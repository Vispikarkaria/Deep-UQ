"""Batch Ensemble and Packed Ensemble methods for efficient uncertainty.

These methods provide memory-efficient alternatives to deep ensembles by
sharing most parameters across ensemble members, differing only in
lightweight per-member perturbations.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from deepuq.types import UQResult


class BatchEnsembleLinear(nn.Module):
    """Linear layer with per-member rank-1 perturbations.

    Parameters
    ----------
    in_features:
        Input dimension.
    out_features:
        Output dimension.
    ensemble_size:
        Number of ensemble members sharing the weight matrix.
    """

    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Shared weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)

        # Per-member rank-1 factors
        self.r = nn.Parameter(torch.empty(ensemble_size, in_features))
        self.s = nn.Parameter(torch.empty(ensemble_size, out_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))

        # Initialize r, s from N(1, 0.5) clipped to positive
        nn.init.normal_(self.r, mean=1.0, std=0.5)
        nn.init.normal_(self.s, mean=1.0, std=0.5)
        with torch.no_grad():
            self.r.clamp_(min=1e-4)
            self.s.clamp_(min=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch * ensemble_size, in_features)``.

        Returns
        -------
        Output tensor of shape ``(batch * ensemble_size, out_features)``.
        """
        batch_total = x.shape[0]
        batch_per_member = batch_total // self.ensemble_size

        # Reshape to (ensemble_size, batch_per_member, in_features)
        x = x.view(self.ensemble_size, batch_per_member, self.in_features)

        # Apply per-member input perturbation: x * r_i
        x = x * self.r.unsqueeze(1)  # (E, B, in)

        # Shared linear: x @ W^T
        out = torch.matmul(x, self.weight.t())  # (E, B, out)

        # Apply per-member output perturbation: out * s_i + bias_i
        out = out * self.s.unsqueeze(1) + self.bias.unsqueeze(1)

        # Reshape back to (batch * ensemble_size, out_features)
        return out.view(batch_total, self.out_features)


class BatchEnsembleWrapper(nn.Module):
    """Wraps a base model to use Batch Ensemble layers.

    Parameters
    ----------
    base_model:
        A PyTorch model whose ``nn.Linear`` layers will be replaced.
    ensemble_size:
        Number of ensemble members.
    """

    method_name = "batch_ensemble"

    def __init__(self, base_model: nn.Module, ensemble_size: int = 4):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.model = copy.deepcopy(base_model)
        self.convert_to_batch_ensemble()

    def convert_to_batch_ensemble(self) -> None:
        """Replace all nn.Linear layers with BatchEnsembleLinear."""
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                be_layer = BatchEnsembleLinear(
                    module.in_features,
                    module.out_features,
                    self.ensemble_size,
                )
                # Copy shared weight
                with torch.no_grad():
                    be_layer.weight.copy_(module.weight)
                    if module.bias is not None:
                        be_layer.bias.copy_(
                            module.bias.unsqueeze(0).expand(self.ensemble_size, -1)
                        )
                # Set the attribute on the parent
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], be_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass replicating input for each ensemble member.

        Parameters
        ----------
        x:
            Input of shape ``(batch, ...)``.

        Returns
        -------
        Output of shape ``(batch * ensemble_size, output_dim)``.
        """
        # Replicate: (batch, feat) -> (ensemble_size, batch, feat) -> (E*B, feat)
        x_rep = x.unsqueeze(0).expand(self.ensemble_size, *x.shape)
        x_rep = x_rep.reshape(-1, *x.shape[1:])
        return self.model(x_rep)

    @torch.inference_mode()
    def predict_uq(self, x: torch.Tensor, aggregate: bool = True) -> UQResult:
        """Return uncertainty estimates aggregated across ensemble members.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, in_features)``.
        aggregate:
            If True, return mean and variance across members.

        Returns
        -------
        UQResult with epistemic uncertainty from member disagreement.
        """
        self.eval()
        out = self.forward(x)  # (E*B, out_dim)
        batch_size = x.shape[0]
        # Reshape to (ensemble_size, batch, out_dim)
        out = out.view(self.ensemble_size, batch_size, -1)

        if aggregate:
            mean = out.mean(dim=0)
            var = out.var(dim=0, unbiased=False)
            return UQResult(
                mean=mean,
                epistemic_var=var,
                aleatoric_var=None,
                total_var=var,
                probs=None,
                probs_var=None,
                metadata={
                    "method": self.method_name,
                    "ensemble_size": self.ensemble_size,
                },
            )
        # Return all members stacked
        mean = out.mean(dim=0)
        var = out.var(dim=0, unbiased=False)
        return UQResult(
            mean=mean,
            epistemic_var=var,
            aleatoric_var=None,
            total_var=var,
            probs=None,
            probs_var=None,
            metadata={
                "method": self.method_name,
                "ensemble_size": self.ensemble_size,
                "member_predictions": out,
            },
        )


class PackedLinear(nn.Module):
    """Grouped linear layer for Packed Ensemble.

    Parameters
    ----------
    in_features:
        Input dimension of the original layer.
    out_features:
        Output dimension of the original layer.
    num_packs:
        Number of ensemble members (groups).
    alpha:
        Width multiplier applied before grouping.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_packs: int = 4,
        alpha: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_packs = num_packs
        self.alpha = alpha

        # Effective dimensions after alpha scaling, rounded up to be divisible
        self.effective_in = in_features * alpha
        # Ensure effective_out is divisible by num_packs
        raw_out = out_features * alpha
        self.effective_out = ((raw_out + num_packs - 1) // num_packs) * num_packs

        # Ensure effective_in is divisible by num_packs
        raw_in = self.effective_in
        self.effective_in = ((raw_in + num_packs - 1) // num_packs) * num_packs

        # Grouped weight: each pack gets its own slice
        self.weight = nn.Parameter(
            torch.empty(self.effective_out, self.effective_in // num_packs)
        )
        self.bias = nn.Parameter(torch.zeros(self.effective_out))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with grouped computation.

        Parameters
        ----------
        x:
            Input of shape ``(batch, effective_in)`` or ``(batch, in_features)``.

        Returns
        -------
        Output of shape ``(batch, effective_out)``.
        """
        batch_size = x.shape[0]

        # If input is original size, expand by repeating channels
        if x.shape[-1] == self.in_features:
            x = x.repeat(1, self.alpha)

        # Pad if needed to match effective_in
        if x.shape[-1] < self.effective_in:
            pad_size = self.effective_in - x.shape[-1]
            x = F.pad(x, (0, pad_size))

        # Grouped linear via conv1d trick
        # x: (batch, effective_in) -> (batch, 1, effective_in) for groups
        x = x.unsqueeze(1)  # (B, 1, effective_in)

        # Reshape for grouped matmul: (B, num_packs, effective_in // num_packs)
        x = x.view(batch_size, self.num_packs, self.effective_in // self.num_packs)

        # Weight: (effective_out, effective_in // num_packs)
        # Reshape to (num_packs, effective_out // num_packs, effective_in // num_packs)
        w = self.weight.view(
            self.num_packs,
            self.effective_out // self.num_packs,
            self.effective_in // self.num_packs,
        )

        # Grouped matmul via einsum
        # (B, P, I_p) x (P, O_p, I_p) -> (B, P, O_p)
        out = torch.einsum("bpi,poi->bpo", x, w)

        # Reshape to (B, effective_out)
        out = out.reshape(batch_size, self.effective_out)
        out = out + self.bias
        return out


class PackedEnsembleWrapper(nn.Module):
    """Wraps a base model to use Packed Ensemble layers.

    Parameters
    ----------
    base_model:
        A PyTorch model whose ``nn.Linear`` layers will be replaced.
    num_packs:
        Number of ensemble members.
    alpha:
        Width multiplier.
    """

    method_name = "packed_ensemble"

    def __init__(self, base_model: nn.Module, num_packs: int = 4, alpha: int = 2):
        super().__init__()
        self.num_packs = num_packs
        self.alpha = alpha
        self.model = copy.deepcopy(base_model)
        self._original_out_features: int | None = None
        self.convert_to_packed()

    def convert_to_packed(self) -> None:
        """Replace all nn.Linear layers with PackedLinear."""
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                packed_layer = PackedLinear(
                    module.in_features,
                    module.out_features,
                    self.num_packs,
                    self.alpha,
                )
                self._original_out_features = module.out_features
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], packed_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the packed model."""
        return self.model(x)

    @torch.inference_mode()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return uncertainty from packed ensemble members.

        The output is split across packs and aggregated.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, in_features)``.

        Returns
        -------
        UQResult with epistemic uncertainty.
        """
        self.eval()
        out = self.forward(x)  # (B, effective_out)
        batch_size = x.shape[0]

        # Split output into packs: (B, num_packs, out_per_pack)
        out_per_pack = out.shape[-1] // self.num_packs
        out = out.view(batch_size, self.num_packs, out_per_pack)

        mean = out.mean(dim=1)
        var = out.var(dim=1, unbiased=False)

        return UQResult(
            mean=mean,
            epistemic_var=var,
            aleatoric_var=None,
            total_var=var,
            probs=None,
            probs_var=None,
            metadata={
                "method": self.method_name,
                "num_packs": self.num_packs,
                "alpha": self.alpha,
            },
        )

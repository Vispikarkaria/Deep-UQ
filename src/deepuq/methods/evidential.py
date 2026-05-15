"""Evidential Deep Learning for regression and classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepuq.types import UQResult


class EvidentialRegression(nn.Module):
    """Evidential regression using Normal-Inverse-Gamma prior."""

    def __init__(self, base_model: nn.Module, output_dim: int = 1) -> None:
        super().__init__()
        self.base_model = base_model
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.base_model(x)
        # Split into 4 groups of output_dim
        gamma, nu_raw, alpha_raw, beta_raw = raw.split(self.output_dim, dim=-1)
        nu = F.softplus(nu_raw) + 1e-6
        alpha = F.softplus(alpha_raw) + 1.0 + 1e-6
        beta = F.softplus(beta_raw) + 1e-6
        return {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

    def loss(self, x: torch.Tensor, y: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
        params = self.forward(x)
        gamma = params["gamma"]
        nu = params["nu"]
        alpha = params["alpha"]
        beta = params["beta"]

        # NIG negative log-likelihood (Student-t)
        omega = beta * (1.0 + nu) / (nu * alpha)
        # log pdf of Student-t: df=2*alpha, loc=gamma, scale=sqrt(omega)
        df = 2.0 * alpha
        diff = y - gamma
        nll = (
            torch.lgamma(df / 2.0)
            - torch.lgamma((df + 1.0) / 2.0)
            + 0.5 * torch.log(torch.pi * omega * df)  # note: scale^2 * df = omega * df
            + ((df + 1.0) / 2.0) * torch.log(1.0 + diff**2 / (omega * df))
        )
        # Correct Student-t: scale = sqrt(omega), so var = omega * df/(df-2)
        # Actually the standard form: scale param s means pdf ~ (1 + (x-loc)^2/(s^2 * df))^...
        # We need: log p = lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi*s^2) - (df+1)/2 * log(1 + (x-mu)^2/(df*s^2))
        # So NLL (negative of above):
        # nll = -lgamma((df+1)/2) + lgamma(df/2) + 0.5*log(df*pi*s^2) + (df+1)/2 * log(1 + (x-mu)^2/(df*s^2))
        # where s^2 = omega = beta*(1+nu)/(nu*alpha)
        # The code above already computes this correctly (just rearranged signs)

        # Evidence regularizer
        reg = torch.abs(y - gamma) * (2.0 * nu + alpha)

        total_loss = (nll + coeff * reg).mean()
        return total_loss

    @torch.no_grad()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        params = self.forward(x)
        gamma = params["gamma"]
        nu = params["nu"]
        alpha = params["alpha"]
        beta = params["beta"]

        aleatoric_var = beta / (alpha - 1.0)
        epistemic_var = beta / (nu * (alpha - 1.0))
        total_var = aleatoric_var + epistemic_var

        return UQResult(
            mean=gamma,
            aleatoric_var=aleatoric_var,
            epistemic_var=epistemic_var,
            total_var=total_var,
        )


class EvidentialClassification(nn.Module):
    """Evidential classification using Dirichlet prior."""

    def __init__(self, base_model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.base_model(x)
        alpha = F.softplus(raw) + 1.0
        return alpha

    def loss(self, x: torch.Tensor, y: torch.Tensor, kl_coeff: float = 1.0) -> torch.Tensor:
        alpha = self.forward(x)
        S = alpha.sum(dim=-1, keepdim=True)

        # One-hot encode y if needed
        if y.dim() == 1 or (y.dim() == 2 and y.shape[-1] == 1):
            y_one_hot = F.one_hot(y.view(-1).long(), self.num_classes).float()
        else:
            y_one_hot = y.float()

        # Bayes risk of cross-entropy under Dirichlet
        bayes_risk = (y_one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        # KL regularizer: KL(Dir(alpha_tilde) || Dir(1)) for incorrect classes
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
        ones = torch.ones_like(alpha_tilde)
        S_ones = ones.sum(dim=-1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde) - torch.lgamma(S_ones)
            - (torch.lgamma(alpha_tilde) - torch.lgamma(ones)).sum(dim=-1, keepdim=True)
            + ((alpha_tilde - ones) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1, keepdim=True)
        ).squeeze(-1)

        total_loss = (bayes_risk + kl_coeff * kl).mean()
        return total_loss

    @torch.no_grad()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        alpha = self.forward(x)
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        epistemic_var = self.num_classes / S.squeeze(-1)

        return UQResult(
            mean=probs,
            epistemic_var=epistemic_var,
            probs=probs,
        )

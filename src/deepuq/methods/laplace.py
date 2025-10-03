from typing import Iterable, List, Optional

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def _find_last_linear_layer(model: nn.Module) -> nn.Module:
    last_linear: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError('Could not locate a linear layer in the model to use for Laplace approximation.')
    return last_linear


class _SimpleDiagonalLaplace:
    """Diagonal Laplace approximation with optional last-layer restriction."""

    def __init__(self, model: nn.Module, likelihood: str = 'regression', subset_of_weights: str = 'last_layer') -> None:
        if likelihood not in {'regression', 'classification'}:
            raise ValueError(f'Unsupported likelihood "{likelihood}". Use "regression" or "classification".')
        if subset_of_weights not in {'last_layer', 'all'}:
            raise ValueError('subset_of_weights must be "last_layer" or "all".')

        self.model = model
        self.likelihood = likelihood
        self.subset_of_weights = subset_of_weights

        if subset_of_weights == 'last_layer':
            self._parameter_modules = list(_find_last_linear_layer(model).parameters())
        else:
            self._parameter_modules = list(model.parameters())

        if len(self._parameter_modules) == 0:
            raise ValueError('No parameters selected for the Laplace approximation.')

        self.device = next(model.parameters()).device
        self._param_dim = parameters_to_vector(self._parameter_modules).numel()

        self.mean_vector: Optional[torch.Tensor] = None
        self.hessian_diag: Optional[torch.Tensor] = None
        self.prior_precision: Optional[torch.Tensor] = None
        self.posterior_variance_diag: Optional[torch.Tensor] = None
        self.posterior_precision_diag: Optional[torch.Tensor] = None
        self.empirical_noise_variance: Optional[torch.Tensor] = None

    def fit(self, train_loader: Iterable, prior_precision: Optional[float] = 1.0) -> '_SimpleDiagonalLaplace':
        self.model.eval()
        param_vector = parameters_to_vector(self._parameter_modules).detach().clone()

        diag_accumulator = torch.zeros_like(param_vector)
        residual_sum_squares = 0.0
        count_outputs = 0

        mse_loss = nn.MSELoss(reduction='sum')
        ce_loss = nn.CrossEntropyLoss(reduction='sum')

        if not hasattr(train_loader, '__iter__'):
            raise TypeError('train_loader must be an iterable over (input, target) pairs.')

        for batch in train_loader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                raise ValueError('Each batch must be a tuple of (inputs, targets).')
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model(inputs)

            if self.likelihood == 'regression':
                if targets.dim() < outputs.dim():
                    targets = targets.unsqueeze(-1)
                loss = 0.5 * mse_loss(outputs, targets)
                residual_sum_squares += torch.sum((outputs.detach() - targets.detach()) ** 2).item()
                count_outputs += targets.numel()
            else:
                if targets.dim() != 1:
                    raise ValueError('Classification targets must be a 1D tensor of class indices.')
                loss = ce_loss(outputs, targets)
                count_outputs += targets.size(0)

            gradients = torch.autograd.grad(loss, self._parameter_modules, retain_graph=False)
            grad_vector = torch.cat([g.detach().reshape(-1) for g in gradients])
            diag_accumulator += grad_vector.pow(2)

        num_datapoints = len(getattr(train_loader, 'dataset', []))
        if num_datapoints == 0:
            num_datapoints = 1
        diag = diag_accumulator / float(num_datapoints)

        self.hessian_diag = diag
        self.mean_vector = param_vector

        if prior_precision is None:
            prior_precision = 1.0
        prior_tensor = torch.full_like(diag, float(prior_precision))
        self.prior_precision = prior_tensor

        eps = torch.tensor(1e-12, device=self.device, dtype=diag.dtype)
        self.posterior_precision_diag = diag + prior_tensor + eps
        self.posterior_variance_diag = 1.0 / self.posterior_precision_diag

        if self.likelihood == 'regression':
            if count_outputs == 0:
                count_outputs = 1
            noise_var = residual_sum_squares / float(count_outputs)
            self.empirical_noise_variance = torch.tensor(noise_var, device=self.device, dtype=diag.dtype)
        else:
            self.empirical_noise_variance = None

        return self

    def optimize_prior_precision(self, value: float = 1.0) -> None:
        if self.posterior_variance_diag is None or self.hessian_diag is None:
            raise RuntimeError('Call fit() before optimising the prior precision.')
        prior_tensor = torch.full_like(self.hessian_diag, float(value))
        self.prior_precision = prior_tensor
        eps = torch.tensor(1e-12, device=self.device, dtype=self.hessian_diag.dtype)
        self.posterior_precision_diag = self.hessian_diag + prior_tensor + eps
        self.posterior_variance_diag = 1.0 / self.posterior_precision_diag

    def predictive(self, x: torch.Tensor, n_samples: int = 50) -> tuple:
        if self.posterior_variance_diag is None or self.mean_vector is None:
            raise RuntimeError('Laplace approximation not fitted yet.')
        if n_samples <= 0:
            raise ValueError('n_samples must be positive.')

        x = x.to(self.device)
        std_vector = torch.sqrt(self.posterior_variance_diag.clamp_min(1e-12))
        mean_vector = self.mean_vector.to(self.device)

        originals = parameters_to_vector(self._parameter_modules).detach().clone()
        samples = mean_vector.unsqueeze(0) + torch.randn(n_samples, mean_vector.numel(), device=self.device) * std_vector.unsqueeze(0)

        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for sample_vec in samples:
                vector_to_parameters(sample_vec, self._parameter_modules)
                outputs.append(self.model(x).detach())
            vector_to_parameters(originals, self._parameter_modules)

        stacked = torch.stack(outputs, dim=0)

        if self.likelihood == 'regression':
            mean = stacked.mean(dim=0)
            var = stacked.var(dim=0, unbiased=False)
            if self.empirical_noise_variance is not None:
                var = var + self.empirical_noise_variance
            return mean, var

        probs = torch.softmax(stacked, dim=-1)
        mean_probs = probs.mean(dim=0)
        return mean_probs, None


class LaplaceWrapper:
    """
    Fit a Laplace approximation around a MAP-trained model using a diagonal covariance.

    Example:
        la = LaplaceWrapper(model, 'classification')
        la.fit(dataloader, prior_precision=1.0)
        probs, var = la.predict(x)
    """

    def __init__(self, model: nn.Module, likelihood: str = 'classification', hessian_structure: str = 'diag', subset_of_weights: str = 'last_layer') -> None:
        if hessian_structure != 'diag':
            raise ValueError('Only diagonal Hessian structure is supported by this implementation.')
        self.model = model
        self.likelihood = likelihood
        self.hessian_structure = hessian_structure
        self.subset_of_weights = subset_of_weights
        self.la: Optional[_SimpleDiagonalLaplace] = None

    def fit(self, train_loader: Iterable, prior_precision: Optional[float] = 1.0, **_) -> _SimpleDiagonalLaplace:
        self.model.eval()
        la = _SimpleDiagonalLaplace(
            self.model,
            likelihood=self.likelihood,
            subset_of_weights=self.subset_of_weights,
        )
        la.fit(train_loader, prior_precision=prior_precision)
        self.la = la
        return la

    def predict(self, x: torch.Tensor, **predict_kwargs):
        if self.la is None:
            raise RuntimeError('Call fit() before predict().')
        return self.la.predictive(x, **predict_kwargs)

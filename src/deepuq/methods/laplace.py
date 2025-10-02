import torch
from laplace import Laplace

class LaplaceWrapper:
    """
    Fit a Laplace approximation around a MAP-trained model.

    Example:
        la = LaplaceWrapper(model, 'classification')
        la.fit(dataloader, prior_precision=1.0)
        probs, var = la.predict(x)
    """
    def __init__(self, model, likelihood='classification', hessian_structure='diag'):
        self.model = model
        self.likelihood = likelihood
        self.hessian_structure = hessian_structure
        self.la = None

    def fit(self, train_loader, prior_precision=1.0):
        self.model.eval()
        la = Laplace(self.model, self.likelihood, self.hessian_structure)
        la.fit(train_loader)
        la.optimize_prior_precision(prior_precision=prior_precision)
        self.la = la
        return la

    @torch.inference_mode()
    def predict(self, x):
        assert self.la is not None, "Call fit() first"
        f = self.la(x)
        probs = self.la.predictive(x)
        return probs, None

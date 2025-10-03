import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianPosterior(nn.Module):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps

    def log_prob(self, w):
        return (-0.5 * ((w - self.mu) / self.sigma).pow(2) - torch.log(self.sigma) - 0.5*math.log(2*math.pi)).sum()

class GaussianPrior:
    def __init__(self, mu=0.0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, w):
        return (-0.5 * ((w - self.mu) / self.sigma).pow(2) - math.log(self.sigma) - 0.5*math.log(2*math.pi)).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_posterior = GaussianPosterior(torch.zeros(out_features, in_features), torch.full((out_features, in_features), -3.0))
        self.bias_posterior   = GaussianPosterior(torch.zeros(out_features), torch.full((out_features,), -3.0))
        self.prior = GaussianPrior(0.0, prior_sigma)

    def forward(self, x, sample=True):
        w = self.weight_posterior.sample() if sample else self.weight_posterior.mu
        b = self.bias_posterior.sample() if sample else self.bias_posterior.mu
        return F.linear(x, w, b)

    def kl(self):
        # KL(q||p) = E_q[log q - log p] approximated by analytic expression for Gaussians
        qw_mu, qw_sigma = self.weight_posterior.mu, self.weight_posterior.sigma
        qb_mu, qb_sigma = self.bias_posterior.mu, self.bias_posterior.sigma
        pw_sigma = self.prior.sigma
        pb_sigma = self.prior.sigma

        kl_w = (torch.log(pw_sigma/qw_sigma) + (qw_sigma**2 + qw_mu**2)/(2*pw_sigma**2) - 0.5).sum()
        kl_b = (torch.log(pb_sigma/qb_sigma) + (qb_sigma**2 + qb_mu**2)/(2*pb_sigma**2) - 0.5).sum()
        return kl_w + kl_b

class BayesByBackpropMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, prior_sigma=0.1):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims)-2):
            layers += [BayesianLinear(dims[i], dims[i+1], prior_sigma), nn.ReLU()]
        layers += [BayesianLinear(dims[-2], dims[-1], prior_sigma)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sample=True):
        h = x
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                h = layer(h, sample=sample)
            else:
                h = layer(h)
        return h

    def kl(self):
        return sum([m.kl() for m in self.layers if isinstance(m, BayesianLinear)])

def vi_elbo_step(model, x, y, n_batches, criterion=None, kl_weight=1.0):
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    logits = model(x, sample=True)
    nll = criterion(logits, y)
    kl = model.kl() / n_batches
    loss = nll + kl_weight * kl
    return loss, nll.detach(), kl.detach()

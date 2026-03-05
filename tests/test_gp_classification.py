import torch

from deepuq.models import GaussianProcessClassifier, OneVsRestGaussianProcessClassifier


def test_binary_gp_classifier_probabilities() -> None:
    torch.manual_seed(0)
    x = torch.randn(48, 2)
    y = (x[:, 1] > 0.4 * x[:, 0]).float()

    model = GaussianProcessClassifier(max_iter=10, tol=1e-4)
    model.fit(x, y)

    x_test = torch.randn(12, 2)
    probs = model.predict_proba(x_test)
    pred = model.predict(x_test)
    uq = model.predict_uq(x_test)

    assert probs.shape == (12, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(12), atol=1e-4)
    assert pred.shape == (12,)
    assert uq.probs is not None
    assert uq.probs_var is not None


def test_ovr_gp_classifier_probabilities() -> None:
    torch.manual_seed(1)
    x = torch.randn(60, 2)
    score = x[:, 0] + 0.8 * x[:, 1]
    y = torch.bucketize(score, boundaries=torch.tensor([-0.3, 0.5]))

    model = OneVsRestGaussianProcessClassifier(max_iter=8, tol=1e-4)
    model.fit(x, y)

    x_test = torch.randn(16, 2)
    probs = model.predict_proba(x_test)
    pred = model.predict(x_test)
    uq = model.predict_uq(x_test)

    assert probs.shape == (16, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-3)
    assert pred.shape == (16,)
    assert uq.probs is not None
    assert uq.probs_var is not None

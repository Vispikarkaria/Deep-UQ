import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import (
    BayesByBackpropMLP,
    DeepEnsembleClassifier,
    DeepEnsembleRegressor,
    DeepEnsembleWrapper,
    HeteroscedasticDeepEnsembleRegressor,
    HeteroscedasticMultiOutputDeepEnsembleRegressor,
    LaplaceWrapper,
    MCDropoutWrapper,
    MultiOutputDeepEnsembleRegressor,
    predict_vi_uq,
    predict_with_samples,
    predict_with_samples_uq,
)
from deepuq.models import (
    MLP,
    DeepKernelGaussianProcessRegressor,
    GaussianProcessClassifier,
    GaussianProcessRegressor,
    HeteroscedasticGaussianProcessRegressor,
    MultiTaskGaussianProcessRegressor,
    OneVsRestGaussianProcessClassifier,
    SparseGaussianProcessRegressor,
    SpectralMixtureGaussianProcessRegressor,
)
from deepuq.types import UQResult


def test_mc_dropout_predict_uq_matches_legacy_shapes():
    model = MLP(4, [8], 1, p_drop=0.2)
    wrapper = MCDropoutWrapper(model, n_mc=8, apply_softmax=False)
    x = torch.randn(6, 4)

    legacy_mean, legacy_var = wrapper.predict(x)
    uq = wrapper.predict_uq(x)

    assert isinstance(uq, UQResult)
    assert uq.mean.shape == legacy_mean.shape
    assert uq.epistemic_var is not None
    assert uq.epistemic_var.shape == legacy_var.shape
    assert uq.total_var is not None
    assert uq.total_var.shape == legacy_var.shape


def test_mc_dropout_predict_uq_classification_probs():
    model = MLP(3, [8], 4, p_drop=0.25)
    wrapper = MCDropoutWrapper(model, n_mc=6, apply_softmax=True)
    x = torch.randn(5, 3)

    uq = wrapper.predict_uq(x)
    assert uq.probs is not None
    assert uq.probs_var is not None
    assert uq.probs.shape == (5, 4)
    assert uq.probs_var.shape == (5, 4)
    row_sums = uq.probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_laplace_predict_uq_regression_shapes():
    x = torch.randn(32, 3)
    y = (x[:, :1] * 0.7) + 0.05 * torch.randn(32, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

    model = MLP(3, [12], 1, p_drop=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(10):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()

    la = LaplaceWrapper(
        model,
        likelihood="regression",
        hessian_structure="diag",
        subset_of_weights="last_layer",
    )
    la.fit(loader, prior_precision=1.0)
    uq = la.predict_uq(torch.randn(5, 3), n_samples=10)

    assert isinstance(uq, UQResult)
    assert uq.mean.shape == (5, 1)
    assert uq.total_var is not None and uq.total_var.shape == (5, 1)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (5, 1)
    if uq.aleatoric_var is not None:
        assert torch.all(uq.total_var >= uq.epistemic_var)


def test_laplace_predict_uq_classification_fields():
    x = torch.randn(48, 4)
    y = torch.randint(0, 3, (48,))
    loader = DataLoader(TensorDataset(x, y), batch_size=12, shuffle=True)

    model = MLP(4, [10], 3, p_drop=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(8):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

    la = LaplaceWrapper(
        model,
        likelihood="classification",
        hessian_structure="diag",
        subset_of_weights="last_layer",
    )
    la.fit(loader, prior_precision=1.0)
    uq = la.predict_uq(torch.randn(6, 4), n_samples=8)
    assert uq.probs is not None
    assert uq.mean.shape == (6, 3)
    assert torch.allclose(uq.probs.sum(dim=1), torch.ones(6), atol=1e-3)
    assert uq.total_var is None


def test_mcmc_predict_with_samples_uq():
    model = MLP(2, [4], 1, p_drop=0.0)
    x = torch.randn(4, 2)

    samples = []
    for _ in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01 * torch.randn_like(p))
        samples.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )

    legacy_mean, legacy_var = predict_with_samples(
        model, samples, x, apply_softmax=False
    )
    uq = predict_with_samples_uq(model, samples, x, apply_softmax=False)

    assert isinstance(uq, UQResult)
    assert uq.mean.shape == legacy_mean.shape
    assert uq.epistemic_var is not None
    assert uq.epistemic_var.shape == legacy_var.shape


def test_vi_predict_uq_shapes():
    model = BayesByBackpropMLP(3, [8], 1)
    x = torch.randn(7, 3)

    uq = predict_vi_uq(model, x, n_samples=6, apply_softmax=False)
    assert isinstance(uq, UQResult)
    assert uq.mean.shape == (7, 1)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (7, 1)
    assert uq.total_var is not None and uq.total_var.shape == (7, 1)


def test_deep_ensemble_predict_uq_shapes():
    models = [MLP(3, [8], 1, p_drop=0.0) for _ in range(3)]
    wrapper = DeepEnsembleWrapper(models)
    x = torch.randn(6, 3)

    uq = wrapper.predict_uq(x)
    assert isinstance(uq, UQResult)
    assert uq.mean.shape == (6, 1)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (6, 1)
    assert uq.total_var is not None and uq.total_var.shape == (6, 1)
    assert uq.metadata["method"] == "deep_ensemble"


def test_additional_deep_ensemble_uq_contracts():
    x = torch.randn(8, 3)

    reg = DeepEnsembleRegressor([MLP(3, [8], 1, p_drop=0.0) for _ in range(2)])
    reg_uq = reg.predict_uq(x)
    assert reg_uq.metadata["method"] == "deep_ensemble_regressor"

    hetero = HeteroscedasticDeepEnsembleRegressor(
        [MLP(3, [8], 2, p_drop=0.0) for _ in range(2)]
    )
    hetero_uq = hetero.predict_uq(x)
    assert hetero_uq.aleatoric_var is not None
    assert hetero_uq.metadata["method"] == "heteroscedastic_deep_ensemble_regressor"

    classifier = DeepEnsembleClassifier([MLP(3, [8], 4, p_drop=0.0) for _ in range(2)])
    classifier_uq = classifier.predict_uq(x)
    assert classifier_uq.probs is not None and classifier_uq.probs.shape == (8, 4)
    assert classifier_uq.probs_var is not None

    multi = MultiOutputDeepEnsembleRegressor(
        [MLP(3, [8], 2, p_drop=0.0) for _ in range(2)]
    )
    multi_uq = multi.predict_uq(x)
    assert multi_uq.mean.shape == (8, 2)

    hetero_multi = HeteroscedasticMultiOutputDeepEnsembleRegressor(
        [MLP(3, [8], 4, p_drop=0.0) for _ in range(2)]
    )
    hetero_multi_uq = hetero_multi.predict_uq(x)
    assert hetero_multi_uq.mean.shape == (8, 2)
    assert hetero_multi_uq.aleatoric_var is not None


def test_gp_predict_uq_shapes():
    x = torch.linspace(-1.0, 1.0, 24).unsqueeze(-1)
    y = torch.sin(2 * torch.pi * x)

    exact = GaussianProcessRegressor(noise=1e-4)
    exact.fit(x, y)
    exact_uq = exact.predict_uq(x[:5])
    assert isinstance(exact_uq, UQResult)
    assert exact_uq.mean.shape == (5,)
    assert exact_uq.total_var is not None and exact_uq.total_var.shape == (5,)

    sparse = SparseGaussianProcessRegressor(
        num_inducing=8, num_iterations=5, verbose=False
    )
    sparse.fit(x, y)
    sparse_uq = sparse.predict_uq(x[:5])
    assert isinstance(sparse_uq, UQResult)
    assert sparse_uq.mean.shape == (5,)
    assert sparse_uq.total_var is not None and sparse_uq.total_var.shape == (5,)


def test_new_gp_regressors_predict_uq_contract():
    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(-1)
    y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)
    x_test = torch.linspace(-1.4, 1.4, 12).unsqueeze(-1)

    hetero = HeteroscedasticGaussianProcessRegressor(num_alternations=3)
    hetero.fit(x, y)
    hetero_uq = hetero.predict_uq(x_test)
    assert hetero_uq.total_var is not None
    assert hetero_uq.aleatoric_var is not None
    assert hetero_uq.metadata["method"] == "heteroscedastic_gp"

    sm = SpectralMixtureGaussianProcessRegressor(num_mixtures=2, opt_steps=30)
    sm.fit(x, y)
    sm_uq = sm.predict_uq(x_test)
    assert sm_uq.total_var is not None
    assert sm_uq.metadata["method"] == "spectral_mixture_gp"

    x_dkl = torch.cat([x, x**2, torch.sin(3 * x)], dim=1)
    x_dkl_test = torch.cat([x_test, x_test**2, torch.sin(3 * x_test)], dim=1)
    dkl = DeepKernelGaussianProcessRegressor(
        feature_dim=8,
        hidden_dims=(16, 16),
        epochs=25,
        lr=1e-3,
    )
    dkl.fit(x_dkl, y)
    dkl_uq = dkl.predict_uq(x_dkl_test)
    assert dkl_uq.total_var is not None
    assert dkl_uq.metadata["method"] == "deep_kernel_gp"


def test_new_gp_classifiers_predict_uq_contract():
    x = torch.randn(36, 2)
    y_bin = (x[:, 0] + 0.4 * x[:, 1] > 0).float()
    y_multi = torch.bucketize(
        x[:, 0] - 0.3 * x[:, 1],
        boundaries=torch.tensor([-0.2, 0.4]),
    )
    x_test = torch.randn(10, 2)

    gpc = GaussianProcessClassifier(max_iter=10, tol=1e-4)
    gpc.fit(x, y_bin)
    gpc_uq = gpc.predict_uq(x_test)
    assert gpc_uq.probs is not None and gpc_uq.probs.shape == (10, 2)
    assert gpc_uq.total_var is None

    ovr = OneVsRestGaussianProcessClassifier(max_iter=8, tol=1e-4)
    ovr.fit(x, y_multi)
    ovr_uq = ovr.predict_uq(x_test)
    assert ovr_uq.probs is not None
    assert ovr_uq.probs.shape[0] == 10
    row_sums = ovr_uq.probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_multitask_gp_predict_uq_contract():
    x = torch.linspace(-1.0, 1.0, 20).unsqueeze(-1)
    y1 = torch.sin(1.2 * x)
    y2 = 0.6 * torch.cos(1.0 * x)
    y = torch.cat([y1, y2], dim=1)

    model = MultiTaskGaussianProcessRegressor(num_tasks=2, opt_steps=25, lr=0.05)
    model.fit(x, y)
    uq = model.predict_uq(x[:6])
    assert uq.mean.shape == (6, 2)
    assert uq.total_var is not None and uq.total_var.shape == (6, 2)
    assert uq.metadata["method"] == "multitask_icm_gp"

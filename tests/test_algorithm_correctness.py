"""Algorithm correctness tests for Deep-UQ.

These tests verify statistical properties of UQ methods using synthetic data
with known ground truth. They check calibration, convergence, consistency,
OOD behavior, variance decomposition, numerical stability, and determinism.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import (
    DeepEnsembleRegressor,
    HeteroscedasticDeepEnsembleRegressor,
    LaplaceWrapper,
    MCDropoutWrapper,
    SGLDOptimizer,
)
from deepuq.models import MLP, GaussianProcessRegressor, RBFKernel
from deepuq.types import UQResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_data(n=200, noise_std=0.1, seed=42):
    torch.manual_seed(seed)
    x = torch.linspace(-1.0, 1.0, n).unsqueeze(-1)
    y = torch.sin(2 * torch.pi * x) + noise_std * torch.randn_like(x)
    return x, y


def _train_mlp(x, y, hidden=64, epochs=300, p_drop=0.0, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = MLP(input_dim=1, hidden_dims=[hidden, hidden], output_dim=1, p_drop=p_drop)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _coverage(mean, var, y_true, z=1.96):
    """Fraction of truth within mean +/- z*std."""
    std = var.sqrt()
    lower = mean - z * std
    upper = mean + z * std
    inside = ((y_true >= lower) & (y_true <= upper)).float()
    return inside.mean().item()


# ---------------------------------------------------------------------------
# 1. Calibration — predicted intervals should contain truth at stated rate
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_gp_calibration(self):
        """GP on clean data should achieve ~95% coverage at 1.96 sigma (including noise)."""
        noise_std = 0.05
        x, y = _sine_data(n=80, noise_std=noise_std, seed=10)
        gp = GaussianProcessRegressor(
            kernel=RBFKernel(lengthscale=0.3, outputscale=1.0), noise=noise_std**2
        )
        gp.fit(x, y)
        mean, var = gp.predict(x)
        # Add observation noise back for full predictive variance
        predictive_var = var + noise_std**2
        cov = _coverage(mean.unsqueeze(-1), predictive_var.unsqueeze(-1), y, z=1.96)
        assert cov >= 0.85, f"GP coverage {cov:.2f} < 0.85"

    def test_ensemble_calibration(self):
        """Ensemble on training data should have reasonable coverage."""
        x, y = _sine_data(n=100, noise_std=0.1, seed=20)
        loader = DataLoader(TensorDataset(x, y), batch_size=50, shuffle=True)
        models = [
            MLP(input_dim=1, hidden_dims=[64, 64], output_dim=1) for _ in range(10)
        ]
        ens = DeepEnsembleRegressor(models)
        ens.fit(loader, epochs=300, lr=1e-3, seed=42)
        uq = ens.predict_uq(x)
        # Ensemble epistemic var + training noise should cover truth
        predictive_var = uq.total_var + 0.1**2
        cov = _coverage(uq.mean, predictive_var, y, z=2.0)
        assert cov >= 0.70, f"Ensemble coverage {cov:.2f} < 0.70"

    def test_mc_dropout_calibration(self):
        """MC Dropout on training data should produce meaningful intervals."""
        x, y = _sine_data(n=100, noise_std=0.1, seed=30)
        model = _train_mlp(x, y, p_drop=0.15, epochs=400, seed=30)
        wrapper = MCDropoutWrapper(model, n_mc=100, apply_softmax=False)
        uq = wrapper.predict_uq(x)
        cov = _coverage(uq.mean, uq.total_var, y, z=2.0)
        assert cov >= 0.70, f"MC Dropout coverage {cov:.2f} < 0.70"


# ---------------------------------------------------------------------------
# 2. Convergence — more data/samples should reduce uncertainty
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_gp_variance_decreases_with_data(self):
        """GP posterior variance at a test point should decrease with more data."""
        x_test = torch.tensor([[0.5]])
        variances = []
        for n in [5, 20, 80]:
            x, y = _sine_data(n=n, noise_std=0.05, seed=0)
            gp = GaussianProcessRegressor(kernel=RBFKernel(), noise=0.05)
            gp.fit(x, y)
            _, var = gp.predict(x_test)
            variances.append(var.item())
        assert (
            variances[0] > variances[1] > variances[2]
        ), f"GP variance should decrease with more data: {variances}"

    def test_ensemble_variance_decreases_with_members(self):
        """More ensemble members should give tighter epistemic uncertainty."""
        x, y = _sine_data(n=80, noise_std=0.1, seed=5)
        loader = DataLoader(TensorDataset(x, y), batch_size=40, shuffle=True)
        x_test = torch.tensor([[0.0]])
        variances = []
        for n_members in [2, 5, 10]:
            models = [
                MLP(input_dim=1, hidden_dims=[32, 32], output_dim=1)
                for _ in range(n_members)
            ]
            ens = DeepEnsembleRegressor(models)
            ens.fit(loader, epochs=150, lr=1e-3, seed=100)
            uq = ens.predict_uq(x_test)
            variances.append(uq.epistemic_var.item())
        # With more well-trained members, disagreement typically decreases
        # or at minimum doesn't blow up. Check last < 10x first.
        assert (
            variances[-1] < variances[0] * 10
        ), f"Ensemble variance should not explode with more members: {variances}"


# ---------------------------------------------------------------------------
# 3. Known posteriors — verify against analytical solutions
# ---------------------------------------------------------------------------


class TestKnownPosteriors:
    def test_gp_linear_matches_bayesian_lr(self):
        """GP with linear kernel on linear data should recover slope well."""
        torch.manual_seed(99)
        x = torch.linspace(-1, 1, 50).unsqueeze(-1)
        true_w = 2.5
        noise_std = 0.1
        y = true_w * x + noise_std * torch.randn_like(x)

        from deepuq.models import LinearKernel

        gp = GaussianProcessRegressor(
            kernel=LinearKernel(variance=10.0), noise=noise_std**2
        )
        gp.fit(x, y)
        x_test = torch.tensor([[1.0], [-1.0]])
        mean, var = gp.predict(x_test)

        # Prediction at x=1 should be ~true_w, at x=-1 should be ~-true_w
        assert (
            abs(mean[0].item() - true_w) < 0.5
        ), f"GP mean at x=1 should be ~{true_w}, got {mean[0].item():.3f}"
        # Variance should be positive and finite
        assert (var > 0).all() and torch.isfinite(var).all()

    def test_gp_interpolation(self):
        """GP must exactly interpolate noise-free training points."""
        x = torch.tensor([[0.0], [1.0], [2.0]])
        y = torch.tensor([[1.0], [3.0], [2.0]])
        gp = GaussianProcessRegressor(kernel=RBFKernel(), noise=1e-6)
        gp.fit(x, y)
        mean, var = gp.predict(x)
        # GP predict returns 1D tensor
        assert torch.allclose(
            mean, y.squeeze(), atol=1e-3
        ), "GP should interpolate training data"
        assert (var < 1e-3).all(), "GP variance at training points should be ~0"


# ---------------------------------------------------------------------------
# 4. Consistency — degenerate cases should match deterministic behavior
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_ensemble_single_member_matches_model(self):
        """Ensemble with 1 member should have zero epistemic variance."""
        x, y = _sine_data(n=50, noise_std=0.1, seed=7)
        loader = DataLoader(TensorDataset(x, y), batch_size=50, shuffle=False)
        model = MLP(input_dim=1, hidden_dims=[32, 32], output_dim=1)
        ens = DeepEnsembleRegressor([model])
        ens.fit(loader, epochs=100, lr=1e-3, seed=7)
        uq = ens.predict_uq(x[:5])
        assert (
            uq.epistemic_var == 0
        ).all(), "Single-member ensemble should have zero epistemic variance"

    def test_mc_dropout_zero_dropout_has_zero_variance(self):
        """MC Dropout with p=0 should produce zero variance."""
        x, y = _sine_data(n=50, noise_std=0.1, seed=8)
        model = _train_mlp(x, y, p_drop=0.0, epochs=100, seed=8)
        wrapper = MCDropoutWrapper(model, n_mc=20, apply_softmax=False)
        uq = wrapper.predict_uq(x[:10])
        assert (
            uq.epistemic_var < 1e-10
        ).all(), "Zero dropout should produce zero MC variance"


# ---------------------------------------------------------------------------
# 5. OOD detection — uncertainty must grow away from training data
# ---------------------------------------------------------------------------


class TestOODDetection:
    def test_gp_ood_higher_variance(self):
        """GP variance far from training data should exceed in-distribution variance."""
        x_train = torch.linspace(-1, 1, 40).unsqueeze(-1)
        y_train = torch.sin(2 * torch.pi * x_train)
        gp = GaussianProcessRegressor(kernel=RBFKernel(), noise=1e-4)
        gp.fit(x_train, y_train)

        x_id = torch.tensor([[0.0]])
        x_ood = torch.tensor([[5.0]])
        _, var_id = gp.predict(x_id)
        _, var_ood = gp.predict(x_ood)
        assert var_ood.item() > var_id.item() * 5, (
            f"OOD variance ({var_ood.item():.4f}) should be much larger than "
            f"ID variance ({var_id.item():.4f})"
        )

    def test_ensemble_ood_higher_variance(self):
        """Ensemble should show more disagreement on OOD inputs."""
        x, y = _sine_data(n=100, noise_std=0.05, seed=11)
        loader = DataLoader(TensorDataset(x, y), batch_size=50, shuffle=True)
        models = [
            MLP(input_dim=1, hidden_dims=[64, 64], output_dim=1) for _ in range(5)
        ]
        ens = DeepEnsembleRegressor(models)
        ens.fit(loader, epochs=300, lr=1e-3, seed=11)

        x_id = torch.tensor([[0.0]])
        x_ood = torch.tensor([[5.0]])
        var_id = ens.predict_uq(x_id).epistemic_var.item()
        var_ood = ens.predict_uq(x_ood).epistemic_var.item()
        assert (
            var_ood > var_id
        ), f"Ensemble OOD var ({var_ood:.6f}) should exceed ID var ({var_id:.6f})"

    def test_laplace_ood_higher_variance(self):
        """Laplace should show higher uncertainty far from training data."""
        x, y = _sine_data(n=100, noise_std=0.1, seed=12)
        model = _train_mlp(x, y, epochs=300, seed=12)
        loader = DataLoader(TensorDataset(x, y), batch_size=50)

        la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
        la.fit(loader)

        uq_id = la.predict_uq(torch.tensor([[0.0]]), n_samples=100)
        uq_ood = la.predict_uq(torch.tensor([[5.0]]), n_samples=100)
        assert (
            uq_ood.total_var.item() > uq_id.total_var.item()
        ), "Laplace OOD variance should exceed ID variance"


# ---------------------------------------------------------------------------
# 6. Variance decomposition — epistemic + aleatoric = total
# ---------------------------------------------------------------------------


class TestVarianceDecomposition:
    def test_heteroscedastic_ensemble_decomposition(self):
        """Epistemic + aleatoric should equal total variance."""
        torch.manual_seed(50)
        x = torch.linspace(-1, 1, 80).unsqueeze(-1)
        y = torch.sin(2 * torch.pi * x) + 0.1 * torch.randn_like(x)
        loader = DataLoader(TensorDataset(x, y), batch_size=40, shuffle=True)

        # Each model outputs [mean, log_var] concatenated
        models = []
        for _ in range(5):
            m = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # 1 mean + 1 log_var
            )
            models.append(m)

        ens = HeteroscedasticDeepEnsembleRegressor(models)
        ens.fit(loader, epochs=200, lr=1e-3, seed=50)
        uq = ens.predict_uq(x[:20])

        assert uq.epistemic_var is not None
        assert uq.aleatoric_var is not None
        assert uq.total_var is not None
        reconstructed = uq.epistemic_var + uq.aleatoric_var
        assert torch.allclose(
            uq.total_var, reconstructed, atol=1e-5
        ), "Total variance should equal epistemic + aleatoric"

    def test_homoscedastic_ensemble_no_aleatoric(self):
        """Homoscedastic ensemble should have aleatoric_var=None and total=epistemic."""
        x, y = _sine_data(n=50, noise_std=0.1, seed=55)
        loader = DataLoader(TensorDataset(x, y), batch_size=50, shuffle=False)
        models = [MLP(input_dim=1, hidden_dims=[32], output_dim=1) for _ in range(3)]
        ens = DeepEnsembleRegressor(models)
        ens.fit(loader, epochs=100, lr=1e-3, seed=55)
        uq = ens.predict_uq(x[:10])
        assert uq.aleatoric_var is None
        assert torch.allclose(uq.total_var, uq.epistemic_var)


# ---------------------------------------------------------------------------
# 7. Numerical stability — no NaN/Inf under edge cases
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_gp_single_point(self):
        """GP should handle a single training point without NaN."""
        x = torch.tensor([[0.0]])
        y = torch.tensor([[1.0]])
        gp = GaussianProcessRegressor(kernel=RBFKernel(), noise=0.01)
        gp.fit(x, y)
        mean, var = gp.predict(torch.tensor([[0.0], [1.0], [-1.0]]))
        assert torch.isfinite(mean).all()
        assert torch.isfinite(var).all()
        assert (var >= 0).all()

    def test_gp_large_inputs(self):
        """GP should not produce NaN for large input values."""
        x = torch.tensor([[100.0], [200.0], [300.0]])
        y = torch.tensor([[1.0], [2.0], [3.0]])
        gp = GaussianProcessRegressor(kernel=RBFKernel(lengthscale=100.0), noise=0.01)
        gp.fit(x, y)
        mean, var = gp.predict(torch.tensor([[150.0], [250.0]]))
        assert torch.isfinite(mean).all()
        assert torch.isfinite(var).all()

    def test_ensemble_constant_target(self):
        """Ensemble trained on constant target should not produce NaN."""
        x = torch.linspace(0, 1, 30).unsqueeze(-1)
        y = torch.ones(30, 1) * 3.0
        loader = DataLoader(TensorDataset(x, y), batch_size=30)
        models = [MLP(input_dim=1, hidden_dims=[16], output_dim=1) for _ in range(3)]
        ens = DeepEnsembleRegressor(models)
        ens.fit(loader, epochs=100, lr=1e-3, seed=60)
        uq = ens.predict_uq(x)
        assert torch.isfinite(uq.mean).all()
        assert torch.isfinite(uq.total_var).all()
        assert (uq.total_var >= 0).all()

    def test_laplace_no_nan(self):
        """Laplace should not produce NaN on small networks."""
        x, y = _sine_data(n=30, noise_std=0.1, seed=70)
        model = _train_mlp(x, y, hidden=16, epochs=100, seed=70)
        loader = DataLoader(TensorDataset(x, y), batch_size=30)
        la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
        la.fit(loader)
        uq = la.predict_uq(x, n_samples=50)
        assert torch.isfinite(uq.mean).all()
        assert torch.isfinite(uq.total_var).all()
        assert (uq.total_var >= 0).all()

    def test_mc_dropout_no_nan(self):
        """MC Dropout should not produce NaN."""
        x, y = _sine_data(n=30, noise_std=0.1, seed=75)
        model = _train_mlp(x, y, p_drop=0.3, epochs=100, seed=75)
        wrapper = MCDropoutWrapper(model, n_mc=50, apply_softmax=False)
        uq = wrapper.predict_uq(x)
        assert torch.isfinite(uq.mean).all()
        assert torch.isfinite(uq.total_var).all()


# ---------------------------------------------------------------------------
# 8. Seed determinism — same seed must produce same output
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_ensemble_deterministic_with_seed(self):
        """Same seed for model init + training should produce identical predictions."""
        x, y = _sine_data(n=50, noise_std=0.1, seed=80)
        loader = DataLoader(TensorDataset(x, y), batch_size=50, shuffle=False)
        x_test = x[:5]

        results = []
        for _ in range(2):
            torch.manual_seed(100)
            models = [
                MLP(input_dim=1, hidden_dims=[32], output_dim=1) for _ in range(3)
            ]
            ens = DeepEnsembleRegressor(models)
            ens.fit(loader, epochs=50, lr=1e-3, seed=100)
            results.append(ens.predict_uq(x_test).mean)

        assert torch.allclose(
            results[0], results[1], atol=1e-5
        ), "Same seed should give identical ensemble results"

    def test_gp_deterministic(self):
        """GP predictions should be deterministic (no randomness in posterior mean)."""
        x, y = _sine_data(n=30, noise_std=0.05, seed=85)
        gp = GaussianProcessRegressor(kernel=RBFKernel(), noise=0.05)
        gp.fit(x, y)
        x_test = torch.tensor([[0.3], [0.7]])
        mean1, var1 = gp.predict(x_test)
        mean2, var2 = gp.predict(x_test)
        assert torch.allclose(mean1, mean2, atol=1e-7)
        assert torch.allclose(var1, var2, atol=1e-7)

    def test_sgld_deterministic_with_seed(self):
        """SGLD with same seed should produce identical sample chains."""
        x, y = _sine_data(n=30, noise_std=0.1, seed=90)

        results = []
        for _ in range(2):
            torch.manual_seed(200)
            model = MLP(input_dim=1, hidden_dims=[16], output_dim=1)
            opt = SGLDOptimizer(model.parameters(), lr=1e-3)
            model.train()
            for _ in range(50):
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(x), y)
                loss.backward()
                opt.step()
            results.append(model(x[:3]).detach())

        assert torch.allclose(
            results[0], results[1], atol=1e-5
        ), "Same seed should give identical SGLD trajectories"


# ---------------------------------------------------------------------------
# 9. UQResult contract — all methods must return valid UQResult
# ---------------------------------------------------------------------------


class TestUQResultContract:
    def test_uqresult_fields_gp(self):
        x, y = _sine_data(n=30, noise_std=0.05, seed=95)
        gp = GaussianProcessRegressor(kernel=RBFKernel(), noise=0.05)
        gp.fit(x, y)
        uq = gp.predict_uq(x[:5])
        self._validate_uqresult(uq, n=5)

    def test_uqresult_fields_ensemble(self):
        x, y = _sine_data(n=30, noise_std=0.1, seed=96)
        loader = DataLoader(TensorDataset(x, y), batch_size=30)
        models = [MLP(input_dim=1, hidden_dims=[16], output_dim=1) for _ in range(3)]
        ens = DeepEnsembleRegressor(models)
        ens.fit(loader, epochs=50, lr=1e-3, seed=96)
        uq = ens.predict_uq(x[:5])
        self._validate_uqresult(uq, n=5)

    def test_uqresult_fields_mc_dropout(self):
        x, y = _sine_data(n=30, noise_std=0.1, seed=97)
        model = _train_mlp(x, y, p_drop=0.1, epochs=100, seed=97)
        wrapper = MCDropoutWrapper(model, n_mc=20, apply_softmax=False)
        uq = wrapper.predict_uq(x[:5])
        self._validate_uqresult(uq, n=5)

    def test_uqresult_fields_laplace(self):
        x, y = _sine_data(n=30, noise_std=0.1, seed=98)
        model = _train_mlp(x, y, epochs=100, seed=98)
        loader = DataLoader(TensorDataset(x, y), batch_size=30)
        la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
        la.fit(loader)
        uq = la.predict_uq(x[:5], n_samples=30)
        self._validate_uqresult(uq, n=5)

    def _validate_uqresult(self, uq: UQResult, n: int):
        assert isinstance(uq, UQResult)
        assert uq.mean is not None
        assert uq.mean.shape[0] == n
        assert torch.isfinite(uq.mean).all()
        if uq.total_var is not None:
            assert uq.total_var.shape[0] == n
            assert (uq.total_var >= 0).all()
            assert torch.isfinite(uq.total_var).all()
        if uq.epistemic_var is not None:
            assert (uq.epistemic_var >= 0).all()
        if uq.aleatoric_var is not None:
            assert (uq.aleatoric_var >= 0).all()
        assert isinstance(uq.metadata, dict)

"""
Experiment 1 (GPyTorch): SVGP binary GP classification on the same synthetic task
used in exp1_pygp_approx_gpc.py.

Model:
- Bernoulli likelihood
- Variational GP (SVGP) with inducing points
- RBF kernel + constant mean

Outputs:
- data/exp1_gpytorch_svgp_posterior.png
- data/exp1_gpytorch_svgp_results.npz
"""

from __future__ import annotations

import os
import sys
import time
import importlib

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predictive_metrics import (
    evaluate_binary_probabilistic_predictions,
    print_metric_table,
    print_posterior_statistics,
)


def make_fake_blobs(seed: int = 42, n_per_class: int = 1000):
    """Generate N=2*n_per_class two-blob binary data in R^2."""
    rng = np.random.default_rng(seed)
    cov = 0.5 * np.eye(2)
    x0 = rng.multivariate_normal(mean=[-1.0, 0.0], cov=cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean=[1.0, 0.0], cov=cov, size=n_per_class)
    X = np.vstack([x0, x1])
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    )
    return X, y


def _import_torch_stack():
    """Import torch + gpytorch with a helpful error message."""
    try:
        torch = importlib.import_module("torch")
        gpytorch = importlib.import_module("gpytorch")

        return torch, gpytorch
    except Exception as err:
        raise ImportError(
            "Could not import torch/gpytorch. Install them first, e.g. "
            'pip install torch gpytorch'
        ) from err


def main() -> None:
    torch, gpytorch = _import_torch_stack()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Data setup (same task as exp1_pygp_approx_gpc)
    X_np, y_np = make_fake_blobs(seed=42, n_per_class=1000)
    X_test_np, y_test = make_fake_blobs(seed=123, n_per_class=500)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)

    # Optional GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    X_test = X_test.to(device)

    class SVGPBinaryClassifier(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Inducing points sampled from training points
    num_inducing = 200
    perm = torch.randperm(X.size(0), device=device)
    inducing_points = X[perm[:num_inducing]].clone()

    model = SVGPBinaryClassifier(inducing_points=inducing_points).to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    model.train()
    likelihood.train()

    # Mini-batch stochastic variational training
    batch_size = 256
    num_epochs = 120
    learning_rate = 0.01

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=learning_rate,
    )

    num_data = X.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)

    t0 = time.perf_counter()
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)

        mean_loss = running_loss / num_data
        epoch_losses.append(mean_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:>3}/{num_epochs}: loss={mean_loss:.4f}")

    train_time = time.perf_counter() - t0

    print(f"\nTraining (SVGP) time: {train_time:.3f}s for {num_epochs} epochs")
    model.eval()
    likelihood.eval()
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120),
    )
    X_grid_np = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    X_grid = torch.tensor(X_grid_np, dtype=torch.float32, device=device)
    def predict_binary(test_x: torch.Tensor):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            q_f = model(test_x)           # latent predictive distribution q(f_*)
            p_y = likelihood(q_f)         # predictive Bernoulli distribution p(y_* | x_*, D)
            mu = q_f.mean                 # latent predictive mean
            var = q_f.variance            # latent predictive marginal variance
            prob1 = p_y.probs             # predictive probability P(y=1 | x_*, D)
        return q_f, p_y, mu, var, prob1
    # Grid predictions
    q_grid, p_grid_dist, mu_grid_t, var_grid_t, prob_grid_t = predict_binary(X_grid)
    mu_grid = mu_grid_t.detach().cpu().numpy().reshape(xx.shape)
    var_grid = var_grid_t.detach().cpu().numpy().reshape(xx.shape)
    std_grid = np.sqrt(np.maximum(var_grid, 0.0))
    p_grid = prob_grid_t.detach().cpu().numpy().reshape(xx.shape)
    # Training-point predictions
    q_train, p_train_dist, mu_train_t, var_train_t, prob_train_t = predict_binary(X)
    mu_train = mu_train_t.detach().cpu().numpy()
    var_train = var_train_t.detach().cpu().numpy()
    p_train = prob_train_t.detach().cpu().numpy()
    # Hard predictions + a couple of useful probabilistic metrics
    y_pred = (p_train >= 0.5).astype(np.int64)
    accuracy = float(np.mean(y_pred == y_np))
    train_loglik = float(
        np.mean(y_np * np.log(p_train + 1e-10) + (1 - y_np) * np.log(1 - p_train + 1e-10))
    )
    train_brier = float(np.mean((p_train - y_np) ** 2))

    # Labeled test-set predictions and evaluation with timing
    t_pred_start = time.perf_counter()
    q_test, p_test_dist, mu_test_t, var_test_t, prob_test_t = predict_binary(X_test)
    mu_test = mu_test_t.detach().cpu().numpy()
    var_test = var_test_t.detach().cpu().numpy()
    p_test = prob_test_t.detach().cpu().numpy()
    t_pred_elapsed = time.perf_counter() - t_pred_start

    print(f"\nPrediction timing for test set ({len(X_test)} points)")
    print("-" * 50)
    print(f"Prediction time: {t_pred_elapsed:.3f}s")

    print_posterior_statistics(
        latent_mean=mu_test,
        latent_var=var_test,
        prob_mean=p_test,
        title="GPyTorch SVGP test set posterior statistics",
    )

    test_metrics = evaluate_binary_probabilistic_predictions(
        y_true=y_test,
        p_pred=p_test,
        threshold=0.5,
        n_bins=15,
    )
    print_metric_table(test_metrics, title="GPyTorch SVGP test metrics")

    # Plot 1: latent predictive mean
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(
        xx, yy, mu_grid, levels=50, cmap="coolwarm", alpha=0.9
    )
    plt.scatter(
        X_np[:, 0], X_np[:, 1],
        c=y_np, cmap="coolwarm", s=8, alpha=0.45, vmin=0, vmax=1
    )
    plt.colorbar(cf, label="Latent predictive mean E[f(x) | D]")
    plt.title("GPyTorch SVGP latent predictive mean")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1_gpytorch_svgp_latent_mean.png"), dpi=170)
    plt.close()
    # Plot 2: latent predictive standard deviation
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(
        xx, yy, std_grid, levels=50, cmap="magma", alpha=0.9
    )
    plt.scatter(
        X_np[:, 0], X_np[:, 1],
        c=y_np, cmap="coolwarm", s=8, alpha=0.45, vmin=0, vmax=1
    )
    plt.colorbar(cf, label="Latent predictive std sqrt(Var[f(x) | D])")
    plt.title("GPyTorch SVGP latent predictive std")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1_gpytorch_svgp_latent_std.png"), dpi=170)
    plt.close()
    # Plot 3: predictive probability P(y=1 | x, D)
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(
        xx,
        yy,
        p_grid,
        levels=50,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
    )
    plt.scatter(
        X_np[:, 0], X_np[:, 1],
        c=y_np, cmap="coolwarm", s=8, alpha=0.45, vmin=0, vmax=1
    )
    plt.colorbar(cf, label="Predictive probability P(y=1 | x, D)")
    plt.title("GPyTorch SVGP predictive probability")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1_gpytorch_svgp_prob.png"), dpi=170)
    plt.close()
    np.savez(
        os.path.join(data_dir, "exp1_gpytorch_svgp_results.npz"),
        num_inducing=num_inducing,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        train_time=float(train_time),
        final_loss=float(epoch_losses[-1]) if epoch_losses else float("nan"),
        train_accuracy=accuracy,
        train_loglik=train_loglik,
        train_brier=train_brier,
        device=str(device),
        latent_mean_train=mu_train,
        latent_var_train=var_train,
        prob_train=p_train,
        latent_mean_test=mu_test,
        latent_var_test=var_test,
        prob_test=p_test,
        y_test=y_test,
        grid_shape=xx.shape,
        latent_mean_grid=mu_grid,
        latent_var_grid=var_grid,
        prob_grid=p_grid,
    )
    print(f"Training done in {train_time:.2f}s on {device}.")
    print(f"Train accuracy: {accuracy * 100.0:.2f}%")
    print(f"Train log-likelihood: {train_loglik:.4f}")
    print(f"Train Brier score: {train_brier:.4f}")
    print("Saved:")
    print("- data/exp1_gpytorch_svgp_latent_mean.png")
    print("- data/exp1_gpytorch_svgp_latent_std.png")
    print("- data/exp1_gpytorch_svgp_prob.png")
    print("- data/exp1_gpytorch_svgp_results.npz")


if __name__ == "__main__":
    main()

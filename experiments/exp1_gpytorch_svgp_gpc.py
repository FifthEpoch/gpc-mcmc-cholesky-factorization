"""
Experiment 1 (GPyTorch): SVGP binary GP classification on the same synthetic task
used in exp1_pygp_approx_gpc.py.

Model:
- Bernoulli likelihood
- Variational GP (SVGP) with inducing points
- RBF kernel + constant mean

Outputs:
- data/exp1_gpytorch_svgp_posterior.png
- data/exp1_gpytorch_svgp_results.npy
"""

from __future__ import annotations

import os
import time
import importlib

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_fake_blobs(seed: int = 42):
    """Generate N=2000 two-blob binary data in R^2."""
    rng = np.random.default_rng(seed)
    n_per_class = 1000
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
    X_np, y_np = make_fake_blobs(seed=42)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    # Optional GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

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

    # Predict posterior probabilities on a 2D grid
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

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_dist = model(X_grid)
        pred_dist = likelihood(latent_dist)
        p_grid = pred_dist.probs.detach().cpu().numpy().reshape(xx.shape)

        train_latent = model(X)
        train_pred = likelihood(train_latent)
        p_train = train_pred.probs.detach().cpu().numpy()

    y_pred = (p_train >= 0.5).astype(np.int64)
    accuracy = float(np.mean(y_pred == y_np))

    # Plot posterior + data points
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
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="coolwarm", s=8, alpha=0.45, vmin=0, vmax=1)
    plt.colorbar(cf, label="Posterior probability P(y=1)")
    plt.title("GPyTorch SVGP posterior (binary classification)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "exp1_gpytorch_svgp_posterior.png"), dpi=170)
    plt.close()

    np.save(
        os.path.join(data_dir, "exp1_gpytorch_svgp_results.npy"),
        {
            "num_inducing": num_inducing,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "train_time": float(train_time),
            "final_loss": float(epoch_losses[-1]) if epoch_losses else float("nan"),
            "train_accuracy": accuracy,
            "device": str(device),
        },
        allow_pickle=True,
    )

    print(f"Training done in {train_time:.2f}s on {device}.")
    print(f"Train accuracy: {accuracy * 100.0:.2f}%")
    print("Saved:")
    print("- data/exp1_gpytorch_svgp_posterior.png")
    print("- data/exp1_gpytorch_svgp_results.npy")


if __name__ == "__main__":
    main()

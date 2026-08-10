# Synthetic conditional-expectation experiments for the smearing mechanism
# Author: Shengning Wang

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsn_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random


class MiniCNN(eqx.Module):
    """Small residual CNN used only for the synthetic experiment."""

    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    out: eqx.nn.Conv1d

    def __init__(self, key: jax.Array, width: int = 32) -> None:
        k1, k2, k3 = random.split(key, 3)
        self.conv1 = eqx.nn.Conv1d(1, width, 5, padding=2, key=k1)
        self.conv2 = eqx.nn.Conv1d(width, width, 5, padding=2, key=k2)
        self.out = eqx.nn.Conv1d(width, 1, 1, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to a field. (B, 1, N) -> (B, 1, N)."""
        h = jax.nn.gelu(jax.vmap(self.conv1)(x))
        h = jax.nn.gelu(jax.vmap(self.conv2)(h))
        return x + 0.1 * jax.vmap(self.out)(h)


def _step_batch(key: jax.Array, n: int, n_grid: int, sigma: float, u_l: float, u_r: float) -> jnp.ndarray:
    """Random-position step fields. (N, 1, N_GRID)."""
    pos = n_grid * jax.random.normal(key, (n, 1))
    pos = pos % n_grid
    x = jnp.arange(n_grid, dtype=jnp.float32)
    return jnp.where(x[None, None, :] > pos[:, None, :], u_r, u_l)


def _ramp_width(u: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    """10%-90% ramp width in cells."""
    x = np.arange(u.size)
    vmin = u.min()
    vmax = u.max()
    span = vmax - vmin
    if span < 1e-8:
        return 0.0
    level_low = vmin + low * span
    level_high = vmin + high * span
    x_low = np.interp(level_low, u, x)
    x_high = np.interp(level_high, u, x)
    return float(x_high - x_low)


def fit_synthetic_mse(
    n_samples: int = 8192,
    n_grid: int = 256,
    sigma: float = 1.0,
    seed: int = 0,
    steps: int = 200,
    batch: int = 256,
) -> float:
    """Train a small CNN on jittered-step pairs and return the fitted ramp width.

    The input is a step at position s + eps and the target is the step at
    position s, with eps ~ N(0, sigma). MSE training pushes the predictor to
    the conditional expectation over eps, a ramp whose width grows with sigma.
    """
    key = random.PRNGKey(seed)
    model = MiniCNN(key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model)

    def make_batch(key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        k1, k2 = random.split(key)
        pos = n_grid * random.uniform(k1, (batch, 1))
        eps = sigma * random.normal(k2, (batch, 1))
        x = jnp.arange(n_grid, dtype=jnp.float32)
        target = jnp.where(x[None, None, :] > pos[:, None, :], 1.0, -1.0)
        source = jnp.where(x[None, None, :] > ((pos + eps) % n_grid)[:, None, :], 1.0, -1.0)
        return source, target

    @eqx.filter_jit
    def step(params, opt_state, key):
        k1, k2 = random.split(key)
        source, target = make_batch(k1)

        def loss_fn(p):
            return jnp.mean((p(source) - target) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    for _ in range(steps):
        key, sk = random.split(key)
        model, opt_state, _ = step(model, opt_state, sk)

    key, sk = random.split(key)
    probe = _step_batch(sk, 1, n_grid, 0.0, -1.0, 1.0)
    pred = np.asarray(model(probe)[0, 0])
    return _ramp_width(pred)


def run_synthetic_ce(
    sigmas: list[float],
    n_samples: int,
    n_grid: int,
    seed: int,
    out_dir: str | Path,
) -> dict:
    """Run the synthetic conditional-expectation experiment.

    Args:
        sigmas (list[float]): Position uncertainty levels in cells.
        n_samples (int): Training samples per sigma.
        n_grid (int): Grid resolution.
        seed (int): Random seed.
        out_dir (str | Path): Output directory for JSON and figure.

    Returns:
        dict: sigmas, analytic_widths, fitted_widths, figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analytic = [2.0 * s * 1.2816 for s in sigmas]
    fitted = [fit_synthetic_mse(n_samples=n_samples, n_grid=n_grid, sigma=s, seed=seed + i) for i, s in enumerate(sigmas)]
    result = {
        "sigmas": sigmas,
        "analytic_widths": analytic,
        "fitted_widths": fitted,
        "n_samples": n_samples,
        "n_grid": n_grid,
    }
    with open(out_dir / "synthetic_ce.json", "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(sigmas, analytic, "k--", label="analytic 2.56 sigma")
    ax.plot(sigmas, fitted, "o-", label="fitted (MSE CNN)")
    ax.set_xlabel("position uncertainty sigma (cells)")
    ax.set_ylabel("10%-90% ramp width (cells)")
    ax.set_title("Conditional-expectation smearing")
    ax.legend()
    fig.tight_layout()
    figure_path = out_dir / "synthetic_ce.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    result["figure"] = str(figure_path)
    return result

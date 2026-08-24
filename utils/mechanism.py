# Mechanism experiments: conditional expectation, scaling law, flow-matching end width
# Author: Shengning Wang

import json
from pathlib import Path

import torch
from torch import nn

from data.synthetic import ce_pairs, fm_pairs, ot_permutation, step_fields, velocity_label
from models.velocity import FlowUNet
from utils.hue_logger import logger
from utils.metrics import ramp_width_1d


class MiniCNN(nn.Module):
    """Small residual 1D CNN used only for the synthetic experiments. (B, 1, N) -> (B, 1, N)."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, width, 5, padding=2)
        self.conv2 = nn.Conv1d(width, width, 5, padding=2)
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.conv1(x))
        h = torch.nn.functional.gelu(self.conv2(h))
        return x + 0.1 * self.out(h)


def fit_mse_ramp(
    model: nn.Module,
    sigma: float,
    n_grid: int,
    jump: float,
    steps: int,
    batch: int,
    device: str,
) -> float:
    """Train a model on jittered-step pairs and return the fitted ramp width at a probe step.

    MSE training pushes the predictor to the conditional expectation over the
    position jitter, a ramp whose width grows with sigma.
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        source, target = ce_pairs(batch, n_grid, sigma, jump, device)
        opt.zero_grad()
        loss = torch.mean((model(source) - target) ** 2)
        loss.backward()
        opt.step()
    model.eval()
    probe = step_fields(torch.full((1,), n_grid / 2, device=device), n_grid, jump)
    with torch.no_grad():
        pred = model(probe)[0, 0].cpu().numpy()
    return ramp_width_1d(pred)


def run_synthetic_ce(
    sigmas: list[float],
    n_grid: int,
    jump: float,
    steps: int,
    batch: int,
    seed: int,
    device: str,
    out_dir: str | Path,
) -> dict:
    """Width vs sigma with a MiniCNN, against the analytic 2.56 sigma line (prediction 2).

    Args:
        sigmas (list[float]): Position uncertainty levels in cells.
        n_grid (int): Grid resolution.
        jump (float): Step amplitude.
        steps (int): Training steps per sigma.
        batch (int): Batch size.
        seed (int): Random seed.
        device (str): Torch device.
        out_dir (str | Path): Output directory for JSON and figure.

    Returns:
        dict: sigmas, analytic_widths, fitted_widths, figure.
    """
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analytic = [2.0 * s * 1.2816 for s in sigmas]
    fitted = [fit_mse_ramp(MiniCNN(), s, n_grid, jump, steps, batch, device) for s in sigmas]
    result = {
        "sigmas": sigmas,
        "analytic_widths": analytic,
        "fitted_widths": fitted,
        "n_grid": n_grid,
        "jump": jump,
    }
    with open(out_dir / "synthetic_ce.json", "w") as f:
        json.dump(result, f, indent=2)
    result["figure"] = str(out_dir / "synthetic_ce.png")
    logger.info(f"mechanism: synthetic CE saved to {out_dir}")
    return result


def run_j_sweep(
    sigma: float,
    jumps: list[float],
    n_grid: int,
    steps: int,
    batch: int,
    seed: int,
    device: str,
) -> dict:
    """Fitted ramp width vs jump height at fixed sigma (prediction 2, J-independence)."""
    torch.manual_seed(seed)
    widths = [fit_mse_ramp(MiniCNN(), sigma, n_grid, j, steps, batch, device) for j in jumps]
    return {"sigma": sigma, "jumps": jumps, "fitted_widths": widths}


def run_cross_arch(
    factories: dict,
    sigma: float,
    n_grid: int,
    jump: float,
    steps: int,
    batch: int,
    seed: int,
    device: str,
) -> dict:
    """Fitted ramp width across architectures at fixed sigma (predictions 1 and 2, cross-arch).

    Args:
        factories (dict): name -> callable() returning an nn.Module.
        sigma (float): Position uncertainty in cells.
        n_grid (int): Grid resolution.
        jump (float): Step amplitude.
        steps (int): Training steps per architecture.
        batch (int): Batch size.
        seed (int): Random seed.
        device (str): Torch device.

    Returns:
        dict: sigma, widths (name -> width), analytic_width.
    """
    torch.manual_seed(seed)
    widths = {}
    for name, factory in factories.items():
        widths[name] = fit_mse_ramp(factory(), sigma, n_grid, jump, steps, batch, device)
    return {"sigma": sigma, "widths": widths, "analytic_width": 2.0 * sigma * 1.2816}


def _fm_train_step(path: str, net: nn.Module, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """One flow-matching training step for a path family, returning the loss."""
    tau = torch.rand(())
    if path == "ot":
        s0 = (x0 > 0).float().argmax(dim=-1)
        s1 = (x1 > 0).float().argmax(dim=-1)
        x1 = x1[ot_permutation(s0, s1)]
    x_tau = (1 - tau) * x0 + tau * x1
    v_star = velocity_label(path, x0, x1, tau)
    v_pred = net(x_tau, tau)
    return torch.mean((v_pred - v_star) ** 2)


def _fm_endpoint_width(
    path: str,
    sigma: float,
    n_grid: int,
    jump: float,
    steps: int,
    batch: int,
    seed: int,
    device: str,
    k_steps: int = 16,
) -> float:
    """Train a velocity network on FM pairs and measure the deterministic ODE endpoint width."""
    torch.manual_seed(seed)
    net = FlowUNet(1, 1, None, width=32, depth=3).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(steps):
        x0, x1 = fm_pairs(batch, n_grid, sigma, jump, device)
        opt.zero_grad()
        loss = _fm_train_step(path, net, x0, x1)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        x = step_fields(torch.full((1,), n_grid / 2, device=device), n_grid, jump)
        for k in range(k_steps):
            tau = (k + 0.5) / k_steps
            x = x + net(x, tau) / k_steps
        pred = x[0, 0].cpu().numpy()
    return ramp_width_1d(pred)


def run_fm_end_width(
    paths: list[str],
    sigmas: list[float],
    n_grid: int,
    jump: float,
    steps: int,
    batch: int,
    seed: int,
    device: str,
) -> dict:
    """Deterministic ODE endpoint width vs sigma per path family (prediction 4).

    Args:
        paths (list[str]): Path families in {"straight", "ot", "transport"}.
        sigmas (list[float]): Target displacement spread in cells.
        n_grid (int): Grid resolution.
        jump (float): Step amplitude.
        steps (int): Training steps per (path, sigma).
        batch (int): Batch size.
        seed (int): Random seed.
        device (str): Torch device.

    Returns:
        dict: path -> {"sigmas": [...], "widths": [...]}.
    """
    result = {}
    for path in paths:
        widths = [_fm_endpoint_width(path, s, n_grid, jump, steps, batch, seed, device) for s in sigmas]
        result[path] = {"sigmas": sigmas, "widths": widths}
    return result

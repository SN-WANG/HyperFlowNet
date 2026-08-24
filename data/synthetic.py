# Synthetic step-family data for the smearing mechanism
# Author: Shengning Wang

import torch


def step_fields(positions: torch.Tensor, n_grid: int, jump: float = 1.0) -> torch.Tensor:
    """Step fields of amplitude jump at random positions. (B,) -> (B, 1, N)."""
    x = torch.arange(n_grid, dtype=positions.dtype, device=positions.device)
    return torch.where(x[None, :] > positions[:, None], jump, -jump)[:, None]


def warp_1d(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Shift a 1D field by per-sample scalars via linear interpolation. (B, C, N), (B,) -> (B, C, N)."""
    n = x.shape[-1]
    idx = torch.arange(n, dtype=x.dtype, device=x.device)[None] - shift[:, None]
    idx = idx % n
    i0 = idx.floor().long()
    i1 = (i0 + 1) % n
    frac = (idx - i0.float()).unsqueeze(1)
    return x.gather(-1, i0.expand_as(x)) * (1 - frac) + x.gather(-1, i1.expand_as(x)) * frac


def ce_pairs(batch: int, n_grid: int, sigma: float, jump: float, device: str) -> tuple:
    """Conditional-expectation pairs: source step at s+eps, target step at s. (B, 1, N) each."""
    pos = n_grid * torch.rand(batch, device=device)
    eps = sigma * torch.randn(batch, device=device)
    target = step_fields(pos, n_grid, jump)
    source = step_fields((pos + eps) % n_grid, n_grid, jump)
    return source, target


def fm_pairs(batch: int, n_grid: int, sigma: float, jump: float, device: str) -> tuple:
    """Flow-matching pairs: source step at s, target step at s+delta. (B, 1, N) each."""
    pos = n_grid * torch.rand(batch, device=device)
    delta = sigma * torch.randn(batch, device=device)
    x0 = step_fields(pos, n_grid, jump)
    x1 = step_fields((pos + delta) % n_grid, n_grid, jump)
    return x0, x1


def ot_permutation(source_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    """Rank-match (monotone rearrangement) pairing indices for 1D OT. (B,) -> (B,)."""
    return torch.argsort(target_pos)[torch.argsort(torch.argsort(source_pos))]


def velocity_label(path: str, x0: torch.Tensor, x1: torch.Tensor, tau: float, eps: float = 1e-3) -> torch.Tensor:
    """Velocity label of a path family at tau. (B, 1, N) each.

    Args:
        path (str): One of "straight", "ot", "transport".
        x0 (torch.Tensor): Source field. (B, C, N).
        x1 (torch.Tensor): Target field. (B, C, N).
        tau (float): Flow-matching time in [0, 1].
        eps (float): Finite-difference step for the transport path.

    Returns:
        torch.Tensor: Velocity label. (B, C, N).
    """
    if path == "transport":
        positions = (x0 > 0).float().argmax(dim=-1)
        targets = (x1 > 0).float().argmax(dim=-1)
        shift = (targets - positions).float()
        x_p = warp_1d(x0, (tau + eps) * shift)
        x_m = warp_1d(x0, (tau - eps) * shift)
        return (x_p - x_m) / (2 * eps)
    return x1 - x0

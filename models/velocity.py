# Conditional velocity networks for flow matching
# Author: Shengning Wang

import torch
import torch.nn.functional as F
from torch import nn

from models.operators import FNO, UNet


def sinkhorn_plan(cost: torch.Tensor, epsilon: float = 1e-2, iters: int = 20) -> torch.Tensor:
    """Entropy-regularized OT plan from a normalized cost matrix. (B, B) -> (B, B)."""
    k = torch.exp(-cost / (epsilon + 1e-8))
    u = torch.ones(cost.shape[0], device=cost.device)
    for _ in range(iters):
        v = 1.0 / (k.T @ u + 1e-8)
        u = 1.0 / (k @ v + 1e-8)
    return u[:, None] * k * v[None, :]


class FlowUNet(nn.Module):
    """Conditional velocity network on a U-Net backbone. (B, C, *S), tau, c -> (B, C, *S)."""

    def __init__(
        self,
        c_in: int,
        ndim: int,
        cfg: dict | None = None,
        width: int | None = None,
        depth: int | None = None,
        n_params: int = 0,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        w = width or int(cfg.get("width", 32))
        d = depth or int(cfg.get("depth", 4))
        self.c_in = c_in
        self.ndim = ndim
        self.n_params = n_params
        self.inject = nn.Linear(n_params, c_in + 1) if n_params > 0 else None
        self.net = UNet(c_in + 1, ndim, cfg, c_out=c_in, width=w, depth=d)

    def forward(self, x: torch.Tensor, tau: float, c: torch.Tensor | None = None) -> torch.Tensor:
        """Return the conditional velocity field.

        Args:
            x (torch.Tensor): Current state. (B, C, *S).
            tau (float): Flow-matching time.
            c (torch.Tensor | None): Conditioning vector. (B, F).

        Returns:
            torch.Tensor: Velocity. (B, C, *S).
        """
        h = torch.cat([x, torch.full_like(x[:, :1], float(tau))], dim=1)
        if c is not None:
            bias = self.inject(c).view(c.shape[0], -1, *([1] * (h.ndim - 2)))
            h = h + bias
        return self.net(h)


class FlowNO(nn.Module):
    """Conditional velocity network on an FNO backbone."""

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, n_params: int = 0) -> None:
        super().__init__()
        self.c_in = c_in
        self.ndim = ndim
        self.n_params = n_params
        self.inject = nn.Linear(n_params, c_in + 1) if n_params > 0 else None
        self.net = FNO(c_in + 1, ndim, cfg, c_out=c_in)

    def forward(self, x: torch.Tensor, tau: float, c: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.cat([x, torch.full_like(x[:, :1], float(tau))], dim=1)
        if c is not None:
            bias = self.inject(c).view(c.shape[0], -1, *([1] * (h.ndim - 2)))
            h = h + bias
        return self.net(h)


class ContextEncoder(nn.Module):
    """History window encoder producing a conditioning vector. (B, H*C, *S) -> (B, F)."""

    def __init__(self, c_hist: int, ndim: int, cfg: dict | None = None, dim: int = 64) -> None:
        super().__init__()
        cfg = cfg or {}
        self.ndim = ndim
        self.dim = dim
        self.net = UNet(c_hist, ndim, cfg, c_out=dim, width=int(cfg.get("width", 32)), depth=int(cfg.get("depth", 2)))
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """Encode a history window. (B, H*C, *S) -> (B, F)."""
        h = self.net(x_hist)
        h = F.adaptive_avg_pool1d(h, 1).flatten(1) if self.ndim == 1 else F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.head(h)


class ConditionalFlowModel(nn.Module):
    """Context-conditioned flow-matching surrogate with straight or OT labels.

    Standard CFM (straight path, independent coupling) and OT-CFM differ only
    in the coupling used to build labels, selected by ``mode``.
    """

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, history: int = 1) -> None:
        super().__init__()
        cfg = cfg or {}
        self.c_in = c_in
        self.ndim = ndim
        self.history = history
        self.context = ContextEncoder(c_in * history, ndim, cfg)
        self.velocity = FlowUNet(c_in, ndim, cfg, n_params=self.context.dim)

    def forward(self, x_hist: torch.Tensor, x1: torch.Tensor, tau: float, mode: str = "straight") -> torch.Tensor:
        """Flow-matching loss for one batch. (B, H*C, *S), (B, C, *S) -> scalar."""
        c = self.context(x_hist)
        x0 = x_hist[:, -self.c_in :]
        x_tau, v_star = self.labels(x0, x1, tau, mode)
        v = self.velocity(x_tau, tau, c)
        return torch.mean((v - v_star) ** 2)

    def labels(self, x0: torch.Tensor, x1: torch.Tensor, tau: float, mode: str) -> tuple:
        """Straight-path labels with independent or OT coupling. (B, C, *S) each."""
        if mode == "ot":
            x1 = self._ot_pair(x0, x1)
        x_tau = (1 - tau) * x0 + tau * x1
        return x_tau, x1 - x0

    def _ot_pair(self, x0: torch.Tensor, x1: torch.Tensor, epsilon: float = 1e-2, iters: int = 20) -> torch.Tensor:
        """Mini-batch Sinkhorn coupling: pair each source with its most probable target. (B, C, *S)."""
        flat0 = x0.flatten(1)
        flat1 = x1.flatten(1)
        c0 = (flat0 * flat0).sum(-1)
        c1 = (flat1 * flat1).sum(-1)
        cost = (c0[:, None] + c1[None, :] - 2.0 * flat0 @ flat1.T) / flat0.shape[-1]
        plan = sinkhorn_plan(cost, epsilon, iters)
        return x1[plan.argmax(dim=1)]

    def predict(self, x_hist: torch.Tensor, k_steps: int = 8) -> torch.Tensor:
        """Deterministic probability-flow ODE from the last history frame. (B, C, *S)."""
        c = self.context(x_hist)
        x = x_hist[:, -self.c_in :]
        with torch.no_grad():
            for k in range(k_steps):
                tau = (k + 0.5) / k_steps
                x = x + self.velocity(x, tau, c) / k_steps
        return x

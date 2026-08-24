# HyperFlowNet: single-stage path-level discontinuity-preserving flow matching
# Author: Shengning Wang

import torch
import torch.nn.functional as F
from torch import nn

from models.blocks import haar_1d, haar_2d
from models.velocity import ContextEncoder, FlowUNet


class HyperFlowNet(nn.Module):
    """Single-stage, path-level flow-matching surrogate with front transport.

    Five components: context encoder, front mask, displacement field
    (phase correlation), transport interpolation path, and velocity network.
    Inference is a deterministic probability-flow ODE.
    """

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, history: int = 1) -> None:
        super().__init__()
        cfg = cfg or {}
        self.c_in = c_in
        self.ndim = ndim
        self.history = history
        self.beta = float(cfg.get("mask_beta", 10.0))
        self.kappa = float(cfg.get("mask_kappa", 0.5))
        self.flow_steps = int(cfg.get("flow_steps", 8))
        self.context = ContextEncoder(c_in * history, ndim, cfg)
        self.velocity = FlowUNet(c_in, ndim, cfg, n_params=self.context.dim)

    def front_mask(self, x0: torch.Tensor) -> torch.Tensor:
        """Front-localization mask in [0, 1]. (B, C, *S) -> (B, 1, *S)."""
        if self.ndim == 1:
            _, hi = haar_1d(x0).chunk(2, dim=-1)
            d = hi.repeat_interleave(2, dim=-1)
        else:
            c = haar_2d(x0)
            d = c[:, self.c_in :].mean(dim=1, keepdim=True)
            d = torch.repeat_interleave(torch.repeat_interleave(d, 2, dim=-1), 2, dim=-2)
        if d.shape[1] > 1:
            d = d.mean(dim=1, keepdim=True)
        m = d.abs()
        m = m / (m.amax(dim=tuple(range(2, m.ndim)), keepdim=True) + 1e-8)
        return torch.sigmoid(self.beta * (m - self.kappa))

    def phase_shift(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Sub-pixel phase-correlation shift between fields. (B, C, *S) -> (B, ndim) in cells."""
        a = x0[:, :1]
        b = x1[:, :1]
        if self.ndim == 1:
            n = a.shape[-1]
            corr = torch.fft.irfft(torch.fft.rfft(a) * torch.conj(torch.fft.rfft(b)), n=n)
            peak = corr.argmax(dim=-1)
            p0 = corr.gather(-1, peak.clamp(1, n - 2).unsqueeze(-1)).squeeze(-1)
            p1 = corr.gather(-1, (peak - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1)
            p2 = corr.gather(-1, (peak + 1).clamp(max=n - 1).unsqueeze(-1)).squeeze(-1)
            delta = 0.5 * (p1 - p2) / (p1 - 2 * p0 + p2 + 1e-8)
            shift = peak + delta
            return ((shift + n / 2) % n - n / 2).reshape(-1)
        h, w = a.shape[-2:]
        corr = torch.fft.irfft2(torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(b)), s=(h, w))
        flat = corr.flatten(1)  # (B, H*W)
        idx = flat.argmax(dim=-1)
        py = idx // w
        px = idx % w
        return torch.stack([(py + h / 2) % h - h / 2, (px + w / 2) % w - w / 2], dim=-1)

    def warp(self, x: torch.Tensor, shift: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Translate a field by scale * shift. (B, C, *S), (B, ndim) -> (B, C, *S)."""
        if self.ndim == 1:
            n = x.shape[-1]
            s = shift.reshape(-1)
            idx = torch.arange(n, device=x.device)[None, None] - scale * s[:, None, None]
            idx = idx % n
            i0 = idx.floor().long()
            i1 = (i0 + 1) % n
            frac = idx - i0.float()  # (B, 1, N)
            return x.gather(-1, i0) * (1 - frac) + x.gather(-1, i1) * frac
        b, c, h, w = x.shape
        gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        gy = (gy.float() - scale * shift[:, 0, None, None]) / (h - 1) * 2.0 - 1.0
        gx = (gx.float() - scale * shift[:, 1, None, None]) / (w - 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).to(x.device)
        return F.grid_sample(x, grid, mode="bilinear", align_corners=True)

    def _labels(self, x0: torch.Tensor, x1: torch.Tensor, tau: float, eps: float = 1e-3) -> tuple:
        """Transport-path interpolation and velocity label. (B, C, *S) each."""
        m = self.front_mask(x0)
        shift = self.phase_shift(x0, x1)
        x_tau = (1 - m) * ((1 - tau) * x0 + tau * x1) + m * self.warp(x0, shift, tau)
        v_trans = (self.warp(x0, shift, tau + eps) - self.warp(x0, shift, tau - eps)) / (2 * eps)
        v_star = (1 - m) * (x1 - x0) + m * v_trans
        return x_tau, v_star

    def train_loss(self, x_hist: torch.Tensor, x1: torch.Tensor, tau: float) -> torch.Tensor:
        """Flow-matching loss on the transport path. (B, H*C, *S), (B, C, *S) -> scalar."""
        c = self.context(x_hist)
        x0 = x_hist[:, -self.c_in :]
        x_tau, v_star = self._labels(x0, x1, tau)
        v = self.velocity(x_tau, tau, c)
        return torch.mean((v - v_star) ** 2)

    def advance(self, x_hist: torch.Tensor, k_steps: int | None = None) -> torch.Tensor:
        """Deterministic probability-flow ODE from the last history frame. (B, C, *S)."""
        c = self.context(x_hist)
        x = x_hist[:, -self.c_in :]
        k = k_steps or self.flow_steps
        for i in range(k):
            tau = (i + 0.5) / k
            x = x + self.velocity(x, tau, c) / k
        return x

    def predict(self, x_hist: torch.Tensor, k_steps: int | None = None) -> torch.Tensor:
        """Inference alias of advance for a uniform model interface."""
        return self.advance(x_hist, k_steps)

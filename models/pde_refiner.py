# PDE-Refiner baseline: denoising refinement of autoregressive predictions
# Author: Shengning Wang

import torch
from torch import nn

from models.operators import UNet


class PDERefiner(nn.Module):
    """Base predictor plus a noise-level-conditioned denoiser (Lippe et al. 2023).

    Training uses an MSE stage for the base predictor and a denoising stage for
    the refiner; inference refines the base prediction over decreasing noise
    levels.
    """

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, c_out: int | None = None) -> None:
        super().__init__()
        cfg = cfg or {}
        self.c_in = c_in
        self.c_out = c_in if c_out is None else c_out
        self.ndim = ndim
        self.base = UNet(c_in, ndim, cfg, c_out=self.c_out)
        self.denoiser = UNet(c_in + self.c_out + 1, ndim, cfg, c_out=self.c_out)
        self.refine_steps = int(cfg.get("refine_steps", 4))

    def denoise(self, x: torch.Tensor, y_noisy: torch.Tensor, sigma: float) -> torch.Tensor:
        """Denoise a corrupted prediction. (B, C, *S) each -> (B, C, *S)."""
        sigma_ch = torch.full_like(x[:, :1], float(sigma))
        return self.denoiser(torch.cat([x, y_noisy, sigma_ch], dim=1))

    def forward(self, x: torch.Tensor, y0: torch.Tensor | None = None) -> torch.Tensor:
        """Predict the next field, refining y0 or the base prediction. (B, C, *S)."""
        y = self.base(x) if y0 is None else y0
        for k in range(self.refine_steps):
            y = self.denoise(x, y, 0.5 ** (k + 1))
        return y

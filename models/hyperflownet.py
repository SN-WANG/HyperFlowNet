# HyperFlowNet: UWNO backbone with a flow-matching residual corrector
# Author: Shengning Wang

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from models.corrector import FlowNO, FlowUNet
from models.operators import UWNO


class HyperFlowNet(eqx.Module):
    """UWNO backbone plus a deterministic flow-matching shape corrector."""

    backbone: UWNO
    corrector: eqx.Module
    flow_steps: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        k1, k2 = random.split(key)
        self.backbone = UWNO(k1, c_in, ndim, cfg)
        if cfg.get("corrector", "fno") == "fno":
            self.corrector = FlowNO(k2, c_in, ndim, cfg)
        else:
            self.corrector = FlowUNet(k2, c_in, ndim, cfg)
        self.flow_steps = int(cfg.get("flow_steps", 8))

    def backbone_call(self, x: jnp.ndarray) -> jnp.ndarray:
        """Backbone prediction without correction. (B, C, *S)."""
        return self.backbone(x)

    def velocity(self, c: jnp.ndarray, x: jnp.ndarray, s: float) -> jnp.ndarray:
        """Correction velocity. (B, C, *S) each."""
        return self.corrector(c, x, s)

    def advance(self, c: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """Backbone prediction followed by deterministic Euler correction."""
        u = self.backbone(c)
        for k in range(self.flow_steps):
            s = (k + 1) / self.flow_steps
            u = u + self.corrector(c, u, s) / self.flow_steps
        return u

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to the next field. (B, C, *S) -> (B, C, *S)."""
        return self.advance(x, None)

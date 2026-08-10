# Flow-matching correctors and deterministic sampling
# Author: Shengning Wang

import equinox as eqx
import jax
import jax.numpy as jnp

from models.operators import FNO, UNet


def _concat_condition(c: jnp.ndarray, x: jnp.ndarray, s: float) -> jnp.ndarray:
    """Concatenate condition, state, and time channel. (B, 2C+1, *S)."""
    s_ch = jnp.full_like(c, s)
    return jnp.concatenate([c, x, s_ch], axis=1)


class FlowNO(eqx.Module):
    """Conditional flow-matching velocity network built on FNO."""

    backbone: FNO

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        self.backbone = FNO(key, 2 * c_in + 1, ndim, cfg, c_out=c_in)

    def __call__(self, c: jnp.ndarray, x: jnp.ndarray, s: float) -> jnp.ndarray:
        """Return the conditional velocity field. (B, C, *S) each."""
        return self.backbone(_concat_condition(c, x, s))


class FlowUNet(eqx.Module):
    """Conditional flow-matching velocity network built on U-Net."""

    backbone: UNet

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        self.backbone = UNet(key, 2 * c_in + 1, ndim, cfg, c_out=c_in)

    def __call__(self, c: jnp.ndarray, x: jnp.ndarray, s: float) -> jnp.ndarray:
        """Return the conditional velocity field. (B, C, *S) each."""
        return self.backbone(_concat_condition(c, x, s))


class DiffNO(eqx.Module):
    """Conditional denoising operator trained on a cosine-schedule DDPM."""

    net: UNet

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        self.net = UNet(key, 2 * c_in + 1, ndim, cfg, c_out=c_in)

    def __call__(self, c: jnp.ndarray, x_t: jnp.ndarray, s: float) -> jnp.ndarray:
        """Predict the clean target from a noisy state. (B, C, *S) each."""
        return self.net(_concat_condition(c, x_t, s))


def make_advance(corrector, flow_steps: int):
    """Deterministic Euler integration of a conditional velocity field."""

    @eqx.filter_jit
    def advance(c: jnp.ndarray, u0: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        u = u0
        for k in range(flow_steps):
            s = (k + 1) / flow_steps
            u = u + corrector(c, u, s) / flow_steps
        return u

    return advance

# Shared building blocks for 1D/2D neural operators
# Author: Shengning Wang

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random


# ============================================================
# Convolutions and pooling
# ============================================================


def _conv(
    ndim: int,
    key: jax.Array,
    c_in: int,
    c_out: int,
    k: int = 3,
) -> eqx.Module:
    """Create a 1D or 2D convolution with same padding."""
    if ndim == 1:
        return eqx.nn.Conv1d(c_in, c_out, k, padding=k // 2, key=key)
    return eqx.nn.Conv2d(c_in, c_out, k, padding=k // 2, key=key)


def _cv(conv: eqx.Module, x: jnp.ndarray) -> jnp.ndarray:
    """Apply a convolution over a batched input. (B, C, *S) -> (B, C_OUT, *S)."""
    return jax.vmap(conv)(x)


def _pool(x: jnp.ndarray) -> jnp.ndarray:
    """Average-pool by a factor of 2 on all spatial axes."""
    if x.ndim == 3:
        return 0.5 * (x[..., 0::2] + x[..., 1::2])
    return 0.25 * (
        x[..., 0::2, 0::2]
        + x[..., 0::2, 1::2]
        + x[..., 1::2, 0::2]
        + x[..., 1::2, 1::2]
    )


def _up(x: jnp.ndarray) -> jnp.ndarray:
    """Nearest-neighbor up-sample by a factor of 2 on all spatial axes."""
    if x.ndim == 3:
        return jnp.repeat(x, 2, axis=-1)
    return jnp.repeat(jnp.repeat(x, 2, axis=-1), 2, axis=-2)


class ConvBlock(eqx.Module):
    """Two convolutions with GELU activations."""

    conv1: eqx.Module
    conv2: eqx.Module

    def __init__(self, ndim: int, key: jax.Array, c_in: int, c_out: int, k: int = 3) -> None:
        k1, k2 = random.split(key)
        self.conv1 = _conv(ndim, k1, c_in, c_out, k=k)
        self.conv2 = _conv(ndim, k2, c_out, c_out, k=k)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply Conv-GELU-Conv-GELU. (B, C_IN, *S) -> (B, C_OUT, *S)."""
        h = jax.nn.gelu(_cv(self.conv1, x))
        return jax.nn.gelu(_cv(self.conv2, h))


# ============================================================
# Spectral convolution
# ============================================================


class SpectralConv(eqx.Module):
    """Spectral multiplication plus local mixing for 1D or 2D fields."""

    w_r: jax.Array
    w_i: jax.Array
    local: eqx.Module
    modes: int = eqx.field(static=True)
    ndim: int = eqx.field(static=True)

    def __init__(self, ndim: int, key: jax.Array, width: int, modes: int) -> None:
        k1, k2, k3 = random.split(key, 3)
        if ndim == 1:
            shape = (width, width, modes)
        else:
            shape = (width, width, modes, modes)
        self.w_r = 0.1 * random.normal(k1, shape)
        self.w_i = 0.1 * random.normal(k2, shape)
        self.local = _conv(ndim, k3, width, width)
        self.modes = modes
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply spectral and local mixing. (B, C, *S) -> (B, C, *S)."""
        w = self.w_r + 1j * self.w_i
        if self.ndim == 1:
            x_hat = jnp.fft.rfft(x, axis=-1)[..., : self.modes]
            m = x_hat.shape[-1]
            y_hat = jnp.einsum("bck,ock->bok", x_hat, w[..., :m])
            y = jnp.fft.irfft(y_hat, n=x.shape[-1], axis=-1)
        else:
            x_hat = jnp.fft.rfft2(x, axes=(-2, -1))[..., : self.modes, : self.modes]
            mh, mw = x_hat.shape[-2:]
            y_hat = jnp.einsum("bchw,ochw->bohw", x_hat, w[..., :mh, :mw])
            y = jnp.fft.irfft2(y_hat, s=x.shape[-2:], axes=(-2, -1))
        return jax.nn.gelu(y + _cv(self.local, x))


# ============================================================
# Haar wavelets
# ============================================================


def haar_split_1d(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """1D Haar split into average and detail. (B, C, N) -> (B, C, N/2) each."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return (even + odd) / math.sqrt(2.0), (even - odd) / math.sqrt(2.0)


def haar_merge_1d(a: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Inverse 1D Haar merge. (B, C, N/2) each -> (B, C, N)."""
    even = (a + d) / math.sqrt(2.0)
    odd = (a - d) / math.sqrt(2.0)
    return jnp.stack([even, odd], axis=-1).reshape(*even.shape[:-1], 2 * even.shape[-1])


def _merge_axis(x: jnp.ndarray, y: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Interleave two half-resolution arrays along one spatial axis."""
    axis = axis % x.ndim
    even = (x + y) / math.sqrt(2.0)
    odd = (x - y) / math.sqrt(2.0)
    return jnp.stack([even, odd], axis=axis + 1).reshape(
        *x.shape[:axis], 2 * x.shape[axis], *x.shape[axis + 1 :]
    )


def haar_split_2d(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """2D Haar split along W then H. (B, C, H, W) -> (B, C, H/2, W/2) each."""
    even_w, odd_w = x[..., 0::2], x[..., 1::2]
    a0 = (even_w + odd_w) / math.sqrt(2.0)
    d0 = (even_w - odd_w) / math.sqrt(2.0)
    a0e, a0o = a0[..., 0::2, :], a0[..., 1::2, :]
    a = (a0e + a0o) / math.sqrt(2.0)
    dy = (a0e - a0o) / math.sqrt(2.0)
    d0e, d0o = d0[..., 0::2, :], d0[..., 1::2, :]
    dx = (d0e + d0o) / math.sqrt(2.0)
    dxy = (d0e - d0o) / math.sqrt(2.0)
    return a, dx, dy, dxy


def haar_merge_2d(a: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray, dxy: jnp.ndarray) -> jnp.ndarray:
    """Inverse 2D Haar merge. (B, C, H/2, W/2) each -> (B, C, H, W)."""
    a0 = _merge_axis(a, dy, axis=-2)
    d0 = _merge_axis(dx, dxy, axis=-2)
    return _merge_axis(a0, d0, axis=-1)


def dwt_1d(x: jnp.ndarray, levels: int) -> list[jnp.ndarray]:
    """1D Haar decomposition, coarse average first. (B, C, N) -> [a, d_L, ..., d_1]."""
    a = x
    details = []
    for _ in range(levels):
        a, d = haar_split_1d(a)
        details.append(d)
    return [a, *details[::-1]]


def idwt_1d(coeffs: list[jnp.ndarray]) -> jnp.ndarray:
    """Inverse 1D Haar transform."""
    out = coeffs[0]
    for d in coeffs[1:]:
        out = haar_merge_1d(out, d)
    return out


def dwt_2d(x: jnp.ndarray, levels: int) -> list:
    """2D Haar decomposition, coarse average first."""
    a = x
    bands = []
    for _ in range(levels):
        a, dx, dy, dxy = haar_split_2d(a)
        bands.append((dx, dy, dxy))
    return [a, *bands[::-1]]


def idwt_2d(coeffs: list) -> jnp.ndarray:
    """Inverse 2D Haar transform."""
    out = coeffs[0]
    for dx, dy, dxy in coeffs[1:]:
        out = haar_merge_2d(out, dx, dy, dxy)
    return out


# ============================================================
# Transformer blocks
# ============================================================


class PatchEmbed(eqx.Module):
    """Patch embedding for 1D or 2D fields."""

    proj: eqx.nn.Linear
    patch_size: int = eqx.field(static=True)
    ndim: int = eqx.field(static=True)

    def __init__(self, ndim: int, key: jax.Array, c_in: int, dim: int, patch_size: int) -> None:
        patch_elems = patch_size if ndim == 1 else patch_size * patch_size
        self.proj = eqx.nn.Linear(c_in * patch_elems, dim, key=key)
        self.patch_size = patch_size
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, tuple[int, ...]]:
        """Embed patches into tokens. (B, C, *S) -> (B, T, D), grid."""
        if self.ndim == 1:
            b, c, n = x.shape
            p = self.patch_size
            t = n // p
            patches = x.reshape(b, c, t, p).transpose(0, 2, 1, 3).reshape(b * t, c * p)
            tokens = jax.vmap(self.proj)(patches).reshape(b, t, -1)
            return tokens, (t,)
        b, c, h, w = x.shape
        p = self.patch_size
        th, tw = h // p, w // p
        patches = (
            x.reshape(b, c, th, p, tw, p)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(b, th * tw, c * p * p)
        )
        tokens = jax.vmap(self.proj)(patches.reshape(b * th * tw, -1)).reshape(b, th * tw, -1)
        return tokens, (th, tw)


class MultiHeadAttention(eqx.Module):
    """Scaled dot-product self-attention over token sequences."""

    w_q: eqx.nn.Linear
    w_k: eqx.nn.Linear
    w_v: eqx.nn.Linear
    w_o: eqx.nn.Linear
    heads: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, dim: int, heads: int) -> None:
        k1, k2, k3, k4 = random.split(key, 4)
        self.w_q = eqx.nn.Linear(dim, dim, key=k1)
        self.w_k = eqx.nn.Linear(dim, dim, key=k2)
        self.w_v = eqx.nn.Linear(dim, dim, key=k3)
        self.w_o = eqx.nn.Linear(dim, dim, key=k4)
        self.heads = heads

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Apply attention. (B, T, D) -> (B, T, D)."""
        b, t, d = tokens.shape
        h = self.heads
        dh = d // h
        flat = tokens.reshape(b * t, d)
        q = jax.vmap(self.w_q)(flat).reshape(b, t, h, dh)
        k = jax.vmap(self.w_k)(flat).reshape(b, t, h, dh)
        v = jax.vmap(self.w_v)(flat).reshape(b, t, h, dh)
        attn = jax.nn.softmax(jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(dh), axis=-1)
        out = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(b, t, d)
        return jax.vmap(self.w_o)(out.reshape(b * t, d)).reshape(b, t, d)


class TransformerBlock(eqx.Module):
    """Pre-norm transformer block with residual attention and FFN."""

    norm1: eqx.nn.LayerNorm
    attn: MultiHeadAttention
    norm2: eqx.nn.LayerNorm
    ffn: list[eqx.nn.Linear]

    def __init__(self, key: jax.Array, dim: int, heads: int, ffn_dim: int) -> None:
        k1, k2, k3 = random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm((dim,))
        self.attn = MultiHeadAttention(k1, dim, heads)
        self.norm2 = eqx.nn.LayerNorm((dim,))
        self.ffn = [
            eqx.nn.Linear(dim, ffn_dim, key=k2),
            eqx.nn.Linear(ffn_dim, dim, key=k3),
        ]

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Apply one transformer block. (B, T, D) -> (B, T, D)."""
        b, t, d = tokens.shape
        normed = jax.vmap(self.norm1)(tokens.reshape(b * t, d)).reshape(b, t, d)
        tokens = tokens + self.attn(normed)
        normed = jax.vmap(self.norm2)(tokens.reshape(b * t, d)).reshape(b, t, d)
        h = jax.nn.gelu(jax.vmap(self.ffn[0])(normed.reshape(b * t, d)))
        h = jax.vmap(self.ffn[1])(h).reshape(b, t, d)
        return tokens + h


def coords_grid(ndim: int, shape: tuple[int, ...]) -> jnp.ndarray:
    """Normalized grid coordinates. (N, D) for 1D or (H*W, 2) for 2D."""
    if ndim == 1:
        n = shape[0]
        return jnp.linspace(0.0, 1.0, n)[:, None]
    h, w = shape
    xs, ys = jnp.meshgrid(jnp.linspace(0.0, 1.0, w), jnp.linspace(0.0, 1.0, h))
    return jnp.stack([ys.ravel(), xs.ravel()], axis=-1)

# Baseline neural operators for 1D and 2D fields
# Author: Shengning Wang

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from models.blocks import (
    ConvBlock,
    PatchEmbed,
    SpectralConv,
    TransformerBlock,
    _conv,
    _cv,
    _pool,
    _up,
    coords_grid,
    dwt_1d,
    dwt_2d,
    idwt_1d,
    idwt_2d,
    haar_split_1d,
    haar_split_2d,
    haar_merge_1d,
    haar_merge_2d,
)


def _cfg(cfg: dict | None) -> dict:
    return cfg or {}


def _width(cfg: dict, default: int = 32) -> int:
    return int(_cfg(cfg).get("width", default))


def _depth(cfg: dict, default: int = 4) -> int:
    return int(_cfg(cfg).get("depth", default))


def _modes(cfg: dict, default: int = 16) -> int:
    return int(_cfg(cfg).get("modes", default))


# ============================================================
# CNN
# ============================================================


class CNN(eqx.Module):
    """Residual convolutional surrogate with GroupNorm and scaled output."""

    in_proj: eqx.Module
    blocks: list[ConvBlock]
    norms: list[eqx.nn.GroupNorm]
    out: eqx.Module
    ndim: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        width = _width(cfg)
        depth = _depth(cfg)
        keys = random.split(key, depth + 2)
        self.in_proj = _conv(ndim, keys[0], c_in, width, k=1)
        self.blocks = [ConvBlock(ndim, k, width, width) for k in keys[1 : 1 + depth]]
        self.norms = [eqx.nn.GroupNorm(4, width) for _ in range(depth)]
        self.out = _conv(ndim, keys[-1], width, c_in, k=1)
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to the next field. (B, C, *S) -> (B, C, *S)."""
        h = jax.nn.gelu(_cv(self.in_proj, x))
        for block, norm in zip(self.blocks, self.norms):
            h = jax.vmap(norm)(h + block(h))
        return x + 0.1 * _cv(self.out, h)


# ============================================================
# U-Net
# ============================================================


class UNet(eqx.Module):
    """Small U-Net with strided pooling for 1D or 2D fields."""

    encoders: list[ConvBlock]
    bottleneck: ConvBlock
    decoders: list[ConvBlock]
    out: eqx.Module
    ndim: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None, c_out: int | None = None) -> None:
        width = _width(cfg, 64)
        channels = (max(8, width // 4), max(16, width // 2), width)
        keys = random.split(key, 2 * len(channels) + 2)
        encoders = []
        c_prev = c_in
        for i, c in enumerate(channels):
            encoders.append(ConvBlock(ndim, keys[i], c_prev, c))
            c_prev = c
        bottleneck = ConvBlock(ndim, keys[len(channels)], c_prev, c_prev)
        decoders = []
        h_channels = channels[-1]
        for i, c in enumerate(reversed(channels)):
            decoders.append(ConvBlock(ndim, keys[len(channels) + 1 + i], h_channels + c, c))
            h_channels = c
        self.encoders = encoders
        self.bottleneck = bottleneck
        self.decoders = decoders
        self.out = _conv(ndim, keys[-1], channels[0], c_in if c_out is None else c_out, k=1)
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to the next field. (B, C, *S) -> (B, C, *S)."""
        skips = []
        h = x
        for block in self.encoders:
            h = block(h)
            skips.append(h)
            h = _pool(h)
        h = self.bottleneck(h)
        for block, skip in zip(self.decoders, skips[::-1]):
            h = _up(h)
            h = block(jnp.concatenate([h, skip], axis=1))
        return _cv(self.out, h)


# ============================================================
# ViT
# ============================================================


class ViT(eqx.Module):
    """Patch-based vision transformer for 1D or 2D fields."""

    embed: PatchEmbed
    blocks: list[TransformerBlock]
    head: eqx.nn.Linear
    patch_size: int = eqx.field(static=True)
    ndim: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        cfg = _cfg(cfg)
        dim = int(cfg.get("dim", 64))
        heads = int(cfg.get("heads", 4))
        patch_size = int(cfg.get("patch_size", 16))
        depth = _depth(cfg, 4)
        keys = random.split(key, depth + 3)
        self.embed = PatchEmbed(ndim, keys[0], c_in, dim, patch_size)
        patch_elems = patch_size if ndim == 1 else patch_size * patch_size
        self.blocks = [TransformerBlock(k, dim, heads, 4 * dim) for k in keys[1 : 1 + depth]]
        self.head = eqx.nn.Linear(dim, c_in * patch_elems, key=keys[-1])
        self.patch_size = patch_size
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to the next field. (B, C, *S) -> (B, C, *S)."""
        tokens, grid = self.embed(x)
        b, t, d = tokens.shape
        for block in self.blocks:
            tokens = block(tokens)
        patches = jax.vmap(self.head)(tokens.reshape(b * t, d)).reshape(b, t, -1)
        p = self.patch_size
        if self.ndim == 1:
            n = grid[0] * p
            out = patches.reshape(b, grid[0], -1, p).transpose(0, 2, 1, 3).reshape(b, -1, n)
        else:
            th, tw = grid
            out = (
                patches.reshape(b, th, tw, -1, p, p)
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(b, -1, th * p, tw * p)
            )
        return x + out


# ============================================================
# DeepONet
# ============================================================


class DeepONet(eqx.Module):
    """Branched deep operator network evaluated on a fixed grid."""

    branch_convs: list[ConvBlock]
    branch_head: eqx.nn.Linear
    trunk_net: list[eqx.nn.Linear]
    p: int = eqx.field(static=True)
    ndim: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        width = _width(cfg)
        p = int(_cfg(cfg).get("dim", 64))
        depth = 2
        keys = random.split(key, 2 * depth + 3)
        self.branch_convs = [ConvBlock(ndim, keys[i], c_in if i == 0 else width, width) for i in range(depth)]
        self.branch_head = eqx.nn.Linear(width, p * c_in, key=keys[depth])
        self.trunk_net = [eqx.nn.Linear(ndim, width, key=keys[depth + 1])]
        for i in range(depth - 1):
            self.trunk_net.append(eqx.nn.Linear(width, width, key=keys[depth + 2 + i]))
        self.trunk_net.append(eqx.nn.Linear(width, p, key=keys[-1]))
        self.p = p
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to the next field. (B, C, *S) -> (B, C, *S)."""
        b, c = x.shape[0], x.shape[1]
        h = x
        for block in self.branch_convs:
            h = block(h)
        branch = jax.vmap(self.branch_head)(h.mean(axis=tuple(range(2, x.ndim))))  # (B, P*C)
        branch = branch.reshape(b, self.p, c)
        coords = coords_grid(self.ndim, x.shape[2:])
        trunk = coords
        for lin in self.trunk_net:
            trunk = jax.nn.gelu(jax.vmap(lin)(trunk))
        out = jnp.einsum("bpc,np->bnc", branch, trunk)  # (B, N, C)
        if self.ndim == 1:
            return out.transpose(0, 2, 1)
        hh, ww = x.shape[2], x.shape[3]
        return out.reshape(b, hh, ww, c).transpose(0, 3, 1, 2)


# ============================================================
# FNO
# ============================================================


class FNO(eqx.Module):
    """Fourier neural operator for 1D or 2D fields."""

    in_proj: eqx.Module
    blocks: list[SpectralConv]
    out_proj: eqx.Module
    ndim: int = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None, c_out: int | None = None) -> None:
        width = _width(cfg)
        modes = _modes(cfg)
        layers = _depth(cfg, 3)
        keys = random.split(key, 3 * layers + 2)
        self.in_proj = _conv(ndim, keys[0], c_in, width, k=1)
        self.blocks = []
        for l in range(layers):
            self.blocks.append(SpectralConv(ndim, keys[1 + l], width, modes))
        self.out_proj = _conv(ndim, keys[-1], width, c_in if c_out is None else c_out, k=1)
        self.ndim = ndim
        self.residual = c_out is None or c_out == c_in

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to the next field. (B, C, *S) -> (B, C, *S)."""
        h = jax.nn.gelu(_cv(self.in_proj, x))
        for block in self.blocks:
            h = block(h)
        out = _cv(self.out_proj, h)
        return x + out if self.residual else out


# ============================================================
# WNO
# ============================================================


class WNO(eqx.Module):
    """Wavelet neural operator using Haar wavelets."""

    blocks: list[ConvBlock]
    levels: int = eqx.field(static=True)
    ndim: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        levels = int(_cfg(cfg).get("depth", 3))
        n_bands = 1 + 3 * levels if ndim == 2 else levels + 1
        keys = random.split(key, n_bands)
        self.blocks = [ConvBlock(ndim, k, c_in, c_in) for k in keys]
        self.levels = levels
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to the next field. (B, C, *S) -> (B, C, *S)."""
        if self.ndim == 1:
            coeffs = dwt_1d(x, self.levels)
            processed = [block(c) for block, c in zip(self.blocks, coeffs)]
            return x + idwt_1d(processed)
        coeffs = dwt_2d(x, self.levels)
        processed = [self.blocks[0](coeffs[0])]
        for i, (dx, dy, dxy) in enumerate(coeffs[1:]):
            j = 1 + 3 * i
            processed.append(
                (self.blocks[j](dx), self.blocks[j + 1](dy), self.blocks[j + 2](dxy))
            )
        return x + idwt_2d(processed)


# ============================================================
# UWNO
# ============================================================


class UWNO(eqx.Module):
    """U-Net with Haar wavelet down/up sampling."""

    in_proj: eqx.Module
    downs: list[ConvBlock]
    skips: list
    bottleneck: ConvBlock
    ups: list[ConvBlock]
    out: eqx.Module
    levels: int = eqx.field(static=True)
    ndim: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        width = _width(cfg, 64)
        levels = int(_cfg(cfg).get("depth", 3))
        keys = random.split(key, 4 * levels + 2)
        self.in_proj = _conv(ndim, keys[-1], c_in, width, k=1)
        self.downs = []
        self.skips = []
        for i in range(levels):
            k1, k2 = random.split(keys[i])
            self.downs.append(ConvBlock(ndim, k1, width, width))
            if ndim == 1:
                self.skips.append(_conv(ndim, k2, width, width, k=1))
            else:
                kx, ky, kxy = random.split(k2, 3)
                self.skips.append(
                    (
                        _conv(ndim, kx, width, width, k=1),
                        _conv(ndim, ky, width, width, k=1),
                        _conv(ndim, kxy, width, width, k=1),
                    )
                )
        self.bottleneck = ConvBlock(ndim, keys[2 * levels], width, width)
        self.ups = []
        for i in range(levels):
            self.ups.append(ConvBlock(ndim, keys[2 * levels + 1 + i], width, width))
        self.out = _conv(ndim, keys[-2], width, c_in, k=1)
        self.levels = levels
        self.ndim = ndim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map a field to the next field. (B, C, *S) -> (B, C, *S)."""
        h = _cv(self.in_proj, x)
        skips = []
        for i in range(self.levels):
            if self.ndim == 1:
                a, d = haar_split_1d(h)
                skips.append(_cv(self.skips[i], d))
            else:
                a, dx, dy, dxy = haar_split_2d(h)
                cx, cy, cxy = self.skips[i]
                skips.append((_cv(cx, dx), _cv(cy, dy), _cv(cxy, dxy)))
            h = self.downs[i](a)
        h = self.bottleneck(h)
        for i in reversed(range(self.levels)):
            if self.ndim == 1:
                h = haar_merge_1d(h, skips[i])
            else:
                h = haar_merge_2d(h, *skips[i])
            h = self.ups[i](h)
        return x + _cv(self.out, h)


# ============================================================
# PDE-Refiner
# ============================================================


class PDERefiner(eqx.Module):
    """FNO backbone plus a learned multi-step refiner."""

    backbone: FNO
    refiner: UNet
    refine_steps: int = eqx.field(static=True)

    def __init__(self, key: jax.Array, c_in: int, ndim: int, cfg: dict | None = None) -> None:
        k1, k2 = random.split(key)
        self.backbone = FNO(k1, c_in, ndim, cfg)
        self.refiner = UNet(k2, c_in, ndim, cfg)
        self.refine_steps = int(_cfg(cfg).get("refine_steps", 2))

    def backbone_call(self, x: jnp.ndarray) -> jnp.ndarray:
        """Backbone prediction without refinement. (B, C, *S)."""
        return self.backbone(x)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict then refine repeatedly. (B, C, *S) -> (B, C, *S)."""
        u = self.backbone(x)
        for _ in range(self.refine_steps):
            u = u + self.refiner(u)
        return u

# Model registry for HyperFlowNet baselines and method
# Author: Shengning Wang

import equinox as eqx
import jax

from models.corrector import DiffNO, FlowNO, FlowUNet
from models.hyperflownet import HyperFlowNet
from models.operators import CNN, DeepONet, FNO, PDERefiner, UNet, UWNO, ViT, WNO


def make_model(
    name: str,
    key: jax.Array,
    c_in: int = 1,
    ndim: int = 1,
    cfg: dict | None = None,
) -> eqx.Module:
    """Create one comparison model by name."""
    mcfg = cfg.get("model", cfg) if cfg else {}
    if name == "CNN":
        return CNN(key, c_in, ndim, mcfg)
    if name == "UNet":
        return UNet(key, c_in, ndim, mcfg)
    if name == "ViT":
        return ViT(key, c_in, ndim, mcfg)
    if name == "DeepONet":
        return DeepONet(key, c_in, ndim, mcfg)
    if name == "FNO":
        return FNO(key, c_in, ndim, mcfg)
    if name == "WNO":
        return WNO(key, c_in, ndim, mcfg)
    if name == "UWNO":
        return UWNO(key, c_in, ndim, mcfg)
    if name == "PDE-Refiner":
        return PDERefiner(key, c_in, ndim, mcfg)
    if name == "FlowNO":
        return FlowNO(key, c_in, ndim, mcfg)
    if name == "DiffNO":
        return DiffNO(key, c_in, ndim, mcfg)
    if name == "HyperFlowNet":
        return HyperFlowNet(key, c_in, ndim, mcfg)
    raise ValueError(f"unknown model: {name}")

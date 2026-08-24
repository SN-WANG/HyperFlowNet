# Model registry for HyperFlowNet baselines and method
# Author: Shengning Wang

from torch import nn

from models.hyperflownet import HyperFlowNet
from models.operators import DeepONet, FNO, UNet, ViT, WNO, UWNO
from models.pde_refiner import PDERefiner
from models.velocity import ConditionalFlowModel

_OPERATORS = {
    "FNO": FNO,
    "DeepONet": DeepONet,
    "U-Net": UNet,
    "UNet": UNet,
    "ViT": ViT,
    "WNO": WNO,
    "UWNO": UWNO,
}


def make_model(name: str, c_in: int, ndim: int, cfg: dict | None = None, history: int = 1) -> nn.Module:
    """Create one comparison model by name.

    Args:
        name (str): Model name in {FNO, DeepONet, U-Net, ViT, WNO, UWNO,
            PDE-Refiner, CFM, OT-CFM, HyperFlowNet}.
        c_in (int): Field channels.
        ndim (int): Spatial dimension (1 or 2).
        cfg (dict | None): Model config.
        history (int): History window length; operators receive history * c_in
            input channels, flow-matching models receive c_in and a context.

    Returns:
        nn.Module: The model.
    """
    mcfg = (cfg or {}).get("model", cfg or {})
    if name in ("CFM", "OT-CFM"):
        return ConditionalFlowModel(c_in, ndim, mcfg, history)
    if name == "HyperFlowNet":
        return HyperFlowNet(c_in, ndim, mcfg, history)
    if name == "PDE-Refiner":
        return PDERefiner(c_in * history, ndim, mcfg, c_out=c_in)
    if name not in _OPERATORS:
        raise ValueError(f"unknown model: {name}")
    return _OPERATORS[name](c_in * history, ndim, mcfg, c_out=c_in)

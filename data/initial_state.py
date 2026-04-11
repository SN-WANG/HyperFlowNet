# Deterministic first-frame construction from label and coordinates
# Author: Shengning Wang

import torch
from torch import Tensor


_LOW_PRESSURE = 1000.0
_HIGH_PRESSURE_X_MAX = 0.2129
_INTERFACE_POINTS = (
    ((0.21299347281455994, 0.0016667942982167006), 8.088444155873731e-05),
    ((0.21297390758991241, 0.003333499189466238), 8.123338920995593e-05),
)


def initial_state_from_label(label: Tensor, coords: Tensor) -> Tensor:
    """
    Build the deterministic four-channel initial state from one label and mesh.

    Args:
        label (Tensor): Pressure-ratio label. (1,) or scalar-like tensor.
        coords (Tensor): Mesh coordinates. (N, D).

    Returns:
        Tensor: Initial state in physical space. (N, 4).
    """
    label_value = float(torch.as_tensor(label, dtype=coords.dtype, device=coords.device).reshape(-1)[0].item())
    high_pressure = label_value * 1000.0

    init_state = coords.new_zeros((coords.shape[0], 4))
    init_state[:, 2] = _LOW_PRESSURE
    init_state[:, 3] = 300.0

    high_mask = coords[:, 0] < _HIGH_PRESSURE_X_MAX
    init_state[high_mask, 2] = high_pressure

    coord_xy = coords[:, :2]
    for point, alpha in _INTERFACE_POINTS:
        target = coord_xy.new_tensor(point)
        idx = (coord_xy - target).square().sum(dim=-1).argmin()
        init_state[idx, 2] = _LOW_PRESSURE + alpha * (high_pressure - _LOW_PRESSURE)

    return init_state

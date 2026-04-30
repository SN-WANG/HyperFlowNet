# Boundary condition utilities for shock-wave rollout prediction
# Author: Shengning Wang

from typing import Any, Dict, List

import torch
from torch import Tensor


class BoundaryCondition:
    """
    Data-driven wall detection and hard boundary condition enforcement.
    """

    def __init__(self) -> None:
        """
        Initialize an empty boundary condition.
        """
        self.wall_mask = torch.empty(0, dtype=torch.bool)
        self.wall_values_std = torch.empty(0)
        self.enforce_channels: List[int] = []
        self._fitted = False

    def fit(
        self,
        raw_dataset: Any,
        state_scaler: Any,
        velocity_channels: List[int],
        velocity_threshold: float = 1e-4,
    ) -> "BoundaryCondition":
        """
        Detect no-slip wall nodes and compute standardized wall values.

        Args:
            raw_dataset (Any): Dataset with raw sequences in ``seqs``. Each sequence is (T, N, C).
            state_scaler (Any): Fitted state scaler with mean and std statistics. (1, 1, C).
            velocity_channels (List[int]): Velocity channel indices to enforce.
            velocity_threshold (float): Maximum absolute velocity for wall detection.

        Returns:
            BoundaryCondition: Fitted boundary condition.
        """
        num_nodes = raw_dataset.seqs[0].shape[1]
        is_wall = torch.ones(num_nodes, dtype=torch.bool)

        for seq in raw_dataset.seqs:
            for ch in velocity_channels:
                is_wall &= seq[:, :, ch].abs().max(dim=0).values < velocity_threshold

        mean = state_scaler.mean.squeeze()
        std = state_scaler.std.squeeze()
        wall_values = torch.zeros_like(mean)
        for ch in velocity_channels:
            wall_values[ch] = (0.0 - mean[ch]) / std[ch]

        self.wall_mask = is_wall
        self.wall_values_std = wall_values
        self.enforce_channels = list(velocity_channels)
        self._fitted = True
        return self

    def enforce(self, pred: Tensor) -> Tensor:
        """
        Replace wall-node predictions with known boundary values.

        Args:
            pred (Tensor): Predicted standardized state. (B, N, C).

        Returns:
            Tensor: Boundary-enforced standardized state. (B, N, C).
        """
        if not self._fitted or self.wall_mask.sum() == 0:
            return pred

        mask = self.wall_mask.to(pred.device)
        values = self.wall_values_std.to(device=pred.device, dtype=pred.dtype)
        out = pred.clone()
        for ch in self.enforce_channels:
            out[:, mask, ch] = values[ch]
        return out

    def state_dict(self) -> Dict[str, Any]:
        """
        Return boundary condition state for checkpoint serialization.

        Returns:
            Dict[str, Any]: Boundary state dictionary.
        """
        return {
            "wall_mask": self.wall_mask,
            "wall_values_std": self.wall_values_std,
            "enforce_channels": self.enforce_channels,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load boundary condition state from a checkpoint.

        Args:
            state_dict (Dict[str, Any]): Boundary state dictionary.
        """
        self.wall_mask = state_dict["wall_mask"]
        self.wall_values_std = state_dict["wall_values_std"]
        self.enforce_channels = state_dict["enforce_channels"]
        self._fitted = state_dict["fitted"]

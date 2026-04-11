# Flow prediction metrics for autoregressive evaluation
# Author: Shengning Wang

from typing import Dict, List

import torch
from torch import Tensor


class Metrics:
    """
    Evaluation metrics for autoregressive flow prediction.
    """

    SUPPORTED_METRICS = ("nmse", "mse", "rmse", "r2", "accuracy")

    def __init__(self, channel_names: List[str], metrics: List[str] | None = None) -> None:
        """
        Initialize the metric evaluator.

        Args:
            channel_names (List[str]): Ordered field names.
            metrics (List[str] | None): Metrics to compute.
        """
        self.channel_names = channel_names
        self.metrics = list(self.SUPPORTED_METRICS) if metrics is None else metrics

    def compute(self, pred: Tensor, target: Tensor) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute global and step-wise metrics.

        Args:
            pred (Tensor): Predicted rollout. (T, N, C).
            target (Tensor): Ground-truth rollout. (T, N, C).

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: Nested metric dictionary.
        """
        results: Dict[str, Dict[str, Dict[str, float]]] = {}

        for ch_idx, ch_name in enumerate(self.channel_names):
            pred_c = pred[..., ch_idx]
            target_c = target[..., ch_idx]

            sq_diff = (pred_c - target_c).square()
            abs_diff = (pred_c - target_c).abs()
            abs_target = target_c.abs()

            channel_metrics = {"global": {}, "step_wise": {}}

            if "nmse" in self.metrics:
                channel_metrics["global"]["nmse"] = float(sq_diff.mean() / target_c.square().mean().clamp_min(1e-8))
                channel_metrics["step_wise"]["nmse"] = (
                    sq_diff.mean(dim=1) / target_c.square().mean(dim=1).clamp_min(1e-8)
                ).tolist()

            if "mse" in self.metrics:
                channel_metrics["global"]["mse"] = float(sq_diff.mean())
                channel_metrics["step_wise"]["mse"] = sq_diff.mean(dim=1).tolist()

            if "rmse" in self.metrics:
                channel_metrics["global"]["rmse"] = float(torch.sqrt(sq_diff.mean()))
                channel_metrics["step_wise"]["rmse"] = torch.sqrt(sq_diff.mean(dim=1)).tolist()

            if "r2" in self.metrics:
                ss_res = sq_diff.sum()
                ss_tot = (target_c - target_c.mean()).square().sum().clamp_min(1e-8)
                channel_metrics["global"]["r2"] = float(1.0 - ss_res / ss_tot)

                step_mean = target_c.mean(dim=1, keepdim=True)
                step_ss_res = sq_diff.sum(dim=1)
                step_ss_tot = (target_c - step_mean).square().sum(dim=1).clamp_min(1e-8)
                channel_metrics["step_wise"]["r2"] = (1.0 - step_ss_res / step_ss_tot).tolist()

            if "accuracy" in self.metrics:
                channel_metrics["global"]["accuracy"] = float(
                    (1.0 - abs_diff.sum() / abs_target.sum().clamp_min(1e-8)) * 100.0
                )
                channel_metrics["step_wise"]["accuracy"] = (
                    (1.0 - abs_diff.sum(dim=1) / abs_target.sum(dim=1).clamp_min(1e-8)) * 100.0
                ).tolist()

            results[ch_name] = channel_metrics

        return results

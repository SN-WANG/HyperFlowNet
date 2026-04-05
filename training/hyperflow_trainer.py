# HyperFlowNet rollout trainer based on BaseTrainer
# Author: Shengning Wang

from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from training.base_trainer import BaseTrainer
from utils.hue_logger import hue, logger


class RolloutCriterion(nn.Module):
    """
    Weighted full-rollout NMSE criterion.
    """

    def __init__(self, channel_weights: Optional[Sequence[float]] = None, eps: float = 1e-8):
        """
        Initialize the rollout criterion.

        Args:
            channel_weights (Optional[Sequence[float]]): Per-channel NMSE weights.
            eps (float): Small constant in the NMSE denominator.
        """
        super().__init__()
        self.eps = eps

        if channel_weights is None:
            self.channel_weights = None
        else:
            self.register_buffer("channel_weights", torch.tensor(channel_weights, dtype=torch.float32))

    def _nmse(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute per-channel NMSE and reduce it to a scalar.

        Args:
            pred (Tensor): Predicted state. (B, N, C).
            target (Tensor): Target state. (B, N, C).

        Returns:
            Tensor: Reduced NMSE loss. ().
        """
        pred = pred.float()
        target = target.float()

        C = pred.shape[-1]
        sq_err = (target - pred).square().reshape(-1, C).sum(dim=0)
        sq_ref = target.square().reshape(-1, C).sum(dim=0).clamp_min(self.eps)
        nmse = sq_err / sq_ref

        if self.channel_weights is None:
            return nmse.mean()

        weights = self.channel_weights.to(device=pred.device, dtype=pred.dtype)
        return (nmse * weights).sum() / weights.sum()

    def forward(self, pred_seq: Tensor, target_seq: Tensor) -> Tensor:
        """
        Compute weighted full-rollout loss.

        Args:
            pred_seq (Tensor): Predicted rollout. (B, T, N, C).
            target_seq (Tensor): Ground-truth rollout. (B, T, N, C).

        Returns:
            Tensor: Scalar weighted rollout loss. ().
        """
        T = pred_seq.shape[1]
        total_weight = T * (T + 1)
        loss = pred_seq.new_tensor(0.0, dtype=torch.float32)

        for t in range(T):
            weight_t = 2.0 * (t + 1) / total_weight
            loss = loss + weight_t * self._nmse(pred_seq[:, t], target_seq[:, t])

        return loss


class HyperFlowTrainer(BaseTrainer):
    """
    Minimal rollout trainer for HyperFlowNet.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-4,
        max_epochs: int = 560,
        weight_decay: float = 1e-4,
        eta_min: float = 1e-6,
        max_rollout_steps: int = 12,
        rollout_patience: int = 55,
        noise_std_init: float = 0.01,
        noise_decay: float = 0.7,
        boundary_condition: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the rollout trainer.

        Args:
            model (nn.Module): HyperFlowNet model.
            lr (float): Initial learning rate for AdamW.
            max_epochs (int): Total training epochs.
            weight_decay (float): AdamW weight decay.
            eta_min (float): Minimum learning rate of cosine annealing.
            max_rollout_steps (int): Maximum rollout horizon.
            rollout_patience (int): Epoch interval between curriculum updates.
            noise_std_init (float): Initial rollout noise std.
            noise_decay (float): Multiplicative decay of rollout noise.
            boundary_condition (Optional[Any]): Optional boundary-condition enforcer.
            **kwargs: Arguments forwarded to BaseTrainer.
        """
        optimizer = kwargs.pop("optimizer", None)
        scheduler = kwargs.pop("scheduler", None)
        criterion = kwargs.pop("criterion", None)

        if optimizer is None:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler is None:
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)

        if criterion is None:
            channel_weights = kwargs.pop("channel_weights", None)
            criterion = RolloutCriterion(channel_weights=channel_weights)

        super().__init__(
            model=model,
            lr=lr,
            max_epochs=max_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            **kwargs,
        )

        self.max_rollout_steps = max_rollout_steps
        self.rollout_patience = rollout_patience
        self.noise_std_init = noise_std_init
        self.noise_decay = noise_decay

        self.rollout_counter = 0
        self.log_update_info = False
        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init

        self.boundary_condition = boundary_condition

    def _update_curriculum(self) -> None:
        """
        Advance the rollout curriculum.
        """
        self.rollout_counter += 1

        if self.rollout_counter >= self.rollout_patience:
            if self.current_rollout_steps < self.max_rollout_steps:
                self.current_rollout_steps += 1
                self.current_noise_std *= self.noise_decay
                self.rollout_counter = 0
                self.log_update_info = True

        if self.log_update_info and self.rollout_counter == 1:
            logger.info(
                f"{hue.y}curriculum update:{hue.q} "
                f"steps = {hue.m}{self.current_rollout_steps}{hue.q}, "
                f"noise = {hue.m}{self.current_noise_std:.4f}{hue.q}"
            )
            self.log_update_info = False

    def _on_epoch_end(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Update the rollout curriculum after each epoch.
        """
        self._update_curriculum()

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Compute weighted full-rollout loss.

        Args:
            batch (Any): Batch tuple `(seq, coords, start_t_norm, dt_norm)`.

        Returns:
            Tensor: Scalar rollout loss. ().
        """
        seq, coords, start_t_norm, dt_norm = batch

        k = min(self.current_rollout_steps, seq.shape[1] - 1)
        input_state = seq[:, 0]
        target_seq = seq[:, 1:k + 1]

        pred_seq = self.model(
            inputs=input_state,
            coords=coords,
            t_norm=start_t_norm,
            dt_norm=dt_norm,
            targets=target_seq,
            teacher_forcing_ratio=0.0,
            noise_std=self.current_noise_std,
            boundary_condition=self.boundary_condition,
        )

        return self.criterion(pred_seq, target_seq)

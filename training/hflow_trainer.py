# HyperFlowNet rollout trainer for shock-wave flow simulation
# Author: Shengning Wang

from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from training.base_trainer import BaseTrainer
from utils.hue_logger import hue, logger


class NMSECriterion(nn.Module):
    """
    Per-channel normalized mean squared error loss.
    """

    def __init__(self, eps: float = 1e-8, channel_weights: Optional[list[float]] = None) -> None:
        """
        Initialize the NMSE criterion.

        Args:
            eps (float): Small denominator stabilizer.
            channel_weights (Optional[list[float]]): Optional channel weights. (C,).
        """
        super().__init__()
        self.eps = eps
        if channel_weights is None:
            self.channel_weights = None
        else:
            self.register_buffer("channel_weights", torch.tensor(channel_weights, dtype=torch.float32))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute channel-wise weighted NMSE.

        Args:
            pred (Tensor): Predicted state. (B, N, C).
            target (Tensor): Target state. (B, N, C).

        Returns:
            Tensor: Scalar NMSE loss. ().
        """
        C = pred.shape[-1]
        mse_c = (target - pred).square().reshape(-1, C).sum(dim=0)
        norm_c = target.square().reshape(-1, C).sum(dim=0).clamp_min(self.eps)
        nmse_c = mse_c / norm_c

        if self.channel_weights is None:
            return nmse_c.mean()

        channel_weights = self.channel_weights.to(device=nmse_c.device, dtype=nmse_c.dtype)
        return (nmse_c * channel_weights).sum() / channel_weights.sum()


class HyperFlowTrainer(BaseTrainer):
    """
    Rollout trainer for HyperFlowNet shock-wave flow prediction.
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
        channel_weights: Optional[list[float]] = None,
        bc: Optional[object] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the HyperFlowNet rollout trainer.

        Args:
            model (nn.Module): HyperFlowNet model.
            lr (float): AdamW learning rate.
            max_epochs (int): Number of training epochs.
            weight_decay (float): AdamW weight decay.
            eta_min (float): Minimum cosine learning rate.
            max_rollout_steps (int): Maximum rollout horizon.
            rollout_patience (int): Epoch interval for curriculum growth.
            noise_std_init (float): Initial rollout noise standard deviation.
            noise_decay (float): Multiplicative rollout noise decay.
            channel_weights (Optional[list[float]]): Optional channel weights. (C,).
            bc (Optional[object]): Boundary condition with an enforce method.
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
            criterion = NMSECriterion(channel_weights=channel_weights)

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

        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init
        self.rollout_counter = 0
        self.log_update_info = False
        self.bc = bc

    def _update_curriculum(self) -> None:
        """
        Advance rollout horizon and rollout noise.
        """
        self.rollout_counter += 1

        if self.rollout_counter >= self.rollout_patience and self.current_rollout_steps < self.max_rollout_steps:
            self.current_rollout_steps += 1
            self.current_noise_std *= self.noise_decay
            self.rollout_counter = 0
            self.log_update_info = True

        if self.log_update_info and self.rollout_counter == 1:
            logger.info(
                f"{hue.y}curriculum update:{hue.q} "
                f"steps={hue.m}{self.current_rollout_steps}{hue.q}, "
                f"noise={hue.m}{self.current_noise_std:.4f}{hue.q}"
            )
            self.log_update_info = False

    def _on_epoch_end(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Update curriculum at the end of each epoch.
        """
        self._update_curriculum()

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Compute weighted autoregressive rollout NMSE.

        Args:
            batch (Any): Batch tuple or list of (seq, coords, t0_norm, dt_norm).

        Returns:
            Tensor: Scalar training loss. ().
        """
        seq, coords, t0_norm, dt_norm = batch
        K = min(self.current_rollout_steps, seq.shape[1] - 1)
        total_weight = K * (K + 1)

        input_state = seq[:, 0]
        loss = seq.new_zeros(())

        for step_idx in range(K):
            step_input = input_state
            if self.model.training and self.current_noise_std > 0.0:
                step_input = step_input + self.current_noise_std * torch.randn_like(step_input)

            step_t_norm = t0_norm + step_idx * dt_norm
            pred_state = self.model(step_input, coords, t_norm=step_t_norm)
            if self.bc is not None:
                pred_state = self.bc.enforce(pred_state)
            target_state = seq[:, step_idx + 1]

            step_weight = 2.0 * (step_idx + 1) / total_weight
            loss = loss + step_weight * self.criterion(pred_state, target_state)
            input_state = pred_state

        return loss

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
    Channel-weighted NMSE criterion for one rollout step.
    """

    def __init__(self, channel_weights: Optional[Sequence[float]] = None, eps: float = 1e-8) -> None:
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

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute channel-weighted NMSE.

        Args:
            pred (Tensor): Predicted state. (B, N, C).
            target (Tensor): Target state. (B, N, C).

        Returns:
            Tensor: Scalar NMSE loss. ().
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


class HyperFlowTrainer(BaseTrainer):
    """
    Rollout trainer for HyperFlowNet with step-wise BPTT curriculum.
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
        teacher_forcing_init: float = 0.25,
        teacher_forcing_decay: float = 0.5,
        teacher_forcing_floor: float = 0.0,
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
            teacher_forcing_init (float): Initial teacher-forcing ratio after the one-step warmup stage.
            teacher_forcing_decay (float): Multiplicative decay of teacher forcing after each curriculum advance.
            teacher_forcing_floor (float): Minimal teacher-forcing ratio.
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
        self.teacher_forcing_init = teacher_forcing_init
        self.teacher_forcing_decay = teacher_forcing_decay
        self.teacher_forcing_floor = teacher_forcing_floor

        self.rollout_counter = 0
        self.log_update_info = False
        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init
        self.current_teacher_forcing_ratio = teacher_forcing_init

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
                self.current_teacher_forcing_ratio = max(
                    self.teacher_forcing_floor,
                    self.current_teacher_forcing_ratio * self.teacher_forcing_decay,
                )
                self.rollout_counter = 0
                self.log_update_info = True

        if self.log_update_info and self.rollout_counter == 1:
            logger.info(
                f"{hue.y}curriculum update:{hue.q} "
                f"steps = {hue.m}{self.current_rollout_steps}{hue.q}, "
                f"noise = {hue.m}{self.current_noise_std:.4f}{hue.q}, "
                f"teacher forcing = {hue.m}{self.current_teacher_forcing_ratio:.3f}{hue.q}"
            )
            self.log_update_info = False

    def _on_epoch_end(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Update the rollout curriculum after each epoch.
        """
        self._update_curriculum()

    def compute_rollout_loss(
        self,
        batch: Any,
        rollout_steps: int,
        teacher_forcing_ratio: float,
        noise_std: float,
    ) -> Tensor:
        """
        Compute weighted rollout loss with step-wise BPTT.

        Args:
            batch (Any): Batch tuple `(seq, coords, start_t_norm, dt_norm)`.
            rollout_steps (int): Rollout horizon used for this pass.
            teacher_forcing_ratio (float): Probability of feeding ground-truth states between rollout steps.
            noise_std (float): Gaussian noise std injected into the rollout input during training.

        Returns:
            Tensor: Scalar rollout loss. ().
        """
        seq, coords, start_t_norm, dt_norm = batch

        num_steps = min(int(rollout_steps), seq.shape[1] - 1)
        input_state = seq[:, 0]
        batch_size = seq.shape[0]
        base_time = start_t_norm.to(device=input_state.device, dtype=input_state.dtype)
        step_dt_norm = dt_norm.to(device=input_state.device, dtype=input_state.dtype)

        total_weight = num_steps * (num_steps + 1)
        loss = seq.new_tensor(0.0, dtype=torch.float32)

        for step_idx in range(num_steps):
            step_input = input_state
            if self.model.training and noise_std > 0.0:
                step_input = step_input + noise_std * torch.randn_like(step_input)

            step_t_norm = base_time + step_idx * step_dt_norm
            pred_state = self.model(step_input, coords, t_norm=step_t_norm)

            if self.boundary_condition is not None:
                pred_state = self.boundary_condition.enforce(pred_state)

            target_state = seq[:, step_idx + 1]
            weight_t = 2.0 * (step_idx + 1) / total_weight
            loss = loss + weight_t * self.criterion(pred_state, target_state)

            if step_idx >= num_steps - 1:
                continue

            if teacher_forcing_ratio >= 1.0:
                input_state = target_state
            elif teacher_forcing_ratio <= 0.0:
                input_state = pred_state
            else:
                use_truth = (torch.rand(batch_size, device=pred_state.device) < teacher_forcing_ratio)
                use_truth = use_truth.to(dtype=pred_state.dtype).view(-1, 1, 1)
                input_state = use_truth * target_state + (1.0 - use_truth) * pred_state

        return loss

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Compute weighted rollout loss.

        Args:
            batch (Any): Batch tuple `(seq, coords, start_t_norm, dt_norm)`.

        Returns:
            Tensor: Scalar rollout loss. ().
        """
        if self.model.training:
            rollout_steps = self.current_rollout_steps
            teacher_forcing_ratio = self.current_teacher_forcing_ratio
            noise_std = self.current_noise_std
        else:
            rollout_steps = self.current_rollout_steps
            teacher_forcing_ratio = 0.0
            noise_std = 0.0

        return self.compute_rollout_loss(batch, rollout_steps, teacher_forcing_ratio, noise_std)

# HyperFlowNet rollout trainer for flow simulation
# Author: Shengning Wang

import math
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
    Rollout trainer for HyperFlowNet flow prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-4,
        max_epochs: int = 560,
        weight_decay: float = 1e-4,
        eta_min: float = 1e-6,
        max_rollout_steps: int = 11,
        rollout_patience: int = 55,
        noise_std_init: float = 0.01,
        noise_decay: float = 0.7,
        max_history_steps: int = 4,
        history_length_alpha: float = 1.0,
        history_sigma_min: float = 0.25,
        history_sigma_max: float = 2.0,
        history_sigma_alpha: float = 1.0,
        use_weighted_loss: bool = True,
        use_causal_weighting: bool = True,
        causal_weight_eps: float = 1.0,
        loss_eps: float = 1e-8,
        channel_weights: Optional[list[float]] = None,
        bc: Optional[object] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the HyperFlowNet rollout trainer.

        Args:
            model (nn.Module): Flow model.
            lr (float): AdamW learning rate.
            max_epochs (int): Number of training epochs.
            weight_decay (float): AdamW weight decay.
            eta_min (float): Minimum cosine learning rate.
            max_rollout_steps (int): Maximum rollout horizon.
            rollout_patience (int): Epoch interval for curriculum growth.
            noise_std_init (float): Initial rollout noise standard deviation.
            noise_decay (float): Multiplicative rollout noise decay.
            max_history_steps (int): Maximum sliding-history length.
            history_length_alpha (float): History length annealing exponent.
            history_sigma_min (float): Final history kernel bandwidth.
            history_sigma_max (float): Initial history kernel bandwidth.
            history_sigma_alpha (float): History bandwidth annealing exponent.
            use_weighted_loss (bool): Whether to use channel, rollout, and causal loss weights.
            use_causal_weighting (bool): Whether to use temporal causal weights.
            causal_weight_eps (float): Causal weighting exponent coefficient.
            loss_eps (float): Small denominator stabilizer for weighted loss.
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

        self.plain_criterion = NMSECriterion()
        self.max_rollout_steps = max_rollout_steps
        self.rollout_patience = rollout_patience
        self.noise_std_init = noise_std_init
        self.noise_decay = noise_decay
        self.max_history_steps = max_history_steps
        self.history_length_alpha = history_length_alpha
        self.history_sigma_min = history_sigma_min
        self.history_sigma_max = history_sigma_max
        self.history_sigma_alpha = history_sigma_alpha
        self.use_weighted_loss = use_weighted_loss
        self.use_causal_weighting = use_causal_weighting
        self.causal_weight_eps = causal_weight_eps
        self.loss_eps = loss_eps

        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init
        self.current_history_steps = max_history_steps
        self.current_history_sigma = history_sigma_max
        self.current_history_weights = torch.ones(max_history_steps)
        self.rollout_counter = 0
        self.log_update_info = False
        self.bc = bc
        self._sync_curriculum_state()

    def _sync_curriculum_state(self) -> None:
        """
        Synchronize history length and history kernel weights with the rollout horizon.
        """
        if self.max_rollout_steps <= 1:
            progress = 1.0
        else:
            progress = (self.current_rollout_steps - 1) / (self.max_rollout_steps - 1)

        history_steps = 1 + math.floor((self.max_history_steps - 1) * (1.0 - progress) ** self.history_length_alpha)
        sigma = self.history_sigma_min + (self.history_sigma_max - self.history_sigma_min) * (
            1.0 - progress
        ) ** self.history_sigma_alpha

        lags = torch.arange(history_steps, dtype=torch.float32, device=self.device)
        rho = torch.exp(-(lags.square()) / (2.0 * sigma * sigma))
        self.current_history_steps = history_steps
        self.current_history_sigma = sigma
        self.current_history_weights = rho / rho.sum().clamp_min(self.loss_eps)

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

    def _on_epoch_start(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Update history schedule at the start of each epoch.
        """
        self._sync_curriculum_state()
        if self.log_update_info:
            logger.info(
                f"{hue.y}curriculum update:{hue.q} "
                f"steps={hue.m}{self.current_rollout_steps}{hue.q}, "
                f"history={hue.m}{self.current_history_steps}{hue.q}, "
                f"sigma_h={hue.m}{self.current_history_sigma:.4f}{hue.q}, "
                f"noise={hue.m}{self.current_noise_std:.4f}{hue.q}"
            )
            self.log_update_info = False

    def _on_epoch_end(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Update rollout curriculum at the end of each epoch.
        """
        self._update_curriculum()

    def _model_step(self, history: Tensor, coords: Tensor, t_norm: Tensor) -> Tensor:
        history_weights = self.current_history_weights.to(device=history.device, dtype=history.dtype)
        if getattr(self.model, "uses_history", False):
            return self.model(history, coords, t_norm=t_norm, history_weights=history_weights)
        return self.model(history[:, 0], coords, t_norm=t_norm[:, 0])

    def _reduce_rollout_losses(self, step_losses: Tensor) -> Tensor:
        K = step_losses.shape[0]
        if not self.use_weighted_loss:
            return step_losses.mean()

        step_idx = torch.arange(K, device=step_losses.device, dtype=step_losses.dtype)
        rollout_weights = 2.0 * (step_idx + 1.0) / (K * (K + 1.0))

        if self.use_causal_weighting and K > 1:
            prefix = torch.cat([step_losses.new_zeros(1), torch.cumsum(step_losses.detach(), dim=0)[:-1]], dim=0)
            causal_weights = torch.exp(-self.causal_weight_eps * prefix)
            rollout_weights = rollout_weights * causal_weights
            return (rollout_weights * step_losses).sum() / rollout_weights.sum().clamp_min(self.loss_eps)

        return (rollout_weights * step_losses).sum()

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Compute autoregressive rollout NMSE with annealed sliding history.

        Args:
            batch (Any): Batch tuple or list of (seq, coords, t0_norm, dt_norm).

        Returns:
            Tensor: Scalar training loss. ().
        """
        seq, coords, t0_norm, dt_norm = batch
        K = min(self.current_rollout_steps, seq.shape[1] - 1)
        L = self.current_history_steps

        criterion = self.criterion if self.use_weighted_loss else self.plain_criterion
        history = [seq[:, 0] for _ in range(L)]
        losses = []
        lag_idx = torch.arange(L, device=seq.device, dtype=seq.dtype)

        for step_idx in range(K):
            model_history = torch.stack(history[:L], dim=1)
            if self.model.training and self.current_noise_std > 0.0:
                model_history = model_history.clone()
                model_history[:, 0] = model_history[:, 0] + self.current_noise_std * torch.randn_like(model_history[:, 0])

            step_t_norm = t0_norm + step_idx * dt_norm
            history_t_norm = step_t_norm[:, None] - lag_idx[None, :] * dt_norm[:, None]
            pred_state = self._model_step(model_history, coords, history_t_norm)
            if self.bc is not None:
                pred_state = self.bc.enforce(pred_state)

            losses.append(criterion(pred_state, seq[:, step_idx + 1]))
            history = [pred_state] + history[:-1]

        return self._reduce_rollout_losses(torch.stack(losses, dim=0))

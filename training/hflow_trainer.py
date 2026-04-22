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
        sq_err = (target - pred).square()
        mse_c = sq_err.reshape(-1, C).sum(dim=0)
        norm_c = target.square().reshape(-1, C).sum(dim=0).clamp_min(self.eps)
        nmse_c = mse_c / norm_c

        if self.channel_weights is None:
            return nmse_c.mean()

        channel_weights = self.channel_weights.to(device=nmse_c.device, dtype=nmse_c.dtype)
        return (nmse_c * channel_weights).sum() / channel_weights.sum()


class HyperFlowTrainer(BaseTrainer):
    """
    Rollout trainer for HyperFlowNet, a spatio-temporal neural operator for shock-wave flow simulation.
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
        frontier_blocks: int = 3,
        frontier_q_low: float = 0.25,
        frontier_q_high: float = 0.75,
        frontier_margin: float = 0.10,
        lambda_smooth: float = 0.05,
        lambda_frontier: float = 0.05,
        **kwargs,
    ) -> None:
        """
        Initialize the HyperFlowNet rollout trainer.

        Args:
            model (nn.Module): HyperFlowNet model for shock-wave flow simulation.
            lr (float): AdamW learning rate.
            max_epochs (int): Number of training epochs.
            weight_decay (float): AdamW weight decay.
            eta_min (float): Minimum cosine learning rate.
            max_rollout_steps (int): Maximum rollout horizon.
            rollout_patience (int): Epoch interval for curriculum growth.
            noise_std_init (float): Initial rollout noise standard deviation.
            noise_decay (float): Multiplicative rollout noise decay.
            channel_weights (Optional[list[float]]): Optional channel weights. (C,).
            frontier_blocks (int): Number of early blocks used by frontier losses.
            frontier_q_low (float): Smooth-edge quantile.
            frontier_q_high (float): Frontier-edge quantile.
            frontier_margin (float): Allowed assignment overlap on frontier edges.
            lambda_smooth (float): Weight of L_smooth.
            lambda_frontier (float): Weight of L_frontier.
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

        self.frontier_blocks = frontier_blocks
        self.frontier_q_low = frontier_q_low
        self.frontier_q_high = frontier_q_high
        self.frontier_margin = frontier_margin
        self.lambda_smooth = lambda_smooth
        self.lambda_frontier = lambda_frontier

        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init
        self.rollout_counter = 0
        self.log_update_info = False

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

    def _compute_cluster_loss(self, weight_bank: list[Tensor], state: Tensor) -> Tensor:
        """
        Compute L_smooth and L_frontier from early slice assignments.

        Args:
            weight_bank (list[Tensor]): Slice assignments from all blocks. Each (B, N, S).
            state (Tensor): Standardized target state. (B, N, C).

        Returns:
            Tensor: Scalar cluster regularization loss. ().
        """
        if self.lambda_smooth <= 0.0 and self.lambda_frontier <= 0.0:
            return state.new_zeros(())

        edge_index = self.model.edge_index
        src, dst = edge_index[0], edge_index[1]
        contrast = (state[:, src, :] - state[:, dst, :]).abs().sum(dim=-1)
        low = torch.quantile(contrast, self.frontier_q_low, dim=1, keepdim=True)
        high = torch.quantile(contrast, self.frontier_q_high, dim=1, keepdim=True)
        smooth_mask = contrast <= low
        frontier_mask = contrast >= high

        num_blocks = min(self.frontier_blocks, len(weight_bank))
        if num_blocks == 0:
            return state.new_zeros(())

        block_losses = []
        for block_idx in range(num_blocks):
            weights = weight_bank[block_idx]
            p_src = weights[:, src, :]
            p_dst = weights[:, dst, :]

            smooth_dist = (p_src - p_dst).square().sum(dim=-1)
            overlap = (p_src * p_dst).sum(dim=-1)
            frontier_penalty = torch.relu(overlap - self.frontier_margin)

            smooth_loss = (smooth_dist * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
            frontier_loss = (frontier_penalty * frontier_mask).sum() / frontier_mask.sum().clamp_min(1.0)
            block_losses.append(self.lambda_smooth * smooth_loss + self.lambda_frontier * frontier_loss)

        return torch.stack(block_losses).mean()

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Compute weighted rollout loss with frontier regularization.

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
            pred_state, weight_bank = self.model(step_input, coords, step_t_norm)
            target_state = seq[:, step_idx + 1]

            rollout_loss = self.criterion(pred_state, target_state)
            cluster_loss = self._compute_cluster_loss(weight_bank, target_state)
            step_loss = rollout_loss + cluster_loss

            weight = 2.0 * (step_idx + 1) / total_weight
            loss = loss + weight * step_loss
            input_state = pred_state

        return loss

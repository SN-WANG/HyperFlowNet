# HyperFlowNet trainer with staged curriculum rollout
# Author: Shengning Wang

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.base_trainer import BaseTrainer
from utils.hue_logger import hue, logger


class HyperFlowCriterion(nn.Module):
    """
    Weighted rollout criterion for HyperFlowNet training.
    """

    def __init__(
        self,
        channel_weights: Optional[Sequence[float]] = None,
        delta_loss_weight: float = 0.0,
        step_weight_power: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize the rollout criterion.

        Args:
            channel_weights (Optional[Sequence[float]]): Per-channel loss weights.
            delta_loss_weight (float): Auxiliary delta loss weight.
            step_weight_power (float): Power used for rollout-step weighting.
            eps (float): Small constant used in NMSE normalization.
        """
        super().__init__()
        self.delta_loss_weight = delta_loss_weight
        self.step_weight_power = step_weight_power
        self.eps = eps

        if channel_weights is None:
            self.channel_weights = None
        else:
            self.register_buffer("channel_weights", torch.tensor(channel_weights, dtype=torch.float32))

    def _nmse(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute channel-wise normalized MSE and reduce it to a scalar.

        Args:
            pred (Tensor): Predicted state. (B, N, C).
            target (Tensor): Target state. (B, N, C).

        Returns:
            Tensor: Reduced NMSE loss. ().
        """
        pred = pred.float()
        target = target.float()
        C = pred.shape[-1]
        sq_err = (pred - target).square().reshape(-1, C).sum(dim=0)
        sq_ref = target.square().reshape(-1, C).sum(dim=0).clamp_min(self.eps)
        nmse = sq_err / sq_ref

        if self.channel_weights is None:
            return nmse.mean()

        weights = self.channel_weights.to(device=pred.device, dtype=pred.dtype)
        return (nmse * weights).sum() / weights.sum()

    def _step_weights(self, steps: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Build normalized rollout-step weights.

        Args:
            steps (int): Number of rollout steps.
            device (torch.device): Target device.
            dtype (torch.dtype): Target dtype.

        Returns:
            Tensor: Rollout-step weights. (T,).
        """
        weights = torch.arange(1, steps + 1, device=device, dtype=dtype)
        weights = weights.pow(self.step_weight_power)
        return weights / weights.sum()

    def forward(self, pred_seq: Tensor, target_seq: Tensor, input_state: Tensor) -> Tensor:
        """
        Compute weighted rollout loss for the full predicted sequence.

        Args:
            pred_seq (Tensor): Predicted rollout. (B, T, N, C).
            target_seq (Tensor): Ground-truth rollout. (B, T, N, C).
            input_state (Tensor): Rollout initial state. (B, N, C).

        Returns:
            Tensor: Scalar rollout loss. ().
        """
        steps = pred_seq.shape[1]
        weights = self._step_weights(steps, pred_seq.device, pred_seq.dtype)

        state_terms = [
            self._nmse(pred_seq[:, step_idx], target_seq[:, step_idx])
            for step_idx in range(steps)
        ]
        state_loss = torch.stack(state_terms).mul(weights).sum()

        if self.delta_loss_weight <= 0.0:
            return state_loss

        prev_state = torch.cat([input_state.unsqueeze(1), target_seq[:, :-1]], dim=1)
        pred_delta = pred_seq - prev_state
        target_delta = target_seq - prev_state

        delta_terms = [
            self._nmse(pred_delta[:, step_idx], target_delta[:, step_idx])
            for step_idx in range(steps)
        ]
        delta_loss = torch.stack(delta_terms).mul(weights).sum()
        return state_loss + self.delta_loss_weight * delta_loss


class HyperFlowTrainer(BaseTrainer):
    """
    HyperFlowNet trainer with staged rollout curriculum and scheduled sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-4,
        max_epochs: int = 480,
        patience: Optional[int] = None,
        weight_decay: float = 1e-5,
        rollout_steps: Sequence[int] = (1, 2, 4, 8, 12),
        stage_ratios: Sequence[float] = (0.15, 0.20, 0.25, 0.20, 0.20),
        teacher_forcing_lows: Sequence[float] = (1.0, 0.9, 0.75, 0.4, 0.0),
        stage_lrs: Sequence[float] = (2e-4, 2e-4, 1.5e-4, 1e-4, 7e-5),
        stage_warmup_ratio: float = 0.2,
        stage_min_lr_ratio: float = 0.05,
        input_noise_std: float = 0.01,
        input_noise_decay: float = 0.85,
        eval_rollout_steps: int = 12,
        channel_weights: Optional[Sequence[float]] = None,
        delta_loss_weight: float = 0.0,
        step_weight_power: float = 1.0,
        loss_eps: float = 1e-6,
        boundary_condition: Optional[Any] = None,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the HyperFlowNet trainer.

        Args:
            model (nn.Module): HyperFlowNet model.
            lr (float): Initial learning rate for AdamW.
            max_epochs (int): Maximum number of epochs.
            patience (Optional[int]): Early stopping patience.
            weight_decay (float): AdamW weight decay.
            rollout_steps (Sequence[int]): Rollout length of each curriculum stage.
            stage_ratios (Sequence[float]): Epoch ratio of each curriculum stage.
            teacher_forcing_lows (Sequence[float]): Final teacher-forcing ratio of each stage.
            stage_lrs (Sequence[float]): Base learning rate of each stage.
            stage_warmup_ratio (float): Warmup ratio inside each stage.
            stage_min_lr_ratio (float): Minimal LR ratio inside stage cosine schedule.
            input_noise_std (float): Initial rollout input noise std.
            input_noise_decay (float): Stage-wise decay factor of rollout input noise.
            eval_rollout_steps (int): Validation rollout length.
            channel_weights (Optional[Sequence[float]]): Per-channel loss weights.
            delta_loss_weight (float): Auxiliary delta loss weight.
            step_weight_power (float): Power used for rollout-step weighting.
            loss_eps (float): Small constant used in NMSE normalization.
            boundary_condition (Optional[Any]): Optional hard boundary-condition enforcer.
            optimizer (Optional[Optimizer]): External optimizer override.
            criterion (Optional[nn.Module]): External criterion override.
            **kwargs: Arguments forwarded to BaseTrainer.
        """
        if optimizer is None:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if criterion is None:
            criterion = HyperFlowCriterion(
                channel_weights=channel_weights,
                delta_loss_weight=delta_loss_weight,
                step_weight_power=step_weight_power,
                eps=loss_eps,
            )

        super().__init__(
            model=model,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            optimizer=optimizer,
            scheduler=None,
            criterion=criterion,
            **kwargs,
        )

        self.rollout_steps = list(rollout_steps)
        self.stage_ratios = list(stage_ratios)
        self.teacher_forcing_lows = list(teacher_forcing_lows)
        self.stage_lrs = list(stage_lrs)

        self.stage_warmup_ratio = stage_warmup_ratio
        self.stage_min_lr_ratio = stage_min_lr_ratio
        self.input_noise_std = input_noise_std
        self.input_noise_decay = input_noise_decay
        self.eval_rollout_steps = eval_rollout_steps
        self.boundary_condition = boundary_condition

        self.stages = self._build_stages()
        self._active_stage_idx = -1

    def _build_stages(self) -> List[Dict[str, Union[int, float]]]:
        """
        Build staged rollout curriculum metadata.

        Returns:
            List[Dict[str, Union[int, float]]]: Curriculum stage definitions.
        """
        lengths = [int(round(ratio * self.max_epochs)) for ratio in self.stage_ratios]
        lengths[-1] += self.max_epochs - sum(lengths)

        stages: List[Dict[str, Union[int, float]]] = []
        start = 0
        for length, rollout, tf_low, base_lr in zip(
            lengths, self.rollout_steps, self.teacher_forcing_lows, self.stage_lrs
        ):
            end = start + length
            stages.append({
                "start": start,
                "end": end,
                "rollout": rollout,
                "tf_hi": 1.0,
                "tf_lo": tf_low,
                "base_lr": base_lr,
            })
            start = end
        return stages

    def _get_stage_index(self, epoch_idx: int) -> int:
        """
        Locate the curriculum stage for the current epoch.

        Args:
            epoch_idx (int): Zero-based epoch index.

        Returns:
            int: Stage index.
        """
        for idx, stage in enumerate(self.stages):
            if stage["start"] <= epoch_idx < stage["end"]:
                return idx
        return len(self.stages) - 1

    def _get_stage(self, epoch_idx: int) -> Dict[str, Union[int, float]]:
        """
        Retrieve the curriculum stage for a given epoch.

        Args:
            epoch_idx (int): Zero-based epoch index.

        Returns:
            Dict[str, Union[int, float]]: Stage definition.
        """
        return self.stages[self._get_stage_index(epoch_idx)]

    def _get_teacher_forcing_ratio(self, epoch_idx: int) -> float:
        """
        Compute the current teacher-forcing ratio.

        Args:
            epoch_idx (int): Zero-based epoch index.

        Returns:
            float: Teacher-forcing ratio.
        """
        stage = self._get_stage(epoch_idx)
        local_epoch = epoch_idx - stage["start"]
        stage_len = stage["end"] - stage["start"]

        if stage_len <= 1:
            return float(stage["tf_lo"])

        alpha = local_epoch / float(stage_len - 1)
        return float(stage["tf_hi"]) + alpha * (float(stage["tf_lo"]) - float(stage["tf_hi"]))

    def _get_noise_std(self, epoch_idx: int) -> float:
        """
        Compute the current rollout input noise std.

        Args:
            epoch_idx (int): Zero-based epoch index.

        Returns:
            float: Rollout input noise std.
        """
        stage_idx = self._get_stage_index(epoch_idx)
        return self.input_noise_std * (self.input_noise_decay ** stage_idx)

    def _get_stage_lr(self, epoch_idx: int) -> float:
        """
        Compute the current stage-wise learning rate.

        Args:
            epoch_idx (int): Zero-based epoch index.

        Returns:
            float: Learning rate for the current epoch.
        """
        stage = self._get_stage(epoch_idx)
        stage_len = stage["end"] - stage["start"]
        local_epoch = epoch_idx - stage["start"]
        base_lr = float(stage["base_lr"])
        min_lr = base_lr * self.stage_min_lr_ratio

        if stage_len <= 1:
            return base_lr

        warmup_epochs = int(math.ceil(stage_len * self.stage_warmup_ratio))
        warmup_epochs = min(warmup_epochs, stage_len - 1)

        if warmup_epochs > 0 and local_epoch < warmup_epochs:
            if warmup_epochs == 1:
                return base_lr
            alpha = local_epoch / float(warmup_epochs - 1)
            return min_lr + alpha * (base_lr - min_lr)

        decay_len = stage_len - warmup_epochs
        decay_pos = local_epoch - warmup_epochs

        if decay_len <= 1:
            return min_lr

        progress = decay_pos / float(decay_len - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (base_lr - min_lr)

    def _train_rollout_steps(self) -> int:
        """
        Get the training rollout length of the current epoch.

        Returns:
            int: Training rollout length.
        """
        epoch_idx = max(self.current_epoch - 1, 0)
        return int(self._get_stage(epoch_idx)["rollout"])

    def _move_batch(self, batch: Any) -> Any:
        """
        Move tensor batches onto the trainer device.

        Args:
            batch (Any): Input batch from DataLoader.

        Returns:
            Any: Device-mapped batch.
        """
        if isinstance(batch, (list, tuple)):
            return [item.to(self.device) if isinstance(item, Tensor) else item for item in batch]
        if isinstance(batch, dict):
            return {key: value.to(self.device) if isinstance(value, Tensor) else value for key, value in batch.items()}
        return batch

    def _apply_learning_rate(self) -> None:
        """
        Apply stage-wise learning rate at the beginning of an epoch.
        """
        epoch_idx = max(self.current_epoch - 1, 0)
        lr = self._get_stage_lr(epoch_idx)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _on_epoch_start(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Update stage-wise learning rate and log the current curriculum stage.
        """
        epoch_idx = max(self.current_epoch - 1, 0)
        stage_idx = self._get_stage_index(epoch_idx)
        self._apply_learning_rate()

        if stage_idx == self._active_stage_idx:
            return

        self._active_stage_idx = stage_idx
        stage = self.stages[stage_idx]
        logger.info(
            f"{hue.y}stage {stage_idx + 1}/{len(self.stages)}{hue.q} | "
            f"rollout: {hue.m}{int(stage['rollout'])}{hue.q} | "
            f"tf: {hue.m}1.00 -> {float(stage['tf_lo']):.2f}{hue.q} | "
            f"lr: {hue.m}{float(stage['base_lr']):.2e}{hue.q} | "
            f"noise: {hue.m}{self._get_noise_std(epoch_idx):.4f}{hue.q}"
        )

    def compute_rollout_loss(
        self,
        batch: Any,
        rollout_steps: int,
        teacher_forcing_ratio: float,
        noise_std: float,
    ) -> Tensor:
        """
        Compute rollout loss under the provided rollout settings.

        Args:
            batch (Any): Batch tuple `(seq, coords, start_t_norm, dt_norm)`.
            rollout_steps (int): Rollout length used for this forward pass.
            teacher_forcing_ratio (float): Scheduled-sampling ratio.
            noise_std (float): Rollout input noise std.

        Returns:
            Tensor: Scalar rollout loss. ().
        """
        seq, coords, start_t_norm, dt_norm = batch

        rollout_steps = min(int(rollout_steps), seq.shape[1] - 1)
        input_state = seq[:, 0]
        target_seq = seq[:, 1:rollout_steps + 1]

        pred_seq = self.model(
            inputs=input_state,
            coords=coords,
            t_norm=start_t_norm,
            dt_norm=dt_norm,
            targets=target_seq,
            teacher_forcing_ratio=teacher_forcing_ratio,
            noise_std=noise_std,
            boundary_condition=self.boundary_condition,
        )
        return self.criterion(pred_seq, target_seq, input_state)

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Compute the loss for a single batch.

        Args:
            batch (Any): Batch tuple `(seq, coords, start_t_norm, dt_norm)`.

        Returns:
            Tensor: Scalar rollout loss. ().
        """
        if self.model.training:
            epoch_idx = max(self.current_epoch - 1, 0)
            rollout_steps = self._train_rollout_steps()
            teacher_forcing_ratio = self._get_teacher_forcing_ratio(epoch_idx)
            noise_std = self._get_noise_std(epoch_idx)
        else:
            rollout_steps = self.eval_rollout_steps
            teacher_forcing_ratio = 0.0
            noise_std = 0.0

        return self.compute_rollout_loss(batch, rollout_steps, teacher_forcing_ratio, noise_std)

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> float:
        """
        Run one training or validation epoch.

        Args:
            loader (DataLoader): Input data loader.
            is_training (bool): Whether the epoch is training or validation.

        Returns:
            float: Mean epoch loss.
        """
        self.model.train(is_training)
        losses: List[float] = []

        epoch_idx = max(self.current_epoch - 1, 0)
        rollout_steps = self._train_rollout_steps() if is_training else self.eval_rollout_steps
        teacher_forcing_ratio = self._get_teacher_forcing_ratio(epoch_idx) if is_training else 0.0

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            pbar = tqdm(loader, desc="Training" if is_training else "Validating", leave=False, dynamic_ncols=True)
            for batch in pbar:
                batch = self._move_batch(batch)

                if is_training:
                    self.optimizer.zero_grad(set_to_none=True)

                loss = self._compute_loss(batch)

                if is_training:
                    loss.backward()
                    self.optimizer.step()

                loss_val = float(loss.detach().item())
                losses.append(loss_val)
                pbar.set_postfix({
                    "loss": f"{loss_val:.4e}",
                    "k": rollout_steps,
                    "tf": f"{teacher_forcing_ratio:.2f}",
                })

        return sum(losses) / max(len(losses), 1)

    def _save_checkpoint(self, val_loss: float, is_best: bool = False, extra_state: Optional[Dict] = None) -> None:
        """
        Save trainer state together with rollout curriculum metadata.

        Args:
            val_loss (float): Validation loss.
            is_best (bool): Whether this checkpoint is the current best one.
            extra_state (Optional[Dict]): Optional additional state.
        """
        state = {
            "curriculum_stages": self.stages,
            "eval_rollout_steps": self.eval_rollout_steps,
        }
        if extra_state is not None:
            state.update(extra_state)
        super()._save_checkpoint(val_loss, is_best=is_best, extra_state=state)

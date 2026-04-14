# HyperFlowNet rollout trainer based on BaseTrainer
# Author: Shengning Wang

import json
import time
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.base_trainer import BaseTrainer
from utils.hue_logger import hue, logger


class HyperFlowTrainer(BaseTrainer):
    """
    Joint rollout trainer for HyperFlowNet with adaptive channel weighting.
    """

    def __init__(
        self,
        model: nn.Module,
        channel_names: list[str],
        lr: float = 5e-4,
        max_epochs: int = 420,
        weight_decay: float = 1e-4,
        eta_min: float = 1e-6,
        max_rollout_steps: int = 12,
        rollout_patience: int = 35,
        noise_std_init: float = 0.01,
        noise_decay: float = 0.80,
        loss_weight_beta: float = 0.90,
        loss_weight_alpha: float = 1.25,
        loss_weight_min: float = 0.5,
        loss_weight_max: float = 4.0,
        loss_weight_warmup: int = 10,
        rollout_eval_interval: int = 5,
        checkpoint_metric: str = "hard_rollout_nmse",
        early_stop_patience: int = 70,
        **kwargs,
    ) -> None:
        """
        Initialize the rollout trainer.

        Args:
            model (nn.Module): HyperFlowNet model.
            channel_names (list[str]): Ordered channel names.
            lr (float): Initial learning rate for AdamW.
            max_epochs (int): Total training epochs.
            weight_decay (float): AdamW weight decay.
            eta_min (float): Minimum learning rate of cosine annealing.
            max_rollout_steps (int): Maximum rollout horizon.
            rollout_patience (int): Epoch interval between curriculum updates.
            noise_std_init (float): Initial rollout noise std.
            noise_decay (float): Multiplicative decay of rollout noise.
            loss_weight_beta (float): EMA momentum for adaptive channel weighting.
            loss_weight_alpha (float): Hardness exponent for adaptive channel weighting.
            loss_weight_min (float): Minimum unclipped adaptive channel weight.
            loss_weight_max (float): Maximum unclipped adaptive channel weight.
            loss_weight_warmup (int): Warmup epochs before adaptive weighting starts.
            rollout_eval_interval (int): Epoch interval for full-horizon rollout evaluation.
            checkpoint_metric (str): Validation metric used for checkpoint selection.
            early_stop_patience (int): Number of rollout evaluations without improvement before stopping.
            **kwargs: Arguments forwarded to BaseTrainer.
        """
        optimizer = kwargs.pop("optimizer", None)
        scheduler = kwargs.pop("scheduler", None)

        if optimizer is None:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler is None:
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)

        super().__init__(
            model=model,
            lr=lr,
            max_epochs=max_epochs,
            patience=early_stop_patience,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

        self.channel_names = channel_names
        self.num_channels = len(channel_names)

        self.max_rollout_steps = max_rollout_steps
        self.rollout_patience = rollout_patience
        self.noise_std_init = noise_std_init
        self.noise_decay = noise_decay

        self.loss_weight_beta = loss_weight_beta
        self.loss_weight_alpha = loss_weight_alpha
        self.loss_weight_min = loss_weight_min
        self.loss_weight_max = loss_weight_max
        self.loss_weight_warmup = loss_weight_warmup
        self.rollout_eval_interval = rollout_eval_interval
        self.checkpoint_metric = checkpoint_metric

        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init
        self.curriculum_counter = 0

        self.channel_ema = torch.ones(self.num_channels, dtype=torch.float32)
        self.channel_weights = torch.ones(self.num_channels, dtype=torch.float32)
        self.best_metric = float("inf")
        self._epoch_stats = self._new_epoch_stats()

    def _new_epoch_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Create one fresh epoch-statistics container.

        Returns:
            Dict[str, Dict[str, Any]]: Train and validation channel-MSE accumulators.
        """
        return {
            "train": {"sum": torch.zeros(self.num_channels, dtype=torch.float32), "count": 0},
            "val": {"sum": torch.zeros(self.num_channels, dtype=torch.float32), "count": 0},
        }

    def _on_epoch_start(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Reset epoch statistics before each training epoch.
        """
        self._epoch_stats = self._new_epoch_stats()

    def _accumulate_channel_mse(self, phase: str, per_channel_mse: Tensor) -> None:
        """
        Accumulate per-step channel MSE statistics.

        Args:
            phase (str): Either ``"train"`` or ``"val"``.
            per_channel_mse (Tensor): Channel-wise MSE for one rollout step. (C,).
        """
        self._epoch_stats[phase]["sum"] += per_channel_mse.detach().cpu().float()
        self._epoch_stats[phase]["count"] += 1

    def _mean_channel_mse(self, phase: str) -> Tensor:
        """
        Compute averaged per-channel MSE for one phase.

        Args:
            phase (str): Either ``"train"`` or ``"val"``.

        Returns:
            Tensor: Averaged per-channel MSE. (C,).
        """
        stats = self._epoch_stats[phase]
        if stats["count"] == 0:
            return torch.zeros(self.num_channels, dtype=torch.float32)
        return stats["sum"] / stats["count"]

    def _update_curriculum(self) -> None:
        """
        Advance the rollout curriculum.
        """
        self.curriculum_counter += 1
        if self.curriculum_counter < self.rollout_patience:
            return

        self.curriculum_counter = 0
        if self.current_rollout_steps >= self.max_rollout_steps:
            return

        self.current_rollout_steps += 1
        self.current_noise_std *= self.noise_decay
        logger.info(
            f"{hue.y}curriculum update:{hue.q} "
            f"steps = {hue.m}{self.current_rollout_steps}{hue.q}, "
            f"noise = {hue.m}{self.current_noise_std:.4f}{hue.q}"
        )

    def _update_channel_weights(self, train_channel_mse: Tensor) -> None:
        """
        Update EMA hardness statistics and adaptive channel weights.

        Args:
            train_channel_mse (Tensor): Epoch-averaged per-channel MSE. (C,).
        """
        train_channel_mse = train_channel_mse.detach().cpu().float()

        if self.current_epoch <= self.loss_weight_warmup:
            self.channel_ema = train_channel_mse.clone()
            self.channel_weights = torch.ones_like(train_channel_mse)
            return

        self.channel_ema = self.loss_weight_beta * self.channel_ema + (1.0 - self.loss_weight_beta) * train_channel_mse
        hardness = (self.channel_ema / self.channel_ema.mean()).pow(self.loss_weight_alpha)
        weights = hardness.clamp(min=self.loss_weight_min, max=self.loss_weight_max)
        self.channel_weights = weights / weights.mean()

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Compute the joint weighted rollout loss.

        Args:
            batch (Any): Batch tuple ``(seq, coords, t0_norm, dt_norm)``.

        Returns:
            Tensor: Scalar rollout loss. ().
        """
        seq, coords, t0_norm, dt_norm = batch

        num_steps = min(int(self.current_rollout_steps), seq.shape[1] - 1)
        input_state = seq[:, 0]
        t0_norm = t0_norm.to(device=input_state.device, dtype=input_state.dtype)
        dt_norm = dt_norm.to(device=input_state.device, dtype=input_state.dtype)
        channel_weights = self.channel_weights.to(device=input_state.device, dtype=input_state.dtype)

        total_weight = num_steps * (num_steps + 1)
        noise_std = self.current_noise_std if self.model.training else 0.0
        phase = "train" if self.model.training else "val"
        loss = input_state.new_tensor(0.0, dtype=torch.float32)

        for step_idx in range(num_steps):
            step_input = input_state
            if self.model.training and noise_std > 0.0:
                step_input = step_input + noise_std * torch.randn_like(step_input)

            step_t_norm = t0_norm + step_idx * dt_norm
            pred_state = self.model(step_input, coords, t_norm=step_t_norm)
            target_state = seq[:, step_idx + 1]

            per_channel_mse = (pred_state - target_state).square().mean(dim=(0, 1))
            self._accumulate_channel_mse(phase, per_channel_mse)

            weight_t = 2.0 * (step_idx + 1) / total_weight
            loss = loss + weight_t * torch.sum(channel_weights * per_channel_mse)

            if step_idx < num_steps - 1:
                input_state = pred_state

        return loss

    def _evaluate_rollout(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run full-horizon rollout evaluation on the validation loader.

        Args:
            loader (DataLoader): Validation loader with standardized sequences.

        Returns:
            Dict[str, float]: Aggregated rollout metrics.
        """
        self.model.eval()

        state_scaler = self.scalers["state_scaler"]
        error_sum = torch.zeros(self.num_channels, dtype=torch.float64)
        target_sum = torch.zeros(self.num_channels, dtype=torch.float64)
        count = 0

        with torch.no_grad():
            pbar = tqdm(loader, desc="Rollout Eval", leave=False, dynamic_ncols=True)
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, Tensor) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

                seq, coords, t0_norm, dt_norm = batch
                pred_std = self.model.predict(
                    inputs=seq[:, 0],
                    coords=coords,
                    steps=seq.shape[1] - 1,
                    t0_norm=t0_norm,
                    dt_norm=dt_norm,
                    show_progress=False,
                )
                target_std = seq.detach().cpu()

                pred = state_scaler.inverse_transform(pred_std)
                target = state_scaler.inverse_transform(target_std)
                diff_sq = (pred - target).square()

                error_sum += diff_sq.sum(dim=(0, 1, 2)).double()
                target_sum += target.square().sum(dim=(0, 1, 2)).double()
                count += pred.shape[0] * pred.shape[1] * pred.shape[2]

        rollout_mse = error_sum / count
        rollout_nmse = error_sum / target_sum.clamp_min(1e-8)

        metrics: Dict[str, float] = {}
        for idx, channel_name in enumerate(self.channel_names):
            metrics[f"rollout_mse_{channel_name}"] = float(rollout_mse[idx])
            metrics[f"rollout_nmse_{channel_name}"] = float(rollout_nmse[idx])

        focus_names = [name for name in ("Vy", "P") if name in self.channel_names]
        if focus_names:
            focus_metric = sum(metrics[f"rollout_nmse_{name}"] for name in focus_names) / len(focus_names)
        else:
            focus_metric = float(rollout_nmse.mean())
        metrics["hard_rollout_nmse"] = focus_metric
        return metrics

    def _build_history_row(
        self,
        train_loss: float,
        val_loss: Optional[float],
        train_channel_mse: Tensor,
        val_channel_mse: Tensor,
        rollout_metrics: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Build one history entry for the current epoch.

        Args:
            train_loss (float): Training loss for the epoch.
            val_loss (Optional[float]): Validation loss for the epoch.
            train_channel_mse (Tensor): Train channel MSE. (C,).
            val_channel_mse (Tensor): Validation channel MSE. (C,).
            rollout_metrics (Optional[Dict[str, float]]): Full-rollout validation metrics.

        Returns:
            Dict[str, Any]: Serializable history entry.
        """
        row: Dict[str, Any] = {
            "epoch": self.current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
            "rollout_steps": self.current_rollout_steps,
            "noise_std": self.current_noise_std,
        }

        for idx, channel_name in enumerate(self.channel_names):
            row[f"train_mse_{channel_name}"] = float(train_channel_mse[idx])
            row[f"val_mse_{channel_name}"] = float(val_channel_mse[idx])
            row[f"weight_{channel_name}"] = float(self.channel_weights[idx])

        if rollout_metrics is not None:
            row.update(rollout_metrics)

        return row

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        Run rollout training with adaptive channel weighting and rollout-based checkpointing.

        Args:
            train_loader (DataLoader): Training loader.
            val_loader (Optional[DataLoader]): Validation loader.
        """
        logger.info(f"start training on {hue.m}{self.device}{hue.q} with {hue.m}{self.max_epochs}{hue.q} epochs")
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            self._on_epoch_start()

            train_loss = self._run_epoch(train_loader, is_training=True)
            val_loss = self._run_epoch(val_loader, is_training=False) if val_loader else None

            train_channel_mse = self._mean_channel_mse("train")
            val_channel_mse = self._mean_channel_mse("val")

            self._update_channel_weights(train_channel_mse)
            self._update_curriculum()

            rollout_metrics = None
            is_best = False
            metric_str = ""
            should_eval_rollout = val_loader is not None and (
                self.current_epoch % self.rollout_eval_interval == 0 or self.current_epoch == self.max_epochs
            )

            if should_eval_rollout:
                rollout_metrics = self._evaluate_rollout(val_loader)
                metric_value = rollout_metrics[self.checkpoint_metric]
                is_best = metric_value < self.best_metric
                if is_best:
                    self.best_metric = metric_value
                    patience_counter = 0
                else:
                    patience_counter += 1
                tag = f" {hue.y}(best){hue.q}" if is_best else ""
                metric_str = f" | {self.checkpoint_metric}: {hue.m}{metric_value:.4e}{hue.q}{tag}"

            if self.scheduler:
                self.scheduler.step()

            extra_state = {
                "channel_ema": self.channel_ema.clone(),
                "channel_weights": self.channel_weights.clone(),
                "best_metric": self.best_metric,
            }
            if rollout_metrics is not None:
                extra_state["rollout_metrics"] = rollout_metrics
            self._save_checkpoint(
                val_loss if val_loss is not None else float("nan"),
                is_best=is_best,
                extra_state=extra_state,
            )

            duration = time.time() - epoch_start
            val_str = f" | val loss: {hue.m}{val_loss:.4e}{hue.q}" if val_loss is not None else ""
            logger.info(
                f"epoch {hue.b}{self.current_epoch:03d}{hue.q} | time: {hue.c}{duration:.1f}s{hue.q} "
                f"| train loss: {hue.m}{train_loss:.4e}{hue.q}{val_str}{metric_str} "
                f"| rollout: {hue.m}{self.current_rollout_steps}{hue.q} | noise: {hue.m}{self.current_noise_std:.4f}{hue.q}"
            )

            train_stats = " | ".join(
                f"{name}={hue.m}{value:.3e}{hue.q}" for name, value in zip(self.channel_names, train_channel_mse.tolist())
            )
            val_stats = " | ".join(
                f"{name}={hue.m}{value:.3e}{hue.q}" for name, value in zip(self.channel_names, val_channel_mse.tolist())
            )
            weight_stats = " | ".join(
                f"{name}={hue.m}{value:.2f}{hue.q}" for name, value in zip(self.channel_names, self.channel_weights.tolist())
            )
            logger.info(f"{hue.y}train mse:{hue.q} {train_stats}")
            logger.info(f"{hue.y}val mse:{hue.q} {val_stats}")
            logger.info(f"{hue.y}channel weights:{hue.q} {weight_stats}")

            row = self._build_history_row(train_loss, val_loss, train_channel_mse, val_channel_mse, rollout_metrics)
            self.history.append(row)

            if should_eval_rollout and patience_counter >= self.patience:
                logger.info(f"early stopping triggered at epoch {hue.m}{self.current_epoch}{hue.q}")
                break

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"{hue.g}training finished in {time.time() - start_time:.1f}s{hue.q}")

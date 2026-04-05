# Main script for HyperFlowNet training, inference, and probing
# Author: Shengning Wang

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import config

from data.flow_data import FlowData
from data.boundary import BoundaryCondition
from data.flow_plot import (
    plot_error_heatmap,
    plot_metrics_comparison,
    plot_rollout_error,
    plot_training_curves,
)
from data.flow_vis import FlowVis
from models.hflownet import HyperFlowNet
from training.hyperflow_trainer import HyperFlowTrainer
from utils.hue_logger import hue, logger
from utils.scaler import MinMaxScalerTensor, StandardScalerTensor
from utils.seeder import seed_everything


class ScaledFlowDataset(Dataset):
    """
    Dataset wrapper that applies feature and coordinate scaling on demand.
    """

    def __init__(
        self,
        dataset: FlowData,
        feature_scaler: StandardScalerTensor,
        coord_scaler: MinMaxScalerTensor,
    ) -> None:
        """
        Initialize the scaled dataset wrapper.

        Args:
            dataset (FlowData): Raw flow dataset.
            feature_scaler (StandardScalerTensor): Fitted feature scaler.
            coord_scaler (MinMaxScalerTensor): Fitted coordinate scaler.
        """
        self.dataset = dataset
        self.feature_scaler = feature_scaler
        self.coord_scaler = coord_scaler

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, float, float]:
        """
        Load and scale one rollout sample.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[Tensor, Tensor, float, float]: Scaled sequence, scaled coordinates,
                normalized start time, and normalized time increment.
        """
        seq, coords, start_t_norm, dt_norm = self.dataset[idx]
        seq_std = self.feature_scaler.transform(seq)
        coords_norm = self.coord_scaler.transform(coords)
        return seq_std, coords_norm, start_t_norm, dt_norm


class Metrics:
    """
    Evaluation metrics for autoregressive flow prediction.
    """

    SUPPORTED_METRICS = ("nmse", "mse", "rmse", "mae", "r2", "accuracy", "max_error")

    def __init__(self, channel_names: List[str], metrics: List[str] = None) -> None:
        """
        Initialize the metric evaluator.

        Args:
            channel_names (List[str]): Ordered field names.
            metrics (List[str], optional): Metrics to compute.
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

            if "mae" in self.metrics:
                channel_metrics["global"]["mae"] = float(abs_diff.mean())
                channel_metrics["step_wise"]["mae"] = abs_diff.mean(dim=1).tolist()

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

            if "max_error" in self.metrics:
                channel_metrics["global"]["max_error"] = float(abs_diff.max())
                channel_metrics["step_wise"]["max_error"] = abs_diff.max(dim=1).values.tolist()

            results[ch_name] = channel_metrics

        return results


def build_case_datasets(args: argparse.Namespace) -> Tuple[FlowData, FlowData, FlowData]:
    """
    Build train, validation, and test datasets before window augmentation.

    Args:
        args (argparse.Namespace): Parsed configuration.

    Returns:
        Tuple[FlowData, FlowData, FlowData]: Raw train, validation, and test datasets.
    """
    all_cases = FlowData.discover_cases(args.data_dir)

    rng = np.random.default_rng(seed=args.seed)
    rng.shuffle(all_cases)

    num_train = len(all_cases) - args.val_cases - args.test_cases
    splits = {
        "train": all_cases[:num_train],
        "val": all_cases[num_train:num_train + args.val_cases],
        "test": all_cases[num_train + args.val_cases:],
    }

    logger.info(
        f"dataset split | train: {hue.m}{len(splits['train'])}{hue.q}, "
        f"val: {hue.m}{len(splits['val'])}{hue.q}, "
        f"test: {hue.m}{len(splits['test'])}{hue.q}"
    )

    train_data = FlowData(args.data_dir, splits["train"], spatial_dim=args.spatial_dim)
    val_data = FlowData(args.data_dir, splits["val"], spatial_dim=args.spatial_dim)
    test_data = FlowData(args.data_dir, splits["test"], spatial_dim=args.spatial_dim)
    return train_data, val_data, test_data


def fit_scalers(train_data: FlowData) -> Dict[str, object]:
    """
    Fit feature and coordinate scalers on raw training cases.

    Args:
        train_data (FlowData): Raw training dataset.

    Returns:
        Dict[str, object]: Dictionary containing fitted scalers.
    """
    train_seqs = torch.cat(train_data.seqs, dim=0)
    train_coords = torch.cat(train_data.coords, dim=0)

    feature_scaler = StandardScalerTensor().fit(train_seqs, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)
    return {
        "feature_scaler": feature_scaler,
        "coord_scaler": coord_scaler,
    }


def build_loaders(
    args: argparse.Namespace,
    train_data: FlowData,
    val_data: FlowData,
    scalers: Dict[str, object],
) -> Tuple[DataLoader, DataLoader]:
    """
    Build scaled training and validation loaders.

    Args:
        args (argparse.Namespace): Parsed configuration.
        train_data (FlowData): Augmented training dataset.
        val_data (FlowData): Augmented validation dataset.
        scalers (Dict[str, object]): Fitted feature and coordinate scalers.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    train_dataset = ScaledFlowDataset(train_data, scalers["feature_scaler"], scalers["coord_scaler"])
    val_dataset = ScaledFlowDataset(val_data, scalers["feature_scaler"], scalers["coord_scaler"])

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_model(args: argparse.Namespace) -> HyperFlowNet:
    """
    Instantiate HyperFlowNet from the parsed configuration.

    Args:
        args (argparse.Namespace): Parsed configuration.

    Returns:
        HyperFlowNet: Initialized model.
    """
    num_channels = len(args.channel_names)
    return HyperFlowNet(
        in_channels=num_channels,
        out_channels=num_channels,
        spatial_dim=args.spatial_dim,
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        num_slices=args.num_slices,
        ffn_ratio=args.ffn_ratio,
        use_spatial_encoding=args.use_spatial_encoding,
        use_temporal_encoding=args.use_temporal_encoding,
        num_fixed_bands=args.num_fixed_bands,
        num_learned_features=args.num_learned_features,
        time_features=args.time_features,
        freq_base=args.freq_base,
        predict_delta=args.predict_delta,
        delta_scale=args.delta_scale,
    )


def build_trainer(
    args: argparse.Namespace,
    model: HyperFlowNet,
    scalers: Dict[str, object],
    output_dir: Path,
    boundary_condition: BoundaryCondition | None = None,
) -> HyperFlowTrainer:
    """
    Build the HyperFlowNet trainer.

    Args:
        args (argparse.Namespace): Parsed configuration.
        model (HyperFlowNet): HyperFlowNet model.
        scalers (Dict[str, object]): Fitted scalers to be stored in checkpoints.
        output_dir (Path): Output directory.
        boundary_condition (BoundaryCondition | None): Optional hard boundary-condition enforcer.

    Returns:
        HyperFlowTrainer: Initialized trainer.
    """
    return HyperFlowTrainer(
        model=model,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        rollout_steps=args.rollout_steps,
        stage_ratios=args.stage_ratios,
        teacher_forcing_lows=args.teacher_forcing_lows,
        stage_lrs=args.stage_lrs,
        stage_warmup_ratio=args.stage_warmup_ratio,
        stage_min_lr_ratio=args.stage_min_lr_ratio,
        input_noise_std=args.input_noise_std,
        input_noise_decay=args.input_noise_decay,
        eval_rollout_steps=args.eval_rollout_steps,
        channel_weights=args.channel_weights,
        delta_loss_weight=args.delta_loss_weight,
        step_weight_power=args.step_weight_power,
        loss_eps=args.loss_eps,
        boundary_condition=boundary_condition,
        scalers=scalers,
        output_dir=output_dir,
        device=args.device,
    )


def save_run_config(args: argparse.Namespace, output_dir: Path, num_params: int) -> None:
    """
    Save the current run configuration.

    Args:
        args (argparse.Namespace): Parsed configuration.
        output_dir (Path): Output directory.
        num_params (int): Number of model parameters.
    """
    payload = vars(args).copy()
    payload["num_params"] = num_params
    with open(output_dir / "config.json", "w") as file:
        json.dump(payload, file, indent=2)


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    """
    Resolve the checkpoint path used during inference.

    Args:
        args (argparse.Namespace): Parsed configuration.

    Returns:
        Path: Checkpoint path.
    """
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / args.checkpoint_name

    if checkpoint_path.exists():
        return checkpoint_path

    fallback_path = output_dir / "ckpt.pt"
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"checkpoint not found under {output_dir}")


def restore_scalers(checkpoint: Dict[str, object]) -> Dict[str, object]:
    """
    Restore feature and coordinate scalers from a checkpoint.

    Args:
        checkpoint (Dict[str, object]): Loaded checkpoint payload.

    Returns:
        Dict[str, object]: Restored scalers and optional boundary condition.
    """
    feature_scaler = StandardScalerTensor()
    feature_scaler.load_state_dict(checkpoint["scaler_state_dict"]["feature_scaler"])

    coord_scaler = MinMaxScalerTensor(norm_range="bipolar")
    coord_scaler.load_state_dict(checkpoint["scaler_state_dict"]["coord_scaler"])

    scalers = {
        "feature_scaler": feature_scaler,
        "coord_scaler": coord_scaler,
    }
    boundary_state = checkpoint["scaler_state_dict"].get("boundary_condition")
    if boundary_state is not None:
        boundary_condition = BoundaryCondition()
        boundary_condition.load_state_dict(boundary_state)
        scalers["boundary_condition"] = boundary_condition

    return scalers


def train_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the training workflow.

    Args:
        args (argparse.Namespace): Parsed configuration.
    """
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading datasets...")
    train_data, val_data, _ = build_case_datasets(args)
    scalers = fit_scalers(train_data)
    boundary_condition = None

    if args.use_hard_bc:
        boundary_condition = BoundaryCondition().fit(
            train_data,
            scalers["feature_scaler"],
            velocity_channels=list(range(args.spatial_dim)),
            velocity_threshold=args.velocity_threshold,
        )
        scalers["boundary_condition"] = boundary_condition

    logger.info("building training windows...")
    FlowData.augment_windows(train_data, args.win_len, args.train_win_stride)
    FlowData.augment_windows(val_data, args.win_len, args.val_win_stride)

    train_loader, val_loader = build_loaders(args, train_data, val_data, scalers)

    model = build_model(args)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"model parameters: {hue.m}{num_params}{hue.q}")
    save_run_config(args, output_dir, num_params)

    trainer = build_trainer(args, model, scalers, output_dir, boundary_condition=boundary_condition)
    trainer.fit(train_loader, val_loader)

    history_path = output_dir / "history.json"
    if history_path.exists():
        plot_training_curves(
            history_paths={"HyperFlowNet": str(history_path)},
            output_path=str(output_dir / "training_curve.png"),
        )


def inference_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the inference workflow.

    Args:
        args (argparse.Namespace): Parsed configuration.
    """
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_path = resolve_checkpoint_path(args)
    device = torch.device(args.device)

    logger.info(f"loading checkpoint: {hue.b}{checkpoint_path.name}{hue.q}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    scalers = restore_scalers(checkpoint)
    boundary_condition = scalers.get("boundary_condition")
    if boundary_condition is not None:
        logger.info(
            f"boundary condition restored: "
            f"{hue.m}{int(boundary_condition.wall_mask.sum())}{hue.q} wall nodes"
        )

    _, _, test_data = build_case_datasets(args)
    test_dataset = ScaledFlowDataset(test_data, scalers["feature_scaler"], scalers["coord_scaler"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = build_model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    num_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"model parameters: {hue.m}{num_params}{hue.q}")

    metrics_evaluator = Metrics(args.channel_names)
    visualizer = FlowVis(output_dir=output_dir, spatial_dim=args.spatial_dim)

    case_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    with torch.no_grad():
        for idx, (seq_std, coords_norm, start_t_norm, dt_norm) in enumerate(test_loader):
            seq_std = seq_std.to(device)
            coords_norm = coords_norm.to(device)
            start_t_norm = start_t_norm.to(device)
            dt_norm = dt_norm.to(device)

            steps = seq_std.shape[1] - 1
            initial_state = seq_std[:, 0]
            pred_seq_std = model.predict(
                inputs=initial_state,
                coords=coords_norm,
                steps=steps,
                start_t_norm=start_t_norm,
                dt_norm=dt_norm,
                boundary_condition=boundary_condition,
            )

            pred_seq = scalers["feature_scaler"].inverse_transform(pred_seq_std).cpu().squeeze(0)
            gt_seq = scalers["feature_scaler"].inverse_transform(seq_std).cpu().squeeze(0)
            coords_raw = test_data.coords[idx].cpu()
            case_name = test_data.case_names[idx]

            metrics = metrics_evaluator.compute(pred_seq, gt_seq)
            case_metrics[case_name] = metrics

            log_items = []
            for ch_name in args.channel_names:
                nmse = metrics[ch_name]["global"]["nmse"]
                r2 = metrics[ch_name]["global"]["r2"]
                accuracy = metrics[ch_name]["global"]["accuracy"]
                log_items.append(
                    f"{hue.c}{ch_name}{hue.q}: NMSE={hue.m}{nmse:.2e}{hue.q}, "
                    f"R2={hue.m}{r2:.4f}{hue.q}, ACC={hue.m}{accuracy:.2f}%{hue.q}"
                )
            logger.info(f"case {hue.b}{case_name}{hue.q} | " + " | ".join(log_items))

            torch.save(pred_seq, output_dir / f"{case_name}_pred.pt")

            visualizer.animate_comparison(
                gt=gt_seq,
                pred=pred_seq,
                coords=coords_raw,
                case_name=case_name,
            )

            plot_rollout_error(
                pred=pred_seq,
                gt=gt_seq,
                channel_names=args.channel_names,
                output_path=str(output_dir / f"{case_name}_rollout_error.png"),
            )

            num_steps = pred_seq.shape[0]
            for step_idx, step_name in [(0, "first"), (num_steps // 2, "mid"), (num_steps - 1, "last")]:
                plot_error_heatmap(
                    gt=gt_seq,
                    pred=pred_seq,
                    coords=coords_raw,
                    timestep=step_idx,
                    channel_names=args.channel_names,
                    output_path=str(output_dir / f"{case_name}_error_t{step_idx}_{step_name}.png"),
                )

    with open(output_dir / "test_metrics.json", "w") as file:
        json.dump(case_metrics, file, indent=2)

    history_path = output_dir / "history.json"
    if history_path.exists():
        plot_training_curves(
            history_paths={"HyperFlowNet": str(history_path)},
            output_path=str(output_dir / "training_curve.png"),
        )

    plot_metrics_comparison(
        metrics_paths={"HyperFlowNet": str(output_dir / "test_metrics.json")},
        output_path=str(output_dir / "metrics_comparison.png"),
        channel_names=args.channel_names,
    )

    logger.info(f"{hue.g}inference finished.{hue.q}")


def probe_pipeline(args: argparse.Namespace) -> None:
    """
    Execute one training-like step to estimate peak GPU memory.

    Args:
        args (argparse.Namespace): Parsed configuration.
    """
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, probe skipped.")
        return

    seed_everything(args.seed)
    train_data, _, _ = build_case_datasets(args)
    scalers = fit_scalers(train_data)
    boundary_condition = None
    if args.use_hard_bc:
        boundary_condition = BoundaryCondition().fit(
            train_data,
            scalers["feature_scaler"],
            velocity_channels=list(range(args.spatial_dim)),
            velocity_threshold=args.velocity_threshold,
        )
        scalers["boundary_condition"] = boundary_condition
    FlowData.augment_windows(train_data, args.win_len, args.train_win_stride)

    probe_dataset = ScaledFlowDataset(train_data, scalers["feature_scaler"], scalers["coord_scaler"])
    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    batch = next(iter(probe_loader))
    batch = [item.to(device) if isinstance(item, Tensor) else item for item in batch]
    seq_std, coords_norm, _, _ = batch

    model = build_model(args)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    trainer = build_trainer(args, model, scalers=scalers, output_dir=Path(args.output_dir),
        boundary_condition=boundary_condition)

    rollout_steps = min(args.probe_rollout_steps, seq_std.shape[1] - 1)
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    logger.info(
        f"{hue.y}probe config:{hue.q} "
        f"batch={hue.m}{seq_std.shape[0]}{hue.q}, "
        f"frames={hue.m}{seq_std.shape[1]}{hue.q}, "
        f"nodes={hue.m}{seq_std.shape[2]}{hue.q}, "
        f"channels={hue.m}{seq_std.shape[3]}{hue.q}, "
        f"rollout={hue.m}{rollout_steps}{hue.q}, "
        f"params={hue.m}{num_params}{hue.q}"
    )

    torch.cuda.reset_peak_memory_stats(device)
    loss = trainer.compute_rollout_loss(
        batch=batch,
        rollout_steps=rollout_steps,
        teacher_forcing_ratio=0.0,
        noise_std=0.0,
    )

    loss.backward()
    trainer.optimizer.step()

    peak = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    ratio = 100.0 * peak / total

    if ratio < 75.0:
        status = f"{hue.g}SAFE{hue.q}"
    elif ratio < 92.0:
        status = f"{hue.y}WARNING{hue.q}"
    else:
        status = f"{hue.r}CRITICAL{hue.q}"

    logger.info(
        f"peak usage: {hue.m}{peak / 1e9:.2f}{hue.q} GB / {hue.m}{total / 1e9:.2f}{hue.q} GB "
        f"({hue.m}{ratio:.1f}%{hue.q}) -> {status}"
    )


if __name__ == "__main__":
    args = config.get_args()
    pipeline_map = {
        "probe": probe_pipeline,
        "train": train_pipeline,
        "infer": inference_pipeline,
    }

    for mode in args.mode:
        pipeline_map[mode](args)

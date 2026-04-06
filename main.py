# Main script for HyperFlowNet training, inference, and probing
# Author: Shengning Wang

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import config

from data.boundary import BoundaryCondition
from data.flow_data import FlowData
from data.flow_vis import FlowVis

from models.hflownet import HyperFlowNet
from training.hyperflow_trainer import HyperFlowTrainer

from utils.seeder import seed_everything
from utils.hue_logger import hue, logger
from utils.scaler import MinMaxScalerTensor, StandardScalerTensor


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


def _log_pipeline_banner(name: str, stage: str, color: str) -> None:
    """
    Log a visible banner for pipeline boundaries.

    Args:
        name (str): Pipeline name.
        stage (str): Pipeline stage label.
        color (str): ANSI color prefix from HueLogger.
    """
    title = f" [{name}] {stage} "
    width = 84
    num_eq = max(8, width - len(title))
    left = "=" * (num_eq // 2)
    right = "=" * (num_eq - len(left))
    logger.info(f"{color}{left}{title}{right}{hue.q}")


def _log_pipeline_start(name: str) -> None:
    """
    Log the start of a pipeline.

    Args:
        name (str): Pipeline name.
    """
    _log_pipeline_banner(name, "START", hue.c)


def _log_pipeline_end(name: str) -> None:
    """
    Log the end of a pipeline.

    Args:
        name (str): Pipeline name.
    """
    _log_pipeline_banner(name, "END", hue.g)


def _build_model(args: argparse.Namespace) -> HyperFlowNet:
    """
    Instantiate HyperFlowNet from the parsed configuration.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

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
        num_slices=args.num_slices,
        num_heads=args.num_heads,
        use_spatial_encoding=args.use_spatial_encoding,
        use_temporal_encoding=args.use_temporal_encoding,
        coords_features=args.coords_features,
        time_features=args.time_features,
        freq_base=args.freq_base,
    )


def _build_trainer(
    args: argparse.Namespace,
    model: HyperFlowNet,
    scalers: Dict[str, object],
    output_dir: Path,
    boundary_condition: BoundaryCondition | None = None,
) -> HyperFlowTrainer:
    """
    Instantiate the HyperFlowNet trainer.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (HyperFlowNet): HyperFlowNet model.
        scalers (Dict[str, object]): Dictionary of data scalers for checkpoint saving.
        output_dir (Path): Directory for saving artifacts.
        boundary_condition (BoundaryCondition | None): Optional boundary-condition enforcer.

    Returns:
        HyperFlowTrainer: Configured trainer.
    """
    return HyperFlowTrainer(
        model=model,
        lr=args.lr,
        max_epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        max_rollout_steps=args.max_rollout_steps,
        rollout_patience=args.rollout_patience,
        noise_std_init=args.noise_std_init,
        noise_decay=args.noise_decay,
        teacher_forcing_init=args.teacher_forcing_init,
        teacher_forcing_decay=args.teacher_forcing_decay,
        teacher_forcing_floor=args.teacher_forcing_floor,
        boundary_condition=boundary_condition,
        channel_weights=args.channel_weights,
        scalers=scalers,
        output_dir=output_dir,
        device=args.device,
    )


def data_pipeline(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, FlowData, Dict[str, object], BoundaryCondition | None]:
    """
    Build datasets and training-time data utilities once for all pipelines.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, FlowData, Dict[str, object], BoundaryCondition | None]:
            Train, validation, and test loaders together with the raw test dataset,
            fitted scalers, and optional boundary condition.
    """
    _log_pipeline_start("DATA PIPELINE")
    train_data, val_data, test_data = FlowData.spawn(
        data_dir=args.data_dir,
        spatial_dim=args.spatial_dim,
        win_len=args.win_len,
        win_stride=args.win_stride,
    )

    train_seqs = torch.cat(train_data.seqs, dim=0)
    train_coords = torch.cat(train_data.coords, dim=0)
    feature_scaler = StandardScalerTensor().fit(train_seqs, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)
    scalers: Dict[str, object] = {
        "feature_scaler": feature_scaler,
        "coord_scaler": coord_scaler,
    }

    boundary_condition = None
    if args.use_hard_bc:
        boundary_condition = BoundaryCondition()
        boundary_condition.fit(
            train_data,
            feature_scaler,
            velocity_channels=list(range(args.spatial_dim)),
            velocity_threshold=args.velocity_threshold,
        )
        scalers["boundary_condition"] = boundary_condition

    processed_splits = []
    for dataset in (train_data, val_data, test_data):
        samples = []
        for seq, coord, t0_norm, dt_norm in dataset:
            seq_std = feature_scaler.transform(seq)
            coords_norm = coord_scaler.transform(coord)
            t0_norm = torch.tensor(t0_norm, dtype=seq_std.dtype)
            dt_norm = torch.tensor(dt_norm, dtype=seq_std.dtype)
            samples.append((seq_std, coords_norm, t0_norm, dt_norm))
        processed_splits.append(samples)

    train_samples, val_samples, test_samples = processed_splits
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_samples,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    _log_pipeline_end("DATA PIPELINE")
    return train_loader, val_loader, test_loader, test_data, scalers, boundary_condition


def probe_pipeline(
    args: argparse.Namespace,
    train_loader: DataLoader,
    boundary_condition: BoundaryCondition | None,
) -> None:
    """
    Run one full-rollout step to estimate peak GPU memory usage.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_loader (DataLoader): Training data loader.
        boundary_condition (BoundaryCondition | None): Optional boundary-condition enforcer.
    """
    _log_pipeline_start("PROBE PIPELINE")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning("No CUDA device - probe skipped.")
        _log_pipeline_end("PROBE PIPELINE")
        return

    seq_std, coords_norm, t0_norm, dt_norm = next(iter(train_loader))
    seq_std = seq_std.to(device)
    coords_norm = coords_norm.to(device)
    t0_norm = t0_norm.to(device)
    dt_norm = dt_norm.to(device)

    output_dir = Path(args.output_dir)
    model = _build_model(args)
    trainer = _build_trainer(args, model, scalers={}, output_dir=output_dir, boundary_condition=boundary_condition)
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    B, T, N, C = seq_std.shape
    k = min(args.max_rollout_steps, T - 1)
    num_params = sum(p.numel() for p in trainer.model.parameters())

    logger.info(
        f"{hue.y}probe config:{hue.q} "
        f"batch={hue.m}{B}{hue.q}, frames={hue.m}{T}{hue.q}, "
        f"nodes={hue.m}{N}{hue.q}, channels={hue.m}{C}{hue.q}, "
        f"max_rollout={hue.m}{k}{hue.q}, params={hue.m}{num_params}{hue.q}"
    )

    torch.cuda.reset_peak_memory_stats(device)

    loss = trainer.compute_rollout_loss(
        batch=(seq_std, coords_norm, t0_norm, dt_norm),
        rollout_steps=k,
        teacher_forcing_ratio=0.0,
        noise_std=0.0,
    )
    loss.backward()
    trainer.optimizer.step()

    peak = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    pct = 100.0 * peak / total

    if pct < 75.0:
        status = f"{hue.g}SAFE{hue.q}"
    elif pct < 92.0:
        status = f"{hue.y}WARNING - close to limit{hue.q}"
    else:
        status = f"{hue.r}CRITICAL - likely OOM in real training{hue.q}"

    logger.info(
        f"{hue.y}device: {hue.b}{torch.cuda.get_device_name(device)}{hue.q} "
        f"({hue.m}{total / 1e9:.1f}{hue.q} GB)"
    )
    logger.info(f"peak usage: {hue.m}{peak / 1e9:.2f}{hue.q} GB ({hue.m}{pct:.1f}{hue.q} %) -> {status}")
    _log_pipeline_end("PROBE PIPELINE")


def train_pipeline(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scalers: Dict[str, object],
    boundary_condition: BoundaryCondition | None,
) -> None:
    """
    Execute the training workflow.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        scalers (Dict[str, object]): Fitted feature and coordinate scalers.
        boundary_condition (BoundaryCondition | None): Optional boundary-condition enforcer.
    """
    _log_pipeline_start("TRAIN PIPELINE")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(args)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    trainer = _build_trainer(args, model, scalers, output_dir, boundary_condition=boundary_condition)
    trainer.fit(train_loader, val_loader)
    _log_pipeline_end("TRAIN PIPELINE")


def inference_pipeline(
    args: argparse.Namespace,
    test_loader: DataLoader,
    test_data: FlowData,
) -> None:
    """
    Execute the inference workflow using the saved training artifacts.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        test_loader (DataLoader): Test data loader from the shared data pipeline.
        test_data (FlowData): Raw test dataset from the shared data pipeline.
    """
    _log_pipeline_start("INFERENCE PIPELINE")
    device = torch.device(args.device)
    run_dir = Path(args.output_dir)
    model_path = run_dir / "ckpt.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"ckpt.pt not found at {model_path}.")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    scaler_state = checkpoint["scaler_state_dict"]

    feature_scaler = StandardScalerTensor()
    feature_scaler.load_state_dict(scaler_state["feature_scaler"])

    coord_scaler = MinMaxScalerTensor(norm_range="bipolar")
    coord_scaler.load_state_dict(scaler_state["coord_scaler"])

    boundary_condition = None
    if "boundary_condition" in scaler_state:
        boundary_condition = BoundaryCondition()
        boundary_condition.load_state_dict(scaler_state["boundary_condition"])

    if boundary_condition is not None:
        logger.info(f"boundary condition: {hue.m}{int(boundary_condition.wall_mask.sum())}{hue.q} wall nodes")

    model = _build_model(args)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    visualizer = FlowVis(output_dir=run_dir, spatial_dim=args.spatial_dim, channel_names=args.channel_names)
    metrics_evaluator = Metrics(channel_names=args.channel_names)

    case_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    with torch.no_grad():
        for i, (seq_std, coords_norm, t0_norm, dt_norm) in enumerate(test_loader):
            seq_std = seq_std.to(device)
            coords_norm = coords_norm.to(device)
            t0_norm = t0_norm.to(device)
            dt_norm = dt_norm.to(device)

            case_name = test_data.case_names[i]
            steps = seq_std.shape[1] - 1
            initial_state = seq_std[:, 0]
            coords_raw = test_data.coords[i].cpu()

            pred_seq_std = model.predict(initial_state, coords_norm, steps, t0_norm, dt_norm, boundary_condition)

            pred_seq = feature_scaler.inverse_transform(pred_seq_std).cpu().squeeze(0)
            gt_seq = feature_scaler.inverse_transform(seq_std).cpu().squeeze(0)

            metrics = metrics_evaluator.compute(pred_seq, gt_seq)
            case_metrics[case_name] = metrics

            log_metrics = []
            for ch_name in args.channel_names:
                nmse = metrics[ch_name]["global"]["nmse"]
                r2 = metrics[ch_name]["global"]["r2"]
                accuracy = metrics[ch_name]["global"]["accuracy"]
                log_metrics.append(
                    f"{hue.c}{ch_name}:{hue.q} NMSE={hue.m}{nmse:.2e}{hue.q}, "
                    f"R2={hue.m}{r2:.4f}{hue.q}, ACC={hue.m}{accuracy:.2f}%{hue.q}"
                )

            logger.info(f"case {hue.b}{case_name}{hue.q} | " + " | ".join(log_metrics))

            torch.save(pred_seq, run_dir / f"{case_name}_pred.pt")

            num_nodes = int(coords_raw.shape[0])
            focus_channel_idx = len(args.channel_names) - 1
            focus_bbox_rel = (0.60, 1.00, 0.00, 1.00) if args.spatial_dim == 2 \
            else (0.60, 1.00, 0.00, 1.00, 0.00, 1.00)

            visualizer.render_full(
                gt=gt_seq,
                pred=pred_seq,
                coords=coords_raw,
                case_name=case_name,
                num_nodes=num_nodes,
                num_params=num_params,
            )
            visualizer.render_focus(
                gt=gt_seq,
                pred=pred_seq,
                coords=coords_raw,
                case_name=case_name,
                num_nodes=num_nodes,
                num_params=num_params,
                focus_channel_idx=focus_channel_idx,
                focus_bbox_rel=focus_bbox_rel,
            )

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(case_metrics, f, indent=4)

    _log_pipeline_end("INFERENCE PIPELINE")


if __name__ == "__main__":
    args = config.get_args()
    seed_everything(args.seed)

    train_loader, val_loader, test_loader, test_data, scalers, boundary_condition = data_pipeline(args)

    if "probe" in args.mode:
        probe_pipeline(args, train_loader, boundary_condition)
    if "train" in args.mode:
        train_pipeline(args, train_loader, val_loader, scalers, boundary_condition)
    if "infer" in args.mode:
        inference_pipeline(args, test_loader, test_data)

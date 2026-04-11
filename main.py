# Main script for HyperFlowNet training, inference, and probing
# Author: Shengning Wang

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

import config

from data.boundary import BoundaryCondition
from data.flow_data import FlowData
from data.flow_metrics import Metrics
from data.flow_vis import FlowVis
from data.initial_state import initial_state_from_label

from models.hflownet import HyperFlowNet
from training.hyperflow_trainer import HyperFlowTrainer

from utils.hue_logger import hue, logger
from utils.scaler import MinMaxScalerTensor, StandardScalerTensor
from utils.seeder import seed_everything


def log_pipeline(name: str, stage: str) -> None:
    """
    Log a visible pipeline banner.

    Args:
        name (str): Pipeline name.
        stage (str): Stage label such as ``START`` or ``END``.
    """
    color = hue.c if stage == "START" else hue.g
    title = f" [{name}] {stage} "
    width = 84
    num_eq = max(8, width - len(title))
    left = "=" * (num_eq // 2)
    right = "=" * (num_eq - len(left))
    logger.info(f"{color}{left}{title}{right}{hue.q}")


def build_model(
    args: argparse.Namespace | None = None,
    model_args: Dict[str, Any] | None = None,
) -> Tuple[HyperFlowNet, Dict[str, Any]]:
    """
    Build one single-channel HyperFlowNet and expose its constructor args.

    Args:
        args (argparse.Namespace | None): Parsed command-line arguments.
        model_args (Dict[str, Any] | None): Explicit model configuration.

    Returns:
        Tuple[HyperFlowNet, Dict[str, Any]]: Model instance and constructor arguments.
    """
    if model_args is None:
        model_args = {
            "in_channels": 2,
            "out_channels": 1,
            "spatial_dim": args.spatial_dim,
            "width": args.width,
            "depth": args.depth,
            "num_slices": args.num_slices,
            "latent_dim": args.latent_dim,
            "num_anchors": args.num_anchors,
            "time_features": args.time_features,
            "freq_base": args.freq_base,
        }
    return HyperFlowNet(**model_args), model_args


def build_channel_data(
    args: argparse.Namespace,
    train_data: FlowData,
    val_data: FlowData,
    channel_name: str,
    channel_idx: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, object], Dict[str, Any], BoundaryCondition | None]:
    """
    Build one channel-specific training runtime.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_data (FlowData): Training split.
        val_data (FlowData): Validation split.
        channel_name (str): Target channel name.
        channel_idx (int): Target channel index.

    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, object], Dict[str, Any], BoundaryCondition | None]:
            Train loader, validation loader, scalers, checkpoint params, and optional BC.
    """
    train_states = torch.cat([seq[..., channel_idx:channel_idx + 1] for seq in train_data.seqs], dim=0)
    train_coords = torch.cat(train_data.coords, dim=0)
    train_labels = torch.stack(train_data.labels, dim=0)

    state_scaler = StandardScalerTensor().fit(train_states, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)
    label_scaler = StandardScalerTensor().fit(train_labels, channel_dim=-1)
    scalers = {
        "state_scaler": state_scaler,
        "coord_scaler": coord_scaler,
        "label_scaler": label_scaler,
    }

    boundary_condition = None
    if args.use_hard_bc and channel_name in {"Vx", "Vy"}:
        boundary_condition = BoundaryCondition()
        boundary_condition.fit(
            SimpleNamespace(seqs=[seq[..., channel_idx:channel_idx + 1] for seq in train_data.seqs]),
            state_scaler,
            velocity_channels=[0],
            velocity_threshold=args.velocity_threshold,
        )

    _, model_args = build_model(args=args)
    params = {
        "model_args": model_args,
        "channel_name": channel_name,
        "channel_idx": channel_idx,
        "use_hard_bc": boundary_condition is not None,
    }
    if boundary_condition is not None:
        params["boundary_condition_state"] = boundary_condition.state_dict()

    train_samples = []
    for seq, coord, label, t0_norm, dt_norm in train_data:
        seq_std = state_scaler.transform(seq[..., channel_idx:channel_idx + 1])
        coords_norm = coord_scaler.transform(coord)
        label_norm = label_scaler.transform(label.view(1, -1)).view(-1)
        train_samples.append((
            seq_std,
            coords_norm,
            label_norm,
            torch.tensor(t0_norm, dtype=seq_std.dtype),
            torch.tensor(dt_norm, dtype=seq_std.dtype),
        ))

    val_samples = []
    for seq, coord, label, t0_norm, dt_norm in val_data:
        seq_std = state_scaler.transform(seq[..., channel_idx:channel_idx + 1])
        coords_norm = coord_scaler.transform(coord)
        label_norm = label_scaler.transform(label.view(1, -1)).view(-1)
        val_samples.append((
            seq_std,
            coords_norm,
            label_norm,
            torch.tensor(t0_norm, dtype=seq_std.dtype),
            torch.tensor(dt_norm, dtype=seq_std.dtype),
        ))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_samples, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=pin_memory)
    val_loader = DataLoader(
        val_samples, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=pin_memory)
    return train_loader, val_loader, scalers, params, boundary_condition


def load_channel_checkpoint(
    channel_dir: Path,
    device: torch.device,
) -> Tuple[HyperFlowNet, Dict[str, object], Dict[str, Any], BoundaryCondition | None]:
    """
    Load one channel checkpoint and its inference artifacts.

    Args:
        channel_dir (Path): Channel run directory.
        device (torch.device): Inference device.

    Returns:
        Tuple[HyperFlowNet, Dict[str, object], Dict[str, Any], BoundaryCondition | None]:
            Model, scalers, checkpoint params, and optional BC.
    """
    checkpoint = torch.load(channel_dir / "ckpt.pt", map_location=device, weights_only=True)
    params = checkpoint["params"]
    scaler_state = checkpoint["scaler_state_dict"]

    state_scaler = StandardScalerTensor()
    state_scaler.load_state_dict(scaler_state["state_scaler"])

    coord_scaler = MinMaxScalerTensor(norm_range="bipolar")
    coord_scaler.load_state_dict(scaler_state["coord_scaler"])

    label_scaler = StandardScalerTensor()
    label_scaler.load_state_dict(scaler_state["label_scaler"])

    boundary_condition = None
    if "boundary_condition_state" in params:
        boundary_condition = BoundaryCondition()
        boundary_condition.load_state_dict(params["boundary_condition_state"])

    model, _ = build_model(model_args=params["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scalers = {
        "state_scaler": state_scaler,
        "coord_scaler": coord_scaler,
        "label_scaler": label_scaler,
    }
    return model, scalers, params, boundary_condition


# ============================================================
# Data Pipeline
# ============================================================


def data_pipeline(args: argparse.Namespace) -> Tuple[FlowData, FlowData, FlowData]:
    """
    Build train / validation / test splits.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple[FlowData, FlowData, FlowData]: Raw train, validation, and test splits.
    """
    log_pipeline("DATA PIPELINE", "START")
    datasets = FlowData.spawn(
        data_dir=args.data_dir,
        spatial_dim=args.spatial_dim,
        win_len=args.win_len,
        win_stride=args.win_stride,
    )
    log_pipeline("DATA PIPELINE", "END")
    return datasets


# ============================================================
# Probing Pipeline
# ============================================================


def probe_pipeline(
    args: argparse.Namespace,
    train_data: FlowData,
    val_data: FlowData,
) -> None:
    """
    Run one parallel four-channel rollout step to estimate peak GPU memory usage.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_data (FlowData): Training split.
        val_data (FlowData): Validation split.
    """
    log_pipeline("PROBE PIPELINE", "START")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning("No CUDA device - probe skipped.")
        log_pipeline("PROBE PIPELINE", "END")
        return

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    trainers = []
    losses = []
    total_params = 0
    B = T = N = k = 0

    for channel_idx, channel_name in enumerate(args.channel_names):
        channel_dir = output_root / channel_name
        channel_dir.mkdir(parents=True, exist_ok=True)

        train_loader, _, scalers, params, boundary_condition = build_channel_data(
            args, train_data, val_data, channel_name, channel_idx)
        model, _ = build_model(model_args=params["model_args"])
        trainer = HyperFlowTrainer(
            model=model,
            lr=args.lr,
            max_epochs=args.max_epochs,
            weight_decay=args.weight_decay,
            eta_min=args.eta_min,
            max_rollout_steps=args.max_rollout_steps,
            rollout_patience=args.rollout_patience,
            noise_std_init=args.noise_std_init,
            noise_decay=args.noise_decay,
            boundary_condition=boundary_condition,
            params=params,
            scalers=scalers,
            output_dir=channel_dir,
            device=args.device,
        )

        seq_std, coords_norm, label_norm, t0_norm, dt_norm = next(iter(train_loader))
        seq_std = seq_std.to(device)
        coords_norm = coords_norm.to(device)
        label_norm = label_norm.to(device)
        t0_norm = t0_norm.to(device)
        dt_norm = dt_norm.to(device)

        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.current_rollout_steps = min(args.max_rollout_steps, seq_std.shape[1] - 1)
        trainer.current_noise_std = 0.0

        B, T, N, _ = seq_std.shape
        k = trainer.current_rollout_steps
        total_params += sum(p.numel() for p in trainer.model.parameters())

        losses.append(trainer._compute_loss((seq_std, coords_norm, label_norm, t0_norm, dt_norm)))
        trainers.append(trainer)

    logger.info(
        f"{hue.y}probe config:{hue.q} "
        f"channels={hue.b}{args.channel_names}{hue.q}, "
        f"batch={hue.m}{B}{hue.q}, frames={hue.m}{T}{hue.q}, "
        f"nodes={hue.m}{N}{hue.q}, max_rollout={hue.m}{k}{hue.q}, "
        f"params={hue.m}{total_params}{hue.q}"
    )

    total_loss = torch.stack(losses).sum()
    total_loss.backward()
    for trainer in trainers:
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
    logger.info(f"{hue.y}peak usage: {hue.m}{peak / 1e9:.2f}{hue.q} GB ({hue.m}{pct:.1f}{hue.q} %) -> {status}")
    log_pipeline("PROBE PIPELINE", "END")


# ============================================================
# Training Pipeline
# ============================================================


def train_pipeline(
    args: argparse.Namespace,
    train_data: FlowData,
    val_data: FlowData,
) -> None:
    """
    Execute the split-channel training workflow.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_data (FlowData): Training split.
        val_data (FlowData): Validation split.
    """
    log_pipeline("TRAIN PIPELINE", "START")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for channel_idx, channel_name in enumerate(args.channel_names):
        channel_dir = output_root / channel_name
        channel_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, scalers, params, boundary_condition = build_channel_data(
            args, train_data, val_data, channel_name, channel_idx)
        model, _ = build_model(model_args=params["model_args"])
        num_params = sum(p.numel() for p in model.parameters())

        logger.info(f"train channel {hue.b}{channel_name}{hue.q} with {hue.m}{num_params}{hue.q} parameters")

        trainer = HyperFlowTrainer(
            model=model,
            lr=args.lr,
            max_epochs=args.max_epochs,
            weight_decay=args.weight_decay,
            eta_min=args.eta_min,
            max_rollout_steps=args.max_rollout_steps,
            rollout_patience=args.rollout_patience,
            noise_std_init=args.noise_std_init,
            noise_decay=args.noise_decay,
            boundary_condition=boundary_condition,
            params=params,
            scalers=scalers,
            output_dir=channel_dir,
            device=args.device,
        )
        trainer.fit(train_loader, val_loader)

    log_pipeline("TRAIN PIPELINE", "END")


# ============================================================
# Inference Pipeline
# ============================================================


def infer_pipeline(
    args: argparse.Namespace,
    test_data: FlowData,
) -> None:
    """
    Execute the label-only split-channel inference workflow.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        test_data (FlowData): Raw test dataset.
    """
    log_pipeline("INFERENCE PIPELINE", "START")
    device = torch.device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    visualizer = FlowVis(output_dir=output_root, spatial_dim=args.spatial_dim, channel_names=args.channel_names)
    metrics_bank = {name: {} for name in args.channel_names}
    channel_bank = {}
    total_params = 0

    for channel_name in args.channel_names:
        channel_dir = output_root / channel_name
        channel_dir.mkdir(parents=True, exist_ok=True)
        model, scalers, params, boundary_condition = load_channel_checkpoint(channel_dir, device)
        channel_bank[channel_name] = (model, scalers, params, boundary_condition)
        total_params += sum(p.numel() for p in model.parameters())

    focus_channel_idx = args.channel_names.index("Vy") if "Vy" in args.channel_names else 1
    focus_bbox_rel = (0.60, 1.00, 0.00, 1.00) if args.spatial_dim == 2 else (0.60, 1.00, 0.00, 1.00, 0.00, 1.00)

    for case_name, seq, coords, label, t0_norm, dt_norm in zip(
        test_data.case_names, test_data.seqs, test_data.coords, test_data.labels, test_data.t0_norm, test_data.dt_norm
    ):
        gt_seq = seq.cpu()
        coords_raw = coords.cpu()
        label_raw = label.cpu()
        init_state = initial_state_from_label(label_raw, coords_raw)
        pred_channels = []
        logs = []

        t0_tensor = torch.tensor([t0_norm], dtype=gt_seq.dtype, device=device)
        dt_tensor = torch.tensor([dt_norm], dtype=gt_seq.dtype, device=device)
        steps = gt_seq.shape[0] - 1

        for channel_idx, channel_name in enumerate(args.channel_names):
            channel_dir = output_root / channel_name
            model, scalers, _, boundary_condition = channel_bank[channel_name]

            init_channel = init_state[:, channel_idx:channel_idx + 1]
            init_channel_std = scalers["state_scaler"].transform(init_channel.unsqueeze(0)).to(device)
            coords_norm = scalers["coord_scaler"].transform(coords_raw.unsqueeze(0)).to(device)
            label_norm = scalers["label_scaler"].transform(label_raw.view(1, -1)).to(device)

            pred_std = model.predict(
                inputs=init_channel_std,
                coords=coords_norm,
                steps=steps,
                t0_norm=t0_tensor,
                dt_norm=dt_tensor,
                boundary_condition=boundary_condition,
                label=label_norm,
            )
            pred_seq = scalers["state_scaler"].inverse_transform(pred_std).cpu().squeeze(0)
            gt_channel = gt_seq[..., channel_idx:channel_idx + 1]
            metrics = Metrics([channel_name]).compute(pred_seq, gt_channel)

            metrics_bank[channel_name][case_name] = metrics
            pred_channels.append(pred_seq)
            torch.save(pred_seq, channel_dir / f"{case_name}_pred.pt")

            acc = metrics[channel_name]["global"]["accuracy"]
            nmse = metrics[channel_name]["global"]["nmse"]
            r2 = metrics[channel_name]["global"]["r2"]
            logs.append(
                f"{hue.c}{channel_name}:{hue.q} "
                f"ACC={hue.m}{acc:.2f}%{hue.q}, "
                f"NMSE={hue.m}{nmse:.2e}{hue.q}, "
                f"R2={hue.m}{r2:.4f}{hue.q}"
            )

        pred_full = torch.cat(pred_channels, dim=-1)
        logger.info(f"case {hue.b}{case_name}{hue.q} | " + " | ".join(logs))

        visualizer.render_full(
            gt=gt_seq,
            pred=pred_full,
            coords=coords_raw,
            case_name=case_name,
            num_nodes=int(coords_raw.shape[0]),
            num_params=total_params,
        )
        visualizer.render_focus(
            gt=gt_seq,
            pred=pred_full,
            coords=coords_raw,
            case_name=case_name,
            num_nodes=int(coords_raw.shape[0]),
            num_params=total_params,
            focus_channel_idx=focus_channel_idx,
            focus_bbox_rel=focus_bbox_rel,
        )

    for channel_name in args.channel_names:
        with open(output_root / channel_name / "metrics.json", "w") as f:
            json.dump(metrics_bank[channel_name], f, indent=4)

    log_pipeline("INFERENCE PIPELINE", "END")


if __name__ == "__main__":
    args = config.get_args()
    seed_everything(args.seed)

    train_data, val_data, test_data = data_pipeline(args)

    if "probe" in args.mode:
        probe_pipeline(args, train_data, val_data)
    if "train" in args.mode:
        train_pipeline(args, train_data, val_data)
    if "infer" in args.mode:
        infer_pipeline(args, test_data)

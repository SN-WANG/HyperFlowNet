# Main script for HyperFlowNet training, inference, and probing
# Author: Shengning Wang

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

import config
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
    Build one joint HyperFlowNet and expose its constructor args.

    Args:
        args (argparse.Namespace | None): Parsed command-line arguments.
        model_args (Dict[str, Any] | None): Explicit model configuration.

    Returns:
        Tuple[HyperFlowNet, Dict[str, Any]]: Model instance and constructor arguments.
    """
    if model_args is None:
        model_args = {
            "in_channels": len(args.channel_names),
            "out_channels": len(args.channel_names),
            "spatial_dim": args.spatial_dim,
            "width": args.width,
            "depth": args.depth,
            "num_slices": args.num_slices,
            "latent_dim": args.latent_dim,
            "num_anchors": args.num_anchors,
            "num_heads": args.num_heads,
            "time_features": args.time_features,
            "freq_base": args.freq_base,
            "attn_dropout": args.attn_dropout,
        }
    return HyperFlowNet(**model_args), model_args


def build_loader(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
) -> DataLoader:
    """
    Build one DataLoader for standardized rollout samples.

    Args:
        samples (list[tuple[Tensor, Tensor, Tensor, Tensor]]): Standardized rollout samples.
        batch_size (int): Mini-batch size.
        shuffle (bool): Whether to shuffle the dataset.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def build_joint_data(
    args: argparse.Namespace,
    train_data: FlowData,
    val_data: FlowData,
) -> Tuple[DataLoader, DataLoader, Dict[str, object], Dict[str, Any]]:
    """
    Build the joint four-channel training runtime.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_data (FlowData): Training split.
        val_data (FlowData): Validation split.

    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, object], Dict[str, Any]]:
            Train loader, validation loader, scalers, and checkpoint params.
    """
    train_states = torch.cat(train_data.seqs, dim=0)
    train_coords = torch.cat(train_data.coords, dim=0)

    state_scaler = StandardScalerTensor().fit(train_states, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)
    scalers = {
        "state_scaler": state_scaler,
        "coord_scaler": coord_scaler,
    }

    _, model_args = build_model(args=args)
    params = {
        "model_args": model_args,
        "channel_names": args.channel_names,
    }

    def _standardize(dataset: FlowData) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        samples = []
        for seq, coord, _, t0_norm, dt_norm in dataset:
            seq_std = state_scaler.transform(seq)
            coords_norm = coord_scaler.transform(coord)
            samples.append((
                seq_std,
                coords_norm,
                torch.tensor(t0_norm, dtype=seq_std.dtype),
                torch.tensor(dt_norm, dtype=seq_std.dtype),
            ))
        return samples

    train_loader = build_loader(_standardize(train_data), args.batch_size, True, args)
    val_loader = build_loader(_standardize(val_data), args.batch_size, False, args)
    return train_loader, val_loader, scalers, params


def load_checkpoint(
    output_dir: Path,
    device: torch.device,
) -> Tuple[HyperFlowNet, Dict[str, object], Dict[str, Any]]:
    """
    Load one joint checkpoint and its inference artifacts.

    Args:
        output_dir (Path): Run directory.
        device (torch.device): Inference device.

    Returns:
        Tuple[HyperFlowNet, Dict[str, object], Dict[str, Any]]:
            Model, scalers, and checkpoint params.
        """
    checkpoint_path = output_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / "ckpt.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    params = checkpoint["params"]
    scaler_state = checkpoint["scaler_state_dict"]

    state_scaler = StandardScalerTensor()
    state_scaler.load_state_dict(scaler_state["state_scaler"])

    coord_scaler = MinMaxScalerTensor(norm_range="bipolar")
    coord_scaler.load_state_dict(scaler_state["coord_scaler"])

    model, _ = build_model(model_args=params["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scalers = {
        "state_scaler": state_scaler,
        "coord_scaler": coord_scaler,
    }
    return model, scalers, params


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
    Run one joint rollout step to estimate peak GPU memory usage.

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

    train_loader, _, scalers, params = build_joint_data(args, train_data, val_data)
    model, _ = build_model(model_args=params["model_args"])
    trainer = HyperFlowTrainer(
        model=model,
        channel_names=args.channel_names,
        lr=args.lr,
        max_epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        max_rollout_steps=args.max_rollout_steps,
        rollout_patience=args.rollout_patience,
        noise_std_init=args.noise_std_init,
        noise_decay=args.noise_decay,
        loss_weight_beta=args.loss_weight_beta,
        loss_weight_alpha=args.loss_weight_alpha,
        loss_weight_min=args.loss_weight_min,
        loss_weight_max=args.loss_weight_max,
        loss_weight_warmup=args.loss_weight_warmup,
        rollout_eval_interval=args.rollout_eval_interval,
        checkpoint_metric=args.checkpoint_metric,
        early_stop_patience=args.early_stop_patience,
        params=params,
        scalers=scalers,
        output_dir=output_root,
        device=args.device,
    )

    seq_std, coords_norm, t0_norm, dt_norm = next(iter(train_loader))
    seq_std = seq_std.to(device)
    coords_norm = coords_norm.to(device)
    t0_norm = t0_norm.to(device)
    dt_norm = dt_norm.to(device)

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.current_rollout_steps = min(args.max_rollout_steps, seq_std.shape[1] - 1)
    trainer.current_noise_std = 0.0

    B, T, N, C = seq_std.shape
    total_params = sum(p.numel() for p in trainer.model.parameters())
    loss = trainer._compute_loss((seq_std, coords_norm, t0_norm, dt_norm))

    logger.info(
        f"{hue.y}probe config:{hue.q} "
        f"channels={hue.b}{args.channel_names}{hue.q}, "
        f"batch={hue.m}{B}{hue.q}, frames={hue.m}{T}{hue.q}, "
        f"nodes={hue.m}{N}{hue.q}, channels_per_node={hue.m}{C}{hue.q}, "
        f"max_rollout={hue.m}{trainer.current_rollout_steps}{hue.q}, "
        f"params={hue.m}{total_params}{hue.q}"
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
    Execute the joint four-channel training workflow.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_data (FlowData): Training split.
        val_data (FlowData): Validation split.
    """
    log_pipeline("TRAIN PIPELINE", "START")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, scalers, params = build_joint_data(args, train_data, val_data)
    model, _ = build_model(model_args=params["model_args"])
    num_params = sum(p.numel() for p in model.parameters())

    logger.info(f"train joint model with {hue.m}{num_params}{hue.q} parameters")

    trainer = HyperFlowTrainer(
        model=model,
        channel_names=args.channel_names,
        lr=args.lr,
        max_epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        max_rollout_steps=args.max_rollout_steps,
        rollout_patience=args.rollout_patience,
        noise_std_init=args.noise_std_init,
        noise_decay=args.noise_decay,
        loss_weight_beta=args.loss_weight_beta,
        loss_weight_alpha=args.loss_weight_alpha,
        loss_weight_min=args.loss_weight_min,
        loss_weight_max=args.loss_weight_max,
        loss_weight_warmup=args.loss_weight_warmup,
        rollout_eval_interval=args.rollout_eval_interval,
        checkpoint_metric=args.checkpoint_metric,
        early_stop_patience=args.early_stop_patience,
        params=params,
        scalers=scalers,
        output_dir=output_root,
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
    Execute the label-only joint inference workflow.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        test_data (FlowData): Raw test dataset.
    """
    log_pipeline("INFERENCE PIPELINE", "START")
    device = torch.device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model, scalers, params = load_checkpoint(output_root, device)
    channel_names = params.get("channel_names", args.channel_names)
    total_params = sum(p.numel() for p in model.parameters())

    visualizer = FlowVis(output_dir=output_root, spatial_dim=args.spatial_dim, channel_names=channel_names)
    metrics = Metrics(channel_names)
    metrics_bank = {}

    focus_channel_idx = channel_names.index("Vy") if "Vy" in channel_names else 1
    focus_bbox_rel = (0.60, 1.00, 0.00, 1.00) if args.spatial_dim == 2 else (0.60, 1.00, 0.00, 1.00, 0.00, 1.00)

    for case_name, seq, coords, label, t0_norm, dt_norm in zip(
        test_data.case_names, test_data.seqs, test_data.coords, test_data.labels, test_data.t0_norm, test_data.dt_norm
    ):
        gt_seq = seq.cpu()
        coords_raw = coords.cpu()
        label_raw = label.cpu()

        init_state = initial_state_from_label(label_raw, coords_raw)
        init_state_std = scalers["state_scaler"].transform(init_state.unsqueeze(0)).to(device)
        coords_norm = scalers["coord_scaler"].transform(coords_raw.unsqueeze(0)).to(device)

        t0_tensor = torch.tensor([t0_norm], dtype=gt_seq.dtype, device=device)
        dt_tensor = torch.tensor([dt_norm], dtype=gt_seq.dtype, device=device)
        pred_std = model.predict(
            inputs=init_state_std,
            coords=coords_norm,
            steps=gt_seq.shape[0] - 1,
            t0_norm=t0_tensor,
            dt_norm=dt_tensor,
        )
        pred_seq = scalers["state_scaler"].inverse_transform(pred_std).cpu().squeeze(0)
        case_metrics = metrics.compute(pred_seq, gt_seq)
        metrics_bank[case_name] = case_metrics

        torch.save(pred_seq, output_root / f"{case_name}_pred.pt")

        logs = []
        for channel_name in channel_names:
            global_metrics = case_metrics[channel_name]["global"]
            logs.append(
                f"{hue.c}{channel_name}:{hue.q} "
                f"ACC={hue.m}{global_metrics['accuracy']:.2f}%{hue.q}, "
                f"NMSE={hue.m}{global_metrics['nmse']:.2e}{hue.q}, "
                f"R2={hue.m}{global_metrics['r2']:.4f}{hue.q}"
            )

        logger.info(f"case {hue.b}{case_name}{hue.q} | " + " | ".join(logs))

        visualizer.render_full(
            gt=gt_seq,
            pred=pred_seq,
            coords=coords_raw,
            case_name=case_name,
            num_nodes=int(coords_raw.shape[0]),
            num_params=total_params,
        )
        visualizer.render_focus(
            gt=gt_seq,
            pred=pred_seq,
            coords=coords_raw,
            case_name=case_name,
            num_nodes=int(coords_raw.shape[0]),
            num_params=total_params,
            focus_channel_idx=focus_channel_idx,
            focus_bbox_rel=focus_bbox_rel,
        )

    with open(output_root / "metrics.json", "w") as f:
        json.dump(metrics_bank, f, indent=4)

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

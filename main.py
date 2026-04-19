# Main script for HyperFlowNet training, inference, and probing
# Author: Shengning Wang

import json
import time
from datetime import datetime, timedelta
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
from training.hflow_trainer import HyperFlowTrainer
from utils.hue_logger import hue, logger
from utils.scaler import MinMaxScalerTensor, StandardScalerTensor
from utils.seeder import seed_everything


def build_model(
    args: Any | None = None,
    model_args: Dict[str, Any] | None = None,
) -> Tuple[HyperFlowNet, Dict[str, Any]]:
    """
    Build HyperFlowNet and return its constructor arguments.

    Args:
        args (Any | None): Parsed command-line arguments.
        model_args (Dict[str, Any] | None): Explicit model arguments.

    Returns:
        Tuple[HyperFlowNet, Dict[str, Any]]: Model instance and model arguments.
    """
    if model_args is None:
        model_args = {
            "in_channels": len(args.channel_names),
            "out_channels": len(args.channel_names),
            "spatial_dim": args.spatial_dim,
            "width": args.width,
            "depth": args.depth,
            "num_slices": args.num_slices,
            "num_heads": args.num_heads,
            "use_spatial_encoding": args.use_spatial_encoding,
            "use_temporal_encoding": args.use_temporal_encoding,
            "coord_features": args.coord_features,
            "time_features": args.time_features,
            "freq_base": args.freq_base,
        }
    return HyperFlowNet(**model_args), model_args


def build_trainer(
    args: Any,
    model: HyperFlowNet,
    scalers: Dict[str, object],
    params: Dict[str, Any],
    output_dir: Path,
) -> HyperFlowTrainer:
    """
    Build the rollout trainer for HyperFlowNet.

    Args:
        args (Any): Parsed command-line arguments.
        model (HyperFlowNet): HyperFlowNet model.
        scalers (Dict[str, object]): Saved scaler objects.
        params (Dict[str, Any]): Checkpoint parameters.
        output_dir (Path): Output directory.

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
        channel_weights=args.channel_weights,
        params=params,
        scalers=scalers,
        output_dir=output_dir,
        device=args.device,
    )


def data_pipeline(args: Any) -> Tuple[DataLoader, DataLoader, FlowData]:
    """
    Build the data runtime for probe, train, and infer.

    Args:
        args (Any): Parsed command-line arguments.

    Returns:
        Tuple[DataLoader, DataLoader, FlowData]: Train loader, validation loader, and raw test data.
    """
    logger.info(f"{hue.c}============================== [DATA PIPELINE] START =============================={hue.q}")

    train_data, val_data, test_data = FlowData.spawn(
        data_dir=args.data_dir,
        spatial_dim=args.spatial_dim,
        win_len=args.win_len,
        win_stride=args.win_stride,
    )

    train_seqs = torch.cat(train_data.seqs, dim=0)
    train_coords = torch.cat(train_data.coords, dim=0)
    args.state_scaler = StandardScalerTensor().fit(train_seqs, channel_dim=-1)
    args.coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)

    train_dataset = [
        (
            args.state_scaler.transform(seq),
            args.coord_scaler.transform(coords),
            torch.tensor(t0_norm, dtype=seq.dtype),
            torch.tensor(dt_norm, dtype=seq.dtype),
        )
        for seq, coords, _, t0_norm, dt_norm in zip(
            train_data.seqs,
            train_data.coords,
            train_data.labels,
            train_data.t0_norm,
            train_data.dt_norm,
        )
    ]
    val_dataset = [
        (
            args.state_scaler.transform(seq),
            args.coord_scaler.transform(coords),
            torch.tensor(t0_norm, dtype=seq.dtype),
            torch.tensor(dt_norm, dtype=seq.dtype),
        )
        for seq, coords, _, t0_norm, dt_norm in zip(
            val_data.seqs,
            val_data.coords,
            val_data.labels,
            val_data.t0_norm,
            val_data.dt_norm,
        )
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"{hue.g}=============================== [DATA PIPELINE] END ==============================={hue.q}")
    return train_loader, val_loader, test_data


def probe_pipeline(args: Any, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Execute one rollout step for peak GPU memory probing.

    Args:
        args (Any): Parsed command-line arguments.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
    """
    logger.info(f"{hue.c}============================= [PROBE PIPELINE] START =============================={hue.q}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning("No CUDA device - probe skipped.")
        logger.info(f"{hue.g}============================== [PROBE PIPELINE] END ==============================={hue.q}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model, model_args = build_model(args=args)
    params = {"model_args": model_args, "channel_names": args.channel_names}
    scalers = {"state_scaler": args.state_scaler, "coord_scaler": args.coord_scaler}
    trainer = build_trainer(args, model, scalers, params, output_dir)

    seq_std, coords_norm, t0_norm, dt_norm = next(iter(train_loader))
    seq_std = seq_std.to(device)
    coords_norm = coords_norm.to(device)
    t0_norm = t0_norm.to(device)
    dt_norm = dt_norm.to(device)
    val_seq_std, val_coords_norm, val_t0_norm, val_dt_norm = next(iter(val_loader))
    val_seq_std = val_seq_std.to(device)
    val_coords_norm = val_coords_norm.to(device)
    val_t0_norm = val_t0_norm.to(device)
    val_dt_norm = val_dt_norm.to(device)

    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.current_rollout_steps = min(args.max_rollout_steps, seq_std.shape[1] - 1)
    trainer.current_noise_std = 0.0

    B, T, N, C = seq_std.shape
    total_params = sum(p.numel() for p in trainer.model.parameters())
    train_steps = len(train_loader)
    torch.cuda.synchronize(device)
    probe_start = time.perf_counter()
    loss = trainer._compute_loss((seq_std, coords_norm, t0_norm, dt_norm))
    loss.backward()
    trainer.optimizer.step()
    torch.cuda.synchronize(device)
    train_batch_time = time.perf_counter() - probe_start

    val_steps = len(val_loader)
    trainer.model.eval()
    torch.cuda.synchronize(device)
    probe_start = time.perf_counter()
    with torch.no_grad():
        trainer._compute_loss((val_seq_std, val_coords_norm, val_t0_norm, val_dt_norm))
    torch.cuda.synchronize(device)
    val_batch_time = time.perf_counter() - probe_start

    rollout_sum = sum(min(1 + epoch // args.rollout_patience, trainer.current_rollout_steps) for epoch in range(args.max_epochs))
    total_seconds = (train_batch_time * train_steps + val_batch_time * val_steps) * rollout_sum / trainer.current_rollout_steps
    finish_at = datetime.now().astimezone() + timedelta(seconds=total_seconds)

    logger.info(
        f"{hue.y}probe config:{hue.q} "
        f"batch={hue.m}{B}{hue.q}, frames={hue.m}{T}{hue.q}, "
        f"nodes={hue.m}{N}{hue.q}, channels={hue.m}{C}{hue.q}, "
        f"max_rollout={hue.m}{trainer.current_rollout_steps}{hue.q}, "
        f"params={hue.m}{total_params}{hue.q}"
    )

    peak = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    pct = 100.0 * peak / total
    if pct < 75.0:
        status = f"{hue.g}SAFE{hue.q}"
    elif pct < 92.0:
        status = f"{hue.y}WARNING - close to limit{hue.q}"
    else:
        status = f"{hue.r}CRITICAL - likely OOM in real training{hue.q}"

    logger.info(f"{hue.y}device: {hue.b}{torch.cuda.get_device_name(device)}{hue.q} ({hue.m}{total / 1e9:.1f}{hue.q} GB)")
    logger.info(f"{hue.y}peak usage: {hue.m}{peak / 1e9:.2f}{hue.q} GB ({hue.m}{pct:.1f}{hue.q} %) -> {status}")
    logger.info(f"{hue.y}train eta: {hue.m}{total_seconds / 3600.0:.1f}{hue.q} h -> {hue.b}{finish_at.strftime('%m-%d %H:%M')}{hue.q}")
    logger.info(f"{hue.g}============================== [PROBE PIPELINE] END ==============================={hue.q}")


def train_pipeline(args: Any, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Execute the training workflow.

    Args:
        args (Any): Parsed command-line arguments.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
    """
    logger.info(f"{hue.c}============================= [TRAIN PIPELINE] START =============================={hue.q}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_args = build_model(args=args)
    params = {"model_args": model_args, "channel_names": args.channel_names}
    scalers = {"state_scaler": args.state_scaler, "coord_scaler": args.coord_scaler}
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"train joint model with {hue.m}{total_params}{hue.q} parameters")

    trainer = build_trainer(args, model, scalers, params, output_dir)
    trainer.fit(train_loader, val_loader)

    logger.info(f"{hue.g}============================== [TRAIN PIPELINE] END ==============================={hue.q}")


def infer_pipeline(args: Any, test_data: FlowData) -> None:
    """
    Execute the inference workflow.

    Args:
        args (Any): Parsed command-line arguments.
        test_data (FlowData): Raw test dataset.
    """
    logger.info(f"{hue.c}=========================== [INFERENCE PIPELINE] START ============================ {hue.q}")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    channel_names = params["channel_names"]
    total_params = sum(p.numel() for p in model.parameters())
    visualizer = FlowVis(output_dir=output_dir, spatial_dim=args.spatial_dim, channel_names=channel_names)
    metrics = Metrics(channel_names)
    metrics_bank = {}

    focus_channel_idx = channel_names.index("Vy") if "Vy" in channel_names else 1
    focus_bbox_rel = (0.60, 1.00, 0.00, 1.00) if args.spatial_dim == 2 else (0.60, 1.00, 0.00, 1.00, 0.00, 1.00)

    for seq, coords, label in zip(
        test_data.seqs,
        test_data.coords,
        test_data.labels,
    ):
        gt_seq = seq.cpu()
        coords_raw = coords.cpu()
        label_raw = label.cpu()
        label_name = str(int(label_raw.reshape(-1)[0].item()))

        init_state = initial_state_from_label(label_raw, coords_raw)
        init_state_std = state_scaler.transform(init_state.unsqueeze(0)).to(device)
        coords_norm = coord_scaler.transform(coords_raw.unsqueeze(0)).to(device)

        pred_std = model.predict(
            inputs=init_state_std,
            coords=coords_norm,
            steps=gt_seq.shape[0] - 1,
        )
        pred_seq = state_scaler.inverse_transform(pred_std).cpu().squeeze(0)
        case_metrics = metrics.compute(pred_seq, gt_seq)
        metrics_bank[label_name] = case_metrics

        torch.save(pred_seq, output_dir / f"{label_name}_pred.pt")

        logs = []
        for channel_name in channel_names:
            global_metrics = case_metrics[channel_name]["global"]
            logs.append(
                f"{hue.c}{channel_name}:{hue.q} "
                f"ACC={hue.m}{global_metrics['accuracy']:.2f}%{hue.q}, "
                f"NMSE={hue.m}{global_metrics['nmse']:.2e}{hue.q}, "
                f"R2={hue.m}{global_metrics['r2']:.4f}{hue.q}"
            )

        logger.info(f"label {hue.b}{label_name}{hue.q} | " + " | ".join(logs))

        visualizer.render_full(
            gt=gt_seq,
            pred=pred_seq,
            coords=coords_raw,
            label=label_name,
            num_nodes=int(coords_raw.shape[0]),
            num_params=total_params,
        )
        visualizer.render_focus(
            gt=gt_seq,
            pred=pred_seq,
            coords=coords_raw,
            label=label_name,
            num_nodes=int(coords_raw.shape[0]),
            num_params=total_params,
            focus_channel_idx=focus_channel_idx,
            focus_bbox_rel=focus_bbox_rel,
        )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_bank, f, indent=4)

    logger.info(f"{hue.g}============================ [INFERENCE PIPELINE] END ============================={hue.q}")


if __name__ == "__main__":
    args = config.get_args()
    seed_everything(args.seed)

    train_loader, val_loader, test_data = data_pipeline(args)

    if "probe" in args.mode:
        probe_pipeline(args, train_loader, val_loader)
    if "train" in args.mode:
        train_pipeline(args, train_loader, val_loader)
    if "infer" in args.mode:
        infer_pipeline(args, test_data)

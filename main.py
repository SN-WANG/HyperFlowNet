# Main script for HyperFlowNet flow simulation workflows
# Author: Shengning Wang

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

import config
from data.boundary import BoundaryCondition
from data.flow_data import FlowData
from data.flow_metrics import Metrics
from data.flow_vis import FlowVis
from data.flow_twin import FlowTwin
from data.initial_state import initial_state_from_label
from models.gcn import GCN
from models.geofno import GeoFNO
from models.gino import GINO
from models.gnot import GNOT
from models.hflownet import HyperFlowNet, build_local_graph
from models.meshgraphnet import MeshGraphNet
from models.transolver import Transolver
from training.hflow_trainer import HyperFlowTrainer
from utils.hue_logger import hue, logger
from utils.scaler import MinMaxScalerTensor, StandardScalerTensor
from utils.seeder import seed_everything


def build_model(
    args: Any | None = None,
    model_args: Dict[str, Any] | None = None,
    adj_indices: Tensor | None = None,
    adj_values: Tensor | None = None,
    edge_index: Tensor | None = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build the configured flow model and return its constructor arguments.

    Args:
        args (Any | None): Parsed arguments.
        model_args (Dict[str, Any] | None): Explicit model arguments.
        adj_indices (Tensor | None): Sparse adjacency indices. (2, E).
        adj_values (Tensor | None): Sparse adjacency values. (E,).
        edge_index (Tensor | None): Undirected edge list. (2, E_EDGE).

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: Model instance and model argument dict.
    """
    if model_args is None:
        model_args = {
            "model_name": args.model_name,
            "in_channels": len(args.channel_names),
            "out_channels": len(args.channel_names),
            "spatial_dim": args.spatial_dim,
            "width": args.width,
            "depth": args.depth,
            "dropout": args.dropout,
            "graph_mode": args.graph_mode,
            "num_slices": args.num_slices,
            "num_heads": args.num_heads,
            "coord_features": args.coord_features,
            "time_features": args.time_features,
            "freq_base": args.freq_base,
            "graph_beta_init": args.graph_beta_init,
            "graph_bias_eps": args.graph_bias_eps,
            "num_experts": args.num_experts,
            "geofno_modes": args.geofno_modes,
            "geofno_grid_size": args.geofno_grid_size,
            "gino_modes": args.gino_modes,
            "gino_grid_size": args.gino_grid_size,
            "gino_neighbors": args.gino_neighbors,
        }

    model_name = model_args["model_name"].lower()
    if model_name == "hflownet":
        model = HyperFlowNet(
            in_channels=model_args["in_channels"],
            out_channels=model_args["out_channels"],
            spatial_dim=model_args["spatial_dim"],
            adj_indices=adj_indices,
            adj_values=adj_values,
            edge_index=edge_index,
            graph_mode=model_args["graph_mode"],
            width=model_args["width"],
            depth=model_args["depth"],
            num_slices=model_args["num_slices"],
            num_heads=model_args["num_heads"],
            coord_features=model_args["coord_features"],
            time_features=model_args["time_features"],
            freq_base=model_args["freq_base"],
            graph_beta_init=model_args["graph_beta_init"],
            graph_bias_eps=model_args["graph_bias_eps"],
        )
    elif model_name == "transolver":
        model = Transolver(
            in_channels=model_args["in_channels"],
            out_channels=model_args["out_channels"],
            spatial_dim=model_args["spatial_dim"],
            width=model_args["width"],
            depth=model_args["depth"],
            num_slices=model_args["num_slices"],
            num_heads=model_args["num_heads"],
            dropout=model_args["dropout"],
        )
    elif model_name == "gnot":
        model = GNOT(
            in_channels=model_args["in_channels"],
            out_channels=model_args["out_channels"],
            spatial_dim=model_args["spatial_dim"],
            width=model_args["width"],
            depth=model_args["depth"],
            num_heads=model_args["num_heads"],
            num_experts=model_args["num_experts"],
            dropout=model_args["dropout"],
        )
    elif model_name == "geofno":
        model = GeoFNO(
            in_channels=model_args["in_channels"],
            out_channels=model_args["out_channels"],
            spatial_dim=model_args["spatial_dim"],
            width=model_args["width"],
            depth=model_args["depth"],
            modes=model_args["geofno_modes"],
            grid_size=model_args["geofno_grid_size"],
        )
    elif model_name == "gino":
        model = GINO(
            in_channels=model_args["in_channels"],
            out_channels=model_args["out_channels"],
            spatial_dim=model_args["spatial_dim"],
            width=model_args["width"],
            depth=model_args["depth"],
            modes=model_args["gino_modes"],
            grid_size=model_args["gino_grid_size"],
            neighbors=model_args["gino_neighbors"],
        )
    elif model_name == "gcn":
        model = GCN(
            in_channels=model_args["in_channels"],
            out_channels=model_args["out_channels"],
            spatial_dim=model_args["spatial_dim"],
            adj_indices=adj_indices,
            adj_values=adj_values,
            width=model_args["width"],
            depth=model_args["depth"],
            dropout=model_args["dropout"],
        )
    elif model_name == "meshgraphnet":
        model = MeshGraphNet(
            in_channels=model_args["in_channels"],
            out_channels=model_args["out_channels"],
            spatial_dim=model_args["spatial_dim"],
            edge_index=edge_index,
            width=model_args["width"],
            depth=model_args["depth"],
        )
    else:
        raise ValueError(f"unknown model_name: {model_name}")
    return model, model_args


def build_trainer(
    args: Any,
    model: nn.Module,
    params: Dict[str, Any],
    scalers: Dict[str, object],
    output_dir: Path,
) -> HyperFlowTrainer:
    """
    Build the rollout trainer.

    Args:
        args (Any): Parsed arguments.
        model (nn.Module): Flow model.
        params (Dict[str, Any]): Checkpoint parameters.
        scalers (Dict[str, object]): Fitted scalers.
        output_dir (Path): Artifact directory.

    Returns:
        HyperFlowTrainer: Configured trainer.
    """
    return HyperFlowTrainer(
        model=model,
        params=params,
        scalers=scalers,
        output_dir=output_dir,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        eta_min=args.eta_min,
        max_rollout_steps=args.max_rollout_steps,
        rollout_patience=args.rollout_patience,
        noise_std_init=args.noise_std_init,
        noise_decay=args.noise_decay,
        channel_weights=args.channel_weights,
        bc=getattr(args, "bc", None),
    )


def data_pipeline(args: Any) -> Tuple[DataLoader, DataLoader, FlowData]:
    """
    Build datasets, scalers, and loaders for the current run.

    Args:
        args (Any): Parsed arguments.

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

    train_states = torch.cat(train_data.seqs, dim=0)
    train_coords = torch.cat(train_data.coords, dim=0)
    args.state_scaler = StandardScalerTensor().fit(train_states, channel_dim=-1)
    args.coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)
    args.bc = None
    if args.use_bc:
        args.bc = BoundaryCondition().fit(
            train_data,
            args.state_scaler,
            velocity_channels=list(range(args.spatial_dim)),
            velocity_threshold=args.bc_threshold,
        )

    train_dataset = [
        (
            args.state_scaler.transform(seq),
            args.coord_scaler.transform(coords),
            torch.tensor(t0_norm, dtype=seq.dtype),
            torch.tensor(dt_norm, dtype=seq.dtype),
        )
        for seq, coords, t0_norm, dt_norm in zip(train_data.seqs, train_data.coords, train_data.t0_norm, train_data.dt_norm)
    ]
    val_dataset = [
        (
            args.state_scaler.transform(seq),
            args.coord_scaler.transform(coords),
            torch.tensor(t0_norm, dtype=seq.dtype),
            torch.tensor(dt_norm, dtype=seq.dtype),
        )
        for seq, coords, t0_norm, dt_norm in zip(val_data.seqs, val_data.coords, val_data.t0_norm, val_data.dt_norm)
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
    Probe peak GPU memory and projected training time.

    Args:
        args (Any): Parsed arguments.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
    """
    logger.info(f"{hue.c}============================= [PROBE PIPELINE] START =============================={hue.q}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, probe skipped.")
        logger.info(f"{hue.g}============================== [PROBE PIPELINE] END ==============================={hue.q}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_coords = train_loader.dataset[0][1]
    adj_indices, adj_values, edge_index = build_local_graph(
        coords=ref_coords,
        k=args.graph_k,
        sigma_scale=args.graph_sigma_scale,
    )
    model, model_args = build_model(
        args=args,
        adj_indices=adj_indices,
        adj_values=adj_values,
        edge_index=edge_index,
    )
    model_name = model_args["model_name"]
    params = {
        "channel_names": args.channel_names,
        "model_args": model_args,
        "graph_k": args.graph_k,
        "graph_sigma_scale": args.graph_sigma_scale,
        "bc": args.bc.state_dict() if args.bc is not None else None,
    }
    scalers = {
        "state_scaler": args.state_scaler,
        "coord_scaler": args.coord_scaler,
    }
    trainer = build_trainer(args, model, params, scalers, output_dir)

    train_batch = tuple(t.to(device) for t in next(iter(train_loader)))
    val_batch = tuple(t.to(device) for t in next(iter(val_loader)))

    reachable_rollout = min(args.max_rollout_steps, train_batch[0].shape[1] - 1)
    if args.rollout_patience > 0:
        scheduled_rollout = 1 + max(args.max_epochs - 1, 0) // args.rollout_patience
        reachable_rollout = min(reachable_rollout, scheduled_rollout)
    sample_steps = sorted({1, max(1, reachable_rollout // 2), reachable_rollout})

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    train_times = []
    for rollout_steps in sample_steps:
        trainer.current_rollout_steps = rollout_steps
        trainer.current_noise_std = trainer.noise_std_init
        trainer.model.train()

        durations = []
        for repeat_idx in range(3):
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            loss = trainer._compute_loss(train_batch)
            loss.backward()
            trainer.optimizer.step()
            torch.cuda.synchronize(device)
            if repeat_idx > 0:
                durations.append(time.perf_counter() - start)
        train_times.append(sum(durations) / len(durations))

    val_times = []
    for rollout_steps in sample_steps:
        trainer.current_rollout_steps = rollout_steps
        trainer.current_noise_std = 0.0
        trainer.model.eval()

        durations = []
        with torch.no_grad():
            for repeat_idx in range(3):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                trainer._compute_loss(val_batch)
                torch.cuda.synchronize(device)
                if repeat_idx > 0:
                    durations.append(time.perf_counter() - start)
        val_times.append(sum(durations) / len(durations))

    if len(sample_steps) == 1:
        train_bias, train_slope = 0.0, train_times[0]
        val_bias, val_slope = 0.0, val_times[0]
    else:
        x = torch.tensor(sample_steps, dtype=torch.float64)
        x_centered = x - x.mean()
        denom = torch.dot(x_centered, x_centered).item()

        y = torch.tensor(train_times, dtype=torch.float64)
        train_slope = float(torch.dot(x_centered, y - y.mean()).item() / denom)
        train_bias = float((y.mean() - train_slope * x.mean()).item())

        y = torch.tensor(val_times, dtype=torch.float64)
        val_slope = float(torch.dot(x_centered, y - y.mean()).item() / denom)
        val_bias = float((y.mean() - val_slope * x.mean()).item())

    rollout_schedule = []
    for epoch_idx in range(args.max_epochs):
        if args.rollout_patience <= 0:
            rollout_schedule.append(reachable_rollout)
        else:
            step = 1 + epoch_idx // args.rollout_patience
            rollout_schedule.append(min(step, reachable_rollout))

    epoch_seconds = []
    for rollout_steps in rollout_schedule:
        train_epoch = len(train_loader) * (max(train_bias, 0.0) + max(train_slope, 0.0) * rollout_steps)
        val_epoch = len(val_loader) * (max(val_bias, 0.0) + max(val_slope, 0.0) * rollout_steps)
        epoch_seconds.append(train_epoch + val_epoch)

    total_seconds = float(sum(epoch_seconds))
    finish_at = datetime.now().astimezone() + timedelta(seconds=total_seconds)

    trainer.current_rollout_steps = reachable_rollout
    trainer.current_noise_std = 0.0
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    loss = trainer._compute_loss(train_batch)
    loss.backward()
    trainer.optimizer.step()
    torch.cuda.synchronize(device)

    B, T, N, C = train_batch[0].shape
    total_params = sum(p.numel() for p in trainer.model.parameters())
    peak = torch.cuda.max_memory_allocated(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    pct = 100.0 * peak / total_mem

    if pct < 75.0:
        status = f"{hue.g}SAFE{hue.q}"
    elif pct < 92.0:
        status = f"{hue.y}WARNING - close to limit{hue.q}"
    else:
        status = f"{hue.r}CRITICAL - likely OOM in real training{hue.q}"

    logger.info(
        f"{hue.y}probe config:{hue.q} "
        f"model={hue.b}{model_name}{hue.q}, class={hue.b}{model.__class__.__name__}{hue.q}, "
        f"batch={hue.m}{B}{hue.q}, frames={hue.m}{T}{hue.q}, "
        f"nodes={hue.m}{N}{hue.q}, channels={hue.m}{C}{hue.q}, "
        f"rollout={hue.m}{reachable_rollout}{hue.q}/{hue.m}{min(args.max_rollout_steps, T - 1)}{hue.q}, "
        f"params={hue.m}{total_params}{hue.q}"
    )
    logger.info(
        f"{hue.y}device:{hue.q} {hue.b}{torch.cuda.get_device_name(device)}{hue.q} "
        f"({hue.m}{total_mem / 1e9:.1f}{hue.q} GB)"
    )
    logger.info(
        f"{hue.y}peak usage:{hue.q} {hue.m}{peak / 1e9:.2f}{hue.q} GB "
        f"({hue.m}{pct:.1f}{hue.q} %) -> {status}"
    )
    logger.info(
        f"{hue.y}train eta:{hue.q} {hue.m}{total_seconds / 3600.0:.1f}{hue.q} h "
        f"-> {hue.b}{finish_at.strftime('%m-%d %H:%M')}{hue.q}"
    )
    logger.info(f"{hue.g}============================== [PROBE PIPELINE] END ==============================={hue.q}")


def train_pipeline(args: Any, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Run the training workflow.

    Args:
        args (Any): Parsed arguments.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
    """
    logger.info(f"{hue.c}============================= [TRAIN PIPELINE] START =============================={hue.q}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_coords = train_loader.dataset[0][1]
    adj_indices, adj_values, edge_index = build_local_graph(
        coords=ref_coords,
        k=args.graph_k,
        sigma_scale=args.graph_sigma_scale,
    )
    model, model_args = build_model(
        args=args,
        adj_indices=adj_indices,
        adj_values=adj_values,
        edge_index=edge_index,
    )
    params = {
        "channel_names": args.channel_names,
        "model_args": model_args,
        "graph_k": args.graph_k,
        "graph_sigma_scale": args.graph_sigma_scale,
        "bc": args.bc.state_dict() if args.bc is not None else None,
    }
    scalers = {
        "state_scaler": args.state_scaler,
        "coord_scaler": args.coord_scaler,
    }

    logger.info(f"train model with {hue.m}{sum(p.numel() for p in model.parameters())}{hue.q} parameters")
    trainer = build_trainer(args, model, params, scalers, output_dir)
    trainer.fit(train_loader, val_loader)

    logger.info(f"{hue.g}============================== [TRAIN PIPELINE] END ==============================={hue.q}")


def infer_pipeline(args: Any, test_data: FlowData) -> None:
    """
    Run the inference workflow.

    Args:
        args (Any): Parsed arguments.
        test_data (FlowData): Raw test dataset.
    """
    logger.info(f"{hue.c}=========================== [INFERENCE PIPELINE] START ============================ {hue.q}")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(output_dir / "ckpt.pt", map_location=device, weights_only=True)
    params = checkpoint["params"]
    scaler_state = checkpoint["scaler_state_dict"]
    bc = None
    if params.get("bc") is not None:
        bc = BoundaryCondition()
        bc.load_state_dict(params["bc"])

    state_scaler = StandardScalerTensor()
    state_scaler.load_state_dict(scaler_state["state_scaler"])
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar")
    coord_scaler.load_state_dict(scaler_state["coord_scaler"])

    ref_coords = coord_scaler.transform(test_data.coords[0])
    adj_indices, adj_values, edge_index = build_local_graph(
        coords=ref_coords,
        k=params["graph_k"],
        sigma_scale=params["graph_sigma_scale"],
    )
    model, _ = build_model(
        model_args=params["model_args"],
        adj_indices=adj_indices,
        adj_values=adj_values,
        edge_index=edge_index,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    channel_names = params["channel_names"]
    total_params = sum(p.numel() for p in model.parameters())
    visualizer = FlowVis(output_dir=output_dir, spatial_dim=args.spatial_dim, channel_names=channel_names)
    flow_twin = FlowTwin(output_dir=output_dir, channel_names=channel_names)
    metrics = Metrics(channel_names)
    metrics_bank = {}

    focus_channel_idx = channel_names.index("Vy")
    focus_bbox_rel = (0.60, 1.00, 0.00, 1.00) if args.spatial_dim == 2 else (0.60, 1.00, 0.00, 1.00, 0.00, 1.00)

    for seq, coords, label in zip(test_data.seqs, test_data.coords, test_data.labels):
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
            bc=bc,
        )
        pred_seq = state_scaler.inverse_transform(pred_std).squeeze(0).cpu()
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
        flow_twin.render(
            pred=pred_seq,
            coords=coords_raw,
            label=label_name,
            num_nodes=int(coords_raw.shape[0]),
            num_params=total_params,
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

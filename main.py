# Main script for HyperFlowNet flow simulation workflows
# Author: Shengning Wang

import argparse
import ast
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.boundary import BoundaryCondition
from data.flow_data import FlowData
from data.flow_metrics import Metrics
from data.flow_twin import FlowTwin
from data.flow_vis import FlowVis
from data.initial_state import initial_state_from_label
from models.gcn import GCN
from models.geofno import GeoFNO
from models.gino import GINO
from models.hflownet import HyperFlowNet
from models.transolver import Transolver
from training.hflow_trainer import HyperFlowTrainer
from utils.hue_logger import hue, logger
from utils.scaler import MinMaxScalerTensor, StandardScalerTensor
from utils.seeder import seed_everything


CONFIG_PATH = Path(__file__).with_name("config.yaml")
MODEL_NAMES = {"hflownet", "transolver", "geofno", "gcn", "gino"}
ABLATIONS = {"none", "time_encoding", "spatial_encoding", "noise", "rollout", "bc", "bias", "gating", "loss"}


# ============================================================
# Configuration
# ============================================================


def _strip_comment(line: str) -> str:
    quote = None
    for idx, char in enumerate(line):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "#" and quote is None:
            return line[:idx]
    return line


def _parse_config_value(raw: str) -> Any:
    raw = raw.strip()
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        return ast.literal_eval(raw)
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return ast.literal_eval(raw)
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _load_config_file(path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line in path.read_text().splitlines():
        clean = _strip_comment(line).rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        key, sep, raw_value = clean.strip().partition(":")
        if sep == "":
            continue

        while indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if raw_value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_config_value(raw_value)

    return root


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(_flatten_config(value))
        else:
            flat[key] = value
    return flat


def _set_config_value(config: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _parse_cli() -> tuple[argparse.Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description="HyperFlowNet: A Spatio-Temporal Neural Operator for Flow Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to a YAML experiment config.")
    parser.add_argument("--set", action="append", default=[], help="Override a config entry, e.g. model.width=256.")
    parser.add_argument("--mode", type=str, nargs="+", help="Execution phases to run.")
    parser.add_argument("--output_dir", type=str, help="Directory to save checkpoints and outputs.")
    parser.add_argument("--model_name", type=str, choices=sorted(MODEL_NAMES), help="Model architecture.")
    parser.add_argument("--ablation", type=str, choices=sorted(ABLATIONS), help="Ablation preset.")

    parsed, unknown = parser.parse_known_args()
    overrides: Dict[str, Any] = {}
    idx = 0
    while idx < len(unknown):
        token = unknown[idx]
        if token.startswith("--no-"):
            overrides[token[5:].replace("-", "_")] = False
            idx += 1
        elif token.startswith("--"):
            key = token[2:].replace("-", "_")
            if idx + 1 < len(unknown) and not unknown[idx + 1].startswith("--"):
                overrides[key] = _parse_config_value(unknown[idx + 1])
                idx += 2
            else:
                overrides[key] = True
                idx += 1
        else:
            idx += 1
    return parsed, overrides


def _expand_spatial_list(values: list[int], spatial_dim: int) -> list[int]:
    if len(values) == spatial_dim:
        return values
    if len(values) == 1:
        return values * spatial_dim
    return values[:spatial_dim]


def _apply_ablation(args: Namespace) -> None:
    if args.ablation == "time_encoding":
        args.use_time_encoding = False
    elif args.ablation == "spatial_encoding":
        args.use_spatial_encoding = False
    elif args.ablation == "noise":
        args.use_noise = False
    elif args.ablation == "rollout":
        args.use_rollout = False
    elif args.ablation == "bc":
        args.use_bc = False
    elif args.ablation == "bias":
        args.use_bias = False
    elif args.ablation == "gating":
        args.use_gating = False
    elif args.ablation == "loss":
        args.use_weighted_loss = False


def _finalize_config(args: Namespace) -> Namespace:
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.model_name = args.model_name.lower()
    if args.model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model_name: {args.model_name}")
    if args.ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {args.ablation}")

    _apply_ablation(args)

    if not args.use_spatial_encoding:
        args.coord_features = 0
    if not args.use_time_encoding:
        args.time_features = 0
        args.lag_features = 0
    if not args.use_noise:
        args.noise_std_init = 0.0
        args.noise_decay = 0.0
    if not args.use_rollout:
        args.max_rollout_steps = 1
        args.max_history_steps = 1
    if not args.use_weighted_loss:
        args.use_causal_weighting = False

    args.geofno_modes = _expand_spatial_list(args.geofno_modes, args.spatial_dim)
    args.geofno_grid_size = _expand_spatial_list(args.geofno_grid_size, args.spatial_dim)
    args.gino_modes = _expand_spatial_list(args.gino_modes, args.spatial_dim)
    args.gino_grid_size = _expand_spatial_list(args.gino_grid_size, args.spatial_dim)
    return args


def get_args() -> Namespace:
    """
    Load YAML config and command-line overrides.

    Returns:
        Namespace: Resolved experiment arguments.
    """
    parsed, unknown_overrides = _parse_cli()
    config_path = Path(parsed.config)
    config = _load_config_file(config_path)

    for assignment in parsed.set:
        key, value = assignment.split("=", 1)
        _set_config_value(config, key, _parse_config_value(value))

    flat = _flatten_config(config)
    flat.update(unknown_overrides)
    for key in ("mode", "output_dir", "model_name", "ablation"):
        value = getattr(parsed, key)
        if value is not None:
            flat[key] = value

    args = Namespace(**flat)
    args.config = str(config_path)
    return _finalize_config(args)


# ============================================================
# Builders
# ============================================================


def build_model(
    args: Any | None = None,
    model_args: Dict[str, Any] | None = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build the configured flow model and return its constructor arguments.

    Args:
        args (Any | None): Parsed arguments.
        model_args (Dict[str, Any] | None): Explicit model arguments.

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
            "num_slices": args.num_slices,
            "num_heads": args.num_heads,
            "coord_features": args.coord_features,
            "time_features": args.time_features,
            "lag_features": args.lag_features,
            "freq_base": args.freq_base,
            "use_spatial_encoding": args.use_spatial_encoding,
            "use_time_encoding": args.use_time_encoding,
            "use_bias": args.use_bias,
            "use_gating": args.use_gating,
            "bias_beta_init": args.bias_beta_init,
            "gate_beta_init": args.gate_beta_init,
            "space_tau_init": args.space_tau_init,
            "transolver_mlp_ratio": args.transolver_mlp_ratio,
            "transolver_use_time_input": args.transolver_use_time_input,
            "transolver_unified_pos": args.transolver_unified_pos,
            "transolver_ref": args.transolver_ref,
            "graph_k": args.graph_k,
            "graph_sigma_scale": args.graph_sigma_scale,
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
            width=model_args["width"],
            depth=model_args["depth"],
            num_slices=model_args["num_slices"],
            num_heads=model_args["num_heads"],
            coord_features=model_args["coord_features"],
            time_features=model_args["time_features"],
            lag_features=model_args["lag_features"],
            freq_base=model_args["freq_base"],
            use_spatial_encoding=model_args["use_spatial_encoding"],
            use_time_encoding=model_args["use_time_encoding"],
            use_bias=model_args["use_bias"],
            use_gating=model_args["use_gating"],
            bias_beta_init=model_args["bias_beta_init"],
            gate_beta_init=model_args["gate_beta_init"],
            space_tau_init=model_args["space_tau_init"],
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
            mlp_ratio=model_args["transolver_mlp_ratio"],
            use_time_input=model_args["transolver_use_time_input"],
            unified_pos=model_args["transolver_unified_pos"],
            ref=model_args["transolver_ref"],
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
            graph_k=model_args["graph_k"],
            graph_sigma_scale=model_args["graph_sigma_scale"],
            width=model_args["width"],
            depth=model_args["depth"],
            dropout=model_args["dropout"],
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
        max_history_steps=args.max_history_steps,
        history_length_alpha=args.history_length_alpha,
        history_sigma_min=args.history_sigma_min,
        history_sigma_max=args.history_sigma_max,
        history_sigma_alpha=args.history_sigma_alpha,
        use_weighted_loss=args.use_weighted_loss,
        use_causal_weighting=args.use_causal_weighting,
        causal_weight_eps=args.causal_weight_eps,
        channel_weights=args.channel_weights,
        bc=getattr(args, "bc", None),
    )


# ============================================================
# Data And Workflows
# ============================================================


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
        for seq, coords, t0_norm, dt_norm in zip(
            train_data.seqs, train_data.coords, train_data.t0_norm, train_data.dt_norm
        )
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


def _checkpoint_params(args: Any, model_args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "channel_names": args.channel_names,
        "model_args": model_args,
        "graph_k": args.graph_k,
        "graph_sigma_scale": args.graph_sigma_scale,
        "bc": args.bc.state_dict() if args.bc is not None else None,
    }


def probe_pipeline(args: Any, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Run a lightweight single-batch compute and memory probe.

    Args:
        args (Any): Parsed arguments.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
    """
    logger.info(f"{hue.c}============================= [PROBE PIPELINE] START =============================={hue.q}")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_args = build_model(args=args)
    scalers = {"state_scaler": args.state_scaler, "coord_scaler": args.coord_scaler}
    trainer = build_trainer(args, model, _checkpoint_params(args, model_args), scalers, output_dir)

    train_batch = tuple(t.to(device) for t in next(iter(train_loader)))
    val_batch = tuple(t.to(device) for t in next(iter(val_loader)))
    reachable_rollout = min(args.max_rollout_steps, train_batch[0].shape[1] - 1)
    sample_steps = sorted({1, max(1, reachable_rollout // 2), reachable_rollout})

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    logs = []
    for rollout_steps in sample_steps:
        trainer.current_rollout_steps = rollout_steps
        trainer.current_noise_std = args.noise_std_init if args.use_noise else 0.0
        trainer._sync_curriculum_state()
        trainer.model.train()

        start = time.perf_counter()
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = trainer._compute_loss(train_batch)
        loss.backward()
        trainer.optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start

        trainer.model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            val_loss = trainer._compute_loss(val_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            val_seconds = time.perf_counter() - start

        logs.append((rollout_steps, trainer.current_history_steps, loss.item(), val_loss.item(), train_seconds, val_seconds))

    B, T, N, C = train_batch[0].shape
    total_params = sum(p.numel() for p in trainer.model.parameters())
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    logger.info(
        f"{hue.y}probe config:{hue.q} "
        f"model={hue.b}{model_args['model_name']}{hue.q}, class={hue.b}{model.__class__.__name__}{hue.q}, "
        f"batch={hue.m}{B}{hue.q}, frames={hue.m}{T}{hue.q}, nodes={hue.m}{N}{hue.q}, "
        f"channels={hue.m}{C}{hue.q}, params={hue.m}{total_params}{hue.q}"
    )
    for rollout_steps, history_steps, train_loss, val_loss, train_seconds, val_seconds in logs:
        logger.info(
            f"{hue.y}probe step:{hue.q} rollout={hue.m}{rollout_steps}{hue.q}, "
            f"history={hue.m}{history_steps}{hue.q}, train={hue.m}{train_loss:.4e}{hue.q} "
            f"({hue.c}{train_seconds:.2f}s{hue.q}), val={hue.m}{val_loss:.4e}{hue.q} "
            f"({hue.c}{val_seconds:.2f}s{hue.q})"
        )
    if device.type == "cuda":
        logger.info(f"{hue.y}peak memory:{hue.q} {hue.m}{peak / 1e9:.2f}{hue.q} GB")

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

    model, model_args = build_model(args=args)
    scalers = {"state_scaler": args.state_scaler, "coord_scaler": args.coord_scaler}

    logger.info(f"train model with {hue.m}{sum(p.numel() for p in model.parameters())}{hue.q} parameters")
    trainer = build_trainer(args, model, _checkpoint_params(args, model_args), scalers, output_dir)
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

    model_args = params["model_args"]
    model, _ = build_model(model_args=model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    channel_names = params["channel_names"]
    spatial_dim = model_args["spatial_dim"]
    total_params = sum(p.numel() for p in model.parameters())
    visualizer = FlowVis(output_dir=output_dir, spatial_dim=spatial_dim, channel_names=channel_names)
    flow_twin = FlowTwin(output_dir=output_dir, channel_names=channel_names)
    metrics = Metrics(channel_names)
    metrics_bank = {}

    focus_channel_idx = channel_names.index("Vy")
    focus_bbox_rel = (0.60, 1.00, 0.00, 1.00) if spatial_dim == 2 else (0.60, 1.00, 0.00, 1.00, 0.00, 1.00)

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
            field_name="Vorticity",
            render_mode="section",
        )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_bank, f, indent=4)

    logger.info(f"{hue.g}============================ [INFERENCE PIPELINE] END ============================={hue.q}")


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    train_loader, val_loader, test_data = data_pipeline(args)

    if "probe" in args.mode:
        probe_pipeline(args, train_loader, val_loader)
    if "train" in args.mode:
        train_pipeline(args, train_loader, val_loader)
    if "infer" in args.mode:
        infer_pipeline(args, test_data)

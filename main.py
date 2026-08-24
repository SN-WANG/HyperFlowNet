# HyperFlowNet command-line entry: generate / train / evaluate / mechanism / benchmark
# Author: Shengning Wang

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from data.datasets import make_data
from models import make_model
from trainer import Trainer
from utils.hue_logger import hue, logger
from utils.mechanism import (
    run_cross_arch,
    run_fm_end_width,
    run_j_sweep,
    run_synthetic_ce,
)
from utils.plotting import plot_cross_arch, plot_fm_width, plot_mechanism, plot_rollout
from utils.seeder import seed_everything

ROOT = Path(__file__).parent


def load_config(path: str | Path) -> dict:
    """Load the YAML configuration file."""
    return yaml.safe_load(Path(path).read_text())


def _device(cfg: dict) -> str:
    dev = str(cfg["training"].get("device", "auto"))
    if dev == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def _data_info(cfg: dict, data: dict) -> tuple[str, int, int, int]:
    """Return (data_kind, channels, history, ndim) for the configured dataset."""
    d = cfg["data"]
    name = d["dataset"]
    if name in ("neptuna_bubble", "neptuna_droplet"):
        nds = d["neptuna"]["bubble" if name == "neptuna_bubble" else "droplet"]
        return "neptuna", int(len(nds["channels"])), int(nds["history"]), 2
    train = data["train"]
    return "mechanism", int(train.shape[2]), int(d.get("history", 1)), int(train.ndim - 3)


def _cmd_generate(args) -> None:
    cfg = load_config(args.config)
    seed_everything(int(cfg["training"]["seed"]))
    data = make_data(cfg)
    logger.info(f"generated {hue.m}{cfg['data']['dataset']}{hue.q}: "
                f"train {data['train'].shape}, test {data['test'].shape}")


def _cmd_train(args) -> None:
    cfg = load_config(args.config)
    if getattr(args, "model", None):
        cfg["model"]["name"] = args.model
    if getattr(args, "objective", None):
        cfg["training"]["objective"] = args.objective
    _run_train(cfg)


def _run_train(cfg: dict) -> None:
    seed_everything(int(cfg["training"]["seed"]))
    device = _device(cfg)
    data = make_data(cfg)
    data_kind, c_in, history, ndim = _data_info(cfg, data)

    model = make_model(cfg["model"]["name"], c_in, ndim, cfg, history)
    logger.info(f"training {hue.m}{cfg['model']['name']}{hue.q} on {hue.m}{cfg['data']['dataset']}{hue.q} "
                f"with objective {hue.m}{cfg['training']['objective']}{hue.q}")

    out_dir = ROOT / cfg["training"]["checkpoint_dir"]
    trainer = Trainer(model, cfg, out_dir, device)
    trainer.fit(data, data_kind, c_in, history)
    ckpt = trainer.save_checkpoint()
    logger.info(f"checkpoint saved to {hue.m}{ckpt}{hue.q}")

    if data_kind == "mechanism":
        test = data["test"]
        metrics = trainer.evaluate(test, int(cfg["eval"]["rollout"]), c_in, history)
        snap_steps = min(40, int(cfg["eval"]["rollout"]))
        snap = _snapshot(trainer, test, snap_steps, c_in, history)
        truth = test[0, history + snap_steps - 1] if test.shape[1] > history + snap_steps else test[0, -1]
        plot_rollout(
            {cfg["model"]["name"]: metrics},
            {cfg["model"]["name"]: snap},
            truth[0] if truth.ndim == 3 else truth,
            np.asarray(data["x"]),
            out_dir / "rollout.png",
        )
        print(f"global_mean={metrics['global_mean']:.4e} shock_mean={metrics['shock_mean']:.4e} tv={metrics['tv_ratio']:.3f}")
    elif data_kind == "neptuna":
        from data.neptuna import load_test_trajectories

        test = load_test_trajectories(
            data["test_ds"].h5_path, data["train_ds"].channels, data["train_ds"].field_stats,
            int(cfg["eval"].get("n_traj", 16)),
        )
        metrics = trainer.evaluate(test, min(int(cfg["eval"]["rollout"]), test.shape[1] - history), c_in, history)
        print(f"global_mean={metrics['global_mean']:.4e} shock_mean={metrics['shock_mean']:.4e} tv={metrics['tv_ratio']:.3f}")


def _snapshot(trainer: Trainer, test: np.ndarray, steps: int, c_in: int, history: int) -> np.ndarray:
    """Roll one trajectory and return the final predicted frame (C, *S)."""
    window = test[0, :history].astype(np.float32)
    for _ in range(steps):
        pred = trainer._step_fn(c_in, history)(window[None])[0]
        window = np.concatenate([window[1:], pred[None]], axis=0)
    return pred


def _cmd_evaluate(args) -> None:
    cfg = load_config(args.config)
    seed_everything(int(cfg["training"]["seed"]))
    device = _device(cfg)
    data = make_data(cfg)
    data_kind, c_in, history, ndim = _data_info(cfg, data)
    model = make_model(cfg["model"]["name"], c_in, ndim, cfg, history)
    out_dir = ROOT / cfg["training"]["checkpoint_dir"]
    trainer = Trainer(model, cfg, out_dir, device)
    trainer.load_checkpoint(args.checkpoint)
    if data_kind == "mechanism":
        metrics = trainer.evaluate(data["test"], int(cfg["eval"]["rollout"]), c_in, history)
        print(f"global_mean={metrics['global_mean']:.4e} shock_mean={metrics['shock_mean']:.4e} tv={metrics['tv_ratio']:.3f}")
    elif data_kind == "neptuna":
        from data.neptuna import load_test_trajectories

        test = load_test_trajectories(
            data["test_ds"].h5_path, data["train_ds"].channels, data["train_ds"].field_stats,
            int(cfg["eval"].get("n_traj", 16)),
        )
        metrics = trainer.evaluate(test, min(int(cfg["eval"]["rollout"]), test.shape[1] - history), c_in, history)
        print(f"global_mean={metrics['global_mean']:.4e} shock_mean={metrics['shock_mean']:.4e} tv={metrics['tv_ratio']:.3f}")


def _cmd_mechanism(args) -> None:
    cfg = load_config(args.config)
    seed_everything(int(cfg["training"]["seed"]))
    device = _device(cfg)
    m = cfg["mechanism"]
    out_dir = ROOT / "runs" / "mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_grid = int(m["n_grid"])
    steps = int(m["steps"])
    batch = int(m["batch"])
    jump = float(m.get("jump", 1.0))
    sigma = float(m.get("sigma", 1.0))

    ce = run_synthetic_ce(
        [float(s) for s in m["sigmas"]], n_grid, jump, steps, batch,
        int(cfg["training"]["seed"]), device, out_dir,
    )
    plot_mechanism(ce, out_dir / "synthetic_ce.png")

    js = run_j_sweep(sigma, [float(j) for j in m.get("jumps", [0.5, 1.0, 2.0])], n_grid, steps, batch, int(cfg["training"]["seed"]), device)
    with open(out_dir / "j_sweep.json", "w") as f:
        json.dump(js, f, indent=2)

    mcfg = {"width": 16, "modes": 8, "depth": 2, "dim": 32, "heads": 2, "patch_size": 8, "basis": 16}
    factories = {name: (lambda n=name: make_model(n, 1, 1, {"model": mcfg})) for name in m["archs"]}
    ca = run_cross_arch(factories, sigma, n_grid, jump, steps, batch, int(cfg["training"]["seed"]), device)
    with open(out_dir / "cross_arch.json", "w") as f:
        json.dump(ca, f, indent=2)
    plot_cross_arch(ca, out_dir / "cross_arch.png")

    fm = run_fm_end_width(
        list(m["fm_paths"]), [float(s) for s in m["sigmas"]], n_grid, jump, steps, batch,
        int(cfg["training"]["seed"]), device,
    )
    with open(out_dir / "fm_end_width.json", "w") as f:
        json.dump(fm, f, indent=2)
    plot_fm_width(fm, out_dir / "fm_end_width.png")
    logger.info(f"mechanism results saved to {hue.m}{out_dir}{hue.q}")


def _cmd_benchmark(args) -> None:
    cfg = load_config(args.config)
    for entry in cfg["benchmark"]["baselines"]:
        cfg["model"]["name"] = entry["model"]
        cfg["training"]["objective"] = entry["objective"]
        cfg["training"]["checkpoint_dir"] = f"runs/bench/{entry['model']}_{entry['objective']}"
        _run_train(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperFlowNet experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--config", default=str(ROOT / "config.yaml"))
    p_gen.set_defaults(func=_cmd_generate)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", default=str(ROOT / "config.yaml"))
    p_train.add_argument("--model", default=None)
    p_train.add_argument("--objective", default=None)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--config", default=str(ROOT / "config.yaml"))
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_mech = sub.add_parser("mechanism")
    p_mech.add_argument("--config", default=str(ROOT / "config.yaml"))
    p_mech.set_defaults(func=_cmd_mechanism)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--config", default=str(ROOT / "config.yaml"))
    p_bench.set_defaults(func=_cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

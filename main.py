# HyperFlowNet command-line entry: generate / train / evaluate / mechanism
# Author: Shengning Wang

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from utils.hue_logger import hue, logger
from utils.plotting import plot_mechanism, plot_rollout
from utils.seeder import seed_everything


ROOT = Path(__file__).parent


def load_config(path: str | Path) -> dict:
    """Load the YAML configuration file."""
    return yaml.safe_load(Path(path).read_text())


def _get_data(cfg: dict) -> dict:
    from data.datasets import generate_dataset, load_dataset, save_dataset, dataset_path

    path = dataset_path(cfg)
    if path.exists():
        logger.info(f"loading dataset from {hue.m}{path}{hue.q}")
        return load_dataset(cfg)
    logger.info(f"generating dataset {hue.m}{cfg['data']['dataset']}{hue.q}")
    data = generate_dataset(cfg)
    saved = save_dataset(cfg, data)
    logger.info(f"dataset saved to {hue.m}{saved}{hue.q}")
    return data


def _snapshot(model, test_s: np.ndarray, steps: int, advance=None) -> np.ndarray:
    key = jax.random.PRNGKey(0)
    state = jnp.asarray(test_s[0, 0][None], dtype=jnp.float32)
    jitted = None if advance is not None else eqx.filter_jit(model)
    for _ in range(steps):
        if advance is not None:
            key, sk = jax.random.split(key)
            state = advance(model, state, sk)
        else:
            state = jitted(state)
    return np.asarray(state[0])


def _cmd_generate(args) -> None:
    cfg = load_config(args.config)
    from data.datasets import generate_dataset, save_dataset

    data = generate_dataset(cfg)
    path = save_dataset(cfg, data)
    print(f"dataset saved: {path}")


def _cmd_train(args) -> None:
    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.objective:
        cfg["training"]["objective"] = args.objective
    key = seed_everything(int(cfg["training"]["seed"]))
    data = _get_data(cfg)
    train_j = jnp.asarray(data["train"], dtype=jnp.float32)
    test_s = np.asarray(data["test"], dtype=np.float32)
    ndim = train_j.ndim - 3
    channels = int(train_j.shape[2])
    rollout = int(cfg["eval"]["rollout"])

    from models import make_model

    name = cfg["model"]["name"]
    key, sk = jax.random.split(key)
    model = make_model(name, sk, c_in=channels, ndim=ndim, cfg=cfg)
    out_dir = ROOT / cfg["training"]["checkpoint_dir"]
    logger.info(f"training {hue.m}{name}{hue.q} on {hue.m}{cfg['data']['dataset']}{hue.q}")

    from trainer import BaseTrainer

    trainer = BaseTrainer(model, cfg, out_dir)
    trainer.fit(train_j, key)
    checkpoint = trainer.save_checkpoint()
    logger.info(f"checkpoint saved to {hue.m}{checkpoint}{hue.q}")
    metrics = trainer.evaluate(test_s, rollout)
    snap_steps = min(40, rollout)
    snap = _snapshot(trainer.model, test_s, snap_steps, advance=trainer._advance())
    plot_rollout(
        {name: metrics},
        {name: snap[0]},
        test_s[0, snap_steps][0] if test_s.shape[1] > snap_steps else test_s[0, -1][0],
        np.asarray(data["x"]),
        out_dir / "rollout.png",
    )
    print(f"global_mean={metrics['global_mean']:.4e} shock_mean={metrics['shock_mean']:.4e} tv={metrics['tv_ratio']:.3f}")


def _cmd_evaluate(args) -> None:
    cfg = load_config(args.config)
    data = _get_data(cfg)
    test_s = np.asarray(data["test"], dtype=np.float32)
    ndim = test_s.ndim - 3
    channels = int(test_s.shape[2])
    rollout = int(cfg["eval"]["rollout"])

    from models import make_model

    key = seed_everything(int(cfg["training"]["seed"]))
    model = make_model(cfg["model"]["name"], key, c_in=channels, ndim=ndim, cfg=cfg)
    out_dir = ROOT / cfg["training"]["checkpoint_dir"]

    from trainer import BaseTrainer

    trainer = BaseTrainer(model, cfg, out_dir)
    trainer.load_checkpoint(Path(args.checkpoint))
    metrics = trainer.evaluate(test_s, rollout)
    print(f"global_mean={metrics['global_mean']:.4e} shock_mean={metrics['shock_mean']:.4e} tv={metrics['tv_ratio']:.3f}")


def _cmd_mechanism(args) -> None:
    cfg = load_config(args.config)
    m = cfg["mechanism"]
    out_dir = ROOT / "runs" / "mechanism"
    from utils.mechanism import run_synthetic_ce

    result = run_synthetic_ce(
        sigmas=[float(s) for s in m["sigmas"]],
        n_samples=int(m["n_samples"]),
        n_grid=int(m["n_grid"]),
        seed=int(cfg["training"]["seed"]),
        out_dir=out_dir,
    )
    plot_mechanism(result, out_dir / "synthetic_ce_plot.png")
    print(f"mechanism result: {result}")


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# Dataset registry, normalization, and npz persistence
# Author: Shengning Wang

import json
from pathlib import Path

import numpy as np

from data.burgers import make_burgers_1d
from data.euler import make_euler_2d_riemann, make_sod_1d
from data.neptuna import load_neptuna
from utils.hue_logger import hue, logger

ROOT = Path(__file__).resolve().parent.parent
_NEPTUNA_KEY = {"neptuna_bubble": "bubble", "neptuna_droplet": "droplet"}


def _add_channel_axis(raw: np.ndarray) -> np.ndarray:
    """Insert the channel axis. (B, T+1, *S) -> (B, T+1, 1, *S)."""
    return raw[:, :, None, ...]


def dataset_path(cfg: dict) -> Path | None:
    """Resolve the npz path for a mechanism dataset; None for Neptuna."""
    d = cfg["data"]
    if d["dataset"] in _NEPTUNA_KEY:
        return None
    data_dir = ROOT / d.get("data_dir", "data")
    return data_dir / f"{d['dataset']}_{int(d['grid'])}.npz"


def generate_dataset(cfg: dict) -> dict:
    """Generate and normalize one mechanism dataset selected by config.

    Args:
        cfg (dict): Config with a data section.

    Returns:
        dict: train/test (B, T+1, C, *S), x, y, mean, std, meta.
    """
    d = cfg["data"]
    seed = int(d["seed"])
    grid = int(d["grid"])
    n_steps = int(d["n_steps"])
    name = d["dataset"]
    meta = {"dataset": name, "grid": grid, "n_steps": n_steps, "channels": int(d["channels"])}
    configs = None
    if name == "burgers_1d":
        x, train, test = make_burgers_1d(
            n_train=int(d["n_train"]), n_test=int(d["n_test"]), n_grid=grid,
            n_steps=n_steps, nu=float(d["nu"]), dt=float(d["dt"]), seed=seed,
        )
        train = _add_channel_axis(train)
        test = _add_channel_axis(test)
        y = None
    elif name == "sod_1d":
        x, train, test = make_sod_1d(
            n_train=int(d["n_train"]), n_test=int(d["n_test"]), n_grid=grid,
            n_steps=n_steps, seed=seed,
        )
        train = np.transpose(train, (0, 1, 3, 2))
        test = np.transpose(test, (0, 1, 3, 2))
        y = None
    elif name == "euler_2d_riemann":
        x, y, train, test, configs = make_euler_2d_riemann(
            n_train=int(d["n_train"]), n_test=int(d["n_test"]), n_grid=grid,
            n_steps=n_steps, seed=seed,
        )
    else:
        raise ValueError(f"unknown dataset: {name}")

    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    mean = train.reshape(train.shape[0], train.shape[1], train.shape[2], -1).mean(axis=(0, 1, 3), keepdims=True)
    std = train.reshape(train.shape[0], train.shape[1], train.shape[2], -1).std(axis=(0, 1, 3), keepdims=True)
    mean = mean.reshape((1, 1, train.shape[2]) + (1,) * (train.ndim - 3))
    std = std.reshape((1, 1, train.shape[2]) + (1,) * (train.ndim - 3))
    std = np.where(std < 1e-8, 1.0, std)
    train = (train - mean) / std
    test = (test - mean) / std
    meta["mean"] = np.asarray(mean, dtype=np.float32).reshape(-1).tolist()
    meta["std"] = np.asarray(std, dtype=np.float32).reshape(-1).tolist()
    data = {
        "train": train,
        "test": test,
        "x": np.asarray(x, dtype=np.float32),
        "y": None if y is None else np.asarray(y, dtype=np.float32),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "meta": meta,
    }
    if configs is not None:
        data["configs"] = configs
    return data


def save_dataset(cfg: dict, data: dict) -> Path:
    """Save a generated mechanism dataset to data/ as a compressed npz."""
    path = dataset_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    arrays["meta_json"] = np.asarray([json.dumps(data["meta"])])
    np.savez_compressed(path, **arrays)
    return path


def load_dataset(cfg: dict) -> dict:
    """Load a mechanism dataset npz and return the same dict shape as generate_dataset."""
    path = dataset_path(cfg)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data.pop("meta_json")[0]))
    mean = np.asarray(meta["mean"], dtype=np.float32).reshape((1, 1, meta["channels"]) + (1,) * (data["train"].ndim - 3))
    std = np.asarray(meta["std"], dtype=np.float32).reshape((1, 1, meta["channels"]) + (1,) * (data["train"].ndim - 3))
    return {
        "train": data["train"],
        "test": data["test"],
        "x": data["x"],
        "y": data.get("y"),
        "mean": mean,
        "std": std,
        "meta": meta,
        "configs": data.get("configs"),
    }


def make_data(cfg: dict) -> dict:
    """Build the dataset selected by config.

    Returns:
        dict: For mechanism datasets, numpy arrays with train/test and stats.
            For Neptuna datasets, train_ds/test_ds torch datasets and meta.
    """
    name = cfg["data"]["dataset"]
    if name in _NEPTUNA_KEY:
        return load_neptuna(cfg["data"]["neptuna"][_NEPTUNA_KEY[name]])
    path = dataset_path(cfg)
    if path.exists():
        logger.info(f"loading dataset from {hue.m}{path}{hue.q}")
        return load_dataset(cfg)
    logger.info(f"generating dataset {hue.m}{name}{hue.q}")
    data = generate_dataset(cfg)
    saved = save_dataset(cfg, data)
    logger.info(f"dataset saved to {hue.m}{saved}{hue.q}")
    return data

# Dataset generation, normalization, and npz persistence
# Author: Shengning Wang

import json
from pathlib import Path

import numpy as np

from data.burgers import make_burgers_1d, make_burgers_2d
from data.euler import make_euler_2d_riemann, make_sod_1d


def _add_channel_axis(raw: np.ndarray) -> np.ndarray:
    """Insert the channel axis. (B, T+1, *S) -> (B, T+1, 1, *S)."""
    return raw[:, :, None, ...]


def generate_dataset(cfg: dict) -> dict:
    """Generate and normalize one dataset selected by config.

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
    elif name == "burgers_2d":
        x, y, train, test = make_burgers_2d(
            n_train=int(d["n_train"]), n_test=int(d["n_test"]), n_grid=grid,
            n_steps=n_steps, nu=float(d["nu"]), dt=float(d["dt"]), seed=seed,
        )
        train = _add_channel_axis(train)
        test = _add_channel_axis(test)
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
    std = np.where(std < 1e-8, 1.0, std)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    train = (train - mean) / std
    test = (test - mean) / std
    meta["mean"] = mean.reshape(-1).tolist()
    meta["std"] = std.reshape(-1).tolist()
    data = {"train": train, "test": test, "x": np.asarray(x, dtype=np.float32), "y": None if y is None else np.asarray(y, dtype=np.float32), "mean": mean, "std": std, "meta": meta}
    if configs is not None:
        data["configs"] = configs
    return data


def dataset_path(cfg: dict) -> Path:
    """Resolve the npz path for a config."""
    d = cfg["data"]
    return Path(d["data_dir"]) / f"{d['dataset']}_{int(d['grid'])}.npz"


def save_dataset(cfg: dict, data: dict) -> Path:
    """Save a generated dataset to data/raw as a compressed npz."""
    path = dataset_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    arrays["meta_json"] = np.asarray([json.dumps(data["meta"])])
    np.savez_compressed(path, **arrays)
    return path


def load_dataset(cfg: dict) -> dict:
    """Load a dataset npz and return the same dict shape as generate_dataset."""
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

# Neptuna engineering data loader (bubble / droplet)
# Author: Shengning Wang

import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.hue_logger import hue, logger

_PARAM_RE = re.compile(r"([A-Za-z]+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_group_params(group_name: str, param_names: list[str] | None = None) -> list[float]:
    """Extract numeric parameters from an HDF5 group name.

    Group names use a key+value token convention, e.g. "005_Mas1.30_sb1_Ax0.0078".
    A leading numeric-only token is treated as an ID prefix and ignored.
    """
    tokens = group_name.split("_")
    if tokens and re.fullmatch(r"\d+", tokens[0]):
        tokens = tokens[1:]
    parsed = {}
    for tok in tokens:
        m = _PARAM_RE.search(tok)
        if not m:
            continue
        parsed[m.group(1)] = float(m.group(2))
    if param_names is None:
        return list(parsed.values())
    return [parsed[name] for name in param_names]


class NeptunaDataset(Dataset):
    """PyTorch dataset over Neptuna h5 trajectories.

    Each h5 group is one trajectory and the group name encodes the conditioning
    parameters. A sample returns a history window, the next frame, and the
    min-max normalized conditioning parameters.
    """

    def __init__(
        self,
        h5_path: str | Path,
        channels: list[str],
        history: int = 10,
        stride: int = 1,
        param_names: list[str] | None = None,
        field_stats: dict | None = None,
        param_stats: dict | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            h5_path (str | Path): Path to train.h5 or test.h5.
            channels (list[str]): h5 field names to load.
            history (int): Number of input frames.
            stride (int): Temporal stride between sampled frames.
            param_names (list[str] | None): Conditioning parameter keys.
            field_stats (dict | None): channel -> (mean, std) for z-normalization.
            param_stats (dict | None): parameter -> (min, max) for min-max normalization.
        """
        self.h5_path = Path(h5_path)
        self.channels = list(channels)
        self.history = int(history)
        self.stride = int(stride)
        self.param_names = list(param_names) if param_names else None
        self.field_stats = field_stats
        self.param_stats = param_stats
        self._h5 = None

        self.groups, self.frames = self._index()
        logger.info(f"Neptuna: {self.h5_path.name} {len(self.groups)} trajectories, first group '{self.groups[0]}' with {self.frames[0]} frames")

    def _index(self) -> tuple[list[str], list[int]]:
        """List (group, frame_count) pairs and the per-group frame counts."""
        with h5py.File(self.h5_path, "r") as f:
            groups = list(f.keys())
            frames = [int(f[g][self.channels[0]].shape[0]) for g in groups]
        return groups, frames

    def _h5file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return int(sum(max(0, fr - self.history * self.stride) for fr in self.frames))

    def __getitem__(self, idx: int) -> dict:
        """Return a history window, target frame, and conditioning parameters."""
        for gi, fr in enumerate(self.frames):
            n_starts = max(0, fr - self.history * self.stride)
            if idx < n_starts:
                start = idx * self.stride
                break
            idx -= n_starts
        else:
            raise IndexError(f"index {idx} out of range")

        group = self._h5file()[self.groups[gi]]
        t_idx = np.arange(start, start + self.history)
        input_frames = [group[ch][t_idx] for ch in self.channels]  # each (H, C_ch, X, Y)
        target_frames = [group[ch][start + self.history] for ch in self.channels]  # each (C_ch, X, Y)
        inputs = input_frames[0] if len(input_frames) == 1 else np.concatenate(input_frames, axis=1)
        targets = target_frames[0] if len(target_frames) == 1 else np.concatenate(target_frames, axis=0)
        inputs, targets = self._normalize(inputs, targets)

        sample = {
            "input": torch.from_numpy(inputs.astype(np.float32, copy=False)),
            "target": torch.from_numpy(targets.astype(np.float32, copy=False)),
        }
        if self.param_names is not None:
            sample["params"] = torch.as_tensor(self._norm_params(self.groups[gi]), dtype=torch.float32)
        return sample

    def _normalize(self, inputs: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel z-normalization using precomputed stats."""
        if self.field_stats is None:
            return inputs, targets
        for ci, ch in enumerate(self.channels):
            mean, std = self.field_stats[ch]
            inputs[:, ci] = (inputs[:, ci] - mean) / std
            targets[ci] = (targets[ci] - mean) / std
        return inputs, targets

    def _norm_params(self, group_name: str) -> np.ndarray:
        """Min-max normalized conditioning parameters from a group name."""
        raw = parse_group_params(group_name, self.param_names)
        if self.param_stats is None:
            return np.asarray(raw, dtype=np.float32)
        norm = []
        for name, val in zip(self.param_names, raw):
            lo, hi = self.param_stats[name]
            norm.append((val - lo) / (hi - lo + 1e-12))
        return np.asarray(norm, dtype=np.float32)


def compute_field_stats(h5_path: str | Path, channels: list[str], frame_stride: int = 10) -> dict:
    """Per-channel (mean, std) over a subsample of frames across all trajectories."""
    sums = {ch: 0.0 for ch in channels}
    sq_sums = {ch: 0.0 for ch in channels}
    counts = {ch: 0 for ch in channels}
    with h5py.File(h5_path, "r") as f:
        for g in f.keys():
            t = f[g][channels[0]].shape[0]
            for idx in range(0, t, frame_stride):
                for ch in channels:
                    x = f[g][ch][idx]
                    sums[ch] += float(x.sum())
                    sq_sums[ch] += float((x.astype(np.float64) ** 2).sum())
                    counts[ch] += int(x.size)
    return {ch: (sums[ch] / counts[ch], np.sqrt(sq_sums[ch] / counts[ch] - (sums[ch] / counts[ch]) ** 2)) for ch in channels}


def compute_param_stats(h5_path: str | Path, param_names: list[str]) -> dict:
    """Per-parameter (min, max) over all group names of a file."""
    stats = {name: [np.inf, -np.inf] for name in param_names}
    with h5py.File(h5_path, "r") as f:
        for g in f.keys():
            for name, val in zip(param_names, parse_group_params(g, param_names)):
                stats[name][0] = min(stats[name][0], val)
                stats[name][1] = max(stats[name][1], val)
    return stats


def load_test_trajectories(
    h5_path: str | Path,
    channels: list[str],
    field_stats: dict | None,
    n_traj: int = 16,
    max_frames: int | None = None,
) -> np.ndarray:
    """Load test trajectories as normalized numpy arrays. (N, T+1, C, H, W)."""
    with h5py.File(h5_path, "r") as f:
        trajs = []
        for g in list(f.keys())[:n_traj]:
            frames = [f[g][ch][:max_frames] for ch in channels]  # each (T, C_ch, H, W)
            x = frames[0] if len(frames) == 1 else np.concatenate(frames, axis=1)
            if field_stats:
                for ci, ch in enumerate(channels):
                    mean, std = field_stats[ch]
                    x[:, ci] = (x[:, ci] - mean) / std
            trajs.append(x.astype(np.float32))
    return np.stack(trajs)


def load_neptuna(cfg: dict) -> dict:
    """Build train/test datasets for a Neptuna config section.

    Args:
        cfg (dict): Neptuna dataset config with path, channels, history,
            stride, param_names.

    Returns:
        dict: train_ds, test_ds, meta.
    """
    data_dir = Path(cfg["path"])
    channels = list(cfg.get("channels", ["density"]))
    history = int(cfg.get("history", 10))
    stride = int(cfg.get("stride", 1))
    param_names = cfg.get("param_names")
    train_path = data_dir / "train.h5"
    test_path = data_dir / "test.h5"
    field_stats = compute_field_stats(train_path, channels)
    param_stats = compute_param_stats(train_path, param_names) if param_names else None
    train_ds = NeptunaDataset(train_path, channels, history, stride, param_names, field_stats, param_stats)
    test_ds = NeptunaDataset(test_path, channels, history, stride, param_names, field_stats, param_stats)
    logger.info(f"{hue.m}Neptuna {data_dir.name}{hue.q}: {len(train_ds)} train / {len(test_ds)} test windows")
    return {
        "train_ds": train_ds,
        "test_ds": test_ds,
        "meta": {"kind": "neptuna", "channels": channels, "history": history},
    }

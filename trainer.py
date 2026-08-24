# Trainer with MSE, CFM, OT-CFM, PDE-Refiner, and HyperFlowNet objectives
# Author: Shengning Wang

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.hue_logger import hue, logger
from utils.metrics import rollout_diagnostics, summarize

_FRONT_STRENGTH = 10.0


class TrajectorySampler:
    """Random window sampler over mechanism trajectories (N, T+1, C, *S)."""

    def __init__(self, data: np.ndarray, history: int) -> None:
        self.data = data
        self.history = int(history)

    def __call__(self, batch: int, device: str) -> dict:
        n, t1 = self.data.shape[:2]
        traj = np.random.randint(0, n, batch)
        t = np.random.randint(self.history - 1, t1 - 2, batch)
        x = np.stack([self.data[traj[i], t[i] - self.history + 1 : t[i] + 1] for i in range(batch)])
        x = x.reshape(batch, -1, *x.shape[3:])  # (B, H*C, *S)
        y = self.data[traj, t + 1]
        y2 = self.data[traj, t + 2]
        return {
            "x": torch.as_tensor(x, dtype=torch.float32, device=device),
            "y": torch.as_tensor(y, dtype=torch.float32, device=device),
            "y2": torch.as_tensor(y2, dtype=torch.float32, device=device),
        }


def _neptuna_collate(batch: list) -> dict:
    x = torch.stack([b["input"] for b in batch])  # (B, H, C, *S)
    y = torch.stack([b["target"] for b in batch])  # (B, C, *S)
    return {"x": x.flatten(1, 2), "y": y, "y2": None}


def _front_weight(y: torch.Tensor) -> torch.Tensor:
    """Shock-region weight for the first channel. (B, C, *S) -> (B, 1, *S)."""
    u = y[:, 0]
    if u.ndim == 2:
        jumps = (u[:, 1:] - u[:, :-1]).abs()
        mask = jumps > 0.08
        mask = mask | torch.roll(mask, 1, -1) | torch.roll(mask, -1, -1)
        mask = torch.nn.functional.pad(mask, (1, 1))
        return 1.0 + _FRONT_STRENGTH * mask[:, None]
    gx = (u[:, :, 1:] - u[:, :, :-1]).abs()
    gy = (u[:, 1:, :] - u[:, :-1, :]).abs()
    mag = torch.zeros_like(u)
    mag[:, :, 1:] = gx
    mag[:, 1:, :] = mag[:, 1:, :] + gy
    mask = mag > 0.2 * mag.max()
    return 1.0 + _FRONT_STRENGTH * mask[:, None]


class Trainer:
    """Training loop, checkpointing, and rollout evaluation for one objective."""

    def __init__(self, model: nn.Module, cfg: dict, out_dir: str | Path, device: str) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history_log: list[dict] = []

    def fit(self, data: dict, data_kind: str, c_in: int, history: int) -> None:
        """Train the model according to the configured objective."""
        tcfg = self.cfg["training"]
        objective = tcfg["objective"]
        steps1 = int(tcfg["steps_stage1"])
        steps2 = int(tcfg["steps_stage2"])
        batch = int(tcfg["batch"])
        amp = bool(tcfg.get("amp", False))

        opt = torch.optim.Adam(self.model.parameters(), lr=float(tcfg["lr"]))
        if data_kind == "neptuna":
            loader = DataLoader(data["train_ds"], batch_size=batch, shuffle=True, collate_fn=_neptuna_collate)
            iterator = iter(loader)
        else:
            sampler = TrajectorySampler(data["train"], history)

        total = steps1 + steps2 if objective in ("pde_refiner", "hyflow") else steps1
        for i in range(total):
            if data_kind == "neptuna":
                try:
                    batch_data = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch_data = next(iterator)
                batch_data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch_data.items()}
            else:
                batch_data = sampler(batch, self.device)
            tau = float(np.random.rand())
            stage2 = i >= steps1
            opt.zero_grad()
            with torch.autocast(
                device_type="cuda" if "cuda" in self.device else "cpu", dtype=torch.bfloat16, enabled=amp
            ):
                loss = self._loss(batch_data, tau, stage2, c_in, history)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            self.history_log.append({"step": i, "loss": float(loss.detach())})

    def _loss(self, b: dict, tau: float, stage2: bool, c_in: int, history: int) -> torch.Tensor:
        """Objective-specific loss for one batch."""
        objective = self.cfg["training"]["objective"]
        x, y = b["x"], b["y"]
        if objective == "mse":
            pred = self.model(x)
            loss = torch.mean((pred - y) ** 2)
            if history == 1 and b["y2"] is not None:
                loss = loss + 0.5 * torch.mean((self.model(pred) - b["y2"]) ** 2)
            return loss
        if objective == "frontloss":
            pred = self.model(x)
            return torch.mean(_front_weight(y) * (pred - y) ** 2)
        if objective == "cfm":
            return self.model.forward(x, y, tau, "straight")
        if objective == "otcfm":
            return self.model.forward(x, y, tau, "ot")
        if objective == "pde_refiner":
            if not stage2:
                return torch.mean((self.model.base(x) - y) ** 2)
            sigma = float(np.random.uniform(0.1, 1.0))
            y_noisy = y + sigma * torch.randn_like(y)
            return torch.mean((self.model.denoise(x, y_noisy, sigma) - y) ** 2)
        if objective == "hyflow":
            loss = self.model.train_loss(x, y, tau)
            if stage2:
                pred = self.model.advance(x, 1)
                x2 = torch.cat([x[:, c_in:], pred], dim=1)
                loss = loss + 0.5 * self.model.train_loss(x2, y, float(np.random.rand()))
            return loss
        raise ValueError(f"unknown objective: {objective}")

    def _step_fn(self, c_in: int, history: int):
        """Numpy step function rolling a history window through the model."""

        def step(window: np.ndarray) -> np.ndarray:
            flat = np.ascontiguousarray(window).reshape(window.shape[0], -1, *window.shape[3:])  # (1, H*C, *S)
            x = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda" if "cuda" in self.device else "cpu",
                    dtype=torch.bfloat16,
                    enabled=bool(self.cfg["training"].get("amp", False)),
                ):
                    pred = self.model.predict(x) if hasattr(self.model, "predict") else self.model(x)
            return pred.detach().cpu().numpy()

        return step

    def evaluate(self, test: np.ndarray, rollout: int, c_in: int, history: int) -> dict:
        """Run rollout diagnostics, save metrics JSON, and return the summary."""
        diag = rollout_diagnostics(self._step_fn(c_in, history), test, rollout, history)
        metrics = summarize(diag)
        with open(self.out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(self.out_dir / "history.json", "w") as f:
            json.dump(self.history_log, f, indent=2)
        logger.info(f"{hue.g}metrics saved to {hue.q}{self.out_dir / 'metrics.json'}")
        return metrics

    def save_checkpoint(self) -> Path:
        path = self.out_dir / "ckpt.pt"
        torch.save(self.model.state_dict(), path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))

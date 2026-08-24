# Plotting helpers for rollout, mechanism, and decomposition results
# Author: Shengning Wang

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsn_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_rollout(metrics: dict, snapshots: dict, truth: np.ndarray, x: np.ndarray, out_path: str | Path) -> None:
    """Save rollout error curves, front bars, and a late-time snapshot."""
    steps = np.arange(1, max(len(m["global"]) for m in metrics.values()) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, m in metrics.items():
        axes[0, 0].semilogy(steps[: len(m["global"])], m["global"], label=name, lw=1.1)
        axes[1, 0].semilogy(steps[: len(m["shock"])], m["shock"], label=name, lw=1.1)
    axes[0, 0].set_title("Global rollout error")
    axes[0, 0].set_xlabel("rollout step")
    axes[0, 0].set_ylabel("relative L2")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].set_title("Shock-region error")
    axes[1, 0].set_xlabel("rollout step")
    axes[1, 0].set_ylabel("relative L2")
    axes[1, 0].legend(fontsize=7, ncol=2)

    names = list(metrics)
    tv = [metrics[n]["tv_ratio"] for n in names]
    off = [metrics[n]["edge_offset"] for n in names]
    idx = np.arange(len(names))
    axes[0, 1].bar(idx - 0.2, tv, 0.4, label="front TV ratio (truth=1)")
    axes[0, 1].bar(idx + 0.2, off, 0.4, label="mean |edge offset| (cells)")
    axes[0, 1].set_xticks(idx)
    axes[0, 1].set_xticklabels(names, rotation=20, fontsize=8)
    axes[0, 1].axhline(1.0, color="k", lw=0.8)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Front sharpness and position")

    if truth.ndim == 1:
        axes[1, 1].plot(x, truth, "k-", lw=1.4, label="truth")
        for name in names:
            if name in snapshots:
                axes[1, 1].plot(x, snapshots[name], "--", lw=1.0, label=name)
    else:
        axes[1, 1].text(0.5, 0.5, "2D snapshot omitted", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("u")
    axes[1, 1].set_title("Snapshot at final rollout step")
    axes[1, 1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mechanism(result: dict, out_path: str | Path) -> None:
    """Save the analytic versus fitted ramp width comparison."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(result["sigmas"], result["analytic_widths"], "k--", label="analytic 2.56 sigma")
    ax.plot(result["sigmas"], result["fitted_widths"], "o-", label="fitted (MSE CNN)")
    ax.set_xlabel("position uncertainty sigma (cells)")
    ax.set_ylabel("10%-90% ramp width (cells)")
    ax.set_title("Conditional-expectation smearing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fm_width(result: dict, out_path: str | Path) -> None:
    """Save the ODE endpoint width vs sigma per path family."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    first = next(iter(result.values()))
    sigmas = first["sigmas"]
    ax.plot(sigmas, [2.0 * s * 1.2816 for s in sigmas], "k--", label="analytic 2.56 sigma")
    for path, entry in result.items():
        ax.plot(entry["sigmas"], entry["widths"], "o-", label=path)
    ax.set_xlabel("target displacement sigma (cells)")
    ax.set_ylabel("endpoint 10%-90% width (cells)")
    ax.set_title("Flow-matching endpoint width by path family")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_arch(result: dict, out_path: str | Path) -> None:
    """Save the fitted ramp width across architectures."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = list(result["widths"])
    widths = [result["widths"][n] for n in names]
    idx = np.arange(len(names))
    ax.bar(idx, widths, 0.55)
    ax.axhline(result["analytic_width"], color="k", ls="--", label=f"analytic 2.56 sigma = {result['analytic_width']:.2f}")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("fitted ramp width (cells)")
    ax.set_title(f"Cross-architecture smearing at sigma = {result['sigma']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decomposition(rq1: dict, out_path: str | Path) -> None:
    """Save transport and shape error growth curves for two models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    steps = np.arange(1, len(next(iter(rq1.values()))["transport"]) + 1)
    for name, diag in rq1.items():
        axes[0].semilogy(steps, diag["transport"], "--", label=f"{name} transport")
        axes[0].semilogy(steps, diag["shape"], "-", label=f"{name} shape")
        axes[1].plot(steps, diag["tv_ratio"], label=name)
    axes[0].set_title("Transport vs shape error")
    axes[0].set_xlabel("rollout step")
    axes[0].set_ylabel("relative L2")
    axes[0].legend()
    axes[1].set_title("Front TV ratio")
    axes[1].set_xlabel("rollout step")
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

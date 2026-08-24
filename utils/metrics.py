# Discontinuity-focused rollout diagnostics
# Author: Shengning Wang

import numpy as np


def ramp_width_1d(u: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    """10%-90% ramp width of a 1D field in cells. (N,) -> float."""
    x = np.arange(u.size)
    vmin = u.min()
    vmax = u.max()
    span = vmax - vmin
    if span < 1e-8:
        return 0.0
    x_low = np.interp(vmin + low * span, u, x)
    x_high = np.interp(vmin + high * span, u, x)
    return float(x_high - x_low)


def shock_mask_1d(truth: np.ndarray, width: int = 3, threshold: float = 0.08) -> np.ndarray:
    """Mask cells near 1D jumps. (N,) -> (N,) bool."""
    jumps = np.abs(np.diff(truth))
    idx = np.where(jumps > threshold)[0]
    mask = np.zeros_like(truth, dtype=bool)
    for i in idx:
        mask[max(0, i - width) : min(len(truth), i + width + 1)] = True
    return mask


def _dilate(mask: np.ndarray, width: int) -> np.ndarray:
    """Morphological dilation of a 2D mask."""
    out = mask.copy()
    for _ in range(width):
        out = out | np.roll(out, 1, axis=0) | np.roll(out, -1, axis=0) | np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1)
    return out


def shock_mask_2d(truth: np.ndarray, width: int = 1, ratio: float = 0.2) -> np.ndarray:
    """Mask cells near 2D gradient magnitude peaks. (H, W) -> (H, W) bool."""
    gx = np.zeros_like(truth)
    gy = np.zeros_like(truth)
    gx[:, :-1] = truth[:, 1:] - truth[:, :-1]
    gy[:-1, :] = truth[1:, :] - truth[:-1, :]
    mag = np.sqrt(gx**2 + gy**2)
    mask = mag > ratio * mag.max()
    return _dilate(mask, width) if width > 0 else mask


def best_shift_1d(pred: np.ndarray, truth: np.ndarray) -> int:
    """Best integer circular shift aligning pred to truth."""
    corr = np.fft.irfft(np.fft.rfft(pred) * np.conj(np.fft.rfft(truth)), n=pred.size)
    return int(np.argmax(corr))


def front_offset_1d(pred: np.ndarray, truth: np.ndarray) -> float:
    """Front offset in cells from the strongest gradient."""
    t_edge = int(np.argmax(np.abs(np.diff(truth))))
    window = slice(max(0, t_edge - 12), min(len(truth), t_edge + 12))
    p_edge = t_edge + int(np.argmax(np.abs(np.diff(pred))[window])) - 12
    return float(p_edge - t_edge)


def tv_ratio_1d(pred: np.ndarray, truth: np.ndarray) -> float:
    """Total variation ratio in a band around the truth front."""
    t_edge = int(np.argmax(np.abs(np.diff(truth))))
    band = np.zeros_like(truth, dtype=bool)
    band[max(0, t_edge - 5) : min(len(truth), t_edge + 5)] = True
    return float(np.abs(np.diff(pred))[band[:-1]].sum() / max(np.abs(np.diff(truth))[band[:-1]].sum(), 1e-12))


def tv_ratio_2d(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Total variation ratio inside the front mask band."""
    gx = np.zeros_like(pred)
    gy = np.zeros_like(pred)
    gx[:, :-1] = pred[:, 1:] - pred[:, :-1]
    gy[:-1, :] = pred[1:, :] - pred[:-1, :]
    p_mag = np.sqrt(gx**2 + gy**2)
    gx = np.zeros_like(truth)
    gy = np.zeros_like(truth)
    gx[:, :-1] = truth[:, 1:] - truth[:, :-1]
    gy[:-1, :] = truth[1:, :] - truth[:-1, :]
    t_mag = np.sqrt(gx**2 + gy**2)
    return float(p_mag[mask].sum() / max(t_mag[mask].sum(), 1e-12))


def spectrum_error_1d(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 error of the 1D Fourier magnitude spectrum."""
    p = np.abs(np.fft.rfft(pred))
    t = np.abs(np.fft.rfft(truth))
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12))


def spectrum_error_2d(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 error of the 2D Fourier magnitude spectrum."""
    p = np.abs(np.fft.rfft2(pred))
    t = np.abs(np.fft.rfft2(truth))
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12))


def _rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-12))


def _spectrum_error(pred: np.ndarray, truth: np.ndarray, ndim: int) -> float:
    return spectrum_error_1d(pred, truth) if ndim == 1 else spectrum_error_2d(pred, truth)


def rollout_diagnostics(step_fn, test: np.ndarray, steps: int, history: int = 1) -> dict:
    """Per-step global, shock, shape, transport, TV, offset, and spectrum errors.

    Args:
        step_fn: Callable mapping a history window (1, H*C, *S) to the next frame.
        test (np.ndarray): Test trajectories (TEST, T+1, C, *S).
        steps (int): Rollout length.
        history (int): History window length in frames.

    Returns:
        dict: Per-step arrays of length steps.
    """
    ndim = test.ndim - 3
    n_test = len(test)
    global_err = np.zeros(steps)
    shock_err = np.zeros(steps)
    counts = np.zeros(steps)
    shape_err = np.zeros(steps)
    transport_err = np.zeros(steps)
    tv_ratio = np.zeros(steps)
    tv_count = np.zeros(steps)
    offsets = np.zeros(steps)
    offset_count = np.zeros(steps)
    spectrum = np.zeros(steps)

    for traj in test:
        window = traj[:history].astype(np.float32)  # (H, C, *S)
        for t in range(steps):
            state = step_fn(window[None])
            truth = traj[history + t]
            pred = np.asarray(state[0])
            window = np.concatenate([window[1:], pred[None]], axis=0)
            global_err[t] += _rel_l2(pred, truth)
            spectrum[t] += _spectrum_error(pred[0], truth[0], ndim)
            if ndim == 1:
                p0, t0 = pred[0], truth[0]
                shift = best_shift_1d(p0, t0)
                aligned = np.roll(p0, -shift)
                shape_err[t] += _rel_l2(aligned, t0)
                transport_err[t] += _rel_l2(p0, aligned)
                mask = shock_mask_1d(t0)
                if mask.sum() > 0:
                    shock_err[t] += _rel_l2(pred[:, mask], truth[:, mask])
                    counts[t] += 1
                    offsets[t] += abs(front_offset_1d(p0, t0))
                    offset_count[t] += 1
                    tv_ratio[t] += tv_ratio_1d(p0, t0)
                    tv_count[t] += 1
            else:
                mask = shock_mask_2d(truth[0])
                if mask.sum() > 0:
                    shock_err[t] += _rel_l2(pred[:, mask], truth[:, mask])
                    counts[t] += 1
                    tv_ratio[t] += tv_ratio_2d(pred[0], truth[0], mask)
                    tv_count[t] += 1

    counts[counts == 0] = 1
    tv_count[tv_count == 0] = 1
    offset_count[offset_count == 0] = 1
    return {
        "global": global_err / n_test,
        "shock": shock_err / counts,
        "shape": shape_err / n_test,
        "transport": transport_err / n_test,
        "tv_ratio": tv_ratio / tv_count,
        "edge_offset": offsets / offset_count,
        "spectrum": spectrum / n_test,
    }


def summarize(diag: dict) -> dict:
    """Convert diagnostics to a JSON-friendly metric dict."""
    steps = len(diag["global"])
    return {
        "global": diag["global"].tolist(),
        "shock": diag["shock"].tolist(),
        "shape": diag["shape"].tolist(),
        "transport": diag["transport"].tolist(),
        "spectrum": diag["spectrum"].tolist(),
        "global_mean": float(diag["global"].mean()),
        "shock_mean": float(diag["shock"].mean()),
        "shape_mean": float(diag["shape"].mean()),
        "transport_mean": float(diag["transport"].mean()),
        "spectrum_mean": float(diag["spectrum"].mean()),
        "tv_ratio": float(diag["tv_ratio"].mean()),
        "edge_offset": float(diag["edge_offset"].mean()),
        "global_step60": float(diag["global"][min(59, steps - 1)]),
    }

# 1D Sod and 2D Riemann problem data generators (MUSCL-Rusanov)
# Author: Shengning Wang

import numpy as np


GAMMA = 1.4
CFL = 0.4


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter of two slopes."""
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _primitive_1d(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conserved to primitive variables for 1D Euler."""
    rho = w[..., 0]
    u = w[..., 1] / rho
    e = w[..., 2]
    p = (GAMMA - 1.0) * (e - 0.5 * rho * u**2)
    c = np.sqrt(GAMMA * p / np.maximum(rho, 1e-12))
    return rho, u, p, c


def _flux_1d(w: np.ndarray) -> np.ndarray:
    """Physical flux for 1D Euler."""
    rho, u, p, _ = _primitive_1d(w)
    f = np.empty_like(w)
    f[..., 0] = rho * u
    f[..., 1] = rho * u**2 + p
    f[..., 2] = u * (w[..., 2] + p)
    return f


def make_sod_1d(
    n_train: int = 96,
    n_test: int = 16,
    n_grid: int = 512,
    n_steps: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 1D Sod shock tube trajectories with random initial states.

    Args:
        n_train (int): Training trajectories.
        n_test (int): Test trajectories.
        n_grid (int): Number of cells.
        n_steps (int): Saved frames (T+1 total).
        seed (int): Random seed.

    Returns:
        tuple: x (N,), train (TRAIN, T+1, N, 3), test (TEST, T+1, N, 3).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False) + 0.5 / n_grid
    dx = x[1] - x[0]

    def solve(w0: np.ndarray, sub_steps: int = 4) -> np.ndarray:
        w = w0.copy()
        frames = np.empty((n_steps + 1, n_grid, 3))
        frames[0] = w
        for t in range(n_steps):
            for _ in range(sub_steps):
                wp = np.pad(w, ((1, 1), (0, 0)), mode="edge")
                slope = _minmod(wp[1:-1] - wp[:-2], wp[2:] - wp[1:-1]) * 0.5
                ql = wp[1:-1] - slope
                qr = wp[1:-1] + slope
                left = np.concatenate([wp[0:1], qr], axis=0)
                right = np.concatenate([ql, wp[-1:]], axis=0)
                fl = _flux_1d(left)
                fr = _flux_1d(right)
                _, ul, pl, cl = _primitive_1d(left)
                _, ur, pr, cr = _primitive_1d(right)
                a = np.maximum(np.abs(ul) + cl, np.abs(ur) + cr)
                flux = 0.5 * (fl + fr) - 0.5 * a[..., None] * (right - left)
                dt = CFL * dx / np.max(a)
                w = w - dt / dx * (flux[1:] - flux[:-1])
            frames[t + 1] = w
        return frames

    def initial_state(rng: np.random.Generator) -> np.ndarray:
        rho_l = rng.uniform(1.0, 1.5)
        p_l = rng.uniform(0.8, 1.5)
        rho_r = rng.uniform(0.1, 0.5)
        p_r = rng.uniform(0.05, 0.4)
        left = np.array([rho_l, 0.0, p_l / (GAMMA - 1.0)])
        right = np.array([rho_r, 0.0, p_r / (GAMMA - 1.0)])
        n_left = n_grid // 2
        return np.concatenate([np.tile(left, (n_left, 1)), np.tile(right, (n_grid - n_left, 1))])

    train = np.stack([solve(initial_state(rng)) for _ in range(n_train)])
    test = np.stack([solve(initial_state(rng)) for _ in range(n_test)])
    return x.astype(np.float32), train.astype(np.float32), test.astype(np.float32)


def _primitive_2d(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conserved to primitive variables for 2D Euler."""
    rho = w[0]
    u = w[1] / rho
    v = w[2] / rho
    e = w[3]
    p = (GAMMA - 1.0) * (e - 0.5 * rho * (u**2 + v**2))
    c = np.sqrt(GAMMA * p / np.maximum(rho, 1e-12))
    return rho, u, v, p, c


def _flux_x(w: np.ndarray) -> np.ndarray:
    """Physical flux in x for 2D Euler."""
    rho, u, v, p, _ = _primitive_2d(w)
    f = np.empty_like(w)
    f[0] = rho * u
    f[1] = rho * u**2 + p
    f[2] = rho * u * v
    f[3] = u * (w[3] + p)
    return f


def _flux_y(w: np.ndarray) -> np.ndarray:
    """Physical flux in y for 2D Euler."""
    rho, u, v, p, _ = _primitive_2d(w)
    f = np.empty_like(w)
    f[0] = rho * v
    f[1] = rho * u * v
    f[2] = rho * v**2 + p
    f[3] = v * (w[3] + p)
    return f


def _interface_flux_2d(w: np.ndarray, axis: int, flux_fn) -> np.ndarray:
    """MUSCL-Rusanov flux at all interfaces along one axis."""
    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (1, 1)
    wp = np.pad(w, pad_width, mode="edge")
    sl = tuple(slice(None) for _ in range(3))
    diff_l = wp[sl[:axis] + (slice(1, -1),) + sl[axis + 1 :]] - wp[sl[:axis] + (slice(0, -2),) + sl[axis + 1 :]]
    diff_r = wp[sl[:axis] + (slice(2, None),) + sl[axis + 1 :]] - wp[sl[:axis] + (slice(1, -1),) + sl[axis + 1 :]]
    slope = _minmod(diff_l, diff_r) * 0.5
    center = wp[sl[:axis] + (slice(1, -1),) + sl[axis + 1 :]]
    ql = center - slope
    qr = center + slope
    if axis == 1:
        left = np.concatenate([wp[:, 0:1, :], qr], axis=1)
        right = np.concatenate([ql, wp[:, -1:, :]], axis=1)
    else:
        left = np.concatenate([wp[:, :, 0:1], qr], axis=2)
        right = np.concatenate([ql, wp[:, :, -1:]], axis=2)
    fl = flux_fn(left)
    fr = flux_fn(right)
    rl, ul, vl, pl, cl = _primitive_2d(left)
    rr, ur, vr, pr, cr = _primitive_2d(right)
    a = np.maximum(np.abs(ul) + cl, np.abs(ur) + cr)
    return 0.5 * (fl + fr) - 0.5 * a[None] * (right - left)


def make_euler_2d_riemann(
    n_train: int = 96,
    n_test: int = 16,
    n_grid: int = 128,
    n_steps: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D Euler Riemann problem trajectories (configs 3 and 6).

    Args:
        n_train (int): Training trajectories.
        n_test (int): Test trajectories.
        n_grid (int): Cells per axis.
        n_steps (int): Saved frames (T+1 total).
        seed (int): Random seed.

    Returns:
        tuple: x (H,), y (W,), train (TRAIN, T+1, 4, H, W),
        test (TEST, T+1, 4, H, W), configs (TRAIN+TEST,).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False) + 0.5 / n_grid
    dx = x[1] - x[0]
    mid = n_grid // 2

    def conserved(rho: float, u: float, v: float, p: float) -> np.ndarray:
        e = p / (GAMMA - 1.0) + 0.5 * rho * (u**2 + v**2)
        return np.array([rho, rho * u, rho * v, e])

    config_states = {
        3: [
            conserved(1.5, 0.0, 0.0, 1.5),
            conserved(0.5323, 1.206, 0.0, 0.3),
            conserved(0.138, 1.206, 1.206, 0.029),
            conserved(0.5323, 0.0, 1.206, 0.3),
        ],
        6: [
            conserved(1.0, 0.75, -0.5, 1.0),
            conserved(2.0, 0.75, 0.5, 1.0),
            conserved(1.0, -0.75, 0.5, 1.0),
            conserved(3.0, -0.75, -0.5, 1.0),
        ],
    }

    def solve(w0: np.ndarray, sub_steps: int = 4) -> np.ndarray:
        w = w0.copy()
        frames = np.empty((n_steps + 1, 4, n_grid, n_grid))
        frames[0] = w
        for t in range(n_steps):
            for _ in range(sub_steps):
                fx = _interface_flux_2d(w, axis=1, flux_fn=_flux_x)
                fy = _interface_flux_2d(w, axis=2, flux_fn=_flux_y)
                rho, u, v, p, c = _primitive_2d(w)
                a = np.maximum(np.abs(u) + c, np.abs(v) + c)
                dt = CFL * dx / np.max(a)
                w = w - dt / dx * (fx[:, 1:, :] - fx[:, :-1, :]) - dt / dx * (fy[:, :, 1:] - fy[:, :, :-1])
            frames[t + 1] = w
        return frames

    def initial_state(config: int, rng: np.random.Generator) -> np.ndarray:
        states = config_states[config]
        w = np.empty((4, n_grid, n_grid))
        w[:, :mid, :mid] = states[0][:, None, None]
        w[:, :mid, mid:] = states[1][:, None, None]
        w[:, mid:, mid:] = states[2][:, None, None]
        w[:, mid:, :mid] = states[3][:, None, None]
        w = w * (1.0 + 0.02 * rng.normal(size=w.shape))
        return np.maximum(w, 1e-8)

    configs = []
    train_frames = []
    test_frames = []
    for i in range(n_train + n_test):
        config = 3 if rng.random() < 0.5 else 6
        configs.append(config)
        frames = solve(initial_state(config, rng))
        if i < n_train:
            train_frames.append(frames)
        else:
            test_frames.append(frames)
    train = np.stack(train_frames)
    test = np.stack(test_frames)
    return x.astype(np.float32), x.astype(np.float32), train.astype(np.float32), test.astype(np.float32), np.asarray(configs, dtype=np.int64)

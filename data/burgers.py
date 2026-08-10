# 1D/2D viscous Burgers data generators with pseudospectral solvers
# Author: Shengning Wang

import numpy as np


def make_burgers_1d(
    n_train: int = 96,
    n_test: int = 16,
    n_grid: int = 256,
    n_steps: int = 200,
    nu: float = 0.02,
    dt: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate periodic 1D Burgers trajectories.

    Args:
        n_train (int): Training trajectories.
        n_test (int): Test trajectories.
        n_grid (int): Spatial resolution.
        n_steps (int): Time steps per trajectory.
        nu (float): Viscosity.
        dt (float): Time step.
        seed (int): Random seed.

    Returns:
        tuple: x (N,), train (TRAIN, T+1, N), test (TEST, T+1, N).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.rfftfreq(n_grid, d=dx) * 2.0 * np.pi
    dealias = np.ones_like(k)
    dealias[int(0.67 * k.size) :] = 0.0
    filter_k = np.exp(-18.0 * (k / k[-1]) ** 8)

    def initial_condition(rng: np.random.Generator) -> np.ndarray:
        u = 0.5 * rng.uniform(-1.0, 1.0, 4).sum() * np.ones(n_grid)
        for j in range(1, 5):
            u = u + rng.uniform(-0.25, 0.25) * np.sin(j * x + rng.uniform(0, 2 * np.pi))
        if rng.random() < 0.7:
            x0 = rng.uniform(0.5, 2.0 * np.pi - 0.5)
            delta = rng.uniform(0.15, 0.25)
            u = u + rng.uniform(0.2, 0.5) * np.tanh((x - x0) / delta)
        return u - u.mean()

    def ifrk4_step(u: np.ndarray) -> np.ndarray:
        def n_hat(u: np.ndarray) -> np.ndarray:
            u_hat = np.fft.rfft(u) * dealias
            u_x = np.fft.irfft(1j * k * u_hat, n=u.size)
            return np.fft.rfft(-u * u_x) * filter_k

        u_hat = np.fft.rfft(u) * filter_k
        k1 = n_hat(u)
        u2 = np.fft.irfft((u_hat + 0.5 * dt * k1) * np.exp(-nu * k**2 * 0.5 * dt), n=u.size)
        k2 = n_hat(u2)
        u3 = np.fft.irfft((u_hat + 0.5 * dt * k2) * np.exp(-nu * k**2 * 0.5 * dt), n=u.size)
        k3 = n_hat(u3)
        u4 = np.fft.irfft((u_hat + dt * k3) * np.exp(-nu * k**2 * dt), n=u.size)
        k4 = n_hat(u4)
        v = u_hat + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return np.fft.irfft(v * np.exp(-nu * k**2 * dt), n=u.size)

    def solve(u0: np.ndarray) -> np.ndarray:
        u = u0.copy()
        traj = np.empty((n_steps + 1, n_grid))
        traj[0] = u0
        for t in range(n_steps):
            u = ifrk4_step(u)
            traj[t + 1] = u
        return traj

    train = np.stack([solve(initial_condition(rng)) for _ in range(n_train)])
    test = np.stack([solve(initial_condition(rng)) for _ in range(n_test)])
    return x.astype(np.float32), train.astype(np.float32), test.astype(np.float32)


def make_burgers_2d(
    n_train: int = 96,
    n_test: int = 16,
    n_grid: int = 128,
    n_steps: int = 100,
    nu: float = 0.01,
    dt: float = 0.002,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate periodic 2D Burgers trajectories.

    Args:
        n_train (int): Training trajectories.
        n_test (int): Test trajectories.
        n_grid (int): Spatial resolution per axis.
        n_steps (int): Time steps per trajectory.
        nu (float): Viscosity.
        dt (float): Time step.
        seed (int): Random seed.

    Returns:
        tuple: x (H,), y (W,), train (TRAIN, T+1, H, W), test (TEST, T+1, H, W).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dx = x[1] - x[0]
    kx = np.fft.rfftfreq(n_grid, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(n_grid, d=dx) * 2.0 * np.pi
    kx2 = np.broadcast_to(kx[None, :], (n_grid, kx.size))
    ky2 = np.broadcast_to(ky[:, None], (n_grid, kx.size))
    k2 = kx2**2 + ky2**2
    dealias = np.ones_like(k2)
    dealias[int(0.67 * n_grid) :, :] = 0.0
    dealias[:, int(0.67 * kx.size) :] = 0.0
    filter_k = np.exp(-18.0 * (k2 / k2.max()) ** 4)

    def initial_condition(rng: np.random.Generator) -> np.ndarray:
        u = 0.5 * rng.uniform(-1.0, 1.0, 4).sum() * np.ones((n_grid, n_grid))
        for j in range(1, 4):
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(-0.3, 0.3)
            u = u + amp * np.sin(j * x[:, None] + phase_x) * np.sin(j * x[None, :] + phase_y)
        if rng.random() < 0.7:
            x0 = rng.uniform(1.0, 2.0 * np.pi - 1.0)
            y0 = rng.uniform(1.0, 2.0 * np.pi - 1.0)
            delta = rng.uniform(0.3, 0.5)
            u = u + rng.uniform(0.2, 0.4) * np.tanh((x[:, None] - x0 + x[None, :] - y0) / delta)
        return u - u.mean()

    def ifrk4_step(u: np.ndarray) -> np.ndarray:
        def n_hat(u: np.ndarray) -> np.ndarray:
            u_hat = np.fft.rfft2(u) * dealias
            ux = np.fft.irfft2(1j * kx2 * u_hat, s=u.shape)
            uy = np.fft.irfft2(1j * ky2 * u_hat, s=u.shape)
            return np.fft.rfft2(-0.5 * u * ux - 0.5 * u * uy) * filter_k

        u_hat = np.fft.rfft2(u) * filter_k
        n1 = n_hat(u)
        u2 = np.fft.irfft2((u_hat + 0.5 * dt * n1) * np.exp(-nu * k2 * 0.5 * dt), s=u.shape)
        n2 = n_hat(u2)
        u3 = np.fft.irfft2((u_hat + 0.5 * dt * n2) * np.exp(-nu * k2 * 0.5 * dt), s=u.shape)
        n3 = n_hat(u3)
        u4 = np.fft.irfft2((u_hat + dt * n3) * np.exp(-nu * k2 * dt), s=u.shape)
        n4 = n_hat(u4)
        v = u_hat + dt * (n1 + 2.0 * n2 + 2.0 * n3 + n4) / 6.0
        return np.fft.irfft2(v * np.exp(-nu * k2 * dt), s=u.shape)

    def solve(u0: np.ndarray) -> np.ndarray:
        u = u0.copy()
        traj = np.empty((n_steps + 1, n_grid, n_grid))
        traj[0] = u0
        for t in range(n_steps):
            u = ifrk4_step(u)
            traj[t + 1] = u
        return traj

    train = np.stack([solve(initial_condition(rng)) for _ in range(n_train)])
    test = np.stack([solve(initial_condition(rng)) for _ in range(n_test)])
    return x.astype(np.float32), x.astype(np.float32), train.astype(np.float32), test.astype(np.float32)

# Trainer with MSE, flow, diffusion, front-loss, and HyperFlowNet objectives
# Author: Shengning Wang

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.hue_logger import hue, logger
from utils.metrics import rollout_diagnostics, summarize


DIFF_T = 50
DIFF_STEPS = 12
FLOW_STEPS = 24


def cosine_alphas(t_max: int = DIFF_T) -> jnp.ndarray:
    """Cosine noise schedule alpha_bar. (T+1,)."""
    t = jnp.arange(t_max + 1, dtype=jnp.float32)
    s = (t / t_max + 0.008) / 1.008
    return jnp.cos(s * jnp.pi / 2.0) ** 2


def _sample_batch(train: jnp.ndarray, batch: int, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample (state, next, next-next) frames from trajectories."""
    k1, k2 = jax.random.split(key)
    traj_idx = jax.random.randint(k1, (batch,), 0, len(train))
    t_max = train.shape[1] - 3
    t_idx = jax.random.randint(k2, (batch,), 0, t_max)
    c = train[traj_idx, t_idx]
    y = train[traj_idx, t_idx + 1]
    y2 = train[traj_idx, t_idx + 2]
    return c, y, y2


def _front_weight(y: jnp.ndarray, strength: float = 10.0) -> jnp.ndarray:
    """Shock-region weight for the first channel. (B, 1, *S)."""
    u = y[:, 0]
    if u.ndim == 2:
        jumps = jnp.abs(u[:, 1:] - u[:, :-1])
        mask = jumps > 0.08
        mask = mask | jnp.roll(mask, 1, axis=-1) | jnp.roll(mask, -1, axis=-1)
        mask = jnp.pad(mask, ((0, 0), (1, 1)))
        return 1.0 + strength * mask[:, None, :]
    gx = jnp.abs(u[:, :, 1:] - u[:, :, :-1])
    gy = jnp.abs(u[:, 1:, :] - u[:, :-1, :])
    mag = jnp.zeros_like(u)
    mag = mag.at[:, :, 1:].set(gx)
    mag = mag.at[:, 1:, :].set(mag[:, 1:, :] + gy)
    mask = mag > 0.2 * jnp.max(mag)
    return 1.0 + strength * mask[:, None]


def _make_mse_step(apply_fn, optimizer):
    """Two-step BPTT MSE step."""

    @eqx.filter_jit
    def step(params, opt_state, c, y, y2):
        def loss_fn(p):
            pred1 = apply_fn(p, c)
            pred2 = apply_fn(p, pred1)
            return jnp.mean((pred1 - y) ** 2) + 0.5 * jnp.mean((pred2 - y2) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def _make_frontloss_step(apply_fn, optimizer):
    """Two-step BPTT MSE with shock-region weighting."""

    @eqx.filter_jit
    def step(params, opt_state, c, y, y2):
        def loss_fn(p):
            pred1 = apply_fn(p, c)
            pred2 = apply_fn(p, pred1)
            w1 = _front_weight(y)
            w2 = _front_weight(y2)
            return jnp.mean(w1 * (pred1 - y) ** 2) + 0.5 * jnp.mean(w2 * (pred2 - y2) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def _make_flow_step(optimizer):
    """Conditional flow-matching training step for FlowNO."""

    @eqx.filter_jit
    def step(params, opt_state, c, y, key):
        k1, k2 = jax.random.split(key)
        s = jax.random.uniform(k1, ())
        eps = jax.random.normal(k2, y.shape)
        x_s = (1.0 - s) * eps + s * y

        def loss_fn(p):
            v = p(c, x_s, s)
            return jnp.mean((v - (y - eps)) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def _make_diff_step(optimizer, alphas):
    """Conditional denoising training step for DiffNO."""

    @eqx.filter_jit
    def step(params, opt_state, c, y, key):
        k1, k2 = jax.random.split(key)
        t = jax.random.randint(k1, (), 1, DIFF_T + 1)
        eps = jax.random.normal(k2, y.shape)
        x_t = jnp.sqrt(alphas[t]) * y + jnp.sqrt(1.0 - alphas[t]) * eps

        def loss_fn(p):
            x0 = p(c, x_t, t / DIFF_T)
            return jnp.mean((x0 - y) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def _make_hyflow_stage2_step(optimizer):
    """Flow-matching corrector step against the detached backbone."""

    @eqx.filter_jit
    def step(params, opt_state, c, y, key):
        s = jax.random.uniform(key, ())

        def loss_fn(p):
            u_bb = jax.lax.stop_gradient(p.backbone_call(c))
            x_s = (1.0 - s) * u_bb + s * y
            v = p.velocity(c, x_s, s)
            return jnp.mean((v - (y - u_bb)) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step


def _make_flow_advance(n_steps: int = FLOW_STEPS):
    """Stochastic Euler integration from noise for FlowNO."""

    @eqx.filter_jit
    def advance(model, c, key):
        x = jax.random.normal(key, c.shape)
        for k in range(n_steps):
            s = (k + 1) / n_steps
            x = x + model(c, x, s) / n_steps
        return x

    return advance


def _make_diff_advance(alphas, n_steps: int = DIFF_STEPS):
    """DDIM sampling for DiffNO."""
    ts = np.unique(np.linspace(DIFF_T, 1, n_steps + 1, dtype=int))[::-1]

    @eqx.filter_jit
    def advance(model, c, key):
        x = jax.random.normal(key, c.shape)
        for t in ts:
            x0 = model(c, x, t / DIFF_T)
            eps_theta = (x - jnp.sqrt(alphas[t]) * x0) / jnp.sqrt(1.0 - alphas[t] + 1e-8)
            x = jnp.sqrt(alphas[t - 1]) * x0 + jnp.sqrt(1.0 - alphas[t - 1]) * eps_theta
        return x

    return advance


class BaseTrainer:
    """Training loop, checkpointing, and evaluation for one objective."""

    def __init__(self, model, cfg: dict, output_dir: str | Path) -> None:
        self.model = model
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    def _optimizer(self) -> optax.GradientTransformation:
        return optax.masked(
            optax.chain(optax.clip_by_global_norm(1.0), optax.adam(float(self.cfg["training"]["lr"]))),
            jax.tree.map(lambda _: True, self.model),
        )

    def fit(self, train: jnp.ndarray, key: jax.Array) -> None:
        """Train the model according to the configured objective."""
        objective = self.cfg["training"]["objective"]
        batch = int(self.cfg["training"]["batch"])
        optimizer = self._optimizer()
        opt_state = optimizer.init(self.model)

        if objective == "hyflow":
            self._fit_hyflow(train, key, batch, optimizer)
            return

        if objective == "flowno":
            step = _make_flow_step(optimizer)
            steps = int(self.cfg["training"]["steps_stage1"])
            for _ in range(steps):
                key, sk = jax.random.split(key)
                c, y, _ = _sample_batch(train, batch, sk)
                key, sk = jax.random.split(key)
                self.model, opt_state, loss = step(self.model, opt_state, c, y, sk)
                self.history.append({"loss": float(loss)})
            return

        if objective == "diffno":
            alphas = cosine_alphas()
            step = _make_diff_step(optimizer, alphas)
            steps = int(self.cfg["training"]["steps_stage1"])
            for _ in range(steps):
                key, sk = jax.random.split(key)
                c, y, _ = _sample_batch(train, batch, sk)
                key, sk = jax.random.split(key)
                self.model, opt_state, loss = step(self.model, opt_state, c, y, sk)
                self.history.append({"loss": float(loss)})
            return

        apply_fn = lambda m, u: m(u)
        make_step = _make_frontloss_step if objective == "frontloss" else _make_mse_step
        step = make_step(apply_fn, optimizer)
        steps = int(self.cfg["training"]["steps_stage1"])
        for _ in range(steps):
            key, sk = jax.random.split(key)
            c, y, y2 = _sample_batch(train, batch, sk)
            self.model, opt_state, loss = step(self.model, opt_state, c, y, y2)
            self.history.append({"loss": float(loss)})

    def _fit_hyflow(self, train, key, batch, optimizer) -> None:
        stage1 = int(self.cfg["training"]["steps_stage1"])
        stage2 = int(self.cfg["training"]["steps_stage2"])
        opt_state = optimizer.init(self.model)
        step1 = _make_mse_step(lambda m, u: m.backbone_call(u), optimizer)
        for _ in range(stage1):
            key, sk = jax.random.split(key)
            c, y, y2 = _sample_batch(train, batch, sk)
            self.model, opt_state, loss = step1(self.model, opt_state, c, y, y2)
            self.history.append({"stage": 1, "loss": float(loss)})
        step2 = _make_hyflow_stage2_step(optimizer)
        for _ in range(stage2):
            key, sk = jax.random.split(key)
            c, y, _ = _sample_batch(train, batch, sk)
            key, sk = jax.random.split(key)
            self.model, opt_state, loss = step2(self.model, opt_state, c, y, sk)
            self.history.append({"stage": 2, "loss": float(loss)})

    def save_checkpoint(self) -> Path:
        """Serialize model leaves to the output directory."""
        path = self.output_dir / "ckpt.eqx"
        eqx.tree_serialise_leaves(path, self.model)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model leaves from a checkpoint."""
        self.model = eqx.tree_deserialise_leaves(path, self.model)

    def _advance(self):
        objective = self.cfg["training"]["objective"]
        if objective == "hyflow":
            return eqx.filter_jit(lambda m, c, key: m.advance(c, key))
        if objective == "flowno":
            return _make_flow_advance()
        if objective == "diffno":
            return _make_diff_advance(cosine_alphas())
        return None

    def evaluate(self, test: np.ndarray, rollout: int) -> dict:
        """Run rollout diagnostics, save metrics JSON, and return the summary."""
        advance = self._advance()
        diag = rollout_diagnostics(self.model, test, rollout, advance=advance)
        metrics = summarize(diag)
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"{hue.g}metrics saved to {hue.q}{self.output_dir / 'metrics.json'}")
        return metrics

# HyperFlowNet Project Specification

This file is the single source of truth for module interfaces. All modules
must match these signatures exactly so that `main.py` can wire them together.

## Stack and Style

- JAX + Equinox + Optax, float32, NumPy/SciPy for offline data generation.
- Every new module starts with a two-line header:
  `# Short description` and `# Author: Shengning Wang`.
- Every public class/function/method has a docstring; tensor interfaces carry
  shape annotations like `(B, C, N)` or `(B, C, H, W)`.
- No defensive programming. No shape checks, fallbacks, or warning paths.
- Comments stay in the background; code blocks are separated by section
  dividers in long files.

## Config Schema

`config.yaml` is the only config source. `main.py` reads it directly and
returns a plain dict.

```yaml
data:
  dataset: burgers_1d        # burgers_1d | burgers_2d | sod_1d | euler_2d_riemann
  data_dir: data/raw
  n_train: 96
  n_test: 16
  grid: 256                  # 1D: N; 2D: H = W = grid
  n_steps: 200               # trajectory length (T+1 frames)
  nu: 0.02
  dt: 0.005
  channels: 1                # 1 for Burgers, 3 for Sod, 4 for 2D Euler
  seed: 0
model:
  name: HyperFlowNet         # CNN | UNet | ViT | DeepONet | FNO | WNO | UWNO | PDE-Refiner | HyperFlowNet | FlowNO | DiffNO
  backbone: UWNO
  corrector: fno             # fno | unet
  modes: 16
  width: 64
  depth: 4
  flow_steps: 8
  patch_size: 16
  heads: 4
  dim: 64
  refine_steps: 2
training:
  steps_stage1: 250
  steps_stage2: 250
  batch: 256
  lr: 1e-3
  bptt: 2
  objective: hyflow          # mse | hyflow | flowno | diffno | frontloss
  seed: 0
  checkpoint_dir: runs/exp
eval:
  rollout: 200
  metrics: [global_l2, shock_l2, tv_ratio, front_offset, shape_error, transport_error, spectrum_error]
mechanism:
  sigmas: [0.25, 0.5, 1.0, 2.0, 4.0]
  n_samples: 8192
  n_grid: 256
```

## Data Module Interfaces

All generators return arrays with dtype float32. Frame index `T` means
`n_steps` intervals and `T+1` stored frames.

### data/burgers.py

- `make_burgers_1d(n_train=96, n_test=16, n_grid=256, n_steps=200, nu=0.02, dt=0.005, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  Returns `x (N,)`, `train (TRAIN, T+1, N)`, `test (TEST, T+1, N)`.
  Port the pseudospectral IFRK4 generator from the gepro prototype.
- `make_burgers_2d(n_train=96, n_test=16, n_grid=128, n_steps=100, nu=0.01, dt=0.002, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  Returns `x (H,)`, `y (W,)`, `train (TRAIN, T+1, H, W)`, `test (TEST, T+1, H, W)`.
  Pseudospectral IFRK4 with periodic boundaries and smooth random initial fields.

### data/euler.py

- `make_sod_1d(n_train=96, n_test=16, n_grid=512, n_steps=100, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  Returns `x (N,)`, `train (TRAIN, T+1, N, C=3)`, `test (TEST, T+1, N, C=3)`.
  Conserved variables order `[rho, rho*u, E]`, gamma = 1.4, MUSCL-Rusanov FV,
  random left/right states.
- `make_euler_2d_riemann(n_train=96, n_test=16, n_grid=128, n_steps=100, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  Returns `x (H,)`, `y (W,)`, `train (TRAIN, T+1, C=4, H, W)`, `test (TEST, T+1, C=4, H, W)`,
  and `configs (TRAIN+TEST,)` (integers 3 or 6).
  Conserved variables `[rho, rho*u, rho*v, E]`, Rusanov + minmod MUSCL,
  zero-gradient boundaries, configurations 3 and 6 from the 2D Riemann
  problem family with small random state perturbations.

### data/datasets.py

- `generate_dataset(cfg: dict) -> dict`
  Calls the generator selected by `cfg["data"]["dataset"]`, computes per-channel
  normalization statistics on train, and returns
  `{"train", "test", "x", "y", "mean", "std", "meta"}` where train/test are
  scaled to `(B, T+1, C, *S)` for all datasets (Burgers has C=1 and gets an
  extra channel axis).
- `save_dataset(cfg: dict, data: dict) -> Path`
  Saves `data/raw/<dataset>_<grid>.npz` with arrays `train`, `test`, `x`, `y`,
  `mean`, `std`, and `meta` (json-compatible dict). Returns the path.
- `load_dataset(cfg: dict) -> dict`
  Loads the npz, applies the stored normalization to raw arrays if present,
  and returns the same dict shape as `generate_dataset`.

## Model Interfaces

All operators are `eqx.Module` subclasses with `__call__(x) -> y`.
Input `x (B, C, *S)`; output `y (B, C, *S)`. `S` is `N` for 1D and `H, W` for 2D.

### models/__init__.py

- `make_model(name: str, key: jax.Array, c_in: int = 1, ndim: int = 1, cfg: dict | None = None) -> eqx.Module`
  Registry for `CNN`, `UNet`, `ViT`, `DeepONet`, `FNO`, `WNO`, `UWNO`,
  `PDE-Refiner`, `FlowNO`, `DiffNO`, `HyperFlowNet`. Unknown names raise
  `ValueError`.

### models/operators.py

Implement 1D and 2D versions of each baseline in one file. Width/modes/depth
come from `cfg["model"]`. Keep parameter counts modest (width 32-64).
- `CNN`: residual conv stack with GroupNorm and 0.1 output scaling (1D and 2D).
- `UNet`: small U-Net with strided pooling (1D and 2D).
- `ViT`: patch embedding, pre-norm transformer encoder, unpatchify; patch size
  from cfg.
- `DeepONet`: conv branch net + MLP trunk evaluated on the fixed grid.
- `FNO`: spectral convolution + local mixing (rfft for 1D, rfft2 for 2D).
- `WNO`: wavelet neural operator using Haar DWT (1D and 2D separable).
- `UWNO`: U-Net with Haar down/up sampling (1D and 2D).
- `PDE-Refiner`: FNO backbone plus a UNet refiner applied `refine_steps` times.

### models/corrector.py

- `FlowNO(key, c_in, modes, width, ndim)` with `__call__(c, x, s) -> v`.
  Inputs `c (B, C, *S)`, `x (B, C, *S)`, `s` scalar; internally concatenates
  `[c, x, s_ch]` giving `2C+1` channels.
- `FlowUNet(key, c_in, ndim)` with the same `__call__` signature.
- `make_advance(corrector, flow_steps: int)`: returns a jitted function
  `advance(c, u0, key)` performing deterministic Euler integration
  `u_{k} = u_{k-1} + v(c, u_{k-1}, s_k)/flow_steps`.

### models/hyperflownet.py

- `HyperFlowNet(key, c_in, ndim, cfg) -> eqx.Module` with fields
  `backbone: UWNO`, `corrector: FlowNO | FlowUNet`, `flow_steps`.
- Methods: `backbone_call(x) -> u_bb`, `velocity(c, x, s) -> v`,
  `advance(c, key) -> u` (backbone prediction followed by deterministic
  Euler correction), and `__call__(x) -> u` (same as advance with a dummy key).

## Trainer Interface (trainer.py)

- `class BaseTrainer`: plain class, not an eqx module.
  - `__init__(self, model, cfg: dict, output_dir: Path)`
  - `fit(self, train: jnp.ndarray, key: jax.Array) -> None`
  - `save_checkpoint(self)`, `load_checkpoint(self, path: Path)`
  - `evaluate(self, test: np.ndarray, rollout: int) -> dict`
- Objective dispatch happens inside `fit` based on `cfg["training"]["objective"]`:
  - `mse`: BPTT-2 MSE on the full model.
  - `hyflow`: stage 1 trains `backbone` with BPTT-2 MSE; stage 2 freezes the
    backbone and trains the corrector with straight-path flow matching
    `x_s = (1-s) u_bb + s y`, velocity target `y - u_bb`.
  - `flowno`: single flow operator trained with flow matching, stochastic
    sampling at inference.
  - `diffno`: conditional denoiser with cosine schedule and DDIM sampling.
  - `frontloss`: MSE BPTT-2 with shock-region weighting on the backbone.
- All variants share the same model instance and optimizer settings
  (Adam 1e-3, global grad norm clip 1.0).

## Metrics Interface (utils/metrics.py)

- `rollout_diagnostics(model, test, steps, advance=None) -> dict[str, np.ndarray]`
  Per-step arrays: `global`, `shock`, `shape`, `transport`, `tv_ratio`,
  `edge_offset`, `spectrum`. 1D semantics are ported from gepro. 2D uses
  gradient-magnitude front masks, TV over the front band, and 2D spectral
  magnitude error; `edge_offset` is skipped in 2D (zeros).
- `summarize(diag) -> dict` returns means plus `global_step60`.
- `shock_mask(truth, width=3, threshold=0.08)` and `front_offset_1d`,
  `tv_ratio_1d`, `spectrum_error_1d` helpers.

## Mechanism Interface (utils/mechanism.py)

- `run_synthetic_ce(sigmas, n_samples, n_grid, seed, out_dir) -> dict`
  Generates step functions with Gaussian-random positions, computes the
  analytic conditional expectation ramp width (10%-90%) for each sigma,
  fits a small CNN on finite samples, measures the fitted ramp width, and
  saves a figure. Returns `{"sigmas", "analytic_widths", "fitted_widths", "figure"}`.
- `fit_synthetic_mse(n_samples, n_grid, sigma, seed, steps=200) -> float`
  Returns the measured 10%-90% width of the MSE-trained surrogate.

## main.py Entry Points

- `python main.py generate --config config.yaml`
- `python main.py train --config config.yaml [--model NAME] [--objective OBJ]`
- `python main.py evaluate --config config.yaml --checkpoint runs/exp/ckpt.eqx`
- `python main.py mechanism --config config.yaml`

`train` writes `runs/<experiment>/metrics.json`, `history.json`, checkpoint,
and figures. `mechanism` writes `runs/mechanism/*.json` and figures.

## Smoke Tests (must pass)

- Each dataset generator runs with `n_train=4, n_test=2, grid=32 (1D) / 16 (2D),
  n_steps=10` and returns finite arrays of the documented shapes.
- Every 1D and 2D model runs one forward and one backward pass on a small
  batch and returns the documented shape.
- `main.py train --model FNO` with 20 training steps completes and writes
  `metrics.json`.
- `main.py mechanism` with small sigma range completes and writes JSON + PNG.

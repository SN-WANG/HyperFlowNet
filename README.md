# HyperFlowNet

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/HyperFlowNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**HyperFlowNet** is the JAX/Equinox transient surrogate repository in the WSNet family. It keeps the autoregressive rollout and mechanism-analysis workflow local to this repository while reusing the lightweight configuration, logging, seeding, and utility style of [WSNet](https://github.com/SN-WANG/WSNet).

## 📌 Overview

HyperFlowNet keeps the full workflow for this task in one place: dataset generation, model training, mechanism experiments, rollout evaluation, metric export, and paper drafting.

The current scope includes:

- pseudospectral 1D/2D Burgers data generation
- MUSCL-Rusanov 1D Sod and 2D Euler Riemann data generation
- autoregressive rollout learning with a UWNO backbone
- deterministic flow-matching residual correction on a frozen backbone
- conditional-expectation smearing analysis for discontinuous dynamics
- transport/shape error decomposition across neural operator baselines
- five training objectives: MSE, HyperFlowNet, FlowNO, DiffNO, and front-loss

## ✨ Highlights

- `HyperFlowNet` as a deterministic flow-corrected autoregressive surrogate
- YAML-driven experiments through `config.yaml`
- Two-stage training with a cached backbone shared by all corrector variants
- Straight-path flow matching on the residual between backbone prediction and truth
- Deterministic Euler integration at inference to avoid sampling-noise accumulation
- 1D/2D baselines: CNN, U-Net, ViT, DeepONet, FNO, WNO, UWNO, PDE-Refiner
- Synthetic conditional-expectation experiments with analytic scaling laws
- Per-step global, shock, TV, offset, transport/shape, and spectrum metrics
- `.npz` dataset persistence with per-channel normalization

## 🧱 Repository Layout

```text
HyperFlowNet/
├── main.py                  # Unified entry point: generate / train / evaluate / mechanism
├── config.yaml              # Experiment configuration
├── trainer.py               # Training loop with five objectives
├── models/
│   ├── __init__.py
│   ├── blocks.py
│   ├── operators.py
│   ├── corrector.py
│   └── hyperflownet.py
├── data/
│   ├── __init__.py
│   ├── burgers.py
│   ├── euler.py
│   └── datasets.py
├── utils/
│   ├── __init__.py
│   ├── hue_logger.py
│   ├── seeder.py
│   ├── metrics.py
│   ├── mechanism.py
│   ├── plotting.py
│   └── sweeper.py
├── README.md
├── LICENSE
```

## 🚀 Running Experiments

### Clone the repository

```bash
git clone https://github.com/SN-WANG/HyperFlowNet.git
cd HyperFlowNet
```

### Install the dependencies you need

```bash
pip install jax equinox optax pyyaml numpy scipy matplotlib
```

### Generate data

```bash
python main.py generate --config config.yaml
```

Generated datasets are saved under `data/raw/` as `.npz` files.

### Train a baseline

```bash
python main.py train --config config.yaml --model FNO --objective mse
```

### Train HyperFlowNet

```bash
python main.py train --config config.yaml --model HyperFlowNet --objective hyflow
```

### Evaluate a checkpoint

```bash
python main.py evaluate --config config.yaml --checkpoint runs/exp/ckpt.eqx
```

### Run the mechanism experiments

```bash
python main.py mechanism --config config.yaml
```

## 📂 Expected Data Format

The default workflow generates trajectories and stores them in a compressed NumPy archive:

```text
data/raw/
├── burgers_1d_256.npz
├── burgers_2d_128.npz
├── sod_1d_512.npz
└── euler_2d_riemann_128.npz
```

Each archive contains:

- `train`: tensor of shape `(TRAIN, T+1, C, *S)` (normalized)
- `test`: tensor of shape `(TEST, T+1, C, *S)` (normalized)
- `x`, `y`: grid coordinates
- `mean`, `std`: per-channel normalization statistics
- `meta_json`: dataset metadata, including grid, steps, channels, and config

`C` is 1 for Burgers, 3 for Sod, and 4 for 2D Euler. `S` is `N` for 1D and `H, W` for 2D.

## 🧾 Workflow Outputs

```text
runs/
├── <experiment>/
│   ├── ckpt.eqx
│   ├── history.json
│   ├── metrics.json
│   └── rollout.png
└── mechanism/
    ├── synthetic_ce.json
    └── synthetic_ce.png
```

Checkpoints serialize the full Equinox model. The backbone is cached separately so that all corrector variants and objective ablations share the same pretrained backbone.

## 🔗 Relationship to WSNet

HyperFlowNet follows the repository organization and utility style of [WSNet](https://github.com/SN-WANG/WSNet). WSNet keeps reusable core modules, while HyperFlowNet keeps the PDE dataset pipeline, the JAX/Equinox model zoo, and the mechanism-analysis workflow in one place.

## 📚 Citation

If this repository is useful in your work, please cite it as a software project.

```bibtex
@software{hyperflownet2026,
  author = {Shengning Wang},
  title = {HyperFlowNet},
  year = {2026},
  url = {https://github.com/SN-WANG/HyperFlowNet}
}
```

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

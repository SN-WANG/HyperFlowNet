# HyperFlowNet

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/HyperFlowNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**HyperFlowNet** is a PyTorch research repository for spatiotemporal prediction of discontinuous physical fields (shocks, interfaces, fronts). It studies why autoregressive neural operators smear discontinuities and proposes a path-level flow-matching fix, following the mechanism-analysis layout of [WSNet](https://github.com/SN-WANG/WSNet).

## Overview

The repository keeps the full workflow in one place: mechanism data generation, model training, mechanism experiments, rollout evaluation, and metric export.

Current scope:

- synthetic step-family data for the conditional-expectation smearing mechanism
- 1D Burgers, 1D Sod, and 2D Euler Riemann data generators
- Neptuna (2D-SABW / 2D-SDBA) engineering data loader
- nine baselines: FNO, DeepONet, U-Net, ViT, WNO, UWNO, CFM, OT-CFM, PDE-Refiner
- HyperFlowNet: single-stage, path-level flow-matching surrogate with front transport
- conditional-expectation, scaling-law, and flow-matching mechanism experiments
- layered discontinuity metrics (front width, edge offset, TV ratio, shape/transport decomposition)

## Highlights

- PyTorch only, single-GPU friendly (RTX 5090), AMP bf16 support
- YAML-driven experiments through `config.yaml`
- path-family ablations: straight-path CFM, OT-coupled CFM, transport-path HyperFlowNet
- deterministic probability-flow ODE inference (no sampling noise accumulation)
- mechanism experiments with analytic scaling laws (2.56σ ramp width)
- per-step global, shock, TV, offset, transport/shape, and spectrum metrics

## Repository Layout

```text
HyperFlowNet/
├── main.py                  # Entry point: generate / train / evaluate / mechanism / benchmark
├── config.yaml              # Experiment configuration
├── trainer.py               # Training objectives, Sinkhorn coupling, evaluation
├── models/
│   ├── __init__.py          # Model registry
│   ├── blocks.py            # Shared 1D/2D building blocks
│   ├── operators.py         # FNO, DeepONet, U-Net, ViT, WNO, UWNO
│   ├── velocity.py          # FlowUNet / FlowNO conditional velocity networks
│   ├── pde_refiner.py       # PDE-Refiner baseline
│   └── hyperflownet.py      # HyperFlowNet (five components)
├── data/
│   ├── __init__.py
│   ├── synthetic.py         # Mechanism synthetic data
│   ├── burgers.py           # 1D Burgers generator
│   ├── euler.py             # 1D Sod and 2D Euler Riemann generators
│   ├── neptuna.py           # Neptuna loader (bubble / droplet)
│   └── datasets.py          # Registry, normalization, persistence
├── utils/
│   ├── __init__.py
│   ├── hue_logger.py        # from WSNet
│   ├── scaler.py            # from WSNet
│   ├── seeder.py            # from WSNet
│   ├── sweeper.py           # from WSNet
│   ├── metrics.py           # Discontinuity-focused rollout diagnostics
│   ├── mechanism.py         # Mechanism experiments
│   └── plotting.py          # Mechanism and rollout figures
├── AGENTS.md                # Project notes for AI agents
├── README.md
└── LICENSE
```

## Running Experiments

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate mechanism datasets

```bash
python main.py generate --config config.yaml
```

Generated datasets are saved under `data/` as `.npz` files.

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
python main.py evaluate --config config.yaml --checkpoint runs/exp/ckpt.pt
```

### Run mechanism experiments

```bash
python main.py mechanism --config config.yaml
```

### Run a benchmark sweep

```bash
python main.py benchmark --config config.yaml
```

## Expected Data Format

Mechanism datasets are generated locally and stored under `data/`:

```text
data/
├── burgers_1d_256.npz
├── sod_1d_512.npz
└── euler_2d_riemann_128.npz
```

Each archive contains:

- `train`, `test`: normalized trajectories of shape `(N, T+1, C, *S)`
- `x`, `y`: grid coordinates
- `mean`, `std`: per-channel normalization statistics
- `meta_json`: dataset metadata

Engineering data (Neptuna) is loaded from the absolute paths in `config.yaml` (`data.neptuna.bubble.path`, `data.neptuna.droplet.path`); the loader reads `train.h5` / `test.h5` directly from those directories.

## Workflow Outputs

```text
runs/
├── <experiment>/
│   ├── ckpt.pt
│   ├── history.json
│   ├── metrics.json
│   └── rollout.png
└── mechanism/
    ├── synthetic_ce.json
    └── synthetic_ce.png
```

## Relationship to WSNet

HyperFlowNet follows the repository organization and utility style of [WSNet](https://github.com/SN-WANG/WSNet). The `utils/` package (logging, scaling, seeding, sweeping) is shared with WSNet; the PDE data pipeline, model zoo, and mechanism-analysis workflow live in this repository.

## Citation

If this repository is useful in your work, please cite it as a software project.

```bibtex
@software{hyperflownet2026,
  author = {Shengning Wang},
  title = {HyperFlowNet},
  year = {2026},
  url = {https://github.com/SN-WANG/HyperFlowNet}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

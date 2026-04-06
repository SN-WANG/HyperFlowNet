# HyperFlowNet

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/HyperFlowNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**HyperFlowNet** is the CFD-focused repository in the WSNet family. It inherits the core training style, scaling tools, and utility conventions from [WSNet](https://github.com/SN-WANG/WSNet), while concentrating on lightweight irregular-mesh autoregressive flow prediction from Fluent-style simulation data.

## 📌 Overview

HyperFlowNet keeps the full workflow for this task in one place:
data loading, memory probing, rollout training, inference, and visualization.

The current scope includes:

- irregular-mesh spatio-temporal flow prediction
- shared data pipeline for train / val / test preparation
- curriculum autoregressive rollout training
- rollout metrics, plots, and animation rendering
- GPU memory probing before full training

## ✨ Highlights

- `HyperFlowNet` as the single maintained model
- A dedicated `HyperFlowTrainer` built on top of WSNet-style `BaseTrainer`
- A lightweight coordinate encoder together with sinusoidal time encoding
- Shared-assignment slice linear attention for irregular meshes
- Step-wise BPTT rollout training with rollout curriculum, teacher forcing decay, and noise injection
- Fluent-style sequence loading, caching, sliding-window augmentation, and normalization
- Built-in rollout metrics, training curves, error heatmaps, and animation rendering
- GPU memory probing before full training

## 🧱 Repository Layout

```text
HyperFlowNet/
├── main.py                  # Unified entry point for probe / train / infer
├── config.py                # Command-line arguments and experiment configuration
├── models/
│   └── hflownet.py
├── data/
│   ├── flow_data.py
│   ├── boundary.py
│   ├── flow_plot.py
│   └── flow_vis.py
├── training/
│   ├── base_trainer.py
│   └── hyperflow_trainer.py
├── utils/
│   ├── scaler.py
│   ├── hue_logger.py
│   ├── seeder.py
│   └── sweeper.py
├── README.md
└── LICENSE
```

## 🚀 Running Experiments

### Clone the repository

```bash
git clone https://github.com/SN-WANG/HyperFlowNet.git
cd HyperFlowNet
```

### Probe GPU memory before training

```bash
python main.py --mode probe
```

### Train HyperFlowNet

```bash
python main.py \
  --mode train \
  --data_dir ./dataset \
  --output_dir ./runs/hyperflownet
```

### Run inference and generate visualizations

```bash
python main.py \
  --mode infer \
  --data_dir ./dataset \
  --output_dir ./runs/hyperflownet
```

### Run the full workflow

```bash
python main.py \
  --mode probe train infer \
  --data_dir ./dataset \
  --output_dir ./runs/hyperflownet
```

## ⚙️ Workflow

`main.py` keeps the workflow in four connected parts:

1. `data_pipeline`: build train / val / test splits, fit scalers, and detect hard boundary conditions.
2. `probe_pipeline`: run one training-like rollout step and estimate peak CUDA memory.
3. `train_pipeline`: train HyperFlowNet with rollout curriculum on the shared processed loaders.
4. `inference_pipeline`: restore artifacts from `ckpt.pt`, run autoregressive rollout, and export metrics and figures.

## 🧠 Model Summary

The current `HyperFlowNet` uses:

- a lightweight spatial encoder: `coords`, learned low-frequency features, and learned Fourier features
- sinusoidal temporal encoding controlled by `time_features`
- shared-assignment slice linear attention to compress node tokens into slice tokens and mix them in linear-attention form
- pre-norm residual blocks with token-wise feed-forward networks

The model predicts one step at a time. Full rollout logic stays in `HyperFlowTrainer`, not inside the model forward pass.

## 🏋️ Training Summary

`HyperFlowTrainer` follows a WSNet-style rollout training path:

- weighted step-wise NMSE loss, with later rollout steps receiving larger weights
- epoch-based rollout curriculum from 1 step up to `max_rollout_steps`
- Gaussian noise injection during training rollout
- teacher forcing that decays during curriculum progression
- AdamW optimizer with cosine annealing learning-rate schedule

Important configuration arguments include:

- `--coords_features`: number of learned spatial encoding features
- `--time_features`: number of temporal sinusoidal frequency pairs
- `--num_slices`: number of slice tokens
- `--max_rollout_steps`: curriculum ceiling for rollout length
- `--teacher_forcing_init`, `--teacher_forcing_decay`, `--teacher_forcing_floor`
- `--use_hard_bc` and `--velocity_threshold`

## 📂 Expected Data Format

```text
dataset/
├── raw_data/
│   ├── case_0001/
│   │   ├── frame_0000.txt
│   │   ├── frame_0001.txt
│   │   └── ...
│   ├── case_0002/
│   └── ...
├── case_0001.pt
├── case_0002.pt
└── ...
```

Each raw text file is expected to follow the Fluent-style convention used by `FlowData`:

- 2D case: `[Index, x, y, P, Vx, Vy, T]`
- 3D case: `[Index, x, y, z, P, Vx, Vy, Vz, T]`

## 🔗 Relationship to WSNet

HyperFlowNet is built on top of [WSNet](https://github.com/SN-WANG/WSNet).
WSNet keeps the reusable core modules, while HyperFlowNet keeps the CFD-specific data pipeline, rollout workflow, and experiment entry points.

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

# HyperFlowNet

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/HyperFlowNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**HyperFlowNet** is the transient CFD repository in the WSNet family. It keeps the shock-wave rollout workflow local to this repository while reusing the lightweight training, normalization, and utility style of [WSNet](https://github.com/SN-WANG/WSNet).

## 📌 Overview

HyperFlowNet keeps the full workflow for this task in one place:
dataset handling, memory probing, model training, case-wise inference, visualization, and metric export.

The current scope includes:

- fixed-mesh shock-wave flow simulation
- autoregressive rollout learning on Fluent-style CFD sequences
- graph-injected slice attention for irregular mesh nodes
- hard no-slip wall boundary condition enforcement
- case-wise comparison, focused-region, and digital-twin visualization

## ✨ Highlights

- `HyperFlowNet` as a spatio-temporal neural operator for shock-wave rollout prediction
- Four graph modes through `--graph_mode`: `bias`, `assign`, `shock_bias`, and `shock_assign`
- Shock-aware graph injection based on local graph residuals
- Hard boundary condition projection through `bc` during training and inference rollout
- Weighted autoregressive NMSE training with rollout curriculum and noise decay
- Deterministic first-frame reconstruction from `case_<label>` and mesh coordinates
- `FlowTwin` axisymmetric `Vy` rendering for 2D-to-3D digital-twin videos

## 🧱 Repository Layout

```text
HyperFlowNet/
├── main.py                  # Unified entry point for probe / train / infer
├── config.py                # Command-line arguments and experiment configuration
├── models/
│   └── hflownet.py
├── data/
│   ├── boundary.py
│   ├── flow_data.py
│   ├── flow_metrics.py
│   ├── flow_twin.py
│   ├── flow_vis.py
│   └── initial_state.py
├── training/
│   ├── base_trainer.py
│   └── hflow_trainer.py
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

### Install the dependencies you need

```bash
pip install numpy scipy torch matplotlib tqdm pyvista pillow
```

MP4 rendering uses system `ffmpeg`. Make sure `ffmpeg` is on `PATH`, or set `FFMPEG_EXE`.

### Probe GPU memory before training

```bash
python main.py --mode probe --data_dir ./dataset --output_dir ./runs
```

### Train HyperFlowNet

```bash
python main.py --mode train --data_dir ./dataset --output_dir ./runs
```

### Run a graph-mode ablation

```bash
python main.py --mode train --graph_mode shock_bias --data_dir ./dataset --output_dir ./runs_shock_bias
```

### Run inference and generate visualizations

```bash
python main.py --mode infer --data_dir ./dataset --output_dir ./runs
```

### Run the full workflow

```bash
python main.py --mode probe train infer --data_dir ./dataset --output_dir ./runs
```

## 📂 Expected Data Format

The default workflow expects 2D cases named `case_<label>`, where the trailing number is used as the operating-condition label.
HyperFlowNet can read cached `.pt` cases directly or raw Fluent-style folders that are cached automatically.

### Cached case format

```text
dataset/
├── case_4500.pt
├── case_5000.pt
└── ...
```

Each case file should be a PyTorch dictionary containing:

- `states`: tensor of shape `(T, N, 4)` with channel order `[Vx, Vy, P, T]`
- `coords`: tensor of shape `(N, 2)`

### Raw Fluent-style format

```text
dataset/
├── raw_data/
│   ├── case_4500/
│   │   ├── frame_0000.txt
│   │   ├── frame_0001.txt
│   │   └── ...
│   ├── case_5000/
│   └── ...
```

Each raw text file is expected to follow the 2D convention used by `FlowData`:

- source columns: `[Index, x, y, P, Vx, Vy, T]`
- cached state order: `[Vx, Vy, P, T]`

## 🧾 Workflow Outputs

```text
runs/
├── ckpt.pt
├── best.pt
├── history.json
├── metrics.json
├── <label>_pred.pt
├── <label>_full.mp4
├── <label>_focus_vy.mp4
└── <label>_twin_vy.mp4
```

Checkpoints store model arguments, graph settings, and `bc` state in `params`, while state and coordinate scalers are stored separately in `scaler_state_dict`.

## 🔗 Relationship to WSNet

HyperFlowNet is built on top of [WSNet](https://github.com/SN-WANG/WSNet).
WSNet keeps the reusable core modules, while HyperFlowNet keeps the CFD dataset pipeline, task-specific model entry point, and experiment workflow.

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

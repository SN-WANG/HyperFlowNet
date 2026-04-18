# HyperFlowNet

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/HyperFlowNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**HyperFlowNet** is the irregular-mesh CFD repository in the WSNet family. It inherits the same lightweight
training, normalization, and utility foundations from [WSNet](https://github.com/SN-WANG/WSNet), while focusing on
autoregressive flow prediction from Fluent-style simulation data.

## 📌 Overview

HyperFlowNet keeps the full workflow for this task in one place:
dataset handling, memory probing, model training, case-wise inference, visualization, and metric export.

The current scope includes:

- irregular-mesh spatio-temporal flow prediction
- autoregressive rollout learning on CFD sequences
- end-to-end probe, train, and infer workflows
- case-wise visualization and diagnostic metrics

## ✨ Highlights

- `HyperFlowNet` as the main model for irregular-mesh autoregressive CFD prediction
- Unified `main.py` workflow for `probe`, `train`, and `infer`
- Fluent-style dataset loading with cached tensor support
- Standardized state normalization and coordinate normalization
- Rollout-based training with checkpointing and metric export
- Case-wise visualization for ground truth, prediction, and error

## 🧱 Repository Layout

```text
HyperFlowNet/
├── main.py                  # Unified entry point for probe / train / infer
├── config.py                # Command-line arguments and experiment configuration
├── models/
│   └── hflownet.py
├── data/
│   ├── flow_data.py
│   ├── flow_metrics.py
│   ├── flow_plot.py
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
pip install numpy torch matplotlib tqdm pyvista
```

### Probe GPU memory before training

```bash
python main.py --mode probe --data_dir ./dataset --output_dir ./runs
```

### Train HyperFlowNet

```bash
python main.py --mode train --data_dir ./dataset --output_dir ./runs
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

HyperFlowNet can read either cached `.pt` cases directly or raw Fluent-style folders that will be cached automatically.

### Cached case format

```text
dataset/
├── case_4500.pt
├── case_5000.pt
└── ...
```

Each case file should be a PyTorch dictionary containing:

- `states`: tensor of shape `(T, N, C)`
- `coords`: tensor of shape `(N, D)`

Case names are expected to end with a numeric operating-condition label such as `4500` in `case_4500`.

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

Each raw text file is expected to follow the convention used by `FlowData`:

- 2D case: `[Index, x, y, P, Vx, Vy, T]`
- 3D case: `[Index, x, y, z, P, Vx, Vy, Vz, T]`

## 🔗 Relationship to WSNet

HyperFlowNet is built on top of [WSNet](https://github.com/SN-WANG/WSNet).
WSNet keeps the reusable core modules, while HyperFlowNet keeps the CFD dataset pipeline, task-specific model entry
point, and experiment workflow.

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

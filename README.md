# HyperFlowNet: A Spatio-Temporal Neural Operator for Shock-Wave Flow Simulation

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/HyperFlowNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**HyperFlowNet** is the transient CFD repository
in the WSNet family. It inherits the same lightweight training, normalization, and utility foundations from
[WSNet](https://github.com/SN-WANG/WSNet), while focusing on autoregressive shock-wave flow prediction on unstructured mesh.

## 📌 Overview

HyperFlowNet keeps the full workflow for this task in one place:
dataset handling, memory probing, model training, case-wise inference, MP4 visualization, and metric export.

The current scope includes:

- fixed-mesh shock-wave flow simulation
- autoregressive rollout learning on Fluent-style CFD sequences
- channel-wise state standardization and coordinate normalization
- end-to-end probe, train, and infer workflows
- case-wise visualization and diagnostic metrics

## ✨ Highlights

- `HyperFlowNet` as a spatio-temporal neural operator for shock-wave rollout prediction
- Unified `main.py` workflow for `probe`, `train`, and `infer`
- Fluent-style raw-text ingestion with cached `.pt` case support
- Standardized flow states and min-max normalized coordinates
- Local graph construction from the reference mesh
- Rollout-based NMSE training with checkpointing and metric export
- Case-wise MP4 visualization for ground truth, prediction, and relative error

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

## 🔗 Relationship to WSNet

HyperFlowNet is built on top of [WSNet](https://github.com/SN-WANG/WSNet).
WSNet keeps the reusable core modules, while HyperFlowNet keeps the CFD dataset pipeline, task-specific model entry point,
and experiment workflow.

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

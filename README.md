# HyperFlowNet

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/HyperFlowNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**HyperFlowNet** is the CFD-focused repository in the WSNet family. It inherits core training, scaling, and utility conventions from [WSNet](https://github.com/SN-WANG/WSNet), while concentrating on irregular-mesh autoregressive flow prediction from Fluent-style simulation data.

## 📌 Overview

HyperFlowNet keeps the full workflow for this task in one place:
data loading, rollout training, inference, visualization, and baseline comparison.

The current scope includes:

- irregular-mesh spatio-temporal flow prediction
- autoregressive rollout training
- baseline comparison with `GeoFNO` and `Transolver`
- rollout metrics, plots, and animation rendering

## ✨ Highlights

- `HyperFlowNet` as the main model, with `GeoFNO` and `Transolver` baselines
- Curriculum-based autoregressive rollout training with noise injection
- Optional hard boundary-condition enforcement during rollout
- Fluent-style sequence loading, caching, sliding-window augmentation, and normalization
- Built-in rollout metrics, training curves, error heatmaps, and animation rendering
- GPU memory probing before full training

## 🧱 Repository Layout

```text
HyperFlowNet/
├── main.py                  # Unified entry point for probe / train / infer
├── config.py                # Command-line arguments and experiment configuration
├── models/
│   ├── hyperflow_net.py
│   ├── geofno.py
│   └── transolver.py
├── data/
│   ├── flow_data.py
│   ├── boundary.py
│   ├── flow_plot.py
│   └── flow_vis.py
├── training/
│   ├── base_trainer.py
│   ├── rollout_trainer.py
│   ├── teacher_forcing_trainer.py
│   └── base_criterion.py
├── utils/
│   ├── scaler.py
│   ├── hue_logger.py
│   ├── seeder.py
│   └── sweep.py
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
python main.py --mode probe --model_type hyperflownet
```

### Train HyperFlowNet

```bash
python main.py \
  --mode train \
  --model_type hyperflownet \
  --trainer_type rollout \
  --data_dir ./dataset \
  --output_dir ./runs/hyperflownet
```

### Run inference and generate visualizations

```bash
python main.py \
  --mode infer \
  --model_type hyperflownet \
  --trainer_type rollout \
  --data_dir ./dataset \
  --output_dir ./runs/hyperflownet
```

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

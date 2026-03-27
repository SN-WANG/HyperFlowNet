# HyperFlowNet

HyperFlowNet is a research codebase for spatio-temporal neural operators on irregular CFD meshes. It targets autoregressive flow prediction from ANSYS Fluent style simulation data and provides a complete experimental workflow for training, long-horizon rollout inference, visualization, and baseline comparison.

This repository is organized as experiment-oriented research code rather than a packaged software library. The focus is on making the core modeling pipeline easy to read, reproduce, and extend: model definitions, CFD data handling, rollout training, boundary-condition enforcement, and post-processing utilities are all kept close to the main experiment entry point.

## Highlights

- Irregular-mesh neural operator modeling for time-dependent flow fields
- `HyperFlowNet` as the main model, with `GeoFNO` and `Transolver` baselines
- Curriculum-based autoregressive rollout training with noise injection
- Optional hard boundary-condition enforcement during rollout
- Fluent-style sequence loading, caching, sliding-window augmentation, and normalization
- Built-in rollout metrics, training curves, error heatmaps, and animation rendering
- Pre-flight GPU memory probing before full training

## Repository Layout

```text
HyperFlowNet/
├── main.py                  # Unified entry point for probe / train / infer
├── config.py                # Command-line arguments and experiment configuration
├── models/                  # HyperFlowNet and baseline neural operators
│   ├── hyperflow_net.py
│   ├── geofno.py
│   └── transolver.py
├── data/                    # CFD data loading, boundary handling, plotting, rendering
│   ├── flow_data.py
│   ├── boundary.py
│   ├── flow_plot.py
│   └── flow_vis.py
├── training/                # Training loops, loss functions, and metrics
│   ├── base_trainer.py
│   ├── rollout_trainer.py
│   ├── teacher_forcing_trainer.py
│   └── base_criterion.py
├── utils/                   # Shared utilities for scaling, logging, reproducibility
│   ├── scaler.py
│   ├── hue_logger.py
│   ├── seeder.py
│   └── sweep.py
└── LICENSE
```

## Core Components

### Models

- `models/hyperflow_net.py`
  The main architecture in this repository. HyperFlowNet combines three ideas:
  learnable Fourier feature encoding for irregular coordinates, sinusoidal temporal encoding for time-dependent rollout, and Physics Attention for compressing mesh-node interactions into slice tokens.

- `models/geofno.py`
  A geometry-aware Fourier neural operator baseline for comparison on the same CFD tasks.

- `models/transolver.py`
  A mesh-based transformer baseline built around Physics Attention, included for fair comparison against HyperFlowNet.

### Data Pipeline

- `data/flow_data.py`
  Loads raw ANSYS Fluent style text files, caches parsed tensors, supports spatial and temporal subsampling, and creates sliding-window training sequences for autoregressive learning.

- `data/boundary.py`
  Detects stationary wall nodes from training data and optionally enforces hard no-slip boundary conditions during rollout.

- `utils/scaler.py`
  Provides tensor and NumPy scalers used for feature standardization and coordinate normalization.

### Training and Evaluation

- `training/rollout_trainer.py`
  Implements weighted full-rollout training with curriculum learning and input noise injection for stable long-horizon prediction.

- `training/teacher_forcing_trainer.py`
  Implements teacher-forcing training for baseline experiments.

- `training/base_criterion.py`
  Defines the normalized MSE loss and a metrics suite with global and step-wise evaluation for each physical channel.

### Visualization and Analysis

- `data/flow_vis.py`
  Renders animated comparisons between prediction and ground truth using PyVista, including support for headless GPU environments.

- `data/flow_plot.py`
  Generates training curves, rollout error plots, error heatmaps, metric summaries, and ablation-oriented visual outputs.

### Experiment Entry Point

- `main.py`
  Orchestrates the full workflow:
  dataset loading, scaler fitting, model construction, training, checkpoint restore, rollout inference, plotting, animation generation, and GPU memory probing.

- `config.py`
  Centralizes model, optimization, curriculum, data, and runtime arguments behind a single command-line interface.

## Expected Data Format

The default dataset layout is:

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

Parsed cases are cached as `.pt` files after the first load.

## Running Experiments

The repository is centered around `main.py`, which supports three execution modes:
`probe`, `train`, and `infer`.

### Probe GPU memory before training

```bash
python main.py --mode probe --model_type hyperflownet
```

This runs a real forward-backward rollout step and reports peak VRAM usage before a full experiment.

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

### Run a baseline

```bash
python main.py \
  --mode train infer \
  --model_type transolver \
  --trainer_type teacher_forcing \
  --data_dir ./dataset \
  --output_dir ./runs/transolver
```

## Experiment Workflow

The default HyperFlowNet workflow is:

1. Parse CFD cases and build sliding-window sequences.
2. Fit feature and coordinate scalers from the training split.
3. Detect wall nodes for optional hard boundary enforcement.
4. Train with weighted autoregressive rollout loss.
5. Increase rollout horizon gradually through curriculum learning.
6. Restore checkpoints for test-time rollout prediction.
7. Generate metrics, plots, animations, and saved prediction tensors.

## Configuration

`config.py` organizes the CLI into the following groups:

- `General`: runtime mode, output paths, random seed, device
- `Data`: dataset location, spatial dimension, batch size, window slicing
- `Model Selection`: model family and trainer type
- `Architecture (Common)`: shared depth and width
- `HyperFlowNet`: slices, heads, temporal encoding, spatial encoding, hard BC
- `GeoFNO`: Fourier modes, latent grid, deformation network
- `Transolver`: slices, heads, MLP ratio, dropout
- `Optimization`: learning rate, weight decay, epoch budget, channel weights
- `Curriculum`: maximum rollout length, patience, noise schedule

## What the Code Produces

For a typical run, the output directory contains:

- `ckpt.pt`: latest checkpoint
- `best.pt`: best validation checkpoint
- `history.json`: training history
- `test_metrics.json`: per-case and per-channel evaluation metrics
- `*_pred.pt`: predicted rollout tensors
- `training_curve.png`: learning curves
- `metrics_comparison.png`: aggregated evaluation plots
- `*_rollout_error.png`: per-case rollout error plots
- `*_error_*.png`: spatial error heatmaps
- rendered animations generated by `FlowVis`

## Code Organization

The repository follows a straightforward research-code layout:

- `main.py` is the single entry point for the full experiment pipeline.
- `config.py` defines the runtime and model configuration space.
- `models/` contains the main architecture and baseline neural operators.
- `data/` contains CFD-specific loading, preprocessing, plotting, and rendering code.
- `training/` contains the rollout logic, baseline trainer, losses, and metrics.
- `utils/` contains shared helpers such as scaling, logging, seeding, and repository utilities.

For a research repository, these names are clear and conventional. They make it easy to navigate the code by function without introducing extra packaging layers that are not needed for day-to-day experimentation.

## Dependency Context

This project is developed alongside `WSNet` and reuses some of its broader core abstractions. HyperFlowNet itself keeps the flow-specific research logic local: the dataset loader, neural operator models, rollout procedure, visualization pipeline, and experiment-facing command-line configuration.

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

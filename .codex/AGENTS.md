# HyperFlowNet Project AGENTS

## Project Snapshot

- HyperFlowNet is `HyperFlowNet: A Spatio-Temporal Neural Operator for Shock-Wave Flow Simulation`.
- It is the transient CFD task repository in the WSNet family.
- The current default workflow is autoregressive shock-wave flow prediction on unstructured mesh.
- Keep this repository focused on task-facing code: data ingestion, rollout training, inference, visualization, metrics, and fast local model iteration.
- Treat the current repository code as the source of truth when older notes, slides, reports, or README text disagree with it.

## Active Code Path

- The active workflow is `main.py -> FlowData -> HyperFlowTrainer -> HyperFlowNet -> Metrics / FlowVis`.
- `config.py` is the canonical source for default command-line options and experiment knobs.
- Keep agent notes limited to modules and mechanisms that exist in the current repository code.

## Dataset Contract

- `FlowData.discover_cases()` finds root-level `case_*.pt` files and `raw_data/case_*` directories.
- Cached cases are PyTorch dictionaries with `states` and `coords`.
- The default workflow uses 2D states with channel order `[Vx, Vy, P, T]` and coordinates shaped `(N, 2)`.
- Raw 2D Fluent-style rows are parsed as `[Index, x, y, P, Vx, Vy, T]` and cached as `[Vx, Vy, P, T]`.
- `FlowData` has a 3D parsing branch, but the current default `main.py` workflow and `initial_state_from_label()` are 2D / four-channel oriented. Do not describe 3D as a validated full workflow unless the code is updated.
- Case labels are parsed from the numeric suffix in names such as `case_4500`.
- Train and validation cases are augmented with sliding temporal windows. Test cases keep full sequences for rollout evaluation.
- The default split behavior is deterministic under seed 42, with `split_counts=(2, 1)` for validation and test case counts.

## Preprocessing And Runtime Flow

- `data_pipeline()` fits `StandardScalerTensor` on training states and `MinMaxScalerTensor(norm_range="bipolar")` on training coordinates.
- Training and validation batches contain standardized `seq`, normalized `coords`, `t0_norm`, and `dt_norm`. Labels are not passed into the trainer.
- Probe and train build one local graph from the first normalized training coordinate set.
- Inference loads `ckpt.pt`, restores saved scalers and model parameters, builds one local graph from the first test coordinate set, constructs the initial physical state with `initial_state_from_label()`, then calls `model.predict()`.
- Inference writes `<label>_pred.pt`, `metrics.json`, and MP4 visualizations through `FlowVis`.
- The current graph is a fixed reference graph per run. Do not claim moving-mesh, variable-cardinality, or per-case graph rebuilding support unless the implementation changes.

## Model And Trainer Contract

- `build_local_graph()` constructs a kNN sparse local operator and an undirected edge list from normalized coordinates.
- `HyperFlowNet` is the local spatio-temporal neural operator implementation. It predicts one next state from `(inputs, coords, t_norm)` and returns `(pred_state, weight_bank)`.
- The model combines current node states, learnable Fourier coordinate encoding, sinusoidal time encoding, frontier-aware slice attention, and residual feed-forward blocks.
- `HyperFlowTrainer` uses AdamW, cosine annealing, channel-weighted NMSE, autoregressive rollout loss, rollout noise injection during training, and frontier regularization from early slice assignments.
- Rollout curriculum advances by `rollout_patience` and `max_rollout_steps`, not by validation-loss triggers.
- Validation loss is computed through the same rollout loss path with model evaluation mode and no injected noise.

## WSNet Relationship

- WSNet is the reusable upstream for mature shared infrastructure.
- Treat `training/base_trainer.py` and `utils/*` as mature WSNet-style shared scripts. Do not casually redesign or fork them inside HyperFlowNet.
- If a real shared-infrastructure fix is needed for `base_trainer` or `utils`, the preferred workflow is to make or finalize the change in WSNet, then sync HyperFlowNet deliberately.
- `models/hflownet.py` is the main local fast-iteration area. Once a model change is validated here, sync or back-port it to WSNet on purpose.
- `main.py`, `config.py`, `data/*`, `training/hflow_trainer.py`, and visualization logic are HyperFlowNet-local and should evolve directly in this repository.

## Practical Change Strategy

- Keep code and documents aligned with the active implementation.
- When changing the data pipeline, preserve the semantic contract:
  - raw or cached default states end up in `[Vx, Vy, P, T]`
  - feature scaling is channel-wise standardization
  - coordinate scaling is min-max normalization to `[-1, 1]`
  - train / validation use sliding windows
  - test data keeps full sequences for long-rollout evaluation
- When changing the trainer, protect the rollout + noise-injection path first.
- When changing the model, keep the reference-graph assumption explicit unless graph rebuilding is implemented end to end.
- Avoid adding moving-mesh or variable-cardinality machinery before the project truly needs it.
- Do not split the trainer into many thin near-empty subclasses when one subclass plus the existing hooks is enough.

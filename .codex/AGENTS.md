# HyperFlowNet Project AGENTS

## Project Snapshot

- HyperFlowNet is the irregular-mesh CFD task repository in the WSNet family.
- The current scope is fixed-mesh autoregressive flow prediction on a shared 2D unstructured mesh.
- Keep this repository focused on task-facing code: data ingestion, rollout training, inference, visualization, and fast local model iteration.

## Canonical Sources of Truth

- Treat the current repository code as the source of truth when notes, slides, or older reports disagree with it.
- The active execution path is `main.py -> FlowData / BoundaryCondition -> HyperFlowTrainer -> HyperFlowNet -> FlowVis`.
- `config.py` is the canonical place for current default experiment knobs.
- Some older internal documents describe Geo-FNO-era designs or older hyperparameters. Do not assume those documents match the present implementation.

## Dataset Mental Model

- The current dataset contains 19 cached transient CFD cases.
- Each case uses the same fixed 2D mesh and stores:
  - `states`: `(1001, 9617, 4)` float32
  - `coords`: `(9617, 2)` float32
- Coordinates are identical across all current cases.
- The approximate coordinate box is:
  - `x in [-0.213, 3.578]`
  - `y in [0.000, 0.578]`
- The channel order after ingestion is always `[Vx, Vy, P, T]`.
- The geometry is a chamber-pipe-chamber style layout with a long narrow connecting section. High activity is concentrated along the pipe and chamber-transition regions.
- Cases differ by operating condition / boundary condition, not by mesh topology.
- In the current cached dataset, coordinates are identical across cases, but this is a dataset fact, not a long-term architectural law.
- Prefer designs that remain compatible with different coordinates for different cases or operating conditions, as long as that compatibility does not add unnecessary complexity to the current code.
- This is an extreme-pressure-ratio transient-flow problem with sharp local structures and long autoregressive horizons.
- Pressure dominates the raw magnitude scale by orders of magnitude. Reason in standardized feature space during training, rollout, and boundary enforcement.
- Wall nodes are detected from near-zero velocity over all timesteps. On the full current cache, this is about 1.3k of 9.6k nodes, but every real run should still fit the wall mask from the training split only.
- With the current 19-case cache and the default `FlowData.spawn()` behavior, the split is 16 train / 2 val / 1 test under seed 42, with full-rollout evaluation on `case_4500`. Treat that as the current default, not an eternal assumption.

## WSNet Relationship

- WSNet is the reusable upstream for mature shared infrastructure.
- `training/base_trainer.py` and `utils/*` should be treated as mature shared scripts. Do not casually redesign or fork them inside HyperFlowNet.
- If a real shared-infrastructure fix is needed for `base_trainer` or `utils`, the preferred workflow is:
  - make or finalize the change in WSNet
  - then sync HyperFlowNet deliberately
- `models/hflownet.py` is the main exception:
  - fast model iteration is allowed and expected in HyperFlowNet first
  - once the model change is validated here, sync or back-port it to WSNet on purpose
- `training/hyperflow_trainer.py`, `data/*`, `main.py`, and visualization logic are HyperFlowNet-local and should evolve directly in this repository.

## Design Philosophy

- Optimize for simple, clear, tidy code.
- Prefer a direct implementation over an abstract one unless abstraction clearly removes real pain.
- Avoid unnecessary new classes, wrappers, managers, or indirection layers.
- Keep the training stack centered on `training/base_trainer.py`.
- All trainer classes should inherit from `training/base_trainer.py`.
- In normal cases, a trainer should only need:
  - a small task-specific `__init__`
  - `_compute_loss`
  - optional use of `_on_epoch_start` and `_on_epoch_end`
- Reuse the built-in hooks instead of building parallel callback systems unless there is a compelling reason.

## Algorithmic Priorities

- This is a shock-sensitive transient-flow problem. Accurate shock localization / identification matters.
- Favor learnable mechanisms that can represent sharp local structures on irregular meshes.
- This is also a long-horizon autoregressive problem. Stability over rollout is a first-class objective, not a secondary metric.
- The current winning training prior is pure rollout plus noise injection.
- Do not quietly fall back to mostly one-step supervision, weak teacher forcing, or curriculum logic that ignores rollout behavior.
- Noise injection is part of the training recipe, not a cosmetic detail. It reduces the training-inference mismatch and should remain visible in trainer design.
- Because the trainer injects noise, validation loss is useful but not sacred.
- Curriculum updates should be driven by rollout difficulty, rollout-step schedule, or explicit rollout-oriented signals, not by raw validation loss alone.
- Hard wall boundary enforcement is part of the rollout design. Preserve it unless there is a strong replacement with clearly better behavior.

## Practical Change Strategy

- When changing the model, tie the change to one of the real project needs:
  - better shock capture
  - better long-rollout stability
  - better geometry encoding on irregular meshes
  - better channel coupling
  - better boundary handling
- When changing the trainer, protect the pure-rollout + noise path first.
- When changing the data pipeline, preserve the semantic contract:
  - raw or cached states end up in `[Vx, Vy, P, T]`
  - feature scaling is channel-wise standardization
  - coordinate scaling is min-max to `[-1, 1]`
  - test data keeps full sequences for long-rollout evaluation
- When code and older notes, reports, or process documents disagree, treat the current code as the primary reference for present behavior.
- Do not rewrite historical or process-tracking documents just to force consistency with the latest code.
- When writing new technical documents, slides, or papers about the current implementation, prefer the code over older documents.

## What Usually Belongs Where

- Keep in HyperFlowNet:
  - `main.py`
  - `config.py`
  - `data/*`
  - `training/hyperflow_trainer.py`
  - `models/hflownet.py` during active architecture iteration
  - rollout diagnostics and visualization
- Keep in WSNet:
  - mature shared utilities
  - reusable trainer-base behavior
  - stable model implementations that are ready to serve as family-wide upstream code

## Non-Goals

- Do not add moving-mesh or variable-cardinality machinery before the project truly needs it.
- Do not hard-code the assumption that all cases must share identical coordinates just because the current cached dataset does.
- Do not overfit the design around validation-loss cosmetics.
- Do not split the trainer into many thin near-empty subclasses when one subclass plus the existing hooks is enough.

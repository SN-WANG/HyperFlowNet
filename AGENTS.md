# HyperFlowNet Agent Entry

Read `.codex/AGENTS.md` before making substantial changes in this repository. That file is the canonical project memory for Codex-style agents working on HyperFlowNet.

Project identity:

- `HyperFlowNet: A Spatio-Temporal Neural Operator for Flow Simulation`

The current active code path is:

- `main.py -> FlowData -> BoundaryCondition -> HyperFlowTrainer -> HyperFlowNet -> Metrics / FlowVis / FlowTwin`

The most important structural rule is:

- Treat `training/base_trainer.py` and `utils/*` as mature WSNet-style shared infrastructure.
- Treat `main.py`, `config.py`, `data/*`, `training/hflow_trainer.py`, and `models/hflownet.py` as HyperFlowNet-local task code.
- Treat `models/hflownet.py` as the local fast-iteration area: model ideas can be developed in HyperFlowNet first, then synced back to WSNet once validated.

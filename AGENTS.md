# HyperFlowNet Agent Entry

Read `.codex/AGENTS.md` before making substantial changes in this repository. That file is the canonical project memory for Codex-style agents working on HyperFlowNet.

The most important structural rule is:

- Treat `training/base_trainer.py` and `utils/*` as mature WSNet-style shared infrastructure.
- Treat `models/hflownet.py` as the local fast-iteration area: model ideas can be developed in HyperFlowNet first, then synced back to WSNet once validated.

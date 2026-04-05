# Argument configuration for HyperFlowNet
# Author: Shengning Wang

import argparse

import torch


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for HyperFlowNet.

    Returns:
        argparse.Namespace: Parsed experiment configuration.
    """
    parser = argparse.ArgumentParser(
        description="HyperFlowNet: Lightweight autoregressive flow prediction on irregular meshes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # 1. General
    # ============================================================

    general = parser.add_argument_group("General")
    general.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used for data splitting and training."
    )
    general.add_argument(
        "--mode", type=str, nargs="+", default=["train"],
        choices=["train", "infer", "probe"],
        help="Execution phases to run."
    )
    general.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device."
    )
    general.add_argument(
        "--data_dir", type=str, default="./dataset",
        help="Directory containing the CFD cases."
    )
    general.add_argument(
        "--output_dir", type=str, default="./runs",
        help="Directory used to save checkpoints, metrics, and figures."
    )
    general.add_argument(
        "--checkpoint_name", type=str, default="best.pt",
        help="Checkpoint filename loaded during inference."
    )

    # ============================================================
    # 2. Data
    # ============================================================

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--spatial_dim", type=int, default=2, choices=[2, 3],
        help="Spatial dimensionality of the mesh."
    )
    data.add_argument(
        "--channel_names", type=str, nargs="+", default=["Vx", "Vy", "P", "T"],
        help="Ordered field names of the state tensor."
    )
    data.add_argument(
        "--val_cases", type=int, default=2,
        help="Number of validation cases."
    )
    data.add_argument(
        "--test_cases", type=int, default=1,
        help="Number of test cases."
    )
    data.add_argument(
        "--win_len", type=int, default=13,
        help="Temporal window length used for training windows."
    )
    data.add_argument(
        "--train_win_stride", type=int, default=2,
        help="Sliding-window stride used for training augmentation."
    )
    data.add_argument(
        "--val_win_stride", type=int, default=5,
        help="Sliding-window stride used for validation augmentation."
    )
    data.add_argument(
        "--batch_size", type=int, default=4,
        help="Mini-batch size."
    )
    data.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of DataLoader workers."
    )

    # ============================================================
    # 3. Model
    # ============================================================

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--width", type=int, default=144,
        help="Hidden channel width."
    )
    model.add_argument(
        "--depth", type=int, default=4,
        help="Number of HyperFlow blocks."
    )
    model.add_argument(
        "--num_heads", type=int, default=8,
        help="Number of attention heads."
    )
    model.add_argument(
        "--num_slices", type=int, default=48,
        help="Number of slice tokens used in linear attention."
    )
    model.add_argument(
        "--ffn_ratio", type=float, default=2.0,
        help="Expansion ratio of the feed-forward network."
    )
    model.add_argument(
        "--num_fixed_bands", type=int, default=6,
        help="Number of fixed Fourier bands in the spatial encoder."
    )
    model.add_argument(
        "--num_learned_features", type=int, default=8,
        help="Number of learned Fourier features in the spatial encoder."
    )
    model.add_argument(
        "--time_features", type=int, default=4,
        help="Number of sinusoidal time-frequency pairs."
    )
    model.add_argument(
        "--freq_base", type=int, default=1000,
        help="Reference scale used by the temporal encoder."
    )
    model.add_argument(
        "--delta_scale", type=float, default=0.5,
        help="Residual scaling factor when delta prediction is enabled."
    )
    model.set_defaults(
        use_spatial_encoding=True,
        use_temporal_encoding=True,
        predict_delta=True,
        use_hard_bc=True,
    )
    model.add_argument(
        "--disable_spatial_encoding", action="store_false", dest="use_spatial_encoding",
        help="Disable the spatial encoder."
    )
    model.add_argument(
        "--disable_temporal_encoding", action="store_false", dest="use_temporal_encoding",
        help="Disable the temporal encoder."
    )
    model.add_argument(
        "--disable_delta_prediction", action="store_false", dest="predict_delta",
        help="Predict the next state directly instead of residual updates."
    )
    model.add_argument(
        "--disable_hard_bc", action="store_false", dest="use_hard_bc",
        help="Disable hard boundary-condition enforcement during rollout."
    )
    model.add_argument(
        "--velocity_threshold", type=float, default=1e-4,
        help="Velocity magnitude threshold used for wall-node detection."
    )

    # ============================================================
    # 4. Optimization
    # ============================================================

    optim = parser.add_argument_group("Optimization")
    optim.add_argument(
        "--lr", type=float, default=2e-4,
        help="Fallback optimizer learning rate."
    )
    optim.add_argument(
        "--weight_decay", type=float, default=1e-5,
        help="AdamW weight decay."
    )
    optim.add_argument(
        "--max_epochs", type=int, default=480,
        help="Maximum number of training epochs."
    )
    optim.add_argument(
        "--patience", type=int, default=120,
        help="Early stopping patience."
    )

    # ============================================================
    # 5. Curriculum
    # ============================================================

    curriculum = parser.add_argument_group("Curriculum")
    curriculum.add_argument(
        "--rollout_steps", type=int, nargs="+", default=[1, 2, 4, 8, 12],
        help="Rollout length of each curriculum stage."
    )
    curriculum.add_argument(
        "--stage_ratios", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.20, 0.20],
        help="Epoch ratio of each curriculum stage."
    )
    curriculum.add_argument(
        "--teacher_forcing_lows", type=float, nargs="+", default=[1.0, 0.9, 0.75, 0.4, 0.0],
        help="Final teacher-forcing ratio of each stage."
    )
    curriculum.add_argument(
        "--stage_lrs", type=float, nargs="+", default=[2e-4, 2e-4, 1.5e-4, 1e-4, 7e-5],
        help="Base learning rate of each curriculum stage."
    )
    curriculum.add_argument(
        "--stage_warmup_ratio", type=float, default=0.2,
        help="Warmup ratio inside each curriculum stage."
    )
    curriculum.add_argument(
        "--stage_min_lr_ratio", type=float, default=0.05,
        help="Minimal LR ratio inside the stage cosine schedule."
    )
    curriculum.add_argument(
        "--input_noise_std", type=float, default=0.01,
        help="Initial rollout input noise std."
    )
    curriculum.add_argument(
        "--input_noise_decay", type=float, default=0.85,
        help="Stage-wise decay factor of rollout input noise."
    )
    curriculum.add_argument(
        "--eval_rollout_steps", type=int, default=12,
        help="Validation rollout length."
    )

    # ============================================================
    # 6. Loss
    # ============================================================

    loss = parser.add_argument_group("Loss")
    loss.add_argument(
        "--channel_weights", type=float, nargs="+", default=[1.0, 3.0, 2.0, 1.0],
        help="Per-channel loss weights."
    )
    loss.add_argument(
        "--delta_loss_weight", type=float, default=0.005,
        help="Auxiliary temporal-delta loss weight."
    )
    loss.add_argument(
        "--step_weight_power", type=float, default=1.0,
        help="Power used to emphasize late rollout steps."
    )
    loss.add_argument(
        "--loss_eps", type=float, default=1e-6,
        help="Small constant used in NMSE normalization."
    )

    # ============================================================
    # 7. Probe
    # ============================================================

    probe = parser.add_argument_group("Probe")
    probe.add_argument(
        "--probe_rollout_steps", type=int, default=12,
        help="Rollout length used during the memory probe."
    )

    args = parser.parse_args()

    num_stages = len(args.rollout_steps)
    if len(args.stage_ratios) != num_stages or len(args.teacher_forcing_lows) != num_stages or len(args.stage_lrs) != num_stages:
        parser.error("rollout_steps, stage_ratios, teacher_forcing_lows, and stage_lrs must have the same length.")

    if abs(sum(args.stage_ratios) - 1.0) > 1e-6:
        parser.error("stage_ratios must sum to 1.0.")

    if len(args.channel_weights) != len(args.channel_names):
        parser.error("channel_weights must have the same length as channel_names.")

    return args

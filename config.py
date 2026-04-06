# Argument configuration for HyperFlowNet
# Author: Shengning Wang

import argparse

import torch


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the HyperFlowNet workflow.

    Returns:
        argparse.Namespace: Parsed experiment arguments.
    """
    parser = argparse.ArgumentParser(
        description="HyperFlowNet: A Spatio-Temporal Neural Operator for Flow Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ============================================================
    # 1. General
    # ============================================================

    general = parser.add_argument_group("General")
    general.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    general.add_argument(
        "--data_dir", type=str, default="./dataset",
        help="Path to directory containing simulation case folders."
    )
    general.add_argument(
        "--output_dir", type=str, default="./runs",
        help="Directory to save checkpoints, logs, and visualizations."
    )
    general.add_argument(
        "--mode", type=str, nargs="+", default=["probe", "train", "infer"],
        choices=["probe", "train", "infer"],
        help="Execution phases to run."
    )
    general.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device."
    )

    # ============================================================
    # 2. Data
    # ============================================================

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--channel_names", type=str, nargs="+", default=["Vx", "Vy", "P", "T"],
        help="Ordered list of output channel names."
    )
    data.add_argument(
        "--spatial_dim", type=int, default=2, choices=[2, 3],
        help="Spatial dimensionality of the mesh."
    )
    data.add_argument(
        "--win_len", type=int, default=13,
        help="Temporal window length for sequence slicing."
    )
    data.add_argument(
        "--win_stride", type=int, default=1,
        help="Stride for sliding-window data augmentation."
    )
    data.add_argument(
        "--batch_size", type=int, default=8,
        help="Mini-batch size for training and validation."
    )
    data.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of DataLoader worker subprocesses."
    )

    # ============================================================
    # 3. Model
    # ============================================================

    model = parser.add_argument_group("HyperFlowNet")
    model.add_argument(
        "--depth", type=int, default=4,
        help="Number of stacked HyperFlow blocks."
    )
    model.add_argument(
        "--width", type=int, default=128,
        help="Hidden channel dimension."
    )
    model.add_argument(
        "--num_slices", type=int, default=32,
        help="Number of mesh slice tokens."
    )
    model.add_argument(
        "--num_heads", type=int, default=8,
        help="Number of attention heads."
    )
    model.add_argument(
        "--use_spatial_encoding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable spatial encoding."
    )
    model.add_argument(
        "--use_temporal_encoding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable temporal encoding."
    )
    model.add_argument(
        "--use_hard_bc", action=argparse.BooleanOptionalAction, default=True,
        help="Enable hard boundary-condition enforcement during rollout."
    )
    model.add_argument(
        "--velocity_threshold", type=float, default=1e-4,
        help="Velocity magnitude threshold for wall-node detection."
    )
    model.add_argument(
        "--coords_features", type=int, default=8,
        help="Number of learned spatial encoding features."
    )
    model.add_argument(
        "--time_features", type=int, default=4,
        help="Number of temporal sinusoidal frequency pairs."
    )
    model.add_argument(
        "--freq_base", type=int, default=1000,
        help="Base for sinusoidal temporal frequencies."
    )

    # ============================================================
    # 4. Optimization
    # ============================================================

    optim = parser.add_argument_group("Optimization")
    optim.add_argument(
        "--lr", type=float, default=5e-4,
        help="Initial learning rate for AdamW."
    )
    optim.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="L2 regularization coefficient for AdamW."
    )
    optim.add_argument(
        "--max_epochs", type=int, default=560,
        help="Maximum training epochs."
    )
    optim.add_argument(
        "--eta_min", type=float, default=1e-6,
        help="Minimum learning rate for cosine annealing."
    )
    optim.add_argument(
        "--channel_weights", type=float, nargs="+", default=[1.0, 3.0, 1.0, 1.0],
        help="Per-channel NMSE loss weights."
    )

    # ============================================================
    # 5. Curriculum
    # ============================================================

    curriculum = parser.add_argument_group("Curriculum")
    curriculum.add_argument(
        "--max_rollout_steps", type=int, default=12,
        help="Maximum autoregressive rollout steps."
    )
    curriculum.add_argument(
        "--rollout_patience", type=int, default=55,
        help="Epochs between curriculum difficulty advances."
    )
    curriculum.add_argument(
        "--noise_std_init", type=float, default=0.01,
        help="Initial std of Gaussian noise injected into the input state."
    )
    curriculum.add_argument(
        "--noise_decay", type=float, default=0.7,
        help="Multiplicative decay for rollout noise."
    )
    curriculum.add_argument(
        "--teacher_forcing_init", type=float, default=0.25,
        help="Initial teacher-forcing ratio used after the one-step stage."
    )
    curriculum.add_argument(
        "--teacher_forcing_decay", type=float, default=0.5,
        help="Multiplicative decay for teacher forcing after each curriculum advance."
    )
    curriculum.add_argument(
        "--teacher_forcing_floor", type=float, default=0.0,
        help="Minimum teacher-forcing ratio."
    )

    return parser.parse_args()

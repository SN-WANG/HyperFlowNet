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
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    general.add_argument(
        "--data_dir", type=str, default="./dataset", help="Path to directory containing cached cases."
    )
    general.add_argument(
        "--output_dir", type=str, default="./runs", help="Directory to save checkpoints and outputs."
    )
    general.add_argument(
        "--mode", type=str, nargs="+", default=["probe", "train", "infer"], help="Execution phases to run."
    )
    general.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device."
    )

    # ============================================================
    # 2. Data
    # ============================================================

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--channel_names", type=str, nargs="+", default=["Vx", "Vy", "P", "T"], help="Field names."
    )
    data.add_argument(
        "--spatial_dim", type=int, default=2, choices=[2, 3], help="Spatial dimensionality."
    )
    data.add_argument(
        "--win_len", type=int, default=13, help="Temporal window length for sequence slicing."
    )
    data.add_argument(
        "--win_stride", type=int, default=1, help="Stride for sliding-window augmentation."
    )
    data.add_argument(
        "--batch_size", type=int, default=8, help="Mini-batch size for training and validation."
    )

    # ============================================================
    # 3. HyperFlowNet
    # ============================================================

    hflownet = parser.add_argument_group("HyperFlowNet")
    hflownet.add_argument(
        "--depth", type=int, default=4, help="Number of stacked HyperFlowNet blocks."
    )
    hflownet.add_argument(
        "--width", type=int, default=128, help="Hidden channel width."
    )
    hflownet.add_argument(
        "--num_slices", type=int, default=32, help="Number of slice tokens."
    )
    hflownet.add_argument(
    "--num_heads", type=int, default=8, help="Number of slice-space attention heads."
    )
    hflownet.add_argument(
        "--use_spatial_encoding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable learnable Fourier spatial encoding.",
    )
    hflownet.add_argument(
        "--use_temporal_encoding", action=argparse.BooleanOptionalAction, default=True,
        help="Enable sinusoidal temporal encoding.",
    )
    hflownet.add_argument(
        "--coord_features", type=int, default=8, help="Half-dimension of Fourier spatial encoding."
    )
    hflownet.add_argument(
        "--time_features", type=int, default=4, help="Half-dimension of temporal encoding."
    )
    hflownet.add_argument(
        "--freq_base", type=int, default=1000, help="Base for temporal frequency decay."
    )

    # ============================================================
    # 4. Trainer
    # ============================================================

    trainer = parser.add_argument_group("Trainer")
    trainer.add_argument(
        "--lr", type=float, default=5e-4, help="Initial learning rate for AdamW."
    )
    trainer.add_argument(
        "--weight_decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    trainer.add_argument(
        "--max_epochs", type=int, default=560, help="Maximum training epochs."
    )
    trainer.add_argument(
        "--eta_min", type=float, default=1e-6, help="Minimum learning rate for cosine annealing."
    )
    trainer.add_argument(
        "--channel_weights", type=float, nargs="+", default=[1.0, 3.0, 1.0, 1.0], help="Per-channel loss weights."
    )

    # ============================================================
    # 5. Curriculum
    # ============================================================

    curriculum = parser.add_argument_group("Curriculum")
    curriculum.add_argument(
        "--max_rollout_steps", type=int, default=12, help="Maximum autoregressive rollout steps."
    )
    curriculum.add_argument(
        "--rollout_patience", type=int, default=55, help="Epochs between curriculum advances."
    )
    curriculum.add_argument(
        "--noise_std_init", type=float, default=0.01, help="Initial rollout noise std."
    )
    curriculum.add_argument(
        "--noise_decay", type=float, default=0.7, help="Multiplicative decay of rollout noise."
    )

    return parser.parse_args()

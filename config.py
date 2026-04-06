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
    general.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    general.add_argument(
        "--data_dir", type=str, default="./dataset", help="Path to directory containing cached cases.")
    general.add_argument(
        "--output_dir", type=str, default="./runs", help="Directory to save checkpoints and outputs.")
    general.add_argument(
        "--mode", type=str, nargs="+", default=["probe", "train", "infer"], choices=["probe", "train", "infer"],
        help="Execution phases to run.")
    general.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device.")

    # ============================================================
    # 2. Data
    # ============================================================

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--channel_names", type=str, nargs="+", default=["Vx", "Vy", "P", "T"], help="Field names.")
    data.add_argument(
        "--spatial_dim", type=int, default=2, choices=[2, 3], help="Spatial dimensionality.")
    data.add_argument(
        "--win_len", type=int, default=13, help="Temporal window length for sequence slicing.")
    data.add_argument(
        "--win_stride", type=int, default=1, help="Stride for sliding-window augmentation.")
    data.add_argument(
        "--batch_size", type=int, default=10, help="Mini-batch size for training and validation.")

    # ============================================================
    # 3. Model
    # ============================================================

    model = parser.add_argument_group("HyperFlowNet")
    model.add_argument(
        "--depth", type=int, default=6, help="Number of recurrent refinement steps.")
    model.add_argument(
        "--width", type=int, default=128, help="Node token width.")
    model.add_argument(
        "--num_slices", type=int, default=24, help="Number of soft slice states.")
    model.add_argument(
        "--latent_dim", type=int, default=32, help="Latent transition width.")
    model.add_argument(
        "--num_anchors", type=int, default=8, help="Number of anchor states in LatentTransition.")
    model.add_argument(
        "--time_features", type=int, default=4, help="Number of temporal sinusoidal frequency pairs.")
    model.add_argument(
        "--freq_base", type=int, default=1000, help="Base for temporal sinusoidal frequencies.")
    model.add_argument(
        "--use_hard_bc", action=argparse.BooleanOptionalAction, default=True,
        help="Enable hard boundary-condition enforcement during rollout.")
    model.add_argument(
        "--velocity_threshold", type=float, default=1e-4,
        help="Velocity magnitude threshold for wall-node detection.")

    # ============================================================
    # 4. Optimization
    # ============================================================

    optim = parser.add_argument_group("Optimization")
    optim.add_argument(
        "--lr", type=float, default=6e-4, help="Initial learning rate for AdamW.")
    optim.add_argument(
        "--weight_decay", type=float, default=1e-4, help="L2 regularization coefficient for AdamW.")
    optim.add_argument(
        "--max_epochs", type=int, default=240, help="Maximum training epochs.")
    optim.add_argument(
        "--eta_min", type=float, default=1e-6, help="Minimum learning rate for cosine annealing.")
    optim.add_argument(
        "--channel_weights", type=float, nargs="+", default=[1.0, 3.0, 1.0, 1.0], help="Per-channel NMSE weights.")

    # ============================================================
    # 5. Curriculum
    # ============================================================

    curriculum = parser.add_argument_group("Curriculum")
    curriculum.add_argument(
        "--max_rollout_steps", type=int, default=12,
        help="Maximum autoregressive rollout steps.")
    curriculum.add_argument(
        "--rollout_patience", type=int, default=18,
        help="Epochs between curriculum difficulty advances.")
    curriculum.add_argument(
        "--noise_std_init", type=float, default=0.01,
        help="Initial std of Gaussian noise injected into rollout input.")
    curriculum.add_argument(
        "--noise_decay", type=float, default=0.75,
        help="Multiplicative decay for rollout noise.")

    return parser.parse_args()

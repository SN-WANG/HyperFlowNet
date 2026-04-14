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
        "--batch_size", type=int, default=8, help="Mini-batch size for training and validation.")
    data.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader worker processes.")
    data.add_argument(
        "--persistent_workers", action=argparse.BooleanOptionalAction, default=True,
        help="Keep DataLoader workers alive across epochs.")

    # ============================================================
    # 3. Model
    # ============================================================

    model = parser.add_argument_group("HyperFlowNet")
    model.add_argument(
        "--depth", type=int, default=4, help="Number of recurrent refinement steps.")
    model.add_argument(
        "--width", type=int, default=128, help="Node token width.")
    model.add_argument(
        "--num_slices", type=int, default=32, help="Number of soft slice states.")
    model.add_argument(
        "--num_heads", type=int, default=8, help="Number of slice-attention heads.")
    model.add_argument(
        "--latent_dim", type=int, default=32, help="Latent transition width.")
    model.add_argument(
        "--num_anchors", type=int, default=8, help="Number of anchor states in LatentTransition.")
    model.add_argument(
        "--time_features", type=int, default=4, help="Number of temporal sinusoidal frequency pairs.")
    model.add_argument(
        "--freq_base", type=int, default=1000, help="Base for temporal sinusoidal frequencies.")
    model.add_argument(
        "--attn_dropout", type=float, default=0.0, help="Dropout probability inside slice attention.")

    # ============================================================
    # 4. Optimization
    # ============================================================

    optim = parser.add_argument_group("Optimization")
    optim.add_argument(
        "--lr", type=float, default=5e-4, help="Initial learning rate for AdamW.")
    optim.add_argument(
        "--weight_decay", type=float, default=1e-4, help="L2 regularization coefficient for AdamW.")
    optim.add_argument(
        "--max_epochs", type=int, default=420, help="Maximum training epochs.")
    optim.add_argument(
        "--eta_min", type=float, default=1e-6, help="Minimum learning rate for cosine annealing.")

    # ============================================================
    # 5. Curriculum
    # ============================================================

    curriculum = parser.add_argument_group("Curriculum")
    curriculum.add_argument(
        "--max_rollout_steps", type=int, default=12,
        help="Maximum autoregressive rollout steps.")
    curriculum.add_argument(
        "--rollout_patience", type=int, default=35,
        help="Epochs between curriculum difficulty advances.")
    curriculum.add_argument(
        "--noise_std_init", type=float, default=0.01,
        help="Initial std of Gaussian noise injected into rollout input.")
    curriculum.add_argument(
        "--noise_decay", type=float, default=0.80,
        help="Multiplicative decay for rollout noise.")

    # ============================================================
    # 6. Loss And Evaluation
    # ============================================================

    evaluate = parser.add_argument_group("Loss And Evaluation")
    evaluate.add_argument(
        "--loss_weight_beta", type=float, default=0.90,
        help="EMA momentum for adaptive channel-loss weighting.")
    evaluate.add_argument(
        "--loss_weight_alpha", type=float, default=1.25,
        help="Exponent applied to normalized channel hardness.")
    evaluate.add_argument(
        "--loss_weight_min", type=float, default=0.5,
        help="Minimum adaptive channel weight before normalization.")
    evaluate.add_argument(
        "--loss_weight_max", type=float, default=4.0,
        help="Maximum adaptive channel weight before normalization.")
    evaluate.add_argument(
        "--loss_weight_warmup", type=int, default=10,
        help="Warmup epochs before adaptive channel weighting starts.")
    evaluate.add_argument(
        "--rollout_eval_interval", type=int, default=5,
        help="Epoch interval for full-horizon rollout evaluation.")
    evaluate.add_argument(
        "--early_stop_patience", type=int, default=70,
        help="Number of rollout evaluations without improvement before early stop.")
    evaluate.add_argument(
        "--checkpoint_metric", type=str, default="hard_rollout_nmse", choices=["hard_rollout_nmse"],
        help="Validation metric used for best-checkpoint selection.")

    return parser.parse_args()

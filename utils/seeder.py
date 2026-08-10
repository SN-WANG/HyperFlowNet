# Reproducibility utilities for JAX experiments
# Author: Shengning Wang

import random

import jax
import numpy as np

from utils.hue_logger import hue, logger


def seed_everything(seed: int = 42) -> jax.Array:
    """
    Seed all global RNGs and return a JAX key.

    Args:
        seed (int): The seed value.

    Returns:
        jax.Array: JAX PRNG key derived from the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    logger.info(f"global seed set to {hue.m}{seed}{hue.q}")
    return key

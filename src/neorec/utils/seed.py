"""Deterministic seeding across all numerical libraries used in NeoRec."""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (+ CUDA), and TensorFlow if available.

    Parameters
    ----------
    seed:
        Global seed.
    deterministic_cuda:
        If True, enables deterministic CUDNN ops (slower but reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover
        pass

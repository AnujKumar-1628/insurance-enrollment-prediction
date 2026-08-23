import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int, default=42
        Random seed value.
    """

    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

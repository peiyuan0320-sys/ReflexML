import os
import random

import numpy as np
import torch


def set_reproducible_seed(seed: int) -> None:
    """Seed libraries and request deterministic PyTorch operations."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # This project uses CPU by default. These settings also reduce variation if
    # a CUDA device is selected explicitly.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def capture_rng_states() -> dict:
    states = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict) -> None:
    random.setstate(states["python_rng_state"])
    np.random.set_state(states["numpy_rng_state"])
    torch.set_rng_state(states["torch_rng_state"])
    if "cuda_rng_states" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda_rng_states"])


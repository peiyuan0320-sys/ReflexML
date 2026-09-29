from pathlib import Path

import torch
from torch import nn

from .config import ExperimentConfig
from .reproducibility import capture_rng_states, restore_rng_states


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    train_generator: torch.Generator,
    split_fingerprint: str,
    history: list[dict] | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_learning_rate": optimizer.param_groups[0]["lr"],
        "random_seed": config.seed,
        "config": config.to_dict(),
        "split_fingerprint": split_fingerprint,
        "train_loader_generator_state": train_generator.get_state(),
        **capture_rng_states(),
    }
    if history is not None:
        checkpoint["history"] = history
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    # weights_only=False is required because a complete research checkpoint also
    # contains Python and NumPy RNG states. Only load checkpoints you trust.
    return torch.load(path, map_location=device, weights_only=False)


def restore_checkpoint(
    checkpoint: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    expected_split_fingerprint: str,
) -> int:
    if checkpoint["split_fingerprint"] != expected_split_fingerprint:
        raise ValueError("Checkpoint and current dataset split do not match")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_generator.set_state(checkpoint["train_loader_generator_state"])
    restore_rng_states(checkpoint)
    return int(checkpoint["epoch"])

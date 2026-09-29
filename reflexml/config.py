from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class ExperimentConfig:
    """All settings needed to understand and repeat one training run."""

    seed: int = 42
    split_seed: int = 2026
    train_size: int = 10_000
    val_size: int = 5_000
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 0.1
    momentum: float = 0.9
    hidden_size: int = 128
    num_workers: int = 0
    device: str = "cpu"
    dataset_name: str = "FashionMNIST"
    optimizer_name: str = "SGD"
    model_structure: str = "Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path, split_fingerprint: str) -> None:
        data = self.to_dict()
        data["dataset_split"] = {
            "source": "FashionMNIST official training set",
            "split_seed": self.split_seed,
            "train_size": self.train_size,
            "validation_size": self.val_size,
            "unused_training_examples": 60_000 - self.train_size - self.val_size,
            "indices_sha256": split_fingerprint,
        }
        data["test_set_usage"] = "final evaluation only"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


import torch
from torch import nn


class FashionMLP(nn.Module):
    """The deliberately small model used by ReflexML v0.1."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


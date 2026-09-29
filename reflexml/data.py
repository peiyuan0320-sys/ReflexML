from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import ExperimentConfig


class IndexedSubset(Subset):
    """审计训练顺序：额外返回官方训练集中的样本索引。"""

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, label, self.indices[index]

    def __getitems__(self, indices):
        # DataLoader 支持批量取样，必须同步覆盖这个入口。
        return [self[index] for index in indices]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_generator: torch.Generator
    train_indices: list[int]
    val_indices: list[int]
    split_fingerprint: str


def make_split_indices(
    dataset_size: int, train_size: int, val_size: int, split_seed: int
) -> tuple[list[int], list[int]]:
    if train_size + val_size > dataset_size:
        raise ValueError("train_size + val_size exceeds the dataset size")

    # A separate generator makes the split independent of model initialization.
    split_generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(dataset_size, generator=split_generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size : train_size + val_size]
    return train_indices, val_indices


def split_fingerprint(train_indices: list[int], val_indices: list[int]) -> str:
    values = torch.tensor(train_indices + [-1] + val_indices, dtype=torch.int64)
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def make_loaders_from_datasets(
    training_dataset: Dataset,
    test_dataset: Dataset,
    config: ExperimentConfig,
    audit_order: bool = False,
) -> DataBundle:
    train_indices, val_indices = make_split_indices(
        len(training_dataset), config.train_size, config.val_size, config.split_seed
    )
    subset_class = IndexedSubset if audit_order else Subset
    train_subset = subset_class(training_dataset, train_indices)
    val_subset = Subset(training_dataset, val_indices)

    # This generator controls RandomSampler, hence the minibatch order. Its
    # state is stored in every checkpoint at the end of an epoch.
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_generator=train_generator,
        train_indices=train_indices,
        val_indices=val_indices,
        split_fingerprint=split_fingerprint(train_indices, val_indices),
    )


def load_fashion_mnist(config: ExperimentConfig, data_dir: Path) -> DataBundle:
    transform = transforms.ToTensor()
    training_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return make_loaders_from_datasets(training_dataset, test_dataset, config)

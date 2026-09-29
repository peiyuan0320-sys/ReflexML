from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from reflexml.checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from reflexml.config import ExperimentConfig
from reflexml.data import make_loaders_from_datasets
from reflexml.model import FashionMLP
from reflexml.reproducibility import set_reproducible_seed
from reflexml.training import evaluate, train_one_epoch


def synthetic_datasets() -> tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(999)
    train_images = torch.rand(80, 1, 28, 28, generator=generator)
    train_labels = torch.randint(0, 10, (80,), generator=generator)
    test_images = torch.rand(20, 1, 28, 28, generator=generator)
    test_labels = torch.randint(0, 10, (20,), generator=generator)
    return TensorDataset(train_images, train_labels), TensorDataset(test_images, test_labels)


def build_experiment(seed: int = 7):
    config = ExperimentConfig(
        seed=seed, split_seed=11, train_size=60, val_size=20, batch_size=10
    )
    set_reproducible_seed(seed)
    training_dataset, test_dataset = synthetic_datasets()
    data = make_loaders_from_datasets(training_dataset, test_dataset, config)
    model = FashionMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return config, data, model, optimizer


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_repeats_split_initial_model_and_curve(self) -> None:
        curves = []
        initial_states = []
        fingerprints = []
        for _ in range(2):
            _, data, model, optimizer = build_experiment()
            fingerprints.append(data.split_fingerprint)
            initial_states.append(deepcopy(model.state_dict()))
            loss_function = nn.CrossEntropyLoss()
            curves.append(
                [
                    train_one_epoch(
                        model,
                        data.train_loader,
                        optimizer,
                        loss_function,
                        torch.device("cpu"),
                    )
                    for _ in range(2)
                ]
            )

        self.assertEqual(fingerprints[0], fingerprints[1])
        for name in initial_states[0]:
            self.assertTrue(torch.equal(initial_states[0][name], initial_states[1][name]))
        self.assertEqual(curves[0], curves[1])

    def test_checkpoint_resume_matches_uninterrupted_training(self) -> None:
        config, data, model, optimizer = build_experiment()
        loss_function = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        train_one_epoch(model, data.train_loader, optimizer, loss_function, device)
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "epoch_001.pt"
            save_checkpoint(
                checkpoint_path,
                1,
                model,
                optimizer,
                config,
                data.train_generator,
                data.split_fingerprint,
            )
            expected_random_values = (
                random.random(),
                np.random.random(),
                torch.rand(1).item(),
            )

            uninterrupted_loss = train_one_epoch(
                model, data.train_loader, optimizer, loss_function, device
            )
            uninterrupted_validation = evaluate(
                model, data.val_loader, loss_function, device
            )
            uninterrupted_state = deepcopy(model.state_dict())

            _, resumed_data, resumed_model, resumed_optimizer = build_experiment()
            checkpoint = load_checkpoint(checkpoint_path, device)
            restored_epoch = restore_checkpoint(
                checkpoint,
                resumed_model,
                resumed_optimizer,
                resumed_data.train_generator,
                resumed_data.split_fingerprint,
            )
            restored_random_values = (
                random.random(),
                np.random.random(),
                torch.rand(1).item(),
            )
            resumed_loss = train_one_epoch(
                resumed_model,
                resumed_data.train_loader,
                resumed_optimizer,
                loss_function,
                device,
            )
            resumed_validation = evaluate(
                resumed_model, resumed_data.val_loader, loss_function, device
            )

        self.assertEqual(restored_epoch, 1)
        self.assertEqual(restored_random_values, expected_random_values)
        self.assertEqual(resumed_loss, uninterrupted_loss)
        self.assertEqual(resumed_validation, uninterrupted_validation)
        for name, tensor in uninterrupted_state.items():
            self.assertTrue(torch.equal(tensor, resumed_model.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()

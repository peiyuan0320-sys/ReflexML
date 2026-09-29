"""Non-confirmatory technical identities for the frozen m=4 kernel."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from reflexml.data import make_split_indices
from reflexml.gradient_noise_attenuation import (
    admit_train_mapping, canonical_train_indices, construct_streams,
    historical_a1_pair, mean_gradient_step, permutation_positions,
    reset_momentum_buffers, run_production, verify_authorities,
)
from reflexml.model import FashionMLP


ROOT = Path(__file__).resolve().parents[1]


class GradientNoiseTechnicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ordered_t, _ = make_split_indices(60000, 10000, 5000, 2026)

    def fixture(self, batch_size=4):
        torch.manual_seed(812)
        model = FashionMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        for p in model.parameters():
            optimizer.state[p]["momentum_buffer"] = torch.ones_like(p) * 0.2
        components = [(torch.randn(batch_size, 1, 28, 28),
                       torch.randint(0, 10, (batch_size,))) for _ in range(4)]
        return model, optimizer, components

    def test_mean_gradient_one_step_and_static_preupdate(self):
        for size in (4, 16):
            model, optimizer, components = self.fixture(size)
            reference = copy.deepcopy(model)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
            reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            parameter_snapshots = []
            forward = model.forward

            def observing_forward(images):
                parameter_snapshots.append([p.detach().clone() for p in model.parameters()])
                return forward(images)

            with patch.object(model, "forward", side_effect=observing_forward), patch.object(
                optimizer, "step", wraps=optimizer.step
            ) as step:
                gradients = mean_gradient_step(model, optimizer, components, nn.CrossEntropyLoss())
                self.assertEqual(step.call_count, 1)
            self.assertEqual(len(parameter_snapshots), 4)
            for snapshot in parameter_snapshots[1:]:
                for left, right in zip(snapshot, parameter_snapshots[0], strict=True):
                    self.assertTrue(torch.equal(left, right))
            self.assertEqual(len(gradients), 4)
            for index, (actual, expected) in enumerate(zip(model.parameters(), reference.parameters(), strict=True)):
                averaged = torch.stack([row[index] for row in gradients]).mean(dim=0)
                self.assertTrue(torch.equal(actual.grad, averaged))
                expected.grad = averaged.clone()
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(optimizer.state[actual]["momentum_buffer"],
                                            reference_optimizer.state[expected]["momentum_buffer"]))

    def test_gradient_equals_mean_component_loss(self):
        for size in (4, 16):
            model, optimizer, components = self.fixture(size)
            reference = copy.deepcopy(model)
            gradients = mean_gradient_step(model, optimizer, components, nn.CrossEntropyLoss())
            reference.zero_grad()
            (sum(nn.CrossEntropyLoss()(reference(x), y) for x, y in components) / 4).backward()
            for index, parameter in enumerate(reference.parameters()):
                expected = torch.stack([row[index] for row in gradients]).mean(dim=0)
                self.assertTrue(torch.allclose(parameter.grad, expected, rtol=1e-5, atol=1e-7))

    def test_final_component_step_preserves_scientific_state_until_update(self):
        model, optimizer, components = self.fixture(16)
        before_parameters = [p.detach().clone() for p in model.parameters()]
        before_buffers = [optimizer.state[p]["momentum_buffer"].clone()
                          for p in model.parameters()]
        observations = []
        forward = model.forward

        def observing_forward(images):
            observations.append((
                [p.detach().clone() for p in model.parameters()],
                [optimizer.state[p]["momentum_buffer"].clone()
                 for p in model.parameters()],
                tuple(model.named_buffers()),
                len(images),
            ))
            return forward(images)

        with patch.object(model, "forward", side_effect=observing_forward), patch.object(
            optimizer, "step", wraps=optimizer.step
        ) as step:
            mean_gradient_step(model, optimizer, components, nn.CrossEntropyLoss())
            self.assertEqual(step.call_count, 1)
        self.assertEqual(len(observations), 4)
        for parameters, buffers, model_buffers, count in observations:
            self.assertEqual(count, 16)
            self.assertEqual(model_buffers, ())
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(parameters, before_parameters, strict=True)))
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(buffers, before_buffers, strict=True)))

    @unittest.skip("Requires excluded historical data/authority/order artifacts; no public substitute")
    def test_streams_determinism_pairing_and_last_batch(self):
        for state in (1, 36):
            ten, five = historical_a1_pair(state)
            first = ten["result"]["realized_order"]
            self.assertEqual(first, five["result"]["realized_order"])
            left = construct_streams(state, self.ordered_t, first)
            right = construct_streams(state, self.ordered_t, first)
            self.assertEqual(left, right)
            self.assertEqual(len(set(left["order_sha256"])), 4)
            for index, stream in enumerate(left["streams"]):
                self.assertEqual([len(batch) for batch in stream], [128] * 78 + [16])
                self.assertEqual(set(sum(stream, [])), set(self.ordered_t))
                if index:
                    key, positions = permutation_positions(state, index + 1)
                    self.assertEqual(key.hex(), left["stream_key_sha256"][str(index + 1)])
                    self.assertEqual(sum(stream, []), [self.ordered_t[p] for p in positions])
            # The two LR constructors receive one immutable stream value.
            for lr in (0.10, 0.05):
                self.assertEqual(construct_streams(state, self.ordered_t, first), left)

    @unittest.skip("Requires excluded historical data/authority/order artifacts; no public substitute")
    def test_ordered_t_rejects_same_set_permutation(self):
        self.assertEqual(admit_train_mapping(self.ordered_t, self.ordered_t, self.registration), self.ordered_t)
        swapped = self.ordered_t.copy()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaises(ValueError):
            admit_train_mapping(self.ordered_t, swapped, self.registration)
        with self.assertRaises(ValueError):
            admit_train_mapping(swapped, swapped, self.registration)

    @unittest.skip("Requires excluded historical data/authority/order artifacts; no public substitute")
    def test_registered_dataset_loader_mapping(self):
        from torchvision.datasets import FashionMNIST
        from torchvision.transforms import ToTensor
        dataset = FashionMNIST(ROOT / "data", train=True, download=False, transform=ToTensor())
        self.assertEqual(canonical_train_indices(ROOT, dataset), self.ordered_t)

    @unittest.skip("Requires excluded historical data/authority/order artifacts; no public substitute")
    def test_historical_a1_pair_and_reset(self):
        ten, five = historical_a1_pair(1)
        self.assertEqual(ten["result"]["realized_order"], five["result"]["realized_order"])
        model, optimizer, _ = self.fixture()
        original_model = copy.deepcopy(model.state_dict())
        original_optimizer = copy.deepcopy(optimizer.state_dict())
        reset_model, reset_optimizer = reset_momentum_buffers(original_model, original_optimizer)
        self.assertEqual(len(reset_optimizer["state"]), 4)
        self.assertEqual(reset_optimizer["param_groups"][0]["momentum"], 0.9)
        for key, original in original_optimizer["state"].items():
            before = original["momentum_buffer"]
            after = reset_optimizer["state"][key]["momentum_buffer"]
            self.assertEqual((after.shape, after.dtype, after.device), (before.shape, before.dtype, before.device))
            self.assertEqual(torch.count_nonzero(after).item(), 0)
        for key in original_model:
            self.assertTrue(torch.equal(reset_model[key], original_model[key]))

    def test_production_guard(self):
        with self.assertRaises(PermissionError):
            run_production()


if __name__ == "__main__":
    unittest.main()

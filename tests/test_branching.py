from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from reflexml.branching import BranchingConfig, action_outcome, paired_branches, states_equal
from reflexml.checkpoint import load_checkpoint, save_checkpoint
from reflexml.features import extract_features
from tests.test_reproducibility import build_experiment, synthetic_datasets
from reflexml.training import train_one_epoch
import torch
from torch import nn


class BranchingTests(unittest.TestCase):
    def test_fair_pair_and_same_lr_five_epochs(self):
        config, data, model, optimizer = build_experiment()
        train_one_epoch(model, data.train_loader, optimizer, nn.CrossEntropyLoss(), torch.device("cpu"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, 1, model, optimizer, config, data.train_generator, data.split_fingerprint)
            checkpoint = load_checkpoint(path, torch.device("cpu"))
        source = deepcopy(checkpoint)
        dataset, _ = synthetic_datasets()
        settings = BranchingConfig()
        same_a, same_b = paired_branches(checkpoint, dataset, settings, lr_factor=1.0)
        self.assertEqual(same_a["history"], same_b["history"])
        self.assertTrue(states_equal(same_a["final_state"], same_b["final_state"]))
        a, b = paired_branches(checkpoint, dataset, settings)
        self.assertTrue(states_equal(a["before"], b["before"]))
        self.assertEqual(a["orders"], b["orders"])
        self.assertEqual(len(a["orders"]), 5)
        self.assertEqual(sorted(sum(a["orders"][0], [])), sorted(data.train_indices))
        self.assertEqual(a["after"]["optimizer"]["param_groups"][0]["lr"], 0.1)
        self.assertEqual(b["after"]["optimizer"]["param_groups"][0]["lr"], 0.05)
        self.assertTrue(states_equal(checkpoint, source))

    def test_features_use_past_only_and_prior_absolute_best(self):
        settings = BranchingConfig()
        # 每次下降都不足 0.5%；不能把累计下降当成相对旧锚点的有效改善。
        history = [{"epoch": t, "train_loss": 6 - t, "val_loss": 1 - (t - 1) * 0.003}
                   for t in range(1, 6)]
        features = extract_features(history, 5, 30, settings)
        self.assertAlmostEqual(features["train_relative_slope"], -1 / (3 + settings.epsilon))
        self.assertEqual(features["epochs_since_improvement"], 0.4)
        self.assertEqual(features["training_progress"], 5 / 30)
        history.append({"epoch": 6, "train_loss": 999, "val_loss": 999})
        self.assertEqual(extract_features(history, 5, 30, settings), features)
        history[-1]["val_loss"] = 0.9
        self.assertEqual(extract_features(history, 6, 30, settings)["epochs_since_improvement"], 0)
        with self.assertRaises(ValueError):
            extract_features(history, 4, 30, settings)

    def test_gain_uses_last_two_and_strict_threshold(self):
        settings = BranchingConfig()
        a = [{"val_loss": x} for x in [100, 100, 100, 2, 2]]
        b = [{"val_loss": x} for x in [0, 0, 0, 1.8, 1.8]]
        outcome = action_outcome(a, b, settings)
        self.assertAlmostEqual(outcome["relative_gain"], 0.1)
        self.assertEqual(outcome["beneficial"], 1)
        self.assertEqual(action_outcome(a, a, settings)["beneficial"], 0)
        self.assertLess(action_outcome(b, a, settings)["relative_gain"], 0)
        settings.meaningful_improvement_threshold = outcome["relative_gain"]
        self.assertEqual(action_outcome(a, b, settings)["beneficial"], 0)


if __name__ == "__main__":
    unittest.main()

"""Synthetic-only tests; never download MNIST or run scientific screening."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from reflexml import collapse_external as ce
from reflexml.branching import BranchingConfig, order_hashes, run_branch, states_equal
from reflexml.checkpoint import load_checkpoint, save_checkpoint
from reflexml.data import make_loaders_from_datasets, make_split_indices, split_fingerprint
from reflexml.training import evaluate, train_one_epoch
from tests.test_reproducibility import build_experiment, synthetic_datasets


TRAIN, VAL = make_split_indices(60000, 10000, 5000, 2026)
ORDER = [TRAIN[start:start + 128] for start in range(0, 10000, 128)]
ORDERS = [ORDER] * 4


def records(gaps=(.125, .125, .125, 0), n=6, per_state=None, pre=1.):
    rows = []
    for i in range(1, n + 1):
        delta = gaps if per_state is None else per_state[i - 1]
        for k in (1, 2):
            for arm in ce.ARMS:
                rows.append({"protocol": ce.PROTOCOL, "state_id": ce.state_id(i),
                             "base_seed": 910000 + i, "future_id": k,
                             "future_seed": ce.future_seed(i, k), "arm": arm,
                             "dataset_identity": "d" * 64,
                             "config_hash": ce.identity(ce.frozen_config(i).to_dict()),
                             "split_hash": split_fingerprint(TRAIN, VAL),
                             "source_checkpoint_path": f"/fixture/{i}.pt",
                             "source_checkpoint_hash": f"{i:064x}",
                             "prebranch_validation_loss": pre,
                             "losses": list(delta) if arm == "Wait3" else [0.] * 4,
                             "learning_rates": [.05] * 4 if arm == "Now" else [.1, .1, .1, .05],
                             "orders": ORDERS, "order_hashes": order_hashes(ORDERS),
                             "success": True, "failure_reason": None})
    return rows


class DatasetTests(unittest.TestCase):
    def test_frozen_config_split_and_batch_geometry(self):
        self.assertEqual(make_split_indices(60000, 10000, 5000, 2026), (TRAIN, VAL))
        self.assertEqual((len(TRAIN), len(VAL), len(set(TRAIN) & set(VAL))), (10000, 5000, 0))
        config = ce.frozen_config(1)
        self.assertEqual((config.seed, config.dataset_name, config.epochs, config.hidden_size,
                          config.batch_size, config.momentum, config.learning_rate),
                         (910001, "MNIST", 14, 128, 128, .9, .1))
        # Integer population suffices to test sampler geometry without MNIST data.
        data = make_loaders_from_datasets(list(range(60000)), [], config)
        self.assertEqual(len(data.train_loader), 79)
        self.assertEqual([len(batch) for batch in data.train_loader], [128] * 78 + [16])
        self.assertFalse(data.train_loader.drop_last)
        self.assertEqual(len(data.test_loader), 0)

    def test_mnist_constructor_only_training_totensor(self):
        with patch.object(ce.datasets, "MNIST") as constructor:
            ce.load_mnist("/fixture/root")
        constructor.assert_called_once()
        kwargs = constructor.call_args.kwargs
        self.assertTrue(kwargs["train"])
        self.assertFalse(kwargs["download"])
        self.assertIs(type(kwargs["transform"]), transforms.ToTensor)

    def test_manifest_content_identity_and_reject_wrong_dataset_transform(self):
        # A synthetic MNIST object exercises identity logic without disk/network access.
        dataset = datasets.MNIST.__new__(datasets.MNIST)
        dataset.train = True
        dataset.root = "/fixture/root"
        dataset.data = torch.zeros(60000, 1, 1, dtype=torch.uint8)
        dataset.targets = torch.zeros(60000, dtype=torch.int64)
        dataset.transform = transforms.ToTensor()
        dataset.target_transform = None
        first = ce.dataset_manifest(dataset)
        self.assertEqual(first["dataset_class"], "torchvision.datasets.MNIST")
        self.assertFalse(first["official_test_enabled"])
        self.assertEqual(first["train_indices"], TRAIN)
        dataset.targets[0] = 1
        self.assertNotEqual(ce.dataset_manifest(dataset)["training_content_sha256"], first["training_content_sha256"])
        dataset.transform = transforms.Compose([transforms.ToTensor()])
        with self.assertRaises(ValueError):
            ce.dataset_manifest(dataset)
        dataset.transform = transforms.ToTensor()
        dataset.train = False
        with self.assertRaises(ValueError):
            ce.dataset_manifest(dataset)
        wrong = datasets.FashionMNIST.__new__(datasets.FashionMNIST)
        with self.assertRaises(ValueError):
            ce.dataset_manifest(wrong)


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        config, data, model, optimizer = build_experiment()
        train_one_epoch(model, data.train_loader, optimizer, nn.CrossEntropyLoss(), ce.CPU)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "base.pt"
            save_checkpoint(path, 14, model, optimizer, config, data.train_generator, data.split_fingerprint)
            cls.checkpoint = load_checkpoint(path, ce.CPU)
        cls.dataset, _ = synthetic_datasets()
        cls.pair = ce.run_pair(cls.checkpoint, cls.dataset, 920001)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_actual_pairing_momentum_and_independent_copies(self):
        source = deepcopy(self.checkpoint)
        now, wait = self.pair["Now"], self.pair["Wait3"]
        self.assertEqual(now["orders"], wait["orders"])
        self.assertTrue(states_equal(now["before"], wait["before"]))
        self.assertTrue(states_equal(now["before"]["optimizer"], source["optimizer_state_dict"]))
        self.assertEqual([r["learning_rate"] for r in now["history"]], [.05] * 4)
        self.assertEqual([r["learning_rate"] for r in wait["history"]], [.1, .1, .1, .05])
        self.assertTrue(states_equal(source, self.checkpoint))
        for arm in (now, wait):
            for key, tensor in arm["before"]["model"].items():
                self.assertNotEqual(tensor.data_ptr(), source["model_state_dict"][key].data_ptr())
                self.assertNotEqual(tensor.data_ptr(), arm["final_state"]["model"][key].data_ptr())
        first_buffer = next(iter(now["before"]["optimizer"]["state"].values()))["momentum_buffer"]
        other_buffer = next(iter(wait["before"]["optimizer"]["state"].values()))["momentum_buffer"]
        self.assertNotEqual(first_buffer.data_ptr(), other_buffer.data_ptr())

    def test_capture_disabled_exact_existing_output(self):
        future = ce.with_future_shuffle(self.checkpoint, 920001)
        old = run_branch(future, self.dataset, BranchingConfig(horizon=4), .5, end_epoch=18)
        explicit = run_branch(future, self.dataset, BranchingConfig(horizon=4), .5, end_epoch=18, capture_epoch=None)
        self.assertNotIn("captured_state", old)
        self.assertTrue(states_equal(old, explicit))
        captured = {k: v for k, v in self.pair["Now"].items() if k != "captured_state"}
        self.assertTrue(states_equal(old, captured))

    def test_capture_deep_copy_and_exact_h4_restore_both_arms(self):
        for name, original in self.pair.items():
            with self.subTest(arm=name):
                capture = original["captured_state"]
                self.assertEqual(capture["epoch"], 17)
                self.assertEqual(capture["epoch_end"]["optimizer"]["param_groups"][0]["lr"],
                                 .05 if name == "Now" else .1)
                self.assertEqual(capture["next_epoch_start"]["optimizer"]["param_groups"][0]["lr"], .05)
                frozen = deepcopy(capture)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "h3.pt"
                    torch.save(capture, path)
                    restored = load_checkpoint(path, ce.CPU)
                replay = run_branch(ce.replay_checkpoint(restored), self.dataset, BranchingConfig(horizon=1),
                                    1., end_epoch=18)
                self.assertEqual(replay["orders"][0], capture["next_epoch_order"])
                self.assertEqual(replay["history"][0], original["history"][3])
                self.assertTrue(states_equal(replay["before"], capture["next_epoch_start"]))
                self.assertTrue(states_equal(replay["final_state"], original["final_state"]))
                self.assertTrue(states_equal(capture, frozen))
                self.assertTrue(states_equal(capture["next_epoch_start"]["rng"], original["epoch_start_random_states"][3]["rng"]))
                for key, tensor in capture["epoch_end"]["model"].items():
                    self.assertNotEqual(tensor.data_ptr(), original["final_state"]["model"][key].data_ptr())
                    self.assertFalse(torch.equal(tensor, original["final_state"]["model"][key]))
                for key, item in capture["epoch_end"]["optimizer"]["state"].items():
                    self.assertNotEqual(item["momentum_buffer"].data_ptr(),
                                        original["final_state"]["optimizer"]["state"][key]["momentum_buffer"].data_ptr())

    def test_state_artifact_persistence_wiring_without_base_training(self):
        config, _, _, _ = build_experiment()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            # Mock the base loop only; branch artifacts are real tiny-fixture kernel outputs.
            def fake_epoch(model, loader, optimizer, criterion, device, order):
                order.extend(ORDER)
                return 1.
            with patch.object(ce, "frozen_config", return_value=config), patch.object(ce, "train_one_epoch", side_effect=fake_epoch) as train, patch.object(ce, "evaluate", return_value=type("Validation", (), {"loss": 1.})()), patch.object(ce, "run_pair", return_value=self.pair) as pair:
                rows = ce._run_state(1, self.dataset, {"fixture": True}, output)
            self.assertEqual(train.call_count, 14)
            self.assertEqual([call.args[2] for call in pair.call_args_list], [920001, 920002])
            self.assertEqual(len(rows), 4)
            metadata = json.loads((output / "state-001" / "state.json").read_text())
            self.assertEqual(metadata["exact_update_count"], 1106)
            self.assertEqual(ce.file_hash(metadata["source_checkpoint_path"]), metadata["source_checkpoint_hash"])
            for row in rows:
                artifact = row["h3_artifact"]
                self.assertEqual(ce.file_hash(artifact["path"]), artifact["sha256"])
                capture = load_checkpoint(Path(artifact["path"]), ce.CPU)
                self.assertEqual(capture["source_checkpoint_hash"], metadata["source_checkpoint_hash"])
                self.assertEqual(capture["config_hash"], metadata["config_hash"])
                self.assertTrue(states_equal(capture["original_h4_endpoint"], self.pair[row["arm"]]["final_state"]))
                self.assertEqual(capture["next_epoch_order"], row["orders"][3])
                self.assertEqual(capture["replay_checkpoint"]["current_learning_rate"], .05)
                self.assertIn("execution", capture)

    def test_invalid_capture_epoch(self):
        for epoch in (14, 18, 19, 17.5):
            with self.assertRaises(ValueError):
                run_branch(self.checkpoint, self.dataset, BranchingConfig(horizon=4), .5,
                           end_epoch=18, capture_epoch=epoch)

    def test_validation_weighted_and_deterministic(self):
        logits = torch.tensor([[4., 0.], [4., 0.], [0., 4.]])
        labels = torch.tensor([0, 0, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        criterion = nn.CrossEntropyLoss()
        first = evaluate(nn.Identity(), loader, criterion, ce.CPU)
        self.assertEqual(first, evaluate(nn.Identity(), loader, criterion, ce.CPU))
        batches = [criterion(logits[:2], labels[:2]).item(), criterion(logits[2:], labels[2:]).item()]
        self.assertEqual(first.loss, (2 * batches[0] + batches[1]) / 3)
        self.assertNotEqual(first.loss, sum(batches) / 2)


class ClassificationTests(unittest.TestCase):
    def test_all_valid_labels_and_estimands(self):
        cases = [((.125, .125, .125, 0), "GO"),
                 ((.125, .125, .125, .125), "NO_GO_COLLAPSE_ABSENT"),
                 ((-.125, -.125, -.125, 0), "NO_GO_DIRECTION_REVERSED"),
                 ((0, 0, 0, 0), "UNINFORMATIVE"),
                 ((.125, 0, .125, .0625), "AMBIGUOUS")]
        for gap, label in cases:
            with self.subTest(label=label):
                result = ce.classify(records(gap))
                self.assertEqual(result["classification"], label)
                self.assertEqual(result["m_h"], list(gap))
                self.assertEqual(result["A34"], gap[3] - gap[2])
                self.assertEqual(result["q_n"], 5)

    def test_signed_collapse_absent_and_non_canceling_uninformative(self):
        self.assertEqual(ce.classify(records((.125, .125, .125, -1)))["classification"], "AMBIGUOUS")
        self.assertEqual(ce.classify(records((1, -1, 0, 0)))["classification"], "AMBIGUOUS")

    def test_invalid_conditions(self):
        base = records()
        mutations = {
            "missing": lambda rows: rows.pop(),
            "duplicate": lambda rows: rows.append(rows[0]),
            "nan": lambda rows: rows[0]["losses"].__setitem__(0, float("nan")),
            "pre_nan": lambda rows: rows[0].__setitem__("prebranch_validation_loss", float("nan")),
            "checkpoint": lambda rows: rows[0].__setitem__("source_checkpoint_hash", "a" * 64),
            "path": lambda rows: rows[0].__setitem__("source_checkpoint_path", "/wrong"),
            "dataset": lambda rows: rows[0].__setitem__("dataset_identity", "a" * 64),
            "config": lambda rows: rows[0].__setitem__("config_hash", "a" * 64),
            "split": lambda rows: rows[0].__setitem__("split_hash", "a" * 64),
            "seed": lambda rows: rows[0].__setitem__("future_seed", 123),
            "lr": lambda rows: rows[0].__setitem__("learning_rates", [.1] * 4),
            "failure": lambda rows: rows[0].__setitem__("success", False),
            "empty_order": lambda rows: rows[0].__setitem__("orders", []),
            "wrong_hash": lambda rows: rows[0].__setitem__("order_hashes", ["a" * 64] * 4),
            "pre_mismatch": lambda rows: rows[0].__setitem__("prebranch_validation_loss", 2.),
            "malformed": lambda rows: rows[0].pop("protocol"),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                rows = deepcopy(base)
                mutate(rows)
                self.assertEqual(ce.classify(rows)["classification"], "INVALID")
        # Valid permutation, wrong pair: hashing alone must not mask mismatch.
        rows = list(base)
        rows[0] = deepcopy(rows[0])
        rows[0]["orders"][0][0].reverse()
        rows[0]["order_hashes"] = order_hashes(rows[0]["orders"])
        self.assertIn("Actual paired minibatch sequences differ", ce.classify(rows)["errors"])
        self.assertEqual(ce.classify(base, 7)["classification"], "INVALID")

    def test_exact_thresholds(self):
        # Binary-exact epsilon = .125, no floating tolerance.
        def classify(gap):
            return ce.classify(records(gap, pre=12.5))["classification"]
        self.assertEqual(classify((.125, .125, .125, 0)), "UNINFORMATIVE")
        self.assertEqual(classify((-.125, -.125, -.125, 0)), "UNINFORMATIVE")
        self.assertEqual(classify((.25, .25, .25, .125)), "GO")  # abs(m4)/E == .5
        self.assertEqual(classify((.1875, .375, .375, 0)), "GO")  # max/min == 2
        self.assertEqual(classify((.25, .375, .25, .125)), "AMBIGUOUS")  # adjacent equality
        self.assertEqual(classify((.25, .25, .25, -.25)), "AMBIGUOUS")
        # .8 ratio boundary uses exactly the same frozen arithmetic.
        self.assertEqual(ce.classify(records((.125, .125, .125, .8 * .125)))["classification"],
                         "NO_GO_COLLAPSE_ABSENT")

    def test_state_counts_q6_q12(self):
        for n, q in ((6, 5), (12, 10)):
            for label, good, bad in [
                ("GO", (.125, .125, .125, 0), (.125, .125, .125, .25)),
                ("NO_GO_COLLAPSE_ABSENT", (.125, .125, .125, .25), (.125, .125, .125, 0)),
                ("NO_GO_DIRECTION_REVERSED", (-.125, -.125, -.125, 0), (0, 0, 0, 0)),
                ("UNINFORMATIVE", (0, 0, 0, 0), (.03125, .03125, .03125, 0))]:
                for count in (q - 1, q):
                    with self.subTest(n=n, label=label, count=count):
                        rows = records(n=n, per_state=[good] * count + [bad] * (n - count))
                        epsilon = ce.freeze_epsilon(rows[:24])
                        result = ce.classify(rows, n, epsilon)
                        self.assertEqual(result["q_n"], q)
                        self.assertEqual(result["classification"], label if count == q else "AMBIGUOUS")

    def test_epsilon_frozen_with_n12_and_provenance(self):
        initial = records()
        epsilon = ce.freeze_epsilon(initial)
        expanded = records(n=12)
        for row in expanded[24:]:
            row["prebranch_validation_loss"] = 1000.
        result = ce.classify(expanded, 12, epsilon)
        self.assertEqual(result["classification"], "GO")
        self.assertEqual(result["epsilon_record"], epsilon)
        self.assertEqual(len(epsilon["sources"]), 6)
        self.assertEqual(epsilon["epsilon"], .01)
        self.assertEqual(ce.classify(expanded, 12)["classification"], "INVALID")
        corrupt = deepcopy(epsilon)
        corrupt["epsilon"] *= 2
        self.assertEqual(ce.classify(expanded, 12, corrupt)["classification"], "INVALID")
        self.assertEqual(ce.freeze_epsilon(records(pre=0))["epsilon"], .001)

    def test_future_replicas_average_before_state_mean(self):
        rows = records()
        rows[3] = {**rows[3], "losses": [.375, .375, .375, 0]}
        result = ce.classify(rows)
        self.assertEqual(result["d_ih"][0], [.25, .25, .25, 0])
        self.assertEqual(result["E"], (.25 + 5 * .125) / 6)


class ExecutionBoundaryTests(unittest.TestCase):
    def test_escalation_frozen(self):
        labels = ["GO", "NO_GO_COLLAPSE_ABSENT", "NO_GO_DIRECTION_REVERSED", "UNINFORMATIVE", "INVALID", "AMBIGUOUS"]
        for label in labels:
            self.assertEqual(ce.escalation_decision(label, 6),
                             "ELIGIBLE_N12" if label == "AMBIGUOUS" else "TECHNICAL_REVIEW" if label == "INVALID" else "STOP")
        self.assertEqual(ce.escalation_decision("AMBIGUOUS", 12, expansions=1), "STOP")
        self.assertEqual(ce.escalation_decision("AMBIGUOUS", 6, expansions=1), "STOP")
        for kwargs in ({"n": 18}, {"n": 6, "k": 3}, {"n": 6, "expansions": 2}):
            with self.assertRaises(ValueError):
                ce.escalation_decision("AMBIGUOUS", **kwargs)
        self.assertEqual([ce.future_seed(12, k) for k in (1, 2)], [920023, 920024])
        with self.assertRaises(ValueError):
            ce.future_seed(1, 3)

    def test_cli_requires_explicit_execution_choice(self):
        import run_collapse_external as cli
        with patch("sys.argv", ["runner", "--data-root", "/fixture", "--output", "/fixture/output"]), patch.object(cli, "run_replication") as run:
            with self.assertRaises(SystemExit):
                cli.main()
            run.assert_not_called()

    def test_screening_orchestration_no_auto_expansion_or_overwrite(self):
        all_rows = records()
        for row in all_rows:
            row["dataset_identity"] = ce.identity({"fixture": True})
        with TemporaryDirectory() as directory:
            output = Path(directory) / "screen"
            with patch.object(ce, "load_mnist") as load, patch.object(ce, "dataset_manifest", return_value={"fixture": True}), patch.object(ce, "_run_state", side_effect=lambda i, *args: all_rows[(i - 1) * 4:i * 4]) as run:
                result = ce.run_replication("/fixture", output)
            self.assertEqual(result["classification"], "GO")
            self.assertEqual([call.args[0] for call in run.call_args_list], list(range(1, 7)))
            self.assertEqual(json.loads((output / "analysis.json").read_text()), result)
            with self.assertRaises(FileExistsError):
                ce.run_replication("/fixture", output)
            with self.assertRaises(ValueError):
                ce.run_replication("/fixture", Path(directory) / "exp", initial_output=output)
            load.assert_called_once()

    def test_expansion_reuses_six_once_and_never_recomputes_epsilon(self):
        all_rows = records((.125, 0, .125, .0625), n=12)
        for row in all_rows:
            row["dataset_identity"] = ce.identity({"fixture": True})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for row in all_rows:
                i = int(row["state_id"].split("-")[1])
                path = root / f"base-{i}.pt"
                path.write_bytes(f"synthetic source {i}".encode())
                row["source_checkpoint_path"] = str(path)
                row["source_checkpoint_hash"] = ce.file_hash(path)
            initial = root / "initial"
            initial.mkdir()
            ce.write_json(initial / "manifest.json", {"fixture": True})
            ce.write_json(initial / "records.json", all_rows[:24])
            saved = ce.classify(all_rows[:24])
            ce.write_json(initial / "analysis.json", saved)
            for row in all_rows[24:]:
                row["prebranch_validation_loss"] = 1000.
            with patch.object(ce, "load_mnist"), patch.object(ce, "dataset_manifest", return_value={"fixture": True}), patch.object(ce, "_run_state", side_effect=lambda i, *args: all_rows[(i - 1) * 4:i * 4]) as run:
                result = ce.run_replication("/fixture", root / "expanded", initial_output=initial)
                self.assertEqual([call.args[0] for call in run.call_args_list], list(range(7, 13)))
                self.assertEqual(result["epsilon_record"], saved["epsilon_record"])
                self.assertEqual(result["classification"], "AMBIGUOUS")
                with self.assertRaises(FileExistsError):
                    ce.run_replication("/fixture", root / "second", initial_output=initial)
            with self.assertRaises(ValueError):
                ce.run_replication("/fixture", root / "third", initial_output=root / "expanded")

    def test_failure_persisted_invalid_without_retry(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with patch.object(ce, "load_mnist"), patch.object(ce, "dataset_manifest", return_value={"fixture": True}), patch.object(ce, "_run_state", side_effect=RuntimeError("fixture failure")) as run:
                with self.assertRaises(RuntimeError):
                    ce.run_replication("/fixture", output)
            self.assertEqual(run.call_count, 1)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["classification"], "INVALID")
            self.assertIn("fixture failure", failure["failure_reason"])


if __name__ == "__main__":
    unittest.main()

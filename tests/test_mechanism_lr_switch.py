"""Synthetic-only checks. No frozen Phase 5 unit is trained here."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from reflexml.branching import order_hashes, states_equal, training_state_finiteness_evidence
from reflexml.config import ExperimentConfig
from reflexml.data import make_loaders_from_datasets
from reflexml.mechanism_lr_switch import (
    APPROVED_STAGE_I_GATE_SHA256, PROTOCOL_ID, PROTOCOL_SHA256, Unit, StageIPass,
    _FROZEN_UNIT_SEAL, _GATE_SEAL,
    _iterate_order, _write_record, analyze_completed,
    check_switch, load_frozen_units, reconstruct_epoch17,
    reconstruct_epoch18_generator, resume_stage_i, run_one_epoch, run_stage_i, run_stage_ii,
)
from reflexml.model import FashionMLP


class MechanismSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(train_size=8, val_size=4, batch_size=4, hidden_size=4)
        images = torch.arange(12 * 28 * 28, dtype=torch.float32).reshape(12, 1, 28, 28) / 10000
        labels = torch.arange(12) % 10
        self.dataset = TensorDataset(images, labels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = FashionMLP(4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(images[:4]), labels[:4]).backward()
        optimizer.step()
        evidence = training_state_finiteness_evidence(model, optimizer)
        data = make_loaders_from_datasets(self.dataset, [], self.config, audit_order=True)
        seed = 93271
        data.train_generator.manual_seed(seed)
        orders = [_iterate_order(data.train_loader) for _ in range(4)]
        history = [
            {"epoch": epoch, "learning_rate": 0.1 if epoch < 18 else 0.05,
             "train_loss": 0.0, "val_loss": 0.0, "val_accuracy": 0.0}
            for epoch in range(15, 19)
        ]
        historical = {
            "wait_history": history,
            "wait_state_finiteness_evidence": [
                {"epoch": epoch, **deepcopy(evidence)} for epoch in range(15, 19)
            ],
            "wait_realized_orders": orders,
            "now_realized_orders": deepcopy(orders),
            "wait_epoch_order_sha256": order_hashes(orders),
        }
        checkpoint = {
            "config": self.config.to_dict(),
            "model_state_dict": deepcopy(model.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            "split_fingerprint": data.split_fingerprint,
        }
        self.unit = Unit(1, 1, seed, checkpoint, historical, "toy-artifact", "toy-checkpoint")

    def test_reconstruction_generator_and_branch_isolation(self):
        with patch("reflexml.mechanism_lr_switch._configuration", return_value=self.config):
            model_state, optimizer_state = reconstruct_epoch17(self.unit)
            canonical = reconstruct_epoch18_generator(self.unit, self.dataset)
            before_model = deepcopy(model_state)
            before_opt = deepcopy(optimizer_state)
            before_generator = canonical.clone()
            switch = run_one_epoch(self.unit, self.dataset, canonical, model_state,
                                   optimizer_state, action="Switch")
            self.assertTrue(states_equal(model_state, before_model))
            self.assertTrue(states_equal(optimizer_state, before_opt))
            self.assertTrue(torch.equal(canonical, before_generator))
            cont = run_one_epoch(self.unit, self.dataset, canonical, model_state,
                                 optimizer_state, action="Continue")
        self.assertEqual(switch["realized_order"], cont["realized_order"])
        self.assertEqual(switch["epoch_order_sha256"], cont["epoch_order_sha256"])
        self.assertNotEqual(switch["learning_rate"], cont["learning_rate"])
        self.assertTrue(states_equal(model_state, before_model))
        self.assertTrue(states_equal(optimizer_state, before_opt))
        self.assertTrue(torch.equal(canonical, before_generator))
        history18 = self.unit.historical["wait_history"][3]
        for key in ("train_loss", "val_loss", "val_accuracy"):
            history18[key] = switch[key]
        self.unit.historical["wait_state_finiteness_evidence"][3].update(
            deepcopy(switch["state_evidence"])
        )
        check_switch(self.unit, switch)
        tampered = deepcopy(switch)
        tampered["state_evidence"]["model_parameters"][0]["state_npy_base64"] = "x"
        with self.assertRaises(ValueError):
            check_switch(self.unit, tampered)

    def test_incomplete_gate_blocks_training(self):
        with self.assertRaisesRegex(ValueError, "exact 72"):
            run_stage_i([self.unit], self.dataset, ".", "/unused")
        with self.assertRaises(PermissionError):
            run_stage_ii([self.unit], self.dataset, ".", None, "/unused")

    def test_momentum_mapping_rejects_missing_buffer(self):
        self.unit.historical["wait_state_finiteness_evidence"][2]["optimizer_state"].pop()
        with patch("reflexml.mechanism_lr_switch._configuration", return_value=self.config):
            with self.assertRaisesRegex(ValueError, "momentum inventory"):
                reconstruct_epoch17(self.unit)

    def _synthetic_analysis_inputs(self):
        units = []
        switch_rows = []
        continue_rows = []
        for state in range(1, 37):
            for replica in (1, 2):
                unit = deepcopy(self.unit)
                object.__setattr__(unit, "base_run_id", state)
                object.__setattr__(unit, "replica_id", replica)
                switch_train = 0.2 + state * 0.0002 + replica * 0.0001
                switch_accuracy = 0.8 + state * 0.0001 + replica * 0.0002
                unit.historical["wait_history"][2]["val_loss"] = 0.4
                unit.historical["wait_history"][3]["val_loss"] = 0.35
                unit.historical["wait_history"][3]["train_loss"] = switch_train
                unit.historical["wait_history"][3]["val_accuracy"] = switch_accuracy
                units.append(unit)
                common = {
                    "base_run_id": state, "replica_id": replica,
                    "source_sha256": unit.source_sha256,
                    "checkpoint_sha256": unit.checkpoint_sha256,
                    "future_seed": unit.future_seed, "epoch": 18,
                    "realized_order": unit.historical["wait_realized_orders"][3],
                    "epoch_order_sha256": unit.historical["wait_epoch_order_sha256"][3],
                }
                switch_rows.append({
                    **common, "action": "Switch", "learning_rate": 0.05,
                    "train_loss": switch_train, "val_loss": 0.35,
                    "val_accuracy": switch_accuracy,
                    "state_evidence": deepcopy(unit.historical["wait_state_finiteness_evidence"][3]),
                })
                switch_rows[-1]["state_evidence"].pop("epoch")
                continue_rows.append({
                    **common, "action": "Continue", "learning_rate": 0.1,
                    "val_loss": 0.36 + state * 0.0001,
                    "train_loss": switch_train + state * 0.001 +
                                  (0.01 if replica == 1 else -0.005),
                    "val_accuracy": switch_accuracy - state * 0.001 +
                                    (0.02 if replica == 1 else -0.01),
                })
        required = frozenset(unit.key for unit in units)
        gate = StageIPass(required, tuple(switch_rows), _GATE_SEAL)
        return units, gate, continue_rows

    def test_frozen_state_level_analysis_on_synthetic_values(self):
        units, gate, continue_rows = self._synthetic_analysis_inputs()
        result = analyze_completed(units, gate, continue_rows)
        self.assertEqual(result["n_states"], 36)
        self.assertEqual(result["bootstrap_replicates"], 10_000)
        self.assertAlmostEqual(result["delta_switch"], 0.01185)
        self.assertAlmostEqual(
            result["D"], result["kappa_C"] + result["delta_switch"]
        )
        self.assertGreater(result["ci_95"]["delta_switch"][0], 0)
        primary_state_values = 0.01 + np.arange(1, 37) * 0.0001
        rng = np.random.Generator(np.random.PCG64(12114954806208443255))
        indices = rng.integers(0, 36, size=(10_000, 36))
        expected_primary_ci = np.percentile(
            primary_state_values[indices].mean(axis=1), [2.5, 97.5]
        )
        np.testing.assert_allclose(
            result["ci_95"]["delta_switch"], expected_primary_ci, atol=1e-15
        )

    def test_secondary_sign_state_means_and_frozen_bootstrap(self):
        units, gate, continue_rows = self._synthetic_analysis_inputs()
        result = analyze_completed(units, gate, continue_rows)
        secondary = result["secondary_outcomes"]
        train = secondary["epoch18_train_loss"]
        accuracy = secondary["epoch18_val_accuracy"]
        self.assertEqual(train["contrast"], "Continue - Switch")
        self.assertEqual(accuracy["contrast"], "Continue - Switch")
        state_ids = np.arange(1, 37, dtype=np.float64)
        expected_state_means = np.column_stack((
            state_ids * 0.001 + 0.0025,
            -state_ids * 0.001 + 0.005,
        ))
        self.assertAlmostEqual(train["estimate"], expected_state_means[:, 0].mean())
        self.assertAlmostEqual(accuracy["estimate"], expected_state_means[:, 1].mean())
        self.assertLess(accuracy["estimate"], 0)  # Switch accuracy is higher.
        rng = np.random.Generator(np.random.PCG64(12114954806208443255))
        indices = rng.integers(0, 36, size=(10_000, 36))
        expected_ci = np.percentile(
            expected_state_means[indices].mean(axis=1), [2.5, 97.5], axis=0
        )
        np.testing.assert_allclose(train["ci_95"], expected_ci[:, 0], atol=1e-15)
        np.testing.assert_allclose(accuracy["ci_95"], expected_ci[:, 1], atol=1e-15)

    def test_secondary_changes_cannot_change_primary_outputs(self):
        units, gate, continue_rows = self._synthetic_analysis_inputs()
        baseline = analyze_completed(units, gate, continue_rows)
        changed_rows = deepcopy(continue_rows)
        for row in changed_rows:
            row["train_loss"] += 0.07
            row["val_accuracy"] -= 0.03
        changed = analyze_completed(units, gate, changed_rows)
        self.assertNotEqual(baseline["secondary_outcomes"], changed["secondary_outcomes"])
        baseline.pop("secondary_outcomes")
        changed.pop("secondary_outcomes")
        self.assertEqual(baseline, changed)

    def test_secondary_analysis_preserves_identity_rejections(self):
        units, gate, continue_rows = self._synthetic_analysis_inputs()
        with self.assertRaisesRegex(ValueError, "exact frozen unit population"):
            analyze_completed(units[:-1], gate, continue_rows)
        with self.assertRaisesRegex(ValueError, "72 Continue outcomes"):
            analyze_completed(units, gate, continue_rows[:-1])
        duplicated_continue = deepcopy(continue_rows)
        duplicated_continue[-1] = deepcopy(duplicated_continue[0])
        with self.assertRaisesRegex(ValueError, "Continue outcome identity/coverage mismatch"):
            analyze_completed(units, gate, duplicated_continue)
        duplicated_switch = list(gate.results)
        duplicated_switch[-1] = deepcopy(duplicated_switch[0])
        duplicate_gate = StageIPass(gate.keys, tuple(duplicated_switch), _GATE_SEAL)
        with self.assertRaisesRegex(ValueError, "Stage I result identity/coverage mismatch"):
            analyze_completed(units, duplicate_gate, continue_rows)

    def test_mocked_72_unit_gate_and_stage_order(self):
        units = [replace(self.unit, base_run_id=state, replica_id=replica,
                         _seal=_FROZEN_UNIT_SEAL)
                 for state in range(1, 37) for replica in (1, 2)]

        def fake_result(unit, _dataset, _generator, _model, _optimizer, *, action):
            return {"base_run_id": unit.base_run_id, "replica_id": unit.replica_id,
                    "source_sha256": unit.source_sha256,
                    "checkpoint_sha256": unit.checkpoint_sha256,
                    "future_seed": unit.future_seed, "action": action}

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch("reflexml.mechanism_lr_switch.validate_production_dataset"),
                  patch("reflexml.mechanism_lr_switch.reconstruct_epoch17", return_value=({}, {})),
                  patch("reflexml.mechanism_lr_switch.reconstruct_epoch18_generator", return_value=torch.zeros(1)),
                  patch("reflexml.mechanism_lr_switch.run_one_epoch", side_effect=fake_result),
                  patch("reflexml.mechanism_lr_switch.check_switch")):
                gate = run_stage_i(units, self.dataset, root, root / "stage_i")
                self.assertEqual(len(gate.results), 72)
                self.assertTrue(gate.manifest_path.is_file())
                self.assertEqual(len(list((root / "stage_i").glob("switch-*.json"))), 72)
                results = run_stage_ii(units, self.dataset, root, gate, root / "stage_ii")
                self.assertEqual(len(results), 72)
                self.assertEqual(len(list((root / "stage_ii").glob("continue-*.json"))), 72)


class StageIResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.units = []
        self.manifest = {
            "protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
            "stage": "I", "status": "72/72 PASS", "unit_count": 72,
            "switch_records": {},
        }
        for state in range(1, 37):
            for replica in (1, 2):
                historical = {
                    "wait_history": [{"epoch": epoch, "learning_rate": 0.05,
                                      "train_loss": 0.2, "val_loss": 0.3,
                                      "val_accuracy": 0.8} for epoch in range(15, 19)],
                    "wait_realized_orders": [[], [], [], [[0, 1]]],
                    "wait_epoch_order_sha256": ["", "", "", order_hashes([[[0, 1]]])[0]],
                    "wait_state_finiteness_evidence": [{}, {}, {}, {
                        "model_parameters": [], "optimizer_state": []}],
                }
                unit = Unit(state, replica, state * 100 + replica, {}, historical,
                            f"{state * 100 + replica:064x}", f"{state:064x}",
                            _FROZEN_UNIT_SEAL)
                self.units.append(unit)
                result = {
                    "base_run_id": state, "replica_id": replica,
                    "source_sha256": unit.source_sha256,
                    "checkpoint_sha256": unit.checkpoint_sha256,
                    "future_seed": unit.future_seed,
                    "action": "Switch", "epoch": 18, "learning_rate": 0.05,
                    "realized_order": [[0, 1]],
                    "epoch_order_sha256": historical["wait_epoch_order_sha256"][3],
                    "train_loss": 0.2, "val_loss": 0.3, "val_accuracy": 0.8,
                    "state_evidence": {"model_parameters": [], "optimizer_state": []},
                }
                _write_record(self.root, unit, "Switch", "PASS", result)
                name = f"switch-{state:03d}-A-r{replica:02d}.json"
                self.manifest["switch_records"][name] = self.digest(self.root / name)
        self.save_manifest()
        self.fixture_approved_sha256 = self.gate_sha256
        gate_authority = patch("reflexml.mechanism_lr_switch.APPROVED_STAGE_I_GATE_SHA256",
                               self.fixture_approved_sha256)
        gate_authority.start()
        self.addCleanup(gate_authority.stop)
        protocol_check = patch("reflexml.mechanism_lr_switch.verify_protocol")
        protocol_check.start()
        self.addCleanup(protocol_check.stop)

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_manifest(self):
        self.gate_path = self.root / "stage_i_gate.json"
        self.gate_path.write_text(json.dumps(self.manifest) + "\n")
        self.gate_sha256 = self.digest(self.gate_path)

    def resume(self):
        with patch("reflexml.mechanism_lr_switch.APPROVED_STAGE_I_GATE_SHA256",
                   self.gate_sha256):
            return resume_stage_i(self.units, self.root, self.root, self.gate_sha256)

    def change_record(self, name, change):
        path = self.root / name
        record = json.loads(path.read_text())
        change(record)
        path.write_text(json.dumps(record) + "\n")
        self.manifest["switch_records"][name] = self.digest(path)
        self.save_manifest()

    def test_valid_gate_and_stage_ii_gate_check(self):
        gate = self.resume()
        self.assertEqual(len(gate.results), 72)
        self.assertEqual(gate.keys, frozenset(unit.key for unit in self.units))
        with patch("reflexml.mechanism_lr_switch.validate_production_dataset",
                   side_effect=RuntimeError("Stage II gate accepted")):
            with self.assertRaisesRegex(RuntimeError, "Stage II gate accepted"):
                run_stage_ii(self.units, None, self.root, gate, self.root / "stage_ii")
        self.assertFalse((self.root / "stage_ii").exists())

    def test_alternate_valid_manifest_with_its_correct_hash_rejected(self):
        self.assertEqual(len(self.resume().results), 72)
        self.gate_path.write_text(self.gate_path.read_text() + " ")
        alternate_sha256 = self.digest(self.gate_path)
        self.assertNotEqual(alternate_sha256, self.fixture_approved_sha256)
        with self.assertRaisesRegex(PermissionError, "not the approved frozen gate"):
            resume_stage_i(self.units, self.root, self.root, alternate_sha256)

    def test_71_of_72_rejected(self):
        self.manifest["switch_records"].pop("switch-036-A-r02.json")
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.resume()

    def test_missing_identity_rejected(self):
        name = "switch-036-A-r02.json"
        path = self.root / name
        path.unlink()
        replacement = "switch-037-A-r02.json"
        (self.root / replacement).write_text("{}\n")
        self.manifest["switch_records"].pop(name)
        self.manifest["switch_records"][replacement] = self.digest(self.root / replacement)
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.resume()

    def test_duplicate_or_substituted_identity_rejected(self):
        name = "switch-036-A-r02.json"
        self.change_record(name, lambda row: row.update(base_run_id=1, replica_id=1))
        with self.assertRaises(ValueError):
            self.resume()

    def test_fail_unit_rejected(self):
        self.change_record("switch-001-A-r01.json", lambda row: row.update(status="INTEGRITY_GATE_FAIL"))
        with self.assertRaises(ValueError):
            self.resume()

    def test_omitted_failure_record_rejected(self):
        (self.root / "switch-extra.json").write_text('{"status": "INTEGRITY_GATE_FAIL"}\n')
        with self.assertRaises(ValueError):
            self.resume()

    def test_rehashed_mismatched_result_rejected(self):
        self.change_record("switch-001-A-r01.json",
                           lambda row: row["result"].update(learning_rate=0.10))
        with self.assertRaisesRegex(ValueError, "exact Switch reproduction mismatch"):
            self.resume()

    def test_corrupt_gate_hash_and_evidence_rejected(self):
        self.gate_path.write_text(self.gate_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.resume()
        self.save_manifest()
        self.manifest["status"] = "71/72 PASS"
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.resume()

    def test_corrupt_referenced_record_rejected(self):
        path = self.root / "switch-001-A-r01.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.resume()

    def test_wrong_protocol_identity_and_hash_rejected(self):
        for field in ("protocol_id", "protocol_sha256"):
            with self.subTest(field=field):
                self.manifest[field] = "wrong"
                self.save_manifest()
                with self.assertRaises(ValueError):
                    self.resume()
                self.manifest[field] = PROTOCOL_ID if field == "protocol_id" else PROTOCOL_SHA256
        self.change_record("switch-001-A-r01.json",
                           lambda row: row.update(protocol_sha256="wrong"))
        with self.assertRaises(ValueError):
            self.resume()

    def test_unvalidated_object_cannot_bypass_stage_ii(self):
        fabricated = StageIPass(frozenset(unit.key for unit in self.units),
                                tuple({} for _ in self.units), object(),
                                self.gate_path, self.gate_sha256)
        with self.assertRaises(PermissionError):
            run_stage_ii(self.units, None, self.root, fabricated, self.root / "stage_ii")
        self.assertFalse((self.root / "stage_ii").exists())


class ApprovedStageIGateTests(unittest.TestCase):
    def test_real_approved_gate_resumes_read_only(self):
        repo = Path(__file__).resolve().parents[1]
        stage_i = Path("internal-artifacts/reflexml-mechanism-lr-switch-v1-stage-i-20260926-001")
        ledger = Path("internal-artifacts/branch_attempts.jsonl")
        base = Path("internal-artifacts/phase5_v1_base_v2")
        if not (stage_i / "stage_i_gate.json").is_file() or not ledger.is_file() or not base.is_dir():
            self.skipTest("Approved production Stage I evidence is unavailable")
        self.assertEqual(APPROVED_STAGE_I_GATE_SHA256,
                         "dd7d4273a215f1c71cdf816bce3334dc846c37e31b92f0e29e46b7748d67dc80")
        units = load_frozen_units(repo, ledger, base)
        gate = resume_stage_i(units, repo, stage_i, APPROVED_STAGE_I_GATE_SHA256)
        self.assertEqual(len(gate.results), 72)
        self.assertEqual(gate.manifest_sha256, APPROVED_STAGE_I_GATE_SHA256)


if __name__ == "__main__":
    unittest.main()

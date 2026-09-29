"""Synthetic Momentum-Reset checks; no historical unit is trained."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from reflexml.branching import order_hashes, states_equal, training_state_finiteness_evidence
from reflexml.config import ExperimentConfig
from reflexml.data import make_loaders_from_datasets
from reflexml.mechanism_lr_switch import Unit, _FROZEN_UNIT_SEAL, _iterate_order
from reflexml.model import FashionMLP
from reflexml.momentum_reset import (
    _run_prepared, _write_prestep, _json_restore, _json_safe, analyze_factorial,
    branch_identity,
    build_completion_manifest, canonical_roster_sha256, kernel_provenance,
    load_canonical_roster, observe_artifact_io_failure,
    observe_source_hash_mismatch, read_scientific_result,
    prepare_reset, reset_momentum_buffers, run_production_reset,
    run_synthetic_reset, write_attempt,
)
from reflexml import momentum_reset, mechanism_lr_switch, training


class MomentumResetSyntheticTests(unittest.TestCase):
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
        historical = {
            "wait_history": [
                {"epoch": epoch, "learning_rate": 0.1 if epoch < 18 else 0.05,
                 "train_loss": 0.0, "val_loss": 0.0, "val_accuracy": 0.0}
                for epoch in range(15, 19)
            ],
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
        self.unit = Unit(1, 1, seed, checkpoint, historical,
                         "toy-artifact", "toy-checkpoint")

    def prepare(self, treatment):
        with patch("reflexml.mechanism_lr_switch._configuration", return_value=self.config):
            return prepare_reset(self.unit, self.dataset, treatment=treatment)

    def observed_io_failure(self, directory: Path):
        try:
            directory.write_bytes(b"cannot write bytes over a directory")
        except OSError as error:
            return observe_artifact_io_failure(error)
        self.fail("Expected an actual directory write failure")

    def official_fixture(self, root: Path, treatment: str = "L_05R"):
        artifact, checkpoint = root / "source.bin", root / "checkpoint.bin"
        artifact.write_bytes(b"synthetic source bytes")
        checkpoint.write_bytes(b"synthetic checkpoint bytes")
        unit = replace(self.unit, source_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        with patch("reflexml.mechanism_lr_switch._configuration", return_value=self.config):
            prepared = prepare_reset(unit, self.dataset, treatment=treatment)
        proof = deepcopy(prepared.pretraining_evidence)
        proof["s17_artifact_path"] = str(artifact)
        proof["source_checkpoint_path"] = str(checkpoint)
        prepared = replace(prepared, unit=replace(unit, _seal=_FROZEN_UNIT_SEAL),
                           pretraining_evidence=proof)
        return prepared, artifact, checkpoint

    def official_run(self, root: Path, prepared, artifact: Path, checkpoint: Path,
                     attempt_id: str, outcome=None, retry_of=None, write_failure=False):
        def mocked_kernel(*args, **kwargs):
            if isinstance(outcome, Exception):
                raise outcome
            return {"learning_rate": prepared.pretraining_evidence["assigned_lr"],
                    "realized_order": prepared.unit.historical["wait_realized_orders"][3],
                    "epoch_order_sha256": prepared.pretraining_evidence["expected_epoch18_order_sha256"],
                    "train_loss": 0.5, "val_loss": outcome, "val_accuracy": 0.75,
                    "state_evidence": {"model_parameters": [], "optimizer_state": []}}
        real_writer = momentum_reset.write_attempt
        def writer(*args, **kwargs):
            if write_failure:
                raise OSError("terminal write failed after result")
            return real_writer(*args, **kwargs)
        with patch.object(momentum_reset, "_require_execution_authorization"), patch.object(
            momentum_reset, "load_canonical_roster", return_value={prepared.unit.key: prepared.unit}
        ), patch.object(momentum_reset, "verify_kernel_candidate"), patch.object(
            momentum_reset, "validate_production_dataset"), patch.object(
            momentum_reset, "verify_frozen_protocol"
        ), patch.object(momentum_reset, "_canonical_source_paths",
                        return_value=(artifact, checkpoint)), patch.object(
            momentum_reset, "prepare_reset", return_value=prepared
        ), patch.object(momentum_reset, "_configuration", return_value=self.config), patch.object(
            mechanism_lr_switch, "_configuration", return_value=self.config
        ), patch.object(momentum_reset, "kernel_provenance",
                        return_value=prepared.pretraining_evidence["kernel_provenance"]), patch.object(
            mechanism_lr_switch, "run_one_epoch", side_effect=mocked_kernel
        ), patch.object(momentum_reset, "write_attempt", side_effect=writer):
            return run_production_reset(1, 1, self.dataset, Path(__file__).resolve().parents[1],
                                        treatment=prepared.treatment, attempt_directory=root,
                                        attempt_id=attempt_id, retry_of=retry_of)

    def test_reset_state_preserves_structure_and_source(self):
        original_model = deepcopy(self.unit.checkpoint["model_state_dict"])
        original_optimizer = deepcopy(self.unit.checkpoint["optimizer_state_dict"])
        model, optimizer = reset_momentum_buffers(original_model, original_optimizer)
        self.assertTrue(states_equal(model, original_model))
        self.assertTrue(states_equal(original_optimizer, self.unit.checkpoint["optimizer_state_dict"]))
        self.assertEqual(optimizer["param_groups"], original_optimizer["param_groups"])
        self.assertEqual(set(optimizer["state"]), set(original_optimizer["state"]))
        for identifier, state in optimizer["state"].items():
            original = original_optimizer["state"][identifier]["momentum_buffer"]
            buffer = state["momentum_buffer"]
            self.assertEqual(set(state), {"momentum_buffer"})
            self.assertEqual((buffer.shape, buffer.dtype, buffer.device),
                             (original.shape, original.dtype, original.device))
            self.assertTrue(torch.all(buffer == 0).item())

    def test_rng_pairing_and_pretraining_provenance(self):
        cpu_rng = torch.random.get_rng_state().clone()
        low = self.prepare("L_05R")
        high = self.prepare("L_10R")
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        self.assertTrue(states_equal(low.model_state, high.model_state))
        self.assertEqual(low.optimizer_state["state"].keys(), high.optimizer_state["state"].keys())
        self.assertTrue(states_equal(low.optimizer_state["state"], high.optimizer_state["state"]))
        self.assertTrue(torch.equal(low.generator_state, high.generator_state))
        self.assertEqual(low.pretraining_evidence["assigned_lr"], 0.05)
        self.assertEqual(high.pretraining_evidence["assigned_lr"], 0.10)
        for prepared in (low, high):
            proof = prepared.pretraining_evidence
            self.assertEqual(len(proof["original_momentum_buffers"]), 4)
            self.assertEqual(len(proof["reset_momentum_buffers"]), 4)
            self.assertTrue(all(row["all_zero_before_training"]
                                for row in proof["reset_momentum_buffers"].values()))
            self.assertTrue(proof["model_unchanged_by_reset"])
            self.assertTrue(proof["canonical_generator_unchanged_by_reset"])
            self.assertTrue(proof["cpu_rng_unchanged_by_reset"])
        def inspect_before_step(*_args, **_kwargs):
            self.assertEqual(len(low.pretraining_evidence["reset_momentum_buffers"]), 4)
            raise RuntimeError("pretraining proof inspected before the numerical kernel")
        with patch("reflexml.momentum_reset._configuration", return_value=self.config), patch(
            "reflexml.mechanism_lr_switch._configuration", return_value=self.config
        ), patch("reflexml.mechanism_lr_switch.run_one_epoch", side_effect=inspect_before_step
        ), patch("reflexml.momentum_reset.kernel_provenance",
                 return_value=low.pretraining_evidence["kernel_provenance"]
        ):
            with self.assertRaisesRegex(RuntimeError, "before the numerical kernel"):
                run_synthetic_reset(low, self.dataset)
        with patch("reflexml.momentum_reset._configuration", return_value=self.config), patch(
            "reflexml.mechanism_lr_switch._configuration", return_value=self.config
        ):
            low_result = run_synthetic_reset(low, self.dataset)
            high_result = run_synthetic_reset(high, self.dataset)
        self.assertEqual(low_result["realized_order"], high_result["realized_order"])
        self.assertEqual(low_result["epoch_order_sha256"], high_result["epoch_order_sha256"])
        self.assertEqual(low_result["realized_order"], self.unit.historical["wait_realized_orders"][3])
        self.assertEqual(low_result["training_step_count"], 2)
        self.assertEqual(len(low_result["final_model_parameters"]), 4)
        self.assertEqual(len(low_result["final_momentum_buffers"]), 4)

    def test_attempt_classification_and_retry_identity(self):
        prepared = self.prepare("L_05R")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nonfinite = write_attempt(root, attempt_id="valid-1", prepared=prepared,
                                      result={"val_loss": float("nan"), "train_loss": float("inf")})
            record = json.loads(nonfinite.read_text())
            self.assertEqual(record["classification"], "scientifically_valid_primary_non_evaluable")
            self.assertTrue(record["scientific_output_generated"])
            self.assertEqual(record["result"]["val_loss"], {"__nonfinite_float__": "NaN"})
            with self.assertRaises(PermissionError):
                write_attempt(root, attempt_id="retry-valid", prepared=prepared,
                              result=None, technical_failure=self.observed_io_failure(root),
                              retry_of="valid-1")
            other = self.prepare("L_10R")
            write_attempt(root, attempt_id="invalid-1", prepared=other, result=None,
                          technical_failure=self.observed_io_failure(root))
            retry = write_attempt(root, attempt_id="retry-invalid", prepared=other,
                                  result={"val_loss": 999999.0}, retry_of="invalid-1")
            self.assertEqual(json.loads(retry.read_text())["classification"],
                             "scientifically_valid_primary_evaluable")
            with self.assertRaises(PermissionError):
                write_attempt(root, attempt_id="valid-1", prepared=prepared, result={})
            with self.assertRaises(ValueError):
                build_completion_manifest(root, root / "manifest.json")

    def test_production_block(self):
        self.assertFalse(momentum_reset.EXECUTION_AUTHORITY.exists())
        with patch.object(mechanism_lr_switch, "run_one_epoch") as kernel:
            with self.assertRaises(PermissionError):
                run_production_reset(1, 1, self.dataset, Path("."), treatment="L_05R",
                                     attempt_directory=Path("/unused"), attempt_id="blocked")
            kernel.assert_not_called()
        sealed = replace(self.unit, _seal=_FROZEN_UNIT_SEAL)
        with self.assertRaises(PermissionError):
            run_synthetic_reset(replace(self.prepare("L_05R"), unit=sealed), self.dataset)

    def test_direct_helper_authorization_bypass_is_closed(self):
        sealed = replace(self.prepare("L_05R"),
                         unit=replace(self.unit, _seal=_FROZEN_UNIT_SEAL))
        disguised = replace(self.prepare("L_05R"), unit=replace(
            self.unit, checkpoint={**self.unit.checkpoint,
                                   "config": ExperimentConfig().to_dict()}))
        with patch("reflexml.mechanism_lr_switch.run_one_epoch") as kernel:
            with self.assertRaises(PermissionError):
                _run_prepared(sealed, self.dataset)
            with self.assertRaises(PermissionError):
                _run_prepared(disguised, self.dataset)
            kernel.assert_not_called()

    @unittest.skip("Requires excluded internal provenance/authorization artifacts; public copies are not historical identities")
    def test_external_authority_schema_and_fail_closed_inputs(self):
        candidate = json.loads(momentum_reset.KERNEL_CANDIDATE.read_text())
        candidate_sha = hashlib.sha256(momentum_reset.KERNEL_CANDIDATE.read_bytes()).hexdigest()
        reattestation = {
            "schema": "ReflexML-Momentum-Reset-Compatibility-Reattestation-v2",
            "verdict": "PASS", "protocol_id": momentum_reset.PROTOCOL_ID,
            "kernel_candidate_v2_sha256": candidate_sha,
            "previous_compatibility_report_sha256": momentum_reset.PREVIOUS_COMPATIBILITY_SHA256,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-reattestation.json"
            path.write_text(json.dumps(reattestation))
            reattestation_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            record = {
                "schema": "ReflexML-Momentum-Reset-Execution-Authorization-v1",
                "execution_authorized": True,
                "protocol_id": momentum_reset.PROTOCOL_ID,
                "protocol_sha256": momentum_reset.PROTOCOL_SHA256,
                "freeze_record_sha256": momentum_reset.FREEZE_RECORD_SHA256,
                "implementation_sha256": hashlib.sha256(momentum_reset.IMPLEMENTATION_SOURCE.read_bytes()).hexdigest(),
                "kernel_candidate_sha256": candidate_sha,
                "roster_sha256": momentum_reset.ROSTER_SHA256,
                "compatibility_reattestation_sha256": reattestation_sha,
                "previous_compatibility_report_sha256": momentum_reset.PREVIOUS_COMPATIBILITY_SHA256,
                "authorized_branch_count": 144,
                "authorized_treatments": [
                    {"treatment": "L_10R", "learning_rate": 0.10},
                    {"treatment": "L_05R", "learning_rate": 0.05}],
                "replicas": ["A1", "A2"], "keep_reruns_authorized": False,
                "source_substitution_authorized": False,
                "kernel_components": candidate["components"],
                "environment": candidate["environment"],
            }
            model_before = deepcopy(self.unit.checkpoint["model_state_dict"])
            optimizer_before = deepcopy(self.unit.checkpoint["optimizer_state_dict"])
            generator_before = torch.random.get_rng_state().clone()
            future_before = self.unit.future_seed
            momentum_reset._validate_authorization_payload(
                record, reattestation, candidate, momentum_reset.ROSTER_SHA256,
                reattestation_sha)
            self.assertTrue(states_equal(model_before, self.unit.checkpoint["model_state_dict"]))
            self.assertTrue(states_equal(optimizer_before, self.unit.checkpoint["optimizer_state_dict"]))
            self.assertTrue(torch.equal(generator_before, torch.random.get_rng_state()))
            self.assertEqual(future_before, self.unit.future_seed)
            for field, wrong in (("execution_authorized", False),
                                 ("protocol_sha256", "0" * 64),
                                 ("implementation_sha256", "0" * 64),
                                 ("kernel_candidate_sha256", "0" * 64),
                                 ("roster_sha256", "0" * 64),
                                 ("compatibility_reattestation_sha256", "0" * 64),
                                 ("previous_compatibility_report_sha256", "0" * 64)):
                with self.subTest(field=field):
                    bad = {**record, field: wrong}
                    with self.assertRaises(PermissionError):
                        momentum_reset._validate_authorization_payload(
                            bad, reattestation, candidate,
                            momentum_reset.ROSTER_SHA256, reattestation_sha)
            missing = dict(record)
            del missing["replicas"]
            with self.assertRaises(PermissionError):
                momentum_reset._validate_authorization_payload(
                    missing, reattestation, candidate,
                    momentum_reset.ROSTER_SHA256, reattestation_sha)
            with self.assertRaises(PermissionError):
                momentum_reset._validate_authorization_payload(
                    record, {**reattestation, "verdict": "FAIL"}, candidate,
                    momentum_reset.ROSTER_SHA256, reattestation_sha)
            with self.assertRaises(PermissionError):
                momentum_reset._validate_authorization_payload(
                    record, reattestation, candidate, "0" * 64, reattestation_sha)
            malformed = Path(directory) / "malformed.json"
            malformed.write_text("{")
            with self.assertRaises(PermissionError):
                momentum_reset._read_authority(malformed)
            with self.assertRaises(PermissionError):
                momentum_reset._read_authority(Path(directory) / "missing.json")

    @unittest.skip("Requires excluded internal provenance/authorization artifacts; public copies are not historical identities")
    def test_numerical_components_match_previous_candidate(self):
        previous = json.loads((momentum_reset.REPO_ROOT / "MOMENTUM_RESET_KERNEL_CANDIDATE.json").read_text())
        current = json.loads(momentum_reset.KERNEL_CANDIDATE.read_text())
        self.assertEqual(current["components"], momentum_reset.kernel_provenance())
        for name in ("lr_switch", "run_one_epoch", "train_one_epoch", "evaluate",
                     "model", "data_loader", "evaluate_decorator"):
            self.assertEqual(current["components"][name], previous["components"][name])

    def test_mutable_prepared_state_is_rechecked_before_kernel(self):
        mutations = {
            "buffer": lambda p: p.optimizer_state["state"][0]["momentum_buffer"].fill_(1),
            "model": lambda p: next(iter(p.model_state.values())).add_(1),
            "lr": lambda p: p.optimizer_state["param_groups"][0].update(lr=0.2),
            "generator": lambda p: p.generator_state.fill_(0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                prepared = self.prepare("L_05R")
                mutate(prepared)
                with patch("reflexml.mechanism_lr_switch._configuration",
                           return_value=self.config), patch(
                    "reflexml.mechanism_lr_switch.run_one_epoch"
                ) as kernel:
                    with self.assertRaises(ValueError):
                        _run_prepared(prepared, self.dataset)
                    kernel.assert_not_called()

    def test_scientific_result_cannot_be_relabelled_technical(self):
        prepared = self.prepare("L_05R")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            failure = self.observed_io_failure(root)
            for index, value in enumerate((float("nan"), float("inf"),
                                           -float("inf"), 1e300)):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        write_attempt(root, attempt_id=f"invalid-{index}", prepared=prepared,
                                      result={"val_loss": value}, technical_failure=failure)
            path = write_attempt(root, attempt_id="actual-integrity-failure", prepared=prepared,
                                 result=None, technical_failure=failure)
            self.assertEqual(json.loads(path.read_text())["classification"], "technical_invalid")

    def test_retry_lineage_rejects_substitutions_and_unlinked_attempts(self):
        prepared = self.prepare("L_05R")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic-source"
            source.write_bytes(b"observed synthetic source")
            failure = observe_source_hash_mismatch(source, "0" * 64)
            with self.assertRaises(ValueError):
                write_attempt(root, attempt_id="boolean", prepared=prepared, result=None,
                              technical_failure=True)
            with self.assertRaises(ValueError):
                observe_source_hash_mismatch(source, hashlib.sha256(source.read_bytes()).hexdigest())
            write_attempt(root, attempt_id="first", prepared=prepared, result=None,
                          technical_failure=failure)
            with self.assertRaises(PermissionError):
                write_attempt(root, attempt_id="unlinked", prepared=prepared,
                              result={"val_loss": 1.0})
            changed = replace(prepared, unit=replace(prepared.unit, source_sha256="changed"))
            with self.assertRaises(PermissionError):
                write_attempt(root, attempt_id="changed", prepared=changed,
                              result={"val_loss": 1.0}, retry_of="first")
            changed_seed = replace(prepared, unit=replace(prepared.unit, future_seed=82))
            with self.assertRaises(PermissionError):
                write_attempt(root, attempt_id="changed-seed", prepared=changed_seed,
                              result={"val_loss": 1.0}, retry_of="first")
            changed_treatment = replace(prepared, treatment="L_10R")
            with self.assertRaises(PermissionError):
                write_attempt(root, attempt_id="changed-treatment", prepared=changed_treatment,
                              result={"val_loss": 1.0}, retry_of="first")
            write_attempt(root, attempt_id="linked", prepared=prepared,
                          result={"val_loss": float("nan")}, retry_of="first")
            with self.assertRaises(PermissionError):
                write_attempt(root, attempt_id="post-valid", prepared=prepared,
                              result={"val_loss": 1.0}, retry_of="first")

    def test_nonfinite_serialization_is_canonical(self):
        prepared = self.prepare("L_05R")
        for value in (1.25, float("nan"), float("inf"), -float("inf")):
            encoded = json.dumps(_json_safe({"value": value}), allow_nan=False)
            restored = _json_restore(json.loads(encoded))["value"]
            self.assertTrue(math.isnan(restored) if math.isnan(value) else restored == value)
            with TemporaryDirectory() as directory:
                path = write_attempt(Path(directory), attempt_id="roundtrip", prepared=prepared,
                                     result={"val_loss": value})
                restored = read_scientific_result(path)["val_loss"]
                self.assertTrue(math.isnan(restored) if math.isnan(value) else restored == value)

    @unittest.skip("Requires excluded internal provenance/authorization artifacts; public copies are not historical identities")
    def test_kernel_provenance_uses_imported_function_files(self):
        import inspect
        from reflexml import mechanism_lr_switch, training
        from reflexml.data import make_loaders_from_datasets
        from reflexml.model import FashionMLP
        from reflexml.phase5 import sha256_file
        provenance = kernel_provenance()
        for name, function in (("run_one_epoch", mechanism_lr_switch.run_one_epoch),
                               ("train_one_epoch", training.train_one_epoch),
                               ("evaluate", training.evaluate),
                               ("model", FashionMLP),
                               ("data_loader", make_loaders_from_datasets)):
            source = Path(inspect.getfile(inspect.unwrap(function)))
            self.assertEqual(provenance[name]["path"], str(source.resolve()))
            self.assertEqual(provenance[name]["sha256"], sha256_file(source))
        self.assertIs(momentum_reset.mechanism_lr_switch.run_one_epoch,
                      mechanism_lr_switch.run_one_epoch)
        self.assertIs(mechanism_lr_switch.train_one_epoch, training.train_one_epoch)
        self.assertIs(mechanism_lr_switch.evaluate, training.evaluate)
        with patch.object(mechanism_lr_switch, "train_one_epoch", lambda: None):
            with self.assertRaisesRegex(ValueError, "binding drift"):
                kernel_provenance()
        with patch.object(mechanism_lr_switch, "run_one_epoch", lambda: None):
            self.assertNotEqual(kernel_provenance()["run_one_epoch"], provenance["run_one_epoch"])
            with self.assertRaisesRegex(ValueError, "candidate identity mismatch"):
                momentum_reset.verify_kernel_candidate()
        momentum_reset.verify_kernel_candidate()

    def test_official_lifecycle_records_scientific_outcomes_and_blocks_reclassification(self):
        for index, value in enumerate((0.25, float("nan"), float("inf"), -float("inf"))):
            with self.subTest(value=value), TemporaryDirectory() as directory:
                root = Path(directory)
                prepared, artifact, checkpoint = self.official_fixture(root)
                path = self.official_run(root, prepared, artifact, checkpoint,
                                         f"official-{index}", outcome=value)
                self.assertIsInstance(path, Path)
                record = json.loads(path.read_text())
                expected = ("scientifically_valid_primary_evaluable" if math.isfinite(value)
                            else "scientifically_valid_primary_non_evaluable")
                self.assertEqual(record["classification"], expected)
                observed = read_scientific_result(path)["val_loss"]
                self.assertTrue(math.isnan(observed) if math.isnan(value) else observed == value)
                with self.assertRaises(PermissionError):
                    write_attempt(root, attempt_id=f"official-{index}", prepared=prepared,
                                  result=None, technical_failure=self.observed_io_failure(root))
                with self.assertRaises(PermissionError):
                    self.official_run(root, prepared, artifact, checkpoint,
                                      f"retry-{index}", outcome=0.1)

    def test_official_post_result_write_failure_remains_unresolved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, artifact, checkpoint = self.official_fixture(root)
            with self.assertRaisesRegex(OSError, "terminal write failed"):
                self.official_run(root, prepared, artifact, checkpoint, "write-failed",
                                  outcome=0.4, write_failure=True)
            self.assertTrue((root / "write-failed.prestep").is_file())
            self.assertFalse((root / "write-failed.json").exists())
            with self.assertRaisesRegex(PermissionError, "unresolved"):
                self.official_run(root, prepared, artifact, checkpoint, "retry", outcome=0.4)

    def test_terminal_file_write_failure_keeps_reconciliation_marker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, artifact, checkpoint = self.official_fixture(root)
            original_open = Path.open
            def fail_terminal_open(path, mode="r", *args, **kwargs):
                if path.name == "file-failed.json" and mode == "x":
                    raise OSError("simulated terminal file failure")
                return original_open(path, mode, *args, **kwargs)
            with patch.object(Path, "open", fail_terminal_open):
                with self.assertRaisesRegex(OSError, "terminal file failure"):
                    self.official_run(root, prepared, artifact, checkpoint,
                                      "file-failed", outcome=0.4)
            self.assertTrue((root / "file-failed.terminal_pending").is_file())
            with self.assertRaisesRegex(PermissionError, "Unreconciled terminal"):
                self.official_run(root, prepared, artifact, checkpoint,
                                  "retry", outcome=0.4)
            with patch.object(momentum_reset, "load_canonical_roster", return_value={
                prepared.unit.key: prepared.unit}):
                with self.assertRaisesRegex(ValueError, "reconciliation"):
                    build_completion_manifest(root, root / "manifest.json")

    def test_official_pre_result_io_failure_is_technical_invalid(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, artifact, checkpoint = self.official_fixture(root)
            def fail_before_result():
                root.write_bytes(b"cannot write bytes over a directory")
            with self.assertRaises(OSError) as caught:
                fail_before_result()
            path = self.official_run(root, prepared, artifact, checkpoint, "io-failed",
                                     outcome=caught.exception)
            record = json.loads(path.read_text())
            self.assertEqual(record["classification"], "technical_invalid")
            self.assertFalse(record["scientific_output_generated"])

    @unittest.skip("Requires excluded internal provenance/authorization artifacts; public copies are not historical identities")
    def test_authoritative_manifest_excludes_synthetic_and_wrong_roster_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, artifact, checkpoint = self.official_fixture(root)
            path = self.official_run(root, prepared, artifact, checkpoint,
                                     "admitted", outcome=0.25)
            roster = {(state, replica): replace(prepared.unit, base_run_id=state,
                                                replica_id=replica)
                      for state in range(1, 37) for replica in (1, 2)}
            with patch.object(momentum_reset, "load_canonical_roster", return_value=roster), patch.object(
                momentum_reset, "_canonical_source_paths", return_value=(artifact, checkpoint)):
                manifest = build_completion_manifest(root, root / "manifest.json")
            self.assertEqual(manifest["scientifically_valid_count"], 1)
            self.assertEqual(manifest["expected_branch_count"], 144)
            (root / "manifest.json").unlink()
            original = json.loads(path.read_text())
            for field, value in (("source_sha256", "toy-artifact"),
                                 ("checkpoint_sha256", "alternate-checkpoint"),
                                 ("future_seed", 123), ("base_run_id", 37),
                                 ("replica_id", 3), ("treatment", "L_05R_alias")):
                with self.subTest(field=field):
                    row = deepcopy(original)
                    row["scientific_identity"][field] = value
                    digest = row.pop("record_sha256")
                    row["record_sha256"] = hashlib.sha256(json.dumps(
                        row, sort_keys=True, ensure_ascii=False, allow_nan=False,
                        separators=(",", ":")).encode()).hexdigest()
                    path.write_text(json.dumps(row, sort_keys=True) + "\n")
                    with patch.object(momentum_reset, "load_canonical_roster", return_value=roster), patch.object(
                        momentum_reset, "_canonical_source_paths", return_value=(artifact, checkpoint)):
                        with self.assertRaises(ValueError):
                            build_completion_manifest(root, root / "rejected.json")
            path.write_text(json.dumps(original, sort_keys=True) + "\n")

    @unittest.skip("Requires excluded internal provenance/authorization artifacts; public copies are not historical identities")
    def test_frozen_roster_has_72_units_and_144_branches(self):
        repo = Path(__file__).resolve().parents[1]
        roster = load_canonical_roster(repo)
        self.assertEqual(len(roster), 72)
        self.assertEqual(set(roster), {(state, replica) for state in range(1, 37)
                                       for replica in (1, 2)})
        self.assertEqual(len({(state, replica, treatment) for state, replica in roster
                              for treatment in ("L_10R", "L_05R")}), 144)
        self.assertNotIn((37, 1), roster)
        self.assertEqual(canonical_roster_sha256(roster),
                         "2e0fa1380561d111183e69609dcfe1b419a986487533344cc01ea95da7fb439e")
        checked = 0
        for unit in roster.values():
            artifact, checkpoint = momentum_reset._canonical_source_paths(unit)
            for treatment in ("L_10R", "L_05R"):
                proof = {
                    "protocol_id": momentum_reset.PROTOCOL_ID,
                    "protocol_sha256": momentum_reset.PROTOCOL_SHA256,
                    "implementation_sha256": momentum_reset.sha256_file(
                        momentum_reset.IMPLEMENTATION_SOURCE),
                    "kernel_provenance": kernel_provenance(),
                    "environment": json.loads(momentum_reset.KERNEL_CANDIDATE.read_text())[
                        "environment"],
                    "base_run_id": unit.base_run_id, "replica_id": unit.replica_id,
                    "treatment": treatment, "s17_source_sha256": unit.source_sha256,
                    "checkpoint_sha256": unit.checkpoint_sha256,
                    "future_seed": unit.future_seed,
                    "s17_artifact_path": str(artifact),
                    "source_checkpoint_path": str(checkpoint),
                    "model_parameters": {}, "reset_momentum_buffers": {},
                    "assigned_lr": 0.10 if treatment == "L_10R" else 0.05,
                    "canonical_post_s17_generator_sha256":
                        momentum_reset._canonical_generator_hash(
                            unit.future_seed, ExperimentConfig(
                                **unit.checkpoint["config"]).train_size),
                    "expected_epoch18_order_sha256":
                        unit.historical["wait_epoch_order_sha256"][3],
                }
                prepared = momentum_reset.PreparedReset(unit, treatment, {}, {},
                                                         torch.empty(0), proof)
                with patch.object(momentum_reset, "_canonical_source_paths",
                                  return_value=(artifact, checkpoint)):
                    momentum_reset._validate_manifest_identity(
                        branch_identity(prepared), proof, roster, "canonical-test")
                    if unit.key == (1, 1) and treatment == "L_05R":
                        for field, bad in (("canonical_post_s17_generator_sha256", "0" * 64),
                                           ("s17_artifact_path", "toy-artifact")):
                            altered = {**proof, field: bad}
                            changed = replace(prepared, pretraining_evidence=altered)
                            with self.subTest(field=field), self.assertRaises(ValueError):
                                momentum_reset._validate_manifest_identity(
                                    branch_identity(changed), altered, roster, "substituted")
                checked += 1
        self.assertEqual(checked, 144)
        arbitrary = replace(roster[1, 1], base_run_id=37)
        with patch("reflexml.momentum_reset.load_frozen_units",
                   return_value=[*roster.values(), arbitrary]):
            with self.assertRaises(ValueError):
                load_canonical_roster(repo)

    def test_prestep_record_is_durable_before_training(self):
        prepared = self.prepare("L_05R")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_prestep(root, "before-step", prepared)
            record = json.loads(path.read_text())
            self.assertEqual(record["scientific_identity"], branch_identity(prepared))
            self.assertEqual(record["pretraining_evidence"], prepared.pretraining_evidence)
            with self.assertRaises(ValueError):
                build_completion_manifest(root, root / "manifest.json")
            with self.assertRaises(PermissionError):
                _write_prestep(root, "second-before-resolution", prepared)
            write_attempt(root, attempt_id="before-step", prepared=prepared,
                          result=None, technical_failure=self.observed_io_failure(root))
            retry = _write_prestep(root, "linked-retry", prepared, retry_of="before-step")
            self.assertTrue(retry.is_file())
            with self.assertRaises(PermissionError):
                _write_prestep(root, "unlinked-retry", prepared)

    def test_synthetic_factorial_state_bootstrap_and_nonfinite(self):
        cells = {}
        for state in range(1, 37):
            for replica in (1, 2):
                for treatment, offset in (("L_10K", 0.04), ("L_05K", 0.0),
                                          ("L_10R", 0.03), ("L_05R", 0.01)):
                    value = state * 0.001 + replica * 0.0001 + offset
                    cells[state, replica, treatment] = {
                        "val_loss": value, "train_loss": value * 2,
                        "val_accuracy": 1 - value,
                    }
        result = analyze_factorial(cells)
        self.assertEqual(result["status"], "EVALUABLE")
        loss = result["outcomes"]["epoch18_val_loss"]
        self.assertAlmostEqual(loss["Delta_K"]["estimate"], 0.04)
        self.assertAlmostEqual(loss["Delta_R"]["estimate"], 0.02)
        self.assertAlmostEqual(loss["I"]["estimate"], 0.02)
        self.assertAlmostEqual(loss["R_.10"]["estimate"] - loss["R_.05"]["estimate"],
                               loss["I"]["estimate"])
        cells[1, 1, "L_05R"]["val_loss"] = float("nan")
        self.assertEqual(analyze_factorial(cells)["status"], "NON-EVALUABLE AS SPECIFIED")
        cells[1, 1, "L_05R"]["val_loss"] = _json_safe(float("inf"))
        self.assertEqual(analyze_factorial(cells)["status"], "NON-EVALUABLE AS SPECIFIED")


if __name__ == "__main__":
    unittest.main()

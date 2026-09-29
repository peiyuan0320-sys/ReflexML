"""Frozen Momentum-Reset implementation candidate; production training is blocked.

The S17 reconstruction and epoch-18 numerical kernel are the LR-Switch ones.
This module adds the joint buffer intervention and evidence before that kernel.
"""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
import inspect
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import RandomSampler
import numpy as np

from .branching import order_hashes, states_equal
from . import mechanism_lr_switch
from . import training, data, model
from .config import ExperimentConfig
from .mechanism_lr_switch import (
    Unit, _FROZEN_UNIT_SEAL, _configuration, _iterate_order, _loader,
    load_frozen_units, reconstruct_epoch17, reconstruct_epoch18_generator,
    validate_production_dataset,
)
from .phase5 import sha256_file


PROTOCOL_ID = "ReflexML-Momentum-Reset-v1"
PROTOCOL_SHA256 = "5f6488f867c5a5663067ef5d5a85aea3aeca295e53260b42b9a21a61dac52f38"
FREEZE_RECORD_SHA256 = "8ce90fa4003c60640c622f966a67536412e273e99d6bf2fbd59ccb6ad57e3992"
ROSTER_SHA256 = "2e0fa1380561d111183e69609dcfe1b419a986487533344cc01ea95da7fb439e"
PREVIOUS_CANDIDATE_SHA256 = "675542ee535929fac9aa38a2a7f0d3438674123db74420f11791395b743913a9"
PREVIOUS_COMPATIBILITY_SHA256 = "bc141514087dc6b9693baf8d4a06a887e779f5dda0d4366d29858377693deb3e"
BOOTSTRAP_SEED = 6630267477959695398
HISTORICAL_BOOTSTRAP_SEED = 12114954806208443255
STAGE_I_GATE_SHA256 = "dd7d4273a215f1c71cdf816bce3334dc846c37e31b92f0e29e46b7748d67dc80"
STAGE_II_MANIFEST_SHA256 = "b4db6b50f176c556ec6ec31182831bbe2a308f174dca499bb44e1503c0e5d473"
PRIMARY_ANALYSIS_SHA256 = "9b7d8a940c3b62590cad1793070c523ddb0c06c8592d2abe4d7c84378454e791"
IMPLEMENTATION_SOURCE = Path(__file__)
REPO_ROOT = IMPLEMENTATION_SOURCE.resolve().parents[1]
KERNEL_CANDIDATE = REPO_ROOT / "MOMENTUM_RESET_KERNEL_CANDIDATE_V2.json"
EXECUTION_AUTHORITY = REPO_ROOT / "MOMENTUM_RESET_EXECUTION_AUTHORIZATION.json"
COMPATIBILITY_REATTESTATION = REPO_ROOT / "MOMENTUM_RESET_COMPATIBILITY_REATTESTATION_V2.json"
FROZEN_BRANCH_LEDGER = Path("internal-artifacts/branch_attempts.jsonl")
FROZEN_BASE_OUTPUT = Path("internal-artifacts/phase5_v1_base_v2")
RESET_TREATMENTS = ("L_10R", "L_05R")
_OBSERVED_FAILURE_SEAL = object()
_LIFECYCLE_SEAL = object()


def verify_frozen_protocol(repo: Path) -> None:
    if (sha256_file(repo / "MOMENTUM_RESET_PROTOCOL.md") != PROTOCOL_SHA256 or
            sha256_file(repo / "MOMENTUM_RESET_PROTOCOL_FREEZE_RECORD.md") != FREEZE_RECORD_SHA256):
        raise ValueError("Momentum-Reset frozen protocol identity mismatch")


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _tensor_row(value: torch.Tensor) -> dict[str, Any]:
    return {
        "sha256": _tensor_hash(value), "shape": list(value.shape),
        "dtype": str(value.dtype), "device": str(value.device),
    }


def _generator_hash(state: torch.Tensor) -> str:
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


@lru_cache(maxsize=144)
def _canonical_generator_hash(future_seed: int, train_size: int) -> str:
    """Recreate three loader sampler advances without fetching images or training."""
    generator = torch.Generator().manual_seed(future_seed)
    sampler = RandomSampler(range(train_size), generator=generator)
    for _ in range(3):
        # DataLoader consumes its base seed before each sampler iteration.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        deque(sampler, maxlen=0)
    return _generator_hash(generator.get_state())


@dataclass(frozen=True)
class PreparedReset:
    unit: Unit
    treatment: str
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    generator_state: torch.Tensor
    pretraining_evidence: dict[str, Any]


@dataclass(frozen=True)
class ObservedTechnicalFailure:
    stage: str
    condition: str
    evidence: dict[str, Any]
    _seal: object
    _error: OSError | None = None


def observe_source_hash_mismatch(path: Path, expected_sha256: str) -> ObservedTechnicalFailure:
    """Issue failure evidence only after hashing actual source bytes."""
    observed = sha256_file(path)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
            observed == expected_sha256):
        raise ValueError("No observed source hash mismatch")
    return ObservedTechnicalFailure(
        "source_identity", "source_hash_mismatch",
        {"source_path": str(path.resolve()), "expected_sha256": expected_sha256,
         "observed_sha256": observed}, _OBSERVED_FAILURE_SEAL)


def observe_artifact_io_failure(error: OSError) -> ObservedTechnicalFailure:
    """Capture the operating-system error from an actual failed artifact operation."""
    if (not isinstance(error, OSError) or error.__traceback__ is None or
            error.errno is None or not error.filename):
        raise ValueError("Artifact failure requires a captured file I/O exception")
    return ObservedTechnicalFailure(
        "artifact_integrity", "artifact_io_failure",
        {"exception_type": type(error).__name__, "errno": error.errno,
         "filename": error.filename, "message": str(error)},
        _OBSERVED_FAILURE_SEAL, error)


def kernel_provenance() -> dict[str, dict[str, str]]:
    """Hash the same module-qualified bindings used by the production kernel."""
    if (mechanism_lr_switch.train_one_epoch is not training.train_one_epoch or
            mechanism_lr_switch.evaluate is not training.evaluate or
            mechanism_lr_switch.FashionMLP is not model.FashionMLP or
            mechanism_lr_switch.make_loaders_from_datasets is not data.make_loaders_from_datasets):
        raise ValueError("LR-Switch numerical callable binding drift")
    functions = {"reset": prepare_reset, "lr_switch": reconstruct_epoch17,
                 "run_one_epoch": mechanism_lr_switch.run_one_epoch,
                 "train_one_epoch": mechanism_lr_switch.train_one_epoch,
                 "evaluate": mechanism_lr_switch.evaluate,
                 "model": mechanism_lr_switch.FashionMLP,
                 "data_loader": mechanism_lr_switch.make_loaders_from_datasets}
    provenance = {}
    for name, function in functions.items():
        path = Path(inspect.getfile(inspect.unwrap(function))).resolve()
        provenance[name] = {"path": str(path), "sha256": sha256_file(path)}
    wrapper = Path(inspect.getfile(mechanism_lr_switch.evaluate)).resolve()
    if wrapper != Path(provenance["evaluate"]["path"]):
        provenance["evaluate_decorator"] = {"path": str(wrapper),
                                            "sha256": sha256_file(wrapper)}
    return provenance


def verify_kernel_candidate() -> None:
    """Fail closed when ordinary production bindings differ from the audited candidate."""
    candidate = json.loads(KERNEL_CANDIDATE.read_text())
    expected = {"protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
                "components": kernel_provenance(),
                "environment": {"torch": torch.__version__,
                                "python": __import__("sys").version,
                                "device": "cpu", "num_threads": torch.get_num_threads()},
                "roster_sha256": ROSTER_SHA256,
                "authorization_path": str(EXECUTION_AUTHORITY),
                "compatibility_reattestation_path": str(COMPATIBILITY_REATTESTATION),
                "previous_candidate_sha256": PREVIOUS_CANDIDATE_SHA256,
                "previous_compatibility_report_sha256": PREVIOUS_COMPATIBILITY_SHA256,
                "implementation_change_classification": "execution authorization plumbing only",
                "status": "AWAITING_COMPATIBILITY_REATTESTATION"}
    if candidate != expected:
        raise ValueError("Momentum-Reset kernel candidate identity mismatch")


def _read_authority(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"Momentum-Reset authority unavailable: {path.name}") from error
    if not isinstance(record, dict):
        raise PermissionError("Momentum-Reset authority must be a JSON object")
    return record


def _validate_authorization_payload(
    record: dict[str, Any], reattestation: dict[str, Any],
    candidate: dict[str, Any], roster_sha256: str, reattestation_sha256: str,
) -> None:
    candidate_sha256 = sha256_file(KERNEL_CANDIDATE)
    expected_reattestation = {
        "schema": "ReflexML-Momentum-Reset-Compatibility-Reattestation-v2",
        "verdict": "PASS",
        "protocol_id": PROTOCOL_ID,
        "kernel_candidate_v2_sha256": candidate_sha256,
        "previous_compatibility_report_sha256": PREVIOUS_COMPATIBILITY_SHA256,
    }
    if reattestation != expected_reattestation:
        raise PermissionError("Momentum-Reset compatibility re-attestation identity mismatch")
    expected = {
        "schema": "ReflexML-Momentum-Reset-Execution-Authorization-v1",
        "execution_authorized": True,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "freeze_record_sha256": FREEZE_RECORD_SHA256,
        "implementation_sha256": sha256_file(IMPLEMENTATION_SOURCE),
        "kernel_candidate_sha256": candidate_sha256,
        "roster_sha256": ROSTER_SHA256,
        "compatibility_reattestation_sha256": reattestation_sha256,
        "previous_compatibility_report_sha256": PREVIOUS_COMPATIBILITY_SHA256,
        "authorized_branch_count": 144,
        "authorized_treatments": [
            {"treatment": "L_10R", "learning_rate": 0.10},
            {"treatment": "L_05R", "learning_rate": 0.05},
        ],
        "replicas": ["A1", "A2"],
        "keep_reruns_authorized": False,
        "source_substitution_authorized": False,
        "kernel_components": candidate["components"],
        "environment": candidate["environment"],
    }
    if record != expected or roster_sha256 != ROSTER_SHA256:
        raise PermissionError("Momentum-Reset execution authorization identity mismatch")


def _require_execution_authorization(repo: Path) -> None:
    """Admit only the fixed, hash-bound external authority before production effects."""
    record = _read_authority(EXECUTION_AUTHORITY)
    reattestation = _read_authority(COMPATIBILITY_REATTESTATION)
    verify_frozen_protocol(repo)
    if (sha256_file(REPO_ROOT / "MOMENTUM_RESET_KERNEL_CANDIDATE.json") != PREVIOUS_CANDIDATE_SHA256 or
            sha256_file(REPO_ROOT / "MOMENTUM_RESET_COMPATIBILITY_GATE.md") != PREVIOUS_COMPATIBILITY_SHA256):
        raise PermissionError("Momentum-Reset previous compatibility authority changed")
    verify_kernel_candidate()
    candidate = json.loads(KERNEL_CANDIDATE.read_text(encoding="utf-8"))
    roster_sha256 = canonical_roster_sha256(load_canonical_roster(repo))
    _validate_authorization_payload(record, reattestation, candidate, roster_sha256,
                                    sha256_file(COMPATIBILITY_REATTESTATION))


def load_canonical_roster(repo: Path) -> dict[tuple[int, int], Unit]:
    """Use the protocol-bound Phase 5 ledger and primary records, not a caller roster."""
    verify_frozen_protocol(repo)
    units = load_frozen_units(repo, FROZEN_BRANCH_LEDGER, FROZEN_BASE_OUTPUT)
    roster = {unit.key: unit for unit in units}
    expected = {(state, replica) for state in range(1, 37) for replica in (1, 2)}
    if len(units) != 72 or set(roster) != expected:
        raise ValueError("Frozen 72-unit source roster mismatch")
    return roster


def canonical_roster_sha256(roster: dict[tuple[int, int], Unit]) -> str:
    if set(roster) != {(state, replica) for state in range(1, 37)
                       for replica in (1, 2)}:
        raise ValueError("Canonical roster keys differ from frozen 72 units")
    rows = [{"base_run_id": state, "replica_id": replica,
             "source_sha256": unit.source_sha256,
             "checkpoint_sha256": unit.checkpoint_sha256,
             "future_seed": unit.future_seed}
            for (state, replica), unit in sorted(roster.items())]
    if len(rows) != 72:
        raise ValueError("Canonical roster must have exactly 72 units")
    return hashlib.sha256(json.dumps(rows, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _canonical_source_paths(unit: Unit) -> tuple[Path, Path]:
    primary = FROZEN_BASE_OUTPUT / f"base_{unit.base_run_id:03d}_attempt_001/primary_state.json"
    checkpoint = Path(json.loads(primary.read_text())["checkpoint_path"])
    matches = []
    with FROZEN_BRANCH_LEDGER.open() as handle:
        for line in handle:
            row = json.loads(line)
            identity = row.get("scientific_identity", {})
            if (row.get("event_type") == "artifact_produced" and
                    (identity.get("base_run_id"), identity.get("block"),
                     identity.get("replica_id")) ==
                    (unit.base_run_id, "A", unit.replica_id)):
                matches.append(row)
    if (len(matches) != 1 or matches[0]["artifact_sha256"] != unit.source_sha256 or
            sha256_file(checkpoint) != unit.checkpoint_sha256 or
            sha256_file(Path(matches[0]["artifact_path"])) != unit.source_sha256):
        raise ValueError("Canonical S17 source path or hash mismatch")
    return Path(matches[0]["artifact_path"]), checkpoint


def branch_identity(prepared: PreparedReset) -> dict[str, Any]:
    evidence = prepared.pretraining_evidence
    return {
        "protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "base_run_id": prepared.unit.base_run_id, "replica_id": prepared.unit.replica_id,
        "treatment": prepared.treatment, "source_sha256": prepared.unit.source_sha256,
        "checkpoint_sha256": prepared.unit.checkpoint_sha256,
        "model_parameters": evidence["model_parameters"],
        "reset_momentum_buffers": evidence["reset_momentum_buffers"],
        "assigned_lr": evidence["assigned_lr"], "future_seed": prepared.unit.future_seed,
        "canonical_generator_sha256": evidence["canonical_post_s17_generator_sha256"],
        "expected_epoch18_order_sha256": evidence["expected_epoch18_order_sha256"],
    }


def reset_momentum_buffers(
    model_state: dict[str, torch.Tensor], optimizer_state: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Clone S17 and zero only the four existing momentum-buffer values."""
    model = deepcopy(model_state)
    optimizer = deepcopy(optimizer_state)
    groups = optimizer["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != 4:
        raise ValueError("Expected one historical SGD group with four parameters")
    ids = groups[0]["params"]
    if len(set(ids)) != 4 or set(optimizer["state"]) != set(ids):
        raise ValueError("Historical optimizer parameter membership mismatch")
    group = groups[0]
    if any(group.get(key) != value for key, value in {
        "momentum": 0.9, "weight_decay": 0, "dampening": 0,
        "nesterov": False,
    }.items()):
        raise ValueError("Historical SGD settings mismatch")
    for identifier in ids:
        state = optimizer["state"][identifier]
        if set(state) != {"momentum_buffer"} or not isinstance(state["momentum_buffer"], torch.Tensor):
            raise ValueError("Historical momentum buffer missing or unexpected optimizer state")
        buffer = state["momentum_buffer"]
        state["momentum_buffer"] = torch.zeros_like(buffer)
    if not states_equal(model, model_state):
        raise AssertionError("Reset changed model parameters")
    return model, optimizer


def prepare_reset(unit: Unit, dataset: Any, *, treatment: str,
                  repo: Path | None = None, s17_artifact_path: Path | None = None,
                  source_checkpoint_path: Path | None = None) -> PreparedReset:
    """Reconstruct and attest the intervention before any scientific training."""
    if treatment not in {"L_10R", "L_05R"}:
        raise ValueError("Unknown Reset treatment")
    if unit._seal is _FROZEN_UNIT_SEAL:
        if repo is None or s17_artifact_path is None or source_checkpoint_path is None:
            raise PermissionError("Frozen units require protocol, S17 artifact, and checkpoint paths")
        verify_frozen_protocol(repo)
        if (sha256_file(s17_artifact_path) != unit.source_sha256 or
                sha256_file(source_checkpoint_path) != unit.checkpoint_sha256):
            raise ValueError("S17 artifact or source checkpoint bytes changed")
    original_model, original_optimizer = reconstruct_epoch17(unit)
    canonical = reconstruct_epoch18_generator(unit, dataset)
    original_model_copy = deepcopy(original_model)
    original_optimizer_copy = deepcopy(original_optimizer)
    canonical_copy = canonical.clone()
    cpu_rng = torch.random.get_rng_state().clone()
    ids = original_optimizer["param_groups"][0]["params"]
    names = list(original_model)
    if len(names) != 4 or len(ids) != 4:
        raise ValueError("Expected four S17 model parameters and buffers")
    original_rows = {
        name: {"optimizer_id": identifier,
               **_tensor_row(original_optimizer["state"][identifier]["momentum_buffer"])}
        for name, identifier in zip(names, ids, strict=True)
    }
    reset_model, reset_optimizer = reset_momentum_buffers(original_model, original_optimizer)
    reset_rows = {}
    for name, identifier in zip(names, ids, strict=True):
        buffer = reset_optimizer["state"][identifier]["momentum_buffer"]
        before = original_rows[name]
        row = {"optimizer_id": identifier, **_tensor_row(buffer),
               "all_zero_before_training": bool(torch.all(buffer == 0).item())}
        if (row["shape"], row["dtype"], row["device"]) != (before["shape"], before["dtype"], before["device"]):
            raise AssertionError("Reset changed buffer structure")
        if not row["all_zero_before_training"]:
            raise AssertionError("Reset buffer is not all zero")
        reset_rows[name] = row
    lr = 0.10 if treatment == "L_10R" else 0.05
    reset_optimizer["param_groups"][0]["lr"] = lr
    if (not states_equal(original_model, original_model_copy) or
            not states_equal(original_optimizer, original_optimizer_copy) or
            not torch.equal(canonical, canonical_copy) or
            not torch.equal(torch.random.get_rng_state(), cpu_rng)):
        raise AssertionError("Reset mutated historical source or relevant RNG")
    evidence = {
        "protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "implementation_sha256": sha256_file(IMPLEMENTATION_SOURCE),
        "kernel_provenance": kernel_provenance(),
        "environment": {"torch": torch.__version__, "python": __import__("sys").version,
                        "device": "cpu", "num_threads": torch.get_num_threads()},
        "base_run_id": unit.base_run_id, "replica_id": unit.replica_id,
        "treatment": treatment, "s17_source_sha256": unit.source_sha256,
        "s17_artifact_path": str(s17_artifact_path) if s17_artifact_path else None,
        "source_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "checkpoint_sha256": unit.checkpoint_sha256, "future_seed": unit.future_seed,
        "model_parameters": {name: _tensor_row(value) for name, value in original_model.items()},
        "parameter_to_optimizer_id": dict(zip(names, ids, strict=True)),
        "original_momentum_buffers": original_rows,
        "reset_momentum_buffers": reset_rows,
        "model_unchanged_by_reset": states_equal(original_model, reset_model),
        "optimizer_groups_before_reset": deepcopy(original_optimizer["param_groups"]),
        "optimizer_groups_for_epoch18": deepcopy(reset_optimizer["param_groups"]),
        "assigned_lr": lr,
        "canonical_post_s17_generator_sha256": _generator_hash(canonical),
        "cpu_rng_sha256": _generator_hash(cpu_rng),
        "canonical_generator_unchanged_by_reset": torch.equal(canonical, canonical_copy),
        "cpu_rng_unchanged_by_reset": torch.equal(torch.random.get_rng_state(), cpu_rng),
        "expected_epoch18_order_sha256": unit.historical["wait_epoch_order_sha256"][3],
    }
    return PreparedReset(unit, treatment, reset_model, reset_optimizer, canonical, evidence)


def _verify_prepared(prepared: PreparedReset, dataset: Any) -> None:
    """Recheck the exact mutable objects handed to the numerical kernel."""
    unit, proof = prepared.unit, prepared.pretraining_evidence
    cpu_rng_before = torch.random.get_rng_state().clone()
    if prepared.treatment not in RESET_TREATMENTS or proof.get("treatment") != prepared.treatment:
        raise ValueError("Reset branch treatment changed")
    original_model, original_optimizer = reconstruct_epoch17(unit)
    canonical = reconstruct_epoch18_generator(unit, dataset)
    names = list(original_model)
    groups = prepared.optimizer_state["param_groups"]
    ids = original_optimizer["param_groups"][0]["params"]
    if (len(names) != 4 or len(ids) != 4 or len(groups) != 1 or
            set(prepared.model_state) != set(original_model) or
            groups != proof.get("optimizer_groups_for_epoch18") or
            proof.get("optimizer_groups_before_reset") != original_optimizer["param_groups"] or
            groups[0]["params"] != ids or
            set(prepared.optimizer_state["state"]) != set(ids) or
            groups[0]["lr"] != (0.10 if prepared.treatment == "L_10R" else 0.05)):
        raise ValueError("Prepared optimizer/model structure or assigned LR changed")
    model_rows = {name: _tensor_row(value) for name, value in original_model.items()}
    if (proof.get("model_parameters") != model_rows or
            proof.get("parameter_to_optimizer_id") != dict(zip(names, ids, strict=True)) or
            {name: _tensor_row(prepared.model_state[name]) for name in names} != model_rows):
        raise ValueError("Prepared model no longer matches reconstructed S17")
    original_rows, reset_rows = {}, {}
    for name, identifier in zip(names, ids, strict=True):
        original = original_optimizer["state"][identifier]["momentum_buffer"]
        state = prepared.optimizer_state["state"][identifier]
        if set(state) != {"momentum_buffer"}:
            raise ValueError("Prepared momentum entry missing or changed")
        buffer = state["momentum_buffer"]
        original_rows[name] = {"optimizer_id": identifier, **_tensor_row(original)}
        reset_rows[name] = {"optimizer_id": identifier, **_tensor_row(buffer),
                            "all_zero_before_training": bool(torch.all(buffer == 0).item())}
        if (not reset_rows[name]["all_zero_before_training"] or
                any(reset_rows[name][key] != original_rows[name][key]
                    for key in ("shape", "dtype", "device"))):
            raise ValueError("Prepared reset momentum buffer changed")
    if (proof.get("original_momentum_buffers") != original_rows or
            proof.get("reset_momentum_buffers") != reset_rows or
            proof.get("assigned_lr") != groups[0]["lr"] or
            proof.get("canonical_post_s17_generator_sha256") != _generator_hash(canonical) or
            _generator_hash(prepared.generator_state) != _generator_hash(canonical) or
            not proof.get("cpu_rng_unchanged_by_reset") or
            not torch.equal(cpu_rng_before, torch.random.get_rng_state()) or
            proof.get("implementation_sha256") != sha256_file(IMPLEMENTATION_SOURCE) or
            proof.get("kernel_provenance") != kernel_provenance() or
            proof.get("base_run_id") != unit.base_run_id or
            proof.get("replica_id") != unit.replica_id or
            proof.get("future_seed") != unit.future_seed or
            proof.get("s17_source_sha256") != unit.source_sha256 or
            proof.get("checkpoint_sha256") != unit.checkpoint_sha256 or
            proof.get("protocol_id") != PROTOCOL_ID or
            proof.get("protocol_sha256") != PROTOCOL_SHA256 or
            proof.get("model_unchanged_by_reset") is not True or
            proof.get("canonical_generator_unchanged_by_reset") is not True or
            proof.get("environment") != {
                "torch": torch.__version__, "python": __import__("sys").version,
                "device": "cpu", "num_threads": torch.get_num_threads()}):
        raise ValueError("Prepared source, RNG, code, or branch evidence changed")


def run_production_reset(base_run_id: int, replica_id: int, dataset: Any, repo: Path, *,
                         treatment: str, attempt_directory: Path,
                         attempt_id: str, retry_of: str | None = None) -> Path:
    """Only authoritative lifecycle: roster, start, pre-step, kernel, terminal.

    Return the immutable terminal path, never an unrecorded scientific result.
    Currently unreachable under frozen execution authorization.
    """
    _require_execution_authorization(repo)
    verify_kernel_candidate()
    unit = load_canonical_roster(repo)[base_run_id, replica_id]
    validate_production_dataset(repo, dataset)
    artifact_path, checkpoint_path = _canonical_source_paths(unit)
    prepared = prepare_reset(unit, dataset, treatment=treatment, repo=repo,
                             s17_artifact_path=artifact_path,
                             source_checkpoint_path=checkpoint_path)
    return _run_prepared(prepared, dataset, attempt_directory=attempt_directory,
                         attempt_id=attempt_id, retry_of=retry_of, repo=repo)


def run_synthetic_reset(prepared: PreparedReset, dataset: Any) -> dict[str, Any]:
    """Controlled fixture path; sealed historical units cannot enter it."""
    if prepared.unit._seal is _FROZEN_UNIT_SEAL:
        raise PermissionError("Frozen units cannot use the synthetic execution path")
    return _run_prepared(prepared, dataset)


def _run_prepared(prepared: PreparedReset, dataset: Any, *,
                  attempt_directory: Path | None = None, attempt_id: str | None = None,
                  retry_of: str | None = None,
                  repo: Path | None = None) -> dict[str, Any]:
    """LR-Switch numerical kernel with the independently prepared Reset state."""
    config_identity = ExperimentConfig(**prepared.unit.checkpoint["config"])
    frozen_config = ExperimentConfig()
    production_like = all(
        getattr(config_identity, field) == getattr(frozen_config, field)
        for field in ("split_seed", "train_size", "val_size", "batch_size", "epochs",
                      "learning_rate", "momentum", "hidden_size", "num_workers", "device",
                      "dataset_name", "optimizer_name", "model_structure")
    )
    if prepared.unit._seal is _FROZEN_UNIT_SEAL or production_like:
        _require_execution_authorization(repo or REPO_ROOT)
    execution = replace(
        prepared,
        unit=replace(prepared.unit, checkpoint=deepcopy(prepared.unit.checkpoint),
                     historical=deepcopy(prepared.unit.historical)),
        model_state=deepcopy(prepared.model_state),
        optimizer_state=deepcopy(prepared.optimizer_state),
        generator_state=prepared.generator_state.clone(),
        pretraining_evidence=deepcopy(prepared.pretraining_evidence),
    )
    _verify_prepared(execution, dataset)
    config = _configuration(execution.unit.checkpoint)
    loader = _loader(config, dataset)
    if loader.split_fingerprint != execution.unit.checkpoint["split_fingerprint"]:
        raise ValueError("Dataset split fingerprint mismatch")
    generator = loader.train_generator
    generator.set_state(execution.generator_state.clone())
    expected_order = _iterate_order(loader.train_loader)
    if (expected_order != execution.unit.historical["wait_realized_orders"][3] or
            order_hashes([expected_order])[0] != execution.pretraining_evidence["expected_epoch18_order_sha256"]):
        raise ValueError("Epoch18 future order mismatch before training")
    action = "Continue" if execution.treatment == "L_10R" else "Switch"
    if execution.unit._seal is _FROZEN_UNIT_SEAL or production_like:
        _require_execution_authorization(repo or REPO_ROOT)
        if repo is None or attempt_directory is None or attempt_id is None:
            raise PermissionError("Production requires frozen roster and durable attempt evidence")
        canonical = load_canonical_roster(repo)[execution.unit.key]
        if (canonical.source_sha256 != execution.unit.source_sha256 or
                canonical.checkpoint_sha256 != execution.unit.checkpoint_sha256 or
                canonical.future_seed != execution.unit.future_seed or
                not states_equal(canonical.checkpoint, execution.unit.checkpoint) or
                not states_equal(canonical.historical, execution.unit.historical)):
            raise ValueError("Prepared production unit differs from frozen roster")
        verify_frozen_protocol(repo)
        artifact_path, checkpoint_path = _canonical_source_paths(canonical)
        if (Path(execution.pretraining_evidence["s17_artifact_path"]).resolve() !=
                artifact_path.resolve() or
                Path(execution.pretraining_evidence["source_checkpoint_path"]).resolve() !=
                checkpoint_path.resolve()):
            raise ValueError("Prepared source paths differ from frozen roster")
        if (sha256_file(Path(execution.pretraining_evidence["s17_artifact_path"])) !=
                execution.unit.source_sha256 or
                sha256_file(Path(execution.pretraining_evidence["source_checkpoint_path"])) !=
                execution.unit.checkpoint_sha256):
            raise ValueError("Frozen source bytes changed before training")
        prestep = _write_prestep(attempt_directory, attempt_id, execution,
                                retry_of=retry_of, _lifecycle_token=_LIFECYCLE_SEAL)
    else:
        prestep = None
    if execution.unit._seal is _FROZEN_UNIT_SEAL or production_like:
        _require_execution_authorization(repo or REPO_ROOT)
    # A pre-step without a terminal remains unresolved and blocks the next start.
    # Only an observed pre-result I/O failure can receive technical-invalid status.
    kernel_provenance()
    try:
        result = mechanism_lr_switch.run_one_epoch(
            execution.unit, dataset, execution.generator_state,
            execution.model_state, execution.optimizer_state, action=action)
    except OSError as error:
        if prestep is None:
            raise
        failure = observe_artifact_io_failure(error)
        return write_attempt(attempt_directory, attempt_id=attempt_id, prepared=execution,
                             result=None, technical_failure=failure, retry_of=retry_of,
                             _lifecycle_token=_LIFECYCLE_SEAL)
    if result["learning_rate"] != execution.pretraining_evidence["assigned_lr"]:
        raise AssertionError("Assigned LR mismatch")
    result["treatment"] = execution.treatment
    result["pretraining_evidence"] = deepcopy(execution.pretraining_evidence)
    if prestep is not None:
        result["prestep_file_sha256"] = sha256_file(prestep)
        result["retry_of"] = retry_of
    result["final_model_parameters"] = {
        row["state_path"]: row["value_sha256"]
        for row in result["state_evidence"]["model_parameters"]
    }
    result["final_momentum_buffers"] = {
        row["state_path"]: row["value_sha256"]
        for row in result["state_evidence"]["optimizer_state"]
    }
    result["training_step_count"] = len(result["realized_order"])
    if prestep is not None:
        return write_attempt(attempt_directory, attempt_id=attempt_id, prepared=execution,
                             result=result, retry_of=retry_of,
                             _lifecycle_token=_LIFECYCLE_SEAL)
    return result


def _write_prestep(directory: Path, attempt_id: str, prepared: PreparedReset, *,
                   retry_of: str | None = None,
                   _lifecycle_token: object | None = None) -> Path:
    """Persist the verified pre-step state before entering the training kernel."""
    if not attempt_id or "/" in attempt_id or ".." in attempt_id:
        raise ValueError("Invalid attempt ID")
    directory.mkdir(parents=True, exist_ok=True)
    _assert_start_admissible(directory, attempt_id, branch_identity(prepared), retry_of)
    path = directory / f"{attempt_id}.prestep"
    record = {"attempt_id": attempt_id, "scientific_identity": branch_identity(prepared),
              "pretraining_evidence": prepared.pretraining_evidence,
              "retry_of": retry_of,
              "authoritative_lifecycle": _lifecycle_token is _LIFECYCLE_SEAL,
              "status": "prepared_before_epoch18_training"}
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def _assert_start_admissible(directory: Path, attempt_id: str,
                             identity: dict[str, Any], retry_of: str | None) -> None:
    """Check lineage before a branch can enter its first scientific step."""
    key = identity["base_run_id"], identity["replica_id"], identity["treatment"]
    if list(directory.glob("*.terminal_pending")):
        raise PermissionError("Unreconciled terminal write blocks another start")
    records = {}
    for path in directory.glob("*.json"):
        if _is_completion_manifest(path):
            continue
        row = _verified_attempt(path)
        prior = row["scientific_identity"]
        if (prior["base_run_id"], prior["replica_id"], prior["treatment"]) == key:
            if prior != identity:
                raise PermissionError("Frozen branch identity changed")
            records[row["attempt_id"]] = row
    if any(row["classification"].startswith("scientifically_valid") for row in records.values()):
        raise PermissionError("Scientifically valid branch cannot be rerun")
    for path in directory.glob("*.prestep"):
        row = json.loads(path.read_text())
        prior = row["scientific_identity"]
        if (prior["base_run_id"], prior["replica_id"], prior["treatment"]) == key:
            if prior != identity:
                raise PermissionError("Pre-step identity changed")
            if row["attempt_id"] not in records:
                raise PermissionError("Prior started attempt remains unresolved")
    if attempt_id in records or (directory / f"{attempt_id}.prestep").exists():
        raise PermissionError("Attempt ID already used")
    if not records:
        if retry_of is not None:
            raise PermissionError("First attempt cannot claim retry lineage")
    else:
        prior = records.get(retry_of)
        if (prior is None or prior["classification"] != "technical_invalid" or
                any(row["retry_of"] == retry_of for row in records.values())):
            raise PermissionError("Retry must link to an unused same-identity invalid attempt")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": "NaN" if math.isnan(value) else
                ("+Inf" if value > 0 else "-Inf")}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_restore(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__nonfinite_float__"}:
            token = value["__nonfinite_float__"]
            if token not in {"NaN", "+Inf", "-Inf"}:
                raise ValueError("Invalid non-finite scientific value encoding")
            return {"NaN": float("nan"), "+Inf": float("inf"),
                    "-Inf": -float("inf")}[token]
        return {key: _json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    return value


def read_scientific_result(path: Path) -> dict[str, Any]:
    """Decode the canonical non-finite representation without dropping outcomes."""
    record = _verified_attempt(path)
    return _json_restore(record["result"])


def _verified_attempt(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Noncanonical JSON constant: {value}")))
    digest = record.pop("record_sha256")
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False,
                           allow_nan=False, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise ValueError(f"Attempt record hash mismatch: {path.name}")
    record["record_sha256"] = digest
    return record


def _is_completion_manifest(path: Path) -> bool:
    row = json.loads(path.read_text())
    return ("expected_branch_count" in row and "branches" in row and
            "attempt_id" not in row)


def _validate_manifest_identity(identity: dict[str, Any], proof: dict[str, Any],
                                roster: dict[tuple[int, int], Unit], filename: str) -> None:
    """Bind the recorded branch and pre-step to the frozen source unit."""
    try:
        state, replica = identity["base_run_id"], identity["replica_id"]
        if type(state) is not int or type(replica) is not int:
            raise ValueError("Noninteger canonical state or replica")
        unit = roster[state, replica]
        treatment = identity["treatment"]
        if treatment not in RESET_TREATMENTS:
            raise ValueError("Unknown Reset treatment")
        expected = branch_identity(PreparedReset(unit, treatment, {}, {},
                                                  torch.empty(0), proof))
        artifact, checkpoint = _canonical_source_paths(unit)
        candidate = json.loads(KERNEL_CANDIDATE.read_text())
        config = ExperimentConfig(**unit.checkpoint["config"])
        if config.num_workers != 0:
            raise ValueError("Unexpected canonical loader worker count")
        if (identity != expected or proof["protocol_id"] != PROTOCOL_ID or
                proof["protocol_sha256"] != PROTOCOL_SHA256 or
                proof["base_run_id"] != state or proof["replica_id"] != replica or
                proof["treatment"] != treatment or
                proof["s17_source_sha256"] != unit.source_sha256 or
                proof["checkpoint_sha256"] != unit.checkpoint_sha256 or
                proof["future_seed"] != unit.future_seed or
                Path(proof["s17_artifact_path"]).resolve() != artifact.resolve() or
                Path(proof["source_checkpoint_path"]).resolve() != checkpoint.resolve() or
                proof["expected_epoch18_order_sha256"] !=
                unit.historical["wait_epoch_order_sha256"][3] or
                proof["assigned_lr"] != (0.10 if treatment == "L_10R" else 0.05) or
                proof["implementation_sha256"] != candidate["components"]["reset"]["sha256"] or
                proof["kernel_provenance"] != candidate["components"] or
                proof["environment"] != candidate["environment"] or
                proof["canonical_post_s17_generator_sha256"] !=
                _canonical_generator_hash(unit.future_seed, config.train_size)):
            raise ValueError("Frozen roster or pre-step identity mismatch")
    except (KeyError, TypeError, IndexError, ValueError) as error:
        raise ValueError(f"Noncanonical Momentum-Reset identity: {filename}") from error


def write_attempt(directory: Path, *, attempt_id: str, prepared: PreparedReset,
                  result: dict[str, Any] | None,
                  technical_failure: ObservedTechnicalFailure | None = None,
                  retry_of: str | None = None,
                  _lifecycle_token: object | None = None) -> Path:
    """Record an immutable attempt; direct helper records are synthetic only."""
    if not attempt_id or "/" in attempt_id or ".." in attempt_id:
        raise ValueError("Invalid attempt ID")
    if result is not None and technical_failure is not None:
        raise ValueError("A generated scientific result cannot be technical-invalid")
    if technical_failure is not None and (
        not isinstance(technical_failure, ObservedTechnicalFailure) or
        technical_failure._seal is not _OBSERVED_FAILURE_SEAL
    ):
        raise ValueError("Technical invalidity requires observed machinery evidence")
    if technical_failure is not None and technical_failure.condition == "source_hash_mismatch":
        evidence = technical_failure.evidence
        if (technical_failure.stage != "source_identity" or
                observe_source_hash_mismatch(
                    Path(evidence["source_path"]), evidence["expected_sha256"]
                ).evidence != evidence):
            raise ValueError("Observed source mismatch evidence changed")
    if technical_failure is not None and technical_failure.condition == "artifact_io_failure":
        error = technical_failure._error
        if (technical_failure.stage != "artifact_integrity" or
                error is None or error.__traceback__ is None or
                technical_failure.evidence != {
                    "exception_type": type(error).__name__, "errno": error.errno,
                    "filename": error.filename, "message": str(error)}):
            raise ValueError("Captured artifact I/O failure evidence changed")
    if technical_failure is not None and technical_failure.condition not in {
        "source_hash_mismatch", "artifact_io_failure"
    }:
        raise ValueError("Unsupported technical failure category")
    if result is None and technical_failure is None:
        raise ValueError("An attempt without output needs an objective technical failure")
    prestep = directory / f"{attempt_id}.prestep"
    authoritative = _lifecycle_token is _LIFECYCLE_SEAL
    if prepared.unit._seal is _FROZEN_UNIT_SEAL and not authoritative:
        raise PermissionError("Frozen attempt must use the official lifecycle")
    if prestep.exists():
        pre_record = json.loads(prestep.read_text())
        if (pre_record["attempt_id"] != attempt_id or
                pre_record["scientific_identity"] != branch_identity(prepared) or
                pre_record["pretraining_evidence"] != prepared.pretraining_evidence or
                pre_record["retry_of"] != retry_of or
                pre_record.get("authoritative_lifecycle") is not authoritative):
            raise ValueError("Attempt disagrees with durable pre-step identity/lineage")
    elif authoritative:
        raise ValueError("Authoritative attempt lacks durable pre-step evidence")
    if technical_failure is None and prepared.unit._seal is _FROZEN_UNIT_SEAL:
        required = {
            "pretraining_evidence", "realized_order", "epoch_order_sha256",
            "training_step_count", "train_loss", "val_loss", "val_accuracy",
            "final_model_parameters", "final_momentum_buffers", "treatment",
            "prestep_file_sha256", "retry_of",
        }
        if (result is None or not required.issubset(result) or
                result["treatment"] != prepared.treatment or
                result["pretraining_evidence"] != prepared.pretraining_evidence):
            raise ValueError("Scientifically valid production record lacks required branch evidence")
        if sha256_file(prestep) != result["prestep_file_sha256"]:
            raise ValueError("Durable pre-step evidence hash mismatch")
        if result["retry_of"] != retry_of:
            raise ValueError("Durable pre-step evidence differs from scientific result")
    identity = branch_identity(prepared)
    key = (identity["base_run_id"], identity["replica_id"], identity["treatment"])
    for other_prestep in directory.glob("*.prestep"):
        if other_prestep == prestep:
            continue
        row = json.loads(other_prestep.read_text())
        prior = row["scientific_identity"]
        if ((prior["base_run_id"], prior["replica_id"], prior["treatment"]) == key and
                not (directory / f"{row['attempt_id']}.json").exists()):
            raise PermissionError("Prior started attempt remains unresolved")
    if prepared.unit._seal is _FROZEN_UNIT_SEAL:
        repo = Path(__file__).resolve().parents[1]
        canonical = load_canonical_roster(repo).get(prepared.unit.key)
        if (canonical is None or
                (canonical.source_sha256, canonical.checkpoint_sha256,
                 canonical.future_seed) !=
                (prepared.unit.source_sha256, prepared.unit.checkpoint_sha256,
                 prepared.unit.future_seed) or
                not states_equal(canonical.checkpoint, prepared.unit.checkpoint) or
                not states_equal(canonical.historical, prepared.unit.historical)):
            raise ValueError("Attempt source differs from frozen 72-unit roster")
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for existing_path in directory.glob("*.json"):
        if _is_completion_manifest(existing_path):
            continue
        existing = _verified_attempt(existing_path)
        prior_identity = existing["scientific_identity"]
        if (prior_identity["base_run_id"], prior_identity["replica_id"],
                prior_identity["treatment"]) == key:
            rows.append(existing)
    if any(row["scientific_identity"] != identity for row in rows):
        raise PermissionError("Branch identity cannot change between attempts")
    if any(row["classification"].startswith("scientifically_valid") for row in rows):
        raise PermissionError("First scientifically valid attempt already recorded")
    if not rows and retry_of is not None:
        raise PermissionError("First attempt cannot be a retry")
    if rows:
        if retry_of is None or len(rows) != 1 + sum(row["retry_of"] is not None for row in rows):
            raise PermissionError("Later attempt requires explicit invalid-attempt lineage")
        prior = next((row for row in rows if row["attempt_id"] == retry_of), None)
        if prior is None or prior["classification"] != "technical_invalid":
            raise PermissionError("Retry must link to a technical-invalid attempt")
        if any(row["retry_of"] == retry_of for row in rows):
            raise PermissionError("Technical-invalid attempt already retried")
    classification = ("technical_invalid" if technical_failure else
                      "scientifically_valid_primary_evaluable" if math.isfinite(
                          float(result["val_loss"])) else
                      "scientifically_valid_primary_non_evaluable")
    failure_record = (None if technical_failure is None else {
        "stage": technical_failure.stage, "condition": technical_failure.condition,
        "evidence": technical_failure.evidence})
    record = {
        "attempt_id": attempt_id, "scientific_identity": identity,
        "classification": classification,
        "technical_failure": failure_record, "scientific_output_generated": result is not None,
        "retry_of": retry_of, "pretraining_evidence": prepared.pretraining_evidence,
        "authoritative_lifecycle": authoritative,
        "prestep_file_sha256": sha256_file(prestep) if prestep.exists() else None,
        "result": result,
    }
    payload = json.dumps(_json_safe(record), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode()
    record["record_sha256"] = hashlib.sha256(payload).hexdigest()
    path = directory / f"{attempt_id}.json"
    pending = directory / f"{attempt_id}.terminal_pending"
    if authoritative:
        with pending.open("x", encoding="utf-8") as handle:
            json.dump({"attempt_id": attempt_id, "scientific_identity": identity}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(record), handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if authoritative:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        pending.unlink()
    return path


def build_completion_manifest(attempt_directory: Path, manifest_path: Path,
                              repo: Path | None = None) -> dict[str, Any]:
    """Admit only canonical lifecycle records into the authoritative 144-branch manifest."""
    repo = repo or Path(__file__).resolve().parents[1]
    roster = load_canonical_roster(repo)
    if list(attempt_directory.glob("*.terminal_pending")):
        raise ValueError("Terminal write requires objective reconciliation")
    expected = {
        (state, replica, treatment)
        for state in range(1, 37) for replica in (1, 2)
        for treatment in ("L_10R", "L_05R")
    }
    attempts: dict[tuple[int, int, str], list[dict[str, Any]]] = {key: [] for key in expected}
    unresolved: dict[tuple[int, int, str], list[dict[str, Any]]] = {key: [] for key in expected}
    for path in sorted(attempt_directory.glob("*.json")):
        if path.resolve() == manifest_path.resolve():
            continue
        record = _verified_attempt(path)
        identity = record["scientific_identity"]
        key = identity["base_run_id"], identity["replica_id"], identity["treatment"]
        if (key not in expected or identity["protocol_id"] != PROTOCOL_ID or
                identity["protocol_sha256"] != PROTOCOL_SHA256):
            raise ValueError(f"Unexpected scientific branch: {path.name}")
        _validate_manifest_identity(identity, record["pretraining_evidence"],
                                    roster, path.name)
        if record.get("authoritative_lifecycle") is not True or not record.get("prestep_file_sha256"):
            raise ValueError(f"Non-authoritative attempt: {path.name}")
        if path.name != f"{record['attempt_id']}.json":
            raise ValueError(f"Attempt filename/identity mismatch: {path.name}")
        result = record["result"]
        if record["scientific_output_generated"] != (result is not None):
            raise ValueError(f"Attempt output presence mismatch: {path.name}")
        if result is None:
            if record["classification"] != "technical_invalid" or not record["technical_failure"]:
                raise ValueError(f"Unsubstantiated technical disposition: {path.name}")
        else:
            expected_classification = (
                "scientifically_valid_primary_evaluable" if math.isfinite(
                    float(_json_restore(result["val_loss"]))) else
                "scientifically_valid_primary_non_evaluable")
            if record["classification"] != expected_classification or record["technical_failure"] is not None:
                raise ValueError(f"Scientific outcome disposition mismatch: {path.name}")
            if (result.get("pretraining_evidence") != record["pretraining_evidence"] or
                    result.get("prestep_file_sha256") != record["prestep_file_sha256"] or
                    result.get("retry_of") != record["retry_of"] or
                    result.get("treatment") != identity["treatment"]):
                raise ValueError(f"Scientific result/pre-step binding mismatch: {path.name}")
        record["file_sha256"] = sha256_file(path)
        record["filename"] = path.name
        attempts[key].append(record)
    for path in sorted(attempt_directory.glob("*.prestep")):
        row = json.loads(path.read_text())
        identity = row["scientific_identity"]
        key = identity["base_run_id"], identity["replica_id"], identity["treatment"]
        if key not in expected or path.name != f"{row['attempt_id']}.prestep":
            raise ValueError(f"Unexpected pre-step scientific branch: {path.name}")
        _validate_manifest_identity(identity, row["pretraining_evidence"], roster, path.name)
        if row.get("authoritative_lifecycle") is not True:
            raise ValueError(f"Non-authoritative pre-step: {path.name}")
        matching = [record for record in attempts[key]
                    if record["attempt_id"] == row["attempt_id"]]
        if matching:
            if (matching[0]["scientific_identity"] != identity or
                    matching[0]["prestep_file_sha256"] != sha256_file(path) or
                    matching[0]["pretraining_evidence"] != row["pretraining_evidence"] or
                    matching[0]["retry_of"] != row["retry_of"]):
                raise ValueError(f"Pre-step/attempt identity mismatch: {path.name}")
        else:
            unresolved[key].append({"attempt_id": row["attempt_id"],
                                    "retry_of": row["retry_of"],
                                    "filename": path.name,
                                    "file_sha256": sha256_file(path),
                                    "scientific_identity": identity})
    for key, rows in attempts.items():
        for row in rows:
            if not (attempt_directory / f"{row['attempt_id']}.prestep").is_file():
                raise ValueError(f"Terminal lacks authoritative pre-step: {row['filename']}")
    branches = {}
    for key in sorted(expected):
        rows = attempts[key]
        valid = [row for row in rows if row["classification"].startswith("scientifically_valid")]
        if len(valid) > 1:
            raise ValueError(f"Multiple valid attempts for {key}")
        if len({row["attempt_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate attempt ID for {key}")
        if rows and any(row["scientific_identity"] != rows[0]["scientific_identity"] for row in rows):
            raise ValueError(f"Changed branch identity in attempt lineage for {key}")
        if len(unresolved[key]) > 1 or (unresolved[key] and valid):
            raise ValueError(f"Ambiguous or post-valid started attempt for {key}")
        if rows and unresolved[key] and unresolved[key][0]["scientific_identity"] != rows[0]["scientific_identity"]:
            raise ValueError(f"Changed pre-step identity for {key}")
        if unresolved[key]:
            pending_retry = unresolved[key][0]["retry_of"]
            if ((not rows and pending_retry is not None) or
                    (rows and not any(row["attempt_id"] == pending_retry and
                                      row["classification"] == "technical_invalid"
                                      for row in rows))):
                raise ValueError(f"Unlinked started retry for {key}")
        if len([row for row in rows if row["retry_of"] is None]) != (1 if rows else 0):
            raise ValueError(f"Unlinked attempt in lineage for {key}")
        for row in rows:
            retry_of = row["retry_of"]
            if retry_of is not None and not any(
                prior["attempt_id"] == retry_of and prior["classification"] == "technical_invalid"
                for prior in rows
            ):
                raise ValueError(f"Invalid retry lineage for {key}")
        if len({row["retry_of"] for row in rows if row["retry_of"] is not None}) != max(0, len(rows)-1):
            raise ValueError(f"Forked retry lineage for {key}")
        disposition = (valid[0]["classification"] if valid else
                       ("attempt_started_unresolved" if unresolved[key] else
                        ("technical_invalid_unresolved" if rows else "missing")))
        branches[f"{key[0]:03d}-A-r{key[1]:02d}-{key[2]}"] = {
            "disposition": disposition,
            "authoritative_attempt": valid[0]["attempt_id"] if valid else None,
            "unresolved_prestep": unresolved[key][0] if unresolved[key] else None,
            "attempts": [{field: row[field] for field in
                          ("attempt_id", "filename", "file_sha256", "record_sha256",
                           "classification", "retry_of", "scientific_output_generated",
                           "technical_failure")}
                         for row in rows],
        }
    manifest = {
        "protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "expected_branch_count": 144,
        "scientifically_valid_count": sum(row["disposition"].startswith("scientifically_valid")
                                          for row in branches.values()),
        "primary_evaluable_branch_count": sum(row["disposition"] == "scientifically_valid_primary_evaluable"
                                              for row in branches.values()),
        "primary_non_evaluable_branch_count": sum(row["disposition"] == "scientifically_valid_primary_non_evaluable"
                                                  for row in branches.values()),
        "technical_invalid_attempt_count": sum(
            attempt["classification"] == "technical_invalid"
            for row in branches.values() for attempt in row["attempts"]),
        "unresolved_technical_branch_count": sum(row["disposition"] == "technical_invalid_unresolved"
                                                 for row in branches.values()),
        "unresolved_started_branch_count": sum(row["disposition"] == "attempt_started_unresolved"
                                               for row in branches.values()),
        "branches": branches,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return manifest


def load_bound_keep_records(stage_i_dir: Path, stage_ii_dir: Path,
                            primary_analysis_path: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Read only the 144 historical cells named and hashed by frozen authority."""
    stage_i_path = stage_i_dir / "stage_i_gate.json"
    stage_ii_path = stage_ii_dir / "stage_ii_completion_manifest.json"
    for path, digest in ((stage_i_path, STAGE_I_GATE_SHA256),
                         (stage_ii_path, STAGE_II_MANIFEST_SHA256),
                         (primary_analysis_path, PRIMARY_ANALYSIS_SHA256)):
        if sha256_file(path) != digest:
            raise ValueError(f"Bound historical authority hash mismatch: {path}")
    gate = json.loads(stage_i_path.read_text())
    manifest = json.loads(stage_ii_path.read_text())
    primary = json.loads(primary_analysis_path.read_text())
    if (gate.get("status") != "72/72 PASS" or
            manifest.get("status") != "72/72 COMPLETE — INTEGRITY VERIFIED" or
            primary.get("stage_i_gate_sha256") != STAGE_I_GATE_SHA256 or
            primary.get("stage_ii_manifest_sha256") != STAGE_II_MANIFEST_SHA256):
        raise ValueError("Historical Keep authority status or chain mismatch")
    bound = {}
    for state in range(1, 37):
        for replica in (1, 2):
            for treatment, root, hashes, filename, action, lr in (
                ("L_05K", stage_i_dir, gate["switch_records"],
                 f"switch-{state:03d}-A-r{replica:02d}.json", "Switch", 0.05),
                ("L_10K", stage_ii_dir, manifest["continue_records"],
                 f"continue-{state:03d}-A-r{replica:02d}.json", "Continue", 0.10),
            ):
                if filename not in hashes or sha256_file(root / filename) != hashes[filename]:
                    raise ValueError(f"Historical Keep record hash mismatch: {filename}")
                record = json.loads((root / filename).read_text())
                result = record["result"]
                if (record["base_run_id"], record["replica_id"], record["action"],
                    result["base_run_id"], result["replica_id"], result["action"],
                    result["epoch"], result["learning_rate"]) != (
                    state, replica, action, state, replica, action, 18, lr
                ) or record["status"] not in {"PASS", "RECORDED"}:
                    raise ValueError(f"Historical Keep scientific identity mismatch: {filename}")
                bound[state, replica, treatment] = result
            keep_high = bound[state, replica, "L_10K"]
            keep_low = bound[state, replica, "L_05K"]
            for field in ("source_sha256", "checkpoint_sha256", "future_seed",
                          "realized_order", "epoch_order_sha256"):
                if keep_high[field] != keep_low[field]:
                    raise ValueError(f"Historical Keep pairing mismatch: {state}, {replica}, {field}")
    if len(gate["switch_records"]) != 72 or len(manifest["continue_records"]) != 72:
        raise ValueError("Historical Keep set has extra or missing records")
    # Match the frozen LR-Switch three-column NumPy aggregation layout exactly.
    historical_values = np.empty((36, 2, 3), dtype=np.float64)
    for state in range(1, 37):
        for index, replica in enumerate((1, 2)):
            historical_values[state - 1, index, 0] = (
                bound[state, replica, "L_10K"]["val_loss"] -
                bound[state, replica, "L_05K"]["val_loss"]
            )
            historical_values[state - 1, index, 1:] = 0
    values = historical_values.mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(HISTORICAL_BOOTSTRAP_SEED))
    indices = rng.integers(0, 36, size=(10_000, 36))
    historical_ci = np.percentile(values[indices].mean(axis=1), [2.5, 97.5], axis=0)
    estimate = float(values.mean(axis=0)[0])
    if (estimate != 0.04110032269702188 or
            historical_ci[:, 0].tolist() !=
            [0.033847031257023194, 0.04871296943619432] or
            primary["delta_switch"] != estimate):
        raise ValueError("Bound Keep row fails frozen historical primary reconstruction")
    return bound


def analyze_factorial(cells: dict[tuple[int, int, str], dict[str, Any]]) -> dict[str, Any]:
    """Generic calculation/testing helper, not authoritative primary-analysis admission."""
    treatments = ("L_10K", "L_05K", "L_10R", "L_05R")
    required = {(state, replica, treatment) for state in range(1, 37)
                for replica in (1, 2) for treatment in treatments}
    if set(cells) != required:
        raise ValueError("Factorial analysis requires exact 36 x 2 x 4 coverage")
    outcomes = {"epoch18_val_loss": "val_loss", "epoch18_train_loss": "train_loss",
                "epoch18_val_accuracy": "val_accuracy"}
    estimates = {}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, 36, size=(10_000, 36))
    for label, field in outcomes.items():
        data = np.array([[[float(_json_restore(cells[state, replica, treatment][field]))
                           for treatment in treatments] for replica in (1, 2)]
                         for state in range(1, 37)], dtype=np.float64)
        if not np.isfinite(data).all():
            affected = [(state + 1, replica + 1, treatments[treatment])
                        for state, replica, treatment in np.argwhere(~np.isfinite(data))]
            if label == "epoch18_val_loss":
                return {"status": "NON-EVALUABLE AS SPECIFIED",
                        "affected_outcome": label, "affected_cells": affected,
                        "bootstrap_not_run": True}
            estimates[label] = {"status": "NON-EVALUABLE AS SPECIFIED",
                                "affected_cells": affected}
            continue
        k10, k05, r10, r05 = (data[:, :, index] for index in range(4))
        contrast = np.stack((k10 - k05, r10 - r05,
                             (k10 - k05) - (r10 - r05), k10 - r10, k05 - r05), axis=2)
        if not np.allclose(contrast[:, :, 2], contrast[:, :, 3] - contrast[:, :, 4],
                           rtol=0, atol=2e-16):
            raise ArithmeticError("Unit-level factorial identity failed")
        state_means = contrast.mean(axis=1)
        estimate = state_means.mean(axis=0)
        if not np.isclose(estimate[2], estimate[3] - estimate[4], rtol=0, atol=2e-16):
            raise ArithmeticError("Aggregate factorial identity failed")
        ci = np.percentile(state_means[indices].mean(axis=1), [2.5, 97.5], axis=0)
        names = ("Delta_K", "Delta_R", "I", "R_.10", "R_.05")
        estimates[label] = {name: {"estimate": float(estimate[index]),
                                   "ci_95": ci[:, index].tolist()}
                            for index, name in enumerate(names)}
    return {"status": "EVALUABLE", "protocol_id": PROTOCOL_ID,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": 10_000,
            "n_states": 36, "replicas_per_state": 2, "outcomes": estimates}

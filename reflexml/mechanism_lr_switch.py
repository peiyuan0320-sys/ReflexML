"""Frozen LR-switch mechanism kernel. No production entry point or auto-execution.

All source paths and expected hashes must come from the audited Phase 5 ledger.
The module deliberately keeps the historical sources read-only. A separate,
independent audit and execution authorization are required before using it on
the real 72 units.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import base64
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .branching import order_hashes, training_state_finiteness_evidence
from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .data import make_loaders_from_datasets
from .model import FashionMLP
from .phase5 import assert_frozen_protocol, sha256_file, validate_dataset_against_registration
from .training import evaluate, train_one_epoch


PROTOCOL_ID = "ReflexML-Mechanism-LR-Switch-v1"
PROTOCOL_SHA256 = "2c5d58f45c5db454eb091023201a40d49cf0b75fd783c9a826a596fe8c217f48"
APPROVED_STAGE_I_GATE_SHA256 = "dd7d4273a215f1c71cdf816bce3334dc846c37e31b92f0e29e46b7748d67dc80"
PROTOCOL_SIZE = 23813
BOOTSTRAP_SEED = 12114954806208443255
STATES = 36
REPLICAS = (1, 2)
GATE_UNITS = STATES * len(REPLICAS)
FROZEN_BRANCH_LEDGER_SHA256 = "9382e1a737a96024ffb77d73344a309dc8e540b71491f2d4062e690962fcc634"
_GATE_SEAL = object()
_FROZEN_UNIT_SEAL = object()
FROZEN_DATASET_REGISTRATION_SHA256 = "18fcff67679cf832d9398aa8251826b57470d5554de9180ace1fda733a923898"


def verify_protocol(repo: Path) -> None:
    repo = repo.resolve()
    record = json.loads((repo / "MECHANISM_LR_SWITCH_PROTOCOL.freeze.json").read_text())
    path = repo / "MECHANISM_LR_SWITCH_PROTOCOL.md"
    if record != {
        "protocol_id": PROTOCOL_ID,
        "protocol_path": str(path),
        "status": "FROZEN",
        "sha256": PROTOCOL_SHA256,
        "size_bytes": PROTOCOL_SIZE,
        "scientific_closure_verdict": "AMENDMENTS CLOSED — READY TO FREEZE",
        "execution_authorized": False,
    } or path.stat().st_size != PROTOCOL_SIZE or sha256_file(path) != PROTOCOL_SHA256:
        raise ValueError("Mechanism protocol bytes or freeze record differ from frozen identity")
    assert_frozen_protocol(repo)


def _decode_tensor(row: dict[str, Any]) -> torch.Tensor:
    if row.get("state_encoding") != "numpy_npy_base64_v1":
        raise ValueError("Unsupported historical tensor encoding")
    raw = base64.b64decode(row["state_npy_base64"], validate=True)
    stream = io.BytesIO(raw)
    array = np.load(stream, allow_pickle=False)
    if array.dtype.kind not in {"f", "c"} or stream.tell() != len(raw):
        raise ValueError("Invalid historical numerical tensor")
    canonical = io.BytesIO()
    np.save(canonical, np.ascontiguousarray(array), allow_pickle=False)
    if canonical.getvalue() != raw:
        raise ValueError("Noncanonical historical numerical bytes")
    checks = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "value_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(array))),
    }
    if any(row.get(key) != value for key, value in checks.items()):
        raise ValueError("Historical tensor summary disagrees with numerical bytes")
    return torch.from_numpy(array.copy())


def _evidence_map(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    result = {}
    for row in rows:
        path = row["state_path"]
        if path != row.get("path") or path in result:
            raise ValueError("Historical state path missing or duplicated")
        result[path] = _decode_tensor(row)
    return result


@dataclass(frozen=True)
class Unit:
    base_run_id: int
    replica_id: int
    future_seed: int
    checkpoint: dict[str, Any]
    historical: dict[str, Any]
    source_sha256: str
    checkpoint_sha256: str
    _seal: object | None = None

    @property
    def key(self) -> tuple[int, int]:
        return self.base_run_id, self.replica_id


def load_unit(
    repo: Path, *, base_run_id: int, replica_id: int,
    checkpoint_path: Path, checkpoint_sha256: str,
    artifact_path: Path, artifact_sha256: str,
) -> Unit:
    """Read exact registered Phase 5 bytes; never infer identities from filenames."""
    verify_protocol(repo)
    if not (1 <= base_run_id <= STATES) or replica_id not in REPLICAS:
        raise ValueError("Unit is outside frozen A1/A2 selection")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("Epoch14 checkpoint hash mismatch")
    if sha256_file(artifact_path) != artifact_sha256:
        raise ValueError("Historical branch artifact hash mismatch")
    artifact = json.loads(artifact_path.read_text())
    identity = artifact["scientific_identity"]
    if any((identity.get(k) != v for k, v in {
        "kind": "branch", "protocol_version": "Phase5-v1",
        "base_run_id": base_run_id, "block": "A", "replica_id": replica_id,
        "checkpoint_sha256": checkpoint_sha256,
    }.items())):
        raise ValueError("Historical artifact identity mismatch")
    if artifact.get("terminal_state") != "terminal_success":
        raise ValueError("Selected historical unit is not a successful terminal")
    import csv
    with (repo / "phase5_preflight_r4/future_seed_mapping.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if (int(row["base_run_id"]), row["block"], int(row["replica_id"]))
                == (base_run_id, "A", replica_id)]
    if len(rows) != 1 or rows[0]["pre_outcome"] != "True":
        raise ValueError("Frozen future mapping missing or ambiguous")
    seed = int(rows[0]["future_seed"])
    if (identity.get("future_seed") != seed or
            identity.get("future_seed_namespace") != rows[0]["namespace"] or
            identity.get("future_seed_counter") != int(rows[0]["counter"])):
        raise ValueError("Historical artifact disagrees with frozen future mapping")
    if sha256_file(repo / "phase5_preflight_r4/future_seed_mapping.csv") != identity.get("future_mapping_sha256"):
        raise ValueError("Frozen future mapping hash mismatch")
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    if checkpoint.get("epoch") != 14 or checkpoint.get("current_learning_rate") != 0.1:
        raise ValueError("Primary checkpoint epoch/LR mismatch")
    return Unit(base_run_id, replica_id, seed, checkpoint, artifact,
                artifact_sha256, checkpoint_sha256)


def load_frozen_units(repo: Path, branch_ledger_path: Path, base_output_root: Path) -> list[Unit]:
    """Resolve the selected 72 from the exact frozen ledger and primary records."""
    verify_protocol(repo)
    if sha256_file(branch_ledger_path) != FROZEN_BRANCH_LEDGER_SHA256:
        raise ValueError("Frozen Phase 5 branch ledger hash mismatch")
    required = {(s, r) for s in range(1, STATES + 1) for r in REPLICAS}
    events: dict[tuple[int, int], list[dict[str, Any]]] = {key: [] for key in required}
    with branch_ledger_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            identity = row.get("scientific_identity", {})
            if identity.get("block") == "A":
                key = identity.get("base_run_id"), identity.get("replica_id")
                if key in events:
                    events[key].append(row)
    units = []
    for base_run_id, replica_id in sorted(required):
        rows = events[base_run_id, replica_id]
        if [row["event_type"] for row in rows] != ["attempt_started", "artifact_produced", "terminal_success"]:
            raise ValueError("Selected unit has incomplete or retry Phase 5 lineage")
        if (rows[0]["attempt_id"] != rows[1]["attempt_id"] or
                rows[0]["attempt_id"] != rows[2]["attempt_id"] or
                rows[0]["scientific_identity"] != rows[1]["scientific_identity"] or
                rows[0]["scientific_identity"] != rows[2]["scientific_identity"] or
                rows[1]["artifact_sha256"] != rows[2]["artifact_sha256"]):
            raise ValueError("Selected Phase 5 event lineage disagrees")
        identity = rows[0]["scientific_identity"]
        primary_path = base_output_root / f"base_{base_run_id:03d}_attempt_001/primary_state.json"
        primary = json.loads(primary_path.read_text())
        if (primary.get("base_run_id") != base_run_id or primary.get("epoch") != 14 or
                primary.get("checkpoint_sha256") != identity["checkpoint_sha256"] or
                primary.get("protocol_version") != "Phase5-v1" or
                primary.get("state_manifested_before_future_outcomes") is not True):
            raise ValueError("Primary state record conflicts with branch identity")
        unit = load_unit(
            repo, base_run_id=base_run_id, replica_id=replica_id,
            checkpoint_path=Path(primary["checkpoint_path"]),
            checkpoint_sha256=identity["checkpoint_sha256"],
            artifact_path=Path(rows[1]["artifact_path"]),
            artifact_sha256=rows[1]["artifact_sha256"],
        )
        units.append(replace(unit, _seal=_FROZEN_UNIT_SEAL))
    return units


def validate_production_dataset(repo: Path, dataset: Any) -> None:
    """Apply the existing canonical Phase 5 dataset identity check."""
    verify_protocol(repo)
    path = repo / "phase5_preflight_r4/dataset_identity.json"
    if sha256_file(path) != FROZEN_DATASET_REGISTRATION_SHA256:
        raise ValueError("Frozen Phase 5 dataset registration hash mismatch")
    validate_dataset_against_registration(
        dataset, json.loads(path.read_text()), canonical_registration=True
    )


def _configuration(checkpoint: dict[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig(**checkpoint["config"])
    expected = ExperimentConfig()
    # The base seed varies by state. All other frozen scientific configuration is fixed.
    for field in ("split_seed", "train_size", "val_size", "batch_size", "epochs",
                  "learning_rate", "momentum", "hidden_size", "num_workers", "device",
                  "dataset_name", "optimizer_name", "model_structure"):
        if getattr(config, field) != getattr(expected, field):
            raise ValueError(f"Frozen configuration mismatch: {field}")
    return config


def reconstruct_epoch17(unit: Unit) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Restore epoch17 numerical bytes and epoch14 SGD group metadata."""
    config = _configuration(unit.checkpoint)
    history = unit.historical["wait_history"]
    if [row["epoch"] for row in history] != [15, 16, 17, 18] or [row["learning_rate"] for row in history] != [0.1, 0.1, 0.1, 0.05]:
        raise ValueError("Historical Wait3 history/schedule mismatch")
    row17 = unit.historical["wait_state_finiteness_evidence"][2]
    if row17["epoch"] != 17:
        raise ValueError("Epoch17 evidence missing")
    with torch.random.fork_rng(devices=[]):
        model = FashionMLP(config.hidden_size)
    names = list(dict(model.named_parameters()))
    model_state = _evidence_map(row17["model_parameters"])
    expected_model = {f"model.{name}" for name in model.state_dict()}
    if set(model_state) != expected_model or set(model.state_dict()) != set(names):
        raise ValueError("Model parameter identity inventory mismatch")
    for name, template in model.state_dict().items():
        value = model_state[f"model.{name}"]
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(f"Model tensor shape/dtype mismatch: {name}")
    checkpoint_model = unit.checkpoint["model_state_dict"]
    if list(checkpoint_model) != list(model.state_dict()):
        raise ValueError("Epoch14 model parameter order mismatch")
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    frozen_opt = unit.checkpoint["optimizer_state_dict"]
    group = frozen_opt["param_groups"]
    constructed = optimizer.state_dict()["param_groups"]
    if len(group) != len(constructed) or len(group) != 1:
        raise ValueError("Frozen optimizer must have one SGD parameter group")
    ids = group[0]["params"]
    if ids != constructed[0]["params"] or len(ids) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Epoch14 optimizer membership/order mismatch")
    if group[0] != constructed[0] or any(group[0][k] != v for k, v in {
        "lr": 0.1, "momentum": 0.9, "weight_decay": 0,
        "dampening": 0, "nesterov": False,
    }.items()):
        raise ValueError("Epoch14 optimizer configuration differs from production SGD")
    if set(frozen_opt["state"]) != set(ids):
        raise ValueError("Epoch14 SGD state membership mismatch")
    if any(set(frozen_opt["state"][identifier]) != {"momentum_buffer"} for identifier in ids):
        raise ValueError("Unexpected epoch14 SGD per-parameter state")
    momentum = _evidence_map(row17["optimizer_state"])
    expected_momentum = {f"optimizer.state.{identifier}.momentum_buffer" for identifier in ids}
    if set(momentum) != expected_momentum:
        raise ValueError("Epoch17 momentum inventory/mapping mismatch")
    for identifier, name in zip(ids, names, strict=True):
        tensor = momentum[f"optimizer.state.{identifier}.momentum_buffer"]
        parameter = model_state[f"model.{name}"]
        if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
            raise ValueError(f"Momentum tensor shape/dtype mismatch: {name}")
    restored_model = {name: model_state[f"model.{name}"].clone() for name in names}
    restored_opt = {
        "state": {identifier: {"momentum_buffer": momentum[f"optimizer.state.{identifier}.momentum_buffer"].clone()}
                  for identifier in ids},
        "param_groups": deepcopy(group),
    }
    return restored_model, restored_opt


def _loader(config: ExperimentConfig, dataset: Any):
    return make_loaders_from_datasets(dataset, [], config, audit_order=True)


def _iterate_order(loader) -> list[list[int]]:
    order = []
    for batch in loader:
        if len(batch) != 3:
            raise ValueError("Expected indexed production DataLoader batches")
        order.append(batch[2].tolist())
    return order


def reconstruct_epoch18_generator(unit: Unit, dataset: Any) -> torch.Tensor:
    config = _configuration(unit.checkpoint)
    data = _loader(config, dataset)
    if data.split_fingerprint != unit.checkpoint["split_fingerprint"]:
        raise ValueError("Frozen dataset split fingerprint mismatch")
    data.train_generator.manual_seed(unit.future_seed)
    historical = unit.historical
    if historical["wait_realized_orders"] != historical["now_realized_orders"]:
        raise ValueError("Historical pair orders disagree")
    expected = historical["wait_realized_orders"]
    hashes = historical["wait_epoch_order_sha256"]
    if len(expected) != 4 or hashes != order_hashes(expected):
        raise ValueError("Historical order evidence invalid")
    for epoch_index in range(3):
        actual = _iterate_order(data.train_loader)
        if actual != expected[epoch_index] or order_hashes([actual])[0] != hashes[epoch_index]:
            raise ValueError(f"Reconstructed epoch{epoch_index + 15} order mismatch")
    canonical = data.train_generator.get_state().clone()
    # Validate O18 through an independent generator and loader, leaving canonical untouched.
    validation = _loader(config, dataset)
    validation.train_generator.set_state(canonical.clone())
    order18 = _iterate_order(validation.train_loader)
    if order18 != expected[3] or order_hashes([order18])[0] != hashes[3]:
        raise ValueError("Reconstructed epoch18 order mismatch")
    if not torch.equal(canonical, data.train_generator.get_state()):
        raise AssertionError("Epoch18 validation mutated canonical generator state")
    return canonical


def run_one_epoch(unit: Unit, dataset: Any, generator_state: torch.Tensor,
                  model_state: dict[str, torch.Tensor], optimizer_state: dict[str, Any],
                  *, action: str) -> dict[str, Any]:
    """Run one independently instantiated branch; never mutates source states."""
    if action not in {"Switch", "Continue"}:
        raise ValueError("Unknown mechanism action")
    config = _configuration(unit.checkpoint)
    with torch.random.fork_rng(devices=[]):
        model = FashionMLP(config.hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    model.load_state_dict(deepcopy(model_state), strict=True)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    data = _loader(config, dataset)
    if data.split_fingerprint != unit.checkpoint["split_fingerprint"]:
        raise ValueError("Frozen dataset split fingerprint mismatch")
    data.train_generator.set_state(generator_state.clone())
    lr = 0.05 if action == "Switch" else 0.10
    optimizer.param_groups[0]["lr"] = lr
    if any(group["lr"] != lr for group in optimizer.param_groups):
        raise AssertionError("Mechanism LR assignment failed")
    criterion = nn.CrossEntropyLoss()
    order: list[list[int]] = []
    loss = train_one_epoch(model, data.train_loader, optimizer, criterion,
                           torch.device("cpu"), order)
    validation = evaluate(model, data.val_loader, criterion, torch.device("cpu"))
    expected_order = unit.historical["wait_realized_orders"][3]
    if order != expected_order:
        raise ValueError("Branch epoch18 minibatch order/boundaries mismatch")
    return {
        "base_run_id": unit.base_run_id, "replica_id": unit.replica_id,
        "source_sha256": unit.source_sha256,
        "checkpoint_sha256": unit.checkpoint_sha256,
        "future_seed": unit.future_seed, "action": action, "epoch": 18,
        "learning_rate": lr, "realized_order": order,
        "epoch_order_sha256": order_hashes([order])[0],
        "train_loss": loss, "val_loss": validation.loss,
        "val_accuracy": validation.accuracy,
        "state_evidence": training_state_finiteness_evidence(model, optimizer),
    }


def check_switch(unit: Unit, result: dict[str, Any]) -> None:
    """Exact Stage I comparison, including canonical tensor bytes."""
    historical = unit.historical
    expected = historical["wait_history"][3]
    checks = {
        "base_run_id": unit.base_run_id, "replica_id": unit.replica_id,
        "action": "Switch", "epoch": 18, "learning_rate": expected["learning_rate"],
        "realized_order": historical["wait_realized_orders"][3],
        "epoch_order_sha256": historical["wait_epoch_order_sha256"][3],
        "train_loss": expected["train_loss"], "val_loss": expected["val_loss"],
        "val_accuracy": expected["val_accuracy"],
        "state_evidence": {
            "model_parameters": historical["wait_state_finiteness_evidence"][3]["model_parameters"],
            "optimizer_state": historical["wait_state_finiteness_evidence"][3]["optimizer_state"],
        },
    }
    for key, value in checks.items():
        if result.get(key) != value:
            raise ValueError(f"Stage I exact Switch reproduction mismatch: {unit.key} {key}")


@dataclass(frozen=True)
class StageIPass:
    keys: frozenset[tuple[int, int]]
    results: tuple[dict[str, Any], ...]
    _seal: object
    manifest_path: Path | None = None
    manifest_sha256: str | None = None


def _write_record(directory: Path, unit: Unit, action: str, status: str,
                  result: dict[str, Any] | None, error: Exception | None = None) -> None:
    """Write one immutable attempt record, including failures."""
    record = {
        "protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "base_run_id": unit.base_run_id, "replica_id": unit.replica_id,
        "source_sha256": unit.source_sha256,
        "checkpoint_sha256": unit.checkpoint_sha256,
        "future_seed": unit.future_seed,
        "action": action, "status": status,
        "result": result,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }
    def json_safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return {"__nonfinite_float__": "NaN" if math.isnan(value) else
                    ("+Inf" if value > 0 else "-Inf")}
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    payload = json.dumps(json_safe(record), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = directory / f"{action.lower()}-{unit.base_run_id:03d}-A-r{unit.replica_id:02d}.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def run_stage_i(units: Sequence[Unit], dataset: Any, repo: Path,
                output_dir: Path) -> StageIPass:
    """All 72 Switch reproductions must pass before a Stage II token exists."""
    required = {(s, r) for s in range(1, STATES + 1) for r in REPLICAS}
    if len(units) != GATE_UNITS or {unit.key for unit in units} != required:
        raise ValueError("Stage I requires the exact 72 frozen A1/A2 identities")
    if any(unit._seal is not _FROZEN_UNIT_SEAL for unit in units):
        raise PermissionError("Stage I units must come from the frozen Phase 5 ledger")
    validate_production_dataset(repo, dataset)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for unit in sorted(units, key=lambda item: item.key):
        result = None
        try:
            model_state, optimizer_state = reconstruct_epoch17(unit)
            generator_state = reconstruct_epoch18_generator(unit, dataset)
            result = run_one_epoch(unit, dataset, generator_state, model_state, optimizer_state,
                                   action="Switch")
            check_switch(unit, result)
        except Exception as exc:
            status = "INTEGRITY_GATE_FAIL" if result is not None else "TECHNICAL_FAILURE"
            _write_record(output_dir, unit, "Switch", status, result, exc)
            raise
        _write_record(output_dir, unit, "Switch", "PASS", result)
        results.append(result)
    files = sorted(output_dir.glob("switch-*.json"))
    if len(files) != GATE_UNITS:
        raise ValueError("Stage I PASS file coverage is incomplete")
    manifest = {
        "protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
        "stage": "I", "status": "72/72 PASS", "unit_count": GATE_UNITS,
        "switch_records": {path.name: sha256_file(path) for path in files},
    }
    manifest_path = output_dir / "stage_i_gate.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return StageIPass(frozenset(required), tuple(results), _GATE_SEAL,
                      manifest_path, sha256_file(manifest_path))


def resume_stage_i(units: Sequence[Unit], repo: Path, stage_i_dir: Path,
                   expected_gate_sha256: str) -> StageIPass:
    """Adopt a persisted 72/72 Switch gate after validating every frozen unit."""
    verify_protocol(repo)
    required = {(s, r) for s in range(1, STATES + 1) for r in REPLICAS}
    if (len(units) != GATE_UNITS or {unit.key for unit in units} != required or
            any(unit._seal is not _FROZEN_UNIT_SEAL for unit in units)):
        raise PermissionError("Resume requires the exact 72 frozen Phase 5 units")
    if len(expected_gate_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_gate_sha256):
        raise ValueError("Expected Stage I gate SHA-256 is invalid")
    if expected_gate_sha256 != APPROVED_STAGE_I_GATE_SHA256:
        raise PermissionError("Stage I gate SHA-256 is not the approved frozen gate")

    def read_hashed_json(path: Path, digest: str) -> dict[str, Any]:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Stage I evidence hash mismatch: {path.name}")
        def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate Stage I JSON key: {key}")
                result[key] = value
            return result
        record = json.loads(raw, object_pairs_hook=unique_keys)
        if not isinstance(record, dict):
            raise ValueError(f"Invalid Stage I JSON object: {path.name}")
        return record

    manifest_path = stage_i_dir / "stage_i_gate.json"
    manifest = read_hashed_json(manifest_path, expected_gate_sha256)
    names = {f"switch-{s:03d}-A-r{r:02d}.json" for s, r in required}
    records = manifest.get("switch_records")
    if (manifest.get("protocol_id") != PROTOCOL_ID or
            manifest.get("protocol_sha256") != PROTOCOL_SHA256 or
            manifest.get("stage") != "I" or manifest.get("status") != "72/72 PASS" or
            manifest.get("unit_count") != GATE_UNITS or
            not isinstance(records, dict) or set(records) != names or
            {path.name for path in stage_i_dir.glob("*.json")} != names | {"stage_i_gate.json"}):
        raise ValueError("Stage I gate identity, status, or record coverage mismatch")

    by_key = {unit.key: unit for unit in units}
    results = []
    for key in sorted(required):
        unit = by_key[key]
        name = f"switch-{key[0]:03d}-A-r{key[1]:02d}.json"
        digest = records[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid Stage I record SHA-256: {name}")
        record = read_hashed_json(stage_i_dir / name, digest)
        identity = {
            "base_run_id": unit.base_run_id, "replica_id": unit.replica_id,
            "source_sha256": unit.source_sha256,
            "checkpoint_sha256": unit.checkpoint_sha256,
            "future_seed": unit.future_seed,
        }
        if (record.get("protocol_id") != PROTOCOL_ID or
                record.get("protocol_sha256") != PROTOCOL_SHA256 or
                record.get("action") != "Switch" or record.get("status") != "PASS" or
                record.get("error_type") is not None or record.get("error_message") is not None or
                any(record.get(field) != value for field, value in identity.items())):
            raise ValueError(f"Stage I record identity or PASS mismatch: {name}")
        result = record.get("result")
        if not isinstance(result, dict) or any(result.get(field) != value for field, value in identity.items()):
            raise ValueError(f"Stage I result identity mismatch: {name}")
        check_switch(unit, result)
        results.append(result)
    return StageIPass(frozenset(required), tuple(results), _GATE_SEAL,
                      manifest_path, expected_gate_sha256)


def run_stage_ii(units: Sequence[Unit], dataset: Any, repo: Path,
                 gate: StageIPass, output_dir: Path) -> list[dict[str, Any]]:
    """Continue only after a sealed complete Stage I PASS."""
    required = {(s, r) for s in range(1, STATES + 1) for r in REPLICAS}
    if (not isinstance(gate, StageIPass) or gate._seal is not _GATE_SEAL or
            gate.keys != frozenset(required) or len(gate.results) != GATE_UNITS):
        raise PermissionError("Complete Stage I 72/72 PASS is required")
    if (gate.manifest_path is None or gate.manifest_sha256 is None or
            sha256_file(gate.manifest_path) != gate.manifest_sha256):
        raise PermissionError("Frozen Stage I gate manifest is missing or changed")
    manifest = json.loads(gate.manifest_path.read_text())
    records = manifest.get("switch_records")
    if (manifest.get("status") != "72/72 PASS" or manifest.get("unit_count") != GATE_UNITS or
            not isinstance(records, dict) or len(records) != GATE_UNITS or
            any(sha256_file(gate.manifest_path.parent / name) != digest
                for name, digest in records.items())):
        raise PermissionError("Stage I gate records are incomplete or changed")
    if len(units) != GATE_UNITS or {unit.key for unit in units} != required:
        raise ValueError("Stage II requires the same 72 frozen identities")
    if any(unit._seal is not _FROZEN_UNIT_SEAL for unit in units):
        raise PermissionError("Stage II units must come from the frozen Phase 5 ledger")
    validate_production_dataset(repo, dataset)
    by_key = {unit.key: unit for unit in units}
    seen = set()
    for result in gate.results:
        key = result["base_run_id"], result["replica_id"]
        if key not in required or key in seen:
            raise ValueError("Stage I results are incomplete or duplicated")
        seen.add(key)
        unit = by_key[key]
        if (result["source_sha256"] != unit.source_sha256 or
                result["checkpoint_sha256"] != unit.checkpoint_sha256 or
                result["future_seed"] != unit.future_seed):
            raise ValueError("Stage I result source identity mismatch")
        check_switch(unit, result)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for unit in sorted(units, key=lambda item: item.key):
        result = None
        try:
            # Reconstruct afresh from the historical artifact, never from Switch.
            model_state, optimizer_state = reconstruct_epoch17(unit)
            generator_state = reconstruct_epoch18_generator(unit, dataset)
            result = run_one_epoch(unit, dataset, generator_state, model_state,
                                   optimizer_state, action="Continue")
        except Exception as exc:
            _write_record(output_dir, unit, "Continue", "TECHNICAL_FAILURE", result, exc)
            raise
        _write_record(output_dir, unit, "Continue", "RECORDED", result)
        results.append(result)
    return results


def analyze_completed(
    units: Sequence[Unit], gate: StageIPass, continue_results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Frozen state-level primary and prespecified secondary analysis."""
    required = {(s, r) for s in range(1, STATES + 1) for r in REPLICAS}
    if not isinstance(gate, StageIPass) or gate._seal is not _GATE_SEAL or gate.keys != frozenset(required):
        raise PermissionError("Analysis requires complete Stage I PASS")
    if len(units) != GATE_UNITS or {unit.key for unit in units} != required:
        raise ValueError("Analysis requires the exact frozen unit population")
    if len(gate.results) != GATE_UNITS:
        raise ValueError("Analysis requires all 72 Stage I records")
    if len(continue_results) != GATE_UNITS:
        raise ValueError("Analysis requires all 72 Continue outcomes")
    by_unit = {unit.key: unit for unit in units}
    switch_keys = set()
    for row in gate.results:
        key = row["base_run_id"], row["replica_id"]
        if key not in required or key in switch_keys:
            raise ValueError("Stage I result identity/coverage mismatch")
        switch_keys.add(key)
        unit = by_unit[key]
        if (row["source_sha256"] != unit.source_sha256 or
                row["checkpoint_sha256"] != unit.checkpoint_sha256 or
                row["future_seed"] != unit.future_seed):
            raise ValueError("Stage I source identity mismatch")
        check_switch(unit, row)
    continuation = {}
    for row in continue_results:
        key = row["base_run_id"], row["replica_id"]
        if key not in required or key in continuation or row["action"] != "Continue":
            raise ValueError("Continue outcome identity/coverage mismatch")
        unit = by_unit[key]
        if (row["source_sha256"] != unit.source_sha256 or
                row["checkpoint_sha256"] != unit.checkpoint_sha256 or
                row["future_seed"] != unit.future_seed or
                row["learning_rate"] != 0.1 or row["epoch"] != 18 or
                row["realized_order"] != unit.historical["wait_realized_orders"][3] or
                row["epoch_order_sha256"] != unit.historical["wait_epoch_order_sha256"][3]):
            raise ValueError("Continue outcome integrity mismatch")
        continuation[key] = row
    if set(continuation) != required:
        raise ValueError("Continue coverage incomplete")
    state_values = np.empty((STATES, len(REPLICAS), 3), dtype=np.float64)
    secondary_values = np.empty((STATES, len(REPLICAS), 2), dtype=np.float64)
    for state in range(1, STATES + 1):
        for index, replica in enumerate(REPLICAS):
            unit = by_unit[state, replica]
            historical = unit.historical["wait_history"]
            loss17 = float(historical[2]["val_loss"])
            switch = float(historical[3]["val_loss"])
            cont = float(continuation[state, replica]["val_loss"])
            if not np.isfinite([loss17, switch, cont]).all():
                raise ValueError("Nonfinite primary mechanism outcome")
            delta = cont - switch
            d = loss17 - switch
            kappa = loss17 - cont
            if not np.isclose(d, kappa + delta, rtol=0, atol=2e-16):
                raise ArithmeticError("Unit-level catch-up decomposition failed")
            state_values[state - 1, index] = (delta, d, kappa)
            train_switch = float(historical[3]["train_loss"])
            train_continue = float(continuation[state, replica]["train_loss"])
            accuracy_switch = float(historical[3]["val_accuracy"])
            accuracy_continue = float(continuation[state, replica]["val_accuracy"])
            if not np.isfinite([train_switch, train_continue,
                                accuracy_switch, accuracy_continue]).all():
                raise ValueError("Nonfinite secondary mechanism outcome")
            secondary_values[state - 1, index] = (
                train_continue - train_switch,
                accuracy_continue - accuracy_switch,
            )
    means = state_values.mean(axis=1)
    estimate = means.mean(axis=0)
    if not np.isclose(estimate[1], estimate[2] + estimate[0], rtol=0, atol=2e-16):
        raise ArithmeticError("Aggregate catch-up decomposition failed")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draw_indices = rng.integers(0, STATES, size=(10_000, STATES))
    draws = means[draw_indices].mean(axis=1)
    ci = np.percentile(draws, [2.5, 97.5], axis=0)
    secondary_means = secondary_values.mean(axis=1)
    secondary_estimate = secondary_means.mean(axis=0)
    secondary_draws = secondary_means[draw_indices].mean(axis=1)
    secondary_ci = np.percentile(secondary_draws, [2.5, 97.5], axis=0)
    return {
        "protocol_id": PROTOCOL_ID,
        "n_states": STATES, "replicas_per_state": 2,
        "delta_switch": float(estimate[0]),
        "D": float(estimate[1]), "kappa_C": float(estimate[2]),
        "R": float(estimate[0] / estimate[1]) if estimate[1] != 0 else None,
        "ci_95": {
            "delta_switch": ci[:, 0].tolist(),
            "D": ci[:, 1].tolist(), "kappa_C": ci[:, 2].tolist(),
        },
        "secondary_outcomes": {
            "epoch18_train_loss": {
                "contrast": "Continue - Switch",
                "estimate": float(secondary_estimate[0]),
                "ci_95": secondary_ci[:, 0].tolist(),
            },
            "epoch18_val_accuracy": {
                "contrast": "Continue - Switch",
                "estimate": float(secondary_estimate[1]),
                "ci_95": secondary_ci[:, 1].tolist(),
            },
        },
        "bootstrap_replicates": 10_000,
        "bootstrap_bit_generator": "PCG64", "bootstrap_seed": BOOTSTRAP_SEED,
    }

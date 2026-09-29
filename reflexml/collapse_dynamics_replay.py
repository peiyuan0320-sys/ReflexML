"""Historical Phase 5A A1 Wait3/Now epoch18 replay with RNG-neutral probes.

Replay, endpoint gate, and offline validation are deliberately separate commands.
No historical artifact is written by this module.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .branching import order_hashes, states_equal, training_state_finiteness_evidence
from .config import ExperimentConfig
from .data import make_loaders_from_datasets
from .mechanism_lr_switch import (
    FROZEN_BRANCH_LEDGER_SHA256, Unit, _configuration, _evidence_map, check_switch,
    load_unit, reconstruct_epoch17, reconstruct_epoch18_generator,
    validate_production_dataset,
)
from .model import FashionMLP
from .phase5 import sha256_file
from .training import evaluate, train_one_epoch


PROTOCOL_ID = "collapse_dynamics_replay_v1"
PROBES = (0, 1, 3, 8, 16, 32, 48, 64, 78, 79)
LEDGER = Path("internal-artifacts/branch_attempts.jsonl")
BASE_OUTPUT = Path("internal-artifacts/phase5_v1_base_v2")
ARMS = ("Wait3", "Now")
CODE_FILES = (
    "reflexml/collapse_dynamics_replay.py", "reflexml/training.py",
    "reflexml/mechanism_lr_switch.py", "reflexml/branching.py",
    "reflexml/model.py", "reflexml/data.py", "reflexml/config.py",
    "run_collapse_dynamics_replay.py",
)


def load_a1_unit(repo: Path, state: int) -> Unit:
    """Resolve one A1 source from the audited branch ledger, without loading A2."""
    if not 1 <= state <= 36 or sha256_file(LEDGER) != FROZEN_BRANCH_LEDGER_SHA256:
        raise ValueError("State or frozen branch ledger identity mismatch")
    events = []
    with LEDGER.open() as handle:
        for line in handle:
            event = json.loads(line)
            identity = event.get("scientific_identity", {})
            if (identity.get("base_run_id"), identity.get("block"),
                    identity.get("replica_id")) == (state, "A", 1):
                events.append(event)
    if [row["event_type"] for row in events] != [
        "attempt_started", "artifact_produced", "terminal_success"
    ]:
        raise ValueError("A1 source lineage is incomplete or ambiguous")
    first, produced, terminal = events
    identity = first["scientific_identity"]
    if (first["attempt_id"] != produced["attempt_id"] or
            first["attempt_id"] != terminal["attempt_id"] or
            identity != produced["scientific_identity"] or
            identity != terminal["scientific_identity"] or
            produced["artifact_sha256"] != terminal["artifact_sha256"]):
        raise ValueError("A1 source lineage disagrees")
    primary_path = BASE_OUTPUT / f"base_{state:03d}_attempt_001/primary_state.json"
    primary = json.loads(primary_path.read_text())
    if (primary.get("base_run_id") != state or primary.get("epoch") != 14 or
            primary.get("protocol_version") != "Phase5-v1" or
            primary.get("checkpoint_sha256") != identity["checkpoint_sha256"] or
            primary.get("state_manifested_before_future_outcomes") is not True):
        raise ValueError("A1 source primary state mismatch")
    return load_unit(
        repo, base_run_id=state, replica_id=1,
        checkpoint_path=Path(primary["checkpoint_path"]),
        checkpoint_sha256=identity["checkpoint_sha256"],
        artifact_path=Path(produced["artifact_path"]),
        artifact_sha256=produced["artifact_sha256"],
    )


def _source_state(unit: Unit, arm: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if arm not in ARMS or unit.replica_id != 1:
        raise ValueError("Only historical A1 Wait3/Now is permitted")
    wait_model, wait_optimizer = reconstruct_epoch17(unit)
    if arm == "Wait3":
        return wait_model, wait_optimizer
    history = unit.historical["now_history"]
    if ([row["epoch"] for row in history] != [15, 16, 17, 18] or
            [row["learning_rate"] for row in history] != [0.05] * 4):
        raise ValueError("Historical Now schedule mismatch")
    row = unit.historical["now_state_finiteness_evidence"][2]
    if row["epoch"] != 17:
        raise ValueError("Historical Now S17 evidence missing")
    model_values = _evidence_map(row["model_parameters"])
    momentum_values = _evidence_map(row["optimizer_state"])
    names = list(wait_model)
    ids = wait_optimizer["param_groups"][0]["params"]
    if (set(model_values) != {f"model.{name}" for name in names} or
            set(momentum_values) != {
                f"optimizer.state.{identifier}.momentum_buffer" for identifier in ids
            }):
        raise ValueError("Historical Now S17 tensor inventory mismatch")
    model = {}
    optimizer = deepcopy(wait_optimizer)
    optimizer["param_groups"][0]["lr"] = 0.05
    for identifier, name in zip(ids, names, strict=True):
        value = model_values[f"model.{name}"]
        buffer = momentum_values[f"optimizer.state.{identifier}.momentum_buffer"]
        if (value.shape != wait_model[name].shape or value.dtype != wait_model[name].dtype or
                buffer.shape != value.shape or buffer.dtype != value.dtype):
            raise ValueError("Historical Now S17 tensor shape/dtype mismatch")
        model[name] = value.clone()
        optimizer["state"][identifier]["momentum_buffer"] = buffer.clone()
    return model, optimizer


def _expected_order(unit: Unit) -> list[list[int]]:
    history = unit.historical
    wait = history["wait_realized_orders"][3]
    now = history["now_realized_orders"][3]
    if (wait != now or [len(batch) for batch in wait] != [128] * 78 + [16] or
            order_hashes([wait])[0] != history["wait_epoch_order_sha256"][3] or
            order_hashes([now])[0] != history["now_epoch_order_sha256"][3] or
            history["wait_epoch_order_sha256"] !=
            history["now_epoch_order_sha256"]):
        raise ValueError("Historical epoch18 order or 79 batch boundaries mismatch")
    return wait


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    # detach/clone reads state without changing model mode, optimizer, or RNG.
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _write_json(path: Path, row: dict[str, Any]) -> None:
    path.write_text(json.dumps(row, sort_keys=True, indent=2, allow_nan=False) + "\n")


def replay(repo: Path, dataset: Any, *, state: int, arm: str,
           output: Path, git_commit: str) -> Path:
    if arm not in ARMS or not git_commit:
        raise ValueError("Arm and code commit are required")
    if output.exists():
        raise FileExistsError(f"Replay output already exists: {output}")
    unit = load_a1_unit(repo, state)
    validate_production_dataset(repo, dataset)
    config = _configuration(unit.checkpoint)
    if config.val_size != 5000 or config.batch_size != 128:
        raise ValueError("Frozen validation population or batch size changed")
    model_state, optimizer_state = _source_state(unit, arm)
    generator_state = reconstruct_epoch18_generator(unit, dataset)
    expected_order = _expected_order(unit)
    with torch.random.fork_rng(devices=[]):
        model = FashionMLP(config.hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                momentum=config.momentum)
    model.load_state_dict(deepcopy(model_state), strict=True)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    optimizer.param_groups[0]["lr"] = 0.05
    if any(group["lr"] != 0.05 for group in optimizer.param_groups):
        raise AssertionError("Historical epoch18 LR assignment failed")
    data = make_loaders_from_datasets(dataset, [], config, audit_order=True)
    if data.split_fingerprint != unit.checkpoint["split_fingerprint"]:
        raise ValueError("Frozen split fingerprint mismatch")
    data.train_generator.set_state(generator_state.clone())
    snapshots = {0: _snapshot(model)}
    optimizer_t0 = deepcopy(optimizer.state_dict())

    def capture(step: int, current: nn.Module) -> None:
        if step in PROBES:
            snapshots[step] = _snapshot(current)

    order: list[list[int]] = []
    loss = train_one_epoch(model, data.train_loader, optimizer, nn.CrossEntropyLoss(),
                           torch.device("cpu"), order, after_step=capture)
    if (order != expected_order or set(snapshots) != set(PROBES) or
            not states_equal(snapshots[79], model.state_dict())):
        raise ValueError("Replay order, update count, or final snapshot mismatch")
    output.mkdir(parents=True)
    files = {}
    for step in PROBES:
        name = f"model_t{step:02d}.pt"
        torch.save(snapshots[step], output / name)
        files[name] = sha256_file(output / name)
    torch.save(optimizer_t0, output / "optimizer_t00.pt")
    files["optimizer_t00.pt"] = sha256_file(output / "optimizer_t00.pt")
    torch.save(optimizer.state_dict(), output / "optimizer_t79.pt")
    files["optimizer_t79.pt"] = sha256_file(output / "optimizer_t79.pt")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "replay_identity": f"{PROTOCOL_ID}:state={state}:A1:{arm}:{output.resolve()}",
        "source_scientific_identity": unit.historical["scientific_identity"],
        "source_artifact_sha256": unit.source_sha256,
        "source_checkpoint_sha256": unit.checkpoint_sha256,
        "base_run_id": state, "replica": "A1", "arm": arm,
        "future_seed": unit.future_seed,
        "future_stream_identity": unit.historical["scientific_identity"]["future_stream_identity"],
        "epoch18_order_sha256": order_hashes([order])[0],
        "probe_steps": list(PROBES), "optimizer_updates": len(order),
        "train_loss": loss, "git_commit": git_commit,
        "implementation_sha256": {name: sha256_file(repo / name) for name in CODE_FILES},
        "files": files,
    }
    _write_json(output / "replay_manifest.json", manifest)
    return output / "replay_manifest.json"


def _read_replay(repo: Path, output: Path) -> tuple[Unit, dict[str, Any]]:
    manifest_path = output / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("protocol_id") != PROTOCOL_ID or
            manifest.get("replica") != "A1" or manifest.get("arm") not in ARMS or
            manifest.get("probe_steps") != list(PROBES) or
            manifest.get("optimizer_updates") != 79 or
            manifest.get("replay_identity") != f"{PROTOCOL_ID}:state={manifest.get('base_run_id')}:A1:{manifest.get('arm')}:{output.resolve()}"):
        raise ValueError("Replay manifest identity or design mismatch")
    if set(manifest["implementation_sha256"]) != set(CODE_FILES):
        raise ValueError("Replay implementation inventory mismatch")
    for name, expected_hash in manifest["implementation_sha256"].items():
        if sha256_file(repo / name) != expected_hash:
            raise ValueError("Replay implementation bytes changed since execution")
    unit = load_a1_unit(repo, manifest["base_run_id"])
    if (manifest["source_scientific_identity"] != unit.historical["scientific_identity"] or
            manifest["source_artifact_sha256"] != unit.source_sha256 or
            manifest["source_checkpoint_sha256"] != unit.checkpoint_sha256 or
            manifest["future_seed"] != unit.future_seed or
            manifest["future_stream_identity"] != unit.historical["scientific_identity"]["future_stream_identity"] or
            manifest["epoch18_order_sha256"] != order_hashes([_expected_order(unit)])[0]):
        raise ValueError("Replay source mapping mismatch")
    expected_files = {f"model_t{step:02d}.pt" for step in PROBES} | {
        "optimizer_t00.pt", "optimizer_t79.pt"
    }
    if set(manifest["files"]) != expected_files:
        raise ValueError("Replay file inventory mismatch")
    for name, expected_hash in manifest["files"].items():
        if sha256_file(output / name) != expected_hash:
            raise ValueError(f"Replay file hash mismatch: {name}")
    return unit, manifest


def _load_model(config: ExperimentConfig, path: Path) -> FashionMLP:
    with torch.random.fork_rng(devices=[]):
        model = FashionMLP(config.hidden_size)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model


def gate(repo: Path, dataset: Any, output: Path) -> Path:
    unit, manifest = _read_replay(repo, output)
    validate_production_dataset(repo, dataset)
    arm = manifest["arm"]
    config = _configuration(unit.checkpoint)
    source_model, source_optimizer = _source_state(unit, arm)
    start = torch.load(output / "model_t00.pt", map_location="cpu", weights_only=True)
    if not states_equal(start, source_model):
        raise ValueError("Replay t=0 model differs from historical S17")
    source_optimizer["param_groups"][0]["lr"] = 0.05
    start_optimizer = torch.load(output / "optimizer_t00.pt", map_location="cpu", weights_only=True)
    if not states_equal(start_optimizer, source_optimizer):
        raise ValueError("Replay t=0 optimizer differs from historical S17")
    model = _load_model(config, output / "model_t79.pt")
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                momentum=config.momentum)
    optimizer.load_state_dict(torch.load(output / "optimizer_t79.pt", map_location="cpu", weights_only=True))
    expected_group = deepcopy(unit.checkpoint["optimizer_state_dict"]["param_groups"])
    expected_group[0]["lr"] = 0.05
    if optimizer.state_dict()["param_groups"] != expected_group:
        raise ValueError("Replay endpoint optimizer metadata mismatch")
    data = make_loaders_from_datasets(dataset, [], config, audit_order=True)
    if data.split_fingerprint != unit.checkpoint["split_fingerprint"] or len(data.val_loader.dataset) != 5000:
        raise ValueError("Frozen full validation set mismatch")
    result = evaluate(model, data.val_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    historical = unit.historical
    prefix = "wait" if arm == "Wait3" else "now"
    expected = historical[f"{prefix}_history"][3]
    state_evidence = training_state_finiteness_evidence(model, optimizer)
    if arm == "Wait3":
        check_switch(unit, {
            "base_run_id": unit.base_run_id, "replica_id": 1,
            "action": "Switch", "epoch": 18, "learning_rate": 0.05,
            "realized_order": _expected_order(unit),
            "epoch_order_sha256": manifest["epoch18_order_sha256"],
            "train_loss": manifest["train_loss"], "val_loss": result.loss,
            "val_accuracy": result.accuracy, "state_evidence": state_evidence,
        })
    checks = {
        "train_loss": manifest["train_loss"] == expected["train_loss"],
        "val_loss": result.loss == expected["val_loss"],
        "val_accuracy": result.accuracy == expected["val_accuracy"],
        "learning_rate": expected["learning_rate"] == 0.05,
        "model_and_optimizer_tensors": state_evidence == {
            "model_parameters": historical[f"{prefix}_state_finiteness_evidence"][3]["model_parameters"],
            "optimizer_state": historical[f"{prefix}_state_finiteness_evidence"][3]["optimizer_state"],
        },
    }
    if not all(checks.values()):
        raise ValueError(f"Historical endpoint exactness FAIL: {checks}")
    gate_path = output / "endpoint_gate.json"
    if gate_path.exists():
        raise FileExistsError(f"Endpoint gate already exists: {gate_path}")
    _write_json(gate_path, {
        "protocol_id": PROTOCOL_ID, "verdict": "PASS", "checks": checks,
        "replay_manifest_sha256": sha256_file(output / "replay_manifest.json"),
        "source_artifact_sha256": unit.source_sha256,
        "base_run_id": unit.base_run_id, "replica": "A1", "arm": arm,
    })
    return gate_path


def _require_gate(repo: Path, dataset: Any, output: Path) -> tuple[Unit, dict[str, Any]]:
    unit, manifest = _read_replay(repo, output)
    gate_row = json.loads((output / "endpoint_gate.json").read_text())
    if (gate_row.get("verdict") != "PASS" or
            gate_row.get("replay_manifest_sha256") != sha256_file(output / "replay_manifest.json") or
            gate_row.get("source_artifact_sha256") != unit.source_sha256 or
            gate_row.get("base_run_id") != unit.base_run_id or
            gate_row.get("replica") != "A1" or gate_row.get("arm") != manifest["arm"] or
            set(gate_row.get("checks", {})) != {
                "train_loss", "val_loss", "val_accuracy", "learning_rate",
                "model_and_optimizer_tensors"
            } or not all(gate_row["checks"].values())):
        raise ValueError("Replay lacks a matching exact endpoint PASS")
    validate_production_dataset(repo, dataset)
    return unit, manifest


def offline_report(repo: Path, dataset: Any, wait_dir: Path, now_dir: Path,
                   report_dir: Path) -> Path:
    wait_unit, wait = _require_gate(repo, dataset, wait_dir)
    now_unit, now = _require_gate(repo, dataset, now_dir)
    if (wait["arm"] != "Wait3" or now["arm"] != "Now" or
            wait_unit.key != now_unit.key or
            wait["future_stream_identity"] != now["future_stream_identity"]):
        raise ValueError("Offline report requires paired A1 Wait3/Now")
    config = _configuration(wait_unit.checkpoint)
    data = make_loaders_from_datasets(dataset, [], config, audit_order=True)
    if data.split_fingerprint != wait_unit.checkpoint["split_fingerprint"] or len(data.val_loader.dataset) != 5000:
        raise ValueError("Frozen full validation set mismatch")
    if report_dir.exists():
        raise FileExistsError(f"Report output already exists: {report_dir}")
    rows = []
    for arm, directory, manifest in (("Wait3", wait_dir, wait), ("Now", now_dir, now)):
        for step in PROBES:
            model = _load_model(config, directory / f"model_t{step:02d}.pt")
            result = evaluate(model, data.val_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
            rows.append({
                "base_run_id": wait_unit.base_run_id, "replica": "A1", "arm": arm,
                "probe_step": step, "validation_loss": result.loss, "gate_pass": True,
                "source_artifact_sha256": manifest["source_artifact_sha256"],
                "future_stream_identity": manifest["future_stream_identity"],
            })
    report_dir.mkdir(parents=True)
    path = report_dir / "trajectory.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(report_dir / "report_manifest.json", {
        "protocol_id": PROTOCOL_ID, "scope": "historical_A1_per_state_trajectory",
        "covered_states": [wait_unit.base_run_id], "probe_steps": list(PROBES),
        "wait_gate_sha256": sha256_file(wait_dir / "endpoint_gate.json"),
        "now_gate_sha256": sha256_file(now_dir / "endpoint_gate.json"),
        "trajectory_sha256": sha256_file(path),
    })
    return path


def summarize_a1(report_root: Path, output: Path) -> Path:
    """Descriptive means across exactly 36 paired A1 states; no resampling."""
    if output.exists():
        raise FileExistsError(f"Summary output already exists: {output}")
    losses: dict[tuple[int, str, int], float] = {}
    for state in range(1, 37):
        folder = report_root / f"state_{state:03d}"
        manifest = json.loads((folder / "report_manifest.json").read_text())
        path = folder / "trajectory.csv"
        if (manifest.get("protocol_id") != PROTOCOL_ID or
                manifest.get("covered_states") != [state] or
                manifest.get("probe_steps") != list(PROBES) or
                manifest.get("trajectory_sha256") != sha256_file(path)):
            raise ValueError(f"A1 report identity/hash mismatch: state {state}")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2 * len(PROBES):
            raise ValueError(f"A1 report coverage mismatch: state {state}")
        for row in rows:
            key = (int(row["base_run_id"]), row["arm"], int(row["probe_step"]))
            if (key[0] != state or row["replica"] != "A1" or
                    row["gate_pass"] != "True" or key[1] not in ARMS or
                    key[2] not in PROBES or key in losses):
                raise ValueError(f"A1 report row identity mismatch: state {state}")
            value = float(row["validation_loss"])
            if not math.isfinite(value):
                raise ValueError(f"A1 report contains nonfinite validation loss: state {state}")
            losses[key] = value
    means = {}
    for step in PROBES:
        wait = sum(losses[state, "Wait3", step] for state in range(1, 37)) / 36
        now = sum(losses[state, "Now", step] for state in range(1, 37)) / 36
        gap = sum(losses[state, "Wait3", step] - losses[state, "Now", step]
                  for state in range(1, 37)) / 36
        means[step] = (wait, now, gap)
    denominator = means[0][2] - means[79][2]
    if denominator == 0:
        raise ValueError("F_t is undefined because endpoint gap closure is zero")
    output.mkdir(parents=True)
    path = output / "historical_a1_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("probe_step", "W_t_A1", "N_t_A1", "C_t_A1", "F_t"))
        for step in PROBES:
            wait, now, gap = means[step]
            writer.writerow((step, wait, now, gap, (means[0][2] - gap) / denominator))
    return path

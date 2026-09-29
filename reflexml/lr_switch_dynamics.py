"""LR-Switch Dynamics v1: A1 Switch/Continue replay with shared sparse probes.

Replay, endpoint gate, and offline validation are deliberately separate commands.
No historical artifact is written by this module.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .branching import order_hashes, states_equal, training_state_finiteness_evidence
from .data import make_loaders_from_datasets
from .mechanism_lr_switch import (
    FROZEN_BRANCH_LEDGER_SHA256, Unit, _configuration, check_switch, reconstruct_epoch17, reconstruct_epoch18_generator,
    validate_production_dataset,
)
from .model import FashionMLP
from .phase5 import sha256_file
from .training import evaluate, train_one_epoch


# Reuse the validated Collapse Dynamics snapshot primitive, grid, and source reader.
from .collapse_dynamics_replay import (
    PROBES, load_a1_unit, _expected_order, _snapshot, _load_model,
)
PROTOCOL_ID = "ReflexML-LR-Switch-Dynamics-v1"
ARMS = ("Switch", "Continue")
CHECKS = {"train_loss", "val_loss", "val_accuracy", "learning_rate",
          "model_and_optimizer_tensors", "source_model", "source_optimizer",
          "endpoint_optimizer_metadata", "exact_order_and_79_updates"}
STAGE_I = Path("internal-artifacts/reflexml-mechanism-lr-switch-v1-stage-i-20260926-001")
STAGE_II = Path("internal-artifacts/reflexml-mechanism-lr-switch-v1-stage-ii-20260926-001")
STAGE_I_SHA = "dd7d4273a215f1c71cdf816bce3334dc846c37e31b92f0e29e46b7748d67dc80"
STAGE_II_SHA = "b4db6b50f176c556ec6ec31182831bbe2a308f174dca499bb44e1503c0e5d473"
CODE_FILES = (
    "reflexml/lr_switch_dynamics.py", "reflexml/collapse_dynamics_replay.py",
    "reflexml/training.py", "reflexml/mechanism_lr_switch.py",
    "reflexml/branching.py", "reflexml/model.py", "reflexml/data.py",
    "reflexml/config.py", "reflexml/checkpoint.py", "reflexml/phase5.py",
    "run_lr_switch_dynamics.py", "run_lr_switch_dynamics_production.py",
    "LR_SWITCH_DYNAMICS_PROTOCOL.md",
)


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def learning_rate(arm: str) -> float:
    if arm not in ARMS:
        raise ValueError("Only Switch/.05 and Continue/.10 are allowed")
    return 0.05 if arm == "Switch" else 0.10


def historical_endpoint(unit: Unit, arm: str) -> tuple[dict, dict]:
    """Read exactly one A1 cell from hash-bound historical authority."""
    learning_rate(arm)
    authority = []
    for root, name, digest in (
        (STAGE_I, "stage_i_gate.json", STAGE_I_SHA),
        (STAGE_II, "stage_ii_completion_manifest.json", STAGE_II_SHA),
    ):
        if sha256_file(root / name) != digest:
            raise ValueError("Historical LR-Switch authority hash mismatch")
        authority.append(json.loads((root / name).read_text()))
    low, high = authority
    if (low["status"] != "72/72 PASS" or
            high["status"] != "72/72 COMPLETE — INTEGRITY VERIFIED" or
            high["stage_i_gate_sha256"] != STAGE_I_SHA or
            high["frozen_branch_ledger_sha256"] != FROZEN_BRANCH_LEDGER_SHA256):
        raise ValueError("Historical LR-Switch authority chain mismatch")
    for records, action in ((low["switch_records"], "switch"),
                            (high["continue_records"], "continue")):
        if set(records) != {f"{action}-{s:03d}-A-r{r:02d}.json"
                            for s in range(1, 37) for r in (1, 2)}:
            raise ValueError("Historical endpoint roster mismatch")
    root = STAGE_I if arm == "Switch" else STAGE_II
    records = low["switch_records"] if arm == "Switch" else high["continue_records"]
    name = f"{arm.lower()}-{unit.base_run_id:03d}-A-r01.json"
    path = root / name
    if sha256_file(path) != records[name]:
        raise ValueError("Historical endpoint record hash mismatch")
    row = json.loads(path.read_text())
    identity = {"base_run_id": unit.base_run_id, "replica_id": 1,
                "source_sha256": unit.source_sha256,
                "checkpoint_sha256": unit.checkpoint_sha256,
                "future_seed": unit.future_seed, "action": arm}
    result = row["result"]
    if (any(row.get(k) != v or result.get(k) != v for k, v in identity.items()) or
            row["status"] != ("PASS" if arm == "Switch" else "RECORDED") or
            row.get("error_type") is not None or row.get("error_message") is not None or
            result["epoch"] != 18 or result["learning_rate"] != learning_rate(arm) or
            result["realized_order"] != _expected_order(unit) or
            result["epoch_order_sha256"] != order_hashes([_expected_order(unit)])[0]):
        raise ValueError("Historical endpoint source/stream/LR identity mismatch")
    if arm == "Switch":
        check_switch(unit, result)
    return result, {"record_path": str(path), "record_sha256": records[name],
                    "stage_i_gate_sha256": STAGE_I_SHA,
                    "stage_ii_manifest_sha256": STAGE_II_SHA}


def _write_json(path: Path, row: dict[str, Any]) -> None:
    path.write_text(json.dumps(row, sort_keys=True, indent=2, allow_nan=False) + "\n")


def replay(repo: Path, dataset: Any, *, state: int, arm: str,
           output: Path, git_commit: str) -> Path:
    if arm not in ARMS or not git_commit:
        raise ValueError("Arm and code commit are required")
    if output.exists():
        raise FileExistsError(f"Replay output already exists: {output}")
    code_hashes = {name: sha256_file(repo / name) for name in CODE_FILES}
    runtime = runtime_identity()
    unit = load_a1_unit(repo, state)
    validate_production_dataset(repo, dataset)
    config = _configuration(unit.checkpoint)
    if config.val_size != 5000 or config.batch_size != 128:
        raise ValueError("Frozen validation population or batch size changed")
    _, endpoint_binding = historical_endpoint(unit, arm)
    model_state, optimizer_state = reconstruct_epoch17(unit)
    generator_state = reconstruct_epoch18_generator(unit, dataset)
    expected_order = _expected_order(unit)
    with torch.random.fork_rng(devices=[]):
        model = FashionMLP(config.hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                momentum=config.momentum)
    model.load_state_dict(deepcopy(model_state), strict=True)
    optimizer.load_state_dict(deepcopy(optimizer_state))
    optimizer.param_groups[0]["lr"] = learning_rate(arm)
    if any(group["lr"] != learning_rate(arm) for group in optimizer.param_groups):
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
    if code_hashes != {name: sha256_file(repo / name) for name in CODE_FILES} or runtime != runtime_identity():
        raise ValueError("Implementation or runtime changed during replay")
    output.mkdir(parents=True, exist_ok=False)
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
        "endpoint_binding": endpoint_binding, "runtime": runtime,
        "implementation_sha256": code_hashes,
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
    _, binding = historical_endpoint(unit, manifest["arm"])
    if manifest.get("endpoint_binding") != binding or manifest.get("runtime") != runtime_identity():
        raise ValueError("Historical endpoint binding or runtime changed")
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


def gate(repo: Path, dataset: Any, output: Path) -> Path:
    unit, manifest = _read_replay(repo, output)
    validate_production_dataset(repo, dataset)
    arm = manifest["arm"]
    config = _configuration(unit.checkpoint)
    source_model, source_optimizer = reconstruct_epoch17(unit)
    start = torch.load(output / "model_t00.pt", map_location="cpu", weights_only=True)
    if (not states_equal(start, source_model) or any(
            start[name].dtype != source_model[name].dtype for name in source_model)):
        raise ValueError("Replay t=0 model differs from historical S17")
    source_optimizer["param_groups"][0]["lr"] = learning_rate(arm)
    start_optimizer = torch.load(output / "optimizer_t00.pt", map_location="cpu", weights_only=True)
    if (not states_equal(start_optimizer, source_optimizer) or any(
            start_optimizer["state"][identifier]["momentum_buffer"].dtype !=
            values["momentum_buffer"].dtype
            for identifier, values in source_optimizer["state"].items())):
        raise ValueError("Replay t=0 optimizer differs from historical S17")
    endpoint_model = torch.load(output / "model_t79.pt", map_location="cpu", weights_only=True)
    if (set(endpoint_model) != set(source_model) or any(
            endpoint_model[name].dtype != source_model[name].dtype or
            endpoint_model[name].shape != source_model[name].shape
            for name in source_model)):
        raise ValueError("Endpoint model inventory/shape/dtype mismatch")
    model = _load_model(config, output / "model_t79.pt")
    if not states_equal(endpoint_model, model.state_dict()):
        raise ValueError("Endpoint model load changed tensor values")
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                momentum=config.momentum)
    endpoint_optimizer = torch.load(output / "optimizer_t79.pt", map_location="cpu", weights_only=True)
    if (set(endpoint_optimizer) != {"state", "param_groups"} or
            set(endpoint_optimizer["state"]) != set(source_optimizer["state"]) or any(
                set(values) != {"momentum_buffer"} or
                values["momentum_buffer"].dtype != source_optimizer["state"][identifier]["momentum_buffer"].dtype or
                values["momentum_buffer"].shape != source_optimizer["state"][identifier]["momentum_buffer"].shape
                for identifier, values in endpoint_optimizer["state"].items())):
        raise ValueError("Endpoint complete optimizer inventory/shape/dtype mismatch")
    optimizer.load_state_dict(endpoint_optimizer)
    if not states_equal(endpoint_optimizer, optimizer.state_dict()):
        raise ValueError("Endpoint optimizer load changed complete state")
    expected_group = deepcopy(unit.checkpoint["optimizer_state_dict"]["param_groups"])
    expected_group[0]["lr"] = learning_rate(arm)
    data = make_loaders_from_datasets(dataset, [], config, audit_order=True)
    if data.split_fingerprint != unit.checkpoint["split_fingerprint"] or len(data.val_loader.dataset) != 5000:
        raise ValueError("Frozen full validation set mismatch")
    result = evaluate(model, data.val_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    expected, _ = historical_endpoint(unit, arm)
    state_evidence = training_state_finiteness_evidence(model, optimizer)
    actual = {**expected, "train_loss": manifest["train_loss"], "val_loss": result.loss,
              "val_accuracy": result.accuracy, "state_evidence": state_evidence}
    if arm == "Switch":
        check_switch(unit, actual)
    checks = {
        "train_loss": manifest["train_loss"] == expected["train_loss"],
        "val_loss": result.loss == expected["val_loss"],
        "val_accuracy": result.accuracy == expected["val_accuracy"],
        "learning_rate": expected["learning_rate"] == learning_rate(arm),
        "model_and_optimizer_tensors": state_evidence == expected["state_evidence"],
        "source_model": states_equal(start, source_model),
        "source_optimizer": states_equal(start_optimizer, source_optimizer),
        "endpoint_optimizer_metadata": optimizer.state_dict()["param_groups"] == expected_group,
        "exact_order_and_79_updates": manifest["optimizer_updates"] == 79 and
            manifest["epoch18_order_sha256"] == expected["epoch_order_sha256"],
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
            gate_row.get("protocol_id") != PROTOCOL_ID or
            set(gate_row.get("checks", {})) != CHECKS or
            any(v is not True for v in gate_row["checks"].values())):
        raise ValueError("Replay lacks a matching exact endpoint PASS")
    validate_production_dataset(repo, dataset)
    return unit, manifest


def offline_report(repo: Path, dataset: Any, switch_dir: Path, continue_dir: Path,
                   report_dir: Path) -> Path:
    switch_unit, low = _require_gate(repo, dataset, switch_dir)
    continue_unit, high = _require_gate(repo, dataset, continue_dir)
    if (low["arm"] != "Switch" or high["arm"] != "Continue" or
            switch_unit.key != continue_unit.key or
            low["future_stream_identity"] != high["future_stream_identity"]):
        raise ValueError("Offline report requires paired A1 Switch/Continue")
    if not states_equal(
        torch.load(switch_dir / "model_t00.pt", weights_only=True),
        torch.load(continue_dir / "model_t00.pt", weights_only=True),
    ):
        raise ValueError("Paired t0 model differs")
    config = _configuration(switch_unit.checkpoint)
    data = make_loaders_from_datasets(dataset, [], config, audit_order=True)
    if data.split_fingerprint != switch_unit.checkpoint["split_fingerprint"] or len(data.val_loader.dataset) != 5000:
        raise ValueError("Frozen full validation set mismatch")
    if report_dir.exists():
        raise FileExistsError(f"Report output already exists: {report_dir}")
    rows = []
    for arm, directory, manifest in (("Switch", switch_dir, low), ("Continue", continue_dir, high)):
        for step in PROBES:
            model = _load_model(config, directory / f"model_t{step:02d}.pt")
            result = evaluate(model, data.val_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
            rows.append({
                "base_run_id": switch_unit.base_run_id, "replica": "A1", "arm": arm,
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
        "protocol_id": PROTOCOL_ID, "scope": "LR_switch_A1_per_state_trajectory",
        "covered_states": [switch_unit.base_run_id], "probe_steps": list(PROBES),
        "switch_gate_sha256": sha256_file(switch_dir / "endpoint_gate.json"),
        "continue_gate_sha256": sha256_file(continue_dir / "endpoint_gate.json"),
        "trajectory_sha256": sha256_file(path),
    })
    return path


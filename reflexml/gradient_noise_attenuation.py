"""Frozen gradient-noise m=4 scientific kernel; production remains unauthorized."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from torch import nn

from .branching import order_hashes
from .config import ExperimentConfig
from .data import make_loaders_from_datasets, make_split_indices
from .model import FashionMLP
from .momentum_reset import _verified_attempt, prepare_reset, reset_momentum_buffers
from .phase5 import sha256_file, validate_dataset_against_registration
from .training import evaluate


PROTOCOL_ID = "ReflexML-Gradient-Noise-Attenuation-v1"
PROTOCOL_SHA256 = "14561bb89471edbbd76b7b528848468f73603db14c72bf758db3113e5f4a99f6"
FREEZE_SHA256 = "baf4d435bed55a9afcf6df1d7f8308e0e8389e745343f420e65090d9bf265a22"
MOMENTUM_CLOSURE_SHA256 = "495fa3363d1631e5efdfea3dfee058bcad6dab4af313d9de64a9531666ad72fb"
DATASET_SHA256 = "18fcff67679cf832d9398aa8251826b57470d5554de9180ace1fda733a923898"
TRAIN_INDICES_SHA256 = "de2fe7734daf05c6820f4906ad4d561661bb952a99549bfcf6c4a61d3bdd113d"
HISTORICAL_MANIFEST_SHA256 = "69260f4fbad9b2db434520949edb2ab1a6464fdc1e0eeaf1d6e554a15b521efa"
MASTER_SEED = bytes.fromhex("6a87c34e90f1b52d46e0a9c78d35f12b8e441096c2d7a5ef39b0684d21c59f73")
PREFIX = b"ReflexML-Gradient-Noise-Attenuation-v1/stream/"
HISTORICAL_DIRECTORY = Path("internal-artifacts/reflexml-momentum-reset-v1-production-20260927-001")
OLD_AUDITED_IMPLEMENTATION_SHA256 = "5c4ff0b5739bbe7604426ad92d29a7089d61211cbc4dfdc8976fe2b13e2a4774"
OLD_KERNEL_CANDIDATE_SHA256 = "3a6bbd857053f35ed59c9c1dbb4b831535aa6e09e088daf6811b00b7917eb2e4"
V2_CANDIDATE_PATH = Path(__file__).resolve().parents[1] / "GRADIENT_NOISE_ATTENUATION_KERNEL_CANDIDATE_V2.json"
V2_CANDIDATE_SHA256 = "2e4c315c29a5381662ffc1dad062de7db745c62c868e7d3dd8e3e2de88ac4b89"
V3_CANDIDATE_PATH = Path(__file__).resolve().parents[1] / "GRADIENT_NOISE_ATTENUATION_KERNEL_CANDIDATE_V3.json"
V3_CANDIDATE_SHA256 = "cab002defc4a3e57c8c95c3ab6fcdfe0e07f9acb007fb3af835fe495eb08b285"
V4_CANDIDATE_PATH = Path(__file__).resolve().parents[1] / "GRADIENT_NOISE_ATTENUATION_KERNEL_CANDIDATE_V4.json"
EXECUTION_AUTHORIZATION_PATH = Path(__file__).resolve().parents[1] / "GRADIENT_NOISE_ATTENUATION_EXECUTION_AUTHORIZATION.json"
PRODUCTION_ROOT = Path("internal-artifacts/reflexml-gradient-noise-v1-production-20260927-001")
M1_ADMISSION_SHA256 = "5691e4637496bf5906c8d5b9927811947eb31cc9d3a4d65a0126289ba6b2a916"
STREAM_MANIFEST_SHA256 = "956c88bd515651c2fd84279ae99a5188bd4a7f7b8c3260f2fc1d6f789cd591ae"
COMPATIBILITY_SHA256 = "fbd8e2ae5f15b02b0f93455a50610bcf114e549e4c5159ef10dc1c2850df33d7"


def verify_authorities(repo: Path) -> dict[str, Any]:
    expected = {
        "GRADIENT_NOISE_ATTENUATION_PROTOCOL.md": PROTOCOL_SHA256,
        "GRADIENT_NOISE_ATTENUATION_PROTOCOL_FREEZE_RECORD.md": FREEZE_SHA256,
        "MOMENTUM_RESET_MECHANISM_CLOSURE.md": MOMENTUM_CLOSURE_SHA256,
        "phase5_preflight_r4/dataset_identity.json": DATASET_SHA256,
    }
    for name, digest in expected.items():
        if sha256_file(repo / name) != digest:
            raise ValueError(f"Frozen authority changed: {name}")
    registration = json.loads((repo / "phase5_preflight_r4/dataset_identity.json").read_text())
    if registration["train_indices_sha256"] != TRAIN_INDICES_SHA256:
        raise ValueError("Ordered training index identity changed")
    return registration


def canonical_train_indices(repo: Path, dataset: Any) -> list[int]:
    registration = verify_authorities(repo)
    validate_dataset_against_registration(dataset, registration, canonical_registration=True)
    ordered, _ = make_split_indices(60000, 10000, 5000, 2026)
    loader = make_loaders_from_datasets(dataset, [], ExperimentConfig(), audit_order=True)
    return admit_train_mapping(ordered, loader.train_loader.dataset.indices, registration)


def admit_train_mapping(expected: Sequence[int], loader_mapping: Sequence[int],
                        registration: dict[str, Any]) -> list[int]:
    """Check the ordered local-to-global mapping, including the historical int64 hash."""
    if len(expected) != 10000 or list(loader_mapping) != list(expected):
        raise ValueError("Loader local-to-global mapping differs in order")
    digest = hashlib.sha256(torch.tensor(expected, dtype=torch.int64).numpy().tobytes()).hexdigest()
    if digest != TRAIN_INDICES_SHA256 or digest != registration["train_indices_sha256"]:
        raise ValueError("Ordered T hash mismatch")
    if len(set(expected)) != 10000:
        raise ValueError("T has duplicate examples")
    return list(expected)


def historical_a1_cell(state: int, lr: float, directory: Path = HISTORICAL_DIRECTORY) -> dict[str, Any]:
    """Read and admit one original Reset A1 cell; never use replay or pooled summaries."""
    if not 1 <= state <= 36 or lr not in (0.10, 0.05):
        raise ValueError("A1 state or LR outside protocol")
    manifest_path = directory / "momentum_reset_completion_manifest.json"
    if sha256_file(manifest_path) != HISTORICAL_MANIFEST_SHA256:
        raise ValueError("Historical manifest changed")
    manifest = json.loads(manifest_path.read_text())
    treatment = "L_10R" if lr == 0.10 else "L_05R"
    key = f"{state:03d}-A-r01-{treatment}"
    branch = manifest["branches"][key]
    attempts = branch["attempts"]
    if (branch["disposition"] != "scientifically_valid_primary_evaluable" or
            branch["unresolved_prestep"] is not None or len(attempts) != 1 or
            branch["authoritative_attempt"] != attempts[0]["attempt_id"] or
            attempts[0]["retry_of"] is not None):
        raise ValueError("Historical A1 attempt lineage not uniquely admitted")
    attempt = attempts[0]
    path = directory / attempt["filename"]
    if path.parent != directory or sha256_file(path) != attempt["file_sha256"]:
        raise ValueError("Historical A1 file identity mismatch")
    record = _verified_attempt(path)
    if (record["record_sha256"] != attempt["record_sha256"] or
            record["attempt_id"] != attempt["attempt_id"] or
            record["classification"] != branch["disposition"] or
            not record["authoritative_lifecycle"] or not record["scientific_output_generated"]):
        raise ValueError("Historical A1 record identity mismatch")
    proof, result, identity = (record[k] for k in
                               ("pretraining_evidence", "result", "scientific_identity"))
    if (proof["base_run_id"] != state or proof["replica_id"] != 1 or
            proof["treatment"] != treatment or proof["assigned_lr"] != lr or
            proof["protocol_id"] != "ReflexML-Momentum-Reset-v1" or
            result["base_run_id"] != state or result["replica_id"] != 1 or
            result["future_seed"] != proof["future_seed"] or
            result["pretraining_evidence"] != proof or
            result["action"] != ("Continue" if lr == 0.10 else "Switch") or
            identity["base_run_id"] != state or identity["replica_id"] != 1 or
            identity["treatment"] != treatment or identity["assigned_lr"] != lr or
            identity["protocol_id"] != proof["protocol_id"] or
            identity["protocol_sha256"] != proof["protocol_sha256"] or
            identity["future_seed"] != proof["future_seed"] or
            identity["model_parameters"] != proof["model_parameters"] or
            identity["reset_momentum_buffers"] != proof["reset_momentum_buffers"] or
            identity["canonical_generator_sha256"] != proof["canonical_post_s17_generator_sha256"] or
            identity["source_sha256"] != result["source_sha256"] or
            identity["source_sha256"] != proof["s17_source_sha256"] or
            identity["checkpoint_sha256"] != result["checkpoint_sha256"] or
            identity["checkpoint_sha256"] != proof["checkpoint_sha256"] or
            identity["expected_epoch18_order_sha256"] != result["epoch_order_sha256"] or
            identity["expected_epoch18_order_sha256"] != proof["expected_epoch18_order_sha256"] or
            result["training_step_count"] != 79 or result["learning_rate"] != lr or
            result["treatment"] != treatment or result["epoch"] != 18 or
            result["prestep_file_sha256"] != record["prestep_file_sha256"]):
        raise ValueError("Historical A1 scientific identity mismatch")
    if sha256_file(directory / f"{record['attempt_id']}.prestep") != record["prestep_file_sha256"]:
        raise ValueError("Historical A1 prestep mismatch")
    order = result["realized_order"]
    if (len(order) != 79 or [len(batch) for batch in order] != [128] * 78 + [16] or
            len(set(x for batch in order for x in batch)) != 10000 or
            order_hashes([order])[0] != result["epoch_order_sha256"]):
        raise ValueError("Historical A1 order or boundaries mismatch")
    if (len(proof["reset_momentum_buffers"]) != 4 or
            not all(row["all_zero_before_training"] for row in proof["reset_momentum_buffers"].values()) or
            len(proof["optimizer_groups_for_epoch18"]) != 1 or
            any(proof["optimizer_groups_for_epoch18"][0][key] != value for key, value in
                {"momentum": 0.9, "weight_decay": 0, "dampening": 0,
                 "nesterov": False, "lr": lr}.items())):
        raise ValueError("Historical A1 Reset condition mismatch")
    if not isinstance(result["val_loss"], float) or not isinstance(result["val_accuracy"], float):
        raise ValueError("Historical A1 validation endpoint missing")
    return record


def historical_a1_pair(state: int, directory: Path = HISTORICAL_DIRECTORY) -> tuple[dict, dict]:
    ten, five = historical_a1_cell(state, 0.10, directory), historical_a1_cell(state, 0.05, directory)
    a, b = ten["pretraining_evidence"], five["pretraining_evidence"]
    for key in ("s17_source_sha256", "checkpoint_sha256", "model_parameters",
                "reset_momentum_buffers", "canonical_post_s17_generator_sha256",
                "future_seed", "expected_epoch18_order_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"Historical A1 LR pair differs: {key}")
    if ten["result"]["realized_order"] != five["result"]["realized_order"]:
        raise ValueError("Historical A1 LR pair stream differs")
    return ten, five


def permutation_positions(state: int, stream: int, size: int = 10000) -> tuple[bytes, list[int]]:
    if not 1 <= state <= 36 or stream not in (2, 3, 4) or size != 10000:
        raise ValueError("Frozen stream identity requires state 1..36 and 10000 positions")
    message = PREFIX + state.to_bytes(4, "big") + stream.to_bytes(1, "big")
    key = hmac.new(MASTER_SEED, message, hashlib.sha256).digest()
    positions = sorted(range(size), key=lambda i: (hashlib.sha256(key + i.to_bytes(4, "big")).digest(), i))
    return key, positions


def batches(order: Sequence[int]) -> list[list[int]]:
    if len(order) != 10000 or len(set(order)) != 10000:
        raise ValueError("Each stream must contain 10000 unique examples")
    result = [list(order[i:i + 128]) for i in range(0, 10000, 128)]
    if [len(part) for part in result] != [128] * 78 + [16]:
        raise AssertionError("Frozen batch boundaries changed")
    return result


def construct_streams(state: int, ordered_t: Sequence[int], historical_order: Sequence[Sequence[int]]) -> dict[str, Any]:
    if len(ordered_t) != 10000 or len(set(ordered_t)) != 10000:
        raise ValueError("Canonical T required")
    first = [list(batch) for batch in historical_order]
    if first != batches([x for batch in first for x in batch]) or set(sum(first, [])) != set(ordered_t):
        raise ValueError("Historical A1 order or boundaries differ from T")
    streams = [first]
    keys = {}
    for j in (2, 3, 4):
        key, positions = permutation_positions(state, j)
        stream = batches([ordered_t[i] for i in positions])
        if set(sum(stream, [])) != set(ordered_t):
            raise AssertionError("Stream permutation identity failed")
        streams.append(stream)
        keys[str(j)] = key.hex()
    hashes = [order_hashes([stream])[0] for stream in streams]
    if len(set(hashes)) != 4:
        raise ValueError("Duplicate stream identities")
    return {"state": state, "streams": streams, "order_sha256": hashes,
            "stream_key_sha256": keys}


def prepare_candidate(unit: Any, dataset: Any, repo: Path, *,
                      s17_artifact_path: Path, source_checkpoint_path: Path) -> dict[str, Any]:
    """Read-only A1 admission and Reset preparation for a future audited branch."""
    if unit.replica_id != 1 or not 1 <= unit.base_run_id <= 36:
        raise ValueError("Only historical A1 states are in scope")
    ordered_t = canonical_train_indices(repo, dataset)
    ten, five = historical_a1_pair(unit.base_run_id)
    first = ten["result"]["realized_order"]
    for record in (ten, five):
        proof = record["pretraining_evidence"]
        if (proof["s17_source_sha256"] != unit.source_sha256 or
                proof["checkpoint_sha256"] != unit.checkpoint_sha256 or
                proof["future_seed"] != unit.future_seed or
                proof["expected_epoch18_order_sha256"] !=
                unit.historical["wait_epoch_order_sha256"][3] or
                first != unit.historical["wait_realized_orders"][3]):
            raise ValueError("Historical A1 differs from selected S17 source")
    streams = construct_streams(unit.base_run_id, ordered_t, first)
    prepared = {}
    for treatment in ("L_10R", "L_05R"):
        prepared[treatment] = prepare_reset(
            unit, dataset, treatment=treatment, repo=repo,
            s17_artifact_path=s17_artifact_path,
            source_checkpoint_path=source_checkpoint_path,
        )
    if (prepared["L_10R"].model_state.keys() != prepared["L_05R"].model_state.keys() or
            any(not torch.equal(prepared["L_10R"].model_state[name],
                                prepared["L_05R"].model_state[name])
                for name in prepared["L_10R"].model_state)):
        raise ValueError("LR pair Reset model differs")
    high, low = prepared["L_10R"], prepared["L_05R"]
    high_optimizer, low_optimizer = high.optimizer_state, low.optimizer_state
    high_group, low_group = high_optimizer["param_groups"], low_optimizer["param_groups"]
    if (len(high_group) != 1 or len(low_group) != 1 or
            {key: value for key, value in high_group[0].items() if key != "lr"} !=
            {key: value for key, value in low_group[0].items() if key != "lr"} or
            high_group[0]["lr"] != 0.10 or low_group[0]["lr"] != 0.05 or
            high_optimizer["state"].keys() != low_optimizer["state"].keys() or
            any(not torch.equal(high_optimizer["state"][key]["momentum_buffer"],
                                low_optimizer["state"][key]["momentum_buffer"])
                for key in high_optimizer["state"]) or
            not torch.equal(high.generator_state, low.generator_state)):
        raise ValueError("LR pair Reset optimizer or generator differs")
    return {"prepared": prepared, "streams": streams,
            "historical_record_sha256": {
                "L_10R": ten["record_sha256"], "L_05R": five["record_sha256"]}}


def assert_static_model(model: nn.Module) -> None:
    if (type(model) is not FashionMLP or
            [type(layer) for layer in model.network] != [nn.Flatten, nn.Linear, nn.ReLU, nn.Linear] or
            list(model.named_buffers())):
        raise ValueError("Historical model architecture or mutable state changed")


def mean_gradient_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                       components: Sequence[tuple[torch.Tensor, torch.Tensor]],
                       criterion: nn.Module) -> list[list[torch.Tensor]]:
    """Four fixed-parameter backward passes followed by one ordinary SGD step."""
    assert_static_model(model)
    if type(criterion) is not nn.CrossEntropyLoss or criterion.reduction != "mean":
        raise ValueError("Historical mean cross-entropy loss required")
    if len(components) != 4 or len({len(y) for _, y in components}) != 1:
        raise ValueError("Exactly four equal-sized component batches required")
    params = list(model.parameters())
    if (type(optimizer) is not torch.optim.SGD or len(optimizer.param_groups) != 1 or
            len(params) != 4 or optimizer.param_groups[0]["params"] != params or
            any(optimizer.param_groups[0][key] != value for key, value in
                {"momentum": 0.9, "weight_decay": 0, "dampening": 0,
                 "nesterov": False}.items())):
        raise ValueError("Historical one-group SGD settings required")
    before = [p.detach().clone() for p in params]
    buffers = [optimizer.state[p].get("momentum_buffer", None) for p in params]
    if any(buffer is None for buffer in buffers):
        raise ValueError("Four inherited momentum buffers required")
    saved_buffers = [None if b is None else b.clone() for b in buffers]
    gradients = []
    model.train()
    for images, labels in components:
        optimizer.zero_grad()
        criterion(model(images), labels).backward()
        if any(not torch.equal(p, original) for p, original in zip(params, before, strict=True)):
            raise AssertionError("Parameters advanced during component evaluation")
        if any(b is not None and not torch.equal(b, original)
               for b, original in zip(buffers, saved_buffers, strict=True)):
            raise AssertionError("Momentum advanced during component evaluation")
        gradients.append([p.grad.detach().clone() for p in params])
    optimizer.zero_grad()
    for index, parameter in enumerate(params):
        parameter.grad = torch.stack([row[index] for row in gradients]).mean(dim=0)
    optimizer.step()
    return gradients


def run_fixture_epoch(model: FashionMLP, optimizer: torch.optim.Optimizer, dataset: Any,
                      streams: Sequence[Sequence[Sequence[int]]], validation_loader: Any) -> dict[str, Any]:
    """Technical-only runner for a non-authoritative fixture; never a production entrypoint."""
    if len(streams) != 4 or any([len(batch) for batch in stream] != [128] * 78 + [16] for stream in streams):
        raise ValueError("Frozen 79-step stream boundaries required")
    criterion = nn.CrossEntropyLoss()
    for position in range(79):
        components = []
        for stream in streams:
            samples = [dataset[index] for index in stream[position]]
            components.append((torch.stack([sample[0] for sample in samples]),
                               torch.tensor([sample[1] for sample in samples])))
        mean_gradient_step(model, optimizer, components, criterion)
    endpoint = evaluate(model, validation_loader, criterion, torch.device("cpu"))
    return {"training_step_count": 79, "val_loss": endpoint.loss,
            "val_accuracy": endpoint.accuracy}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ordered_index_hash(values: Sequence[int]) -> str:
    return _sha256(torch.tensor(values, dtype=torch.int64).numpy().tobytes())


def _read_bound_json(repo: Path, name: str, expected_sha256: str) -> dict[str, Any]:
    path = repo / name
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Production authority changed: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_branch_roster(repo: Path) -> list[dict[str, Any]]:
    """Prospective 72-cell roster; contains identities, never scientific outcomes."""
    verify_authorities(repo)
    admission = _read_bound_json(repo, "GRADIENT_NOISE_M1_A1_ADMISSION.json", M1_ADMISSION_SHA256)
    manifest = _read_bound_json(repo, "GRADIENT_NOISE_ATTENUATION_STREAM_MANIFEST.json",
                                STREAM_MANIFEST_SHA256)
    if sha256_file(repo / "GRADIENT_NOISE_ATTENUATION_COMPATIBILITY_GATE.md") != COMPATIBILITY_SHA256:
        raise ValueError("Compatibility authority changed")
    if sha256_file(repo / "GRADIENT_NOISE_ATTENUATION_KERNEL_CANDIDATE.json") != OLD_KERNEL_CANDIDATE_SHA256:
        raise ValueError("Audited kernel candidate changed")
    if (admission.get("protocol_id") != PROTOCOL_ID or admission.get("expected") != 72 or
            admission.get("admitted") != 72 or admission.get("rejected") != 0 or
            admission.get("dataset_identity_sha256") != DATASET_SHA256 or
            admission.get("ordered_train_indices_sha256") != TRAIN_INDICES_SHA256 or
            manifest.get("protocol_id") != PROTOCOL_ID or
            manifest.get("state_count") != 36 or manifest.get("stream_count") != 144 or
            manifest.get("streams_per_state") != 4 or
            manifest.get("boundary_sizes") != [128] * 78 + [16] or
            manifest.get("dataset_identity_sha256") != DATASET_SHA256 or
            manifest.get("ordered_train_indices_sha256") != TRAIN_INDICES_SHA256 or
            manifest.get("master_seed_hex") != MASTER_SEED.hex()):
        raise ValueError("Historical admission or stream roster identity changed")
    cells = admission["cells"]
    rows = manifest["streams"]
    cell_by_key = {(cell["state"], cell["LR"]): cell for cell in cells}
    row_by_key = {(row["state"], row["stream_index"]): row for row in rows}
    expected_cells = {(state, lr) for state in range(1, 37) for lr in (0.10, 0.05)}
    expected_rows = {(state, stream) for state in range(1, 37) for stream in range(1, 5)}
    if (len(cells) != 72 or set(cell_by_key) != expected_cells or
            len(rows) != 144 or set(row_by_key) != expected_rows):
        raise ValueError("Historical admission or stream roster coverage changed")
    roster = []
    for state, lr in sorted(expected_cells):
        cell = cell_by_key[state, lr]
        streams = [row_by_key[state, stream] for stream in range(1, 5)]
        if (cell["replica"] != "A1" or cell["treatment"] != "Reset" or
                cell["admission_verdict"] != "ADMITTED" or
                cell["branch_key"] != f"{state:03d}-A-r01-{'L_10R' if lr == 0.10 else 'L_05R'}" or
                any(row["example_count"] != 10000 or
                    row["batch_sizes"] != "128 x 78, 16 x 1" or
                    row["LR_pair_uses_identical_stream"] is not True
                    for row in streams) or
                streams[0]["historical_order_hash_sha256"] != cell["realized_order_sha256"] or
                streams[0]["ordered_global_example_sha256"] != cell["ordered_global_example_sha256"]):
            raise ValueError("Admitted A1 cell differs from prospective stream authority")
        identity = {
            "protocol_id": PROTOCOL_ID, "base_run_id": state, "replica": "A1", "m": 4,
            "learning_rate": lr, "s17_source_sha256": cell["s17_source_sha256"],
            "checkpoint_sha256": cell["checkpoint_sha256"],
            "stream_manifest_sha256": STREAM_MANIFEST_SHA256,
            "stream_global_order_sha256": [row["ordered_global_example_sha256"] for row in streams],
            "stream_boundary_sha256": [row["minibatch_boundary_sha256"] for row in streams],
        }
        roster.append(identity)
    return roster


def canonical_branch_identity(repo: Path, state: int, lr: float) -> dict[str, Any]:
    if type(state) is not int or state not in range(1, 37) or type(lr) is not float or lr not in (0.10, 0.05):
        raise ValueError("Only canonical A1 m=4 state/LR branches are permitted")
    return next(row for row in canonical_branch_roster(repo)
                if row["base_run_id"] == state and row["learning_rate"] == lr)


def _verify_materialized_streams(identity: dict[str, Any], streams: Sequence[Sequence[Sequence[int]]]) -> None:
    if len(streams) != 4:
        raise ValueError("Four frozen streams required")
    for index, stream in enumerate(streams):
        if [len(batch) for batch in stream] != [128] * 78 + [16]:
            raise ValueError("Frozen batch boundaries changed")
        flat = [example for batch in stream for example in batch]
        if _ordered_index_hash(flat) != identity["stream_global_order_sha256"][index]:
            raise ValueError("Materialized stream differs from prospective manifest")


def _tensor_hashes(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: _sha256(value.detach().cpu().contiguous().numpy().tobytes())
            for name, value in tensors.items()}


def _optimizer_buffer_hashes(optimizer: torch.optim.Optimizer,
                             model: nn.Module) -> dict[str, str]:
    return _tensor_hashes({name: optimizer.state[parameter]["momentum_buffer"]
                           for name, parameter in model.named_parameters()})


def _durable_exclusive_json(path: Path, record: dict[str, Any]) -> str:
    content = _canonical_bytes(record) + b"\n"
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return _sha256(content)


def _endpoint(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return value


def _run_lifecycle(identity: dict[str, Any], model: FashionMLP,
                   optimizer: torch.optim.Optimizer, dataset: Any,
                   streams: Sequence[Sequence[Sequence[int]]], validation_loader: Any,
                   output_directory: Path) -> Path:
    """Internal technical entry; public production checks authorization first."""
    _verify_materialized_streams(identity, streams)
    if not output_directory.name.startswith("reflexml-gradient-noise-v1-production-"):
        raise ValueError("Isolated Gradient-Noise output directory required")
    if sha256_file(V2_CANDIDATE_PATH) != V2_CANDIDATE_SHA256:
        raise ValueError("Audited V2 production bridge candidate changed")
    if sha256_file(V3_CANDIDATE_PATH) != V3_CANDIDATE_SHA256:
        raise ValueError("Audited V3 production bridge candidate changed")
    candidate = json.loads(V4_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if (candidate.get("implementation_sha256") != sha256_file(Path(__file__)) or
            candidate.get("old_audited_implementation_sha256") != OLD_AUDITED_IMPLEMENTATION_SHA256 or
            candidate.get("previous_candidate_v2_sha256") != V2_CANDIDATE_SHA256 or
            candidate.get("previous_candidate_v3_sha256") != V3_CANDIDATE_SHA256 or
            candidate.get("production_root") != str(PRODUCTION_ROOT) or
            candidate.get("status") != "AUTHORIZATION-PLUMBING COMPLETE — EXECUTION AUTHORIZATION PENDING"):
        raise ValueError("V4 production bridge candidate identity changed")
    components = candidate["reused_components"]
    if any(sha256_file(Path(__file__).resolve().parents[1] / name) != digest
           for name, digest in components.items()):
        raise ValueError("Reused numerical component identity changed")
    candidate_sha256 = sha256_file(V4_CANDIDATE_PATH)
    branch_sha256 = _sha256(_canonical_bytes(identity))
    prefix = f"{identity['base_run_id']:03d}-A1-m4-lr{'10' if identity['learning_rate'] == 0.10 else '05'}-{branch_sha256[:16]}-a001"
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.glob(prefix + ".*")):
        raise PermissionError("Branch has an existing attempt or unresolved lifecycle")
    start = output_directory / (prefix + ".start.json")
    prestep = output_directory / (prefix + ".prestep.json")
    terminal = output_directory / (prefix + ".terminal.json")
    attempt = {"schema": "gradient_noise_attempt_start_v1", "attempt_id": prefix,
               "branch_identity": identity, "branch_identity_sha256": branch_sha256}
    start_sha256 = _durable_exclusive_json(start, attempt)
    proof = {
        "schema": "gradient_noise_prestep_v1", "attempt_id": prefix,
        "attempt_start_sha256": start_sha256, "branch_identity": identity,
        "protocol_sha256": PROTOCOL_SHA256, "freeze_record_sha256": FREEZE_SHA256,
        "implementation_sha256": sha256_file(Path(__file__)),
        "old_kernel_candidate_sha256": OLD_KERNEL_CANDIDATE_SHA256,
        "kernel_candidate_v2_sha256": V2_CANDIDATE_SHA256,
        "kernel_candidate_v3_sha256": V3_CANDIDATE_SHA256,
        "kernel_candidate_v4_sha256": candidate_sha256,
        "historical_admission_sha256": M1_ADMISSION_SHA256,
        "stream_manifest_sha256": STREAM_MANIFEST_SHA256,
        "compatibility_report_sha256": COMPATIBILITY_SHA256,
        "dataset_identity_sha256": DATASET_SHA256,
        "train_indices_sha256": TRAIN_INDICES_SHA256,
        "starting_model_tensor_sha256": _tensor_hashes(model.state_dict()),
        "starting_momentum_buffer_sha256": _optimizer_buffer_hashes(optimizer, model),
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "device": "cpu", "num_threads": torch.get_num_threads()},
        "kernel_components": components,
    }
    prestep_sha256 = _durable_exclusive_json(prestep, proof)
    # This is the unchanged audited 79-step numerical function.
    result = run_fixture_epoch(model, optimizer, dataset, streams, validation_loader)
    if result["training_step_count"] != 79:
        raise ValueError("Scientific kernel did not complete 79 updates")
    status = ("scientifically_valid_primary_evaluable" if math.isfinite(result["val_loss"])
              else "scientifically_valid_primary_non_evaluable")
    record = {
        "schema": "gradient_noise_terminal_v1", "attempt_id": prefix,
        "branch_identity": identity, "attempt_start_sha256": start_sha256,
        "prestep_sha256": prestep_sha256, "classification": status,
        "val_loss": _endpoint(result["val_loss"]),
        "val_accuracy": _endpoint(result["val_accuracy"]),
        "training_step_count": result["training_step_count"],
        "stream_global_order_sha256": identity["stream_global_order_sha256"],
        "stream_boundary_sha256": identity["stream_boundary_sha256"],
        "final_model_tensor_sha256": _tensor_hashes(model.state_dict()),
        "final_momentum_buffer_sha256": _optimizer_buffer_hashes(optimizer, model),
        "environment": proof["environment"],
        "implementation_sha256": proof["implementation_sha256"],
        "kernel_candidate_v2_sha256": V2_CANDIDATE_SHA256,
        "kernel_candidate_v3_sha256": V3_CANDIDATE_SHA256,
        "kernel_candidate_v4_sha256": candidate_sha256,
        "kernel_components": components,
    }
    _durable_exclusive_json(terminal, record)
    return terminal


def _require_execution_authorization() -> None:
    """Read the one project-local authority before any production side effect."""
    try:
        record = json.loads(EXECUTION_AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PermissionError("Gradient-Noise execution authorization unavailable") from error
    _validate_execution_authorization(record)


def _expected_execution_authorization() -> dict[str, Any]:
    verify_authorities(V4_CANDIDATE_PATH.parent)
    for name, digest in (
        ("GRADIENT_NOISE_M1_A1_ADMISSION.json", M1_ADMISSION_SHA256),
        ("GRADIENT_NOISE_ATTENUATION_STREAM_MANIFEST.json", STREAM_MANIFEST_SHA256),
        ("GRADIENT_NOISE_ATTENUATION_COMPATIBILITY_GATE.md", COMPATIBILITY_SHA256),
    ):
        if sha256_file(V4_CANDIDATE_PATH.parent / name) != digest:
            raise PermissionError(f"Gradient-Noise upstream authority changed: {name}")
    if (sha256_file(V3_CANDIDATE_PATH) != V3_CANDIDATE_SHA256 or
            sha256_file(V2_CANDIDATE_PATH) != V2_CANDIDATE_SHA256):
        raise PermissionError("Gradient-Noise predecessor candidate identity changed")
    candidate = json.loads(V4_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if (candidate.get("implementation_sha256") != sha256_file(Path(__file__)) or
            candidate.get("previous_candidate_v3_sha256") != V3_CANDIDATE_SHA256 or
            candidate.get("protocol_sha256") != PROTOCOL_SHA256 or
            candidate.get("freeze_record_sha256") != FREEZE_SHA256 or
            candidate.get("historical_admission_sha256") != M1_ADMISSION_SHA256 or
            candidate.get("stream_manifest_sha256") != STREAM_MANIFEST_SHA256 or
            candidate.get("compatibility_report_sha256") != COMPATIBILITY_SHA256 or
            candidate.get("dataset_identity_sha256") != DATASET_SHA256 or
            candidate.get("train_indices_sha256") != TRAIN_INDICES_SHA256 or
            candidate.get("production_root") != str(PRODUCTION_ROOT) or
            candidate.get("expected_new_branch_count") != 72 or
            candidate.get("status") != "AUTHORIZATION-PLUMBING COMPLETE — EXECUTION AUTHORIZATION PENDING"):
        raise PermissionError("Gradient-Noise V4 candidate identity mismatch")
    expected_environment = {
        "python": "3.14.3 (v3.14.3:323c59a5e34, Feb  3 2026, 11:41:37) [Clang 16.0.0 (clang-1600.0.26.6)]",
        "torch": "2.12.0", "device": "cpu", "num_threads": 4,
    }
    if candidate.get("environment") != expected_environment:
        raise PermissionError("Gradient-Noise audited environment identity mismatch")
    if any(sha256_file(V4_CANDIDATE_PATH.parent / name) != digest
           for name, digest in candidate["reused_components"].items()):
        raise PermissionError("Gradient-Noise numerical component identity mismatch")
    return {
        "schema": "ReflexML-Gradient-Noise-Attenuation-Execution-Authorization-v1",
        "execution_authorized": True,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "freeze_record_sha256": FREEZE_SHA256,
        "implementation_sha256": candidate["implementation_sha256"],
        "kernel_candidate_sha256": sha256_file(V4_CANDIDATE_PATH),
        "historical_admission_sha256": M1_ADMISSION_SHA256,
        "stream_manifest_sha256": STREAM_MANIFEST_SHA256,
        "compatibility_report_sha256": COMPATIBILITY_SHA256,
        "dataset_identity_sha256": DATASET_SHA256,
        "train_indices_sha256": TRAIN_INDICES_SHA256,
        "production_root": str(PRODUCTION_ROOT),
        "authorized_branch_count": 72,
        "authorized_states": list(range(1, 37)),
        "authorized_replica": "A1",
        "authorized_m": 4,
        "authorized_lrs": [0.10, 0.05],
        "m1_reruns_authorized": False,
        "a2_authorized": False,
        "source_substitution_authorized": False,
        "stream_substitution_authorized": False,
        "required_environment": expected_environment,
    }


def _validate_execution_authorization(record: Any) -> None:
    """Exact-key, exact-value authority; rejects bool/int and extra fields."""
    expected = _expected_execution_authorization()
    if not _same_exact(record, expected):
        raise PermissionError("Gradient-Noise execution authorization identity mismatch")


def _same_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return set(left) == set(right) and all(_same_exact(left[key], value)
                                                for key, value in right.items())
    if isinstance(right, list):
        return len(left) == len(right) and all(_same_exact(a, b)
                                               for a, b in zip(left, right, strict=True))
    return left == right


def _require_production_environment() -> None:
    """Admit only the exact audited CPU runtime; never repair it by mutation."""
    candidate = json.loads(V4_CANDIDATE_PATH.read_text(encoding="utf-8"))
    expected = candidate.get("environment")
    current = {"python": sys.version, "torch": torch.__version__,
               "device": torch.device("cpu").type,
               "num_threads": torch.get_num_threads()}
    if (candidate.get("implementation_sha256") != sha256_file(Path(__file__)) or
            candidate.get("production_root") != str(PRODUCTION_ROOT) or
            expected != current or expected != {
                "python": "3.14.3 (v3.14.3:323c59a5e34, Feb  3 2026, 11:41:37) [Clang 16.0.0 (clang-1600.0.26.6)]",
                "torch": "2.12.0", "device": "cpu", "num_threads": 4,
            }):
        raise ValueError("Gradient-Noise production numerical environment mismatch")


def run_production(state: int | None = None, lr: float | None = None,
                   dataset: Any = None, repo: Path | None = None,
                   output_directory: Path | None = None) -> Path:
    """One canonical branch; authorization precedes all production side effects."""
    _require_execution_authorization()
    if output_directory is not None and output_directory != PRODUCTION_ROOT:
        raise ValueError("Alternate Gradient-Noise production root is forbidden")
    _require_production_environment()
    if repo is None:
        raise ValueError("Frozen repository required")
    identity = canonical_branch_identity(repo, state, lr)
    from .momentum_reset import _canonical_source_paths, load_canonical_roster
    unit = load_canonical_roster(repo)[state, 1]
    if (unit.source_sha256 != identity["s17_source_sha256"] or
            unit.checkpoint_sha256 != identity["checkpoint_sha256"]):
        raise ValueError("Canonical S17 source differs from branch roster")
    artifact, checkpoint = _canonical_source_paths(unit)
    candidate = prepare_candidate(unit, dataset, repo, s17_artifact_path=artifact,
                                  source_checkpoint_path=checkpoint)
    streams = candidate["streams"]["streams"]
    _verify_materialized_streams(identity, streams)
    treatment = "L_10R" if lr == 0.10 else "L_05R"
    prepared = candidate["prepared"][treatment]
    config = ExperimentConfig(**unit.checkpoint["config"])
    with torch.random.fork_rng(devices=[]):
        model = FashionMLP(config.hidden_size)
    model.load_state_dict(prepared.model_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer.load_state_dict(prepared.optimizer_state)
    validation_loader = make_loaders_from_datasets(dataset, [], config).val_loader
    return _run_lifecycle(identity, model, optimizer, dataset, streams,
                          validation_loader, PRODUCTION_ROOT)

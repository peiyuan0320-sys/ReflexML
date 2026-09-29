"""Frozen Phase5-v1 implementation contracts and preflight-safe utilities.

This module contains no automatic experiment entry point.  Real base or branch
execution requires an explicit authorization argument and preflight artifacts;
the repository's current backup hard gate remains unresolved.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .branching import BranchingConfig, order_hashes, run_branch, states_equal
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import make_loaders_from_datasets, make_split_indices, split_fingerprint
from .model import FashionMLP
from .reproducibility import set_reproducible_seed
from .stability import with_future_shuffle
from .training import evaluate, train_one_epoch


PROTOCOL_VERSION = "Phase5-v1"
PROTOCOL_COMMIT = "dc0b6a7e3447e656c4499931a6fb4c32e8d4c765"
PROTOCOL_SHA256 = "8eff292a2e5027bec893d6cf43e327f961ae91bf390d9ed28a65a996a0799352"
PREFLIGHT_REVISION = 3
TRUST_ROOT_PATH = "phase5_trust_root.json"
CANONICAL_PREFLIGHT_PATH = "phase5_preflight_r3"
TRUST_ROOT_SHA256 = "65682e72b4a06e7165ac89443ed147d9d7f9bad8d0ee47551011d5dc59e07316"
REVIEWED_PREFLIGHT_MANIFEST_SHA256 = (
    "9f5cd6bfdbefc95fed45de38478668a49ddc15d2db3f532f20b7494e52b84bd5"
)
PREDECESSOR_PREFLIGHT_PATH = "phase5_preflight_r2/preflight_manifest.json"
PREDECESSOR_MANIFEST_SHA256 = "276169afb3cd8ca3db1376a8c0ffcdf12704c4ae2776c57688b590c574dbfec8"
CANDIDATE_STATUS = "CANDIDATE / NOT APPROVED FOR REAL PHASE 5 EXECUTION"
# Compatibility export for callers that only need the unchanged status text.
R2_STATUS = CANDIDATE_STATUS
SYNTHETIC_PROTOCOL_VERSION = "SyntheticPhase5Fixture-v1"
PRIMARY_CHECKPOINT_EPOCH = 14
N_STATES = 36
K_A = 15
K_B = 10
TOTAL_FUTURE_REPLICAS = N_STATES * (K_A + K_B)
TOTAL_PHASE5_SEEDS = N_STATES + TOTAL_FUTURE_REPLICAS
BOOTSTRAP_NAMESPACE = "ReflexML|Phase5|v1|bootstrap|primary"
BOOTSTRAP_SHA256 = "94778b440d5865af8a3d6d482f184d7eb1c40cde8aaa65a3f3134ad2c289f770"
BOOTSTRAP_SEED = 10698172564239836591
BOOTSTRAP_REPLICATES = 20_000
NON_EVALUABLE = "NON-EVALUABLE AS SPECIFIED"
EVALUABLE = "EVALUABLE AS SPECIFIED"
SEED_MODULUS = 2**31 - 2


@dataclass(frozen=True)
class Phase5Design:
    protocol_version: str = PROTOCOL_VERSION
    checkpoint_epoch: int = PRIMARY_CHECKPOINT_EPOCH
    state_count: int = N_STATES
    a_replicas: int = K_A
    b_replicas: int = K_B
    a_end_epoch: int = 18
    b_end_epoch: int = 30
    wait_epochs: int = 3
    initial_learning_rate: float = 0.1
    reduced_learning_rate: float = 0.05
    final_window: tuple[int, int, int] = (28, 29, 30)
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_bit_generator: str = "PCG64"


DESIGN = Phase5Design()


_PREFLIGHT_SEAL = object()
_CANONICAL_PREFLIGHT_SEAL = object()
_GATE_SEAL = object()
_PRIMARY_FIXTURE_SEAL = object()
_PRIMARY_PRODUCTION_SEAL = object()
_DATASET_PRODUCTION_SEAL = object()
_DATASET_FIXTURE_SEAL = object()


@dataclass(frozen=True)
class CandidatePreflightEvidence:
    """Hash-validated r3 evidence.  This is not execution authorization."""

    repo: Path
    path: Path
    manifest_sha256: str
    base_manifest_sha256: str
    future_manifest_sha256: str
    historical_base_registry_sha256: str
    historical_checkpoint_registry_sha256: str
    base_rows: tuple[dict[str, Any], ...]
    future_rows: tuple[dict[str, Any], ...]
    historical_base_seeds: frozenset[int]
    historical_checkpoint_hashes: frozenset[str]
    dataset_identity: dict[str, Any]
    gate_authority_registry: dict[str, Any]
    primary_checkpoint_registry: dict[str, Any]
    branch_artifact_registry: dict[str, Any]
    primary_checkpoint_registry_sha256: str
    branch_artifact_registry_sha256: str
    trust_root_sha256: str | None
    _seal: object


@dataclass(frozen=True)
class ProductionGateContext:
    """A context issued only after file-backed gate evidence is revalidated."""

    stage: str
    preflight: CandidatePreflightEvidence
    independent_audit_sha256: str
    hard_gate_evidence_sha256: str
    authorization_sha256: str
    independent_audit_path: Path
    hard_gate_evidence_path: Path
    authorization_path: Path
    _seal: object


@dataclass(frozen=True)
class ValidatedPrimaryData:
    """Frozen-shape primary arrays constructed only from provenance-bound rows."""

    a_delta: np.ndarray
    b_now_final_losses: np.ndarray
    b_wait_final_losses: np.ndarray
    divergence_records: tuple[dict[str, Any], ...]
    terminal_assignments: tuple[dict[str, Any], ...]
    preflight_manifest_sha256: str
    source_record_count: int
    canonical_production_records: bool
    _seal: object


@dataclass(frozen=True)
class DatasetValidationResult:
    """Mechanically checked dataset identity; fixtures can never become canonical."""

    class_identity: str
    content_sha256: str
    split_fingerprint: str
    canonical_phase5_dataset: bool
    _seal: object


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_protocol(repo: Path) -> str:
    """Fail closed if the authoritative Phase5-v1 protocol bytes changed."""
    protocol_path = repo.resolve() / "PHASE5_DESIGN.md"
    if not protocol_path.is_file():
        raise ValueError("Frozen Phase5-v1 protocol file is missing")
    actual = sha256_file(protocol_path)
    if actual != PROTOCOL_SHA256:
        raise ValueError(
            f"Frozen Phase5-v1 protocol SHA-256 mismatch: expected {PROTOCOL_SHA256}, got {actual}"
        )
    return actual


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid required JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Required JSON artifact is not an object: {path}")
    return value


def _read_seed_csv(path: Path, *, future: bool) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = (
        ("base_run_id", "counter", "future_seed", "replica_id", "start_epoch", "end_epoch")
        if future
        else ("base_run_id", "index", "counter", "base_seed")
    )
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        if row.get("pre_outcome") not in ("True", True):
            raise ValueError(f"Seed manifest row lacks pre-outcome status: {path}")
        row["pre_outcome"] = True
    return rows


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def dataset_class_identity(dataset: Any) -> str:
    cls = type(dataset)
    return f"{cls.__module__}.{cls.__qualname__}"


def dataset_content_sha256(dataset: Any) -> str:
    """Hash the in-memory image/label population used by a dataset.

    Canonical Fashion-MNIST exposes tensors named ``data`` and ``targets``.  A
    fixture may use the same representation to exercise the hashing mechanics,
    but class and registration checks keep that fixture noncanonical.
    """
    data = getattr(dataset, "data", None)
    targets = getattr(dataset, "targets", None)
    if not isinstance(data, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError("Dataset identity requires tensor data and targets")
    data_cpu = data.detach().cpu().contiguous()
    targets_cpu = targets.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(b"FashionMNIST|train=True|data|")
    digest.update(data_cpu.numpy().tobytes())
    digest.update(b"|targets|")
    digest.update(targets_cpu.numpy().tobytes())
    return digest.hexdigest()


def _indices_sha256(indices: Iterable[int]) -> str:
    values = torch.tensor(list(indices), dtype=torch.int64)
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def validate_dataset_against_registration(
    dataset: Any,
    registration: dict[str, Any],
    *,
    canonical_registration: bool,
) -> DatasetValidationResult:
    """Validate class, transform, population bytes, and the exact frozen split."""
    expected_class = str(registration.get("dataset_class", ""))
    actual_class = dataset_class_identity(dataset)
    if actual_class != expected_class:
        raise ValueError("Training dataset class/source identity is not registered")
    if len(dataset) != int(registration.get("population_size", -1)):
        raise ValueError("Training dataset population size differs from its registration")
    if bool(getattr(dataset, "train", False)) is not bool(registration.get("train", False)):
        raise ValueError("Training dataset source split differs from its registration")
    transform = getattr(dataset, "transform", None)
    transform_identity = (
        f"{type(transform).__module__}.{type(transform).__qualname__}:{repr(transform)}"
    )
    if transform_identity != registration.get("transform_identity"):
        raise ValueError("Training dataset transform/preprocessing identity is not registered")
    if getattr(dataset, "target_transform", None) is not None:
        raise ValueError("Phase5-v1 forbids target transforms")
    content_sha = dataset_content_sha256(dataset)
    if content_sha != registration.get("content_sha256"):
        raise ValueError("Training dataset content fingerprint is not registered")
    split_seed = int(registration.get("split_seed", -1))
    train_size = int(registration.get("train_size", -1))
    val_size = int(registration.get("validation_size", -1))
    train_indices, val_indices = make_split_indices(len(dataset), train_size, val_size, split_seed)
    actual_split = split_fingerprint(train_indices, val_indices)
    if (
        actual_split != registration.get("split_fingerprint")
        or _indices_sha256(train_indices) != registration.get("train_indices_sha256")
        or _indices_sha256(val_indices) != registration.get("validation_indices_sha256")
    ):
        raise ValueError("Training dataset exact train/validation subset identity changed")
    if canonical_registration:
        from torchvision.datasets import FashionMNIST
        from torchvision.transforms import ToTensor

        if type(dataset) is not FashionMNIST or type(transform) is not ToTensor:
            raise ValueError("Canonical dataset must use the exact torchvision FashionMNIST/ToTensor classes")
        required = {
            "registry_type": "phase5_v1_frozen_dataset_identity",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "dataset_source": "FashionMNIST official training set",
            "dataset_class": "torchvision.datasets.mnist.FashionMNIST",
            "train": True,
            "population_size": 60_000,
            "split_seed": 2026,
            "train_size": 10_000,
            "validation_size": 5_000,
            "transform_identity": "torchvision.transforms.transforms.ToTensor:ToTensor()",
            "split_fingerprint": "d697ce4e11ce24f9217f3d61e6bf4252ef918c6db88c658d0d029b7216bf016f",
        }
        if any(registration.get(key) != value for key, value in required.items()):
            raise ValueError("Dataset registration is not the frozen canonical Phase5-v1 identity")
        raw_hashes = registration.get("raw_file_sha256")
        if not isinstance(raw_hashes, dict):
            raise ValueError("Canonical dataset registration lacks raw-file integrity")
        raw_folder = Path(str(getattr(dataset, "raw_folder", "")))
        for relative, expected in raw_hashes.items():
            raw_path = raw_folder / Path(relative).name
            if not raw_path.is_file() or sha256_file(raw_path) != expected:
                raise ValueError("Canonical Fashion-MNIST raw-file integrity check failed")
    return DatasetValidationResult(
        class_identity=actual_class,
        content_sha256=content_sha,
        split_fingerprint=actual_split,
        canonical_phase5_dataset=canonical_registration,
        _seal=_DATASET_PRODUCTION_SEAL if canonical_registration else _DATASET_FIXTURE_SEAL,
    )


def validate_phase5_training_dataset(
    dataset: Any, evidence: CandidatePreflightEvidence
) -> DatasetValidationResult:
    """Production validation resolves only the implementation-anchored registration."""
    evidence = _require_canonical_candidate(evidence)
    return validate_dataset_against_registration(
        dataset, evidence.dataset_identity, canonical_registration=True
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def protocol_seed(namespace: str) -> int:
    if namespace != namespace.strip() or "\n" in namespace or "\r" in namespace:
        raise ValueError("Seed namespace must have no surrounding whitespace or newline")
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return 1 + value % SEED_MODULUS


def bootstrap_seed_derivation() -> dict[str, Any]:
    encoded = BOOTSTRAP_NAMESPACE.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    seed = int.from_bytes(bytes.fromhex(digest)[:8], "big", signed=False)
    if digest != BOOTSTRAP_SHA256 or seed != BOOTSTRAP_SEED:
        raise AssertionError("Frozen bootstrap namespace derivation changed")
    return {
        "namespace": BOOTSTRAP_NAMESPACE,
        "encoding": "UTF-8, exact bytes, no surrounding whitespace or newline",
        "sha256": digest,
        "first_8_bytes_hex": digest[:16],
        "derived_integer_seed": seed,
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "protocol_version": PROTOCOL_VERSION,
        "replicates": BOOTSTRAP_REPLICATES,
    }


def primary_bootstrap_rng() -> np.random.Generator:
    """Frozen canonical RNG; it intentionally exposes no override."""
    bootstrap_seed_derivation()
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))


def _source_record(repo: Path, relative: str) -> dict[str, str]:
    path = repo / relative
    if not path.is_file():
        raise ValueError(f"Required historical provenance is missing: {relative}")
    return {"path": relative, "sha256": sha256_file(path)}


def build_historical_base_seed_registry(repo: Path) -> dict[str, Any]:
    """Fail-closed reconstruction from Phase 3 identity/provenance only."""
    repo = repo.resolve()
    source_names = [
        "runs/phase3/design.json",
        "runs/phase3/timing_dataset.csv",
        "runs/phase3/audit.json",
        "runs/phase3/independent_audit.json",
    ]
    sources = [_source_record(repo, name) for name in source_names]
    design = json.loads((repo / source_names[0]).read_text(encoding="utf-8"))
    audit = json.loads((repo / source_names[2]).read_text(encoding="utf-8"))
    independent = json.loads((repo / source_names[3]).read_text(encoding="utf-8"))
    try:
        all_seeds = [int(seed) for seed in design["all_seeds"]]
        override = [int(seed) for seed in design["timing_seeds_override"]]
        split_seeds = sorted(
            int(seed) for values in design["seed_splits"].values() for seed in values
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Phase 3 design lacks reliable base-seed provenance") from error
    with (repo / source_names[1]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        dataset_seeds = [int(row["seed"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Phase 3 dataset lacks reliable base-seed identities") from error
    unique = sorted(set(dataset_seeds))
    counts = {seed: dataset_seeds.count(seed) for seed in unique}
    reliable = (
        len(all_seeds) == len(set(all_seeds))
        and sorted(all_seeds) == sorted(override) == split_seeds == unique
        and len(rows) == len(unique) * 6 == 120
        and set(counts.values()) == {6}
        and audit.get("sample_count") == 120
        and audit.get("all_pairs_checked") is True
        and independent.get("samples_recomputed") == 120
        and independent.get("old_artifacts_unchanged") is True
    )
    if not reliable:
        raise ValueError("Historical base-seed evidence is incomplete or inconsistent; refuse to guess")
    entries = [{"seed": seed, "phase3_rows": counts[seed]} for seed in unique]
    return {
        "registry_type": "historical_base_training_seeds",
        "protocol_version": PROTOCOL_VERSION,
        "construction": (
            "Intersection-by-exact-agreement of Phase 3 all_seeds, timing override, "
            "seed splits, and the audited 120-row timing dataset; six rows required per seed."
        ),
        "sources": sources,
        "entry_count": len(entries),
        "entries": entries,
        "entries_sha256": canonical_sha256(entries),
        "outcome_blind_membership": True,
        "prohibited_outcome_fields_used": [],
        "reliable": True,
    }


def build_historical_checkpoint_registry(repo: Path) -> dict[str, Any]:
    """Reconstruct checkpoint identities from the completed Phase 4A protected manifest."""
    repo = repo.resolve()
    manifest_name = "runs/phase4a_horizon/input_manifest.json"
    source = _source_record(repo, manifest_name)
    manifest = json.loads((repo / manifest_name).read_text(encoding="utf-8"))
    protected = manifest.get("protected_file_sha256")
    if not isinstance(protected, dict):
        raise ValueError("Phase 4A input manifest lacks protected file hashes")
    rows = []
    for relative, recorded_hash in sorted(protected.items()):
        if not relative.endswith(".pt"):
            continue
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"Historical checkpoint is missing: {relative}")
        actual = sha256_file(path)
        if actual != recorded_hash:
            raise ValueError(f"Historical checkpoint hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    if not rows:
        raise ValueError("No reliable historical checkpoints found")
    unique_hashes = sorted({row["sha256"] for row in rows})
    return {
        "registry_type": "historical_checkpoint_hashes",
        "protocol_version": PROTOCOL_VERSION,
        "construction": (
            "Select every .pt entry from the completed Phase 4A protected-input manifest "
            "and require each current file to match its recorded SHA-256."
        ),
        "scope": "All .pt checkpoints protected by the completed Phase 4A input manifest",
        "sources": [source],
        "path_count": len(rows),
        "unique_hash_count": len(unique_hashes),
        "entries": rows,
        "unique_hashes": unique_hashes,
        "entries_sha256": canonical_sha256(rows),
        "outcome_blind_membership": True,
        "prohibited_outcome_fields_used": [],
        "reliable": True,
    }


def generate_seed_manifests(
    historical_base_seeds: Iterable[int],
    explicit_base_exclusions: Iterable[int] = (),
    explicit_future_exclusions: Iterable[int] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical = {int(seed) for seed in historical_base_seeds}
    base_exclusions = historical | {int(seed) for seed in explicit_base_exclusions}
    future_exclusions = {int(seed) for seed in explicit_future_exclusions}
    used: set[int] = set()
    base_rows: list[dict[str, Any]] = []
    for index in range(1, N_STATES + 1):
        counter = 0
        while True:
            namespace = f"ReflexML|Phase5|v1|base|{index}|{counter}"
            seed = protocol_seed(namespace)
            if seed not in used and seed not in base_exclusions:
                break
            counter += 1
        used.add(seed)
        base_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "base_run_id": index,
                "index": index,
                "counter": counter,
                "namespace": namespace,
                "base_seed": seed,
                "pre_outcome": True,
            }
        )
    future_rows: list[dict[str, Any]] = []
    for base_run_id in range(1, N_STATES + 1):
        for block, count in (("A", K_A), ("B", K_B)):
            for replica_id in range(1, count + 1):
                counter = 0
                while True:
                    namespace = (
                        f"ReflexML|Phase5|v1|future|{base_run_id}|{block}|"
                        f"{replica_id}|{counter}"
                    )
                    seed = protocol_seed(namespace)
                    if seed not in used and seed not in future_exclusions:
                        break
                    counter += 1
                used.add(seed)
                future_rows.append(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "base_run_id": base_run_id,
                        "block": block,
                        "replica_id": replica_id,
                        "counter": counter,
                        "namespace": namespace,
                        "future_seed": seed,
                        "start_epoch": 15,
                        "end_epoch": 18 if block == "A" else 30,
                        "pre_outcome": True,
                    }
                )
    validate_seed_manifests(base_rows, future_rows, historical)
    return base_rows, future_rows


def validate_seed_manifests(
    base_rows: list[dict[str, Any]],
    future_rows: list[dict[str, Any]],
    historical_base_seeds: Iterable[int],
) -> None:
    if len(base_rows) != N_STATES or len(future_rows) != TOTAL_FUTURE_REPLICAS:
        raise ValueError("Phase 5 seed manifest row counts differ from the frozen design")
    if [row["base_run_id"] for row in base_rows] != list(range(1, N_STATES + 1)):
        raise ValueError("Base-run IDs must be exactly 1..36")
    expected_keys = [
        (base, block, replica)
        for base in range(1, N_STATES + 1)
        for block, count in (("A", K_A), ("B", K_B))
        for replica in range(1, count + 1)
    ]
    actual_keys = [(r["base_run_id"], r["block"], r["replica_id"]) for r in future_rows]
    if actual_keys != expected_keys:
        raise ValueError("Future mapping order or A/B assignments differ from the protocol")
    seeds = [row["base_seed"] for row in base_rows] + [row["future_seed"] for row in future_rows]
    if len(seeds) != TOTAL_PHASE5_SEEDS or len(set(seeds)) != TOTAL_PHASE5_SEEDS:
        raise ValueError("All 936 Phase 5 seeds must be globally unique")
    if {r["base_seed"] for r in base_rows} & {int(s) for s in historical_base_seeds}:
        raise ValueError("Phase 5 base seed overlaps the historical base-seed registry")
    for row in base_rows:
        if row.get("protocol_version") != PROTOCOL_VERSION or row.get("pre_outcome") is not True:
            raise ValueError("Base-seed row has invalid protocol provenance")
        expected_namespace = (
            f"ReflexML|Phase5|v1|base|{int(row['base_run_id'])}|{int(row['counter'])}"
        )
        if row.get("namespace") != expected_namespace or int(row["index"]) != int(row["base_run_id"]):
            raise ValueError("Base-seed namespace/counter identity is invalid")
        if protocol_seed(row["namespace"]) != row["base_seed"]:
            raise ValueError("Base-seed row is not reproducible")
    for row in future_rows:
        if row.get("protocol_version") != PROTOCOL_VERSION or row.get("pre_outcome") is not True:
            raise ValueError("Future-seed row has invalid protocol provenance")
        expected_namespace = (
            f"ReflexML|Phase5|v1|future|{int(row['base_run_id'])}|{row['block']}|"
            f"{int(row['replica_id'])}|{int(row['counter'])}"
        )
        expected_end = 18 if row["block"] == "A" else 30
        if (
            row.get("namespace") != expected_namespace
            or int(row["start_epoch"]) != 15
            or int(row["end_epoch"]) != expected_end
        ):
            raise ValueError("Future-seed namespace/counter or block boundary is invalid")
        if protocol_seed(row["namespace"]) != row["future_seed"]:
            raise ValueError("Future-seed row is not reproducible")


def _load_preflight_kernel(
    repo: Path,
    preflight_dir: Path,
    *,
    expected_manifest_sha256: str | None = None,
    static_artifact_sha256: dict[str, str] | None = None,
    dynamic_registry_authority: dict[str, Any] | None = None,
    trust_root_sha256: str | None = None,
    seal: object = _PREFLIGHT_SEAL,
) -> CandidatePreflightEvidence:
    """Generic validation kernel; it is not the canonical production trust boundary."""
    repo = repo.resolve()
    preflight_dir = preflight_dir.resolve()
    assert_frozen_protocol(repo)
    manifest_path = preflight_dir / "preflight_manifest.json"
    manifest = _read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise PermissionError("Canonical reviewed preflight manifest identity mismatch")
    revision = manifest.get("preflight_evidence_revision")
    if revision == PREFLIGHT_REVISION:
        required_identity = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_sha256": PROTOCOL_SHA256,
            "preflight_evidence_revision": PREFLIGHT_REVISION,
            "predecessor_preflight_path": PREDECESSOR_PREFLIGHT_PATH,
            "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
            "predecessor_status": "historical_candidate_r2_preserved_byte_identically",
            "status": CANDIDATE_STATUS,
        }
        predecessor = repo / PREDECESSOR_PREFLIGHT_PATH
        predecessor_sha256 = PREDECESSOR_MANIFEST_SHA256
    elif revision == 4:
        required_identity = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_sha256": PROTOCOL_SHA256,
            "preflight_evidence_revision": 4,
            "predecessor_preflight_path": (
                f"{CANONICAL_PREFLIGHT_PATH}/preflight_manifest.json"
            ),
            "predecessor_manifest_sha256": REVIEWED_PREFLIGHT_MANIFEST_SHA256,
            "predecessor_status": "historical_candidate_r3_preserved_byte_identically",
            "status": CANDIDATE_STATUS,
        }
        predecessor = repo / CANONICAL_PREFLIGHT_PATH / "preflight_manifest.json"
        predecessor_sha256 = REVIEWED_PREFLIGHT_MANIFEST_SHA256
    else:
        raise ValueError("Unsupported Phase5-v1 candidate evidence revision")
    mismatched = [key for key, value in required_identity.items() if manifest.get(key) != value]
    if mismatched:
        raise ValueError(f"Revision-3 preflight identity mismatch: {', '.join(mismatched)}")
    if manifest.get("independent_conformance_audit", {}).get("status") != "ABSENT":
        raise ValueError("Candidate r3 must not claim independent conformance approval")
    if manifest.get("real_execution_authorization", {}).get("status") != "ABSENT":
        raise ValueError("Candidate r3 must not claim real execution authorization")
    if sha256_file(predecessor) != predecessor_sha256:
        raise ValueError("Predecessor preflight manifest changed")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("Revision-3 artifact hash manifest is missing")
    actual_artifacts = {
        path.name for path in preflight_dir.iterdir() if path.is_file() and path.name != "preflight_manifest.json"
    }
    if actual_artifacts != set(artifact_hashes):
        raise ValueError("Revision-3 artifact set differs from its manifest")
    dynamic_names = {
        str(authority.get("path"))
        for authority in (dynamic_registry_authority or {}).values()
        if isinstance(authority, dict)
    }
    for name, expected in artifact_hashes.items():
        if name in dynamic_names:
            continue
        if sha256_file(preflight_dir / name) != expected:
            raise ValueError(f"Revision-3 artifact hash mismatch: {name}")
    if static_artifact_sha256 is not None:
        if set(static_artifact_sha256) & dynamic_names:
            raise ValueError("Dynamic registries must not be content-hash pinned by the trust root")
        for name, expected in static_artifact_sha256.items():
            path = preflight_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise PermissionError(f"Implementation-anchored static identity mismatch: {name}")
            if artifact_hashes.get(name) != expected:
                raise PermissionError(f"Reviewed manifest disagrees with trust root: {name}")
    code_hashes = manifest.get("code_test_sha256")
    if not isinstance(code_hashes, dict):
        raise ValueError("Revision-3 code/test hashes are missing")
    for relative, expected in code_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Revision-3 code/test hash record is malformed")

    base_rows = _read_seed_csv(preflight_dir / "base_seed_manifest.csv", future=False)
    future_rows = _read_seed_csv(preflight_dir / "future_seed_mapping.csv", future=True)
    base_registry = _read_json(preflight_dir / "historical_base_seed_registry.json")
    if base_registry.get("reliable") is not True or base_registry.get("outcome_blind_membership") is not True:
        raise ValueError("Historical base-seed registry is not reliable and outcome-blind")
    historical_seeds = frozenset(int(row["seed"]) for row in base_registry["entries"])
    validate_seed_manifests(base_rows, future_rows, historical_seeds)
    checkpoint_metadata = _read_json(preflight_dir / "historical_checkpoint_registry_metadata.json")
    checkpoint_csv = preflight_dir / "historical_checkpoint_registry.csv"
    if checkpoint_metadata.get("registry_csv_sha256") != sha256_file(checkpoint_csv):
        raise ValueError("Historical checkpoint registry identity mismatch")
    with checkpoint_csv.open(newline="", encoding="utf-8") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    historical_checkpoint_hashes = frozenset(row["sha256"] for row in checkpoint_rows)
    if (
        checkpoint_metadata.get("reliable") is not True
        or checkpoint_metadata.get("outcome_blind_membership") is not True
        or len(historical_checkpoint_hashes) != int(checkpoint_metadata["unique_hash_count"])
    ):
        raise ValueError("Historical checkpoint registry is not reliable and outcome-blind")
    dataset_identity = _read_json(preflight_dir / "dataset_identity.json")
    if (
        dataset_identity.get("registry_type") != "phase5_v1_frozen_dataset_identity"
        or dataset_identity.get("protocol_version") != PROTOCOL_VERSION
        or dataset_identity.get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise ValueError("Frozen dataset identity registry is invalid")
    gate_registry = _read_json(preflight_dir / "gate_authority_registry.json")
    if (
        gate_registry.get("registry_type") != "phase5_v1_gate_authority_registry"
        or gate_registry.get("protocol_version") != PROTOCOL_VERSION
        or gate_registry.get("protocol_sha256") != PROTOCOL_SHA256
        or gate_registry.get("status") != "NO_TRUSTED_POSITIVE_AUTHORITY_ARTIFACTS"
    ):
        raise ValueError("Gate authority registry is invalid")
    primary_authority = (dynamic_registry_authority or {}).get("primary_checkpoint_registry")
    primary_name = (
        str(primary_authority.get("path"))
        if isinstance(primary_authority, dict)
        else "primary_checkpoint_registry.json"
    )
    checkpoint_registry = _read_json(preflight_dir / primary_name)
    if isinstance(primary_authority, dict):
        entries = checkpoint_registry.get("entries")
        status = checkpoint_registry.get("status")
        if (
            checkpoint_registry.get("registry_type") != primary_authority.get("registry_type")
            or checkpoint_registry.get("protocol_version") != PROTOCOL_VERSION
            or checkpoint_registry.get("protocol_sha256") != PROTOCOL_SHA256
            or checkpoint_registry.get("required_count") != primary_authority.get("required_count")
            or status not in primary_authority.get("allowed_statuses", [])
            or not isinstance(entries, list)
            or (status == "ABSENT" and entries != [])
            or (status == "APPROVED" and len(entries) != N_STATES)
        ):
            raise PermissionError("Canonical primary-checkpoint registry authority is invalid")
    elif (
        checkpoint_registry.get("registry_type") != "phase5_v1_primary_checkpoint_registry"
        or checkpoint_registry.get("status") != "ABSENT"
        or checkpoint_registry.get("entries") != []
    ):
        raise ValueError("Candidate r3 must not claim a reviewed primary-checkpoint set")
    branch_authority = (dynamic_registry_authority or {}).get("branch_artifact_registry")
    branch_name = (
        str(branch_authority.get("path"))
        if isinstance(branch_authority, dict)
        else "branch_artifact_registry.json"
    )
    branch_registry = _read_json(preflight_dir / branch_name)
    if isinstance(branch_authority, dict):
        entries = branch_registry.get("entries")
        status = branch_registry.get("status")
        if (
            branch_registry.get("registry_type") != branch_authority.get("registry_type")
            or branch_registry.get("protocol_version") != PROTOCOL_VERSION
            or branch_registry.get("protocol_sha256") != PROTOCOL_SHA256
            or branch_registry.get("required_assignment_count")
            != branch_authority.get("required_assignment_count")
            or status not in branch_authority.get("allowed_statuses", [])
            or not isinstance(entries, list)
            or (status == "ABSENT" and entries != [])
            or (status == "REVIEWED" and len(entries) != TOTAL_FUTURE_REPLICAS)
        ):
            raise PermissionError("Canonical branch-artifact registry authority is invalid")
    elif (
        branch_registry.get("registry_type") != "phase5_v1_branch_artifact_registry"
        or branch_registry.get("status") != "ABSENT"
        or branch_registry.get("entries") != []
    ):
        raise ValueError("Candidate r3 must not claim registered branch artifacts")
    return CandidatePreflightEvidence(
        repo=repo,
        path=preflight_dir,
        manifest_sha256=manifest_sha,
        base_manifest_sha256=sha256_file(preflight_dir / "base_seed_manifest.csv"),
        future_manifest_sha256=sha256_file(preflight_dir / "future_seed_mapping.csv"),
        historical_base_registry_sha256=sha256_file(
            preflight_dir / "historical_base_seed_registry.json"
        ),
        historical_checkpoint_registry_sha256=sha256_file(checkpoint_csv),
        base_rows=tuple(base_rows),
        future_rows=tuple(future_rows),
        historical_base_seeds=historical_seeds,
        historical_checkpoint_hashes=historical_checkpoint_hashes,
        dataset_identity=dataset_identity,
        gate_authority_registry=gate_registry,
        primary_checkpoint_registry=checkpoint_registry,
        branch_artifact_registry=branch_registry,
        primary_checkpoint_registry_sha256=sha256_file(
            preflight_dir / primary_name
        ),
        branch_artifact_registry_sha256=sha256_file(
            preflight_dir / branch_name
        ),
        trust_root_sha256=trust_root_sha256,
        _seal=seal,
    )


def load_candidate_preflight(repo: Path, preflight_dir: Path) -> CandidatePreflightEvidence:
    """Read historical or synthetic candidate evidence; never authorizes production."""
    return _load_preflight_kernel(repo, preflight_dir)


def load_canonical_preflight(repo: Path) -> CandidatePreflightEvidence:
    """Resolve canonical Phase5-v1 evidence only from the implementation-anchored root."""
    repo = repo.resolve()
    assert_frozen_protocol(repo)
    trust_root_path = repo / TRUST_ROOT_PATH
    if not trust_root_path.is_file() or sha256_file(trust_root_path) != TRUST_ROOT_SHA256:
        raise PermissionError("Implementation-anchored Phase5-v1 trust root identity mismatch")
    trust_root = _read_json(trust_root_path)
    required_root = {
        "trust_root_type": "phase5_v1_reviewed_bootstrap_trust_root",
        "trust_root_schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "canonical_preflight_path": CANONICAL_PREFLIGHT_PATH,
        "reviewed_preflight_manifest_sha256": REVIEWED_PREFLIGHT_MANIFEST_SHA256,
        "current_reviewed_state": "BLOCKED",
        "status": CANDIDATE_STATUS,
    }
    if any(trust_root.get(key) != value for key, value in required_root.items()):
        raise PermissionError("Implementation-anchored Phase5-v1 trust root is malformed")
    static_hashes = trust_root.get("static_artifact_sha256")
    dynamic_authority = trust_root.get("dynamic_registry_authority")
    if not isinstance(static_hashes, dict) or not isinstance(dynamic_authority, dict):
        raise PermissionError("Implementation-anchored Phase5-v1 trust root is incomplete")
    preflight_dir = (repo / CANONICAL_PREFLIGHT_PATH).resolve()
    if preflight_dir != (repo / str(trust_root["canonical_preflight_path"])).resolve():
        raise PermissionError("Canonical preflight path differs from the anchored trust root")
    return _load_preflight_kernel(
        repo,
        preflight_dir,
        expected_manifest_sha256=REVIEWED_PREFLIGHT_MANIFEST_SHA256,
        static_artifact_sha256=static_hashes,
        dynamic_registry_authority=dynamic_authority,
        trust_root_sha256=TRUST_ROOT_SHA256,
        seal=_CANONICAL_PREFLIGHT_SEAL,
    )


def _require_candidate(evidence: CandidatePreflightEvidence) -> CandidatePreflightEvidence:
    if not isinstance(evidence, CandidatePreflightEvidence) or evidence._seal not in {
        _PREFLIGHT_SEAL,
        _CANONICAL_PREFLIGHT_SEAL,
    }:
        raise PermissionError("Hash-validated revision-3 preflight evidence is required")
    current = (
        load_canonical_preflight(evidence.repo)
        if evidence._seal is _CANONICAL_PREFLIGHT_SEAL
        else load_candidate_preflight(evidence.repo, evidence.path)
    )
    if (
        current.manifest_sha256 != evidence.manifest_sha256
        or current.primary_checkpoint_registry_sha256
        != evidence.primary_checkpoint_registry_sha256
        or current.branch_artifact_registry_sha256 != evidence.branch_artifact_registry_sha256
        or current.trust_root_sha256 != evidence.trust_root_sha256
    ):
        raise PermissionError("Revision-3 preflight evidence changed after validation")
    return current


def _require_canonical_candidate(
    evidence: CandidatePreflightEvidence,
) -> CandidatePreflightEvidence:
    if (
        not isinstance(evidence, CandidatePreflightEvidence)
        or evidence._seal is not _CANONICAL_PREFLIGHT_SEAL
        or evidence.trust_root_sha256 != TRUST_ROOT_SHA256
    ):
        raise PermissionError("Implementation-anchored canonical Phase5-v1 evidence is required")
    return _require_candidate(evidence)


def resolve_base_manifest_row(
    evidence: CandidatePreflightEvidence, base_run_id: int, *, requested_seed: int | None = None
) -> dict[str, Any]:
    evidence = _require_candidate(evidence)
    matches = [row for row in evidence.base_rows if int(row["base_run_id"]) == int(base_run_id)]
    if len(matches) != 1:
        raise ValueError("Base identity does not resolve to exactly one frozen manifest row")
    row = dict(matches[0])
    if requested_seed is not None and int(requested_seed) != int(row["base_seed"]):
        raise ValueError("Caller-provided base seed differs from the frozen manifest")
    return row


def resolve_future_manifest_row(
    evidence: CandidatePreflightEvidence,
    base_run_id: int,
    block: str,
    replica_id: int,
    *,
    requested_seed: int | None = None,
) -> dict[str, Any]:
    evidence = _require_candidate(evidence)
    key = (int(base_run_id), str(block), int(replica_id))
    matches = [
        row
        for row in evidence.future_rows
        if (int(row["base_run_id"]), row["block"], int(row["replica_id"])) == key
    ]
    if len(matches) != 1:
        raise ValueError("Branch identity does not resolve to exactly one frozen mapping row")
    row = dict(matches[0])
    if requested_seed is not None and int(requested_seed) != int(row["future_seed"]):
        raise ValueError("Caller-provided future seed differs from the frozen mapping")
    return row


def checkpoint_nonoverlap_audit(
    primary_checkpoint_rows: Iterable[dict[str, Any]], historical_hashes: Iterable[str]
) -> dict[str, Any]:
    rows = list(primary_checkpoint_rows)
    expected_ids = set(range(1, N_STATES + 1))
    ids = [int(row["base_run_id"]) for row in rows]
    if len(rows) != N_STATES or set(ids) != expected_ids or len(set(ids)) != N_STATES:
        raise ValueError("Exactly one primary checkpoint is required for every base run 1..36")
    if any(int(row["epoch"]) != PRIMARY_CHECKPOINT_EPOCH for row in rows):
        raise ValueError("Every primary checkpoint must be epoch 14")
    hashes = [str(row["checkpoint_sha256"]) for row in rows]
    if len(set(hashes)) != N_STATES:
        raise ValueError("Primary checkpoint identities must be unique")
    overlap = sorted(set(hashes) & set(historical_hashes))
    return {
        "status": "passed" if not overlap else "failed",
        "primary_path_count": len(rows),
        "primary_unique_hash_count": len(set(hashes)),
        "overlap_count": len(overlap),
        "overlap_sha256": overlap,
    }


def block_epochs(block: str) -> list[int]:
    if block == "A":
        return list(range(15, 19))
    if block == "B":
        return list(range(15, 31))
    raise ValueError("Phase 5 block must be A or B")


def expected_learning_rates(block: str, action: str) -> list[float]:
    epochs = block_epochs(block)
    if action == "Now":
        return [DESIGN.reduced_learning_rate] * len(epochs)
    if action == "Wait":
        return [
            DESIGN.initial_learning_rate if epoch <= 17 else DESIGN.reduced_learning_rate
            for epoch in epochs
        ]
    raise ValueError("Action must be Now or Wait")


def load_production_gate_context(
    repo: Path,
    preflight_dir: Path,
    stage: str,
    *,
    independent_audit_path: Path,
    hard_gate_evidence_path: Path,
    authorization_path: Path,
) -> ProductionGateContext:
    """Issue a context only for artifacts registered below the anchored trust root.

    The current r3 registry deliberately contains no positive audit, backup, or
    authorization identities, so every canonical stage remains fail-closed.
    """
    if stage not in {"base", "branch", "analysis"}:
        raise ValueError("Execution gate stage must be base, branch, or analysis")
    repo = repo.resolve()
    canonical_path = (repo / CANONICAL_PREFLIGHT_PATH).resolve()
    if preflight_dir.resolve() != canonical_path:
        raise PermissionError("Caller-selected preflight directories cannot define canonical trust")
    raise PermissionError(
        "No reviewed independent r5 authority is configured; the r4 dynamic-registry "
        "production authority path is obsolete"
    )
    evidence = load_canonical_preflight(repo)
    trusted = evidence.gate_authority_registry.get("trusted_artifact_sha256")
    if not isinstance(trusted, dict):
        raise PermissionError("Trusted gate authority registry is incomplete")
    requested = {
        "independent_conformance_audit": independent_audit_path,
        "hard_gate_evidence": hard_gate_evidence_path,
        "execution_authorization": authorization_path,
    }
    registered_keys = {
        "independent_conformance_audit": "independent_conformance_audit",
        "hard_gate_evidence": f"hard_gate_evidence_{stage}",
        "execution_authorization": f"execution_authorization_{stage}",
    }
    for label, path in requested.items():
        registered_sha = trusted.get(registered_keys[label])
        if not isinstance(registered_sha, str) or len(registered_sha) != 64:
            raise PermissionError(
                f"No reviewed {label.replace('_', ' ')} identity is registered for {stage}"
            )
        if not path.is_file() or sha256_file(path) != registered_sha:
            raise PermissionError(
                f"Caller-selected {label.replace('_', ' ')} is not the registered artifact"
            )
    audit = _read_json(independent_audit_path)
    required_audit = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_evidence_revision": PREFLIGHT_REVISION,
        "preflight_manifest_sha256": evidence.manifest_sha256,
        "audit_type": "independent_read_only_conformance",
        "status": "PASSED",
    }
    if any(audit.get(key) != value for key, value in required_audit.items()):
        raise PermissionError("Independent conformance-audit evidence is absent or invalid")
    audit_sha = sha256_file(independent_audit_path)

    hard_gate = _read_json(hard_gate_evidence_path)
    required_hard_gate = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_manifest_sha256": evidence.manifest_sha256,
        "stage": stage,
        "status": "PASSED",
        "unresolved_hard_gates": [],
    }
    if any(hard_gate.get(key) != value for key, value in required_hard_gate.items()):
        raise PermissionError("Stage-specific hard-gate evidence is absent or unresolved")
    if hard_gate.get("independent_audit_sha256") != audit_sha:
        raise PermissionError("Hard-gate evidence is not bound to the independent audit")
    bindings = hard_gate.get("evidence_artifacts")
    if not isinstance(bindings, dict):
        raise PermissionError("Hard-gate evidence lacks provenance-bound supporting artifacts")

    def bound_json(label: str) -> tuple[Path, dict[str, Any]]:
        binding = bindings.get(label)
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise PermissionError(f"Required hard-gate artifact binding is missing: {label}")
        path = Path(binding["path"])
        if not path.is_absolute():
            path = repo.resolve() / path
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise PermissionError(f"Required hard-gate artifact hash is invalid: {label}")
        registered_sha = trusted.get(label)
        if not isinstance(registered_sha, str) or registered_sha != sha256_file(path):
            raise PermissionError(f"Hard-gate artifact is not review-registered: {label}")
        return path, _read_json(path)

    _, test_report = bound_json("implementation_test_report")
    if (
        test_report.get("status") != "PASSED"
        or test_report.get("failed") != 0
        or test_report.get("protocol_sha256") != PROTOCOL_SHA256
        or test_report.get("preflight_manifest_sha256") != evidence.manifest_sha256
    ):
        raise PermissionError("Implementation-test evidence is not valid for this protocol/evidence")
    _, seed_review = bound_json("seed_manifest_review")
    if (
        seed_review.get("status") != "PASSED"
        or seed_review.get("base_seed_manifest_sha256") != evidence.base_manifest_sha256
        or seed_review.get("future_seed_mapping_sha256") != evidence.future_manifest_sha256
        or seed_review.get("historical_exclusion_status") != "PASSED"
        or seed_review.get("global_uniqueness_status") != "PASSED"
    ):
        raise PermissionError("Seed-manifest review evidence is missing or inconsistent")
    _, backup = bound_json("backup_verification")
    if (
        backup.get("status") != "PASSED"
        or backup.get("independent_destination_verified") is not True
    ):
        raise PermissionError("Independent backup evidence is missing or inconsistent")
    working_root = Path(str(backup.get("working_root", "")))
    backup_root = Path(str(backup.get("backup_root", "")))
    if not working_root.is_absolute():
        working_root = repo.resolve() / working_root
    if not backup_root.is_absolute():
        backup_root = repo.resolve() / backup_root
    if working_root.resolve() == backup_root.resolve():
        raise PermissionError("Backup destination is not independent from the working copy")
    try:
        working_manifest = checksum_manifest(working_root)
        backup_manifest = checksum_manifest(backup_root)
    except ValueError as error:
        raise PermissionError("Independent backup roots cannot be verified") from error
    if (
        working_manifest != backup_manifest
        or canonical_sha256(working_manifest) != backup.get("working_manifest_sha256")
        or canonical_sha256(backup_manifest) != backup.get("backup_manifest_sha256")
    ):
        raise PermissionError("Backup verification must match independently recomputed files")
    if stage in {"branch", "analysis"}:
        checkpoint_audit = hard_gate.get("primary_checkpoint_nonoverlap")
        if not isinstance(checkpoint_audit, dict) or (
            checkpoint_audit.get("status") != "PASSED"
            or checkpoint_audit.get("primary_checkpoint_count") != N_STATES
            or checkpoint_audit.get("historical_checkpoint_registry_sha256")
            != evidence.historical_checkpoint_registry_sha256
        ):
            raise PermissionError("Complete primary-checkpoint non-overlap evidence is required")
        if hard_gate.get("checkpoint_restoration_audit") != "PASSED":
            raise PermissionError("Checkpoint restoration audit evidence is required")
        _, checkpoint_report = bound_json("primary_checkpoint_audit")
        checkpoint_rows = checkpoint_report.get("primary_checkpoints")
        if not isinstance(checkpoint_rows, list):
            raise PermissionError("Primary-checkpoint audit lacks checkpoint rows")
        registered_rows = evidence.primary_checkpoint_registry.get("entries")
        if (
            evidence.primary_checkpoint_registry.get("status") != "APPROVED"
            or checkpoint_rows != registered_rows
        ):
            raise PermissionError("Checkpoint audit does not resolve the reviewed approved set")
        for row in checkpoint_rows:
            path = Path(row.get("checkpoint_path", ""))
            if not path.is_absolute():
                path = repo.resolve() / path
            if not path.is_file() or sha256_file(path) != row.get("checkpoint_sha256"):
                raise PermissionError("Primary-checkpoint audit does not match actual checkpoint bytes")
        recomputed = checkpoint_nonoverlap_audit(
            checkpoint_rows, evidence.historical_checkpoint_hashes
        )
        if recomputed["status"] != "passed":
            raise PermissionError("Primary checkpoints overlap historical checkpoint identities")
        _, restoration = bound_json("checkpoint_restoration_audit")
        restoration_rows = restoration.get("states")
        if (
            not isinstance(restoration_rows, list)
            or {int(row.get("base_run_id", -1)) for row in restoration_rows}
            != set(range(1, N_STATES + 1))
            or any(row.get("status") != "PASSED" for row in restoration_rows)
        ):
            raise PermissionError("Complete checkpoint-restoration evidence is required")
    if stage == "analysis":
        order_audit = hard_gate.get("complete_order_stream_audit")
        if not isinstance(order_audit, dict) or (
            order_audit.get("status") != "PASSED"
            or order_audit.get("assignment_count") != TOTAL_FUTURE_REPLICAS
        ):
            raise PermissionError("Complete 900-assignment order-stream audit evidence is required")
        _, order_report = bound_json("complete_order_stream_audit")
        audit_future_order_records(order_report.get("records", []), require_complete=True)
    hard_gate_sha = sha256_file(hard_gate_evidence_path)

    authorization = _read_json(authorization_path)
    required_authorization = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_manifest_sha256": evidence.manifest_sha256,
        "stage": stage,
        "authorization": "AUTHORIZED",
        "independent_audit_sha256": audit_sha,
        "hard_gate_evidence_sha256": hard_gate_sha,
    }
    if any(authorization.get(key) != value for key, value in required_authorization.items()):
        raise PermissionError("Separate user execution authorization is absent or invalid")
    if authorization.get("registry_sha256") != sha256_file(
        evidence.path / "gate_authority_registry.json"
    ):
        raise PermissionError("Execution authorization is not bound to the reviewed authority registry")
    return ProductionGateContext(
        stage=stage,
        preflight=evidence,
        independent_audit_sha256=audit_sha,
        hard_gate_evidence_sha256=hard_gate_sha,
        authorization_sha256=sha256_file(authorization_path),
        independent_audit_path=independent_audit_path.resolve(),
        hard_gate_evidence_path=hard_gate_evidence_path.resolve(),
        authorization_path=authorization_path.resolve(),
        _seal=_GATE_SEAL,
    )


def validate_execution_gate(
    receipt: Any, stage: str, snapshot: dict[str, Any] | None = None
) -> Any:
    """Require the r5 gate; r4 receipts and caller assertions have no authority."""
    if stage not in {"base", "branch", "analysis"}:
        raise ValueError("Execution gate stage must be base, branch, or analysis")
    from .phase5_evidence import R5ExecutionGateContext, validate_r5_execution_gate

    if isinstance(receipt, R5ExecutionGateContext):
        return validate_r5_execution_gate(receipt, stage, snapshot)
    if not isinstance(receipt, ProductionGateContext) or receipt._seal is not _GATE_SEAL:
        raise PermissionError(
            f"Phase 5 {stage} execution requires provenance-bound gate evidence; receipts are insufficient"
        )
    raise PermissionError(
        "Phase 5 execution requires provenance-bound r5 authority; r4 receipts are obsolete"
    )


def _run_pair_kernel(
    checkpoint: dict,
    training_dataset,
    block: str,
    future_seed: int,
) -> dict[str, Any]:
    """Private deterministic pair kernel; it cannot register canonical artifacts."""
    if int(checkpoint.get("epoch", -1)) != PRIMARY_CHECKPOINT_EPOCH:
        raise ValueError("Phase 5 branches require an epoch-14 primary checkpoint")
    end_epoch = block_epochs(block)[-1]
    seeded = with_future_shuffle(checkpoint, future_seed)
    original = deepcopy(seeded)
    common = BranchingConfig()
    now = run_branch(seeded, training_dataset, common, 0.5, end_epoch=end_epoch)
    wait = run_branch(
        seeded,
        training_dataset,
        common,
        0.5,
        expected_initial=now["before"],
        delay_epochs=3,
        end_epoch=end_epoch,
    )
    if not states_equal(now["before"], wait["before"]):
        raise AssertionError("Now/Wait initial states differ")
    if now["orders"] != wait["orders"]:
        raise AssertionError("Now/Wait actual minibatch orders differ")
    if not states_equal(now["epoch_start_random_states"], wait["epoch_start_random_states"]):
        raise AssertionError("Now/Wait epoch-start RNG states differ")
    if not states_equal(now["final_state"]["rng"], wait["final_state"]["rng"]):
        raise AssertionError("Now/Wait final global RNG states differ")
    if not states_equal(now["final_state"]["generator"], wait["final_state"]["generator"]):
        raise AssertionError("Now/Wait final loader streams differ")
    if not states_equal(seeded, original):
        raise AssertionError("Source checkpoint was mutated")
    epochs = block_epochs(block)
    if [row["epoch"] for row in now["history"]] != epochs:
        raise AssertionError("Now coverage differs from the frozen block boundary")
    if [row["epoch"] for row in wait["history"]] != epochs:
        raise AssertionError("Wait coverage differs from the frozen block boundary")
    if [row["learning_rate"] for row in now["history"]] != expected_learning_rates(block, "Now"):
        raise AssertionError("Now schedule differs from Phase5-v1")
    if [row["learning_rate"] for row in wait["history"]] != expected_learning_rates(block, "Wait"):
        raise AssertionError("Wait schedule differs from Phase5-v1")
    return {
        "block": block,
        "future_seed": int(future_seed),
        "now": now,
        "wait": wait,
        "audit": {
            "initial_state_equal": True,
            "actual_orders_equal": True,
            "source_checkpoint_immutable": True,
            "now_order_sha256": order_hashes(now["orders"]),
            "wait_order_sha256": order_hashes(wait["orders"]),
            "epoch_numbers": epochs,
        },
    }


def run_synthetic_pair(
    checkpoint: dict, training_dataset, block: str, future_seed: int
) -> dict[str, Any]:
    """Run a tiny test pair in a namespace that cannot be canonical Phase5-v1."""
    result = _run_pair_kernel(checkpoint, training_dataset, block, int(future_seed))
    result.update(
        {
            "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
            "artifact_class": "synthetic_test_fixture",
            "canonical_phase5_observation": False,
        }
    )
    return result


def validate_registered_checkpoint_member(
    checkpoint_path: Path,
    primary_state: dict[str, Any],
    registry: dict[str, Any],
    historical_checkpoint_hashes: Iterable[str],
    *,
    expected_base_run_id: int,
    expected_protocol_manifest_sha256: str,
    canonical_registry: bool,
) -> str:
    """Resolve actual checkpoint bytes through an exact registered-set member."""
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise ValueError("Actual Phase 5 checkpoint file is missing")
    actual = sha256_file(checkpoint_path)
    if primary_state.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Primary checkpoint provenance has the wrong protocol version")
    if primary_state.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Primary checkpoint provenance has the wrong protocol hash")
    if int(primary_state.get("base_run_id", -1)) != int(expected_base_run_id):
        raise ValueError("Primary checkpoint belongs to a different base_run_id")
    if int(primary_state.get("epoch", -1)) != PRIMARY_CHECKPOINT_EPOCH:
        raise ValueError("Primary checkpoint is not the frozen epoch-14 state")
    if Path(primary_state.get("checkpoint_path", "")).resolve() != checkpoint_path:
        raise ValueError("Primary-state checkpoint path does not bind the actual file")
    if primary_state.get("checkpoint_sha256") != actual:
        raise ValueError("Caller-supplied checkpoint hash cannot substitute for the actual file hash")
    if actual in set(historical_checkpoint_hashes):
        raise ValueError("Phase 5 primary checkpoint overlaps the historical checkpoint registry")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Primary checkpoint registry has no registered entries")
    matches = [row for row in entries if row.get("checkpoint_sha256") == actual]
    if len(matches) != 1:
        raise ValueError("Actual checkpoint hash is not a unique approved-set member")
    registered = matches[0]
    registered_path = Path(str(registered.get("checkpoint_path", "")))
    if not registered_path.is_absolute():
        registered_path = checkpoint_path.parent / registered_path
    if registered_path.resolve() != checkpoint_path:
        raise ValueError("Approved checkpoint registry path does not identify the actual file")
    required_registered = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_manifest_sha256": expected_protocol_manifest_sha256,
        "base_run_id": int(expected_base_run_id),
        "epoch": PRIMARY_CHECKPOINT_EPOCH,
    }
    if any(registered.get(key) != value for key, value in required_registered.items()):
        raise ValueError("Approved checkpoint entry has the wrong base/epoch/protocol identity")
    if canonical_registry:
        if (
            registry.get("registry_type") != "phase5_v1_primary_checkpoint_registry"
            or registry.get("status") != "APPROVED"
            or registry.get("protocol_version") != PROTOCOL_VERSION
            or registry.get("protocol_sha256") != PROTOCOL_SHA256
            or registry.get("preflight_manifest_sha256") != expected_protocol_manifest_sha256
            or len(entries) != N_STATES
        ):
            raise ValueError("Exact approved 36-checkpoint registry is unavailable")
        checkpoint_nonoverlap_audit(entries, historical_checkpoint_hashes)
    return actual


def validate_checkpoint_for_branch(
    checkpoint_path: Path,
    primary_state: dict[str, Any],
    evidence: CandidatePreflightEvidence,
    *,
    expected_base_run_id: int,
) -> str:
    """Canonical wrapper: registry membership, not caller metadata, is authoritative."""
    evidence = _require_canonical_candidate(evidence)
    base_row = resolve_base_manifest_row(evidence, expected_base_run_id)
    if int(primary_state.get("base_seed", -1)) != int(base_row["base_seed"]):
        raise ValueError("Primary checkpoint base seed differs from the frozen manifest")
    if primary_state.get("base_seed_manifest_sha256") != evidence.base_manifest_sha256:
        raise ValueError("Primary checkpoint is not bound to the frozen base manifest")
    return validate_registered_checkpoint_member(
        checkpoint_path,
        primary_state,
        evidence.primary_checkpoint_registry,
        evidence.historical_checkpoint_hashes,
        expected_base_run_id=expected_base_run_id,
        expected_protocol_manifest_sha256=evidence.manifest_sha256,
        canonical_registry=True,
    )


def phase5_pair(
    checkpoint_path: Path,
    training_dataset,
    base_run_id: int,
    block: str,
    replica_id: int,
    *,
    primary_state: dict[str, Any],
    execution_gate: Any,
    checkpoint_snapshot: dict[str, Any] | None = None,
    attempt_ledger_path: Path | None = None,
    artifact_path: Path | None = None,
    attempt_id: str | None = None,
    retry_of: str | None = None,
) -> dict[str, Any]:
    """Canonical r5 branch entry point; outputs remain non-authoritative working evidence."""
    from .phase5_evidence import (
        append_working_event,
        canonical_sha256 as r5_canonical_sha256,
        current_r5_implementation_identity,
        expected_branch_identities,
        load_r5_static_baseline,
        load_working_ledger,
        resolve_r5_retry_identity,
        sha256_file as r5_sha256_file,
        validate_attempt_chronology,
        validate_checkpoint_for_branch_r5,
    )

    if checkpoint_snapshot is None:
        raise PermissionError("Canonical branching requires a specifically authorized checkpoint snapshot")
    gate = validate_execution_gate(execution_gate, "branch", checkpoint_snapshot)
    if gate.authority_domain != "production":
        raise PermissionError("Synthetic r5 authority cannot enter canonical branching")
    if attempt_ledger_path is None or artifact_path is None or not attempt_id:
        raise ValueError("Canonical branching requires attempt ledger, artifact path, and attempt ID")
    baseline = load_r5_static_baseline(gate.repo)
    evidence = load_canonical_preflight(gate.repo)
    validate_phase5_training_dataset(training_dataset, evidence)
    mapping = resolve_future_manifest_row(evidence, base_run_id, block, replica_id)
    checkpoint_sha = validate_checkpoint_for_branch_r5(
        checkpoint_path, base_run_id, checkpoint_snapshot, gate
    )
    identities = expected_branch_identities(baseline, checkpoint_snapshot)
    scientific_identity = identities[(int(base_run_id), str(block), int(replica_id))]
    prior_events = load_working_ledger(attempt_ledger_path)
    if retry_of is None:
        chronology = validate_attempt_chronology(
            prior_events,
            protected_lineage=gate.provider,
            ledger_id="phase5-branch-attempt-ledger",
            repo=gate.repo,
        )
        if r5_canonical_sha256(scientific_identity) in chronology["latest_by_identity"]:
            raise ValueError("A later branch attempt must use authenticated retry lineage")
    else:
        resolved_identity, predecessor = resolve_r5_retry_identity(
            baseline,
            checkpoint_snapshot,
            prior_events,
            gate,
            base_run_id=base_run_id,
            block=block,
            replica_id=replica_id,
        )
        if resolved_identity != scientific_identity or predecessor != retry_of:
            raise ValueError("Retry predecessor differs from authenticated attempt lineage")
    start_type = "retry_started" if retry_of is not None else "attempt_started"
    append_working_event(
        attempt_ledger_path,
        {
            "event_id": f"{attempt_id}:start",
            "attempt_id": attempt_id,
            "event_type": start_type,
            "predecessor_attempt_id": retry_of,
            "scientific_identity": scientific_identity,
        },
    )
    try:
        checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
        result = _run_pair_kernel(
            checkpoint, training_dataset, block, int(mapping["future_seed"])
        )
        divergence_rows = []
        for action_key, action_name in (("now", "Now"), ("wait", "Wait")):
            for history_row in result[action_key]["history"]:
                for field in ("train_loss", "val_loss"):
                    value = float(history_row[field])
                    if not math.isfinite(value):
                        divergence_rows.append(
                            {
                                "base_run_id": int(base_run_id),
                                "block": block,
                                "replica_id": int(replica_id),
                                "action": action_name,
                                "epoch": int(history_row["epoch"]),
                                "nonfinite_field": field,
                            }
                        )
            for state_row in result[action_key]["state_finiteness_evidence"]:
                for category in ("model_parameters", "optimizer_state"):
                    for tensor_row in state_row[category]:
                        if int(tensor_row["nonfinite_count"]) > 0:
                            divergence_rows.append(
                                {
                                    "base_run_id": int(base_run_id),
                                    "block": block,
                                    "replica_id": int(replica_id),
                                    "action": action_name,
                                    "epoch": int(state_row["epoch"]),
                                    "nonfinite_field": category,
                                    "evidence_path": tensor_row["path"],
                                    "nonfinite_count": int(tensor_row["nonfinite_count"]),
                                }
                            )
        if block == "B":
            now_mean = float(np.mean([row["val_loss"] for row in result["now"]["history"][-3:]]))
            wait_mean = float(np.mean([row["val_loss"] for row in result["wait"]["history"][-3:]]))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                gain = float((wait_mean - now_mean) / max(wait_mean, 1e-12))
            if not math.isfinite(gain):
                divergence_rows.append(
                    {
                        "base_run_id": int(base_run_id),
                        "block": block,
                        "replica_id": int(replica_id),
                        "action": "derived_W_minus_N",
                        "epoch": "28..30",
                        "nonfinite_field": "G",
                    }
                )

        def json_safe(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                if math.isnan(value):
                    return {"__nonfinite_float__": "NaN"}
                return {"__nonfinite_float__": "+Inf" if value > 0 else "-Inf"}
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            return value

        terminal_state = (
            "substantive_divergence" if divergence_rows else "terminal_success"
        )
        artifact = {
            "artifact_type": "phase5_v1_r5_branch_working_artifact",
            "schema_version": 3,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_identity": current_r5_implementation_identity(gate.repo),
            "preflight_manifest_sha256": evidence.manifest_sha256,
            "future_manifest_sha256": evidence.future_manifest_sha256,
            "scientific_identity": scientific_identity,
            "attempt_id": attempt_id,
            "terminal_state": terminal_state,
            "now_history": json_safe(result["now"]["history"]),
            "wait_history": json_safe(result["wait"]["history"]),
            "now_state_finiteness_evidence": result["now"]["state_finiteness_evidence"],
            "wait_state_finiteness_evidence": result["wait"]["state_finiteness_evidence"],
            "divergence_rows": divergence_rows,
            "now_realized_orders": result["now"]["orders"],
            "wait_realized_orders": result["wait"]["orders"],
            "now_epoch_order_sha256": result["audit"]["now_order_sha256"],
            "wait_epoch_order_sha256": result["audit"]["wait_order_sha256"],
            "execution_audit": {
                "initial_state_equal": result["audit"]["initial_state_equal"],
                "actual_orders_equal": result["audit"]["actual_orders_equal"],
                "source_checkpoint_immutable": result["audit"]["source_checkpoint_immutable"],
                "epoch_numbers": result["audit"]["epoch_numbers"],
            },
        }
        if artifact_path.exists():
            raise FileExistsError("Refuse to overwrite a branch working artifact")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        artifact_sha = r5_sha256_file(artifact_path)
        append_working_event(
            attempt_ledger_path,
            {
                "event_id": f"{attempt_id}:artifact",
                "attempt_id": attempt_id,
                "event_type": "artifact_produced",
                "predecessor_attempt_id": retry_of,
                "scientific_identity": scientific_identity,
                "artifact_path": str(artifact_path.resolve()),
                "artifact_sha256": artifact_sha,
            },
        )
        append_working_event(
            attempt_ledger_path,
            {
                "event_id": f"{attempt_id}:success",
                "attempt_id": attempt_id,
                "event_type": terminal_state,
                "predecessor_attempt_id": retry_of,
                "scientific_identity": scientific_identity,
                "artifact_sha256": artifact_sha,
            },
        )
    except Exception as error:
        append_working_event(
            attempt_ledger_path,
            {
                "event_id": f"{attempt_id}:failure",
                "attempt_id": attempt_id,
                "event_type": "technical_failure",
                "predecessor_attempt_id": retry_of,
                "scientific_identity": scientific_identity,
                "error_type": type(error).__name__,
            },
        )
        raise
    result.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "r5_static_baseline_identity": baseline.baseline_identity,
            "checkpoint_snapshot_identity": checkpoint_snapshot["snapshot_identity"],
            "base_run_id": int(base_run_id),
            "replica_id": int(replica_id),
            "checkpoint_sha256": checkpoint_sha,
            "future_seed": int(mapping["future_seed"]),
            "future_seed_namespace": mapping["namespace"],
            "future_seed_counter": int(mapping["counter"]),
            "future_manifest_sha256": evidence.future_manifest_sha256,
            "preflight_manifest_sha256": evidence.manifest_sha256,
            "future_stream_identity": scientific_identity["future_stream_identity"],
            "realized_order_stream_sha256": r5_canonical_sha256(
                result["audit"]["now_order_sha256"]
            ),
            "attempt_id": attempt_id,
            "artifact_path": str(artifact_path.resolve()),
            "artifact_sha256": artifact_sha,
            "terminal_state": terminal_state,
            "divergence_rows": divergence_rows,
            "artifact_class": "r5_working_evidence_only",
            "canonical_phase5_observation": False,
        }
    )
    return result


def order_sequence_sha256(epoch_order_hashes: list[str]) -> str:
    return canonical_sha256(epoch_order_hashes)


def audit_future_order_records(
    records: Iterable[dict[str, Any]], *, require_complete: bool = False
) -> dict[str, Any]:
    records = list(records)
    keys: set[tuple[int, str, int]] = set()
    sequences: set[str] = set()
    for row in records:
        key = (int(row["base_run_id"]), str(row["block"]), int(row["replica_id"]))
        if key in keys:
            raise ValueError("Duplicate state/block/replica order record")
        keys.add(key)
        if row["now_order_sha256"] != row["wait_order_sha256"]:
            raise ValueError("Within-pair minibatch-order hashes differ")
        now_hashes = row["now_order_sha256"]
        if not isinstance(now_hashes, list) or not now_hashes:
            raise ValueError("Order evidence must contain epoch-level hash lists")
        sequence = order_sequence_sha256(now_hashes)
        if sequence in sequences:
            raise ValueError("Future order sequence was reused across assignments")
        sequences.add(sequence)
    expected = {
        (base, block, replica)
        for base in range(1, N_STATES + 1)
        for block, count in (("A", K_A), ("B", K_B))
        for replica in range(1, count + 1)
    }
    complete = keys == expected
    if require_complete and not complete:
        missing = len(expected - keys)
        unexpected = len(keys - expected)
        raise ValueError(
            f"Complete Phase5-v1 order audit requires all 900 assignments; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "status": "passed" if complete else "partial_evidence_only",
        "audit_scope": "complete_phase5_v1_assignment_set" if complete else "partial",
        "assignment_count": len(keys),
        "expected_assignment_count": TOTAL_FUTURE_REPLICAS,
        "unique_sequence_count": len(sequences),
        "complete": complete,
    }


def canonical_phase5_base_config(base_seed: int) -> ExperimentConfig:
    """Return the production Phase5-v1 config for an authorized base seed."""
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("Phase 5 base seed must be an integer")
    return ExperimentConfig(seed=base_seed)


def _run_base_kernel(
    base_row: dict[str, Any],
    training_dataset,
    output_dir: Path,
    *,
    fixture_config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Private base-training kernel; public wrappers determine artifact namespace."""
    base_run_id = int(base_row["base_run_id"])
    if not 1 <= base_run_id <= N_STATES:
        raise ValueError("Invalid base_run_id")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = canonical_phase5_base_config(int(base_row["base_seed"]))
    if fixture_config is not None:
        config = replace(fixture_config, seed=int(base_row["base_seed"]))
    set_reproducible_seed(config.seed)
    data = make_loaders_from_datasets(training_dataset, [], config)
    model = FashionMLP(config.hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, PRIMARY_CHECKPOINT_EPOCH + 1):
        loss = train_one_epoch(model, data.train_loader, optimizer, criterion, torch.device("cpu"))
        validation = evaluate(model, data.val_loader, criterion, torch.device("cpu"))
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_loss": validation.loss,
                "val_accuracy": validation.accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    checkpoint_path = output_dir / "epoch_014.pt"
    save_checkpoint(
        checkpoint_path,
        PRIMARY_CHECKPOINT_EPOCH,
        model,
        optimizer,
        config,
        data.train_generator,
        data.split_fingerprint,
        history,
    )
    state = {
        "base_run_id": base_run_id,
        "base_seed": config.seed,
        "epoch": PRIMARY_CHECKPOINT_EPOCH,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_fingerprint": data.split_fingerprint,
        "state_manifested_before_future_outcomes": True,
    }
    write_json(output_dir / "primary_state.json", state)
    return state


def run_synthetic_base_fixture(
    base_seed: int,
    training_dataset,
    output_dir: Path,
    *,
    fixture_config: ExperimentConfig,
) -> dict[str, Any]:
    """Create only a clearly non-canonical tiny test artifact."""
    row = {"base_run_id": 1, "base_seed": int(base_seed)}
    state = _run_base_kernel(row, training_dataset, output_dir, fixture_config=fixture_config)
    state.update(
        {
            "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
            "artifact_class": "synthetic_test_fixture",
            "canonical_phase5_observation": False,
        }
    )
    write_json(output_dir / "primary_state.json", state)
    return state


def run_base_to_primary_state(
    base_run_id: int,
    training_dataset,
    output_dir: Path,
    *,
    execution_gate: Any,
    attempt_ledger_path: Path | None = None,
    attempt_id: str | None = None,
    retry_of: str | None = None,
) -> dict[str, Any]:
    """Canonical r5 base entry point; success creates working evidence only."""
    from .phase5_evidence import (
        append_working_event,
        canonical_sha256 as r5_canonical_sha256,
        expected_base_identities,
        load_r5_static_baseline,
        load_working_ledger,
        validate_attempt_chronology,
    )

    gate = validate_execution_gate(execution_gate, "base")
    if gate.authority_domain != "production":
        raise PermissionError("Synthetic r5 authority cannot enter canonical base execution")
    if attempt_ledger_path is None or not attempt_id:
        raise ValueError("Canonical base execution requires attempt ledger and attempt ID")
    baseline = load_r5_static_baseline(gate.repo)
    evidence = load_canonical_preflight(gate.repo)
    validate_phase5_training_dataset(training_dataset, evidence)
    base_row = resolve_base_manifest_row(evidence, int(base_run_id))
    scientific_identity = expected_base_identities(baseline)[int(base_run_id)]
    prior_events = load_working_ledger(attempt_ledger_path)
    chronology = validate_attempt_chronology(
        prior_events,
        protected_lineage=gate.provider,
        ledger_id="phase5-base-attempt-ledger",
        repo=gate.repo,
    )
    identity_sha = r5_canonical_sha256(scientific_identity)
    predecessor = chronology["latest_by_identity"].get(identity_sha)
    if retry_of is None:
        if predecessor is not None:
            raise ValueError("A later base attempt must use authenticated retry lineage")
    elif (
        predecessor != retry_of
        or chronology["attempts"][predecessor]["terminal"] != "technical_failure"
    ):
        raise ValueError("Base retry requires an authenticated terminal technical failure")
    append_working_event(
        attempt_ledger_path,
        {
            "event_id": f"{attempt_id}:start",
            "attempt_id": attempt_id,
            "event_type": "retry_started" if retry_of is not None else "attempt_started",
            "predecessor_attempt_id": retry_of,
            "scientific_identity": scientific_identity,
        },
    )
    try:
        state = _run_base_kernel(base_row, training_dataset, output_dir)
    except Exception as error:
        append_working_event(
            attempt_ledger_path,
            {
                "event_id": f"{attempt_id}:failure",
                "attempt_id": attempt_id,
                "event_type": "technical_failure",
                "predecessor_attempt_id": retry_of,
                "scientific_identity": scientific_identity,
                "error_type": type(error).__name__,
            },
        )
        raise
    state.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "r5_static_baseline_identity": baseline.baseline_identity,
            "base_seed_namespace": base_row["namespace"],
            "base_seed_counter": int(base_row["counter"]),
            "base_seed_manifest_sha256": evidence.base_manifest_sha256,
            "preflight_manifest_sha256": evidence.manifest_sha256,
            "dataset_identity_sha256": r5_canonical_sha256(baseline.dataset_identity),
            "scientific_identity_sha256": r5_canonical_sha256(scientific_identity),
            "attempt_id": attempt_id,
            "artifact_class": "r5_working_evidence_only",
            "canonical_phase5_observation": False,
        }
    )
    write_json(output_dir / "primary_state.json", state)
    append_working_event(
        attempt_ledger_path,
        {
            "event_id": f"{attempt_id}:artifact",
            "attempt_id": attempt_id,
            "event_type": "artifact_produced",
            "predecessor_attempt_id": retry_of,
            "scientific_identity": scientific_identity,
            "checkpoint_path": state["checkpoint_path"],
            "checkpoint_sha256": state["checkpoint_sha256"],
        },
    )
    append_working_event(
        attempt_ledger_path,
        {
            "event_id": f"{attempt_id}:success",
            "attempt_id": attempt_id,
            "event_type": "terminal_success",
            "predecessor_attempt_id": retry_of,
            "scientific_identity": scientific_identity,
            "checkpoint_sha256": state["checkpoint_sha256"],
        },
    )
    return state


def validate_primary_state_rows(rows: Iterable[dict[str, Any]], base_rows: list[dict[str, Any]]) -> None:
    rows = list(rows)
    if len(rows) != N_STATES:
        raise ValueError("Exactly 36 primary-state rows are required")
    base_lookup = {int(row["base_run_id"]): int(row["base_seed"]) for row in base_rows}
    seen = set()
    for row in rows:
        key = int(row["base_run_id"])
        if key in seen or key not in base_lookup:
            raise ValueError("Primary-state base_run_id is missing or duplicated")
        seen.add(key)
        if int(row["base_seed"]) != base_lookup[key] or int(row["epoch"]) != 14:
            raise ValueError("Primary state does not match the frozen base manifest")
        if row.get("state_manifested_before_future_outcomes") is not True:
            raise ValueError("Primary state lacks pre-outcome provenance")


def per_replica_gain(now_final_losses: np.ndarray, wait_final_losses: np.ndarray) -> np.ndarray:
    now = np.asarray(now_final_losses, dtype=float)
    wait = np.asarray(wait_final_losses, dtype=float)
    if now.shape != wait.shape or now.ndim < 1 or now.shape[-1] != 3:
        raise ValueError("B records must provide paired epoch 28-30 loss triples")
    if not np.isfinite(now).all() or not np.isfinite(wait).all():
        raise ValueError("Required B outcome is non-finite")
    if np.any(now < 0) or np.any(wait < 0):
        raise ValueError("Validation loss cannot be negative")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean_now = now.mean(axis=-1)
        mean_wait = wait.mean(axis=-1)
        gains = (mean_wait - mean_now) / np.maximum(mean_wait, 1e-12)
    if not np.isfinite(gains).all():
        raise ValueError("Calculated required B gain G is non-finite")
    return gains


def primary_estimates(
    a_delta: np.ndarray,
    b_now_final_losses: np.ndarray,
    b_wait_final_losses: np.ndarray,
    *,
    require_frozen_shape: bool = True,
) -> dict[str, Any]:
    a = np.asarray(a_delta, dtype=float)
    now = np.asarray(b_now_final_losses, dtype=float)
    wait = np.asarray(b_wait_final_losses, dtype=float)
    if a.ndim != 3 or a.shape[-1] != 4:
        raise ValueError("A must retain complete four-horizon trajectories")
    if now.shape != wait.shape or now.ndim != 3 or now.shape[-1] != 3:
        raise ValueError("B must retain paired epoch 28-30 records")
    if a.shape[0] != now.shape[0]:
        raise ValueError("A and B state axes differ")
    if require_frozen_shape and (a.shape != (N_STATES, K_A, 4) or now.shape != (N_STATES, K_B, 3)):
        raise ValueError("Primary arrays differ from N=36, K_A=15, K_B=10")
    if not np.isfinite(a).all():
        raise ValueError("Required A outcome is non-finite")
    gains = per_replica_gain(now, wait)
    d = a[:, :, :3].mean(axis=2)
    rho = d.mean(axis=1)
    tau = gains.mean(axis=1)
    n = a.shape[0]
    if n < 2 or gains.shape[1] < 2:
        raise ValueError("Covariance and deconvolution require at least two states and B replicas")
    psi = float(np.sum((rho - rho.mean()) * (tau - tau.mean())) / (n - 1))
    within_variance = gains.var(axis=1, ddof=1)
    sigma_tau2 = float(tau.var(ddof=1) - np.mean(within_variance / gains.shape[1]))
    return {
        "rho_hat": rho,
        "tau_hat": tau,
        "per_replica_g": gains,
        "mu_rho_hat": float(rho.mean()),
        "A_34_hat": float((a[:, :, 3] - a[:, :, 2]).mean(axis=1).mean()),
        "mu_h_hat": a[:, :, :3].mean(axis=1).mean(axis=0),
        "psi_hat": psi,
        "sigma_tau2_hat": sigma_tau2,
        "sigma_tau_display": math.sqrt(max(0.0, sigma_tau2)),
        "within_tau_variance": within_variance,
    }


def evaluability_status(
    a_delta: np.ndarray, b_now_final_losses: np.ndarray, b_wait_final_losses: np.ndarray
) -> dict[str, str]:
    a_finite = np.isfinite(np.asarray(a_delta, dtype=float)).all()
    b_input_finite = (
        np.isfinite(np.asarray(b_now_final_losses, dtype=float)).all()
        and np.isfinite(np.asarray(b_wait_final_losses, dtype=float)).all()
    )
    b_finite = b_input_finite
    if b_input_finite:
        try:
            per_replica_gain(b_now_final_losses, b_wait_final_losses)
        except ValueError:
            b_finite = False
    return {
        "phase5a": EVALUABLE if a_finite else NON_EVALUABLE,
        "rho": EVALUABLE if a_finite else NON_EVALUABLE,
        "tau": EVALUABLE if b_finite else NON_EVALUABLE,
        "sigma_tau2": EVALUABLE if b_finite else NON_EVALUABLE,
        "psi": EVALUABLE if a_finite and b_finite else NON_EVALUABLE,
    }


def classify_primary_claims(
    estimates: dict[str, Any],
    intervals: dict[str, tuple[float, float] | np.ndarray],
    *,
    evaluability: dict[str, str] | None = None,
) -> dict[str, Any]:
    if evaluability is not None and evaluability.get("phase5a") == NON_EVALUABLE:
        phase5b = NON_EVALUABLE if evaluability.get("psi") == NON_EVALUABLE else "UNCLASSIFIED"
        return {
            "phase5a": NON_EVALUABLE,
            "phase5a_failed_conditions": [],
            "phase5b": phase5b,
            "omnibus_claim": "prohibited",
        }
    mu_low, mu_high = map(float, intervals["mu_rho_hat"])
    a34_low, a34_high = map(float, intervals["A_34_hat"])
    psi_low, psi_high = map(float, intervals["psi_hat"])
    horizons = np.asarray(estimates["mu_h_hat"], dtype=float)
    conditions = [
        ("mu_rho 95% CI entirely > 0", mu_low > 0),
        ("A_34 95% CI entirely < 0", a34_high < 0),
        ("mu_1 > 0", horizons[0] > 0),
        ("mu_2 > 0", horizons[1] > 0),
        ("mu_3 > 0", horizons[2] > 0),
    ]
    failed = [name for name, passed in conditions if not passed]
    summary_supported = conditions[0][1] and conditions[1][1]
    concordant = all(passed for _, passed in conditions[2:])
    if summary_supported and concordant:
        phase5a = "5A signature supported"
    elif summary_supported:
        phase5a = "summary estimands supported, but horizon-level directional concordance was not fully reproduced"
    else:
        phase5a = "5A signature not fully supported"
    if evaluability is not None and evaluability.get("psi") == NON_EVALUABLE:
        phase5b = NON_EVALUABLE
    elif psi_low > 0:
        phase5b = "positive state-level response-effect association"
    elif psi_high < 0:
        phase5b = (
            "negative state-level association: stronger positive short response corresponds to "
            "weaker or more negative historical final intervention effect"
        )
    else:
        phase5b = (
            "no sufficiently precise evidence of the preregistered linear state-level association "
            "at the achieved Phase 5 precision"
        )
    return {
        "phase5a": phase5a,
        "phase5a_failed_conditions": failed,
        "phase5b": phase5b,
        "omnibus_claim": "prohibited",
    }


def bootstrap_resample_indices(
    rng: np.random.Generator, n_states: int, k_a: int, k_b: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer = rng.integers(0, n_states, size=n_states)
    # Each outer occurrence gets separate draws, including repeated source states.
    inner_a = rng.integers(0, k_a, size=(n_states, k_a))
    inner_b = rng.integers(0, k_b, size=(n_states, k_b))
    return outer, inner_a, inner_b


def nested_bootstrap(
    a_delta: np.ndarray,
    b_now_final_losses: np.ndarray,
    b_wait_final_losses: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    return_first_trace: bool = False,
) -> dict[str, Any]:
    a = np.asarray(a_delta, dtype=float)
    now = np.asarray(b_now_final_losses, dtype=float)
    wait = np.asarray(b_wait_final_losses, dtype=float)
    # Validate semantics without imposing full design dimensions on unit fixtures.
    primary_estimates(a, now, wait, require_frozen_shape=False)
    n, k_a, _ = a.shape
    k_b = now.shape[1]
    rng = np.random.Generator(np.random.PCG64(seed))
    names = ("mu_rho_hat", "A_34_hat", "mu_1_hat", "mu_2_hat", "mu_3_hat", "psi_hat", "sigma_tau2_hat")
    draws = {name: np.empty(replicates, dtype=float) for name in names}
    trace = None
    for draw in range(replicates):
        outer, inner_a, inner_b = bootstrap_resample_indices(rng, n, k_a, k_b)
        sampled_a = np.empty((n, k_a, 4), dtype=float)
        sampled_now = np.empty((n, k_b, 3), dtype=float)
        sampled_wait = np.empty((n, k_b, 3), dtype=float)
        for occurrence, source_state in enumerate(outer):
            sampled_a[occurrence] = a[source_state, inner_a[occurrence], :]
            sampled_now[occurrence] = now[source_state, inner_b[occurrence], :]
            sampled_wait[occurrence] = wait[source_state, inner_b[occurrence], :]
        estimate = primary_estimates(sampled_a, sampled_now, sampled_wait, require_frozen_shape=False)
        draws["mu_rho_hat"][draw] = estimate["mu_rho_hat"]
        draws["A_34_hat"][draw] = estimate["A_34_hat"]
        draws["mu_1_hat"][draw], draws["mu_2_hat"][draw], draws["mu_3_hat"][draw] = estimate["mu_h_hat"]
        draws["psi_hat"][draw] = estimate["psi_hat"]
        # Deliberately untruncated.
        draws["sigma_tau2_hat"][draw] = estimate["sigma_tau2_hat"]
        if draw == 0 and return_first_trace:
            trace = {"outer": outer, "inner_a": inner_a, "inner_b": inner_b}
    intervals = {name: np.quantile(values, [0.025, 0.975]) for name, values in draws.items()}
    return {
        "draws": draws,
        "intervals": intervals,
        "trace": trace,
        "rng": {"bit_generator": "PCG64", "seed": seed, "numpy_version": np.__version__},
    }


_BRANCH_BINDING_FIELDS = (
    "protocol_version",
    "protocol_sha256",
    "preflight_manifest_sha256",
    "base_run_id",
    "block",
    "replica_id",
    "checkpoint_sha256",
    "future_seed",
    "future_seed_namespace",
    "future_seed_counter",
    "future_manifest_sha256",
    "realized_order_stream_sha256",
)


def _branch_record_payload(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "branch_artifact_id",
        "branch_artifact_sha256",
        "branch_artifact_registry_sha256",
    }
    def normalized(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return {"__nonfinite_float__": "NaN"}
            return {"__nonfinite_float__": "+Inf" if value > 0 else "-Inf"}
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalized(item) for item in value]
        return value

    return {
        key: normalized(value) for key, value in row.items() if key not in excluded
    }


def _branch_group_sha256(rows: Iterable[dict[str, Any]]) -> str:
    payload = sorted(
        (_branch_record_payload(row) for row in rows),
        key=lambda row: (str(row["action"]), int(row["epoch"])),
    )
    return canonical_sha256(payload)


def register_synthetic_branch_artifacts(
    records: Iterable[dict[str, Any]], evidence: CandidatePreflightEvidence
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Create a clearly noncanonical registry solely for integrity unit tests."""
    evidence = _require_candidate(evidence)
    rows = [dict(row) for row in records]
    groups: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("synthetic_test_fixture") is not True:
            raise ValueError("Synthetic artifact registration requires explicit fixture records")
        key = (int(row["base_run_id"]), str(row["block"]), int(row["replica_id"]))
        row.setdefault("realized_order_stream_sha256", canonical_sha256(list(key)))
        groups.setdefault(key, []).append(row)
    entries = []
    for key, group in sorted(groups.items()):
        base, block, replica = key
        mapping = resolve_future_manifest_row(evidence, base, block, replica)
        artifact_id = f"synthetic/{base:02d}/{block}/{replica:02d}"
        artifact_sha = _branch_group_sha256(group)
        entries.append(
            {
                "artifact_id": artifact_id,
                "artifact_sha256": artifact_sha,
                "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
                "protocol_sha256": PROTOCOL_SHA256,
                "preflight_manifest_sha256": evidence.manifest_sha256,
                "base_run_id": base,
                "block": block,
                "replica_id": replica,
                "checkpoint_sha256": group[0]["checkpoint_sha256"],
                "future_seed": int(mapping["future_seed"]),
                "future_seed_namespace": mapping["namespace"],
                "future_seed_counter": int(mapping["counter"]),
                "future_manifest_sha256": evidence.future_manifest_sha256,
                "realized_order_stream_sha256": group[0]["realized_order_stream_sha256"],
                "record_count": len(group),
            }
        )
        for row in group:
            row["branch_artifact_id"] = artifact_id
            row["branch_artifact_sha256"] = artifact_sha
    registry = {
        "registry_type": "synthetic_phase5_branch_artifact_registry",
        "protocol_version": SYNTHETIC_PROTOCOL_VERSION,
        "status": "SYNTHETIC_TEST_ONLY",
        "entries": entries,
    }
    registry_sha = canonical_sha256(registry)
    for row in rows:
        row["branch_artifact_registry_sha256"] = registry_sha
    return rows, registry, registry_sha


def validate_registered_branch_records(
    records: Iterable[dict[str, Any]],
    evidence: CandidatePreflightEvidence,
    registry: dict[str, Any],
    registry_sha256: str,
    *,
    canonical_registry: bool,
) -> None:
    """Bind record bytes and identities to exact reviewed branch-registry entries."""
    rows = list(records)
    if canonical_registry:
        if (
            registry.get("registry_type") != "phase5_v1_branch_artifact_registry"
            or registry.get("protocol_version") != PROTOCOL_VERSION
            or registry.get("status") != "REVIEWED"
            or registry_sha256 != evidence.branch_artifact_registry_sha256
        ):
            raise PermissionError("Reviewed canonical branch-artifact registry is unavailable")
    else:
        if (
            registry.get("registry_type") != "synthetic_phase5_branch_artifact_registry"
            or registry.get("status") != "SYNTHETIC_TEST_ONLY"
            or registry_sha256 != canonical_sha256(registry)
        ):
            raise ValueError("Synthetic branch-artifact registry identity is invalid")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Branch-artifact registry entries are missing")
    entry_lookup = {
        (int(row["base_run_id"]), str(row["block"]), int(row["replica_id"])): row
        for row in entries
    }
    groups: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["base_run_id"]), str(row["block"]), int(row["replica_id"]))
        groups.setdefault(key, []).append(row)
    if set(groups) != set(entry_lookup):
        raise ValueError("Analysis records do not equal the registered branch-artifact set")
    for key, group in groups.items():
        registered = entry_lookup[key]
        if _branch_group_sha256(group) != registered.get("artifact_sha256"):
            raise ValueError("Branch artifact hash does not match registered record bytes")
        expected_protocol = PROTOCOL_VERSION if canonical_registry else SYNTHETIC_PROTOCOL_VERSION
        required = {
            "protocol_version": expected_protocol,
            "protocol_sha256": PROTOCOL_SHA256,
            "preflight_manifest_sha256": evidence.manifest_sha256,
            "base_run_id": key[0],
            "block": key[1],
            "replica_id": key[2],
            "record_count": len(group),
        }
        if any(registered.get(field) != value for field, value in required.items()):
            raise ValueError("Registered branch artifact has inconsistent protocol identity")
        for row in group:
            if (
                row.get("branch_artifact_id") != registered.get("artifact_id")
                or row.get("branch_artifact_sha256") != registered.get("artifact_sha256")
                or row.get("branch_artifact_registry_sha256") != registry_sha256
            ):
                raise ValueError("Analysis row is not bound to its registered branch artifact")
            for field in _BRANCH_BINDING_FIELDS:
                if field == "protocol_version" and not canonical_registry:
                    continue
                if row.get(field) != registered.get(field):
                    raise ValueError(f"Branch artifact provenance mismatch: {field}")


def construct_primary_data_from_records(
    records: Iterable[dict[str, Any]],
    evidence: CandidatePreflightEvidence,
    *,
    preflight_synthetic_validation: bool = False,
    execution_gate: ProductionGateContext | None = None,
    branch_artifact_registry: dict[str, Any] | None = None,
    branch_artifact_registry_sha256: str | None = None,
    branch_snapshot: dict[str, Any] | None = None,
) -> ValidatedPrimaryData:
    """Construct frozen arrays from a complete SUCCESS/divergence terminal universe."""
    rows = [dict(row) for row in records]
    r5_snapshot_admission = branch_snapshot is not None
    if r5_snapshot_admission:
        from .phase5_evidence import (
            branch_analysis_records_from_snapshot,
            canonical_json_bytes as r5_canonical_json_bytes,
        )

        gate = validate_execution_gate(execution_gate, "analysis", branch_snapshot)
        if gate.authority_domain not in {"production", "synthetic_test_fixture"}:
            raise PermissionError("r5 analysis authority domain is invalid")
        snapshot_rows = branch_analysis_records_from_snapshot(branch_snapshot, gate)
        if r5_canonical_json_bytes(rows) != r5_canonical_json_bytes(snapshot_rows):
            raise PermissionError("Analysis rows differ from the authorized branch snapshot")
        evidence = load_canonical_preflight(gate.repo)
        canonical_production = gate.authority_domain == "production"
    else:
        synthetic_flags = {row.get("synthetic_test_fixture") is True for row in rows}
        if synthetic_flags == {True} and not preflight_synthetic_validation:
            raise PermissionError("Synthetic fixture records cannot enter canonical Phase5-v1 analysis")
        if preflight_synthetic_validation and synthetic_flags != {True}:
            raise ValueError("Synthetic preflight validation requires every record to be explicitly synthetic")
        if synthetic_flags == {False} and preflight_synthetic_validation:
            raise ValueError("Production records cannot be relabeled as synthetic validation data")
        if len(synthetic_flags) != 1:
            raise ValueError("Synthetic and production records cannot be mixed")
        canonical_production = synthetic_flags == {False}
        if canonical_production:
            evidence = _require_canonical_candidate(evidence)
            gate = validate_execution_gate(execution_gate, "analysis")
            if gate.preflight.manifest_sha256 != evidence.manifest_sha256:
                raise PermissionError("Analysis gate and production records use different evidence")
            validate_registered_branch_records(
                rows,
                evidence,
                evidence.branch_artifact_registry,
                evidence.branch_artifact_registry_sha256,
                canonical_registry=True,
            )
        else:
            evidence = _require_candidate(evidence)
            if branch_artifact_registry is None or branch_artifact_registry_sha256 is None:
                raise ValueError("Synthetic validation requires an explicit synthetic artifact registry")
            validate_registered_branch_records(
                rows,
                evidence,
                branch_artifact_registry,
                branch_artifact_registry_sha256,
                canonical_registry=False,
            )
    mapping_lookup = {
        (int(row["base_run_id"]), row["block"], int(row["replica_id"])): row
        for row in evidence.future_rows
    }
    a = np.full((N_STATES, K_A, 4), np.nan, dtype=float)
    now = np.full((N_STATES, K_B, 3), np.nan, dtype=float)
    wait = np.full((N_STATES, K_B, 3), np.nan, dtype=float)
    seen: set[tuple[int, str, int, str, int]] = set()
    checkpoint_by_base: dict[int, str] = {}
    divergence: list[dict[str, Any]] = []
    terminal_by_assignment: dict[tuple[int, str, int], dict[str, Any]] = {}
    terminal_rows = [
        row
        for row in rows
        if row.get("record_type") == "phase5_v1_r5_assignment_terminal"
    ]
    ordinary_rows = [
        row
        for row in rows
        if row.get("record_type") != "phase5_v1_r5_assignment_terminal"
    ]
    if terminal_rows and not r5_snapshot_admission:
        raise PermissionError("Divergence terminals require an authorized r5 branch snapshot")
    terminal_required = {
        "terminal_state",
        "protocol_version",
        "protocol_sha256",
        "preflight_manifest_sha256",
        "base_run_id",
        "block",
        "replica_id",
        "checkpoint_sha256",
        "future_seed",
        "future_seed_namespace",
        "future_seed_counter",
        "future_manifest_sha256",
        "scientific_identity",
        "scientific_identity_sha256",
        "attempt_id",
        "artifact_path",
        "artifact_sha256",
        "audit_status",
        "realized_order_stream_sha256",
        "divergence_rows",
    }
    for row in terminal_rows:
        if not terminal_required <= row.keys() or row["terminal_state"] != "substantive_divergence":
            raise ValueError("Canonical divergence terminal is malformed")
        if (
            row["protocol_version"] != PROTOCOL_VERSION
            or row["protocol_sha256"] != PROTOCOL_SHA256
            or row["preflight_manifest_sha256"] != evidence.manifest_sha256
            or row["future_manifest_sha256"] != evidence.future_manifest_sha256
            or row["audit_status"] != "passed"
        ):
            raise ValueError("Divergence terminal has invalid protocol or manifest provenance")
        identity = row["scientific_identity"]
        if not isinstance(identity, dict) or canonical_sha256(identity) != row["scientific_identity_sha256"]:
            raise ValueError("Divergence terminal scientific identity is malformed")
        base = int(row["base_run_id"])
        block = str(row["block"])
        replica = int(row["replica_id"])
        assignment = (base, block, replica)
        mapping = mapping_lookup.get(assignment)
        if mapping is None:
            raise ValueError("Divergence terminal is outside the frozen assignment universe")
        if (
            int(row["future_seed"]) != int(mapping["future_seed"])
            or row["future_seed_namespace"] != mapping["namespace"]
            or int(row["future_seed_counter"]) != int(mapping["counter"])
        ):
            raise ValueError("Divergence terminal differs from its frozen future mapping")
        if assignment in terminal_by_assignment:
            raise ValueError("Canonical assignment has duplicate divergence terminals")
        if not isinstance(row["divergence_rows"], list) or not row["divergence_rows"]:
            raise ValueError("Substantive-divergence terminal lacks derived evidence")
        terminal_by_assignment[assignment] = row
        checkpoint_sha = str(row["checkpoint_sha256"])
        if len(checkpoint_sha) != 64:
            raise ValueError("Divergence terminal checkpoint identity is malformed")
        previous = checkpoint_by_base.setdefault(base, checkpoint_sha)
        if previous != checkpoint_sha:
            raise ValueError("A state uses inconsistent checkpoint identities")
        for item in row["divergence_rows"]:
            if (
                not isinstance(item, dict)
                or int(item.get("base_run_id", -1)) != base
                or item.get("block") != block
                or int(item.get("replica_id", -1)) != replica
            ):
                raise ValueError("Divergence location differs from its terminal assignment")
            divergence.append(
                {
                    **item,
                    "checkpoint_sha256": checkpoint_sha,
                    "future_manifest_sha256": evidence.future_manifest_sha256,
                    "future_seed_namespace": mapping["namespace"],
                    "future_seed_counter": int(mapping["counter"]),
                    "attempt_id": row["attempt_id"],
                    "artifact_path": row["artifact_path"],
                    "artifact_sha256": row["artifact_sha256"],
                    "audit_status": "passed",
                }
            )
    required = {
        "protocol_version",
        "protocol_sha256",
        "preflight_manifest_sha256",
        "base_run_id",
        "block",
        "replica_id",
        "action",
        "epoch",
        "val_loss",
        "checkpoint_sha256",
        "future_seed",
        "future_seed_namespace",
        "future_seed_counter",
        "future_manifest_sha256",
        "audit_status",
        "realized_order_stream_sha256",
    }
    if not r5_snapshot_admission:
        required |= {
            "branch_artifact_id",
            "branch_artifact_sha256",
            "branch_artifact_registry_sha256",
        }
    ordinary_assignment_rows: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for row in ordinary_rows:
        if not required <= row.keys():
            raise ValueError("Production record lacks required state/block/replica/epoch provenance")
        if (
            row["protocol_version"] != PROTOCOL_VERSION
            or row["protocol_sha256"] != PROTOCOL_SHA256
            or row["preflight_manifest_sha256"] != evidence.manifest_sha256
            or row["future_manifest_sha256"] != evidence.future_manifest_sha256
            or row["audit_status"] != "passed"
        ):
            raise ValueError("Production record has invalid protocol, manifest, or audit provenance")
        base = int(row["base_run_id"])
        block = str(row["block"])
        replica = int(row["replica_id"])
        action = str(row["action"])
        epoch = int(row["epoch"])
        if action not in {"Now", "Wait"}:
            raise ValueError("Production record action must be Now or Wait")
        mapping = mapping_lookup.get((base, block, replica))
        if mapping is None:
            raise ValueError("Production record does not match a frozen A/B assignment")
        if (
            int(row["future_seed"]) != int(mapping["future_seed"])
            or row["future_seed_namespace"] != mapping["namespace"]
            or int(row["future_seed_counter"]) != int(mapping["counter"])
        ):
            raise ValueError("Production record differs from its frozen future mapping")
        checkpoint_sha = str(row["checkpoint_sha256"])
        if len(checkpoint_sha) != 64:
            raise ValueError("Production record checkpoint identity is malformed")
        previous = checkpoint_by_base.setdefault(base, checkpoint_sha)
        if previous != checkpoint_sha:
            raise ValueError("A state uses inconsistent checkpoint identities")
        assignment = (base, block, replica)
        if assignment in terminal_by_assignment:
            raise ValueError("Divergent assignment also supplied ordinary success rows")
        ordinary_assignment_rows.setdefault(assignment, []).append(row)
        key = (base, block, replica, action, epoch)
        if key in seen:
            raise ValueError("Duplicate production record identity")
        seen.add(key)
        value = float(row["val_loss"])
        if value < 0:
            raise ValueError("Validation loss cannot be negative")
        if not np.isfinite(value):
            divergence.append(
                {
                    "base_run_id": base,
                    "block": block,
                    "replica_id": replica,
                    "action": action,
                    "epoch": epoch,
                    "nonfinite_field": "val_loss",
                    "checkpoint_sha256": checkpoint_sha,
                    "future_manifest_sha256": evidence.future_manifest_sha256,
                    "future_seed_namespace": mapping["namespace"],
                    "future_seed_counter": int(mapping["counter"]),
                    "audit_status": "passed",
                }
            )
        if block == "A":
            if not 1 <= replica <= K_A or epoch not in (15, 16, 17, 18):
                raise ValueError("A records require replicas 1..15 and exact horizons h=1..4")
            horizon = epoch - PRIMARY_CHECKPOINT_EPOCH
            if "horizon" in row and int(row["horizon"]) != horizon:
                raise ValueError("A horizon does not match its absolute epoch")
            index = (base - 1, replica - 1, horizon - 1)
            if action == "Now":
                # Delta is filled once its Wait mate is available below.
                pass
        elif block == "B":
            if not 1 <= replica <= K_B or epoch not in DESIGN.final_window:
                raise ValueError("B records require replicas 1..10 and exact epochs 28..30")
            index = (base - 1, replica - 1, DESIGN.final_window.index(epoch))
            (now if action == "Now" else wait)[index] = value
        else:
            raise ValueError("A/B production provenance cannot be swapped")

    expected_a_keys = {
        (base, "A", replica, action, epoch)
        for base in range(1, N_STATES + 1)
        for replica in range(1, K_A + 1)
        for action in ("Now", "Wait")
        for epoch in (15, 16, 17, 18)
    }
    expected_b_keys = {
        (base, "B", replica, action, epoch)
        for base in range(1, N_STATES + 1)
        for replica in range(1, K_B + 1)
        for action in ("Now", "Wait")
        for epoch in DESIGN.final_window
    }
    expected = expected_a_keys | expected_b_keys
    divergent_row_keys = {
        key
        for key in expected
        if (key[0], key[1], key[2]) in terminal_by_assignment
    }
    expected_success_keys = expected - divergent_row_keys
    if seen != expected_success_keys:
        raise ValueError("Production records are incomplete, malformed, or contain unexpected identities")

    expected_assignments = set(mapping_lookup)
    observed_assignments = set(ordinary_assignment_rows) | set(terminal_by_assignment)
    if observed_assignments != expected_assignments:
        raise ValueError("Canonical terminal universe is incomplete, extra, or duplicated")

    row_lookup = {
        (int(row["base_run_id"]), row["block"], int(row["replica_id"]), row["action"], int(row["epoch"])): row
        for row in ordinary_rows
    }
    for base in range(1, N_STATES + 1):
        for replica in range(1, K_A + 1):
            if (base, "A", replica) in terminal_by_assignment:
                continue
            for horizon, epoch in enumerate((15, 16, 17, 18)):
                n_value = float(row_lookup[(base, "A", replica, "Now", epoch)]["val_loss"])
                w_value = float(row_lookup[(base, "A", replica, "Wait", epoch)]["val_loss"])
                delta = w_value - n_value
                for action in ("Now", "Wait"):
                    reported = row_lookup[(base, "A", replica, action, epoch)].get("delta_l")
                    if reported is not None and not np.isclose(
                        float(reported), delta, rtol=0.0, atol=0.0, equal_nan=True
                    ):
                        raise ValueError("Reported A delta must use the exact W-N sign")
                a[base - 1, replica - 1, horizon] = delta

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean_now = now.mean(axis=2)
        mean_wait = wait.mean(axis=2)
        gains = (mean_wait - mean_now) / np.maximum(mean_wait, 1e-12)
    for state, replica in np.argwhere(~np.isfinite(gains)):
        assignment = (int(state) + 1, "B", int(replica) + 1)
        if assignment in terminal_by_assignment:
            continue
        mapping = mapping_lookup[assignment]
        divergence.append(
            {
                "base_run_id": int(state) + 1,
                "block": "B",
                "replica_id": int(replica) + 1,
                "action": "derived_W_minus_N",
                "epoch": "28..30",
                "nonfinite_field": "G",
                "checkpoint_sha256": checkpoint_by_base[int(state) + 1],
                "future_manifest_sha256": evidence.future_manifest_sha256,
                "future_seed_namespace": mapping["namespace"],
                "future_seed_counter": int(mapping["counter"]),
                "audit_status": "passed",
            }
        )
    terminal_assignments: list[dict[str, Any]] = []
    for assignment in sorted(expected_assignments):
        if assignment in terminal_by_assignment:
            terminal_assignments.append(dict(terminal_by_assignment[assignment]))
            continue
        sample = ordinary_assignment_rows[assignment][0]
        terminal_assignments.append(
            {
                "record_type": "phase5_v1_r5_assignment_terminal",
                "terminal_state": "terminal_success",
                "base_run_id": assignment[0],
                "block": assignment[1],
                "replica_id": assignment[2],
                "checkpoint_sha256": sample["checkpoint_sha256"],
                "future_seed": sample["future_seed"],
                "future_seed_namespace": sample["future_seed_namespace"],
                "future_seed_counter": sample["future_seed_counter"],
                "future_manifest_sha256": sample["future_manifest_sha256"],
                "audit_status": sample["audit_status"],
                "realized_order_stream_sha256": sample[
                    "realized_order_stream_sha256"
                ],
            }
        )
    return ValidatedPrimaryData(
        a_delta=a,
        b_now_final_losses=now,
        b_wait_final_losses=wait,
        divergence_records=tuple(divergence),
        terminal_assignments=tuple(terminal_assignments),
        preflight_manifest_sha256=evidence.manifest_sha256,
        source_record_count=len(rows),
        canonical_production_records=canonical_production,
        _seal=_PRIMARY_PRODUCTION_SEAL if canonical_production else _PRIMARY_FIXTURE_SEAL,
    )


def _require_validated_primary_data(data: ValidatedPrimaryData) -> None:
    if not isinstance(data, ValidatedPrimaryData):
        raise ValueError("Canonical primary inference requires validated production-record data")
    expected_seal = (
        _PRIMARY_PRODUCTION_SEAL if data.canonical_production_records else _PRIMARY_FIXTURE_SEAL
    )
    if data._seal is not expected_seal:
        raise ValueError("Primary-data namespace/seal mismatch")
    if (
        data.a_delta.shape != (N_STATES, K_A, 4)
        or data.b_now_final_losses.shape != (N_STATES, K_B, 3)
        or data.b_wait_final_losses.shape != (N_STATES, K_B, 3)
    ):
        raise ValueError("Canonical primary inference requires exact 36/15/10 frozen shapes")
    terminal_keys = [
        (int(row["base_run_id"]), str(row["block"]), int(row["replica_id"]))
        for row in data.terminal_assignments
    ]
    expected_terminal_keys = {
        (base, block, replica)
        for base in range(1, N_STATES + 1)
        for block, count in (("A", K_A), ("B", K_B))
        for replica in range(1, count + 1)
    }
    if len(terminal_keys) != len(expected_terminal_keys) or set(terminal_keys) != expected_terminal_keys:
        raise ValueError("Canonical primary inference requires one terminal per frozen assignment")
    if any(
        row.get("terminal_state") not in {"terminal_success", "substantive_divergence"}
        for row in data.terminal_assignments
    ):
        raise ValueError("Canonical terminal state must be SUCCESS or SUBSTANTIVE_DIVERGENCE")


def frozen_primary_bootstrap(data: ValidatedPrimaryData) -> dict[str, Any]:
    """Canonical Phase5-v1 bootstrap with no caller-overridable settings."""
    _require_validated_primary_data(data)
    status = evaluability_status(
        data.a_delta, data.b_now_final_losses, data.b_wait_final_losses
    )
    if any(value != EVALUABLE for value in status.values()):
        raise ValueError("Frozen primary bootstrap cannot replace non-evaluable quantities")
    result = nested_bootstrap(
        data.a_delta,
        data.b_now_final_losses,
        data.b_wait_final_losses,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    if result["rng"]["bit_generator"] != "PCG64" or result["rng"]["seed"] != BOOTSTRAP_SEED:
        raise AssertionError("Frozen primary bootstrap RNG changed")
    result["rng"]["protocol_version"] = PROTOCOL_VERSION
    result["replicates"] = BOOTSTRAP_REPLICATES
    result["artifact_class"] = (
        "canonical_phase5_primary_bootstrap"
        if data.canonical_production_records
        else "synthetic_preflight_validation"
    )
    return result


def _frozen_a_only_bootstrap(a_delta: np.ndarray) -> dict[str, np.ndarray]:
    """Preserve frozen outer/inner draws when B is substantively non-evaluable."""
    if a_delta.shape != (N_STATES, K_A, 4) or not np.isfinite(a_delta).all():
        raise ValueError("A-only bootstrap requires the complete finite frozen A sample")
    rng = primary_bootstrap_rng()
    names = ("mu_rho_hat", "A_34_hat", "mu_1_hat", "mu_2_hat", "mu_3_hat")
    draws = {name: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for name in names}
    for draw in range(BOOTSTRAP_REPLICATES):
        outer, inner_a, _ = bootstrap_resample_indices(rng, N_STATES, K_A, K_B)
        sampled = np.empty((N_STATES, K_A, 4), dtype=float)
        for occurrence, source_state in enumerate(outer):
            sampled[occurrence] = a_delta[source_state, inner_a[occurrence], :]
        draws["mu_rho_hat"][draw] = sampled[:, :, :3].mean(axis=2).mean()
        draws["A_34_hat"][draw] = (sampled[:, :, 3] - sampled[:, :, 2]).mean()
        (
            draws["mu_1_hat"][draw],
            draws["mu_2_hat"][draw],
            draws["mu_3_hat"][draw],
        ) = sampled[:, :, :3].mean(axis=(0, 1))
    return {name: np.quantile(values, [0.025, 0.975]) for name, values in draws.items()}


def divergence_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report divergence counts, rates, and identities by action and block."""
    rows = [dict(row) for row in records]
    action_denominator = N_STATES * K_A * 4 + N_STATES * K_B * 3
    block_action_denominators = {
        "A:Now": N_STATES * K_A * 4,
        "A:Wait": N_STATES * K_A * 4,
        "B:Now": N_STATES * K_B * 3,
        "B:Wait": N_STATES * K_B * 3,
        "B:derived_W_minus_N": N_STATES * K_B,
    }

    def locations(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fields = ("base_run_id", "block", "replica_id", "action", "epoch", "nonfinite_field")
        return [{field: row[field] for field in fields} for row in selected]

    by_action = {}
    for action in ("Now", "Wait"):
        selected = [row for row in rows if row.get("action") == action]
        by_action[action] = {
            "count": len(selected),
            "denominator": action_denominator,
            "rate": len(selected) / action_denominator,
            "locations": locations(selected),
        }
    by_block_action = {}
    for key, denominator in block_action_denominators.items():
        block, action = key.split(":", 1)
        selected = [
            row for row in rows
            if row.get("block") == block and row.get("action") == action
        ]
        by_block_action[key] = {
            "count": len(selected),
            "denominator": denominator,
            "rate": len(selected) / denominator,
            "locations": locations(selected),
        }
    return {
        "total_count": len(rows),
        "by_action": by_action,
        "by_block_action": by_block_action,
    }


def frozen_primary_inference(data: ValidatedPrimaryData) -> dict[str, Any]:
    """Integrate evaluability before emitting any canonical claim."""
    _require_validated_primary_data(data)
    status = evaluability_status(
        data.a_delta, data.b_now_final_losses, data.b_wait_final_losses
    )
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "evaluability": status,
        "divergence_records": list(data.divergence_records),
        "divergence_summary": divergence_summary(data.divergence_records),
        "terminal_assignments": list(data.terminal_assignments),
        "terminal_universe_summary": {
            "expected": N_STATES * (K_A + K_B),
            "observed": len(data.terminal_assignments),
            "success": sum(
                row["terminal_state"] == "terminal_success"
                for row in data.terminal_assignments
            ),
            "substantive_divergence": sum(
                row["terminal_state"] == "substantive_divergence"
                for row in data.terminal_assignments
            ),
        },
        "bootstrap_specification": bootstrap_seed_derivation(),
        "artifact_class": (
            "canonical_phase5_primary_analysis"
            if data.canonical_production_records
            else "synthetic_preflight_validation"
        ),
        "canonical_phase5_analysis": data.canonical_production_records,
    }
    a_divergence = any(
        row.get("block") == "A" for row in data.divergence_records
    )
    b_divergence = any(
        row.get("block") == "B" for row in data.divergence_records
    )
    result["non_evaluable_reasons"] = {
        "phase5a": ["required_A_substantive_divergence"] if a_divergence else [],
        "rho": ["required_A_substantive_divergence"] if a_divergence else [],
        "tau": ["required_B_substantive_divergence"] if b_divergence else [],
        "sigma_tau2": ["required_B_substantive_divergence"] if b_divergence else [],
        "psi": [
            reason
            for condition, reason in (
                (a_divergence, "required_A_substantive_divergence"),
                (b_divergence, "required_B_substantive_divergence"),
            )
            if condition
        ],
    }
    if status["phase5a"] == NON_EVALUABLE:
        if status["tau"] == EVALUABLE:
            gains = per_replica_gain(
                data.b_now_final_losses, data.b_wait_final_losses
            )
            tau = gains.mean(axis=1)
            within_variance = gains.var(axis=1, ddof=1)
            sigma_tau2 = float(
                tau.var(ddof=1) - np.mean(within_variance / gains.shape[1])
            )
            result["b_point_estimates"] = {
                "tau_hat": tau,
                "per_replica_g": gains,
                "sigma_tau2_hat": sigma_tau2,
                "sigma_tau_display": math.sqrt(max(0.0, sigma_tau2)),
            }
        result["claims"] = {
            "phase5a": NON_EVALUABLE,
            "phase5a_failed_conditions": [],
            "phase5b": NON_EVALUABLE,
            "omnibus_claim": "prohibited",
        }
        return result
    if status["tau"] == NON_EVALUABLE:
        a = data.a_delta
        a_estimates = {
            "mu_rho_hat": float(a[:, :, :3].mean(axis=2).mean()),
            "A_34_hat": float((a[:, :, 3] - a[:, :, 2]).mean()),
            "mu_h_hat": a[:, :, :3].mean(axis=(0, 1)),
        }
        intervals = _frozen_a_only_bootstrap(a)
        intervals["psi_hat"] = np.array([np.nan, np.nan])
        result["a_point_estimates"] = a_estimates
        result["a_bootstrap_intervals"] = intervals
        result["claims"] = classify_primary_claims(
            a_estimates, intervals, evaluability=status
        )
        return result
    estimates = primary_estimates(
        data.a_delta, data.b_now_final_losses, data.b_wait_final_losses
    )
    bootstrap = frozen_primary_bootstrap(data)
    result["estimates"] = estimates
    result["bootstrap"] = bootstrap
    result["claims"] = classify_primary_claims(
        estimates, bootstrap["intervals"], evaluability=status
    )
    return result


def canonical_primary_analysis(
    records: Iterable[dict[str, Any]],
    order_records: Iterable[dict[str, Any]],
    *,
    execution_gate: Any,
    branch_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical analysis admits only records sealed by the authorized r5 branch snapshot."""
    if branch_snapshot is None:
        raise PermissionError("Canonical analysis requires a sealed r5 branch snapshot")
    gate = validate_execution_gate(execution_gate, "analysis", branch_snapshot)
    if gate.authority_domain not in {"production", "synthetic_test_fixture"}:
        raise PermissionError("Canonical analysis authority domain is invalid")
    from .phase5_evidence import branch_order_records_from_snapshot

    snapshot_orders = branch_order_records_from_snapshot(branch_snapshot, gate)
    supplied_orders = list(order_records)
    if canonical_json_bytes(supplied_orders) != canonical_json_bytes(snapshot_orders):
        raise PermissionError("Order records differ from the authorized branch snapshot")
    audit_future_order_records(snapshot_orders, require_complete=True)
    evidence = load_canonical_preflight(gate.repo)
    data = construct_primary_data_from_records(
        records,
        evidence,
        execution_gate=gate,
        branch_snapshot=branch_snapshot,
    )
    if data.preflight_manifest_sha256 != evidence.manifest_sha256:
        raise PermissionError("Production records are bound to different preflight evidence")
    return frozen_primary_inference(data)


def analysis_ready_tables(
    a_delta: np.ndarray, b_now_final_losses: np.ndarray, b_wait_final_losses: np.ndarray
) -> dict[str, list[dict[str, Any]]]:
    a = np.asarray(a_delta, dtype=float)
    now = np.asarray(b_now_final_losses, dtype=float)
    wait = np.asarray(b_wait_final_losses, dtype=float)
    estimates = primary_estimates(a, now, wait)
    a_rows = []
    for state in range(N_STATES):
        for replica in range(K_A):
            row = {"base_run_id": state + 1, "block": "A", "replica_id": replica + 1}
            for horizon in range(4):
                row[f"delta_l_h{horizon + 1}"] = float(a[state, replica, horizon])
            row["d"] = float(a[state, replica, :3].mean())
            row["h4_minus_h3"] = float(a[state, replica, 3] - a[state, replica, 2])
            a_rows.append(row)
    b_rows = []
    gains = estimates["per_replica_g"]
    for state in range(N_STATES):
        for replica in range(K_B):
            row = {"base_run_id": state + 1, "block": "B", "replica_id": replica + 1}
            for offset, epoch in enumerate((28, 29, 30)):
                row[f"now_val_loss_e{epoch}"] = float(now[state, replica, offset])
                row[f"wait_val_loss_e{epoch}"] = float(wait[state, replica, offset])
            row["now_mean_e28_30"] = float(now[state, replica].mean())
            row["wait_mean_e28_30"] = float(wait[state, replica].mean())
            row["G"] = float(gains[state, replica])
            row["formula_version"] = "per-replica-ratio-v1"
            b_rows.append(row)
    state_rows = [
        {
            "base_run_id": state + 1,
            "rho_hat": float(estimates["rho_hat"][state]),
            "tau_hat": float(estimates["tau_hat"][state]),
            "within_tau_variance": float(estimates["within_tau_variance"][state]),
        }
        for state in range(N_STATES)
    ]
    return {"a_replica_rows": a_rows, "b_replica_rows": b_rows, "state_rows": state_rows}


MAPPING_ID_FIELDS = (
    "protocol_version",
    "protocol_sha256",
    "preflight_manifest_sha256",
    "base_run_id",
    "checkpoint_sha256",
    "block",
    "replica_id",
    "future_seed",
    "future_seed_namespace",
    "future_seed_counter",
    "future_manifest_sha256",
    "realized_order_stream_sha256",
    "branch_artifact_id",
    "branch_artifact_sha256",
    "branch_artifact_registry_sha256",
)


def resolve_retry_identity(
    event: dict[str, Any],
    evidence: CandidatePreflightEvidence,
    *,
    branch_registry: dict[str, Any] | None = None,
    branch_registry_sha256: str | None = None,
    synthetic_test_fixture: bool = False,
) -> dict[str, Any]:
    """Resolve retry identity from the frozen mapping and registered artifact."""
    evidence = (
        _require_candidate(evidence)
        if synthetic_test_fixture
        else _require_canonical_candidate(evidence)
    )
    base = int(event.get("base_run_id", -1))
    block = str(event.get("block", ""))
    replica = int(event.get("replica_id", -1))
    mapping = resolve_future_manifest_row(evidence, base, block, replica)
    registry = branch_registry if synthetic_test_fixture else evidence.branch_artifact_registry
    registry_sha = (
        branch_registry_sha256
        if synthetic_test_fixture
        else evidence.branch_artifact_registry_sha256
    )
    if registry is None or registry_sha is None:
        raise ValueError("Retry requires a registered branch-artifact identity")
    if synthetic_test_fixture:
        if (
            registry.get("registry_type") != "synthetic_phase5_branch_artifact_registry"
            or registry.get("status") != "SYNTHETIC_TEST_ONLY"
            or registry_sha != canonical_sha256(registry)
        ):
            raise ValueError("Synthetic retry registry is invalid")
        protocol_version = SYNTHETIC_PROTOCOL_VERSION
    else:
        if (
            registry.get("registry_type") != "phase5_v1_branch_artifact_registry"
            or registry.get("status") != "REVIEWED"
            or registry_sha != evidence.branch_artifact_registry_sha256
        ):
            raise PermissionError("Canonical retry requires the reviewed branch-artifact registry")
        protocol_version = PROTOCOL_VERSION
    entries = registry.get("entries")
    matches = [
        row
        for row in entries if (
            int(row.get("base_run_id", -1)),
            str(row.get("block", "")),
            int(row.get("replica_id", -1)),
        ) == (base, block, replica)
    ] if isinstance(entries, list) else []
    if len(matches) != 1:
        raise ValueError(
            "Retry base_run_id/block/replica_id does not resolve to one registered branch artifact"
        )
    artifact = matches[0]
    if not synthetic_test_fixture:
        checkpoint_matches = [
            row
            for row in evidence.primary_checkpoint_registry.get("entries", [])
            if row.get("checkpoint_sha256") == artifact.get("checkpoint_sha256")
            and int(row.get("base_run_id", -1)) == base
            and int(row.get("epoch", -1)) == PRIMARY_CHECKPOINT_EPOCH
        ]
        if evidence.primary_checkpoint_registry.get("status") != "APPROVED" or len(checkpoint_matches) != 1:
            raise PermissionError("Retry source checkpoint is not in the approved 36-checkpoint set")
    resolved = {
        "protocol_version": protocol_version,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_manifest_sha256": evidence.manifest_sha256,
        "base_run_id": base,
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "block": block,
        "replica_id": replica,
        "future_seed": int(mapping["future_seed"]),
        "future_seed_namespace": mapping["namespace"],
        "future_seed_counter": int(mapping["counter"]),
        "future_manifest_sha256": evidence.future_manifest_sha256,
        "realized_order_stream_sha256": artifact["realized_order_stream_sha256"],
        "branch_artifact_id": artifact["artifact_id"],
        "branch_artifact_sha256": artifact["artifact_sha256"],
        "branch_artifact_registry_sha256": registry_sha,
    }
    return resolved


def append_attempt_event(
    path: Path, event: dict[str, Any], evidence: CandidatePreflightEvidence
) -> None:
    """Append a canonical retry only after resolving trusted r3 registrations."""
    frozen_mapping = resolve_retry_identity(event, evidence)
    for field in MAPPING_ID_FIELDS:
        if event.get(field) != frozen_mapping.get(field):
            raise ValueError(
                f"Technical retry must retain frozen checkpoint/mapping/stream identity: {field}"
            )
    required = {*MAPPING_ID_FIELDS, "attempt_id", "event_type", "status"}
    if not required <= event.keys():
        raise ValueError("Attempt event is missing required fields")
    prior = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            prior = [json.loads(line) for line in handle if line.strip()]
    if any(row["attempt_id"] == event["attempt_id"] for row in prior):
        raise ValueError("Attempt IDs are append-only and unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def append_synthetic_attempt_event(
    path: Path,
    event: dict[str, Any],
    evidence: CandidatePreflightEvidence,
    branch_registry: dict[str, Any],
    branch_registry_sha256: str,
) -> None:
    """Test-only counterpart; its namespace cannot authorize canonical retries."""
    frozen_mapping = resolve_retry_identity(
        event,
        evidence,
        branch_registry=branch_registry,
        branch_registry_sha256=branch_registry_sha256,
        synthetic_test_fixture=True,
    )
    for field in MAPPING_ID_FIELDS:
        if event.get(field) != frozen_mapping.get(field):
            raise ValueError(
                f"Technical retry must retain frozen checkpoint/mapping/stream identity: {field}"
            )
    required = {*MAPPING_ID_FIELDS, "attempt_id", "event_type", "status"}
    if not required <= event.keys():
        raise ValueError("Attempt event is missing required fields")
    prior = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            prior = [json.loads(line) for line in handle if line.strip()]
    if any(row["attempt_id"] == event["attempt_id"] for row in prior):
        raise ValueError("Attempt IDs are append-only and unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def create_versioned_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def checksum_manifest(root: Path) -> list[dict[str, str]]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Checksum root must be a directory")
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(p for p in root.rglob("*") if p.is_file())
    ]


def copy_and_verify_backup(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError("Refuse to overwrite backup destination")
    before = checksum_manifest(source)
    shutil.copytree(source, destination)
    after = checksum_manifest(destination)
    if before != after:
        raise IOError("Backup checksum verification failed")
    return {
        "status": "passed",
        "file_count": len(before),
        "manifest_sha256": canonical_sha256(before),
    }


def phase5_schema() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "artifact_status": "PRE-OUTCOME DESIGN/PREFLIGHT ONLY",
        "design_environment_required": [
            "protocol_version", "protocol_sha256", "git_commit", "git_worktree_status", "code_sha256",
            "test_sha256", "python_version", "torch_version", "torchvision_version", "numpy_version",
            "platform", "device", "torch_threads", "num_workers", "training_config", "intervention_config",
            "dataset_source", "split_seed", "split_fingerprint", "dataset_file_integrity",
            "historical_base_seed_registry_sha256", "historical_checkpoint_registry_sha256",
            "base_seed_manifest_sha256", "future_seed_mapping_sha256", "bootstrap_seed",
            "bootstrap_bit_generator", "bootstrap_protocol_version"
        ],
        "base_seed_manifest": [
            "protocol_version", "base_run_id", "index", "counter", "namespace", "base_seed", "pre_outcome"
        ],
        "future_seed_mapping": [
            "protocol_version", "base_run_id", "block", "replica_id", "counter", "namespace",
            "future_seed", "start_epoch", "end_epoch", "pre_outcome"
        ],
        "primary_state": [
            "protocol_version", "base_run_id", "base_seed", "epoch", "checkpoint_path",
            "checkpoint_sha256", "split_fingerprint", "state_manifested_before_future_outcomes"
        ],
        "base_metrics": [
            "base_run_id", "base_seed", "attempt_id", "epoch", "train_loss", "val_loss",
            "val_accuracy", "learning_rate"
        ],
        "replica_registry": [
            "protocol_version", "base_run_id", "checkpoint_sha256", "block", "replica_id",
            "future_seed", "namespace", "counter", "mapping_manifest_sha256", "attempt_id",
            "start_epoch", "end_epoch"
        ],
        "branch_metrics": [
            "base_run_id", "block", "replica_id", "action", "epoch", "horizon", "learning_rate",
            "train_loss", "val_loss", "val_accuracy", "order_sha256"
        ],
        "integrity_record": [
            "base_run_id", "checkpoint_sha256", "block", "replica_id", "attempt_id",
            "initial_state_equal", "source_checkpoint_immutable", "lr_only_intervention",
            "rng_states_equal", "actual_orders_equal", "now_order_sha256", "wait_order_sha256",
            "cross_assignment_order_unique", "schedule_valid", "coverage_valid"
        ],
        "failure_retry_ledger": [
            "base_run_id", "block", "replica_id", "future_seed", "attempt_id", "event_type",
            "status", "reason_code", "timestamp", "parent_attempt_id"
        ],
        "divergence_record": [
            "base_run_id", "block", "replica_id", "action", "epoch", "audit_status",
            "nonfinite_field", "non_evaluable_components"
        ],
        "checksum_manifest": ["path", "sha256"],
        "a_analysis": [
            "base_run_id", "block", "replica_id", "delta_l_h1", "delta_l_h2", "delta_l_h3",
            "delta_l_h4", "d", "h4_minus_h3"
        ],
        "b_analysis": [
            "base_run_id", "block", "replica_id", "now_val_loss_e28", "now_val_loss_e29",
            "now_val_loss_e30", "wait_val_loss_e28", "wait_val_loss_e29", "wait_val_loss_e30",
            "now_mean_e28_30", "wait_mean_e28_30", "G", "formula_version"
        ],
        "formula_semantics": {
            "G": "(mean_wait_e28_30 - mean_now_e28_30) / max(mean_wait_e28_30, 1e-12), per replica",
            "tau": "mean of per-replica G; never ratio of state-level mean losses",
            "rho": "A-only mean of per-replica mean(delta_l_h1..h3)",
        },
    }


def _validated_predecessor_manifest(repo: Path) -> dict[str, Any]:
    predecessor_path = repo / PREDECESSOR_PREFLIGHT_PATH
    if not predecessor_path.is_file() or sha256_file(predecessor_path) != PREDECESSOR_MANIFEST_SHA256:
        raise ValueError("Historical pre-repair preflight manifest changed")
    predecessor = _read_json(predecessor_path)
    if (
        predecessor.get("protocol_version") != PROTOCOL_VERSION
        or predecessor.get("protocol_commit") != PROTOCOL_COMMIT
        or predecessor.get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise ValueError("Historical preflight protocol identity is inconsistent")
    root = predecessor_path.parent
    for name, expected in predecessor.get("artifact_sha256", {}).items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Historical preflight artifact changed: {name}")
    return predecessor


def materialize_preflight_revision3(repo: Path, output: Path) -> dict[str, Any]:
    """Create outcome-free r3 evidence after the second conformance audit."""
    repo = repo.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Refuse to overwrite revision-3 preflight artifacts")
    assert_frozen_protocol(repo)
    predecessor = _validated_predecessor_manifest(repo)
    predecessor_root = (repo / PREDECESSOR_PREFLIGHT_PATH).parent

    # Recompute frozen identities, then copy r2 scientific artifacts byte-for-byte.
    base_registry = build_historical_base_seed_registry(repo)
    checkpoint_registry = build_historical_checkpoint_registry(repo)
    historical_seeds = [entry["seed"] for entry in base_registry["entries"]]
    base_rows, future_rows = generate_seed_manifests(historical_seeds)
    predecessor_base = _read_seed_csv(predecessor_root / "base_seed_manifest.csv", future=False)
    predecessor_future = _read_seed_csv(predecessor_root / "future_seed_mapping.csv", future=True)
    if base_rows != predecessor_base or future_rows != predecessor_future:
        raise ValueError("Frozen Phase5-v1 seed mappings differ from predecessor evidence")
    if base_registry != _read_json(predecessor_root / "historical_base_seed_registry.json"):
        raise ValueError("Frozen historical base-seed registry changed")
    predecessor_checkpoint_metadata = _read_json(
        predecessor_root / "historical_checkpoint_registry_metadata.json"
    )
    current_checkpoint_metadata = {
        key: value for key, value in checkpoint_registry.items() if key != "entries"
    }
    current_checkpoint_metadata["registry_csv_sha256"] = sha256_file(
        predecessor_root / "historical_checkpoint_registry.csv"
    )
    if current_checkpoint_metadata != predecessor_checkpoint_metadata:
        raise ValueError("Frozen historical checkpoint registry changed")
    if bootstrap_seed_derivation() != _read_json(predecessor_root / "bootstrap_rng.json"):
        raise ValueError("Frozen bootstrap specification changed")
    if phase5_schema() != _read_json(predecessor_root / "phase5_schema.json"):
        raise ValueError("Frozen Phase5-v1 schema semantics changed")

    output.mkdir(parents=True)
    unchanged_names = [
        "historical_base_seed_registry.json",
        "historical_checkpoint_registry.csv",
        "historical_checkpoint_registry_metadata.json",
        "base_seed_manifest.csv",
        "future_seed_mapping.csv",
        "phase5_schema.json",
        "bootstrap_rng.json",
    ]
    for name in unchanged_names:
        shutil.copyfile(predecessor_root / name, output / name)
        if sha256_file(output / name) != predecessor["artifact_sha256"][name]:
            raise IOError(f"Revision-3 copy verification failed: {name}")

    raw_hashes = {
        "data/FashionMNIST/raw/train-images-idx3-ubyte":
            "c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888",
        "data/FashionMNIST/raw/train-labels-idx1-ubyte":
            "bad3541b69d912435c50bb6ba87bec294ff4f6a2e1246121d8633921760443d9",
    }
    for relative, expected in raw_hashes.items():
        if sha256_file(repo / relative) != expected:
            raise ValueError(f"Frozen Fashion-MNIST raw-file identity changed: {relative}")
    dataset_identity = {
        "registry_type": "phase5_v1_frozen_dataset_identity",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_source": "FashionMNIST official training set",
        "dataset_class": "torchvision.datasets.mnist.FashionMNIST",
        "train": True,
        "population_size": 60_000,
        "content_sha256": "3abd1c978a6ae509d023c293629b2f2b22ac7117d1b7a99e4d9149445dc0c938",
        "raw_file_sha256": raw_hashes,
        "transform_identity": "torchvision.transforms.transforms.ToTensor:ToTensor()",
        "target_transform": None,
        "split_seed": 2026,
        "train_size": 10_000,
        "validation_size": 5_000,
        "split_fingerprint": "d697ce4e11ce24f9217f3d61e6bf4252ef918c6db88c658d0d029b7216bf016f",
        "train_indices_sha256": "de2fe7734daf05c6820f4906ad4d561661bb952a99549bfcf6c4a61d3bdd113d",
        "validation_indices_sha256": "9da7b4014e6f83728a893c8a64c79536f43113c8eb6f574b40735f91c0e25726",
        "no_stochastic_augmentation": True,
    }
    write_json(output / "dataset_identity.json", dataset_identity)
    gate_registry = {
        "registry_type": "phase5_v1_gate_authority_registry",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "NO_TRUSTED_POSITIVE_AUTHORITY_ARTIFACTS",
        "trusted_artifact_sha256": {
            "independent_conformance_audit": None,
            "implementation_test_report": None,
            "seed_manifest_review": None,
            "backup_verification": None,
            "hard_gate_evidence_base": None,
            "hard_gate_evidence_branch": None,
            "hard_gate_evidence_analysis": None,
            "execution_authorization_base": None,
            "execution_authorization_branch": None,
            "execution_authorization_analysis": None,
            "primary_checkpoint_audit": None,
            "checkpoint_restoration_audit": None,
            "complete_order_stream_audit": None,
        },
        "note": "No legitimate positive audit, backup, or authorization artifact exists at r3.",
    }
    write_json(output / "gate_authority_registry.json", gate_registry)
    write_json(
        output / "primary_checkpoint_registry.json",
        {
            "registry_type": "phase5_v1_primary_checkpoint_registry",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "status": "ABSENT",
            "required_count": N_STATES,
            "entries": [],
            "phase5_scientific_outcomes_exist": False,
        },
    )
    write_json(
        output / "branch_artifact_registry.json",
        {
            "registry_type": "phase5_v1_branch_artifact_registry",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "status": "ABSENT",
            "required_assignment_count": TOTAL_FUTURE_REPLICAS,
            "entries": [],
            "phase5_scientific_outcomes_exist": False,
        },
    )

    unresolved = [
        "revision-3 evidence has not passed an independent read-only conformance audit",
        "implementation, tests, and revision-3 evidence are not committed",
        "independent backup destination remains TBD and unverified",
        "separate user authorization for real Phase 5 base training is absent",
        "36 real epoch-14 Phase 5 primary checkpoints do not exist",
        "real primary-checkpoint non-overlap and restoration audits remain pending",
        "complete 900-assignment CRN/order-stream audit remains pending",
        "separate user authorization for real Phase 5 branching is absent",
        "separate user authorization for canonical primary analysis is absent",
    ]
    gate_status = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_evidence_revision": PREFLIGHT_REVISION,
        "status": CANDIDATE_STATUS,
        "current_hard_gate_status": "BLOCKED",
        "unresolved_execution_hard_gates": unresolved,
        "independent_conformance_audit": "ABSENT",
        "real_execution_authorization": "ABSENT",
        "backup_destination": "TBD / EXECUTION BLOCKED",
        "training_approved": False,
        "branching_approved": False,
        "analysis_approved": False,
        "phase5_scientific_outcomes_exist": False,
    }
    write_json(output / "hard_gate_status.json", gate_status)
    created_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_evidence_revision": PREFLIGHT_REVISION,
        "predecessor_preflight_path": PREDECESSOR_PREFLIGHT_PATH,
        "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
        "predecessor_status": "historical_candidate_r2_preserved_byte_identically",
        "revision_reason": "narrow hardening after second independent conformance audit",
        "created_at_utc": created_at,
        "creation_mode": "synthetic/unit/integrity validation only; no Phase 5 execution",
        "scientific_protocol_changed": False,
        "phase5_scientific_outcomes_exist": False,
        "status": CANDIDATE_STATUS,
    }
    write_json(output / "revision_provenance.json", provenance)

    artifact_names = unchanged_names + [
        "dataset_identity.json",
        "gate_authority_registry.json",
        "primary_checkpoint_registry.json",
        "branch_artifact_registry.json",
        "hard_gate_status.json",
        "revision_provenance.json",
    ]
    artifact_hashes = {name: sha256_file(output / name) for name in artifact_names}
    code_test_names = [
        "reflexml/phase5.py",
        "prepare_phase5.py",
        "tests/test_phase5.py",
        "reflexml/checkpoint.py",
        "reflexml/branching.py",
        "reflexml/timing.py",
        "reflexml/stability.py",
    ]
    code_test_hashes = {name: sha256_file(repo / name) for name in code_test_names}
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_evidence_revision": PREFLIGHT_REVISION,
        "predecessor_preflight_path": PREDECESSOR_PREFLIGHT_PATH,
        "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
        "predecessor_status": "historical_candidate_r2_preserved_byte_identically",
        "revision_reason": "narrow hardening after second independent conformance audit",
        "status": CANDIDATE_STATUS,
        "artifact_status": "PRE-OUTCOME CANDIDATE EVIDENCE; CONTAINS NO PHASE 5 OUTCOMES",
        "creation_provenance": {
            "created_at_utc": created_at,
            "generator": "prepare_phase5.py / materialize_preflight_revision3",
            "r1_and_r2_preserved": True,
            "scientific_design_artifacts_byte_identical": True,
        },
        "generated_counts": predecessor["generated_counts"],
        "audits": predecessor["audits"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "num_workers": 0,
        },
        "bootstrap": {
            "bit_generator": "PCG64",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "protocol_version": PROTOCOL_VERSION,
            "numpy_version": np.__version__,
        },
        "seed_manifest_identity": {
            "base_seed_manifest_sha256": artifact_hashes["base_seed_manifest.csv"],
            "future_seed_mapping_sha256": artifact_hashes["future_seed_mapping.csv"],
        },
        "historical_registry_identity": {
            "base_seed_registry_sha256": artifact_hashes["historical_base_seed_registry.json"],
            "checkpoint_registry_sha256": artifact_hashes["historical_checkpoint_registry.csv"],
            "checkpoint_registry_metadata_sha256": artifact_hashes[
                "historical_checkpoint_registry_metadata.json"
            ],
        },
        "production_registration_identity": {
            "dataset_identity_sha256": artifact_hashes["dataset_identity.json"],
            "gate_authority_registry_sha256": artifact_hashes["gate_authority_registry.json"],
            "primary_checkpoint_registry_sha256": artifact_hashes[
                "primary_checkpoint_registry.json"
            ],
            "branch_artifact_registry_sha256": artifact_hashes[
                "branch_artifact_registry.json"
            ],
        },
        "code_test_sha256": code_test_hashes,
        "frozen_design": asdict(DESIGN),
        "scientific_design_artifacts_changed": False,
        "current_hard_gate_status": "BLOCKED",
        "current_unresolved_hard_gates": unresolved,
        "independent_conformance_audit": {"status": "ABSENT", "artifact_created": False},
        "real_execution_authorization": {"status": "ABSENT", "artifact_created": False},
        "phase5_scientific_outcomes": {"exists": False, "count": 0},
        "artifact_sha256": artifact_hashes,
    }
    write_json(output / "preflight_manifest.json", manifest)
    return manifest


def materialize_preflight_revision4(repo: Path, output: Path) -> dict[str, Any]:
    """Create outcome-free r4 evidence for only the trust-root anchoring repair."""
    repo = repo.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Refuse to overwrite revision-4 preflight artifacts")
    evidence = load_canonical_preflight(repo)
    source = evidence.path
    output.mkdir(parents=True)
    copied_names = [
        "historical_base_seed_registry.json",
        "historical_checkpoint_registry.csv",
        "historical_checkpoint_registry_metadata.json",
        "base_seed_manifest.csv",
        "future_seed_mapping.csv",
        "phase5_schema.json",
        "bootstrap_rng.json",
        "dataset_identity.json",
        "gate_authority_registry.json",
        "primary_checkpoint_registry.json",
        "branch_artifact_registry.json",
    ]
    for name in copied_names:
        shutil.copyfile(source / name, output / name)

    unresolved = [
        "revision-4 evidence has not passed an independent read-only conformance audit",
        "implementation, tests, and revision-4 evidence are not committed",
        "independent backup destination remains TBD and unverified",
        "separate user authorization for real Phase 5 base training is absent",
        "36 real epoch-14 Phase 5 primary checkpoints do not exist",
        "real primary-checkpoint non-overlap and restoration audits remain pending",
        "complete 900-assignment CRN/order-stream audit remains pending",
        "separate user authorization for real Phase 5 branching is absent",
        "separate user authorization for canonical primary analysis is absent",
    ]
    gate_status = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_evidence_revision": 4,
        "trust_root_sha256": TRUST_ROOT_SHA256,
        "status": CANDIDATE_STATUS,
        "current_hard_gate_status": "BLOCKED",
        "unresolved_execution_hard_gates": unresolved,
        "independent_conformance_audit": "ABSENT",
        "real_execution_authorization": "ABSENT",
        "backup_destination": "TBD / EXECUTION BLOCKED",
        "training_approved": False,
        "branching_approved": False,
        "analysis_approved": False,
        "phase5_scientific_outcomes_exist": False,
    }
    write_json(output / "hard_gate_status.json", gate_status)
    created_at = datetime.now(timezone.utc).isoformat()
    revision_reason = "narrow trust-root anchoring repair after independent r3 conformance audit"
    predecessor_path = f"{CANONICAL_PREFLIGHT_PATH}/preflight_manifest.json"
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_evidence_revision": 4,
        "predecessor_preflight_path": predecessor_path,
        "predecessor_manifest_sha256": REVIEWED_PREFLIGHT_MANIFEST_SHA256,
        "predecessor_status": "historical_candidate_r3_preserved_byte_identically",
        "revision_reason": revision_reason,
        "created_at_utc": created_at,
        "creation_mode": "synthetic/unit/integrity validation only; no Phase 5 execution",
        "scientific_protocol_changed": False,
        "phase5_scientific_outcomes_exist": False,
        "status": CANDIDATE_STATUS,
    }
    write_json(output / "revision_provenance.json", provenance)

    artifact_names = copied_names + ["hard_gate_status.json", "revision_provenance.json"]
    artifact_hashes = {name: sha256_file(output / name) for name in artifact_names}
    code_test_names = [
        "reflexml/phase5.py",
        "prepare_phase5.py",
        "tests/test_phase5.py",
    ]
    code_test_hashes = {name: sha256_file(repo / name) for name in code_test_names}
    source_manifest = _read_json(source / "preflight_manifest.json")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_evidence_revision": 4,
        "predecessor_preflight_path": predecessor_path,
        "predecessor_manifest_sha256": REVIEWED_PREFLIGHT_MANIFEST_SHA256,
        "predecessor_status": "historical_candidate_r3_preserved_byte_identically",
        "revision_reason": revision_reason,
        "status": CANDIDATE_STATUS,
        "artifact_status": "PRE-OUTCOME CANDIDATE EVIDENCE; CONTAINS NO PHASE 5 OUTCOMES",
        "creation_provenance": {
            "created_at_utc": created_at,
            "generator": "prepare_phase5.py / materialize_preflight_revision4",
            "r1_r2_and_r3_preserved": True,
            "scientific_design_artifacts_byte_identical": True,
        },
        "generated_counts": source_manifest["generated_counts"],
        "audits": source_manifest["audits"],
        "bootstrap": source_manifest["bootstrap"],
        "trust_root": {
            "path": TRUST_ROOT_PATH,
            "sha256": TRUST_ROOT_SHA256,
            "canonical_preflight_path": CANONICAL_PREFLIGHT_PATH,
            "reviewed_preflight_manifest_sha256": REVIEWED_PREFLIGHT_MANIFEST_SHA256,
        },
        "static_registry_authority_sha256": {
            name: artifact_hashes[name]
            for name in (
                "base_seed_manifest.csv",
                "future_seed_mapping.csv",
                "historical_base_seed_registry.json",
                "historical_checkpoint_registry.csv",
                "historical_checkpoint_registry_metadata.json",
                "phase5_schema.json",
                "bootstrap_rng.json",
                "dataset_identity.json",
                "gate_authority_registry.json",
            )
        },
        "dynamic_registry_state_sha256": {
            "primary_checkpoint_registry.json": artifact_hashes[
                "primary_checkpoint_registry.json"
            ],
            "branch_artifact_registry.json": artifact_hashes[
                "branch_artifact_registry.json"
            ],
        },
        "code_test_sha256": code_test_hashes,
        "frozen_design": asdict(DESIGN),
        "scientific_design_artifacts_changed": False,
        "current_hard_gate_status": "BLOCKED",
        "current_unresolved_hard_gates": unresolved,
        "independent_conformance_audit": {"status": "ABSENT", "artifact_created": False},
        "real_execution_authorization": {"status": "ABSENT", "artifact_created": False},
        "phase5_scientific_outcomes": {"exists": False, "count": 0},
        "artifact_sha256": artifact_hashes,
    }
    write_json(output / "preflight_manifest.json", manifest)
    return manifest


def materialize_preflight(repo: Path, output: Path) -> dict[str, Any]:
    """Compatibility name creates only revision-3 candidate evidence."""
    return materialize_preflight_revision3(repo, output)

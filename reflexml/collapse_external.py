"""Frozen Collapse External Replication v0.1; no Phase 5 execution dependencies.

The CLI is an explicit execution entrypoint, never invoked by importing this module.
Scientific runs require separate review/authorization; tests use synthetic fixtures.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
from torch import nn
from torchvision import datasets, transforms

from .branching import BranchingConfig, order_hashes, run_branch, states_equal
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import make_loaders_from_datasets, make_split_indices, split_fingerprint
from .model import FashionMLP
from .reproducibility import set_reproducible_seed
from .stability import with_future_shuffle
from .training import evaluate, train_one_epoch

PROTOCOL = "ReflexML Collapse External Replication v0.1"
ARMS = {"Now": 0, "Wait3": 3}
CPU = torch.device("cpu")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_id(i):
    if type(i) is not int or not 1 <= i <= 12:
        raise ValueError("State index must be 1..12")
    return f"state-{i:03d}"


def future_seed(i, k):
    state_id(i)
    if type(k) is not int or k not in (1, 2):
        raise ValueError("K is frozen at 2")
    return 920001 + 2 * (i - 1) + k - 1


def frozen_config(i):
    state_id(i)
    return ExperimentConfig(seed=910000 + i, epochs=14, dataset_name="MNIST")


def load_mnist(root, *, download=False):
    # No official test dataset is constructed.
    return datasets.MNIST(root=str(root), train=True, download=download,
                          transform=transforms.ToTensor())


def dataset_manifest(dataset):
    if type(dataset) is not datasets.MNIST or not dataset.train or len(dataset) != 60000:
        raise ValueError("Requires official MNIST training population")
    if type(dataset.transform) is not transforms.ToTensor or dataset.target_transform is not None:
        raise ValueError("Transform must be ToTensor only")
    digest = hashlib.sha256()
    for name, tensor in (("images", dataset.data), ("targets", dataset.targets)):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.cpu().contiguous().numpy().tobytes())
    train, val = make_split_indices(60000, 10000, 5000, 2026)
    return {
        "protocol": PROTOCOL, "dataset_class": "torchvision.datasets.MNIST",
        "source": "official MNIST training population", "root": str(Path(dataset.root).resolve()),
        "training_content_sha256": digest.hexdigest(), "transform": "ToTensor()",
        "split_seed": 2026, "train_indices": train, "validation_indices": val,
        "split_hash": split_fingerprint(train, val), "official_test_enabled": False,
        "population_size": 60000,
    }


def write_json(path, value):
    # Exclusive creation: a repeated command cannot overwrite scientific artifacts.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_records(rows, n):
    """Validate complete identities, schedules, orders and finite required outcomes."""
    if type(n) is not int or n not in (6, 12):
        return ["N must be 6 or 12"]
    expected = {(state_id(i), k, arm) for i in range(1, n + 1)
                for k in (1, 2) for arm in ARMS}
    keyed = {}
    errors = []
    train, val = make_split_indices(60000, 10000, 5000, 2026)
    expected_split = split_fingerprint(train, val)
    train_set = set(train)
    dataset_hashes = set()
    state_sources = {}
    try:
        for row in rows:
            key = (row["state_id"], row["future_id"], row["arm"])
            if key not in expected or key in keyed:
                errors.append("Unexpected or duplicate branch identity")
                continue
            keyed[key] = row
            i = int(row["state_id"].split("-")[1])
            if (row["protocol"] != PROTOCOL or row["base_seed"] != 910000 + i
                    or row["future_seed"] != future_seed(i, row["future_id"])
                    or row["config_hash"] != identity(frozen_config(i).to_dict())
                    or row["split_hash"] != expected_split):
                errors.append("Frozen protocol/config/seed/split mismatch")
            if not row["success"] or row["failure_reason"] is not None:
                errors.append("Failed branch")
            if not _finite(row["prebranch_validation_loss"]) or len(row["losses"]) != 4 or not all(
                    _finite(x) for x in row["losses"]):
                errors.append("Missing or nonfinite loss")
            if not _sha(row["dataset_identity"]) or not _sha(row["source_checkpoint_hash"]) or not row["source_checkpoint_path"]:
                errors.append("Missing dataset/source identity")
            dataset_hashes.add(row["dataset_identity"])
            source = (row["source_checkpoint_path"], row["source_checkpoint_hash"], row["prebranch_validation_loss"])
            if row["state_id"] in state_sources and state_sources[row["state_id"]] != source:
                errors.append("Inconsistent state source or prebranch loss")
            state_sources[row["state_id"]] = source
            lrs = [.05] * 4 if row["arm"] == "Now" else [.1, .1, .1, .05]
            if row["learning_rates"] != lrs:
                errors.append("Wrong intervention geometry")
            orders = row["orders"]
            if len(orders) != 4 or row["order_hashes"] != order_hashes(orders):
                errors.append("Missing/corrupt order evidence")
            for order in orders:
                flat = [index for batch in order for index in batch]
                if ([len(batch) for batch in order] != [128] * 78 + [16]
                        or len(flat) != len(set(flat)) or set(flat) != train_set
                        or any(type(index) is not int for index in flat)):
                    errors.append("Invalid minibatch identities or boundaries")
        if set(keyed) != expected:
            errors.append("Incomplete frozen roster")
        if len(dataset_hashes) != 1:
            errors.append("Dataset identities differ")
        for i in range(1, n + 1):
            for k in (1, 2):
                now = keyed.get((state_id(i), k, "Now"))
                wait = keyed.get((state_id(i), k, "Wait3"))
                if now is not None and wait is not None and now["orders"] != wait["orders"]:
                    errors.append("Actual paired minibatch sequences differ")
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        errors.append("Malformed required branch evidence")
    return sorted(set(errors))


def freeze_epsilon(rows):
    errors = validate_records(rows, 6)
    if errors:
        raise ValueError("Cannot freeze epsilon from invalid screening data: " + "; ".join(errors))
    sources = []
    for i in range(1, 7):
        row = next(r for r in rows if r["state_id"] == state_id(i))
        sources.append({"state_id": state_id(i), "base_seed": 910000 + i,
                        "source_checkpoint_hash": row["source_checkpoint_hash"],
                        "prebranch_validation_loss": row["prebranch_validation_loss"]})
    return {"epsilon": max(.001, .01 * statistics.mean(
        r["prebranch_validation_loss"] for r in sources)), "source_n": 6, "sources": sources}


def classify(rows, n=6, epsilon_record=None):
    errors = validate_records(rows, n)
    if errors:
        return {"classification": "INVALID", "errors": errors, "n": n}
    first_six = [r for r in rows if r["state_id"] in {state_id(i) for i in range(1, 7)}]
    expected_epsilon = freeze_epsilon(first_six)
    if epsilon_record is None and n == 6:
        epsilon_record = expected_epsilon
    if epsilon_record != expected_epsilon:
        return {"classification": "INVALID", "errors": ["Missing or altered frozen N=6 epsilon provenance"], "n": n}
    epsilon = epsilon_record["epsilon"]
    keyed = {(r["state_id"], r["future_id"], r["arm"]): r for r in rows}
    d = [[statistics.mean(keyed[(state_id(i), k, "Wait3")]["losses"][h] -
                          keyed[(state_id(i), k, "Now")]["losses"][h] for k in (1, 2))
          for h in range(4)] for i in range(1, n + 1)]
    if not all(_finite(x) for row in d for x in row):
        return {"classification": "INVALID", "errors": ["Nonfinite derived contrast"], "n": n}
    m = [statistics.mean(row[h] for row in d) for h in range(4)]
    early = [statistics.mean(row[:3]) for row in d]
    e = statistics.mean(m[:3])
    q = 5 if n == 6 else 10
    go_count = sum(ei > 0 and row[3] < ei for ei, row in zip(early, d))
    absent_count = sum(ei > epsilon and row[3] >= .8 * ei for ei, row in zip(early, d))
    reversed_count = sum(ei < -epsilon for ei in early)
    small_count = sum(max(abs(x) for x in row[:3]) <= epsilon for row in d)
    if (min(m[:3]) > epsilon and max(m[:3]) / min(m[:3]) <= 2
            and e > epsilon and abs(m[3]) / e <= .5
            and m[2] - m[3] > max(abs(m[1] - m[0]), abs(m[2] - m[1])) and go_count >= q):
        label = "GO"
    elif e > epsilon and m[3] / e >= .8 and absent_count >= q:
        label = "NO_GO_COLLAPSE_ABSENT"
    elif max(m[:3]) < -epsilon and reversed_count >= q:
        label = "NO_GO_DIRECTION_REVERSED"
    elif max(abs(x) for x in m[:3]) <= epsilon and small_count >= q:
        label = "UNINFORMATIVE"
    else:
        label = "AMBIGUOUS"
    return {"protocol": PROTOCOL, "classification": label, "n": n, "k": 2, "q_n": q,
            "epsilon_record": deepcopy(epsilon_record), "state_ids": [state_id(i) for i in range(1, n + 1)],
            "d_ih": d, "m_h": m, "E_i": early, "E": e, "A34": m[3] - m[2],
            "counts": {"go": go_count, "absent": absent_count, "reversed": reversed_count, "small": small_count},
            "errors": []}


def escalation_decision(classification, n, *, expansions=0, k=2):
    if n not in (6, 12) or k != 2 or expansions not in (0, 1):
        raise ValueError("Frozen N/K/one-expansion rule violated")
    if classification not in {"INVALID", "GO", "NO_GO_COLLAPSE_ABSENT", "NO_GO_DIRECTION_REVERSED", "UNINFORMATIVE", "AMBIGUOUS"}:
        raise ValueError("Unknown classification")
    if classification == "INVALID":
        return "TECHNICAL_REVIEW"
    if n == 6 and classification == "AMBIGUOUS" and expansions == 0:
        return "ELIGIBLE_N12"
    return "STOP"


def run_pair(checkpoint, dataset, seed):
    """Kernel adapter; tests may use tiny synthetic checkpoints/datasets."""
    original = deepcopy(checkpoint)
    future = with_future_shuffle(checkpoint, seed)
    settings = BranchingConfig(horizon=4)
    now = run_branch(future, dataset, settings, .5, end_epoch=18, capture_epoch=17)
    wait = run_branch(future, dataset, settings, .5, expected_initial=now["before"],
                      delay_epochs=3, end_epoch=18, capture_epoch=17)
    if now["orders"] != wait["orders"]:
        raise ValueError("Actual Now/Wait3 minibatch order mismatch")
    if not states_equal(original, checkpoint) or not states_equal(future, with_future_shuffle(original, seed)):
        raise AssertionError("Branch mutated its source checkpoint")
    return {"Now": now, "Wait3": wait}


def replay_checkpoint(capture):
    """Convert persisted h4-start state to the existing checkpoint restore format."""
    state = capture["next_epoch_start"]
    return deepcopy({"epoch": capture["epoch"], "model_state_dict": state["model"],
                     "optimizer_state_dict": state["optimizer"],
                     "train_loader_generator_state": state["generator"], **state["rng"],
                     "config": capture["config"], "split_fingerprint": capture["split_fingerprint"],
                     "current_learning_rate": state["optimizer"]["param_groups"][0]["lr"]})


def _run_state(i, dataset, manifest, output):
    directory = output / state_id(i)
    directory.mkdir()  # No retry/overwrite policy is silently introduced.
    config = frozen_config(i)
    set_reproducible_seed(config.seed)
    data = make_loaders_from_datasets(dataset, [], config, audit_order=True)
    model = FashionMLP(config.hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    updates = 0
    history = []
    for epoch in range(1, 15):
        order = []
        train_one_epoch(model, data.train_loader, optimizer, criterion, CPU, order)
        updates += len(order)
        validation = evaluate(model, data.val_loader, criterion, CPU)
        history.append({"epoch": epoch, "val_loss": validation.loss})
    if updates != 1106:
        raise AssertionError("Base training must have exactly 1106 updates")
    path = directory / "base-epoch14.pt"
    save_checkpoint(path, 14, model, optimizer, config, data.train_generator, data.split_fingerprint, history)
    source_hash = file_hash(path)
    pre = history[-1]["val_loss"]
    meta = {"protocol": PROTOCOL, "state_id": state_id(i), "base_seed": config.seed,
            "dataset_identity": identity(manifest), "config_hash": identity(config.to_dict()),
            "split_hash": data.split_fingerprint, "source_checkpoint_path": str(path.resolve()),
            "source_checkpoint_hash": source_hash, "prebranch_validation_loss": pre,
            "exact_update_count": updates}
    # JSON cannot encode NaN; preserve failed numeric outcomes as null + failure.
    if not _finite(pre):
        write_json(directory / "failure.json", {**meta, "prebranch_validation_loss": None,
                                                "success": False, "failure_reason": "Nonfinite base validation"})
        raise ValueError("Nonfinite base validation")
    write_json(directory / "state.json", meta)
    rows = []
    checkpoint = load_checkpoint(path, CPU)
    for k in (1, 2):
        common = {**meta, "future_id": k, "future_seed": future_seed(i, k)}
        try:
            branches = run_pair(checkpoint, dataset, future_seed(i, k))
            if file_hash(path) != source_hash:
                raise AssertionError("Source checkpoint file changed")
            for arm, branch in branches.items():
                losses = [r["val_loss"] for r in branch["history"]]
                finite = all(_finite(x) for x in losses)
                capture = branch["captured_state"]
                artifact = directory / f"future-{k}-{arm}-h3.pt"
                torch.save({**capture, **common, "arm": arm,
                            "original_h4_validation_loss": losses[-1],
                            "original_h4_endpoint": branch["final_state"],
                            "replay_checkpoint": replay_checkpoint(capture)}, artifact)
                row = {**common, "arm": arm, "losses": [x if _finite(x) else None for x in losses],
                       "learning_rates": [r["learning_rate"] for r in branch["history"]],
                       "orders": branch["orders"], "order_hashes": order_hashes(branch["orders"]),
                       "success": finite, "failure_reason": None if finite else "Nonfinite horizon loss",
                       "h3_artifact": {"path": str(artifact.resolve()), "sha256": file_hash(artifact)}}
                write_json(directory / f"future-{k}-{arm}.json", row)
                rows.append(row)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            for arm in ARMS:
                arm_path = directory / f"future-{k}-{arm}.json"
                if not arm_path.exists():
                    write_json(arm_path, {**common, "arm": arm, "losses": [None] * 4,
                               "learning_rates": [], "orders": [], "order_hashes": [],
                               "h3_artifact": None, "success": False, "failure_reason": failure})
            write_json(directory / f"future-{k}-failure.json", {**common, "success": False,
                        "failure_reason": failure})
            raise  # No automatic retry, substitution or available-case analysis.
    return rows


def run_replication(root, output, *, download=False, initial_output=None):
    """Explicit screening or one-shot expansion; never automatically expand N."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output directory already exists; technical repair needs a separate rule")
    old_rows = []
    epsilon_record = None
    if initial_output is not None:
        initial = Path(initial_output).resolve()
        old_rows = json.loads((initial / "records.json").read_text())
        saved = json.loads((initial / "analysis.json").read_text())
        checked = classify(old_rows, 6, saved.get("epsilon_record"))
        if saved != checked or escalation_decision(checked["classification"], 6) != "ELIGIBLE_N12":
            raise ValueError("Expansion requires verified N=6 AMBIGUOUS result")
        epsilon_record = checked["epsilon_record"]
        # Recheck the original base files, never rerun the initial six states.
        for row in old_rows:
            if file_hash(row["source_checkpoint_path"]) != row["source_checkpoint_hash"]:
                raise ValueError("Initial source checkpoint hash changed")
    dataset = load_mnist(root, download=download)
    manifest = dataset_manifest(dataset)
    if initial_output is not None and json.loads((initial / "manifest.json").read_text()) != manifest:
        raise ValueError("Expansion dataset/root/split identity changed")
    if initial_output is not None and any(r["dataset_identity"] != identity(manifest) for r in old_rows):
        raise ValueError("Initial branch dataset identity does not bind the manifest")
    if initial_output is not None:
        # Exclusive claim makes this a one-shot expansion, including failed attempts.
        write_json(initial / "expansion.json", {"protocol": PROTOCOL, "output": str(output),
                                               "n": 12, "k": 2})
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    try:
        rows = list(old_rows)
        for i in (range(1, 7) if initial_output is None else range(7, 13)):
            rows.extend(_run_state(i, dataset, manifest, output))
        write_json(output / "records.json", rows)
        result = classify(rows, 6 if initial_output is None else 12, epsilon_record)
        write_json(output / "analysis.json", result)
        return result
    except Exception as exc:
        write_json(output / "failure.json", {"protocol": PROTOCOL, "classification": "INVALID",
                   "success": False, "failure_reason": f"{type(exc).__name__}: {exc}"})
        raise

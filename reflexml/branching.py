"""固定动作的配对实验。这里只生成监督数据，不进行任何策略决策。"""

from copy import deepcopy
from dataclasses import dataclass
import base64
import hashlib
import io
import json

import numpy as np
import torch
from torch import nn

from .checkpoint import restore_checkpoint
from .config import ExperimentConfig
from .data import make_loaders_from_datasets
from .model import FashionMLP
from .reproducibility import capture_rng_states
from .training import evaluate, train_one_epoch


@dataclass
class BranchingConfig:
    seeds: tuple[int, ...] = tuple(range(42, 50))
    checkpoint_epochs: tuple[int, ...] = (8, 11, 14, 17, 20, 23)
    horizon: int = 5
    lr_factor: float = 0.5
    meaningful_improvement_threshold: float = 0.005
    window: int = 5
    final_average_epochs: int = 2
    improvement_cap: int = 10
    epsilon: float = 1e-12


def states_equal(left, right):
    """比较实际状态值，而不是 .pt 文件的序列化字节。"""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            states_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            states_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def training_state(model, optimizer, generator):
    return deepcopy({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "rng": capture_rng_states(),
    })


def _floating_tensor_evidence(value, prefix=""):
    """Return lossless per-tensor evidence used to classify non-finite state.

    Each floating tensor is stored as a canonical NumPy ``.npy`` byte stream.
    The later verifier reloads those numeric bytes and independently derives
    dtype, shape, value hash, and non-finite count.  The summary fields here are
    cross-checks only; they are never the verifier's source of truth.
    """
    rows = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            tensor = value.detach().cpu().contiguous()
            array = tensor.numpy()
            state_buffer = io.BytesIO()
            np.save(state_buffer, array, allow_pickle=False)
            state_bytes = state_buffer.getvalue()
            finite = torch.isfinite(tensor)
            rows.append(
                {
                    "state_path": prefix,
                    "path": prefix,
                    "state_encoding": "numpy_npy_base64_v1",
                    "state_npy_base64": base64.b64encode(state_bytes).decode("ascii"),
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "value_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                    "nonfinite_count": int((~finite).sum().item()),
                }
            )
        return rows
    if isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_floating_tensor_evidence(value[key], child))
        return rows
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            rows.extend(_floating_tensor_evidence(item, child))
    return rows


def _floating_tensor_inventory(value, prefix=""):
    """Describe floating tensor structure without trusting persisted evidence."""
    rows = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            array = value.detach().cpu().contiguous().numpy()
            rows.append(
                {
                    "state_path": prefix,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
            )
        return rows
    if isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_floating_tensor_inventory(value[key], child))
        return rows
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            rows.extend(_floating_tensor_inventory(item, child))
    return rows


def training_state_tensor_inventory(model, optimizer, *, observation_stage):
    """Return the complete floating model/optimizer tensor inventory."""
    return {
        "observation_stage": observation_stage,
        "model_parameters": _floating_tensor_inventory(model.state_dict(), "model"),
        "optimizer_state": _floating_tensor_inventory(
            optimizer.state_dict().get("state", {}), "optimizer.state"
        ),
    }


def expected_training_state_tensor_inventory(
    config, *, observation_stage="post_training_epoch"
):
    """Derive the required inventory from the frozen implementation semantics.

    Phase 5 records state evidence only after a non-empty training epoch, so
    every trainable parameter has participated in at least one optimizer step.
    PyTorch SGD creates momentum buffers lazily on that first step.  A zero-grad
    step on an isolated model therefore initializes the same structural state
    without executing training or depending on scientific data or outcomes.
    """
    if observation_stage != "post_training_epoch":
        raise ValueError("Unsupported Phase 5 state-evidence observation stage")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = FashionMLP(config.hidden_size).to(dtype=torch.float32)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, momentum=config.momentum
        )
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        return training_state_tensor_inventory(
            model, optimizer, observation_stage=observation_stage
        )


def training_state_finiteness_evidence(model, optimizer):
    """Persist enough canonical state evidence for protocol divergence checks."""
    return {
        "model_parameters": _floating_tensor_evidence(model.state_dict(), "model"),
        "optimizer_state": _floating_tensor_evidence(
            optimizer.state_dict().get("state", {}), "optimizer.state"
        ),
    }


def change_lr_only(model, optimizer, generator, new_lr):
    """每次固定干预时都核查：只主动改变 LR，不重置动量或随机源。"""
    before = training_state(model, optimizer, generator)
    optimizer.param_groups[0]["lr"] = new_lr
    expected = deepcopy(before)
    expected["optimizer"]["param_groups"][0]["lr"] = new_lr
    if not states_equal(training_state(model, optimizer, generator), expected):
        raise AssertionError("Changing LR unexpectedly modified other training state")


def run_branch(checkpoint, training_dataset, settings, lr_factor, expected_initial=None,
               delay_epochs=0, end_epoch=None, *, capture_epoch=None):
    config = ExperimentConfig(**checkpoint["config"])
    if config.device != "cpu" or config.num_workers != 0:
        raise ValueError("Phase 2 audits currently require CPU and num_workers=0")
    # 不载入官方测试集。空的 test_dataset 只满足已有 DataBundle 的接口，永不迭代。
    data = make_loaders_from_datasets(training_dataset, [], config, audit_order=True)
    model = FashionMLP(config.hidden_size)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum
    )
    # optimizer.load_state_dict 可能引用输入张量，必须隔离各分支的动量存储。
    restore_checkpoint(
        deepcopy(checkpoint), model, optimizer, data.train_generator,
        data.split_fingerprint,
    )
    before = training_state(model, optimizer, data.train_generator)
    expected_restored = {
        "model": checkpoint["model_state_dict"],
        "optimizer": checkpoint["optimizer_state_dict"],
        "generator": checkpoint["train_loader_generator_state"],
        "rng": {key: checkpoint[key] for key in before["rng"]},
    }
    if not states_equal(before, expected_restored):
        raise AssertionError("Restoration did not reproduce checkpoint state")
    if expected_initial is not None and not states_equal(before, expected_initial):
        raise AssertionError("A/B initial states differ before B's first update")
    final_epoch = checkpoint["epoch"] + settings.horizon if end_epoch is None else end_epoch
    first_reduced_epoch = checkpoint["epoch"] + delay_epochs + 1
    if delay_epochs < 0 or first_reduced_epoch > final_epoch:
        raise ValueError("Delay must leave at least one reduced-LR training epoch")
    if capture_epoch is not None and (type(capture_epoch) is not int
                                     or not checkpoint["epoch"] < capture_epoch < final_epoch):
        raise ValueError("Capture epoch must precede a following training epoch")
    initial_lr = optimizer.param_groups[0]["lr"]
    if delay_epochs == 0:
        change_lr_only(model, optimizer, data.train_generator, initial_lr * lr_factor)
    after = training_state(model, optimizer, data.train_generator)

    history = []
    orders = []
    random_states = []
    state_finiteness_evidence = []
    criterion = nn.CrossEntropyLoss()
    captured = None
    for epoch in range(checkpoint["epoch"] + 1, final_epoch + 1):
        if delay_epochs > 0 and epoch == first_reduced_epoch:
            change_lr_only(model, optimizer, data.train_generator, initial_lr * lr_factor)
        if capture_epoch is not None and epoch == capture_epoch + 1:
            # After any scheduled LR change, before iterator creation or updates.
            captured["next_epoch_start"] = training_state(model, optimizer, data.train_generator)
        random_states.append(deepcopy({
            "rng": capture_rng_states(), "generator": data.train_generator.get_state(),
        }))
        batch_order = []
        loss = train_one_epoch(
            model, data.train_loader, optimizer, criterion, torch.device("cpu"), batch_order
        )
        validation = evaluate(model, data.val_loader, criterion, torch.device("cpu"))
        orders.append(batch_order)
        history.append({
            "epoch": epoch, "train_loss": loss, "val_loss": validation.loss,
            "val_accuracy": validation.accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        state_finiteness_evidence.append(
            {
                "epoch": epoch,
                **training_state_finiteness_evidence(model, optimizer),
            }
        )
        if epoch == capture_epoch:
            captured = {
                "epoch": epoch,
                "epoch_end": training_state(model, optimizer, data.train_generator),
                "config": deepcopy(checkpoint["config"]),
                "split_fingerprint": checkpoint["split_fingerprint"],
                "source_epoch": checkpoint["epoch"],
                "execution": {
                    "device": config.device, "num_workers": config.num_workers,
                    "model_training": model.training,
                    "torch_version": str(torch.__version__),
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                },
            }
    result = {
        "before": before, "after": after, "history": history, "orders": orders,
        "final_state": training_state(model, optimizer, data.train_generator),
        "epoch_start_random_states": random_states,
        "state_finiteness_evidence": state_finiteness_evidence,
    }
    if capture_epoch is not None:
        captured["next_epoch_order"] = deepcopy(orders[capture_epoch - checkpoint["epoch"]])
        result["captured_state"] = captured
    return result


def paired_branches(checkpoint, training_dataset, settings, lr_factor=None):
    factor = settings.lr_factor if lr_factor is None else lr_factor
    original = deepcopy(checkpoint)
    branch_a = run_branch(checkpoint, training_dataset, settings, 1.0)
    branch_b = run_branch(checkpoint, training_dataset, settings, factor, branch_a["before"])
    if not states_equal(branch_a["before"], branch_b["before"]):
        raise AssertionError("A/B initial states differ")
    if branch_a["orders"] != branch_b["orders"]:
        raise AssertionError("A/B minibatch index order differs")
    if not states_equal(original, checkpoint):
        raise AssertionError("A branch mutated the source checkpoint")
    if factor == 1.0:
        if (branch_a["history"] != branch_b["history"]
                or not states_equal(branch_a["final_state"], branch_b["final_state"])):
            raise AssertionError("Equal-LR branches failed exact reproduction")
    return branch_a, branch_b


def action_outcome(branch_a_history, branch_b_history, settings):
    if len(branch_a_history) != settings.horizon or len(branch_b_history) != settings.horizon:
        raise ValueError("Both branches must have exactly horizon epochs")
    mean_a = float(np.mean([r["val_loss"] for r in branch_a_history[-settings.final_average_epochs:]]))
    mean_b = float(np.mean([r["val_loss"] for r in branch_b_history[-settings.final_average_epochs:]]))
    if not np.isfinite([mean_a, mean_b]).all() or min(mean_a, mean_b) < 0:
        raise ValueError("Cannot label nonfinite or negative validation loss")
    gain = (mean_a - mean_b) / max(mean_a, settings.epsilon)
    return {
        "branch_a_final_val_loss": mean_a,
        "branch_b_final_val_loss": mean_b,
        "relative_gain": gain,
        "beneficial": int(gain > settings.meaningful_improvement_threshold),
    }


def order_hashes(orders):
    # orders 保留 batch 边界；哈希仅用于小体积审计存档，运行时直接比较完整索引列表。
    return [hashlib.sha256(json.dumps(order).encode()).hexdigest() for order in orders]

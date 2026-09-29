"""固定预算下的降 LR 时机实验；不学习或选择动作。"""

from copy import deepcopy
from dataclasses import dataclass
import math
import statistics

from .branching import BranchingConfig, run_branch, states_equal


@dataclass
class TimingConfig:
    seeds: tuple[int, ...] = (42, 43, 44)
    checkpoint_epochs: tuple[int, ...] = (8, 11, 14, 17, 20, 23)
    end_epoch: int = 30
    delay_epochs: int = 3
    lr_factor: float = 0.5
    final_average_epochs: int = 3
    approximate_tie_limit: float = 0.002
    epsilon: float = 1e-12


def timing_pair(checkpoint, dataset, settings, same_schedule=False):
    t = checkpoint["epoch"]
    delayed_epoch = t + settings.delay_epochs + 1
    # 保证延迟分支在最后三轮评估窗口中已经使用 reduced LR。
    if settings.delay_epochs < 0 or delayed_epoch > settings.end_epoch - settings.final_average_epochs + 1:
        raise ValueError("Checkpoint leaves fewer than three reduced-LR epochs after waiting")
    original = deepcopy(checkpoint)
    common = BranchingConfig()
    now = run_branch(checkpoint, dataset, common, settings.lr_factor, end_epoch=settings.end_epoch)
    wait = run_branch(checkpoint, dataset, common, settings.lr_factor,
                      expected_initial=now["before"],
                      delay_epochs=0 if same_schedule else settings.delay_epochs,
                      end_epoch=settings.end_epoch)
    if not states_equal(now["before"], wait["before"]):
        raise AssertionError("Initial training states differ")
    if now["orders"] != wait["orders"]:
        raise AssertionError("Actual minibatch index order differs")
    if not states_equal(now["epoch_start_random_states"], wait["epoch_start_random_states"]):
        raise AssertionError("RNG or loader generator diverged between branches")
    for key in ("rng", "generator"):
        if not states_equal(now["final_state"][key], wait["final_state"][key]):
            raise AssertionError("Final random states differ")
    if not states_equal(checkpoint, original):
        raise AssertionError("Source checkpoint was mutated")
    if same_schedule and (now["history"] != wait["history"] or
                          not states_equal(now["final_state"], wait["final_state"])):
        raise AssertionError("Identical schedules failed exact reproduction")
    return now, wait


def timing_outcome(now_history, wait_history, settings):
    if [r["epoch"] for r in now_history] != [r["epoch"] for r in wait_history]:
        raise ValueError("Branches must cover the same epochs")
    expected = list(range(settings.end_epoch - settings.final_average_epochs + 1, settings.end_epoch + 1))
    if [r["epoch"] for r in now_history[-settings.final_average_epochs:]] != expected:
        raise ValueError("Final evaluation window must end at the fixed budget")
    loss_now = statistics.mean(r["val_loss"] for r in now_history[-settings.final_average_epochs:])
    loss_wait = statistics.mean(r["val_loss"] for r in wait_history[-settings.final_average_epochs:])
    if not all(math.isfinite(v) and v >= 0 for v in (loss_now, loss_wait)):
        raise ValueError("Cannot label nonfinite/negative validation loss")
    gain = (loss_wait - loss_now) / max(loss_wait, settings.epsilon)
    return {"reduce_now_final_val_loss": loss_now, "wait_final_val_loss": loss_wait,
            "timing_relative_gain": gain, "reduce_now_better": int(gain > 0),
            "approximate_tie": int(abs(gain) < settings.approximate_tie_limit)}


def summarize_timing(rows):
    gains = [r["timing_relative_gain"] for r in rows]
    return {"sample_count": len(rows), "reduce_now_better_count": sum(g > 0 for g in gains),
            "wait_better_count": sum(g < 0 for g in gains), "exact_tie_count": sum(g == 0 for g in gains),
            "approximate_tie_count": sum(r["approximate_tie"] for r in rows),
            "mean": statistics.mean(gains), "median": statistics.median(gains),
            "min": min(gains), "max": max(gains)}

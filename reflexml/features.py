"""四个可解释特征；不读取 checkpoint 之后的数据。"""

import numpy as np


def extract_features(history, current_epoch, total_epochs, settings):
    past = [row for row in history if row["epoch"] <= current_epoch]
    if [row["epoch"] for row in past] != list(range(1, current_epoch + 1)):
        raise ValueError("History must contain each epoch from 1 to current_epoch")
    if len(past) < settings.window:
        raise ValueError("Insufficient history for the feature window")
    values = np.array([[r["train_loss"], r["val_loss"]] for r in past])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Loss history must be finite and nonnegative")

    def relative_slope(column):
        window = values[-settings.window:, column]
        times = np.arange(settings.window, dtype=float)
        centered_times = times - times.mean()
        slope = np.dot(centered_times, window - window.mean()) / np.dot(
            centered_times, centered_times
        )
        return float(slope / (window.mean() + settings.epsilon))

    # 第一次观测只初始化，不作为一次有证据的改善。
    # best 总是此前所有观测的最小值，包括不足阈值的小幅改善。
    best = past[0]["val_loss"]
    last_improvement_epoch = past[0]["epoch"]
    for row in past[1:]:
        relative_improvement = (best - row["val_loss"]) / max(best, settings.epsilon)
        if relative_improvement >= settings.meaningful_improvement_threshold:
            last_improvement_epoch = row["epoch"]
        best = min(best, row["val_loss"])
    elapsed = current_epoch - last_improvement_epoch
    return {
        "train_relative_slope": relative_slope(0),
        "val_relative_slope": relative_slope(1),
        "epochs_since_improvement": min(elapsed, settings.improvement_cap)
        / settings.improvement_cap,
        "training_progress": current_epoch / total_epochs,
    }

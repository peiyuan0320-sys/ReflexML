"""固定训练状态，只重新指定未来 DataLoader shuffle 随机流。"""

from copy import deepcopy
import statistics

import torch

from .branching import states_equal


def with_future_shuffle(checkpoint, future_seed):
    """返回独立副本；模型、动量、历史和全局 RNG 均保持原样。"""
    replica = deepcopy(checkpoint)
    generator = torch.Generator().manual_seed(future_seed)
    replica['train_loader_generator_state'] = generator.get_state()
    unchanged = deepcopy(replica)
    unchanged['train_loader_generator_state'] = checkpoint['train_loader_generator_state']
    if not states_equal(unchanged, checkpoint):
        raise AssertionError('Future shuffle initialization changed other checkpoint state')
    return replica


def summarize_replicas(rows, original_gain, epsilon=1e-12):
    gains = [row['timing_gain'] for row in rows]
    mean = statistics.mean(gains)
    std = statistics.stdev(gains)  # K-1：样本标准差，不是均值标准误。
    positive = sum(g > 0 for g in gains)
    negative = sum(g < 0 for g in gains)
    tie = sum(g == 0 for g in gains)
    return {
        'replica_count': len(gains), 'original_phase3_gain': original_gain,
        'mean_gain': mean, 'std_gain': std, 'variance_gain': std ** 2,
        'min_gain': min(gains), 'max_gain': max(gains),
        'signs': ''.join('+' if g > 0 else '-' if g < 0 else '0' for g in gains),
        'positive_count': positive, 'negative_count': negative, 'tie_count': tie,
        'approximate_tie_count': sum(abs(g) < .002 for g in gains),
        'dominant_action_consistency': max(positive, negative) / len(gains),
        'effect_to_spread_ratio': abs(mean) / (std + epsilon),
        'original_mean_same_sign': int((original_gain > 0) - (original_gain < 0) == (mean > 0) - (mean < 0)),
        'original_outside_replica_range': int(original_gain < min(gains) or original_gain > max(gains)),
        'original_distance_in_replica_std': abs(original_gain - mean) / (std + epsilon),
    }

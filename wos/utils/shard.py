"""
    For sharding repeative operations
"""

from typing import Callable

import torch


def shard_op(op: Callable, n_per_shard: int, *args, **kwargs):
    """
    op: operation to be sharded
    n_shard: number of shards
    args: arguments of op
    kwargs: keyword arguments of op
    """
    n_sample = kwargs["n_sample"]

    n_shard = max(n_sample // n_per_shard, 1)

    running_mean = torch.zeros([n_sample, 1], device=kwargs["device"])
    running_var = torch.zeros_like([n_sample, 1], device=kwargs["device"])

    if n_shard == 1:
        return op(*args, **kwargs)
    else:
        for i in range(n_shard):
            n_sample_per_shard = min(n_per_shard, n_sample - i * n_per_shard)
            y, y2 = op(n_sample_per_shard=n_sample_per_shard, *args, **kwargs)
            running_mean += y
            running_var += y2
        running_mean /= n_shard
        running_var = running_var / n_shard - running_mean * running_mean
        return running_mean, running_var

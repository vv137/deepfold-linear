"""Scatter utilities."""

import torch


def scatter_mean(
    values: torch.Tensor,
    indices: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Compute mean of values grouped by indices.

    Args:
        values:       (M, D) values to aggregate
        indices:      (M,) int segment indices in [0, num_segments)
        num_segments: number of output segments

    Returns:
        (num_segments, D) mean-aggregated values
    """
    D = values.shape[-1]
    agg = torch.zeros(num_segments, D, device=values.device, dtype=values.dtype)
    count = torch.zeros(num_segments, 1, device=values.device, dtype=values.dtype)
    idx_expand = indices.unsqueeze(-1).expand_as(values)
    agg.scatter_add_(0, idx_expand, values)
    count.scatter_add_(
        0,
        indices.unsqueeze(-1),
        torch.ones(values.shape[0], 1, device=values.device, dtype=values.dtype),
    )
    return agg / count.clamp(min=1)

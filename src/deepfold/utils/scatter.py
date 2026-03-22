"""Scatter utilities."""

import torch


def _scatter_mean_2d(
    values: torch.Tensor,
    indices: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Unbatched scatter_mean: values (M, D), indices (M,) -> (num_segments, D)."""
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


def scatter_mean(
    values: torch.Tensor,
    indices: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Compute mean of values grouped by indices.

    Supports both unbatched and batched inputs:
        Unbatched: values (M, D), indices (M,) -> (num_segments, D)
        Batched:   values (B, M, D), indices (B, M) -> (B, num_segments, D)

    Uses batch-offset trick for efficient batched scatter without loops.

    Args:
        values:       (M, D) or (B, M, D) values to aggregate
        indices:      (M,) or (B, M) int segment indices in [0, num_segments)
        num_segments: number of output segments

    Returns:
        (num_segments, D) or (B, num_segments, D) mean-aggregated values
    """
    if values.dim() == 2:
        return _scatter_mean_2d(values, indices, num_segments)

    # Batched path: (B, M, D) values, (B, M) indices
    B, M, D = values.shape

    # Offset indices per batch element so they don't collide when flattened
    offsets = torch.arange(B, device=indices.device).unsqueeze(1) * num_segments  # (B, 1)
    flat_indices = (indices + offsets).reshape(-1)  # (B*M,)
    flat_values = values.reshape(-1, D)  # (B*M, D)

    result = _scatter_mean_2d(flat_values, flat_indices, B * num_segments)
    return result.reshape(B, num_segments, D)

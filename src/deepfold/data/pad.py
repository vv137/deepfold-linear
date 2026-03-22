"""Padding utilities for variable-length tensors in collate functions.

Adapted from Boltz-1's pad.py for use in DeepFold-Linear data loaders.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import pad


def pad_dim(data: Tensor, dim: int, pad_len: int, value: float = 0) -> Tensor:
    """Pad a tensor along a single dimension.

    Parameters
    ----------
    data : Tensor
        Input tensor of any shape.
    dim : int
        The dimension to pad (0-indexed).
    pad_len : int
        Number of elements to pad at the end of ``dim``.
    value : float
        Fill value for padded positions.

    Returns
    -------
    Tensor
        Padded tensor.
    """
    if pad_len == 0:
        return data

    total_dims = len(data.shape)
    # torch.nn.functional.pad expects padding from last dim backwards:
    # (last_left, last_right, ..., first_left, first_right)
    padding = [0] * (2 * (total_dims - dim))
    padding[2 * (total_dims - 1 - dim) + 1] = pad_len
    return pad(data, tuple(padding), value=value)


def pad_to_max(
    data: list[Tensor],
    value: float = 0,
) -> tuple[Tensor, Tensor]:
    """Pad a list of tensors to the maximum size in each dimension, then stack.

    This is the core utility for collating variable-length samples into a
    batch.  It returns both the padded batch tensor and a mask tensor
    (1 = real data, 0 = padding).

    Parameters
    ----------
    data : list[Tensor]
        List of tensors with the same number of dimensions but potentially
        different sizes.  All tensors must share the same dtype.
    value : float
        Fill value for padded positions.

    Returns
    -------
    batched : Tensor
        (B, *max_shape) stacked and padded tensor.
    mask : Tensor
        (B, *max_shape) float tensor — 1 where data is real, 0 where padded.
        If no padding was needed (all shapes equal), returns a scalar 0
        as a sentinel (callers can check ``mask is not 0``).
    """
    # Fast path: strings (pass through unchanged)
    if isinstance(data[0], str):
        return data, 0

    # Fast path: all same shape — no padding needed
    if all(d.shape == data[0].shape for d in data):
        return torch.stack(data, dim=0), 0

    num_dims = len(data[0].shape)
    max_dims = [max(d.shape[i] for d in data) for i in range(num_dims)]

    # Build per-sample padding specs (torch pad format: last dim first)
    pad_specs = []
    for d in data:
        spec = []
        for i in range(num_dims):
            spec.append(0)  # pad-left = 0
            spec.append(max_dims[num_dims - 1 - i] - d.shape[num_dims - 1 - i])
        pad_specs.append(spec)

    # Pad data and build masks
    padded = [pad(d, p, value=value) for d, p in zip(data, pad_specs)]
    masks = [
        pad(torch.ones_like(d, dtype=torch.float32), p, value=0)
        for d, p in zip(data, pad_specs)
    ]

    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


def collate_field(
    samples: list[dict[str, Tensor]],
    key: str,
    value: float = 0,
) -> tuple[Tensor, Tensor]:
    """Extract a field from a list of sample dicts and pad to max.

    Convenience wrapper around :func:`pad_to_max` for use in custom
    collate functions.

    Parameters
    ----------
    samples : list[dict[str, Tensor]]
        Batch of sample dictionaries.
    key : str
        Dictionary key to extract.
    value : float
        Padding fill value.

    Returns
    -------
    tuple[Tensor, Tensor]
        Padded batch tensor and mask tensor (see :func:`pad_to_max`).
    """
    tensors = [s[key] for s in samples]
    return pad_to_max(tensors, value=value)

"""Spatial cropping for DeepFold-Linear (SPEC §10).

Spatial crop: pick a random seed token, take the N_crop nearest tokens
by C-alpha (or center atom) distance. All atoms belonging to selected
tokens are included. Crop indices preserve global_idx for correct
position bin computation (SPEC §4).

Crop schedule (SPEC §10.3):
    Stage 1 (0-100K steps):   256 tokens
    Stage 2 (100K-300K):      384 tokens
    Stage 3 (300K-500K):      512 tokens
    Fine-tuning (500K+):      768 tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Crop schedule
# ---------------------------------------------------------------------------

DEFAULT_CROP_SCHEDULE = [
    (0, 256),
    (100_000, 384),
    (300_000, 512),
    (500_000, 768),
]

# Module-level schedule, overridable via set_crop_schedule()
_crop_schedule = list(DEFAULT_CROP_SCHEDULE)


def set_crop_schedule(schedule: list[tuple[int, int] | list[int]]) -> None:
    """Override the crop schedule (e.g. from config YAML)."""
    global _crop_schedule
    _crop_schedule = [(int(s), int(c)) for s, c in schedule]


def get_crop_size(step: int) -> int:
    """Return the crop size (number of tokens) for the current training step."""
    crop = _crop_schedule[0][1]
    for threshold, size in _crop_schedule:
        if step >= threshold:
            crop = size
    return crop


# ---------------------------------------------------------------------------
# Crop result
# ---------------------------------------------------------------------------


@dataclass
class CropResult:
    """Container for crop outputs.

    Attributes
    ----------
    token_indices : np.ndarray
        (N_crop,) int — indices into the original token array.
    atom_indices : np.ndarray
        (N_atom_crop,) int — indices into the original atom array.
    token_to_atom_start : np.ndarray
        (N_crop,) int — for each cropped token, start offset in cropped atom array.
    token_to_atom_count : np.ndarray
        (N_crop,) int — number of atoms per cropped token.
    """

    token_indices: np.ndarray
    atom_indices: np.ndarray
    token_to_atom_start: np.ndarray
    token_to_atom_count: np.ndarray


# ---------------------------------------------------------------------------
# Spatial crop
# ---------------------------------------------------------------------------


def spatial_crop(
    center_coords: np.ndarray,
    n_crop: int,
    atom_starts: np.ndarray,
    atom_counts: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
    seed_idx: Optional[int] = None,
) -> CropResult:
    """Pure spatial crop: pick seed token, take N_crop nearest by distance.

    Parameters
    ----------
    center_coords : np.ndarray
        (N, 3) — center atom (C-alpha / C1') coordinates for each token.
    n_crop : int
        Number of tokens to keep.
    atom_starts : np.ndarray
        (N,) int — start index of each token's atoms in the full atom array.
    atom_counts : np.ndarray
        (N,) int — number of atoms per token.
    rng : np.random.RandomState, optional
        Random state. Uses np.random if None.
    seed_idx : int, optional
        If provided, use this token as the seed instead of a random one.

    Returns
    -------
    CropResult
        Crop indices for tokens and corresponding atoms.
    """
    if rng is None:
        rng = np.random

    N = len(center_coords)

    # If already small enough, keep everything
    if N <= n_crop:
        token_indices = np.arange(N, dtype=np.int64)
        all_atom_indices = []
        starts = []
        counts = []
        offset = 0
        for i in range(N):
            s = atom_starts[i]
            c = atom_counts[i]
            all_atom_indices.append(np.arange(s, s + c, dtype=np.int64))
            starts.append(offset)
            counts.append(c)
            offset += c
        atom_indices = (
            np.concatenate(all_atom_indices)
            if all_atom_indices
            else np.array([], dtype=np.int64)
        )
        return CropResult(
            token_indices=token_indices,
            atom_indices=atom_indices,
            token_to_atom_start=np.array(starts, dtype=np.int64),
            token_to_atom_count=np.array(counts, dtype=np.int64),
        )

    # Pick seed token
    if seed_idx is None:
        seed_idx = rng.randint(N)

    # Compute distances from seed
    diffs = center_coords - center_coords[seed_idx]
    dists = np.linalg.norm(diffs, axis=-1)

    # Take N_crop nearest
    sorted_indices = np.argsort(dists)
    token_indices = np.sort(sorted_indices[:n_crop])  # keep original order

    # Gather corresponding atoms
    all_atom_indices = []
    starts = []
    counts = []
    offset = 0
    for ti in token_indices:
        s = atom_starts[ti]
        c = atom_counts[ti]
        all_atom_indices.append(np.arange(s, s + c, dtype=np.int64))
        starts.append(offset)
        counts.append(c)
        offset += c

    atom_indices = (
        np.concatenate(all_atom_indices)
        if all_atom_indices
        else np.array([], dtype=np.int64)
    )

    return CropResult(
        token_indices=token_indices.astype(np.int64),
        atom_indices=atom_indices,
        token_to_atom_start=np.array(starts, dtype=np.int64),
        token_to_atom_count=np.array(counts, dtype=np.int64),
    )


def spatial_crop_with_resolved_preference(
    center_coords: np.ndarray,
    resolved_mask: np.ndarray,
    n_crop: int,
    atom_starts: np.ndarray,
    atom_counts: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
) -> CropResult:
    """Spatial crop that prefers resolved tokens as seed.

    Picks a random resolved token as seed. Falls back to any token if
    none are resolved. Adapted from Boltz BoltzCropper.

    Parameters
    ----------
    center_coords : np.ndarray
        (N, 3) center coordinates per token.
    resolved_mask : np.ndarray
        (N,) bool — True if the token has resolved coordinates.
    n_crop : int
        Number of tokens to keep.
    atom_starts : np.ndarray
        (N,) int — start index in atom array.
    atom_counts : np.ndarray
        (N,) int — number of atoms per token.
    rng : np.random.RandomState, optional
        Random state.

    Returns
    -------
    CropResult
    """
    if rng is None:
        rng = np.random

    # Pick seed from resolved tokens if possible
    resolved_idx = np.where(resolved_mask)[0]
    if len(resolved_idx) > 0:
        seed_idx = resolved_idx[rng.randint(len(resolved_idx))]
    else:
        seed_idx = rng.randint(len(center_coords))

    return spatial_crop(
        center_coords=center_coords,
        n_crop=n_crop,
        atom_starts=atom_starts,
        atom_counts=atom_counts,
        rng=rng,
        seed_idx=seed_idx,
    )

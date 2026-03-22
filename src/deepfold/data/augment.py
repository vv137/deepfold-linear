"""Coordinate augmentation for SE(3) invariance (Boltz Algorithm 19).

Training: center + uniform SO(3) rotation + small Gaussian translation
Inference: center + uniform SO(3) rotation (no translation)

Two implementations:
  - NumPy: used in featurize() during data loading (CPU, single sample)
  - Torch: used in diffusion multiplicity (GPU, batched M samples)
"""

import numpy as np
import torch


def random_quaternion(rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a unit quaternion uniformly from SO(3).

    Uses the Gaussian-normalisation method: sample q ~ N(0, I_4),
    then normalise to the unit sphere.  This is uniform on S^3,
    which double-covers SO(3).

    Returns (4,) float64 quaternion [w, x, y, z].
    """
    if rng is None:
        rng = np.random.default_rng()
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    # Canonicalise: ensure w >= 0 so the same rotation isn't sampled twice
    if q[0] < 0:
        q = -q
    return q


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix.

    Returns (3, 3) float64 rotation matrix R such that v' = v @ R.T
    (or equivalently v' = R @ v for column vectors).
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def random_rotation_matrix(rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a 3x3 rotation matrix uniformly from SO(3).

    Returns (3, 3) float64.
    """
    return quaternion_to_matrix(random_quaternion(rng))


def center_random_augmentation(
    *coord_arrays: np.ndarray,
    mask: np.ndarray | None = None,
    s_trans: float = 1.0,
    training: bool = True,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Center and randomly augment coordinate arrays (Boltz Algorithm 19).

    All arrays share the SAME rotation and translation so the
    coordinate frame stays consistent.

    Parameters
    ----------
    *coord_arrays : np.ndarray
        Each is (M_i, 3) float.  All are transformed identically.
    mask : np.ndarray, optional
        (M,) bool for the first array — centre of mass computed over
        masked atoms only.  If None, all atoms used.
    s_trans : float
        Standard deviation of Gaussian translation (Angstroms).
        Only applied during training.
    training : bool
        If True, apply rotation + translation.
        If False, apply rotation only (no translation).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    list[np.ndarray]
        Augmented copies of each input array (same order, float32).
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(coord_arrays) == 0:
        return []

    # --- 1. Center (using first array + mask) ---
    ref = coord_arrays[0]
    if mask is not None and mask.any():
        center = ref[mask].mean(axis=0)
    elif len(ref) > 0:
        center = ref.mean(axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)

    centered = [arr - center for arr in coord_arrays]

    # --- 2. Random rotation (uniform SO(3)) ---
    R = random_rotation_matrix(rng)  # (3, 3)
    rotated = [(c.astype(np.float64) @ R.T).astype(np.float32) for c in centered]

    # --- 3. Random translation (training only) ---
    if training:
        t = (rng.standard_normal(3) * s_trans).astype(np.float32)
        rotated = [r + t for r in rotated]

    return rotated


# ===========================================================================
# Torch (GPU) augmentation — for diffusion multiplicity
# ===========================================================================


def random_rotation_matrices_torch(
    n: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Sample n uniform SO(3) rotation matrices via random quaternions.

    Returns (n, 3, 3) rotation matrices.
    """
    q = torch.randn(n, 4, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)  # unit quaternions
    w, x, y, z = q.unbind(-1)

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(n, 3, 3)
    return R


def batch_augment(
    x: torch.Tensor,
    M: int,
    s_trans: float = 1.0,
    training: bool = True,
) -> torch.Tensor:
    """Augment coordinates M times with independent random rotations + translations.

    Each of the M copies gets centered, then a different random SO(3) rotation
    (and, if training, a different small Gaussian translation).

    Args:
        x:        (N_atom, 3) coordinates (already centered from featurize).
        M:        number of augmented copies.
        s_trans:  translation std (Å). Only applied during training.
        training: if False, skip translation.

    Returns:
        (M, N_atom, 3) augmented coordinate copies.
    """
    # x is (N_atom, 3) → (M, N_atom, 3) via independent rotations
    R = random_rotation_matrices_torch(M, x.device, x.dtype)  # (M, 3, 3)
    x_exp = x.unsqueeze(0).expand(M, -1, -1)  # (M, N_atom, 3)
    x_rot = torch.bmm(x_exp, R.transpose(1, 2))  # (M, N_atom, 3)

    if training:
        t = torch.randn(M, 1, 3, device=x.device, dtype=x.dtype) * s_trans
        x_rot = x_rot + t

    return x_rot

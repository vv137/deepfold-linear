"""Dataset and DataLoader for DeepFold-Linear.

Implements:
    - DeepFoldDataset(torch.utils.data.Dataset): loads, tokenizes, crops, featurizes
    - collate_fn: pads variable-length tensors for batching
    - create_dataloader: helper to build a DataLoader

Design decisions (per SPEC and CLAUDE.md):
    - No batch dimension in model forward — single sample per forward pass.
    - Pure PyTorch, no Lightning.
    - Spatial crop with configurable size (schedule managed by trainer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from deepfold.data import const
from deepfold.data.crop import (
    spatial_crop_with_resolved_preference,
)
from deepfold.data.featurize import featurize
from deepfold.data.symmetry import compute_chain_symmetries, compute_amino_acid_symmetries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for DeepFoldDataset."""

    data_dir: str = ""  # directory containing NPZ or mmCIF files
    manifest_path: Optional[str] = None  # JSON manifest listing structures
    max_tokens: int = 256  # crop size (tokens); updated by trainer per step
    max_atoms: Optional[int] = None  # optional atom budget
    max_msa_seqs: int = 128  # MSA depth cap (S)
    training: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# Structure loading helpers
# ---------------------------------------------------------------------------


def load_structure_npz(path: Path) -> dict:
    """Load a Boltz-style NPZ structure file.

    Returns a dict with keys: atoms, residues, chains, tokens, bonds,
    mask, interfaces, connections (whatever is available).
    """
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _ensure_fields(arr: np.ndarray, required: list[str]) -> bool:
    """Check that a structured array has all required fields."""
    if arr.dtype.names is None:
        return False
    return all(f in arr.dtype.names for f in required)


# Boltz mol_type IDs match DeepFold exactly (after aligning to Boltz-1):
# 0=PROTEIN, 1=DNA, 2=RNA, 3=NONPOLYMER

# Token dtype matching what _process() expects
_TOKEN_DTYPE = np.dtype(
    [
        ("token_idx", "i8"),
        ("asym_id", "i8"),
        ("res_idx", "i8"),
        ("res_type", "i8"),
        ("mol_type", "i8"),
        ("atom_idx", "i8"),
        ("atom_num", "i8"),
        ("center_coords", "f4", (3,)),
        ("resolved_mask", "?"),
    ]
)

_TOKEN_BOND_DTYPE = np.dtype([("token_1", "i8"), ("token_2", "i8")])


def _atom_name_hash_vectorized(name_arr: np.ndarray) -> np.ndarray:
    """Vectorized hash of (N, 4) int8 atom names → (N,) int64 in [0, 64)."""
    n = name_arr.astype(np.int64)
    h = n[:, 0]
    for i in range(1, name_arr.shape[1]):
        h = (h * 31 + n[:, i]) & 0xFFFFFFFF
    return h % 64


def tokenize_boltz_structure(struct: dict) -> dict:
    """Convert raw Boltz NPZ (residues+chains+atoms) → DeepFold tokens format.

    Vectorized: builds tokens from residues/chains using bulk numpy ops.
    Non-standard residues are expanded to one token per atom.
    """
    atoms = struct["atoms"]
    residues = struct["residues"]
    chains = struct["chains"]
    mask = struct.get("mask", np.ones(len(chains), dtype=bool))
    raw_bonds = struct.get(
        "bonds",
        np.array([], dtype=[("atom_1", "i4"), ("atom_2", "i4"), ("type", "i1")]),
    )
    connections = struct.get("connections", np.array([]))

    # --- Build per-residue chain info (vectorized) ---
    # Map each residue to its chain's mol_type and asym_id
    n_res = len(residues)
    res_mol_type = np.zeros(n_res, dtype=np.int64)
    res_asym_id = np.zeros(n_res, dtype=np.int64)

    for ci, chain in enumerate(chains):
        if not mask[ci]:
            continue
        rs = int(chain["res_idx"])
        rn = int(chain["res_num"])
        res_mol_type[rs : rs + rn] = int(chain["mol_type"])
        res_asym_id[rs : rs + rn] = int(chain["asym_id"])

    # --- Identify valid residues (belonging to unmasked chains) ---
    valid_res_mask = np.zeros(n_res, dtype=bool)
    for ci, chain in enumerate(chains):
        if mask[ci]:
            rs = int(chain["res_idx"])
            rn = int(chain["res_num"])
            valid_res_mask[rs : rs + rn] = True

    # --- Separate standard vs non-standard residues ---
    is_standard = residues["is_standard"] & valid_res_mask

    # Standard residues: one token per residue
    std_idx = np.where(is_standard)[0]
    n_std = len(std_idx)

    # Non-standard residues: one token per atom
    nonstd_idx = np.where(valid_res_mask & ~is_standard)[0]
    nonstd_atom_counts = residues["atom_num"][nonstd_idx].astype(np.int64)
    n_nonstd_tokens = int(nonstd_atom_counts.sum()) if len(nonstd_idx) > 0 else 0

    n_tokens = n_std + n_nonstd_tokens
    if n_tokens == 0:
        result = dict(struct)
        result["tokens"] = np.array([], dtype=_TOKEN_DTYPE)
        result["token_bonds"] = np.array([], dtype=_TOKEN_BOND_DTYPE)
        result["atom_name_indices"] = (
            _atom_name_hash_vectorized(
                atoms["name"].reshape(-1, atoms["name"].shape[-1])
                if atoms["name"].ndim > 1
                else np.zeros((len(atoms), 4), dtype=np.int8)
            )
            if len(atoms) > 0
            else np.array([], dtype=np.int64)
        )
        return result

    tokens = np.zeros(n_tokens, dtype=_TOKEN_DTYPE)

    # Fill standard tokens (vectorized)
    if n_std > 0:
        std_res = residues[std_idx]
        center_indices = std_res["atom_center"].astype(np.int64)

        tokens["token_idx"][:n_std] = np.arange(n_std)
        tokens["asym_id"][:n_std] = res_asym_id[std_idx]
        tokens["res_idx"][:n_std] = std_res["res_idx"]
        tokens["res_type"][:n_std] = std_res["res_type"]
        tokens["mol_type"][:n_std] = res_mol_type[std_idx]
        tokens["atom_idx"][:n_std] = std_res["atom_idx"]
        tokens["atom_num"][:n_std] = std_res["atom_num"]
        tokens["center_coords"][:n_std] = atoms["coords"][center_indices]
        tokens["resolved_mask"][:n_std] = (
            std_res["is_present"] & atoms["is_present"][center_indices]
        )

    # Fill non-standard tokens (one per atom)
    if n_nonstd_tokens > 0:
        off = n_std
        for ri in nonstd_idx:
            res = residues[ri]
            a_start = int(res["atom_idx"])
            a_count = int(res["atom_num"])
            sl = slice(off, off + a_count)
            tokens["token_idx"][sl] = np.arange(off, off + a_count)
            tokens["asym_id"][sl] = res_asym_id[ri]
            tokens["res_idx"][sl] = int(res["res_idx"])
            tokens["res_type"][sl] = int(res["res_type"])
            tokens["mol_type"][sl] = res_mol_type[ri]
            tokens["atom_idx"][sl] = np.arange(a_start, a_start + a_count)
            tokens["atom_num"][sl] = 1
            tokens["center_coords"][sl] = atoms["coords"][a_start : a_start + a_count]
            tokens["resolved_mask"][sl] = atoms["is_present"][
                a_start : a_start + a_count
            ]
            off += a_count

    # --- Build atom_to_token map (vectorized for standard residues) ---
    atom_to_token = np.full(len(atoms), -1, dtype=np.int64)
    if n_std > 0:
        for ti, ri in enumerate(std_idx):
            a_s = int(residues[ri]["atom_idx"])
            a_c = int(residues[ri]["atom_num"])
            atom_to_token[a_s : a_s + a_c] = ti

    if n_nonstd_tokens > 0:
        off = n_std
        for ri in nonstd_idx:
            a_s = int(residues[ri]["atom_idx"])
            a_c = int(residues[ri]["atom_num"])
            atom_to_token[a_s : a_s + a_c] = np.arange(off, off + a_c)
            off += a_c

    # --- Convert atom-level bonds to token-level (vectorized) ---
    bond_set = set()
    for bond_arr in [raw_bonds]:
        if len(bond_arr) == 0:
            continue
        a1 = bond_arr["atom_1"].astype(np.int64)
        a2 = bond_arr["atom_2"].astype(np.int64)
        valid = (a1 < len(atom_to_token)) & (a2 < len(atom_to_token))
        a1, a2 = a1[valid], a2[valid]
        t1 = atom_to_token[a1]
        t2 = atom_to_token[a2]
        cross = (t1 != t2) & (t1 >= 0) & (t2 >= 0)
        for i in np.where(cross)[0]:
            lo, hi = (
                (int(t1[i]), int(t2[i])) if t1[i] < t2[i] else (int(t2[i]), int(t1[i]))
            )
            bond_set.add((lo, hi))

    if (
        len(connections) > 0
        and connections.dtype.names
        and "atom_1" in connections.dtype.names
    ):
        a1 = connections["atom_1"].astype(np.int64)
        a2 = connections["atom_2"].astype(np.int64)
        valid = (a1 < len(atom_to_token)) & (a2 < len(atom_to_token))
        a1, a2 = a1[valid], a2[valid]
        t1 = atom_to_token[a1]
        t2 = atom_to_token[a2]
        cross = (t1 != t2) & (t1 >= 0) & (t2 >= 0)
        for i in np.where(cross)[0]:
            lo, hi = (
                (int(t1[i]), int(t2[i])) if t1[i] < t2[i] else (int(t2[i]), int(t1[i]))
            )
            bond_set.add((lo, hi))

    if bond_set:
        token_bonds = np.array(list(bond_set), dtype=_TOKEN_BOND_DTYPE)
    else:
        token_bonds = np.array([], dtype=_TOKEN_BOND_DTYPE)

    # --- Atom name hashing (vectorized) ---
    name_data = atoms["name"]
    if name_data.ndim == 1:
        # Scalar name field — use directly
        atom_name_indices = name_data.astype(np.int64) % 64
    else:
        atom_name_indices = _atom_name_hash_vectorized(name_data)

    result = dict(struct)
    result["tokens"] = tokens
    result["token_bonds"] = token_bonds
    result["atom_name_indices"] = atom_name_indices
    return result


# ---------------------------------------------------------------------------
# DeepFoldDataset
# ---------------------------------------------------------------------------


def load_msa_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load Boltz-style MSA NPZ and return (residue_types, deletions, taxonomy).

    Returns
    -------
    (msa_seqs, msa_dels, taxonomy) or None if loading fails.
        msa_seqs: (S, L) int array of residue type IDs
        msa_dels: (S, L) float array of deletion counts
        taxonomy: (S,) int array of species IDs (-1 = unknown)
    """
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None

    sequences = data["sequences"]
    residues = data["residues"]
    deletions = data.get(
        "deletions", np.array([], dtype=[("res_idx", "i2"), ("deletion", "i2")])
    )

    S = len(sequences)
    if S == 0:
        return None

    # Determine sequence length from first sequence
    s0 = sequences[0]
    L = int(s0["res_end"] - s0["res_start"])
    if L == 0:
        return None

    msa_seqs = np.zeros((S, L), dtype=np.int64)
    msa_dels = np.zeros((S, L), dtype=np.float32)

    for i, seq in enumerate(sequences):
        rs, re = int(seq["res_start"]), int(seq["res_end"])
        n = re - rs
        if n != L:
            # Variable-length alignment — truncate/pad to query length
            n = min(n, L)
        msa_seqs[i, :n] = residues["res_type"][rs : rs + n]

        # Sparse deletions → dense
        ds, de = int(seq["del_start"]), int(seq["del_end"])
        for d in deletions[ds:de]:
            ridx = int(d["res_idx"])
            if 0 <= ridx < L:
                msa_dels[i, ridx] = float(d["deletion"])

    # Taxonomy: (S,) int array of species IDs, -1 = unknown
    if "taxonomy" in data:
        taxonomy = data["taxonomy"].astype(np.int64)
        if len(taxonomy) != S:
            taxonomy = np.full(S, -1, dtype=np.int64)
    elif (
        sequences.dtype.names is not None
        and "taxonomy" in sequences.dtype.names
    ):
        taxonomy = sequences["taxonomy"].astype(np.int64)
    else:
        taxonomy = np.full(S, -1, dtype=np.int64)

    return msa_seqs, msa_dels, taxonomy


def subsample_msa(
    msa_seqs: np.ndarray,
    msa_dels: np.ndarray,
    max_seqs: int,
    rng: np.random.RandomState,
    training: bool = True,
    min_seqs: int = 1,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample MSA to max depth, always keeping query (row 0).

    During training, randomly picks depth in [min_seqs, max_seqs] (Boltz convention).
    During inference, uses full max_seqs.

    If *return_indices* is True, also returns the row indices used.
    """
    S, L = msa_seqs.shape
    if S <= 1:
        if return_indices:
            return msa_seqs, msa_dels, np.arange(S)
        return msa_seqs, msa_dels

    if training:
        target = rng.randint(min_seqs, max_seqs + 1)
    else:
        target = max_seqs

    if S <= target:
        if return_indices:
            return msa_seqs, msa_dels, np.arange(S)
        return msa_seqs, msa_dels

    # Always keep query (row 0), subsample the rest
    other_idx = rng.choice(S - 1, size=min(target - 1, S - 1), replace=False) + 1
    idx = np.concatenate([[0], np.sort(other_idx)])
    if return_indices:
        return msa_seqs[idx], msa_dels[idx], idx
    return msa_seqs[idx], msa_dels[idx]


def pair_chain_msas(
    chain_msas: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_paired: int = 512,
    max_unpaired: int = 512,
    rng: np.random.RandomState | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Pair MSAs across chains by taxonomy, then add unpaired rows.

    Each element of *chain_msas* is ``(msa_seqs, msa_dels, taxonomy)`` for one
    chain, where:
      - ``msa_seqs``: ``(S, L)`` int — residue type IDs (row 0 = query)
      - ``msa_dels``: ``(S, L)`` float — deletion counts
      - ``taxonomy``: ``(S,)`` int — species IDs (-1 = unknown)

    Returns
    -------
    paired_seqs : list of (S_total, L_chain) int arrays per chain
    paired_dels : list of (S_total, L_chain) float arrays per chain
    is_paired   : (S_total,) bool — True for taxonomy-paired rows
    """
    GAP_TOKEN = 1  # gap residue type index

    n_chains = len(chain_msas)
    chain_lengths = [ms[0].shape[1] for ms in chain_msas]

    if rng is None:
        rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    # Single-chain fast path: no pairing possible
    # ------------------------------------------------------------------
    if n_chains <= 1:
        seqs, dels, _tax = chain_msas[0]
        is_paired = np.zeros(seqs.shape[0], dtype=bool)
        is_paired[0] = True  # query row is always "paired" (it's the query)
        return [seqs], [dels], is_paired

    # ------------------------------------------------------------------
    # Row 0: query sequences from every chain (always present)
    # ------------------------------------------------------------------
    row_seqs = [ms[0][[0]] for ms in chain_msas]   # list of (1, L_c)
    row_dels = [ms[1][[0]] for ms in chain_msas]

    paired_rows_seqs = [r.copy() for r in row_seqs]
    paired_rows_dels = [r.copy() for r in row_dels]
    is_paired_list = [True]  # query row counts as paired

    # ------------------------------------------------------------------
    # Build taxonomy → {chain_idx: [seq_indices]} map (skip row 0, skip -1)
    # ------------------------------------------------------------------
    taxonomy_map: dict[int, dict[int, list[int]]] = {}
    for ci, (_, _, tax) in enumerate(chain_msas):
        for si in range(1, len(tax)):
            t = int(tax[si])
            if t == -1:
                continue
            taxonomy_map.setdefault(t, {}).setdefault(ci, []).append(si)

    # Keep only taxonomies shared by >= 2 chains, sort by chain coverage desc
    shared_taxa = [
        (t, chain_dict)
        for t, chain_dict in taxonomy_map.items()
        if len(chain_dict) >= 2
    ]
    shared_taxa.sort(key=lambda x: len(x[1]), reverse=True)

    # Track which (chain, seq_idx) are used in pairing
    used: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Paired rows: one row per shared-taxonomy occurrence
    # ------------------------------------------------------------------
    for _taxon, chain_dict in shared_taxa:
        max_occ = max(len(v) for v in chain_dict.values())
        for occ_i in range(max_occ):
            row_s = []
            row_d = []
            any_paired = False
            for ci in range(n_chains):
                if ci in chain_dict:
                    idx = chain_dict[ci][occ_i % len(chain_dict[ci])]
                    row_s.append(chain_msas[ci][0][[idx]])
                    row_d.append(chain_msas[ci][1][[idx]])
                    used.add((ci, idx))
                    any_paired = True
                else:
                    # Gap row for this chain
                    L = chain_lengths[ci]
                    row_s.append(np.full((1, L), GAP_TOKEN, dtype=np.int64))
                    row_d.append(np.zeros((1, L), dtype=np.float32))

            if any_paired:
                for ci in range(n_chains):
                    paired_rows_seqs[ci] = np.concatenate(
                        [paired_rows_seqs[ci], row_s[ci]], axis=0
                    )
                    paired_rows_dels[ci] = np.concatenate(
                        [paired_rows_dels[ci], row_d[ci]], axis=0
                    )
                is_paired_list.append(True)

            if len(is_paired_list) >= max_paired + 1:  # +1 for query row
                break
        if len(is_paired_list) >= max_paired + 1:
            break

    # ------------------------------------------------------------------
    # Subsample paired rows to max_paired (keep query row 0)
    # ------------------------------------------------------------------
    n_paired = len(is_paired_list) - 1  # exclude query
    if n_paired > max_paired:
        keep = rng.choice(n_paired, size=max_paired, replace=False) + 1
        keep = np.concatenate([[0], np.sort(keep)])
        for ci in range(n_chains):
            paired_rows_seqs[ci] = paired_rows_seqs[ci][keep]
            paired_rows_dels[ci] = paired_rows_dels[ci][keep]
        is_paired_list = [is_paired_list[i] for i in keep]

    # ------------------------------------------------------------------
    # Unpaired rows: remaining sequences, one chain at a time
    # ------------------------------------------------------------------
    unpaired_seqs = [[] for _ in range(n_chains)]
    unpaired_dels = [[] for _ in range(n_chains)]
    unpaired_count = 0

    for ci in range(n_chains):
        S_c = chain_msas[ci][0].shape[0]
        for si in range(1, S_c):
            if (ci, si) in used:
                continue
            if unpaired_count >= max_unpaired:
                break
            # Build a row: this chain has sequence, others get gap
            for cj in range(n_chains):
                if cj == ci:
                    unpaired_seqs[cj].append(chain_msas[ci][0][[si]])
                    unpaired_dels[cj].append(chain_msas[ci][1][[si]])
                else:
                    L = chain_lengths[cj]
                    unpaired_seqs[cj].append(
                        np.full((1, L), GAP_TOKEN, dtype=np.int64)
                    )
                    unpaired_dels[cj].append(
                        np.zeros((1, L), dtype=np.float32)
                    )
            unpaired_count += 1
        if unpaired_count >= max_unpaired:
            break

    # ------------------------------------------------------------------
    # Concatenate paired + unpaired
    # ------------------------------------------------------------------
    result_seqs = []
    result_dels = []
    for ci in range(n_chains):
        parts_s = [paired_rows_seqs[ci]]
        parts_d = [paired_rows_dels[ci]]
        if unpaired_seqs[ci]:
            parts_s.append(np.concatenate(unpaired_seqs[ci], axis=0))
            parts_d.append(np.concatenate(unpaired_dels[ci], axis=0))
        result_seqs.append(np.concatenate(parts_s, axis=0))
        result_dels.append(np.concatenate(parts_d, axis=0))

    is_paired_arr = np.array(
        is_paired_list + [False] * unpaired_count, dtype=bool
    )

    return result_seqs, result_dels, is_paired_arr


class DeepFoldDataset(Dataset):
    """Dataset that loads structures, crops, and featurizes.

    Each ``__getitem__`` returns a dict of tensors matching the model's
    ``forward()`` signature (no batch dimension).

    The dataset supports two data formats:
        1. **NPZ files** (Boltz-style): directory of ``<id>.npz`` files containing
           structured arrays (atoms, tokens, bonds, chains, residues, etc.).
        2. **Pre-parsed dicts** (in-memory): for testing / programmatic use.
    """

    def __init__(
        self,
        data_paths: list[Path],
        max_tokens: int = 256,
        max_msa_seqs: int = 128,
        min_msa_seqs: int = 1,
        msa_dir: Optional[Union[str, Path]] = None,
        max_msa_cycles: int = 3,
        max_paired: int = 512,
        max_unpaired: int = 512,
        training: bool = True,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        data_paths : list[Path]
            Paths to NPZ structure files.
        max_tokens : int
            Crop size in tokens. Updated externally via ``set_crop_size()``.
        max_msa_seqs : int
            Maximum MSA depth.
        min_msa_seqs : int
            Minimum random MSA depth during training.
        msa_dir : str or Path, optional
            Directory with MSA NPZ files (named ``{pdb}_{chain}.npz``).
        max_paired : int
            Maximum taxonomy-paired MSA rows across chains.
        max_unpaired : int
            Maximum unpaired MSA rows (one chain at a time, others get gap).
        training : bool
            If True, include ground-truth coordinates and use random cropping.
        seed : int
            Random seed for reproducibility.
        """
        super().__init__()
        self.data_paths = list(data_paths)
        self.max_tokens = max_tokens
        self.max_msa_seqs = max_msa_seqs
        self.min_msa_seqs = min_msa_seqs
        self.msa_dir = Path(msa_dir) if msa_dir else None
        self.max_msa_cycles = max_msa_cycles
        self.max_paired = max_paired
        self.max_unpaired = max_unpaired
        self.training = training
        self.rng = np.random.RandomState(seed)

    def set_crop_size(self, n_crop: int) -> None:
        """Update the crop size (called by trainer when schedule changes)."""
        self.max_tokens = n_crop

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Load, crop, and featurize a single structure."""
        path = self.data_paths[idx]

        try:
            return self._process(path)
        except Exception as e:
            logger.warning("Failed to process %s: %s. Trying another sample.", path, e)
            # Fall back to a random other sample
            alt_idx = self.rng.randint(len(self.data_paths))
            try:
                return self._process(self.data_paths[alt_idx])
            except Exception as e2:
                logger.error(
                    "Fallback also failed on %s: %s", self.data_paths[alt_idx], e2
                )
                raise

    def _process(self, path: Path) -> dict[str, Tensor]:
        """Core processing pipeline for a single structure."""
        # 1. Load
        struct = load_structure_npz(path)
        struct["_source_path"] = str(path)

        # Tokenize if raw Boltz format (has 'residues' but no 'tokens')
        if "tokens" not in struct and "residues" in struct:
            struct = tokenize_boltz_structure(struct)

        tokens = struct["tokens"]
        if len(tokens) == 0:
            raise ValueError(f"No valid tokens (all chains masked): {path.name}")
        atoms = struct["atoms"]
        # Use token-level bonds (from tokenization or pre-existing)
        bonds = struct.get(
            "token_bonds",
            struct.get(
                "bonds", np.array([], dtype=[("token_1", "i8"), ("token_2", "i8")])
            ),
        )

        # Ensure bonds has correct dtype
        if bonds.dtype.names is None or "token_1" not in bonds.dtype.names:
            bonds = np.array([], dtype=[("token_1", "i8"), ("token_2", "i8")])

        len(tokens)

        # 2. Extract token center coords and atom info for cropping (vectorized)
        center_coords = tokens["center_coords"].astype(np.float32)
        resolved_mask = tokens["resolved_mask"].astype(bool)
        atom_starts = tokens["atom_idx"].astype(np.int64)
        atom_counts = tokens["atom_num"].astype(np.int64)

        # 3. Crop
        crop = spatial_crop_with_resolved_preference(
            center_coords=center_coords,
            resolved_mask=resolved_mask,
            n_crop=self.max_tokens,
            atom_starts=atom_starts,
            atom_counts=atom_counts,
            rng=self.rng,
        )

        # 4. Build cropped arrays
        cropped_tokens = tokens[crop.token_indices]
        N = len(cropped_tokens)

        # Rebuild contiguous atom table from cropped tokens (vectorized)
        token_types = cropped_tokens["mol_type"].astype(np.int64)
        token_res_types = cropped_tokens["res_type"].astype(np.int64)
        token_chain_ids = cropped_tokens["asym_id"].astype(np.int64)
        token_global_idx = cropped_tokens["res_idx"].astype(np.int64)
        token_resolved = cropped_tokens["resolved_mask"].astype(bool)
        token_center_coords = cropped_tokens["center_coords"].astype(np.float32)

        # Build atom arrays from the cropped atom indices
        cropped_atoms = (
            atoms[crop.atom_indices] if len(crop.atom_indices) > 0 else atoms[:0]
        )
        N_atom = len(cropped_atoms)

        atom_coords_all = (
            cropped_atoms["coords"].astype(np.float32)
            if N_atom > 0
            else np.zeros((0, 3), dtype=np.float32)
        )
        atom_ref_pos_all = (
            cropped_atoms["conformer"].astype(np.float32)
            if N_atom > 0 and "conformer" in cropped_atoms.dtype.names
            else atom_coords_all.copy()
        )
        atom_charge_all = (
            cropped_atoms["charge"].astype(np.float32)
            if N_atom > 0 and "charge" in cropped_atoms.dtype.names
            else np.zeros(N_atom, dtype=np.float32)
        )
        atom_element_all = (
            cropped_atoms["element"].astype(np.int64)
            if N_atom > 0 and "element" in cropped_atoms.dtype.names
            else np.zeros(N_atom, dtype=np.int64)
        )
        # Atom name: use pre-computed indices if available (Boltz format has
        # 4-byte name arrays, not single ints).  atom_name_indices is built
        # by tokenize_boltz_structure().
        if "atom_name_indices" in struct and N_atom > 0:
            atom_name_all = struct["atom_name_indices"][crop.atom_indices].astype(
                np.int64
            )
        elif N_atom > 0 and "name" in cropped_atoms.dtype.names:
            name_raw = cropped_atoms["name"]
            if name_raw.ndim > 1:
                raise ValueError(
                    "Multi-byte atom names require pre-computed atom_name_indices "
                    "(run tokenize_boltz_structure first)"
                )
            atom_name_all = name_raw.astype(np.int64)
        else:
            atom_name_all = np.zeros(N_atom, dtype=np.int64)
        atom_mask_all = (
            cropped_atoms["is_present"].astype(bool)
            if N_atom > 0 and "is_present" in cropped_atoms.dtype.names
            else np.ones(N_atom, dtype=bool)
        )

        # Atom → token mapping (local indices) — vectorized
        atom_to_token_all = np.zeros(N_atom, dtype=np.int64)
        starts = crop.token_to_atom_start
        counts = crop.token_to_atom_count
        for tok_id in range(N):
            atom_to_token_all[starts[tok_id] : starts[tok_id] + counts[tok_id]] = tok_id

        # Re-map token bonds to cropped token space (vectorized lookup)
        orig_ids = cropped_tokens["token_idx"].astype(np.int64)
        orig_to_local = np.full(
            int(orig_ids.max()) + 1 if N > 0 else 0, -1, dtype=np.int64
        )
        orig_to_local[orig_ids] = np.arange(N)
        bond_pairs = []
        n_dropped = 0
        if len(bonds) > 0:
            b_t1 = bonds["token_1"].astype(np.int64)
            b_t2 = bonds["token_2"].astype(np.int64)
            max_orig = int(orig_ids.max()) + 1 if N > 0 else 0
            valid = (b_t1 < max_orig) & (b_t2 < max_orig)
            b_t1, b_t2 = b_t1[valid], b_t2[valid]
            l1 = orig_to_local[b_t1]
            l2 = orig_to_local[b_t2]
            keep = (l1 >= 0) & (l2 >= 0)
            bond_pairs = list(zip(l1[keep].tolist(), l2[keep].tolist()))
            n_dropped = int((~keep).sum()) + int((~valid).sum())
        if n_dropped > 0:
            logger.debug(
                "Dropped %d/%d bonds outside crop window", n_dropped, len(bonds)
            )

        # MSA mask (vectorized)
        msa_mask = (token_types == const.MOL_PROTEIN) | (token_types == const.MOL_RNA)

        # Load real MSA if available
        msa_data_np = None
        msa_del_np = None
        full_msa_np = None
        full_del_np = None

        if self.msa_dir is not None:
            msa_data_np, msa_del_np, full_msa_np, full_del_np = (
                self._load_chain_msa(struct, cropped_tokens, msa_mask)
            )

        # 5. Featurize
        features = featurize(
            token_types=token_types,
            token_res_types=token_res_types,
            token_chain_ids=token_chain_ids,
            token_global_idx=token_global_idx,
            token_resolved_mask=token_resolved,
            atom_coords=atom_coords_all,
            atom_ref_pos=atom_ref_pos_all,
            atom_charge=atom_charge_all,
            atom_element=atom_element_all,
            atom_name=atom_name_all,
            atom_mask=atom_mask_all,
            atom_to_token=atom_to_token_all,
            token_atom_starts=crop.token_to_atom_start,
            token_atom_counts=crop.token_to_atom_count,
            token_center_coords=token_center_coords,
            token_bonds=bond_pairs,
            msa_data=msa_data_np,
            msa_deletion=msa_del_np,
            full_msa_data=full_msa_np,
            full_msa_deletion=full_del_np,
            msa_mask=msa_mask,
            training=self.training,
        )

        # 6. Symmetry features (training only)
        if self.training and "chains" in struct:
            sym = compute_chain_symmetries(
                struct, cropped_tokens, crop.atom_indices,
            )
            aa_sym = compute_amino_acid_symmetries(
                cropped_tokens, crop.token_to_atom_start, crop.token_to_atom_count,
            )
            features["sym_all_coords"] = sym["all_coords"]
            features["sym_all_resolved_mask"] = sym["all_resolved_mask"]
            features["sym_crop_to_all_atom_map"] = sym["crop_to_all_atom_map"]
            # Non-tensor: store as Python lists (handled by collate_fn)
            features["sym_chain_symmetries"] = sym["chain_symmetries"]
            features["sym_amino_acid_symmetries"] = aa_sym

        return features

    def _load_chain_msa(
        self,
        struct: dict,
        cropped_tokens: np.ndarray,
        msa_mask: np.ndarray,
    ) -> tuple:
        """Load MSA for ALL protein/RNA chains, pair by taxonomy, subsample.

        Returns (msa_crops, del_crops, full_msa_crop, full_del_crop):
          - msa_crops: list of (S_i, N_prot) subsampled MSA per cycle
          - del_crops: list of (S_i, N_prot) subsampled deletions per cycle
          - full_msa_crop: (S_full, N_prot) full MSA (for profile/del_mean)
          - full_del_crop: (S_full, N_prot) full deletions
        Returns (None, None, None, None) if no MSA is available.
        """
        N_prot = int(msa_mask.sum())
        if N_prot == 0 or self.msa_dir is None:
            return None, None, None, None

        chains = struct.get("chains", None)
        if chains is None:
            return None, None, None, None

        pdb_id = None
        src = struct.get("_source_path", None)
        if src is not None:
            pdb_id = Path(src).stem
        if pdb_id is None:
            return None, None, None, None

        # Find unique protein/RNA chain asym_ids in the crop
        prot_chain_ids = np.unique(cropped_tokens["asym_id"][msa_mask])

        # ------------------------------------------------------------------
        # Load per-chain MSAs and build column-alignment info
        # ------------------------------------------------------------------
        # Each entry: (msa_seqs, msa_dels, taxonomy, prot_local_indices)
        # prot_local_indices: positions in the N_prot dimension for this chain
        chain_infos: list[dict] = []
        prot_positions = np.where(msa_mask)[0]

        for chain in chains:
            chain_asym = int(chain["asym_id"])
            if chain_asym not in prot_chain_ids:
                continue
            mol = int(chain["mol_type"])
            if mol != const.MOL_PROTEIN and mol != const.MOL_RNA:
                continue

            chain_letter = str(chain["name"]).strip().lower()
            chain_letter = chain_letter.rstrip("0123456789")
            msa_path = self.msa_dir / f"{pdb_id}_{chain_letter}.npz"

            if not msa_path.exists():
                continue

            result = load_msa_npz(msa_path)
            if result is None:
                continue

            msa_seqs, msa_dels, taxonomy = result
            L_msa = msa_seqs.shape[1]

            chain_res_start = int(chain["res_idx"])
            this_chain_mask = msa_mask & (cropped_tokens["asym_id"] == chain_asym)
            if not this_chain_mask.any():
                continue

            prot_global_idx = cropped_tokens["res_idx"][this_chain_mask].astype(np.int64)
            local_idx = np.clip(prot_global_idx - chain_res_start, 0, L_msa - 1)

            chain_positions = np.where(this_chain_mask)[0]
            prot_local = np.searchsorted(prot_positions, chain_positions)

            chain_infos.append({
                "msa_seqs": msa_seqs,
                "msa_dels": msa_dels,
                "taxonomy": taxonomy,
                "local_idx": local_idx,     # columns in MSA NPZ
                "prot_local": prot_local,   # columns in N_prot space
            })

        if not chain_infos:
            return None, None, None, None

        # ------------------------------------------------------------------
        # Pair MSAs across chains
        # ------------------------------------------------------------------
        chain_msas_for_pairing = [
            (ci["msa_seqs"], ci["msa_dels"], ci["taxonomy"])
            for ci in chain_infos
        ]
        paired_seqs, paired_dels, is_paired = pair_chain_msas(
            chain_msas_for_pairing,
            max_paired=self.max_paired,
            max_unpaired=self.max_unpaired,
            rng=self.rng,
        )
        # paired_seqs[k]: (S_total, L_chain_k) — full chain length
        # We need to map to (S_total, N_prot) using each chain's alignment info

        S_total = paired_seqs[0].shape[0]

        # ------------------------------------------------------------------
        # Build full (pre-subsampled) MSA in N_prot space
        # ------------------------------------------------------------------
        full_msa_crop = np.full((S_total, N_prot), 1, dtype=np.int64)  # gap=1
        full_del_crop = np.zeros((S_total, N_prot), dtype=np.float32)

        for k, ci in enumerate(chain_infos):
            local_idx = ci["local_idx"]
            prot_local = ci["prot_local"]
            # paired_seqs[k] has full chain-length columns; extract aligned ones
            full_msa_crop[:, prot_local] = paired_seqs[k][:, local_idx]
            full_del_crop[:, prot_local] = paired_dels[k][:, local_idx]

        # ------------------------------------------------------------------
        # Produce max_msa_cycles independent subsamplings
        # ------------------------------------------------------------------
        msa_crops, del_crops = [], []
        for _ in range(self.max_msa_cycles):
            sub_seqs, sub_dels = subsample_msa(
                full_msa_crop, full_del_crop, self.max_msa_seqs, self.rng,
                self.training, min_seqs=self.min_msa_seqs,
            )
            msa_crops.append(sub_seqs)
            del_crops.append(sub_dels)

        return msa_crops, del_crops, full_msa_crop, full_del_crop


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

# Keys whose padding value is 0.0 (marks padding positions in masks).
# These are float mask tensors where 1.0 = real, 0.0 = padding.
_MASK_KEYS = {"token_pad_mask", "atom_pad_mask", "pair_valid_mask", "msa_mask"}

# Keys that hold plain Python lists (not tensors) — passed through as lists.
_LIST_KEYS = {"sym_chain_symmetries", "sym_amino_acid_symmetries"}


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a batch of samples, padding variable-length tensors.

    For batch_size=1 returns the single sample dict (no batch dim) for
    backward compatibility.

    For batch_size>1, pads each tensor to the max size along every
    dimension and stacks along a new dim 0 to produce (B, ...) tensors.
    Padding conventions:
        - Float/int tensors: zero-padded
        - Bool tensors (e.g. bond_matrix): False-padded
        - Mask tensors (token_pad_mask, atom_pad_mask, pair_valid_mask):
          padded with 0.0 (marks padding positions)
        - List keys (symmetry specs): collected as list-of-lists
    """
    if len(batch) == 1:
        result = {}
        for k, v in batch[0].items():
            if k in _LIST_KEYS:
                result[k] = [v]
            else:
                result[k] = v.unsqueeze(0)
        return result

    # Collect all keys (use first sample; all samples have the same keys)
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [b[key] for b in batch]

        # Non-tensor keys: collect as plain list
        if key in _LIST_KEYS:
            collated[key] = values
            continue

        # Determine pad value based on key/dtype
        pad_value: float | bool
        if values[0].dtype == torch.bool:
            pad_value = False
        else:
            # For mask keys, 0.0 is the correct pad (marks padding).
            # For all other float/int keys, 0 is also the correct pad.
            pad_value = 0

        collated[key] = _pad_and_stack(values, pad_value=pad_value)

    # msa_mask (B, C, S, N_msa) is already produced by featurize and
    # padded by _pad_and_stack — no additional generation needed here.

    return collated


def _pad_and_stack(
    tensors: list[Tensor],
    pad_value: float | bool = 0,
) -> Tensor:
    """Pad tensors to the same shape and stack along a new batch dim.

    Parameters
    ----------
    tensors : list[Tensor]
        Tensors of the same ndim but potentially different sizes.
    pad_value : float | bool
        Value to use for padding (0 for int/float, False for bool).
    """
    ndim = tensors[0].ndim
    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        for d in range(ndim):
            max_shape[d] = max(max_shape[d], t.shape[d])

    # Fast path: all shapes already match
    if all(list(t.shape) == max_shape for t in tensors):
        return torch.stack(tensors, dim=0)

    padded = []
    for t in tensors:
        pad_widths: list[int] = []
        # F.pad expects padding in reverse dimension order:
        # (last_dim_left, last_dim_right, second_last_left, ...)
        for d in reversed(range(ndim)):
            pad_widths.extend([0, max_shape[d] - t.shape[d]])
        padded.append(torch.nn.functional.pad(t, pad_widths, value=pad_value))

    return torch.stack(padded, dim=0)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_dataloader(
    data_dir: Union[str, Path],
    max_tokens: int = 256,
    max_msa_seqs: int = 128,
    training: bool = True,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    file_pattern: str = "*.npz",
) -> DataLoader:
    """Create a DataLoader for DeepFold-Linear.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing NPZ structure files.
    max_tokens : int
        Initial crop size. Updated during training via dataset.set_crop_size().
    max_msa_seqs : int
        Maximum MSA depth.
    training : bool
        Training mode (includes ground-truth, random crop).
    batch_size : int
        Batch size. Default 1 (model expects no batch dim).
    num_workers : int
        DataLoader workers.
    pin_memory : bool
        Pin memory for CUDA.
    seed : int
        Random seed.
    file_pattern : str
        Glob pattern for structure files.

    Returns
    -------
    DataLoader
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob(file_pattern))

    if not paths:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {data_dir}")

    dataset = DeepFoldDataset(
        data_paths=paths,
        max_tokens=max_tokens,
        max_msa_seqs=max_msa_seqs,
        training=training,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=training,
    )

    return loader

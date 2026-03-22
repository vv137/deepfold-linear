"""MSA processing for DeepFold-Linear.

Adapted from Boltz-1 ``parse/a3m.py`` and ``feature/featurizer.py``.

Provides:
  - A3M file parsing (with gzip support)
  - MSA feature computation: profile (32-dim), deletion_mean, has_msa
  - MSA matrix construction: (S, N_prot, 34) raw features
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
import torch
import torch.nn.functional as F

from deepfold.data.parse import (
    NUM_TOKENS,
    PROT_LETTER_TO_TOKEN,
    TOKEN_IDS,
)

# ---------------------------------------------------------------------------
# MSA structured dtypes (compatible with Boltz NPZ)
# ---------------------------------------------------------------------------

MSAResidueDtype = np.dtype([("res_type", "i1")])
MSADeletionDtype = np.dtype([("res_idx", "i2"), ("deletion", "i2")])
MSASequenceDtype = np.dtype(
    [
        ("seq_idx", "i2"),
        ("taxonomy", "i4"),
        ("res_start", "i4"),
        ("res_end", "i4"),
        ("del_start", "i4"),
        ("del_end", "i4"),
    ]
)


@dataclass(frozen=True)
class MSA:
    """Parsed MSA data for a single chain."""

    sequences: np.ndarray  # MSASequenceDtype
    deletions: np.ndarray  # MSADeletionDtype
    residues: np.ndarray  # MSAResidueDtype

    def save(self, path: Path) -> None:
        np.savez_compressed(
            str(path),
            sequences=self.sequences,
            deletions=self.deletions,
            residues=self.residues,
        )

    @classmethod
    def load(cls, path: Path) -> "MSA":
        data = np.load(path, allow_pickle=True)
        return cls(
            sequences=data["sequences"],
            deletions=data["deletions"],
            residues=data["residues"],
        )


# ---------------------------------------------------------------------------
# A3M parsing
# ---------------------------------------------------------------------------


def _parse_a3m_lines(
    lines: TextIO,
    taxonomy: Optional[dict[str, str]] = None,
    max_seqs: Optional[int] = None,
) -> MSA:
    """Parse A3M formatted lines into an MSA object.

    Follows Boltz-1 logic: lowercase characters are insertions (counted as
    deletions), uppercase + gaps are alignment columns.  Duplicate sequences
    (ignoring gaps) are skipped.
    """
    visited: set[str] = set()
    sequences: list[tuple] = []
    deletions: list[tuple] = []
    residues: list[int] = []

    seq_idx = 0
    taxonomy_id = -1

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith(">"):
            header = line.split()[0]
            if taxonomy and header.startswith(">UniRef100"):
                uniref_id = header.split("_")[1]
                taxonomy_id = int(taxonomy.get(uniref_id, -1))
            else:
                taxonomy_id = -1
            continue

        # Skip duplicate sequences
        str_seq = line.replace("-", "").upper()
        if str_seq in visited:
            continue
        visited.add(str_seq)

        # Process alignment columns
        residue: list[int] = []
        deletion: list[tuple[int, int]] = []
        count = 0
        res_idx = 0

        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            # Map character to token id
            token_name = PROT_LETTER_TO_TOKEN.get(c, c)
            token_id = TOKEN_IDS.get(token_name, TOKEN_IDS.get("-", 1))
            residue.append(token_id)
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)
        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if max_seqs is not None and seq_idx >= max_seqs:
            break

    return MSA(
        residues=np.array(residues, dtype=MSAResidueDtype),
        deletions=np.array(deletions, dtype=MSADeletionDtype),
        sequences=np.array(sequences, dtype=MSASequenceDtype),
    )


def parse_a3m(
    path: str | Path,
    taxonomy: Optional[dict[str, str]] = None,
    max_seqs: Optional[int] = None,
) -> MSA:
    """Parse an A3M (or A3M.gz) file.

    Parameters
    ----------
    path : str | Path
        Path to the ``.a3m`` or ``.a3m.gz`` file.
    taxonomy : dict, optional
        Mapping from UniRef100 ids to taxonomy ids.
    max_seqs : int, optional
        Maximum number of sequences to keep.

    Returns
    -------
    MSA
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            return _parse_a3m_lines(f, taxonomy, max_seqs)
    else:
        with path.open("r") as f:
            return _parse_a3m_lines(f, taxonomy, max_seqs)


def dummy_msa(n_residues: int, res_type: int = 1) -> MSA:
    """Create a dummy single-sequence MSA (gap-only).

    Useful for chains without MSA data (ligands, ions, etc.).

    Parameters
    ----------
    n_residues : int
        Number of residues in the chain.
    res_type : int
        Token id to fill (default: 1 = gap ``-``).
    """
    residues = np.full(n_residues, res_type, dtype=MSAResidueDtype)
    sequences = np.array([(0, -1, 0, n_residues, 0, 0)], dtype=MSASequenceDtype)
    deletions = np.array([], dtype=MSADeletionDtype)
    return MSA(sequences=sequences, deletions=deletions, residues=residues)


# ---------------------------------------------------------------------------
# MSA -> dense matrix
# ---------------------------------------------------------------------------


def msa_to_dense(
    msa: MSA,
    query_length: int,
    max_seqs: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a compact MSA into dense (S, L) matrices.

    Returns
    -------
    msa_matrix : np.ndarray, shape (S, L), int
        Token ids per alignment position.
    deletion_matrix : np.ndarray, shape (S, L), float32
        Deletion counts per alignment position.
    """
    seqs = msa.sequences
    if max_seqs is not None and len(seqs) > max_seqs:
        seqs = seqs[:max_seqs]

    S = len(seqs)
    L = query_length
    msa_mat = np.full((S, L), TOKEN_IDS["-"], dtype=np.int64)
    del_mat = np.zeros((S, L), dtype=np.float32)

    for seq in seqs:
        s = int(seq["seq_idx"])
        if s >= S:
            break
        rs = int(seq["res_start"])
        re = int(seq["res_end"])
        residues = msa.residues[rs:re]
        length = min(re - rs, L)
        msa_mat[s, :length] = residues["res_type"][:length]

        ds = int(seq["del_start"])
        de = int(seq["del_end"])
        for d in msa.deletions[ds:de]:
            ridx = int(d["res_idx"])
            if ridx < L:
                del_mat[s, ridx] = float(d["deletion"])

    return msa_mat, del_mat


# ---------------------------------------------------------------------------
# MSA feature computation (SPEC sections 3.1 and 3.2)
# ---------------------------------------------------------------------------


def compute_msa_features(
    msa_dict: dict[int, MSA],
    token_types: np.ndarray,
    chain_id: np.ndarray,
    global_idx: np.ndarray,
    max_seqs: int = 128,
) -> dict[str, torch.Tensor]:
    """Compute MSA features for the full token set.

    This handles multi-chain pairing by simple concatenation along the
    residue dimension (protein tokens only, ordered by token index).

    Parameters
    ----------
    msa_dict : dict[int, MSA]
        MSA per chain (keyed by asym_id / chain index).
    token_types : np.ndarray, shape (N,)
        Per-token type (0=protein, 1=RNA, 2=DNA, 3=ligand).
    chain_id : np.ndarray, shape (N,)
        Per-token chain (asym_id).
    global_idx : np.ndarray, shape (N,)
        Per-token global residue index.
    max_seqs : int
        Maximum MSA depth S.

    Returns
    -------
    dict with keys:
        profile       : (N, 32) float  -- MSA frequency profile (first 32 token classes)
        deletion_mean : (N, 1)  float  -- mean deletion value
        has_msa       : (N, 1)  float  -- 1.0 for protein/RNA tokens
        msa_feat      : (S, N_prot, 34) float -- raw MSA features for MSA module
        msa_mask      : (S, N_prot) float     -- 1 where MSA row exists
        n_prot        : int                    -- number of protein tokens
    """
    N = len(token_types)

    # Identify protein (and optionally RNA) tokens that have MSA data
    # SPEC: has_msa = (token_type <= 1), i.e. protein and RNA
    has_msa_mask = token_types <= 1  # bool (N,)
    prot_indices = np.where(has_msa_mask)[0]
    N_prot = len(prot_indices)

    # Build per-chain dense MSAs for protein chains, then concatenate
    # along the residue dimension
    chain_order = []
    chain_lengths = []
    chain_msa_matrices = []
    chain_del_matrices = []

    # Group protein tokens by chain in order
    seen_chains: list[int] = []
    for idx in prot_indices:
        cid = int(chain_id[idx])
        if cid not in seen_chains:
            seen_chains.append(cid)

    for cid in seen_chains:
        mask_c = (chain_id == cid) & has_msa_mask
        n_res = int(mask_c.sum())
        chain_lengths.append(n_res)
        chain_order.append(cid)

        if cid in msa_dict:
            msa_mat, del_mat = msa_to_dense(msa_dict[cid], n_res, max_seqs)
        else:
            # Dummy: single row of gaps
            msa_mat = np.full((1, n_res), TOKEN_IDS["-"], dtype=np.int64)
            del_mat = np.zeros((1, n_res), dtype=np.float32)

        chain_msa_matrices.append(msa_mat)
        chain_del_matrices.append(del_mat)

    if N_prot == 0:
        # No protein tokens -- return zeros
        profile = torch.zeros(N, 32, dtype=torch.float32)
        del_mean = torch.zeros(N, 1, dtype=torch.float32)
        has_msa_t = torch.zeros(N, 1, dtype=torch.float32)
        msa_feat = torch.zeros(1, 0, 34, dtype=torch.float32)
        msa_mask = torch.zeros(1, 0, dtype=torch.float32)
        return {
            "profile": profile,
            "deletion_mean": del_mean,
            "has_msa": has_msa_t,
            "msa_feat": msa_feat,
            "msa_mask": msa_mask,
            "n_prot": 0,
        }

    # Pad all chain MSAs to the same depth and concatenate
    S_max = max(m.shape[0] for m in chain_msa_matrices)
    S = min(S_max, max_seqs)

    padded_msa: list[np.ndarray] = []
    padded_del: list[np.ndarray] = []
    for msa_mat, del_mat in zip(chain_msa_matrices, chain_del_matrices):
        s = msa_mat.shape[0]
        if s > S:
            msa_mat = msa_mat[:S]
            del_mat = del_mat[:S]
            s = S
        if s < S:
            pad_m = np.full((S - s, msa_mat.shape[1]), TOKEN_IDS["-"], dtype=np.int64)
            pad_d = np.zeros((S - s, del_mat.shape[1]), dtype=np.float32)
            msa_mat = np.concatenate([msa_mat, pad_m], axis=0)
            del_mat = np.concatenate([del_mat, pad_d], axis=0)
        padded_msa.append(msa_mat)
        padded_del.append(del_mat)

    # (S, N_prot) concatenated across chains
    full_msa = np.concatenate(padded_msa, axis=1)  # (S, N_prot)
    full_del = np.concatenate(padded_del, axis=1)  # (S, N_prot)

    # Convert to torch
    msa_t = torch.from_numpy(full_msa).long()  # (S, N_prot)
    del_t = torch.from_numpy(full_del).float()  # (S, N_prot)

    # --- Profile: mean one-hot over MSA depth (SPEC: 32-dim) ---
    # Boltz uses NUM_TOKENS classes then takes first 32; we use NUM_TOKENS which is 32
    msa_onehot = F.one_hot(
        msa_t, num_classes=NUM_TOKENS
    ).float()  # (S, N_prot, NUM_TOKENS)
    profile_prot = msa_onehot.mean(dim=0)  # (N_prot, NUM_TOKENS)
    # Truncate / pad to 32 dims
    if profile_prot.shape[-1] >= 32:
        profile_prot = profile_prot[..., :32]
    else:
        pad = torch.zeros(N_prot, 32 - profile_prot.shape[-1])
        profile_prot = torch.cat([profile_prot, pad], dim=-1)

    # --- Deletion mean ---
    # Transform: arctan scaling (same as Boltz)
    del_transformed = (math.pi / 2) * torch.arctan(del_t / 3.0)
    del_mean_prot = del_transformed.mean(dim=0)  # (N_prot,)

    # --- Scatter to full token dimension ---
    profile = torch.zeros(N, 32, dtype=torch.float32)
    del_mean = torch.zeros(N, 1, dtype=torch.float32)
    has_msa_t = torch.zeros(N, 1, dtype=torch.float32)

    prot_idx_t = torch.from_numpy(prot_indices).long()
    profile[prot_idx_t] = profile_prot
    del_mean[prot_idx_t] = del_mean_prot.unsqueeze(-1)
    has_msa_t[prot_idx_t] = 1.0

    # --- MSA feature matrix for MSA module (SPEC section 3.2) ---
    # cat(one_hot(restype, 32), has_deletion[1], deletion_value[1]) = 34
    has_deletion = (del_t > 0).float().unsqueeze(-1)  # (S, N_prot, 1)
    del_value = del_transformed.unsqueeze(-1)  # (S, N_prot, 1)
    msa_restype_oh = F.one_hot(msa_t, num_classes=NUM_TOKENS).float()
    if msa_restype_oh.shape[-1] >= 32:
        msa_restype_oh = msa_restype_oh[..., :32]
    else:
        pad_size = 32 - msa_restype_oh.shape[-1]
        msa_restype_oh = F.pad(msa_restype_oh, (0, pad_size))

    msa_feat = torch.cat(
        [msa_restype_oh, has_deletion, del_value], dim=-1
    )  # (S, N_prot, 34)

    # MSA mask: 1 where we have real rows
    msa_mask = torch.ones(S, N_prot, dtype=torch.float32)

    return {
        "profile": profile,  # (N, 32)
        "deletion_mean": del_mean,  # (N, 1)
        "has_msa": has_msa_t,  # (N, 1)
        "msa_feat": msa_feat,  # (S, N_prot, 34)
        "msa_mask": msa_mask,  # (S, N_prot)
        "n_prot": N_prot,
    }

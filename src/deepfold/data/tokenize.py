"""Tokenization for DeepFold-Linear.

Adapted from Boltz-1 ``BoltzTokenizer``.

Token type system (SPEC section 3.1):
    0: protein  -- 1 token per residue
    1: RNA      -- 1 token per nucleotide
    2: DNA      -- 1 token per nucleotide
    3: ligand / ion / water -- 1 token per atom

Builds:
    - tokens array with per-token metadata
    - token_idx mapping (atom -> token)
    - token-level bond list
    - bond_matrix for 68-bin position encoding (SPEC section 4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from deepfold.data.parse import (
    CHAIN_TYPES,
    UNK_TOKEN_IDS,
    Structure,
)

# ---------------------------------------------------------------------------
# Token / bond dtypes
# ---------------------------------------------------------------------------

TokenDtype = np.dtype(
    [
        ("token_idx", "i4"),
        ("atom_idx", "i4"),
        ("atom_num", "i4"),
        ("res_idx", "i4"),
        ("res_type", "i4"),
        ("sym_id", "i4"),
        ("asym_id", "i4"),
        ("entity_id", "i4"),
        ("mol_type", "i4"),  # SPEC: 0=protein, 1=RNA, 2=DNA, 3=ligand
        ("center_idx", "i4"),
        ("disto_idx", "i4"),
        ("center_coords", "3f4"),
        ("disto_coords", "3f4"),
        ("resolved_mask", "?"),
        ("disto_mask", "?"),
    ]
)

TokenBondDtype = np.dtype(
    [
        ("token_1", "i4"),
        ("token_2", "i4"),
    ]
)


# ---------------------------------------------------------------------------
# Token type mapping
# ---------------------------------------------------------------------------


def _chain_mol_type_to_token_type(mol_type: int) -> int:
    """Map Boltz chain mol_type to our 4-class token_type.

    Boltz uses: PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3
    SPEC uses:  protein=0, RNA=1, DNA=2, ligand=3
    """
    if mol_type == CHAIN_TYPES["PROTEIN"]:
        return 0
    elif mol_type == CHAIN_TYPES["RNA"]:
        return 1
    elif mol_type == CHAIN_TYPES["DNA"]:
        return 2
    else:  # NONPOLYMER
        return 3


# ---------------------------------------------------------------------------
# Tokenized result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tokenized:
    """Result of tokenization."""

    tokens: np.ndarray  # TokenDtype, shape (N_tokens,)
    bonds: np.ndarray  # TokenBondDtype, shape (N_bonds,)
    atom_to_token: (
        np.ndarray
    )  # int32, shape (N_atoms,) -- maps atom index to token index
    structure: Structure
    # Convenience
    token_types: np.ndarray  # int32 (N_tokens,) -- 0/1/2/3 per SPEC
    chain_id: np.ndarray  # int32 (N_tokens,) -- asym_id per token
    global_idx: np.ndarray  # int32 (N_tokens,) -- residue-level global index per token


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def tokenize(structure: Structure) -> Tokenized:
    """Tokenize a parsed structure.

    Standard polymer residues (protein / RNA / DNA) become one token each.
    Non-standard residues (ligands, ions, modified residues flagged as
    non-standard) are tokenized per atom.

    Returns
    -------
    Tokenized
    """
    token_data: list[tuple] = []
    token_types_list: list[int] = []
    chain_ids_list: list[int] = []
    global_idx_list: list[int] = []
    atom_to_token = np.full(len(structure.atoms), -1, dtype=np.int32)

    token_idx = 0
    valid_chains = (
        structure.chains[structure.mask]
        if len(structure.mask) > 0
        else structure.chains
    )

    for chain in valid_chains:
        mol_type = int(chain["mol_type"])
        tt = _chain_mol_type_to_token_type(mol_type)
        asym_id = int(chain["asym_id"])
        sym_id = int(chain["sym_id"])
        entity_id = int(chain["entity_id"])

        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])

        for res in structure.residues[res_start:res_end]:
            atom_start = int(res["atom_idx"])
            atom_end = atom_start + int(res["atom_num"])

            if res["is_standard"]:
                # One token per residue
                center = structure.atoms[int(res["atom_center"])]
                disto = structure.atoms[int(res["atom_disto"])]
                is_present = bool(res["is_present"]) and bool(center["is_present"])
                is_disto = bool(res["is_present"]) and bool(disto["is_present"])

                token_data.append(
                    (
                        token_idx,
                        atom_start,
                        int(res["atom_num"]),
                        int(res["res_idx"]),
                        int(res["res_type"]),
                        sym_id,
                        asym_id,
                        entity_id,
                        tt,
                        int(res["atom_center"]),
                        int(res["atom_disto"]),
                        tuple(center["coords"]),
                        tuple(disto["coords"]),
                        is_present,
                        is_disto,
                    )
                )
                token_types_list.append(tt)
                chain_ids_list.append(asym_id)
                global_idx_list.append(int(res["res_idx"]))

                atom_to_token[atom_start:atom_end] = token_idx
                token_idx += 1

            else:
                # Non-standard: one token per atom
                unk_id = UNK_TOKEN_IDS.get("PROTEIN", 0)
                for i in range(atom_start, atom_end):
                    atom = structure.atoms[i]
                    is_present = bool(res["is_present"]) and bool(atom["is_present"])
                    coords = tuple(atom["coords"])

                    token_data.append(
                        (
                            token_idx,
                            i,
                            1,
                            int(res["res_idx"]),
                            unk_id,
                            sym_id,
                            asym_id,
                            entity_id,
                            tt,
                            i,
                            i,
                            coords,
                            coords,
                            is_present,
                            is_present,
                        )
                    )
                    token_types_list.append(tt)
                    chain_ids_list.append(asym_id)
                    global_idx_list.append(int(res["res_idx"]))

                    atom_to_token[i] = token_idx
                    token_idx += 1

    # Build token bond list
    token_bonds: list[tuple[int, int]] = []

    # Intra-residue bonds from the structure bond table
    for bond in structure.bonds:
        a1 = int(bond["atom_1"])
        a2 = int(bond["atom_2"])
        t1 = atom_to_token[a1] if a1 < len(atom_to_token) else -1
        t2 = atom_to_token[a2] if a2 < len(atom_to_token) else -1
        if t1 >= 0 and t2 >= 0 and t1 != t2:
            token_bonds.append((t1, t2))

    # Connection (covalent cross-residue) bonds
    for conn in structure.connections:
        a1 = int(conn["atom_1"])
        a2 = int(conn["atom_2"])
        t1 = atom_to_token[a1] if a1 < len(atom_to_token) else -1
        t2 = atom_to_token[a2] if a2 < len(atom_to_token) else -1
        if t1 >= 0 and t2 >= 0 and t1 != t2:
            token_bonds.append((t1, t2))

    # Peptide bonds: consecutive standard residues in the same polymer chain
    # (not encoded explicitly in Boltz bond table, but needed for 68-bin encoding)
    for chain in valid_chains:
        if int(chain["mol_type"]) == CHAIN_TYPES["NONPOLYMER"]:
            continue
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        prev_token: Optional[int] = None
        for res in structure.residues[res_start:res_end]:
            if not res["is_standard"]:
                prev_token = None
                continue
            cur_token = atom_to_token[int(res["atom_idx"])]
            if cur_token < 0:
                prev_token = None
                continue
            if prev_token is not None:
                token_bonds.append((prev_token, cur_token))
            prev_token = cur_token

    # Deduplicate bonds (keep canonical direction)
    bond_set: set[tuple[int, int]] = set()
    for t1, t2 in token_bonds:
        bond_set.add((min(t1, t2), max(t1, t2)))
    token_bonds_dedup = sorted(bond_set)

    # Build numpy arrays
    tokens = (
        np.array(token_data, dtype=TokenDtype)
        if token_data
        else np.empty(0, dtype=TokenDtype)
    )
    bonds_arr = (
        np.array(token_bonds_dedup, dtype=TokenBondDtype)
        if token_bonds_dedup
        else np.empty(0, dtype=TokenBondDtype)
    )
    token_types = np.array(token_types_list, dtype=np.int32)
    chain_id_arr = np.array(chain_ids_list, dtype=np.int32)
    global_idx_arr = np.array(global_idx_list, dtype=np.int32)

    return Tokenized(
        tokens=tokens,
        bonds=bonds_arr,
        atom_to_token=atom_to_token,
        structure=structure,
        token_types=token_types,
        chain_id=chain_id_arr,
        global_idx=global_idx_arr,
    )


# ---------------------------------------------------------------------------
# 68-bin bond matrix (SPEC section 4)
# ---------------------------------------------------------------------------


def build_bond_set(tokenized: Tokenized) -> set[tuple[int, int]]:
    """Return the set of (i, j) pairs that are covalently bonded at token level.

    The set is symmetric: if (i, j) is present, (j, i) is also present.
    """
    bond_set: set[tuple[int, int]] = set()
    for b in tokenized.bonds:
        t1, t2 = int(b["token_1"]), int(b["token_2"])
        bond_set.add((t1, t2))
        bond_set.add((t2, t1))
    return bond_set


def compute_rel_pos_bin(
    i: int,
    j: int,
    chain_id: np.ndarray,
    global_idx: np.ndarray,
    bond_set: set[tuple[int, int]],
) -> int:
    """Compute the 68-bin relative position / bond index for token pair (i, j).

    SPEC section 4.2:
        bin(i,j) =
          clip(g_i - g_j, -32, 32) + 32   if same chain, no bond   -> [0, 64]
          65                                if cross-chain, no bond
          66                                if covalent bond, same chain
          67                                if covalent bond, cross chain
    """
    same_chain = int(chain_id[i] == chain_id[j])
    bond = int((i, j) in bond_set)
    sep = int(np.clip(global_idx[i] - global_idx[j], -32, 32)) + 32

    # Branchless (matches SPEC section 4.4)
    return (
        sep * (1 - bond) * same_chain
        + 65 * (1 - bond) * (1 - same_chain)
        + 66 * bond * same_chain
        + 67 * bond * (1 - same_chain)
    )


def build_bond_matrix_dense(tokenized: Tokenized) -> np.ndarray:
    """Build a dense (N, N) int8 bond matrix for the 68-bin encoding.

    Entry (i, j) contains the bin index in [0, 67].
    This is used for debugging / small structures.  In production the
    bins are computed on-the-fly per attention tile.
    """
    N = len(tokenized.tokens)
    mat = np.zeros((N, N), dtype=np.int8)
    bond_set = build_bond_set(tokenized)
    chain_id = tokenized.chain_id
    global_idx = tokenized.global_idx

    for i in range(N):
        for j in range(N):
            mat[i, j] = compute_rel_pos_bin(i, j, chain_id, global_idx, bond_set)
    return mat

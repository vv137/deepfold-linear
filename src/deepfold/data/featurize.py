"""Feature generation for DeepFold-Linear (SPEC §3).

Takes a parsed + tokenized + cropped structure and produces all tensors
needed by ``DeepFoldLinear.forward()``.

Key outputs (matching the model's forward signature):
    token_type   (N,)       int    — 0=protein, 1=RNA, 2=DNA, 3=ligand
    profile      (N, 32)    float  — MSA residue-type frequencies, zero for non-protein
    del_mean     (N, 1)     float  — MSA deletion mean, zero for non-protein
    has_msa      (N, 1)     float  — 1 for protein/RNA, 0 otherwise
    msa_feat     (S, N_prot, 34)   — cat(restype_onehot[32], has_del[1], del_val[1])
    c_atom       (N_atom, D_ref)   — ref conformer features (cat of ref_pos, charge, mask, elem, name)
    p_lm         (n_pairs, 16_raw) — intra-token atom pair raw features (disp[3], inv_dist[1], valid[1])
    p_lm_idx     (n_pairs, 2)      — atom index pairs for p_lm (into cropped atom array)
    token_idx    (N_atom,)  int    — atom → token mapping
    chain_id     (N,)       int    — chain id per token
    global_idx   (N,)       int    — global residue index (preserved after crop)
    bond_matrix  (N, N)     bool   — covalent bond adjacency
    protein_mask (N,)       bool   — True for tokens with MSA data
    x_atom_true  (N_atom, 3) float — ground-truth atom coords (training)
    x_res_true   (N, 3)     float  — ground-truth token (C-alpha/center) coords (training)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from deepfold.data import const
from deepfold.data.augment import center_random_augmentation


# ---------------------------------------------------------------------------
# D_ref: 3(pos) + 1(charge) + 1(mask) + 128(element_onehot) + 64(atom_name_onehot) = 197
# ---------------------------------------------------------------------------
D_REF = 197
NUM_ATOM_NAME_CLASSES = 64
NUM_MSA_RESTYPE_CLASSES = 32  # num_tokens covers protein+RNA+DNA+pad+gap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot_np(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Simple numpy one-hot encoding."""
    out = np.zeros((len(indices), num_classes), dtype=np.float32)
    valid = (indices >= 0) & (indices < num_classes)
    out[np.arange(len(indices))[valid], indices[valid]] = 1.0
    return out


def _build_intra_token_pairs(
    atom_starts: np.ndarray,
    atom_counts: np.ndarray,
    max_atoms_per_token: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Build all intra-token atom pairs.

    Returns
    -------
    pair_indices : (n_pairs, 2) int — (atom_i, atom_j) in cropped atom space
    pair_token   : (n_pairs,) int  — which token each pair belongs to
    """
    pair_list = []
    tok_list = []
    for tok_id, (start, count) in enumerate(zip(atom_starts, atom_counts)):
        c = min(count, max_atoms_per_token)
        for i in range(c):
            for j in range(c):
                pair_list.append([start + i, start + j])
                tok_list.append(tok_id)
    if pair_list:
        return np.array(pair_list, dtype=np.int64), np.array(tok_list, dtype=np.int64)
    return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)


# ---------------------------------------------------------------------------
# Main featurizer
# ---------------------------------------------------------------------------


def featurize(
    # Token-level arrays (already cropped)
    token_types: np.ndarray,  # (N,) int: mol type per token
    token_res_types: np.ndarray,  # (N,) int: residue type ids
    token_chain_ids: np.ndarray,  # (N,) int: chain/asym id
    token_global_idx: np.ndarray,  # (N,) int: global residue index
    token_resolved_mask: np.ndarray,  # (N,) bool: is resolved
    # Atom-level arrays (already cropped)
    atom_coords: np.ndarray,  # (N_atom, 3)
    atom_ref_pos: np.ndarray,  # (N_atom, 3) reference conformer
    atom_charge: np.ndarray,  # (N_atom,) float
    atom_element: np.ndarray,  # (N_atom,) int (atomic number)
    atom_name: np.ndarray,  # (N_atom,) int (name index)
    atom_mask: np.ndarray,  # (N_atom,) bool — is_present
    atom_to_token: np.ndarray,  # (N_atom,) int — maps atom → token
    # Token → atom mapping
    token_atom_starts: np.ndarray,  # (N,) int — start in atom array
    token_atom_counts: np.ndarray,  # (N,) int — count per token
    # Center atom coords per token (for x_res_true)
    token_center_coords: np.ndarray,  # (N, 3)
    # Bond info: list of (tok_i, tok_j) pairs
    token_bonds: Optional[list[tuple[int, int]]] = None,
    # MSA data (optional)
    msa_data: Optional[
        np.ndarray
    ] = None,  # (S, N_prot) int — residue types in MSA rows
    msa_deletion: Optional[np.ndarray] = None,  # (S, N_prot) float — deletion counts
    msa_mask: Optional[np.ndarray] = None,  # (N,) bool — which tokens have MSA
    # Training flag
    training: bool = True,
    # Augmentation
    s_trans: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Convert parsed structure arrays into tensors for model forward.

    All inputs are numpy arrays. Output is a dict of torch tensors (no batch dim).
    """
    N = len(token_types)
    N_atom = len(atom_coords)

    # ------------------------------------------------------------------
    # 1. token_type (N,) — 0=protein, 1=RNA, 2=DNA, 3=ligand
    # ------------------------------------------------------------------
    token_type = torch.from_numpy(token_types.astype(np.int64))

    # ------------------------------------------------------------------
    # 2. protein_mask (N,) bool — True for protein/RNA tokens
    # ------------------------------------------------------------------
    if msa_mask is not None:
        protein_mask = torch.from_numpy(msa_mask.astype(bool))
    else:
        protein_mask = torch.tensor(
            [(t == const.MOL_PROTEIN or t == const.MOL_RNA) for t in token_types],
            dtype=torch.bool,
        )
    N_prot = int(protein_mask.sum().item())

    # ------------------------------------------------------------------
    # 3. profile (N, 32), del_mean (N, 1), has_msa (N, 1)
    # ------------------------------------------------------------------
    profile = torch.zeros(N, NUM_MSA_RESTYPE_CLASSES, dtype=torch.float32)
    del_mean = torch.zeros(N, 1, dtype=torch.float32)
    has_msa = torch.zeros(N, 1, dtype=torch.float32)

    if msa_data is not None and N_prot > 0:
        # msa_data: (S, N_prot) int residue type indices
        S = msa_data.shape[0]
        msa_onehot = _one_hot_np(msa_data.ravel(), NUM_MSA_RESTYPE_CLASSES)
        msa_onehot = msa_onehot.reshape(S, N_prot, NUM_MSA_RESTYPE_CLASSES)
        prot_profile = msa_onehot.mean(axis=0)  # (N_prot, 32)

        prot_indices = protein_mask.nonzero(as_tuple=True)[0]
        for local_i, global_i in enumerate(prot_indices):
            profile[global_i] = torch.from_numpy(prot_profile[local_i])

        if msa_deletion is not None:
            # arctan transform as in Boltz
            del_transformed = (np.pi / 2.0) * np.arctan(msa_deletion / 3.0)
            del_mean_prot = del_transformed.mean(axis=0)  # (N_prot,)
            for local_i, global_i in enumerate(prot_indices):
                del_mean[global_i, 0] = float(del_mean_prot[local_i])

    # has_msa
    has_msa[protein_mask] = 1.0

    # ------------------------------------------------------------------
    # 4. msa_feat (S, N_prot, 34) — cat(restype_onehot[32], has_del[1], del_val[1])
    # ------------------------------------------------------------------
    if msa_data is not None and N_prot > 0:
        S = msa_data.shape[0]
        msa_restype_oh = torch.from_numpy(
            _one_hot_np(msa_data.ravel(), NUM_MSA_RESTYPE_CLASSES).reshape(
                S, N_prot, NUM_MSA_RESTYPE_CLASSES
            )
        )

        if msa_deletion is not None:
            has_del = torch.from_numpy((msa_deletion > 0).astype(np.float32)).unsqueeze(
                -1
            )  # (S, N_prot, 1)
            del_val = torch.from_numpy(
                ((np.pi / 2.0) * np.arctan(msa_deletion / 3.0)).astype(np.float32)
            ).unsqueeze(-1)  # (S, N_prot, 1)
        else:
            has_del = torch.zeros(S, N_prot, 1)
            del_val = torch.zeros(S, N_prot, 1)

        msa_feat = torch.cat(
            [msa_restype_oh, has_del, del_val], dim=-1
        )  # (S, N_prot, 34)
    else:
        # Dummy MSA: single row with query sequence one-hot
        prot_res_types = (
            token_res_types[protein_mask.numpy()]
            if N_prot > 0
            else np.array([], dtype=np.int64)
        )
        if N_prot > 0:
            query_oh = _one_hot_np(prot_res_types, NUM_MSA_RESTYPE_CLASSES)
            msa_feat = torch.from_numpy(
                np.concatenate(
                    [
                        query_oh,
                        np.zeros((N_prot, 1), dtype=np.float32),
                        np.zeros((N_prot, 1), dtype=np.float32),
                    ],
                    axis=-1,
                )
            ).unsqueeze(0)  # (1, N_prot, 34)
        else:
            msa_feat = torch.zeros(1, 0, 34)

    # ------------------------------------------------------------------
    # 5. c_atom (N_atom, D_ref) — reference conformer features
    # ------------------------------------------------------------------
    ref_pos_t = atom_ref_pos.astype(np.float32)  # (N_atom, 3)
    ref_charge_t = atom_charge.astype(np.float32).reshape(-1, 1)  # (N_atom, 1)
    ref_mask_t = atom_mask.astype(np.float32).reshape(-1, 1)  # (N_atom, 1)
    ref_element_oh = _one_hot_np(atom_element, const.num_elements)  # (N_atom, 128)
    ref_name_oh = _one_hot_np(
        atom_name % NUM_ATOM_NAME_CLASSES, NUM_ATOM_NAME_CLASSES
    )  # (N_atom, 64)

    c_atom_np = np.concatenate(
        [ref_pos_t, ref_charge_t, ref_mask_t, ref_element_oh, ref_name_oh], axis=-1
    )  # (N_atom, D_REF=197)
    c_atom = torch.from_numpy(c_atom_np)

    # ------------------------------------------------------------------
    # 6. p_lm (n_pairs, 5) and p_lm_idx (n_pairs, 2) — intra-token atom pairs
    #    Raw features: displacement(3), inv_dist(1), validity(1)
    #    The model's AtomPairEmbedding will project these to (n_pairs, 16)
    # ------------------------------------------------------------------
    pair_idx_np, _pair_tok = _build_intra_token_pairs(
        token_atom_starts, token_atom_counts
    )  # (n_pairs, 2)

    if len(pair_idx_np) > 0:
        pos_i = atom_ref_pos[pair_idx_np[:, 0]]
        pos_j = atom_ref_pos[pair_idx_np[:, 1]]
        disp = (pos_i - pos_j).astype(np.float32)  # (n_pairs, 3)
        dist_sq = (disp**2).sum(axis=-1, keepdims=True)
        inv_dist = (1.0 / (1.0 + dist_sq)).astype(np.float32)  # (n_pairs, 1)
        valid = np.ones(
            (len(pair_idx_np), 1), dtype=np.float32
        )  # all intra-token → valid
        p_lm_raw = np.concatenate([disp, inv_dist, valid], axis=-1)  # (n_pairs, 5)
    else:
        p_lm_raw = np.zeros((0, 5), dtype=np.float32)

    p_lm = torch.from_numpy(p_lm_raw)
    p_lm_idx = torch.from_numpy(pair_idx_np)

    # ------------------------------------------------------------------
    # 7. token_idx (N_atom,) — atom → token mapping
    # ------------------------------------------------------------------
    token_idx = torch.from_numpy(atom_to_token.astype(np.int64))

    # ------------------------------------------------------------------
    # 8. chain_id (N,), global_idx (N,)
    # ------------------------------------------------------------------
    chain_id = torch.from_numpy(token_chain_ids.astype(np.int64))
    global_idx = torch.from_numpy(token_global_idx.astype(np.int64))

    # ------------------------------------------------------------------
    # 9. bond_matrix (N, N) bool — covalent bond adjacency
    # ------------------------------------------------------------------
    bond_matrix = torch.zeros(N, N, dtype=torch.bool)
    if token_bonds is not None:
        for ti, tj in token_bonds:
            if 0 <= ti < N and 0 <= tj < N:
                bond_matrix[ti, tj] = True
                bond_matrix[tj, ti] = True

    # ------------------------------------------------------------------
    # 10. Ground truth coordinates + augmentation (Boltz Algorithm 19)
    #     Training: center + random SO(3) rotation + small translation
    #     Inference: center + random SO(3) rotation (no translation)
    #     Same rotation/translation applied to atom_coords, token_center_coords,
    #     and atom_ref_pos so the coordinate frame stays consistent.
    # ------------------------------------------------------------------
    valid_mask = atom_mask.astype(bool)
    aug_coords, aug_centers, aug_ref = center_random_augmentation(
        atom_coords,
        token_center_coords,
        atom_ref_pos,
        mask=valid_mask,
        s_trans=s_trans,
        training=training,
        rng=rng,
    )
    x_atom_true = torch.from_numpy(aug_coords)
    x_res_true = torch.from_numpy(aug_centers)
    # Patch c_atom's first 3 columns (ref_pos) with augmented ref_pos
    c_atom[:, :3] = torch.from_numpy(aug_ref)

    # ------------------------------------------------------------------
    # Assemble output dict
    # ------------------------------------------------------------------
    features = {
        "token_type": token_type,  # (N,) int
        "profile": profile,  # (N, 32) float
        "del_mean": del_mean,  # (N, 1) float
        "has_msa": has_msa,  # (N, 1) float
        "msa_feat": msa_feat,  # (S, N_prot, 34) float
        "c_atom": c_atom,  # (N_atom, 197) float
        "p_lm": p_lm,  # (n_pairs, 5) float — raw feats for AtomPairEmbedding
        "p_lm_idx": p_lm_idx,  # (n_pairs, 2) int
        "token_idx": token_idx,  # (N_atom,) int
        "chain_id": chain_id,  # (N,) int
        "global_idx": global_idx,  # (N,) int
        "bond_matrix": bond_matrix,  # (N, N) bool
        "protein_mask": protein_mask,  # (N,) bool
    }

    if training:
        features["x_atom_true"] = x_atom_true  # (N_atom, 3) float
        features["x_res_true"] = x_res_true  # (N, 3) float
        # Resolved masks for loss masking (Boltz-1 convention)
        features["atom_resolved_mask"] = torch.from_numpy(
            atom_mask.astype(np.float32)
        )  # (N_atom,) float
        features["token_resolved_mask"] = torch.from_numpy(
            token_resolved_mask.astype(np.float32)
        )  # (N,) float

    return features


# ---------------------------------------------------------------------------
# Convenience: featurize from NPZ data (Boltz-style structure arrays)
# ---------------------------------------------------------------------------


def featurize_from_npz(
    atoms: np.ndarray,
    residues: np.ndarray,
    chains: np.ndarray,
    tokens: np.ndarray,
    token_bonds_arr: np.ndarray,
    msa: Optional[dict] = None,
    token_indices: Optional[np.ndarray] = None,
    atom_indices: Optional[np.ndarray] = None,
    training: bool = True,
    s_trans: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Featurize from Boltz-style NPZ structured arrays.

    Parameters
    ----------
    atoms : structured np.ndarray
        Columns: coords(3), conformer(3), element(int), charge(float),
        name(int), is_present(bool).
    residues : structured np.ndarray
        Columns: res_type(int), ...
    chains : structured np.ndarray
        Columns: asym_id(int), mol_type(int), ...
    tokens : structured np.ndarray
        Columns: token_idx, asym_id, res_idx, res_type, mol_type,
        atom_idx, atom_num, center_idx, center_coords(3),
        resolved_mask, ...
    token_bonds_arr : structured np.ndarray
        Columns: token_1, token_2.
    msa : dict, optional
        Per-chain MSA data.
    token_indices : np.ndarray, optional
        Subset of token indices to use (after cropping).
    atom_indices : np.ndarray, optional
        Subset of atom indices to use (after cropping).
    training : bool
        Whether to include ground-truth coordinates.

    Returns
    -------
    dict[str, torch.Tensor]
    """
    # Apply cropping if indices provided
    if token_indices is not None:
        tokens = tokens[token_indices]
    if atom_indices is not None:
        atoms = atoms[atom_indices]

    N = len(tokens)

    # Build token-level arrays
    token_types = np.array([t["mol_type"] for t in tokens], dtype=np.int64)
    token_res_types = np.array([t["res_type"] for t in tokens], dtype=np.int64)
    token_chain_ids = np.array([t["asym_id"] for t in tokens], dtype=np.int64)
    token_global_idx = np.array([t["res_idx"] for t in tokens], dtype=np.int64)
    token_resolved = np.array([t["resolved_mask"] for t in tokens], dtype=bool)
    token_center_coords = np.array(
        [t["center_coords"] for t in tokens], dtype=np.float32
    )

    # Build atom-level arrays; re-index into the cropped atom table
    # Each token has atom_num atoms starting at some offset.
    # After cropping, we need to rebuild the contiguous atom table.
    atom_coords_list = []
    atom_ref_pos_list = []
    atom_charge_list = []
    atom_element_list = []
    atom_name_list = []
    atom_mask_list = []
    atom_to_token_list = []
    atom_starts = np.zeros(N, dtype=np.int64)
    atom_counts = np.zeros(N, dtype=np.int64)

    offset = 0
    for tok_local_id, tok in enumerate(tokens):
        start = tok["atom_idx"]
        count = tok["atom_num"]

        if atom_indices is not None:
            # Atoms were already subsetted; we need to find them
            # In the cropped case, atoms is already the subset.
            # We need to build from the original atom start/count.
            # Since atom_indices handles the global → local mapping,
            # we work with the pre-subsetted atoms array.
            pass

        atom_starts[tok_local_id] = offset
        atom_counts[tok_local_id] = count
        offset += count

        # Extract atom data for this token from the (potentially cropped) atoms array
        # If we did direct atom subsetting, atoms is already cropped.
        # If not, we read from the original atoms array using token's atom_idx.
        tok_atoms = (
            atoms[start : start + count]
            if atom_indices is None
            else _get_token_atoms(atoms, tok, atom_indices)
        )

        atom_coords_list.append(tok_atoms["coords"])
        atom_ref_pos_list.append(
            tok_atoms["conformer"]
            if "conformer" in tok_atoms.dtype.names
            else tok_atoms["coords"]
        )
        atom_charge_list.append(
            tok_atoms["charge"]
            if "charge" in tok_atoms.dtype.names
            else np.zeros(count)
        )
        atom_element_list.append(tok_atoms["element"])
        atom_name_list.append(
            tok_atoms["name"]
            if "name" in tok_atoms.dtype.names
            else np.zeros(count, dtype=np.int64)
        )
        atom_mask_list.append(
            tok_atoms["is_present"]
            if "is_present" in tok_atoms.dtype.names
            else np.ones(count, dtype=bool)
        )
        atom_to_token_list.extend([tok_local_id] * count)

    if offset > 0:
        atom_coords_all = np.concatenate(atom_coords_list, axis=0).astype(np.float32)
        atom_ref_pos_all = np.concatenate(atom_ref_pos_list, axis=0).astype(np.float32)
        atom_charge_all = np.concatenate(atom_charge_list, axis=0).astype(np.float32)
        atom_element_all = np.concatenate(atom_element_list, axis=0).astype(np.int64)
        atom_name_all = np.concatenate(atom_name_list, axis=0).astype(np.int64)
        atom_mask_all = np.concatenate(atom_mask_list, axis=0).astype(bool)
    else:
        atom_coords_all = np.zeros((0, 3), dtype=np.float32)
        atom_ref_pos_all = np.zeros((0, 3), dtype=np.float32)
        atom_charge_all = np.zeros((0,), dtype=np.float32)
        atom_element_all = np.zeros((0,), dtype=np.int64)
        atom_name_all = np.zeros((0,), dtype=np.int64)
        atom_mask_all = np.zeros((0,), dtype=bool)

    atom_to_token_all = np.array(atom_to_token_list, dtype=np.int64)

    # Build bond pairs in the cropped token space
    tok_orig_to_local = {}
    for local_id, tok in enumerate(tokens):
        tok_orig_to_local[int(tok["token_idx"])] = local_id

    bond_pairs = []
    for bond in token_bonds_arr:
        t1, t2 = int(bond["token_1"]), int(bond["token_2"])
        if t1 in tok_orig_to_local and t2 in tok_orig_to_local:
            bond_pairs.append((tok_orig_to_local[t1], tok_orig_to_local[t2]))

    # MSA: simplified — extract for protein tokens
    msa_data_np = None
    msa_del_np = None
    msa_mask_np = np.array(
        [(t == const.MOL_PROTEIN or t == const.MOL_RNA) for t in token_types],
        dtype=bool,
    )

    # TODO: Integrate full MSA pairing logic when MSA data is available.
    # For now, generate dummy MSA from query sequence.

    return featurize(
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
        token_atom_starts=atom_starts,
        token_atom_counts=atom_counts,
        token_center_coords=token_center_coords,
        token_bonds=bond_pairs,
        msa_data=msa_data_np,
        msa_deletion=msa_del_np,
        msa_mask=msa_mask_np,
        training=training,
        s_trans=s_trans,
        rng=rng,
    )


def _get_token_atoms(atoms_cropped, token, atom_indices):
    """When atoms were globally subsetted, find atoms belonging to a token."""
    start = token["atom_idx"]
    count = token["atom_num"]
    # atom_indices maps original → position in cropped array
    # We need atoms in [start, start+count)
    mask = (atom_indices >= start) & (atom_indices < start + count)
    result = (
        atoms_cropped[mask]
        if hasattr(atoms_cropped, "__getitem__")
        else atoms_cropped[np.where(mask)]
    )
    # If we don't have exactly count atoms, the crop trimmed some
    return result

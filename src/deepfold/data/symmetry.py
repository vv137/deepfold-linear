"""Symmetry handling for homo-oligomers and symmetric residues.

Adapted from Boltz-1. Three levels:
1. Chain symmetries: permutations of chains with same entity_id
2. Amino acid atom symmetries: e.g. ASP OD1/OD2 swap (from const.ref_symmetries)
3. (Future) Ligand atom symmetries from CCD automorphisms

At loss time, find the ground-truth arrangement closest to prediction
so the model isn't penalized for picking an equivalent symmetric state.
"""

import itertools
import random

import numpy as np
import torch

from deepfold.data import const


# ============================================================================
# Feature extraction (data pipeline, numpy)
# ============================================================================


def compute_chain_symmetries(
    struct: dict,
    cropped_tokens: np.ndarray,
    crop_atom_indices: np.ndarray,
    max_n_symmetries: int = 100,
) -> dict:
    """Compute chain-level symmetries for homo-oligomers.

    Returns dict with:
        all_coords: (N_all_atoms, 3) float32 — full structure coords
        all_resolved_mask: (N_all_atoms,) bool
        crop_to_all_atom_map: (N_crop_atoms,) int64
        chain_symmetries: list of permutation specs
    """
    chains = struct["chains"]
    atoms = struct["atoms"]

    # Build per-chain info from full structure
    chain_starts = []   # start index in flattened all_coords
    chain_counts = []   # atom count per chain
    chain_entity = []   # entity_id per chain
    chain_asym = []     # asym_id per chain
    chain_in_crop = set()  # asym_ids that appear in crop

    all_coords_list = []
    all_resolved_list = []

    for asym_id in np.unique(cropped_tokens["asym_id"]):
        chain_in_crop.add(int(asym_id))

    offset = 0
    for chain in chains:
        aidx = int(chain["atom_idx"])
        anum = int(chain["atom_num"])
        chain_starts.append(offset)
        chain_counts.append(anum)
        chain_entity.append(int(chain["entity_id"]))
        chain_asym.append(int(chain["asym_id"]))

        all_coords_list.append(atoms["coords"][aidx:aidx + anum].astype(np.float32))
        if "is_present" in atoms.dtype.names:
            all_resolved_list.append(atoms["is_present"][aidx:aidx + anum].astype(bool))
        else:
            all_resolved_list.append(np.ones(anum, dtype=bool))
        offset += anum

    all_coords = np.concatenate(all_coords_list, axis=0)
    all_resolved = np.concatenate(all_resolved_list, axis=0)

    # Build crop_to_all_atom_map: for each cropped atom, its index in all_coords
    # We need to map from the original atom indices (crop_atom_indices into struct atoms)
    # to our flattened all_coords array.
    # all_coords is ordered by chain, so chain i starts at chain_starts[i].
    # Original atom index for chain i starts at chains[i]["atom_idx"].
    orig_to_flat = np.empty(len(atoms), dtype=np.int64)
    for i, chain in enumerate(chains):
        aidx = int(chain["atom_idx"])
        anum = int(chain["atom_num"])
        orig_to_flat[aidx:aidx + anum] = np.arange(chain_starts[i], chain_starts[i] + anum)

    crop_to_all = orig_to_flat[crop_atom_indices]

    # Enumerate chain permutations for chains sharing entity_id and in crop
    swaps_per_chain = []
    for i in range(len(chains)):
        if chain_asym[i] not in chain_in_crop:
            continue
        possible = []
        for j in range(len(chains)):
            if chain_entity[i] == chain_entity[j] and chain_counts[i] == chain_counts[j]:
                possible.append((
                    chain_starts[i], chain_starts[i] + chain_counts[i],
                    chain_starts[j], chain_starts[j] + chain_counts[j],
                    i, j,
                ))
        swaps_per_chain.append(possible)

    # Generate combinations, filter for valid permutations
    combinations = list(itertools.islice(
        itertools.product(*swaps_per_chain),
        max_n_symmetries * 10,
    ))

    def _all_different(combo):
        targets = [s[-1] for s in combo]
        return len(targets) == len(set(targets))

    combinations = [c for c in combinations if _all_different(c)]

    if len(combinations) > max_n_symmetries:
        combinations = random.sample(combinations, max_n_symmetries)

    if not combinations:
        combinations.append(())  # identity permutation

    return {
        "all_coords": torch.from_numpy(all_coords),
        "all_resolved_mask": torch.from_numpy(all_resolved),
        "crop_to_all_atom_map": torch.from_numpy(crop_to_all),
        "chain_symmetries": combinations,
    }


def compute_amino_acid_symmetries(
    cropped_tokens: np.ndarray,
    crop_token_to_atom_start: np.ndarray,
    crop_token_to_atom_count: np.ndarray,
) -> list:
    """Compute atom-level symmetries for standard amino acids and nucleotides.

    Returns list of per-residue swap lists. Each swap is a list of (i, j) pairs
    where i and j are atom indices in the cropped atom array.
    """
    swaps = []
    for tok_i in range(len(cropped_tokens)):
        res_type = int(cropped_tokens[tok_i]["res_type"])
        token_name = const.tokens[res_type] if res_type < len(const.tokens) else "PAD"
        symmetries = const.ref_symmetries.get(token_name, [])
        if symmetries:
            atom_start = int(crop_token_to_atom_start[tok_i])
            residue_swaps = []
            for sym in symmetries:
                residue_swaps.append(
                    [(i + atom_start, j + atom_start) for i, j in sym]
                )
            swaps.append(residue_swaps)
    return swaps


# ============================================================================
# Symmetry correction at loss time (torch, no_grad)
# ============================================================================


@torch.no_grad()
def apply_symmetry_correction(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    resolved_mask: torch.Tensor,
    atom_weights: torch.Tensor,
    all_coords: torch.Tensor,
    all_resolved_mask: torch.Tensor,
    crop_to_all_atom_map: torch.Tensor,
    chain_symmetries: list,
    amino_acid_symmetries: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the best symmetric arrangement of ground truth.

    Args:
        pred_coords: (N_atom, 3) predicted coordinates
        true_coords: (N_atom, 3) ground truth (augmented)
        resolved_mask: (N_atom,) bool, resolved atoms
        atom_weights: (N_atom,) per-atom loss weights
        all_coords: (N_all, 3) full structure coordinates
        all_resolved_mask: (N_all,) bool
        crop_to_all_atom_map: (N_crop,) int64, maps crop atoms to all_coords
        chain_symmetries: list of permutation specs
        amino_acid_symmetries: list of per-residue swap specs

    Returns:
        (corrected_true_coords, corrected_resolved_mask) — best symmetric match
    """
    N_crop = crop_to_all_atom_map.shape[0]
    N_atom = pred_coords.shape[0]

    all_c = all_coords.to(pred_coords.device, pred_coords.dtype)
    all_rm = all_resolved_mask.to(pred_coords.device)
    c2a = crop_to_all_atom_map.to(pred_coords.device, torch.long)

    w = (resolved_mask.float() * atom_weights).unsqueeze(-1)  # (N_atom, 1)

    # --- Phase 1: Chain permutations → pick best by weighted RMSD ---
    best_true = true_coords
    best_rm = resolved_mask
    best_mse = float("inf")

    for combo in chain_symmetries:
        perm_all_c = all_c.clone()
        perm_all_rm = all_rm.clone()
        for start1, end1, start2, end2, ci, cj in combo:
            perm_all_c[start1:end1] = all_c[start2:end2]
            perm_all_rm[start1:end1] = all_rm[start2:end2]

        # Extract cropped atoms and align
        perm_crop_c = perm_all_c[c2a]  # (N_crop, 3)
        perm_crop_rm = perm_all_rm[c2a]

        # Pad to full atom count if crop < total atoms in batch
        if N_crop < N_atom:
            pad = N_atom - N_crop
            perm_true = torch.cat([perm_crop_c, true_coords[N_crop:]], dim=0)
            perm_rm = torch.cat([perm_crop_rm, resolved_mask[N_crop:]], dim=0)
        else:
            perm_true = perm_crop_c
            perm_rm = perm_crop_rm

        # Kabsch align
        aligned = _kabsch_align_single(perm_true, pred_coords, w)

        # Weighted MSE
        diff = pred_coords - aligned
        mse = (diff * diff).sum(-1)
        mask_f = perm_rm.float() * atom_weights
        mse_val = (mse * mask_f).sum() / mask_f.sum().clamp(min=1e-8)

        if mse_val.item() < best_mse:
            best_mse = mse_val.item()
            best_true = aligned
            best_rm = perm_rm

    # --- Phase 2: Amino acid atom swaps (greedy, no realignment) ---
    # Re-align best chain result for MSE comparison
    w_cmp = (best_rm.float() * atom_weights).unsqueeze(-1)
    best_true = _kabsch_align_single(best_true, pred_coords, w_cmp)

    for residue_swaps in amino_acid_symmetries:
        for swap in residue_swaps:
            new_true = best_true.clone()
            new_rm = best_rm.clone()
            for i, j in swap:
                if i < N_atom and j < N_atom:
                    new_true[i] = best_true[j]
                    new_rm[i] = best_rm[j]

            # Compare MSE without realignment
            mask_f = best_rm.float() * atom_weights
            old_mse = ((pred_coords - best_true) ** 2).sum(-1)
            old_mse = (old_mse * mask_f).sum()
            new_mask_f = new_rm.float() * atom_weights
            new_mse = ((pred_coords - new_true) ** 2).sum(-1)
            new_mse = (new_mse * new_mask_f).sum()

            if new_mse < old_mse:
                best_true = new_true
                best_rm = new_rm

    # Return permuted coords — loss functions do their own Kabsch alignment
    return best_true, best_rm


def _kabsch_align_single(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Kabsch align true to pred. All (N, 3), weights (N, 1)."""
    w = weights
    w_sum = w.sum().clamp(min=1e-8)

    true_c = (true_coords * w).sum(0, keepdim=True) / w_sum
    pred_c = (pred_coords * w).sum(0, keepdim=True) / w_sum

    true_cen = true_coords - true_c
    pred_cen = pred_coords - pred_c

    H = (w * pred_cen).T @ true_cen  # (3, 3)

    with torch.amp.autocast(pred_coords.device.type, enabled=False):
        H_f32 = H.float()
        U, S, Vh = torch.linalg.svd(H_f32)
        d = torch.det(U @ Vh).sign()
        D = torch.ones(3, device=H.device, dtype=H_f32.dtype)
        D[-1] = d
        R = U @ torch.diag(D) @ Vh
    R = R.to(true_coords.dtype)

    aligned = true_cen @ R.T + pred_c
    return aligned

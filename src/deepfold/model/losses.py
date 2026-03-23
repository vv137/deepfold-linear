"""Loss functions (SPEC §11)."""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from deepfold.data import const
from deepfold.model.diffusion import SIGMA_DATA

# Per-atom-type loss weights (Boltz-1 convention)
_NUCLEOTIDE_LOSS_WEIGHT = 5.0
_LIGAND_LOSS_WEIGHT = 10.0


def _mol_type_to_weights(atom_mol: torch.Tensor) -> torch.Tensor:
    """Convert per-atom molecule types to loss weights. Works on any shape."""
    w = torch.ones_like(atom_mol, dtype=torch.float32)
    w = w + _NUCLEOTIDE_LOSS_WEIGHT * (
        (atom_mol == const.MOL_DNA).float() + (atom_mol == const.MOL_RNA).float()
    )
    w = w + _LIGAND_LOSS_WEIGHT * (atom_mol == const.MOL_NONPOLYMER).float()
    return w


def _atom_type_weights(
    token_idx: torch.Tensor,
    token_type: torch.Tensor,
) -> torch.Tensor:
    """Per-atom loss weights based on molecule type (Boltz-1 convention).

    Protein=1x, nucleotide=5x, nonpolymer=10x.

    Args:
        token_idx:  (B, N_atom) or (N_atom,) int — atom -> token mapping
        token_type: (B, N) or (N,) int — mol_type per token

    Returns:
        (B, N_atom) or (N_atom,) float weights
    """
    if token_idx.dim() == 1:
        return _mol_type_to_weights(token_type[token_idx])

    # Batched: (B, N_atom), (B, N) -> (B, N_atom)
    return _mol_type_to_weights(torch.gather(token_type, 1, token_idx))


# ============================================================================
# Weighted Rigid Alignment (AF3 Algorithm 28, Boltz-1)
# ============================================================================


def weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Align ground truth to prediction via weighted Kabsch (SVD).

    Aligns true_coords to pred_coords under no_grad — gradients only
    flow through the MSE against the aligned target, not through SVD.

    Args:
        true_coords:  (B, N, 3) ground truth
        pred_coords:  (B, N, 3) predicted (alignment target)
        weights:      (B, N) per-atom alignment weights
        mask:         (B, N) 1=valid, 0=pad

    Returns:
        (B, N, 3) true_coords rigidly aligned to pred_coords (detached)
    """
    w = (mask * weights).unsqueeze(-1)  # (B, N, 1)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Weighted centroids
    true_c = (true_coords * w).sum(dim=1, keepdim=True) / w_sum
    pred_c = (pred_coords * w).sum(dim=1, keepdim=True) / w_sum

    # Center
    true_centered = true_coords - true_c
    pred_centered = pred_coords - pred_c

    # Weighted cross-covariance: H = P^T W X
    H = torch.einsum("bni,bnj->bij", w * pred_centered, true_centered)

    # SVD in float32 for numerical stability
    H_f32 = H.float()
    U, S, Vh = torch.linalg.svd(H_f32)
    V = Vh.mH  # (B, 3, 3)

    # Ensure proper rotation (det=+1)
    d = torch.det(U @ Vh).sign()  # (B,)
    D = torch.ones(H.shape[0], 3, device=H.device, dtype=H_f32.dtype)
    D[:, -1] = d
    R = torch.einsum("bij,bjk,blk->bil", U, torch.diag_embed(D), V)
    R = R.to(true_coords.dtype)

    # Apply rotation + translation
    aligned = torch.einsum("bni,bji->bnj", true_centered, R) + pred_c
    return aligned.detach()


# ============================================================================
# EDM Diffusion Loss (SPEC §11.1, Boltz-1 convention)
# ============================================================================


def edm_diffusion_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    sigma: torch.Tensor,
    resolved_mask: torch.Tensor | None = None,
    atom_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    EDM-weighted diffusion loss with resolved masking (Boltz-1).

    L_diff = weight(sigma) * sum(mse * w * mask) / sum(3 * w * mask)

    Args:
        x_pred:        (B, N_atom, 3) or (N_atom, 3) predicted denoised coords
        x_true:        (B, N_atom, 3) or (N_atom, 3) ground truth coords
        sigma:         (B,) or scalar noise level
        resolved_mask: (B, N_atom) or (N_atom,) float, 1 for resolved, 0 otherwise
        atom_weights:  (B, N_atom) or (N_atom,) float, per-atom type weights
    """
    sigma_data = SIGMA_DATA

    # Unbatched path → add batch dim, run batched, squeeze
    if x_pred.dim() == 2:
        return edm_diffusion_loss(
            x_pred.unsqueeze(0), x_true.unsqueeze(0), sigma.unsqueeze(0),
            resolved_mask=resolved_mask.unsqueeze(0) if resolved_mask is not None else None,
            atom_weights=atom_weights.unsqueeze(0) if atom_weights is not None else None,
        )

    # Batched path: (B, N_atom, 3)
    B = x_pred.shape[0]
    N_atom = x_pred.shape[1]

    if resolved_mask is None:
        resolved_mask = torch.ones(B, N_atom, device=x_pred.device)
    if atom_weights is None:
        atom_weights = torch.ones(B, N_atom, device=x_pred.device)

    # Weighted Kabsch alignment: align ground truth to prediction (AF3 Alg 28)
    # Under no_grad — gradients only flow through x_pred in the MSE.
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        x_true_aligned = weighted_rigid_align(
            x_true.detach().float(),
            x_pred.detach().float(),
            atom_weights.detach().float(),
            mask=resolved_mask.detach().float(),
        ).to(x_pred.dtype)

    # sigma: scalar or (B,) -> edm_weight (B,)
    if sigma.dim() == 0:
        sigma = sigma.expand(B)
    edm_weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2  # (B,)

    # Per-atom MSE against aligned ground truth: (B, N_atom)
    mse = ((x_pred - x_true_aligned) ** 2).sum(dim=-1)

    # Per-sample loss: (B,)
    numerator = (mse * atom_weights * resolved_mask).sum(dim=1)  # (B,)
    denominator = (3.0 * atom_weights * resolved_mask).sum(dim=1).clamp(min=1.0)  # (B,)

    per_sample = edm_weight * numerator / denominator  # (B,)
    return per_sample.mean()


# ============================================================================
# Smooth LDDT Loss (SPEC §11.2)
# ============================================================================


def smooth_lddt(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    cutoff: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    slope: float = 10.0,
    resolved_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Differentiable smooth LDDT loss (SPEC §11.2).

    L_lddt = 1 - mean over valid pairs of mean over thresholds of
             sigmoid((threshold - |d_pred - d_true|) * slope)

    Args:
        x_pred: (B, M, 3) or (M, 3) predicted coordinates
        x_true: (B, M, 3) or (M, 3) ground truth coordinates
        cutoff: distance cutoff for valid pairs
        thresholds: LDDT thresholds in Angstroms
        slope: sigmoid steepness
        resolved_mask: (B, M) or (M,) float, 1 for resolved, 0 otherwise
    """
    # Unbatched path
    if x_pred.dim() == 2:
        return _smooth_lddt_single(x_pred, x_true, cutoff, thresholds, slope, resolved_mask)

    # Batched path: (B, M, 3)
    B, M, _ = x_pred.shape

    # Pairwise distances: (B, M, M)
    d_pred = torch.cdist(x_pred, x_pred)
    d_true = torch.cdist(x_true, x_true)

    # Valid pairs: true distance < cutoff, exclude diagonal, both resolved
    eye = torch.eye(M, device=x_pred.device).unsqueeze(0)  # (1, M, M)
    valid = (d_true < cutoff) & (eye == 0)

    if resolved_mask is not None:
        # (B, M) -> (B, M, M) outer product mask
        pair_mask = resolved_mask.unsqueeze(-1) * resolved_mask.unsqueeze(-2)
        valid = valid & (pair_mask > 0)

    # Per-sample LDDT
    dev = (d_pred - d_true).abs()  # (B, M, M)

    # Compute per-sample losses
    losses = []
    for b in range(B):
        valid_b = valid[b]
        if not valid_b.any():
            losses.append((x_pred[b] * 0).sum())
            continue
        dev_b = dev[b]
        scores = []
        for t in thresholds:
            s = torch.sigmoid((t - dev_b[valid_b]) * slope)
            scores.append(s)
        lddt_b = torch.stack(scores, dim=0).mean(dim=0).mean()
        losses.append(1.0 - lddt_b)

    return torch.stack(losses).mean()


def _smooth_lddt_single(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    cutoff: float,
    thresholds: tuple[float, ...],
    slope: float,
    resolved_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Original unbatched smooth LDDT."""
    M = x_pred.shape[0]
    d_pred = torch.cdist(x_pred, x_pred)
    d_true = torch.cdist(x_true, x_true)

    valid = (d_true < cutoff) & (torch.eye(M, device=x_pred.device) == 0)
    if resolved_mask is not None:
        pair_mask = resolved_mask[:, None] * resolved_mask[None, :]
        valid = valid & (pair_mask > 0)

    if not valid.any():
        return (x_pred * 0).sum()

    dev = (d_pred - d_true).abs()
    scores = []
    for t in thresholds:
        s = torch.sigmoid((t - dev[valid]) * slope)
        scores.append(s)

    lddt = torch.stack(scores, dim=0).mean(dim=0).mean()
    return 1.0 - lddt


# ============================================================================
# Low-Rank Bilinear Distogram Loss (SPEC §11.3)
# ============================================================================


class DistogramLoss(nn.Module):
    """Low-rank bilinear distogram loss with tiling (SPEC §11.3)."""

    def __init__(
        self,
        d_model: int = 512,
        d_low: int = 64,
        num_bins: int = 39,
        tile_size: int = 64,
        dist_min: float = 2.0,
        dist_max: float = 22.0,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_u = nn.Linear(d_model, d_low)
        self.w_v = nn.Linear(d_model, d_low)
        self.to_bins = nn.Linear(d_low, num_bins)
        self.tile_size = tile_size
        self.num_bins = num_bins
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.bin_width = (dist_max - dist_min) / (num_bins - 1)  # 0.5A

    def distance_to_bins(self, d: torch.Tensor) -> torch.Tensor:
        """Convert distances to bin indices (0-38)."""
        bins = ((d - self.dist_min) / self.bin_width).long()
        return bins.clamp(0, self.num_bins - 1)

    def forward(
        self,
        h_res: torch.Tensor,
        x_true: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        token_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h_res:          (B, N, 512) or (N, 512) token representations
            x_true:         (B, N, 3) or (N, 3) ground truth token coordinates
            valid_mask:     (B, N, N) or (N, N) optional explicit mask
            token_pad_mask: (B, N) float, 1=valid 0=pad. Used to build valid_mask
                            if valid_mask is not provided. Batched mode only.
        """
        # Unbatched path
        if h_res.dim() == 2:
            return self._forward_unbatched(h_res, x_true, valid_mask)

        # Batched path: (B, N, d)
        B, N, _ = h_res.shape

        # Build valid_mask from token_pad_mask if not provided
        if valid_mask is None and token_pad_mask is not None:
            valid_mask = token_pad_mask.unsqueeze(-1) * token_pad_mask.unsqueeze(-2)

        # Ground truth distance bins (FP32 for precision)
        d_true = torch.cdist(x_true.float(), x_true.float())  # (B, N, N)
        target_bins = self.distance_to_bins(d_true)  # (B, N, N)

        # O(N) projections
        h = self.ln(h_res.float()).to(h_res.dtype)
        U = self.w_u(h)  # (B, N, d_low)
        V = self.w_v(h)  # (B, N, d_low)

        # Accumulate per-sample losses
        total_loss = torch.zeros(1, device=h_res.device, dtype=torch.float32)
        total_count = 0
        T = self.tile_size

        for b in range(B):
            sample_loss = torch.zeros(1, device=h_res.device, dtype=torch.float32)
            sample_count = 0

            for i0 in range(0, N, T):
                ie = min(i0 + T, N)
                U_tile = U[b, i0:ie]

                for j0 in range(0, N, T):
                    je = min(j0 + T, N)
                    V_tile = V[b, j0:je]

                    Z = U_tile[:, None, :] * V_tile[None, :, :]  # (ti, tj, d_low)
                    logits = self.to_bins(Z)  # (ti, tj, num_bins)
                    targets = target_bins[b, i0:ie, j0:je]

                    if valid_mask is not None:
                        m = valid_mask[b, i0:ie, j0:je] > 0
                        if m.any():
                            tile_n = m.sum().item()
                            sample_loss = sample_loss + F.cross_entropy(
                                logits[m].float(), targets[m], reduction="mean"
                            ) * tile_n
                            sample_count += tile_n
                    else:
                        tile_n = (ie - i0) * (je - j0)
                        sample_loss = sample_loss + F.cross_entropy(
                            logits.reshape(-1, self.num_bins).float(),
                            targets.reshape(-1),
                            reduction="mean",
                        ) * tile_n
                        sample_count += tile_n

            if sample_count > 0:
                total_loss = total_loss + sample_loss / sample_count
                total_count += 1

        if total_count == 0:
            return total_loss.squeeze()
        return (total_loss / total_count).squeeze()

    def _forward_unbatched(
        self,
        h_res: torch.Tensor,
        x_true: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Original unbatched forward path."""
        N = h_res.shape[0]

        d_true = torch.cdist(x_true.float(), x_true.float())
        target_bins = self.distance_to_bins(d_true)

        h = self.ln(h_res.float()).to(h_res.dtype)
        U = self.w_u(h)
        V = self.w_v(h)

        total_loss = torch.zeros(1, device=h_res.device, dtype=torch.float32)
        count = 0
        T = self.tile_size

        for i0 in range(0, N, T):
            ie = min(i0 + T, N)
            U_tile = U[i0:ie]

            for j0 in range(0, N, T):
                je = min(j0 + T, N)
                V_tile = V[j0:je]

                Z = U_tile[:, None, :] * V_tile[None, :, :]
                logits = self.to_bins(Z)
                targets = target_bins[i0:ie, j0:je]

                if valid_mask is not None:
                    m = valid_mask[i0:ie, j0:je] > 0
                    if m.any():
                        tile_n = m.sum().item()
                        total_loss = total_loss + F.cross_entropy(
                            logits[m].float(), targets[m], reduction="mean"
                        ) * tile_n
                        count += tile_n
                else:
                    tile_n = (ie - i0) * (je - j0)
                    total_loss = total_loss + F.cross_entropy(
                        logits.reshape(-1, self.num_bins).float(),
                        targets.reshape(-1),
                        reduction="mean",
                    ) * tile_n
                    count += tile_n

        return (total_loss / max(count, 1)).squeeze()


# ============================================================================
# Chain Permutation Symmetry (SPEC §11.4)
# ============================================================================


def permutation_invariant_loss(
    pred_x: torch.Tensor,
    true_x: torch.Tensor,
    chain_ids: torch.Tensor,
    loss_fn,
) -> torch.Tensor:
    """
    Chain permutation-invariant loss (SPEC §11.4).
    Hungarian for K>4 chains, enumerate for K<=4.
    """
    unique_chains = torch.unique(chain_ids)
    K = len(unique_chains)

    if K <= 1:
        return loss_fn(pred_x, true_x)

    def permute_chains(x, perm):
        """Reorder chains in x according to permutation."""
        result = x.clone()
        for i, chain in enumerate(unique_chains):
            mask = chain_ids == chain
            target_chain = unique_chains[perm[i]]
            target_mask = chain_ids == target_chain
            result[mask] = x[target_mask]
        return result

    if K <= 4:
        best_loss = torch.tensor(float("inf"), device=pred_x.device)
        for perm in itertools.permutations(range(K)):
            permuted_true = permute_chains(true_x, perm)
            loss = loss_fn(pred_x, permuted_true)
            best_loss = torch.minimum(best_loss, loss)
        return best_loss
    else:
        # Hungarian on chain-chain RMSD
        rmsd_matrix = torch.zeros(K, K)
        for i, ci in enumerate(unique_chains):
            mask_i = chain_ids == ci
            for j, cj in enumerate(unique_chains):
                mask_j = chain_ids == cj
                if mask_i.sum() == mask_j.sum():
                    rmsd = (
                        ((pred_x[mask_i] - true_x[mask_j]) ** 2).sum(-1).mean().sqrt()
                    )
                    rmsd_matrix[i, j] = rmsd.item()
                else:
                    rmsd_matrix[i, j] = float("inf")

        row_ind, col_ind = linear_sum_assignment(rmsd_matrix.cpu().numpy())
        permuted_true = permute_chains(true_x, col_ind)
        return loss_fn(pred_x, permuted_true)


# ============================================================================
# Total Loss (SPEC §11.6)
# ============================================================================


def total_loss(
    l_diff: torch.Tensor,
    l_lddt: torch.Tensor,
    l_disto: torch.Tensor,
    l_trunk_coord: torch.Tensor,
    w_diff: float = 1.0,
    w_lddt: float = 1.0,
    w_disto: float = 0.2,
    w_trunk_coord: float = 0.5,
) -> torch.Tensor:
    """Weighted total loss (SPEC §11.6).

    Default weights: L_diff + L_lddt + 0.2*L_disto + 0.5*L_trunk_coord.
    Configurable via loss_weights in model.yaml.
    """
    return w_diff * l_diff + w_lddt * l_lddt + w_disto * l_disto + w_trunk_coord * l_trunk_coord

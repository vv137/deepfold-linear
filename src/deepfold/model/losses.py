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


def _atom_type_weights(
    token_idx: torch.Tensor,
    token_type: torch.Tensor,
) -> torch.Tensor:
    """Per-atom loss weights based on molecule type (Boltz-1 convention).

    Protein=1×, nucleotide=5×, nonpolymer=10×.

    Args:
        token_idx: (N_atom,) int — atom → token mapping
        token_type: (N,) int — mol_type per token

    Returns:
        (N_atom,) float weights
    """
    atom_mol = token_type[token_idx]  # (N_atom,)
    w = torch.ones_like(atom_mol, dtype=torch.float32)
    w = w + _NUCLEOTIDE_LOSS_WEIGHT * (
        (atom_mol == const.MOL_DNA).float() + (atom_mol == const.MOL_RNA).float()
    )
    w = w + _LIGAND_LOSS_WEIGHT * (atom_mol == const.MOL_NONPOLYMER).float()
    return w


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

    L_diff = weight(σ) × Σ(mse × w × mask) / Σ(3 × w × mask)

    Args:
        x_pred:        (N_atom, 3) predicted denoised coords
        x_true:        (N_atom, 3) ground truth coords
        sigma:         scalar noise level
        resolved_mask: (N_atom,) float, 1 for resolved atoms, 0 otherwise
        atom_weights:  (N_atom,) float, per-atom type weights
    """
    sigma_data = SIGMA_DATA
    edm_weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2

    # Per-atom squared error summed over xyz
    mse = ((x_pred - x_true) ** 2).sum(dim=-1)  # (N_atom,)

    if resolved_mask is None:
        resolved_mask = torch.ones(mse.shape[0], device=mse.device)
    if atom_weights is None:
        atom_weights = torch.ones(mse.shape[0], device=mse.device)

    # Weighted mean following Boltz-1: divide by 3 * sum(weights) for xyz normalization
    numerator = (mse * atom_weights * resolved_mask).sum()
    denominator = (3.0 * atom_weights * resolved_mask).sum().clamp(min=1.0)

    return edm_weight * numerator / denominator


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
        x_pred: (M, 3) predicted coordinates (atom or token level)
        x_true: (M, 3) ground truth coordinates
        cutoff: distance cutoff for valid pairs
        thresholds: LDDT thresholds in Angstroms
        slope: sigmoid steepness
        resolved_mask: (M,) float, 1 for resolved, 0 otherwise
    """
    M = x_pred.shape[0]

    # Pairwise distances
    d_pred = torch.cdist(x_pred, x_pred)  # (M, M)
    d_true = torch.cdist(x_true, x_true)  # (M, M)

    # Valid pairs: true distance < cutoff, exclude diagonal, both resolved
    valid = (d_true < cutoff) & (torch.eye(M, device=x_pred.device) == 0)

    if resolved_mask is not None:
        pair_mask = resolved_mask[:, None] * resolved_mask[None, :]  # (M, M)
        valid = valid & (pair_mask > 0)

    if not valid.any():
        return (x_pred * 0).sum()

    # Distance deviation
    dev = (d_pred - d_true).abs()  # (M, M)

    # Smooth LDDT per threshold
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
    ) -> torch.Tensor:
        """
        Args:
            h_res:      (N, 512) token representations
            x_true:     (N, 3) ground truth token coordinates
            valid_mask: (N, N) optional mask
        """
        N = h_res.shape[0]

        # Ground truth distance bins (FP32 for precision with BF16 models)
        d_true = torch.cdist(x_true.float(), x_true.float())
        target_bins = self.distance_to_bins(d_true)  # (N, N)

        # O(N) projections (LayerNorm stabilizes input from deep trunk)
        h = self.ln(h_res.float()).to(h_res.dtype)
        U = self.w_u(h)  # (N, d_low)
        V = self.w_v(h)  # (N, d_low)

        # Accumulate in FP32 to avoid precision loss across tiles
        total_loss = torch.zeros(1, device=h_res.device, dtype=torch.float32)
        count = 0
        T = self.tile_size

        for i0 in range(0, N, T):
            ie = min(i0 + T, N)
            U_tile = U[i0:ie]

            for j0 in range(0, N, T):
                je = min(j0 + T, N)
                V_tile = V[j0:je]

                Z = U_tile[:, None, :] * V_tile[None, :, :]  # (ti, tj, d_low)
                logits = self.to_bins(Z)  # (ti, tj, num_bins)
                targets = target_bins[i0:ie, j0:je]

                if valid_mask is not None:
                    m = valid_mask[i0:ie, j0:je] > 0  # ensure bool
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

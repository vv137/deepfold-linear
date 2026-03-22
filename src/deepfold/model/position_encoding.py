"""68-bin position/bond encoding (SPEC §4)."""

import torch
import torch.nn as nn


def compute_bins(
    chain_id: torch.Tensor,
    global_idx: torch.Tensor,
    bond_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 68-bin position/bond encoding for all pairs (i, j).

    Args:
        chain_id:    (N,) int — chain identifier per token
        global_idx:  (N,) int — global residue index (preserved after cropping)
        bond_matrix: (N, N) bool — covalent bond adjacency

    Returns:
        bins: (N, N) int in [0, 67]
    """
    N = chain_id.shape[0]

    same_chain = chain_id.unsqueeze(1) == chain_id.unsqueeze(0)  # (N, N)
    bond = bond_matrix.bool()

    # Sequence separation: clip to [-32, 32] + 32 -> [0, 64]
    sep = global_idx.unsqueeze(1) - global_idx.unsqueeze(0)  # (N, N)
    sep = sep.clamp(-32, 32) + 32  # [0, 64]

    # Branchless formula (SPEC §4.4)
    bins = (
        sep * (~bond & same_chain).long()
        + 65 * (~bond & ~same_chain).long()
        + 66 * (bond & same_chain).long()
        + 67 * (bond & ~same_chain).long()
    )

    return bins


class PositionBias(nn.Module):
    """
    Learnable per-head position bias from 68-bin encoding.

    w_rel_res: (H_res, 68) for token UOT blocks — zeros init, AdamW decay
    w_rel_msa: (H_msa, 68) for MSA row attention — zeros init, AdamW decay
    """

    def __init__(self, num_heads: int, num_bins: int = 68):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_heads, num_bins))

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bins: (N, N) int in [0, num_bins-1]
        Returns:
            bias: (H, N, N) per-head position bias
        """
        return self.weight[:, bins.long()]  # (H, N, N) via advanced indexing

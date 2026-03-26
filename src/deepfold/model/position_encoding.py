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

    Supports both unbatched and batched inputs:
        Unbatched: chain_id (N,), global_idx (N,), bond_matrix (N,N) -> (N,N)
        Batched:   chain_id (B,N), global_idx (B,N), bond_matrix (B,N,N) -> (B,N,N)

    Args:
        chain_id:    (N,) or (B,N) int — chain identifier per token
        global_idx:  (N,) or (B,N) int — global residue index
        bond_matrix: (N,N) or (B,N,N) bool — covalent bond adjacency

    Returns:
        bins: (N,N) or (B,N,N) int in [0, 67]
    """
    unbatched = chain_id.dim() == 1
    if unbatched:
        chain_id = chain_id.unsqueeze(0)
        global_idx = global_idx.unsqueeze(0)
        bond_matrix = bond_matrix.unsqueeze(0)

    # (B, N, N)
    same_chain = chain_id.unsqueeze(2) == chain_id.unsqueeze(1)
    bond = bond_matrix.bool()

    # Sequence separation: clip to [-32, 32] + 32 -> [0, 64]
    sep = global_idx.unsqueeze(2) - global_idx.unsqueeze(1)  # (B, N, N)
    sep = sep.clamp(-32, 32) + 32  # [0, 64]

    # Branchless formula (SPEC §4.4)
    bins = (
        sep * (~bond & same_chain).long()
        + 65 * (~bond & ~same_chain).long()
        + 66 * (bond & same_chain).long()
        + 67 * (bond & ~same_chain).long()
    )

    bins = bins.to(torch.int32)
    if unbatched:
        return bins.squeeze(0)
    return bins


class PositionBias(nn.Module):
    """
    Learnable per-head position bias from 68-bin encoding.

    w_rel_res: (H_res, 68) per UOT block — zeros init, no decay
    w_rel_msa: (H_msa, 68) for MSA row attention — zeros init, no decay
    """

    def __init__(self, num_heads: int, num_bins: int = 68):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_heads, num_bins))

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bins: (N, N) or (B, N, N) int in [0, num_bins-1]
        Returns:
            bias: (H, N, N) or (B, H, N, N) per-head position bias
        """
        if bins.dim() == 2:
            return self.weight[:, bins.long()]  # (H, N, N)
        # Batched: (B, N, N) -> (B, H, N, N) via F.embedding + permute
        # F.embedding: (B, N, N) -> (B, N, N, H), then permute to (B, H, N, N)
        return torch.nn.functional.embedding(bins.long(), self.weight.t()).permute(
            0, 3, 1, 2
        )

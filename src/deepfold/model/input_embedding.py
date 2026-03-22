"""Input embeddings (SPEC §3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenSingleEmbedding(nn.Module):
    """
    Token single representation (SPEC §3.1).
    Input: cat(token_type_onehot[4], profile[32], del_mean[1], has_msa[1]) = 38
    Output: (N, 512)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.proj = nn.Linear(38, d_model)

    def forward(
        self,
        token_type: torch.Tensor,
        profile: torch.Tensor,
        del_mean: torch.Tensor,
        has_msa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_type: (N,) int in {0,1,2,3}
            profile:    (N, 32) MSA frequencies, zero for non-protein
            del_mean:   (N, 1) deletion mean, zero for non-protein
            has_msa:    (N, 1) 1 for protein/RNA, 0 otherwise
        """
        token_onehot = F.one_hot(token_type.long(), 4).float()  # (N, 4)
        feat = torch.cat([token_onehot, profile, del_mean, has_msa], dim=-1)  # (N, 38)
        return self.proj(feat)  # (N, 512)


class MSAEmbedding(nn.Module):
    """
    MSA matrix embedding (SPEC §3.2).
    Input: cat(msa_restype_onehot[32], has_del[1], del_val[1]) = 34
    Output: (S, N_prot, 64)
    """

    def __init__(self, d_msa: int = 64):
        super().__init__()
        self.proj = nn.Linear(34, d_msa)

    def forward(self, msa_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa_feat: (S, N_prot, 34) concatenated MSA features
        """
        return self.proj(msa_feat)


class AtomSingleEmbedding(nn.Module):
    """
    Atom single representation — frozen during diffusion (SPEC §3.3).
    Input: cat(ref_pos[3], ref_charge[1], ref_mask[1], ref_element[128], ref_atom_name[64]) = D_ref=197
    Output: (N_atom, 128)
    """

    D_REF = 197  # 3 + 1 + 1 + 128 + 64

    def __init__(self, d_ref: int = 197, d_atom: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_ref, d_atom)

    def forward(self, ref_feat: torch.Tensor) -> torch.Tensor:
        """ref_feat: (N_atom, D_ref)"""
        return self.proj(ref_feat)


class AtomPairEmbedding(nn.Module):
    """
    Atom pair representation — frozen during diffusion (SPEC §3.4).
    Three small projections from reference conformer geometry.
    Output: (local_pairs, 16)
    """

    def __init__(self, d_pair: int = 16):
        super().__init__()
        self.disp_proj = nn.Linear(3, d_pair)
        self.inv_dist_proj = nn.Linear(1, d_pair)
        self.valid_proj = nn.Linear(1, d_pair)

    def forward(
        self,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            d_lm: (local_pairs, 3) displacement vectors between atom pairs
            v_lm: (local_pairs, 1) validity mask (same token)
        """
        dist_sq = (d_lm**2).sum(dim=-1, keepdim=True)  # (local, 1)
        inv_dist = 1.0 / (1.0 + dist_sq)  # (local, 1)

        p_lm = (
            self.disp_proj(d_lm) * v_lm
            + self.inv_dist_proj(inv_dist) * v_lm
            + self.valid_proj(v_lm)
        )
        return p_lm  # (local_pairs, 16)

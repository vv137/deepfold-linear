"""Integration tests for MSA, trunk block, and full model."""

import torch

from deepfold.model.msa import MSABlock
from deepfold.model.trunk_block import TokenUOTBlock
from deepfold.model.position_encoding import compute_bins
from deepfold.model.diffusion import (
    AtomBlock,
    FourierEmbedding,
    edm_preconditioning,
    karras_schedule,
)
from deepfold.model.losses import smooth_lddt, DistogramLoss, edm_diffusion_loss


class TestMSABlock:
    def test_shape(self):
        block = MSABlock(d_model=64, d_msa=16, h_msa=4, h_res=4, coevol_rank=4)
        S, N_prot, N = 4, 8, 8
        m = torch.randn(S, N_prot, 16)
        h_res = torch.randn(N, 64)
        protein_mask = torch.ones(N, dtype=torch.bool)
        pos_bias = torch.zeros(4, N_prot, N_prot)

        m_out, h_out, c_bar = block(
            m, h_res, protein_mask, pos_bias, training=False
        )

        assert m_out.shape == (S, N_prot, 16)
        assert h_out.shape == (N, 64)
        assert c_bar.shape == (N, 4)  # coevol_rank=4


class TestTokenUOTBlock:
    def test_shape(self):
        N, H = 10, 4
        block = TokenUOTBlock(d_model=64, n_heads=H, block_idx=0)
        # Override eps for smaller head count
        block.eps = torch.tensor([0.5, 1.0, 2.0, 4.0])

        h = torch.randn(N, 64)
        x_res = torch.randn(N, 3)
        mu = torch.ones(H, N) / N
        nu = torch.ones(H, N) / N

        chain_id = torch.zeros(N, dtype=torch.long)
        global_idx = torch.arange(N, dtype=torch.long)
        bond_matrix = torch.zeros(N, N, dtype=torch.bool)
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)
        h_out, x_out, log_u, log_v = block(
            h,
            x_res,
            mu,
            nu,
            None,
            None,
            pos_bins,
        )

        assert h_out.shape == (N, 64)
        assert x_out.shape == (N, 3)
        assert log_u.shape == (H, N)

    def test_egnn_equivariance(self):
        """EGNN coordinate update should be SE(3) equivariant (SPEC §8.1)."""
        torch.manual_seed(42)
        N, H = 8, 4
        block = TokenUOTBlock(d_model=64, n_heads=H, block_idx=0)
        block.eps = torch.tensor([0.5, 1.0, 2.0, 4.0])
        # Set nonzero gamma to test equivariance
        block.gamma.data = torch.randn(H) * 0.1

        h = torch.randn(N, 64)
        mu = torch.ones(H, N) / N
        nu = torch.ones(H, N) / N
        chain_id = torch.zeros(N, dtype=torch.long)
        global_idx = torch.arange(N, dtype=torch.long)
        bond_matrix = torch.zeros(N, N, dtype=torch.bool)
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)
        x_res = torch.randn(N, 3)

        # Forward pass 1: original
        _, x_out1, _, _ = block(
            h, x_res, mu, nu, None, None, pos_bins
        )

        # Forward pass 2: rotated input
        # Random rotation matrix
        q = torch.randn(3, 3)
        R, _ = torch.linalg.qr(q)
        if R.det() < 0:
            R[:, 0] *= -1
        t = torch.randn(3)  # translation

        x_rotated = x_res @ R.T + t
        _, x_out2, _, _ = block(
            h, x_rotated, mu, nu, None, None, pos_bins
        )

        # x_out2 should be x_out1 @ R.T + t
        x_expected = x_out1 @ R.T + t
        assert torch.allclose(x_out2, x_expected, atol=1e-4), (
            f"EGNN equivariance failed: max diff = {(x_out2 - x_expected).abs().max()}"
        )


class TestDiffusion:
    def test_timestep_embedding(self):
        fourier = FourierEmbedding(128)
        t = torch.tensor(0.5)
        emb = fourier(t)
        assert emb.shape == (128,)

    def test_edm_preconditioning(self):
        sigma = torch.tensor(1.0)
        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma)
        assert torch.isfinite(c_skip)
        assert torch.isfinite(c_out)

    def test_karras_schedule(self):
        sigmas = karras_schedule(200, torch.device("cpu"))
        assert sigmas.shape == (200,)
        assert sigmas[0] > sigmas[-1]  # decreasing

    def test_atom_block_shape(self):
        block = AtomBlock(d_atom=32, d_model=64, n_heads=2)
        N_atom, N = 20, 8
        q = torch.randn(N_atom, 32)
        c_atom = torch.randn(N_atom, 32)
        h_res = torch.randn(N, 64)
        p_lm = torch.randn(0, 16)
        p_lm_idx = torch.zeros(0, 2, dtype=torch.long)
        t_emb = torch.randn(32)
        token_idx = torch.arange(N).repeat_interleave(N_atom // N + 1)[:N_atom]

        out = block(q, c_atom, h_res, p_lm, p_lm_idx, t_emb, token_idx)
        assert out.shape == (N_atom, 32)


class TestLosses:
    def test_smooth_lddt(self):
        x_pred = torch.randn(20, 3, requires_grad=True)
        x_true = torch.randn(20, 3)
        loss = smooth_lddt(x_pred, x_true)
        assert 0 <= loss.item() <= 1
        loss.backward()
        assert x_pred.grad is not None

    def test_distogram_loss(self):
        disto = DistogramLoss(d_model=64, d_low=16, num_bins=39, tile_size=8)
        h = torch.randn(10, 64, requires_grad=True)
        x_true = torch.randn(10, 3)
        loss = disto(h, x_true)
        loss.backward()
        assert h.grad is not None

    def test_edm_loss(self):
        x_pred = torch.randn(20, 3)
        x_true = torch.randn(20, 3)
        sigma = torch.tensor(1.0)
        loss = edm_diffusion_loss(x_pred, x_true, sigma)
        assert loss.item() >= 0

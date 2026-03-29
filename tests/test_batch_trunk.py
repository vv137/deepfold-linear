"""Tests for batched MSA module, trunk block, and trunk orchestrator."""

import torch


class TestTokenOTBlockBatch:
    """Test batched TokenOTBlock."""

    def _make_block(self, d_model=64, n_heads=4):
        from deepfold.model.trunk_block import TokenOTBlock

        return TokenOTBlock(d_model=d_model, n_heads=n_heads, block_idx=0)

    def test_unbatched_compat(self):
        """B=1 batched should match unbatched output."""
        torch.manual_seed(42)
        d_model, n_heads, N = 64, 4, 8
        block = self._make_block(d_model, n_heads)
        block.eval()

        h = torch.randn(N, d_model)
        x_res = torch.randn(N, 3)
        pos_bins = torch.randint(0, 68, (N, N))

        # Unbatched
        with torch.no_grad():
            h_ub, x_ub = block(h, x_res, pos_bins)

        # Batched B=1
        with torch.no_grad():
            h_b, x_b = block(
                h.unsqueeze(0),
                x_res.unsqueeze(0),
                pos_bins.unsqueeze(0),
            )

        torch.testing.assert_close(h_ub, h_b.squeeze(0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x_ub, x_b.squeeze(0), atol=1e-4, rtol=1e-4)

    def test_batched_masked(self):
        """B=2 with mask: padded positions should have zero updates."""
        torch.manual_seed(123)
        d_model, n_heads = 64, 4
        B, N = 2, 10
        N_real = [10, 6]  # sample 0: all real, sample 1: 6 real

        block = self._make_block(d_model, n_heads)
        block.eval()

        h = torch.randn(B, N, d_model)
        x_res = torch.randn(B, N, 3)
        pos_bins = torch.randint(0, 68, (B, N, N))

        mask = torch.ones(B, N)
        mask[1, N_real[1] :] = 0  # pad positions 6-9 for sample 1

        with torch.no_grad():
            h_out, x_out = block(h, x_res, pos_bins, mask=mask)

        assert h_out.shape == (B, N, d_model)
        assert x_out.shape == (B, N, 3)

        # Padded x_res should not move
        torch.testing.assert_close(
            x_out[1, N_real[1]:], x_res[1, N_real[1]:], atol=1e-6, rtol=1e-6,
            msg="Padded x_res positions should not move",
        )

    def test_gradient_flows(self):
        """Verify gradients flow through batched block."""
        d_model, n_heads = 64, 4
        B, N = 2, 6
        block = self._make_block(d_model, n_heads)

        h = torch.randn(B, N, d_model, requires_grad=True)
        x_res = torch.randn(B, N, 3, requires_grad=True)
        pos_bins = torch.randint(0, 68, (B, N, N))

        h_out, x_out = block(h, x_res, pos_bins)
        loss = h_out.sum() + x_out.sum()
        loss.backward()

        assert h.grad is not None
        assert x_res.grad is not None
        assert h.grad.shape == h.shape


class TestMSABlockBatch:
    """Test batched MSABlock."""

    def _make_module(self, d_model=64, d_msa=16, h_msa=2, h_res=4):
        from deepfold.model.msa import MSABlock

        return MSABlock(
            d_model=d_model,
            d_msa=d_msa,
            h_msa=h_msa,
            h_res=h_res,
            coevol_rank=4,
            tile_size=4,
        )

    def test_unbatched_compat(self):
        """B=1 batched should match unbatched."""
        torch.manual_seed(77)
        d_model, d_msa, h_msa, h_res = 64, 16, 2, 4
        N, N_prot, S = 10, 6, 3

        block = self._make_module(d_model, d_msa, h_msa, h_res)
        block.eval()

        m = torch.randn(S, N_prot, d_msa)
        h = torch.randn(N, d_model)
        msa_token_mask = torch.zeros(N, dtype=torch.bool)
        msa_token_mask[:N_prot] = True

        from deepfold.model.position_encoding import PositionBias

        pb = PositionBias(h_msa, 68)
        msa_bins = torch.randint(0, 68, (N_prot, N_prot))
        pos_bias = pb(msa_bins)

        with torch.no_grad():
            m_ub, h_ub, c_bar_ub = block(m, h, msa_token_mask, pos_bias, training=False)

        with torch.no_grad():
            m_b, h_b, c_bar_b = block(
                m.unsqueeze(0), h.unsqueeze(0), msa_token_mask.unsqueeze(0),
                pos_bias.unsqueeze(0), training=False,
            )

        torch.testing.assert_close(m_ub, m_b.squeeze(0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(h_ub, h_b.squeeze(0), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(c_bar_ub, c_bar_b.squeeze(0), atol=1e-4, rtol=1e-4)


class TestTrunkBatch:
    """Integration test for batched Trunk."""

    def _make_inputs(self, B, N, N_atom, N_prot, S, device="cpu"):
        token_type = torch.randint(0, 4, (B, N), device=device)
        profile = torch.randn(B, N, 32, device=device)
        del_mean = torch.randn(B, N, 1, device=device)
        has_msa = torch.ones(B, N, 1, device=device)
        msa_feat = torch.randn(B, 1, S, N_prot, 34, device=device)
        c_atom = torch.randn(B, N_atom, 128, device=device)
        p_lm = torch.randn(B, N_atom, 16, device=device)
        p_lm_idx = torch.zeros(B, N_atom, 2, dtype=torch.long, device=device)
        token_idx = torch.arange(N_atom, device=device).unsqueeze(0).expand(B, -1) % N
        chain_id = torch.zeros(B, N, dtype=torch.long, device=device)
        global_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        bond_matrix = torch.zeros(B, N, N, dtype=torch.long, device=device)
        msa_token_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        msa_token_mask[:, :N_prot] = True
        return (
            token_type, profile, del_mean, has_msa, msa_feat,
            c_atom, p_lm, p_lm_idx, token_idx, chain_id, global_idx,
            bond_matrix, msa_token_mask,
        )

    def test_trunk_forward_batched(self):
        """Trunk forward with B=2 runs without error."""
        from deepfold.model.trunk import Trunk

        torch.manual_seed(42)
        B, N, N_atom, N_prot, S = 2, 8, 12, 4, 2
        trunk = Trunk(
            d_model=64, d_msa=16, d_atom=128, h_res=4, h_msa=2,
            n_msa_blocks=1, n_trunk_blocks=1, max_cycles=1, inference_cycles=1,
        )
        trunk.eval()
        inputs = self._make_inputs(B, N, N_atom, N_prot, S)

        with torch.no_grad():
            h_res, x_res = trunk(*inputs)

        assert h_res.shape == (B, N, 64)
        assert x_res.shape == (B, N, 3)


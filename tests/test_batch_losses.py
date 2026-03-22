"""Tests for batched diffusion module and loss functions."""

import torch

from deepfold.model.diffusion import AtomBlock, DiffusionModule
from deepfold.model.losses import (
    DistogramLoss,
    _atom_type_weights,
    edm_diffusion_loss,
    smooth_lddt,
)


def _make_atom_block_inputs(B, N, N_atom, n_pairs, d_atom=128, d_model=512, seed=42):
    """Create random inputs for AtomBlock testing."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, N_atom, d_atom, generator=g)
    c_atom = torch.randn(B, N_atom, d_atom, generator=g)
    h_cond = torch.randn(B, N, d_model, generator=g)
    p_lm = torch.randn(B, n_pairs, 16, generator=g)
    p_lm_idx = torch.stack(
        [torch.randint(0, N_atom, (B, n_pairs), generator=g) for _ in range(2)],
        dim=-1,
    )
    t_emb = torch.randn(B, d_atom, generator=g)
    token_idx = torch.randint(0, N, (B, N_atom), generator=g)
    atom_pad_mask = torch.ones(B, N_atom)
    pair_valid_mask = torch.ones(B, n_pairs)
    return q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx, atom_pad_mask, pair_valid_mask


class TestAtomBlockBatched:
    """Test AtomBlock with batch dimension."""

    def test_atom_block_batched_shape(self):
        B, N, N_atom, n_pairs = 2, 8, 20, 30
        block = AtomBlock(128, 512, 4)
        q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx, atom_pad_mask, pair_valid_mask = (
            _make_atom_block_inputs(B, N, N_atom, n_pairs)
        )
        out = block(q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx,
                     atom_pad_mask=atom_pad_mask, pair_valid_mask=pair_valid_mask)
        assert out.shape == (B, N_atom, 128)

    def test_atom_block_unbatched_compat(self):
        """B=1 batched should match unbatched for same input."""
        N, N_atom, n_pairs = 8, 20, 30
        block = AtomBlock(128, 512, 4)
        block.eval()

        g = torch.Generator().manual_seed(0)
        q = torch.randn(N_atom, 128, generator=g)
        c_atom = torch.randn(N_atom, 128, generator=g)
        h_cond = torch.randn(N, 512, generator=g)
        p_lm = torch.randn(n_pairs, 16, generator=g)
        p_lm_idx = torch.stack(
            [torch.randint(0, N_atom, (n_pairs,), generator=g) for _ in range(2)],
            dim=-1,
        )
        t_emb = torch.randn(128, generator=g)
        token_idx = torch.randint(0, N, (N_atom,), generator=g)

        with torch.no_grad():
            out_ub = block(q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx)

        assert out_ub.shape == (N_atom, 128)

    def test_atom_block_padding_isolation(self):
        """Padded atoms in one sample should not affect the other."""
        B, N, N_atom, n_pairs = 2, 4, 10, 5
        block = AtomBlock(128, 512, 4)
        block.eval()

        q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx, atom_pad_mask, pair_valid_mask = (
            _make_atom_block_inputs(B, N, N_atom, n_pairs, seed=7)
        )

        # Mask out last 3 atoms in sample 1
        atom_pad_mask[1, 7:] = 0

        with torch.no_grad():
            out = block(q, c_atom, h_cond, p_lm, p_lm_idx, t_emb, token_idx,
                        atom_pad_mask=atom_pad_mask, pair_valid_mask=pair_valid_mask)

        # Padded positions should be zero
        assert (out[1, 7:] == 0).all()


class TestDiffusionModuleBatched:
    """Test DiffusionModule with batch dimension."""

    def test_diffusion_batched_shape(self):
        B, N, N_atom, n_pairs = 2, 8, 20, 30
        mod = DiffusionModule(512, 128, n_atom_blocks=2)
        mod.eval()

        g = torch.Generator().manual_seed(42)
        h_res = torch.randn(B, N, 512, generator=g)
        c_atom = torch.randn(B, N_atom, 128, generator=g)
        p_lm = torch.randn(B, n_pairs, 16, generator=g)
        p_lm_idx = torch.stack(
            [torch.randint(0, N_atom, (B, n_pairs), generator=g) for _ in range(2)],
            dim=-1,
        )
        x_noisy = torch.randn(B, N_atom, 3, generator=g)
        sigma = torch.tensor([1.0, 2.0])
        token_idx = torch.randint(0, N, (B, N_atom), generator=g)
        atom_pad_mask = torch.ones(B, N_atom)

        with torch.no_grad():
            out = mod(h_res, c_atom, p_lm, p_lm_idx, x_noisy, sigma, token_idx,
                      atom_pad_mask=atom_pad_mask)

        assert out.shape == (B, N_atom, 3)

    def test_diffusion_unbatched_compat(self):
        """Unbatched path still works."""
        N, N_atom, n_pairs = 8, 20, 30
        mod = DiffusionModule(512, 128, n_atom_blocks=2)
        mod.eval()

        g = torch.Generator().manual_seed(42)
        h_res = torch.randn(N, 512, generator=g)
        c_atom = torch.randn(N_atom, 128, generator=g)
        p_lm = torch.randn(n_pairs, 16, generator=g)
        p_lm_idx = torch.stack(
            [torch.randint(0, N_atom, (n_pairs,), generator=g) for _ in range(2)],
            dim=-1,
        )
        x_noisy = torch.randn(N_atom, 3, generator=g)
        sigma = torch.tensor(1.0)
        token_idx = torch.randint(0, N, (N_atom,), generator=g)

        with torch.no_grad():
            out = mod(h_res, c_atom, p_lm, p_lm_idx, x_noisy, sigma, token_idx)

        assert out.shape == (N_atom, 3)

    def test_diffusion_scalar_sigma(self):
        """Scalar sigma should broadcast to batch."""
        B, N, N_atom, n_pairs = 2, 4, 10, 5
        mod = DiffusionModule(512, 128, n_atom_blocks=1)
        mod.eval()

        g = torch.Generator().manual_seed(0)
        h_res = torch.randn(B, N, 512, generator=g)
        c_atom = torch.randn(B, N_atom, 128, generator=g)
        p_lm = torch.randn(B, n_pairs, 16, generator=g)
        p_lm_idx = torch.stack(
            [torch.randint(0, N_atom, (B, n_pairs), generator=g) for _ in range(2)],
            dim=-1,
        )
        x_noisy = torch.randn(B, N_atom, 3, generator=g)
        sigma = torch.tensor(1.5)
        token_idx = torch.randint(0, N, (B, N_atom), generator=g)
        atom_pad_mask = torch.ones(B, N_atom)

        with torch.no_grad():
            out = mod(h_res, c_atom, p_lm, p_lm_idx, x_noisy, sigma, token_idx,
                      atom_pad_mask=atom_pad_mask)

        assert out.shape == (B, N_atom, 3)


class TestEdmLossBatched:
    """Test EDM diffusion loss with batch dimension."""

    def test_edm_loss_batched_shape(self):
        B, N_atom = 2, 20
        x_pred = torch.randn(B, N_atom, 3)
        x_true = torch.randn(B, N_atom, 3)
        sigma = torch.tensor([1.0, 2.0])
        loss = edm_diffusion_loss(x_pred, x_true, sigma)
        assert loss.dim() == 0  # scalar

    def test_edm_loss_per_sample_isolation(self):
        """Changing sample 1 should not affect sample 0's contribution."""
        B, N_atom = 2, 10
        x_pred = torch.randn(B, N_atom, 3)
        x_true = torch.randn(B, N_atom, 3)
        sigma = torch.tensor([1.0, 1.0])

        # Compute with original
        loss1 = edm_diffusion_loss(x_pred, x_true, sigma)

        # Modify sample 1 pred
        x_pred2 = x_pred.clone()
        x_pred2[1] = x_pred2[1] + 10.0
        loss2 = edm_diffusion_loss(x_pred2, x_true, sigma)

        # Loss should differ since sample 1 changed
        assert not torch.allclose(loss1, loss2)

    def test_edm_loss_unbatched_compat(self):
        """Unbatched should still work."""
        N_atom = 15
        x_pred = torch.randn(N_atom, 3)
        x_true = torch.randn(N_atom, 3)
        sigma = torch.tensor(1.0)
        loss = edm_diffusion_loss(x_pred, x_true, sigma)
        assert loss.dim() == 0

    def test_edm_loss_batched_with_masks(self):
        """Batched with resolved_mask and atom_weights."""
        B, N_atom = 2, 10
        x_pred = torch.randn(B, N_atom, 3)
        x_true = torch.randn(B, N_atom, 3)
        sigma = torch.tensor([1.0, 2.0])
        resolved_mask = torch.ones(B, N_atom)
        resolved_mask[0, 8:] = 0  # mask last 2 atoms in sample 0
        atom_weights = torch.ones(B, N_atom)
        loss = edm_diffusion_loss(x_pred, x_true, sigma, resolved_mask, atom_weights)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestSmoothLddtBatched:
    """Test smooth LDDT loss with batch dimension."""

    def test_smooth_lddt_batched_shape(self):
        B, M = 2, 15
        x_pred = torch.randn(B, M, 3)
        x_true = torch.randn(B, M, 3)
        loss = smooth_lddt(x_pred, x_true)
        assert loss.dim() == 0

    def test_smooth_lddt_batched_matches_unbatched(self):
        """Per-sample LDDT should match unbatched computation."""
        M = 10
        torch.manual_seed(42)
        x_pred_0 = torch.randn(M, 3)
        x_true_0 = torch.randn(M, 3)
        x_pred_1 = torch.randn(M, 3)
        x_true_1 = torch.randn(M, 3)

        loss_0 = smooth_lddt(x_pred_0, x_true_0)
        loss_1 = smooth_lddt(x_pred_1, x_true_1)
        expected = (loss_0 + loss_1) / 2.0

        x_pred_b = torch.stack([x_pred_0, x_pred_1], dim=0)
        x_true_b = torch.stack([x_true_0, x_true_1], dim=0)
        loss_b = smooth_lddt(x_pred_b, x_true_b)

        torch.testing.assert_close(loss_b, expected, atol=1e-5, rtol=1e-5)

    def test_smooth_lddt_batched_with_mask(self):
        B, M = 2, 12
        x_pred = torch.randn(B, M, 3)
        x_true = torch.randn(B, M, 3)
        resolved_mask = torch.ones(B, M)
        resolved_mask[1, 8:] = 0
        loss = smooth_lddt(x_pred, x_true, resolved_mask=resolved_mask)
        assert loss.dim() == 0

    def test_smooth_lddt_unbatched_compat(self):
        M = 10
        x_pred = torch.randn(M, 3)
        x_true = torch.randn(M, 3)
        loss = smooth_lddt(x_pred, x_true)
        assert loss.dim() == 0


class TestDistogramBatched:
    """Test DistogramLoss with batch dimension."""

    def test_distogram_batched_shape(self):
        B, N = 2, 16
        disto = DistogramLoss(d_model=64, d_low=16, tile_size=8)
        h_res = torch.randn(B, N, 64)
        x_true = torch.randn(B, N, 3)
        token_pad_mask = torch.ones(B, N)
        loss = disto(h_res, x_true, token_pad_mask=token_pad_mask)
        assert loss.dim() == 0

    def test_distogram_batched_with_padding(self):
        B, N = 2, 16
        disto = DistogramLoss(d_model=64, d_low=16, tile_size=8)
        h_res = torch.randn(B, N, 64)
        x_true = torch.randn(B, N, 3)
        token_pad_mask = torch.ones(B, N)
        token_pad_mask[1, 12:] = 0  # pad last 4 tokens in sample 1
        loss = disto(h_res, x_true, token_pad_mask=token_pad_mask)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_distogram_unbatched_compat(self):
        N = 16
        disto = DistogramLoss(d_model=64, d_low=16, tile_size=8)
        h_res = torch.randn(N, 64)
        x_true = torch.randn(N, 3)
        loss = disto(h_res, x_true)
        assert loss.dim() == 0


class TestAtomTypeWeightsBatched:
    """Test _atom_type_weights with batch dimension."""

    def test_batched_weights(self):
        from deepfold.data import const

        B, N, N_atom = 2, 5, 10
        token_idx = torch.randint(0, N, (B, N_atom))
        token_type = torch.zeros(B, N, dtype=torch.long)
        token_type[0, 0] = const.MOL_DNA
        token_type[1, 2] = const.MOL_NONPOLYMER

        w = _atom_type_weights(token_idx, token_type)
        assert w.shape == (B, N_atom)
        assert w.dtype == torch.float32

    def test_unbatched_weights(self):
        N, N_atom = 5, 10
        token_idx = torch.randint(0, N, (N_atom,))
        token_type = torch.zeros(N, dtype=torch.long)
        w = _atom_type_weights(token_idx, token_type)
        assert w.shape == (N_atom,)

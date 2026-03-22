"""Tests for batch dimension support in input embeddings and atom encoder."""

import torch

from deepfold.model.input_embedding import (
    TokenSingleEmbedding,
    MSAEmbedding,
    AtomSingleEmbedding,
    AtomPairEmbedding,
)
from deepfold.model.atom_encoder import AtomToTokenEncoder


class TestTokenSingleEmbedding:
    def test_unbatched(self):
        m = TokenSingleEmbedding(d_model=64)
        N = 5
        out = m(
            torch.randint(0, 4, (N,)),
            torch.randn(N, 32),
            torch.randn(N, 1),
            torch.randn(N, 1),
        )
        assert out.shape == (N, 64)

    def test_batched_shape(self):
        m = TokenSingleEmbedding(d_model=64)
        B, N = 3, 5
        out = m(
            torch.randint(0, 4, (B, N)),
            torch.randn(B, N, 32),
            torch.randn(B, N, 1),
            torch.randn(B, N, 1),
        )
        assert out.shape == (B, N, 64)

    def test_batched_matches_unbatched(self):
        m = TokenSingleEmbedding(d_model=64)
        B, N = 2, 5
        tt = torch.randint(0, 4, (B, N))
        prof = torch.randn(B, N, 32)
        dm = torch.randn(B, N, 1)
        hm = torch.randn(B, N, 1)

        batched = m(tt, prof, dm, hm)
        for b in range(B):
            single = m(tt[b], prof[b], dm[b], hm[b])
            assert torch.allclose(batched[b], single, atol=1e-6)


class TestMSAEmbedding:
    def test_unbatched(self):
        m = MSAEmbedding(d_msa=32)
        out = m(torch.randn(4, 8, 34))
        assert out.shape == (4, 8, 32)

    def test_batched(self):
        m = MSAEmbedding(d_msa=32)
        out = m(torch.randn(2, 4, 8, 34))
        assert out.shape == (2, 4, 8, 32)

    def test_batched_matches_unbatched(self):
        m = MSAEmbedding(d_msa=32)
        x = torch.randn(2, 4, 8, 34)
        batched = m(x)
        for b in range(2):
            single = m(x[b])
            assert torch.allclose(batched[b], single, atol=1e-6)


class TestAtomSingleEmbedding:
    def test_unbatched(self):
        m = AtomSingleEmbedding(d_ref=197, d_atom=64)
        out = m(torch.randn(10, 197))
        assert out.shape == (10, 64)

    def test_batched(self):
        m = AtomSingleEmbedding(d_ref=197, d_atom=64)
        out = m(torch.randn(3, 10, 197))
        assert out.shape == (3, 10, 64)

    def test_batched_matches_unbatched(self):
        m = AtomSingleEmbedding(d_ref=197, d_atom=64)
        x = torch.randn(2, 10, 197)
        batched = m(x)
        for b in range(2):
            single = m(x[b])
            assert torch.allclose(batched[b], single, atol=1e-6)


class TestAtomPairEmbedding:
    def test_unbatched(self):
        m = AtomPairEmbedding(d_pair=8)
        out = m(torch.randn(20, 3), torch.randn(20, 1))
        assert out.shape == (20, 8)

    def test_batched(self):
        m = AtomPairEmbedding(d_pair=8)
        out = m(torch.randn(2, 20, 3), torch.randn(2, 20, 1))
        assert out.shape == (2, 20, 8)

    def test_batched_matches_unbatched(self):
        m = AtomPairEmbedding(d_pair=8)
        d = torch.randn(2, 20, 3)
        v = torch.randn(2, 20, 1)
        batched = m(d, v)
        for b in range(2):
            single = m(d[b], v[b])
            assert torch.allclose(batched[b], single, atol=1e-6)


class TestAtomToTokenEncoder:
    def _make_unbatched_inputs(self, n_tokens=4, atoms_per_token=3, n_pairs=8):
        """Helper to create consistent unbatched test inputs."""
        N_atom = n_tokens * atoms_per_token
        d_atom = 128
        d_pair = 16

        c_atom = torch.randn(N_atom, d_atom)
        # token_idx: each group of atoms_per_token atoms belongs to one token
        token_idx = torch.arange(n_tokens).repeat_interleave(atoms_per_token)

        # Pair indices: random intra-token pairs
        src = torch.randint(0, N_atom, (n_pairs,))
        dst = torch.randint(0, N_atom, (n_pairs,))
        p_lm_idx = torch.stack([src, dst], dim=-1)
        p_lm = torch.randn(n_pairs, d_pair)

        return c_atom, p_lm, p_lm_idx, token_idx, n_tokens

    def test_unbatched(self):
        m = AtomToTokenEncoder(d_atom=128, d_model=64, n_heads=4)
        c_atom, p_lm, p_lm_idx, token_idx, n_tokens = self._make_unbatched_inputs()
        out = m(c_atom, p_lm, p_lm_idx, token_idx, n_tokens)
        assert out.shape == (n_tokens, 64)

    def test_batched_b1_matches_unbatched(self):
        """B=1 batched output matches unbatched output."""
        m = AtomToTokenEncoder(d_atom=128, d_model=64, n_heads=4)
        c_atom, p_lm, p_lm_idx, token_idx, n_tokens = self._make_unbatched_inputs()

        unbatched_out = m(c_atom, p_lm, p_lm_idx, token_idx, n_tokens)

        # Wrap as batch of 1
        batched_out = m(
            c_atom.unsqueeze(0),
            p_lm.unsqueeze(0),
            p_lm_idx.unsqueeze(0),
            token_idx.unsqueeze(0),
            n_tokens,
        )
        assert batched_out.shape == (1, n_tokens, 64)
        assert torch.allclose(batched_out.squeeze(0), unbatched_out, atol=1e-5)

    def test_batched_b2_shape(self):
        """B=2 produces correct output shape."""
        m = AtomToTokenEncoder(d_atom=128, d_model=64, n_heads=4)
        B, N_atom, n_tokens, n_pairs = 2, 12, 4, 8

        c_atom = torch.randn(B, N_atom, 128)
        token_idx = torch.arange(n_tokens).repeat_interleave(3).unsqueeze(0).expand(B, -1)
        p_lm = torch.randn(B, n_pairs, 16)
        src = torch.randint(0, N_atom, (B, n_pairs))
        dst = torch.randint(0, N_atom, (B, n_pairs))
        p_lm_idx = torch.stack([src, dst], dim=-1)

        out = m(c_atom, p_lm, p_lm_idx, token_idx, n_tokens)
        assert out.shape == (B, n_tokens, 64)

    def test_batched_with_padding(self):
        """B=2 with padding masks, verify padded positions are zero."""
        m = AtomToTokenEncoder(d_atom=128, d_model=64, n_heads=4)
        B, N_atom, n_tokens, n_pairs = 2, 12, 4, 10

        c_atom = torch.randn(B, N_atom, 128)
        token_idx = torch.arange(n_tokens).repeat_interleave(3).unsqueeze(0).expand(B, -1).clone()

        p_lm = torch.randn(B, n_pairs, 16)
        src = torch.randint(0, N_atom, (B, n_pairs))
        dst = torch.randint(0, N_atom, (B, n_pairs))
        p_lm_idx = torch.stack([src, dst], dim=-1)

        # Sample 1 has all atoms real, sample 2 has last 3 atoms padded
        atom_pad_mask = torch.ones(B, N_atom, dtype=torch.bool)
        atom_pad_mask[1, 9:] = False

        pair_pad_mask = torch.ones(B, n_pairs, dtype=torch.bool)
        pair_pad_mask[1, 7:] = False

        # Sample 2 has last token padded
        token_pad_mask = torch.ones(B, n_tokens, dtype=torch.bool)
        token_pad_mask[1, 3] = False

        out = m(
            c_atom, p_lm, p_lm_idx, token_idx, n_tokens,
            atom_pad_mask=atom_pad_mask,
            pair_pad_mask=pair_pad_mask,
            token_pad_mask=token_pad_mask,
        )
        assert out.shape == (B, n_tokens, 64)
        # Padded token positions should be zero
        assert torch.allclose(out[1, 3], torch.zeros(64), atol=1e-7)
        # Non-padded positions should be non-zero (with high probability)
        assert out[0, 0].abs().sum() > 0

    def test_batched_per_sample_matches(self):
        """Batched output matches looped per-sample without padding."""
        torch.manual_seed(42)
        m = AtomToTokenEncoder(d_atom=128, d_model=64, n_heads=4)
        m.eval()  # avoid dropout if any

        B, N_atom, n_tokens, n_pairs = 2, 12, 4, 6

        c_atom = torch.randn(B, N_atom, 128)
        token_idx = torch.arange(n_tokens).repeat_interleave(3).unsqueeze(0).expand(B, -1)
        p_lm = torch.randn(B, n_pairs, 16)
        src = torch.randint(0, N_atom, (B, n_pairs))
        dst = torch.randint(0, N_atom, (B, n_pairs))
        p_lm_idx = torch.stack([src, dst], dim=-1)

        with torch.no_grad():
            batched = m(c_atom, p_lm, p_lm_idx, token_idx, n_tokens)
            for b in range(B):
                single = m(c_atom[b], p_lm[b], p_lm_idx[b], token_idx[b], n_tokens)
                assert torch.allclose(batched[b], single, atol=1e-5), (
                    f"Mismatch at batch {b}: max diff = {(batched[b] - single).abs().max()}"
                )

    def test_no_pairs(self):
        """Works with zero atom pairs (both batched and unbatched)."""
        m = AtomToTokenEncoder(d_atom=128, d_model=64, n_heads=4)
        N_atom, n_tokens = 6, 2
        c_atom = torch.randn(N_atom, 128)
        token_idx = torch.arange(n_tokens).repeat_interleave(3)
        p_lm = torch.zeros(0, 16)
        p_lm_idx = torch.zeros(0, 2, dtype=torch.long)

        out = m(c_atom, p_lm, p_lm_idx, token_idx, n_tokens)
        assert out.shape == (n_tokens, 64)

        # Batched with zero pairs
        out_b = m(
            c_atom.unsqueeze(0),
            p_lm.unsqueeze(0),
            p_lm_idx.unsqueeze(0),
            token_idx.unsqueeze(0),
            n_tokens,
        )
        assert out_b.shape == (1, n_tokens, 64)

    def test_gradient_flows_batched(self):
        """Gradients flow through batched atom encoder."""
        m = AtomToTokenEncoder(d_atom=128, d_model=64, n_heads=4)
        B, N_atom, n_tokens, n_pairs = 2, 6, 2, 4

        c_atom = torch.randn(B, N_atom, 128, requires_grad=True)
        token_idx = torch.arange(n_tokens).repeat_interleave(3).unsqueeze(0).expand(B, -1)
        p_lm = torch.randn(B, n_pairs, 16)
        src = torch.randint(0, N_atom, (B, n_pairs))
        dst = torch.randint(0, N_atom, (B, n_pairs))
        p_lm_idx = torch.stack([src, dst], dim=-1)

        out = m(c_atom, p_lm, p_lm_idx, token_idx, n_tokens)
        out.sum().backward()
        assert c_atom.grad is not None
        assert c_atom.grad.shape == (B, N_atom, 128)

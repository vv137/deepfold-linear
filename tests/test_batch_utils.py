"""Tests for batch dimension support in utility modules."""

import torch

from deepfold.utils.scatter import scatter_mean
from deepfold.model.position_encoding import compute_bins, PositionBias
from deepfold.model.primitives import SwiGLU, LNLinear


class TestScatterMeanBatched:
    def test_matches_per_sample(self):
        """Batched scatter_mean matches looped per-sample results."""
        B, M, D, S = 2, 8, 16, 4
        values = torch.randn(B, M, D)
        indices = torch.randint(0, S, (B, M))

        batched = scatter_mean(values, indices, S)
        assert batched.shape == (B, S, D)

        for b in range(B):
            single = scatter_mean(values[b], indices[b], S)
            assert torch.allclose(batched[b], single, atol=1e-6)

    def test_backward_compat_2d(self):
        """2D input still works as before."""
        M, D, S = 10, 8, 3
        values = torch.randn(M, D)
        indices = torch.randint(0, S, (M,))
        result = scatter_mean(values, indices, S)
        assert result.shape == (S, D)

    def test_gradient_flows(self):
        """Gradients flow through batched scatter_mean."""
        B, M, D, S = 2, 6, 4, 3
        values = torch.randn(B, M, D, requires_grad=True)
        indices = torch.randint(0, S, (B, M))
        result = scatter_mean(values, indices, S)
        result.sum().backward()
        assert values.grad is not None
        assert values.grad.shape == (B, M, D)


class TestComputeBinsBatched:
    def test_matches_per_sample(self):
        """Batched compute_bins matches per-sample results."""
        B, N = 2, 6
        chain_id = torch.zeros(B, N, dtype=torch.long)
        chain_id[1, 3:] = 1  # second sample has two chains
        global_idx = torch.arange(N, dtype=torch.long).unsqueeze(0).expand(B, -1)
        bond_matrix = torch.zeros(B, N, N, dtype=torch.bool)
        bond_matrix[0, 0, 1] = True
        bond_matrix[0, 1, 0] = True

        batched = compute_bins(chain_id, global_idx, bond_matrix)
        assert batched.shape == (B, N, N)

        for b in range(B):
            single = compute_bins(chain_id[b], global_idx[b], bond_matrix[b])
            assert torch.equal(batched[b], single)

    def test_backward_compat_1d(self):
        """1D input still works as before."""
        N = 5
        chain_id = torch.zeros(N, dtype=torch.long)
        global_idx = torch.arange(N, dtype=torch.long)
        bond_matrix = torch.zeros(N, N, dtype=torch.bool)
        bins = compute_bins(chain_id, global_idx, bond_matrix)
        assert bins.shape == (N, N)

    def test_cross_chain_batched(self):
        """Cross-chain pairs get bin 65 in batched mode."""
        B, N = 1, 4
        chain_id = torch.tensor([[0, 0, 1, 1]])
        global_idx = torch.arange(N).unsqueeze(0)
        bond_matrix = torch.zeros(B, N, N, dtype=torch.bool)
        bins = compute_bins(chain_id, global_idx, bond_matrix)
        assert bins[0, 0, 2] == 65
        assert bins[0, 1, 3] == 65


class TestPositionBiasBatched:
    def test_unbatched_shape(self):
        """Unbatched input produces (H, N, N)."""
        H, N = 8, 10
        pb = PositionBias(H, 68)
        bins = torch.randint(0, 68, (N, N))
        out = pb(bins)
        assert out.shape == (H, N, N)

    def test_batched_shape(self):
        """Batched input produces (B, H, N, N)."""
        B, H, N = 3, 8, 10
        pb = PositionBias(H, 68)
        bins = torch.randint(0, 68, (B, N, N))
        out = pb(bins)
        assert out.shape == (B, H, N, N)

    def test_batched_matches_unbatched(self):
        """Batched output matches per-sample unbatched output."""
        B, H, N = 2, 4, 6
        pb = PositionBias(H, 68)
        bins = torch.randint(0, 68, (B, N, N))
        batched = pb(bins)
        for b in range(B):
            single = pb(bins[b])
            assert torch.allclose(batched[b], single)


class TestPrimitivesBatchCompat:
    def test_swiglu_batched(self):
        """SwiGLU handles (B, N, d) input."""
        B, N, d_in, d_h, d_out = 2, 10, 64, 128, 64
        m = SwiGLU(d_in, d_h, d_out)
        x = torch.randn(B, N, d_in)
        out = m(x)
        assert out.shape == (B, N, d_out)

    def test_lnlinear_batched(self):
        """LNLinear handles (B, N, d) input."""
        B, N, d_in, d_out = 2, 10, 512, 64
        m = LNLinear(d_in, d_out)
        x = torch.randn(B, N, d_in)
        out = m(x)
        assert out.shape == (B, N, d_out)

    def test_swiglu_2d_still_works(self):
        """SwiGLU still works with 2D (N, d) input."""
        m = SwiGLU(32, 64, 32)
        x = torch.randn(5, 32)
        assert m(x).shape == (5, 32)

    def test_lnlinear_2d_still_works(self):
        """LNLinear still works with 2D (N, d) input."""
        m = LNLinear(64, 32)
        x = torch.randn(5, 64)
        assert m(x).shape == (5, 32)

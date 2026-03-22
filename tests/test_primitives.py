"""Unit tests for core primitives and modules."""

import torch

from deepfold.model.primitives import SwiGLU, LNLinear, zero_init_linear
from deepfold.model.position_encoding import compute_bins, PositionBias
from deepfold.model.input_embedding import TokenSingleEmbedding, MSAEmbedding
from deepfold.model.atom_encoder import AtomToTokenEncoder
from deepfold.model.sinkhorn import sinkhorn_solve, compute_transport_output


class TestSwiGLU:
    def test_shape(self):
        m = SwiGLU(64, 256, 64)
        x = torch.randn(10, 64)
        assert m(x).shape == (10, 64)

    def test_grad(self):
        m = SwiGLU(32, 128, 32)
        x = torch.randn(5, 32, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


class TestLNLinear:
    def test_shape(self):
        m = LNLinear(512, 64)
        x = torch.randn(10, 512)
        assert m(x).shape == (10, 64)


class TestZeroInit:
    def test_zeros(self):
        lin = zero_init_linear(64, 32)
        assert (lin.weight == 0).all()
        assert (lin.bias == 0).all()


class TestPositionEncoding:
    def test_bins_same_chain(self):
        N = 10
        chain_id = torch.zeros(N, dtype=torch.long)
        global_idx = torch.arange(N, dtype=torch.long)
        bond_matrix = torch.zeros(N, N, dtype=torch.bool)

        bins = compute_bins(chain_id, global_idx, bond_matrix)

        # Diagonal: sep=0 -> bin 32
        assert (bins.diag() == 32).all()
        # bins[i, i+1]: sep = g_i - g_{i+1} = -1 -> bin 31
        # bins[i+1, i]: sep = g_{i+1} - g_i = +1 -> bin 33
        for i in range(N - 1):
            assert bins[i, i + 1] == 31
            assert bins[i + 1, i] == 33

    def test_bins_cross_chain(self):
        chain_id = torch.tensor([0, 0, 1, 1])
        global_idx = torch.arange(4, dtype=torch.long)
        bond_matrix = torch.zeros(4, 4, dtype=torch.bool)

        bins = compute_bins(chain_id, global_idx, bond_matrix)
        assert bins[0, 2] == 65  # cross-chain
        assert bins[1, 3] == 65

    def test_bins_covalent(self):
        chain_id = torch.zeros(3, dtype=torch.long)
        global_idx = torch.arange(3, dtype=torch.long)
        bond_matrix = torch.zeros(3, 3, dtype=torch.bool)
        bond_matrix[0, 1] = True
        bond_matrix[1, 0] = True

        bins = compute_bins(chain_id, global_idx, bond_matrix)
        assert bins[0, 1] == 66  # covalent same-chain
        assert bins[1, 0] == 66

    def test_position_bias_shape(self):
        pb = PositionBias(16, 68)
        bins = torch.randint(0, 68, (20, 20))
        out = pb(bins)
        assert out.shape == (16, 20, 20)


class TestTokenEmbedding:
    def test_shape(self):
        emb = TokenSingleEmbedding(512)
        N = 15
        out = emb(
            torch.zeros(N, dtype=torch.long),
            torch.randn(N, 32),
            torch.zeros(N, 1),
            torch.ones(N, 1),
        )
        assert out.shape == (N, 512)


class TestMSAEmbedding:
    def test_shape(self):
        emb = MSAEmbedding(64)
        out = emb(torch.randn(8, 20, 34))
        assert out.shape == (8, 20, 64)


class TestAtomEncoder:
    def test_shape(self):
        enc = AtomToTokenEncoder(128, 512)
        N_atom = 30
        N = 10
        c_atom = torch.randn(N_atom, 128)
        p_lm = torch.randn(0, 16)
        p_lm_idx = torch.zeros(0, 2, dtype=torch.long)
        token_idx = torch.arange(N).repeat_interleave(3)
        out = enc(c_atom, p_lm, p_lm_idx, token_idx, N)
        assert out.shape == (N, 512)


class TestSinkhorn:
    def test_basic_convergence(self):
        H, N = 4, 10
        C = torch.randn(H, N, N)
        log_mu = torch.log_softmax(torch.randn(H, N), dim=-1)
        log_nu = torch.log_softmax(torch.randn(H, N), dim=-1)
        eps = torch.tensor([0.5, 1.0, 2.0, 4.0])

        log_u, log_v = sinkhorn_solve(C, log_mu, log_nu, eps, lam=1.0, K=20)

        # Check that dual variables are finite
        assert torch.isfinite(log_u).all()
        assert torch.isfinite(log_v).all()

    def test_transport_output_shape(self):
        H, N, d_h = 4, 10, 32
        V = torch.randn(H, N, d_h)
        G = torch.randn(H, N, d_h)
        log_u = torch.zeros(H, N)
        log_v = torch.zeros(H, N)
        C = torch.randn(H, N, N)
        eps = torch.tensor([0.5, 1.0, 2.0, 4.0])
        x_res = torch.randn(N, 3)

        o, T_norm, x_centroid = compute_transport_output(
            V, G, log_u, log_v, C, eps, x_res
        )

        assert o.shape == (N, H * d_h)
        assert T_norm.shape == (H, N, N)
        assert x_centroid.shape == (H, N, 3)

        # T_norm rows should sum to ~1
        row_sums = T_norm.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01)

    def test_unrolled_gradient(self):
        """Verify unrolled gradients flow through sinkhorn_solve (SPEC v4.4)."""
        H, N = 2, 5
        C = torch.randn(H, N, N, requires_grad=True)
        log_mu = torch.log_softmax(torch.randn(H, N), dim=-1)
        log_nu = torch.log_softmax(torch.randn(H, N), dim=-1)
        eps = torch.tensor([1.0, 2.0])

        log_u, log_v = sinkhorn_solve(C, log_mu, log_nu, eps, K=7)
        loss = (log_u**2 + log_v**2).sum()
        loss.backward()
        assert C.grad is not None
        assert torch.isfinite(C.grad).all()

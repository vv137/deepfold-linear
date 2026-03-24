"""Tests for cross-attention Triton kernels (atom↔token).

Compares Triton kernels against Python references for:
  - AtomToToken (sparse): tokens query their own atoms
  - TokenToAtom (dense): atoms query all tokens
"""

import pytest
import torch

from deepfold.model.kernels.cross_attn_kernel import (
    atom_to_token_attn,
    atom_to_token_ref,
    token_to_atom_attn,
    token_to_atom_ref,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"


def _make_token_atom_mapping(B, N, M_approx, device):
    """Create token_atom_starts/counts for ~M_approx atoms across N tokens."""
    # Roughly uniform: each token gets M_approx // N atoms
    atoms_per_token = max(1, M_approx // N)
    counts = torch.full((B, N), atoms_per_token, device=device, dtype=torch.int32)
    # Last token gets remainder
    M = atoms_per_token * N
    starts = torch.zeros(B, N, device=device, dtype=torch.int32)
    for b in range(B):
        starts[b] = torch.arange(N, device=device) * atoms_per_token
    return starts, counts, M


# ============================================================================
# AtomToToken tests
# ============================================================================


class TestAtomToTokenForward:

    @pytest.mark.parametrize("B,H,N,atoms_per_tok,D", [
        (1, 4, 32, 5, 32),
        (2, 4, 64, 3, 32),
        (1, 4, 128, 1, 32),   # ligand-like: 1 atom per token
        (2, 4, 16, 14, 32),   # protein-like: ~14 atoms per residue
    ])
    def test_forward_matches_ref(self, B, H, N, atoms_per_tok, D):
        M = N * atoms_per_tok
        starts = torch.arange(N, device=DEVICE).unsqueeze(0).expand(B, -1) * atoms_per_tok
        starts = starts.to(torch.int32)
        counts = torch.full((B, N), atoms_per_tok, device=DEVICE, dtype=torch.int32)
        token_mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)

        Q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16)
        K = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16)
        V = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16)

        with torch.no_grad():
            out_tri = atom_to_token_attn(Q, K, V, starts, counts, token_mask)
            out_ref = atom_to_token_ref(Q.float(), K.float(), V.float(),
                                        starts, counts, token_mask)

        torch.testing.assert_close(
            out_tri.float(), out_ref.float(), atol=5e-2, rtol=5e-2
        )

    def test_forward_with_padding(self):
        B, H, N, D = 2, 4, 32, 32
        atoms_per_tok = 5
        M = N * atoms_per_tok
        starts = torch.arange(N, device=DEVICE).unsqueeze(0).expand(B, -1) * atoms_per_tok
        starts = starts.to(torch.int32)
        counts = torch.full((B, N), atoms_per_tok, device=DEVICE, dtype=torch.int32)
        token_mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)
        token_mask[:, -8:] = 0.0  # pad last 8 tokens

        Q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16)
        K = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16)
        V = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16)

        with torch.no_grad():
            out_tri = atom_to_token_attn(Q, K, V, starts, counts, token_mask)

        # Padded tokens should have zero output
        assert (out_tri[:, :, -8:, :] == 0).all()


class TestAtomToTokenBackward:

    def test_backward_matches_ref(self):
        B, H, N, D = 1, 4, 32, 32
        atoms_per_tok = 5
        M = N * atoms_per_tok

        starts = (torch.arange(N, device=DEVICE) * atoms_per_tok).unsqueeze(0).to(torch.int32)
        counts = torch.full((1, N), atoms_per_tok, device=DEVICE, dtype=torch.int32)
        token_mask = torch.ones(1, N, device=DEVICE, dtype=torch.float32)

        Q_ref = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
        K_ref = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
        V_ref = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32, requires_grad=True)

        out_ref = atom_to_token_ref(Q_ref, K_ref, V_ref, starts, counts, token_mask)
        out_ref.sum().backward()

        Q_tri = Q_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)
        K_tri = K_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)
        V_tri = V_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)

        out_tri = atom_to_token_attn(Q_tri, K_tri, V_tri, starts, counts, token_mask)
        out_tri.float().sum().backward()

        torch.testing.assert_close(Q_tri.grad.float(), Q_ref.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(K_tri.grad.float(), K_ref.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(V_tri.grad.float(), V_ref.grad, atol=5e-2, rtol=5e-2)

    def test_grad_flows(self):
        B, H, N, D = 1, 4, 16, 32
        atoms_per_tok = 5
        M = N * atoms_per_tok

        starts = (torch.arange(N, device=DEVICE) * atoms_per_tok).unsqueeze(0).to(torch.int32)
        counts = torch.full((1, N), atoms_per_tok, device=DEVICE, dtype=torch.int32)
        token_mask = torch.ones(1, N, device=DEVICE, dtype=torch.float32)

        Q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        K = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        V = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

        out = atom_to_token_attn(Q, K, V, starts, counts, token_mask)
        out.float().sum().backward()

        assert Q.grad is not None and Q.grad.abs().sum() > 0
        assert K.grad is not None and K.grad.abs().sum() > 0
        assert V.grad is not None and V.grad.abs().sum() > 0


# ============================================================================
# TokenToAtom tests
# ============================================================================


class TestTokenToAtomForward:

    @pytest.mark.parametrize("B,H,M,N,D", [
        (1, 4, 64, 32, 32),
        (2, 4, 128, 64, 32),
        (1, 4, 256, 128, 32),
        (2, 4, 97, 43, 32),   # non-power-of-2
    ])
    def test_forward_matches_ref(self, B, H, M, N, D):
        Q = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16)
        K = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16)
        V = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16)
        atom_mask = torch.ones(B, M, device=DEVICE, dtype=torch.float32)
        token_mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            out_tri = token_to_atom_attn(Q, K, V, atom_mask, token_mask)
            out_ref = token_to_atom_ref(Q.float(), K.float(), V.float(),
                                        atom_mask, token_mask)

        torch.testing.assert_close(
            out_tri.float(), out_ref.float(), atol=5e-2, rtol=5e-2
        )

    def test_forward_with_padding(self):
        B, H, M, N, D = 2, 4, 128, 64, 32
        Q = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16)
        K = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16)
        V = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16)
        atom_mask = torch.ones(B, M, device=DEVICE, dtype=torch.float32)
        atom_mask[:, -20:] = 0.0
        token_mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)
        token_mask[:, -10:] = 0.0

        with torch.no_grad():
            out_tri = token_to_atom_attn(Q, K, V, atom_mask, token_mask)

        # Padded atoms should be zero
        assert (out_tri[:, :, -20:, :] == 0).all()


class TestTokenToAtomBackward:

    def test_backward_matches_ref(self):
        B, H, M, N, D = 1, 4, 64, 32, 32

        Q_ref = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
        K_ref = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
        V_ref = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32, requires_grad=True)
        atom_mask = torch.ones(B, M, device=DEVICE, dtype=torch.float32)
        token_mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)

        out_ref = token_to_atom_ref(Q_ref, K_ref, V_ref, atom_mask, token_mask)
        out_ref.sum().backward()

        Q_tri = Q_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)
        K_tri = K_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)
        V_tri = V_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)

        out_tri = token_to_atom_attn(Q_tri, K_tri, V_tri, atom_mask, token_mask)
        out_tri.float().sum().backward()

        torch.testing.assert_close(Q_tri.grad.float(), Q_ref.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(K_tri.grad.float(), K_ref.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(V_tri.grad.float(), V_ref.grad, atol=5e-2, rtol=5e-2)

    def test_grad_flows(self):
        B, H, M, N, D = 1, 4, 64, 32, 32

        Q = torch.randn(B, H, M, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        K = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        V = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        atom_mask = torch.ones(B, M, device=DEVICE, dtype=torch.float32)
        token_mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)

        out = token_to_atom_attn(Q, K, V, atom_mask, token_mask)
        out.float().sum().backward()

        assert Q.grad is not None and Q.grad.abs().sum() > 0
        assert K.grad is not None and K.grad.abs().sum() > 0
        assert V.grad is not None and V.grad.abs().sum() > 0

"""Tests for flash windowed atom attention Triton kernel.

Compares Triton kernel against Python reference for forward and backward.
"""

import pytest
import torch

from deepfold.model.kernels.flash_atom_attn import (
    flash_atom_attn,
    flash_atom_attn_ref,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"


def _random_inputs(B, H, M, D, dtype=torch.bfloat16, pad_frac=0.0):
    Q = torch.randn(B, H, M, D, device=DEVICE, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, M, D, device=DEVICE, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, M, D, device=DEVICE, dtype=dtype, requires_grad=True)
    mask = torch.ones(B, M, device=DEVICE, dtype=torch.float32)
    if pad_frac > 0:
        pad_start = int(M * (1 - pad_frac))
        mask[:, pad_start:] = 0.0
    return Q, K, V, mask


class TestFlashAtomAttnForward:

    @pytest.mark.parametrize("B,H,M,D", [
        (1, 4, 64, 32),
        (2, 4, 128, 32),
        (1, 4, 256, 32),
        (2, 4, 97, 32),     # non-multiple of W=32
        (1, 4, 512, 64),
    ])
    def test_forward_matches_ref(self, B, H, M, D):
        Q, K, V, mask = _random_inputs(B, H, M, D)

        with torch.no_grad():
            out_tri = flash_atom_attn(Q, K, V, mask)
            out_ref = flash_atom_attn_ref(Q.float(), K.float(), V.float(), mask)

        torch.testing.assert_close(
            out_tri.float(), out_ref.float(), atol=1e-2, rtol=1e-2
        )

    def test_forward_with_padding(self):
        B, H, M, D = 2, 4, 256, 32
        Q, K, V, mask = _random_inputs(B, H, M, D, pad_frac=0.2)

        with torch.no_grad():
            out_tri = flash_atom_attn(Q, K, V, mask)
            out_ref = flash_atom_attn_ref(Q.float(), K.float(), V.float(), mask)

        # Padded positions should be zero
        for b in range(B):
            pad_idx = (mask[b] == 0).nonzero(as_tuple=True)[0]
            if len(pad_idx) > 0:
                assert (out_tri[b, :, pad_idx[0]:, :] == 0).all()

        torch.testing.assert_close(
            out_tri.float(), out_ref.float(), atol=1e-2, rtol=1e-2
        )

    def test_unbatched(self):
        H, M, D = 4, 128, 32
        Q = torch.randn(H, M, D, device=DEVICE, dtype=torch.bfloat16)
        K = torch.randn(H, M, D, device=DEVICE, dtype=torch.bfloat16)
        V = torch.randn(H, M, D, device=DEVICE, dtype=torch.bfloat16)

        with torch.no_grad():
            out = flash_atom_attn(Q, K, V)

        assert out.shape == (H, M, D)


class TestFlashAtomAttnBackward:

    def _check_grads(self, B, H, M, D, pad_frac=0.0):
        Q, K, V, mask = _random_inputs(B, H, M, D, dtype=torch.float32, pad_frac=pad_frac)

        # Reference
        Q_ref = Q.detach().clone().requires_grad_(True)
        K_ref = K.detach().clone().requires_grad_(True)
        V_ref = V.detach().clone().requires_grad_(True)
        out_ref = flash_atom_attn_ref(Q_ref, K_ref, V_ref, mask)
        out_ref.sum().backward()

        # Triton
        Q_tri = Q.detach().clone().to(torch.bfloat16).requires_grad_(True)
        K_tri = K.detach().clone().to(torch.bfloat16).requires_grad_(True)
        V_tri = V.detach().clone().to(torch.bfloat16).requires_grad_(True)
        out_tri = flash_atom_attn(Q_tri, K_tri, V_tri, mask)
        out_tri.float().sum().backward()

        torch.testing.assert_close(Q_tri.grad.float(), Q_ref.grad.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(K_tri.grad.float(), K_ref.grad.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(V_tri.grad.float(), V_ref.grad.float(), atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("B,H,M,D", [
        (1, 4, 64, 32),
        (2, 4, 128, 32),
        (1, 4, 256, 32),
    ])
    def test_backward_matches_ref(self, B, H, M, D):
        self._check_grads(B, H, M, D)

    def test_backward_with_padding(self):
        self._check_grads(2, 4, 256, 32, pad_frac=0.2)

    def test_grad_flows(self):
        B, H, M, D = 1, 4, 128, 32
        Q, K, V, mask = _random_inputs(B, H, M, D, dtype=torch.bfloat16)
        out = flash_atom_attn(Q, K, V, mask)
        out.float().sum().backward()

        assert Q.grad is not None and Q.grad.abs().sum() > 0
        assert K.grad is not None and K.grad.abs().sum() > 0
        assert V.grad is not None and V.grad.abs().sum() > 0

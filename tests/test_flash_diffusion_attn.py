"""Tests for flash diffusion attention Triton kernel.

Compares Triton kernel output against Python reference for:
  - Forward: output values match
  - Backward: gradients (dQ, dK, dV, dW_pos) match
  - Edge cases: padding masks, single-element batches, unbatched mode
"""

import pytest
import torch

from deepfold.model.kernels.flash_diffusion_attn import (
    flash_diffusion_attn,
    flash_diff_attn_ref,
)

# Skip all tests if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)

DEVICE = "cuda"
NUM_BINS = 68


def _random_inputs(B, H, N, D, dtype=torch.bfloat16, with_mask=False):
    """Generate random test inputs."""
    Q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=True)
    pos_weight = torch.randn(H, NUM_BINS, device=DEVICE, dtype=torch.float32, requires_grad=True)
    pos_bins = torch.randint(0, NUM_BINS, (B, N, N), device=DEVICE, dtype=torch.int32)

    if with_mask:
        # Create mask with some padded positions at the end
        mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)
        for b in range(B):
            pad_start = N - torch.randint(0, N // 4 + 1, (1,)).item()
            mask[b, pad_start:] = 0.0
    else:
        mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)

    return Q, K, V, pos_weight, pos_bins, mask


class TestFlashDiffusionAttnForward:
    """Forward pass: Triton kernel vs Python reference."""

    @pytest.mark.parametrize("B,H,N,D", [
        (1, 4, 64, 32),
        (2, 16, 128, 32),
        (1, 16, 384, 32),
        (2, 8, 97, 64),     # non-power-of-2 N
    ])
    def test_forward_matches_ref(self, B, H, N, D):
        Q, K, V, pos_weight, pos_bins, mask = _random_inputs(B, H, N, D)

        with torch.no_grad():
            out_triton = flash_diffusion_attn(Q, K, V, pos_weight, pos_bins, mask)
            out_ref = flash_diff_attn_ref(Q.float(), K.float(), V.float(),
                                          pos_weight, pos_bins, mask)

        torch.testing.assert_close(
            out_triton.float(), out_ref.float(),
            atol=1e-2, rtol=1e-2,
        )

    def test_forward_with_padding(self):
        B, H, N, D = 2, 8, 128, 32
        Q, K, V, pos_weight, pos_bins, mask = _random_inputs(B, H, N, D, with_mask=True)

        with torch.no_grad():
            out_triton = flash_diffusion_attn(Q, K, V, pos_weight, pos_bins, mask)
            out_ref = flash_diff_attn_ref(Q.float(), K.float(), V.float(),
                                          pos_weight, pos_bins, mask)

        # Padded positions should be zero
        for b in range(B):
            pad_start = (mask[b] == 0).nonzero(as_tuple=True)[0]
            if len(pad_start) > 0:
                assert (out_triton[b, :, pad_start[0]:, :] == 0).all()

        torch.testing.assert_close(
            out_triton.float(), out_ref.float(),
            atol=1e-2, rtol=1e-2,
        )

    def test_unbatched(self):
        H, N, D = 8, 64, 32
        Q = torch.randn(H, N, D, device=DEVICE, dtype=torch.bfloat16)
        K = torch.randn(H, N, D, device=DEVICE, dtype=torch.bfloat16)
        V = torch.randn(H, N, D, device=DEVICE, dtype=torch.bfloat16)
        pos_weight = torch.randn(H, NUM_BINS, device=DEVICE, dtype=torch.float32)
        pos_bins = torch.randint(0, NUM_BINS, (N, N), device=DEVICE, dtype=torch.int32)

        with torch.no_grad():
            out = flash_diffusion_attn(Q, K, V, pos_weight, pos_bins)

        assert out.shape == (H, N, D)


class TestFlashDiffusionAttnBackward:
    """Backward pass: Triton gradients vs PyTorch autograd on reference."""

    def _check_grads(self, B, H, N, D, with_mask=False):
        Q, K, V, pos_weight, pos_bins, mask = _random_inputs(
            B, H, N, D, dtype=torch.float32, with_mask=with_mask
        )

        # Reference forward+backward (float32, autograd)
        Q_ref = Q.detach().clone().requires_grad_(True)
        K_ref = K.detach().clone().requires_grad_(True)
        V_ref = V.detach().clone().requires_grad_(True)
        pw_ref = pos_weight.detach().clone().requires_grad_(True)

        out_ref = flash_diff_attn_ref(Q_ref, K_ref, V_ref, pw_ref, pos_bins, mask)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Triton forward+backward (bfloat16 compute, but float32 inputs for grad check)
        Q_tri = Q.detach().clone().to(torch.bfloat16).requires_grad_(True)
        K_tri = K.detach().clone().to(torch.bfloat16).requires_grad_(True)
        V_tri = V.detach().clone().to(torch.bfloat16).requires_grad_(True)
        pw_tri = pos_weight.detach().clone().requires_grad_(True)

        out_tri = flash_diffusion_attn(Q_tri, K_tri, V_tri, pw_tri, pos_bins, mask)
        loss_tri = out_tri.float().sum()
        loss_tri.backward()

        # Compare gradients (relaxed tolerance for bf16)
        torch.testing.assert_close(
            Q_tri.grad.float(), Q_ref.grad.float(), atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(
            K_tri.grad.float(), K_ref.grad.float(), atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(
            V_tri.grad.float(), V_ref.grad.float(), atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(
            pw_tri.grad.float(), pw_ref.grad.float(), atol=5e-2, rtol=5e-2
        )

    @pytest.mark.parametrize("B,H,N,D", [
        (1, 4, 64, 32),
        (2, 8, 128, 32),
    ])
    def test_backward_matches_ref(self, B, H, N, D):
        self._check_grads(B, H, N, D)

    def test_backward_with_padding(self):
        self._check_grads(2, 8, 128, 32, with_mask=True)

    def test_grad_flows(self):
        """Verify gradients are non-zero and flow to all parameters."""
        B, H, N, D = 1, 4, 64, 32
        Q, K, V, pos_weight, pos_bins, mask = _random_inputs(
            B, H, N, D, dtype=torch.bfloat16
        )

        out = flash_diffusion_attn(Q, K, V, pos_weight, pos_bins, mask)
        out.float().sum().backward()

        assert Q.grad is not None and Q.grad.abs().sum() > 0
        assert K.grad is not None and K.grad.abs().sum() > 0
        assert V.grad is not None and V.grad.abs().sum() > 0
        assert pos_weight.grad is not None and pos_weight.grad.abs().sum() > 0

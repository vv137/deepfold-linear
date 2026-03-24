"""Test consistency between training (dense) and inference (flash Triton) paths.

Verifies that TokenUOTBlock produces the same outputs in train vs eval mode,
which exercises the materialized pos_bias path vs the gather-in-kernel path.
"""

import pytest
import torch

from deepfold.model.position_encoding import PositionBias, compute_bins
from deepfold.model.trunk_block import TokenUOTBlock


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonFallbackConsistency:
    """Compare training (dense Sinkhorn) vs inference (flash Sinkhorn) outputs."""

    def _make_block(self, d_model=64, n_heads=4):
        block = TokenUOTBlock(d_model=d_model, n_heads=n_heads, block_idx=0)
        block.eps = torch.tensor([0.5, 1.0, 2.0, 4.0])
        return block

    def test_train_vs_eval_consistency(self):
        """h, x_res, log_u, log_v should be close between dense and flash paths."""
        torch.manual_seed(42)
        N, d_model, n_heads = 16, 64, 4

        block = self._make_block(d_model, n_heads).cuda()

        h = torch.randn(1, N, d_model, device="cuda")
        x_res = torch.randn(1, N, 3, device="cuda")
        mu = torch.softmax(torch.randn(1, n_heads, N, device="cuda"), dim=-1)
        nu = torch.softmax(torch.randn(1, n_heads, N, device="cuda"), dim=-1)
        log_u = torch.zeros(1, n_heads, N, device="cuda")
        log_v = torch.zeros(1, n_heads, N, device="cuda")

        chain_id = torch.zeros(1, N, dtype=torch.long, device="cuda")
        global_idx = torch.arange(N, device="cuda").unsqueeze(0)
        bond_matrix = torch.zeros(1, N, N, dtype=torch.bool, device="cuda")
        pos_bins = compute_bins(chain_id, global_idx, bond_matrix)

        pb = PositionBias(n_heads, 68).cuda()
        # Give it nonzero weights for a meaningful test
        pb.weight.data = torch.randn_like(pb.weight) * 0.1
        pos_weight = pb.weight

        # Training path (dense Sinkhorn)
        block.train()
        with torch.no_grad():
            h_train, x_train, lu_train, lv_train = block(
                h.clone(), x_res.clone(), mu, nu, log_u, log_v,
                pos_weight, pos_bins,
            )

        # Inference path (flash Sinkhorn / Triton)
        block.eval()
        with torch.no_grad():
            h_eval, x_eval, lu_eval, lv_eval = block(
                h.clone(), x_res.clone(), mu, nu, log_u, log_v,
                pos_weight, pos_bins,
            )

        # They won't be bit-exact (different computation order, FP32 accumulation),
        # but should be close since both solve the same Sinkhorn problem.
        atol, rtol = 1e-3, 1e-3
        torch.testing.assert_close(
            h_train, h_eval, atol=atol, rtol=rtol,
            msg="h mismatch between train (dense) and eval (flash)"
        )
        torch.testing.assert_close(
            x_train, x_eval, atol=atol, rtol=rtol,
            msg="x_res mismatch between train (dense) and eval (flash)"
        )
        torch.testing.assert_close(
            lu_train, lu_eval, atol=atol, rtol=rtol,
            msg="log_u mismatch between train (dense) and eval (flash)"
        )
        torch.testing.assert_close(
            lv_train, lv_eval, atol=atol, rtol=rtol,
            msg="log_v mismatch between train (dense) and eval (flash)"
        )

    def test_pos_bias_gather_equivalence(self):
        """Verify that weight[:, bins] == kernel gather for position bias."""
        torch.manual_seed(123)
        H, N = 4, 32

        weight = torch.randn(H, 68, device="cuda")
        bins = torch.randint(0, 68, (N, N), dtype=torch.int32, device="cuda")

        # Materialized
        pos_bias = weight[:, bins.long()]  # (H, N, N)

        # Gather (same as what kernel does)
        for h in range(H):
            for i in range(N):
                for j in range(N):
                    expected = weight[h, bins[i, j].item()]
                    actual = pos_bias[h, i, j]
                    assert expected == actual, f"Mismatch at h={h}, i={i}, j={j}"

    def test_batched_train_vs_eval(self):
        """Same as above but with B=2 and padding."""
        torch.manual_seed(7)
        B, N, d_model, n_heads = 2, 12, 64, 4

        block = self._make_block(d_model, n_heads).cuda()

        h = torch.randn(B, N, d_model, device="cuda")
        x_res = torch.randn(B, N, 3, device="cuda")
        mu = torch.softmax(torch.randn(B, n_heads, N, device="cuda"), dim=-1)
        nu = torch.softmax(torch.randn(B, n_heads, N, device="cuda"), dim=-1)
        log_u = torch.zeros(B, n_heads, N, device="cuda")
        log_v = torch.zeros(B, n_heads, N, device="cuda")
        pos_bins = torch.randint(0, 68, (B, N, N), dtype=torch.int32, device="cuda")

        pb = PositionBias(n_heads, 68).cuda()
        pb.weight.data = torch.randn_like(pb.weight) * 0.1
        pos_weight = pb.weight

        mask = torch.ones(B, N, device="cuda")
        mask[1, 8:] = 0  # pad last 4 positions in sample 1

        block.train()
        with torch.no_grad():
            h_train, x_train, _, _ = block(
                h.clone(), x_res.clone(), mu, nu, log_u, log_v,
                pos_weight, pos_bins, mask=mask,
            )

        block.eval()
        with torch.no_grad():
            h_eval, x_eval, _, _ = block(
                h.clone(), x_res.clone(), mu, nu, log_u, log_v,
                pos_weight, pos_bins, mask=mask,
            )

        atol, rtol = 1e-3, 1e-3
        torch.testing.assert_close(h_train, h_eval, atol=atol, rtol=rtol)
        torch.testing.assert_close(x_train, x_eval, atol=atol, rtol=rtol)

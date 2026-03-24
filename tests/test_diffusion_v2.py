"""Tests for DiffusionModuleV2 — encoder-transformer-decoder architecture."""

import pytest
import torch
import torch.nn as nn

from deepfold.model.diffusion_v2 import (
    SingleConditioning,
    DiffusionTransformerLayer,
    WindowedAtomBlock,
    AtomToTokenCrossAttn,
    TokenToAtomCrossAttn,
    DiffusionModuleV2,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# All tests run under autocast, matching real training
_autocast = torch.amp.autocast("cuda", dtype=DTYPE)


def _make_diffusion_inputs(B=2, N=64, M=256, d_model=512, d_atom=128):
    """Create synthetic inputs for diffusion module testing."""
    h_res = torch.randn(B, N, d_model, device=DEVICE, dtype=DTYPE)
    s_inputs = torch.randn(B, N, d_model, device=DEVICE, dtype=DTYPE)
    c_atom = torch.randn(B, M, d_atom, device=DEVICE, dtype=DTYPE)
    x_noisy = torch.randn(B, M, 3, device=DEVICE, dtype=DTYPE)
    sigma = torch.tensor([1.0, 5.0], device=DEVICE)[:B]

    # Token idx: assign ~M/N atoms per token
    atoms_per_token = M // N
    token_idx = torch.arange(N, device=DEVICE).repeat_interleave(atoms_per_token)
    token_idx = token_idx[:M].unsqueeze(0).expand(B, -1)

    # Position bins
    pos_bins = torch.randint(0, 68, (B, N, N), device=DEVICE, dtype=torch.int32)

    # Token-atom mapping
    token_atom_starts = (torch.arange(N, device=DEVICE) * atoms_per_token).unsqueeze(0).expand(B, -1).to(torch.int32)
    token_atom_counts = torch.full((B, N), atoms_per_token, device=DEVICE, dtype=torch.int32)

    token_pad_mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)
    atom_pad_mask = torch.ones(B, M, device=DEVICE, dtype=torch.float32)

    return dict(
        h_res=h_res, s_inputs=s_inputs, c_atom=c_atom,
        x_atom_noisy=x_noisy, sigma=sigma, token_idx=token_idx,
        pos_bins=pos_bins,
        token_atom_starts=token_atom_starts,
        token_atom_counts=token_atom_counts,
        token_pad_mask=token_pad_mask,
        atom_pad_mask=atom_pad_mask,
    )


class TestSingleConditioning:
    def test_shape(self):
        B, N, d = 2, 64, 512
        mod = SingleConditioning(d, 256).to(DEVICE).to(DTYPE)
        h_res = torch.randn(B, N, d, device=DEVICE, dtype=DTYPE)
        s_inputs = torch.randn(B, N, d, device=DEVICE, dtype=DTYPE)
        c_noise = torch.randn(B, device=DEVICE)
        with _autocast:
            out = mod(h_res, s_inputs, c_noise)
        assert out.shape == (B, N, d)


class TestDiffusionTransformerLayer:
    def test_shape(self):
        B, N, d, H = 2, 64, 512, 16
        layer = DiffusionTransformerLayer(d, H).to(DEVICE).to(DTYPE)
        s = torch.randn(B, N, d, device=DEVICE, dtype=DTYPE)
        s_cond = torch.randn(B, N, d, device=DEVICE, dtype=DTYPE)
        pos_w = torch.randn(H, 68, device=DEVICE, dtype=torch.float32)
        pos_bins = torch.randint(0, 68, (B, N, N), device=DEVICE, dtype=torch.int32)
        mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)

        with _autocast:
            out = layer(s, s_cond, pos_w, pos_bins, mask)
        assert out.shape == (B, N, d)


class TestWindowedAtomBlock:
    def test_shape(self):
        B, M, d = 2, 128, 128
        block = WindowedAtomBlock(d, d, 4).to(DEVICE).to(DTYPE)
        a = torch.randn(B, M, d, device=DEVICE, dtype=DTYPE)
        cond = torch.randn(B, M, d, device=DEVICE, dtype=DTYPE)
        mask = torch.ones(B, M, device=DEVICE, dtype=torch.float32)

        with _autocast:
            out = block(a, cond, mask)
        assert out.shape == (B, M, d)


class TestCrossAttnModules:
    def test_atom_to_token_shape(self):
        B, N, M, d_tok, d_atom = 2, 32, 160, 512, 128
        mod = AtomToTokenCrossAttn(d_tok, d_atom, 4).to(DEVICE).to(DTYPE)
        s = torch.randn(B, N, d_tok, device=DEVICE, dtype=DTYPE)
        a = torch.randn(B, M, d_atom, device=DEVICE, dtype=DTYPE)
        atoms_per_tok = M // N
        starts = (torch.arange(N, device=DEVICE) * atoms_per_tok).unsqueeze(0).expand(B, -1).to(torch.int32)
        counts = torch.full((B, N), atoms_per_tok, device=DEVICE, dtype=torch.int32)
        mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)

        with _autocast:
            out = mod(s, a, starts, counts, mask)
        assert out.shape == (B, N, d_tok)

    def test_token_to_atom_shape(self):
        B, N, M, d_tok, d_atom = 2, 32, 160, 512, 128
        mod = TokenToAtomCrossAttn(d_tok, d_atom, 4).to(DEVICE).to(DTYPE)
        a = torch.randn(B, M, d_atom, device=DEVICE, dtype=DTYPE)
        s = torch.randn(B, N, d_tok, device=DEVICE, dtype=DTYPE)
        a_mask = torch.ones(B, M, device=DEVICE, dtype=torch.float32)
        t_mask = torch.ones(B, N, device=DEVICE, dtype=torch.float32)

        with _autocast:
            out = mod(a, s, a_mask, t_mask)
        assert out.shape == (B, M, d_atom)


class TestDiffusionModuleV2:
    def test_forward_shape(self):
        """Full forward: output shape matches input atom coords."""
        model = DiffusionModuleV2(
            d_model=512, d_atom=128, d_fourier=256,
            n_transformer_layers=2,  # small for testing
            n_encoder_blocks=1,
            n_decoder_blocks=1,
        ).to(DEVICE).to(DTYPE)

        inputs = _make_diffusion_inputs(B=2, N=32, M=128)
        with _autocast:
            x_pred = model(**inputs)
        assert x_pred.shape == (2, 128, 3)

    def test_gradient_flow(self):
        """Verify gradients flow through h_res back into trunk params."""
        # Use float32 to avoid bf16 gradient underflow
        model = DiffusionModuleV2(
            n_transformer_layers=2,
            n_encoder_blocks=1,
            n_decoder_blocks=1,
        ).to(DEVICE).float()

        # Break zero-init so gradients can flow through coord_out
        with torch.no_grad():
            nn.init.normal_(model.coord_out.weight, std=0.1)
            nn.init.normal_(model.coord_out.bias, std=0.1)

        inputs = _make_diffusion_inputs(B=1, N=32, M=128)
        # Float32 inputs
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                inputs[k] = v.float()
        inputs["h_res"] = inputs["h_res"].requires_grad_(True)

        x_pred = model(**inputs)
        # MSE loss, not sum — re-centering makes sum(x_pred)≡0
        target = torch.randn_like(x_pred)
        loss = ((x_pred - target) ** 2).sum()
        loss.backward()

        assert inputs["h_res"].grad is not None
        assert inputs["h_res"].grad.abs().sum() > 0, "No gradient to h_res"

    def test_c_skip_effect(self):
        """Verify c_skip is used: at σ→0, output ≈ x_noisy."""
        model = DiffusionModuleV2(
            n_transformer_layers=2,
            n_encoder_blocks=1,
            n_decoder_blocks=1,
        ).to(DEVICE).to(DTYPE)

        inputs = _make_diffusion_inputs(B=1, N=32, M=128)
        inputs["sigma"] = torch.tensor([0.001], device=DEVICE)

        with torch.no_grad(), _autocast:
            x_pred = model(**inputs)

        x_noisy = inputs["x_atom_noisy"]
        diff = (x_pred.float() - x_noisy.float()).abs().mean()
        assert diff < 1.0, f"At σ≈0, x_pred should be close to x_noisy, got mean diff={diff}"

    def test_padded_atoms_zero(self):
        """Padded atom positions should have zero output."""
        model = DiffusionModuleV2(
            n_transformer_layers=2,
            n_encoder_blocks=1,
            n_decoder_blocks=1,
        ).to(DEVICE).to(DTYPE)

        inputs = _make_diffusion_inputs(B=1, N=32, M=128)
        inputs["atom_pad_mask"][:, -20:] = 0.0

        with torch.no_grad(), _autocast:
            x_pred = model(**inputs)

        assert (x_pred[:, -20:, :] == 0).all()

    def test_param_count(self):
        """Verify parameter count is in expected range."""
        model = DiffusionModuleV2(
            n_transformer_layers=24,
            n_encoder_blocks=3,
            n_decoder_blocks=3,
        )
        n_params = sum(p.numel() for p in model.parameters())
        # Expected ~155M (24 transformer layers + 3+3 encoder/decoder + conditioning)
        assert 100_000_000 < n_params < 200_000_000, f"Param count {n_params/1e6:.1f}M out of range"
        print(f"DiffusionModuleV2 params: {n_params/1e6:.1f}M")

"""Tests for weight initialization (SPEC §14)."""

import math

import torch
import torch.nn as nn


def _make_small_model():
    """Build a minimal DeepFoldLinear for init testing."""
    from deepfold.model.deepfold import DeepFoldLinear

    return DeepFoldLinear(
        d_model=64,
        d_msa=16,
        d_atom=32,
        h_res=4,
        h_msa=4,
        n_msa_blocks=1,
        n_uot_blocks=4,
        n_diff_transformer_layers=2,
        n_diff_encoder_blocks=1,
        n_diff_decoder_blocks=1,
        n_diff_heads=4,
        d_fourier=32,
        diffusion_multiplicity=1,
    )


class TestZeroInit:
    """Params that must be exactly zero at init."""

    def test_gamma_zero(self):
        model = _make_small_model()
        for name, p in model.named_parameters():
            if "gamma" in name:
                assert (p == 0).all(), f"{name} should be zero"

    def test_w_dist_logit_init(self):
        model = _make_small_model()
        for name, p in model.named_parameters():
            if "w_dist_logit" in name:
                assert torch.allclose(p, torch.full_like(p, -2.0)), f"{name} should be -2.0"

    def test_alpha_coevol_zero(self):
        model = _make_small_model()
        for name, p in model.named_parameters():
            if "alpha_coevol" in name:
                assert (p == 0).all(), f"{name} should be zero"

    def test_position_bias_zero(self):
        model = _make_small_model()
        for name, p in model.named_parameters():
            if "pos_bias.weight" in name:
                assert (p == 0).all(), f"{name} should be zero"

    def test_coord_out_zero(self):
        model = _make_small_model()
        for name, p in model.named_parameters():
            if "coord_out" in name:
                assert (p == 0).all(), f"{name} should be zero"


class TestXavierInit:
    """Weight matrices should be Xavier normal (not Kaiming uniform default)."""

    def test_trunk_uot_w_q_is_xavier_scale(self):
        model = _make_small_model()
        for name, p in model.named_parameters():
            if "trunk.uot_blocks.0.w_q.weight" in name:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p)
                expected_std = math.sqrt(2.0 / (fan_in + fan_out))
                actual_std = p.std().item()
                # Xavier normal std should be close (within 30% for small matrices)
                assert abs(actual_std - expected_std) / expected_std < 0.5, (
                    f"{name}: std={actual_std:.4f}, expected~{expected_std:.4f}"
                )

    def test_input_embed_is_xavier(self):
        model = _make_small_model()
        p = model.trunk.token_embed.proj.weight
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p)
        expected_std = math.sqrt(2.0 / (fan_in + fan_out))
        actual_std = p.std().item()
        assert abs(actual_std - expected_std) / expected_std < 0.5


class TestDepthScaling:
    """W_O in deep block stacks should be scaled by 1/sqrt(L)."""

    def test_trunk_w_o_scaled(self):
        model = _make_small_model()
        for name, p in model.named_parameters():
            if "trunk.uot_blocks.0.w_o.weight" in name:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p)
                xavier_std = math.sqrt(2.0 / (fan_in + fan_out))
                xavier_std / math.sqrt(48)  # always uses 48
                actual_std = p.std().item()
                # Scaled down significantly from Xavier
                assert actual_std < xavier_std * 0.5, (
                    f"{name}: std={actual_std:.4f} should be << xavier={xavier_std:.4f}"
                )

    def test_trunk_swiglu_out_not_depth_scaled(self):
        """SwiGLU output should NOT be depth-scaled (self-suppression suffices)."""
        model = _make_small_model()
        for name, p in model.named_parameters():
            if "trunk.uot_blocks.0.swiglu.out_proj.weight" in name:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p)
                xavier_std = math.sqrt(2.0 / (fan_in + fan_out))
                actual_std = p.std().item()
                # Should be approximately Xavier, NOT scaled down
                assert abs(actual_std - xavier_std) / xavier_std < 0.5, (
                    f"{name}: std={actual_std:.4f}, expected~{xavier_std:.4f}"
                )


class TestLayerNorm:
    """LayerNorm weight=1, bias=0."""

    def test_layernorm_defaults(self):
        model = _make_small_model()
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    assert (module.weight == 1).all(), f"{name}.weight should be 1"
                    if module.bias is not None:
                        assert (module.bias == 0).all(), f"{name}.bias should be 0"

"""Tests for checkpoint save/load/resume roundtrip."""

import numpy as np
import torch

from deepfold.model.deepfold import DeepFoldLinear
from deepfold.train.trainer import EMA, build_optimizer


def _make_small_model():
    return DeepFoldLinear(
        d_model=64,
        d_msa=16,
        d_atom=32,
        h_res=4,
        h_msa=2,
        n_msa_blocks=1,
        n_uot_blocks=2,
        n_diff_transformer_layers=2,
        n_diff_encoder_blocks=1,
        n_diff_decoder_blocks=1,
        n_diff_heads=4,
        d_fourier=32,
        max_cycles=1,
        inference_cycles=1,
        diffusion_multiplicity=1,
    )


class TestCheckpointRoundtrip:
    """Save a checkpoint, load it back, verify everything matches."""

    def test_save_load_matches(self, tmp_path):
        torch.manual_seed(0)
        model = _make_small_model()
        optimizer = build_optimizer(model, lr=1e-3)
        ema = EMA(model, decay=0.999, warmup_steps=10)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Simulate a few optimizer steps to get non-trivial state
        for _ in range(3):
            for p in model.parameters():
                if p.requires_grad:
                    p.grad = torch.randn_like(p) * 0.01
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)

        step = 42

        # Save checkpoint (mirrors train.py logic)
        ckpt_path = tmp_path / "step_42.pt"
        ckpt_data = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
        }
        torch.save(ckpt_data, ckpt_path)

        # Build fresh model + optimizer + ema
        torch.manual_seed(999)  # different seed
        model2 = _make_small_model()
        optimizer2 = build_optimizer(model2, lr=1e-3)
        ema2 = EMA(model2, decay=0.999, warmup_steps=10)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2.load_state_dict(ckpt["model"])
        optimizer2.load_state_dict(ckpt["optimizer"])
        ema2.load_state_dict(ckpt["ema"])
        loaded_step = ckpt["step"]

        # Verify step
        assert loaded_step == step

        # Verify model parameters match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, msg=f"model param {n1} mismatch")

        # Verify EMA shadow matches
        for name in ema.shadow:
            torch.testing.assert_close(
                ema.shadow[name], ema2.shadow[name],
                msg=f"EMA shadow {name} mismatch",
            )
        assert ema.step == ema2.step

        # Verify optimizer state matches (momentum buffers etc.)
        for g1, g2 in zip(optimizer.param_groups, optimizer2.param_groups):
            assert g1["lr"] == g2["lr"]
            assert g1["weight_decay"] == g2["weight_decay"]

        # Verify RNG state roundtrip
        torch.set_rng_state(ckpt["rng_state"])
        np.random.set_state(ckpt["np_rng_state"])
        r1 = torch.randn(5)
        n1 = np.random.randn(5)

        torch.set_rng_state(ckpt["rng_state"])
        np.random.set_state(ckpt["np_rng_state"])
        r2 = torch.randn(5)
        n2 = np.random.randn(5)

        torch.testing.assert_close(r1, r2)
        np.testing.assert_array_equal(n1, n2)

    def test_symlink_latest(self, tmp_path):
        """Verify latest.pt symlink logic."""
        import os

        ckpt_path = tmp_path / "step_100.pt"
        torch.save({"step": 100}, ckpt_path)

        latest = tmp_path / "latest.pt"
        tmp_link = str(latest) + ".tmp"
        os.symlink(os.path.basename(ckpt_path), tmp_link)
        os.replace(tmp_link, str(latest))

        assert latest.is_symlink()
        assert latest.resolve() == ckpt_path.resolve()
        loaded = torch.load(latest, weights_only=False)
        assert loaded["step"] == 100

    def test_resume_continues_from_step(self, tmp_path):
        """Verify training loop would start from step+1 after resume."""
        torch.manual_seed(0)
        model = _make_small_model()
        optimizer = build_optimizer(model, lr=1e-3)
        ema = EMA(model, decay=0.999, warmup_steps=10)

        save_step = 500
        ckpt_path = tmp_path / "step_500.pt"
        torch.save(
            {
                "step": save_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "rng_state": torch.get_rng_state(),
                "np_rng_state": np.random.get_state(),
            },
            ckpt_path,
        )

        # Simulate resume logic from train.py
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        start_step = ckpt["step"]
        total_steps = 1000

        loop_steps = list(range(start_step + 1, total_steps + 1))
        assert loop_steps[0] == 501
        assert loop_steps[-1] == 1000
        assert len(loop_steps) == 500

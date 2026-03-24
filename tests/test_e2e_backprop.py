"""End-to-end backpropagation tests (SPEC §13.4 v4.5).

Verifies that gradients flow correctly from all four losses through the
entire model:
  - L_diff, L_lddt → atom blocks → h_cond → h_res → trunk (end-to-end)
  - L_disto → h_res → trunk
  - L_trunk_coord → x_res → EGNN γ → trunk
"""

import torch

from deepfold.model.deepfold import DeepFoldLinear
from deepfold.train.trainer import build_optimizer, get_lr


def _make_model(device="cpu", multiplicity=2):
    """Build a minimal model for testing."""
    return DeepFoldLinear(
        d_model=64,
        d_msa=16,
        d_atom=32,
        h_res=4,
        h_msa=4,
        n_msa_blocks=1,
        n_uot_blocks=2,
        n_diff_transformer_layers=2,
        n_diff_encoder_blocks=1,
        n_diff_decoder_blocks=1,
        n_diff_heads=4,
        d_fourier=32,
        max_cycles=1,
        inference_cycles=1,
        diffusion_multiplicity=multiplicity,
    ).to(device)


def _make_batch(N=8, N_atom=16, S=2, device="cpu"):
    """Create a synthetic batch matching model forward() signature."""
    token_type = torch.zeros(N, dtype=torch.long, device=device)
    profile = torch.randn(N, 32, device=device).softmax(dim=-1)
    del_mean = torch.zeros(N, 1, device=device)
    has_msa = torch.ones(N, 1, device=device)
    protein_mask = torch.ones(N, dtype=torch.bool, device=device)
    msa_feat = torch.randn(S, N, 34, device=device)

    # D_REF = 197: pos(3) + charge(1) + mask(1) + element_onehot(128) + name_onehot(64)
    c_atom = torch.randn(N_atom, 197, device=device)
    # Raw atom pair features: disp(3) + inv_dist(1) + valid(1) = 5
    p_lm = torch.randn(0, 5, device=device)
    p_lm_idx = torch.zeros(0, 2, dtype=torch.long, device=device)

    token_idx = torch.arange(N, device=device).repeat_interleave(N_atom // N)[:N_atom]

    chain_id = torch.zeros(N, dtype=torch.long, device=device)
    global_idx = torch.arange(N, dtype=torch.long, device=device)
    bond_matrix = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(N - 1):
        bond_matrix[i, i + 1] = True
        bond_matrix[i + 1, i] = True

    x_atom_true = torch.randn(N_atom, 3, device=device) * 5.0
    x_res_true = torch.zeros(N, 3, device=device)
    for i in range(N):
        mask = token_idx == i
        if mask.any():
            x_res_true[i] = x_atom_true[mask].mean(dim=0)

    atom_resolved_mask = torch.ones(N_atom, device=device)
    token_resolved_mask = torch.ones(N, device=device)

    return {
        "token_type": token_type,
        "profile": profile,
        "del_mean": del_mean,
        "has_msa": has_msa,
        "msa_feat": msa_feat,
        "c_atom": c_atom,
        "p_lm": p_lm,
        "p_lm_idx": p_lm_idx,
        "token_idx": token_idx,
        "chain_id": chain_id,
        "global_idx": global_idx,
        "bond_matrix": bond_matrix,
        "protein_mask": protein_mask,
        "x_atom_true": x_atom_true,
        "x_res_true": x_res_true,
        "atom_resolved_mask": atom_resolved_mask,
        "token_resolved_mask": token_resolved_mask,
    }


class TestEndToEndBackprop:
    """Verify gradient flow through entire model."""

    def test_forward_backward_no_nan(self):
        """Single forward+backward produces finite loss and no NaN grads."""
        torch.manual_seed(0)
        model = _make_model()
        batch = _make_batch()
        model.train()

        outputs = model(**batch)
        loss = outputs["loss"]

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"

    def test_all_losses_present(self):
        """All four loss components are returned and finite."""
        torch.manual_seed(1)
        model = _make_model()
        batch = _make_batch()
        model.train()

        outputs = model(**batch)
        for key in ["loss", "l_diff", "l_lddt", "l_disto", "l_trunk_coord"]:
            assert key in outputs, f"Missing loss: {key}"
            assert torch.isfinite(outputs[key]), f"{key} not finite: {outputs[key]}"

    def test_trunk_receives_gradient_from_diffusion(self):
        """v4.5: h_res not frozen — diffusion loss backprops into trunk params.

        At init, coord_out is zero-initialized so L_diff gradient through
        coord_out → q → h_cond → h_res is zero (by design, SPEC §13.5).
        After a few steps, coord_out has non-zero weights and gradient flows.
        We simulate this by perturbing coord_out weights before the test.
        """
        torch.manual_seed(2)
        model = _make_model()
        batch = _make_batch()
        model.train()

        # Perturb coord_out from zero-init so gradient can flow through
        for name, p in model.named_parameters():
            if "coord_out" in name and p.dim() == 2:
                p.data.normal_(0, 0.01)

        outputs = model(**batch)
        outputs["l_diff"].backward()

        trunk_with_grad = [
            (name, p.grad.abs().sum().item())
            for name, p in model.named_parameters()
            if "trunk" in name and p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(trunk_with_grad) > 0, (
            "No trunk parameter received gradient from L_diff — "
            "end-to-end gradient flow is broken"
        )

    def test_coord_out_zero_init_blocks_trunk_gradient(self):
        """At init, zero coord_out blocks L_diff → trunk gradient (by design)."""
        torch.manual_seed(2)
        model = _make_model()
        batch = _make_batch()
        model.train()

        # Verify coord_out is zero
        for name, p in model.named_parameters():
            if "coord_out" in name and p.dim() == 2:
                assert p.abs().sum() == 0, f"coord_out should be zero-init: {name}"

        outputs = model(**batch)
        outputs["l_diff"].backward()

        # L_diff gradient to coord_out weights should be non-zero
        # (the weights themselves get gradient even though they're zero)
        coord_out_grad = [
            p.grad.abs().sum().item()
            for name, p in model.named_parameters()
            if "coord_out" in name and p.grad is not None
        ]
        assert any(g > 0 for g in coord_out_grad), "coord_out should receive gradient"

        # But trunk params should get zero from L_diff alone (gradient dies at zero W)
        trunk_grad_sum = sum(
            p.grad.abs().sum().item()
            for name, p in model.named_parameters()
            if "trunk" in name and p.grad is not None
        )
        assert trunk_grad_sum == 0, (
            "Trunk should get zero gradient from L_diff at init (zero coord_out)"
        )

    def test_egnn_gamma_receives_gradient(self):
        """L_trunk_coord provides gradient to EGNN γ parameters."""
        torch.manual_seed(3)
        model = _make_model()
        batch = _make_batch()
        model.train()

        outputs = model(**batch)
        outputs["l_trunk_coord"].backward()

        gamma_grads = []
        for name, p in model.named_parameters():
            if "gamma" in name and p.grad is not None:
                gamma_grads.append((name, p.grad.abs().sum().item()))

        assert len(gamma_grads) > 0, "No EGNN γ parameters received gradient"
        assert any(g > 0 for _, g in gamma_grads), (
            f"All EGNN γ gradients are zero: {gamma_grads}"
        )

    def test_multi_step_training(self):
        """3 training steps: loss stays finite, parameters change."""
        torch.manual_seed(42)
        model = _make_model()
        optimizer = build_optimizer(model, lr=1e-3)
        batch = _make_batch()

        initial_params = {name: p.clone() for name, p in model.named_parameters()}
        losses = []

        for step in range(3):
            model.train()
            optimizer.zero_grad()

            lr = get_lr(step, warmup_steps=1, total_steps=10)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            outputs = model(**batch)
            loss = outputs["loss"]
            assert torch.isfinite(loss), f"Step {step}: loss={loss.item()}"

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        # Parameters should have changed
        n_changed = sum(
            1
            for name, p in model.named_parameters()
            if not torch.equal(p, initial_params[name])
        )
        assert n_changed > 0, "No parameters changed after 3 steps"

        # All losses finite
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), (
            f"NaN/Inf in losses: {losses}"
        )

    def test_multi_step_loss_trend(self):
        """5 steps on same batch with high LR: loss should decrease or stay finite."""
        torch.manual_seed(99)
        model = _make_model()
        optimizer = build_optimizer(model, lr=5e-3)
        batch = _make_batch()

        losses = []
        for step in range(5):
            model.train()
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Must not explode
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), (
            f"Loss exploded: {losses}"
        )

    def test_optimizer_three_param_groups(self):
        """Optimizer has 3 groups: decay, no-decay, gamma."""
        model = _make_model()
        optimizer = build_optimizer(model, lr=1e-4)

        assert len(optimizer.param_groups) == 3
        # Group 0: weight decay
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        # Group 1: no decay (LN, bias)
        assert optimizer.param_groups[1]["weight_decay"] == 0.0
        # Group 2: gamma with decay
        assert optimizer.param_groups[2]["weight_decay"] == 0.01

    def test_distogram_gradient_to_trunk(self):
        """L_disto backprops into trunk h_res parameters."""
        torch.manual_seed(4)
        model = _make_model()
        batch = _make_batch()
        model.train()

        outputs = model(**batch)
        outputs["l_disto"].backward()

        # Distogram loss head params should have gradient
        disto_grads = [
            (name, p.grad.abs().sum().item())
            for name, p in model.named_parameters()
            if "distogram" in name and p.grad is not None
        ]
        assert len(disto_grads) > 0 and any(g > 0 for _, g in disto_grads)

        # Trunk params should also have gradient (through h_res)
        trunk_grads = [
            (name, p.grad.abs().sum().item())
            for name, p in model.named_parameters()
            if "trunk" in name and p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(trunk_grads) > 0, "L_disto did not backprop into trunk"

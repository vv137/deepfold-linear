"""Smoke test: full training loop for a few steps with synthetic data.

Verifies the entire pipeline: model construction → forward → loss → backward → optimizer step.
"""

import torch
import time

from deepfold.model.deepfold import DeepFoldLinear
from deepfold.train.trainer import build_optimizer, get_lr


def make_synthetic_batch(N=32, N_atom=64, S=4, device="cuda"):
    """Create a minimal synthetic batch matching DeepFoldLinear.forward() signature.

    Uses raw featurizer output shapes: c_atom (N_atom, 197), p_lm (n_pairs, 5).
    """
    D_REF = 197  # 3(pos) + 1(charge) + 1(mask) + 128(element) + 64(name)

    # Token-level
    token_type = torch.randint(0, 4, (N,), device=device)
    profile = torch.randn(N, 32, device=device).softmax(dim=-1)
    del_mean = torch.randn(N, 1, device=device).abs() * 0.1
    has_msa = torch.ones(N, 1, device=device)
    msa_token_mask = torch.ones(N, device=device, dtype=torch.bool)

    # MSA
    msa_feat = torch.randn(1, S, N, 34, device=device)

    # Atom-level: raw ref conformer features (before embedding)
    c_atom = torch.randn(N_atom, D_REF, device=device)

    # Intra-token atom pairs: raw features [disp(3), inv_dist(1), valid(1)]
    atoms_per_token = N_atom // N
    pair_list = []
    idx_list = []
    offset = 0
    for tok in range(N):
        c = min(atoms_per_token, N_atom - offset)
        for i in range(c):
            for j in range(c):
                idx_list.append([offset + i, offset + j])
                disp = torch.randn(3, device=device)
                inv_d = 1.0 / (1.0 + (disp**2).sum())
                pair_list.append(
                    torch.cat([disp, inv_d.unsqueeze(0), torch.ones(1, device=device)])
                )
        offset += c
    if pair_list:
        p_lm = torch.stack(pair_list)  # (n_pairs, 5)
        p_lm_idx = torch.tensor(idx_list, dtype=torch.long, device=device)
    else:
        p_lm = torch.zeros(0, 5, device=device)
        p_lm_idx = torch.zeros(0, 2, dtype=torch.long, device=device)

    # Atom-to-token mapping (2 atoms per token)
    token_idx = torch.arange(N, device=device).repeat_interleave(N_atom // N)
    if token_idx.shape[0] < N_atom:
        token_idx = torch.cat(
            [
                token_idx,
                torch.full(
                    (N_atom - token_idx.shape[0],),
                    N - 1,
                    device=device,
                    dtype=torch.long,
                ),
            ]
        )
    token_idx = token_idx[:N_atom]

    # Chain/position
    chain_id = torch.zeros(N, device=device, dtype=torch.long)
    global_idx = torch.arange(N, device=device)
    bond_matrix = torch.zeros(N, N, device=device, dtype=torch.bool)
    for i in range(N - 1):
        bond_matrix[i, i + 1] = True
        bond_matrix[i + 1, i] = True

    # Ground truth
    x_atom_true = torch.randn(N_atom, 3, device=device) * 10.0
    # Token coords from atom coords
    x_res_true = torch.zeros(N, 3, device=device)
    for i in range(N):
        mask = token_idx == i
        if mask.any():
            x_res_true[i] = x_atom_true[mask].mean(dim=0)

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
        "msa_token_mask": msa_token_mask,
        "x_atom_true": x_atom_true,
        "x_res_true": x_res_true,
    }


def main():
    device = "cuda"
    torch.manual_seed(42)

    # Full-size dims, reduced depth for smoke test
    print("Building model...")
    model = DeepFoldLinear(
        d_model=512,
        d_msa=64,
        d_atom=128,
        h_res=16,
        h_msa=8,
        n_msa_blocks=1,
        n_uot_blocks=4,  # 4 instead of 48
        n_diff_transformer_layers=2,
        n_diff_encoder_blocks=1,
        n_diff_decoder_blocks=1,
        n_diff_heads=4,
        d_fourier=32,
        max_cycles=1,
        inference_cycles=1,
        diffusion_multiplicity=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = build_optimizer(model, lr=1e-4)

    N, N_atom = 512, 4096
    batch = make_synthetic_batch(N=N, N_atom=N_atom, S=4, device=device)

    print(f"\nTraining {5} steps (N={N}, N_atom={N_atom})...\n")

    losses = []
    for step in range(5):
        model.train()
        optimizer.zero_grad()

        lr = get_lr(step, warmup_steps=2, total_steps=10)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.perf_counter()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs["loss"]

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        losses.append(loss.item())
        print(
            f"  step {step}: loss={loss.item():.4f}  "
            f"l_diff={outputs['l_diff'].item():.4f}  "
            f"l_lddt={outputs['l_lddt'].item():.4f}  "
            f"l_disto={outputs['l_disto'].item():.4f}  "
            f"l_trunk={outputs['l_trunk_coord'].item():.4f}  "
            f"grad_norm={grad_norm.item():.2f}  "
            f"lr={lr:.2e}  "
            f"time={dt:.0f}ms"
        )

    # Verify loss decreased (or at least didn't explode)
    print(f"\nLoss: {losses[0]:.4f} → {losses[-1]:.4f}")
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "NaN/Inf in loss!"
    print("No NaN/Inf — smoke test PASSED")

    # Quick memory report
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak GPU memory: {peak_mem:.0f}MB")


if __name__ == "__main__":
    main()

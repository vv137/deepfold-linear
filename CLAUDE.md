# DeepFold-Linear

## Instructions

* Follow AF3 unless SPEC.md specifies otherwise.
* See SPEC.md for full design specification.
* Use uv. Standard repo layout. PyTorch only. No Lightning.
* Tests: `uv run python -m pytest tests/ -x -v`

## Project Layout

* Package: `deepfold-linear` (import as `deepfold`)
* Source: `src/deepfold/{model,data,train,utils}`
* Config: `configs/model.yaml`
* References: `references/boltz/`, `references/flash-sinkhorn/` (gitignored)

## Conventions

* Mixed precision: BF16 general, FP32 for Sinkhorn log-domain ops
* **Bias**: `bias=False` after LayerNorm (trunk Q/K/V/G/O, SwiGLU, LN_Lin). Exception: AtomBlock W_Q has `bias=True` (AF3 Alg 24). `bias=True` on standalone projections.
* Einsum output dim order: `(H, N, N)` for attention, `(H, N, d_h)` for projected

## Architecture (~375M params)

| Component | Module | Key Design |
|-----------|--------|------------|
| Input embedding | `model/input_embedding.py` | Token (38→512) + atom encoder (scatter_mean) |
| MSA module | `model/msa.py` | 4 blocks, low-rank co-evolution (rank 16), Triton coevol kernel (inference) |
| Trunk | `model/trunk.py` | 48 UOT+EGNN blocks, random cycles 1-5, flash Sinkhorn attention |
| Sinkhorn attention | `model/kernels/sinkhorn_kernel.py` | Triton fwd+bwd, O(N) memory, CG-based IFT backward |
| EGNN | `model/trunk_block.py` | Transport-weighted centroid, per-head γ (noise init 1e-4), per-layer pos_bias |
| Diffusion | `model/diffusion_v2.py` | AF3 style: 3 encoder + 24 transformer + 3 decoder, c_skip EDM |
| Losses | `model/losses.py` | EDM diffusion (Kabsch-aligned), smooth LDDT, distogram (Triton eval) |

## Triton Kernels

| Kernel | File | Batch | Training | Inference |
|--------|------|-------|----------|-----------|
| Flash Sinkhorn (fwd+bwd) | `kernels/sinkhorn_kernel.py` | ✅ (B*H, n_tiles) | ✅ exact unrolled backward | ✅ |
| Flash Sinkhorn wrapper | `kernels/flash_sinkhorn_attn.py` | ✅ | ✅ (training+inference) | ✅ |
| Flash diffusion attn | `kernels/flash_diffusion_attn.py` | ✅ (B*H, n_tiles) | ✅ fwd+bwd | ✅ |
| Windowed atom attn | `kernels/flash_atom_attn.py` | ✅ (B*H, n_windows) | ✅ fwd+bwd | ✅ |
| Cross-attention | `kernels/cross_attn_kernel.py` | ✅ | ✅ fwd+bwd (kernel LSE) | ✅ |
| Co-evolution | `kernels/coevol_kernel.py` | ✅ (B, n_tiles) | ✅ fwd+bwd (cached w_tile) | ✅ |
| Distogram loss | `kernels/distogram_kernel.py` | ✅ (B, n_i, n_j) | ✅ fwd+bwd (recompute from x_true) | ✅ |

## Training Infrastructure

* Script: `scripts/train.py` (DDP, gradient accumulation, EMA)
* Optimizer: AdamW, 3 param groups (decay, no-decay, EGNN γ)
* LR: AF3 schedule — linear warmup 1000 steps + plateau + exponential decay (0.95× every 50k)
* Checkpoint: model + optimizer + EMA + scaler + RNG → `latest.pt` symlink
* Resume: `--resume runs/.../checkpoints/latest.pt`
* DDP: `find_unused_parameters=True`, OOM broadcast across ranks

## Development History

| Phase | Description |
|-------|-------------|
| 0-8 | Core architecture, data pipeline, training infra |
| 9 | SPEC v4.4: unrolled Sinkhorn, 10 atom blocks, Xavier init |
| 10 | Augmentation, stability fixes, Heun sampler |
| 11 | SPEC v4.5: remove diffusion UOT, end-to-end gradient |
| 12 | SPEC v4.6: AF3 alignment (AdaLN-Zero, FourierEmbed, c_in scaling, Kabsch) |
| 13 | Flash Sinkhorn: Triton fwd+CG-IFT bwd, batch dims, mask support, kernel wiring |
| 14 | SPEC v5: AF3 diffusion (encoder-transformer-decoder), Triton flash/windowed/cross-attn kernels, proper c_skip, ~375M total |
| 15 | SPEC v5.4: per-layer marginals, per-cycle MSA subsampling, h_res-conditioned column scoring |

## Known AF3 Divergences

**Trunk:**
* No pair representation (O(N²) → O(N) persistent state)
* UOT-Sinkhorn replaces softmax attention in trunk
* CG-IFT backward (not unrolled autograd through Sinkhorn iterations)
* EGNN replaces IPA for structure refinement
* No triangle attention/updates
* Random cycle count 1-3 (AF3 uses fixed recycling)
* Low-rank co-evolution (rank 16) instead of full outer product
* 68-bin position encoding instead of RoPE

**Diffusion (v5, mostly aligned):**
* 16 diffusion multiplicity (Boltz-1) instead of AF3's 48
* 68-bin PositionBias replaces AF3's O(N²) pair bias z in transformer
* Cross-attention (learnable) replaces scatter_mean/gather (hard) for atom↔token
* Single conditioning track (s = features AND conditioning) vs AF3's separate s/a tracks
* dim=512 (vs AF3 768), ~155M params (vs AF3 ~445M)
* Atom pair bias p_lm not used in v2 atom blocks (TODO)
* Diffusion transformer pos_bias may be redundant — s already inherits position from trunk h_res conditioning (TODO: ablation)
* Custom Triton kernels for all attention (AF3 uses PyTorch SDPA)

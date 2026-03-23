# DeepFold-Linear

* Follow Boltz-1 unless SPEC.md specifies otherwise.
* Update CLAUDE.md before starting work.
* See SPEC.md for full design specification.
* Clone Boltz-1 into `references/boltz/` as reference.
* Use uv. Standard repo layout.
* PyTorch only. No Lightning.
* Investigate and report any conflicts between SPEC.md and Boltz-1.

## Project Setup (established)

* Package name: `deepfold-linear` (import as `deepfold`)
* Source layout: `src/deepfold/{model,data,train,utils}`
* Boltz-1 reference: `references/boltz/` (gitignored)
* Flash-Sinkhorn reference: `references/flash-sinkhorn/` (gitignored)
* Tests: `uv run python -m pytest tests/ -x -v`
* Config: `configs/model.yaml`
* No batch dimension — single sample per forward pass (variable-length proteins)
* Mixed precision: BF16 general, FP32 for Sinkhorn log-domain ops
* Einsum convention: output dim order = `(H, N, N)` for attention, `(H, N, d_h)` for projected
* **Bias convention (SPEC §0 v4.4)**: `bias=False` on all projections after LayerNorm (attention Q/K/V/G/O, SwiGLU, LN_Lin). `bias=True` on standalone projections (input embed, coord output, loss heads, AdaLN, cond_proj, cond_gate).

## Architecture Status

* Phase 0: Project setup ✅
* Phase 1: Primitives + input embeddings ✅
* Phase 2: MSA module (4 blocks) ✅
* Phase 3: UOT-Sinkhorn + unrolled backward + EGNN ✅
* Phase 4: Trunk / recycling loop ✅
* Phase 5: Diffusion module (2 UOT + 10 atom blocks) ✅
* Phase 6: Loss functions ✅
* Phase 7: Training infrastructure ✅
* Phase 8: Data pipeline ✅
  * `src/deepfold/data/crop.py` — Spatial cropping (SPEC §10): seed token + N nearest by Cα distance
  * `src/deepfold/data/featurize.py` — Featurizer producing all model forward() tensors (D_REF=197)
  * `src/deepfold/data/dataset.py` — DeepFoldDataset + collate_fn + create_dataloader (PyTorch only)
  * Crop schedule: 256→384→512→768 at steps 0/100K/300K/500K (SPEC §10.3)
  * Supports Boltz-style NPZ data format; MSA integration placeholder ready
* Phase 9: SPEC v4.4 updates ✅
  * IFT backward → unrolled differentiable Sinkhorn (plain autograd through K iterations)
  * Atom blocks 3 → 10 (AF3 uses 24; 3 was undersized for sidechain packing)
  * Weight initialization: Xavier + 1/√L depth scaling on W_O (`src/deepfold/model/init.py`)
  * ~226M → ~229M parameters (+2.8M from 7 additional atom blocks) [pre-v4.5]
* Phase 10: Coordinate augmentation + stability fixes ✅
  * `src/deepfold/data/augment.py` — Boltz Algorithm 19: center + uniform SO(3) rotation + Gaussian translation
  * Training: center + rotate + translate (s_trans=1.0Å). Inference: center + rotate (no translation)
  * Same rotation/translation applied to x_atom_true, x_res_true, and ref_pos in c_atom
  * Diffusion COM re-centering after each denoising step (float32 hygiene)
  * FP32 for distogram loss distance computation
  * Fixed phantom gradient in smooth_lddt empty-set return
  * Marginals clamped before log (trunk_block.py)
  * Fixed Heun sampler: removed wrong c_in scaling, correct score = (x - D) / σ
  * Consolidated crop schedule (single source in crop.py)
* Phase 11: SPEC v4.5 — remove diffusion UOT, end-to-end gradient ✅
  * Removed 2 diffusion UOT blocks + associated params (w_dist_diff, geo_gate_proj, pos_bias)
  * h_res NOT frozen before diffusion — L_diff, L_lddt backprop into trunk end-to-end
  * Timestep conditioning: h_cond = h_res + Lin(128→512)(t_emb) at token level
  * μ, ν removed from diffusion interface (only needed for UOT)
  * EMA warmup: copy params directly for first 1000 steps
  * Optimizer: 3 param groups (decay, no-decay, EGNN γ with decay)
  * ~229M → ~220M parameters (−9M from diffusion UOT removal)
* Phase 12: SPEC v4.6 — Diffusion stability, AF3/Boltz-1 alignment ✅
  * FourierEmbedding: random frozen projection cos(2π(t·w+b)) replaces deterministic sinusoidal (AF3 Alg 22)
  * AdaLN: sigmoid(Lin(s))·LN(a,affine=False)+LinNoBias(s) replaces unbounded (1+γ)·LN(a)+β (AF3 Alg 26)
  * AdaLN-Zero gates: sigmoid(Lin(cond,w=0,b=−2)) on attention + transition outputs (AF3 Alg 24)
  * Removed cond_gate1/cond_gate2 (c_atom gating) — replaced by AdaLN-Zero conditioning gates
  * Each atom block starts as near-identity at init (sigmoid(−2)≈0.12 gate factor)
  * c_noise = log(σ/σ_data)·0.25 (was log(σ)/4) — matches AF3/Boltz-1
  * Coordinate input scaling: c_in = 1/√(σ²+σ_data²) replaces 1/σ_data (AF3 Alg 20 line 2)
  * LayerNorm on Fourier features before t_proj (AF3 Alg 21 line 9)
  * LayerNorm on pair features before pair_bias_proj (AF3 Alg 24 line 8)
  * t_proj and pair_bias_proj → bias=False (AF3 uses LinearNoBias)

## Known Boltz-1 Divergences

* No pair representation (biggest change — O(N²) → O(N))
* UOT-Sinkhorn replaces softmax attention in trunk
* Unrolled Sinkhorn backward (not IFT)
* EGNN replaces IPA for structure refinement
* No triangle attention/updates
* Random cycle count 1-5 (Boltz uses fixed recycling)
* Low-rank co-evolution (rank 16) instead of full outer product
* 68-bin position encoding instead of RoPE

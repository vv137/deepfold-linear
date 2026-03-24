# Protein Complex Structure Prediction Model: Full Design Specification v4.6

---

## 0. Notation and Definitions

| Symbol | Definition |
|---|---|
| N | Number of tokens (after crop). Token = residue (protein), nucleotide (RNA/DNA), or atom/group (ligand) |
| N_atom | Number of atoms (after crop) |
| S | MSA depth (128) |
| d | Token representation dimension (512) |
| d_msa | MSA representation dimension (64) |
| d_atom | Atom representation dimension (128) |
| d_pair | Atom pair representation dimension (16) |
| H_res | Token-level UOT attention heads (16) |
| d_h | Per-head dimension = d / H_res = 32 |
| H_msa | MSA attention heads (8) |
| H_atom | Atom attention heads (4) |
| r | Co-evolution rank (16) |
| L | Number of Token UOT blocks in trunk (48) |
| L_atom | Number of Atom blocks per diffusion step (10) |
| r_0 | Geometry characteristic distance (10 Å, fixed) |
| d_low | Distogram low-rank interaction dimension (64) |
| σ_data | EDM data noise scale (16 Å) |
| token_idx | (N_atom,) int — maps each atom to its token |
| Lin(a→b) | nn.Linear(a, b, bias=True) — standalone projections (input embed, coord output, loss heads) |
| Lin_nb(a→b) | nn.Linear(a, b, bias=False) — projections after LayerNorm (attention Q/K/V/G/O, SwiGLU) |
| LN(a) | nn.LayerNorm(a) — with learnable γ, β |
| LN_Lin(a→b) | LayerNorm(a) followed by Linear(a, b, bias=False). LN β provides the bias; Linear bias redundant |

**Bias convention**: Projections that follow a LayerNorm use `bias=False` because LN's learnable β already provides the additive shift. This includes trunk attention projections (W_Q, W_K, W_V, W_G, W_O), SwiGLU projections, and LN_Lin composites. **Exception**: In diffusion AtomBlock (AF3 Alg 24), W_Q uses `bias=True` per AF3 convention. Standalone projections that don't follow LN (input embedding, coordinate output, loss heads, atom-to-token encoder output) use `bias=True`.

**Terminology**: This document uses "token" as the basic unit of the single-track representation. For proteins, one token = one residue. For RNA/DNA, one token = one nucleotide. For ligands/ions/water, one token = one atom or functional group. Legacy references to "residue" in variable names (e.g., h_res) are retained for readability but apply to all token types.

**Notation shorthand**: For readability, this document writes `Lin(a→b)` uniformly in pseudocode. The bias convention (True/False) is determined by context — after LN means no bias. Standalone means bias. Implementers should follow the bias convention table above.

---

## 1. Design Motivation

### 1.1 Limitations of Existing Models

**Memory**: AF2/AF3 pair representation z_ij ∈ R^{N×N×d} causes O(N²) memory. Tens of GB at 1000 residues.

**Softmax attention**: Row-stochastic constraint forces probability mass onto all residues. In practice, 10–20% of residues determine structure. Antibody–antigen systems where a handful of CDR–epitope residues dominate binding are especially affected.

**Co-evolution**: Single representation alone cannot capture 2nd-order co-evolution statistics. Single-sequence models struggle with complex prediction.

### 1.2 Core Design Decisions

| Decision | Rationale |
|---|---|
| Remove pair representation | O(N²) → O(N) memory (after tiling) |
| UOT-Sinkhorn attention | Automatic hot-spot focus, sparse transport |
| EGNN coordinate update in UOT blocks | SE(3) equivariant structure refinement, no augmentation needed for trunk |
| Diffusion module for sampling | Stochastic multi-modal sampling; AF3-style augmentation (lightweight, local) |
| Remove triangle attention | Unnecessary with explicit coordinates |
| Low-rank co-evolution (rank 16) | Capture multi-modal 2nd-order statistics without pair rep |
| x_res → cost (geometry-aware UOT) | Geometry modulates attention; coordinates refine through EGNN each block |
| Transport-weighted average | Length-invariant output |
| Unrolled Sinkhorn backward | Exact gradient for K-iteration computation; IFT gives wrong gradients at incomplete convergence |

### 1.3 Core Contributions

1. **UOT-Flash Attention**: Biology-aware sparse transport without pair representation.
2. **O(N) memory architecture**: No N×N stored matrices. Reference implementation uses O(N²) for co-evolution and distogram; tiled versions planned.
3. **Low-rank co-evolution aggregation (rank 16)**: 2nd-order co-evolution statistics compressed into single representation via rank-16 outer product; r-dimensional co-evolution profile drives per-head marginal bias; co-evolving positions become UOT hubs through ν marginals.
4. **Bond-aware position encoding**: 68-bin scheme unifying sequence separation, chain identity, and covalent bonds.
5. **EGNN coordinate update in UOT blocks**: Transport-weighted centroid displacement with per-head signed γ. SE(3) equivariant by construction — no augmentation in trunk. Coordinates refine across 48 blocks × 3 cycles. Fused into Flash-Sinkhorn kernel at ~10% extra cost.
6. **Geometry-aware UOT**: x_res → metric cost; coordinates improve each block via EGNN → geometry cost becomes increasingly reliable across blocks.
7. **Unrolled Sinkhorn backward**: Standard autograd through K iterations; exact gradient for actual computation, not theoretical fixed point.
8. **Transport-weighted average**: Length-invariant, ν_i controls how much other residues attend to position i.

### 1.4 Role of Marginals μ and ν

In Sinkhorn OT, T*_ij = u_i K_ij v_j. The transport-weighted average normalizes by row sum:

    o_i = sigmoid(G_i) ⊙ (Σ_j T*_ij V_j) / (Σ_j T*_ij + ε)

This makes o_i scale-invariant w.r.t. μ_i — the output magnitude is O(1) regardless of μ_i.

However, the **column marginal ν_i** controls how much mass arrives at position i from other positions. When ν_i is large, v_i grows through Sinkhorn iterations, and Σ_j T*_ji increases — **other residues allocate more transport to position i**. Co-evolving functional sites get large ν_i through the co-evolution bias, automatically becoming UOT hubs.

The **row marginal μ_i** controls how much mass leaves position i. Large μ_i means position i exports more total transport, distributing its information more broadly. Together, μ and ν allow asymmetric hub behavior: a position can be a strong receiver (large ν) without being a strong sender (μ can differ).

In this architecture, μ and ν share the co-evolution bias c̄_i but have separate learned projections W_μ and W_ν, allowing partial independence.

### 1.5 Honest Complexity Statement

The architecture is **designed** for O(N) memory: no component inherently requires N×N storage. The **reference PyTorch implementation** uses Python-level tiling (tile_size=64) for co-evolution aggregation and distogram loss, keeping their peak memory at O(tile²·d) ≈ constant. The remaining O(N²) bottleneck is `cdist` in Residue UOT blocks, which will be eliminated by the Flash-Sinkhorn kernel (computing distances on-the-fly per tile). Until that kernel is implemented, the reference implementation is O(N²) memory due to cdist only.

---

## 2. Full Architecture Overview

```
Input: token_types, MSA (protein-only), bond_matrix, reference conformer

[Input Embedding]
  h_res ← Lin(38→512)(cat(token_type, profile, del_mean, has_msa))  (N, 512)
  c_atom ← Lin(D_ref→128)(ref_features)                             (N_atom, 128) frozen
  p_lm   ← atom_pair_embed(ref_conformer)                           (local, 16)   frozen
  m      ← Lin(34→64)(cat(msa_restype, has_del, del_val))           (S, N_prot, 64)

[Atom-to-Token Encoder — runs once]
  c_atom → local self-attn (1 block, pair bias p_lm) → SwiGLU
  → Lin(128→512) → scatter_mean by token_idx → atom_agg            (N, 512)
  h_res ← h_res + atom_agg

[Trunk — random 1–5 recycling passes (training), configurable (inference)]
  x_res ← randn(N, 3) * σ_data, centered                           initial noise
  num_cycles ~ Uniform{1..5} (training) or configurable (inference)
  All but last cycle: torch.no_grad. Last cycle: backprop + checkpoint.

  for cycle in [0 .. num_cycles-1]:
    MSA blocks × 4 → m, h_res, μ, ν                                 (invariant, no coords)
    Token UOT+EGNN blocks × 48:
      Each block:
        UOT attention → h_res update                                 (invariant)
        EGNN → x_res update: x += Σ_h γ_h · (x − T_norm^h @ x)     (equivariant)
      Geometry cost uses latest x_res each block
    Re-center x_res (once per cycle, float32 hygiene)

  freeze h_res, μ, ν, x_res
  L_trunk_coord = smooth_lddt(x_res, x_0)                           direct EGNN supervision

[Diffusion — 200 steps (inference), sampled σ (training)]
  AF3-style augmentation for SE(3).
  for each denoise step:
    x_res ← scatter_mean(x_atom, token_idx)
    h_step ← diffusion UOT blocks × 2 (frozen μ,ν, σ-gated geometry)
    q ← c_atom + Lin(3→128)(x_atom / σ_data)
    Atom blocks × 10 → q
    Δx ← zero_init_Lin(LN(q))
    x_atom ← x_atom + c_out(σ) · Δx

Output: denoised x_atom
```

---

## 3. Input Representations

### 3.1 Token Single Representation

```python
# Token type: 4 broad classes
# 0: protein   1: RNA   2: DNA   3: ligand (ions, water, modifications, etc.)
token_type_onehot = one_hot(token_type, 4)            # (N, 4)

# MSA features: defined for protein (and optionally RNA) tokens, zero for others
profile  = msa_frequencies                             # (N, 32), zero-padded for non-protein
del_mean = deletion_mean                               # (N, 1), zero for non-protein
has_msa  = (token_type <= 1).float().unsqueeze(-1)     # (N, 1), 1 for protein/RNA

h_res = Lin(38 → 512)(                                 # (N, 512)
    cat(token_type_onehot, profile, del_mean, has_msa, dim=-1)
)
```

**Design**: The token type vocabulary is deliberately minimal (4 classes). Fine-grained chemical information comes from the Atom-to-Token Encoder (§3.5), not from a large residue-type vocabulary. The `has_msa` flag lets the network distinguish "zero MSA because ligand" from "zero MSA because orphan protein."

**Ligand atom-level tokens**: Non-polymer molecules (ligands, water, ions) are tokenized at the atom level — one token per atom. This means the trunk's EGNN operates at all-atom resolution for these entities, natively predicting ligand atom coordinates. The 68-bin bond matrix (§4) encodes the exact 2D molecular graph of the ligand through bins 66/67 (covalent bonds). Repulsive EGNN heads (γ < 0) maintain proper Van der Waals separation and bond geometry.

### 3.2 MSA Matrix

```python
msa_feat = cat(
    one_hot(msa_restype, 32),                 # (S, N_prot, 32)
    has_deletion,                              # (S, N_prot, 1)
    deletion_value                             # (S, N_prot, 1)
)                                             # (S, N_prot, 34)
m = Lin(34 → 64)(msa_feat)                    # (S, N_prot, 64)
```

MSA is defined only for protein (and optionally RNA/DNA) tokens. N_prot ≤ N is the number of tokens with MSA data. The MSA module (§6) operates on these tokens only; non-MSA tokens receive no co-evolution signal and get uniform UOT marginals.

### 3.3 Atom Single Representation (Frozen)

```python
ref_feat = cat(
    ref_pos,          # (N_atom, 3)   reference conformer positions
    ref_charge,       # (N_atom, 1)
    ref_mask,         # (N_atom, 1)
    ref_element,      # (N_atom, 128) one-hot
    ref_atom_name     # (N_atom, 64)  one-hot
)                     # (N_atom, D_ref)
c_atom = Lin(D_ref → 128)(ref_feat)           # (N_atom, 128)
```

Frozen during diffusion — serves as residual reference. Encodes per-atom chemistry independent of structural context. Follows AF3 Algorithm 5.

### 3.4 Atom Pair Representation (Frozen)

```python
# Reference conformer intra-token distances and connectivity
d_lm = ref_pos[l] - ref_pos[m]               # displacement vectors
v_lm = (same_token[l, m]).float()             # validity mask (same token)

p_lm = (Lin(3 → 16)(d_lm) * v_lm
       + Lin(1 → 16)(1/(1 + ||d_lm||²)) * v_lm
       + Lin(1 → 16)(v_lm))                   # (local_pairs, 16)
```

Encodes intra-token local geometry only. Frozen during diffusion. Follows AF3 Algorithm 5.

### 3.5 Atom-to-Token Encoder

Aggregates atom-level chemical information into the token representation. Runs once before recycling. For standard amino acids, this adds sidechain geometry details beyond the token type. For ligands, modified residues, and ions, this is the **primary** source of chemical information — the token type only says "ligand."

Follows AF3's AtomAttentionEncoder pattern: local self-attention with pair bias lets atoms within a token (and nearby tokens) exchange information before aggregation.

```python
def atom_to_token_encoder(c_atom, p_lm, token_idx, N):
    """
    Single block of local atom self-attention, then mean-pool to token level.
    Runs once before recycling. ~400K params.
    
    c_atom:    (N_atom, 128)  frozen atom reference features
    p_lm:      (local, 16)    frozen intra-token atom pair features  
    token_idx: (N_atom,)      maps each atom to its token
    N:         int             number of tokens
    
    Returns:   (N, 512)       per-token atom summary
    """
    q = c_atom                                                  # (N_atom, 128)
    
    # --- One block of gated local self-attention ---
    q_n = LN(q)                                                 # (N_atom, 128)
    Q = Lin(128, 128)(q_n).view(-1, 4, 32)                     # (N_atom, 4, 32)
    K = Lin(128, 128)(q_n).view(-1, 4, 32)
    V = Lin(128, 128)(q_n).view(-1, 4, 32)
    G = Lin(128, 128)(q_n).view(-1, 4, 32)
    
    b = Lin(16, 4)(p_lm)                                       # intra-token pair bias
    A = softmax(einsum('ihd,jhd->hij', Q, K) / sqrt(32) + b,
                dim=-1, window=32)
    out = sparse_einsum(A, V)                                   # (N_atom, 4, 32)
    out = rearrange(out, 'n h d -> n (h d)')                    # (N_atom, 128)
    q = q + sigmoid(rearrange(G, 'n h d -> n (h d)')) * Lin(128, 128)(out)
    
    # --- SwiGLU transition ---
    q_n = LN(q)                                                 # (N_atom, 128)
    q = q + Lin(512, 128)(SiLU(Lin(128, 512)(q_n)) * Lin(128, 512)(q_n))
    
    # --- Project and aggregate to token level ---
    atom_feat = Lin(128, 512)(q)                                # (N_atom, 512)
    
    # scatter_mean: sum then divide by count per token
    agg = torch.zeros(N, 512, device=q.device)
    count = torch.zeros(N, 1, device=q.device)
    agg.scatter_add_(0, token_idx[:, None].expand_as(atom_feat), atom_feat)
    count.scatter_add_(0, token_idx[:, None],
                       torch.ones(N_atom, 1, device=q.device))
    return agg / count.clamp(min=1)                             # (N, 512)
```

Integration into input embedding:

```python
h_res = Lin(38, 512)(cat(token_type, profile, del_mean, has_msa))
h_res = h_res + atom_to_token_encoder(c_atom, p_lm, token_idx, N)
```

---

## 4. Position / Bond Encoding

### 4.1 Design Intent

RoPE breaks under multi-chain and spatial cropping. Infinity masking is numerically unstable. Instead: 68-bin lookup unifying sequence separation, chain identity, and covalent bonds. Computed online — no N×N storage.

**Bond matrix scope**: The bond_matrix must include **all** covalent bonds — not just peptide backbone bonds. This covers: peptide bonds, disulfide bonds, nucleotide phosphodiester bonds, ligand internal bonds, covalent protein-ligand bonds (e.g., covalent inhibitors), and any other covalent connectivity. The 68-bin scheme handles all of these uniformly through bins 66 (same chain) and 67 (cross chain).

### 4.2 Bin Definition

```
bin(i, j) =
  clip(g_i − g_j, −32, 32) + 32     if same chain, no bond     → bins [0, 64]
  65                                   if cross-chain, no bond
  66                                   if covalent bond, same chain
  67                                   if covalent bond, cross-chain
```

g_i, g_j: global residue indices, preserved after cropping.

### 4.3 Parameters

```
w_rel_res:  (H_res, 68)  — Residue UOT position bias.    Init: zeros. AdamW decay.
w_rel_msa:  (H_msa, 68)  — MSA attention position bias.   Init: zeros. AdamW decay.
```

### 4.4 Branchless Online Computation

```c
int same = (chain_id[i] == chain_id[j]);
int bond = bond_matrix[i * N + j];
int sep  = clamp(global_idx[i] - global_idx[j], -32, 32) + 32;
int bin  = sep * (1 - bond) * same
         + 65  * (1 - bond) * (1 - same)
         + 66  * bond       * same
         + 67  * bond       * (1 - same);
```

No warp divergence. w_rel cached in shared memory for kernel fusion.

---

## 5. Recycling Loop (Random Cycle Count)

### 5.1 Design Intent

The trunk runs a **randomly sampled** number of recycling passes (1 to max_cycles). Each cycle: (1) refine h_res and marginals with MSA blocks (invariant, sequence/co-evolution), (2) refine h_res (invariant) and x_res (equivariant) with UOT+EGNN blocks. Only the last cycle backpropagates; all earlier cycles run under `torch.no_grad()`.

**Random cycle count** makes the architecture an anytime algorithm. The network must produce good output whether it gets 1 cycle (48 UOT+EGNN blocks from noise) or 5 cycles (240 blocks of refinement). Since γ is shared across cycles, the network learns step sizes that work for both aggressive folding and gentle refinement. At inference, the user chooses: 1 cycle (fast), 3 (standard), 5+ (difficult targets).

**No coordinate embedding into h_res**: Geometry enters through UOT cost → T* → attention output.

**No augmentation for trunk**: EGNN updates use only relative vectors — SE(3) equivariant by construction.

### 5.2 Trunk Loop

```python
def trunk_forward(seq_features, msa_features, c_atom, p_lm,
                  bond_matrix, chain_id, global_idx, token_idx,
                  x_0_res=None):  # ground truth for L_trunk_coord (training only)

    # ---- Input embedding ----
    h_res = Lin(38, 512)(cat(token_type, profile, del_mean, has_msa))   # (N, 512)
    h_res = h_res + atom_to_token_encoder(c_atom, p_lm, token_idx, N)   # + atom info
    m     = Lin(34, 64)(msa_features)                                    # (S, N_prot, 64)
    mu    = torch.ones(H_res, N) / N
    nu    = torch.ones(H_res, N) / N

    # Initial coordinates: random noise, centered
    x_res = torch.randn(N, 3) * sigma_data
    x_res = x_res - x_res.mean(dim=0, keepdim=True)

    # ---- Sample cycle count ----
    if training:
        num_cycles = randint(1, max_cycles + 1)       # e.g., uniform over {1, 2, 3, 4, 5}
    else:
        num_cycles = inference_cycles                  # default 3, configurable

    log_u_carry, log_v_carry = None, None

    for cycle in range(num_cycles):
        is_last = (cycle == num_cycles - 1)

        # ---- MSA blocks × 4 (invariant, no coordinates) ----
        for b in range(4):
            m, h_res, mu_new, nu_new = msa_block(m, h_res, mu, nu, block_idx=b)

        if cycle > 0:
            mu = softmax(log(mu_new + 1e-8) + 0.5 * log(mu + 1e-8), dim=-1)
            nu = softmax(log(nu_new + 1e-8) + 0.5 * log(nu + 1e-8), dim=-1)
        else:
            mu, nu = mu_new, nu_new

        # ---- Token UOT+EGNN blocks × 48 ----
        log_u_prev = log_u_carry
        log_v_prev = log_v_carry

        for block in residue_uot_blocks:
            if is_last:
                h_res, x_res, log_u_prev, log_v_prev = checkpoint(
                    block, h_res, x_res, mu, nu, log_u_prev, log_v_prev
                )
            else:
                with torch.no_grad():
                    h_res, x_res, log_u_prev, log_v_prev = block(
                        h_res, x_res, mu, nu, log_u_prev, log_v_prev
                    )

        # ---- Re-center coordinates (once per cycle, not per block) ----
        # EGNN is exactly translation-equivariant; re-centering is float32 hygiene
        x_res = x_res - x_res.mean(dim=0, keepdim=True)

        if not is_last:
            h_res = h_res.detach()
            x_res = x_res.detach()
            mu = mu.detach()
            nu = nu.detach()
            log_u_carry = log_u_prev.detach()
            log_v_carry = log_v_prev.detach()
        else:
            log_u_carry = log_u_prev

    # ---- Losses (last cycle only, in graph) ----
    L_disto = distogram_loss(h_res, x_0_res)
    L_trunk_coord = smooth_lddt(x_res, x_0_res) if x_0_res is not None else 0.0

    return h_res, mu, nu, x_res, L_disto, L_trunk_coord
```

### 5.3 Cycle Count Distribution

| Parameter | Value |
|---|---|
| max_cycles (training) | 5 |
| Sampling distribution | Uniform over {1, 2, 3, 4, 5} |
| inference_cycles (default) | 3 |
| inference_cycles (fast) | 1 |
| inference_cycles (difficult) | 5+ |

The uniform distribution ensures the network trains equally on all cycle counts. Alternative: geometric distribution (P(k) ∝ 0.5^(k-1)) to bias toward fewer cycles, teaching efficiency. Hyperparameter to tune.

### 5.4 Recycling Design Rationale

| Mechanism | Purpose |
|---|---|
| Random cycle count | Anytime algorithm; works with 1–5+ cycles; no fixed schedule |
| no_grad on all but last cycle | Memory: always 52 blocks in graph regardless of cycle count |
| EGNN updates every UOT block | Coordinates refine continuously (48 updates/cycle) |
| Re-centering once per cycle | Float32 hygiene; EGNN is exactly translation-equivariant in exact arithmetic |
| L_trunk_coord on final x_res | Direct gradient to all EGNN γ parameters; fixes dead-gradient on last block |
| No coordinate embedding into h_res | Geometry enters h_res through UOT cost → T* → attention output |
| No augmentation | EGNN is SE(3) equivariant by construction |
| Marginal carry (log-space blend) | Preserves transport patterns from previous cycle |
| Sinkhorn warm-start carry | log_u, log_v from cycle k → cycle k+1 (detached except last) |

---

## 6. MSA Module

### 6.1 Design Intent

- **Scope**: MSA processing operates on protein (and optionally RNA/DNA) tokens only. Non-MSA tokens (ligands, ions) receive no co-evolution signal and get uniform UOT marginals. The MSA block internally works on the protein token subset; results are scattered back to the full token set.
- **Row-wise attention**: Within-sequence residue interactions with position bias.
- **Column weighted mean**: Learned aggregation capturing 1st-order co-evolution.
- **Low-rank co-evolution aggregation**: c_ij = (1/S) Σ_s U_si^T V_sj captures 2nd-order co-evolution statistics. Rank r=16 allows multiple co-evolution modes (active site, allosteric site, interface, fold stability). The r-dimensional co-evolution vector per pair is reduced to a scalar weight for aggregation but retained as a full profile for marginal bias.
- **Marginal update**: Per-layer, per-head scalar coefficients gate the co-evolution signal. The r-dimensional co-evolution profile c̄_i ∈ R^16 projects to per-head marginal bias via Lin(16→H_res), letting each head attend to different co-evolution modes. Non-MSA tokens retain uniform marginals.

### 6.2 Full Block

```python
def msa_block(m, h_res, mu, nu, block_idx):
    """
    m:         (S, N_prot, 64)   MSA for protein tokens only
    h_res:     (N, 512)          all tokens
    mu:        (H_res, N)        all tokens
    nu:        (H_res, N)        all tokens
    block_idx: int (0–3), selects per-block alpha_coevol coefficients
    """

    # ---- 1. Single → MSA injection (protein tokens only) ----
    # protein_mask: (N,) bool — True for tokens with MSA data
    h_prot = h_res[protein_mask]                                # (N_prot, 512)
    m = m + Lin(512 → 64)(h_prot)[None]                        # broadcast (1, N_prot, 64)

    # ---- 2. Row-wise attention (per sequence, over protein residues) ----
    m_n = LN(m)                                                 # (S, N_prot, 64)
    Q = Lin(64 → 64)(m_n)                                      # (S, N, 64) → (S, N, H_msa=8, 8)
    K = Lin(64 → 64)(m_n)                                      # same
    V = Lin(64 → 64)(m_n)                                      # same
    G = Lin(64 → 64)(m_n)                                      # same

    # Position bias: (H_msa, 68) indexed by bin(i,j) → (H_msa, N, N)
    # Computed on-the-fly, not stored
    b = w_rel_msa[bin_ij]                                       # (H_msa, N, N)

    # Standard gated attention per head per sequence
    A = softmax(
        einsum('shid,shjd->shij', Q, K) / sqrt(8) + b[None],
        dim=-1
    )                                                           # (S, H, N, N)
    att_out = einsum('shij,shjd->shid', A, V)                  # (S, H, N, 8)
    att_out = rearrange(att_out, 's n h d -> s n (h d)')        # (S, N, 64)
    m = m + sigmoid(rearrange(G, 's n h d -> s n (h d)')) * Lin(64 → 64)(att_out)

    # ---- 3. Column weighted mean → single rep (protein → all tokens) ----
    m_n = LN(m)                                                 # (S, N_prot, 64)
    alpha = softmax(Lin(64 → 1)(m_n), dim=0)                   # (S, N_prot, 1) softmax over S
    col_agg = (alpha * m_n).sum(dim=0)                          # (N_prot, 64)
    h_prot_update = Lin(64 → 512)(col_agg)                     # (N_prot, 512)
    h_res[protein_mask] = h_res[protein_mask] + h_prot_update   # scatter back

    # ---- 4. Low-rank co-evolution aggregation (rank 16, tiled) ----
    # c_ij ∈ R^r: multi-modal co-evolution vector per pair
    # Scalar weight for aggregation, full r-dim profile for marginal bias
    m_n = LN(m)                                                 # (S, N, 64)
    U = Lin(64 → 16)(m_n)                                      # (S, N, 16)
    V_ = Lin(64 → 16)(m_n)                                     # (S, N, 16)

    # Precompute value track for aggregation
    h_coevol = LN_Lin(512 → 512)(h_res)                        # (N, 512)

    # Tiled: peak memory O(tile² · r + tile · d) per tile
    TILE = 64
    h_agg = torch.zeros(N, 512)
    c_bar_accum = torch.zeros(N, 16)                            # r-dim profile accumulator

    for i0 in range(0, N, TILE):
        ie = min(i0 + TILE, N)
        U_i = U[:, i0:ie, :]                                   # (S, ti, 16)

        for j0 in range(0, N, TILE):
            je = min(j0 + TILE, N)
            V_j = V_[:, j0:je, :]                              # (S, tj, 16)

            # Co-evolution vector per pair: (ti, tj, 16)
            c_tile = einsum('sir,sjr->ijr', U_i, V_j) / S     # (ti, tj, 16)

            # Scalar weight for aggregation: project r→1
            w_tile = sigmoid(Lin(16 → 1)(c_tile).squeeze(-1))  # (ti, tj)
            h_agg[i0:ie] += w_tile @ h_coevol[j0:je]           # (ti, 512)

            # r-dim profile for marginal bias
            c_bar_accum[i0:ie] += c_tile.sum(dim=1)            # (ti, 16)

    h_res = h_res + Lin(512 → 512)(h_agg)                      # (N, 512)
    c_bar = c_bar_accum / N                                     # (N, 16) per-position profile

    # ---- 5. Marginal update (per-layer, per-head gated) ----
    # alpha_coevol: (4, H_res) — per MSA block, per head scalar coefficients
    # Zero-init + AdamW decay: co-evolution bias must earn its influence
    # Marginals computed over ALL tokens; co-evolution bias only for protein tokens
    mu_logit = Lin(512 → H_res)(h_res)                         # (N, H_res)
    nu_logit = Lin(512 → H_res)(h_res)                         # (N, H_res)

    # r-dim co-evolution profile → per-head bias (protein tokens only)
    # Non-protein tokens: coevol_bias = 0, so they get marginals from h_res alone
    coevol_bias = torch.zeros(N, H_res, device=h_res.device)
    coevol_bias[protein_mask] = Lin(16 → H_res)(c_bar[protein_mask])
    bias = (alpha_coevol[block_idx][:, None] * coevol_bias.T)  # (H_res, N)

    mu = softmax((mu_logit.T + bias), dim=-1)                  # (H_res, N)
    nu = softmax((nu_logit.T + bias), dim=-1)                  # (H_res, N)

    # ---- 6. SwiGLU transition ----
    m_n = LN(m)                                                 # (S, N, 64)
    gate  = Lin(64 → 256)(m_n)                                  # (S, N, 256)
    value = Lin(64 → 256)(m_n)                                  # (S, N, 256)
    m = m + Lin(256 → 64)(SiLU(gate) * value)                   # (S, N, 64)

    # ---- 7. Row dropout ----
    # Drop entire MSA rows (sequences) during training
    if training:
        mask = torch.bernoulli(torch.full((S, 1, 1), 0.85))    # (S, 1, 1)
        m = m * mask / 0.85

    return m, h_res, mu, nu
```

### 6.3 Why Standard Softmax in MSA Row Attention

MSA row-wise attention operates within individual homologous sequences where every position is informative — there are no "irrelevant" residues within a single alignment row. The sparse-transport motivation for UOT applies to cross-chain and long-range residue interactions, not within-sequence MSA processing. Standard softmax is appropriate and cheaper here.

---

## 7. UOT-Flash Attention

### 7.1 Cost Matrix

For each head h, between residue positions i and j:

```
C_ij^(h) = −(LN(Q_i)^T LN(K_j)) / sqrt(d_h)       content term
          + w_rel^(h)[bin(i,j)]                       position/bond term
          + w_dist^(h) · d_ij / (r_0 + d_ij)          geometry term
```

where d_ij = ||x_res_i − x_res_j||₂.

**LN on Q, K**: LayerNorm normalizes each head's query/key vector to approximately unit norm. This bounds the content term to approximately [−1, 1] after division by sqrt(d_h). This is critical for Sinkhorn stability — all three cost terms are O(1), so ε=1.0 produces meaningful entropy regularization.

**Geometry term design**:

- f(d) = d/(r_0 + d) is concave, f(0) = 0, bounded in [0, 1).
- f ∘ d satisfies the triangle inequality (metric).
- r_0 = 10 Å fixed: roughly contact range (~8 Å) scale.
- w_dist is per-block, per-head: `w_dist = sigmoid(w_dist_logit)`, bounded to (0, 1). `w_dist_logit` initialized to -2.0 (sigmoid ≈ 0.12, weak geometry initially). No weight decay on `w_dist_logit` (sigmoid already bounds it).
- Per-block, per-head w_dist allows multi-scale behavior: some heads/layers develop strong geometry sensitivity, others stay weak. The transport-weighted average across heads provides implicit noise robustness.
- In the trunk: coordinates have variable quality across recycling noise levels. The per-head diversity handles this — geometry-sensitive heads contribute when coordinates are good, geometry-insensitive heads carry the load when coordinates are noisy.
- In the diffusion module: σ-conditioned geometry gating explicitly modulates the geometry term (see §9.2).

**Scale budget** (all terms O(1)):

- Content: LN → approximately [−3, 3]
- w_rel[bin]: AdamW decay → approximately [−1, 1]
- w_dist · f: sigmoid(logit) · f ∈ [0,1) → [0, 1)

### 7.2 Transport Problem

```
T*^(h) = argmin_{T ≥ 0}  ⟨C^(h), T⟩
                         + ε^(h) · KL(T ‖ μ^(h) ⊗ ν^(h))
                         + λ · KL(T1 ‖ μ^(h))
                         + λ · KL(T^T 1 ‖ ν^(h))
```

**Multi-scale ε (fixed per head, not learned)**:

```python
# Registered as buffer — no gradients
self.register_buffer('eps', torch.tensor([
    0.5, 0.5, 0.5, 0.5,     # heads 0–3:   sparse (contacts, hot-spots)
    1.0, 1.0, 1.0, 1.0,     # heads 4–7:   balanced (structural reasoning)
    2.0, 2.0, 2.0, 2.0,     # heads 8–11:  smooth (secondary structure)
    4.0, 4.0, 4.0, 4.0,     # heads 12–15: diffuse (allostery, global)
]))
```

**Empirical convergence (cold-start, measured across N=128–512)**:

| K_iter | Residual | Status |
|---|---|---|
| 4 | 26–34% | Unusable |
| 7 | 2.3–3.0% | Marginal |
| 10 | 0.2–0.3% | Decent |
| 14 | ~1e-4 | Good |
| 20 | ~1e-6 | Converged |

| Head group | ε | κ = λ/(λ+ε) | Transport character |
|---|---|---|---|
| 0–3 | 0.5 | 0.67 | Sparse: mass flows to specific contacts |
| 4–7 | 1.0 | 0.5 | Balanced: standard structural interactions |
| 8–11 | 2.0 | 0.33 | Smooth: broad environment sensing |
| 12–15 | 4.0 | 0.2 | Diffuse: global state, allostery |

| Parameter | Value | Rationale |
|---|---|---|
| ε | Fixed per-head [0.5, 1.0, 2.0, 4.0] × 4 | Multi-scale transport without gradient complexity |
| λ | 1.0 (uniform all heads) | κ ranges 0.2–0.67 |
| K | 20 (all blocks, uniform) | Converges to ~1e-6 residual; no warm/cold distinction; unrolled backprop needs correct forward output |
| K_max_infer | 20 (or adaptive tol=1e-4 → ~14 iterations) | Inference can use early stopping for ~30% speedup |
| tol (inference) | 1e-4 | Early stopping threshold at inference |

**Why K=20 uniform**: With unrolled differentiable backward, gradient quality depends directly on forward output quality. K=4 gives 30% residual — the network trains on a substantially wrong transport plan. K=20 reaches ~1e-6 residual, giving a correct forward output and therefore correct gradients. The FlashSinkhorn Triton kernel (§18) makes K=20 affordable — fused IO-aware streaming achieves 32× speedup over naive PyTorch per iteration.

**No warm/cold distinction**: With K=20, all blocks converge fully regardless of initialization quality. Warm-starting from the previous block's log_u, log_v still helps (convergence may be reached by K=10–14 with warm start), but K=20 provides a safe ceiling. At inference, adaptive early stopping with tol=1e-4 typically terminates at K≈14.

**Why fixed, not learned**: Learnable ε creates gradient complications — ε enters both the kernel scaling (log_K = -C/ε) and the damping factor (κ = λ/(λ+ε)), producing competing gradient components. Per-head κ varies with ε, making convergence analysis ε-dependent. Fixed ε avoids all of this while still providing multi-scale transport. The network adapts through its existing per-head parameters (W_Q, W_K, W_V, W_G, W_O, γ, w_dist) — ample capacity for head specialization.

**Why λ=1.0 uniform**: At K=20, convergence is not a constraint — all ε values converge fully. λ=1.0 gives κ ranging from 0.67 (sparse heads, strong marginal enforcement) to 0.2 (diffuse heads, weak enforcement). This is the right physical behavior.

### 7.3 Sinkhorn Iterations (Log-Domain, Per-Head ε)

```python
def sinkhorn_log_domain(C, log_mu, log_nu, eps, lam, K, log_u_init=None, log_v_init=None):
    """
    C:      (H, N, N) cost matrix
    log_mu: (H, N) log row marginal
    log_nu: (H, N) log column marginal
    eps:    (H,) fixed per-head entropic regularization (buffer, no grad)
    lam:    scalar marginal penalty
    Returns: log_u (H, N), log_v (H, N)
    """
    kappa = lam / (lam + eps)                          # (H,) fixed per-head damping

    log_u = log_u_init if log_u_init is not None else torch.zeros(H, N)
    log_v = log_v_init if log_v_init is not None else torch.zeros(H, N)

    log_K = -C / eps[:, None, None]                    # (H, N, N) per-head scaling

    for k in range(K):
        # Row update
        log_u = kappa[:, None] * (log_mu - logsumexp(log_K + log_v[:, None, :], dim=-1))
        # Column update
        log_v = kappa[:, None] * (log_nu - logsumexp(log_K + log_u[:, :, None], dim=-2))

    return log_u, log_v
```

**Per-head broadcasting**: Each head's tile uses its own fixed ε_h — a single scalar read from shared memory. κ_h is also fixed. Zero extra memory, zero extra compute. No gradient through ε — backward only handles ∂L/∂C, ∂L/∂μ, ∂L/∂ν via unrolled Sinkhorn autograd.

### 7.4 Output: Transport-Weighted Average (Numerically Stable)

```python
def uot_attention_output(V, G, log_u, log_v, C, eps):
    """
    After Sinkhorn converges, compute gated transport-weighted average.
    Uses running-max trick (same as FlashAttention) for numerical stability.
    
    Reference implementation (non-tiled):
    """
    log_K = -C / eps                                                # (H, N, N)

    # Per-row running max for numerical stability
    log_score = log_u[:, :, None] + log_K + log_v[:, None, :]      # (H, N, N)
    row_max = log_score.max(dim=-1, keepdim=True).values            # (H, N, 1)

    T = exp(log_score - row_max)                                    # (H, N, N) safe: ≤ 1
    T_sum = T.sum(dim=-1, keepdim=True)                             # (H, N, 1)
    O_avg = einsum('hnm,hmd->hnd', T, V) / (T_sum + 1e-6)         # (H, N, d_h)

    # Gating
    o = sigmoid(G) * O_avg                                          # (H, N, d_h)
    o = rearrange(o, 'h n d -> n (h d)')                            # (N, 512)
    return Lin(512 → 512)(o)                                         # (N, 512)
```

**Numerical stability**: Without the row_max subtraction, `exp(log_u + log_K + log_v)` can overflow (large positive exponents) or underflow (large negative). The running-max trick subtracts the per-row maximum before exp, keeping all values ≤ 1. The division by T_sum cancels the subtracted constant. In the tiled (Flash) implementation, this uses the online logsumexp accumulator identical to FlashAttention.

**Length invariance**: Division by T_sum makes output O(1) regardless of N. No gradient dilution.

### 7.5 Backward: Unrolled Differentiable Sinkhorn

Standard PyTorch autograd differentiates through the K Sinkhorn iterations directly. No custom backward, no adjoint solve, no fixed-point assumption.

```python
def sinkhorn_differentiable(C, log_mu, log_nu, eps, lam, K, log_u_init, log_v_init):
    """Differentiable Sinkhorn. Standard autograd handles backward."""
    kappa = lam / (lam + eps)                          # (H,) per-head
    log_u = log_u_init.clone() if log_u_init is not None else torch.zeros_like(log_mu)
    log_v = log_v_init.clone() if log_v_init is not None else torch.zeros_like(log_nu)
    log_K = -C / eps[:, None, None]                    # (H, N, N)

    for k in range(K):
        log_u = kappa[:, None] * (log_mu - torch.logsumexp(log_K + log_v[:, None, :], dim=-1))
        log_v = kappa[:, None] * (log_nu - torch.logsumexp(log_K + log_u[:, :, None], dim=-2))

    return log_u, log_v
```

**Why not IFT**: The Implicit Function Theorem computes exact gradients at the Sinkhorn fixed point (infinite-iteration limit). However, empirical testing shows that IFT gradients for cost-matrix parameters (Q, K, w_dist) diverge from unrolled gradients by 1.5–3.5× median relative error, and this error does NOT decrease with more Sinkhorn iterations. The IFT gradient is "exact" at a point the network never reaches — the actual K-iteration output is the operating regime. Unrolled backprop gives exact gradients for the computation that was actually performed.

**Memory**: Autograd stores K intermediate (log_u, log_v) states per block. With K=20: 20 × H × N × 4 bytes = 20 × 16 × N × 4 = 1280N bytes per block. For N=512: 640KB per block. With gradient checkpointing, only one block's intermediates exist at a time. Negligible compared to the (N, 512) h_res boundary state.

**Gradient checkpointing interaction**: The enclosing `torch.utils.checkpoint.checkpoint` recomputes the entire block forward (including Sinkhorn iterations) during backward. The K intermediate states are created fresh during this recomputation and consumed immediately by autograd. No persistent storage across blocks.

---

## 8. Token UOT+EGNN Block

48 blocks in the trunk, 2 in diffusion (diffusion blocks have σ-gated geometry, see §9.2). Each block updates h_res (invariant) and x_res (equivariant) from the same transport plan.

```python
class TokenUOTBlock(nn.Module):
    def __init__(self, d=512, H=16):
        # Attention projections (after LN → bias=False)
        self.ln_attn = LayerNorm(d)
        self.W_Q = Lin(d, d, bias=False)
        self.W_K = Lin(d, d, bias=False)
        self.W_V = Lin(d, d, bias=False)
        self.W_G = Lin(d, d, bias=False)
        self.W_O = Lin(d, d, bias=False)

        # EGNN: per-head signed geometry step size
        self.gamma = nn.Parameter(torch.zeros(H))    # (16,) zeros init

        # Per-head entropic regularization (fixed, multi-scale transport)
        self.register_buffer('eps', torch.tensor([
            0.5, 0.5, 0.5, 0.5,     # sparse: contacts, hot-spots
            1.0, 1.0, 1.0, 1.0,     # balanced: structural reasoning
            2.0, 2.0, 2.0, 2.0,     # smooth: secondary structure
            4.0, 4.0, 4.0, 4.0,     # diffuse: allostery, global
        ]))

        # Transition (after LN → bias=False)
        self.ln_ff = LayerNorm(d)
        self.ff_gate  = Lin(d, d * 4, bias=False)
        self.ff_value = Lin(d, d * 4, bias=False)
        self.ff_out   = Lin(d * 4, d, bias=False)

    def forward(self, h, x_res, mu, nu, log_u_prev=None, log_v_prev=None):
        """
        h:     (N, 512)   token representation (invariant)
        x_res: (N, 3)     token coordinates (equivariant)
        mu:    (H, N)      row marginals
        nu:    (H, N)      column marginals
        Returns: h, x_res, log_u, log_v
        """

        # ---- Pre-norm ----
        h_n = self.ln_attn(h)                                    # (N, 512)
        Q = self.W_Q(h_n).view(N, H, 32)
        K = self.W_K(h_n).view(N, H, 32)
        V = self.W_V(h_n).view(N, H, 32)
        G = self.W_G(h_n).view(N, H, 32)

        # ---- Cost matrix (online, per head) ----
        Q_ln = layer_norm(Q, dim=-1)
        K_ln = layer_norm(K, dim=-1)
        content = -einsum('ihd,jhd->hij', Q_ln, K_ln) / sqrt(32)

        pos_bias = w_rel_res[compute_bins(chain_id, global_idx, bond_matrix)]
        dist = torch.cdist(x_res, x_res)
        geo_bias = w_dist[:, None, None] * dist / (r_0 + dist)

        C = content + pos_bias + geo_bias                        # (H, N, N)

        # ---- Sinkhorn (per-head fixed ε, K=20 uniform) ----
        K_iter = 20
        init_u = log_u_prev    # warm-start from previous block (or None for block 0)
        init_v = log_v_prev
        log_u, log_v = sinkhorn_log_domain(
            C, log(mu), log(nu), eps=self.eps, lam=1.0, K=K_iter,
            log_u_init=init_u, log_v_init=init_v
        )

        # ---- Row-normalized transport plan ----
        log_K = -C / self.eps[:, None, None]                     # (H, N, N) per-head
        log_score = log_u[:, :, None] + log_K + log_v[:, None, :]
        row_max = log_score.max(dim=-1, keepdim=True).values
        T = exp(log_score - row_max)                              # numerically stable
        T_norm = T / (T.sum(dim=-1, keepdim=True) + 1e-6)        # (H, N, N) row-stochastic

        # ---- h_res update (invariant): gated transport-weighted average ----
        O_avg = einsum('hnm,hmd->hnd', T_norm, V)                # (H, N, d_h)
        o = sigmoid(G) * O_avg
        o = rearrange(o, 'h n d -> n (h d)')                     # (N, 512)
        h = h + self.W_O(o)

        # ---- x_res update (equivariant): EGNN ----
        # Transport-weighted centroid per head (fused in Flash kernel with V aggregation)
        x_centroid = einsum('hnm,mc->hnc', T_norm, x_res)        # (H, N, 3)
        # Relative displacement (equivariant: depends only on relative vectors)
        delta = x_res[None] - x_centroid                          # (H, N, 3)
        # Multi-head combination: signed γ allows attraction AND repulsion
        x_res = x_res + einsum('h,hnc->nc', self.gamma, delta)   # (N, 3)
        # NOTE: No per-block re-centering. EGNN is exactly translation-equivariant
        # (T_norm is row-stochastic). Re-centering done once per cycle (§5.2)
        # to handle float32 drift only.

        # ---- SwiGLU transition (h_res only, invariant) ----
        h_n = self.ln_ff(h)
        gate  = self.ff_gate(h_n)
        value = self.ff_value(h_n)
        h = h + self.ff_out(SiLU(gate) * value)

        return h, x_res, log_u, log_v
```

### 8.1 EGNN Coordinate Update: SE(3) Equivariance Proof

The update is: `x_new = x + Σ_h γ_h · (x − T_norm^(h) @ x)`.

**Rotation** (x → xR, R ∈ O(3)): T_norm is invariant (computed from invariant cost C). So `(xR) − T_norm(xR) = (x − T_norm·x)R`. The rotation factors out. ✓

**Translation** (x → x + 1t^T): T_norm is row-stochastic (rows sum to 1), so `T_norm·(x + 1t^T) = T_norm·x + 1t^T`. The displacement `(x + 1t^T) − T_norm·(x + 1t^T) = x − T_norm·x`. Translation cancels completely. ✓

### 8.2 Per-Head Signed γ: Attraction and Repulsion

| γ sign | Effect | Physical interpretation |
|---|---|---|
| γ > 0 | Token moves AWAY from centroid | Repulsion: maintains steric separation |
| γ < 0 | Token moves TOWARD centroid | Attraction: pulls co-evolving residues together |

Different heads learn different signs. The multi-head combination produces complex force fields from 16 scalar parameters per block.

Zeros init: coordinates don't move initially — EGNN must earn its influence. AdamW decay: step sizes stay small. **768 total scalars** (48 blocks × 16 heads) for the entire trunk's coordinate refinement.

### 8.3 Flash Kernel Integration

`T_norm @ x_res` (H, N, 3) is computed alongside `T_norm @ V` (H, N, 32) in the same Flash-Sinkhorn Pass 2 kernel. The transport plan tiles are already in SRAM. Extra cost: 3 dims / 32 dims ≈ **10% on the value aggregation pass**. No additional memory.

**Warm-starting across layers**: All blocks use K=20 iterations with warm-start from previous block's log_u, log_v. Convergence typically reached by K=10–14 with warm start; K=20 provides a safe ceiling. At inference, adaptive stopping with tol=1e-4 typically terminates at K≈14.

---

## 9. Diffusion Module

### 9.1 Design Intent

The diffusion module is a local atom-level refinement engine. It receives the trunk's h_res (which encodes all inter-token structural context from 48 UOT+EGNN blocks) and refines noisy atom coordinates toward the ground truth structure.

**No UOT blocks in diffusion**: The trunk's h_res already contains the complete inter-token structural information. Re-running UOT attention per diffusion step with noisy coordinates adds expense without benefit — at high σ, geometry is garbage (UOT would suppress it anyway); at low σ, the trunk's encoding is already accurate. AF3 similarly does not re-run its Pairformer per diffusion step.

**No μ, ν in diffusion**: The learned marginals were only needed for UOT attention. Without UOT blocks, the diffusion module has no use for marginals.

**End-to-end gradient (no freezing)**: h_res is NOT detached before the diffusion module. The diffusion loss (L_diff, L_lddt) backpropagates through the atom blocks, through the AdaLN conditioning, into h_res, and back through the trunk. This teaches the trunk to produce h_res representations that help with atom-level placement — not just distogram classification. One σ is sampled per training step (standard EDM), so only one diffusion forward/backward is added to the graph.

### 9.2 Diffusion Step

```python
def diffusion_step(h_res, c_atom, p_lm, x_atom_noisy, sigma, token_idx):
    """
    h_res:        (N, 512)     from trunk (NOT detached — gradients flow back)
    c_atom:       (N_atom, 128) frozen reference conformer embedding
    p_lm:         (local, 16)   frozen atom pair representation (intra-token)
    x_atom_noisy: (N_atom, 3)   current noisy coordinates
    sigma:        scalar         current noise level
    token_idx:    (N_atom,)      maps atoms to tokens
    """

    # 1. Timestep embedding + token conditioning (AF3 Alg 21-22)
    # FourierEmbedding: frozen random projection, cos(2π(c_noise·w + b))
    # w, b ~ N(0, 1), requires_grad=False (sampled once, never updated)
    c_noise = log(sigma / sigma_data) * 0.25                     # AF3/Boltz c_noise
    t_emb = FourierEmbedding(d=128)(c_noise)                     # (128,)
    h_cond = h_res + LinNoBias(128 → 512)(LN(t_emb))            # LN on Fourier features

    # 2. Atom embedding: scale by c_in (AF3 Alg 20 line 2)
    c_in = 1 / sqrt(sigma² + sigma_data²)                       # EDM input scaling
    q = c_atom + Lin(3 → 128)(x_atom_noisy * c_in)              # (N_atom, 128)

    # 3. Atom blocks × 10
    for block in atom_blocks:
        q = block(q, c_atom, h_cond, p_lm, sigma, token_idx)

    # 4. Coordinate update
    delta_x = zero_init_Lin(128 → 3)(LN(q))                     # (N_atom, 3)
    x_atom_new = x_atom_noisy + c_out(sigma) * delta_x

    return x_atom_new
```

**Gradient flow**: L_diff → x_atom_new → delta_x → atom blocks → h_cond → h_res → trunk (48 UOT+EGNN blocks). All four losses now train the trunk end-to-end.

**zero_init_Lin**: Output linear layer initialized to zeros so initial prediction is the identity (no update). Standard practice in diffusion models.

### 9.3 Atom Block

```python
class AtomBlock(nn.Module):
    def __init__(self, d_atom=128, H_atom=4):
        # Conditioning projection
        self.cond_proj = Lin(512, 128)          # h_cond → atom dimension

        # AdaLN (AF3 Algorithm 26): sigmoid-bounded scale, LN on conditioning
        # a = sigmoid(Lin(s)) * LayerNorm(a, affine=False) + LinNoBias(s)
        # s is normalized by its own LayerNorm before producing scale/shift
        self.adaln1 = AdaLN(128, 128)           # pre-attention
        self.adaln2 = AdaLN(128, 128)           # pre-transition

        # Self-attention
        self.W_Q = Lin(128, 128, bias=True)     # AF3: Q has bias
        self.W_K = Lin(128, 128, bias=False)
        self.W_V = Lin(128, 128, bias=False)
        self.W_G = Lin(128, 128, bias=False)
        self.W_O = Lin(128, 128, bias=False)

        # Atom pair bias projection
        self.pair_bias_proj = Lin(16, H_atom)   # p_lm → per-head bias

        # AdaLN-Zero output gates (AF3 Algorithm 24, lines 12-14)
        # weight=0, bias=-2 → sigmoid(-2) ≈ 0.12 at init → near-identity block
        self._attn_gate       = Lin(128, 128, w_init=0, b_init=-2)
        self._transition_gate = Lin(128, 128, w_init=0, b_init=-2)

        # Transition (SwiGLU)
        self.swiglu = SwiGLU(128, 512, 128)

    def forward(self, q, c_atom, h_cond, p_lm, t_emb, token_idx):
        """
        q:            (N_atom, 128)  atom representation
        c_atom:       (N_atom, 128)  frozen reference
        h_cond:       (N, 512)       timestep-conditioned token repr
        p_lm:         (local, 16)    frozen atom pair representation
        t_emb:        (128,)         Fourier timestep embedding
        token_idx:    (N_atom,)      maps atoms to tokens
        """

        # Conditioning signal: timestep + token info
        cond = t_emb[None, :] + self.cond_proj(h_cond[token_idx])    # (N_atom, 128)

        # AdaLN-normalized input (AF3 Algorithm 26)
        # sigmoid-bounded scale: a = sigmoid(Lin(s)) * LN(q, affine=False) + LinNoBias(s)
        q_n = self.adaln1(q, cond)                                   # (N_atom, 128)

        # ---- Self-attention: sequence-local window ----
        Q = self.W_Q(q_n).view(N_atom, H_atom, 32)
        K = self.W_K(q_n).view(N_atom, H_atom, 32)
        V = self.W_V(q_n).view(N_atom, H_atom, 32)
        G = self.W_G(q_n).view(N_atom, H_atom, 32)

        # Atom pair bias — LN on z_ij before projection (AF3 Alg 24 line 8)
        b_pair = self.pair_bias_proj(LN(p_lm))                      # (local, H_atom)

        # Sequence-local attention, window size 32 (AF3 style)
        A = softmax(
            einsum('ihd,jhd->hij', Q, K) / sqrt(32) + b_pair,
            dim=-1,
            window=32
        )                                                            # sparse (N_atom, H, 32)
        att_out = sparse_einsum(A, V)                                # (N_atom, H, 32)
        att_out = rearrange(att_out, 'n h d -> n (h d)')             # (N_atom, 128)
        attn_update = sigmoid(rearrange(G, 'n h d -> n (h d)')) * self.W_O(att_out)

        # AdaLN-Zero gate on attention output (AF3 Algorithm 24, lines 12-14)
        # sigmoid(Lin(cond, w=0, b=-2)) ≈ 0.12 at init → near-identity block
        q = q + sigmoid(self._attn_gate(cond)) * attn_update

        # ---- SwiGLU transition with AdaLN + AdaLN-Zero gate ----
        q_n2 = self.adaln2(q, cond)                                  # (N_atom, 128)
        transition_update = self.swiglu(q_n2)
        q = q + sigmoid(self._transition_gate(cond)) * transition_update

        return q
```

**Window size 32 (not 32→128)**: Following AF3, atom self-attention uses a fixed local window of 32 atoms. This covers approximately 2–3 residues of local context, sufficient for sidechain packing and local backbone geometry. Long-range information enters through the AdaLN conditioning from h_res. AF3 demonstrates this is sufficient — atom-level long-range attention is unnecessary when residue-level attention handles long-range communication.

**No inter-residue atom pair bias**: Atom pair representation p_lm covers intra-token geometry only (same as AF3). Inter-residue atom-atom interactions rely on the learned Q-K dot product and the implicit geometry from coordinate embedding. This is identical to the AF3 design and is empirically validated there.

---

## 10. Cropping

### 10.1 Strategy

Spatial crop: pick a random seed residue, take the N_crop nearest residues by Cα distance (from reference or current noisy coordinates). All atoms belonging to selected residues are included.

```python
def spatial_crop(x_res, N_crop):
    seed = randint(0, len(x_res))
    dists = torch.norm(x_res - x_res[seed], dim=-1)
    crop_idx = dists.argsort()[:N_crop]
    return crop_idx

crop_data = {
    'h':           h[crop_idx],
    'x':           x[atom_mask],            # atoms belonging to cropped residues
    'chain_id':    chain_id[crop_idx],
    'global_idx':  global_idx[crop_idx],    # preserved for correct bin computation
    'bond_matrix': bond_matrix[crop_idx][:, crop_idx],
    'm':           m[:, crop_idx, :],
    'residue_idx': reindex(residue_idx, crop_idx),
}
```

### 10.2 Crop Boundary Effects

UOT handles crop boundaries naturally: unbalanced marginals allow residues at the crop boundary to export/import less mass than their "natural" share. No special masking or padding needed.

### 10.3 Crop Size Schedule

| Training Stage | Crop Size (residues) | Rationale |
|---|---|---|
| Stage 1 (0–100K steps) | 256 | Fast iteration, learn local structure |
| Stage 2 (100K–300K) | 384 | Medium-range interactions |
| Stage 3 (300K–500K) | 512 | Long-range, complex contacts |
| Fine-tuning (500K+) | 768 | Full complex contexts |

---

## 11. Loss Functions

### 11.1 EDM Diffusion Loss

```
# 1. Weighted Kabsch alignment: align ground truth to prediction (AF3 Alg 28)
#    Under no_grad — gradients only flow through x_pred in the MSE.
x_aligned = weighted_rigid_align(x_true, x_pred, atom_weights, resolved_mask)  # detached

# 2. MSE against aligned target, weighted by EDM schedule
L_diff = ((σ² + σ_data²) / (σ · σ_data)²) · ||D_θ(x_σ; σ) − x_aligned||²
```

**Weighted rigid alignment** (Kabsch/SVD): Before computing MSE, the ground truth coordinates are rigidly aligned to the prediction via weighted SVD. This removes the rotational/translational degrees of freedom that the SE(3)-invariant model cannot control. The alignment is computed under `no_grad` with `.detach()` — gradients flow only through `x_pred` in the MSE, not through the SVD. Boltz-1 and AF3 both do this.

EDM preconditioning:

| Function | Formula |
|---|---|
| c_skip | σ_data² / (σ² + σ_data²) |
| c_out | σ · σ_data / sqrt(σ² + σ_data²) |
| c_in | 1 / sqrt(σ² + σ_data²) |
| c_noise | ln(σ/σ_data) · 0.25 |

The network predicts the denoised signal. The EDM weighting ensures equal contribution across noise levels.

### 11.2 Smooth LDDT Loss

```
L_lddt = 1 − (1/|P|) Σ_{(i,j)∈P} (1/4) Σ_{δ∈{0.5,1,2,4}} sigmoid((δ − |d_ij^pred − d_ij^true|) / 0.1)
```

P: atom pairs with d_ij^true < 15 Å. Differentiable approximation of LDDT.

### 11.3 Low-Rank Bilinear Distogram Loss (Tiled)

The distogram loss forces h_res to encode pairwise distance information. Since this architecture has no pair representation, the distogram is one of only two training signals for pairwise reasoning (the other being UOT geometry cost in the forward pass). Expressivity here matters.

**Why not simple element-wise**: A naive `(W_u h_i)_b · (W_v h_j)_b` treats each distance bin as independent. But protein distances are continuous — predicting 4Å should share features with 3.5Å and 4.5Å. Element-wise products prevent this cross-bin interaction.

**Low-rank bilinear formulation**: Project to a shared interaction space d_low, compute Hadamard interaction, then project to bins:

```
U_i = W_u(h_i)          ∈ R^{d_low}     (512 → 64)
V_j = W_v(h_j)          ∈ R^{d_low}     (512 → 64)
Z_ij = U_i ⊙ V_j        ∈ R^{d_low}     Hadamard product
logits_ij = W_bin(Z_ij)  ∈ R^{39}        (64 → 39)
```

Each bin's logit is a different linear combination of all d_low interaction features. Neighboring bins can share features (similar rows in W_bin). Strictly more expressive than element-wise, at minimal parameter cost (64 × 39 = 2,496 extra parameters).

**Tiled implementation** (peak memory: tile_size² × d_low × 4 bytes ≈ 1 MB at tile=64):

```python
class LowRankDistogramLoss(nn.Module):
    def __init__(self, d_model=512, d_low=64, num_bins=39):
        super().__init__()
        self.W_u = nn.Linear(d_model, d_low)       # Xavier init
        self.W_v = nn.Linear(d_model, d_low)       # Xavier init
        self.to_bins = nn.Linear(d_low, num_bins)   # Xavier init
        self.tile_size = 64

    def forward(self, h_res, target_bins, valid_mask=None):
        """
        h_res:       (N, 512)
        target_bins: (N, N)   ground truth distance bin indices (0–38)
        valid_mask:  (N, N)   optional: filter by distance cutoff, chain, etc.
        """
        N = h_res.shape[0]

        # O(N) projections — computed once
        U = self.W_u(h_res)                          # (N, d_low)
        V = self.W_v(h_res)                          # (N, d_low)

        total_loss = 0.0
        count = 0
        T = self.tile_size

        for i0 in range(0, N, T):
            ie = min(i0 + T, N)
            U_tile = U[i0:ie]                        # (ti, d_low)

            for j0 in range(0, N, T):
                je = min(j0 + T, N)
                V_tile = V[j0:je]                    # (tj, d_low)

                # Interaction + bin projection
                Z = U_tile[:, None, :] * V_tile[None, :, :]   # (ti, tj, d_low)
                logits = self.to_bins(Z)                        # (ti, tj, 39)
                targets = target_bins[i0:ie, j0:je]

                if valid_mask is not None:
                    m = valid_mask[i0:ie, j0:je]
                    if m.any():
                        total_loss += F.cross_entropy(
                            logits[m], targets[m], reduction='sum')
                        count += m.sum().item()
                else:
                    total_loss += F.cross_entropy(
                        logits.reshape(-1, 39), targets.reshape(-1),
                        reduction='sum')
                    count += (ie - i0) * (je - j0)

        return total_loss / max(count, 1)
```

**Distance bin specification**: 39 bins covering 2Å to 22Å in 0.5Å increments (bins 0–38), plus underflow (< 2Å, mapped to bin 0) and overflow (> 22Å, mapped to bin 38).

### 11.4 Chain Permutation Symmetry Loss

For homo-oligomeric complexes, the loss must be invariant to chain permutation.

```python
def permutation_invariant_loss(pred_x, true_x, chain_ids, loss_fn):
    """
    For K identical chains, compute loss over all K! permutations.
    Use Hungarian algorithm for K > 4.
    """
    unique_chains = torch.unique(chain_ids)
    K = len(unique_chains)

    if K <= 4:
        # Enumerate all permutations
        best_loss = float('inf')
        for perm in itertools.permutations(range(K)):
            permuted_true = permute_chains(true_x, chain_ids, perm)
            loss = loss_fn(pred_x, permuted_true)
            best_loss = min(best_loss, loss)
        return best_loss
    else:
        # Hungarian algorithm on chain-chain RMSD matrix
        rmsd_matrix = compute_chain_rmsd_matrix(pred_x, true_x, chain_ids)
        row_ind, col_ind = linear_sum_assignment(rmsd_matrix)
        permuted_true = permute_chains(true_x, chain_ids, col_ind)
        return loss_fn(pred_x, permuted_true)
```

Applied to L_diff and L_lddt. L_disto uses intra-chain pairs only (no permutation needed).

### 11.5 Trunk Coordinate Loss

Direct structural supervision on the trunk's EGNN-refined coordinates. This is critical: without it, the last UOT+EGNN block's γ receives zero gradient (its x_res output is not used by any other loss). With it, every γ^(b) in all 48 blocks receives direct gradient through the chain of EGNN updates.

```
L_trunk_coord = smooth_lddt(x_res_final, x_0_res)
```

Uses the same smooth LDDT formulation as §11.2, but evaluated on token-level (Cα) coordinates from the trunk, not atom-level coordinates from the diffusion module. Since smooth LDDT uses internal pairwise distances, it is invariant to rigid-body transformations — no alignment needed, consistent with the equivariant architecture.

### 11.6 Total Loss

```
L = L_diff + L_lddt + 0.2 · L_disto + 0.5 · L_trunk_coord

Fine-tuning: L += α_bond · L_bond
```

**Gradient flow**:

- L_diff, L_lddt, L_bond → diffusion module parameters (atom blocks, diffusion UOT, coord output)
- L_disto → trunk h_res → all trunk parameters (representation supervision)
- L_trunk_coord → trunk x_res → all EGNN γ parameters + all UOT params via geometry cost (coordinate supervision)

L_disto and L_trunk_coord are complementary: L_disto teaches h_res to encode distance information (pairwise distance classification into 39 bins), L_trunk_coord teaches EGNN γ to produce accurate coordinates (pairwise distance regression via smooth LDDT). Together they provide the trunk with both representation-level and coordinate-level structural signals.

where L_bond penalizes deviations from ideal bond lengths and angles (standard force-field terms, fine-tuning only).

---

## 12. Diffusion Schedule

| Parameter | Value |
|---|---|
| σ_data | 16 Å |
| σ_max | 160 Å |
| σ_min | 0.002 Å |
| P_mean | −1.2 |
| P_std | 1.2 |
| Training | ln σ ~ N(−1.2, 1.2²) |
| Inference steps | 200 (development: 50) |
| Sampler | Heun 2nd order |
| ρ (schedule curvature) | 7 |

Inference σ schedule (Karras):

```python
sigmas = (sigma_max^(1/rho) + i/(steps-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
for i in range(steps)
```

---

## 13. Optimizer and Training Stability

### 13.1 Optimizer

```python
param_groups = [
    {   # Weight matrices: weight decay
        'params': [p for n, p in model.named_parameters()
                   if 'layernorm' not in n.lower()
                   and 'bias' not in n
                   and 'gamma' not in n],       # EGNN γ handled separately
        'weight_decay': 0.01
    },
    {   # LayerNorm γ,β and standalone biases: no weight decay
        'params': [p for n, p in model.named_parameters()
                   if 'layernorm' in n.lower() or 'bias' in n],
        'weight_decay': 0.0
    },
    {   # EGNN γ: weight decay (pull toward zero = no coordinate update)
        'params': [p for n, p in model.named_parameters()
                   if 'gamma' in n],
        'weight_decay': 0.01
    },
]

optimizer = AdamW(param_groups, lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

# Learning rate schedule: linear warmup + cosine decay
warmup_steps = 5000
total_steps = 500000

ema = EMA(model, decay=0.999)
```

**Note on bias=False**: Most Linear layers in the model use `bias=False` (all post-LN projections). Only standalone projections (input embedding, coordinate output, loss heads) have bias. This means the "standalone biases" group is small. LayerNorm γ and β are the main no-decay parameters.

### 13.2 Exponential Moving Average (EMA)

EMA maintains a shadow copy of all model parameters, updated each training step:

```python
θ_ema ← decay · θ_ema + (1 - decay) · θ_train
```

with decay = 0.999 (each step blends 0.1% of current weights into the shadow).

**Why EMA**: Neural network training is noisy — individual gradient steps can overshoot or oscillate around the optimum. EMA smooths these oscillations by averaging over the last ~1000 steps (the effective window for decay=0.999). The EMA parameters typically generalize better than the raw training parameters.

**Training**: All loss computation and gradient updates use the raw training parameters θ_train. EMA is a passive observer — it reads θ_train after each optimizer step and updates θ_ema. No extra gradient computation.

**Inference**: Always use θ_ema, never θ_train. The EMA parameters are the deployed model. This is standard practice (AF2, AF3, all modern diffusion models).

**Interaction with recycling**: During training, all cycles (including the backprop cycle) use θ_train. The EMA update happens after the full training step completes. At inference, all cycles use θ_ema.

**Memory**: θ_ema is a full copy of all model parameters (~220M × 4 bytes = ~880MB in FP32). This doubles the parameter memory but not the activation memory. For a 220M parameter model on modern GPUs, this is negligible.

**EMA warmup**: During the first few thousand steps, θ_train changes rapidly and θ_ema lags far behind. Common practice: don't update EMA for the first 1000 steps (let θ_train stabilize), then start EMA from the current θ_train. Alternative: use a lower decay (0.99) initially, increasing to 0.999 over 10K steps.

```python
class EMA:
    def __init__(self, model, decay=0.999, warmup_steps=1000):
        self.shadow = {n: p.clone() for n, p in model.named_parameters()}
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0

    def update(self, model):
        self.step += 1
        if self.step <= self.warmup_steps:
            # Copy current params directly during warmup
            for n, p in model.named_parameters():
                self.shadow[n].copy_(p)
        else:
            for n, p in model.named_parameters():
                self.shadow[n].lerp_(p, 1 - self.decay)

    def apply(self, model):
        """Swap EMA params into model for inference."""
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])
```

### 13.2 Stability Mechanisms

| Risk | Mitigation | How |
|---|---|---|
| Sinkhorn divergence | LN on Q, K | Cost C is O(1) |
| Sinkhorn convergence | K=20 uniform | Empirically: ~1e-6 residual at K=20; unrolled backprop needs correct forward |
| Allosteric signal loss | Fixed per-head ε = [0.5, 1.0, 2.0, 4.0] | Diffuse heads (ε=4) maintain global receptive field |
| Head collapse | Per-head W_μ, W_ν | Separate marginal learning |
| Length bias | Transport-weighted average | Division by T_sum |
| Transport output overflow | Running-max subtraction | Same trick as FlashAttention |
| EGNN coordinate collapse | Per-head signed γ | Repulsive heads counteract attractive heads |
| EGNN coordinate explosion | γ zeros init + AdamW decay | Step sizes stay small; must earn influence |
| EGNN last-block dead gradient | L_trunk_coord on final x_res | Direct gradient to all γ through EGNN chain |
| EGNN float32 centroid drift | Re-center once per cycle | Not per-block; EGNN is exactly translation-equivariant |
| Geometry bias at high noise (trunk) | Per-head w_dist diversity | Some heads geometry-sensitive, others robust |
| Geometry bias at high noise (diffusion) | σ-conditioned geo_gate | Learned suppression at high σ |
| Geometry bias domination | w_dist zeros init + AdamW decay | Starts at zero, regularized |
| Position bias scale | w_rel zeros init + AdamW decay | Starts at zero, regularized |
| Co-evolution noise | Per-layer per-head α_coevol zeros init + decay | Must earn influence; per-head specialization |
| Marginal reset across cycles | Log-space geometric blend with previous cycle | Preserves transport patterns |
| SE(3) equivariance (trunk) | EGNN: relative vectors only | No augmentation needed |
| SE(3) equivariance (diffusion) | AF3-style augmentation | Standard, lightweight |
| MSA overfitting | Row dropout p=0.15 | Drop entire sequences |
| Gradient bias (Sinkhorn) | Unrolled backprop through K iterations | Exact gradient for actual computation; IFT gives biased gradients on Q,K,w_dist |
| Parameter oscillation | EMA decay=0.999 | Smoothed parameters |
| Crop boundary | UOT unbalanced marginals | Natural mass adjustment |
| Diffusion output at init | Zero-init final Linear | Identity prediction initially |
| Large coordinate scale (diffusion) | Division by σ_data in atom embedding | O(1) embedding input |

### 13.4 Backward Pass

Only the last recycling cycle is in the computation graph. All earlier cycles run under `torch.no_grad()`. The graph contains: 4 MSA blocks + 48 UOT+EGNN blocks (with gradient checkpointing) + loss heads. The diffusion module has its own separate graph.

**Gradient sources** (four losses):

```
L = L_diff + L_lddt + 0.2 · L_disto + 0.5 · L_trunk_coord
```

| Loss | Gradient target | What it trains |
|---|---|---|
| L_diff | ∂/∂x_atom_pred → atom blocks → h_cond → h_res | Atom blocks + entire trunk (end-to-end) |
| L_lddt | ∂/∂x_atom_pred → atom blocks → h_cond → h_res | Same path as L_diff |
| L_disto | ∂/∂h_res^(48) | All trunk params: MSA blocks, UOT attention, SwiGLU |
| L_trunk_coord | ∂/∂x_res^(48) | All EGNN γ + all trunk params via geometry cost |

**Backward through diffusion module**: Standard backprop through 10 atom blocks. Gradients flow through h_cond into h_res and back through the entire trunk. One σ sampled per training step — only one diffusion forward/backward in the graph. Cost: ~2× diffusion forward.

**Backward through UOT+EGNN blocks** (the complex part):

Each block b receives upstream gradients ∂L/∂h^(b+1) and ∂L/∂x^(b+1) and propagates backward through 4 stages in reverse:

**Stage 4 — SwiGLU backward**: Standard backprop through SiLU, element-wise multiply, linear projections. Produces ∂L/∂(ff_gate, ff_value, ff_out, ln_ff params) and ∂L/∂h_pre_transition.

**Stage 3 — EGNN backward**: The EGNN update is `x_new = x + Σ_h γ_h · (x − T_norm^(h) @ x)`.

```
∂L/∂γ_h = Σ_i grad_x_i^T · (x_i − centroid_i^(h))      scalar per head

∂L/∂x_j += Σ_i Σ_h γ_h · (−T_norm_ij^(h)) · ∂L/∂x_new_i  coordinate gradient

∂L/∂T_norm_ij^(h) = −γ_h · (∂L/∂x_new_i)^T · x_j          transport plan gradient
```

This produces gradients for γ (trains EGNN step sizes), for input x (cascades to previous block), and for T_norm (enters Sinkhorn unrolled backward).

**Stage 2 — UOT attention backward (unrolled through Sinkhorn)**:

The attention output produces ∂L/∂T_norm from the h_res path (through V aggregation). Combined with ∂L/∂T_norm from EGNN (stage 3), autograd differentiates backward through the K Sinkhorn iterations:

```
Iteration K → K-1 → ... → 1 → 0:
    Each logsumexp backward produces ∂L/∂C (accumulated across iterations)
    ∂L/∂C chains to:
        ∂L/∂W_Q, ∂L/∂W_K    via content term (-LN(Q)^T LN(K)/√d_h)
        ∂L/∂w_rel             via position bias
        ∂L/∂w_dist            via geometry term
        ∂L/∂x_res             via distance in geometry term
        ∂L/∂μ, ∂L/∂ν         via marginal terms in Sinkhorn update
```

No custom backward needed — standard autograd through the Sinkhorn loop. The K intermediate (log_u, log_v) states are recomputed during gradient checkpointing and consumed by autograd.

**x_res accumulates gradients from THREE sources per block**:

1. EGNN backward: through centroid displacement
2. Geometry cost backward: through distance term in C_ij (via unrolled Sinkhorn)
3. Pass-through from block b+1

**Stage 1 — Pre-norm + projections backward**: Standard linear backward for W_Q, W_K, W_V, W_G, ln_attn.

**Backward through MSA blocks**: Standard backprop through 4 blocks. Marginal gradients ∂L/∂μ, ∂L/∂ν from the Sinkhorn backward flow through the marginal projection → α_coevol → co-evolution aggregation → MSA representation. This is how UOT transport quality trains the co-evolution computation.

**Gradient checkpointing**: Each of the 48 UOT+EGNN blocks is checkpointed. During backward, the block's forward pass (including K Sinkhorn iterations) is recomputed from boundary activations (h_res, x_res, log_u, log_v). The K intermediate states are created fresh and consumed by autograd. Memory per block during backward: K × H × N × 4 bytes for Sinkhorn intermediates + O(N²) for cost matrix (eliminated by Flash-Sinkhorn tiling).

**Total backward memory**: O(48 × N × d) for boundary activations + O(N²) for one block's recomputed cost matrix. With Flash-Sinkhorn: O(N × d) total.

**Complete gradient flow**:

```
L_trunk_coord → x_res^(48) → EGNN^(48) → γ^(48) ✓
                                ↓
                            T_norm^(48) → Sinkhorn^(48) unrolled → C^(48) → W_Q,W_K,w_dist,w_rel
                                                                     ↓
                                                                  x_res^(47) → EGNN^(47) → γ^(47) ✓
                                                                     ↓
                                                                  ... cascades through all 48 blocks ...
                                                                     ↓
                                                                  x_res^(0) → block 0 params

L_disto → h_res^(48) → W_O^(48), SwiGLU^(48)
                          ↓
                      T_norm^(48) → Sinkhorn^(48) unrolled → (same cascade as above)
                          ↓
                      h_res^(47) → block 47 → ... → h_res^(0) → MSA blocks → MSA params

Both paths merge at Sinkhorn backward: the transport plan receives gradients from both
h_res (representation quality) and x_res (coordinate quality).
```

**Training cost per step**:

```
(num_cycles − 1) × forward + 1 × forward + 1 × backward
≈ (num_cycles + 2.5) × single_cycle_forward

num_cycles=3: ~5.5× single cycle forward
num_cycles=1: ~3.5× single cycle forward
Plus: diffusion forward + backward (~3× diffusion forward, per sampled σ)
```

### 13.5 Initialization

Two-tier philosophy: (1) the representation path (h_res) starts as near-identity across all 48 blocks, (2) the coordinate path (x_res) starts completely dormant. This mirrors the denoising process — content first, then geometry, then fine refinement — without needing a curriculum schedule.

**Attention output scaling (1/√L)**: W_O is scaled by 1/√48 to prevent the residual stream from growing with depth. Without this, 48 blocks of residual additions would grow h_res magnitude ~√48 ≈ 7×, destabilizing LayerNorm and loss computation. This follows GPT-2/3/LLaMA practice.

```python
def init_model(model, num_blocks=48):
    for name, param in model.named_parameters():
        if param.dim() < 2:
            # --- Scalars and vectors ---
            if 'gamma' in name:                    # EGNN γ: dormant at init
                nn.init.zeros_(param)
            elif 'w_dist_logit' in name:             # geometry bias: sigmoid(-2.0) ≈ 0.12
                nn.init.constant_(param, -2.0)
            elif 'w_rel' in name:                  # position bias: off at init
                nn.init.zeros_(param)
            elif 'alpha_coevol' in name:           # co-evolution gate: off at init
                nn.init.zeros_(param)
            elif 'layernorm.weight' in name:       # LN γ = 1
                nn.init.ones_(param)
            elif 'layernorm.bias' in name:         # LN β = 0
                nn.init.zeros_(param)
            elif 'bias' in name:                   # standalone biases: zero
                nn.init.zeros_(param)

        elif param.dim() == 2:
            # --- Weight matrices ---
            if 'coord_output' in name or 'zero_init' in name:
                nn.init.zeros_(param)              # diffusion coord output: identity at init
            elif 'W_O' in name:
                nn.init.xavier_normal_(param)
                param.data /= math.sqrt(num_blocks)  # 1/√48 residual scaling
            else:
                nn.init.xavier_normal_(param)      # all other weight matrices
```

| Component | Init | Rationale |
|---|---|---|
| W_Q, W_K, W_V, W_G | Xavier | Standard; LN on Q,K controls scale to O(1) |
| W_O (attention output) | Xavier / √48 | Residual depth scaling; prevents h_res magnitude growth |
| SwiGLU gate, value | Xavier | SiLU(gate) × value product naturally self-suppresses at init |
| SwiGLU output (ff_out) | Xavier (not scaled) | Product structure already suppresses; double-scaling would starve transitions |
| γ (EGNN) | zeros | EGNN dormant at init; must earn influence via L_trunk_coord gradient |
| w_dist_logit | -2.0 (sigmoid ≈ 0.12) | Weak geometry at init; per-block per-head; sigmoid-bounded (0,1); no weight decay |
| w_rel | zeros | Position bias off at init; learns sequence-distance priors during training |
| α_coevol | zeros | Co-evolution bias off at init; MSA processing must stabilize first |
| LN γ, β | (1, 0) | Identity normalization at init |
| Input embedding Lin(38→512) | Xavier + bias=zeros | Standard; network adjusts column scales for different input features |
| Atom encoder projections | Xavier | Standard; encoder output is O(1) at init |
| Diffusion coord output Lin(128→3) | **all zeros** | Identity denoising at init (EDM preconditioning assumes this) |
| Distogram head (W_u, W_v, W_bin) | Xavier | Random predictions at init → strong initial gradients to h_res |
| FourierEmbedding w, b | N(0,1), **frozen** | Random Fourier features; never updated (AF3 Algorithm 22) |
| AdaLN-Zero gates (_attn_gate, _transition_gate) | **w=0, b=−2** | sigmoid(−2)≈0.12 at init → atom blocks start as near-identity (AF3 Algorithm 24) |
| AdaLN s_scale, s_bias | Xavier | Conditioning projections; sigmoid bounds scale to [0,1] |

**Staged activation dynamics**: At the start of training:

1. Content cost (`-LN(Q)^T LN(K)/√d_h`) is the only active cost term → attention based on sequence/MSA features
2. Transport plans are approximately uniform (all costs similar) → EGNN centroids ≈ global mean → γ=0 means no coordinate movement
3. L_disto provides strong gradients (random predictions, high loss) → h_res rapidly learns distance-relevant features
4. As h_res improves → content cost becomes informative → transport plans sharpen → EGNN centroids become meaningful
5. L_trunk_coord gradient pushes γ away from zero → coordinates begin to refine
6. Better coordinates → w_dist grows from zero → geometry cost activates → positive feedback loop

This ordering emerges naturally from the initialization without any explicit curriculum.

### 13.6 Backward Implementation (Reference PyTorch)

The reference implementation uses standard PyTorch autograd — no custom backward functions. Sinkhorn is a plain differentiable loop; autograd handles backward through all K iterations automatically. Gradient checkpointing bounds memory for the 48-block stack.

**Differentiable Sinkhorn (no custom autograd)**:

The Sinkhorn function from §7.3 is directly differentiable by PyTorch autograd. No `torch.autograd.Function` needed:

```python
# sinkhorn_differentiable from §7.3 — just a plain loop
# PyTorch autograd records the K iterations and differentiates through them
# Gradient checkpointing on the enclosing block recomputes iterations during backward
```

**Gradient checkpointing on the 48-block stack**:

```python
# In the recycling loop (last cycle, backprop enabled):
for block in residue_uot_blocks:
    h_res, x_res, log_u_prev, log_v_prev = torch.utils.checkpoint.checkpoint(
        block, h_res, x_res, mu, nu, log_u_prev, log_v_prev,
        use_reentrant=False
    )
```

Each block's activations (including K Sinkhorn intermediate states) are discarded after forward and recomputed during backward. Memory: O(N × d) for 48 boundary states (h_res, x_res per block). The Sinkhorn intermediates are recreated on-the-fly during the checkpointed recomputation and consumed immediately by autograd.

**Full training step (reference)**:

```python
def train_step(model, batch, optimizer, ema):
    optimizer.zero_grad()

    # 1. Trunk forward (last cycle backpropagates)
    h_res, x_res, L_disto, L_trunk_coord = model.trunk_forward(batch)

    # 2. Diffusion forward — h_res NOT detached, gradients flow back to trunk
    x_atom_pred = model.diffusion_step(h_res, batch.c_atom, batch.p_lm,
                                        batch.x_atom_noisy, batch.sigma,
                                        batch.token_idx)
    L_diff = edm_loss(x_atom_pred, batch.x_0_atom, batch.sigma)
    L_lddt = smooth_lddt_loss(x_atom_pred, batch.x_0_atom)

    # 3. Total loss — all four losses train the trunk end-to-end
    L = L_diff + L_lddt + 0.2 * L_disto + 0.5 * L_trunk_coord

    # 4. Backward through everything (trunk + diffusion, one graph)
    L.backward()

    # 5. Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 6. Optimizer step
    optimizer.step()

    # 7. EMA update
    ema.update(model)

    return L.item()
```

---

## 14. Parameter Inventory

### 14.1 Trunk Parameters

All post-LN projections use `bias=False`. Standalone projections (input embed, coord output, loss heads) use `bias=True`.

| Component | Shape | Count | Init | Decay |
|---|---|---|---|---|
| Input: Lin(38→512, bias=True) | (38, 512) + (512,) | 20K | Xavier | ✅ |
| MSA input: Lin(34→64, bias=True) | (34, 64) + (64,) | 2K | Xavier | ✅ |
| MSA inject: Lin(512→64, bias=F) | per block × 4 | 131K | Xavier | ✅ |
| MSA row attn: Q,K,V,G,O Lin(64→64, bias=F) | per block × 4 | 4 × 5 × 4K = 82K | Xavier | ✅ |
| MSA col: Lin(64→1, bias=F), Lin(64→512, bias=F) | per block × 4 | 4 × 33K = 131K | Xavier | ✅ |
| MSA co-evol: U,V Lin(64→16, bias=F) | per block × 4 | 4 × 2 × 1K = 8K | Xavier | ✅ |
| MSA co-evol scalar weight: Lin(16→1, bias=F) | per block × 4 | 4 × 16 = 64 | Xavier | ✅ |
| MSA co-evol agg: LN_Lin(512→512), Lin(512→512, bias=F) | per block × 4 | 4 × 2 × 262K = 2.1M | Xavier | ✅ |
| MSA marginals: Lin(512→16, bias=F) × 2 | per block × 4 | 4 × 2 × 8K = 66K | zeros | ✅ |
| MSA marginal co-evol proj: Lin(16→16, bias=F) | per block × 4 | 4 × 256 = 1K | Xavier | ✅ |
| MSA alpha_coevol | (4, 16) | 64 | zeros | ✅ |
| MSA SwiGLU: Lin(64→256, bias=F)×2, Lin(256→64, bias=F) | per block × 4 | 4 × 49K = 196K | Xavier | ✅ |
| MSA position: w_rel_msa | (8, 68) | 544 | zeros | ✅ |
| **MSA total** | | **~2.7M** | | |
| Token UOT+EGNN: LN + Q,K,V,G Lin(512→512, bias=F) | per block × 48 | 48 × 4 × 262K = 50.3M | Xavier | ✅ |
| Token UOT+EGNN: W_O Lin(512→512, bias=F) | per block × 48 | 48 × 262K = 12.6M | **Xavier/√48** | ✅ |
| Token UOT+EGNN: SwiGLU Lin(512→2048, bias=F)×2, Lin(2048→512, bias=F) | per block × 48 | 48 × 3.1M = 150M | Xavier | ✅ |
| Token UOT+EGNN: γ (EGNN step sizes) | per block × 48, (16,) each | 768 | **zeros** | ✅ |
| Token UOT+EGNN: ε (per-head, fixed) | buffer (16,), shared all blocks | 0 (not a parameter) | [1.0]×4+[2.0]×4+[4.0]×4+[8.0]×4 | N/A |
| Token position: w_rel_res | (16, 68) | 1K | zeros | ✅ |
| Geometry: w_dist_logit | per block × 48, (16,) each | 768 | **-2.0** (sigmoid ≈ 0.12) | ❌ (sigmoid-bounded) |
| LN γ, β (attention + transition) | per block × 48, 2 × (512,) each | 48 × 2K = 98K | (1, 0) | ❌ |
| **Token UOT+EGNN total** | | **~213M** | | |
| **Trunk total** | | **~216M** | | |

### 14.2 Diffusion Parameters

| Component | Shape | Count | Init | Decay |
|---|---|---|---|---|
| Timestep conditioning: Lin(128→512) | (128, 512) + (512,) | 66K | Xavier | ✅ |
| Atom embed: Lin(3→128) | (3, 128) + (128,) | 512 | Xavier | ✅ |
| Atom ref embed: Lin(D_ref→128) | depends on D_ref | ~25K | Xavier | ✅ |
| Atom pair embed | (three small projections) | ~1K | Xavier | ✅ |
| Atom blocks × 10 | per block ~400K | 4M | Xavier | ✅ |
| Coord output: Lin(128→3) | (128, 3) + (3,) | 387 | **zeros** | ✅ |
| Distogram: W_u Lin(512→64), W_v Lin(512→64), W_bin Lin(64→39) | (512,64)+(512,64)+(64,39) | 68K | Xavier | ✅ |
| **Diffusion total** | | **~4.2M** | | |

### 14.3 Grand Total

**~220M parameters**

AF3 uses c_s=384 (single) and c_z=128 (pair) with 48 Pairformer blocks and 24 atom transformer blocks per diffusion step. Our model uses d=512 (single only) with 48 UOT+EGNN blocks and 10 atom blocks per diffusion step. No UOT blocks in diffusion — the trunk's h_res provides all inter-token context. End-to-end gradient from diffusion loss through h_res into the trunk.

---

## 15. Complexity Analysis

| Component | Compute | Memory (reference) | Memory (tiled) |
|---|---|---|---|
| MSA row attention (×4) | O(S · N² · d_msa) | O(S · N · d_msa) | same (FlashAttn) |
| Co-evolution outer product (×4) | O(S · N² · r) | O(tile² · r) per tile | O(tile² · r) per tile |
| Co-evolution aggregation (×4) | O(N² · d) | O(tile² + tile · d) per tile | O(tile² + tile · d) per tile |
| Residue UOT (×48, K iters) | O(48 · K · N² · d_h) | **O(N²)** per cdist | O(H · N) with FlashSinkhorn |
| Distogram | O(N² · d_low) | O(tile² · d_low) per tile | O(tile² · d_low) per tile |
| Atom self-attention | O(N_atom · w · d_atom) | O(N_atom · d_atom) | same |
| **Reference total** | | **O(N²)** for cdist only | |
| **Tiled total** | | | **O(N · d + S · N · d_msa)** |

**Tiling status**: Co-evolution and distogram use Python-level tiling (tile=64) in the reference implementation. torch.compile or Triton kernels will fuse the loops. The remaining O(N²) bottleneck is cdist in Residue UOT blocks, eliminated by the Flash-Sinkhorn kernel which computes distances on-the-fly per tile.

---

## 16. Information Flow Summary

```
[TRUNK — random 1–5 recycling passes, SE(3) equivariant]

h_res = Lin(38→512)(cat(token_type, profile, del_mean, has_msa))  (invariant)
h_res += atom_to_token_encoder(c_atom, p_lm)                      (invariant)
x_res = randn(N, 3) * σ_data, centered                            (equivariant)
num_cycles ~ Uniform{1..5} (training)

Recycling × num_cycles (all but last: no_grad; last: backprop):
  MSA Block × 4 (no coordinates — pure sequence/co-evolution):
    row attn(w_rel_msa) → m                           ← within-sequence context
    col weighted mean → h_res                         ← 1st-order co-evolution
    rank-16 outer_product(U, V)/S → c_ij ∈ R^16       ← 2nd-order co-evolution
    Lin(16→1)(c_ij) → scalar weight → h_res           ← co-evolution aggregation
    Lin(16→H)(c̄) · α_coevol[block,head] → μ, ν bias  ← per-head marginal modulation
    log-blend with previous cycle marginals             ← marginal carry

  Token UOT+EGNN Block × 48 (warm-start carry across cycles):
    LN(Q)^T LN(K) + w_rel + w_dist·f(d) → C          ← content + position + geometry
    Sinkhorn(C, μ, ν) → T*, T_norm                     ← sparse transport plan
    T_norm @ V → gated → h_res                         ← long-range interaction (invariant)
    T_norm @ x_res → centroid                           ← transport-weighted centroid
    x += Σ_h γ_h · (x − centroid_h)                    ← EGNN update (equivariant)
    Geometry cost in next block uses updated x_res      ← iterative refinement

  Re-center x_res (once per cycle)                      ← float32 hygiene

    ↓ L_trunk_coord = smooth_lddt(x_res, x_0)          ← direct EGNN supervision
    ↓ L_disto = distogram(h_res, x_0)                   ← representation supervision

[DIFFUSION — per step, AF3-style augmentation, end-to-end gradient through h_res]

  h_cond = h_res + Lin(t_emb)                          ← timestep + trunk conditioning (NOT frozen)
  x_atom_noisy / σ_data + c_atom → q                   ← atom embed
  Atom Block × 10:
    AdaLN(ln(σ)/4 + Lin(h_cond)) → q                   ← timestep + token conditioning
    local self-attn(w=32, p_lm bias) → q                ← local atom interactions
    sigmoid(Lin(c_atom)) · q → q                        ← reference gating
    SwiGLU + AdaLN → q
  LN(q) → zero_init_Lin → Δx                           ← coordinate update
  x_atom += c_out(σ) · Δx

  L_diff, L_lddt → backprop through atom blocks → h_cond → h_res → trunk (end-to-end)
```

---

## 17. Comparison with AF3

| Aspect | AF3 | This Model (v4.6) |
|---|---|---|
| Single representation | c_s = 384 | d = 512 |
| Pair representation | c_z = 128, O(N²) stored | ❌ none |
| Trunk blocks | 48 Pairformer (pair + single tracks) | 48 UOT+EGNN (single track + equivariant coords) |
| Triangle attention/update | ✅ | ❌ |
| Co-evolution signal | Outer product → z_ij (full pair) | Rank-16 outer product → scalar agg weight + r-dim marginal profile (tiled) |
| Token-level attention | Softmax + gate | UOT-Sinkhorn (multi-scale ε) + gate |
| Structure refinement (trunk) | N/A (Pairformer has no structure module) | EGNN via transport-weighted centroid displacement |
| SE(3) equivariance (trunk) | N/A | EGNN equivariant (no augmentation) |
| SE(3) equivariance (diffusion) | Augmentation | Augmentation (same) |
| Position encoding | Relative bins (clipped ±32) + chain offset | 68-bin + covalent bond |
| Geometry in attention | Via pair rep (implicit) | Explicit x_res → metric cost, updated every block via EGNN |
| Atom coordinate embed | c_atom = 128 per diffusion step | d_atom = 128 per diffusion step (same) |
| Atom attention | Local window 32→128 | Local window 32 (same pattern) |
| Atom pair bias | c_atompair = 16, intra-token | d_pair = 16, intra-token (same) |
| Recycling | 3 fixed | 1–5 random (EGNN refines coords across 48 blocks/cycle) |
| Marginal handling | N/A (softmax) | Per-layer per-head α_coevol, log-blend across cycles |
| Template | ✅ | ❌ (v1 excluded) |
| Confidence heads | pLDDT, PAE, pTM | Planned (§18) |
| Distogram | In pair rep (c_z track) | Low-rank bilinear (d_low=64, tiled, separate loss head) |
| Loss | EDM + FAPE + LDDT + distogram | EDM + LDDT + distogram + trunk_coord + chain perm |
| Gradient through attention | Standard backprop (softmax) | Unrolled backprop through K Sinkhorn iterations |
| EMA | decay=0.999 | decay=0.999 (same) |
| Memory | O(N²) | O(N) arch, O(N²) ref (cdist only) |
| Diffusion atom blocks | 24 per step | 10 per step |
| Parameters | ~226M (est. for trunk + diffusion) | ~220M |

---

## 18. Remaining Work

### Design

1. **Confidence heads**: pLDDT from single rep (standard MLP on h_res). PAE equivalent: bilinear from single rep, same decomposition as distogram but with error bins. pTM: derived from PAE.
2. **Template integration** (v2): Template coordinates → pairwise distances → binned → bias in UOT cost matrix. No pair representation needed — distance bins directly added to C_ij as another bias term, analogous to w_rel.
3. **Detailed training stages**: Learning rate schedule per stage, crop size ramp (Section 10.3).

### Engineering (Priority Order)

0. ~~**Diffusion module stability (AF3/Boltz-1 alignment)**~~ ✅ v4.6
   - FourierEmbedding (frozen random), AdaLN (sigmoid-bounded), AdaLN-Zero output gates
   - c_in coordinate scaling, c_noise formula, LayerNorm on Fourier/pair features, LinearNoBias

0.5. ~~**Flash-Sinkhorn Triton kernels + gradient correctness**~~ ✅
   - All 13 Triton kernels support batch dim: grid `(B*H, n_tiles)`
   - Mask support: `-1e9` bias on padded positions in all score computations
   - Flash forward (Triton): N=384 in 0.6ms, O(N) memory
   - Training backward: dense unrolled (autograd through K iterations) — correct gradients for ALL parameters including mu_proj, nu_proj, coevol_to_marginal
   - IFT backward (CG-based): implemented but gives incorrect gradients for UOT with row-normalized output (∂T_norm/∂log_u = 0 at fixed point). Reserved for future O(N) backward via tiled unrolled approach.
   - FP32 marginal fix: mu.float() before log() prevents BF16 underflow in softmax Jacobian backward
   - Kabsch alignment: disable autocast for SVD/det (BF16 not supported)
   - Coevol kernel wired into MSA (inference), distogram kernel wired into losses (eval)
   - trunk_block.py: dense unrolled for training, flash for inference

1. **Flash-Sinkhorn kernel — further optimization from [flash-sinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn) (MIT license)**

   Foundation: FlashSinkhorn (Ye et al. 2026, arXiv:2602.03067) provides fused Triton kernels for IO-aware streaming Sinkhorn with O(nd) memory. Already supports unbalanced OT via `reach` parameter, transport application `P*V`, analytic gradients, early stopping, and half-precision. 32× forward / 161× end-to-end speedups on A100.

   **Adaptations needed** (fork and extend):

   | Component | FlashSinkhorn (current) | Our requirement | Adaptation |
   |---|---|---|---|
   | Cost function | Squared Euclidean `‖x−y‖²` | `-LN(Q)^T LN(K)/√d_h + w_rel[bin] + w_dist·f(d)` | Replace distance computation per tile with dot-product + two additive biases (position lookup, geometry from cdist) |
   | ε | Scalar (`blur²`) | Per-head (H,) buffer `[0.5,1,2,4]×4` | Broadcast: load ε_h once per head per tile |
   | UOT parameterization | `reach` (KL penalty) | κ = λ/(λ+ε) damping per head | Map: their `reach` → our λ; verify half-step matches `κ·(log_μ − LSE(...))` |
   | Transport application | `P*V` (apply_plan_vec/mat) | `P*V` (H,N,d_h) + `P*x_res` (H,N,3) for EGNN | Extend kernel: accumulate V (d_h dims) and x_res (3 dims) in same streaming pass |
   | Backward | Analytic gradients (custom) | Unrolled autograd through K iterations | Use their forward kernel as single-iteration building block; call K times in differentiable loop; autograd handles backward |
   | Position encoding | None | 68-bin w_rel[bin(i,j)] per tile | Compute bin indices from chain_id, global_idx, bond_matrix per tile; lookup w_rel as additive bias |
   | Geometry bias | Implicit in squared Euclidean cost | Explicit `w_dist·d/(r_0+d)` per head | Compute cdist per tile (already done for their cost); apply per-head w_dist scaling |

   **Migration strategy**:
   - Phase 1: Fork flash-sinkhorn. Replace squared-Euclidean cost with general cost (content + position + geometry). Validate forward output matches reference PyTorch.
   - Phase 2: Add per-head ε broadcast. Extend transport application kernel with EGNN centroid accumulation (+3 dims alongside V's d_h dims).
   - Phase 3: Wrap single-iteration kernel in differentiable loop for unrolled backward. Validate gradients match reference PyTorch autograd.
   - Phase 4: Profile. Tune tile sizes for our d_h=32 (vs their typical d=64 point clouds).

   This eliminates the last O(N²) bottleneck (cdist) and is the critical path for scaling to large complexes.

2. **Co-evolution tiling Triton kernel** (fuse the Python tile loops in Section 6.2)
3. **Distogram tiling Triton kernel** (fuse the Python tile loops in Section 11.3)
4. **Mixed precision strategy** (BF16 for most, FP32 for Sinkhorn log-domain)

**Current memory profile** (no checkpointing, crop=384, B200 192GB):
- Cost matrix per block: `(B, H, N, N)` = `(1, 16, 384, 384)` × 4B = 9.4MB
- Autograd stores 48 blocks' intermediates: ~450MB total
- At crop=1024: ~3.2GB — still fits comfortably in 192GB
- Checkpointing removed for DDP compatibility; B200 has sufficient memory

**Status**: Flash-Sinkhorn (item 1) is the critical path for scaling beyond crop ~2048 where the O(N²) Sinkhorn intermediates (~50GB at N=2048) would exceed single-GPU memory. Until then, dense PyTorch Sinkhorn on high-memory GPUs (B200, H100 80GB) is sufficient. Items 2–3 are performance optimizations. Item 4 is active (BF16 autocast for trunk, FP32 for Sinkhorn log-domain).

### Inference Optimizations

1. **Trunk-initialized diffusion**: Instead of starting diffusion from pure noise (σ_max=160Å), initialize atom coordinates from the trunk's refined x_res plus noise at σ_init ≈ 3–5Å (matching trunk's typical RMSD). Enter the Karras schedule at σ_init instead of σ_max, skipping ~30–40% of denoising steps. No training change — the model has trained to denoise at all σ levels. Pure inference speedup.
2. **Adaptive cycle count**: Run 1 cycle for fast screening, 3 for standard, 5+ for difficult targets. The random-cycle training (§5) enables this.

### Validation Experiments

1. **Ablation: co-evolution rank** r ∈ {4, 8, 16, 32} — diminishing returns threshold; compare with and without vector marginal profile
2. **UOT sparsity**: Measure effective support of T* across heads; verify per-head ε learns multi-scale behavior
3. **Multi-scale ε effect**: Compare fixed multi-scale ε [0.5,1,2,4] vs uniform ε=1.0; measure per-head transport sparsity and downstream accuracy
4. **Transport vs hot-spots**: Correlate T* mass with known functional sites
5. **ν distribution on Ag-Ab**: CDR-epitope residues should show high ν
6. **EGNN trajectory**: Track x_res across 48 blocks — verify progressive structure formation, measure per-block RMSD improvement
7. **EGNN γ distribution**: After training, visualize γ signs across heads/blocks; verify attraction/repulsion pattern
8. **Anytime performance**: Plot structure quality (LDDT, DockQ) vs cycle count (1–5) at inference
9. **AF3 parity benchmark**: CASP15/16 targets, DockQ for complexes

---

## Appendix A: Review Decisions Log

### v4.5 → v4.6 Changes

| Change | Section | Rationale |
|---|---|---|
| FourierEmbedding: frozen random projection replaces deterministic sinusoidal | §9.2 | AF3 Algorithm 22: `cos(2π(t·w + b))` with w,b~N(0,1) frozen. Incoherent basis uniformly represents all noise levels. Boltz-1 identical. |
| AdaLN: sigmoid-bounded scale replaces unbounded (1+γ) | §9.3 | AF3 Algorithm 26: `sigmoid(Lin(s)) * LN(a, affine=False) + LinNoBias(s)`. Scale bounded [0,1], conditioning LN'd. Prevents scale explosion. |
| AdaLN-Zero output gates on attention + transition | §9.3 | AF3 Algorithm 24: `sigmoid(Lin(s, w=0, b=-2)) ⊙ output`. Each block starts as near-identity (sigmoid(-2)≈0.12). Critical for training stability. |
| c_noise = log(σ/σ_data)·0.25 (was log(σ)/4) | §9.2, §12 | Boltz-1 `c_noise`: `log(sigma/sigma_data) * 0.25`. Matches AF3 Algorithm 21 line 8. |
| Coordinate input scaling: c_in = 1/√(σ²+σ_data²) (was 1/σ_data) | §9.2 | AF3 Algorithm 20 line 2: `r_noisy = x_noisy / √(t̂²+σ_data²)`. Normalizes to unit variance at all noise levels. Was leaving high-σ inputs unnormalized (std≈σ/σ_data≈10 at σ=160). |
| LayerNorm on Fourier features before t_proj | §9.2 | AF3 Algorithm 21 line 9: `s += LinNoBias(LN(n))`. Boltz-1 `norm_fourier`. Stabilizes conditioning injection. |
| LayerNorm on pair features before pair_bias_proj | §9.3 | AF3 Algorithm 24 line 8: `b_ij = LinNoBias(LN(z_ij))`. Normalizes frozen pair features. |
| t_proj and pair_bias_proj → LinearNoBias | §9.2, §9.3 | AF3 uses LinearNoBias for Fourier-to-single and pair bias projections. |

### v4.4 → v4.5 Changes

| Change | Section | Rationale |
|---|---|---|
| Diffusion UOT blocks removed (2 blocks, 9M params) | §9.1, §9.2, §14.2 | Trunk h_res already encodes all inter-token structural context. At high σ, geometry is noise (UOT suppresses it). At low σ, trunk encoding is already accurate. AF3 similarly does not re-run Pairformer per diffusion step. |
| μ, ν removed from diffusion interface | §9.2 | Marginals only needed for UOT attention; no UOT blocks in diffusion means no marginals needed |
| h_res NOT frozen before diffusion (end-to-end gradient) | §9.1, §9.2, §13.4, §13.6 | AF3 does not freeze trunk representations. End-to-end gradient teaches trunk to produce h_res useful for atom-level placement, not just distogram. One σ per training step → one diffusion forward/backward in graph. |
| All four losses train trunk end-to-end | §13.4 | L_diff, L_lddt backprop through atom blocks → h_cond → h_res → trunk. Previously only L_disto, L_trunk_coord reached trunk. |
| ε reverted to [0.5, 1.0, 2.0, 4.0], K=20 uniform | §7.2, §8 | Empirical convergence: K=20 reaches ~1e-6 for all ε values. FlashSinkhorn makes K=20 affordable. No warm/cold distinction. |
| Mixed precision: BF16 trunk representation, FP32 coordinates + Sinkhorn | §13 (planned) | h_res path tolerates BF16; EGNN accumulates small displacements across 200+ updates; Sinkhorn log-domain needs FP32 |
| Parameters: ~229M → ~220M | §14.3 | −9M from removing diffusion UOT blocks |

### v4.3 → v4.4 Changes

| Change | Section | Rationale |
|---|---|---|
| IFT backward replaced with unrolled differentiable Sinkhorn | §7.5, §13.4, §13.6 | Empirical: IFT gradients for Q, K, w_dist show 1.5–3.5× median error vs unrolled, error does NOT decrease with more Sinkhorn iterations. IFT computes gradients at theoretical fixed point the network never reaches. Unrolled gives exact gradients for actual K-iteration computation. |
| SinkhornFunction custom autograd removed | §13.6 | Plain differentiable loop; standard autograd through K iterations; no custom backward needed |
| IFT backward Triton kernel removed from engineering priorities | §18 | No longer needed; Flash-Sinkhorn backward stores K intermediate states per tile instead |
| IFT section (adjoint derivation, convergence analysis) removed | §7.5 | Replaced with unrolled backward description |
| Atom blocks increased from 3 to 10 | §9, §14, §17 | AF3 uses 24 atom transformer blocks per diffusion step (not 3). Our 3 was severely undersized for sidechain packing and ligand pose refinement. 10 is conservative given our trunk already handles global structure via EGNN. |
| Diffusion parameter total updated: ~10M → ~13M | §14.2 | 10 atom blocks × ~400K each = 4M (was 1.2M) |
| Grand total updated: ~226M → ~229M | §14.3 | +2.8M from atom blocks |
| ε stays at [0.5, 1.0, 2.0, 4.0]; K raised to 20 uniform | §7.2, §8 | Empirical convergence data: K=4 gives 30% residual (unusable), K=7 gives 3% (marginal), K=20 gives ~1e-6 (converged). With unrolled backprop, forward output quality directly determines gradient quality. FlashSinkhorn Triton kernel makes K=20 affordable. No warm/cold distinction needed. |
| Flash-Sinkhorn engineering: migrate from ot-triton-lab/flash-sinkhorn | §18 | MIT-licensed Triton implementation provides IO-aware tiling, streaming LSE, O(nd) memory, 32× forward speedup. Adapt: general cost, per-head ε, EGNN centroid, unrolled backward. 4-phase migration plan. |

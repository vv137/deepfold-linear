# Protein Complex Structure Prediction Model: Full Design Specification v7.0

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
| H_res | Token-level attention heads (16), used in both MHA and Sinkhorn |
| d_h | Per-head dimension = d / H_res = 32 |
| H_msa | MSA attention heads (8) |
| H_atom | Atom attention heads (4) |
| r | Co-evolution rank (16) |
| L | Number of Token OT blocks in trunk (48) |
| L_atom | Number of Atom blocks per diffusion step (10) |
| r_h | Per-head characteristic distance (learnable, init 10 Å) |
| d_low | Distogram low-rank interaction dimension (64) |
| σ_data | EDM data noise scale (16 Å) |
| token_idx | (N_atom,) int — maps each atom to its token |
| Lin(a→b) | nn.Linear(a, b, bias=True) — standalone projections (input embed, coord output, loss heads) |
| Lin_nb(a→b) | nn.Linear(a, b, bias=False) — projections after LayerNorm (attention Q/K/V/O, Sinkhorn Q/K, SwiGLU) |
| LN(a) | nn.LayerNorm(a) — with learnable γ, β |
| LN_Lin(a→b) | LayerNorm(a) followed by Linear(a, b, bias=False). LN β provides the bias; Linear bias redundant |

**Bias convention**: Projections that follow a LayerNorm use `bias=False` because LN's learnable β already provides the additive shift. This includes trunk MHA projections (W_Q, W_K, W_V, W_O), Sinkhorn projections (W_Q_sink, W_K_sink), SwiGLU projections, and LN_Lin composites. **Exception**: W_gate uses `bias=True` (provides per-residue offset for tanh gate). In diffusion AtomBlock (AF3 Alg 24), W_Q uses `bias=True` per AF3 convention. Standalone projections that don't follow LN (input embedding, coordinate output, loss heads, atom-to-token encoder output) use `bias=True`.

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
| Balanced Sinkhorn attention | Automatic hot-spot focus, sparse transport |
| EGNN coordinate update in OT blocks | SE(3) equivariant structure refinement, no augmentation needed for trunk |
| Diffusion module for sampling | Stochastic multi-modal sampling; AF3-style augmentation (lightweight, local) |
| Remove triangle attention | Unnecessary with explicit coordinates |
| Low-rank co-evolution (rank 16) | Capture multi-modal 2nd-order statistics without pair rep |
| x_res → cost (geometry-aware Sinkhorn) | Geometry modulates attention; coordinates refine through EGNN each block |
| Transport-weighted average | Length-invariant output |
| Unrolled Sinkhorn backward | Exact gradient for K-iteration computation; IFT gives wrong gradients at incomplete convergence |

### 1.3 Core Contributions

1. **Flash Sinkhorn Attention**: Biology-aware sparse transport without pair representation.
2. **O(N) memory architecture**: No N×N stored matrices. Reference implementation uses O(N²) for co-evolution and distogram; tiled versions planned.
3. **Low-rank co-evolution aggregation (rank 16)**: 2nd-order co-evolution statistics compressed into single representation via rank-16 outer product; r-dimensional co-evolution profile provides per-head bias for Sinkhorn cost.
4. **Bond-aware position encoding**: 68-bin scheme unifying sequence separation, chain identity, and covalent bonds.
5. **EGNN coordinate update in OT blocks**: Transport-weighted centroid displacement with input-dependent per-head γ gate (residue-specific attraction/repulsion). SE(3) equivariant by construction — no augmentation in trunk. Coordinates refine across 48 blocks × 3 cycles. Fused into Flash-Sinkhorn kernel at ~10% extra cost.
6. **Geometry-aware Sinkhorn**: x_res → metric cost; coordinates improve each block via EGNN → geometry cost becomes increasingly reliable across blocks.
7. **Unrolled Sinkhorn backward**: Standard autograd through K iterations; exact gradient for actual computation, not theoretical fixed point.
8. **Transport-weighted average**: Length-invariant output via row-sum normalization of the transport plan.

### 1.4 Balanced Transport with Uniform Marginals

Balanced Sinkhorn transport uses uniform marginals (1/N for all positions). The transport plan T satisfies T·1 = 1/N and T^T·1 = 1/N — every position sends and receives equal total mass. The transport-weighted centroid normalizes naturally:

    centroid_i = (Σ_j T_ij x_j) / (Σ_j T_ij)

WHERE mass flows is determined by the cost matrix (feature similarity + geometry + position bias), not by learned marginals. This eliminates the complexity of per-layer marginal projections while MHA handles all feature mixing.

### 1.5 Honest Complexity Statement

The architecture is **designed** for O(N) memory: no component inherently requires N×N storage. The **reference PyTorch implementation** uses Python-level tiling (tile_size=64) for co-evolution aggregation and distogram loss, keeping their peak memory at O(tile²·d) ≈ constant. The remaining O(N²) bottleneck is `cdist` in Token OT blocks, which will be eliminated by the Flash-Sinkhorn kernel (computing distances on-the-fly per tile). Until that kernel is implemented, the reference implementation is O(N²) memory due to cdist only.

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
    MSA blocks × 4 → m, h_res, coevol_bias                           (invariant, no coords)
    Token OT blocks × 48:
      Each block:
        MHA → h_res update                                          (invariant)
        Sinkhorn transport + EGNN → x_res update: x += Σ_h γ_h(h_i) · (x_i − T_norm^h @ x) (equivariant)
      Geometry cost uses latest x_res each block
    Re-center x_res (once per cycle, float32 hygiene)

  freeze h_res, x_res
  L_trunk = log_distance_mse(x_res, x_0) and/or smooth_lddt         direct EGNN supervision

[Diffusion — 200 steps (inference), sampled σ (training)]
  AF3-style augmentation for SE(3).
  for each denoise step:
    x_res ← scatter_mean(x_atom, token_idx)
    h_step ← SingleConditioning(h_res, σ)                            conditioning
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

MSA is defined only for protein (and optionally RNA/DNA) tokens. N_prot ≤ N is the number of tokens with MSA data. The MSA module (§6) operates on these tokens only; non-MSA tokens receive no co-evolution signal and get zero co-evolution bias.

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
w_rel_res:  (H_res, 68) per layer — Token OT position bias.  Init: zeros. With decay (unlike Swin — unbounded pos_bias destabilizes Sinkhorn backward at small ε). Each of the 48 Token OT blocks has its own w_rel_res.
w_rel_msa:  (H_msa, 68)           — MSA attention position bias.  Init: zeros. With decay (same reason). Shared across MSA blocks.
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

The trunk runs a **randomly sampled** number of recycling passes (1 to max_cycles). Each cycle: (1) embed fresh MSA and refine h_res with MSA blocks (invariant, sequence/co-evolution), (2) refine h_res (invariant, via MHA) and x_res (equivariant, via balanced Sinkhorn transport) with Token OT blocks. Only the last cycle backpropagates; all earlier cycles run under `torch.no_grad()`.

**Random cycle count** makes the architecture an anytime algorithm. The network must produce good output whether it gets 1 cycle (48 OT blocks from noise) or 3 cycles (144 blocks of refinement). At inference, the user chooses: 1 cycle (fast), 3 (standard).

**No coordinate embedding into h_res**: Geometry enters through Sinkhorn cost → transport plan → coordinate update.

**No augmentation for trunk**: Coordinate updates use only relative vectors — SE(3) equivariant by construction.

### 5.2 Trunk Loop

```python
def trunk_forward(seq_features, msa_features, c_atom, p_lm,
                  bond_matrix, chain_id, global_idx, token_idx,
                  msa_token_mask, x_0_res=None):

    # ---- Input embedding ----
    h_res = Lin(38, 512)(cat(token_type, profile, del_mean, has_msa))   # (N, 512)
    h_res = h_res + atom_to_token_encoder(c_atom, p_lm, token_idx, N)   # + atom info

    # Initial coordinates: random noise, centered
    x_res = torch.randn(N, 3) * sigma_data
    x_res = x_res - x_res.mean(dim=0, keepdim=True)

    # Precompute position bins (68-bin encoding)
    pos_bins = compute_bins(chain_id, global_idx, bond_matrix)  # (N, N)

    # ---- Sample cycle count ----
    if training:
        num_cycles = randint(1, max_cycles + 1)       # uniform over {1, 2, 3}
    else:
        num_cycles = inference_cycles                  # default 3, configurable

    for cycle in range(num_cycles):
        is_last = (cycle == num_cycles - 1)

        # ---- Per-cycle MSA subsampling: fresh MSA each cycle ----
        m = Lin(34, 64)(subsample_msa(msa_features))   # (S, N_prot, 64)

        # ---- MSA module: 4 blocks (invariant, no coordinates) ----
        m, h_res, coevol_bias = msa_module(m, h_res, msa_token_mask)

        # ---- Token OT blocks × 48 ----
        for block in ot_blocks:
            if is_last:
                h_res, x_res = block(h_res, x_res, pos_bins)
            else:
                with torch.no_grad():
                    h_res, x_res = block(h_res, x_res, pos_bins)

        # ---- Re-center coordinates (once per cycle, not per block) ----
        x_res = x_res - x_res.mean(dim=0, keepdim=True)

        if not is_last:
            h_res = h_res.detach()
            x_res = x_res.detach()

    # ---- Losses (last cycle only, in graph) ----
    L_disto = distogram_loss(h_res, x_0_res)
    L_trunk_coord = log_distance_mse(x_res, x_0_res) if x_0_res is not None else 0.0

    return h_res, x_res, L_disto, L_trunk_coord
```

### 5.3 Cycle Count Distribution

| Parameter | Value |
|---|---|
| max_cycles (training) | 3 |
| Sampling distribution | Uniform over {1, 2, 3} |
| inference_cycles (default) | 3 |
| inference_cycles (fast) | 1 |

### 5.4 Recycling Design Rationale

| Mechanism | Purpose |
|---|---|
| Random cycle count | Anytime algorithm; works with 1–3+ cycles; no fixed schedule |
| no_grad on all but last cycle | Memory: always 52 blocks in graph regardless of cycle count |
| MHA + Sinkhorn every OT block | Features contextualized, coordinates refined continuously (48 updates/cycle) |
| Re-centering once per cycle | Float32 hygiene; coordinate updates are exactly translation-equivariant |
| L_trunk on final x_res (log-distance MSE) | Direct gradient to gate and Sinkhorn parameters |
| No coordinate embedding into h_res | Geometry enters through Sinkhorn cost → transport plan → displacement |
| No augmentation | Coordinate updates are SE(3) equivariant by construction |
| Per-cycle MSA subsampling | Fresh MSA embedding each cycle; m does not carry across cycles |
| No warm-start carry | Balanced OT converges fast; each block starts from uniform init |

---

## 6. MSA Module

### 6.1 Design Intent

- **Scope**: MSA processing operates on protein (and optionally RNA/DNA) tokens only. Non-MSA tokens (ligands, ions) receive no co-evolution signal and get zero co-evolution bias. The MSA block internally works on the protein token subset; results are scattered back to the full token set.
- **Row-wise attention**: Within-sequence residue interactions with position bias.
- **Column weighted mean**: Learned aggregation capturing 1st-order co-evolution.
- **Low-rank co-evolution aggregation**: c_ij = (1/S) Σ_s U_si^T V_sj captures 2nd-order co-evolution statistics. Rank r=16 allows multiple co-evolution modes (active site, allosteric site, interface, fold stability). The r-dimensional co-evolution vector per pair is reduced to a scalar weight for aggregation and retained as a full profile for per-head cost bias.
- **Co-evolution bias**: Computed once after all 4 MSA blocks. The r-dimensional co-evolution profile c̄_i ∈ R^16 projects to per-head bias via zero-init coevol_to_marginal Lin(16→H_res). Non-MSA tokens get zero bias. The bias is added to the Sinkhorn cost matrix in each trunk block.

### 6.2 Full Block

```python
def msa_block(m, h_res, msa_mask):
    """
    m:         (B, S, N_prot, 64)   MSA for protein tokens only
    h_res:     (B, N, 512)          all tokens
    msa_mask:  (B, S, N_prot)       True where MSA data valid (S-axis masking)
    """

    # ---- 1. Single → MSA injection (protein tokens only) ----
    # msa_token_mask: (N,) bool — True for tokens with MSA data
    h_prot = h_res[msa_token_mask]                              # (N_prot, 512)
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
    # h_res-conditioned: col_query(h_prot) adds to column scores
    m_n = LN(m)                                                 # (S, N_prot, 64)
    col_scores = Lin(64 → 1)(m_n)                              # (S, N_prot, 1) per-sequence scores
    col_scores = col_scores + Lin(512 → 1)(h_prot)[None]       # (1, N_prot, 1) h_res query bias
    col_scores = col_scores.masked_fill(~msa_mask.unsqueeze(-1), -1e9)  # mask invalid S entries
    alpha = softmax(col_scores, dim=0)                          # (S, N_prot, 1) softmax over S
    col_agg = (alpha * m_n).sum(dim=0)                          # (N_prot, 64)
    h_prot_update = Lin(64 → 512)(col_agg)                     # (N_prot, 512)
    h_res[msa_token_mask] = h_res[msa_token_mask] + h_prot_update  # scatter back

    # ---- 4. Low-rank co-evolution aggregation (rank 16, tiled) ----
    # c_ij ∈ R^r: multi-modal co-evolution vector per pair
    # Scalar weight for aggregation, full r-dim profile for cost bias
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

            # r-dim profile for cost bias
            c_bar_accum[i0:ie] += c_tile.sum(dim=1)            # (ti, 16)

    h_res = h_res + Lin(512 → 512)(h_agg)                      # (N, 512)
    c_bar = c_bar_accum / N                                     # (N, 16) per-position profile

    # ---- 5. SwiGLU transition ----
    m_n = LN(m)                                                 # (S, N, 64)
    gate  = Lin(64 → 256)(m_n)                                  # (S, N, 256)
    value = Lin(64 → 256)(m_n)                                  # (S, N, 256)
    m = m + Lin(256 → 64)(SiLU(gate) * value)                   # (S, N, 64)

    # ---- 6. Row dropout (protect query row 0) ----
    # Drop entire MSA rows (sequences) during training; row 0 (query) always kept
    if training:
        mask = torch.bernoulli(torch.full((S, 1, 1), 0.85))    # (S, 1, 1)
        mask[0] = 1.0                                            # protect query row
        m = m * mask / 0.85

    return m, h_res, c_bar


def msa_module(m, h_res, msa_token_mask, msa_mask):
    """
    Run 4 MSA blocks, then compute co-evolution bias for Sinkhorn cost.
    Returns m, h_res, coevol_bias

    The coevol_bias is added to the Sinkhorn cost matrix in each trunk block.
    """
    c_bar = None
    for b in range(4):
        m, h_res, c_bar = msa_block(m, h_res, msa_mask)

    # ---- Co-evolution bias (post-loop) ----
    # r-dim co-evolution profile → per-head bias (protein tokens only)
    # c_bar from last block only (earlier blocks' c_bar not accumulated)
    # Non-protein tokens: coevol_bias = 0
    # coevol_to_marginal: zero-init, so bias starts at zero
    coevol_bias = torch.zeros(N, H_res, device=h_res.device)
    coevol_bias[msa_token_mask] = coevol_to_marginal(c_bar[msa_token_mask])  # Lin(16→H_res), zero-init

    return m, h_res, coevol_bias
```

### 6.3 Why Standard Softmax in MSA Row Attention

MSA row-wise attention operates within individual homologous sequences where every position is informative — there are no "irrelevant" residues within a single alignment row. The sparse-transport motivation for Sinkhorn OT applies to cross-chain and long-range residue interactions, not within-sequence MSA processing. Standard softmax is appropriate and cheaper here.

---

## 7. Balanced Sinkhorn Transport

### 7.1 Cost Matrix

For each head h, between token positions i and j:

```
C_feat^(h) = 1 − cos(Q_s_i, K_s_j)                  feature cost ∈ [0, 2]
C_geom^(h) = d_ij / (r_h + d_ij)                     geometry cost ∈ [0, 1)
C_ij^(h)   = (1 − w_h) · C_feat + w_h · C_geom       weighted blend
```

where:

- `Q_s = L2_norm(W_Q_sink(LN(h)))`, `K_s = L2_norm(W_K_sink(LN(h)))` — separate projections from MHA
- `w_h = sigmoid(α_h)` — per-head mixing gate (init −1.0 → feature-heavy start)
- `r_h` — per-head characteristic distance (learnable, init 10 Å)
- `d_ij = ||x_res_i − x_res_j||₂`

**Three-gate coordinate update**:

| Gate | Type | Shape | Controls |
|---|---|---|---|
| Mixing α_h | `sigmoid(α_h)` | (H,) per-head | Feature vs geometry cost blend |
| Intensity λ_h | `tanh(λ_h_raw) / H` | (H,) per-head | Force magnitude and direction |
| Mobility G_i | `sigmoid(W_gate(LN(h)))` | (N, 1) per-residue | Whether residue responds to forces |

**Cosine feature cost**: L2-normalization bounds cos(Q_s, K_s) ∈ [-1, 1], so C_feat ∈ [0, 2]. All cost terms are O(1).

**Mixing gate initialization**: α_h = −1.0 → sigmoid(−1) ≈ 0.27 → starts feature-heavy. Expected: front layers feature-heavy → back layers geometry-heavy.

**Learnable r_h**: Per-head characteristic distance. Heads specialize in different spatial scales — contacts (~5 Å) to domains (~30 Å).

**Geometry term design**:

- f(d) = d/(r_h + d) is concave, f(0) = 0, bounded in [0, 1).
- f ∘ d satisfies the triangle inequality (metric).
- r_h learnable per-head (init 10 Å): roughly contact range (~8 Å) scale.
- Per-block, per-head r_h + α_h allows multi-scale behavior: some heads/layers develop strong geometry sensitivity, others stay feature-heavy.

**Scale budget** (all terms O(1)):

- C_feat: cosine distance → [0, 2]
- C_geom: d/(r_h+d) → [0, 1)
- Weighted sum: [0, 2]

### 7.2 Transport Problem

Balanced optimal transport with uniform marginals:

```
T*^(h) = argmin_{T ≥ 0}  ⟨C^(h), T⟩ + ε^(h) · KL(T ‖ 1/N ⊗ 1/N)
         subject to:  T·1 = 1/N,  T^T·1 = 1/N
```

**Multi-scale ε (fixed per head, not learned)**:

```python
self.register_buffer('eps', torch.tensor([
    0.5, 0.5, 0.5, 0.5,     # heads 0–3:   sparse (contacts)
    1.0, 1.0, 1.0, 1.0,     # heads 4–7:   balanced (structural)
    2.0, 2.0, 2.0, 2.0,     # heads 8–11:  smooth (secondary structure)
    4.0, 4.0, 4.0, 4.0,     # heads 12–15: diffuse (global)
]))
```

| Head group | ε | Transport character |
|---|---|---|
| 0–3 | 0.5 | Sparse: mass flows to specific contacts |
| 4–7 | 1.0 | Balanced: standard structural interactions |
| 8–11 | 2.0 | Smooth: broad environment sensing |
| 12–15 | 4.0 | Diffuse: global state, allostery |

| Parameter | Value | Rationale |
|---|---|---|
| ε | Fixed per-head [0.5, 1.0, 2.0, 4.0] × 4 | Multi-scale transport without gradient complexity |
| Marginals | Uniform 1/N (balanced OT) | No learned marginal projections needed |
| K | 20 (all blocks, uniform) | Converges to ~1e-6 residual |

**Why balanced OT** (vs UOT): Uniform marginals eliminate mu_proj/nu_proj, avoid FP32 marginal headaches, and converge faster. MHA handles feature mixing — Sinkhorn only determines WHERE to transport coordinates.

**Why fixed ε**: Learnable ε creates gradient complications. The mixing gate α_h provides sufficient expressivity for head specialization.

### 7.3 Sinkhorn Iterations (Log-Domain, Balanced)

```python
def balanced_sinkhorn(C, eps, K, mask=None):
    """
    C:    (B, H, N, N) cost matrix
    eps:  (H,) fixed per-head entropic regularization
    K:    number of iterations
    Returns: T (B, H, N, N) transport plan
    """
    log_K = -C / eps[None, :, None, None]
    log_marginal = -log(N_valid)                        # uniform 1/N

    log_u = zeros(B, H, N, 1)
    log_v = zeros(B, H, 1, N)

    for k in range(K):
        log_u = log_marginal - logsumexp(log_K + log_v, dim=-1, keepdim=True)
        log_v = log_marginal - logsumexp(log_K + log_u, dim=-2, keepdim=True)

    T = exp(log_u + log_K + log_v)
    return T
```

**Balanced = simpler iterations**: No κ damping (κ=1 always). Each iteration is a simple row/col normalization in log-domain. Faster and more stable convergence than UOT.

### 7.4 Output: Transport-Weighted Centroid

```python
def transport_centroid(T, x_res):
    """Transport-weighted centroid for coordinate update. No value aggregation."""
    x_centroid = einsum('bhij,bjc->bhic', T, x_res)                # (B, H, N, 3)
    return x_centroid
```

Feature mixing is handled by MHA (§8 Step 1). The transport plan T is used ONLY for coordinate movement.

### 7.5 Backward: Triton Unrolled (exact gradients, O(N) memory)

Autograd through K Sinkhorn iterations, implemented as Triton tiled kernels. Cost tiles recomputed on-the-fly — no N×N matrix ever stored in HBM.

**Iteration order**: col-first, row-last. The last row update ensures T's row sums are exactly 1/N (machine precision), making the centroid numerically stable without explicit normalization.

**Saved for backward**: iteration history K × 2 × (B,H,N) ≈ 1-5 MB (negligible).

**Backward kernels** (4 Triton kernels in `balanced_sinkhorn_bwd.py`):

1. `_centroid_bwd_D_kernel` — D[i] = Σ_j T_norm·(grad_xc·x) contraction
2. `_centroid_bwd_kernel` — g_u, g_v, grad_x_transport + direct cost gradient
3. `_balanced_row_bwd_kernel` — backward through row update (2-pass: LSE then softmax grad)
4. `_balanced_col_bwd_kernel` — backward through col update

**Gradient accuracy** (vs PyTorch autograd, same input):

| Parameter | Rel error | Note |
|---|---|---|
| x_res | < 1e-4 | Exact |
| Q_s, K_s | < 1e-3 | Exact (FP32 rounding only) |
| α_h | < 1e-3 | Exact |
| r_h | < 6e-5 | Exact |

**Performance** (CUDA, B=1, H=16, d_h=32, K=20):

| N | Triton | Autograd | Speedup | Memory ratio |
|---|---|---|---|---|
| 64 | 1.6ms, 1MB | 3.0ms, 30MB | 1.9× | 0.04× |
| 256 | 4.5ms, 22MB | 3.0ms, 218MB | 0.7× | 0.10× |
| 512 | 8.3ms, 28MB | 5.5ms, 816MB | 0.7× | 0.03× |
| 768 | 21.4ms, 35MB | 11.4ms, 1807MB | 0.5× | 0.02× |

Speed: Triton backward is compute-bound at large N (cost tile recomputation × K iterations × atomic_add contention). Memory: O(N) — **up to 52× reduction** vs materialized O(N²).

---

## 8. Token OT Block (MHA + Balanced Sinkhorn)

48 blocks in the trunk. Each block separates feature contextualization (MHA) from coordinate transport (balanced Sinkhorn). Returns h (invariant) and x_res (equivariant).

```python
class TokenOTBlock(nn.Module):
    def __init__(self, d=512, H=16, r_0=10.0, dropout=0.0):
        # ---- [Step 1] MHA: Contextualization ----
        self.ln_mha = LayerNorm(d)
        self.W_Q = Lin(d, d, bias=False)
        self.W_K = Lin(d, d, bias=False)
        self.W_V = Lin(d, d, bias=False)
        self.W_O = Lin(d, d, bias=False)
        self.W_G = Lin(d, d, bias=False)                  # output gate
        self.pos_bias = PositionBias(H, 68)

        # ---- FFN (SwiGLU) ----
        self.ln_ff = LayerNorm(d)
        self.swiglu = SwiGLU(d, d*4, d)
        self.ff_dropout = Dropout(dropout)

        # ---- [Step 3] Sinkhorn Transport ----
        self.ln_sink = LayerNorm(d)
        self.W_Q_sink = Lin(d, d, bias=False)
        self.W_K_sink = Lin(d, d, bias=False)
        self.alpha_h = Parameter(full(H, -1.0))         # mixing gate → feature-heavy
        self.r_h = Parameter(full(H, r_0))               # per-head char. distance
        self.register_buffer('eps', ...)                  # [0.5, 1.0, 2.0, 4.0] × 4

        # ---- [Step 4] Coordinate Update ----
        self.ln_gate = LayerNorm(d)
        self.W_gate = Lin(d, 1, bias=True)               # mobility gate (sigmoid)
        self.lambda_h_raw = Parameter(zeros(H))           # intensity gate (tanh/H)

    def forward(self, h, x_res, pos_bins, mask=None):
        # Returns: h, x_res

        # ---- [Step 1] Flash MHA with 68-bin position bias ----
        h_n = self.ln_mha(h)
        Q, K, V, G = project_qkvg(h_n)
        att_out = flash_diffusion_attn(Q, K, V, pos_bias.weight, pos_bins, mask)
        h = h + W_O(sigmoid(G) * att_out)                 # AF3 Alg 24: gate before W_O

        # ---- FFN (SwiGLU) ----
        h = h + dropout(swiglu(ln_ff(h)))

        # ---- [Step 2–3] Balanced Sinkhorn Transport ----
        Q_s = L2_norm(W_Q_sink(ln_sink(h)))
        K_s = L2_norm(W_K_sink(ln_sink(h)))
        x_centroid = balanced_sinkhorn_transport(Q_s, K_s, x_res, eps, alpha_h, r_h)

        # ---- [Step 4] Three-gate coordinate update ----
        vec_h = x_centroid - x_res[:, None]               # (B, H, N, 3)
        G_i = sigmoid(W_gate(ln_gate(h)))                  # (B, N, 1) mobility
        lambda_h = tanh(lambda_h_raw) / H                  # (H,) intensity
        x_res = x_res + G_i * einsum('h,bhic->bic', lambda_h, vec_h)

        return h, x_res
```

### 8.1 Coordinate Update: SE(3) Equivariance Proof

The update is: `Δx_i = G_i · Σ_h λ_h · (centroid_h(i) − x_i)`, where G_i = sigmoid(W @ LN(h_i) + b) is an invariant scalar, λ_h = tanh(raw_h)/H is a constant scalar, and centroid_h(i) = Σ_j T_ij x_j.

**Rotation** (x → xR, R ∈ O(3)): T is invariant (computed from invariant cost). G_i is invariant. So `centroid(xR) − xR = (centroid(x) − x)R`. Rotation factors out. ✓

**Translation**: For balanced OT with marginals 1/N, T rows sum to 1/N. The centroid uses normalized transport: `centroid(x+t) − (x+t) = centroid(x) − x` when rows are properly normalized. Re-centering once per cycle (§5.2) handles float32 drift.

### 8.2 Three-Gate System

| Gate | Formula | Range | Init | Purpose |
|---|---|---|---|---|
| Mixing α_h | `sigmoid(α_h)` | (0, 1) | σ(−1)≈0.27 | Feature vs geometry blend |
| Intensity λ_h | `tanh(λ_h_raw) / H` | (−1/H, 1/H) | 0 (dormant) | Attraction (λ>0) or repulsion (λ<0) |
| Mobility G_i | `sigmoid(W_gate(LN(h)) + b)` | (0, 1) | ~0.5 | Per-residue readiness to move |

**Dormant at init**: λ_h_raw = 0 → tanh(0) = 0 → no coordinate movement at init. The network learns to activate coordinate updates as training progresses.

**No weight decay** on α_h, r_h, λ_h_raw (small scalar parameters). W_gate gets weight decay.

### 8.3 Separation of Concerns

| Component | Role | Parameters |
|---|---|---|
| MHA | Feature contextualization | W_Q, W_K, W_V, W_O, pos_bias |
| SwiGLU | Nonlinear feature refinement | gate_proj, value_proj, out_proj |
| Sinkhorn | Coordinate transport plan | W_Q_sink, W_K_sink, α_h, r_h, eps |
| Gate | Coordinate update | W_gate, λ_h_raw |

MHA handles all feature mixing. Sinkhorn handles all geometry. The two share no projections. Position bias (68-bin) lives in MHA — Sinkhorn gets positional information implicitly through feature cost.

### 8.4 Dropout Rules

- MHA attention weights: **dropout** (standard)
- MHA output (W_O), SwiGLU output: **dropout** (standard)
- Transport plan T: **NEVER dropout** (must stay doubly-stochastic)

---

## 9. Diffusion Module (v5 — AF3 style, no pair representation)

### 9.1 Design Intent

The diffusion module is an encoder-transformer-decoder that refines noisy atom coordinates. It adopts AF3's architecture — atom encoder aggregates atom features to token level, a token-level transformer reasons globally, then an atom decoder broadcasts back and predicts coordinate updates. Diffusion multiplicity follows Boltz-1 (16 samples) rather than AF3 (48 samples).

**Key divergence from AF3**: No O(N²) pair representation z. The 68-bin PositionBias (same as trunk) replaces AF3's pair attention bias. All attention uses custom Triton kernels for O(N) memory.

**End-to-end gradient**: h_res is NOT detached. L_diff, L_lddt backpropagate through the full diffusion module → SingleConditioning → h_res → trunk.

**Proper EDM preconditioning**: `x_pred = c_skip · x_noisy + c_out · F_θ(c_in · x_noisy)`. The v4.5 residual form (`x_noisy + c_out · Δx`) is replaced.

### 9.2 Architecture (~155M params)

```
Input: h_res(B,N,512), s_inputs(B,N,512), c_atom(B,M,128), x_noisy(B,M,3), σ, pos_bins(B,N,N)

1. SingleConditioning (~4M)
   c_noise = log(σ/σ_data) * 0.25
   t_emb = FourierEmbedding(256)(c_noise)                → (B, 256)
   s = h_res + s_inputs + Lin(LN(t_emb))                 → (B, N, 512)
   s = s + SwiGLU_transition(LN(s))  ×2

2. AtomEncoder (3 layers, ~4.5M)
   a = c_atom + Lin(x_noisy * c_in)                      → (B, M, 128)
   atom_cond = Lin(gather(s, token_idx))                  → (B, M, 128)
   ×3:
     a = WindowedAtomSelfAttn(a, atom_cond, W=32, H=128)  — Triton kernel
     s = s + AtomToTokenCrossAttn(Q=s, K/V=a)              — sparse Triton kernel
   a_enc = a                                               — cached for decoder skip

3. DiffusionTransformer (24 layers, ~101M)
   ×24:
     s = s + AdaLN-Zero gated FlashSelfAttn(s, pos_bias[pos_bins])  — Triton kernel
     s = s + AdaLN-Zero gated SwiGLU(s)
   s = LN(s)

4. AtomDecoder (3 layers, ~4.5M)
   a = a_enc + zero_init_Lin(gather(s, token_idx))        — skip + broadcast
   atom_cond_dec = Lin(gather(s, token_idx))
   ×3:
     a = a + TokenToAtomCrossAttn(Q=a, K/V=s)              — dense Triton kernel
     a = WindowedAtomSelfAttn(a, atom_cond_dec)
   Δx = zero_init_Lin(LN(a))                              → (B, M, 3)

5. EDM Output
   x_pred = c_skip · x_noisy + c_out · Δx
   x_pred = recenter(x_pred, atom_mask)
```

### 9.3 Triton Attention Kernels

All diffusion attention uses custom Triton kernels for O(N) memory:

| Kernel | File | Grid | Memory |
|--------|------|------|--------|
| Flash token self-attn | `kernels/flash_diffusion_attn.py` | (B\*H, ⌈N/64⌉) | O(N) — online softmax, pos_bias gathered per-tile from (H,68) |
| Windowed atom self-attn | `kernels/flash_atom_attn.py` | (B\*H, ⌈M/32⌉) | O(M) — 32×128 fits in SRAM, direct softmax |
| Atom→token cross-attn | `kernels/cross_attn_kernel.py` | (B\*H, N) | O(N) — sparse, each token gathers its atoms |
| Token→atom cross-attn | `kernels/cross_attn_kernel.py` | (B\*H, ⌈M/64⌉) | O(M) — tiled over atom queries |

All kernels include forward + backward + Python reference for testing.

### 9.4 Cross-Attention Design

**AtomToTokenCrossAttn** (encoder): Tokens query, atoms key/value. Sparse — each token only attends to its own atoms via `token_atom_starts/counts`. Replaces scatter_mean with a learnable soft aggregation.

**TokenToAtomCrossAttn** (decoder): Atoms query, tokens key/value. Dense — each atom attends to all N tokens (N ≤ 384 in training). Replaces hard gather/broadcast with a learnable distribution.

Both use sigmoid(G)·W_O gating pattern, LN on Q and K/V inputs.

### 9.5 Stability Features

- **AdaLN-Zero gates**: w=0, b=-2 → sigmoid(-2) ≈ 0.12 at init → near-identity blocks
- **zero_init_Lin**: Coordinate output and token-to-atom broadcast projection initialized to zeros
- **Proper c_skip**: At σ→0, c_skip→1 and c_out→0, so x_pred → x_noisy (identity)
- **FP32 FourierEmbedding**: Frozen weights stay float32 even under bf16 autocast
- **Re-centering**: Subtract center-of-mass per sample after each denoising step

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

Balanced Sinkhorn transport handles crop boundaries through masking: padded positions are masked out of the Sinkhorn iterations, and uniform marginals are computed over valid positions only (1/N_valid). No special boundary treatment needed beyond standard padding masks.

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

**Weighted rigid alignment** (Kabsch/SVD): Before computing MSE, the ground truth coordinates are rigidly aligned to the prediction via weighted SVD. This removes the rotational/translational degrees of freedom that the SE(3)-invariant model cannot control. The alignment is computed under `no_grad` with `.detach()` — gradients flow only through `x_pred` in the MSE, not through the SVD. AF3 does this.

EDM preconditioning:

| Function | Formula |
|---|---|
| c_skip | σ_data² / (σ² + σ_data²) |
| c_out | σ · σ_data / sqrt(σ² + σ_data²) |
| c_in | 1 / sqrt(σ² + σ_data²) |
| c_noise | ln(σ/σ_data) · 0.25 |

The network predicts the denoised signal. The EDM weighting ensures equal contribution across noise levels.

### 11.2 Smooth LDDT Loss (AF3 convention)

```
L_lddt = 1 − (1/|P|) Σ_{(i,j)∈P} (1/4) Σ_{δ∈{0.5,1,2,4}} sigmoid(δ − |d_ij^pred − d_ij^true|)
```

**Slope = 1** (no scaling factor). AF3 uses implicit slope=1 in `sigmoid(t - dev)`. A steeper slope (e.g. 10) causes sigmoid saturation at both extremes, killing gradients early in training when predictions are poor.

**Per-type pair cutoff** (AF3 convention):

- Nucleotide atom pairs: d_ij^true < 30 Å (nucleic acids have larger inter-residue spacing)
- All other pairs: d_ij^true < 15 Å
- A pair is "nucleotide" if either atom belongs to a DNA/RNA chain

**No σ-weighting**: Unlike L_diff, L_lddt is not weighted by σ. AF3 adds it to the total loss with weight 1.0 at all noise levels.

**Zero-pair fallback**: When no valid pairs exist for a sample (e.g. all atoms masked), the loss returns 1.0 (worst case), not 0.0.

**Memory**: O(M²) where M = number of atoms — materializes pairwise distance matrices. At crop=384 tokens (~2000 atoms), this is ~100–200 MB, comparable to the trunk's Sinkhorn cost matrix.

### 11.3 Low-Rank Bilinear Distogram Loss (Tiled)

The distogram loss forces h_res to encode pairwise distance information. Since this architecture has no pair representation, the distogram is one of only two training signals for pairwise reasoning (the other being Sinkhorn geometry cost in the forward pass). Expressivity here matters.

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

### 11.5 Trunk Coordinate Losses

Direct structural supervision on the trunk's EGNN-refined coordinates. This is critical: without it, the last Token OT block's γ gate receives zero gradient (its x_res output is not used by any other loss). With it, every γ gate in all 48 blocks receives direct gradient through the chain of EGNN updates.

Two complementary trunk coordinate losses are available (either or both can be enabled via null weights):

**Log-distance MSE** (default, `w_trunk_logmse`):

```
L_trunk_logmse = mean_{i≠j, valid} [log(d_pred_ij + 1) - log(d_true_ij + 1)]²
```

Soft distance weighting via log compression: large-distance errors are dampened, providing global topology supervision without hard cutoffs. Gradient ∝ 1/((d+1)·d) — natural soft decay. Triton-fused fwd+bwd with O(1) extra memory (no N² materialization).

**Smooth LDDT** (optional, `w_trunk_slddt`):

```
L_trunk_slddt = smooth_lddt(x_res_final, x_0_res)
```

Same formulation as §11.2 (slope=1, per-type cutoff 15/30 Å). Hard cutoff for local precision.

Both operate on token-level (Cα) coordinates and are invariant to rigid-body transformations — no alignment needed. Log-distance MSE provides global topology signal; smooth LDDT provides local precision.

### 11.6 Total Loss

```
L = w_diff · L_diff + w_lddt · L_lddt + w_disto · L_disto
    + w_trunk_slddt · L_trunk_slddt + w_trunk_logmse · L_trunk_logmse

Default: L = L_diff + L_lddt + 0.2 · L_disto + 0.1 · L_trunk_logmse
(w_trunk_slddt = null → disabled)

Fine-tuning: L += α_bond · L_bond
```

Any weight set to null disables that loss term entirely (no computation).

**Gradient flow**:

- L_diff, L_lddt, L_bond → diffusion module parameters (atom blocks, coord output)
- L_disto → trunk h_res → all trunk parameters (representation supervision)
- L_trunk_logmse / L_trunk_slddt → trunk x_res → all EGNN γ gate parameters (W_gamma weight + bias) + all Sinkhorn params via geometry cost (coordinate supervision)

L_disto and L_trunk are complementary: L_disto teaches h_res to encode distance information (pairwise distance classification into 39 bins), L_trunk teaches the EGNN γ gate to produce accurate coordinates. Together they provide the trunk with both representation-level and coordinate-level structural signals.

where L_bond penalizes deviations from ideal bond lengths and angles (standard force-field terms, fine-tuning only).

---

## 12. Diffusion Schedule

| Parameter | Value |
|---|---|
| σ_data | 16 Å |
| σ_max | 160 Å |
| σ_min | 0.002 Å |
| P_mean | −1.2 |
| P_std | 1.5 (AF3 aligned) |
| Training | σ = σ_data · exp(N(−1.2, 1.5²)), median ≈ 4.8 Å |
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
# Post-LN projection names — scale-invariant under LayerNorm, no decay
POST_LN = {'w_q', 'w_k', 'w_v', 'w_g', 'w_o', 'swiglu'}
# Bounded or zeros-init gating params — no decay
NO_DECAY_SPECIAL = {'w_dist_raw', 'pos_bias'}  # gamma handled by explicit check

param_groups = [
    {   # Standalone weight matrices (not post-LN): weight decay
        'params': [...],  # input embed, atom encoder, cond_proj, loss heads, AdaLN gates
        'weight_decay': 0.01
    },
    {   # No decay: LN γ/β, biases, post-LN projections (scale-invariant),
        #           bounded params (w_dist_raw, W_gamma)
        #   NOTE: pos_bias gets decay — unbounded, and large values destabilize
        #         Sinkhorn backward at small ε (exp(-pos/ε) saturates).
        'params': [...],  # layernorm, ln, bias, w_q/k/v/g/o, swiglu,
                          # w_dist_raw, w_gamma
        'weight_decay': 0.0
    },
]

optimizer = AdamW(param_groups, lr=0.0, betas=(0.9, 0.95), eps=1e-8)

# Learning rate schedule: AF3 warmup + plateau + exponential decay
scheduler = AlphaFoldLRScheduler(
    optimizer,
    base_lr=0.0,              # start from zero
    max_lr=1e-3,              # peak LR
    warmup_steps=1000,        # linear warmup
    start_decay_after=50000,  # plateau until this step
    decay_every=50000,        # decay interval
    decay_factor=0.95,        # multiplicative factor per interval
)
max_grad_norm = 10.0

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
| Head collapse | Per-head multi-scale ε [0.5,1,2,4] | Heads forced to different transport regimes |
| Length bias | Transport-weighted average | Division by T_sum |
| Transport output overflow | Running-max subtraction | Same trick as FlashAttention |
| EGNN coordinate collapse | Per-head signed γ | Repulsive heads counteract attractive heads |
| EGNN coordinate explosion | γ zeros init + AdamW decay + tanh bound | Step sizes bounded (-1,1); must earn influence |
| EGNN last-block dead gradient | L_trunk_coord on final x_res | Direct gradient to all γ through EGNN chain |
| EGNN float32 centroid drift | Re-center once per cycle | Not per-block; EGNN is exactly translation-equivariant |
| Geometry bias at high noise (trunk) | Per-head w_dist diversity | Some heads geometry-sensitive, others robust |
| Geometry bias at high noise (diffusion) | σ-conditioned geo_gate | Learned suppression at high σ |
| Geometry bias domination | w_dist sigmoid-bounded (0,1), init ≈0.12 | Per-block per-head; can't exceed 1.0 |
| Position bias scale | w_rel zeros init (per-layer), with decay | Starts at zero; per-layer allows specialization; decay prevents unbounded growth that destabilizes Sinkhorn backward at small ε |
| Co-evolution noise | Zero-init coevol_to_marginal | Must earn influence; starts at zero bias |
| Uniform marginals | Balanced OT with 1/N marginals | No learned marginal projections; simpler and more stable |
| SE(3) equivariance (trunk) | EGNN: relative vectors only | No augmentation needed |
| SE(3) equivariance (diffusion) | AF3-style augmentation | Standard, lightweight |
| MSA overfitting | Row dropout p=0.15 | Drop entire sequences |
| Gradient bias (Sinkhorn) | Unrolled backprop through K iterations | Exact gradient for actual computation; IFT gives biased gradients on Q,K,w_dist_raw |
| Parameter oscillation | EMA decay=0.999 | Smoothed parameters |
| Crop boundary | Padding mask in Sinkhorn iterations | Uniform marginals over valid positions only |
| Diffusion output at init | Zero-init final Linear | Identity prediction initially |
| Large coordinate scale (diffusion) | Division by σ_data in atom embedding | O(1) embedding input |

### 13.4 Backward Pass

Only the last recycling cycle is in the computation graph. All earlier cycles run under `torch.no_grad()`. The graph contains: 4 MSA blocks + 48 Token OT blocks (with gradient checkpointing) + loss heads. The diffusion module has its own separate graph.

**Gradient sources** (four losses):

```
L = L_diff + L_lddt + 0.2 · L_disto + 0.5 · L_trunk_coord
```

| Loss | Gradient target | What it trains |
|---|---|---|
| L_diff | ∂/∂x_atom_pred → atom blocks → h_cond → h_res | Atom blocks + entire trunk (end-to-end) |
| L_lddt | ∂/∂x_atom_pred → atom blocks → h_cond → h_res | Same path as L_diff |
| L_disto | ∂/∂h_res^(48) | All trunk params: MSA blocks, MHA, Sinkhorn, SwiGLU |
| L_trunk_coord | ∂/∂x_res^(48) | All EGNN γ + all trunk params via geometry cost |

**Backward through diffusion module**: Standard backprop through 10 atom blocks. Gradients flow through h_cond into h_res and back through the entire trunk. One σ sampled per training step — only one diffusion forward/backward in the graph. Cost: ~2× diffusion forward.

**Backward through Token OT blocks** (the complex part):

Each block b receives upstream gradients ∂L/∂h^(b+1) and ∂L/∂x^(b+1) and propagates backward through 4 stages in reverse:

**Stage 4 — SwiGLU backward**: Standard backprop through SiLU, element-wise multiply, linear projections. Produces ∂L/∂(ff_gate, ff_value, ff_out, ln_ff params) and ∂L/∂h_pre_transition.

**Stage 3 — EGNN backward**: The EGNN update is `x_new = x + Σ_h γ_h · (x − T_norm^(h) @ x)`.

```
∂L/∂γ_h = Σ_i grad_x_i^T · (x_i − centroid_i^(h))      scalar per head

∂L/∂x_j += Σ_i Σ_h γ_h · (−T_norm_ij^(h)) · ∂L/∂x_new_i  coordinate gradient

∂L/∂T_norm_ij^(h) = −γ_h · (∂L/∂x_new_i)^T · x_j          transport plan gradient
```

This produces gradients for γ (trains EGNN step sizes), for input x (cascades to previous block), and for T_norm (enters Sinkhorn unrolled backward).

**Stage 2 — Sinkhorn transport backward (unrolled through iterations)**:

The attention output produces ∂L/∂T_norm from the h_res path (through V aggregation). Combined with ∂L/∂T_norm from EGNN (stage 3), autograd differentiates backward through the K Sinkhorn iterations:

```
Iteration K → K-1 → ... → 1 → 0:
    Each logsumexp backward produces ∂L/∂C (accumulated across iterations)
    ∂L/∂C chains to:
        ∂L/∂W_Q, ∂L/∂W_K    via content term (-LN(Q)^T LN(K)/√d_h)
        ∂L/∂w_rel             via position bias
        ∂L/∂w_dist_raw        via geometry term (through algebraic sigmoid)
        ∂L/∂x_res             via distance in geometry term
```

No custom backward needed — standard autograd through the Sinkhorn loop. The K intermediate (log_u, log_v) states are recomputed during gradient checkpointing and consumed by autograd.

**x_res accumulates gradients from THREE sources per block**:

1. EGNN backward: through centroid displacement
2. Geometry cost backward: through distance term in C_ij (via unrolled Sinkhorn)
3. Pass-through from block b+1

**Stage 1 — Pre-norm + projections backward**: Standard linear backward for W_Q, W_K, W_V, W_G, ln_attn.

**Backward through MSA module**: Standard backprop through 4 blocks + post-loop coevol_bias. Gradients from Sinkhorn cost bias flow through coevol_to_marginal → co-evolution c_bar (last block) → MSA representation → all 4 blocks' h_res.

**Gradient checkpointing**: Each of the 48 Token OT blocks is checkpointed. During backward, the block's forward pass (including K Sinkhorn iterations) is recomputed from boundary activations (h_res, x_res, log_u, log_v). The K intermediate states are created fresh and consumed by autograd. Memory per block during backward: K × H × N × 4 bytes for Sinkhorn intermediates + O(N²) for cost matrix (eliminated by Flash-Sinkhorn tiling).

**Total backward memory**: O(48 × N × d) for boundary activations + O(N²) for one block's recomputed cost matrix. With Flash-Sinkhorn: O(N × d) total.

**Complete gradient flow**:

```
L_trunk_coord → x_res^(48) → EGNN^(48) → γ^(48) ✓
                                ↓
                            T_norm^(48) → Sinkhorn^(48) unrolled → C^(48) → W_Q,W_K,w_dist_raw,w_rel
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
            if 'gamma' in name:                    # EGNN γ: noise init for symmetry breaking
                param.data.normal_(0, 1e-4)
            elif 'w_dist_raw' in name:               # geometry: alg_sigmoid(0) = 0.5 midpoint
                nn.init.zeros_(param)
            elif 'w_rel' in name:                  # position bias: off at init
                nn.init.zeros_(param)
            elif 'coevol_to_marginal' in name:      # co-evolution bias: off at init (zero-init replaces alpha_coevol)
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
| γ (EGNN) | randn × 1e-4 | Near-dormant with symmetry-breaking noise; each layer starts with unique γ |
| w_dist_raw | 0 (alg_sigmoid = 0.5) | Midpoint geometry at init; per-block per-head; algebraic sigmoid bounded (0,1); no weight decay |
| w_rel | zeros (per-layer for trunk) | Position bias off at init; each Token OT block learns its own sequence-distance priors |
| coevol_to_marginal | **all zeros** | Co-evolution bias off at init; MSA processing must stabilize first |
| LN γ, β | (1, 0) | Identity normalization at init |
| Input embedding Lin(38→512) | Xavier + bias=zeros | Standard; network adjusts column scales for different input features |
| Atom encoder projections | Xavier | Standard; encoder output is O(1) at init |
| Diffusion coord output Lin(128→3) | **all zeros** | Identity denoising at init (EDM preconditioning assumes this) |
| Distogram head (W_u, W_v, W_bin) | Xavier | Random predictions at init → strong initial gradients to h_res |
| FourierEmbedding w, b | N(0,1), **frozen** | Random Fourier features; never updated (AF3 Algorithm 22) |
| AdaLN-Zero gates (_attn_gate, _transition_gate) | **w=0, b=−2** | sigmoid(−2)≈0.12 at init → atom blocks start as near-identity (AF3 Algorithm 24) |
| AdaLN s_scale, s_bias | Xavier | Conditioning projections; sigmoid bounds scale to [0,1] |

**Staged activation dynamics**: At the start of training:

1. Content cost (`-LN(Q)^T LN(K)/√d_h`) and geometry cost (w_dist starts at midpoint 0.5) both active from the start
2. Transport plans are approximately uniform (all costs similar) → EGNN centroids ≈ global mean → γ≈0 (noise init) means minimal coordinate movement, with symmetry broken across layers
3. L_disto provides strong gradients (random predictions, high loss) → h_res rapidly learns distance-relevant features
4. As h_res improves → content cost becomes informative → transport plans sharpen → EGNN centroids become meaningful
5. L_trunk_coord gradient pushes γ away from zero (no weight decay on γ) → coordinates begin to refine
6. Better coordinates → geometry cost becomes informative → positive feedback loop. Heads that don't benefit from geometry can reduce w_dist; the algebraic sigmoid's heavy tail keeps gradient alive at both boundaries

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
for block in token_ot_blocks:
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
| MSA col: Lin(64→1, bias=F), Lin(512→1, bias=F), Lin(64→512, bias=F) | per block × 4 | 4 × 33K = 133K | Xavier | ✅ |
| MSA co-evol: U,V Lin(64→16, bias=F) | per block × 4 | 4 × 2 × 1K = 8K | Xavier | ✅ |
| MSA co-evol scalar weight: Lin(16→1, bias=F) | per block × 4 | 4 × 16 = 64 | Xavier | ✅ |
| MSA co-evol agg: LN_Lin(512→512), Lin(512→512, bias=F) | per block × 4 | 4 × 2 × 262K = 2.1M | Xavier | ✅ |
| MSA coevol_to_marginal: Lin(16→16, bias=F) | shared (post-loop) | 256 | **zeros** | ✅ |
| MSA SwiGLU: Lin(64→256, bias=F)×2, Lin(256→64, bias=F) | per block × 4 | 4 × 49K = 196K | Xavier | ✅ |
| MSA position: w_rel_msa | (8, 68) | 544 | zeros | ✅ |
| **MSA total** | | **~2.7M** | | |
| Token OT: LN + Q,K,V,G Lin(512→512, bias=F) | per block × 48 | 48 × 4 × 262K = 50.3M | Xavier | ✅ |
| Token OT: W_O Lin(512→512, bias=F) | per block × 48 | 48 × 262K = 12.6M | **Xavier/√48** | ✅ |
| Token OT: SwiGLU Lin(512→2048, bias=F)×2, Lin(2048→512, bias=F) | per block × 48 | 48 × 3.1M = 150M | Xavier | ✅ |
| Token OT: γ (EGNN step sizes) | per block × 48, (16,) each | 768 | **N(0,1e-4)** | ❌ (tanh-bounded) |
| Token OT: ε (per-head, fixed) | buffer (16,), shared all blocks | 0 (not a parameter) | [1.0]×4+[2.0]×4+[4.0]×4+[8.0]×4 | N/A |
| Token position: w_rel_res (×48) | (16, 68) per block | 52K | zeros | ✅ |
| Geometry: w_dist_raw | per block × 48, (16,) each | 768 | **0** (alg_sigmoid = 0.5) | ❌ (alg_sigmoid-bounded) |
| LN γ, β (attention + transition) | per block × 48, 2 × (512,) each | 48 × 2K = 98K | (1, 0) | ❌ |
| **Token OT total** | | **~213M** | | |
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

AF3 uses c_s=384 (single) and c_z=128 (pair) with 48 Pairformer blocks and 24 atom transformer blocks per diffusion step. Our model uses d=512 (single only) with 48 Token OT blocks and 10 atom blocks per diffusion step. No Sinkhorn blocks in diffusion — the trunk's h_res provides all inter-token context. End-to-end gradient from diffusion loss through h_res into the trunk.

---

## 15. Complexity Analysis

| Component | Compute | Memory (reference) | Memory (tiled) |
|---|---|---|---|
| MSA row attention (×4) | O(S · N² · d_msa) | O(S · N · d_msa) | same (FlashAttn) |
| Co-evolution outer product (×4) | O(S · N² · r) | O(tile² · r) per tile | O(tile² · r) per tile |
| Co-evolution aggregation (×4) | O(N² · d) | O(tile² + tile · d) per tile | O(tile² + tile · d) per tile |
| Token OT (×48, K iters) | O(48 · K · N² · d_h) | **O(N²)** per cdist | O(H · N) with FlashSinkhorn |
| Distogram | O(N² · d_low) | O(tile² · d_low) per tile | O(tile² · d_low) per tile |
| Atom self-attention | O(N_atom · w · d_atom) | O(N_atom · d_atom) | same |
| **Reference total** | | **O(N²)** for cdist only | |
| **Tiled total** | | | **O(N · d + S · N · d_msa)** |

**Tiling status**: Co-evolution and distogram now use Triton kernels with autograd (forward + backward) for training. Both eliminate O(N²) autograd storage via flash-style recomputation. The remaining O(N²) bottleneck is cdist in Token OT blocks, eliminated by the Flash-Sinkhorn kernel which computes distances on-the-fly per tile.

---

## 16. Information Flow Summary

```
[TRUNK — random 1–5 recycling passes, SE(3) equivariant]

h_res = Lin(38→512)(cat(token_type, profile, del_mean, has_msa))  (invariant)
h_res += atom_to_token_encoder(c_atom, p_lm)                      (invariant)
x_res = randn(N, 3) * σ_data, centered                            (equivariant)
num_cycles ~ Uniform{1..5} (training)

Recycling × num_cycles (all but last: no_grad; last: backprop):
  Fresh MSA subsample → m = Lin(34→64)(msa_feat)       ← per-cycle re-embedding
  MSA Block × 4 (no coordinates — pure sequence/co-evolution):
    row attn(w_rel_msa) → m                           ← within-sequence context
    col weighted mean(h_res-conditioned) → h_res      ← 1st-order co-evolution
    rank-16 outer_product(U, V)/S → c_ij ∈ R^16       ← 2nd-order co-evolution
    Lin(16→1)(c_ij) → scalar weight → h_res           ← co-evolution aggregation
  coevol_to_marginal(c̄) → coevol_bias                 ← zero-init, per-head bias

  Token OT Block × 48:
    MHA(Q,K,V,G, pos_bias) → h_res                     ← feature contextualization (invariant)
    LN(Q_s)^T LN(K_s) + w_rel + w_dist·f(d) → C      ← content + position + geometry
    Sinkhorn(C, uniform 1/N) → T*                       ← balanced transport plan
    T* @ x_res → centroid                               ← transport-weighted centroid
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
| Trunk blocks | 48 Pairformer (pair + single tracks) | 48 Token OT (single track + equivariant coords) |
| Triangle attention/update | ✅ | ❌ |
| Co-evolution signal | Outer product → z_ij (full pair) | Rank-16 outer product → scalar agg weight + r-dim cost bias profile (tiled) |
| Token-level attention | Softmax + gate | MHA + balanced Sinkhorn transport (multi-scale ε) |
| Structure refinement (trunk) | N/A (Pairformer has no structure module) | EGNN via transport-weighted centroid displacement |
| SE(3) equivariance (trunk) | N/A | EGNN equivariant (no augmentation) |
| SE(3) equivariance (diffusion) | Augmentation | Augmentation (same) |
| Position encoding | Relative bins (clipped ±32) + chain offset | 68-bin + covalent bond |
| Geometry in attention | Via pair rep (implicit) | Explicit x_res → metric cost, updated every block via EGNN |
| Atom coordinate embed | c_atom = 128 per diffusion step | d_atom = 128 per diffusion step (same) |
| Atom attention | Local window 32→128 | Local window 32 (same pattern) |
| Atom pair bias | c_atompair = 16, intra-token | d_pair = 16, intra-token (same) |
| Recycling | 3 fixed | 1–3 random (EGNN refines coords across 48 blocks/cycle) |
| Marginal handling | N/A (softmax) | Uniform 1/N (balanced OT); no learned marginals |
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
2. **Template integration** (v2): Template coordinates → pairwise distances → binned → bias in Sinkhorn cost matrix. No pair representation needed — distance bins directly added to C_ij as another bias term, analogous to w_rel.
3. **Detailed training stages**: Learning rate schedule per stage, crop size ramp (Section 10.3).

### Engineering (Priority Order)

1. ~~**Diffusion module stability (AF3 alignment)**~~ ✅ v4.6
   - FourierEmbedding (frozen random), AdaLN (sigmoid-bounded), AdaLN-Zero output gates
   - c_in coordinate scaling, c_noise formula, LayerNorm on Fourier/pair features, LinearNoBias

0.5. ~~**Flash-Sinkhorn Triton kernels + gradient correctness**~~ ✅

- All 13 Triton kernels support batch dim: grid `(B*H, n_tiles)`
- Mask support: `-1e9` bias on padded positions in all score computations
- Flash forward (Triton): N=384 in 0.6ms, O(N) memory
- Training backward: Triton exact unrolled (autograd through K iterations) — correct gradients for ALL parameters including cost projections and coevol_to_marginal
- IFT backward (CG-based): implemented but gives incorrect gradients for Sinkhorn with row-normalized output (∂T_norm/∂log_u = 0 at fixed point). Reserved for future O(N) backward via tiled unrolled approach.
- FP32 log-domain: all Sinkhorn iterations in FP32 to prevent BF16 underflow
- Kabsch alignment: disable autocast for SVD/det (BF16 not supported)
- Coevol kernel wired into MSA (training + inference), distogram kernel wired into losses (eval)
- trunk_block.py: flash Sinkhorn for both training and inference (O(N) memory)

1. **Flash-Sinkhorn kernel — further optimization from [flash-sinkhorn](https://github.com/ot-triton-lab/flash-sinkhorn) (MIT license)**

   Foundation: FlashSinkhorn (Ye et al. 2026, arXiv:2602.03067) provides fused Triton kernels for IO-aware streaming Sinkhorn with O(nd) memory. Already supports unbalanced OT via `reach` parameter, transport application `P*V`, analytic gradients, early stopping, and half-precision. 32× forward / 161× end-to-end speedups on A100.

   **Adaptations needed** (fork and extend):

   | Component | FlashSinkhorn (current) | Our requirement | Adaptation |
   |---|---|---|---|
   | Cost function | Squared Euclidean `‖x−y‖²` | `-LN(Q)^T LN(K)/√d_h + w_rel[bin] + w_dist·f(d)` | Replace distance computation per tile with dot-product + two additive biases (position lookup, geometry from cdist) |
   | ε | Scalar (`blur²`) | Per-head (H,) buffer `[0.5,1,2,4]×4` | Broadcast: load ε_h once per head per tile |
   | OT parameterization | `reach` (KL penalty for UOT) | Balanced OT (uniform 1/N marginals) | Set `reach=∞` (no KL penalty) or use balanced mode directly |
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

2. ~~**Co-evolution tiling Triton kernel**~~ ✅ (fused Python tile loops from Section 6.2)
   - Forward kernel with mask support + w_tile (B,N,N) cache for backward
   - Two backward kernels: dU (i-centric, includes dw_weight/db_weight via c_r recompute) + dV (j-centric, fused D-chunk loop for dh_coevol)
   - IEEE FP32 precision in all tl.dot calls (R=16 matrices too small for TF32 benefit)
   - S_CHUNK ≥ 16 enforced for tl.dot K constraint; handles S < 16 MSA depths
   - Enabled for training in msa.py (was inference-only); PyTorch fallback retained for CPU
3. ~~**Distogram tiling Triton kernel**~~ ✅ (fused Python tile loops from Section 11.3)
   - Forward kernel with mask support (token_pad_mask → per-tile validity)
   - Two backward kernels: dU+dW+dbias (i-centric) + dV (j-centric)
   - Zero O(N²) backward storage: recomputes Z/logits/softmax and target bins from x_true per tile
   - N=512: 222MB → 24MB (9.4x memory reduction vs PyTorch tiled autograd)
   - Enabled for training in losses.py; PyTorch fallback retained for CPU
4. **Mixed precision strategy** (BF16 for most, FP32 for Sinkhorn log-domain)

**Current memory profile** (no checkpointing, crop=384, B200 192GB):

- Flash Sinkhorn: O(N) per block, cost tiles computed on the fly in Triton kernels
- No O(N²) cost matrix stored; autograd saves only O(N) dual variables (log_u, log_v) per iteration
- At crop=1024: ~3.2GB — still fits comfortably in 192GB
- Checkpointing removed for DDP compatibility; B200 has sufficient memory

**Status**: Flash-Sinkhorn (item 1) is the critical path for scaling beyond crop ~2048 where the O(N²) Sinkhorn intermediates (~50GB at N=2048) would exceed single-GPU memory. Until then, dense PyTorch Sinkhorn on high-memory GPUs (B200, H100 80GB) is sufficient. Items 2–3 are performance optimizations. Item 4 is active (BF16 autocast for trunk, FP32 for Sinkhorn log-domain).

### Inference Optimizations

1. **Trunk-initialized diffusion**: Instead of starting diffusion from pure noise (σ_max=160Å), initialize atom coordinates from the trunk's refined x_res plus noise at σ_init ≈ 3–5Å (matching trunk's typical RMSD). Enter the Karras schedule at σ_init instead of σ_max, skipping ~30–40% of denoising steps. No training change — the model has trained to denoise at all σ levels. Pure inference speedup.
2. **Adaptive cycle count**: Run 1 cycle for fast screening, 3 for standard. The random-cycle training (§5) enables this.

### Validation Experiments

1. **Ablation: co-evolution rank** r ∈ {4, 8, 16, 32} — diminishing returns threshold; compare with and without co-evolution cost bias
2. **Transport sparsity**: Measure effective support of T* across heads; verify per-head ε produces multi-scale behavior
3. **Multi-scale ε effect**: Compare fixed multi-scale ε [0.5,1,2,4] vs uniform ε=1.0; measure per-head transport sparsity and downstream accuracy
4. **Transport vs hot-spots**: Correlate T* mass with known functional sites
5. **Transport mass on Ag-Ab**: CDR-epitope residues should receive high transport mass from partner chain
6. **EGNN trajectory**: Track x_res across 48 blocks — verify progressive structure formation, measure per-block RMSD improvement
7. **EGNN γ distribution**: After training, visualize γ signs across heads/blocks; verify attraction/repulsion pattern
8. **Anytime performance**: Plot structure quality (LDDT, DockQ) vs cycle count (1–5) at inference
9. **AF3 parity benchmark**: CASP15/16 targets, DockQ for complexes

---

## Appendix A: Review Decisions Log

### v4.5 → v4.6 Changes

| Change | Section | Rationale |
|---|---|---|
| FourierEmbedding: frozen random projection replaces deterministic sinusoidal | §9.2 | AF3 Algorithm 22: `cos(2π(t·w + b))` with w,b~N(0,1) frozen. Incoherent basis uniformly represents all noise levels. |
| AdaLN: sigmoid-bounded scale replaces unbounded (1+γ) | §9.3 | AF3 Algorithm 26: `sigmoid(Lin(s)) * LN(a, affine=False) + LinNoBias(s)`. Scale bounded [0,1], conditioning LN'd. Prevents scale explosion. |
| AdaLN-Zero output gates on attention + transition | §9.3 | AF3 Algorithm 24: `sigmoid(Lin(s, w=0, b=-2)) ⊙ output`. Each block starts as near-identity (sigmoid(-2)≈0.12). Critical for training stability. |
| c_noise = log(σ/σ_data)·0.25 (was log(σ)/4) | §9.2, §12 | AF3 Algorithm 21 line 8: `c_noise = log(sigma/sigma_data) * 0.25`. |
| Coordinate input scaling: c_in = 1/√(σ²+σ_data²) (was 1/σ_data) | §9.2 | AF3 Algorithm 20 line 2: `r_noisy = x_noisy / √(t̂²+σ_data²)`. Normalizes to unit variance at all noise levels. Was leaving high-σ inputs unnormalized (std≈σ/σ_data≈10 at σ=160). |
| LayerNorm on Fourier features before t_proj | §9.2 | AF3 Algorithm 21 line 9: `s += LinNoBias(LN(n))`. Stabilizes conditioning injection. |
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

# MHC Custom Operator Layer

## Location

`vllm/model_executor/kernels/mhc/` — 4 files, 4 platform-specific implementations of the same 4 operations, plus a dispatch layer at `vllm/model_executor/layers/mhc.py`.

## The Four Operations

| Op | Input | Output | Purpose |
|----|-------|--------|---------|
| `mhc_pre` | residual (..., hc_mult, H) bf16 + learned params | `post_mix`, `comb_mix`, `layer_input` (..., H) bf16 | Generate gating coefficients and reduce hc_mult streams to one |
| `mhc_post` | x, residual, `post_layer_mix`, `comb_res_mix` | `residual_out` (..., hc_mult, H) | Inject layer output back into each HC residual stream |
| `mhc_fused_post_pre` | post inputs + pre params | `residual_cur`, `post_mix_cur`, `comb_mix_cur`, `layer_input_cur` | Single-kernel post + next pre |
| `hc_head_fused` | hidden_states (..., hc_mult, H), hc_fn | out (..., H) | Terminal reduction at model head, collapses hc_mult → 1 |

## Mathematical Formulas

### mhc_pre (Mix Coefficients + Sinkhorn Normalization)

Let `T` = num_tokens, `hc_mult` = M, `H` = hidden_size, `M3 = M*2 + M*M`.

```
x = residual_flat → (T, M*H)  fp32
mixes = x @ fn^T               (T, M3)
rms = rsqrt( mean(x^2) + rms_eps )
mixes = mixes * rms

# Pre-mix (learned sigmoid gate)
pre_mix  = sigmoid(mixes[:, :M]        * scale[0] + base[:M])        + pre_eps

# Post-mix (learned sigmoid gate with fixed multiplier)
post_mix = sigmoid(mixes[:, M:2*M]     * scale[1] + base[M:2*M])     * post_mult_value

# Comb-mix (softmax + Sinkhorn, per-token doubly-stochastic matrix)
comb_logits = mixes[:, 2*M:].view(T, M, M) * scale[2] + base[2*M:].view(1, M, M)
comb_mix = softmax(comb_logits, dim=-1) + sinkhorn_eps
comb_mix = comb_mix / (comb_mix.sum(-2) + sinkhorn_eps)
for _ in range(sinkhorn_repeat - 1):     # row/col normalize iteratively
    comb_mix = comb_mix / (comb_mix.sum(-1) + sinkhorn_eps)
    comb_mix = comb_mix / (comb_mix.sum(-2) + sinkhorn_eps)

# Reduce hc_mult streams → single stream via weighted average
layer_input = sum_i pre_mix_i * residual_i  →  (T, H) bf16
```

The Sinkhorn normalization iteratively forces the comb_mix matrix to be doubly stochastic (rows and columns each sum to 1). This is a critical component for the MHC mixing mechanism.

### mhc_post (Residual Mixing)

```
out_j = post_layer_mix_j * x  +  sum_i comb_res_mix_ij * residual_i
```

Where:
- `post_layer_mix` is a per-stream scalar gate (hc_mult values)
- `comb_res_mix` is an (hc_mult × hc_mult) mixing matrix
- `x` is the layer output (broadcast across hc_mult streams)

Equivalently in Einstein notation: `out[jh] = post_mix[j] * x[h] + sum_i comb_mix[ij] * residual[ih]`

### hc_head_fused (Terminal Reduction)

Same core logic as mhc_pre's pre-mix + weighted sum, but without post_mix/comb_mix output:

```
x_flat = hidden_states.flatten(-2)    # (T, M*H)
x_normed = rmsnorm_nw(x_flat)          # weight-free RMSNorm
mixes = linear(x_normed, hc_fn)        # (T, M)
pre = sigmoid(mixes * scale + base) + eps
out[h] = sum_i pre[i] * hidden_states[i, h]   # weighted reduction
```

## Tensor Shape Table

### mhc_pre / mhc_pre_tilelang

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `residual` | (..., M, H) | bf16 | HC residual input |
| `fn` | (M3, M*H) | fp32 | Weight matrix for mix logits |
| `hc_scale` | (3,) | fp32 | Per-mix-type logit scale |
| `hc_base` | (M3,) | fp32 | Per-mix-type logit bias |
| `norm_weight` | (H,) | bf16 | Optional fused RMSNorm weight |
| `post_mix` | (..., M, 1) | fp32 | Post-mix gating coefficients |
| `comb_mix` | (..., M, M) | fp32 | Doubly-stochastic comb mixing matrix |
| `layer_input` | (..., H) | bf16 | Reduced single-stream output |

Where `M = hc_mult`, `M3 = 2*M + M*M`.

### mhc_post / mhc_post_tilelang

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `x` | (..., H) | bf16 | Layer output |
| `residual` | (..., M, H) | bf16 | HC residual input |
| `post_layer_mix` | (..., M, 1) | fp32 | Per-stream post multiplier |
| `comb_res_mix` | (..., M, M) | fp32 | Comb mixing matrix |
| `out` | (..., M, H) | bf16 | Updated residual |

### mhc_fused_post_pre_tilelang

Returns all outputs from both mhc_post and mhc_pre:

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `residual_cur` | (..., M, H) | bf16 |
| `post_mix_cur` | (..., M, 1) | fp32 |
| `comb_mix_cur` | (..., M, M) | fp32 |
| `layer_input_cur` | (..., H) | bf16 |

### hc_head_fused

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `hidden_states` | (..., M, H) | bf16 |
| `hc_fn` | (M, M*H) | fp32 |
| `hc_scale` | (1,) | fp32 |
| `hc_base` | (M,) | fp32 |
| `out` | (..., H) | bf16 |

## Four Implementation Files

### `torch.py` — Pure PyTorch Reference (107 lines)

- `mhc_pre_torch`: Straightforward PyTorch implementation for correctness testing.
- `mhc_post_torch`: Uses `torch.einsum("...ij,...ih->...jh", comb, residual)`.
- `mhc_fused_post_pre_torch`: Not defined (missing from this file; only individual ops).
- `hc_head_fused_torch`: Not defined (missing from this file).

This is the **ground truth** for numerical validation of all other implementations.

### `tilelang.py` — TileLang CUDA Custom Ops (547 lines)

The primary high-performance path for NVIDIA GPUs. All 4 ops are registered via `direct_register_custom_op` with fake impls for meta-device shape inference.

**mhc_pre_tilelang architecture:**

1. **Split-k GEMM** (`tf32_hc_prenorm_gemm` from DeepGEMM): performs the large `x @ fn^T` matmul while also computing `sum(x^2)` as a byproduct. The k-dimension is split into `n_splits` pieces to maximize SM utilization. Outputs two buffers:
   - `gemm_out_mul`: (n_splits, T, M3) — partial GEMM results
   - `gemm_out_sqrsum`: (n_splits, T) — partial squared sums

2. **big_fuse kernel** (`mhc_pre_big_fuse_tilelang`): a single TileLang kernel that consumes the split-k outputs and does everything else:
   - Accumulates `rms` from all splits' `gemm_out_sqrsum`
   - Accumulates `mixes` from all splits' `gemm_out_mul`, then multiplies by `rms`
   - Computes `post_mix` (warp 0-31, thread binding < 32)
   - Computes `comb_mix` with full Sinkhorn iterations
   - Computes `pre_mix` and weighted sum reduction into `layer_input` (warp 32+)

**mhc_post_tilelang:** Direct TileLang kernel performing the einsum + broadcast.

**mhc_fused_post_pre_tilelang:** Full fusion of post + pre:
- Small batch (≤16 tokens): single `mhc_fused_tilelang` kernel that computes post mapping, writes updated residual, and performs GEMM FMA for pre in one pass
- Large batch (>16 tokens): sequential `mhc_post_tilelang` + split-k GEMM + `mhc_pre_big_fuse_tilelang` — avoids register pressure

**hc_head_fused_tilelang:** Two-pass TileLang kernel using cross-thread reducers to avoid materializing intermediate tensors to global memory.

### `aiter.py` — AMD ROCm Aiter Wrapper (139 lines)

- Delegates to `rocm_aiter_ops.mhc_pre` / `rocm_aiter_ops.mhc_post` (from `vllm._aiter_ops`).
- Constraint: requires `hidden_size % 256 == 0`.
- Currently **disabled** in the dispatch layer due to a known accuracy bug at large token counts (`aiter commit b639cb6`).
- Only provides `mhc_pre` and `mhc_post` ops; no fused_post_pre or hc_head.

### `triton.py` — Triton HC Head Fallback (175 lines)

- `rmsnorm_nw`: weight-free RMSNorm Triton kernel (for hc_head).
- `hc_head_reduce_triton_kernel`: x → x_flat → rmsnorm → linear → sigmoid → weighted sum.
- Uses a Triton kernel `_hc_head_reduce_store_kernel` for the reduction across hc_mult streams.
- Only provides `hc_head` op registered as `hc_head_triton`.

## Platform-Specific Dispatch

Dispatch is controlled by `vllm/model_executor/layers/mhc.py` via the `CustomOp` framework:

| Platform | mhc_pre | mhc_post | mhc_fused_post_pre | hc_head |
|----------|---------|----------|--------------------|---------|
| **CUDA** (NVIDIA) | `mhc_pre_tilelang` | `mhc_post_tilelang` | `mhc_fused_post_pre_tilelang` | `hc_head_fused_kernel_tilelang` |
| **HIP** (AMD ROCm) | `mhc_pre_torch` (aiter disabled) | `mhc_post_torch` (aiter disabled) | Not available (raises) | `hc_head_triton` |
| **Native** (CPU) | Not available | Not available | Not available | Not available |

All ops are registered as PyTorch custom ops via `direct_register_custom_op` (`vllm.utils.torch_utils`), which gives them fake implementations for torch.compile / meta-device tracing.

## TileLang Architecture: Split-k GEMM + Big Fuse

The core performance strategy for the pre block on NVIDIA GPUs:

```
residual (T, M, H) bf16
        │
        ▼
┌─────────────────────────────────────────────────┐
│ tf32_hc_prenorm_gemm  (split-k, from DeepGEMM)  │
│   x @ fn^T  +  x^2 sum  in tf32                 │
│   k-dim split into n_splits                     │
│   Consumer grid: n_splits = f(64, M*H, T')      │
└────────────┬───────────────────────┬─────────────┘
             │                       │
    gemm_out_mul            gemm_out_sqrsum
    (n_splits, T, M3)       (n_splits, T)
             │                       │
             ▼                       ▼
┌─────────────────────────────────────────────────────┐
│ mhc_pre_big_fuse_tilelang  (fully fused kernel)     │
│   per-token:                                        │
│     1. rms = rsqrt( sum(sqrsum_splits) / D + eps ) │
│     2. mixes = sum(mul_splits) * rms                │
│     3. post_mix = sigmoid(mixes[M:2M]) * mult       │
│     4. comb_mix = softmax(mixes[2M:]) → Sinkhorn    │
│     5. pre_mix = sigmoid(mixes[:M]) + eps           │
│     6. layer_input = sum(pre * residual[stream])    │
│        (optionally fused with RMSNorm)              │
└─────────────────────┬───────────────────────────────┘
                      │
        post_mix, comb_mix, layer_input
```

The number of splits is auto-tuned:

```python
# From _tilelang_ops.py
def compute_num_split(block_k, k, grid_size):
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    split_k = n_sms // grid_size
    num_block_k = cdiv(k, block_k)
    split_k = min(split_k, num_block_k // 4)   # avoid over-splitting small k
    return max(split_k, 1)
```

This ensures the GEMM tiles evenly across all available SMs while avoiding excessive overhead from many tiny splits.

## Fusion Strategies

### Post-Pre Fusion (`mhc_fused_post_pre_tilelang`)

Two paths based on `num_tokens ≤ fma_token_threshold` (= 16):

**Small batch (fully fused):**
```
mhc_fused_tilelang  (single kernel)
  ┌───────────────────────────────────────────────┐
  │ For each token, thread owns h_iters elements: │
  │   new_r[j] = pm[j]*x[h] + sum cm[k,j]*res[k] │
  │   residual_out[j,h] = new_r[j]                │
  │   acc[n] += sum w[n,j,h] * new_r[j]           │
  │   sqr += sum(new_r[j]^2)                      │
  │ warp_reduce_sum acc/sqr → shared              │
  │ warp 0 finalizes → gemm_out, rp_out           │
  └───────────────────────────────────────────────┘
```
- Auto-heuristics: `tile_n = 2` if `tokens < 8` else `3`
- `n_splits = 8` if `(tokens < 8 and H ≤ 4096)` else `4`
- Eliminates the post residual writeback and re-read entirely.

**Large batch (partially fused):**
- `mhc_post_tilelang` → write residual_cur to HBM
- `tf32_hc_prenorm_gemm` → gemm_out_mul + gemm_out_sqrsum
- `mhc_pre_big_fuse_tilelang` → post_mix + comb_mix + layer_input

At large batch sizes, full fusion would exhaust registers, so the split-k GEMM and big_fuse kernel remain the optimal path.

### Norm Fusion

When `norm_weight` is provided to `mhc_pre_tilelang`:

- `mhc_pre_big_fuse_with_norm_tilelang` fuses RMSNorm into the layer_input write path:
  - **Pass 1**: Compute weighted sum, store temporary bf16 output in shared memory while accumulating squared sum per position
  - **Compute**: `rsqrt = 1 / sqrt(mean(sumsq) + norm_eps)`
  - **Pass 2**: Re-read temporary, scale by `rsqrt * norm_weight`, write to HBM
- Eliminates the materialization of an intermediate un-normalized tensor and a separate RMSNorm kernel launch.

## Design Rationale

### Why Split-k?

The GEMM `x @ fn^T` has shape `(T, M*H) @ (M3, M*H)^T = (T, M3)`. The hidden dimension `M*H` can be very large (model scaling), while `T` may be small (inference batch). A standard 2D grid over (token, M3) would underutilize SMs. Split-k partitions the reduction dimension (k = M*H) across SMs so every SM has work, then the big_fuse kernel does a cheap inter-split reduction.

### Why Two Paths for Fused Post-Pre?

The fused kernel (`mhc_fused_tilelang`) packs both the post and pre GEMM FMA into one kernel but uses many registers. At small T this is tolerable (high arithmetic intensity per thread). At large T, the register pressure from holding all tiles simultaneously would limit occupancy, so the sequential path (three separate kernels) performs better.

### Why Sinkhorn Normalization?

The comb_mix matrix must be doubly stochastic (rows and columns each sum to 1) to represent a valid soft assignment between input and output HC streams. Sinkhorn's algorithm iteratively row/column normalizes a non-negative matrix to achieve this. The number of iterations is a configurable hyperparameter (`sinkhorn_repeat`).

### Platform Strategy

- **CUDA**: TileLang provides a JIT compilation framework that targets native PTX with full control over register usage, shared memory, and warp-level primitives — ideal for the intricate fusion patterns. DeepGEMM handles the split-k GEMM in tf32.
- **ROCm**: Aiter is the AMD-native kernel library but currently has precision issues at large token counts. Falls back to pure PyTorch (torch.py) as a safe bridge.
- **hc_head on ROCm**: Triton provides a portable implementation since it runs on both CUDA and HIP backends. The hc_head kernel is simpler (no Sinkhorn, no gemm_out_mul buffering) so Triton is sufficient.

## Related Notes

- [[V4Architecture]] — Overall DeepSeek V4 architecture and the role of MHC layers
- [[TileLangKernels]] — TileLang JIT kernel infrastructure in vLLM
- [[DeepGEMMIntegration]] — DeepGEMM split-k GEMM wrapper (`vllm/utils/deep_gemm.py`)
- [[CustomOpDispatch]] — The CustomOp framework for platform dispatch

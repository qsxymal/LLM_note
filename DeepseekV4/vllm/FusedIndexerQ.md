# FusedIndexerQ -- Fused RoPE + Quantize Q for the Sparse Indexer

**File:** `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` (Triton fallback)

Fused kernel that applies **forward GPT-J interleaved RoPE** to the indexer's Q projection, quantizes the result, and folds the Q scale into indexer weights. Two quantization paths are supported: **FP8** (default) and **MXFP4** (`use_fp4=True`).

**Relationship with [[FusedInvRopeFP8Quant]]:** These two are the RoPE + quantization fusion pair for opposite ends of the attention pipeline. `FusedIndexerQ` processes the **Q-side** (pre-attention, applying forward RoPE). `FusedInvRopeFP8Quant` processes the **O-side** (post-attention, applying inverse RoPE). Both use GPT-J interleaved RoPE, block-scaled quantization, and Triton backends, but differ in their scale management strategy.

## 1. Inputs and Outputs

### Inputs

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `positions` | `(T,)` | int64 | Token positions for RoPE cos/sin lookup. |
| `index_q` | `(T, H, head_dim)` | bf16 | Q projection from the indexer. `head_dim = 128`. |
| `index_q_cos_sin_cache` | `(max_pos, rope_dim)` | bf16/f32 | Precomputed RoPE cos/sin. `rope_dim = 64` for H128 indexer head. |
| `index_weights` | `(T, H)` | bf16 | Scalar weight per (token, head) from the sparse indexer. |

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_weights_softmax_scale` | float | Softmax temperature scaling factor. |
| `index_weights_head_scale` | float | Per-head scaling factor (typically loaded from weights). |
| `use_fp4` | bool | Selects MXFP4 path when True (default: False = FP8). |

### Outputs

| Path | Output | Shape | dtype | Description |
|------|--------|-------|-------|-------------|
| FP8 | `q_fp8` | `(T, H, head_dim)` | float8_e4m3fn | Quantized Q, **no companion scale tensor**. Scale is folded into `weights_out`. |
| FP8 | `weights_out` | `(T, H)` | float32 | `weights * q_scale * softmax_scale * head_scale` |
| MXFP4 | `q_packed` | `(T, H, head_dim // 2)` | uint8 | 2 E2M1 nibbles per byte (packed values). |
| MXFP4 | `q_scale` | `(T, H, num_blocks)` | uint8 (ue8m0) | Per-block exponent-only scale. `num_blocks = head_dim // 32`. |
| MXFP4 | `weights_out` | `(T, H)` | float32 | `weights * softmax_scale * head_scale` (NO q_scale fold). |

## 2. Weight-Fold Design Rationale

The central design difference between the two paths:

### FP8 Path (Default)

```
weights_out = index_weights * q_scale * softmax_scale * head_scale
```

- Q uses a **single per-token per-head scalar** scale factor.
- This scalar is folded into `weights_out` so the downstream FP8 logits kernel does not need to load and multiply it separately.
- No companion scale tensor is emitted for `q_fp8`.
- The downstream `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` kernels use the pre-scaled `weights` to apply the per-token Q scale inline.

### MXFP4 Path

```
weights_out = index_weights * softmax_scale * head_scale
```

- Q uses **per-block** (32-element) scales that live alongside the packed values in a separate tensor.
- These per-block scales **cannot** be folded into a single per-token weight scalar.
- The `q_scale` tensor is emitted separately (ue8m0 format).
- The downstream MXFP4 logits kernel dequantizes Q by reading both `q_packed` and `q_scale` together.
- `weights_out` carries only the softmax and head scales.

### Why Not Always Fold?

| Aspect | FP8 fold | MXFP4 no-fold |
|--------|----------|---------------|
| Scale granularity | 1 scalar per (t, h) | 1 scale per 32 elements (`num_blocks` per head) |
| Fold feasible? | Yes -- single scalar is trivially folded into weights | No -- `num_blocks` scales cannot be collapsed into a single weight |
| Extra tensor | None | `q_scale` output tensor needed |
| Downstream kernel shape | Uses pre-scaled weights directly | Must load + apply per-block scales during dequant |

## 3. RoPE Algorithm (GPT-J Interleaved, Forward)

Applied to the **last** `rope_dim` elements of each head. The leading `nope_dim = head_dim - rope_dim` elements pass through unchanged.

### Forward Rotation

```
For each i in [0, rope_dim/2):
    even_idx = nope_dim + 2*i        # even index in rope region
    odd_idx  = nope_dim + 2*i + 1    # odd index in rope region

    x_even = q[even_idx]
    x_odd  = q[odd_idx]

    r_even = x_even * cos[i] - x_odd * sin[i]    # forward rotation (even)
    r_odd  = x_odd * cos[i] + x_even * sin[i]    # forward rotation (odd)
```

Contrast with [[FusedInvRopeFP8Quant]] where the inverse rotation uses `+ partner*sin` on the even branch.

### BFloat16 Roundtrip for Numerical Parity

Both the nope and rope halves follow this sequence before absmax:

```
fp32 -> bf16 -> fp32
```

This is done explicitly in the kernel (L123-124 for FP8 kernel, L259-260 for MXFP4 kernel). The comment notes: *"Same pattern as the K-side compressor kernel"* ([[DeepseekCompressor]]). This ensures numerical parity between the fused kernel and the unfused reference (which would have naturally rounded through a bf16 write-back between RoPE and quant).

In the FP8 kernel:
```python
r_even = r_even.to(tl.bfloat16).to(tl.float32)
r_odd  = r_odd.to(tl.bfloat16).to(tl.float32)
```

### Cos/Sin Lookup

```python
cos = cos_sin_cache[pos, 0:HALF_ROT_DIM]
sin = cos_sin_cache[pos, HALF_ROT_DIM:ROT_DIM]
```

The cache stores cos and sin contiguously: `[cos_0..cos_{H-1}, sin_0..sin_{H-1}]`.

## 4. FP8 Quant Path

```
Scale computation:
    amax = max(|nope_values|, |rotated_even|, |rotated_odd|)
    q_scale = round_nearest(amax, tol=1e-4) / 448.0    # div_rn
    q_scale = 2^ceil(log2(q_scale))                    # round up to pow2

Quantization:
    q_fp8[t, h, d] = clamp(q_bf16 / q_scale, -448, 447)  cast to float8_e4m3fn
```

- The entire head shares a single scalar scale.
- `448.0` = max representable value in `float8_e4m3fn`.
- Scale is rounded UP to a power of 2 (ensures no overflow).

## 5. MXFP4 Quant Path

### MXFP4 Format

| Component | Bits | Description |
|-----------|------|-------------|
| Element | E2M1 (4-bit) | 2 exponent bits, 1 mantissa bit, sign. Range: [-6, +6]. |
| Block size | 32 elements | 1 scale per block. |
| Packing | 2 nibbles/byte | Even-index value in low nibble, odd-index in high nibble. |
| Scale | ue8m0 (8-bit) | Unsigned exponent-only: `scale = 2^(ue8m0 - 127)`. |

### Block Quantization (`_quantize_mxfp4_pair`)

The kernel processes 32-element blocks in groups of 16 pairs:

```python
# Each invocation handles 16 even + 16 odd values (= 32 elements = 1 MXFP4 block)
amax = max(|x_lo|, |x_hi|)
amax = max(amax, 6.0 * 2^-126)        # floor from DeepSeek reference kernel

log2_ratio = ceil(log2(amax / 6.0))    # 6.0 = max of E2M1
log2_ratio = clamp(log2_ratio, -127, 127)
scale = 2^log2_ratio
ue8m0 = uint8(log2_ratio + 127)        # bias = 127

# FP4 quant via inline PTX:
#   cvt.rn.satfinite.e2m1x2.f32 tmp, x_hi, x_lo;
packed = inline_asm_elementwise(...)  # returns uint8
```

The PTX intrinsic `cvt.rn.satfinite.e2m1x2.f32` converts two f32 values to a pair of E2M1 values packed into one byte.

### MXFP4 RoPE Block Loop

In the MXFP4 path, the RoPE region is processed in `NUM_ROPE_BLOCKS` iterations, each handling `MXFP4_BLOCK_SIZE = 32` elements:

```python
for b in static_range(NUM_ROPE_BLOCKS):
    # Load 16 pairs of cos/sin for this block
    pair_off = b * 16 + tid  # tid = [0, 16)

    # Apply GPT-J RoPE to the block's 16 pairs
    r_even = x_even * cos - x_odd * sin
    r_odd  = x_odd * cos + x_even * sin

    # bf16 roundtrip (numerical parity)
    r_even = r_even.to(bf16).to(fp32)
    r_odd  = r_odd.to(bf16).to(fp32)

    # Quantize pair and store packed result + ue8m0 scale
    packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
    store(out_base + rope_byte_off + half_off, packed)
    store(scale_base + NUM_NOPE_BLOCKS + b, ue8m0)
```

## 6. Code Structure: CuteDSL vs Triton Fallback

The common entry point `fused_indexer_q_rope_quant` dispatches based on hardware availability:

```
fused_indexer_q_rope_quant()
    |
    ├── has_cutedsl() == True (SM100 CUDA)
    │   └── vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py
    │       ├── fused_indexer_q_rope_quant_mxfp4_cutedsl()
    │       └── fused_indexer_q_rope_quant_fp8_cutedsl()
    │
    └── has_cutedsl() == False (SM90 or non-NVIDIA)
        └── vllm/models/deepseek_v4/common/ops/fused_indexer_q.py (THIS FILE)
            ├── _fused_indexer_q_rope_quant_kernel    (Triton, FP8 path)
            └── _fused_indexer_q_rope_mxfp4_kernel     (Triton, MXFP4 path)
```

### Triton Kernel Details

Both Triton kernels use a **1D launch grid** of `(num_tokens, num_index_q_heads)` -- one program per (token, head) pair with **1 warp each**.

| Aspect | FP8 Kernel | MXFP4 Kernel |
|--------|------------|--------------|
| Triton function | `_fused_indexer_q_rope_quant_kernel` | `_fused_indexer_q_rope_mxfp4_kernel` |
| Output values | `index_q_fp8` (float8_e4m3fn) | `index_q_packed` (uint8) + `index_q_scale` (uint8 ue8m0) |
| Scale fold | Yes, into `weights_out` | No, separate `q_scale` output |
| Input layout | Direct `index_q` (no transpose) | Same |
| NoPE blocks | Single `tl.load` | Loop over `NUM_NOPE_BLOCKS` |
| RoPE blocks | Single load + rotate | Loop over `NUM_ROPE_BLOCKS` |
| Inline PTX | None | `cvt.rn.satfinite.e2m1x2.f32` for FP4 packing |

### Return Format

```python
# FP8 path
return index_q_fp8, weights_out
# index_q_fp8: (T, H, head_dim) float8_e4m3fn

# MXFP4 path
return (index_q_packed, index_q_scale.view(int32).squeeze(-1)), weights_out
# index_q_packed: (T, H, head_dim//2) uint8
# index_q_scale:  (T, H) int32 (after squeeze; 4 ue8m0 bytes packed per int32)
```

The MXFP4 scale is reinterpreted as int32 and squeezed from `(T, H, 1)` to `(T, H)` to match DeepGEMM's expected `q_sf` rank (2-D for prefill, 3-D for decode after reshape).

## Cross-References

- [[FusedInvRopeFP8Quant]] -- counterpart for O-side inverse RoPE + quant.
- [[DeepseekV4Indexer]] -- the indexer pipeline that consumes this kernel's output.
- [[DeepseekV4MultiHeadLatentAttentionWrapper]] -- the MLA wrapper that calls this kernel on the auxiliary attention stream.
- [[DeepseekCompressor]] -- another fused kernel using the same bf16-roundtrip pattern for numerical parity.
- `fused_indexer_q_cutedsl.py` -- CuTeDSL implementation of the same operation (SM100 only).

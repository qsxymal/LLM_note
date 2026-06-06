# FusedInvRopeFP8Quant -- Fused Inverse RoPE + FP8 Quantization for Attention Output

**File:** `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`

Fused kernel that applies **inverse GPT-J interleaved RoPE** to the attention output `o` (bf16) followed by **block-scaled FP8 quantization**. The result feeds directly into `fp8_einsum` for the V-contribution term in MLA.

**Relationship with [[FusedIndexerQ]]:** These two kernels handle the RoPE + quantization fusion for opposite ends of the attention pipeline. `FusedInvRopeFP8Quant` processes the **attention output** (post-attention, on the O side), applying *inverse* RoPE to convert back from rotary to absolute position encoding. `FusedIndexerQ` processes the **Q projection** (pre-attention, on the indexer side), applying *forward* RoPE before quantizing. Both use GPT-J interleaved RoPE and block-scaled quantization, but differ in their scale management strategy (block-scaled vs. per-token scalar) and output transposition.

## 1. Why Fused

Without fusion, the attention output would need:
1. Read `o` from HBM, apply inverse RoPE, write back to HBM
2. Read `o` from HBM again, compute absmax per block, quantize to FP8, write

Fusion eliminates the intermediate HBM roundtrip of `o`, keeping all computation and the temporary rotated values in registers.

## 2. Inputs and Outputs

### Inputs

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `o` | `(T, num_heads, head_dim)` | bf16 | Attention output before inverse RoPE. `head_dim = nope_dim + rope_dim = 448 + 64 = 512`. |
| `positions` | `(T,)` | int64 | Token positions used to index `cos_sin_cache`. |
| `cos_sin_cache` | `(max_pos, rope_dim)` | float32 | Precomputed cos/sin values. `rope_dim = 64`. |

### Outputs

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `o_fp8` | `(n_groups, T, d)` | float8_e4m3fn | Transposed to group-major for `fp8_einsum`. `d = heads_per_group * head_dim`. |
| `o_scale` | Varies by SM architecture | fp32 or int32 | See scale format below. |

### Scale Format

| Platform | dtype | Shape (per group) | Format |
|----------|-------|-------------------|--------|
| SM90 (`TMA_ALIGNED_SCALES=False`) | fp32 | `(T, num_scale_blocks)` | Each block's fp32 scale factor. |
| SM100 (`TMA_ALIGNED_SCALES=True`) | int32 | `(T, packed_sf_k)` | 4 UE8M0 exponents packed per int32. `packed_sf_k = ceil(num_scale_blocks / 4)`. |

The scale tensor is **pre-transformed** into the layout expected by `fp8_einsum`, so the downstream kernel can skip `transform_sf_into_required_layout`.

## 3. Kernel Launch Configuration

```
Grid: (tma_aligned_T, n_groups * heads_per_group)
```

- One Triton program per `(token, head)` pair.
- `tma_aligned_T` = `get_tma_aligned_size(num_tokens, 4)` -- padding to TMA alignment boundary.
- **1 warp / 1 stage** per program: each program processes a single head's `head_dim=512` values.

## 4. Inverse RoPE (GPT-J Interleaved)

The kernel applies inverse RoPE to the **last `rope_dim=64` elements** of each head. The algorithm reconstructs the *un-rotated* representation from the rotated one.

### Quant Block Boundary

`QUANT_GROUP_SIZE = 128`, `CHUNKS_PER_HEAD = 512 / 128 = 4`. The last quant block (block index 3, elements `[384, 512)`) spans the boundary between nope (elements `[0, 448)`) and rope (elements `[448, 512)`):

```
|                head_dim = 512                          |
|  block 0 (128) | block 1 (128) | block 2 (128) | block 3 (128)           |
|  nope=448                       |<--- nope ---->|<--- rope=64 ---->|
                                                  ^  rope_abs_start = 384 + 64
```

`rope_abs_start = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START = 3*128 + 64 = 448`.

Where `ROPE_START = nope_dim % quant_group_size = 448 % 128 = 64`.

### Rotation Algorithm

```python
# Load partner element using XOR with 1 (GPT-J interleaved pairing)
x_partner = o[offsets ^ 1]    # even <-> odd partner

# Load cos/sin from cache for this position
cos_v = cos_sin_cache[pos, cs_idx]
sin_v = cos_sin_cache[pos, HALF_ROPE + cs_idx]

# Inverse rotation (note: x_add = o*cos + partner*sin instead of o*cos - partner*sin)
x_add = x * cos_v + x_partner * sin_v
x_sub = x * cos_v - x_partner * sin_v

# Interleave: even indices get x_add, odd indices get x_sub
rotated = x_add if is_even else x_sub
```

The key difference from forward RoPE is the sign pattern: `+ partner*sin` (inverse) vs `- partner*sin` (forward). See [[FusedIndexerQ]] for the forward version.

## 5. Block-Scaled FP8 Quantization

### Scale Computation

1. Reshape absolute values into `(CHUNKS_PER_HEAD, QUANT_GROUP_SIZE)` = `(4, 128)`.
2. Compute `block_absmax = max(|x|)` per block, clamped to `eps=1e-10`.
3. `scale_raw = block_absmax / fp8_max` where `fp8_max = 448.0` (max of float8_e4m3fn).
4. **Round scale up to next power of 2**: `scale = 2^ceil(log2(scale_raw))`.
5. Quantize: `x_quant = clamp(x / scale, -fp8_max, fp8_max)` cast to float8_e4m3fn.

### Output Transposition

The output tensor `o_fp8` has shape `(n_groups, T, d)` with strides `(d, T*d, 1)`. This is the group-major layout expected by `fp8_einsum`.

## 6. Custom Op Registration

```python
direct_register_custom_op(
    op_name="fused_inv_rope_fp8_quant_kernel",
    op_func=_fused_inv_rope_fp8_quant_kernel_impl,
    fake_impl=_fused_inv_rope_fp8_quant_kernel_fake,
)
```

The operation is registered as a PyTorch custom op so that TorchInductor sees an opaque boundary (workaround for [vllm issue #41106](https://github.com/vllm-project/vllm/issues/41106)). The `fake_impl` returns empty tensors of the correct shape for inductor's tracing pass without executing the Triton kernel.

The `launch_pdl=False` parameter is passed on CUDA (not ROCm/XPU) to disable persistent distributed launch.

## 7. Numerical Details

- Input `o` is loaded as fp32 (via `.to(tl.float32)`) for the rotation computation to maintain precision.
- Scale rounding `ceil(log2(...))` ensures the quantized values always fit in the FP8 range without overflow.
- `eps = 1e-10` prevents division by zero for degenerate blocks.
- The `(offsets ^ 1)` trick pairs each element with its partner by flipping the lowest bit, which works for the GPT-J interleaved layout where even/odd pairs are rotated.

## Cross-References

- [[FusedIndexerQ]] -- counterpart for Q-side forward RoPE + quant.
- [[DeepseekV4MLAAttention]] -- the attention mechanism that produces the `o` tensor.
- [[SparseAttnCompress]] -- the compressor on the KV side uses similar FP8 quantization patterns.
- [[DequantGatherKCache]] -- another fused quant + memory access kernel in the same family.

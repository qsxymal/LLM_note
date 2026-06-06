# DeepseekV4 KVCache Ops

## FusedCompressQuantCache

**Source:** `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`

Fused Triton kernel that chains together: **compressor -> RMSNorm -> RoPE -> quantize -> insert into KV cache** in a single kernel launch, avoiding redundant memory round-trips through global memory.

### Selector Function

`compress_norm_rope_store_triton()` (line 31) dispatches to one of three kernels based on `head_dim` and `use_fp4_cache`:

| head_dim | use_fp4_cache | Kernel | num_warps |
|----------|---------------|--------|-----------|
| 512 | N/A | `_fused_kv_compress_norm_rope_insert_sparse_attn` (line 113) | 4 |
| 128 | False | `_fused_kv_compress_norm_rope_insert_indexer_attn` (line 303) | 1 |
| 128 | True | `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` (line 480) | 1 |

All three share an identical launch signature; the selector passes `**pdl_kwargs` for persistent-dynamic-launch metadata (line 105).

### Program Model: One Program Per Token

Each kernel launches `num_actual` programs (one per token). The first action in every kernel is an early-exit guard (e.g. line 163):

```
if (position + 1) % COMPRESS_RATIO != 0:
    return
```

Non-boundary positions (where the compression condition is not met) exit immediately, so only 1-of-`COMPRESS_RATIO` tokens performs actual work.

A second guard checks `slot_mapping[token_idx] >= 0` (line 159) to skip padding tokens.

### Compressor

The compression step gathers state cache entries, computes a softmax over the sliding window, and produces a weighted sum. All three kernels use identical logic:

- **Window start** (line 169):
  ```
  start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
  ```
  Gathers `(1+OVERLAP)*COMPRESS_RATIO` tokens total (past sliding window + current token).

- **KV slot lookup** (lines 174-193): For each gathered position, the block_table translates `(req_idx, position // block_size)` into a physical block number; then `(block_number, block_offset, head_offset)` indexes into the state cache.

- **Score loading and softmax** (lines 198-203): Score values are read from the state cache at offset `STATE_WIDTH` (half of the per-entry width). Softmax is applied along `dim=0` (across gathered tokens), producing `[TRITON_BLOCK_SIZE]` weights.

- **Weighted sum** (line 211):
  ```
  compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32
  ```

### RMSNorm

Applied in fp32 throughout (line 213-217):

```python
variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
rrms = tl.rsqrt(variance + rms_norm_eps)
normed = compressed_kv * rrms * rms_w
```

### RoPE

All three kernels implement **GPT-J** (interleaved) rotary position embedding. The MXFP4 variant (line 629) skips the final `tl.interleave` because the downstream MXFP4 quantizer naturally operates on even/odd halves.

Common approach (example from sparse attn, lines 275-296):
1. `tl.reshape(normed, (NUM_PAIRS, 2))` -- pair up even/odd elements.
2. `tl.split(...)` -- split into `even`, `odd` vectors.
3. `cs_idx = pair_idx - NOPE_PAIRS` -- only the rope half of the dimension uses cos/sin.
4. `cos`, `sin` loaded from `cos_sin_cache` at `compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO`.
5. `new_even = even * cos_v - odd * sin_v`, `new_odd = odd * cos_v + even * sin_v`.
6. `tl.interleave(new_even, new_odd)` -- restore original interleaved layout.

The bf16 roundtrip (line 242, line 457, line 631) `normed.to(tl.bfloat16).to(tl.float32)` exists for numeric parity with the reference kernel / Q-side kernels.

### Quantize and Store into Paged KV Cache

**Cache block layout** (same for all three kernels):

```
[0 : bs*token_stride)           token data (fp8 bytes or packed nibbles)
[bs*token_stride : +bs*scale_dim]  scales (uint8 UE8M0 exponents or float32)
```

**KV cache pointer computation** (lines 220-232, 412-424, 591-603):
- `kv_slot_mapping[token_idx]` gives the flat slot index.
- `kv_block_idx = slot // kv_cache_block_size`, `kv_pos_in_block = slot % kv_cache_block_size`.
- `cache_block_ptr = k_cache_ptr + kv_block_idx * KV_BLOCK_STRIDE`.
- `fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE`.
- `scale_ptr = cache_block_ptr + bs*TOKEN_STRIDE + kv_pos_in_block * SCALE_DIM`.

#### Kernel 1: Sparse Attn (head=512)

| Parameter | Value | Line Defined |
|-----------|-------|--------------|
| HEAD_SIZE | 512 | line 93 |
| NOPE_HEAD_DIM | 448 (= HEAD_SIZE - ROPE_HEAD_DIM) | line 234 |
| ROPE_HEAD_DIM | 64 | line 141 |
| HALF_ROPE | 32 | line 235 |
| TOKEN_STRIDE | 576 (= 448 fp8 + 128 bf16) | line 144 |
| SCALE_DIM | 8 (7 real + 1 pad) | line 145 |
| QUANT_BLOCK | 64 | line 143 |
| N_QUANT_BLOCKS | 8 (= TRITON_BLOCK_SIZE // QUANT_BLOCK) | line 238 |
| N_NOPE_BLOCKS | 7 (= NOPE_HEAD_DIM // QUANT_BLOCK) | line 239 |

UE8M0 quantization (lines 242-269):
1. `normed.to(tl.bfloat16).to(tl.float32)` -- bf16 roundtrip for parity.
2. `tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))` -- tile into register-level blocks.
3. Per-block absmax -> `scale = 2^ceil(log2(absmax / 448.0))`.
4. `x_scaled = x / scale`, clamped to `[-448, 448]`, `to(tl.float8e4nv)`, bitcast to `uint8`.
5. Store fp8 data: `tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)`.
6. Store scales: exponent + 127 (UE8M0 encoding), last slot `[7]` zero-padded.
7. Rope portion stored as bf16 at `fp8_ptr + NOPE_HEAD_DIM` via `tl.pointer_type(tl.bfloat16)`.

#### Kernel 2: Indexer FP8 (head=128)

| Parameter | Value | Line Defined |
|-----------|-------|--------------|
| HEAD_SIZE | 128 | line 326 |
| TOKEN_STRIDE | 128 | line 335 |
| SCALE_DIM | 4 (= 1 float32) | line 336 |
| QUANT_BLOCK | 128 | line 334 |

Because `TRITON_BLOCK_SIZE == QUANT_BLOCK`, a single flat `tl.max` reduction is used (line 458), skipping the `[N_QUANT_BLOCKS, QUANT_BLOCK]` reshape. The scale is stored as a single `float32` value (line 472-473), not UE8M0.

#### Kernel 3: Indexer MXFP4 (head=128)

| Parameter | Value | Line Defined |
|-----------|-------|--------------|
| HEAD_SIZE | 128 | line 503 |
| TOKEN_STRIDE | 64 (= HEAD_SIZE // 2, packed nibbles) | line 511 |
| SCALE_DIM | 4 (= HEAD_SIZE // QUANT_BLOCK, ue8m0 bytes) | line 512 |
| QUANT_BLOCK | 32 (MXFP4 block size) | line 510 |
| HALF_BLOCK | 16 (= QUANT_BLOCK // 2) | line 638 |
| N_QUANT_BLOCKS | 4 (= HEAD_SIZE // QUANT_BLOCK) | line 637 |

MXFP4 format:
- E2M1 4-bit values packed two per byte (low nibble first, then high).
- Per-32-element block scale = `2^ceil(log2(amax / 6.0))`, stored as ue8m0 byte (= exponent + 127).
- Max representable magnitude = 6.0.

RoPE output kept as even/odd halves (no `tl.interleave`), tiled into `(N_QUANT_BLOCKS, HALF_BLOCK)` for even and odd separately (lines 644-645). Absmax computed pairwise (line 647-650). Packed via `_fp32x2_to_fp4x2` (line 660, defined in `fused_indexer_q.py` line 29), which uses `cvt.rn.satfinite.e2m1x2.f32` inline PTX.

### Cross References

- [[#quantize_and_insert_k_cache]] uses the same UE8M0 scheme but as a standalone kernel (without compression). The compute `exponent = ceil(log2(block_max / FP8_MAX))` is identical (cache_utils.py line 108 vs fused_compress_quant_cache.py line 249).
- [[#dequantize_and_gather_k_cache]] is the inverse: loads fp8 + UE8M0 scale, dequantizes back to bf16 by `x_fp8 * 2^(stored - 127)`.
- [[#combine_topk_swa_indices]] consumes the compressed tokens produced by this kernel.

---

## CacheUtils

**Source:** `vllm/models/deepseek_v4/common/ops/cache_utils.py`

Four exported Triton kernels for paged K-cache management and sparse-attention index preparation. All operate on the same KV-cache block layout defined in [[#FusedCompressQuantCache]].

---

### quantize_and_insert_k_cache

**Python wrapper:** line 142 | **Triton kernel:** `quantize_and_insert_k_kernel` line 24

**Purpose:** Quantize a bf16 K tensor into UE8M0 FP8 and write into the paged KV cache. Standalone version of the quantize + store stage from [[#FusedCompressQuantCache]], used when compression is not fused (or for the offline/fallback path).

**Tensor shapes:**

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `k` | `(num_tokens, 512)` | bf16 |
| `k_cache` | `(num_blocks, block_bytes)` | uint8 |
| `slot_mapping` | `(num_tokens,)` | int64 |

**K Cache Block Layout (block_size=64 tokens):**

```
Offset range                          Contents
[0, 64*576)                           Token data: each token = 448 fp8 + 128 bf16
[64*576, 64*576 + 64*8)               Scales: each token = 8 uint8 (7 real + 1 pad)
[64*576 + 64*8, block_stride)         Padding
```

**Constants** (lines 170-175):

| Constant | Value | Meaning |
|----------|-------|---------|
| TOKEN_FP8_DIM | 448 | nope dimensions quantized to FP8 |
| TOKEN_BF16_DIM | 64 | rope dimensions stored as bf16 |
| TOKEN_SCALE_DIM | 8 | UE8M0 scale bytes per token (7 real + 1 pad) |
| QUANT_BLOCK_SIZE | 64 | elements per quantization block |
| FP8_MAX | 448.0 | max representable value for fp8e4nv |
| TOKEN_DATA_SIZE | 576 (= 448 + 64*2) | total data bytes per token |
| n_quant_blocks | 8 | number of 64-element quant blocks |

**Algorithm** (lines 87-139):

```
for each of 8 quant blocks (qblock_idx in tl.static_range(8)):
    load bf16[offsets] for this block
    block_max = max(abs(x))
    block_max = max(block_max, 1e-4)           # match CUDA: fmaxf(amax, 1e-4)
    exponent = ceil(log2(block_max / FP8_MAX)) # UE8M0: scale is power-of-2
    scale = 2^exponent
    x_scaled = x / scale
    x_clamped = clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(float8e4nv)
    x_uint8 = bitcast(x_fp8, uint8)            # store as raw bytes
    store(x_uint8, fp8_ptr + offsets)
    store(uint8(exponent + 127), scale_ptr + qblock_idx)  # UE8M0 encoding

store 0 at scale_ptr[7]                         # padding scale

for each of 4 bf16 chunks (16 elements each):
    load bf16[fp8_dim + chunk_offsets]          # rope portion
    store bf16 at bf16_out_ptr + chunk_offsets  # stored directly
```

Lines 123-124 show the UE8M0 encoding: `stored = exponent + 127`, clamped to `[0, 255]`. On dequant: `scale = 2^(stored - 127)`.

**Key design notes:**
- Uses `block_idx.to(tl.int64) * block_stride` (line 71) because `block_stride ~37K * 57K+ blocks` can exceed `2^31`.
- When using DP, `slot_mapping.shape[0]` may be less than `k.shape[0]` due to padding; the token count always comes from `slot_mapping` (line 167).

---

### dequantize_and_gather_k_cache

**Python wrapper:** line 353 | **Triton kernel:** `_dequantize_and_gather_k_kernel` line 198 | **CuteDSL fallback:** line 369

**Purpose:** Inverse of `quantize_and_insert_k_cache`. Gathers FP8 K from the paged cache, dequantizes to bf16, for use in sparse/SWA prefill attention.

**Tensor shapes:**

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `out` | `(num_reqs, max_num_tokens, 512)` | bf16 (output) |
| `k_cache` | `(num_blocks, block_bytes)` | uint8 |
| `seq_lens` | `(num_reqs,)` | int32 |
| `gather_lens` | `(num_reqs,)` or `None` | int32 |
| `block_table` | `(num_reqs, max_blocks_per_seq)` | int32 |

**Launch config:** `(num_reqs, 128)` grid (line 330) -- 128 workers cooperatively process each request's tokens.

**Algorithm** (lines 232-304):

```
for each token assigned to this worker (stride = num_workers):
    pos = (seq_len - gather_len) + i         # position in sequence
    block_in_seq = pos // cache_block_size
    physical_block = block_table[batch_idx, block_in_seq]
    
    cache_block_ptr = k_cache + physical_block * block_stride
    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
    token_scale_ptr = cache_block_ptr + bs*token_data_size + pos_in_block * scale_dim
    
    for each of 7 quant blocks:
        load uint8(fp8 raw bytes)
        x_fp8 = float8e4nv(bitcast uint8)
        x_float = fp32(x_fp8)
        encoded_scale = load(scale_ptr + qblock_idx)
        exponent = float32(encoded_scale) - 127.0
        scale = 2^exponent
        
        x_dequant = x_float * scale          # bf16
        store(x_dequant, out[token_idx, offset])
    
    # bf16 portion: direct copy
    load bf16 from cache at offset fp8_dim, store to output at same offset
```

**Key notes:**
- `gather_lens` allows gathering only the trailing `N` tokens (for SWA). When `None`, gathers all `seq_len` tokens (line 226-229).
- `physical_block_idx.to(tl.int64) * block_stride` (line 246) prevents int32 overflow, same reasoning as the insert path.
- `n_quant_blocks = 7` (no padding block) vs `8` in the insert kernel (line 349 vs line 193).
- CuteDSL fallback (lines 367-380): when `has_cutedsl()` is true, delegates to `dequantize_and_gather_k_cache_cutedsl` (NVIDIA-optimized CUDA implementation).

---

### compute_global_topk_indices_and_lens

**Python wrapper:** line 383 | **Triton kernel:** `_compute_global_topk_indices_and_lens_kernel` line 418

**Purpose:** Maps local topk indices (indices within a request's logical sequence) to global KV cache slot IDs (physical positions in the paged cache). Fuses three operations into one kernel.

**Tensor shapes:**

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `topk_indices` | `(T, topk)` | int32 |
| `token_to_req_indices` | `(T,)` | int32 |
| `block_table` | `(num_reqs, max_blocks_per_seq)` | int32 |
| `is_valid_token` | `(T,)` | int32 |
| `global_topk_indices` | `(T, topk)` | int32 (output) |
| `topk_lens` | `(T,)` | int32 (output) |

**Algorithm** (lines 441-465):

```
for each token (1 program):
    for each block of up to 1024 indices:
        local_idx = load(topk_indices[token, offset])
        is_valid = (local_idx >= 0)
        
        block_number = block_table[req_idx, local_idx // block_size]
        block_offset = local_idx % block_size
        slot_id = block_number * block_size + block_offset
        slot_id = where(is_valid, slot_id, -1)
        store(global_topk_indices[token, offset], slot_id)
        count += sum(is_valid)
    
    store(topk_lens[token], is_valid_token ? count : 0)
```

**Design notes:**
- The `block_size` parameter matches the paged KV cache block size (typically 64).
- Valid-entry counting (line 436, 462): only entries with `local_idx >= 0` contribute to `topk_lens`.
- Padding tokens are masked to length 0 (line 465), ensuring downstream scatter operations are no-ops.
- Output `global_topk_indices` uses `-1` as sentinel for invalid slots.

---

### combine_topk_swa_indices

**Python wrapper:** line 476 | **Triton kernel:** `_combine_topk_swa_indices_kernel` line 525

**Purpose:** Concatenates topk (compressed) indices with SWA (sliding window) indices into a single sorted index array for FlashMLA sparse prefill.

**Tensor shapes:**

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `topk_indices` | `(T, topk)` | int32 (local indices) |
| `query_start_loc` | `(num_reqs+1,)` | int32 |
| `seq_lens` | `(num_reqs,)` | int32 |
| `gather_lens` | `(num_reqs,)` | int32 |
| `combined_indices` | `(T, P)` | int32 (output, P = padded) |
| `combined_lens` | `(T,)` | int32 (output) |

Where `P = ceil((topk + window_size) / 128) * 128` (aligned to `_SPARSE_PREFILL_TOPK_ALIGNMENT`, line 473).

**Launch config:** `(num_reqs, 128)` grid (line 505) -- workers iterate over tokens in a request.

**Algorithm** (lines 559-594):

```
for each request (1 program per request, 128 workers):
    base = query_start_loc[0]
    query_start = query_start_loc[batch_idx] - base
    query_end = query_start_loc[batch_idx+1] - base
    start_pos = seq_len - (query_end - query_start)
    gather_start = seq_len - gather_len

    for each token assigned to this worker:
        pos = start_pos + token_idx_in_query
        
        topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = min(pos + 1, WINDOW_SIZE)
        
        # Store topk indices (offset by M*batch_idx for global indexing)
        store(combined_indices[topk_offset], topk_indices + M*batch_idx)
        
        # Store SWA indices
        swa_start = N + pos - swa_len + 1 - gather_start
        store(combined_indices[topk_len + swa_offset], M*batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start)
        
        store(combined_lens[token_idx], topk_len + swa_len)
```

**Key constants and design:**

- `_SPARSE_PREFILL_TOPK_ALIGNMENT = 128` (line 473): FlashMLA sparse prefill requires `topk % 128 == 0` (see comment at line 468-472). Combined output is padded to this alignment with `-1` sentinel values.
- `PADDED_TOP_K` (line 519) = `next_power_of_2(topk)`, used to load topk indices in aligned chunks.
- `TOP_K=0` (line 516) is used for SWA-only layers to zero out the topk portion.
- `M * batch_idx` and `N` are global offset constants (defined by the caller, passed as tensor values) that shift indices into the correct region of the global KV cache buffer.
- `gather_start` offset (line 557): the SWA portion of the gathered buffer starts from `seq_len - gather_len`, not position 0, so this offset adjusts the SWA index computation.

### Cross References

- [[#FusedCompressQuantCache]] defines the KV cache block layout that all CacheUtils kernels operate on.
- `quantize_and_insert_k_cache` shares the UE8M0 algorithm with the fused kernel's quant step. The `exponent = ceil(log2(block_max / FP8_MAX))` computation is byte-identical (cache_utils.py:108 vs fused_compress_quant_cache.py:249).
- `dequantize_and_gather_k_cache` is the inverse of the quant step in [[#FusedCompressQuantCache]].
- `compute_global_topk_indices_and_lens` produces the global indices consumed by `combine_topk_swa_indices`.
- `combine_topk_swa_indices` feeds into FlashMLA sparse prefill.

### Shared Design Elements

- **Overflow protection:** Both `quantize_and_insert_k_cache` (line 71) and `dequantize_and_gather_k_cache` (line 246) use `tl.int64` for `block_idx * block_stride` multiplication because the product can exceed 2^31 when the KV cache contains 57K+ blocks with a stride of ~37K bytes.
- **Worker parallelism:** `dequantize_and_gather_k_cache` and `combine_topk_swa_indices` both use 128 workers per batch for fine-grained parallelism.
- **UE8M0 format:** Scale = 2^(exponent), stored as uint8(exponent + 127). Encoded as `ceil(log2(block_max / FP8_MAX)) + 127`; decoded as `2^(stored - 127)`.

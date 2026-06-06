# ROCm Attention Backend for DeepSeek V4

## Overview

The file `vllm/models/deepseek_v4/amd/rocm.py` provides the **AMD ROCm-specific** implementation of DeepSeek V4's sparse MLA (Multi-head Latent Attention) backend. It is the ROCm counterpart of [[DeepseekV4_KVCache_Ops#NVIDIA Backend|the NVIDIA flashmla.py backend]].

ROCm cannot use NVIDIA's `flash_mla_with_kvcache` or `flash_mla_sparse_fwd` CUDA kernels (which depend on cuDNN/CUTLASS for Hopper GPUs). Instead, it delegates to Triton-based sparse attention kernels in the `rocm_aiter_mla_sparse` module. The defining architectural difference is that ROCm uses a **ragged (COO-like) format** for sparse indices, whereas NVIDIA uses a **dense padded format**.

---

## Class Hierarchy

```
AttentionBackend                          (vllm.v1.attention.backend)
  └── FlashMLASparseBackend               (flashmla_sparse.py)
        └── DeepseekV4FlashMLASparseBackend  (nvidia/flashmla.py)
              └── DeepseekV4ROCMAiterMLASparseBackend   [L576-587]

SparseMLAAttentionImpl                    (vllm.v1.attention.backend)
  └── DeepseekV4SparseMLAAttentionImpl    (nvidia/flashmla.py, abstract)
        └── DeepseekV4FlashMLASparseImpl  (nvidia/flashmla.py)
              └── DeepseekV4ROCMAiterMLASparseImpl       [L590-856]

FlashMLASparseMetadata                    (flashmla_sparse.py)
  └── DeepseekV4ROCMAiterMLASparseMetadata               [L451-456]

FlashMLASparseMetadataBuilder             (flashmla_sparse.py)
  └── DeepseekV4ROCMAiterMLASparseMetadataBuilder        [L465-519]

DeepseekSparseSWAMetadata                 (sparse_swa.py)
  └── DeepseekV4ROCMAiterSparseSWAMetadata               [L459-462]

DeepseekSparseSWAMetadataBuilder          (sparse_swa.py)
  └── DeepseekV4ROCMAiterSparseSWAMetadataBuilder        [L522-573]
```

The ROCm classes override the NVIDIA versions to inject ragged-format metadata and wire in ROCm-specific attention kernels. They do not add new layers to the inheritance hierarchy but rather slot in as alternative backends selected at runtime based on `current_platform.is_rocm()`.

---

## Dense vs. Ragged Format

This is the central design distinction between the NVIDIA and ROCm backends.

### Dense Format (NVIDIA)

Used by `flash_mla_with_kvcache` and `flash_mla_sparse_fwd`:

- Every token has a **fixed-width** 2D tensor of indices, padded with `-1` sentinels.
- Each row has shape `(1, TOP_K)` or `(1, combined_topk)`.
- The kernel uses `topk_length` values to know how many entries per row are valid.
- Padding guarantees aligned memory accesses for CUDA kernel launch configurations.

### Ragged Format (ROCm)

Required by ROCm's `rocm_sparse_attn_decode` / `rocm_sparse_attn_prefill`:

- A **flat 1D tensor** containing only the valid (non-padded) indices concatenated across all tokens.
- A companion **indptr** tensor of shape `(num_tokens + 1,)` delimits each token's range via CSR-style pointers: `indices[indptr[i]:indptr[i+1]]` are the valid entries for token `i`.
- Also carries a **lengths** tensor for compatibility with the NVIDIA code path.

The ragged format is analogous to CSR (Compressed Sparse Row) or COO encoding. It eliminates wasted memory from padding tokens. ROCm uses ragged because its Triton-based sparse kernels (originally designed for the DeepSeek V3 Aiter backend) operate on this representation and lack the padded-index kernel support that the NVIDIA CUDA kernels provide.

### Building Ragged from Dense

The helper `build_ragged_indices_from_dense` (defined in `rocm_aiter_mla_sparse.py`, called from both metadata builders) performs the conversion:

1. Clamp the lengths tensor to `[0, max_width]` to guard against out-of-range entries.
2. Compute the indptr via `cumsum(lengths)`.
3. Allocate a flat tensor of size `indptr[-1]`.
4. Launch `_pack_dense_prefix_to_ragged_kernel` to scatter valid entries into the flat buffer.

---

## Ragged Metadata Dataclasses

### `DeepseekV4ROCMAiterMLASparseMetadata` [L451-456]
Extends `FlashMLASparseMetadata` with two extra ragged fields for C128A decode:

```python
c128a_decode_topk_ragged_indices: torch.Tensor | None  # flat 1D
c128a_decode_topk_ragged_indptr: torch.Tensor | None    # (max_tokens+1,)
```

### `DeepseekV4ROCMAiterSparseSWAMetadata` [L459-462]
Extends `DeepseekSparseSWAMetadata` with ragged SWA indices for decode:

```python
decode_swa_ragged_indices: torch.Tensor | None  # flat 1D
decode_swa_ragged_indptr: torch.Tensor | None   # (max_tokens+1,)
```

---

## Metadata Building Pipeline

### C128A (Compress Ratio = 128)

The `DeepseekV4ROCMAiterMLASparseMetadataBuilder` [L465-519] follows this pipeline:

1. **Pre-allocate buffers** at constructor time (if `compress_ratio == 128`): ragged indices buffer of shape `(max_tokens * c128a_max_compressed,)` and indptr buffer of shape `(max_tokens + 1,)`. These are persistent for [[V1AttentionBackend#CUDA Graphs|CUDA graph compatibility]].

2. **`build()`**: Calls the parent `FlashMLASparseMetadataBuilder.build()` which produces the **dense** `c128a_global_decode_topk_indices` and `c128a_decode_topk_lens`.

3. **Ragged conversion**: Calls `build_ragged_indices_from_dense(dense_decode, decode_lens)` to produce ragged indices + indptr from the dense parent output.

4. **Copy to graph buffers**: Calls `_copy_ragged_to_graph_buffers()` to copy the ragged metadata into the pre-allocated persistent buffers.

5. Return a `DeepseekV4ROCMAiterMLASparseMetadata` with both dense (inherited) and ragged (new) fields populated.

### SWA (Sliding Window Attention)

The `DeepseekV4ROCMAiterSparseSWAMetadataBuilder` [L522-573] follows the same pattern:

1. **Pre-allocate** buffers of size `(max_tokens * window_size,)` for ragged indices and `(max_tokens + 1,)` for indptr.

2. **`build()`**: Calls parent `DeepseekSparseSWAMetadataBuilder.build()` to produce dense `decode_swa_indices` and `decode_swa_lens`.

3. **Ragged conversion**: Converts dense decode SWA indices to ragged.

4. **Copy to graph buffers**: Copies into pre-allocated persistent storage.

5. Return `DeepseekV4ROCMAiterSparseSWAMetadata` with ragged fields populated.

### C4A (Compress Ratio = 4)

C4A does **not** go through the metadata builder for its ragged indices. Instead, `_forward_decode()` [L700-712] calls `compute_global_topk_ragged_indices_and_indptr` **inline** during each decode step. This is because C4A topk indices differ per layer (they are filled by the Indexer at every step), so they cannot be pre-computed in the metadata builder.

The function `compute_global_topk_ragged_indices_and_indptr` [L236-277] is the AMD-specific replacement for the common `compute_global_topk_indices_and_lens` (which outputs dense). It:

1. Counts valid entries per token via `_compute_topk_lens_kernel`.
2. Builds indptr from lengths via `_build_indptr_from_lengths`.
3. Packs ragged indices from local -> global (block table lookup) via `_pack_global_topk_ragged_kernel`.

---

## CUDA Graph Compatibility Strategy

ROCm uses the same approach as NVIDIA for [[V1AttentionBackend#CUDA Graphs|CUDA graph compatibility]]: pre-allocate persistent buffers so kernel argument addresses are stable across graph replay.

The function `_copy_ragged_to_graph_buffers` [L427-448] handles the ragged-specific graph copy:

```python
def _copy_ragged_to_graph_buffers(
    ragged_indices, ragged_indptr,
    ragged_indices_buffer, ragged_indptr_buffer,
    num_rows, max_entries_per_row,
) -> tuple[Tensor, Tensor]:
```

- Copies indptr as a slice `[:num_rows + 1]` with `non_blocking=True`.
- Copies indices as a slice `[:nnz]` capped at `num_rows * max_entries_per_row`, also with `non_blocking=True`.
- Returns sliced views backed by the stable buffer storage so kernel argument addresses remain the same across graph capture and replay.
- The `non_blocking=True` flag enables H2D copy overlap with computation.

---

## Forward Pass Flow

### `forward_mqa()` [L600-675] -- Entry Point

Same dispatch structure as NVIDIA:

1. If `attn_metadata is None` (dummy warmup run): reserve workspace and zero output.
2. Split into prefill and decode segments.
3. Call `_forward_prefill()` on `q[num_decode_tokens:]`.
4. Call `_forward_decode()` on `q[:num_decode_tokens]`.

### `_forward_decode()` [L677-738]

```
forward_decode
  ├── C4A (compress_ratio==4)
  │     └── compute_global_topk_ragged_indices_and_indptr(layer.topk_indices_buffer)
  │           ├── _compute_topk_lens_kernel (count valid per token)
  │           ├── _build_indptr_from_lengths (cumsum -> indptr)
  │           └── _pack_global_topk_ragged_kernel (block-table lookup, flat output)
  │
  ├── C128A (compress_ratio==128)
  │     └── Use pre-built ragged from metadata:
  │         attn_metadata.c128a_decode_topk_ragged_indices
  │         attn_metadata.c128a_decode_topk_ragged_indptr
  │
  └── rocm_sparse_attn_decode(q, kv_cache, swa_k_cache, topk_*, swa_*, ...)
        ├── topk_ragged_indices + topk_ragged_indptr
        ├── swa_ragged_indices + swa_ragged_indptr  (from SWA metadata)
        └── Output: attention result
```

Key differences from NVIDIA decode:
- NVIDIA passes `topk_indices` (dense) into `flash_mla_with_kvcache` via `extra_indices_in_kvcache`.
- ROCm passes `topk_ragged_indices` + `topk_ragged_indptr` as separate arguments to `rocm_sparse_attn_decode`.
- ROCm also passes `swa_ragged_indices` + `swa_ragged_indptr` (the ragged conversion of the SWA decode indices), while NVIDIA passes dense `swa_indices` directly.

### `_forward_prefill()` [L740-856]

```
forward_prefill (chunked, chunk_size=4)
  │
  ├── for each chunk:
  │     ├── Dequantize + gather compressed K cache (if not SWA-only)
  │     │     └── dequantize_and_gather_k_cache (shared ops, same as NVIDIA)
  │     ├── Dequantize + gather SWA K cache
  │     │     └── dequantize_and_gather_k_cache (shared ops, same as NVIDIA)
  │     ├── Combine topk + SWA indices
  │     │     └── combine_topk_swa_indices (AMD-specific version, L118-165)
  │     └── rocm_sparse_attn_prefill(q, kv, combined_indices, combined_lens, ...)
  │
  └── Output: chunked attention result
```

Key differences from NVIDIA prefill:
- Both use the shared `dequantize_and_gather_k_cache` from common ops.
- Both call `combine_topk_swa_indices` but ROCm uses its **AMD-specific** version (not the shared one from `cache_utils.py`).
- NVIDIA calls `flash_mla_sparse_fwd` with `combined_indices.unsqueeze(1)`; ROCm calls `rocm_sparse_attn_prefill`.

---

## The Two `combine_topk_swa_indices` Variants

### Shared/Common Version (`common/ops/cache_utils.py`)

Used by both NVIDIA prefill and available for general use. Outputs **dense padded** combined indices.

- Padding aligns to `_SPARSE_PREFILL_TOPK_ALIGNMENT=128`.
- The kernel assumes all `topk_indices` entries are valid within `topk_len`.
- Constraint enforced by the caller: `topk_indices` **must not contain -1 padding** at the shared level.

### AMD-Specific Version (`amd/rocm.py`, L118-165)

Used by ROCm prefill. Also outputs **dense padded** combined indices (prefill still uses dense for combine), but includes an extra `TOPK_WIDTH` parameter.

**Why `TOPK_WIDTH` is needed**: ROCm's topk indices may have a physical width (`topk_width`) larger than the logical `TOP_K`. For example, the allocated buffer might be sized for `max_topk=256` while only `topk=64` entries are valid per token. The AMD kernel:

1. Loads safely using `TOPK_WIDTH` (the actual stride/width of `topk_indices`).
2. Masks to `topk_len` via `TOP_K`.
3. Filters invalid indices: `(topk_indices >= 0) & (topk_indices < N)` -- the shared version does not re-check validity because the common caller ensures no -1 entries make it into the combine step.

The `PADDED_TOP_K` is computed as `triton.next_power_of_2(topk_width)` for Triton block sizing, while the NVIDIA version uses `next_power_of_2` over `topk` (not `topk_width`).

### AMD-Specific Ragged Variant (`amd/rocm.py`, L370-424)

`combine_topk_swa_indices_ragged` produces ragged output directly instead of dense padded. It:

1. Pre-computes combined lengths via `_compute_combined_lens_kernel` (separate pass).
2. Builds indptr from those lengths.
3. Launches `_combine_topk_swa_indices_ragged_kernel` with a 3D grid `(num_reqs, 128 workers, cdiv(topk+window, 128) blocks)` to write directly into the ragged flat buffer.

This ragged variant is not currently used in the main forward pass but exists as a building block for potential future use where the ROCm prefill kernel could accept ragged input directly.

---

## Key Differences from NVIDIA's FlashMLASparseImpl

| Aspect | NVIDIA (`flashmla.py`) | AMD (`rocm.py`) |
|--------|----------------------|-----------------|
| **Sparse indices format** | Dense padded 2D tensors | Ragged 1D + indptr (CSR-like) |
| **Decode attention kernel** | `flash_mla_with_kvcache` (CUDA) | `rocm_sparse_attn_decode` (Triton) |
| **Prefill attention kernel** | `flash_mla_sparse_fwd` (CUDA) | `rocm_sparse_attn_prefill` (Triton) |
| **Q head padding** | Pads to 64 or 128 (`get_padded_num_q_heads`) | No padding (returns `num_heads` unchanged) |
| **C4A decode topk** | `compute_global_topk_indices_and_lens` (dense output) | `compute_global_topk_ragged_indices_and_indptr` (ragged output) |
| **C128A ragged build** | Not needed (kernel consumes dense) | Built from dense parent via `build_ragged_indices_from_dense` |
| **SWA ragged build** | Not needed | Built from dense parent via `build_ragged_indices_from_dense` |
| **`combine_topk_swa_indices`** | Uses shared version from `cache_utils.py` | Uses AMD-specific version with `TOPK_WIDTH` guard |
| **Prefill chunk size** | 4 (same constant) | 4 (same constant) |
| **`_forward_decode` signature** | Takes `FlashMLASparseMetadata` | Takes `DeepseekV4ROCMAiterMLASparseMetadata` |
| **Dequantize/gather KV** | Same shared `dequantize_and_gather_k_cache` | Same shared `dequantize_and_gather_k_cache` |

---

## Helper Utilities

### `_build_indptr_from_lengths` [L40-44]

Converts a lengths tensor to a CSR-style indptr via `cumsum`:

```python
indptr = zeros(len(lengths) + 1)
cumsum(lengths, out=indptr[1:])
```

### `compute_global_topk_ragged_indices_and_indptr` [L236-277]

Three-step pipeline for C4A decode:
1. `_compute_topk_lens_kernel`: Counts valid (>=0) entries per token, zeros out invalid tokens.
2. `_build_indptr_from_lengths`: Builds the ragged indptr.
3. `_pack_global_topk_ragged_kernel`: Resolves local indices to global KV cache slots via block table lookup and writes to the flat ragged buffer.

### `_copy_ragged_to_graph_buffers` [L427-448]

Copies ragged metadata into pre-allocated persistent CUDA graph buffers using `non_blocking=True` H2D copies.

---

## Related Cross-References

- [[DeepseekV4_KVCache_Ops]] -- Shared K-cache operations and the common `combine_topk_swa_indices`.
- [[V1AttentionBackend]] -- The framework of attention backends, metadata builders, and CUDA graph integration.
- [[NvidiaVsAMD]] -- Architectural comparison between the NVIDIA FlashMLA and AMD ROCm Aiter backends.

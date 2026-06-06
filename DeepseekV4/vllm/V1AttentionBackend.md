# V1AttentionBackend -- DeepseekV4FlashMLASparseBackend

## File: `vllm/models/deepseek_v4/nvidia/flashmla.py`

## Class Hierarchy

```
SparseMLAAttentionImpl[FlashMLASparseMetadata]   (vllm.v1.attention.backend)
  |
  +-- DeepseekV4SparseMLAAttentionImpl   (abstract, line 37)
        |
        +-- DeepseekV4FlashMLASparseImpl   (concrete, line 114)

FlashMLASparseBackend   (vllm.v1.attention.backends.mla.flashmla_sparse)
  |
  +-- DeepseekV4FlashMLASparseBackend   (line 79)
```

## DeepseekV4SparseMLAAttentionImpl (line 37-76)

Abstract parent class that overrides the V1 framework's expected interface.

**Key design decision -- Liskov-broken override (line 40-44):**

```python
class DeepseekV4SparseMLAAttentionImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
    """V4 sparse MLA is driven by the layer (``DeepseekV4MLAAttention.forward``)
    rather than the v1 framework, so ``forward_mqa`` is overridden with a
    classmethod that takes the layer as its first argument. This Liskov-broken
    override is intentional: the grandparent's instance-method ``forward_mqa``
    is never called on V4 layers."""
```

In V1, `forward_mqa` is normally an instance method called by the framework. For V4, `DeepseekV4MLAAttention.forward` calls `cls.forward_mqa(layer, ...)` directly, making the framework's orchestration a no-op.

**`PREFILL_CHUNK_SIZE = 4`** (line 52):
- Prefill is processed in chunks of 4 sequences at a time.
- This limits the bf16 gather workspace to `(4, M, head_dim)` bytes.
- The same constant is read by `DeepseekV4MLAAttention`'s dummy-run path to pre-reserve workspace during CUDA graph warmup.

**Abstract methods:**
- `forward_mqa(cls, layer, q, kv, positions, output)` (line 54-64) -- classmethod, first arg is the layer, not self.
- `get_padded_num_q_heads(cls, num_heads) -> int` (line 67-76) -- returns padded head count the wrapper should allocate q/output buffers at.

## DeepseekV4FlashMLASparseBackend (line 79-111)

Extends `FlashMLASparseBackend` with V4-specific overrides:

| Method | Return Value | Notes |
|---|---|---|
| `get_supported_kernel_block_sizes()` | `[256]` | Single fixed block size |
| `get_name()` | `"V4_FLASHMLA_SPARSE"` | Backend identifier |
| `get_impl_cls()` | `DeepseekV4FlashMLASparseImpl` | Concrete impl class |
| `get_supported_head_sizes()` | `[512]` | V4 layout: 448 NoPE + 64 RoPE (overrides V3.2 default of 576) |
| `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)` | `(num_blocks, block_size, 584)` when `cache_dtype_str=="fp8_ds_mla"` | 584 = 448 NoPE + 128 RoPE + 8 fp8 scale |

## DeepseekV4FlashMLASparseImpl (line 114-425)

### Head Padding (line 119-127)

```python
@classmethod
def get_padded_num_q_heads(cls, num_heads: int) -> int:
    # FP8 decode kernel only supports h_q = 64 or 128.
    return 64 if num_heads <= 64 else 128
```

The FP8 decode kernel (`flash_mla_with_kvcache` with FP8 KV cache) requires the Q head dimension to be exactly 64 or 128. The MLA wrapper allocates q/output at `[N, padded_heads, head_dim]`.

### Forward Dispatch (line 129-208)

`forward_mqa` splits into decode and prefill paths:

```
forward_mqa()
  |-- attn_metadata is None  -> warmup dummy run (line 149-165)
  |-- num_prefills > 0       -> _forward_prefill() (line 188-198)
  |-- num_decodes > 0        -> _forward_decode() (line 199-208)
```

**Warmup path** (lines 149-165): When `attn_metadata is None`, reserves the bf16 gather workspace via `current_workspace_manager().get_simultaneous()` and zeroes output. The dequantize/topk/sparse_fwd kernels are skipped. This is the CUDA graph warmup hook.

- `swa_only` is determined by `layer.compress_ratio <= 1`.
- `N` = `(max_model_len + compress_ratio - 1) // compress_ratio` (size of compressed KV pool).
- `M` = `N + window_size + max_num_batched_tokens`.

### Decode Path (line 210-302)

```
_forward_decode()
  |-- Determine topk_indices/topk_lens:
  |     compress_ratio == 4  (C4A)  -> compute_global_topk_indices_and_lens() (line 234)
  |     compress_ratio == 128 (C128A) -> pre-computed in metadata (line 244)
  |-- SWA indices from swa_metadata.decode_swa_indices / decode_swa_lens
  |-- q.unsqueeze(1)  -> (num_decode_tokens, 1, 1, head_dim)
  |-- swa_cache = swa_cache_layer.kv_cache.unsqueeze(-2)
  |-- kv_cache = kv_cache.unsqueeze(-2) if not swa_only
  |-- Select tile_scheduler_metadata per layer type (line 268-284):
  |     compress_ratio <= 1   -> tile_sched_swaonly
  |     compress_ratio == 4    -> tile_sched_c4a
  |     compress_ratio == 128  -> tile_sched_c128a
  |-- flash_mla_with_kvcache()  (line 286-302)
```

**Tile scheduler metadata** (lines 262-284): One metadata entry per layer type (SWA-only / C4A / C128A), shared across all same-type layers within a decode step. On the first call per type, the in-kernel planner runs (allocating `tile_scheduler_metadata` and `num_splits` via PyTorch's graph-aware allocator). Subsequent same-type layers see `have_initialized=True` on the metadata and skip the planner.

This is critical for CUDA graph capture -- the kernel arguments must live at stable addresses across graph replays.

**C4A topk indices** (lines 231-241): Local indices differ per layer, filled by the Indexer. Global indices are computed on the fly. **C128A topk indices** (lines 243-244): Pre-computed during metadata build, stored in `attn_metadata.c128a_global_decode_topk_indices`.

### Prefill Path (lines 304-424)

```
_forward_prefill()
  |-- Derive prefill-local token offsets from query_start_loc_cpu (line 335)
  |-- For non-SWA-only layers:
  |     C4A   -> topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
  |     C128A -> attn_metadata.c128a_prefill_topk_indices
  |-- Compute N, M (same as warmup)
  |-- For each chunk of PREFILL_CHUNK_SIZE (4) sequences (line 360-424):
  |     |-- Workspace: get_simultaneous((chunk_size, M, head_dim), bf16)
  |     |-- Dequantize & gather compressed KV cache (line 373-381)
  |     |-- Dequantize & gather SWA KV cache (line 385-393)
  |     |-- Combine topk + SWA indices (line 403-415)
  |     |-- flash_mla_sparse_fwd() (line 416-424)
```

### KV Cache Shapes

| Cache | Shape | Description |
|---|---|---|
| Main MLA KV cache | `(num_blocks, block_size, 584)` | 448 NoPE + 128 RoPE + 8 fp8 scale bytes |
| SWA KV cache | `(num_blocks, swa_block_size, head_dim)` | Sliding window attention cache |
| Compressed KV cache | Per-layer, `compress_ratio` buckets | Top-k compressed key-value pool |

## Metadata Types

```python
FlashMLASparseMetadata    # from vllm.v1.attention.backends.mla.flashmla_sparse
  - block_table, block_size
  - c128a_global_decode_topk_indices
  - c128a_decode_topk_lens
  - c128a_prefill_topk_indices

DeepseekSparseSWAMetadata # from vllm.v1.attention.backends.mla.sparse_swa
  - num_decodes, num_decode_tokens
  - num_prefills, num_prefill_tokens
  - decode_swa_indices, decode_swa_lens
  - prefill_seq_lens, prefill_gather_lens
  - query_start_loc, query_start_loc_cpu
  - block_table, block_size
  - tile_sched_swaonly, tile_sched_c4a, tile_sched_c128a
  - is_valid_token
```

## Attention Flow: Who Calls Whom

```
DeepseekV4Model.forward()
  -> DeepseekV4DecoderLayer.forward()
    -> DeepseekV4MLAAttention.forward()      # drives attention, not the V1 framework
      -> attn_fn = attn_backend.get_impl_cls().forward_mqa
      = DeepseekV4FlashMLASparseImpl.forward_mqa(layer, q, kv, positions, output)
```

The V1 framework provides metadata (block_table, seq_lens, tile_scheduler_metadata through metadata builders), but `DeepseekV4MLAAttention.forward` orchestrates the call, not the V1 attention runner.

Related notes: [[DeepseekV4MLAAttention]], [[DeepseekV4FlashMLASparseImpl]], [[CUDAGraphIntegration]], [[DeepseekV4Attention]]

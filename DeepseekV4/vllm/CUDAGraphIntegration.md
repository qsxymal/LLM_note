# CUDAGraphIntegration -- How DeepSeek V4 works with CUDA graphs

## Overview

DeepSeek V4's CUDA graph integration involves several mechanisms to handle the tension between CUDA graph's requirement for fixed tensor addresses and V4's dynamic runtime (variable batch size, dynamic topk indices, sparse attention, MegaMoE).

The constraints for CUDA graph capture are:
1. All tensor shapes and addresses must be fixed at graph capture time.
2. Dynamic data (indices, lengths, block tables) must use pre-allocated buffers with `copy_` updates.
3. Custom CUDA kernels (FlashMLA, MegaMoE) must be opaque to torch.compile or placed in `no_compile_layers`.

## 1. Static Forward Context for MegaMoE

File: `vllm/models/deepseek_v4/nvidia/model.py`, lines 214-219

```python
# Register in the static forward context so the custom-op wrapper
# can look up this module by name from within a torch.compile graph.
compilation_config = vllm_config.compilation_config
if prefix in compilation_config.static_forward_context:
    raise ValueError(f"Duplicate layer name: {prefix}")
compilation_config.static_forward_context[prefix] = self
```

During `DeepseekV4MegaMoEExperts.__init__`, each expert module registers itself by `prefix` into `compilation_config.static_forward_context`. This is a dictionary that maps layer name strings to module instances.

**Why this is needed:** Inside a `torch.compile`-captured graph, `self` (the module instance) is not accessible. The custom-op wrapper function `_deepseek_v4_mega_moe_experts_op` needs the actual module to dispatch to `_run_mega_moe`. It retrieves the module via the forward context:

```python
# Line 422
self = get_forward_context().no_compile_layers[layer_name]
self._run_mega_moe(hidden_states, topk_weights, topk_ids, out, ...)
```

## 2. no_compile_layers

File: `vllm/models/deepseek_v4/nvidia/model.py`, line 422

`get_forward_context().no_compile_layers[layer_name]` is populated by vLLM's compilation framework. Layers listed in `no_compile_layers` are excluded from `torch.compile` graph capture. The MegaMoE expert op is a custom CUDA kernel that:

- Cannot be traced by torch.compile (contains custom triton/CUDA kernels).
- Has dynamic dispatch (expert routing based on `topk_ids`).
- Needs access to module state (weight scales, quantization parameters).

The wrapper function signature (lines 413-421) takes all tensor inputs explicitly plus `layer_name` as a string:

```python
def _deepseek_v4_mega_moe_experts_op(
    hidden_states, topk_weights, topk_ids, out,
    layer_name, activation_clamp, fast_math,
) -> None:
    self = get_forward_context().no_compile_layers[layer_name]
    self._run_mega_moe(...)
```

This pattern allows `torch.compile` to trace the surrounding graph while treating the MegaMoE call as a black-box custom op.

## 3. MTP Hidden Buffer (Outside CUDAGraph Pool)

File: `vllm/models/deepseek_v4/nvidia/model.py`, lines 1169-1177

```python
if get_pp_group().is_last_rank:
    self._mtp_hidden_buffer = torch.empty(
        vllm_config.scheduler_config.max_num_batched_tokens,
        self.hc_dim,
        dtype=vllm_config.model_config.dtype,
        device=self.device,
    )
else:
    self._mtp_hidden_buffer = None
```

The MTP draft head needs to receive hidden states from the main model via `copy_`. By allocating this buffer outside the CUDA graph memory pool, the `copy_` operation in `forward()` correctly refreshes the data across captured shapes without triggering graph replay issues. The buffer is sized at `max_num_batched_tokens * hc_dim` and lives on the last PP rank only.

## 4. CUDA Graph Warmup in Attention

File: `vllm/models/deepseek_v4/nvidia/flashmla.py`, lines 149-165

When `attn_metadata is None` during CUDA graph warmup, the `forward_mqa` path pre-reserves the bf16 gather workspace:

```python
if attn_metadata is None:
    swa_only = layer.compress_ratio <= 1
    N = (0 if swa_only
         else (layer.max_model_len + layer.compress_ratio - 1) // layer.compress_ratio)
    M = N + layer.window_size + layer.max_num_batched_tokens
    current_workspace_manager().get_simultaneous(
        ((cls.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
    )
    output.zero_()
    return
```

This ensures the `workspace_manager` allocates the workspace with stable addresses that will be reused during graph replay. The dequantize/topk/sparse_fwd kernels are skipped during warmup since there is no real metadata.

## 5. Tile Scheduler Metadata (Graph-Aware Allocator)

File: `vllm/models/deepseek_v4/nvidia/flashmla.py`, lines 262-284

```python
if layer.compress_ratio <= 1:
    tile_metadata = swa_metadata.tile_sched_swaonly
elif layer.compress_ratio == 4:
    tile_metadata = swa_metadata.tile_sched_c4a
elif layer.compress_ratio == 128:
    tile_metadata = swa_metadata.tile_sched_c128a
```

The `tile_scheduler_metadata` is allocated via PyTorch's graph-aware allocator during metadata building (in `DeepseekSparseSWAMetadataBuilder.build_tile_scheduler`). This means:

- The first call per layer type triggers the in-kernel planner, which allocates `tile_scheduler_metadata` and `num_splits` tensors.
- Subsequent same-type layers see `have_initialized=True` and skip the planner, reusing the same tensors.
- During CUDA graph capture, these tensors have stable addresses, so the `flash_mla_with_kvcache` kernel arguments are graph-compatible.

Three separate tile scheduler metadata entries exist:
| Entry | Layer Type |
|---|---|
| `tile_sched_swaonly` | `compress_ratio <= 1` |
| `tile_sched_c4a` | `compress_ratio == 4` |
| `tile_sched_c128a` | `compress_ratio == 128` |

## 6. Opaque Custom CUDA Kernels

The following kernels are opaque to `torch.compile` and are never traced:

| Kernel | Location | Purpose |
|---|---|---|
| `flash_mla_with_kvcache` | `vllm.v1.attention.ops.flashmla` | Multi-head latent attention with KV cache (decode) |
| `flash_mla_sparse_fwd` | `vllm.v1.attention.ops.flashmla` | Sparse MLA forward (prefill) |
| MegaMoE experts | `vllm.models.deepseek_v4.nvidia.model` | Fused MoE expert computation |
| `dequantize_and_gather_k_cache` | `vllm.models.deepseek_v4.common.ops` | FP8 dequant + KV gather |

These are excluded from the `torch.compile` graph either via `no_compile_layers` or because they are custom CUDA/Triton kernels registered as PyTorch ops.

## 7. ROCm: Copy Ragged to Graph Buffers

File: `vllm/models/deepseek_v4/amd/rocm.py`, lines 427-448

```python
def _copy_ragged_to_graph_buffers(
    ragged_indices, ragged_indptr,
    ragged_indices_buffer, ragged_indptr_buffer,
    num_rows, max_entries_per_row,
) -> tuple[Tensor, Tensor]:
    indptr_out = ragged_indptr_buffer[:num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)
    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    nnz = ragged_indices.numel()
    if nnz > 0:
        ragged_out[:nnz].copy_(ragged_indices, non_blocking=True)
    return ragged_out, indptr_out
```

On ROCm, the dynamic ragged indices/indptr (from combining topk + SWA indices) must be copied into pre-allocated graph-stable buffers. This ensures CUDA graph replay sees the same tensor addresses every time. The buffers are sized to the maximum possible entries (`num_rows * max_entries_per_row`) and slicing keeps the storage address stable.

## Summary: CUDA Graph Compatibility per Component

| Component | Mechanism | Stable Addresses? |
|---|---|---|
| Q/KV/output tensors | Fixed max batch allocation, padded heads | Yes |
| MegaMoE experts | `no_compile_layers` lookup by name | N/A (excluded from graph) |
| Tile scheduler metadata | PyTorch graph-aware allocator | Yes |
| bf16 gather workspace | `workspace_manager.get_simultaneous()` | Yes (pre-reserved during warmup) |
| MTP hidden buffer | Pre-allocated outside graph pool | Yes |
| Ragged indices (ROCm) | Pre-allocated buffers, `copy_` update | Yes |
| `flash_mla_with_kvcache` | Custom CUDA kernel, opaque to compile | Yes (fixed args from graph) |
| Topk indices buffer | Pre-allocated per layer (`topk_indices_buffer`) | Yes |
| SWA indices | From metadata, backed by graph-aware storage | Yes |
| Attention sink | `layer.attn_sink` (constant scalar or tensor) | Yes |

Related notes: [[V1AttentionBackend]], [[DeepseekV4ForCausalLM]], [[DeepseekV4MLAAttention]], [[DeepseekV4MoE]], [[NvidiaVsAMD]]

# CUDAGraphIntegration -- DeepSeek V4 如何与 CUDA 图协同工作

## 概述

DeepSeek V4 的 CUDA 图集成涉及多种机制，以处理 CUDA 图对固定张量地址的要求与 V4 动态运行时（可变批大小、动态 topk 索引、稀疏注意力、MegaMoE）之间的矛盾。

CUDA 图捕获的约束条件如下：
1. 所有张量形状和地址必须在图捕获时固定。
2. 动态数据（索引、长度、块表）必须使用预分配的缓冲区并通过 `copy_` 更新。
3. 自定义 CUDA 内核（FlashMLA、MegaMoE）必须对 `torch.compile` 不透明，或放置在 `no_compile_layers` 中。

## 1. MegaMoE 的静态前向上下文

文件：`vllm/models/deepseek_v4/nvidia/model.py`，第 214-219 行

```python
# Register in the static forward context so the custom-op wrapper
# can look up this module by name from within a torch.compile graph.
compilation_config = vllm_config.compilation_config
if prefix in compilation_config.static_forward_context:
    raise ValueError(f"Duplicate layer name: {prefix}")
compilation_config.static_forward_context[prefix] = self
```

在 `DeepseekV4MegaMoEExperts.__init__` 期间，每个专家模块通过 `prefix` 将自己注册到 `compilation_config.static_forward_context` 中。这是一个将层名称字符串映射到模块实例的字典。

**为什么需要这样做：** 在 `torch.compile` 捕获的图内部，`self`（模块实例）是不可访问的。自定义算子包装函数 `_deepseek_v4_mega_moe_experts_op` 需要实际的模块来分派到 `_run_mega_moe`。它通过前向上下文检索模块：

```python
# Line 422
self = get_forward_context().no_compile_layers[layer_name]
self._run_mega_moe(hidden_states, topk_weights, topk_ids, out, ...)
```

## 2. no_compile_layers

文件：`vllm/models/deepseek_v4/nvidia/model.py`，第 422 行

`get_forward_context().no_compile_layers[layer_name]` 由 vLLM 的编译框架填充。列在 `no_compile_layers` 中的层被排除在 `torch.compile` 图捕获之外。MegaMoE 专家算子是一个自定义 CUDA 内核，它：

- 无法被 torch.compile 追踪（包含自定义 Triton/CUDA 内核）。
- 具有动态分派（基于 `topk_ids` 的专家路由）。
- 需要访问模块状态（权重缩放因子、量化参数）。

包装函数签名（第 413-421 行）显式接收所有张量输入以及作为字符串的 `layer_name`：

```python
def _deepseek_v4_mega_moe_experts_op(
    hidden_states, topk_weights, topk_ids, out,
    layer_name, activation_clamp, fast_math,
) -> None:
    self = get_forward_context().no_compile_layers[layer_name]
    self._run_mega_moe(...)
```

这种模式允许 `torch.compile` 追踪周围的图，同时将 MegaMoE 调用视为不透明的自定义算子。

## 3. MTP 隐藏状态缓冲区（在 CUDAGraph 内存池之外）

文件：`vllm/models/deepseek_v4/nvidia/model.py`，第 1169-1177 行

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

MTP 草稿头需要通过 `copy_` 从主模型接收隐藏状态。通过将此缓冲区分配在 CUDA 图内存池之外，`forward()` 中的 `copy_` 操作能够正确地在已捕获的形状之间刷新数据，而不会触发图重放问题。该缓冲区的大小为 `max_num_batched_tokens * hc_dim`，且仅存在于最后一个 PP 等级上。

## 4. 注意力中的 CUDA 图预热

文件：`vllm/models/deepseek_v4/nvidia/flashmla.py`，第 149-165 行

当 CUDA 图预热期间 `attn_metadata is None` 时，`forward_mqa` 路径会预保留 bf16 gather 工作空间：

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

这确保了 `workspace_manager` 分配具有稳定地址的工作空间，该地址将在图重放期间被重用。由于没有真实的元数据，去量化/topk/sparse_fwd 内核在预热期间被跳过。

## 5. Tile 调度器元数据（图感知分配器）

文件：`vllm/models/deepseek_v4/nvidia/flashmla.py`，第 262-284 行

```python
if layer.compress_ratio <= 1:
    tile_metadata = swa_metadata.tile_sched_swaonly
elif layer.compress_ratio == 4:
    tile_metadata = swa_metadata.tile_sched_c4a
elif layer.compress_ratio == 128:
    tile_metadata = swa_metadata.tile_sched_c128a
```

`tile_scheduler_metadata` 在元数据构建期间（在 `DeepseekSparseSWAMetadataBuilder.build_tile_scheduler` 中）通过 PyTorch 的图感知分配器分配。这意味着：

- 每层类型的第一次调用会触发内核内规划器，该规划器分配 `tile_scheduler_metadata` 和 `num_splits` 张量。
- 后续相同类型的层看到 `have_initialized=True` 并跳过规划器，重用相同的张量。
- 在 CUDA 图捕获期间，这些张量具有稳定的地址，因此 `flash_mla_with_kvcache` 内核参数是与图兼容的。

存在三个独立的 tile 调度器元数据条目：
| 条目 | 层类型 |
|---|---|
| `tile_sched_swaonly` | `compress_ratio <= 1` |
| `tile_sched_c4a` | `compress_ratio == 4` |
| `tile_sched_c128a` | `compress_ratio == 128` |

## 6. 不透明的自定义 CUDA 内核

以下内核对于 `torch.compile` 是不透明的，永远不会被追踪：

| 内核 | 位置 | 用途 |
|---|---|---|
| `flash_mla_with_kvcache` | `vllm.v1.attention.ops.flashmla` | 带 KV 缓存的多头潜在注意力（解码） |
| `flash_mla_sparse_fwd` | `vllm.v1.attention.ops.flashmla` | 稀疏 MLA 前向（预填充） |
| MegaMoE experts | `vllm.models.deepseek_v4.nvidia.model` | 融合 MoE 专家计算 |
| `dequantize_and_gather_k_cache` | `vllm.models.deepseek_v4.common.ops`（详见 [[DeepseekV4_KVCache_Ops#dequantize_and_gather_k_cache]]） | FP8 去量化 + KV gather |

这些内核要么通过 `no_compile_layers` 被排除在 `torch.compile` 图之外，要么因为它们是注册为 PyTorch 算子的自定义 CUDA/Triton 内核。

## 7. ROCm：复制不规则数据到图缓冲区

文件：`vllm/models/deepseek_v4/amd/rocm.py`，第 427-448 行

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

在 ROCm 上，动态的不规则索引/indptr（来自组合 topk + SWA 索引）必须复制到预分配的图稳定缓冲区中。这确保了 CUDA 图重放每次看到相同的张量地址。缓冲区大小设置为最大可能条目数（`num_rows * max_entries_per_row`），切片操作保持存储地址稳定。

## 总结：各组件的 CUDA 图兼容性

| 组件 | 机制 | 地址稳定？ |
|---|---|---|
| Q/KV/输出张量 | 固定最大批分配，填充头数 | 是 |
| MegaMoE 专家 | 按名称的 `no_compile_layers` 查找 | 不适用（被排除在图外） |
| Tile 调度器元数据 | PyTorch 图感知分配器 | 是 |
| bf16 gather 工作空间 | `workspace_manager.get_simultaneous()` | 是（预热期间预保留） |
| MTP 隐藏状态缓冲区 | 在图池外预分配 | 是 |
| 不规则索引（ROCm） | 预分配缓冲区，`copy_` 更新 | 是 |
| `flash_mla_with_kvcache` | 自定义 CUDA 内核，对编译不透明 | 是（来自图的固定参数） |
| Topk 索引缓冲区 | 每层预分配（`topk_indices_buffer`） | 是 |
| SWA 索引 | 来自元数据，由图感知存储支持 | 是 |
| Attention sink | `layer.attn_sink`（常量标量或张量） | 是 |

相关笔记：[[V1AttentionBackend]], [[DeepseekV4ForCausalLM]], [[DeepseekV4MLAAttention]], [[DeepseekV4MoE]], [[NvidiaVsAMD]]

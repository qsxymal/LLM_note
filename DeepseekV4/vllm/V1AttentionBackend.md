# V1AttentionBackend -- DeepseekV4FlashMLASparseBackend

## 文件：`vllm/models/deepseek_v4/nvidia/flashmla.py`

## 类层次结构

```
SparseMLAAttentionImpl[FlashMLASparseMetadata]   (vllm.v1.attention.backend)
  |
  +-- DeepseekV4SparseMLAAttentionImpl   (抽象类, 第 37 行)
        |
        +-- DeepseekV4FlashMLASparseImpl   (具体类, 第 114 行)

FlashMLASparseBackend   (vllm.v1.attention.backends.mla.flashmla_sparse)
  |
  +-- DeepseekV4FlashMLASparseBackend   (第 79 行)
```

## DeepseekV4SparseMLAAttentionImpl（第 37-76 行）

覆盖 V1 框架期望接口的抽象父类。

**关键设计决策 -- 违反里氏替换原则的覆写（第 40-44 行）：**

```python
class DeepseekV4SparseMLAAttentionImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
    """V4 sparse MLA is driven by the layer (``DeepseekV4MLAAttention.forward``)
    rather than the v1 framework, so ``forward_mqa`` is overridden with a
    classmethod that takes the layer as its first argument. This Liskov-broken
    override is intentional: the grandparent's instance-method ``forward_mqa``
    is never called on V4 layers."""
```

在 V1 中，`forward_mqa` 通常是一个由框架调用的实例方法。对于 V4，`DeepseekV4MLAAttention.forward` 直接调用 `cls.forward_mqa(layer, ...)`，使得框架的编排成为空操作。

**`PREFILL_CHUNK_SIZE = 4`**（第 52 行）：
- Prefill 每次以 4 个序列为一块进行处理。
- 这限制了 bf16 收集工作空间的大小为 `(4, M, head_dim)` 字节。
- 同样的常量被 `DeepseekV4MLAAttention` 的 dummy-run 路径读取，用于在 CUDA 图预热期间预保留工作空间。

**抽象方法：**
- `forward_mqa(cls, layer, q, kv, positions, output)`（第 54-64 行）-- 类方法，第一个参数是层，不是 self。
- `get_padded_num_q_heads(cls, num_heads) -> int`（第 67-76 行）-- 返回封装器应分配的 q/output 缓冲区的填充后头数。

## DeepseekV4FlashMLASparseBackend（第 79-111 行）

使用 V4 特定覆写扩展 `FlashMLASparseBackend`：

| 方法 | 返回值 | 备注 |
|---|---|---|
| `get_supported_kernel_block_sizes()` | `[256]` | 单个固定块大小 |
| `get_name()` | `"V4_FLASHMLA_SPARSE"` | 后端标识符 |
| `get_impl_cls()` | `DeepseekV4FlashMLASparseImpl` | 具体实现类 |
| `get_supported_head_sizes()` | `[512]` | V4 布局：448 NoPE + 64 RoPE（覆盖 V3.2 默认的 576） |
| `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)` | `(num_blocks, block_size, 584)` 当 `cache_dtype_str=="fp8_ds_mla"` 时 | 584 = 448 NoPE + 128 RoPE + 8 fp8 缩放 |

## DeepseekV4FlashMLASparseImpl（第 114-425 行）

### 头填充（第 119-127 行）

```python
@classmethod
def get_padded_num_q_heads(cls, num_heads: int) -> int:
    # FP8 decode kernel only supports h_q = 64 or 128.
    return 64 if num_heads <= 64 else 128
```

FP8 解码内核（`flash_mla_with_kvcache` 配合 FP8 KV 缓存）要求 Q 头维度恰好为 64 或 128。MLA 封装器在 `[N, padded_heads, head_dim]` 处分配 q/output。

### 前向分发（第 129-208 行）

`forward_mqa` 拆分为解码和 prefill 路径：

```
forward_mqa()
  |-- attn_metadata is None  -> 预热 dummy 运行（第 149-165 行）
  |-- num_prefills > 0       -> _forward_prefill()（第 188-198 行）
  |-- num_decodes > 0        -> _forward_decode()（第 199-208 行）
```

**预热路径**（第 149-165 行）：当 `attn_metadata is None` 时，通过 `current_workspace_manager().get_simultaneous()` 预留 bf16 收集工作空间并将输出置零。反量化/topk/sparse_fwd 内核被跳过。这是 CUDA 图预热钩子。

- `swa_only` 由 `layer.compress_ratio <= 1` 确定。
- `N` = `(max_model_len + compress_ratio - 1) // compress_ratio`（压缩 KV 池的大小）。
- `M` = `N + window_size + max_num_batched_tokens`。

### 解码路径（第 210-302 行）

[[DeepseekV4_KVCache_Ops#compute_global_topk_indices_and_lens|compute_global_topk_indices_and_lens]]

```
_forward_decode()
  |-- 确定 topk_indices/topk_lens：
  |     compress_ratio == 4  (C4A)  -> compute_global_topk_indices_and_lens()（第 234 行）
  |     compress_ratio == 128 (C128A) -> 在元数据中预计算（第 244 行）
  |-- SWA 索引来自 swa_metadata.decode_swa_indices / decode_swa_lens
  |-- q.unsqueeze(1)  -> (num_decode_tokens, 1, 1, head_dim)
  |-- swa_cache = swa_cache_layer.kv_cache.unsqueeze(-2)
  |-- kv_cache = kv_cache.unsqueeze(-2)（如果不是仅 SWA）
  |-- 按层类型选择 tile_scheduler_metadata（第 268-284 行）：
  |     compress_ratio <= 1   -> tile_sched_swaonly
  |     compress_ratio == 4    -> tile_sched_c4a
  |     compress_ratio == 128  -> tile_sched_c128a
  |-- flash_mla_with_kvcache()（第 286-302 行）
```

**Tile 调度器元数据**（第 262-284 行）：每种层类型（仅 SWA / C4A / C128A）一个元数据条目，在一个解码步骤中所有同类型层共享。在首次调用每种类型时，内核内规划器运行（通过 PyTorch 的图感知分配器分配 `tile_scheduler_metadata` 和 `num_splits`）。后续的同类型层看到元数据上的 `have_initialized=True` 并跳过规划器。

这对 CUDA 图捕获至关重要——内核参数必须在图重放期间驻留在稳定的地址上。

**C4A topk 索引**（第 231-241 行）：局部索引每层不同，由 Indexer 填充。全局索引即时计算。**C128A topk 索引**（第 243-244 行）：在元数据构建期间预计算，存储在 `attn_metadata.c128a_global_decode_topk_indices`。

### Prefill 路径（第 304-424 行）

```
_forward_prefill()
  |-- 从 query_start_loc_cpu 派生 prefill 局部的 token 偏移（第 335 行）
  |-- 对于非仅 SWA 层：
  |     C4A   -> topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
  |     C128A -> attn_metadata.c128a_prefill_topk_indices
  |-- 计算 N, M（与预热相同）
  |-- 对于每个 PREFILL_CHUNK_SIZE（4）序列的块（第 360-424 行）：
  |     |-- 工作空间：get_simultaneous((chunk_size, M, head_dim), bf16)
  |     |-- 反量化并收集压缩的 KV 缓存（第 373-381 行）
  |     |-- 反量化并收集 SWA KV 缓存（第 385-393 行）
  |     |-- 合并 topk + SWA 索引（第 403-415 行）
  |     |-- flash_mla_sparse_fwd()（第 416-424 行）
```

### KV 缓存形状

| 缓存 | 形状 | 描述 |
|---|---|---|
| 主 MLA KV 缓存 | `(num_blocks, block_size, 584)` | 448 NoPE + 128 RoPE + 8 fp8 缩放字节 |
| SWA KV 缓存 | `(num_blocks, swa_block_size, head_dim)` | 滑动窗口注意力缓存 |
| 压缩 KV 缓存 | 每层，`compress_ratio` 桶 | Top-k 压缩键值池 |

## 元数据类型

```python
FlashMLASparseMetadata    # 来自 vllm.v1.attention.backends.mla.flashmla_sparse
  - block_table, block_size
  - c128a_global_decode_topk_indices
  - c128a_decode_topk_lens
  - c128a_prefill_topk_indices

DeepseekSparseSWAMetadata # 来自 vllm.v1.attention.backends.mla.sparse_swa
  - num_decodes, num_decode_tokens
  - num_prefills, num_prefill_tokens
  - decode_swa_indices, decode_swa_lens
  - prefill_seq_lens, prefill_gather_lens
  - query_start_loc, query_start_loc_cpu
  - block_table, block_size
  - tile_sched_swaonly, tile_sched_c4a, tile_sched_c128a
  - is_valid_token
```

## 注意力流程：调用关系

```
DeepseekV4Model.forward()
  -> DeepseekV4DecoderLayer.forward()
    -> DeepseekV4MLAAttention.forward()      # 驱动注意力，而不是 V1 框架
      -> attn_fn = attn_backend.get_impl_cls().forward_mqa
      = DeepseekV4FlashMLASparseImpl.forward_mqa(layer, q, kv, positions, output)
```

V1 框架提供元数据（block_table、seq_lens、tile_scheduler_metadata 通过元数据构建器），但 `DeepseekV4MLAAttention.forward` 负责编排调用，而不是 V1 注意力运行器。

相关笔记：[[DeepseekV4MLAAttention]], [[DeepseekV4FlashMLASparseImpl]], [[CUDAGraphIntegration]], [[DeepseekV4Attention]]

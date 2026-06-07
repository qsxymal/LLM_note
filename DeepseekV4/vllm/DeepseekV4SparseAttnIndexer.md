# DeepseekV4SparseAttnIndexer -- 稀疏注意力索引器

**文件:** `vllm/model_executor/layers/sparse_attn_indexer.py` (537 行)

---

## 1. 模块定位：Indexer 在 DeepSeek V4 注意力中的作用

在 DeepSeek V4 的 C4A（Compress-4-Attention）层中，模型需要在完整的历史 KV 缓存中仅选出与当前 Query 最相关的少量 token 进行注意力计算，以降低计算量。`SparseAttnIndexer` 正是负责这一 **稀疏索引选择** 的核心组件：

1. **接收**：经过 RoPE 旋转和量化后的 Q（FP8 或 MXFP4），以及 KV 缓存中的 K。
2. **计算**：通过 `mqa_logits`（prefill）或 `paged_mqa_logits`（decode）计算 Q 与 K 的注意力 logits。
3. **选择**：通过 `top_k_per_row` 或 `persistent_topk` 从 logits 中选出 top-k 个 token 索引。
4. **输出**：填充 `topk_indices_buffer`，供下游的稀疏注意力后端（如 [[DeepseekV4FlashMLASparseImpl]]）使用。

Indexer 是 DeepSeek V4 稀疏注意力机制中的 **门控/筛选** 环节，其输出质量直接决定了后续稀疏注意力是在有用信息上计算还是在噪声上计算。

**在整体数据流中的位置：**

```
DeepseekV4Indexer (生成 Q 量化、K 缓存写入)
    |
    v
SparseAttnIndexer (Q @ K 得到 logits, 选出 top-k 索引)  <-- 本模块
    |
    v
DeepseekV4FlashMLASparseImpl (根据 top-k 索引执行稀疏 MLA 注意力)
```

与 [[DeepseekV4Indexer]] 的关系：
- `DeepseekV4Indexer` 负责 **准备输入**：将 Query 投影、RoPE 旋转、量化、写入 K 缓存。
- `SparseAttnIndexer` 负责 **执行索引**：用 Q 去 K 缓存中检索最相关的 token。

---

## 2. 类结构与方法

### 2.1 `SparseAttnIndexer(CustomOp)` (L406-537)

通过 `@CustomOp.register("sparse_attn_indexer")` 注册为自定义算子层。

**构造函数参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `k_cache` | `KVCache` | K 缓存对象，包含 `prefix` 和 `kv_cache` |
| `quant_block_size` | `int` | 量化块大小（如 128） |
| `scale_fmt` | `str` | 缩放因子格式（如 "UE8M0"） |
| `topk_tokens` | `int` | 每个 head 选择的 top-k 数量 |
| `head_dim` | `int` | head 维度（DeepSeek V4 为 128） |
| `max_model_len` | `int` | 模型最大长度 |
| `max_total_seq_len` | `int` | 总序列最大长度（用于 workspace 分配） |
| `topk_indices_buffer` | `torch.Tensor` | 共享的 top-k 索引输出缓冲区 |
| `skip_k_cache_insert` | `bool` | 是否跳过 K 缓存写入（默认 False） |
| `use_fp4_cache` | `bool` | 是否使用 MXFP4 K 缓存（默认 False） |

**重要检查** (L443-446)：CUDA 平台必须安装 DeepGEMM，否则抛出 `RuntimeError`。

### 2.2 方法

| 方法 | 说明 |
|------|------|
| `forward_native()` | 根调度：CUDA/XPU -> `forward_cuda`，ROCm -> `forward_hip` |
| `forward_cuda()` | 解包 q_quant（tuple 时拆为 values + scales），调用 `torch.ops.vllm.sparse_attn_indexer` |
| `forward_xpu()` | 直接委托给 `forward_cuda`（XPU 仅支持 FP8） |
| `forward_hip()` | 通过 `rocm_aiter_ops.rocm_aiter_sparse_attn_indexer` 执行，要求 `VLLM_ROCM_USE_AITER=1`，不支持 FP4 |

### 2.3 q_quant 输入格式

```python
# FP8 路径：单张量 (per-token scale 已折叠进 weights)
q_quant: torch.Tensor  # shape: (T, H, head_dim), float8_e4m3fn

# MXFP4 路径：双张量元组
q_quant: tuple[
    torch.Tensor,  # values: (T, H, head_dim // 2), uint8 (2值压缩为1字节)
    torch.Tensor,  # scales:  (T, H, head_dim // 32), uint8 (ue8m0)
]
```

---

## 3. 自定义算子 `sparse_attn_indexer` 详解 (L81-373)

该函数通过 `direct_register_custom_op` 注册为 PyTorch 自定义算子，包含两条路径。

### 3.1 算子注册 (L397-403)

```python
direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],  # 原地修改！
    fake_impl=sparse_attn_indexer_fake,    # 虚设实现
    dispatch_key=current_platform.dispatch_key,
)
```

- `mutates_args=["topk_indices_buffer"]`：声明该算子会原地修改 `topk_indices_buffer`。
- `fake_impl`：用于 torch.compile 的虚设实现，直接返回 `topk_indices_buffer`。

### 3.2 Dummy/Profiling 路径 (L106-141)

当 `attn_metadata` 不是 dict 类型时（处于 profiling 或 dummy 运行阶段）：

1. **预留 workspace**：调用 `_gather_workspace_shapes` 计算 gather workspace 的 shape/dtype，然后通过 `current_workspace_manager().get_simultaneous()` 预留。
2. **预留 radix topk workspace**：大小为 `RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024` 字节的 uint8 张量。
3. **预留 peak logits 张量**：大小为 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` MB 的假张量，用于模拟推理时的峰值显存。
4. **返回 fake 结果**：调用 `sparse_attn_indexer_fake` 直接返回 `topk_indices_buffer`。

### 3.3 真实运行路径 (L142-373)

从 `attn_metadata` 中提取当前层（通过 `k_cache_prefix` 索引）的元数据：

```python
attn_metadata_narrowed = attn_metadata[k_cache_prefix]
assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
```

元数据来自 [[V1AttentionBackend|注意力后端]] 中的 `DeepseekV32IndexerMetadata`，包含 `slot_mapping`、`num_decodes`、`num_prefills`、`prefill`、`decode` 等字段。

**K 缓存写入 (L163-173)**：若 `skip_k_cache_insert=False`，调用 `ops.indexer_k_quant_and_cache` 将当前 token 的 K 量化后写入 KV 缓存。

**初始化 topk_indices_buffer (L175)**：将前 `hidden_states.shape[0]` 行置为 -1（无效索引标记）。

---

## 4. Prefill 阶段详细流程 (L176-256)

对 `prefill_metadata.chunks` 中的每个 chunk 依次处理：

### 步骤 1：分配 gather workspace (L184-194)

```python
k_quant_full, k_scale_full = workspace_manager.get_simultaneous(values_spec, scales_spec)
k_quant = k_quant_full[:chunk.total_seq_lens]
k_scale = k_scale_full[:chunk.total_seq_lens]
```

Gather workspace 大小由 `_gather_workspace_shapes()` 决定：

| 路径 | values shape | values dtype | scales shape | scales dtype |
|------|-------------|-------------|-------------|-------------|
| FP8 | `(total_seq_lens, head_dim)` | `fp8_dtype` | `(total_seq_lens, 4)` | `uint8` |
| MXFP4 | `(total_seq_lens, head_dim//2)` | `uint8` | `(total_seq_lens, head_dim//32)` | `uint8` |

### 步骤 2：从 KV 缓存 gather K (L196-203)

若 `chunk.skip_kv_gather=False`，调用 `ops.cp_gather_indexer_k_quant_cache`：

```
输入：kv_cache (3D uint8), block_table, cu_seq_lens
输出：k_quant (已量化 K), k_scale (量化 scale)
```

该算子从分页 KV 缓存中根据 block table 取出指定序列的 K，量化为 FP8/MXFP4 格式，写入 workspace。

### 步骤 3：计算 MQA logits (L211-240)

调用 `fp8_fp4_mqa_logits`（或 XPU 的 `torch.ops.vllm.xpu_fp8_mqa_logits`）计算注意力 logits：

```
logits = fp8_fp4_mqa_logits(
    (q_slice_cast, q_scale_slice),  # Q 量化值 + 可选的 MXFP4 scale
    (k_quant_cast, k_scale_cast),    # K 量化值 + scale
    weights,                         # indexer_weights (per-head 加权)
    cu_seqlen_ks,                    # 每个序列的 K 起始位置
    cu_seqlen_ke,                    # 每个序列的 K 结束位置
    clean_logits=False,              # 保留 logits 实际值（不清理为 0）
)
```

**重要 — weights 已融合 Q 缩放因子 (weight-fold)**：

传递给 `fp8_fp4_mqa_logits` 的 `weights` 参数是来自 `FusedIndexerQ` 的 `weights_out`，不是原始的 `indexer_weights`。在 **FP8 路径**中，per-token 的 Q 量化缩放因子 `q_scale` 已被折叠（fold）进 weights：

```
# FP8 路径：weights 融合了 q_scale
weights_out = index_weights × q_scale × softmax_scale × head_scale  # (T, H) float32

# MXFP4 路径：q_scale 无法折叠（per-block 粒度），单独传递
weights_out = index_weights × softmax_scale × head_scale            # (T, H) float32
q_scale → 单独作为 (q_slice_cast, q_scale_slice) 元组传入
```

这意味着 FP8 路径下 `weights` 的 dtype 是 **float32**（非 bf16），且已编码了 Q 的量化缩放信息。详细推导见 [[FusedIndexerQ#2. 权重折叠设计原理]]。

**DeepGEMM 类型转换 (L213-220)**：
- MXFP4 路径：values `-> int8` (kPackedFP4 标签)，scales `-> int32` 并 squeeze 最后一维。
- FP8 路径：保持原 dtype，scales `-> float32` 并 squeeze 最后一维。

### 步骤 4：Top-K 选择 (L247-256)

```python
ops.top_k_per_row_prefill(
    logits,
    cu_seqlen_ks, cu_seqlen_ke,     # 每个序列的范围
    topk_indices,                     # 输出缓冲区
    num_rows, topk_tokens,            # 行数和 top-k 数量
    logits.stride(0), logits.stride(1),  # stride 信息
)
```

该算子对每个序列（由 `cu_seqlen_ks/ke` 界定）独立执行 top-k 选择，将结果写入 `topk_indices_buffer` 的对应行。

---

## 5. Decode 阶段详细流程 (L258-372)

### 步骤 1：KV 缓存视图转换 (L261)

```python
kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
```

将 3D KV 缓存（`[num_blocks, block_size, page_bytes]`）转换为 DeepGEMM 需要的 4D 视图（`[num_blocks, block_size, 1, fp4_bytes]`）。详见 [[DeepseekV4_KVCache_Ops]].

### 步骤 2：Q Padding (L263-294)

当 `decode_metadata.requires_padding=True` 时，需要对 Q 进行 padding 以适应 CUDA graph 的固定 batch size：

```python
padded_q_quant_decode_tokens = pack_seq_triton(
    q_quant[:num_decode_tokens], decode_lens, pad_value=0
)
```

- **MXFP4 路径**：`pad_value=0` 确保 padding 位置的 dequant 结果为 0，不会产生 NaN/Inf。
- **FP8 路径**：使用 `pack_seq_triton` 默认的 fp32 pad 路径（因为下游 `context_lens` 会 mask 掉 padding 位置）。

Padding 逻辑通过 `pack_seq_triton` / `unpack_seq_triton` 实现，这保证了与 CUDA graph 兼容的固定输入形状。

### 步骤 3：Paged MQA Logits (L308-333)

调用 `fp8_fp4_paged_mqa_logits`（或 XPU 替代实现）：

```
logits = fp8_fp4_paged_mqa_logits(
    (padded_q_quant_cast, padded_q_scale),  # 已 padding 的 Q
    kv_cache,                                 # 4D KV 缓存视图
    weights[:num_padded_tokens],             # indexer_weights
    seq_lens,                                 # 每个序列的当前长度（2D: (B, next_n)）
    decode_metadata.block_table,              # block table
    decode_metadata.schedule_metadata,        # 调度元数据
    max_model_len=max_model_len,
    clean_logits=False,
)
```

**关键参数**：
- `seq_lens` 始终是 2D：`(B, next_n)` 用于原生 speculative decode，`(B, 1)` 用于普通 decode。DeepGEMM 的 `paged_mqa_logits` 要求 2D `context_lens`。
- `decode_metadata.schedule_metadata`：包含序列调度信息（如哪些 block 需要加载）。

### 步骤 4：Decode Top-K 选择 (L337-360)

有两个实现路径：

**CUDA + 特定 topk_tokens (512/1024/2048)**：使用 `persistent_topk` 内核 (L342-349)：

```python
torch.ops._C.persistent_topk(
    logits, seq_lens, topk_indices, topk_workspace,
    topk_tokens, attn_metadata_narrowed.max_seq_len,
)
```

- 需要额外的 `RADIX_TOPK_WORKSPACE_SIZE` workspace。
- 性能更优，适合较大的 top-k 值。

**其他情况**：使用 `ops.top_k_per_row_decode` (L351-360)：

```
支持更通用的 top-k 数量，但性能可能不如 persistent_topk。
```

### 步骤 5：Unpack 结果 (L362-371)

若 `requires_padding=True`，需要对 topk_indices 执行 unpadding：

```python
topk_indices = unpack_seq_triton(
    topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
    decode_lens,
)
topk_indices_buffer[:topk_indices.shape[0], :topk_indices.shape[-1]] = topk_indices
```

将 padding 的 token 索引去除，只保留实际 token 的索引。

---

## 6. 关键 Tensor 的 shape/dtype 表

| Tensor | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `hidden_states` | `(T, H, head_dim)` | `bf16` | 输入 hidden states（用于 profiling） |
| `q_quant` (FP8) | `(T, H, head_dim)` | `float8_e4m3fn` | 量化的 Q 值 |
| `q_quant` (MXFP4) | `(T, H, head_dim//2)` | `uint8` | 压缩的 MXFP4 Q 值（2值/字节） |
| `q_scale` (MXFP4) | `(T, H, head_dim//32)` | `uint8` (ue8m0) | MXFP4 Q 的 scale |
| `k` | `(T, K_dim)` | `fp8_dtype` | 当前 token 的 K（用于缓存写入） |
| `weights` | `(T, H)` | `float32` | 融合了 Q 量化缩放因子的 per-head 加权（FP8 路径下 `q_scale` 已折叠入 weights，见 [[FusedIndexerQ#2. 权重折叠设计原理]]） |
| `kv_cache` (FP8) | `(num_blocks, block_size, page_bytes)` | `uint8` | 3D KV 缓存原始视图 |
| `kv_cache` (MXFP4) | `(num_blocks, block_size, 1, fp4_bytes)` | `uint8` | 4D KV 缓存视图 |
| `k_quant` (FP8) | `(total_seq_lens, head_dim)` | `fp8_dtype` | Gathered K 量化值 |
| `k_scale` (FP8) | `(total_seq_lens, 4)` | `uint8` | Gathered K 的 scale |
| `k_quant` (MXFP4) | `(total_seq_lens, head_dim//2)` | `uint8` | Gathered MXFP4 K |
| `k_scale` (MXFP4) | `(total_seq_lens, head_dim//32)` | `uint8` (ue8m0) | K 的 ue8m0 scale |
| `logits` | `(num_queries, total_seq_len)` | `f32` | MQA logits 结果 |
| `topk_indices_buffer` | `(max_batch, topk_tokens)` | `int32` | 输出的 top-k 索引 |
| `topk_workspace` | `(1024*1024,)` | `uint8` | Persistent topk 工作区 |
| `slot_mapping` | `(num_tokens,)` | `int64` | token 到缓存 slot 的映射 |

---

## 7. 平台差异

### CUDA (默认路径)

- 使用 `forward_cuda()`，调用 `torch.ops.vllm.sparse_attn_indexer`。
- 依赖 DeepGEMM（`has_deep_gemm()` 检查，未安装时抛出 `RuntimeError`）。
- Decode 阶段支持 `persistent_topk` 优化（当 `topk_tokens in (512, 1024, 2048)`）。
- 完整支持 FP8 和 MXFP4 两条路径。
- K 缓存插入通过 `ops.indexer_k_quant_and_cache` 完成（L167）。

### ROCm (AMD GPU)

- 使用 `forward_hip()`，不通过 DeepGEMM。
- **必须**设置 `VLLM_ROCM_USE_AITER=1` 环境变量以启用 AITER 库。
- 只支持单 FP8 Q tensor（`assert isinstance(q_quant, torch.Tensor)`），**不支持** MXFP4 缓存。
- 调用 `torch.ops.vllm.rocm_aiter_sparse_attn_indexer` 实现。

### XPU (Intel GPU)

- 使用 `forward_xpu()`，直接委托给 `forward_cuda()`。
- 在 prefill 和 decode 阶段使用 `torch.ops.vllm.xpu_fp8_mqa_logits` 和 `torch.ops.vllm.xpu_fp8_paged_mqa_logits`。
- **不支持** MXFP4 Q（`q_scale_slice is not None` 时抛出 `RuntimeError`）。

### 平台差异总结

| 特性 | CUDA | ROCm | XPU |
|------|------|------|-----|
| 核心算子 | `torch.ops.vllm.sparse_attn_indexer` | `rocm_aiter_sparse_attn_indexer` | 同 CUDA |
| DeepGEMM | 必需 | 不依赖 | 不依赖 |
| FP8 支持 | 完整 | 完整 | 完整 |
| MXFP4 支持 | 完整 | 不支持 | 不支持 |
| persistent_topk | 支持 | 不支持 | 不支持 |
| K 缓存插入 | `indexer_k_quant_and_cache` | 待确认 | 待确认 |

---

## 8. MXFP4 vs FP8 路径差异

### 数据格式

| 特性 | FP8 | MXFP4 |
|------|-----|-------|
| 元素类型 | `float8_e4m3fn` | `uint8`（2值压缩为1字节） |
| Scale 类型 | `float32` (per-128-elements) | `uint8` `ue8m0` (per-32-elements) |
| 精度 | 8-bit (约3位尾数) | 4-bit (约2位尾数) |
| 存储压缩比 | 2x (相对 bf16) | 4x (相对 bf16) |
| 硬件需求 | 原生 FP8 支持 (H100+) | 无原生支持，需特殊 dequant |

### 代码差异

| 场景 | FP8 | MXFP4 |
|------|-----|-------|
| Gather workspace values | `(T, head_dim), fp8_dtype` | `(T, head_dim//2), uint8` |
| Gather workspace scales | `(T, 4), uint8` | `(T, head_dim//32), uint8` |
| DeepGEMM 类型转换 | 保持原 dtype | values `-> int8`，scales `-> int32` |
| Q padding pad_value | 默认（fp32 pad） | `pad_value=0` |
| K 缓存插入 | `ops.indexer_k_quant_and_cache` | 不支持（`use_fp4_cache=True` 时断言） |
| q_scale | `None`（已折叠进 weights） | `not None`（独立传递） |

### 选择逻辑

通过 `use_fp4_cache` 布尔标志控制：

```python
if use_fp4_cache:
    assert q_scale is not None, "use_fp4_cache=True requires q_scale"
else:
    assert q_scale is None, "q_scale must be None when use_fp4_cache=False"
```

---

## 9. CUDA Graph 兼容设计

CUDA graph 要求每次推理的形状是固定的，但注意力中的 token 数量是动态的。SparseAttnIndexer 通过以下机制实现兼容：

### 9.1 Workspace 预留

在 profiling/dummy 路径中（L106-141），所有 workspace 被一次性预留：

```python
# Gather workspace (形状由 _gather_workspace_shapes 决定)
workspace_manager.get_simultaneous(values_spec, scales_spec)

# Radix topk workspace (固定大小 1MB)
workspace_manager.get_simultaneous(((1024*1024,), torch.uint8))

# Max logits tensor (固定大小，由环境变量控制)
_ = torch.empty(max_logits_elems, dtype=torch.uint8, ...)
```

这些预留操作确保 CUDA graph 捕获时所有内存分配已被记录，不会在后续推理中引入动态分配。

### 9.2 pack_seq_triton / unpack_seq_triton

decode 阶段通过 padding 将所有序列填充到相同的最大长度：

- **pack**: 将不规则的 Q 序列填充为固定形状 `(batch_size, next_n, ...)`。
- **unpack**: 将 top-k 结果中的 padding token 索引去除，只保留实际 token。

```python
# 填充 (L274-284)
padded_q_quant_decode_tokens = pack_seq_triton(
    q_quant[:num_decode_tokens], decode_lens, pad_value=0
)

# 解填充 (L365-371)
topk_indices = unpack_seq_triton(
    topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
    decode_lens,
)
```

### 9.3 K 的截断 (L159-161)

在进行 speculative decoding 时，`k` 可能被 padding 到 CUDA graph 的 batch size，但 `slot_mapping` 只覆盖实际 token。截断 `k` 防止越界读取：

```python
num_tokens = slot_mapping.shape[0]
if k is not None:
    k = k[:num_tokens]
```

---

## 10. 与注意力模块的关系

### 被 [[DeepseekV4MLAAttention]] 调用

在 `DeepseekV4MLAAttention` 中，`indexer` 属性仅在 C4A 层（`compress_ratio > 1`）非 `None`：

```python
if self.indexer is not None:
    topk_indices = self.indexer(
        None,     # hidden_states (未使用)
        q_quant,  # 来自 FusedIndexerQ 的输出
        None,     # k (已在 indexer 内部处理)
        indexer_weights,
    )
```

调用方式见 [[DeepseekV4MLAAttention|DeepseekV4MLAAttention 第 2.6 节]]。

### 与 [[FusedIndexerQ]] 的配合

`FusedIndexerQ` 负责将 indexer 的 Q 投影经 RoPE 旋转后量化为 FP8/MXFP4，生成 `q_quant` 和（MXFP4 路径的）`q_scale`，这些直接作为 `SparseAttnIndexer` 的输入。

### 与 [[DeepseekV4_KVCache_Ops]] 的关系

- `kv_cache_as_quant_view()` (L61-78) 将 3D KV 缓存视图转换为 DeepGEMM 需要的 4D 视图，涉及 `torch.as_strided` 操作，非数据拷贝。
- `ops.cp_gather_indexer_k_quant_cache` 从分页 KV 缓存中 gather K 并量化。
- `ops.indexer_k_quant_and_cache` 将当前 K 量化并写入缓存。

### 与 [[DeepseekV4Indexer]] 的关系

`DeepseekV4Indexer` 是上层模块，包含完整的 indexer 逻辑：
1. 生成 Q 的 RoPE + 量化（调用 `FusedIndexerQ`）。
2. 执行 compression（调用 `DeepseekCompressor`）。
3. 写入 K 缓存。
4. **调用 `SparseAttnIndexer` 执行 top-k 索引选择**。

在整个 indexer 流程中，`SparseAttnIndexer` 是 **最后一步**，也是 **最核心的计算密集步骤**。

---

## 11. 辅助函数详解

### `_gather_workspace_shapes()` (L40-58)

根据 `use_fp4_cache` 返回不同的 workspace shape/dtype：

```python
def _gather_workspace_shapes(total_seq_lens, head_dim, fp8_dtype, use_fp4_cache):
    if use_fp4_cache:
        # MXFP4: 2值/字节 + ue8m0 scale (每32元素1字节)
        return ((T, head_dim//2), uint8), ((T, head_dim//32), uint8)
    else:
        # FP8: 1值/字节 + fp32 scale (每128元素4字节)
        return ((T, head_dim), fp8_dtype), ((T, 4), uint8)
```

### `kv_cache_as_quant_view()` (L61-78)

将 3D KV 缓存转换为 DeepGEMM 期望的 4D 视图：

```python
def kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache):
    if use_fp4_cache:
        # MXFP4: 创建 strided 4D 视图
        num_blocks, block_size, _ = kv_cache.shape
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1))
    else:
        # FP8: 简单 unsqueeze
        return kv_cache.unsqueeze(-2)
```

### `sparse_attn_indexer_fake()` (L376-394)

虚设实现，直接返回 `topk_indices_buffer`，用于 torch.compile 的 schema 推导。

---

## 12. 常量与环境变量

| 常量/环境变量 | 默认值 | 说明 |
|---------------|--------|------|
| `RADIX_TOPK_WORKSPACE_SIZE` | `1024 * 1024` | Persistent topk 的工作区大小（字节） |
| `MXFP4_BLOCK_SIZE` | `32` | MXFP4 的 block size（每 32 元素一个 scale） |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 环境变量 | 用于预留 peak logits 张量的显存大小（MB） |

---

## 13. 设计要点总结

1. **提取为独立 CustomOp**：将 mqa_logits、paged_mqa_logits、top_k_per_row 等重 kernel 封装为单一自定义算子，便于对不同硬件后端进行针对性优化。

2. **Workspace 管理模式**：通过 `current_workspace_manager().get_simultaneous()` 一次性分配所有临时 workspace，避免动态分配的开销和 CUDA graph 兼容性问题。

3. **两阶段分离**：Prefill 和 Decode 使用不同的注意力计算方式（dense mqa_logits vs paged_mqa_logits）和不同的 top-k 算法（top_k_per_row_prefill vs persistent_topk/top_k_per_row_decode）。

4. **MXFP4 前瞻设计**：虽然当前 ROCm 和 XPU 不支持 MXFP4，但 CUDA 路径已完整实现，为未来更激进的内存压缩铺平了道路。

5. **CUDA Graph 最优兼容**：通过 workspace 预留、pack/unpack padding、tensor 截断等机制，确保该模块在 CUDA graph 捕获后仍能正确处理动态形状。

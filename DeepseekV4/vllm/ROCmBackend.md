# DeepSeek V4 的 ROCm 注意力后端

## 概述

文件 `vllm/models/deepseek_v4/amd/rocm.py` 提供了 DeepSeek V4 稀疏 MLA（多头潜在注意力）的 **AMD ROCm 特定**实现。它是 [[DeepseekV4_KVCache_Ops#NVIDIA 后端|NVIDIA flashmla.py 后端]]的 ROCm 对应实现。

ROCm 无法使用 NVIDIA 的 `flash_mla_with_kvcache` 或 `flash_mla_sparse_fwd` CUDA 内核（这些内核依赖于 Hopper GPU 上的 cuDNN/CUTLASS）。相反，它委托给 `rocm_aiter_mla_sparse` 模块中基于 Triton 的稀疏注意力内核。两者架构上的关键区别在于：ROCm 使用**不规则（类 COO）格式**表示稀疏索引，而 NVIDIA 使用**稠密填充格式**。

---

## 类层次结构

```
AttentionBackend                          (vllm.v1.attention.backend)
  └── FlashMLASparseBackend               (flashmla_sparse.py)
        └── DeepseekV4FlashMLASparseBackend  (nvidia/flashmla.py)
              └── DeepseekV4ROCMAiterMLASparseBackend   [L576-587]

SparseMLAAttentionImpl                    (vllm.v1.attention.backend)
  └── DeepseekV4SparseMLAAttentionImpl    (nvidia/flashmla.py, 抽象类)
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

ROCm 类覆盖了 NVIDIA 版本，以注入不规则格式的元数据并接入 ROCm 特定的注意力内核。它们没有在继承层次中添加新层级，而是在运行时基于 `current_platform.is_rocm()` 作为替代后端被选择。

---

## 稠密格式 vs. 不规则格式

这是 NVIDIA 和 ROCm 后端之间的核心设计区别。

### 稠密格式（NVIDIA）

由 `flash_mla_with_kvcache` 和 `flash_mla_sparse_fwd` 使用：

- 每个 token 都有一个**固定宽度**的二维索引张量，使用 `-1` 哨兵值填充。
- 每行形状为 `(1, TOP_K)` 或 `(1, combined_topk)`。
- 内核使用 `topk_length` 值来知道每行有多少个有效条目。
- 填充保证了 CUDA 内核启动配置的内存对齐访问。

### 不规则格式（ROCm）

ROCm 的 `rocm_sparse_attn_decode` / `rocm_sparse_attn_prefill` 要求：

- 一个**扁平的一维张量**，只包含所有 token 的有效（非填充）索引，连接在一起。
- 一个伴随的 **indptr** 张量，形状为 `(num_tokens + 1,)`，通过类似 CSR 的指针界定每个 token 的范围：`indices[indptr[i]:indptr[i+1]]` 是 token `i` 的有效条目。
- 还带有一个 **lengths** 张量，用于与 NVIDIA 代码路径的兼容性。

不规则格式类似于 CSR（压缩稀疏行）或 COO 编码。它消除了填充 token 带来的内存浪费。ROCm 使用不规则格式，因为其基于 Triton 的稀疏内核（最初为 DeepSeek V3 Aiter 后端设计）操作在此表示上，并且缺少 NVIDIA CUDA 内核所支持的填充索引内核。

### 从稠密构建不规则

辅助函数 `build_ragged_indices_from_dense`（定义在 `rocm_aiter_mla_sparse.py` 中，从两个元数据构建器中调用）执行转换：

1. 将 lengths 张量限制在 `[0, max_width]` 范围内，以防止越界条目。
2. 通过 `cumsum(lengths)` 计算 indptr。
3. 分配一个大小为 `indptr[-1]` 的扁平张量。
4. 启动 `_pack_dense_prefix_to_ragged_kernel` 将有效条目散布到扁平缓冲区中。

---

## 不规则元数据数据类

### `DeepseekV4ROCMAiterMLASparseMetadata` [L451-456]
扩展 `FlashMLASparseMetadata`，为 C128A 解码添加两个不规则字段：

```python
c128a_decode_topk_ragged_indices: torch.Tensor | None  # 扁平一维
c128a_decode_topk_ragged_indptr: torch.Tensor | None    # (max_tokens+1,)
```

### `DeepseekV4ROCMAiterSparseSWAMetadata` [L459-462]
扩展 `DeepseekSparseSWAMetadata`，为解码添加不规则 SWA 索引：

```python
decode_swa_ragged_indices: torch.Tensor | None  # 扁平一维
decode_swa_ragged_indptr: torch.Tensor | None   # (max_tokens+1,)
```

---

## 元数据构建流程

### C128A（压缩率 = 128）

`DeepseekV4ROCMAiterMLASparseMetadataBuilder` [L465-519] 遵循以下流程：

1. **在构造函数时预分配缓冲区**（如果 `compress_ratio == 128`）：形状为 `(max_tokens * c128a_max_compressed,)` 的不规则索引缓冲区和形状为 `(max_tokens + 1,)` 的 indptr 缓冲区。这些是持久化的，以实现 [[V1AttentionBackend#CUDA 图|CUDA 图兼容性]]。

2. **`build()`**：调用父类的 `FlashMLASparseMetadataBuilder.build()`，生成**稠密**的 `c128a_global_decode_topk_indices` 和 `c128a_decode_topk_lens`。

3. **不规则转换**：调用 `build_ragged_indices_from_dense(dense_decode, decode_lens)` 从稠密的父类输出生成不规则索引 + indptr。

4. **复制到图缓冲区**：调用 `_copy_ragged_to_graph_buffers()` 将不规则元数据复制到预分配的持久缓冲区中。

5. 返回一个 `DeepseekV4ROCMAiterMLASparseMetadata`，其中稠密字段（继承）和不规则字段（新增）都已填充。

### SWA（滑动窗口注意力）

`DeepseekV4ROCMAiterSparseSWAMetadataBuilder` [L522-573] 遵循相同的模式：

1. **预分配**大小为 `(max_tokens * window_size,)` 的不规则索引缓冲区和 `(max_tokens + 1,)` 的 indptr 缓冲区。

2. **`build()`**：调用父类的 `DeepseekSparseSWAMetadataBuilder.build()` 生成稠密的 `decode_swa_indices` 和 `decode_swa_lens`。

3. **不规则转换**：将稠密的解码 SWA 索引转换为不规则格式。

4. **复制到图缓冲区**：复制到预分配的持久存储中。

5. 返回填充了不规则字段的 `DeepseekV4ROCMAiterSparseSWAMetadata`。

### C4A（压缩率 = 4）

C4A **不**通过元数据构建器处理其不规则索引。相反，`_forward_decode()` [L700-712] 在每个解码步骤中**内联**调用 `compute_global_topk_ragged_indices_and_indptr`。这是因为 C4A topk 索引每层都不同（它们由 Indexer 在每一步填充），因此无法在元数据构建器中预先计算。

函数 `compute_global_topk_ragged_indices_and_indptr` [L236-277] 是通用 `compute_global_topk_indices_and_lens`（输出稠密）的 AMD 特定替代。它：

1. 通过 `_compute_topk_lens_kernel` 统计每个 token 的有效条目数。
2. 通过 `_build_indptr_from_lengths` 从 lengths 构建 indptr。
3. 通过 `_pack_global_topk_ragged_kernel` 将不规则索引从局部打包为全局（块表查找）。

---

## CUDA 图兼容性策略

ROCm 使用与 NVIDIA 相同的 [[V1AttentionBackend#CUDA 图|CUDA 图兼容性]]方法：预分配持久缓冲区，使得内核参数地址在图重放时保持稳定。

函数 `_copy_ragged_to_graph_buffers` [L427-448] 处理不规则特定的图复制：

```python
def _copy_ragged_to_graph_buffers(
    ragged_indices, ragged_indptr,
    ragged_indices_buffer, ragged_indptr_buffer,
    num_rows, max_entries_per_row,
) -> tuple[Tensor, Tensor]:
```

- 以切片 `[:num_rows + 1]` 的形式复制 indptr，使用 `non_blocking=True`。
- 以切片 `[:nnz]` 的形式复制索引，上限为 `num_rows * max_entries_per_row`，也使用 `non_blocking=True`。
- 返回由稳定缓冲区存储支持的切片视图，使得内核参数地址在图捕获和重放之间保持不变。
- `non_blocking=True` 标志允许 H2D 复制与计算重叠。

---

## 前向传播流程

### `forward_mqa()` [L600-675] -- 入口点

与 NVIDIA 相同的分发结构：

1. 如果 `attn_metadata is None`（虚拟预热运行）：预留工作空间并将输出置零。
2. 拆分为 prefill 和解码段。
3. 对 `q[num_decode_tokens:]` 调用 `_forward_prefill()`。
4. 对 `q[:num_decode_tokens]` 调用 `_forward_decode()`。

### `_forward_decode()` [L677-738]

```
forward_decode
  ├── C4A (compress_ratio==4)
  │     └── compute_global_topk_ragged_indices_and_indptr(layer.topk_indices_buffer)
  │           ├── _compute_topk_lens_kernel (统计每个 token 的有效条目)
  │           ├── _build_indptr_from_lengths (cumsum -> indptr)
  │           └── _pack_global_topk_ragged_kernel (块表查找，扁平输出)
  │
  ├── C128A (compress_ratio==128)
  │     └── 使用从元数据预构建的不规则格式：
  │         attn_metadata.c128a_decode_topk_ragged_indices
  │         attn_metadata.c128a_decode_topk_ragged_indptr
  │
  └── rocm_sparse_attn_decode(q, kv_cache, swa_k_cache, topk_*, swa_*, ...)
        ├── topk_ragged_indices + topk_ragged_indptr
        ├── swa_ragged_indices + swa_ragged_indptr  (来自 SWA 元数据)
        └── 输出：注意力结果
```

与 NVIDIA 解码的关键区别：
- NVIDIA 通过 `extra_indices_in_kvcache` 将 `topk_indices`（稠密）传递给 `flash_mla_with_kvcache`。
- ROCm 将 `topk_ragged_indices` + `topk_ragged_indptr` 作为单独的参数传递给 `rocm_sparse_attn_decode`。
- ROCm 还传递了 `swa_ragged_indices` + `swa_ragged_indptr`（SWA 解码索引的不规则转换），而 NVIDIA 直接传递稠密的 `swa_indices`。

### `_forward_prefill()` [L740-856]

```
forward_prefill (分块，chunk_size=4)
  │
  ├── 对每个 chunk：
  │     ├── 反量化并收集压缩的 K 缓存（如果不是仅 SWA）
  │     │     └── dequantize_and_gather_k_cache（共享算子，与 NVIDIA 相同）
  │     ├── 反量化并收集 SWA K 缓存
  │     │     └── dequantize_and_gather_k_cache（共享算子，与 NVIDIA 相同）
  │     ├── 合并 topk + SWA 索引
  │     │     └── combine_topk_swa_indices（AMD 特定版本，L118-165）
  │     └── rocm_sparse_attn_prefill(q, kv, combined_indices, combined_lens, ...)
  │
  └── 输出：分块注意力结果
```

与 NVIDIA prefill 的关键区别：
- 两者都使用来自公共算子的共享 `dequantize_and_gather_k_cache`。
- 两者都调用 `combine_topk_swa_indices`，但 ROCm 使用其 **AMD 特定**版本（而不是来自 `cache_utils.py` 的共享版本）。
- NVIDIA 使用 `combined_indices.unsqueeze(1)` 调用 `flash_mla_sparse_fwd`；ROCm 调用 `rocm_sparse_attn_prefill`。

---

## 两种 `combine_topk_swa_indices` 变体

### 共享/通用版本（`common/ops/cache_utils.py`，详见 [[DeepseekV4_KVCache_Ops#combine_topk_swa_indices]]）

被 NVIDIA prefill 使用，也可供通用使用。输出**稠密填充**的合并索引。

- 填充对齐到 `_SPARSE_PREFILL_TOPK_ALIGNMENT=128`。
- 内核假定所有 `topk_indices` 条目在 `topk_len` 范围内都是有效的。
- 调用方强制约束：`topk_indices` 在共享级别**不得包含 -1 填充**。

### AMD 特定版本（`amd/rocm.py`，L118-165）

由 ROCm prefill 使用。也输出**稠密填充**的合并索引（prefill 仍使用稠密进行合并），但包含一个额外的 `TOPK_WIDTH` 参数。

**为什么需要 `TOPK_WIDTH`**：ROCm 的 topk 索引可能有一个大于逻辑 `TOP_K` 的物理宽度（`topk_width`）。例如，分配的缓冲区可能按 `max_topk=256` 大小设置，而每个 token 只有 `topk=64` 个有效条目。AMD 内核：

1. 使用 `TOPK_WIDTH`（`topk_indices` 的实际步长/宽度）安全地加载。
2. 通过 `TOP_K` 掩码到 `topk_len` 范围。
3. 过滤无效索引：`(topk_indices >= 0) & (topk_indices < N)` —— 共享版本不重新检查有效性，因为通用调用方确保没有 -1 条目进入合并步骤。

`PADDED_TOP_K` 计算为 `triton.next_power_of_2(topk_width)` 用于 Triton 块大小，而 NVIDIA 版本使用 `next_power_of_2` 作用于 `topk`（而不是 `topk_width`）。

### AMD 特定不规则变体（`amd/rocm.py`，L370-424）

`combine_topk_swa_indices_ragged` 直接输出不规则格式而不是稠密填充。它：

1. 通过 `_compute_combined_lens_kernel` 预计算合并后的 lengths（单独 pass）。
2. 从这些 lengths 构建 indptr。
3. 启动 `_combine_topk_swa_indices_ragged_kernel`，使用三维网格 `(num_reqs, 128 workers, cdiv(topk+window, 128) blocks)` 直接写入不规则扁平缓冲区。

这个不规则变体目前未在主前向传递中使用，但作为一个构件存在，用于未来 ROCm prefill 内核可能直接接受不规则输入的潜在用途。

---

## 与 NVIDIA FlashMLASparseImpl 的关键区别

| 方面 | NVIDIA（`flashmla.py`） | AMD（`rocm.py`） |
|--------|----------------------|-----------------|
| **稀疏索引格式** | 稠密填充的二维张量 | 不规则一维 + indptr（类 CSR） |
| **解码注意力内核** | `flash_mla_with_kvcache`（CUDA） | `rocm_sparse_attn_decode`（Triton） |
| **Prefill 注意力内核** | `flash_mla_sparse_fwd`（CUDA） | `rocm_sparse_attn_prefill`（Triton） |
| **Q 头填充** | 填充到 64 或 128（`get_padded_num_q_heads`） | 无填充（原样返回 `num_heads`） |
| **C4A 解码 topk** | `compute_global_topk_indices_and_lens`（稠密输出） | `compute_global_topk_ragged_indices_and_indptr`（不规则输出） |
| **C128A 不规则构建** | 不需要（内核直接消费稠密） | 通过 `build_ragged_indices_from_dense` 从稠密父类构建 |
| **SWA 不规则构建** | 不需要 | 通过 `build_ragged_indices_from_dense` 从稠密父类构建 |
| **`combine_topk_swa_indices`** | 使用来自 `cache_utils.py` 的共享版本 | 使用带有 `TOPK_WIDTH` 保护的 AMD 特定版本 |
| **Prefill 块大小** | 4（相同常量） | 4（相同常量） |
| **`_forward_decode` 签名** | 接收 `FlashMLASparseMetadata` | 接收 `DeepseekV4ROCMAiterMLASparseMetadata` |
| **反量化/收集 KV** | 相同的共享 `dequantize_and_gather_k_cache` | 相同的共享 `dequantize_and_gather_k_cache` |

---

## 辅助工具

### `_build_indptr_from_lengths` [L40-44]

通过 `cumsum` 将 lengths 张量转换为 CSR 风格的 indptr：

```python
indptr = zeros(len(lengths) + 1)
cumsum(lengths, out=indptr[1:])
```

### `compute_global_topk_ragged_indices_and_indptr` [L236-277]

C4A 解码的三步流程：
1. `_compute_topk_lens_kernel`：统计每个 token 的有效（>=0）条目数，将无效 token 置零。
2. `_build_indptr_from_lengths`：构建不规则 indptr。
3. `_pack_global_topk_ragged_kernel`：通过块表查找将局部索引解析为全局 KV 缓存槽位，并写入扁平不规则缓冲区。

### `_copy_ragged_to_graph_buffers` [L427-448]

使用 `non_blocking=True` 的 H2D 复制，将不规则元数据复制到预分配的持久 CUDA 图缓冲区中。

---

## 相关交叉引用

- [[DeepseekV4_KVCache_Ops]] -- 共享的 K 缓存操作和通用的 `combine_topk_swa_indices`。
- [[V1AttentionBackend]] -- 注意力后端、元数据构建器和 CUDA 图集成的框架。
- [[NvidiaVsAMD]] -- NVIDIA FlashMLA 和 AMD ROCm Aiter 后端之间的架构比较。

下面详细解析 `DeepseekV4SparseMLAAttentionImpl` 及其具体实现 `DeepseekV4FlashMLASparseImpl` 的代码。这两个类共同构成了 **DeepSeek-V4 稀疏 MLA 注意力** 的后端实现，负责与 `FlashMLASparseBackend` 交互，执行实际的注意力计算（包括 prefill 和 decode）。代码采用类方法（classmethod）风格，以便在无实例的情况下被上层调用，同时便于 `torch.compile` 捕获。

---

## 1. 抽象基类 `DeepseekV4SparseMLAAttentionImpl`

```python
class DeepseekV4SparseMLAAttentionImpl(SparseMLAAttentionImpl[FlashMLASparseMetadata]):
```

- 继承自 `SparseMLAAttentionImpl`（泛型参数为 `FlashMLASparseMetadata`），为 DeepSeek-V4 的稀疏 MLA 实现提供抽象骨架。
- 该基类重写了 `forward_mqa` 为 **类方法**（而不是实例方法），因为 V4 的调用方式与 v1 框架不同：由 `DeepseekV4MLAAttention.forward` 直接调用 `impl_cls.forward_mqa`，而不是通过框架的 `AttentionBackend`。这是有意为之的 Liskov 替换原则“破坏”。

### 关键类属性

```python
backend_cls: ClassVar[type[AttentionBackend]]
PREFILL_CHUNK_SIZE: ClassVar[int] = 4
```

- `backend_cls`：关联的 `AttentionBackend` 子类（例如 `DeepseekV4FlashMLASparseBackend`），用于注册和元数据获取。
- `PREFILL_CHUNK_SIZE`：prefill 阶段每次处理的最大序列数（chunk），用于限制临时工作区的大小。

### 抽象方法

```python
@classmethod
@abstractmethod
def forward_mqa(cls, layer, q, kv, positions, output) -> None: ...
```

- 核心前向方法，接收 `DeepseekV4MLAAttention` 实例 `layer` 以及 Q、KV（实际未使用）、positions、output 缓冲区。需要在子类中实现完整的稀疏注意力逻辑。

```python
@classmethod
@abstractmethod
def get_padded_num_q_heads(cls, num_heads: int) -> int:
```

- 返回底层内核要求的 **填充后** 的 Q 头数。例如 FlashMLA 的 FP8 decode 内核仅支持 64 或 128 头，因此对于 `num_heads <= 64` 返回 64，否则返回 128。该值用于上层分配 Q 和输出缓冲区。

---

## 2. `DeepseekV4FlashMLASparseBackend` 后端类

```python
class DeepseekV4FlashMLASparseBackend(FlashMLASparseBackend):
```

- 继承自 vLLM 的 `FlashMLASparseBackend`，为 DeepSeek-V4 定制后端参数。

### 方法详解

`get_supported_kernel_block_sizes() -> [256]`
- 返回支持的 KV 缓存块大小（block size）。FlashMLA 要求块大小为 256。

 `get_name() -> "V4_FLASHMLA_SPARSE"`
- 后端名称，用于日志和选择。

 `get_impl_cls() -> DeepseekV4FlashMLASparseImpl`
- 返回关联的实现类（即稍后定义的 `DeepseekV4FlashMLASparseImpl`）。

 `get_supported_head_sizes() -> [512]`
- 返回支持的 head 维度大小。DeepSeek-V4 的 head_dim 为 512（448 NoPE + 64 RoPE），覆盖了 V3.2 默认的 576。

 `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)`
- 决定 KV 缓存的形状。
  - 若 `cache_dtype_str == "fp8_ds_mla"`（DeepSeek-V4 特有格式），则每个 token 使用 584 字节（448 NoPE + 128 RoPE + 8 字节缩放因子）。返回 `(num_blocks, block_size, 584)`。
  - 否则返回 `(num_blocks, block_size, head_size)`（通用布局）。
- 注意：`head_size` 参数传入的是语义上的 head_dim (512)，但实际缓存大小按 584 字节分配。

---

## 3. `DeepseekV4FlashMLASparseImpl` 实现类

### 类属性

```python
backend_cls = DeepseekV4FlashMLASparseBackend
```

### `get_padded_num_q_heads`

```python
@classmethod
def get_padded_num_q_heads(cls, num_heads: int) -> int:
    if num_heads > 128:
        raise ValueError(...)
    return 64 if num_heads <= 64 else 128
```

- 确保头数在 64 或 128。若模型实际头数超过 128 则不支持（当前 DeepSeek-V4 配置应在此范围内）。

### `forward_mqa` — 主入口

```python
@classmethod
def forward_mqa(cls, layer, q, kv, positions, output):
    assert output.shape == q.shape
    assert output.dtype == q.dtype
```

- `q` 形状为 `[num_tokens, padded_heads, head_dim]`（已经填充到 64 或 128）。
- `output` 形状必须与 `q` 相同，且 dtype 一致（通常是 `bfloat16`）。

#### 获取 forward context 和元数据

```python
forward_context = get_forward_context()
attn_metadata = forward_context.attn_metadata
```

- `attn_metadata` 是 vLLM 在每次前向时设置的全局注意力元数据，是一个字典，key 为层前缀（`layer.prefix`），value 为 `FlashMLASparseMetadata`；另外还有一个 key 为 `layer.swa_cache_layer.prefix` 对应 `DeepseekSparseSWAMetadata`。

#### 处理 dummy run（warmup / profile）

```python
if attn_metadata is None:
    swa_only = layer.compress_ratio <= 1
    N = 0 if swa_only else (layer.max_model_len + layer.compress_ratio - 1) // layer.compress_ratio
    M = N + layer.window_size + layer.max_num_batched_tokens
    current_workspace_manager().get_simultaneous(
        ((cls.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
    )
    output.zero_()
    return
```

- 在 CUDA graph 捕获前的 profile run 中，没有真实 `attn_metadata`。此时需要预分配一个临时工作区（workspace），其形状为 `(PREFILL_CHUNK_SIZE, M, head_dim)`，以便后续 `_forward_prefill` 中的 `dequantize_and_gather_k_cache` 使用。`M` 是压缩缓存长度 + 滑动窗口长度 + 最大 batch tokens 的总和。
- 输出置零，不执行实际注意力。

#### 真实运行：获取元数据

```python
flashmla_metadata = cast(FlashMLASparseMetadata | None, attn_metadata.get(layer.prefix))
swa_metadata = cast(DeepseekSparseSWAMetadata, attn_metadata.get(layer.swa_cache_layer.prefix))
assert swa_metadata is not None
```

- `flashmla_metadata` 仅当 `compress_ratio > 1` 时存在（因为压缩层才有额外的稀疏元数据）。
- `swa_metadata` 始终存在，包含滑动窗口的调度信息、序列长度、token 到请求的映射等。

#### 分离 prefill 和 decode tokens

```python
num_decodes = swa_metadata.num_decodes
num_prefills = swa_metadata.num_prefills
num_decode_tokens = swa_metadata.num_decode_tokens

if num_prefills > 0:
    cls._forward_prefill(
        layer=layer,
        q=q[num_decode_tokens:],
        positions=positions[num_decode_tokens:],
        compressed_k_cache=self_kv_cache,
        swa_k_cache=swa_kv_cache,
        output=output[num_decode_tokens:],
        attn_metadata=flashmla_metadata,
        swa_metadata=swa_metadata,
    )
if num_decodes > 0:
    cls._forward_decode(
        layer=layer,
        q=q[:num_decode_tokens],
        kv_cache=self_kv_cache,
        swa_metadata=swa_metadata,
        attn_metadata=flashmla_metadata,
        swa_only=swa_only,
        output=output[:num_decode_tokens],
    )
```

- `q` 和 `output` 的前 `num_decode_tokens` 是 decode token，后面是 prefill token。分别调用 `_forward_decode` 和 `_forward_prefill`。

---

## 4. `_forward_decode` 方法

处理 decode 阶段（每个 token 逐个生成）的注意力。

### 准备稀疏索引

```python
topk_indices = None
topk_lens = None
if not swa_only:
    assert attn_metadata is not None
    assert swa_metadata.is_valid_token is not None
    block_size = attn_metadata.block_size // layer.compress_ratio
    is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
    if layer.compress_ratio == 4:
        # C4A: local indices per layer (filled by Indexer)
        global_indices, topk_lens = [[DeepseekV4_KVCache_Ops#compute_global_topk_indices_and_lens|compute_global_topk_indices_and_lens]](
            layer.topk_indices_buffer[:num_decode_tokens],
            swa_metadata.token_to_req_indices,
            attn_metadata.block_table[:num_decodes],
            block_size,
            is_valid,
        )
        topk_indices = global_indices.view(num_decode_tokens, 1, -1)
    else:
        # C128A: pre-computed during metadata build.
        topk_indices = attn_metadata.c128a_global_decode_topk_indices
        topk_lens = attn_metadata.c128a_decode_topk_lens
```

- 对于压缩层（`compress_ratio > 1`），需要知道每个 query 应该关注哪些压缩后的 KV 位置（top‑k 索引）。
- **C4A**（ratio=4）情况下，索引由 `Indexer` 动态计算并存储在 `layer.topk_indices_buffer` 中，需要通过 `compute_global_topk_indices_and_lens` 转换为全局索引（考虑序列内的 block table）。输出 `global_indices` 形状 `[num_tokens, topk]`，随后扩展为 `[num_tokens, 1, topk]`（因为 FlashMLA 期望的维度）。
- **C128A**（ratio=128）情况下，索引在元数据构建时已预先计算好，直接使用 `attn_metadata.c128a_global_decode_topk_indices` 和 `c128a_decode_topk_lens`。

### 准备滑动窗口索引

```python
swa_indices = swa_metadata.decode_swa_indices
swa_lens = swa_metadata.decode_swa_lens
```

- 滑动窗口索引，形状 `[num_tokens, window_size]`，表示每个 token 需要关注的 SWA 位置（相对偏移）。

### Q 的维度调整

```python
q = q.unsqueeze(1)   # [b = num_tokens, s = 1, padded_heads, head_dim]
```

- FlashMLA 期望 Q 形状为 `[num_tokens, 1, num_heads, head_dim]`。这里num_tokens定义成batch，1代表的是seqlen_q= 1，可以这样表达的原因如下：
	- KV采用的是page attention方式管理的
	- num_tokens其实总token总长度
	- 每个token对应的KV由indices决定（mla接口也不需要block_table了）
	- 所以，token所属的batch已经不重要了

### 准备 KV 缓存张量

```python
swa_cache = layer.swa_cache_layer.kv_cache.unsqueeze(-2)  # [num_blocks, block_size, 1, head_bytes]
if kv_cache is not None:
    kv_cache = kv_cache.unsqueeze(-2)
```

- 每个缓存形状为 `(num_blocks, block_size, 1, head_bytes)`。`unsqueeze(-2)` 添加一个 KV 头维度（`num_kv_heads=1`）。

### 获取 tile 调度元数据

```python
if layer.compress_ratio <= 1:
    tile_metadata = swa_metadata.tile_sched_swaonly
elif layer.compress_ratio == 4:
    tile_metadata = swa_metadata.tile_sched_c4a
elif layer.compress_ratio == 128:
    tile_metadata = swa_metadata.tile_sched_c128a
else:
    raise ValueError(...)
```

- `tile_sched_*` 是 FlashMLA 内核需要的预计算调度信息（如各个 warp 处理的任务划分），由 `DeepseekSparseSWAMetadataBuilder` 在元数据构建时根据序列长度、稀疏模式提前生成。不同压缩比对应不同的稀疏模式，因此需要不同的调度元数据。

### 调用 FlashMLA 内核

```python
out, _ = flash_mla_with_kvcache(
    q=q,
    k_cache=swa_cache,
    block_table=None,
    head_dim_v=512,
    tile_scheduler_metadata=tile_metadata,
    cache_seqlens=None,
    is_fp8_kvcache=True,
    indices=swa_indices,
    topk_length=swa_lens,
    softmax_scale=layer.scale,
    attn_sink=layer.attn_sink,
    extra_k_cache=kv_cache if not swa_only else None,
    extra_indices_in_kvcache=topk_indices,
    extra_topk_length=topk_lens,
    out=output.unsqueeze(1),
)
```

- `flash_mla_with_kvcache` 是 FlashMLA 提供的融合内核，支持：
  - 主 KV 缓存（`k_cache`）用于滑动窗口（SWA）。
  - 可选额外 KV 缓存（`extra_k_cache`）用于压缩区域（通过 `extra_indices_in_kvcache` 索引）。
  - `indices` 和 `topk_length` 控制滑动窗口内的索引。
  - `attn_sink` 作为偏置加到 logits 上。
  - 输出直接写入 `output`（已 unsqueeze 为 `[num_tokens, 1, padded_heads, head_dim]`）。
- `head_dim_v=512` 指定 V 的维度（V 没有压缩，保持全维度）。

---

## 5. `_forward_prefill` 方法

处理 prefill 阶段（一次性处理整个输入序列）的稀疏注意力。prefill 时，需要将压缩 KV 和 SWA KV 从缓存中 **收集（gather）** 到一个连续的临时缓冲区 `kv` 中，然后执行一次稀疏注意力。

### 参数说明

- `q`, `positions`, `output`：仅包含 prefill token 部分（已经通过切片分离）。
- `compressed_k_cache`：压缩层的 KV 缓存（`compress_ratio>1` 时存在）。
- `swa_k_cache`：滑动窗口 KV 缓存。
- `attn_metadata`：稀疏元数据（`compress_ratio>1` 时存在）。
- `swa_metadata`：SWA 元数据（始终存在）。

### 准备基本变量

```python
swa_only = attn_metadata is None
num_prefills = swa_metadata.num_prefills
num_prefill_tokens = swa_metadata.num_prefill_tokens
...
seq_lens = swa_metadata.prefill_seq_lens          # 每个 prefill 序列的实际长度
gather_lens = swa_metadata.prefill_gather_lens    # 每个 prefill 需要 gather 的 SWA 长度
```

- `num_prefills` 是序列数（不是 token 数）。
- `prefill_seq_lens` 长度等于 `num_prefills`。

### 获取 token 偏移量

```python
prefill_token_base = query_start_loc_cpu[num_decodes]
```

- `query_start_loc_cpu` 是所有请求（decode + prefill）的 token 起始索引，`num_decodes` 之前的索引属于 decode token，prefill token 从 `prefill_token_base` 开始。

### 确定稀疏索引

```python
if not swa_only:
    if layer.compress_ratio == 4:
        topk_indices = layer.topk_indices_buffer[num_decode_tokens:][:num_prefill_tokens]
    else: # C128A
        topk_indices = attn_metadata.c128a_prefill_topk_indices
    top_k = topk_indices.shape[-1]
    N = (layer.max_model_len + layer.compress_ratio - 1) // layer.compress_ratio
else:
    topk_indices = layer.topk_indices_buffer[num_decode_tokens:][:num_prefill_tokens]
    top_k = 0
    N = 0
```

- 对于压缩层，`N` 是压缩后的最大长度（即整个序列长度除以压缩比后上取整）。用于临时缓冲区 `kv` 的第二维大小。

### 临时缓冲区大小

```python
M = N + layer.window_size + layer.max_num_batched_tokens
chunk_size_const = cls.PREFILL_CHUNK_SIZE   # 4
num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const
workspace_manager = current_workspace_manager()
kv = workspace_manager.get_simultaneous(
    ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
)[0]
```

- `M` 是临时缓冲区中每个序列的最大 token 数（压缩区域 + SWA 区域 + 新增 token）。
- 每次处理最多 4 个 prefill 序列（chunk），循环处理所有序列。
- `kv` 形状 `(chunk_size, M, head_dim)`，用于存放 gather 后的 KV 数据。

### 循环处理每个 chunk

```python
for chunk_idx in range(num_chunks):
    chunk_start = chunk_idx * chunk_size_const
    chunk_end = min(chunk_start + chunk_size_const, num_prefills)
    chunk_size = chunk_end - chunk_start
```

- 提取当前 chunk 对应的序列范围。

#### 1. 收集压缩 KV（若存在）

```python
if not swa_only:
    [[DeepseekV4_KVCache_Ops#dequantize_and_gather_k_cache|dequantize_and_gather_k_cache]](
        block_table=block_table[chunk_start:chunk_end],
        block_size=attn_metadata.block_size // layer.compress_ratio,
        offset=0,
    )
```

- `dequantize_and_gather_k_cache` 从压缩缓存中读取每个序列的 KV（经过 FP8 量化），反量化为 BF16，并写入 `kv` 的前 `chunk_size` 行，从 `offset=0` 开始。
- 该操作会读取每个序列的全部压缩 KV（长度 `seq_len / compress_ratio`），并按块索引拼接。

#### 2. 收集 SWA KV

```python
dequantize_and_gather_k_cache(
    kv[:chunk_size],
    swa_k_cache,
    seq_lens=seq_lens[chunk_start:chunk_end],
    gather_lens=gather_lens[chunk_start:chunk_end],
    block_table=swa_block_table[chunk_start:chunk_end],
    block_size=swa_metadata.block_size,
    offset=N,
)
```

- 将 SWA 区域（滑动窗口）的 KV 写入 `kv`，偏移 `offset=N`（即放在压缩区域的后面）。
- `gather_lens` 控制每个序列实际需要收集的 SWA 长度（通常等于 `window_size` 加上某些额外长度）。

#### 3. 组合索引（topk + SWA）

```python
combined_indices, combined_lens = [[DeepseekV4_KVCache_Ops#combine_topk_swa_indices|combine_topk_swa_indices]](
    topk_indices[query_start:query_end],
    query_start_loc[num_decodes + chunk_start : num_decodes + chunk_end + 1],
    seq_lens[chunk_start:chunk_end],
    gather_lens[chunk_start:chunk_end],
    layer.window_size,
    layer.compress_ratio,
    top_k,
    M,
    N,
)
```

- `combine_topk_swa_indices` 将稀疏 top‑k 索引（在压缩区域中的位置）和 SWA 索引（在滑动窗口中的位置）合并为一个总的索引列表，并映射到临时缓冲区 `kv` 中的实际行号。
- 输出 `combined_indices` 形状 `[total_tokens, topk+swa_len]`，`combined_lens` 为每个 token 实际有效的索引数量。

#### 4. 执行稀疏注意力

```python
flash_mla_sparse_fwd(
    q=q[query_start:query_end],
    kv=kv.view(-1, 1, q.shape[-1]),
    indices=combined_indices.unsqueeze(1),
    sm_scale=layer.scale,
    attn_sink=layer.attn_sink,
    topk_length=combined_lens,
    out=output[query_start:query_end],
)
```

- `flash_mla_sparse_fwd` 是另一个 FlashMLA 内核，接受 Q、连续的 KV 缓冲区（`kv` 已 gather 好）、以及合并后的索引，执行稀疏 attention 并写入 `output`。
- `kv` 需要 reshape 为 `[-1, 1, head_dim]`（因为 `num_kv_heads=1`）。
- `indices` 需要增加一个维度 `unsqueeze(1)` 变成 `[total_tokens, 1, total_indices]`。
- 这里，其实是**变长的attention**，与dense区别在于，每个q对应的KV由indices决定，q所属的batch已经不重要了，故接口不需要cu_seqlens_q积分和

---

## 6. 量化与并行策略总结

### 量化

- **KV 缓存**：全部为 FP8 格式（`is_fp8_kvcache=True`）。压缩缓存使用 `"fp8_ds_mla"` 布局（每 token 584 字节）。`dequantize_and_gather_k_cache` 负责在 prefill 时反量化为 BF16。
- **Q 和输出**：保持 BF16。
- **注意力计算**：内核内部使用 FP8 乘法累加，最终输出 BF16。

### 并行策略

- **tile 调度元数据**：提前规划每个 warp 的工作负载，实现高效的块级并行。
- **prefill 分块处理**：每次处理最多 4 个序列，避免临时缓冲区过大，同时利用 workspace 复用。
- **decode 阶段**：每个 token 独立处理，FlashMLA 内核内部已实现高效的 warp 级并行。
- **与 wrapper 的多流并行**：`DeepseekV4MultiHeadLatentAttentionWrapper` 中的 `attn_gemm_parallel_execute` 与 `attention_impl` 中的任务级并行与本类无关；本类只负责注意力核心计算。

---

## 7. 设计亮点

1. **灵活的稀疏支持**：通过 `compress_ratio` 区分 SWA only (1)、C4A (4)、C128A (128)，分别对应不同的稀疏模式和索引生成方式。
2. **高效 prefill**：通过 `dequantize_and_gather_k_cache` 批量收集压缩 KV 和 SWA KV，合并索引后一次调用 `flash_mla_sparse_fwd`，避免逐 token 循环。
3. **decode 融合**：`flash_mla_with_kvcache` 直接读取缓存的 KV，支持同时使用滑动窗口和额外稀疏索引，无需中间 gather。
4. **元数据驱动**：所有调度信息（tile 调度、索引）都在 vLLM 的元数据构建阶段提前计算，减少运行时开销，并支持 CUDA graph 捕获。
5. **平台特定优化**：通过 `get_padded_num_q_heads` 和 `get_kv_cache_shape` 适配 FlashMLA 的具体要求（64/128 头、584B 缓存行）。

总之，`DeepseekV4FlashMLASparseImpl` 是 DeepSeek-V4 注意力性能的关键，它紧密耦合了 FlashMLA 内核，并利用 vLLM 的元数据基础设施，实现了大规模稀疏注意力的高效推理。

---

## 8. 源码行号标注

> 以下行号基于 `vllm/models/deepseek_v4/nvidia/flashmla.py`。

### 8.1 `forward_mqa` 中区分 decode/prefill (L183-208)

```python
# L184-186: 从 swa_metadata 中提取统计量
num_decodes = swa_metadata.num_decodes
num_prefills = swa_metadata.num_prefills
num_decode_tokens = swa_metadata.num_decode_tokens

# L188-198: prefill 分支 —— 处理完整序列
if num_prefills > 0:
    cls._forward_prefill(
        layer=layer,
        q=q[num_decode_tokens:],              # 从 decode token 之后开始
        positions=positions[num_decode_tokens:],
        compressed_k_cache=self_kv_cache,
        swa_k_cache=swa_kv_cache,
        output=output[num_decode_tokens:],
        attn_metadata=flashmla_metadata,
        swa_metadata=swa_metadata,
    )
# L199-208: decode 分支 —— 逐个 token 生成
if num_decodes > 0:
    cls._forward_decode(
        layer=layer,
        q=q[:num_decode_tokens],              # 取前 num_decode_tokens
        kv_cache=self_kv_cache,
        swa_metadata=swa_metadata,
        attn_metadata=flashmla_metadata,
        swa_only=swa_only,
        output=output[:num_decode_tokens],
    )
```

- `q` 和 `output` 的第一维是 `[num_decode_tokens + num_prefill_tokens]`，直接按索引切片。
- prefill 和 decode 是两个独立路径，分别调用不同的 FlashMLA 内核。

### 8.2 `_forward_decode` 中 `flash_mla_with_kvcache` 调用 (L286-302)

```python
out, _ = flash_mla_with_kvcache(
    q=q,                                    # [num_decode_tokens, 1, padded_heads, head_dim]
    k_cache=swa_cache,                      # 滑动窗口 KV 缓存 [num_blocks, block_size, 1, head_bytes]
    block_table=None,                        # 不使用 block_table（索引驱动）
    head_dim_v=512,                          # V 的完整维度（未压缩）
    tile_scheduler_metadata=tile_metadata,   # 预计算的 warp 调度元数据
    cache_seqlens=None,                      # 不使用 seqlens（索引驱动）
    is_fp8_kvcache=True,                     # KV 缓存为 FP8 格式
    indices=swa_indices,                     # 滑动窗口索引 [num_tokens, window_size]
    topk_length=swa_lens,                    # 每个 token 实际 SWA 长度
    softmax_scale=layer.scale,               # attention scale 因子
    attn_sink=layer.attn_sink,               # attention sink 偏置
    extra_k_cache=kv_cache if not swa_only else None,  # 压缩区域 KV 缓存（可选）
    extra_indices_in_kvcache=topk_indices,    # 压缩区域 topk 索引
    extra_topk_length=topk_lens,              # 每个 token 的 topk 数量
    out=output.unsqueeze(1),                  # 输出 [num_tokens, 1, padded_heads, head_dim]
)
```

参数含义详解：
- **`k_cache`**：主 KV 缓存，存储滑动窗口区域。形状 `[num_blocks, block_size, 1, head_bytes]`，其中 `head_bytes` 是实际存储字节数（含量化 scale）。
- **`head_dim_v=512`**：V 的维度。DeepSeek-V4 中 V 没有压缩，保持完整的 512 维（448 NoPE + 64 RoPE）。
- **`tile_scheduler_metadata`**：FlashMLA 内核需要的 tile 调度元数据，由 `DeepseekSparseSWAMetadataBuilder` 在元数据构建时根据序列长度、稀疏模式预计算。不同压缩比（1/4/128）使用不同的调度元数据（`tile_sched_swaonly` / `tile_sched_c4a` / `tile_sched_c128a`）。
- **`indices` / `topk_length`**：控制滑动窗口内的索引。`indices` 形状 `[num_tokens, window_size]`，表示每个 token 需要关注的 SWA 位置；`topk_length` 是每个 token 实际有效的索引数（变长支持）。
- **`extra_k_cache` / `extra_indices_in_kvcache` / `extra_topk_length`**：可选参数，当 `compress_ratio > 1` 时传入压缩区域的 KV 缓存和 top-k 索引，实现 SWA + 压缩区域的联合注意力。
- **`attn_sink`**：作为偏置加到 logits 上，用于 attention sink 技术。
- **`is_fp8_kvcache=True`**：KV 缓存以 FP8 格式存储，内核内部使用 FP8 乘法累加。

### 8.3 `_forward_prefill` 中 chunk 循环处理 (L365-424)

```python
# L365: 按 chunk 循环
for chunk_idx in range(num_chunks):
    chunk_start = chunk_idx * chunk_size_const
    chunk_end = min(chunk_start + chunk_size_const, num_prefills)
    chunk_size = chunk_end - chunk_start

    # L369-381: 收集压缩区域的 KV（若存在压缩层）
    if not swa_only:
        dequantize_and_gather_k_cache(
            kv[:chunk_size],
            compressed_k_cache,
            seq_lens=seq_lens[chunk_start:chunk_end] // layer.compress_ratio,
            gather_lens=None,
            block_table=block_table[chunk_start:chunk_end],
            block_size=attn_metadata.block_size // layer.compress_ratio,
            offset=0,                           # 从 kv 缓冲区头开始写入
        )

    # L384-393: 收集滑动窗口区域的 KV
    dequantize_and_gather_k_cache(
        kv[:chunk_size],
        swa_k_cache,
        seq_lens=seq_lens[chunk_start:chunk_end],
        gather_lens=gather_lens[chunk_start:chunk_end],
        block_table=swa_block_table[chunk_start:chunk_end],
        block_size=swa_metadata.block_size,
        offset=N,                               # 在压缩区域之后写入
    )

    # L403-415: 合并 topk 索引和 SWA 索引
    combined_indices, combined_lens = [[DeepseekV4_KVCache_Ops#combine_topk_swa_indices|combine_topk_swa_indices]](
        topk_indices[query_start:query_end],
        query_start_loc[num_decodes + chunk_start : num_decodes + chunk_end + 1],
        seq_lens[chunk_start:chunk_end],
        gather_lens[chunk_start:chunk_end],
        layer.window_size, layer.compress_ratio, top_k, M, N,
    )

    # L416-424: 执行稀疏注意力
    flash_mla_sparse_fwd(
        q=q[query_start:query_end],
        kv=kv.view(-1, 1, q.shape[-1]),         # reshape 为 [total_kv, 1, head_dim]
        indices=combined_indices.unsqueeze(1),  # [total_tokens, 1, total_indices]
        sm_scale=layer.scale,
        attn_sink=layer.attn_sink,
        topk_length=combined_lens,
        out=output[query_start:query_end],
    )
```

- 每次处理最多 `PREFILL_CHUNK_SIZE=4` 个序列，避免临时缓冲区过大。
- 使用 `dequantize_and_gather_k_cache` 将 FP8 缓存的 KV 反量化为 BF16 并收集到连续缓冲区 `kv`。
- `flash_mla_sparse_fwd` 是 prefill 专用内核，与 decode 的 `flash_mla_with_kvcache` 不同：后者直接从缓存读取，前者从已 gather 的连续缓冲区读取。

---

## 9. 设计决策思考

### 9.1 为什么 `padded_heads` 是 64 或 128？

```python
# L119-127 (flashmla.py)
@classmethod
def get_padded_num_q_heads(cls, num_heads: int) -> int:
    if num_heads > 128:
        raise ValueError(...)
    return 64 if num_heads <= 64 else 128
```

根本原因：**FlashMLA 的 FP8 decode 内核（`flash_mla_with_kvcache`）的 warp 级并行调度要求 Q head 数必须是 64 或 128。**

具体来说：
1. **硬件约束**：FlashMLA 内核在解码阶段使用固定的 warp 调度策略，每个 warp 处理固定数量的 head。当 `h_q=64` 时，warp 调度为 `(4 warps × 16 heads)`；当 `h_q=128` 时，调度为 `(8 warps × 16 heads)`。其他 head 数无法映射到内核的固定调度模板。
2. **填充策略**：若模型实际 head 数 `<= 64`（例如 DeepSeek-V4 的标准配置 `n_local_heads=16`），填充到 64；若在 `(64, 128]` 之间，填充到 128。填充的 head 位置值为 0，不影响 softmax 输出。
3. **与 wrapper 的协作**：`DeepseekV4MultiHeadLatentAttentionWrapper` 在分配 Q 和输出缓冲区时使用 `self.padded_heads`（L255: `self.padded_heads = self.mla_attn.padded_heads`），形状为 `[num_tokens, padded_heads, head_dim]`。计算完成后通过 `o = o_padded[:, :self.n_local_heads, :]` 切片取出有效 head（L302）。
4. **性能 vs 显存权衡**：填充少量 head（64-16=48 个无效 head）带来的额外计算量很小，但使得内核可以使用最优的 warp 调度，大幅提升吞吐。这是一种"以少量冗余计算换取更大并行度"的经典优化策略。

---

## 10. 交叉引用

- [[DeepseekV4MultiHeadLatentAttentionWrapper]]：`DeepseekV4FlashMLASparseImpl` 的上层调用者。Wrapper 负责 MLA 的预处理（QR/KV 投影、RoPE、KV 缓存插入）和输出投影（逆 RoPE + WoA + WoB），而 `DeepseekV4FlashMLASparseImpl` 只负责注意力核心计算（`flash_mla_with_kvcache` / `flash_mla_sparse_fwd`）。wrapper 在 `attention_impl` 方法（L409）中调用 `self.mla_attn(q, kv, positions, output=out)`，其中 `mla_attn` 是 `DeepseekV4MLAAttention` 实例（L235），其 `forward` 方法（L727）直接调用 `self.impl_cls.forward_mqa(self, q, kv, positions, output)`，即 `DeepseekV4FlashMLASparseImpl.forward_mqa`。
- [[DeepseekV4MLAAttention]]：`DeepseekV4MLAAttention` 是连接 Wrapper 和 Impl 的中间层，负责初始化 `impl_cls`、管理 KV 缓存和 `padded_heads`。
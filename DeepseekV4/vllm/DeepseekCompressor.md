这段代码实现了 **DeepSeek V4 模型中用于压缩 KV 缓存和注意力分数的核心模块**，被称为 `DeepseekCompressor`。它通过高效的压缩、归一化、旋转位置编码（RoPE）和存储操作，显著减少显存占用和计算量。以下是对代码的逐步详细解析。

---

## 1. 整体背景

DeepSeek V4 采用了一种**压缩注意力机制**：将原始的高维 KV 状态和注意力分数（score）通过线性变换压缩成低维表示，然后存储到专用的状态缓存中。后续的解压缩和注意力计算利用这些压缩状态，从而降低内存带宽和存储成本。

代码中涉及的主要组件：
- `CompressorBackend`：自定义的注意力后端，用于管理压缩状态的元数据和内存布局。
- `CompressorMetadataBuilder`：构建压缩注意力所需的元数据（如块表、令牌到请求的映射）。
- `CompressorStateCache`：实际存储压缩状态（KV 部分 + 分数部分）的张量模块。
- `DeepseekCompressor`：核心模块，执行压缩、归一化、RoPE 和存储的全流程。

---

## 2. `CompressorBackend` 类

```python
class CompressorBackend(AttentionBackend):
    ...
```

这是一个自定义的注意力后端，专门用于处理压缩状态的注意力计算。

- `get_name()`：返回后端标识 `"CompressorBackend"`。
- `get_supported_kernel_block_sizes()`：支持任意块大小（`MultipleOf(1)`）。
- `get_supported_head_sizes()`：支持头维度 512 和 1024（与 DeepSeek V4 配置匹配）。
- `get_builder_cls()`：返回 `CompressorMetadataBuilder` 类，用于构建元数据。
- `get_kv_cache_shape()`：压缩状态缓存形状为 `(num_blocks, block_size, head_size)`，其中 `num_kv_heads` 固定为 1（单头存储压缩状态）。
- `get_kv_cache_stride_order()`：定义内存布局的步长顺序。

---

## 3. `CompressorMetadata` 与 `CompressorMetadataBuilder`

### `CompressorMetadata` 数据类

存储压缩注意力步骤所需的元数据：
- `block_table`：形状 `[num_blocks, block_size]`，记录每个块中每个位置的物理索引。
- `slot_mapping`：将每个逻辑 token 位置映射到缓存中的实际槽位。
- `block_size`：块大小。
- `token_to_req_indices`：可选的映射，从 token 索引到所属请求（request）的索引，用于后续按请求处理。

### `CompressorMetadataBuilder`

构建 `CompressorMetadata` 对象。关键点：
- 构造函数中获取 `SlidingWindowMLASpec` 或 `MLAAttentionSpec` 以确定 `block_size`。
- 分配 `token_to_req_indices` 张量（预分配最大批大小）。
- `build()` 方法：根据 `CommonAttentionMetadata` 计算每个 token 所属的请求索引（通过 `repeat_interleave`），然后封装返回 `CompressorMetadata`。

---

## 4. `CompressorStateCache` 类

继承自 `torch.nn.Module` 和 `AttentionLayerBase`，负责管理压缩状态的实际存储。

以下是对 `CompressorStateCache` 类的详细代码解析。该类继承自 `torch.nn.Module` 和 `AttentionLayerBase`，用于在 vLLM 框架中管理**压缩器状态缓存**，它与传统的 KV 缓存协同工作，但针对不同的压缩比（`compress_ratio`）采用不同的存储块结构。

---

### 类定义与初始化参数

```python
class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
```

- **`state_dim`**：压缩器状态向量的维度，通常与模型 hidden size 或 KV head dim 相关。
- **`dtype`**：缓存中存储的数据类型，代码中强制要求为 `torch.float32`（见后文断言）。
- **`compress_ratio`**：压缩比，允许取值 `4` 或 `128`，决定了状态的汇聚方式以及内部块大小。
- **`prefix`**：当前缓存层在模型中的唯一标识符（如 `"layer_0"`），用于在全局编译配置中注册自身。

---

### 成员变量初始化

```python
self.state_dim = state_dim
self.dtype = dtype
self.prefix = prefix
self.kv_cache = torch.tensor([])   # 初始为空 tensor，后续会动态扩展
```

- `kv_cache` 是实际存储压缩状态的张量，开始时为空，将在推理过程中按需分配和填充。

---

### 全局编译上下文注册

```python
compilation_config = get_current_vllm_config().compilation_config
if prefix in compilation_config.static_forward_context:
    raise ValueError(f"Duplicate layer name: {prefix}")
compilation_config.static_forward_context[prefix] = self
```

- `get_current_vllm_config()` 获取当前 vLLM 运行时的全局配置对象。
- `compilation_config.static_forward_context` 是一个字典，用于在静态编译（如 `torch.compile`）期间记录各个压缩器缓存层，以便在模型前向传播时能够根据 `prefix` 找到对应的缓存对象。
- 如果同一个 `prefix` 已经被注册过，则抛出异常，避免冲突。

---

### 类型与压缩比检查

```python
assert self.dtype == torch.float32
assert compress_ratio in [4, 128]
```

- 当前实现仅支持 `float32` 精度（可能与某些融合 kernel 或数值稳定性要求有关）。
- 压缩比只允许 `4` 或 `128`，对应两种不同的压缩策略（例如 C4 和 C128）。

---

### 滑动窗口大小计算

```python
coff = 1 + (compress_ratio == 4)
self.sliding_window = coff * compress_ratio
```

- 当 `compress_ratio == 4` 时，布尔表达式 `(compress_ratio == 4)` 为 `True`（Python 中 `True` 等于 `1`），因此 `coff = 2`，`sliding_window = 2 * 4 = 8`。
- 当 `compress_ratio == 128` 时，`coff = 1 + 0 = 1`，`sliding_window = 128`。
- **含义**：滑动窗口大小控制着压缩器能回顾的历史状态步数。对于压缩比 `4`，窗口较小（8）；对于压缩比 `128`，窗口较大（128）。这可能是为了匹配底层 kernel 对状态对齐的要求。

---

### 块大小 (`block_size`) 的确定

```python
if compress_ratio == 4:
    self.block_size = 4
elif compress_ratio == 128:
    self.block_size = 8
else:
    raise ValueError(f"Invalid compress ratio: {compress_ratio}")
```

注释中解释了块大小的约束来源：

> Block size is constrained by tensor sharing between compressor states and KV blocks. Since compressor states share the same physical tensor as KV blocks, they must use the same page size.

- **压缩器状态与 KV 块共享同一个物理张量**，因此必须使用相同的分页大小（page size）。
- KV 块的形状是 `[256//4, head_dim] = [64, 584]`，这决定了压缩器的块大小：
  - **C4 压缩器**（`compress_ratio=4`）：每个块形状为 `[4, 2*512*2*4]`，因此 `block_size = 4`
  - **C128 压缩器**（`compress_ratio=128`）：每个块形状为 `[8, 512*2*4]`，因此 `block_size = 8`
- 这里的 `head_dim=584` 是特定模型（如某些 Llama 变体）的数值，注释中的 `2*512*2*4` 等是内部 kernel 的布局参数，整体保证了不同压缩比下物理内存的连续性和对齐。

---

### 总结

`CompressorStateCache` 是一个用于高效管理可变长序列压缩状态的模块，它：

- 通过 `prefix` 在全局编译上下文中注册自己，便于静态图捕获。
- 只支持 `float32` 和两种压缩比（4 或 128）。
- 根据压缩比自动计算滑动窗口大小和内存块大小，以满足与 KV 缓存共享物理内存时的对齐要求。
- `kv_cache` 初始为空，后续会在推理时根据实际序列长度和压缩策略进行动态扩展。

该设计是 vLLM 中长序列高效推理（如 StreamingLLM 或基于压缩的注意力机制）的关键组成部分。

---

## 5. `DeepseekCompressor` 核心类

这是整个压缩流程的入口和协调者。

### 5.1 初始化 (`__init__`)

接收的关键参数：
- `compress_ratio`：压缩比（4 或 128）。
- `hidden_size`：模型隐藏维度。
- `head_dim`：注意力头的维度（512 或 128）。
- `rotate`：是否应用旋转。
- `use_fp4_cache`：是否使用 FP4 量化缓存（仅用于 head_dim=128 的索引器路径）。

**组件创建**：

1. **可学习参数 `ape`**（Adaptive Position Encoding）：  
   形状 `(compress_ratio, coff * head_dim)`，数据类型 `float32`，不可训练。用于添加到压缩状态中，提供位置信息。

2. **`fused_wkv_wgate`**：  
   合并的列并行线性层（`MergedColumnParallelLinear`），将输入 `hidden_size` 映射到两个输出：`kv` 和 `score`，每个维度为 `coff * head_dim`。这里禁用张量并行（`disable_tp=True`），可能因为压缩操作本身已分片。

3. **`norm`**：RMSNorm 层，作用于 `head_dim` 维度。

4. **`state_cache`**：`CompressorStateCache` 实例，管理压缩状态的存储。

5. **静态前向上下文引用**：将自身注册到 `static_forward_context` 中，以便在 `forward` 时快速获取 KV 缓存。

6. **量化相关参数**（根据 `head_dim` 设置）：
   - `head_dim == 512`：不使用 FP4 缓存；`_quant_block=64`，`_token_stride = nope_head_dim + rope_head_dim*2`，`_scale_dim = nope_head_dim//64+1`。
   - `head_dim == 128`：若用 FP4 缓存，`_quant_block=32`（`MXFP4_BLOCK_SIZE`），`_token_stride=head_dim//2`，`_scale_dim=head_dim//MXFP4_BLOCK_SIZE`；否则 `_quant_block=128`，`_token_stride=head_dim`，`_scale_dim=4`。

### 5.2 `forward` 方法详解

输入：
- `kv_score`：形状 `[num_tokens, 2 * coff * head_dim]`，是原始压缩输入（由之前的线性层产生），包含 KV 部分和分数部分。
- `positions`：每个 token 的位置索引。
- `rotary_emb`：旋转位置编码模块，提供 `cos_sin_cache`。

**执行步骤**：

#### Step 1: 拆分 kv 和 score
```python
kv, score = kv_score.split([self.coff * self.head_dim, ...], dim=-1)
```
两个张量形状均为 `[num_tokens, coff * head_dim]`，类型为 bfloat16，之后会被转换为 float32 存储。

#### Step 2: 获取压缩注意力元数据
```python
attn_metadata = get_forward_context().attn_metadata
if not isinstance(attn_metadata, dict):
    return
state_metadata = cast(CompressorMetadata, attn_metadata[self.state_cache.prefix])
```
通过前向上下文获取当前步骤的注意力元数据。如果没有（如 profiling 阶段），则直接返回。从字典中取出对应压缩器的元数据。

#### Step 3: 提取元数据字段
- `token_to_req_indices`：token → 请求索引。
- `slot_mapping`：token → 缓存槽位。
- `block_table`：物理块表。
- `block_size`：块大小。

#### Step 4: 获取状态缓存张量
```python
state_cache = self.state_cache.kv_cache   # shape: [num_blocks, block_size, 2 * state_width]
state_width = state_cache.shape[-1] // 2   # 一半用于 kv_state，一半用于 score_state
```
`state_cache` 是预先分配的物理张量，每个块包含 `block_size` 个压缩状态向量，每个向量包含 KV 和分数两部分，各占 `state_width` 维度。

#### Step 5: 存储原始 KV 和分数（未压缩）
调用 `save_partial_states` 函数（来自 `vllm.models.deepseek_v4.common.ops.save_partial_states`）：
```python
save_partial_states(
    kv=kv, score=score, ape=self.ape, positions=positions,
    state_cache=state_cache, slot_mapping=slot_mapping, block_size=block_size,
    state_width=state_width, compress_ratio=self.compress_ratio, ...
)
```
该函数执行：
- 将 `kv` 和 `score` 分别与 `ape` 的对应部分相加（根据 `positions` 索引取 `ape[position % compress_ratio]`）。
- 根据 `slot_mapping` 将结果写入 `state_cache` 的对应槽位。

#### Step 6: 获取 KV 缓存和元数据（用于后续写入）
```python
k_cache_metadata = attn_metadata[self.k_cache_prefix]
kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache
```
这里 `k_cache_prefix` 通常指向 `CompressorStateCache` 的另一个实例（可能用于最终存储压缩后的 KV），而当前压缩器的状态缓存是中间存储。

#### Step 7: 选择压缩内核函数
```python
if current_platform.is_cuda():
    if self.head_dim == 512:
        from .nvidia.ops.sparse_attn_compress_cutedsl import compress_norm_rope_store_cutedsl
        compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
    else:
        compress_norm_rope_store_fn = compress_norm_rope_store_triton
else:
    compress_norm_rope_store_fn = compress_norm_rope_store_triton
```
- NVIDIA GPU 且 `head_dim==512`：使用 `cutedsl` 优化内核（cuDNN 加速的稀疏注意力压缩）。
- 其他情况（包括 head_dim=128 或非 NVIDIA 平台）：使用 Triton 实现的内核。

#### Step 8: 调用压缩内核
```python
compress_norm_rope_store_fn(
    state_cache=state_cache,
    num_actual=num_actual,
    token_to_req_indices=token_to_req_indices,
    positions=positions,
    slot_mapping=slot_mapping,
    block_table=block_table,
    block_size=block_size,
    state_width=state_width,
    cos_sin_cache=cos_sin_cache,
    kv_cache=kv_cache,
    k_cache_metadata=k_cache_metadata,
    ...
    rms_norm_weight=self.norm.weight,
    rms_norm_eps=self.rms_norm_eps,
    quant_block=self._quant_block,
    token_stride=self._token_stride,
    scale_dim=self._scale_dim,
)
```
这个融合内核执行以下操作（按顺序）：
1. **从 `state_cache` 中读取**之前存储的原始 KV 和分数（每个 token 对应一个压缩组，如 4 或 128 个 token）。
2. **压缩**：将组内的多个状态通过加权融合为单个状态（具体算法由内核实现）。
3. **RMSNorm**：对压缩后的向量进行归一化。
4. **RoPE**：对向量的最后 `rope_head_dim` 维应用旋转位置编码（使用 `cos_sin_cache`，以组起始位置为基准）。
5. **量化**（可选）：如果启用 FP4 缓存，将结果量化为 FP4 格式；否则保持为 FP8 或 bfloat16。
6. **写入 `kv_cache`**：将处理后的状态存储到最终的 KV 缓存中（按 `block_table` 和 `slot_mapping` 索引）。

---

## 6. 关键设计思想

### 6.1 两级缓存
- **状态缓存（`state_cache`）**：存储未压缩的原始 KV 和分数（float32），用于后续压缩时跨 token 聚合。
- **KV 缓存（`kv_cache`）**：存储最终压缩后的值，供注意力计算使用。类型可能是量化格式（FP4/FP8）或低精度浮点数。

### 6.2 压缩比与重叠模式
- 压缩比 4 时，`coff=2` 表示每个压缩块包含 4 个 token，但为了重叠滑动窗口，实际使用 2 倍的头部维度。
- 压缩比 128 时为单重模式。

### 6.3 融合内核优势
- 将压缩、归一化、RoPE、量化、写入缓存全部融合到一个 GPU 内核中，减少内存读写和内核启动开销。
- 针对不同硬件（NVIDIA cutedsl、Triton）提供优化路径。

### 6.4 元数据驱动的异步处理
- 通过 `token_to_req_indices` 和 `block_table`，内核可以按请求（request）并行处理，并高效地写入块状缓存。

---

## 7. 与 vLLM 框架的集成

- `DeepseekCompressor` 作为一个 `nn.Module`，插入到模型层中，通常位于注意力模块之前。
- 通过 `CompressorBackend` 和自定义的注意力元数据，使 vLLM 的调度器能够正确管理压缩状态的分配和复用。
- 利用 `static_forward_context` 在编译时捕获 KV 缓存引用，支持 `torch.compile` 的静态图优化。

---

## 总结

这段代码展示了 **DeepSeek V4 模型中高效率的压缩注意力实现**，通过融合内核、两级缓存和硬件特定优化，大幅减少了 KV 缓存的内存占用和带宽消耗。整个设计体现了以下几个核心原则：

- **数据局部性**：压缩操作使用邻近 token 的状态，适合向量化。
- **融合执行**：减少内核启动和中间数据传输。
- **可扩展性**：支持不同压缩比、头维度，以及多种后端（CUDA cutedsl / Triton，ROCm Triton）。
- **与 vLLM 深度集成**：复用 vLLM 的 KV 缓存管理、注意力元数据传递和编译优化框架。

---

## 8. 源码行号标注

> 以下行号基于 `vllm/models/deepseek_v4/compressor.py`。

### 8.1 关键方法行号汇总

| 类 / 方法 | 文件行号 | 说明 |
|-----------|---------|------|
| `CompressorBackend` | L37-74 | 注意力后端，管理压缩状态元数据和内存布局 |
| `CompressorMetadata` | L78-83 | 元数据 data class |
| `CompressorMetadataBuilder.__init__` | L89-99 | 构造函数，获取 block_size |
| `CompressorMetadataBuilder.build` | L101-118 | 构建元数据，计算 token_to_req_indices |
| `CompressorStateCache.__init__` | L121-155 | 初始化状态缓存，注册到 static_forward_context |
| `CompressorStateCache.get_kv_cache_spec` | L157-165 | 返回缓存规格（alignment=576） |
| `DeepseekCompressor.__init__` | L184-268 | 初始化压缩器组件 |
| `DeepseekCompressor.forward` | L270-380 | 前向传播主流程 |

### 8.2 `forward` 方法 8 个步骤的行号标注

```python
# L270-380: DeepseekCompressor.forward

# Step 1 (L280-282): 拆分 kv 和 score
kv, score = kv_score.split([self.coff * self.head_dim, ...], dim=-1)

# Step 2 (L285-291): 获取压缩注意力元数据
attn_metadata = get_forward_context().attn_metadata
if not isinstance(attn_metadata, dict):
    return
state_metadata = cast(CompressorMetadata, attn_metadata[self.state_cache.prefix])

# Step 3 (L292-296): 提取元数据字段
token_to_req_indices = state_metadata.token_to_req_indices
slot_mapping = state_metadata.slot_mapping
num_actual = slot_mapping.shape[0]
block_table = state_metadata.block_table
block_size = state_metadata.block_size

# Step 4 (L298-301): 获取状态缓存张量
state_cache = self.state_cache.kv_cache  # [num_blocks, block_size, 2 * state_width]
state_width = state_cache.shape[-1] // 2

# Step 5 (L314-325): 存储原始 KV 和分数（save_partial_states）
save_partial_states(kv=kv, score=score, ape=self.ape, positions=positions,
                    state_cache=state_cache, slot_mapping=slot_mapping,
                    block_size=block_size, state_width=state_width,
                    compress_ratio=self.compress_ratio, ...)

# Step 6 (L334-336): 获取 KV 缓存和元数据
k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache

# Step 7 (L338-355): 选择压缩内核函数
if current_platform.is_cuda():
    if self.head_dim == 512:
        compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
    else:
        compress_norm_rope_store_fn = compress_norm_rope_store_triton
else:
    compress_norm_rope_store_fn = compress_norm_rope_store_triton

# Step 8 (L357-380): 调用融合压缩内核
compress_norm_rope_store_fn(state_cache=state_cache, ...,
                            rms_norm_weight=self.norm.weight,
                            rms_norm_eps=self.rms_norm_eps,
                            quant_block=self._quant_block,
                            token_stride=self._token_stride,
                            scale_dim=self._scale_dim, ...)
```

---

## 9. 两级缓存（state_cache vs kv_cache）Shape/Dtype 表

| 属性 | 缓存名称 | Shape | Dtype | 存储内容 |
|------|---------|-------|-------|---------|
| **状态缓存** | `state_cache` (即 `self.state_cache.kv_cache`) | `[num_blocks, block_size, 2 * state_width]` | `float32` | 前半: 原始 KV 状态 (未压缩)；后半: 原始 Score 状态 (未压缩) |
| **KV 缓存** | `kv_cache` (即 `self._static_forward_context[self.k_cache_prefix].kv_cache`) | 由 `CompressorStateCache.get_kv_cache_spec` 决定 (alignment=576) | `uint8` (量化后) | 压缩、归一化、RoPE、量化后的 KV，供注意力计算使用 |

其中：
- `state_width = state_cache.shape[-1] // 2`，即状态缓存的一半宽度。
- `state_cache` 的 `block_size` 在 `CompressorStateCache` 中确定：C4 为 4，C128 为 8。
- `kv_cache` 的布局由 FlashMLA 的 `fp8_ds_mla` 格式决定，每 token 584 字节（448 NoPE + 128 RoPE + 8 字节缩放因子）。
- `state_cache` 存储浮点格式（`float32`）以避免量化误差在压缩聚合时累积；`kv_cache` 使用量化格式（FP8/MXFP4）以节省显存。

---

## 10. 设计决策思考

### 10.1 为什么使用 float32 存储状态缓存？

`CompressorStateCache` 中强制要求 `self.dtype == torch.float32`（L139）。原因如下：

1. **压缩聚合的数值精度要求**：`state_cache` 存储的是**压缩前的原始 KV/Score 状态**。每个压缩组（C4 包含 4 个 token，C128 包含 128 个 token）需要在 `compress_norm_rope_store` 融合内核中进行跨 token 的加权聚合。如果使用 bfloat16 或 float16，聚合过程中的累加误差会显著放大。float32 提供足够的精度保证压缩后的状态质量。
2. **累积过程中误差不可接受**：压缩操作本质是一个组内约简（reduction），其数值精度直接决定了后续注意力计算的质量。低精度累加可能导致信息丢失，尤其是在 C128（128 个 token 压缩为 1 个）这种高压缩比场景下。
3. **存储不是瓶颈**：`state_cache` 的存储量远小于 `kv_cache`（因为 state_cache 只存在于压缩阶段，不需要长期保留），使用 float32 带来的显存开销是可以接受的。
4. **与内核实现的匹配**：融合内核 `compress_norm_rope_store_cutedsl` 和 `compress_norm_rope_store_triton` 的输入预期为 float32，这与其内部使用 float32 算术单元的设计一致。

### 10.2 量化锚点的物理含义

```python
# L249-268 (compressor.py)
if self.head_dim == 512:
    self._quant_block = 64                    # (1)
    self._token_stride = self.nope_head_dim + self.rope_head_dim * 2  # (2)
    self._scale_dim = self.nope_head_dim // 64 + 1  # (3)
elif self.head_dim == 128:
    if use_fp4_cache:
        self._quant_block = MXFP4_BLOCK_SIZE  # 32
        self._token_stride = self.head_dim // 2
        self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
    else:
        self._quant_block = 128
        self._token_stride = self.head_dim
        self._scale_dim = 4
```

**三个量化锚点的物理含义：**

1. **`_quant_block`（量化块大小）**：FP8/MXFP4 量化时每多少个元素共享一个 scale 因子。`head_dim=512` 时为 64（即每 64 个元素一个 scale）；`head_dim=128` 时为 128（FP8）或 32（MXFP4）。这个值直接决定了 scale 因子的数量和量化粒度：_quant_block 越小，量化粒度越细，精度越高。

2. **`_token_stride`（token 步长）**：在 FP8 量化后的缓存中，每个 token 的 KV 数据占据的连续元素数量。对于 `head_dim=512`：`_token_stride = nope_head_dim + rope_head_dim * 2 = 448 + 128 = 576`。注意 RoPE 部分占 2 倍的 rope_head_dim（因为 RoPE 部分存储为 bf16，而非 fp8）。对于 `head_dim=128`：FP8 时 stride=128，MXFP4 时 stride=64（因为是 4-bit 量化，每个元素占半字节）。这个值决定了内核在遍历 token 时的内存访问步长。

3. **`_scale_dim`（scale 维度）**：每个 token 的 scale 因子总个数。对于 `head_dim=512`：`nope_head_dim // 64 + 1 = 7 + 1 = 8`（7 个实际 scale + 1 个填充字节）。对于 `head_dim=128`：FP8 时为 4（128/128*4=4 字节），MXFP4 时为 4（128/32=4）。这个值决定了在缓存中为 scale 预留的空间大小。

**这些锚点的作用**：它们定义了融合压缩内核在将压缩后的状态写入 `kv_cache` 时的量化参数和内存布局。内核需要知道每个 token 的数据如何排列、scale 因子放在哪里、每个量化块有多大，才能正确执行 FP8/MXFP4 量化并写入缓存。

---

## 11. 交叉引用

- [[DeepseekV4MultiHeadLatentAttentionWrapper]]：`DeepseekCompressor` 的调用者。Wrapper 在 `attention_impl` 方法（`attention.py` L409-495）中，当 `compress_ratio > 1` 时调用 `compressor(kv_score, positions, self.rotary_emb)`（L462/L483）。该调用通过 `maybe_execute_in_parallel` 与 `wq_b_kv_insert` 并行执行，隐藏访存延迟。Wrapper 的 `attn_gemm_parallel_execute` 方法（L349-407）还负责计算 `kv_score`（即 `compressor.fused_wkv_wgate` 的输入）。
- [[DeepseekV4FlashMLASparseImpl]]：`DeepseekCompressor` 的输出消费者。压缩后的 KV 被写入 `kv_cache`，然后在 `_forward_decode` 和 `_forward_prefill` 中通过 `flash_mla_with_kvcache` 和 `flash_mla_sparse_fwd` 读取使用。
- [[DeepseekV4Attention]]：管理 `DeepseekCompressor` 的创建（`attention.py` L268-278），将 `k_cache_prefix` 指向 `self.mla_attn.prefix`。
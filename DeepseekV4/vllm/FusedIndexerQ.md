# FusedIndexerQ -- 稀疏 Indexer 的融合 RoPE + 量化 Q

**文件：** `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`（Triton 备选）
**所属模块族：** [[common_ops]] 跨平台融合操作算子（Triton 实现）；另有 `nvidia/ops/fused_indexer_q_cutedsl.py`（CuteDSL 实现，仅 SM100）

融合内核，对 indexer 的 Q 投影应用**正向 GPT-J 交错 RoPE**，量化结果，并将 Q 缩放因子折叠到 indexer 权重中。支持两种量化路径：**FP8**（默认）和 **MXFP4**（`use_fp4=True`）。

**与 [[FusedInvRopeFP8Quant]] 的关系：** 这两个是注意力流水线两端的 RoPE + 量化融合对。`FusedIndexerQ` 处理 **Q 侧**（注意力之前，应用正向 RoPE）。`FusedInvRopeFP8Quant` 处理 **O 侧**（注意力之后，应用逆 RoPE）。两者都使用 GPT-J 交错 RoPE、块缩放量化和 Triton 后端，但在缩放因子管理策略上有所不同。

## 1. 输入和输出

### 输入

| 张量 | 形状 | dtype | 描述 |
|--------|-------|-------|-------------|
| `positions` | `(T,)` | int64 | 用于 RoPE cos/sin 查找的 token 位置。 |
| `index_q` | `(T, H, head_dim)` | bf16 | 来自 indexer 的 Q 投影。`head_dim = 128`。 |
| `index_q_cos_sin_cache` | `(max_pos, rope_dim)` | bf16/f32 | 预计算的 RoPE cos/sin。对于 H128 indexer 头，`rope_dim = 64`。 |
| `index_weights` | `(T, H)` | bf16 | 来自稀疏 indexer 的每个 (token, head) 标量权重。 |

### 参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `index_weights_softmax_scale` | float | Softmax 温度缩放因子。 |
| `index_weights_head_scale` | float | 每头缩放因子（通常从权重加载）。 |
| `use_fp4` | bool | 为 True 时选择 MXFP4 路径（默认：False = FP8）。 |

### 输出

| 路径 | 输出 | 形状 | dtype | 描述 |
|------|--------|-------|-------|-------------|
| FP8 | `q_fp8` | `(T, H, head_dim)` | float8_e4m3fn | 量化后的 Q，**没有伴随的缩放张量**。缩放因子被折叠到 `weights_out` 中。 |
| FP8 | `weights_out` | `(T, H)` | float32 | `weights * q_scale * softmax_scale * head_scale` |
| MXFP4 | `q_packed` | `(T, H, head_dim // 2)` | uint8 | 每字节 2 个 E2M1 半字节（打包值）。 |
| MXFP4 | `q_scale` | `(T, H, num_blocks)` | uint8 (ue8m0) | 每块仅指数缩放因子。`num_blocks = head_dim // 32`。 |
| MXFP4 | `weights_out` | `(T, H)` | float32 | `weights * softmax_scale * head_scale`（不折叠 q_scale）。 |

## 2. 权重折叠设计原理

两条路径之间的核心设计区别：

### FP8 路径（默认）

```
weights_out = index_weights * q_scale * softmax_scale * head_scale
```

- Q 使用**单个每 token 每头标量**缩放因子。
- 该标量被折叠到 `weights_out` 中，因此下游 FP8 logits 内核无需单独加载和乘以它。
- 不为 `q_fp8` 生成伴随的缩放张量。
- 下游 `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` 内核使用预先缩放的 `weights` 来内联应用每 token Q 缩放。

### MXFP4 路径

```
weights_out = index_weights * softmax_scale * head_scale
```

- Q 使用**每块**（32 元素）缩放因子，这些缩放因子与打包值一起存在于一个单独的张量中。
- 这些每块缩放因子**无法**折叠到单个每 token 权重标量中。
- `q_scale` 张量单独输出（ue8m0 格式）。
- 下游 MXFP4 logits 内核通过同时读取 `q_packed` 和 `q_scale` 来去量化 Q。
- `weights_out` 仅携带 softmax 和头缩放因子。

### 为什么不全折叠？

| 方面 | FP8 折叠 | MXFP4 不折叠 |
|--------|----------|---------------|
| 缩放因子粒度 | 每个 (t, h) 1 个标量 | 每 32 元素 1 个缩放因子（每头 `num_blocks` 个） |
| 折叠可行？ | 是——单个标量轻松折叠到权重中 | 否——`num_blocks` 个缩放因子无法合并到单个权重中 |
| 额外张量 | 无 | 需要 `q_scale` 输出张量 |
| 下游内核形状 | 直接使用预先缩放的权重 | 必须在去量化期间加载并应用每块缩放因子 |

## 3. RoPE 算法（GPT-J 交错，正向）

应用于每头的**最后** `rope_dim` 个元素。前导的 `nope_dim = head_dim - rope_dim` 元素保持不变。

### 正向旋转

```
对于每个 i in [0, rope_dim/2):
    even_idx = nope_dim + 2*i        # rope 区域中的偶数索引
    odd_idx  = nope_dim + 2*i + 1    # rope 区域中的奇数索引

    x_even = q[even_idx]
    x_odd  = q[odd_idx]

    r_even = x_even * cos[i] - x_odd * sin[i]    # 正向旋转（偶数）
    r_odd  = x_odd * cos[i] + x_even * sin[i]    # 正向旋转（奇数）
```

与 [[FusedInvRopeFP8Quant]] 对比，其中逆旋转在偶数分支上使用 `+ partner*sin`。

### 用于数值一致性的 BFloat16 往返

nope 和 rope 两半在 absmax 之前都遵循以下序列：

```
fp32 -> bf16 -> fp32
```

在内核中显式执行（FP8 内核的第 123-124 行，MXFP4 内核的第 259-260 行）。注释指出：*"与 K 侧压缩器内核相同的模式"*（[[DeepseekCompressor]]）。这确保了融合内核与非融合参考（后者在 RoPE 和量化之间会自然通过 bf16 写回进行舍入）之间的数值一致性。

在 FP8 内核中：
```python
r_even = r_even.to(tl.bfloat16).to(tl.float32)
r_odd  = r_odd.to(tl.bfloat16).to(tl.float32)
```

### Cos/Sin 查找

```python
cos = cos_sin_cache[pos, 0:HALF_ROT_DIM]
sin = cos_sin_cache[pos, HALF_ROT_DIM:ROT_DIM]
```

缓存连续存储 cos 和 sin：`[cos_0..cos_{H-1}, sin_0..sin_{H-1}]`。

## 4. FP8 量化路径

```
缩放因子计算：
    amax = max(|nope_values|, |rotated_even|, |rotated_odd|)
    q_scale = round_nearest(amax, tol=1e-4) / 448.0    # div_rn
    q_scale = 2^ceil(log2(q_scale))                    # 向上舍入到 2 的幂

量化：
    q_fp8[t, h, d] = clamp(q_bf16 / q_scale, -448, 447) 转换为 float8_e4m3fn
```

- 整个头共享一个单一的标量缩放因子。
- `448.0` = `float8_e4m3fn` 中的最大可表示值。
- 缩放因子向上舍入到 2 的幂（确保不会溢出）。

## 5. MXFP4 量化路径

### MXFP4 格式

| 组件 | 比特数 | 描述 |
|-----------|------|-------------|
| 元素 | E2M1（4 位） | 2 个指数位，1 个尾数位，带符号。范围：[-6, +6]。 |
| 块大小 | 32 元素 | 每块 1 个缩放因子。 |
| 打包 | 每字节 2 个半字节 | 偶数索引值在低半字节，奇数索引值在高半字节。 |
| 缩放因子 | ue8m0（8 位） | 无符号仅指数：`scale = 2^(ue8m0 - 127)`。 |

### 块量化（`_quantize_mxfp4_pair`）

内核以 16 对为一组处理 32 元素块：

```python
# 每次调用处理 16 个偶数 + 16 个奇数值（= 32 元素 = 1 个 MXFP4 块）
amax = max(|x_lo|, |x_hi|)
amax = max(amax, 6.0 * 2^-126)        # 来自 DeepSeek 参考内核的下限

log2_ratio = ceil(log2(amax / 6.0))    # 6.0 = E2M1 的最大值
log2_ratio = clamp(log2_ratio, -127, 127)
scale = 2^log2_ratio
ue8m0 = uint8(log2_ratio + 127)        # 偏置 = 127

# 通过内联 PTX 进行 FP4 量化：
#   cvt.rn.satfinite.e2m1x2.f32 tmp, x_hi, x_lo;
packed = inline_asm_elementwise(...)  # 返回 uint8
```

PTX 内联函数 `cvt.rn.satfinite.e2m1x2.f32` 将两个 f32 值转换为一对 E2M1 值，打包到一个字节中。

### MXFP4 RoPE 块循环

在 MXFP4 路径中，RoPE 区域在 `NUM_ROPE_BLOCKS` 次迭代中处理，每次处理 `MXFP4_BLOCK_SIZE = 32` 个元素：

```python
for b in static_range(NUM_ROPE_BLOCKS):
    # 为该块加载 16 对 cos/sin
    pair_off = b * 16 + tid  # tid = [0, 16)

    # 将 GPT-J RoPE 应用于块的 16 对
    r_even = x_even * cos - x_odd * sin
    r_odd  = x_odd * cos + x_even * sin

    # bf16 往返（数值一致性）
    r_even = r_even.to(bf16).to(fp32)
    r_odd  = r_odd.to(bf16).to(fp32)

    # 量化对并存储打包结果 + ue8m0 缩放因子
    packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
    store(out_base + rope_byte_off + half_off, packed)
    store(scale_base + NUM_NOPE_BLOCKS + b, ue8m0)
```

## 6. 代码结构：CuteDSL vs Triton 备选

公共入口点 `fused_indexer_q_rope_quant` 根据硬件可用性进行分派：

```
fused_indexer_q_rope_quant()
    |
    ├── has_cutedsl() == True（SM100 CUDA）
    │   └── vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py
    │       ├── fused_indexer_q_rope_quant_mxfp4_cutedsl()
    │       └── fused_indexer_q_rope_quant_fp8_cutedsl()
    │
    └── has_cutedsl() == False（SM90 或非 NVIDIA）
        └── vllm/models/deepseek_v4/common/ops/fused_indexer_q.py（本文件）
            ├── _fused_indexer_q_rope_quant_kernel    （Triton，FP8 路径）
            └── _fused_indexer_q_rope_mxfp4_kernel     （Triton，MXFP4 路径）
```

### Triton 内核详情

两个 Triton 内核都使用 `(num_tokens, num_index_q_heads)` 的 **1D 启动网格**——每个 (token, head) 对一个程序，**每个程序 1 个 warp**。

| 方面 | FP8 内核 | MXFP4 内核 |
|--------|------------|--------------|
| Triton 函数 | `_fused_indexer_q_rope_quant_kernel` | `_fused_indexer_q_rope_mxfp4_kernel` |
| 输出值 | `index_q_fp8`（float8_e4m3fn） | `index_q_packed`（uint8）+ `index_q_scale`（uint8 ue8m0） |
| 缩放因子折叠 | 是，折叠到 `weights_out` 中 | 否，单独的 `q_scale` 输出 |
| 输入布局 | 直接 `index_q`（不转置） | 相同 |
| NoPE 块 | 单个 `tl.load` | 循环遍历 `NUM_NOPE_BLOCKS` |
| RoPE 块 | 单次加载 + 旋转 | 循环遍历 `NUM_ROPE_BLOCKS` |
| 内联 PTX | 无 | `cvt.rn.satfinite.e2m1x2.f32` 用于 FP4 打包 |

### 返回格式

```python
# FP8 路径
return index_q_fp8, weights_out
# index_q_fp8: (T, H, head_dim) float8_e4m3fn

# MXFP4 路径
return (index_q_packed, index_q_scale.view(int32).squeeze(-1)), weights_out
# index_q_packed: (T, H, head_dim//2) uint8
# index_q_scale:  (T, H) int32（squeeze 后；每个 int32 打包 4 个 ue8m0 字节）
```

MXFP4 缩放因子重新解释为 int32 并从 `(T, H, 1)` squeeze 到 `(T, H)`，以匹配 DeepGEMM 期望的 `q_sf` 秩（预填充时为 2-D，reshape 后的解码时为 3-D）。

## 交叉引用

- [[common_ops]] — 同目录的跨平台算子总览
- [[FusedInvRopeFP8Quant]] — O 侧逆 RoPE + 量化的对应核（同属 `common/ops/`）
- [[DeepseekV4_KVCache_Ops#FusedCompressQuantCache]] — 同目录下使用相同 bf16 往返模式的另一融合算子
- [[DeepseekV4Indexer]] — 消费此内核输出的 indexer 流水线
- [[DeepseekV4MultiHeadLatentAttentionWrapper]] — 在辅助注意力流上调用此内核的 MLA 包装器
- [[DeepseekCompressor]] — 另一个使用相同 bf16 往返模式以实现数值一致性的融合内核
- `fused_indexer_q_cutedsl.py` — 相同操作的 CuTeDSL 实现（仅 SM100），见 [[DequantGatherKCache#相关笔记]]

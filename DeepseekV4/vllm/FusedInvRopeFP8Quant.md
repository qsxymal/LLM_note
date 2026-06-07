# FusedInvRopeFP8Quant -- 注意力输出的融合逆 RoPE + FP8 量化

**文件：** `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`
**所属模块族：** [[common_ops]] 跨平台融合操作算子

融合内核，对注意力输出 `o`（bf16）应用**逆 GPT-J 交错 RoPE**，然后进行**块缩放 FP8 量化**。结果直接馈送到 MLA 中 V 贡献项的 `fp8_einsum`。

**与 [[FusedIndexerQ]] 的关系：** 这两个内核处理注意力流水线两端的 RoPE + 量化融合。`FusedInvRopeFP8Quant` 处理**注意力输出**（注意力之后，在 O 侧），应用*逆* RoPE 从旋转位置编码转换回绝对位置编码。`FusedIndexerQ` 处理 **Q 投影**（注意力之前，在 indexer 侧），在量化之前应用*正向* RoPE。两者都使用 GPT-J 交错 RoPE 和块缩放量化，但在缩放因子管理策略（块缩放 vs. 每 token 标量）和输出转置方面有所不同。

## 1. 为什么需要融合

如果没有融合，注意力输出将需要：
1. 从 HBM 读取 `o`，应用逆 RoPE，写回 HBM
2. 再次从 HBM 读取 `o`，计算每块的 absmax，量化为 FP8，写入

融合消除了 `o` 的中间 HBM 往返，将所有计算和临时旋转值保留在寄存器中。

## 2. 输入和输出

### 输入

| 张量 | 形状 | dtype | 描述 |
|--------|-------|-------|-------------|
| `o` | `(T, num_heads, head_dim)` | bf16 | 逆 RoPE 之前的注意力输出。`head_dim = nope_dim + rope_dim = 448 + 64 = 512`。 |
| `positions` | `(T,)` | int64 | 用于索引 `cos_sin_cache` 的 token 位置。 |
| `cos_sin_cache` | `(max_pos, rope_dim)` | float32 | 预计算的 cos/sin 值。`rope_dim = 64`。 |

### 输出

| 张量 | 形状 | dtype | 描述 |
|--------|-------|-------|-------------|
| `o_fp8` | `(n_groups, T, d)` | float8_e4m3fn | 转置为组主序，用于 `fp8_einsum`。`d = heads_per_group * head_dim`。 |
| `o_scale` | 因 SM 架构而异 | fp32 或 int32 | 参见下面的缩放格式。 |

### 缩放格式

| 平台 | dtype | 形状（每分组） | 格式 |
|----------|-------|-------------------|--------|
| SM90（`TMA_ALIGNED_SCALES=False`） | fp32 | `(T, num_scale_blocks)` | 每块的 fp32 缩放因子。 |
| SM100（`TMA_ALIGNED_SCALES=True`） | int32 | `(T, packed_sf_k)` | 每个 int32 打包 4 个 UE8M0 指数。`packed_sf_k = ceil(num_scale_blocks / 4)`。 |

缩放张量**预先转换**为 `fp8_einsum` 期望的布局，因此下游内核可以跳过 `transform_sf_into_required_layout`。

## 3. 内核启动配置

```
网格: (tma_aligned_T, n_groups * heads_per_group)
```

- 每个 `(token, head)` 对对应一个 Triton 程序。
- `tma_aligned_T` = `get_tma_aligned_size(num_tokens, 4)` — 填充到 TMA 对齐边界。
- **每个程序 1 个 warp / 1 个阶段**：每个程序处理单个头的 `head_dim=512` 个值。

## 4. 逆 RoPE（GPT-J 交错）

内核将逆 RoPE 应用于每个头的**最后 `rope_dim=64` 个元素**。该算法从旋转后的表示中重建*未旋转*的表示。

### 量化块边界

`QUANT_GROUP_SIZE = 128`，`CHUNKS_PER_HEAD = 512 / 128 = 4`。最后一个量化块（块索引 3，元素 `[384, 512)`）跨越了 nope（元素 `[0, 448)`）和 rope（元素 `[448, 512)`）之间的边界：

```
|                head_dim = 512                          |
|  块 0 (128) | 块 1 (128) | 块 2 (128) | 块 3 (128)           |
|  nope=448                       |<--- nope ---->|<--- rope=64 ---->|
                                                  ^  rope_abs_start = 384 + 64
```

`rope_abs_start = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START = 3*128 + 64 = 448`。

其中 `ROPE_START = nope_dim % quant_group_size = 448 % 128 = 64`。

### 旋转算法

```python
# 使用 XOR 与 1 加载伙伴元素（GPT-J 交错配对）
x_partner = o[offsets ^ 1]    # 偶数 <-> 奇数伙伴

# 从缓存中加载此位置的 cos/sin
cos_v = cos_sin_cache[pos, cs_idx]
sin_v = cos_sin_cache[pos, HALF_ROPE + cs_idx]

# 逆旋转（注意：x_add = o*cos + partner*sin 而不是 o*cos - partner*sin）
x_add = x * cos_v + x_partner * sin_v
x_sub = x * cos_v - x_partner * sin_v

# 交错：偶数索引得到 x_add，奇数索引得到 x_sub
rotated = x_add if is_even else x_sub
```

与正向 RoPE 的关键区别在于符号模式：`+ partner*sin`（逆） vs `- partner*sin`（正向）。参见 [[FusedIndexerQ]] 了解正向版本。

## 5. 块缩放 FP8 量化

### 缩放因子计算

1. 将绝对值重塑为 `(CHUNKS_PER_HEAD, QUANT_GROUP_SIZE)` = `(4, 128)`。
2. 计算每块的 `block_absmax = max(|x|)`，限制为 `eps=1e-10`。
3. `scale_raw = block_absmax / fp8_max`，其中 `fp8_max = 448.0`（float8_e4m3fn 的最大值）。
4. **将缩放因子向上舍入到下一个 2 的幂**：`scale = 2^ceil(log2(scale_raw))`。
5. 量化：`x_quant = clamp(x / scale, -fp8_max, fp8_max)` 转换为 float8_e4m3fn。

### 输出转置

输出张量 `o_fp8` 的形状为 `(n_groups, T, d)`，步长为 `(d, T*d, 1)`。这是 `fp8_einsum` 期望的组主序布局。

## 6. 自定义算子注册

```python
direct_register_custom_op(
    op_name="fused_inv_rope_fp8_quant_kernel",
    op_func=_fused_inv_rope_fp8_quant_kernel_impl,
    fake_impl=_fused_inv_rope_fp8_quant_kernel_fake,
)
```

该操作注册为 PyTorch 自定义算子，以便 TorchInductor 看到一个不透明的边界（作为 [vllm issue #41106](https://github.com/vllm-project/vllm/issues/41106) 的解决方案）。`fake_impl` 在 inductor 的追踪阶段返回正确形状的空张量，而不执行 Triton 内核。

在 CUDA（而非 ROCm/XPU）上传递 `launch_pdl=False` 参数以禁用持久分布式启动。

## 7. 数值细节

- 输入 `o` 以 fp32 加载（通过 `.to(tl.float32)`）用于旋转计算以保持精度。
- 缩放因子舍入 `ceil(log2(...))` 确保量化值始终适合 FP8 范围而不会溢出。
- `eps = 1e-10` 防止退化块除零。
- `(offsets ^ 1)` 技巧通过翻转最低位将每个元素与其伙伴配对，这适用于偶数/奇数对旋转的 GPT-J 交错布局。

## 交叉引用

- [[common_ops]] — 同目录的跨平台算子总览
- [[DeepseekV4_KVCache_Ops]] — 同目录的 KV Cache 量化/压缩算子族（共享 UE8M0 量化算法）
- [[FusedIndexerQ]] — Q 侧正向 RoPE + 量化的对应核（同属 `common/ops/`）
- [[DeepseekV4MLAAttention]] — 产生 `o` 张量的注意力机制
- [[SparseAttnCompress]] — KV 侧的压缩器使用类似的 FP8 量化模式（`nvidia/ops/` CuteDSL 实现）
- [[DequantGatherKCache]] — 另一融合量化 + 内存访问内核（`nvidia/ops/` CuteDSL 实现）

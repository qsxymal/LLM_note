# MoE 激活与权重量化详解

完整覆盖 MegaMoE 和 FusedMoE 两条路径的量化策略、数据类型、缩放因子格式及量化公式。

---

## 1. 量化路径总览

```
                DeepSeek V4 MoE
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     MegaMoE                FusedMoE
  (SM100 Blackwell)        (通用 GPU)
          │                     │
     ┌────┴────┐          ┌────┴────┐
     │         │          │         │
  激活量化    权重量化   权重量化    激活量化
  FP8 E4M3    FP4        │         │
          ┌───────────────┼─────────┼──────┐
          ▼               ▼         ▼      ▼
       MXFP4           NVFP4      FP8    BF16
      (E8M0)      (fp8+fp32)  (float32)  (无)
```

---

## 2. MegaMoE 路径的量化(W4A8)

### 2.1 激活量化（FP8 E4M3 + E8M0 Scale）

**位置：** `PrepareMegaMoEInputs.md`（Triton kernel `_prepare_megamoe_inputs_kernel`）

#### 量化公式

```
输入: x ∈ BF16, shape [num_tokens, hidden_size]
隐藏: hidden_size % 128 == 0

1. 分块: 每 128 个元素为一组 → hidden_size / 128 组
2. 子分组: 每组再分为 4 个 GROUP_K=32 的子组
   num_groups = 128 / 32 = 4 个子组
3. amax: 每个子组计算 |x| 的最大值
   amax = max(|x[g*32:(g+1)*32]|), g ∈ [0, 4)
4. Scale 计算:
   scale_raw = amax / 448.0           # 448 = FP8 E4M3 最大值
5. E8M0 舍入: float32 → 纯指数格式
   scale_bits = bitcast<float32→uint32>(scale_raw)
   scale_exp = ((scale_bits >> 23) & 0xFF) + (尾数非零 ? 1 : 0)
   scale_exp = clamp(scale_exp, 1, 254)
   scale_e8m0 = bitcast<uint32→float32>(scale_exp << 23)
6. 量化:
   x_quant = x / scale_e8m0           # 每子组除以对应 scale
   x_fp8 = to_fp8_e4m3(x_quant)       # 转为 FP8 E4M3 格式
7. 打包 scale: 4 个 E8M0 指数打包为 1 个 int32
   packed = scale_exp[0] | (scale_exp[1]<<8) | (scale_exp[2]<<16) | (scale_exp[3]<<24)
```

#### 数据类型

| 变量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `hidden_states`（输入） | `(T, H)` | bf16 | T = num_tokens, H = hidden_size |
| `x_fp8`（输出） | `(T, H)` | float8_e4m3 | 量化后的激活值 |
| `x_sf`（输出） | `(T, H/128)` | int32 | 打包的 E8M0 缩放因子 |
| 中间 `amax` | `(T, H/32)` | float32 | 每子组 absmax |
| 中间 `scale_exp` | `(T, H/32)` | uint32 | E8M0 指数值 `[1, 254]` |

#### E8M0 格式详解

```
E8M0（unsigned exponent 8-bit, mantissa 0）:
┌─位 7─────────────────────────────位 0┐
│         指数 E（无符号 8 位）         │
└──────────────────────────────────────┘

实际缩放因子 = 2^(E - 127)

范围: [2^(-126), 2^(127)] ≈ [1.18e-38, 1.70e38]
精度: 纯指数，无尾数 → 缩放因子都是 2 的幂

转换 uint8 → float32:
  float32_value = (uint8_value << 23)  # 将 E 放到 float32 的指数位
                    .view(float32)      # 尾数位已清零
```

**为什么用 E8M0？** DeepGEMM 内核期望缩放因子为 2 的幂（纯指数），这样可以**通过移位而非乘法**进行反量化，节省硬件乘法器。E8M0 的 `[1, 254]` 范围（排除 0 和 255）覆盖了实际需要的所有缩放值。

**为什么除以 448？** FP8 E4M3 格式：
```
E4M3: 1 位符号 + 4 位指数 + 3 位尾数
```
* 严格遵循 IEEE 754 标准的 1-4-3 浮点数
   * 在标准 IEEE 754 中，指数位全 1 (1111) 被保留用于表示特殊值 (无穷大 / NaN)，不能表示普通数值。
   * 最大指数存储值：1110₂ = 14₁₀
   * 真实指数：14 - 7 = 7
   * 最大尾数：111₂ = 7/8 = 0.875，加上隐含前导 1 得 1.875
   * 最大值 = 1.875 × 2^7 = 1.875 × 128 = 240
* AI 硬件中使用的 E4M3FN 格式（最常用）
为了扩大数值范围以适应深度学习需求，NVIDIA、ARM 等厂商对 E4M3 进行了修改，形成了 **E4M3FN (Float8 E4M3)** 格式：
   * 不支持无穷大 (Inf)
   * 只有当指数位全 1 且尾数位全 1时才表示 NaN
   * 指数位全 1 (1111) 可以用来表示普通数值
   * 最大指数存储值：1111₂ = 15₁₀
   * 真实指数：15 - 7 = 8
   * 最大尾数：不能全 1（否则是 NaN），所以是 110₂ = 6/8 = 0.75，加上隐含前导 1 得 1.75
   * 最大值 = 1.75 × 2^8 = 1.75 × 256 = 448

使用 `amax / 448` 确保量化后的值不会超过 FP8 的表示范围，同时最大化利用可用位。

### 2.2 权重量化（FP4 + E8M0 Scale）

**位置：** `DeepseekV4MegaMoEExperts`（`nvidia/model.py:L138-L316`）

#### 权重格式

FP4 是 4-bit 浮点格式，两个 FP4 值打包在一个 uint8 字节中：

```
FP4: 1 位符号 + 2 位指数 + 1 位尾数
范围: ±(0, 0.5, 1, 1.5, 2, 3, 4, 6]
     (从指数和尾数的组合推导，具体值的集合)
```

#### 权重存储参数

| 参数 | 原始形状 | dtype | 含义 |
|------|---------|-------|------|
| `w13_weight` | `(E, 2*I, H/2)` | uint8 | gate+up 融合权重，每字节 2 个 FP4 |
| `w13_weight_scale` | `(E, 2*I, H/32)` | uint8 | E8M0 缩放因子，block_size=32 |
| `w2_weight` | `(E, H, I/2)` | uint8 | down 投影权重，每字节 2 个 FP4 |
| `w2_weight_scale` | `(E, H, I/32)` | uint8 | E8M0 缩放因子，block_size=32 |

其中：E = n_local_experts, H = hidden_size, I = intermediate_size

`weight_loader` 属性标注：
```python
self.w13_weight_scale.quant_method = "block"  # model.py:L182
```

**形状解释：** `hidden_size // 2` 因为 FP4 是 4-bit，一个 uint8 存两个值。`hidden_size // 32` 因为 block_size=32，每 32 个元素一个 scale。

#### Scale 存储细节

MegaMoE 路径的 checkpoint 中 scale 以 `float8_e8m0fnu` dtype 存储。加载时需 view 为 uint8（而非 copy_，否则会产生数值转换）：

```python
if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
    loaded_weight = loaded_weight.view(torch.uint8)
```

#### 权重布局变换（finalize_weights）

`finalize_weights()` 执行以下步骤：

1. **E8M0 uint8 → float32**：
   ```python
   scale_float32 = (sf_uint8.to(torch.int32) << 23).view(torch.float32)
   # 结果: 2^(E-127) 的 float32 值
   ```

2. **缩放因子重排**（`transform_sf_into_required_layout`）：
   - 输入形状: `(E, 2*I, H/32)` → DeepGEMM 内部布局
   - DeepGEMM 期望的 scale 布局与标准格式不同，需要按 ((1,32),) tile 重排

3. **权重交错**（`transform_weights_for_mega_moe`）：
   - 将 w13 和 w2 重新排列为 DeepGEMM 内核的交错格式
   - `_interleave_l1_weights` 将 w13 权重交错排列以优化内存访问模式

4. **释放原始参数**：
   ```python
   self.w13_weight = None  # 原始 uint8 参数
   self.w13_weight_scale = None
   self.w2_weight = None
   self.w2_weight_scale = None
   ```

**为什么丢弃原始参数？** DeepGEMM 的权重布局与原始格式不同，保留原始参数会浪费显存。推理场景不需要原始格式。

---

## 3. FusedMoE 路径的量化

FusedMoE 路径根据 `quant_config.get_quant_method()` 返回的量化方法，支持多种量化方案。

### 3.1 MXFP4（W4A16）

**位置：** `Mxfp4MoEMethod`

#### 权重格式

| 属性 | 值 |
|------|-----|
| 权重 dtype | uint8（每字节 2 个 FP4 nibble） |
| 激活 dtype | bf16（不量化激活） |
| Block size | 32（每 32 元素一个 scale） |
| Scale dtype | uint8（E8M0 格式） |
| 计算模式 | W4A16：权重 FP4 + 激活 BF16 |

#### 与 MegaMoE FP4 的差异

| 维度 | MegaMoE FP4 | MXFP4 (FusedMoE) |
|------|-------------|-------------------|
| 激活 | FP8 量化 | BF16 无量化 |
| Kernel | DeepGEMM `fp8_fp4_mega_moe` | Triton/FlashInfer MXFP4 MoE kernel |
| 后端 EP | EP 内部 all-to-all | TP 列切分 |
| Scale 布局 | `transform_sf_into_required_layout` | `convert_weight_to_mxfp4_moe_kernel_format` |
| 权重最终化 | `finalize_weights()` | `process_weights_after_loading()` |

#### 量化配置生成

```python
# select_deepseek_v4_mxfp4_moe_backend 选择最优后端
mxfp4_moe_quant_config = make_mxfp4_moe_quant_config(
    w1_weight=w13_weight,        # uint8 FP4 packed
    w1_scale=w13_weight_scale,   # uint8 E8M0
    w2_weight=w2_weight,
    w2_scale=w2_weight_scale,
    sparse=True,                  # DeepSeek V4 启用稀疏 MoE
)
```

#### MoeBackend 的选择

```python
# mxfp4.py: 从 TRITON_BACKENDS 中选择
# 可选后端: Triton, FlashInfer TRTLLM, AITER
backend = select_deepseek_v4_mxfp4_moe_backend()
```

### 3.2 NVFP4（W4A16 / W4A4）

**位置：** `ModelOptNvFp4FusedMoE`（`modelopt.py:L1384-1658`）

#### 权重格式

| 属性 | 值 |
|------|-----|
| 权重 dtype | `uint8`（每字节 2 个 FP4 nibble） |
| 激活 dtype | `bf16`（W4A16）/ `fp4`（W4A4） |
| Group size | 16（NVFP4 特有，`self.quant_config.group_size`） |
| Block scale dtype | `float8_e4m3fn`（NaN-free variant） |
| 实现 | ModelOpt CUTLASS/Marlin NVFP4 内核 |
| 后端选择 | `select_nvfp4_moe_backend(config, weight_key, activation_key)` |

#### W4A16 vs W4A4 后端选择

```python
self.use_a16 = quant_config.quant_method == "W4A16_NVFP4"
self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
    config=self.moe,
    weight_key=kNvfp4Static,             # 权重侧 NVFP4 量化
    activation_key=None if self.use_a16 else kNvfp4Dynamic,
    # W4A16: activation_key=None → 仅 Marlin 接受此配置
    # W4A4:  activation_key=kNvfp4Dynamic → 其他后端也参与选择
)
```
（`modelopt.py:L1397-1402`）

- **W4A16**：`activation_key=None` → 只有 Marlin 后端符合条件。Marlin 在 `convert_to_nvfp4_moe_kernel_format` 中丢弃 `input_scale`，因此参数虽创建但不参与计算
- **W4A4**：`activation_key=kNvfp4Dynamic` → 支持激活量化的后端（CUTLASS 等）也可选。`input_scale` 用于激活 FP8 量化

#### 三层缩放因子

NVFP4 使用三层缩放，与 MXFP4 的单一 E8M0 不同（`modelopt.py:L1437-1537`）：

**w13（gate+up 融合权重）**

| 层级 | 参数名 | dtype | shape | input_dim | 说明 |
|------|--------|-------|-------|-----------|------|
| Per-block | `w13_weight_scale` | float8_e4m3fn | `(E, 2*I, H/16)` | 1 | 每 16 原始元素一个 FP8 scale |
| Per-tensor global | `w13_weight_scale_2` | float32 | `(E, 2)` | — | act_and_mul 时 w1 和 w3 各一个全局 scale |
| Per-tensor input | `w13_input_scale` | float32 | `(E, 2)` | — | 仅 W4A4 有效；W4A16 被 Marlin 丢弃 |

**w2（down 投影权重）**

| 层级 | 参数名 | dtype | shape | input_dim | 说明 |
|------|--------|-------|-------|-----------|------|
| Per-block | `w2_weight_scale` | float8_e4m3fn | `(E, H, I/16)` | 1 | 每 16 原始元素一个 FP8 scale |
| Per-tensor global | `w2_weight_scale_2` | float32 | `(E,)` | — | 每个专家一个全局 scale |
| Per-tensor input | `w2_input_scale` | float32 | `(E,)` | — | 仅 W4A4 有效 |

> **注意**：shape 中的 `H` 与 `I` 均为未打包前的原始元素计数。`H/16` 是因为 group_size=16 且维度为原始 hidden_size。权重本身因 FP4 打包为 `H/2`，但 scale 的计数按 `H/16`（每 16 元素 1 scale）。

**权重最终化（`process_weights_after_loading`, `modelopt.py:L1539-1595`）：**

1. **w13_weight_scale_2 合并**：is_act_and_mul 时验证 `[:, 0]` 与 `[:, 1]` 近似相等，取 `[:, 0]` 为单一全局 scale
2. **格式转换**：`convert_to_nvfp4_moe_kernel_format` 将所有参数转为内核所需的布局
3. **参数替换**：`replace_parameter` 将原始参数替换为转换后的参数
4. **内核初始化**：`make_nvfp4_moe_kernel` → `process_weights_after_loading`

#### `use_global_sf` 标志

```python
self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(self.nvfp4_backend)
```
（`modelopt.py:L1404-1406`）

当 `use_global_sf=True` 时，`input_scale` 的 shape 使用 `global_num_experts` 而非 `num_experts`。这影响 EP（专家并行）场景：各 rank 需持有全局专家索引的 input_scale。

#### NVFP4 vs MXFP4 对比

| 属性 | MXFP4 | NVFP4 |
|------|-------|-------|
| Block size | 32 | 16 |
| Block scale dtype | uint8（E8M0） | float8_e4m3fn |
| Global scale | 无 | float32 per-tensor（`weight_scale_2`） |
| 激活量化 | 无（W4A16） | 是（W4A4） |
| Input activation scale | 无 | `input_scale`（仅 W4A4 有效） |
| 精度 | 较低（block 大、无全局 scale） | 较高（block 小 + 全局 scale） |
| 参数结构 | 1 层 scale | 3 层 scale（block + global + input） |
| kernel 后端 | Triton/FlashInfer/TRTLLM/AITER | CUTLASS/Marlin |

### 3.3 FP8 Block Quantization

**位置：** `Fp8MoEMethod`（`fp8.py`）

#### 权重格式

| 属性 | 值 |
|------|-----|
| 权重 dtype | float8_e4m3fn |
| Block size | 可配置（如 128×128） |
| Scale dtype | float32 |
| 激活量化 | 动态 per-token FP8（float32 scale，见下方详述） |
| 参数名 | `w13_weight_scale_inv`, `w2_weight_scale_inv` |

#### 激活量化

激活在 kernel 内部做**动态 per-token FP8 量化**，每 token 独立计算 scale。

```
输入: x ∈ BF16, shape [num_tokens, hidden_size]

1. amax: 每 token 计算 absmax（沿 hidden 维度）
   amax[t] = max(|x[t, :]|), t ∈ [0, num_tokens)

2. Scale: 每个 token 一个 float32 scale
   scale_a[t] = amax[t] / 448.0      # 448 = FP8 E4M3 最大值

3. 量化: 逐 token 缩放后转 FP8
   x_quant[t, :] = to_fp8_e4m3(x[t, :] / scale_a[t])

结果: x_quant ∈ float8_e4m3fn, scale_a ∈ float32, shape [num_tokens]
```

**数据通路：** scale_a 不经过 TMA，而是在 kernel 内计算后留在寄存器中，逐 block 与 weight scale 合并反量化。

#### Scale 维度

```
w13_weight_scale_inv: (E, 2*ceil(I/block_n), ceil(H/block_k))
w2_weight_scale_inv:  (E, ceil(H/block_n), ceil(I/block_k))
```

典型 block_n=128, block_k=128 时：
```
w13_scale: (E, 2*ceil(I/128), ceil(H/128))    float32
w2_scale:  (E, ceil(H/128), ceil(I/128))       float32
```

#### 反量化方式（fused_moe kernel 内部）

```python
# Block 反量化：在 K 循环内部逐 block 应用 scale
accumulator += tl.dot(a_block, b_block) * a_scale[:, None] * b_scale[None, :]
# a_scale: per-token 动态 scale（float32）
# b_scale: per-block 静态 weight scale（float32）
```

与 per-tensor 方案不同，block 反量化在 K 循环内部完成，避免大 scale 值导致的精度损失。

---

## 4. 共享专家量化(W8A16)

### 数据类型

共享专家 `DeepseekV4MLP` 不是一个 `FusedMoE` 层，所以其量化由 `Fp8Config.get_quant_method` 处理（返回 `Fp8LinearMethod`）。

| 组件 | 权重 dtype | Scale dtype | Scale shape |
|------|-----------|-------------|-------------|
| `gate_up_proj` | float8_e4m3fn | float32 | per-block |
| `down_proj` | float8_e4m3fn | float32 | per-block |
| 激活 | bf16 | — | — |

### reduce_results 策略

```python
self.shared_experts = DeepseekV4MLP(
    ...
    reduce_results=self.use_mega_moe,   # MegaMoE: True → 不 all-reduce
    disable_tp=is_sequence_parallel,    # MegaMoE: True → 复制层
)
```

- **MegaMoE 路径**：`disable_tp=True`，每个 rank 持有完整共享专家权重，无 TP 通信
- **FusedMoE 路径**：标准 TP 切分，`gate_up_proj` 列切分，`down_proj` 行切分 + all-reduce

---

## 5. 数据类型对比总结

### 5.1 所有权重格式对比

| 方案 | 存储 dtype | 有效格式 | Scale dtype | Block size | 2 值/字节 | 参数名 |
|------|-----------|---------|-------------|-----------|----------|--------|
| MegaMoE FP4 | uint8 | FP4 | uint8（E8M0） | 32 | ✓ | `weight_scale` |
| MXFP4 | uint8 | FP4 | uint8（E8M0） | 32 | ✓ | `weight_scale` |
| NVFP4 | uint8 | FP4 | float8_e4m3fn | 16 | ✓ | `weight_scale` |
| FP8 block | float8_e4m3fn | FP8 E4M3 | float32 | 128×128 | ✗ | `weight_scale_inv` |
| BF16（无量化） | bfloat16 | BF16 | — | — | ✗ | — |

### 5.2 所有激活量化格式对比

| 方案 | 量化位置 | 激活 dtype | Scale dtype | 实际粒度 | 备注 |
|------|---------|-----------|-------------|---------|------|
| MegaMoE 输入 | `prepare_megamoe_inputs` | float8_e4m3 | uint32（打包 4×E8M0） | 32（子分组）→ 128（打包） | |
| FP8 block (FusedMoE) | `fused_experts` 内部 | float8_e4m3 | float32 | per-token（精确） | scale 留在寄存器 |
| MXFP4 / NVFP4 W4A16 | 无 | bf16 | — | — | 激活不量化 |
| NVFP4 W4A4 | `fused_experts` 内部 | fp4 | float32（动态 per-16-group）+ float32（静态 per-tensor input_scale） | 16 | kernel 内 per-group + 全局 input_scale 偏置 |

### 5.3 E8M0 格式在系统中的应用

E8M0（unsigned exponent 8-bit, mantissa 0）是 DeepSeek V4 中最广泛使用的 scale 格式：

| 使用位置 | 用途 | 打包方式 |
|---------|------|---------|
| MegaMoE 激活量化 | 每 128 元素的 FP8 缩放因子 | 4 个 E8M0 打包为 int32 |
| MegaMoE 权重量化 | 每 32 元素的 FP4 缩放因子 | 每个 E8M0 存为 uint8 |
| MXFP4 权重量化 | 每 32 元素的 FP4 缩放因子 | 每个 E8M0 存为 uint8 |
| KV Cache FP8 | 每 64 元素的 FP8 缩放因子 | 7 + 1 padding 字节 |
| wo_a 输入量化 | group_shape=(1,128) | 打包格式可选 DeepGEMM |

**E8M0 转换为 float32 的统一公式：**
```python
float32_value = (uint8_e8m0_value.to(torch.int32) << 23).view(torch.float32)
# 等价于 float32_value = 2^(E8M0_value - 127)
```

---

## 6. Intra-MMoE 量化数据流图

### MegaMoE

```
hidden_states [T, H] bf16
    │
    ├─ prepare_megamoe_inputs
    │   └─ 动态 FP8 量化 (BF16→FP8 E4M3, E8M0 scale, block=128)
    │
    ├─ x_fp8 [T, H] float8_e4m3
    │   x_sf  [T, H/128] int32 (packed)
    │
    ├─ deep_gemm.fp8_fp4_mega_moe
    │   ├─ 权重 w1: [E, 2I, H/2] uint8 (FP4 packed) + scale [E, 2I, H/32] uint8 (E8M0)
    │   ├─ 权重 w2: [E, H, I/2] uint8 (FP4 packed) + scale [E, H, I/32] uint8 (E8M0)
    │   │
    │   ├─ 输入反量化: x_bf16 = x_fp8 * 2^(sf-127)   # E8M0 → float32 反量化
    │   ├─ 权重反量化: w_bf16 = unpack_fp4(w_uint8) * 2^(w_scale-127)
    │   ├─ FP4 x FP8 GEMM (float32 累加)
    │   └─ 输出: [T, H] bf16
    │
    ├─ shared_experts (FP8 block quant)
    └─ final_hidden_states [T, H] bf16
```

### FusedMoE (MXFP4)

```
hidden_states [T, H] bf16
    │
    ├─ FusedMoE kernel (Triton/FlashInfer)
    │   ├─ 激活: bf16 (无量化)
    │   ├─ 权重 w1: uint8 (FP4 packed) + uint8 (E8M0 scale)
    │   ├─ 权重 w2: uint8 (FP4 packed) + uint8 (E8M0 scale)
    │   ├─ 内部: FP4 × BF16 GEMM (float32 累加)
    │   └─ 输出: [T, H] bf16
    │
    ├─ shared_experts (FP8 block quant)
    └─ final_hidden_states [T, H] bf16
```

### FusedMoE (FP8)

```
hidden_states [T, H] bf16
    │
    ├─ FusedMoE kernel (Triton)
    │   ├─ 激活量化: per-token-group FP8 (float8_e4m3 + float32 scale)
    │   ├─ 权重 w1: float8_e4m3 + float32 per-block scale
    │   ├─ 权重 w2: float8_e4m3 + float32 per-block scale
    │   ├─ 内部: FP8 × FP8 GEMM (float32 累加, 逐 block 反量化)
    │   └─ 输出: [T, H] bf16
    │
    ├─ shared_experts (FP8 block quant)
    └─ final_hidden_states [T, H] bf16
```

---

## 7. 量化参数与显存收益

| 方案 | 每专家权重大小 | 与 BF16 比 | 说明 |
|------|--------------|-----------|------|
| BF16（基线） | `4 × H × I` bytes | 1× | gate+up+down 各 ~2H×I |
| FP8 | `2 × H × I` bytes | 0.5× | 各参数减半 |
| FP4（MXFP4/NVFP4） | `1 × H × I` bytes | 0.25× | 每字节 2 个值 + scale 开销 |
| MegaMoE FP4 | `~1 × H × I` bytes | ~0.25× | 额外 scale 布局开销 |

以 DeepSeek V4 典型配置（H=4096, I=12288, E=256）：
- BF16: 256 × 4 × 4096 × 12288 × 2 bytes ≈ 96 GB
- FP4: 256 × 1 × 4096 × 12288 × 2 bytes ≈ 24 GB（节省 72 GB）

---

个人认为主要的几个点：
* QAT：原始精度转FP4，再转FP8进行训练，模拟FP4量化
* SwiGLU clamping：适应FP4的精度范围
* Anticipatory routing：gate的权重用历史时刻的，非当前最新的

---


> **注：** FP4 QAT 训练方法及激活量化 Outlier 处理已移至 [[QuantAndParallelStrategy]] §7-§8。

## 相关笔记

- [[PrepareMegaMoEInputs]] — MegaMoE 输入 FP8 量化 Triton 内核详解
- [[DeepseekV4MoE]] — MoE 层完整实现
- [[DeepseekV4FP8Config]] — 量化配置分发中心
- [[QuantAndParallelStrategy]] — 量化策略全景
- [[DeepseekV4_KVCache_Ops]] — KV Cache 的 FP8 量化

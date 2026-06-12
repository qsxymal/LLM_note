# 大模型量化浮点数据类型

## 通用计算框架

```
value = (-1)^S × 2^(E - bias) × (1 + M / 2^m)      bias = 2^(e-1) - 1
```

> **IEEE vs AI：** IEEE 754 为 Inf/NaN 保留 E=全 1 的编码。AI 硬件在低位宽格式中去掉 Inf/NaN 保留，将 E=全 1 释放为正常数以扩大范围。以下用 **AI** 标注实际硬件范围，用 **IEEE** 标注严格遵循 754 的理论值（仅列出有差异的类型）。

> **量化粒度说明：**
> - **per-tensor**：整个 tensor 共享一个 scale
> - **per-channel**：权重每输出通道一个 scale，沿 reduction dim 分组（1D）
> - **per-token**：激活每 token 一个 scale，沿序列维度分组（1D）
> - **per-group**：权重沿 reduction dim 按 group_size 分组（1D，即 per-channel 但粒度更细），常见 gs=32/64/128
> - **per-block**：权重矩阵按 block_n×block_k 划分为 2D tile，每 tile 一个 scale。同时沿 input/reduction dim 和 output dim 分组，能捕获两个方向的数值变化，但 scale 数量更多
> - **block(MX)**：概念同 per-group（沿 reduction dim，32元素一组），但 scale 固定为 E8M0 共享指数（纯 2^n），是 OCP 标准定义的硬件原生格式

---

## 1. 16-bit

### 1.1 BF16 (1-8-7)

```
bias=127, E_max=254 (E=255 保留), M_max=127
max = 2^(254-127) × (1+127/128) = 255 × 2^120 ≈ 3.39×10^38
min = 2^(1-127) × 2^-7 = 2^-133 ≈ 9.18×10^-41
```

**范围：** ±[9.18×10⁻⁴¹, 3.39×10³⁸]

训练主格式、精度基准，非量化格式。

### 1.2 FP16 (1-5-10)

```
bias=15, E_max=30 (E=31 保留), M_max=1023
max = 2^(30-15) × (1+1023/1024) = 65504
min = 2^(1-15) × 2^-10 = 2^-24 ≈ 5.96×10^-8
```

**范围：** ±[5.96×10⁻⁸, 6.55×10⁴]

传统混合精度训练格式，非量化格式。已逐步被 BF16 取代。

---

## 2. 8-bit

### 2.1 FP8 E5M2 (1-5-2)

```
bias=15, E_max=30 (E=31 保留), M_max=3
max = 2^(30-15) × (1+3/4) = 57344
min = 2^(1-15) × 2^-2 = 2^-16 ≈ 1.53×10^-5
```

**范围：** ±[1.53×10⁻⁵, 5.73×10⁴]

范围大但精度极低（2-bit 尾数，每指数区间仅 4 级）。H100 FP8 训练的反向通路。

### 2.2 FP8 E4M3 (1-4-3)

| | IEEE | AI |
|---|---|---|
| E_max | 14（E=15 保留） | 15（仅 M=7 做 NaN）|
| 最大值 | 2^(14-7)×(1+7/8) = **240** | 2^(15-7)×(1+6/8) = **448** |
| min_subnormal | 2^(1-7)×2^-3 = 2^-9 | 同左 |

**范围：** IEEE ±[1.95×10⁻³, 240] / AI **±[1.95×10⁻³, 448]**

**使用：** W8A8 推理主流（vLLM+H100）；训练前向

**优点：** 近无损（PPL↑<0.1），H100 Tensor Core 2× 吞吐

**缺点：** 对激活 outlier 敏感，需 SmoothQuant

### 2.3 MXFP8

局部格式 E4M3 或 E5M2，每 32 元素加 1 个 E8M0 8-bit 共享指数。

```
32 × 8-bit (数据) + 1 × 8-bit (E8M0 scale) = 33 字节/block
→ 8.25 bit/element
```

E8M0：纯 8-bit 指数（bias=127），存 2^n，范围 2⁻¹²⁷~2¹²⁷。

B200 端到端训练/推理统一格式。硬件内建 block 级自适应缩放，无需软件量化校准。需 B200+ 硬件。

---

## 3. 4-bit

### 3.1 FP4 E2M1 (1-2-1)

| | IEEE | AI |
|---|---|---|
| E_max | 2（E=3 保留） | 3（无 Inf/NaN）|
| 最大值 | 2^(2-1)×(1+1/2) = **3** | 2^(3-1)×(1+1/2) = **6** |
| min_subnormal | 2^(1-1)×2^-1 = 0.5 | 同左 |

**范围：** IEEE ±[0.5, 3]（8 个正值：0, 0.5, 1, 1.5, 2, 3）/ AI **±[0.5, 6]**（+4, 6）

E=3 释放，增加 4 和 6 两个可取值。每字节打包 2 个 FP4，配合 E8M0 scale（纯 2^n 移位反量化）。

### 3.2 NVFP4（NVIDIA Blackwell）

局部格式 E2M1（范围 ±6），每 **16** 元素加 1 个 **FP8 E4M3** 8-bit scale。
**量化：** weight per-group (gs=16, FP8 E4M3 scale + FP32 tensor scale) | W4A16 / W4A4

```
16 × 4-bit (数据) + 1 × 8-bit (FP8 E4M3 scale) = 72 bit/block
→ 4.5 bit/element
```

**双重量化：** `dequant = E2M1 × block_scale_FP8 × tensor_scale_FP32`

| 层级 | 精度 | 粒度 | bit/el |
|------|------|------|--------|
| E2M1 | 4-bit | 每元素 | 4 |
| block scale | FP8 E4M3 | 每 16 元素 | 0.5 |
| tensor scale | FP32 | 每 tensor | ≈0 |
| **合计** | | | **4.5** |

| vs MXFP4 | NVFP4 | MXFP4 |
|---|---|---|
| block 大小 | **16** 元素（更细）| 32 元素 |
| scale 格式 | **FP8 E4M3**（±448） | E8M0（纯 2^n）|
| 额外 tensor scale | FP32 | 无 |
| 存储 | **4.5 bit/el** | 4.25 bit/el |

NVFP4 用更多的 scale 开销换取更细的 block 粒度和更精确的 scale。Blackwell Tensor Core 原生支持，TensorRT-LLM / vLLM≥0.8 可用。

### 3.3 MXFP4

局部格式 E2M1（范围 ±6），每 32 元素加 1 个 E8M0 共享指数。
**量化：** weight per-group (gs=32, E8M0 scale) | W4A16（激活不量化）

```
32 × 4-bit + 1 × 8-bit = 17 字节/block → 4.25 bit/element
```

压缩比：7.5× vs FP32，4× vs BF16。

B200 推理主打格式。敏感层（首尾层、Attention 投影）需回退高精度。

---

## 综合对比表

| 类型 | 位宽 | e:m | IEEE 最大 | AI 最大 | 量化对象 | 实际存储 | 主要用途 |
|------|------|-----|-----------|---------|---------|---------|---------|
| BF16 | 16 | 8:7 | 3.39e38 | 同左 | — | 16 bit/el | 训练主格式 |
| FP16 | 16 | 5:10 | 65504 | 同左 | — | 16 bit/el | 传统训练 |
| E5M2 | 8 | 5:2 | 57344 | 同左 | gradient | 8 bit/el | 训练梯度 |
| E4M3 | 8 | 4:3 | 240 | **448** | weight+activation | 8 bit/el | W8A8 推理/前向 |
| MXFP8 | 8+E8M0 | 4:3/5:2 | 240/57344 | 448/57344 | weight+activation+gradient | 8.25 bit/el | 下一代训练/推理 |
| E2M1 | 4 | 2:1 | 3 | **6** | weight（2值/字节） | 4 bit/el | 极端压缩 |
| **NVFP4** | 4+E4M3 | 2:1 | 3 | **6** | weight(W4A16)/weight+activation(W4A4) | **4.5 bit/el** | B200 推理 |
| **MXFP4** | 4+E8M0 | 2:1 | 3 | **6** | weight(W4A16) | 4.25 bit/el | B200 推理 |

* FP4和MXFP4可以做weight量化，但不适合激活量化，因为激活值对精度更敏感
* NVFP4激活和weight量化都支持

DeepseekV4有：
* W8A16: Attention的Linear(O仅W_b过程)；Attention decode；MoE的共享专家
* W8A8: Attention的O Linear的W_a过程；FusedMoE的FP8 block quant / 通用推理
* W4A16: FusedMoE，支持MXFP4和NVFP4
* W4A8: MegaMoE
* W4A4: FusedMoE的NVFP4; Indexer的mqa_logits
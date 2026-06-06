# DeepseekV4DecoderLayer —— 多头流混合解码器层

**文件路径：** `vllm/models/deepseek_v4/nvidia/model.py`（NVIDIA 主实现），`vllm/models/deepseek_v4/amd/model.py`（AMD）
**类定义：** `nvidia/model.py:L845-L1071`
**平台选择：** `__init__.py` 根据 `current_platform` 自动导入 NVIDIA 或 AMD 实现

下面详细讲解 `DeepseekV4DecoderLayer` 类的设计、初始化、核心算子以及前向传播逻辑。该类是 DeepSeek V4 模型的基础构建块，集成了**多头流（multi‑stream）处理**、**混合残差连接**、**注意力机制**和 **MoE（混合专家）FFN**，并且针对 CUDA 平台做了大量融合优化。

---

## 1. 类概述

`DeepseekV4DecoderLayer` 代表模型的一层 Transformer 解码器。与标准解码器的主要区别在于：

- **输入/输出不是普通的 hidden states**，而是带有 **`hc_mult` 个并行流**的张量，形状为 `(num_tokens, hc_mult, hidden_size)`。
- 在每一层内部，通过 **HC Pre‑processing** 和 **HC Post‑processing** 将残差连接与多个流的混合/归一化整合为高效的融合算子。
- 支持两种模式：  
  - `_forward_cuda`（`model.py:L958`）：在 NVIDIA GPU 上使用高度融合的自定义内核（`mhc_fused_post_pre`），将多个操作合并，减少内核启动次数。  
  - `_forward_native`（`model.py:L1027`）：在非 CUDA 平台（ROCm、XPU 或纯 PyTorch 回退）上使用分步操作，保证兼容性。

### 关键行号速查

| 方法/属性 | 行号 | 说明 |
|----------|------|------|
| `__init__` | L846-L923 | 初始化所有子模块和 HC 参数 |
| `hc_pre` | L925-L947 | HC 预处理：从残差生成 layer_input + 混合系数 |
| `hc_post` | L949-L956 | HC 后处理：合并子层输出与残差 |
| `_forward_cuda` | L958-L1025 | CUDA 融合前向（两个阶段 + fused_post_pre） |
| `_forward_native` | L1027-L1053 | 原生回退前向（分步执行） |
| `forward` | L1055-L1071 | 路由：根据平台选择实现路径 |

---

## 2. `__init__` 初始化

```python
def __init__(self, vllm_config, prefix, topk_indices_buffer=None, aux_stream_list=None):
```

**位置：** `model.py:L846-L923`

### 2.1 导入 MHC 算子（L857）

```python
import vllm.model_executor.layers.mhc  # noqa: F401
```

- `mhc` 模块注册了三个 PyTorch 自定义算子：`torch.ops.vllm.mhc_pre`、`mhc_post`、`mhc_fused_post_pre`。这些算子用 TileLang 或 CUDA 实现，专门用于 DeepSeek V4 的多头流操作。
- 使用 lazy import 避免顶层 `tilelang` 依赖（L856 注释）。

### 2.2 基础组件（L863-L872）

[[DeepseekV4Attention]]和[[DeepseekV4MoE]]

```python
self.attn = DeepseekV4Attention(...)          # 多头注意力
self.ffn = DeepseekV4MoE(...)                 # 混合专家 FFN
self.attn_norm = RMSNorm(...)                 # 注意力前的 RMSNorm
self.ffn_norm = RMSNorm(...)                  # FFN 前的 RMSNorm
```

- 注意：在 CUDA 融合路径中，`attn_norm` 和 `ffn_norm` 的权重会被传入 HC 预处理算子内部，从而省略显式的 `norm` 调用。
- `attn_norm` / `ffn_norm` 是 `RMSNorm`（均方根归一化），对输入的最后维做归一化：`x * rsqrt(mean(x^2) + eps)`。

### 2.3 HC 相关可学习参数（L873-L920）

```python
mix_hc = (2 + self.hc_mult) * self.hc_mult
hc_dim = self.hc_mult * self.hidden_size
```

- `hc_mult`（L873）：流的数量。`mix_hc` 是 HC 混合矩阵的行数，与 `hc_mult` 的关系来自特定的线性变换（可能用于生成组合系数）。  
- `hc_dim`（L878）：单个流展平后的总维度。

然后定义四个参数对（每个对应一个变换，一个用于 attention 分支，一个用于 ffn 分支）：

```python
self.hc_attn_fn = nn.Parameter(...)     # 形状 (mix_hc, hc_dim)
self.hc_attn_base = nn.Parameter(...)   # 长度 mix_hc
self.hc_attn_scale = nn.Parameter(...)  # 长度 3（缩放因子）
self.hc_ffn_fn = nn.Parameter(...)      # 类似
self.hc_ffn_base = nn.Parameter(...)
self.hc_ffn_scale = nn.Parameter(...)
```

- `_fn`：用于流混合的权重矩阵。  
- `_base`：偏置项。  
- `_scale`：三个标量（可能用于控制不同阶段的缩放）。  
- 这些参数从 checkpoint 加载，且 `requires_grad=False`（推理时冻结）。

### HC 参数规格总览

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `hc_attn_fn` | `(mix_hc, hc_dim)` | float32 | Attention 分支流混合权重矩阵 |
| `hc_attn_base` | `(mix_hc,)` | float32 | Attention 分支偏置 |
| `hc_attn_scale` | `(3,)` | float32 | Attention 分支缩放因子 |
| `hc_ffn_fn` | `(mix_hc, hc_dim)` | float32 | FFN 分支流混合权重矩阵 |
| `hc_ffn_base` | `(mix_hc,)` | float32 | FFN 分支偏置 |
| `hc_ffn_scale` | `(3,)` | float32 | FFN 分支缩放因子 |

> **为什么是 float32？** 这些参数是混合系数生成矩阵，精度要求高但计算量小（相对于 bf16 的主干计算），float32 可保证 Sinkhorn 迭代的数值稳定性。

### 2.4 MHC 算子实例（L921-L923）

```python
self.mhc_pre = MHCPreOp()
self.mhc_post = MHCPostOp()
self.mhc_fused_post_pre = MHCFusedPostPreOp()
```

- `MHCPreOp`：实现 `hc_pre` 逻辑，输入 `residual`（形状 `[B, hc_mult, D]`），输出混合后的 `layer_input` 以及两个中间张量 `post_mix` 和 `res_mix`。  
- `MHCPostOp`：实现 `hc_post` 逻辑，将当前层输出 `x`、残差 `residual`、以及 `post_mix`、`res_mix` 结合，产生最终的层输出。  
- `MHCFusedPostPreOp`：将上一层的 `hc_post` 和当前层的 `hc_pre` 融合成一个内核，以减少内存读写和内核启动开销。这是 CUDA 专用优化。

---

## 3. `hc_pre` 方法（L925-L947）

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base, norm_weight=None, norm_eps=1e-6):
    post_mix, res_mix, layer_input = self.mhc_pre(
        residual=x,
        fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=self.rms_norm_eps,
        hc_pre_eps=self.hc_eps,
        hc_sinkhorn_eps=self.hc_eps,
        hc_post_mult_value=self.hc_post_alpha,
        sinkhorn_repeat=self.hc_sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return layer_input, post_mix, res_mix
```

- **输入**：`x` 是上一层的输出（形状 `[B, hc_mult, D]`），同时也是当前层的残差。  
- 内部调用 `mhc_pre` 算子，该算子会：
  - 可能执行 RMSNorm（若 `norm_weight` 非空）。
  - 使用 `hc_fn`、`hc_scale`、`hc_base` 对输入进行线性变换，产生两个混合矩阵 `post_mix` 和 `res_mix`（形状可能为 `[B, mix_hc]` 或类似），以及最终的 `layer_input`（形状 `[B, hc_mult, D]`），后者将传入注意力或 FFN。
- `hc_sinkhorn_iters`（L874）：Sinkhorn 迭代次数，用于稳定双随机归一化（这里可能用于生成混合系数）。

`hc_pre` 的功能可以理解为：**基于残差流，为后续的 attention/FFN 准备输入，同时产生后处理所需的混合系数**。

### 输入输出规格

| 变量 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` (residual) | `(B, hc_mult, H)` | bf16 | 上层输出/残差流 |
| `hc_fn` | `(mix_hc, hc_dim)` | float32 | 混合权重（attention 或 FFN 版本） |
| `hc_scale` | `(3,)` | float32 | 缩放因子 |
| `hc_base` | `(mix_hc,)` | float32 | 偏置 |
| `hc_post_alpha` | scalar | float32 | `2.0`（硬编码 L876），后处理乘性因子 |
| `layer_input` 输出 | `(B, hc_mult, H)` | bf16 | 传入子层（attention/FFN）的输入 |
| `post_mix` 输出 | `(B, mix_hc)` | float32 | 后处理混合系数 |
| `res_mix` 输出 | `(B, mix_hc)` | float32 | 残差混合系数 |

> `hc_mult=2` 时：`mix_hc = (2+2)*2 = 8`，`hc_dim = 2 * 隐藏层维度`

---

## 4. `hc_post` 方法（L949-L956）

```python
def hc_post(self, x, residual, post, comb):
    return self.mhc_post(x, residual, post, comb)
```

- 将 attention/FFN 的输出 `x` 与残差 `residual` 结合，使用 `post` 和 `comb` 系数进行加权混合，最终返回更新后的 `x`（形状不变）。

### 输入输出规格

| 变量 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `(B, hc_mult, H)` | bf16 | 子层输出 |
| `residual` | `(B, hc_mult, H)` | bf16 | 残差连接 |
| `post` | `(B, mix_hc)` | float32 | hc_pre 产生的后处理系数 |
| `comb` | `(B, mix_hc)` | float32 | hc_pre 产生的组合系数 |
| 返回值 | `(B, hc_mult, H)` | bf16 | 混合后的层输出 |

---

## 5. `_forward_cuda`（CUDA 融合路径，L958-L1025）

这是为 NVIDIA GPU 高度优化的前向函数。整个计算分为两个大阶段：**attention 阶段** 和 **FFN 阶段**，每个阶段内部使用 `mhc_fused_post_pre` 将多个操作融合。

### 5.1 第一阶段：Attention（L969-L1001）

```python
attn_norm_weight = self.attn_norm.weight.data
attn_norm_eps = self.attn_norm.variance_epsilon
if residual is None:
    # 第一层（没有传入上一层的 post/res_mix）: 单独调用 hc_pre
    residual = x
    x, post_mix, res_mix = self.hc_pre(
        x,
        self.hc_attn_fn,
        self.hc_attn_scale,
        self.hc_attn_base,
        norm_weight=attn_norm_weight,
        norm_eps=attn_norm_eps,
    )
else:
    # 非第一层: 融合上一层的 hc_post 和当前层的 hc_pre
    residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
        x,
        residual,
        post_mix,
        res_mix,
        self.hc_attn_fn,
        self.hc_attn_scale,
        self.hc_attn_base,
        self.rms_norm_eps,
        self.hc_eps,
        self.hc_eps,
        self.hc_post_alpha,
        self.hc_sinkhorn_iters,
        n_splits=1,
        tile_n=1,
        norm_weight=attn_norm_weight,
        norm_eps=attn_norm_eps,
    )
```

- `residual` 是上一层传递来的 `residual` 流（跨层传递）。  
- `mhc_fused_post_pre` 首先完成**上一层的 `hc_post`**（将上一层的输出与残差混合），然后**立即执行当前层的 `hc_pre`**（准备 attention 输入）—— 二者被融合进一个内核。  
- 该算子输出四个值：更新后的 `residual`（新残差）、新的 `post_mix`、新的 `res_mix`，以及当前层 attention 的输入 `x`。  

```python
x = self.attn(positions, x, None)
```

- `x` 现在是经过 `hc_pre` 处理后的输入（形状 `[B, hc_mult, D]`，但可能与原始 `hc_mult` 不同？实际上经过 `hc_pre` 后的 `x` 仍为相同形状，因为 `layer_input` 保持 `hc_mult` 个流）。
- 传入的第三个参数 `None` 是 `llama_4_scaling`（跳过了 scaling 参数）。

### 5.2 第二阶段：FFN（L1003-L1025）

```python
ffn_norm_weight = self.ffn_norm.weight.data
ffn_norm_eps = self.ffn_norm.variance_epsilon
residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
    x,
    residual,
    post_mix,
    res_mix,
    self.hc_ffn_fn,
    self.hc_ffn_scale,
    self.hc_ffn_base,
    self.rms_norm_eps,
    self.hc_eps,
    self.hc_eps,
    self.hc_post_alpha,
    self.hc_sinkhorn_iters,
    n_splits=1,
    tile_n=1,
    norm_weight=ffn_norm_weight,
    norm_eps=ffn_norm_eps,
)
x = self.ffn(x, input_ids)
return x, residual, post_mix, res_mix
```

- 同样的融合操作，但使用 FFN 的 HC 参数。  
- `self.ffn` 是 MoE 层，输出形状与输入相同（`[B, hc_mult, D]`）。  
- 最后返回 `(x, residual, post_mix, res_mix)`，其中 `x` 是这一层最终的隐藏状态（即将作为下一层的输入），而 `residual`、`post_mix`、`res_mix` 被传递到下一层，以便在下一层的 `mhc_fused_post_pre` 中继续融合。

**关键**：`residual`、`post_mix`、`res_mix` 跨层传递，使得相邻两层之间的后处理+预处理可以合并，极大减少内核调用和全局内存访问。

### 5.3 CUDA 融合路径流程图

```
第一层 (residual=None)：
  x_in → hc_pre(attn) → attn() → mhc_fused_post_pre(ffn) → ffn() → (x, residual, post_mix, res_mix)
         └─ norm fused in  └─融合了 attn hc_post + ffn hc_pre

后续层 (residual≠None)：
  (x, residual, post_mix, res_mix) 
    → mhc_fused_post_pre(attn) → attn() 
    → mhc_fused_post_pre(ffn) → ffn() 
    → (x, residual, post_mix, res_mix) 传给下一层
       ↑                            ↑
       └─融合了 上一层hc_post        └─融合了 attn hc_post 
          + 当前层attn hc_pre           + ffn hc_pre
```

**关键观察**：每层只调用 `mhc_fused_post_pre` **两次**（attention 和 FFN 各一次），而非四次（`hc_post` + `hc_pre` × 2）。第一层特殊处理：由于没有上一层的 `hc_post` 需要融合，attention 前退化为单独调用 `hc_pre`。

---

## 6. `_forward_native`（回退路径，L1027-L1053）

```python
def _forward_native(...):
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    x = self.attn_norm(x)
    x = self.attn(positions, x, None)
    x = self.hc_post(x, residual, post, comb)

    residual = x
    x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post(x, residual, post, comb)
    return x, None, None, None
```

- 没有融合，每个 `hc_pre` 和 `hc_post` 都是独立调用，且显式执行 RMSNorm。  
- 不返回 `post_mix, res_mix`（因为后续没有融合需求）。  
- 用于 ROCm、XPU 或 CPU 调试，保证正确性但性能较低。

### 原生路径流程图

```
x_in → residual = x
     → hc_pre(attn) → attn_norm → attn() → hc_post
     → residual = x
     → hc_pre(ffn) → ffn_norm → ffn() → hc_post
     → return (x, None, None, None)
```

### 为什么 `_forward_native` 返回 `None, None, None`？

融合路径中 `residual, post_mix, res_mix` 必须在层间传递以供 `mhc_fused_post_pre` 使用。但在 native 路径中，所有 `hc_post` / `hc_pre` 均独立执行，跨层状态不再需要。返回 `None` 确保调用方（[[DeepseekV4Model]]）不会在下一层尝试使用这些值进行融合。

---

## 7. `forward` 路由（L1055-L1071）

```python
def forward(...):
    if current_platform.is_rocm() or current_platform.is_xpu():
        return self._forward_native(...)
    return self._forward_cuda(...)
```

- ROCm/XPU 上使用 native 实现（因为自定义 CUDA 算子无法运行）。  
- 其他平台（主要是 CUDA）使用融合实现。

### AMD 实现的差异

AMD 平台在 `amd/model.py` 中提供自己的 `DeepseekV4DecoderLayer`。差异点包括：
- 可能使用不同的 MHC 算子实现（`amd_mhc_pre`、`amd_mhc_post`）
- `_forward_cuda` 被替换为自定义的 AMD 融合路径（可能使用 rocBLAS 或 Composable Kernel）
- 共享相同的 HC 参数结构，权重兼容

---

## 8. 数据流形状总结

| 位置 | 形状 | dtype | 说明 |
|------|------|-------|------|
| 层输入 `x` | `(B, hc_mult, H)` | bf16 | B = total tokens, H = hidden_size |
| `hc_pre` 输出 `layer_input` | `(B, hc_mult, H)` | bf16 | 仍保持流结构，attn_norm 已融合 |
| Attention 输入 | `(B, hc_mult, H)` | bf16 | 经过 hc_pre 预处理后的多流状态 |
| Attention 输出 | `(B, hc_mult, H)` | bf16 | 多流注意力结果，形状不变 |
| FFN 输入 | `(B, hc_mult, H)` | bf16 | 经过 attn hc_post + ffn hc_pre 后的多流状态 |
| FFN 输出 | `(B, hc_mult, H)` | bf16 | MoE 输出，形状不变 |
| 层输出 `x` | `(B, hc_mult, H)` | bf16 | 层最终结果 |
| `residual` | `(B, hc_mult, H)` | bf16 | 跨层传递的残差流 |
| `post_mix` | `(B, mix_hc)` | float32 | Sinkhorn 归一化的后处理混合系数 |
| `res_mix` | `(B, mix_hc)` | float32 | Sinkhorn 归一化的残差混合系数 |

> **关键观察**：整个层内所有主要张量保持 `(B, hc_mult, H)` 形状不变。混合系数 `post_mix` / `res_mix` 使用 float32 精度以保证 Sinkhorn 迭代的稳定性。

---

## 9. 设计亮点

- **高融合**：CUDA 上通过 `mhc_fused_post_pre` 将相邻层的后处理+预处理合并，减少内核启动和全局内存读写。  
- **RMSNorm 融合**：Norm 的权重和 eps 被传入 HC 算子内部，消除单独的归一化内核。  
- **跨层状态传递**：`residual`、`post_mix`、`res_mix` 在层间流动，形成一个链式的混合状态机。  
- **平台自适应**：自动选择 native 实现以兼容非 CUDA 设备。  

这种设计是 DeepSeek V4 实现极致推理性能的关键，尤其是在大规模 MoE 和多头流并行的情况下。

---

## 10. 设计决策分析

### 10.1 为什么采用两阶段（Attention → FFN）而非单阶段融合？

标准 Transformer 层中 attention 和 FFN 的顺序是固定的（attn → ffn）。融合为单个阶段没有好处，因为 attention 的输出必须经过 `hc_post` + `hc_pre` 转换才能送入 FFN。两阶段设计使得：
- 每个阶段内部可以熔合 `hc_post`（上一子层）+ `hc_pre`（下一子层）
- `mhc_fused_post_pre` 的参数签名复用，attention 和 FFN 版本只差 `hc_fn`/`hc_scale`/`hc_base` 三组参数（L971-L977 vs L1005-L1021）

### 10.2 为什么 `hc_pre` 接受 `norm_weight` 参数而非内部持有 RMSNorm？

这使得 `hc_pre` 可以复用外部的 `attn_norm` / `ffn_norm` 权重，避免重复存储。更关键的是，`mhc_fused_post_pre` 融合算子可以同时执行归一化和流混合，需要直接访问 norm 权重——如果 norm 在 `hc_pre` 内部持有，就无法在融合路径中被外层访问。

### 10.3 为什么 `hc_post_alpha = 2.0` 硬编码（L876）？

这是一个固定的后处理缩放因子，与模型架构绑定而非可学习参数。将其硬编码可以：
- 减少 checkpoint 中需要存储的参数数量
- 避免每层重复存储相同的数值
- 可能在 Sinkhorn 归一化后起平衡作用（控制混合后激活值的量级）

### 10.4 CUDA 融合 vs Native 回退的精度差异

两个路径的计算图不同：融合路径在同一个 kernel 内完成 `norm + hc_pre`，而 native 路径先执行 `hc_pre` 再单独调用 `attn_norm`（L1039-L1042）。这可能导致极小的数值差异（由于融合顺序不同），但不会影响推理质量。

---

## 11. 量化与并行策略锚点

### 11.1 量化影响

`DeepseekV4DecoderLayer` 本身不执行量化计算，但其子模块受量化配置影响：

| 量化点 | 影响模块 | 说明 |
|--------|---------|------|
| FP8 激活量化 | `self.attn`（见 [[DeepseekV4MultiHeadLatentAttentionWrapper]]） | O 投影中对激活做 FP8 量化 |
| FP4/FP8 专家权重 | `self.ffn`（见 [[DeepseekV4MoE]]） | MegaMoE 使用 FP4，FusedMoE 支持 FP4/FP8/BF16 |
| KV cache 压缩 | `self.attn`（见 [[DeepseekCompressor]]） | compress_ratio > 1 时压缩 KV cache |
| HC 参数 | `hc_*` 参数 | **始终为 float32**，不受量化配置影响 |

### 11.2 并行策略影响

| 并行维度 | 对 DecoderLayer 的影响 | 文件位置 |
|---------|----------------------|---------|
| **张量并行（TP）** | Attention 和 MoE 的列/行切分由子模块处理，DecoderLayer 内部无感知 | `model.py:L863-L869` |
| **流水线并行（PP）** | 层被切分到不同 PP stage，层间传递的 `residual / post_mix / res_mix` 需要在 PP 边界通过通信传递 | `model.py:L1025` 返回的元组 |
| **专家并行（EP）** | 仅影响 `self.ffn`（MegaMoE 路径），DecoderLayer 主体无感知 | `model.py:L869` |
| **多 CUDA Stream 并行** | `topk_indices_buffer` 和 `aux_stream_list` 被传递给 `DeepseekV4Attention`（L865-L868），用于 MLA 中的多流 GEMM 重叠 | `model.py:L850-L851` |

---

## 12. 边界场景与异常处理

### 12.1 第一层处理（residual=None，L969-L979）

第一层解码器没有上一层的 `post_mix`/`res_mix`/`residual`，无法调用 `mhc_fused_post_pre`。代码退化为独立调用 `hc_pre`：
```python
residual = x        # 将输入 x 本身作为残差起始
x, post_mix, res_mix = self.hc_pre(x, ...)  # 独立的 hc_pre
```
这意味着第一层没有 "上一层的 hc_post" 需要执行，只做当前层的 hc_pre。

### 12.2 ROCm/XPU 兼容性（L1066-L1069）

在 AMD ROCm 或 Intel XPU 上，`_forward_native` 被选中。需要注意：
- native 路径的 `hc_pre` 调用未传入 `norm_weight`（L1039），这意味着 HC 算子内部的 Norm 融合不会生效
- 返回 `(x, None, None, None)`，调用方（[[DeepseekV4Model]]）接收后需正确处理 None 值（见 `model.py:L1220`）

### 12.3 `input_ids=None` 的情况

`input_ids` 仅在 MoE 路由需要 token ID（Hash MoE）时使用。当 `input_ids=None` 时（L1001 的 `self.attn(positions, x, None)` 和可能的 FFN 调用），路由将使用 fallback 方式（如仅使用 logits 分数）。这是安全的，因为：
- Attention 始终不需要 `input_ids`（`None` 仅作为占位符）
- FFN 的 Gate 路由在 `input_ids=None` 时会跳过 hash-based 路由（见 [[DeepseekV4MoE]]）

### 12.4 `post_mix`/`res_mix` 形状不匹配

如果 `hc_mult` 配置不对（例如 checkpoint 与 config 不一致），`mix_hc = (2 + hc_mult) * hc_mult` 计算出的形状与加载的权重不匹配。这会在 `self.hc_attn_fn = nn.Parameter(torch.empty((mix_hc, hc_dim), ...))` 处暴露，但只有在权重加载时才会体现为 shape mismatch 错误。

---

## 13. 交叉引用

| 相关笔记 | 关联说明 |
|---------|---------|
| [[DeepseekV4Attention]] | Attention 子模块的构造和参数组织 |
| [[DeepseekV4MultiHeadLatentAttentionWrapper]] | MLA 计算编排，被 `self.attn` 调用 |
| [[DeepseekV4MoE]] | MoE FFN 子模块，支持 MegaMoE / FusedMoE 双路径 |
| [[DeepseekV4Model]] | 调用 DecoderLayer 的模型主干，管理层间状态传递 |
| [[ArchitectureOverview]] | 整体架构层次分析，DecoderLayer 是第 3 层抽象 |
| [[QuantAndParallelStrategy]] | 跨模块的量化和并行策略总览 |

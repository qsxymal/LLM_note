# MHC 自定义算子层

## 位置

`vllm/model_executor/kernels/mhc/` — 4 个文件，4 个平台特定的实现，以及位于 `vllm/model_executor/layers/mhc.py` 的分发层。

## 四个算子

| 算子 | 输入 | 输出 | 用途 |
|----|-------|--------|---------|
| `mhc_pre` | 残差 (..., hc_mult, H) bf16 + 学习参数 | `post_mix`, `comb_mix`, `layer_input` (..., H) bf16 | 生成门控系数并将 hc_mult 流归约为一个流 |
| `mhc_post` | x, 残差, `post_layer_mix`, `comb_res_mix` | `residual_out` (..., hc_mult, H) | 将层输出注入每个 HC 残差流 |
| `mhc_fused_post_pre` | post 输入 + pre 参数 | `residual_cur`, `post_mix_cur`, `comb_mix_cur`, `layer_input_cur` | 单内核 post + 下一个 pre |
| `hc_head_fused` | hidden_states (..., hc_mult, H), hc_fn | out (..., H) | 模型头部的终端归约，将 hc_mult 折叠为 1 |

## 数学公式

### mhc_pre（混合系数 + Sinkhorn 归一化）

令 `T` = num_tokens, `hc_mult` = M, `H` = hidden_size, `M3 = M*2 + M*M`。

```
x = residual_flat → (T, M*H)  fp32
mixes = x @ fn^T               (T, M3)
rms = rsqrt( mean(x^2) + rms_eps )
mixes = mixes * rms

# Pre-mix（学习得到的 sigmoid 门控）
pre_mix  = sigmoid(mixes[:, :M]        * scale[0] + base[:M])        + pre_eps

# Post-mix（学习得到的 sigmoid 门控，带固定乘数）
post_mix = sigmoid(mixes[:, M:2*M]     * scale[1] + base[M:2*M])     * post_mult_value

# Comb-mix（softmax + Sinkhorn，每个 token 的双随机矩阵）
comb_logits = mixes[:, 2*M:].view(T, M, M) * scale[2] + base[2*M:].view(1, M, M)
comb_mix = softmax(comb_logits, dim=-1) + sinkhorn_eps
comb_mix = comb_mix / (comb_mix.sum(-2) + sinkhorn_eps)
for _ in range(sinkhorn_repeat - 1):     # 迭代地进行行/列归一化
    comb_mix = comb_mix / (comb_mix.sum(-1) + sinkhorn_eps)
    comb_mix = comb_mix / (comb_mix.sum(-2) + sinkhorn_eps)

# 通过加权平均将 hc_mult 流归约为单流
layer_input = sum_i pre_mix_i * residual_i  →  (T, H) bf16
```

Sinkhorn 归一化迭代地迫使 comb_mix 矩阵成为双随机矩阵（行和列各自和为 1）。这是 MHC 混合机制的关键组成部分。

### mhc_post（残差混合）

```
out_j = post_layer_mix_j * x  +  sum_i comb_res_mix_ij * residual_i
```

其中：
- `post_layer_mix` 是每个流的标量门控（hc_mult 个值）
- `comb_res_mix` 是一个 (hc_mult × hc_mult) 混合矩阵
- `x` 是层输出（广播到所有 hc_mult 流）

爱因斯坦记号等价形式：`out[jh] = post_mix[j] * x[h] + sum_i comb_mix[ij] * residual[ih]`

### hc_head_fused（终端归约）

与 mhc_pre 的 pre-mix + 加权求和核心逻辑相同，但没有 post_mix/comb_mix 输出：

```
x_flat = hidden_states.flatten(-2)    # (T, M*H)
x_normed = rmsnorm_nw(x_flat)          # 无权重的 RMSNorm
mixes = linear(x_normed, hc_fn)        # (T, M)
pre = sigmoid(mixes * scale + base) + eps
out[h] = sum_i pre[i] * hidden_states[i, h]   # 加权归约
```

## 张量形状表

### mhc_pre / mhc_pre_tilelang

| 张量 | 形状 | 数据类型 | 描述 |
|--------|-------|-------|-------------|
| `residual` | (..., M, H) | bf16 | HC 残差输入 |
| `fn` | (M3, M*H) | fp32 | 混合 logits 的权重矩阵 |
| `hc_scale` | (3,) | fp32 | 每种混合类型的 logit 缩放 |
| `hc_base` | (M3,) | fp32 | 每种混合类型的 logit 偏置 |
| `norm_weight` | (H,) | bf16 | 可选融合 RMSNorm 权重 |
| `post_mix` | (..., M, 1) | fp32 | Post-mix 门控系数 |
| `comb_mix` | (..., M, M) | fp32 | 双随机 comb 混合矩阵 |
| `layer_input` | (..., H) | bf16 | 归约后的单流输出 |

其中 `M = hc_mult`, `M3 = 2*M + M*M`。

### mhc_post / mhc_post_tilelang

| 张量 | 形状 | 数据类型 | 描述 |
|--------|-------|-------|-------------|
| `x` | (..., H) | bf16 | 层输出 |
| `residual` | (..., M, H) | bf16 | HC 残差输入 |
| `post_layer_mix` | (..., M, 1) | fp32 | 每流 post 乘数 |
| `comb_res_mix` | (..., M, M) | fp32 | Comb 混合矩阵 |
| `out` | (..., M, H) | bf16 | 更新后的残差 |

### mhc_fused_post_pre_tilelang

返回 mhc_post 和 mhc_pre 的所有输出：

| 张量 | 形状 | 数据类型 |
|--------|-------|-------|
| `residual_cur` | (..., M, H) | bf16 |
| `post_mix_cur` | (..., M, 1) | fp32 |
| `comb_mix_cur` | (..., M, M) | fp32 |
| `layer_input_cur` | (..., H) | bf16 |

### hc_head_fused

| 张量 | 形状 | 数据类型 |
|--------|-------|-------|
| `hidden_states` | (..., M, H) | bf16 |
| `hc_fn` | (M, M*H) | fp32 |
| `hc_scale` | (1,) | fp32 |
| `hc_base` | (M,) | fp32 |
| `out` | (..., H) | bf16 |

## 四个实现文件

### `torch.py` — 纯 PyTorch 参考实现（107 行）

- `mhc_pre_torch`：用于正确性测试的直接 PyTorch 实现。
- `mhc_post_torch`：使用 `torch.einsum("...ij,...ih->...jh", comb, residual)`。
- `mhc_fused_post_pre_torch`：未定义（此文件中缺少；仅有单独算子）。
- `hc_head_fused_torch`：未定义（此文件中缺少）。

这是所有其他实现的数值验证**基准参考**。

### `tilelang.py` — TileLang CUDA 自定义算子（547 行）

NVIDIA GPU 的主要高性能路径。所有 4 个算子均通过 `direct_register_custom_op` 注册，并带有用于元设备形状推断的 fake 实现。

**mhc_pre_tilelang 架构：**

1. **Split-k GEMM**（来自 DeepGEMM 的 `tf32_hc_prenorm_gemm`）：执行大型 `x @ fn^T` 矩阵乘法，同时作为副产品计算 `sum(x^2)`。k 维度被拆分为 `n_splits` 块以最大化 SM 利用率。输出两个缓冲区：
   - `gemm_out_mul`：(n_splits, T, M3) — 部分 GEMM 结果
   - `gemm_out_sqrsum`：(n_splits, T) — 部分平方和

2. **big_fuse 内核**（`mhc_pre_big_fuse_tilelang`）：一个单一的 TileLang 内核，消费 split-k 输出并完成所有其他工作：
   - 从所有 split 的 `gemm_out_sqrsum` 累加 `rms`
   - 从所有 split 的 `gemm_out_mul` 累加 `mixes`，然后乘以 `rms`
   - 计算 `post_mix`（warp 0-31，线程绑定 < 32）
   - 计算带有完整 Sinkhorn 迭代的 `comb_mix`
   - 计算 `pre_mix` 和到 `layer_input` 的加权和归约（warp 32+）

**mhc_post_tilelang：** 执行 einsum + 广播的直接 TileLang 内核。

**mhc_fused_post_pre_tilelang：** post + pre 的完整融合：
- 小批次（<= 16 个 token）：单个 `mhc_fused_tilelang` 内核，在一个 pass 中计算 post 映射、写入更新后的残差并执行 GEMM FMA 用于 pre
- 大批次（> 16 个 token）：顺序执行 `mhc_post_tilelang` + split-k GEMM + `mhc_pre_big_fuse_tilelang` — 避免寄存器压力

**hc_head_fused_tilelang：** 两遍 TileLang 内核，使用跨线程归约器避免将中间张量实例化到全局内存。

### `aiter.py` — AMD ROCm Aiter 封装（139 行）

- 委托给 `rocm_aiter_ops.mhc_pre` / `rocm_aiter_ops.mhc_post`（来自 `vllm._aiter_ops`）。
- 约束条件：要求 `hidden_size % 256 == 0`。
- 由于已知的大 token 数量精度错误（`aiter commit b639cb6`），目前在分发层中**已禁用**。
- 仅提供 `mhc_pre` 和 `mhc_post` 算子；没有 fused_post_pre 或 hc_head。

### `triton.py` — Triton HC Head 回退（175 行）

- `rmsnorm_nw`：无权重 RMSNorm Triton 内核（用于 hc_head）。
- `hc_head_reduce_triton_kernel`：x → x_flat → rmsnorm → linear → sigmoid → 加权和。
- 使用一个 Triton 内核 `_hc_head_reduce_store_kernel` 进行跨 hc_mult 流的归约。
- 仅提供以 `hc_head_triton` 注册的 `hc_head` 算子。

## 平台特定分发

分发由 `vllm/model_executor/layers/mhc.py` 通过 `CustomOp` 框架控制：

| 平台 | mhc_pre | mhc_post | mhc_fused_post_pre | hc_head |
|----------|---------|----------|--------------------|---------|
| **CUDA**（NVIDIA） | `mhc_pre_tilelang` | `mhc_post_tilelang` | `mhc_fused_post_pre_tilelang` | `hc_head_fused_kernel_tilelang` |
| **HIP**（AMD ROCm） | `mhc_pre_torch`（aiter 已禁用） | `mhc_post_torch`（aiter 已禁用） | 不可用（抛出异常） | `hc_head_triton` |
| **Native**（CPU） | 不可用 | 不可用 | 不可用 | 不可用 |

所有算子均通过 `direct_register_custom_op`（`vllm.utils.torch_utils`）注册为 PyTorch 自定义算子，这使得它们拥有用于 torch.compile / 元设备追踪的 fake 实现。

## TileLang 架构：Split-k GEMM + Big Fuse

NVIDIA GPU 上 pre 块的核心性能策略：

```
residual (T, M, H) bf16
        │
        ▼
┌─────────────────────────────────────────────────┐
│ tf32_hc_prenorm_gemm  (split-k, from DeepGEMM)  │
│   x @ fn^T  +  x^2 sum  in tf32                 │
│   k-dim split into n_splits                     │
│   Consumer grid: n_splits = f(64, M*H, T')      │
└────────────┬───────────────────────┬─────────────┘
             │                       │
    gemm_out_mul            gemm_out_sqrsum
    (n_splits, T, M3)       (n_splits, T)
             │                       │
             ▼                       ▼
┌─────────────────────────────────────────────────────┐
│ mhc_pre_big_fuse_tilelang  (fully fused kernel)     │
│   per-token:                                        │
│     1. rms = rsqrt( sum(sqrsum_splits) / D + eps ) │
│     2. mixes = sum(mul_splits) * rms                │
│     3. post_mix = sigmoid(mixes[M:2M]) * mult       │
│     4. comb_mix = softmax(mixes[2M:]) → Sinkhorn    │
│     5. pre_mix = sigmoid(mixes[:M]) + eps           │
│     6. layer_input = sum(pre * residual[stream])    │
│        (optionally fused with RMSNorm)              │
└─────────────────────┬───────────────────────────────┘
                      │
        post_mix, comb_mix, layer_input
```

split 数量是自动调优的：

```python
# From _tilelang_ops.py
def compute_num_split(block_k, k, grid_size):
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    split_k = n_sms // grid_size
    num_block_k = cdiv(k, block_k)
    split_k = min(split_k, num_block_k // 4)   # 避免对小 k 过度分拆
    return max(split_k, 1)
```

这确保了 GEMM 块均匀分布在所有可用的 SM 上，同时避免了许多微小的 split 带来的过多开销。

## 融合策略

### Post-Pre 融合（`mhc_fused_post_pre_tilelang`）

基于 `num_tokens ≤ fma_token_threshold`（= 16）的两条路径：

**小批次（完全融合）：**
```
mhc_fused_tilelang  (single kernel)
  ┌───────────────────────────────────────────────┐
  │ For each token, thread owns h_iters elements: │
  │   new_r[j] = pm[j]*x[h] + sum cm[k,j]*res[k] │
  │   residual_out[j,h] = new_r[j]                │
  │   acc[n] += sum w[n,j,h] * new_r[j]           │
  │   sqr += sum(new_r[j]^2)                      │
  │ warp_reduce_sum acc/sqr → shared              │
  │ warp 0 finalizes → gemm_out, rp_out           │
  └───────────────────────────────────────────────┘
```
- 自动启发式：如果 `tokens < 8` 则 `tile_n = 2`，否则为 `3`
- 如果 `(tokens < 8 and H ≤ 4096)` 则 `n_splits = 8`，否则为 `4`
- 完全消除了 post 残差的写回和重新读取。

**大批次（部分融合）：**
- `mhc_post_tilelang` → 将 residual_cur 写入 HBM
- `tf32_hc_prenorm_gemm` → gemm_out_mul + gemm_out_sqrsum
- `mhc_pre_big_fuse_tilelang` → post_mix + comb_mix + layer_input

在较大的批次大小时，完全融合会耗尽寄存器，因此 split-k GEMM 和 big_fuse 内核仍然是最优路径。

### Norm 融合

当向 `mhc_pre_tilelang` 提供 `norm_weight` 时：

- `mhc_pre_big_fuse_with_norm_tilelang` 将 RMSNorm 融合到 layer_input 写入路径中：
  - **第 1 遍**：计算加权和，在共享内存中存储临时 bf16 输出，同时累加每个位置的平方和
  - **计算**：`rsqrt = 1 / sqrt(mean(sumsq) + norm_eps)`
  - **第 2 遍**：重新读取临时值，通过 `rsqrt * norm_weight` 缩放，写入 HBM
- 消除了中间未归一化张量的实例化和单独的 RMSNorm 内核启动。

## 设计原理

### 为什么使用 Split-k？

GEMM `x @ fn^T` 的形状为 `(T, M*H) @ (M3, M*H)^T = (T, M3)`。隐藏维度 `M*H` 可能非常大（模型扩展），而 `T` 可能很小（推理批次）。标准的 (token, M3) 二维网格会使得 SM 利用率不足。Split-k 将归约维度（k = M*H）分割到多个 SM 上，使每个 SM 都有工作可做，然后 big_fuse 内核进行廉价的 split 间归约。

### 为什么融合 Post-Pre 有两条路径？

融合内核（`mhc_fused_tilelang`）将 post 和 pre GEMM FMA 打包到一个内核中，但使用了大量寄存器。在小 T 时这是可容忍的（每个线程的高算术强度）。在大 T 时，同时持有所有片带来的寄存器压力会限制驻留率，因此顺序路径（三个独立内核）表现更好。

### 为什么使用 Sinkhorn 归一化？

comb_mix 矩阵必须是双随机的（行和列各自和为 1），才能表示输入和输出 HC 流之间的有效软分配。Sinkhorn 算法通过迭代地行/列归一化非负矩阵来实现这一点。迭代次数是一个可配置的超参数（`sinkhorn_repeat`）。

### 平台策略

- **CUDA**：TileLang 提供了一个 JIT 编译框架，目标为原生 PTX，完全控制寄存器使用、共享内存和 warp 级原语——非常适合复杂的融合模式。DeepGEMM 处理 tf32 下的 split-k GEMM。
- **ROCm**：Aiter 是 AMD 原生的内核库，但目前在大 token 数量时存在精度问题。回退到纯 PyTorch（torch.py）作为安全桥梁。
- **ROCm 上的 hc_head**：Triton 提供了一个可移植的实现，因为它同时运行在 CUDA 和 HIP 后端上。hc_head 内核更简单（没有 Sinkhorn，没有 gemm_out_mul 缓冲），因此 Triton 已经足够。

## 相关笔记

- [[V4Architecture]] — DeepSeek V4 整体架构和 MHC 层的作用
- [[TileLangKernels]] — vLLM 中的 TileLang JIT 内核基础设施
- [[DeepGEMMIntegration]] — DeepGEMM split-k GEMM 封装（`vllm/utils/deep_gemm.py`）
- [[CustomOpDispatch]] — 用于平台分发的 CustomOp 框架

# DeepSeek V4 MoE 门控路由

## 概述

DeepSeek V4 使用一个学习得到的门控网络将每个 token 路由到一组专家子集。门控是一个线性投影（`GateLinear`），从 `hidden_size` 映射到 `n_routed_experts`，生成原始 logits。然后，这些 logits 通过一个**评分函数**生成每个专家的分数，并选择得分最高的 top-k 个专家。

路由流程位于 `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`（`fused_topk_bias()` 函数），由 `DeepseekV4MoE.forward()` 在 `vllm/models/deepseek_v4/nvidia/model.py` 中调用。

---

## 评分函数

支持三种评分函数，由 `scoring_func` 配置属性控制（默认值：`"sqrtsoftplus"`）。

### 1. softmax

`ops.topk_softmax` -- 对门控 logits `(M, n_experts)` 应用标准 softmax，生成概率分布。返回 top-k 的索引及其概率。

被许多传统 MoE 模型（Mixtral、DeepSeek V2 等）使用。不是 DeepSeek V4 的默认配置。

### 2. sigmoid

`ops.topk_sigmoid` -- 对门控 logits 逐元素应用 sigmoid。每个专家获得一个独立的 (0, 1) 范围内的分数。选择 sigmoid 分数最高的 top-k 个专家。

被许多 MoE 模型广泛使用。不是 DeepSeek V4 的默认配置。

### 3. sqrtsoftplus（默认）

`ops.topk_hash_softplus_sqrt`（CUDA 自定义算子）或 `_topk_softplus_sqrt_torch`（PyTorch 回退实现）。

评分函数为：

```
score = sqrt(softplus(x))   其中 softplus(x) = log(1 + exp(x))
```

这是 DeepSeek V4 的默认配置。当启用 MegaMoE 时，仅支持 `sqrtsoftplus`（`DeepseekV4MoE.__init__` 在 `scoring_func != "sqrtsoftplus"` 时抛出异常）。

---

## sqrtsoftplus 详解

**数学公式**：`score_i = sqrt(ln(1 + exp(gate_i)))`

与 softmax（产生耦合的概率分布）或 sigmoid（取值范围在 0 到 1 之间）不同，sqrtsoftplus 具有以下有用特性：

- **无界且单调**：每个专家的分数独立于其他专家。与 softmax 不同，专家之间不存在对概率质量的"竞争"。
- **负 logits 无梯度消失**：对于非常负的 `x`，softplus 近似为 `exp(x)`；对于非常正的 `x`，近似为 `x`，从而在任何地方都能提供平滑的梯度。平方根运算压缩了动态范围。
- **可与 `e_score_correction_bias` 组合使用**：由于分数是按专家独立计算的，添加一个学习得到的偏置直接改变了每个专家的"激活阈值"，这在 softmax 下更难解释。

`vllm_topk_softplus_sqrt` 函数（`fused_topk_bias_router.py` 第 106 行）会分发到以下两种路径之一：

- **CUDA 路径**：`ops.topk_hash_softplus_sqrt(...)` -- 一个融合的 CUDA 内核，通过 `torch.library.impl` 在 `csrc/moe/torch_bindings.cpp` 中注册。内核实现在 `csrc/moe/topk_softplus_sqrt_kernels.cu`。
- **PyTorch 回退路径**（XPU/CPU）：`_topk_softplus_sqrt_torch(...)` -- 纯 PyTorch 实现。

**CUDA 内核融合了**：softplus + sqrt + 偏置加法 + topk 选择 + 哈希查找 + 重归一化，全部在一个内核启动中完成。这避免了在 HBM 上实例化中间张量。

---

## 三种路由模式

路由模式在 `DeepseekV4MoE.__init__()` 中确定（model.py 第 505-524 行）：

### 哈希 MoE

当 `extract_layer_index(prefix) < config.num_hash_layers` 时激活，即早期层使用基于哈希的路由。

- **不创建** `e_score_correction_bias`。
- 分配一个形状为 `(vocab_size, top_k)` 的 `tid2eid` 表作为 `nn.Parameter`（冻结，`requires_grad=False`），预先填充随机的专家 ID（使用 `torch.randint` 以避免在 dummy 模式中访问垃圾值内存）。
- 在路由过程中：从表中**查找**专家 ID（`hash_indices_table[input_tokens]`），而不是通过 topk 选择。这意味着路由是确定性的且静态的——这些层没有学习得到的门控。

当 `use_mega_moe` 为 True 时，`tid2eid` 的数据类型为 `torch.int64`，否则为 `torch.int32`。

### Noaux TC

当 `config.topk_method == "noaux_tc"` 时激活。

- 创建一个形状为 `(n_routed_experts,)`、数据类型为 `float32` 的 `e_score_correction_bias` 作为 `nn.Parameter`（冻结，`requires_grad=False`）。
- 该偏置在**训练期间学习**，用于在无需辅助损失的情况下平衡专家负载。
- 该偏置仅用于**选择**——实际权重值来自于无偏置的分数（参见 [[#偏置处理]]）。

"Noaux TC" 代表**无辅助 token 计数器**。门控在训练期间不产生辅助负载均衡损失；相反，`e_score_correction_bias` 是用于平衡专家选择的学习机制。

### 标准模式

既不是哈希 MoE 也不是 noaux TC。使用配置的评分函数进行普通的 top-k 路由。没有 `e_score_correction_bias` 和 `tid2eid` 表。

---

## 偏置处理

`e_score_correction_bias` 是一个形状为 `(n_routed_experts,)`、数据类型为 `float32` 的一维张量。

**关键设计决策**：偏置仅用于**选择**，不用于权重计算。

```python
# Line 75-81 in fused_topk_bias_router.py
# Bias is used for expert SELECTION only, not for weight computation.
# Using biased scores as weights flattens the distribution when the bias
# is near-uniform (e.g., DSv4-Flash where all biases ≈ 8.08).
if e_score_correction_bias is not None:
    scores_for_choice = scores + e_score_correction_bias.float()
else:
    scores_for_choice = scores

# Selection uses biased scores:
_, indices = torch.topk(scores_for_choice, k=topk, dim=-1)
# Weights are gathered from UNBIASED scores:
weights = scores.gather(1, indices)
```

**设计理由**：在 DeepSeek V4-Flash 中，学习得到的偏置值都大约是 8.08（接近均匀）。如果偏置直接加到权重上，每个专家将获得几乎相同的权重，从而压平了分布，丢失了门控网络的信号。通过仅将偏置用于选择，门控网络的相对分数仍然决定了权重的大小。

偏置在 `fused_topk_bias()` 中也通过 ROCm AITER 路径（第 245-256 行）和最终回退路径（第 267-289 行）进行处理，始终遵循相同的模式：偏置加到 `scores_for_choice`，权重从无偏置的 `scores` 中收集。

---

## 纯 PyTorch 回退与自定义 CUDA 内核

### PyTorch 回退（`_topk_softplus_sqrt_torch`，第 60-103 行）

在 XPU/CPU 平台上使用。完整的流程：

```
1. scores = sqrt(softplus(gating_output.float()))
2. 如果有偏置：scores_for_choice = scores + bias.float()
3a. 如果是哈希：expert_ids = hash_indices_table[input_tokens]   （查找）
3b. 如果是标准：indices = torch.topk(scores_for_choice, k=topk)
4. weights = scores.gather(1, indices)                        （无偏置）
5. 如果需要重归一化：weights /= weights.sum(dim=-1).clamp(min=1e-20)
6. weights *= routed_scaling_factor
```

注意：`scores` 在 `sqrt(softplus(...))` 计算之前始终转换为 `float32` 以保证数值稳定性。

### CUDA 自定义算子（`ops.topk_hash_softplus_sqrt`）

在 `csrc/moe/torch_bindings.cpp` 中以 `topk_softplus_sqrt` 名称注册为自定义算子。分发到 `csrc/moe/topk_softplus_sqrt_kernels.cu` 中的 `dispatch_topk_softplus_sqrt_launch`。

支持 float32、float16（`__half`）和 bfloat16（`__nv_bfloat16`）输入数据类型。

**融合操作**：softplus + sqrt + 偏置加法 + topk 选择 + 哈希表查找 + 重归一化 + 缩放。全部在一个内核中完成以最小化全局内存访问。

### Softmax/Sigmoid 的 CUDA 路径

当评分函数为 `softmax` 或 `sigmoid` 时，使用相应的算子（`ops.topk_softmax`、`ops.topk_sigmoid`）。`topk_sigmoid` 路径还有 ROCm AITER 路径（`rocm_aiter_ops.biased_grouped_topk`）。

### `fused_topk_bias()` 中的回退路径（第 258-289 行）

当启用了 `rocm_aiter_ops` 但条件不满足时，或因其他原因自定义算子不可用时，使用纯 PyTorch 回退：

```python
scores = F.softplus(gating_output).sqrt()  # 针对 sqrtsoftplus
scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
topk_indices = torch.topk(scores_for_choice, k=topk, sorted=True)[1]
topk_weights = scores.gather(1, topk_indices)
```

注意使用 `sorted=True` 以保证批次不变性（确定性专家排序）。

---

## `fused_topk_bias()` 如何被 `DeepseekV4MoE.forward()` 调用

### MegaMoE 路径（`use_mega_moe=True`）

```python
# model.py lines 613-627
router_logits, _ = self.gate(hidden_states)                         # [M, n_experts]
topk_weights, topk_ids = fused_topk_bias(
    hidden_states=hidden_states,
    gating_output=router_logits,
    scoring_func=self.scoring_func,                                  # "sqrtsoftplus"
    e_score_correction_bias=self.gate.e_score_correction_bias.data,  # 或 None
    topk=self.n_activated_experts,                                   # num_experts_per_tok
    renormalize=self.renormalize,                                    # config.norm_topk_prob
    indices_type=self.hash_indices_dtype,                            # megamoe 为 int64，否则为 int32
    input_tokens=input_ids,                                          # [M] 或 None
    hash_indices_table=self.gate.tid2eid,                            # 或 None
    routed_scaling_factor=self.routed_scaling_factor,
)
final_hidden_states = self.experts(hidden_states, topk_weights, topk_ids, ...)
```

`gate` 是一个 `GateLinear` 实例（从 `vllm/model_executor/layers/fused_moe/router/gate_linear.py` 导入）。它是一个三层 GEMM 层：

1. **第一层**（DSV3 专用内核，SM90+，批次 <= 16）：`ops.dsv3_router_gemm`
2. **第二层**（cuBLAS bf16 x bf16 -> fp32）：`torch.mm(x, W.T, out_dtype=float32)`
3. **第三层**（通过 `ReplicatedLinear` 回退）：`F.linear`

`GateLinear.forward()` 的输出数据类型为 `torch.float32`（在 model.py 第 600 行设置：`router_logits_dtype=torch.float32`）。

### FusedMoE 路径（`use_mega_moe=False`）

门控和路由在 `FusedMoE` 类内部处理（`vllm/model_executor/layers/fused_moe`）。门控先运行：

```python
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
    input_ids=input_ids,
)
```

`FusedMoE` 内部使用 `FusedTopKBiasRouter`（如果设置了 `scoring_func`）来执行路由。

---

## 形状/数据类型表

### 门控（线性投影）

| 张量 | 形状 | 数据类型 | 描述 |
|--------|-------|-------|-------------|
| `hidden_states` | `(M, hidden_size)` | bf16/fp16 | 来自上一层的 token 隐藏状态 |
| `gate.weight` | `(n_routed_experts, hidden_size)` | bf16/fp16 | 门控投影权重 |
| `router_logits` | `(M, n_routed_experts)` | float32 | 原始门控 logits（门控输出） |

### 路由输出

| 张量 | 形状 | 数据类型 | 描述 |
|--------|-------|-------|-------------|
| `topk_weights` | `(M, top_k)` | float32 | 最终路由权重（选择 + 重归一化 + 缩放后的分数） |
| `topk_ids` | `(M, top_k)` | int32 / int64 | 选中的专家 ID；MegaMoE 为 int64，否则为 int32 |

### 偏置和哈希表

| 张量 | 形状 | 数据类型 | 描述 |
|--------|-------|-------|-------------|
| `e_score_correction_bias` | `(n_routed_experts,)` | float32 | 用于专家选择均衡的学习偏置 |
| `tid2eid`（哈希表） | `(vocab_size, top_k)` | int32 / int64 | 哈希 MoE 层的 Token ID -> 专家 ID 查找表 |

### 中间张量（PyTorch 回退）

| 张量 | 形状 | 数据类型 | 描述 |
|--------|-------|-------|-------------|
| `scores` | `(M, n_routed_experts)` | float32 | `sqrt(softplus(router_logits))` |
| `scores_for_choice` | `(M, n_routed_experts)` | float32 | `scores + bias`（仅用于选择） |
| `token_expert_indices` | `(M, top_k)` | int32 | 自定义算子的临时张量（PyTorch 回退不使用） |

---

## 设计原理

### sqrtsoftplus vs softmax

| 属性 | softmax | sigmoid | sqrtsoftplus |
|----------|---------|---------|--------------|
| 专家竞争 | 是（总和=1） | 否（独立） | 否（独立） |
| 分数范围 | (0, 1) | (0, 1) | (0, inf) |
| 负 logits 的梯度 | 接近 0 时很小 | 接近 0 时很小 | 平滑（softplus 对负 x 约为 exp(x)） |
| 偏置语义 | 移动概率质量 | 移动阈值 | 移动阈值（自然） |
| 数值稳定性 | 需要 `-max` 技巧 | 稳定 | 稳定（softplus 无饱和） |

**DeepSeek V4 为什么选择 sqrtsoftplus**：

- **独立的专家评分**：与 softmax 不同，每个专家的分数是独立的。当使用 `e_score_correction_bias` 独立地偏置每个专家的选择阈值时，这一点至关重要。
- **无界分数**：分数没有固定的上限，这允许门控网络表达高置信度。
- **梯度特性**：Softplus 为所有输入提供平滑的梯度，避免了 sigmoid 在非常负或非常正值时的饱和问题。平方根运算压缩了动态范围。
- **MegaMoE 兼容性**：自定义 CUDA 内核融合了整个流程，这对 MegaMoE 的性能至关重要。

### 偏置仅用于选择

仅将偏置用于选择而不用于权重的模式，是由 DeepSeek V4-Flash 中观察到的行为所驱动的：

- 在 V4-Flash 中，学习得到的 `e_score_correction_bias` 值都约为 8.08——一个接近均匀的偏置。
- 如果应用于权重：`weights = (scores + 8.08).gather(...)`，均匀偏移占主导地位，导致所有选中的专家具有几乎相同的权重。门控网络的信号就丢失了。
- 如果仅用于选择：`scores_for_choice = scores + 8.08` 改变了哪些专家被选中，而 `weights = scores.gather(1, indices)` 保留了来自门控网络的相对大小。

这种将选择（选中哪些专家）与加权（每个专家贡献多少）分离的做法是关键的设计思想。偏置扮演的是**专家选择阈值调节器**的角色，而不是权重修改器。

---

## 相关文件和类

- `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` -- `fused_topk_bias()` 和 `FusedTopKBiasRouter`
- `vllm/model_executor/layers/fused_moe/router/gate_linear.py` -- `GateLinear`
- `vllm/models/deepseek_v4/nvidia/model.py` -- `DeepseekV4MoE`
- `vllm/model_executor/layers/fused_moe` -- `FusedMoE`
- `csrc/moe/topk_softplus_sqrt_kernels.cu` -- 融合 sqrtsoftplus + topk 的 CUDA 内核
- `csrc/moe/torch_bindings.cpp` -- 自定义算子注册

相关笔记：[[MegaMoE]], [[FusedMoE]]

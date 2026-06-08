# DeepSeek V4 MoE（混合专家）

**文件路径：** `vllm/models/deepseek_v4/nvidia/model.py`（核心），`vllm/models/deepseek_v4/amd/model.py`（AMD 回退），`vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`（输入准备），`vllm/models/deepseek_v4/quant_config.py`（量化分发）

## 1. 模块定位

- **职责：** 实现 DeepSeek V4 的 MoE FFN 层，支持两条互斥的执行路径：**MegaMoE**（基于 DeepGEMM 的高性能 FP4 内核，需 SM100 Blackwell GPU）和 **FusedMoE**（标准 vLLM 融合 MoE 层）。
- **上游调用者：** `DeepseekV4DecoderLayer`（`vllm/models/deepseek_v4/nvidia/model.py:L869`），在 HC 后处理之后调用 `self.ffn(x, input_ids)`。
- **下游依赖：**
  - `vllm.model_executor.layers.fused_moe`：标准 FusedMoE 实现
  - `vllm.third_party.deep_gemm`：DeepGEMM MegaMoE 内核（仅 SM100）
  - `vllm.model_executor.layers.quantization.fp8` / `mxfp4` / `modelopt`：量化方法
  - `vllm.model_executor.layers.fused_moe.router`：路由计算

## 2. 两条执行路径

DeepSeek V4 MoE 有两条完全不同的执行路径，通过 `use_mega_moe` 标志选择。

### 路径一：MegaMoE（`deep_gemm_mega_moe`）

| 属性 | 值 |
|------|-----|
| 触发条件 | `kernel_config.moe_backend == "deep_gemm_mega_moe"`（`model.py:L465`） |
| 硬件要求 | SM100 (Blackwell) GPU（`model.py:L273`） |
| 专家精度 | 仅 FP4（`expert_dtype == "fp4"`）（`model.py:L488`） |
| 评分函数 | 仅 `sqrtsoftplus`（`model.py:L484`） |
| 并行策略 | 专家并行（EP）（`model.py:L468-473`） |
| 激活值 | FP8 E8M0 动态量化（准备阶段，`prepare_megamoe.py`） |
| 权重格式 | FP4（`uint8` 存储）+ ue8m0 缩放因子 |

### 路径二：FusedMoE（标准 vLLM MoE）

| 属性 | 值 |
|------|-----|
| 触发条件 | `moe_backend != "deep_gemm_mega_moe"` |
| 硬件要求 | 通用 |
| 专家精度 | FP4 / FP8 / BF16（取决于 `expert_dtype` 和 `quant_config`） |
| 并行策略 | 张量并行（TP），按 TP rank 分配专家（`model.py:L579-583`） |
| 量化方法 | `Mxfp4MoEMethod`（FP4）、`Fp8MoEMethod`（FP8）、`ModelOptNvFp4FusedMoE`（NVFP4，见 `quant_config.py:L132-153`） |

---

## 3. 对外接口与数据类型契约

### `DeepseekV4MoE.forward()`

```python
def forward(
    self,
    hidden_states: torch.Tensor,    # [num_tokens, *extra_dims, hidden_size], bf16
    input_ids: torch.Tensor | None, # [num_tokens], int32/int64 (hash MoE 必需)
) -> torch.Tensor
```

| 位置 | Tensor | shape | dtype | device | 说明 |
|------|--------|-------|-------|--------|------|
| 输入 | `hidden_states` | `(num_tokens, hc_mult, hidden_size)` 或 `(num_tokens, hidden_size)` | bf16 | GPU | DecoderLayer 的 HC 后处理输出 |
| 输入 | `input_ids` | `(num_tokens,)` | int32/int64 | CPU/GPU | hash MoE 路由时传入 token ID |
| 输出 | `final_hidden_states` | 与输入 `hidden_states` 相同 | bf16 | GPU | MoE 输出 + 共享专家输出 |

**关键 Tensor 的 shape/dtype 速查：**

| 变量名 | shape | dtype | 来源 |
|--------|-------|-------|------|
| `router_logits` | `(num_tokens, n_routed_experts)` | float32 | `self.gate(hidden_states)`（`model.py:L613`） |
| `topk_weights` | `(num_tokens, top_k)` | bf16（FusedMoE）/ float32（MegaMoE） | `fused_topk_bias(...)`（`model.py:L614`） |
| `topk_ids` | `(num_tokens, top_k)` | int32/int64 | `fused_topk_bias(...)`（`model.py:L614`） |
| `x_fp8`（MegaMoE 输入准备） | `(num_tokens, hidden_size)` | fp8 | `prepare_megamoe_inputs` 中量化（`prepare_megamoe.py`） |
| `x_sf`（MegaMoE 缩放因子） | `(num_tokens, hidden_size//128)` | int32（打包） | `prepare_megamoe_inputs`（`prepare_megamoe.py`） |
| `w13_weight`（MegaMoE） | `(n_local_experts, 2*intermediate_size, hidden_size//2)` | uint8 | checkpoint 加载（`model.py:L165`） |
| `w2_weight`（MegaMoE） | `(n_local_experts, hidden_size, intermediate_size//2)` | uint8 | checkpoint 加载（`model.py:L188`） |
| `w13_weight_scale`（MegaMoE） | `(n_local_experts, 2*intermediate_size, hidden_size//32)` | uint8 | checkpoint 加载（`model.py:L176`） |
| `w2_weight_scale`（MegaMoE） | `(n_local_experts, hidden_size, intermediate_size//32)` | uint8 | checkpoint 加载（`model.py:L199`） |

---

## 4. 内部类关系图

```
DeepseekV4MoE (nvidia/model.py:L453 / amd/model.py:L110)
  ├── GateLinear — 路由网络（hidden_size → n_routed_experts）
  ├── DeepseekV4MegaMoEExperts (nvidia/model.py:L138) — [仅 MegaMoE 路径]
  │     ├── w13_weight / w13_weight_scale — FP4 gate+up 融合权重
  │     ├── w2_weight / w2_weight_scale — FP4 down 投影权重
  │     ├── finalize_weights() — 转换为 DeepGEMM 布局
  │     └── _run_mega_moe() — 调用 deep_gemm.fp8_fp4_mega_moe
  ├── FusedMoE (vllm FusedMoE 层) — [仅 FusedMoE 路径]
  │     └── 包含 n_local_experts 个专家，使用 FusedMoE kernel
  └── DeepseekV4MLP (nvidia/model.py:L68) — 共享专家
        ├── gate_up_proj (MergedColumnParallelLinear) — SiLU 门控
        ├── act_fn (SiluAndMul)
        └── down_proj (RowParallelLinear)
```

---

## 5. 关键设计决策及源码证据

### 5.1 为什么 MegaMoE 需要强制开启专家并行？

```python
self.use_mega_moe = (vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe")
if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
    raise NotImplementedError(...)
```
（`model.py:L465-473`）

**原因：** DeepGEMM 的 MegaMoE 内核（`deep_gemm.fp8_fp4_mega_moe`）内部已经集成了 expert parallel 的 all-to-all 通信和 expert dispatch。它通过 `get_symm_buffer_for_mega_moe` 获取通信缓冲区（`model.py:L334`），并在内核内部处理 `ep_group` 的同步。如果不使用 EP，则各 rank 持有全部专家权重，这违背了 DeepGEMM 的设计假设，也无法发挥 SM100 上 NVLink 的快速通信能力。

### 5.2 为什么 MegaMoE 权重用 uint8 存储而不是 FP4？

```python
self.w13_weight = nn.Parameter(
    torch.zeros(num_local_experts, 2 * intermediate_size, hidden_size // 2,
                dtype=torch.uint8), requires_grad=False)
```
（`model.py:L165-173`）

**原因：** PyTorch 没有原生 `fp4` / `mxfp4` 数据类型。`uint8` 作为原始字节容器，每个 uint8 元素存储两个 FP4 值（FP4 是 4-bit，所以 `hidden_size // 2`）。缩放因子同理以 `uint8` 存储 ue8m0（纯指数）格式。在 `finalize_weights()` 中通过 `_ue8m0_uint8_to_float()` 将 ue8m0 转换为 float32（`model.py:L261`），然后调用 `deep_gemm.transform_sf_into_required_layout()` 重新布局缩放因子供内核使用。

### 5.3 为什么评分函数固定为 sqrtsoftplus？

```python
self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
if self.use_mega_moe and self.scoring_func != "sqrtsoftplus":
    raise NotImplementedError(...)
```
（`model.py:L483-486`）

**原因：** DeepGEMM MegaMoE 内核期望路由权重经过 sqrtsoftplus 归一化，这样与 FP8 激活的量化范围配合更好。softplus 提供平滑的梯度，sqrt 进一步压缩路由权重的动态范围，与 FP8 的有限指数范围匹配。

### 5.4 FusedMoE 路径中，gate 是否作为内部路由器？

```python
def _forward_fused_moe(self, hidden_states, input_ids=None):
    if self.experts.is_internal_router:
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states, input_ids=input_ids)
    else:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits, input_ids=input_ids)
```
（`model.py:L644-663`）

当 `FusedMoE` 的 `is_internal_router` 为 `True` 时，gate 计算被融合在 FusedMoE 内核内部（`DeepseekV4MoE` 传 `router_logits=hidden_states` 作为占位符）。这是为了内核融合优化——将 gate 矩阵乘与 expert 计算内核合并，减少全局内存读写和 kernel launch 开销。

### 5.5 MegaMoE 权重变换（`finalize_weights`）的时机

```python
def finalize_weights(self) -> None:
    if self._transformed_l1_weights is not None:
        return
    self._check_runtime_supported()
    import vllm.third_party.deep_gemm as deep_gemm
    # 将原始 uint8/ue8m0 权重转换为 DeepGEMM 专用布局
    ...
    self.w13_weight = None   # 释放原始权重
```
（`model.py:L280-316`）

**设计意图：**
1. **延迟执行**：`finalize_weights()` 在 `load_weights()` 之后调用（见 `DeepseekV4Model.load_weights` 中调用 `self.finalize_mega_moe_weights()`，`model.py:L1366`），确保所有权重已正确加载。
2. **就地转换**：转换后的权重使用 DeepGEMM 的内部布局（交错排列 L1 层权重），原始 `w13_weight` 被释放以节省显存（`model.py:L313-316`）。
3. **幂等性**：`_transformed_l1_weights is not None` 检查确保只转换一次（`model.py:L281`）。但如果用 dummy 权重加载后又被真实权重覆盖，需要在 forward 时重新调用（`model.py:L396` 的注释特别说明了这一点）。

---

## 6. 输入数据流详解

### 6.1 MegaMoE 完整流程

```
hidden_states [num_tokens, hc_mult, hidden_size]
    │
    │  view() 展平 hc_mult 维度（由 DecoderLayer 保证输入形状）
    ▼
hidden_states [num_tokens, hidden_size]
    │
    ├─ Step 1: 路由计算
    │   gate(hidden_states) → router_logits [num_tokens, n_routed_experts] (float32)
    │   fused_topk_bias(router_logits, ...) → topk_weights [num_tokens, top_k], topk_ids [num_tokens, top_k]
    │   （model.py:L613-627）
    │
    ├─ Step 2: 输入准备（prepare_megamoe_inputs）
    │   └─ 对 hidden_states 进行 FP8 动态量化：
    │      - 每 128 元素一组，计算 E8M0 缩放因子
    │      - 量化为 FP8 E4M3，同时打包 int32 缩放因子
    │      - topk_ids 转换为 int64，topk_weights 保持 float32
    │   → x_fp8 [num_tokens, hidden_size] (fp8)
    │   → x_sf [num_tokens, hidden_size/128] (int32, 打包)
    │   → topk_idx_out [num_tokens, top_k] (int64)
    │   → topk_weights_out [num_tokens, top_k] (float32)
    │   （prepare_megamoe.py:L118-173）
    │
    ├─ Step 3: 专家计算
    │   experts.forward(hidden_states, topk_weights, topk_ids)
    │   └─ 内部通过 custom op → _run_mega_moe()
    │      └─ prepare_megamoe_inputs(...) → symm_buffer 填充
    │      └─ deep_gemm.fp8_fp4_mega_moe(...) → y [num_tokens, hidden_size] (bf16)
    │   （model.py:L345-407）
    │
    ├─ Step 4: 共享专家
    │   shared_experts(hidden_states) → shared_output [num_tokens, hidden_size] (bf16)
    │   final_hidden_states = y + shared_output
    │   （model.py:L638-640）
    │
    └─ Step 5: 恢复形状
        return final_hidden_states.view(org_shape)
        （model.py:L642）
```

### 6.2 FusedMoE 完整流程

```
hidden_states [num_tokens, hidden_size]
    │
    ├─ Step 1: 路由计算
    │   gate(hidden_states) → router_logits [num_tokens, n_routed_experts] (float32)
    │
    ├─ Step 2: 专家计算（FusedMoE kernel 内部完成排序、dispatch、计算、combine）
    │   experts(hidden_states, router_logits) → [num_tokens, hidden_size] (bf16)
    │
    └─ Step 3: 共享专家
        shared_experts(hidden_states) → add to experts output
```

---

## 7. 路由机制详解

### 7.1 路由类型

DeepSeek V4 支持三种路由方式，由 `config` 中的 `num_hash_layers` 和 `topk_method` 控制：

| 类型 | 触发条件 | 实现 |
|------|---------|------|
| Hash MoE | `extract_layer_index(prefix) < config.num_hash_layers`（`model.py:L505`） | 使用 `tid2eid` 哈希表将 token ID 映射到固定专家，无 gate 学习 |
| Noaux TC | `topk_method == "noaux_tc"`（`model.py:L520`） | 使用 `e_score_correction_bias` 修正路由得分 |
| 默认 | 上述条件不满足 | gate 直接输出 logits 经 sqrtsoftplus + topk 选择 |

### 7.2 Hash MoE 设计

```python
if is_hash_moe:
    self.gate.tid2eid = nn.Parameter(
        torch.randint(0, config.n_routed_experts,
                      (config.vocab_size, config.num_experts_per_tok),
                      dtype=self.hash_indices_dtype),
        requires_grad=False,
    )
```
（`model.py:L510-518`）

- 哈希表形状 `(vocab_size, num_experts_per_tok)`，为每个 token ID 预先分配固定的 top-k 专家。
- 用 `randint` 而非 `empty` 避免 dummy mode 下的垃圾值导致非法内存访问（`model.py` 注释说明）。
- MegaMoE 路径使用 `int64`（`model.py:L506`），FusedMoE 路径使用 `int32`（`amd/model.py:L144`）。

### 7.3 `fused_topk_bias` 的调用

```python
topk_weights, topk_ids = fused_topk_bias(
    hidden_states=hidden_states,
    gating_output=router_logits,
    scoring_func=self.scoring_func,
    e_score_correction_bias=self.gate.e_score_correction_bias.data if ... else None,
    topk=self.n_activated_experts,
    renormalize=self.renormalize,
    indices_type=self.hash_indices_dtype,
    input_tokens=input_ids,
    hash_indices_table=self.gate.tid2eid,
    routed_scaling_factor=self.routed_scaling_factor,
)
```
（`model.py:L614-627`）

`fused_topk_bias` 是一个融合算子（`vllm/model_executor/layers/fused_moe/router/fused_topk_bias.py`），一次性完成：
1. 评分函数应用（sqrtsoftplus）
2. e_score_correction_bias 修正
3. Hash 路由查找（如果 `hash_indices_table` 非空）
4. topk 选择
5. 权重归一化（`renormalize=True` 时）

---

## 8. 量化策略（MoE 相关）

> 详细的数据类型、Scale 格式、量化公式等参见 [[MoEQuantization]]。

### 8.1 MegaMoE 路径的量化

**三层量化管道：**

| 层 | 存储格式 | 计算格式 | 缩放因子 |
|----|---------|---------|---------|
| 输入激活 | bf16 | → FP8 E4M3 | E8M0 逐块缩放（block_size=128，`prepare_megamoe.py:L60`） |
| 专家权重（w13/w2） | FP4（uint8 打包） | FP4 | ue8m0 逐块缩放（block_size=32，`model.py:L179`） |
| 输出激活 | bf16 | bf16 | 无量化 |

**缩放因子的打包格式：**
- 对 FP8 激活缩放，每 128 元素计算 E8M0 指数，打包为 int32（每字节一个指数，`prepare_megamoe.py:L79`）
- 对 FP4 权重缩放，每 32 元素存储一个 ue8m0 指数，`model.py:L286-292` 中通过 `transform_sf_into_required_layout` 重排为 DeepGEMM 所需格式

### 8.2 FusedMoE 路径的量化

取决于 `quant_config.py` 中的 `expert_dtype` 分发：

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, FusedMoE):
        if self.expert_dtype == "fp4":
            if self.moe_quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(...)  # group_size=16
            return Mxfp4MoEMethod(...)              # MXFP4
        # expert_dtype == "fp8": Fp8MoEMethod with block-wise float32 scales
```
（`quant_config.py:L132-153`）

| `expert_dtype` | `moe_quant_algo` | 量化方法 | 缩放因子格式 |
|---------------|-----------------|---------|-------------|
| `"fp4"` | `"NVFP4"` | `ModelOptNvFp4FusedMoE` | group_size=16 |
| `"fp4"` | 其他/空 | `Mxfp4MoEMethod` | ue8m0 |
| `"fp8"` | — | `Fp8MoEMethod` | float32 block scales |
| （默认） | — | `UnquantizedFusedMoEMethod` | 无量化 |

---

## 9. 并行策略

### 9.1 MegaMoE — 专家并行（EP）

```python
self.ep_group = get_ep_group()
self.ep_size = self.ep_group.world_size
self.ep_rank = self.ep_group.rank_in_group
self.n_local_experts = config.n_routed_experts // self.ep_size
self.experts_start_idx = self.ep_rank * self.n_local_experts
```
（`model.py:L552-559`）

- EP 比 TP 更适合 MoE：专家权重在不同 rank 间独立，只在 all-to-all 阶段通信。
- `DeepseekV4MegaMoEExperts` 在 `get_symm_buffer()` 中获取 EP group 的同步缓冲区（`model.py:L319-343`），`deep_gemm.get_symm_buffer_for_mega_moe` 内部使用 `ep_group` 创建通信缓冲区。
- `fused_topk_bias` 输出的 topk_ids 在所有 rank 上相同（gate 是复制层），后续 DeepGEMM 内核内部根据 `ep_rank` 分发本地专家。

### 9.2 FusedMoE — 张量并行（TP）

```python
self.n_local_experts = config.n_routed_experts // self.tp_size
self.experts_start_idx = self.tp_rank * self.n_local_experts
```
（`amd/model.py:L182-183`，也是 `nvidia/model.py` FusedMoE 路径）

- 每个 TP rank 持有 `n_routed_experts / tp_size` 个完整专家。
- FusedMoE 层内部通过 all-reduce 合并各专家的输出。
- gate 路由网络是复制层（`ReplicatedLinear`），每个 rank 独立计算完整路由。
- **AMD 路径仅支持 FusedMoE**：`amd/model.py` 中没有 `use_mega_moe` 标志，始终使用 FusedMoE + TP。AMD 不实现 MegaMoE，因为其依赖的 DeepGEMM 内核（含 EP all-to-all）是 NVIDIA SM100 专有的。

### 9.3 共享专家 — TP 策略

```python
self.shared_experts = DeepseekV4MLP(
    ...
    reduce_results=self.use_mega_moe,  # MegaMoE: True 表示不 all-reduce（后续由 EP 处理）
    disable_tp=is_sequence_parallel,    # MegaMoE 路径禁用 TP
)
```
（`model.py:L531-538`）

- **MegaMoE 路径**：`is_sequence_parallel=True`，禁用 TP，每个 rank 持有完整共享专家权重。
- **FusedMoE 路径**：使用标准 TP 切分（gate_up_proj 列切分，down_proj 行切分 + all-reduce）。

---

## 10. 关键算子：`prepare_megamoe_inputs`

**位置：** `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`，详见 [[PrepareMegaMoEInputs]]

这是一个 Triton kernel，将 `hidden_states`（bf16）量化为 FP8 E4M3 并打包成 DeepGEMM 所需的输入布局。

### 量化流程（Triton kernel 内部）：

```
输入: hidden_states [num_tokens, hidden_size] (bf16)
    │
    ├─ 1. 按行加载，reshape 为 (num_groups, GROUP_K=32)
    ├─ 2. 计算每组 absmax → amax
    ├─ 3. scale = amax / 448.0 (448 = FP8 E4M3 最大值)
    ├─ 4. 将 scale 四舍五入到 E8M0 格式（纯指数，8 位）
    ├─ 5. hidden / rounded_scale → 量化 → fp8
    └─ 6. 将 4 个 E8M0 指数打包成一个 int32（每字节一个指数）
```
（`prepare_megamoe.py:L44-83`）

**输出布局说明：**
- `x_fp8` 仍为 `(num_tokens, hidden_size)`，但 dtype 是 fp8e4nv。
- `x_sf` 为 `(num_tokens, hidden_size // 128)`，每个 int32 打包了 4 个 uint8 指数（因为 128/32=4 个 group）。
- `topk_idx_out` 为 `int64`（DeepGEMM 要求），`topk_weights_out` 为 `float32`。

### 为什么除以 448.0？

FP8 E4M3 的能表示的最大有限值是 448.0（`2^(8-1) × (1 + 3/8) = 128 × 1.375 = 448`）。将输入缩放到 `[-448, 448]` 范围内可以最大化利用 FP8 的精度。

---

## 11. 共享专家（`DeepseekV4MLP`）

### 结构

```python
class DeepseekV4MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, ...):
        self.gate_up_proj = MergedColumnParallelLinear(...)  # hidden → 2×intermediate
        self.down_proj = RowParallelLinear(...)               # intermediate → hidden
        self.act_fn = SiluAndMul()
```
（`model.py:L68-116`）

是一个标准 SwiGLU MLP，但与普通 MLP 的区别在于：

1. **`gate_up_proj` 是融合的**：`MergedColumnParallelLinear` 将 gate 和 up 两个线性层合并为一个 GEMM，输出维度 `2 * intermediate_size`（前一半为 gate，后一半为 up），然后 `SiluAndMul` 计算 `SiLU(gate) * up`。
2. **`intermediate_size` 乘以 `n_shared_experts`**：共享专家的中间大小是 `moe_intermediate_size * n_shared_experts`（`model.py:L529`），即共享专家等价于多个 standard expert 的融合。

### forward

```python
def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x, _ = self.down_proj(x)
    return x
```
（`model.py:L112-116`）

形状变化：
- `gate_up`: `[num_tokens, 2 * intermediate_size]` (bf16)
- `act_fn` 后: `[num_tokens, intermediate_size]` (bf16)
- `down_proj`: `[num_tokens, hidden_size]` (bf16)

---

## 12. DeepseekV4MegaMoEExperts 详解

### 12.1 权重布局

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `w13_weight` | `(n_local, 2*intermediate, hidden//2)` | uint8 | gate+up FP4 权重（half 因为 4-bit） |
| `w13_weight_scale` | `(n_local, 2*intermediate, hidden//32)` | uint8 | ue8m0 缩放因子，block_size=32 |
| `w2_weight` | `(n_local, hidden, intermediate//2)` | uint8 | down FP4 权重 |
| `w2_weight_scale` | `(n_local, hidden, intermediate//32)` | uint8 | ue8m0 缩放因子，block_size=32 |

### 12.2 `finalize_weights()` 转换流程

```python
def finalize_weights(self) -> None:
    # 1. 硬件检查：必须为 SM100、CUDA、维度对齐
    self._check_runtime_supported()
    
    # 2. ue8m0 uint8 → float32（重塑指数位）
    w13_scale = self._ue8m0_uint8_to_float(self.w13_weight_scale)
    
    # 3. DeepGEMM 的 transform_sf_into_required_layout
    #    将缩放因子重排为内核所需格式
    w13_scale = deep_gemm.transform_sf_into_required_layout(
        w13_scale, 2*intermediate_size, hidden_size, (1, 32), n_local)
    
    # 4. transform_weights_for_mega_moe
    #    将权重和缩放因子打包为 DeepGEMM 内部数据格式
    self._transformed_l1_weights, self._transformed_l2_weights = \
        deep_gemm.transform_weights_for_mega_moe(
            (w13_weight_int8, w13_scale), (w2_weight_int8, w2_scale))
    
    # 5. 释放原始权重以节省显存
    self.w13_weight = None
```
（`model.py:L280-316`）

### 12.3 `_ue8m0_uint8_to_float()`

```python
@staticmethod
def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
    return (sf.to(torch.int32) << 23).view(torch.float32)
```
（`model.py:L260-262`）

ue8m0 是一个无符号 8 位指数格式（没有尾数位），指数范围 `[0, 255]`。存储的值是原始指数值，需要将其放入 IEEE 754 float32 的指数位（bit 23-30）并将尾数位清零，得到 `2^(E-127)` 的效果。

### 12.4 `get_symm_buffer()`

```python
def get_symm_buffer(self):
    key = (id(group), device, self.num_experts, self.max_num_tokens,
           self.top_k, self.hidden_size, self.intermediate_size)
    symm_buffer = self._symm_buffer_cache.get(key)
    if symm_buffer is None:
        symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(...)
        self._symm_buffer_cache[key] = symm_buffer
    return symm_buffer
```
（`model.py:L318-343`）

- 对称缓冲区（`symm_buffer`）通过 `(ep_group, GPU, expert config)` 的元组缓存，跨 EP rank 共享。
- 包含 `x`、`x_sf`、`topk_idx`、`topk_weights` 等输入字段，由 `prepare_megamoe_inputs` 填充。
- 使用全局类缓存 `_symm_buffer_cache`（`model.py:L139`），避免创建多个解码层时重复分配。

---

## 13. 设计亮点与思考

### 13.1 为什么 MegaMoE 选择 FP4 权重 + FP8 激活？

FP8 权重已经可以显著减少显存占用（相比 BF16 减半），而 FP4 进一步减半，使得 DeepSeek V4 的 MoE（256 个路由专家）可以在有限的显存中装下。FP8 激活则避免了激活值量化成为精度瓶颈——因为路由权重对精度敏感，FP8 的 E4M3 格式相比 FP4 保留更多动态范围。这种组合（FP4 权重 + FP8 激活）代表了精度与显存之间在 Blackwell 上的最佳平衡选择。

### 13.2 共享专家 vs 路由专家

- 路由专家：256 个专家，每个 token 激活 top_k 个。
- 共享专家：一个（或多个合并的）`DeepseekV4MLP`，所有 token 无论路由结果都会计算。

共享专家确保每个 token 都能获得一部分非稀疏的 FFN 计算，防止路由过于稀疏导致信息丢失。`n_shared_experts` 乘以 `moe_intermediate_size` 作为 intermediate_size 使得共享专家的计算量等价于多个标准专家。

### 13.3 为什么 MegaMoE 的 gate 路由输出 topk_ids 在所有 EP rank 上相同？

gate 是 `ReplicatedLinear`（复制层），每个 rank 独立计算完整的 `router_logits`。这看似冗余，但避免了 EP 中 all-gather 路由信息的通信开销。由于 `fused_topk_bias` 的计算量相对于专家计算很小，复制计算比通信更高效。DeepGEMM 的 MegaMoE 内核在内部使用 `ep_group` 的 all-to-all 来分发 token 到正确的 Expert rank，而 topk_ids 是全局一致的。

### 13.4 如果 token 数超过 `max_num_tokens` 会怎样？

```python
if hidden_states.shape[0] > self.max_num_tokens:
    raise ValueError(...)
```
（`model.py:L354-357`）

`symm_buffer` 的大小在创建时由 `max_num_batched_tokens` 固定。如果实际 token 数超过该值，会导致缓冲区溢出，因此在 forward 入口处显式检查并报错。这是 static graph 优化的一个限制——缓冲区的分配无法动态增长。

### 13.5 weight_loader 的 `return_success` 机制

```python
success = weight_loader(param, loaded_weight, name_mapped,
                        shard_id=expert_shard_id, expert_id=expert_id, return_success=True)
if success:
    break
```
（`model.py:L1315-1328`）

在 FusedMoE 路径中，专家的权重映射是通过 `FusedMoE.make_expert_params_mapping` 生成的（`model.py:L1358`），每个 checkpoint 的 `.experts.X.w1` 会匹配多个 vLLM 参数条目。`return_success=True` 避免了一个 weight 被应用多次——只有 expert_id 落在本地范围内的映射才会返回 `True`。

---

## 14. 边界场景与异常处理

| 场景 | 行为 | 代码位置 |
|------|------|---------|
| hash MoE 但未传入 `input_ids` | 抛出 `ValueError` | `model.py:L606-607` |
| MegaMoE 但评分函数不是 `sqrtsoftplus` | 抛出 `NotImplementedError` | `model.py:L484-486` |
| MegaMoE 但 `expert_dtype` 不是 `"fp4"` | 抛出 `NotImplementedError` | `model.py:L488-492` |
| MegaMoE 但未启用 EP | 抛出 `NotImplementedError` | `model.py:L468-473` |
| 非 SM100 GPU 上运行 MegaMoE | `finalize_weights` 中抛出 `NotImplementedError` | `model.py:L264-273` |
| hidden/intermediate 不是 128 的倍数 | 抛出 `ValueError` | `model.py:L274-277` |
| token 数超过 `max_num_tokens` | 抛出 `ValueError` | `model.py:L354-357` |

---

## 15. AMD MoE 差异

AMD 平台在 `amd/model.py` 中提供独立的 `DeepseekV4MoE` 实现（L110-L226），与 NVIDIA 版本相比有以下关键差异：

### 15.1 无 MegaMoE 路径

```python
# amd/model.py L110 — 无 use_mega_moe 判断，直接使用 FusedMoE
self.experts = FusedMoE(
    gate=self.gate,
    num_experts=config.n_routed_experts,
    top_k=config.num_experts_per_tok,
    ...
)
```

| 特性 | NVIDIA | AMD |
|------|--------|-----|
| MegaMoE 支持 | ✓（SM100 Blackwell 专用） | ✗ |
| 专家并行（EP） | ✓（MegaMoE 强制） | ✗（仅 TP） |
| 专家精度 | FP4（MegaMoE）/ FP4/FP8/BF16（FusedMoE） | FP4/FP8/BF16（取决于 quant_config） |
| FP8 激活量化 | ✓（`prepare_megamoe_inputs`） | ✗（无输入量化步骤） |
| 对称缓冲区 | ✓（`symm_buffer` 管理） | ✗ |

### 15.2 简化的架构

- **gate 不再需要 `fused_topk_bias` 的独立调用**：AMD 将 gate 和 scoring_func 直接传入 `FusedMoE`，路由在 FusedMoE 内部通过 `is_internal_router` 机制完成（`amd/model.py:L186-202`）。
- **无 `finalize_weights`**：AMD 路径不需要 DeepGEMM 布局转换，权重直接加载使用。
- **`hash_indices_dtype` 为 `int32`**（而非 NVIDIA MegaMoE 的 `int64`），因为标准 FusedMoE 期望 `int32` 索引（`amd/model.py:L144`）。

### 15.3 共享专家配置

```python
self.shared_experts = DeepseekV4MLP(
    ...
    reduce_results=False,  # AMD 路径：不额外 all-reduce
)
```

`reduce_results=False` 与 NVIDIA FusedMoE 路径的 `reduce_results=self.use_mega_moe`（MegaMoE=True 时不 reduce）形成对比。在 AMD 上，共享专家始终交给 FusedMoE 内部的通信逻辑处理。

### 15.4 整体区别

AMD 的 MoE 实现本质上是**纯 FusedMoE 路径**，没有 MegaMoE 特有的 FP4 权重变换、EP all-to-all 通信和 FP8 激活量化。这意味着 AMD 的推理路径更简单，但也失去了 Blackwell GPU 上 FP4+EP 带来的极致性能。

---

## 16. 下一步分析建议

- **DeepGEMM 内核分析**：`deep_gemm.fp8_fp4_mega_moe` 的 CUDA 内核实现和 all-to-all 通信模式
- **与 V3/V3.2 MoE 的对比**：DeepSeek V3 的 MoE 实现与 V4 的差异（尤其是 FP4 支持）
- **EP vs TP 的性能对比**：在 DeepSeek V4 规模下 EP 替代 TP 带来的通信收益量化
- **Hash MoE 的效果**：前 `num_hash_layers` 层的 hash 路由对负载均衡的影响
- **FP8 激活量化路径对比**：MegaMoE 的 `prepare_megamoe_inputs`（Triton）与标准 FusedMoE 的量化路径差异

---

## 相关笔记

- [[PrepareMegaMoEInputs]] — `nvidia/ops/` 下的 MegaMoE 输入 FP8 量化 Triton 内核
- [[MoEQuantization]] — 激活与权重量化完整详解（数据类型、Scale 格式、量化公式）
- [[DeepseekV4FP8Config]] — 量化配置分发，控制 `expert_dtype`（FP4 vs FP8）
- [[GateRouting]] — 门控路由数学：sqrtsoftplus、Hash MoE、Noaux TC
- [[QuantAndParallelStrategy]] — MoE 路径的量化与并行策略
- [[ROCmBackend]] — AMD ROCm 平台的注意力后端（不含 MegaMoE）

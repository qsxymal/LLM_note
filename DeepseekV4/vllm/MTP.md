# MTP（Multi-Token Prediction）推测解码

**文件路径：** `vllm/models/deepseek_v4/nvidia/mtp.py`（NVIDIA），`vllm/models/deepseek_v4/amd/mtp.py`（AMD）
**核心类：**
- `DeepSeekV4MTP`（`L255-L515`）— 顶层 MTP 包装器
- `DeepSeekV4MultiTokenPredictor`（`L160-L251`）— 多 token 预测器
- `DeepSeekV4MultiTokenPredictorLayer`（`L61-L157`）— 单层 MTP 实现

## 1. 模块定位

- **职责：** 实现 DeepSeek V4 的 MTP（Multi-Token Prediction）draft 模型，用于推测解码（speculative decoding）。MTP 模型在目标模型（`DeepseekV4ForCausalLM`）的顶部添加额外的 Transformer 层，每层预测一个未来的 token。
- **上游调用者：** vLLM 的推测解码调度器（`vllm/spec_decode`）
- **下游依赖：** `DeepseekV4DecoderLayer`（复用主模型的解码器层）、`SharedHead`（共享词表投影头）

## 2. 架构设计

```
DeepSeekV4MTP                                      (L255)
  └── DeepSeekV4MultiTokenPredictor                (L160)
        ├── embed_tokens (VocabParallelEmbedding) — MTP 输入嵌入
        ├── logits_processor (LogitsProcessor)
        └── layers: Dict[str, DeepSeekV4MultiTokenPredictorLayer]  (L61)
              ├── layer_{num_hidden_layers}:
              │     ├── enorm — 输入嵌入的 RMSNorm
              │     ├── hnorm — 目标模型隐藏状态的 RMSNorm
              │     ├── e_proj (ReplicatedLinear) — 输入嵌入投影 × hidden_size
              │     ├── h_proj (ReplicatedLinear) — 目标状态投影 × hidden_size
              │     ├── hc_head_fn / hc_head_base / hc_head_scale — HC Head 参数
              │     ├── shared_head (SharedHead) — 共享词表投影
              │     ├── hc_head_op — HC Head 算子
              │     └── mtp_block (DeepseekV4DecoderLayer) — 完整的解码器层
              ├── layer_{num_hidden_layers+1}: ...
              └── layer_{num_hidden_layers+num_mtp_layers-1}: ...
```

### 2.1 与 V3 MTP 的区别

V4 MTP 从 V3 的 `deepseek_mtp.py` 分离出来，主要差异见文件注释（`mtp.py:L5-10`）：
- **分离的 `e_proj` / `h_proj`**（V3 使用融合的 `eh_proj`），各使用独立的 FP8 线性量化。
- **HC Head**：MTP 的 logits 计算前需要经过 `hc_head_op` 将多流隐藏状态合并。
- **`DeepseekV4DecoderLayer`** 复用，包含辅助流管理。
- **V4 特有的 checkpoint 权重重映射**。

## 3. 对外接口与数据类型

### `DeepSeekV4MultiTokenPredictor.forward()`（`L209-L226`）

```python
def forward(
    self,
    input_ids: torch.Tensor,               # [num_tokens]
    positions: torch.Tensor,               # [num_tokens]
    previous_hidden_states: torch.Tensor,  # [num_tokens, hc_dim] — 目标模型预 HC Head 隐藏状态
    inputs_embeds: torch.Tensor | None,    # [num_tokens, hidden_size]
    spec_step_idx: int,                    # 当前推测步索引
) -> torch.Tensor
```

**关键 Tensor：**

| 变量 | shape | dtype | 说明 |
|------|-------|-------|------|
| `inputs_embeds` | `(num_tokens, hidden_size)` | bf16 | 输入 token 嵌入，由 `ernorm` 归一化 |
| `previous_hidden_states` 输入 | `(num_tokens, hc_dim)` | bf16 | 目标模型的预 HC Head 残差流缓存 |
| `previous_hidden_states` view | `(num_tokens, hc_mult, hidden_size)` | bf16 | reshape 为多流格式 |
| `hidden_states` 输出 | `(num_tokens, hc_dim)` | bf16 | 展平的多流隐藏状态 |

### `DeepSeekV4MultiTokenPredictor.compute_logits()`（`L228-L251`）

```python
def compute_logits(
    self,
    hidden_states: torch.Tensor,  # [num_tokens, hc_dim] — forward 的原始输出
    spec_step_idx: int,
) -> torch.Tensor:
```

输出前先通过 `hc_head_op` 将多流合并为单流 `[num_tokens, hidden_size]`，然后通过 `shared_head` + `logits_processor` 计算 logits。

### `DeepSeekV4MTP` 包装类（`L255-L286`）

```python
class DeepSeekV4MTP(nn.Module):
    def forward(self, input_ids, positions, hidden_states, ...):
        return self.model(input_ids, positions, hidden_states, ...)

    def compute_logits(self, hidden_states, spec_step_idx):
        return self.model.compute_logits(hidden_states, spec_step_idx)
```

`DeepSeekV4MTP` 是一个轻量包装器，持有 `DeepSeekV4MultiTokenPredictor` 实例，将 `forward` 和 `compute_logits` 委托给内部 `self.model`。被 `@support_torch_compile` 修饰以支持 torch.compile 图捕获。

## 4. 数据流

### 单层 forward（`L125-L157`）

```python
# 1. 在 position=0 处 mask 掉输入（MTP 的 position 0 无意义）
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)

# 2. 归一化
inputs_embeds = self.enorm(inputs_embeds)
previous_hidden_states = previous_hidden_states.view(-1, self.hc_mult, self.config.hidden_size)
previous_hidden_states = self.hnorm(previous_hidden_states)

# 3. 融合：投影 + 广播加法
hidden_states = self.h_proj(previous_hidden_states) + \
                self.e_proj(inputs_embeds).unsqueeze(-2)

# 4. 解码器层（包含注意力 + MoE + HC 混合）
hidden_states, residual, post_mix, res_mix = self.mtp_block(
    positions=positions, x=hidden_states, input_ids=None)

# 5. CUDA 上执行 hc_post（ROCm 回退路径中 mtp_block 内部已处理）
if current_platform.is_cuda():
    hidden_states = self.mtp_block.hc_post(hidden_states, residual, post_mix, res_mix)

return hidden_states.flatten(1)  # 展平多流 → 返回 pre-HC-Head 残差
```

### 融合公式

```
hidden_states = h_proj(hnorm(target_hidden_states)) +  ← 目标模型状态
              e_proj(enorm(inputs_embeds))              ← 输入嵌入
```

`h_proj` 的输出 shape `[num_tokens, hc_mult, hidden_size]`，`e_proj` 的输出通过 `unsqueeze(-2)` 广播为 `[num_tokens, 1, hidden_size]`，实现逐元素相加。这种设计的直觉是：
- `h_proj` 将目标模型的残差流映射到 MTP 层需要的多流空间。
- `e_proj` 提供当前 step 的输入信息，通过广播与目标状态融合。

### HC Head 延迟到 compute_logits（`L236-L247`）

```python
# compute_logits 中
hidden_states = hidden_states.view(-1, hc_mult, hidden_size)
hidden_states = hc_head_op(       # 合并多流 → 单流
    hidden_states,
    hc_head_fn, hc_head_scale, hc_head_base, ...)
logits = logits_processor(shared_head.head, shared_head(hidden_states))
```

**设计意图**：MTP 的 forward 输出 `pre-hc_head` 的残差流（而不是合并后的单流），这样当 `num_speculative_tokens > 1` 时，该输出可以直接作为下一推测步的 `previous_hidden_states` 输入，避免反复合并与展开。HC Head 被推迟到 `compute_logits` 才执行，那里才真正需要单流 hidden state 来计算词表概率。

## 5. 推测解码步调度（`L209-L226`）

```python
current_step_idx = spec_step_idx % self.num_mtp_layers
return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
    input_ids, positions, previous_hidden_states, inputs_embeds, current_step_idx)
```

- 每个推测步（`spec_step_idx`）循环使用一个 MTP 层（`num_speculative_tokens` 个层）。
- MTP 层的索引从 `num_hidden_layers` 开始（在目标模型层之后）。
- 当 `spec_step_idx > num_mtp_layers` 时，`%` 取模使有限的 MTP 层被循环使用。

## 6. 与目标模型的交互

### 隐藏状态缓存

目标模型 [[DeepseekV4Model]] 在 forward 中保存预 HC Head 的残差流（`model.py:L1239-1240`）：

```python
self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
```

形状：`[max_num_batched_tokens, hc_dim]`，地址固定在 CUDA graph 池外（`model.py:L1169` 注释），确保每次 `copy_` 能正确更新。

MTP 模型通过 `DeepseekV4ForCausalLM.get_mtp_target_hidden_states()`（`model.py:L1463`）获取此缓冲区。

### 消费时序

```
目标模型 forward → _mtp_hidden_buffer 写入
    → 推测解码调度器调用 MTP forward
        → MTP 层读取 _mtp_hidden_buffer 作为 previous_hidden_states
        → MTP 层输出新的 hidden_states
            → 重复（下一推测步）
    → 目标模型下一轮 forward → _mtp_hidden_buffer 被覆盖
```

### 权重加载的协作

MTP 权重加载不经过主模型的 `AutoWeightsLoader`（通过 `skip_substrs=["mtp."]` 跳过），而是由 `DeepSeekV4MTP.load_weights` 独立处理（`mtp.py:L288`）。这允许 MTP checkpoint 作为独立文件加载，不干扰主模型权重。

## 7. 权重加载详情（`L288-L475`）

### 层索引重写

MTP 权重在 checkpoint 中存储为 `mtp.{i}.*`，加载时映射到 `model.layers.{num_hidden_layers + i}.*`（`L361-L364`）：

```python
name = name.replace(
    f"mtp.{mtp_layer_idx}.",
    f"model.layers.{self.config.num_hidden_layers + mtp_layer_idx}.",
)
```

这样 `get_spec_layer_idx_from_weight_name`（来自 `deepseek_v2.py`）可以正确识别层索引。

### `_rewrite_spec_layer_name`（`L481-L515`）

根据权重名决定是否添加 `.mtp_block` 前缀：

| 权重名包含 | 处理方式 | 示例 |
|-----------|---------|------|
| `embed_tokens` 等共享名 | 提升为顶层：`model.layers.X.embed_tokens` → `model.embed_tokens` | 跨层共享的嵌入权重 |
| `enorm`, `hnorm`, `e_proj`, `h_proj`, `shared_head` 等 | 保留在层内，不加 `.mtp_block` | 层专用的 MTP 参数 |
| 其他（attn, ffn, norm 等） | 添加 `.mtp_block` 前缀：`model.layers.X.attn` → `model.layers.X.mtp_block.attn` | 交给 `DeepseekV4DecoderLayer` |

### 专家 scale 后缀选择（`L350-L354`）

```python
expert_scale_suffix = (
    ".weight_scale"   if expert_dtype == "fp4"
    else ".weight_scale_inv"
)
```

FP4/MXFP4 专家注册 `w{1,2,3}_weight_scale`，FP8 专家注册 `w{1,2,3}_weight_scale_inv`。权重名的 `.scale` 后缀根据需要替换为正确版本（`L375-L381`）。

### E8M0 类型转换（`L400-L404`）

与主模型相同——FP4 专家的 weight_scale 以 `float8_e8m0fnu` 存储，通过 `.view(torch.uint8)` 按位读取。

### 缺失层检查（`L456-L472`）

```python
for layer_idx in range(mtp_start, mtp_start + num_mtp_layers):
    if layer_idx not in loaded_layers:
        raise ValueError(
            f"MTP layer {layer_idx} weights missing from checkpoint. ...")
```

加载完成后验证所有 MTP 层都有权重被加载。如果某层缺失（例如 checkpoint 被量化时未包含 MTP 层），显式报错而非静默失败。

### `finalize_mega_moe_weights()`（`L477-L479`）

```python
def finalize_mega_moe_weights(self) -> None:
    for layer in self.model.layers.values():
        layer.mtp_block.ffn.finalize_mega_moe_weights()
```

与主模型的 `finalize_mega_moe_weights` 等价——对每个 MTP 层的 FFN 执行 DeepGEMM 权重重排。

## 8. 量化策略

与目标模型共享相同的 `quant_config`：
- **`e_proj` / `h_proj`**：`ReplicatedLinear` 使用 FP8 量化（`mtp.py:L82-95`）。
- **MTP 解码器层** `DeepseekV4DecoderLayer`：继承目标模型的注意力 + MoE 量化。
- **`SharedHead`**：词表投影头使用 FP8 量化。

## 9. 与 MoE 的关系

- MTP 层使用 `DeepseekV4DecoderLayer`（包含 `DeepseekV4MoE`），因此也支持 MegaMoE 或 FusedMoE。
- 专家权重映射和 `finalize_mega_moe_weights()` 在 `DeepSeekV4MTP.load_weights` 中同样处理（`mtp.py:L477-479`）。
- MTP 层使用 `make_deepseek_v4_expert_params_mapping` 或 `FusedMoE.make_expert_params_mapping` 生成专家映射（`mtp.py:L333-345`）。

## 10. 边界场景

- **`inputs_embeds=None`**（`L217`）：自动通过 `embed_tokens(input_ids)` 生成嵌入。
- **`position=0`**（`L135`）：MTP 的 position 0 没有语义意义（第一个推测步总是基于目标模型输出），通过 `torch.where` mask 为零。
- **`num_mtp_layers=0`**：`forward` 中 `0 % 0` 会触发除零错误，但配置上不应出现。
- **`spec_step_idx >= num_mtp_layers`**：`%` 取模循环使用有限的 MTP 层。
- **ROCm 回退路径**（`L150`）：`hc_post` 在 `_forward_native` 中已在 `mtp_block` 内部执行，外层不重复调用。
- **MTP 层权重缺失**：抛出 `ValueError`，引导用户使用包含 MTP 层的 checkpoint 或关闭推测解码。

## Cross-References

- [[DeepseekV4ForCausalLM]]：目标模型，主 `load_weights` 中跳过 `mtp.` 前缀
- [[DeepseekV4Model]]：提供 `_mtp_hidden_buffer` 供 MTP 使用
- [[DeepseekV4DecoderLayer]]：MTP 层复用的解码器层实现
- [[DeepseekV4MoE]]：MTP 层同样支持 MegaMoE / FusedMoE
- [[load_weights]]：主模型的权重加载流程，与 MTP 独立加载对比

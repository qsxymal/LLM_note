# DeepseekV4ForCausalLM

**文件路径：** `vllm/models/deepseek_v4/nvidia/model.py`，L1409-L1477

## 1. 模块定位

- **职责：** DeepSeek V4 模型的顶层因果语言模型（Causal Language Model）封装，持有 `DeepseekV4Model` 和 `lm_head`，负责权重加载映射、前向传播调度、logits 计算，以及与流水线并行（PP）和推测解码（MTP）的对接。
- **继承关系：** `torch.nn.Module` + `SupportsPP`（`model.py:L1409`）— 支持流水线并行。
- **上游调用者：** vLLM 引擎 `ModelRunner`（通过 `forward` 方法）
- **下游依赖：** `DeepseekV4Model` → `DeepseekV4DecoderLayer` → `DeepseekV4Attention` / `DeepseekV4MoE`
- **平台适配：** `__init__.py` 根据平台选择 NVIDIA 或 AMD 实现（`vllm/models/deepseek_v4/__init__.py:L19-24`）

---

## 2. 类属性

```python
model_cls = DeepseekV4Model                                          # L1410
hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")          # L1414
```
（`model.py:L1410-1414`）

### `model_cls`
- 指定底层 Transformer 模型类（不含 lm_head）。`__init__` 中实例化此类。
- 这是一个类属性，被子类可以直接继承。

### `hf_to_vllm_mapper`
- 默认的 HuggingFace → vLLM 权重名称映射器，FP4 专家格式版本。
- 由 `_make_deepseek_v4_weights_mapper("fp4")` 工厂函数创建（`model.py:L1371`），返回 `WeightsMapper` 对象。
- 如果 `expert_dtype != "fp4"`，`__init__` 中会动态替换此映射器。
- 该映射器处理四层名称替换：前缀（`layers.` → `model.layers.`）、后缀（`head.weight` → `lm_head.weight`）、子串（`.attn.compressor.` → `.attn.mla_attn.compressor.`）、正则（`\.experts\.\d+\.w[123]\.scale$` → `.weight_scale`）。详见 [[load_weights]]。

---

## 3. `__init__` 方法

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):    # L1416
```
（`model.py:L1416-1438`）

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `vllm_config` | `VllmConfig` | vLLM 顶层配置聚合（含 `model_config.hf_config`, `quant_config`, `parallel_config`, `cache_config` 等）。详见 [[vllmconfig]] |
| `prefix` | `str` | 参数名前缀，默认 `""`。通过 `maybe_prefix(prefix, "model")` 标记子模块命名空间 |

### 内部逻辑

#### Step 1: 提取 `expert_dtype`（L1419-L1423）

```python
expert_dtype = getattr(config, "expert_dtype", "fp4")
```

从 HuggingFace 配置中读取专家权重的数据类型。DeepSeek V4 checkpoint 有两种变体：
| 变体 | `expert_dtype` | 专家格式 | 缩放因子格式 |
|------|---------------|---------|-------------|
| DeepSeek-V4-Flash | `"fp4"`（默认） | FP4（MegaMoE/MXFP4/NVFP4） | ue8m0（`float8_e8m0fnu`） |
| DeepSeek-V4-Flash-Base | `"fp8"` | FP8 block | float32 |

`getattr(..., "fp4")` 的默认值确保旧 checkpoint 无需修改即可使用。

#### Step 2: 动态替换权重映射器（L1422-L1423）

```python
if expert_dtype != "fp4":
    self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
```

FP4 和 FP8 专家的 checkpoint 缩放因子映射规则不同：
- FP4：`w{1,2,3}.scale` → `w{1,2,3}_weight_scale`（ue8m0 格式）
- FP8：`w{13,2}.scale` → `w{13,2}_weight_scale_inv`（float32 格式）

详见 [[load_weights#HF-到-vLLM-的名称映射]]。

#### Step 3: 实例化底层模型（L1425-L1426）

```python
self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
```

- 创建 [[DeepseekV4Model]] 实例，参数名前缀加 `"model"`。
- `maybe_prefix` 避免重复前缀：如果 `prefix=""`，结果为 `"model"`；如果 `prefix="model"`，结果仍为 `"model"`。

#### Step 4: 创建 lm_head（L1428-L1434）

```python
if get_pp_group().is_last_rank:
    self.lm_head = ParallelLMHead(...)       # 末 rank：并行化语言模型头
else:
    self.lm_head = PPMissingLayer()           # 非末 rank：占位符
```

- `get_pp_group()` 获取当前流水线并行通信组。
- `PPMissingLayer` 在非末 rank 上替换为占位符（`forward` 返回 `None`），确保 PP 各 rank 的 `named_parameters` 仍然对齐。
- `ParallelLMHead` 继承自 [[VocabParallelEmbedding]]，将 lm_head 的权重矩阵按词汇表维度进行 TP 切分。详见 [[ParallelLMHead]]。

#### Step 5: 其他组件（L1436-L1439）

```python
self.logits_processor = LogitsProcessor(config.vocab_size)  # logits 后处理
self.make_empty_intermediate_tensors = (
    self.model.make_empty_intermediate_tensors               # PP 中间张量工厂
)
```

- **`LogitsProcessor`**：可插拔层（`@PluggableLayer.register("logits_processor")`），负责 logits 的后处理（`lm_head` 投影、TP gather、soft_cap、scale 缩放等）。详见 [[LogitsProcessor]]。
- **`make_empty_intermediate_tensors`**：引用 `DeepseekV4Model` 的同名方法（`model.py:L1182`），用于 PP 中创建空的 `IntermediateTensors`，形状为 `(batch_size, hc_mult, hidden_size)`。类型别名标记（`# type: ignore[method-assign]`）因为这是赋值而非方法重写。

---

## 4. `embed_input_ids` 方法

```python
def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:  # L1441
    return self.model.embed_input_ids(input_ids)                     # L1442
```
（`model.py:L1441-1442`）

| 参数/返回 | shape | dtype | 说明 |
|-----------|-------|-------|------|
| `input_ids` | `(num_tokens,)` | int32/int64 | 输入 token ID |
| 返回 | `(num_tokens, hidden_size)` | bf16 | token 嵌入向量 |

- 直接委托给 [[DeepseekV4Model]].`embed_input_ids`（`model.py:L1179`），后者调用 `self.embed_tokens(input_ids)`。
- 仅在 PP 首 rank 有效（非首 rank 的 `embed_tokens` 是 `PPMissingLayer`）。

---

## 5. `compute_logits` 方法

```python
def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:  # L1444
    logits = self.logits_processor(self.lm_head, hidden_states)                 # L1448
    return logits                                                               # L1449
```
（`model.py:L1444-1449`）

| 参数/返回 | shape | dtype | 说明 |
|-----------|-------|-------|------|
| `hidden_states` | `(num_tokens, hidden_size)` | bf16 | Transformer 输出（HC Head 合并 + norm 后） |
| `logits` | `(num_tokens, vocab_size)` 或 `None` | float32/bf16 | TP rank 0 返回完整 logits；非 rank 0 返回 `None` |

### 处理流水线

```
hidden_states [num_tokens, hidden_size] (bf16)
    │
    ├─ lm_head (ParallelLMHead): weight [vocab_size_per_partition, hidden_size] (bf16/FP8)
    │   └─ quant_method.apply(hidden_states) → [num_tokens, vocab_size_per_partition]
    │
    ├─ LogitsProcessor._gather_logits:
    │   └─ tensor_model_parallel_gather (CUDA) or all_gather (TPU)
    │   └─ → [num_tokens, vocab_size_padded] (仅 rank 0)
    │
    └─ LogitsProcessor: 截取 org_vocab_size → [num_tokens, org_vocab_size]
```

### 关键设计：分离式 logits 计算

`forward` 只返回隐藏状态，logits 通过 `compute_logits` 单独计算。这种分离有两个原因：

1. **PP 异步调度**：在 PP 中，末 rank 收到隐藏状态后需要返回给引擎，引擎再在当前 rank 上调用 `compute_logits`。如果 logits 计算融合在 `forward` 中，PP 各 stage 的调度会变得复杂（非末 rank 不需要 lm_head）。
2. **零填充张量优化**：`compute_logits` 可以被 `torch.compile` 或 CUDA graph 捕获，与主模型 forward 独立缓存。

### TP 下的 logits gather 策略

| 平台 | 方法 | 行为 |
|------|------|------|
| CUDA | `tensor_model_parallel_gather` | 仅 rank 0 返回完整 logits，其余 rank 返回 `None` |
| TPU | `tensor_model_parallel_all_gather` | 所有 rank 都返回完整 logits（SPMD 要求） |

详见 [[LogitsProcessor#5-gather_logits-方法]]。

---

## 6. `forward` 方法

```python
def forward(
    self,
    input_ids: torch.Tensor,                                    # L1452
    positions: torch.Tensor,                                    # L1453
    intermediate_tensors: IntermediateTensors | None = None,    # L1454
    inputs_embeds: torch.Tensor | None = None,                  # L1455
) -> torch.Tensor | IntermediateTensors:                       # L1456
    hidden_states = self.model(                                 # L1458
        input_ids, positions, intermediate_tensors, inputs_embeds
    )
    return hidden_states                                        # L1460
```
（`model.py:L1451-1461`）

### 参数详情

| 参数 | shape | dtype | 来源 | 说明 |
|------|-------|-------|------|------|
| `input_ids` | `(num_tokens,)` | int32/int64 | 请求 token 序列 | 经过 PP 广播后所有 rank 获得相同值 |
| `positions` | `(num_tokens,)` | int32 | 位置编码 | 从 `attn_metadata` 计算 |
| `intermediate_tensors` | — | — | 上一 PP stage | 仅在 PP 非首 rank 传入 |
| `inputs_embeds` | `(num_tokens, hidden_size)` | bf16 | 可选 | 跳过 embedding 查找时使用 |

### 返回类型

| PP rank | 返回类型 | shape | 说明 |
|---------|---------|-------|------|
| 非末 rank | `IntermediateTensors` | `{"hidden_states": (num_tokens, hc_mult, hidden_size)}` | 传递到下一 stage |
| 末 rank | `torch.Tensor` | `(num_tokens, hidden_size)` | 最终隐藏状态（HC Head 合并 + norm 后） |

### 处理流程

```
forward(input_ids, positions, intermediate_tensors, inputs_embeds)
    │
    └─ DeepseekV4Model.forward() (model.py:L1202)
        │
        ├─ PP 首 rank：embed + unsqueeze + repeat → [num_tokens, hc_mult, hidden_size]
        ├─ 非首 rank：从 intermediate_tensors 中提取 hidden_states
        │
        ├─ 逐层推理（islice: start_layer → end_layer）
        │   └─ DecoderLayer → hc_post → 跨层 residual/post_mix/res_mix
        │
        ├─ 非末 rank：返回 IntermediateTensors
        │
        └─ 末 rank：
            ├─ 缓存 MTP 目标状态到 _mtp_hidden_buffer
            ├─ HCHeadOp 合并多流 → [num_tokens, hidden_size]
            └─ RMSNorm → 返回 final hidden_states
```

---

## 7. `get_mtp_target_hidden_states` 方法

```python
def get_mtp_target_hidden_states(self) -> torch.Tensor | None:   # L1463
    """Pre-hc_head residual stream buffer ... for the MTP draft model."""
    return getattr(self.model, "_mtp_hidden_buffer", None)       # L1467
```
（`model.py:L1463-1467`）

| 返回 | shape | dtype | device | 说明 |
|------|-------|-------|--------|------|
| `_mtp_hidden_buffer` | `(max_num_batched_tokens, hc_dim)` | bf16（`vllm_config.model_config.dtype`） | GPU | HC Head 前的多流隐藏状态 |

- 仅 PP 末 rank 持有（`model.py:L1169`）。
- 缓冲区地址固定在 CUDA graph 池外（`model.py` 注释说明），确保 `copy_` 在 CUDA graph 重放时正确更新。
- 由 `DeepseekV4Model.forward()` 在每步推理末尾填充（`model.py:L1240`）。
- [[MTP]] draft 模型通过此缓冲区获取目标模型的中间表示，用于推测解码。

---

## 8. `load_weights` 方法

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:  # L1469
    loader = AutoWeightsLoader(self, skip_substrs=["mtp."])                       # L1470
    loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)   # L1471
    self.model.finalize_mega_moe_weights()                                        # L1472
    return loaded_params                                                          # L1473
```
（`model.py:L1469-1473`）

### 参数详情

| 参数/返回 | 类型 | 说明 |
|-----------|------|------|
| `weights` | `Iterable[tuple[str, torch.Tensor]]` | 权重迭代器，每项为 `(checkpoint_name, tensor)` |
| 返回 | `set[str]` | 成功加载的 vLLM 参数名集合 |

### 三级处理机制

**第一级：`AutoWeightsLoader`**
- 自动遍历 `self.named_parameters()`，为每个 checkpoint weight 查找匹配的 vLLM 参数。
- `skip_substrs=["mtp."]`：跳过 MTP 权重。因为 [[MTP]] 权重由 `DeepSeekV4MTP.load_weights`（`mtp.py:L288`）独立加载，主模型不处理。

**第二级：`hf_to_vllm_mapper`**
- 在执行匹配前对 checkpoint weight 名称进行四层重写。详见 [[load_weights#HF-到-vLLM-的名称映射]]。
- 特别说明共享专家的权重重写：`.shared_experts.w2` → `.shared_experts.down_proj`（`model.py:L1404`），因为 `DeepseekV4MLP` 将 down_proj 命名为 `down_proj` 而非 `w2`。

**第三级：`finalize_mega_moe_weights()`**
- 委托给 `DeepseekV4Model.finalize_mega_moe_weights()`（`model.py:L1366`）。
- 对每层的 `layer.ffn.finalize_mega_moe_weights()` 调用，触发 `DeepseekV4MegaMoEExperts.finalize_weights()`（`model.py:L280`）。
- 将原始 uint8/ue8m0 格式的 FP4 专家权重转换为 DeepGEMM 内部数据格式。幂等性通过 `_transformed_l1_weights is not None` 检查保障。
- 详见 [[DeepseekV4MoE#12-2-finalize-weights-转换流程]]。

---

## 9. `get_expert_mapping` 方法

```python
def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:   # L1475
    return self.model.get_expert_mapping()                          # L1476
```
（`model.py:L1475-1476`）

- 委托给 [[DeepseekV4Model]].`get_expert_mapping()`（`model.py:L1352`）。
- 根据第一层（`first_layer.ffn.use_mega_moe`）选择映射生成方式：

| MoE 路径 | 映射生成函数 | 映射格式 |
|---------|-------------|---------|
| MegaMoE | `make_deepseek_v4_expert_params_mapping(num_experts)`（`model.py:L119`） | `(experts.w13_, experts.X.w1., expert_id, w1)` |
| FusedMoE | `FusedMoE.make_expert_params_mapping(...)`（`model.py:L1358`） | 标准 vLLM expert 映射 |

- 返回的映射列表在 `load_weights` 中用于将 checkpoint 中的专家权重分发到正确的 TP/EP rank。

---

## 10. 关键设计决策总结

| 设计决策 | 动机 | 源码证据 |
|---------|------|---------|
| `forward` 与 `compute_logits` 分离 | 支持 PP 异步调度，允许 `torch.compile` 的独立图捕获 | 见 6、5 节 |
| MTP 权重跳过 (`skip_substrs=["mtp."]`) | MTP 权重由 `DeepSeekV4MTP` 独立加载，主模型不关心 | `model.py:L1470` |
| `expert_dtype` 延迟到 `__init__` 中读取 | `DeepseekV4FP8Config` 在 `VllmConfig` 设置前创建，无法在构造时读取 hf_config | `quant_config.py:L56` |
| 类属性 `hf_to_vllm_mapper` + 实例级重写 | 默认 FP4 映射器适用于大多数 checkpoint；实例级重写处理 FP8 Base 变体 | `model.py:L1414, L1423` |
| `PPMissingLayer` 占位 | 保持 PP 各 rank 的 `named_parameters` 对齐，使权重加载器无需感知 PP 拓扑 | `model.py:L1431, L1434` |
| `make_empty_intermediate_tensors` 委托 | PP 需要创建空的中间张量用于首 stage 的输入占位 | `model.py:L1438, L1182` |

---

## 11. 量化与并行策略锚点

### 量化

- **权重精度**：所有线性层（`lm_head`、shared experts 等）的量化通过 `quant_config` 传入的 `DeepseekV4FP8Config` 控制。详见 [[QuantAndParallelStrategy#1-3-量化配置分发]]。
- **专家量化**：由 `expert_dtype` 决定——FP4（MXFP4，ue8m0 缩放）或 FP8（block 缩放，float32 缩放因子）。

### 并行

- **PP**：`SupportsPP` 混入 + `get_pp_group()` 判断首末 rank。`make_empty_intermediate_tensors` 提供 stage 间传递的中间张量。PP stage 间通过 `IntermediateTensors` 传递形状为 `(num_tokens, hc_mult, hidden_size)` 的多流状态。
- **TP**：`ParallelLMHead`（＝ `VocabParallelEmbedding`）将 lm_head 权重沿词汇表维度切分。`LogitsProcessor._gather_logits` 负责 TP gather。
- **EP**：`get_expert_mapping` 为专家并行提供权重分发映射，MegaMoE 路径强制开启 EP。

---

## 12. 边界场景与异常

| 场景 | 代码行 | 行为 |
|------|--------|------|
| `expert_dtype` 不是 `"fp4"` 或 `"fp8"` | `quant_config.py:L65-68` | 抛出 `ValueError` |
| MTP 权重缺失 | `mtp.py:L461-472` | 抛出 `ValueError`（如 checkpoint 量化时未含 MTP 层） |
| checkpoint 中 `.scale` 后缀在非专家层 | `model.py:L1377-1381` | 映射到 `.weight_scale_inv`（FP8 Linear 层） |
| FP4 专家缩放因子为 `float8_e8m0fnu` | `model.py:L1299-1303` | `.view(torch.uint8)` 避免数值转换 |

---

## 13. 与其他模块的引用关系

```
DeepseekV4ForCausalLM
  ├── DeepseekV4Model                    → [[DeepseekV4Model]]
  │     ├── DeepseekV4DecoderLayer       → [[DeepseekV4DecoderLayer]]
  │     │     ├── DeepseekV4Attention    → [[DeepseekV4Attention]]
  │     │     ├── DeepseekV4MoE          → [[DeepseekV4MoE]]
  │     │     └── HC 操作                → [[ArchitectureOverview]]
  │     ├── 权重加载                     → [[load_weights]]
  │     └── MTP 缓存                    → [[MTP]]
  ├── ParallelLMHead                     → [[ParallelLMHead]]
  ├── LogitsProcessor                    → [[LogitsProcessor]]
  └── 量化与并行配置                      → [[QuantAndParallelStrategy]]
```

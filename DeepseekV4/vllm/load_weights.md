# 权重加载（load_weights）

## 概述

DeepSeek V4 的权重加载由 `DeepseekV4ForCausalLM.load_weights` 和 `DeepseekV4Model.load_weights` 两级完成，涉及多层映射和特殊处理。

## 加载入口（`model.py:L1469-L1473`）

```python
# DeepseekV4ForCausalLM (nvidia/model.py:L1469)
def load_weights(self, weights):
    loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
    loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
    self.model.finalize_mega_moe_weights()
    return loaded_params
```

1. `skip_substrs=["mtp."]`：MTP 权重在 `DeepSeekV4MTP.load_weights` 中单独加载（`mtp.py:L288`）。
2. `hf_to_vllm_mapper`：由 `_make_deepseek_v4_weights_mapper` 根据 `expert_dtype` 动态创建。
3. `finalize_mega_moe_weights()`：所有权重加载完成后，对 MegaMoE 路径执行权重重排（DeepGEMM 布局转换）。

### 类级默认值与实例级覆盖（`model.py:L1412-L1423`）

```python
# 类属性默认假设 FP4 专家 checkpoint 布局
hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

def __init__(self, *, vllm_config, prefix=""):
    expert_dtype = getattr(config, "expert_dtype", "fp4")
    if expert_dtype != "fp4":
        # 运行时根据实际 expert_dtype 覆盖
        self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
```

- **设计意图**：大部分 checkpoint 使用 FP4，将 `"fp4"` 设为类默认值避免每次实例化都重新计算。仅在加载 FP8 专家 checkpoint（如 Flash-Base）时才在 `__init__` 中覆盖。

---

## HF 到 vLLM 的名称映射

`_make_deepseek_v4_weights_mapper(expert_dtype)`（`model.py:L1371-L1406`）返回 `WeightsMapper`，包含四层映射，按 `前缀 → 后缀 → 子串 → 正则` 依次应用。

### 前缀映射（`orig_to_new_prefix`）

| HF Checkpoint 中的前缀 | vLLM 参数名前缀 |
|------------------------|----------------|
| `layers.` | `model.layers.` |
| `embed.` | `model.embed.` |
| `norm.` | `model.norm.` |
| `hc_head` | `model.hc_head` |
| `mtp.` | `model.mtp.` |

### 后缀映射（`orig_to_new_suffix`）

| HF 后缀 | vLLM 后缀 |
|---------|-----------|
| `head.weight` | `lm_head.weight` |
| `embed.weight` | `embed_tokens.weight` |
| `.ffn.gate.bias` | `.ffn.gate.e_score_correction_bias` |

### 子串映射（`orig_to_new_substr`）

| HF 子串 | vLLM 子串 |
|---------|-----------|
| `.attn.compressor.` | `.attn.mla_attn.compressor.` |
| `.shared_experts.w2` | `.shared_experts.down_proj` |

### 缩放因子正则映射（`orig_to_new_regex`）

根据 `expert_dtype` 决定：

**FP4 专家**（`model.py:L1372-L1380`）：
```
\.experts\.\d+\.w[123]\.scale$  →  .weight_scale     # 专家权重的 E8M0 缩放因子
\.scale$                          →  .weight_scale_inv  # 其他 FP8 block 缩放因子
```

**FP8 专家**（`model.py:L1381-L1387`）：
```
\.scale$                          →  .weight_scale_inv  # 所有缩放因子统一映射
```

> **为什么 FP4 和 FP8 的 scale 映射不同？**
>
> FP4 专家使用 `Mxfp4MoEMethod`，该模块将专家权重缩放因子注册为 `w{1,2,3}_weight_scale`（无 `_inv` 后缀），存储在 uint8（E8M0 格式）中。而 FP8 线性层和共享专家使用 `Fp8LinearMethod`，其 block 缩放因子注册为 `weight_scale_inv`（float32）。FP4 路径需要两条正则分别处理两类缩放因子；FP8 路径中所有 `*.scale` 统一映射到 `*_inv`。

---

## DeepseekV4Model.load_weights 的内部处理（`model.py:L1253-L1350`）

### 堆叠参数映射（`stacked_params_mapping`）

```python
stacked_params_mapping = [
    # (param_name, shard_name, shard_id)
    ("gate_up_proj", "w1", 0),        # 前半部分：gate
    ("gate_up_proj", "w3", 1),        # 后半部分：up
    ("attn.fused_wqa_wkv", "attn.wq_a", 0),
    ("attn.fused_wqa_wkv", "attn.wkv", 1),
    ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
    ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
]
```

（`model.py:L1254-L1261`）

checkpoint 中分开存储的 `w1` / `w3`、`attn.wq_a` / `attn.wkv` 等张量在加载时通过 `shard_id` 拼接到同一个参数中。

#### 各 stacked 参数的拼接细节

| 目标参数 | Checkpoint 源 | shard_id | 拼接方向 | 拼接后的 shape |
|---------|--------------|----------|---------|---------------|
| `gate_up_proj` | `w1` (gate) | 0 | `[0:intermediate]` | `(2*intermediate, hidden)` |
| `gate_up_proj` | `w3` (up) | 1 | `[intermediate:2*intermediate]` | 同上 |
| `attn.fused_wqa_wkv` | `attn.wq_a` | 0 | `[0:q_lora_rank]` | `(q_lora_rank+head_dim, hidden)` |
| `attn.fused_wqa_wkv` | `attn.wkv` | 1 | `[q_lora_rank:q_lora_rank+kv_lora_rank]` | 同上 |
| `compressor.fused_wkv_wgate` | `compressor.wkv` | 0 | `[0:kv_lora_rank]` | `(2*kv_lora_rank, hidden)` |
| `compressor.fused_wkv_wgate` | `compressor.wgate` | 1 | `[kv_lora_rank:2*kv_lora_rank]` | 同上 |

> **设计意图**：合并小参数为单个大参数（例如 `w1` + `w3` → `gate_up_proj`）可以减少 PyTorch 的 `nn.Parameter` 数量、降低优化器状态开销（训练场景），并使得单个 `weight_loader` 调用能一次性完成加载（对推理的权重映射效率也有好处）。

### 专家权重加载（`model.py:L1293-L1329`）

```python
if ".experts." in name:
    # E8M0 scales: view to uint8
    if "weight_scale" in name and dtype == float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    for mapping in expert_mapping:
        param_name, weight_name, expert_id, expert_shard_id = mapping
        if weight_name not in name:
            continue
        name_mapped = name.replace(weight_name, param_name)
        if is_pp_missing_parameter(name_mapped, self):
            continue
        param = params_dict[name_mapped]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        success = weight_loader(
            param, loaded_weight, name_mapped,
            shard_id=expert_shard_id,
            expert_id=expert_id,
            return_success=True,
        )
        if success:
            name = name_mapped
            break
    loaded_params.add(name_mapped)
    continue
```

#### `expert_mapping` 的生成（`model.py:L1352-L1364`）

```python
def get_expert_mapping(self):
    first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
    if first_layer.ffn.use_mega_moe:
        return make_deepseek_v4_expert_params_mapping(num_experts)
    return FusedMoE.make_expert_params_mapping(...)
```

**MegaMoE 路径**（`model.py:L119-L135`）：
```python
# make_deepseek_v4_expert_params_mapping:
# 对每个 expert_id × (w1, w2, w3) 生成:
#   ("experts.w13_", f"experts.{id}.w1.", id, "w1")  ← gate
#   ("experts.w13_", f"experts.{id}.w3.", id, "w3")  ← up   → 共享 w13_ 参数
#   ("experts.w2_",  f"experts.{id}.w2.", id, "w2")  ← down
```

FusedMoE 路径使用 `FusedMoE.make_expert_params_mapping`，`ckpt_gate_proj_name="w1"`, `ckpt_down_proj_name="w2"`, `ckpt_up_proj_name="w3"`。

#### `return_success` 机制（`model.py:L1312-L1328`）

```python
success = weight_loader(
    param, loaded_weight, name_mapped,
    shard_id=expert_shard_id,
    expert_id=expert_id,
    return_success=True,  # ← 关键
)
if success:
    name = name_mapped
    break
```

**为什么需要 `return_success`？**

`expert_mapping` 是全局生成的（包含全部 `n_routed_experts` 个专家的映射条目），但当前 rank 在 EP/TP 下只持有本地专家子集。例如 `ep_size=8, n_routed_experts=256` 时，每个 rank 只加载 32 个本地专家。如果没有 `return_success`，非本地专家的权重会错误地尝试加载到本地参数中。

`DeepseekV4MegaMoEExperts.weight_loader`（`model.py:L226-L258`）内部通过 `_map_global_expert_id` 判断：
```python
def _map_global_expert_id(self, expert_id: int) -> int:
    if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
        return -1          # ← 非本地专家
    return expert_id - self.experts_start_idx  # ← 映射到本地索引
```

当 `return_success=True` 且 `expert_id` 不在本地范围时，返回 `False`，上层循环继续尝试下一个映射条目。

### E8M0 缩放因子的类型转换（`model.py:L1299-L1303`）

```python
if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
    loaded_weight = loaded_weight.view(torch.uint8)
```

checkpoint 中 E8M0 缩放因子以 `float8_e8m0fnu` 类型存储（实质是 8 位无符号指数）。直接 `copy_` 会触发数值转换（例如 `2^(-7)` → `0`），破坏原始指数字节。因此必须通过 `.view(torch.uint8)` 按字节 reinterpret。

#### E8M0 格式详解

ue8m0（e8m0fnu）是**纯指数格式**——8 位指数，零位尾数：
- 值范围：`[0, 255]`，表示指数 E
- 实际缩放因子：`2^(E-127)`（转换见 `DeepseekV4MegaMoEExperts._ue8m0_uint8_to_float`，`model.py:L260-L262`）
- 在 checkpoint 中存储为 `float8_e8m0fnu` 类型，但 `view(torch.uint8)` 后按字节读取

```python
@staticmethod
def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
    # uint8 → int32 → 左移 23 位（填入 IEEE 754 float32 的指数部分）→ reinterpet as float32
    return (sf.to(torch.int32) << 23).view(torch.float32)
```

> **为什么不能用 `loaded_weight.float()`？**
> PyTorch 的 `float8_e8m0fnu → float32` 转换会执行实际的数值转换：`2^(-7)` → `0.0078125`，丢失了"缩放因子是纯指数"的语义。`view` 完全绕过了数值转换，保留原始位模式。这是 DeepSeek V4 checkpoint 特有的 hack。

### attn_sink 的 TP 切分（`model.py:L1331-L1337`）

```python
elif "attn_sink" in name:
    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
    n = narrow_weight.shape[0]
    params_dict[name][:n].copy_(narrow_weight)
```

`attn_sink` 的参数是按 `padded_heads` 分配的（`≥ n_local_heads`，对齐到 64 的倍数），但 checkpoint 存储的是全局完整的 `n_heads` 个头的 sink。这里：

1. 按 TP rank 的头范围窄取（`head_rank_start:head_rank_end`）
2. 复制到参数的前 `n_local_heads` 个位置
3. 剩余位置保持初始化值（`-inf`）

**为什么保留剩余位置为 `-inf`？** FlashMLA 内核要求 head 数量对齐到 `padded_heads`（64 的倍数）。对齐多出的位置必须填充为 `-inf`，以避免它们在 softmax 中贡献非零概率。TP 切分后，每个 rank 的参数总大小仍为 `padded_heads`，但有效数据只占前 `n_local_heads` 个位置。

### 默认加载路径（`model.py:L1339-L1348`）

```python
else:
    if is_pp_missing_parameter(name, self):
        continue
    param = params_dict[name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)
    loaded_params.add(name)
```

对于非 stacked、非 expert、非 attn_sink 的常规参数，使用参数的 `weight_loader` 属性或 `default_weight_loader`。`is_pp_missing_parameter` 检查该参数是否属于当前 PP rank 不需要的层（例如非末 rank 的 `lm_head`）。

---

## MegaMoE 权重后处理（`model.py:L1366-L1368`）

```python
def finalize_mega_moe_weights(self) -> None:
    for layer in islice(self.layers, self.start_layer, self.end_layer):
        layer.ffn.finalize_mega_moe_weights()
```

调用链：
```
DeepseekV4ForCausalLM.load_weights(L1472)
  └→ DeepseekV4Model.finalize_mega_moe_weights(L1366)
       └→ 对本 PP stage 的每层:
            DeepseekV4MoE.finalize_mega_moe_weights(L578-580)
              └→ self.experts.finalize_weights()
                   └→ DeepseekV4MegaMoEExperts.finalize_weights(L280)
```

### `finalize_weights()` 内部流程（`model.py:L280-L316`）

```python
def finalize_weights(self) -> None:
    if self._transformed_l1_weights is not None:
        return  # 已转换，跳过（幂等）

    self._check_runtime_supported()  # SM100 + CUDA 检查

    # 1. 将 E8M0 uint8 缩放因子转换为 float32
    w13_scale = self._ue8m0_uint8_to_float(self.w13_weight_scale.data)

    # 2. 转换缩放到 DeepGEMM 要求的布局
    w13_scale = deep_gemm.transform_sf_into_required_layout(
        w13_scale, 2*intermediate, hidden, block_shape=(1,32), num_local_experts)
    w2_scale = deep_gemm.transform_sf_into_required_layout(...)

    # 3. 转换权重到 DeepGEMM 内部布局（L1 权重重排 + L2 权重别名）
    self._transformed_l1_weights, self._transformed_l2_weights = (
        deep_gemm.transform_weights_for_mega_moe(
            (w13_weight_int8, w13_scale), (w2_weight_int8, w2_scale))
    )

    # 4. 丢弃原始参数（L1 已重新分配，L2 由 _transformed 持有引用）
    self.w13_weight = None
    self.w13_weight_scale = None
    self.w2_weight = None
    self.w2_weight_scale = None
```

**设计意图**：
- DeepGEMM 需要权重的特定内存布局（`_interleave_l1_weights` 对 L1 权重做交错排列），加载后的原始布局不能直接用于计算。
- `w13_weight` 等原始参数在 `finalize_weights` 后被设为 `None`，因为它们的存储已由 `_transformed_l1_weights` 和 `_transformed_l2_weights` 接管。L2 weight 实际上是原始存储的别名（浅拷贝），因此 L2 的存储不会因 `w2_weight=None` 而被回收。
- `finalize_weights` 在 `_run_mega_moe` 中也调用一次（`model.py:L396`），覆盖 dummy weight 加载的 case。

---

## MTP 的独立权重加载

`DeepSeekV4MTP.load_weights`（`mtp.py:L288`）是一个独立的加载器，包含：

1. **层索引重写**：checkpoint 中的 `mtp.{i}.*` 映射到 `model.layers.{num_hidden_layers + i}.*`
2. **`_rewrite_spec_layer_name`**：区分共享权重（`embed_tokens`）和层内权重（添加 `.mtp_block` 前缀）
3. **缺失层检查**：加载后验证所有 MTP 层的权重都已加载（`mtp.py:L461-472`），否则抛出 `ValueError`
4. **与主模型相同的 E8M0 类型转换**和专家映射

---

## 调试与验证

### 打印映射前后的权重名

在 `DeepseekV4Model.load_weights` 循环中插入临时日志可观察映射效果：

```python
# 在 model.py:L1277 后插入
if "experts.0.w1" in name:
    print(f"  HF name: {name}")
    print(f"  vLLM name: {name_mapped}")
    # FP4:  "experts.0.w1.scale" → "experts.w13_.weight_scale"
    # FP8:  "experts.0.w1.scale" → "experts.w13_.weight_scale_inv"
```

### 验证专家权重加载

```python
# 检查当前 rank 的本地专家权重
moe_layer = model.model.layers[0].ffn
if moe_layer.use_mega_moe:
    # finalize 前: 检查 w13_weight shape
    print(f"Local experts: {moe_layer.experts.num_local_experts}")
    print(f"w13_weight shape: {moe_layer.experts.w13_weight.shape}")
    print(f"w13_weight dtype: {moe_layer.experts.w13_weight.dtype}")  # uint8
else:
    # FusedMoE 路径: 检查 weight_loader 加载状态
    print(f"Gate shape: {moe_layer.gate.weight.shape}")
```

### finalize 前后的权重值对比

```python
# finalize_mega_moe_weights() 前
raw_w13 = moe_layer.experts.w13_weight.data
print("Raw w13 [0,0,:4]:", raw_w13[0,0,:4])  # uint8 打包的 FP4

# finalize_mega_moe_weights() 后
print("Transformed L1 exists:", moe_layer.experts._transformed_l1_weights is not None)
print("Transformed L1 shape:", moe_layer.experts._transformed_l1_weights.shape)
# 注意: 此时 w13_weight 已被设为 None
```

### 常见错误及根因

| 错误 | 可能原因 | 排查方向 |
|------|---------|---------|
| `KeyError: 'model.layers.X.ffn.experts.w13_.weight_scale'` | FP4 checkpoint 使用了 FP8 mapper | 检查 `expert_dtype` 配置 |
| `Shape mismatch for expert weight` | expert_dtype 与 checkpoint 实际格式不符 | 检查 checkpoint 中 expert 权重的 dtype |
| Missing key(s): `mtp.*` | MTP 权重未加载 | 确认 `skip_substrs` 排除了 MTP，且 `DeepSeekV4MTP.load_weights` 被调用 |
| `ValueError: DeepGEMM MegaMoE requires SM100` | 非 Blackwell GPU 上尝试加载 MegaMoE | 检查 `use_mega_moe` 是否被误启用 |
| E8M0 scale 值全为 0 | `view(uint8)` 未被调用，`copy_` 做了数值转换 | 确认 `weight_scale` 分支中执行了 `view` |

---

## 权重加载流程总结

```
weights iterator (HF format)
    │
    ├─ hf_to_vllm_mapper (前缀/后缀/子串/正则替换)
    │   └─ FP4 路径: 两条正则分别处理 expert scale vs linear scale
    │   └─ FP8 路径: 统一映射
    │
    ├─ DeepseekV4ForCausalLM.load_weights
    │   └─ AutoWeightsLoader (跳过 "mtp.")
    │       └─ 对每个 weight 调用 DeepseekV4Model.load_weights:
    │           ├─ stacked_params_mapping？→ 按 shard_id 拼接参数 (L1278-L1292)
    │           ├─ .experts.？→ expert_mapping 分发 (L1293-L1329)
    │           │   ├─ weight_scale + float8_e8m0fnu → view(uint8) (L1299-L1303)
    │           │   └─ return_success 过滤非本地专家 (L1318-L1326)
    │           ├─ attn_sink？→ TP head 维度窄取 + padding (L1331-L1337)
    │           └─ 其他 → 默认 weight_loader (L1339-L1348)
    │
    └─ finalize_mega_moe_weights() (L1472)
        └─ 对每层:
            DeepseekV4MoE.finalize_mega_moe_weights()
              └─ DeepseekV4MegaMoEExperts.finalize_weights()
                  ├─ E8M0 uint8 → float32 缩放因子转换
                  ├─ transform_sf_into_required_layout (DeepGEMM 布局)
                  ├─ transform_weights_for_mega_moe (L1 重排 + L2 别名)
                  └─ 丢弃原始 uint8 参数
```

## Cross-References

- [[DeepseekV4MoE]]：MegaMoE 权重变换和 `finalize_weights` 的完整逻辑
- [[DeepseekV4MegaMoEExperts]]：`weight_loader` 的具体实现和 `_map_global_expert_id` 路由
- [[MTP]]：MTP 权重的独立加载流程
- [[QuantAndParallelStrategy]]：量化格式（FP4 vs FP8）如何影响权重映射和 scale 处理
- [[DeepseekV4ForCausalLM]]：`load_weights` 的主入口

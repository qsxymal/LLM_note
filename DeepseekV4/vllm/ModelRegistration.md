# ModelRegistration -- DeepseekV4ForCausalLM 如何注册到 vLLM

## 注册表字典（TEXT_GENERATION_MODELS）

文件：`vllm/model_executor/models/registry.py`

中央分派表是 `_TEXT_GENERATION_MODELS`（从第 71 行开始）。它将 HuggingFace `architectures` 字符串映射到 `(module_path, class_name)` 元组：

```python
# Line 101
"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
# Line 616
"DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
```

关键点：
- 该集合名为 `_TEXT_GENERATION_MODELS`，包含所有因果 LM 架构，与 `_VISION_MODELS` 和 `_SPEC_DECODING_MODELS` 分开。
- 模块路径 `"vllm.models.deepseek_v4"` 使用硬件隔离布局（位于 `vllm/models/deepseek_v4/` 下），而不是大多数其他模型使用的扁平 `vllm.model_executor.models.<name>` 约定。
- MTP 变体注册（第 616 行）使用 `"DeepSeekV4MTPModel"`（Seek 中的 S 大写）作为 HF 架构键，指向 `DeepSeekV4MTP`。

## 解析流程

`_ModelRegistry.resolve_model_cls`（第 1201 行）是将 HF `architectures` 列表转换为具体 `nn.Module` 类的入口点。

1. 从 `hf_config.architectures[0]` 接收 `architectures`（列表中的第一个架构）。
2. 遍历架构，通过 `_normalize_arch` 规范化每个架构。
3. 调用 `_try_load_model_cls(normalized_arch)`（第 1238 行）。
4. `_try_load_model_cls`（第 928 行）带有 `@lru_cache` 装饰器，并调用 `model.load_model_cls()`。
5. 对于延迟注册的模型（`_LazyRegisteredModel` 数据类，第 807 行），`load_model_cls()`（第 922 行）执行：

```python
mod = importlib.import_module(self.module_name)  # "vllm.models.deepseek_v4"
return getattr(mod, self.class_name)              # "DeepseekV4ForCausalLM"
```

这会触发 `vllm/models/deepseek_v4/__init__.py`，这是平台切换的入口点。

## 平台特定的重新导出（`__init__.py`）

文件：`vllm/models/deepseek_v4/__init__.py`（第 1-31 行）

```python
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
else:
    from .amd.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .amd.mtp import DeepSeekV4MTP            # type: ignore[assignment]
```

- **类型检查时（mypy/Pyright）：** NVIDIA 类是类型检查器看到的静态默认值。
- **CUDA 运行时：** `current_platform.is_rocm()` 返回 `False`，因此使用 NVIDIA 类。
- **ROCm 运行时：** `current_platform.is_rocm()` 返回 `True`，AMD 类动态替换 NVIDIA 类。`# type: ignore[assignment]` 抑制类型检查器警告。
- `DeepseekV4FP8Config` 无条件导入（第 14 行）来自 `vllm/models/deepseek_v4/quant_config.py`，因此无论平台如何，它始终可用于 quant_config 自动检测。
- `__all__`（第 26-30 行）导出所有三个公开名称：`DeepSeekV4MTP`、`DeepseekV4FP8Config`、`DeepseekV4ForCausalLM`。

## QuantConfig 自动检测

文件：`vllm/config/speculative.py`（第 314 行）

当 `hf_config.model_type == "deepseek_v4"` 时：
- MTP 推测器配置将 `model_type` 重写为 `"deepseek_mtp"`，并将 `architectures` 设置为 `["DeepSeekV4MTPModel"]`（第 314-319 行）。
- 这导致 `resolve_model_cls` 为 MTP 头加载 `DeepSeekV4MTP`。

另外，`DeepseekV4FP8Config` 自动检测由量化配置解析路径中的 `model_type="deepseek_v4"` 触发（在 `vllm/config.py` 中），该路径检查模型是否使用 FP8 KV 缓存和权重，并注册适当的量化方法。

## 加载权重覆盖

`DeepseekV4ForCausalLM.load_weights`（nvidia/model.py 第 1253 行）覆盖了默认的权重加载：
- **堆叠参数**（`attn.fused_wqa_wkv`、`gate_up_proj`、`compressor.fused_wkv_wgate`）通过 `stacked_params_mapping` 解堆叠（第 1254-1261 行）。
- **专家权重**通过 `get_expert_mapping()` 的 `expert_mapping` 分派（第 1352 行），该映射处理 MegaMoE 和标准 FusedMoE 布局。
- **MTP 权重跳过**：如果 `".experts."` 在权重名称中但未匹配任何本地专家 ID，则静默跳过（第 1294-1330 行）。
- **E8M0 缩放因子字节**通过在 `copy_` 之前将 `float8_e8m0fnu` 张量视为 `uint8` 来保留（第 1299-1303 行），防止数值转换。

`hf_to_vllm_mapper` 类属性（第 1414 行）使用 `WeightsMapper`（第 1371 行）处理检查点键重映射：
- 前缀映射：`"layers."` -> `"model.layers."`，`"embed."` -> `"model.embed."` 等。
- 后缀映射：`"head.weight"` -> `"lm_head.weight"`。
- 子串映射：`".attn.compressor."` -> `".attn.mla_attn.compressor."`。
- 用于 FP4 与 FP8 专家数据类型的缩放因子正则表达式。

## 设计总结

| 步骤 | 发生了什么 |
|---|---|
| 1. 读取 HF 配置 | `config.architectures[0]` = `"DeepseekV4ForCausalLM"` |
| 2. 注册表查找 | `_TEXT_GENERATION_MODELS["DeepseekV4ForCausalLM"]` -> `("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")` |
| 3. 延迟导入 | `importlib.import_module("vllm.models.deepseek_v4")` 触发 `__init__.py` |
| 4. 平台切换 | `__init__.py` 根据 `current_platform.is_rocm()` 导入 NVIDIA 或 AMD 变体 |
| 5. 返回类 | `getattr(mod, "DeepseekV4ForCausalLM")` 返回平台特定的类 |
| 6. 权重加载 | `load_weights()` 处理堆叠参数、专家映射、MTP 跳过、缩放因子 dtype 保留 |

相关笔记：[[DeepseekV4ForCausalLM]], [[DeepseekV4FP8Config]], [[load_weights]], [[MTP]]

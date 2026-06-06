# ModelRegistration -- How DeepseekV4ForCausalLM is registered into vLLM

## The Registry Dict (TEXT_GENERATION_MODELS)

File: `vllm/model_executor/models/registry.py`

The central dispatch is `_TEXT_GENERATION_MODELS` (line 71 onward). It maps HuggingFace `architectures` strings to `(module_path, class_name)` tuples:

```python
# Line 101
"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
# Line 616
"DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
```

Key points:
- The set is named `_TEXT_GENERATION_MODELS` containing all causal LM architectures, distinct from `_VISION_MODELS` and `_SPEC_DECODING_MODELS`.
- The module path `"vllm.models.deepseek_v4"` uses the hardware-isolated layout (under `vllm/models/deepseek_v4/`) rather than the flat `vllm.model_executor.models.<name>` convention used by most other models.
- The MTP variant is registered (line 616) using `"DeepSeekV4MTPModel"` (capital S in Seek) as the HF architecture key pointing to `DeepSeekV4MTP`.

## Resolution Flow

`_ModelRegistry.resolve_model_cls` (line 1201) is the entry point that converts an HF `architectures` list into a concrete `nn.Module` class.

1. Receives `architectures` from `hf_config.architectures[0]` (the first architecture in the list).
2. Iterates through architectures, normalizing each via `_normalize_arch`.
3. Calls `_try_load_model_cls(normalized_arch)` (line 1238).
4. `_try_load_model_cls` (line 928) is `@lru_cache`-decorated and calls `model.load_model_cls()`.
5. For lazy-registered models (the `_LazyRegisteredModel` dataclass, line 807), `load_model_cls()` (line 922) does:

```python
mod = importlib.import_module(self.module_name)  # "vllm.models.deepseek_v4"
return getattr(mod, self.class_name)              # "DeepseekV4ForCausalLM"
```

This triggers `vllm/models/deepseek_v4/__init__.py`, which is the platform-switching entry point.

## Platform-Specific Re-Export (`__init__.py`)

File: `vllm/models/deepseek_v4/__init__.py` (lines 1-31)

```python
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
else:
    from .amd.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .amd.mtp import DeepSeekV4MTP            # type: ignore[assignment]
```

- **Type-checking time (mypy/Pyright):** NVIDIA classes are the static default that type-checkers see.
- **Runtime on CUDA:** `current_platform.is_rocm()` returns `False`, so NVIDIA classes are used.
- **Runtime on ROCm:** `current_platform.is_rocm()` returns `True`, AMD classes dynamically replace the NVIDIA ones. `# type: ignore[assignment]` suppresses type-checker warnings.
- `DeepseekV4FP8Config` is imported unconditionally (line 14) from `vllm/models/deepseek_v4/quant_config.py` so it is always available for quant_config auto-detection regardless of platform.
- `__all__` (line 26-30) exports all three public names: `DeepSeekV4MTP`, `DeepseekV4FP8Config`, `DeepseekV4ForCausalLM`.

## QuantConfig Auto-Detection

File: `vllm/config/speculative.py` (line 314)

When `hf_config.model_type == "deepseek_v4"`:
- The MTP speculator config rewrites `model_type` to `"deepseek_mtp"` and sets `architectures` to `["DeepSeekV4MTPModel"]` (lines 314-319).
- This causes `resolve_model_cls` to load `DeepSeekV4MTP` for the MTP head.

Separately, `DeepseekV4FP8Config` auto-detection is triggered by `model_type="deepseek_v4"` in the quant config resolution path (in `vllm/config.py`), which checks if the model uses FP8 KV cache and weights, registering the appropriate quant method.

## Load Weights Override

`DeepseekV4ForCausalLM.load_weights` (nvidia/model.py line 1253) overrides the default weight loading:
- **Stacked params** (`attn.fused_wqa_wkv`, `gate_up_proj`, `compressor.fused_wkv_wgate`) are unstacked via `stacked_params_mapping` (lines 1254-1261).
- **Expert weights** are dispatched via `expert_mapping` from `get_expert_mapping()` (line 1352), which handles both MegaMoE and standard FusedMoE layouts.
- **MTP weight skipping**: If `".experts."` is in the weight name and it does not match any local expert ID, it is silently skipped (lines 1294-1330).
- **E8M0 scale bytes** are preserved by viewing `float8_e8m0fnu` tensors as `uint8` before `copy_` (lines 1299-1303), preventing numeric conversion.

The `hf_to_vllm_mapper` class attribute (line 1414) uses `WeightsMapper` (line 1371) to handle checkpoint key remapping:
- Prefix mapping: `"layers."` -> `"model.layers."`, `"embed."` -> `"model.embed."`, etc.
- Suffix mapping: `"head.weight"` -> `"lm_head.weight"`.
- Substring mapping: `".attn.compressor."` -> `".attn.mla_attn.compressor."`.
- Scale regex for FP4 vs FP8 expert dtypes.

## Design Summary

| Step | What happens |
|---|---|
| 1. HF config read | `config.architectures[0]` = `"DeepseekV4ForCausalLM"` |
| 2. Registry lookup | `_TEXT_GENERATION_MODELS["DeepseekV4ForCausalLM"]` -> `("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM")` |
| 3. Lazy import | `importlib.import_module("vllm.models.deepseek_v4")` triggers `__init__.py` |
| 4. Platform switch | `__init__.py` imports NVIDIA or AMD variant based on `current_platform.is_rocm()` |
| 5. Class returned | `getattr(mod, "DeepseekV4ForCausalLM")` returns the platform-specific class |
| 6. Weight loading | `load_weights()` handles stacked params, expert mapping, MTP skipping, scale dtype preservation |

Related notes: [[DeepseekV4ForCausalLM]], [[DeepseekV4FP8Config]], [[load_weights]], [[MTP]]

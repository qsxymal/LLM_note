# NVIDIA vs AMD 平台对比

DeepSeek V4 的 `__init__.py` 根据 `current_platform.is_rocm()` 自动选择 NVIDIA 或 AMD 实现。本文对比两个平台实现的关键差异。

## 1. 选择机制（`__init__.py:L19-L24`）

```python
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
else:
    from .amd.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .amd.mtp import DeepSeekV4MTP
```

静态类型检查（mypy）始终看到 NVIDIA 实现。AMD 实现在运行时动态替换，通过 `# type: ignore[assignment]` 抑制类型错误。

## 2. MoE 实现对比

| 特性 | NVIDIA | AMD |
|------|--------|-----|
| MegaMoE 路径 | ✅ FP4 权重 + FP8 激活，DeepGEMM 内核 | ❌ 不支持 |
| FusedMoE 路径 | ✅ 支持 FP4/FP8/BF16 | ✅ 始终使用 FusedMoE（仅 FP8/BF16） |
| 专家并行（EP） | ✅ MegaMoE 路径支持 | ❌ 仅 TP（`n_local_experts = n_routed / tp_size`） |
| 共享专家 | ✅ `reduce_results=False` 的 MLP | ✅ 相同 |
| Gate 路由 | ✅ 完全相同（`GateLinear` + `fused_topk_bias`） | ✅ 完全相同 |

**影响**：AMD 平台不支持 SM100 专属的 MegaMoE 路径，因此：
- 不会使用 FP4 专家权重
- 不支持专家并行（EP），使用 TP 替代
- `DeepseekV4MegaMoEExperts` 类不存在

## 3. 解码器层对比

| 特性 | NVIDIA | AMD |
|------|--------|-----|
| `_forward_cuda` | ✅ 使用 `mhc_fused_post_pre` 融合 | ✅ 相同融合路径 |
| `_forward_native` | ✅ 分步执行，返回 `(x, None, None, None)` | ❌ 使用 `_forward_rocm` |
| `_forward_rocm` | ❌ 不存在 | ✅ ROCm 专用回退路径 |
| HC 算子 | `torch.ops.vllm.mhc_*`（CUDA） | `torch.ops.vllm.mhc_*`（ROCm 绑定） |
| 3 辅助流 | ✅ 是 | ❌ `None`（串行执行） |

AMD 的 `_forward_rocm` 在结构上与 `_forward_native` 相同，但：
```python
# amd/model.py:L558-L583
def _forward_rocm(self, x, positions, input_ids, ...):
    residual = x
    x, post, comb = self.hc_pre(x, hc_attn_*)      # 独立调用 hc_pre
    x = self.attn_norm(x)                            # 显式 norm
    x = self.attn(positions, x, None)
    x = self.hc_post(x, residual, post, comb)

    residual = x
    x, post, comb = self.hc_pre(x, hc_ffn_*)        # 独立调用 hc_pre
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post(x, residual, post, comb)
    return x, residual, post_mix, res_mix    # 注意：返回了 post_mix/res_mix！
```

**与 `_forward_native` 的关键区别**：AMD 的 `_forward_rocm` 返回 `(x, residual, post_mix, res_mix)`（四个值），而 NVIDIA 的 `_forward_native` 返回 `(x, None, None, None)`。这是因为 ROCm 可能在某些场景下仍需要跨层状态传递。

## 4. 注意力实现对比

两者的 `DeepseekV4Attention` 和 `DeepseekV4MultiHeadLatentAttentionWrapper` 基本相同，共享同一个 `attention.py` 文件。差异在于：

| 特性 | NVIDIA | AMD |
|------|--------|-----|
| MLA 计算路径 | FlashMLA（SM90+） | 回退到通用注意力（可能使用 Triton） |
| FP8 KV 缓存 | ✅ 强制启用 | ✅ 强制启用 |
| 多流并行 | ✅ 3 个辅助 CUDA 流 | ❌ 无辅助流 |
| CuteDSL 内核 | ✅ `dequant_gather_k`、`fused_indexer_q` 等 | ❌ 使用 `common.ops` 中的替代实现 |

## 5. 权重加载对比

| 特性 | NVIDIA | AMD |
|------|--------|-----|
| `stacked_params_mapping` | ✅ 相同 | ✅ 相同 |
| 专家映射 | ✅ 双路径（MegaMoE / FusedMoE） | ✅ 仅 FusedMoE 路径 |
| E8M0 view(uint8) | ✅ | ✅ |
| attn_sink 切分 | ✅ | ✅ |
| `return_success` | ✅ | ✅ |
| `finalize_mega_moe_weights` | ✅ | ❌ 不需要 |
| `_make_deepseek_v4_weights_mapper` | ✅（E8M0 感知） | ❌ 可能使用固定映射 |

## 6. MTP 对比

| 特性 | NVIDIA | AMD |
|------|--------|-----|
| 类名 | `DeepSeekV4MTP` | `DeepSeekV4MTP` |
| 权重加载 | 完整实现（含 expert mapping） | 完整实现（仅 FusedMoE 路径） |
| HC Head | ✅ | ✅ |
| 辅助流 | ✅ 3 个 | ❌ `None` |

## 7. 总结

| 维度 | NVIDIA 优势 | AMD 优势 |
|------|-----------|---------|
| MoE 性能 | MegaMoE（FP4 + EP）大幅降低显存和通信 | 无 |
| 注意力性能 | FlashMLA + 多流并行 + CuteDSL 内核 | 无 |
| 兼容性 | SM100 专属，H100 回退 | 原生 ROCm 支持 |
| 代码复杂度 | 更高（多路径、多内核） | 更低（单一 FusedMoE 路径） |
| 维护策略 | 积极性能优化（CuteDSL、DeepGEMM） | 功能等价，跟随上游 |

**核心结论**：AMD 版本是 NVIDIA 版本的功能等价子集——去掉了 SM100 专属的优化（MegaMoE、CuteDSL 内核、多流并行），保留完整的模型结构和推理正确性。NVIDIA 的算子通过 `torch.ops.vllm.*` 注册，AMD 平台可提供同名 ROCm 绑定。

## Cross-References

- [[DeepseekV4MoE]]：MegaMoE vs FusedMoE 路径细节
- [[DeepseekV4DecoderLayer]]：`_forward_cuda` vs `_forward_native` / `_forward_rocm`
- [[DeepseekV4MultiHeadLatentAttentionWrapper]]：多流并行（仅 NVIDIA）
- [[PrepareMegaMoEInputs]]：MegaMoE 专用的 Triton 内核（仅 NVIDIA）
- [[DequantGatherKCache]]、[[FusedIndexerQ]]、[[SparseAttnCompress]]：CuteDSL 内核族（仅 NVIDIA）

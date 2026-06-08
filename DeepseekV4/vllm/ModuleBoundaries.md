# 模块边界与协同机制

本文分析 DeepSeek V4 各模块之间如何定义边界、如何传递数据、如何协同工作。

---

## 1. 总体架构的分层边界

```
ForCausalLM          ← 模型顶层外观
    │
    ├─ Model         ← Transformer 主干（嵌入 + 层列表 + norm + HC 头）
    │     │
    │     ├─ DecoderLayer  ← 单层：HC 融合 + 子层编排
    │     │    ├─ Attention    ← 注意力组件工厂（结构声明）
    │     │    │    └─ Wrapper    ← MLA 计算编排（PluggableLayer）
    │     │    │         ├─ Compressor  ← KV 压缩（Wrapper 内部）
    │     │    │         ├─ Indexer     ← 稀疏索引（Attention 声明）
    │     │    │         └─ MLAAttention ← 核心抽象（委托给 Impl）
    │     │    │              └─ FlashMLASparseImpl ← 内核调度（Impl）
    │     │    └─ MoE          ← 混合专家 FFN
    │     │
    │     ├─ norm（最终）
    │     └─ hc_head_op（多流合并）
    │
    ├─ lm_head       ← 词表投影头
    └─ LogitsProcessor ← logits 后处理
```

**关键边界原则**：
- 上层不知道下层的内部细节（例如 `Model` 不知道 `MoE` 内部是用 MegaMoE 还是 FusedMoE）
- 同层模块通过中间张量交互（`DecoderLayer` 内部 `Attention → MoE` 通过 `hidden_states`）
- 跨层状态通过返回值传递（`residual`, `post_mix`, `res_mix` 在 `DecoderLayer` 间流动）

---

## 2. 平台隔离边界

### `__init__.py` 的运行时调度（`deepseek_v4/__init__.py:L19-L24`）

```python
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
else:
    from .amd.model import DeepseekV4ForCausalLM
    from .amd.mtp import DeepSeekV4MTP
```

**边界定义**：平台选择发生在模块导入级，**一旦进入运行时，整个模型要么是 NVIDIA 实现要么是 AMD 实现**，不存在混合。`TYPE_CHECKING` 时始终走 NVIDIA 分支以便 mypy 静态分析。

### `common/ops/` vs `nvidia/ops/` 的边界

| 方面 | `common/ops/` | `nvidia/ops/` |
|------|---------------|---------------|
| 依赖 | 纯 PyTorch/Triton | CuteDSL/CUTLASS |
| 导入时机 | 正常导入 | 惰性导入（门控在 `has_cutedsl()` 后） |
| 导出模式 | `__init__.py` 导出 9 个符号 | `__init__.py` 为空 |
| CUDA 驱动影响 | 无 | `import cutlass` 初始化 CUDA 驱动 |

`nvidia/ops/` 故意为空 `__init__.py`（[[PrepareMegaMoEInputs#7-关于-nvidia-ops-init-py-的设计说明]]），防止 fork 子进程中的 CUDA 驱动初始化问题。

### 注意力实现的平台选择

`_select_v4_sparse_impl()`（`attention.py:L79-L91`）在 `DeepseekV4MLAAttention.__init__` 时调用一次，确定当前平台使用哪个 Impl 类。这是注意力子系统的**单次固定派发**。

### HC 后端的平台选择

`DeepseekV4DecoderLayer.forward`（`nvidia/model.py:L1055-L1071`）：
- CUDA：`_forward_cuda`（融合路径，使用 `mhc_fused_post_pre`）
- ROCm/XPU：`_forward_native`（顺序路径，`hc_pre → sublayer → hc_post`）

---

## 3. 数据契约边界

### 解码器层的接口契约（`nvidia/model.py:L958-L965`）

```python
def _forward_cuda(
    x: (B, hc_mult, H) bf16,
    positions: (B,) int64,
    input_ids: (B,) int64 | None,
    post_mix: (B, mix_hc) float32 | None,
    res_mix: (B, mix_hc) float32 | None,
    residual: (B, hc_mult, H) bf16 | None,
) -> tuple[(B, hc_mult, H), (B, mix_hc), (B, mix_hc), (B, hc_mult, H)]:
```

**边界约束**：
- `residual` 在首层为 None → 触发独立 `hc_pre`（`nvidia/model.py:L969`）
- 非首层必须传入非 None 的 `residual`, `post_mix`, `res_mix`（来自上一层）
- `input_ids` 在 Hash MoE 路由时必须非 None
- 返回值始终为 4 元组

### PP stage 间张量契约

`IntermediateTensors`（`nvidia/model.py:L1182-L1200`）：
```python
{"hidden_states": (B, hc_mult, H) bf16}
```

PP 末 rank 处理后返回 `(B, H) bf16` 给 `lm_head`。

### MLA 注意力子系统的内部契约

`DeepseekV4MLAModules`（`attention.py:L94-L111`）是结构层（`DeepseekV4Attention`）和执行层（`Wrapper`）之间的数据契约：

```python
@dataclass
class DeepseekV4MLAModules:
    fused_wqa_wkv: Module      # MergedColumnParallelLinear
    q_norm: Module              # RMSNorm
    wq_b: Module                # ColumnParallelLinear
    kv_norm: Module             # RMSNorm
    wo_a: Module                # ColumnParallelLinear
    wo_b: Module                # RowParallelLinear
    attn_sink: Module           # sink attention parameters
    rotary_emb: Module          # RotaryEmbedding
    indexer: Module | None      # DeepseekV4Indexer 或 None
    indexer_rotary_emb: Module
    topk_indices_buffer: Tensor | None
    aux_stream_list: list[Stream] | None
```

**边界意义**：`Attention` 准备好所有子模块，`Wrapper` 使用它们。`indexer=None` 表示该层不需要稀疏注意力（C128A 或 SWA-only）。

---

## 4. 协同模式

### 模式 A：逐层委派

```
ForCausalLM.forward → Model.forward → [DecoderLayer.forward × num_layers]
```

这是最基本的模式，每层封装自己的计算逻辑，Model 负责循环驱动。

### 模式 B：组件工厂 + 计算编排

`DeepseekV4Attention` 创建 `DeepseekV4MultiHeadLatentAttentionWrapper`，然后 `forward` 完全委派给 Wrapper。

**协同数据流**：
```
Attention.__init__
  ├─ 创建投影层、norm、RoPE
  ├─ 创建 Indexer（如果 compress_ratio == 4）
  ├─ 打包 DeepseekV4MLAModules
  └─ 创建 Wrapper（传入 mla_modules）

Attention.forward → Wrapper.forward
  ├─ Wrapper.attn_gemm_parallel_execute（多流并行 GEMM）
  ├─ Wrapper.attention_impl（norm + RoPE + KV insert + indexer + compressor + MLA）
  └─ Wrapper.forward（O 投影：inv_rope + fp8 einsum + wo_b）
```

### 模式 C：策略模式 + 平台 Impl

```
DeepseekV4MLAAttention.forward（抽象基类）
  └─ self.impl_cls.forward_mqa(self, q, kv, positions, output)
       ├─ NVIDIA: DeepseekV4FlashMLASparseImpl.forward_mqa
       └─ ROCm: DeepseekV4ROCMAiterMLASparseImpl.forward_mqa
```

`Impl` 类的选择在 `MLAAttention.__init__` 时通过 `_select_v4_sparse_impl()` 固定。

### 模式 D：Custom Op + 静态 forward context

```
torch.ops.vllm.deepseek_v4_attention(hidden_states, positions, out, layer_name)
  └─ deepseek_v4_attention()                 # custom op 函数
       └─ forward_context.no_compile_layers[layer_name]  # 静态查找
            └─ Wrapper.attention_impl(...)    # 实际执行
```

**为什么用 custom op 而非直接调用**：`torch.compile` 能 将 custom op 作为一个整体进行融合和优化，但如果直接调用 Python 方法则会尝试追踪内部图，导致各种编译问题。

**`no_compile_layers`** 是 `ForwardContext` 中的一个字典，存储了所有不应该被 `torch.compile` 追踪的层实例。通过 `layer_name` 在编译图中进行运行时查找。

### 模式 E：事件驱动的多流同步

```
默认流：         fused_wqa_wkv → event0.wait() → wq_b+kv_insert → event[1..3].wait()
辅流 0：         event0.record() → compressor_kv_score → event[1].record()
辅流 1：         event0.record() → indexer_weights_proj → event[2].record()
辅流 2：         event0.record() → indexer_compressor_kv_score → event[3].record()
```

事件模式由 `execute_in_parallel`（`multi_stream_utils.py:L60-L128`）实现。两个阶段的多流并行分别使用不同的 event 组合。

详见 [[MultiStreamParallelism]]。

### 模式 F：Compressor 与 Indexer 的"层内层"

Compressor 和 Indexer 内部还有自己的子注意力结构：

```
Indexer.forward:
  ├─ wq_b（Q 投影）
  ├─ fused_indexer_q_rope_quant（Q 量化）
  ├─ Compressor（内部压缩）
  └─ SparseAttnIndexer（稀疏索引）
```

Indexer 内部有自己的 Compressor 实例和 KV 缓存，通过 `maybe_execute_in_parallel` 与外部 Compressor 并行执行。

---

## 5. 边界违规与设计约束

### 已知的边界突破

**`static_forward_context`**（`attention.py:L260-L264`, `compressor.py:L135-L137`）：正常模块间应通过方法参数传递依赖，但 Compressor 和 Wrapper 通过 `compilation_config.static_forward_context` 这个全局字典来查找对方实例。原因：
- `torch.compile` 环境下，模块实例无法通过常规的 Python 引用访问
- `layer_name` 字符串作为编译图中的 "指针"
- 注释标注为 `# HACK`（`attention.py:L261`）

**`no_compile_layers`**：与 `static_forward_context` 类似的运行时全局字典，用于 custom op 查找层实例。

### 模块间隐式耦合

- `Wrapper` 的 `self.padded_heads` 来自 `self.mla_attn.padded_heads`（`attention.py:L255`），`MLAAttention` 的 `padded_heads` 来自 `self.impl_cls.get_padded_num_q_heads(num_heads)`（`attention.py:L653`）。Wrapper 依赖于 Impl 类的具体返回值。
- Compressor 通过 `k_cache_prefix`（`compressor.py:L277`）查找主 KV 缓存的元数据，这是一种字符串级别的弱耦合。

---

## 6. 协同中的数据流拓扑

```
输入 token
  │
  ▼
VocabParallelEmbedding  ───────── (B, H) bf16
  │
  ▼  unsqueeze + repeat
(B, hc_mult, H) bf16
  │
  ▼  [DecoderLayer × N]
  ├─ hc_pre / mhc_fused_post_pre
  │    ├─ attn_norm fused in hc_pre
  │    └─ 产生 layer_input + post_mix + res_mix
  ├─ Attention
  │    ├─ attn_gemm_parallel_execute
  │    │    ├─ fused_wqa_wkv → Q_low_rank + KV_low_rank（主流）
  │    │    ├─ compressor_kv_score（辅流0）
  │    │    └─ indexer 投影（辅流1+2）
  │    ├─ attention_impl
  │    │    ├─ Q norm + RoPE + KV insert（融合 op）
  │    │    ├─ indexer forward（辅流0）
  │    │    ├─ compressor（辅流1）
  │    │    └─ mla_attn（FlashMLA / ROCm 内核）
  │    └─ O 投影（inv_rope + fp8 einsum + wo_b）
  ├─ mhc_fused_post_pre（attn_hc_post + ffn_hc_pre 融合）
  ├─ MoE
  │    ├─ gate（路由）
  │    ├─ [MegaMoE | FusedMoE]（专家计算）
  │    └─ shared_experts
  └─ 返回 (x, residual, post_mix, res_mix) 给下一层
  │
  ▼  last layer hc_post
(B, hc_mult, H) bf16
  │
  ▼  hc_head_op（多流合并）
(B, H) bf16
  │
  ▼  norm
(B, H) bf16
  │
  ▼  lm_head → logits_processor → 采样
```

---

## 7. 模块边界总结

| 边界类型 | 分隔对象 | 连接机制 |
|---------|---------|---------|
| 平台隔离 | NVIDIA vs AMD | 导入时 `__init__.py` 条件选择 |
| 注意力后端 | FlashMLASparseImpl vs ROCmAiterImpl | `_select_v4_sparse_impl()` 策略选择 |
| 计算 vs 结构 | `Attention` vs `Wrapper` | `DeepseekV4MLAModules` dataclass |
| 层间状态 | DecoderLayer → DecoderLayer | 返回元组 `(x, residual, post_mix, res_mix)` |
| PP stage 间 | PP rank → PP rank | `IntermediateTensors` |
| 编译图内外 | Python 代码 vs torch.compile | `custom_op` + `static_forward_context` |
| 多流同步 | 默认流 ↔ 辅流 | `torch.cuda.Event` |

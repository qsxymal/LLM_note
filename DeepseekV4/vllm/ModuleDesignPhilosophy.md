# 模块职责与设计原理

本文从"每个模块解决什么问题、为什么这么设计"的角度，对 DeepSeek V4 在 vLLM 中的实现进行逐模块分析。

---

## 1. `DeepseekV4ForCausalLM` — 因果 LM 顶层外观

### 解决的问题
vLLM 的模型注册、PP 切分、权重加载都需要一个统一的模型入口。同时，MTP 需要从主模型获取中间隐藏状态。

### 为什么这么设计
**vLLM 统一范式**：所有 `*ForCausalLM` 都遵循 `model + lm_head + logits_processor + load_weights` 的契约，使得 vLLM 的调度器、PP 框架、`AutoWeightsLoader` 可以通用方式操作不同模型。

**职责委派**：`forward` 直接委托给 `self.model()`，`load_weights` 委托给 `AutoWeightsLoader`。`ForCausalLM` 本身是一个**零逻辑外观**，核心价值在于：
- 维护 `hf_to_vllm_mapper`（权重键名映射，根据 `expert_dtype` 动态选择 FP4 或 FP8 的 scale 映射规则）（`nvidia/model.py:L1371-L1406`）
- 暴露 `get_mtp_target_hidden_states()` 给 MTP draft model，获取 pre-hc_head 残差流（`nvidia/model.py:L1463-L1467`）
- PP 非末 rank 不创建 `lm_head`（`PPMissingLayer`），避免无效参数

### 边界场景
- `expert_dtype != "fp4"` 时，`hf_to_vllm_mapper` 的 scale 映射规则不同（FP4 用 `weight_scale`，FP8 用 `weight_scale_inv`），在 `__init__` 中覆盖类默认值（`nvidia/model.py:L1421-L1423`）

---

## 2. `DeepseekV4Model` — Transformer 主干 + 多流管理

### 解决的问题
管理 Transformer 的公共资源：嵌入层、解码器层列表、最终 norm、HC 头。关键问题是**多流状态（hc_mult）如何在模型中流动**。

### 为什么这么设计
**多流展开**：`embed_input_ids` 后立即 `unsqueeze(-2).repeat(1, hc_mult, 1)` 将单流 `(B, H)` 展开为多流 `(B, hc_mult, H)`（`nvidia/model.py:L1214`）。这个 shape 贯穿所有解码器层直到 HC 头。

**PP 中间张量**：`make_empty_intermediate_tensors` 创建的 `hidden_states` 形状为 `(B, hc_mult, H)`，比普通模型多一个维度，确保 PP stage 间正确传递多流状态（`nvidia/model.py:L1182-L1200`）。

**最后层收尾**：末 PP rank 对最后一层额外调用 `hc_post`（`nvidia/model.py:L1232-L1233`），将残差流合并回层输出。vLLM 标准 `finalize_layer` 钩子在 `DecoderLayer` 末尾。

**MTP 缓存**：`_mtp_hidden_buffer` 在末 PP rank 上预分配，`copy_` 在 forward 中执行，确保 CUDA Graph 捕获时地址稳定（`nvidia/model.py:L1166-L1177`）。

**HC 头**：`hc_head_op` 将 `(B, hc_mult, H)` 的多流合并为 `(B, H)` 的单流，使用可训练的 `hc_head_fn` 权重（`nvidia/model.py:L1242-L1249`）。

---

## 3. `DeepseekV4DecoderLayer` — HC 融合层

### 解决的问题
DeepSeek V4 的 HC（Hybrid Composition）机制要求注意力子层和 FFN 子层的**前后处理深度融合**，不能简单写为"残差 → norm → 子层 → 残差 → norm → 子层"的顺序。

### 为什么这么设计
**融合路径（`_forward_cuda`）** 是核心设计（`nvidia/model.py:L958-L1025`）：
```
第一层: residual=None → 独立 hc_pre(attn)  → attn() → mhc_fused_post_pre(ffn) → ffn()
后续层: mhc_fused_post_pre(attn) → attn() → mhc_fused_post_pre(ffn) → ffn()
```

`mhc_fused_post_pre` 将**上一子层的 hc_post 与当前子层的 hc_pre 融合为一个算子**（`nvidia/model.py:L1005-L1022`），原因：
1. 减少 kernel launch 次数：两个操作融合为一个 TileLang/Triton 内核
2. 减少全局内存往返：hc_post 的输出无需写回再读取
3. 残差跨层流动：`residual`, `post_mix`, `res_mix` 在层间传递

**原生路径（`_forward_native`）** 是未融合的回退：hc_pre → 子层 → hc_post（`nvidia/model.py:L1027-L1053`）。ROCm/XPU 使用此路径。

**norm 融合**：`_forward_cuda` 中 `attn_norm` 和 `ffn_norm` 的权重被直接传入 `hc_pre` / `mhc_fused_post_pre`，在 MHCPreOp 内部完成归一化（`nvidia/model.py:L967-L968`），避免单独 kernel launch。

### 为什么 attn_norm 和 ffn_norm 作为独立模块声明（`nvidia/model.py:L871-L872`）
- 为了权重加载时参数名称匹配（`state_dict` 中需要有 `attn_norm.weight`）
- 在 CUDA 路径中，参数被取出传给 MHC 算子，module 本身不参与前向
- 在原生路径中，module 正常参与前向

---

## 4. `DeepseekV4Attention` — 注意力组件工厂

### 解决的问题
MLA 注意力涉及众多子组件（投影、norm、RoPE、索引器、SWA 缓存、压缩器），需要一个装配点来声明"这个层需要哪些参数和子模块"。

### 为什么这么设计
**零计算 forward**：`forward` 只有一行 `self.mla_attn(...)`（`nvidia/model.py:L836-L842`）。计算完全委托给 `DeepseekV4MultiHeadLatentAttentionWrapper`。

**为何 Indexer 在此声明**（`nvidia/model.py:L780-L800`）：
- Indexer 是模型结构的一部分，有可训练参数和独立的 KV 缓存
- 它需要参与 `state_dict`、权重加载、缓存管理器注册
- compress_ratio == 4 时才存在，条件分支在结构层处理

**为何 Compressor 不在此声明**：
- Compressor 是计算流程的内部环节，属于执行细节
- 与多流并行深度融合，归属 Wrapper 更自然
- Wrapper 可替换性考虑

详见 [[ArchitectureOverview]] 第 II 节的 Indexer/Compressor 归属分析。

---

## 5. `DeepseekV4MultiHeadLatentAttentionWrapper` — MLA 计算编排器

### 解决的问题
MLA 注意力包含多个并行 GEMM + 融合算子 + 索引器 + 压缩器，需要一个编排点来协调它们的执行顺序和并行关系。

### 为什么这么设计
**作为自定义算子注册**（`attention.py:L114`）：
```python
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
```
注册为 `PluggableLayer`，使整个 MLA 流程可以被不同后端整体替换。

**`torch.ops.vllm.deepseek_v4_attention`**（`attention.py:L546-L554`）：将 `forward` → `attention_impl` 封装为自定义 op，使得 `torch.compile` 可以将内部融合为一个编译图。

**两层并行**：
- 第一层在 `attn_gemm_parallel_execute`：`fused_wqa_wkv`（主流）与 compressor `kv_score`、indexer 投影并行（3 个辅流）
- 第二层在 `attention_impl`：`wq_b + kv_insert` 与 indexer `forward`、compressor 的压缩 + 量化 + 缓存写入并行（2 个辅流）

**O 投影的 GPU 自适应**（`attention.py:L197-L200`）：
- SM90（H100）：FP32 block scales `(1, 128, 128)` 分块
- SM100（Blackwell）：INT32 packed scales `(1, 1, 128)` 逐元素缩放，支持 TMA

---

## 6. `DeepseekV4MLAAttention` — 核心注意力抽象

### 解决的问题
提供一个与 vLLM V1 注意力框架兼容的接口层，使调度器、缓存管理器能统一操作 MLA 注意力。

### 为什么这么设计
**策略模式**：`forward` 委托给 `self.impl_cls.forward_mqa(...)`（`attention.py:L720-L727`），`impl_cls` 由 `_select_v4_sparse_impl()` 根据平台选择：
- NVIDIA：`DeepseekV4FlashMLASparseImpl`
- ROCm：`DeepseekV4ROCMAiterMLASparseImpl`

**强制 FP8 KV 缓存**（`attention.py:L669-L688`）：`kv_cache_dtype` 必须是 FP8 系列，自动转为 `fp8_ds_mla` 布局。FlashMLA 要求 576B 对齐的缓存布局。

**KV 缓存规格**：`get_kv_cache_spec` 返回 `MLAAttentionSpec`，包含 `compress_ratio` 和 `alignment=576`（`attention.py:L704-L718`）。这使得 vLLM 的缓存管理器可以按正确的 shape 分配块。

---

## 7. `DeepseekV4FlashMLASparseImpl` — 内核调度器

### 解决的问题
FlashMLA 内核（CUDA）需要特定的输入数据布局和 tile 调度策略，prefill 和 decode 的路径不同。

### 为什么这么设计
**物理含义**：这是最靠近硬件的层，负责将"Q 在哪里、KV 在哪里、输出写哪里"翻译成 `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` 的调用。

**Tile scheduler metadata**（`nvidia/flashmla.py:L262-L284`）：按层类型（SWA-only / C4A / C128A）缓存 tile 规划结果，避免同一类型多次调用内核内规划器。CUDA Graph 兼容的关键——规划器只在首次运行，之后复用。

**Head padding**（`nvidia/flashmla.py:L119-L127`）：FP8 解码内核要求 Q 头数为 64 或 128，因此对 `n_local_heads < 64` 的配置做 padding。

**C4A topk 索引在线计算**：每层的 `topk_indices_buffer` 由 Indexer 填充，全局索引在 `_forward_decode` 中即时计算。C128A 的 topk 在元数据构建时预计算。

详见 [[V1AttentionBackend]] 和 [[DeepseekV4FlashMLASparseImpl]]。

---

## 8. `DeepseekV4MoE` — 混合专家 FFN

### 解决的问题
256 个路由专家 + 1 个共享专家的 MoE 需要灵活的精度策略（FP4/FP8/BF16）和并行策略（EP/TP）支持。

### 为什么这么设计
**双路径互斥**（`nvidia/model.py:L465-L473`）：
- `use_mega_moe=True`：DeepGEMM FP4 权重 + FP8 激活 + EP。仅 SM100 Blackwell。
- `use_mega_moe=False`：vLLM 标准 FusedMoE + TP。通用 GPU。

两个路径互斥，通过 `moe_backend` 内核配置选择。MegaMoE 强制要求 EP（`nvidia/model.py:L468-L473`）。

**路由机制**（`nvidia/model.py:L495-L524`）：
- 默认 sqrtsoftplus 评分函数（DeepSeek V4 特有，非标准 softmax）
- Hash MoE（前 `num_hash_layers` 层）：通过 token ID 哈希表固定路由，`tid2eid` 参数（`nvidia/model.py:L511-L519`）
- Noaux TC：`e_score_correction_bias` 辅助路由偏置（`nvidia/model.py:L520-L524`）
- MegaMoE 路径强制使用 `hash_indices_dtype=torch.int64`（`nvidia/model.py:L506`），FusedMoE 用 int32

**共享专家**（`nvidia/model.py:L526-L539`）：所有 token 都经过。`reduce_results=self.use_mega_moe` 控制是否在共享专家的 `down_proj` 后做 all-reduce。MegaMoE 路径下 disabled（DeepGEMM 内部处理通信）。

**权重最终化**（`nvidia/model.py:L280-L316`）：MegaMoE 权重在加载后需要 `finalize_weights()`，将原始 checkpoint 格式转为 DeepGEMM 内核要求的交错布局。转换后丢弃原始 `w13_weight` 等参数释放显存。

详见 [[DeepseekV4MoE]]。

---

## 9. `DeepseekCompressor` — KV 压缩器

### 解决的问题
将注意力 KV 投影后的输出压缩（比率 4 或 128）并写入压缩 KV 缓存，同时计算 score 状态用于稀疏注意力。

### 为什么这么设计
**两阶段写入**（`compressor.py:L270-L380`）：
1. `save_partial_states`：将 kv + score 以 float32 精度写入 state cache（带 APE 偏置）
2. `compress_norm_rope_store`：从 state cache 读取 → 压缩 → RMSNorm → RoPE → FP8 量化 → 写入主 KV 缓存

分两阶段的原因是：压缩需要 state cache 中的历史上下文（滑动窗口），而 state cache 以 float32 保存以保证精度。

**ROCm/NVIDIA 共享代码**：`DeepseekCompressor.forward` 中只有 `compress_norm_rope_store_fn` 根据平台选择不同实现（CUDA 用 cutedsl 或 Triton，ROCm 全用 Triton），其余逻辑完全共享（`compressor.py:L338-L355`）。

**FP4 indexer cache**：`use_fp4_cache` 标志控制 indexer 压缩缓存是否使用 MXFP4 格式（`compressor.py:L256-L268`）。

---

## 10. `DeepseekV4Indexer` — 稀疏注意力索引器

### 解决的问题
决定每个 token 在稀疏注意力中关注压缩 KV 池的哪些位置（top-k 选择）。

### 为什么这么设计
**"层内层"架构**：Indexer 本身是一个**完整的子注意力网络**（`attention.py:L771-L910`）：
- 自己的 Q 投影（`wq_b`，`ReplicatedLinear`）
- 自己的权重投影（`weights_proj`）
- 自己的 Compressor 和 state cache
- 自己的 KV 缓存（`DeepseekV4IndexerCache`）
- 自己的稀疏索引算子（`SparseAttnIndexer`）

**独立 KV 缓存**（`attention.py:L730-L768`）：`DeepseekV4IndexerCache` 继承 `AttentionLayerBase`，向 vLLM 缓存管理器声明自己的 `MLAAttentionSpec`。这意味着 Indexer 的 KV 缓存由框架分配和管理。

**为何需要 indexer**：C4A 压缩率（4）下，压缩 KV 池仍然很大（是 C128A 的 32 倍），需要稀疏注意力从中选择最相关的 token。Indexer 的 Q 投影专门学习如何选择这些 token。

**并行友好**：`forward` 中的 `wq_b_and_q_quant` 与 `compressor` 可以并行执行（`attention.py:L903-L909`），通过 `maybe_execute_in_parallel` 实现。

---

## 11. `FlashMLASparseBackend / DeepseekV4FlashMLASparseBackend` — V1 注意力框架集成

### 解决的问题
vLLM V1 的注意力框架要求每个注意力后端提供：元数据构建器、KV 缓存形状描述、后端标识。需要适配 FlashMLA 的特定约束。

### 为什么这么设计
**类层次结构**：
```
FlashMLASparseBackend (vllm.v1.attention.backends.mla.flashmla_sparse)
  └── DeepseekV4FlashMLASparseBackend (nvidia/flashmla.py)
```

子类只需覆盖少量方法（`get_supported_head_sizes`、`get_kv_cache_shape` 等），框架的元数据构建、CUDA Graph 支持等从父类继承。

**关键覆写**：`get_kv_cache_shape` 返回 `(num_blocks, block_size, 584)`（`nvidia/flashmla.py`），584 = 448 NoPE + 128 RoPE + 8 缩放因子。这个数字是 FlashMLA 内核要求的固定布局。

---

## 核心设计原则总结

| 原则 | 体现 |
|------|------|
| **计算与结构分离** | Attention（结构）vs Wrapper（计算） |
| **硬件隔离分层** | common/ops 跨平台共享，nvidia/ 和 amd/ 各自实现 |
| **融合优先** | HC pre/post 融合、Q norm+RoPE+KV insert 融合 |
| **延迟隐藏** | 多流并行重叠 GEMM 与计算 |
| **精度异构** | 不同模块用不同精度（FP8/BF16/FP4/float32） |
| **框架兼容** | 通过 AttentionLayerBase、PluggableLayer、CustomOp 与 vLLM V1 集成 |
| **CUDA Graph 优先** | 预分配缓冲区、tile scheduler 缓存、静态 forward context |

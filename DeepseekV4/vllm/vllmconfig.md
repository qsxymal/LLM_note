# VllmConfig —— 配置聚合与 DeepSeek V4 特有字段

## 概述

`VllmConfig` 是 vLLM 的统一配置对象，聚合了模型初始化所需的全部配置项。在 DeepSeek V4 的场景中，以下字段具有特殊作用。

## 重要字段

| 字段 | 类型 | DeepSeek V4 中的关键影响 |
|------|------|------------------------|
| `model_config` | `ModelConfig` | 持有 `hf_config`（Hugging Face 原始配置），是 V4 架构检测的源头。 |
| `quant_config` | `QuantizationConfig` | 在 V4 中具体化为 `DeepseekV4FP8Config`，控制 FP8/FP4 量化分发。 |
| `parallel_config` | `ParallelConfig` | TP、PP 配置，与 EP/多流并行（`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`）联动。 |
| `scheduler_config` | `SchedulerConfig` | `max_num_batched_tokens` 控制 MegaMoE `symm_buffer` 分配大小。 |
| `cache_config` | `CacheConfig` | KV 缓存 dtype 强制为 `fp8_ds_mla`。 |
| `speculative_config` | `SpeculativeConfig` | MTP 的 `draft_model_config.hf_config` 来源。 |
| `compilation_config` | `CompilationConfig` | `static_forward_context` 用于 MegaMoE 的 `no_compile_layers` 注册。 |

## DeepSeek V4 特有的 `hf_config` 字段

DeepSeek V4 checkpoint 的 `config.json` 包含以下对推理至关重要的字段：

### 架构基础

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `model_type` | `"deepseek_v4"` | 架构自动检测，触发 `DeepseekV4FP8Config` 和模型注册 |
| `hidden_size` | 4096 | 隐藏层维度 |
| `num_attention_heads` | 64 | 注意力头数 |
| `num_hidden_layers` | 60 | 解码器层数 |

### HC（Hyper-Composed）多头流

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `hc_mult` | 2 | 多头流数，控制 `DeepseekV4DecoderLayer` 中流并行数 |
| `hc_sinkhorn_iters` | 1 | Sinkhorn 归一化迭代次数，控制混合系数生成精度 |
| `hc_eps` | 1e-6 | HC 操作的 epsilon，防止除零 |
| `hc_post_alpha` | 2.0 | **硬编码**（非 config 字段），HC 后处理的缩放因子 |

### MLA（Multi-head Latent Attention）

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `q_lora_rank` | 1536 | Q 低秩投影维度 |
| `kv_lora_rank` | 512 | KV 低秩投影维度（= `qk_nope_head_dim + qk_rope_head_dim`） |
| `qk_nope_head_dim` | 128 | QK 无位置编码的 head 维度 |
| `qk_rope_head_dim` | 64 | QK RoPE 部分的 head 维度 |
| `v_head_dim` | 128 | Value head 维度 |
| `attn_sink_token_num` | ？ | 注意力 sink token 数量 |

### MoE（Mixture of Experts）

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `n_routed_experts` | 256 | 路由专家总数 |
| `num_experts_per_tok` | 8 | 每 token 激活的专家数（top-k） |
| `moe_intermediate_size` | 2048 | MoE FFN 的中间维度 |
| `shared_expert_intermediate_size` | ？ | 共享专家的中间维度 |
| `expert_dtype` | `"fp4"` 或 `"fp8"` | 专家权重精度，控制量化方法路由 |
| `moe_quant_algo` | `"NVFP4"` 或空 | MoE 量化算法，空表示默认 MXFP4 |
| `index_topk` | ？ | Indexer 使用的 topk 数量 |

### 压缩与稀疏注意力

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `compress_ratio` | 4 或 128 | KV 缓存压缩比，控制稀疏注意力模式（C4A vs C128） |
| `sliding_window_size` | ？ | 滑动窗口大小 |
| `attn_cache_compress_ratio` | ？ | 注意力缓存的压缩比（与 compress_ratio 联动） |

### 推测解码

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `num_nextn_predict_layers` | ？ | MTP 额外预测层数，即 `num_mtp_layers` |

## DeepSeek V4 配置交互矩阵

| hc_mult | compress_ratio | expert_dtype | use_mega_moe | 影响 |
|---------|---------------|-------------|-------------|------|
| 2 | 4 | fp4 | True | 完整 V4 架构：C4A 稀疏注意力 + FP4 MegaMoE（SM100） |
| 2 | 4 | fp8 | False | C4A + FP8 FusedMoE（通用 GPU） |
| 2 | 128 | fp4 | True | C128 稀疏注意力 + FP4 MegaMoE |
| 2 | 1 | fp4 | False | SWA-only（无压缩）+ FP4 FusedMoE |

## 关键设计：lazy `expert_dtype` 解析

`DeepseekV4FP8Config` 在 `VllmConfig` 构造阶段创建，此时 `set_current_vllm_config` 尚未激活。`expert_dtype` 在首次实际访问时才从 `hf_config` 读取（`quant_config.py:L56-L76`），确保 Flash-Base checkpoint 能被正确路由。

详见 [[DeepseekV4FP8Config#2-关键设计：Lazy-expert_dtype-解析]]。

## Cross-References

- [[DeepseekV4FP8Config]]：`quant_config` 的具体实现和执行分发
- [[QuantAndParallelStrategy]]：`quant_config` 与 `parallel_config` 联动
- [[DeepseekV4MoE]]：`moe_intermediate_size` 和 `n_routed_experts` 如何影响 MoE 层初始化
- [[DeepseekV4DecoderLayer]]：`hc_mult` 控制流并行度
- [[DeepseekV4Model]]：`num_hidden_layers` 和 PP 切分
- [[MTP]]：`num_nextn_predict_layers` 控制 MTP 规模

# 端到端数据流追踪

一个 batch 的 token 从输入到输出的完整路径。

## 约定

```
variable_name (shape) dtype [文件:行号]
```

## 1. 嵌入

```
input_ids (num_tokens,) int64
    │ [DeepseekV4ForCausalLM.forward → DeepseekV4Model.forward]
    ▼
hidden_states (num_tokens, hidden_size) bf16 [model.py:L1177]
    │ 来自 embed_tokens(input_ids) — VocabParallelEmbedding
    │ 如果是 PP 首 rank，否则来自 intermediate_tensors
    │ 此后 hidden_states 维度在每层中从 hidden_size → hc_dim
    ▼
hidden_states (num_tokens, hc_mult, hidden_size) bf16 [model.py:L1182-L1183]
    │ view(hc_mult, -1).transpose(0, 1) — 展开为多流
```

## 2. 解码器层（第 0 层 — 特殊处理）

```
hidden_states (B, hc_mult, H) bf16
    │ [DeepseekV4DecoderLayer._forward_cuda] [model.py:L969-L979]
    │ 第一层: residual=None, 退化为独立 hc_pre
    ├─ hc_pre(attn_*) → layer_input(B, hc_mult, H) bf16 + post_mix/res_mix
    │                 → attn_norm_weight 传入 hc_pre 内部（融合 norm）
    ▼
x (B, hc_mult, H) bf16 — attention 输入
    │ [DeepseekV4Attention.forward → DeepseekV4MultiHeadLatentAttentionWrapper]
    ▼
    ┌─ fused_wqa_wkv: (B, q_lora_rank+kv_lora_rank) bf16 [model.py:L54]
    │  └─ Q 低秩投影 + KV 低秩投影（并行多流 GEMM）
    ├─ compressor: kv_score (B, kv_lora_rank) float32 (辅流 0)
    ├─ indexer: indexer_weights (辅流 1+2) 
    ├─ Q/KV norm + RoPE + KV cache insert (融合自定义 op)
    ├─ flash_mla_with_kvcache / flash_mla_sparse_fwd
    │  └─ 输出: (B, padded_heads, head_dim) bf16
    └─ O 投影: wo_a (FP8 einsum) + wo_b (bf16 Linear + all-reduce)
    │         → x (B, hc_mult, H) bf16
    ▼
x (B, hc_mult, H) bf16 — attention 输出
    │ [mhc_fused_post_pre(ffn)] [model.py:L1005]
    │ 融合: attn hc_post + ffn hc_pre（使用上一阶段产生的 post_mix/res_mix）
    │ 输出: residual/post_mix/res_mix + x (layer_input for FFN)
    ▼
x (B, hc_mult, H) bf16 — FFN 输入
    │ [DeepseekV4MoE.forward]
    │ 对于非第一层: 融合路径已有 norm 在 mhc_fused_post_pre 中完成
    ▼
    ┌─ gate: (B, n_routed_experts) float32 [model.py:L470/amd:L133]
    │  ├─ fused_topk_bias → topk_ids (B, top_k) int64
    │  └─ topk_weights (B, top_k) bf16
    ├─ [MegaMoE]: prepare_megamoe_inputs (FP8 量化 hidden_states)
    │  └─ deep_gemm.fp8_fp4_mega_moe → y (B, hc_mult, H) bf16
    └─ [FusedMoE]: FusedMoE.forward → y (B, hc_mult, H) bf16
    ▼
x (B, hc_mult, H) bf16 — FFN 输出
    │ [model.py:L1025]
    │ 返回 (x, residual, post_mix, res_mix) 给下一层
```

## 3. 解码器层（第 1 层及后续 — 融合路径）

```
上层的 (x, residual, post_mix, res_mix)
    │ [model.py:L981-L998]
    │ mhc_fused_post_pre: 融合上层 hc_post + 当前层 hc_pre(attn)
    │ 输出: residual/post_mix/res_mix + x (attention layer_input)
    ▼
x → attn() → mhc_fused_post_pre(ffn) → ffn() → (x, residual, post_mix, res_mix) → 下一层
```

**跨层状态**：`residual(B, hc_mult, H) bf16 + post_mix(B, mix_hc) float32 + res_mix(B, mix_hc) float32` 在每层间流动，支持 `mhc_fused_post_pre` 的融合。

## 4. 最终层输出

```
hidden_states (B, hc_mult, H) bf16 — 经所有层处理后的多流状态
    │ [DeepseekV4Model.forward 结尾] [model.py:L1239-L1250]
    │
    ├─ 保存到 _mtp_hidden_buffer（供 MTP 使用）
    │  _mtp_hidden_buffer[:B] = hidden_states.flatten(1)  # (B, hc_dim) bf16
    │
    ├─ HC Head 合并多流
    │  hidden_states = hc_head_op(hidden_states)  # (B, H) bf16
    │
    └─ final norm
       hidden_states = norm(hidden_states)  # (B, H) bf16
```

**PP 中间 stage**：通过 `IntermediateTensors` 传递 `hidden_states`（`(B, hc_mult, H)`）到下一 PP stage。

## 5. LM Head + Logits

```
hidden_states (B, H) bf16 [model.py:L1444-L1448]
    │ PP 末 rank 才有 lm_head
    │
    ├─ lm_head (ParallelLMHead): (B, H) → (B, vocab_size/tp) bf16
    │   └─ FP8 quant 矩阵乘（quant_method.apply）
    │
    ├─ LogitsProcessor: gather + 处理
    │   └─ TP gather: 从各 rank 收集完整 logits
    │   └─ (B, vocab_size) float32
    │
    └─ 采样 → 输出 token ID
```

## 6. 关键张量形状速查

### 解码器层内部

| 变量 | 形状 | dtype | 产生位置 |
|------|------|-------|---------|
| hidden_states（多流） | `(B, hc_mult, H)` | bf16 | 嵌入后 reshape |
| attn layer_input | `(B, hc_mult, H)` | bf16 | hc_pre 输出 |
| Q low-rank | `(B, q_lora_rank)` | bf16 | fused_wqa_wkv |
| KV low-rank | `(B, kv_lora_rank)` | bf16 | fused_wqa_wkv |
| Q after norm+RoPE | `(B, padded_heads, head_dim)` | bf16 | fused_norm_rope插入 op |
| KV cache（每 token） | 584 字节 | fp8 + bf16 | FP8 KV 缓存 |
| Attention 输出 | `(B, padded_heads, head_dim)` | bf16 | FlashMLA |
| wo_a 输入 | `(B, padded_heads, 128)` | fp8 | FP8 einsum |
| wo_b 输出 | `(B, hc_mult, H)` | bf16 | RowParallel + all-reduce |
| gate logits | `(B, n_routed)` | float32 | GateLinear |
| topk_ids | `(B, top_k)` | int64 | fused_topk_bias |
| MoE 输出 | `(B, hc_mult, H)` | bf16 | MegaMoE / FusedMoE |
| post_mix / res_mix | `(B, mix_hc)` | float32 | hc_pre 输出 |
| residual | `(B, hc_mult, H)` | bf16 | 跨层残差 |

### 关键常量

| 常量 | 典型值 | 说明 |
|------|-------|------|
| `H` = hidden_size | 4096 | 隐藏层维度 |
| `hc_mult` | 2 | 多头流数 |
| `hc_dim` = hc_mult × H | 8192 | 展平后的多流维度 |
| `mix_hc` = (2+hc_mult)×hc_mult | 8 | HC 混合矩阵行数 |
| `q_lora_rank` | 1536 | Q 低秩投影维度 |
| `kv_lora_rank` | 512 | KV 低秩投影维度 |
| `head_dim` | 512 | 注意力 head 维度（= qk_nope + qk_rope + v） |
| `padded_heads` | 64 或 128 | FlashMLA 对齐后的 head 数 |
| `n_routed_experts` | 256 | 路由专家数 |
| `top_k` | 8 | 每 token 激活专家数 |

## Cross-References

- 每步涉及的模块参见：[[ArchitectureOverview]]
- 形状变换中的量化点参见：[[QuantAndParallelStrategy]]
- 多流展开和 HC 混合参见：[[DeepseekV4DecoderLayer]]
- MTP 分支参见：[[MTP]]

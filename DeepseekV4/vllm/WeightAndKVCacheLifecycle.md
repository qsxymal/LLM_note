# 权重与 KV Cache 全生命周期

---

## 第一部分：权重的完整生命周期

### 1.1 从 Checkpoint 到 GPU 的完整路径

```
HF Checkpoint（.safetensors）
  │  格式：fp8_e8m0fnu / bf16
  │
  ▼  AutoWeightsLoader.load_weights
Weights Mapper（hf_to_vllm_mapper）
  │  └─ 键名重映射 + scale 格式转换
  │
  ▼  DeepseekV4ForCausalLM.load_weights
  │
  ├─ stacked_params_mapping        ← 合并权重（wq_a + wkv → fused_wqa_wkv）
  ├─ expert_mapping                ← 专家分布式映射（EP/TP 分片）
  ├─ attn_sink 特殊处理            ← head 切分
  └─ 普通参数 default_weight_loader ← Linear 层权重直接加载
  │
  ▼
DeepseekV4Model.finalize_mega_moe_weights
  └─ 仅 MegaMoE 路径：权重布局变换 + 丢弃原始参数
```

### 1.2 权重映射器（hf_to_vllm_mapper）

`_make_deepseek_v4_weights_mapper(expert_dtype)`（`nvidia/model.py:L1371-L1406`）根据 `expert_dtype` 选择不同映射规则：

**FP4 专家（`expert_dtype="fp4"`）**：
```python
scale_regex = {
    re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
    re.compile(r"\.scale$"): ".weight_scale_inv",          # 共享专家 + Linear
}
```

- MXFP4 专家的 scale 键为 `weight_scale`（不含 `_inv` 后缀）
- 共享专家和普通 Linear 层的 scale 键为 `weight_scale_inv`

**FP8 专家（`expert_dtype="fp8"`）**：
```python
scale_regex = {
    re.compile(r"\.scale$"): ".weight_scale_inv",
}
```

- 所有 scale 统一映射到 `weight_scale_inv`
- 因为 `Fp8MoEMethod` 注册的 scale 参数名为 `weight_scale_inv`

**前缀/后缀映射**：
```python
orig_to_new_prefix = {
    "layers.": "model.layers.",          # Checkpoint → vLLM 命名
    "embed.": "model.embed.",
    "norm.": "model.norm.",
    "hc_head": "model.hc_head",
    "mtp.": "model.mtp.",
}
orig_to_new_suffix = {
    "head.weight": "lm_head.weight",
    "embed.weight": "embed_tokens.weight",
    ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
}
orig_to_new_substr = {
    ".attn.compressor.": ".attn.mla_attn.compressor.",
    ".shared_experts.w2": ".shared_experts.down_proj",
}
```

**为什么要做这么多映射**：
- Checkpoint 来自 DeepSeek 原始训练框架，其命名约定与 vLLM 的 `nn.Module` 命名不同
- 不同 `expert_dtype` 下 scale 参数的注册名称不同（`weight_scale` vs `weight_scale_inv`）
- 一些层在模型中嵌套层次较深（如 compressor 在 `mla_attn` 内部）

### 1.3 合并权重（stacked_params_mapping）

```python
stacked_params_mapping = [
    ("gate_up_proj", "w1", 0),      # w1 → gate_up_proj 的前一半
    ("gate_up_proj", "w3", 1),      # w3 → gate_up_proj 的后一半
    ("attn.fused_wqa_wkv", "attn.wq_a", 0),   # wq_a → fused_wqa_wkv 前半
    ("attn.fused_wqa_wkv", "attn.wkv", 1),    # wkv → fused_wqa_wkv 后半
    ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
    ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
]
```

**设计意图**：
- checkpoint 中 `w1` 和 `w3` 是分开的，加载时合并为 `gate_up_proj` 的 MergedColumnParallelLinear
- `wq_a` 和 `wkv` 合并为 `fused_wqa_wkv`
- `wkv` 和 `wgate` 合并为 `fused_wkv_wgate`
- 合并的好处：前向时一次 GEMM 计算两个投影，减少 kernel launch 次数

### 1.4 专家权重的分布式映射

**MegaMoE 路径**（`make_deepseek_v4_expert_params_mapping`，`nvidia/model.py:L119-L135`）：
```python
return [
    ("experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
     f"experts.{expert_id}.{weight_name}.", expert_id, shard_id)
    for expert_id in range(num_experts)
    for shard_id, weight_name in [("w1","w1"), ("w2","w2"), ("w3","w3")]
]
```

**FusedMoE 路径**：使用 `FusedMoE.make_expert_params_mapping`（标准 vLLM MoE 映射）。

**MegaMoE 的 EP 分片**（`nvidia/model.py:L552-L559`）：
```python
self.n_local_experts = config.n_routed_experts // self.ep_size
self.experts_start_idx = self.ep_rank * self.n_local_experts
```
每个 EP rank 只加载 `n_local_experts` 个专家权重，`weight_loader` 中通过 `_map_global_expert_id`（`nvidia/model.py:L221-L224`）将全局 expert ID 映射到本地索引，非本地专家的权重直接跳过。

### 1.5 attn_sink 的特殊加载

`attn_sink` 不是线性层权重，而是一个 `(padded_heads,)` 的 float32 sink 偏置（`nvidia/model.py:L710-L714`）：
```python
self.attn_sink = nn.Parameter(
    torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
    requires_grad=False,
)
```

加载时需要按 TP rank 切分 head（`nvidia/model.py:L1331-L1337`）：
```python
narrow_weight = loaded_weight[head_rank_start:head_rank_end]
params_dict[name][:narrow_weight.shape[0]].copy_(narrow_weight)
```
因为 checkpoint 中 `attn_sink` 包含所有 head 的值，每个 TP rank 只取自己的 head 子集。

### 1.6 MegaMoE 权重最终化（finalize_weights）

`DeepseekV4MegaMoEExperts.finalize_weights()`（`nvidia/model.py:L280-L316`）在权重加载完成后执行：

```python
def finalize_weights(self):
    # 1. 将 E8M0 uint8 scale 转为 float32
    w13_scale = deep_gemm.transform_sf_into_required_layout(
        self._ue8m0_uint8_to_float(self.w13_weight_scale.data), ...)
    w2_scale = deep_gemm.transform_sf_into_required_layout(...)
    
    # 2. 将权重转成 DeepGEMM 的交叉布局
    self._transformed_l1_weights, self._transformed_l2_weights = \
        deep_gemm.transform_weights_for_mega_moe(
            (w13_weight_int8, w13_scale), (w2_weight_int8, w2_scale))
    
    # 3. 丢弃原始参数释放显存
    self.w13_weight = None
    self.w13_weight_scale = None
    self.w2_weight = None
    self.w2_weight_scale = None
```

**为什么需要布局变换**：DeepGEMM 的 MegaMoE 内核要求权重以特定交错格式存储，以优化 FP4 矩阵乘法的内存访问模式。原始 uint8 打包的 FP4 权重需要被重新排列。

**为什么丢弃原始参数**：变换后的权重已经包含了所有需要的信息，原始 uint8 参数占用额外显存。只在推理时有效——如果未来需要 fine-tune 则需要保留。

### 1.7 E8M0 Scale 的存储格式

Checkpoint 中 scale 以 `float8_e8m0fnu` dtype 存储，但 MoE 参数是 `uint8`：

```python
if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
    loaded_weight = loaded_weight.view(torch.uint8)
```

**为什么 view 而非转换**：`uint8` 和 `float8_e8m0fnu` 的底层字节布局相同（都是 8 位指数，bits 相同），只是语义解释不同。`copy_()` 会做数值转换（如 `2⁻⁷ → 0`），破坏原始指数字节。

**E8M0 转 float32**（`nvidia/model.py:L261-L262`）：
```python
def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
    return (sf.to(torch.int32) << 23).view(torch.float32)
```
将 uint8 指数左移 23 位放到 float32 的指数位。

---

## 第二部分：KV Cache 的完整生命周期

### 2.1 三种 KV 缓存

| 缓存类型 | 所属模块 | 形状 | 分配时机 |
|---------|---------|------|---------|
| 主 MLA KV 缓存 | `DeepseekV4MLAAttention` | `(num_blocks, 256, 584)` | vLLM 缓存管理器 |
| SWA KV 缓存 | `DeepseekV4SWACache` | `(num_blocks, swa_block_size, head_dim)` | `DeepseekV4SWACache` 构造函数 |
| Compressor State Cache | `CompressorStateCache` | `(num_blocks, block_size, state_dim)` | 缓存管理器 |
| Indexer KV 缓存 | `DeepseekV4IndexerCache` | `(num_blocks, block_size, k_cache_head_dim)` | 缓存管理器 |

### 2.2 缓存分配的契约机制

vLLM V1 框架通过 `get_kv_cache_spec()` 方法了解每个注意力组件需要什么形状的缓存：

**主 MLA KV Cache**（`attention.py:L704-L718`）：
```python
def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec | None:
    if self.compress_ratio <= 1:  # SWA-only
        return None
    return MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=self.head_dim,
        dtype=torch.uint8,
        compress_ratio=self.compress_ratio,
        cache_dtype_str="fp8_ds_mla",
        alignment=576,           # FlashMLA 要求的对齐
        model_version="deepseek_v4",
    )
```

**Compressor State Cache**（`compressor.py:L157-L165`）：
```python
def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
    return SlidingWindowMLASpec(
        block_size=self.block_size,    # C4 = 4, C128 = 8
        num_kv_heads=1,
        head_size=self.state_dim,      # 2*coff*head_dim
        dtype=torch.float32,           # state cache 用 float32
        sliding_window=self.sliding_window,
        alignment=576,
    )
```

**Indexer KV Cache**（`attention.py:L751-L763`）：
```python
def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
    return MLAAttentionSpec(
        block_size=self.cache_config.block_size,
        num_kv_heads=1,
        head_size=k_cache_head_dim,    # = head_dim + head_dim // 128 * 4
        dtype=torch.uint8,
        compress_ratio=self.compress_ratio,
        alignment=576,
    )
```

### 2.3 KV Cache 布局（584 字节/token）

```
一个 block = 256 tokens
每个 token 占 584 字节:
┌─────────────────┬─────────────────┬──────────────────────────┐
│ NoPE (448B)     │ RoPE (128B)     │ 缩放因子 (8B)           │
│ FP8 E4M3        │ BF16            │ 7 × UE8M0 + 1 pad       │
│ dim=448         │ dim=64×2=128    │ NoPE 分 7 个 64 块量化  │
└─────────────────┴─────────────────┴──────────────────────────┘
```

- FP8 量化仅应用于 NoPE 部分（448 维），RoPE 部分保持 BF16
- 448 FP8 字节 + 64 × 2 BF16 字节 + 8 元数据字节 = 584 字节
- 缩放因子为 7 个 UE8M0 指数（每 64 元素一个）+ 1 字节 padding

### 2.4 KV Cache 写入路径

```
KV cache write path（单层，单次 forward）:
  ┌─ fused_wqa_wkv GEMM
  │    └─ kv = (B, head_dim=512) bf16
  │
  ├─ (仅在 compress_ratio > 1 时)
  │  compressor.forward:
  │    1. kv_score ← fused_wkv_wgate(hidden_states)    # GEMM
  │    2. split(kv, score)                              # [coff*head_dim] × 2
  │    3. save_partial_states(kv, score, ape)
  │       → state_cache (float32)
  │    4. compress_norm_rope_store(state_cache)
  │       → read state_cache → compress → RMSNorm → RoPE → FP8 quant
  │       → write to main KV cache (FP8, 584 bytes/token)
  │
  ├─ kv_norm(kv) → fused_q_kv_rmsnorm
  │
  └─ _fused_qnorm_rope_kv_insert:  # 融合自定义 op
       └─ 对 kv（剩余部分）：
          RoPE（GPT-J style，仅最后 rope_head_dim=64 维）
          → FP8 quant（UE8M0）
          → insert into SWA cache (FP8)
```

**关键设计**：
- Compressor 先写 state cache（float32），再通过 `compress_norm_rope_store` 写主 KV cache（FP8）。两步分离的原因是 state cache 用于后续的滑动窗口压缩计算，需要高精度
- SWA cache 写入与 Q norm + RoPE 融合在同一个自定义 op 中（`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`），减少 kernel launch
- RoPE 只应用于 K 的最后 `rope_head_dim` 维，NoPE 部分保持不变

### 2.5 KV Cache 读取路径（注意力 forward）

```
Decode 路径:
  ┌─ flash_mla_with_kvcache(
  │      q,                    # (B, padded_heads, head_dim) bf16
  │      kv_cache,             # (num_blocks, 256, 584) fp8
  │      block_table,
  │      topk_indices,         # 稀疏选择位置
  │      topk_lens,
  │      swa_kv_cache,         # 滑动窗口缓存
  │      tile_scheduler_metadata,
  │   )
  └─ 输出: (B, padded_heads, head_dim) bf16

Prefill 路径（分块）:
  对每 4 个序列一个块:
    1. dequantize_and_gather_k_cache(
         cache, block_table, slot_mapping, seq_lens, ...
       ) → (chunk_size, M, head_dim) bf16
       └─ 从主 KV cache 中读取并反量化 FP8 → BF16
    2. 同上读取 SWA cache
    3. combine_topk_swa_indices(topk_indices, swa_indices)
       → 合并稀疏索引与 SWA 索引
    4. flash_mla_sparse_fwd(q, k_collected, combined_indices, ...)
```

**为什么 prefill 需要分块**：Prefill 的序列可能很长，一次性收集所有 KV 会占用大量显存（`M * head_dim`，其中 `M = N + window_size + num_tokens`）。`PREFILL_CHUNK_SIZE = 4` 限制了一次处理的序列数。

**decode 直接消费 FP8**：`flash_mla_with_kvcache` 内核内部处理 FP8 反量化，无需先反量化再传输。这是 FlashMLA 的关键优化——避免中间 BF16 缓冲区的读写。

### 2.6 Compressor State Cache 的读写

```
写入（compressor.forward 每次被调用）:
  save_partial_states(kv, score, ape, positions, state_cache, slot_mapping)
  └─ 按 slot_mapping 将 kv_state + score_state + APE 写入 state_cache
  └─ 然后 compress_norm_rope_store 从 state_cache 中读取历史窗口

读取（隐式，compress_norm_rope_store 内部）:
  └─ 从 state_cache 读取 [pos-window:pos+1] 范围的 state
  └─ 加权压缩 → norm → RoPE → FP8 quant → 写入主 KV cache
```

**为什么需要 state cache**：Compression ratio 4 意味着每 4 个 token 才产生一个压缩 KV。state cache 以 float32 保存了完整精度的中间状态，在压缩时提供滑动窗口上下文。

![compressor state cache flow]

### 2.7 Indexer KV Cache 的读写

Indexer 有自己独立的 Compressor 和 KV Cache：

```
Indexer 内部 compressor（indexer.compressor）:
  ┌─ 与主 compressor 共用 hidden_states
  ├─ fused_wkv_wgate → kv_score（在 attn_gemm_parallel_execute 辅流2 计算）
  └─ forward: save_partial_states → compress_norm_rope_store
       └─ 写入 indexer.k_cache（FP8 或 MXFP4）

Indexer forward 读取:
  SparseAttnIndexer(hidden_states, q_quant, k, weights)
  └─ 基于 q_quant 和 k 计算稀疏注意力分数
  └─ 选择 topk 索引 → 写入 topk_indices_buffer
```

**与主 KV cache 的交互**：Indexer 的主 KV cache 由 `DeepseekV4IndexerCache` 管理，这是一个独立的 `AttentionLayerBase` 子类，向 vLLM 框架独立声明缓存规格。因此框架会为它分配独立的 block table。

### 2.8 SWA Cache 的读写

SWA（Sliding Window Attention）缓存是一个固定大小的环形缓冲区：

```python
self.swa_cache_layer = DeepseekV4SWACache(
    head_dim=self.head_dim,
    window_size=self.window_size,
    dtype=torch.uint8,
    cache_config=cache_config,
)
```

写入：通过 `_fused_qnorm_rope_kv_insert` 中的 `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` 完成（`attention.py:L532-L542`）。

读取：在 `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` 中作为参数传入。

### 2.9 CUDA Graph 下的缓存管理

关键挑战：CUDA Graph 要求所有被捕获的内核参数地址在重放时不变。

- **预分配策略**：`_mtp_hidden_buffer`、`topk_indices_buffer`、`symm_buffer` 都在 `__init__` 中预分配，地址在整个生命周期固定
- **Tile scheduler metadata**在首次调用时由内核内规划器分配（PyTorch 图感知分配器），之后复用（`have_initialized=True`）
- **`non_blocking=True`**：元数据更新（如 block_table 更新）使用异步复制，不与计算流冲突

---

## 3. 生命周期总结

```
权重生命周期:
  Checkpoint → 映射 → 加载 → [MegaMoE: finalize] → GPU 参数 → 推理使用

MLA KV Cache 生命周期:
  缓存管理器分配 → 写入（量化+存储） → 读取（反量化+注意力） → 释放

Compressor State Cache 生命周期:
  缓存管理器分配 → 写入（float32 状态） → 读取（压缩计算） → 释放

Indexer KV Cache 生命周期:
  缓存管理器分配 → 写入（Indexer 内 compressor） → 读取（稀疏索引） → 释放

SWA Cache 生命周期:
  Wrapper __init__ 创建 → 写入（融合 op） → 读取（注意力内核） → 释放
```

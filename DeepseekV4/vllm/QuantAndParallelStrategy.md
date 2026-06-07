# DeepSeek V4 量化和并行策略

## 1. 量化策略全景

### 1.1 支持的量化和数据类型

| 量化层级 | 格式 | 缩放因子 | 适用场景 | 核心依赖 |
|---------|------|---------|---------|---------|
| 注意力权重 | FP8（block-wise） | weight_scale_inv（float32/uint8） | 所有 Linear 层 | `quant_config` → `Fp8Config` |
| KV 缓存 | FP8 E4M3（`fp8_ds_mla`） | 无（动态 per-tensor） | FlashMLA 稀疏注意力 | `DeepseekV4MLAAttention` 强制设置（`attention.py:L89`） |
| KV 缓存（每 token 584 字节） | FP8 对齐布局 | — | FlashMLA 要求 576B 对齐 | `FlashMLASparseBackend.get_kv_cache_shape` |
| MoE 专家权重（MegaMoE） | FP4（MXFP4，uint8 打包） | ue8m0，block_size=32 | SM100 Blackwell | `DeepseekV4MegaMoEExperts` |
| MoE 专家权重（FusedMoE FP4） | MXFP4 | ue8m0 | 通用 GPU | `Mxfp4MoEMethod` |
| MoE 专家权重（FusedMoE FP8） | FP8 block | float32 block scales | 通用 GPU | `Fp8MoEMethod` |
| MoE 专家权重（NVFP4） | NVFP4 | group_size=16 | 特定 GPU | `ModelOptNvFp4FusedMoE` |
| MoE 输入激活（MegaMoE） | FP8 E4M3 | E8M0，block_size=128 | DeepGEMM 输入准备 | `prepare_megamoe_inputs` (Triton kernel) |
| 注意力输出到 wo_a | FP8（UE8M0 动态） | group_shape=(1,128) | O 投影计算 | `DeepseekV4MultiHeadLatentAttentionWrapper`（`nvidia/model.py:L54`） |

### 1.2 整体混合精度

```
激活（BF16） → 注意力权重（FP8） → KV Cache（FP8）
                                  → MoE 专家权重（FP4/FP8）
                                  → 共享专家权重（FP8）
                                  → 注意力输出（FP8 量化，UE8M0）
                                  → 输出投影 FP8 Einsum → 输出（BF16）
```

- 全模型保持 BF16 激活主流计算。
- KV 缓存全为 FP8，每 token 584 字节（448 NoPE + 128 RoPE + 8 字节缩放因子/元数据）。
- MoE 路径额外增加一道量化：MegaMoE 下激活在送入专家前被量化为 FP8 E4M3，专家权重为 FP4。

### 1.3 量化配置分发

`DeepseekV4FP8Config`（`vllm/models/deepseek_v4/quant_config.py`）是量化配置的分发中心：

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, FusedMoE):
        if self.expert_dtype == "fp4":
            if self.moe_quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(group_size=16)
            return Mxfp4MoEMethod()
        # expert_dtype == "fp8": 回退到 Fp8Config 的 Fp8MoEMethod
    return super().get_quant_method(layer, prefix)  # → Fp8LinearMethod for Linear layers
```
（`quant_config.py:L132-153`）

关键设计：`expert_dtype` 的解析被延迟（lazy property），因为 `DeepseekV4FP8Config` 在 `VllmConfig` 设置阶段创建，此时 `hf_config` 尚不可用。首次访问时才从 `hf_config` 读取真实的 `expert_dtype`（`quant_config.py:L56-76`）。

### 1.4 FP8 KV Cache 的布局

```
每个 token 的 KV 在缓存中占 584 字节：
  ┌─────────────────┬─────────────────┬──────┐
  │ NoPE (448B)     │ RoPE (128B)     │ meta │
  │ FP8 E4M3        │ BF16            │ 8B   │
  └─────────────────┴─────────────────┴──────┘

> **注意**：RoPE 部分不做量化，以 BF16 格式直接存储。64 个 RoPE 元素 × 2 字节(BF16) = 128 字节。缩放因子 8 字节中 7 个为 UE8M0（7 个 FP8 量化块的缩放），第 8 个为 padding。
```

- 由 `FlashMLASparseBackend.get_kv_cache_shape()` 定义（`attention.py?`），block_size=256 时每块大小 `256 × 584`。
- 对齐要求 576B 来自 FlashMLA 的缓存 block 对齐特性。

### 1.5 FP8 Einsum 的硬件自适应

```python
cap = current_platform.get_device_capability()
self._einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
self._tma_aligned_scales = cap.major >= 10
```
（`nvidia/model.py:L71-73`）

- SM90 (H100)：`(1, 128, 128)` 块缩放。
- SM100 (Blackwell)：`(1, 1, 128)` 逐元素缩放，支持 TMA 对齐。

### 1.6 缩放因子的 E8M0 格式

ue8m0（也称 e8m0fnu）是无符号 8 位指数格式——只有 8 位指数，没有尾数位：
- 值范围：`[0, 255]`，表示指数 E
- 实际缩放因子：`2^(E-127)`（转换见 `DeepseekV4MegaMoEExperts._ue8m0_uint8_to_float`）
- 主要用于 FP8 和 FP4 的逐块缩放因子存储

---

## 2. 并行策略全景

### 2.1 支持的并行维度

| 并行维度 | 配置参数 | 模型范围 | 通信模式 |
|---------|---------|---------|---------|
| 张量并行（TP） | `tensor_parallel_size` | 所有 Linear 层 | all-reduce（行并行）/ 无（列并行） |
| 流水线并行（PP） | `pipeline_parallel_size` | 按层切分 | point-to-point（PP rank 间） |
| 专家并行（EP） | `enable_expert_parallel` | MoE 专家层 | all-to-all（DeepGEMM 内核内部） |

### 2.2 TP 在注意力层中的分布

| 层 | TP 类型 | 策略 | 通信 |
|----|---------|------|------|
| `fused_wqa_wkv` | `MergedColumnParallelLinear` | 禁用 TP（ReplicatedLinear） | 无（复制） |
| `wq_b` | `ColumnParallelLinear` | 按头切分（`n_local_heads`） | 无（列并行） |
| `wo_a` | `ColumnParallelLinear` | 按组切分（`n_local_groups`） | 无（列并行） |
| `wo_b` | `RowParallelLinear` | 输入切分 + all-reduce | **all-reduce** |

**设计思考：** `fused_wqa_wkv` 输出维度小（`q_lora_rank + head_dim`，约 1536+512=2048），复制比切分后 all-gather 更高效。`wo_b` 的 all-reduce 无法避免，因为需要聚合所有 TP rank 的输出以得到完整的 `hidden_size`。

### 2.3 TP 在 MoE 层中的分布（FusedMoE 路径）

| 层 | TP 类型 | 说明 |
|----|---------|------|
| `gate` | `ReplicatedLinear` | 各 rank 独立计算完整路由 |
| `experts` | `FusedMoE` 层 | 各 rank 持有 `n_routed_experts / tp_size` 个专家 |
| `shared_experts.gate_up_proj` | `MergedColumnParallelLinear` | 列切分 |
| `shared_experts.down_proj` | `RowParallelLinear` | 行切分 + all-reduce |

### 2.4 PP（流水线并行）在 DeepSeek V4 中的切片

`DeepseekV4Model` 使用 `make_layers`（`vllm/model_executor/models/utils.py`）按层切分：

```python
start_layer, end_layer = get_pp_indices(
    num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size)
```
（`nvidia/model.py:L1128-1130`）

- 嵌入层仅在 PP 首 rank 创建（`PPMissingLayer` 占位其余 rank）。
- 解码器层按 `[start_layer, end_layer)` 区间分配。
- 最终 norm 仅在 PP 末 rank 创建。
- `IntermediateTensors` 在 PP 各 stage 间传递 `hidden_states`（`[num_tokens, hc_mult, hidden_size]`）。

### 2.5 EP（专家并行）在 MegaMoE 中的分配

```python
self.n_local_experts = config.n_routed_experts // self.ep_size
self.experts_start_idx = self.ep_rank * self.n_local_experts
```
（`nvidia/model.py:L555-558`）

- DeepGEMM MegaMoE 内核内部处理 all-to-all 通信（`deep_gemm.fp8_fp4_mega_moe`）。
- 对称缓冲区（`symm_buffer`）包含通信所需的中间缓冲区（`model.py:L319-343`）。
- MTP module 同样使用 expert mapping（`mtp.py` 引用了 `make_deepseek_v4_expert_params_mapping`）。

### 2.6 多 CUDA 流并行（Micro-parallelism）

`DeepseekV4Model` 创建 3 个辅助 CUDA 流（仅 CUDA 平台，ROCm/XPU 禁用）：

```python
aux_stream_list = (
    None if current_platform.is_rocm() or current_platform.is_xpu()
    else [torch.cuda.Stream() for _ in range(3)])
```
（`nvidia/model.py:L1102-1106`）

三个流的用途（在 `DeepseekV4MultiHeadLatentAttentionWrapper.attn_gemm_parallel_execute` 中）：

| 流索引 | 执行任务 | 与主流的依赖 |
|--------|---------|-------------|
| 0 | Compressor 的 `kv_score`（`fused_wkv_wgate` × hidden_states） | 无（独立 GEMM） |
| 1 | Indexer 的 `weights_proj`（线性投影） | 无（独立 GEMM） |
| 2 | Indexer 的 Compressor 的 `kv_score` | 无（独立 GEMM） |
| 默认流（主） | `fused_wqa_wkv`（Q/KV 低秩投影） | 无 |

同步使用 `torch.cuda.Event` 实现：

```python
qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
    fused_wqa_wkv, aux_fns,
    self.ln_events[0], self.ln_events[1:4], aux_streams,
    enable=hidden_states.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
)
```
（`attention.py`）

小 batch 时多流无法充分利用 GPU，通过 `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` 控制回退到串行。

---

## 3. 全模型量化 + 并行决策矩阵

| 模块 | 数据类型 | TP | PP | EP | 多流 | 量化方法 |
|------|---------|-----|-----|-----|------|---------|
| VocabParallelEmbedding | bf16 | ✓（按词表切分） | ✓（首 rank） | — | — | — |
| DeepseekV4DecoderLayer | bf16/hc_mult | — | ✓（分层） | — | — | — |
| fused_wqa_wkv (Linear) | bf16 权重+激活 | 复制 | — | — | — | FP8 block |
| wq_b (ColumnParallel) | bf16 | ✓（按头） | — | — | — | FP8 block |
| wo_a (ColumnParallel) | bf16 权重，FP8 输入 | ✓（按组） | — | — | — | FP8 + FP8 Einsum |
| wo_b (RowParallel) | bf16 | ✓（+all-reduce） | — | — | — | FP8 block |
| DeepseekV4MLAAttention | bf16 Q，FP8 KVCache | — | — | — | ✓（3 个流） | KV Cache FP8 |
| MoE FusedMoE 路径 | FP4/FP8/BF16 权重 | ✓（按专家） | — | — | — | Mxfp4/Fp8 |
| MoE MegaMoE 路径 | FP4 权重，FP8 激活 | — | — | ✓（all-to-all） | — | FP4 weight + FP8 act |
| 共享专家 MLP | FP8/BF16 权重 | ✓ 或 复制 | — | — | — | FP8 block |
| HC 操作 | float32 参数，bf16 数据 | — | — | — | — | 无 |
| ParallelLMHead | bf16 | ✓（按词表） | ✓（末 rank） | — | — | FP8 block |
| LogitsProcessor | bf16 → float32 | 无 | ✓（末 rank） | — | — | 无 |
| DeepseekCompressor | float32 缓存，bf16 输入 | 复制 | — | — | ✓（流 0） | FP8 KVCache |

---

## 4. 量化与并行的交互影响

### 4.1 PP 下的 MTP 中间状态

PP 非末 rank 的隐藏状态通过 `IntermediateTensors` 传递，不涉及量化格式转换。末 rank 才进行 `HCHeadOp` 合流和 norm。

### 4.2 EP 下的量化差异

EP 下每个 rank 持有不同的专家子集，因此每个 rank 只需加载和量化其本地专家。gate 路由在所有 rank 上复制计算以产生全局一致的 `topk_ids`。

### 4.3 TP + EP 混合时的问题

当前 DeepSeek V4 的 MegaMoE 不支持 TP + EP 混合（会抛出 `NotImplementedError`）。FusedMoE 路径支持 TP 但不支持 EP。二者通过 `use_mega_moe` 标志互斥。

### 4.4 多流并行与量化的交互

辅助流上执行的 `kv_score` 计算（`compressor.fused_wkv_wgate`）的输出 dtype 为 `float32`（`attention.py`）。这避免了在辅助 GEMM 中插入额外的量化步骤，简化同步逻辑。

---

## 5. 调试与验证

### 验证 KV Cache dtype

```python
# DeepseekV4MLAAttention 强制检查
assert kv_cache_dtype.startswith("fp8"), "DeepseekV4 only supports fp8 kv-cache format"
# 自动转换为 fp8_ds_mla 布局
if kv_cache_dtype != "fp8_ds_mla":
    cache_config.cache_dtype = "fp8_ds_mla"
```
（`attention.py`）

### 验证 MegaMoE 的 FP4 要求

```python
assert self.expert_dtype == "fp4", "MegaMoE only supports fp4 experts"
```

### 多流并行的环境变量控制

```bash
# 设置小 batch 阈值，控制多流退化为串行
export VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=4096
# 设置为 0 强制始终使用多流
export VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0
```

### 内存占用计算

| 组件 | 公式 | 说明 |
|------|------|------|
| KV Cache（注意力） | `num_layers × num_blocks × block_size × 584` | FP8，每 token 584 字节 |
| KV Cache（SWA） | `num_layers × window_size × head_dim` | FP8，滑动窗口 |
| MoE 专家权重（MegaMoE FP4） | `n_local_experts × (2*intermediate*hidden//2 + hidden*intermediate//2)` × 2 bytes | uint8 存储，约等于 BF16 的 1/4 |
| MoE 专家权重（FusedMoE FP4） | 同上，但 TP rank 持有 | MXFP4 打包 |
| Symm Buffer（MegaMoE） | `max_num_tokens × (hidden_size + hidden//128 × 4 + top_k × 12)` | fp8 + int32 + int64 + float32 |

---

## 6. 设计亮点总结

1. **分层量化**：不同模块使用不同精度（FP8 KV cache、FP4 专家权重、FP8 激活），在精度和带宽之间取得平衡。
2. **硬件自适应量化**：FP8 Einsum 的 recipe 根据 SM 版本自动选择（SM90 块缩放 vs SM100 逐元素缩放）。
3. **复制 vs 切分的智能选择**：小维度层复制避免通信，大维度层切分减少计算。
4. **EP 与 TP 互斥**：简化了运行时调度，避免嵌套并行导致的复杂性。
5. **多流并行与量化无关**：辅助流避免量化，减少同步复杂度。
6. **晚量化**：MegaMoE 的输入在送入 DeepGEMM 前才做 FP8 量化（`prepare_megamoe_inputs`），保持中间计算精度。

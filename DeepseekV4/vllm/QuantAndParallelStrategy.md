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

> MoE 相关的详细量化数据格式见 [[MoEQuantization]]。

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

### 1.7 模块级量化详情

#### 1.7.1 Attention Linear 层（fused_wqa_wkv / wq_b / wo_b）

标准 FP8 block 权重量化（W8A16），激活保持 BF16 不量化。

| 参数 | 激活 | 权重 |
|------|------|------|
| 数据类型 | BF16 | FP8 (E4M3) |
| Scale dtype | — | float32 |
| gran_k (block_k) | — | 128 |
| gran_n (block_n) | — | 128 |
| Scale shape | — | `[out_dim/128, in_dim/128]` |
| 量化模式 | 无 | 静态 per-block (`weight_scale_inv`) |
| 反量化方式 | — | 软件 (kernel 内逐 block 乘 scale) |

#### 1.7.2 wo_a + FP8 Einsum（FP8 权重 + FP8 激活）

wo_a 的权重和输入均为 FP8，通过 FP8 Einsum 完成 FP8×FP8 计算。

| 参数 | 输入（注意力输出） | 权重 |
|------|-------------------|------|
| 数据类型 | FP8 (E4M3) | FP8 (E4M3) |
| Scale dtype | UE8M0 (activation scale) | float32 (weight_scale_inv) |
| gran_k | 128 (group_shape=(1,128)) | 128 (per-block) |
| Scale shape | SM90: `(1, 128, 128)` block；SM100: `(1, 1, 128)` per-elem | per-block `[out_dim/128, in_dim/128]` |
| 量化模式 | 动态 per-token-group（UE8M0） | 静态 per-block |
| 反量化方式 | SM90 软件 / SM100 硬件 UMMA | 同左 |

Einsum recipe 根据 SM 版本自适应（§1.5），激活和权重的 scale 数据经由 TMA 加载。

**关键数据通路：**
```
attention_output (BF16) → FP8 量化 + UE8M0 scale → wo_a FP8 Einsum (FP8 weight + weight_scale_inv) → output (BF16)
```

#### 1.7.3 KV Cache（FP8 E4M3 + BF16 RoPE）

| 参数 | NoPE | RoPE |
|------|------|------|
| 数据类型 | FP8 (E4M3) | BF16 |
| Scale dtype | UE8M0 | — |
| gran_k | 64（448 元素 / 7 scale） | — |
| Scale shape | 每 token 7B (7×UE8M0) + 1B padding | — |
| 量化模式 | 动态 per-block（FlashMLA 内部） | 无量化 |

- 每 token 共 584 字节：448B NoPE FP8 + 128B RoPE BF16 + 8B 元数据
- FlashMLA 后端要求 576B 对齐，缓存 block 大小 256 token
- KV cache dtype 强制为 `fp8_ds_mla`（`attention.py:L89`）

#### 1.7.4 MegaMoE（W4A8：FP4 权重 + FP8 激活）

MegaMoE 路径同时量化激活和权重，分 L1（gate+up）和 L2（down）两阶段。

**激活量化（`prepare_megamoe_inputs` Triton kernel）：**

| 参数 | 输入激活 |
|------|---------|
| 输入 dtype | BF16 |
| 输出 dtype | FP8 (E4M3) |
| Scale dtype | E8M0（4 个打包为 int32） |
| gran_k | 32（子分组）→ 128（分组打包） |
| Scale shape | `[T, H/128]` (int32) |
| 量化模式 | 动态 per-group |

**权重量化：**

| 参数 | w13（gate+up） | w2（down） |
|------|---------------|-----------|
| 数据类型 | FP4 (E2M1) | FP4 (E2M1) |
| 存储 dtype | uint8（每字节 2 FP4） | uint8 |
| Scale dtype | E8M0 (uint8) | E8M0 (uint8) |
| gran_k | 32 | 32 |
| Scale shape | `[E, 2×I, H/32]` | `[E, H, I/32]` |
| 量化模式 | 静态 per-group | 静态 per-group |
| 反量化方式 | 硬件 UMMA (SM100 DeepGEMM) | 同左 |

**数据通路：**
```
BF16 act → FP8 quant (E8M0, gs=32) → DeepGEMM fp8_fp4_mega_moe (L1: FP8×FP4 + SwiGLU)
  → L1 out re-quant (FP8+UE8M0) → L2: FP8×FP4 → BF16 out
```

#### 1.7.5 FusedMoE MXFP4（W4A16）

| 参数 | 激活 | 权重 |
|------|------|------|
| 数据类型 | BF16 | FP4 (E2M1) |
| 存储 dtype | BF16 | uint8（每字节 2 FP4） |
| Scale dtype | — | E8M0 (uint8) |
| gran_k | — | 32 |
| Scale shape | — | `[E, 2×I, H/32]` / `[E, H, I/32]` |
| 量化模式 | 无 | 静态 per-group |
| 后端 | — | Triton / FlashInfer / TRTLLM / AITER |

激活不量化，仅在权重侧做 MXFP4 per-group 量化。

#### 1.7.6 FusedMoE NVFP4（W4A16 / W4A4）

NVFP4 使用三层缩放（block + global + input），group_size=16。

**W4A16 模式（激活不量化）：**

| 参数 | 激活 | 权重 |
|------|------|------|
| 数据类型 | BF16 | FP4 (E2M1) |
| Block scale dtype | — | FP8 E4M3 |
| Global scale dtype | — | FP32 per-tensor |
| gran_k (group_size) | — | 16 |
| Scale shape | — | block: `[E, 2×I, H/16]` + global: `[E, 2]`/`[E,]` |
| 量化模式 | 无 | 静态 per-group + global |

**W4A4 模式（激活也量化）：**

NVFP4 W4A4 的激活量化有两层，容易被混淆：

| 层级 | 粒度 | 存在形式 | 作用 |
|------|------|---------|------|
| `input_scale` 参数 | **per-tensor**（每专家 1 个 FP32） | 静态，存储在 checkpoint | 全局激活 scale 偏置，三层缩放体系的一环 |
| kernel 内动态量化 | **per-16-group**（每 16 元素） | 动态，kernel 内部实时计算 | 实际将激活量化为 FP4 的粒度 |

数据流：
```
activation_bf16
  → kernel 内 per-16-group 动态量化 → act_fp4（float32 scale，暂存寄存器）
  → 反量化: act = dequant(act_fp4 × per_group_scale × input_scale)
  → FP4 weight × FP4/FP8 activation GEMM
```

| 参数 | 激活 | 权重 |
|------|------|------|
| 数据类型 | FP4 (E2M1) | FP4 (E2M1) |
| Input scale（全局偏置） | FP32 per-tensor（静态参数） | — |
| Block scale dtype | — | FP8 E4M3 |
| Global scale dtype | — | FP32 per-tensor |
| 实际量化粒度 (gran_k) | 16（kernel 内动态 per-group） | 16 |
| 量化模式 | 动态 per-16-group + 静态 per-tensor input_scale | 静态 per-group + global |

#### 1.7.7 FusedMoE FP8 Block（W8A8）

| 参数 | 激活 | 权重 |
|------|------|------|
| 数据类型 | FP8 (E4M3) | FP8 (E4M3) |
| Scale dtype | float32（动态 per-token） | float32（静态 per-block） |
| gran_k / block | per-token（沿 hidden 维度） | 128×128 block |
| Scale shape | `[T]` (per-token) | `[E, 2×I/128, H/128]` / `[E, H/128, I/128]` |
| 量化模式 | 动态 per-token | 静态 per-block |
| 反量化方式 | 软件 (K 循环内逐 block 乘 scale) | 同左 |

**激活量化细节：**
```
每 token 计算 absmax → scale = amax / 448 → x_fp8 = x_bf16 / scale
scale 在 kernel 内部计算并留在寄存器中，不经过 TMA。
```

#### 1.7.8 共享专家 MLP（W8A16）

共享专家使用 `Fp8LinearMethod`，不与 MoE 共享量化路径。

| 参数 | 激活 | 权重 |
|------|------|------|
| 数据类型 | BF16 | FP8 (E4M3) |
| Scale dtype | — | float32 |
| gran_k | — | 128 (block) |
| Scale shape | — | per-block `[out_dim/128, in_dim/128]` |
| 量化模式 | 无 | 静态 per-block |

#### 1.7.9 Indexer MQA Logits（FP8 / FP4）

**SM90 FP8 MQA Logits（软件反量化）：**

| 参数 | Q | KV |
|------|------|------|
| 数据类型 | FP8 (E4M3) | FP8 (E4M3) |
| Scale dtype | — | FP32 kv_scales |
| gran_k | — | 128 |
| Scale shape | — | `[ceil(seq_len_kv, BLOCK_KV)]` (1D) |
| 量化模式 | — | 静态 per-KV-block |
| 反量化方式 | 软件 (标量乘) | 同左 |

**SM100 FP4 MQA Logits（硬件 block-scaling）：**

| 参数 | Q | KV |
|------|------|------|
| 数据类型 | FP4 (E2M1) | FP4 (E2M1) |
| Scale dtype | UE8M0 sf_q | UE8M0 sf_kv |
| gran_k | per-query | 128 |
| Scale shape | `[nHeads, seq_len]` | `[seq_len_kv, 1]` (1D) |
| 量化模式 | 动态 per-head per-query | 静态 per-KV-block |
| 反量化方式 | 硬件 UMMA block-scaling | 同左 |

#### 1.7.10 HC 操作（无量化）

| 参数 | 值 |
|------|-----|
| 数据类型 | BF16 (数据) + FP32 (参数) |
| Scale | 无 |
| 计算精度 | TF32 / FP32 |
| 量化模式 | 无量化 |

HC 操作的参数（如 head covariance 矩阵）以 FP32 精度存储，数据流保持 BF16。使用 `tf32_hc_prenorm_gemm` 内核，在 A 加载阶段手动 BF16→FP32 转换并累加平方和。

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

---

## 7. FP4 精度保障：量化感知训练（FP4 QAT）

> 以下内容综合 DeepSeek-V4 技术报告 §3.4、§4.2.3、§5.2.1 及第三方技术分析，原载 [[MoEQuantization]]。


> **来源说明：** 本节内容综合两类信息来源：
> - **\[Paper\]**：DeepSeek-V4 技术报告（arXiv, 2026）§3.4 FP4 Quantization-Aware Training、§4.2.3 Mitigating Training Instability、§5.2.1 FP4 Quantization Integration
> - **\[Third-party\]**：第三方技术博客和分析文章，具体来源如下：
>   - \[1\] [Fireworks AI: Notes on DeepSeek-V4's training system](https://fireworks.ai/blog/what-deepseek-v4-says-about-training-platforms) — Anticipatory Routing 分析、overhead、spike detector
>   - \[2\] [EET China: DeepSeek V4 MXFP4 量化感知训练技术](https://www.eet-china.com/mp/a494319.html) — MXFP4 QAT 技术解读
>   - \[3\] [Bestaifor: DeepSeek V4 paper full version FP4 QAT details](https://www.bestaifor.com/blog/deepseek-v4-paper-full-version-is-out-fp4-qat-details-and-st) — FP4 QAT 细节、benchmark 推测数据
>   - \[4\] [Jianyu's Blog: DeepSeek-V4 Infra](https://jianyuh.github.io/llm/2026/04/25/DeepSeek-V4-Infra.html) — FP4 QAT、TileLang、Hybrid KV Cache
>   - \[5\] [Reddit r/LocalLLaMA: DeepSeek V4 paper full version discussion](https://www.reddit.com/r/LocalLLaMA/comments/1t7yrvr/deepseek_v4_paper_full_version_is_out_fp4_qat/) — FP4 QAT benchmark 讨论
>   - \[6\] [LMSYS Blog: DeepSeek-V4 on Day 0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) — 整体技术解读
>   - \[7\] [BlockBeats: DeepMind 研究员评价 DeepSeek V4 稳定性方案](https://en.theblockbeats.news/flash/342862) — Susan Zhang "band-aid" 评论

### 7.1 问题：为什么需要 QAT 而非 PTQ

FP4 E2M1 只有 8 个可表示的正值 {0, 0.5, 1, 1.5, 2, 3, 4, 6}。如果直接用 PTQ（训练完直接 round 到 FP4），精度损失通常超过 5%。原因是 BF16 训练出的权重分布没有被约束——权重可以任意取值，round 到 FP4 后信息大量丢失。

DeepSeek V4 的解决方案：在 **post-training stage** 引入 **Quantization-Aware Training (QAT)**，让模型主动适应 FP4 精度损失。\[Paper: §3.4\]

**FP4 量化的两个目标：**
1. **MoE 专家权重**——GPU 显存的主要占用来源
2. **CSA indexer 的 QK 路径**——QK 激活值的缓存、加载、矩阵乘全部在 FP4 精度下完成，加速长序列的注意力分数计算

额外量化：index scores 从 FP32 压缩到 BF16，top-k selector 加速 2×，KV recall 保持 99.7%。\[Paper: §3.4\]

### 7.2 QAT 前向-反向机制

**前向传播（模拟量化 → 无损反量化 → FP8 计算）：** \[Paper: §3.4\]

```
FP32 master weights (优化器维护)
  → quantize: fp32 → fp4 (per-group scale)
  → dequantize: fp4 × scale → fp8 (E4M3)    ← 无损反量化
  → FP8 × FP8 GEMM (复用现有 FP8 训练框架)
  → loss
```

**反向传播（STE 通过 FP8 权重直接回传）：** \[Paper: §3.4\]

```
loss
  → ∂Loss/∂w_fp8 (计算图用 FP8 权重)
  → Straight-Through Estimator: 梯度直接传回 FP32 master weights
  → 更新 FP32 master weights
  → 下一轮前向再重新 round 到 FP4
```

关键设计：
- **Master weights 是 FP32，不是 BF16**（论文原文 "the FP32 master weights maintained by the optimizer"）\[Paper: §3.4\]
- 梯度以 FP8 权重为基准计算，通过 STE 直接回传到 FP32 master weights，**避免了重用量化转置权重** \[Paper: §3.4\]
- 前向计算中 FP4→FP8 反量化是 lossless 的（见 §8.4），因此整个 QAT 管线可**完全复用现有 FP8 训练框架，反向传播零修改** \[Paper: §3.4\]

> **什么是 STE（Straight-Through Estimator）？**
>
> 量化的根本难题：`round()` 和 `floor()` 这类操作的梯度几乎处处为零（或未定义），无法用于反向传播。
>
> ```
> 前向:  w_quant = round(w_fp32 / scale)        ← round() 不可导
> 反向:  ∂Loss/∂w_fp32 = ∂Loss/∂w_quant * ∂w_quant/∂w_fp32
>                        = ∂Loss/∂w_quant * 0     ← 梯度消失，参数永远不更新！
> ```
>
> STE 的解决方案极其简单——**在前向传播时照常做不可导的量化，但反向传播时假装 round() 是恒等映射**：
>
> ```
> 前向:  w_quant = round(w_fp32 / scale)    ← 该怎么做怎么做
> 反向:  ∂w_quant/∂w_fp32 ≈ 1               ← STE: 梯度直接穿过量化函数
>        ∂Loss/∂w_fp32 ≈ ∂Loss/∂w_quant     ← 梯度无损传回
> ```
>
> 数学上等价于将 `round()` 替换为一个恒等函数（"直通"名字的来源），但前向结果仍保持离散取值。这允许梯度更新 FP32 master weights，而 master weights 下次前向会被重新量化到 FP4：
>
> ```
> w_fp32 ← w_fp32 - lr * ∂Loss/∂w_quant    ← STE 让梯度更新真正生效
> w_fp4  ← round(w_fp32 / scale)           ← 下一轮前向重新量化
> ```
>
> **为什么 STE 避免了重用量化转置权重？** 反向传播时需要计算 `∂Loss/∂X = ∂Loss/∂Y · W^T`，其中 `W` 是权重。如果 `W` 以 FP8 保存在计算图中且梯度通过 STE 直接回传，就不需要把 FP4 权重重新量化到 FP8 再转置——FP8 权重已经在前向计算时存在于图中。如果使用标准的 fake quantization 框架（如 PyTorch's `torch.ao.quantization`），每一轮前向都需重新反量化权重，再交给 GEMM，而这里 FP4→FP8 反量化后的结果直接就是 FP8，可以被计算图保留重用于反向的转置 GEMM。\[Paper: §3.4\]

### 7.3 Rollout 和推理阶段：直接使用原生 FP4

论文明确区分了两种模式：\[Paper: §3.4\]

| 阶段 | 权重格式 | 说明 |
|------|---------|------|
| **QAT 训练 step** | FP4 → FP8 模拟反量化 | FP32 master weights → FP4 → FP8，需要 backward |
| **RL rollout / 推理** | **原生 FP4 权重** | 直接加载真实的 FP4 权重，不做模拟反量化 |

> **什么是 RL Rollout？**
> 
> RL 训练的核心矛盾：**没有现成的标准答案**，模型必须自己"试一下"才知道好坏。
> 
> ```
> SFT（监督微调）:  有标准答案 → 直接算 loss → 不需要 rollout
> RL（强化学习）:  没有标准答案 → 先生成再看 → 必须先 rollout
> ```
> 
> "Rollout" 来自 RL 术语——agent（模型）在环境中"展开"一段轨迹，如同卷尺滚出（roll out）。每步生成一个 token 看作 RL 中的一步 action，整个序列就是一条 trajectory。
> 
> 以 DeepSeek V4 的 OPD（On-Policy Distillation）为例：
> ```
> [Rollout]
>   给定 prompt "写一个排序函数"
>   student 自回归生成 → "def sort(arr):\n    return sorted(arr)"  ← 每一步都是 student 当前策略下的真实选择
> 
> [Training]
>   把 student 生成的每个 token 喂给 teacher
>   teacher: "这一步选 sorted(arr) 的概率应该是 0.95"
>   student 实际: 概率 0.7
>   → KL loss 推动 student 分布向 teacher 靠拢
> ```
> 
> 必须在 student **自己生成的 token** 上评估 policy（on-policy），不能用预存的标准答案替代——这正是 RL 区别于 SFT 的价值：**从自己的错误中学习，而非模仿标准答案。**
> 
> 回到 FP4：rollout 阶段没有 backward，因此可以**直接用原生 FP4 权重**而非 QAT 的模拟反量化。这保证了：
> - **采样行为与线上部署完全一致**（无 simulation-reality gap）
> - **减少 kernel 内存加载，实际加速**
> - **显著降低显存消耗**

\[Paper: §3.4\]

### 7.4 FP4→FP8 无损反量化的数学条件

FP8 (E4M3) 比 FP4 (E2M1) 多 2 个指数位，动态范围更大。因此反量化无损的条件是：\[Paper: §3.4\]

```
FP8 的宽动态范围能吸收 FP4 per-group scale 的差异

具体条件:
  FP4 sub-block (1×32 tile) 之间的 max_scale / min_scale 比值
  不超过 FP8 E4M3 可表示的阈值

  论文验证: 当前权重满足此条件 ✓
```

FP4 per-group scale 的细粒度信息被 FP8 的宽动态范围完全吸收。这是整个 QAT 管线能复用 FP8 框架的理论基础。\[Paper: §3.4\]

> **\[Third-party\]\[3\]\[4\] 补充解读：** 第三方分析认为，FP4→FP8 的"无损"并不绝对——论文报告的 benchmark 结果中数学推理仍有 ~1-2% 的精度差距。此外，per-group scale 的精度（E8M0 纯 2ⁿ 指数 vs FP8 E4M3 非 2ⁿ）会影响 scale 自身的信息损失，这一点论文未展开讨论。

### 7.5 训练稳定性保障

#### 7.5.1 问题根源：Loss Spike

**Loss spike（损失尖峰）** 指训练过程中损失值突然剧烈飙升——正常训练 loss 平稳下降，但在某个 step 突然跳变数倍甚至数十倍。严重时会直接发散到 NaN，导致训练崩溃。

spike 的两大威胁：
1. **破坏已学习的权重**：spike 伴随的巨大梯度会严重扰动模型参数，之前的训练成果可能被部分"洗掉"
2. **简单回滚不够**：回滚能暂时恢复状态，但不阻止复发——论文明确指出 "simple rollbacks... proved inadequate as a long-term solution because they do not prevent the recurrence of loss spikes"

训练万亿参数 MoE 模型时反复出现 loss spike。论文定位到：\[Paper: §4.2.3\]
- Loss spike 的根源是 **MoE 层的 outlier**
- **路由机制本身放大了这些 outlier**——论文用"vicious cycle induced by routing"描述

具体机制（基于论文推断）：
```
MoE 层出现 outlier 激活值
  → gate 路由被 outlier 扰动，将 token 误路由到不匹配的专家
  → 错误专家收到异常输入，输出更加极端
  → 梯度剧烈放大
  → gate 权重被大梯度冲偏
  → 下一轮路由更乱 → 更多 token 误路由 → loss 飙升
```

论文由此提出两个维度的解法：**打断路由的恶性循环**（Anticipatory Routing）和**直接抑制异常值**（SwiGLU Clamping）。

#### 7.5.2 Anticipatory Routing（超前路由）

论文提出解耦 backbone 网络和路由网络的同步更新：\[Paper: §4.2.3\]

```
Step t:
  特征计算:       当前参数 θ_t
  路由索引计算:   历史参数 θ_{t-Δt}    ← 路由决策滞后 Δt 步
```

**工程实现（Prefetch + Cache）：** \[Paper: §4.2.3\]

```
Step t-Δt:  提前 fetch step t 的数据
            用 θ_{t-Δt} 预计算路由索引 → 缓存
Step t:     直接读缓存路由索引
            用 θ_t 做特征计算
```

优化：
- 路由索引只需一次前向，通过 pipeline 编排和 EP 通信重叠，**额外 wall-clock 开销约 20%** \[Paper: §4.2.3\]
- **自动检测 + 短回滚机制**：loss spike 检测器触发时，先执行 **short rollback**（回退几步），再激活 Anticipatory Routing；运行一段时间后自动切回标准训练 \[Paper: §4.2.3\]
- 动态启用使整体额外训练开销可忽略，不损害模型性能 \[Paper: §4.2.3\]

> **\[Third-party\]\[1\] 类比解释：** 超前路由的直觉是"餐厅客人看昨天的评分选厨师，而非看今天手抖的表现"。路由滞后于状态变化，给异常一个缓冲期。

> **\[Third-party\]\[1\]\[7\] 评价：** 第三方分析（Fireworks AI 博客）评价 Anticipatory Routing 为条件运行时干预而非干净的训练目标。Google DeepMind 研究员 Susan Zhang 将稳定性方案评价为"创可贴"（band-aid）——工程上有效，但理论上未根本解决 MoE 的流形撕裂问题。

#### 7.5.3 SwiGLU Clamping

\[Paper: §4.2.3\]

```
linear component: clamp to [-10, 10]
gate component:   upper bound clamped to 10
```

论文实验验证：clamping 有效消除 outlier、稳定训练，且不损害性能。V4-Flash 和 V4-Pro 全程使用此设置。\[Paper: §4.2.3\]

> **\[Third-party\]\[2\] 补充分析：** FP4 的表示范围只有 ±6。SwiGLU 激活函数的中间值可以到数百，如果不做约束，量化后信息完全丢失。Clamp 到 ±10 确保所有值落入 FP4 可表示范围（结合 per-group scale 后有效范围扩大约 ±10×32/6 ≈ ±53）。

#### 7.5.4 论文的坦诚

论文明确说明：\[Paper: §4.2.3\]

> **"Although a comprehensive theoretical understanding of their underlying mechanisms remains an open question for now"**

——这些方案的底层原理仍是开放问题。属于工程上有效、理论上还未完全理解的实用技术。

### 7.6 关于 Master Weights 精度的说明

\[Paper: §3.4\] 论文原文：
> "the FP32 master weights maintained by the optimizer are first quantized to FP4, then dequantized back to FP8 for computation"

明确写的是 **FP32** master weights，不是 BF16。

> **\[Third-party\]\[3\]\[5\] 为何会有 BF16 的说法：** 部分第三方解读推测 FP32 可能在实际实现中被降级为 BF16（因为 MoE 训练中混合精度通常用 BF16 作为 master weights）。但论文原文明确是 FP32。这值得注意，因为 FP32 master weights 意味着优化器状态（momentum、variance）和 master weights 都占用 FP32 存储，显存开销高于 BF16 master weights。

### 7.7 关于 Per-group Scaling 粒度

论文原文仅提到：FP4 sub-block 为 **1×32 tiles**，FP8 quantization block 为 **128×128 tiles**。\[Paper: §3.4\]

> **\[Third-party\]\[2\]\[4\] 补充数值（来自 vLLM 代码分析和第三方解读）：**
> 
> | gran_k | 每组元素 | 推测适用位置 | 信息来源 |
> |--------|---------|-------------|---------|
> | **32** | **32** | **MoE FFN 专家权重** | 论文 sub-block 1×32 tile + vLLM 代码中 scale shape `[E, 2×I, H/32]` |
> | **16** | **16** | **MoE FFN 专家权重** | NVFP4 group_size=16（`modelopt.py`） |
> 
> **解释：** 论文提及的 1×32 "FP4 sub-block within each FP8 quantization block" 针对的是 FP8 训练框架兼容性描述。实际的 FP4 per-group scale 粒度需要从 checkpoint 的 scale tensor shape 推断。vLLM 代码中 MegaMoE 权重的 scale shape 为 `[E, 2×I, H/32]`（`nvidia/model.py:L176-186`），即每 32 个元素一个 scale，与 1×32 tile 描述一致。MXFP4 同样使用 gs=32。NVFP4 使用更细的 gs=16 + FP8 E4M3 scale（`modelopt.py`），精度更高但 scale 开销更大。
>
> **消融实验趋势（第三方推测）：** 理论上 gs 越小精度越高——per-tensor 全矩阵单 scale 不可接受，gs=128 损失仍较大，gs=32 可控制损失在 ~1-2%，gs=16 可进一步降低。但论文本身未发布此消融数据。

### 7.8 关于 Phased QAT Schedule

> **\[Third-party\]\[3\]\[5\] （原始说法，后被论文证伪）：** 一些早期第三方分析推测 DeepSeek V4 使用分阶段量化调度——先量化 MoE FFN 专家权重（占参数量 90%+，对每 token 精度贡献分散），再量化更敏感的 Attention 权重。每个 phase 之间用 warmup 过渡。加速此调度会导致路由 logits 噪声增大、训练 loss 发散。

> **\[Paper\] 实际情况：** 论文 **未提及** 此类分阶段量化调度。FP4 QAT 作为一个整体在 post-training stage 中应用，没有描述对专家和注意力权重分步量化的策略。上述调度描述属于第三方推测，未被论文证实。

### 7.9 FP4 QAT 在 Post-Training 中的集成

FP4 QAT 集成到 post-training 的 RL + OPD 管线中：\[Paper: §5.2.1\]

```
Post-Training 流程:

  多个 Specialist 模型 (SFT + GRPO RL 独立训练)
        ↓
  On-Policy Distillation (OPD)
        ↓
  FP4 QAT: 在 rollout 和推理 forward 中使用原生 FP4 权重
           在训练 step 中使用 FP4→FP8 模拟反量化 + FP32 master weights
        ↓
  导出 FP4 checkpoint
```

其中 teacher 和 reference 模型的 forward 也全部使用 FP4 加速。\[Paper: §5.2.1\]

### 7.10 关于 Benchmark 精度数据

> **\[Paper\] 论文报告的评估方式：** 论文的评估主要针对 base model 和 chat model 的整体性能（MMLU、HumanEval、MATH 等），在 benchmark 表格中对比 V4-Flash/V4-Pro 与 V3.2 等基线，**未单独列出 FP4 QAT 相对于 BF16 基线的精度损失数值**。

> **\[Third-party\]\[3\]\[5\]\[6\] 补充数据（来自第三方博客的引用或推测）：**
> 
> | Benchmark | FP4 QAT vs BF16（第三方报道） |
> |-----------|------------------------------|
> | MMLU（通用推理） | 基线 ± noise |
> | HumanEval（代码） | ~1% 下降 |
> | MATH（数学） | ~1-2% 下降 |
> 
> **注意：** 这些数值在论文中未出现，可能是第三方从论文整体评估结果中间接推算或从其他渠道获取，仅供参考。

### 7.11 总结：FP4 保精度的技术栈

| 技术 | 解决的问题 | 来源 |
|------|-----------|------|
| **QAT + STE** | 让权重分布主动适应 FP4 的离散取值 | \[Paper: §3.4\] |
| **FP32 master weights** | 防止梯度被 FP4 round 吞噬，累积微小更新 | \[Paper: §3.4\] |
| **FP4→FP8 无损反量化** | 复用现有 FP8 计算管线，理论保证反量化精度 | \[Paper: §3.4\] |
| **RL rollout 用原生 FP4** | 消除 simulation-reality gap，保证推理行为一致 | \[Paper: §3.4\] |
| **SwiGLU clamping** | 消除 FP4 范围外的极端 outlier，稳定训练 | \[Paper: §4.2.3\] |
| **Anticipatory routing** | 切断路由→outlier 的正反馈发散回路 | \[Paper: §4.2.3\] |
| **Index scores 量化到 BF16** | 加速 top-k selector 2×，recall 99.7% | \[Paper: §3.4\] |
| **Per-group scaling (gs=32/16)** | 用高密度 scale 补偿 FP4 的数值稀疏性 | \[Third-party\]\[2\]\[4\] / 代码分析 |
| **Phased QAT schedule** | 先量化不敏感层再量化敏感层 | \[Third-party\]\[3\]\[5\]，未获论文证实 |

个人认为主要的几个点：
* QAT：原始精度转FP4，再转FP8进行训练，模拟FP4量化
* SwiGLU clamping：适应FP4的精度范围
* Anticipatory routing：gate的权重用历史时刻的，非当前最新的

---


## 8. 激活量化与 Outlier 处理


### 8.1 什么是 Outlier

**Outlier（异常值/离群值）** 指激活张量中量级远大于同通道其他值的个别元素。例如：

```
正常激活: [0.3, -0.5, 0.2, 0.8, -0.1, 0.4, 0.1, -0.3, ...]   ← 都在 ±1 以内
出现 outlier: [0.3, -0.5, 0.2, 52.7, -0.1, 0.4, 0.1, -0.3, ...]   ← 52.7，比其他值大 100×
```

LLM 中 outlier 的来源：
1. **Attention 的 softmax**：某些 token 对特定位置的注意力权重极高，放大对应 value 的分量
2. **FFN 的激活函数**：SwiGLU/GELU 的非线性在某些输入下产生很大输出
3. **层叠放大**：一层的小波动经过多层矩阵乘后逐渐放大
4. **个别 hidden 维度**：实证发现某些维度天生容易积累大值（与训练数据和初始化相关）

### 8.2 为什么 Outlier 是量化的大敌

以 per-tensor 量化到 FP8（范围 ±448）为例：

```
无 outlier:                        有 outlier:
x = [0.3, 0.8, 1.2, 0.5]          x = [0.3, 52.7, 1.2, 0.5]
amax = 1.2                         amax = 52.7
scale = 1.2/448 = 0.0027           scale = 52.7/448 = 0.118
                                        ↓
x_fp8 = [111, 296, 448, 185]       x_fp8 = [2.5, 448, 10, 4.2]
→ 全部 4 个值精度良好              → 前/后 3 个值被"压扁"到接近 0
```

**一个 outlier 劫持了整个 group 的 scale**。其他正常值除以过大的 scale 后压缩到接近 0，信息几乎完全丢失。这就是 per-tensor 量化几乎不可行的根本原因。

### 8.3 DeepSeek V4 的激活量化全景

共 4 个激活量化阶段：

| 阶段 | 位置 | 输入→输出 | 量化粒度 | Scale dtype | Scale 舍入 | 内核 |
|------|------|----------|---------|-------------|-----------|------|
| **1** | 注意力输出→wo_a | BF16→FP8 | per-128-group | UE8M0 | `2^ceil(log2(amax/448))` | `fused_inv_rope_fp8_quant` |
| **2** | MegaMoE 输入 | BF16→FP8 | per-32-subgroup→128-group | E8M0(int32) | round_to_e8m0(amax/448) | `prepare_megamoe_inputs` |
| **3** | FusedMoE FP8(W8A8) | BF16→FP8 | per-token | float32 | amax[t]/448 (精确) | `fused_moe` kernel 内部 |
| **4** | FusedMoE NVFP4(W4A4) | BF16→FP4 | per-16-group | float32 | 动态 per-group | NVFP4 kernel 内部 |

### 8.4 四种粒度如何各自解决 Outlier——数值案例

假设有一段 512 维的激活，维度 32 处有一个 outlier 值 50，其余 511 个维度在 [-1, 1] 范围内。

#### Per-tensor（仅作对比，DeepSeek V4 不使用）

```
整个 512 维共享 1 个 scale
amax = 50 → scale = 50/448 ≈ 0.112
正常值: 0.5 / 0.112 ≈ 4.5 → FP8 = ~4    (理想值应 ~224)
→ 511 个正常值全部被压扁，信息几乎全丢
scope of damage: 511 / 512 = 99.8% 的正常值被污染
```

#### 阶段 1：per-128-group（wo_a）

```
512 维 → 4 个 128 维 block，每 block 独立 scale
Block 0 (dims 0-127):   包含 outlier 52.7 → amax=52.7 → 128 个值被污染
Block 1-3 (dims 128-511): amax≈1 → scale≈0.0022 → 正常量化 ✓
scope of damage: 1/4 = 25% 维度受影响

额外保护: scale = 2^ceil(log2(amax/448))
  → scale 向上舍入到 2 的幂，确保不会因舍入误差漏掉 outlier
  → 代价：正常值也被略微压低，但远好于被污染
```

#### 阶段 2：per-32-subgroup（MegaMoE）

```
512 维 → 4 个 128-block → 每个 block 内有 4 个 32-subgroup
Subgroup (dims 32-63):  包含 outlier 52.7 → 32 个值被污染
其余 15 个 subgroup:     amax≈1 → 正常量化 ✓
scope of damage: 1/16 = 6.25% 维度受影响

关键设计: 嵌套分组
  128-block: 外层，用于 TMA 对齐和显存布局
  32-subgroup: 内层，实际量化粒度
  → 对齐开销低 (128) 但 outlier 影响范围窄 (32)
```

#### 阶段 3：per-token（FusedMoE FP8）

```
每个 token 沿 hidden 维度取一个 scale
Token A 维度 32 有 outlier: amax[A] = 52.7
  → Token A 的 512 维全部受影响
Token B 无 outlier: amax[B] = 1
  → Token B 完全不受影响 ✓
Token C 无 outlier: amax[C] = 1
  → Token C 完全不受影响 ✓

scope of damage for Token A: 512/512 = 100% (but only 1 token)
scope of damage across batch: 1/n_tokens

额外优势: scale = exact float32 (amax/448)
  → 无舍入，scale 精度最高，outlier 影响最小
  → 但 scale 在 kernel 内计算，不写回显存
```

#### 阶段 4：per-16-group + per-tensor input_scale（NVFP4 W4A4）

NVFP4 W4A4 的激活量化有两层叠加，注意不要混淆：

```
第一层 — kernel 内动态 per-16-group 量化（真正的量化粒度）:
  每 16 元素一组，实时计算 amax → scale → round_to_fp4
  scale 在 kernel 内计算后留在寄存器，不经过 TMA/显存

第二层 — 静态 per-tensor input_scale（全局校正系数）:
  每专家 1 个 FP32，存储在 checkpoint 中
  作用: 补偿跨专家的全局激活分布偏差
  
反量化时两者合并: act = dequant(act_fp4 × per_group_scale × input_scale)
```

数值案例：
```
512 维 → 32 个 16 维 group，每 group 独立 scale
Group 2 (dims 32-47): 包含 outlier 52.7 → 16 个值被污染
Group 3 (dims 48-63): 若 outlier 的 scale 泄漏则可能受影响，否则正常
其余 30 个 group:      amax≈1 → 正常量化 ✓
scope of damage: 1-2/32 ≈ 3-6% 维度受影响

注意: FP4 只有 8 个正值，即使 scale 完美，数值精度也不及 FP8
  → 用最细粒度 (gs=16) + 三层 scaling 补偿 FP4 的精度不足
  → input_scale 是三层缩放中专门服务于激活侧的一层，权重侧有对应的 block_scale + global_scale
```

### 8.5 四阶段对比总结

| | 阶段 1 (wo_a) | 阶段 2 (MegaMoE) | 阶段 3 (FusedMoE FP8) | 阶段 4 (NVFP4 W4A4) |
|------|:--:|:--:|:--:|:--:|
| **粒度** | per-128 | per-32 | per-token | per-16 |
| **outlier 污染范围** | 128 维 | 32 维 | 1 token 的所有维 | 16 维 |
| **scale 精度** | 2 的幂 (较粗) | E8M0 (中等) | 精确 float32 (最高) | 精确 float32 |
| **scale 开销/元素** | 0.0625B (FP32) | 0.25B (int32/128) | 0 (暂存寄存器) | ~0.5B |
| **数据精度** | FP8 | FP8 | FP8 | FP4 |
| **优势** | 向上舍入保守 | 嵌套分组，对齐+精度兼顾 | 跨 token 完全隔离 | 最细粒度 |

### 8.6 互补防线

| 防线 | 机制 | 位置 |
|------|------|------|
| **源头抑制** | SwiGLU clamping 到 [-10, 10]，使 outlier 在产生之前就被消除 | 所有 FFN 层 |
| **粒度隔离** | per-group/per-token 量化，outlier 只污染其所在 group，不扩散 | 所有激活量化阶段 |
| **尺度保护** | amax 下限 eps (1e-4/1e-10)，防止全零块除零 | 所有量化 kernel |
| **保守舍入** (阶段 1) | scale 向上取整到 2 的幂，确保 outlier 不溢出 | wo_a |
| **嵌套分组** (阶段 2) | 128 对齐存储 + 32 实际量化粒度，兼得带宽效率与精度 | MegaMoE |
| **跨 token 隔离** (阶段 3) | per-token scale 使 outlier token 完全不影响 batch 内其他 token | FusedMoE FP8 |

### 8.7 DeepSeek V4 没有用的技术

**SmoothQuant（per-channel 迁移）：** 不将激活 outlier 通过 per-channel scale "迁移" 到权重侧。

原因推测：
- RMSNorm 已使激活分布在各层前保持稳定
- SwiGLU clamping 在源头消除了大部分 outlier
- per-group scale 粒度已够处理残存的局部波动
- 不做 SmoothQuant 避免了权重侧引入额外的 per-channel scale 复杂度

---

## 9. 为什么不需要 AWQ 等 PTQ 时代的补丁

AWQ（Activation-aware Weight Quantization）等权重量化补偿技术曾是 PTQ 场景下的标配，但 DeepSeek V4 没有采用。本节解释为什么。

### 9.1 AWQ 解决什么问题

AWQ 的核心洞察：

```
GEMM: Y = X × W

某些 W 的通道与 X 中量级大的激活维度相乘 → 该通道的权重量化误差被显著放大
→ 这些 "salient channels" 需要更精准的 scale
→ AWQ: 根据激活的统计量（per-channel magnitude）给每个 W 通道分配不同的 scale 或保护位宽
```

本质：**观察激活分布 → 补偿 PTQ 下权重量化的精度损失**。

### 9.2 DeepSeek V4 为什么不需要

#### 原因 1：QAT 替代 PTQ，从根上消除了补丁的必要性

```
AWQ 路线:   BF16 训练 → PTQ round 到低精度（出现显著误差）
              → AWQ 用激活统计补偿误差
              → 问题：round 是硬性信息丢失，补偿只能减轻不能消除

DeepSeek:   BF16 训练 → QAT 模拟量化继续训练
              → 权重本身已经在量化约束下优化到收敛
              → 问题：不需要补偿，因为训练目标直接包含了量化
```

PTQ 是"训练完了再压缩"，QAT 是"带着量化约束继续训练"。前者需要各种启发式补丁弥补 round 误差，后者直接优化了量化后的损失函数。

#### 原因 2：per-group 粒度碾压 per-channel

| 方案 | 缩放粒度 | 每 scale 覆盖元素数 (K=4096) |
|------|---------|--------------------------|
| AWQ per-channel | 1 个输出通道 × 所有 K 维 | 4096 |
| DeepSeek per-32-group | 每 32 个连续 K 元素 | **32**（细 128×） |
| DeepSeek NVFP4 | 每 16 个连续 K 元素 | **16**（细 256×） |

AWQ 试图用单一 per-channel scale 补偿整个输入通道的量化误差，而同一通道内 4096 个权重值的量级可能差异很大。per-32-group 每组只有 32 个元素，**组内数值分布本身已经足够均匀**，不存在 AWQ 面对的那种"一个 scale 覆盖跨度太大的分布"的问题。

#### 原因 3：SwiGLU clamping 把 outlier 在源头掐掉了

AWQ 需要"激活感知"的根本动机：某些激活维度上存在 outlier，大激活乘以权重量化误差后被放大。DeepSeek 的 SwiGLU clamping 直接限制了激活范围：

```
有 outlier（AWQ 的动机场景）:
  x = [0.3, 52.7, 1.2, 0.5]  → 维度 1 的激活比邻域大 100×
  → 对应权重通道如量化有误差，会被 52.7× 放大
  → AWQ 需要给该通道更高精度的 scale

SwiGLU clamping 后:
  x = [0.3, 10.0, 1.2, 0.5]  → 最大差异降至 33×
  → per-32-group 的粒度完全能处理这个跨度的差异
  → 不再需要 AWQ 式的激活感知补偿
```

#### 原因 4：NVFP4 的三层 scale 本身就是比 AWQ 更强的机制

| | AWQ | NVFP4 |
|------|-----|-------|
| 粒度 | per-channel (1D, 粗) | per-16-group (1D, 细 256×) |
| Scale 精度 | 静态统计值 | QAT 中学习（FP8 E4M3 scale） |
| 多层校正 | 单层 | 三层：block + global + input |
| 何时确定 | 训练后，一次性 | 训练中，与权重共同优化 |

NVFP4 的 block_scale（per-16-group, FP8 E4M3）、global_scale（per-tensor, FP32）、input_scale（per-tensor, FP32）三层体系，等价于把 AWQ 要**事后补偿**的东西**纳入训练优化**了。每层的 scale 都在 QAT 中与权重共同学习收敛，而非训练结束后的一次性统计。

### 9.3 总结

| | AWQ 的假设 | DeepSeek V4 的现实 |
|------|-----------|-------------------|
| **何时量化** | 训练后 (PTQ) | 训练中 (QAT) |
| **量化粒度** | per-channel（K 维聚合） | per-32/16-group（比 AWQ 细 128-256×） |
| **激活分布** | 有 outlier，需要感知补偿 | SwiGLU clamp 在源头抑制 outlier |
| **scale 来源** | 静态统计，一次性 | QAT 中学习，与权重共同优化 |

**AWQ 是 PTQ 时代的精巧补丁。DeepSeek V4 用 QAT + 细粒度 + 源头 clamp 从根上消除了这个补丁的动机。**

---

## 10. 未来量化技术发展路径

### 10.1 基于现有趋势的预测（保守，有明确证据支撑）

#### 1. FP4 成为推理标配，FP8 退为训练格式

DeepSeek V4 证明了 FP4 QAT 在万亿参数 MoE 上可行（损失 <2%）。这条技术路线的扩散路径已经清晰：前沿实验室的下一代模型将默认采用 FP4 QAT，FP8 逐步退为训练专用格式。时间线预计 1-2 年内。

#### 2. 原生 FP4 硬件成为准入门槛

Blackwell 的 FP4 tensor core 只是开始。DeepSeek V4 论文明确暗示 FP4×FP8 GEMM 在"未来硬件上可以比 FP8×FP8 效率提升 1/3"。后续 B300、MI400、昇腾 960 必然原生支持 FP4/FP8 混合精度。硬件-软件协同设计从"锦上添花"变成"必要条件"。

#### 3. QAT 成为标准管线，PTQ 退居边缘

历史规律：FP16 混合精度训练从"可选优化"变成"默认方式"，纯 FP32 训练几乎消失。低比特时代会重复这个剧本——QAT 将在 1-2 年内从"前沿实验室的技术"变成"开源框架的标准功能"（如 PyTorch 原生 FP4 QAT API），PTQ 退居个人部署和 hobbyist 场景。

#### 4. 量化粒度继续细化，scale 格式持续进化

| 代际 | 典型粒度 | Scale 格式 | 存储开销 |
|------|---------|-----------|---------|
| 2023 (LLaMA-FP8) | per-tensor / per-channel | FP32 | ~0% / <1% |
| 2024 (DeepSeek-V3) | per-128 | FP32 / E8M0 | 3.1% / 0.8% |
| 2025 (DeepSeek-V4) | per-32 / per-16 | E8M0 / FP8 E4M3 | 6.3% |
| 2026-27 (预测) | per-8 / per-4 | 更紧凑的指数格式 | 12-25% |

约束条件：当 scale 开销达到有效比特数的 20-30% 时，进一步细化粒度将被 scale 存储成本抵消。临界点大约在 gs=4~8。

#### 5. 激活量化追平权重量化

当前方案大多是 W4A16 或 W4A8，激活侧量化比权重侧保守（FP8 vs FP4）。随着 SmoothQuant 类技术被 QAT 内化、训练稳定性技术成熟，W4A4 将不再是 NVFP4 独占，而成为通用方案。

#### 6. 量化覆盖范围从 MoE 扩展到全模型

DeepSeek V4 的 FP4 只覆盖了 MoE 专家权重和 indexer QK 路径。下一步将向 Attention 权重、Embedding 层、甚至所有缓存计算扩展。每扩展一个模块，意味着 QAT 稳定性需要新的技巧。

### 10.2 大胆预测（推测性，可能错误）

#### 1. FP2/FP3 推理可行，但只对"不重要的 80% 参数"

不是所有参数都需要 FP4。未来模型将出现**精度异构 2.0**——每个 layer、每个 expert、每个 block 都有不同的比特宽度，由训练中自动学习的"精度分配器"决定。高频使用的专家保持 FP4，低频专家降到 FP2/FP3。

#### 2. 可微分的量化网格——Learn The Grid Itself

当前的 E2M1/E4M3 格式是人为设计的固定网格。未来模型的量化网格本身将成为可学习参数——非均匀量化点由梯度下降优化，每个 layer 甚至每个 block 有自己的最优量化点集合。这比固定网格更接近"无损压缩"的极限。

#### 3. 从 FP4 QAT 到 FP4 Native Training

当前路径：BF16 预训练 → FP4 QAT。但如果 QAT 足够稳定，未来的模型可以直接在 FP4 精度下训练大部分 pre-training token。这需要解决 FP4 下的优化器收敛问题（FP4 的 8 个值做 momentum 更新极难），但混合精度训练的历史告诉我们：曾经被认为不可能的事情，配上正确的工程技巧后会变成标准实践。

#### 4. 稀疏性 + 低精度收敛为单一范式

当前稀疏性（2:4 pruning、MoE routing）和低精度（FP4/FP8）是两个独立领域。未来它们将在格式层面融合——一个统一表示同时描述"哪些值是零"和"非零值用多少 bit 表示"。structurally sparse FP4 可能是下一代 GEMM 的基本操作数。

#### 5. KV Cache 降到 2-bit

当前 FP8 KV Cache。随着 attention 对 softmax 的精度敏感性被更深入理解、"重要性感知量化"（长距离 KV 可以比近距离 KV 更激进的量化）被系统化，2-bit 甚至 1.5-bit KV Cache 将出现，使百万 token 上下文在单卡上运行成为可能。

#### 6. 神经量化器——NeuQuant

不再用 `amax/448` 这种硬编码公式计算 scale。一个小型神经网络（可能是几层 MLP）**根据输入分布实时预测最优量化参数**。这个量化器本身在 QAT 中与被量化的模型共同训练。

```
当前:    act → amax → scale = amax/fp8_max → quantize
未来:    act → NeuQuant(act_context) → per-element scale → quantize
```

#### 7. 量化从"精度损失补偿"变为"泛化增强手段"

已有证据表明低精度训练具有隐式正则化效果。未来这个视角可能反转：量化不是"为了节省显存不得已损失精度"，而是**主动的正则化工具**。低精度 backprop 的梯度噪声可以替代 dropout/weight decay。DeepSeek V4 论文中的 "underlying principles are not yet fully understood" 正是这种反转的前兆——我们还没理解量化为什么工作得这么好，当理解了，就可以主动利用它。

#### 8. 任意比特宽度的柔性硬件

固定 FP4/FP8/FP16 的硬件设计将被打破。下一代张量核心将支持**运行时配置比特宽度**——同一个 kernel 内不同区域可以动态切换精度。软件侧配合"精度调度器"，在延迟约束下自动分配每层的比特预算。

#### 9. Zero-Bit Experts——不计算就是最好的量化

MoE 中某些专家对于特定输入分布实际上是"沉睡"的。未来将出现**专家级稀疏性**——对于某个 batch，不仅路由不到某些专家（当前机制），而且主动识别并跳过那些"即使路由到也贡献接近零"的专家。量化到极致就是不算。

#### 10. 量化 + 推理时训练（TTT）+ RL 的三角融合

量化降低了推理的显存门槛。当推理成本足够低时，**推理时微调**（test-time training）变得可行。模型在推理过程中根据输入动态调整量化策略甚至少量权重。RL rollout 阶段的"原生 FP4 权重"可以演化为"FP4 权重 + 推理时在线微调"。这将模糊训练和推理的边界——不是训练完了再量化推理，而是推理本身就是轻量训练。

---
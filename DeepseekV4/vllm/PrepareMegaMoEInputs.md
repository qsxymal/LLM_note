# prepare_megamoe_inputs —— MegaMoE 输入 FP8 量化 Triton 内核

**文件路径：** `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`
**所属模块族：** `nvidia/ops/` — NVIDIA SM100 算力族
**内核函数：** `_prepare_megamoe_inputs_kernel`（`L15-L115`，Triton JIT）
**Host 封装：** `prepare_megamoe_inputs`（`L118-L173`）

## 1. 模块定位

- **职责：** 将 MegaMoE 的输入 hidden states 量化为 FP8 E4M3（附带 E8M0 group scales），并将路由结果（`topk_ids`、`topk_weights`）重排为 DeepGEMM `fp8_fp4_mega_moe` 内核需要的布局。
- **上游调用者：** `DeepseekV4MegaMoEExperts._run_mega_moe`（`model.py:L384-L392`），在执行 DeepGEMM 之前调用。
- **下游消费者：** `deep_gemm.fp8_fp4_mega_moe`，写入到 `symm_buffer.x` 和 `symm_buffer.x_sf`。

## 2. 调用点

```python
# model.py:L384-L392
prepare_megamoe_inputs(
    hidden_states,      # BF16 输入
    topk_weights,       # 路由权重
    topk_ids,           # 路由索引
    symm_buffer.x,      # FP8 输出缓冲区
    symm_buffer.x_sf,   # E8M0 缩放因子输出
    symm_buffer.topk_idx,      # int64 重排后的路由索引
    symm_buffer.topk_weights,  # float32 重排后的路由权重
)
```

`symm_buffer` 由 `deep_gemm.get_symm_buffer_for_mega_moe` 分配（`model.py:L334`），包含预先分配的 FP8 输入缓冲区、缩放因子缓冲区和重排后的路由元数据。

## 3. 输入输出规格

### 主机函数 `prepare_megamoe_inputs`

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `hidden_states` | `(num_tokens, hidden_size)` | bf16 | MoE 门控输出，即当前 token 的激活值 |
| `topk_weights` | `(num_tokens, top_k)` | bf16 | 路由权重，每个 token 选出的 top-k 专家的权重 |
| `topk_ids` | `(num_tokens, top_k)` | int64 | 路由索引，每个 token 选出的 top-k 专家的 ID |
| `x_fp8` 输出 | `(num_tokens, hidden_size)` | float8_e4m3 | FP8 量化后的 hidden states，供 DeepGEMM 消费 |
| `x_sf` 输出 | `(num_tokens, hidden_size//128)` | int32 | 打包的 E8M0 缩放因子（每 128 元素一个，4 字节打包 4 个） |
| `topk_idx_out` 输出 | `(num_tokens, top_k)` | int64 | DeepGEMM 布局的 top-k 索引 |
| `topk_weights_out` 输出 | `(num_tokens, top_k)` | float32 | DeepGEMM 布局的 top-k 权重 |

### 约束条件

- `hidden_size % 128 == 0`（L130-L134），否则抛出 `ValueError`。
- `topk_weights.shape == topk_ids.shape`（L136-L139）。
- `top_k` 必须是 32 的倍数吗？不需要，`BLOCK_TOPK = next_power_of_2(top_k)`（L144）自动向上取整到 2 的幂。

## 4. Triton 内核流水线

```python
@triton.jit
def _prepare_megamoe_inputs_kernel(hidden_states, x_fp8, x_sf, topk_ids, topk_weights,
                                   topk_idx_out, topk_weights_out, ...):
```

**Grid 配置**（L143）：
```python
grid = (num_tokens, triton.cdiv(hidden_size, 128))
```
二维 grid：`token_id × hidden_block_id`，每 block 处理 128 个 hidden 维度。

### 阶段 1：加载 + FP8 量化（L49-L83）

```python
# 1. 加载 BF16 hidden states（每 block 128 元素）
hidden = tl.load(hidden_states + token_id * stride_m + k_offsets * stride_k, ...).to(tl.float32)

# 2. 分组求 amax（GROUP_K=32，每组 4 个元素）
num_groups = BLOCK_K // GROUP_K  # = 128 / 32 = 4
hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
amax = tl.max(hidden_groups, axis=1)
amax = tl.maximum(amax, 1e-4)  # 防除零

# 3. 计算 FP8 缩放因子：amax / 448.0（FP8 E4M3 最大可表示值）
scale = amax / 448.0

# 4. 将 float32 scale 舍入到最近的 E8M0 纯指数格式
#    E8M0 = 无尾数的 8 位指数：提取指数部分，调整范围 [1, 254]
scale_bits = scale.to(tl.uint32, bitcast=True)
scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(tl.uint32)
scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

# 5. 量化到 FP8 E4M3
scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
fp8 = scaled.to(tl.float8e4nv)
tl.store(x_fp8 + ..., fp8, ...)

# 6. 将 4 个 E8M0 指数打包到一个 int32 中（每字节一个指数）
packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
tl.store(x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k, packed_scale)
```

**量化公式**：
```
scale = max(|x|) / 448.0          # 448 = FP8 E4M3 最大值
scale_e8m0 = round_to_e8m0(scale) # 舍入到最近的 E8M0 指数
x_fp8 = x / scale_e8m0            # 缩放到 FP8 范围
x_sf  = pack_e8m0(scale_e8m0)     # 4 个 E8M0 打包为 1 个 int32
```

> **为什么除以 448？** FP8 E4M3 的最大可表示值为 448（`2^(4-1) × (1 + 3/8) = 8 × 1.375 = 448`）。`amax / 448` 确保量化后的值不超过 FP8 的表示范围。

> **E8M0 舍入逻辑（L60-L66）**：scale 的 float32 尾数非零即表示 scale 在两个可表示的 E8M0 值之间，需要向上取整（`+1`）。最终范围限制在 `[1, 254]`，0 和 255 被排除（分别表示 0 和无穷）。

### 阶段 2：路由元数据重排（L85-L115）

```python
if k_block_id == 0:  # 仅在 hidden 的第一个 block 执行一次
    # 加载 topk_ids，转换为 int64
    ids = tl.load(topk_ids + token_id * stride + ...).to(tl.int64)
    tl.store(topk_idx_out + ..., ids, ...)

    # 加载 topk_weights，保持为 float32
    weights = tl.load(topk_weights + ...)
    tl.store(topk_weights_out + ..., weights, ...)
```

仅在 `k_block_id == 0` 时执行（每个 token 一次），避免重复写入。将 BF16 topk_weights 提升到 float32（DeepGEMM 要求的精度）。

## 5. 与 DeepGEMM 的数据流

```
hidden_states (bf16)
    │
    ▼
prepare_megamoe_inputs (Triton kernel)
    │
    ├─ x_fp8 (float8_e4m3) → symm_buffer.x
    ├─ x_sf  (int32 packed E8M0) → symm_buffer.x_sf
    ├─ topk_idx_out (int64) → symm_buffer.topk_idx
    └─ topk_weights_out (float32) → symm_buffer.topk_weights
    │
    ▼
deep_gemm.fp8_fp4_mega_moe (DeepGEMM kernel)
    │
    └─ y (bf16)
```

`symm_buffer` 在 `DeepseekV4MegaMoEExperts.get_symm_buffer`（`model.py:L318-L343`）中分配，包含对称通信所需的额外元数据缓冲区。写入到 `symm_buffer` 的 `x[:num_tokens]` 等切片，而非完整缓冲区，以支持变长 batch。

## 6. 边界场景

- **`num_tokens == 0`**（L128）：直接返回，跳过空批次的推理。
- **`hidden_size` 非 128 倍数**（L130-L134）：显式错误，因为 DeepGEMM 要求 hidden_size 对齐到 128。
- **`top_k` 非 2 的幂**（L144）：`triton.next_power_of_2(top_k)` 自动填充，Triton 的 mask 机制保证超出的位置不会影响计算。
- **极小的激活值**（L58）：`amax = max(amax, 1e-4)` 避免除以 0 导致的无穷大缩放因子。

## 相关笔记

- [[SparseAttnCompress]]：同 `nvidia/ops/` 目录的另一 Triton/CuteDSL 内核，共享 FP8 量化模式
- [[DeepseekV4MoE]]：MegaMoE 路径的整体流程和 `_run_mega_moe` 的调用链
- [[DeepseekV4MegaMoEExperts]]：`get_symm_buffer` 和 `finalize_weights` 的配套逻辑
- [[QuantAndParallelStrategy]]：MegaMoE 路径的量化策略（FP4 权重 + FP8 激活）
- [[DeepseekV4FP8Config]]：`expert_dtype` 控制是否启用 FP4 专家

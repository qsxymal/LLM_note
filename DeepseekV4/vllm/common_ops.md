# common/ops -- 跨平台融合操作算子

## 概览

`vllm/models/deepseek_v4/common/ops/` 目录包含跨平台（NVIDIA + AMD）共享的 Triton 算子。
通过 `__init__.py` 导出 8 个函数，供 NVIDIA 和 AMD 两端的注意力/压缩逻辑消费。

当前目录下的文件：

- `cache_utils.py` -- KV cache 量化/去量化与全局 top-k 索引
- `fused_compress_quant_cache.py`
- `fused_indexer_q.py` -- 带 RoPE + 量化 (MXFP4) 的索引器 Q 融合
- `fused_inv_rope_fp8_quant.py` -- 逆 RoPE + FP8 量化融合
- `fused_qk_rmsnorm.py` -- Q 和 KV 的 RMSNorm 融合 kernel（本笔记）
- `save_partial_states.py` -- 向压缩器状态缓存写入 partial states（本笔记）

---

## fused_q_kv_rmsnorm

**文件：** `vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py`

**调用者：** `vllm/models/deepseek_v4/attention.py` 第 423 行，在 `DeepseekV4Attention.forward()` 中对 `qr` 和 `kv` 分别做 RMSNorm（第 422 行先通过 `qr_kv.split([q_lora_rank, head_dim], dim=-1)` 切分 GEMM 输出）。

### 功能

将 Query 和 KV 各自的 RMSNorm 融合为一次 kernel launch。输入 `qr` 和 `kv` 共享同一 batch 维度（token 数相同），分别做 RMSNorm 后写出结果。

### 输入/输出

| 符号 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `qr` | `(T, Q_SIZE)` | bf16/f16 | Q 输入 |
| `kv` | `(T, KV_SIZE)` | bf16/f16 | KV 输入 |
| `q_weight` | `(Q_SIZE,)` | bf16/f16 | Q 的 RMSNorm weight |
| `kv_weight` | `(KV_SIZE,)` | bf16/f16 | KV 的 RMSNorm weight |
| `eps` | scalar | float32 | RMSNorm epsilon |
| `qr_out` | `(T, Q_SIZE)` | 同输入 | Q 输出 |
| `kv_out` | `(T, KV_SIZE)` | 同输入 | KV 输出 |

### Kernel 设计

**Grid 布局**（第 80 行）：`[(num_tokens, 2)]`

- `pid_task = tl.program_id(1)`：0 处理 Q，1 处理 KV（第 31-42 行）
- `token_idx = tl.program_id(0).to(tl.int64)`：处理第几个 token（第 30 行）

**为什么 `num_tokens` 放 grid-x 而不是 grid-y？（第 25-29 行注释）**

CUDA 的 grid-y 和 grid-z 维度有 65535 的上限。如果 grid 写成 `(2, num_tokens)`，当 `num_tokens >= 65536`（chunked prefill 的 typical 场景），launch 会失败并报 `Triton Error [CUDA]: invalid argument`。把 `num_tokens` 放在 grid-x 维度后其上限为 `2^31 - 1`。

**为什么用 `tl.int64`？（第 28-30 行）**

`q_in_stride` 可能是约 24K（例如 128 heads x 192）。`token_idx * stride` 用 `int32` 计算时，在 `num_tokens ~ 87K` 附近就会溢出。在大 chunked prefill 场景下 token 数可能远超此值，所以升格为 `int64`。

**计算流程（第 47-54 行）：**

```
block = tl.arange(0, BLOCK_SIZE)        # 连续地址，coalesced access
mask = block < SIZE
x = load(row_in + block, mask).to(fp32)  # 输入转 fp32
variance = sum(x * x) / SIZE             # 均方差
rrms = rsqrt(variance + eps)             # 倒数平方根
w = load(weight_ptr + block, mask).to(fp32)
y = x * rrms * w                         # fp32 计算
store(row_out + block, y.to(dtype))      # 最后转回输入 dtype 写出
```

**精度策略（第 44-46 行注释）：**

全程 fp32 计算——`x`, `rrms`, `w` 都在 fp32 域，仅在最后 store 时做一次 cast。这与 `csrc/layernorm_kernels.cu` 以及 DeepseekV4 压缩器 kernel 的表现一致：中间保持 fp32 精度，最终写出时回退到低精度。

**Block size（第 79 行）：**

`triton.next_power_of_2(max(Q_SIZE, KV_SIZE))`——用较宽的那个决定 block size，确保单个程序能覆盖全部元素。

### 边界情况

- **空 tensor（第 76-77 行）：** `num_tokens == 0` 时直接返回 `torch.empty_like` 的结果，不 launch kernel。

### 测试

对应测试文件 `tests/kernels/core/test_fused_q_kv_rmsnorm.py`：

- `test_fused_q_kv_rmsnorm_correctness`：参数化 `num_tokens in {1, 17, 1024, 8192}` + `dtype in {bf16, f16}`，与 PyTorch 参考实现比较
- `test_fused_q_kv_rmsnorm_launches_past_grid_y_cap`：参数化 `num_tokens in {65535, 65536, 131072}`，回归测试 grid-y 上限问题（第 54-58 行注释明确记录了此问题）

---

## save_partial_states

**文件：** `vllm/models/deepseek_v4/common/ops/save_partial_states.py`

**调用者：** `vllm/models/deepseek_v4/compressor.py` 第 314 行，属于 `DeepseekCompressor.forward()` 的 prologue 阶段。该 kernel 之后紧跟 fused compress->norm->RoPE->store kernel（`compress_norm_rope_store_triton` 或 cuTEDSL 变体）。

### 功能

将 [kv, score+ape] 的 partial states 写入压缩器的内部状态缓存（[[state_cache.md|state_cache]]）。每个 token 一个 program，padded token（`slot_mapping == -1`）跳过。

### 输入

| 符号 | 形状 | 说明 |
|------|------|------|
| `kv` | `(T, head_size)` | 待缓存的 KV 状态 |
| `score` | `(T, head_size)` | 待缓存的 score 状态 |
| `ape` | `(compress_ratio, head_size)` | 绝对位置编码（Absolute Position Embedding）查找表 |
| `positions` | `(T,)` | token 位置，int64 |
| `state_cache` | `(num_blocks, block_size, 2 * state_width)` | 压缩器内部状态缓存，最后一维 packed：[kv_state \| score_state] |
| `slot_mapping` | `(T,)` | slot_id 映射，-1 表示 padding |
| `block_size` | scalar | cache 的 block size |
| `state_width` | scalar | 单个 state 的宽度（`state_cache.shape[-1] // 2`） |
| `compress_ratio` | scalar | APE 查找用的压缩比 |

### Kernel 设计

**Grid 布局**（第 27 行）：`[(num_actual,)]`——每个 token 一个 program。

**地址计算（第 68-83 行）：**

```python
token_idx = tl.program_id(0)
slot_id = load(slot_mapping_ptr + token_idx)
if slot_id < 0:
    return                                # 跳过 padding

block_idx = slot_id // block_size
pos_in_block = slot_id % block_size
base_ptr = state_cache_ptr
           + block_idx * state_cache_stride0    # block 层 stride
           + pos_in_block * state_cache_stride1 # block 内 token 层 stride
```

**存储流程（第 86-101 行）：**

```
block = tl.arange(0, TRITON_BLOCK_SIZE)  # 连续地址
mask = block < HEAD_SIZE

# 前半段：写 kv
kv = load(kv_ptr + token_idx * kv_stride + block, mask)
store(base_ptr + block, kv, mask)

# 后半段：写 score + ape（融合）
position = load(positions_ptr + token_idx)
ape_row = position % COMPRESS_RATIO
ape = load(ape_ptr + ape_row * ape_stride + block, mask)
score = load(score_ptr + token_idx * score_stride + block, mask)
store(base_ptr + STATE_WIDTH + block, score + ape, mask)
```

**核心融合技巧（第 92-101 行）：**

`score += ape[position % compress_ratio]` 被内联进 store——一次访存加载 `ape`，就地完成加法，无需单独 launch element-wise kernel。`ape` 是一个循环查找表，其大小为 `compress_ratio`，索引为 `position % compress_ratio`。

### Design Decisions

- **PAD 跳过（第 71-76 行）：** 当 `slot_id == -1` 时立即 `return`。这是 vLLM 的 padding sentinel，CUDA graph replay 时 batch 中可能包含无效 token，若不跳过会写入非法地址 `kv_state[-1]`。
- **PDL 禁用（compressor.py 第 303-313 行注释）：** `pdl_kwargs` 中 `launch_pdl=False`。因为该 kernel 和其后继的压缩 kernel 都依赖前序 GEMM 的输出，但两者都没有发出/等待 PDL grid dependency，`launch_pdl=True` 会导致读后写竞争和非确定性输出。
- **State cache 布局（第 64-65 行 constexpr 注释）：** state cache 最后一维长度 `2 * state_width`，前 `state_width` 存 `kv_state`，后 `state_width` 存 `score_state`。`base_ptr + STATE_WIDTH + block` 就是 score 部分的起始地址。
- **Block size（第 41 行）：** `triton.next_power_of_2(head_size)` 保证一个 program 覆盖全部 head dim。

### 边界情况

- **Padding tokens（第 71-76 行）：** `slot_id < 0` 时跳过——这是保护 CUDA graph batch 中无效 slot 的唯一手段。
- **APE 索引越界保护：** `position % COMPRESS_RATIO` 保证 `ape_row` 永远落在 `[0, COMPRESS_RATIO)` 范围内。

# 多流并行机制

---

## 1. 概述

DeepSeek V4 的 MLA 注意力计算涉及多个可并行的 GEMM 和融合算子。为了最大化 GPU 利用率，vLLM 使用辅助 CUDA 流将独立计算重叠执行。

### 核心目标
将注意力计算中**不依赖彼此的计算**分配到不同 CUDA stream 上重叠执行，隐藏单个 kernel 的延迟。

### 适用条件
- 仅 CUDA 平台（NVIDIA GPU）
- 不适用于 ROCm / XPU（路径上会 hang）（`nvidia/model.py:L1102-L1106`）
- 可通过 `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` 环境变量控制启用阈值

---

## 2. Stream 配置

### 3 个辅助流（`nvidia/model.py:L1097-L1106`）

```python
aux_stream_list = (
    None if current_platform.is_rocm() or current_platform.is_xpu()
    else [torch.cuda.Stream() for _ in range(3)]
)
```

| 流索引 | 用途 | 创建位置 |
|--------|------|---------|
| 主流（默认） | `fused_wqa_wkv` GEMM + `wq_b` + KV insert | `torch.cuda.current_stream()` |
| 辅流 0 | Compressor `kv_score` GEMM | `aux_stream_list[0]` |
| 辅流 1 | Indexer `weights_proj` | `aux_stream_list[1]` |
| 辅流 2 | Indexer compressor `kv_score` | `aux_stream_list[2]` |

### 4 个同步事件（`attention.py:L224`）

```python
self.ln_events = [torch.cuda.Event() for _ in range(4)]
```

| 事件 | 角色 | 阶段 |
|------|------|------|
| `ln_events[0]` | GEMM phase fan-out（主流 → 所有辅流） | 阶段 1 |
| `ln_events[1]` | 辅流 0 完成通知 | 阶段 1 + 阶段 2 |
| `ln_events[2]` | 辅流 1 完成通知 | 阶段 1 |
| `ln_events[3]` | 辅流 2 完成通知 | 阶段 1 |

**注意**：`ln_events[1]` 在阶段 1 中作为辅流 0 的 done event，在阶段 2 中复用作辅流 0 的 start event（GEMM 完全 join 后才开始阶段 2）。

---

## 3. 两阶段重叠

MLA 的多流并行分为两个连续的阶段，分布在 `Wrapper.attn_gemm_parallel_execute` 和 `Wrapper.attention_impl` 中。

### 阶段 1：输入 GEMM 并行（`attention.py:L349-L407`）

```
               默认流                              辅流 0                       辅流 1                        辅流 2
               ──────                             ──────                       ──────                        ──────
               event0.record()
               fused_wqa_wkv()          [wait event0]                [wait event0]                 [wait event0]
                                         compressor_kv_score()        indexer_weights_proj()        indexer_cmpr_kv_score()
                                         event1.record()             event2.record()                event3.record()
               [wait events 1..3]
```

**执行内容**：

| 流 | 计算 | Kernel | 输出 shape | dtype |
|----|------|--------|-----------|-------|
| 默认 | `fused_wqa_wkv(hidden_states)` | GEMM (MergedColumnParallel) | `(B, q_lora_rank + head_dim)` | bf16 |
| 辅流 0 | `compressor.fused_wkv_wgate.weight @ hidden_states` | torch.mm | `(B, 2*coff*head_dim)` | **float32** |
| 辅流 1 | `indexer.weights_proj(hidden_states)` | GEMM (ReplicatedLinear) | `(B, n_head)` | bf16 |
| 辅流 2 | `indexer.compressor.fused_wkv_wgate.weight @ hidden_states` | torch.mm | `(B, 2*coff*head_dim)` | **float32** |

**关键设计**：
- `fused_wqa_wkv` 是最重的 GEMM（输出维度最大，~2048），放在默认流
- Compressor 和 indexer 的投影输出维度小，适合放在辅流
- Compressor 的 `kv_score` 输出为 float32（`out_dtype=torch.float32`），避免精度损失影响后续 softmax

### 阶段 2：后处理并行（`attention.py:L409-L495`）

取决于 `self.indexer is not None` 和 `self.compressor is not None`，有三种分支。

**分支 A：有 Indexer + 有 Compressor（C4A 层）**（`attention.py:L435-L468`）

```
默认流                              辅流 0                       辅流 1
──────                              ──────                       ──────
event0.record()
wq_b_kv_insert()         [wait event0]                   [wait event0]
                         indexer.forward():               compressor(kv_score):
                           ┌─ wq_b                         └─ save_partial_states
                           ├─ fused_indexer_q_rope_quant       → compress_norm_rope_store
                           └─ SparseAttnIndexer
                         event1.record()                  event2.record()
[wait event1]
[wait event2]
↓ mla_attn(q, kv, positions, output)
```

**为什么辅流 0 同时用于 indexer 和 compressor 的**：
阶段 1 完成后，`ln_events[1]` 已经被记录（辅流 0 完成 `compressor_kv_score`），阶段 2 中辅流 0 执行 indexer，辅流 1 执行 compressor。这是安全的，因为事件 `[0]` 只在阶段 2 的 `execute_in_parallel` 中重新记录。

**辅流 2 在阶段 2 中空闲**：`aux_stream_list[2]` 在 `attention_impl` 中被复用作 C4A Indexer 内部子重叠（`attention.py:L786-L788`），即 Indexer 内部的 `wq_b_and_q_quant` vs `compressor` 的并行。

**分支 B：无 Indexer + 有 Compressor（C128A 层）**（`attention.py:L469-L487`）

```
默认流                              辅流 0
──────                              ──────
event0.record()
wq_b_kv_insert()         [wait event0]
                         compressor(kv_score)
                         event1.record()
[wait event1]
↓ mla_attn(q, kv, positions, output)
```

**分支 C：无 Indexer + 无 Compressor（SWA-only 层）**（`attention.py:L488-L495`）

无重叠，串行执行。

---

## 4. 并行工具函数

### `execute_in_parallel`（`multi_stream_utils.py:L60-L128`）

N 个辅流的通用并行框架：

```python
def execute_in_parallel(
    default_fn,               # 主流执行函数
    aux_fns,                  # 辅流执行函数列表（可含 None）
    start_event,              # fan-out 事件
    done_events,              # 每个辅流的完成事件
    aux_streams,              # 辅流列表
    enable,                   # 控制是否启用
) -> (default_result, [aux_results]):
```

**实现逻辑**：
1. 如果 `aux_streams is None` 或 `enable=False`：顺序执行（先 default_fn，再 aux_fns 顺序）
2. 否则：
   - `start_event.record()`：在当前流记录
   - 对每个非 None 的 aux_fn，切到对应辅流，等待 `start_event`，执行 `aux_fn`，记录 `done_event`
   - 切回默认流执行 `default_fn`
   - 等待所有 `done_event`

**为什么先启动辅流再执行主流**：避免主流先完成但辅流还没启动导致的空闲。

### `maybe_execute_in_parallel`（`multi_stream_utils.py:L20-L58`）

2 个函数的简化版本：

```python
def maybe_execute_in_parallel(
    fn0, fn1, event0, event1, aux_stream=None
) -> (fn0_result, fn1_result):
```

逻辑相同，但通过 `aux_stream is None` 控制启用，无需 `enable` 参数。

---

## 5. 控制策略

### Token 阈值控制

`attn_gemm_parallel_execute` 中的 `enable` 条件（`attention.py:L403-L405`）：

```python
enable = hidden_states.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
```

**设计意图**：
- 小 batch 时，多流的 kernel 启动开销（event 记录/等待、stream 切换）可能超过 overlap 收益
- 大 batch 时，GPU 已被主流充分占用，辅流的 overlap 空间有限
- 中等 batch 时，多流最有效

**默认阈值**：`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` 环境变量控制。设置 0 强制始终启用。

### ROCm 回退

ROCm 平台上：
- `aux_stream_list = None`（`nvidia/model.py:L1102-L1106`）
- `execute_in_parallel` 中的 `enable` 无效（`aux_streams is None` 导致顺序执行）
- 原因：ROCm 的 stream 同步存在已知的 hang 问题

---

## 6. 重叠深度分析

### 理论重叠收益

假设各 kernel 的执行时间：
- `T_fused_wqa_wkv` ~ 100μs（主流，不可隐藏）
- `T_compressor_kv_score` ~ 20μs（辅流 0，可隐藏）
- `T_indexer_weights` ~ 15μs（辅流 1，可隐藏）
- `T_indexer_kv_score` ~ 15μs（辅流 2，可隐藏）

无多流：`T_total = 100 + 20 + 15 + 15 = 150μs`
有多流：`T_total = 100 + max(20, 15, 15) = 120μs`
节省：~20%

**实际收益取决于**：
- batch size（影响 GEMM 计算量）
- GPU 型号（SM 数量影响并行度）
- hidden_size 配置

### 第二层重叠

阶段 2 的 `wq_b_kv_insert` 和 compressor 的重叠更微妙：
- `wq_b_kv_insert` = `wq_b` GEMM + `fused_qnorm_rope_kv_insert`（融合 op）
- `compressor.forward` = `save_partial_states` + `compress_norm_rope_store`
- 两者在计算图上无数据依赖，但在 KV cache 写入上有**隐式依赖**（使用不同的缓存区域）
- 通过 CUDA event 保证写入顺序正确

---

## 7. 与其他系统的对比

| 系统 | 多流策略 | 辅流数 | 同步机制 |
|------|---------|--------|---------|
| DeepSeek V4 vLLM | 两阶段 GEMM + 后处理并行 | 3 | CUDA Event |
| TRT-LLM (PR #14142) | Level 1: 类似设计 | 2+ | CUDA Event |
| FasterTransformer | 仅 encoder/decoder 分离 | 1 | NCCL |
| vLLM 标准模型 | 无显式多流（除通信） | 0 | — |

DeepSeek V4 的多流设计参考了 TRT-LLM PR #14142 的 Level 1 方案（`attention.py:L449` 注释提及）。

---

## 8. CUDA Graph 兼容性

多流并行的 CUDA Graph 兼容需要特别注意：

**问题**：CUDA Graph 捕获期间，辅流上的操作也可能被捕获。如果 event 地址在重放时变化，会导致错误。

**解决方案**（`attention.py:L224`）：
```python
self.ln_events = [torch.cuda.Event() for _ in range(4)]
```
Event 在 `__init__` 中创建，地址固定。`record()` 和 `wait()` 是对同一 event 实例的操作，Graph 重放时仍有效。

**预分配缓冲区**：`symm_buffer`、`topk_indices_buffer` 等也在 `__init__` 中预分配，确保地址稳定。

---

## 9. 不启用多流的场景

| 场景 | 原因 |
|------|------|
| ROCm/XPU 平台 | 已知 stream 同步 hang 问题 |
| 小 batch（token 数 < 阈值） | 多流开销超过收益 |
| SWA-only 层（无 compressor） | 无可并行的独立计算 |
| 预热 dummy run | `attn_metadata is None` 跳过所有计算 |

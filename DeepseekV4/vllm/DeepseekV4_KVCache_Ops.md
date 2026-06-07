# DeepseekV4 KV 缓存算子

**文件路径：** `vllm/models/deepseek_v4/common/ops/`
**涉及文件：** `fused_compress_quant_cache.py`（FusedCompressQuantCache 节）、`cache_utils.py`（CacheUtils 节）
**所属模块族：** [[common_ops]] 跨平台融合操作算子

## FusedCompressQuantCache

**来源：** `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`

融合 Triton 内核，将**压缩器 -> RMSNorm -> RoPE -> 量化 -> 插入 KV 缓存**链接在单个内核启动中，避免了通过全局内存的冗余内存往返。

### 选择器函数

`compress_norm_rope_store_triton()`（第 31 行）根据 `head_dim` 和 `use_fp4_cache` 分派到三个内核之一：

| head_dim | use_fp4_cache | 内核 | num_warps |
|----------|---------------|--------|-----------|
| 512 | 不适用 | `_fused_kv_compress_norm_rope_insert_sparse_attn`（第 113 行） | 4 |
| 128 | False | `_fused_kv_compress_norm_rope_insert_indexer_attn`（第 303 行） | 1 |
| 128 | True | `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`（第 480 行） | 1 |

三个内核共享相同的启动签名；选择器传递 `**pdl_kwargs` 用于持久动态启动元数据（第 105 行）。

### 程序模型：每 token 一个程序

每个内核启动 `num_actual` 个程序（每 token 一个）。每个内核中的第一个操作是提前退出守卫（例如第 163 行）：

```
if (position + 1) % COMPRESS_RATIO != 0:
    return
```

非边界位置（不满足压缩条件的位置）立即退出，因此只有 `1/COMPRESS_RATIO` 的 token 执行实际工作。

第二个守卫检查 `slot_mapping[token_idx] >= 0`（第 159 行）以跳过填充 token。

### 压缩器

压缩步骤收集状态缓存条目，计算滑动窗口上的 softmax，并生成加权和。三个内核使用相同的逻辑：

- **窗口起始**（第 169 行）：
  ```
  start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
  ```
  总共收集 `(1+OVERLAP)*COMPRESS_RATIO` 个 token（过去的滑动窗口 + 当前 token）。

- **KV 槽查找**（第 174-193 行）：对于每个收集的位置，block_table 将 `(req_idx, position // block_size)` 转换为物理块号；然后 `(block_number, block_offset, head_offset)` 索引到状态缓存中。

- **分数加载和 softmax**（第 198-203 行）：分数值从状态缓存在偏移量 `STATE_WIDTH` 处读取（每条目宽度的一半）。沿 `dim=0`（跨收集的 token）应用 softmax，产生 `[TRITON_BLOCK_SIZE]` 个权重。

- **加权和**（第 211 行）：
  ```
  compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32
  ```

### RMSNorm

全程以 fp32 应用（第 213-217 行）：

```python
variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
rrms = tl.rsqrt(variance + rms_norm_eps)
normed = compressed_kv * rrms * rms_w
```

### RoPE

三个内核都实现了 **GPT-J**（交错）旋转位置嵌入。MXFP4 变体（第 629 行）跳过最后的 `tl.interleave`，因为下游 MXFP4 量化器自然地在偶数/奇数半部上操作。

通用方法（以稀疏注意力为例，第 275-296 行）：
1. `tl.reshape(normed, (NUM_PAIRS, 2))` — 将偶数/奇数元素配对。
2. `tl.split(...)` — 分割为 `even`、`odd` 向量。
3. `cs_idx = pair_idx - NOPE_PAIRS` — 只有维度中的 rope 半部分使用 cos/sin。
4. 在 `compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO` 处从 `cos_sin_cache` 加载 `cos`、`sin`。
5. `new_even = even * cos_v - odd * sin_v`，`new_odd = odd * cos_v + even * sin_v`。
6. `tl.interleave(new_even, new_odd)` — 恢复原始交错布局。

bf16 往返（第 242 行，第 457 行，第 631 行）`normed.to(tl.bfloat16).to(tl.float32)` 用于与参考内核 / Q 侧内核的数值一致性。

### 量化和存储到分页 KV 缓存

**缓存块布局**（三个内核相同）：

```
[0 : bs*token_stride)           token 数据（fp8 字节或打包的半字节）
[bs*token_stride : +bs*scale_dim]  缩放因子（uint8 UE8M0 指数或 float32）
```

**KV 缓存指针计算**（第 220-232 行，第 412-424 行，第 591-603 行）：
- `kv_slot_mapping[token_idx]` 给出扁平槽索引。
- `kv_block_idx = slot // kv_cache_block_size`，`kv_pos_in_block = slot % kv_cache_block_size`。
- `cache_block_ptr = k_cache_ptr + kv_block_idx * KV_BLOCK_STRIDE`。
- `fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE`。
- `scale_ptr = cache_block_ptr + bs*TOKEN_STRIDE + kv_pos_in_block * SCALE_DIM`。

#### 内核 1：稀疏注意力（head=512）

| 参数 | 值 | 定义行 |
|-----------|-------|--------------|
| HEAD_SIZE | 512 | 第 93 行 |
| NOPE_HEAD_DIM | 448（= HEAD_SIZE - ROPE_HEAD_DIM） | 第 234 行 |
| ROPE_HEAD_DIM | 64 | 第 141 行 |
| HALF_ROPE | 32 | 第 235 行 |
| TOKEN_STRIDE | 576（= 448 fp8 + 128 bf16） | 第 144 行 |
| SCALE_DIM | 8（7 个实际 + 1 个填充） | 第 145 行 |
| QUANT_BLOCK | 64 | 第 143 行 |
| N_QUANT_BLOCKS | 8（= TRITON_BLOCK_SIZE // QUANT_BLOCK） | 第 238 行 |
| N_NOPE_BLOCKS | 7（= NOPE_HEAD_DIM // QUANT_BLOCK） | 第 239 行 |

UE8M0 量化（第 242-269 行）：
1. `normed.to(tl.bfloat16).to(tl.float32)` — 用于一致性的 bf16 往返。
2. `tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))` — 平铺为寄存器级块。
3. 每块 absmax -> `scale = 2^ceil(log2(absmax / 448.0))`。
4. `x_scaled = x / scale`，限制在 `[-448, 448]`，`to(tl.float8e4nv)`，按位转换为 `uint8`。
5. 存储 fp8 数据：`tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)`。
6. 存储缩放因子：指数 + 127（UE8M0 编码），最后一个槽 `[7]` 零填充。
7. RoPE 部分作为 bf16 通过 `tl.pointer_type(tl.bfloat16)` 存储在 `fp8_ptr + NOPE_HEAD_DIM` 处。

#### 内核 2：Indexer FP8（head=128）

| 参数 | 值 | 定义行 |
|-----------|-------|--------------|
| HEAD_SIZE | 128 | 第 326 行 |
| TOKEN_STRIDE | 128 | 第 335 行 |
| SCALE_DIM | 4（= 1 个 float32） | 第 336 行 |
| QUANT_BLOCK | 128 | 第 334 行 |

因为 `TRITON_BLOCK_SIZE == QUANT_BLOCK`，所以使用单个扁平的 `tl.max` 归约（第 458 行），跳过 `[N_QUANT_BLOCKS, QUANT_BLOCK]` 重塑。缩放因子存储为单个 `float32` 值（第 472-473 行），而不是 UE8M0。

#### 内核 3：Indexer MXFP4（head=128）

| 参数 | 值 | 定义行 |
|-----------|-------|--------------|
| HEAD_SIZE | 128 | 第 503 行 |
| TOKEN_STRIDE | 64（= HEAD_SIZE // 2，打包的半字节） | 第 511 行 |
| SCALE_DIM | 4（= HEAD_SIZE // QUANT_BLOCK，ue8m0 字节） | 第 512 行 |
| QUANT_BLOCK | 32（MXFP4 块大小） | 第 510 行 |
| HALF_BLOCK | 16（= QUANT_BLOCK // 2） | 第 638 行 |
| N_QUANT_BLOCKS | 4（= HEAD_SIZE // QUANT_BLOCK） | 第 637 行 |

MXFP4 格式：
- E2M1 4 位值，每字节打包两个（先是低半字节，然后是高半字节）。
- 每 32 元素块的缩放因子 = `2^ceil(log2(amax / 6.0))`，存储为 ue8m0 字节（= 指数 + 127）。
- 最大可表示量级 = 6.0。

RoPE 输出保持为偶数/奇数半部（没有 `tl.interleave`），分别平铺成 `(N_QUANT_BLOCKS, HALF_BLOCK)` 用于偶数和奇数（第 644-645 行）。成对计算 Absmax（第 647-650 行）。通过 `_fp32x2_to_fp4x2` 打包（第 660 行，定义在 `fused_indexer_q.py` 第 29 行），其使用 `cvt.rn.satfinite.e2m1x2.f32` 内联 PTX。

### 交叉引用

- [[#quantize_and_insert_k_cache]] 使用相同的 UE8M0 方案，但作为一个独立内核（无压缩）。计算 `exponent = ceil(log2(block_max / FP8_MAX))` 完全相同（cache_utils.py 第 108 行 vs fused_compress_quant_cache.py 第 249 行）。
- [[#dequantize_and_gather_k_cache]] 是逆操作：加载 fp8 + UE8M0 缩放因子，通过 `x_fp8 * 2^(stored - 127)` 去量化回 bf16。
- [[#combine_topk_swa_indices]] 消费此内核生成的压缩 token。

---

## CacheUtils

**来源：** `vllm/models/deepseek_v4/common/ops/cache_utils.py`

四个导出的 Triton 内核，用于分页 K 缓存管理和稀疏注意力索引准备。所有内核都操作 [[#FusedCompressQuantCache]] 中定义的相同 KV 缓存块布局。

---

### quantize_and_insert_k_cache

**Python 包装器：** 第 142 行 | **Triton 内核：** `quantize_and_insert_k_kernel` 第 24 行

**目的：** 将 bf16 K 张量量化为 UE8M0 FP8 并写入分页 KV 缓存。来自 [[#FusedCompressQuantCache]] 的量化和存储阶段的独立版本，用于当压缩未被融合时（或用于离线/备选路径）。

**张量形状：**

| 张量 | 形状 | Dtype |
|--------|-------|-------|
| `k` | `(num_tokens, 512)` | bf16 |
| `k_cache` | `(num_blocks, block_bytes)` | uint8 |
| `slot_mapping` | `(num_tokens,)` | int64 |

**K 缓存块布局（block_size=64 token）：**

```
偏移范围                          内容
[0, 64*576)                           Token 数据：每个 token = 448 fp8 + 128 bf16
[64*576, 64*576 + 64*8)               缩放因子：每个 token = 8 uint8（7 个实际 + 1 个填充）
[64*576 + 64*8, block_stride)         填充
```

**常量**（第 170-175 行）：

| 常量 | 值 | 含义 |
|----------|-------|---------|
| TOKEN_FP8_DIM | 448 | 量化为 FP8 的 nope 维度 |
| TOKEN_BF16_DIM | 64 | 存储为 bf16 的 rope 维度 |
| TOKEN_SCALE_DIM | 8 | 每 token 的 UE8M0 缩放因子字节数（7 个实际 + 1 个填充） |
| QUANT_BLOCK_SIZE | 64 | 每个量化块的元素数 |
| FP8_MAX | 448.0 | fp8e4nv 的最大可表示值 |
| TOKEN_DATA_SIZE | 576（= 448 + 64*2） | 每 token 的总数据字节数 |
| n_quant_blocks | 8 | 64 元素量化块的数量 |

**算法**（第 87-139 行）：

```
对于 8 个量化块中的每一个（qblock_idx in tl.static_range(8)）：
    加载此块的 bf16[offsets]
    block_max = max(abs(x))
    block_max = max(block_max, 1e-4)           # 匹配 CUDA: fmaxf(amax, 1e-4)
    exponent = ceil(log2(block_max / FP8_MAX))  # UE8M0: 缩放因子是 2 的幂
    scale = 2^exponent
    x_scaled = x / scale
    x_clamped = clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(float8e4nv)
    x_uint8 = bitcast(x_fp8, uint8)            # 存储为原始字节
    store(x_uint8, fp8_ptr + offsets)
    store(uint8(exponent + 127), scale_ptr + qblock_idx)  # UE8M0 编码

在 scale_ptr[7] 处存储 0                          # 填充缩放因子

对于 4 个 bf16 块中的每一个（16 元素每个）：
    加载 bf16[fp8_dim + chunk_offsets]           # rope 部分
    在 bf16_out_ptr + chunk_offsets 处存储 bf16  # 直接存储
```

第 123-124 行展示了 UE8M0 编码：`stored = exponent + 127`，限制在 `[0, 255]`。去量化时：`scale = 2^(stored - 127)`。

**关键设计说明：**
- 使用 `block_idx.to(tl.int64) * block_stride`（第 71 行），因为 `block_stride ~37K * 57K+ blocks` 可能超过 `2^31`。
- 使用 DP 时，`slot_mapping.shape[0]` 可能小于 `k.shape[0]` 由于填充；token 计数始终来自 `slot_mapping`（第 167 行）。

---

### dequantize_and_gather_k_cache

**Python 包装器：** 第 353 行 | **Triton 内核：** `_dequantize_and_gather_k_kernel` 第 198 行 | **CuteDSL 备选：** 第 369 行

**目的：** `quantize_and_insert_k_cache` 的逆操作。从分页缓存中收集 FP8 K，去量化为 bf16，用于稀疏/SWA 预填充注意力。

**张量形状：**

| 张量 | 形状 | Dtype |
|--------|-------|-------|
| `out` | `(num_reqs, max_num_tokens, 512)` | bf16（输出） |
| `k_cache` | `(num_blocks, block_bytes)` | uint8 |
| `seq_lens` | `(num_reqs,)` | int32 |
| `gather_lens` | `(num_reqs,)` 或 `None` | int32 |
| `block_table` | `(num_reqs, max_blocks_per_seq)` | int32 |

**启动配置：** `(num_reqs, 128)` 网格（第 330 行）——128 个工作者协作处理每个请求的 token。

**算法**（第 232-304 行）：

```
对于分配给此工作者的每个 token（步长 = num_workers）：
    pos = (seq_len - gather_len) + i         # 序列中的位置
    block_in_seq = pos // cache_block_size
    physical_block = block_table[batch_idx, block_in_seq]
    
    cache_block_ptr = k_cache + physical_block * block_stride
    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
    token_scale_ptr = cache_block_ptr + bs*token_data_size + pos_in_block * scale_dim
    
    对于 7 个量化块中的每一个：
        加载 uint8（fp8 原始字节）
        x_fp8 = float8e4nv(bitcast uint8)
        x_float = fp32(x_fp8)
        encoded_scale = load(scale_ptr + qblock_idx)
        exponent = float32(encoded_scale) - 127.0
        scale = 2^exponent
        
        x_dequant = x_float * scale          # bf16
        store(x_dequant, out[token_idx, offset])
    
    # bf16 部分：直接复制
    从缓存中偏移 fp8_dim 处加载 bf16，存储到输出的相同偏移处
```

**关键说明：**
- `gather_lens` 允许只收集最后的 `N` 个 token（用于 SWA）。当为 `None` 时，收集所有 `seq_len` 个 token（第 226-229 行）。
- `physical_block_idx.to(tl.int64) * block_stride`（第 246 行）防止 int32 溢出，与插入路径理由相同。
- `n_quant_blocks = 7`（无填充块）vs 插入内核中的 `8`（第 349 行 vs 第 193 行）。
- CuteDSL 备选（第 367-380 行）：当 `has_cutedsl()` 为 true 时，委托给 `dequantize_and_gather_k_cache_cutedsl`（NVIDIA 优化的 CUDA 实现）。

---

### compute_global_topk_indices_and_lens

**Python 包装器：** 第 383 行 | **Triton 内核：** `_compute_global_topk_indices_and_lens_kernel` 第 418 行

**目的：** 将本地 topk 索引（请求的逻辑序列内的索引）映射到全局 KV 缓存槽 ID（分页缓存中的物理位置）。将三个操作融合到一个内核中。

**张量形状：**

| 张量 | 形状 | Dtype |
|--------|-------|-------|
| `topk_indices` | `(T, topk)` | int32 |
| `token_to_req_indices` | `(T,)` | int32 |
| `block_table` | `(num_reqs, max_blocks_per_seq)` | int32 |
| `is_valid_token` | `(T,)` | int32 |
| `global_topk_indices` | `(T, topk)` | int32（输出） |
| `topk_lens` | `(T,)` | int32（输出） |

**算法**（第 441-465 行）：

```
对于每个 token（1 个程序）：
    对于最多 1024 个索引的每个块：
        local_idx = load(topk_indices[token, offset])
        is_valid = (local_idx >= 0)
        
        block_number = block_table[req_idx, local_idx // block_size]
        block_offset = local_idx % block_size
        slot_id = block_number * block_size + block_offset
        slot_id = where(is_valid, slot_id, -1)
        store(global_topk_indices[token, offset], slot_id)
        count += sum(is_valid)
    
    store(topk_lens[token], is_valid_token ? count : 0)
```

**设计说明：**
- `block_size` 参数与分页 KV 缓存块大小匹配（通常为 64）。
- 有效条目计数（第 436 行，第 462 行）：只有 `local_idx >= 0` 的条目才计入 `topk_lens`。
- 填充 token 被屏蔽为长度 0（第 465 行），确保下游 scatter 操作为空操作。
- 输出 `global_topk_indices` 使用 `-1` 作为无效槽的哨兵值。

---

### combine_topk_swa_indices

**Python 包装器：** 第 476 行 | **Triton 内核：** `_combine_topk_swa_indices_kernel` 第 525 行

**目的：** 将 topk（压缩）索引与 SWA（滑动窗口）索引连接成一个排序后的索引数组，用于 FlashMLA 稀疏预填充。

**张量形状：**

| 张量 | 形状 | Dtype |
|--------|-------|-------|
| `topk_indices` | `(T, topk)` | int32（本地索引） |
| `query_start_loc` | `(num_reqs+1,)` | int32 |
| `seq_lens` | `(num_reqs,)` | int32 |
| `gather_lens` | `(num_reqs,)` | int32 |
| `combined_indices` | `(T, P)` | int32（输出，P = 填充后） |
| `combined_lens` | `(T,)` | int32（输出） |

其中 `P = ceil((topk + window_size) / 128) * 128`（对齐到 `_SPARSE_PREFILL_TOPK_ALIGNMENT`，第 473 行）。

**启动配置：** `(num_reqs, 128)` 网格（第 505 行）——工作者迭代请求中的 token。

**算法**（第 559-594 行）：

```
对于每个请求（每个请求 1 个程序，128 个工作者）：
    base = query_start_loc[0]
    query_start = query_start_loc[batch_idx] - base
    query_end = query_start_loc[batch_idx+1] - base
    start_pos = seq_len - (query_end - query_start)
    gather_start = seq_len - gather_len

    对于分配给此工作者的每个 token：
        pos = start_pos + token_idx_in_query
        
        topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = min(pos + 1, WINDOW_SIZE)
        
        # 存储 topk 索引（偏移 M*batch_idx 用于全局索引）
        store(combined_indices[topk_offset], topk_indices + M*batch_idx)
        
        # 存储 SWA 索引
        swa_start = N + pos - swa_len + 1 - gather_start
        store(combined_indices[topk_len + swa_offset], M*batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start)
        
        store(combined_lens[token_idx], topk_len + swa_len)
```

**关键常量和设计：**

- `_SPARSE_PREFILL_TOPK_ALIGNMENT = 128`（第 473 行）：FlashMLA 稀疏预填充要求 `topk % 128 == 0`（参见第 468-472 行的注释）。组合输出使用 `-1` 哨兵值填充到此对齐。
- `PADDED_TOP_K`（第 519 行）= `next_power_of_2(topk)`，用于以对齐的块加载 topk 索引。
- `TOP_K=0`（第 516 行）用于仅 SWA 层，以将 topk 部分清零。
- `M * batch_idx` 和 `N` 是全局偏移常量（由调用者定义，作为张量值传入），将索引移位到全局 KV 缓存缓冲区的正确区域。
- `gather_start` 偏移（第 557 行）：收集缓冲区的 SWA 部分从 `seq_len - gather_len` 开始，而不是位置 0，因此此偏移调整了 SWA 索引的计算。

### 交叉引用

- [[#FusedCompressQuantCache]] 定义了所有 CacheUtils 内核操作的 KV 缓存块布局。
- `quantize_and_insert_k_cache` 与融合内核的量化步骤共享 UE8M0 算法。`exponent = ceil(log2(block_max / FP8_MAX))` 计算在字节级别相同（cache_utils.py:108 vs fused_compress_quant_cache.py:249）。
- `dequantize_and_gather_k_cache` 是 [[#FusedCompressQuantCache]] 中量化步骤的逆操作。
- `compute_global_topk_indices_and_lens` 生成 `combine_topk_swa_indices` 消费的全局索引。
- `combine_topk_swa_indices` 馈送到 FlashMLA 稀疏预填充中。

### 共享设计元素

- **溢出保护：** `quantize_and_insert_k_cache`（第 71 行）和 `dequantize_and_gather_k_cache`（第 246 行）都对 `block_idx * block_stride` 乘法使用 `tl.int64`，因为当 KV 缓存包含 57K+ 块且步长为约 37K 字节时，乘积可能超过 2^31。
- **工作者并行性：** `dequantize_and_gather_k_cache` 和 `combine_topk_swa_indices` 都使用每个批次 128 个工作者以实现细粒度并行。
- **UE8M0 格式：** 缩放因子 = 2^(exponent)，存储为 uint8(exponent + 127)。编码为 `ceil(log2(block_max / FP8_MAX)) + 127`；解码为 `2^(stored - 127)`。

---

## 相关笔记

- [[common_ops]] — 同目录的跨平台算子总览
- [[FusedInvRopeFP8Quant]] — 同目录的逆 RoPE + FP8 量化融合算子
- [[FusedIndexerQ]] — 同目录的 Indexer Q 正向 RoPE + 量化融合算子（Triton 实现）
- [[DequantGatherKCache]] — `nvidia/ops/` 下的 CuteDSL 备选实现（功能等价，SM100 优化）
- [[SparseAttnCompress]] — `nvidia/ops/` 下的 CuteDSL 实现（与 Triton 版本 `sparse_attn_compress.py` 功能等价）
- [[DeepseekCompressor]] — 调用 `fused_compress_quant_cache` 中内核的上游压缩器模块

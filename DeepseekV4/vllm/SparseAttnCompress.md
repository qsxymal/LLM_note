# SparseAttnCompress —— 稀疏注意力 KV 压缩 CuTe DSL 内核族

**文件路径：** `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`
**公开函数：** `compress_norm_rope_store_cutedsl`（`L1244-L1332`）— 统一入口

## 1. 模块定位

- **职责：** 将稀疏注意力的 state cache（float32 格式的 KV 流聚合状态）压缩为 FP8 格式的 KV 缓存行，同时执行 RMSNorm 和 RoPE 变换。支持 C4（`compress_ratio=4`）和 C128（`compress_ratio=128`）两种压缩模式。
- **上游调用者：** `DeepseekV4IndexerCache` 的 `compress_kv_cache` 方法（在注意力 forward 中，每个 token 达到压缩边界时触发）。
- **与 Triton 的关系：** 此文件是 CuteDSL 实现，与 `vllm/v1/attention/ops/sparse_attn_compress.py` 的 Triton 实现功能等价，但性能更优（针对 SM100 优化）。

## 2. 统一入口

```python
def compress_norm_rope_store_cutedsl(
    state_cache: torch.Tensor,    # (num_blocks, block_size, 2*state_width) float32
    num_actual: int,              # 实际 token 数
    token_to_req_indices,         # (num_tokens,) int32
    positions: torch.Tensor,      # (num_tokens,) int64
    slot_mapping: torch.Tensor,   # (num_tokens,) int64
    block_table: torch.Tensor,    # (num_reqs, max_blocks) int32
    block_size: int,              # state cache 的 block 大小
    state_width: int,             # 状态宽度（= 2 * head_dim 对于 C4，= head_dim 对于 C128）
    cos_sin_cache: torch.Tensor,  # (max_pos, rope_dim) float32
    kv_cache: torch.Tensor,       # (num_blocks, block_size, 584) uint8 — 输出 FP8 KV 缓存
    k_cache_metadata,             # 元数据，包含 slot_mapping
    pdl_kwargs, ...
    compress_ratio: int,          # 4 (C4A) 或 128 (C128)
    overlap: bool,                # 使用重叠窗口（C4 使用，C128 不使用）
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,# (head_dim,) bf16/f32
    rms_norm_eps: float,
    quant_block: int = 64,
    token_stride: int = 576,
    scale_dim: int = 8,
) -> None
```

### 路由逻辑（`L1268-L1332`）

```python
def compress_norm_rope_store_cutedsl(...):
    if compress_ratio == 4:
        # C4A: 单个融合内核更快
        fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(...)
    else:
        # C128: 两个分离内核更快
        compressed_kv = torch.empty((num_actual, head_dim), float32)
        compress_kv_sparse_attn_cutedsl(...)     # 步骤 1: 压缩
        norm_rope_insert_sparse_attn_cutedsl(...) # 步骤 2: 归一化 + RoPE + 量化写入
```

## 3. 三类内核

### 3.1 `SparseAttnCompressKernel` — C128 压缩（`L463-L815`）

```
输入: state_cache (float32) — KV 流聚合状态
输出: compressed_kv (float32) — softmax 加权聚合结果

流程:
  1. 读取 window 个 position 的 KV 和 score
  2. 对 score 做 online softmax（safe softmax 算法）
  3. 输出加权平均的 KV 值
```

- Grid: `(num_tokens * num_splits, 1, 1)`，其中 `num_splits = head_dim // 64`
- 每个 threadblock 处理一个 token × 64 维的 split
- `head_tile=64`，`rows_per_warp=8`，`elems_per_lane=4`
- 使用 5 级 shuffle 的 warp reduce 做跨 warp 归约

### 3.2 `SparseAttnNormRopeStoreKernel` — C128 后处理（`L818-L1087`）

```
输入: compressed_kv (float32) — 压缩后的 KV
输出: kv_cache (uint8 FP8) — 写入 Paged KV 缓存

流程:
  1. RMSNorm(compressed_kv)
  2. RoPE: 对 rope 维应用旋转位置编码
  3. FP8 量化 + E8M0 scale 打包
  4. 写入 FP8 KV 缓存（584 字节/token 格式）
```

- NoPE 部分：FP8 E4M3 量化 + E8M0 per-block scales（每 64 元素）
- RoPE 部分：BF16 直接存储（不量化）
- 每 token stride = 576（`448 FP8 + 128 BF16 + 8 scale` → 为 576 对齐）

### 3.3 `SparseAttnCompressNormRopeStoreC4Kernel` — C4 融合（`L75-L460`）

```
输入: state_cache (float32) — KV 流聚合状态
输出: kv_cache (uint8 FP8) — 直接写入 Paged KV 缓存

功能: 对 C4 压缩，将压缩 + RMSNorm + RoPE + FP8 量化融合为一个内核
```

- `token_stride=576`，`scale_dim=8`，`quant_block=64`
- 与 C128 不同的是：C4 使用重叠窗口（`overlap=True`），窗口大小 = `(1+1)*4=8`

## 4. 压缩比对比

| 特性 | C4（compress_ratio=4） | C128（compress_ratio=128） |
|------|----------------------|--------------------------|
| 窗口大小 | 8（重叠） | 128（不重叠） |
| 内核模式 | 单融合内核 | 双分离内核 |
| state_width | 2 × head_dim = 1024 | head_dim = 512 |
| 输出格式 | 直接 FP8 KV 缓存 | 先 float32 再转换 |
| 适用层 | Indexer 层（C4A） | 非 Indexer 压缩层 |

## 5. 输出格式

所有路径最终输出到 `kv_cache`，格式为 `(num_blocks, block_size, 584)` 的 FP8 布局：

```
每行 584 字节:
  [0..447]    NoPE FP8 E4M3 (量化, block_size=64)
  [448..575]  RoPE BF16 (不量化)
  [576..583]  E8M0 scales (8 字节, 每 64 元素一个)
```

## Cross-References

- [[DeepseekV4Indexer]]：Indexer 中触发 KV 压缩的调用点
- [[DeepseekCompressor]]：Compressor 模块负责 KV 压缩和缓存管理
- [[QuantAndParallelStrategy]]：FP8 KV 缓存布局（584 字节/token）
- [[PrepareMegaMoEInputs]]：类似的 FP8 量化 + E8M0 scale 打包逻辑（Triton 实现）

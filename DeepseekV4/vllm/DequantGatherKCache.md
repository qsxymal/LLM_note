# DequantGatherKCache —— FP8 K 缓存反量化 + 聚集 CuTe DSL 内核

**文件路径：** `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`
**公开函数：** `dequantize_and_gather_k_cache_cutedsl`（`L17-L29`）

## 1. 定位

- **职责：** 将 FP8 格式的 K 缓存页反量化为 BF16，并聚集到连续的输出缓冲区中，供稀疏注意力 prefill 阶段使用。
- **上游调用者：** `vllm.models.deepseek_v4.common.ops.dequantize_and_gather_k_cache`（common 封装），在 `DeepseekV4FlashMLASparseImpl._forward_prefill` 的每个 chunk 循环中被调用（`flashmla.py:L373-L381`）。
- **处理的数据：** DeepSeek V4 的 FP8 K 缓存布局：每 token 448 字节 NoPE（FP8 E4M3）+ 64 字节 BF16（RoPE）+ 8 字节缩放因子 = 584 字节/行。

## 2. 公开 API

```python
def dequantize_and_gather_k_cache_cutedsl(
    out: torch.Tensor,          # (num_reqs, gather_len, head_dim=512) bf16 — 输出
    k_cache: torch.Tensor,      # (num_blocks, block_size, 584) uint8 — FP8 K 缓存
    seq_lens: torch.Tensor,     # (num_reqs,) int32 — 每个序列的完整长度
    gather_lens: torch.Tensor | None,  # (num_reqs,) int32 — 实际需要聚集的长度（可选）
    block_table: torch.Tensor,  # (num_reqs, max_blocks) int32 — 页表
    block_size: int,            # Paged KV cache 的 block 大小（通常 256）
    offset: int,                # 输出缓冲区的起始偏移
) -> None:
```

## 3. 输入输出

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `out` | `(R, G, 512)` | bf16 | 输出：反量化后的连续 K 张量 |
| `k_cache` | `(B, BS, 584)` | uint8 | FP8 K 缓存（584 = 448 NoPE + 128 RoPE + 8 scale） |
| `seq_lens` | `(R,)` | int32 | 各序列的 KV 长度 |
| `gather_lens` | `(R,)` | int32 | 可选：限制聚集范围（短于 seq_len） |
| `block_table` | `(R, MB)` | int32 | PagedAttention 页表 |
| `offset` | scalar | int | 写入 out 的起始位置 |

## 4. 计算流程

```
k_cache: [block_size=256, 584B] FP8 页 → 逐 token 反量化
  │
  ├─ 前 448 字节 = NoPE FP8 E4M3 → 读取 FP8 + E8M0 scale → 反量化为 BF16
  ├─ 后 128 字节 = RoPE BF16 → 直接复制为 BF16（无需反量化）
  └─ 最后 8 字节 = E8M0 缩放因子 → 按 group_size=64 解析
    
out: [gather_len, 512] BF16 连续输出
```

**内核细节**（`L32-L331`）：
- `head_dim=512`，`group_size=64`（每 64 元素共享一个 E8M0 scale）
- `fp8_dim=448`（前 448 维为 FP8），`bf16_dim=64`（后 64 维为 BF16 RoPE，实际上用 `bf16_dim * 2 = 128` 字节存储）
- 使用 `cp.async` 预取（4 stage），warp 粒度并行

**反量化数学**（`L253-L275`）：
```
scale_u8 = k_cache_scale[lane_id * 8 // 64]    # 每 group 一个 E8M0
scale_bf16 = (scale_u8 << 7) | (scale_u8 << 23) # E8M0 → BF16 魔数展开
bf16_val = fp8_val * scale_bf16                  # FP8 → BF16 反量化
```

## 5. 调用示例

```python
# 在 _forward_prefill 中（flashmla.py:L373-L381）
for chunk_idx in range(num_chunks):
    dequantize_and_gather_k_cache(
        kv[:chunk_size],          # BF16 输出缓冲区
        compressed_k_cache,       # FP8 K 缓存
        seq_lens=seq_lens // compress_ratio,
        gather_lens=None,
        block_table=block_table,
        block_size=attn_metadata.block_size // compress_ratio,
        offset=0,
    )
```

## Cross-References

- [[DeepseekV4FlashMLASparseImpl]]：在 `_forward_prefill` 中调用此内核
- [[QuantAndParallelStrategy]]：FP8 KV 缓存布局（584 字节/token）

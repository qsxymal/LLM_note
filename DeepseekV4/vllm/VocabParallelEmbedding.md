# VocabParallelEmbedding —— 词表并行嵌入

**文件路径：** `vllm/model_executor/layers/vocab_parallel_embedding.py`

## 1. 模块定位

- **职责：** 实现词表切分的嵌入层（ColumnParallel），每个 TP rank 持有词汇表的一个分片，避免整个词表被复制到所有 GPU。
- **上游调用者：** `DeepseekV4Model` 的 `embed_tokens` 层和 MTP 的 `embed_tokens` 层。
- **下游消费者：** `ParallelLMHead`（子类），`LogitsProcessor`（消费 `lm_head` 的输出）。

## 2. 词表并行策略

### 嵌入查表（`embedding` 模式）

- **分片方式：** 嵌入权重按词表维度切分。每个 rank 持有 `vocab_size // tp_size` 个 token 的嵌入向量。
- **前向传播：** `nn.Embedding` 的列并行——输入 token ID 直接查本地表，超出本地范围的 ID 查不到（应为零）。
- **TP 通信：** 前向传播中的 `all_reduce` 聚合各 rank 的部分嵌入（`EmbeddingAllReduce` 算子）。这使得每个 rank 的输出形状已经是 `[num_tokens, hidden_size]`，后续计算无需额外通信。
- **设计意图：** 嵌入层占参数量不大（`vocab_size × hidden_size`），但 TP 切分后可减少 HBM 带宽占用。对于大词表模型（如 128K+），节省显著。

### 逻辑回归（`lm_head` 模式，即 `ParallelLMHead`）

`ParallelLMHead` 继承 `VocabParallelEmbedding` 并设置 `quant_method` 为 `"apply"` 模式，此时：
- **前向传播：** 执行 `quant_method.apply`（FP8 量化矩阵乘）替代嵌入查表。
- **TP 通信：** `LogitsProcessor` 在 `apply` 后执行 `gather` 而非 `all_reduce`（因为词表切分后需要收集所有分片才能得到完整 logits）。
- **参见 [[ParallelLMHead]] 和 [[LogitsProcessor]]。**

## 3. DeepSeek V4 中的特殊处理

### MTP 共享嵌入

DeepSeek V4 的 MTP（Multi-Token Prediction）draft 模型使用独立的 `VocabParallelEmbedding` 实例（`mtp.py:L199-L203`）：

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    prefix=maybe_prefix(prefix, "embed_tokens"),
)
```

这个嵌入层与目标模型的 `embed_tokens` 是**独立的参数**（不是权重绑定），在 MTP checkpoint 中独立存储，通过 `_rewrite_spec_layer_name`（`mtp.py:L481`）将共享权重提升到顶层命名空间。

### 量化配置的传递

`VocabParallelEmbedding` 的 `quant_config` 来自 `VllmConfig`，最终由 [[DeepseekV4FP8Config]] 控制。嵌入层使用 `Fp8LinearMethod`（embedding 模式），而 `lm_head` 使用 `Fp8LinearMethod`（apply 模式），两者由 `quant_method` 属性区分。

## 4. 对外接口

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, ...):
        # 按 TP 切分词表
        self.weight = nn.Parameter(...)  # shape: [vocab_size // tp_size, hidden_size]

    def forward(self, input_ids):
        # 嵌入查表 + all_reduce
        return output  # [num_tokens, hidden_size]
```

## 5. 关键 tensor

| 变量 | shape | dtype | 说明 |
|------|-------|-------|------|
| `weight` | `(vocab_size // tp_size, hidden_size)` | bf16/fp8 | 切分后的嵌入矩阵 |
| `input_ids` | `(num_tokens,)` | int64 | 输入 token ID |
| output | `(num_tokens, hidden_size)` | bf16 | `all_reduce` 后的完整嵌入 |

## 6. 设计亮点

- **统一的分片基类**：同一份权重可以既作嵌入（查表）又作输出投影（`lm_head`），通过 `quant_method` 区分行为模式。
- **TP 通信隐藏**：`all_reduce` 被封装在 `forward` 中，上层（`DeepseekV4Model`）感知不到通信开销。
- **MTP 权重的命名隔离**：MTP 的 `embed_tokens` 通过 `prefix` 区分命名空间，与主模型权重互不干扰。

## Cross-References

- [[ParallelLMHead]]：子类，用于 `lm_head` 输出投影
- [[LogitsProcessor]]：消费 `ParallelLMHead` 的输出，执行 TP gather
- [[QuantAndParallelStrategy]]：`quant_method` 在嵌入层和 `lm_head` 中的不同角色（`embedding` vs `apply`），以及 TP 通信模式（all-reduce vs gather）对整体推理性能的影响。
- [[MTP]]：MTP 中独立的 `embed_tokens` 实例

**源码路径**: `vllm/model_executor/layers/vocab_parallel_embedding.py`  
**类定义**: `class ParallelLMHead(VocabParallelEmbedding)` — 第 510 行  
**注册**: `@PluggableLayer.register("parallel_lm_head")` — 第 509 行

下面详细讲解 `ParallelLMHead` 类的实现。该类继承自 `VocabParallelEmbedding`，但语义完全不同：它代表语言模型的**输出投影层**（通常称为 `lm_head`），负责将隐藏状态映射到词汇表大小的 logits。尽管与嵌入层形状相同（`[vocab_size, hidden_size]`），但它的权重矩阵是**按行切分**（与嵌入层一致），且支持可选的偏置项。

---

## 1. 类声明与注册

```python
@PluggableLayer.register("parallel_lm_head")
class ParallelLMHead(VocabParallelEmbedding):
```

- 通过 `@PluggableLayer.register` 注册为可插拔层，允许外部后端提供自定义实现。
- 继承 `VocabParallelEmbedding` 的目的是**复用其词汇表并行逻辑**（分片、填充、权重加载、TP 通信等），因为 `lm_head` 的权重矩阵与嵌入矩阵具有完全相同的形状和并行切分策略。

---

## 2. `__init__` 构造函数（第 528 行）

```python
def __init__(
    self,
    num_embeddings: int,          # 总词汇表大小（原始 + LoRA 添加）
    embedding_dim: int,           # 隐藏状态维度
    bias: bool = False,           # 是否使用偏置项
    params_dtype: torch.dtype | None = None,
    org_num_embeddings: int | None = None,  # 原始词汇表大小（不含 LoRA）
    padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
):
    super().__init__(...)
    self.quant_config = quant_config
    if bias:
        self.bias = Parameter(
            torch.empty(self.num_embeddings_per_partition, dtype=params_dtype)
        )
        set_weight_attrs(
            self.bias,
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )
    else:
        self.register_parameter("bias", None)
```

- 首先调用父类 `VocabParallelEmbedding` 的初始化，完成以下工作：
  - 计算填充后的词汇表大小 `num_embeddings_padded`。
  - 根据 TP size 计算当前 rank 的分片大小 `num_embeddings_per_partition`。
  - 创建 `shard_indices` 用于索引映射。
  - 通过量化方法创建权重张量（形状 `[num_embeddings_per_partition, embedding_dim]`）。
- **偏置处理**：
  - 如果 `bias=True`，则创建一个与本地分片大小相同的偏置参数（形状 `[num_embeddings_per_partition]`）。
  - 为该偏置设置权重加载属性：`output_dim=0` 表示偏置的切分维度也是词汇表维度，`weight_loader` 复用父类的 `weight_loader` 方法（该方法会自动按 `shard_indices` 从完整偏置张量中切取当前 rank 的部分）。
  - 如果 `bias=False`，则 `self.bias` 为 `None`。

**设计要点**：
- 偏置也按词汇表维度并行切分，每个 TP rank 只保存自己负责的那一段偏置值。
- 权重加载时，父类的 `weight_loader` 同样适用于偏置，因为偏置被标记了 `output_dim=0`。

---

## 3. `tie_weights` 方法——权重绑定（第 563 行）

```python
def tie_weights(self, embed_tokens: VocabParallelEmbedding):
    """Tie the weights with word embeddings."""
    # GGUF quantized embed_tokens.
    if self.quant_config and self.quant_config.get_name() == "gguf":
        return embed_tokens
    else:
        self.weight = embed_tokens.weight
        return self
```

- **权重绑定**是一种常见技巧：将 `lm_head` 的权重矩阵与输入嵌入层 `embed_tokens` 的权重共享，从而减少参数量（两者形状完全相同）。这在 GPT-2、LLaMA 等模型中常见。
- **GGUF 量化例外**：如果模型使用了 GGUF 量化，嵌入层可能使用了特殊的量化格式（例如将权重打包），此时不能直接共享 `weight` 引用，因为 `lm_head` 期望的权重布局可能不同。因此直接返回 `embed_tokens`，由调用方自行处理（通常意味着 `lm_head` 不会与嵌入层绑定，而是使用独立的权重）。
- 对于非 GGUF 情况，将 `self.weight` 直接指向 `embed_tokens.weight`（Python 引用赋值）。之后 `lm_head` 的前向计算将使用嵌入层的权重，共享同一份显存。

**设计思考：关于 weight binding 的深层含义**：
- **为什么需要 `tie_weights` 而非直接在 `__init__` 中绑定？** 因为权重的加载顺序是先加载嵌入层（`embed_tokens`），后加载 `lm_head`。如果在 `__init__` 时就绑定，此时嵌入层的权重可能还未加载（仍是随机初始化）。`tie_weights` 作为一个单独的方法，允许在所有权重加载完成后、模型前向计算之前被调用。
- **绑定后的权重共享语义**：`self.weight = embed_tokens.weight` 是 Python 的引用赋值，修改 `embed_tokens.weight` 会同时影响 `lm_head`，反之亦然。这对梯度计算和优化器更新有重要影响——如果进行微调，梯度会累积到嵌入层（而非 `lm_head`）的权重上。
- **量化兼容性**：即使 `lm_head` 自身的 `quant_config` 指定了某种量化方案，通过 `tie_weights` 共享权重后，实际使用的权重格式取决于嵌入层的量化配置。因此 vLLM 需要在模型定义时确保嵌入层和 `lm_head` 的量化方式兼容。
- **GGUF 的特殊性**：GGUF 格式将权重打包为特定布局，其 `quant_method` 可能不兼容嵌入层的查表操作（需要 `embedding` 方法）与 `lm_head` 的线性投影操作（需要 `apply` 方法）。因此 GGUF 下放弃绑定，让两者各自独立量化，确保每种操作都使用正确的内核。

**注意**：绑定后，`lm_head` 的 `weight` 属性不再是由量化方法创建的独立张量，而是指向嵌入层的权重。这在加载权重时需要特别小心（通常嵌入层先加载，然后 `tie_weights` 再建立引用）。

---

## 4. `forward` 方法——禁用（第 572 行）

```python
def forward(self, input_):
    del input_
    raise RuntimeError("LMHead's weights should be used in the sampler.")
```

- 该方法**故意不实现**，而是直接抛出 `RuntimeError`。
- **原因**：在 vLLM 的设计中，`ParallelLMHead` 仅作为权重容器存在，真正的 logits 计算发生在 `Sampler` 类中（[[LogitsProcessor]]）。`Sampler` 会直接访问 `lm_head.weight` 和 `lm_head.bias`（如果存在），并调用底层的线性运算（例如 `torch.matmul` 或量化的 `apply` 方法）。因此 `ParallelLMHead` 本身不需要 `forward`。
- 这样的设计分离了权重管理与计算逻辑，使得 `Sampler` 可以灵活选择计算方式（例如使用自定义内核、FP8 矩阵乘等），同时避免了在模型 forward 中误用 `lm_head`。

---

## 5. 与父类 `VocabParallelEmbedding` 的差异

| 特性 | `VocabParallelEmbedding` | `ParallelLMHead` |
|------|---------------------------|------------------|
| 用途 | 输入 token id → embedding | 隐藏状态 → logits |
| 前向调用 | `forward(input_ids)` 执行查表 | 禁用，`forward` 抛出异常 |
| 权重形状 | `[vocab_size_partition, hidden_size]` | 相同 |
| 偏置 | 不支持（嵌入层通常无偏置） | 支持可选偏置 |
| 权重绑定 | 不适用 | 提供 `tie_weights` 方法与嵌入层共享权重 |
| 量化兼容 | 通过 `quant_method.embedding` | 通过 `quant_method.apply`（线性层） |
| TP 通信 | all‑reduce（嵌入输出） | 不通信，由 `Sampler` 负责 gather |

尽管 `ParallelLMHead` 复用了父类的词汇表并行分片和权重加载逻辑，但它的**计算语义完全不同**。继承的目的是代码复用（避免重复实现分片索引、权重加载、填充处理等），而不是表示 is‑a 关系。

---

## 6. 使用场景与设计意义

在 vLLM 的推理流程中，`ParallelLMHead` 通常作为 `DeepseekV4ForCausalLM` 的一部分创建（仅在最后一个流水线并行 rank 上实例化）。权重加载后，可能会调用 `tie_weights` 与嵌入层绑定。随后在每次生成迭代中，`Sampler` 从 `forward_context` 中获取模型的隐藏状态，然后：

```python
logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=self.lm_head.bias)
logits = self.logits_processor._gather_logits(logits)
```

- `lm_head.quant_method.apply` 会使用 `lm_head.weight` 和 `lm_head.bias` 执行线性变换（支持量化）。
- 由于 `ParallelLMHead` 不提供 `forward`，所有对它的调用都应通过 `Sampler` 显式进行。

这种设计解耦了模型定义和采样逻辑，允许 `Sampler` 对 logits 进行高效的后处理（如温度缩放、top‑k 过滤、rejection sampling 等），同时避免了在模型 forward 中不必要地计算 logits（只在需要时计算）。

---

## 7. 总结

`ParallelLMHead` 是一个专门用于大模型 TP 推理的**输出投影权重容器**，它：

- 继承 `VocabParallelEmbedding` 复用词汇表并行分片和权重加载逻辑。
- 支持可选的偏置项，同样按 TP 分片。
- 提供 `tie_weights` 实现权重绑定，减少显存占用。
- **禁用 `forward` 方法**，强制使用者通过 `Sampler` 直接访问权重进行 logits 计算。
- 配合量化框架，支持 FP8 等低精度推理。

这种设计使得 vLLM 能够在保持模型结构清晰的同时，灵活地优化采样阶段的性能。

---

### Cross-References

- [[LogitsProcessor]]：`ParallelLMHead` 的实际消费者，使用 `quant_method.apply` 执行对数投影并完成 gather 通信。
- [[VocabParallelEmbedding]]：父类，提供词汇表并行分片、填充处理、权重加载等基础设施。
- [[QuantAndParallelStrategy]]：`quant_config` 与 `quant_method` 的联动，以及 `tie_weights` 在量化场景中遇到的特例（GGUF）。

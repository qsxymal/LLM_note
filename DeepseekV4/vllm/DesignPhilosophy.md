# 整体设计哲学

从 DeepSeek V4 在 vLLM 中的实现中提炼出的**设计哲学**——这些是贯穿整个代码库的深层原则。

---

## 原则一：分层抽象，每层一个关注点

### 表现
```
ForCausalLM          → 顶层外观（权重加载 + PP）
  Model              → 主干（嵌入、层列表、norm、HC 头）
    DecoderLayer     → HC 融合 + 子层编排
      Attention      → 组件工厂（结构声明）
        Wrapper      → 计算编排（多流 + 融合）
          MLAAttention → 核心抽象（接口）
            Impl     → 内核调度（硬件特定）
      MoE            → 混合专家（策略选择）
```

### 为什么这么做
- **可替换性**：每一层都可以在不影响其他层的情况下替换。例如换注意力后端只需要改 Impl 层
- **测试性**：可以单独验证每一层的输入输出契约
- **平台隔离**：`Impl` 层的平台选择在 `MLAAttention.__init__` 时一次性完成，上层无需关心

### 违反原则的地方（HACK）
- `static_forward_context` 全局字典（`attention.py:L261`）：被注释标注为 `# HACK`，因为 `torch.compile` 环境无法传递 Python 引用，只能通过字符串查找
- Compressor 通过 `k_cache_prefix` 字符串查找主 KV 缓存（`compressor.py:L277`）：弱耦合，但设计上更清晰的方案应该通过参数传递

---

## 原则二：计算与结构分离

### 表现
`DeepseekV4Attention` 只声明子模块（结构），不执行任何计算；`DeepseekV4MultiHeadLatentAttentionWrapper` 只编排计算，不声明新子模块。两者的桥梁是 `DeepseekV4MLAModules` dataclass。

### Indexer vs Compressor 的归属差异
- **Indexer 在 Attention**：因为它有自己的参数、KV 缓存、独立性，是模型结构的一部分
- **Compressor 在 Wrapper**：因为它是计算流程中的预处理步骤，与多流并行深度融合

### 为什么这么做
- 结构层的稳定性：模型架构变化时会修改 Attention 中的子模块，但不影响执行逻辑
- 计算层的灵活性：Wrapper 可以频繁优化（调整并行策略、添加融合算子），而不影响权重加载或缓存分配
- 可替换性：Wrapper 作为 `PluggableLayer`，不同平台可以提供完全不同的实现

---

## 原则三：融合优先

### 表现
代码中反复出现"将多个操作合并为一个 kernel"的模式：

| 融合名称 | 包含的操作 | 实现方式 |
|---------|-----------|---------|
| `mhc_fused_post_pre` | 上一子层 hc_post + 当前子层 hc_pre + norm | TileLang/Triton kernel |
| `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | Q head norm + RoPE + KV RoPE + FP8 quant + cache insert | C++ custom op |
| `fused_q_kv_rmsnorm` | Q RMSNorm + KV RMSNorm | Triton kernel |
| `fused_inv_rope_fp8_quant` | inverse RoPE + FP8 quantization | Triton kernel |
| `compress_norm_rope_store` | compress + RMSNorm + RoPE + FP8 quant + cache store | CuteDSL/Triton |
| `fused_indexer_q_rope_quant` | Q projection + RoPE + quantization | Triton kernel |

### 为什么这么做
- **减少 kernel launch 开销**：每个 kernel 启动有固定延迟（PCIe 提交、调度器唤醒），融合后启动次数减半
- **减少全局内存往返**：中间结果不写回 HBM，直接留在寄存器/SMEM 中传递
- **减少带宽消耗**：例如 `fused_qnorm_rope_kv_insert` 将 Q 的 post-norm 和 KV 的 pre-insert 合并，避免对 Q 的单独读写

### 代价
- **复杂度上升**：每个融合算子都是一个自定义 kernel，开发和调试成本高
- **灵活性下降**：融合后的算子不易拆解或修改内部某一步
- **可移植性受限**：融合 kernel 通常有硬件依赖性（如 CuteDSL 需要 SM90+）

---

## 原则四：延迟隐藏（通过多流并行）

### 表现
MLA 计算中的独立 GEMM 分配到不同 CUDA stream 上重叠执行。两阶段并行覆盖了：

1. 输入 GEMM 阶段：`fused_wqa_wkv`（主流）与 compressor/indexer 的投影（辅流）重叠
2. 后处理阶段：`wq_b` + KV insert（主流）与 indexer/compressor 的后处理（辅流）重叠

### 为什么这么做
- GPU 的 SM 资源可以被多个 kernel 共享（如果 SM 有空闲）
- 在主流 kernel （如 `fused_wqa_wkv` 大 GEMM）执行时，辅流的轻量 kernel 可以利用空闲 SM 完成，隐藏它们的延迟
- 多流并行的收益在中等 batch 时最大

### 代价
- **事件同步开销**：CUDA event 的 record/wait 加起来约 2-5μs 延迟
- **复杂性代价**：需要管理 stream 切换、事件等待顺序、race condition

详见 [[MultiStreamParallelism]]。

---

## 原则五：精度异构

### 表现
不同模块使用不同的数据类型：

| 模块 | 数据类型 | 原因 |
|------|---------|------|
| 激活流（hidden_states） | BF16 | 主流计算精度 |
| 注意力权重 | FP8 block-wise | 带宽友好，精度损失在注意力可接受范围 |
| KV Cache | FP8 E4M3 + BF16 RoPE | 节省 4× 存储（vs FP32），NoPE 部分量化 |
| Compressor State Cache | float32 | 需要高精度用于滑动窗口加权 |
| MoE 专家权重（MegaMoE） | FP4 + UE8M0 scale | 极致压缩，256 专家显存友好 |
| MoE 专家权重（FusedMoE） | FP4/FP8/BF16 | 灵活选择 |
| MoE 输入激活（MegaMoE） | FP8 E4M3 | 输入到 DeepGEMM 前动态量化 |
| HC 混合矩阵 | float32 | 混合权重精度要求高 |
| Routing logits | float32 | softmax 精度敏感 |

### 为什么这么做
- **存储带宽是瓶颈**：推理时显存带宽比计算更受限，量化可以显著减少数据移动
- **不同模块的精度敏感度不同**：注意力 QKV 投影对精度不太敏感（FP8 足够），但 compressor state cache 的 sliding window 计算需要高精度
- **"费力不费力"原则**：把精度放在需要的地方（routing、state cache），对带宽瓶颈的地方用低精度

### 量化点的设计权衡
- KV Cache 只量化 NoPE 部分，RoPE 部分保留 BF16：因为 RoPE 的频率信息对位置敏感，量化后位置区分度下降
- Compressor 内部用 float32 保持中间精度：因为滑动窗口压缩是累积计算，误差会放大
- MegaMoE 用 FP4 权重 + FP8 激活 + float32 内部：DeepGEMM 内部使用 float32 累加，外存带宽降 4× 但计算精度保持

---

## 原则六：CUDA Graph 优先设计

### 表现
几乎每个模块的初始化都在考虑 CUDA Graph 兼容性：

- **预分配缓冲区**：`topk_indices_buffer`、`_mtp_hidden_buffer`、`symm_buffer` 在 `__init__` 中分配，地址固定
- **Tile scheduler caching**：FlashMLA 的 tile 规划只在首次运行，结果缓存供后续图重放使用
- **`static_forward_context`**：通过 `layer_name` 字符串在编译图中定位模块实例
- **Custom op + fake impl**：为 `torch.compile` 提供 fake 实现，使编译图可以静态推断 shape
- **`non_blocking=True`**：元数据复制与计算重叠

### 为什么这么做
- CUDA Graph 消除了 host-side kernel launch 开销（每次 launch 约 5-50μs）
- 对于小 batch decode（每次只需要 launch 少量 kernel），graph 可以显著提升吞吐
- 但 CUDA Graph 要求：捕获期内所有内核参数地址不变、所有控制流不变
- 因此必须预分配所有缓冲区，缓存所有一次计算结果

详见 [[CUDAGraphIntegration]]。

---

## 原则七：平台隔离而非条件分支

### 表现
NVIDIA 和 AMD 的实现通过**导入时选择**隔离，而非运行时 `if is_rocm()` 条件：

```python
# __init__.py: 导入时选择
if not current_platform.is_rocm():
    from .nvidia.model import DeepseekV4ForCausalLM
else:
    from .amd.model import DeepseekV4ForCausalLM
```

- `nvidia/model.py` 和 `amd/model.py` 是完全独立的文件，没有共享代码
- 共享的逻辑放在 `common/` 目录（如 `common/ops/` 中的算子）
- 注意力后端通过 `_select_v4_sparse_impl()` 策略函数选择

### 为什么这么做
- **避免运行时分支**：运行时 `if is_rocm()` 会在每次 forward 时执行，但代码中两个分支只有一个是可达的
- **编译优化**：导入时选择后，JIT 编译器可以看到完整的计算图，进行更好的优化
- **依赖隔离**：NVIDIA 代码可能导入 CuteDSL（初始化 CUDA 驱动），AMD 平台不应该执行这些 import
- **静态类型检查**：mypy 至少在 TYPE_CHECKING 下能看到 NVIDIA 路径的完整类型

### 平台隔离的例外
- `DeepseekCompressor.forward` 中有少量运行时平台选择（`compressor.py:L338-L355`），因为 compressor 逻辑共享但底层 kernel 不同（CUDA 用 cutedsl，ROCm 用 Triton）
- `DeepseekV4DecoderLayer.forward` 中有 `is_rocm()` 分支（`nvidia/model.py:L1066-L1068`），选择融合或原生路径

---

## 原则八：Custom Op + PluggableLayer — 编译图边界

### 表现
关键计算路径封装为自定义 op：

```python
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
class DeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer): ...

torch.ops.vllm.deepseek_v4_attention(...)  # custom op
torch.ops.vllm.deepseek_v4_mega_moe_experts(...)  # custom op
torch.ops.vllm.deepseek_v4_fp8_einsum(...)  # custom op
```

### 为什么这么做
- **`torch.compile` 边界**：Custom op 是 `torch.compile` 的"原子操作"，编译图不会穿透 custom op 内部。这防止了 `torch.compile` 尝试追踪复杂的 Python 控制流
- **可替换的 PluggableLayer**：`PluggableLayer` 是 vLLM 提供的机制，允许不同后端注册自己的实现层。一个 OOT（out-of-tree）平台可以替换整个 MLA 层而无需修改模型代码
- **fake impl**：为 custom op 提供 fake 实现（如 `deepseek_v4_attention_fake`）：

```python
def deepseek_v4_attention_fake(...):
    return None
```
使 `torch.compile` 可以在跟踪模式下跳过实际计算，只推断 shape 和 dtype

---

## 九条原则的交互

```
┌──────────────────────────────────────────────────────────────┐
│                   整体设计哲学                                │
│                                                              │
│  1. 分层抽象 ──── 每一层一个关注点                            │
│       │                                                      │
│       ├── 2. 计算与结构分离 ──── Attention vs Wrapper        │
│       │                                                      │
│       ├── 3. 融合优先 ──── 减少 kernel launch + 带宽         │
│       │      │                                               │
│       │      └── 4. 延迟隐藏 ──── 多流并行 hide latency      │
│       │                                                      │
│       ├── 5. 精度异构 ──── 把精度给需要的地方                │
│       │                                                      │
│       ├── 6. CUDA Graph 优先 ──── 预分配 + 缓存              │
│       │                                                      │
│       ├── 7. 平台隔离 ──── 导入时选择而非条件分支            │
│       │                                                      │
│       └── 8. Custom Op ──── 编译图边界 + 可替换              │
│                                                              │
│  9. 工程务实主义 ──── HACK 存在但是有注释说明原因            │
└──────────────────────────────────────────────────────────────┘
```

## 附：代码中的 HACK 标注统计

这些是设计哲学未能完美贯彻的地方：

| 位置 | HACK 内容 | 原因 |
|------|----------|------|
| `attention.py:L261` | `# HACK` 标注 `self.layer_name` 的存储方式 | `torch.compile` 无法传递 Python 引用 |
| `attention.py:L790` | `compress_ratio` 的 0 值处理 | MTP 层不包含在 compress ratio list 中 |
| `nvidia/flashmla.py` | 违反 LSP 的 classmethod override | V4 注意力由层驱动而非 V1 框架驱动 |
| `compressor.py:L312` | PDL 禁用 | PDL grid 依赖原语与 compress kernel 不兼容 |

这些 HACK 的必要性源于**现有框架的限制**，而非设计失误。每个 HACK 都附有注释解释原因。

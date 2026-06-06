# 任务汇总

本文件汇总用户在整个分析过程中下达的所有指令/任务。

---

## 阶段零：初始任务

**来源：** `task.md`

### Task1 — 基础笔记生成

分析 `vllm/models/deepseek_v4/` 目录下的代码，补充 MoE 相关笔记，完善 `my_note/` 中现有的笔记。要求：
- 每行代码的说明
- 设计说明
- 每个模块的输入输出及其数据类型说明
- 量化和并行策略说明
- 思考
- 笔记格式为 Obsidian
- 现有笔记可修改，所有内容重新分析一遍，优化现有笔记
- 原笔记内容格式保持，在任务过程中完善优化，最终输出高质量文档

### Task2 — 笔记补充（保留原内容）

对已有笔记进行再补充，要求：
- 原笔记内容不要删减
- 在原笔记基础上新增内容
- 原内容与新增内容合并，以新的更优的排版写入原文件
- 先以 `my_note/DeepseekV4ForCausalLM.md` 为例开展工作，写入前需询问确认

### Task3 — 深度分析

在前两阶段基础上进行更深入的代码分析。

---

## 阶段一：基础优化

### 指令 1：中文文件重命名

> "好的。将中文文件根据内容重命名成英文文件"

将 `my_note/` 中中文命名的笔记文件根据内容重命名为英文名：
- 结构思考.md → `ArchitectureOverview.md`
- 量化和并行策略.md → `QuantAndParallelStrategy.md`

### 指令 2：Gap 分析 + 示例

> "好的。先思考下还能怎么补充；2、以某个笔记文件为例进行分析"

- 对现有笔记做缺口分析，找出未覆盖的模块
- 以某个笔记文件为例展示补充方式
- 批准后全面实施

### 指令 3：批准计划

> "好的，请继续"

批准了基于 Gap 分析的笔记补充计划。

### 指令 4：按顺序执行

> "依次进行"

要求按以下顺序依次执行所有补充任务：
1. `DeepseekV4ForCausalLM.md` / `DeepseekV4Model.md` — 模型入口
2. `DeepseekV4DecoderLayer.md` — 解码器层
3. `DeepseekV4MultiHeadLatentAttentionWrapper.md` — MLA 注意力
4. `load_weights.md` — 权重加载
5. `DeepseekV4FP8Config.md` + `DeepseekV4MoE.md` — 量化和 MoE
6. `MTP.md` + `VocabParallelEmbedding.md` — MTP 和嵌入
7. `QuantAndParallelStrategy.md` + `ArchitectureOverview.md` — 跨层专题
8. `NvidiaVsAMD.md` + `EndToEndDataFlow.md` — 平台对比和数据流
9. 5 个 CuteDSL 内核文件笔记
10. `LogitsProcessor.md` + `ParallelLMHead.md` — 输出头

### 指令 5：确认进度

> "当前是第几个? 还有四个: load_weights.md 那是第四个. 原来是分了两个阶段，现在在第1.5阶段"

确认当前执行进度。当时处于第 4/10 项（load_weights.md），属于阶段一和阶段二之间的过渡。

> "current is the 4th, 5 more items left"

再次确认进度，剩余 5 项待完成。

---

## 阶段二：深度分析

### 指令 6：询问能否更深入

> "其他模块还能不能再尽可能详细分析了？"

询问是否还有模块可以进一步深入分析，要求列出所有可能的深度分析方向。

### 指令 7：依次深入开展

> "依次开展"

要求对列出的所有未覆盖/可深入模块依次进行分析，包括：
1. `common/ops/` — 跨平台融合操作算子（6 个文件）
2. MHC 自定义算子层 — `mhc_pre` / `mhc_post` / `mhc_fused_post_pre`
3. AMD ROCm 后端 — `rocm.py`（856 行）
4. Gate routing 数学 — `fused_topk_bias` / `sqrtsoftplus` / `Noaux TC` / `Hash MoE`
5. Model Registration — 模型注册机制
6. V1 注意力后端 — `DeepseekV4SparseMLAAttentionImpl` / `FlashMLASparseBackend`
7. CUDA Graph 集成 — `static_forward_context` / `no_compile_layers` / 预热

---

## 输出产物总览

### 基础优化阶段（新写/重写）
| 笔记 | 说明 |
|------|------|
| `ArchitectureOverview.md` | 原名"结构思考.md"，架构概览 |
| `QuantAndParallelStrategy.md` | 原名"量化和并行策略.md"，量化与并行专题 |
| `NvidiaVsAMD.md` | NVIDIA vs AMD 平台对比 |
| `EndToEndDataFlow.md` | 端到端数据流追踪 |
| 5 个 CuteDSL 内核笔记 | `DequantGatherKCache.md`, `FusedIndexerQ.md` 等 |

### 深度分析阶段（新写）
| 笔记 | 说明 |
|------|------|
| `common_ops.md` | `fused_qk_rmsnorm` + `save_partial_states` |
| `FusedInvRopeFP8Quant.md` | 融合逆 RoPE + FP8 量化 |
| `FusedIndexerQ.md` | Indexer Q RoPE + 量化（重写，覆盖 Triton 路径） |
| `DeepseekV4_KVCache_Ops.md` | `fused_compress_quant_cache` + `cache_utils` |
| `MHCOps.md` | MHC 算子层四实现 |
| `GateRouting.md` | 门控路由数学 |
| `ROCmBackend.md` | AMD ROCm 后端 |
| `ModelRegistration.md` | 模型注册机制 |
| `V1AttentionBackend.md` | FlashMLA 稀疏后端 |
| `CUDAGraphIntegration.md` | CUDA 图集成 |

### 深度补充阶段（在原笔记上增强）
| 笔记 | 补充内容 |
|------|---------|
| `DeepseekV4ForCausalLM.md` | 模型入口完整分析 |
| `DeepseekV4Model.md` | 模型主体完整分析 |
| `DeepseekV4DecoderLayer.md` | 解码器层完整分析 |
| `DeepseekV4MultiHeadLatentAttentionWrapper.md` | MLA 注意力完整分析 |
| `load_weights.md` | 权重加载完整分析 |
| `DeepseekV4FP8Config.md` | FP8 配置完整分析 |
| `DeepseekV4MoE.md` | MoE 完整分析 |
| `MTP.md` | MTP 完整分析 |
| `VocabParallelEmbedding.md` | 嵌入完整分析 |
| `DeepseekV4MLAAttention.md` | MLA 注意力补充 |
| `DeepseekCompressor.md` | 压缩器补充 |
| `DeepseekV4Indexer.md` | Indexer 补充 |
| `attention.py.md` | 注意力代码补充 |
| 其他原笔记 | 按计划逐项补充完善 |

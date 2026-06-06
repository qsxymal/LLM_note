# DeepSeek V4 笔记导读

本文档说明所有笔记文件之间的关系及最佳阅读顺序。

```mermaid
flowchart TB
    subgraph L0["第零层：基础概念"]
        direction TB
        A0[ArchitectureOverview\n架构全景]
        A1[vllmconfig\n配置聚合]
        A2[QuantAndParallelStrategy\n量化与并行策略]
        A3[NvidiaVsAMD\n平台对比]
    end

    subgraph L1["第一层：模型入口"]
        direction TB
        B1[ModelRegistration\n模型注册机制]
        B2[DeepseekV4ForCausalLM\n模型入口类]
        B3[DeepseekV4Model\n模型主体]
        B4[EndToEndDataFlow\n端到端数据流 ★]
    end

    subgraph L2["第二层：解码器层"]
        direction TB
        C1[DeepseekV4DecoderLayer\n解码器层核心]
    end

    subgraph L3["第三层：注意力子系统"]
        direction TB
        D1[attention.py\n注意力代码总览]
        D2[DeepseekV4MLAAttention\nMLA 注意力层]
        D3[DeepseekV4MultiHeadLatentAttentionWrapper\nMLA 包装器 + 多流]
        D4[V1AttentionBackend\nFlashMLA 稀疏后端]
    end

    subgraph L4["第四层：KV Cache 与压缩"]
        direction TB
        E1[DeepseekCompressor\n压缩器]
        E2[DeepseekV4Indexer\n稀疏索引器]
        E3[SparseAttnCompress\n稀疏注意力压缩]
        E4[FusedInvRopeFP8Quant\n融合逆 RoPE + FP8 量化]
        E5[FusedIndexerQ\nIndexer Q 融合]
        E6[DeepseekV4_KVCache_Ops\nKVCache 算子全集]
        E7[common_ops\n公共算子]
        E8[DequantGatherKCache\n反量化 Gather]
    end

    subgraph L5["第五层：MoE 子系统"]
        direction TB
        F1[DeepseekV4MoE\nMoE 层]
        F2[GateRouting\n门控路由数学]
        F3[PrepareMegaMoEInputs\nMegaMoE 输入准备]
    end

    subgraph L6["第六层：MHC 算子层"]
        direction TB
        G1[MHCOps\nMHC 算子四实现]
    end

    subgraph L7["第七层：输出头"]
        direction TB
        H1[ParallelLMHead\n并行 LM Head]
        H2[LogitsProcessor\nLogits 处理器]
    end

    subgraph L8["第八层：推测解码"]
        direction TB
        I1[MTP\n多 Token 预测]
    end

    subgraph L9["第九层：权重加载"]
        direction TB
        J1[load_weights\n权重加载流程]
        J2[VocabParallelEmbedding\n词表并行嵌入]
    end

    subgraph L10["第十层：量化配置"]
        direction TB
        K1[DeepseekV4FP8Config\nFP8 量化配置]
    end

    subgraph L11["进阶主题"]
        direction TB
        L1[ROCmBackend\nAMD ROCm 后端]
        L2[CUDAGraphIntegration\nCUDA Graph 集成]
    end

    %% 跨层引用
    A0 --> A1 --> A2 --> A3
    A2 -.->|量化策略传递| K1
    A3 --> B3

    B1 --> B2 --> B3 --> B4
    B3 --> C1

    C1 --> D1 --> D2

    D2 --> D3
    D3 --> D4
    D2 --> E1

    E1 --> E2
    E2 --> E3
    E3 --> E4
    E2 --> E5
    E4 --> E6
    E5 --> E6
    E6 --> E7
    E7 --> E8

    D3 -.->|多流管理| G1
    C1 --> F1
    F1 --> F2
    F1 --> F3
    F3 --> G1

    B3 --> H1 --> H2
    B3 --> I1
    I1 -.->|依赖| B3

    J2 -.->|嵌入层| B2
    J1 --> J2

    K1 -.->|量化配置| F1
    K1 -.->|量化配置| D2

    L1 -.->|AMD 替代| D4
    L2 -.->|图捕获| D4
    L2 -.->|图捕获| F1

    %% 阅读顺序标注
    style A0 fill:#4a90d9,color:#fff
    style B4 fill:#4a90d9,color:#fff
    style C1 fill:#4a90d9,color:#fff
    style D3 fill:#4a90d9,color:#fff
    style F1 fill:#50b86c,color:#fff
    style J1 fill:#d9a845,color:#fff
```

## 推荐阅读路径

### 🟦 主线路径（从入门到精通）

```
第零层：基础概念
  └── ArchitectureOverview → vllmconfig → QuantAndParallelStrategy → NvidiaVsAMD

第一层：模型入口
  └── ModelRegistration → DeepseekV4ForCausalLM → DeepseekV4Model → EndToEndDataFlow ★

第二层：解码器
  └── DeepseekV4DecoderLayer

第三层：注意力
  └── attention.py → DeepseekV4MLAAttention → DeepseekV4MultiHeadLatentAttentionWrapper
                                                      └── V1AttentionBackend

第四层：KV Cache
  └── DeepseekCompressor → DeepseekV4Indexer → SparseAttnCompress
       └── FusedInvRopeFP8Quant / FusedIndexerQ → DeepseekV4_KVCache_Ops → common_ops

第五层：MoE
  └── DeepseekV4MoE → GateRouting → PrepareMegaMoEInputs → MHCOps

第六层：输出 → 第七层：推测解码 → 第八层：权重加载 → 第九层：量化配置
  └── ParallelLMHead → LogitsProcessor → MTP → load_weights → DeepseekV4FP8Config

进阶：ROCmBackend → CUDAGraphIntegration
```

### 具体阅读顺序建议

#### 🟦 第一遍：整体认知（适合快速通读）

| 步骤 | 笔记 | 预期收获 |
|------|------|---------|
| 1 | [[ArchitectureOverview]] | 了解 V4 整体架构、有哪些创新点 |
| 2 | [[vllmconfig]] | 熟悉关键配置项及其影响 |
| 3 | [[QuantAndParallelStrategy]] | 理解量化在哪里发生、并行怎么切分 |
| 4 | [[NvidiaVsAMD]] | 了解两个平台的差异 |
| 5 | [[DeepseekV4ForCausalLM]] → [[DeepseekV4Model]] | 模型入口和主体结构 |
| 6 | [[EndToEndDataFlow]] ★ | **单条数据完整路径速览** |
| 7 | [[DeepseekV4DecoderLayer]] | 解码器层的融合设计 |

#### 🟩 第二遍：深入关键模块（选择性阅读）

| 步骤 | 笔记 | 预期收获 |
|------|------|---------|
| 8 | [[attention.py]] → [[DeepseekV4MLAAttention]] | MLA 注意力机制 |
| 9 | [[DeepseekV4MultiHeadLatentAttentionWrapper]] | 多流并行 + HC 混合 |
| 10 | [[DeepseekV4MoE]] → [[GateRouting]] | MoE 路由逻辑 |
| 11 | [[DeepseekCompressor]] → [[DeepseekV4Indexer]] | KV 压缩和稀疏索引 |
| 12 | [[MHCOps]] | 理解 mhc_pre/mhc_post 底层实现 |
| 13 | [[MTP]] | 推测解码的工作方式 |

#### 🟨 第三遍：工程细节（按需查阅）

| 步骤 | 笔记 | 预期收获 |
|------|------|---------|
| 14 | [[load_weights]] + [[VocabParallelEmbedding]] | 权重加载流程 |
| 15 | [[FusedInvRopeFP8Quant]] + [[FusedIndexerQ]] | 融合算子的底层细节 |
| 16 | [[DeepseekV4_KVCache_Ops]] + [[common_ops]] | KV Cache 的全部算子 |
| 17 | [[ROCmBackend]] + [[CUDAGraphIntegration]] | AMD 和 CUDA Graph 适配 |
| 18 | [[V1AttentionBackend]] | V1 框架集成 |

---

## 笔记分类速查

### 全景类（先读）
| 笔记 | 核心内容 |
|------|---------|
| [[ArchitectureOverview]] | V4 架构全景，HC 机制解释 |
| [[vllmconfig]] | 所有配置项及其效果 |
| [[QuantAndParallelStrategy]] | 量化点和并行策略 |
| [[NvidiaVsAMD]] | 双平台实现对比 |
| [[EndToEndDataFlow]] | 一条 token 的完整数据流（含 shape 变化） |

### 模型结构类
| 笔记 | 核心内容 |
|------|---------|
| [[DeepseekV4ForCausalLM]] | 入口：forward、load_weights、MTP 接口 |
| [[DeepseekV4Model]] | 主体：embed → layers → norm → lm_head |
| [[DeepseekV4DecoderLayer]] | 单层：融合 attn 和 ffn 的 HC 前后处理 |
| [[ParallelLMHead]] | 词表投影头 |
| [[LogitsProcessor]] | Logits 后处理 |

### 注意力类
| 笔记 | 核心内容 |
|------|---------|
| [[attention.py]] | Attention 模块代码总览 |
| [[DeepseekV4MLAAttention]] | MLA 层：QKV 低秩投影 + 稀疏注意力 |
| [[DeepseekV4MultiHeadLatentAttentionWrapper]] | MLA 包装器：多流并行 + fused_wqa_wkv |
| [[V1AttentionBackend]] | V1 框架的 FlashMLA 后端 |

### KV Cache 类
| 笔记 | 核心内容 |
|------|---------|
| [[DeepseekCompressor]] | KV 压缩器：state cache + softmax 加权 |
| [[DeepseekV4Indexer]] | 稀疏索引器：topk 选择 + 稀疏注意力元数据 |
| [[SparseAttnCompress]] | 稀疏注意力压缩机制 |
| [[FusedInvRopeFP8Quant]] | 注意力输出融合逆 RoPE + FP8 量化 |
| [[FusedIndexerQ]] | Indexer Q 的 RoPE + 量化融合 |
| [[DeepseekV4_KVCache_Ops]] | fused_compress_quant_cache + cache_utils |
| [[common_ops]] | 跨平台公共算子 |
| [[DequantGatherKCache]] | K Cache 反量化 gather |

### MoE 类
| 笔记 | 核心内容 |
|------|---------|
| [[DeepseekV4MoE]] | MoE 层：gate + experts + shared experts |
| [[GateRouting]] | 门控路由数学：sqrtsoftplus / Hash MoE / Noaux TC |
| [[PrepareMegaMoEInputs]] | MegaMoE 的 Triton 预处理 |

### 算子层
| 笔记 | 核心内容 |
|------|---------|
| [[MHCOps]] | MHC 四实现（TileLang/Triton/Torch/Aiter） |

### 推测解码类
| 笔记 | 核心内容 |
|------|---------|
| [[MTP]] | Multi-Token Prediction 完整实现 |

### 配置与加载类
| 笔记 | 核心内容 |
|------|---------|
| [[DeepseekV4FP8Config]] | FP8/FP4 量化配置分发 |
| [[load_weights]] | 权重加载流程 |
| [[VocabParallelEmbedding]] | 词表并行嵌入 |
| [[ModelRegistration]] | 模型注册机制 |

### 进阶类
| 笔记 | 核心内容 |
|------|---------|
| [[ROCmBackend]] | AMD ROCm 注意力后端 |
| [[CUDAGraphIntegration]] | CUDA Graph 兼容策略 |

---

## 依赖关系速查

要理解笔记 A，建议先读笔记 B：

| 笔记 | 前置依赖 |
|------|---------|
| [[DeepseekV4ForCausalLM]] | [[ArchitectureOverview]], [[vllmconfig]] |
| [[DeepseekV4Model]] | [[DeepseekV4ForCausalLM]] |
| [[DeepseekV4DecoderLayer]] | [[DeepseekV4Model]] |
| [[attention.py]] | [[DeepseekV4DecoderLayer]] |
| [[DeepseekV4MLAAttention]] | [[attention.py]] |
| [[DeepseekV4MultiHeadLatentAttentionWrapper]] | [[DeepseekV4MLAAttention]], [[QuantAndParallelStrategy]] |
| [[DeepseekV4MoE]] | [[DeepseekV4DecoderLayer]], [[DeepseekV4FP8Config]] |
| [[GateRouting]] | [[DeepseekV4MoE]] |
| [[MHCOps]] | [[DeepseekV4DecoderLayer]], [[QuantAndParallelStrategy]] |
| [[MTP]] | [[DeepseekV4Model]], [[DeepseekV4DecoderLayer]], [[load_weights]] |
| [[EndToEndDataFlow]] | [[DeepseekV4Model]], [[DeepseekV4DecoderLayer]], [[DeepseekV4MLAAttention]], [[DeepseekV4MoE]] |
| [[ROCmBackend]] | [[V1AttentionBackend]], [[NvidiaVsAMD]], [[DeepseekV4_KVCache_Ops]] |
| [[CUDAGraphIntegration]] | [[DeepseekV4Model]], [[V1AttentionBackend]] |

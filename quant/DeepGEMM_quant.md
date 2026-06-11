# DeepGEMM CUTLASS-style GEMM Kernel — 数据流与架构分析

> 分析对象：`deep_gemm/include/deep_gemm/impls/` 下所有 `.cuh` / `.hpp` 文件（非 JIT 编译内核）
>
> 架构覆盖：**SM90 (Hopper)** — WGMMA + TMA · **SM100 (Blackwell)** — UMMA + TMEM + UTCCP

---

## GEMM 统一框架

**GEMM 公式：** `C[M,N] = A[M,K] @ B[K,N]`

| 组件 | SM90 (Hopper) | SM100 (Blackwell) |
|------|--------------|-------------------|
| 计算指令 | WGMMA (warp-group MMA) | UMMA (uniform MMA) |
| 累加器 | **寄存器** (RF) | **TMEM** (tensor memory, `max 128KB`) |
| 数据搬运 | TMA (tensor memory accelerator) | TMA |
| Scale 搬运 | TMA → smem → register (软件) | TMA → smem → **UTCCP** → TMEM (硬件) |
| 共享内存写回 | STSM (swizzled) | STSM (swizzled) |
| 全局内存写回 | **TMA store** (多数) / AtomicAdd (少数) | TMA store |
| Block scaling | 软件反量化 (寄存器乘 scale) | **硬件反量化** (UMMA 指令描述符) |
| 同步原语 | ClusterTransactionBarrier (mbarrier) | 同左 |

### 量化统一

- **量化发生在 K (reduction) 维度**，每 `gran_k` 个 K 元素共享一个 scale
- **Scale tensor 以 MN-major 布局存储**，最内层维度为 M 或 N
- **反量化累加形式：** `C[m,n] += Σ_k scale_a[m,kb] × scale_b[n,kb] × A[m,k] × B[k,n]`

---

## 量化体系全景

### 数据通路符号约定

```
原始数据              量化后                    Scale
Activation:  float  →  A (FP8/FP4/BF16)    +  scale_a (FP32 / UE8M0)
Weight:      float  →  B (FP8/FP4/BF16)    +  scale_b (FP32 / UE8M0)
                                            └── 量化粒度 gran_k = 128(FP8) / 32(FP4)
```

### 所有内核量化参数总表

| 内核 | A dtype | B dtype | Scale A dtype | Scale B dtype | granKA | granKB | Scale Shape | 输出 C/D dtype | 反量化实现 |
|------|---------|---------|--------------|--------------|--------|--------|------------|---------------|-----------|
| **sm90_bf16_gemm** | BF16 | BF16 | — | — | — | — | — | BF16/BF16 | 无量化 |
| **sm100_bf16_gemm** | BF16 | BF16 | — | — | — | — | — | BF16/BF16 | 无量化 |
| **sm90_bmn_bnk_mn** | BF16 | BF16 | — | — | — | — | — | FP32 (无D) | 无量化 |
| **sm100_bmn_bnk_mn** | BF16 | BF16 | — | — | — | — | — | FP32 (无D) | 无量化 |
| **sm90_fp8_gemm_1d1d** | FP8(e4m3) | FP8(e4m3) | FP32 | FP32 | 128 | 128 | `[M,K/128]/[N,K/128]` | **FP32**/FP32 | **软件** |
| **sm90_fp8_gemm_1d2d** | FP8(e4m3) | FP8(e4m3) | FP32 | FP32(裸指针) | 128 | 128 | `[M,K/128]`/裸指针2D | **BF16**/BF16 | **软件** |
| **sm100_fp8_fp4_gemm_1d1d** | FP8/FP4 | FP8/FP4 | UE8M0(int32) | UE8M0(int32) | **32/128** | **32/128** | `[M,K/(gKA×4)]/[N,K/(gKB×4)]` | FP32/FP32 | **硬件** |
| **sm100_fp8_gemm_1d1d** | FP8 | FP8 | UE8M0 | UE8M0 | 128 | 128 | `[M,K/512]/[N,K/512]` | FP32/FP32 | **硬件** |
| **sm100_fp8_fp4_mega_moe** (L1) | FP8(e4m3) | **FP4**(e2m1) | UE8M0 | UE8M0 | 32 | 32 | `[pool,hid/128]/[2×inter,hid/128×E]` | FP8+SF/BF16 | **硬件** |
| **sm100_fp8_fp4_mega_moe** (L2) | FP8 | FP4 | UE8M0 | UE8M0 | 32 | 32 | `[pool,inter/128]/[hid,inter/128×E]` | BF16 (无D) | **硬件** |
| **sm90_fp8_mqa_logits** | FP8 | FP8 | — | **FP32** kv_scales | — | 128 | —/kv_scales:`[seq_len_kv,1]` | FP32/可配 | **软件** |
| **sm100_fp8_mqa_logits** | FP8 | FP8 | — | **FP32** kv_scales | — | 128 | —/kv_scales:`[seq_len_kv,1]` | FP32/可配 | **软件** |
| **sm100_fp4_mqa_logits** | **FP4** | **FP4** | **UE8M0** sf_q | **UE8M0** sf_kv | per-q | 128 | sf_q:`[nHeads,seq]`/kv:`[seq_len_kv,1]` | FP32/可配 | **硬件** |
| **sm90/100_tf32_hc_prenorm** | BF16 | FP32 | — | — | — | — | — | FP32/FP32 | 无量化 |

### 量化粒度与 Scale 开销

| 数据类型 | gran_k | 字节/元素 | K=4096 时 scale 数 | 每 scale 字节 | Scale 总字节 (K=4096) | 数据总字节 (K=4096) | 存储开销比 |
|---------|--------|----------|-------------------|--------------|---------------------|-------------------|-----------|
| **FP8 + FP32** | 128 | 1 B | 32 | 4 B | 128 B | 4096 B | 128/4096 = **3.125%** |
| **FP8 + UE8M0** | 128 | 1 B | 32 | 1 B | 32 B | 4096 B | 32/4096 = **0.78%** |
| **FP4 + UE8M0** | 32 | 0.5 B | 128 | 1 B | 128 B | 2048 B | 128/2048 = **6.25%** |

**关键观察：** FP4 + UE8M0 的总 scale 存储（128B）等于 FP8 + FP32 的总 scale 存储（128B）。

原因：FP4 的量化粒度是 FP8 的 4 倍（gran_k=32 vs 128，即 scale 密度 4x），但 UE8M0 的 4:1 压缩（FP32 4B → UE8M0 1B）恰好抵消了这 4 倍的 scale 数量增长。所以 FP4+UE8M0 比 FP8+FP32 节省了 2x 的数据存储（0.5B vs 1B/elem），但 scale 存储相同。

```
FP8 (gran_k=128, FP32 scale):    32 scales × 4B = 128B scale
FP4 (gran_k=32, UE8M0 scale):   128 scales × 1B = 128B scale   ← 相同！
                                    ↑ 4x 更多 scale    ↑ 4:1 压缩
```

而如果与 FP8+UE8M0 相比（SM100），FP8 的 scale 开销小得多（0.78%），因为 gran_k 大且用了压缩。

### Scale Tensor 通用公式

```
Scale tensor TMA 逻辑 shape = [align(MN_dim, 16/elem_size), outer_dim]

outer_dim = ceil_div(K, gran_k × pack_factor) × num_groups

pack_factor:
  FP32 (SM90):          pack_factor = 1  (1 scale = 1 FP32, 4B)
  UE8M0 (SM100):        pack_factor = 4  (4 scales = 1 packed uint32, 4B)
```

- TMA 逻辑视图为 2D：内层 = MN 维度（连续），外层 = K-SF 维度（stride）
- TMA box size = `[BLOCK_MN, 1]`，每次沿 M 或 N 方向加载 `BLOCK_MN` 个 scale
- Host 侧经由 `make_tma_sf_desc`（`runtime_utils.hpp:246`）创建描述符

---

## 各内核量化实现详解

### BF16 GEMM — 无量化

**文件：** `sm90_bf16_gemm.cuh` · `sm100_bf16_gemm.cuh`

所有 BF16 GEMM 不使用量化，A、B 均为原生 BF16，直接进行 WGMMA/UMMA。

```
WGMMA/BF16:  A_bf16 × B_bf16 → accum_fp32    // SM90 WGMMA 原生 BF16 支持
UMMA/BF16:    A_bf16 × B_bf16 → TMEM_fp32     // SM100 UMMA 原生 BF16 支持
```

**BMK-BNK-MN 变体**（`sm90_bmk_bnk_mn.cuh` · `sm100_bmk_bnk_mn.cuh`）同样 BF16 无量化。

---

### SM90 FP8 1D:1D — 软件反量化

**文件：** `sm90_fp8_gemm_1d1d.cuh`

#### 量化参数

| 参数 | Activation | Weight |
|------|-----------|--------|
| 数据类型 | FP8 (e4m3) | FP8 (e4m3) |
| Scale 类型 | FP32 | FP32 |
| gran_k | 128 | 128 |
| 反量化方式 | 软件 (寄存器乘, rank-1 外积) | 同左 |
| Scale 形状 | `[align(M,4), ceil(K/128)]` | `[align(N,4), ceil(K/128)]` |

#### 数据通路

```
Global Memory          Shared Memory              Registers
┌─────────┐   TMA     ┌──────────┐   WGMMA       ┌──────┐
│ A_fp8   │──────────→│ smem_a   │──────────────→│      │
│ B_fp8   │──────────→│ smem_b   │──────────────→│ accum│
│         │           │          │               │      │
│ scale_a │──────────→│ smem_sfa │──ld_shared──→│ sfa  │──┐
│ (FP32)  │   TMA     │ (FP32)   │               │ reg  │  │
│         │           │          │               │      │  │
│ scale_b │──────────→│ smem_sfb │──ld_shared──→│ sfb  │──┤
│ (FP32)  │   TMA     │ (FP32)   │               │ reg  │  │
└─────────┘           └──────────┘               └──┬───┘  │
                                                    │      │
                 ┌──────────────────────────────────┘      │
                 │  final_accum += scale_a × scale_b × accum
                 └─────────────────────────────────────────┘
```

#### Scale 加载

```cpp
// TMA 描述符以 MN-major 布局访问 scale tensor
// scale_a shape: [num_groups, ceil(K/gran_k), M], gran_k=128
// scale_b shape: [num_groups, ceil(K/gran_k), N]
tma::copy<BLOCK_M, 1, 0>(&tensor_map_sfa, ..., smem_sfa[stage_idx], m_idx, sf_k_idx);
tma::copy<BLOCK_N, 1, 0>(&tensor_map_sfb, ..., smem_sfb[stage_idx], n_idx, sf_k_idx);
// 每次加载 BLOCK_M 个 FP32 scale（对 scale_a）或 BLOCK_N 个（对 scale_b）
```

#### 反量化代码（`sm90_fp8_gemm_1d1d.cuh:300-309`）

```cpp
// WGMMA 结果 accum[i] 是 FP8×FP8→FP32 的原始累加值（未 scale）
// 每个 K-block 结束后，用 scale_a[BLOCK_M] × scale_b[BLOCK_N] 的外积逐个修正

// scale_a_0 = sfa[r_0], scale_a_1 = sfa[r_1]  (r_0 = row, r_1 = row+8)
// scales_b[i] = {sfb[n_0], sfb[n_1]}  (每 4 个 accum 对应 2 个 N 方向 scale)

for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
    final_accum[i*4+0] += scale_a_0 * scale_b.x * accum[i*4+0];  // m_row,   n_col_0
    final_accum[i*4+1] += scale_a_0 * scale_b.y * accum[i*4+1];  // m_row,   n_col_1
    final_accum[i*4+2] += scale_a_1 * scale_b.x * accum[i*4+2];  // m_row+8, n_col_0
    final_accum[i*4+3] += scale_a_1 * scale_b.y * accum[i*4+3];  // m_row+8, n_col_1
}
```

**关键约束：**
- `BLOCK_K == 128` 固定（`DG_STATIC_ASSERT`），与 gran_k=128 一致
- C 输出必须为 FP32（`cd_dtype_t = float`）
- TMA multicast 支持（空 barrier 用 `lane_idx` arrive）

**K-Grouped 变体：**
- 多组 weight 共享同一 activation
- TMA 描述符动态更新：`tensor_map_replace_global_addr_in_smem` + `tensor_map_acquire_gpu`
- 每组独立 scale_b
- Scale shape 变化：host 将各组 K 拼接，`sum_sf_k = Σ ceil(K_i, 128)`，device 端通过 `scheduler.current_sf_k_cumsum + k_block_idx` 索引

---

### SM90 FP8 1D:2D — 2D 分段 Weight Scale + 软件反量化

**文件：** `sm90_fp8_gemm_1d2d.cuh`

#### 量化参数

| 参数 | Activation | Weight |
|------|-----------|--------|
| 数据类型 | FP8 (e4m3) | FP8 (e4m3) |
| Scale 类型 | FP32 (TMA) | FP32 (裸指针, math threads 直读) |
| gran_k | 128 | 128 |
| 反量化方式 | 软件 (寄存器乘, 1D:2D 双组 predicate) | 同左 |
| Scale 形状 | `[align(M,4), ceil(K/128)×num_groups]` | **裸指针 2D 分段**，M-block 偏移由 scheduler 决定 |

#### 反量化公式

```
C[m,n] += scale_a[m, kb] × scale_b[kb, n_group] × Σ A_fp8[m,k] × B_fp8[k,n]
```

其中 `n_group = 1 或 2`（取决于 `BLOCK_N` 与 `BLOCK_K` 的关系），scale_b 索引同时依赖 k_block_idx 和 n_block_idx。

#### Scale A 数据通路（与 1D:1D 相同）

```
scale_a (FP32, MN-major TMA) → TMA → smem_sfa → ld_shared → register
```

#### Scale B 数据通路（与 1D:1D 的本质区别）

```
scale_b (FP32, 裸指针) → math threads 直读全局内存 → st_shared → smem_sfb → ld_shared → register
```

#### Scale B 加载细节（`sm90_fp8_gemm_1d2d.cuh:236-244`）

```cpp
// 2D scale_b 的寻址：
// - previous_group_offset: M-block 对应的偏移（scheduler 决定）
// - (n_block_idx * BLOCK_N) / BLOCK_K: 当前 N-block 在 K 方向上的组号
// - stride_n_sfb / stride_k_sfb: 由 kMajorSFB (MN/K-major) 决定
uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
auto local_sfb = sfb + previous_group_offset
               + ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;

// 加载 2 组 scale_b（第 1 组 + 可能存在的第 2 组）
for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
    ptx::st_shared(smem_sfb + i,
        i < shape_k_scales ?
            local_sfb[i * stride_k_sfb] :                    // 第 1 组
            local_sfb[(i - shape_k_scales) * stride_k_sfb + stride_n_sfb]);  // 第 2 组
```

#### 动态循环调度（`dispatch_num_former_iters`）

```cpp
// 对于 BLOCK_N > BLOCK_K（例如 BLOCK_N=256, BLOCK_K=128），
// 一个 N-block 会跨越 2 个 K-group，前一半 iter 用第 1 组 scale，后一半用第 2 组
//
// BLOCK_N=128, BLOCK_K=128 → kMustUseUniformedScaleB = true,
//   所有 iter 使用同一组 scale_b（num_former_iters = num_full_iters）
//
// BLOCK_N=256, BLOCK_K=128 → kMustUseUniformedScaleB = false,
//   前 128/8=16 个 scale 用组 1，后 16 个用组 2

num_former_iters = min(BLOCK_N, BLOCK_K - (n_block_idx * BLOCK_N) % BLOCK_K) / 8;
```

#### 反量化逻辑

```cpp
float scale_b_0 = smem_sfb[k_block_idx];                      // 第 1 组
float scale_b_1 = smem_sfb[k_block_idx + shape_k_scales];     // 第 2 组

// predicate 决定当前 iter 使用哪组 scale_b
const bool predicate = i < num_former_iters;
shifted_accum[i*4+0] += (predicate ? scale_b_0 : scale_b_1) * scale_a * accum[i*4+0];
```

**核心特性：**
- C/D 输出为 **BF16** 而非 FP32（与其他 FP8 内核不同）
- Scale B 由 math warp 直读全局内存（不经过 TMA）
- Scale B 物理上**不是规整 tensor**，由 scheduler 分段

---

### SM100 FP8/FP4 1D:1D — 硬件 block-scaling

**文件：** `sm100_fp8_fp4_gemm_1d1d.cuh` · `sm100_fp8_gemm_1d1d.cuh`

#### 量化参数

| 参数 | Activation | Weight |
|------|-----------|--------|
| 数据类型 | FP8(e4m3) / FP4(e2m1) | FP8(e4m3) / FP4(e2m1) |
| Scale 类型 | UE8M0 (int32) | UE8M0 (int32) |
| gran_k | 32 或 128, 独立可配 | 同左 |
| 反量化方式 | 硬件 UMMA block-scaling | 同左 |
| Scale 形状 | `[align(M,4), ceil(K/gKA×4)]` | `[align(N,4), ceil(K/gKB×4)]` |

#### 数据通路

```
Global Memory       Shared Memory          TMEM              UMMA
┌─────────┐  TMA   ┌──────────┐  UTCCP   ┌──────┐  硬件自动  ┌──────┐
│ A_fp8   │───────→│ smem_a   │─────────→│      │──────────→│      │
│ B_fp8   │───────→│ smem_b   │          │      │           │ TMEM │
│         │        │          │          │      │           │ accum│
│ scale_a │───────→│ smem_sfa │─────────→│ sfa  │──┐        │      │
│ (FP32)  │  TMA   │ (FP32)   │ UTCCP    │ TMEM │  │        │      │
│         │        │          │          │      │  │        │      │
│ scale_b │───────→│ smem_sfb │─────────→│ sfb  │──┤        │      │
│ (FP32)  │  TMA   │ (FP32)   │ UTCCP    │ TMEM │  │        │      │
└─────────┘        └──────────┘          └──────┘  │        └──────┘
                                                    │
              UMMA 指令描述符: scale_id_sfa, scale_id_sfb 指定 TMEM 位置
              UMMA 硬件自动:  accum += scale_a × scale_b × A × B
```

#### 关键差异：UE8M0 打包

SM100 不在寄存器中做乘法，而是利用 **UMMA 硬件 block-scaling**：

```cpp
// 1. 创建指令描述符，指定数据类型和 gran_k
auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
    a_dtype_t,          // A: FP8(e4m3) 或 FP4(e2m1)
    b_dtype_t,          // B: FP8(e4m3) 或 FP4(e2m1)
    float,              // 累加器: FP32
    cutlass::float_ue8m0_t  // Scale: UE8M0 (unsigned E8M0)
>(kGranKA, kGranKB);    // gran_k = 32 或 128

// 2. UTCCP 将 smem 中排好序的 FP32 scale 按 128 元素一组写入 TMEM
//    硬件自动将 FP32 的 exponent 提取为 UE8M0
cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFA);

// 3. UMMA 执行时，每 UMMA_K=32 个 K 元素，硬件自动读取 TMEM 中的 scale
//    granKA=128 → 每 4 个 UMMA_K step 读一次 scale（k % 128 == 0）
//    granKA=32  → 每个 UMMA_K step 读一次 scale
mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
           k_block_idx > 0 or k > 0,  // 是否为首次累积
           runtime_instr_desc,
           kTmemStartColOfSFA, kTmemStartColOfSFB);
```

#### SM100 scale 加载频率优化

```cpp
// granKA=128 → kNumSFAStagesPerLoad = 4 → 每 4 个 K-block 加载一次 scale
// granKA=32  → kNumSFAStagesPerLoad = 1 → 每个 K-block 都加载
// 原因是：K 方向跨度 = BLOCK_K * kNumSFAStagesPerLoad = 128 * 4 = 512 元素
// granKA=128 需要 ceil(512/128)=4 个 scale，一次 TMA 加载完
// granKA=32  需要 ceil(512/32)=16 个 scale，需 4 次 TMA 加载
constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
```

**在 UMMA 中，sf_id 动态选择：**
```cpp
// 每个 UMMA_K=32 的微步，选择对应的 scale
const uint32_t sfa_id = (kGranKA == 32 ? k : sfa_stage_in_group_idx);
// granKA=32: sf_id = k (0..3) → 每个微步不同 scale
// granKA=128: sf_id = sfa_stage_in_group_idx (始终为 0) → 4 个微步同一 scale
```

#### gran_k 可配性约束

| gran_k | FP8 适用 | FP4 适用 | scale 有效范围 | K 方向每组元素 |
|--------|---------|---------|---------------|--------------|
| 128 | 是 | 否 | 1 个 scale/128 元素 | 128 |
| 32 | 是 | **是** | 1 个 scale/32 元素 | 32 |

FP4 是 4-bit 数据，精度更低，需要更细的量化粒度（gran_k=32）。

**模式间差异：**

| 模式 | sfa shape | sfb shape |
|------|----------|----------|
| Normal | `[align(M,4), ceil(K, gKA×4)]` | `[align(N,4), ceil(K, gKB×4)]` |
| MGroupedContiguous | 同上 | `[align(N,4), ceil(K, gKB×4)×num_groups]` |
| MGroupedMasked | `[align(M,4), ceil(K, gKA×4)×num_groups]` | 同上 |
| KGroupedContiguous | K-SF = Σ ceil(K_i, gK×4) | 同左，且 gKA=gKB |
| Batched | `[align(M,4), ceil(K, gKA×4)×batch]` | `[align(N,4), ceil(K, gKB×4)×batch]` |

---

### SM100 Mega MoE — L1 FP8×FP4 + L2 FP8×FP4 硬件量化

**文件：** `sm100_fp8_fp4_mega_moe.cuh`

#### L1 GEMM 量化参数

| 参数 | Activation | Weight (w1) |
|------|-----------|-------------|
| 数据类型 | **FP8 (e4m3)** | **FP4 (e2m1)** |
| Scale 类型 | UE8M0 (int32) | UE8M0 (int32) |
| gran_k | **32** | **32** |
| 实现方式 | 硬件 UMMA block-scaling | 同左 |
| Scale 形状 | `[align(pool_tokens,4), ceil(hidden/128)]` | `[align(2×inter_hidden,4), ceil(hidden/128)×experts]` |

#### L1 输出量化（SwiGLU 后重新量化）

```cpp
// silu(gate) * up → 结果用 FP8 存储，同时生成 scale
float gate_val = tmem_load(tmem_gate, idx);
float up_val = tmem_load(tmem_up, idx);
float result = silu(gate_val) * up_val;

// 量化为 FP8 + UE8M0 scale
l1_out_buffer[out_idx] = float_to_fp8(result);
l1_sf_buffer[sf_idx] = fp32_to_ue8m0(result);  // L2 读取时反量化
```

#### L2 GEMM 量化参数

| 参数 | Activation (L1 output) | Weight (w2) |
|------|----------------------|-------------|
| 数据类型 | **FP8 (e4m3)** | **FP4 (e2m1)** |
| Scale 类型 | UE8M0 | UE8M0 |
| gran_k | 32 | 32 |
| Scale 形状 | `[align(pool_tokens,4), ceil(inter_hidden/128)]` | `[align(hidden,4), ceil(inter_hidden/128)×experts]` |

注意这里 gran_k 虽然是 32，但 scale 存储时仍按 128 打包（UE8M0 把 4 个 scale 打包成 1 个 uint32），因此 shape 中用 `ceil(hidden/128)` 而非 `ceil(hidden/32)`。

---

### MQA Logits — 软件与硬件反量化

#### SM90 FP8 MQA Logits — 软件反量化

**文件：** `sm90_fp8_mqa_logits.cuh`

| 参数 | Q | KV |
|------|------|------|
| 数据类型 | FP8 (e4m3) | FP8 (e4m3) |
| Scale 类型 | — | FP32 kv_scales |
| gran_k | — | 128 (head_dim<=128) |
| 反量化方式 | 软件 (标量乘) | 同左 |
| Scale 形状 | — | `[align(seq_len_kv, 4)]` (1D, per-KV-block) |

**公式：** `logits[q, kv] = Σ_d Q[q, d] × KV[kv, d] × kv_scale[kv]`

```
Q: FP8(e4m3), shape [BLOCK_Q × kNumHeads, kHeadDim]
KV: FP8(e4m3), shape [BLOCK_KV, kHeadDim]
kv_scales: FP32, per-KV-block, shape [ceil(seq_len_kv, BLOCK_KV)]

数据流:
WGMMA(Q_fp8, KV_fp8) → partial_accum (FP32, K 未规约)
  ↓ warp_reduce_sum (沿 K 方向 → 每个 thread 一个标量)
  ↓ 标量 × kv_scale[kv_block]（反量化）
  ↓ fmaxf(0) × weights[q][kv_block]（Weighted ReLU）
logits[q, kv]
```

#### SM100 FP4 MQA Logits — 硬件 block-scaling

**文件：** `sm100_fp4_mqa_logits.cuh`

| 参数 | Q | KV |
|------|------|------|
| 数据类型 | FP4 (e2m1) | FP4 (e2m1) |
| Scale 类型 | UE8M0 sf_q | UE8M0 sf_kv |
| gran_k | per-query | 128 (head_dim<=128>) |
| 反量化方式 | 硬件 UMMA block-scaling | 同左 |
| Scale 形状 | `[num_heads, seq_len]` (per-head per-query) | `[align(seq_len_kv, 16/elem_size)]` (1D) |

**双路 scale （SF_Q + SF_KV）：**

```
Q: FP4(e2m1), shape [BLOCK_Q × kNumHeads, kHeadDim]
   sf_q: UE8M0, per-head per-query, TMA 加载 → smem → UTCCP → TMEM

KV: FP4(e2m1), shape [BLOCK_KV, kHeadDim]
   sf_kv: UE8M0, per-KV-block, TMA 加载 → smem → UTCCP → TMEM

UMMA 指令描述符:
auto instr_desc = make_instr_desc_block_scaled<
    float_e2m1_t, float_e2m1_t, float, float_ue8m0_t
>(granKQ, granKKV);
```

**TMEM 分配：**
```
TMEM 布局:
  [accum 区域]      : kNumAccumTmemCols = BLOCK_Q × kNumHeads × kNumTmemStages
  [sf_q 区域]       : kTmemStartColOfSFQ = kNumAccumTmemCols, kNumSFQ/32 列
  [sf_kv 区域]      : kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ/32, kNumSFKV/32 列
```

**paged 变体额外：**
- `block_table` 实现 KV cache 物理页映射
- `kNextNAtom` / `kNumNextNAtoms` 处理跨 page 边界

---

### TF32 预归一化 GEMM — 无量化，但带 sqr_sum

**文件：** `sm90_tf32_hc_prenorm_gemm.cuh` · `sm100_tf32_hc_prenorm_gemm.cuh`

**特殊性：** 这不是量化 GEMM，但在 A 加载阶段手动进行 BF16→FP32 转换，同时累加平方和用于后续 normalization。

```cpp
// SM90: Math threads 手动从 smem 加载 BF16 A，转 FP32，累加 sqr_sum
nv_bfloat16 a_bf16 = smem_a[stage_idx][row * BLOCK_K + bank_group_idx * 16 + elem_offset];
float a_float = __bfloat162float(a_bf16);
sqr_sum_acc += a_float * a_float;

// WGMMA 使用 TF32 精度 × FP32 weight
using WGMMA = mma::sm90::TF32MMASelector<WGMMA_N, true>::type;
```

---

## UE8M0 格式详解

### 定义

UE8M0 是 **unsigned E8M0** 浮点格式：
- 8-bit exponent (biased, E8M0 = 无 sign bit、无 mantissa bits)
- 值 = 2^(exponent - 127) × 2^sign_bias
- 用于表示 FP32 scale 的指数部分（FP32 的 bit[30:23]）

### 打包算法（`smxx_layout.cuh:100-107`）

```cpp
// 4 个 FP32 exponent → 1 个 uint32
// FP32 bit layout: S[31] | E[30:23] | M[22:0]
// UE8M0 bit layout: E[7:0]

uint32_t packed = 0;
packed |= (values[0] >> 23u);   // FP32[30:23] → uint32[7:0]
packed |= (values[1] >> 15u);   // FP32[30:23] → uint32[15:8]
packed |= (values[2] >>  7u);   // FP32[30:23] → uint32[23:16]
packed |= (values[3] <<  1u);   // FP32[30:23] → uint32[31:24]
```

### UE8M0 读取路径

```
FP32 scale 数组 → transpose + pack → uint32 array (MN-major)
  → TMA → smem (uint32 array)
  → UTCCP → TMEM (硬件自动解包为 4 个独立 UE8M0)
  → UMMA 指令解码器
```

### 硬件解包原理

SM100 UTCCP 指令 `SM100_UTCCP_4x32dp128bit_1cta` 从 smem 读取 128 字节（32 个 uint32），转置后写入 TMEM。UMMA 硬件自动：
1. 从 TMEM 读取每个 tile 对应的 UE8M0
2. 将其左移 23 位恢复为 FP32（在 exponent 位置）
3. 用这个 FP32 scale 乘对应的 FP8/FP4 block

---

## Scale 布局转换工具

**文件：** `smxx_layout.cuh` (189 行)

### 三个 Utility Kernel

```cpp
// 1. FP32 转置: (SF_K, MN) → (MN, SF_K) — 使最内层为 MN 维度
//    用途: SM90 FP8 1D:1D/1D:2D 的 scale
transpose_fp32<kNumThreads, BLOCK_MN, SF_K>(sf, out, mn);

// 2. FP32 转置 + 打包为 UE8M0
//    用途: SM100 的 scale 预处理
transpose_and_pack_fp32_into_ue8m0<kNumThreads, BLOCK_MN, SF_K>(sf, out, mn);

// 3. 已转置的 FP32 打包为 UE8M0（K-Group aware）
//    用途: 多组 K-Grouped scale 的打包
pack_fp32_into_ue8m0<kNumGroups, kNumThreads, ...>(sf, out, ks, mn, sf_k, packed_sf_k, gran_k);
```

### 三种转换模式对比

| 内核 | Scale 原始布局 | Scale 中间布局 | 最终 Scale 格式 |
|------|--------------|--------------|---------------|
| SM90 1D:1D | K-major: `(SF_K, MN)` | MN-major: `(MN, SF_K)` **FP32** | FP32（TMA 直接消费） |
| SM90 1D:2D | K-major | MN-major **FP32** | FP32（math threads 直读） |
| SM100 所有 | K-major | MN-major **UE8M0 packed int32** | UE8M0（TMA + UTCCP） |

---
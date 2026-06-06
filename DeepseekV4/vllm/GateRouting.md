# DeepSeek V4 MoE Gate Routing

## Overview

DeepSeek V4 uses a learned gating network to route each token to a subset of experts. The gate is a linear projection (`GateLinear`) from `hidden_size` -> `n_routed_experts`, producing raw logits. These logits are then passed through a **scoring function** to produce per-expert scores, and the top-k experts by score are selected.

The routing pipeline lives at `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` (the `fused_topk_bias()` function) and is called from `DeepseekV4MoE.forward()` at `vllm/models/deepseek_v4/nvidia/model.py`.

---

## Scoring Functions

Three scoring functions are supported, controlled by the `scoring_func` config attribute (default: `"sqrtsoftplus"`).

### 1. softmax

`ops.topk_softmax` -- standard softmax over the gating logits `(M, n_experts)`, producing a probability distribution. The top-k indices and their probabilities are returned.

Used by many traditional MoE models (Mixtral, DeepSeek V2, etc.). Not the default for DeepSeek V4.

### 2. sigmoid

`ops.topk_sigmoid` -- element-wise sigmoid applied to gating logits. Each expert gets an independent score in (0, 1). The top-k by sigmoid score are selected.

Widely used by many MoE models. Not the default for DeepSeek V4.

### 3. sqrtsoftplus (default)

`ops.topk_hash_softplus_sqrt` (CUDA custom op) or `_topk_softplus_sqrt_torch` (PyTorch fallback).

The scoring function is:

```
score = sqrt(softplus(x))   where softplus(x) = log(1 + exp(x))
```

This is DeepSeek V4's default. When MegaMoE is enabled, only `sqrtsoftplus` is supported (`DeepseekV4MoE.__init__` raises if `scoring_func != "sqrtsoftplus"`).

---

## sqrtsoftplus in Detail

**Math**: `score_i = sqrt(ln(1 + exp(gate_i)))`

Unlike softmax (which produces a coupled probability distribution) or sigmoid (bounded between 0 and 1), sqrtsoftplus has useful properties:

- **Unbounded and monotonic**: each expert's score is independent of others. There is no "competition" between experts for probability mass, unlike softmax.
- **No vanishing gradients for negative logits**: softplus is approximately `exp(x)` for very negative `x` and approximately `x` for very positive `x`, giving smooth gradients everywhere. The sqrt compresses the dynamic range.
- **Composable with `e_score_correction_bias`**: because scores are independent per-expert, adding a learned bias directly shifts the "activation threshold" for each expert, which is harder to interpret with softmax.

The `vllm_topk_softplus_sqrt` function (line 106 in `fused_topk_bias_router.py`) dispatches to either:

- **CUDA path**: `ops.topk_hash_softplus_sqrt(...)` -- a fused CUDA kernel registered via `torch.library.impl` at `csrc/moe/torch_bindings.cpp`. The kernel implementation is at `csrc/moe/topk_softplus_sqrt_kernels.cu`.
- **PyTorch fallback** (XPU/CPU): `_topk_softplus_sqrt_torch(...)` -- a pure PyTorch implementation.

**The CUDA kernel fuses**: softplus + sqrt + bias addition + topk selection + hash lookup + renormalization into a single kernel launch. This avoids materializing intermediate tensors on HBM.

---

## Three Routing Modes

The routing mode is determined in `DeepseekV4MoE.__init__()` (model.py lines 505-524):

### Hash MoE

Activated when `extract_layer_index(prefix) < config.num_hash_layers`, i.e. early layers use hash-based routing.

- **No `e_score_correction_bias`** is created.
- A `tid2eid` table of shape `(vocab_size, top_k)` is allocated as an `nn.Parameter` (frozen, `requires_grad=False`), pre-filled with random expert IDs (using `torch.randint` to avoid garbage-value memory access in dummy mode).
- During routing: expert IDs are **looked up** from the table (`hash_indices_table[input_tokens]`) instead of selecting via topk. This means routing is deterministic and static -- no learned gating for these layers.

The dtype of `tid2eid` is `torch.int64` when `use_mega_moe` is True, else `torch.int32`.

### Noaux TC

Activated when `config.topk_method == "noaux_tc"`.

- An `e_score_correction_bias` of shape `(n_routed_experts,)` with dtype `float32` is created as an `nn.Parameter` (frozen, `requires_grad=False`).
- This bias is **learned during training** to balance expert load without an auxiliary loss.
- The bias is added to scores for **selection only** -- the actual weight values come from unbiased scores (see [[#Bias Handling]]).

"Noaux TC" stands for **no auxiliary token counter**. The gating does not produce an auxiliary load-balancing loss during training; instead, the `e_score_correction_bias` is the learned mechanism that balances expert selection.

### Standard

Neither hash MoE nor noaux TC. Plain top-k routing with the configured scoring function. No `e_score_correction_bias` and no `tid2eid` table.

---

## Bias Handling

The `e_score_correction_bias` is a 1D tensor of shape `(n_routed_experts,)` with dtype `float32`.

**Critical design decision**: The bias is added to scores for **selection only**, not for weight computation.

```python
# Line 75-81 in fused_topk_bias_router.py
# Bias is used for expert SELECTION only, not for weight computation.
# Using biased scores as weights flattens the distribution when the bias
# is near-uniform (e.g., DSv4-Flash where all biases ≈ 8.08).
if e_score_correction_bias is not None:
    scores_for_choice = scores + e_score_correction_bias.float()
else:
    scores_for_choice = scores

# Selection uses biased scores:
_, indices = torch.topk(scores_for_choice, k=topk, dim=-1)
# Weights are gathered from UNBIASED scores:
weights = scores.gather(1, indices)
```

**Rationale**: In DeepSeek V4-Flash, the learned bias values are all approximately 8.08 (near-uniform). If the bias were added to the weights directly, every expert would get roughly the same weight, flattening the distribution and losing the signal from the gating network. By using the bias only for selection, the gating network's relative scores still determine the weight magnitudes.

The bias is also handled in `fused_topk_bias()` via the ROCm AITER path (line 245-256) and the final fallback path (line 267-289), always with the same pattern: bias added to `scores_for_choice`, weights gathered from unbiased `scores`.

---

## Pure PyTorch Fallback vs Custom CUDA Kernel

### PyTorch Fallback (`_topk_softplus_sqrt_torch`, line 60-103)

Used on XPU/CPU platforms. The complete pipeline:

```
1. scores = sqrt(softplus(gating_output.float()))
2. If bias: scores_for_choice = scores + bias.float()
3a. If hash: expert_ids = hash_indices_table[input_tokens]   (lookup)
3b. If standard: indices = torch.topk(scores_for_choice, k=topk)
4. weights = scores.gather(1, indices)                        (UNBIASED)
5. If renormalize: weights /= weights.sum(dim=-1).clamp(min=1e-20)
6. weights *= routed_scaling_factor
```

Note: `scores` is always cast to `float32` before the sqrt(softplus(...)) computation for numerical stability.

### CUDA Custom Op (`ops.topk_hash_softplus_sqrt`)

Registered at `csrc/moe/torch_bindings.cpp` as a custom op under the `topk_softplus_sqrt` name. Dispatches to `dispatch_topk_softplus_sqrt_launch` in `csrc/moe/topk_softplus_sqrt_kernels.cu`.

Supports float32, float16 (`__half`), and bfloat16 (`__nv_bfloat16`) input dtypes.

**Fused operations**: softplus + sqrt + bias addition + topk selection + hash table lookup + renormalization + scaling. All done in a single kernel to minimize global memory traffic.

### CUDA Path for Softmax/Sigmoid

When the scoring function is `softmax` or `sigmoid`, the corresponding ops (`ops.topk_softmax`, `ops.topk_sigmoid`) are used. The `topk_sigmoid` path also has a ROCm AITER path (`rocm_aiter_ops.biased_grouped_topk`).

### Fallback Path in `fused_topk_bias()` (line 258-289)

When `rocm_aiter_ops` is enabled but conditions are not met, or for any other reason the custom ops don't apply, a pure PyTorch fallback is used:

```python
scores = F.softplus(gating_output).sqrt()  # for sqrtsoftplus
scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
topk_indices = torch.topk(scores_for_choice, k=topk, sorted=True)[1]
topk_weights = scores.gather(1, topk_indices)
```

Note `sorted=True` for batch invariance (deterministic expert ordering).

---

## How `fused_topk_bias()` Is Called from `DeepseekV4MoE.forward()`

### MegaMoE Path (`use_mega_moe=True`)

```python
# model.py lines 613-627
router_logits, _ = self.gate(hidden_states)                         # [M, n_experts]
topk_weights, topk_ids = fused_topk_bias(
    hidden_states=hidden_states,
    gating_output=router_logits,
    scoring_func=self.scoring_func,                                  # "sqrtsoftplus"
    e_score_correction_bias=self.gate.e_score_correction_bias.data,  # or None
    topk=self.n_activated_experts,                                   # num_experts_per_tok
    renormalize=self.renormalize,                                    # config.norm_topk_prob
    indices_type=self.hash_indices_dtype,                            # int64 for megamoe, int32 otherwise
    input_tokens=input_ids,                                          # [M] or None
    hash_indices_table=self.gate.tid2eid,                            # or None
    routed_scaling_factor=self.routed_scaling_factor,
)
final_hidden_states = self.experts(hidden_states, topk_weights, topk_ids, ...)
```

The `gate` is a `GateLinear` instance (imported from `vllm/model_executor/layers/fused_moe/router/gate_linear.py`). It is a three-tier GEMM layer:

1. **Tier 1** (DSV3 specialized kernel, SM90+, batch <= 16): `ops.dsv3_router_gemm`
2. **Tier 2** (cuBLAS bf16 x bf16 -> fp32): `torch.mm(x, W.T, out_dtype=float32)`
3. **Tier 3** (fallback via `ReplicatedLinear`): `F.linear`

The output dtype of `GateLinear.forward()` is `torch.float32` (set at model.py line 600: `router_logits_dtype=torch.float32`).

### FusedMoE Path (`use_mega_moe=False`)

The gate and routing are handled inside the `FusedMoE` class (`vllm/model_executor/layers/fused_moe`). The gate runs first:

```python
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=hidden_states,
    router_logits=router_logits,
    input_ids=input_ids,
)
```

The `FusedMoE` internally uses `FusedTopKBiasRouter` (if `scoring_func` is set) to perform the routing.

---

## Shape / Dtype Tables

### Gate (Linear Projection)

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `hidden_states` | `(M, hidden_size)` | bf16/fp16 | Token hidden states from previous layer |
| `gate.weight` | `(n_routed_experts, hidden_size)` | bf16/fp16 | Gate projection weights |
| `router_logits` | `(M, n_routed_experts)` | float32 | Raw gating logits (gate output) |

### Routing Outputs

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `topk_weights` | `(M, top_k)` | float32 | Final routing weights (scores after selection + renormalize + scaling) |
| `topk_ids` | `(M, top_k)` | int32 / int64 | Selected expert IDs; int64 for MegaMoE, int32 otherwise |

### Bias and Hash Table

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `e_score_correction_bias` | `(n_routed_experts,)` | float32 | Learned bias for expert selection balancing |
| `tid2eid` (hash table) | `(vocab_size, top_k)` | int32 / int64 | Token ID -> Expert ID lookup table for hash MoE layers |

### Intermediate Tensors (PyTorch Fallback)

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `scores` | `(M, n_routed_experts)` | float32 | `sqrt(softplus(router_logits))` |
| `scores_for_choice` | `(M, n_routed_experts)` | float32 | `scores + bias` (selection only) |
| `token_expert_indices` | `(M, top_k)` | int32 | Scratch tensor for custom ops (not used by the PyTorch fallback) |

---

## Design Rationale

### sqrtsoftplus vs softmax

| Property | softmax | sigmoid | sqrtsoftplus |
|----------|---------|---------|--------------|
| Expert competition | Yes (sum=1) | No (independent) | No (independent) |
| Score range | (0, 1) | (0, 1) | (0, inf) |
| Gradient for negative logits | Small near 0 | Small near 0 | Smooth (softplus is ~exp(x) for neg x) |
| Bias semantics | Shifts probability mass | Shifts threshold | Shifts threshold (natural) |
| Numerical stability | Needs `-max` trick | Stable | Stable (softplus has no saturation) |

**Why sqrtsoftplus for DeepSeek V4**:

- **Independent expert scoring**: Unlike softmax, each expert's score is independent. This is critical when `e_score_correction_bias` is used to independently bias each expert's selection threshold.
- **Unbounded scores**: There is no fixed upper bound on scores, which allows the gating network to express high confidence.
- **Gradient properties**: Softplus provides smooth gradients for all inputs, avoiding the saturation issues of sigmoid for very negative or very positive values. The sqrt compresses the dynamic range.
- **MegaMoE compatibility**: The custom CUDA kernel fuses the entire pipeline, which is essential for MegaMoE's performance.

### Bias for Selection Only

The pattern of using bias for selection but not weights is motivated by the behavior observed in DeepSeek V4-Flash:

- In V4-Flash, the learned `e_score_correction_bias` values are all approximately 8.08 -- a near-uniform bias.
- If applied to weights: `weights = (scores + 8.08).gather(...)`, the uniform offset dominates, making all selected experts have nearly the same weight. The gating network's signal is lost.
- If applied only to selection: `scores_for_choice = scores + 8.08` changes which experts are selected, but `weights = scores.gather(1, indices)` preserves the relative magnitudes from the gating network.

This separation of selection (which experts) from weighting (how much each expert contributes) is the key insight. The bias acts as an **expert selection threshold adjuster** rather than a weight modifier.

---

## Related Files and Classes

- `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` -- `fused_topk_bias()` and `FusedTopKBiasRouter`
- `vllm/model_executor/layers/fused_moe/router/gate_linear.py` -- `GateLinear`
- `vllm/models/deepseek_v4/nvidia/model.py` -- `DeepseekV4MoE`
- `vllm/model_executor/layers/fused_moe` -- `FusedMoE`
- `csrc/moe/topk_softplus_sqrt_kernels.cu` -- CUDA kernel for fused sqrtsoftplus + topk
- `csrc/moe/torch_bindings.cpp` -- custom op registration

Related notes: [[MegaMoE]], [[FusedMoE]]

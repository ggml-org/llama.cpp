---
name: ssm-linear-attention-architectures
description: >
  Guide to SSM and linear attention model architectures as implemented in llama.cpp.
  Covers Delta-Net, Mamba, GLA, RWKV, and how they differ from standard transformers.
  Use when working with recurrent/SSM models, implementing new SSM ops, or understanding
  which GGML ops map to which architectures. Also covers state management, GQA handling,
  and available test models on Hugging Face.
---

# SSM and Linear Attention Architectures in llama.cpp

## Architecture Overview

| Architecture | GGML Op | State Shape | Key Feature | Source File |
|-------------|---------|-------------|-------------|-------------|
| Mamba (S6) | `GGML_OP_SSM_SCAN` | [d_state, d_inner, n_seqs] | Selective state spaces, input-dependent gating | `src/models/mamba.cpp` |
| Mamba-2 | `GGML_OP_SSM_SCAN` | [d_state, head_dim, n_head, n_seqs] | Multi-head SSM, structured state | `src/models/mamba.cpp` |
| Delta-Net | `GGML_OP_DELTA_NET_RECURRENCE` | [S, S, H, n_seqs] | Innovation-based update, beta gating | `src/models/delta-net-base.cpp` |
| GLA | `GGML_OP_GATED_LINEAR_ATTN` | [S, S, H, n_seqs] | Gated linear attention, direct k*v outer product | `src/models/delta-net-base.cpp` |
| RWKV-6 | `GGML_OP_RWKV_WKV6` | varies | Time-mixing with decay | `src/models/rwkv6*.cpp` |
| RWKV-7 | `GGML_OP_RWKV_WKV7` | varies | Enhanced time-mixing | `src/models/rwkv7.cpp` |
| Standard Transformer | `GGML_OP_MUL_MAT` + attention | KV cache | Full quadratic attention | most model files |
| MoE Transformer | `GGML_OP_MUL_MAT_ID` | KV cache | Sparse expert routing | `src/models/qwen35moe.cpp` etc. |

## Delta-Net Recurrence Math

The Delta-Net SSM computes per-head recurrence (GDA variant):

```
s_dec = s * exp(gate)           // Decay old state
sk = dot(s_dec[j,:], k)         // Project state onto key (per row j)
d_j = (v[j] - sk) * beta       // Innovation: how much v differs from prediction
s_new[j,i] = s_dec[j,i] + k[i] * d_j   // Rank-1 update to state
o[j] = dot(s_new[j,:], q)      // Read output from updated state
```

Key properties:
- **No shared memory needed**: Each thread j computes d_j independently
- **State is [S, S] per head**: Quadratic in state dimension (typically S=128)
- **Gate is scalar per head**: exp(gate) decays the entire state uniformly (GDA mode)
- **Beta is scalar per head**: Controls innovation strength
- **Combined output**: Output tensor packs [o | s_new] with `s_off` offset, following GLA/SSM_SCAN convention

### GDA vs KDA

- **GDA (Gate-Decay-All)**: gate is [1] per head, decays entire state uniformly. Simpler, fuseable.
- **KDA (Key-Decay-Attention)**: gate is [S] per head, different decay per state dimension. More expressive, harder to fuse.
- The fused `GGML_OP_DELTA_NET_RECURRENCE` only handles GDA. KDA falls back to the unfused 13-op implementation.

## GQA Handling in SSM Models

- **GQA is handled by the caller**, not by the recurrence op
- `build_delta_net_autoregressive()` receives q/k already repeated from H_k to H_v
- Inside the op: H_k == H_v always, no GQA mapping needed in shaders
- This simplifies the kernel: each workgroup handles one (head, sequence) pair

## State Management

SSM models carry recurrent state across tokens (unlike transformers which use KV cache):

- State is stored in `ggml_tensor` with shape `[S, S, H, n_seqs]`
- During token generation, previous state feeds into current step as `src[0]`
- Output tensor combines output and new state: `ne = {S*H, n_seqs*(1+S), 1, 1}`
- The caller extracts output via `ggml_view_1d` at offset 0, state via view at `s_off`

## Existing Fusions

| Fusion | What It Does | Architecture |
|--------|-------------|--------------|
| `GGML_OP_SSM_SCAN` | Full Mamba S6/S4 scan (not just recurrence) | Mamba |
| `GGML_OP_GATED_LINEAR_ATTN` | GLA recurrence: s = s*gate + k*v, o = s@q | GLA |
| `GGML_OP_DELTA_NET_RECURRENCE` | Delta-Net GDA recurrence (our new op) | Delta-Net |
| `GGML_OP_RWKV_WKV6/7` | RWKV time-mixing | RWKV |

## View Ops Are No-Ops in Vulkan

`TRANSPOSE`, `PERMUTE`, `RESHAPE`, `VIEW` all return `true` from `ggml_vk_is_empty()` in the Vulkan backend. They just change metadata (strides/shape), no GPU work. This means:
- They don't generate dispatches but DO appear in the graph
- Fusion pattern matching must skip/handle interleaved view ops
- Graph-level fusion is fragile because view ops break contiguous sequences

## Test Models on Hugging Face

Search HF for GGUF models of these architectures:
- **Delta-Net**: Search for `delta-net gguf` or `DeltaNet` in model names
- **Mamba**: `mamba gguf`, `state-spaces/mamba-*`
- **RWKV**: `rwkv gguf`, multiple GGUF conversions available
- **GLA**: `gla gguf`, less common

Use `mcp__claude_ai_Hugging_Face__hub_repo_search` to find models:
```
query: "delta-net gguf"
filter: "pipeline_tag:text-generation"
```

## Key Differences from Transformers

| Aspect | Transformer | SSM/Linear Attention |
|--------|------------|---------------------|
| Memory per token | O(1) new KV, O(n) total KV cache | O(1) fixed state |
| Compute per token | O(n) attention over all past tokens | O(S^2) fixed recurrence |
| Long context | Slower as context grows (KV attention) | Constant speed regardless of context |
| Token gen speed | Dominated by KV cache reads at long context | Dominated by weight reads (constant) |
| Parallelism | Fully parallel in prefill | Sequential in recurrence (limits prefill) |

## Architecture Detection in llama.cpp

The model architecture string in GGUF metadata maps to the implementation:
- `"delta-net"` -> `delta-net-base.cpp`
- `"mamba"` -> `mamba.cpp`
- `"rwkv6"` -> `rwkv6*.cpp`
- `"qwen35moe"` -> `qwen35moe.cpp` (standard MoE transformer, NOT SSM)

Check with: `llama-cli --model <file> --verbose-prompt -n 0` to see which architecture loads.

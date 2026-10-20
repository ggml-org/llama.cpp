# GGML_OP_LIGHTNING_INDEXER

Fused DSA (DeepSeek Sparse Attention) lightning indexer.

Models that use sparse attention first score every past key against the current
query heads, then keep only the top-k keys for the main attention. This op is
that scoring step: it turns the indexer Q, K, head weights, and causal mask into
a dense score tensor of shape `[n_kv, n_tokens]`.

Without the fused op, the same work is a graph of `mul_mat`, `relu`, `mul`,
`sum_rows`, `permute`, and `add`. The fused form avoids those large intermediate
tensors (especially the `[n_kv, n_head, n_tokens]` KQ matrix).

API: `ggml_lightning_indexer()` in `ggml/include/ggml.h`.
Reference: `ggml_compute_forward_lightning_indexer()` in `ggml/src/ggml-cpu/ops.cpp`.

## Inputs and output

| Tensor   | Shape                                      | Type              | Notes |
|----------|--------------------------------------------|-------------------|-------|
| `q`      | `[D, n_head, n_tokens, n_stream]`          | F32               | indexer queries |
| `k`      | `[D, 1, n_kv, n_stream]`                   | any convertible   | indexer keys (one head in the cache) |
| `weights`| `[n_head, n_tokens, 1, n_stream]`          | F32               | **prescaled** head weights |
| `mask`   | `[n_kv, n_tokens, 1, n_mask_stream]`       | F16               | causal / padding mask |
| `dst`    | `[n_kv, n_tokens, 1, n_stream]`            | F32               | indexer scores |

Constraints (from `ggml_lightning_indexer`):

- `q.ne[0] == k.ne[0]` (head dim `D`)
- `k.ne[1] == 1` (single key head in the indexer cache)
- `q.ne[1] == weights.ne[0]` (head count)
- `q.ne[2] == weights.ne[1] == mask.ne[1]` (token / batch count)
- `mask.ne[0] == k.ne[2]` (KV length)
- `weights.ne[2] == mask.ne[2] == 1`
- `q.ne[3] == k.ne[3] == weights.ne[3]`
- mask stream broadcast: `weights.ne[3] % mask.ne[3] == 0`

Typical model values: `D = 128`, `n_head` in `{4, 32, 64}`.

`weights` are expected to already include the indexer scale
`1 / sqrt(D * n_head)` (see model graphs). The op does not scale again.

## Compute math

For each stream `s`, token `t`, and key position `ik`:

```
score[ik, t, s] =
    sum_{h = 0 .. n_head-1}
        ReLU( < q[:, h, t, s] , k[:, 0, ik, s] > ) * weights[h, t, s]
  + mask[ik, t, s % n_mask_stream]
```

Where:

- `<a, b>` is the length-`D` dot product: `sum_d a[d] * b[d]`
- `ReLU(x) = max(x, 0)`
- `mask` is added after the head sum (usually `0` for valid keys, `-inf` for
  masked ones). On backends that use qf32 / similar math that does not preserve
  IEEE inf, implementations must restore `-inf` after the add.

Element-wise (same thing expanded):

```
qk[h, ik, t, s] = sum_{d = 0 .. D-1}  q[d, h, t, s] * k[d, 0, ik, s]

score[ik, t, s] = sum_h  max(qk[h, ik, t, s], 0) * weights[h, t, s]
                + mask[ik, t, s % n_mask_stream]
```

As a small matrix multiply per `(t, s)` (useful for HMX / GEMM thinking):

```
Q_ts : [n_head, D]     = q[:, :, t, s]^T
K_s  : [D, n_kv]       = k[:, 0, :, s]
W_ts : [n_head]        = weights[:, t, s]

S_ts = ReLU(Q_ts @ K_s)        # [n_head, n_kv]
score[:, t, s] = W_ts^T @ S_ts + mask[:, t, ...]
```

So even decode (`n_tokens = 1`) is a GEMM of size `[n_head x D] x [D x n_kv]`,
followed by ReLU, a length-`n_head` weighted sum, and a mask add.

## Unfused equivalent graph

When `cparams.fused_lid` is false, model code builds the same result as:

1. `mul_mat(k, q)` after permutes -> KQ `[n_kv, n_head, n_tokens, ...]`
2. `relu(KQ)`
3. `mul` by `weights` (broadcast over `n_kv`)
4. `sum_rows` over the head dimension
5. permute / cont to `[n_kv, n_tokens, ...]`
6. `add` the KQ mask

The fused op is enabled automatically when a backend reports
`supports_op(LIGHTNING_INDEXER)` (probe `llm_fused_op_lid_probe` in
`src/llama-context.cpp`).

## FLOPs (approx.)

Per output element there are `n_head` dots of length `D`, plus ReLU / mul / add:

```
FLOPs ~= (2 * D * n_head + 2 * n_head + 1) * n_kv * n_tokens * n_stream
```

(`test-backend-ops` uses `((2 * hsk + 2) * nh + 1) * kv * nb * ns`.)

## Role in the model

After this op, the graph usually runs `top_k` on `score` and gathers the
selected keys/values for the main attention. The indexer KV cache is separate
from the main attention cache and stores one key head per layer (`type_k`).

Used by DSA-style models in `src/models/` (e.g. deepseek32, glm-dsa, hy-v4,
dots3note, deepseek4, and related).

## Backend notes

- CPU: any `k` type with a `to_float` converter (or F32).
- Hexagon: F16 and Q8_0 `k` only. HVX for decode / small tiles; HMX for prefill
  when `n_hmx > 0`, `D % 32 == 0`, even `n_head`, and enough rows (`n_tok >= 8`,
  `n_tok * n_head >= 32`).
- Other backends (Metal, Vulkan, CUDA, ...) may support additional `k` types;
  see each backend's `supports_op` / pipeline tables.

## Hexagon notes (profile + HMX)

Traces on SM8650 (`--hex-profile 3`), HVX path:

| Stage | Decode nh=4 | Decode nh=64 | Prefill nh=64 (HVX) |
|-------|-------------|--------------|---------------------|
| DMA | ~35-45% | ~17-24% | ~5% |
| HVX_A_PREP (Q) | ~4% | ~15% | ~22% |
| HVX_W_DEQUANT + W_PREP | ~14% (Q8_0) | ~7% | ~2% |
| HVX_COMP | ~9% | ~45% | ~72% |

HMX prefill (SM8650, vs CPU `us/run`, F16 unless noted):

| Shape | HTP | CPU | speedup |
|-------|-----|-----|---------|
| nh=4, kv=256, nb=512 | 0.42 ms | 5.8 ms | ~14x |
| nh=32, kv=256, nb=512 | 2.3 ms | 22 ms | ~9x |
| nh=64, kv=256, nb=512 | 4.5 ms | 40 ms | ~9x |
| nh=4, kv=4096, nb=512 | 3.8 ms | 90 ms | ~23x |
| nh=32, kv=4096, nb=512 | 18 ms | 330 ms | ~18x |

Takeaways:

- Prefill is compute-bound on HVX; HMX QK tiles (reuse FA `hmx_fa_qk_dot_tile`)
  plus HVX ReLU*w+mask reduce are the win. Tile buffers must stay 2KB-aligned
  (pad VTCM after odd `n_tok` weight regions).
- Decode stays on HVX (small `n_tok`). It is compute-bound (DMA overlaps): about
  half K transpose, half the dot loop.
- Q8_0 decode transposes K by `vgather` straight from the raw rows (one halfword
  gather = `q[d], q[d+1]` of 64 rows; block scales gathered the same way), then
  widens and scales in HVX. This replaced per-row dequant + shuffle transpose:
  Q8_0 decode went from ~1.6x to ~1.1x the F16 time.
- Tried and dropped for decode: scales applied to per-block partial dots (breaks
  the accumulator pipeline), scalar scale gather (VTCM scalar access is slow),
  overlapping next-block gathers with compute (gathers stall on issue and
  contend with compute, ~5%). `vdmpy` (sf += hf*hf) is v79+ only.
- Next decode lever: the dot loop runs ~12 packets per d step (5 loads, 4 vmpy,
  8 qf32 adds); reusing each Q splat load across two kv blocks should help both
  F16 and Q8_0.
- Short-KV head-split and splat-in-loop were tried and lost; left off.

## Source pointers

- API / shapes: `ggml/include/ggml.h` (`ggml_lightning_indexer`)
- Shape asserts: `ggml/src/ggml.c`
- CPU reference: `ggml/src/ggml-cpu/ops.cpp`
- Tests: `tests/test-backend-ops.cpp` (`test_lightning_indexer`)
- Fused enable: `src/llama-context.cpp` (`llm_fused_op_lid_probe`)
- Example graph: `src/models/deepseek32.cpp` (fused vs unfused branches)

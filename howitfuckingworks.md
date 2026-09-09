# How It Works: Weight-Tied Recurrent Transformer Layers in llama.cpp

This document describes the inference-time recurrence implemented in the `llamar.cpp` fork: its rationale, the exact equations executed, the graph construction in the model builders, and the stability properties of the scheme. It is the authoritative reference for the recurrence mechanism; the README is the quick overview.

## 1. Problem statement

A standard transformer applies every layer exactly once per token. The effective depth — and therefore the reasoning capacity — of a single forward pass is fixed by the architecture, independent of token difficulty. The usual workaround is to spend reasoning effort in *context space*: the model emits many intermediate tokens, which grow the KV cache quadratically in the number of such tokens and increase latency linearly.

The alternative explored here is to spend effort in *latent space*: apply a subset of the network more than once per token, without allocating any new parameters and without feeding additional tokens to the attention cache. Weight-tied recurrence reuses one existing layer as the "reasoning core" and iterates it, so the effective depth of that stage becomes an input-dependent hyperparameter.

## 2. Design overview

The network of `N` layers is split into three sequential stages for each token:

```
Layers:   [ 0 .. L-1 ]        [ L ] recurred T times          [ L+1 .. N-1 ]
Stage:    prelude (1 pass)    recurrent core (T passes)       coda (1 pass)
```

- **Prelude** decodes the token through the lower layers once. Its output `e` is the **frozen anchor**: computed once per token and kept constant for the whole recurrence.
- **Recurrent core** is a *single* layer `L` applied `T` times. Each application receives `h_t + e` (the sum of the previous recurrent state and the anchor), and the state is updated by a linear time-invariant (LTI) rule.
- **Coda** decodes the final recurrent state through the upper layers once, then a normalisation and the LM head produce logits.

Choosing `L` near 38% of the model depth preserves the lower layers (positional and syntactic grounding) and the upper layers (logit calibration) as single-pass.

## 3. Mathematics

Let `F_l(h)` be the forward function of layer `l`. For the architectures involved `F_l` is a residual block; for example, an attention–SSM–FFN hybrid layer computes

```
F_l(h) = h + FFN(RMSNorm( Attn(RMSNorm(h)) + SSM(RMSNorm(h)) ))
```

where the SSM term is present only for hybrid architectures (Falcon-H1, Qwen3.5 DeltaNet). Define `L = RECURRENT_LAYER` and `T = RECURRENT_T`.

### 3.1 Prelude

```
h_0  = embed(x)
e    = F_{L-1}( … F_1(F_0(h_0)) … )            // anchor, frozen
```

### 3.2 Recurrent core

Initialise the recurrent state to the anchor, `h = e`. For `t = 0 … T-1`:

```
combined  = h + e                               // anchor injection into input
block_out = F_L(combined)                       // weight-tied pass through core
h         = A·h + B·e + block_out               // LTI state update
```

With `A = RECURRENT_A` and `B = RECURRENT_B`. The same tensor `F_L` (attention weights, SSM, FFN) is reused for every pass — hence *weight-tied*.

### 3.3 Coda and output

```
c   = F_{N-1}( … F_{L+2}(F_{L+1}(h)) … )
out = LMHead(RMSNorm(c))
```

### 3.4 Relation to OpenMythos-style recurrence

The update is the discrete version of a continuous-time state-space increment:

```
ḣ = (A-1)·h + (B+1)·e + F_L(h + e)      // continuous interpretation
```

The `A·h + B·e` term is a linear time-invariant (LTI) system in the sense of a scalar state: constant, input-independent coefficients, no nonlinearity. It is the minimal construction that simultaneously

1. bounds the state norm (decay), and
2. preserves the decoded input semantics (anchor re-injection).

All nonlinear "thinking" happens inside `F_L`; the LTI term only stabilises and carries memory.

## 4. Stability analysis

### 4.1 Contraction

Ignoring the bounded activation of `F_L` (a residual block whose norm is bounded for well-trained models), the homogeneous part of the recurrence is

```
h_{t+1} = A · h_t
```

Its spectral radius is `ρ(A) = |A|`. For `|A| < 1` (default `A = 0.90`) the homogeneous component decays geometrically:

```
|A^t · h_0| ≤ |A|^t · |h_0|  →  0
```

The recurrence is therefore a contraction with respect to the anchor. This makes the iteration unconditionally stable for any `T`, without needing the harmonic-decay or exit-damping schedules of earlier recurrence designs.

### 4.2 Anchor conservation

The scalar `B` controls how strongly the anchor is re-injected each step. Setting `B = 1 - A` (default `0.10` with the default `A = 0.90`) makes the LTI term `A·h + B·e` a convex combination of the current state and the anchor when `block_out` is ignored, fully conserving the decoded context weight. This prevents the "drift to a dominant eigenvector" failure mode: even at large `T`, the input to the core is never far from `e`.

### 4.3 KV cache locality

Because the core layer index is constant across passes, the attention of every pass writes to the *same* KV rows for the current token's position (in the last pass the cache must retain the representation, but intermediate passes may freely overwrite). The graph reuses one `build_layer`/decoder call; native SSM per-layer state is recomputed fresh each pass, while the LTI state `h` is the only cross-pass memory.

### 4.4 Fixed point

A fixed point of the loop satisfies, with the same `F_L` applied,

```
h* = A·h* + B·e + F_L(h* + e)   ⟺   F_L(h* + e) = (1-A)·h* - B·e
```

For `B = 1 - A` this becomes `F_L(h* + e) = (1-A)·(h* - e)`: the layer output at a fixed point compensates the decaying residual exactly. The contraction bound above guarantees the iteration approaches a neighbourhood of such a point.

## 5. Graph construction (exact code path)

The canonical implementation is `src/models/falcon-h1.cpp`; `src/models/qwen2.cpp`, `src/models/qwen2vl.cpp` and `src/models/qwen35.cpp` follow the identical structure.

### 5.1 Configuration (per builder)

```cpp
const int   RECURRENT_T   = [] { const char * v = std::getenv("RECURRENT_T");   return v ? std::atoi(v) : 1; }();
const float RECURRENT_A   = [] { const char * v = std::getenv("RECURRENT_A");   return v ? std::atof(v) : 0.90f; }();
const float RECURRENT_B   = [] { const char * v = std::getenv("RECURRENT_B");   return v ? std::atof(v) : 0.10f; }();
const int   n_rec_layer   = [] { const char * v = std::getenv("RECURRENT_LAYER"); return v ? std::atoi(v) : -1; }();
```

### 5.2 Decoder lambda

A single decoder closure takes a layer index and an input tensor and returns the layer output (SSM, attention, aggregation, FFN). For the recurrent core this closure is invoked `T` times with the *same* `il`, which is exactly what makes the layer weight-tied in the compute graph. The SSM cache entry for the core layer is reset before each pass so the native recurrent state does not carry stale activations across passes.

### 5.3 Control flow

```cpp
if (RECURRENT_T > 1 && n_rec_layer >= 0 && n_rec_layer < n_layer) {
    // 1. Prelude
    for (int il = 0; il < n_rec_layer; ++il) inpL = falcon_decoder(il, inpL);

    // 2. Freeze anchor
    ggml_tensor * anchor_e = inpL;

    // 3. Weight-tied loop (LTI update)
    ggml_tensor * h = inpL;
    for (int t = 0; t < RECURRENT_T; ++t) {
        ggml_tensor * combined  = ggml_add(ctx0, h, anchor_e);
        ggml_tensor * block_out = falcon_decoder(n_rec_layer, combined);
        h = ggml_add(ctx0,
                     ggml_add(ctx0,
                              ggml_scale(ctx0, h, RECURRENT_A),
                              ggml_scale(ctx0, anchor_e, RECURRENT_B)),
                     block_out);
    }
    inpL = h;

    // 4. Coda
    for (int il = n_rec_layer + 1; il < n_layer; ++il) inpL = falcon_decoder(il, inpL);
} else {
    // Vanilla: single pass over all layers (exact upstream behavior)
    for (int il = 0; il < n_layer; ++il) inpL = falcon_decoder(il, inpL);
}
```

The vanilla branch is taken whenever `RECURRENT_T = 1` (the default) or `RECURRENT_LAYER` is unset, so default inference is graph-identical to upstream.

## 6. Empirical status

Measured on Falcon-H1R-7B-IQ4_XS with `llama-cli`:

| Configuration | Prompt evaluation (t/s) | Generation (t/s) |
|---------------|-------------------------|------------------|
| Baseline (`RECURRENT_T=1`) | 149.5 | 18.5 |
| `RECURRENT_T=3 RECURRENT_LAYER=16` | 111.4 | 15.6 |

The recurrent path runs without crash or numerical divergence at `T = 3`; throughput drops in proportion to the extra core passes. Accuracy evaluation (GSM8K chain-of-thought, MBPP unit tests) against an unmodified upstream baseline is pending and will be published in a separate benchmark report.

## 7. Known limitations

- **Latency vs. reasoning trade-off.** Each pass adds a full layer execution; total cost scales linearly with `T`.
- **Single layer core.** The current scheme recurses one layer. Recurring a contiguous *block* of layers would multiply compute proportionally and is deliberately avoided in this build.
- **SSM hybrid layers.** For architectures with native SSM state, only the LTI scalar state `h` carries memory across passes; the native SSM is re-evaluated per pass. This is a deliberate, documented choice, not a latent bug.
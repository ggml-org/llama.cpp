# dory port — work contract for the parallel build

Read `docs/dory-arch-spec.md` first: it holds the algorithm, confirmed against
`transformers/models/dory` (not inferred from tensor shapes). This file is the
**ownership and acceptance** contract so several agents can work at once without
stepping on each other.

## Acceptance — what "done" means (no partial credit)

A build that merely LOADS the GGUF is not done. Done is:

1. `cmake --build build --target llama-cli` succeeds with no new warnings that
   are errors, AND
2. `llama-cli -m <gguf> -p "<prompt>" -n 24 --temp 0` produces the token ids in
   `/Volumes/Widekeep/oracle/oracle-dory.json`, in order.

The oracle was produced by the reference implementation:

```
prompt          "The capital of France is"
prompt_ids      [1784, 8961, 1307, 5498, 1395]
generated_ids   [6993, 1046, 1531, 8961, 1307, 10787, 1395, 10874, 15421, …] (24 total in the file)
generated_text  " Paris. The capital of Germany is Berlin. The capital of Italy is Rome. …"
```

Model: `/Volumes/Safehold/models/dory-1.0-4b/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`
Reference weights (for re-deriving the oracle if ever needed):
`/Volumes/Widekeep/oracle/dory-bf16` (BF16, via the isolated venv
`/Volumes/Widekeep/oracle/venv`; the flock's transformers must stay untouched).

**If the ids differ, the port is wrong.** Do not relax the check, do not
compare text, do not accept "plausible". Report the first differing position and
the logits you saw there.

## File ownership — one owner per path, strictly

| owner | paths | notes |
|---|---|---|
| **A: registration** | `src/llama-arch.h`, `src/llama-arch.cpp`, `src/llama-model.cpp`, `src/llama-hparams.h`, `src/llama-hparams.cpp` | `LLM_ARCH_DORY` + name; new `LLM_TENSOR_*` enums and their name strings; `LLM_KV_*` for the `dory.*` keys; hparams fields; the dispatch case that constructs the model |
| **B: model** | `src/models/dory.cpp` (**new file, plus its own `#include` wiring if CMake needs it — CMake globs `models/*.cpp`, so normally nothing else**) | tensor loading + the looped graph. Owns NO other file |
| **C: harness** | `scripts/dory_oracle_check.sh` or `.py` (**new file**) | builds nothing; runs the binary against the oracle, prints first-divergence |
| **D: integration** | none | runs the build, triages failures, routes them to A or B |

Templates to copy from, in order of usefulness:
`src/models/inkling2-loop.cpp` (a LOOPED arch — closest relative),
`src/models/ricursive.cpp` (small, odd naming), `src/models/llama.cpp` (baseline).

## The algorithm, condensed (full text in the spec)

```
h = embed(tokens)
for blk in 0..15        : h = block(h)                  # input phase (16 modules)
input_hidden = L2Norm(h)
for loop in 0..n_loops-1:
    h = L2Norm(h) + input_hidden
    for blk in 16..55   : h = block(h)                  # recurrent group (40 modules)
for blk in 56..71       : h = block(h)                  # output phase (16 modules)
h = L2Norm(h); logits = h @ W_out * output_scale
```

A **module** is one physical `blk.N`: even indices are attention, odd are MLP.

```
L2Norm(x)        = x * rsqrt(sum(x^2) + 1e-6)            # unit sphere, NOT rms
attention: q,k   = L2Norm over head_dim, then RoPE(theta 1e4 sliding / 1e7 full)
           q     = q * square(attn_qk_scale)
           g     = sigmoid(square(gate_proj(x) * attn_gate_scale))
           out   = o_proj( attn_sub_norm( attn(q,k,v) * g ) )
mlp:       gate * ffn_gate_scale ; up * ffn_up_scale ; down( norm( silu(gate) * up ) )
residual:  rate = abs(residual_alpha)                    # per-hidden-dim vector
           h = h + rate * (f(norm(h)) - h)
```

GGUF metadata: `dory.block_count 72`, `dory.input_layer_count 16`,
`dory.recurrent_layer_count 40`, `dory.output_layer_count 16`,
`dory.num_loops 4`, `dory.attention.sliding_window 1024`,
`dory.rope.freq_base 1e7`, `dory.rope.freq_base_2 1e4`,
`dory.recurrent_kv_cache_mode per_loop`, `dory.alpha_init 0.05`.
The HF reference halves these (72→36 etc.) because it counts attn+MLP pairs;
**llama.cpp sees 72 modules, so use the GGUF numbers directly.**

Two `rope.freq_base`s means per-layer-type theta: sliding layers use
`freq_base_2 = 1e4`, full-attention layers use `freq_base = 1e7`. The
`layer_pattern` string (S / *) says which is which per module.

## Rules

* Never edit a path you do not own; if you need a change there, report it.
* Compile before reporting: `cmake --build build --target llama-cli -j8`.
  A build takes minutes; run it, do not guess.
* Never claim the oracle matched unless you ran it and pasted the ids.
* If a design decision is genuinely ambiguous (e.g. which norm applies inside
  the attention sub-block), say so and cite the line of
  `transformers/models/dory/modeling_dory.py` — the fetched copy is at
  `/tmp/modeling_dory.py` on this host.

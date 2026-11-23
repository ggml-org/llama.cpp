# `dory` architecture — spec captured from the GGUF, implementation plan

Status: **specification only**. No forward pass yet. Branch `dory-arch` (off
`origin/master` @ c212d25ad). Written 2026-11-23 from the shipped file
`NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` (3,673,288,864 B, 507 tensors,
44 metadata keys), because the reference implementation is not public: the
transformers repo `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B` returns HTTP 401 and
the GGUF repo (public, HTTP 200) ships only the .gguf, a README, an
architecture PNG and `explainability.md`.

## What the model is

NVIDIA **Dory1.0-4B — "Looped (latent-recurrent) normalized-GPT"**, per the
model card:

* layer stack in three groups — **Input 8 blocks / Recurrent 20 blocks (shared,
  applied `num_loops` times) / Output 8 blocks**, input reinjected every loop;
* interleaved **sliding-window attention (SWA)** and **full attention**;
* nGPT backbone → learned per-layer scales (hence the `*_scale` tensors);
* last-loop ("per_loop") KV caching so loop count costs no cache memory;
* 256K context; natively NVFP4-pretrained, shipped here as Q5_K_M.

## Metadata (verbatim, `dory.*`)

```
block_count = 72            input_layer_count = 16
num_loops = 4               recurrent_layer_count = 40
embedding_length = 2560     output_layer_count = 16
feed_forward_length = 7168  attention.head_count = 16
context_length = 262144     attention.head_count_kv = 4
rope.dimension_count = 256  attention.key_length = 256
rope.freq_base = 1e7        attention.value_length = 256
rope.freq_base_2 = 1e4      attention.sliding_window = 1024
rope.profile = [2,1,2,1, … 72 entries]
layer_pattern = S-S-S-*-S-S-S-*-… (72 entries; S = SWA, * = full)
alpha_init = 0.05  gate_init = 50.596…  init_std = 0.0177
recurrent_kv_cache_mode = per_loop
```

**The counting trap:** `block_count = 72` counts *physical* layers, and the
tensors alternate attention / FFN — `blk.0`, `blk.2`, … carry attention,
`blk.1`, `blk.3`, … carry FFN. The README's "8 / 20 / 8 blocks" is half of
16 / 40 / 16 physical layers. So the recurrent group is **40 physical layers =
20 (attn+ffn) pairs**, applied `num_loops = 4` times.

## Per-layer tensors

Attention layers (`blk.2k`):
```
attn_q.weight      [2560, 4096]   attn_k.weight        [2560, 1024]
attn_v.weight      [2560, 1024]   attn_output.weight   [4096, 2560]
attn_gate.weight   [2560, 4096]   attn_qk_scale.weight [4096]
attn_gate_scale.weight [4096]     residual_alpha.weight [2560]
```
FFN layers (`blk.2k+1`):
```
ffn_gate.weight [2560, 7168]   ffn_up.weight  [2560, 7168]
ffn_down.weight [7168, 2560]   ffn_gate_scale.weight [7168]
ffn_up_scale.weight [7168]     residual_alpha.weight [2560]
```
Globals: `token_embd.weight [2560, 131072]`, `output.weight [2560, 131072]`,
`output_scale.weight [131072]`.

## Implementation plan for llama.cpp

1. `src/llama-arch.cpp`: `LLM_ARCH_DORY` + arch name `dory`; tensor-name
   mappings for the eight attention suffixes, six FFN suffixes and the three
   globals above; metadata keys `dory.block_count`, `dory.num_loops`,
   `dory.input_layer_count`, `dory.recurrent_layer_count`,
   `dory.output_layer_count`, `dory.attention.sliding_window`,
   `dory.rope.profile`, `dory.rope.freq_base_2`,
   `dory.recurrent_kv_cache_mode`, `dory.alpha_init`, `dory.gate_init`.
2. `src/llama-model.cpp`: a looped graph builder — build layers `[0,16)` once,
   then the recurrent region `[16,56)` `num_loops` times with the hidden state
   reinjected (and a per-loop KV slot when `per_loop`), then `[56,72)` once.
   Interleaved attn/ffn means each physical index is one op, not a block.
3. nGPT semantics — **the part that must be matched, not guessed**: how
   `residual_alpha` scales the residual add, the order of `attn_qk_scale`
   (pre-softmax QK normalisation) and `attn_gate_scale`, whether `ffn_*_scale`
   are applied pre- or post-activation, and the `output_scale` over the logits
   (131072 = vocab). The tensor *shapes* pin the dimensionality but not the
   order of operations.
4. Dual RoPE: `rope.profile = [2,1,2,1,…]` over `dimension_count = 256` with two
   base frequencies (1e7 / 1e4) is a per-dimension-pair split — this needs the
   reference to confirm which pairs take which base.
5. `llama.cpp` has no `dory`, and `origin/master` has no `dory` either (checked
   at c212d25ad); the closest existing arches are `nemotron_h` /
   `nemotron_h_moe`, which do not carry `residual_alpha`/`*_scale`/`attn_gate`.

## What is needed before a certification means anything

A loader that accepts the file would emit a receipt, and the receipt would be
worthless: the queue's pass criterion is a coherent chat reply, and a wrong op
order can produce fluent-looking garbage. Three ways to settle it, in order of
cost:

1. **Get `config.json` + the model class** from the (gated) base repo or from
   NVIDIA — that pins steps 3 and 4 exactly.
2. **Reference logits**: run the model in its native implementation on a fixed
   prompt and store `argmax` per position; the llama.cpp port is correct when
   its logits match. This is the only mechanical proof.
3. Fall back to `dory_arch.png` + `explainability.md` in the GGUF repo and
   accept that the port is a hypothesis until (1) or (2) arrives.

Until one of those exists, this branch carries the specification and the
tensor/metadata mapping — deliberately not a guessed forward pass.

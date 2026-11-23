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

---

# CONFIRMED from the reference (2026-11-23, later the same day)

The block above says the spec was blocked on a 401. That was a WRONG REPO NAME,
not a gate. The public repository is `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`
(config.json answers 200); the bare `...-4B` name does not exist. The reference
implementation ships in **transformers itself** (no `trust_remote_code`), so
`src/transformers/models/dory/modeling_dory.py` (653 lines) +
`configuration_dory.py` (130 lines) are readable and authoritative. Local
transformers 5.3.0 predates `models/dory`, so the source was read from
`raw.githubusercontent.com/huggingface/transformers/main/...`.

## config.json (public, verbatim values that matter)

```
architectures = ["DoryForCausalLM"]   model_type = "dory"
hidden_size 2560   intermediate_size 7168   num_hidden_layers 72
num_attention_heads 16   num_key_value_heads 4   head_dim 256
n_input_layers 16   n_recurrent_layers 40   n_output_layers 16
n_recurrent_loops 2 (runtime-elastic; the GGUF was exported with 4)
recurrent_kv_cache_mode "per_loop"    sliding_window 1024
ngpt_alpha_init 0.05   initializer_range 0.0177   hidden_act silu
layer_types = 72 entries, alternating ("sliding_attention"|"full_attention", "mlp"),
              full_attention every 4th attention slot
rope_parameters.sliding_attention.rope_theta = 1e4
rope_parameters.full_attention.rope_theta    = 1e7
```

So the 72 `blk.N` in the GGUF are 72 PHYSICAL MODULES: even = attention, odd =
MLP. 16 + 40 + 16 = 72 is a module count; the README's 8 / 20 / 8 counts
(attn+MLP) pairs. Both statements describe the same stack.

## The forward pass, as the reference implements it

nGPT normalisation is a **unit-sphere** projection, not RMS:
`L2Norm(x) = x * rsqrt(sum(x²) + 1e-6)`; `eps = 1e-6`.

A module (attention or MLP) is an **interpolation toward its output**, never a
plain residual add:

```
rate = abs(lambda * lambda_scale)          # per-layer scalar x per-dim vector
h = h + rate * (f(norm(h)) - h)
```
For attention modules `lambda`/`lambda_scale` come from `residual_alpha`
(2560 values, one per hidden dim); the MLP module has its own.

Attention (`DoryAttention.forward`):
```
q = q_proj(x) -> q_norm  (L2 over head_dim) -> RoPE(theta 1e4 for sliding, 1e7 for full)
k = k_proj(x) -> k_norm  (L2)                -> RoPE(same profile)
v = v_proj(x)
q = q * square(sqk * sqk_scale)            # gguf: attn_qk_scale.weight[4096] = 16*256, folded
attn = softmax(q k^T * scaling + mask) v   # mask = sliding window 1024 or full
g = sigmoid( square(gate_proj(x) * (s_gate * s_gate_scale)) )
out = o_proj( attn_sub_norm( attn * g ) )  # gguf: attn_gate.weight[2560,4096], attn_gate_scale[4096]
```
Note the square: the qk scale and the gate scale are applied SQUARED, and the
gate is a sigmoid of that square. Both GGUF tensors are already the folded
product (`s* *_scale`), so the port multiplies them once and squares.

MLP (`DoryMLP.forward`) — SwiGLU with a scale on EACH branch:
```
gate = gate_proj(x) * (suv_gate * suv_scale)   # gguf: ffn_gate_scale[7168]
up   = up_proj(x)   * (suv_up   * suv_scale)   # gguf: ffn_up_scale[7168]
out  = down_proj( ffn_sub_norm( silu(gate) * up ) )
```
LM head: `logits = (h @ W).float() * (logit_scale * factor)` — matches
`output_scale.weight [131072]`, one scalar per vocab entry.

## The loop (DoryModel.forward)

```
h = embed(input)
for layer in layers[0:16]:        h = layer(h)          # input phase, ONCE
input_hidden = L2Norm(h)
for loop_idx in range(n_loops):
    h = L2Norm(h) + input_hidden                        # reinjection, every loop
    for layer in layers[16:56]:   h = layer(h)          # the shared recurrent group
for layer in layers[56:72]:       h = layer(h)          # output phase, ONCE
h = L2Norm(h)
```
`per_loop` caching: loops 1..n-1 write into deep copies of the recurrent
layers' caches (`cache.layers += [copy.deepcopy(c) for c in recurrent_cache *
(n_loops - 1)]`); `last_loop` instead appends `DoryPastReader` wrappers so every
loop before the last reads without writing, i.e. one KV set regardless of loop
count.

## What this changes for the port

The remaining risk is gone: op order is no longer a hypothesis. The work is now
mechanical but large — register the arch and 17 tensor kinds, implement the two
nGPT ops (`L2Norm`, interpolated residual with per-dim rate), one attention
builder with the gate and squared qk scale, one scaled-SwiGLU builder, and a
graph loop over `[16,56)` with reinjection plus the per-loop KV offset. A
validation harness still needs reference logits from `DoryForCausalLM` on a
fixed prompt; that is now cheap to produce (`pip install transformers>=…` with
`models/dory`, load the BF16 repo, greedy-decode 32 tokens, store token ids).

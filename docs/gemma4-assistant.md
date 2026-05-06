# Gemma 4 Multi-Token Prediction (MTP) drafter

This page documents the `gemma4-assistant` GGUF architecture, which is the
file-format counterpart to Google's official Multi-Token Prediction (MTP)
drafters released alongside the Gemma 4 family on 2026-05-05.

- Announcement: <https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/>
- Hugging Face card (E2B): <https://huggingface.co/google/gemma-4-E2B-it-assistant>
- Hugging Face card (E4B): <https://huggingface.co/google/gemma-4-E4B-it-assistant>
- Reference impl: <https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4_assistant>

## What it is

Each Gemma 4 model ships with a paired ~78 M-param "assistant" — a 4-layer
transformer that proposes several tokens ahead of the target model. The
target then verifies the proposal in parallel, the same way as classical
speculative decoding ([Leviathan et al. 2022](https://arxiv.org/abs/2211.17192)),
but with an architecture co-designed with the target backbone:

- **Q-only attention.** The drafter does not have its own K/V projections;
  every layer reads K/V from the target backbone's last layer of the matching
  attention type (one entry for full-attention, one for sliding-window).
- **2× backbone input.** The forward signature is
  `[target_last_hidden, embed(next_token)]` projected through `pre_projection`
  to the small assistant hidden size (256 for both released drafters).
- **Centroid-clustered output head.** Instead of dense logits over the full
  262 144-row vocabulary, the assistant scores 2 048 centroids, picks the top
  32, and only computes fine-grained logits inside those clusters. The chosen
  positions are scattered into the vocab; everything else is masked.
- **Per-layer scalar gate.** A single `layer_scalar` weight per layer rescales
  the residual stream — same shape as the Gemma 4 backbone uses.

## Status of this implementation

| Capability | Status |
|---|---|
| Convert Hugging Face checkpoint → GGUF | ✅ |
| `gguf-dump` inspection | ✅ |
| Model load (`llama_model_load_from_file`) | ✅ |
| Forward pass / `llama_decode` | ⛔ aborts with a clear message |
| Speculative-decoding integration | ⛔ follow-up PR |

The runtime path is intentionally gated: completing it requires plumbing the
target context's last-layer K/V tensors into the drafter context, which is a
new piece of public API surface. Landing the model architecture first lets
the GGUF format stabilise so the ecosystem (Ollama, LM Studio, the
quantization tools, etc.) can ingest the released checkpoints while the
runtime piece is reviewed separately.

## Conversion

```sh
# Download the assistant checkpoint (E2B; swap for `E4B-it-assistant` to get the
# E4B drafter). Both are ~158 MB at bf16.
hf download google/gemma-4-E2B-it-assistant --local-dir gemma-4-E2B-it-assistant

python3 convert_hf_to_gguf.py gemma-4-E2B-it-assistant \
    --outfile gemma-4-E2B-it-assistant.gguf \
    --outtype bf16
```

The converter is registered against the `Gemma4AssistantForCausalLM`
architecture, mirrors the `gemma4` tokenizer (262 144 tokens) and writes the
following metadata:

- `gemma4-assistant.backbone_hidden_size` — 1536 for E2B, 2560 for E4B.
- `gemma4-assistant.assistant.num_centroids` — 2048.
- `gemma4-assistant.assistant.centroid_intermediate_top_k` — 32.
- `gemma4-assistant.assistant.use_ordered_embeddings` — `true`.

Tensor schema (per-block tensors are written for `blk.0..3`):

| GGUF tensor | Shape | Source name |
|---|---|---|
| `token_embd.weight` | `[hidden, vocab]` | `model.embed_tokens.weight` |
| `output_norm.weight` | `[hidden]` | `model.norm.weight` |
| `assist_pre_proj.weight` | `[2*backbone, hidden]` | `pre_projection.weight` |
| `assist_post_proj.weight` | `[hidden, backbone]` | `post_projection.weight` |
| `assist_embed_centroids.weight` | `[hidden, num_centroids]` | `masked_embedding.centroids.weight` |
| `assist_token_ordering.weight` | `[vocab]` (F32, integer values) | `masked_embedding.token_ordering` |
| `blk.{i}.attn_q.weight` | `[hidden, n_head*head_dim]` | `model.layers.{i}.self_attn.q_proj.weight` |
| `blk.{i}.attn_q_norm.weight` | `[head_dim]` | `model.layers.{i}.self_attn.q_norm.weight` |
| `blk.{i}.attn_output.weight` | `[n_head*head_dim, hidden]` | `model.layers.{i}.self_attn.o_proj.weight` |
| `blk.{i}.attn_norm.weight` | `[hidden]` | `model.layers.{i}.input_layernorm.weight` |
| `blk.{i}.post_attention_norm.weight` | `[hidden]` | `model.layers.{i}.post_attention_layernorm.weight` |
| `blk.{i}.ffn_norm.weight` | `[hidden]` | `model.layers.{i}.pre_feedforward_layernorm.weight` |
| `blk.{i}.post_ffw_norm.weight` | `[hidden]` | `model.layers.{i}.post_feedforward_layernorm.weight` |
| `blk.{i}.ffn_gate.weight` | `[hidden, intermediate]` | `model.layers.{i}.mlp.gate_proj.weight` |
| `blk.{i}.ffn_up.weight` | `[hidden, intermediate]` | `model.layers.{i}.mlp.up_proj.weight` |
| `blk.{i}.ffn_down.weight` | `[intermediate, hidden]` | `model.layers.{i}.mlp.down_proj.weight` |
| `blk.{i}.layer_output_scale.weight` | `[1]` | `model.layers.{i}.layer_scalar` |

`token_ordering` keeps F32 storage because the converter pipeline force-stores
1-D tensors as F32; every value is below 2^18 so the cast is lossless.

## Variants supported

- **E2B**: 4 layers, hidden 256, KV-heads 1, backbone 1536. ~78 M params.
- **E4B**: 4 layers, hidden 256, KV-heads 2, backbone 2560. ~78 M params.
- 26B-A4B and 31B drafters use the same architecture and convert with the
  same command (only the backbone size changes); they are not exercised by
  this PR but should work — please file an issue if not.

## Why the runtime is staged

Implementing decoding faithfully needs three pieces that don't exist yet
in llama.cpp:

1. **Cross-context K/V exposure.** Reading the target context's last-layer
   K/V at decode time and passing those tensors as graph inputs to the
   drafter context. The transformers reference takes them as the
   `shared_kv_states` keyword argument of the drafter's `forward`.
2. **Bidirectional attention with external K/V.** Each drafter layer attends
   over the target's existing tokens (causal-by-construction since they're
   already decoded), with a flipped sliding window. ggml has no `flip` op
   yet — we'll most likely materialise the index map at graph build time.
3. **Centroid masked logits.** A `top_k` + `get_rows` + scatter combination
   over the LM head. ggml's existing `top_k` covers the first step; the
   scatter requires either a dense fill plus `set_rows`, or a new graph op.

These will be addressed in a follow-up PR so they can be reviewed apart
from the data-format changes here. The architecture-level guard
(`build_arch_graph` aborts with a pointer back to this document) prevents
silently-wrong logits in the meantime.

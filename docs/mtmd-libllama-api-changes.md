# API and ABI changes of the mtmd-in-libllama prototype

Branch `mtmd-in-libllama`, based on master `e8fbcad85`. This file lists every API, ABI, behaviour and format change the prototype makes, so a reviewer can judge the cost of the approach. It is updated with each milestone.

Status legend: **done** (in the branch and tested), **planned** (part of the accepted plan, not written yet).

## Summary

| Surface | Change | ABI effect |
|---|---|---|
| `include/llama.h` | none | none |
| `src/llama-ext.h` (staging, C++) | new media API, see below | new exported symbols only |
| `tools/mtmd/mtmd.h` | one transitional function | new exported symbol only |
| GGUF | none (mmproj files load as they are) | - |
| internal structs (`llama_ubatch`, `llama_batch_ext`, `llama_batch_allocr`, `llama_model`, `llm_ffn_op_type`, `LLM_KV`) | new fields and values | internal, not part of any ABI |

No existing function changed its signature or meaning. No public struct changed its layout.

## `src/llama-ext.h` (staging API)

All new functions are `LLAMA_API` and exported from libllama. The header is C++ and explicitly unstable.

### Context role

| Item | Status | Notes |
|---|---|---|
| `LLAMA_CONTEXT_TYPE_ENCODER` | done | `static const enum llama_context_type` with value 16, outside the enum for now. When promoted, it becomes a member of `enum llama_context_type` in `llama.h`; adding an enum value does not change the ABI. |
| `llama_model_has_context_type(model, type)` | done | mmproj models report ENCODER only (when their projector is ported); text models report DEFAULT, and MTP when they have nextn layers. |

Rules enforced by `llama_init_from_model`: an mmproj (`general.architecture = clip`) model needs an ENCODER context, and an ENCODER context needs an mmproj model. An ENCODER context always has `embeddings = true`, non-causal attention and no pooling, whatever the params say. Its `n_ubatch` bounds the output rows of one media and sets the compute reservation.

### Preprocessing and layout

| Function | Status |
|---|---|
| `llama_media_proc_default_params`, `llama_media_proc_init(ctx_enc, model_dec, params)`, `llama_media_proc_free`, `llama_media_proc_has`, `llama_media_proc_sample_rate` | done (images; `sample_rate` returns -1 until audio is ported) |
| `llama_media_layout_init(proc, pos_0)`, `_free`, `_add_tokens`, `_add_image`, `_add_frames`, `_n_pieces`, `_get_tokens`, `_get_media` | done (`_add_frames` lands with the Qwen-VL milestone) |

Difference from the accepted plan (`llama-ext-plan.md`):
- `llama_media_proc_init` takes the decoder **model**, not the decoder context. The only decoder-context value the proc needed was `n_ubatch`, which the caller passes as `params.n_span_max`. This lets libmtmd, which receives only the text model, create a proc.
- The video builder (`video_begin` / `add_video_frame` / `add_video_audio` / `video_end`) is replaced for now by `llama_media_layout_add_frames(lay, nx, ny, rgb0, rgb1, id)`, one call per frame pair that the model merges. This is what Qwen-VL needs; the full video builder stays planned.
- `llama_media_proc_video_hint` is not written yet (planned).

### Media handle

| Function | Status |
|---|---|
| `llama_media_get_type`, `_n_tokens`, `_n_pos`, `_is_atomic`, `_get_id`, `_is_placeholder`, `_is_encoded`, `_ref`, `_free`, `_release_rows`, `_get_rows` | done |

`llama_media` is refcounted and thread-safe to reference from several threads. Encoded rows live in host memory in this version (phase 2 of the plan moves them to the device behind the same calls).

### Batch entry

| Function | Status |
|---|---|
| `llama_batch_ext_add_media(batch, seq_id, media, row0, n_rows, pos_0)` | done |

On a batch of an ENCODER context it queues the media for `llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, batch)`, which writes the rows into the media. On a decoder batch it adds the rows as embedding entries, with the positions computed from the media geometry and, for decoders that attend bidirectionally inside media (Gemma 3), the span bounds of the media. Error codes: -1 batch full, -3 bad seq_id, -4 not encoded / placeholder, -5 would split an atomic media, -6 the context cannot take this media.

### Planned additions

| Phase | Items |
|---|---|
| 1 (rest) | `llama_media_proc_video_hint`, full video builder, audio input (`add_audio`) |
| 3 | `llama_audio_stream_*` (10 functions), removal of `llama_model_n_tokens_delay` and `llama_model_get_tok_embd_rows` |
| 4 | `LLAMA_CONTEXT_TYPE_GEN_CODE/GEN_WAV/GEN_NAR`, `llama_get_sampled_tokens_ith`, `llama_batch_ext_add_codes`, `llama_set_seed_seq`, `llama_set_cfg_scale`, `llama_batch_ext_set_cond`, `llama_set_cond_media`, voice queries; removal of `llama_model_get_tok_embd` |

## `tools/mtmd/mtmd.h`

| Item | Status | Notes |
|---|---|---|
| `mtmd_input_chunk_get_llama_media(chunk)` | done | TRANSITIONAL. Returns the `llama_media` behind an image chunk, nullptr for chunks of the old engine. Needed because `mtmd-helper` may only use the public mtmd API. |

Behaviour of libmtmd with the shim:
- For ported projectors (Gemma 3 so far), `mtmd_init_from_file` also loads the mmproj with `llama_model_load_from_file`, creates an ENCODER context and a media proc, and routes images through them. Everything else (marker parsing, chunk API, audio, video helpers, TTS) is unchanged.
- The mmproj is loaded twice (old engine and libllama) while the shim exists. This costs one extra copy of the vision weights. The old engine is still needed by functions that read its hparams directly (video config, memory estimate).
- `MTMD_LEGACY_ENGINE=1` forces the old engine at runtime. `MTMD_LM_LOAD_MODE=dio` loads the libllama copy with direct I/O (mtmd has no load-mode parameter).
- If the libllama engine fails to load, mtmd logs a warning and falls back to the old engine.
- `mtmd_get_output_embd` and `mtmd_batch_get_output_embd` still return host rows (copied from the media).
- `mtmd_helper_decode_image_chunk` uses `llama_batch_ext_add_media` for libllama media when no post-decode callback is set: no `llama_set_causal_attn` toggle, positions and span set by libllama. With a callback (speculative decoding mirror), it keeps the old path.
- `MTMD_DEBUG_EMBEDDINGS=<path>` dumps the libllama rows in the same format as the old engine.

## Internal changes (no API, no ABI)

| Where | Change |
|---|---|
| `llama_ubatch` | `span` (`[2*n_tokens]` first/last position of the bidirectional span of each row, NULL if none); `media` (the media an encoder ubatch encodes) |
| `llama_batch_ext` | per-entry `span_lo` / `span_hi`; `model` and `ctx_type` of the context it was made for; `media` list (references) for encoder batches; a destructor that drops those references |
| `llama_batch_allocr` | copies spans into ubatches; `split_simple` and `split_seq` never cut a bidirectional span (they give back its head), following the `n_keep_tail` precedent |
| `llama_kv_cache::set_input_attn_mask` | a query inside a bidirectional span sees the later keys of its own span; everything else stays causal |
| `llama_context` | `encode()` routes batches with media to `encode_media()`: one ubatch per media, `n_tokens` = output rows, rows copied into the media |
| `llama_model` | virtual `media_bidirectional(type)` (false by default, true for images in Gemma 3); `print_info` skips the vocab of vocab-less models |
| `llama_model_base::load_hparams` | for `clip`, calls the arch's `load_arch_hparams` instead of returning early |
| `llama_model::create_memory` | `clip` keeps no memory |
| `src/llama.cpp` | the "CLIP cannot be used as main model" error is gone; the context-type check replaces it |
| `src/models/clip.cpp` | real arch: vision tower hparams (`clip.vision.*` via `LLM_KV` prefix), tensors through `create_tensor` (mmap, `-ot`, device split, repack apply), legacy FFN up/down swap, Gemma 3 graph on libllama helpers (`build_norm`, `build_ffn`, `build_attn_mha`, `build_lora_mm`) |
| `src/llama-media*.{h,cpp}` | media objects, layout, preprocessing; the image resampler is a copy of mtmd's `img_tool` (kept bit-identical, to be shared once mtmd uses it) |
| `llm_ffn_op_type` | new `LLM_FFN_GELU_QUICK` |
| `LLM_KV` | optional prefix that replaces the arch name (`clip.vision`) |
| `llm_tensor` / `LLM_TENSOR_NAMES` / `LLM_TENSOR_INFOS` | `V_ENC_*`, `V_PRE_NORM`, `V_POST_NORM`, `V_MMPROJ`, `V_MM_INP_PROJ`, `V_MM_SOFT_EMB_NORM`, `V_DS_*`, names identical to gguf-py `MODEL_TENSOR.V_*`; stem and projector tensors follow the output device |
| `llama_model_loader` | explicit instantiations of the string-key getters for `bool`, `int32_t`, `std::vector<float>`, `std::vector<int32_t>` |

## GGUF

No format change. mmproj files keep `general.architecture = clip`, `clip.*` keys and `v.*` / `mm.*` tensor names. Integer `clip.vision.*` keys are accepted as u32 or i32 (converters differ). Unlike mtmd, libllama checks every tensor shape; tensors whose shape is not given by hparams (projector, legacy layouts) are created with the shape read from the file.

## Known limitations of the prototype

- Only the projectors listed in `mtmd_lm_ported()` use libllama.
- Rows go through host memory (phase 2 is not written).
- Span-aware splitting covers `split_simple` and `split_seq`; `split_equal` (multi-sequence, non-unified KV) does not know about spans yet.
- The no-cache attention path (embedding models without a KV cache) does not read spans.
- Serialized chunks (`mtmd_input_chunk_save`) of libllama media load back as old-engine placeholders.

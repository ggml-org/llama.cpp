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

### Audio generation: code predictor

| Item | Status | Notes |
|---|---|---|
| `LLAMA_CONTEXT_TYPE_GEN_CODE` | done | staging value 17, outside the enum like ENCODER. Offered by mmproj files whose generation tower is ported (Qwen3-TTS). |
| `llama_get_sampled_tokens_ith(ctx, i, out, n_max)` | done | the codes of output row i (16 for Qwen3-TTS). |

A GEN_CODE batch entry carries a token (the code the backbone sampled) and a state embedding (the backbone hidden state), set with the existing `llama_batch_ext_add_token` and `llama_batch_ext_set_embd_state`. `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)` runs one frame per entry. The feedback embedding comes back through the existing `llama_get_embeddings_ith`. The codes are sampled in the graph by the sampler chain of the sequence (`llama_set_sampler`, existing API), greedy without one: the code predictor calls the chain's `backend_apply` once per codebook, 15 times in one graph. The k/v of the predictor lives in the graph, so no `llama_memory` is involved. Temperature is not applied, like the mtmd graph ("TODO: handle this on graph").

### Audio generation: codec (code2wav)

| Item | Status | Notes |
|---|---|---|
| `LLAMA_CONTEXT_TYPE_GEN_WAV` | done | staging value 18. Offered by mmproj files with a ported codec (Qwen3-TTS). |
| `llama_batch_ext_add_codes(batch, seq_id, codes, n_codes)` | done | one entry per frame, one code per codebook. |
| `llama_n_embd_out(ctx)` | done | width of the output rows of a context. Needed because one mmproj model now outputs rows of different widths per context type (feedback embedding of GEN_CODE, PCM of one frame for GEN_WAV). |

The entries of one `llama_process(DECODE)` call are consecutive frames of one sequence; output row i (`llama_get_embeddings_ith`) is the PCM of frame i (1920 samples for Qwen3-TTS). The decoder state of each sequence lives in the context memory: a new memory type, `llama_memory_slots` (internal), holds per-sequence state tensors of any shape that the model declares (38 for Qwen3-TTS: position counter, k/v window prefix of 8 layers, conv left-contexts, transposed-conv tails). The graph reads them as views and writes them back in place at the end. `llama_memory_seq_rm(mem, seq, -1, -1)` (existing API) resets a sequence; removing part of a sequence is refused, state save/load is not implemented yet.

### Preprocessing and layout

| Function | Status |
|---|---|
| `llama_media_proc_default_params`, `llama_media_proc_init(ctx_enc, model_dec, params)`, `llama_media_proc_free`, `llama_media_proc_has`, `llama_media_proc_sample_rate` | done (images; `sample_rate` returns -1 until audio is ported) |
| `llama_media_layout_init(proc, pos_0)`, `_free`, `_add_tokens`, `_add_image`, `_add_frames`, `_n_pieces`, `_get_tokens`, `_get_media` | done |

Difference from the accepted plan (`llama-ext-plan.md`):
- `llama_media_proc_init` takes the decoder **model**, not the decoder context. The only decoder-context value the proc needed was `n_ubatch`, which the caller passes as `params.n_span_max`. This lets libmtmd, which receives only the text model, create a proc.
- The video builder (`video_begin` / `add_video_frame` / `add_video_audio` / `video_end`) is replaced for now by `llama_media_layout_add_frames(lay, nx, ny, rgb0, rgb1, id)`, one call per frame pair that the model merges. This is what Qwen-VL needs; the full video builder stays planned.
- `llama_media_proc_video_hint` is not written yet (planned).

### Media handle

| Function | Status |
|---|---|
| `llama_media_get_type`, `_n_tokens`, `_n_pos`, `_is_atomic`, `_get_id`, `_is_placeholder`, `_is_encoded`, `_ref`, `_free`, `_release_rows`, `_get_rows` | done |
| `llama_media_get_row_pos(m, pos_0, row, pos, n_pos)` | done, TRANSITIONAL |

`llama_media_get_row_pos` is not in the accepted plan. Callers that still decode host rows with `llama_batch_ext_add_embd` (the server mixed batch, the speculative-decoding callback of `mtmd-helper`) need the decoder positions of each row; the libmtmd shim uses it to keep `mtmd_image_tokens_get_decoder_pos` working. It goes away once every caller uses `llama_batch_ext_add_media`.

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
| 4 | `LLAMA_CONTEXT_TYPE_GEN_NAR`, `llama_set_seed_seq`, `llama_set_cfg_scale`, `llama_batch_ext_set_cond`, `llama_set_cond_media`, voice queries; removal of `llama_model_get_tok_embd` (GEN_CODE, GEN_WAV, `llama_get_sampled_tokens_ith` and `llama_batch_ext_add_codes` are done) |

## `tools/mtmd/mtmd.h`

| Item | Status | Notes |
|---|---|---|
| `mtmd_input_chunk_get_llama_media(chunk)` | done | TRANSITIONAL. Returns the `llama_media` behind an image chunk, nullptr for chunks of the old engine. Needed because `mtmd-helper` may only use the public mtmd API. |

Behaviour of libmtmd with the shim:
- For ported projectors (Gemma 3, Qwen2-VL, Qwen2.5-VL, Qwen3-VL), `mtmd_init_from_file` also loads the mmproj with `llama_model_load_from_file`, creates an ENCODER context and a media proc, and routes images through them. Everything else (marker parsing, chunk API, audio, video helpers, TTS) is unchanged.
- The mmproj is loaded twice (old engine and libllama) while the shim exists. This costs one extra copy of the vision weights. The old engine is still needed by functions that read its hparams directly (video config, memory estimate).
- `MTMD_LEGACY_ENGINE=1` forces the old engine at runtime. `MTMD_LM_LOAD_MODE=dio` loads the libllama copy with direct I/O (mtmd has no load-mode parameter).
- If the libllama engine fails to load, mtmd logs a warning and falls back to the old engine.
- `mtmd_get_output_embd` and `mtmd_batch_get_output_embd` still return host rows (copied from the media).
- `mtmd_helper_decode_image_chunk` uses `llama_batch_ext_add_media` for libllama media when no post-decode callback is set: no `llama_set_causal_attn` toggle, positions and span set by libllama. With a callback (speculative decoding mirror), it keeps the old path.
- `MTMD_DEBUG_EMBEDDINGS=<path>` dumps the libllama rows in the same format as the old engine.
- Qwen3-TTS: the code predictor (`mtmd_gen_audio_process` with `MTMD_GEN_PROCESS_TYPE_GEN_CODE`) runs on a GEN_CODE `llama_context`, one per `mtmd_gen_state`, with a sampler chain built from the call's `top_k` / `top_p` and the state seed. Greedy output is identical to the old engine; sampled output differs because the random draws come from libllama's dist sampler instead of mtmd's generator. code2wav (`MTMD_GEN_PROCESS_TYPE_GEN_WAV`) runs on a GEN_WAV `llama_context` per gen state; `MTMD_LEGACY_CODEC=1` keeps the old codec. A short last window is padded to 72 frames and trimmed, as the old engine does: the graph shape decides which CPU matmul kernels run, so this keeps the PCM bit-identical (unpadded, the samples differ by a few LSB from frame 1 on). The speaker encoder stays on the old engine.
- Image chunks of libllama media carry their grid (`nx`, `ny`) and one placeholder entry, so `mtmd_image_tokens_get_nx/ny`, `mtmd_input_chunk_get_n_tokens` and serialized chunks keep the same counts.
- Qwen3-VL video frames stay on the old engine: the model sizes video frames differently from images and that preprocessing is not ported yet.

## Internal changes (no API, no ABI)

| Where | Change |
|---|---|
| `llama_ubatch` | `span` (`[2*n_tokens]` first/last position of the bidirectional span of each row, NULL if none); `media` (the media an encoder ubatch encodes) |
| `llama_batch_ext` | per-entry `span_lo` / `span_hi`; `model` and `ctx_type` of the context it was made for; `media` list (references) for encoder batches; a destructor that drops those references |
| `llama_batch_allocr` | copies spans into ubatches; `split_simple` and `split_seq` never cut a bidirectional span (they give back its head), following the `n_keep_tail` precedent |
| `llama_kv_cache::set_input_attn_mask` | a query inside a bidirectional span sees the later keys of its own span; everything else stays causal |
| `llama_context` | `encode()` routes batches with media to `encode_media()`: one ubatch per media, `n_tokens` = output rows, rows copied into the media |
| `llama_model` | virtual `media_bidirectional(type)` (false by default, true for images in Gemma 3); virtual `graph_max_nodes_hint()` (8192 for vision towers, 32768 for the unrolled code predictor); virtuals `ctx_n_vocab(type)` / `ctx_n_embd_state(type)` giving the batch input widths per context type; `print_info` skips the vocab of vocab-less models; `done_getting_tensors` runs in partial mode for `clip`, so towers that are not ported stay in the file unread |
| `llm_graph_result` | `t_codes` (codes sampled in the graph) |
| `llama_context` | `gen_code()`: one ubatch per entry for GEN_CODE contexts, codes kept per output row; `gen_wav()`: one ubatch per call for GEN_WAV contexts; every use of the output width goes through `n_embd_out_ctx()` (the model's width for the context type) |
| `llama_ubatch` | `codes` / `n_codes` (codes per row, audio generation) |
| `llama_batch_ext` | per-entry `codes_off` / `n_codes` and a `codes` array |
| `llama_memory_slots` (new, `src/llama-memory-slots.{h,cpp}`) | per-sequence state tensors declared by the model, in one buffer on the output device; implements `llama_memory_i` |
| `llama_model` | virtual `ctx_n_embd_out(type)`; `create_memory` asks the `clip` arch for the memory of its context type |
| `llama_model_base::load_hparams` | for `clip`, calls the arch's `load_arch_hparams` instead of returning early |
| `llama_model::create_memory` | `clip` keeps no memory |
| `src/llama.cpp` | the "CLIP cannot be used as main model" error is gone; the context-type check replaces it |
| `src/models/clip.cpp` | real arch: vision tower hparams (`clip.vision.*` via `LLM_KV` prefix), tensors through `create_tensor` (mmap, `-ot`, device split, repack apply), legacy FFN up/down swap, graphs on libllama helpers (`build_norm`, `build_ffn`, `build_attn_mha`, `build_lora_mm`): Gemma 3 (SigLIP + pooling), Qwen2-VL / Qwen2.5-VL (M-RoPE patch positions, temporal merge, window attention with permutation and block mask), Qwen3-VL (learned position embedding resized in graph, deepstack rows concatenated after the main row). Per-media inputs are computed from the media geometry in `llm_graph_input_media` |
| `src/llama-media*.{h,cpp}` | media objects, layout, preprocessing; the image resampler is a copy of mtmd's `img_tool` (kept bit-identical, to be shared once mtmd uses it) |
| `llm_ffn_op_type` | new `LLM_FFN_GELU_QUICK` |
| `LLM_KV` | optional prefix that replaces the arch name (`clip.vision`) |
| `llm_tensor` / `LLM_TENSOR_NAMES` / `LLM_TENSOR_INFOS` | `V_ENC_*`, `V_PRE_NORM`, `V_POST_NORM`, `V_MMPROJ`, `V_MM_INP_PROJ`, `V_MM_SOFT_EMB_NORM`, `V_DS_*`, names identical to gguf-py `MODEL_TENSOR.V_*`; stem and projector tensors follow the output device |
| `llama_model_loader` | explicit instantiations of the string-key getters for `bool`, `int32_t`, `std::vector<float>`, `std::vector<int32_t>` |

## GGUF

No format change. mmproj files keep `general.architecture = clip`, `clip.*` keys and `v.*` / `mm.*` tensor names. Integer `clip.vision.*` keys are accepted as u32 or i32 (converters differ). Unlike mtmd, libllama checks every tensor shape; tensors whose shape is not given by hparams (projector, legacy layouts) are created with the shape read from the file.

## Verification

Reference: the same binary with `MTMD_LEGACY_ENGINE=1` (mtmd's own engine), so both sides share the text model, the sampler and the build. Encoder rows are compared through `MTMD_DEBUG_EMBEDDINGS` dumps; text and audio as greedy outputs (`--temp 0` / `--top-k 1`, seed 1). Strix Halo (gfx1151), ROCm build.

| Model | What runs on libllama | CPU | GPU (HIP) |
|---|---|---|---|
| Gemma 3 4B (Q8_0, mmproj f16) | vision tower, layout, decoder rows with a bidirectional image span | rows bit-exact (256 x 2560), text identical | rows bit-exact with FA off and on, text identical |
| Qwen2.5-VL 3B (Q8_0, mmproj f16) | vision tower with window attention, M-RoPE rows | rows bit-exact (391 x 2048), text identical | same, FA off and on |
| Qwen3-VL 2B (Q8_0, mmproj Q8_0) | vision tower with deepstack, M-RoPE rows | rows bit-exact (300 x 8192), text identical | same, FA off and on |
| Qwen3-TTS 1.7B (Q8_0, mmproj bf16) | code predictor (GEN_CODE), codec (GEN_WAV) | WAV byte-identical, FA off and on, one window and 4 windows (221 frames) | byte-identical with `GGML_CUDA_DISABLE_FUSION=1`; with fusion on, the codes drift after a few frames |

Checks that must fail did fail:
- Gemma 3 with the image span forced causal changes the generated text, so the span mask is in effect.
- Qwen3-TTS with `--top-k 50` differs between the engines (different random draws) while repeated runs of each engine are identical, so the libllama predictor is the one running.
- The first Gemma 3 run that looked bit-exact had silently fallen back to the old engine (a u32/i32 key mismatch); the run scripts now report which engine ran.

Numerical notes:
- The GPU difference in TTS comes from kernel fusion: the HIP backend fuses neighbouring ops depending on node order, and the predictor graph orders its nodes differently from mtmd's. With fusion off, both engines agree byte for byte.
- An unpadded last codec window (34 frames instead of 72) differs by a few LSB from frame 1 on: the CPU matmul kernels depend on the column count. The shim pads like mtmd; libllama itself takes any frame count.
- Ported encoders reuse `build_attn_mha`, whose non-FA path sets BF16 accumulation on `kq`; this changed nothing on CPU or HIP for these models.

Regression tests: `ctest` on CPU 64/64 (download tests and a git-lfs vocab fixture excluded), `test-llama-archs` 152/152.

Text-only speed (the batch allocator, the KV mask and the output width changed for every model): standard matrix, Gemma 3 4B Q8_0, `llama-bench -lm dio -ngl 99 -fa 1 -p 1024 -n 128 -d 0,40000 -ub 1024,2048,4096 -b 4096`, master `e8fbcad85` then the branch, one after the other.

| ubatch | test | master t/s | branch t/s |
|---|---|---|---|
| 1024 | pp1024 | 2393 ± 102 | 2356 ± 79 |
| 1024 | tg128 | 45.46 ± 0.23 | 45.49 ± 0.12 |
| 1024 | pp1024 @ d40000 | 1276 ± 28 | 1313 ± 18 |
| 1024 | tg128 @ d40000 | 35.95 ± 0.81 | 37.96 ± 0.06 |
| 2048 | pp1024 | 2191 ± 114 | 2461 ± 21 |
| 2048 | tg128 | 43.17 ± 0.35 | 47.65 ± 0.06 |
| 2048 | pp1024 @ d40000 | 1205 ± 71 | 1396 ± 7 |
| 2048 | tg128 @ d40000 | 36.34 ± 0.19 | 37.99 ± 0.03 |
| 4096 | pp1024 | 2327 ± 34 | 2462 ± 19 |
| 4096 | tg128 | 45.70 ± 0.07 | 47.58 ± 0.10 |
| 4096 | pp1024 @ d40000 | 1287 ± 46 | 1403 ± 26 |
| 4096 | tg128 @ d40000 | 36.34 ± 0.12 | 37.85 ± 0.10 |

No slowdown. The branch is not faster either: master ran first and its spread is much wider, so the gap is the order of the runs, not the change.

## Known limitations of the prototype

- Only the projectors listed in `mtmd_lm_ported()` use libllama.
- Rows go through host memory (phase 2 is not written).
- Span-aware splitting covers `split_simple` and `split_seq`; `split_equal` (multi-sequence, non-unified KV) does not know about spans yet.
- The no-cache attention path (embedding models without a KV cache) does not read spans.
- Serialized chunks (`mtmd_input_chunk_save`) of libllama media load back as old-engine placeholders.
- Generation contexts process one sequence per call (GEN_CODE: one frame per ubatch; GEN_WAV: one sequence per call). Batching several sequences per ubatch is the next step and needs no API change.
- The codec's attention layers pass the main tower's last layer index to `build_attn_mha`, because the fused-op probe maps layer indices to devices and the codec layers are not layers of the main tower.
- The mmproj is loaded once per engine part: the shim loads it for the encoder (vision), or for the generation stages, besides the old engine's own copy.

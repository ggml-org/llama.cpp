# API and ABI changes of mtmd-in-libllama

Branch `mtmd-in-libllama`, based on master `e8fbcad85`. mtmd's engine (model loader, graphs of every family, image and audio preprocessing, the layout of media in the prompt, realtime audio input, the audio generation pipelines) now lives in libllama, under `src/clip/`. libmtmd keeps its public API and becomes a client of `src/llama-ext.h`. This file lists every API, ABI, behaviour and format change, so a reviewer can judge the cost of the approach.

## Summary

| Surface | Change | ABI effect |
|---|---|---|
| `include/llama.h` | none | none |
| `src/llama-ext.h` (staging, C++) | media, streaming and generation API, see below (26 -> 83 `LLAMA_API` declarations) | new exported symbols only |
| `tools/mtmd/mtmd.h` | one new function (`mtmd_input_chunk_get_llama_media`); the other 86 keep their signatures | new exported symbol only |
| `tools/mtmd/mtmd-helper.h` | none | none |
| serialized chunks (`mtmd_input_chunk_save`) | format version 6, counts and ids only; versions 2 to 5 no longer load | buffers saved by older builds are rejected |
| GGUF | none (mmproj files load as they are) | - |
| internal structs (`llama_ubatch`, `llama_batch_ext`, `llama_batch_allocr`, `llama_model`, `llama_model_loader`, `LLM_TN_IMPL`, `llm_graph_*`) | new fields and values | internal, not part of any ABI |

No existing libllama function changed its signature or meaning. No public struct changed its layout.

## Where the code went

| Before (master) | After |
|---|---|
| `tools/mtmd/mtmd-model.{h,cpp}` (loader, hparams of every projector) | `src/clip/clip-model.{h,cpp}`; tensors created through `llama_model_loader::create_tensor` |
| `tools/mtmd/mtmd-graph.{h,cpp}`, `tools/mtmd/models/*` (66 files) | `src/clip/clip-graph.{h,cpp}`, `src/clip/models/*`; built inside `llm_graph_context` |
| `tools/mtmd/mtmd-subcontext.{h,cpp}` (backends, scheduler, input setters) | `src/clip/clip-subcontext.{h,cpp}`: input setters and output readers only, the graph runs on the scheduler of the `llama_context` |
| `tools/mtmd/mtmd-image.*`, `mtmd-audio.*` (preprocessing, DSP) | `src/clip/clip-image.*`, `src/clip/clip-audio.*` |
| `mtmd.cpp`: per-projector markers, tokenizer media logic, chunk structs, decoder positions | `src/clip/clip-media.{h,cpp}`, behind `llama_media_layout_*` |
| `mtmd.cpp`: realtime audio input | `src/clip/clip-stream.{h,cpp}`, behind `llama_audio_stream_*` |
| `mtmd.cpp`: audio generation pipelines, chatterbox reference encode, YuE2 NAR loader | `src/clip/clip-gen.{h,cpp}`, behind `llama_batch_ext_add_gen` on GEN contexts |
| `mtmd.cpp` (6.1k lines) | `mtmd.cpp` (2.2k lines): marker parsing, lazy media, chunk API, forwarders |

All engine names were renamed `mtmd_` -> `clip_` and `MTMD_` -> `CLIP_`. The copy scripts are kept with the branch's notes; the code of the families is otherwise unchanged.

## `src/llama-ext.h` (staging API)

All functions are `LLAMA_API` and exported from libllama. The header is C++ and explicitly unstable.

### Context roles

| Item | Notes |
|---|---|
| `LLAMA_CONTEXT_TYPE_ENCODER` (16), `LLAMA_CONTEXT_TYPE_GEN_CODE` (17), `LLAMA_CONTEXT_TYPE_GEN_WAV` (18) | `static const enum llama_context_type` values outside the enum for now; promoting them into the enum does not change the ABI |
| `llama_model_has_context_type(model, type)` | an mmproj offers ENCODER when it has a vision or audio tower, GEN_CODE / GEN_WAV when it has a generation tower |
| `llama_n_embd_out(ctx)` | width of the output rows of a context; one mmproj outputs rows of different widths per context type |

An mmproj (`general.architecture = clip`) model needs a media context type, and these context types need an mmproj model. ENCODER and GEN contexts always have `embeddings = true`, non-causal attention and no pooling.

### Preprocessing and layout

| Function | Notes |
|---|---|
| `llama_media_proc_default_params`, `llama_media_proc_init(ctx_enc, model_dec, params)`, `_free`, `_has`, `_sample_rate`, `_n_frame_merge` | the proc holds the per-projector layout of the clip engine (markers, separators, preprocessors); `model_dec` may be null for the queries only |
| `llama_media_proc_video_frame_size`, `llama_media_proc_video_markers` | what mtmd's video helper needs from the model |
| `llama_model_media_n_tokens_max(model, type, &atomic)` | the largest span of one media and whether the decoder takes it in one ubatch (server check of `n_ubatch`) |
| `llama_media_layout_init(proc, pos_0)`, `_free`, `_add_tokens`, `_add_image`, `_add_frames`, `_add_audio`, `_n_pieces`, `_get_tokens`, `_get_media` | one layout per prompt; the moved tokenizer turns one media into pieces (tokens and media spans), as mtmd did |
| `llama_media_debug_encode`, `llama_media_debug_preprocess` | for `llama-mtmd-debug` (model ports), unstable |

The user limits of the image token count are kv overrides of the mmproj model, `clip.vision.image_min_tokens` and `clip.vision.image_max_tokens` (int); `llama_media_proc_params` lost its two fields. The audio preprocessor computes its log-mel on the encoder context's scheduler (`llama_context::sched_aux`, internal).

### Media handle and batch entry

| Function | Notes |
|---|---|
| `llama_media_get_type`, `_n_tokens`, `_n_pos`, `_is_atomic`, `_get_id`, `_is_placeholder`, `_is_encoded`, `_ref`, `_free`, `_release_rows`, `_get_rows` | refcounted, thread-safe to reference |
| `llama_media_get_row_pos(m, pos_0, row, pos, n_pos)` | TRANSITIONAL: decoder positions for callers that still decode host rows with `llama_batch_ext_add_embd` |
| `llama_batch_ext_add_media(batch, seq_id, media, row0, n_rows, pos_0)` | ENCODER batch: queues the media for `llama_process(ENCODE)`; decoder batch: adds the rows with their positions and span bounds |

A media is a chunk of the clip engine (`clip_image_tokens` / `clip_audio_tokens`) plus its encoded rows. `encode_media` runs consecutive images of equal geometry in one graph when the tower takes a batch, as `mtmd_batch` did.

### Realtime audio input

| Function | Notes |
|---|---|
| `llama_audio_stream_params`, `llama_audio_stream_default_params`, `llama_media_proc_has_audio_stream` | |
| `llama_audio_stream_init(proc, params)`, `_free`, `_push`, `_flush`, `_step`, `_get_output_embd`, `_is_seq_start`, `_get_n_rows`, `_get_n_rows_flushed` | the semantics of `mtmd_audio_stream_*`; streams of one proc are serialized, they encode on its ENCODER context |

### Audio generation

| Item | Notes |
|---|---|
| `llama_get_sampled_tokens_ith`, `llama_batch_ext_add_codes` | native graphs (Qwen3-TTS): K codes per output row sampled in the graph by the sequence's sampler chain; frames of codes into the codec |
| `llama_gen_info`, `llama_gen_audio_type`, `llama_gen_stage`, `llama_gen_inp`, `llama_gen_out` | the types of `mtmd_gen_*` |
| `llama_model_gen_info`, `llama_gen_inp_default`, `llama_model_gen_read_tensor`, `llama_model_gen_voice_name` | model queries |
| `llama_media_init_voice(model, name)` | a predefined voice as an encoded AUDIO media with its gen-side reference |
| `llama_batch_ext_add_gen(batch, seq_id, inp)`, `llama_get_gen_out_ith(ctx, i, out)` | one step of one sequence; `llama_process(DECODE)` runs it on a GEN_CODE / GEN_WAV context; both stages run in one context |
| `llama_gen_set_seed(ctx, seq_id, seed)`, `llama_gen_set_ref(ctx, seq_id, media)` | the RNG of a sequence; the speaker reference from an encoded media (chatterbox reference encode, predefined voice) |

The per-sequence state of the moved pipelines (RNG, decoder slots, host DSP such as the LFM3 ISTFT and the ZONOS3 iSTFT head) is the memory of the GEN context (`llama_memory_clip_gen`); `llama_memory_seq_rm(mem, seq, -1, -1)` is a cold start, the state cannot be saved. Qwen3-TTS keeps its native graphs (GEN_CODE with the in-graph sampler, GEN_WAV with `llama_memory_slots`); `LLAMA_CLIP_NATIVE=0` runs it on the moved pipeline instead.

## `tools/mtmd/mtmd.h`

| Item | Notes |
|---|---|
| `mtmd_input_chunk_get_llama_media(chunk)` | TRANSITIONAL: the `llama_media` behind an image or audio chunk, for `mtmd-helper`, which may only use the public mtmd API |

Behaviour of libmtmd:
- `mtmd_init_from_file` loads the mmproj once, with `llama_model_load_from_file`, creates an ENCODER context and a media proc when the file has encoder towers, and a GEN context per `mtmd_gen_state` when it has a generation tower. The engine moved, the API did not.
- `mtmd_tokenize` keeps the marker parsing, the lazy media (video) and the BOS/EOS handling; one `llama_media_layout` per prompt lays out the media. Images of a prompt are numbered and aligned within it, as before.
- `mtmd_batch_*` hands all its media to one ENCODER batch; libllama groups the ones that share a graph.
- `mtmd_get_memory_usage` loads the mmproj with `no_alloc` and sums `llama_get_memory_breakdown` of the encoder context, as `common_fit` does for text models.
- `mtmd_get_cap_from_file` reads the GGUF keys with the gguf API.
- `MTMD_DEBUG_EMBEDDINGS=<path>` dumps the encoded rows in the old format; `MTMD_DEBUG_CHUNKS=1` prints the layout of each prompt.
- `MTMD_LM_LOAD_MODE=dio` loads the mmproj with direct I/O (mtmd has no load-mode parameter).
- Serialized chunks hold counts and ids (version 6); they load back as placeholders, as before.

## Internal changes (no API, no ABI)

| Where | Change |
|---|---|
| `LLM_TN_IMPL`, `llm_tensor` | `LLM_TENSOR_NAMED` and `LLM_TN_IMPL::named(arch, name)`: a tensor whose name is taken from the file (output device, MUL_MAT probe, ADD for `.bias`); `is_bias()` |
| `llama_model_loader` | `read_tensor_host(name, dst, size)` for the small tensors the engine keeps on the host (mel filters, voices); `fname`; string-key getters for `bool`, `int32_t`, arrays |
| `llama_model_clip` (`src/models/clip.cpp`, `clip-engine.cpp`, `clip-qwen3tts-wav.cpp`) | holds the engine's models of every modality; native graphs for Gemma 3, Qwen2/2.5/3-VL and Qwen3-TTS share their weights with the engine; `graph_engine` builds the moved graphs inside `llm_graph_context` and registers their flash-attention nodes for the auto-detection |
| `llama_context` | `encode_media`, `encode_media_engine(_group)`, `clip_exec` (one engine graph through `process_ubatch`), `clip_sub` (engine runtime per tower), `gen_engine`, `sched_aux` |
| `llm_graph_context::build_pooling` | a clip graph without an embedding output (a codec stage with alternative branches) skips pooling |
| `llama_ubatch` / `llama_batch_ext` / `llama_batch_allocr` | media, spans, codes and gen entries; span-aware splitting |
| `llama_kv_cache::set_input_attn_mask` | a query inside a bidirectional span sees the later keys of its span |
| `llama_memory_slots`, `llama_memory_clip_gen` | per-sequence state of the codec (native) and of the moved pipelines |

## GGUF

No format change. mmproj files keep `general.architecture = clip`, `clip.*` keys and their tensor names.

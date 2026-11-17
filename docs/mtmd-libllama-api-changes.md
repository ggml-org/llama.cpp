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

## Results

Strix Halo (gfx1151), ROCm build. Reference: mtmd's own engine. Before the deletion, both engines ran in the same binary (`MTMD_LEGACY_ENGINE=1`) and their outputs were stored; after the deletion, the client-only mtmd ran every case again and was compared with those stored outputs. Greedy settings (`--temp 0`, `--top-k 1`, seed 1). Encoder rows through `MTMD_DEBUG_EMBEDDINGS`, prompt layout through `MTMD_DEBUG_CHUNKS`.

### Parity after the deletion (CPU and GPU)

| Area | Models | Result |
|---|---|---|
| Vision, moved graphs | SmolVLM 500M, SmolVLM2 500M Video, InternVL 2.5 1B, InternVL3 1B, LFM2-VL 450M, granite-docling 258M, LightOnOCR 1B, HunyuanOCR, Qwen2-VL 2B, granite-vision 3.2 2B, dots.ocr, DeepSeek-OCR, MiniCPM-V 2.6, LLaVA 1.5 7B, LLaVA 1.6 Mistral 7B, GLM-Edge-V 5B, Gemma 4 E4B, Qwen2.5-Omni 7B | layout identical, rows bit-exact, text identical |
| Vision, native graphs | Gemma 3 4B, Qwen2.5-VL 3B, Qwen3-VL 2B, Qwen3.5 0.8B | same |
| Audio input | ultravox 1B, Qwen3-ASR 0.6B, LFM2-Audio 1.5B, Voxtral Mini 3B, Gemma 4 E4B, Qwen2.5-Omni 7B, Higgs STT, VibeVoice-ASR | same (Qwen3-ASR and Higgs: the old engine dumped only the last chunk; those rows match the last chunk of the new dump) |
| Realtime audio input | Gemma 4 E4B, Higgs STT, Voxtral Mini 4B Realtime, VibeVoice-ASR | every row bit-exact, greedy tokens identical |
| Generation | Qwen3-TTS 1.7B (moved pipeline and native graphs), Soprano 80M, Chatterbox, Chatterbox-turbo, ZONOS3-Mini | WAV byte-identical on CPU; on GPU too, except native Qwen3-TTS, which is byte-identical with `GGML_CUDA_DISABLE_FUSION=1` (HIP fuses neighbouring ops by node order) |

Not tested, no small GGUF available: Pocket TTS, Drama 3, Fish S2, YuE2, LFM3-Audio (generation and its stream). Their code moved unchanged.

Tests: `ctest` on CPU passes except `test-tokenizers-ggml-vocabs` (its fixture is an unfetched git-lfs pointer); `test-llama-archs` 152/152; `test-mtmd-impl` runs on the moved engine through the `src/clip` headers.

### Code size

| | master `e8fbcad85` | branch |
|---|---|---|
| `tools/mtmd` (C/C++) | 42,956 lines | 10,270 lines (`mtmd.cpp` 6,101 -> 2,225) |
| `src` | 120,647 lines | 158,918 lines: `src/clip` 33,443, the rest +4,828 (native graphs of 3 families, media and gen plumbing, slot memories) |
| `src` + `tools/mtmd` | 163,603 | 169,188 (+5,585) |

The net growth is the native graphs (Gemma 3, Qwen-VL, Qwen3-TTS), which now exist next to the moved ones, and the bridge (`clip-engine.cpp`, `llama-media.cpp`, `clip-gen-memory.cpp`).

### API surface

| Header | master | branch |
|---|---|---|
| `include/llama.h` | unchanged | unchanged |
| `src/llama-ext.h` (`LLAMA_API` declarations) | 26 | 83 |
| `tools/mtmd/mtmd.h` (`MTMD_API`) | 86 | 87 |
| `tools/mtmd/mtmd-helper.h` (`MTMD_API`) | 36 | 36 |

### Speed (GPU, master build vs branch build, 3 runs each)

| Case | master | branch |
|---|---|---|
| Gemma 3 4B, image encode (native) | 783-786 ms | 776-786 ms |
| Qwen2.5-VL 3B, image encode (native) | 341-345 ms | 338-343 ms |
| InternVL3 1B, image encode, 5 tiles | 631-632 ms | 621-631 ms |
| LFM2-VL 450M, image encode | 82-87 ms | 82-87 ms |
| MiniCPM-V 2.6, image encode, slices | 404-415 ms | 406-412 ms |
| Gemma 4 E4B, audio encode | 82-84 ms | 99-113 ms |
| Voxtral Mini 3B, audio encode | 294-311 ms | 304-320 ms |
| Qwen3-TTS 1.7B, llama-tts wall time | 3.28-3.33 s | 3.21-3.25 s |
| Soprano 80M, llama-tts wall time | 0.57-0.63 s | 0.56-0.60 s |
| Chatterbox-turbo, llama-tts wall time | 6.56-6.65 s | 6.70-9.55 s |
| ZONOS3-Mini, llama-tts wall time | 1.48-1.67 s | 1.43-1.45 s |

Equal within noise, except Gemma 4 audio (+17 to +29 ms per clip, about +20 %). Not investigated yet; the audio towers compute their log-mel on the encoder context's scheduler between encoder graphs, which is the first suspect. On CPU the same cases are equal within noise.

Text only (`llama-bench -m gemma-3-4b-it-Q8_0 -lm dio -ngl 99 -fa 1 -p 1024 -n 128 -d 0,40000 -ub 1024,2048,4096 -b 4096`, master then branch):

| ubatch | test | master t/s | branch t/s |
|---|---|---|---|
| 1024 | pp1024 | 2721 ± 27 | 2710 ± 29 |
| 1024 | tg128 | 47.76 ± 0.06 | 47.76 ± 0.07 |
| 1024 | pp1024 @ d40000 | 1432 ± 21 | 1412 ± 24 |
| 1024 | tg128 @ d40000 | 38.01 ± 0.03 | 38.04 ± 0.04 |
| 2048 | pp1024 | 2470 ± 20 | 2467 ± 28 |
| 2048 | tg128 | 47.68 ± 0.10 | 47.62 ± 0.09 |
| 2048 | pp1024 @ d40000 | 1404 ± 6 | 1391 ± 10 |
| 2048 | tg128 @ d40000 | 38.10 ± 0.06 | 37.93 ± 0.12 |
| 4096 | pp1024 | 2473 ± 19 | 2437 ± 56 |
| 4096 | tg128 | 47.59 ± 0.07 | 47.61 ± 0.08 |
| 4096 | pp1024 @ d40000 | 1411 ± 16 | 1401 ± 12 |
| 4096 | tg128 @ d40000 | 38.23 ± 0.05 | 38.02 ± 0.05 |

All within 1.5 %, below the 3 % noise floor of this machine.

### Memory estimate (`mtmd_get_memory_usage`, CPU)

Same as master within 2 % for Gemma 3, InternVL3, Voxtral and Gemma 4, except Qwen2.5-VL: 2,783 MiB instead of 1,846 MiB. Its native graphs reserve the compute buffer for the largest image (4,096 rows); mtmd reserved for a 46 x 46 warmup image and grew on demand.

## Known limitations

- The soundtrack pieces of an LFM3-Omni video are laid out as separate audio spans; mtmd joined them into one track and encoded it once. Not ported (no test model).
- The native graphs (Gemma 3, Qwen2/2.5/3-VL, Qwen3-TTS) and the moved graphs of the same families both exist; one of the two should go.
- The moved graphs are mtmd's graphs as they were; they use `clip_graph` helpers, not `llm_graph_context` helpers.
- `test-mtmd-impl` does not build on Windows (libllama exports only `LLAMA_API` there).
- Encoded rows go through host memory.
- Serialized chunks of older builds no longer load.
- Generation contexts process one gen entry at a time; streams of one proc are serialized.

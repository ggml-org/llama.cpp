# API and ABI changes of mtmd-in-libllama

Branch `mtmd-in-libllama`, based on master `e8fbcad85`. mtmd's engine (model loader, graphs of every family, image and audio preprocessing, the layout of media in the prompt, realtime audio input, the audio generation pipelines) now lives in libllama, under `src/clip/`. libmtmd keeps its public API and becomes a client of `src/llama-ext.h`. This file lists every API, ABI, behaviour and format change, so a reviewer can judge the cost of the approach.

## Summary

| Surface | Change | ABI effect |
|---|---|---|
| `include/llama.h` | none | none |
| `src/llama-ext.h` (staging, C++) | media, streaming and generation API, see below (26 -> 89 `LLAMA_API` declarations) | new exported symbols only |
| `tools/mtmd/mtmd.h` | one new function (`mtmd_input_chunk_get_llama_media`); the other 86 keep their signatures | new exported symbol only |
| `tools/mtmd/mtmd-helper.h` | none | none |
| serialized chunks (`mtmd_input_chunk_save`) | format version 7: counts, ids and the decoder positions of image rows; versions 2 to 6 still load | none for readers; buffers written by this build need it or a later one |
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
| `mtmd.cpp` (6.1k lines) | `mtmd.cpp` (2.9k lines): marker parsing, lazy media, chunk API, forwarders |

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
| `llama_media_proc_video_frame_size`, `llama_media_proc_video_markers`, `llama_media_proc_video_template` (`enum llama_media_video_template`) | what mtmd's video helper needs from the model: frame size, the text around a video, how frames are sampled and marked (default, Qwen3-VL, LFM3, LFM3-Omni) |
| `llama_model_media_n_tokens_max(model, type, &atomic)` | the largest span of one media and whether the decoder takes it in one ubatch (server check of `n_ubatch`) |
| `llama_media_layout_init(proc, pos_0)`, `_free`, `_add_tokens`, `_add_image`, `_add_frames`, `_add_audio`, `_n_pieces`, `_get_tokens`, `_get_media` | one layout per prompt; the moved tokenizer turns one media into pieces (tokens and media spans), as mtmd did |
| `llama_media_layout_begin_track(lay, pcm, n_samples, n_pieces)`, `llama_media_layout_add_track_piece(lay, id)` | an audio track cut in pieces that sit in different places of the prompt (the soundtrack of an LFM3-Omni video, a piece after each group of frames): encoded once as one clip, each piece takes its rows; LFM3-Omni only; a piece without PCM makes the track's media placeholders |
| `llama_media_debug_encode`, `llama_media_debug_preprocess` | for `llama-mtmd-debug` (model ports), unstable |

The user limits of the image token count are kv overrides of the mmproj model, `clip.vision.image_min_tokens` and `clip.vision.image_max_tokens` (int); `llama_media_proc_params` lost its two fields. The audio preprocessors compute their log-mel on a CPU scheduler owned by the proc (`clip_cpu_compute`, internal), in blocks of 1024 frames, under a lock of that scheduler: the media of one proc can be encoded on several contexts at once.

### Media handle and batch entry

| Function | Notes |
|---|---|
| `llama_media_get_type`, `_n_tokens`, `_n_pos`, `_is_atomic`, `_get_id`, `_is_placeholder`, `_is_encoded`, `_ref`, `_free`, `_release_rows`, `_get_rows` | refcounted, thread-safe to reference |
| `llama_media_get_row_pos(m, pos_0, row, pos, n_pos)` | TRANSITIONAL: decoder positions for callers that still decode host rows with `llama_batch_ext_add_embd` |
| `llama_media_get_grid(m, nx, ny)` | the token grid the engine records for an image, for mtmd's deprecated `mtmd_image_tokens_get_nx/ny` |
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
| `llama_get_sampled_tokens_ith`, `llama_batch_ext_add_codes` | native graphs (Qwen3-TTS): K codes per output row sampled in the graph by the sequence's sampler chain, which must run on the backend as a whole (`llama_set_sampler` refuses a chain with a host-only sampler; a backend sampler that leaves candidates is mapped back to codes); frames of codes into the codec, each frame checked against the codebooks when it is added |
| `llama_gen_info`, `llama_gen_audio_type`, `llama_gen_stage`, `llama_gen_inp`, `llama_gen_out` | the types of `mtmd_gen_*` |
| `llama_model_gen_info`, `llama_gen_inp_default`, `llama_model_gen_read_tensor`, `llama_model_gen_voice_name` | model queries |
| `llama_media_init_voice(model, name)` | a predefined voice as an encoded AUDIO media with its gen-side reference; the voices of lfm3a are names for the chat template and have no rows |
| `llama_batch_ext_add_gen(batch, seq_id, inp)`, `llama_get_gen_out_ith(ctx, i, out)` | one step of one sequence; `llama_process(DECODE)` runs it on a GEN_CODE / GEN_WAV context; both stages run in one context |
| `llama_gen_set_seed(ctx, seq_id, seed)`, `llama_gen_set_ref(ctx, seq_id, media)`, `llama_gen_reset(ctx, seq_id)` | the RNG of a sequence; the speaker reference from an encoded media (chatterbox reference encode, predefined voice); a new utterance that keeps both (`llama_memory_seq_rm` of the whole sequence drops them); on a Qwen3-TTS codec context the reset clears the codec slots |

The per-sequence state of the moved pipelines (RNG, decoder slots, host DSP such as the LFM3 ISTFT and the ZONOS3 iSTFT head) is the memory of the GEN context (`llama_memory_clip_gen`); `llama_memory_seq_rm(mem, seq, -1, -1)` is a cold start, the state cannot be saved. Every projector has one graph. Gemma 3, Qwen2/2.5/3-VL (also the vision tower of a legacy `qwen2.5o` file, which the engine resolves to Qwen2.5-VL) and the Qwen3-TTS generation tower run on native graphs built with libllama's helpers: GEN_CODE samples the codes in the graph with the sampler of the sequence (`llama_set_sampler`), GEN_WAV keeps the codec state in `llama_memory_slots`, allocated per sequence on first use; the slots of a sequence are removed whole (a partial `llama_memory_seq_rm` returns false) and its position is the last frame decoded. Every other tower, the Qwen3-TTS speaker encoder included, runs on the moved graphs of the clip engine, which also lays out and preprocesses every media.

Flash attention follows the context, except that a tower which opts out of auto flash attention (the lfm3a depthformer) runs without it when the context asked for auto, as in mtmd.

## `tools/mtmd/mtmd.h`

| Item | Notes |
|---|---|
| `mtmd_input_chunk_get_llama_media(chunk)` | TRANSITIONAL: the `llama_media` behind an image or audio chunk, for `mtmd-helper`, which may only use the public mtmd API |

Behaviour of libmtmd:
- `mtmd_init_from_file` loads the mmproj once, with `llama_model_load_from_file`, creates an ENCODER context and a media proc when the file has encoder towers, or, when it has a generation tower, the GEN contexts of the generation states (64 sequences, one per `mtmd_gen_state`) and the reference audio: one GEN context for the engine's pipelines, a code predictor and a codec context for Qwen3-TTS; the GEN context, or the codec context, also encodes the references. A 65th live state adds another set of GEN contexts, so there is no limit on states. The engine moved, the API did not.
- `mtmd_context_params::warmup` is ignored: libllama reserves the compute buffers of a context when it creates it.
- `mtmd_tokenize` keeps the marker parsing, the lazy media (video) and the BOS/EOS handling; one `llama_media_layout` per prompt lays out the media. Images of a prompt are numbered and aligned within it, as before.
- `mtmd_batch_*` hands all its media to one ENCODER batch; libllama groups the ones that share a graph.
- `mtmd_get_output_embd` and `mtmd_batch_get_output_embd` return a copy of the rows owned by the context or batch, valid until its next encode, as before; writes to it do not reach the media, which can serve other encodes of the same chunk. `mtmd_helper_decode_image_chunk` decodes the rows it is given: rows equal to the media's take the media path (positions and attention span from libllama), edited rows the path of explicit rows, as on master.
- Generation states and realtime streams can be freed before or after their context; after it, the stream calls fail and `mtmd_gen_state_reset` does nothing. A state is checked against the context it is passed with, also when a new context takes the address of a freed one. For Qwen3-TTS the codes are sampled in the code predictor graph by a backend sampler of the state that reproduces the clip engine's sampling and random draws (top-k, top-p, one draw per codebook from an `mt19937` seeded by `mtmd_gen_state_params::seed`). `mtmd_gen_state_reset` clears the decoder state and keeps the RNG position and the speaker reference (`llama_gen_reset`). A Qwen3-TTS codec call checks the codes of every frame before it decodes any, so a bad code leaves the state as it was. Realtime streams take the same lock as the generation stages, which run on the same context for LFM3.
- `mtmd_log_set` sets the logger of mtmd only; the encoder runs in libllama and logs through `llama_log_set`.
- `mtmd_get_memory_usage` loads the mmproj with `no_alloc` and sums `llama_get_memory_breakdown` of the contexts `mtmd_init_from_file` creates (the weights once), as `common_fit` does for text models. It silences libllama and ggml while it loads and restores both loggers (`llama_log_get_llama`, as `llama_log_get` returns ggml's).
- `mtmd_get_cap_from_file` reads the GGUF keys with the gguf API.
- `MTMD_DEBUG_EMBEDDINGS=<path>` dumps the encoded rows in the old format; `MTMD_DEBUG_CHUNKS=1` prints the layout of each prompt.
- `MTMD_LM_LOAD_MODE=dio` loads the mmproj with direct I/O (mtmd has no load-mode parameter).
- Serialized chunks hold counts, ids and the decoder position of every image row (version 7); they load back as placeholders, as before, and `mtmd_image_tokens_get_decoder_pos` answers from the stored positions. A placeholder (`mtmd_input_chunk_get_placeholder`, which the server keeps) keeps the positions of the image. Chunks of versions 2 to 5 (the old engine) load too: counts and positions follow from the stored geometry by the old rules, computed when asked (an early version-3 MiniCPM-V 4.7 canvas chunk, whose position type 3 later became SenseNova-U1, is recognised by its separators); a chunk whose geometry gives more than `INT32_MAX` rows, or canvas slices without a grid, fails to load. Version 6 chunks load without positions.

## Internal changes (no API, no ABI)

| Where | Change |
|---|---|
| `LLM_TN_IMPL`, `llm_tensor` | `LLM_TENSOR_NAMED` and `LLM_TN_IMPL::named(arch, name)`: a tensor whose name is taken from the file (output device, MUL_MAT probe, ADD for `.bias`); `is_bias()` |
| `llama_model_loader` | `read_tensor_host(name, dst, size)` for the small tensors the engine keeps on the host (mel filters, voices); `fname`; string-key getters for `bool`, `int32_t`, arrays |
| `llama_model_clip` (`src/models/clip.cpp`, `clip-engine.cpp`, `clip-qwen3tts-wav.cpp`) | holds the engine's models of every modality; native graphs for Gemma 3, Qwen2/2.5/3-VL and Qwen3-TTS share their weights with the engine, whose graphs of these projectors are gone (the Qwen3-TTS code predictor and codec helpers stay, Fish S2 and Drama 3 build on them); `graph_engine` builds the moved graphs inside `llm_graph_context` and registers their flash-attention nodes for the auto-detection |
| `llama_context` | `encode_media`, `encode_media_engine(_group)`, `clip_exec` (one engine graph through `process_ubatch`), `clip_sub` (engine runtime per tower), `gen_engine`; a clip context reserves its graph once (it does not depend on the ubatch) and skips the DeepSeek V4 probes; `llama_cparams::auto_fa_user` |
| `llm_graph_context::build_pooling` | a clip graph without an embedding output (a codec stage with alternative branches) skips pooling |
| `llama_ubatch` / `llama_batch_ext` / `llama_batch_allocr` | media, spans, codes and gen entries; span-aware splitting |
| `llama_kv_cache::set_input_attn_mask` | a query inside a bidirectional span sees the later keys of its span |
| `llama_memory_slots`, `llama_memory_clip_gen` | per-sequence state of the codec (native, allocated by the first use of a sequence, freed by its removal) and of the moved pipelines |

## GGUF

No format change. mmproj files keep `general.architecture = clip`, `clip.*` keys and their tensor names.

## Results

Strix Halo (gfx1151), ROCm build. Two references: mtmd's own engine, whose outputs were stored before its deletion (both engines ran in the same binary, `MTMD_LEGACY_ENGINE=1`), and the master build (`e8fbcad85`) for what has no stored output (video, LFM3-Omni, chunk files, timing, memory). Greedy settings (`--temp 0`, `--top-k 1`, seed 1). Encoder rows through `MTMD_DEBUG_EMBEDDINGS`, prompt layout through `MTMD_DEBUG_CHUNKS`.

### Parity (CPU and GPU)

| Area | Models | Result |
|---|---|---|
| Vision | SmolVLM 500M, SmolVLM2 500M Video, InternVL 2.5 1B, InternVL3 1B, LFM2-VL 450M, granite-docling 258M, LightOnOCR 1B, HunyuanOCR, Qwen2-VL 2B, granite-vision 3.2 2B, dots.ocr, DeepSeek-OCR, MiniCPM-V 2.6, LLaVA 1.5 7B, LLaVA 1.6 Mistral 7B, GLM-Edge-V 5B, Gemma 3 4B, Qwen2.5-VL 3B, Qwen3-VL 2B, Qwen3.5 0.8B, Gemma 4 E4B, Qwen2.5-Omni 7B | layout identical, rows bit-exact, text identical |
| Audio input | ultravox 1B, Qwen3-ASR 0.6B, LFM2-Audio 1.5B, Voxtral Mini 3B, Gemma 4 E4B, Qwen2.5-Omni 7B, Higgs STT, VibeVoice-ASR | same (Qwen3-ASR and Higgs: the old engine dumped only the last chunk; those rows match the last chunk of the new dump) |
| Video (CPU, vs master) | SmolVLM2 Video, InternVL3, LFM2-VL, Qwen2-VL, Gemma 3, Qwen2.5-VL, Qwen3-VL, Qwen3.5, Gemma 4, LFM3-Omni 2B with its soundtrack | layout identical (21 to 241 chunks), text identical; LFM3-Omni on GPU too |
| Realtime audio input | Gemma 4 E4B, Higgs STT, Voxtral Mini 4B Realtime, VibeVoice-ASR | every row bit-exact, greedy tokens identical |
| Generation | Qwen3-TTS 1.7B, Soprano 80M, Chatterbox, Chatterbox-turbo, ZONOS3-Mini, LFM3-Omni 2B (vs master) | WAV byte-identical on CPU and GPU |
| Serialized chunks | chunks of all 30 models above saved by master (version 5), loaded by the branch | same type, token count, position count and id |

Gemma 3, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5, the Qwen2.5-Omni vision tower and Qwen3-TTS generation run on their native graphs, the only graphs of these projectors; the moved graphs they replace gave the same results. Two Qwen3-TTS generation states driven alternately give the codes and audio of one state alone (shared code predictor and codec contexts). Qwen2.5-Omni 7B (image and audio): the text on CPU differs from the stored reference in one token; master gives the branch's text, rows bit-exact, so the reference is the outlier. `llama-mtmd-debug -p encode` on the native towers prints the same rows as master (Gemma 3, Qwen2.5-VL).

On GPU, the native Qwen3-TTS code predictor builds the RoPE position and the mask of a position once for all its layers, as the engine did: from the second layer on, the RMS norm, the norm weight and the RoPE of Q are neighbours and HIP fuses them as it did for the engine, which keeps the WAV byte-identical (with `GGML_CUDA_DISABLE_FUSION=1` both master and the branch give another WAV, the same one).

Not tested, no small GGUF available: Pocket TTS, Drama 3, Fish S2, YuE2, the LFM3-Audio stream. Their code moved unchanged.

Tests: `test-mtmd-impl` (19 cases) and `test-mtmd-c-api` pass; on Windows `test-mtmd-impl` runs the 5 cases outside the engine internals only (libllama exports only `LLAMA_API` there), checked on Linux by compiling without the internals and resolving every symbol from libmtmd, ggml and libc.

### Code size

| | master `e8fbcad85` | branch |
|---|---|---|
| `tools/mtmd` (C/C++) | 42,956 lines | 10,947 lines (`mtmd.cpp` 6,101 -> 2,884) |
| `src` | 120,647 lines | 158,674 lines: `src/clip` 32,920, the rest +5,107 (native graphs of 3 families, media and gen plumbing, slot memories) |
| `src` + `tools/mtmd` | 163,603 | 169,621 (+6,018) |

The net growth is the native graphs (Gemma 3, Qwen-VL, Qwen3-TTS), which replaced the engine's graphs of these projectors, and the bridge (`clip-engine.cpp`, `llama-media.cpp`, `clip-gen-memory.cpp`).

### API surface

| Header | master | branch |
|---|---|---|
| `include/llama.h` | unchanged | unchanged |
| `src/llama-ext.h` (`LLAMA_API` declarations) | 26 | 89 |
| `tools/mtmd/mtmd.h` (`MTMD_API`) | 86 | 87 |
| `tools/mtmd/mtmd-helper.h` (`MTMD_API`) | 36 | 36 |

### Speed (GPU, master build vs branch build, 3 runs each, interleaved)

| Case | master | branch |
|---|---|---|
| Gemma 3 4B, image encode (native) | 777-782 ms | 770-782 ms |
| Qwen2.5-VL 3B, image encode (native) | 337-347 ms | 331-339 ms |
| InternVL3 1B, image encode, 5 tiles | 624-631 ms | 628-631 ms |
| LFM2-VL 450M, image encode | 78-88 ms | 78-86 ms |
| MiniCPM-V 2.6, image encode, slices | 396-412 ms | 406-415 ms |
| Gemma 4 E4B, audio encode | 82-86 ms | 83 ms |
| Voxtral Mini 3B, audio encode | 303-361 ms | 307-315 ms |
| Qwen3-VL 2B, video encode, 20 frames (native) | 1172-2084 ms | 1187-1900 ms |
| LFM3-Omni 2B, video encode with soundtrack | 998-1074 ms | 984-1007 ms |
| Qwen3-TTS 1.7B, llama-tts wall time (native) | 4.97-5.73 s | 4.68-4.86 s |
| Soprano 80M, llama-tts wall time | 1.03-1.10 s | 0.93-1.08 s |
| Chatterbox-turbo, llama-tts wall time | 14.56-16.70 s | 14.14-14.69 s |
| ZONOS3-Mini, llama-tts wall time | 2.21-2.63 s | 1.91-1.95 s |
| LFM3-Omni 2B, llama-tts wall time | 3.12-3.24 s | 3.02-3.18 s |

Equal within noise, after the removal of the moved graphs (native graphs marked); the machine was busier than in the earlier series, both builds alike. Earlier series on a quiet machine: the video runs spread with the machine, both builds slowing down in the same runs (quiet runs 1168-1219 ms master, 1175-1204 ms branch, chunk by chunk the same); two 9.5 s Chatterbox-turbo runs of the branch did not repeat in 4 more runs of each build, with the same timings inside the process (generation 2.01 s, vocoder 1.29-1.34 s). On CPU the machine was shared with another job (load average 7 to 10); a recheck of 4 cases, 5 interleaved runs with 8 threads, gave the branch's best run within 1 % of master's or below it (Gemma 3 image 6566 -> 6084 ms, Voxtral audio 2703 -> 2734 ms, Qwen3-VL video 11824 -> 10611 ms, LFM3-Omni video 6433 -> 5399 ms) and medians within 9 %, which is the noise of that load.

Text only (`llama-bench -m gemma-3-4b-it-Q8_0 -lm dio -ngl 99 -fa 1 -p 1024 -n 128 -d 0,40000 -ub 1024,2048,4096 -b 4096`, master then branch):

| ubatch | test | master t/s | branch t/s |
|---|---|---|---|
| 1024 | pp1024 | 2775 ± 28 | 2707 ± 27 |
| 1024 | tg128 | 47.75 ± 0.02 | 47.63 ± 0.03 |
| 1024 | pp1024 @ d40000 | 1434 ± 28 | 1411 ± 16 |
| 1024 | tg128 @ d40000 | 38.29 ± 0.04 | 38.04 ± 0.03 |
| 2048 | pp1024 | 2483 ± 16 | 2468 ± 26 |
| 2048 | tg128 | 47.66 ± 0.11 | 47.63 ± 0.07 |
| 2048 | pp1024 @ d40000 | 1424 ± 14 | 1402 ± 8 |
| 2048 | tg128 @ d40000 | 38.08 ± 0.11 | 38.01 ± 0.02 |
| 4096 | pp1024 | 2466 ± 21 | 2462 ± 22 |
| 4096 | tg128 | 47.54 ± 0.06 | 47.65 ± 0.07 |
| 4096 | pp1024 @ d40000 | 1417 ± 14 | 1422 ± 31 |
| 4096 | tg128 @ d40000 | 38.07 ± 0.15 | 38.01 ± 0.04 |

All within 2.5 %, below the 3 % noise floor of this machine; the text path does not run the changed code.

### Initialization (`mtmd_init_from_file` with warmup, best of 3)

CPU: the branch is faster for all 30 mmprojs, 6x to 15x (SmolVLM 53 -> 8 ms, Gemma 3 190 -> 14 ms, Qwen2.5-VL 292 -> 25 ms, Qwen3.5 54 -> 4 ms, Gemma 4 216 -> 34 ms, Qwen2.5-Omni 513 -> 49 ms). The weights are mapped instead of read, and an encoder context reserves its graph once, on a placeholder warmup batch. After the removal of the moved graphs, on a CPU loaded by another job, the native-graph models stayed 9 to 13x faster than master (Gemma 3 374 -> 40 ms, Qwen2.5-VL 600 -> 52 ms, Qwen2.5-Omni 1055 -> 104 ms). GPU, after the removal: faster for 29 of 30, equal for SmolVLM2 (49 ms) (Gemma 3 192 -> 152 ms, Qwen2.5-VL 321 -> 299 ms, Qwen3.5 64 -> 52 ms, Qwen2.5-Omni 546 -> 480 ms).

### Memory

`mtmd_get_memory_usage` (CPU) equals master for 33 of 37 mmprojs, Qwen2-VL (1,537 MiB) and Qwen2.5-VL (1,846 MiB) included. Gemma 3 estimates 12 MiB lower (921 vs 933 MiB). Qwen3-TTS estimates 10 MiB higher (1,517 vs 1,507 MiB): the codec state of the first sequence (5 MiB) and the code predictor context (0.8 MiB) come on top of a codec compute buffer of master's size. The three files with both a vision and an audio tower (Qwen2.5-Omni, Gemma 4, LFM3-Omni) estimate 2 to 6 MiB lower: the branch's estimate is its compute buffers plus its weights, which equal master's at runtime (Gemma 4: 143.65 MiB compute, 945.5 MiB weights in both builds); master's estimate is above its own runtime buffers.

Peak RSS of `llama-mtmd-cli` (CPU, Gemma 4 E4B): image +5 MB (8,424 vs 8,429 MB), audio +22 MB (8,528 vs 8,551 MB, two runs each), 0.1 to 0.3 %. After `mtmd_init_from_file` the branch is 4 MB lower; the difference comes during encode: about 5 MB is the compute buffer of the log-mel's CPU scheduler (master ran it on the encoder's scheduler, whose buffers already fit it), the rest host copies of the input and the output copy of the encoded rows next to the rows the media keeps. Qwen3-TTS (`llama-tts`, CPU): +30 MB (8,587 vs 8,617 MB, 0.35 %), WAV identical, 10 % faster (11.1 vs 12.3 s wall).

## Known limitations

- The helper classes `clip_graph_qwen3tts_gen::code_gen` and `::code2wav` keep their Qwen3-TTS name, but only Fish S2 and Drama 3 use them now.
- The moved graphs are mtmd's graphs as they were; they use `clip_graph` helpers, not `llm_graph_context` helpers.
- On Windows, `test-mtmd-impl` runs only the cases that need no engine internals.
- Encoded rows go through host memory.
- Generation contexts process one gen entry at a time; streams of one proc are serialized.
- Peak RSS is 5 to 22 MB above master during media encode (see Memory).

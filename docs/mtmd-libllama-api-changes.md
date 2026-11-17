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
| `llama_media_proc_video_frame_size`, `llama_media_proc_video_markers`, `llama_media_proc_video_template` (`enum llama_media_video_template`) | what mtmd's video helper needs from the model: frame size, the text around a video, how frames are sampled and marked (default, Qwen3-VL, LFM3, LFM3-Omni) |
| `llama_model_media_n_tokens_max(model, type, &atomic)` | the largest span of one media and whether the decoder takes it in one ubatch (server check of `n_ubatch`) |
| `llama_media_layout_init(proc, pos_0)`, `_free`, `_add_tokens`, `_add_image`, `_add_frames`, `_add_audio`, `_n_pieces`, `_get_tokens`, `_get_media` | one layout per prompt; the moved tokenizer turns one media into pieces (tokens and media spans), as mtmd did |
| `llama_media_layout_begin_track(lay, pcm, n_samples, n_pieces)`, `llama_media_layout_add_track_piece(lay, id)` | an audio track cut in pieces that sit in different places of the prompt (the soundtrack of an LFM3-Omni video, a piece after each group of frames): encoded once as one clip, each piece takes its rows |
| `llama_media_debug_encode`, `llama_media_debug_preprocess` | for `llama-mtmd-debug` (model ports), unstable |

The user limits of the image token count are kv overrides of the mmproj model, `clip.vision.image_min_tokens` and `clip.vision.image_max_tokens` (int); `llama_media_proc_params` lost its two fields. The audio preprocessors compute their log-mel on a CPU scheduler owned by the proc (`clip_cpu_compute`, internal), in blocks of 1024 frames.

### Media handle and batch entry

| Function | Notes |
|---|---|
| `llama_media_get_type`, `_n_tokens`, `_n_pos`, `_is_atomic`, `_get_id`, `_is_placeholder`, `_is_encoded`, `_ref`, `_free`, `_release_rows`, `_get_rows` | refcounted, thread-safe to reference |
| `llama_media_rows_host(m)` | the encoded rows in place (NULL if not encoded or not in host memory); valid while the media lives and is not encoded again |
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
| `llama_media_init_voice(model, name)` | a predefined voice as an encoded AUDIO media with its gen-side reference; the voices of lfm3a are names for the chat template and have no rows |
| `llama_batch_ext_add_gen(batch, seq_id, inp)`, `llama_get_gen_out_ith(ctx, i, out)` | one step of one sequence; `llama_process(DECODE)` runs it on a GEN_CODE / GEN_WAV context; both stages run in one context |
| `llama_gen_set_seed(ctx, seq_id, seed)`, `llama_gen_set_ref(ctx, seq_id, media)` | the RNG of a sequence; the speaker reference from an encoded media (chatterbox reference encode, predefined voice) |

The per-sequence state of the moved pipelines (RNG, decoder slots, host DSP such as the LFM3 ISTFT and the ZONOS3 iSTFT head) is the memory of the GEN context (`llama_memory_clip_gen`); `llama_memory_seq_rm(mem, seq, -1, -1)` is a cold start, the state cannot be saved. The moved graphs and pipelines are the default for every model; `LLAMA_CLIP_NATIVE=1` selects the native graphs of Gemma 3, Qwen2/2.5/3-VL and Qwen3-TTS (GEN_CODE with the in-graph sampler, GEN_WAV with `llama_memory_slots`).

Flash attention follows the context, except that a tower which opts out of auto flash attention (the lfm3a depthformer) runs without it when the context asked for auto, as in mtmd.

## `tools/mtmd/mtmd.h`

| Item | Notes |
|---|---|
| `mtmd_input_chunk_get_llama_media(chunk)` | TRANSITIONAL: the `llama_media` behind an image or audio chunk, for `mtmd-helper`, which may only use the public mtmd API |

Behaviour of libmtmd:
- `mtmd_init_from_file` loads the mmproj once, with `llama_model_load_from_file`, creates an ENCODER context and a media proc when the file has encoder towers, or one GEN context (64 sequences, one per `mtmd_gen_state`) that also encodes the reference audio when it has a generation tower. The engine moved, the API did not.
- `mtmd_tokenize` keeps the marker parsing, the lazy media (video) and the BOS/EOS handling; one `llama_media_layout` per prompt lays out the media. Images of a prompt are numbered and aligned within it, as before.
- `mtmd_batch_*` hands all its media to one ENCODER batch; libllama groups the ones that share a graph.
- `mtmd_get_output_embd` and `mtmd_batch_get_output_embd` point into the rows of the encoded media (no copy); valid until the next encode, as before.
- `mtmd_log_set` sets the logger of mtmd only; the encoder runs in libllama and logs through `llama_log_set`.
- `mtmd_get_memory_usage` loads the mmproj with `no_alloc` and sums `llama_get_memory_breakdown` of the encoder context, as `common_fit` does for text models.
- `mtmd_get_cap_from_file` reads the GGUF keys with the gguf API.
- `MTMD_DEBUG_EMBEDDINGS=<path>` dumps the encoded rows in the old format; `MTMD_DEBUG_CHUNKS=1` prints the layout of each prompt.
- `MTMD_LM_LOAD_MODE=dio` loads the mmproj with direct I/O (mtmd has no load-mode parameter).
- Serialized chunks hold counts and ids (version 6); they load back as placeholders, as before. Chunks of versions 2 to 5 (the old engine) load too: their counts follow from the stored geometry by the old rules.

## Internal changes (no API, no ABI)

| Where | Change |
|---|---|
| `LLM_TN_IMPL`, `llm_tensor` | `LLM_TENSOR_NAMED` and `LLM_TN_IMPL::named(arch, name)`: a tensor whose name is taken from the file (output device, MUL_MAT probe, ADD for `.bias`); `is_bias()` |
| `llama_model_loader` | `read_tensor_host(name, dst, size)` for the small tensors the engine keeps on the host (mel filters, voices); `fname`; string-key getters for `bool`, `int32_t`, arrays |
| `llama_model_clip` (`src/models/clip.cpp`, `clip-engine.cpp`, `clip-qwen3tts-wav.cpp`) | holds the engine's models of every modality; native graphs for Gemma 3, Qwen2/2.5/3-VL and Qwen3-TTS share their weights with the engine; `graph_engine` builds the moved graphs inside `llm_graph_context` and registers their flash-attention nodes for the auto-detection |
| `llama_context` | `encode_media`, `encode_media_engine(_group)`, `clip_exec` (one engine graph through `process_ubatch`), `clip_sub` (engine runtime per tower), `gen_engine`; a clip context reserves its graph once (it does not depend on the ubatch) and skips the DeepSeek V4 probes; `llama_cparams::auto_fa_user` |
| `llm_graph_context::build_pooling` | a clip graph without an embedding output (a codec stage with alternative branches) skips pooling |
| `llama_ubatch` / `llama_batch_ext` / `llama_batch_allocr` | media, spans, codes and gen entries; span-aware splitting |
| `llama_kv_cache::set_input_attn_mask` | a query inside a bidirectional span sees the later keys of its span |
| `llama_memory_slots`, `llama_memory_clip_gen` | per-sequence state of the codec (native) and of the moved pipelines |

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

Qwen2.5-Omni 7B (image and audio): the text on CPU now differs from the stored reference in one token; master gives the branch's text, rows bit-exact, so the reference is the outlier. The native graphs (`LLAMA_CLIP_NATIVE=1`) of Qwen3-TTS give a byte-identical WAV on GPU only with `GGML_CUDA_DISABLE_FUSION=1` (HIP fuses neighbouring ops by node order); they are not the default.

Not tested, no small GGUF available: Pocket TTS, Drama 3, Fish S2, YuE2, the LFM3-Audio stream. Their code moved unchanged.

Tests: `test-mtmd-impl` (17 cases) and `test-mtmd-c-api` pass; on Windows `test-mtmd-impl` runs the 3 cases of the mtmd helpers only (libllama exports only `LLAMA_API` there), checked on Linux by compiling without the internals and resolving every symbol from libmtmd, ggml and libc.

### Code size

| | master `e8fbcad85` | branch |
|---|---|---|
| `tools/mtmd` (C/C++) | 42,956 lines | 10,492 lines (`mtmd.cpp` 6,101 -> 2,450) |
| `src` | 120,647 lines | 159,042 lines: `src/clip` 33,476, the rest +4,919 (native graphs of 3 families, media and gen plumbing, slot memories) |
| `src` + `tools/mtmd` | 163,603 | 169,534 (+5,931) |

The net growth is the native graphs (Gemma 3, Qwen-VL, Qwen3-TTS), which exist next to the moved ones, and the bridge (`clip-engine.cpp`, `llama-media.cpp`, `clip-gen-memory.cpp`).

### API surface

| Header | master | branch |
|---|---|---|
| `include/llama.h` | unchanged | unchanged |
| `src/llama-ext.h` (`LLAMA_API` declarations) | 26 | 87 |
| `tools/mtmd/mtmd.h` (`MTMD_API`) | 86 | 87 |
| `tools/mtmd/mtmd-helper.h` (`MTMD_API`) | 36 | 36 |

### Speed (GPU, master build vs branch build, 3 runs each, interleaved)

| Case | master | branch |
|---|---|---|
| Gemma 3 4B, image encode | 783-788 ms | 777-782 ms |
| Qwen2.5-VL 3B, image encode | 338-345 ms | 333-343 ms |
| InternVL3 1B, image encode, 5 tiles | 624-630 ms | 625-633 ms |
| LFM2-VL 450M, image encode | 86-89 ms | 78-87 ms |
| MiniCPM-V 2.6, image encode, slices | 412 ms | 406-411 ms |
| Gemma 4 E4B, audio encode | 82-86 ms | 83-84 ms |
| Voxtral Mini 3B, audio encode | 306-311 ms | 303-305 ms |
| Qwen3-VL 2B, video encode, 20 frames (6 runs) | 1168-1624 ms | 1175-1717 ms |
| LFM3-Omni 2B, video encode with soundtrack | 549-553 ms | 539-551 ms |
| Qwen3-TTS 1.7B, llama-tts wall time | 3.31-3.40 s | 3.28-3.30 s |
| Soprano 80M, llama-tts wall time | 0.58-0.65 s | 0.57-0.59 s |
| Chatterbox-turbo, llama-tts wall time (4 runs) | 6.63-6.84 s | 6.60-6.64 s |
| ZONOS3-Mini, llama-tts wall time | 1.48-1.51 s (one 5.50 s outlier) | 1.41-1.42 s |
| LFM3-Omni 2B, llama-tts wall time | 2.25-2.28 s | 2.22-2.24 s |

Equal within noise. The video runs spread with the machine: both builds slowed down in the same runs, and in quiet runs they took 1168-1219 ms (master) and 1175-1204 ms (branch), chunk by chunk the same. An earlier Chatterbox-turbo series had two 9.5 s runs on the branch; 4 more runs of each build did not repeat it, and the timings inside the process (generation 2.01 s, vocoder 1.29-1.34 s) are the same in both builds. On CPU the machine was shared with another job (load average 7 to 10); a recheck of 4 cases, 5 interleaved runs with 8 threads, gave the branch's best run at or below master's (Gemma 3 image 6566 -> 6084 ms, Voxtral audio 2703 -> 2734 ms, Qwen3-VL video 11824 -> 10611 ms, LFM3-Omni video 6433 -> 5399 ms) and medians within 9 %, which is the noise of that load.

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

CPU: the branch is faster for all 30 mmprojs, 6x to 15x (SmolVLM 53 -> 8 ms, Gemma 3 190 -> 14 ms, Qwen2.5-VL 292 -> 25 ms, Qwen3.5 54 -> 4 ms, Gemma 4 216 -> 34 ms, Qwen2.5-Omni 513 -> 49 ms). The weights are mapped instead of read, and an encoder context reserves its graph once, on a placeholder warmup batch. GPU: the branch is faster for all 30, by 2 % to 42 % (Gemma 3 118 -> 100 ms, Qwen2.5-VL 210 -> 193 ms, Qwen3.5 51 -> 38 ms, Qwen2.5-Omni 363 -> 314 ms).

### Memory

`mtmd_get_memory_usage` (CPU) equals master for 34 of 37 mmprojs, Qwen2.5-VL included (1,846 MiB). The three files with both a vision and an audio tower (Qwen2.5-Omni, Gemma 4, LFM3-Omni) estimate 2 to 6 MiB lower: the branch's estimate is its compute buffers plus its weights, which equal master's at runtime (Gemma 4: 143.65 MiB compute, 945.5 MiB weights in both builds); master's estimate is above its own runtime buffers.

Peak RSS of `llama-mtmd-cli` (CPU, Gemma 4 E4B): image +5 MB (8,424 vs 8,429 MB), audio +15 MB (8,531 vs 8,546 MB), 0.1 to 0.2 %. After `mtmd_init_from_file` the branch is 4 MB lower; the difference comes during encode: about 5 MB is the compute buffer of the log-mel's CPU scheduler (master ran it on the encoder's scheduler, whose buffers already fit it), the rest host copies of the input.

## Known limitations

- The native graphs (Gemma 3, Qwen2/2.5/3-VL, Qwen3-TTS) and the moved graphs of the same families both exist; the native ones are opt-in (`LLAMA_CLIP_NATIVE=1`), one of the two should go.
- The moved graphs are mtmd's graphs as they were; they use `clip_graph` helpers, not `llm_graph_context` helpers.
- On Windows, `test-mtmd-impl` runs only the cases that need no engine internals.
- Encoded rows go through host memory.
- Generation contexts process one gen entry at a time; streams of one proc are serialized.
- Peak RSS is 5 to 15 MB above master during media encode (see Memory).

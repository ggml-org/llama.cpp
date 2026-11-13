# libmtmd dev guide

## History

Please refer to [multimodal.md](../../docs/multimodal.md) for a broader context.

In short:
- `libmtmd` started as a wrapper around `libllava` / `clip.cpp`
- Various components that used to be in `clip.cpp` are moved progressively to mtmd. For example, preprocessor is now part of mtmd
- `clip.cpp` has since been split into `mtmd-model` (hparams and weights, one `mtmd_model` per modality), `mtmd-subcontext` (backends and compute, one `mtmd_subcontext` per modality) and `mtmd-graph` (graph builders, see `models/`). Only the GGUF format keeps the `clip.` name (metadata keys and the `clip` architecture), for compatibility with existing mmproj files

## Terminologies

- mtmd: **M**ul**T**i**M**o**D**al
- bitmap: representing a raw input data, for example: RGB image, PCM audio
- tiles / slices: for llava-uhd-style models, the preprocessor breaks a large input into smaller square images called tiles or slices
- chunk: a mtmd_input_chunk represents a preprocessed input that can then be passed through `mtmd_encode()`

## Structure

Overview of the `libmtmd` structure:

- `mtmd_context` (`mtmd.cpp`): the public handle
    - `mtmd_models`: one mmproj file, loaded by `mtmd_model_load()`
        - `mtmd_model` `v` / `a` / `gen_a` (`mtmd-model.h`): one per modality, holds hparams, weights (`ctx_data` + `buf`) and requirements
    - `mtmd_backends` (`mtmd-subcontext.h`): GPU/CPU backend instances, CPU threadpool and one scheduler, shared by the subcontexts
    - `compute_mutex`: held by every public function that runs a compute pass, so only one pass at a time runs on the backends
    - `mtmd_subcontext` `ctx_v` / `ctx_a` / `ctx_gen_a` (`mtmd-subcontext.h`): one per model, holds a const ref to it and a ref to the backends
        - `buf_compute_meta` and the resolved flash attn of its graphs
        - `mtmd_graph` (`mtmd-graph.h`, `models/`): built for each `encode()` call, temporary
    - `mtmd_image_preprocessor` / `mtmd_audio_preprocessor` (`mtmd-image.h`, `mtmd-audio.h`): read the hparams of their model
- `mtmd_gen_state` / `mtmd_audio_stream`: public, one per sequence / stream
    - `mtmd_subctx_gen_state` / `mtmd_subctx_enc_state`: allocated on the backend of a subcontext, see `gen_state_init()` / `enc_state_init()`

Rules of thumb:
- `mtmd_model` is read-only once loaded. Anything that depends on the backend (flash attn resolution, warmup) belongs to `mtmd_subcontext`
- Like the pp/tg graphs of `llama_context`, the subcontexts share one scheduler: each warmup reserves its worst-case graph on it, so the compute buffers fit the largest one, and switching modality re-plans inside them
- A model declares its runtime needs (e.g. `max_nodes`, `flash_attn_auto`) in `load_hparams`, the subcontext applies them
- Models and backends must outlive the subcontexts, `mtmd_context` declares them first for that reason
- Locking happens only at the public API boundary (`mtmd.cpp`): a new public function that runs a graph must take `compute_mutex`. The internals (subcontexts, preprocessors, `mtmd_audio_log_mel()`) assume it is held and never lock

## Pipeline

A typical pipeline of the core libmtmd is as follows:
- A bitmap (RGB image or PCM audio) is created
- Bitmap and the text prompt is provided to `mtmd_tokenize()` that breaks the input into chunks
    - Alternatively, `mtmd_tokenize_from_parts()` takes a list of pre-split text/media parts instead of a marker-based prompt
    - The tokenizer function first expands a "lazy" bitmap if it finds one. Typically, this is used by video, so that one media token corresponds to one input bitmap
    - For models that support "fused" temporal frames like Qwen-VL, the tokenizer tries to merge pair of consecutive frames into one batch. Only bitmaps marked by `mtmd_bitmap_set_mergeable()` are merged
    - The preprocessor will then be called, which produces a list of chunks
    - Depending on the model itself, special tokens will be injected to separate image chunks (i.e. llava-uhd-style models)
- Multiple bitmaps may be batched together to form a larger `mtmd_batch()`
- Single image or batch is encoded, via `mtmd_encode()` or `mtmd_batch_encode()`
- Get the output embeddings

## Helper

We provide a set of helper functions via `mtmd_helper` to make using libmtmd easier. The helper provides:
- Image, audio and video file decoding (for example, decode raw JPEG into RGB bitmap)
- Manage `llama_batch` and calls to `llama_decode`

## Audio generation support

Audio generation is added to mtmd in PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Currently, we support the 3-stage pipeline below which should cover most TTS models:
- Stage 1: Backbone / Semantic Stage: Backbone model accepts text prompt and reference voice as input
- Stage 2: Acoustic Detail Generator: A model takes the hidden state from backbone and generate audio details (usually as audio codes or mel-spectrogram)
- Stage 3: Waveform Reconstruction: Convert the semantic and acoustic data from previous stages to the final waveform

For example, Qwen3-TTS:
- Reference voice is encoded using ECAPA-TDNN speaker encoder (`speaker_encoder`)
- Text prompt and reference voice are processed via a backbone (`talker.model`)
- A model converts sampled semantic token and hidden state from stage 2 into a list of 15 acoustic codes (`talker.code_predictor`)
- 16 generated codes are converted into waveform (`code2wav`)

### API design constraints

Due to wide variety of audio generation pipelines, the `mtmd_gen_audio` system is designed to be flexible and reusable by new models.

`mtmd_gen_audio` is split into 2 main API:
- Core API `mtmd.h`: handles main inference. Important: `mtmd_gen_state` holds only the per-sequence data that the mmproj itself writes and reads back (RNG, decoder KV / conv state, output buffers). Data that goes to or comes from the backbone (hidden states, codes, PCM) must stay explicit in `mtmd_gen_inp` / `mtmd_gen_out`; the caller drives the backbone and handles audio frame accumulation.
- Helper API `mtmd-helper.h`: provides a model-agnostic stateful API. Usage example can be found in the `tools/tts` directory.

### Checklist for porting new audio generation models to mtmd

1. Make sure to consult merged PRs about adding new TTS models, especially reviewer comments
    - Example: https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+mtmd+tts+is%3Amerged
2. Establish a list of reusable and missing components from the current mtmd implementation.
3. For GGUF conversion:
    - Backbone model should be converted to a normal text model (loadable via `libllama`)
        - If model used hard-coded embedding row ID, append them to token embeddings and assign token name for them (see `qwen3tts.py`)
        - If model have a specific output logits head for audio codes (usually semantic code), keep the head as-is and pad the logits at inference time (see `src/models/qwen3vl.cpp`)
    - Sidecar models (code2wav, bigvgan, etc) must live inside the mmproj GGUF (but can be in different `mtmd_subcontext` if necessary)
        - Note: it should use `ggml_build_forward_select` to select graphs if multiple graphs living in the same context
    - Reuse existing GGUF metadata key name and tensor name whenever possible; think twice before adding extensive changes to GGUF writer. For example, Qwen3-TTS hard-code part of the hparams in `mtmd-model.cpp` as they won't likely to change.
    - For tensor naming:
        - Prefixed with `a.*` for tensors used by speaker encoder pipeline
        - Prefixed with `a.gen.*` for generation stages (code / mel-spectrogram / PCM generation)
    - For GGUF metadata:
        - Reuse as many existing keys as possible
        - In most cases, you can hard-code model configs in the model graph class, or in `mtmd_hparams`
        - If some values need to be exposed to the `mtmd_helper` layer, hard-code them in `mtmd_helper` and distinguish by pipeline and `mtmd_gen_audio_info::model_variant` if necessary
        - Do NOT add new GGUF metadata or new fields to `mtmd_gen_audio_info` unless you can prove that you absolutely need them
4. Make sure most of the changes happen inside `mtmd-helper-gen.cpp`. A good PR looks like this:
    - 10-20% changes is to add new backbone (text) model and conversion
    - 60% changes inside `mtmd-helper-gen.cpp`
    - 10% changes inside `libmtmd` core (`mtmd-model`, `mtmd-subcontext`, `models/`)
    - The rest downstream code (CLI, server) should have no changes at all
5. Update usage documentation in `tools/tts/README.md`

IMPORTANT: If your model needs changes that don't fit the existing infrastructure, **open an issue first for discussion**.

No-go checklist (these will get the PR rejected and require discussion before proceeding):
- Violating the API design constraints stated above
- Adding a new model-specific binary: the API and binary surface must stay model-agnostic
- Adding new flags to conversion script: this script must stay model-agnostic

# libmtmd dev guide

## History

Please refer to [multimodal.md](../../docs/multimodal.md) for a broader context.

In short:
- `libmtmd` started as a wrapper around `libllava` / `clip.cpp`
- Various components that used to be in `clip.cpp` are moved progressively to mtmd. For example, preprocessor is now part of mtmd

## Terminologies

- mtmd: **M**ul**T**i**M**o**D**al
- bitmap: representing a raw input data, for example: RGB image, PCM audio
- tiles / slices: for llava-uhd-style models, the preprocessor breaks a large input into smaller square images called tiles or slices
- chunk: a mtmd_input_chunk represents a preprocessed input that can then be passed through `mtmd_encode()`

## Pipeline

A typical pipeline of the core libmtmd is as follows:
- A bitmap (RGB image or PCM audio) is created
- Bitmap and the text prompt is provided to `mtmd_tokenize()` that breaks the input into chunks
    - The tokenizer function first expands a "lazy" bitmap if it finds one. Typically, this is used by video, so that one media token corresponds to one input bitmap
    - For models that support "fused" temporal frames like Qwen-VL, the tokenizer tries to merge pair of consecutive frames into one batch
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
- Core API `mtmd.h`: handles main inference. Important: the API surface must be stateless; caller must handle state management and audio frame accumulation.
- Helper API `mtmd-helper.h`: provides a model-agnostic stateful API. Usage example can be found in the `tools/tts` directory.

### Checklist for porting new audio generation models to mtmd

1. Establish a list of reusable and missing components from the current mtmd implementation.
2. For GGUF conversion:
    - Backbone model should be converted to a normal text model (loadable via `libllama`)
        - If model used hard-coded embedding row ID, append them to token embeddings and assign token name for them (see `qwen3tts.py`)
        - If model have a specific output logits head for audio codes (usually semantic code), keep the head as-is and pad the logits at inference time (see `src/models/qwen3vl.cpp`)
    - Sidecar models (code2wav, bigvgan, etc) must live inside the mmproj GGUF (but can be in different `clip_context` if necessary)
        - Note: it should use `ggml_build_forward_select` to select graphs if multiple graphs living in the same context
    - Reuse existing GGUF metadata key name and tensor name whenever possible; think twice before adding extensive changes to GGUF writer. For example, Qwen3-TTS hard-code part of the hparams to `clip.cpp` as they won't likely to change.
    - For tensor naming:
        - Prefixed with `a.*` for tensors used by speaker encoder pipeline
        - Prefixed with `a.gen.*` for generation stages (code / mel-spectrogram / PCM generation)
3. Make sure most of the changes happen inside `mtmd-helper-gen.cpp`. A good PR looks like this:
    - 10-20% changes is to add new backbone (text) model and conversion
    - 60% changes inside `mtmd-helper-gen.cpp`
    - 10% changes inside `libmtmd` and `clip.cpp` systems
    - The rest downstream code (CLI, server) should have no changes at all
4. Update usage documentation in `tools/tts/README.md`

IMPORTANT: If your model needs changes that don't fit the existing infrastructure, **open an issue first for discussion**.

No-go checklist (these will get the PR rejected and require discussion before proceeding):
- Violating the API design constraints stated above
- Adding a new model-specific binary: the API and binary surface must stay model-agnostic

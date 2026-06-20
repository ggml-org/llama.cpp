# Output Multimodal Transfer/Decoder

`libomtd` defines the output-side companion boundary for llama.cpp tools.

- `libmtmd`: input media -> model-readable embeddings/tokens
- `libomtd`: model output -> generated media

This patch only adds the OMTD framework and public API. Concrete output models
and frontend wiring are added by follow-up patches.

## Intended Frontend Path

OMTD is intended to be used from the same user-facing tools that already host
model interaction:

```text
llama-cli     --omtd FILE -> generated output file
llama-server  --omtd FILE -> generated output endpoint or stream
llama-tts     --omtd FILE -> generated speech/audio file
```

The framework-only patch does not add those flags yet. It adds the common
library target and API first, so the later frontend patch can route all three
frontends through the same OMTD boundary without introducing a model-specific
backend at the same time.

## Patch Path

A clean review sequence is:

```text
1. OMTD core library and API
2. llama-cli / llama-server / llama-tts frontend wiring
3. First concrete OMTD backend
```

This keeps the generic OMTD design separate from any one output model.

## Scope

OMTD is for generated outputs. It does not replace MTMD and it does not load
`--mmproj` files.

Use:

- `--mmproj FILE` for MTMD input-media projectors
- `--omtd FILE` for OMTD output-media companions, once a frontend/backend patch
  wires a concrete output model to this API

## API

The public C API is declared in `omtd.h`.

Detection helpers:

- `omtd_is_output_companion_gguf(path)`
- `omtd_get_model_type(path)`
- `omtd_get_model_modality(path)`

Generation entry point:

- `omtd_audio_generate_file(params, error, error_size)`

The framework patch returns `OMTD_STATUS_UNSUPPORTED` for audio generation until
an audio backend is registered by a model-specific patch.

## OMTD vs MTMD

| Area | MTMD | OMTD |
| --- | --- | --- |
| Direction | Media input to text model | Text model to generated media output |
| Main flag | `--mmproj` | `--omtd` |
| File role | Input projector/encoder GGUF | Output companion/decoder GGUF |
| Current API shape | tokenize/encode helpers | output generation helpers |

MTMD prepares input media so a language model can consume it. OMTD provides the
library boundary for media generated from model output.

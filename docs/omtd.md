# Output Multimodal Transfer/Decoder

OMTD is a proposed output-side companion library for llama.cpp.

It mirrors MTMD conceptually, but runs in the opposite direction:

```text
MTMD: media input -> embeddings/tokens -> language model
OMTD: language model output -> output companion -> generated media
```

This framework patch adds only the library boundary. Model-specific output
implementations, such as audio, image, or video generators, should be layered on
top in separate patches.

## Intended Frontend Support

The intended frontend path is:

```text
llama-cli
llama-server
llama-tts
    |
    v
  libomtd
    |
    v
concrete output backend
```

Planned frontend responsibilities:

- `llama-cli`: accept `--omtd FILE` for local output generation from a prompt.
- `llama-server`: accept `--omtd FILE` and expose output-generation endpoints.
- `llama-tts`: use OMTD as the shared TTS/audio-output backend path.

This OMTD-only patch does not wire those flags yet. It creates the stable shared
library and API first. The frontend wiring should be a separate patch, followed
by model-specific backend patches.

## Review Path

Recommended patch order:

```text
1. OMTD core library and docs
2. Frontend wiring for llama-cli, llama-server, and llama-tts
3. First concrete OMTD backend
```

This keeps the generic output-runtime boundary reviewable without tying it to a
specific model implementation.

## Responsibilities

OMTD owns:

- A public C API for output companion detection.
- A public C API for structured output generation requests.
- Common status/error reporting for output generation backends.
- Placeholder modality and model-type enums for future backends.

OMTD does not own:

- Input media processing. That remains MTMD.
- `--mmproj` handling.
- A concrete output model in this framework-only patch.
- Application-specific file or HTTP response handling.

## Current API

Detection:

```text
omtd_is_output_companion_gguf(path)
omtd_get_model_type(path)
omtd_get_model_modality(path)
```

Audio generation boundary:

```text
omtd_audio_generate_file(params, error, error_size)
```

The audio generation entry point validates required fields and returns
`OMTD_STATUS_UNSUPPORTED` until a concrete audio backend is added.

## Diagram

```text
llama.cpp frontend
        |
        v
  OMTD public API
        |
        +--> companion detection
        |
        +--> structured output request
        |
        v
 model-specific output backend
   added by later patches
        |
        v
generated media output
```

## Relationship To MTMD

| Area | MTMD | OMTD |
| --- | --- | --- |
| Direction | input -> model | model -> output |
| Media role | user-provided input media | generated output media |
| File role | input projector | output companion |
| API style | tokenize/encode input | generate/decode output |

# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Note: it's highly recommended to always provide a speaker reference voice (via `--tts-speaker-file`); otherwise, the model's performance may be degraded.

Common params:
- Sampling params such as `--top-k`, `--top-p`, `--temp`, etc. `--temp` also sets the temperature of the audio frames for models that support it, without it they use the model default
- `-n <number_of_frames>` limits the output length, e.g. `-n 500`. Note that how many milliseconds each frame represents varies by model
- Core inference params such as `-ngl`, `-b`, `-ub`, etc.

## Supported models

See [docs/multimodal/tts.md](../../docs/multimodal/tts.md) for the list of supported models, their params and GGUF conversion notes.

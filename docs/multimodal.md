# Multimodal

llama.cpp supports multimodal input via `libmtmd`. Currently, there are 2 tools support this feature:
- [llama-mtmd-cli](../tools/mtmd/README.md)
- [llama-server](../tools/server/README.md) via OpenAI-compatible `/chat/completions` API

Currently, we support **image** and **audio** input. Audio is highly experimental and may have reduced quality.

To enable it, you can use one of the 2 methods below:

- Use `-hf` option with a supported model (see a list of pre-quantized model below)
    - To load a model using `-hf` while disabling multimodal, use `--no-mmproj`
    - To load a model using `-hf` while using a custom mmproj file, use `--mmproj local_file.gguf`
- Use `-m model.gguf` option with `--mmproj file.gguf` to specify text and multimodal projector respectively

By default, multimodal projector will be offloaded to GPU. To disable this, add `--no-mmproj-offload`

For example:

```sh
# simple usage with CLI
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF

# simple usage with server
llama-server -hf ggml-org/gemma-3-4b-it-GGUF

# using local file
llama-server -m gemma-3-4b-it-Q4_K_M.gguf --mmproj mmproj-gemma-3-4b-it-Q4_K_M.gguf

# no GPU offload
llama-server -hf ggml-org/gemma-3-4b-it-GGUF --no-mmproj-offload
```

## Pre-quantized models

These are ready-to-use models, most of them come with `Q4_K_M` quantization by default. They can be found at the Hugging Face page of the ggml-org: https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc

Replaces the `(tool_name)` with the name of binary you want to use. For example, `llama-mtmd-cli` or `llama-server`

NOTE: some models may require large context window, for example: `-c 8192`

**Vision models**:

```sh
# Gemma 3
(tool_name) -hf ggml-org/gemma-3-4b-it-GGUF
(tool_name) -hf ggml-org/gemma-3-12b-it-GGUF
(tool_name) -hf ggml-org/gemma-3-27b-it-GGUF

# SmolVLM
(tool_name) -hf ggml-org/SmolVLM-Instruct-GGUF
(tool_name) -hf ggml-org/SmolVLM-256M-Instruct-GGUF
(tool_name) -hf ggml-org/SmolVLM-500M-Instruct-GGUF
(tool_name) -hf ggml-org/SmolVLM2-2.2B-Instruct-GGUF
(tool_name) -hf ggml-org/SmolVLM2-256M-Video-Instruct-GGUF
(tool_name) -hf ggml-org/SmolVLM2-500M-Video-Instruct-GGUF

# Pixtral 12B
(tool_name) -hf ggml-org/pixtral-12b-GGUF

# Qwen 2 VL
(tool_name) -hf ggml-org/Qwen2-VL-2B-Instruct-GGUF
(tool_name) -hf ggml-org/Qwen2-VL-7B-Instruct-GGUF

# Qwen 2.5 VL
(tool_name) -hf ggml-org/Qwen2.5-VL-3B-Instruct-GGUF
(tool_name) -hf ggml-org/Qwen2.5-VL-7B-Instruct-GGUF
(tool_name) -hf ggml-org/Qwen2.5-VL-32B-Instruct-GGUF
(tool_name) -hf ggml-org/Qwen2.5-VL-72B-Instruct-GGUF

# Mistral Small 3.1 24B (IQ2_M quantization)
(tool_name) -hf ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF

# InternVL 2.5 and 3
(tool_name) -hf ggml-org/InternVL2_5-1B-GGUF
(tool_name) -hf ggml-org/InternVL2_5-4B-GGUF
(tool_name) -hf ggml-org/InternVL3-1B-Instruct-GGUF
(tool_name) -hf ggml-org/InternVL3-2B-Instruct-GGUF
(tool_name) -hf ggml-org/InternVL3-8B-Instruct-GGUF
(tool_name) -hf ggml-org/InternVL3-14B-Instruct-GGUF

# Llama 4 Scout
(tool_name) -hf ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF

# Moondream2 20250414 version
(tool_name) -hf ggml-org/moondream2-20250414-GGUF

```

**Audio models**:

```sh
# Ultravox 0.5
(tool_name) -hf ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF
(tool_name) -hf ggml-org/ultravox-v0_5-llama-3_1-8b-GGUF

# Qwen2-Audio and SeaLLM-Audio
# note: no pre-quantized GGUF this model, as they have very poor result
# ref: https://github.com/ggml-org/llama.cpp/pull/13760
```

**Mixed modalities**:

```sh
# Qwen2.5 Omni
# Capabilities: audio input, vision input
(tool_name) -hf ggml-org/Qwen2.5-Omni-3B-GGUF
(tool_name) -hf ggml-org/Qwen2.5-Omni-7B-GGUF
```

## Finding more models:

GGUF models on Huggingface with vision capabilities can be found here: https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending&search=gguf

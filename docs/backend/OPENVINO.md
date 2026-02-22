# OpenVINO Backend for llama.cpp

This document describes the [OpenVINO](https://docs.openvino.ai/) backend for `llama.cpp`, which enables hardware-accelerated inference on **Intel® CPUs, GPUs, and NPUs** while remaining compatible with the existing **GGUF model ecosystem**.

The backend translates GGML compute graphs into OpenVINO graphs and leverages graph compilation, kernel fusion, and device-specific optimizations to improve inference performance on supported Intel hardware.

## Overview

The OpenVINO backend is implemented in ggml/src/ggml-openvino and provides a translation layer for core GGML operations. It supports FP16 and BF16 models, as well as selected quantized GGUF formats. This backend enables accelerated inference on Intel CPUs, integrated and discrete GPUs, and NPUs, while integrating seamlessly with the existing `llama.cpp` execution flow.

## Supported Devices

OpenVINO backend supports the following hardware:

- Intel CPUs
- Intel integrated and discrete GPUs
- Intel NPUs (Requires UD32+ driver)

Although OpenVINO supports a wide range of [Intel hardware](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html), the llama.cpp OpenVINO backend has been validated specifically on AI PCs such as the Intel® Core™ Ultra Series 1 and Series 2.

## Supported Model Precisions

- `FP16`
- `BF16` (on Intel Xeon)
- `Q4_0`
- `Q4_1`
- `Q4_K_M`
- `Q6_K`

Accuracy and performance optimizations for quantized models are still work in progress.

## Quantization Support Details

### CPU and GPU

- **`Q4_0`, `Q4_1`, `Q4_K_M`, `Q6_K` models are supported**
- `Q5_K` and `Q6_K` tensors are converted to `Q8_0_C`

### NPU

- **Primary supported quantization scheme is `Q4_0`**
- `Q6_K` tensors are requantized to `Q4_0_128` in general. For embedding weights, `Q6_K` tensors are requantized to `Q8_0_C` except for the token embedding matrix which is dequantized to fp16

### Additional Notes

- Both `Q4_0` and `Q4_1` models use `Q6_K` for the token embedding tensor and the final matmul weight tensor (often the same tensor)
- `Q4_0` models may produce some `Q4_1` tensors if an imatrix is provided during quantization using `llama-quantize`
- `Q4_K_M` models may include both `Q6_K` and `Q5_K` tensors (observed in Phi-3)

## Validated Models

The following models have been validated for functionality on Intel® Core™ Ultra Series 1 and Series 2:

- [Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF)
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
- [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [openbmb/MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16)
- [tencent/Hunyuan-7B-Instruct](https://huggingface.co/tencent/Hunyuan-7B-Instruct)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF)

## Build Instructions

For detailed build instructions, refer to [build.md](../build.md#openvino)

## Runtime Configuration

The OpenVINO backend can be configured using the following environment variables at runtime to control device selection, caching, debugging, and profiling behavior.

### Configuration Options

| Variable | Description |
|--------|-------------|
| `GGML_OPENVINO_DEVICE` | Specify the target device (`CPU`, `GPU`, `NPU`). When set to `NPU`, static compilation mode is enabled for optimal performance. |
| `GGML_OPENVINO_CACHE_DIR` | Directory for OpenVINO model caching (recommended: `/tmp/ov_cache`). Enables model caching when set. **Not supported on NPU devices.** |
| `GGML_OPENVINO_PROFILING` | Enable execution-time profiling. |
| `GGML_OPENVINO_DUMP_CGRAPH` | Dump the GGML compute graph to `cgraph.txt`. |
| `GGML_OPENVINO_DUMP_IR` | Export OpenVINO IR files with timestamps. |
| `GGML_OPENVINO_DEBUG_INPUT` | Enable input debugging. |
| `GGML_OPENVINO_DEBUG_OUTPUT` | Enable output debugging. |
| `GGML_OPENVINO_STATEFUL_EXECUTION` | Enable stateful execution for better performance |

> [!NOTE]
>`GGML_OPENVINO_STATEFUL_EXECUTION` is an **Experimental** feature to allow stateful execution for managing the KV cache internally inside the OpenVINO model, improving performance on CPUs and GPUs. Stateful execution is not effective on NPUs, and not all models currently support this feature. This feature is experimental and has been validated only with the llama-simple, llama-cli, llama-bench, and llama-run applications and is recommended to enable for the best performance. Other applications, such as llama-server and llama-perplexity, are not yet supported.

### Example Usage

#### GPU Inference with Profiling

```bash
export GGML_OPENVINO_CACHE_DIR=/tmp/ov_cache
export GGML_OPENVINO_PROFILING=1
export GGML_OPENVINO_DEVICE=GPU

./build/ReleaseOV/bin/llama-simple \
  -m ~/models/Llama-3.2-1B-Instruct.fp16.gguf \
  -n 50 \
  "The story of AI is "
```

#### llama-bench

```bash
GGML_OPENVINO_DEVICE=GPU ./llama-bench -fa 1
```
-fa 1 is required when running llama-bench with the OpenVINO backend.

### NPU Notes

- Model caching is not yet supported
- Does not support llama-server -np > 1 (multiple parallel sequences)
- Only supports llama-perplexity -b 512 or smaller

## Llama.cpp Tools

The following tools work with the OpenVINO backend on CPU and GPU: llama-simple, llama-run, llama-cli, llama-server, llama-bench, llama-perplexity.

## Work in Progress

- Performance and memory optimizations
- Broader quantization coverage
- Support for additional model architectures
- Extensive accuracy validation

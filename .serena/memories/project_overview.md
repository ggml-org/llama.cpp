# llama.cpp Project Overview

## Purpose
llama.cpp is a C/C++ inference engine for large language models (LLMs) using the GGUF model format. It provides cross-platform CPU and GPU inference with minimal dependencies.

## Tech Stack
- **Language**: C/C++ (C11, C++17)
- **Build system**: CMake 3.14+ with Ninja preferred
- **GPU backends**: CUDA, Metal, Vulkan, SYCL (Intel), CPU (AVX/NEON)
- **Platform**: Linux (primary dev), macOS, Windows
- **Python**: Used for model conversion scripts (convert_hf_to_gguf.py)

## Key Directories
- `src/` - Main llama library (llama.cpp, llama-*.cpp)
- `include/llama.h` - Public C API header (~2000 lines)
- `ggml/` - Core tensor library (vendored ggml framework)
- `ggml/src/ggml-sycl/` - Intel SYCL backend
- `ggml/src/ggml-cuda/` - NVIDIA CUDA backend
- `ggml/src/ggml-cpu/` - CPU backend
- `common/` - Shared utility code for examples
- `examples/` and `tools/` - 40+ CLI tools
- `tests/` - CTest integration

## Key Binaries (build/bin/)
- `llama-cli` - Interactive chat/inference
- `llama-completion` - Non-interactive text completion
- `llama-server` - OpenAI-compatible HTTP server
- `llama-bench` - Performance benchmarking
- `llama-quantize` - Model quantization

## Machine-Specific
- Intel GPUs: Arc B580 (device 0), Arc Pro B50 (device 1), iGPU (device 2)
- Models stored in /Storage/GenAI/models/
- Default benchmark model: mistral-7b-v0.1.Q4_0.gguf
- Must use ONEAPI_DEVICE_SELECTOR=level_zero:0 for single-GPU operation

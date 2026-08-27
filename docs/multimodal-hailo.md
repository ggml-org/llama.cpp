# Hailo NPU vision backend

This backend offloads the multimodal vision encoder (CLIP/ViT) to a Hailo NPU.
Everything else — text decode, sampling, server — keeps running on CPU exactly as upstream.
Currently only Qwen3-VL is supported.

## Prerequisites

- HailoRT installed.
  See the [HailoRT install guide](https://hailo.ai/developer-zone/documentation/).
- An `hef` file for your VLM, produced by Hailo's Dataflow Compiler.

## Build

```bash
cmake -B build -DLLAMA_HAILO=ON -DGGML_NATIVE=ON
cmake --build build -j$(nproc)
```

`LLAMA_HAILO` defaults to `OFF`; with it off, no Hailo code is compiled and behavior is unchanged from upstream.

## Usage

Across every interface, the integration point is the same: pass the `.hef` wherever you would normally pass the mmproj GGUF.

### Command line

```bash
./build/bin/llama-mtmd-cli \
    -m model.gguf \
    --mmproj /path/to/encoder.hef \
    --image photo.jpg \
    -p "Describe this image"
```

### Server

```bash
./build/bin/llama-server \
    -m model.gguf \
    --mmproj /path/to/encoder.hef \
    --host 127.0.0.1 --port 8080
```

### C++

```cpp
#include "mtmd.h"

mtmd_context_params params = mtmd_context_params_default();
mtmd_context * ctx = mtmd_init_from_file("/path/to/encoder.hef", llama_model, params);
```

See `tools/mtmd/mtmd-cli.cpp` for a full embedding example.

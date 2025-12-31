# llama.cpp + Qwen3-Omni

> **Fork with Qwen3-Omni multimodal architecture support**

[![Qwen3-Omni](https://img.shields.io/badge/Qwen3--Omni-Supported-green)](https://huggingface.co/phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF)
[![Models](https://img.shields.io/badge/GGUF%20Models-HuggingFace-yellow)](https://huggingface.co/phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF)

## What's Added

Support for **Qwen3-Omni**, Alibaba's multimodal LLM:

- `LLM_ARCH_QWEN3OMNI` - Main LLM architecture (MoE: 48 layers, 128 experts)
- `PROJECTOR_TYPE_QWEN3O` - Vision encoder support
- IMROPE position encoding for multimodal inputs

**Note:** Audio encoder support is WIP.

## Quick Start

```bash
# Clone this fork
git clone https://github.com/phnxsystms/llama.cpp.git
cd llama.cpp

# Build with CUDA
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Download models from HuggingFace
huggingface-cli download phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF --local-dir models/

# Text inference
./bin/llama-cli -m models/qwen3-omni-30B-Q8_0.gguf -p "Hello!" -ngl 99

# Vision inference
./bin/llama-mtmd-cli \
    -m models/qwen3-omni-30B-Q8_0.gguf \
    --mmproj models/mmproj-qwen3-omni-30B-F16-fixed.gguf \
    --image your_image.jpg \
    -p "What's in this image?"
```

## Distributed Inference (RPC)

For large models, use llama.cpp's RPC backend to distribute across multiple machines:

```bash
# On worker nodes - start RPC server
./bin/llama-rpc-server --host 0.0.0.0 --port 50052

# On main node - connect to workers
./bin/llama-cli \
    -m models/qwen3-omni-30B-Q8_0.gguf \
    --rpc worker1:50052,worker2:50052 \
    -ngl 99 \
    -p "Hello!"
```

## Models

| Model | Size | Description |
|-------|------|-------------|
| [qwen3-omni-30B-Q8_0.gguf](https://huggingface.co/phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/qwen3-omni-30B-Q8_0.gguf) | 31GB | Main LLM (Q8_0) |
| [mmproj-qwen3-omni-30B-F16-fixed.gguf](https://huggingface.co/phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/mmproj-qwen3-omni-30B-F16-fixed.gguf) | 2.3GB | Vision projector (F16) |

## Performance

Tested on multi-GPU distributed setup:
- **41-44 tokens/sec** inference speed
- Text and vision inference working

## Files Changed

```
src/llama-arch.cpp      # Architecture registration
src/llama-model.cpp     # Model loading & graph building
tools/mtmd/clip.cpp     # Vision projector support
tools/mtmd/mtmd.cpp     # Multimodal pipeline
```

## License

MIT (same as llama.cpp)

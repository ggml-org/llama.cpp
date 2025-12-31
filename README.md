# llama.cpp + Qwen3-Omni

> **Fork with Qwen3-Omni multimodal architecture support**

## Quick Start

```bash
# Clone and build
git clone https://github.com/phnxsystms/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j

# Download models
huggingface-cli download phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF --local-dir models/
```

## Run Server (Recommended)

Spin up an OpenAI-compatible API server:

```bash
./build/bin/llama-server \
    -m models/qwen3-omni-30B-Q8_0.gguf \
    --mmproj models/mmproj-qwen3-omni-30B-F16-fixed.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99
```

Then use it:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

## CLI Usage

```bash
# Text
./build/bin/llama-cli -m models/qwen3-omni-30B-Q8_0.gguf -p "Hello!" -ngl 99

# Vision
./build/bin/llama-mtmd-cli \
    -m models/qwen3-omni-30B-Q8_0.gguf \
    --mmproj models/mmproj-qwen3-omni-30B-F16-fixed.gguf \
    --image photo.jpg \
    -p "Describe this image"
```

## Multi-GPU / Distributed

Model is 31GB - for multi-GPU or distributed inference:

```bash
# Distributed: start RPC on worker machines
./build/bin/llama-rpc-server --host 0.0.0.0 --port 50052

# Main: connect to workers
./build/bin/llama-server \
    -m models/qwen3-omni-30B-Q8_0.gguf \
    --rpc worker1:50052,worker2:50052 \
    -ngl 99
```

## Models

| File | Size |
|------|------|
| [qwen3-omni-30B-Q8_0.gguf](https://huggingface.co/phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF) | 31GB |
| [mmproj-qwen3-omni-30B-F16-fixed.gguf](https://huggingface.co/phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF) | 2.3GB |

## Status

- ✅ Text inference
- ✅ Vision inference  
- 🚧 Audio (WIP)

## License

MIT

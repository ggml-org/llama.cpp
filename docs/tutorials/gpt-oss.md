# Guide: Running gpt-oss with llama.cpp

This guide covers running OpenAI's gpt-oss models locally using llama.cpp with optimal performance across various hardware configurations.

## Model variants

| Model | Parameters | Size on disk | Max context |
|-------|-----------|-------------|-------------|
| gpt-oss 20B | 20.91 B (MoE) | ~12 GB | 128K tokens |
| gpt-oss 120B | 116.83 B (MoE) | ~61 GB | 128K tokens |

Both models use native MXFP4 format for efficient storage and inference.

## Quick start

```sh
# Start llama-server with auto-download from HuggingFace
llama-server -hf ggml-org/gpt-oss-20b-GGUF --jinja -c 0

# 120B model
llama-server -hf ggml-org/gpt-oss-120b-GGUF --jinja -c 0
```

Open `http://127.0.0.1:8080` for the WebUI, or use the OpenAI-compatible API at `http://127.0.0.1:8080/v1`.

## Memory requirements

| Model | Model data | Compute buffers | KV cache per 8K ctx | Total @ 8K | Total @ 32K | Total @ 128K |
|-------|-----------|----------------|--------------------|------------|-------------|-------------|
| gpt-oss 20B | 12.0 GB | 2.7 GB | 0.2 GB | 14.9 GB | 15.5 GB | 17.9 GB |
| gpt-oss 120B | 61.0 GB | 2.7 GB | 0.3 GB | 64.0 GB | 64.9 GB | 68.5 GB |

You do not need to fit the entire model in VRAM. Offloading only attention tensors and KV cache to GPU while keeping the rest in CPU RAM still provides decent performance.

## Key CLI arguments

| Argument | Purpose |
|----------|---------|
| `-hf MODEL_ID` | HuggingFace model ID for auto-download |
| `-c N` | Context size; `-c 0` uses model default (128K max) |
| `-ub N -b N` | Max batch size; larger = more memory but potentially faster |
| `-fa` | Enable Flash Attention for improved performance |
| `--n-cpu-moe N` | Number of MoE layers to keep on CPU (for GPU-limited setups) |
| `--jinja` | Use Jinja chat template from the model file |

## Apple Silicon

Apple Silicon devices share unified memory between CPU and GPU. Stay under 70% of total device memory for best results.

### Devices with >96 GB RAM

M2 Max, M3 Max, M4 Max, M1/M2/M3 Ultra can run both models at full context:

```sh
# gpt-oss 20B
llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 0 --jinja -ub 2048 -b 2048

# gpt-oss 120B
llama-server -hf ggml-org/gpt-oss-120b-GGUF -c 0 --jinja -ub 2048 -b 2048
```

### Devices with 48-96 GB RAM

M4 Pro, M3/M4 Max (48-64 GB):

```sh
# gpt-oss 20B at full context
llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 0 --jinja

# gpt-oss 120B with reduced context
llama-server -hf ggml-org/gpt-oss-120b-GGUF -c 16384 --jinja
```

### Devices with 24-48 GB RAM

```sh
# gpt-oss 20B at moderate context
llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 16384 --jinja
```

### Devices with 16-24 GB RAM

```sh
# gpt-oss 20B with 8K context
llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 8192 --jinja --n-cpu-moe 12
```

### Devices with 8-16 GB RAM

```sh
# gpt-oss 20B minimal config
llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 4096 --jinja --n-cpu-moe 24
```

## Benchmark examples

### M3 Ultra (512 GB, 80 GPU cores) -- gpt-oss 20B

```
| test  | t/s      |
|-------|----------|
| pp2K  | 2816.47  |
| pp8K  | 2308.17  |
| pp16K | 1879.98  |
| pp32K | 1351.67  |
| tg128 | 115.52   |
```

### M2 Ultra (192 GB, 76 GPU cores) -- gpt-oss 20B

```
| test  | t/s      |
|-------|----------|
| pp2K  | 2191.13  |
| pp8K  | 1889.83  |
| pp16K | 1594.51  |
| pp32K | 1218.99  |
| tg128 | 116.08   |
```

Run your own benchmarks:

```sh
llama-bench -m gpt-oss-20b-mxfp4.gguf -t 1 -fa 1 -b 2048 -ub 2048 -p 512,2048,8192
```

## NVIDIA GPU

```sh
# Install with CUDA support first, then:
llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 0 --jinja -fa -ngl auto

# Offload all layers (-ngl all) for max GPU throughput
llama-server -hf ggml-org/gpt-oss-20b-GGUF -c 0 --jinja -fa -ngl all

# 120B on multi-GPU
llama-server -hf ggml-org/gpt-oss-120b-GGUF -c 0 --jinja -fa -ngl all -sm row --tensor-split 3,1
```

## Using gpt-oss programmatically

The server exposes a standard OpenAI-compatible API:

```sh
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Explain quantum computing in 3 sentences."}]
  }'
```

## Converting from original HuggingFace weights

The original models are at:
- https://huggingface.co/openai/gpt-oss-20b
- https://huggingface.co/openai/gpt-oss-120b

Pre-converted GGUF versions are at:
- https://huggingface.co/ggml-org/gpt-oss-20b-GGUF
- https://huggingface.co/ggml-org/gpt-oss-120b-GGUF

To convert manually:

```sh
python convert_hf_to_gguf.py /path/to/gpt-oss-20b --outfile gpt-oss-20b.gguf --outtype mxfp4
```

## Performance tips

- Enable Flash Attention with `-fa` for faster prompt processing
- Tune `--n-cpu-moe` for hybrid CPU/GPU setups by experimenting with values
- Use `-c 0` to automatically use the model's maximum context
- For batch processing, adjust `-b` and `-ub` to match your workload
- The gpt-oss models are MoE; only a subset of parameters are active per token, making them efficient despite large parameter counts

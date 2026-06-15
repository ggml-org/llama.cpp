# Tutorial: Parallel inference using Hugging Face Inference Endpoints

Hugging Face Inference Endpoints natively support GGUF models, allowing you to deploy llama.cpp-powered inference in the cloud with a few clicks. This tutorial covers deploying and using GGUF models on HF Endpoints as well as managing parallel inference workloads.

## Deploying a GGUF model on HF Endpoints

### Option 1: Via the web UI

1. Go to https://ui.endpoints.huggingface.co/
2. Click "New Endpoint"
3. Select "GGUF" as the model format
4. Choose a GGUF model from the HF Hub (e.g., `ggml-org/gemma-3-1b-it-GGUF`)
5. Select hardware configuration (CPU, GPU, multi-GPU)
6. Click "Create Endpoint"

The endpoint deploys llama-server built from the master branch, providing an OpenAI-compatible API.

### Option 2: Via huggingface_hub SDK

```python
from huggingface_hub import create_inference_endpoint

endpoint = create_inference_endpoint(
    name="my-llama-endpoint",
    model_id="ggml-org/gemma-3-1b-it-GGUF",
    framework="gguf",
    task="text-generation",
    type="protected",
    hardware="cpu-up-to-16gb",
    namespace="your-username",
)
```

### Option 3: Via the CLI

```sh
huggingface-cli endpoints create my-llama-endpoint \
  --model-id ggml-org/gemma-3-1b-it-GGUF \
  --framework gguf \
  --task text-generation \
  --type protected \
  --hardware cpu-up-to-16gb
```

## Using the deployed endpoint

The endpoint exposes the standard llama-server API:

```sh
# OpenAI-compatible chat completions
curl https://YOUR-ENDPOINT.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -d '{
    "model": "tgi",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# List available models
curl https://YOUR-ENDPOINT.hf.space/v1/models \
  -H "Authorization: Bearer $HF_TOKEN"
```

## Parallel inference strategies

### Strategy 1: Multi-replica endpoints

For high-throughput workloads, deploy multiple replicas of the same model:

```python
endpoint = create_inference_endpoint(
    name="parallel-llama",
    model_id="ggml-org/gemma-3-1b-it-GGUF",
    framework="gguf",
    hardware="gpu-t4-small",
    replicas=3,  # 3 parallel inference servers
)
```

Each replica is an independent llama-server instance. The endpoint load-balances requests across replicas.

### Strategy 2: Multi-slot llama-server

Within a single llama-server instance, enable parallel slot processing:

```python
# Custom endpoint with environment variables for llama-server CLI args
endpoint = create_inference_endpoint(
    name="multi-slot-llama",
    model_id="ggml-org/gemma-3-1b-it-GGUF",
    framework="gguf",
    hardware="gpu-a10g-large",
    env={
        "LLAMA_ARG_N_PARALLEL": "4",
        "LLAMA_ARG_CTX_SIZE": "32768",
        "LLAMA_ARG_FLASH_ATTN": "1",
    }
)
```

This enables 4 concurrent requests within one instance, sharing the model weights but with independent KV caches.

### Strategy 3: Model-specific endpoints

Deploy separate endpoints for different models, each optimized for its use case:

| Model | Use Case | Hardware | -np |
|-------|----------|----------|-----|
| gemma-3-1b | Embedding/batch | cpu-up-to-8gb | 8 |
| gemma-3-12b | Interactive chat | gpu-a10g-small | 4 |
| gpt-oss-20b | Coding agent | gpu-a10g-large | 1 |
| Llama-3.1-70B | Complex reasoning | gpu-a100-large | 2 |

### Strategy 4: Batch processing via the API

For offline batch inference, use the `/v1/chat/completions` endpoint with multiple concurrent requests:

```python
from concurrent.futures import ThreadPoolExecutor
import requests

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
url = "https://YOUR-ENDPOINT.hf.space/v1/chat/completions"

prompts = ["Translate to French: Hello", "Summarize: ...", "Explain: ..."]

def query(prompt):
    return requests.post(url, headers=headers, json={
        "model": "tgi",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100
    }).json()

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(query, prompts))
```

## Monitoring endpoint performance

```sh
# Get endpoint metrics
curl https://api.endpoints.huggingface.cloud/v2/endpoint/YOUR_USERNAME/my-llama-endpoint \
  -H "Authorization: Bearer $HF_TOKEN" | jq .status

# View logs
huggingface-cli endpoints logs my-llama-endpoint
```

## Cost optimization

- Use **protected** endpoints for lower cost (cold start on first request)
- Use **public** endpoints for always-warm instances (higher cost, zero cold start)
- Scale replicas to zero when idle (`--min-replicas 0`)
- Choose the smallest GPU that fits your model and throughput needs
- Use CPU instances for embedding and batch workloads

## Self-hosted llama-server as HF Endpoint alternative

If you prefer managing your own infrastructure:

```sh
# Start llama-server with HF-compatible settings
llama-server -m model.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 32768 -np 4 --jinja --flash-attn

# Place behind nginx/caddy for HTTPS and authentication
```

Behind a load balancer, you can run multiple llama-server instances for horizontal scaling.

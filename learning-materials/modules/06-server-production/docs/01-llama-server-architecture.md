# LLaMA Server Architecture

**Learning Module**: Module 6 - Server & Production
**Estimated Reading Time**: 25 minutes
**Prerequisites**: Module 1-2 complete, basic understanding of HTTP servers
**Related Content**:
- [RESTful API Design](./02-restful-api-design.md)
- [Deployment Patterns](./03-deployment-patterns.md)
- [Production Best Practices](./06-production-best-practices.md)

---

## Overview

The `llama-server` is llama.cpp's built-in HTTP server that provides an OpenAI-compatible API for LLM inference. It transforms llama.cpp from a command-line tool into a production-ready inference service that can power web applications, mobile apps, and microservices.

### Key Features

1. **OpenAI Compatibility**: Drop-in replacement for OpenAI API
2. **High Performance**: C++ implementation with minimal overhead
3. **Multi-Model Support**: Serve multiple models simultaneously
4. **Streaming Responses**: Real-time token generation via SSE
5. **Embeddings**: Vector generation for RAG and semantic search
6. **Grammar-Guided Generation**: Structured output (JSON, XML)
7. **Built-in UI**: Web interface for testing and demos

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     llama-server Architecture                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐      ┌──────────────┐      ┌─────────────┐ │
│  │   HTTP/HTTPS │──────▶│   Request   │──────▶│   Request  │ │
│  │   Listener   │      │   Router     │      │   Handler  │ │
│  │  (port 8080) │◀──────│             │◀──────│            │ │
│  └─────────────┘      └──────────────┘      └─────────────┘ │
│                              │                      │         │
│                              ▼                      ▼         │
│                       ┌──────────────┐      ┌─────────────┐ │
│                       │   Endpoint   │      │   Model     │ │
│                       │   Handlers   │──────▶│   Manager  │ │
│                       └──────────────┘      └─────────────┘ │
│                              │                      │         │
│  ┌───────────────────────────┴──────────────────────┘        │
│  │                                                            │
│  ▼                                                            │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │              Inference Engine (llama.cpp)                │ │
│ │  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  │ │
│ │  │ Completion │  │  Embeddings │  │  Chat Completion │  │ │
│ │  │   Engine   │  │   Engine    │  │      Engine      │  │ │
│ │  └────────────┘  └─────────────┘  └──────────────────┘  │ │
│ │          │               │                  │            │ │
│ │          └───────────────┴──────────────────┘            │ │
│ │                          │                               │ │
│ │                          ▼                               │ │
│ │                  ┌──────────────┐                        │ │
│ │                  │   KV Cache   │                        │ │
│ │                  │  Management  │                        │ │
│ │                  └──────────────┘                        │ │
│ │                          │                               │ │
│ │                          ▼                               │ │
│ │                  ┌──────────────┐                        │ │
│ │                  │ GPU Backend  │                        │ │
│ │                  │ CUDA/Metal/  │                        │ │
│ │                  │    CPU       │                        │ │
│ │                  └──────────────┘                        │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. HTTP/HTTPS Listener
- **Purpose**: Accept incoming HTTP connections
- **Implementation**: Built on httplib.h (C++ header-only library)
- **Features**:
  - TLS/SSL support
  - CORS handling
  - Connection pooling
  - Request timeout management

#### 2. Request Router
- **Purpose**: Map URLs to endpoint handlers
- **Routes**:
  - `/v1/completions` - Text completion
  - `/v1/chat/completions` - Chat-style completion
  - `/v1/embeddings` - Vector embeddings
  - `/v1/models` - Model information
  - `/health` - Health check
  - `/metrics` - Prometheus metrics

#### 3. Endpoint Handlers
- **Completion Handler**: Process text generation requests
- **Chat Handler**: Handle chat-format conversations
- **Embeddings Handler**: Generate vector representations
- **Model Info Handler**: Return model metadata

#### 4. Model Manager
- **Model Loading**: Load GGUF models into memory
- **Context Management**: Create and manage llama_context instances
- **Multi-Model Support**: Serve multiple models concurrently
- **Memory Management**: Track memory usage and limits

#### 5. Inference Engine
- **Tokenization**: Convert text to tokens
- **Forward Pass**: Run model inference
- **Sampling**: Apply sampling strategies
- **Detokenization**: Convert tokens back to text

---

## Starting the Server

### Basic Usage

```bash
# Start with a single model
llama-server -m models/llama-2-7b.Q4_K_M.gguf

# Specify port and host
llama-server -m model.gguf --host 0.0.0.0 --port 8080

# Enable GPU acceleration
llama-server -m model.gguf -ngl 35

# Set context size
llama-server -m model.gguf -c 4096

# Multiple parallel sequences
llama-server -m model.gguf --parallel 4
```

### Advanced Configuration

```bash
llama-server \
  -m models/llama-2-7b.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  -ngl 35 \
  --parallel 4 \
  --cont-batching \
  --flash-attn \
  --metrics \
  --log-format text \
  --api-key "your-secret-key"
```

**Parameter Breakdown**:
- `-m`: Model path
- `--host`: Bind address (0.0.0.0 for all interfaces)
- `--port`: HTTP port
- `-c`: Context size (token limit)
- `-ngl`: GPU layers (0 = CPU only)
- `--parallel`: Max parallel requests
- `--cont-batching`: Enable continuous batching
- `--flash-attn`: Use FlashAttention
- `--metrics`: Enable Prometheus metrics
- `--api-key`: Authentication key

---

## API Endpoints

### 1. Chat Completion (`/v1/chat/completions`)

**Purpose**: OpenAI-compatible chat API

**Request**:
```json
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama-2-7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false
}
```

**Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "llama-2-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

### 2. Text Completion (`/v1/completions`)

**Purpose**: Raw text completion without chat formatting

**Request**:
```json
POST /v1/completions
Content-Type: application/json

{
  "model": "llama-2-7b",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.95,
  "n": 1,
  "stream": false
}
```

### 3. Embeddings (`/v1/embeddings`)

**Purpose**: Generate vector embeddings

**Request**:
```json
POST /v1/embeddings
Content-Type: application/json

{
  "model": "llama-2-7b",
  "input": "The quick brown fox jumps over the lazy dog"
}
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "llama-2-7b",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### 4. Model Information (`/v1/models`)

**Purpose**: List available models

**Request**:
```bash
GET /v1/models
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-2-7b",
      "object": "model",
      "created": 1699999999,
      "owned_by": "local"
    }
  ]
}
```

---

## Streaming Responses

### Server-Sent Events (SSE)

For real-time token generation:

**Request**:
```json
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama-2-7b",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}
```

**Response Stream**:
```
data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Once"},"index":0}]}

data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" upon"},"index":0}]}

data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" a"},"index":0}]}

...

data: [DONE]
```

### Client Implementation

**Python**:
```python
import requests

url = "http://localhost:8080/v1/chat/completions"
data = {
    "model": "llama-2-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": True
}

with requests.post(url, json=data, stream=True) as response:
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data != '[DONE]':
                    import json
                    chunk = json.loads(data)
                    if chunk['choices'][0]['delta'].get('content'):
                        print(chunk['choices'][0]['delta']['content'], end='', flush=True)
```

**JavaScript**:
```javascript
const response = await fetch('http://localhost:8080/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'llama-2-7b',
    messages: [{ role: 'user', content: 'Hello!' }],
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data !== '[DONE]') {
        const parsed = JSON.parse(data);
        const content = parsed.choices[0].delta?.content;
        if (content) {
          process.stdout.write(content);
        }
      }
    }
  }
}
```

---

## Performance Optimization

### 1. Parallel Request Handling

```bash
# Handle up to 8 requests simultaneously
llama-server -m model.gguf --parallel 8
```

**Benefits**:
- Higher throughput for multiple users
- Better hardware utilization
- Reduced average latency

**Memory Impact**:
```
Memory per slot = context_size * bytes_per_token * 2 (KV cache)
Example: 2048 ctx * 2 bytes * 2 = 8MB per slot
8 slots = 64MB additional memory
```

### 2. Continuous Batching

```bash
llama-server -m model.gguf --cont-batching
```

**How it works**:
- Process multiple requests in a single forward pass
- Add new requests without waiting for existing ones to complete
- Remove completed requests immediately

**Performance Impact**:
- 2-4x throughput improvement
- Better GPU utilization
- Lower latency under load

### 3. Flash Attention

```bash
llama-server -m model.gguf --flash-attn
```

**Benefits**:
- Faster attention computation
- Reduced memory usage
- Supports longer contexts
- Requires compatible GPU (Ampere+)

### 4. GPU Offloading Strategy

```bash
# Optimal GPU layer allocation
llama-server -m model.gguf -ngl 35  # 7B model on 8GB VRAM
llama-server -m model.gguf -ngl 45  # 13B model on 16GB VRAM
llama-server -m model.gguf -ngl 80  # 70B model on 80GB VRAM
```

**Finding optimal `-ngl`**:
```bash
# Start with all layers on GPU
llama-server -m model.gguf -ngl 999 --verbose

# Monitor VRAM usage
# If OOM, reduce by 5-10 layers
# Repeat until stable
```

---

## Production Considerations

### 1. Health Checks

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "slots_available": 3,
  "slots_total": 4
}
```

**Kubernetes Liveness Probe**:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### 2. Metrics & Monitoring

**Endpoint**: `GET /metrics`

**Prometheus Format**:
```
# HELP llama_requests_total Total number of requests
# TYPE llama_requests_total counter
llama_requests_total{endpoint="/v1/chat/completions"} 1234

# HELP llama_request_duration_seconds Request duration
# TYPE llama_request_duration_seconds histogram
llama_request_duration_seconds_bucket{le="0.1"} 45
llama_request_duration_seconds_bucket{le="0.5"} 120
llama_request_duration_seconds_bucket{le="1.0"} 180

# HELP llama_tokens_generated_total Total tokens generated
# TYPE llama_tokens_generated_total counter
llama_tokens_generated_total 567890

# HELP llama_slots_processing Current active slots
# TYPE llama_slots_processing gauge
llama_slots_processing 2
```

### 3. Authentication

**API Key Authentication**:
```bash
llama-server -m model.gguf --api-key "your-secret-key"
```

**Client Request**:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-2-7b", "messages": [...]}'
```

### 4. Rate Limiting

**Not built-in** - Implement at reverse proxy level:

**Nginx Example**:
```nginx
http {
    limit_req_zone $binary_remote_addr zone=llama:10m rate=10r/s;

    server {
        location / {
            limit_req zone=llama burst=20 nodelay;
            proxy_pass http://llama-server:8080;
        }
    }
}
```

---

## Multi-Model Serving

### Approach 1: Multiple Server Instances

```bash
# Model 1 - 7B chat model
llama-server -m llama-2-7b-chat.gguf --port 8080 &

# Model 2 - 13B instruct model
llama-server -m llama-2-13b-instruct.gguf --port 8081 &

# Model 3 - Code generation model
llama-server -m codellama-7b.gguf --port 8082 &
```

**Load Balancer Routes**:
```nginx
upstream chat_backend {
    server localhost:8080;
}

upstream instruct_backend {
    server localhost:8081;
}

upstream code_backend {
    server localhost:8082;
}

server {
    location /chat/ {
        proxy_pass http://chat_backend/;
    }

    location /instruct/ {
        proxy_pass http://instruct_backend/;
    }

    location /code/ {
        proxy_pass http://code_backend/;
    }
}
```

### Approach 2: Model Swapping

**Dynamic Model Loading** (requires custom wrapper):
```python
import subprocess
import signal

class ModelManager:
    def __init__(self):
        self.current_process = None
        self.current_model = None

    def load_model(self, model_path):
        # Stop current server
        if self.current_process:
            self.current_process.send_signal(signal.SIGTERM)
            self.current_process.wait()

        # Start new server
        self.current_process = subprocess.Popen([
            'llama-server',
            '-m', model_path,
            '--port', '8080'
        ])
        self.current_model = model_path
```

---

## Troubleshooting

### Issue: Slow First Request

**Symptom**: First request takes 10+ seconds

**Cause**: Model loading on first request

**Solution**: Warm up the server
```bash
# After starting server, send warmup request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-2-7b","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}'
```

### Issue: Out of Memory

**Symptom**: Server crashes with OOM

**Solutions**:
1. Reduce context size: `-c 2048` instead of `-c 4096`
2. Reduce parallel slots: `--parallel 2` instead of `--parallel 4`
3. Use more aggressive quantization: Q4_K_M instead of Q5_K_M
4. Reduce GPU layers: `-ngl 30` instead of `-ngl 35`

### Issue: High Latency

**Symptom**: Requests take >5 seconds

**Diagnosis**:
```bash
# Check metrics
curl http://localhost:8080/metrics | grep duration

# Check server logs
llama-server -m model.gguf --log-format json | jq '.duration_ms'
```

**Solutions**:
1. Enable continuous batching
2. Enable Flash Attention
3. Increase GPU layers
4. Use faster quantization (Q4_0 vs Q5_K_M)
5. Reduce context size

---

## Best Practices

### 1. Resource Allocation

**CPU-Only Deployment**:
```yaml
resources:
  requests:
    cpu: "4"
    memory: "8Gi"
  limits:
    cpu: "8"
    memory: "16Gi"
```

**GPU Deployment**:
```yaml
resources:
  requests:
    nvidia.com/gpu: 1
    memory: "16Gi"
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi"
```

### 2. Logging

**Structured JSON Logging**:
```bash
llama-server -m model.gguf --log-format json | \
  jq -r '[.timestamp, .level, .message] | @tsv'
```

**Log Levels**:
- `--log-disable`: No logging
- Default: Info level
- `--verbose`: Debug level

### 3. Security

**Checklist**:
- ✅ Use `--api-key` for authentication
- ✅ Enable TLS/SSL for production
- ✅ Implement rate limiting at proxy level
- ✅ Run with non-root user
- ✅ Use network policies in Kubernetes
- ✅ Regularly update llama.cpp
- ✅ Monitor for abuse patterns

### 4. Monitoring

**Key Metrics**:
- Request rate (req/s)
- Token generation rate (tokens/s)
- Request duration (p50, p95, p99)
- Active slots
- Memory usage
- GPU utilization
- Error rate

---

## Example Production Setup

```bash
#!/bin/bash
# production-llama-server.sh

MODEL_PATH="/models/llama-2-7b-chat.Q4_K_M.gguf"
PORT=8080
CTX_SIZE=4096
PARALLEL=8
GPU_LAYERS=35

llama-server \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  -c "$CTX_SIZE" \
  -ngl "$GPU_LAYERS" \
  --parallel "$PARALLEL" \
  --cont-batching \
  --flash-attn \
  --metrics \
  --log-format json \
  --api-key "${LLAMA_API_KEY}" \
  2>&1 | tee -a /var/log/llama-server.log
```

---

## Summary

The llama-server provides a production-ready HTTP API for LLM inference:

**Key Takeaways**:
1. **OpenAI Compatible**: Drop-in replacement for OpenAI API
2. **High Performance**: Continuous batching, Flash Attention, GPU acceleration
3. **Production Ready**: Metrics, health checks, streaming, authentication
4. **Flexible**: Supports multiple deployment patterns
5. **Efficient**: Minimal overhead, optimized C++ implementation

**Next Steps**:
- [RESTful API Design](./02-restful-api-design.md) - Design custom endpoints
- [Deployment Patterns](./03-deployment-patterns.md) - Deploy to production
- [Load Balancing & Scaling](./04-load-balancing-scaling.md) - Scale horizontally

---

**Learning Resources**:
- Lab 6.1: Deploy llama-server
- Tutorial: Building a production inference API
- Code Example: Custom server wrapper

**Interview Topics**:
- Server architecture design
- API endpoint design
- Performance optimization strategies
- Production deployment considerations

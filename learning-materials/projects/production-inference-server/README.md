# Capstone Project: Production Inference Server

**Difficulty**: Advanced (Senior Level)
**Estimated Time**: 40-60 hours
**Modules Required**: 1-6, 9
**Prerequisites**: Python, C++, Docker, Kubernetes basics

---

## Project Overview

Build a production-ready inference server using llama.cpp with enterprise features: high availability, monitoring, rate limiting, and security.

**Learning Outcomes**:
- ✅ Production server architecture
- ✅ High-performance API design
- ✅ Continuous batching implementation
- ✅ Observability and monitoring
- ✅ Security and compliance
- ✅ Deployment and scaling

---

## System Requirements

### Functional Requirements

1. **API Endpoints**:
   - POST /v1/completions (OpenAI-compatible)
   - POST /v1/chat/completions (streaming support)
   - POST /v1/embeddings
   - GET /v1/models
   - GET /health
   - GET /metrics (Prometheus)

2. **Performance Targets**:
   - 1000 requests/sec sustained throughput
   - p99 latency < 2 seconds
   - 99.9% uptime SLA
   - Support 100 concurrent connections

3. **Features**:
   - Continuous batching for efficiency
   - Request queue with prioritization
   - Rate limiting per API key
   - Streaming response support
   - Graceful shutdown
   - Health checks and readiness probes

### Non-Functional Requirements

1. **Security**:
   - API key authentication
   - HTTPS/TLS termination
   - Rate limiting (token bucket)
   - Input validation and sanitization
   - PII detection (optional)

2. **Observability**:
   - Prometheus metrics export
   - Structured JSON logging
   - Distributed tracing (OpenTelemetry)
   - Custom dashboards (Grafana)

3. **Reliability**:
   - Circuit breaker for dependencies
   - Request timeout handling
   - Graceful degradation
   - Automatic recovery

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer (nginx)                    │
└────────────┬────────────────────────────────────────────────┘
             │
     ┌───────┴────────┐
     │                │
┌────▼─────┐   ┌─────▼────┐       ┌──────────────┐
│ Server 1 │   │ Server 2 │  ...  │   Server N   │
└────┬─────┘   └─────┬────┘       └──────┬───────┘
     │               │                    │
┌────▼───────────────▼────────────────────▼────┐
│         llama.cpp Inference Engine            │
│    ┌──────────────────────────────────┐      │
│    │   Continuous Batching Manager    │      │
│    │  ┌────────────────────────────┐  │      │
│    │  │  Request Queue (Priority)  │  │      │
│    │  └────────────────────────────┘  │      │
│    │  ┌────────────────────────────┐  │      │
│    │  │    KV Cache Manager        │  │      │
│    │  └────────────────────────────┘  │      │
│    └──────────────────────────────────┘      │
└───────────────────────────────────────────────┘
             │                   │
    ┌────────▼────────┐   ┌─────▼──────┐
    │   Prometheus    │   │  Grafana   │
    │   (Metrics)     │   │ (Dashboard)│
    └─────────────────┘   └────────────┘
```

---

## Implementation Guide

### Phase 1: Basic Server (Week 1)

**Tasks**:
1. Setup FastAPI/Flask application
2. Implement /v1/completions endpoint
3. Load llama.cpp model
4. Basic request handling (no batching)
5. Error handling and validation

**Deliverables**:
- `server.py` - Main server implementation
- `config.yaml` - Configuration
- `requirements.txt` - Dependencies
- Basic tests

**Example Code**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()
model = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf")

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    try:
        output = model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return {"choices": [{"text": output["choices"][0]["text"]}]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Phase 2: Continuous Batching (Week 2)

**Tasks**:
1. Implement request queue
2. Continuous batching engine
3. KV cache management
4. Request prioritization
5. Performance optimization

**Key Algorithm**:
```python
class ContinuousBatcher:
    def __init__(self, max_batch_size=32):
        self.active_requests = []
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size

    async def process_loop(self):
        while True:
            # Remove completed requests
            self.active_requests = [r for r in self.active_requests 
                                    if not r.is_complete()]

            # Add new requests
            while len(self.active_requests) < self.max_batch_size:
                try:
                    req = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=0.01
                    )
                    self.active_requests.append(req)
                except asyncio.TimeoutError:
                    break

            if not self.active_requests:
                await asyncio.sleep(0.001)
                continue

            # Generate one token for each request
            batch_inputs = [r.get_next_input() for r in self.active_requests]
            outputs = await self.batch_inference(batch_inputs)

            for req, output in zip(self.active_requests, outputs):
                await req.add_token(output)
```

### Phase 3: Monitoring & Observability (Week 3)

**Tasks**:
1. Prometheus metrics integration
2. Structured logging
3. OpenTelemetry tracing
4. Grafana dashboard
5. Alerting rules

**Metrics to Track**:
```python
from prometheus_client import Counter, Histogram, Gauge

request_count = Counter(
    'llama_requests_total',
    'Total requests',
    ['endpoint', 'status']
)

latency = Histogram(
    'llama_request_duration_seconds',
    'Request latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

active_requests = Gauge(
    'llama_active_requests',
    'Currently active requests'
)

tokens_per_second = Gauge(
    'llama_tokens_per_second',
    'Inference throughput'
)
```

### Phase 4: Security & Rate Limiting (Week 4)

**Tasks**:
1. API key authentication
2. Rate limiting (token bucket)
3. Input validation
4. CORS configuration
5. Security headers

**Rate Limiter**:
```python
class TokenBucketRateLimiter:
    def __init__(self, rate: int, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()

    async def acquire(self, cost: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
```

### Phase 5: Production Deployment (Week 5-6)

**Tasks**:
1. Docker containerization
2. Kubernetes deployment
3. Horizontal Pod Autoscaler
4. Ingress configuration
5. CI/CD pipeline

**Docker File**:
```dockerfile
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git

# Install llama.cpp
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama-server
  template:
    metadata:
      labels:
        app: llama-server
    spec:
      containers:
      - name: llama
        image: llama-server:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## Testing Plan

### Unit Tests
- API endpoint validation
- Request parsing
- Rate limiter logic
- KV cache management

### Integration Tests
- End-to-end API calls
- Concurrent request handling
- Error scenarios
- Graceful shutdown

### Load Tests
- Apache Bench / Locust
- 1000 req/sec sustained
- Latency distribution
- Resource utilization

**Example Load Test**:
```python
from locust import HttpUser, task, between

class LlamaUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def completion(self):
        self.client.post("/v1/completions", json={
            "prompt": "Hello, how are you?",
            "max_tokens": 50
        }, headers={"Authorization": "Bearer test-key"})
```

---

## Evaluation Criteria

### Functionality (40%)
- ✅ All API endpoints working
- ✅ Continuous batching implemented
- ✅ Rate limiting functional
- ✅ Streaming support

### Performance (30%)
- ✅ Meets throughput target (1000 req/sec)
- ✅ Meets latency target (p99 < 2s)
- ✅ Efficient resource utilization
- ✅ Graceful degradation

### Code Quality (20%)
- ✅ Clean, modular code
- ✅ Comprehensive tests (>80% coverage)
- ✅ Documentation
- ✅ Error handling

### Production Readiness (10%)
- ✅ Deployment automation
- ✅ Monitoring and alerting
- ✅ Security best practices
- ✅ Operational runbook

---

## Extensions (Optional)

1. **Advanced Features**:
   - Multi-model support
   - A/B testing framework
   - Request caching
   - Speculative decoding

2. **Scale**:
   - Multi-region deployment
   - Auto-scaling based on queue depth
   - Cost optimization
   - GPU resource pooling

3. **Integrations**:
   - S3 model loading
   - Redis for distributed rate limiting
   - Kafka for event streaming
   - DataDog/NewRelic integration

---

## Resources

**Code Examples**: `/learning-materials/projects/production-inference-server/examples/`
**Reference Implementation**: `/learning-materials/projects/production-inference-server/reference/`
**Documentation**: `/learning-materials/modules/06-server-production/`

---

**Project Maintainer**: Agent 8 (Integration Coordinator)
**Last Updated**: 2025-11-18
**Estimated Completion**: 6 weeks part-time, 3 weeks full-time

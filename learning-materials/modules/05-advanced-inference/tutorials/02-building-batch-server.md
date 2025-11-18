# Tutorial 2: Building a Production Batch Inference Server

**Module 5 - Advanced Inference**
**Duration**: 90-120 minutes

## Overview

Build a production-grade batching server with continuous batching, request queuing, monitoring, and auto-scaling capabilities.

## Architecture

```
┌─────────────┐
│   Clients   │
└──────┬──────┘
       │
┌──────▼──────────────────┐
│   Load Balancer         │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│   Request Queue         │
│   (Priority-based)      │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│  Batch Scheduler        │
│  (Continuous Batching)  │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│   GPU Inference         │
│   (llama.cpp)           │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│   Monitoring            │
│   (Prometheus/Grafana)  │
└─────────────────────────┘
```

## Step 1: Core Server (30 min)

### FastAPI Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

app = FastAPI(title="LLaMA Batch Inference Server")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    priority: int = 0  # 0=low, 1=medium, 2=high

class GenerateResponse(BaseModel):
    request_id: str
    text: str
    tokens_generated: int
    latency_ms: float
    batch_wait_ms: float

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Submit generation request"""
    import uuid
    request_id = str(uuid.uuid4())

    # Submit to batch scheduler
    result = await batch_scheduler.submit(
        request_id=request_id,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        priority=request.priority
    )

    return GenerateResponse(**result)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_requests": batch_scheduler.active_count(),
        "queue_depth": batch_scheduler.queue_depth(),
        "gpu_utilization": get_gpu_utilization()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    return batch_scheduler.get_metrics()
```

## Step 2: Continuous Batch Scheduler (40 min)

### Scheduler Implementation

```python
import heapq
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    timestamp: float
    request_id: str = field(compare=False)
    prompt: List[int] = field(compare=False)
    max_tokens: int = field(compare=False)
    future: asyncio.Future = field(compare=False)

class ContinuousBatchScheduler:
    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Load model
        self.model = load_llama_model(model_path)

        # Request queue (priority heap)
        self.queue: List[PrioritizedRequest] = []

        # Active sequences
        self.active: Dict[str, SequenceState] = {}

        # Metrics
        self.metrics = BatchMetrics()

        # Start background processing
        self.running = True
        asyncio.create_task(self._processing_loop())

    async def submit(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        priority: int
    ):
        """Submit request and wait for result"""
        import time

        # Tokenize
        tokens = self.model.tokenize(prompt)

        # Create future for result
        future = asyncio.Future()

        # Add to priority queue
        req = PrioritizedRequest(
            priority=-priority,  # Negate for max-heap
            timestamp=time.time(),
            request_id=request_id,
            prompt=tokens,
            max_tokens=max_tokens,
            future=future
        )

        heapq.heappush(self.queue, req)

        # Wait for result
        result = await future
        return result

    async def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            # Step 1: Remove completed sequences
            self._remove_completed()

            # Step 2: Add new requests from queue
            await self._fill_batch()

            # Step 3: Generate one token for all active
            if self.active:
                await self._generation_step()

            # Small sleep if no work
            if not self.active:
                await asyncio.sleep(0.001)

    def _remove_completed(self):
        """Remove completed sequences"""
        completed_ids = [
            req_id for req_id, seq in self.active.items()
            if seq.is_complete()
        ]

        for req_id in completed_ids:
            seq = self.active.pop(req_id)

            # Return result
            seq.future.set_result({
                "request_id": req_id,
                "text": self.model.detokenize(seq.tokens),
                "tokens_generated": seq.generated_tokens(),
                "latency_ms": seq.latency_ms(),
                "batch_wait_ms": seq.wait_time_ms()
            })

            self.metrics.record_completion(seq)

    async def _fill_batch(self):
        """Fill batch with waiting requests"""
        import time

        deadline = time.time() + self.max_wait_ms / 1000.0

        while len(self.active) < self.max_batch_size and self.queue:
            # Check deadline
            if time.time() > deadline and self.active:
                break

            # Pop highest priority request
            req = heapq.heappop(self.queue)

            # Create sequence state
            seq = SequenceState(
                request_id=req.request_id,
                tokens=req.prompt,
                max_tokens=req.max_tokens,
                future=req.future,
                start_time=time.time(),
                enqueue_time=req.timestamp
            )

            self.active[req.request_id] = seq

    async def _generation_step(self):
        """Generate one token for all active sequences"""
        # Prepare batch
        batch_tokens = [
            seq.tokens for seq in self.active.values()
        ]

        # Batched inference
        logits_batch = await self.model.forward_batch(batch_tokens)

        # Sample for each sequence
        for seq, logits in zip(self.active.values(), logits_batch):
            token = self.model.sample(logits, temperature=seq.temperature)
            seq.tokens.append(token)

        self.metrics.record_iteration(len(self.active))
```

## Step 3: Monitoring (20 min)

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

class BatchMetrics:
    def __init__(self):
        # Counters
        self.requests_total = Counter(
            'requests_total',
            'Total requests processed'
        )
        self.tokens_generated = Counter(
            'tokens_generated_total',
            'Total tokens generated'
        )

        # Histograms
        self.latency = Histogram(
            'request_latency_seconds',
            'Request latency in seconds'
        )
        self.batch_size = Histogram(
            'batch_size',
            'Batch size per iteration'
        )

        # Gauges
        self.active_requests = Gauge(
            'active_requests',
            'Currently active requests'
        )
        self.queue_depth = Gauge(
            'queue_depth',
            'Requests waiting in queue'
        )

    def record_completion(self, seq):
        """Record completed request"""
        self.requests_total.inc()
        self.tokens_generated.inc(seq.generated_tokens())
        self.latency.observe(seq.latency_ms() / 1000.0)

    def record_iteration(self, batch_size):
        """Record iteration metrics"""
        self.batch_size.observe(batch_size)
        self.active_requests.set(batch_size)
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log key events
logger.info(f"Request {request_id} submitted (priority={priority})")
logger.info(f"Batch iteration: size={batch_size}, util={util:.1f}%")
logger.warning(f"Queue depth high: {queue_depth}")
logger.error(f"Request {request_id} failed: {error}")
```

## Step 4: Deployment (20 min)

### Docker Container

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    cmake \
    build-essential

# Build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp
WORKDIR /llama.cpp
RUN mkdir build && cd build && \
    cmake .. -DLLAMA_CUDA=ON && \
    cmake --build . --config Release

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  batch-server:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    environment:
      - MODEL_PATH=/models/llama-13b.gguf
      - MAX_BATCH_SIZE=32
      - MAX_WAIT_MS=50
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

## Step 5: Testing (20 min)

### Load Testing

```python
import asyncio
import aiohttp
import time

async def load_test(
    num_requests: int = 100,
    concurrent: int = 10
):
    """Load test the batch server"""
    async with aiohttp.ClientSession() as session:
        tasks = []

        for i in range(num_requests):
            task = submit_request(
                session,
                prompt=f"Explain topic {i}",
                priority=i % 3  # Mix priorities
            )
            tasks.append(task)

            # Limit concurrency
            if len(tasks) >= concurrent:
                results = await asyncio.gather(*tasks)
                tasks = []

                # Analyze results
                latencies = [r['latency_ms'] for r in results]
                print(f"Batch complete: avg_latency={np.mean(latencies):.1f}ms")

        # Complete remaining
        if tasks:
            await asyncio.gather(*tasks)

async def submit_request(session, prompt, priority):
    """Submit single request"""
    start = time.time()

    async with session.post(
        'http://localhost:8000/v1/generate',
        json={
            'prompt': prompt,
            'max_tokens': 100,
            'priority': priority
        }
    ) as resp:
        result = await resp.json()
        result['total_latency_ms'] = (time.time() - start) * 1000
        return result

# Run load test
asyncio.run(load_test(num_requests=1000, concurrent=50))
```

## Production Checklist

- [ ] Continuous batching implemented
- [ ] Priority queuing working
- [ ] Metrics collection enabled
- [ ] Logging configured
- [ ] Docker deployment tested
- [ ] Load testing passed
- [ ] Monitoring dashboards created
- [ ] Auto-scaling configured

## Performance Targets

✅ **Throughput**: >500 tokens/sec
✅ **P50 Latency**: <200ms
✅ **P95 Latency**: <500ms
✅ **GPU Utilization**: >80%
✅ **Batch Efficiency**: >90%

## Next Steps

1. Deploy to Kubernetes
2. Add auto-scaling
3. Implement request caching
4. Add A/B testing
5. Set up alerting

See complete code:
- `../code/continuous_batching_simulator.py`
- Lab: `../labs/lab2-batch-optimization.md`

🚀 **Your production batch server is ready!**

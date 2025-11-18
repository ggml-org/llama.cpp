# Performance at Scale

## Introduction

Scaling LLM inference systems to handle production workloads requires careful attention to performance, resource utilization, and system architecture. This lesson covers strategies for achieving high throughput and low latency at scale.

## Understanding Performance Metrics

### Key Metrics

1. **Latency**
   - Time to First Token (TTFT)
   - Time per Output Token (TPOT)
   - End-to-end latency

2. **Throughput**
   - Requests per second (RPS)
   - Tokens per second (TPS)
   - Concurrent requests supported

3. **Resource Utilization**
   - GPU utilization %
   - Memory bandwidth
   - CPU usage
   - Network bandwidth

4. **Quality**
   - Error rate
   - Timeout rate
   - P50, P95, P99 latencies

### Measuring Performance

```python
# monitoring/metrics.py
import time
from dataclasses import dataclass
from typing import List
import statistics

@dataclass
class PerformanceMetrics:
    time_to_first_token: float
    total_latency: float
    tokens_generated: int
    prompt_tokens: int

    @property
    def time_per_output_token(self) -> float:
        if self.tokens_generated == 0:
            return 0
        return (self.total_latency - self.time_to_first_token) / self.tokens_generated

    @property
    def throughput_tps(self) -> float:
        if self.total_latency == 0:
            return 0
        return self.tokens_generated / self.total_latency

class PerformanceMonitor:
    def __init__(self):
        self.measurements = []

    def measure_inference(self, model, prompt: str, max_tokens: int) -> PerformanceMetrics:
        """Measure single inference performance"""
        start_time = time.perf_counter()
        first_token_time = None
        tokens_generated = 0

        # Tokenize prompt
        prompt_tokens = len(model.tokenize(prompt.encode()))

        # Generate with streaming to measure TTFT
        for chunk in model(prompt, max_tokens=max_tokens, stream=True):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens_generated += 1

        end_time = time.perf_counter()

        metrics = PerformanceMetrics(
            time_to_first_token=first_token_time - start_time if first_token_time else 0,
            total_latency=end_time - start_time,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens
        )

        self.measurements.append(metrics)
        return metrics

    def get_statistics(self) -> dict:
        """Calculate aggregate statistics"""
        if not self.measurements:
            return {}

        ttfts = [m.time_to_first_token for m in self.measurements]
        latencies = [m.total_latency for m in self.measurements]
        tpots = [m.time_per_output_token for m in self.measurements]

        return {
            'count': len(self.measurements),
            'ttft': {
                'mean': statistics.mean(ttfts),
                'median': statistics.median(ttfts),
                'p95': sorted(ttfts)[int(len(ttfts) * 0.95)],
                'p99': sorted(ttfts)[int(len(ttfts) * 0.99)],
            },
            'latency': {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': sorted(latencies)[int(len(latencies) * 0.95)],
                'p99': sorted(latencies)[int(len(latencies) * 0.99)],
            },
            'tpot': {
                'mean': statistics.mean(tpots),
                'median': statistics.median(tpots),
            },
            'throughput_tps': sum(m.throughput_tps for m in self.measurements) / len(self.measurements)
        }
```

## Batching Strategies

### Continuous Batching

```python
# server/continuous_batching.py
import asyncio
from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass
class Request:
    request_id: str
    prompt: str
    max_tokens: int
    generated_tokens: int = 0
    result: List[int] = None
    finished: bool = False

class ContinuousBatcher:
    def __init__(self, model, max_batch_size: int = 8, max_wait_ms: int = 100):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()
        self.active_requests: List[Request] = []

    async def add_request(self, request: Request):
        """Add request to queue"""
        await self.queue.put(request)

    async def process_loop(self):
        """Main processing loop"""
        while True:
            # Collect batch
            batch = await self._collect_batch()

            if not batch:
                await asyncio.sleep(0.01)
                continue

            # Process batch
            await self._process_batch(batch)

    async def _collect_batch(self) -> List[Request]:
        """Collect requests for batching"""
        batch = []
        deadline = time.time() + (self.max_wait_ms / 1000)

        # Add ongoing requests
        batch.extend([r for r in self.active_requests if not r.finished])

        # Add new requests up to batch size
        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                request = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=max(0, deadline - time.time())
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        return batch

    async def _process_batch(self, batch: List[Request]):
        """Process a batch of requests"""
        # Prepare batch inputs
        prompts = [r.prompt for r in batch]

        # Generate one token for each request in batch
        outputs = self.model.generate_batch(prompts, num_tokens=1)

        # Update requests
        for i, request in enumerate(batch):
            token = outputs[i]

            if request.result is None:
                request.result = []
            request.result.append(token)
            request.generated_tokens += 1

            # Check if finished
            if (request.generated_tokens >= request.max_tokens or
                token == self.model.token_eos()):
                request.finished = True

        # Update active requests
        self.active_requests = [r for r in batch if not r.finished]

# Usage
batcher = ContinuousBatcher(model, max_batch_size=8)

async def handle_request(prompt: str, max_tokens: int):
    request = Request(
        request_id=generate_id(),
        prompt=prompt,
        max_tokens=max_tokens
    )

    await batcher.add_request(request)

    # Wait for completion
    while not request.finished:
        await asyncio.sleep(0.01)

    return model.detokenize(request.result)
```

### Dynamic Batching with vLLM-style PagedAttention

```python
# server/paged_attention_batch.py
from typing import List, Dict
import numpy as np

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = set(range(num_blocks))
        self.allocations: Dict[str, List[int]] = {}

    def allocate(self, request_id: str, num_tokens: int) -> List[int]:
        """Allocate blocks for request"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Out of KV cache memory")

        blocks = []
        for _ in range(num_blocks_needed):
            block = self.free_blocks.pop()
            blocks.append(block)

        self.allocations[request_id] = blocks
        return blocks

    def free(self, request_id: str):
        """Free blocks for request"""
        if request_id in self.allocations:
            blocks = self.allocations.pop(request_id)
            self.free_blocks.update(blocks)

    def usage_percentage(self) -> float:
        """Get cache utilization"""
        used = self.num_blocks - len(self.free_blocks)
        return (used / self.num_blocks) * 100

class PagedAttentionBatcher:
    def __init__(self, model, block_size: int = 16, num_blocks: int = 1024):
        self.model = model
        self.kv_cache = KVCacheManager(num_blocks, block_size)
        self.active_requests: Dict[str, Request] = {}

    def add_request(self, request: Request):
        """Add new request"""
        # Allocate KV cache
        prompt_tokens = len(self.model.tokenize(request.prompt.encode()))
        blocks = self.kv_cache.allocate(request.request_id, prompt_tokens)

        request.kv_blocks = blocks
        self.active_requests[request.request_id] = request

    def process_batch(self):
        """Process all active requests in one batch"""
        if not self.active_requests:
            return

        # All requests can be batched together thanks to paged attention
        requests = list(self.active_requests.values())

        # Generate tokens
        for request in requests:
            # Each request's KV cache is in different blocks
            # Model can efficiently handle them together
            token = self.model.generate_with_blocks(
                request.prompt,
                request.kv_blocks
            )

            request.result.append(token)
            request.generated_tokens += 1

            if request.generated_tokens >= request.max_tokens:
                request.finished = True
                self.kv_cache.free(request.request_id)
                del self.active_requests[request.request_id]
```

## Load Balancing

### NGINX Load Balancer Configuration

```nginx
# nginx.conf
upstream llama_servers {
    least_conn;  # Route to server with least connections

    server llama-server-1:8080 max_fails=3 fail_timeout=30s;
    server llama-server-2:8080 max_fails=3 fail_timeout=30s;
    server llama-server-3:8080 max_fails=3 fail_timeout=30s;

    # Health check
    server llama-server-4:8080 max_fails=3 fail_timeout=30s backup;

    keepalive 32;
}

server {
    listen 80;
    server_name api.inference.example.com;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    location /v1/ {
        proxy_pass http://llama_servers;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Buffering
        proxy_buffering off;  # For streaming responses
        proxy_request_buffering off;

        # Retry logic
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 2;
    }

    location /health {
        access_log off;
        proxy_pass http://llama_servers/health;
    }
}
```

### HAProxy Configuration

```haproxy
# haproxy.cfg
global
    maxconn 10000
    log stdout format raw local0

defaults
    mode http
    timeout connect 5s
    timeout client 120s
    timeout server 120s
    option httplog
    option http-server-close

frontend llama_frontend
    bind *:80

    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny if { sc_http_req_rate(0) gt 100 }

    default_backend llama_backend

backend llama_backend
    balance leastconn

    option httpchk GET /health
    http-check expect status 200

    server llama1 llama-server-1:8080 check inter 5s rise 2 fall 3 maxconn 100
    server llama2 llama-server-2:8080 check inter 5s rise 2 fall 3 maxconn 100
    server llama3 llama-server-3:8080 check inter 5s rise 2 fall 3 maxconn 100

    # Connection pooling
    http-reuse safe
```

## Auto-Scaling

### Kubernetes Horizontal Pod Autoscaler

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  # Scale on CPU
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # Scale on memory
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  # Scale on GPU
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "75"

  # Scale on queue depth
  - type: Pods
    pods:
      metric:
        name: request_queue_depth
      target:
        type: AverageValue
        averageValue: "10"

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minutes
      policies:
      - type: Percent
        value: 50  # Scale down max 50% of current replicas
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100  # Double capacity quickly
        periodSeconds: 30
      - type: Pods
        value: 4  # Add max 4 pods at a time
        periodSeconds: 30
      selectPolicy: Max
```

### Custom Metrics for Autoscaling

```python
# monitoring/custom_metrics.py
from prometheus_client import Gauge, Counter
import time

# Custom metrics for autoscaling
request_queue_depth = Gauge('request_queue_depth', 'Number of requests waiting')
gpu_utilization = Gauge('gpu_utilization', 'GPU utilization percentage')
active_requests = Gauge('active_requests', 'Number of requests being processed')
avg_latency = Gauge('avg_latency_ms', 'Average request latency in ms')

class AutoscalingMetrics:
    def __init__(self):
        self.queue_size = 0
        self.processing = 0
        self.recent_latencies = []

    def update_queue(self, size: int):
        """Update queue depth"""
        self.queue_size = size
        request_queue_depth.set(size)

    def request_started(self):
        """Track request start"""
        self.processing += 1
        active_requests.set(self.processing)

    def request_finished(self, latency_ms: float):
        """Track request completion"""
        self.processing -= 1
        active_requests.set(self.processing)

        # Track latency
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > 100:
            self.recent_latencies.pop(0)

        avg_latency.set(sum(self.recent_latencies) / len(self.recent_latencies))

    def update_gpu_utilization(self, utilization: float):
        """Update GPU utilization"""
        gpu_utilization.set(utilization)
```

## Caching Strategies

### Response Caching

```python
# caching/response_cache.py
import redis
import hashlib
import json
from typing import Optional

class ResponseCache:
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl

    def _make_key(self, prompt: str, params: dict) -> str:
        """Create cache key from prompt and parameters"""
        # Create deterministic hash
        content = f"{prompt}:{json.dumps(params, sort_keys=True)}"
        return f"cache:response:{hashlib.sha256(content.encode()).hexdigest()}"

    def get(self, prompt: str, params: dict) -> Optional[str]:
        """Get cached response"""
        key = self._make_key(prompt, params)
        cached = self.redis.get(key)

        if cached:
            return cached.decode('utf-8')
        return None

    def set(self, prompt: str, params: dict, response: str):
        """Cache response"""
        key = self._make_key(prompt, params)
        self.redis.setex(key, self.ttl, response)

    def invalidate(self, prompt: str, params: dict):
        """Invalidate cached response"""
        key = self._make_key(prompt, params)
        self.redis.delete(key)

# Usage
cache = ResponseCache(redis.Redis())

async def generate_with_cache(prompt: str, **params):
    # Check cache
    cached = cache.get(prompt, params)
    if cached:
        return cached

    # Generate
    response = await model.generate(prompt, **params)

    # Cache if deterministic
    if params.get('temperature', 1.0) == 0.0:
        cache.set(prompt, params, response)

    return response
```

### KV Cache Sharing

```python
# caching/kv_cache_sharing.py
from typing import Dict, List, Tuple

class KVCacheSharer:
    def __init__(self):
        self.prefix_cache: Dict[str, Tuple[int, List]] = {}

    def find_shared_prefix(self, prompt: str) -> Tuple[str, int]:
        """Find longest matching prefix in cache"""
        best_match = ""
        best_length = 0

        for cached_prompt, (length, kv_cache) in self.prefix_cache.items():
            if prompt.startswith(cached_prompt):
                if length > best_length:
                    best_match = cached_prompt
                    best_length = length

        return best_match, best_length

    def cache_prompt(self, prompt: str, kv_cache: List):
        """Cache KV for prompt"""
        # Only cache common prefixes
        if len(prompt) > 50:  # Minimum length
            self.prefix_cache[prompt] = (len(prompt), kv_cache)

    def process_with_sharing(self, prompt: str):
        """Process prompt with KV cache sharing"""
        # Find shared prefix
        prefix, length = self.find_shared_prefix(prompt)

        if length > 0:
            # Reuse cached KV
            print(f"Reusing {length} tokens from cache")
            kv_cache = self.prefix_cache[prefix][1]
            remaining_prompt = prompt[length:]

            # Process only remaining tokens
            output = model.generate(remaining_prompt, kv_cache=kv_cache)
        else:
            # Process full prompt
            output = model.generate(prompt)

            # Cache for future use
            self.cache_prompt(prompt, model.get_kv_cache())

        return output
```

## Database Optimization for Embeddings

### Vector Database with FAISS

```python
# storage/vector_db.py
import faiss
import numpy as np
from typing import List, Tuple

class VectorDatabase:
    def __init__(self, dimension: int, index_type: str = "IVF"):
        self.dimension = dimension

        if index_type == "IVF":
            # Inverted file index for faster search
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif index_type == "HNSW":
            # Hierarchical NSW for even faster search
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            # Flat index (exact search)
            self.index = faiss.IndexFlatL2(dimension)

        self.metadata = []

    def train(self, vectors: np.ndarray):
        """Train index (for IVF)"""
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.train(vectors)

    def add(self, vectors: np.ndarray, metadata: List[dict]):
        """Add vectors to index"""
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[float, dict]]:
        """Search for nearest neighbors"""
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append((float(dist), self.metadata[idx]))

        return results

    def save(self, filepath: str):
        """Save index to disk"""
        faiss.write_index(self.index, filepath)

    def load(self, filepath: str):
        """Load index from disk"""
        self.index = faiss.read_index(filepath)

# Usage
db = VectorDatabase(dimension=4096, index_type="HNSW")

# Add embeddings
embeddings = model.embed(documents)
db.add(embeddings, metadata=[{"doc_id": i, "text": doc} for i, doc in enumerate(documents)])

# Search
query_embedding = model.embed(["What is llama.cpp?"])
results = db.search(query_embedding, k=5)
```

## Performance Optimization Checklist

### Infrastructure
- [ ] Use GPU acceleration when available
- [ ] Enable tensor cores (FP16/BF16)
- [ ] Configure optimal batch sizes
- [ ] Implement request batching
- [ ] Use load balancing
- [ ] Set up auto-scaling

### Model Optimization
- [ ] Choose appropriate quantization (Q4_K_M recommended)
- [ ] Enable flash attention if available
- [ ] Use KV cache efficiently
- [ ] Implement speculative decoding for long generations
- [ ] Share KV cache for common prefixes

### Application Layer
- [ ] Implement response caching
- [ ] Use connection pooling
- [ ] Enable HTTP/2 or gRPC
- [ ] Implement rate limiting
- [ ] Add request timeouts
- [ ] Use async/await patterns

### Monitoring
- [ ] Track latency (P50, P95, P99)
- [ ] Monitor throughput
- [ ] Watch resource utilization
- [ ] Set up alerting
- [ ] Create dashboards

## Summary

Key strategies for performance at scale:
- **Batching**: Continuous batching for better GPU utilization
- **Caching**: Response and KV cache sharing
- **Load Balancing**: Distribute traffic efficiently
- **Auto-Scaling**: Scale based on demand
- **Optimization**: Quantization, flash attention, efficient inference

---

**Authors**: Agent 5 (Documentation Specialist)
**Last Updated**: 2025-11-18
**Estimated Reading Time**: 40 minutes

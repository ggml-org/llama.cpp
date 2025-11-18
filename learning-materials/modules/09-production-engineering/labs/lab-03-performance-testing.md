# Lab 3: Performance Testing

## Objectives

- ✅ Benchmark inference performance
- ✅ Conduct load testing
- ✅ Identify bottlenecks
- ✅ Optimize for production
- ✅ Set up continuous performance monitoring

**Estimated Time**: 2-3 hours

## Part 1: Baseline Benchmarks

### Task 1.1: Single Request Latency

```python
import time
from llama_cpp import Llama

model = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf")

# Measure Time to First Token (TTFT)
prompt = "Explain quantum computing:"

start = time.time()
first_token = None
tokens = 0

for chunk in model(prompt, max_tokens=100, stream=True):
    if first_token is None:
        first_token = time.time()
    tokens += 1

end = time.time()

print(f"TTFT: {first_token - start:.3f}s")
print(f"Total: {end - start:.3f}s")
print(f"Tokens: {tokens}")
print(f"TPS: {tokens / (end - start):.2f}")
```

**✏️ Task**: Run with different prompt lengths and measure variance.

### Task 1.2: Throughput Testing

```python
import asyncio
import aiohttp

async def measure_throughput(url, num_requests=100):
    async def make_request(session, i):
        start = time.time()
        async with session.post(
            f"{url}/completion",
            json={"prompt": f"Test {i}", "max_tokens": 10}
        ) as resp:
            await resp.json()
            return time.time() - start

    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, i) for i in range(num_requests)]
        latencies = await asyncio.gather(*tasks)

    total_time = max(latencies)
    throughput = num_requests / total_time

    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Mean latency: {sum(latencies)/len(latencies):.3f}s")

asyncio.run(measure_throughput("http://localhost:8080", num_requests=100))
```

## Part 2: Load Testing

### Task 2.1: Gradual Load Increase

```python
import time
import statistics

def load_test_gradual(url, max_concurrent=50, step=5, duration_per_step=60):
    results = {}

    for concurrent in range(step, max_concurrent + 1, step):
        print(f"\nTesting with {concurrent} concurrent requests...")

        latencies = []
        errors = 0

        start = time.time()
        while time.time() - start < duration_per_step:
            # Make concurrent requests
            # ... implementation ...

        results[concurrent] = {
            'mean_latency': statistics.mean(latencies),
            'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)],
            'error_rate': errors / len(latencies) if latencies else 1.0,
            'throughput': len(latencies) / duration_per_step
        }

        print(f"  Mean: {results[concurrent]['mean_latency']:.3f}s")
        print(f"  P95: {results[concurrent]['p95_latency']:.3f}s")
        print(f"  Errors: {results[concurrent]['error_rate']:.1%}")

    return results
```

**✏️ Task**: Find the maximum sustainable load.

### Task 2.2: Stress Testing

Use `locust` for distributed load testing:

```python
# locustfile.py
from locust import HttpUser, task, between

class LLaMAUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def completion(self):
        self.client.post("/completion", json={
            "prompt": "Hello world",
            "max_tokens": 20
        })

    @task(3)
    def chat(self):
        self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 20
        })
```

Run:
```bash
locust -f locustfile.py --host=http://localhost:8080
# Open http://localhost:8089 for UI
```

## Part 3: Profiling

### Task 3.1: CPU Profiling

```bash
# Profile with perf
perf record -g ./llama-server -m model.gguf
perf report

# Or with valgrind
valgrind --tool=callgrind ./llama-server -m model.gguf
kcachegrind callgrind.out.*
```

### Task 3.2: GPU Profiling

```bash
# NVIDIA Nsight
nsys profile --stats=true ./llama-server -m model.gguf

# CUDA profiler
nvprof --print-gpu-trace ./llama-server -m model.gguf
```

**✏️ Task**: Identify top 3 hotspots.

## Part 4: Optimization

### Task 4.1: Batch Processing

Implement request batching:

```python
class BatchProcessor:
    def __init__(self, max_batch_size=8, max_wait_ms=100):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []

    async def add_request(self, prompt):
        # Add to queue
        self.queue.append(prompt)

        # Process when batch is full or timeout
        if len(self.queue) >= self.max_batch_size:
            return await self.process_batch()

    async def process_batch(self):
        # Process all queued requests together
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]

        # Batch inference
        results = model.generate_batch(batch)
        return results
```

**✏️ Task**: Measure throughput improvement.

### Task 4.2: Caching

```python
from functools import lru_cache
import hashlib

class ResponseCache:
    def __init__(self):
        self.cache = {}

    def get(self, prompt, params):
        key = self._make_key(prompt, params)
        return self.cache.get(key)

    def set(self, prompt, params, response):
        key = self._make_key(prompt, params)
        self.cache[key] = response

    def _make_key(self, prompt, params):
        content = f"{prompt}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()

# Usage with deterministic generation
if temperature == 0.0:
    cached = cache.get(prompt, params)
    if cached:
        return cached
```

## Part 5: Continuous Monitoring

### Task 5.1: Prometheus Metrics

```python
from prometheus_client import Histogram, Counter

request_latency = Histogram(
    'request_latency_seconds',
    'Request latency'
)

@request_latency.time()
def process_request(prompt):
    return model(prompt)
```

### Task 5.2: Performance Dashboard

Create Grafana dashboard with:
- Request latency (P50, P95, P99)
- Throughput (req/s, tokens/s)
- Error rate
- GPU utilization
- Memory usage

## Verification

Run complete performance test suite:

```bash
python performance_tests.py --baseline baseline.json --output results.json
```

Expected output:
```
Performance Test Results:
✅ TTFT: 0.145s (target: <0.2s)
✅ Throughput: 125 req/s (target: >100 req/s)
✅ P95 latency: 0.8s (target: <1.0s)
⚠️  Max concurrent: 45 (target: >50)
```

## Deliverables

- ✅ Performance baseline report
- ✅ Load testing results
- ✅ Bottleneck analysis
- ✅ Optimization recommendations
- ✅ Continuous monitoring setup

## Challenge Tasks

1. Implement speculative decoding
2. Add KV cache optimization
3. Test multi-GPU scaling
4. Benchmark different quantizations
5. Create auto-scaling based on metrics

---

**Next**: Lab 4: Security Audit

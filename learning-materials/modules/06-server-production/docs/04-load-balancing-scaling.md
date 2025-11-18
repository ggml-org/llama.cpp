# Load Balancing & Scaling for LLM Services

**Learning Module**: Module 6 - Server & Production
**Estimated Reading Time**: 30 minutes
**Prerequisites**: Understanding of distributed systems, Module 6.1-6.3
**Related Content**:
- [Deployment Patterns](./03-deployment-patterns.md)
- [Monitoring & Observability](./05-monitoring-observability.md)
- [Production Best Practices](./06-production-best-practices.md)

---

## Overview

Scaling LLM inference services presents unique challenges due to:
- **Stateful nature**: Long-running requests with KV cache
- **Variable latency**: Completion time depends on output length
- **Resource intensity**: High memory and compute requirements
- **GPU constraints**: Limited GPU sharing capabilities

This guide covers strategies for horizontal scaling and load balancing.

---

## Load Balancing Strategies

### 1. Round Robin

**Simplest approach** - Distribute requests evenly across servers.

**Nginx Configuration**:
```nginx
upstream llama_backend {
    # Simple round-robin
    server 10.0.1.10:8080;
    server 10.0.1.11:8080;
    server 10.0.1.12:8080;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://llama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Important for streaming
        proxy_buffering off;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        # Long timeout for LLM generation
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

**Pros**:
- Simple to implement
- Predictable distribution
- No state required

**Cons**:
- Ignores server load
- Doesn't account for request complexity
- May overload slow servers

### 2. Least Connections

**Better for LLM** - Route to server with fewest active connections.

**Nginx Configuration**:
```nginx
upstream llama_backend {
    least_conn;

    server 10.0.1.10:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8080 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8080 max_fails=3 fail_timeout=30s;
}
```

**Pros**:
- Accounts for long-running requests
- Better load distribution
- Handles variable request duration

**Cons**:
- Doesn't consider actual server load
- Connections ≠ resource usage

### 3. Weighted Load Balancing

**For heterogeneous hardware** - Assign more load to powerful servers.

**Nginx Configuration**:
```nginx
upstream llama_backend {
    least_conn;

    # High-end GPU server (A100)
    server 10.0.1.10:8080 weight=5 max_fails=3;

    # Mid-range GPU servers (T4)
    server 10.0.1.11:8080 weight=2 max_fails=3;
    server 10.0.1.12:8080 weight=2 max_fails=3;

    # CPU-only fallback
    server 10.0.1.13:8080 weight=1 backup;
}
```

**Weight Calculation**:
```
Weight = (GPU TFLOPS / Baseline TFLOPS) * (VRAM GB / Baseline VRAM)

Example:
A100:    (312 / 65) * (40 / 16) = 12
T4:      (65 / 65)  * (16 / 16) = 1
```

### 4. Slot-Based Routing

**Custom solution** - Route based on available inference slots.

**Python Implementation**:
```python
import httpx
import asyncio
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Backend:
    url: str
    total_slots: int
    available_slots: int = 0
    latency_ms: float = 0
    last_check: float = 0

class SlotBasedLoadBalancer:
    def __init__(self, backends: List[str]):
        self.backends = [
            Backend(url=url, total_slots=0)
            for url in backends
        ]

    async def update_backend_status(self, backend: Backend):
        """Query backend for available slots"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{backend.url}/health",
                    timeout=5.0
                )
                data = response.json()

                backend.total_slots = data.get("slots_total", 0)
                backend.available_slots = data.get("slots_available", 0)
                backend.last_check = asyncio.get_event_loop().time()

        except Exception as e:
            backend.available_slots = 0

    async def get_best_backend(self) -> Backend:
        """Select backend with most available slots"""
        # Update all backends
        await asyncio.gather(*[
            self.update_backend_status(b) for b in self.backends
        ])

        # Filter healthy backends
        healthy = [b for b in self.backends if b.available_slots > 0]

        if not healthy:
            raise Exception("No available backends")

        # Return backend with most slots
        return max(healthy, key=lambda b: b.available_slots)

    async def route_request(self, request_data: dict):
        """Route request to best backend"""
        backend = await self.get_best_backend()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{backend.url}/v1/chat/completions",
                json=request_data,
                timeout=300.0
            )
            return response.json()

# Usage
lb = SlotBasedLoadBalancer([
    "http://server1:8080",
    "http://server2:8080",
    "http://server3:8080"
])

response = await lb.route_request({
    "model": "llama-2-7b",
    "messages": [{"role": "user", "content": "Hello"}]
})
```

### 5. Model-Based Routing

**Route by model type** - Different models on different servers.

**Nginx Configuration**:
```nginx
map $request_uri $backend_pool {
    ~*/chat/.*model=llama-2-7b     7b_pool;
    ~*/chat/.*model=llama-2-13b    13b_pool;
    ~*/chat/.*model=codellama      code_pool;
    default                        7b_pool;
}

upstream 7b_pool {
    server gpu-server-1:8080;
    server gpu-server-2:8080;
}

upstream 13b_pool {
    server gpu-server-3:8080;
    server gpu-server-4:8080;
}

upstream code_pool {
    server gpu-server-5:8080;
}

server {
    location / {
        proxy_pass http://$backend_pool;
    }
}
```

---

## Horizontal Scaling Patterns

### 1. Manual Scaling

**Simple approach** - Add/remove instances manually.

**Kubernetes Example**:
```bash
# Scale to 5 replicas
kubectl scale deployment llama-server --replicas=5

# Scale down to 2
kubectl scale deployment llama-server --replicas=2

# Check status
kubectl get pods -l app=llama-server
```

**When to use**:
- Predictable traffic patterns
- Small scale deployments
- Cost-sensitive environments

### 2. Auto-Scaling Based on Metrics

**Kubernetes HPA (Horizontal Pod Autoscaler)**:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-server-hpa
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  # Scale based on CPU
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # Scale based on memory
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  # Scale based on custom metric (requests per second)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max

    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min
```

**Custom Metrics with Prometheus**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-metrics
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
spec:
  selector:
    app: llama-server
  ports:
  - port: 8080
    name: metrics
```

### 3. Queue-Based Scaling

**For batch processing** - Scale based on queue depth.

**Architecture**:
```
┌─────────┐      ┌──────────┐      ┌────────────────┐
│ Clients │─────▶│  Queue   │◀─────│  Workers       │
│         │      │ (Redis/  │      │ (llama-server) │
└─────────┘      │  RabbitMQ)│      └────────────────┘
                 └──────────┘              ▲
                      │                    │
                      │                    │
                      ▼                    │
                 ┌──────────┐              │
                 │ Auto-    │──────────────┘
                 │ Scaler   │ (Scale based on queue depth)
                 └──────────┘
```

**Implementation**:
```python
import asyncio
import redis
import httpx
from typing import List

class QueueWorker:
    def __init__(self, redis_url: str, llama_url: str):
        self.redis = redis.from_url(redis_url)
        self.llama_url = llama_url

    async def process_queue(self):
        """Process requests from queue"""
        while True:
            # Pop request from queue
            request_data = self.redis.blpop("inference_queue", timeout=1)

            if request_data:
                _, request_json = request_data
                request = json.loads(request_json)

                # Process with llama-server
                result = await self.process_request(request)

                # Store result
                self.redis.set(
                    f"result:{request['id']}",
                    json.dumps(result),
                    ex=3600  # Expire after 1 hour
                )

    async def process_request(self, request: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.llama_url}/v1/chat/completions",
                json=request["data"],
                timeout=300.0
            )
            return response.json()

# Auto-scaler based on queue depth
class QueueBasedAutoscaler:
    def __init__(self, redis_url: str, min_workers: int, max_workers: int):
        self.redis = redis.from_url(redis_url)
        self.min_workers = min_workers
        self.max_workers = max_workers

    def get_desired_replicas(self) -> int:
        """Calculate desired replicas based on queue depth"""
        queue_depth = self.redis.llen("inference_queue")

        # Scale: 1 worker per 10 queued items
        desired = max(
            self.min_workers,
            min(queue_depth // 10, self.max_workers)
        )

        return desired

    async def scale_loop(self):
        """Continuously adjust replica count"""
        while True:
            desired = self.get_desired_replicas()

            # Update Kubernetes deployment
            subprocess.run([
                "kubectl", "scale", "deployment", "llama-server",
                f"--replicas={desired}"
            ])

            await asyncio.sleep(30)
```

**Kubernetes Job-based approach**:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llama-inference-job
spec:
  parallelism: 5  # Run 5 workers in parallel
  completions: 100  # Process 100 items total
  template:
    spec:
      containers:
      - name: worker
        image: llama-worker:latest
        command: ["python", "worker.py"]
        env:
        - name: REDIS_URL
          value: "redis://redis:6379"
```

### 4. Predictive Scaling

**Machine learning-based** - Predict traffic and pre-scale.

**Concept**:
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime

class PredictiveScaler:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.training_data = []

    def train(self, historical_data: pd.DataFrame):
        """Train on historical traffic patterns"""
        # Features: hour, day_of_week, month, previous_hour_traffic
        X = historical_data[['hour', 'day_of_week', 'month', 'prev_hour_rps']]
        y = historical_data['requests_per_second']

        self.model.fit(X, y)

    def predict_next_hour_traffic(self) -> int:
        """Predict traffic for next hour"""
        now = datetime.datetime.now()

        features = [[
            now.hour,
            now.weekday(),
            now.month,
            self.get_current_rps()
        ]]

        predicted_rps = self.model.predict(features)[0]

        # Convert to replica count
        replicas = max(2, int(predicted_rps / 10))  # 1 replica per 10 RPS

        return replicas

    def get_current_rps(self) -> float:
        # Query Prometheus or metrics system
        pass
```

---

## Vertical Scaling Considerations

### When to Scale Vertically

**Scale Up** (Increase resources per instance):
- ✅ Running larger models (13B → 70B)
- ✅ Increasing context length
- ✅ Enabling more parallel slots
- ✅ GPU memory constraints

**Resource Calculation**:
```python
def calculate_required_resources(
    model_size_b: int,
    quantization: str,
    context_length: int,
    parallel_slots: int
) -> dict:
    """Calculate required CPU, RAM, VRAM"""

    # Bytes per parameter for quantization
    bytes_per_param = {
        "Q4_K_M": 0.5,
        "Q5_K_M": 0.625,
        "Q8_0": 1.0,
        "F16": 2.0
    }

    bpp = bytes_per_param.get(quantization, 0.5)

    # Model size
    model_size_gb = (model_size_b * 1e9 * bpp) / (1024**3)

    # KV cache per slot
    kv_cache_gb = (context_length * 2 * 0.002) * parallel_slots

    # Total memory
    total_memory_gb = model_size_gb + kv_cache_gb + 2  # +2GB overhead

    # GPU selection
    if total_memory_gb <= 24:
        gpu = "RTX 4090 / A10"
    elif total_memory_gb <= 40:
        gpu = "A100-40GB"
    elif total_memory_gb <= 80:
        gpu = "A100-80GB"
    else:
        gpu = "Multiple GPUs required"

    return {
        "model_size_gb": model_size_gb,
        "kv_cache_gb": kv_cache_gb,
        "total_memory_gb": total_memory_gb,
        "recommended_gpu": gpu,
        "cpu_cores": max(4, parallel_slots),
        "ram_gb": total_memory_gb * 1.5  # 50% overhead
    }

# Example
resources = calculate_required_resources(
    model_size_b=13,
    quantization="Q4_K_M",
    context_length=4096,
    parallel_slots=8
)
print(resources)
# {
#   "model_size_gb": 6.5,
#   "kv_cache_gb": 0.13,
#   "total_memory_gb": 8.63,
#   "recommended_gpu": "RTX 4090 / A10",
#   "cpu_cores": 8,
#   "ram_gb": 12.95
# }
```

---

## Traffic Management

### 1. Request Prioritization

**Implement priority queues**:
```python
from enum import Enum
import heapq
import asyncio

class Priority(Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3

class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.counter = 0

    async def enqueue(self, request: dict, priority: Priority):
        # Lower number = higher priority
        heapq.heappush(
            self.queue,
            (priority.value, self.counter, request)
        )
        self.counter += 1

    async def dequeue(self):
        if not self.queue:
            return None

        _, _, request = heapq.heappop(self.queue)
        return request

# Usage
queue = PriorityQueue()

# High priority (paid users)
await queue.enqueue(paid_user_request, Priority.HIGH)

# Normal priority
await queue.enqueue(normal_request, Priority.NORMAL)

# Low priority (free tier)
await queue.enqueue(free_tier_request, Priority.LOW)
```

### 2. Circuit Breaker Pattern

**Prevent cascade failures**:
```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

            raise e

# Usage
breaker = CircuitBreaker()

try:
    response = breaker.call(
        httpx.post,
        "http://llama-server:8080/v1/chat/completions",
        json=request_data
    )
except Exception as e:
    # Circuit is open, use fallback
    response = fallback_response()
```

### 3. Rate Limiting

**Token bucket implementation** (from Module 6.2):
```python
class DistributedRateLimiter:
    """Redis-based rate limiter for distributed systems"""

    def __init__(self, redis_client, requests_per_minute: int):
        self.redis = redis_client
        self.rpm = requests_per_minute

    async def allow_request(self, user_id: str) -> bool:
        key = f"ratelimit:{user_id}"
        now = time.time()

        # Sliding window
        window_start = now - 60

        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        count = await self.redis.zcard(key)

        if count < self.rpm:
            # Add this request
            await self.redis.zadd(key, {str(now): now})
            await self.redis.expire(key, 60)
            return True

        return False
```

---

## Handling Failures

### 1. Health Checks

**Multi-level health checks**:
```python
from fastapi import FastAPI, Response
import httpx

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic liveness check"""
    return {"status": "ok"}

@app.get("/health/ready")
async def readiness_check():
    """Readiness check - can serve traffic"""
    try:
        # Check backend
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8080/health",
                timeout=5.0
            )

        if response.status_code == 200:
            data = response.json()
            if data.get("slots_available", 0) > 0:
                return {"status": "ready", "slots": data["slots_available"]}

        return Response(status_code=503, content='{"status": "not_ready"}')

    except Exception as e:
        return Response(status_code=503, content=f'{{"error": "{str(e)}"}}')

@app.get("/health/startup")
async def startup_check():
    """Startup check - model loaded"""
    # Check if model is loaded
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8080/v1/models",
                timeout=10.0
            )

        if response.status_code == 200:
            return {"status": "started"}

        return Response(status_code=503, content='{"status": "starting"}')

    except Exception:
        return Response(status_code=503, content='{"status": "starting"}')
```

### 2. Graceful Shutdown

**Handle shutdown gracefully**:
```python
import signal
import asyncio

class GracefulShutdown:
    def __init__(self):
        self.is_shutting_down = False
        self.active_requests = 0

    async def shutdown_handler(self):
        """Handle shutdown signal"""
        self.is_shutting_down = True
        print("Received shutdown signal, waiting for requests to complete...")

        # Wait for active requests to complete (max 60s)
        for _ in range(60):
            if self.active_requests == 0:
                print("All requests completed, shutting down")
                break
            await asyncio.sleep(1)
        else:
            print(f"Shutdown timeout, {self.active_requests} requests remaining")

    def setup_signals(self):
        signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(self.shutdown_handler()))
        signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(self.shutdown_handler()))

shutdown_manager = GracefulShutdown()

@app.middleware("http")
async def track_requests(request, call_next):
    if shutdown_manager.is_shutting_down:
        return Response(status_code=503, content="Server is shutting down")

    shutdown_manager.active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        shutdown_manager.active_requests -= 1
```

---

## Summary

**Key Strategies**:
1. **Load Balancing**: Use least-connections or slot-based routing
2. **Horizontal Scaling**: Auto-scale based on metrics (CPU, queue depth, custom)
3. **Vertical Scaling**: Calculate resource requirements accurately
4. **Traffic Management**: Implement priority queues and circuit breakers
5. **Failure Handling**: Multi-level health checks and graceful shutdown

**Best Practices**:
- ✅ Monitor slot availability, not just CPU/memory
- ✅ Use weighted routing for heterogeneous hardware
- ✅ Implement gradual scaling (don't scale too aggressively)
- ✅ Set appropriate stabilization windows
- ✅ Test failure scenarios regularly

**Next Steps**:
- [Monitoring & Observability](./05-monitoring-observability.md)
- Lab 6.4: Implement auto-scaling

---

**Interview Topics**:
- Load balancing algorithms
- Auto-scaling strategies
- Circuit breaker pattern
- Graceful shutdown implementation
- Capacity planning for LLM services

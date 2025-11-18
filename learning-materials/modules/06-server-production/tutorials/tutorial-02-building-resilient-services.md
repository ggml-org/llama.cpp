# Tutorial: Building Resilient LLM Services

**Duration**: 60 minutes
**Level**: Advanced

---

## Overview

Learn to build resilient LLM services that handle failures gracefully, recover automatically, and maintain high availability.

---

## Part 1: Circuit Breaker Pattern (15 min)

### Implementation

```python
import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
breaker = CircuitBreaker()

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    try:
        response = await breaker.call(call_llama_server, request)
        return response
    except CircuitBreakerError:
        # Use fallback or cached response
        return get_cached_response(request)
```

---

## Part 2: Retry with Exponential Backoff (10 min)

```python
import asyncio
import random
from typing import Callable, Tuple, Type

async def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            if attempt == max_retries:
                raise

            delay = min(
                base_delay * (exponential_base ** attempt),
                max_delay
            )

            if jitter:
                delay *= (0.5 + random.random() * 0.5)

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s"
            )

            await asyncio.sleep(delay)

# Usage
async def call_with_retry(request):
    return await retry_with_backoff(
        call_llama_server,
        request,
        max_retries=3,
        retryable_exceptions=(httpx.HTTPError, TimeoutError)
    )
```

---

## Part 3: Graceful Degradation (15 min)

```python
from enum import Enum

class ServiceMode(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"

class ServiceController:
    def __init__(self):
        self.mode = ServiceMode.NORMAL
        self.health_checks = []

    async def determine_mode(self):
        # Check backend health
        backend_healthy = await self.check_backend()
        queue_depth = await self.get_queue_depth()
        error_rate = await self.get_error_rate()

        if not backend_healthy or error_rate > 0.1:
            self.mode = ServiceMode.EMERGENCY
        elif queue_depth > 100 or error_rate > 0.05:
            self.mode = ServiceMode.DEGRADED
        else:
            self.mode = ServiceMode.NORMAL

    async def handle_request(self, request: ChatRequest):
        await self.determine_mode()

        if self.mode == ServiceMode.EMERGENCY:
            return await self.emergency_response(request)
        elif self.mode == ServiceMode.DEGRADED:
            return await self.degraded_response(request)
        else:
            return await self.normal_response(request)

    async def degraded_response(self, request):
        # Reduce quality for speed
        request.max_tokens = min(request.max_tokens, 256)
        request.temperature = 0.0  # Greedy decoding
        return await call_llama_server(request)

    async def emergency_response(self, request):
        # Use cache or simple response
        cached = await get_from_cache(request)
        if cached:
            return cached

        return {
            "status": "degraded",
            "message": "Service temporarily degraded, please try again"
        }
```

---

## Part 4: Health Checks & Readiness Probes (10 min)

```python
from fastapi import Response
import httpx

@app.get("/health")
async def health_check():
    """Liveness probe - is the service alive?"""
    return {"status": "ok"}

@app.get("/health/ready")
async def readiness_check():
    """Readiness probe - can the service handle traffic?"""
    checks = {
        "backend": await check_backend_health(),
        "database": await check_database(),
        "cache": await check_cache()
    }

    all_healthy = all(checks.values())

    if all_healthy:
        return {"status": "ready", "checks": checks}
    else:
        return Response(
            status_code=503,
            content={"status": "not_ready", "checks": checks}
        )

async def check_backend_health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://llama-server:8080/health",
                timeout=5.0
            )
            data = response.json()
            return data.get("slots_available", 0) > 0
    except:
        return False
```

---

## Part 5: Bulkhead Pattern (10 min)

```python
import asyncio
from asyncio import Semaphore

class BulkheadController:
    def __init__(self):
        self.semaphores = {
            "critical": Semaphore(10),   # 10 concurrent critical requests
            "normal": Semaphore(20),     # 20 concurrent normal requests
            "batch": Semaphore(5)        # 5 concurrent batch requests
        }

    async def execute(self, priority: str, func, *args, **kwargs):
        semaphore = self.semaphores.get(priority, self.semaphores["normal"])

        async with semaphore:
            return await func(*args, **kwargs)

bulkhead = BulkheadController()

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, priority: str = "normal"):
    return await bulkhead.execute(
        priority,
        call_llama_server,
        request
    )
```

---

## Part 6: Timeout Management (10 min)

```python
import asyncio

class TimeoutManager:
    def __init__(self):
        self.default_timeout = 30.0
        self.max_timeout = 300.0

    async def execute_with_timeout(
        self,
        func,
        *args,
        timeout: float = None,
        **kwargs
    ):
        timeout = timeout or self.default_timeout
        timeout = min(timeout, self.max_timeout)

        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout}s")
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {timeout}s"
            )

timeout_manager = TimeoutManager()

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    # Longer timeout for larger requests
    timeout = calculate_timeout(request.max_tokens)

    return await timeout_manager.execute_with_timeout(
        call_llama_server,
        request,
        timeout=timeout
    )

def calculate_timeout(max_tokens: int) -> float:
    # Base: 10s + 0.1s per token
    return min(10 + (max_tokens * 0.1), 300)
```

---

## Testing Resilience

### Chaos Engineering Tests

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_backend_failure():
    \"\"\"Test behavior when backend fails\"\"\"
    # Simulate backend failure
    with mock.patch('httpx.AsyncClient.post') as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        response = await client.post("/v1/chat/completions", json={...})

        # Should fail gracefully
        assert response.status_code in [503, 504]
        assert "error" in response.json()

@pytest.mark.asyncio
async def test_slow_backend():
    \"\"\"Test timeout handling\"\"\"
    with mock.patch('httpx.AsyncClient.post') as mock_post:
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(60)
            return mock_response

        mock_post.side_effect = slow_response

        start = time.time()
        response = await client.post("/v1/chat/completions", json={...})
        duration = time.time() - start

        # Should timeout within reasonable time
        assert duration < 35
        assert response.status_code == 504

@pytest.mark.asyncio
async def test_circuit_breaker():
    \"\"\"Test circuit breaker opens after failures\"\"\"
    # Make 5 failing requests
    for _ in range(5):
        await client.post("/v1/chat/completions", json={...})

    # Circuit should be open now
    response = await client.post("/v1/chat/completions", json={...})
    assert "circuit breaker" in response.json()["error"].lower()
```

---

## Summary: Resilience Patterns

| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| Circuit Breaker | Prevent cascade failures | Backend instability |
| Retry + Backoff | Handle transient errors | Network issues |
| Graceful Degradation | Maintain availability | Overload situations |
| Bulkhead | Isolate failures | Resource contention |
| Timeout | Prevent hanging | Slow operations |
| Health Checks | Detect failures | Load balancing |

---

## Next Steps

1. Implement chaos engineering tests
2. Add distributed tracing
3. Set up alerting for resilience metrics
4. Test failure scenarios regularly

**Your service is now resilient!**

# Production Best Practices for LLM Services

**Learning Module**: Module 6 - Server & Production
**Estimated Reading Time**: 35 minutes
**Prerequisites**: Completion of Module 6.1-6.5
**Related Content**:
- [LLaMA Server Architecture](./01-llama-server-architecture.md)
- [Deployment Patterns](./03-deployment-patterns.md)
- [Monitoring & Observability](./05-monitoring-observability.md)

---

## Overview

This guide consolidates production best practices for running LLM inference services at scale, covering security, reliability, performance, and cost optimization.

---

## Security

### 1. Authentication & Authorization

**Multi-Layer Security**:
```python
from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()
security = HTTPBearer()

# JWT Configuration
SECRET_KEY = "your-secret-key-from-env"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class User:
    def __init__(self, user_id: str, tier: str, permissions: list):
        self.user_id = user_id
        self.tier = tier
        self.permissions = permissions

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
    """Verify JWT token and return user"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        tier = payload.get("tier")
        permissions = payload.get("permissions", [])

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return User(user_id=user_id, tier=tier, permissions=permissions)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_permission(permission: str):
    """Decorator to require specific permission"""
    async def permission_checker(user: User = Depends(verify_token)):
        if permission not in user.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permission: {permission}"
            )
        return user
    return permission_checker

# Usage
@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    user: User = Depends(verify_token)
):
    # User is authenticated
    pass

@app.delete("/v1/models/{model_id}")
async def delete_model(
    model_id: str,
    user: User = Depends(require_permission("models:delete"))
):
    # User has admin permissions
    pass
```

### 2. Input Validation & Sanitization

**Comprehensive Validation**:
```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import re

class Message(BaseModel):
    role: str
    content: str

    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ['system', 'user', 'assistant']
        if v not in allowed_roles:
            raise ValueError(f'role must be one of {allowed_roles}')
        return v

    @validator('content')
    def sanitize_content(cls, v):
        # Remove null bytes
        v = v.replace('\x00', '')

        # Limit length
        max_length = 50000
        if len(v) > max_length:
            raise ValueError(f'content exceeds maximum length of {max_length}')

        # Check for injection attempts
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'onerror=',
            r'onclick='
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('content contains suspicious patterns')

        return v.strip()

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., regex=r'^[a-zA-Z0-9\-]+$')
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError('messages cannot be empty')

        if len(v) > 100:
            raise ValueError('too many messages (max 100)')

        # Check total token count estimate
        total_chars = sum(len(msg.content) for msg in v)
        if total_chars > 100000:
            raise ValueError('total message content too large')

        return v
```

### 3. Rate Limiting & DDoS Protection

**Multi-Tier Rate Limiting**:
```python
import redis
import time
from functools import wraps
from fastapi import HTTPException

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Check if rate limit is exceeded"""
        now = time.time()
        window_start = now - window_seconds

        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = self.redis.zcard(key)

        if request_count >= max_requests:
            return False

        # Add current request
        self.redis.zadd(key, {str(now): now})
        self.redis.expire(key, window_seconds)

        return True

    async def check_tiered_limits(self, user_id: str, tier: str) -> dict:
        """Check multiple rate limits"""
        limits = {
            "free": {
                "per_minute": 10,
                "per_hour": 100,
                "per_day": 1000
            },
            "premium": {
                "per_minute": 60,
                "per_hour": 1000,
                "per_day": 50000
            },
            "enterprise": {
                "per_minute": 600,
                "per_hour": 10000,
                "per_day": 1000000
            }
        }

        tier_limits = limits.get(tier, limits["free"])

        checks = [
            ("minute", tier_limits["per_minute"], 60),
            ("hour", tier_limits["per_hour"], 3600),
            ("day", tier_limits["per_day"], 86400)
        ]

        for period, max_requests, window in checks:
            key = f"ratelimit:{user_id}:{period}"
            if not await self.check_rate_limit(key, max_requests, window):
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {max_requests} requests per {period}",
                    headers={"Retry-After": str(window)}
                )

        return {"allowed": True}

# Middleware
rate_limiter = RateLimiter(redis_client)

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # Extract user ID from token
    user = await get_user_from_request(request)

    if user:
        await rate_limiter.check_tiered_limits(user.user_id, user.tier)

    response = await call_next(request)
    return response
```

### 4. Content Filtering

**Moderation API Integration**:
```python
import httpx
from typing import List

class ContentModerator:
    def __init__(self, moderation_api_key: str):
        self.api_key = moderation_api_key
        self.endpoint = "https://api.openai.com/v1/moderations"

    async def check_content(self, text: str) -> dict:
        """Check content for policy violations"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": text},
                timeout=10.0
            )

        data = response.json()
        results = data["results"][0]

        return {
            "flagged": results["flagged"],
            "categories": results["categories"],
            "category_scores": results["category_scores"]
        }

    async def moderate_request(self, messages: List[dict]) -> bool:
        """Check all user messages"""
        for msg in messages:
            if msg["role"] == "user":
                result = await self.check_content(msg["content"])

                if result["flagged"]:
                    # Log violation
                    logger.warning(
                        "Content policy violation",
                        extra={
                            "categories": result["categories"],
                            "scores": result["category_scores"]
                        }
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Content violates usage policy"
                    )

        return True

# Usage
moderator = ContentModerator(os.getenv("MODERATION_API_KEY"))

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest, user: User = Depends(verify_token)):
    # Moderate input
    await moderator.moderate_request(request.messages)

    # Process request
    pass
```

---

## Reliability

### 1. Graceful Degradation

**Fallback Strategies**:
```python
from enum import Enum
from typing import Optional

class FallbackStrategy(Enum):
    QUEUE = "queue"
    REJECT = "reject"
    CACHE = "cache"
    SIMPLIFIED = "simplified"

class ServiceController:
    def __init__(self):
        self.is_degraded = False
        self.fallback_strategy = FallbackStrategy.QUEUE

    async def handle_request(self, request: ChatCompletionRequest):
        # Check server health
        health = await self.check_backend_health()

        if health["slots_available"] == 0:
            return await self.handle_no_slots(request)

        if health["response_time"] > 10.0:
            self.is_degraded = True
            return await self.handle_degraded_service(request)

        # Normal processing
        return await self.process_normal(request)

    async def handle_no_slots(self, request):
        """Handle when all slots are busy"""
        if self.fallback_strategy == FallbackStrategy.QUEUE:
            # Add to queue
            await self.enqueue_request(request)
            return {
                "status": "queued",
                "message": "Request queued due to high load",
                "estimated_wait_time": await self.estimate_wait_time()
            }

        elif self.fallback_strategy == FallbackStrategy.REJECT:
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable",
                headers={"Retry-After": "60"}
            )

    async def handle_degraded_service(self, request):
        """Handle degraded service"""
        # Reduce quality for speed
        request.max_tokens = min(request.max_tokens, 256)
        request.temperature = 0.0  # Use greedy decoding

        logger.warning("Service degraded, using simplified parameters")

        return await self.process_normal(request)
```

### 2. Circuit Breaker with Auto-Recovery

**Enhanced Circuit Breaker**:
```python
import asyncio
import time
from enum import Enum
from collections import deque

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

        # Track recent failures for metrics
        self.recent_failures = deque(maxlen=100)

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                logger.info("Circuit breaker entering HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN, retry after {self.timeout}s"
                )

        try:
            result = await func(*args, **kwargs)

            # Success handling
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    logger.info("Circuit breaker closing after successful recovery")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.recent_failures.append({
                "time": time.time(),
                "error": str(e)
            })

            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker opening after {self.failure_count} failures"
                )
                self.state = CircuitState.OPEN

            raise e

    def get_state(self) -> dict:
        """Get circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "recent_failures": len(self.recent_failures),
            "time_until_retry": max(
                0,
                self.timeout - (time.time() - self.last_failure_time)
            ) if self.state == CircuitState.OPEN else 0
        }

# Usage
breaker = CircuitBreaker(failure_threshold=5, timeout=60)

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        response = await breaker.call(
            call_llama_server,
            request
        )
        return response
    except CircuitBreakerOpenError:
        # Use fallback
        return await fallback_handler(request)
```

### 3. Retry Logic with Exponential Backoff

**Intelligent Retry**:
```python
import asyncio
from typing import Callable, Any
import random

async def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    **kwargs
) -> Any:
    """Retry function with exponential backoff"""

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except retryable_exceptions as e:
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded")
                raise

            # Calculate delay
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)

            logger.warning(
                f"Request failed (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {delay:.2f}s: {str(e)}"
            )

            await asyncio.sleep(delay)

    # Should never reach here
    raise Exception("Retry logic error")

# Usage
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    response = await retry_with_backoff(
        call_llama_server,
        request,
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(httpx.HTTPError, TimeoutError)
    )
    return response
```

---

## Performance Optimization

### 1. Connection Pooling

**HTTP Connection Pooling**:
```python
import httpx
from typing import Optional

class LlamaServerClient:
    _instance: Optional['LlamaServerClient'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'client'):
            # Connection pool configuration
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )

            timeout = httpx.Timeout(
                connect=5.0,
                read=300.0,
                write=10.0,
                pool=5.0
            )

            self.client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                http2=True  # Enable HTTP/2
            )

    async def chat_completion(self, request: dict) -> dict:
        """Make chat completion request"""
        response = await self.client.post(
            "http://llama-server:8080/v1/chat/completions",
            json=request
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close client connections"""
        await self.client.aclose()

# Usage
llama_client = LlamaServerClient()

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    response = await llama_client.chat_completion(request.dict())
    return response
```

### 2. Response Caching

**Semantic Caching**:
```python
import hashlib
import json
import redis
from typing import Optional

class ResponseCache:
    def __init__(self, redis_client, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl

    def generate_cache_key(self, request: dict) -> str:
        """Generate cache key from request"""
        # Normalize request for caching
        cache_data = {
            "model": request["model"],
            "messages": request["messages"],
            "temperature": request.get("temperature", 0.7),
            "max_tokens": request.get("max_tokens", 512)
        }

        # Create deterministic hash
        request_str = json.dumps(cache_data, sort_keys=True)
        return f"cache:{hashlib.sha256(request_str.encode()).hexdigest()}"

    async def get(self, request: dict) -> Optional[dict]:
        """Get cached response"""
        # Only cache deterministic requests (temperature=0)
        if request.get("temperature", 0.7) != 0.0:
            return None

        key = self.generate_cache_key(request)
        cached = self.redis.get(key)

        if cached:
            logger.info("Cache hit", extra={"cache_key": key})
            return json.loads(cached)

        return None

    async def set(self, request: dict, response: dict):
        """Cache response"""
        if request.get("temperature", 0.7) != 0.0:
            return

        key = self.generate_cache_key(request)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(response)
        )
        logger.info("Response cached", extra={"cache_key": key})

# Usage
cache = ResponseCache(redis_client, ttl=3600)

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    # Check cache
    cached_response = await cache.get(request.dict())
    if cached_response:
        return cached_response

    # Process request
    response = await process_request(request)

    # Cache response
    await cache.set(request.dict(), response)

    return response
```

### 3. Request Batching

**Dynamic Batching**:
```python
import asyncio
from collections import defaultdict
from typing import List

class RequestBatcher:
    def __init__(self, max_batch_size: int = 8, max_wait_ms: int = 100):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = defaultdict(list)
        self.batch_tasks = {}

    async def add_request(self, model: str, request: dict) -> dict:
        """Add request to batch"""
        future = asyncio.Future()

        self.pending_requests[model].append({
            "request": request,
            "future": future
        })

        # Start batch timer if not already running
        if model not in self.batch_tasks:
            self.batch_tasks[model] = asyncio.create_task(
                self._process_batch_after_delay(model)
            )

        # Process immediately if batch is full
        if len(self.pending_requests[model]) >= self.max_batch_size:
            await self._process_batch(model)

        # Wait for result
        return await future

    async def _process_batch_after_delay(self, model: str):
        """Process batch after delay"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        await self._process_batch(model)

    async def _process_batch(self, model: str):
        """Process accumulated requests as batch"""
        if model not in self.pending_requests or not self.pending_requests[model]:
            return

        batch = self.pending_requests[model]
        self.pending_requests[model] = []

        # Cancel timer task
        if model in self.batch_tasks:
            self.batch_tasks[model].cancel()
            del self.batch_tasks[model]

        logger.info(f"Processing batch of {len(batch)} requests")

        # Process batch in parallel
        tasks = [
            call_llama_server(item["request"])
            for item in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Set results on futures
        for item, result in zip(batch, results):
            if isinstance(result, Exception):
                item["future"].set_exception(result)
            else:
                item["future"].set_result(result)

# Usage
batcher = RequestBatcher(max_batch_size=8, max_wait_ms=100)

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    response = await batcher.add_request(request.model, request.dict())
    return response
```

---

## Cost Optimization

### 1. Model Selection & Routing

**Cost-Aware Routing**:
```python
from typing import List

class CostOptimizedRouter:
    def __init__(self):
        self.model_costs = {
            "llama-2-7b": {"compute": 0.0001, "memory_gb": 7, "speed": 1.0},
            "llama-2-13b": {"compute": 0.0002, "memory_gb": 13, "speed": 0.7},
            "llama-2-70b": {"compute": 0.001, "memory_gb": 70, "speed": 0.3}
        }

    def select_model(
        self,
        task_complexity: str,
        max_cost: float,
        latency_requirement: float
    ) -> str:
        """Select most cost-effective model for task"""

        # Simple tasks can use smaller models
        if task_complexity == "simple":
            return "llama-2-7b"

        # For complex tasks with tight latency
        if latency_requirement < 2.0:
            # Need fast model even if more expensive
            return "llama-2-13b"

        # For complex tasks without tight latency, optimize cost
        if max_cost < 0.001:
            return "llama-2-13b"

        return "llama-2-70b"

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        max_completion_tokens: int
    ) -> float:
        """Estimate request cost"""
        total_tokens = prompt_tokens + max_completion_tokens
        cost_per_token = self.model_costs[model]["compute"]
        return total_tokens * cost_per_token
```

### 2. Resource Scheduling

**Off-Peak Processing**:
```python
from datetime import datetime, time

class ResourceScheduler:
    def __init__(self):
        self.peak_hours = [(9, 17)]  # 9 AM to 5 PM

    def is_peak_time(self) -> bool:
        """Check if current time is peak"""
        now = datetime.now().time()

        for start_hour, end_hour in self.peak_hours:
            start = time(start_hour, 0)
            end = time(end_hour, 0)

            if start <= now <= end:
                return True

        return False

    async def schedule_request(
        self,
        request: dict,
        priority: str = "normal"
    ):
        """Schedule request based on priority and time"""

        is_peak = self.is_peak_time()

        if priority == "low" and is_peak:
            # Queue low priority during peak
            await self.queue_for_off_peak(request)
            return {"status": "queued", "scheduled_for": "off-peak"}

        # Process immediately
        return await self.process_now(request)

    async def queue_for_off_peak(self, request: dict):
        """Queue for off-peak processing"""
        # Store in database or queue
        await db.execute(
            "INSERT INTO off_peak_queue (request_data, created_at) VALUES ($1, $2)",
            json.dumps(request),
            datetime.utcnow()
        )
```

---

## Deployment Checklist

### Pre-Production

- [ ] **Security**
  - [ ] Enable HTTPS/TLS
  - [ ] Implement authentication
  - [ ] Set up API key rotation
  - [ ] Enable rate limiting
  - [ ] Configure CORS properly
  - [ ] Run security audit

- [ ] **Performance**
  - [ ] Load test with expected traffic
  - [ ] Optimize model quantization
  - [ ] Configure appropriate context size
  - [ ] Set up connection pooling
  - [ ] Enable caching where appropriate

- [ ] **Reliability**
  - [ ] Implement health checks
  - [ ] Configure auto-scaling
  - [ ] Set up circuit breakers
  - [ ] Test failover scenarios
  - [ ] Implement graceful shutdown

- [ ] **Observability**
  - [ ] Set up metrics collection
  - [ ] Configure log aggregation
  - [ ] Create dashboards
  - [ ] Define alerts
  - [ ] Test alerting pipeline

- [ ] **Cost**
  - [ ] Estimate costs
  - [ ] Set up cost tracking
  - [ ] Configure budget alerts
  - [ ] Optimize resource usage

### Production Operations

- [ ] **Monitoring**
  - [ ] Monitor SLOs continuously
  - [ ] Review metrics daily
  - [ ] Investigate anomalies
  - [ ] Track costs

- [ ] **Maintenance**
  - [ ] Regular security updates
  - [ ] Model updates
  - [ ] Performance tuning
  - [ ] Capacity planning

- [ ] **Incident Response**
  - [ ] On-call rotation
  - [ ] Runbooks for common issues
  - [ ] Postmortem process
  - [ ] Communication plan

---

## Summary

**Security Essentials**:
1. Multi-layer authentication
2. Input validation and sanitization
3. Rate limiting
4. Content moderation

**Reliability Patterns**:
1. Circuit breakers
2. Retry with exponential backoff
3. Graceful degradation
4. Health checks

**Performance Optimization**:
1. Connection pooling
2. Response caching
3. Request batching
4. Efficient resource allocation

**Cost Optimization**:
1. Right-size models
2. Off-peak scheduling
3. Resource monitoring
4. Efficient quantization

**Production Readiness**:
- Comprehensive monitoring
- Automated alerting
- Disaster recovery plan
- Documentation

---

**Interview Topics**:
- Security best practices
- Reliability engineering patterns
- Performance optimization techniques
- Cost optimization strategies
- Production incident handling

**Next Steps**:
- Complete all Module 6 labs
- Build capstone project: Production-ready inference API
- Review interview questions
- Practice system design scenarios

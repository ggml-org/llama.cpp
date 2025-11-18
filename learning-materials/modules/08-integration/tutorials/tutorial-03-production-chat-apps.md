# Tutorial: Building Production-Ready Chat Applications

**Estimated Time**: 75 minutes
**Level**: Advanced

## Overview

Learn to build scalable, production-ready chat applications with llama.cpp, covering architecture, performance, security, and deployment.

## 1. Architecture Patterns

### Microservices Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Frontend  │────▶│  API Gateway │────▶│  Auth Service│
│  (React/Vue)│     │   (FastAPI)  │     │             │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
          ┌─────▼──────┐      ┌──────▼────────┐
          │ Chat Service│      │ Model Service │
          │  (WebSocket)│      │  (llama.cpp) │
          └─────┬──────┘      └───────────────┘
                │
          ┌─────▼──────┐
          │  Database  │
          │ (Postgres) │
          └────────────┘
```

## 2. Real-Time Communication

### WebSocket Handler

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, List[dict]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.user_sessions[client_id] = []

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def broadcast(self, message: dict, exclude: str = None):
        for client_id, connection in self.active_connections.items():
            if client_id != exclude:
                await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()

            # Process message
            response = await process_message(client_id, data)

            # Send response
            await manager.send_message(client_id, response)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
```

## 3. Message Processing

### Message Queue Pattern

```python
import asyncio
from collections import deque

class MessageQueue:
    def __init__(self, max_workers: int = 4):
        self.queue = deque()
        self.workers = max_workers
        self.processing = False

    async def add_message(self, message: dict):
        """Add message to queue."""
        self.queue.append(message)

        if not self.processing:
            await self.process_queue()

    async def process_queue(self):
        """Process messages in queue."""
        self.processing = True

        while self.queue:
            message = self.queue.popleft()
            await self.process_message(message)

        self.processing = False

    async def process_message(self, message: dict):
        """Process single message."""
        # Generate response
        response = await generate_response(message)

        # Send to client
        await send_response(message['client_id'], response)
```

## 4. Conversation Management

### Session State Management

```python
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

class ConversationManager:
    def __init__(self, max_history: int = 20, max_tokens: int = 2000):
        self.sessions: Dict[str, List[Message]] = {}
        self.max_history = max_history
        self.max_tokens = max_tokens

    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        message = Message(role=role, content=content)
        self.sessions[session_id].append(message)

        # Trim if needed
        self._trim_history(session_id)

    def get_messages(self, session_id: str) -> List[Dict]:
        """Get conversation history."""
        messages = self.sessions.get(session_id, [])
        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

    def _trim_history(self, session_id: str):
        """Trim history to fit limits."""
        messages = self.sessions[session_id]

        # Trim by count
        if len(messages) > self.max_history:
            # Keep system message + recent messages
            system_msgs = [m for m in messages if m.role == "system"]
            recent_msgs = messages[-(self.max_history - len(system_msgs)):]
            self.sessions[session_id] = system_msgs + recent_msgs

        # Trim by tokens
        total_tokens = sum(len(m.content) // 4 for m in self.sessions[session_id])
        while total_tokens > self.max_tokens and len(self.sessions[session_id]) > 1:
            # Remove oldest non-system message
            for i, msg in enumerate(self.sessions[session_id]):
                if msg.role != "system":
                    removed = self.sessions[session_id].pop(i)
                    total_tokens -= len(removed.content) // 4
                    break

    def clear_session(self, session_id: str):
        """Clear session history."""
        if session_id in self.sessions:
            # Keep system messages
            system_msgs = [m for m in self.sessions[session_id] if m.role == "system"]
            self.sessions[session_id] = system_msgs
```

## 5. Rate Limiting

### Token Bucket Algorithm

```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, rate: int = 10, per: int = 60):
        """
        Rate limiter using token bucket.

        Args:
            rate: Number of requests
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = defaultdict(lambda: rate)
        self.last_check = defaultdict(lambda: time.time())

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        current = time.time()
        time_passed = current - self.last_check[client_id]
        self.last_check[client_id] = current

        # Add tokens based on time passed
        self.allowance[client_id] += time_passed * (self.rate / self.per)

        # Cap at rate limit
        if self.allowance[client_id] > self.rate:
            self.allowance[client_id] = self.rate

        # Check if we have tokens
        if self.allowance[client_id] < 1.0:
            return False

        # Consume a token
        self.allowance[client_id] -= 1.0
        return True

# Usage in endpoint
rate_limiter = RateLimiter(rate=10, per=60)

@app.post("/chat")
async def chat(request: ChatRequest, client_id: str):
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return await process_chat(request)
```

## 6. Error Handling & Resilience

### Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset."""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
```

## 7. Performance Optimization

### Response Caching

```python
from functools import lru_cache
import hashlib

class ResponseCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str):
        """Get cached response."""
        if key in self.cache:
            # Check if expired
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None

            return self.cache[key]

        return None

    def set(self, key: str, value):
        """Cache response."""
        # Evict if full
        if len(self.cache) >= self.max_size:
            # Remove oldest
            oldest = min(self.timestamps.items(), key=lambda x: x[1])
            del self.cache[oldest[0]]
            del self.timestamps[oldest[0]]

        self.cache[key] = value
        self.timestamps[key] = time.time()

    def cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key from messages."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
```

### Connection Pooling

```python
from llama_cpp import Llama
from queue import Queue
from threading import Lock

class ModelPool:
    def __init__(self, model_path: str, pool_size: int = 4):
        self.pool = Queue(maxsize=pool_size)
        self.pool_size = pool_size

        # Initialize pool
        for _ in range(pool_size):
            model = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=35 // pool_size
            )
            self.pool.put(model)

    def acquire(self) -> Llama:
        """Get model from pool."""
        return self.pool.get()

    def release(self, model: Llama):
        """Return model to pool."""
        self.pool.put(model)

    async def generate(self, *args, **kwargs):
        """Generate using pooled model."""
        model = self.acquire()
        try:
            result = model(*args, **kwargs)
            return result
        finally:
            self.release(model)
```

## 8. Monitoring & Logging

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
requests_total = Counter('chat_requests_total', 'Total chat requests')
request_duration = Histogram('chat_request_duration_seconds', 'Request duration')
active_connections = Gauge('active_websocket_connections', 'Active WebSocket connections')
errors_total = Counter('chat_errors_total', 'Total errors', ['error_type'])

class MetricsMiddleware:
    async def __call__(self, request, call_next):
        requests_total.inc()

        start_time = time.time()
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            errors_total.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            request_duration.observe(time.time() - start_time)
```

## 9. Security Best Practices

### Input Sanitization

```python
import re
from html import escape

class InputSanitizer:
    @staticmethod
    def sanitize_message(text: str) -> str:
        """Sanitize user input."""
        # Remove HTML
        text = escape(text)

        # Limit length
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length]

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    @staticmethod
    def validate_message(text: str) -> bool:
        """Validate message."""
        if not text or len(text) > 5000:
            return False

        # Check for spam patterns
        if re.search(r'(.)\1{20,}', text):  # Repeated characters
            return False

        return True
```

## 10. Production Deployment

### Health Checks

```python
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check model
        model_healthy = await check_model_health()

        # Check database
        db_healthy = await check_database()

        # Check Redis
        cache_healthy = await check_cache()

        return {
            "status": "healthy" if all([
                model_healthy,
                db_healthy,
                cache_healthy
            ]) else "unhealthy",
            "components": {
                "model": model_healthy,
                "database": db_healthy,
                "cache": cache_healthy
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

### Graceful Shutdown

```python
import signal
import asyncio

class GracefulShutdown:
    def __init__(self, app):
        self.app = app
        self.is_shutting_down = False

        signal.signal(signal.SIGTERM, self.handle_sigterm)
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigterm(self, signum, frame):
        """Handle SIGTERM."""
        asyncio.create_task(self.shutdown())

    def handle_sigint(self, signum, frame):
        """Handle SIGINT (Ctrl+C)."""
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        """Graceful shutdown."""
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        print("Shutting down gracefully...")

        # Stop accepting new connections
        # Wait for active requests to complete
        await asyncio.sleep(5)

        # Close connections
        await self.close_connections()

        # Cleanup resources
        await self.cleanup()

        print("Shutdown complete")
```

## Summary

Production chat app checklist:
- [X] WebSocket communication
- [X] Message queue processing
- [X] Conversation management
- [X] Rate limiting
- [X] Error handling & resilience
- [X] Performance optimization
- [X] Security measures
- [X] Monitoring & logging
- [X] Health checks
- [X] Graceful shutdown

---

**Tutorial**: 03 - Production Chat Apps
**Module**: 08 - Integration & Applications
**Version**: 1.0

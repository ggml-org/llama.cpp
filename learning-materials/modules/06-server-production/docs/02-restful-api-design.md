# RESTful API Design for LLM Services

**Learning Module**: Module 6 - Server & Production
**Estimated Reading Time**: 30 minutes
**Prerequisites**: Understanding of HTTP, REST principles, Module 6.1
**Related Content**:
- [LLaMA Server Architecture](./01-llama-server-architecture.md)
- [Production Best Practices](./06-production-best-practices.md)

---

## Introduction

While llama-server provides an OpenAI-compatible API, production applications often require custom endpoints, business logic, and integration with existing systems. This guide covers designing RESTful APIs for LLM services.

### Why Custom APIs?

1. **Business Logic**: Implement domain-specific validation and processing
2. **Integration**: Connect with databases, auth systems, payment processors
3. **Rate Limiting**: Control usage and prevent abuse
4. **Cost Tracking**: Monitor and bill for API usage
5. **Custom Features**: Add functionality not in standard OpenAI API

---

## API Design Principles

### 1. Resource-Oriented Design

**Good - Resource-based**:
```
POST   /api/v1/conversations
GET    /api/v1/conversations/{id}
POST   /api/v1/conversations/{id}/messages
GET    /api/v1/models
POST   /api/v1/embeddings
```

**Bad - Action-based**:
```
POST   /api/v1/createConversation
POST   /api/v1/sendMessage
POST   /api/v1/getModels
```

### 2. Use HTTP Methods Correctly

| Method | Purpose | Idempotent | Safe |
|--------|---------|------------|------|
| GET | Retrieve resource | ✅ | ✅ |
| POST | Create resource | ❌ | ❌ |
| PUT | Replace resource | ✅ | ❌ |
| PATCH | Update resource | ❌ | ❌ |
| DELETE | Delete resource | ✅ | ❌ |

### 3. Versioning Strategy

**URL Versioning** (Recommended):
```
https://api.example.com/v1/chat/completions
https://api.example.com/v2/chat/completions
```

**Header Versioning**:
```
GET /chat/completions
Accept: application/vnd.myapi.v1+json
```

### 4. Consistent Error Responses

**Standard Error Format**:
```json
{
  "error": {
    "code": "invalid_request",
    "message": "Missing required field: model",
    "details": {
      "field": "model",
      "required": true
    },
    "request_id": "req_abc123"
  }
}
```

**HTTP Status Codes**:
- `200 OK`: Successful request
- `201 Created`: Resource created
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Missing/invalid auth
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource doesn't exist
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Server overloaded

---

## Endpoint Design Patterns

### 1. Chat Completion API

**Endpoint**: `POST /v1/chat/completions`

**Request Schema**:
```json
{
  "model": "llama-2-7b-chat",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false,
  "user": "user_123"
}
```

**Response Schema**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "llama-2-7b-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 12,
    "total_tokens": 37
  }
}
```

**Implementation** (Python/FastAPI):
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    user: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    # Validate model
    if request.model not in ["llama-2-7b-chat", "llama-2-13b-chat"]:
        raise HTTPException(status_code=400, detail="Invalid model")

    # Validate message format
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    # Call llama-server backend
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/v1/chat/completions",
            json=request.dict(),
            timeout=30.0
        )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()
```

### 2. Embeddings API

**Endpoint**: `POST /v1/embeddings`

**Request**:
```json
{
  "model": "llama-2-7b",
  "input": "The quick brown fox",
  "user": "user_123"
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
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

**Batch Processing Support**:
```json
{
  "model": "llama-2-7b",
  "input": [
    "First document",
    "Second document",
    "Third document"
  ]
}
```

### 3. Model Management API

**List Models**: `GET /v1/models`

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-2-7b-chat",
      "object": "model",
      "created": 1699999999,
      "owned_by": "meta",
      "context_length": 4096,
      "quantization": "Q4_K_M",
      "parameters": "7B",
      "architecture": "llama"
    }
  ]
}
```

**Get Model Details**: `GET /v1/models/{model_id}`

**Response**:
```json
{
  "id": "llama-2-7b-chat",
  "object": "model",
  "created": 1699999999,
  "owned_by": "meta",
  "details": {
    "architecture": "llama",
    "parameters": 7016656896,
    "context_length": 4096,
    "embedding_dim": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "vocab_size": 32000,
    "quantization": "Q4_K_M",
    "file_size": 3829063104
  },
  "capabilities": {
    "chat": true,
    "completion": true,
    "embeddings": true,
    "function_calling": false
  }
}
```

### 4. Conversation Management

**Create Conversation**: `POST /v1/conversations`

**Request**:
```json
{
  "model": "llama-2-7b-chat",
  "system_prompt": "You are a helpful coding assistant.",
  "user_id": "user_123"
}
```

**Response**:
```json
{
  "id": "conv_abc123",
  "created": 1699999999,
  "model": "llama-2-7b-chat",
  "system_prompt": "You are a helpful coding assistant.",
  "message_count": 0
}
```

**Add Message**: `POST /v1/conversations/{conv_id}/messages`

**Request**:
```json
{
  "content": "How do I reverse a list in Python?",
  "stream": false
}
```

**Get Conversation**: `GET /v1/conversations/{conv_id}`

**Response**:
```json
{
  "id": "conv_abc123",
  "created": 1699999999,
  "model": "llama-2-7b-chat",
  "messages": [
    {
      "id": "msg_001",
      "role": "user",
      "content": "How do I reverse a list in Python?",
      "created": 1699999999
    },
    {
      "id": "msg_002",
      "role": "assistant",
      "content": "You can reverse a list in Python using list.reverse()...",
      "created": 1700000005
    }
  ]
}
```

---

## Authentication & Authorization

### 1. API Key Authentication

**Implementation**:
```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets

security = HTTPBearer()

API_KEYS = {
    "sk-abc123": {"user_id": "user_001", "tier": "premium"},
    "sk-def456": {"user_id": "user_002", "tier": "free"}
}

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    api_key = credentials.credentials

    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return API_KEYS[api_key]

@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    user_info: dict = Security(verify_api_key)
):
    # user_info contains {"user_id": "...", "tier": "..."}
    # Apply tier-based rate limiting
    pass
```

**Request**:
```bash
curl -X POST https://api.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-abc123" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-2-7b", "messages": [...]}'
```

### 2. OAuth 2.0

**For user-facing applications**:

```python
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify JWT token
    user = verify_jwt_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return user

@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    current_user: User = Depends(get_current_user)
):
    # Use current_user for authorization
    pass
```

---

## Rate Limiting

### 1. Token Bucket Algorithm

**Implementation**:
```python
import time
from collections import defaultdict
from fastapi import HTTPException

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.buckets = defaultdict(lambda: {"tokens": requests_per_minute, "last_update": time.time()})

    def allow_request(self, user_id: str) -> bool:
        now = time.time()
        bucket = self.buckets[user_id]

        # Refill tokens based on time elapsed
        time_passed = now - bucket["last_update"]
        bucket["tokens"] += time_passed * (self.requests_per_minute / 60.0)
        bucket["tokens"] = min(bucket["tokens"], self.requests_per_minute)
        bucket["last_update"] = now

        # Check if request is allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False

rate_limiter = RateLimiter(requests_per_minute=60)

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # Get user ID from auth token
    user_id = extract_user_id(request)

    if not rate_limiter.allow_request(user_id):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={"Retry-After": "60"}
        )

    response = await call_next(request)
    return response
```

### 2. Tiered Rate Limits

**Configuration**:
```python
RATE_LIMITS = {
    "free": {
        "requests_per_minute": 10,
        "tokens_per_day": 10000,
        "max_context": 2048
    },
    "premium": {
        "requests_per_minute": 60,
        "tokens_per_day": 1000000,
        "max_context": 4096
    },
    "enterprise": {
        "requests_per_minute": 600,
        "tokens_per_day": 10000000,
        "max_context": 8192
    }
}

async def enforce_rate_limits(user_info: dict, request: ChatCompletionRequest):
    tier = user_info["tier"]
    limits = RATE_LIMITS[tier]

    # Check token usage
    daily_usage = get_daily_token_usage(user_info["user_id"])
    if daily_usage >= limits["tokens_per_day"]:
        raise HTTPException(status_code=429, detail="Daily token limit exceeded")

    # Enforce context limit
    if request.max_tokens > limits["max_context"]:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens exceeds tier limit of {limits['max_context']}"
        )
```

---

## Request Validation

### 1. Input Validation

**Pydantic Schemas**:
```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('role must be system, user, or assistant')
        return v

    @validator('content')
    def validate_content(cls, v):
        if len(v) > 10000:
            raise ValueError('content exceeds maximum length of 10000 characters')
        if not v.strip():
            raise ValueError('content cannot be empty')
        return v

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError('messages cannot be empty')
        if v[-1].role != 'user':
            raise ValueError('last message must be from user')
        return v

    @validator('model')
    def validate_model(cls, v):
        allowed_models = ['llama-2-7b-chat', 'llama-2-13b-chat', 'codellama-7b']
        if v not in allowed_models:
            raise ValueError(f'model must be one of {allowed_models}')
        return v
```

### 2. Content Filtering

**Profanity/Toxicity Check**:
```python
from typing import List

BLOCKED_PATTERNS = [
    # Add patterns to block
    r'\bexplicit_word\b',
    # ...
]

def contains_blocked_content(text: str) -> bool:
    import re
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

async def filter_request(request: ChatCompletionRequest):
    # Check all user messages
    for msg in request.messages:
        if msg.role == 'user' and contains_blocked_content(msg.content):
            raise HTTPException(
                status_code=400,
                detail="Content violates usage policy"
            )
```

---

## Response Streaming

### 1. Server-Sent Events (SSE)

**FastAPI Implementation**:
```python
from fastapi.responses import StreamingResponse
import httpx
import json

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request),
            media_type="text/event-stream"
        )
    else:
        # Regular response
        pass

async def stream_chat_completion(request: ChatCompletionRequest):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8080/v1/chat/completions",
            json=request.dict(),
            timeout=None
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n\n"
```

### 2. WebSocket Streaming

**Alternative for bidirectional communication**:
```python
from fastapi import WebSocket
import json

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            request = json.loads(data)

            # Process with llama-server
            async for chunk in stream_completion(request):
                await websocket.send_text(json.dumps(chunk))

    except WebSocketDisconnect:
        print("Client disconnected")
```

---

## Error Handling

### 1. Comprehensive Error Types

```python
class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: dict = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class InvalidRequestError(APIError):
    def __init__(self, message: str, details: dict = None):
        super().__init__("invalid_request", message, 400, details)

class AuthenticationError(APIError):
    def __init__(self, message: str = "Invalid API key"):
        super().__init__("authentication_error", message, 401)

class RateLimitError(APIError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__("rate_limit_exceeded", message, 429)

class InferenceError(APIError):
    def __init__(self, message: str, details: dict = None):
        super().__init__("inference_error", message, 500, details)

@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "request_id": request.state.request_id
            }
        }
    )
```

### 2. Request ID Tracking

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response

app.add_middleware(RequestIDMiddleware)
```

---

## Usage Tracking & Billing

### 1. Token Usage Tracking

```python
import asyncpg
from datetime import datetime

class UsageTracker:
    def __init__(self, db_pool):
        self.db_pool = db_pool

    async def track_usage(
        self,
        user_id: str,
        request_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float
    ):
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO usage_logs
                (user_id, request_id, model, prompt_tokens, completion_tokens, total_tokens, cost, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                user_id,
                request_id,
                model,
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                cost,
                datetime.utcnow()
            )

    async def get_daily_usage(self, user_id: str) -> dict:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost) as total_cost
                FROM usage_logs
                WHERE user_id = $1
                  AND DATE(created_at) = CURRENT_DATE
                """,
                user_id
            )
            return dict(row) if row else {}

@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    user_info: dict = Security(verify_api_key)
):
    # Make request
    response = await call_llama_server(request)

    # Track usage
    usage = response["usage"]
    cost = calculate_cost(request.model, usage["total_tokens"])

    await usage_tracker.track_usage(
        user_id=user_info["user_id"],
        request_id=response["id"],
        model=request.model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        cost=cost
    )

    return response
```

### 2. Cost Calculation

```python
PRICING = {
    "llama-2-7b-chat": {
        "prompt": 0.0001,  # per 1K tokens
        "completion": 0.0002
    },
    "llama-2-13b-chat": {
        "prompt": 0.0002,
        "completion": 0.0004
    }
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING.get(model, {"prompt": 0, "completion": 0})

    prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1000) * pricing["completion"]

    return prompt_cost + completion_cost
```

---

## Documentation

### 1. OpenAPI/Swagger

**Automatic Generation with FastAPI**:
```python
from fastapi import FastAPI

app = FastAPI(
    title="LLM Inference API",
    description="Production-ready LLM inference service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Endpoints automatically documented
# Visit http://localhost:8000/docs for Swagger UI
# Visit http://localhost:8000/redoc for ReDoc
```

### 2. API Examples

**Include in documentation**:
```python
@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "id": "chatcmpl-abc123",
                        "object": "chat.completion",
                        "model": "llama-2-7b-chat",
                        "choices": [...]
                    }
                }
            }
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"}
    }
)
async def chat_completion(request: ChatCompletionRequest):
    pass
```

---

## Best Practices Summary

### API Design
- ✅ Use RESTful resource-oriented design
- ✅ Version your API (v1, v2)
- ✅ Provide consistent error responses
- ✅ Implement proper HTTP status codes
- ✅ Support pagination for list endpoints

### Security
- ✅ Require authentication for all endpoints
- ✅ Implement rate limiting
- ✅ Validate and sanitize all inputs
- ✅ Log requests for audit trails
- ✅ Use HTTPS in production

### Performance
- ✅ Support streaming for long responses
- ✅ Implement request queuing
- ✅ Use async/await for I/O operations
- ✅ Cache model metadata
- ✅ Monitor response times

### Reliability
- ✅ Handle errors gracefully
- ✅ Provide request IDs for debugging
- ✅ Implement health checks
- ✅ Set appropriate timeouts
- ✅ Use circuit breakers for backend calls

---

## Summary

**Key Takeaways**:
1. Design APIs around resources, not actions
2. Implement robust authentication and authorization
3. Enforce rate limits to prevent abuse
4. Validate all inputs thoroughly
5. Provide clear, consistent error messages
6. Track usage for billing and optimization
7. Document your API comprehensively

**Next Steps**:
- [Deployment Patterns](./03-deployment-patterns.md)
- [Load Balancing & Scaling](./04-load-balancing-scaling.md)
- Lab 6.2: Build a custom API wrapper

---

**Interview Topics**:
- RESTful API design principles
- Authentication strategies
- Rate limiting algorithms
- Error handling best practices
- API versioning strategies

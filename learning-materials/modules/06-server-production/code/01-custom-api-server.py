"""
Custom LLM API Server with Production Features

This server wraps llama-server with:
- Authentication & authorization
- Rate limiting
- Request validation
- Metrics collection
- Logging
- Error handling
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import List, Optional, AsyncGenerator
import httpx
import time
import logging
import json
import jwt
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
LLAMA_SERVER_URL = "http://localhost:8080"

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Metrics
# ============================================================================

request_counter = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'Request duration',
    ['endpoint']
)

active_requests = Gauge(
    'api_active_requests',
    'Currently active requests'
)

tokens_generated = Counter(
    'api_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

# ============================================================================
# Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str

    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('Invalid role')
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
            raise ValueError('Messages cannot be empty')
        if len(v) > 50:
            raise ValueError('Too many messages')
        return v

class User(BaseModel):
    user_id: str
    tier: str
    rate_limit: int

# ============================================================================
# Rate Limiter
# ============================================================================

class InMemoryRateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self):
        self.requests = defaultdict(list)

    def check_rate_limit(self, user_id: str, limit: int, window: int = 60) -> bool:
        """Check if user is within rate limit"""
        now = time.time()
        cutoff = now - window

        # Remove old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if req_time > cutoff
        ]

        # Check limit
        if len(self.requests[user_id]) >= limit:
            return False

        # Add current request
        self.requests[user_id].append(now)
        return True

rate_limiter = InMemoryRateLimiter()

# ============================================================================
# Authentication
# ============================================================================

security = HTTPBearer()

def create_access_token(user_id: str, tier: str = "free") -> str:
    """Create JWT token"""
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode = {
        "sub": user_id,
        "tier": tier,
        "exp": expire
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> User:
    """Verify JWT token and return user"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        tier = payload.get("tier", "free")

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Determine rate limit based on tier
        rate_limits = {
            "free": 10,
            "premium": 100,
            "enterprise": 1000
        }

        return User(
            user_id=user_id,
            tier=tier,
            rate_limit=rate_limits.get(tier, 10)
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# HTTP Client
# ============================================================================

class LlamaClient:
    """HTTP client for llama-server with connection pooling"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> dict:
        """Call llama-server chat completion API"""
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=request.dict()
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LLaMA server error: {response.text}"
            )

        return response.json()

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[bytes, None]:
        """Stream chat completion from llama-server"""
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=request.dict(),
            timeout=None
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

llama_client = LlamaClient(LLAMA_SERVER_URL)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LLM Inference API",
    description="Production-ready LLM inference service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    active_requests.inc()

    try:
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        request_duration.labels(endpoint=request.url.path).observe(duration)
        request_counter.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        return response

    finally:
        active_requests.dec()

# ============================================================================
# Endpoints
# ============================================================================

@app.post("/auth/token")
async def get_token(user_id: str, tier: str = "free"):
    """Generate authentication token"""
    token = create_access_token(user_id, tier)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 86400
    }

@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    user: User = Depends(verify_token)
):
    """Chat completion endpoint with authentication and rate limiting"""

    # Check rate limit
    if not rate_limiter.check_rate_limit(user.user_id, user.rate_limit):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {user.rate_limit} requests per minute"
        )

    logger.info(
        f"Chat completion request from user {user.user_id}",
        extra={
            "user_id": user.user_id,
            "model": request.model,
            "num_messages": len(request.messages)
        }
    )

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            llama_client.chat_completion_stream(request),
            media_type="text/event-stream"
        )

    # Non-streaming request
    start_time = time.time()
    response = await llama_client.chat_completion(request)
    duration = time.time() - start_time

    # Track tokens
    usage = response.get("usage", {})
    tokens_generated.labels(model=request.model).inc(
        usage.get("completion_tokens", 0)
    )

    logger.info(
        f"Request completed in {duration:.2f}s",
        extra={
            "user_id": user.user_id,
            "duration": duration,
            "tokens": usage.get("total_tokens", 0)
        }
    )

    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if llama-server is healthy
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{LLAMA_SERVER_URL}/health",
                timeout=5.0
            )

        if response.status_code == 200:
            return {
                "status": "healthy",
                "llama_server": "up"
            }
        else:
            return {
                "status": "degraded",
                "llama_server": "down"
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await llama_client.close()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

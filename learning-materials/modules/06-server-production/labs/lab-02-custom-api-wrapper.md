# Lab 6.2: Build Custom API Wrapper

**Difficulty**: Intermediate
**Estimated Time**: 60-90 minutes
**Prerequisites**: Lab 6.1, Python programming, FastAPI basics

---

## Learning Objectives

1. Build a custom API wrapper around llama-server
2. Implement authentication and authorization
3. Add rate limiting
4. Collect custom metrics
5. Handle errors gracefully

---

## Part 1: Setup FastAPI Project (15 min)

```bash
# Create project
mkdir llm-api-wrapper
cd llm-api-wrapper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn httpx pydantic python-jose[cryptography] prometheus-client

# Create project structure
mkdir -p app tests
touch app/__init__.py app/main.py app/auth.py app/rate_limiter.py
```

---

## Part 2: Implement Core API (20 min)

Create `app/main.py`:

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import httpx

app = FastAPI(title="LLM API Wrapper", version="1.0.0")

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "llama-2-7b"
    temperature: float = 0.7
    max_tokens: int = 512

# HTTP client for llama-server
client = httpx.AsyncClient(base_url="http://localhost:8080")

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    \"\"\"Proxy chat completion to llama-server\"\"\"
    try:
        response = await client.post(
            "/v1/chat/completions",
            json=request.dict(),
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    \"\"\"Check if service is healthy\"\"\"
    try:
        response = await client.get("/health", timeout=5.0)
        if response.status_code == 200:
            return {"status": "healthy"}
    except:
        pass
    return {"status": "unhealthy"}

# Run: uvicorn app.main:app --reload
```

**Test it**:
```bash
uvicorn app.main:app --reload --port 8000

# In another terminal:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

**✅ Checkpoint**: API forwards requests to llama-server

---

## Part 3: Add Authentication (20 min)

Create `app/auth.py`:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
security = HTTPBearer()

def create_token(user_id: str, tier: str = "free") -> str:
    expire = datetime.utcnow() + timedelta(hours=24)
    payload = {"sub": user_id, "tier": tier, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return {"user_id": payload["sub"], "tier": payload.get("tier", "free")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

Update `app/main.py`:

```python
from app.auth import verify_token, create_token

@app.post("/auth/token")
async def get_token(user_id: str, tier: str = "free"):
    token = create_token(user_id, tier)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatRequest,
    user = Depends(verify_token)
):
    # Now requires authentication!
    # ... rest of implementation
```

**Test authentication**:
```bash
# Get token
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/token?user_id=test_user&tier=premium" | jq -r '.access_token')

# Use token
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

**✅ Checkpoint**: Authentication required for API access

---

## Part 4: Implement Rate Limiting (20 min)

Create `app/rate_limiter.py`:

```python
import time
from collections import defaultdict
from fastapi import HTTPException

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            "free": 10,
            "premium": 100,
            "enterprise": 1000
        }

    def check_limit(self, user_id: str, tier: str) -> bool:
        limit = self.limits.get(tier, 10)
        now = time.time()
        cutoff = now - 60  # 1 minute window

        # Clean old requests
        self.requests[user_id] = [
            t for t in self.requests[user_id] if t > cutoff
        ]

        if len(self.requests[user_id]) >= limit:
            return False

        self.requests[user_id].append(now)
        return True

rate_limiter = RateLimiter()
```

Update endpoint:

```python
from app.rate_limiter import rate_limiter

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, user = Depends(verify_token)):
    # Check rate limit
    if not rate_limiter.check_limit(user["user_id"], user["tier"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # ... process request
```

**✅ Checkpoint**: Rate limiting enforced based on user tier

---

## Part 5: Add Metrics (15 min)

```python
from prometheus_client import Counter, Histogram, generate_latest

request_counter = Counter('api_requests_total', 'Total requests', ['endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])

@app.middleware("http")
async def track_metrics(request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    request_duration.labels(endpoint=request.url.path).observe(duration)
    request_counter.labels(endpoint=request.url.path, status=response.status_code).inc()

    return response

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

**✅ Checkpoint**: Metrics available at `/metrics`

---

## Deliverables

1. Working API wrapper with all features
2. Test script demonstrating:
   - Authentication
   - Rate limiting
   - Error handling
3. Metrics screenshot showing request counts

---

## Challenge

Add request caching for deterministic requests (temperature=0)

**Next Lab**: [Lab 6.3 - Docker Containerization](./lab-03-docker-containerization.md)

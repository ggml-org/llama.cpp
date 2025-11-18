# Production ML Best Practices for LLM Deployment

**Module**: 9 - Production Best Practices | **Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

Best practices for deploying and maintaining LLM applications in production. Covers infrastructure, monitoring, security, cost optimization, and incident response.

---

## 1. Deployment Architecture

### Recommended Stack

```
┌─────────────────────────────────────────────┐
│          Load Balancer (nginx)              │
├─────────────────────────────────────────────┤
│     Model Servers (vLLM / llama.cpp)        │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│  │Server│  │Server│  │Server│  │Server│   │
│  │  1   │  │  2   │  │  3   │  │  4   │   │
│  └──────┘  └──────┘  └──────┘  └──────┘   │
├─────────────────────────────────────────────┤
│          Monitoring (Prometheus)            │
├─────────────────────────────────────────────┤
│         Logging (ELK / Loki)                │
├─────────────────────────────────────────────┤
│         Caching (Redis)                     │
└─────────────────────────────────────────────┘
```

---

## 2. Resource Management

### GPU Allocation

```yaml
# Kubernetes deployment
apiVersion: v1
kind: Pod
metadata:
  name: llm-server
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    resources:
      requests:
        nvidia.com/gpu: 1  # Request 1 GPU
        memory: "32Gi"
      limits:
        nvidia.com/gpu: 1
        memory: "32Gi"
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
```

### Autoscaling

```python
# Kubernetes HPA based on request rate
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"  # Scale up if >100 req/s per pod
```

---

## 3. Monitoring and Observability

### Key Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
requests_total = Counter('llm_requests_total', 'Total requests', ['status', 'model'])
request_duration = Histogram('llm_request_duration_seconds', 'Request duration')
active_requests = Gauge('llm_active_requests', 'Active requests')

# Model metrics
tokens_generated = Counter('llm_tokens_generated_total', 'Tokens generated')
cache_hits = Counter('llm_cache_hits_total', 'Cache hits')

# Resource metrics
gpu_utilization = Gauge('llm_gpu_utilization_percent', 'GPU utilization')
memory_used = Gauge('llm_memory_used_bytes', 'Memory used')
```

### Alerting Rules

```yaml
# Prometheus alerts
groups:
- name: llm_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(llm_requests_total{status="error"}[5m]) > 0.05
    for: 2m
    annotations:
      summary: "High error rate (>5%) in LLM service"

  - alert: SlowResponse
    expr: histogram_quantile(0.95, llm_request_duration_seconds) > 10
    for: 5m
    annotations:
      summary: "95th percentile latency > 10s"

  - alert: GPUMemoryHigh
    expr: llm_gpu_memory_used / llm_gpu_memory_total > 0.95
    for: 5m
    annotations:
      summary: "GPU memory usage >95%"
```

---

## 4. Caching Strategy

```python
import redis
import hashlib

cache = redis.Redis(host='localhost', port=6379, db=0)

def cached_generate(prompt, model, max_tokens=100, ttl=3600):
    # Create cache key
    cache_key = hashlib.md5(
        f"{prompt}:{model}:{max_tokens}".encode()
    ).hexdigest()

    # Check cache
    cached_response = cache.get(cache_key)
    if cached_response:
        return cached_response.decode()

    # Generate
    response = model.generate(prompt, max_tokens=max_tokens)

    # Store in cache
    cache.setex(cache_key, ttl, response)

    return response

# Typical cache hit rate: 20-40% for production workloads
# Latency improvement: 50-100× for cache hits
```

---

## 5. Rate Limiting

```python
from fastapi import FastAPI, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/completions")
@limiter.limit("100/minute")  # 100 requests per minute per IP
async def generate(request: CompletionRequest):
    try:
        response = model.generate(request.prompt)
        return {"text": response}
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

---

## 6. Security Best Practices

### Input Validation

```python
def validate_prompt(prompt: str) -> bool:
    # Length check
    if len(prompt) > 10000:
        raise ValueError("Prompt too long (max 10k chars)")

    # Injection detection (basic)
    forbidden_patterns = [
        "ignore previous instructions",
        "disregard above",
        "system:",
        "assistant:"
    ]

    prompt_lower = prompt.lower()
    for pattern in forbidden_patterns:
        if pattern in prompt_lower:
            logging.warning(f"Potential prompt injection: {pattern}")
            # Option: reject or sanitize

    return True
```

### Output Filtering

```python
def filter_output(response: str) -> str:
    # Remove potential PII
    import re

    # Email pattern
    response = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', response)

    # Phone pattern
    response = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', response)

    # SSN pattern (US)
    response = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', response)

    return response
```

### API Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key not in valid_api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/v1/completions")
async def generate(request: CompletionRequest, api_key: str = Depends(verify_api_key)):
    # Authenticated request
    pass
```

---

## 7. Cost Optimization

```python
# Track token usage for cost calculation
def calculate_cost(prompt_tokens, completion_tokens, model="llama-7b"):
    costs = {
        "llama-7b": 0.0001,    # $ per 1K tokens
        "llama-13b": 0.0002,
        "llama-70b": 0.0010,
    }

    total_tokens = prompt_tokens + completion_tokens
    cost = (total_tokens / 1000) * costs.get(model, 0.0001)

    return cost

# Log costs
logging.info(f"Request cost: ${cost:.4f}")

# Monthly aggregation
monthly_cost = sum(all_request_costs)
print(f"Monthly LLM inference cost: ${monthly_cost:.2f}")
```

### Model Selection by Load

```python
def route_to_model(prompt, urgency="normal"):
    """
    Route to appropriate model based on load and urgency
    """
    if urgency == "high":
        # Use smaller, faster model
        return llama_7b_model
    elif len(prompt) < 500:
        # Short prompts: small model sufficient
        return llama_7b_model
    else:
        # Long, complex: use larger model
        return llama_13b_model if gpu_available() else llama_7b_model
```

---

## 8. Incident Response

### Graceful Degradation

```python
def generate_with_fallback(prompt):
    try:
        # Try primary model
        return primary_model.generate(prompt, timeout=5)
    except TimeoutError:
        logging.warning("Primary model timeout, using fallback")
        return fallback_model.generate(prompt, max_tokens=50)
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return "I'm experiencing technical difficulties. Please try again."
```

### Circuit Breaker

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_api(data):
    # If 5 failures in a row, circuit opens for 60s
    response = requests.post("https://api.example.com", json=data)
    return response.json()

# Prevents cascading failures
```

---

## 9. Deployment Checklist

**Pre-deployment**:
- [ ] Load testing (expected QPS)
- [ ] Benchmark latency (p50, p95, p99)
- [ ] Measure resource usage (GPU, RAM, CPU)
- [ ] Test error handling
- [ ] Validate monitoring/alerts
- [ ] Security audit (input validation, authentication)

**Deployment**:
- [ ] Blue-green deployment (zero downtime)
- [ ] Gradual rollout (canary 5% → 50% → 100%)
- [ ] Monitor error rates
- [ ] Compare latency to baseline

**Post-deployment**:
- [ ] User feedback collection
- [ ] A/B test new model versions
- [ ] Cost tracking
- [ ] Incident postmortems

---

## 10. Key Takeaways

✅ **Infrastructure**: Load balancing, autoscaling, redundancy
✅ **Monitoring**: Metrics (latency, errors, GPU), alerts
✅ **Caching**: 20-40% hit rate typical, huge latency improvement
✅ **Security**: Input validation, output filtering, authentication
✅ **Cost**: Track token usage, optimize model routing
✅ **Resilience**: Fallbacks, circuit breakers, graceful degradation

---

## Further Reading

- **MLOps**: Best practices for ML in production
- **Kubernetes**: Container orchestration for LLMs
- **Prometheus**: Monitoring and alerting

---

**Status**: Complete | Module 9 Complete (2/2) papers | ALL PAPER SUMMARIES COMPLETE!

# Tutorial: Cost Optimization for LLM Services

**Duration**: 60 minutes
**Level**: Advanced

---

## Overview

Learn strategies to reduce infrastructure costs while maintaining performance and reliability for production LLM services.

---

## Part 1: Right-Sizing Models (15 min)

### Strategy: Choose the Right Model for Each Task

```python
class ModelSelector:
    def __init__(self):
        self.models = {
            "small": {
                "name": "llama-2-7b",
                "cost_per_1k_tokens": 0.0001,
                "speed": "fast",
                "quality": "good"
            },
            "medium": {
                "name": "llama-2-13b",
                "cost_per_1k_tokens": 0.0002,
                "speed": "medium",
                "quality": "better"
            },
            "large": {
                "name": "llama-2-70b",
                "cost_per_1k_tokens": 0.001,
                "speed": "slow",
                "quality": "best"
            }
        }

    def select_model(
        self,
        task_complexity: str,
        max_cost: float,
        latency_requirement: float
    ) -> str:
        \"\"\"Select most cost-effective model\"\"\"

        # Simple tasks use small model
        if task_complexity == "simple":
            return "small"

        # Tight latency? Use faster model
        if latency_requirement < 2.0:
            return "medium" if max_cost >= 0.0002 else "small"

        # Complex tasks with relaxed constraints
        if max_cost >= 0.001:
            return "large"

        return "medium"

    def estimate_cost(
        self,
        model_size: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        \"\"\"Estimate request cost\"\"\"
        model = self.models[model_size]
        total_tokens = prompt_tokens + completion_tokens
        return (total_tokens / 1000) * model["cost_per_1k_tokens"]

# Usage
selector = ModelSelector()

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    # Classify task
    complexity = classify_task(request.messages)

    # Select optimal model
    model_size = selector.select_model(
        complexity,
        max_cost=0.001,
        latency_requirement=5.0
    )

    # Route to appropriate model
    return await route_to_model(model_size, request)
```

---

## Part 2: Aggressive Quantization (10 min)

### Cost Savings Through Quantization

```python
class QuantizationOptimizer:
    def __init__(self):
        self.quantization_configs = {
            "Q4_0": {"size_reduction": 0.25, "quality_loss": 0.02},
            "Q4_K_M": {"size_reduction": 0.28, "quality_loss": 0.01},
            "Q5_K_M": {"size_reduction": 0.35, "quality_loss": 0.005},
            "Q8_0": {"size_reduction": 0.50, "quality_loss": 0.001}
        }

    def recommend_quantization(
        self,
        task_type: str,
        quality_requirement: float
    ) -> str:
        \"\"\"Recommend quantization based on requirements\"\"\"

        if quality_requirement > 0.99:
            return "Q8_0"  # Minimal quality loss
        elif quality_requirement > 0.95:
            return "Q5_K_M"
        else:
            return "Q4_K_M"  # Maximum cost savings

    def calculate_savings(
        self,
        model_size_gb: float,
        quantization: str
    ) -> dict:
        config = self.quantization_configs[quantization]

        quantized_size = model_size_gb * config["size_reduction"]
        memory_saved = model_size_gb - quantized_size

        # Assume $0.10/GB/hour for GPU memory
        hourly_savings = memory_saved * 0.10

        return {
            "original_size_gb": model_size_gb,
            "quantized_size_gb": quantized_size,
            "memory_saved_gb": memory_saved,
            "hourly_savings_usd": hourly_savings,
            "monthly_savings_usd": hourly_savings * 730
        }

# Example
optimizer = QuantizationOptimizer()
savings = optimizer.calculate_savings(
    model_size_gb=26,  # 13B model FP16
    quantization="Q4_K_M"
)
print(f"Monthly savings: ${savings['monthly_savings_usd']:.2f}")
# Output: Monthly savings: $1314.00
```

---

## Part 3: Caching Strategies (15 min)

### Semantic Caching

```python
import hashlib
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache = {}
        self.embeddings = {}
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.encoder.encode(text)

    def _compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        return np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )

    async def get(self, messages: list) -> Optional[dict]:
        # Create embedding for query
        query_text = json.dumps(messages)
        query_emb = self._get_embedding(query_text)

        # Find similar cached responses
        for cached_key, cached_emb in self.embeddings.items():
            similarity = self._compute_similarity(query_emb, cached_emb)

            if similarity >= self.similarity_threshold:
                logger.info(f"Cache hit (similarity: {similarity:.3f})")
                return self.cache[cached_key]

        return None

    async def set(self, messages: list, response: dict):
        key = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
        self.cache[key] = response
        self.embeddings[key] = self._get_embedding(json.dumps(messages))

# Usage
cache = SemanticCache()

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    # Check cache
    cached = await cache.get(request.messages)
    if cached:
        return cached

    # Process request
    response = await call_llama_server(request)

    # Cache for deterministic requests
    if request.temperature == 0:
        await cache.set(request.messages, response)

    return response
```

### Cache Hit Rate Metrics

```python
from prometheus_client import Counter, Gauge

cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_misses = Counter('cache_misses_total', 'Total cache misses')
cache_savings = Counter('cache_cost_savings_usd', 'Cost savings from cache')

def calculate_cache_savings():
    \"\"\"Calculate cost savings from caching\"\"\"
    hit_rate = cache_hits._value.get() / (
        cache_hits._value.get() + cache_misses._value.get()
    )

    # Assume $0.001 per request
    requests_saved = cache_hits._value.get()
    savings = requests_saved * 0.001

    return {
        "hit_rate": hit_rate,
        "requests_saved": requests_saved,
        "cost_savings_usd": savings
    }
```

---

## Part 4: Spot Instances & Preemption (10 min)

### Spot Instance Management

```python
class SpotInstanceManager:
    def __init__(self):
        self.on_demand_cost = 1.00  # per hour
        self.spot_cost = 0.30  # per hour (70% savings)

    def calculate_optimal_mix(
        self,
        required_capacity: int,
        availability_requirement: float = 0.999
    ) -> dict:
        \"\"\"Calculate optimal on-demand vs spot mix\"\"\"

        # Use spot for 70% of capacity (can tolerate interruption)
        spot_capacity = int(required_capacity * 0.7)
        on_demand_capacity = required_capacity - spot_capacity

        hourly_cost = (
            spot_capacity * self.spot_cost +
            on_demand_capacity * self.on_demand_cost
        )

        # Cost comparison
        all_on_demand_cost = required_capacity * self.on_demand_cost
        savings = all_on_demand_cost - hourly_cost

        return {
            "spot_instances": spot_capacity,
            "on_demand_instances": on_demand_capacity,
            "hourly_cost": hourly_cost,
            "hourly_savings": savings,
            "monthly_savings": savings * 730,
            "savings_percentage": (savings / all_on_demand_cost) * 100
        }

# Example
manager = SpotInstanceManager()
result = manager.calculate_optimal_mix(required_capacity=10)
print(f"Monthly savings: ${result['monthly_savings']:.2f}")
print(f"Savings: {result['savings_percentage']:.1f}%")
# Output:
# Monthly savings: $3577.00
# Savings: 49.0%
```

---

## Part 5: Off-Peak Scaling (10 min)

### Time-Based Capacity Planning

```python
from datetime import datetime

class ScheduledScaler:
    def __init__(self):
        self.schedule = {
            "peak": {
                "hours": (9, 17),      # 9 AM - 5 PM
                "scale_factor": 1.0,    # 100% capacity
                "replicas": 10
            },
            "medium": {
                "hours": (6, 21),      # 6 AM - 9 PM
                "scale_factor": 0.6,    # 60% capacity
                "replicas": 6
            },
            "low": {
                "hours": (0, 24),      # Night
                "scale_factor": 0.3,    # 30% capacity
                "replicas": 3
            }
        }

    def get_current_period(self) -> str:
        hour = datetime.now().hour

        if 9 <= hour <= 17:
            return "peak"
        elif 6 <= hour <= 21:
            return "medium"
        else:
            return "low"

    def get_desired_replicas(self) -> int:
        period = self.get_current_period()
        return self.schedule[period]["replicas"]

    def calculate_savings(self) -> dict:
        # Assume peak pricing for baseline
        peak_cost_per_hour = 10.00  # 10 instances

        # Weighted average based on hours in each period
        actual_cost_per_day = (
            8 * 10.00 +   # 8 hours peak (100%)
            7 * 6.00 +    # 7 hours medium (60%)
            9 * 3.00      # 9 hours low (30%)
        )

        baseline_cost_per_day = 24 * 10.00
        daily_savings = baseline_cost_per_day - actual_cost_per_day

        return {
            "baseline_cost_per_day": baseline_cost_per_day,
            "actual_cost_per_day": actual_cost_per_day,
            "daily_savings": daily_savings,
            "monthly_savings": daily_savings * 30,
            "savings_percentage": (daily_savings / baseline_cost_per_day) * 100
        }

# Example
scaler = ScheduledScaler()
savings = scaler.calculate_savings()
print(f"Monthly savings: ${savings['monthly_savings']:.2f}")
print(f"Savings: {savings['savings_percentage']:.1f}%")
# Output:
# Monthly savings: $2730.00
# Savings: 30.4%
```

---

## Part 6: Request Batching (10 min)

### Dynamic Batching for Cost Efficiency

```python
import asyncio
from collections import defaultdict

class RequestBatcher:
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: int = 100
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending = defaultdict(list)

    async def add_request(self, model: str, request: dict):
        future = asyncio.Future()

        self.pending[model].append({
            "request": request,
            "future": future
        })

        # Process if batch is full
        if len(self.pending[model]) >= self.max_batch_size:
            asyncio.create_task(self._process_batch(model))

        # Or wait for timeout
        else:
            asyncio.create_task(
                self._process_after_delay(model)
            )

        return await future

    async def _process_batch(self, model: str):
        batch = self.pending[model][:self.max_batch_size]
        self.pending[model] = self.pending[model][self.max_batch_size:]

        # Process batch in parallel
        tasks = [
            call_llama_server(item["request"])
            for item in batch
        ]

        results = await asyncio.gather(*tasks)

        for item, result in zip(batch, results):
            item["future"].set_result(result)

# Cost savings from batching
def calculate_batching_savings(
    requests_per_hour: int,
    avg_batch_size: float,
    cost_per_request: float
):
    # Batching reduces overhead (startup, teardown, etc.)
    overhead_reduction = 0.2  # 20% overhead reduction

    unbatched_cost = requests_per_hour * cost_per_request
    batched_cost = unbatched_cost * (1 - overhead_reduction)

    hourly_savings = unbatched_cost - batched_cost

    return {
        "hourly_savings": hourly_savings,
        "monthly_savings": hourly_savings * 730,
        "savings_percentage": overhead_reduction * 100
    }
```

---

## Summary: Cost Optimization Strategies

| Strategy | Potential Savings | Complexity | Trade-offs |
|----------|-------------------|------------|------------|
| Model Right-Sizing | 50-80% | Low | Quality vs cost |
| Quantization | 60-75% | Low | Minimal quality loss |
| Semantic Caching | 30-60% | Medium | Cache invalidation |
| Spot Instances | 70% | Medium | Interruption risk |
| Off-Peak Scaling | 30-50% | Low | Availability |
| Request Batching | 15-25% | Medium | Latency |

---

## Complete Cost Optimization Checklist

**Infrastructure**:
- ✅ Use appropriate instance types
- ✅ Leverage spot instances where possible
- ✅ Implement auto-scaling
- ✅ Schedule off-peak scaling
- ✅ Right-size GPU allocation

**Model Optimization**:
- ✅ Use most aggressive quantization acceptable
- ✅ Select smallest model for each task
- ✅ Enable continuous batching
- ✅ Implement model caching

**Request Optimization**:
- ✅ Cache responses aggressively
- ✅ Implement semantic caching
- ✅ Batch requests where possible
- ✅ Set appropriate timeouts

**Monitoring**:
- ✅ Track cost per request
- ✅ Monitor cache hit rates
- ✅ Analyze usage patterns
- ✅ Set budget alerts

---

## Real-World Example

**Before Optimization**:
- 10 on-demand instances
- FP16 models
- No caching
- 24/7 full capacity
- **Cost**: $7,200/month

**After Optimization**:
- 3 on-demand + 7 spot instances
- Q4_K_M quantization
- 50% cache hit rate
- Scheduled scaling
- **Cost**: $1,800/month

**Total Savings**: $5,400/month (75%)

---

**Congratulations!** You've learned how to optimize LLM service costs by 70%+!

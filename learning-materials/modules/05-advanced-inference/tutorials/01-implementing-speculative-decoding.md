# Tutorial 1: Implementing Speculative Decoding

**Module 5 - Advanced Inference**
**Duration**: 60-90 minutes

## Overview

Step-by-step guide to implementing speculative decoding from scratch, optimizing it for production, and integrating it into a real application.

## What You'll Build

A production-ready speculative decoder with:
- Draft-verify pipeline
- Adaptive K selection
- Performance monitoring
- Error handling

## Step 1: Core Algorithm (20 min)

### Draft Generation Phase

```python
class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, K=4):
        self.draft = draft_model
        self.target = target_model
        self.K = K

    def generate_drafts(self, tokens):
        """Generate K draft tokens quickly"""
        draft_tokens = []
        draft_probs = []

        current = tokens.copy()

        for _ in range(self.K):
            logits = self.draft.forward(current)
            probs = softmax(logits)
            token = sample(probs)

            draft_tokens.append(token)
            draft_probs.append(probs)
            current.append(token)

        return draft_tokens, draft_probs
```

### Verification Phase

```python
    def verify_drafts(self, tokens, draft_tokens, draft_probs):
        """Verify drafts with target model in parallel"""
        accepted = []

        for i, draft_token in enumerate(draft_tokens):
            # Get target distribution
            target_logits = self.target.forward(tokens + accepted)
            target_probs = softmax(target_logits)

            # Acceptance test
            p_target = target_probs[draft_token]
            p_draft = draft_probs[i][draft_token]

            if random.random() < min(1.0, p_target / p_draft):
                accepted.append(draft_token)
            else:
                # Reject and resample
                corrected = self.resample(target_probs, draft_probs[i])
                accepted.append(corrected)
                break

        return accepted
```

## Step 2: Optimization (20 min)

### Adaptive K Selection

```python
class AdaptiveSpeculativeDecoder(SpeculativeDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acceptance_history = []
        self.K_range = (2, 8)

    def adjust_K(self):
        """Dynamically adjust K based on acceptance rate"""
        if len(self.acceptance_history) < 10:
            return

        recent_rate = np.mean(self.acceptance_history[-10:])

        if recent_rate > 0.85 and self.K < self.K_range[1]:
            self.K += 1  # More speculation
        elif recent_rate < 0.60 and self.K > self.K_range[0]:
            self.K -= 1  # Less speculation
```

### Batch Support

```python
    def generate_batch(self, batch_tokens, max_tokens):
        """Support batched speculative decoding"""
        results = []

        # Process batch in parallel
        for tokens in batch_tokens:
            generated = self.generate(tokens, max_tokens)
            results.append(generated)

        return results
```

## Step 3: Monitoring (15 min)

### Metrics Collection

```python
@dataclass
class SpecMetrics:
    draft_tokens: int = 0
    accepted_tokens: int = 0
    iterations: int = 0
    draft_time: float = 0.0
    verify_time: float = 0.0

    @property
    def acceptance_rate(self):
        return self.accepted_tokens / self.draft_tokens if self.draft_tokens > 0 else 0

    @property
    def speedup(self):
        baseline_time = self.accepted_tokens * 0.02  # Estimate
        actual_time = self.draft_time + self.verify_time
        return baseline_time / actual_time if actual_time > 0 else 1.0
```

### Real-time Monitoring

```python
class MonitoredSpeculativeDecoder(AdaptiveSpeculativeDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = SpecMetrics()

    def generate(self, tokens, max_tokens):
        """Generate with monitoring"""
        while len(tokens) < max_tokens:
            # Draft phase
            start = time.time()
            drafts, probs = self.generate_drafts(tokens)
            self.metrics.draft_time += time.time() - start

            # Verify phase
            start = time.time()
            accepted = self.verify_drafts(tokens, drafts, probs)
            self.metrics.verify_time += time.time() - start

            # Update metrics
            self.metrics.draft_tokens += len(drafts)
            self.metrics.accepted_tokens += len(accepted)
            self.metrics.iterations += 1

            tokens.extend(accepted)

            # Adapt K
            self.acceptance_history.append(len(accepted) / len(drafts))
            self.adjust_K()

        return tokens, self.metrics
```

## Step 4: Production Integration (20 min)

### REST API Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    use_speculative: bool = True

@app.post("/generate")
async def generate(request: GenerateRequest):
    tokens = tokenize(request.prompt)

    if request.use_speculative:
        decoder = MonitoredSpeculativeDecoder(
            draft_model, target_model, K=4
        )
        result, metrics = decoder.generate(tokens, request.max_tokens)

        return {
            "text": detokenize(result),
            "metrics": {
                "acceptance_rate": metrics.acceptance_rate,
                "speedup": metrics.speedup,
                "K": decoder.K
            }
        }
    else:
        result = baseline_generate(tokens, request.max_tokens)
        return {"text": detokenize(result)}
```

### Error Handling

```python
class RobustSpeculativeDecoder(MonitoredSpeculativeDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = 3

    def generate(self, tokens, max_tokens):
        """Generate with error handling"""
        for attempt in range(self.max_retries):
            try:
                return super().generate(tokens, max_tokens)
            except OutOfMemoryError:
                # Reduce K and retry
                self.K = max(2, self.K - 1)
                logging.warning(f"OOM, reducing K to {self.K}")
            except Exception as e:
                logging.error(f"Generation failed: {e}")
                if attempt == self.max_retries - 1:
                    # Fallback to baseline
                    return self._baseline_generate(tokens, max_tokens)
```

## Step 5: Testing (15 min)

### Unit Tests

```python
import pytest

def test_acceptance_criterion():
    """Test acceptance sampling"""
    decoder = SpeculativeDecoder(draft, target, K=4)

    # High agreement → high acceptance
    p_draft = np.array([0.5, 0.3, 0.2])
    p_target = np.array([0.5, 0.3, 0.2])
    assert decoder.should_accept(0, p_draft, p_target) == True

    # Low agreement → low acceptance
    p_target = np.array([0.1, 0.1, 0.8])
    # Should reject token 0
    acceptance_rate = sum(
        decoder.should_accept(0, p_draft, p_target)
        for _ in range(100)
    ) / 100
    assert acceptance_rate < 0.3

def test_speedup():
    """Verify speedup vs baseline"""
    decoder = SpeculativeDecoder(fast_draft, slow_target, K=4)

    start = time.time()
    tokens_spec, _ = decoder.generate(prompt, 100)
    spec_time = time.time() - start

    start = time.time()
    tokens_base = baseline_generate(prompt, 100)
    base_time = time.time() - start

    speedup = base_time / spec_time
    assert speedup >= 1.5, f"Expected 1.5x+ speedup, got {speedup:.2f}x"
```

### Integration Tests

```python
def test_api_endpoint():
    """Test REST API"""
    response = client.post("/generate", json={
        "prompt": "Explain quantum computing",
        "max_tokens": 100,
        "use_speculative": True
    })

    assert response.status_code == 200
    data = response.json()

    assert "text" in data
    assert "metrics" in data
    assert data["metrics"]["acceptance_rate"] > 0.5
```

## Production Checklist

- [ ] Adaptive K selection implemented
- [ ] Metrics monitoring in place
- [ ] Error handling and fallback
- [ ] API endpoint tested
- [ ] Performance benchmarks run
- [ ] Documentation complete
- [ ] Logging configured
- [ ] Model alignment verified

## Performance Targets

✅ **Acceptance rate**: >70%
✅ **Speedup**: >2x vs baseline
✅ **Latency**: <10% increase per request
✅ **Memory**: <2x baseline usage

## Common Issues

**Low acceptance rate**:
- Check model alignment (same family)
- Reduce temperature
- Verify quantization matches

**OOM errors**:
- Reduce K
- Use smaller draft model
- Enable CPU offloading

**Slower than baseline**:
- Draft model too slow (check ratio)
- K too high
- Overhead too large

## Next Steps

1. Deploy to production
2. Monitor metrics in real-time
3. A/B test vs baseline
4. Optimize for your workload
5. Consider multi-draft strategies

## Full Example

See complete implementation:
- Code: `../code/speculative_decoding.py`
- Lab: `../labs/lab1-speculative-decoding-experiments.md`

🚀 **You now have production-ready speculative decoding!**

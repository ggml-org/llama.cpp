# Testing and Validation for ML Systems

**Module**: 9 - Production Best Practices | **Impact**: ⭐⭐⭐⭐

---

## Executive Summary

Testing LLM applications requires different approaches than traditional software. Covers unit tests, integration tests, evaluation benchmarks, and monitoring.

---

## 1. Testing Pyramid for LLM Systems

```
     ┌─────────────────┐
     │  End-to-End     │ ← Full system, human eval
     │  (5%)           │
     ├─────────────────┤
     │  Integration    │ ← Model + retrieval + API
     │  (25%)          │
     ├─────────────────┤
     │  Unit Tests     │ ← Individual components
     │  (70%)          │
     └─────────────────┘
```

---

## 2. Unit Tests

### 2.1 Deterministic Components

```python
import pytest

def test_chunk_text():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=256, overlap=50)

    assert len(chunks) == 5  # Deterministic
    assert all(len(c) <= 256 for c in chunks)
    assert chunks[0][-50:] == chunks[1][:50]  # Overlap check

def test_prompt_formatting():
    prompt = format_prompt(context="...", question="...")

    assert "[INST]" in prompt  # LLaMA format
    assert "Context:" in prompt
    assert "Question:" in prompt
```

### 2.2 Model Loading

```python
def test_model_loads():
    model = load_model("model.gguf")
    assert model is not None
    assert model.n_ctx >= 2048

def test_model_inference_shape():
    model = load_model("model.gguf")
    output = model("Test prompt", max_tokens=10)

    assert isinstance(output, str)
    assert len(output) > 0
```

---

## 3. Integration Tests

### 3.1 RAG Pipeline

```python
def test_rag_pipeline():
    # Setup
    rag = RAGSystem(model_path="...", vector_db_path="...")

    # Test retrieval
    docs = rag.retrieve("What is Python?")
    assert len(docs) > 0
    assert "python" in docs[0].lower()  # Relevant

    # Test generation
    answer = rag.query("What is Python?")
    assert len(answer) > 20  # Non-trivial answer
    assert "program" in answer.lower() or "language" in answer.lower()

def test_rag_with_no_results():
    rag = RAGSystem(...)
    answer = rag.query("xyzabc nonsense")
    # Should handle gracefully
    assert "don't know" in answer.lower() or "no information" in answer.lower()
```

### 3.2 API Tests

```python
import requests

def test_api_health():
    response = requests.get("http://localhost:8080/health")
    assert response.status_code == 200

def test_api_generation():
    response = requests.post(
        "http://localhost:8080/v1/completions",
        json={"prompt": "Hello", "max_tokens": 10}
    )

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
```

---

## 4. Evaluation Benchmarks

### 4.1 Standard Benchmarks

```python
from lm_eval import evaluator

# MMLU, HellaSwag, etc.
results = evaluator.simple_evaluate(
    model="llama-2-7b-q4_K_M.gguf",
    tasks=["mmlu", "hellaswag", "truthfulqa"],
    num_fewshot=5
)

print(results)
# {'mmlu': {'acc': 0.461}, 'hellaswag': {'acc': 0.756}, ...}
```

### 4.2 Custom Evaluation Sets

```python
test_cases = [
    {
        "question": "What is the capital of France?",
        "expected_answer": "Paris",
        "category": "factual"
    },
    {
        "question": "Explain quantum entanglement simply",
        "evaluation": "contains_keywords",
        "keywords": ["particles", "connected", "distance"],
        "category": "explanation"
    }
]

def evaluate_model(model, test_cases):
    results = {"correct": 0, "total": len(test_cases)}

    for case in test_cases:
        answer = model.generate(case["question"])

        if case.get("expected_answer"):
            if case["expected_answer"].lower() in answer.lower():
                results["correct"] += 1
        elif case.get("evaluation") == "contains_keywords":
            if all(kw.lower() in answer.lower() for kw in case["keywords"]):
                results["correct"] += 1

    results["accuracy"] = results["correct"] / results["total"]
    return results
```

---

## 5. Regression Testing

```python
# Snapshot testing for LLM outputs

def test_output_regression(snapshot):
    model = load_model("model.gguf")

    prompt = "Explain gravity in one sentence."
    output = model(prompt, max_tokens=50, temperature=0.0)  # Deterministic

    # Compare to saved snapshot
    snapshot.assert_match(output, "gravity_explanation")

# If output changes, test fails → manual review required
```

---

## 6. Performance Testing

```python
import time

def test_latency():
    model = load_model("model.gguf")

    start = time.time()
    _ = model("Test prompt", max_tokens=100)
    latency = time.time() - start

    assert latency < 5.0  # Should complete in <5s

def test_throughput():
    # Batch processing test
    prompts = ["Test prompt"] * 100

    start = time.time()
    for prompt in prompts:
        model(prompt, max_tokens=10)
    total_time = time.time() - start

    throughput = len(prompts) / total_time
    assert throughput > 5  # >5 req/s
```

---

## 7. Monitoring in Production

```python
import logging
from prometheus_client import Counter, Histogram

# Metrics
request_count = Counter('llm_requests_total', 'Total requests')
request_latency = Histogram('llm_request_latency_seconds', 'Request latency')
error_count = Counter('llm_errors_total', 'Total errors')

def monitored_generate(prompt):
    request_count.inc()

    with request_latency.time():
        try:
            response = model.generate(prompt)
            log_output_quality(prompt, response)  # Custom metrics
            return response
        except Exception as e:
            error_count.inc()
            logging.error(f"Generation failed: {e}")
            raise

def log_output_quality(prompt, response):
    # Custom metrics
    if len(response) < 10:
        logging.warning(f"Short response: {len(response)} chars")

    if "error" in response.lower() or "sorry" in response.lower():
        logging.info("Model indicated uncertainty")
```

---

## 8. A/B Testing

```python
def ab_test_models(model_a, model_b, test_prompts):
    results = {"model_a": [], "model_b": []}

    for prompt in test_prompts:
        # Generate from both
        response_a = model_a(prompt)
        response_b = model_b(prompt)

        # Human or automated evaluation
        winner = compare_responses(response_a, response_b)

        results[winner].append(prompt)

    print(f"Model A wins: {len(results['model_a'])}")
    print(f"Model B wins: {len(results['model_b'])}")

    return results
```

---

## 9. Key Takeaways

✅ **Unit tests**: Deterministic components (chunking, formatting)
✅ **Integration tests**: Full pipeline (RAG, API)
✅ **Benchmarks**: Standard (MMLU) + custom evaluation sets
✅ **Regression**: Snapshot testing for critical prompts
✅ **Monitoring**: Latency, errors, output quality
✅ **A/B testing**: Compare models or prompts

---

## Further Reading

- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **Pytest for ML**: Best practices
- **Evidently AI**: ML monitoring

---

**Status**: Complete | Module 9 (1/2) papers

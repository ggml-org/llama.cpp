# Tutorial: Testing LLM Inference Systems

## Introduction

This tutorial walks you through building a comprehensive test suite for llama.cpp inference systems, from unit tests to production monitoring.

**What You'll Build**: A complete testing framework with unit, integration, performance, and quality tests.

**Time Required**: 1-2 hours

## Step 1: Project Setup

Create test directory structure:

```bash
mkdir -p tests/{unit,integration,performance,quality}
touch tests/__init__.py

# Install dependencies
pip install pytest pytest-asyncio pytest-cov requests
```

## Step 2: Unit Tests - Tokenization

Create `tests/unit/test_tokenization.py`:

```python
"""Unit tests for tokenization"""
import pytest
from llama_cpp import Llama

@pytest.fixture(scope="module")
def model():
    """Load model once for all tests"""
    return Llama(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=512,
        verbose=False
    )

class TestTokenization:
    """Tokenization test suite"""

    def test_encode_decode_roundtrip(self, model):
        """Test that encoding and decoding preserves text"""
        text = "The quick brown fox jumps over the lazy dog"

        # Encode
        tokens = model.tokenize(text.encode('utf-8'))
        assert len(tokens) > 0

        # Decode
        decoded = model.detokenize(tokens).decode('utf-8')

        # Should preserve text (allowing whitespace differences)
        assert text.replace(" ", "") in decoded.replace(" ", "")

    def test_special_tokens(self, model):
        """Verify special tokens are defined"""
        bos = model.token_bos()
        eos = model.token_eos()

        assert bos >= 0, "BOS token should be non-negative"
        assert eos >= 0, "EOS token should be non-negative"
        assert bos != eos, "BOS and EOS should be different"

    def test_empty_string(self, model):
        """Handle empty input gracefully"""
        tokens = model.tokenize(b"")
        assert isinstance(tokens, list)

    def test_unicode_handling(self, model):
        """Handle Unicode text correctly"""
        unicode_text = "Hello 世界 🌍"
        tokens = model.tokenize(unicode_text.encode('utf-8'))

        assert len(tokens) > 0
        decoded = model.detokenize(tokens).decode('utf-8', errors='ignore')
        # Should contain at least some of the original text
        assert "Hello" in decoded
```

**Run**: `pytest tests/unit/test_tokenization.py -v`

## Step 3: Integration Tests - API Server

Create `tests/integration/test_server_api.py`:

```python
"""Integration tests for inference server"""
import pytest
import requests
import subprocess
import time

@pytest.fixture(scope="module")
def inference_server():
    """Start server for testing"""
    # Start server
    proc = subprocess.Popen([
        "./build/bin/llama-server",
        "-m", "models/llama-2-7b-chat.Q4_K_M.gguf",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--ctx-size", "2048"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for readiness
    url = "http://127.0.0.1:8080"
    for _ in range(30):
        try:
            requests.get(f"{url}/health", timeout=1)
            break
        except:
            time.sleep(1)

    yield url

    # Cleanup
    proc.terminate()
    proc.wait(timeout=10)

class TestServerAPI:
    """API endpoint tests"""

    def test_health_check(self, inference_server):
        """Server responds to health checks"""
        response = requests.get(f"{inference_server}/health")

        assert response.status_code == 200
        assert response.json()['status'] == 'ok'

    def test_basic_completion(self, inference_server):
        """Generate text completion"""
        response = requests.post(
            f"{inference_server}/completion",
            json={
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0.0
            },
            timeout=30
        )

        assert response.status_code == 200

        data = response.json()
        assert 'content' in data
        assert len(data['content']) > 0
        assert 'Paris' in data['content'] or 'paris' in data['content'].lower()

    def test_streaming_response(self, inference_server):
        """Test streaming completion"""
        response = requests.post(
            f"{inference_server}/completion",
            json={
                "prompt": "Count: one, two,",
                "max_tokens": 20,
                "stream": True
            },
            stream=True,
            timeout=30
        )

        assert response.status_code == 200

        chunks = []
        for line in response.iter_lines():
            if line:
                chunks.append(line)

        assert len(chunks) > 0

    def test_error_handling(self, inference_server):
        """Server handles errors gracefully"""
        # Missing required field
        response = requests.post(
            f"{inference_server}/completion",
            json={"max_tokens": 10},
            timeout=10
        )

        assert response.status_code in [400, 422]
```

**Run**: `pytest tests/integration/test_server_api.py -v`

## Step 4: Performance Tests

Create `tests/performance/test_benchmarks.py`:

```python
"""Performance benchmarks"""
import pytest
import requests
import time
import statistics
import json

def measure_latency(url, num_runs=10):
    """Measure request latency"""
    latencies = []

    for _ in range(num_runs):
        start = time.time()

        response = requests.post(
            f"{url}/completion",
            json={"prompt": "Test", "max_tokens": 10}
        )

        latencies.append(time.time() - start)

        assert response.status_code == 200

    return {
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'p95': sorted(latencies)[int(len(latencies) * 0.95)],
        'min': min(latencies),
        'max': max(latencies)
    }

class TestPerformance:
    """Performance test suite"""

    def test_latency_requirements(self, inference_server):
        """Verify latency meets requirements"""
        results = measure_latency(inference_server, num_runs=20)

        print(f"\nLatency Results:")
        print(f"  Mean: {results['mean']:.3f}s")
        print(f"  Median: {results['median']:.3f}s")
        print(f"  P95: {results['p95']:.3f}s")

        # Assert requirements
        assert results['mean'] < 5.0, "Mean latency too high"
        assert results['p95'] < 10.0, "P95 latency too high"

    def test_throughput(self, inference_server):
        """Measure request throughput"""
        num_requests = 50
        start = time.time()

        for _ in range(num_requests):
            response = requests.post(
                f"{inference_server}/completion",
                json={"prompt": "Hi", "max_tokens": 5}
            )
            assert response.status_code == 200

        duration = time.time() - start
        throughput = num_requests / duration

        print(f"\nThroughput: {throughput:.2f} req/s")

        assert throughput > 1.0, "Throughput too low"
```

## Step 5: Quality Tests

Create `tests/quality/test_output_quality.py`:

```python
"""Model output quality tests"""
import pytest
from llama_cpp import Llama

@pytest.fixture(scope="module")
def model():
    return Llama(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=2048
    )

class TestOutputQuality:
    """Quality assurance tests"""

    def test_factual_qa(self, model):
        """Test factual knowledge"""
        qa_pairs = [
            ("What is 2+2?", "4"),
            ("The capital of France is", "Paris"),
            ("The sun rises in the", "east"),
        ]

        for question, expected in qa_pairs:
            output = model(question, max_tokens=10, temperature=0.0)
            response = output['choices'][0]['text'].lower()

            assert expected.lower() in response, \
                f"Expected '{expected}' in response to '{question}'"

    def test_consistency(self, model):
        """Verify consistent outputs with temp=0"""
        prompt = "The capital of Spain is"

        outputs = [
            model(prompt, max_tokens=5, temperature=0.0, seed=42)['choices'][0]['text']
            for _ in range(3)
        ]

        # All outputs should be identical
        assert len(set(outputs)) == 1, "Outputs not consistent"

    def test_instruction_following(self, model):
        """Test ability to follow instructions"""
        prompt = """List exactly 3 programming languages:
1."""

        output = model(prompt, max_tokens=50, temperature=0.0)
        text = output['choices'][0]['text']

        # Should contain numbered list
        assert "2." in text or "3." in text
```

## Step 6: Automated Test Execution

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short

markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    quality: Quality tests
```

Create `run_tests.sh`:

```bash
#!/bin/bash

echo "Running Test Suite"
echo "=================="

# Unit tests (fast)
echo -e "\n[1/4] Unit Tests"
pytest tests/unit/ -v

# Integration tests
echo -e "\n[2/4] Integration Tests"
pytest tests/integration/ -v

# Performance tests
echo -e "\n[3/4] Performance Tests"
pytest tests/performance/ -v

# Quality tests
echo -e "\n[4/4] Quality Tests"
pytest tests/quality/ -v

# Coverage report
echo -e "\n[Coverage Report]"
pytest tests/ --cov=. --cov-report=term-missing

echo -e "\n✅ Test suite complete!"
```

Run all tests:

```bash
chmod +x run_tests.sh
./run_tests.sh
```

## Step 7: Continuous Integration

Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install pytest pytest-cov requests

    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Summary

You've built a complete test suite with:

- ✅ **Unit tests** for tokenization
- ✅ **Integration tests** for API endpoints
- ✅ **Performance tests** for latency/throughput
- ✅ **Quality tests** for model outputs
- ✅ **Automated execution** with scripts
- ✅ **CI integration** with GitHub Actions

## Next Steps

1. Add more test coverage
2. Implement regression detection
3. Add load testing
4. Set up test reporting dashboard
5. Create test documentation

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- Module 9 Documentation: Testing Strategies

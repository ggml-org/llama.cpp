# Lab 1: Building a Comprehensive Test Suite

## Objectives

By completing this lab, you will:
- ✅ Implement unit tests for llama.cpp inference code
- ✅ Create integration tests for the API server
- ✅ Build performance regression tests
- ✅ Set up automated test execution
- ✅ Generate test coverage reports

**Estimated Time**: 2-3 hours
**Difficulty**: Intermediate

## Prerequisites

- Completed Modules 1-6
- llama.cpp built from source
- Python 3.8+ installed
- Basic understanding of testing concepts

## Setup

### 1. Install Testing Dependencies

```bash
# Python testing tools
pip install pytest pytest-asyncio pytest-cov requests aiohttp

# C++ testing (Google Test)
cd llama.cpp
git submodule update --init --recursive
```

### 2. Verify Environment

```bash
# Check pytest installation
pytest --version

# Verify llama.cpp build
./build/bin/llama-server --help
```

## Part 1: Unit Tests

### Task 1.1: Test Tokenization

Create `tests/unit/test_tokenization.py`:

```python
import pytest
from llama_cpp import Llama

@pytest.fixture(scope="module")
def model():
    return Llama(
        model_path="models/test-model.gguf",
        n_ctx=512,
        verbose=False
    )

def test_basic_tokenization(model):
    """Test that text is tokenized correctly"""
    text = "Hello, world!"
    tokens = model.tokenize(text.encode('utf-8'))

    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)

def test_detokenization(model):
    """Test round-trip tokenization"""
    text = "The quick brown fox jumps over the lazy dog"
    tokens = model.tokenize(text.encode('utf-8'))
    detokenized = model.detokenize(tokens).decode('utf-8')

    # Should preserve the text (with possible whitespace changes)
    assert text.replace(" ", "") in detokenized.replace(" ", "")

def test_special_tokens(model):
    """Test special token handling"""
    bos = model.token_bos()
    eos = model.token_eos()

    assert bos >= 0
    assert eos >= 0
    assert bos != eos

# Add your tests here
def test_empty_string_tokenization(model):
    # TODO: Implement test for empty string
    pass

def test_unicode_tokenization(model):
    # TODO: Implement test for Unicode text
    pass
```

**✏️ Your Task**: Implement the two TODO test functions above.

### Task 1.2: Test Sampling Methods

Create `tests/unit/test_sampling.py`:

```python
import pytest
from llama_cpp import Llama

@pytest.fixture(scope="module")
def model():
    return Llama(model_path="models/test-model.gguf", n_ctx=512, verbose=False)

def test_deterministic_sampling(model):
    """Test that temperature=0 is deterministic"""
    prompt = "The capital of France is"

    output1 = model(prompt, max_tokens=5, temperature=0.0, seed=42)
    output2 = model(prompt, max_tokens=5, temperature=0.0, seed=42)

    assert output1['choices'][0]['text'] == output2['choices'][0]['text']

# TODO: Add more sampling tests
# - Test temperature effect on randomness
# - Test top_k filtering
# - Test top_p (nucleus) sampling
# - Test repetition penalty
```

**✏️ Your Task**: Implement at least 3 additional sampling tests.

### Verification 1

Run your unit tests:

```bash
pytest tests/unit/ -v

# Expected output:
# tests/unit/test_tokenization.py::test_basic_tokenization PASSED
# tests/unit/test_tokenization.py::test_detokenization PASSED
# tests/unit/test_sampling.py::test_deterministic_sampling PASSED
# ...
```

## Part 2: Integration Tests

### Task 2.1: Test Server Endpoints

Create `tests/integration/test_server.py`:

```python
import pytest
import requests
import time
import subprocess
import os

@pytest.fixture(scope="module")
def server():
    """Start llama-server for testing"""
    # Start server in background
    process = subprocess.Popen([
        "./build/bin/llama-server",
        "--model", "models/test-model.gguf",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--ctx-size", "2048"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for server to start
    max_retries = 10
    for i in range(max_retries):
        try:
            requests.get("http://127.0.0.1:8080/health", timeout=1)
            break
        except requests.ConnectionError:
            time.sleep(1)

    yield "http://127.0.0.1:8080"

    # Cleanup
    process.terminate()
    process.wait()

def test_health_endpoint(server):
    """Test health check"""
    response = requests.get(f"{server}/health")
    assert response.status_code == 200

def test_completion_endpoint(server):
    """Test basic completion"""
    response = requests.post(
        f"{server}/completion",
        json={
            "prompt": "Hello",
            "max_tokens": 10
        },
        timeout=30
    )

    assert response.status_code == 200
    data = response.json()
    assert 'content' in data
    assert len(data['content']) > 0

# TODO: Add more integration tests
# - Test streaming responses
# - Test error handling
# - Test concurrent requests
# - Test chat endpoint
```

**✏️ Your Task**: Implement the TODO integration tests.

### Task 2.2: End-to-End Workflow Test

Create `tests/integration/test_e2e_workflow.py`:

```python
def test_complete_inference_workflow(server):
    """Test complete workflow from prompt to response"""
    # 1. Submit request
    response = requests.post(
        f"{server}/completion",
        json={
            "prompt": "Explain quantum computing in one sentence:",
            "max_tokens": 50,
            "temperature": 0.7
        }
    )

    assert response.status_code == 200
    result = response.json()

    # 2. Verify response structure
    assert 'content' in result
    assert 'tokens_predicted' in result
    assert 'tokens_evaluated' in result

    # 3. Verify content quality
    content = result['content']
    assert len(content) > 20  # Should be substantial
    assert result['tokens_predicted'] > 0

    print(f"\nGenerated: {content}")
```

### Verification 2

```bash
pytest tests/integration/ -v -s

# Should see server start, tests run, server stop
```

## Part 3: Performance Tests

### Task 3.1: Benchmark Framework

Create `tests/performance/test_benchmarks.py`:

```python
import pytest
import requests
import time
import statistics

def benchmark_latency(server, num_runs=10):
    """Benchmark request latency"""
    latencies = []

    for i in range(num_runs):
        start = time.time()
        response = requests.post(
            f"{server}/completion",
            json={"prompt": "Test", "max_tokens": 10}
        )
        end = time.time()

        assert response.status_code == 200
        latencies.append(end - start)

    return {
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'p95': sorted(latencies)[int(len(latencies) * 0.95)],
        'min': min(latencies),
        'max': max(latencies)
    }

def test_latency_benchmarks(server):
    """Test that latency is acceptable"""
    results = benchmark_latency(server, num_runs=20)

    print(f"\nLatency Benchmarks:")
    print(f"  Mean: {results['mean']:.3f}s")
    print(f"  Median: {results['median']:.3f}s")
    print(f"  P95: {results['p95']:.3f}s")
    print(f"  Range: {results['min']:.3f}s - {results['max']:.3f}s")

    # Assert performance requirements
    assert results['p95'] < 10.0, f"P95 latency too high: {results['p95']:.3f}s"
    assert results['mean'] < 5.0, f"Mean latency too high: {results['mean']:.3f}s"

# TODO: Add more performance tests
# - Throughput test (requests per second)
# - Concurrent request handling
# - Memory usage under load
# - Long-running inference test
```

**✏️ Your Task**: Implement the TODO performance tests.

### Task 3.2: Regression Detection

Create `tests/performance/test_regression.py`:

```python
import json
from pathlib import Path

def test_performance_regression():
    """Check for performance regressions"""
    baseline_file = Path("tests/performance/baseline.json")

    # Load baseline (if exists)
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)

        # Run current benchmarks
        current = benchmark_latency("http://127.0.0.1:8080", num_runs=20)

        # Check for regression (>5% slower)
        threshold = 0.05
        for metric in ['mean', 'median', 'p95']:
            change = (current[metric] - baseline[metric]) / baseline[metric]

            print(f"{metric}: {baseline[metric]:.3f}s -> {current[metric]:.3f}s ({change*100:+.1f}%)")

            assert change <= threshold, \
                f"Performance regression detected: {metric} increased by {change*100:.1f}%"

    else:
        print("No baseline found, creating new baseline")
        current = benchmark_latency("http://127.0.0.1:8080", num_runs=20)

        with open(baseline_file, 'w') as f:
            json.dump(current, f, indent=2)
```

### Verification 3

```bash
pytest tests/performance/ -v -s

# Should show benchmark results and regression checks
```

## Part 4: Test Coverage

### Task 4.1: Generate Coverage Report

```bash
# Run tests with coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

**✏️ Your Task**: Aim for >80% test coverage. Identify untested code and add tests.

### Task 4.2: Coverage Analysis

1. Review the coverage report
2. Identify critical untested code paths
3. Add tests for uncovered areas
4. Re-run coverage analysis

## Part 5: Automated Test Execution

### Task 5.1: Create Test Configuration

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings

markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests

# Coverage settings
[coverage:run]
source = .
omit =
    tests/*
    venv/*
    */site-packages/*
```

### Task 5.2: Create Test Script

Create `run_tests.sh`:

```bash
#!/bin/bash

set -e

echo "Running LLaMA Inference Test Suite"
echo "=================================="

# Unit tests
echo ""
echo "1. Running unit tests..."
pytest tests/unit/ -m "not slow" --tb=short

# Integration tests
echo ""
echo "2. Running integration tests..."
pytest tests/integration/ --tb=short

# Performance tests
echo ""
echo "3. Running performance tests..."
pytest tests/performance/ --tb=short

# Coverage report
echo ""
echo "4. Generating coverage report..."
pytest tests/ --cov=. --cov-report=term --cov-report=html

echo ""
echo "✅ All tests passed!"
echo "📊 Coverage report: htmlcov/index.html"
```

### Verification 4

```bash
chmod +x run_tests.sh
./run_tests.sh
```

## Deliverables

At the end of this lab, you should have:

1. ✅ **Unit test suite** with ≥10 tests
2. ✅ **Integration test suite** with ≥5 tests
3. ✅ **Performance benchmarks** with regression detection
4. ✅ **Coverage report** showing ≥80% coverage
5. ✅ **Automated test script** that runs all tests

## Challenge Tasks

For extra practice:

1. **Property-based Testing**: Use `hypothesis` library for property-based tests
2. **Fuzzing**: Implement fuzzing tests for input validation
3. **Stress Testing**: Create tests that push the system to its limits
4. **Mock Testing**: Add tests using mocks for external dependencies

## Submission

Create a summary report:

```markdown
# Test Suite Report

## Coverage
- Unit Tests: X tests, Y% coverage
- Integration Tests: X tests
- Performance Tests: X benchmarks

## Results
- Total Tests: X
- Passed: X
- Failed: X
- Coverage: X%

## Performance Baselines
- Mean Latency: Xs
- P95 Latency: Xs
- Throughput: X req/s

## Key Findings
[Your observations about test coverage, performance, etc.]
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [Google Test documentation](https://google.github.io/googletest/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**Next**: Proceed to Lab 2: CI/CD Pipeline Setup

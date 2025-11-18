"""
Comprehensive test suite for llama.cpp inference system
Covers unit tests, integration tests, and performance tests
"""

import pytest
import requests
import asyncio
import aiohttp
import time
import statistics
from pathlib import Path
from typing import List, Dict
import json

# ============================================================================
# Configuration and Fixtures
# ============================================================================

TEST_MODEL_PATH = "models/test-model.gguf"
SERVER_URL = "http://localhost:8080"

@pytest.fixture(scope="session")
def test_model():
    """Ensure test model exists"""
    model_path = Path(TEST_MODEL_PATH)
    if not model_path.exists():
        pytest.skip(f"Test model not found at {TEST_MODEL_PATH}")
    return str(model_path)

@pytest.fixture(scope="session")
def server_url():
    """Check if server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            return SERVER_URL
    except requests.ConnectionError:
        pass

    pytest.skip("Server not running. Start with: ./llama-server -m models/test-model.gguf")

# ============================================================================
# Unit Tests
# ============================================================================

class TestTokenization:
    """Test tokenization functionality"""

    def test_basic_tokenization(self):
        """Test basic text tokenization"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)
        text = "Hello, world!"

        tokens = model.tokenize(text.encode('utf-8'))

        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_detokenization(self):
        """Test token to text conversion"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)
        text = "The capital of France is Paris."

        tokens = model.tokenize(text.encode('utf-8'))
        detokenized = model.detokenize(tokens).decode('utf-8')

        # Detokenization should preserve the text (with possible whitespace differences)
        assert text.replace(" ", "") in detokenized.replace(" ", "")

    def test_special_tokens(self):
        """Test special token handling"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH)

        bos = model.token_bos()
        eos = model.token_eos()

        assert bos >= 0
        assert eos >= 0
        assert bos != eos

class TestSampling:
    """Test sampling methods"""

    def test_deterministic_sampling(self):
        """Test that temperature=0 produces deterministic results"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)
        prompt = "The capital of France is"

        # Generate twice with same seed
        output1 = model(prompt, max_tokens=10, temperature=0.0, seed=42)
        output2 = model(prompt, max_tokens=10, temperature=0.0, seed=42)

        assert output1['choices'][0]['text'] == output2['choices'][0]['text']

    def test_temperature_effect(self):
        """Test that temperature affects randomness"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)
        prompt = "Once upon a time"

        # Low temperature should be more deterministic
        low_temp_outputs = [
            model(prompt, max_tokens=20, temperature=0.1, seed=i)['choices'][0]['text']
            for i in range(5)
        ]

        # High temperature should be more random
        high_temp_outputs = [
            model(prompt, max_tokens=20, temperature=1.5, seed=i)['choices'][0]['text']
            for i in range(5)
        ]

        # Low temperature outputs should be more similar
        low_temp_unique = len(set(low_temp_outputs))
        high_temp_unique = len(set(high_temp_outputs))

        assert high_temp_unique >= low_temp_unique

    def test_max_tokens_limit(self):
        """Test that max_tokens is respected"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)
        prompt = "Count to one hundred:"

        output = model(prompt, max_tokens=10, temperature=0.0)
        tokens = model.tokenize(output['choices'][0]['text'].encode())

        assert len(tokens) <= 10

# ============================================================================
# Integration Tests
# ============================================================================

class TestServerAPI:
    """Test server API endpoints"""

    def test_health_endpoint(self, server_url):
        """Test health check endpoint"""
        response = requests.get(f"{server_url}/health")

        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] in ['ok', 'ready']

    def test_completion_endpoint(self, server_url):
        """Test completion endpoint"""
        response = requests.post(
            f"{server_url}/completion",
            json={
                "prompt": "The capital of France is",
                "max_tokens": 20,
                "temperature": 0.0
            },
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        assert 'content' in data
        assert len(data['content']) > 0
        assert 'tokens_predicted' in data
        assert data['tokens_predicted'] <= 20

    def test_chat_endpoint(self, server_url):
        """Test chat completions endpoint"""
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "max_tokens": 50,
                "temperature": 0.0
            },
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        assert 'choices' in data
        assert len(data['choices']) > 0
        assert 'message' in data['choices'][0]
        assert 'content' in data['choices'][0]['message']

    def test_streaming_completion(self, server_url):
        """Test streaming completion"""
        response = requests.post(
            f"{server_url}/completion",
            json={
                "prompt": "Count to five:",
                "max_tokens": 30,
                "stream": True
            },
            stream=True,
            timeout=30
        )

        assert response.status_code == 200

        chunks = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8').replace('data: ', ''))
                    if 'content' in data:
                        chunks.append(data['content'])
                except json.JSONDecodeError:
                    pass

        assert len(chunks) > 0
        full_text = ''.join(chunks)
        assert len(full_text) > 0

    def test_error_handling(self, server_url):
        """Test error handling"""
        # Test missing required field
        response = requests.post(
            f"{server_url}/completion",
            json={"max_tokens": 10},  # Missing prompt
            timeout=10
        )

        assert response.status_code in [400, 422]  # Bad request

        # Test invalid parameters
        response = requests.post(
            f"{server_url}/completion",
            json={
                "prompt": "Test",
                "max_tokens": -1  # Invalid
            },
            timeout=10
        )

        assert response.status_code in [400, 422]

# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and load testing"""

    def test_latency_bounds(self, server_url):
        """Test that latency is within acceptable bounds"""
        latencies = []

        for _ in range(10):
            start = time.time()
            response = requests.post(
                f"{server_url}/completion",
                json={
                    "prompt": "Hello",
                    "max_tokens": 10,
                    "temperature": 0.0
                },
                timeout=30
            )
            end = time.time()

            assert response.status_code == 200
            latencies.append(end - start)

        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nAverage latency: {avg_latency:.3f}s")
        print(f"P95 latency: {p95_latency:.3f}s")

        # Assertions (adjust based on your requirements)
        assert avg_latency < 5.0, f"Average latency too high: {avg_latency}s"
        assert p95_latency < 10.0, f"P95 latency too high: {p95_latency}s"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, server_url):
        """Test handling concurrent requests"""
        async def make_request(session, i):
            async with session.post(
                f"{server_url}/completion",
                json={
                    "prompt": f"Request {i}:",
                    "max_tokens": 10
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                return response.status

        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, i) for i in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should succeed
        assert all(r == 200 for r in results if not isinstance(r, Exception))

        # Count successes
        successes = sum(1 for r in results if r == 200)
        print(f"\n{successes}/10 concurrent requests succeeded")

        assert successes >= 8, f"Too many failed requests: {10 - successes}"

    def test_throughput(self, server_url):
        """Test request throughput"""
        num_requests = 20
        start = time.time()

        for i in range(num_requests):
            response = requests.post(
                f"{server_url}/completion",
                json={
                    "prompt": "Test",
                    "max_tokens": 5,
                    "temperature": 0.0
                },
                timeout=30
            )
            assert response.status_code == 200

        end = time.time()
        duration = end - start
        throughput = num_requests / duration

        print(f"\nThroughput: {throughput:.2f} requests/second")

        assert throughput > 0.5, f"Throughput too low: {throughput} req/s"

# ============================================================================
# Quality Tests
# ============================================================================

class TestOutputQuality:
    """Test model output quality"""

    def test_factual_knowledge(self):
        """Test basic factual knowledge"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)

        qa_pairs = [
            ("What is 2+2?", "4"),
            ("The capital of France is", "Paris"),
        ]

        for question, expected in qa_pairs:
            output = model(question, max_tokens=10, temperature=0.0)
            response = output['choices'][0]['text'].lower()

            assert expected.lower() in response, \
                f"Expected '{expected}' in response to '{question}', got: {response}"

    def test_instruction_following(self):
        """Test instruction following"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)

        prompt = "List exactly three colors:\n1."
        output = model(prompt, max_tokens=50, temperature=0.0)
        text = output['choices'][0]['text']

        # Should contain numbered items
        assert "2." in text or "3." in text

    def test_consistency(self):
        """Test output consistency with temperature=0"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)

        outputs = []
        for _ in range(3):
            output = model(
                "The quick brown fox",
                max_tokens=10,
                temperature=0.0,
                seed=42
            )
            outputs.append(output['choices'][0]['text'])

        # All outputs should be identical
        assert len(set(outputs)) == 1, "Outputs should be consistent with temp=0"

# ============================================================================
# Stress Tests
# ============================================================================

class TestStress:
    """Stress and edge case testing"""

    def test_long_prompt(self):
        """Test handling of long prompts"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=2048, verbose=False)

        # Create a long prompt (but within context)
        long_prompt = "word " * 500

        try:
            output = model(long_prompt, max_tokens=10, temperature=0.0)
            assert output is not None
        except Exception as e:
            # Should handle gracefully if too long
            assert "context" in str(e).lower() or "memory" in str(e).lower()

    def test_empty_prompt(self, server_url):
        """Test handling of empty prompt"""
        response = requests.post(
            f"{server_url}/completion",
            json={
                "prompt": "",
                "max_tokens": 10
            },
            timeout=10
        )

        # Should either reject or handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_special_characters(self):
        """Test handling of special characters"""
        from llama_cpp import Llama

        model = Llama(model_path=TEST_MODEL_PATH, n_ctx=512, verbose=False)

        special_prompts = [
            "Hello 你好 مرحبا",  # Unicode
            "Test\n\nNew line",  # Newlines
            "Special: @#$%^&*()",  # Symbols
        ]

        for prompt in special_prompts:
            try:
                output = model(prompt, max_tokens=10, temperature=0.0)
                assert output is not None
            except Exception as e:
                pytest.fail(f"Failed on prompt '{prompt}': {e}")

# ============================================================================
# Test Report
# ============================================================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate test summary report"""
    print("\n" + "="*70)
    print("Test Summary Report")
    print("="*70)

    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {passed + failed + skipped}")

    if failed == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {failed} test(s) failed")

    print("="*70)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

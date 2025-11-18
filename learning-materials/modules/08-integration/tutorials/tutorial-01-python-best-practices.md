# Tutorial: Python Best Practices for llama.cpp Integration

**Estimated Time**: 60 minutes
**Level**: Intermediate

## Overview

Learn production-ready patterns and best practices for integrating llama.cpp into Python applications.

## 1. Project Structure

### Recommended Layout

```
your-app/
├── src/
│   ├── __init__.py
│   ├── models/          # Model management
│   ├── services/        # Business logic
│   ├── api/             # API endpoints
│   └── utils/           # Utilities
├── tests/
├── config/
├── requirements.txt
├── setup.py
└── README.md
```

## 2. Configuration Management

### Use Environment Variables

```python
# config/settings.py
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    model_path: str = Field(..., env='MODEL_PATH')
    n_ctx: int = Field(2048, env='N_CTX')
    n_gpu_layers: int = Field(0, env='N_GPU_LAYERS')

    class Config:
        env_file = '.env'

settings = Settings()
```

### Configuration Factory Pattern

```python
# config/model_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_path: str
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0

    @classmethod
    def production(cls, model_path: str) -> 'ModelConfig':
        return cls(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35
        )

    @classmethod
    def development(cls, model_path: str) -> 'ModelConfig':
        return cls(
            model_path=model_path,
            n_ctx=512,
            n_threads=2,
            n_gpu_layers=0
        )
```

## 3. Model Management

### Singleton Pattern

```python
# models/llama_singleton.py
from llama_cpp import Llama
from threading import Lock

class LlamaSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, model_path: str, **kwargs):
        if not hasattr(self, 'llm'):
            self.llm = Llama(model_path=model_path, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm(*args, **kwargs)
```

### Context Manager Pattern

```python
# models/llama_context.py
from contextlib import contextmanager
from llama_cpp import Llama

@contextmanager
def llama_context(model_path: str, **kwargs):
    """Context manager for automatic cleanup."""
    llm = Llama(model_path=model_path, **kwargs)
    try:
        yield llm
    finally:
        del llm

# Usage
with llama_context("model.gguf") as llm:
    result = llm("Hello")
```

## 4. Error Handling

### Robust Error Handling

```python
# utils/error_handling.py
import logging
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model errors."""
    pass

class ModelLoadError(ModelError):
    """Error loading model."""
    pass

class GenerationError(ModelError):
    """Error during generation."""
    pass

def handle_model_errors(func: Callable) -> Callable:
    """Decorator for model error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise ModelLoadError(f"Model file not found: {e}")
        except MemoryError as e:
            logger.error(f"Out of memory: {e}")
            raise ModelError(f"Insufficient memory: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise GenerationError(f"Generation failed: {e}")
    return wrapper

# Usage
@handle_model_errors
def load_model(path: str):
    return Llama(model_path=path)
```

## 5. Async Support

### Async Wrapper

```python
# services/async_llama.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama

class AsyncLlama:
    def __init__(self, model_path: str, **kwargs):
        self.llm = Llama(model_path=model_path, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def generate_async(self, prompt: str, **kwargs):
        """Async generation using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.llm(prompt, **kwargs)
        )

    async def stream_async(self, prompt: str, **kwargs):
        """Async streaming generation."""
        loop = asyncio.get_event_loop()

        def generate_tokens():
            for output in self.llm(prompt, stream=True, **kwargs):
                yield output['choices'][0]['text']

        # Stream tokens asynchronously
        for token in generate_tokens():
            await asyncio.sleep(0)  # Yield control
            yield token
```

## 6. Logging

### Structured Logging

```python
# utils/logging_config.py
import logging
import sys
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# Usage in application
import logging

logger = logging.getLogger(__name__)

logger.info("Loading model", extra={
    "model_path": model_path,
    "n_ctx": n_ctx
})
```

## 7. Testing

### Unit Testing

```python
# tests/test_llama_service.py
import pytest
from unittest.mock import Mock, patch
from services.llama_service import LlamaService

@pytest.fixture
def mock_llama():
    with patch('llama_cpp.Llama') as mock:
        mock.return_value.return_value = {
            'choices': [{'text': 'test response'}]
        }
        yield mock

def test_generate(mock_llama):
    service = LlamaService(model_path="test.gguf")
    result = service.generate("test prompt")

    assert result == 'test response'
    mock_llama.return_value.assert_called_once()

def test_error_handling(mock_llama):
    mock_llama.side_effect = FileNotFoundError()

    with pytest.raises(ModelLoadError):
        LlamaService(model_path="nonexistent.gguf")
```

## 8. Performance Monitoring

### Metrics Collection

```python
# utils/metrics.py
import time
from functools import wraps
from dataclasses import dataclass
from typing import Callable

@dataclass
class GenerationMetrics:
    duration_ms: float
    tokens_generated: int
    tokens_per_second: float

def measure_generation(func: Callable) -> Callable:
    """Decorator to measure generation performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start) * 1000

        # Estimate tokens (simple approximation)
        tokens = len(result.split())

        metrics = GenerationMetrics(
            duration_ms=duration,
            tokens_generated=tokens,
            tokens_per_second=tokens / (duration / 1000)
        )

        logger.info(f"Generation metrics: {metrics}")
        return result

    return wrapper
```

## 9. Caching

### Response Caching

```python
# utils/cache.py
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path

class DiskCache:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, key: str):
        """Get cached value."""
        cache_file = self.cache_dir / self._hash_key(key)
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, key: str, value):
        """Set cached value."""
        cache_file = self.cache_dir / self._hash_key(key)
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

# Usage with decorator
def cached_generation(cache: DiskCache):
    def decorator(func):
        @wraps(func)
        def wrapper(prompt: str, *args, **kwargs):
            cached = cache.get(prompt)
            if cached:
                return cached

            result = func(prompt, *args, **kwargs)
            cache.set(prompt, result)
            return result
        return wrapper
    return decorator
```

## 10. Production Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] Logging properly set up
- [ ] Error handling comprehensive
- [ ] Tests passing (>80% coverage)
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Security review done
- [ ] Resource limits configured
- [ ] Monitoring and alerting set up
- [ ] Graceful shutdown implemented

### Code Quality

```bash
# Run linters
black src/
flake8 src/
mypy src/

# Run tests
pytest tests/ --cov=src --cov-report=html

# Security check
bandit -r src/
```

## Summary

Key takeaways:
- Use proper project structure
- Implement robust error handling
- Add comprehensive logging
- Write tests for all components
- Monitor performance
- Cache where appropriate
- Follow Python best practices

---

**Tutorial**: 01 - Python Best Practices
**Module**: 08 - Integration & Applications
**Version**: 1.0

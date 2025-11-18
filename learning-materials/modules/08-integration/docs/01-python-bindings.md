# Python Bindings Deep Dive

**Module 8, Lesson 1**
**Estimated Time**: 3 hours
**Difficulty**: Intermediate

## Overview

This lesson explores llama-cpp-python, the official Python binding for llama.cpp, enabling seamless integration of high-performance LLM inference into Python applications.

## Learning Objectives

By the end of this lesson, you will be able to:
- Install and configure llama-cpp-python with various backends
- Use the high-level and low-level Python APIs
- Implement streaming inference and chat completions
- Optimize Python bindings for production use
- Debug common Python binding issues

## Prerequisites

- Module 1: Foundations (llama.cpp basics)
- Python 3.8+ with pip
- Understanding of Python async programming (helpful)
- Basic knowledge of GPU acceleration (for GPU support)

---

## 1. Introduction to llama-cpp-python

### What is llama-cpp-python?

llama-cpp-python is the official Python binding for llama.cpp, providing:
- **High-level API**: OpenAI-compatible interface
- **Low-level API**: Direct access to llama.cpp functions
- **Streaming support**: Token-by-token generation
- **Multiple backends**: CPU, CUDA, Metal, OpenCL
- **Type hints**: Full typing support for better IDE integration

### Architecture

```
┌─────────────────────────────────────┐
│   Python Application                │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  llama-cpp-python (High API) │  │
│  │  - Llama class               │  │
│  │  - ChatCompletion            │  │
│  │  - Embedding                 │  │
│  └────────────┬─────────────────┘  │
│               │                     │
│  ┌────────────▼─────────────────┐  │
│  │  llama-cpp-python (Low API)  │  │
│  │  - ctypes bindings           │  │
│  │  - Direct C function calls   │  │
│  └────────────┬─────────────────┘  │
└───────────────┼─────────────────────┘
                │
┌───────────────▼─────────────────────┐
│   llama.cpp (C/C++)                 │
│   - Model loading                   │
│   - Tokenization                    │
│   - Inference engine                │
│   - CUDA/Metal backends             │
└─────────────────────────────────────┘
```

---

## 2. Installation and Setup

### Basic Installation

```bash
# CPU-only (basic installation)
pip install llama-cpp-python

# Verify installation
python -c "from llama_cpp import Llama; print('Success!')"
```

### GPU-Accelerated Installation

#### CUDA (NVIDIA GPUs)

```bash
# Set CUDA compilation flags
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# For specific CUDA architectures
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;86" pip install llama-cpp-python
```

#### Metal (Apple Silicon)

```bash
# Metal is automatically enabled on macOS
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

#### OpenCL (AMD/Intel GPUs)

```bash
CMAKE_ARGS="-DGGML_CLBLAST=on" pip install llama-cpp-python
```

### Advanced Build Options

```bash
# Enable all optimizations
CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on -DGGML_CUBLAS=on" \
  pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Build with specific features
CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_CURL=on -DLLAMA_METAL=on" \
  pip install llama-cpp-python
```

### Installation Verification

```python
#!/usr/bin/env python3
"""Verify llama-cpp-python installation and capabilities."""

from llama_cpp import Llama
import llama_cpp

# Check version
print(f"llama-cpp-python version: {llama_cpp.__version__}")

# Check backend support
backends = []
if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
    if llama_cpp.llama_supports_gpu_offload():
        backends.append("GPU offload")

print(f"Available backends: {', '.join(backends) if backends else 'CPU only'}")

# Test basic model loading (requires a model file)
# llm = Llama(model_path="./models/model.gguf", n_ctx=512, n_gpu_layers=0)
# print("Model loaded successfully!")
```

---

## 3. High-Level API

### The Llama Class

The primary interface for model interaction:

```python
from llama_cpp import Llama

# Initialize model
llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,           # Context window
    n_threads=8,          # CPU threads
    n_gpu_layers=35,      # GPU layers (0 for CPU-only)
    verbose=False         # Disable verbose output
)

# Basic text generation
output = llm(
    "Q: What is the capital of France? A:",
    max_tokens=32,
    temperature=0.7,
    top_p=0.9,
    echo=True             # Include prompt in output
)

print(output['choices'][0]['text'])
```

### Streaming Generation

Real-time token-by-token generation:

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/model.gguf")

# Streaming generator
stream = llm(
    "Write a poem about recursion:",
    max_tokens=200,
    stream=True
)

# Process tokens as they arrive
for output in stream:
    token = output['choices'][0]['text']
    print(token, end='', flush=True)

print()  # Newline after completion
```

### Chat Completions API

OpenAI-compatible chat interface:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    chat_format="llama-2"  # Use Llama-2 chat template
)

# Chat completion
response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms."
        }
    ],
    temperature=0.7,
    max_tokens=200
)

print(response['choices'][0]['message']['content'])
```

### Streaming Chat Completions

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/model.gguf",
    chat_format="chatml"
)

# Streaming chat
stream = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Tell me a story about a robot."}
    ],
    stream=True
)

for chunk in stream:
    if 'content' in chunk['choices'][0]['delta']:
        print(chunk['choices'][0]['delta']['content'], end='', flush=True)
```

---

## 4. Low-Level API

### Direct C Function Access

For advanced use cases requiring fine-grained control:

```python
from llama_cpp import llama_cpp, Llama
import ctypes

# Load model
llm = Llama(model_path="./models/model.gguf")

# Access context (C pointer)
ctx = llm.ctx

# Get context size
n_ctx = llama_cpp.llama_n_ctx(ctx)
print(f"Context size: {n_ctx}")

# Get vocabulary size
n_vocab = llama_cpp.llama_n_vocab(llm.model)
print(f"Vocabulary size: {n_vocab}")

# Tokenize text
text = b"Hello, world!"
tokens = (llama_cpp.llama_token * (len(text) + 1))()
n_tokens = llama_cpp.llama_tokenize(
    llm.model,
    text,
    len(text),
    tokens,
    len(tokens),
    True,  # add_bos
    False  # special
)

print(f"Tokens: {[tokens[i] for i in range(n_tokens)]}")
```

### Custom Tokenization

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/model.gguf")

# Tokenize
text = "The quick brown fox"
tokens = llm.tokenize(text.encode('utf-8'))
print(f"Tokens: {tokens}")

# Detokenize
decoded = llm.detokenize(tokens)
print(f"Decoded: {decoded.decode('utf-8')}")

# Token to piece (get text representation)
for token in tokens:
    piece = llm.model.token_to_piece(token)
    print(f"Token {token}: '{piece.decode('utf-8')}'")
```

### Manual Inference Loop

```python
from llama_cpp import Llama, llama_cpp

llm = Llama(model_path="./models/model.gguf", n_ctx=512)

# Tokenize prompt
prompt = "The meaning of life is"
tokens = llm.tokenize(prompt.encode('utf-8'))

# Reset context
llm.reset()

# Evaluate prompt
llm.eval(tokens)

# Generate tokens one by one
max_tokens = 50
for _ in range(max_tokens):
    # Get logits
    logits = llm.logits()

    # Sample next token
    next_token = llm.sample(
        top_k=40,
        top_p=0.95,
        temp=0.8
    )

    # Stop on EOS
    if next_token == llm.token_eos():
        break

    # Decode and print
    text = llm.detokenize([next_token]).decode('utf-8', errors='ignore')
    print(text, end='', flush=True)

    # Evaluate next token
    llm.eval([next_token])

print()
```

---

## 5. Advanced Features

### Embeddings Generation

```python
from llama_cpp import Llama

# Load model for embeddings
llm = Llama(
    model_path="./models/model.gguf",
    embedding=True  # Enable embedding mode
)

# Generate embedding
text = "This is a sample text for embedding."
embedding = llm.create_embedding(text)

# Access embedding vector
vector = embedding['data'][0]['embedding']
print(f"Embedding dimension: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
```

### Logit Bias and Token Control

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/model.gguf")

# Bias against certain tokens
logit_bias = {
    llm.tokenize(b"hate")[0]: -100.0,  # Suppress "hate"
    llm.tokenize(b"love")[0]: 10.0,    # Boost "love"
}

output = llm(
    "I",
    max_tokens=50,
    logit_bias=logit_bias
)

print(output['choices'][0]['text'])
```

### Grammar-Constrained Generation

```python
from llama_cpp import Llama, LlamaGrammar

# Define JSON grammar
json_grammar = LlamaGrammar.from_string(r'''
root ::= object
object ::= "{" pair ("," pair)* "}"
pair ::= string ":" value
value ::= "true" | "false" | "null" | number | string | array | object
string ::= "\"" [^"]* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
array ::= "[" (value ("," value)*)? "]"
''')

llm = Llama(model_path="./models/model.gguf")

# Generate JSON
output = llm(
    "Generate a JSON object with name and age: ",
    max_tokens=100,
    grammar=json_grammar
)

print(output['choices'][0]['text'])
```

### Function Calling

```python
from llama_cpp import Llama
import json

llm = Llama(
    model_path="./models/functionary-7b.gguf",
    chat_format="functionary"
)

# Define functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Chat with function calling
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Check for function call
message = response['choices'][0]['message']
if 'tool_calls' in message:
    for tool_call in message['tool_calls']:
        print(f"Function: {tool_call['function']['name']}")
        print(f"Arguments: {tool_call['function']['arguments']}")
```

---

## 6. Performance Optimization

### GPU Layer Offloading

```python
from llama_cpp import Llama

# Optimize GPU layer distribution
llm = Llama(
    model_path="./models/llama-2-13b.Q4_K_M.gguf",
    n_gpu_layers=40,      # Offload 40 layers to GPU
    n_batch=512,          # Batch size for prompt processing
    n_threads=6,          # CPU threads for remaining layers
    n_ctx=4096            # Large context window
)
```

### Batched Inference

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/model.gguf")

prompts = [
    "Translate to French: Hello",
    "Translate to French: Goodbye",
    "Translate to French: Thank you"
]

# Process in batch (implemented via iteration)
results = []
for prompt in prompts:
    output = llm(prompt, max_tokens=20)
    results.append(output['choices'][0]['text'])

for prompt, result in zip(prompts, results):
    print(f"{prompt} -> {result}")
```

### Memory Management

```python
from llama_cpp import Llama

# Memory-efficient configuration
llm = Llama(
    model_path="./models/model.gguf",
    n_ctx=2048,           # Moderate context
    n_batch=256,          # Smaller batch size
    use_mmap=True,        # Memory-map model file (default)
    use_mlock=True,       # Lock model in RAM (prevents swapping)
    low_vram=True         # Optimize for low VRAM (if using GPU)
)
```

### Caching and Reuse

```python
from llama_cpp import Llama

# Initialize once, reuse many times
llm = Llama(model_path="./models/model.gguf")

# Save state for later
state = llm.save_state()

# Generate something
llm("First generation", max_tokens=50)

# Restore to original state
llm.load_state(state)

# Generate again from same state
llm("Second generation", max_tokens=50)
```

---

## 7. Error Handling and Debugging

### Common Issues and Solutions

#### Issue: Model Loading Fails

```python
from llama_cpp import Llama

try:
    llm = Llama(
        model_path="./models/model.gguf",
        verbose=True  # Enable verbose logging
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Possible causes:")
    print("- Incorrect model path")
    print("- Corrupted model file")
    print("- Insufficient memory")
    print("- Incompatible GGUF version")
```

#### Issue: GPU Out of Memory

```python
from llama_cpp import Llama

# Start with fewer GPU layers
n_gpu_layers = 35

while n_gpu_layers > 0:
    try:
        llm = Llama(
            model_path="./models/model.gguf",
            n_gpu_layers=n_gpu_layers
        )
        print(f"Successfully loaded with {n_gpu_layers} GPU layers")
        break
    except Exception as e:
        if "out of memory" in str(e).lower():
            n_gpu_layers -= 5
            print(f"Reducing to {n_gpu_layers} GPU layers...")
        else:
            raise
```

#### Issue: Slow Inference

```python
from llama_cpp import Llama
import time

llm = Llama(
    model_path="./models/model.gguf",
    n_threads=8,    # Increase CPU threads
    n_batch=512,    # Increase batch size
    n_gpu_layers=0  # Try GPU offloading
)

# Benchmark
prompt = "Explain Python decorators."
start = time.time()
output = llm(prompt, max_tokens=100)
elapsed = time.time() - start

tokens = len(llm.tokenize(output['choices'][0]['text'].encode()))
print(f"Generated {tokens} tokens in {elapsed:.2f}s")
print(f"Speed: {tokens/elapsed:.2f} tokens/sec")
```

### Logging and Diagnostics

```python
import logging
from llama_cpp import Llama

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

llm = Llama(
    model_path="./models/model.gguf",
    verbose=True  # Enable llama.cpp verbose output
)

# Print model metadata
print(f"Model type: {llm.metadata.get('general.architecture', 'unknown')}")
print(f"Parameter count: {llm.metadata.get('general.parameter_count', 'unknown')}")
print(f"Context length: {llm.n_ctx()}")
```

---

## 8. Production Best Practices

### Configuration Management

```python
from dataclasses import dataclass
from llama_cpp import Llama

@dataclass
class ModelConfig:
    """Model configuration for different deployment scenarios."""
    model_path: str
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0
    n_batch: int = 512
    verbose: bool = False

    @classmethod
    def for_production(cls, model_path: str) -> 'ModelConfig':
        """Production-optimized config."""
        return cls(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35,
            n_batch=512,
            verbose=False
        )

    @classmethod
    def for_development(cls, model_path: str) -> 'ModelConfig':
        """Development config with verbose output."""
        return cls(
            model_path=model_path,
            n_ctx=512,
            n_threads=2,
            n_gpu_layers=0,
            verbose=True
        )

# Usage
config = ModelConfig.for_production("./models/model.gguf")
llm = Llama(**config.__dict__)
```

### Thread Safety

```python
from llama_cpp import Llama
from threading import Lock
from typing import List, Dict

class ThreadSafeLlama:
    """Thread-safe wrapper for Llama."""

    def __init__(self, model_path: str, **kwargs):
        self.llm = Llama(model_path=model_path, **kwargs)
        self.lock = Lock()

    def __call__(self, prompt: str, **kwargs) -> Dict:
        """Thread-safe generation."""
        with self.lock:
            return self.llm(prompt, **kwargs)

    def create_chat_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Thread-safe chat completion."""
        with self.lock:
            return self.llm.create_chat_completion(messages, **kwargs)

# Usage in multi-threaded environment
llm = ThreadSafeLlama(model_path="./models/model.gguf")
```

### Resource Cleanup

```python
from llama_cpp import Llama
from contextlib import contextmanager

@contextmanager
def llama_context(model_path: str, **kwargs):
    """Context manager for automatic cleanup."""
    llm = Llama(model_path=model_path, **kwargs)
    try:
        yield llm
    finally:
        # Explicit cleanup (Python will GC, but this is cleaner)
        del llm

# Usage
with llama_context("./models/model.gguf") as llm:
    output = llm("Hello, world!", max_tokens=50)
    print(output['choices'][0]['text'])
# Model automatically cleaned up here
```

---

## 9. Integration Examples

### Flask API Server

```python
from flask import Flask, request, jsonify, Response
from llama_cpp import Llama
import json

app = Flask(__name__)

# Initialize model once
llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=35,
    chat_format="llama-2"
)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat endpoint."""
    data = request.json
    messages = data.get('messages', [])
    stream = data.get('stream', False)

    if stream:
        def generate():
            for chunk in llm.create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=data.get('max_tokens', 512)
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(generate(), mimetype='text/event-stream')
    else:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.7)
        )
        return jsonify(response)

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Text completion endpoint."""
    data = request.json
    prompt = data.get('prompt', '')

    response = llm(
        prompt,
        max_tokens=data.get('max_tokens', 512),
        temperature=data.get('temperature', 0.7),
        top_p=data.get('top_p', 0.95)
    )

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### FastAPI with Async

```python
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize model
llm = Llama(
    model_path="./models/model.gguf",
    n_ctx=2048,
    n_gpu_layers=35
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

@app.post("/chat")
async def chat(request: ChatRequest):
    """Async chat endpoint."""
    messages = [msg.dict() for msg in request.messages]

    # Run blocking operation in thread pool
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        executor,
        lambda: llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
    )

    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 10. Testing

### Unit Tests

```python
import unittest
from llama_cpp import Llama

class TestLlamaCppPython(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize model once for all tests."""
        cls.llm = Llama(
            model_path="./models/test-model.gguf",
            n_ctx=512,
            verbose=False
        )

    def test_basic_generation(self):
        """Test basic text generation."""
        output = self.llm("Test prompt", max_tokens=10)
        self.assertIn('choices', output)
        self.assertGreater(len(output['choices']), 0)

    def test_tokenization(self):
        """Test tokenization."""
        text = "Hello, world!"
        tokens = self.llm.tokenize(text.encode('utf-8'))
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Test detokenization
        decoded = self.llm.detokenize(tokens).decode('utf-8')
        self.assertIn("Hello", decoded)

    def test_chat_completion(self):
        """Test chat completion."""
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "user", "content": "Hi"}
            ],
            max_tokens=20
        )
        self.assertIn('choices', response)
        self.assertIn('message', response['choices'][0])

if __name__ == '__main__':
    unittest.main()
```

---

## Summary

In this lesson, you learned:
- ✅ How to install llama-cpp-python with various backends
- ✅ High-level and low-level API usage
- ✅ Streaming inference and chat completions
- ✅ Advanced features (embeddings, function calling, grammars)
- ✅ Performance optimization techniques
- ✅ Production best practices
- ✅ Integration patterns with web frameworks

## Next Steps

- **Lesson 2**: RAG Systems - Build retrieval-augmented generation pipelines
- **Lab 8.1**: Build a production-ready Python application
- **Project**: Create an OpenAI-compatible API server

## Additional Resources

- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [Examples Repository](https://github.com/abetlen/llama-cpp-python/tree/main/examples)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## Practice Exercises

1. **Exercise 1**: Install llama-cpp-python with GPU support and verify CUDA acceleration
2. **Exercise 2**: Implement streaming chat with conversation history
3. **Exercise 3**: Create a custom sampling strategy
4. **Exercise 4**: Build a simple REST API with rate limiting
5. **Exercise 5**: Optimize inference for your specific hardware

---

**Module**: 08 - Integration & Applications
**Lesson**: 01 - Python Bindings Deep Dive
**Version**: 1.0
**Last Updated**: 2025-11-18

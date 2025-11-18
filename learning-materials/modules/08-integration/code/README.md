# Module 8 - Integration & Applications - Code Examples

This directory contains practical code examples for integrating llama.cpp into applications.

## Examples Overview

| File | Description | Difficulty | Topics |
|------|-------------|------------|--------|
| `01_simple_chat_app.py` | Command-line chat application | Beginner | Chat, Streaming, History |
| `02_simple_rag_system.py` | RAG with vector search | Intermediate | RAG, Embeddings, Retrieval |
| `03_function_calling_agent.py` | Agent with tool use | Advanced | Function Calling, Tools, Agents |
| `04_flask_api_server.py` | OpenAI-compatible API | Intermediate | Flask, REST API, Streaming |
| `05_batch_processing.py` | Parallel batch processing | Intermediate | Threading, CSV, JSONL |

## Prerequisites

```bash
# Install required packages
pip install llama-cpp-python flask numpy tqdm

# For GPU support (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

## Usage

### 1. Simple Chat App

```bash
# Basic usage
python 01_simple_chat_app.py ./models/llama-2-7b-chat.Q4_K_M.gguf

# Interactive chat
You: Hello! Tell me about Python.
Assistant: Python is a high-level programming language...
```

### 2. Simple RAG System

```bash
# With documents directory
python 02_simple_rag_system.py ./models/model.gguf ./documents

# With sample documents (no directory)
python 02_simple_rag_system.py ./models/model.gguf

# Interactive querying
Question: What is Python?
Assistant: Based on the documents, Python is a high-level programming...
```

### 3. Function Calling Agent

```bash
# Run with example queries
python 03_function_calling_agent.py ./models/model.gguf

# The agent can use tools like:
# - get_current_time()
# - calculate(expression)
# - get_weather(location)
# - search_web(query)

# Example queries:
You: What's the weather in Paris?
[Iteration 1]
Assistant: {"function": "get_weather", "arguments": {"location": "Paris"}}
Executing: get_weather({'location': 'Paris'})
Result: {"location": "Paris", "temperature": 18, "condition": "Sunny"}
[Iteration 2]
Assistant: The weather in Paris is sunny with a temperature of 18°C.
```

### 4. Flask API Server

```bash
# Start server
python 04_flask_api_server.py ./models/model.gguf --port 8000

# Test endpoints
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# Streaming chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### 5. Batch Processing

```bash
# Create sample CSV
python 05_batch_processing.py model.gguf --create-sample

# Process CSV file
python 05_batch_processing.py model.gguf \
  --input sample_prompts.csv \
  --output results.csv \
  --workers 4 \
  --max-tokens 256

# Input CSV format:
# id,prompt,category
# 1,"What is Python?",programming
# 2,"Explain ML",AI

# Output includes 'output' and 'success' columns
```

## Code Patterns

### Pattern 1: Streaming Responses

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf")

# Streaming
stream = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

for chunk in stream:
    delta = chunk['choices'][0]['delta']
    if 'content' in delta:
        print(delta['content'], end='', flush=True)
```

### Pattern 2: Conversation History

```python
conversation = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
]

response = llm.create_chat_completion(messages=conversation)
conversation.append({
    "role": "assistant",
    "content": response['choices'][0]['message']['content']
})
```

### Pattern 3: Error Handling

```python
try:
    response = llm(prompt, max_tokens=512)
except Exception as e:
    print(f"Generation failed: {e}")
    # Fallback strategy
```

## Performance Tips

1. **GPU Acceleration**: Set `n_gpu_layers` to offload layers to GPU
2. **Context Window**: Adjust `n_ctx` based on your needs (smaller = faster)
3. **Batch Size**: Increase `n_batch` for prompt processing
4. **Threads**: Set `n_threads` to number of CPU cores
5. **Model Selection**: Use Q4_K_M for best quality/speed tradeoff

## Common Issues

### Issue: Model loading fails

```python
# Solution: Check model path and format
from pathlib import Path
if not Path(model_path).exists():
    print("Model file not found!")
```

### Issue: Out of memory

```python
# Solution: Reduce GPU layers or context size
llm = Llama(
    model_path=model_path,
    n_gpu_layers=20,  # Reduced from 35
    n_ctx=2048        # Reduced from 4096
)
```

### Issue: Slow inference

```python
# Solution: Optimize settings
llm = Llama(
    model_path=model_path,
    n_threads=8,      # Increase CPU threads
    n_batch=512,      # Increase batch size
    n_gpu_layers=35   # Use GPU if available
)
```

## Next Steps

- **Labs**: Try hands-on labs in `../labs/`
- **Projects**: Build full applications in `../projects/`
- **Tutorials**: Read best practices in `../tutorials/`

## Additional Resources

- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

**Module**: 08 - Integration & Applications
**Version**: 1.0
**Last Updated**: 2025-11-18

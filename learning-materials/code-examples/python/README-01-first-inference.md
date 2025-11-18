# Example 01: First Inference

**Module**: 1.4 - Basic Inference
**Difficulty**: Beginner
**Estimated Time**: 15 minutes

## Overview

This example demonstrates the fundamental workflow of loading a GGUF model and performing basic text generation using llama.cpp Python bindings. This is your first hands-on experience with LLM inference.

## Learning Objectives

By completing this example, you will:
- ✅ Understand how to load a GGUF model file
- ✅ Configure basic model parameters (context size, GPU layers)
- ✅ Perform simple text generation
- ✅ Handle common errors during model loading
- ✅ Understand the structure of generation parameters

## Prerequisites

- Python 3.8 or higher
- llama-cpp-python installed (`pip install llama-cpp-python`)
- A GGUF model file (see "Getting a Model" below)
- Basic understanding of Python

## Getting a Model

Before running this example, you need a GGUF model file. Here are some options:

### Option 1: Small Model (Recommended for testing)
TinyLlama (1.1B parameters, ~600MB)
```bash
mkdir -p models
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### Option 2: Standard Model
Llama-2-7B (7B parameters, ~4GB)
```bash
# Download from HuggingFace using git-lfs or huggingface-cli
# Example: https://huggingface.co/TheBloke/Llama-2-7B-GGUF
```

### Option 3: Browse Models
Visit [HuggingFace GGUF Models](https://huggingface.co/models?search=gguf) to find more options.

## Installation

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# For GPU support (CUDA):
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# For GPU support (Metal on macOS):
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

## Usage

### Basic Usage

```bash
# Run with default model path
python 01-first-inference.py

# Run with custom model path
python 01-first-inference.py path/to/your/model.gguf
```

### Expected Output

```
============================================================
LLaMA.cpp Python - First Inference Example
============================================================

Model path: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

Step 1: Loading model...
------------------------------------------------------------
Loading model from: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Context size: 2048 tokens
GPU layers: 0
This may take a few moments...
✓ Model loaded successfully!
  - Context window: 2048 tokens
  - Vocabulary size: 32000 tokens

Step 2: Preparing prompt...
------------------------------------------------------------
Prompt: The capital of France is

Step 3: Generating text...
------------------------------------------------------------
Generating... (this may take a few seconds)

Step 4: Results
------------------------------------------------------------
Prompt: The capital of France is
Generated:  Paris. It is located in the north-central part of the country...

============================================================
Additional Example: Longer prompt
============================================================
...
```

## Key Concepts Explained

### 1. Model Loading
```python
model = Llama(
    model_path="models/model.gguf",
    n_ctx=2048,         # Context window size
    n_gpu_layers=0,     # GPU offloading
    verbose=True        # Show loading info
)
```

**Parameters**:
- `model_path`: Path to your GGUF model file
- `n_ctx`: Maximum context length in tokens (input + output)
  - Larger = more memory but longer conversations
  - Typical values: 2048, 4096, 8192
- `n_gpu_layers`: Number of layers to offload to GPU
  - 0 = CPU only
  - -1 = all layers on GPU
  - 20-35 = partial offloading (good for limited VRAM)

### 2. Text Generation
```python
output = model(
    prompt="Your prompt here",
    max_tokens=128,      # Maximum tokens to generate
    temperature=0.7,     # Randomness (0.0 = deterministic)
    echo=False          # Don't repeat the prompt
)
```

**Parameters**:
- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Controls randomness (0.0 - 2.0)
  - 0.0: Always picks most likely token (deterministic)
  - 0.7: Balanced (recommended for most tasks)
  - 1.0+: More creative but less coherent
- `stop`: List of strings that stop generation
- `echo`: Whether to include prompt in output

### 3. Error Handling

The example demonstrates proper error handling for:
- ✅ Missing model files
- ✅ Invalid model format
- ✅ Insufficient memory
- ✅ Generation failures

## Common Issues and Solutions

### Issue 1: Model file not found
```
FileNotFoundError: Model file not found: models/model.gguf
```
**Solution**: Download a GGUF model first (see "Getting a Model" above)

### Issue 2: Out of memory
```
Error loading model: ...
```
**Solution**:
- Try a smaller model (e.g., TinyLlama instead of Llama-2-7B)
- Try a more quantized version (Q4_K_M instead of Q6_K)
- Reduce `n_ctx` parameter

### Issue 3: llama-cpp-python not installed
```
ModuleNotFoundError: No module named 'llama_cpp'
```
**Solution**: `pip install llama-cpp-python`

### Issue 4: Slow generation on CPU
**Solution**:
- Enable GPU offloading: `n_gpu_layers=-1`
- Use a smaller model
- Use a more quantized model

## Code Structure

```
01-first-inference.py
├── load_model()           # Handles model loading with error checking
├── generate_text()        # Wraps generation with error handling
└── main()                 # Demonstrates complete workflow
```

## Experiments to Try

1. **Temperature Exploration**:
   - Try temperature values: 0.0, 0.5, 0.7, 1.0, 1.5
   - Observe how randomness affects output quality

2. **Context Size Impact**:
   - Compare n_ctx values: 512, 2048, 4096
   - Monitor memory usage with different sizes

3. **Different Prompts**:
   - Factual: "The capital of Japan is"
   - Creative: "Write a story about"
   - Technical: "Explain quantum computing in simple terms"

4. **Max Tokens**:
   - Try: 16, 64, 128, 256, 512
   - Observe generation speed vs output length

## Performance Tips

1. **Use GPU acceleration** if available (`n_gpu_layers=-1`)
2. **Start with smaller models** for testing
3. **Use Q4_K_M quantization** for good balance of speed/quality
4. **Monitor memory usage** with `htop` or Task Manager
5. **Reduce context size** if you don't need long conversations

## Next Steps

After completing this example:
- ✅ Try `02-basic-chat.py` for interactive conversations
- ✅ Experiment with different models and quantizations
- ✅ Read Module 1.4 documentation on inference parameters
- ✅ Complete Lab 1.4: First Inference

## Related Documentation

- [Module 1.4: Basic Inference](../../modules/01-foundations/docs/)
- [GGUF Format Deep Dive](../../modules/01-foundations/docs/)
- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)

## Interview Topics

This example covers these common interview topics:
- Model loading and initialization
- Memory management for LLM inference
- Generation parameter tuning
- Error handling in production systems

**Typical Interview Question**:
> "How would you load and serve a 7B parameter model with limited memory?"

**Answer**: Use quantization (Q4_K_M), adjust context size, consider GPU offloading, and implement proper error handling.

---

**Author**: Agent 3 (Code Developer)
**Last Updated**: 2025-11-18
**Module**: 1.4 - Basic Inference

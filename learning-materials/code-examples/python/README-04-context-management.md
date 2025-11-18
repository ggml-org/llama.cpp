# Example 04: Context Management

**Module**: 1.4 - Basic Inference
**Difficulty**: Intermediate
**Estimated Time**: 30 minutes

## Overview

This example explores context windows, KV cache, and strategies for managing long sequences. You'll learn how to work within token limits and optimize memory usage for efficient inference.

## Learning Objectives

By completing this example, you will:
- ✅ Understand context window limits and their implications
- ✅ Learn how KV cache works and why it matters
- ✅ Master strategies for handling long documents
- ✅ Implement truncation and sliding window techniques
- ✅ Monitor and optimize memory usage
- ✅ Choose appropriate context sizes for different tasks

## Prerequisites

- Completed examples 01-03
- Understanding of how transformers process sequences (helpful)
- A GGUF model file

## Installation

```bash
pip install llama-cpp-python
```

## Usage

```bash
python 04-context-management.py [model_path]
```

## Key Concepts Explained

### 1. Context Window

**Definition**: Maximum total tokens the model can process in one forward pass.

```python
model = Llama(
    model_path="model.gguf",
    n_ctx=2048  # Context window size
)
```

**Key points**:
- Includes both input prompt AND generated output
- Example: n_ctx=2048 means prompt + generation ≤ 2048 tokens
- Larger context = more memory usage
- Exceeding context causes truncation or errors

**Typical sizes**:
- 2048: Standard for most tasks
- 4096: Long conversations, documents
- 8192: Very long documents
- 16384+: Specialized long-context models

### 2. Token Estimation

Approximating tokens from text:

```python
def estimate_tokens(text: str) -> int:
    # Rough approximation
    return len(text) // 4  # ~1 token per 4 characters
```

**More accurate methods**:
```python
# Use model's tokenizer (if available)
tokens = model.tokenize(text.encode('utf-8'))
token_count = len(tokens)
```

**Rules of thumb**:
- English text: ~1 token per 4 characters
- Code: ~1 token per 3 characters
- Special characters: Variable, can be >1 token each

### 3. KV Cache

**What it is**: Cached attention keys and values from previous tokens.

**Why it matters**:
```
Without KV cache:
  Generate token 1: Process all tokens (1 token)
  Generate token 2: Process all tokens (2 tokens)
  Generate token 3: Process all tokens (3 tokens)
  Total: 1+2+3 = 6 operations

With KV cache:
  Generate token 1: Process 1 token, cache KV
  Generate token 2: Process 1 token, reuse cache
  Generate token 3: Process 1 token, reuse cache
  Total: 1+1+1 = 3 operations
```

**Memory usage**:
```
KV cache size = n_layers × n_tokens × hidden_dim × bytes_per_value

Example (Llama-2-7B, 2048 context):
32 layers × 2048 tokens × 4096 hidden × 2 bytes (FP16)
= ~512 MB just for KV cache!
```

**Performance impact**:
- Enables O(n) generation instead of O(n²)
- Critical for multi-turn conversations
- Cache reused when context is reused
- Invalidated when context changes

### 4. Context Management Strategies

#### Strategy 1: Head Truncation
Keep beginning, drop end.

```python
max_length = (n_ctx - reserve_for_generation) * 4
truncated = long_text[:max_length]
```

**Pros**: Simple, preserves introduction
**Cons**: Loses ending
**Use case**: When beginning contains key info

#### Strategy 2: Tail Truncation
Keep ending, drop beginning.

```python
truncated = long_text[-max_length:]
```

**Pros**: Preserves conclusion
**Cons**: Loses context
**Use case**: When ending is most important

#### Strategy 3: Middle Truncation
Keep both ends, drop middle.

```python
head = long_text[:max_length//2]
tail = long_text[-(max_length//2):]
truncated = head + "\n[...]\n" + tail
```

**Pros**: Preserves context and conclusion
**Cons**: Loses middle content
**Use case**: When intro and conclusion both important

#### Strategy 4: Sliding Window
Process in overlapping chunks.

```python
window_size = n_ctx * 4
stride = window_size // 2  # 50% overlap

position = 0
while position < len(document):
    chunk = document[position:position + window_size]
    process(chunk)
    position += stride
```

**Pros**: Can handle unlimited length
**Cons**: More complex, multiple passes
**Use case**: Analyzing long documents

#### Strategy 5: Hierarchical Summarization
Compress old context.

```python
# First pass: Summarize chunks
summaries = [summarize(chunk) for chunk in chunks]

# Second pass: Combine summaries
final_summary = summarize("\n".join(summaries))
```

**Pros**: Preserves key information efficiently
**Cons**: Information loss, complexity
**Use case**: Very long documents, conversations

## Demonstrations

The example includes 4 demonstrations:

### Demo 1: Context Window Limits
Shows what happens when approaching token limits.
- Small prompts (plenty of room)
- Large prompts (approaching limits)
- Strategies to avoid overflow

### Demo 2: Truncation Strategies
Compares different truncation approaches.
- Head truncation
- Tail truncation
- Middle truncation

### Demo 3: Sliding Window
Processes long document in overlapping chunks.
- Window size and stride configuration
- Overlap for context preservation
- Scalability to unlimited length

### Demo 4: KV Cache Behavior
Explains and demonstrates KV cache.
- Memory implications
- Performance benefits
- Timing comparisons

## Memory Calculations

### Model Memory
```
Base model memory = parameters × bytes_per_parameter

Examples:
- 7B model, Q4_K_M: 7B × 0.5 bytes ≈ 3.5 GB
- 7B model, FP16: 7B × 2 bytes ≈ 14 GB
- 13B model, Q4_K_M: 13B × 0.5 bytes ≈ 6.5 GB
```

### KV Cache Memory
```
Cache memory = n_layers × n_ctx × hidden_dim × 2 (K and V) × bytes

Examples (all with n_ctx=2048):
- 7B model (32 layers, 4096 hidden): ~512 MB
- 13B model (40 layers, 5120 hidden): ~800 MB
- 70B model (80 layers, 8192 hidden): ~2.5 GB
```

### Total Memory Budget
```
Total = Model + KV Cache + Activations + Overhead

Example (7B Q4_K_M, ctx=2048):
  Model: 3.5 GB
  KV Cache: 0.5 GB
  Activations: ~0.5 GB
  Overhead: ~0.5 GB
  Total: ~5 GB
```

## Best Practices

### 1. Choose Appropriate Context Size
```python
# Don't default to maximum
n_ctx = 2048  # Good for most tasks

# Only increase when needed
n_ctx = 4096  # Long conversations
n_ctx = 8192  # Long documents
```

### 2. Reserve Tokens for Generation
```python
# Always leave room for output
max_prompt_tokens = n_ctx - max_generation_tokens - 100  # 100 token buffer
```

### 3. Monitor Token Usage
```python
prompt_tokens = len(model.tokenize(prompt.encode('utf-8')))
if prompt_tokens + max_tokens > n_ctx:
    # Handle overflow
    prompt = truncate_or_summarize(prompt)
```

### 4. Manage Conversation History
```python
class Chat:
    def __init__(self, max_history_tokens=1500):
        self.max_history_tokens = max_history_tokens

    def add_message(self, message):
        # Add message
        # If too long, remove oldest messages
        while total_tokens > self.max_history_tokens:
            remove_oldest()
```

### 5. Use Sliding Window for Long Documents
```python
def process_long_document(doc, model, window_size):
    results = []
    for chunk in sliding_window(doc, window_size):
        result = model(chunk)
        results.append(result)
    return combine(results)
```

## Common Issues and Solutions

### Issue 1: "Context exceeded" error
```
Error: Requested tokens exceed context window
```
**Solutions**:
1. Reduce `max_tokens` in generation
2. Truncate input prompt
3. Increase `n_ctx` when loading model (if you have RAM)
4. Use sliding window approach

### Issue 2: Slow generation with long contexts
```
Generation speed decreases as context grows
```
**Solutions**:
1. Use GPU acceleration (`n_gpu_layers=-1`)
2. Reduce context size if possible
3. Use smaller model
4. Consider quantization

### Issue 3: High memory usage
```
System runs out of RAM
```
**Solutions**:
1. Reduce `n_ctx`
2. Use more aggressive quantization (Q4 instead of Q6)
3. Use smaller model
4. Enable GPU offloading
5. Clear cache between unrelated generations

### Issue 4: Lost conversation context
```
Model doesn't remember earlier messages
```
**Check**:
1. Are you including history in prompts?
2. Is history being truncated too aggressively?
3. Is context window large enough?
4. Are you using separate model instances?

## Experiments to Try

### 1. Context Size Impact
```python
for n_ctx in [512, 1024, 2048, 4096]:
    model = Llama(model_path=path, n_ctx=n_ctx)
    # Measure memory usage
    # Time generation with long prompts
```

### 2. Truncation Strategy Comparison
```python
strategies = [head_truncate, tail_truncate, middle_truncate]
for strategy in strategies:
    truncated = strategy(long_text, max_length)
    result = model(truncated)
    # Compare result quality
```

### 3. Sliding Window Parameters
```python
# Try different window sizes and overlaps
for window_size in [1024, 2048, 4096]:
    for overlap in [0.0, 0.25, 0.5, 0.75]:
        process_with_sliding_window(doc, window_size, overlap)
```

## Interview Topics

**Context Management**:
> Q: "How would you handle a 100-page document with a 2048 token context window?"
>
> A: Use sliding window with 50% overlap. Process each chunk, extract key information, then combine results. Alternatively, use hierarchical summarization: summarize chunks, then summarize summaries.

**Memory Optimization**:
> Q: "Explain the trade-offs between context size and performance."
>
> A: Larger context allows longer inputs but requires more memory (both model and KV cache). Generation may slow down with very long contexts. Choose context size based on actual needs, not maximum possible.

**KV Cache**:
> Q: "What is KV cache and why is it important?"
>
> A: KV cache stores attention keys and values for previously processed tokens, avoiding recomputation. Without it, generation would be O(n²). With it, it's O(n). Critical for multi-turn conversations and long sequences.

**Production Systems**:
> Q: "Design a chatbot that can discuss arbitrarily long documents."
>
> A:
> 1. Store document in vector database (embeddings)
> 2. Retrieve relevant chunks for each query
> 3. Include only relevant context in prompt
> 4. Maintain conversation history with smart truncation
> 5. Summarize old exchanges if conversation gets long

## Advanced Topics

### Context Window Extension
Some models support extended contexts via techniques like:
- **ALiBi**: Attention with Linear Biases
- **RoPE Scaling**: Scale Rotary Position Embeddings
- **Sliding Window Attention**: Local attention mechanism

### Infinite Context (Concept)
Approaches to handle unlimited length:
- **Retrieval-augmented**: Query relevant segments
- **Memory networks**: Separate context storage
- **Hierarchical processing**: Multi-level summarization

## Next Steps

After completing this example:
- ✅ Try `05-batch-inference.py` for efficient multi-prompt processing
- ✅ Experiment with different context sizes and measure impact
- ✅ Implement a production chat system with smart context management
- ✅ Read Module 2 on KV cache implementation details

## Related Documentation

- [Module 1.5: Memory Management Basics](../../modules/01-foundations/docs/)
- [Module 2.3: KV Cache Implementation](../../modules/02-core-implementation/docs/)
- [Lab 1.5: Memory Profiling](../../modules/01-foundations/labs/)

---

**Author**: Agent 3 (Code Developer)
**Last Updated**: 2025-11-18
**Module**: 1.4 - Basic Inference

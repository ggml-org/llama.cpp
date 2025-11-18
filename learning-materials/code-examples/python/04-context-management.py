"""
Module: 04-context-management.py
Purpose: Demonstrate context window management and KV cache usage
Learning Objectives:
    - Understand context window limits and token counting
    - Learn about KV cache and its role in inference
    - Manage long conversations and documents
    - Implement context window strategies (truncation, sliding window)
    - Monitor memory usage and performance

Prerequisites: Module 1 Lesson 1.4 complete
Estimated Time: 30 minutes
Module: 1.4 - Basic Inference
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple
import time

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Install it with: pip install llama-cpp-python")
    sys.exit(1)


def load_model(model_path: str, n_ctx: int = 2048) -> Optional[Llama]:
    """Load a GGUF model with specified context size."""
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return None

    try:
        print(f"Loading model: {model_path}")
        print(f"Context window: {n_ctx} tokens\n")
        return Llama(
            model_path=str(model_file),
            n_ctx=n_ctx,
            n_gpu_layers=0,
            verbose=False
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.

    This is a rough approximation. Actual tokenization depends on the model's
    vocabulary and tokenizer.

    Rule of thumb:
    - English: ~1 token per 4 characters
    - Code: ~1 token per 3 characters
    - Special chars: May use more tokens

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Simple character-based estimate
    # Real tokenizers use BPE, SentencePiece, etc.
    return len(text) // 4


def demonstrate_context_limits(model: Llama) -> None:
    """
    Demonstrate what happens at context limits.

    Key concepts:
    - Context window: Maximum total tokens (prompt + generation)
    - If prompt + max_tokens > n_ctx, generation will be truncated
    - KV cache grows with sequence length
    """
    print("=" * 70)
    print("DEMO 1: Context Window Limits")
    print("=" * 70)
    print()

    n_ctx = model.n_ctx()
    print(f"Model context window: {n_ctx} tokens\n")

    # Test 1: Small prompt (well within limits)
    print("Test 1: Small prompt (fits easily)")
    print("-" * 70)
    prompt1 = "Count from 1 to 10: "
    estimated_tokens = estimate_tokens(prompt1)
    print(f"Prompt: {prompt1}")
    print(f"Estimated tokens: {estimated_tokens}")
    print(f"Max generation tokens: 100")
    print(f"Total: {estimated_tokens + 100} / {n_ctx} tokens")
    print()

    output1 = model(prompt1, max_tokens=100, temperature=0.7, echo=False)
    print(f"Generated: {output1['choices'][0]['text'][:200]}...")
    print(f"✓ Success: Plenty of room in context\n")

    # Test 2: Large prompt (approaching limits)
    print("Test 2: Large prompt (approaching limits)")
    print("-" * 70)
    # Create a long prompt
    long_text = " ".join([f"This is sentence {i}." for i in range(200)])
    prompt2 = f"Summarize the following text:\n{long_text}\n\nSummary:"
    estimated_tokens = estimate_tokens(prompt2)
    print(f"Prompt length: {len(prompt2)} characters")
    print(f"Estimated tokens: {estimated_tokens}")
    print(f"Max generation tokens: 100")
    print(f"Total: {estimated_tokens + 100} / {n_ctx} tokens")
    print()

    if estimated_tokens + 100 < n_ctx:
        output2 = model(prompt2, max_tokens=100, temperature=0.7, echo=False)
        print(f"Generated: {output2['choices'][0]['text'][:200]}...")
        print(f"✓ Success: Still fits in context\n")
    else:
        print(f"⚠ Warning: Would exceed context window!")
        print(f"Strategy needed: Truncate prompt or reduce max_tokens\n")


def demonstrate_truncation_strategies(model: Llama) -> None:
    """
    Demonstrate different strategies for handling long text.

    Strategies:
    1. Head truncation: Keep beginning, drop end
    2. Tail truncation: Keep end, drop beginning
    3. Middle truncation: Keep beginning and end, drop middle
    4. Sliding window: Process in chunks
    5. Summarization: Compress context
    """
    print("=" * 70)
    print("DEMO 2: Truncation Strategies")
    print("=" * 70)
    print()

    # Create a long document
    long_document = "\n".join([
        f"Paragraph {i}: This is the content of paragraph {i}. "
        f"It contains important information about topic {i}. "
        f"We need to preserve this content somehow."
        for i in range(1, 51)  # 50 paragraphs
    ])

    n_ctx = model.n_ctx()
    doc_tokens = estimate_tokens(long_document)
    print(f"Document: {len(long_document)} characters, ~{doc_tokens} tokens")
    print(f"Context window: {n_ctx} tokens")
    print()

    # Strategy 1: Head truncation (keep beginning)
    print("Strategy 1: Head Truncation (keep beginning)")
    print("-" * 70)
    max_chars = (n_ctx - 200) * 4  # Reserve 200 tokens for prompt/generation
    truncated_head = long_document[:max_chars]
    prompt_head = f"Summarize this text:\n{truncated_head}\n\nSummary:"
    print(f"Truncated to: {len(truncated_head)} characters")
    print("Use case: When beginning contains most important info")
    print("Pros: Simple, preserves introduction")
    print("Cons: Loses ending/conclusion\n")

    # Strategy 2: Tail truncation (keep ending)
    print("Strategy 2: Tail Truncation (keep ending)")
    print("-" * 70)
    truncated_tail = long_document[-max_chars:]
    prompt_tail = f"Summarize this text:\n{truncated_tail}\n\nSummary:"
    print(f"Truncated to: {len(truncated_tail)} characters")
    print("Use case: When ending contains most important info")
    print("Pros: Preserves conclusion")
    print("Cons: Loses context from beginning\n")

    # Strategy 3: Middle truncation (keep beginning and end)
    print("Strategy 3: Middle Truncation (keep both ends)")
    print("-" * 70)
    head_size = max_chars // 2
    tail_size = max_chars // 2
    truncated_middle = long_document[:head_size] + "\n[...]\n" + long_document[-tail_size:]
    print(f"Kept: {head_size} chars from start + {tail_size} chars from end")
    print("Use case: When intro and conclusion are both important")
    print("Pros: Preserves context and conclusion")
    print("Cons: Loses middle content\n")


def demonstrate_sliding_window(model: Llama) -> None:
    """
    Demonstrate sliding window approach for long documents.

    Sliding window:
    - Process document in overlapping chunks
    - Each chunk fits in context window
    - Useful for: summarization, analysis, search
    """
    print("=" * 70)
    print("DEMO 3: Sliding Window Processing")
    print("=" * 70)
    print()

    # Create a long document
    chapters = [
        f"Chapter {i}: This chapter discusses topic {i}. "
        f"It contains important information and insights. "
        f"The main point is about {i * 2}. "
        f"We learn many things in this chapter. "
        for i in range(1, 11)
    ]
    full_document = "\n\n".join(chapters)

    print(f"Document: {len(chapters)} chapters")
    print(f"Total length: {len(full_document)} characters")
    print()

    # Define window size (in characters, approximating tokens)
    n_ctx = model.n_ctx()
    window_size = (n_ctx - 200) * 4  # Reserve tokens for prompt/generation
    stride = window_size // 2  # 50% overlap

    print(f"Window size: ~{window_size} characters")
    print(f"Stride: ~{stride} characters (50% overlap)")
    print(f"Overlap helps maintain context between windows\n")

    # Process in windows
    print("Processing windows:")
    print("-" * 70)
    summaries = []

    position = 0
    window_num = 1
    while position < len(full_document):
        # Extract window
        window = full_document[position:position + window_size]

        # Create prompt
        prompt = f"Briefly summarize this text in one sentence:\n{window}\n\nSummary:"

        # Generate (for demo, we'll just show the setup)
        print(f"Window {window_num}:")
        print(f"  Position: {position}-{position + len(window)}")
        print(f"  Length: {len(window)} characters")
        print(f"  Preview: {window[:100]}...")
        print()

        # In real usage, generate here:
        # output = model(prompt, max_tokens=50, temperature=0.7, echo=False)
        # summaries.append(output['choices'][0]['text'])

        # Move to next window
        position += stride
        window_num += 1

    print(f"Total windows processed: {window_num - 1}")
    print("Use case: Analyzing long documents, books, reports")
    print("Benefit: Can process unlimited length content\n")


def demonstrate_kv_cache(model: Llama) -> None:
    """
    Demonstrate KV cache behavior.

    Key-Value (KV) Cache:
    - Stores attention keys and values for processed tokens
    - Avoids recomputing attention for previous tokens
    - Critical for efficient multi-turn conversations
    - Memory usage grows with sequence length

    Memory usage per token in cache:
    - Depends on: model layers, hidden size, precision
    - Typical: ~2-4 bytes per token per layer
    - Example: 32 layers, 2048 tokens → ~256MB
    """
    print("=" * 70)
    print("DEMO 4: Understanding KV Cache")
    print("=" * 70)
    print()

    print("What is KV Cache?")
    print("-" * 70)
    print("""
The KV (Key-Value) cache is a crucial optimization for transformer inference:

1. Without cache:
   - Each new token requires recomputing attention over ALL previous tokens
   - Complexity: O(n²) for sequence length n
   - Very slow for long sequences

2. With cache:
   - Store computed keys and values for previous tokens
   - Only compute for new token
   - Complexity: O(n) for sequence length n
   - Much faster!

Memory implications:
- Cache size = n_layers × n_tokens × hidden_dim × precision
- For Llama-2-7B with 2048 context:
  - 32 layers × 2048 tokens × 4096 hidden × 2 bytes ≈ 512MB
    """)

    n_ctx = model.n_ctx()
    n_vocab = model.n_vocab()

    print(f"Model configuration:")
    print(f"  - Context window: {n_ctx} tokens")
    print(f"  - Vocabulary size: {n_vocab} tokens")
    print()

    print("Timing comparison (demonstration):")
    print("-" * 70)

    # First generation (cold cache)
    prompt1 = "Write a short story:"
    print("Generation 1 (empty cache):")
    print(f"Prompt: {prompt1}")
    start = time.time()
    output1 = model(prompt1, max_tokens=50, temperature=0.7, echo=False)
    elapsed1 = time.time() - start
    print(f"Time: {elapsed1:.2f}s")
    print(f"Tokens generated: ~50")
    print(f"Speed: ~{50/elapsed1:.1f} tokens/sec\n")

    # Second generation (warm cache if using same context)
    # Note: In llama-cpp-python, cache is managed automatically
    print("Generation 2 (continuing):")
    prompt2 = output1['choices'][0]['text'] + " Furthermore,"
    start = time.time()
    output2 = model(prompt2, max_tokens=50, temperature=0.7, echo=False)
    elapsed2 = time.time() - start
    print(f"Prompt: [previous output] + 'Furthermore,'")
    print(f"Time: {elapsed2:.2f}s")
    print(f"Tokens generated: ~50")
    print(f"Speed: ~{50/elapsed2:.1f} tokens/sec\n")

    print("Cache behavior:")
    print("  - Cache reused when possible (same context)")
    print("  - Cache invalidated when context changes")
    print("  - Cache size limits max sequence length")
    print()


def print_best_practices() -> None:
    """Print best practices for context management."""
    print("=" * 70)
    print("BEST PRACTICES: Context Management")
    print("=" * 70)
    print("""
1. Choose appropriate context size:
   - Larger context = more memory usage
   - Don't overallocate: n_ctx=2048 is good for most tasks
   - Increase only when needed: long conversations, documents

2. Estimate token counts:
   - ~1 token per 4 characters (rough estimate)
   - Use model's tokenizer for accuracy
   - Always leave room for generation

3. Handle long inputs:
   - Truncation: Simple but loses information
   - Sliding window: Process in chunks
   - Summarization: Compress old context
   - Hierarchical: Summarize → process summaries

4. Conversation management:
   - Keep last N turns
   - Summarize old messages
   - Use system prompt efficiently
   - Monitor token count

5. Memory optimization:
   - Clear cache when starting new task
   - Use smaller context when possible
   - Consider quantization for larger models
   - Monitor memory usage in production

6. Performance tips:
   - Batch similar-length sequences
   - Reuse cache when possible
   - Profile token generation speed
   - Consider GPU for longer contexts

Common Pitfalls:
❌ Exceeding context window (prompt too long)
❌ Not accounting for generation tokens
❌ Ignoring memory usage growth
❌ Not managing conversation history
❌ Using max context when not needed

Interview Topics:
- "How would you handle a conversation longer than the context window?"
- "Explain the trade-offs between different context management strategies"
- "How does KV cache impact memory and performance?"
- "Design a system to chat about a 1000-page document"
""")


def main() -> None:
    """Main demonstration of context management concepts."""
    print("=" * 70)
    print("LLaMA.cpp Python - Context Management Demonstration")
    print("=" * 70)
    print()
    print("This example demonstrates context windows, KV cache, and")
    print("strategies for handling long sequences.\n")

    # Get model path
    model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Load model with moderate context size
    model = load_model(model_path, n_ctx=2048)
    if model is None:
        sys.exit(1)

    # Run demonstrations
    try:
        demonstrate_context_limits(model)
        demonstrate_truncation_strategies(model)
        demonstrate_sliding_window(model)
        demonstrate_kv_cache(model)
        print_best_practices()

    except Exception as e:
        print(f"Error during demonstration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Context Management Demonstration Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Context window limits total tokens (prompt + generation)")
    print("2. KV cache enables efficient long-sequence generation")
    print("3. Multiple strategies exist for handling long inputs")
    print("4. Choose context size based on your use case")
    print("5. Monitor memory usage, especially for long contexts")
    print("\nNext: Try 05-batch-inference.py for processing multiple prompts")


if __name__ == "__main__":
    main()

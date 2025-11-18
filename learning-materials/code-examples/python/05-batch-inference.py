"""
Module: 05-batch-inference.py
Purpose: Demonstrate efficient batch processing of multiple prompts
Learning Objectives:
    - Process multiple prompts efficiently
    - Understand batching benefits and limitations
    - Implement parallel processing strategies
    - Optimize throughput for production workloads
    - Handle variable-length inputs

Prerequisites: Module 1 Lesson 1.4 complete
Estimated Time: 25 minutes
Module: 1.4 - Basic Inference
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Install it with: pip install llama-cpp-python")
    sys.exit(1)


def load_model(model_path: str, n_ctx: int = 2048) -> Optional[Llama]:
    """Load a GGUF model for batch processing."""
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return None

    try:
        print(f"Loading model: {model_path}\n")
        return Llama(
            model_path=str(model_file),
            n_ctx=n_ctx,
            n_gpu_layers=0,
            verbose=False,
            n_batch=512  # Batch size for prompt processing
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None


def sequential_processing(
    model: Llama,
    prompts: List[str],
    max_tokens: int = 50
) -> List[str]:
    """
    Process prompts one at a time (baseline).

    This is the simplest approach but also the slowest.
    Each prompt is processed completely before starting the next.

    Args:
        model: Loaded Llama model
        prompts: List of input prompts
        max_tokens: Maximum tokens to generate per prompt

    Returns:
        List of generated texts
    """
    print("Sequential Processing")
    print("-" * 70)
    print(f"Processing {len(prompts)} prompts sequentially...")
    print()

    results = []
    start_time = time.time()

    for i, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {i}/{len(prompts)}...", end=" ", flush=True)
        prompt_start = time.time()

        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            echo=False
        )
        generated = output['choices'][0]['text']
        results.append(generated)

        prompt_time = time.time() - prompt_start
        print(f"({prompt_time:.2f}s)")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
    print(f"Throughput: {len(prompts)/total_time:.2f} prompts/second\n")

    return results


def batch_processing_simulation(
    model: Llama,
    prompts: List[str],
    max_tokens: int = 50
) -> List[str]:
    """
    Simulate batch processing with a single model.

    Note: Standard llama-cpp-python doesn't support true parallel batching
    on a single model instance. This demonstrates the concept.

    Real batching requires:
    - Server mode (llama-server) with continuous batching
    - Or multiple model instances (memory intensive)
    - Or specialized frameworks (vLLM, TensorRT-LLM)

    Args:
        model: Loaded Llama model
        prompts: List of input prompts
        max_tokens: Maximum tokens to generate per prompt

    Returns:
        List of generated texts
    """
    print("Batch Processing Simulation")
    print("-" * 70)
    print(f"Processing {len(prompts)} prompts with optimizations...")
    print()

    results = []
    start_time = time.time()

    # Group prompts by similar length for efficiency
    # (In real batching, this improves GPU utilization)
    prompts_with_idx = [(i, p) for i, p in enumerate(prompts)]
    prompts_with_idx.sort(key=lambda x: len(x[1]))

    for i, (original_idx, prompt) in enumerate(prompts_with_idx, 1):
        print(f"Processing batch item {i}/{len(prompts)}...", end=" ", flush=True)
        prompt_start = time.time()

        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            echo=False
        )
        generated = output['choices'][0]['text']
        results.append((original_idx, generated))

        prompt_time = time.time() - prompt_start
        print(f"({prompt_time:.2f}s)")

    # Restore original order
    results.sort(key=lambda x: x[0])
    results = [text for _, text in results]

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
    print(f"Throughput: {len(prompts)/total_time:.2f} prompts/second\n")

    return results


def parallel_processing_multi_instance(
    model_path: str,
    prompts: List[str],
    max_tokens: int = 50,
    max_workers: int = 2
) -> List[str]:
    """
    Process prompts in parallel using multiple model instances.

    This demonstrates true parallelism but requires more memory.
    Each worker loads a separate model instance.

    Pros:
    - True parallel processing
    - Better throughput for multiple requests

    Cons:
    - High memory usage (N × model size)
    - Slower startup (must load N models)
    - Better suited for server deployment

    Args:
        model_path: Path to GGUF model
        prompts: List of input prompts
        max_tokens: Maximum tokens to generate
        max_workers: Number of parallel workers (model instances)

    Returns:
        List of generated texts
    """
    print(f"Parallel Processing ({max_workers} workers)")
    print("-" * 70)
    print(f"Processing {len(prompts)} prompts with {max_workers} parallel workers...")
    print(f"Note: Each worker loads a separate model copy (high memory usage)\n")

    def process_prompt(args):
        """Worker function to process a single prompt."""
        idx, prompt = args
        # Each worker loads its own model instance
        worker_model = Llama(
            model_path=model_path,
            n_ctx=512,  # Use smaller context to save memory
            n_gpu_layers=0,
            verbose=False
        )
        output = worker_model(prompt, max_tokens=max_tokens, temperature=0.7, echo=False)
        return (idx, output['choices'][0]['text'])

    results = []
    start_time = time.time()

    # Use ThreadPoolExecutor for parallel processing
    # Note: GIL limitations mean this may not be fully parallel for CPU inference
    # For true parallelism, use ProcessPoolExecutor or GPU acceleration
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_prompt = {
            executor.submit(process_prompt, (i, prompt)): i
            for i, prompt in enumerate(prompts)
        }

        # Collect results as they complete
        for future in as_completed(future_to_prompt):
            idx = future_to_prompt[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed prompt {len(results)}/{len(prompts)}")
            except Exception as e:
                print(f"Error processing prompt {idx}: {e}")
                results.append((idx, "[ERROR]"))

    # Sort results by original order
    results.sort(key=lambda x: x[0])
    results = [text for _, text in results]

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
    print(f"Throughput: {len(prompts)/total_time:.2f} prompts/second")
    print(f"Speedup vs sequential: {len(prompts)/(total_time * 0.5):.2f}x (approximate)\n")

    return results


def demonstrate_prompt_grouping(prompts: List[str]) -> None:
    """
    Demonstrate the importance of grouping similar-length prompts.

    In real batching systems (GPU), processing variable-length sequences
    requires padding, which wastes computation. Grouping by length
    minimizes padding overhead.

    This is a conceptual demonstration.
    """
    print("Prompt Grouping Strategy")
    print("-" * 70)
    print("For efficient batching, group prompts by similar length:\n")

    # Calculate lengths
    lengths = [(i, len(p), p[:50] + "...") for i, p in enumerate(prompts)]

    # Show original order
    print("Original order:")
    for idx, length, preview in lengths:
        print(f"  {idx}: {length:4d} chars - {preview}")

    print()

    # Show sorted by length
    lengths_sorted = sorted(lengths, key=lambda x: x[1])
    print("Sorted by length (better for batching):")
    for idx, length, preview in lengths_sorted:
        print(f"  {idx}: {length:4d} chars - {preview}")

    print()
    print("Why this matters:")
    print("  - GPU batching requires padding to max length in batch")
    print("  - Mixed lengths → more padding → wasted computation")
    print("  - Similar lengths → less padding → better efficiency")
    print("  - Can improve throughput by 2-5x in production systems\n")


def demonstrate_throughput_optimization(model: Llama) -> None:
    """
    Demonstrate throughput optimization techniques.

    Key techniques:
    1. Prompt batching: Process multiple prompts together
    2. Length grouping: Group similar-length sequences
    3. Dynamic batching: Add prompts to batches as they arrive
    4. KV cache reuse: Share common prefixes
    5. Request queueing: Maintain steady stream of work
    """
    print("Throughput Optimization Techniques")
    print("=" * 70)
    print("""
1. PROMPT BATCHING
   - Process multiple prompts simultaneously
   - Better GPU utilization
   - Higher throughput, potentially higher latency per request
   - Best for: High-volume inference servers

2. LENGTH GROUPING
   - Group prompts by similar length before batching
   - Reduces padding overhead
   - Improves GPU efficiency
   - Can increase throughput 2-5x

3. DYNAMIC BATCHING (Continuous Batching)
   - Add new requests to ongoing batches
   - Minimal latency impact
   - Maximizes GPU utilization
   - Used by: vLLM, TensorRT-LLM, llama-server

4. KV CACHE SHARING
   - Share cache for common prompt prefixes
   - Example: System prompt shared across requests
   - Reduces memory and computation
   - Used in multi-tenant systems

5. REQUEST QUEUEING
   - Buffer incoming requests
   - Form optimal batches
   - Balance latency vs throughput
   - Critical for production systems

Metrics to track:
   - Throughput: requests/second or tokens/second
   - Latency: time to first token, time to completion
   - GPU utilization: target >80%
   - Memory usage: monitor cache size
   - Queue depth: avoid unbounded growth

Production frameworks that do this well:
   - vLLM: Continuous batching, PagedAttention
   - TensorRT-LLM: Optimized kernels, batching
   - llama.cpp server: Basic batching support
   - Text Generation Inference (TGI): Dynamic batching
""")


def main() -> None:
    """
    Main demonstration of batch inference concepts.
    """
    print("=" * 70)
    print("LLaMA.cpp Python - Batch Inference Demonstration")
    print("=" * 70)
    print()
    print("This example demonstrates efficient processing of multiple prompts.\n")

    # Get model path
    model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Load model once for sequential and batch demos
    print("Loading model for demonstrations...\n")
    model = load_model(model_path, n_ctx=2048)
    if model is None:
        sys.exit(1)

    # Create test prompts of varying lengths
    test_prompts = [
        "What is Python?",
        "Explain machine learning in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of using type hints in Python?",
        "Describe the difference between a list and a tuple.",
        "How does garbage collection work?",
        "What is the GIL in Python?",
        "Explain async/await in Python."
    ]

    print(f"Test dataset: {len(test_prompts)} prompts\n")
    print("=" * 70)
    print()

    # Demo 1: Prompt grouping strategy
    demonstrate_prompt_grouping(test_prompts)

    # Demo 2: Sequential processing (baseline)
    print("=" * 70)
    results_sequential = sequential_processing(model, test_prompts, max_tokens=30)

    # Demo 3: Batch processing simulation
    print("=" * 70)
    results_batch = batch_processing_simulation(model, test_prompts, max_tokens=30)

    # Demo 4: Show optimization techniques
    print("=" * 70)
    demonstrate_throughput_optimization(model)

    # Summary
    print("=" * 70)
    print("SUMMARY: Key Takeaways")
    print("=" * 70)
    print("""
1. Batching Improves Throughput
   - Process multiple requests together
   - Critical for production deployment
   - Balance between throughput and latency

2. Standard llama-cpp-python Limitations
   - Single model instance = sequential processing
   - For real batching, use llama-server or specialized frameworks
   - Or use multiple model instances (memory intensive)

3. GPU Acceleration is Key
   - Batching benefits most from GPU
   - CPU batching limited by GIL in Python
   - Consider GPU offloading for production

4. Prompt Preprocessing Helps
   - Group by length
   - Pre-tokenize if possible
   - Cache common prefixes

5. Production Recommendations
   - Use llama.cpp server mode for APIs
   - Or consider vLLM, TensorRT-LLM for advanced features
   - Monitor throughput and latency metrics
   - Implement request queueing
   - Profile and optimize for your workload

Next Steps:
   - Experiment with llama-server for true batching
   - Try GPU acceleration (n_gpu_layers=-1)
   - Implement a production API with queueing
   - Profile your specific workload
""")

    print("\n" + "=" * 70)
    print("Batch Inference Demonstration Complete!")
    print("=" * 70)
    print("\nYou've completed all 5 basic inference examples!")
    print("Ready to move on to Module 1.5 or try the labs.")


if __name__ == "__main__":
    main()

# Example 05: Batch Inference

**Module**: 1.4 - Basic Inference
**Difficulty**: Intermediate to Advanced
**Estimated Time**: 25 minutes

## Overview

This example demonstrates efficient batch processing of multiple prompts, covering throughput optimization techniques essential for production deployment. You'll learn strategies to maximize inference efficiency when handling many requests.

## Learning Objectives

By completing this example, you will:
- ✅ Understand batching benefits and limitations
- ✅ Implement sequential vs batch processing
- ✅ Learn prompt grouping strategies
- ✅ Understand throughput optimization techniques
- ✅ Know when to use different batching approaches
- ✅ Prepare for production-scale inference systems

## Prerequisites

- Completed examples 01-04
- Understanding of throughput vs latency trade-offs
- Basic knowledge of parallel processing concepts
- A GGUF model file

## Installation

```bash
pip install llama-cpp-python
```

## Usage

```bash
python 05-batch-inference.py [model_path]
```

## Key Concepts Explained

### 1. Sequential vs Batch Processing

**Sequential Processing**:
```python
# Process one at a time
for prompt in prompts:
    result = model(prompt)
    results.append(result)
```

**Characteristics**:
- Simple to implement
- Predictable latency per request
- Low throughput
- Poor GPU utilization
- Good for: Single-user applications

**Batch Processing**:
```python
# Process multiple prompts together
results = model.batch_generate(prompts)  # Conceptual
```

**Characteristics**:
- Higher throughput
- Better resource utilization
- More complex implementation
- Potential latency increase per request
- Good for: Servers, APIs, batch jobs

### 2. Throughput vs Latency Trade-off

```
Sequential:
  Request 1: [==============] 1.0s
  Request 2:                  [==============] 1.0s
  Request 3:                                   [==============] 1.0s
  Total: 3.0s, Throughput: 1 req/s

Batched (size=3):
  Requests 1-3: [==================] 1.5s
  Total: 1.5s, Throughput: 2 req/s
  But each request took 1.5s instead of 1.0s
```

**Key insight**: Batching increases total throughput but may increase individual request latency.

### 3. Prompt Grouping

Grouping prompts by similar length reduces padding overhead:

```python
# Before grouping:
prompts = [
    "Short",           # 5 chars → pads to 100
    "Very long prompt...",  # 100 chars → no padding
    "Medium length",   # 20 chars → pads to 100
]
# Wastes: 95 + 0 + 80 = 175 padded tokens

# After grouping:
group1 = ["Short", "Medium length"]  # Max 20 → less padding
group2 = ["Very long prompt..."]     # Max 100 → no padding
# Wastes: 15 + 0 = 15 padded tokens
```

**Benefit**: Can improve throughput by 2-5x in GPU inference.

### 4. Dynamic Batching (Continuous Batching)

Advanced technique used by production systems:

```
Time →
Batch 1: [A, B, C] → Processing
Batch 2: [D, E] → Waiting
    ↓
Batch 1: [A, B] + [D, E] → A done, add new requests
    ↓
Batch 1: [B, D, E, F] → B done, C done, add F
```

**Benefits**:
- Maintains high GPU utilization
- Minimal latency impact
- Handles variable arrival rates
- Used by: vLLM, TensorRT-LLM, llama-server

### 5. KV Cache Sharing

Share common prompt prefixes across requests:

```python
# System prompt (shared across all requests):
system = "You are a helpful assistant."

# User prompts:
prompts = [
    system + "\nUser: What is Python?",
    system + "\nUser: What is Java?",
    system + "\nUser: What is C++?",
]

# With cache sharing:
# 1. Compute KV cache for system prompt once
# 2. Reuse for all requests
# 3. Only compute unique portions
```

**Benefit**: Reduces computation and memory for shared prefixes.

## Implementations Demonstrated

### Implementation 1: Sequential Processing

Baseline approach - process one prompt at a time.

```python
def sequential_processing(model, prompts):
    results = []
    for prompt in prompts:
        output = model(prompt)
        results.append(output)
    return results
```

**Pros**:
- Simple
- Predictable
- Low memory

**Cons**:
- Slow
- Poor utilization
- Not scalable

### Implementation 2: Batch Simulation

Optimized sequential with length sorting.

```python
def batch_processing_simulation(model, prompts):
    # Sort by length
    prompts_sorted = sorted(prompts, key=len)

    # Process in order
    results = []
    for prompt in prompts_sorted:
        output = model(prompt)
        results.append(output)

    # Restore original order
    return restore_order(results)
```

**Note**: Standard llama-cpp-python doesn't support true parallel batching on a single model instance. This shows the concept.

### Implementation 3: Multi-Instance Parallel

Use multiple model instances for true parallelism.

```python
def parallel_processing(model_path, prompts, workers=2):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Each worker loads separate model
        results = executor.map(process_with_new_model, prompts)
    return list(results)
```

**Pros**:
- True parallelism
- Higher throughput

**Cons**:
- High memory (N × model size)
- Slower startup
- Better for servers

## Batching Strategies Comparison

| Strategy | Throughput | Latency | Memory | Complexity | Use Case |
|----------|-----------|---------|---------|-----------|----------|
| Sequential | Low | Low | Low | Simple | Single user |
| Length-sorted | Medium | Medium | Low | Easy | Batch jobs |
| Multi-instance | High | Medium | High | Medium | Small servers |
| True batching* | Very High | Low | Medium | High | Production |

*Requires specialized frameworks (vLLM, llama-server, TensorRT-LLM)

## Performance Metrics

### Key Metrics to Track

**1. Throughput**:
```
Throughput = Total requests / Total time
Example: 100 requests / 50 seconds = 2 req/s
```

**2. Latency**:
```
Time to First Token (TTFT): Time until generation starts
Time to Completion (TTC): Total time for request
```

**3. Token Throughput**:
```
Token Throughput = Total tokens generated / Total time
Example: 5000 tokens / 50 seconds = 100 tokens/s
```

**4. GPU Utilization**:
```
Target: >80% for efficient batching
Monitor with: nvidia-smi, nvtop
```

### Example Measurements

```
Sequential (baseline):
  8 prompts, 30 tokens each
  Time: 16.0s
  Throughput: 0.5 req/s, 15 tokens/s

Batched (size=4):
  8 prompts, 30 tokens each
  Time: 9.0s
  Throughput: 0.9 req/s, 27 tokens/s
  Speedup: 1.8x

Multi-instance (2 workers):
  8 prompts, 30 tokens each
  Time: 8.5s
  Throughput: 0.94 req/s, 28 tokens/s
  Speedup: 1.9x
```

## Production Batching Solutions

### Option 1: llama.cpp Server

Built-in server with basic batching support:

```bash
# Start server
./llama-server -m model.gguf -c 2048 --parallel 4

# Send requests via API
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```

**Features**:
- OpenAI-compatible API
- Basic continuous batching
- Relatively simple setup
- Good for moderate load

### Option 2: vLLM

Advanced inference engine with optimized batching:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="model.gguf")
outputs = llm.generate(prompts, sampling_params)
```

**Features**:
- PagedAttention for efficient memory
- Continuous batching
- High throughput
- GPU-optimized

### Option 3: TensorRT-LLM

NVIDIA's optimized inference:

```python
# Requires TensorRT-LLM setup
# Highly optimized for NVIDIA GPUs
# Complex setup but best performance
```

**Features**:
- Highly optimized kernels
- Advanced batching
- Multi-GPU support
- Best for production at scale

### Option 4: Text Generation Inference (TGI)

HuggingFace's inference server:

```bash
docker run --gpus all \
  -p 8080:80 \
  -v $PWD:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id model
```

**Features**:
- Dynamic batching
- Token streaming
- Good integration with HF ecosystem

## Best Practices

### 1. Choose the Right Approach

```python
# For single user, simple apps:
model = Llama(model_path)
result = model(prompt)

# For batch jobs (offline):
results = [model(p) for p in prompts]

# For production API:
# Use llama-server or vLLM with proper batching
```

### 2. Optimize Prompt Length

```python
# Group by length
short_prompts = [p for p in prompts if len(p) < 100]
long_prompts = [p for p in prompts if len(p) >= 100]

# Process groups separately
results = (
    process_batch(short_prompts) +
    process_batch(long_prompts)
)
```

### 3. Implement Request Queueing

```python
from queue import Queue
import threading

request_queue = Queue()
result_queue = Queue()

def worker():
    while True:
        batch = []
        # Collect batch
        for _ in range(BATCH_SIZE):
            if not request_queue.empty():
                batch.append(request_queue.get())

        # Process batch
        if batch:
            results = process_batch(batch)
            for r in results:
                result_queue.put(r)

# Start workers
for _ in range(NUM_WORKERS):
    threading.Thread(target=worker, daemon=True).start()
```

### 4. Monitor Performance

```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()

    def record(self, num_tokens):
        self.total_requests += 1
        self.total_tokens += num_tokens

    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {
            "throughput_req": self.total_requests / elapsed,
            "throughput_tok": self.total_tokens / elapsed,
            "avg_tokens_per_req": self.total_tokens / self.total_requests
        }
```

### 5. Handle Variable Load

```python
# Adaptive batch size based on queue depth
def get_batch_size(queue_depth):
    if queue_depth < 5:
        return 1  # Low latency for light load
    elif queue_depth < 20:
        return 4  # Balanced
    else:
        return 8  # High throughput for heavy load
```

## Common Issues and Solutions

### Issue 1: High Memory Usage

**Symptom**: OOM errors with multiple model instances

**Solutions**:
1. Use single model with sequential processing
2. Reduce number of workers
3. Use more aggressive quantization
4. Consider model splitting across devices

### Issue 2: Poor GPU Utilization

**Symptom**: GPU usage <50% during inference

**Solutions**:
1. Increase batch size
2. Use proper batching framework (vLLM)
3. Group prompts by length
4. Reduce CPU-GPU transfer overhead

### Issue 3: High Latency

**Symptom**: Individual requests take too long

**Solutions**:
1. Reduce batch size
2. Use dynamic batching
3. Prioritize latency-sensitive requests
4. Consider multiple model replicas

### Issue 4: Inconsistent Throughput

**Symptom**: Throughput varies widely

**Solutions**:
1. Implement request buffering
2. Use consistent batch sizes
3. Profile and identify bottlenecks
4. Monitor queue depths

## Interview Topics

**Batching Strategy**:
> Q: "How would you design an inference API to handle 1000 requests per second?"
>
> A: Use dynamic batching with request queueing. Key components:
> 1. Request queue with priorities
> 2. Batching engine that forms optimal batches (8-16 requests)
> 3. Multiple model instances for parallelism
> 4. Load balancer across instances
> 5. Monitoring and auto-scaling
> 6. Consider vLLM or TensorRT-LLM for production

**Trade-offs**:
> Q: "Explain the trade-offs between throughput and latency in batched inference."
>
> A: Larger batches increase throughput (more requests per second) but increase latency per request (each waits for batch to complete). Small batches have lower latency but lower throughput. Production systems use dynamic batching to balance: small batches during light load (low latency), larger during heavy load (high throughput).

**Optimization**:
> Q: "How would you optimize batch inference for a mix of short and long prompts?"
>
> A: Group by length into separate batches. Short prompts in one batch (less padding), long in another. Prevents wasting computation on padding. Can improve throughput 2-5x. May need multiple processing queues and dynamic routing.

## Advanced Topics

### Speculative Decoding
```
Use small model to generate candidates
Large model verifies in parallel
Can speed up generation 2-3x
```

### Continuous Batching
```
Add/remove requests from batches dynamically
Maintains high utilization
Used by vLLM, TGI
```

### Multi-GPU Batching
```
Split batch across GPUs
Tensor parallelism or pipeline parallelism
Handle very large batches
```

## Next Steps

After completing this example:
- ✅ Experiment with llama-server for real batching
- ✅ Try vLLM for production-grade inference
- ✅ Implement a production API with queueing
- ✅ Profile different batching strategies
- ✅ Move on to Module 1.5 or Module 2

## Related Documentation

- [Module 5: Advanced Inference](../../modules/05-advanced-inference/docs/)
- [Module 6: Server & Production](../../modules/06-server-production/docs/)
- [Lab 6.2: Building an Inference API](../../modules/06-server-production/labs/)

## External Resources

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - PagedAttention
- [Orca Paper](https://www.usenix.org/conference/osdi22/presentation/yu) - Continuous batching
- [llama.cpp Server Documentation](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)

---

**Author**: Agent 3 (Code Developer)
**Last Updated**: 2025-11-18
**Module**: 1.4 - Basic Inference

**Congratulations!** You've completed all 5 basic inference examples. You're now ready for more advanced topics or hands-on labs.

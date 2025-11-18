# Lab 2: Batch Optimization and Parallel Inference

**Module 5 - Advanced Inference**
**Estimated Time**: 2-3 hours
**Difficulty**: Advanced

## Objectives

By the end of this lab, you will:
- Implement and compare static vs dynamic batching
- Optimize batch size for maximum throughput
- Measure GPU utilization across different batch sizes
- Handle variable-length sequences efficiently
- Build a production-ready batching system

## Prerequisites

- Completed Module 4 (GPU Acceleration)
- llama.cpp built with CUDA support
- Python 3.8+ with numpy, matplotlib
- NVIDIA GPU with CUDA

## Part 1: Baseline Single Request (20 minutes)

### Measure Sequential Performance

```bash
cd ~/llama.cpp-learn

# Single request benchmark
time ./llama-cli \
    -m models/llama-2-13b.Q4_K_M.gguf \
    -p "Explain machine learning" \
    -n 100 \
    --log-disable

# Record: Time: _______s
```

Now run 8 requests sequentially:

```bash
for i in {1..8}; do
    echo "Request $i"
    ./llama-cli \
        -m models/llama-2-13b.Q4_K_M.gguf \
        -p "Explain AI topic $i" \
        -n 100 \
        --log-disable
done
```

**Question 1**: What is the total time for 8 sequential requests?

**Question 2**: What is the throughput (total tokens / total time)?

## Part 2: Static Batching (45 minutes)

### Using llama-parallel

llama.cpp includes a parallel processing tool:

```bash
# Create test prompts
cat > batch_prompts.txt << EOF
Explain machine learning in detail
Describe deep learning architectures
What are neural networks
How do transformers work
Explain attention mechanisms
Describe backpropagation
What is gradient descent
Explain overfitting and regularization
EOF

# Run with parallel processing (static batch)
time ./llama-parallel \
    -m models/llama-2-13b.Q4_K_M.gguf \
    -f batch_prompts.txt \
    -n 100 \
    -np 8 \  # Number of parallel sequences
    --log-disable
```

**Task**: Record results:
- Total time: _______s
- Total tokens: _______
- Throughput: _______ tok/s

### Compare with Sequential

```python
import matplotlib.pyplot as plt

methods = ['Sequential', 'Batched']
times = [___,  ___]  # Your measured times
throughputs = [___, ___]  # Your measured throughputs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(methods, times)
ax1.set_ylabel('Time (s)')
ax1.set_title('Processing Time')

ax2.bar(methods, throughputs)
ax2.set_ylabel('Throughput (tok/s)')
ax2.set_title('Throughput Comparison')

plt.tight_layout()
plt.savefig('batch_comparison.png')
```

**Question 3**: What speedup did you achieve with batching?

**Question 4**: Why isn't the speedup 8x despite having 8 sequences?

### GPU Utilization Monitoring

Monitor GPU usage during batching:

```bash
# In one terminal, monitor GPU
watch -n 0.5 nvidia-smi

# In another terminal, run batched inference
./llama-parallel \
    -m models/llama-2-13b.Q4_K_M.gguf \
    -f batch_prompts.txt \
    -n 100 \
    -np 8
```

**Task**: Record GPU metrics:
- GPU utilization: _______%
- Memory used: _______GB
- Memory utilization: _______%

**Question 5**: Is your GPU compute-bound or memory-bound?

## Part 3: Batch Size Optimization (45 minutes)

### Find Optimal Batch Size

Test different batch sizes:

```bash
#!/bin/bash
# save as test_batch_sizes.sh

for NP in 1 2 4 8 16 32; do
    echo "Testing batch size: $NP"

    # Create prompts file with NP prompts
    > test_prompts.txt
    for i in $(seq 1 $NP); do
        echo "Explain topic number $i" >> test_prompts.txt
    done

    # Benchmark
    TIME=$(time ./llama-parallel \
        -m models/llama-2-13b.Q4_K_M.gguf \
        -f test_prompts.txt \
        -n 50 \
        -np $NP \
        --log-disable 2>&1 | grep real | awk '{print $2}')

    echo "Batch size $NP: $TIME"
    echo "---"
done
```

**Task**: Fill in the table:

| Batch Size | Time (s) | Throughput (tok/s) | GPU Memory (GB) | Notes |
|-----------|----------|-------------------|----------------|-------|
| 1 | | | | Baseline |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |
| 32 | | | | OOM? |

### Analyze Results

```python
import numpy as np
import matplotlib.pyplot as plt

batch_sizes = [1, 2, 4, 8, 16, 32]
throughputs = [___]  # Your measurements
memory_usage = [___]  # GB

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Throughput vs Batch Size
ax1.plot(batch_sizes, throughputs, marker='o', linewidth=2)
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Throughput (tok/s)')
ax1.set_title('Throughput Scaling')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# Memory vs Batch Size
ax2.plot(batch_sizes, memory_usage, marker='s', linewidth=2, color='red')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('GPU Memory (GB)')
ax2.set_title('Memory Usage')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

plt.tight_layout()
plt.savefig('batch_size_analysis.png')
```

**Question 6**: What is the optimal batch size for your GPU?

**Question 7**: Does throughput scale linearly with batch size? Explain.

**Question 8**: What limits the maximum batch size?

## Part 4: Variable-Length Sequences (45 minutes)

### The Padding Problem

Create prompts with varying lengths:

```python
# create_varied_prompts.py
import random

prompts = [
    "AI" * random.randint(5, 50)  # Varies from ~10 to ~100 words
    for _ in range(8)
]

# Write to file
with open('varied_prompts.txt', 'w') as f:
    for p in prompts:
        f.write(p + '\n')

# Print stats
lengths = [len(p.split()) for p in prompts]
print(f"Prompt lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
```

### Measure Padding Waste

```bash
# Run with varied lengths
time ./llama-parallel \
    -m models/llama-2-13b.Q4_K_M.gguf \
    -f varied_prompts.txt \
    -n 50 \
    -np 8 \
    --log-disable
```

**Task**: Calculate padding waste:

```python
def calculate_padding_waste(lengths, max_tokens_per_seq):
    """
    lengths: List of prompt lengths
    max_tokens_per_seq: Fixed generation length
    """
    # In static batching, all wait for longest
    max_length = max(lengths) + max_tokens_per_seq

    # Actual work
    actual_work = sum(l + max_tokens_per_seq for l in lengths)

    # Padded work (all sequences process max_length steps)
    padded_work = max_length * len(lengths)

    waste_pct = (padded_work - actual_work) / padded_work * 100

    return waste_pct

prompt_lengths = [___]  # Your measured lengths
waste = calculate_padding_waste(prompt_lengths, 50)
print(f"Padding waste: {waste:.1f}%")
```

**Question 9**: What percentage of compute is wasted due to padding?

### Bucketing Strategy

Implement bucketing to reduce waste:

```python
def bucket_requests(requests, num_buckets=3):
    """
    Group requests by similar length
    """
    # Sort by length
    sorted_reqs = sorted(requests, key=len)

    # Create buckets
    bucket_size = len(requests) // num_buckets
    buckets = []

    for i in range(0, len(sorted_reqs), bucket_size):
        bucket = sorted_reqs[i:i+bucket_size]
        buckets.append(bucket)

    return buckets

# Test bucketing
requests = ["short", "medium length", "very long prompt here"]
bucketed = bucket_requests(requests, num_buckets=3)

for i, bucket in enumerate(bucketed):
    print(f"Bucket {i}: {bucket}")
```

**Question 10**: How much does bucketing reduce padding waste?

## Part 5: Dynamic Batching (45 minutes)

### Implement Simple Dynamic Batching

Use the provided code:

```python
# See: ../code/parallel_batch_processing.py

# Run dynamic batching benchmark
python ../code/parallel_batch_processing.py
```

### Compare Static vs Dynamic

**Task**: Analyze the output and fill in:

| Metric | Static Batching | Dynamic Batching | Improvement |
|--------|----------------|------------------|-------------|
| Throughput | | | |
| Avg Latency | | | |
| P95 Latency | | | |
| GPU Utilization | | | |

**Question 11**: Why does dynamic batching have better latency?

**Question 12**: When would static batching be preferable to dynamic?

## Part 6: Production Batching System (Optional, 60 minutes)

### Design a Request Queue

Implement a production-ready batching system:

```python
import asyncio
import time
from dataclasses import dataclass
from typing import List

@dataclass
class Request:
    id: int
    prompt: str
    max_tokens: int
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class ProductionBatchScheduler:
    def __init__(
        self,
        max_batch_size: int = 16,
        max_wait_ms: float = 50.0
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()

        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.batch_sizes = []

    async def submit_request(self, request: Request):
        """Submit a new request"""
        await self.queue.put(request)
        self.total_requests += 1

        # TODO: Wait for result
        result = await self._wait_for_result(request.id)
        return result

    async def process_batches(self):
        """Background task to process batches"""
        while True:
            batch = await self._collect_batch()

            if batch:
                # Process batch
                results = await self._process_batch(batch)

                # Return results
                for request, result in zip(batch, results):
                    self._complete_request(request.id, result)

    async def _collect_batch(self):
        """Collect requests into a batch"""
        batch = []
        deadline = time.time() + self.max_wait_ms / 1000.0

        while len(batch) < self.max_batch_size:
            timeout = max(0, deadline - time.time())

            try:
                request = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=timeout
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        self.batch_sizes.append(len(batch))
        self.total_batches += 1

        return batch

    async def _process_batch(self, batch: List[Request]):
        """Process a batch of requests (mock implementation)"""
        # TODO: Call actual inference
        await asyncio.sleep(0.1)  # Simulate inference
        return [f"Result for {req.id}" for req in batch]

    def get_metrics(self):
        """Get performance metrics"""
        import numpy as np
        return {
            'total_requests': self.total_requests,
            'total_batches': self.total_batches,
            'avg_batch_size': np.mean(self.batch_sizes),
            'batch_efficiency': np.mean(self.batch_sizes) / self.max_batch_size
        }

# Test the scheduler
async def test_scheduler():
    scheduler = ProductionBatchScheduler(max_batch_size=8, max_wait_ms=50)

    # Start processing
    asyncio.create_task(scheduler.process_batches())

    # Submit requests
    requests = [
        Request(i, f"Prompt {i}", 100)
        for i in range(32)
    ]

    results = await asyncio.gather(*[
        scheduler.submit_request(req)
        for req in requests
    ])

    # Print metrics
    metrics = scheduler.get_metrics()
    print(f"Metrics: {metrics}")

# Run test
# asyncio.run(test_scheduler())
```

**Challenge**: Complete the implementation to:
1. Handle request timeouts
2. Implement priority queues
3. Add fair scheduling across users
4. Monitor and log performance metrics

## Deliverables

Submit a report containing:

1. **Benchmarking Results**
   - Sequential vs batched performance
   - Optimal batch size for your hardware
   - GPU utilization measurements

2. **Analysis**
   - Throughput scaling analysis
   - Memory usage patterns
   - Padding waste calculations

3. **Graphs**
   - Batch size vs throughput plot
   - Memory usage vs batch size plot
   - Static vs dynamic batching comparison

4. **Answers to Questions**
   - All 12 questions with detailed explanations

5. **Production Recommendations** (1-2 pages)
   - Recommended batch size
   - Strategies for variable-length sequences
   - When to use dynamic vs static batching

## Evaluation Criteria

- **Correctness** (40%): Accurate measurements and calculations
- **Analysis** (30%): Insightful explanations
- **Completeness** (20%): All sections attempted
- **Code Quality** (10%): Clean, well-documented code

## Extensions (Optional)

1. **Multi-GPU Batching**: Distribute batches across multiple GPUs
2. **Priority Scheduling**: Implement QoS tiers
3. **Adaptive Batch Sizing**: Auto-tune based on load
4. **Prefill Chunking**: Split long prompts into chunks

## Resources

- llama.cpp parallel docs: `examples/parallel/README.md`
- Code examples: `../code/parallel_batch_processing.py`
- CUDA batch processing guide: NVIDIA docs

## Tips

- Start with batch_size=8 as baseline
- Monitor `nvidia-smi` during experiments
- Use uniform-length sequences for initial tests
- Account for warmup time in measurements
- Profile with `nsys` for detailed analysis

Good luck! 🚀

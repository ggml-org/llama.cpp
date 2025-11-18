# Lab 1: Speculative Decoding Experiments

**Module 5 - Advanced Inference**
**Estimated Time**: 2-3 hours
**Difficulty**: Advanced

## Objectives

By the end of this lab, you will:
- Implement speculative decoding from scratch
- Benchmark speedup with different draft/target model pairs
- Analyze acceptance rates under various conditions
- Optimize K (draft tokens per iteration) for your hardware
- Understand when speculative decoding helps vs hurts

## Prerequisites

- Completed Module 4 (GPU Acceleration)
- llama.cpp built with CUDA support
- Two models: one large (target), one small (draft)
- Python 3.8+ with numpy

## Part 1: Setup and Baseline (30 minutes)

### Download Models

```bash
# Navigate to models directory
cd ~/llama.cpp-learn/models

# Download draft model (7B)
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Download target model (13B)
wget https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_M.gguf
```

### Baseline Performance

Measure standard inference performance:

```bash
# Benchmark target model alone
time ./llama-cli \
    -m models/llama-2-13b.Q4_K_M.gguf \
    -p "Write a detailed explanation of quantum computing" \
    -n 100 \
    --temp 0.7 \
    --log-disable

# Record:
# - Total time: _______
# - Tokens/sec: _______
```

**Question 1**: What is your baseline throughput (tokens/sec)?

## Part 2: Basic Speculative Decoding (45 minutes)

### Run with Speculative Decoding

```bash
# With draft model
time ./llama-cli \
    -m models/llama-2-13b.Q4_K_M.gguf \
    --draft models/llama-2-7b.Q4_K_M.gguf \
    -p "Write a detailed explanation of quantum computing" \
    -n 100 \
    --temp 0.7 \
    --log-disable

# Record:
# - Total time: _______
# - Tokens/sec: _______
# - Speedup: _______
```

### Calculate Speedup

```python
baseline_time = ___  # Your baseline time
spec_time = ___      # Speculative decoding time

speedup = baseline_time / spec_time
print(f"Speedup: {speedup:.2f}x")
```

**Question 2**: What speedup did you achieve?

**Question 3**: Is it close to the theoretical maximum? Why or why not?

### Measure Acceptance Rate

Enable debug output to see acceptance rate:

```bash
./llama-cli \
    -m models/llama-2-13b.Q4_K_M.gguf \
    --draft models/llama-2-7b.Q4_K_M.gguf \
    -p "Write a detailed explanation of quantum computing" \
    -n 100 \
    --temp 0.7 \
    --draft-debug
```

Look for output like:
```
draft: accepted 3/4 tokens (75.0%)
draft: accepted 4/4 tokens (100.0%)
draft: accepted 2/4 tokens (50.0%)
...
```

**Task**: Calculate average acceptance rate over all iterations.

**Question 4**: What is your average acceptance rate?

## Part 3: Optimize K Parameter (45 minutes)

### Experiment with Different K Values

Test K = 2, 4, 6, 8 to find optimal value:

```bash
# Create benchmark script
cat > benchmark_k.sh << 'EOF'
#!/bin/bash

for K in 2 4 6 8; do
    echo "Testing K=$K"

    time ./llama-cli \
        -m models/llama-2-13b.Q4_K_M.gguf \
        --draft models/llama-2-7b.Q4_K_M.gguf \
        --draft-n $K \
        -p "Explain artificial intelligence" \
        -n 100 \
        --temp 0.7 \
        --log-disable \
        2>&1 | grep "tokens per second"

    echo "---"
done
EOF

chmod +x benchmark_k.sh
./benchmark_k.sh
```

**Task**: Fill in the table:

| K | Time (s) | Tokens/sec | Speedup |
|---|----------|------------|---------|
| 2 | | | |
| 4 | | | |
| 6 | | | |
| 8 | | | |

**Question 5**: What is the optimal K for your setup?

**Question 6**: Why doesn't K=8 give 2x speedup over K=4?

### Analyze the Trade-offs

Plot the results:

```python
import matplotlib.pyplot as plt

K_values = [2, 4, 6, 8]
throughputs = [___]  # Fill in your measured values

plt.figure(figsize=(10, 6))
plt.plot(K_values, throughputs, marker='o', linewidth=2, markersize=8)
plt.xlabel('K (Draft Tokens)')
plt.ylabel('Throughput (tokens/sec)')
plt.title('Speculative Decoding: K vs Throughput')
plt.grid(True, alpha=0.3)
plt.savefig('k_optimization.png')
plt.show()
```

## Part 4: Temperature Effects (30 minutes)

### Test Different Temperatures

Acceptance rate depends on temperature:

```bash
for TEMP in 0.1 0.5 0.7 1.0 1.5; do
    echo "Temperature: $TEMP"

    ./llama-cli \
        -m models/llama-2-13b.Q4_K_M.gguf \
        --draft models/llama-2-7b.Q4_K_M.gguf \
        -p "Explain deep learning" \
        -n 50 \
        --temp $TEMP \
        --draft-debug \
        2>&1 | grep "accepted"

    echo "---"
done
```

**Task**: Record acceptance rates:

| Temperature | Avg Acceptance | Throughput | Notes |
|-------------|----------------|------------|-------|
| 0.1 | | | Focused |
| 0.5 | | | Balanced |
| 0.7 | | | Default |
| 1.0 | | | Creative |
| 1.5 | | | Very random |

**Question 7**: How does temperature affect acceptance rate? Explain why.

**Question 8**: At what temperature does speculative decoding stop being beneficial?

## Part 5: Model Alignment Analysis (30 minutes)

### Test Misaligned Models

What happens with different model families?

```bash
# If you have a Mistral model, test it as draft for LLaMA target
# (This should show poor alignment)

./llama-cli \
    -m models/llama-2-13b.Q4_K_M.gguf \
    --draft models/mistral-7b.Q4_K_M.gguf \  # If available
    -p "Explain neural networks" \
    -n 100 \
    --draft-debug

# Compare to aligned models
./llama-cli \
    -m models/llama-2-13b.Q4_K_M.gguf \
    --draft models/llama-2-7b.Q4_K_M.gguf \
    -p "Explain neural networks" \
    -n 100 \
    --draft-debug
```

**Question 9**: Why is model alignment important for speculative decoding?

**Question 10**: What acceptance rate would make speculative decoding slower than baseline?

## Part 6: Custom Implementation (Optional, 45 minutes)

### Implement Your Own Speculative Decoder

Use the provided Python code:

```python
# See: ../code/speculative_decoding.py

# Run it
python ../code/speculative_decoding.py

# Modify it to experiment with:
# 1. Different acceptance thresholds
# 2. Adaptive K selection
# 3. Multi-draft models
```

**Challenge**: Implement adaptive K selection based on recent acceptance rate:

```python
def adaptive_speculative_decode(draft, target, prompt, initial_K=4):
    """
    Dynamically adjust K based on acceptance rate
    """
    K = initial_K
    acceptance_history = []

    # TODO: Implement adaptive K logic
    # If acceptance_rate > 0.8: increase K
    # If acceptance_rate < 0.5: decrease K
```

## Part 7: Production Scenarios (30 minutes)

### Scenario 1: Batch Processing

Test speculative decoding with batching:

```bash
# Create multiple prompts
cat > prompts.txt << EOF
Explain quantum computing
Describe machine learning
What is blockchain
How do neural networks work
EOF

# Run with batching
./llama-parallel \
    -m models/llama-2-13b.Q4_K_M.gguf \
    --draft models/llama-2-7b.Q4_K_M.gguf \
    -f prompts.txt \
    -n 100
```

**Question 11**: Does speculative decoding work well with batching? Why or why not?

### Scenario 2: Memory Constraints

Calculate memory requirements:

```python
def calculate_memory(draft_params, target_params, batch_size=1):
    """
    Calculate memory for speculative decoding

    draft_params: Draft model parameters (billions)
    target_params: Target model parameters (billions)
    """
    # Model weights (fp16)
    draft_mem_gb = draft_params * 2
    target_mem_gb = target_params * 2

    # KV cache (simplified)
    kv_cache_gb = (draft_params + target_params) * 0.1 * batch_size

    total_gb = draft_mem_gb + target_mem_gb + kv_cache_gb

    return {
        'draft_model': draft_mem_gb,
        'target_model': target_mem_gb,
        'kv_cache': kv_cache_gb,
        'total': total_gb
    }

# Example: 7B draft + 70B target
mem = calculate_memory(7, 70)
print(f"Total memory needed: {mem['total']:.1f} GB")
```

**Question 12**: What's the memory overhead of speculative decoding compared to baseline?

## Deliverables

Submit a report containing:

1. **Benchmarking Results**
   - Baseline vs speculative throughput
   - Optimal K value for your hardware
   - Acceptance rates for different temperatures

2. **Analysis**
   - Speedup calculations and explanations
   - Trade-offs between K values
   - Temperature effects on acceptance rate

3. **Graphs**
   - K vs Throughput plot
   - Temperature vs Acceptance Rate plot
   - Speedup comparison chart

4. **Answers to Questions**
   - All 12 questions answered with detailed explanations

5. **Production Recommendations** (1-2 paragraphs)
   - When to use speculative decoding
   - Recommended parameter settings
   - Potential pitfalls to avoid

## Evaluation Criteria

- **Correctness** (40%): Accurate measurements and calculations
- **Analysis** (30%): Insightful explanations of results
- **Completeness** (20%): All sections attempted
- **Presentation** (10%): Clear, well-organized report

## Extensions (Optional)

For extra credit:

1. **Multi-Draft Models**: Test using multiple draft models and picking the best
2. **Adaptive Speculation**: Implement runtime K adjustment
3. **Tree-Based Speculation**: Explore branching speculation strategies
4. **Production Deployment**: Design a speculative decoding service API

## Resources

- llama.cpp speculative decoding docs: `examples/speculative/README.md`
- Original paper: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
- Code examples: `../code/speculative_decoding.py`

## Tips

- Start with K=4 as a baseline
- Use temperature ≤ 0.7 for best acceptance rates
- Monitor GPU memory usage with `nvidia-smi`
- Same model family (draft and target) is crucial
- Quantization level should match between draft and target

Good luck! 🚀

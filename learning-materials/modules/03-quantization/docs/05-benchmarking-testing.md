# Benchmarking & Testing Guide

**Module 3, Lesson 5** | **Estimated Time**: 2 hours | **Difficulty**: Intermediate to Advanced

## Table of Contents
1. [Introduction](#introduction)
2. [Benchmarking Fundamentals](#benchmarking-fundamentals)
3. [Performance Metrics](#performance-metrics)
4. [Quality Metrics](#quality-metrics)
5. [Perplexity Measurement](#perplexity-measurement)
6. [Benchmark Suites](#benchmark-suites)
7. [Statistical Significance](#statistical-significance)
8. [Continuous Benchmarking](#continuous-benchmarking)
9. [Best Practices](#best-practices)
10. [Interview Questions](#interview-questions)

---

## Introduction

Benchmarking is essential for optimizing LLM inference. You need to measure both **performance** (speed, throughput) and **quality** (accuracy, perplexity) to make informed decisions about quantization and optimization.

**Learning Objectives:**
- Design effective benchmark experiments
- Measure perplexity and other quality metrics
- Analyze performance characteristics
- Conduct statistically valid comparisons
- Build automated benchmark pipelines

**Prerequisites:**
- Understanding of quantization
- Basic statistics knowledge
- Familiarity with llama.cpp tools

---

## Benchmarking Fundamentals

### Why Benchmark?

1. **Validate optimizations**: Ensure changes improve performance
2. **Compare configurations**: Choose best quantization, hardware, etc.
3. **Detect regressions**: Catch performance degradation
4. **Guide decisions**: Data-driven optimization priorities
5. **Report results**: Communicate improvements to stakeholders

### Types of Benchmarks

#### 1. Microbenchmarks

Test individual operations:
```bash
# Benchmark specific operation
./llama-bench -m model.gguf -op matmul -n 1000
```

**Use for:**
- Optimizing specific functions
- Comparing SIMD implementations
- Testing hardware-specific code paths

#### 2. End-to-End Benchmarks

Test complete inference pipeline:
```bash
# Full inference benchmark
./llama-bench -m model.gguf -p "Complete prompt" -n 512
```

**Use for:**
- Overall system performance
- Quantization comparisons
- Production performance estimation

#### 3. Regression Benchmarks

Track performance over time:
```bash
# Run nightly benchmarks
./run_benchmarks.sh --store-results results/2025-11-18.json
```

**Use for:**
- Catching performance regressions
- Long-term tracking
- CI/CD integration

---

## Performance Metrics

### 1. Tokens Per Second (TPS)

Primary inference speed metric:

```
TPS = number_of_tokens / inference_time_seconds
```

**Example:**
```bash
# Generate 100 tokens
time ./llama-cli -m model.gguf -p "Test" -n 100

# Output: Generated 100 tokens in 5.2 seconds
# TPS = 100 / 5.2 = 19.2 tokens/sec
```

**Considerations:**
- **Prompt processing** vs **token generation**: Different speeds
- **First token latency**: Important for user experience
- **Sustained throughput**: After warmup

### 2. Throughput vs Latency

**Latency**: Time to process single request
**Throughput**: Requests processed per unit time

```
Single request: 50ms latency
Batch of 8:     200ms latency, 8 requests / 0.2s = 40 req/sec throughput
```

**Trade-off:**
- Low latency: Small batches (better UX)
- High throughput: Large batches (better efficiency)

### 3. Time To First Token (TTFT)

Critical for interactive applications:

```bash
# Measure prompt processing time
./llama-cli -m model.gguf -p "Long prompt..." --log-timing

# Look for: "prompt_eval_time"
```

**Impact factors:**
- Prompt length
- KV cache initialization
- Model size

### 4. Memory Bandwidth Utilization

```bash
# Monitor memory bandwidth
sudo perf stat -e \
    memory_bandwidth_read,\
    memory_bandwidth_write \
    ./llama-cli -m model.gguf -p "Test" -n 100

# Calculate utilization:
# actual_bandwidth / max_theoretical_bandwidth
```

### 5. GPU Utilization

For GPU inference:

```bash
# NVIDIA GPU
nvidia-smi dmon -s u

# Look for:
# - GPU utilization (should be >80% for compute-bound)
# - Memory utilization
# - Power consumption
```

---

## Quality Metrics

### 1. Perplexity

Most common quality metric for language models:

```
Perplexity = exp(cross_entropy_loss)
```

**Interpretation:**
- Lower is better
- Measures how "surprised" the model is by test data
- Perplexity of 10 = model assigns ~10% probability to correct token on average

**Example:**
```bash
./llama-perplexity \
    -m model-q4_k_m.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    --perplexity

# Output:
# Final perplexity: 5.68 +/- 0.03
```

### 2. Benchmark Task Performance

#### Common Benchmarks:

**MMLU (Massive Multitask Language Understanding)**:
- 57 tasks across diverse subjects
- Multiple choice questions
- Measures broad knowledge

**HellaSwag**:
- Commonsense reasoning
- Sentence completion
- Measures understanding

**TruthfulQA**:
- Factual accuracy
- Resistance to misconceptions
- Important for reliability

**GSM8K**:
- Grade school math problems
- Tests reasoning ability
- Sensitive to quantization

**HumanEval**:
- Code generation
- Python programming problems
- Critical for coding models

### 3. Task-Specific Metrics

#### Text Generation:
- **BLEU**: Machine translation quality
- **ROUGE**: Summarization quality
- **METEOR**: Semantic similarity

#### Question Answering:
- **Exact Match (EM)**: Exact string match
- **F1 Score**: Token overlap

#### Code Generation:
- **Pass@k**: Proportion passing tests
- **Compilation rate**: Syntactically correct
- **Functional correctness**: Passes test cases

---

## Perplexity Measurement

### Using llama-perplexity

```bash
# Basic perplexity test
./llama-perplexity \
    -m model.gguf \
    -f test_data.txt \
    --perplexity

# With specific parameters
./llama-perplexity \
    -m model.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    --perplexity \
    -ngl 32 \        # GPU layers
    -t 8 \           # Threads
    -c 2048          # Context size
```

### Interpreting Results

```
Example output:
perplexity: 5.68 +/- 0.03
time:       125.5 s
samples:    148
tokens:     245823
```

**Good perplexity values** (LLaMA-7B on WikiText-2):
- FP16: ~5.68
- Q8_0: ~5.70 (<0.5% increase) ✅
- Q6_K: ~5.72 (~0.7% increase) ✅
- Q5_K_M: ~5.75 (~1.2% increase) ✅
- Q4_K_M: ~5.82 (~2.5% increase) ✅
- Q4_0: ~5.90 (~3.9% increase) ⚠️
- Q3_K_M: ~6.10 (~7.4% increase) ⚠️
- Q2_K: ~6.80 (~20% increase) ❌

### Custom Perplexity Testing

```python
import math
from transformers import AutoTokenizer

def calculate_perplexity(model, text):
    """Calculate perplexity on text"""
    tokenizer = AutoTokenizer.from_pretrained("model")
    tokens = tokenizer.encode(text)

    total_log_prob = 0
    count = 0

    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]

        # Get model prediction
        logits = model.forward(context)
        probs = softmax(logits)

        # Log probability of actual token
        log_prob = math.log(probs[target])
        total_log_prob += log_prob
        count += 1

    # Calculate perplexity
    avg_log_prob = total_log_prob / count
    perplexity = math.exp(-avg_log_prob)

    return perplexity
```

---

## Benchmark Suites

### llama.cpp Built-in Benchmarks

#### 1. llama-bench

Comprehensive benchmark tool:

```bash
# Basic benchmark
./llama-bench -m model.gguf

# Multiple configurations
./llama-bench \
    -m model-q4_k_m.gguf,model-q5_k_m.gguf,model-q8_0.gguf \
    -p 512 \           # Prompt length
    -n 128 \           # Generation length
    -t 4,8,16          # Thread counts

# Output format
# model                    | size |  pp 512 |   tg 128 |
# -------------------------|------|---------|----------|
# llama-7b-q4_k_m.gguf     | 4.1G | 45.2 ms | 15.3 t/s |
# llama-7b-q5_k_m.gguf     | 4.8G | 48.1 ms | 14.8 t/s |
# llama-7b-q8_0.gguf       | 7.2G | 52.3 ms | 13.9 t/s |
```

#### 2. Custom Benchmark Script

```bash
#!/bin/bash
# benchmark_suite.sh

MODELS=(
    "llama-7b-q4_k_m.gguf"
    "llama-7b-q5_k_m.gguf"
    "llama-7b-q8_0.gguf"
)

THREADS=(1 2 4 8 16)
PROMPTS=(
    "Short prompt"
    "Medium length prompt with more context to process"
    "Very long prompt with extensive context that will test the model's ability to handle larger inputs and maintain coherence across multiple sentences and ideas"
)

for model in "${MODELS[@]}"; do
    for threads in "${THREADS[@]}"; do
        for prompt in "${PROMPTS[@]}"; do
            echo "Testing: $model, $threads threads, prompt length ${#prompt}"

            ./llama-bench \
                -m "$model" \
                -t $threads \
                -p "$prompt" \
                -n 100 \
                -r 3 \
                --output benchmark_results.csv
        done
    done
done
```

### External Benchmark Frameworks

#### 1. lm-evaluation-harness

```bash
# Install
pip install lm-eval

# Run benchmarks
lm_eval --model llama-cpp \
        --model_args path=model.gguf,n_gpu_layers=32 \
        --tasks mmlu,hellaswag,truthfulqa \
        --batch_size 8 \
        --output_path results/

# Results:
# {
#   "mmlu": {"acc": 0.623},
#   "hellaswag": {"acc_norm": 0.789},
#   "truthfulqa": {"mc2": 0.412}
# }
```

#### 2. Custom Task Evaluation

```python
# evaluate_model.py
import json
from llama_cpp import Llama

def evaluate_gsm8k(model_path):
    """Evaluate on GSM8K math problems"""
    llm = Llama(model_path=model_path, n_ctx=2048)

    with open("gsm8k_test.json") as f:
        problems = json.load(f)

    correct = 0
    total = len(problems)

    for problem in problems:
        prompt = f"Question: {problem['question']}\nAnswer:"
        response = llm(prompt, max_tokens=256)

        # Extract answer and check
        if check_answer(response['choices'][0]['text'],
                       problem['answer']):
            correct += 1

    accuracy = correct / total
    print(f"GSM8K Accuracy: {accuracy:.2%}")
    return accuracy
```

---

## Statistical Significance

### Why Statistics Matter

Single runs can be misleading due to:
- Noise (system load, thermal throttling)
- Cache effects (warm vs cold start)
- Random initialization

### Running Multiple Trials

```bash
# Run benchmark 10 times
for i in {1..10}; do
    ./llama-bench -m model.gguf -n 100 >> results.txt
done

# Analyze results
python analyze_results.py results.txt
```

### Statistical Analysis

```python
import numpy as np
from scipy import stats

def analyze_benchmark_results(measurements):
    """Analyze benchmark measurements"""

    # Calculate statistics
    mean = np.mean(measurements)
    std = np.std(measurements)
    sem = stats.sem(measurements)  # Standard error of mean

    # 95% confidence interval
    confidence = 0.95
    ci = stats.t.interval(confidence, len(measurements)-1,
                          loc=mean, scale=sem)

    print(f"Mean: {mean:.2f}")
    print(f"Std Dev: {std:.2f}")
    print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    return mean, std, ci

# Example
results_q4 = [15.2, 15.5, 15.1, 15.3, 15.4]  # tokens/sec
results_q8 = [13.8, 14.1, 13.9, 14.0, 13.8]

analyze_benchmark_results(results_q4)
# Mean: 15.30
# Std Dev: 0.15
# 95% CI: [15.11, 15.49]
```

### Comparing Two Configurations

```python
def compare_benchmarks(results_a, results_b, alpha=0.05):
    """Statistical comparison of two configurations"""

    # T-test for independent samples
    t_stat, p_value = stats.ttest_ind(results_a, results_b)

    is_significant = p_value < alpha

    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant: {is_significant}")

    # Effect size (Cohen's d)
    mean_a = np.mean(results_a)
    mean_b = np.mean(results_b)
    pooled_std = np.sqrt((np.var(results_a) + np.var(results_b)) / 2)
    cohens_d = (mean_a - mean_b) / pooled_std

    print(f"Effect size (Cohen's d): {cohens_d:.3f}")

    return is_significant, cohens_d

# Example
results_baseline = [15.2, 15.5, 15.1, 15.3, 15.4]
results_optimized = [17.1, 17.3, 16.9, 17.2, 17.0]

compare_benchmarks(results_baseline, results_optimized)
# T-statistic: -13.856
# P-value: 0.0001
# Significant: True
# Effect size (Cohen's d): -11.000 (very large improvement)
```

---

## Continuous Benchmarking

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Build llama.cpp
        run: |
          cmake -B build -DLLAMA_BUILD_TESTS=ON
          cmake --build build --config Release

      - name: Download test model
        run: |
          wget https://huggingface.co/models/test-model-q4.gguf

      - name: Run benchmarks
        run: |
          ./build/bin/llama-bench \
            -m test-model-q4.gguf \
            -p 512 -n 128 -r 5 \
            --output results.json

      - name: Compare with baseline
        run: |
          python scripts/compare_benchmarks.py \
            results.json \
            baseline.json \
            --threshold 0.05  # 5% regression threshold

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results.json
```

### Regression Detection

```python
# compare_benchmarks.py
import json
import sys

def detect_regression(current, baseline, threshold=0.05):
    """Detect performance regression"""

    current_tps = current['tokens_per_second']
    baseline_tps = baseline['tokens_per_second']

    change = (current_tps - baseline_tps) / baseline_tps

    if change < -threshold:
        print(f"❌ REGRESSION DETECTED: {change*100:.1f}% slower")
        print(f"   Current: {current_tps:.2f} t/s")
        print(f"   Baseline: {baseline_tps:.2f} t/s")
        sys.exit(1)
    elif change > threshold:
        print(f"✅ IMPROVEMENT: {change*100:.1f}% faster")
    else:
        print(f"✅ STABLE: {change*100:.1f}% change (within threshold)")

    return change

# Load results
with open(sys.argv[1]) as f:
    current = json.load(f)
with open(sys.argv[2]) as f:
    baseline = json.load(f)

detect_regression(current, baseline, threshold=0.05)
```

### Performance Dashboard

```python
# dashboard.py - Streamlit app
import streamlit as st
import pandas as pd
import plotly.express as px

# Load benchmark history
df = pd.read_csv("benchmark_history.csv")

st.title("LLaMA.cpp Performance Dashboard")

# Tokens per second over time
fig = px.line(df, x='date', y='tokens_per_second',
              color='quantization',
              title='Tokens/Second Over Time')
st.plotly_chart(fig)

# Perplexity comparison
fig = px.bar(df.groupby('quantization')['perplexity'].mean(),
             title='Average Perplexity by Quantization')
st.plotly_chart(fig)

# Model size vs performance
fig = px.scatter(df, x='model_size_gb', y='tokens_per_second',
                 color='quantization', size='perplexity',
                 title='Size vs Performance Trade-off')
st.plotly_chart(fig)
```

---

## Best Practices

### 1. Benchmark Design

**Do:**
- Run multiple trials (at least 5)
- Test with realistic workloads
- Include warmup runs
- Control for system variability
- Document hardware and configuration

**Don't:**
- Cherry-pick best results
- Run on loaded system
- Forget to disable turbo boost (for reproducibility)
- Compare different hardware directly

### 2. Reproducibility Checklist

```yaml
# benchmark_config.yaml
hardware:
  cpu: "Intel Xeon E5-2680 v4"
  ram: "64 GB DDR4-2400"
  gpu: "NVIDIA A100 40GB"
  storage: "NVMe SSD"

system:
  os: "Ubuntu 22.04 LTS"
  kernel: "5.15.0-generic"
  cpu_governor: "performance"
  turbo_boost: "disabled"

model:
  name: "llama-2-7b"
  quantization: "Q4_K_M"
  size_gb: 4.1

benchmark:
  tool: "llama-bench"
  version: "1.0.0"
  prompt_length: 512
  generation_length: 128
  batch_size: 1
  threads: 8
  trials: 10
```

### 3. Reporting Results

```markdown
## Benchmark Results

**Configuration:**
- Model: LLaMA-2-7B-Q4_K_M
- Hardware: AMD Ryzen 9 5950X, 32GB RAM
- Settings: 8 threads, batch size 1

**Performance:**
| Metric | Mean | Std Dev | 95% CI |
|--------|------|---------|--------|
| Prompt Processing | 45.2 ms | 2.1 ms | [43.8, 46.6] |
| Token Generation | 15.3 t/s | 0.4 t/s | [15.0, 15.6] |
| Memory Usage | 4.8 GB | 0.1 GB | [4.7, 4.9] |

**Quality:**
| Benchmark | Score | vs FP16 |
|-----------|-------|---------|
| Perplexity (WikiText-2) | 5.82 | +2.5% |
| MMLU | 62.3% | -0.8% |
| HellaSwag | 78.5% | -1.2% |

**Conclusion:** Q4_K_M provides 3.4x speedup over FP16 with minimal quality loss.
```

---

## Interview Questions

### Conceptual

1. **Q: What is perplexity and how do you interpret it?**

   A: Perplexity measures how well a model predicts text:
   ```
   Perplexity = exp(average cross-entropy loss)
   = exp(-Σ log P(token_i | context_i) / N)
   ```

   Interpretation:
   - Lower is better
   - Perplexity of 10 ≈ model is as uncertain as choosing from 10 equally likely options
   - 5% perplexity increase ≈ acceptable for quantization
   - 20% increase ≈ noticeable quality degradation

   Limitations:
   - Doesn't measure factual accuracy
   - Task-agnostic (may not reflect real performance)
   - Sensitive to domain of test data

2. **Q: Why is statistical significance important in benchmarking?**

   A: Single measurements can be misleading due to:
   - System noise (background processes, thermal variation)
   - Measurement error
   - Random variation in execution

   Statistical testing:
   - Determines if difference is real vs random chance
   - Quantifies uncertainty (confidence intervals)
   - Prevents false conclusions from noise

   Example: 15.3 vs 15.1 tokens/sec might be noise, but 15.3 vs 17.2 with p<0.01 is real improvement.

3. **Q: What's the difference between latency and throughput? When does each matter?**

   A:
   - **Latency**: Time for single request (ms)
   - **Throughput**: Requests per time unit (req/sec or tokens/sec)

   Trade-off:
   ```
   Batch size 1: Low latency (50ms), low throughput (20 req/s)
   Batch size 16: High latency (400ms), high throughput (40 req/s)
   ```

   When each matters:
   - **Interactive apps** (chat, autocomplete): Latency critical
   - **Batch processing** (translation, summarization): Throughput critical
   - **API servers**: Balance both with dynamic batching

### Practical

4. **Q: You observe 20% performance variation between benchmark runs. How do you diagnose the cause?**

   A: Systematic investigation:
   ```bash
   # 1. Check system load
   top, htop  # Other processes consuming resources?

   # 2. Check thermal throttling
   sensors  # CPU temperature stable?
   cat /proc/cpuinfo | grep MHz  # Frequency stable?

   # 3. Check CPU governor
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   # Should be "performance" for benchmarking

   # 4. Check memory
   free -h  # Swapping occurring?

   # 5. Run controlled test
   sudo nice -n -20 taskset -c 0-7 ./llama-bench ...
   # Pin to cores, high priority

   # 6. Measure variance
   for i in {1..20}; do
       ./llama-bench ... 2>&1 | grep "tokens/s"
   done | calculate_variance.py
   ```

   Common causes:
   - Background processes
   - Thermal throttling
   - Inconsistent CPU frequency
   - Memory swapping
   - NUMA effects

5. **Q: How would you design a benchmark to compare two quantization formats (Q4_K_M vs Q5_K_M)?**

   A: Comprehensive comparison:
   ```yaml
   1. Performance Benchmark:
      - Tool: llama-bench
      - Prompts: Short (128), Medium (512), Long (2048)
      - Generations: 128, 256 tokens
      - Batch sizes: 1, 4, 8
      - Threads: 1, 4, 8, 16
      - Trials: 10 each
      - Metrics: TPS, TTFT, memory

   2. Quality Benchmark:
      - Perplexity: WikiText-2, C4
      - Tasks: MMLU, HellaSwag, GSM8K
      - Human eval: 50 sample prompts

   3. Analysis:
      - Statistical tests (t-test)
      - Effect sizes
      - Cost-benefit: performance gain vs quality loss

   4. Recommendation:
      if quality_loss < 2% and speedup > 1.2x:
          recommend Q4_K_M
      else:
          recommend Q5_K_M
   ```

### Advanced

6. **Q: You need to benchmark a model that will serve 1000 QPS in production. How do you design a realistic benchmark?**

   A: Production-realistic setup:
   ```python
   # 1. Load testing
   from locust import HttpUser, task, between

   class LLMUser(HttpUser):
       wait_time = between(1, 5)  # Simulate real users

       @task
       def generate(self):
           self.client.post("/generate", json={
               "prompt": get_random_prompt(),
               "max_tokens": 128
           })

   # Run with 1000 concurrent users
   # locust -f loadtest.py --users 1000 --spawn-rate 10

   # 2. Monitor metrics
   - Response time (p50, p95, p99)
   - Error rate
   - Queue depth
   - GPU/CPU utilization
   - Memory usage

   # 3. Test failure modes
   - Spike to 2000 QPS
   - Long prompts (edge cases)
   - Sustained load (24 hours)

   # 4. Optimize based on bottleneck
   if gpu_util < 70%:
       increase_batch_size()
   if p99_latency > SLA:
       add_more_instances()
   ```

---

## Summary

**Key Takeaways:**

1. **Measure both performance and quality** - speed means nothing if quality degrades
2. **Use statistical rigor** - run multiple trials, calculate confidence intervals
3. **Benchmark realistically** - use production-like workloads and configurations
4. **Automate benchmarking** - continuous monitoring catches regressions early
5. **Report transparently** - document hardware, configuration, and methodology

**Essential Metrics:**
- Performance: Tokens/sec, latency, throughput
- Quality: Perplexity, task accuracy
- Efficiency: Memory usage, power consumption

**Tools:**
- llama-bench, llama-perplexity (built-in)
- lm-evaluation-harness (comprehensive benchmarks)
- Custom scripts (task-specific evaluation)

**Next Steps:**
- Lab 3: Comprehensive benchmarking lab
- Tutorial: Building a continuous benchmarking pipeline
- Module 3 Capstone: Optimization challenge

---

**Further Reading:**

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Perplexity Explained](https://huggingface.co/docs/transformers/perplexity)
- [Statistical Testing for ML](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
- [Load Testing Best Practices](https://locust.io/docs)

**Author**: Agent 5 (Documentation Specialist)
**Module**: 3 - Quantization & Optimization
**Last Updated**: 2025-11-18

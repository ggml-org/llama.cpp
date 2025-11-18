# Speculative Decoding: Accelerating LLM Inference

**Module 5, Lesson 1**
**Estimated Time**: 3-4 hours
**Difficulty**: Advanced

## Overview

Speculative decoding (also called assisted generation or draft-target decoding) is a revolutionary technique that can speed up LLM inference by 2-3x without changing the output distribution. This lesson explores how it works, when to use it, and how to implement it in llama.cpp.

## Learning Objectives

By the end of this lesson, you will:
- ✅ Understand the theory behind speculative decoding
- ✅ Implement draft-verify pipelines
- ✅ Optimize acceptance rates
- ✅ Measure and benchmark speedup gains
- ✅ Know when speculative decoding helps vs hurts

## What is Speculative Decoding?

### The Core Idea

Traditional autoregressive generation generates one token at a time:
```
Target Model: [prompt] → token₁ → token₂ → token₃ → ...
```

Speculative decoding uses a smaller, faster **draft model** to generate multiple tokens speculatively, then validates them with the larger **target model** in parallel:

```
1. Draft Model:   [prompt] → draft₁, draft₂, draft₃, draft₄
2. Target Model:  Verify all 4 drafts in parallel
3. Accept:        Keep valid tokens, reject rest
4. Repeat from step 1
```

### Why It Works

**Key Insight**: The target model can process multiple tokens in a single forward pass (parallel verification) much faster than generating them sequentially.

**Mathematical Guarantee**: The output distribution is **identical** to standard sampling - no quality degradation!

### Performance Characteristics

| Scenario | Speedup |
|----------|---------|
| High acceptance rate (>80%) | 2-3x faster |
| Medium acceptance rate (50-80%) | 1.5-2x faster |
| Low acceptance rate (<50%) | May be slower |

## Theory: How Speculative Decoding Works

### Algorithm Overview

```python
def speculative_decode(prompt, draft_model, target_model, K=4):
    """
    K: Number of speculative tokens to generate
    """
    tokens = tokenize(prompt)

    while not done:
        # Step 1: Draft model generates K tokens
        draft_tokens = []
        draft_probs = []

        for _ in range(K):
            draft_logits = draft_model(tokens + draft_tokens)
            draft_prob = softmax(draft_logits)
            draft_token = sample(draft_prob)
            draft_tokens.append(draft_token)
            draft_probs.append(draft_prob)

        # Step 2: Target model verifies all K tokens in ONE pass
        target_logits = target_model(tokens + draft_tokens[:-1])

        # Step 3: Accept/reject each token
        accepted = 0
        for i in range(K):
            target_prob = softmax(target_logits[i])

            # Acceptance criterion
            if should_accept(draft_tokens[i], draft_probs[i], target_prob):
                tokens.append(draft_tokens[i])
                accepted += 1
            else:
                # Reject and resample from adjusted distribution
                corrected_token = resample(target_prob, draft_probs[i])
                tokens.append(corrected_token)
                break

        # Step 4: Continue if no rejections
        if accepted == K:
            # All accepted, generate one more from target
            final_logits = target_model(tokens)
            tokens.append(sample(softmax(final_logits)))

    return tokens
```

### Acceptance Criterion

The key to maintaining the correct distribution is the **acceptance test**:

```python
def should_accept(draft_token, p_draft, p_target):
    """
    Accept with probability min(1, p_target[token] / p_draft[token])
    """
    acceptance_prob = p_target[draft_token] / p_draft[draft_token]
    return random.random() < min(1.0, acceptance_prob)
```

### Resampling on Rejection

When a token is rejected, we sample from the **adjusted distribution**:

```python
def resample(p_target, p_draft):
    """
    Sample from: max(0, p_target - p_draft) / Z
    where Z is the normalization constant
    """
    adjusted_probs = np.maximum(0, p_target - p_draft)
    adjusted_probs /= adjusted_probs.sum()
    return np.random.choice(len(adjusted_probs), p=adjusted_probs)
```

## Implementation in llama.cpp

### Using Speculative Decoding

llama.cpp supports speculative decoding through the `--draft` parameter:

```bash
# Use a small model as draft, large model as target
./llama-cli \
    -m models/llama-70b-q4.gguf \
    --draft models/llama-7b-q4.gguf \
    -n 512 \
    -p "Explain quantum computing"
```

### Configuration Parameters

```bash
# Number of draft tokens per iteration
--draft-n 4

# Draft model path
--draft models/draft-model.gguf

# Enable debug output
--draft-debug
```

### C++ API Usage

```cpp
#include "llama.h"

// Initialize models
llama_model* target_model = llama_load_model_from_file(
    "models/llama-70b.gguf", params
);
llama_model* draft_model = llama_load_model_from_file(
    "models/llama-7b.gguf", params
);

// Create contexts
llama_context* target_ctx = llama_new_context_with_model(target_model, ctx_params);
llama_context* draft_ctx = llama_new_context_with_model(draft_model, ctx_params);

// Configure speculative parameters
llama_sampling_params sampling;
sampling.n_draft = 4;  // Generate 4 draft tokens

// Generate with speculative decoding
while (!done) {
    // Draft generation
    std::vector<llama_token> draft_tokens;
    for (int i = 0; i < sampling.n_draft; i++) {
        llama_decode(draft_ctx, ...);
        draft_tokens.push_back(llama_sample(draft_ctx));
    }

    // Target verification (parallel)
    llama_decode_batch(target_ctx, draft_tokens);

    // Accept/reject logic
    int n_accept = llama_verify_draft(target_ctx, draft_ctx, draft_tokens);

    // Update position
    tokens.insert(tokens.end(),
                  draft_tokens.begin(),
                  draft_tokens.begin() + n_accept);
}
```

## Optimizing Acceptance Rates

### Choosing the Right Draft Model

**Best Practices**:

1. **Same architecture**: Draft and target should be the same model family
   - ✅ LLaMA-7B → LLaMA-70B (good)
   - ❌ Mistral-7B → LLaMA-70B (poor alignment)

2. **Size ratio**: 8-10x difference is optimal
   - ✅ 7B draft + 70B target
   - ✅ 1B draft + 13B target
   - ⚠️ 7B draft + 13B target (marginal gains)

3. **Same fine-tuning**: If target is instruction-tuned, draft should be too

### Tuning K (Number of Draft Tokens)

```python
# Experiment to find optimal K
def find_optimal_k(draft_model, target_model, prompts):
    results = {}

    for K in [2, 4, 6, 8, 10]:
        total_time = 0
        total_tokens = 0

        for prompt in prompts:
            start = time.time()
            tokens = speculative_decode(prompt, draft_model, target_model, K)
            elapsed = time.time() - start

            total_time += elapsed
            total_tokens += len(tokens)

        throughput = total_tokens / total_time
        results[K] = throughput

    return results

# Typical findings:
# K=4: Best for most cases (2-3x speedup)
# K=6-8: Better for high-similarity models
# K=2-3: Better for low-similarity models
```

### Temperature and Sampling Effects

**Important**: Acceptance rate depends on sampling settings!

```python
# High temperature = lower acceptance
generate(temp=1.0)  # ~70% acceptance
generate(temp=0.7)  # ~80% acceptance
generate(temp=0.1)  # ~90% acceptance (near-greedy)

# Top-k/top-p affects alignment
generate(top_k=50)   # Good acceptance
generate(top_k=1)    # Perfect acceptance (greedy)
```

## Performance Analysis

### Theoretical Speedup

Let:
- `t_draft` = time to generate 1 token with draft model
- `t_target` = time to generate 1 token with target model
- `K` = number of draft tokens
- `α` = average acceptance rate

**Speedup formula**:
```
Speedup = (K × α + 1) / (K × t_draft/t_target + 1)
```

Example with `t_target/t_draft = 10`, `K = 4`, `α = 0.75`:
```
Speedup = (4 × 0.75 + 1) / (4 × 0.1 + 1) = 4 / 1.4 = 2.86x
```

### Practical Benchmarks

Real-world results from llama.cpp:

| Model Pair | K | Acceptance | Tokens/sec | Speedup |
|------------|---|------------|------------|---------|
| 7B → 70B | 4 | 78% | 42 | 2.6x |
| 7B → 70B | 6 | 72% | 45 | 2.8x |
| 1B → 13B | 4 | 81% | 85 | 2.1x |
| Mistral-7B → LLaMA-70B | 4 | 42% | 18 | 0.9x ❌ |

### When NOT to Use Speculative Decoding

Speculative decoding can be **slower** when:

1. **Low acceptance rate** (<50%)
   - Draft and target models too different
   - High temperature sampling
   - Creative generation tasks

2. **Memory-bound scenarios**
   - Both models don't fit in VRAM
   - Swapping overhead dominates

3. **Small batch sizes**
   - Verification overhead too high
   - Better to use larger batches instead

4. **Very short generations**
   - Startup overhead not amortized

## Advanced Techniques

### Multi-Draft Models

Use multiple draft models and pick the best:

```python
def multi_draft_decode(prompt, drafts, target, K=4):
    """
    drafts: List of draft models
    Pick the draft with highest expected acceptance
    """
    best_drafts = []

    for draft in drafts:
        tokens = draft.generate(K)
        score = target.score_sequence(tokens)  # Log probability
        best_drafts.append((score, tokens))

    # Use highest-scoring draft
    best_tokens = max(best_drafts, key=lambda x: x[0])[1]
    return verify_and_accept(best_tokens, target)
```

### Adaptive K Selection

Adjust K based on recent acceptance rates:

```python
class AdaptiveSpeculativeDecoder:
    def __init__(self):
        self.K = 4
        self.acceptance_history = []

    def decode_step(self, draft, target):
        # Generate K tokens
        drafts = draft.generate(self.K)
        n_accept = target.verify(drafts)

        # Track acceptance rate
        acceptance_rate = n_accept / self.K
        self.acceptance_history.append(acceptance_rate)

        # Adjust K based on recent performance
        if len(self.acceptance_history) >= 10:
            avg_acceptance = np.mean(self.acceptance_history[-10:])

            if avg_acceptance > 0.85 and self.K < 8:
                self.K += 1  # Increase speculation
            elif avg_acceptance < 0.60 and self.K > 2:
                self.K -= 1  # Reduce speculation
```

### Tree-Based Speculation

Generate a tree of possibilities instead of a sequence:

```
                    root
                   /  |  \
                 t1  t2  t3
                /|\  |   |
              t4 t5 t6  t7
```

This improves acceptance rate but requires more memory.

## Production Considerations

### Model Selection Strategy

```python
def select_draft_model(target_model, available_drafts):
    """
    Choose best draft model based on:
    1. Architecture compatibility
    2. Size ratio
    3. Available memory
    4. Benchmark results
    """
    candidates = []

    for draft in available_drafts:
        # Check architecture compatibility
        if draft.architecture != target_model.architecture:
            continue

        # Check size ratio
        ratio = target_model.params / draft.params
        if ratio < 5 or ratio > 15:
            continue

        # Check memory fit
        if draft.memory_required + target_model.memory_required > available_memory:
            continue

        # Benchmark on sample prompts
        acceptance = benchmark_acceptance(draft, target_model)
        speedup = estimate_speedup(draft, target_model, acceptance)

        candidates.append((speedup, draft))

    return max(candidates, key=lambda x: x[0])[1] if candidates else None
```

### Monitoring and Metrics

Key metrics to track in production:

```python
class SpeculativeMetrics:
    def __init__(self):
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.total_time_draft = 0.0
        self.total_time_verify = 0.0
        self.iterations = 0

    def log_iteration(self, n_draft, n_accept, time_draft, time_verify):
        self.total_draft_tokens += n_draft
        self.total_accepted_tokens += n_accept
        self.total_time_draft += time_draft
        self.total_time_verify += time_verify
        self.iterations += 1

    def summary(self):
        return {
            'acceptance_rate': self.total_accepted_tokens / self.total_draft_tokens,
            'avg_draft_time_ms': 1000 * self.total_time_draft / self.iterations,
            'avg_verify_time_ms': 1000 * self.total_time_verify / self.iterations,
            'efficiency': self.total_accepted_tokens / self.iterations,
        }
```

### Cost-Benefit Analysis

Calculate whether speculative decoding is worth it:

```python
def roi_analysis(baseline_cost, draft_cost, target_cost, acceptance_rate, K=4):
    """
    baseline_cost: Cost per token without speculation (e.g., $0.001)
    draft_cost: Cost per draft token (e.g., $0.0001)
    target_cost: Cost per target verification (e.g., $0.0008)
    """
    # Baseline: N tokens at baseline cost
    baseline_total = baseline_cost

    # Speculative: Draft + verify costs
    iterations = 1.0 / (K * acceptance_rate + 1)  # Iterations per final token
    spec_total = iterations * (K * draft_cost + target_cost)

    savings = (baseline_total - spec_total) / baseline_total * 100

    return {
        'baseline_cost': baseline_total,
        'speculative_cost': spec_total,
        'savings_percent': savings,
        'break_even_acceptance': compute_break_even(K, draft_cost, target_cost)
    }
```

## Interview Questions

1. **Explain speculative decoding in 2 minutes. Why does it work?**
   - Draft model generates candidates, target validates in parallel
   - Key: Parallel verification faster than sequential generation
   - Maintains exact output distribution via acceptance sampling

2. **What acceptance rate do you need for 2x speedup?**
   - Depends on draft/target speed ratio and K
   - With 10x ratio and K=4: need ~60% acceptance
   - With 10x ratio and K=8: need ~70% acceptance

3. **Design a system that automatically chooses optimal K per request.**
   - Profile acceptance rates for different K values
   - Consider prompt characteristics (length, domain)
   - Use online learning to adapt
   - Cache optimal K per user/use case

4. **When would you NOT use speculative decoding in production?**
   - Memory constraints (can't fit both models)
   - Low acceptance rate scenarios (<50%)
   - Creative tasks requiring high diversity
   - Very short generations (overhead not amortized)

5. **How would you debug poor acceptance rates?**
   - Check model alignment (same architecture/fine-tuning)
   - Verify sampling parameters match
   - Profile per-token acceptance patterns
   - Test with greedy decoding (should be 100%)
   - Check temperature/top-k settings

## Hands-On Exercises

### Exercise 1: Basic Speculative Decoding

Implement and benchmark speculative decoding:

```bash
# Download models
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF
wget https://huggingface.co/TheBloke/Llama-2-70B-GGUF

# Run with speculation
time ./llama-cli \
    -m llama-70b-q4.gguf \
    --draft llama-7b-q4.gguf \
    --draft-n 4 \
    -n 100 \
    -p "Write a story about" \
    --draft-debug

# Compare without speculation
time ./llama-cli \
    -m llama-70b-q4.gguf \
    -n 100 \
    -p "Write a story about"

# Calculate speedup
```

### Exercise 2: Acceptance Rate Analysis

Measure acceptance rates under different conditions:

```python
# See code/speculative_benchmark.py for full implementation
python code/speculative_benchmark.py \
    --draft models/llama-7b.gguf \
    --target models/llama-70b.gguf \
    --temperatures 0.1,0.5,0.7,1.0 \
    --k-values 2,4,6,8 \
    --output acceptance_matrix.csv
```

### Exercise 3: Optimal K Selection

Find the best K for your use case:

```python
# See code/optimal_k_finder.py
python code/optimal_k_finder.py \
    --draft models/draft.gguf \
    --target models/target.gguf \
    --prompts prompts.txt \
    --k-range 2,10 \
    --plot results.png
```

## Further Reading

### Research Papers

1. **"Fast Inference from Transformers via Speculative Decoding"** (2023)
   - Leviathan et al., original speculative decoding paper
   - https://arxiv.org/abs/2211.17192

2. **"Accelerating Large Language Model Decoding with Speculative Sampling"** (2023)
   - Chen et al., DeepMind's approach
   - https://arxiv.org/abs/2302.01318

3. **"SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference"** (2024)
   - Miao et al., tree-based extension
   - https://arxiv.org/abs/2305.09781

### Implementation References

- **llama.cpp speculative decoding**: `examples/speculative/`
- **Hugging Face Transformers**: `assisted_generation()`
- **vLLM speculative sampling**: `vllm.SpeculativeModel`

## Summary

Speculative decoding is a powerful technique that can provide 2-3x speedup with **zero quality degradation**. Key takeaways:

✅ **How it works**: Draft model proposes, target model verifies in parallel
✅ **When to use**: High acceptance rate (>70%), memory allows both models
✅ **Optimization**: Choose aligned models, tune K, monitor acceptance
✅ **Production**: Track metrics, adjust dynamically, cost-benefit analysis

In the next lesson, we'll explore **parallel inference** techniques for serving multiple requests efficiently.

---

**Next**: [02-parallel-inference.md](./02-parallel-inference.md)

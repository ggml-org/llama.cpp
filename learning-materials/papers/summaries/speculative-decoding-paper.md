# Speculative Decoding: Accelerating LLM Inference

**Paper**: "Fast Inference from Transformers via Speculative Decoding"
**Authors**: Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
**Published**: February 2023
**Link**: https://arxiv.org/abs/2211.17192
**Module**: 5 - Advanced Inference Optimization
**Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

Speculative decoding achieves 2-3× speedup for LLM inference by using a small "draft" model to predict multiple tokens ahead, then verifying them in parallel with the large model. This maintains the exact same output distribution as standard autoregressive decoding while dramatically reducing latency.

**Key Innovation**: Parallel verification of multiple speculative tokens with rejection sampling to preserve distribution.

---

## 1. Problem: Autoregressive Decoding is Sequential

```python
# Standard generation: One token at a time
def standard_generate(model, prompt, max_tokens):
    tokens = tokenize(prompt)

    for i in range(max_tokens):
        logits = model(tokens)  # Full forward pass
        next_token = sample(logits[-1])  # Sample from distribution
        tokens.append(next_token)

    return tokens

# Problem: Can't parallelize—each token depends on previous
# Latency = max_tokens × time_per_token
```

**For LLaMA 7B**:
- Time per token: ~50ms
- Generate 100 tokens: 5 seconds
- GPU utilization: Low (memory-bound, not compute-bound)

---

## 2. Speculative Decoding Algorithm

### 2.1 Core Concept

```
Use two models:
1. Draft model (small, fast): Predict K tokens speculatively
2. Target model (large, accurate): Verify all K tokens in parallel

If speculation correct: Accept all K tokens (K× speedup)
If speculation wrong: Reject and resample (no quality loss)
```

### 2.2 Algorithm

```python
def speculative_decoding(draft_model, target_model, prompt, max_tokens, K=4):
    """
    K: Number of speculative tokens (typically 4-8)
    """
    tokens = tokenize(prompt)
    total_accepted = 0

    while len(tokens) < max_tokens:
        # Step 1: Draft model generates K speculative tokens
        draft_tokens = []
        draft_probs = []

        x = tokens
        for _ in range(K):
            logits_draft = draft_model(x)
            p_draft = softmax(logits_draft[-1])
            t = sample(p_draft)
            draft_tokens.append(t)
            draft_probs.append(p_draft)
            x = x + [t]

        # Step 2: Target model verifies ALL K+1 positions in parallel
        # (original context + K draft tokens)
        logits_target = target_model(tokens + draft_tokens)  # Single forward pass!

        # Step 3: Verify each speculative token
        accepted = 0
        for i in range(K):
            p_target = softmax(logits_target[len(tokens) + i])
            p_draft = draft_probs[i]
            t = draft_tokens[i]

            # Rejection sampling: accept with probability min(1, p_target[t] / p_draft[t])
            accept_prob = min(1.0, p_target[t] / p_draft[t])

            if random.uniform(0, 1) < accept_prob:
                tokens.append(t)
                accepted += 1
            else:
                # Rejection: resample from adjusted distribution
                p_adjusted = torch.max(0, p_target - p_draft)
                p_adjusted = p_adjusted / p_adjusted.sum()
                t_new = sample(p_adjusted)
                tokens.append(t_new)
                break  # Stop at first rejection

        # If all K accepted, sample one more token from target model
        if accepted == K:
            p_target_final = softmax(logits_target[len(tokens)])
            tokens.append(sample(p_target_final))

        total_accepted += accepted

    return tokens, total_accepted / (len(tokens) - len(tokenize(prompt)))

# Expected speedup = (K × acceptance_rate + 1) / (K + 1)
# If acceptance_rate = 0.75, K = 4:
# Speedup = (4 × 0.75 + 1) / 5 = 0.8× faster per step
# But each step generates multiple tokens!
```

---

## 3. Theoretical Guarantees

### 3.1 Distribution Preservation

**Theorem**: Speculative decoding produces the EXACT same token distribution as standard autoregressive decoding from the target model.

**Proof sketch**:
- Acceptance probability ensures correct target distribution
- Rejection sampling corrects for draft model bias
- Result: Mathematically equivalent to sampling from target model

### 3.2 Expected Speedup

```python
def expected_speedup(K, acceptance_rate):
    """
    K: Speculation depth
    acceptance_rate: Fraction of tokens accepted (0-1)

    Returns: Speedup factor
    """
    # Expected tokens generated per iteration
    E_accepted = sum(k * acceptance_rate**k * (1 - acceptance_rate)
                     for k in range(K))
    E_accepted += K * acceptance_rate**K

    # Cost: 1 draft forward pass + 1 target forward pass
    # Assume draft is α× faster than target (α = 10-20 typically)
    alpha = 10
    cost_draft = K / alpha  # K sequential draft steps
    cost_target = 1          # 1 parallel target step
    total_cost = cost_draft + cost_target

    speedup = E_accepted / total_cost
    return speedup

# Example: K=5, acceptance_rate=0.7, alpha=10
# Speedup ≈ 2.3× (empirically validated)
```

---

## 4. Draft Model Selection

### 4.1 Options

**1. Smaller model from same family**:
```
Target: LLaMA 70B
Draft: LLaMA 7B (10× faster, similar distribution)
Acceptance rate: 60-80%
```

**2. Quantized version**:
```
Target: LLaMA 13B FP16
Draft: LLaMA 13B Q4_K_M (4× faster)
Acceptance rate: 80-90% (very similar!)
```

**3. Distilled model**:
```
Target: LLaMA 70B
Draft: Student model (distilled from 70B)
Acceptance rate: 70-85%
```

**4. Fine-tuned small model**:
```
Target: Domain-specific LLaMA 70B
Draft: LLaMA 7B fine-tuned on same domain
Acceptance rate: 75-90% (better than base 7B)
```

---

## 5. Implementation

### 5.1 Practical Implementation

```python
import torch
from transformers import AutoModelForCausalLM

# Load models
target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
draft_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

def speculative_decode_step(
    input_ids,
    draft_model,
    target_model,
    K=5,
    temperature=1.0
):
    # Draft phase: Generate K tokens
    draft_tokens = []
    draft_logits_list = []

    draft_input = input_ids
    for _ in range(K):
        with torch.no_grad():
            outputs_draft = draft_model(draft_input)
            logits_draft = outputs_draft.logits[:, -1, :] / temperature
            probs_draft = torch.softmax(logits_draft, dim=-1)
            next_token = torch.multinomial(probs_draft, num_samples=1)

            draft_tokens.append(next_token)
            draft_logits_list.append(logits_draft)
            draft_input = torch.cat([draft_input, next_token], dim=-1)

    # Verification phase: Single forward pass
    verify_input = torch.cat([input_ids] + draft_tokens, dim=-1)
    with torch.no_grad():
        outputs_target = target_model(verify_input)

    # Verify each token
    accepted_tokens = []
    for i, draft_token in enumerate(draft_tokens):
        target_logits = outputs_target.logits[:, len(input_ids) + i - 1, :] / temperature
        target_probs = torch.softmax(target_logits, dim=-1)
        draft_probs = torch.softmax(draft_logits_list[i], dim=-1)

        # Acceptance probability
        p_target = target_probs[0, draft_token.item()]
        p_draft = draft_probs[0, draft_token.item()]
        acceptance_prob = min(1.0, (p_target / p_draft).item())

        if torch.rand(1).item() < acceptance_prob:
            accepted_tokens.append(draft_token)
        else:
            # Rejection sampling
            adjusted_probs = torch.clamp(target_probs - draft_probs, min=0)
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            resampled_token = torch.multinomial(adjusted_probs, num_samples=1)
            accepted_tokens.append(resampled_token)
            break

    return torch.cat(accepted_tokens, dim=-1)
```

---

## 6. llama.cpp Integration

### 6.1 Draft Model Support

```bash
# Speculative decoding in llama.cpp
./llama-cli \
  -m llama-70b-q4_K_M.gguf \        # Target model
  -md llama-7b-q4_K_M.gguf \        # Draft model
  --draft 5 \                        # K=5 speculative tokens
  -p "Tell me about quantum computing" \
  -n 200

# Output includes acceptance rate statistics:
# Draft tokens: 1000
# Accepted: 720 (72%)
# Effective speedup: 2.1×
```

### 6.2 Self-Speculative Decoding

```bash
# Use quantized version as draft model
./llama-cli \
  -m llama-13b-f16.gguf \      # Target (FP16)
  -md llama-13b-q4_0.gguf \    # Draft (quantized, 4× faster)
  --draft 6 \
  -p "Explain..."

# Acceptance rate: ~85% (same model, different precision)
# Speedup: ~2.5×
```

---

## 7. Optimizations and Variants

### 7.1 Adaptive Speculation Depth

```python
def adaptive_K(acceptance_history, K_min=2, K_max=8):
    """
    Adjust K based on recent acceptance rate
    """
    recent_rate = sum(acceptance_history[-10:]) / 10

    if recent_rate > 0.8:
        return min(K_max, current_K + 1)  # Increase speculation
    elif recent_rate < 0.5:
        return max(K_min, current_K - 1)  # Decrease speculation
    else:
        return current_K

# Better performance across different contexts
```

### 7.2 Tree-Based Speculation (Medusa)

```
Instead of linear chain:
       t1 → t2 → t3 → t4

Use tree:
       t1 → t2 → t3
        ↘ t2' → t3'
         ↘ t2''

Verify all paths in parallel, pick best
Speedup: 3-4× (vs 2-3× for linear)
```

---

## 8. Benchmarks

### LLaMA 2 70B + 7B (A100 GPU)

| Task | Standard (tokens/s) | Speculative (tokens/s) | Speedup |
|------|---------------------|------------------------|---------|
| Summarization | 12.3 | 28.1 | 2.3× |
| Code generation | 11.8 | 31.4 | 2.7× |
| Creative writing | 10.9 | 24.2 | 2.2× |
| Q&A | 13.1 | 32.6 | 2.5× |

**Average speedup**: 2.4× with no quality loss

---

## 9. Key Takeaways

### Principles
✅ **Distribution-preserving**: Exact same output as standard decoding
✅ **Latency reduction**: 2-3× speedup with draft model
✅ **No training needed**: Works with existing models
✅ **Parallelization**: Verifies K tokens in single forward pass

### For llama.cpp Users
- Enable with `-md` flag (draft model)
- Works best with same-family models (e.g., 70B + 7B)
- Self-speculation with quantized draft is very effective
- Monitor acceptance rate to tune K

---

## Further Reading

- **Paper**: https://arxiv.org/abs/2211.17192
- **Medusa**: https://arxiv.org/abs/2401.10774 (tree-based)
- **llama.cpp draft**: See `examples/speculative` directory

---

**Status**: Complete | Module 5 (1/3) papers

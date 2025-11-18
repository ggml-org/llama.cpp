# Sampling Strategies

**Learning Module**: Module 2 - Core Implementation
**Estimated Reading Time**: 26 minutes
**Prerequisites**: Basic probability, understanding of inference pipeline
**Related Content**:
- [Inference Pipeline](./04-inference-pipeline.md)
- [Grammar Constraints](./06-grammar-constraints.md)

---

## Overview

Sampling strategies determine how models select the next token from probability distributions. The choice of sampling method dramatically affects output quality, diversity, and coherence.

### Learning Objectives

After completing this lesson, you will:
- ✅ Understand all major sampling algorithms
- ✅ Configure sampling parameters effectively
- ✅ Balance creativity and coherence
- ✅ Debug generation quality issues
- ✅ Implement custom sampling strategies

---

## From Logits to Tokens

### The Sampling Process

```
┌─────────────────────────────────────────────────────┐
│           Token Sampling Pipeline                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Forward Pass                                     │
│     Model → Logits (raw scores)                     │
│     Shape: [vocab_size] (e.g., 32000 values)        │
│                                                      │
│  2. Logits Transformation                            │
│     - Temperature scaling                            │
│     - Repetition penalty                             │
│     - Frequency penalty                              │
│                                                      │
│  3. Convert to Probabilities                         │
│     Softmax → Probability distribution               │
│     Sum = 1.0                                        │
│                                                      │
│  4. Apply Sampling Method                            │
│     - Greedy: Pick highest                           │
│     - Top-K: Filter to top K                         │
│     - Top-P: Filter to cumulative probability P      │
│     - Min-P: Filter below minimum probability        │
│     - Typical: Filter by local entropy               │
│     - Mirostat: Dynamic adjustment                   │
│                                                      │
│  5. Sample Token                                     │
│     Random selection from filtered distribution      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Basic Example

```python
# After forward pass
logits = model.forward(input_ids)  # Shape: [32000]

# Example logits (simplified)
logits = [2.5, 1.2, 3.8, 0.9, 2.1, ...]  # 32000 values

# Apply temperature
temperature = 0.8
logits = logits / temperature

# Convert to probabilities
probs = softmax(logits)  # [0.12, 0.05, 0.41, 0.03, 0.08, ...]

# Sample token
next_token = sample_from(probs)  # e.g., token 2 (prob 0.41)
```

---

## Sampling Methods

### 1. Greedy Sampling

Always pick the highest probability token.

```python
def greedy_sample(logits):
    """Deterministic: always same output"""
    return argmax(logits)

# Example
logits = [1.2, 3.5, 2.1, 0.8]
token = greedy_sample(logits)  # Always returns 1 (highest)
```

**Characteristics**:
- ✅ Deterministic (reproducible)
- ✅ Often high quality for factual tasks
- ❌ No diversity
- ❌ Can get stuck in loops
- ❌ Repetitive output

**Use Cases**:
- Code generation
- Translation
- Question answering
- When reproducibility is critical

### 2. Temperature Sampling

Scale logits before softmax to control randomness.

```python
def temperature_sample(logits, temperature):
    """
    temperature < 1.0: More focused (sharper distribution)
    temperature = 1.0: Unchanged
    temperature > 1.0: More random (flatter distribution)
    """
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    return sample_from_distribution(probs)
```

**Temperature Effect**:
```
Original logits: [1.0, 2.0, 3.0]

Temperature = 0.5 (focused):
  Probs: [0.09, 0.24, 0.67]  # Most weight on highest
  Output: More deterministic, less diverse

Temperature = 1.0 (unchanged):
  Probs: [0.09, 0.24, 0.67]  # Original distribution

Temperature = 2.0 (random):
  Probs: [0.18, 0.26, 0.56]  # More evenly distributed
  Output: More diverse, less focused
```

**Guidelines**:
```
Temperature = 0.1-0.5:  Very focused, deterministic
Temperature = 0.7-0.9:  Good balance (recommended for most tasks)
Temperature = 1.0-1.5:  More creative, diverse
Temperature = 2.0+:     Very random, often incoherent
```

**llama.cpp**:
```cpp
// Apply temperature
for (size_t i = 0; i < n_vocab; i++) {
    logits[i] /= temperature;
}
```

### 3. Top-K Sampling

Only consider top K most probable tokens.

```python
def top_k_sample(logits, k):
    """
    Keep top K tokens, set rest to -inf
    """
    # Sort logits
    top_k_indices = argsort(logits)[-k:]

    # Zero out non-top-k
    filtered_logits = [-inf] * len(logits)
    for idx in top_k_indices:
        filtered_logits[idx] = logits[idx]

    # Sample from filtered distribution
    probs = softmax(filtered_logits)
    return sample_from_distribution(probs)
```

**Example**:
```
Vocabulary: ["the", "a", "an", "cat", "dog", "elephant", "zebra", ...]
Logits:     [5.2,   4.8, 3.1, 2.9,  2.1,  0.5,       0.3,    ...]

Top-K = 3:
  Keep: ["the", "a", "an"]
  Sample from these 3 only

Top-K = 10:
  Keep: ["the", "a", "an", "cat", "dog", ...]
  More diverse but still filtered
```

**Trade-offs**:
```
K = 1:      Equivalent to greedy
K = 10:     Moderate diversity
K = 50:     High diversity
K = vocab:  Unfiltered (just temperature)
```

**Limitations**:
- Fixed K may be too restrictive (peaked distribution) or too permissive (flat distribution)
- Doesn't adapt to distribution shape

### 4. Top-P (Nucleus) Sampling

Keep tokens until cumulative probability reaches P.

```python
def top_p_sample(logits, p):
    """
    Nucleus sampling: dynamic vocabulary size
    """
    # Convert to probabilities
    probs = softmax(logits)

    # Sort by probability
    sorted_probs, sorted_indices = sort(probs, descending=True)

    # Find cutoff where cumulative probability > p
    cumsum = 0.0
    cutoff_idx = 0
    for i, prob in enumerate(sorted_probs):
        cumsum += prob
        if cumsum > p:
            cutoff_idx = i + 1
            break

    # Keep only top nucleus
    filtered_probs = [0.0] * len(probs)
    for i in range(cutoff_idx):
        idx = sorted_indices[i]
        filtered_probs[idx] = probs[idx]

    # Renormalize and sample
    filtered_probs = normalize(filtered_probs)
    return sample_from_distribution(filtered_probs)
```

**Example**:
```
Probs: [0.50, 0.20, 0.15, 0.08, 0.04, 0.02, 0.01]

Top-P = 0.9:
  Cumulative: 0.50, 0.70, 0.85, 0.93
  Keep: First 4 tokens (cumsum = 0.93 > 0.9)
  Effective vocabulary: 4 tokens

Top-P = 0.95:
  Keep: First 5 tokens (cumsum = 0.97 > 0.95)
  Effective vocabulary: 5 tokens
```

**Advantages**:
- ✅ Adapts to distribution shape
- ✅ Peaked distribution → few tokens (focused)
- ✅ Flat distribution → many tokens (diverse)
- ✅ Better than fixed Top-K

**Guidelines**:
```
P = 0.9:   Conservative, high quality (recommended)
P = 0.95:  Balanced
P = 0.99:  Diverse but may include low-quality tokens
P = 1.0:   Equivalent to temperature-only sampling
```

### 5. Min-P Sampling

Filter tokens below a minimum probability threshold.

```python
def min_p_sample(logits, min_p):
    """
    Remove tokens with probability < min_p × max_probability
    """
    probs = softmax(logits)
    max_prob = max(probs)
    threshold = min_p * max_prob

    # Filter
    filtered_probs = [p if p >= threshold else 0.0 for p in probs]

    # Renormalize and sample
    filtered_probs = normalize(filtered_probs)
    return sample_from_distribution(filtered_probs)
```

**Example**:
```
Probs: [0.50, 0.20, 0.15, 0.08, 0.04, 0.02, 0.01]
Max prob: 0.50

Min-P = 0.1:
  Threshold: 0.1 × 0.50 = 0.05
  Keep: [0.50, 0.20, 0.15, 0.08]  (4 tokens)

Min-P = 0.3:
  Threshold: 0.3 × 0.50 = 0.15
  Keep: [0.50, 0.20, 0.15]  (3 tokens)
```

**Benefits**:
- Scales with confidence (high confidence → strict, low confidence → permissive)
- Often better than fixed Top-K
- Good alternative to Top-P

### 6. Typical Sampling

Select tokens with "typical" information content.

```python
def typical_sample(logits, tau):
    """
    Locally typical sampling
    Keep tokens with entropy near conditional entropy
    """
    probs = softmax(logits)

    # Calculate entropy
    entropy = -sum(p * log(p) for p in probs if p > 0)

    # Calculate surprise for each token
    surprises = [-log(p) for p in probs]

    # Keep tokens near expected surprise (entropy)
    filtered_probs = []
    for p, surprise in zip(probs, surprises):
        if abs(surprise - entropy) < tau:
            filtered_probs.append(p)
        else:
            filtered_probs.append(0.0)

    # Renormalize and sample
    filtered_probs = normalize(filtered_probs)
    return sample_from_distribution(filtered_probs)
```

**Key Idea**:
- Don't just pick high probability tokens
- Pick tokens with "typical" information content
- Avoids both too-obvious and too-surprising tokens

**Use Cases**:
- Creative writing
- When diversity needed but coherence important
- Alternative to Top-P

### 7. Mirostat Sampling

Adaptive sampling that maintains target perplexity.

```python
class MirostatSampler:
    def __init__(self, target_tau=5.0, learning_rate=0.1):
        self.target_tau = target_tau  # Target perplexity
        self.mu = 2.0 * target_tau     # Initial mu
        self.learning_rate = learning_rate

    def sample(self, logits):
        """
        Dynamically adjust sampling to maintain target perplexity
        """
        probs = softmax(logits)

        # Sort by probability
        sorted_probs, sorted_indices = sort(probs, descending=True)

        # Find cutoff based on current mu
        cumsum = 0.0
        cutoff_idx = 0
        for i, prob in enumerate(sorted_probs):
            cumsum += prob
            surprise = -log(prob)
            if surprise > self.mu:
                cutoff_idx = i
                break

        # Sample from filtered distribution
        token = sample_from(sorted_probs[:cutoff_idx])

        # Update mu based on observed surprise
        selected_prob = probs[token]
        surprise = -log(selected_prob)
        self.mu = self.mu - self.learning_rate * (surprise - self.target_tau)

        return token
```

**Benefits**:
- Self-adjusting based on model confidence
- Maintains consistent quality over long generation
- Good for long-form generation

**Parameters**:
```
target_tau = 5.0:    Target perplexity (higher = more diverse)
learning_rate = 0.1: How quickly to adapt
```

---

## Penalty Methods

### Repetition Penalty

Penalize tokens that have already appeared.

```python
def apply_repetition_penalty(logits, previous_tokens, penalty):
    """
    penalty > 1.0: Discourage repetition
    penalty < 1.0: Encourage repetition (rare)
    """
    for token in previous_tokens:
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty
    return logits

# Example
penalty = 1.1  # 10% penalty for repeated tokens
logits = apply_repetition_penalty(logits, previous_tokens, penalty)
```

**Guidelines**:
```
1.0:   No penalty (default)
1.05:  Mild (subtle reduction)
1.1:   Moderate (recommended)
1.2:   Strong (may affect quality)
1.3+:  Very strong (can break coherence)
```

### Frequency Penalty

Penalize based on how often token has appeared.

```python
def apply_frequency_penalty(logits, token_counts, penalty):
    """
    Linear penalty based on frequency
    """
    for token, count in token_counts.items():
        logits[token] -= penalty * count
    return logits

# More sophisticated: presence penalty
def apply_presence_penalty(logits, seen_tokens, penalty):
    """
    Binary: was token seen or not?
    """
    for token in seen_tokens:
        logits[token] -= penalty
    return logits
```

**Use Cases**:
- Prevent repetitive phrases
- Encourage vocabulary diversity
- Reduce looping behavior

---

## Combined Strategies

Most practical applications combine multiple methods:

```python
def advanced_sampling(
    logits,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    min_p=0.05,
    repetition_penalty=1.1,
    previous_tokens=[]
):
    """
    Typical production sampling pipeline
    """
    # 1. Apply repetition penalty
    logits = apply_repetition_penalty(logits, previous_tokens, repetition_penalty)

    # 2. Apply temperature
    logits = logits / temperature

    # 3. Convert to probabilities
    probs = softmax(logits)

    # 4. Apply Top-K filtering
    if top_k > 0:
        probs = top_k_filter(probs, top_k)

    # 5. Apply Top-P filtering
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)

    # 6. Apply Min-P filtering
    if min_p > 0.0:
        probs = min_p_filter(probs, min_p)

    # 7. Sample
    return sample_from_distribution(probs)
```

---

## Parameter Tuning Guide

### For Different Use Cases

**1. Creative Writing**
```python
temperature = 0.9          # More diverse
top_p = 0.95              # Allow variety
repetition_penalty = 1.1  # Prevent loops
```

**2. Code Generation**
```python
temperature = 0.2         # Very focused
top_p = 0.95             # Standard
repetition_penalty = 1.0  # Don't penalize (code repeats naturally)
```

**3. Factual Q&A**
```python
temperature = 0.1         # Almost greedy
top_p = 0.9              # Conservative
repetition_penalty = 1.0  # Neutral
```

**4. Chat/Conversation**
```python
temperature = 0.7         # Balanced
top_p = 0.9              # Standard
repetition_penalty = 1.15 # Mild penalty
```

**5. Long-form Generation**
```python
# Use Mirostat to maintain quality
mirostat = 2
mirostat_tau = 5.0
mirostat_eta = 0.1
```

---

## Implementation in llama.cpp

### Sampling Configuration

```cpp
// From llama-sampling.h
struct llama_sampling_params {
    int32_t     n_prev                = 64;        // Tokens to consider for penalties
    int32_t     top_k                 = 40;        // Top-K sampling
    float       top_p                 = 0.95f;     // Top-P (nucleus) sampling
    float       min_p                 = 0.05f;     // Min-P sampling
    float       tfs_z                 = 1.00f;     // Tail-free sampling
    float       typical_p             = 1.00f;     // Typical sampling
    float       temp                  = 0.80f;     // Temperature
    float       penalty_repeat        = 1.10f;     // Repetition penalty
    float       penalty_freq          = 0.00f;     // Frequency penalty
    float       penalty_present       = 0.00f;     // Presence penalty
    int32_t     mirostat              = 0;         // Mirostat: 0=disabled, 1=v1, 2=v2
    float       mirostat_tau          = 5.00f;     // Mirostat target entropy
    float       mirostat_eta          = 0.10f;     // Mirostat learning rate
    bool        penalize_nl           = true;      // Penalize newlines
};
```

### Sampling Function

```cpp
llama_token llama_sampling_sample(
    struct llama_sampling_context * ctx_sampling,
    struct llama_context * ctx_main,
    struct llama_context * ctx_cfg,  // Classifier-free guidance context
    int idx = -1
) {
    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    // Get logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    // Apply penalties
    llama_sample_repetition_penalties(
        ctx_main,
        &ctx_sampling->cur,
        ctx_sampling->params.penalty_repeat,
        ctx_sampling->params.penalty_freq,
        ctx_sampling->params.penalty_present
    );

    // Apply temperature
    llama_sample_temp(ctx_main, &ctx_sampling->cur, ctx_sampling->params.temp);

    // Apply Top-K
    llama_sample_top_k(ctx_main, &ctx_sampling->cur, ctx_sampling->params.top_k, 1);

    // Apply Top-P
    llama_sample_top_p(ctx_main, &ctx_sampling->cur, ctx_sampling->params.top_p, 1);

    // Apply Min-P
    llama_sample_min_p(ctx_main, &ctx_sampling->cur, ctx_sampling->params.min_p, 1);

    // Sample token
    llama_token token = llama_sample_token(ctx_main, &ctx_sampling->cur);

    return token;
}
```

---

## Debugging Generation Issues

### Problem: Repetitive Output

```
Output: "The cat sat on the mat. The cat sat on the mat. The cat..."

Solutions:
1. Increase repetition_penalty (1.1 → 1.2)
2. Increase temperature (0.7 → 0.9)
3. Decrease top_k (40 → 20) or top_p (0.95 → 0.9)
4. Use Mirostat sampling
```

### Problem: Incoherent Output

```
Output: "The xqz flarb gribbled the snorkel..."

Solutions:
1. Decrease temperature (0.9 → 0.7)
2. Decrease top_p (0.99 → 0.95)
3. Increase min_p (0.0 → 0.05)
4. Check model quality (may be undertrained)
```

### Problem: Too Conservative/Boring

```
Output: "The cat is a cat. The dog is a dog. The bird is a bird."

Solutions:
1. Increase temperature (0.5 → 0.8)
2. Increase top_p (0.9 → 0.95)
3. Decrease repetition_penalty (1.2 → 1.1)
4. Use typical sampling
```

---

## Interview Questions

**Q1: Explain the difference between Top-K and Top-P sampling.**

**Answer**: Top-K keeps exactly K tokens regardless of their probabilities, while Top-P keeps tokens until cumulative probability reaches P. Top-P adapts to distribution shape: peaked distributions keep few tokens (focused), flat distributions keep many (diverse). Top-K is fixed regardless of distribution. Top-P is generally preferred for this adaptive property.

**Q2: What is temperature and how does it affect generation?**

**Answer**: Temperature scales logits before softmax. Temperature < 1 makes the distribution sharper (more peaked), making high-probability tokens even more likely. Temperature > 1 makes it flatter (more uniform), giving lower-probability tokens more chance. Temperature = 0 is greedy decoding. Practical range: 0.7-0.9 for most tasks.

**Q3: When would you use Mirostat over standard sampling methods?**

**Answer**: Mirostat maintains target perplexity through adaptive sampling, making it ideal for:
- Long-form generation (maintains consistency)
- When model confidence varies significantly
- Creative writing where quality should stay consistent
It dynamically adjusts the sampling cutoff based on observed entropy, rather than using fixed thresholds.

---

## Further Reading

### Code References
- [llama-sampling.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampling.cpp): Sampling implementations
- [llama-sampling.h](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h): API definitions

### Research Papers
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751): Top-P (nucleus) sampling
- [Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity](https://arxiv.org/abs/2007.14966)
- [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)

### Tutorials
- [Lab 4: Custom Sampling Experiments](../labs/lab-04-custom-sampling.ipynb)
- [Tutorial: Building Custom Samplers](../tutorials/tutorial-02-custom-samplers.ipynb)
- [Code Example: Sampling Comparison](../code/sampling_comparison.py)

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Module**: 2 - Core Implementation

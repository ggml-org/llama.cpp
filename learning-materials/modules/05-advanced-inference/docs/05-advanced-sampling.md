# Advanced Sampling: Beyond Temperature and Top-K

**Module 5, Lesson 5**
**Estimated Time**: 4-5 hours
**Difficulty**: Advanced

## Overview

Sampling is the process of selecting the next token from the model's probability distribution. While basic methods like temperature scaling and top-k are well-known, advanced algorithms like Mirostat, min-p, and custom samplers can dramatically improve generation quality.

## Learning Objectives

By the end of this lesson, you will:
- ✅ Master all sampling algorithms in llama.cpp
- ✅ Understand Mirostat and perplexity control
- ✅ Implement min-p and locally typical sampling
- ✅ Design custom sampling algorithms
- ✅ Optimize sampling for different use cases

## Sampling Fundamentals

### The Sampling Problem

Given logits from the model, how do we choose the next token?

```python
# Model output (simplified)
logits = {
    "the": 4.2,
    "a": 3.8,
    "an": 1.5,
    "some": 0.8,
    "many": -0.3,
    "several": -1.2,
    ...  # 32,000 more tokens
}

# Convert to probabilities
probs = softmax(logits)
# {
#   "the": 0.45,
#   "a": 0.35,
#   "an": 0.08,
#   "some": 0.04,
#   ...
# }

# Now what? How to choose?
next_token = ???
```

### Greedy Decoding (Baseline)

Always choose the highest probability token:

```python
def greedy_sample(logits):
    return argmax(logits)

# Result: Always picks "the"
# Problem: Boring, repetitive, no creativity
```

**Characteristics**:
- ✅ Deterministic
- ✅ Highest likelihood
- ❌ Repetitive
- ❌ No diversity

## Temperature Scaling

### How It Works

Scale logits before softmax to control randomness:

```python
def temperature_sample(logits, temperature=1.0):
    """
    temperature < 1.0: More focused (less random)
    temperature = 1.0: Original distribution
    temperature > 1.0: More diverse (more random)
    """
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    return sample(probs)
```

### Effect of Temperature

```python
logits = [4.2, 3.8, 1.5, 0.8]  # "the", "a", "an", "some"

# Temperature = 0.1 (very focused)
probs_0.1 = [0.62, 0.37, 0.01, 0.00]  # Almost greedy

# Temperature = 0.7 (balanced)
probs_0.7 = [0.48, 0.38, 0.10, 0.04]  # Good diversity

# Temperature = 1.0 (original)
probs_1.0 = [0.45, 0.35, 0.12, 0.08]  # Natural distribution

# Temperature = 2.0 (very diverse)
probs_2.0 = [0.32, 0.30, 0.20, 0.18]  # Too random!
```

### Choosing Temperature

| Use Case | Temperature | Rationale |
|----------|-------------|-----------|
| Code generation | 0.1 - 0.3 | Need correctness |
| Factual QA | 0.3 - 0.5 | Balance accuracy and naturalness |
| Creative writing | 0.7 - 0.9 | Want diversity |
| Brainstorming | 1.0 - 1.5 | Maximum creativity |

## Top-K Sampling

### Algorithm

Only consider the top K most likely tokens:

```python
def top_k_sample(logits, k=40):
    """
    Sample from top-k most likely tokens
    """
    # Get top-k tokens
    top_k_indices = torch.topk(logits, k).indices
    top_k_logits = logits[top_k_indices]

    # Set others to -inf
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits[top_k_indices] = top_k_logits

    # Sample from filtered distribution
    probs = softmax(filtered_logits)
    return sample(probs)
```

### Example

```python
Probabilities (sorted):
1. "the" (0.45)
2. "a" (0.35)
3. "an" (0.08)
4. "some" (0.04)
5. "many" (0.02)
...
50. "several" (0.0001)

k=2: Only consider "the", "a"
k=5: Consider top 5
k=50: Consider top 50
```

**Problem**: K is context-independent. Sometimes we want more/fewer options.

## Top-P (Nucleus) Sampling

### Algorithm

Dynamically choose the smallest set of tokens whose cumulative probability ≥ p:

```python
def top_p_sample(logits, p=0.9):
    """
    Sample from smallest set with cumulative probability >= p
    """
    # Sort by probability (descending)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = softmax(sorted_logits)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # Find cutoff index
    cutoff_idx = torch.where(cumulative_probs >= p)[0][0] + 1

    # Keep only top-p tokens
    top_p_indices = sorted_indices[:cutoff_idx]
    top_p_logits = logits[top_p_indices]

    # Sample from filtered distribution
    probs = softmax(top_p_logits)
    return sample(probs)
```

### Example

```python
Probabilities (cumulative):
1. "the" (0.45)  → 0.45
2. "a" (0.35)    → 0.80
3. "an" (0.08)   → 0.88
4. "some" (0.04) → 0.92  ← p=0.9 cutoff here
5. "many" (0.02) → 0.94
...

p=0.9: Consider tokens 1-4 (dynamic!)
p=0.95: Consider tokens 1-5
```

**Advantage**: Adapts to distribution shape (more tokens when uncertain, fewer when confident).

## Min-P Sampling

### The Problem with Top-P

```python
# High confidence case
probs = [0.90, 0.05, 0.03, 0.02]  # Very confident
top_p(0.9) → keeps 1 token  # Good!

# Low confidence case
probs = [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, ...]  # Very uncertain
top_p(0.9) → keeps 7 tokens  # Still too restrictive!
```

### Min-P Algorithm

Keep tokens with probability ≥ p × max_prob:

```python
def min_p_sample(logits, min_p=0.05):
    """
    Keep tokens with prob >= min_p * max_prob
    """
    probs = softmax(logits)
    max_prob = probs.max()

    # Filter tokens below threshold
    threshold = min_p * max_prob
    mask = probs >= threshold

    filtered_logits = torch.where(
        mask,
        logits,
        torch.full_like(logits, float('-inf'))
    )

    # Sample from filtered distribution
    filtered_probs = softmax(filtered_logits)
    return sample(filtered_probs)
```

### Example

```python
# High confidence
probs = [0.90, 0.05, 0.03, 0.02]
max_prob = 0.90
threshold = 0.05 * 0.90 = 0.045
Keeps: [0.90, 0.05]  ✅ Top 2

# Low confidence (flat distribution)
probs = [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08]
max_prob = 0.15
threshold = 0.05 * 0.15 = 0.0075
Keeps: All 8 tokens  ✅ More diversity when uncertain
```

**Advantage**: Adapts better to distribution flatness.

## Mirostat Sampling

### The Perplexity Problem

**Problem**: Traditional sampling doesn't control output quality/coherence.

**Solution**: Mirostat actively controls perplexity (a measure of surprise).

### Mirostat V1 Algorithm

```python
class MirostatSampler:
    def __init__(self, target_tau=5.0, learning_rate=0.1):
        """
        target_tau: Target perplexity (higher = more diverse)
        learning_rate: How fast to adjust
        """
        self.target_tau = target_tau
        self.learning_rate = learning_rate
        self.mu = 2 * target_tau  # Initial threshold

    def sample(self, logits):
        """
        Dynamically adjust threshold to maintain target perplexity
        """
        probs = softmax(logits)

        # Sort by probability
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Compute cumulative probabilities
        cumsum = torch.cumsum(sorted_probs, dim=0)

        # Find cutoff based on current mu
        # Keep tokens until cumulative surprise > mu
        surprises = -torch.log(sorted_probs)
        cumulative_surprise = torch.cumsum(surprises * sorted_probs, dim=0)

        cutoff_idx = torch.where(cumulative_surprise >= self.mu)[0][0]

        # Sample from filtered set
        top_indices = sorted_indices[:cutoff_idx+1]
        top_logits = logits[top_indices]
        top_probs = softmax(top_logits)

        token_idx = torch.multinomial(top_probs, 1)
        token = top_indices[token_idx]

        # Compute observed surprise
        observed_surprise = -torch.log(probs[token])

        # Update mu to approach target
        error = observed_surprise - self.target_tau
        self.mu -= self.learning_rate * error

        return token
```

### Mirostat V2 Algorithm

Improved version with better stability:

```python
class MirostatV2Sampler:
    def __init__(self, target_tau=5.0, learning_rate=0.1):
        self.target_tau = target_tau
        self.learning_rate = learning_rate
        self.mu = 2 * target_tau

    def sample(self, logits):
        probs = softmax(logits)

        # Truncate to tokens with prob > e^(-mu)
        threshold = math.exp(-self.mu)
        mask = probs > threshold

        # Sample from truncated distribution
        filtered_logits = torch.where(
            mask,
            logits,
            torch.full_like(logits, float('-inf'))
        )

        filtered_probs = softmax(filtered_logits)
        token = torch.multinomial(filtered_probs, 1)

        # Update mu
        observed_surprise = -torch.log(probs[token])
        error = observed_surprise - self.target_tau
        self.mu -= self.learning_rate * error

        # Clamp mu to reasonable range
        self.mu = max(0.1, min(self.mu, 20.0))

        return token
```

### Choosing Mirostat Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| tau | 2.0 - 3.0 | Focused, coherent |
| tau | 4.0 - 6.0 | Balanced (recommended) |
| tau | 7.0 - 10.0 | Diverse, creative |
| learning_rate | 0.05 - 0.2 | Adaptation speed |

## Locally Typical Sampling

### Information-Theoretic Approach

**Idea**: Sample tokens that are "typically surprising" for the context.

```python
def locally_typical_sample(logits, epsilon=0.1):
    """
    Sample tokens with surprise close to conditional entropy
    """
    probs = softmax(logits)

    # Compute conditional entropy
    entropy = -torch.sum(probs * torch.log(probs))

    # Compute surprise for each token
    surprises = -torch.log(probs)

    # Keep tokens with surprise close to entropy
    diff = torch.abs(surprises - entropy)
    mask = diff < epsilon

    # Sample from filtered distribution
    filtered_logits = torch.where(
        mask,
        logits,
        torch.full_like(logits, float('-inf'))
    )

    filtered_probs = softmax(filtered_logits)
    return sample(filtered_probs)
```

**Effect**: Avoids both very likely and very unlikely tokens, focusing on the "typical" range.

## Repetition Penalties

### Frequency Penalty

Penalize tokens based on how often they've appeared:

```python
def frequency_penalty_sample(logits, token_history, penalty=1.0):
    """
    Reduce probability of repeated tokens
    """
    # Count token frequencies
    token_counts = Counter(token_history)

    # Apply penalty
    for token, count in token_counts.items():
        logits[token] -= penalty * count

    return sample(softmax(logits))
```

### Presence Penalty

Binary version - penalize any token that has appeared:

```python
def presence_penalty_sample(logits, token_history, penalty=1.0):
    """
    Penalize tokens that have appeared at all
    """
    appeared_tokens = set(token_history)

    for token in appeared_tokens:
        logits[token] -= penalty

    return sample(softmax(logits))
```

### Recency-Weighted Penalty

Penalize recent repetitions more:

```python
def recency_penalty_sample(logits, token_history, penalty=1.0, decay=0.9):
    """
    Penalize recent tokens more than older ones
    """
    for i, token in enumerate(reversed(token_history)):
        age = i
        weight = penalty * (decay ** age)
        logits[token] -= weight

    return sample(softmax(logits))
```

## Combined Sampling Strategies

### llama.cpp Default Pipeline

```cpp
// Typical llama.cpp sampling pipeline
std::vector<llama_token_data> candidates;

// 1. Start with all tokens
for (int i = 0; i < n_vocab; i++) {
    candidates.push_back({i, logits[i], 0.0f});
}

// 2. Apply repetition penalty
llama_sample_repetition_penalties(
    ctx, &candidates,
    last_tokens, penalty_repeat, penalty_freq, penalty_present
);

// 3. Apply grammar constraints (if any)
if (grammar) {
    llama_sample_grammar(ctx, &candidates, grammar);
}

// 4. Apply top-k
if (top_k > 0) {
    llama_sample_top_k(ctx, &candidates, top_k, min_keep);
}

// 5. Apply temperature
llama_sample_temp(ctx, &candidates, temperature);

// 6. Apply top-p
if (top_p < 1.0) {
    llama_sample_top_p(ctx, &candidates, top_p, min_keep);
}

// 7. Apply min-p (if enabled)
if (min_p > 0.0) {
    llama_sample_min_p(ctx, &candidates, min_p, min_keep);
}

// 8. Sample final token
llama_token token = llama_sample_token(ctx, &candidates);
```

### Custom Sampling Chain

```python
class CustomSamplingPipeline:
    def __init__(self):
        self.stages = []

    def add_stage(self, stage):
        """Add a sampling stage to the pipeline"""
        self.stages.append(stage)

    def sample(self, logits, context):
        """Run through all stages"""
        current_logits = logits.clone()

        for stage in self.stages:
            current_logits = stage(current_logits, context)

        return torch.multinomial(softmax(current_logits), 1)

# Example usage
pipeline = CustomSamplingPipeline()
pipeline.add_stage(FrequencyPenalty(penalty=1.0))
pipeline.add_stage(Temperature(temp=0.7))
pipeline.add_stage(MinP(min_p=0.05))
pipeline.add_stage(TopP(p=0.9))

token = pipeline.sample(logits, context)
```

## Advanced: Custom Sampling Algorithms

### Context-Aware Sampling

Adjust sampling based on context:

```python
class ContextAwareSampler:
    def sample(self, logits, context):
        """
        Adjust sampling based on what we're generating
        """
        # Detect context type
        if is_code_context(context):
            # Low temperature for code
            temperature = 0.2
            top_p = 0.95
        elif is_creative_context(context):
            # High temperature for creative writing
            temperature = 0.9
            top_p = 0.8
        else:
            # Balanced for general text
            temperature = 0.7
            top_p = 0.9

        return self.sample_with_params(logits, temperature, top_p)
```

### Ensemble Sampling

Combine multiple models or sampling strategies:

```python
def ensemble_sample(logits_list, weights):
    """
    Sample from weighted ensemble of distributions
    """
    # Compute weighted average of probabilities
    ensemble_probs = sum(
        w * softmax(logits)
        for w, logits in zip(weights, logits_list)
    )

    return sample(ensemble_probs)
```

### Beam Search

Non-sampling approach - keep top-k hypotheses:

```python
def beam_search(model, prompt, beam_size=4, max_len=100):
    """
    Beam search for most likely sequence
    """
    beams = [(prompt, 0.0)]  # (tokens, score)

    for _ in range(max_len):
        candidates = []

        for tokens, score in beams:
            # Get next token probabilities
            logits = model(tokens)
            probs = softmax(logits)

            # Consider top-k tokens
            top_k_probs, top_k_tokens = torch.topk(probs, beam_size)

            for prob, token in zip(top_k_probs, top_k_tokens):
                new_tokens = tokens + [token]
                new_score = score + math.log(prob)
                candidates.append((new_tokens, new_score))

        # Keep top beam_size candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    # Return best beam
    return beams[0][0]
```

## Benchmarking Sampling Strategies

### Quality Metrics

```python
def evaluate_sampling_strategy(sampler, test_prompts):
    """
    Evaluate sampling quality
    """
    metrics = {
        'perplexity': [],
        'diversity': [],
        'repetition_rate': [],
        'coherence_score': []
    }

    for prompt in test_prompts:
        # Generate multiple samples
        samples = [sampler.generate(prompt) for _ in range(10)]

        # Measure perplexity
        avg_ppl = np.mean([calculate_perplexity(s) for s in samples])
        metrics['perplexity'].append(avg_ppl)

        # Measure diversity (unique n-grams)
        diversity = calculate_diversity(samples)
        metrics['diversity'].append(diversity)

        # Measure repetition
        repetition = np.mean([calculate_repetition(s) for s in samples])
        metrics['repetition_rate'].append(repetition)

        # Measure coherence (human eval or model-based)
        coherence = np.mean([score_coherence(s) for s in samples])
        metrics['coherence_score'].append(coherence)

    return {k: np.mean(v) for k, v in metrics.items()}
```

### Performance Benchmarks

| Algorithm | Overhead | Quality | Use Case |
|-----------|----------|---------|----------|
| Greedy | 0% | Low diversity | Deterministic output |
| Temperature | <1% | Good | General purpose |
| Top-K | 1-2% | Good | Fast, simple |
| Top-P | 2-3% | Better | Adaptive |
| Min-P | 2-3% | Best | Handling uncertainty |
| Mirostat | 5-10% | Excellent | Quality control |
| Beam Search | 400% | High likelihood | Translation, summarization |

## Production Recommendations

### General Text Generation

```python
config = {
    'temperature': 0.7,
    'top_p': 0.9,
    'min_p': 0.05,
    'repeat_penalty': 1.1,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
```

### Code Generation

```python
config = {
    'temperature': 0.2,
    'top_p': 0.95,
    'min_p': 0.0,
    'repeat_penalty': 1.0,  # Don't penalize valid code repetition
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}
```

### Creative Writing

```python
config = {
    'temperature': 0.8,
    'top_p': 0.85,
    'min_p': 0.1,
    'repeat_penalty': 1.2,
    'frequency_penalty': 0.5,
    'presence_penalty': 0.3
}
```

### Mirostat for Quality

```python
config = {
    'mirostat': 2,  # Use Mirostat V2
    'mirostat_tau': 5.0,  # Target perplexity
    'mirostat_eta': 0.1,  # Learning rate
    'temperature': 1.0  # Mirostat handles temp internally
}
```

## Interview Questions

1. **Explain the difference between top-k and top-p sampling.**
   - Top-k: Fixed number of tokens (context-independent)
   - Top-p: Dynamic cutoff based on cumulative probability (adapts to distribution)
   - Top-p better for varying confidence levels

2. **What is the purpose of temperature in sampling?**
   - Controls randomness by scaling logits before softmax
   - temp < 1: More focused (sharper distribution)
   - temp > 1: More random (flatter distribution)
   - temp = 0: Greedy (deterministic)

3. **Design a sampling strategy for a medical chatbot.**
   - Low temperature (0.1-0.3) for factual accuracy
   - Top-p (0.95) to maintain some flexibility
   - No min-p (avoid filtering when very confident)
   - Repetition penalty (1.0) to avoid penalizing medical terms
   - Consider Mirostat for quality control

4. **How does Mirostat differ from temperature?**
   - Temperature: Static adjustment to distribution
   - Mirostat: Dynamic feedback loop controlling perplexity
   - Mirostat adapts based on actual output quality
   - Better for maintaining consistent quality

5. **Calculate the effect of temperature on probabilities.**
   ```python
   logits = [2.0, 1.0, 0.0]

   temp = 1.0: softmax([2, 1, 0]) = [0.66, 0.24, 0.09]
   temp = 0.5: softmax([4, 2, 0]) = [0.84, 0.14, 0.02]
   temp = 2.0: softmax([1, 0.5, 0]) = [0.49, 0.30, 0.18]
   ```

## Summary

Advanced sampling enables fine-grained control over generation:

✅ **Temperature**: Universal randomness control
✅ **Top-p/min-p**: Adaptive token filtering
✅ **Mirostat**: Perplexity-based quality control
✅ **Custom pipelines**: Combine strategies for optimal results

**Key insight**: Different tasks need different sampling strategies. Experiment and measure!

---

**Module 5 Complete!** You now understand advanced inference techniques for production LLM systems.

**Next Module**: [06-server-production](../../06-server-production/docs/)

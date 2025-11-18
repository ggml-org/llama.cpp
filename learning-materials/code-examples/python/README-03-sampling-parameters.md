# Example 03: Sampling Parameters

**Module**: 1.4 - Basic Inference
**Difficulty**: Intermediate
**Estimated Time**: 25 minutes

## Overview

This example provides a comprehensive exploration of sampling parameters (temperature, top-k, top-p, repeat penalty) and their effects on text generation. Through systematic experiments, you'll learn how to tune these parameters for different use cases.

## Learning Objectives

By completing this example, you will:
- ✅ Understand how temperature affects generation randomness
- ✅ Master top-k and top-p (nucleus) sampling
- ✅ Learn when to use greedy vs stochastic sampling
- ✅ Apply repeat penalty to prevent loops
- ✅ Choose optimal parameters for different tasks
- ✅ Understand the trade-offs between quality and creativity

## Prerequisites

- Completed examples 01 and 02
- Understanding of probability distributions (helpful but not required)
- A GGUF model file

## Installation

```bash
pip install llama-cpp-python
```

## Usage

```bash
# Run with default model
python 03-sampling-parameters.py

# Run with custom model
python 03-sampling-parameters.py path/to/model.gguf
```

## Experiments Overview

The script runs 5 systematic experiments:

1. **Temperature Effects** - Varies from 0.0 (deterministic) to 1.5 (very random)
2. **Top-K Sampling** - Tests different k values (1, 10, 40, 100, disabled)
3. **Top-P Sampling** - Tests nucleus sampling (0.1 to 1.0)
4. **Repeat Penalty** - Demonstrates repetition control (1.0 to 1.5)
5. **Combined Settings** - Shows recommended combinations for different tasks

## Key Concepts Explained

### 1. Temperature

**What it does**: Controls randomness by scaling logits before softmax.

```python
# Low temperature (0.0-0.3): Deterministic, factual
output = model(prompt, temperature=0.0)  # Always same output

# Medium temperature (0.5-0.8): Balanced
output = model(prompt, temperature=0.7)  # Recommended default

# High temperature (1.0+): Creative, diverse
output = model(prompt, temperature=1.2)  # Very random
```

**Mathematical intuition**:
```
Probability of token i = softmax(logits / temperature)

temperature → 0: Distribution becomes peaked (one token dominates)
temperature → ∞: Distribution becomes uniform (all tokens equally likely)
```

**When to use**:
- **Low (0.0-0.3)**: Facts, Q&A, code, structured output
- **Medium (0.5-0.8)**: General chat, explanations, balanced tasks
- **High (0.9-1.5)**: Creative writing, brainstorming, poetry

### 2. Top-K Sampling

**What it does**: Only considers the K most likely tokens.

```python
# Very focused (k=10)
output = model(prompt, top_k=10, temperature=0.7)

# Balanced (k=40) - common default
output = model(prompt, top_k=40, temperature=0.7)

# Diverse (k=100)
output = model(prompt, top_k=100, temperature=0.7)

# Disabled (k=0) - consider all tokens
output = model(prompt, top_k=0, temperature=0.7)
```

**How it works**:
```
1. Sort tokens by probability: [0.3, 0.2, 0.15, 0.1, ...]
2. Keep only top K tokens
3. Renormalize and sample from remaining tokens
```

**Pros**:
- Simple and effective
- Prevents sampling unlikely tokens

**Cons**:
- Fixed K may be too rigid
- Doesn't adapt to confidence level

### 3. Top-P (Nucleus) Sampling

**What it does**: Dynamically selects smallest set of tokens whose cumulative probability exceeds P.

```python
# Very focused (p=0.5)
output = model(prompt, top_p=0.5, temperature=0.7)

# Balanced (p=0.9) - recommended default
output = model(prompt, top_p=0.9, temperature=0.7)

# Diverse (p=0.95)
output = model(prompt, top_p=0.95, temperature=0.7)
```

**How it works**:
```
Token probabilities: [0.4, 0.3, 0.2, 0.05, 0.03, 0.02, ...]
Cumulative sum:      [0.4, 0.7, 0.9, 0.95, 0.98, 1.0,  ...]

For p=0.9:
- Include tokens until cumulative > 0.9
- Result: Keep [0.4, 0.3, 0.2] (sum = 0.9)
```

**Advantages over top-k**:
- Adaptive to model confidence
- High confidence → fewer tokens
- Low confidence → more tokens
- Generally better than top-k

### 4. Repeat Penalty

**What it does**: Penalizes tokens that already appeared.

```python
# No penalty (default)
output = model(prompt, repeat_penalty=1.0)

# Light penalty (recommended)
output = model(prompt, repeat_penalty=1.1)

# Strong penalty (prevent loops)
output = model(prompt, repeat_penalty=1.3)
```

**How it works**:
```
For each already-seen token:
    adjusted_prob = original_prob / repeat_penalty

Higher penalty → already-seen tokens less likely to be selected
```

**When to use**:
- 1.0: No issues with repetition
- 1.1: General purpose (slight penalty)
- 1.3: Model is looping or repeating phrases
- 1.5+: Desperate measures (may hurt coherence)

## Recommended Settings by Task

### Factual Q&A
```python
params = {
    "temperature": 0.0,      # Deterministic
    "top_k": 1,              # Greedy
    "top_p": 1.0,            # Disabled
    "repeat_penalty": 1.0    # None
}
```
**Why**: Facts should be consistent and deterministic.

### General Purpose Chat
```python
params = {
    "temperature": 0.7,      # Balanced
    "top_k": 40,             # Moderate diversity
    "top_p": 0.9,            # Balanced
    "repeat_penalty": 1.1    # Light penalty
}
```
**Why**: Good balance of coherence and variety.

### Creative Writing
```python
params = {
    "temperature": 0.9,      # More creative
    "top_k": 100,            # More diverse
    "top_p": 0.95,           # Diverse
    "repeat_penalty": 1.1    # Prevent loops
}
```
**Why**: Creativity requires more randomness and diversity.

### Code Generation
```python
params = {
    "temperature": 0.2,      # Focused
    "top_k": 40,             # Moderate
    "top_p": 0.9,            # Balanced
    "repeat_penalty": 1.05   # Slight penalty
}
```
**Why**: Code needs precision but some creativity for problem-solving.

### Translation
```python
params = {
    "temperature": 0.3,      # Focused
    "top_k": 20,             # Limited diversity
    "top_p": 0.9,            # Balanced
    "repeat_penalty": 1.0    # None (phrases may repeat)
}
```
**Why**: Accuracy is paramount, but some flexibility helps.

## Common Issues and Solutions

### Issue 1: Output is too repetitive
```
Output: "The cat sat on the mat. The cat sat on the mat. The cat..."
```
**Solutions**:
1. Increase repeat_penalty to 1.3
2. Increase temperature to 0.8
3. Increase top_k or top_p

### Issue 2: Output is incoherent
```
Output: "xylophone quantum butterfly telescope harmonious..."
```
**Solutions**:
1. Decrease temperature to 0.5
2. Decrease top_k to 20
3. Decrease top_p to 0.8

### Issue 3: Output is always the same
```
Output always identical, no variation
```
**Solutions**:
1. Increase temperature above 0.0
2. Enable top_k (set to 40) or top_p (set to 0.9)
3. Check if model supports stochastic sampling

### Issue 4: Output ignores prompt
```
Output not following instructions or context
```
**Solutions**:
1. Decrease temperature to 0.3
2. Improve prompt clarity
3. Try different model (may be model quality issue)

## Experiments to Try

### 1. Temperature Sweep
Run the same prompt with temperatures [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]:
```python
for temp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]:
    output = model("Once upon a time", temperature=temp, max_tokens=100)
    print(f"temp={temp}: {output}\n")
```

### 2. Compare Top-K vs Top-P
```python
# Top-K: Fixed number of tokens
output1 = model(prompt, temperature=0.7, top_k=40, top_p=1.0)

# Top-P: Adaptive number of tokens
output2 = model(prompt, temperature=0.7, top_k=0, top_p=0.9)

# Which produces better results?
```

### 3. Repetition Challenge
Prompt that triggers repetition:
```python
prompt = "The word 'the' is"
# Try different repeat_penalty values to fix it
```

### 4. Task-Specific Tuning
Test your task (e.g., summarization, question answering):
```python
# Start with defaults
params = {"temperature": 0.7, "top_k": 40, "top_p": 0.9}

# Adjust based on results
# Too creative? Lower temperature
# Too repetitive? Increase repeat_penalty
# Too deterministic? Increase temperature
```

## Performance Notes

- **Generation speed**: Not significantly affected by sampling parameters
- **Quality**: Proper tuning dramatically improves output quality
- **Consistency**: Lower temperature = more consistent results
- **Diversity**: Higher top_p/top_k = more diverse outputs

## Interview Topics

This example covers critical interview topics:

**Sampling Strategies**:
> Q: "Explain the difference between top-k and top-p sampling."
>
> A: Top-k considers a fixed number of tokens (e.g., top 40), while top-p (nucleus sampling) dynamically selects tokens until cumulative probability exceeds a threshold (e.g., 0.9). Top-p is more adaptive to model confidence and generally preferred.

**Parameter Tuning**:
> Q: "How would you tune generation parameters for a factual Q&A system vs. a creative writing assistant?"
>
> A: Q&A needs deterministic output (temperature=0.0, top_k=1) for consistency. Creative writing needs diversity (temperature=0.9, top_p=0.95) for interesting output. The key difference is the reliability vs. creativity trade-off.

**Production Considerations**:
> Q: "How do sampling parameters affect production inference systems?"
>
> A: Higher temperature and top_k increase generation time slightly but not significantly. The main consideration is quality: poorly tuned parameters can cause repetition, incoherence, or determinism. Production systems should have task-specific parameter profiles.

## Advanced Topics

### Combining Temperature with Top-K/Top-P
```python
# Both together for maximum control
output = model(
    prompt,
    temperature=0.7,  # Adjust distribution shape
    top_p=0.9,        # Filter tokens
    top_k=40          # Additional filtering
)
```

### Min-P Sampling (Advanced)
Some implementations support min-p:
```python
# Only consider tokens with prob > min_p * max_prob
output = model(prompt, min_p=0.05)
```

### Mirostat Sampling (Advanced)
Dynamically adjusts sampling to maintain perplexity:
```python
output = model(prompt, mirostat_mode=2, mirostat_tau=5.0)
```

## Next Steps

After completing this example:
- ✅ Experiment with your own prompts and tasks
- ✅ Try `04-context-management.py` for advanced context handling
- ✅ Build a parameter tuning tool for your application
- ✅ Read papers on sampling strategies (nucleus sampling, etc.)

## Related Documentation

- [Module 1.4: Basic Inference](../../modules/01-foundations/docs/)
- [Sampling Strategies Deep Dive](../../modules/02-core-implementation/docs/)
- [Lab 2.5: Sampling Experiments](../../modules/02-core-implementation/labs/)

## References

- Holtzman et al. (2019): "The Curious Case of Neural Text Degeneration" (introduces nucleus sampling)
- Fan et al. (2018): "Hierarchical Neural Story Generation" (explores sampling for creativity)

---

**Author**: Agent 3 (Code Developer)
**Last Updated**: 2025-11-18
**Module**: 1.4 - Basic Inference

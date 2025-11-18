# Lab 4: Sampling Algorithm Comparison

**Module 5 - Advanced Inference**
**Estimated Time**: 2-3 hours
**Difficulty**: Advanced

## Objectives

- Implement and compare all sampling algorithms
- Measure quality metrics (perplexity, diversity, coherence)
- Optimize sampling for different use cases
- Design custom sampling strategies
- Understand trade-offs between algorithms

## Part 1: Algorithm Implementation (45 min)

### Implement All Samplers

Using `../code/advanced_sampling.py` as reference, implement:

1. **Greedy** (baseline)
2. **Temperature** (0.1, 0.5, 0.7, 1.0, 1.5)
3. **Top-K** (k=10, 40, 80)
4. **Top-P** (p=0.8, 0.9, 0.95)
5. **Min-P** (min_p=0.01, 0.05, 0.1)
6. **Mirostat V2** (tau=3, 5, 8)
7. **Locally Typical** (epsilon=0.1, 0.2, 0.5)

**Test each** on same prompt, compare outputs

## Part 2: Quality Metrics (45 min)

### Measure Generation Quality

For each sampler, measure:

#### 1. Perplexity

```python
def calculate_perplexity(tokens, model):
    """Lower = more confident predictions"""
    logprobs = model.get_logprobs(tokens)
    return np.exp(-np.mean(logprobs))

# Run for all samplers
results = {}
for name, sampler in samplers.items():
    tokens = generate(prompt, sampler, n=100)
    ppl = calculate_perplexity(tokens, model)
    results[name] = ppl

print(f"Perplexity comparison:")
for name, ppl in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name}: {ppl:.2f}")
```

#### 2. Diversity (Unique N-grams)

```python
def calculate_diversity(samples):
    """Higher = more diverse outputs"""
    all_ngrams = set()
    for sample in samples:
        ngrams = zip(sample[:-1], sample[1:])  # Bigrams
        all_ngrams.update(ngrams)

    return len(all_ngrams) / len(samples)
```

#### 3. Repetition Rate

```python
def calculate_repetition(tokens):
    """Lower = less repetitive"""
    from collections import Counter
    counts = Counter(tokens)
    max_count = max(counts.values())
    return max_count / len(tokens)
```

**Create comparison table**:

| Algorithm | Perplexity | Diversity | Repetition | Coherence |
|-----------|------------|-----------|------------|-----------|
| Greedy | | | | |
| Temp=0.7 | | | | |
| Top-K | | | | |
| Top-P | | | | |
| Min-P | | | | |
| Mirostat | | | | |

## Part 3: Use Case Optimization (45 min)

### Test on Different Tasks

#### Task 1: Code Generation

**Goal**: Syntactically correct code

```python
prompts = [
    "def fibonacci(n):",
    "class BinaryTree:",
    "import numpy as np\n\ndef "
]

# Test samplers
for prompt in prompts:
    for sampler in samplers.values():
        code = generate(prompt, sampler)
        try:
            compile(code, '<string>', 'exec')
            valid = True
        except:
            valid = False

        print(f"{sampler}: {'✓' if valid else '✗'}")
```

**Best sampler for code**: ___________

#### Task 2: Creative Writing

**Goal**: Diverse, interesting stories

```python
prompt = "Once upon a time in a magical forest,"

# Generate 5 stories with each sampler
for name, sampler in samplers.items():
    stories = [generate(prompt, sampler) for _ in range(5)]

    # Measure diversity
    diversity = calculate_diversity(stories)
    print(f"{name}: diversity={diversity:.2f}")
```

**Best sampler for creative writing**: ___________

#### Task 3: Factual Q&A

**Goal**: Accurate, concise answers

```python
questions = [
    "What is the capital of France?",
    "When was Python created?",
    "Who invented the telephone?"
]

# Test accuracy
for question in questions:
    for name, sampler in samplers.items():
        answer = generate(question, sampler)
        correct = verify_answer(question, answer)
        print(f"{name}: {question} → {'✓' if correct else '✗'}")
```

**Best sampler for Q&A**: ___________

## Part 4: Mirostat Deep Dive (30 min)

### Perplexity Control

Test Mirostat's ability to maintain target perplexity:

```python
mirostat = MirostatV2Sampler(tau=5.0, eta=0.1)

surprises = []
for _ in range(100):
    logits = model.forward(tokens)
    token = mirostat.sample(logits)

    probs = softmax(logits)
    surprise = -np.log(probs[token])
    surprises.append(surprise)

    tokens.append(token)

# Plot surprise over time
plt.figure(figsize=(12, 6))
plt.plot(surprises, alpha=0.7, label='Observed Surprise')
plt.axhline(y=5.0, color='r', linestyle='--', label='Target (tau=5.0)')
plt.xlabel('Token')
plt.ylabel('Surprise')
plt.title('Mirostat Perplexity Control')
plt.legend()
plt.savefig('mirostat_control.png')
```

**Questions**:
1. Does Mirostat converge to target tau?
2. How many tokens until convergence?
3. Effect of different tau values?

## Part 5: Custom Sampling Strategy (30 min)

### Design Task-Aware Sampler

Implement sampler that adapts to context:

```python
class ContextAwareSampler:
    def __init__(self):
        self.code_sampler = TemperatureSampler(0.2)
        self.creative_sampler = MirostatV2Sampler(tau=6.0)
        self.factual_sampler = TemperatureSampler(0.5)

    def sample(self, logits, context):
        # Detect context type
        if self._is_code_context(context):
            return self.code_sampler.sample(logits)
        elif self._is_creative_context(context):
            return self.creative_sampler.sample(logits)
        else:
            return self.factual_sampler.sample(logits)

    def _is_code_context(self, context):
        # Check for code keywords
        code_keywords = ['def', 'class', 'import', 'function']
        return any(kw in context for kw in code_keywords)

    def _is_creative_context(self, context):
        creative_keywords = ['story', 'once upon', 'imagine']
        return any(kw in context for kw in creative_keywords)
```

**Test** on mixed prompts, measure improvement

## Part 6: Production Recommendations (30 min)

### Create Sampling Guidelines

Based on experiments, create decision matrix:

```markdown
## Sampling Strategy Guide

### Code Generation
- Algorithm: Temperature (0.1-0.3)
- Why: Need deterministic, correct output
- Settings: temp=0.2, top_p=0.95

### Creative Writing
- Algorithm: Mirostat V2
- Why: Balance quality and diversity
- Settings: tau=6.0, eta=0.1

### Factual Q&A
- Algorithm: Temperature (0.3-0.5)
- Why: Accurate but natural
- Settings: temp=0.4, top_p=0.9

### Brainstorming
- Algorithm: Temperature (1.0-1.5)
- Why: Maximum diversity
- Settings: temp=1.2, min_p=0.1

### General Text
- Algorithm: Combined (Temp + Top-P + Min-P)
- Why: Balanced approach
- Settings: temp=0.7, top_p=0.9, min_p=0.05
```

## Deliverables

1. **Benchmarking Report**
   - Quality metrics for all samplers
   - Performance comparison table
   - Use case recommendations

2. **Graphs**
   - Perplexity comparison
   - Diversity vs repetition plot
   - Mirostat convergence plot

3. **Code Implementation**
   - All samplers implemented
   - Custom context-aware sampler
   - Test suite

4. **Production Guide**
   - Sampling strategy decision tree
   - Parameter recommendations
   - Troubleshooting tips

## Evaluation

- **Correctness** (40%): Accurate implementations
- **Analysis** (30%): Insightful comparisons
- **Completeness** (20%): All sections done
- **Recommendations** (10%): Practical guidelines

## Extensions

1. **Ensemble Sampling**: Combine multiple samplers
2. **Beam Search**: Implement for comparison
3. **Human Evaluation**: A/B test outputs
4. **Parameter Tuning**: Auto-optimize settings

## Resources

- Code: `../code/advanced_sampling.py`
- Paper: "Mirostat: A Neural Text Decoding Algorithm"
- llama.cpp sampling: `src/llama-sampling.cpp`

## Tips

- Run each sampler multiple times (variance)
- Use same prompts for fair comparison
- Consider both quality AND speed
- Temperature 0.7 is good starting point
- Mirostat great for long-form generation

Good luck! 🎯

# Tutorial: Choosing the Right Quantization

**Module 3** | **Estimated Time**: 45 minutes | **Difficulty**: Beginner to Intermediate

## Introduction

Choosing the right quantization format is one of the most important decisions when deploying an LLM. This tutorial will guide you through a systematic process to select the optimal quantization for your specific use case.

## The Decision Framework

### Step 1: Define Your Requirements

Before looking at any numbers, answer these questions:

**1. What's your memory budget?**
- Unlimited (32GB+ RAM): Consider Q8_0 or Q6_K
- Comfortable (16-24GB RAM): Q5_K_M is ideal
- Moderate (8-16GB RAM): Q4_K_M recommended
- Tight (4-8GB RAM): Q4_K_S or Q3_K_M
- Very tight (<4GB RAM): Q3_K_S or smaller model

**2. What's your quality requirement?**
- Maximum quality (research, production): Q8_0
- Excellent quality (most production): Q5_K_M or Q6_K
- Good quality (general use): Q4_K_M
- Acceptable trade-off (size-constrained): Q4_K_S
- Experimental (testing limits): Q3_K_M or Q2_K

**3. What type of task?**
- Code generation: Higher precision (Q5_K_M minimum)
- Math/reasoning: Higher precision (Q5_K_M or Q6_K)
- General chat: Q4_K_M works well
- Creative writing: Can tolerate Q4_K_S or Q3_K_M
- Translation: Q5_K_M recommended

**4. What's your hardware?**
- Modern CPU (AVX2/AVX-512): K-quants optimal
- Older CPU: Legacy formats (Q4_0, Q5_0) may be faster
- Apple Silicon: K-quants well-optimized
- GPU (CUDA): All formats work, choose based on VRAM
- Mobile/Edge: Q4_K_S or Q3_K_M

### Step 2: The Quick Selection Guide

Based on your answers, here's a quick reference:

```
Start Here (Universal Recommendation):
└─ Q4_K_M

If you need better quality:
└─ Q5_K_M

If you need better than Q5_K_M:
└─ Q6_K or Q8_0

If Q4_K_M doesn't fit:
└─ Q4_K_S

If Q4_K_S still doesn't fit:
└─ Q3_K_M (test quality carefully)
```

## Detailed Walk-Through

### Example 1: Chatbot Application

**Scenario**: Building a customer service chatbot

**Requirements**:
- Will run on cloud VMs with 16GB RAM
- Needs to handle 100+ concurrent users
- Responses should be coherent and helpful
- Cost optimization important

**Analysis**:

1. **Memory**: 16GB allows for 7B model at any quantization
2. **Quality**: General chat doesn't need maximum precision
3. **Scale**: Multiple concurrent users → want fastest inference
4. **Cost**: Lower precision = faster = fewer servers

**Decision Process**:

```
Starting point: Q4_K_M (4.1 GB)
✓ Fits easily (4.1 GB < 16 GB)
✓ Fast inference (~25 tokens/sec on typical CPU)
✓ Quality excellent for chat (perplexity increase ~2-3%)

Test Q4_K_S (3.9 GB)?
→ Only marginally smaller
→ Slightly lower quality
→ Not worth trade-off for server deployment

Test Q5_K_M (4.8 GB)?
→ Still fits easily
→ Better quality (~1% perplexity increase)
→ Slightly slower (~22 tokens/sec)
→ For general chat, Q4_K_M sufficient
```

**Final Choice**: **Q4_K_M**

**Reasoning**: Perfect balance of speed and quality for chat application. The quality improvement of Q5_K_M not necessary for customer service use case, and Q4_K_M's speed advantage helps with concurrent users.

### Example 2: Code Generation Tool

**Scenario**: Building a code completion assistant

**Requirements**:
- Will run on developer laptops (8-32GB RAM)
- Must generate syntactically correct code
- Speed important for IDE integration
- Quality critical (incorrect code is useless)

**Analysis**:

1. **Memory**: Varies, but assume 16GB minimum
2. **Quality**: CRITICAL - code must compile
3. **Speed**: Important for UX
4. **Task**: Code generation very sensitive to quantization

**Decision Process**:

```
Starting point: Q4_K_M
✗ Code generation quality may degrade
→ Test on code benchmarks (HumanEval)

Test Q5_K_M (4.8 GB):
✓ Fits on 8GB+ laptops
✓ Better quality for complex reasoning
✓ Small speed reduction acceptable

Test Q6_K (5.5 GB):
✓ Fits on 16GB laptops easily
✓ Near-perfect quality
✗ Slower than Q5_K_M
→ Diminishing returns vs Q5_K_M

Benchmark Results (HumanEval):
- Q4_K_M: 48% pass rate
- Q5_K_M: 52% pass rate
- Q6_K: 53% pass rate
```

**Final Choice**: **Q5_K_M**

**Reasoning**: Best trade-off for code generation. Q6_K only slightly better but noticeably slower. Q4_K_M has meaningful quality degradation on code tasks. Q5_K_M provides near-optimal quality with good speed.

### Example 3: Mobile App

**Scenario**: LLM-powered mobile writing assistant

**Requirements**:
- Must run on mobile devices (4-6GB RAM total)
- App needs ~2GB for UI, OS, etc.
- Quality should be good but not perfect
- Size is PRIMARY constraint

**Analysis**:

1. **Memory**: 4-6GB total, ~2-3GB available for model
2. **Quality**: Important but can trade for size
3. **Size**: CRITICAL constraint
4. **Platform**: Mobile (ARM, NEON optimized)

**Decision Process**:

```
Maximum model size: ~3GB

Options that fit:
- Q4_K_S: 3.9 GB ✗ Too large
- Q3_K_M: 3.3 GB ✓ Fits
- Q3_K_S: 3.0 GB ✓ Fits with headroom
- Q2_K: 2.5 GB ✓ Plenty of room

Test Q3_K_M:
✓ Fits (3.3 GB)
✓ Reasonable quality for writing
✗ Tight fit, may cause issues

Test Q3_K_S:
✓ Fits with margin (3.0 GB)
✓ Slightly lower quality but acceptable
✓ Better stability (more RAM for OS)

Test Q2_K:
✓ Fits easily
✗ Significant quality degradation
✗ Not acceptable for writing assistant
```

**Final Choice**: **Q3_K_S**

**Reasoning**: Q3_K_M would work but tight fit risks out-of-memory errors. Q3_K_S provides sufficient quality for writing assistance with comfortable margin. Q2_K too aggressive for production use.

## The Testing Protocol

Don't just choose based on theory—test!

### Step 1: Create Test Set

```bash
# Create representative prompts
cat > test_prompts.txt << 'EOF'
Explain quantum computing in simple terms.
Write a Python function to sort a list.
Translate the following to French: Hello, how are you?
Solve this math problem: If x + 5 = 12, what is x?
Write a creative story beginning with "Once upon a time..."
EOF
```

### Step 2: Test Multiple Quantizations

```bash
#!/bin/bash
# test_quantizations.sh

FORMATS=(q4_k_s q4_k_m q5_k_s q5_k_m q6_k)
MODEL_BASE="llama-2-7b"

for format in "${FORMATS[@]}"; do
    echo "Testing $format..."

    # Generate outputs
    ./llama-cli \
        -m ${MODEL_BASE}-${format}.gguf \
        -f test_prompts.txt \
        -n 100 \
        --temp 0.7 \
        --seed 42 \
        > outputs_${format}.txt

    # Measure perplexity
    ./llama-perplexity \
        -m ${MODEL_BASE}-${format}.gguf \
        -f wikitext-2-test.txt \
        > perplexity_${format}.txt

    # Measure speed
    ./llama-bench \
        -m ${MODEL_BASE}-${format}.gguf \
        -n 128 \
        -r 5 \
        > benchmark_${format}.txt
done
```

### Step 3: Compare and Decide

Review outputs side-by-side:

```bash
# Compare code generation quality
diff outputs_q4_k_m.txt outputs_q5_k_m.txt

# Compare metrics
grep "perplexity" perplexity_*.txt
grep "tokens" benchmark_*.txt
```

## Common Pitfalls to Avoid

### Pitfall 1: Choosing Too Aggressive Quantization

**Problem**: "Q2_K is so small, let's use that!"

**Reality**: Quality degradation makes it unusable for most tasks

**Solution**: Always test quality on your specific use case

### Pitfall 2: Not Considering Hardware

**Problem**: "Q8_0 has best quality, let's use that everywhere"

**Reality**: May not fit in memory on deployment hardware

**Solution**: Match quantization to deployment constraints

### Pitfall 3: Ignoring Task Sensitivity

**Problem**: Using Q4_K_M for code generation because it works for chat

**Reality**: Different tasks have different quality requirements

**Solution**: Test on task-specific benchmarks

### Pitfall 4: Optimizing for Wrong Metric

**Problem**: "This format is 0.3% better on perplexity!"

**Reality**: User experience depends on actual generation quality

**Solution**: Evaluate end-to-end task performance, not just perplexity

## Advanced Considerations

### Multi-Format Deployment

Sometimes best strategy is using multiple formats:

```python
class AdaptiveModelServer:
    def __init__(self):
        self.models = {
            'fast': load_model('model-q4_k_s.gguf'),     # Quick responses
            'balanced': load_model('model-q4_k_m.gguf'), # Default
            'quality': load_model('model-q5_k_m.gguf'),  # Premium requests
        }

    def generate(self, prompt, priority='balanced'):
        model = self.models[priority]
        return model.generate(prompt)
```

### A/B Testing in Production

Don't trust benchmarks alone:

```python
# Gradual rollout
if user.id % 10 < 3:
    model = 'q4_k_m'  # 30% of users
else:
    model = 'q5_k_m'  # 70% of users

# Collect metrics
log_user_satisfaction(user, model, response)

# Analyze after 1 week
# Switch all users to winner
```

## Quick Reference Chart

| Use Case | Model Size | Recommended | Alternative | Avoid |
|----------|------------|-------------|-------------|-------|
| Chatbot | 7B | Q4_K_M | Q5_K_M | Q2_K, Q3_K_S |
| Code Gen | 7B-13B | Q5_K_M | Q6_K | <Q4_K_M |
| Creative Writing | 7B | Q4_K_S | Q3_K_M | Q2_K |
| Translation | 7B-13B | Q5_K_M | Q6_K | <Q4_K_M |
| Math/Reasoning | 13B+ | Q5_K_M | Q6_K, Q8_0 | <Q5_K_M |
| Mobile | 7B | Q3_K_S | Q3_K_M | Q4_* |
| Edge Device | 7B | Q3_K_M | Q4_K_S | >Q4_K_M |
| Server (unlimited) | Any | Q5_K_M | Q6_K, Q8_0 | - |
| API (cost-optimized) | 7B | Q4_K_M | Q4_K_S | <Q4_K_S |

## Practical Exercise

**Your Turn**: You need to deploy a model for these scenarios. What quantization do you choose?

1. **Scenario**: Q&A bot for technical documentation, running on AWS t3.xlarge (16GB RAM)
   - **Your Choice**: ________________
   - **Reasoning**: ________________

2. **Scenario**: Creative writing assistant for iOS app, must fit in 4GB
   - **Your Choice**: ________________
   - **Reasoning**: ________________

3. **Scenario**: Code review tool running on developer workstations (32GB RAM), quality critical
   - **Your Choice**: ________________
   - **Reasoning**: ________________

4. **Scenario**: Batch translation service, processing millions of documents, cost is primary concern
   - **Your Choice**: ________________
   - **Reasoning**: ________________

## Summary

**Decision Process**:
1. ✅ Define constraints (memory, quality, speed)
2. ✅ Start with Q4_K_M as baseline
3. ✅ Move up/down based on requirements
4. ✅ Test on representative workload
5. ✅ Measure actual task performance
6. ✅ Validate in production-like environment

**Remember**:
- Q4_K_M is the "safe default" for most uses
- Test quality on YOUR specific use case
- Consider deployment constraints
- Don't over-optimize for benchmarks
- When in doubt, choose one tier higher than minimum

## Next Steps

- [Tutorial 2: Optimizing for Your Hardware](./02-optimizing-for-hardware.md)
- [Lab 1: Quantization Impact Analysis](../labs/lab1-quantization-impact-analysis.md)
- [GGUF Quantization Formats Guide](../docs/02-gguf-quantization-formats.md)

---

**Author**: Agent 4 (Tutorial Specialist)
**Module**: 3 - Quantization & Optimization
**Last Updated**: 2025-11-18

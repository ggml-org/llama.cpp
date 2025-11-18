# Lab 1: Quantization Impact Analysis

**Module 3** | **Estimated Time**: 2-3 hours | **Difficulty**: Intermediate

## Learning Objectives

By completing this lab, you will:
- Quantize a model to multiple formats
- Measure perplexity impact of different quantizations
- Analyze quality vs size trade-offs
- Make data-driven quantization decisions
- Create a comprehensive comparison report

## Prerequisites

- Completed Module 3, Lessons 1-2
- llama.cpp built and working
- A GGUF model file (FP16 or unquantized)
- WikiText-2 test dataset (or similar)
- Python with pandas and matplotlib installed

## Lab Setup

### 1. Download Test Data

```bash
# Create data directory
mkdir -p lab_data
cd lab_data

# Download WikiText-2 test set
wget https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/wiki.test.raw \
    -O wikitext-2-test.txt

# Verify download
wc -l wikitext-2-test.txt
# Should show ~4358 lines
```

### 2. Verify Tools

```bash
# Check llama-quantize
./llama-quantize --help

# Check llama-perplexity
./llama-perplexity --help

# Check llama-cli
./llama-cli --help
```

## Part 1: Model Quantization (30 minutes)

### Step 1: Prepare Base Model

If you don't have an FP16 GGUF model, you'll need to start with one. For this lab, we'll assume you have `llama-2-7b-fp16.gguf`.

```bash
# List your model
ls -lh llama-2-7b-fp16.gguf
# Note the size (should be ~13-14 GB for 7B model)
```

### Step 2: Quantize to Multiple Formats

Let's create several quantizations to compare:

```bash
# Q8_0 - Nearly lossless
./llama-quantize llama-2-7b-fp16.gguf llama-2-7b-q8_0.gguf Q8_0

# Q6_K - High quality
./llama-quantize llama-2-7b-fp16.gguf llama-2-7b-q6_k.gguf Q6_K

# Q5_K_M - Excellent balance
./llama-quantize llama-2-7b-fp16.gguf llama-2-7b-q5_k_m.gguf Q5_K_M

# Q4_K_M - Recommended default
./llama-quantize llama-2-7b-fp16.gguf llama-2-7b-q4_k_m.gguf Q4_K_M

# Q4_K_S - Smaller variant
./llama-quantize llama-2-7b-fp16.gguf llama-2-7b-q4_k_s.gguf Q4_K_S

# Q3_K_M - Aggressive compression
./llama-quantize llama-2-7b-fp16.gguf llama-2-7b-q3_k_m.gguf Q3_K_M
```

### Step 3: Record Model Sizes

```bash
# List all quantizations
ls -lh llama-2-7b-*.gguf

# Save sizes to file
ls -lh llama-2-7b-*.gguf > model_sizes.txt
```

**Question 1**: Fill in the table below with your model sizes:

| Format | Size (GB) | Compression Ratio vs FP16 |
|--------|-----------|---------------------------|
| FP16   | _____     | 1.0x                      |
| Q8_0   | _____     | _____                     |
| Q6_K   | _____     | _____                     |
| Q5_K_M | _____     | _____                     |
| Q4_K_M | _____     | _____                     |
| Q4_K_S | _____     | _____                     |
| Q3_K_M | _____     | _____                     |

## Part 2: Perplexity Measurement (45 minutes)

### Step 1: Measure Baseline Perplexity (FP16)

```bash
./llama-perplexity \
    -m llama-2-7b-fp16.gguf \
    -f lab_data/wikitext-2-test.txt \
    --perplexity \
    | tee results_fp16.txt
```

**Note**: This may take several minutes. Record the final perplexity value.

### Step 2: Measure Each Quantization

Run perplexity tests for each quantization:

```bash
for format in q8_0 q6_k q5_k_m q4_k_m q4_k_s q3_k_m; do
    echo "Testing $format..."
    ./llama-perplexity \
        -m llama-2-7b-${format}.gguf \
        -f lab_data/wikitext-2-test.txt \
        --perplexity \
        | tee results_${format}.txt
done
```

### Step 3: Extract and Compare Results

Create a results file `perplexity_results.txt`:

```
# Extract perplexity values
for file in results_*.txt; do
    echo "$file:"
    grep "perplexity" $file
done
```

**Question 2**: Fill in your perplexity measurements:

| Format | Perplexity | Increase vs FP16 (%) |
|--------|------------|----------------------|
| FP16   | _____      | 0%                   |
| Q8_0   | _____      | _____                |
| Q6_K   | _____      | _____                |
| Q5_K_M | _____      | _____                |
| Q4_K_M | _____      | _____                |
| Q4_K_S | _____      | _____                |
| Q3_K_M | _____      | _____                |

**Question 3**: Which quantizations maintain <2% perplexity increase?

**Question 4**: Which quantization would you eliminate from production use based on perplexity alone?

## Part 3: Quality Analysis (30 minutes)

### Step 1: Generate Sample Text

Create a test prompt file `test_prompts.txt`:

```
Prompt 1: Once upon a time
Prompt 2: The capital of France is
Prompt 3: To solve this math problem, first we need to
Prompt 4: In the field of artificial intelligence,
Prompt 5: Step-by-step reasoning: If x = 5 and y = 3, then x + y =
```

### Step 2: Generate Outputs from Each Model

```bash
#!/bin/bash
# generate_outputs.sh

PROMPTS=(
    "Once upon a time"
    "The capital of France is"
    "To solve this math problem, first we need to"
    "In the field of artificial intelligence,"
    "Step-by-step reasoning: If x = 5 and y = 3, then x + y ="
)

MODELS=(fp16 q8_0 q6_k q5_k_m q4_k_m q4_k_s q3_k_m)

for model in "${MODELS[@]}"; do
    echo "Generating from $model..."
    mkdir -p outputs/$model

    for i in "${!PROMPTS[@]}"; do
        echo "Prompt $((i+1)): ${PROMPTS[$i]}"
        ./llama-cli \
            -m llama-2-7b-${model}.gguf \
            -p "${PROMPTS[$i]}" \
            -n 100 \
            --temp 0.7 \
            --top-p 0.9 \
            --seed 42 \
            > outputs/${model}/prompt_$((i+1)).txt
    done
done
```

### Step 3: Qualitative Comparison

**Question 5**: Compare outputs for Prompt 2 ("The capital of France is"):

FP16 output: _______________________________________________

Q5_K_M output: _______________________________________________

Q4_K_M output: _______________________________________________

Q3_K_M output: _______________________________________________

**Question 6**: Do you notice quality degradation in any quantizations? Which ones?

**Question 7**: For reasoning tasks (Prompt 5), which quantization is the minimum acceptable?

## Part 4: Performance Measurement (30 minutes)

### Step 1: Measure Inference Speed

```bash
#!/bin/bash
# benchmark_performance.sh

MODELS=(fp16 q8_0 q6_k q5_k_m q4_k_m q4_k_s q3_k_m)
PROMPT="This is a benchmark test prompt for measuring inference speed."

echo "Model,PromptTime_ms,GenerationTime_ms,TokensPerSec" > performance_results.csv

for model in "${MODELS[@]}"; do
    echo "Benchmarking $model..."

    # Run inference and capture timing
    output=$(./llama-cli \
        -m llama-2-7b-${model}.gguf \
        -p "$PROMPT" \
        -n 100 \
        --log-disable 2>&1)

    # Extract timing info (you'll need to parse the output)
    # This is a simplified example
    echo "$model,$output" >> performance_results.csv
done
```

**Question 8**: Fill in performance results:

| Format | Tokens/Second | Speedup vs FP16 |
|--------|---------------|-----------------|
| FP16   | _____         | 1.0x            |
| Q8_0   | _____         | _____           |
| Q6_K   | _____         | _____           |
| Q5_K_M | _____         | _____           |
| Q4_K_M | _____         | _____           |
| Q4_K_S | _____         | _____           |
| Q3_K_M | _____         | _____           |

### Step 2: Memory Profiling

Monitor memory usage during inference:

```bash
# Run with monitoring
/usr/bin/time -v ./llama-cli \
    -m llama-2-7b-q4_k_m.gguf \
    -p "Test" \
    -n 10 2>&1 | grep "Maximum resident"
```

**Question 9**: Which quantization would you choose if you have only 8GB of RAM?

## Part 5: Trade-off Analysis (30 minutes)

### Create Comparison Matrix

Based on your measurements, create a decision matrix:

| Format | Size | Perplexity | Speed | Quality | Best For |
|--------|------|------------|-------|---------|----------|
| FP16   | ★    | ★★★★★      | ★     | ★★★★★   | Reference |
| Q8_0   | ★★   | ★★★★★      | ★★    | ★★★★★   | _________ |
| Q6_K   | ★★★  | ★★★★★      | ★★    | ★★★★★   | _________ |
| Q5_K_M | ★★★  | ★★★★★      | ★★★   | ★★★★    | _________ |
| Q4_K_M | ★★★★ | ★★★★       | ★★★   | ★★★★    | _________ |
| Q4_K_S | ★★★★★| ★★★        | ★★★★  | ★★★     | _________ |
| Q3_K_M | ★★★★★| ★★         | ★★★★  | ★★      | _________ |

### Visualization

Create a Python script to visualize trade-offs:

```python
import matplotlib.pyplot as plt
import pandas as pd

# Enter your data
data = {
    'format': ['FP16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q4_K_S', 'Q3_K_M'],
    'size_gb': [13.5, 7.2, 5.5, 4.8, 4.1, 3.9, 3.3],  # Replace with your values
    'perplexity': [5.68, 5.70, 5.72, 5.75, 5.82, 5.85, 6.10],  # Replace with your values
    'tokens_per_sec': [12.5, 18.2, 20.1, 22.3, 24.5, 25.8, 27.1]  # Replace with your values
}

df = pd.DataFrame(data)

# Plot 1: Size vs Perplexity
plt.figure(figsize=(10, 6))
plt.scatter(df['size_gb'], df['perplexity'], s=100)
for i, txt in enumerate(df['format']):
    plt.annotate(txt, (df['size_gb'][i], df['perplexity'][i]))
plt.xlabel('Model Size (GB)')
plt.ylabel('Perplexity (lower is better)')
plt.title('Size vs Quality Trade-off')
plt.grid(alpha=0.3)
plt.savefig('size_vs_quality.png')

# Plot 2: Size vs Performance
plt.figure(figsize=(10, 6))
plt.scatter(df['size_gb'], df['tokens_per_sec'], s=100)
for i, txt in enumerate(df['format']):
    plt.annotate(txt, (df['size_gb'][i], df['tokens_per_sec'][i]))
plt.xlabel('Model Size (GB)')
plt.ylabel('Tokens per Second')
plt.title('Size vs Performance Trade-off')
plt.grid(alpha=0.3)
plt.savefig('size_vs_performance.png')

plt.show()
```

## Lab Questions

Answer the following questions based on your analysis:

**Question 10**: If you had to choose ONE quantization for production deployment, which would it be and why?

**Answer**: _________________________________________________________________

**Question 11**: What is the "sweet spot" quantization that balances all factors (size, quality, speed)?

**Answer**: _________________________________________________________________

**Question 12**: For which use cases would you recommend Q8_0 over Q4_K_M?

**Answer**: _________________________________________________________________

**Question 13**: What is the minimum acceptable quantization for:
- Chatbot: _______
- Code generation: _______
- Creative writing: _______
- Math reasoning: _______

**Question 14**: Based on perplexity alone, can you accurately predict generation quality? Why or why not?

**Answer**: _________________________________________________________________

## Challenge Exercise

### Advanced: Multi-Metric Scoring

Create a scoring system that combines all metrics:

```python
def calculate_score(size_gb, perplexity, tokens_per_sec, fp16_baseline):
    """
    Calculate overall score (higher is better)

    Weights:
    - Size: 30% (smaller is better)
    - Quality: 40% (lower perplexity is better)
    - Speed: 30% (higher is better)
    """
    # Normalize metrics (0-1 scale)
    size_score = 1 - (size_gb / fp16_baseline['size_gb'])
    quality_score = fp16_baseline['perplexity'] / perplexity
    speed_score = tokens_per_sec / fp16_baseline['tokens_per_sec']

    # Weighted average
    total_score = (0.3 * size_score +
                   0.4 * quality_score +
                   0.3 * speed_score)

    return total_score

# Calculate for all formats
# Which format has the highest score?
```

**Question 15**: Which format has the highest combined score with these weights?

**Answer**: _________________________________________________________________

**Question 16**: If you change the weights to prioritize speed (50%) over size (20%) and quality (30%), does the winner change?

**Answer**: _________________________________________________________________

## Lab Deliverables

Submit the following:
1. Completed tables with all measurements
2. Answers to all questions
3. Generated visualizations (PNG files)
4. Sample outputs from different quantizations
5. Your final recommendation report (1 page)

## Reflection

**Question 17**: What was the most surprising finding from this lab?

**Answer**: _________________________________________________________________

**Question 18**: How would you explain the quality vs size trade-off to a non-technical stakeholder?

**Answer**: _________________________________________________________________

## Further Exploration

- Test with different datasets (code, math, creative writing)
- Measure quality on specific benchmarks (MMLU, HellaSwag)
- Compare k-quant formats to legacy formats (Q4_0 vs Q4_K_M)
- Investigate extreme quantizations (Q2_K, IQ formats)
- Test on different hardware (ARM, Apple Silicon, AMD)

## Resources

- [Quantization Fundamentals](../docs/01-quantization-fundamentals.md)
- [GGUF Quantization Formats](../docs/02-gguf-quantization-formats.md)
- [Benchmarking Guide](../docs/05-benchmarking-testing.md)
- [quantization_comparison.py](../code/quantization_comparison.py)

---

**Completion Criteria**:
- [ ] All models quantized successfully
- [ ] Perplexity measured for all formats
- [ ] Performance benchmarked
- [ ] Qualitative comparison completed
- [ ] Visualizations generated
- [ ] All questions answered
- [ ] Final recommendation written

**Estimated Time**: 2-3 hours
**Difficulty**: Intermediate
**Author**: Agent 4 (Lab Designer)
**Module**: 3 - Quantization & Optimization

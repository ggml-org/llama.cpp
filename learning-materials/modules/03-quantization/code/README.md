# Module 3 Code Examples

This directory contains comprehensive code examples for quantization and optimization.

## Available Scripts

### 1. quantization_comparison.py

**Purpose**: Compare different quantization formats for quality and performance.

**Usage**:
```bash
# Compare specific formats
python quantization_comparison.py \
    --model model.gguf \
    --formats Q4_K_M Q5_K_M Q8_0 \
    --test-file wikitext-2-test.txt

# Generate visualization and report
python quantization_comparison.py \
    --model model.gguf \
    --formats Q4_K_M Q5_K_M Q6_K Q8_0 \
    --test-file wikitext-2-test.txt \
    --output-viz comparison.png \
    --output-report comparison_report.md
```

**Features**:
- Automatic quantization of models
- Perplexity measurement
- Performance benchmarking
- Visualization generation
- Markdown report generation

**Dependencies**: pandas, matplotlib

---

### 2. format_converter.py

**Purpose**: Batch convert models to multiple quantization formats.

**Usage**:
```bash
# Convert to specific formats
python format_converter.py --model model.gguf --formats Q4_K_M Q5_K_M Q8_0

# Use presets
python format_converter.py --model model.gguf --preset production
python format_converter.py --model model.gguf --preset mobile

# Convert to all formats
python format_converter.py --model model.gguf --all

# Generate README for converted models
python format_converter.py \
    --model model.gguf \
    --preset production \
    --generate-readme
```

**Presets**:
- `mobile`: Q4_0, Q4_K_S, Q4_K_M
- `balanced`: Q4_K_M, Q5_K_M, Q6_K
- `quality`: Q5_K_M, Q6_K, Q8_0
- `experimental`: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L
- `production`: Q4_K_M, Q5_K_M, Q8_0

**Features**:
- Batch conversion
- Format verification
- MD5 checksum calculation
- Automatic README generation
- Conversion logging

---

### 3. performance_profiler.py

**Purpose**: Comprehensive performance profiling with thread scaling analysis.

**Usage**:
```bash
# Profile thread scaling
python performance_profiler.py \
    --model model.gguf \
    --profile threads \
    --threads 1,2,4,8,16

# Profile prompt length impact
python performance_profiler.py \
    --model model.gguf \
    --profile prompts \
    --prompt-lengths 50,100,200,500

# Full profiling suite
python performance_profiler.py \
    --model model.gguf \
    --profile all \
    --threads 1,2,4,8 \
    --prompt-lengths 50,100,200
```

**Features**:
- Thread scaling analysis
- Memory usage monitoring
- CPU utilization tracking
- Tokens per second measurement
- Visualization generation
- Markdown report generation

**Dependencies**: pandas, matplotlib, psutil

---

### 4. ggml_operations_example.cpp

**Purpose**: Demonstrate core GGML tensor operations.

**Compilation**:
```bash
# From llama.cpp root directory
g++ -std=c++17 \
    -I./include \
    -I./ggml/include \
    learning-materials/modules/03-quantization/code/ggml_operations_example.cpp \
    -L./build \
    -lggml \
    -o ggml_example

# Run
./ggml_example
```

**Examples Included**:
1. Matrix Multiplication
2. RMS Normalization (LLaMA-style)
3. Element-wise Operations (add, mul, GELU, SiLU)
4. Scaled Dot-Product Attention
5. Quantization/Dequantization
6. Computation Graph Building

**Features**:
- Performance measurement
- Memory management examples
- Graph-based computation
- Random data generation
- Multiple operation types

---

### 5. benchmark_automation.py

**Purpose**: Automated benchmarking suite for CI/CD integration.

**Usage**:
```bash
# Benchmark multiple models
python benchmark_automation.py \
    --models model1.gguf model2.gguf model3.gguf

# Use glob pattern
python benchmark_automation.py \
    --model-pattern "llama-7b-*.gguf"

# Compare with baseline
python benchmark_automation.py \
    --models model-new.gguf \
    --baseline baseline_results.json

# Use configuration file
python benchmark_automation.py \
    --config benchmark_config.yaml \
    --models model.gguf
```

**Configuration Example** (benchmark_config.yaml):
```yaml
llama_cpp_dir: '../../../'

benchmarks:
  perplexity:
    enabled: true
    test_file: 'wikitext-2-test.txt'
    context_size: 512

  performance:
    enabled: true
    prompt: 'Hello, world! This is a benchmark test.'
    n_predict: 128
    trials: 5
    thread_counts: [1, 2, 4, 8]

  memory:
    enabled: true

thresholds:
  perplexity_increase_percent: 5.0
  performance_decrease_percent: 10.0
  memory_increase_percent: 15.0

output:
  json: 'benchmark_results_{timestamp}.json'
  csv: 'benchmark_results_{timestamp}.csv'
  plots: true
  plots_dir: 'benchmark_plots'
```

**Features**:
- Multiple benchmark types (perplexity, performance, memory)
- Baseline comparison
- Regression detection
- Automated plotting
- CI/CD integration
- Configurable thresholds

**Dependencies**: pandas, matplotlib, pyyaml

---

## Installation

Install required Python dependencies:

```bash
pip install pandas matplotlib psutil pyyaml
```

## Common Workflows

### 1. Evaluate a New Model

```bash
# Step 1: Convert to different formats
python format_converter.py \
    --model base-model.gguf \
    --preset production \
    --generate-readme

# Step 2: Compare formats
python quantization_comparison.py \
    --model base-model.gguf \
    --formats Q4_K_M Q5_K_M Q8_0 \
    --test-file wikitext-2-test.txt

# Step 3: Profile performance
python performance_profiler.py \
    --model base-model-q4_k_m.gguf \
    --profile all
```

### 2. Optimize for Production

```bash
# Find optimal thread count
python performance_profiler.py \
    --model model.gguf \
    --profile threads \
    --threads 1,2,4,8,16,32

# Analyze results and choose best configuration
```

### 3. Continuous Benchmarking (CI/CD)

```bash
# In CI pipeline
python benchmark_automation.py \
    --models new-model-q4_k_m.gguf \
    --baseline production_baseline.json

# Exit code 0 if no regressions, 1 if regressions detected
```

## Output Files

Scripts generate various output files:

| Script | JSON | CSV | Plots | Report |
|--------|------|-----|-------|--------|
| quantization_comparison.py | ✅ | - | ✅ | ✅ |
| format_converter.py | ✅ (log) | - | - | ✅ (README) |
| performance_profiler.py | ✅ | - | ✅ | ✅ |
| benchmark_automation.py | ✅ | ✅ | ✅ | - |

## Tips and Best Practices

1. **Always use test data for perplexity**: Download WikiText-2 or use your own test set
2. **Run multiple trials**: Performance can vary, use at least 5 trials
3. **Control system load**: Close unnecessary applications during benchmarking
4. **Document your setup**: Save configuration, hardware specs, and environment
5. **Version control results**: Track benchmark results over time

## Troubleshooting

### "llama-quantize not found"

Ensure llama.cpp is built and the path is correct:
```bash
# Build llama.cpp first
cd ../../../../  # Navigate to llama.cpp root
mkdir build && cd build
cmake ..
make -j
```

### "Test file not found"

Download test data:
```bash
wget https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/wiki.test.raw -O wikitext-2-test.txt
```

### Import errors

Install dependencies:
```bash
pip install pandas matplotlib psutil pyyaml
```

## Further Reading

- [Quantization Fundamentals](../docs/01-quantization-fundamentals.md)
- [GGUF Quantization Formats](../docs/02-gguf-quantization-formats.md)
- [Performance Optimization Guide](../docs/03-performance-optimization.md)
- [Benchmarking Best Practices](../docs/05-benchmarking-testing.md)

## Author

Agent 3 (Code Examples Specialist)
Module 3 - Quantization & Optimization
Last Updated: 2025-11-18

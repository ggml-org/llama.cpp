# Lab 2: Format Comparison and Selection

**Module 3** | **Estimated Time**: 2-3 hours | **Difficulty**: Intermediate to Advanced

## Learning Objectives

By completing this lab, you will:
- Compare k-quant formats against legacy formats
- Understand the internal structure of quantization formats
- Make informed format selection decisions
- Build an automated format comparison pipeline
- Create production deployment recommendations

## Prerequisites

- Completed Lab 1
- Understanding of GGUF quantization formats
- Python programming skills
- llama.cpp tools installed

## Lab Overview

In this lab, you'll build a comprehensive format comparison system and use it to select the optimal quantization for specific deployment scenarios.

## Part 1: Legacy vs K-Quants Comparison (45 minutes)

### Step 1: Prepare Comparison Set

We'll compare legacy formats against their k-quant equivalents:

| Legacy | K-Quant | Nominal Bits |
|--------|---------|--------------|
| Q4_0   | Q4_K_M  | 4-bit        |
| Q4_1   | Q4_K_M  | 4-bit        |
| Q5_0   | Q5_K_M  | 5-bit        |
| Q5_1   | Q5_K_M  | 5-bit        |

### Step 2: Quantize Models

```bash
# Legacy formats
./llama-quantize base-model.gguf model-q4_0.gguf Q4_0
./llama-quantize base-model.gguf model-q4_1.gguf Q4_1
./llama-quantize base-model.gguf model-q5_0.gguf Q5_0
./llama-quantize base-model.gguf model-q5_1.gguf Q5_1

# K-quant formats
./llama-quantize base-model.gguf model-q4_k_s.gguf Q4_K_S
./llama-quantize base-model.gguf model-q4_k_m.gguf Q4_K_M
./llama-quantize base-model.gguf model-q5_k_s.gguf Q5_K_S
./llama-quantize base-model.gguf model-q5_k_m.gguf Q5_K_M
```

### Step 3: Size Comparison

```bash
# Compare sizes
ls -lh model-q*.gguf | awk '{print $9, $5}'
```

**Question 1**: Fill in the size comparison:

| Format | Size (GB) | Bits per Weight |
|--------|-----------|-----------------|
| Q4_0   | _____     | ~4.5            |
| Q4_1   | _____     | ~5.0            |
| Q4_K_S | _____     | ~4.3            |
| Q4_K_M | _____     | ~4.8            |
| Q5_0   | _____     | ~5.5            |
| Q5_1   | _____     | ~6.0            |
| Q5_K_S | _____     | ~5.5            |
| Q5_K_M | _____     | ~5.9            |

**Question 2**: Why do k-quants sometimes have slightly larger sizes than legacy formats at the same nominal bit-width?

**Answer**: _________________________________________________________________

### Step 4: Quality Comparison

Run perplexity tests:

```bash
#!/bin/bash
# compare_legacy_vs_kquant.sh

FORMATS=(q4_0 q4_1 q4_k_s q4_k_m q5_0 q5_1 q5_k_s q5_k_m)
TEST_FILE="wikitext-2-test.txt"

echo "Format,Perplexity,Time_seconds" > comparison_results.csv

for format in "${FORMATS[@]}"; do
    echo "Testing $format..."

    start=$(date +%s)
    output=$(./llama-perplexity \
        -m model-${format}.gguf \
        -f $TEST_FILE \
        --perplexity 2>&1)

    end=$(date +%s)
    elapsed=$((end - start))

    # Extract perplexity
    ppl=$(echo "$output" | grep "perplexity" | awk '{print $NF}')

    echo "$format,$ppl,$elapsed" >> comparison_results.csv
    echo "  Perplexity: $ppl (${elapsed}s)"
done
```

**Question 3**: Compare Q4_0 vs Q4_K_M:

- Q4_0 perplexity: _______
- Q4_K_M perplexity: _______
- Improvement: _______  %

**Question 4**: Is the perplexity improvement of k-quants worth the slightly larger size?

**Answer**: _________________________________________________________________

## Part 2: S/M/L Variant Analysis (30 minutes)

### Understanding Variants

K-quants come in S (Small), M (Medium), and L (Large) variants for some bit-widths.

### Step 1: Test All Variants

```bash
# Q3_K variants
./llama-quantize base-model.gguf model-q3_k_s.gguf Q3_K_S
./llama-quantize base-model.gguf model-q3_k_m.gguf Q3_K_M
./llama-quantize base-model.gguf model-q3_k_l.gguf Q3_K_L

# Q4_K variants (S and M only)
# Already done above

# Q5_K variants (S and M only)
# Already done above
```

### Step 2: Measure Quality Difference

**Question 5**: For Q3_K formats, fill in the comparison:

| Variant | Size (GB) | Perplexity | Use Case |
|---------|-----------|------------|----------|
| Q3_K_S  | _____     | _____      | ________ |
| Q3_K_M  | _____     | _____      | ________ |
| Q3_K_L  | _____     | _____      | ________ |

**Question 6**: Is the quality improvement from S → M → L progressive, or are there diminishing returns?

**Answer**: _________________________________________________________________

**Question 7**: At what point does Q3_K_L become less attractive than Q4_K_S?

**Answer**: _________________________________________________________________

## Part 3: Building an Automated Comparison Tool (45 minutes)

### Create a Format Comparison Script

```python
#!/usr/bin/env python3
"""
format_selector.py - Automated format selection tool
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

@dataclass
class FormatMetrics:
    format_name: str
    size_gb: float
    perplexity: float
    tokens_per_sec: float
    bits_per_weight: float
    quality_score: float  # Higher is better
    efficiency_score: float  # tokens/sec per GB

    def __post_init__(self):
        # Calculate derived metrics
        if self.size_gb > 0:
            self.efficiency_score = self.tokens_per_sec / self.size_gb
        else:
            self.efficiency_score = 0

class FormatSelector:
    """Select optimal quantization format based on requirements"""

    def __init__(self, base_model: str, llama_cpp_dir: str = "./"):
        self.base_model = Path(base_model)
        self.llama_cpp_dir = Path(llama_cpp_dir)
        self.metrics: List[FormatMetrics] = []

    def measure_format(self, format_name: str,
                      model_path: Path,
                      test_file: str) -> FormatMetrics:
        """Measure all metrics for a format"""

        # Size
        size_gb = model_path.stat().st_size / (1024**3)

        # Perplexity
        ppl = self._measure_perplexity(model_path, test_file)

        # Performance
        tps = self._measure_performance(model_path)

        # Estimate bits per weight (lookup table)
        bpw = self._get_bits_per_weight(format_name)

        # Quality score (inverse of perplexity)
        quality_score = 1.0 / ppl if ppl > 0 else 0

        return FormatMetrics(
            format_name=format_name,
            size_gb=size_gb,
            perplexity=ppl,
            tokens_per_sec=tps,
            bits_per_weight=bpw,
            quality_score=quality_score,
            efficiency_score=0  # Calculated in __post_init__
        )

    def _measure_perplexity(self, model_path: Path, test_file: str) -> float:
        """Measure perplexity"""
        # Implementation similar to previous labs
        # Return perplexity value
        pass

    def _measure_performance(self, model_path: Path) -> float:
        """Measure tokens/second"""
        # Implementation similar to previous labs
        # Return tokens per second
        pass

    def _get_bits_per_weight(self, format_name: str) -> float:
        """Get approximate bits per weight for format"""
        bpw_map = {
            'Q2_K': 2.5,
            'Q3_K_S': 3.4, 'Q3_K_M': 3.7, 'Q3_K_L': 4.0,
            'Q4_0': 4.5, 'Q4_1': 5.0,
            'Q4_K_S': 4.3, 'Q4_K_M': 4.8,
            'Q5_0': 5.5, 'Q5_1': 6.0,
            'Q5_K_S': 5.5, 'Q5_K_M': 5.9,
            'Q6_K': 6.6,
            'Q8_0': 8.5
        }
        return bpw_map.get(format_name.upper(), 0.0)

    def recommend_format(self, requirements: Dict) -> str:
        """Recommend format based on requirements

        Args:
            requirements: Dict with keys:
                - priority: 'quality', 'size', 'speed', 'balanced'
                - max_size_gb: Optional[float]
                - min_quality: Optional[float] (max acceptable perplexity)
                - min_speed: Optional[float] (min tokens/sec)
        """

        priority = requirements.get('priority', 'balanced')
        max_size = requirements.get('max_size_gb', float('inf'))
        min_quality = requirements.get('min_quality', float('inf'))
        min_speed = requirements.get('min_speed', 0)

        # Filter candidates
        candidates = [
            m for m in self.metrics
            if m.size_gb <= max_size
            and m.perplexity <= min_quality
            and m.tokens_per_sec >= min_speed
        ]

        if not candidates:
            return None

        # Select based on priority
        if priority == 'quality':
            return min(candidates, key=lambda m: m.perplexity).format_name
        elif priority == 'size':
            return min(candidates, key=lambda m: m.size_gb).format_name
        elif priority == 'speed':
            return max(candidates, key=lambda m: m.tokens_per_sec).format_name
        else:  # balanced
            # Multi-objective score
            return max(candidates, key=lambda m: (
                m.quality_score * 0.4 +
                (1.0 / m.size_gb) * 0.3 +
                (m.tokens_per_sec / 30.0) * 0.3  # Normalize to ~30 tps
            )).format_name

# Usage example
selector = FormatSelector("base-model.gguf")

# Measure formats
formats = ['Q4_0', 'Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q8_0']
for fmt in formats:
    model_path = Path(f"model-{fmt.lower()}.gguf")
    if model_path.exists():
        metrics = selector.measure_format(fmt, model_path, "test.txt")
        selector.metrics.append(metrics)

# Get recommendations
print(selector.recommend_format({'priority': 'balanced'}))
print(selector.recommend_format({'priority': 'quality', 'max_size_gb': 5.0}))
print(selector.recommend_format({'priority': 'speed', 'min_quality': 6.0}))
```

**Task**: Complete the implementation of the `FormatSelector` class.

## Part 4: Real-World Scenarios (45 minutes)

### Scenario 1: Mobile Deployment

**Requirements**:
- Maximum size: 4GB
- Acceptable perplexity increase: <5%
- Priority: Size first, then speed

**Question 8**: Using your measurements, which format would you select?

**Answer**: _________________________________________________________________

**Justification**: _________________________________________________________________

### Scenario 2: Cloud API Service

**Requirements**:
- Priority: Throughput (can batch requests)
- Acceptable perplexity increase: <2%
- Will run on 32GB RAM instance

**Question 9**: Which format would you select?

**Answer**: _________________________________________________________________

**Justification**: _________________________________________________________________

### Scenario 3: Edge Device (Raspberry Pi 4)

**Requirements**:
- Maximum size: 3GB
- Must fit in 4GB RAM (including OS)
- Quality: Best possible within size constraint

**Question 10**: Which format would you select?

**Answer**: _________________________________________________________________

**Justification**: _________________________________________________________________

### Scenario 4: Code Generation Service

**Requirements**:
- Use case: GitHub Copilot-like feature
- Quality: Critical (code must compile)
- Speed: Important (need fast suggestions)
- Size: Not a major constraint

**Question 11**: Which format would you select?

**Answer**: _________________________________________________________________

**Justification**: _________________________________________________________________

## Part 5: Production Decision Matrix (30 minutes)

### Create a Decision Framework

Build a decision tree for format selection:

```
1. What's your memory budget?
   ├─ Unlimited → Q8_0
   ├─ Comfortable → Q5_K_M or Q6_K
   ├─ Moderate → Q4_K_M
   └─ Tight → Q4_K_S or Q3_K_M

2. What's your quality requirement?
   ├─ Maximum → Q8_0 or Q6_K
   ├─ High → Q5_K_M
   ├─ Balanced → Q4_K_M
   └─ Acceptable degradation → Q4_K_S, Q3_K_M

3. What's your hardware?
   ├─ Modern CPU (AVX2/AVX-512) → K-quants work great
   ├─ Older CPU → Legacy formats may be faster
   ├─ ARM (Apple Silicon) → K-quants optimized
   └─ GPU → All formats work, prefer higher for quality
```

**Question 12**: Create your own decision matrix for your organization:

**Answer**: _________________________________________________________________

## Challenge Exercise

### Multi-Format Ensemble

Some applications run multiple formats for different use cases:

```python
class MultiFormatDeployment:
    """Deploy different formats for different use cases"""

    def __init__(self):
        self.models = {
            'fast': 'model-q4_k_s.gguf',      # Quick responses
            'balanced': 'model-q4_k_m.gguf',  # Default
            'quality': 'model-q5_k_m.gguf',   # Important requests
            'reference': 'model-q8_0.gguf'    # Quality check
        }

    def select_model(self, request):
        """Select appropriate model for request"""
        if request.priority == 'fast':
            return self.models['fast']
        elif request.requires_quality:
            return self.models['quality']
        else:
            return self.models['balanced']
```

**Question 13**: Design a multi-format deployment strategy for:
- A customer service chatbot
- A code generation tool
- A content writing assistant

**Answer**: _________________________________________________________________

## Lab Deliverables

Submit:
1. Completed comparison tables
2. Format selector implementation
3. Scenario analysis and recommendations
4. Decision matrix for your use case
5. Multi-format deployment strategy

## Key Takeaways

Write a one-paragraph summary of what you learned about format selection:

**Summary**: _________________________________________________________________

## Further Exploration

- Compare IQ (importance quantization) formats
- Test on different model sizes (7B vs 13B vs 70B)
- Measure format impact on specific tasks (code, math, creative)
- Build automated format testing pipeline
- Create A/B testing framework for production

## Resources

- [GGUF Quantization Formats Guide](../docs/02-gguf-quantization-formats.md)
- [Format Converter Tool](../code/format_converter.py)
- [llama.cpp Quantization Documentation](https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize)

---

**Completion Criteria**:
- [ ] All format comparisons completed
- [ ] Legacy vs K-quant analysis done
- [ ] Format selector implemented
- [ ] All scenarios analyzed
- [ ] Decision matrix created
- [ ] Deployment strategy designed

**Estimated Time**: 2-3 hours
**Difficulty**: Intermediate to Advanced
**Author**: Agent 4 (Lab Designer)
**Module**: 3 - Quantization & Optimization

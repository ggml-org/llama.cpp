# Model Info Tool - Usage Examples

Quick examples to get you started with the model inspection tool.

## Quick Start

```bash
# No installation needed! Just run it
cd model-info-tool
python3 -m model_info_tool.cli info /path/to/model.gguf

# Or install and use anywhere
pip install -e .
model-info info /path/to/model.gguf
```

## Basic Examples

### 1. Quick Model Check

```bash
$ model-info info llama-2-7b-chat.Q4_K_M.gguf

============================================================
  MODEL INFORMATION
============================================================

Name: Llama 2 7B Chat
Architecture: llama
File: llama-2-7b-chat.Q4_K_M.gguf
Path: /models/llama-2-7b-chat.Q4_K_M.gguf
Size: 3.83 GB (3897.45 MB)
Parameters: 6.74B (6,738,415,616)
Tensors: 291
```

### 2. Summary View

```bash
$ model-info info model.gguf --summary

============================================================
MODEL SUMMARY
============================================================
Name: Llama 2 7B Chat
Architecture: llama
Parameters: 6.74B
File Size: 3.83 GB (3897.45 MB)
Quantization: Q4_K_M
Tensor Count: 291

ARCHITECTURE DETAILS
------------------------------------------------------------
embedding_length: 4096
block_count: 32
feed_forward_length: 11008
attention.head_count: 32
attention.head_count_kv: 32

MEMORY REQUIREMENTS (ESTIMATED)
------------------------------------------------------------
Model Size: 3.83 GB
CPU-only RAM: ~4.59 GB
GPU-full VRAM: ~4.21 GB

============================================================
```

### 3. Detailed Information

```bash
# Show everything
$ model-info info model.gguf -v

# Or show specific sections
$ model-info info model.gguf --architecture
$ model-info info model.gguf --quantization
$ model-info info model.gguf --memory
$ model-info info model.gguf --tokenizer
```

## Comparison Examples

### Compare Quantization Levels

```bash
$ model-info compare \
  llama-2-7b.Q2_K.gguf \
  llama-2-7b.Q4_K_M.gguf \
  llama-2-7b.Q6_K.gguf \
  llama-2-7b.Q8_0.gguf

============================================================
  MODEL COMPARISON
============================================================

Model                          Architecture    Parameters   Size (GB)  Quantization
----------------------------------------------------------------------------------
llama-2-7b.Q2_K.gguf          llama           6.74B        2.51       Q2_K
llama-2-7b.Q4_K_M.gguf        llama           6.74B        3.83       Q4_K_M
llama-2-7b.Q6_K.gguf          llama           6.74B        5.15       Q6_K
llama-2-7b.Q8_0.gguf          llama           6.74B        7.17       Q8_0

Memory Requirements Comparison
------------------------------------------------------------
Model                          CPU RAM (GB)    GPU VRAM (GB)
------------------------------------------------------------
llama-2-7b.Q2_K.gguf          3.012          2.761
llama-2-7b.Q4_K_M.gguf        4.596          4.213
llama-2-7b.Q6_K.gguf          6.180          5.665
llama-2-7b.Q8_0.gguf          8.604          7.887
```

### Compare Different Models

```bash
$ model-info compare \
  llama-2-7b-chat.Q4_K_M.gguf \
  llama-2-13b-chat.Q4_K_M.gguf \
  mistral-7b-v0.1.Q4_K_M.gguf \
  codellama-7b.Q4_K_M.gguf

Model                          Architecture    Parameters   Size (GB)  Quantization
----------------------------------------------------------------------------------
llama-2-7b-chat.Q4_K_M.gguf   llama           6.74B        3.83       Q4_K_M
llama-2-13b-chat.Q4_K_M.gguf  llama           13.02B       7.37       Q4_K_M
mistral-7b-v0.1.Q4_K_M.gguf   llama           7.24B        4.78       Q4_K_M
codellama-7b.Q4_K_M.gguf      llama           6.74B        3.83       Q4_K_M
```

## Export Examples

### Export to JSON

```bash
# Print to terminal
$ model-info export model.gguf -f json

# Save to file
$ model-info export model.gguf -f json -o model_info.json

# Include tensor details (large output!)
$ model-info export model.gguf -f json --include-tensors -o full_info.json

# Custom indentation
$ model-info export model.gguf -f json -i 4 -o model_info.json
```

Example JSON output:

```json
{
  "export_info": {
    "timestamp": "2024-01-15T10:30:00",
    "tool_version": "1.0.0"
  },
  "basic_info": {
    "name": "Llama 2 7B Chat",
    "architecture": "llama",
    "file_path": "/models/llama-2-7b-chat.Q4_K_M.gguf",
    "file_name": "llama-2-7b-chat.Q4_K_M.gguf",
    "file_size_mb": 3897.45,
    "file_size_gb": 3.83,
    "parameter_count": 6738415616,
    "parameter_count_formatted": "6.74B",
    "tensor_count": 291
  },
  "quantization": {
    "predominant_type": "Q4_K_M",
    "tensor_types": {
      "Q4_K": 225,
      "Q6_K": 64,
      "F32": 2
    }
  }
}
```

### Export to Markdown

```bash
# Print to terminal
$ model-info export model.gguf -f markdown

# Save to file
$ model-info export model.gguf -f markdown -o MODEL_INFO.md
```

Great for documentation!

### Export to CSV

```bash
# Print to terminal
$ model-info export model.gguf -f csv

# Save to file
$ model-info export model.gguf -f csv -o model_info.csv
```

Output:

```csv
Name,Architecture,Parameters,File_Size_GB,Quantization,Tensor_Count,RAM_CPU_GB,VRAM_GPU_GB
Llama 2 7B Chat,llama,6738415616,3.83,Q4_K_M,291,4.596,4.213
```

Perfect for spreadsheets!

## Metadata Examples

### List All Metadata

```bash
$ model-info metadata model.gguf

============================================================
  ALL METADATA
============================================================
general.architecture: llama
general.file_type: 15
general.name: Llama 2 7B Chat
general.quantization_version: 2
llama.attention.head_count: 32
llama.attention.head_count_kv: 32
llama.attention.layer_norm_rms_epsilon: 1e-05
llama.block_count: 32
llama.context_length: 4096
llama.embedding_length: 4096
llama.feed_forward_length: 11008
llama.rope.dimension_count: 128
llama.rope.freq_base: 10000.0
tokenizer.ggml.bos_token_id: 1
tokenizer.ggml.eos_token_id: 2
tokenizer.ggml.model: llama
tokenizer.ggml.vocab_size: 32000
```

## Practical Use Cases

### 1. Choosing the Right Model for Your Hardware

```bash
# Check memory requirements
$ model-info info model.gguf --memory

Memory Requirements (Estimated)
------------------------------------------------------------
  Model Size: 3.83 GB
  CPU-only RAM: ~4.59 GB      # Need at least 8GB RAM
  GPU-full VRAM: ~4.21 GB     # Need at least 6GB VRAM

# Compare different sizes
$ model-info compare 7b-model.gguf 13b-model.gguf 30b-model.gguf
```

### 2. Documenting Your Model Collection

```bash
# Create documentation for all models
for model in models/*.gguf; do
  echo "Processing $model..."
  model-info export "$model" -f markdown -o "docs/$(basename "$model" .gguf).md"
done
```

### 3. Building a Model Database

```bash
# Export all models to JSON
mkdir -p model_catalog

for model in models/*.gguf; do
  name=$(basename "$model" .gguf)
  model-info export "$model" -f json -o "model_catalog/${name}.json"
done

# Combine into single database
jq -s '.' model_catalog/*.json > all_models.json
```

### 4. Pre-Download Verification

```bash
# Before downloading a large model, check if similar model works
$ model-info info similar-model.gguf --memory

# If it fits, download the model you want
# If not, look for smaller quantization
```

### 5. Quality vs Size Trade-off Analysis

```bash
# Compare quantizations to choose best trade-off
$ model-info compare \
  model.Q4_K_M.gguf \  # 3.8 GB
  model.Q5_K_M.gguf \  # 4.6 GB
  model.Q6_K.gguf      # 5.2 GB

# Check size difference
# Q5_K_M is only 0.8GB larger than Q4_K_M
# Might be worth it for better quality!
```

## Scripting Examples

### Python Script: Batch Analysis

```python
#!/usr/bin/env python3
from pathlib import Path
from model_info_tool import ModelAnalyzer

models_dir = Path("models")

for model_file in models_dir.glob("*.gguf"):
    print(f"\nAnalyzing {model_file.name}...")

    analyzer = ModelAnalyzer(str(model_file))
    basic = analyzer.get_basic_info()
    memory = analyzer.get_memory_requirements()

    print(f"  Size: {basic['file_size_gb']} GB")
    print(f"  Parameters: {basic['parameter_count_formatted']}")
    print(f"  RAM needed: {memory['estimated_ram_cpu_only_gb']} GB")
```

### Shell Script: Model Inventory

```bash
#!/bin/bash

echo "Model Inventory Report"
echo "======================"
echo ""

for model in models/*.gguf; do
  if [ -f "$model" ]; then
    echo "Model: $(basename "$model")"
    model-info info "$model" --summary | grep -E "(Parameters|Size|Quantization)"
    echo "---"
  fi
done
```

### Find Models That Fit Your Hardware

```bash
#!/bin/bash

MAX_RAM_GB=8

echo "Models that fit in ${MAX_RAM_GB}GB RAM:"
echo ""

for model in models/*.gguf; do
  # Export to JSON and extract RAM requirement
  ram=$(model-info export "$model" -f json | \
    jq -r '.memory_requirements.estimated_ram_cpu_only_gb')

  if (( $(echo "$ram < $MAX_RAM_GB" | bc -l) )); then
    name=$(basename "$model")
    echo "$name - RAM: ${ram}GB"
  fi
done
```

## Advanced Examples

### Compare and Export Results

```bash
# Compare models and save comparison to JSON
$ python3 << 'EOF'
from model_info_tool.exporter import ModelExporter

models = [
    "models/llama-2-7b.Q4_K_M.gguf",
    "models/llama-2-13b.Q4_K_M.gguf",
    "models/mistral-7b.Q5_K_M.gguf"
]

ModelExporter.compare_models_to_json(models, "comparison.json")
EOF
```

### Custom Analysis Script

```python
#!/usr/bin/env python3
from model_info_tool import ModelAnalyzer

analyzer = ModelAnalyzer("model.gguf")

# Get all information
basic = analyzer.get_basic_info()
arch = analyzer.get_architecture_details()
quant = analyzer.get_quantization_info()
memory = analyzer.get_memory_requirements()

# Custom analysis
params_billions = basic['parameter_count'] / 1_000_000_000
size_gb = basic['file_size_gb']
bits_per_param = (size_gb * 1024 * 1024 * 1024 * 8) / basic['parameter_count']

print(f"Model: {basic['name']}")
print(f"Parameters: {params_billions:.2f}B")
print(f"Average bits per parameter: {bits_per_param:.2f}")
print(f"Compression ratio: {32/bits_per_param:.1f}x from FP32")

# Efficiency score (subjective!)
if bits_per_param < 3:
    print("Efficiency: Excellent (very compressed)")
elif bits_per_param < 5:
    print("Efficiency: Good (balanced)")
elif bits_per_param < 7:
    print("Efficiency: Moderate (higher quality)")
else:
    print("Efficiency: Low (minimal compression)")
```

### Generate HTML Report

```bash
# Export to markdown
model-info export model.gguf -f markdown -o report.md

# Convert to HTML (requires pandoc)
pandoc report.md -o report.html --standalone --css=style.css
```

## Integration Examples

### With Model Download Script

```bash
#!/bin/bash

MODEL_URL="https://huggingface.co/..."
MODEL_NAME="model.gguf"

# Download model
echo "Downloading $MODEL_NAME..."
wget -q "$MODEL_URL" -O "$MODEL_NAME"

# Verify and inspect
if [ -f "$MODEL_NAME" ]; then
  echo "Download complete. Inspecting model..."
  model-info info "$MODEL_NAME" --summary

  # Check if it fits in RAM
  ram_needed=$(model-info export "$MODEL_NAME" -f json | \
    jq -r '.memory_requirements.estimated_ram_cpu_only_gb')
  echo "RAM needed: ${ram_needed}GB"
fi
```

### With llama.cpp Testing

```bash
#!/bin/bash

MODEL="$1"

echo "Model Information:"
model-info info "$MODEL" --memory

echo ""
echo "Running test inference..."
./llama-cli -m "$MODEL" -p "Hello" -n 10 --verbose
```

## Tips and Tricks

### 1. Quick Memory Check

```bash
alias check-model='model-info info --memory'
check-model model.gguf
```

### 2. Compare All Quantizations

```bash
model-info compare model.Q*_*.gguf | less
```

### 3. Find Largest Models

```bash
for m in models/*.gguf; do
  size=$(stat -f%z "$m" 2>/dev/null || stat -c%s "$m")
  echo "$size $m"
done | sort -rn | head -5
```

### 4. Export for Spreadsheet

```bash
# Create CSV with all models
echo "Name,Architecture,Parameters,Size_GB,Quantization" > models.csv
for m in models/*.gguf; do
  model-info export "$m" -f csv | tail -n1 >> models.csv
done
```

### 5. JSON Query with jq

```bash
# Get specific info with jq
model-info export model.gguf -f json | jq '.basic_info.parameter_count'
model-info export model.gguf -f json | jq '.quantization.predominant_type'
model-info export model.gguf -f json | jq '.memory_requirements.estimated_ram_cpu_only_gb'
```

## Common Workflows

### Workflow 1: Model Selection

```bash
# 1. See what's available
ls -lh models/

# 2. Compare options
model-info compare models/llama*.gguf

# 3. Check memory for top choice
model-info info models/llama-2-7b.Q4_K_M.gguf --memory

# 4. Export info for reference
model-info export models/llama-2-7b.Q4_K_M.gguf -f markdown -o CURRENT_MODEL.md
```

### Workflow 2: Model Documentation

```bash
# 1. Inspect model
model-info info model.gguf -v > model_analysis.txt

# 2. Export structured data
model-info export model.gguf -f json -o model_data.json

# 3. Create user-friendly doc
model-info export model.gguf -f markdown -o README_MODEL.md

# 4. Add to project
git add model_data.json README_MODEL.md
```

### Workflow 3: Batch Processing

```bash
# Process all models in directory
for model in models/*.gguf; do
  name=$(basename "$model" .gguf)
  echo "Processing $name..."

  # Summary
  model-info info "$model" -s > "reports/${name}_summary.txt"

  # Full JSON
  model-info export "$model" -f json -o "data/${name}.json"
done

echo "All models processed!"
```

# Lab 7.4: Model Conversion Pipeline

**Estimated Time**: 90 minutes
**Difficulty**: Intermediate to Advanced
**Prerequisites**: Module 7.5 documentation complete

## Learning Objectives

- Convert HuggingFace models to GGUF format
- Quantize models to different precision levels
- Build automated conversion pipelines
- Validate converted models
- Deploy converted models

---

## Part 1: Basic Conversion (30 minutes)

### Setup

```bash
# Clone llama.cpp if not already
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install dependencies
pip install -r requirements.txt

# Build quantize tool
make quantize
```

### Task 1: Download and Convert a Model

**Download TinyLlama** (1.1B parameters, good for testing):

```bash
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./TinyLlama-1.1B-Chat
```

**Convert to GGUF**:

```bash
python convert_hf_to_gguf.py \
    ./TinyLlama-1.1B-Chat \
    --outfile TinyLlama-1.1B-Chat-f16.gguf \
    --outtype f16
```

**Record**:
- Conversion time
- Output file size
- Any warnings or errors

**Inspect the GGUF file**:
```bash
# List metadata
python gguf-py/scripts/gguf-dump.py TinyLlama-1.1B-Chat-f16.gguf | head -50

# Count tensors
python gguf-py/scripts/gguf-dump.py TinyLlama-1.1B-Chat-f16.gguf | grep -c "tensor"
```

---

## Part 2: Quantization (30 minutes)

### Task 2: Quantize to Multiple Formats

**Quantize to different precision levels**:

```bash
# Q4_K_M (4-bit, medium)
./quantize TinyLlama-1.1B-Chat-f16.gguf TinyLlama-1.1B-Chat-Q4_K_M.gguf q4_k_m

# Q5_K_M (5-bit, better quality)
./quantize TinyLlama-1.1B-Chat-f16.gguf TinyLlama-1.1B-Chat-Q5_K_M.gguf q5_k_m

# Q8_0 (8-bit, minimal loss)
./quantize TinyLlama-1.1B-Chat-f16.gguf TinyLlama-1.1B-Chat-Q8_0.gguf q8_0
```

**Create comparison table**:

```bash
ls -lh TinyLlama-*.gguf > sizes.txt
```

| Format | Size (MB) | Reduction | Notes |
|--------|-----------|-----------|-------|
| F16 | | Baseline | |
| Q8_0 | | | |
| Q5_K_M | | | |
| Q4_K_M | | | |

### Task 3: Test Each Quantization

**Create test script** (`test_quant.sh`):

```bash
#!/bin/bash

PROMPT="Once upon a time"
MODELS=(
    "TinyLlama-1.1B-Chat-f16.gguf"
    "TinyLlama-1.1B-Chat-Q8_0.gguf"
    "TinyLlama-1.1B-Chat-Q5_K_M.gguf"
    "TinyLlama-1.1B-Chat-Q4_K_M.gguf"
)

for MODEL in "${MODELS[@]}"; do
    echo "Testing: $MODEL"
    time ./main -m "$MODEL" -p "$PROMPT" -n 50 --temp 0.7 2>&1 | tail -20
    echo "---"
done
```

Run and compare:
- Generation quality
- Inference speed
- Memory usage

---

## Part 3: Automated Pipeline (30 minutes)

### Task 4: Build Conversion Pipeline

Use the provided `convert_model_helper.py`:

```bash
python convert_model_helper.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output ./converted \
    --llamacpp-dir . \
    --quant-types q4_k_m q5_k_m q8_0 \
    --keep-f16
```

**Examine the manifest**:
```bash
cat converted/TinyLlama-1.1B-Chat-v1.0_manifest.json
```

### Task 5: Batch Conversion

**Create batch conversion script** (`batch_convert.py`):

```python
import json
from pathlib import Path
from convert_model_helper import ModelConverter

# Models to convert
models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",  # 2.7B
    # Add more as needed
]

converter = ModelConverter("./llama.cpp")

results = []
for model_id in models:
    print(f"\n{'='*80}")
    print(f"Converting: {model_id}")
    print('='*80)

    try:
        result = converter.full_conversion_pipeline(
            model_id=model_id,
            output_dir="./converted",
            quant_types=["q4_k_m", "q5_k_m"],
            keep_f16=False
        )
        results.append({
            "model": model_id,
            "status": "success",
            "files": result["files"]
        })
    except Exception as e:
        print(f"Error: {e}")
        results.append({
            "model": model_id,
            "status": "failed",
            "error": str(e)
        })

# Save summary
with open("conversion_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("BATCH CONVERSION COMPLETE")
print("="*80)
for r in results:
    status = "✓" if r["status"] == "success" else "✗"
    print(f"{status} {r['model']}")
```

---

## Part 4: Validation (20 minutes)

### Task 6: Compare with Reference

**Create validation script** (`validate_conversion.py`):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
import numpy as np

def compare_outputs(hf_model_path, gguf_model_path, prompt="Hello, world!"):
    """Compare HuggingFace and GGUF model outputs"""

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

    # Load GGUF model
    print("Loading GGUF model...")
    gguf_model = Llama(model_path=gguf_model_path, verbose=False)

    # Generate with HuggingFace
    print("\nHuggingFace generation:")
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        hf_output = hf_model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,  # Low temp for reproducibility
            do_sample=True
        )
    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
    print(hf_text)

    # Generate with GGUF
    print("\nGGUF generation:")
    gguf_output = gguf_model.create_completion(
        prompt,
        max_tokens=50,
        temperature=0.1,
        seed=42
    )
    gguf_text = gguf_output['choices'][0]['text']
    print(prompt + gguf_text)

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Match: {hf_text == (prompt + gguf_text)}")

    # Check token overlap
    hf_tokens = hf_text.split()
    gguf_tokens = (prompt + gguf_text).split()
    overlap = len(set(hf_tokens) & set(gguf_tokens))
    print(f"Token overlap: {overlap}/{len(hf_tokens)} ({100*overlap/len(hf_tokens):.1f}%)")

# Run
compare_outputs(
    "./TinyLlama-1.1B-Chat",
    "TinyLlama-1.1B-Chat-Q8_0.gguf"
)
```

**Expected**: High token overlap (>80%), similar outputs

### Task 7: Perplexity Testing

**Measure perplexity** to quantify quality:

```bash
# Download test data
wget https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip

# Run perplexity test
./perplexity -m TinyLlama-1.1B-Chat-f16.gguf -f wikitext-2-raw/wiki.test.raw

# Test all quantizations
for MODEL in TinyLlama-*.gguf; do
    echo "Testing: $MODEL"
    ./perplexity -m "$MODEL" -f wikitext-2-raw/wiki.test.raw | tail -5
done
```

**Record perplexity** for each quantization:

| Format | Perplexity | Δ from F16 |
|--------|------------|------------|
| F16 | | 0.00 |
| Q8_0 | | |
| Q5_K_M | | |
| Q4_K_M | | |

---

## Part 5: Production Deployment (20 minutes)

### Task 8: Model Registry

**Create model registry** (`model_registry.json`):

```json
{
  "models": [
    {
      "id": "tinyllama-1.1b-chat",
      "name": "TinyLlama 1.1B Chat",
      "source": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "variants": [
        {
          "format": "q4_k_m",
          "file": "TinyLlama-1.1B-Chat-Q4_K_M.gguf",
          "size_mb": 669,
          "use_case": "Fast inference, mobile"
        },
        {
          "format": "q5_k_m",
          "file": "TinyLlama-1.1B-Chat-Q5_K_M.gguf",
          "size_mb": 794,
          "use_case": "Balanced quality/speed"
        },
        {
          "format": "q8_0",
          "file": "TinyLlama-1.1B-Chat-Q8_0.gguf",
          "size_mb": 1211,
          "use_case": "High quality, server"
        }
      ],
      "metadata": {
        "architecture": "llama",
        "parameters": "1.1B",
        "context_length": 2048,
        "license": "Apache 2.0"
      }
    }
  ]
}
```

### Task 9: Model Serving

**Create model server** (`model_server.py`):

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import json

app = FastAPI()

# Load registry
with open("model_registry.json") as f:
    registry = json.load(f)

# Load models (lazy loading in production)
models = {}

def get_model(model_id, variant="q4_k_m"):
    """Load model on-demand"""
    key = f"{model_id}:{variant}"

    if key not in models:
        # Find model in registry
        for m in registry["models"]:
            if m["id"] == model_id:
                for v in m["variants"]:
                    if v["format"] == variant:
                        models[key] = Llama(
                            model_path=v["file"],
                            n_ctx=2048,
                            n_gpu_layers=35
                        )
                        return models[key]

        raise HTTPException(404, "Model not found")

    return models[key]

class GenerateRequest(BaseModel):
    model: str
    variant: str = "q4_k_m"
    prompt: str
    max_tokens: int = 512

@app.post("/generate")
async def generate(req: GenerateRequest):
    model = get_model(req.model, req.variant)

    response = model.create_completion(
        req.prompt,
        max_tokens=req.max_tokens
    )

    return response

@app.get("/models")
async def list_models():
    return registry

# Run with: uvicorn model_server:app --reload
```

Test:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat",
    "variant": "q4_k_m",
    "prompt": "Explain quantum computing in one sentence.",
    "max_tokens": 100
  }'
```

---

## Deliverables

1. **Conversion Report**:
   - Models converted
   - File sizes
   - Conversion times
   - Issues encountered

2. **Quantization Analysis**:
   - Size comparison table
   - Perplexity measurements
   - Quality assessment

3. **Validation Results**:
   - Output comparisons
   - Perplexity scores
   - Accuracy metrics

4. **Production Assets**:
   - Model registry
   - Automated conversion scripts
   - Model server implementation
   - Deployment documentation

---

## Bonus Challenges

### Challenge 1: Vision Model Conversion

Convert a vision-language model:
- LLaVA
- MiniCPM-V
- Separate vision encoder and LLM

### Challenge 2: LoRA Conversion

Convert and merge LoRA adapters:
```bash
python convert_lora_to_gguf.py \
    --base base_model.gguf \
    --lora lora_adapter/ \
    --output merged_model.gguf
```

### Challenge 3: Optimization Pipeline

Build a pipeline that:
- Converts model
- Tests multiple quantizations
- Benchmarks performance
- Selects best variant for use case
- Deploys automatically

---

## Reflection

1. What's the optimal quantization for your use case?
2. How do you validate conversion quality?
3. What are the trade-offs between formats?
4. How would you manage a large model fleet?

---

**Lab Complete!** You've mastered the complete model conversion pipeline for llama.cpp!

# Model Conversion: HuggingFace to GGUF

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Model Formats](#understanding-model-formats)
3. [Conversion Pipeline Overview](#conversion-pipeline-overview)
4. [Using convert_hf_to_gguf.py](#using-convert_hf_to_ggufpy)
5. [Advanced Conversion Techniques](#advanced-conversion-techniques)
6. [Post-Conversion Quantization](#post-conversion-quantization)
7. [Troubleshooting](#troubleshooting)
8. [Automation and Best Practices](#automation-and-best-practices)

---

## Introduction

Converting models from HuggingFace format to GGUF enables running them in llama.cpp. This process involves:
1. Loading PyTorch/Safetensors weights
2. Mapping tensor names to GGUF format
3. Extracting and writing metadata
4. Optionally quantizing weights

### Why Convert Models?

**Benefits of GGUF format**:
- **Efficient Inference**: Optimized for CPU/GPU inference
- **Quantization Support**: Easy to quantize after conversion
- **Single File**: Everything in one portable file
- **Memory Mapping**: Fast loading with mmap
- **Cross-Platform**: Works on any platform llama.cpp supports

---

## Understanding Model Formats

### HuggingFace Model Structure

```
model-name/
├── config.json              # Model architecture configuration
├── tokenizer.json           # Tokenizer vocabulary
├── tokenizer_config.json    # Tokenizer settings
├── special_tokens_map.json  # Special token definitions
└── pytorch_model.bin        # or .safetensors files
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── model-00003-of-00003.safetensors
```

**config.json** example:
```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "vocab_size": 32000,
  "rope_theta": 10000.0,
  "max_position_embeddings": 2048,
  "rms_norm_eps": 1e-5
}
```

### GGUF Structure

```
model.gguf
├── Header
│   ├── Magic number (GGUF)
│   ├── Version
│   └── Metadata count
├── Metadata (key-value pairs)
│   ├── general.architecture = "llama"
│   ├── llama.context_length = 2048
│   ├── llama.embedding_length = 4096
│   └── ... (all hyperparameters)
├── Tensor Info
│   ├── Tensor count
│   ├── Tensor 1: name, dimensions, type, offset
│   ├── Tensor 2: name, dimensions, type, offset
│   └── ...
└── Tensor Data
    ├── Aligned binary data for all tensors
    └── ...
```

### Format Comparison

| Aspect | HuggingFace | GGUF |
|--------|-------------|------|
| Files | Multiple (config + weights) | Single file |
| Loading | Load to RAM first | Memory-mapped |
| Metadata | JSON files | Binary KV pairs |
| Tensors | PyTorch format | Custom format |
| Quantization | Limited | Native support |
| Platform | Python-centric | Cross-platform |

---

## Conversion Pipeline Overview

### High-Level Process

```
┌────────────────────┐
│ HuggingFace Model  │
│ (PyTorch/Safetens) │
└──────────┬─────────┘
           ↓
┌────────────────────┐
│  Load Config & Weights │
└──────────┬─────────┘
           ↓
┌────────────────────┐
│  Map Tensor Names  │  ← Architecture-specific
└──────────┬─────────┘
           ↓
┌────────────────────┐
│ Extract Metadata   │
└──────────┬─────────┘
           ↓
┌────────────────────┐
│ Write GGUF File    │
└──────────┬─────────┘
           ↓
┌────────────────────┐
│   GGUF Model (F16) │
└──────────┬─────────┘
           ↓
┌────────────────────┐
│ Quantize (Optional)│
└──────────┬─────────┘
           ↓
┌────────────────────┐
│  GGUF Model (Q4/Q5)│
└────────────────────┘
```

### Conversion Scripts in llama.cpp

```bash
llama.cpp/
├── convert_hf_to_gguf.py          # Main conversion script
├── convert_hf_to_gguf_update.py   # Update existing GGUF
├── convert_lora_to_gguf.py        # Convert LoRA adapters
└── gguf-py/                       # GGUF library
    ├── gguf/
    │   ├── gguf.py                # GGUF writer
    │   ├── constants.py           # Type definitions
    │   └── vocab.py               # Tokenizer conversion
```

---

## Using convert_hf_to_gguf.py

### Basic Conversion

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install dependencies
pip install -r requirements.txt

# Convert model
python convert_hf_to_gguf.py \
  /path/to/huggingface/model \
  --outfile model.gguf \
  --outtype f16
```

### Command-Line Options

```bash
python convert_hf_to_gguf.py --help

Options:
  model_dir              Path to HuggingFace model directory
  --outfile, -o          Output GGUF file path (default: based on model name)
  --outtype              Output tensor type: f32, f16, q8_0 (default: f16)
  --vocab-only           Convert only tokenizer, not weights
  --awq-path             Path to AWQ quantized model
  --metadata             Override metadata values (key=value)
  --pad-vocab            Pad vocabulary to multiple of 32
  --concurrency          Number of parallel workers (default: 4)
```

### Example: Convert LLaMA-2-7B

```bash
# Download from HuggingFace
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./Llama-2-7b-hf

# Convert to GGUF (F16)
python convert_hf_to_gguf.py ./Llama-2-7b-hf \
  --outfile llama-2-7b.gguf \
  --outtype f16

# Output:
# Loading model: Llama-2-7b-hf
# - Architecture: llama
# - Layers: 32
# - Embedding: 4096
# - Vocabulary: 32000
# Converting tensors...
# [████████████████████████████] 291/291 tensors
# Writing GGUF file: llama-2-7b.gguf
# Done! File size: 13.5 GB
```

### Example: Convert Mistral-7B

```bash
# Download model
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./Mistral-7B

# Convert
python convert_hf_to_gguf.py ./Mistral-7B \
  --outfile mistral-7b-v0.1.gguf \
  --outtype f16 \
  --metadata "general.name=Mistral-7B-v0.1"

# Mistral-specific metadata automatically detected:
# - Sliding window size
# - RoPE parameters
# - GQA configuration
```

### Example: Convert Custom Model

```bash
# For models with custom architectures
python convert_hf_to_gguf.py ./my-custom-model \
  --outfile custom.gguf \
  --outtype f16 \
  --metadata "general.architecture=myarch" \
  --metadata "myarch.custom_param=42"
```

---

## Advanced Conversion Techniques

### Tensor Name Mapping

The conversion script maps HuggingFace tensor names to GGUF names:

```python
# In convert_hf_to_gguf.py

# Example mapping for LLaMA
TENSOR_MAP = {
    # HuggingFace name → GGUF name
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",

    # Per-layer tensors
    "model.layers.{}.self_attn.q_proj.weight": "blk.{}.attn_q.weight",
    "model.layers.{}.self_attn.k_proj.weight": "blk.{}.attn_k.weight",
    "model.layers.{}.self_attn.v_proj.weight": "blk.{}.attn_v.weight",
    "model.layers.{}.self_attn.o_proj.weight": "blk.{}.attn_output.weight",

    "model.layers.{}.mlp.gate_proj.weight": "blk.{}.ffn_gate.weight",
    "model.layers.{}.mlp.up_proj.weight": "blk.{}.ffn_up.weight",
    "model.layers.{}.mlp.down_proj.weight": "blk.{}.ffn_down.weight",

    "model.layers.{}.input_layernorm.weight": "blk.{}.attn_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "blk.{}.ffn_norm.weight",
}
```

### Custom Tensor Mapping

```python
# For custom architectures, extend the mapping
class CustomConverter(Converter):
    def __init__(self):
        super().__init__()

    def get_tensor_name_map(self):
        return {
            # Your custom mappings
            "model.custom_layer.weight": "custom.weight",
            # ... more mappings ...
        }

    def set_gguf_parameters(self):
        # Set architecture-specific metadata
        self.gguf_writer.add_string("general.architecture", "myarch")
        self.gguf_writer.add_uint32("myarch.block_count", self.hparams["n_layers"])
        # ... more parameters ...
```

### Handling Special Cases

#### 1. Tied Embeddings

Some models share input and output embeddings:

```python
def handle_tied_embeddings(self):
    """
    Handle models where input embeddings == output embeddings
    """
    if self.config.get("tie_word_embeddings", False):
        # Don't duplicate the embedding tensor
        # GGUF will reference the same data
        self.gguf_writer.add_tensor(
            "output.weight",
            self.tensors["token_embd.weight"],  # Reference same tensor
            raw_dtype=gguf.GGMLQuantizationType.F16
        )
```

#### 2. Vocabulary Padding

```python
def pad_vocabulary(self, vocab_size):
    """
    Pad vocabulary to multiple of 32 for efficiency
    """
    target_size = ((vocab_size + 31) // 32) * 32
    padding = target_size - vocab_size

    if padding > 0:
        # Add dummy tokens
        for i in range(padding):
            self.vocab.add_token(f"<pad_{i}>")

        # Pad embedding matrix
        embed_tensor = self.tensors["token_embd.weight"]
        pad_tensor = torch.zeros(padding, embed_tensor.shape[1])
        self.tensors["token_embd.weight"] = torch.cat([embed_tensor, pad_tensor])

    return target_size
```

#### 3. Grouped Query Attention

```python
def convert_gqa_weights(self):
    """
    Convert GQA weights where n_head_kv < n_head
    """
    n_head = self.hparams["num_attention_heads"]
    n_head_kv = self.hparams.get("num_key_value_heads", n_head)

    if n_head_kv < n_head:
        # K and V have fewer heads
        # Weights are already in correct format
        # Just set metadata
        self.gguf_writer.add_uint32("llama.attention.head_count", n_head)
        self.gguf_writer.add_uint32("llama.attention.head_count_kv", n_head_kv)
```

### Metadata Extraction

```python
def extract_metadata(self, config):
    """
    Extract all required metadata from config
    """
    # General metadata
    metadata = {
        "general.architecture": self.detect_architecture(config),
        "general.name": config.get("_name_or_path", "unknown"),
        "general.file_type": 1,  # F16
    }

    # Architecture-specific (e.g., LLaMA)
    if metadata["general.architecture"] == "llama":
        metadata.update({
            "llama.context_length": config["max_position_embeddings"],
            "llama.embedding_length": config["hidden_size"],
            "llama.block_count": config["num_hidden_layers"],
            "llama.feed_forward_length": config["intermediate_size"],
            "llama.attention.head_count": config["num_attention_heads"],
            "llama.attention.head_count_kv": config.get("num_key_value_heads",
                                                        config["num_attention_heads"]),
            "llama.rope.dimension_count": config.get("rope_dim",
                                                     config["hidden_size"] // config["num_attention_heads"]),
            "llama.rope.freq_base": config.get("rope_theta", 10000.0),
            "llama.attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
        })

    return metadata
```

---

## Post-Conversion Quantization

After converting to F16 GGUF, quantize for smaller size:

### Quantization Command

```bash
# Build quantization tool
make quantize

# Quantize model
./quantize model-f16.gguf model-q4_k_m.gguf q4_k_m

# Options:
# q4_0      - 4-bit, small, lower quality
# q4_k_s    - 4-bit, k-quant, small
# q4_k_m    - 4-bit, k-quant, medium (recommended)
# q5_k_s    - 5-bit, k-quant, small
# q5_k_m    - 5-bit, k-quant, medium
# q6_k      - 6-bit, k-quant, large
# q8_0      - 8-bit, very large, minimal loss
```

### Batch Quantization

```bash
#!/bin/bash
# quantize_all.sh - Quantize to multiple formats

MODEL_F16="model-f16.gguf"
OUTPUT_DIR="quantized"

mkdir -p $OUTPUT_DIR

# Define quantization formats
QUANTS=(
    "q4_k_s"
    "q4_k_m"
    "q5_k_s"
    "q5_k_m"
    "q6_k"
    "q8_0"
)

for quant in "${QUANTS[@]}"; do
    echo "Quantizing to $quant..."
    ./quantize $MODEL_F16 "$OUTPUT_DIR/model-$quant.gguf" $quant
done

echo "Quantization complete!"
ls -lh $OUTPUT_DIR/
```

### Quantization Quality Comparison

```python
# test_quantization.py
from llama_cpp import Llama
import numpy as np

def test_perplexity(model_path, test_file="wikitext-2-raw/wiki.test.raw"):
    """
    Measure perplexity to assess quantization quality
    """
    model = Llama(model_path=model_path, n_ctx=512)

    with open(test_file) as f:
        text = f.read()

    # Calculate perplexity
    tokens = model.tokenize(text.encode())
    total_logprob = 0
    count = 0

    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]

        # Get logits
        logits = model.eval(context)
        logprob = logits[target] - np.log(np.sum(np.exp(logits)))

        total_logprob += logprob
        count += 1

    perplexity = np.exp(-total_logprob / count)
    return perplexity

# Test all quantizations
models = {
    "f16": "model-f16.gguf",
    "q8_0": "model-q8_0.gguf",
    "q5_k_m": "model-q5_k_m.gguf",
    "q4_k_m": "model-q4_k_m.gguf",
    "q4_k_s": "model-q4_k_s.gguf",
}

results = {}
for name, path in models.items():
    ppl = test_perplexity(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    results[name] = {"perplexity": ppl, "size_mb": size_mb}

# Print results
for name, metrics in results.items():
    print(f"{name:10} - Size: {metrics['size_mb']:7.1f} MB, "
          f"Perplexity: {metrics['perplexity']:.2f}")

# Example output:
# f16        - Size: 13543.2 MB, Perplexity: 5.68
# q8_0       - Size:  7289.4 MB, Perplexity: 5.69 (+0.01)
# q5_k_m     - Size:  4974.6 MB, Perplexity: 5.72 (+0.04)
# q4_k_m     - Size:  4217.3 MB, Perplexity: 5.78 (+0.10)
# q4_k_s     - Size:  3964.1 MB, Perplexity: 5.85 (+0.17)
```

---

## Troubleshooting

### Issue 1: Unknown Architecture

**Error**: `Unknown architecture: custom_model_type`

**Solution**:
```bash
# Check config.json
cat model_dir/config.json | grep architecture

# If architecture not supported, add manually
python convert_hf_to_gguf.py ./model_dir \
  --metadata "general.architecture=llama" \
  --outfile model.gguf
```

### Issue 2: Tensor Shape Mismatch

**Error**: `Tensor shape mismatch for 'blk.0.attn_q.weight'`

**Cause**: Model uses different tensor layout

**Solution**:
```python
# Create custom converter with transpose
class CustomConverter(Converter):
    def prepare_tensors(self):
        for name, tensor in self.tensors.items():
            if "attn_q" in name or "attn_k" in name:
                # Transpose if needed
                if tensor.shape != expected_shape:
                    tensor = tensor.transpose(0, 1)
            self.tensors[name] = tensor
```

### Issue 3: Missing Tensors

**Error**: `Required tensor 'output.weight' not found`

**Cause**: Tied embeddings or different naming

**Solution**:
```python
# Check for tied embeddings
if "lm_head.weight" not in state_dict:
    # Use input embeddings for output
    state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]
```

### Issue 4: Vocabulary Issues

**Error**: `Tokenizer mismatch: expected 32000, got 32001`

**Solution**:
```bash
# Pad vocabulary
python convert_hf_to_gguf.py ./model_dir \
  --pad-vocab \
  --outfile model.gguf

# Or manually set vocab size
--metadata "llama.vocab_size=32000"
```

### Issue 5: Large Model Conversion

**Error**: `MemoryError` when converting 70B+ models

**Solution**:
```python
# Use lazy loading and process tensors one at a time
class LazyConverter(Converter):
    def convert_tensors(self):
        for tensor_name in self.tensor_names:
            # Load one tensor
            tensor = self.load_tensor(tensor_name)

            # Convert and write
            self.gguf_writer.add_tensor(
                self.map_tensor_name(tensor_name),
                tensor
            )

            # Free memory
            del tensor
            gc.collect()
```

---

## Automation and Best Practices

### Automated Conversion Pipeline

```python
#!/usr/bin/env python3
# auto_convert.py - Automated conversion pipeline

import subprocess
import os
import json
from pathlib import Path

class ModelConverter:
    def __init__(self, hf_model_path, output_dir):
        self.hf_model_path = Path(hf_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_config(self):
        """Load model configuration"""
        config_path = self.hf_model_path / "config.json"
        with open(config_path) as f:
            return json.load(f)

    def get_model_info(self, config):
        """Extract model information"""
        arch = config.get("architectures", ["Unknown"])[0]
        n_params = self.estimate_parameters(config)

        return {
            "architecture": arch,
            "parameters": n_params,
            "name": config.get("_name_or_path", "unknown")
        }

    def estimate_parameters(self, config):
        """Estimate parameter count"""
        hidden = config.get("hidden_size", 4096)
        layers = config.get("num_hidden_layers", 32)
        vocab = config.get("vocab_size", 32000)
        ffn = config.get("intermediate_size", 11008)

        # Rough estimate
        params = (vocab * hidden +  # Embeddings
                 layers * (4 * hidden * hidden +  # Attention
                          3 * hidden * ffn) +    # FFN
                 vocab * hidden)  # Output

        return params

    def convert_to_gguf(self, outtype="f16"):
        """Convert HF model to GGUF"""
        model_name = self.hf_model_path.name
        output_file = self.output_dir / f"{model_name}-{outtype}.gguf"

        cmd = [
            "python", "convert_hf_to_gguf.py",
            str(self.hf_model_path),
            "--outfile", str(output_file),
            "--outtype", outtype
        ]

        print(f"Converting {model_name} to GGUF ({outtype})...")
        subprocess.run(cmd, check=True)

        return output_file

    def quantize(self, f16_file, quant_type):
        """Quantize GGUF model"""
        output_file = f16_file.with_name(
            f16_file.stem.replace("-f16", f"-{quant_type}") + ".gguf"
        )

        cmd = [
            "./quantize",
            str(f16_file),
            str(output_file),
            quant_type
        ]

        print(f"Quantizing to {quant_type}...")
        subprocess.run(cmd, check=True)

        return output_file

    def test_model(self, model_file):
        """Quick test of converted model"""
        from llama_cpp import Llama

        print(f"Testing {model_file.name}...")

        model = Llama(model_path=str(model_file), n_ctx=512)

        output = model.create_completion(
            "Hello, world!",
            max_tokens=10
        )

        print(f"✓ Model loads and generates: {output['choices'][0]['text'][:50]}")

    def full_conversion(self, quant_types=None):
        """Complete conversion pipeline"""
        if quant_types is None:
            quant_types = ["q4_k_m", "q5_k_m", "q8_0"]

        # Load and display info
        config = self.load_config()
        info = self.get_model_info(config)
        print(f"Model: {info['name']}")
        print(f"Architecture: {info['architecture']}")
        print(f"Parameters: ~{info['parameters'] / 1e9:.1f}B")

        # Convert to F16
        f16_file = self.convert_to_gguf("f16")
        print(f"✓ F16 conversion complete: {f16_file.stat().st_size / 1024**3:.2f} GB")

        # Test F16
        self.test_model(f16_file)

        # Quantize
        quantized_files = []
        for quant in quant_types:
            try:
                quant_file = self.quantize(f16_file, quant)
                print(f"✓ {quant} quantization complete: {quant_file.stat().st_size / 1024**3:.2f} GB")

                # Test quantized
                self.test_model(quant_file)

                quantized_files.append(quant_file)
            except Exception as e:
                print(f"✗ Failed to quantize to {quant}: {e}")

        print("\n=== Conversion Summary ===")
        print(f"F16: {f16_file.stat().st_size / 1024**3:.2f} GB")
        for qf in quantized_files:
            print(f"{qf.stem.split('-')[-1].upper()}: {qf.stat().st_size / 1024**3:.2f} GB")

# Usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auto_convert.py <huggingface_model_path> [output_dir]")
        sys.exit(1)

    hf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./converted"

    converter = ModelConverter(hf_path, output_dir)
    converter.full_conversion()
```

### Usage:

```bash
# Download model
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./Mistral-7B-v0.1

# Run automated conversion
python auto_convert.py ./Mistral-7B-v0.1 ./output

# Output:
# Model: mistralai/Mistral-7B-v0.1
# Architecture: MistralForCausalLM
# Parameters: ~7.2B
# Converting to GGUF (f16)...
# ✓ F16 conversion complete: 13.48 GB
# ✓ Model loads and generates
# Quantizing to q4_k_m...
# ✓ q4_k_m quantization complete: 4.07 GB
# ...
```

---

## Summary

Model conversion enables running any HuggingFace model in llama.cpp:

✅ **Formats**: Understand HuggingFace and GGUF structures
✅ **Conversion**: Use convert_hf_to_gguf.py for most models
✅ **Advanced**: Custom tensor mapping and metadata handling
✅ **Quantization**: Post-conversion quantization for efficiency
✅ **Troubleshooting**: Common issues and solutions
✅ **Automation**: Batch conversion and testing pipelines

**Next Steps**:
- Complete Lab 7.4 on model conversion
- Convert a HuggingFace model to GGUF
- Quantize and benchmark different formats
- Build an automated conversion pipeline

---

**References**:
- GGUF Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- llama.cpp Conversion Guide: https://github.com/ggerganov/llama.cpp/discussions/2948
- HuggingFace Hub: https://huggingface.co/models
- Quantization Guide: llama.cpp quantization documentation

# GGUF Format Specification: Technical Deep Dive
## From Binary Layout to Practical Implementation

**Format**: GGUF (GPT-Generated Unified Format)
**Specification**: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
**Version**: 3 (Current)
**Learning Module**: Module 1 - Foundations
**Estimated Reading Time**: 18 minutes
**Relevance to llama.cpp**: ⭐⭐⭐⭐⭐ Critical - This is THE storage format

---

## Executive Summary

GGUF is a binary file format designed specifically for efficient storage and loading of large language models in inference scenarios. It replaces earlier formats (GGML, GGMF, GGJT) with a **self-contained, extensible, and mmap-friendly** design that enables:

- ✅ Single-file deployment (no external dependencies)
- ✅ Instant loading via memory mapping
- ✅ Rich metadata for model configuration
- ✅ Multiple quantization formats in one specification
- ✅ Forward/backward compatibility
- ✅ Cross-platform portability

**For practitioners**: GGUF is what makes llama.cpp fast. Understanding it helps you choose the right quantization, debug loading issues, and optimize inference performance.

---

## The Problem GGUF Solves

### Legacy Format Limitations

**GGML/GGMF/GGJT formats had critical issues:**

1. **Limited metadata**: Model configuration stored externally or inferred from filename
2. **Poor versioning**: Breaking changes required new format variants (GGMF → GGJT → etc.)
3. **No extensibility**: Adding new tensor types or metadata required format changes
4. **Fragile loading**: Missing metadata caused silent failures or incorrect behavior
5. **Tokenizer separation**: Vocabulary stored separately, causing deployment headaches

**Example of the old way:**
```bash
# Multiple files required
model.ggml           # Model weights
config.json          # Model configuration
tokenizer.json       # Tokenizer
vocab.txt            # Vocabulary
# Missing any file = broken model
```

**The GGUF way:**
```bash
# Single file contains everything
llama-2-7b.Q4_K_M.gguf   # Model + config + tokenizer + metadata
# One file, guaranteed to work
```

---

## Design Goals and Philosophy

### 1. **Single-File Deployment**

**Goal**: One file contains everything needed for inference.

**What's included:**
- Model architecture specification
- Tensor weights (with multiple quantization levels)
- Tokenizer vocabulary and merges
- Model hyperparameters
- Training configuration metadata
- Custom metadata (chat templates, system prompts, etc.)

**Benefit**: Simplifies deployment, versioning, and distribution. No dependency hell.

### 2. **Memory-Mapped File Compatibility**

**What is mmap?**
Memory mapping treats files as if they were RAM, enabling:
- **Zero-copy loading**: OS loads file pages on-demand
- **Instant startup**: No upfront loading time
- **Shared memory**: Multiple processes share one model copy
- **Virtual memory**: Model larger than RAM? No problem (uses swap)

**Technical requirement:**
All tensor data must be properly aligned (typically 32-byte alignment) to enable efficient mmap operations.

**Real-world impact:**
```python
# Without mmap: 30-60 seconds to load 7B model
model = load_model_traditional("model.bin")

# With mmap: <1 second to "load" same model
model = load_model_mmap("model.gguf")  # Actually just maps file
```

### 3. **Extensibility**

**Design pattern**: New capabilities added via metadata, not format changes.

**Example evolution:**
```
GGUF v1 → v2: Added metadata types (arrays)
GGUF v2 → v3: Added alignment specification
Future: Add new metadata keys without breaking existing readers
```

**Backward compatibility**: Older readers skip unknown metadata gracefully.

### 4. **Language Agnostic**

**Goal**: Any language can read/write GGUF.

**Implementations exist in:**
- Python (reference: `gguf` library)
- C/C++ (llama.cpp)
- Rust (llama-rs, candle)
- JavaScript/TypeScript (web inference)
- Go, Swift, Java, etc.

**Why it matters**: Enables ecosystem growth without vendor lock-in.

---

## File Structure: A Deep Dive

### High-Level Layout

```
┌─────────────────────────────────────┐
│  Magic Number (4 bytes)             │  "GGUF" in ASCII
├─────────────────────────────────────┤
│  Version (4 bytes)                  │  Currently: 3
├─────────────────────────────────────┤
│  Tensor Count (8 bytes)             │  Number of tensors
├─────────────────────────────────────┤
│  Metadata KV Count (8 bytes)        │  Number of metadata pairs
├─────────────────────────────────────┤
│                                     │
│  Metadata Section                   │  Variable size
│  (Key-Value Pairs)                  │  Architecture, config, etc.
│                                     │
├─────────────────────────────────────┤
│                                     │
│  Tensor Information Section         │  Variable size
│  (Name, dims, type, offset)         │  Describes each tensor
│                                     │
├─────────────────────────────────────┤
│  Padding (alignment)                │  Ensure proper alignment
├─────────────────────────────────────┤
│                                     │
│  Tensor Data Section                │  Bulk of file size
│  (Raw tensor weights)               │  Aligned, quantized data
│                                     │
└─────────────────────────────────────┘
```

### Section 1: Header (20 bytes fixed)

#### Magic Number (4 bytes)
```
Hex: 0x47 0x47 0x55 0x46
ASCII: "GGUF"
Purpose: File format identification
```

**Verification code:**
```python
def is_gguf_file(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        return magic == b'GGUF'
```

#### Version (4 bytes, uint32)
```
Current: 3
Format: Little-endian unsigned 32-bit integer
```

**Version history:**
- v1: Initial release
- v2: Added array metadata types
- v3: Added alignment specification, current standard

**Breaking vs. Non-breaking changes:**
- Version increment = structural change (breaking)
- New metadata keys = non-breaking (extensibility)

#### Tensor Count (8 bytes, uint64)
```
Range: 0 to 2^64 - 1
Typical: 200-400 for LLaMA models
Example: LLaMA-7B has 291 tensors
```

#### Metadata Count (8 bytes, uint64)
```
Range: 0 to 2^64 - 1
Typical: 20-50 entries
```

### Section 2: Metadata (Variable Size)

Metadata uses a **type-length-value (TLV)** encoding pattern.

#### Metadata Value Types (13 types)

| Type ID | Name | Size | Description |
|---------|------|------|-------------|
| 0 | UINT8 | 1 byte | Unsigned 8-bit integer |
| 1 | INT8 | 1 byte | Signed 8-bit integer |
| 2 | UINT16 | 2 bytes | Unsigned 16-bit integer |
| 3 | INT16 | 2 bytes | Signed 16-bit integer |
| 4 | UINT32 | 4 bytes | Unsigned 32-bit integer |
| 5 | INT32 | 4 bytes | Signed 32-bit integer |
| 6 | FLOAT32 | 4 bytes | 32-bit floating point |
| 7 | BOOL | 1 byte | Boolean (0=false, 1=true) |
| 8 | STRING | Variable | UTF-8 string with length prefix |
| 9 | ARRAY | Variable | Array of any type (homogeneous) |
| 10 | UINT64 | 8 bytes | Unsigned 64-bit integer |
| 11 | INT64 | 8 bytes | Signed 64-bit integer |
| 12 | FLOAT64 | 8 bytes | 64-bit floating point |

#### Metadata Key Format

**Rules:**
- Valid ASCII characters only
- Hierarchical: segments separated by dots (.)
- Segments in `lower_snake_case`
- Maximum 65,535 bytes length

**Examples:**
```
general.architecture         → "llama"
general.name                 → "LLaMA 2 7B"
general.file_type            → 2 (mostly Q4_K_M)
llama.context_length         → 4096
llama.embedding_length       → 4096
llama.block_count            → 32
llama.attention.head_count   → 32
tokenizer.ggml.model         → "gpt2"
tokenizer.ggml.tokens        → ["<s>", "</s>", ...]
```

**Namespace convention:**
- `general.*`: General model information
- `{arch}.*`: Architecture-specific params (llama, gpt2, etc.)
- `tokenizer.*`: Tokenizer configuration
- `{custom}.*`: User-defined metadata

#### Common Metadata Keys

**Essential keys (required):**
```python
{
    "general.architecture": "llama",      # Model type
    "general.name": "LLaMA 2 7B",        # Human-readable name
    "general.file_type": 2,               # Quantization type
    "general.alignment": 32,              # Tensor alignment

    # Architecture-specific (LLaMA example)
    "llama.context_length": 4096,
    "llama.embedding_length": 4096,
    "llama.block_count": 32,              # Number of transformer layers
    "llama.feed_forward_length": 11008,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 32,  # For GQA
    "llama.rope.dimension_count": 128,
    "llama.rope.freq_base": 10000.0,

    # Tokenizer
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.tokens": [...],       # Vocabulary array
    "tokenizer.ggml.scores": [...],       # Token scores
    "tokenizer.ggml.token_type": [...],   # Token types
    "tokenizer.ggml.merges": [...],       # BPE merges
    "tokenizer.ggml.bos_token_id": 1,
    "tokenizer.ggml.eos_token_id": 2,
}
```

**Optional but useful:**
```python
{
    "general.description": "Instruction-tuned model",
    "general.license": "llama2",
    "general.source.url": "https://...",
    "general.source.huggingface.repository": "meta-llama/Llama-2-7b-hf",
    "general.tags": ["chat", "instruct"],

    # Chat template for instruction models
    "tokenizer.chat_template": "{% for message in messages %}...",

    # Training info
    "general.training.dataset": "mixture",
    "general.training.epochs": 1.0,
}
```

### Section 3: Tensor Information (Variable Size)

Each tensor described by:

#### Tensor Info Structure
```
┌─────────────────────────────────────┐
│  Name Length (8 bytes, uint64)      │
├─────────────────────────────────────┤
│  Name (variable, UTF-8 string)      │
├─────────────────────────────────────┤
│  Dimension Count (4 bytes, uint32)  │  Typically 1-4
├─────────────────────────────────────┤
│  Dimensions (8 × n_dims bytes)      │  Each dimension: uint64
├─────────────────────────────────────┤
│  Data Type (4 bytes, uint32)        │  Quantization format
├─────────────────────────────────────┤
│  Offset (8 bytes, uint64)           │  Offset in tensor data section
└─────────────────────────────────────┘
```

#### Tensor Naming Convention

**Standard names follow pattern:**
```
{component}.{block_id}.{layer}.{quantization_suffix}

Examples:
token_embd.weight                    # Embedding layer
blk.0.attn_norm.weight              # Block 0, attention norm
blk.0.attn_q.weight                 # Block 0, query projection
blk.0.attn_k.weight                 # Block 0, key projection
blk.0.attn_v.weight                 # Block 0, value projection
blk.0.attn_output.weight            # Block 0, attention output
blk.0.ffn_norm.weight               # Block 0, FFN norm
blk.0.ffn_gate.weight               # Block 0, FFN gate (SwiGLU)
blk.0.ffn_up.weight                 # Block 0, FFN up projection
blk.0.ffn_down.weight               # Block 0, FFN down projection
output_norm.weight                   # Final norm
output.weight                        # LM head
```

**Naming must end with:**
- `.weight` for weights
- `.bias` for biases

This convention is used by quantization tools to identify tensor types.

#### Tensor Dimensions

**Storage order: reversed from PyTorch!**

Example: PyTorch shape `[4096, 4096]` stored as `[4096, 4096]` in metadata.

**Common shapes (LLaMA-7B):**
- Embedding: `[32000, 4096]` - vocab_size × hidden_dim
- Attention QKV: `[4096, 4096]` - hidden_dim × hidden_dim
- FFN Gate/Up: `[4096, 11008]` - hidden_dim × intermediate_dim
- FFN Down: `[11008, 4096]` - intermediate_dim × hidden_dim
- Norm weights: `[4096]` - hidden_dim (1D)

### Section 4: Tensor Data Types (40+ formats)

#### Standard Floating Point

| Type | ID | Bytes/Value | Description |
|------|----|----|-------------|
| F32 | 0 | 4 | Standard 32-bit float |
| F16 | 1 | 2 | 16-bit float (half precision) |
| BF16 | 30 | 2 | Brain float 16 (better range than F16) |
| F64 | 28 | 8 | 64-bit double precision |

#### Integer Types

| Type | ID | Description |
|------|----|----|
| I8 | 24 | 8-bit signed integer |
| I16 | 25 | 16-bit signed integer |
| I32 | 26 | 32-bit signed integer |
| I64 | 27 | 64-bit signed integer |

#### Quantization Formats (K-Quants)

**Q4 variants (4-bit):**
- `Q4_0` (ID 2): Basic 4-bit quantization, 32 values per block
- `Q4_1` (ID 3): 4-bit with offset, better accuracy
- `Q4_K_S` (ID 12): K-quant small, most aggressive
- `Q4_K_M` (ID 13): K-quant medium, **best balance** ⭐
- `Q4_K_L` (ID 14): K-quant large, higher quality

**Q5 variants (5-bit):**
- `Q5_0` (ID 6): Basic 5-bit
- `Q5_1` (ID 7): 5-bit with offset
- `Q5_K_S` (ID 15): K-quant small
- `Q5_K_M` (ID 16): K-quant medium, **good quality** ⭐
- `Q5_K_L` (ID 17): K-quant large

**Q6 and Q8 (higher quality):**
- `Q6_K` (ID 18): 6-bit K-quant, **near-lossless** ⭐
- `Q8_0` (ID 8): 8-bit, almost no quality loss
- `Q8_1` (ID 9): 8-bit with offset

**IQ variants (Improved Quantization):**
- `IQ2_XXS` to `IQ4_XS`: Even more aggressive quantization
- Lower memory, higher quality loss
- Experimental, for extreme memory constraints

**Special formats:**
- `TQ1_0`, `TQ2_0`: Ternary quantization
- `MXFP4`: Mixed precision FP4 (NVIDIA collaboration)

#### K-Quant Explained

**What makes K-quants special?**

Traditional quantization (Q4_0, Q5_0):
- Uniform quantization across all weights
- Same precision everywhere
- Simple but suboptimal

**K-quants (Q4_K_M, Q5_K_M, Q6_K):**
- **Mixed precision** within each tensor
- Important weights get higher precision
- Less important weights get lower precision
- Automatic sensitivity detection

**Block structure example (Q4_K_M):**
```
Super-block (256 values):
  ├─ Block 1 (64 values): Q4 + Q6 scales
  ├─ Block 2 (64 values): Q4 + Q6 scales
  ├─ Block 3 (64 values): Q4 + Q6 scales
  └─ Block 4 (64 values): Q4 + Q6 scales

Scales stored in Q6 (higher precision)
Values stored in Q4 (lower precision)
= Better quality than pure Q4
```

**Practical impact:**
```
Model: LLaMA-7B
Q4_0:   3.5 GB, perplexity: 6.58
Q4_K_M: 3.8 GB, perplexity: 6.45  ← Better quality, slightly larger
Q5_K_M: 4.4 GB, perplexity: 6.31  ← Even better
Q6_K:   5.2 GB, perplexity: 6.20  ← Near original
FP16:   13 GB,  perplexity: 6.18  ← Baseline
```

### Section 5: Alignment and Padding

**Purpose**: Enable memory-mapped loading and efficient access.

**Alignment value**: Specified in metadata (`general.alignment`), typically 32 bytes.

**Where alignment matters:**
1. Start of tensor data section
2. Start of each individual tensor
3. Padding between tensors

**Example calculation:**
```python
def calculate_padding(current_offset, alignment=32):
    remainder = current_offset % alignment
    if remainder == 0:
        return 0
    return alignment - remainder

# Example:
current_offset = 12345
padding = calculate_padding(12345, 32)  # Returns: 7
next_aligned_offset = 12345 + 7 = 12352  # Divisible by 32
```

**Binary representation:**
```
Metadata end: byte 12345
Padding: 7 bytes of 0x00
Tensor data start: byte 12352 (aligned to 32)
```

---

## Endianness and Portability

### Default: Little-Endian

**All values stored as little-endian:**
- Header values (version, counts)
- Metadata values
- Tensor data (floats, ints)

**Why little-endian?**
- x86, x86-64, ARM are little-endian
- Covers 99%+ of inference hardware
- No byte-swapping needed on most platforms

### Big-Endian Support

**For big-endian systems** (PowerPC, some RISC-V):
- Separate GGUF files with big-endian encoding
- Typically named with `.be.gguf` suffix
- Same structure, different byte order

**Portability consideration:**
```python
import struct
import sys

# Reading uint32 portably
def read_uint32(file):
    bytes_data = file.read(4)
    # Always interpret as little-endian for GGUF
    return struct.unpack('<I', bytes_data)[0]
```

---

## File Naming Convention

### Standard Format
```
<BaseName>-<SizeLabel>-<FineTune>-<Version>-<Encoding>-<Type>-<Shard>.gguf
```

### Components Breakdown

#### Required
- **BaseName**: Model family (e.g., `llama`, `mistral`, `phi`)
- **SizeLabel**: Model size (e.g., `7b`, `13b`, `70b`)
- **Version**: Model version (e.g., `v1`, `v2`, `v2.1`)

#### Optional
- **FineTune**: Fine-tuning variant (e.g., `chat`, `instruct`, `code`)
- **Encoding**: Quantization format (e.g., `Q4_K_M`, `Q5_K_M`, `F16`)
- **Type**: Special type (e.g., `lora`, `vocab`)
- **Shard**: Multi-file shard number (e.g., `00001-of-00004`)

### Examples
```
# Standard model
llama-2-7b-v2-Q4_K_M.gguf

# Chat fine-tune
llama-2-7b-chat-v2-Q5_K_M.gguf

# Code-specialized
codellama-13b-python-v1-Q4_K_M.gguf

# Full precision
mistral-7b-v0.1-F16.gguf

# Multi-shard (for large models)
llama-2-70b-v2-Q4_K_M-00001-of-00004.gguf
llama-2-70b-v2-Q4_K_M-00002-of-00004.gguf
llama-2-70b-v2-Q4_K_M-00003-of-00004.gguf
llama-2-70b-v2-Q4_K_M-00004-of-00004.gguf

# LoRA adapter
llama-2-7b-alpaca-lora-Q8_0.gguf
```

**Parsing example:**
```python
import re

pattern = r'([^-]+)-(\d+[bm])-?(.*?)-?(v[\d.]+)?-?([QF][\w_]+)?\.gguf'
filename = "llama-2-7b-chat-v2-Q4_K_M.gguf"

match = re.match(pattern, filename)
if match:
    base, size, finetune, version, quant = match.groups()
    print(f"Base: {base}")      # llama
    print(f"Size: {size}")      # 7b
    print(f"Tune: {finetune}")  # chat
    print(f"Ver: {version}")    # v2
    print(f"Quant: {quant}")    # Q4_K_M
```

---

## Practical Implications for Inference

### 1. **Loading Performance**

**Memory-mapped loading:**
```python
# Traditional loading: reads entire file into RAM
# Time: O(file_size / disk_bandwidth)
# LLaMA-7B Q4_K_M: ~3.8GB / 500MB/s = 7.6 seconds

# mmap loading: maps file, loads pages on-demand
# Time: O(1) - instant!
# Actual page loads happen during inference
```

**llama.cpp implementation:**
```cpp
// Enable mmap (default)
llama_model_params params = llama_model_default_params();
params.use_mmap = true;  // Fast startup
params.use_mlock = true; // Prevent swapping (optional)

llama_model* model = llama_load_model_from_file(
    "llama-2-7b.Q4_K_M.gguf",
    params
);
```

### 2. **Memory Requirements**

**Formula:**
```
Total_RAM = Model_size + KV_cache + Activations + Overhead

Where:
- Model_size: Tensor data size (from GGUF file size)
- KV_cache: 2 × n_layers × d_model × ctx_len × bytes_per_value
- Activations: ~2-4 GB during inference
- Overhead: ~500 MB - 1 GB
```

**Example (LLaMA-7B, Q4_K_M):**
```
Model: 3.8 GB
KV-cache (ctx=2048): 2 × 32 × 4096 × 2048 × 2 bytes (FP16) = 1.0 GB
Activations: ~2 GB
Overhead: ~500 MB
Total: ~7.3 GB minimum RAM
```

**Optimization: KV-cache quantization**
```
KV-cache with Q8_0: 1.0 GB → 500 MB
KV-cache with Q4_0: 1.0 GB → 250 MB (experimental)
```

### 3. **Quantization Selection Guide**

**Decision tree:**

```
Do you have enough VRAM/RAM for F16?
├─ Yes → Use F16 (maximum quality)
└─ No ↓

Can you fit Q8_0?
├─ Yes → Use Q8_0 (near-lossless)
└─ No ↓

Can you fit Q6_K?
├─ Yes → Use Q6_K (excellent quality)
└─ No ↓

Can you fit Q5_K_M?
├─ Yes → Use Q5_K_M (good balance)
└─ No ↓

Use Q4_K_M (standard choice)
└─ Still too big? → Q4_K_S or IQ variants
```

**Quality vs. Size trade-off:**
```
Format    Size    Perplexity    Speed    Use Case
------------------------------------------------------
F16       100%    Baseline      Slow     Research, benchmarking
Q8_0      50%     +0.02        Fast      Near-lossless needed
Q6_K      40%     +0.04        Fast      High quality production
Q5_K_M    35%     +0.15        Faster   Good balance
Q4_K_M    25%     +0.30        Fastest  Standard production ⭐
Q4_K_S    23%     +0.45        Fastest  Constrained memory
IQ4_XS    20%     +0.60        Fastest  Extreme constraints
```

### 4. **Multi-GPU and Sharding**

**When to use sharding:**
- Model doesn't fit in single GPU VRAM
- Want to distribute across multiple GPUs

**Example: LLaMA-70B on 2x A100 (80GB each)**
```bash
# Single shard (won't fit)
llama-2-70b-Q4_K_M.gguf  # 38 GB - fits, but tight

# Multi-shard (safer, more flexible)
llama-2-70b-Q4_K_M-00001-of-00002.gguf  # 19 GB on GPU 0
llama-2-70b-Q4_K_M-00002-of-00002.gguf  # 19 GB on GPU 1
```

**Loading sharded models:**
```python
from llama_cpp import Llama

model = Llama(
    model_path="llama-2-70b-Q4_K_M.gguf",  # Automatically finds shards
    n_gpu_layers=80,  # Offload all layers
    tensor_split=[0.5, 0.5]  # 50% on each GPU
)
```

### 5. **Metadata Utilization**

**Auto-configuration from metadata:**
```python
# llama.cpp automatically reads:
# - Context length from llama.context_length
# - Rope frequency from llama.rope.freq_base
# - Attention heads from llama.attention.head_count
# - GQA configuration from llama.attention.head_count_kv
# - Tokenizer from tokenizer.ggml.*

# You rarely need to specify these manually!
model = Llama("model.gguf")  # Just works
```

**Custom metadata for applications:**
```python
# Reading custom metadata
import gguf

reader = gguf.GGUFReader("model.gguf")

# Check if it's a chat model
metadata = reader.fields
if "tokenizer.chat_template" in metadata:
    print("This is a chat model!")
    chat_template = metadata["tokenizer.chat_template"]
```

---

## Common Issues and Debugging

### Issue 1: "File is not a valid GGUF file"

**Causes:**
- File corrupted during download
- Partial download
- Wrong format (GGML, not GGUF)

**Solution:**
```bash
# Check magic number
xxd -l 4 model.gguf
# Should show: 47 47 55 46 (GGUF in hex)

# Verify file size matches expected
ls -lh model.gguf

# Re-download if corrupted
wget -c https://... # -c resumes interrupted downloads
```

### Issue 2: "Invalid version"

**Cause:** File is GGUF but wrong version (v1, v2 vs. v3)

**Solution:**
```bash
# Check version
xxd -s 4 -l 4 model.gguf
# Should show: 03 00 00 00 (version 3 in little-endian)

# Convert old GGUF to new version
python convert_gguf_version.py old.gguf new.gguf
```

### Issue 3: "Alignment error" or "Cannot mmap"

**Cause:** Tensor data not properly aligned

**Solution:**
```python
# Re-quantize with proper alignment
from llama_cpp import llama_model_quantize

llama_model_quantize(
    input="model_unaligned.gguf",
    output="model_aligned.gguf",
    qtype="Q4_K_M"
)
```

### Issue 4: "Out of memory" during loading

**Cause:** Model larger than available RAM/VRAM

**Solutions:**
```python
# 1. Use smaller quantization
# Q5_K_M → Q4_K_M saves ~20-25%

# 2. Disable mmap (uses less memory during load)
params.use_mmap = False

# 3. Offload fewer layers to GPU
model = Llama("model.gguf", n_gpu_layers=20)  # Instead of 32

# 4. Use CPU inference with memory mapping
model = Llama("model.gguf", n_gpu_layers=0, use_mmap=True)
```

### Issue 5: "Tokenizer not found"

**Cause:** Old GGUF file without embedded tokenizer

**Solution:**
```python
# Specify tokenizer explicitly
model = Llama(
    "model.gguf",
    vocab_only=False,
    # Use external tokenizer
    tokenizer_path="tokenizer.json"
)
```

---

## Tools and Utilities

### 1. **Inspecting GGUF Files**

**Python library:**
```python
from gguf import GGUFReader

reader = GGUFReader("llama-2-7b.Q4_K_M.gguf")

# Print all metadata
for key, field in reader.fields.items():
    print(f"{key}: {field.parts}")

# List all tensors
for tensor in reader.tensors:
    print(f"{tensor.name}: {tensor.shape} ({tensor.tensor_type})")
```

**Command-line tool:**
```bash
# Using gguf-py
pip install gguf

# Dump metadata
python -m gguf.gguf_dump model.gguf

# Extract specific metadata
python -m gguf.gguf_dump model.gguf | grep "general\."
```

### 2. **Converting to GGUF**

**From HuggingFace:**
```bash
# Using llama.cpp convert script
python convert_hf_to_gguf.py \
    --model meta-llama/Llama-2-7b-hf \
    --outfile llama-2-7b-F16.gguf \
    --outtype f16

# Then quantize
./llama-quantize \
    llama-2-7b-F16.gguf \
    llama-2-7b-Q4_K_M.gguf \
    Q4_K_M
```

**From PyTorch:**
```python
# Using custom conversion
from gguf import GGUFWriter, GGMLQuantizationType

writer = GGUFWriter("output.gguf", "llama")

# Add metadata
writer.add_architecture("llama")
writer.add_context_length(4096)
writer.add_embedding_length(4096)
# ... more metadata

# Add tensors
writer.add_tensor("token_embd.weight", embedding_tensor)
writer.add_tensor("blk.0.attn_q.weight", q_weight)
# ... more tensors

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

### 3. **Quantizing GGUF**

**Using llama.cpp:**
```bash
# Quantize F16 → Q4_K_M
./llama-quantize \
    input-f16.gguf \
    output-q4.gguf \
    Q4_K_M

# Quantize with importance matrix (better quality)
./llama-quantize \
    input-f16.gguf \
    output-q4.gguf \
    Q4_K_M \
    --imatrix imatrix.dat
```

**Using Python:**
```python
from llama_cpp import llama_model_quantize

llama_model_quantize(
    input="model-f16.gguf",
    output="model-q4.gguf",
    qtype="Q4_K_M",
    nthread=8
)
```

### 4. **Validating GGUF Files**

**Integrity check:**
```python
def validate_gguf(filepath):
    from gguf import GGUFReader

    try:
        reader = GGUFReader(filepath)

        # Check version
        assert reader.version == 3, f"Version {reader.version} not supported"

        # Check required metadata
        required = [
            "general.architecture",
            "general.file_type",
            "general.alignment"
        ]
        for key in required:
            assert key in reader.fields, f"Missing required key: {key}"

        # Check tensor count
        assert len(reader.tensors) > 0, "No tensors found"

        print(f"✓ Valid GGUF file: {filepath}")
        print(f"  Architecture: {reader.fields['general.architecture'].parts}")
        print(f"  Tensors: {len(reader.tensors)}")
        return True

    except Exception as e:
        print(f"✗ Invalid GGUF file: {e}")
        return False
```

---

## Advanced Topics

### 1. **Custom Quantization Schemes**

**Implementing IQ (Improved Quantization):**

IQ formats use importance matrix to determine per-tensor quantization levels.

```bash
# Generate importance matrix
./llama-imatrix \
    -m model-f16.gguf \
    -f calibration_data.txt \
    -o imatrix.dat

# Quantize with importance matrix
./llama-quantize \
    model-f16.gguf \
    model-iq4.gguf \
    IQ4_XS \
    --imatrix imatrix.dat
```

**How it works:**
1. Run calibration data through model
2. Measure activation magnitudes per tensor
3. Assign higher precision to high-importance tensors
4. Assign lower precision to low-importance tensors

**Result:** Better quality than uniform quantization at same average bit-width.

### 2. **Mixed-Precision Models**

**Creating custom mixed-precision:**

```python
# Different quantization per layer type
quantization_map = {
    "token_embd": "Q8_0",       # Embeddings: higher precision
    "output": "Q8_0",            # LM head: higher precision
    "attn_q": "Q4_K_M",         # Attention: medium
    "attn_k": "Q4_K_M",
    "attn_v": "Q5_K_M",         # Values: slightly higher
    "attn_output": "Q4_K_M",
    "ffn_gate": "Q4_K_M",
    "ffn_up": "Q4_K_M",
    "ffn_down": "Q5_K_M",       # FFN down: slightly higher
    "norm": "F32"                # Norms: full precision
}

# Use llama.cpp quantize with custom map
# (requires custom quantize implementation)
```

### 3. **LoRA Adapters in GGUF**

**GGUF supports LoRA adapters:**

```python
# Load base model
base_model = Llama("llama-2-7b.Q4_K_M.gguf")

# Load LoRA adapter (also in GGUF format)
model_with_lora = Llama(
    "llama-2-7b.Q4_K_M.gguf",
    lora_path="alpaca-lora.gguf",
    lora_scale=1.0  # Adapter strength
)
```

**LoRA GGUF contains:**
- Only the low-rank adaptation matrices
- Same metadata format
- Much smaller file size (~100-500 MB)

### 4. **Streaming and Partial Loading**

**Loading specific layers only:**

```python
# Advanced: selective layer loading
# Useful for distributed inference or debugging

from gguf import GGUFReader

reader = GGUFReader("model.gguf")

# Load only attention layers
for tensor in reader.tensors:
    if "attn" in tensor.name:
        # Load this tensor
        tensor_data = tensor.data
        # Process...
```

---

## Future Directions

### Upcoming Features (Community Discussions)

1. **Metadata versioning**: Independent version for metadata schema
2. **Compression**: Tensor data compression (LZ4, ZSTD) for storage
3. **Streaming support**: HTTP range requests for cloud-hosted models
4. **Multi-modal tensors**: Vision, audio embeddings in same file
5. **Training metadata**: Checkpointing information for continued training

### GGUF vs. Competitors

| Format | Strengths | Weaknesses |
|--------|-----------|------------|
| **GGUF** | Single-file, mmap, rich metadata | Larger than some formats |
| **SafeTensors** | Fast, secure, PyTorch-native | No metadata, separate config needed |
| **GPTQ** | Excellent quantization quality | Format tied to specific hardware |
| **AWQ** | Great 4-bit quality | Requires specific kernels |
| **ONNX** | Wide tool support | Large files, complex graph |

**GGUF advantages:**
- ✅ Single file deployment
- ✅ Works on CPU, GPU, Apple Silicon
- ✅ Memory-mapped loading
- ✅ Rich ecosystem (llama.cpp, koboldcpp, etc.)

---

## Hands-On Example: Reading GGUF

### Complete Python Example

```python
#!/usr/bin/env python3
"""
GGUF File Inspector
Demonstrates reading and analyzing GGUF files
"""

from gguf import GGUFReader
import sys

def inspect_gguf(filepath):
    """Inspect a GGUF file and print comprehensive information"""

    reader = GGUFReader(filepath)

    print("=" * 80)
    print(f"GGUF File Inspection: {filepath}")
    print("=" * 80)

    # Header info
    print(f"\n📋 HEADER")
    print(f"  Version: {reader.version}")
    print(f"  Tensor Count: {len(reader.tensors)}")
    print(f"  Metadata Count: {len(reader.fields)}")

    # Essential metadata
    print(f"\n🏗️  ARCHITECTURE")
    arch = reader.fields.get("general.architecture")
    name = reader.fields.get("general.name")
    file_type = reader.fields.get("general.file_type")

    if arch:
        print(f"  Architecture: {arch.parts[0]}")
    if name:
        print(f"  Name: {name.parts[0]}")
    if file_type:
        print(f"  File Type: {file_type.parts[0]}")

    # Model parameters
    print(f"\n⚙️  PARAMETERS")
    for key in sorted(reader.fields.keys()):
        if key.startswith(f"{arch.parts[0]}."):
            value = reader.fields[key]
            print(f"  {key}: {value.parts[0] if value.parts else value}")

    # Tokenizer info
    print(f"\n📝 TOKENIZER")
    tok_model = reader.fields.get("tokenizer.ggml.model")
    tok_tokens = reader.fields.get("tokenizer.ggml.tokens")

    if tok_model:
        print(f"  Model: {tok_model.parts[0]}")
    if tok_tokens:
        print(f"  Vocabulary Size: {len(tok_tokens.parts)}")
        print(f"  Sample tokens: {tok_tokens.parts[:5]}")

    # Tensor analysis
    print(f"\n📊 TENSORS ({len(reader.tensors)} total)")

    # Group by type
    tensor_types = {}
    total_size = 0

    for tensor in reader.tensors:
        dtype = tensor.tensor_type.name
        tensor_types[dtype] = tensor_types.get(dtype, 0) + 1

        # Calculate size
        elements = 1
        for dim in tensor.shape:
            elements *= dim
        size = elements * tensor.tensor_type.itemsize
        total_size += size

    print(f"  Total Size: {total_size / 1e9:.2f} GB")
    print(f"\n  By Type:")
    for dtype, count in sorted(tensor_types.items()):
        print(f"    {dtype}: {count} tensors")

    # Sample tensors
    print(f"\n  Sample Tensors:")
    for tensor in reader.tensors[:10]:
        shape_str = " × ".join(map(str, tensor.shape))
        print(f"    {tensor.name:<40} {shape_str:<20} {tensor.tensor_type.name}")

    if len(reader.tensors) > 10:
        print(f"    ... and {len(reader.tensors) - 10} more")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_gguf.py <model.gguf>")
        sys.exit(1)

    inspect_gguf(sys.argv[1])
```

**Run it:**
```bash
python inspect_gguf.py llama-2-7b.Q4_K_M.gguf
```

---

## Key Takeaways

### ✅ For Practitioners

1. **GGUF = Efficiency**: Single file, mmap-enabled, instant loading
2. **Metadata is King**: Everything needed for inference is embedded
3. **Quantization Flexibility**: 40+ formats, choose based on your constraints
4. **K-quants are optimal**: Q4_K_M, Q5_K_M, Q6_K for best quality/size ratio
5. **Alignment matters**: Proper alignment enables memory mapping

### ✅ For Developers

1. **Language-agnostic**: Design enables implementations in any language
2. **Extensible**: Add new metadata without breaking format
3. **Versioned**: Structural changes get version increments
4. **Self-documenting**: Metadata describes model completely

### ✅ For Researchers

1. **Reproducible**: Same file works identically across platforms
2. **Inspectable**: All parameters visible in metadata
3. **Quantization testbed**: Easy to compare different quantization schemes
4. **Portable**: Share single file, includes everything

---

## Related Resources

### Official Documentation
- GGUF Specification: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp: https://github.com/ggml-org/llama.cpp
- gguf-py library: https://pypi.org/project/gguf/

### Community Resources
- GGUF discussion: https://github.com/ggml-org/ggml/discussions
- llama.cpp discussions: https://github.com/ggml-org/llama.cpp/discussions
- Model quantization guide: llama.cpp wiki

### Related Papers
- "LLM.int8(): 8-bit Matrix Multiplication for Transformers" (Dettmers et al.)
- "GPTQ: Accurate Post-Training Quantization" (Frantar et al.)
- "AWQ: Activation-aware Weight Quantization" (Lin et al.)

---

**Document Created By**: Agent 1 (Research Curator)
**Last Updated**: 2025-11-18
**Related Materials**:
- LLaMA Paper Summary: `llama-paper-summary.md`
- Lab 2: GGUF Format Exploration
- Lab 5: Quantization Techniques Comparison

**Next Steps**:
1. Complete Lab 2 to explore GGUF files hands-on
2. Experiment with different quantization formats
3. Understand memory trade-offs for your use case

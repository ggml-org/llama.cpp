# LLaVA: Large Language and Vision Assistant

**Paper**: "Visual Instruction Tuning" (LLaVA)
**Authors**: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
**Published**: April 2023 | **Module**: 7 - Multimodal Models
**Link**: https://arxiv.org/abs/2304.08485 | **Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

LLaVA combines a vision encoder (CLIP) with a large language model (LLaMA/Vicuna) via a simple projection layer, enabling multimodal understanding. It demonstrates that instruction-following capabilities can emerge through visual instruction tuning on GPT-4 generated data.

**Key Innovation**: Vision encoder → Projection → Frozen LLM = Simple but effective multimodal architecture

---

## 1. Architecture

```
┌──────────────────────────────────────────────────┐
│                  LLaVA Model                     │
├──────────────────────────────────────────────────┤
│                                                  │
│  Image                                           │
│    ↓                                             │
│  ┌─────────────────┐                             │
│  │ CLIP Vision     │ (Frozen during training)    │
│  │ Encoder         │                             │
│  │ (ViT-L/14)      │                             │
│  └────────┬────────┘                             │
│           │ Image features [1, 256, 1024]        │
│           ↓                                      │
│  ┌─────────────────┐                             │
│  │  Projection     │ (Trainable)                 │
│  │  Matrix W       │                             │
│  └────────┬────────┘                             │
│           │ Text-aligned features [1, 256, 4096] │
│           ↓                                      │
│  ┌─────────────────┐                             │
│  │  Language Model │ (Frozen in stage 1,         │
│  │  (LLaMA/Vicuna) │  fine-tuned in stage 2)     │
│  └────────┬────────┘                             │
│           │                                      │
│           ↓                                      │
│    Generated Text                                │
└──────────────────────────────────────────────────┘
```

---

## 2. Training Procedure

### Stage 1: Pre-training (Projection Only)
```python
# Goal: Align vision and language features
# Dataset: CC3M (595K image-caption pairs)
# Trainable: Projection matrix only
# Frozen: Vision encoder + LLM

loss = CrossEntropy(
    predictions=llm(projection(vision_features)),
    targets=caption_tokens
)
# Only projection weights updated
```

### Stage 2: Fine-tuning (End-to-End)
```python
# Goal: Instruction following
# Dataset: 150K GPT-4 generated instruction-response pairs
# Trainable: Projection + LLM
# Frozen: Vision encoder

instruction_examples = [
    {
        "image": "cat.jpg",
        "instruction": "Describe this image in detail",
        "response": "A fluffy orange cat sitting on a windowsill..."
    },
    {
        "image": "chart.png",
        "instruction": "What trend does this chart show?",
        "response": "The chart shows an upward trend in sales..."
    }
]
```

---

## 3. Prompt Format

```python
# System prompt
system = "You are a helpful vision assistant."

# User prompt (multimodal)
user_prompt = f"""<image>
{image_embedding}
</image>

{text_instruction}
"""

# Example
instruction = "What objects are in this image?"
# Model sees: image features + text instruction
# Model generates: "The image contains a dog, a frisbee, and a park bench."
```

---

## 4. llama.cpp Support

### LLaVA Integration

```bash
# Build with LLaVA support
cmake -B build -DGGML_LLAVA=ON
cmake --build build

# Convert LLaVA model to GGUF
python convert-llava-to-gguf.py \
  --model liuhaotian/llava-v1.5-7b \
  --output llava-v1.5-7b-f16.gguf

# Quantize
./llama-quantize llava-v1.5-7b-f16.gguf llava-v1.5-7b-q4_K_M.gguf Q4_K_M

# Run inference
./llama-llava-cli \
  -m llava-v1.5-7b-q4_K_M.gguf \
  --mmproj clip-vit-large-patch14.gguf \
  --image cat.jpg \
  -p "Describe this image"
```

### Architecture in GGUF

```
llava-v1.5-7b.gguf:
- LLM weights (LLaMA/Vicuna)
- Projection matrix

clip-vit-large.gguf:
- Vision encoder (separate file)

# Inference:
# 1. Load image → CLIP encoder → features
# 2. Project features → LLM space
# 3. Concatenate with text tokens
# 4. LLM generates response
```

---

## 5. Applications

**Image Understanding**:
```
Q: "What is happening in this image?"
A: "A group of people are playing volleyball on a beach at sunset."
```

**Visual Reasoning**:
```
Q: "Why might this person be wearing a helmet?"
A: "The person is riding a bicycle, and wearing a helmet provides safety..."
```

**OCR and Document Understanding**:
```
Q: "What does the sign say?"
A: "The sign reads 'No Parking' in red letters."
```

---

## Key Takeaways

✅ **Simple architecture**: Vision encoder + projection + LLM
✅ **Instruction tuning**: GPT-4 generated data enables strong performance
✅ **llama.cpp support**: Quantized multimodal inference on CPU
✅ **Foundation for multimodal**: Many variants (LLaVA-1.5, LLaVA-Next, etc.)

---

## Further Reading

- **Paper**: https://arxiv.org/abs/2304.08485
- **GitHub**: https://github.com/haotian-liu/LLaVA
- **llama.cpp LLaVA**: Examples in repository

---

**Status**: Complete | Module 7 (1/2) papers

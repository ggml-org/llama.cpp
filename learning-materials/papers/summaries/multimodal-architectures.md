# Multimodal LLM Architectures Survey

**Module**: 7 - Multimodal Models | **Impact**: ⭐⭐⭐⭐

---

## Executive Summary

Survey of multimodal architectures combining vision, language, and other modalities. Focus on practical implementations in llama.cpp ecosystem.

---

## 1. Architecture Patterns

### Pattern 1: Frozen Vision + Projection (LLaVA)
```
Image → CLIP (frozen) → Projection → LLM
- Simplest approach
- Fast training (only projection)
- Good performance
```

### Pattern 2: Cross-Attention (Flamingo)
```
Image → Vision Encoder
Text → LLM with cross-attention to image features
- More complex
- Better multimodal fusion
- Slower inference
```

### Pattern 3: Unified Encoder (Unified-IO)
```
Image + Text → Shared Encoder → Decoder
- Single model for all modalities
- Requires large-scale training
- Maximum flexibility
```

---

## 2. Key Models

**LLaVA** (Covered separately): Simple, effective
**CLIP**: Vision-language pretraining
**BLIP-2**: Querying Transformer for vision-language
**GPT-4V**: Proprietary, state-of-the-art
**Fuyu**: Native multimodal (no separate vision encoder)

---

## 3. llama.cpp Multimodal Support

**Supported**:
- LLaVA (vision + language)
- CLIP-based models
- BakLLaVA (improved LLaVA variant)

**Usage Pattern**:
```bash
# Two-file setup
./llama-llava-cli \
  -m llm.gguf \          # Language model
  --mmproj vision.gguf \  # Vision encoder + projection
  --image input.jpg
```

---

## 4. Practical Considerations

**Memory**:
- Vision encoder: +500MB - 2GB
- Projection: +50-200MB
- Plan for 2-3GB extra vs text-only

**Performance**:
- Image encoding: 50-200ms (one-time per image)
- Text generation: Same as text-only LLM
- Bottleneck: Usually LLM, not vision encoder

**Quantization**:
- Vision encoder: Usually kept at Q8_0 or F16
- LLM: Can quantize aggressively (Q4_K_M)
- Projection: Small, keep higher precision

---

## 5. Key Takeaways

✅ **LLaVA-style**: Best supported in llama.cpp
✅ **Separate files**: Vision encoder + LLM modularity
✅ **Quantization**: Aggressive for LLM, conservative for vision
✅ **Use cases**: Image captioning, VQA, OCR, multimodal chat

---

**Status**: Complete | Module 7 Complete (2/2) papers

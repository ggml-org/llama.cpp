# Vision-Language Models in LLaMA.cpp

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [LLaVA: Large Language and Vision Assistant](#llava-large-language-and-vision-assistant)
4. [MiniCPM-V and Other Models](#minicpm-v-and-other-models)
5. [Implementation in llama.cpp](#implementation-in-llamacpp)
6. [Image Preprocessing](#image-preprocessing)
7. [Inference Pipeline](#inference-pipeline)
8. [Performance Optimization](#performance-optimization)
9. [Common Issues and Solutions](#common-issues-and-solutions)
10. [Best Practices](#best-practices)

---

## Introduction

Vision-language models (VLMs) combine visual understanding with language generation, enabling AI systems to perceive and reason about images. In llama.cpp, these models allow you to:

- **Visual Question Answering**: Ask questions about image content
- **Image Captioning**: Generate descriptive text for images
- **Visual Reasoning**: Perform multi-step reasoning combining vision and language
- **Document Understanding**: Extract information from documents with figures

### Why Vision-Language Models?

Traditional language models are text-only, limiting their application to scenarios where visual context is crucial. VLMs bridge this gap by:

1. **Multimodal Understanding**: Processing both visual and textual information
2. **Rich Context**: Incorporating visual details into responses
3. **Real-World Applications**: Enabling applications like document analysis, accessibility tools, and autonomous systems

---

## Architecture Overview

### Core Components

Vision-language models in llama.cpp consist of three main components:

```
┌─────────────────┐
│  Image Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vision Encoder  │  ← Pre-trained (e.g., CLIP-ViT)
│  (CLIP/ViT)     │
└────────┬────────┘
         │ Image Features
         ▼
┌─────────────────┐
│   Projector     │  ← Maps vision to language space
│  (MLP/Linear)   │
└────────┬────────┘
         │ Visual Tokens
         ▼
┌─────────────────┐
│ Language Model  │  ← Pre-trained LLM (e.g., LLaMA)
│   (LLaMA/etc)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Output     │
└─────────────────┘
```

### Component Details

#### 1. Vision Encoder
- **Purpose**: Convert images to feature representations
- **Common Architectures**: CLIP-ViT (Vision Transformer), ConvNext
- **Output**: Grid of visual features (e.g., 576 tokens for 336×336 image)
- **Typical Dimensions**: 1024, 1408, or higher dimensional vectors

#### 2. Projector Network
- **Purpose**: Align visual features with language model's embedding space
- **Architecture**: Usually 2-layer MLP or linear projection
- **Mapping**: Vision dimension → Language model dimension
- **Training**: Trained during visual instruction tuning phase

#### 3. Language Model
- **Purpose**: Generate text conditioned on visual tokens
- **Models**: LLaMA, Vicuna, Mistral, and derivatives
- **Integration**: Visual tokens prepended or interleaved with text tokens
- **Context**: Treats visual tokens as additional context

---

## LLaVA: Large Language and Vision Assistant

### Overview

LLaVA (Large Language and Vision Assistant) is one of the most popular vision-language models, notable for its:
- Strong visual understanding capabilities
- Efficient training methodology
- Open-source availability
- Active community support

### Architecture Details

```python
# LLaVA Architecture Breakdown
{
    "vision_encoder": {
        "type": "CLIP-ViT-L/14",
        "input_resolution": 336,  # or 224, 448
        "patch_size": 14,
        "hidden_dim": 1024,
        "layers": 24,
        "num_patches": 576  # (336/14)^2
    },
    "projector": {
        "type": "MLP",
        "input_dim": 1024,  # CLIP output
        "hidden_dim": 4096,
        "output_dim": 4096,  # LLaMA embedding dim
        "layers": 2
    },
    "language_model": {
        "type": "LLaMA-2-7B / 13B / Vicuna",
        "vocab_size": 32000,
        "context_length": 4096
    }
}
```

### Training Methodology

LLaVA uses a two-stage training approach:

#### Stage 1: Pre-training (Feature Alignment)
- **Data**: Image-caption pairs (CC3M, ~600K samples)
- **Frozen**: Vision encoder and LLM
- **Trainable**: Only the projector
- **Goal**: Align vision features with language space
- **Duration**: ~4-8 hours on 8×A100

#### Stage 2: Fine-tuning (Visual Instruction Tuning)
- **Data**: Instruction-following data with images (~150K samples)
- **Frozen**: Vision encoder
- **Trainable**: Projector and LLM
- **Goal**: Teach instruction following with visual context
- **Duration**: ~12-24 hours on 8×A100

### Model Variants

| Model | LLM Base | Vision Encoder | Parameters | Best For |
|-------|----------|----------------|------------|----------|
| LLaVA-1.5-7B | Vicuna-7B | CLIP-ViT-L/14-336 | ~7B | General use, efficiency |
| LLaVA-1.5-13B | Vicuna-13B | CLIP-ViT-L/14-336 | ~13B | Better reasoning |
| LLaVA-1.6-7B | Vicuna-7B | CLIP-ViT-L/14-336 | ~7B | Improved performance |
| LLaVA-1.6-34B | Hermes-34B | CLIP-ViT-L/14-336 | ~34B | Highest quality |

### Input Format

LLaVA uses a special token `<image>` to represent the image position:

```
USER: <image>\nWhat is shown in this image?
ASSISTANT: The image shows...
```

Internally, `<image>` is replaced with 576 visual tokens from the vision encoder.

---

## MiniCPM-V and Other Models

### MiniCPM-V

**Overview**: Efficient vision-language model optimized for resource-constrained environments.

**Key Features**:
- **Compact Size**: 2B-8B parameters vs 7B+ for LLaVA
- **Efficient Design**: Optimized architecture for mobile/edge deployment
- **Strong Performance**: Competitive with larger models
- **Multilingual**: Better support for non-English languages

**Architecture Differences**:
```python
{
    "vision_encoder": "SigLIP-400M",  # More efficient than CLIP
    "image_resolution": 448,          # Higher than LLaVA's 336
    "compression": "Q-Former",        # Additional compression layer
    "language_model": "MiniCPM-2B"    # Smaller but efficient
}
```

**Use Cases**:
- Mobile applications
- Edge devices
- Real-time processing
- Resource-constrained environments

### Other Notable Models

#### 1. **Obsidian-3B**
- Ultra-compact vision-language model
- Based on Phi-2/StableLM
- ~3B parameters total
- Good for quick prototyping

#### 2. **CogVLM**
- Integrated vision-language architecture
- No separate projector needed
- Better visual grounding
- Larger context windows

#### 3. **Qwen-VL**
- Multilingual vision-language model
- Strong performance on Asian languages
- Supports higher resolution images
- Good OCR capabilities

---

## Implementation in llama.cpp

### Model File Structure

LLaVA models in llama.cpp use a multi-file format:

```
llava-v1.5-7b/
├── mmproj-model-f16.gguf      # Vision encoder + projector
└── ggml-model-q4_0.gguf        # Language model (quantized)
```

**mmproj** contains:
- CLIP vision encoder weights
- Projector network weights
- Image preprocessing parameters

**ggml-model** contains:
- Language model weights
- Tokenizer vocabulary
- Generation parameters

### Loading Models

```cpp
// C++ API
struct llava_context {
    struct clip_ctx *ctx_clip;      // Vision encoder
    struct llama_context *ctx_llama; // Language model
};

// Initialize vision model
clip_ctx *clip = clip_model_load(mmproj_path, verbosity);

// Initialize language model
llama_model *model = llama_load_model_from_file(model_path, params);
llama_context *ctx = llama_new_context_with_model(model, ctx_params);
```

```python
# Python API (llama-cpp-python with llava support)
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

chat_handler = Llava15ChatHandler(clip_model_path="mmproj-model-f16.gguf")

llm = Llama(
    model_path="ggml-model-q4_0.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,
    logits_all=True,
)
```

### Architecture Registration

In llama.cpp, vision models use the CLIP backend:

```cpp
// src/llava/clip.cpp
enum clip_projector_type {
    CLIP_PROJECTOR_MLP,      // LLaVA style
    CLIP_PROJECTOR_LINEAR,   // Simple linear projection
    CLIP_PROJECTOR_LDPNET,   // MiniCPM style
};

struct clip_vision_model {
    clip_vision_hparams hparams;

    struct ggml_tensor *class_embedding;
    struct ggml_tensor *patch_embedding;
    struct ggml_tensor *position_embedding;

    std::vector<clip_layer> layers;
};
```

---

## Image Preprocessing

### Standard Preprocessing Pipeline

Vision models require specific image preprocessing:

```python
def preprocess_image(image_path, target_size=336):
    """
    Standard LLaVA preprocessing pipeline
    """
    from PIL import Image
    import numpy as np

    # 1. Load image
    image = Image.open(image_path).convert('RGB')

    # 2. Resize with aspect ratio preservation
    image = resize_with_padding(image, target_size)

    # 3. Normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # 4. Normalize with CLIP statistics
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image_array = (image_array - mean) / std

    # 5. Convert to CHW format
    image_array = np.transpose(image_array, (2, 0, 1))

    return image_array

def resize_with_padding(image, target_size):
    """
    Resize image while maintaining aspect ratio, pad to square
    """
    # Calculate scaling to fit within target_size
    scale = target_size / max(image.size)
    new_size = tuple(int(dim * scale) for dim in image.size)

    # Resize
    image = image.resize(new_size, Image.Resampling.BICUBIC)

    # Create padded image
    padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_pos = ((target_size - new_size[0]) // 2,
                 (target_size - new_size[1]) // 2)
    padded.paste(image, paste_pos)

    return padded
```

### Resolution Considerations

Different models use different resolutions:

| Model | Resolution | Patches | Visual Tokens | Notes |
|-------|------------|---------|---------------|-------|
| LLaVA-1.5 | 336×336 | 24×24 | 576 | Standard |
| LLaVA-1.6 | 336×336 or 672×672 | Variable | 576 or 2304 | Dynamic resolution |
| MiniCPM-V | 448×448 | 32×32 | 1024 | Higher resolution |
| Qwen-VL | 448×448 | Variable | Variable | Adaptive |

**Trade-offs**:
- **Higher Resolution**: Better detail, more compute, longer context
- **Lower Resolution**: Faster inference, less memory, may miss details

---

## Inference Pipeline

### Step-by-Step Process

```python
def llava_inference(model, image_path, prompt):
    """
    Complete LLaVA inference pipeline
    """
    # Step 1: Preprocess image
    image_tensor = preprocess_image(image_path)

    # Step 2: Encode image through CLIP
    # This happens inside llama.cpp's clip_image_encode
    # Returns: [num_patches, vision_dim] e.g., [576, 1024]

    # Step 3: Project to language space
    # Projector: [576, 1024] → [576, 4096]
    # This happens inside clip_image_load

    # Step 4: Prepare text prompt
    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

    # Step 5: Tokenize text
    # <image> token is replaced with image embeddings

    # Step 6: Forward pass through LLM
    # Input: [visual_tokens (576) + text_tokens] → Output: generated text

    # Step 7: Decode and return
    response = model.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": prompt}
            ]
        }]
    )

    return response['choices'][0]['message']['content']
```

### Token Flow Visualization

```
Image (336×336×3)
    ↓ [CLIP Vision Encoder]
Visual Features (576×1024)
    ↓ [Projector MLP]
Visual Tokens (576×4096)
    ↓ [Concatenate with text]
Combined Input: [IMG_TOK_1, IMG_TOK_2, ..., IMG_TOK_576, "What", "is", "in", "this", "image", "?"]
    ↓ [LLaMA Forward Pass]
Output Tokens: ["The", "image", "shows", "a", "cat", ...]
```

### Context Window Management

With 576 visual tokens:
- **Available text context**: `n_ctx - 576 - system_prompt_length`
- **Example**: 2048 context → ~1400 tokens for conversation
- **Recommendation**: Use `n_ctx=4096` for multimodal models

---

## Performance Optimization

### Memory Optimization

**Model Quantization**:
```bash
# Vision encoder: Keep at F16 for quality
mmproj-model-f16.gguf  # 1.7 GB

# Language model: Can quantize aggressively
llava-v1.5-7b-Q4_K_M.gguf   # 4.1 GB
llava-v1.5-7b-Q5_K_M.gguf   # 4.8 GB (better quality)
llava-v1.5-7b-Q8_0.gguf     # 7.2 GB (minimal loss)
```

**Recommendations**:
- **Vision Encoder**: Always use F16 or higher (quality matters)
- **Projector**: F16 (small, critical for alignment)
- **Language Model**: Q4_K_M or Q5_K_M (good quality/size balance)

### Compute Optimization

**Batching Visual Tokens**:
```cpp
// Process all visual tokens in one batch
llama_batch batch = llama_batch_init(n_tokens, 0, 1);

// Add all 576 visual tokens
for (int i = 0; i < n_visual_tokens; i++) {
    llama_batch_add(&batch, visual_token_ids[i], i, {0}, false);
}

// Add text tokens
for (int i = 0; i < n_text_tokens; i++) {
    llama_batch_add(&batch, text_token_ids[i],
                    n_visual_tokens + i, {0}, i == n_text_tokens - 1);
}

llama_decode(ctx, batch);
```

**GPU Offloading**:
```python
# Offload vision encoder and LLM to GPU
llm = Llama(
    model_path="model.gguf",
    chat_handler=handler,
    n_gpu_layers=35,  # Offload LLM layers
    n_ctx=4096,
)
# Note: Vision encoder automatically uses GPU if available
```

### Throughput Optimization

**Pre-encode Images**:
```python
# For multiple queries on same image, cache visual embeddings
visual_embeddings = clip_encode_image(image)  # Do once
responses = []
for question in questions:
    response = llm.generate(visual_embeddings, question)  # Reuse
    responses.append(response)
```

**Benchmark Results** (LLaVA-1.5-7B, A100 GPU):
```
Image Encoding:    ~50-100ms
Visual Projection: ~10-20ms
Text Generation:   ~20-30 tokens/sec
Total Latency:     ~200-500ms (first token)
```

---

## Common Issues and Solutions

### Issue 1: Image Not Recognized

**Symptoms**: Model ignores image or generates unrelated text

**Causes**:
- Incorrect `<image>` token placement
- Image preprocessing mismatch
- Wrong model architecture

**Solutions**:
```python
# Ensure correct format
prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"

# Verify preprocessing matches training
# LLaVA uses CLIP normalization
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

# Check model compatibility
# LLaVA mmproj must match LLaVA LLM base
```

### Issue 2: Out of Memory

**Symptoms**: Crash or slow performance with large images

**Solutions**:
```python
# Reduce context size
n_ctx = 2048  # Instead of 4096

# Use smaller batch size
n_batch = 128  # Instead of 512

# Quantize more aggressively
# Use Q4_K_M instead of Q8_0

# Reduce image resolution (if model supports)
image_size = 224  # Instead of 336
```

### Issue 3: Poor Visual Understanding

**Symptoms**: Incorrect or vague descriptions

**Solutions**:
- **Use higher quality quantization** (Q5_K_M, Q8_0)
- **Increase image resolution** (if supported)
- **Keep mmproj at F16** (never quantize vision encoder)
- **Use better base model** (13B instead of 7B)
- **Improve prompting**:
  ```python
  # Weak prompt
  "What is this?"

  # Better prompt
  "Describe this image in detail, including objects, colors, and actions."
  ```

---

## Best Practices

### Model Selection

1. **For Production**:
   - Use LLaVA-1.5 or LLaVA-1.6 (well-tested)
   - Choose size based on latency requirements
   - Test multiple quantization levels

2. **For Mobile/Edge**:
   - Use MiniCPM-V (2B-4B)
   - Aggressive quantization (Q4_K_S)
   - Lower resolution (224×224)

3. **For Quality**:
   - Use LLaVA-1.6-34B or LLaVA-NeXT
   - Minimal quantization (Q8_0 or F16)
   - Higher resolution (672×672)

### Prompting Strategies

```python
# Generic prompting
prompt = "What is in this image?"

# Detailed prompting
prompt = """Analyze this image and provide:
1. Main objects and their locations
2. Colors and visual characteristics
3. Any text visible in the image
4. The overall scene or context
Be specific and detailed."""

# Task-specific prompting
prompt = """You are a medical imaging assistant. Analyze this X-ray image and:
- Identify any abnormalities
- Describe their location
- Assess severity (if applicable)
Note: This is for educational purposes only."""
```

### Error Handling

```python
def safe_vision_inference(model, image_path, prompt, max_retries=3):
    """
    Robust vision model inference with error handling
    """
    for attempt in range(max_retries):
        try:
            # Validate image
            img = Image.open(image_path)
            if img.size[0] > 4096 or img.size[1] > 4096:
                # Resize very large images
                img.thumbnail((4096, 4096))
                img.save(image_path)

            # Run inference
            response = llava_inference(model, image_path, prompt)

            # Validate response
            if len(response.strip()) < 10:
                raise ValueError("Response too short, likely failed")

            return response

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return f"Error processing image: {str(e)}"
            time.sleep(1)
```

### Performance Monitoring

```python
import time

def benchmark_vision_model(model, test_images):
    """
    Benchmark vision model performance
    """
    metrics = {
        'encoding_time': [],
        'generation_time': [],
        'total_time': [],
        'tokens_generated': []
    }

    for image_path in test_images:
        start = time.time()

        # Track encoding
        encode_start = time.time()
        # (encoding happens internally)
        encode_time = time.time() - encode_start

        # Generate
        gen_start = time.time()
        response = model.generate(image_path, "Describe this image.")
        gen_time = time.time() - gen_start

        total_time = time.time() - start

        metrics['encoding_time'].append(encode_time)
        metrics['generation_time'].append(gen_time)
        metrics['total_time'].append(total_time)
        metrics['tokens_generated'].append(len(response.split()))

    # Calculate statistics
    return {
        'avg_encoding_ms': np.mean(metrics['encoding_time']) * 1000,
        'avg_generation_ms': np.mean(metrics['generation_time']) * 1000,
        'avg_total_ms': np.mean(metrics['total_time']) * 1000,
        'tokens_per_sec': np.mean([t / (g / 1000) for t, g in
                                   zip(metrics['tokens_generated'],
                                       metrics['generation_time'])])
    }
```

---

## Summary

Vision-language models in llama.cpp provide powerful multimodal capabilities:

✅ **Architecture**: Vision encoder + Projector + Language model
✅ **Models**: LLaVA, MiniCPM-V, and growing ecosystem
✅ **Implementation**: Efficient GGUF format with multi-file support
✅ **Optimization**: Quantization, GPU offloading, caching strategies
✅ **Applications**: Visual QA, document understanding, accessibility

**Next Steps**:
- Complete Lab 7.1 to run LLaVA
- Experiment with different models and resolutions
- Build a visual question-answering application
- Explore Lesson 7.2 on embedding models

---

**References**:
- Liu et al. (2023). "Visual Instruction Tuning" (LLaVA paper)
- Radford et al. (2021). "Learning Transferable Visual Models from Natural Language Supervision" (CLIP)
- llama.cpp documentation: https://github.com/ggerganov/llama.cpp
- LLaVA project: https://llava-vl.github.io/

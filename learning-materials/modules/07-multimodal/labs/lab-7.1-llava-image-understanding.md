# Lab 7.1: LLaVA Image Understanding

**Estimated Time**: 60-90 minutes
**Difficulty**: Advanced
**Prerequisites**: Modules 1-3 complete, GPU recommended

## Learning Objectives

By the end of this lab, you will be able to:
- Load and run LLaVA vision-language models
- Perform image understanding and visual question answering
- Optimize multimodal model performance
- Build a practical image analysis application

## Prerequisites

### Software Requirements
```bash
# Install dependencies
pip install llama-cpp-python pillow numpy

# Optional: GPU support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Model Download
```bash
# Download LLaVA 1.5 7B (recommended)
# Language model (~4 GB quantized)
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf

# Vision projector (~1.7 GB)
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf

# Or use llama.cpp model list
ls ~/.cache/huggingface/hub/
```

### Test Images
Prepare 3-5 test images in a `test_images/` directory:
- Natural photos
- Documents/screenshots
- Charts/diagrams
- Artistic images

---

## Part 1: Basic LLaVA Inference (20 minutes)

### Step 1: Verify Installation

Create `test_llava.py`:

```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

print("Testing LLaVA setup...")

# Initialize (will error if models missing)
try:
    handler = Llava15ChatHandler(clip_model_path="mmproj-model-f16.gguf")
    llm = Llama(
        model_path="ggml-model-q4_k.gguf",
        chat_handler=handler,
        n_ctx=2048,
        logits_all=True,
        verbose=False
    )
    print("✓ LLaVA loaded successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Check that model files are in the current directory")
```

Run:
```bash
python test_llava.py
```

**Expected Output**: "✓ LLaVA loaded successfully!"

### Step 2: First Image Query

Use the provided `llava_inference.py` script:

```bash
python llava_inference.py \
    --model ggml-model-q4_k.gguf \
    --mmproj mmproj-model-f16.gguf \
    --image test_images/sample.jpg \
    --prompt "Describe this image in detail."
```

**Questions to Answer**:
1. What resolution does LLaVA use for the input image?
2. How many visual tokens are generated?
3. What is the approximate latency for the first query?

---

## Part 2: Visual Question Answering (30 minutes)

### Challenge 1: Object Detection

Test various prompts on an image:

```bash
# General description
python llava_inference.py \
    --model ggml-model-q4_k.gguf \
    --mmproj mmproj-model-f16.gguf \
    --image test_images/room.jpg \
    --prompt "List all objects visible in this image."

# Specific questions
--prompt "What color is the sofa?"
--prompt "How many windows are in this room?"
--prompt "What is the lighting like?"
```

**Task**: Create a table comparing LLaVA's responses:

| Prompt | Response Quality (1-5) | Accuracy | Notes |
|--------|------------------------|----------|-------|
| List objects | | | |
| Count items | | | |
| Describe colors | | | |
| Spatial relationships | | | |

### Challenge 2: Document Understanding

Test on a document/screenshot:

```bash
python llava_inference.py \
    --model ggml-model-q4_k.gguf \
    --mmproj mmproj-model-f16.gguf \
    --image test_images/document.jpg \
    --prompt "What is the main topic of this document?"
```

**Tasks**:
1. Extract key information from an invoice/receipt
2. Identify text in a chart or diagram
3. Summarize a slide or infographic

**Document**: What works well? What are limitations?

---

## Part 3: Batch Processing (20 minutes)

### Task: Image Gallery Analysis

Process multiple images:

```bash
python llava_inference.py \
    --model ggml-model-q4_k.gguf \
    --mmproj mmproj-model-f16.gguf \
    --image test_images/ \
    --prompt "Describe this image in one sentence."
```

Create `analyze_gallery.py`:

```python
import os
from llava_inference import LLaVAInference

# Initialize model once
llava = LLaVAInference(
    model_path="ggml-model-q4_k.gguf",
    mmproj_path="mmproj-model-f16.gguf",
    n_ctx=4096,
    n_gpu_layers=35  # Adjust for your GPU
)

# Process images
image_dir = "test_images/"
prompts = [
    "What is the main subject?",
    "What is the setting or location?",
    "What emotions or mood does this convey?"
]

for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, image_file)
        print(f"\n{'='*80}")
        print(f"Image: {image_file}")
        print('='*80)

        for prompt in prompts:
            result = llava.query(image_path, prompt, max_tokens=100)
            print(f"\nQ: {prompt}")
            print(f"A: {result['text']}")
```

**Deliverable**: Create a CSV with results:
```csv
image,subject,setting,mood
image1.jpg,"cat","indoor living room","peaceful"
...
```

---

## Part 4: Performance Optimization (20 minutes)

### Experiment 1: Model Quantization

Compare different quantization levels:

```bash
# Q4_K_M (4-bit, medium)
time python llava_inference.py --model ggml-model-q4_k.gguf ...

# Q5_K_M (5-bit, better quality)
time python llava_inference.py --model ggml-model-q5_k.gguf ...

# Q8_0 (8-bit, minimal loss)
time python llava_inference.py --model ggml-model-q8_0.gguf ...
```

**Record**:
- Load time
- First token latency
- Tokens per second
- Memory usage
- Response quality

### Experiment 2: GPU Offloading

Test GPU layer offloading:

```python
import time
from llava_inference import LLaVAInference

configs = [
    {"n_gpu_layers": 0, "name": "CPU only"},
    {"n_gpu_layers": 10, "name": "Partial GPU"},
    {"n_gpu_layers": 35, "name": "Full GPU"},
]

image = "test_images/sample.jpg"
prompt = "Describe this image."

for config in configs:
    print(f"\nTesting: {config['name']}")

    llava = LLaVAInference(
        model_path="ggml-model-q4_k.gguf",
        mmproj_path="mmproj-model-f16.gguf",
        n_gpu_layers=config['n_gpu_layers']
    )

    start = time.time()
    result = llava.query(image, prompt)
    latency = time.time() - start

    print(f"Latency: {latency:.2f}s")
    print(f"Response: {result['text'][:100]}...")
```

**Create a performance chart** showing GPU layers vs. latency.

---

## Part 5: Application Development (30 minutes)

### Project: Visual QA System

Build a simple web interface or CLI app:

**Features**:
1. Upload an image
2. Ask multiple questions
3. Get contextual answers
4. Save Q&A history

**Starter Code** (`visual_qa_app.py`):

```python
import argparse
from llava_inference import LLaVAInference

class VisualQAApp:
    def __init__(self, model_path, mmproj_path):
        self.llava = LLaVAInference(model_path, mmproj_path)
        self.current_image = None
        self.history = []

    def load_image(self, image_path):
        self.current_image = image_path
        print(f"Loaded image: {image_path}")

    def ask(self, question):
        if not self.current_image:
            return "Please load an image first."

        result = self.llava.query(self.current_image, question)
        answer = result['text']

        self.history.append({
            'image': self.current_image,
            'question': question,
            'answer': answer
        })

        return answer

    def run_interactive(self):
        print("Visual QA Interactive Mode")
        print("Commands: /load <path>, /ask <question>, /history, /quit")

        while True:
            cmd = input("\n> ").strip()

            if cmd.startswith("/load "):
                path = cmd[6:]
                self.load_image(path)

            elif cmd.startswith("/ask "):
                question = cmd[5:]
                answer = self.ask(question)
                print(f"Answer: {answer}")

            elif cmd == "/history":
                for i, item in enumerate(self.history, 1):
                    print(f"\n{i}. {item['image']}")
                    print(f"   Q: {item['question']}")
                    print(f"   A: {item['answer'][:100]}...")

            elif cmd == "/quit":
                break

if __name__ == "__main__":
    app = VisualQAApp(
        "ggml-model-q4_k.gguf",
        "mmproj-model-f16.gguf"
    )
    app.run_interactive()
```

**Enhancement Ideas**:
- Add image preview (using PIL)
- Support batch uploads
- Export Q&A to PDF/HTML
- Add suggested questions based on image content

---

## Deliverables

Submit the following:

1. **Results Table** (Part 2): VQA performance on different prompt types
2. **Gallery Analysis CSV** (Part 3): Batch processing results
3. **Performance Report** (Part 4):
   - Quantization comparison
   - GPU offloading analysis
   - Recommendations for production deployment
4. **Visual QA Application** (Part 5):
   - Working code
   - 3 example Q&A sessions
   - Screenshots/demo

---

## Bonus Challenges

### Challenge 1: Multi-Image Comparison
Compare two images and describe differences:
```python
# Pseudo-code - concatenate image descriptions
desc1 = llava.query("image1.jpg", "Describe this image.")
desc2 = llava.query("image2.jpg", "Describe this image.")
comparison = llm.generate(f"Compare: {desc1} vs {desc2}")
```

### Challenge 2: Visual Chain-of-Thought
Implement multi-step reasoning:
```python
# Step 1: Identify objects
objects = llava.query(image, "List all objects.")

# Step 2: Analyze relationships
relationships = llava.query(image, f"How are these objects related: {objects}?")

# Step 3: Make inference
conclusion = llava.query(image, f"Based on {relationships}, what is happening?")
```

### Challenge 3: Fine-tune Prompts
Experiment with prompt engineering:
- System prompts
- Few-shot examples
- Chain-of-thought prompting
- Role-based prompts

---

## Reflection Questions

1. What types of images does LLaVA handle well vs. poorly?
2. How does visual token count impact context window?
3. What are the trade-offs between model size and quality?
4. How would you deploy LLaVA in a production system?
5. What are potential applications in your domain?

---

## Resources

- **LLaVA Paper**: https://arxiv.org/abs/2304.08485
- **llama.cpp Docs**: https://github.com/ggerganov/llama.cpp
- **LLaVA Models**: https://huggingface.co/models?search=llava
- **Vision Transformer**: https://arxiv.org/abs/2010.11929

---

**Lab Complete!** You've mastered LLaVA for visual understanding. Next: Lab 7.2 - Building RAG with Embeddings.

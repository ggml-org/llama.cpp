# DeepSeek-OCR Server

## Quick start

```bash
./build/bin/llama-server \
  -m "/path/to/deepseek-ocr-q4_k_m.gguf" \
  --mmproj "/path/to/mmproj-deepseek-ocr-f16.gguf" \
  --temp 0 --flash-attn off \
  --chat-template deepseek-ocr \
  -ngl 0 -c 2048 --host 0.0.0.0 --port 8000
```

Flags:
- `-ngl 0` — CPU only (required on GPUs with <6GB VRAM)
- `--flash-attn off` — avoids CUDA OOM on low-VRAM GPUs
- `--chat-template deepseek-ocr` — enables the correct prompt format
- `--mmproj-gpu false` — if you still get OOM, also forces mmproj to CPU

## API: `/v1/chat/completions`

```
POST http://localhost:8000/v1/chat/completions
Content-Type: application/json
```

### Request

```json
{
  "model": "deepseek-ocr",
  "max_tokens": 512,
  "temperature": 0,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}},
        {"type": "text", "text": "<|grounding|>Convert the document to markdown."}
      ]
    }
  ]
}
```

### Response

```json
{
  "choices": [{
    "message": {
      "content": "text[[76, 149, 945, 288]]\n<ocr text here>\nequation[[104, 299, 691, 351]]\n\\[latex...\\]"
    }
  }],
  "usage": {
    "prompt_tokens": 277,
    "completion_tokens": 128,
    "total_tokens": 405
  }
}
```

## Tips

**VRAM issues:** The Q4_K_M model + F16 mmproj needs ~2.7GB + compute buffers. On 4GB GPUs (RTX 3050), use `-ngl 0`. For partial GPU offload try `-ngl 12 --flash-attn on -c 1024`.

**Prompt prefix:** Always include `<|grounding|>` in the text to get document OCR mode. Without it the model may behave like a generic chatbot.

**Output tags:** The model outputs `<|ref|>type<|/ref|><|det|>[x1,y1,x2,y2]<|/det|>` bounding boxes in CLI mode. The server strips the `<|ref|>`/`<|det|>` wrapping depending on the chat template prefix.

## Python example

```python
import requests, base64

with open("document.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "deepseek-ocr",
    "max_tokens": 512,
    "temperature": 0,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text",
             "text": "<|grounding|>Convert the document to markdown."}
        ]
    }]
})

print(r.json()["choices"][0]["message"]["content"])
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `500 failed to process image` | GPU OOM — add `-ngl 0 --mmproj-gpu false` |
| `number of bitmaps (1) does not match number of markers (0)` | Missing `--chat-template deepseek-ocr` flag |
| `GGML_ASSERT(batch.n_tokens > 0)` | Outdated build — rebuild with latest patches |
| Output is `<__media__><|grounding|>...` literal text | Missing `--chat-template deepseek-ocr` |

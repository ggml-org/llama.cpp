# Tutorial: Measuring time to first token (TTFT) and time between tokens (TBT)

When deploying LLM applications, understanding latency is critical. This tutorial explains how to measure **TTFT** (Time To First Token) and **TBT** (Time Between Tokens) using llama.cpp.

## Key metrics

| Metric | Definition | Field in response |
|--------|-----------|-------------------|
| **TTFT** | Time from request submission to the first generated token | `timings.prompt_ms` + time to first generated token in streaming |
| **TBT** | Average time between consecutive generated tokens | `timings.predicted_per_token_ms` |
| **Prompt throughput** | Tokens processed per second during prompt evaluation | `timings.prompt_per_second` |
| **Generation throughput** | Tokens generated per second during text generation | `timings.predicted_per_second` |

## Using `llama-server` response timings

Every completion response from `llama-server` includes a `timings` object:

```json
{
  "timings": {
    "cache_n": 236,
    "prompt_n": 1,
    "prompt_ms": 30.958,
    "prompt_per_token_ms": 30.958,
    "prompt_per_second": 32.30,
    "predicted_n": 35,
    "predicted_ms": 661.064,
    "predicted_per_token_ms": 18.887,
    "predicted_per_second": 52.94
  }
}
```

### Interpreting the fields

- `prompt_ms` -- total time spent processing the entire prompt (including non-cached tokens)
- `predicted_ms` -- total time spent generating output tokens
- `predicted_n` -- number of tokens generated
- `prompt_n` -- number of prompt tokens that were actually processed (not cached)
- `cache_n` -- number of prompt tokens reused from KV cache

### Calculating TTFT

In **non-streaming** mode, TTFT is included in `prompt_ms` (all prompt processing happens before the first token is generated):

```
TTFT = prompt_ms (non-streaming)
```

In **streaming** mode, you need to measure it client-side by recording the time between sending the request and receiving the first SSE event containing a generated token:

```python
import time
import requests

start = time.time()
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    stream=True
)

ttft = None
for line in response.iter_lines():
    if line and ttft is None:
        ttft = time.time() - start
        print(f"TTFT: {ttft:.3f}s")
```

### Calculating TBT

TBT is the average time between tokens:

```
TBT = predicted_ms / predicted_n = predicted_per_token_ms
```

For per-token timing in streaming mode, enable `timings_per_token`:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "timings_per_token": true
  }'
```

This adds per-token timing information to each streaming chunk.

## Using `llama-perplexity` for benchmarking

For reproducible benchmarks, use `llama-perplexity` which reports detailed timing:

```sh
llama-perplexity -m model.gguf -f test.txt
```

## Using `llama-bench` for throughput testing

```sh
# Benchmark prompt processing speed
llama-bench -m model.gguf -p 512 -n 128

# Benchmark generation speed
llama-bench -m model.gguf -p 0 -n 128
```

Output columns include `pp512` (prompt processing for 512 tokens) and `tg128` (token generation for 128 tokens), both in tokens per second.

## Enabling internal libllama performance timings

Add `--perf` to any llama.cpp tool for detailed per-operation timings:

```sh
llama-cli -m model.gguf --perf -p "Hello"
```

This prints a breakdown after inference, showing time spent in each operation (matrix multiply, attention, etc.).

## Common latency targets

| Use case | Target TTFT | Target TBT |
|----------|------------|------------|
| Real-time chat | < 500ms | < 50ms |
| Interactive coding | < 1s | < 100ms |
| Batch processing | < 5s | < 200ms |
| Document analysis | < 10s | N/A |

## Factors affecting TTFT and TBT

- **Model size**: Larger models increase both TTFT and TBT
- **Prompt length**: Longer prompts increase TTFT (more tokens to process)
- **Quantization**: Lower-bit quantization reduces latency
- **GPU offloading**: Offloading layers to GPU dramatically reduces TBT
- **Batch size**: Small ubatch sizes can bottleneck prompt processing
- **KV cache reuse**: With `cache_prompt: true`, prompt prefixes are cached, reducing TTFT for repeated prefixes
- **Flash attention**: Enable with `--flash-attn` for faster prompt processing on supported backends

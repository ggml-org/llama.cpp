# Tutorial: KV cache reuse with llama-server

KV cache reuse eliminates redundant prompt processing when the same or similar prompts are sent repeatedly. This tutorial explains how to leverage KV cache reuse in llama-server to reduce latency and improve throughput.

## How it works

When `cache_prompt` is enabled (default), llama-server stores the key-value (KV) states computed during prompt processing. On subsequent requests, if the prompt shares a common prefix with a cached prompt, only the differing suffix is processed.

```
Request 1: "Translate to French: Hello, how are you?"
           |-------- KV cached --------|

Request 2: "Translate to French: Hello, what is your name?"
           |---- reused from cache ----|--- new ---|
```

The response includes `tokens_cached` indicating how many prompt tokens were reused:

```json
{
  "tokens_cached": 5,
  "tokens_evaluated": 8
}
```

Here, 5 tokens were reused from cache, and only 8 new tokens needed processing (the total prompt length is 5 + 8 = 13).

## Enabling KV cache reuse

KV cache reuse is enabled by default. The relevant server options:

```sh
# Launch with prompt caching enabled (default)
llama-server -m model.gguf --cache-prompt

# Set minimum chunk size for cache reuse via KV shifting
llama-server -m model.gguf --cache-reuse 256
```

### Key options

| Option | Description |
|--------|-------------|
| `--cache-prompt` | Enable/disable prompt caching (default: enabled) |
| `--cache-reuse N` | Min chunk size to attempt reusing via KV shifting (default: 0) |
| `--cache-ram N` | Size of the RAM cache in GiB |
| `--cache-idle-slots` | Save idle slots to prompt cache on new task |

## Using cache_prompt in API requests

Pass `cache_prompt: true` in completion requests (it's `true` by default):

```sh
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "prompt": "Translate the following to French:\n\nHello, how are you?",
    "cache_prompt": true
  }'
```

The same applies to the chat completions endpoint:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "system", "content": "You are a French translator."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "cache_prompt": true
  }'
```

## Verifying cache hits

Inspect the `timings` object in the response:

```json
{
  "timings": {
    "cache_n": 25,
    "prompt_n": 5,
    "prompt_ms": 12.3,
    "predicted_n": 20,
    "predicted_ms": 380.5
  }
}
```

- `cache_n > 0` means cache was hit successfully
- Total context tokens = `cache_n + prompt_n + predicted_n`

## Reusing multiple prompt prefixes with slots

llama-server supports parallel slots (`-np`) for concurrent request handling. Each slot maintains its own KV cache.

### Enabling parallel slots

```sh
# Start with 4 parallel slots, each with a dedicated KV cache
llama-server -m model.gguf -np 4 --cache-prompt
```

Each API request can target a specific slot or let the server assign one automatically.

### Slot-specific cache behavior

```sh
# Request using slot 0 (reuses slot 0's KV cache)
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "...", "id_slot": 0, "cache_prompt": true}'
```

With `-np 4`, the server maintains 4 independent KV caches, enabling:
- Dedicated cache per user/session
- Simultaneous requests without cache interference
- Pre-warming multiple caches for common prefixes

### Pre-warming the cache

Send a completion with `n_predict: 0` to process the prompt and cache it without generating output:

```sh
# Cache a system prompt without generating text
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "You are a helpful assistant. Answer concisely.",
    "n_predict": 0,
    "cache_prompt": true
  }'
```

Subsequent requests with the same prefix will benefit from the cached KV state.

## Inspecting cache state

Use the slots monitoring endpoint to see cache usage:

```sh
curl http://localhost:8080/slots
```

Response:

```json
[
  {
    "id": 0,
    "state": 1,
    "tokens_cached": 10
  },
  {
    "id": 1,
    "state": 0,
    "tokens_cached": 0
  }
]
```

- `state: 1` means the slot is idle but has cached tokens
- `state: 2` means the slot is processing

## Saving and restoring slot KV caches

Save a slot's KV cache to disk for later reuse:

```sh
# Start server with slot save path
llama-server -m model.gguf --slot-save-path ./kv-cache/

# Save slot 0's cache via API
curl http://localhost:8080/slots/0/save
```

## Considerations

- **Determinism**: With `cache_prompt: true`, results may not be bit-for-bit identical to processing the full prompt from scratch, because prompt processing and token generation use different batch sizes
- **Memory**: Each active slot consumes memory proportional to context size. Balance `--ctx-size` and `-np` against available RAM/VRAM
- **Cache invalidation**: Changing sampling parameters does not invalidate the cache, but changing the model or context size does
- **Unified KV cache**: When `-np` is not set or set to 1, the server may use a unified KV cache. For guaranteed per-slot caching, explicitly set `-np` to your desired slot count

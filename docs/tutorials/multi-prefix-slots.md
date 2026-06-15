# Tutorial: Reusing multiple prompt prefixes with slots (-np)

When serving an LLM, many requests share common prefixes (system prompts, few-shot examples, document templates). This tutorial shows how to cache multiple distinct prefixes simultaneously using parallel slots.

## The problem with single-slot caching

With a single slot, each new request that changes the prefix invalidates the previous cache:

```
Request 1: "You are a translator.\n\nEnglish: Hello\nFrench:"
Request 2: "You are a summarizer.\n\nText: ..."
```

Request 2 overwrites the KV cache from request 1. When request 1's pattern repeats, the entire prefix must be re-processed.

## Solution: Parallel slots with -np

Use `-np N` to allocate N independent slots, each with its own KV cache:

```sh
# Before: single slot
llama-server -m model.gguf -c 1024

# After: 8 parallel slots (scale context proportionally)
llama-server -m model.gguf -c 8192 -np 8
```

Server slots assignment example:
- Slot 0: Translator system prompt cached
- Slot 1: Summarizer system prompt cached
- Slot 2: Code reviewer system prompt cached
- ...

The server automatically assigns incoming requests to the slot whose cached prefix best matches the new prompt.

## How it works

1. Each slot stores its own independent KV cache.
2. When a request arrives, the server compares its prompt against each slot's cached tokens.
3. The request is assigned to the slot with the longest matching prefix.
4. Only the non-matching suffix is processed.

```sh
# Pre-warm multiple caches with --cache-idle-slots
llama-server -m model.gguf -c 16384 -np 8 --cache-prompt --cache-idle-slots

# Send requests with different prefixes to populate caches
# Prefix 1: Translator
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are a French translator.\n\nEnglish: cat\nFrench: ", "n_predict": 5}'

# Prefix 2: Code reviewer
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are a code reviewer.\n\nCode:\ndef add(a,b):\n    return a-b\n\nReview: ", "n_predict": 30}'
```

## Monitoring slot cache state

```sh
curl http://localhost:8080/slots
```

```json
[
  {"id": 0, "state": 1, "tokens_cached": 45},
  {"id": 1, "state": 1, "tokens_cached": 38},
  {"id": 2, "state": 0, "tokens_cached": 0},
  ...
]
```

- `state: 0` -- idle, no cached tokens
- `state: 1` -- idle, has cached tokens (ready for reuse)
- `state: 2` -- busy, processing a request

Verify cache hits by checking `tokens_cached` in the response `timings`.

## Saving and restoring slot caches

Save slot KV caches to disk:

```sh
# Start server with slot save path
llama-server -m model.gguf -np 4 --slot-save-path ./kv-cache/

# Save all occupied slots
curl -X POST http://localhost:8080/slots/0/save
curl -X POST http://localhost:8080/slots/1/save
```

On restart, caches are automatically restored from the save path.

## Performance considerations

- **Context scaling**: With `-np N`, total VRAM/RAM usage is `N x ctx_size x cache_bytes_per_token`. Scale `-c` accordingly.
- **Generation speed impact**: Parallel slots improve prompt processing latency but may reduce generation throughput due to scheduler overhead. Benchmark with your workload.
- **Slot count**: Match `-np` to the number of distinct prefixes in your workload. Too many slots waste memory.
- **Idle slot caching**: `--cache-idle-slots` saves idle slots to prompt cache and clears them for unified KV usage, useful when slots exceed concurrent requests.

## Use cases

| Scenario | Recommended -np | -c per slot |
|----------|----------------|-------------|
| Multi-tenant SaaS (per-tenant system prompts) | 4-8 | 4096-8192 |
| Document Q&A (per-document prefixes) | 8-16 | 8192+ |
| Agent system (per-agent personality) | 4 | 4096 |
| Translation service (per-language pair) | 8 | 2048 |

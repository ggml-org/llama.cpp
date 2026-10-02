# parallel-decision

Answer a finite JSON schema in one batched forward pass, instead of generating the JSON token by token.

Every field of a schema has a fixed set of allowed values (enums, booleans, bounded integers, numbers on a grid).
After the context, each field's allowed values are scored as token paths that fork from the same KV cache, so all
fields are answered in one `llama_decode` and cannot see each other. Each answer comes back with a probability, and
the JSON object is assembled by code, so it always matches the schema.

This directory holds the engine (`decision-engine.*`), a CLI (`llama-parallel-decision`), and the engine is also
served by `llama-server` as `POST /v1/decision`.

## Build

Same as llama.cpp:

```bash
cmake -B build -DGGML_CUDA=ON        # or plain `cmake -B build` for CPU / Metal
cmake --build build --config Release -j
```

## Run the server

`--decision-seqs N` reserves the sequence slots the decisions need: one holds the cached instructions, one per context
in flight, the rest are the parallel questions. It also switches the KV cache to unified, which is what lets the
branches share the context's cells.

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -fa on -c 32768 --decision-seqs 24 --port 8096
```

With a presets file, one loaded model serves chat and decisions:

```ini
[*]
ngl = 99
fa = on
jinja = 1
parallel = 1
cache-type-k = q8_0
cache-type-v = q8_0
decision-seqs = 24

[gemma-4-12b]
model = ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf
ctx-size = 32768
ubatch-size = 512
decision-seqs = 12
```

```bash
./build/bin/llama-server --models-preset models.ini --models-max 1 --port 8096
```

How many sequences a model affords depends on its attention. A plain-attention model shares the context's cells, so
128 sequences cost almost nothing. A sliding-window model (Gemma) allocates its window per sequence, so keep it low
(12 on a 12 GB card). Hybrid models with recurrent layers (Qwen3.5, Nemotron-H) keep a recurrent state per
sequence (about 50 MB each for Qwen3.5 4B and 9B), and llama.cpp only batches their sequences together when they hold
the same number of tokens. The engine right-pads each group of branches to its longest one, so they still score in a
single pass; the padding comes after the token that is read, so it doesn't change the result.

## POST /v1/decision

`contexts` is a list of 1-256 strings. They share one schema, one set of instructions, and one cached prefix; results
come back in the same order.

```bash
curl http://localhost:8096/v1/decision -H "Content-Type: application/json" -d '{
  "model": "gemma-4-12b",
  "instructions": "Answer each question about this support request from its state.",
  "schema": {
    "category": {"type": "enum", "choices": ["billing","technical","cancellation","other"],
                 "description": "What type of support request is this?"},
    "urgent":   {"type": "boolean", "description": "Does this need urgent handling?"},
    "priority": {"type": "enum", "choices": ["low","medium","high","critical"],
                 "description": "Rate support priority."}
  },
  "contexts": ["I was charged twice and need this fixed today."]
}'
```

```json
{
  "object": "decision",
  "results": [
    {
      "decision": {"category": "billing", "urgent": true, "priority": "high"},
      "fields": {
        "category": {"value": "billing",  "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "urgent":   {"value": true,       "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "priority": {"value": "high",     "probability": 0.74, "scored_nodes": 1, "tree": true}
      },
      "usage": {"context_tokens": 21, "scored_rows": 14}
    }
  ],
  "usage": {"prompt_tokens": 137, "cached_tokens": 116, "context_tokens": 21, "scored_rows": 14},
  "timings": {"prefill_ms": 50.7, "scoring_ms": 50.0, "total_ms": 100.7, "rounds": 1, "per_decision_ms": 100.7}
}
```

(That response is a real one: Gemma 4 12B on an RTX 3060, warm cache.)

### Schema

Compact fields, or a JSON Schema object with `properties`:

| type | keys | notes |
|---|---|---|
| `enum` | `choices` (or `enum`) | 1-255 values |
| `boolean` | - | true / false |
| `integer` | `minimum`, `maximum` | 1-255 values |
| `number` | `minimum`, `maximum`, `step` (`multipleOf` in JSON Schema) | fixed-width decimals |

Numeric fields take `aggregate`: `mode` (default), `median` or `mean`.

### Options

| field | default | meaning |
|---|---|---|
| `instructions` | `""` | prepended to the generated field catalogue; cached with it |
| `mode` | `auto` | `tree` scores every divergence node and returns exact probabilities; `greedy` walks the trie; `auto` picks tree up to `tree_max` values |
| `tree_max` | 128 | per-field switch between tree and greedy |
| `cache_prompt` | true | reuse the cached instructions + schema prefix |

## CLI

`llama-parallel-decision` runs the same engine from a worker process (stdin/stdout protocol, one JSON request per
line). Environment: `DECIDE_TREE`, `DECIDE_TREE_MAX`, `DECIDE_NSEQ`, `DECIDE_SPLIT_BOUNDARY`.

## A UI for it

[decision-playground](https://github.com/thecodacus/decision-playground) is a browser-only playground: it talks
straight to your llama-server, runs a decision and the same question as a chat completion side by side with live
timers, and has a small game whose agents decide through the endpoint.

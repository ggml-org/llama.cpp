# Tutorial: Compute embeddings using llama.cpp

This tutorial shows how to compute text embeddings with llama.cpp using the dedicated `llama-embedding` tool and the `llama-server` API.

## What are embeddings?

Embeddings are dense vector representations of text. They capture semantic meaning, enabling tasks like semantic search, clustering, classification, and reranking. Each embedding is a fixed-length array of floating-point numbers.

## Prerequisites

- A built llama.cpp (see [build guide](../build.md))
- An embedding or general-purpose GGUF model (e.g. [bge-small-en-v1.5](https://huggingface.co/ggml-org/bge-small-en-v1.5-GGUF), or any model with pooling support)

## Method 1: Using `llama-embedding`

The `llama-embedding` tool is the simplest way to compute embeddings from the command line.

### Basic usage

```sh
# Download a model
llama-embedding -hf ggml-org/bge-small-en-v1.5-GGUF --embd-normalize 2 -p "Hello world"
```

### Multiple inputs

Separate inputs with a newline:

```sh
llama-embedding -hf ggml-org/bge-small-en-v1.5-GGUF --embd-normalize 2 -p "Hello world\nHow are you?"
```

### Output formats

```sh
# Default: plain text
llama-embedding -m model.gguf -p "text" --embd-out raw

# JSON output
llama-embedding -m model.gguf -p "text" --embd-out json

# JSON with cosine similarity matrix
llama-embedding -m model.gguf -p "text A\ntext B" --embd-out json+

# Compact array format
llama-embedding -m model.gguf -p "text" --embd-out array
```

### Pooling types

Different models use different pooling strategies. Specify it with `--pooling`:

| Pooling | Description |
|---------|-------------|
| `none` | Returns per-token embeddings |
| `mean` | Average of all token embeddings |
| `cls` | Use the CLS token embedding |
| `last` | Use the last token embedding |
| `rank` | For reranker models (e.g. bge-reranker) |

```sh
# Per-token embeddings
llama-embedding -m model.gguf --pooling none -p "hello"

# Mean pooling (most common for sentence embeddings)
llama-embedding -m model.gguf --pooling mean --embd-normalize 2 -p "hello"
```

### Reranking

For reranker models that score document relevance to a query:

```sh
llama-embedding -m bge-reranker.gguf --pooling rank --cls-sep "|||" -p "What is Python?|||Python is a programming language"
```

### Normalization options

Control embedding normalization with `--embd-normalize`:

| Value | Method |
|-------|--------|
| -1 | No normalization |
| 0 | Max absolute |
| 1 | Taxicab (L1) |
| 2 | Euclidean (L2) |
| >2 | P-norm |

## Method 2: Using `llama-server` API

The HTTP server provides OpenAI-compatible embeddings endpoints.

### Start the server

```sh
llama-server -m model.gguf --pooling mean --embeddings
```

### OpenAI-compatible endpoint: `POST /v1/embeddings`

```sh
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "input": "Hello world",
    "model": "GPT-4",
    "encoding_format": "float"
  }'
```

Multiple inputs:

```sh
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "input": ["Hello", "World"],
    "model": "GPT-4",
    "encoding_format": "float"
  }'
```

### Non-OpenAI endpoint: `POST /embeddings`

This endpoint supports `--pooling none` (per-token embeddings) in addition to all other pooling types:

```sh
curl http://localhost:8080/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "input": ["hello", "world"],
    "model": "GPT-4"
  }'
```

### Reranking endpoint: `POST /reranking`

Requires `--pooling rank` and a reranker model:

```sh
llama-server -m bge-reranker.gguf --pooling rank --reranking

curl http://localhost:8080/reranking \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
    "query": "What is Python?",
    "documents": [
      "Python is a programming language.",
      "The weather is nice today."
    ]
  }'
```

## Method 3: Using the C API

For programmatic use, see `examples/embedding/embedding.cpp` in the repository. The key API calls are:

```cpp
// Set up embedding mode
params.embedding = true;

// Initialize
common_init_from_params(params);

// Tokenize input
std::vector<llama_token> tokens = common_tokenize(ctx, "your text", true, true);

// Decode
llama_batch batch = llama_batch_init(n_tokens, 0, 1);
// ... add tokens to batch ...
llama_decode(ctx, batch);

// Get embeddings based on pooling type
if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
    embd = llama_get_embeddings_ith(ctx, i);     // per-token
} else {
    embd = llama_get_embeddings_seq(ctx, seq_id); // pooled
}

// Normalize
common_embd_normalize(embd, output, n_embd, embd_norm);
```

## Performance tips

- Use `--parallel` (`-np`) to set the number of concurrent sequences
- Set `--batch-size` and `--ubatch-size` appropriately for your input lengths
- Enable `--embd-normalize 2` (Euclidean/L2) for best cosine similarity results
- For batch processing of many short texts, combine them in one call with newline separation

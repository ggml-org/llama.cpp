# Computing embeddings with llama.cpp

This guide shows how to run `llama-server` in embedding mode and get vector
embeddings back over an OpenAI-compatible API. It's aimed at anyone building
retrieval, semantic search, or RAG (retrieval-augmented generation) on top of
llama.cpp.

## What embedding mode does

`llama-server` can run in two modes: normal text generation, or embedding
mode. In embedding mode, the server loads a dedicated embedding model and
exposes an endpoint that turns text into a fixed-size vector instead of
generating new tokens. Those vectors can then be compared (e.g. with cosine
similarity) to find semantically related text, which is the basis of most
retrieval and RAG pipelines.

## Starting the server in embedding mode

Use the `--embedding` (or `--embeddings`) flag together with a dedicated
embedding model. For a quick start, llama.cpp ships a convenience flag that
downloads a small, capable embedding model automatically:

```sh
llama-server --embd-gemma-default --embedding
```

This downloads and serves [EmbeddingGemma](https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF)
and starts the server (by default on `http://127.0.0.1:8080`, though the
port may differ if 8080 is already in use — check the `listening on`
line in the startup log).

You can also point `--embedding` at any GGUF embedding model you already
have locally:

```sh
llama-server -m /path/to/model.gguf --embedding
```

### Relevant flags

| Flag | Purpose |
|---|---|
| `--embedding`, `--embeddings` | Restrict the server to embedding-only use (default: disabled) |
| `--pooling {none,mean,cls,last,rank}` | How token-level embeddings are combined into one vector. If unspecified, the model's default is used. |
| `--embd-normalize N` | Normalization applied to the output vector. `-1`=none, `0`=max absolute, `1`=taxicab, `2`=euclidean (default) |

Pooling strategy matters for retrieval quality: most modern embedding
models expect `mean` or a model-specific default, so it's usually safest to
leave `--pooling` unset unless you know your model needs an override.

## Windows note: HTTPS downloads require OpenSSL

On Windows, a default CMake build of llama.cpp may not link OpenSSL, which
disables HTTPS support entirely. If you see this on startup:

```
E get_repo_commit: error: HTTPS is not supported. Please rebuild with one of:
  -DLLAMA_BUILD_BORINGSSL=ON
  -DLLAMA_BUILD_LIBRESSL=ON
  -DLLAMA_OPENSSL=ON (default, requires OpenSSL dev files installed)
```

it means flags like `--embd-gemma-default` or `-hf` can't reach Hugging
Face to download a model. To fix it:

1. Install OpenSSL (e.g. via conda: `conda install -c conda-forge openssl`).
2. Reconfigure CMake, pointing it at the installed OpenSSL root so
   `FindOpenSSL` can locate it (conda installs outside CMake's default
   search paths):

   ```powershell
   cmake -B build -DLLAMA_OPENSSL=ON -DOPENSSL_ROOT_DIR="<path-to-your-conda-env>\Library"
   ```

   You should see `-- Found OpenSSL: ... (found version "3.x.x")` in the
   configure output before rebuilding.
3. Rebuild: `cmake --build build --config Release -j`

If you'd rather not deal with this, download the GGUF model manually from
Hugging Face and load it with `-m /path/to/model.gguf` instead, which
doesn't require any network access from the server itself.

## Querying the embeddings endpoint

Once the server is running, send a POST request to `/v1/embeddings`
(OpenAI-compatible):

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "embedding"}'
```

> **Windows PowerShell note:** PowerShell handles escaped quotes inside
> double-quoted strings differently from `cmd`/bash, which can mangle a
> `-d "{\"input\": ...}"` argument passed inline. The most reliable
> approach is to write the JSON body to a file first:
>
> ```powershell
> '{"input": "hello world", "model": "embedding"}' | Out-File -Encoding utf8 body.json
> curl.exe http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" --data-binary "@body.json"
> ```

The response includes the embedding vector plus token usage:

```json
{
  "model": "embedding",
  "object": "list",
  "usage": { "prompt_tokens": 4, "total_tokens": 4 },
  "data": [
    {
      "embedding": [0.0627, 0.0341, -0.0266, ...],
      "index": 0,
      "object": "embedding"
    }
  ]
}
```

The length of the `embedding` array depends on the model (EmbeddingGemma
300M outputs 768-dimensional vectors).

## Using embeddings for retrieval

A minimal similarity search pipeline looks like this:

1. Embed each document in your corpus once, and store the vectors
   alongside the source text (in memory, a file, or a vector database).
2. At query time, embed the user's query with the same model and settings.
3. Compute cosine similarity between the query vector and each stored
   document vector.
4. Return the top-N most similar documents as retrieved context.

Because `llama-server` in embedding mode is stateless per request, this
pipeline can be implemented in any language that can make HTTP requests to
the `/v1/embeddings` endpoint — the server itself doesn't need to know
anything about your corpus or storage layer.

## See also

- [server README](../tools/server/README.md) for the full list of server
  flags and endpoints
- [HOWTO-add-model.md](./HOWTO-add-model.md) if you want to add support for
  a new embedding model architecture

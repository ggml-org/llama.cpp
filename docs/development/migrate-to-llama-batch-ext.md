# Migrate from `llama_batch` to `llama_batch_ext`

The old `llama_batch` API was removed in [#30565](https://github.com/ggml-org/llama.cpp/pull/30565). This includes `llama_batch`, `llama_batch_init()`, `llama_batch_get_one()`, `llama_batch_free()`, `llama_encode()` and `llama_decode()`.

Use `llama_batch_ext` and `llama_process()` instead. This guide shows how to port existing code.

Working examples:
- [examples/simple/simple.cpp](/examples/simple/simple.cpp) - single sequence, token input, encoder-decoder
- [tools/mtmd/mtmd-helper-common.h](/tools/mtmd/mtmd-helper-common.h) - embedding input, M-RoPE positions

## Main differences

| Old (`llama_batch`)                                  | New (`llama_batch_ext`)                                         |
|------------------------------------------------------|-----------------------------------------------------------------|
| Plain struct with public arrays                      | Opaque object, use the `llama_batch_ext_*` functions            |
| `llama_batch_init(n_tokens, embd, n_seq_max)`        | `llama_batch_ext_init(ctx)`                                     |
| `llama_batch_get_one(tokens, n_tokens)`              | `llama_batch_ext_init(ctx)` + `llama_batch_ext_add_token()`     |
| `llama_batch_free(batch)`                            | `llama_batch_ext_free(batch)`, or `llama_batch_ext_ptr` in C++  |
| `batch.n_tokens = 0`                                 | `llama_batch_ext_clear(batch)`                                  |
| `batch.token[i] = id`                                | `llama_batch_ext_add_token(batch, seq_id, id)`                  |
| `batch.embd + i*n_embd`                              | `llama_batch_ext_add_embd(batch, seq_id, embd)`                 |
| `batch.pos[i] = pos`                                 | `llama_batch_ext_set_pos(batch, idx, &pos)`                     |
| `batch.seq_id[i][j] = seq_id`                        | `llama_batch_ext_add_seq(batch, idx, seq_id)`                   |
| `batch.logits[i] = true`                             | `llama_batch_ext_set_output_logits(batch, idx, true)`           |
| `llama_encode(ctx, batch)`                           | `llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, batch)`          |
| `llama_decode(ctx, batch)`                           | `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)`          |

Things to keep in mind:

- The batch is tied to a context. Its capacity is `n_batch` of that context, and token IDs, embedding sizes and sequence IDs are checked against it. Do not reuse a batch with another context.
- There are no default values. Each entry must have its position set with `llama_batch_ext_set_pos()`. Positions are no longer tracked automatically.
- No entry is an output by default. Set at least the entries you want logits or embeddings for with `llama_batch_ext_set_output_logits()` (or `llama_batch_ext_set_output_embd()`, they are equivalent for now).
- The `llama_batch_ext_add_*()` functions return the batch index `idx` of the new entry. Use `idx` for the `_set_*()` functions, and for `llama_get_logits_ith()` / `llama_get_embeddings_ith()` after processing.
- The batch copies the token IDs and embeddings. You can free your own buffers after the `add` call.
- `llama_process()` returns the same values as the old `llama_encode()` / `llama_decode()`.

## Examples

### 1. Single sequence (replace `llama_batch_get_one`)

Old code:

```cpp
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
if (llama_decode(ctx, batch)) {
    // error
}
```

New code:

```cpp
llama_batch_ext * batch = llama_batch_ext_init(ctx);

// continue from the last position in the memory (this is what llama_batch_get_one did)
const llama_pos pos_0 = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

for (int32_t i = 0; i < (int32_t) tokens.size(); i++) {
    const int32_t   idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
    const llama_pos pos = pos_0 + i;
    llama_batch_ext_set_pos(batch, idx, &pos);
}
llama_batch_ext_set_output_logits(batch, tokens.size() - 1, true);

if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
    // error
}

llama_batch_ext_free(batch);
```

If you already know the positions (for example, you track `n_past` yourself), use them directly instead of reading the memory.

### 2. Multiple sequences (replace `llama_batch_init`)

Old code:

```cpp
llama_batch batch = llama_batch_init(n_batch, 0, n_seq_max);

batch.n_tokens = 0;
for (...) {
    const int i = batch.n_tokens;
    batch.token   [i]    = id;
    batch.pos     [i]    = pos;
    batch.n_seq_id[i]    = 1;
    batch.seq_id  [i][0] = seq_id;
    batch.logits  [i]    = output;
    batch.n_tokens++;
}

llama_decode(ctx, batch);
float * logits = llama_get_logits_ith(ctx, i_out);

llama_batch_free(batch);
```

New code:

```cpp
llama_batch_ext * batch = llama_batch_ext_init(ctx);

llama_batch_ext_clear(batch);
for (...) {
    const int32_t idx = llama_batch_ext_add_token(batch, seq_id, id);
    if (idx < 0) {
        // -1: batch is full, -2: invalid token, -3: invalid seq_id
    }
    llama_batch_ext_set_pos(batch, idx, &pos);
    llama_batch_ext_set_output_logits(batch, idx, output);
}

llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
float * logits = llama_get_logits_ith(ctx, i_out); // i_out is the idx returned by add

llama_batch_ext_free(batch);
```

To share one entry between several sequences (old `n_seq_id[i] > 1`), add it once, then call `llama_batch_ext_add_seq()` for each extra sequence. Call it before the other `_set_*()` functions:

```cpp
const int32_t idx = llama_batch_ext_add_token(batch, seq_ids[0], id);
for (size_t s = 1; s < seq_ids.size(); s++) {
    llama_batch_ext_add_seq(batch, idx, seq_ids[s]);
}
llama_batch_ext_set_pos(batch, idx, &pos);
```

### 3. Embedding input

Old code used `llama_batch_init(n_tokens, n_embd, 1)` and filled `batch.embd`. Now each entry gets its own `llama_embd`:

```cpp
for (int32_t i = 0; i < n_tokens; i++) {
    llama_embd e = {
        /* data   */ embd + (size_t) i * n_embd,
        /* n_rows */ 1,
        /* n_embd */ (size_t) n_embd,
    };
    const int32_t   idx = llama_batch_ext_add_embd(batch, seq_id, e);
    const llama_pos pos = n_past + i;
    llama_batch_ext_set_pos(batch, idx, &pos);
}
```

`n_rows * n_embd` must match the input embedding size of the model, otherwise the call returns `-2`.

For M-RoPE models (`llama_model_rope_type()` returns `LLAMA_ROPE_TYPE_MROPE` or `LLAMA_ROPE_TYPE_IMROPE`), each embedding entry takes 4 positions in order `[t, y, x, z]`. The old API used a section-major layout (`pos[j*n_tokens + i]`); the new API takes the 4 positions of one entry at a time:

```cpp
const llama_pos p[4] = { t, y, x, z }; // t is also the KV cache position
llama_batch_ext_set_pos(batch, idx, p);
```

Text tokens always take a single position, also for M-RoPE models.

If an entry needs both a token ID and an embedding, use `llama_batch_ext_add_token()` then `llama_batch_ext_set_embd_token()`.

### 4. Encoder-decoder models

Replace `llama_encode()` with `LLAMA_PROCESS_TYPE_ENCODE`. The same batch object can be reused for the decoder after `llama_batch_ext_clear()`:

```cpp
llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, batch);

llama_batch_ext_clear(batch);
const llama_pos pos = 0;
const int32_t   idx = llama_batch_ext_add_token(batch, 0, decoder_start_token_id);
llama_batch_ext_set_pos(batch, idx, &pos);
llama_batch_ext_set_output_logits(batch, idx, true);

llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
```

### 5. Splitting a large batch

With the old API, a large batch was often split by creating views with pointer offsets (`batch.token + i`, `batch.pos + i`, ...). This is no longer possible. Fill a new batch for each chunk of at most `llama_n_batch(ctx)` entries instead.

## C++ helpers

In C++, use `llama_batch_ext_ptr` from [llama-cpp.h](/include/llama-cpp.h) to free the batch automatically:

```cpp
llama_batch_ext_ptr batch(llama_batch_ext_init(ctx));
llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
```

Code inside this repository that links `common` can use `common_batch` (see [common/common.h](/common/common.h)). It keeps a readable copy of the entries, supports more entries than `n_batch` via `get_sub_batch()`, and `common_batch_get_one()` replaces `llama_batch_get_one()`.

## Language bindings

Bindings that mirrored the `llama_batch` struct layout (for example Swift or Python `ctypes`) must switch to the function-based API. `llama_batch_ext` is an opaque pointer, its layout is not part of the public API.

# llama.cpp/example/paged

Minimal end-to-end demo of the paged KV cache and continuous-batching scheduler. Queues N requests up front, then runs a loop of `llama_paged_scheduler_prepare_batch` → `llama_decode` → `llama_paged_scheduler_update` until every request finishes.

## Example

Run 10 sequences in parallel with greedy sampling, up to 50 generated tokens per sequence on a single GPU:

```bash
llama-paged -m model.gguf -kvp -ngl 100 -sm none -mg 0 \
            -ns 10 -np 10 -n 50 -b 512 -ub 512 \
            -ngpub 500 -ncpub 100
```

- `-kvp` / `--kv-paged`: enable the paged KV cache (required)
- `-ngpub N`: number of GPU KV blocks to allocate (auto-fitted from free VRAM if omitted)
- `-ncpub N`: number of CPU KV blocks for swap-out (optional)
- `-ns N`: number of sequences to queue
- `-np N`: maximum sequences decoded in parallel (must equal `-ns` in this example)

## Phase 1 restrictions
The paged path currently requires:
- Single device (fully loaded on one GPU or CPU). Use `-sm none -mg <id>` to pin to one GPU on multi-GPU machines.
- Full offload of the model (`-ngl` must cover all layers).
- `-b == -ub` (batch and ubatch sizes must match).

Note: SWA architectures (gemma3, llama4, etc.) are not yet supported.

## Output
- Prints all the generated outputs at the end.
- Prints performance timings (general and per-request).

# llama.cpp/examples/verifiable-inference

This example demonstrates (a simplified version of) the **commit-and-open** idea from `paper/2026-541.pdf`:

- The prover runs inference and **commits** to an execution trace by hashing each GGML node output tensor and building a **Merkle tree** over those hashes.
- A verifier challenge is derived from the commitment (Fiat–Shamir style), and the prover re-runs inference to **open** a small number of randomly sampled trace entries, providing:
  - the tensor bytes,
  - the leaf hash,
  - and a Merkle inclusion path to the root.

This is intended as a practical integration point for llama.cpp’s existing `cb_eval` trace callback; it is **not** a full cryptographic proof of correct inference.

## Build

From the repo root:

```sh
cmake -S . -B build
cmake --build build -j
```

## Run

```sh
./build/bin/llama-verifiable-inference \
  --hf-repo ddh0/GPT-2-GGUF \
  --hf-file GPT-2-q4_K_S.gguf \
  -ngl 0 \
  -p "Once upon a time" \
  --vi-samples 16 \
  --vi-out vi-out
```

Outputs:

- `vi-out/commit.json`: commitment root + trace metadata
- `vi-out/openings.json`: sampled openings + Merkle paths
- `vi-out/openings/*.bin`: raw tensor bytes for each opened trace entry

## Notes / caveats

- To keep the example self-contained, the “trace” is defined as the sequence of GGML node output tensors observed via the `llama_context_params.cb_eval` callback.
- If you want smaller traces, use `--vi-tensor-filter` to restrict the tensor names (regex, can be repeated).
- Exact node ordering is backend/scheduler dependent; the example uses the callback order as the trace order and then replays it for openings.

# Laguna-S-2.1 on RDNA2 tensor parallelism

Laguna-S-2.1 builds a deeper tensor-parallel graph than the models originally used to validate this fork. With the affected Meta-backend implementation, the command can load all model tensors and then segfault immediately after reserving the approximately 4,807-node compute graph.

## Diagnosis

The failure is host-side process-stack exhaustion, not a GPU fault and not the RDNA2 typical-MoE MMQ picker.

`ggml_backend_meta_get_split_state()` and lazy per-device tensor-shard initialization previously followed graph dependencies recursively. Their frames contain large `ggml_backend_meta_split_state` values. Laguna's graph can exceed the normal 8 MiB Linux stack while resolving those dependencies.

The `ggml-meta: avoid deep graph stack exhaustion` change converts both dependency walks to iterative post-order and adds a 2,048-node regression in `test-meta-split`. It does not alter tensor-split results or model math.

## Immediate workaround for older images

Give the container a 64 MiB process stack. This is a deployment workaround for binaries that do not contain the source fix.

Docker Compose:

```yaml
services:
  Laguna S 2.1:
    ulimits:
      stack:
        soft: 67108864
        hard: 67108864
```

Equivalent `docker run` option:

```bash
--ulimit stack=67108864:67108864
```

A shell entrypoint can instead run `ulimit -s 65536` before `llama-server`, provided the container's hard stack limit permits it.

## Fixed-build verification

From a clean shell on the four-V620 test host:

```bash
cd /path/to/llama.cpp-rdna2
./scripts/verify-laguna-tensor-split.sh
```

The verifier deliberately enforces the ordinary 8 MiB stack, runs the Meta/tensor-split tests, starts the exact reported Laguna server configuration, waits for `/health`, submits a deterministic eight-token request, and stops the server. Success ends with:

```text
LAGUNA_TENSOR_SPLIT_VERIFY_OK
```

## About commit `52ad29f`

`52ad29f` did change RDNA2's `use_typical_moe_ncols` setting from false to true, and Laguna uses 256 routed experts with top-10 selection. However, that picker is used when dispatching routed quantized matrix multiplication. The observed crash happens during host-side Meta graph setup, before a request dispatches the routed MMQ path. The unchanged picker also completes Laguna inference with either the stack workaround or the source fix. It should therefore be evaluated separately as a performance/correctness experiment, not reverted as a fix for this load crash.

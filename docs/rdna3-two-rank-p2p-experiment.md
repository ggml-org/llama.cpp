# Experimental two-rank native P2P AllReduce

This document records the guarded two-rank tensor-parallel experiment for
qualified native RDNA2/gfx1030 and RDNA3/gfx1100 topologies. It is not enabled
by the launcher or by either architecture's Auto profile.

## Design

The opt-in path is implemented in `ggml/src/ggml-cuda/ggml-cuda.cu` and is
available only in HIP+Linux+RCCL builds after all of these checks pass:

- exactly two selected, non-aliased CUDA/HIP backends;
- both selected GPUs pass the exact architecture/device identity and
  bidirectional peer-access guard (gfx1030 requires the exact architecture;
  gfx1100 additionally requires RX 7900 XT);
- RCCL communicators initialize successfully;
- each supported width passes a startup FP32 self-test against RCCL for four
  input patterns.

The candidate uses portable mapped-host snapshots and separate start/end phase
flags. Snapshot storage has eight slots and the dispatch requires both ranks to
use the same backend stream; unsupported shape, topology, or stream cases
return to RCCL. The ordinary candidate supports hidden-state widths 1 through 6
(`[5120,width,1,1]` F32). Each width is qualified independently, so a width
that does not match the installed RCCL behavior falls back without disabling
other widths.

Direct remote-device pointer reads were tested in a standalone microbenchmark,
but failed exact validation against real model allocations. That route was not
retained; the model-tested candidate uses mapped snapshots instead.

## Activation

These are supervised A/B controls only:

```bash
GGML_HIP_P2P_ALLREDUCE=1 \
  GGML_CUDA_ALLREDUCE=nccl \
  scripts/run-qwen38-rdna-unified.sh
```

The earlier `GGML_HIP_RDNA3_P2P_ALLREDUCE` name remains accepted as an
alias. The two-rank path is ordinary AllReduce only; the existing four-rank
consumer-fused path remains separately gated. The switch remains unset in
normal production launches. No NCCL algorithm, protocol, channel, or topology
override is implied.

## Measurements

Matched Qwen3.8-27B Q4_0 two-GPU tensor-mode requests used the verified model,
F16 KV, the same prompt/seed, and exact response hashing. Results below are
preliminary screening measurements on the supervised 180 W/`-50mV` validation
boot; they are not a production performance claim:

| Workload | RCCL control | Two-rank snapshot | Output |
|---|---:|---:|---|
| Target-only, 128 tokens, graphs disabled | 38.16/38.61 tok/s | 38.85/39.35 tok/s | identical response hash |
| MTP sidecar, 128 tokens, graphs disabled | 59.86/83.10 tok/s | 69.10/78.02 tok/s | identical response hash |

The MTP run exercised widths 2, 3, and 4, with all startup self-tests passing.
Acceptance varied between the short runs (`0.88462` RCCL versus `0.85047`
snapshot), so longer stock-boot A/B testing is required before any claim.

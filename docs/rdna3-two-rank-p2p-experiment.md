# Experimental two-rank RDNA3 AllReduce

This document records the guarded two-rank tensor-parallel experiment for the
native RX 7900 XT/gfx1100 topology. It is not enabled by the launcher or by
`GGML_HIP_RDNA3_AUTO`.

## Design

The opt-in path is implemented in `ggml/src/ggml-cuda/ggml-cuda.cu` and is
available only in HIP+Linux+RCCL builds after all of these checks pass:

- exactly two CUDA/HIP backends in logical order `0,1`;
- every visible physical GPU passes the existing native RX 7900 XT/gfx1100
  identity and bidirectional peer-access guard;
- RCCL communicators initialize successfully;
- a startup FP32 self-test matches RCCL bit-for-bit for four input patterns at
  5,120 elements.

The candidate uses portable mapped-host snapshots and separate start/end phase
flags. Snapshot storage has eight slots and the dispatch requires both ranks to
use the same backend stream; unsupported shape, topology, or stream cases
return to RCCL. The ordinary candidate is restricted to the hot
`[5120,1,1,1]` F32 boundary. A separate consumer-fused switch covers the
`reshape -> add -> RMSNorm -> mul` prefix for the same shape.

Direct remote-device pointer reads were tested in a standalone microbenchmark,
but failed exact validation against real model allocations. That route was not
retained; the model-tested candidate uses mapped snapshots instead.

## Activation

These are supervised A/B controls only:

```bash
GGML_HIP_RDNA3_P2P_ALLREDUCE=1 \
  GGML_HIP_RDNA3_AUTO=1 \
  scripts/run-qwen38-rdna-unified.sh
```

The consumer-fused variant is selected independently:

```bash
GGML_HIP_RDNA3_P2P_FUSED_ALLREDUCE=1 \
  GGML_HIP_RDNA3_AUTO=1 \
  scripts/run-qwen38-rdna-unified.sh
```

Both switches remain unset in normal production launches. No NCCL algorithm,
protocol, channel, or topology override is implied.

## Measurements

Matched Qwen3.8-27B Q4_0 two-GPU tensor-mode requests used the verified model,
F16 KV, the same prompt/seed, and exact response hashing. Results below are
screening measurements on the supervised 180 W/`-50mV` validation boot:

| Workload | RCCL control | Ordinary snapshot | Consumer-fused snapshot | Output |
|---|---:|---:|---:|---|
| Target-only, 1,024 tokens, graphs enabled | 39.70 tok/s warm | 40.51 tok/s warm | 39.91 tok/s warm | byte-identical |
| MTP sidecar, 512 tokens, graphs enabled | 91.24 tok/s warm | no candidate calls (verification width >1 falls back) | no candidate calls | candidate path not exercised |

The ordinary candidate's target-only improvement is small and requires more
repeated production-policy measurements before it can be considered a default.
MTP width-3/4 verification deliberately remains on RCCL; the experiment did
not claim an MTP gain.

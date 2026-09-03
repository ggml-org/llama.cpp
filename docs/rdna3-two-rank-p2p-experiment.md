# Experimental two-rank native P2P AllReduce

This document records the guarded two-rank tensor-parallel candidate for
qualified native RDNA2/gfx1030 and RDNA3/gfx1100 topologies. Qualified gfx1030
TP2 follows the existing `GGML_HIP_RDNA2_AUTO` policy. On gfx1100,
`GGML_HIP_RDNA3_AUTO=1` retains the validated RCCL/direct-P2P policy; the
preliminary host-snapshot candidate remains explicit because matched testing
found it slower than RCCL. RCCL remains the fallback.

## Design

The path is implemented in `ggml/src/ggml-cuda/ggml-cuda.cu` and is
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
return to RCCL. RDNA2 Auto qualifies only width-one decode. Explicit mode
supports hidden-state widths 1 through 6 (`[5120,width,1,1]` F32), qualifying
each independently so a rejected width does not disable the others.

Direct remote-device pointer reads were tested in a standalone microbenchmark,
but failed exact validation against real model allocations. That route was not
retained; the model-tested candidate uses mapped snapshots instead.

## Activation

On a qualified gfx1030 TP2 topology, the existing RDNA2 Auto policy arms the
candidate unless disabled:

```bash
GGML_HIP_RDNA2_AUTO=1 GGML_CUDA_ALLREDUCE=nccl \
  scripts/run-qwen38-rdna-unified.sh
```

On a qualified gfx1100 TP2 topology, opt in to the validated RDNA3 Auto
profile. This intentionally leaves the preliminary host-snapshot candidate
on RCCL/direct-P2P:

```bash
GGML_HIP_RDNA3_AUTO=1 GGML_CUDA_ALLREDUCE=nccl \
  scripts/run-qwen38-rdna-unified.sh
```

`GGML_HIP_P2P_ALLREDUCE=1` remains a supervised explicit override for the
host-snapshot candidate on either architecture. `GGML_HIP_P2P_ALLREDUCE=0`
suppresses it. With the explicit enable, `GGML_HIP_RDNA3_P2P_CHUNKED=1`
enables the guarded two-block protocol for wider rows. The earlier
`GGML_HIP_RDNA3_P2P_ALLREDUCE` name remains an alias. The two-rank path is
ordinary AllReduce only; the existing four-rank
consumer-fused path remains separately gated. No NCCL algorithm, protocol,
channel, or topology override is implied.

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
snapshot). On the later MQ-IQ4_XS_1 workload, disabling the host-snapshot
candidate improved the warm result from `91.70` to `96.97` tok/s, so the
candidate remains explicit on gfx1100 pending a different implementation.
On V620/gfx1030, repeated target-only decode improved `38.44 → 40.40` tok/s,
while speculative multi-row one-block and two-block paths remained below RCCL.
Consequently RDNA2 Auto uses width one only; wider rows remain explicit.

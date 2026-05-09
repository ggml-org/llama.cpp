# B60 SYCL Performance Observations

Working notes from benchmarking llama.cpp SYCL backend on Intel Arc Pro B60 (24 GB),
branch `fa-overhead-sycl` at commit 9dd668777. All numbers from `llama-bench`,
power from `xpu-smi dump -d 0 -m 1,2,18 -i 1000` (1 s sampling).

## Headline measurements

| Workload | Format | pp t/s | tg t/s | Peak power | Sustained power |
|---|---|---|---|---|---|
| Qwen3 35B-A3B MoE | Q4_K_M | 349 (pp512) | 48.6 | 98 W | ~95-98 W |
| Qwen3 4B dense | Q4_K_M | 1129 (pp512) | 65.7 | 123 W | ~120-123 W |
| Qwen2.5 7B dense | F16 | 631 (pp4096) | 24.9 | 143 W | ~135-143 W |
| Qwen2.5 7B dense, `-fa 1` | F16 | 539 (pp4096) | 26.3 | ~140 W | ~135-143 W |

Idle: 35 W at 400-1550 MHz. Boost clock during compute: 2400 MHz throughout.

## Pattern: power scales with compute density, not "GPU activity"

Each step up in arithmetic-per-byte adds ~25 W to the sustained ceiling:

- MoE Q4: scattered routing, small per-expert matmuls, dequant on critical path -> 98 W
- Dense Q4: dequant still on critical path, but matmuls are larger -> 123 W
- Dense F16: no dequant, matmuls go straight to XMX -> 143 W

The B60 is rated 200 W. We do not approach it. The gap is not "the GPU is idle";
the gap is "the GPU is doing the wrong kind of work." Memory-bound passes
(dequant kernels, decode-phase matmuls, scattered MoE access) draw less power
than compute-bound passes regardless of how busy the queues look.

## Why FA is currently a regression on SYCL prefill

`-fa 1` on F16 prefill: -15 % throughput, no power change. On decode: +6 %,
also no power change. Attention is not the bottleneck preventing higher
utilization on prefill - the FFN matmuls dominate. The `fa-overhead-sycl` branch
is the place where this is being investigated; the data above just confirms
that FA-on does not currently buy time or watts on prefill, only a small
decode improvement consistent with reduced KV-cache traffic.

## The 200 W spec is peak/burst, not sustained

Practical sustained ceiling on this card for compute-bound F16 prefill is
about 145 W with brief touches to ~150 W. There is no workload short of
multi-stream concurrent matmuls that will pull the chip to its rated 200 W in
single-card inference.

## Multi-stream infrastructure exists but is a stub

`ggml/src/ggml-sycl/common.hpp:329-334` allocates 8 stream slots per device
(`GGML_SYCL_MAX_STREAMS = 8`) but every slot resolves to the same
`default_queue()`, which is the in-order queue. All "streams" alias to one
serializing queue. The CUDA backend at `ggml/src/ggml-cuda/common.cuh:1436-1442`
has the same shape but actually creates a fresh non-blocking stream per slot.

Consequence: the split-mode multi-stream code in `ggml_sycl_op_mul_mat`
(around `ggml-sycl.cpp:2795`) submits work to a `is` index that has no effect
on a single device. There is no intra-device kernel concurrency today.

## ROI ranking for closing the power gap

1. **Fused INT4/FP8 matmul kernels.** Removes the dequant pass, feeds XMX
   directly. Highest single win because dequant is memory-bound dead weight
   that cannot use the matrix engines. `vllm-xpu-kernels/` (already cloned in
   this repo) does this for Intel GPUs and is the obvious source.
2. **MoE expert parallelism via real per-device streams.** Only matters for
   MoE, but MoE is where the floor was 98 W. Experts within a layer are
   independent by construction.
3. **Wire real per-slot queues** (replace `default_queue()` aliasing with
   `create_in_order_queue()` per slot, store ownership). Foundation for (2).
   Mechanical change, no algorithmic risk.
4. **Out-of-order queue + event DAG experiment.** Cheap to try, unknown how
   aggressive L0 is at scheduling. May or may not match hand-rolled per-slot
   scheduling.

Compute/copy overlap and CUDA-graph-style replay are second-order on
single-card inference.

## Sampling caveat

xpu-smi 5 s buckets undersample short prefills. 1 s sampling shows the
sustained band sits remarkably flat (130-143 W on F16, very little
kernel-to-kernel variance). The chip is not "idle then busy" - it is at a
ceiling, continuously.

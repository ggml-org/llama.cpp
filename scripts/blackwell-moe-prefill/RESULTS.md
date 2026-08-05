# GPT-OSS-120B prefill on Blackwell

On one RTX PRO 6000 Blackwell, the bitwise-compatible path improves
GPT-OSS-120B MXFP4 prefill from 9,168 to 17,708 tok/s at pp8192. This is a
1.93x speedup over the existing llama.cpp path. A faster experimental
Attention path reaches 23,301 tok/s, or 2.54x, but changes the logits and is
included only as a performance ceiling.

The same-machine vLLM result was about 36.7k tok/s. The bitwise-compatible
path is still 2.07x slower than vLLM, while the ceiling is 1.58x slower.

A later direct CUTLASS run reached 24,763 tok/s. Its W13 and W2 grouped GEMMs
took 111.73 ms, compared with 105.07 ms in vLLM. A follow-up replaced the three
slow support stages and reached 25,714 tok/s. The optimized support path is
bitwise identical to the earlier CUTLASS support path.

These are direct prefill measurements, not serving throughput. I did not
compare them directly with the llama-benchy number because llama-benchy
estimates prefill from TTFR after subtracting request and first-token latency.

## Test setup

- GPU: RTX PRO 6000 Blackwell Server Edition, 96 GB, SM120
- llama.cpp baseline: `c745be2`
- Model: GPT-OSS-120B MXFP4 with fused W13
- Run: pp8192, batch 8192, 25 CPU threads, FlashAttention
- Measurement: three timed repetitions, with each mode in a separate process
- Earlier vLLM run: `0.1.dev1+g045293d82`, PyTorch 2.12.1+cu130, FlashInfer
  0.6.1, using the FlashInfer CUTLASS FP8xFP4 path

## End-to-end performance

This table uses the best measured ubatch for each path. The two CUTLASS results
are shown next to the earlier native experiments and the vLLM reference.

| Version | Best ubatch | Prefill | Speedup over existing best | Share of vLLM | Numerical comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| Existing llama.cpp | 2048 | 11,738.58 tok/s | 1.00x | 31.9% | Reference |
| Canonical persistent | 2048 | 14,875 tok/s | 1.27x | 40.4% | Bitwise with existing |
| Strict native CUDA | 8192 | 17,708 tok/s | 1.51x | 48.1% | Bitwise with existing |
| Native CUDA full ceiling | 8192 | 23,301 tok/s | 1.98x | 63.3% | Relaxed Attention |
| Direct CUTLASS | 8192 | 24,762.74 tok/s | 2.11x | 67.3% | CUTLASS numerical ceiling |
| Optimized CUTLASS support | 8192 | 25,713.80 tok/s | 2.19x | 69.8% | Bitwise with direct CUTLASS |
| vLLM FlashInfer CUTLASS | 8192 | 36,819 tok/s | 3.14x | 100% | External reference |

The earlier native sweep also recorded the ubatch sensitivity:

| Mode | ubatch 2048 | ubatch 8192 |
| --- | ---: | ---: |
| Existing path | 11,722 tok/s | 9,168 tok/s |
| Canonical persistent | 14,875 tok/s | 11,708 tok/s |
| Strict tuned | 15,092 tok/s | 17,708 tok/s |
| Full ceiling | 15,317 tok/s | 23,301 tok/s |

The larger ubatch hurts the existing and canonical paths, but helps the TMA
versions substantially. I have not isolated this completely. The current
evidence suggests that the TMA scheduler needs the larger microbatch to expose
enough grouped-MMQ work, while the generic path loses efficiency at that size.

The ceiling gains another 5,593 tok/s over the strict path. Nsys attributes
nearly all of that difference to Attention and RoPE rather than another MoE
change.

## Direct CUTLASS follow-up

The direct CUTLASS experiment uses MXFP8 activations, MXFP4 expert weights,
BF16 grouped-GEMM outputs, and the SM120 block-scaled tensor core path. It is a
separate numerical ceiling, not the bitwise-compatible native path above. The
measured pp8192 times were 330.820 ms for direct CUTLASS, 318.587 ms after the
support-kernel changes, and 222.494 ms for vLLM. The existing llama.cpp control
used its best ubatch and took 697.870 ms. The same CUTLASS run repeated the
native CUDA ceiling at 377.154 ms and 21,720.56 tok/s; the combined table uses
the earlier 23,301 tok/s best result for that path.

The optimized path reaches 69.8% of the vLLM prefill throughput and is 1.43x
slower end to end. The CUTLASS code only owns W13 and W2. The other stages are
CUDA kernels integrated with the existing llama.cpp graph and memory pool.

The latest two-pass Nsys capture gives the following steady-state split. The
one-time weight transforms are excluded.

| MoE component | Native CUDA ceiling | Direct CUTLASS | vLLM CUTLASS |
| --- | ---: | ---: | ---: |
| Input activation quantization | 4.852 ms | 16.338 ms | about 4.01 ms |
| Expert scheduling | 7.891 ms | 10.501 ms | about 0.93 ms |
| W13 plus W2 | 235.519 ms | 111.730 ms | 105.071 ms |
| W13 activation and A2 quantization | included above | 57.187 ms | 13.409 ms |
| W2 finalization | 11.715 ms | 8.837 ms | 5.863 ms |
| MoE steady total | about 259.98 ms | 204.59 ms | about 129.28 ms |

Direct CUTLASS spends 73.813 ms in W13 and 37.917 ms in W2. The combined GEMM
gap against vLLM is only 6.66 ms. The direct path also transforms the expert
weights once on first use; this took 378.205 ms in the trace and is not part of
the steady total.

The vLLM support-kernel values are grouped by kernel name. Its trace does not
contain matching NVTX ranges, so the scheduling and input figures are an
approximate classification of the prefix-sum, stride, expansion, and block-
quantization kernels.

### Optimized CUTLASS support follow-up

The follow-up replaces the per-expert scan with a histogram and prefix sum,
quantizes one input token per CTA, and processes one routed row per CTA in the
W13 activation stage. It sets
`GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS=1`. A same-build, unprofiled A/B run
measured:

| Support path | Time | Prefill |
| --- | ---: | ---: |
| Earlier CUTLASS support | 379.120 ms | 21,608.22 tok/s |
| Optimized CUTLASS support | 318.587 ms | 25,713.80 tok/s |

This is a 19.0% throughput improvement. The complete logits comparison covered
201,088 values and was bitwise identical, with NMSE 0 and max absolute error 0.

The corrected two-pass Nsys run used one pp8192 warmup followed by one measured
pass. The stage ranges and kernel totals below are normalized to one 36-layer
pass. The one-time weight transform is excluded.

| Component | Earlier range | Optimized range | Optimized kernels | vLLM named kernels |
| --- | ---: | ---: | ---: | ---: |
| Expert scheduling | 8.649 ms | 6.578 ms | 0.345 ms | about 0.924 ms |
| Input expansion and MXFP8 quantization | 15.588 ms | 9.526 ms | 3.945 ms | about 4.012 ms |
| W13 plus W2 | 127.514 ms | 127.863 ms | 127.863 ms | 105.041 ms |
| W13 activation and A2 quantization | 58.301 ms | 12.973 ms | 12.129 ms | 13.384 ms |
| W2 finalization | 8.767 ms | 8.785 ms | 8.785 ms | 5.856 ms |
| MoE steady total | 218.819 ms | 165.725 ms | - | about 129.217 ms |

The scheduling kernel itself is 25.1x faster, input quantization is 3.62x
faster, and the W13 activation kernel is 4.68x faster. The scheduling range is
still wider because it contains three launches and the gaps between them.

Nsys adds substantial overhead to the full process. Its cumulative ablation is
useful for attribution, not for the acceptance throughput above:

| Nsys case | Time | Prefill |
| --- | ---: | ---: |
| Earlier support | 493.273 ms | 16,607.4 tok/s |
| Prefix scheduling | 488.516 ms | 16,769.2 tok/s |
| Prefix plus CTA input quantization | 477.747 ms | 17,147.2 tok/s |
| Complete optimized support | 433.665 ms | 18,890.1 tok/s |

The complete support path removes 53.094 ms from the steady MoE range. The
remaining 36.5 ms against the vLLM MoE split is about 22.8 ms in the current
W13/W2 capture, 11.2 ms in scheduling and input range overhead, and 2.9 ms in
W2 finalization. The earlier best W13/W2 capture was within 6.66 ms of vLLM, so
the grouped-GEMM result is sensitive to the selected tiles and run conditions.

## Implemented versions

The final comparisons all use the same fused-W13 GGUF. Disabling the
experimental launcher restores the original MoE graph and generic MMQ path;
it does not change the model weights.

The strict path addresses three groups of overhead.

**MoE data movement and scheduling.** W13 and W2 share the expert permutation
and the broadcast activation is quantized once. Expert weights are transformed
in place for TMA loads, avoiding a second packed copy of the roughly 65 GB of
expert weights. W13 uses a 64-row tile and W2 is scheduled output-major.

**Intermediate tensors and launches.** W13 bias, SwiGLU, and the second
activation quantization are folded into the W13 stage. W2 bias, routing
weights, and expert reduction are folded into W2.

**Work outside MoE.** The strict path also includes direct causal Attention,
which avoids materializing the KQ mask, and an add-plus-RMSNorm kernel.

Unsupported devices, shapes, old separate gate/up GGUF files, and small-token
decode workloads fall back to the existing path.

The full-ceiling version adds Q RoPE fusion and a separate SM120 causal
Attention schedule. It reduces Attention launches and avoids mask reads. The
floating-point evaluation order is different, so this version is useful for
measuring the remaining performance headroom but not for a bitwise-compatible
change.

### Development history

Before the TMA work, I measured graph fusion and the canonical persistent
scheduler separately:

| Version | pp512 | pp2048 | pp8192 |
| --- | ---: | ---: | ---: |
| Existing path | 8,429 tok/s | 10,947 tok/s | 11,640 tok/s |
| Generic fused | 8,791 tok/s | 12,130 tok/s | 13,040 tok/s |
| Canonical persistent | 9,442 tok/s | 13,596 tok/s | 14,725 tok/s |

Graph fusion improved pp8192 by 12.0%. The canonical scheduler raised the total
gain to 26.5% while still reading the original 17-byte MXFP4 GGUF blocks.

## Correctness

- Add-plus-RMSNorm backend tests: 3/3 passed.
- MoE backend tests: 5/5 passed. These cover contiguous and strided expert
  IDs, views-first graphs, uniform routing, and highly skewed routing.
- Strict pp8192 logits: all 201,088 values are bitwise identical to the
  existing path, with NMSE 0 and max absolute error 0.
- Optimized CUTLASS support logits: all 201,088 values are bitwise identical
  to the earlier CUTLASS support path, with NMSE 0 and max absolute error 0.

The full-ceiling path has measurable numerical drift:

| Prompt | NMSE | Max abs | Mean abs | RMSE |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 0.0037914378 | 0.78913784 | 0.1222908 | 0.1490868 |
| 8192 | 0.004538735 | 0.6545565 | 0.1366402 | 0.1644045 |

## Native CUDA and Attention Nsys profile

This earlier profile compares the native CUDA and Attention variants. The
llama.cpp captures contain one cold pp8192 pass. The vLLM
column comes from the earlier same-GPU pp8192 trace using the FlashInfer
CUTLASS FP8xFP4 backend. Times are summed GPU kernel time in milliseconds.
`All steady kernels` excludes the one-time llama.cpp weight transform.

| Component | Existing path | Strict | Full ceiling | vLLM CUTLASS |
| --- | ---: | ---: | ---: | ---: |
| W13 | 179.391 | 133.345 | 134.475 | not separable |
| W2 | 87.487 | 58.123 | 58.810 | not separable |
| W13 plus W2 | 266.878 | 191.468 | 193.285 | 105.041 |
| Activation quantization | 22.397 | 6.436 | 6.414 | 0.551 |
| Expert routing | 14.978 | 10.291 | 10.297 | 4.385 |
| MoE epilogue | 82.427 | 11.828 | 11.863 | 19.240 |
| MoE subtotal shown | 386.680 | 220.023 | 221.859 | 129.217 |
| Attention | 141.448 | 145.785 | 50.520 | included below |
| RoPE | 22.527 | 23.758 | 2.648 | included below |
| Attention plus RoPE/fixup | 163.975 | 169.543 | 53.168 | 32.422 |
| One-time TMA repack | 0 | 357.405 | 356.601 | done at load time |
| All steady kernels | 681.936 | 486.844 | 372.599 | 215.030 |

W13 plus W2 falls from 266.878 ms to 191.468 ms in the strict path, but vLLM
still needs only 105.041 ms. The activation, routing, and epilogue work falls
from 119.802 ms to 28.555 ms, close to vLLM's 24.176 ms. In the ceiling run,
Attention plus RoPE falls from 163.975 ms to 53.168 ms; vLLM takes 32.422 ms.

The in-place expert-weight transform is lazy and runs at first use. It launches
72 kernels and takes 357.405 ms in the strict run and 356.601 ms in the ceiling
run. The corresponding NVTX spans are 533.7 and 532.6 ms. Warmed acceptance
benchmarks hide this cost, but the first request pays it. A production version
would need to move the transform to model load or an explicit warmup step.

<details>
<summary>Earlier canonical persistent Nsys breakdown</summary>

The generic and canonical versions were not repeated in the final comparison.
An earlier profile measured:

| MoE component | Existing path | Canonical persistent |
| --- | ---: | ---: |
| Complete MoE pipeline | about 487 ms | about 332 ms |
| W13 plus W2 MMQ | 354 ms | 275 ms |
| Activation quantization | 21.1 ms | 14.5 ms |
| Expert permutation | 15.2 ms | 5.6 ms |
| Bias, SwiGLU, weight, and reduce | 96.9 ms | 36.2 ms |

</details>

The matching vLLM benchmark took 222.494 ms, or 36,819 input tok/s. Its trace
contains 787 CUDA kernel launches. The 72 CUTLASS grouped-GEMM launches are W13
and W2 for all 36 layers, but Nsys reports both under the same kernel name and
does not provide a reliable W13/W2 split. A matching Marlin trace spent 213.117
ms in those 72 expert GEMMs, roughly twice the CUTLASS time.

The vLLM support-kernel split is 13.384 ms for activation, 5.856 ms for final
routing, 3.461 ms for input expansion, 0.551 ms for input block quantization,
and 0.924 ms for prefix sums, TMA strides, and top-k. Dense projection GEMMs
take about 40.0 ms across 108 launches. Earlier notes quoted about 106.7 ms for
the grouped GEMMs and 32.8 ms for Attention after broader grouping; the table
uses the raw named-kernel totals of 105.041 and 32.422 ms.

## Remaining gap

The three support-kernel experiments reached their compute targets. Input
quantization and the W13 activation are now close to the matching vLLM named
kernels, and the scheduling kernel is faster. The next MoE work is narrower:
reduce launch gaps around scheduling and input quantization, keep the best
W13/W2 tile configuration stable, and trim the W2 finalization kernel.

The larger end-to-end gap is outside this support pipeline. Even the relaxed
Attention ceiling takes 53.168 ms for Attention plus RoPE, compared with
32.422 ms in vLLM. The strict Attention path is slower. Dense projections,
normalization, casts, residual operations, and ggml graph launch boundaries
make up the rest.

<details>
<summary>Planning estimates used for the support-kernel follow-up</summary>

| Target | Measured gap before implementation | Strict experiment | Reusable implementation |
| --- | ---: | ---: | ---: |
| Histogram and prefix-sum expert scheduling | 9.57 ms | 300-500 lines | 600-900 lines |
| Input expansion and MXFP8 quantization | 12.33 ms | 250-450 lines | 450-750 lines |
| W13 bias, SwiGLU, and A2 quantization | 43.78 ms | 400-700 lines | 700-1,100 lines |

Shared dispatch, fallback checks, correctness tests, and profiling were
estimated at 250-450 lines for a strict GPT-OSS experiment or 400-700 lines
for a reusable implementation. The combined estimates were 1,200-2,100 lines
and 2,150-3,450 lines respectively.

</details>

For the original investigation, direct CUTLASS closes most of the expert GEMM
gap, and the optimized CUDA support kernels close the three surrounding compute
gaps. Further gains now depend more on launch integration and Attention than on
another rewrite of the grouped-GEMM mainloop.

I implemented the fused W13 layout, shared MoE scheduling and epilogues, SM120
TMA W13/W2, direct causal Attention, add-plus-RMSNorm, and the faster numerical-
relaxed Attention version in the same branch so their effects could be measured
separately. The code and benchmark switches keep these versions independently
selectable for review and cherry-picking.

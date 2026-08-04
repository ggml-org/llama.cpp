# GPT-OSS-120B prefill on Blackwell

On one RTX PRO 6000 Blackwell, the bitwise-compatible path improves
GPT-OSS-120B MXFP4 prefill from 9,168 to 17,708 tok/s at pp8192. This is a
1.93x speedup over the existing llama.cpp path. A faster experimental
Attention path reaches 23,301 tok/s, or 2.54x, but changes the logits and is
included only as a performance ceiling.

The same-machine vLLM result was about 36.7k tok/s. The bitwise-compatible
path is still 2.07x slower than vLLM, while the ceiling is 1.58x slower.

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

| Mode | ubatch 2048 | ubatch 8192 | Speedup at ubatch 8192 |
| --- | ---: | ---: | ---: |
| Existing path | 11,722 tok/s | 9,168 tok/s | 1.00x |
| Canonical persistent | 14,875 tok/s | 11,708 tok/s | 1.28x |
| Strict tuned | 15,092 tok/s | 17,708 tok/s | 1.93x |
| Full ceiling | 15,317 tok/s | 23,301 tok/s | 2.54x |

The larger ubatch hurts the existing and canonical paths, but helps the TMA
versions substantially. I have not isolated this completely. The current
evidence suggests that the TMA scheduler needs the larger microbatch to expose
enough grouped-MMQ work, while the generic path loses efficiency at that size.

The ceiling gains another 5,593 tok/s over the strict path. Nsys attributes
nearly all of that difference to Attention and RoPE rather than another MoE
change.

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

The full-ceiling path has measurable numerical drift:

| Prompt | NMSE | Max abs | Mean abs | RMSE |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 0.0037914378 | 0.78913784 | 0.1222908 | 0.1490868 |
| 8192 | 0.004538735 | 0.6545565 | 0.1366402 | 0.1644045 |

## Nsys profile

The final llama.cpp Nsys captures contain one cold pp8192 pass. The vLLM
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

The profile leaves two clear targets. W13 plus W2 takes 191.5 ms in the strict
path, compared with 105.041 ms for the named vLLM CUTLASS kernels. Attention
takes 50.5 ms in the fast experimental path, compared with 32.422 ms for the
vLLM FlashInfer kernel. Closing the first gap needs a better grouped-MMQ
kernel. Closing the second needs an Attention schedule that retains the
reference numerical behavior.

I implemented the fused W13 layout, shared MoE scheduling and epilogues, SM120
TMA W13/W2, direct causal Attention, add-plus-RMSNorm, and the faster numerical-
relaxed Attention version in the same branch so their effects could be measured
separately. The code and benchmark switches keep these versions independently
selectable for review and cherry-picking.

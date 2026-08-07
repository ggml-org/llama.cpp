# RDNA2 V620 Optimization Summary

Date: 2026-08-07

## Hardware and standard environment

- 4x AMD Radeon Pro V620, `gfx1030` / RDNA2
- PCIe 4.0 x16; no high-speed GPU interconnect
- ROCm 7.14, bundled RCCL
- `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- `HSA_NO_SCRATCH_RECLAIM=1`
- `GGML_CUDA_ALLREDUCE=nccl`
- `GGML_CUDA_P2P=1`
- `NCCL_P2P_LEVEL=PXB`
- `--flash-attn on`

## Final experimental branch

```text
exp/rdna2-q4k-mmid-batch6-pr24546-rdna2
```

Current HEAD:

```text
76671da65 docs: summarize RDNA2 V620 optimization results
```

Code HEAD before the documentation commit:

```text
8f21e1113 fix: apply routed width to RDNA2 MMQ tile search
```

Lineage:

```text
457b30a43  synchronized fork/master
9bdd850d7  six-row Q4_K/Q6_K MMVQ and Qwen MTP graph change
4cb5e6774  ported PR #24546 routed-MoE MMQ picker
52ad29fd0  enabled the picker by default on RDNA2
8f21e1113  fixed the picker integration to use ncols_picker
```

`master` remains at `457b30a43`; the experimental branch has not been merged into master.

## What the final branch contains

### Routed-MoE MMQ picker

The PR #24546 typical-expert-width picker is enabled by default for RDNA2. It sizes MMQ tile selection from the typical routed expert width while retaining the worst-case launch grid for coverage.

This applies to routed MMQ generally, not only Qwen3.6.

### Existing J16 tuning

The earlier RDNA2 J16 selector remains active:

- Qwen3.5-122B-A10B: exact four-way equal tensor split
- Qwen3.6-35B-A3B: exact four-way equal layer split
- Other models/topologies are not given the model-specific J16 hint

For the exact Qwen3.6 Q4_K path, the existing J16 selector can override the generic picker. Other routed types, including Q6_K paths, use the typical-width picker.

### Six-row MMVQ/MTP change

The model-scoped six-row hint applies to Qwen3.6-35B-A3B in the exact four-GPU layer topology. It covers Q4_K and Q6_K routed tensors and moves the practical graph-safe MTP limit from `n_max=4` to `n_max=5`.

This six-row change is not automatically applied to Qwen3.5-122B tensor split.

## Baseline and final performance

### Qwen3.6-35B-A3B Q4_K_M, four-GPU layer split

Prompt processing, `--batch-size 2048`, four-GPU tensor split `1/1/1/1`, picker disabled versus enabled:

| Prompt | ubatch | Disabled | Enabled | Gain |
|---:|---:|---:|---:|---:|
| 512 | 32 | 651 | 751 | +15.3% |
| 512 | 64 | 928 | 1321 | +42.4% |
| 512 | 128 | 2214 | 2964 | +33.9% |
| 512 | 256 | 2794 | 2907 | +4.0% |
| 512 | 512 | 2099 | 2319 | +10.5% |
| 512 | 1024 | 2107 | 2311 | +9.7% |
| 4096 | 128 | 1680 | 2268 | +35.0% |
| 4096 | 256 | 4912 | 5418 | +10.3% |
| 8192 | 128 | 1638 | 2199 | +34.2% |
| 8192 | 256 | 4963 | 5612 | +13.1% |

All values are prompt tokens/second from three-repetition `llama-bench` runs.

The best tested prompt-processing setting was generally `--ubatch-size 128`. The existing `ubatch=256` setting still benefits, especially at longer prompts.

### Qwen3.5-122B-A10B Q4_K, four-GPU tensor split

`Qwopus3.5-122B-A10B.gguf`, `--split-mode tensor`, `--tensor-split 1/1/1/1`, PP512, ubatch=128:

| Picker | PP512 |
|---|---:|
| Disabled | 759.5 tok/s |
| Enabled | 896.4 tok/s |
| Gain | +18.0% |

This validates that the generic RDNA2 picker and the Qwen3.5-122B tensor-split J16 path operate on the 122B model.

### Qwen3.6 raw decode graph comparison

Four-GPU layer mode raw generation:

| Graphs | TG |
|---|---:|
| Disabled | 72.22 tok/s |
| Enabled | 78.21 tok/s |
| Gain | +8.3% |

### MTP sweep before the six-row change

| `n_max` | TG | Acceptance |
|---:|---:|---:|
| 1 | 95.71 | 93.85% |
| 2 | 107.85 | 81.25% |
| 3 | 121.72 | 78.07% |
| 4 | 117.67 | 66.19% |
| 5 | 108.18 | 62.75% |
| 6 | 95.58 | 50.81% |

`n_max=3` remains the best measured effective throughput setting from that sweep.

After the six-row change, a release `n_max=5` run reached approximately 112.5 tok/s, with graph unsupported-node events limited to initialization. Debug diagnostics showed:

```text
n_max=5: 5 initialization events
n_max=6: 45 recurring events
```

### Earlier baseline work

- DeepSeek Q4_K_M and IQ3: approximately 20 tok/s generation, effectively identical
- DeepSeek split mode: tensor 20.7 tok/s; layer 24.4 tok/s
- Best tested RCCL setting: `NCCL_P2P_LEVEL=PXB`, approximately 22.2 tok/s in the relevant tensor-mode test
- Three versus four GPUs: approximately 24.3 versus 24.4 tok/s
- Qwopus context sweep: PP approximately 368–972 tok/s and TG approximately 57–70 tok/s over 512–65k context

## Correctness and validation

- Q4_K six-row MMVQ graph-on/off A/B: 0 mismatches over 24,576 values
- Q6_K six-row MMVQ graph-on/off A/B: 0 mismatches over 24,576 values
- Q4_K routed MMQ picker A/B: 0 mismatches over 262,144 values
- Q6_K routed MMQ picker A/B: 0 mismatches over 262,144 values
- Qwen/Qwen3.5/DeepSeek automatic RDNA2 MMQ configuration tests: PASS
- Production and debug `llama-server` builds: PASS

## Interpretation and remaining work

The MMQ picker affects prompt processing. It does not replace or directly fix the MMVQ MTP graph-cliff path. The six-row MMVQ change and the MMQ picker are complementary.

The branch is experimentally strong enough for continued testing. Before changing the final production command, repeat the MTP/server sweep with `ubatch=128` and `ubatch=256`, using matched longer workloads. Keep `n_max=3` as the current effective-throughput reference until that sweep is complete.

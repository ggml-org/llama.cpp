# Micro-Graph Experiment Results (llama.cpp-5pki)

**Date**: February 25, 2026
**Branch**: feature/sycl-coalescing
**Commits**: b9374dc75 (Step 0 benchmark), 8482f2d97 (micro-graph implementation)

## Step 0: Graph Overhead Benchmark

Per-node SYCL graph replay latency on Intel Arc B580 (Level Zero 1.14, oneAPI 2025.3):

| Test | Nodes | us/replay | us/node |
|------|-------|-----------|---------|
| single_task | 350 | 164.7 | 0.47 |
| parallel_for nd=256 | 350 | 189.5 | 0.54 |
| parallel_for nd=32768 | 350 | 315.5 | 0.90 |
| parallel_for+SLM=16K | 350 | 313.5 | 0.90 |
| Mixed realistic | 256 | 184.7 | 0.72 |
| No-graph baseline | 350 | 616.1 | 1.76 |
| Scaling N=700 | 700 | 367.5 | 0.53 |

**Decision**: STRONG PROCEED (< 1 us/node, well under 3 us threshold).

## Performance Results

| Mode | TG128 (tok/s) | ms/token | PP512 |
|------|--------------|----------|-------|
| Non-persistent (graph replay) | 70.14 | 14.3 | ~1335 |
| **Micro-graph persistent** | **38.49** | **26.0** | 1335 |
| Monolithic persistent (phase) | 34.49 | 29.0 | ~1335 |

Micro-graph is 11% faster than monolithic persistent. Zero PP regression.

## Key Finding

Barriers are NOT the primary bottleneck. The persistent kernel's generic dequant
compute path is ~2x slower than the dedicated MMVQ kernels. Graph overhead is
only ~200-400us (vs 15ms barriers), confirming the graph approach works, but
the bigger win requires optimizing per-op compute to match MMVQ quality.

## Architecture

- MicroPhaseArgs struct for per-phase kernel arguments
- PersistentTGKernelImpl::run_micro_phase() static method
- Per-phase tile counters (malloc_device, bulk-zeroed once per token)
- Graph recorded once, replayed each token
- Ops table (malloc_host) updated by UPDATE recipe, read via PCIe zero-copy
- build_only flag on launch_persistent_kernel() to skip monolithic kernel launch

## Env Vars

- `GGML_SYCL_PERSISTENT_TG_MICRO_GRAPH=1`: Enable micro-graph mode
- `GGML_SYCL_PERSISTENT_TG_BENCH_GRAPH=1`: Run graph overhead benchmark on first token

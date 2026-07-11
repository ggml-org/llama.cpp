# Standard SYCL fork-vs-upstream A/B — 2026-07-11

## Decision

The private fork can align its standard f16 path with upstream, but it should
not wholesale-drop its standard q8_0 VEC delta yet.

Across all three service-class models, fork and upstream f16/f16 were
correctness-equivalent and effectively throughput-parity. For q8_0/q8_0, both
sides passed the paired PPL gate, while the fork decoded 5.3-6.7% faster.
A source diff shows one generic standard-path difference in
`ggml/src/ggml-sycl/fattn-vec.hpp`: the fork folds `scale_h2` into each
`Q_reg` load, while upstream performs a separate post-load scaling loop.

The correlation is strong but causality is **[UNVERIFIED]** until that hunk is
A/B-tested alone. Keep this single generic delta isolated temporarily; do not
use the result to retain unrelated standard-SYCL divergences. A narrow
follow-up should patch upstream or revert the fork hunk and rerun only the
q8_0 cells with the same PPL gate.

No TurboQuant or InnerQ path was included.

## Pins and builds

- Host/device: `vinbonesjr`, Intel Arc A770, Level Zero device 0.
- Fork: clean detached `Raudbjorn-fork` commit
  `9a43bf4de97551bb4f30016e0f620ded18b9c795` at
  `/home/svnbjrn/llama-fork-p43-9a43bf4de`.
- Upstream: `ggml-org/llama.cpp` master pinned through the GitHub API at
  `1d1d9a9ed7a4f09c4225ea4cc8fd3bd1cf2c940f`, clean detached worktree
  `/home/svnbjrn/llama-upstream-p43-1d1d9a9ed`.
- Build mode: identical Release IntelLLVM/oneAPI 2026.0 JIT configuration;
  `GGML_SYCL=ON`, target INTEL, F16 ON, Level Zero/graph/host fallback ON,
  `GGML_SYCL_DEVICE_ARCH` empty, compiler launchers empty.
- Fork build: `/home/svnbjrn/build-p43-fork-jit-9a43bf4de`.
- Upstream build: `/home/svnbjrn/build-p43-upstream-jit-1d1d9a9ed`.
- Artifact hashes: `/tmp/p43-upstream-ab/build-sha256.txt`.
- Raw matrix: `/tmp/p43-upstream-ab/`; runner exit `MATRIX_EXIT=0`.

Every fork JSONL row reports `build_commit: 9a43bf4de`; every upstream row
reports `build_commit: 1d1d9a9ed`. `ldd` with each runner-scoped
`LD_LIBRARY_PATH` resolved `libllama`, `libggml-base`, `libggml-cpu`, and
`libggml-sycl` exclusively from that variant's own build directory. This
prevents cross-loading the shared `libggml-sycl.so.0` SONAME.

## Method

The A/B runner executed 12 cells:

```text
{fork, upstream} x {Llama-3.1-8B, Mistral-7B, Qwen3-Coder-30B-A3B}
                 x {f16/f16, q8_0/q8_0}
```

Throughput command shape:

```text
llama-bench -m MODEL -ngl 99 -fa on -ctk KV -ctv KV \
  -p 512 -d 2048 -n 256 -b 512 -ub 512 -r 3 \
  --no-warmup -o jsonl
```

Correctness command shape:

```text
llama-perplexity -m MODEL -ngl 99 -fa on -ctk KV -ctv KV \
  -c 512 -b 512 -ub 512 \
  -f /mnt/mrgr/projects/llama-cpp-turboquant/wikitext-2-raw/wiki.test.raw \
  --chunks 8 --no-warmup --no-mmap
```

Every leg had a fail-closed render-node occupancy check. All 24 `.fuser`
guard files were empty. The correctness gate required finite PPL and less than
1% relative q8_0-vs-f16 PPL change within each build/model.

## Results

Values are mean +/- sample standard deviation over three throughput runs.

| Model | KV | Fork pp512 | Upstream pp512 | Fork tg256@d2048 | Upstream tg256@d2048 | Fork-vs-upstream tg | Fork PPL | Upstream PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | f16 | 576.355 +/- 0.185 | 576.602 +/- 0.182 | 22.579 +/- 0.483 | 22.582 +/- 0.514 | -0.01% | 8.0287 | 8.0291 |
| Llama-3.1-8B | q8_0 | 573.605 +/- 0.391 | 569.056 +/- 0.386 | 20.787 +/- 0.856 | 19.744 +/- 0.481 | +5.28% | 8.0334 | 8.0314 |
| Mistral-7B | f16 | 576.717 +/- 0.210 | 575.389 +/- 2.740 | 23.507 +/- 0.527 | 23.460 +/- 0.524 | +0.20% | 8.2270 | 8.2269 |
| Mistral-7B | q8_0 | 574.202 +/- 0.428 | 570.174 +/- 0.174 | 21.754 +/- 0.471 | 20.388 +/- 0.431 | +6.70% | 8.2318 | 8.2256 |
| Qwen3-Coder-30B-A3B | f16 | 161.641 +/- 1.398 | 163.516 +/- 1.266 | 13.977 +/- 0.021 | 13.916 +/- 0.063 | +0.44% | 9.5895 | 9.5918 |
| Qwen3-Coder-30B-A3B | q8_0 | 161.515 +/- 1.590 | 162.667 +/- 1.406 | 12.792 +/- 0.026 | 12.047 +/- 0.052 | +6.19% | 9.5700 | 9.5943 |

Fork-vs-upstream PPL differences were at most 0.253% absolute-relative, far
below the eight-chunk uncertainty. Within each build, every q8_0-vs-f16 PPL
change was below 1%.

## Source mapping

The standard VEC thread count and subgroup barrier are already structurally
aligned in fork and upstream. The remaining relevant diff includes:

```text
upstream: Q_reg[...] = half2(tmp.x(), tmp.y());
          ... separate unrolled Q_reg *= scale_h2 loop
fork:     Q_reg[...] = half2(tmp.x(), tmp.y()) * scale_h2;
```

The fork also adds compile-time TurboQuant routing and template declarations;
those are outside this A/B because only f16 and q8_0 were run. Git blame maps
the scale-fold form to fork commit `b1566523e`, where it landed amid a broad
backend-prune/TurboQuant reconciliation commit. That mixed provenance is why
an isolated-hunk A/B is required before calling it the cause of the q8_0 gain.

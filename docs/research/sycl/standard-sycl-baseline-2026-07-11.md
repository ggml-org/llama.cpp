# Standard SYCL f16/q8_0 baseline - 2026-07-11

## Verdict

PASS. On the Intel Arc A770, all six standard-SYCL configurations completed a
sole-tenancy throughput run and an eight-chunk WikiText-2 perplexity check.
Symmetric q8_0 KV changed PPL by at most 0.21% relative to symmetric f16 KV on
the three-model fleet. q8_0 decode throughput was 10-11% lower than f16 in
this depth-2048 probe; this is acceptable for the capacity framing, where
throughput parity is not the gate.

This is a pinned baseline, not a current-HEAD claim.

## Provenance

- Host/device: `vinbonesjr`, Intel Arc A770, Level Zero device 0.
- Source: `Raudbjorn-fork`, detached clean checkout at
  `10a70cde8161b5f2d3a50ed6e95ea74ec862ce25`.
- Build mode: Release, IntelLLVM/oneAPI 2026.0, SYCL AOT `acm-g10`,
  `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`.
- Throughput binary:
  `/home/svnbjrn/build-turbo-aot-decode-10a70cde8/bin/llama-bench`, SHA-256
  `64366b6d3f6f5c23a0721f5fdb49da7e80d7dba44eba7bd83002ff7f8416b63b`.
- Correctness binary:
  `/home/svnbjrn/build-turbo-aot-discriminator/bin/llama-perplexity`, SHA-256
  `73770bee05c9f71af5c91dbf18f1e3b42a0e88536f994da0601208025d38a0dd`.
- Loaded SYCL library:
  `/home/svnbjrn/build-turbo-aot-discriminator/bin/libggml-sycl.so.0.15.1`,
  SHA-256
  `4bbccb54b2f034307eb13656e936b432a1e786cfde8ce068d551d30557a338ed`.
- Both executables report version `10041 (10a70cde8)`; every benchmark JSONL
  row reports `build_commit: 10a70cde8`.
- A post-run loader audit used the runner's `LD_LIBRARY_PATH`, displayed here
  in abbreviated form:
  `build-turbo-aot-discriminator/bin:build-turbo-aot-decode-10a70cde8/bin:...`.
  `ldd` showed both executables resolving `libllama`, `libggml-base`,
  `libggml-cpu`, and `libggml-sycl` from
  `build-turbo-aot-discriminator/bin`; therefore throughput and PPL used the
  same runtime compute stack. The loaded SYCL-library hash above matches the
  pristine `10a70cde8` discriminator manifest at
  `/tmp/p324a-discriminator-manifest.txt`.
- The discriminator build directory was subsequently reconfigured against a
  moving checkout, so its generated CMake metadata is not artifact provenance.
  The executable's live `--version`, matching content hashes, pristine-build
  manifest, and loader resolution are the binding evidence.
- Raw evidence: `/tmp/p41-standard-baseline-10a70cde8/`.

The live branch was `turbo-sycl-opt` at `32c4e1e37` when this baseline ran.
`git diff --stat 10a70cde8..32c4e1e37` contains InnerQ APIs/wrappers,
clear-data-only cache lifecycle changes, test/quality-gate work, and corrected
block-layout comments; it contains no standard f16/q8_0 SYCL kernel edit.
The numbers below nevertheless apply only to the pinned commit and binaries.

## Method

The user authorized terminating other user processes holding the Arc. Chrome
PID 3652856 and Codium Insiders PID 3661268 were terminated. The runner then
required `fuser /dev/dri/renderD128` to return no holder before every leg; all
12 guard files are empty and the runner wrote `EXIT=0`.

Each throughput leg used:

```text
llama-bench -m MODEL -ngl 99 -fa on -ctk KV -ctv KV \
  -p 512 -d 2048 -n 256 -b 512 -ub 512 -r 3 \
  --no-warmup -o jsonl
```

`pp512` is prompt throughput. `tg256@d2048` is generation throughput after a
2048-token depth prefill. Values are mean +/- sample standard deviation over
three repetitions.

Each paired correctness leg used the same model, KV types, FA setting, GPU
offload, and batch sizes:

```text
llama-perplexity -m MODEL -ngl 99 -fa on -ctk KV -ctv KV \
  -c 512 -b 512 -ub 512 \
  -f /mnt/mrgr/projects/llama-cpp-turboquant/wikitext-2-raw/wiki.test.raw \
  --chunks 8 --no-warmup --no-mmap
```

The correctness gate was successful completion with finite PPL and less than
1% relative q8_0-vs-f16 PPL change per model. This is a short coherence/PPL
check, not a publication-quality corpus estimate.

## Results

| Model | KV K/V | pp512 tok/s | tg256@d2048 tok/s | PPL |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | f16/f16 | 652.368 +/- 0.290 | 25.611 +/- 0.739 | 8.0287 +/- 0.43592 |
| Llama-3.1-8B | q8_0/q8_0 | 648.821 +/- 0.509 | 23.083 +/- 0.632 | 8.0334 +/- 0.43617 |
| Mistral-7B | f16/f16 | 651.653 +/- 0.570 | 26.475 +/- 0.758 | 8.2270 +/- 0.46891 |
| Mistral-7B | q8_0/q8_0 | 648.891 +/- 0.447 | 23.681 +/- 0.614 | 8.2318 +/- 0.46946 |
| Qwen3-Coder-30B-A3B | f16/f16 | 190.324 +/- 1.818 | 14.438 +/- 0.018 | 9.5895 +/- 0.64126 |
| Qwen3-Coder-30B-A3B | q8_0/q8_0 | 189.702 +/- 2.127 | 12.950 +/- 0.025 | 9.5700 +/- 0.64132 |

Relative q8_0-vs-f16 PPL changes were +0.059% (Llama-3.1), +0.058%
(Mistral), and -0.203% (Qwen3-Coder). All pass the predeclared 1% gate and
are much smaller than the reported eight-chunk uncertainty.

## Interpretation

The standard SYCL path is coherent on all three service-class models for both
symmetric f16 and q8_0 KV. q8_0 is not a decode-speed win in this probe; it is
the lower-capacity-cost standard baseline against which turbo KV should be
judged. No result here validates turbo KV, InnerQ, non-FA block-quantized KV,
or an upstream-vs-fork comparison.

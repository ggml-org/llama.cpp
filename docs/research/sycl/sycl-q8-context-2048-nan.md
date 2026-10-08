# SYCL Q8 KV: context >= 1024 perplexity NaNs (resolved 2026-09-05)

Status: resolved. Root cause found and fixed on 2026-09-05; the oracle now covers
the failing route. This note tracks the defect because repository issues are
disabled. The original open note is preserved below the resolution.

## Root cause

With q8_0 K and V and flash attention on, the tested MKL-routed perplexity
workloads on the Arc A770 produced NaN at contexts 1024, 1536, and 2048.
Context 512 was finite in the same configuration (model and commands below).

The failing shape routes to the fork-local MKL GEMM flash-attention prefill
kernel (`ggml/src/ggml-sycl/fattn-mkl.cpp`). `ggml_sycl_get_best_fattn_kernel()`
selects it for eligible non-turbo K/V with a mask, `gqa_ratio >= 2`, supported
head dimensions, no sinks/ALiBi/softcap, compatible batch dimensions and valid
type/stride conditions, when `Q->ne[1] >= 32` and `K->ne[1] >= 1024`
(`GGML_SYCL_ENABLE_MKL_FA`, default 1). With the tested ubatch 512 configuration,
a context-2048 chunk walks n_kv 512, 1024, 1536, 2048 and every ubatch after the
first takes MKL; the context-512 run never reaches it.

`mkl_fa_dequant_chunk` stages K/V to dense f16 per chunk. It called
`ggml_get_to_fp16_sycl(type, dst)` with the destination tensor and the one-argument
`ggml_get_to_fp16_nc_sycl(type)`, so the q8_0 converters never saw the
`GGML_TENSOR_FLAG_KV_Q8_QUANTS_FIRST` flag on K/V and decoded the quants-first
row layout (default-on since #33, `754dc99e2`, for SYCL q8_0 K and V with
128-wide heads) as canonical q8_0. Quant bytes were reinterpreted as fp16 scales,
Inf/NaN entered the staged K chunk and poisoned the online softmax.

The MKL kernel landed with the #50 upstream sync (2026-09-03), after the
quants-first default, and nothing exercised the two together: the oracle used
n_q=8 and n_kv=256 (never MKL) and `test-backend-ops` cannot set the layout flag.

## Bisect that isolated it

`~/build-pr51-sycl/bin/llama-perplexity`, Llama-3.1-8B-Instruct Q4_K_M, Wikitext-2
raw test, 2 chunks, `-fa on`, ubatch 512, `ONEAPI_DEVICE_SELECTOR=level_zero:0`:

| K/V | ctx | knob | PPL |
| --- | --- | --- | ---: |
| q8_0/q8_0 | 2048 | default | NaN |
| q8_0/q8_0 | 1024, 1536 | default | NaN |
| q8_0/q8_0 | 2048 | ubatch 256 | NaN |
| f16/f16 | 2048 | | 8.0997 |
| q8_0/f16 | 2048 | | 8.0957 |
| f16/q8_0 | 2048 | | 8.0974 |
| q8_0/q8_0 | 2048 | `GGML_SYCL_Q8_KV_QUANTS_FIRST=0` | 8.0984 |
| q8_0/q8_0 | 2048 | `GGML_SYCL_ENABLE_MKL_FA=0` | 8.0939 |

`GGML_SYCL_FA_FORCE_VEC_STANDARD` and `GGML_SYCL_FA_Q8_GQA_TILE` did not
discriminate because they apply to decode (`Q->ne[1] == 1`) only. The route
logger used to skip MKL and ONEDNN, so `GGML_SYCL_FA_ROUTE` output was silent.

## Fix

- `fattn-mkl.cpp`: the KV descriptor carries the source tensor; both quantized
  dequant modes call the tensor-aware converters
  (`ggml_get_to_fp16_sycl(type, tensor)`, `ggml_get_to_fp16_nc_sycl(type, tensor)`),
  which return the quants-first kernels when the flag is set.
- `fattn.cpp`: `ggml_sycl_log_fattn_route_once` now reports MKL and ONEDNN.
- `tests/test-sycl-turbo-correctness.cpp`: `probe_flash_attn` and `probe_fa_f16`
  take `n_kv`; new GATE section [4c] runs d=128, n_q=64, GQA 4:1, n_kv 1024 and
  2048 for f16 and q8_0. Each probe checks MKL selection; q8_0 explicitly tests
  both canonical and quants-first rows in one run. Before the fix the quants-first cases
  failed with `nmse=-nan`; after it they pass with the same error as canonical
  rows (nmse 5.8e-05 and 6.3e-05, cosine 0.99997).

## Verification (build `~/build-master-435f47bb8`, fix applied)

- Oracle default sweep: `0 GATE-FAIL, 0 XPASS, 0 xfail, 0 SKIP` (56 cases incl. [4c]).
- Oracle with `LLAMA_TEST_TURBO_FA=1`: `0 GATE-FAIL, 0 XPASS, 6 xfail` (unchanged set).
- Oracle with `GGML_SYCL_Q8_KV_QUANTS_FIRST=0`: all green.
- Real model, q8_0/q8_0, `-fa on`, ubatch 512 (same command as below):

| ctx | chunks | PPL |
| ---: | ---: | ---: |
| 512 | 8 | 7.9861 (control, unchanged) |
| 1024 | 2 | 5.8843 |
| 1536 | 2 | 6.5355 |
| 2048 | 2 | 8.0984 |
| 2048 | 2 | 8.0939 with `GGML_SYCL_ENABLE_MKL_FA=0` |
| 2048 | 8 | 5.8632 |

The 2048/2-chunk value equals the pre-fix `GGML_SYCL_Q8_KV_QUANTS_FIRST=0` result
to four decimals, and the MKL-on and MKL-off numbers differ by 0.0045.

Host-specific evidence command (requires the SYCL build, model, and Wikitext-2
corpus at the recorded paths):

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=1
timeout 300 ~/build-master-435f47bb8/bin/llama-perplexity \
  -m /mnt/mrgr/models/llama31-8b-q4km/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -f /home/svnbjrn/.cache/pr51-integration/wikitext-2-raw/wiki.test.raw \
  -t 4 -tb 4 -ngl 99 -fa on -fit off --device SYCL0 \
  -c 2048 -b 512 -ub 512 --chunks 2 -ctk q8_0 -ctv q8_0
```

Not claimed: no throughput measurement of the MKL route; no KLD run against f16
logits at ctx 2048; no wave64 or non-A770 execution; the 6 turbo-FA xfails are
the pre-existing turbo2 set, not touched here.

## Original note (2026-09-05, before the root cause)

On the Arc A770 with oneAPI 2026.0, Llama-3.1-8B-Instruct Q4_K_M and
Wikitext-2 raw test data, Q8_0 K/V with flash attention produced NaN perplexity
at context 2048. The isolated pre-integration build (tree `fdfa74483`) also
produced `[1]-nan,[2]-nan,` at context/batch/ubatch 2048. The integrated build
reproduced it with both 512-token and 2048-token batches. The context-512
eight-chunk Q8 control was finite (PPL 7.9861) in both builds. Saved logits from
the NaN runs are invalid KLD reference data. Evidence under
`/home/svnbjrn/.cache/pr51-integration/`: `baseline-sycl-probe.sh`,
`ppl-baseline-sycl-q8-2048.log`, `ppl-q8-2048.log`, `ppl-q8-2048-singlebatch.log`.

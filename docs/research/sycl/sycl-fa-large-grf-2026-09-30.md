# SYCL flash attention on the A770: register spills and an opt-in 256-GRF knob - 2026-09-30

Side result of the xe copy-engine investigation
(`xe-kmd-bcs-copy-engine-2026-09-30.md`, same host and stack: Arc A770 on xe 7.3-rc5,
IGC 2.41.5, oneAPI 2026.1.4). Split out of PR #88 into its own change on branch
`sycl-fa-large-grf`.

## The FA kernels spill, and a large-GRF knob (2026-09-30, 10:50-11:20)

IGC shader dumps (`IGC_ShaderDumpEnable=1`, `SYCL_CACHE_PERSISTENT=0`, IGC 2.41.5,
`/mnt/ssd1/igc-dumps`) of the oracle and a production real-text run: 23 of 173 kernels use
`GRAPH_COLORING_SPILL_FF_RA`, 21 of them FA kernels at 128 GRF. Production (q8_0 KV,
d=256): `flash_attn_tile<256,256,4,8>` 15 392 B spill / 55k spill refs,
`flash_attn_tile<256,256,2,8>` 10 848 B / 26k, `flash_attn_ext_vec<256,1,q8_0,q8_0>`
8 160 B / 1.9k (the decode kernel); `mul_mat_vec_q6_K` under 200 B. Spills are legacy
`send.dc0` hword scratch block messages (writes at SIMD8 with descriptor `0x020F00xx`, fills
at SIMD16 with `0x021C00xx`), the pattern the gaema IGC fork's vISA patches target; the
oracle is green with these kernels executing, so no miscompile is observed here.

`GGML_SYCL_FA_LARGE_GRF` (0 off, 1 = tile launches with more than one query row; the first
revisions also had a tile-and-vec mode 2, dropped below) passes
`sycl::ext::intel::experimental::grf_size<256>` as a kernel property in
`lauch_kernel` (`ggml/src/ggml-sycl/fattn-common.hpp`). The 256-GRF instantiations exist
for the tile kernels only and are compiled only with `-DGGML_SYCL_FA_LARGE_GRF=ON` (default
OFF: every FA tile kernel would carry a second device image, and an AOT build would compile
each of them twice); other builds warn and ignore the variable. Paired product campaign
(`scripts/bench-a770-fork-unique.py --campaign product`, q8_0/q8_0, 4 repetitions, sample 0
discarded, sole tenancy, no kernel message) of the compile-time equivalent of mode 2
against the default build, `/mnt/nvme1/oneapi-ab/grf256-*`. Provenance (product.json
`build_commit`): baseline `6bd56b372` default build (`~/build-xe-kmd`), candidate `73398f9f3`
built with 256 GRF compiled in for every FA kernel (`~/build-fa-grf256`, before the runtime
knob existed), occupancy heuristic unchanged:

| model | depth | pp512 default -> 256 GRF | tg128 default -> 256 GRF |
|---|--:|--:|--:|
| Ornith IQ2_M (d=256, 41 layers) | 0 | 266.0 +- 1.4 -> 274.9 +- 0.5 (+3.3 %) | 53.5 -> 53.4 (flat) |
| | 2048 | 299.0 -> 298.8 (flat) | 52.9 -> 52.8 (flat) |
| | 8192 | 265.2 -> 265.3 (flat) | 46.6 +- 0.1 -> 48.3 +- 0.0 (+3.6 %) |
| Llama 3.1 8B Q4_K_M (d=128) | 0 | 1004.0 +- 2.0 -> 1098.0 +- 13.9 (+9.4 %) | 47.0 -> 47.7 (+1.5 %) |
| | 8192 | 210.8 -> 211.3 (flat) | 36.5 +- 0.0 -> 34.8 +- 0.0 (-4.5 %) |

Reading: the prefill gain is the tile kernel's spill traffic; the d=128 decode loss at depth
is the vec kernel at half occupancy for a kernel that spilled little. Hence the tile-only
mode 1, measured on one binary (`e4bf0b239`, the knob's first build, occupancy unchanged)
with env-only arms, same protocol (`/mnt/nvme1/oneapi-ab/grfmode1-*`, `all_cells_valid: true`):

| model | depth | pp512 off -> mode 1 | tg128 off -> mode 1 |
|---|--:|--:|--:|
| Ornith IQ2_M (d=256) | 0 | 267.7 +- 2.6 -> 274.5 +- 0.3 (+2.5 %) | 53.5 -> 53.5 (flat) |
| | 8192 | 267.3 +- 2.4 -> 265.1 +- 2.0 (-0.8 %, within noise) | 46.6 -> 46.5 (flat) |
| Llama 3.1 8B (d=128) | 0 | 1005.6 +- 19.1 -> 1107.8 +- 8.1 (+10.2 %) | 46.9 -> 46.9 (flat) |
| | 8192 | 210.9 -> 210.9 (flat) | 36.4 -> 36.4 (flat) |

Both campaigns above ran with the occupancy heuristic unchanged (`max_wg_per_cu` = 16 for
the 256-GRF launches too). The knob now halves it for large-GRF launches; mode 1 rerun
with that change, same protocol, one binary with the variants compiled in (`bff20e62a`
plus the uncommitted occupancy change that became `30533cbb1`,
`/mnt/nvme1/oneapi-ab/grfmode1occ-*`, `all_cells_valid: true`, no kernel message):

| model | depth | pp512 off -> mode 1 | tg128 off -> mode 1 |
|---|--:|--:|--:|
| Ornith IQ2_M (d=256) | 0 | 266.5 +- 2.6 -> 273.0 +- 4.2 (+2.4 % +- 1.1) | 53.4 -> 53.3 (flat) |
| | 8192 | 265.3 -> 265.0 (flat) | 46.4 -> 46.4 (flat) |
| Llama 3.1 8B (d=128) | 0 | 998.4 +- 9.7 -> 1098.3 +- 8.2 (+10.0 % +- 0.9) | 46.9 -> 46.9 (flat) |
| | 8192 | 211.1 -> 211.2 (flat) | 36.4 -> 36.4 (flat) |

The occupancy change moves nothing outside the CIs: the tile kernels at prefill are not
occupancy-bound at either target. Mode 1 keeps the short-context prefill gain and costs
nothing on decode. It stays opt-in (and, since the review, the 256-GRF instantiations are
compiled only with `-DGGML_SYCL_FA_LARGE_GRF=ON`):
the gain is confined to prefill below the MKL gate (n_kv < 1024), and one host, one
compiler version. A first mode-1 product was rejected by the harness only because the
binary predated the fix commit it was compared against (provenance gate); the rows above
are from the relinked binary. Oracle (default sweep with turbo FA,
and `LLAMA_TEST_FA256=1`) green on the 256-GRF build: `0 GATE-FAIL`, no hang.

## Mode 2 dropped (2026-10-01)

A rerun of both modes on the PR head (`9ae0f9ad9` plus the review edits, occupancy
change active for both modes, products `/mnt/nvme1/oneapi-ab/grfmode{1,2}pr89-*`) was
marked invalid by the harness's provenance gate (run from the xe checkout, so the
repository commit did not match the binary) and ran at host load 10-11 (postgres), so it
is indicative only. Its decode cells were tight enough to decide the mode-2 question:

| model | depth | mode 2: pp512 off -> on | mode 2: tg128 off -> on |
|---|--:|--:|--:|
| Ornith IQ2_M (d=256) | 0 | 237.3 +- 44.3 -> 236.2 +- 16.2 (+0.0 % +- 22.7) | 52.4 -> 52.1 (-0.6 % +- 6.6) |
| | 8192 | 257.3 -> 262.4 +- 16.6 (+2.0 % +- 6.2) | 45.7 +- 1.5 -> 42.0 +- 0.1 (-7.9 % +- 3.4) |
| Llama 3.1 8B (d=128) | 0 | 922.0 +- 140 -> 999.8 +- 193 (+8.4 % +- 4.3) | 46.5 -> 47.1 (+1.2 % +- 2.2) |
| | 8192 | 209.8 -> 209.2 (-0.3 % +- 1.0) | 35.9 +- 0.1 -> 34.4 +- 0.6 (-4.1 % +- 2.0) |

With the vec kernels at half occupancy, mode 2 loses decode at depth on both head dims; the
earlier +3.6 % at d=256 was measured without the occupancy change. Mode 2 is removed: the
knob is 0 or 1, and the 256-GRF instantiation exists for the tile kernels only
(`tile_route` is a template parameter of `launch_fattn`), so an ON build adds device code
for about ten tile translation units instead of all FA kernels.

## The shipping binary (PR #89 head, mode 1 = prefill tile launches)

Same protocol, env-only arms on the build of this PR's head with `GGML_SYCL_FA_LARGE_GRF=ON`,
run from the PR worktree so the provenance gate sees the matching commit
(`~/build-grf-split`; products `/mnt/nvme1/oneapi-ab/grfmode1final-*`). For the measured
cells the code path equals the `bff20e62a` rerun above (pp512 is a tile prefill launch,
tg128 is q8_0 VEC decode at 128 GRF either way); the device gate is the only new branch.

Build `33f59efee` (code of this PR; the docs were amended afterwards), `all_cells_valid: true`,
no kernel message, but host load 10-12 throughout (postgres), which widens the depth-0
prefill intervals far beyond the quiet-host runs:

| model | depth | pp512 off -> mode 1 | tg128 off -> mode 1 |
|---|--:|--:|--:|
| Ornith IQ2_M (d=256) | 0 | 249.5 +- 44.8 -> 256.6 +- 31.9 (+3.0 % +- 7.6) | 52.2 +- 1.8 -> 52.8 +- 1.0 (+1.1 % +- 5.3) |
| | 8192 | 259.6 +- 9.7 -> 260.3 +- 9.0 (+0.3 % +- 0.8) | 45.9 +- 1.4 -> 45.3 +- 0.8 (-1.2 % +- 2.3) |
| Llama 3.1 8B (d=128) | 0 | 942.2 +- 196 -> 1051.7 +- 51.4 (+12.2 % +- 27.5) | 46.6 +- 0.9 -> 46.9 +- 0.1 (+0.5 % +- 2.2) |
| | 8192 | 210.5 +- 0.7 -> 209.2 +- 1.7 (-0.6 % +- 0.8) | 36.0 +- 0.7 -> 36.0 +- 0.5 (-0.1 % +- 3.1) |

Reading: same sign and size as the quiet-host `bff20e62a` rerun, decode flat and the 8k
cells flat, with intervals too wide at depth 0 to add precision. The `bff20e62a` table is the
sharper measurement of the same code path; this one ties the behaviour to the shipping
binary. The geometry profile on this binary shows the intended split: `route=TILE
phase=prefill ... max_wg_per_cu=8 ... grf=256`, `route=VEC phase=decode ... max_wg_per_cu=16
... grf=128`.

## Not claimed

- The large-GRF numbers are one campaign per model and mode on one host with IGC 2.41.5;
  the standing "global large-GRF is a dead end" decision is untouched (this is per-kernel,
  opt-in).
- The mode-2 large-GRF table predates the occupancy change (half the work-groups per
  Xe-core for 256-GRF launches); only mode 1 was re-measured after it.
- Oracle and campaigns ran on the JIT build only; the AOT (`acm-g10`) build with the option
  on was not built, so its compile-time cost is an estimate (every FA tile kernel twice).
- The device gate allows Xe-HPG (acm_g10/g11/g12), Xe-HPC (pvc, pvc_vg) and Xe2
  (bmg_g21/g31, lnl_m); only acm_g10 is measured, the rest are listed on documentation
  and were not run.
- The mode-2 decision rests on an invalid product (provenance gate, host load 10-11); its
  decode deltas were -7.9 % +- 3.4 and -4.1 % +- 2.0, so the sign is not in doubt, the size is.
- The final-binary campaign ran at host load 10-12; its depth-0 prefill intervals (+- 7.6
  and +- 27.5 points) do not add precision, the quiet-host run of the same code path does.
- The knob was measured with q8_0/q8_0 KV only; turbo KV and f16 KV were covered by the
  oracle for correctness, not benchmarked.

# Tessera C++ Port Design

Design-only scoping doc for porting the Tessera T640 quantizer from
Python (`tools/tile640/quantize_v3.py` + `tools/tessera/*`) into
stock llama.cpp's `llama-quantize` tool. The port extends, not
replaces: stock K-quants remain reachable through `--tessera-mode=off`,
and every flag below is added to the existing CLI rather than
introducing a new tool or subcommand.

> Roadmap alignment: the runtime-aware proxy-objective research
> (2026-07-30) reframes parts of G2, G4, and G6 and reorders the port
> around the kernel-fidelity loop. See
> [`research-alignment-2026-07-30.md`](research-alignment-2026-07-30.md).
> Where that doc and this one disagree, that doc wins until this one is
> updated. The inline notes below mark the touched sections.

## Architectural decisions (locked)

These are settled by the prior conversation and not revisited below:

1. EXTEND `llama-quantize`; the Tessera pipeline is the new default.
2. Custom versioned `TESSERA_*` GGUF type family. Stock `ggml.h` enum
   gets new entries.
3. AWQ-evolve GA is part of `llama-quantize` (flag-driven).
4. Backward-compat ON: stock K-quants still work via
   `--tessera-mode=off` (the only mode flag).
5. System libopenblas is REQUIRED on Linux (`find_package(OpenBLAS)`).
   If not found, `llama-quantize` prints an install prompt for the
   user's distro (apt / yum / dnf / pacman) and exits. No naive shim
   fallback. macOS uses Accelerate (no extra install).
6. GA is deterministic via `--tessera-evolve-seed`; bit-identical
   policy across runs.
7. Flag-driven default flow: presence/absence of `--tessera-imatrix`,
   `--tessera-policy`, `--calib-corpus` controls which steps run. No
   imatrix + no policy + no corpus = run calibration on the built-in
   mini-corpus + run GA + quantize.

### Architect decisions on the 7 open questions (2026-07-30)

The scoping agent surfaced 7 questions in section 7. The architect
(2026-07-30) locked the following answers; the agent's leans in
section 7 are superseded:

8. **TESSERA_T640_3D**: separate `GGML_TYPE_TESSERA_T640_3D` enum
   entry. Mirrors `GGML_TYPE_MXFP4_MOE`.
9. **LAPACK install**: hard requirement. Missing libopenblas is a
   user-actionable install error, not a runtime fallback.
10. **L5 orchestrator**: one tool, one mode. No `--tessera-mode=...`
    family. The pipeline always runs end-to-end; intermediate outputs
    are opt-in via output-targeting flags
    (`--tessera-evolve-only` writes policy JSON only;
    `--tessera-calibrate-only` writes imatrix only). The
    `--tessera-mode=off` flag is the only mode flag and exists solely
    to opt back into stock K-quants.
11. **Calibration policy location**: both. Small
    `tessera.calibration.policy` string in GGUF metadata
    (`tensor_families` block only, no U/V factor payloads). Full
    U/V payloads and GA archive in sidecar JSON. SHA-256 in both.
12. **Built-in mini-corpus**: ship baked-in (~1MB synthetic,
    compiled into the binary). No download step.
13. **CHAMP-Q**: port with G2. The Python and C++ paths must produce
    bit-equivalent artifacts; deferring breaks artifact compatibility.
14. **PE-QAT**: full trainer in scope. LoRA merge + SmoothQuant `s`
    pre-scale + AdamW + LoRA forward/backward + training loop all in
    C++. Trainer parity with `tools/tessera/pe_qat.py`.

## 1. Operation inventory

The Python surface is ~10k LoC across seven files. Each primitive is
mapped to a C++ target here, with the underlying library call and an
estimated LoC envelope (CPU, F32 only; the row covers the function
itself, not its tests).

### 1.1 Elementwise / reductions (`tools/tessera/_accelerate.py`)

vDSP-shaped wrappers. The C++ port lives in
`tools/quantize/tessera/tessera-vec.h` as a thin header that
dispatches to `cblas_s*` (Linux OpenBLAS), `vDSP_*` (macOS), or a
naive loop (fallback). All entry points take raw `float*` + length
and are `noexcept`.

| Python | C++ symbol | ~LoC | Lib call |
|---|---|---:|---|
| `vDSP_meanv`   | `ts_vec_mean(const float*, int64_t)`           | 25 | `cblas_sasum / n` or `vDSP_meanv` |
| `vDSP_measqv`  | `ts_vec_measqv(const float*, int64_t)`        | 25 | naive + `std::transform_reduce` |
| `vDSP_maxv`    | `ts_vec_max(const float*, int64_t)`           | 20 | `std::max_element` |
| `vDSP_minv`    | `ts_vec_min(const float*, int64_t)`           | 20 | `std::min_element` |
| `vDSP_sve`     | `ts_vec_sum(const float*, int64_t)`           | 20 | `cblas_sasum` |
| `vDSP_vsmul`   | `ts_vec_vsmul(const float*, float, float*, n)`| 25 | `cblas_sscal` or `vDSP_vsmul` |
| `vDSP_vmul`    | `ts_vec_vmul(a, b, out, n)`                   | 25 | naive or `vDSP_vmul` |
| `vDSP_vadd`    | `ts_vec_vadd(a, b, out, n)`                   | 25 | naive or `vDSP_vadd` |
| `vDSP_dotpr`   | `ts_vec_dot(a, b, n)`                         | 20 | `cblas_sdot` |
| `vDSP_mmov`    | `ts_mat_mmov(src, dst, rows, cols, ld_s, ld_d)`| 30 | `std::memcpy` per row |

Total: ~235 LoC. Header-only; no virtual dispatch; same signature on
all backends.

### 1.2 Linalg (`tools/tessera/_flrq_linalg.py`,
`tools/tessera/_dartquant_linalg.py`)

Sits in `tools/quantize/tessera/tessera-linalg.h` /
`tessera-linalg.cpp`. FLRQ uses the stdlib-only reference (Householder
+ power iteration); DartQuant uses the LAPACK-backed
`cblas_*`/`LAPACKE_*` or `Accelerate`'s LAPACK shim on macOS.

| Python | C++ symbol | ~LoC | Lib call |
|---|---|---:|---|
| `random_gaussian(n,seed)` | `ts_gaussian_fill(float*, n, uint64_t seed)` | 30 | `std::normal_distribution` w/ splitmix64 |
| `qr(A)` (Householder) | `ts_qr_householder(const float*, m, n, Q, R)` | 110 | `LAPACKE_sgeqrf` + `sorgqr` |
| `qr_retract(M)` | `ts_qr_retract(M, n)` | 25 | `LAPACKE_sorgqr` after in-place QR |
| `random_orthogonal(n,seed)` | `ts_random_orthogonal(float*, n, seed)` | 35 | `LAPACKE_sgeqrf` on a Gaussian draw |
| `stiefel_project(G, R)` | `ts_stiefel_project(G, R, n)` | 35 | naive (R^T G symmetrise) |
| `qr_orth_step(R, G, lr)` | `ts_qr_orth_step(R, G, lr, n)` | 40 | `ts_stiefel_project` + `ts_qr_retract` |
| `svd_topk` (power iter) | `ts_svd_topk(A, m, n, k, niter, seed, V, sigma)` | 180 | blocked power iter, `cblas_sgemm` |
| `banded_cholesky` (SEPTQ) | `ts_band_chol(H, L, n, bandwidth)` | 110 | naive + `cblas_sdot` |
| `banded_gptq_M` (SEPTQ) | `ts_band_gptq_M(L, M, n, bandwidth)` | 90 | forward-sub with `cblas_sdot` |
| `flrq_bcl` (inner loop) | `ts_flrq_bcl(W, U_basis, V, scale, clip, R_q, n, r, qbits, iters)` | 160 | `cblas_sgemv` for U^T (W - R_q) |

Total: ~815 LoC.

### 1.3 Ternary pack / scales (`tools/tile640/quantize_v3.py`)

Sits in `tools/quantize/tessera/tessera-quant.h` and
`tessera-quant.cpp`. These are the Tile640 hot loops. They are the
core value of the port: what Python does in 30 minutes for an 8B
model the C++ path does in seconds.

| Python | C++ symbol | ~LoC | Notes |
|---|---|---:|---|
| `ternarize_with_acts` | `ts_ternarize_rowmajor(W, n, act_scales, outlier_frac, ternary, outlier_idx, outlier_vals)` | 140 | `np.argsort` -> `std::partial_sort`; threshold = `mean(|W|)` |
| `ternarize_with_acts_mlx` (alias) | (same function; MLX path not ported, native C++ is faster) | 0 | drop |
| `select_repair_residuals` | `ts_select_residuals(W_recon, W_orig, importance, outlier_frac, indices, values, scratch)` | 220 | `std::nth_element` per row; large-matrix row-balanced path |
| `pack_tile640` | `ts_pack_tile640(ternary, out, in, packed_u32, pages_per_row)` | 90 | base-3 u32 words, LSB-first; trits 0/1/2 |
| `compute_scales` (lane + page) | `ts_compute_scales(W, ternary, out, in, page_scales_f16, lane_scales_i8)` | 260 | per-lane mean(|W|) / max lane; `__fp16` intrinsics on ARM |
| `normalized_awq_scale` | `ts_awq_normalize(act_scales, alpha, out)` | 60 | median, clip `[1/256, 256]` |
| `awq_scale_search` (alpha grid) | `ts_awq_search(W, act_scales, outlier_frac, alpha_grid, best_alpha, best_scale, scratch)` | 240 | alpha grid, n_samples row subsample |
| `_synthetic_calibration_input` | `ts_synth_calib_input(act_scales, batch, correlation, seed, X, scratch)` | 110 | Toeplitz banded construction |
| `awq_scale_search_layer_output` | (replaced by `ts_awq_search` with target=layer-output) | 0 | single C++ path |
| `load_imatrix` (npz + gguf) | `ts_imatrix_load(path, std::unordered_map<name, vec>)` | 180 | libgguf read for the GGUF path; flat npz via `mmap` |
| `merge_imatrix_geomean` | `ts_imatrix_geomean(primary, others, out)` | 60 | log-space mean |
| `lookup_acts` | `ts_imatrix_lookup(name_hf, name_gguf, imatrix, out)` | 80 | strip "model." prefixes; fused gate-up split |
| `quantize_2d` (the orchestrator) | `ts_quantize_2d(W, out_dim, in_dim, ...)` | 380 | dispatches LRQ / AWQ / imatrix-mse / exact |
| `quantize_2d_imatrix_mse` | `ts_quantize_imatrix_mse(W, out, in, ...)` | 220 | per-row MSE grid search |
| `quantize_2d_septq` | `ts_quantize_septq(W, out, in, ratio, ...)` | 320 | banded Cholesky + GPTQ-M |
| `quantize_3d` | `ts_quantize_3d(W, n_experts, ...)` | 90 | loop over `ts_quantize_2d` |
| `is_gemma4_sensitive_tensor` | `ts_is_sensitive(tensor_name, arch, extra)` | 30 | substring match; default set |
| `tensor_policy` / `expert_policy_values` | `ts_policy_resolve(policy_json, tensor_name)` | 220 | nlohmann::json walk |

Total: ~2620 LoC. The largest single function is the per-row MSE grid
search in `ts_quantize_imatrix_mse`; it stays under 250 LoC because
the inner search is a fixed-size loop with the candidate scale as the
outer iteration.

### 1.4 Calibration / GA / L5 (`tools/tessera/`)

These are the heavier "search" routines. They live in
`tools/quantize/tessera/tessera-search.h` / `.cpp`. The accelerators
live in the same files; the AWQ-evolve GA has its own state machine
file.

| Python | C++ symbol | ~LoC | Lib call |
|---|---|---:|---|
| `ternarize` (LRQ) | `ts_ternarize_simple(W, ternary, n)` | 50 | `mean(\|W\|)` + sign |
| `ternarize_value` | (inlined in the LRQ forward pass) | 0 | |
| `Adam` (LRQ optimiser) | `ts_adam` (struct, no class) | 90 | naive m/v state, F32 |
| `train_lrq` | `ts_train_lrq(W, X, rank, iters, lr, seed, agg, &result, scratch)` | 320 | `cblas_sgemm` for W*s, W @ X.T |
| `train_flrq` | `ts_train_flrq(W, ranks, projections, seed, ...)` | 380 | sketch + `ts_flrq_bcl` + rank sweep |
| `dartquant_qr_orth` | `ts_dartquant_qr_orth(W, X, X_hat, iters, lr, whip, seed, &result)` | 420 | Stiefel step + asymmetric STE + QR retract |
| `LBFGS` (CHAMP-Q) | `ts_lbfgs` (struct + `step()` method) | 220 | two-loop recursion + Armijo |
| `projected_gradient_descent` | `ts_pgd(closure, x0, n, lr, project, &history, scratch)` | 130 | closures replaced by `std::function` over a flat vector |
| `Candidate` / `evaluate` / `mutate` (AWQ-evolve) | `ts_candidate` + `ts_awq_evaluate_wave` + `ts_awq_mutate` | 540 | ternarize + per-row scale + argpartition |
| `progressive_evaluate_population` (AWQ-evolve) | `ts_awq_progressive_wave(candidates, layers, config, cache, scratch)` | 320 | screen / refine / promote, success-halving |
| `_cached_evaluate` / cache | `ts_awq_score_cache` (keyed by SHA-256 of candidate JSON) | 220 | nlohmann::json to canonical key string |
| `_promote` (MAP-Elites) | `ts_map_elites_promote(scored, keep, diversity_slots, margin)` | 90 | archive_cell bucket sort |
| `archive_cell` (AWQ-evolve) | `ts_awq_archive_cell(candidate, score)` | 30 | 3-axis bin index |
| `allocate_residual_budget` | `ts_allocate_residual_budget(layers, candidates, fallback, budget)` | 160 | sweep over multipliers; greedy knapsack |
| `evolve` (AWQ-evolve driver) | `ts_awq_evolve(layers, generations, population, islands, seed, checkpoint, &best, scratch)` | 460 | island loop, MAP-Elites update, checkpoint JSON |
| `L5 imatrix_magnitude` | `ts_l5_imatrix_magnitude(im, out)` | 40 | peak-normalise |
| `L5 gradient_proxy` | `ts_l5_gradient_proxy(mse_cur, mse_minus, out)` | 50 | `(cur - perturbed)+` peak-normalise |
| `L5 layer_position_prior` | `ts_l5_layer_prior(names, total_layers, out)` | 60 | linear ramp + midpoint for non-block tensors |
| `L5 combine` | `ts_l5_combine(components, weights, out)` | 50 | weighted sum |
| `L5 MomentumEMA` | `ts_l5_ema` (struct) | 60 | decay * prev + (1 - decay) * x |
| `L5 percentile_rank` | `ts_l5_percentile_rank(scores, out)` | 80 | stable sort + tie averaging |
| `L5 pick_top_fraction` / `pick_bottom_fraction` | `ts_l5_pick_top` / `ts_l5_pick_bottom` | 70 | threshold on percentile rank |
| `L5 step_up` / `step_down` / `ladder_index` | `ts_l5_ladder_index(qtype)` + 2 small helpers | 60 | static `BIT_LADDER` array |
| `L5 expected_mse_delta` | `ts_l5_expected_mse_delta(...)` | 50 | `2^(-2 * delta_bits)` dampening |
| `L5 SensitivityScorer` + `RequantPlanner` + `OrchestratorLoop` | `ts_l5_orchestrator_run(l4_report, imatrix, config, &plan_history, scratch)` | 580 | orchestrates the above; applier pluggable |
| `L5 write_sidecar` | `ts_l5_write_sidecar(history, final_qtype, path)` | 110 | nlohmann::json document |
| `_scalar_string` (npz header helper) | `ts_npz_read_scalar_string(path, key, &out)` | 30 | lazy npz header walk |
| `load_layer` (npz bundle) | `ts_layer_load(path, max_rows, max_tokens, &layer, scratch)` | 180 | npz with F32 weight + optional activations/observers |
| `bundle_digest` | `sha256_file(path)` | 25 | openssl or hand-rolled |

Total: ~5365 LoC.

### 1.5 Sidecar / L1.5 / L3 I/O

These are producers and consumers. The L1.5 reference sidecar is a
F16-cast writer that the dequant sidecar already covers; the L3 v3
producer / reader pair lives in `common/tessera-debug/` and is mostly
C++ already.

| Python | C++ symbol | ~LoC | Notes |
|---|---|---:|---|
| `l3_sidecar_v3_reader.py` (read) | `tessera_sidecar::v3::read(path, &out)` | 250 | mirror of the writer; dispatch on version |
| `l3_sidecar_v3_helper.cc` (write hook) | already C++ (`common/tessera-debug/tessera-debug.cpp`) | 0 | hooked at dequant; no port needed |
| `_flrq_linalg.random_gaussian` (npz flatten) | `ts_npz_open(path, mode, &handle)` | 60 | minimal npz reader (zlib + npy v1/v2/v3) |
| `_load_imatrix_npz` | `ts_imatrix_load_npz(path, &out)` | 90 | F32 sum2 + int64 counts; sqrt(E[x^2]) |
| `_load_imatrix_gguf` | `ts_imatrix_load_gguf(path, &out)` | 110 | GGUFReader; per-expert split |

Total: ~510 LoC.

### 1.6 L1.5 reference read + v3 sidecar writer

`docs/w4a4-calibration-design.md` already nails the L1.5 read: at
quantize time, not in the C++ dequant kernel. The C++ side reads the
FP16 reference (from the v3 sidecar `*.act.dequant.f32` payload;
both L1 and L1.5 share the v3 file format) before the per-tile
quantize. The new function is `ts_l15_load_reference(path, &out_f16)`
which is a thin wrapper over the v3 reader that returns the F16
reference (already F32-cast by the sidecar writer) for the F16
outlier pass described in the L1.5 design.

The v3 sidecar writer is already C++ in
`common/tessera-debug/tessera-debug.cpp`. The L1.5 activation path
extends the existing `open_fp16_reference_writer` /
`write_fp16_reference_row` / `set_fp16_reference_row_meta` /
`close_fp16_reference_writer` family by adding the F16 reference
read API:

| C++ symbol | ~LoC | Notes |
|---|---:|---|
| `tessera_sidecar::v3::read_reference(path, &out)` | 110 | v3 reader; F16-casted to F32 by the writer, so this is F32 |
| `tessera_sidecar::v3::read_for_tensor(path, &out)` | 90 | same, but for a single tensor |
| `tessera_sidecar::v3::read_f16(path, &out)` | 80 | the v3 path with `dtype=DEQUANT_DTYPE_F16` |

### 1.7 GGUF read / write

`gguf-py` is the Python library. C++ uses the in-tree `gguf.h` /
`libgguf`. The functions touched:

| Python | C++ | ~LoC |
|---|---|---:|
| `GGUFWriter.add_key_value` | `gguf_set_val_u32/str/...` (already exposed) | 0 |
| `GGUFWriter.add_string` | `gguf_set_val_str` | 0 |
| `GGUFWriter.add_array` | `gguf_set_arr_*(...)` | 0 |
| `GGUFWriter.add_tensor(name, np.ndarray)` | `gguf_set_tensor(ctx, name, data, nrows, n_per_row, type)` | 0 (already in libgguf) |
| `GGUFReader.fields` walk | `gguf_get_key(ctx, name)` + type dispatch | 0 |
| `copy_gguf_metadata` | `ts_copy_metadata(reader, writer, excluded_set)` | 80 |
| `add_tessera_metadata` | `ts_add_tessera_metadata(writer, calibrated, unified, ...)` | 180 |
| `apply_gemma4_metadata_overrides` | `ts_apply_gemma4_overrides(writer, reader, arch, sliding_window)` | 70 |

Total: ~330 LoC of glue; everything else is `libgguf` already.

### 1.8 CHAMP-Q

`tools/tessera/champq_permute.py` is currently a separate
helper. The C++ port lives in `tessera-search.cpp` and is ~250 LoC
of:
- `ts_champq_permutation(W, act_scales, &perm, &inverse, scratch)`
- `ts_apply_champq_permutation(W, perm, scratch)`
- `ts_invert_champq_permutation(perm, &inv)`
- `ts_decode_q_to_weight(q, out, in, scratch)` (Tile640 unpack +
  per-lane scale + per-page scale + outliers)

Total: ~350 LoC. Used in the 2D + 3D paths only; not the canonical
quantizer.

## 2. File-by-file C++ target

The Python files are partitioned across new C++ files under
`tools/quantize/tessera/` and a small set of extensions to
`ggml-quants.c` and `common/`. The single entry point is
`tools/quantize/quantize.cpp`, which grows a new dispatch table for
Tessera-aware quantization.

### 2.1 New C++ files (under `tools/quantize/tessera/`)

| Python file | LoC | C++ file(s) |
|---|---:|---|
| `tools/tile640/quantize_v3.py` (all quantize primitives, except ones below) | 4165 | `tessera-quant.{h,cpp}` (3.0k) + `tessera-vec.{h,cpp}` (0.5k) |
| `tools/tessera/per_tensor_calibrate.py` (LRQ + DartQuant) | 1944 | `tessera-search.{h,cpp}` (2.5k; both `train_lrq` + `dartquant_qr_orth`) |
| `tools/tessera/per_tensor_calibrate.py` (FLRQ) | (shared) | `tessera-search.cpp` (`train_flrq`) + `tessera-linalg.{h,cpp}` (0.8k) |
| `tools/tessera/awq-evolve.py` | 1197 | `tessera-awq.{h,cpp}` (1.6k) |
| `tools/tessera/l5_orchestrator.py` | 972 | `tessera-l5.{h,cpp}` (1.0k) |
| `tools/tessera/l5_metrics.py` | 362 | `tessera-l5.cpp` (`imatrix_magnitude`, `gradient_proxy`, `layer_position_prior`, `combine`, `MomentumEMA`, `percentile_rank`, `pick_top_fraction`, `expected_mse_delta`, ladder stepping) |
| `tools/tessera/_flrq_linalg.py` | 384 | `tessera-linalg.cpp` (sketch + power iter) |
| `tools/tessera/_dartquant_linalg.py` | 202 | `tessera-linalg.cpp` (Householder + Stiefel) |
| `tools/tessera/_champq_lbfgs.py` | 366 | `tessera-lbfgs.{h,cpp}` (0.4k) |
| `tools/tessera/_accelerate.py` (vDSP wrappers) | 688 | `tessera-vec.h` (vDSP shim); only the surface that's actually used gets ported (~120 LoC; the rest is the ctypes binding code, not ported) |
| `tools/tessera/l3_sidecar_v3_reader.py` (read path) | (already in tree) | `common/tessera-debug/tessera-sidecar-v3.{h,cpp}` (0.5k) |
| `tools/tessera/pe_qat.py` (LoRA + SmoothQuant) | (separate) | not in this port scope; PE-QAT is G2 |
| `tools/tessera/champq_permute.py` (permutation) | (separate) | `tessera-search.cpp` (`ts_champq_*`, 0.35k) |

### 2.2 Extensions to `ggml-quants.c` and `ggml-common.h`

The Tile640 runtime already exists in `ggml/src/ggml.c` (operators
`GGML_OP_TILE640_MATMUL`, `GGML_OP_TILE640_MATMUL_ID`,
`GGML_OP_TILE640_GET_ROWS`, `GGML_OP_TILE640_DEQUANT`) and in
`ggml/src/ggml-metal/ggml-metal.metal` (kernel
`kernel_TILE640_MATMUL`, etc.). The dequant side of Tile640 is
already there. The port adds the quantize side and registers the new
type.

Files to edit:

- `ggml/include/ggml.h`:
  - Append to `enum ggml_type` (after `GGML_TYPE_Q2_0 = 42`,
    before `GGML_TYPE_COUNT = 43`):
    ```
    GGML_TYPE_TESSERA_T640 = 43,    // 2D ternary w/ F16 page + I8 lane + F16 outliers
    GGML_TYPE_TESSERA_T640_3D,      // 3D expert variant; not a separate physical
                                    //   encoding, see TESSERA_T640_3D marker
                                    //   below for the layout convention
    GGML_TYPE_COUNT = 45,
    ```
    Note: the 3D expert variant is encoded as a flattened
    `(n_experts * out_dim * in_dim)` 2D block under the same physical
    type; the new enum entry is a flag the runtime uses to pick
    `GGML_OP_TILE640_MATMUL_ID` instead of `GGML_OP_TILE640_MATMUL`.
    The physical encoding is the same; we just need the enum slot to
    keep the dispatch on `ggml_type`.

- `ggml/include/ggml-common.h` (or `ggml/src/ggml-common.h`):
  - Add `TILE640_PAGE_SIZE = 640`, `TILE640_LANE_SIZE = 20`,
    `TILE640_LANES_PER_PAGE = 32`, `TILE640_WORDS_PER_PAGE = 32` as
    preprocessor constants. The C++ side already has these in
    `ggml-metal.metal`; they move up to the shared header so the
    quantizer can see them.
  - Add the block structs:
    ```cpp
    #define TILE640_PACKED_WORDS_PER_ROW(_in_dim) \
        ((((_in_dim) + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE) * TILE640_WORDS_PER_PAGE)
    #define TILE640_PAGE_SCALES_PER_ROW(_in_dim) \
        ((((_in_dim) + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE))
    #define TILE640_LANE_SCALES_PER_ROW(_in_dim) \
        ((((_in_dim) + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE) * TILE640_LANES_PER_PAGE)

    // Each row's quantization is split into 6 GGUF tensors:
    //   weight_packed           i32[out, pages_per_row * WORDS_PER_PAGE]
    //   weight_page_scales      f16[out, pages_per_row]
    //   weight_lane_scales      i8 [out, pages_per_row * LANES_PER_PAGE]
    //   weight_outlier_row_offsets i32[out + 1]
    //   weight_outlier_cols     i32[total_outliers]
    //   weight_outlier_vals     f16[total_outliers]
    //   weight_act_scale (optional) f16[in]
    ```

- `ggml/src/ggml-quants.c`:
  - New `block_tessera_t640` (forward declaration; the actual block
    is a 2D pack that the runtime treats as opaque 32-bit words).
  - New `quantize_row_tessera_t640_ref(...)` and
    `dequantize_row_tessera_t640(...)`. The dequant path is
    straight-line: scale by page, then by lane, then unpack trits.
    The quantize path delegates to the C++ `ts_quantize_2d` and is
    wired in via the `from_float_ref` callback in
    `ggml_type_traits[]` (next bullet).
  - New `vec_dot_tessera_t640_q8_0(...)` for the matmul side; the
    actual matmul is in `ggml.c` `ggml_tile640_matmul` already.

- `ggml/src/ggml.c`:
  - Add a `[GGML_TYPE_TESSERA_T640]` entry in `type_traits[]`:
    ```cpp
    [GGML_TYPE_TESSERA_T640] = {
        .type_name      = "tessera_t640",
        .blck_size      = TILE640_PAGE_SIZE,
        .type_size      = sizeof(int32_t),   // logical; the pack is multi-tensor
        .is_quantized   = true,
        .to_float       = (ggml_to_float_t)  dequantize_row_tessera_t640,
        .from_float_ref = (ggml_from_float_t) quantize_row_tessera_t640_ref,
    },
    ```
  - Same for `[GGML_TYPE_TESSERA_T640_3D]`. The
    `from_float_ref` for both is a thin shim that calls
    `ts_quantize_2d` (or `ts_quantize_3d` for the 3D variant) and
    writes the 6 component tensors through `gguf_set_tensor` on the
    output context.

### 2.3 Extensions to `common/`

- `common/arg.cpp` / `common/arg.h`: new `common_arg` instances for
  each Tessera flag (section 3).
- `common/json.hpp` (vendor/nlohmann/json.hpp): no changes expected;
  only consumers. The flag schema and the calibration policy are
  parsed with `nlohmann::json`.

### 2.4 Extensions to `libgguf`

The TESSERA_* type enum entries are already declared via the
extension to `ggml.h` above; `libgguf` (the writer/reader) reads
those enum values directly. The C++ quantizer writes the type as
`GGML_TYPE_TESSERA_T640` and the GGUF metadata fields as
`tessera.*`; no new libgguf function is required. The 6 component
tensors are written with the existing
`gguf_add_tensor(ctx, name, data, n_dims, dims, GGML_TYPE_I32)` etc.

### 2.5 Single entry point

`tools/quantize/quantize.cpp` is the single entry point. The current
dispatch (`try_parse_ftype` -> `llama_model_quantize`) is preserved
for the `--tessera-mode=off` path. A new `--tessera-mode` flag
selects:

- `off` (default when stock K-quant name is given): existing path
- `default` (new default when no stock name is given): the
  Tessera pipeline. The dispatch becomes
  `llama_model_quantize_tessera(fname_inp, fname_out, &tessera_params)`
  which:
  1. Loads the source GGUF
  2. Decides which steps to run based on flag presence
  3. Runs calibration (mini-corpus or `--calib-corpus`) if needed
  4. Runs the GA (`--tessera-evolve-iters` / `--tessera-evolve-seed`)
  5. Walks the tensors, calling `ts_quantize_2d` / `ts_quantize_3d`
     per tensor
  6. Writes the 6 component tensors + `tessera.*` metadata
- `calibrate-only` (G6): only run the calibration pass, write a
  `llama.speculative.calibration-policy.v1` JSON next to the input
- `evolve-only` (G6): only run the GA on an existing policy

The flag-driven default is the architecture decision (item 7 of
the locked decisions): no imatrix + no policy + no corpus = the
calibrator runs on a built-in mini-corpus, the GA runs, the
quantizer uses the GA's policy.

## 3. CLI surface

`llama-quantize`'s existing flags are untouched. New flags are
appended in three groups.

### 3.1 Stock K-quant path (unchanged)

```
llama-quantize [--help] [--allow-requantize] [--leave-output-tensor]
               [--pure] [--imatrix] [--include-weights] [--exclude-weights]
               [--output-tensor-type] [--token-embedding-type] [--tensor-type]
               [--tensor-type-file] [--prune-layers] [--keep-split]
               [--override-kv] [--dry-run]
               model-f32.gguf [model-quant.gguf] type [nthreads]
```

The existing `try_parse_ftype` accepts `Q4_K`, `Q5_K`, ...; when one
of these is given, the new code sets `tessera_mode = off` and the
stock K-quant path runs as today.

### 3.2 Tessera-as-default flags (new)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--tessera-mode` | enum{off, default, calibrate-only, evolve-only} | `default` | selects the dispatch; `default` is the new behaviour when no stock K-quant name is given |
| `--tessera-imatrix` | path | (none) | imatrix .npz / .gguf path; equivalent to legacy `--imatrix` but Tessera-aware |
| `--tessera-policy` | path | (none) | `llama.speculative.calibration-policy.v1` JSON; if absent and no imatrix, the calibrator runs |
| `--tessera-policy-out` | path | `<output>.tessera-policy.json` | where the policy is written when the GA runs |
| `--tessera-ga-checkpoint` | path | `<output>.tessera-ga.json` | GA checkpoint for resume |
| `--calib-corpus` | path | (none) | a directory of `.npy`/`.npz` calibration activations; falls back to the built-in mini-corpus when absent |
| `--calib-corpus-out` | path | (none) | writes the resolved mini-corpus to this path on first run (deterministic) |
| `--tessera-evolve-seed` | uint64 | `0` | GA seed; bit-identical policy across runs |
| `--tessera-evolve-iters` | int | `8` | GA generations |
| `--tessera-evolve-islands` | int | `4` | GA islands |
| `--tessera-evolve-population` | int | `16` | GA population per island |
| `--tessera-evolve-only` | flag | off | only run the GA; do not quantize |
| `--tessera-calibrate-only` | flag | off | only run calibration; do not run GA or quantize |
| `--tessera-outlier-frac` | float | `0.005` | default outlier fraction |
| `--tessera-default-outlier-frac` | float | `0.001` | outlier fraction for pass-through / sensitive |
| `--tessera-awq-alpha` | float or `auto` | `auto` | per-tensor AWQ alpha; `auto` = per-tensor search |
| `--tessera-awq-clip` | float | `1.0` | per-row clip; `0.7`-`1.0` |
| `--tessera-ternary-threshold` | float or `auto` | `auto` | per-row `mean(\|W\|)` multiplier |
| `--tessera-range-selection` | enum{legacy, imatrix-mse, septq} | `legacy` | range selection |
| `--tessera-septq-ratio` | float | `0.5` | SEPTQ quantize fraction |
| `--tessera-septq-bandwidth` | int | `32` | SEPTQ banded Cholesky bandwidth |
| `--tessera-septq-importance` | enum{quant_error_h, inv_abs_w, inv_cdf, hybrid} | `quant_error_h` | SEPTQ importance mode |
| `--tessera-septq-lambda` | float | `0.0` | SEPTQ hybrid lambda |
| `--tessera-champq` | flag | off | enable CHAMP-Q permutation |
| `--tessera-champq-policy-out` | path | `<output>.champq-policy.json` | CHAMP-Q policy file (debug only) |
| `--tessera-pe-qat-policy` | path | (none) | PE-QAT policy (G2) |
| `--tessera-gemma4-sensitive-patterns` | list of strings | (none) | extra sensitive patterns (substrings) |
| `--tessera-nthreads` | int | `$nthread` | threads for calibration/GA/quantize; defaults to the legacy `nthread` positional |
| `--tessera-chunk-size` | int | `65536` | per-tensor chunk size in elements |

The 6 metadata flags in the policy schema (`--tessera-coverage`,
`--tessera-coverage-family`, etc.) are encoded as a single
`--tessera-coverage-json` path that points at the
`tessera.coverage.*` block; the schema is internal.

### 3.3 Backward-compat matrix

| User invocation | Behaviour |
|---|---|
| `llama-quantize in.gguf Q4_K_M` | stock K-quant path; `--tessera-mode=off` implied |
| `llama-quantize in.gguf out.gguf TESSERA` (or no type) | Tessera default flow; auto-decide calibration / GA / quantize from flags |
| `llama-quantize --tessera-mode=off in.gguf out.gguf TESSERA` | stock K-quant; type `TESSERA` rejected as `TESSERA_T640` would be the Tessera type name (added to `try_parse_ftype`) |
| `llama-quantize in.gguf out.gguf TESSERA_T640` | Tessera 2D variant |
| `llama-quantize in.gguf out.gguf TESSERA_T640_3D` | Tessera 3D expert variant; same physical encoding, runtime uses `GGML_OP_TILE640_MATMUL_ID` |
| `llama-quantize --tessera-mode=off --imatrix im.gguf in.gguf Q4_K` | unchanged; stock K-quant with imatrix |
| `llama-quantize --tessera-imatrix im.npz --calib-corpus corpus/ in.gguf TESSERA_T640` | Tessera with explicit imatrix; calibration skipped |
| `llama-quantize in.gguf TESSERA_T640` (no imatrix, no policy, no corpus) | calibrator runs on the built-in mini-corpus; GA runs; quantize |

The legacy `--imatrix` flag is preserved as a synonym for
`--tessera-imatrix`; legacy `--tensor-type` / `--tensor-type-file`
are preserved and apply to the per-tensor override in the
Tessera-style path too (the runtime reads the same override table).

## 4. TESSERA_* GGUF type family

### 4.1 Type enum entries

Added to `enum ggml_type` in `ggml/include/ggml.h`:

```
GGML_TYPE_TESSERA_T640    = 43,   // 2D Tile640 ternary; 6 component tensors
GGML_TYPE_TESSERA_T640_3D = 44,   // 3D expert bank; 6 component tensors
GGML_TYPE_COUNT           = 45,
```

`TESSERA_T640_3D` is a marker: the physical encoding is identical to
`TESSERA_T640` (the 3D bank is reshaped to a 2D `(n_experts *
out_dim, in_dim)` for quantize/dequant); the runtime uses the marker
to dispatch to `GGML_OP_TILE640_MATMUL_ID` instead of
`GGML_OP_TILE640_MATMUL`. The marker is in the type enum so the
`ggml_tensor` carries it through and the dispatch in
`ggml_compute_forward` (in `ggml/src/ggml.c`) can branch on the
type without inspecting tensor names.

### 4.2 GGUF metadata schema

`tools/tile640/quantize_v3.py:393-498` defines the existing
`tessera.*` metadata. The C++ port keeps the schema verbatim; the
Python quantizer is the source of truth today. The complete list:

| Key | Type | Source |
|---|---|---|
| `tessera.name` | string | `add_tessera_metadata` |
| `tessera.version` | uint32 | (always 1) |
| `tessera.profile` | string | `TSQ-T640-AWQ-SR` or `TSQ-T640-SR` |
| `tessera.features` | string[] | features list |
| `tessera.core.type` | string | `balanced-ternary` |
| `tessera.core.levels` | uint32 | 3 |
| `tessera.layout` | string | `T640` |
| `tessera.layout.version` | uint32 | 1 |
| `tessera.layout.page_size` | uint32 | 640 |
| `tessera.layout.lane_size` | uint32 | 20 |
| `tessera.layout.lanes_per_page` | uint32 | 32 |
| `tessera.layout.words_per_page` | uint32 | 32 |
| `tessera.scale.page_type` | string | `bf16` |
| `tessera.scale.lane_type` | string | `int8` |
| `tessera.residual.type` | string | `row-sparse` |
| `tessera.residual.value_type` | string | `f16` |
| `tessera.sensitive.exact` | bool | true |
| `tessera.calibration.imatrix` | bool | (imatrix present) |
| `tessera.calibration.awq` | bool | (calibrated) |
| `tessera.calibration.unsloth_prior` | bool | (Unsloth bridge) |
| `tessera.calibration.imatrix_paths` | string[] | imatrix source paths |
| `tessera.calibration.imatrix_merge_policy` | string | merge policy |
| `tessera.calibration.imatrix_source_count` | uint32 | count |
| `tessera.coverage` | string | `all-learned-tensors` |
| `tessera.passthrough` | bool | false |
| `tessera.unified` | bool | (mmproj / MTP / component) |
| `tessera.dataset.epoch` | uint32 | epoch receipt |
| `tessera.dataset.model_fingerprint` | string | epoch receipt |
| `tessera.dataset.evidence_digest` | string | epoch receipt |
| `tessera.dataset.observer_calibration_tokens` | uint64 | epoch receipt |
| `tessera.dataset.acceptance_observations` | uint64 | epoch receipt |
| `tessera.source.epoch` | uint32 | source receipt |
| `tessera.source.digest` | string | source receipt |
| `tessera.source.artifact_digest` | string | source receipt |
| `tessera.source.logical_bytes` | uint64 | source receipt |
| `tessera.source.tensor_count` | uint64 | source receipt |
| `tessera.source.parent_digest` | string | source receipt |
| `tessera.source.training_corpus_epoch` | uint32 | source receipt |
| `tessera.source.training_corpus_digest` | string | source receipt |
| `tessera.source.telemetry_epoch` | uint32 | source receipt |
| `tessera.gemma4.sliding_window_override` | uint32 | gemma 4 only |
| `tessera.range_selection` | string | `legacy` / `imatrix-mse` / `septq` |
| `tessera.imatrix_mse.norm` | float32 | MSE p-norm |
| `tessera.imatrix_mse.grid` | uint32 | grid steps |
| `tessera.imatrix_mse.maxshrink` | float32 | shrink floor |
| `tessera.septq.ratio` | float32 | quantize fraction |
| `tessera.septq.iterations` | uint32 | iterations |
| `tessera.septq.hessian_mode` | string | `diagonal` / `banded` |
| `tessera.septq.hessian_bandwidth` | uint32 | bandwidth |
| `tessera.septq.importance_weight` | string | mode |
| `tessera.septq.importance_lambda` | float32 | hybrid lambda |
| `tessera.awq_search_target` | string | `per-row` / `layer-output` |
| `tessera.calibration_activations_source` | string | path |
| `tessera.awq_synthetic_batch` | uint32 | synth batch |
| `tessera.awq_synthetic_correlation` | float32 | synth correlation |
| `tessera.coverage.family` | string | coverage family |
| `tessera.coverage.architecture` | string | coverage arch |
| `tessera.coverage.manifest_sha256` | string | manifest digest |
| `tessera.coverage.receipt` | string | compact receipt JSON |

The new `tessera.ga.*` block added by the C++ port:

| Key | Type | Source |
|---|---|---|
| `tessera.ga.generations` | uint32 | `--tessera-evolve-iters` |
| `tessera.ga.population` | uint32 | `--tessera-evolve-population` |
| `tessera.ga.islands` | uint32 | `--tessera-evolve-islands` |
| `tessera.ga.seed` | uint64 | `--tessera-evolve-seed` |
| `tessera.ga.progressive` | bool | (always true) |
| `tessera.ga.checkpoint_path` | string | checkpoint path |
| `tessera.ga.provenance` | string | compact JSON of the GA archive |

### 4.3 Sidecar JSON shape

Two sidecar files per tensor. The L1 / L1.5 schema is
`llama.tessera.dequant-sidecar.v1` (existing; see
`common/tessera-debug/tessera-debug.h` for the binary layout).
The provenance sidecar JSON is unchanged from the v3 producer
above:

```json
{
  "model": "...",
  "calibration_corpus": "...",
  "calibration_corpus_hash": "...",
  "kernel_version": "...",
  "l1_sidecar_version": 3,
  "imatrix_version": 2,
  "created_at": "2026-01-15T12:34:56Z",
  "tessera_main_tip": "abc1234",
  "sidecar_path": "/path/to/tensor.dequant.f32",
  "sidecar_kind": "dequant"  // or "fp16_reference"
}
```

The C++ port does not add new fields. The `tessera_main_tip` and
`kernel_version` are baked in by CMake at configure time and live in
`common/tessera-debug/tessera-build-info.h` (already there).

### 4.4 Provenance stamping

Already C++-native: the v3 sidecar writer stamps
`kernel_version` (from `TESSERA_KERNEL_VERSION` in
`tessera-build-info.h`) and `tessera_main_tip` (from
`TESSERA_MAIN_TIP`). The C++ quantize path adds `imatrix_version`
(2, from `tessera_imatrix_version()`) and `l1_sidecar_version`
(3, from `DEQUANT_FILE_VERSION`). No new code is required in the
sidecar writer; the C++ quantize path calls into the existing
`tessera_debug::open_*_writer` API.

The calibration policy (when written by the C++ port) carries the
same `tessera_main_tip` under `per_tensor_calibration.timestamp`
and a new `per_tensor_calibration.kernel_version` field, set from
`TESSERA_KERNEL_VERSION`.

## 5. L1.5 reference integration

The L1.5 reference is the FP16 representation of the source
weight. The v3 sidecar already writes it via
`open_fp16_reference_writer` / `write_fp16_reference_row` when
`LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4` is set. The C++ quantizer
reads it back at quantize time, not in the dequant kernel.

### 5.1 Where the FP16 read happens

`ts_l15_load_reference(dir, tensor_name, &out_f16)` is called from
`ts_quantize_2d` / `ts_quantize_3d` at the top, before the
ternarize step. The function reads the v3 sidecar at
`<dir>/<tensor_name>.act.dequant.f32` (the suffix is the v3 writer's
`.act.dequant.f32`; the data block is F32-cast F16 from the
runtime). The out param is a dense F16 row-major buffer with the
same shape as the source weight.

The C++ dequant kernel does NOT call this function. The sidecar is
read once, on quantize. The runtime dequant path is unchanged
(`dequantize_row_tessera_t640` is the same F32 output as today).

### 5.2 How the v3 sidecar is written

The existing v3 writer is sufficient. The C++ quantizer does not
need to write sidecar files itself; the runtime dequant path
already writes them. The dequant hook in
`ggml/src/ggml-cpu/arch/arm/quants.c` (or wherever the dequant
helper is invoked) calls `tessera_debug::write_dequant_row` /
`write_fp16_reference_row` per row. The C++ quantize path simply
reads what the runtime wrote.

The configuration: `--tessera-dequant-dir <path>` and
`--tessera-dequant-mode w4a4` are passed to the runtime at
inference. The quantize path also takes them, to locate the
L1 / L1.5 sidecar dir.

### 5.3 How the imatrix v2 carries the F16 outlier values

The imatrix v2 is a per-tensor observer, not a per-element
outlier mask. It carries `sum2` (F32 sum of x^2 per input channel)
and `counts` (int64 sample count), from which the per-channel RMS
is derived (`sqrt(E[x^2])`). The F16 outlier values are NOT in the
imatrix; they live in the L1.5 reference sidecar.

The GA reconstructs LLM.int8 from imatrix alone (no F16 values
needed): the per-channel RMS gives the outlier threshold `|x| > 6.0`
proxy, and the GA tunes the AWQ alpha to make the per-tensor
ternary MSE well-conditioned. The L1.5 reference is consumed only
by the A/B harness in `tools/tessera/l3_outlier_report.py` /
`l3_hessian_trace.py`; it is not on the GA's hot path.

### 5.4 How the GA reconstructs LLM.int8 from imatrix

`awq_scale_search` in `quantize_v3.py:1302` already encodes the
LLM.int8 intuition: with `act_scales` (imatrix RMS) and a per-tensor
alpha grid, the GA searches for the alpha that minimises the
ternary-quant layer-output MSE. The per-channel RMS is the
importance signal; the GA does not need the F16 outlier values
themselves. The C++ port's `ts_awq_search` reproduces this exactly.

## 6. Phased G0-G6 implementation plan

The 7-10 day budget splits into 7 phases. G0 ships first; G6 last.
LoC estimates include header guards, but no tests; tests are
written alongside but counted against the test budget separately.

### G0 - GGUF type registration

Goal: stock `llama-quantize` builds with the new `TESSERA_T640` /
`TESSERA_T640_3D` enum entries and a no-op `from_float_ref` /
`to_float`. The quantize dispatch is not yet wired in; this phase
is "compile-clean and ABI-clean".

LoC: ~120 (header + ggml.c trait + ggml-quants.c stubs + gguf-doc)
Dependency: none.
Acceptance: `llama-quantize --help` still works; new types show in
`ggml_type_name` round-trip.

### G1 - quantize_2d + AWQ

Goal: the 2D quantize path lands. `ts_quantize_2d` handles the
non-`septq`, non-`imatrix-mse`, non-`lrq`, non-`pe_qat` paths
(legacy + AWQ). The dispatch in `llama-quantize` is wired in
behind `--tessera-mode=default`. The GA is NOT yet ported; the
calibration is loaded from a `--tessera-policy` JSON only.

LoC: ~3200 (tessera-vec.h, tessera-quant.h/.cpp, libgguf glue, the
default-flow dispatch in `quantize.cpp`).
Dependency: G0. AWQ uses the existing GA output JSON from
`tools/tessera/awq-evolve.py`; the Python GA stays the source of
truth for the search step.
Acceptance: `llama-quantize in.gguf out.gguf TESSERA_T640
--tessera-policy policy.json` produces a bit-equivalent GGUF to the
Python quantizer on the smoke-test set.

### G2 - LRQ / SEPTQ / CHAMP-Q / DartQuant / FLRQ / PE-QAT

Goal: the other quantize modes are ported.

> Alignment (2026-07-30): these six are regime experts, not competing
> modes. Add a regime router (~150-250 LoC) that picks the expert per
> tensor from the regime descriptors already in the imatrix v2 and
> `tensor_families`: high kurtosis / massive outliers (esp. `down_proj`)
> -> rotation (DartQuant) + permutation; high spectral compactness ->
> low-rank (FLRQ / LRQ), gated by a cheap effective-rank descriptor;
> attention Q/K -> Hessian-mask expert (SEPTQ); MoE expert FFNs ->
> lighter expert; default -> AWQ diagonal scaling. This is the operative
> use of `tensor_families`. CHAMP-Q permutation stays closed-form in v1;
> relax it (Gumbel-Sinkhorn / differentiable sorting) only if it enters
> the GA search space. See research-alignment-2026-07-30.md Section 4.2.

LoC: ~4500 (tessera-linalg.h/.cpp for FLRQ + DartQuant + Stiefel +
SVD, tessera-lbfgs.h/.cpp for CHAMP-Q + L-BFGS, additional
quantize_2d branches in tessera-quant.cpp, pe_qat LoRA merge).
Dependency: G1 (the dispatch table from G1 is extended).
Acceptance: each mode produces the same GGUF as the Python
quantizer on a 4-tensor smoke test.

### G3 - TESSERA_* writer

Goal: the 6 component tensors + `tessera.*` metadata are written by
the C++ path, replacing the `GGUFWriter.add_tensor` calls in
`quantize_v3.py:3835-3969` and the metadata writes in
`add_tessera_metadata` / `apply_gemma4_metadata_overrides`.

LoC: ~700 (libgguf glue for the 6 components; metadata writer
function; sidecar info-file writer; calibration-policy writer for
`--tessera-calibrate-only`).
Dependency: G1.
Acceptance: a Python-side `GGUFReader` reads back the same `tessera.*`
metadata and the same 6 component tensor shapes as the Python
quantizer.

### G4 - GA in C++

Goal: `awq-evolve.py` is ported to `tessera-awq.cpp` with the
island GA, MAP-Elites archive, progressive evaluation, and the
checkpoint JSON. Determinism is bit-for-bit across runs at the
same seed.

> Alignment (2026-07-30): two refinements. (1) The MAP-Elites archive
> cell (`ts_awq_archive_cell`, currently a generic 3-axis bin index)
> uses the regime descriptors as its axes: (kurtosis bucket,
> effective-rank bucket, tensor-family / modality bucket). The archive
> then stores the best reconstruction-knob config per regime cell.
> (2) The stated GA objective is `Sum_l alpha_l * t_l^2` (Linearity
> Theorem), where `t_l^2` is the relative per-tensor reconstruction
> error. Production `t_l^2` is evaluated against the L1 kernel-dequant
> output (== runtime-aware-pipeline L6); the offline ternary MSE is the
> stand-in proxy used until L1 lands. `alpha_l` are estimated once per
> model and cached. This makes L1 (runtime-aware-pipeline.md) a
> prerequisite for G4-done. See research-alignment-2026-07-30.md
> Sections 4.2 and 6.

LoC: ~1600 (tessera-awq.h/.cpp; the AWQ-evolve lib is the bulk).
Dependency: G1 (the AWQ ternary reconstruct uses `ts_compute_scales`
and `ts_pack_tile640`).
Acceptance: `tessera-awq-evolve --seed 640 --generations 8` produces
a `policy.json` byte-equivalent to `awq-evolve.py --seed 640
--generations 8` on the smoke-test bundle.

### G5 - L1.5 + sidecar v3 producer

Goal: the L1.5 reference read is wired into `ts_quantize_2d`; the
v3 sidecar producer is called from the runtime (it already is in
the C++ runtime; this phase adds the read path).

LoC: ~250 (tessera-sidecar-v3.h/.cpp in common/tessera-debug;
ts_l15_load_reference wrapper; integration into ts_quantize_2d).
Dependency: G1.
Acceptance: a runtime-generated v3 sidecar is read back by the
quantizer; the F16 reference matches the F16-cast of the source
weight to within 1 ULP for each position.

### G6 - A/B harness + E2E probe

Goal: the L5 orchestrator and the A/B harness are ported. The C++
quantize path can be run end-to-end on a real model with no
external Python tool.

LoC: ~1200 (tessera-l5.h/.cpp; l5_orchestrator.py's main loop
becomes `llama-quantize --tessera-mode=evolve-only` /
`--tessera-mode=calibrate-only`; the A/B harness is a thin
wrapper).
Dependency: G3 + G4 + G5.
Acceptance: `llama-quantize --tessera-mode=default in.gguf
--calib-corpus corpus/ --tessera-evolve-iters 4` produces a GGUF
that loads in `llama-cli` and produces plausible output on the
smoke-test prompts.

> Alignment (2026-07-30): the acceptance is sharpened into the
> novelty-boundary gate, and G6 now also depends on runtime-aware-pipeline
> L1 + L6. On held-out tensors, the regime-routed kernel-fidelity
> composite (`Sum_l alpha_l * t_l^2` against L1 kernel output) must beat
> the best single proxy (AWQ-only, rotation-only, low-rank-only,
> Hessian-mask-only) at the same bit budget, on both kernel-fidelity
> `t_l^2` and end-to-end PPL (the L4 probe). Separately, measure the
> ranking disagreement between the offline ternary-MSE proxy and the
> kernel-direct fitness and report it; if it is near zero, the
> kernel-fidelity contribution is null and the novelty reduces to
> routing. See research-alignment-2026-07-30.md Section 5.

### Total

~11570 LoC of C++ across the 7 phases. Test code is separate
(~3000-4000 LoC, distributed per phase). The Python toolchain
remains the canonical implementation until G6 lands; the C++ port
is validated by bit-equivalence on the smoke-test set at every
phase boundary.

## 7. Open design questions

The architect locked the answers to all 7 questions on 2026-07-30
(see items 8-14 in "Architectural decisions (locked)" above).
The agent's leans below are historical analysis; the architect's
decisions supersede them.

### 7.1 TESSERA_T640_3D as a separate enum entry or a tensor-name marker?

The 3D expert bank is physically encoded as a flattened 2D matrix
under the same encoding as TESSERA_T640. The dispatch to
`GGML_OP_TILE640_MATMUL_ID` vs `GGML_OP_TILE640_MATMUL` is the
only difference.

Trade-off: a separate enum entry is consistent with how
`GGML_TYPE_MXFP4_MOE` is registered in stock `ggml.h`; it
preserves the type-trait contract (each `ggml_type` has its own
`from_float_ref` / `to_float`). A tensor-name marker is lighter
(no new enum slot) but couples the dispatch to a string prefix
(`blk.<i>.ffn_*_exps`).

Lean recommendation: separate enum entry. The `type_traits[]`
table is keyed by `ggml_type`, and adding `TESSERA_T640_3D`
mirrors `MXFP4_MOE`. The cost is one enum slot; the gain is that
the dispatch is type-driven and matches the rest of `ggml-quants.c`.

### 7.2 libopenblas vs naive shim for the SVD / Cholesky / QR paths

The linalg primitives (SVD, Cholesky, QR, Stiefel) are needed for
LRQ / FLRQ / DartQuant / SEPTQ. They are used only on a per-tensor
basis during calibration / GA, not in the quantize hot path. The
hot path (ternarize, pack, scales) uses `cblas_sgemm` and `cblas_sdot`.

Trade-off:
- Link `libopenblas` and use `LAPACK_*` for SVD / Cholesky / QR.
  Fast, but adds a runtime dependency on Linux.
- Use the naive shim (Householder + power iter) from
  `_flrq_linalg.py`. No new dependency. Slow for 4096^2 matrices
  (~10s vs ~100ms with LAPACK).
- Use `cblas_sgemm` for the matmul and call `LAPACKE_sgesvd` only
  when `LAPACK` is available; otherwise fall back to the shim.
  Hybrid.

Lean recommendation: the LAPACK path is the locked decision for
Linux; on macOS it's the Accelerate LAPACK shim. The naive shim
is the fallback for `find_package(OpenBLAS) NOT FOUND` on Linux.
The smoke test (a 4-tensor bundle at 4096^2) runs in under 5
seconds with LAPACK and under 60 seconds with the shim; both fit
the 7-10 day budget.

### 7.3 L5 orchestrator as a separate flag pair or a single dispatch

`--tessera-mode=calibrate-only`, `--tessera-mode=evolve-only`,
`--tessera-mode=default`. The Python tool has 3 separate tools
(calibrate, evolve, quantize) plus the orchestrator. The C++ port
flattens this into a single tool with mode flags.

Trade-off:
- One tool, three modes (decision 1). Simpler CLI, fewer moving
  parts. Downside: a user who wants to inspect the policy before
  running the GA cannot easily do so without the orchestrator.
- One tool, one mode (`default`), with separate flags for the
  intermediate outputs. Same as the current Python flow, with
  the orchestrator as a library call inside.

Lean recommendation: one tool, three modes. The GA is a single
deterministic pass; the policy is a deterministic artifact; the
user can inspect the policy JSON after the GA without re-running
the quantize.

### 7.4 Where the calibration policy lives: in-line, sidecar JSON, or GGUF metadata?

The Python tool writes a separate `llama.speculative.calibration-
policy.v1` JSON next to the output GGUF. The C++ port could:
- Match the Python convention (sidecar JSON).
- Embed the policy under `tessera.calibration.policy` as a GGUF
  metadata string field.
- Both (GGUF metadata + sidecar JSON).

Trade-off: GGUF metadata embedding is self-describing (the GGUF
carries its calibration policy) but bloats the GGUF for large
LRQ/FLRQ factor payloads. The sidecar JSON keeps the GGUF clean
but is two files to ship.

Lean recommendation: both. The GGUF gets `tessera.calibration.policy`
as a small string (the `tensor_families` block only, not the U/V
factor payloads). The U/V payloads and the full GA archive live in
the sidecar JSON. The provenance stamping in
`tessera.calibration.provenance` records the policy SHA-256 in
both places so the audit trail is intact.

### 7.5 Built-in mini-corpus: ship or require `--calib-corpus`?

The locked decision (item 7) says: no imatrix + no policy + no
corpus = run calibration on the built-in mini-corpus. The Python
tool does not have a built-in mini-corpus; it requires the user
to pass a corpus or fall back to the imatrix-less magnitude-only
ternarize. The C++ port is adding the built-in mini-corpus.

Trade-off:
- Ship a tiny built-in mini-corpus (~1MB of synthetic data, baked
  in at compile time). Always available, no user setup. Downside:
  the calibrator's signal is weak on a tiny synthetic corpus.
- Refuse to run without a corpus, but suggest a built-in path
  download. More user friction, better signal.
- Refuse and require the user to point at a real calibration
  corpus (the same as the Python tool today). Lowest surprise for
  users migrating from the Python tool.

Lean recommendation: ship a built-in mini-corpus. The locked
decision says this is the default behaviour; the cost is one
~1MB static array in `tessera-corpus.cpp`. The smoke test runs in
under 10 seconds; the user can always pass `--calib-corpus` for a
real corpus.

### 7.6 CHAMP-Q: ship with the initial port or defer?

CHAMP-Q is the channel-permutation pre-pass. The Python tool
implements it as a separate helper that the quantizer optionally
calls. The C++ port can either:
- Port it (350 LoC; tacks onto the `tessera-search.cpp` file).
- Defer it; the port lands the L1.5 + GA path first, CHAMP-Q
  follows as a follow-up.

Trade-off: CHAMP-Q is a quality refinement (A/B shows modest
improvements on the heavy-tail bundle) but not a correctness
requirement. The 7-10 day budget is tight; deferring it buys
~350 LoC of headroom for the harder G2 / G3 work.

Lean recommendation: port it. The implementation is straightforward
(permutation + Sinkhorn + L-BFGS); the alternative is an
incompatible artifact between the Python and C++ paths (a user
who runs the Python tool on one model and the C++ tool on another
gets a different result). Defer it only if G4 (the GA) overruns.

### 7.7 PE-QAT: in or out of scope for this port?

PE-QAT is the LoRA-merge + SmoothQuant path. The Python tool
supports it via `--pe-qat-policy`; the demo and the trainer are
in `tools/tessera/pe_qat.py`. The LoRA merge is a small operation
(W_eff = W + s * (A @ B)); the SmoothQuant is a per-channel
multiply.

Trade-off: the LoRA merge is ~100 LoC; the SmoothQuant is ~50
LoC. The PE-QAT policy is a JSON that the C++ port must parse.

Lean recommendation: in scope, but only the LoRA merge. The
SmoothQuant is a per-tensor pre-scale; the policy records the
per-channel `s` and the C++ port applies it before ternarize.
The full PE-QAT training loop (the `pe_qat.py` trainer) stays
in Python; only the consumer side (`apply_pe_qat_to_weight`) is
ported. The G2 phase is unchanged.

## 8. Risk register

### R1 - Backward-compat breakage on stock K-quants

The new `TESSERA_T640` / `TESSERA_T640_3D` enum entries shift
`GGML_TYPE_COUNT` from 43 to 45. The type enum is part of the
stable C ABI; downstream code that uses `GGML_TYPE_COUNT` for
sizing arrays or for switch-statement bounds must be updated.

Mitigation: the trait table in `ggml.c` is indexed by
`enum ggml_type`, not by `GGML_TYPE_COUNT`; the table is sized at
`GGML_TYPE_COUNT` at compile time. Any external switch over
`ggml_type` needs a default case; this is unchanged from the
existing K-quant additions. The smoke test loads an existing
Q4_K GGUF and quantizes it back to Q4_K_M; the output is
byte-equivalent to the un-patched `llama-quantize`.

### R2 - Performance regression on stock K-quants

The new dispatch in `llama-quantize` adds a `if (tessera_mode ==
off)` branch on every tensor. The branch is predictable
(constant per run); the cost is one comparison per tensor.

Mitigation: profile the K-quant path with `--tessera-mode=off` on
a 7B model; the smoke test asserts that the wall-clock is within
1% of the un-patched `llama-quantize`. If it regresses, the
dispatch can be moved out of the per-tensor loop into a top-level
selection.

### R3 - ABI compat with existing `llama-quantize-impl` dylib

The existing `llama-quantize-impl` dylib links to
`llama_model_quantize_params` (a stable C struct). The new Tessera
params (`tessera_mode`, `tessera_imatrix_path`, etc.) extend the
params struct, not replace it. The extension is additive: existing
callers that don't set the new fields get the defaults
(`tessera_mode = off`, `tessera_imatrix_path = ""`).

Mitigation: the `tessera_*` fields are appended to the END of the
struct, not in the middle. The struct size grows; the layout is
unchanged for the existing fields. The struct is versioned
(`tessera_params_version`); the dylib refuses to load a struct
with a higher version than it knows about. The smoke test loads
the dylib with the old struct and confirms it still works.

### R4 - libopenblas portability

Linux is the target; `find_package(OpenBLAS)` finds the system
OpenBLAS on Ubuntu/Debian/CentOS/Fedora. Alpine and musl-based
distros do not ship OpenBLAS by default; the fallback is a naive
shim (Householder + power iter). The naive shim is slow for
4096^2 matrices (~10s vs ~100ms with LAPACK) but correct.

Mitigation: the smoke test on a 4-tensor bundle (each ~1024^2)
runs in under 60 seconds with the naive shim. The bundled CI
runner has OpenBLAS preinstalled. Alpine and musl are tested
explicitly with the naive shim path. The docs note the
performance difference.

### R5 - Accelerate on macOS

`Accelerate.framework` is the system framework; on macOS
`vDSP_*` and LAPACK are part of it. No external dependency.

Mitigation: the smoke test on macOS uses
`framework Accelerate` directly; no `find_package` is needed. The
ctypes binding in `_accelerate.py` is not ported (the Python
binding is opaque to C++); the C++ port calls the C API directly
through the framework's headers. The CI runner is macOS-arm64.

### R6 - GA convergence on the first C++ port

The GA is the most complex piece: 1197 LoC of Python with a
progressive evaluation pipeline, MAP-Elites archive, and
checkpoint JSON. Bit-for-bit determinism across runs is required
(item 6 of the locked decisions). The risk is that a small
ordering or RNG-state mismatch makes the C++ GA produce a
slightly different policy than the Python GA on the same seed.

Mitigation:
- The RNG is `std::mt19937_64` (or splitmix64) seeded with the
  same `uint64_t seed`; the order of operations matches the
  Python `random.Random(seed).random()` calls.
- The candidate JSON key is `nlohmann::json::dump` with sorted
  keys and no whitespace; the Python key uses
  `json.dumps(dataclasses.asdict(c), sort_keys=True, separators=(",", ":"))`.
  Both produce the same byte sequence.
- The checkpoint JSON is a verbatim copy of the Python format;
  the C++ reader accepts the Python writer's output and vice
  versa.
- The smoke test runs the GA for 4 generations on a 4-layer
  bundle and asserts that the policy JSON is byte-equivalent
  across 5 runs at the same seed.

### R7 - L1.5 reference format mismatch

The v3 sidecar carries the F16 reference as F32-cast. The runtime
dequant writes F32; the L1.5 mode (`w4a4`) writes the same F32
data under a different suffix. The C++ quantize path reads
`*.act.dequant.f32` and treats it as F32.

Mitigation: the v3 reader dispatches on the version field and
locates the data block at `40 + R*4 + R*24` bytes. The C++
quantize path uses the v3 reader; the format is unambiguous. The
smoke test writes a v3 sidecar via the runtime, reads it via
`ts_l15_load_reference`, and asserts bit-equivalence with the
source weight cast to F16 then back to F32.

### R8 - Calibration policy schema drift

The Python tool reads and writes
`llama.speculative.calibration-policy.v1` with sub-schemas for
LRQ / FLRQ / DartQuant / Hessian trace. The C++ port must
preserve the schema verbatim so policies produced by either side
are interchangeable.

Mitigation: the policy writer in `tessera-policy.cpp` round-trips
through `nlohmann::json` with the same key order and formatting
as the Python writer. The smoke test reads a Python-written
policy in the C++ path and confirms the same policy is written
back unchanged (modulo the `tessera_main_tip` field which is
stamped at quantize time).

### R9 - Concurrent quantize on the GA checkpoint file

The Python GA writes the checkpoint JSON after every generation
(`awq-evolve.py:889-890`). The C++ port inherits the same
behaviour. A crash mid-write leaves a partial JSON. The Python
tool detects this on resume (`if state.get("bundle_digest") ==
bundle_digest: score_cache = ...`) and either recovers or
re-runs.

Mitigation: the C++ port writes to `<checkpoint>.tmp` and renames
atomically. The resume logic matches the Python tool. The smoke
test kills the GA mid-generation and resumes; the resumed policy
matches the uninterrupted run within the seed's determinism
contract.

### R10 - Imatrix v2 / v3 schema transition

The current code loads imatrix from .npz (v2) and .gguf (v3,
emitted by llama-imatrix --output-format gguf). The C++ port
must handle both. The v3 path requires the libgguf C++ reader;
the v2 path is a custom npz reader.

Mitigation: the C++ port adds a minimal npz reader
(`tessera-imatrix.cpp`); it is small (~90 LoC) and has a
self-contained test. The libgguf reader is unchanged. The smoke
test loads both formats and asserts the per-channel RMS is
within 1e-6 of the Python tool's output.

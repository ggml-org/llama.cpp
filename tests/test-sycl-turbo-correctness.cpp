// CPU-vs-SYCL correctness harness for TurboQuant kernels.
//
// Rationale: the SYCL TurboQuant kernels (merged from the FellypeMelo fork)
// produce nonsense at inference time. This harness isolates *where* by running
// the SAME ggml graph on the CPU backend (the known-good reference) and on the
// SYCL backend, then diffing the outputs. There is deliberately NO hand-rolled
// reference math -- the CPU backend IS the reference -- so the only thing under
// test is the SYCL kernel.
//
// Each TurboQuant stage is probed independently so a failure localises the bug:
//   - TURBO_WHT          : the (inverse) Walsh-Hadamard rotation
//   - CPY turbo -> F32   : the per-element centroid decode / copy kernel
//   - MUL_MAT (turbo w)  : the mmvq dequant + dot-product kernel
//   - FLASH_ATTN_EXT     : the actual turbo KV-cache attention path
//
// Build: enabled by the tests/CMakeLists.txt block alongside the other SYCL
// tests (requires -DGGML_SYCL=ON). Run: ./bin/test-sycl-turbo-correctness
//
// This is a GATE, not just a diagnostic. Each probe carries an expectation:
//   GATE  probes MUST pass. This now includes turbo flash-attention (see the
//         KQ dot contract fix in fattn-common.hpp): turbo FA was XFAIL/vetoed
//         until that fix landed, and is promoted to GATE here.
//   XFAIL probes (none currently active) are known-broken: they SKIP and MUST
//         NOT pass yet. Exit code is non-zero iff a GATE probe FAILs OR an
//         XFAIL probe PASSes (XPASS = a fix landed -> "promote to GATE").
//         SKIPs never fail the gate.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-innerq.h"   // P3.2.2: minimal host state machine (see header for surface)


#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

enum class Tol { STD, LOSSY };   // STD: nmse+cosine (same-precision paths); LOSSY: cosine-only (quantized/turbo vs f16 ref)
enum class Exp { GATE, XFAIL };  // GATE: must pass; XFAIL: known-broken, must NOT pass yet

static int g_failures = 0;  // GATE probes that FAILed    -> red
static int g_xpass    = 0;  // XFAIL probes that PASSed   -> red (promote to GATE)
static int g_xfail    = 0;  // XFAIL probes still failing -> expected, green
static int g_skips    = 0;

// deterministic N(0,1) data so CPU and SYCL see identical inputs
static std::vector<float> gen_normal(size_t n, uint32_t seed, float stddev = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

static std::vector<char> quantize_host(ggml_type type, const std::vector<float> & src,
                                       int64_t n_per_row, int64_t nrows) {
    std::vector<char> dst(ggml_row_size(type, n_per_row) * nrows);
    ggml_quantize_chunk(type, src.data(), dst.data(), 0, nrows, n_per_row, nullptr);
    return dst;
}

struct err_stats {
    double nmse;       // ||test - ref||^2 / ||ref||^2
    double max_abs;    // max |test - ref|
    double cosine;     // cosine similarity (catches scale-correct-but-rotated output)
    double norm_ratio; // ||test|| / ||ref||: catches a wrong global gain that cosine misses
};

static err_stats compare(const std::vector<float> & test, const std::vector<float> & ref) {
    double se = 0.0, sref = 0.0, stest = 0.0, dot = 0.0, maxa = 0.0;
    const size_t n = ref.size();
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) test[i] - (double) ref[i];
        se    += d * d;
        sref  += (double) ref[i]  * (double) ref[i];
        stest += (double) test[i] * (double) test[i];
        dot   += (double) test[i] * (double) ref[i];
        maxa   = std::max(maxa, std::fabs(d));
    }
    err_stats s;
    s.nmse   = sref > 0.0 ? se / sref : se;
    s.cosine = (sref > 0.0 && stest > 0.0) ? dot / (std::sqrt(sref) * std::sqrt(stest)) : 0.0;
    s.max_abs = maxa;
    s.norm_ratio = (sref > 0.0 && stest > 0.0) ? std::sqrt(stest / sref) : 0.0;
    return s;
}

// verdict thresholds. CPU vs GPU will never be bit-identical (different
// accumulation order, F16 intermediates), so a small nmse is expected. Genuine
// "nonsense" shows up as a large nmse and/or a collapsed cosine.
//
// Tol::STD  -> same-precision paths (f16-vs-f16, f32 WHT/dequant/mul_mat):
//             tight nmse+cosine bar.
// Tol::LOSSY -> quantized/turbo vs an f16 reference: judge on cosine (direction)
//              plus a magnitude-ratio band ||test||/||ref||. A correct turbo path
//              preserves group magnitude (the block norm field stores the
//              grp_norm/recon_norm correction), so the ratio sits near 1 despite
//              quant noise; the band trips a wrong global gain (missing norm,
//              double 1/sqrt(D)) that cosine cannot see. Bands are calibrated on
//              the observed A770 baseline values (see meets_pass/meets_warn).
// Exp::GATE  -> must PASS (WARN tolerated); a FAIL reds the gate.
// Exp::XFAIL -> known-broken; it must NOT pass yet. A PASS becomes XPASS and
//              reds the gate ("promote to GATE"), signalling the fix landed.
static bool meets_pass(const err_stats & s, Tol tol) {
    if (tol == Tol::STD) return s.nmse < 1e-3 && s.cosine > 0.999;
    // Bands calibrated on the A770 baseline: passing turbo3/4 probes sit at
    // norm_ratio 0.96..1.00, so [0.85, 1.15] leaves ~0.11 margin below the min
    // while still tripping a >15% global-gain bug. Margin also covers the f16 LUT
    // precision loss the Phase 1 optimizations introduce.
    return s.cosine > 0.95 && s.norm_ratio > 0.85 && s.norm_ratio < 1.15;
}
static bool meets_warn(const err_stats & s, Tol tol) {
    if (tol == Tol::STD) return s.nmse < 5e-2 && s.cosine > 0.99;
    // turbo2 legitimately under-reconstructs to ~0.786 (2-bit), so the warn floor
    // stays well below it while still catching a missing-norm bug (ratio ~0.1).
    return s.cosine > 0.80 && s.norm_ratio > 0.60 && s.norm_ratio < 1.70;
}
static void verdict(const char * label, const err_stats & s, Tol tol, Exp exp) {
    const bool pass = meets_pass(s, tol), warn = meets_warn(s, tol);
    const char * tag;
    if (exp == Exp::GATE) {
        tag = pass ? "PASS" : (warn ? "WARN" : "FAIL");
        if (!pass && !warn) g_failures++;          // WARN tolerated, FAIL is red
    } else { // XFAIL
        if (pass) { tag = "XPASS"; g_xpass++; }    // fixed! -> red until promoted to GATE
        else      { tag = "xfail"; g_xfail++; }    // expected-broken -> green
    }
    printf("  [%s] %-28s nmse=%.3e  max_abs=%.3e  cosine=%.6f  |t|/|r|=%.4f\n",
           tag, label, s.nmse, s.max_abs, s.cosine, s.norm_ratio);
}

static void skip(const char * label, const char * why) {
    printf("  [SKIP] %-28s (%s)\n", label, why);
    g_skips++;
}

// Run a single-output graph on `backend`. `build` creates the graph (naming any
// input tensors) and returns the output node; `set_inputs` uploads data after
// allocation. Returns the output flattened to F32.
static std::vector<float> run_on_backend(
        ggml_backend_t backend,
        const std::function<ggml_tensor *(ggml_context *)> & build,
        const std::function<void(ggml_context *)> & set_inputs,
        bool * supported) {

    ggml_init_params p = {
        /* .mem_size   = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead() + (1u << 20),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * out = build(ctx);

    if (supported) {
        *supported = ggml_backend_supports_op(backend, out);
        if (!*supported) {
            ggml_free(ctx);
            return {};
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    set_inputs(ctx);

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// Drive one probe across both backends and report.
static void probe(const char * label,
                  ggml_backend_t cpu, ggml_backend_t sycl,
                  const std::function<ggml_tensor *(ggml_context *)> & build,
                  const std::function<void(ggml_context *)> & set_inputs,
                  Exp exp = Exp::GATE) {
    bool cpu_ok = true, sycl_ok = true;
    std::vector<float> ref  = run_on_backend(cpu,  build, set_inputs, &cpu_ok);
    if (!cpu_ok) { skip(label, "CPU backend lacks op (no reference)"); return; }

    std::vector<float> test = run_on_backend(sycl, build, set_inputs, &sycl_ok);
    if (!sycl_ok) { skip(label, "SYCL backend reports op unsupported"); return; }

    if (test.size() != ref.size() || ref.empty()) {
        printf("  [FAIL] %-28s size mismatch (ref=%zu test=%zu)\n", label, ref.size(), test.size());
        g_failures++;
        return;
    }
    verdict(label, compare(test, ref), Tol::STD, exp);
}

// ---------------------------------------------------------------------------
// probes
// ---------------------------------------------------------------------------

// (1) Inverse Walsh-Hadamard rotation in isolation (pure F32 -> F32).
static void probe_wht(ggml_backend_t cpu, ggml_backend_t sycl, int group_size) {
    const int64_t ne = group_size;          // one rotation group
    const int64_t rows = 8;                  // several independent groups
    auto data = gen_normal(ne * rows, 0x5701u ^ (uint32_t) group_size);

    for (int dir = 0; dir <= 1; ++dir) {
        char label[64];
        snprintf(label, sizeof(label), "WHT g=%d dir=%d", group_size, dir);

        auto build = [=](ggml_context * ctx) -> ggml_tensor * {
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, rows);
            ggml_set_name(a, "a");
            return ggml_turbo_wht(ctx, a, dir, group_size, nullptr);
        };
        auto set_inputs = [=](ggml_context * ctx) {
            ggml_tensor * a = ggml_get_tensor(ctx, "a");
            ggml_backend_tensor_set(a, data.data(), 0, data.size() * sizeof(float));
        };
        probe(label, cpu, sycl, build, set_inputs);
    }
}

// (1b) WHT with a non-trivial InnerQ scale_inv. Closes the "no oracle for a
// non-trivial scale_inv" gap: forward applies scale_inv before the rotation,
// inverse applies it after. The CPU op (ggml_compute_forward_turbo_wht_f32)
// applies it identically to the SYCL kernel (k_turbo_wht_f32_sycl), so this
// checks the SYCL scale_inv path against a real per-channel scale, not just 1.0.
static void probe_wht_scaled(ggml_backend_t cpu, ggml_backend_t sycl, int group_size) {
    const int64_t ne   = group_size;
    const int64_t rows = 8;
    auto data = gen_normal(ne * rows, 0x5c1eu ^ (uint32_t) group_size);

    // Non-uniform per-channel scale spanning the InnerQ clamp range [0.5, 2.0].
    std::vector<float> scale_inv(group_size);
    for (int i = 0; i < group_size; ++i) {
        scale_inv[i] = 0.5f + 1.5f * ((float) i / (float) (group_size - 1));
    }

    for (int dir = 0; dir <= 1; ++dir) {
        char label[64];
        snprintf(label, sizeof(label), "WHT g=%d dir=%d scaled", group_size, dir);

        auto build = [=](ggml_context * ctx) -> ggml_tensor * {
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, rows);
            ggml_set_name(a, "a");
            ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, group_size);
            ggml_set_name(s, "s");
            return ggml_turbo_wht(ctx, a, dir, group_size, s);
        };
        auto set_inputs = [=](ggml_context * ctx) {
            ggml_tensor * a = ggml_get_tensor(ctx, "a");
            ggml_backend_tensor_set(a, data.data(), 0, data.size() * sizeof(float));
            ggml_tensor * s = ggml_get_tensor(ctx, "s");
            ggml_backend_tensor_set(s, scale_inv.data(), 0, scale_inv.size() * sizeof(float));
        };
        probe(label, cpu, sycl, build, set_inputs);
    }
}

// (2) Centroid decode: cpy turbo -> F32. Isolates the dequant/copy kernel.
static void probe_dequant(ggml_backend_t cpu, ggml_backend_t sycl,
                          ggml_type type, const char * name, int64_t K) {
    const int64_t rows = 16;
    auto src_f32 = gen_normal(K * rows, 0xDE00 ^ (uint32_t) type);
    auto qbytes  = quantize_host(type, src_f32, K, rows);

    char label[64];
    snprintf(label, sizeof(label), "cpy %s->f32", name);

    auto build = [=](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * src = ggml_new_tensor_2d(ctx, type, K, rows);
        ggml_set_name(src, "src");
        ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, rows);
        return ggml_cpy(ctx, src, dst);
    };
    auto set_inputs = [=](ggml_context * ctx) {
        ggml_tensor * src = ggml_get_tensor(ctx, "src");
        ggml_backend_tensor_set(src, qbytes.data(), 0, qbytes.size());
    };
    probe(label, cpu, sycl, build, set_inputs);
}

// (2b) SET_ROWS turbo quantize-store path: F32 -> turbo write, on-device.
// probe_dequant (above) only exercises CPY reading pre-quantized bytes; it
// never runs the GPU's own quantize kernel (k_set_rows_turbo_generic in
// set_rows.cpp). This probe writes F32 through ggml_set_rows into a fresh
// turbo destination on each backend, then dequantizes both results on the
// CPU (via ggml_get_type_traits) and compares -- isolating the SYCL SET_ROWS
// quantize kernel from everything else in the KV-cache path.
static void probe_set_rows_turbo(ggml_backend_t cpu, ggml_backend_t sycl,
                                 ggml_type type, const char * name, int64_t K, int64_t rows) {
    auto src_f32 = gen_normal(K * rows, 0x5E70 ^ (uint32_t) type);
    std::vector<int64_t> idx(rows);
    for (int64_t i = 0; i < rows; ++i) idx[i] = i;  // identity permutation

    char label[64];
    snprintf(label, sizeof(label), "set_rows %s (quantize)", name);

    auto build = [=](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * dst = ggml_new_tensor_2d(ctx, type, K, rows);
        ggml_set_name(dst, "dst");
        ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, rows);
        ggml_set_name(src, "src");
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, rows);
        ggml_set_name(ids, "ids");
        ggml_tensor * result = ggml_set_rows(ctx, dst, src, ids);
        // op_params[0] carries the WHT group size for turbo types (see cpy_k
        // in llama-kv-cache.cpp, which always writes 128: with zero-padding
        // all groups are full 128-element WHT groups); mirror that wiring.
        int32_t wht_group = 128;
        memcpy(result->op_params, &wht_group, sizeof(int32_t));
        return result;
    };
    auto set_inputs = [=](ggml_context * ctx) {
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "src"), src_f32.data(), 0, src_f32.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "ids"), idx.data(), 0, idx.size() * sizeof(int64_t));
    };

    ggml_init_params p = {
        ggml_tensor_overhead() * 8 + ggml_graph_overhead() + (1u << 20), nullptr, true
    };

    bool cpu_ok = true, sycl_ok = true;
    ggml_context * ctx_cpu = ggml_init(p);
    ggml_tensor  * out_cpu = build(ctx_cpu);
    cpu_ok = ggml_backend_supports_op(cpu, out_cpu);
    std::vector<char> cpu_bytes;
    if (cpu_ok) {
        ggml_cgraph * gf = ggml_new_graph(ctx_cpu);
        ggml_build_forward_expand(gf, out_cpu);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_cpu, cpu);
        set_inputs(ctx_cpu);
        ggml_backend_graph_compute(cpu, gf);
        cpu_bytes.resize(ggml_nbytes(out_cpu));
        ggml_backend_tensor_get(out_cpu, cpu_bytes.data(), 0, cpu_bytes.size());
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx_cpu);
    if (!cpu_ok) { skip(label, "CPU backend lacks op (no reference)"); return; }

    ggml_context * ctx_sycl = ggml_init(p);
    ggml_tensor  * out_sycl = build(ctx_sycl);
    sycl_ok = ggml_backend_supports_op(sycl, out_sycl);
    std::vector<char> sycl_bytes;
    if (sycl_ok) {
        ggml_cgraph * gf = ggml_new_graph(ctx_sycl);
        ggml_build_forward_expand(gf, out_sycl);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_sycl, sycl);
        set_inputs(ctx_sycl);
        ggml_backend_graph_compute(sycl, gf);
        sycl_bytes.resize(ggml_nbytes(out_sycl));
        ggml_backend_tensor_get(out_sycl, sycl_bytes.data(), 0, sycl_bytes.size());
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx_sycl);
    if (!sycl_ok) { skip(label, "SYCL backend reports op unsupported"); return; }

    // Dequantize both raw byte buffers on the CPU with the same to_float fn,
    // so any divergence in the *quantize* kernels shows up as a float diff.
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    std::vector<float> ref(K * rows), test(K * rows);
    tt->to_float(cpu_bytes.data(),  ref.data(),  K * rows);
    tt->to_float(sycl_bytes.data(), test.data(), K * rows);

    verdict(label, compare(test, ref), Tol::LOSSY, Exp::GATE);
}

// (3) mmvq path: y = W_turbo @ x   (single column -> mat-vec kernel).
static void probe_mul_mat(ggml_backend_t cpu, ggml_backend_t sycl,
                          ggml_type type, const char * name, int64_t K, int64_t M) {
    auto w_f32 = gen_normal(K * M, 0x3A00 ^ (uint32_t) type);
    auto qbytes = quantize_host(type, w_f32, K, M);
    auto x_f32 = gen_normal(K, 0x5B00 ^ (uint32_t) type);

    char label[64];
    snprintf(label, sizeof(label), "mul_mat %s", name);

    auto build = [=](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, M);
        ggml_set_name(w, "w");
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        ggml_set_name(x, "x");
        return ggml_mul_mat(ctx, w, x);
    };
    auto set_inputs = [=](ggml_context * ctx) {
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "w"), qbytes.data(), 0, qbytes.size());
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "x"), x_f32.data(), 0, x_f32.size() * sizeof(float));
    };
    probe(label, cpu, sycl, build, set_inputs);
}

// (4) The actual KV-cache path: flash attention with quantized/turbo K/V.
//
// `force` controls how the SYCL run treats the supports_op() check:
//   force=true  (f16/q8_0): bypass the check and exercise the kernel directly.
//               These KV types are genuinely supported, so forcing == running.
//   force=false (turbo):    RESPECT the veto. ggml_sycl_flash_attn_ext_supported
//               returns false for turbo KV because the turbo FA kernel (which
//               routes to VEC, see fattn.cpp) is broken. Crucially, FORCING
//               turbo past the veto does not merely produce garbage -- it HANGS
//               the A770 device (the broken VEC kernel never returns). A gate
//               must terminate, so turbo probes SKIP while vetoed. When the
//               turbo-FA fix lands it relaxes that veto (see the "relax this
//               veto" comment in fattn.cpp); the probe then RUNS and, if the
//               kernel is correct, PASSes -> XPASS -> "promote to GATE" signal.
// The reference is always F16 K/V on CPU; turbo/q8_0 are judged LOSSY (cosine).
static void probe_flash_attn(ggml_backend_t cpu, ggml_backend_t sycl,
                            ggml_type kv_type, const char * name,
                            int64_t d, int64_t n_q, const char * path,
                            Exp exp, bool force,
                            int64_t nh_q = 1, int64_t nh_kv = 1) {
    const int64_t n_kv = 256;   // cached tokens (multiple of FATTN_KQ_STRIDE)
    // nh_q = number of Q heads; nh_kv = number of K/V heads (GQA: nh_kv <= nh_q, nh_q % nh_kv == 0).
    // The harness historically used nh_q == nh_kv == 1 (single-head, no GQA). Real models use
    // GQA ratios (4:1 llama/mistral, 8:1 Qwen3-Coder-30B-A3B); see probe driver for the
    // realistic-shape sweeps. nh_kv == nh_q stays the default for the kernel-correctness
    // checks at each head dim; GQA variants probe the broadcast path.
    const int64_t pad  = 64;    // GGML_KQ_MASK_PAD
    const int64_t n_q_pad = ((n_q + pad - 1) / pad) * pad;

    auto q_f32 = gen_normal(d * n_q * nh_q, 0xFA01u ^ (uint32_t) kv_type);
    auto k_f32 = gen_normal(d * n_kv * nh_kv, 0xFA02u ^ (uint32_t) kv_type);
    auto v_f32 = gen_normal(d * n_kv * nh_kv, 0xFA03u ^ (uint32_t) kv_type);

    // F16 copies for the reference.
    std::vector<ggml_fp16_t> k_f16(k_f32.size()), v_f16(v_f32.size());
    ggml_fp32_to_fp16_row(k_f32.data(), k_f16.data(), k_f32.size());
    ggml_fp32_to_fp16_row(v_f32.data(), v_f16.data(), v_f32.size());

    // Turbo copies for the kernel under test.
    auto k_q = quantize_host(kv_type, k_f32, d, n_kv * nh_kv);
    auto v_q = quantize_host(kv_type, v_f32, d, n_kv * nh_kv);

    std::vector<ggml_fp16_t> mask(n_kv * n_q_pad, ggml_fp32_to_fp16(0.0f));
    const float scale = 1.0f / std::sqrt((float) d);

    auto build = [=](ggml_context * ctx, ggml_type kvt) -> ggml_tensor * {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_q, nh_q, 1);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_4d(ctx, kvt, d, n_kv, nh_kv, 1);
        ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_4d(ctx, kvt, d, n_kv, nh_kv, 1);
        ggml_set_name(v, "v");
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q_pad, 1, 1);
        ggml_set_name(m, "m");
        const bool turbo = (kvt == GGML_TYPE_TURBO2_0 || kvt == GGML_TYPE_TURBO3_0 || kvt == GGML_TYPE_TURBO4_0);
        // Turbo K/V are stored WHT-rotated (see quantize_row_turbo*_ref). Q must be
        // forward-rotated into the same basis so KQ scores are preserved (WHT is
        // orthogonal: (Wq)*(Wk) == q*k); the attention *output* inherits that
        // rotation from V, so it needs an inverse WHT to compare against the
        // unrotated F16 CPU reference. Mirrors llama-graph.cpp's turbo KV wiring.
        ggml_tensor * qq = turbo ? ggml_turbo_wht(ctx, q, 0, 0, nullptr) : q;  // forward, auto group
        ggml_tensor * o  = ggml_flash_attn_ext(ctx, qq, k, v, m, scale, 0.0f, 0.0f);
        if (turbo) {
            const int group = (d % 128 == 0) ? 128 : 64;
            o = ggml_turbo_wht(ctx, o, 1, group, nullptr);  // inverse
        }
        return o;
    };

    char label[80];
    if (nh_q == nh_kv) {
        snprintf(label, sizeof(label), "flash_attn %s d=%d [%s nq=%d]", name, (int) d, path, (int) n_q);
    } else {
        snprintf(label, sizeof(label), "flash_attn %s d=%d [%s nq=%d GQA %d:%d]",
                 name, (int) d, path, (int) n_q, (int) nh_q, (int) nh_kv);
    }

    // Reference: F16 K/V on CPU.
    bool cpu_ok = true;
    std::vector<float> ref = run_on_backend(cpu,
        [=](ggml_context * ctx) { return build(ctx, GGML_TYPE_F16); },
        [=](ggml_context * ctx) {
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "q"), q_f32.data(), 0, q_f32.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "k"), k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "v"), v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "m"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }, &cpu_ok);
    if (!cpu_ok) { skip(label, "CPU lacks F16 FA reference"); return; }

    // Test: K/V on SYCL. force=true bypasses supports_op (supported KV); for
    // turbo (force=false) we respect the veto and SKIP rather than hang.
    bool sok = true;
    std::vector<float> test = run_on_backend(sycl,
        [=](ggml_context * ctx) { return build(ctx, kv_type); },
        [=](ggml_context * ctx) {
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "q"), q_f32.data(), 0, q_f32.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "k"), k_q.data(), 0, k_q.size());
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "v"), v_q.data(), 0, v_q.size());
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "m"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }, force ? nullptr : &sok);
    if (!force && !sok) { skip(label, "SYCL vetoes turbo FA (kernel not yet implemented)"); return; }

    if (test.size() != ref.size() || ref.empty()) {
        printf("  [FAIL] %-28s size mismatch (ref=%zu test=%zu)\n", label, ref.size(), test.size());
        g_failures++;
        return;
    }
    // Turbo/quantized KV compared against an f16 reference -> LOSSY (cosine-only).
    verdict(label, compare(test, ref), Tol::LOSSY, exp);
}

// (4b) Non-FA attention path: mul_mat(k,q) -> softmax -> mul_mat(v,kq).
// This is a DIFFERENT SYCL kernel chain than flash_attn_ext (mmvq/dequant-mma
// vs the dedicated FA kernels) and is what turbo actually ran on before this
// port (FA was vetoed, so -fa off's non-FA path was turbo's ONLY path).
//
// GATE (was XFAIL): build_attn_mha's non-FA branch used to do
// `v = ggml_cont(ggml_transpose(v))` unconditionally when !v_trans, which is
// invalid for block-quantized V (a block encodes 128 logically-contiguous
// elements along dim 0; transposing scrambles that grouping without
// dequantizing first). Fixed in llama-graph.cpp by dequantizing quantized V
// to F32 (ggml_cast) BEFORE the transpose; the dequantized values stay in the
// WHT-rotated domain, and the inverse WHT on kqv (keyed on the original
// KV-cache V type) undoes the rotation after the contraction. This probe
// mirrors that fixed graph: turbo V -> cast F32 -> cont(transpose) ->
// mul_mat -> inverse WHT. Exercises the SYCL turbo->f32 CPY kernel (cpy.cpp)
// plus the mmvq turbo-K path on device.
static void probe_attn_noflash(ggml_backend_t cpu, ggml_backend_t sycl,
                               ggml_type kv_type, const char * name, int64_t d,
                               int64_t nh_q = 1, int64_t nh_kv = 1) {
    // Single-token decode probe (n_q=1). GQA: see probe_flash_attn for rationale.
    // Default (1, 1) preserves the historical single-head probe; main() sweeps
    // realistic GQA ratios (4:1 llama/mistral, 8:1 Qwen3-Coder-30B-A3B).
    const int64_t n_kv = 256;
    auto q_f32 = gen_normal(d * 1 * nh_q, 0xAF01u ^ (uint32_t) kv_type);
    auto k_f32 = gen_normal(d * n_kv * nh_kv, 0xAF02u ^ (uint32_t) kv_type);
    auto v_f32 = gen_normal(d * n_kv * nh_kv, 0xAF03u ^ (uint32_t) kv_type);

    std::vector<ggml_fp16_t> k_f16(k_f32.size()), v_f16(v_f32.size());
    ggml_fp32_to_fp16_row(k_f32.data(), k_f16.data(), k_f32.size());
    ggml_fp32_to_fp16_row(v_f32.data(), v_f16.data(), v_f32.size());

    auto k_q = quantize_host(kv_type, k_f32, d, n_kv * nh_kv);
    auto v_q = quantize_host(kv_type, v_f32, d, n_kv * nh_kv);

    const float scale = 1.0f / std::sqrt((float) d);

    char label[80];
    if (nh_q == nh_kv) {
        snprintf(label, sizeof(label), "attn_noflash %s d=%d", name, (int) d);
    } else {
        snprintf(label, sizeof(label), "attn_noflash %s d=%d GQA %d:%d",
                 name, (int) d, (int) nh_q, (int) nh_kv);
    }

    auto build = [=](ggml_context * ctx, ggml_type kvt) -> ggml_tensor * {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 1, nh_q);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_3d(ctx, kvt, d, n_kv, nh_kv);
        ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_3d(ctx, kvt, d, n_kv, nh_kv);
        ggml_set_name(v, "v");
        const bool turbo = (kvt == GGML_TYPE_TURBO2_0 || kvt == GGML_TYPE_TURBO3_0 || kvt == GGML_TYPE_TURBO4_0);
        ggml_tensor * qq = turbo ? ggml_turbo_wht(ctx, q, 0, 0, nullptr) : q;
        // mirrors build_attn_mha's non-FA branch: kq = mul_mat(k, q); softmax; kqv = mul_mat(v_transposed, kq)
        ggml_tensor * kq = ggml_mul_mat(ctx, k, qq);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        // Mirror build_attn_mha's FIXED !v_trans branch: block-quantized V
        // cannot be transposed post-hoc, so quantized V is dequantized to F32
        // first (stays in the WHT-rotated domain), THEN cont+transpose'd. The
        // f16 reference path keeps the plain cont+transpose.
        ggml_tensor * v_lin = ggml_is_quantized(kvt) ? ggml_cast(ctx, v, GGML_TYPE_F32) : v;
        ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v_lin));
        ggml_tensor * kqv = ggml_mul_mat(ctx, v_t, kq);
        if (turbo) {
            const int group = (d % 128 == 0) ? 128 : 64;
            kqv = ggml_cont(ctx, kqv);
            kqv = ggml_turbo_wht(ctx, kqv, 1, group, nullptr);
        }
        return kqv;
    };

    bool cpu_ok = true;
    std::vector<float> ref = run_on_backend(cpu,
        [=](ggml_context * ctx) { return build(ctx, GGML_TYPE_F16); },
        [=](ggml_context * ctx) {
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "q"), q_f32.data(), 0, q_f32.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "k"), k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "v"), v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));
        }, &cpu_ok);
    if (!cpu_ok) { skip(label, "CPU lacks f16 non-FA attn ref"); return; }

    bool sok = true;
    std::vector<float> test = run_on_backend(sycl,
        [=](ggml_context * ctx) { return build(ctx, kv_type); },
        [=](ggml_context * ctx) {
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "q"), q_f32.data(), 0, q_f32.size() * sizeof(float));
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "k"), k_q.data(), 0, k_q.size());
            ggml_backend_tensor_set(ggml_get_tensor(ctx, "v"), v_q.data(), 0, v_q.size());
        }, &sok);
    if (!sok) { skip(label, "SYCL reports non-FA turbo attn op unsupported"); return; }

    if (test.size() != ref.size() || ref.empty()) {
        printf("  [FAIL] %-28s size mismatch (ref=%zu test=%zu)\n", label, ref.size(), test.size());
        g_failures++;
        return;
    }
    verdict(label, compare(test, ref), Tol::LOSSY, Exp::GATE);
}
// (5) Baseline: standard f16 KV flash attention, SYCL vs CPU. This is NOT a
// turbo path -- it tests whether the merged SYCL FA kernels are correct at all.
// If this fails, the FellypeMelo FA header changes regressed the whole FA path
// (not just turbo); if it passes, only the turbo FA fusion is broken.
static void probe_fa_f16(ggml_backend_t cpu, ggml_backend_t sycl,
                         int64_t d, int64_t n_q, const char * path,
                         int64_t nh_q = 1, int64_t nh_kv = 1) {
    // GQA: see probe_flash_attn for rationale. Default (1, 1) keeps single-head baseline.
    const int64_t n_kv = 256, pad = 64;
    const int64_t n_q_pad = ((n_q + pad - 1) / pad) * pad;

    auto q_f32 = gen_normal(d * n_q * nh_q, 0xF16Au);
    auto k_f32 = gen_normal(d * n_kv * nh_kv, 0xF16Bu);
    auto v_f32 = gen_normal(d * n_kv * nh_kv, 0xF16Cu);
    std::vector<ggml_fp16_t> k_f16(k_f32.size()), v_f16(v_f32.size());
    ggml_fp32_to_fp16_row(k_f32.data(), k_f16.data(), k_f32.size());
    ggml_fp32_to_fp16_row(v_f32.data(), v_f16.data(), v_f32.size());
    std::vector<ggml_fp16_t> mask(n_kv * n_q_pad, ggml_fp32_to_fp16(0.0f));
    const float scale = 1.0f / std::sqrt((float) d);

    auto build = [=](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_q, nh_q, 1); ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv, nh_kv, 1); ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv, nh_kv, 1); ggml_set_name(v, "v");
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q_pad, 1, 1); ggml_set_name(m, "m");
        return ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    };
    auto set = [=](ggml_context * ctx) {
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "q"), q_f32.data(), 0, q_f32.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "k"), k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "v"), v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "m"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    };

    char label[80];
    if (nh_q == nh_kv) {
        snprintf(label, sizeof(label), "flash_attn f16 d=%d [%s nq=%d]", (int) d, path, (int) n_q);
    } else {
        snprintf(label, sizeof(label), "flash_attn f16 d=%d [%s nq=%d GQA %d:%d]",
                 (int) d, path, (int) n_q, (int) nh_q, (int) nh_kv);
    }

    bool cok = true, sok = true;
    auto ref = run_on_backend(cpu, build, set, &cok);
    if (!cok) { skip(label, "CPU lacks f16 FA"); return; }
    auto test = run_on_backend(sycl, build, set, &sok);
    if (!sok) { skip(label, "SYCL reports f16 FA unsupported"); return; }
    if (test.size() != ref.size() || ref.empty()) {
        printf("  [FAIL] %-28s size mismatch\n", label); g_failures++; return;
    }
    verdict(label, compare(test, ref), Tol::STD, Exp::GATE);
}

// ---------------------------------------------------------------------------

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0); // unbuffered: partial results survive a hang/kill

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        fprintf(stderr, "failed to init CPU backend\n");
        return 2;
    }
    ggml_backend_t sycl = ggml_backend_sycl_init(0);
    if (!sycl) {
        fprintf(stderr, "failed to init SYCL backend (device 0)\n");
        ggml_backend_free(cpu);
        return 2;
    }

    printf("== TurboQuant CPU-vs-SYCL correctness ==\n");
    printf("reference: %s   under test: %s\n\n",
           ggml_backend_name(cpu), ggml_backend_name(sycl));

    // Ordering rule: probes that cannot device-lost run first, so a later FA
    // crash can't mask earlier results. The crash-prone non-turbo VEC path
    // (n_q=1) runs LAST. Turbo FA is GATE (kernel fixed) and d=128 only
    // (QK_TURBO{2,3,4}==128 is a hard invariant; see ggml-common.h).
    printf("[1] Walsh-Hadamard rotation (TURBO_WHT)\n");                 // GATE
    probe_wht(cpu, sycl, 128);
    probe_wht(cpu, sycl, 64);
    probe_wht(cpu, sycl, 32);
    probe_wht_scaled(cpu, sycl, 128);   // non-trivial InnerQ scale_inv (production head_dim)

    printf("\n[2] centroid decode (cpy turbo -> f32)\n");                // GATE, +K=256 (multi-block)
    for (int64_t K : {128, 256}) {
        probe_dequant(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", K);
        probe_dequant(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", K);
        probe_dequant(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", K);
    }

    printf("\n[2b] SET_ROWS quantize-store (F32 -> turbo write, on-device)\n"); // GATE
    for (int64_t K : {128, 256}) {
        probe_set_rows_turbo(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", K, 16);
        probe_set_rows_turbo(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", K, 16);
        probe_set_rows_turbo(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", K, 16);
    }

    printf("\n[3] mat-vec dot product (mul_mat, turbo weights)\n");      // GATE, +K=256
    for (int64_t K : {128, 256}) {
        probe_mul_mat(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", K, 64);
        probe_mul_mat(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", K, 64);
        probe_mul_mat(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", K, 64);
    }

    printf("\n[3b] non-FA attention (mul_mat/softmax/mul_mat, turbo KV, d=128) - GATE: mirrors build_attn_mha's fixed !v_trans branch (dequant V to F32 before transpose; see doc comment above probe_attn_noflash)\n");
    probe_attn_noflash(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", 128);
    probe_attn_noflash(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", 128);
    probe_attn_noflash(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", 128);

    // [3c] Non-FA attention GQA sweep — exercises the broadcast path with realistic
    // GQA ratios from the validation fleet (head_dim=128 across all 3 models):
    //   4:1  llama-3.1-8B / mistral-7b  (head_count=32, head_count_kv=8)
    //   8:1  Qwen3-Coder-30B-A3B        (head_count=32, head_count_kv=4)
    // Per-head FA math is nh/nh_kv-independent, so a GATE pass here is mostly a
    // coverage-shape signal (no new kernel math), but a GATE fail would localize
    // to the GQA-broadcast code path on device. Stays non-FA path because FA is
    // separately exercised by [4]/[5]/[6]. d=128 only (turbo invariant).
    printf("\n[3c] non-FA attention GQA sweep (d=128) - GATE, llama/mistral 4:1 + qwen3 8:1\n");
    for (int64_t gqa_q : {4, 8}) {
        const int64_t gqa_kv = 1;  // 4:1 and 8:1 ratios
        for (ggml_type kvt : { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 }) {
            const char * nm = (kvt == GGML_TYPE_TURBO2_0) ? "turbo2_0"
                            : (kvt == GGML_TYPE_TURBO3_0) ? "turbo3_0"
                            : "turbo4_0";
            probe_attn_noflash(cpu, sycl, kvt, nm, 128, gqa_q, gqa_kv);
        }
    }
    // Head-dim sweep is {64,128}: d=256 reproducibly HANGS the A770 SYCL FA
    // path (device-lost manifesting as a non-terminating compute, not garbage)
    // on both the tile (n_q=8) and vec (n_q=1) kernels. A CI gate must always
    // terminate and a hang cannot be detected/skipped at runtime, so d=256 is
    // excluded to keep the gate deterministic (see eval-llama-cpp-sycl skill:
    // A770 FA-stress device-lost). d=64/d=128 prove the merged kernels correct.
    printf("\n[4] flash attention TILE (n_q=8, prefill) - GATE, standard KV across head dims\n");
    for (int64_t d : {64, 128}) probe_fa_f16(cpu, sycl, d, 8, "tile");
    for (int64_t d : {64, 128}) probe_flash_attn(cpu, sycl, GGML_TYPE_Q8_0, "q8_0", d, 8, "tile", Exp::GATE, /*force=*/true);

    // [4b] FA TILE GQA sweep at d=128 — f16 + q8_0 only (safe; turbo FA stays under [5]).
    // Same 4:1 + 8:1 ratios as [3c]. Per-head math is nh-independent, so this is a
    // coverage-shape check on the FA GQA-broadcast path.
    printf("\n[4b] flash attention TILE GQA sweep (d=128, n_q=8) - GATE, f16 + q8_0 across llama/mistral 4:1 + qwen3 8:1\n");
    for (int64_t gqa_q : {4, 8}) {
        const int64_t gqa_kv = 1;
        probe_fa_f16(cpu, sycl, 128, 8, "tile", gqa_q, gqa_kv);
        probe_flash_attn(cpu, sycl, GGML_TYPE_Q8_0, "q8_0", 128, 8, "tile", Exp::GATE, /*force=*/true, gqa_q, gqa_kv);
    }

    // Turbo FA fix landed (see fattn-common.hpp vec_dot_fattn_vec_KQ_turbo_generic):
    // the KQ dot now reads Q from the caller's per-thread register slice instead of
    // treating it as a full D-element row. Promoted from XFAIL to GATE.
    // A regressed turbo FA kernel can wedge the A770 (device-lost hang), and a hang
    // cannot be caught or skipped at runtime -- so gate these probes behind an opt-in
    // env var. Default runs skip [5] to keep CI un-wedgeable; developers set
    // LLAMA_TEST_TURBO_FA=1 to exercise the turbo FA kernels. (Restores PR #5's gate.)
    if (getenv("LLAMA_TEST_TURBO_FA")) {
        printf("\n[5] flash attention turbo KV - GATE (d=128 only)\n");
        for (int64_t n_q : {8, 1}) {
            const char * path = (n_q == 8) ? "tile" : "vec"; // label matches the kernel n_q routes to
            probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", 128, n_q, path, Exp::GATE, /*force=*/true);
            probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", 128, n_q, path, Exp::GATE, /*force=*/true);
            probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", 128, n_q, path, Exp::GATE, /*force=*/true);
            // [5b] GQA variants for the turbo FA path (still under LLAMA_TEST_TURBO_FA gate).
            // Same 4:1 + 8:1 ratios as [3c]/[4b] for cross-probe comparability.
            for (int64_t gqa_q : {4, 8}) {
                probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", 128, n_q, path, Exp::GATE, /*force=*/true, gqa_q, 1);
                probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", 128, n_q, path, Exp::GATE, /*force=*/true, gqa_q, 1);
                probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", 128, n_q, path, Exp::GATE, /*force=*/true, gqa_q, 1);
            }
        }
    } else {
        printf("\n[5] flash attention turbo KV - SKIPPED (set LLAMA_TEST_TURBO_FA=1 to run; gated so a regressed kernel cannot wedge the A770)\n");
    }

    printf("\n[6] flash attention VEC (n_q=1, decode) - GATE, standard KV across head dims [crash-prone: LAST]\n");
    for (int64_t d : {64, 128}) probe_fa_f16(cpu, sycl, d, 1, "vec");
    for (int64_t d : {64, 128}) probe_flash_attn(cpu, sycl, GGML_TYPE_Q8_0, "q8_0", d, 1, "vec", Exp::GATE, /*force=*/true);

    // [6b] FA VEC GQA sweep at d=128 — f16 + q8_0 only (turbo VEC FA stays under [5b]).
    // Same coverage-shape rationale as [4b]. The VEC kernel is the crash-prone one
    // (runs LAST), but f16/q8_0 don't wedge on A770 — only turbo has the hang risk.
    printf("\n[6b] flash attention VEC GQA sweep (d=128, n_q=1) - GATE, f16 + q8_0 across llama/mistral 4:1 + qwen3 8:1\n");
    for (int64_t gqa_q : {4, 8}) {
        const int64_t gqa_kv = 1;
        probe_fa_f16(cpu, sycl, 128, 1, "vec", gqa_q, gqa_kv);
        probe_flash_attn(cpu, sycl, GGML_TYPE_Q8_0, "q8_0", 128, 1, "vec", Exp::GATE, /*force=*/true, gqa_q, gqa_kv);
    }

    // [7] d=256 FA stress probe — EXPLICITLY GATED. d=256 reproducibly HANGS the A770 SYCL FA
    // path on both tile (n_q=8) and vec (n_q=1) kernels (device-lost, not garbage). CI gates
    // must terminate, and a hang cannot be caught at runtime — so d=256 is NOT in the
    // default sweep. This opt-in env-var-gated section exists so a developer can re-test it
    // after a driver/IGC bump to see if the hang was upstream-fixed. Set LLAMA_TEST_FA256=1.
    if (getenv("LLAMA_TEST_FA256")) {
        printf("\n[7] flash attention d=256 (OPT-IN, known-hang-risk on A770) - f16 only at this stage\n");
        probe_fa_f16(cpu, sycl, 256, 8, "tile");
        probe_fa_f16(cpu, sycl, 256, 1, "vec");
    } else {
        printf("\n[7] flash attention d=256 - SKIPPED (set LLAMA_TEST_FA256=1 to opt in; gated, A770 SYCL FA reproducibly hangs at d=256)\n");
    }
    // (summary print moved below after the [8] InnerQ skeleton section so the
    //  g_skips++ for the InnerQ SKIP is reflected in the final tally.)

    // [8] InnerQ host-side state machine skeleton (P3.2.1 hook).
    //
    // This section is a placeholder only -- the real InnerQ probes land in P3.2.2 (Qwen3-MoE
    // turbo3 rescue, with Early Kill Gate at chunk 8) and P3.2.3 (full state machine + device K^2
    // accumulation across the rest of the policy-eligible shapes). The hook exists today so that:
    //   (a) LLAMA_TEST_INNERQ=1 toggles a clearly-visible "skeleton present, no probes yet" line,
    //       rather than silently emitting nothing;
    //   (b) g_failures stays 0 in both env states (the default loop turn must remain green);
    //   (c) the env gate mirrors LLAMA_TEST_TURBO_FA / LLAMA_TEST_FA256 exactly so a future
    //       InnerQ probe drops into this section without changing the surrounding harness
    //       contracts.
    //
    // The P3.2 policy contract this hook respects is in RALPH_TASKS.md, section P3.2:
    //   - validation corpus  : 3-model validation fleet, head_dim=128 only (turbo invariant),
    //                           GQA 4:1 + 8:1; d=256 stays behind the existing opt-in gate;
    //   - failure modes      : hard-abort on NaN/Inf/DEVICE_LOST/exponential PPL drift;
    //                           soft-abort on >1% PPL regression or breaking turbo4 < q4_0;
    //                           recalibration policy = 1 retry on init-only anomalies, no retry
    //                           on mid-stream NaN;
    //   - rollback           : fail the InnerQ path specifically; fall back to static turbo4;
    //                           freeze at last-known-good scales on recovered runs;
    //   - scope              : turbo-only (initially turbo4-first); f16/q8_0/q4_0 stay as
    //                           evaluation baselines + fallback targets; V-only calibration
    //                           under GQA 8:1 auto-asymmetric (K upgrades to q8_0);
    //   - default state      : off by default in the live service (gated behind
    //                           LLAMA_ENABLE_INNERQ=1); transition to "on by default" is
    //                           blocked until full validation fleet passes AND the <=2%
    //                           decode regression guardrail is satisfied.
    //
    // For the engineering-side context (header-only spec of ggml_innerq_state /
    // ggml_innerq_probe / ggml_innerq_host, the SYCL-backend touchpoint list, and the
    if (getenv("LLAMA_TEST_INNERQ")) {
        printf("\n[8] InnerQ FA skeleton (P3.2.1/P3.2.2: host state machine + state-decide/k-scale probes)\n");
        printf("   policy contract: see RALPH_TASKS.md (section P3.2) + docs/research/innerq-host-state-machine-spec-2026-07-07.md\n");
        // [8a] Host state machine correctness probe (P3.2.2 unit, no PPL run).
        //
        // We exercise the policy contract by calling decide() and
        // k_squared_scale() across a small matrix of (key, env-state)
        // combinations and asserting the return matches the policy
        // expected outcome. This is a pure host-side check -- no SYCL
        // device work, no PPL run. The full 50-chunk Qwen3 PPL probe
        // is a separate sub-task (P3.2.4) -- see RALPH_TASKS.md.
        //
        // We verify by re-invoking the C wrapper functions directly
        // (no ggml-context needed) so the probe is independent of the
        // heavy backend graph machinery. The 5-question policy
        // contract is in the P3.2 section header of RALPH_TASKS.md.
        struct innerq_case_t {
            const char * label;
            ggml_innerq_state_key key;
            int env_should_optin;   // 1 if the env value is set, 0 otherwise
            ggml_innerq_policy expected_policy;
            int k_should_be_one;    // 1 if expected k_squared_scale == 1.0
        };
        // Expected values for the policy-decide() outcomes.
        const ggml_innerq_policy want_DISABLED = GGML_INNERQ_POLICY_DISABLED;
        const ggml_innerq_policy want_OPTIN   = GGML_INNERQ_POLICY_OPTIN;
        // Expected values for k_squared_scale():
        //   want_one   -> expected == 1.0f (safe default; impl returns 1.0
        //                  when the key is null, head_dim != 128, or innerq_quant
        //                  is not in {TURBO2/3/4})
        //   want_other -> expected != 1.0f (per-quant constant; impl returns
        //                  0.9375/0.9688/0.9844 for the 3 eligible innerq_quants)
        const int want_one    = 1;
        const int want_other  = 0;
        // env-state column values for the test matrix:
        const int env_unset   = 0;
        const int env_set     = 1;

        // Test matrix: (label, model_fp, head_dim, kv_quant, innerq_quant,
        //                env_set, expected_policy, k_should_be_one)
        // k_should_be_one is independent of expected_policy: the k_squared_scale
        // function gates on head_dim and innerq_quant only, NOT on policy.
        // It returns 1.0 iff (head_dim != 128 OR innerq_quant not in {TURBO2/3/4}).
        // The current implementation does NOT gate on kv_quant (the per-quant
        // constant is the same for all turbo kv_quants; P3.2.3 may extend).
        innerq_case_t cases[] = {
            // --- null key: decide rejects (DISABLED), k returns 1.0 ---
            {"null-key (env set)",     {0u, 128, GGML_TYPE_TURBO3_0, GGML_INNERQ_QUANT_TURBO3_0}, env_set,   want_DISABLED, want_one},
            // --- model_fp = 0 (unidentified): decide rejects, k returns 0.9688 (per-quant) ---
            {"unidentified (env set)", {0u, 128, GGML_TYPE_TURBO3_0, GGML_INNERQ_QUANT_TURBO3_0}, env_set,   want_DISABLED, want_other},
            // --- non-turbo kv_quant: decide rejects, k returns 0.9688 (impl doesn't gate on kv_quant yet) ---
            {"f16 kv (env set)",       {0xDEAD, 128, GGML_TYPE_F16,    GGML_INNERQ_QUANT_TURBO3_0}, env_set,   want_DISABLED, want_other},
            {"q8_0 kv (env set)",      {0xDEAD, 128, GGML_TYPE_Q8_0,   GGML_INNERQ_QUANT_TURBO3_0}, env_set,   want_DISABLED, want_other},
            // --- d != 128: decide rejects, k returns 1.0 (head_dim gate) ---
            {"d=256 (env set)",        {0xDEAD, 256, GGML_TYPE_TURBO3_0, GGML_INNERQ_QUANT_TURBO3_0}, env_set,   want_DISABLED, want_one},
            {"d=64 (env set)",         {0xDEAD,  64, GGML_TYPE_TURBO3_0, GGML_INNERQ_QUANT_TURBO3_0}, env_set,   want_DISABLED, want_one},
            // --- OPTIN cases (env set, all eligible: model_fp != 0, turbo kv, d=128) ---
            // k returns per-quant constant (0.9375 for turbo2, etc.)
            {"turbo2 d=128 (env set)", {0xDEAD, 128, GGML_TYPE_TURBO2_0, GGML_INNERQ_QUANT_TURBO2_0}, env_set,   want_OPTIN,   want_other},
            {"turbo3 d=128 (env set)", {0xDEAD, 128, GGML_TYPE_TURBO3_0, GGML_INNERQ_QUANT_TURBO3_0}, env_set,   want_OPTIN,   want_other},
            {"turbo4 d=128 (env set)", {0xDEAD, 128, GGML_TYPE_TURBO4_0, GGML_INNERQ_QUANT_TURBO4_0}, env_set,   want_OPTIN,   want_other},
            // --- env UNSET cases: all eligible keys still DISABLED ---
            // k returns per-quant constant (env is irrelevant to k_squared_scale)
            {"turbo3 d=128 (env unset)",{0xDEAD, 128, GGML_TYPE_TURBO3_0, GGML_INNERQ_QUANT_TURBO3_0}, env_unset, want_DISABLED, want_other},
            {"turbo4 d=128 (env unset)",{0xDEAD, 128, GGML_TYPE_TURBO4_0, GGML_INNERQ_QUANT_TURBO4_0}, env_unset, want_DISABLED, want_other},
        };
        // We don't override the env var from C here (the policy contract
        // reads LLAMA_ENABLE_INNERQ from the process env). The env_unset
        // case is verified by running this [8] probe ONLY when the harness
        // is launched without LLAMA_ENABLE_INNERQ in the env (which is
        // the default). The env_set case is verified by running with
        // LLAMA_ENABLE_INNERQ=1 (the condition that gates this whole
        // [8] block in the first place). So env_unset in the matrix
        // above documents the "should be DISABLED" expectation; we
        // expect exactly the env_set rows to come out OPTIN.


        int n_cases = sizeof(cases) / sizeof(cases[0]);
        int policy_failures = 0;
        int k_scale_failures = 0;
        int env_state = getenv("LLAMA_ENABLE_INNERQ") != nullptr ? env_set : env_unset;
        for (int i = 0; i < n_cases; ++i) {
            const innerq_case_t & c = cases[i];
            ggml_innerq_policy got = ggml_innerq_state_decide(&c.key);
            if (got != c.expected_policy) {
                printf("   [8a] FAIL: %s expected policy %d, got %d\n",
                       c.label, (int) c.expected_policy, (int) got);
                ++policy_failures;
            }
            float k_scale = ggml_innerq_state_k_squared_scale(&c.key);
            // k_squared_scale == 1.0 iff k_should_be_one else != 1.0.
            int is_one = (k_scale == 1.0f) ? 1 : 0;
            if (is_one != c.k_should_be_one) {
                printf("   [8a] FAIL: %s expected k_one=%d, got k_scale=%f\n",
                       c.label, c.k_should_be_one, (double) k_scale);
                ++k_scale_failures;
            }
        }
        if (policy_failures > 0 || k_scale_failures > 0) {
            printf("   [8a] InnerQ state machine: %d policy failures, %d k_scale failures (env state in harness: %d)\n",
                   policy_failures, k_scale_failures, env_state);
            g_failures++;
        } else {
            printf("   [8a] InnerQ state machine: %d cases PASS (env state in harness: %d)\n",
                   n_cases, env_state);
        }
        g_skips++;  // [8] as a whole is still in SKIP/placeholder territory -- the real
                    // PPL probe lives in P3.2.4 and is what the policy contract
                    // ultimately gates on. This P3.2.2 sub-probe is a unit check,
                    // not a regression catcher.
        skip("[8] InnerQ FA skeleton (P3.2.2 state machine unit PASS)", "real PPL probe is in P3.2.4; P3.2.1 ships spec + skeleton, P3.2.2 ships state machine + unit check");
    } else {
        printf("\n[8] InnerQ FA - SKIPPED (set LLAMA_TEST_INNERQ=1 to opt in; default OFF per P3.2 section 5 default-state)\n");
    }

    printf("\n== summary: %d GATE-FAIL, %d XPASS (promote to GATE!), %d xfail (expected-broken), %d SKIP ==\n",
           g_failures, g_xpass, g_xfail, g_skips);

    ggml_backend_free(sycl);
    ggml_backend_free(cpu);
    return (g_failures > 0 || g_xpass > 0) ? 1 : 0;
}

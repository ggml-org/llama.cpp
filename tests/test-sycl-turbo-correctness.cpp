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
//   GATE  probes (standard f16/q8_0 + turbo WHT/dequant/mul_mat) MUST pass.
//   XFAIL probes (turbo flash-attention) are known-broken: they SKIP while the
//         SYCL veto stands and MUST NOT pass yet.
// Exit code is non-zero iff a GATE probe FAILs OR an XFAIL probe PASSes (XPASS,
// = the turbo-FA fix has landed -> "promote to GATE"). SKIPs never fail the gate.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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
    double nmse;     // ||test - ref||^2 / ||ref||^2
    double max_abs;  // max |test - ref|
    double cosine;   // cosine similarity (catches scale-correct-but-rotated output)
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
    return s;
}

// verdict thresholds. CPU vs GPU will never be bit-identical (different
// accumulation order, F16 intermediates), so a small nmse is expected. Genuine
// "nonsense" shows up as a large nmse and/or a collapsed cosine.
//
// Tol::STD  -> same-precision paths (f16-vs-f16, f32 WHT/dequant/mul_mat):
//             tight nmse+cosine bar.
// Tol::LOSSY -> quantized/turbo vs an f16 reference: judge on cosine only,
//              since the lossy codec legitimately shifts magnitudes.
// Exp::GATE  -> must PASS (WARN tolerated); a FAIL reds the gate.
// Exp::XFAIL -> known-broken; it must NOT pass yet. A PASS becomes XPASS and
//              reds the gate ("promote to GATE"), signalling the fix landed.
static bool meets_pass(const err_stats & s, Tol tol) {
    return tol == Tol::STD ? (s.nmse < 1e-3 && s.cosine > 0.999) : (s.cosine > 0.95);
}
static bool meets_warn(const err_stats & s, Tol tol) {
    return tol == Tol::STD ? (s.nmse < 5e-2 && s.cosine > 0.99)  : (s.cosine > 0.80);
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
    printf("  [%s] %-28s nmse=%.3e  max_abs=%.3e  cosine=%.6f\n",
           tag, label, s.nmse, s.max_abs, s.cosine);
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
                             Exp exp, bool force) {
    const int64_t n_kv = 256;   // cached tokens (multiple of FATTN_KQ_STRIDE)
    const int64_t nh   = 1;     // heads
    const int64_t pad  = 64;    // GGML_KQ_MASK_PAD
    const int64_t n_q_pad = ((n_q + pad - 1) / pad) * pad;

    auto q_f32 = gen_normal(d * n_q * nh, 0xFA01u ^ (uint32_t) kv_type);
    auto k_f32 = gen_normal(d * n_kv * nh, 0xFA02u ^ (uint32_t) kv_type);
    auto v_f32 = gen_normal(d * n_kv * nh, 0xFA03u ^ (uint32_t) kv_type);

    // F16 copies for the reference.
    std::vector<ggml_fp16_t> k_f16(k_f32.size()), v_f16(v_f32.size());
    ggml_fp32_to_fp16_row(k_f32.data(), k_f16.data(), k_f32.size());
    ggml_fp32_to_fp16_row(v_f32.data(), v_f16.data(), v_f32.size());

    // Turbo copies for the kernel under test.
    auto k_q = quantize_host(kv_type, k_f32, d, n_kv * nh);
    auto v_q = quantize_host(kv_type, v_f32, d, n_kv * nh);

    std::vector<ggml_fp16_t> mask(n_kv * n_q_pad, ggml_fp32_to_fp16(0.0f));
    const float scale = 1.0f / std::sqrt((float) d);

    auto build = [=](ggml_context * ctx, ggml_type kvt) -> ggml_tensor * {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_q, nh, 1);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_4d(ctx, kvt, d, n_kv, nh, 1);
        ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_4d(ctx, kvt, d, n_kv, nh, 1);
        ggml_set_name(v, "v");
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q_pad, 1, 1);
        ggml_set_name(m, "m");
        return ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    };

    char label[64];
    snprintf(label, sizeof(label), "flash_attn %s d=%d [%s nq=%d]", name, (int) d, path, (int) n_q);

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

// (5) Baseline: standard f16 KV flash attention, SYCL vs CPU. This is NOT a
// turbo path -- it tests whether the merged SYCL FA kernels are correct at all.
// If this fails, the FellypeMelo FA header changes regressed the whole FA path
// (not just turbo); if it passes, only the turbo FA fusion is broken.
static void probe_fa_f16(ggml_backend_t cpu, ggml_backend_t sycl,
                         int64_t d, int64_t n_q, const char * path) {
    const int64_t n_kv = 256, nh = 1, pad = 64;
    const int64_t n_q_pad = ((n_q + pad - 1) / pad) * pad;

    auto q_f32 = gen_normal(d * n_q * nh, 0xF16Au);
    auto k_f32 = gen_normal(d * n_kv * nh, 0xF16Bu);
    auto v_f32 = gen_normal(d * n_kv * nh, 0xF16Cu);
    std::vector<ggml_fp16_t> k_f16(k_f32.size()), v_f16(v_f32.size());
    ggml_fp32_to_fp16_row(k_f32.data(), k_f16.data(), k_f32.size());
    ggml_fp32_to_fp16_row(v_f32.data(), v_f16.data(), v_f32.size());
    std::vector<ggml_fp16_t> mask(n_kv * n_q_pad, ggml_fp32_to_fp16(0.0f));
    const float scale = 1.0f / std::sqrt((float) d);

    auto build = [=](ggml_context * ctx) -> ggml_tensor * {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, n_q, nh, 1); ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv, nh, 1); ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_kv, nh, 1); ggml_set_name(v, "v");
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q_pad, 1, 1); ggml_set_name(m, "m");
        return ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    };
    auto set = [=](ggml_context * ctx) {
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "q"), q_f32.data(), 0, q_f32.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "k"), k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "v"), v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "m"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    };

    char label[64];
    snprintf(label, sizeof(label), "flash_attn f16 d=%d [%s nq=%d]", (int) d, path, (int) n_q);

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
    // (n_q=1) runs LAST. Turbo FA is XFAIL (kernel still broken) and d=128 only
    // (QK_TURBO{2,3,4}==128 is a hard invariant; see ggml-common.h).
    printf("[1] Walsh-Hadamard rotation (TURBO_WHT)\n");                 // GATE
    probe_wht(cpu, sycl, 128);
    probe_wht(cpu, sycl, 64);
    probe_wht(cpu, sycl, 32);

    printf("\n[2] centroid decode (cpy turbo -> f32)\n");                // GATE, +K=256 (multi-block)
    for (int64_t K : {128, 256}) {
        probe_dequant(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", K);
        probe_dequant(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", K);
        probe_dequant(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", K);
    }

    printf("\n[3] mat-vec dot product (mul_mat, turbo weights)\n");      // GATE, +K=256
    for (int64_t K : {128, 256}) {
        probe_mul_mat(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", K, 64);
        probe_mul_mat(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", K, 64);
        probe_mul_mat(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", K, 64);
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

    // Turbo FA is vetoed on SYCL today (routes to the broken VEC kernel; forcing
    // it HANGS the A770), so these probes SKIP while the veto stands -- green,
    // deterministic. force=false respects the veto. When the turbo-FA fix relaxes
    // the veto and corrects the kernel, the probes RUN and PASS -> XPASS (red,
    // "promote to GATE"): flip those probes' Exp::XFAIL -> Exp::GATE then.
    printf("\n[5] flash attention turbo KV - XFAIL (SKIP while vetoed; d=128 only)\n");
    for (int64_t n_q : {8, 1}) {
        const char * path = (n_q == 8) ? "tile" : "vec"; // label matches the kernel n_q routes to
        probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO2_0, "turbo2_0", 128, n_q, path, Exp::XFAIL, /*force=*/false);
        probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO3_0, "turbo3_0", 128, n_q, path, Exp::XFAIL, /*force=*/false);
        probe_flash_attn(cpu, sycl, GGML_TYPE_TURBO4_0, "turbo4_0", 128, n_q, path, Exp::XFAIL, /*force=*/false);
    }

    printf("\n[6] flash attention VEC (n_q=1, decode) - GATE, standard KV across head dims [crash-prone: LAST]\n");
    for (int64_t d : {64, 128}) probe_fa_f16(cpu, sycl, d, 1, "vec");
    for (int64_t d : {64, 128}) probe_flash_attn(cpu, sycl, GGML_TYPE_Q8_0, "q8_0", d, 1, "vec", Exp::GATE, /*force=*/true);

    printf("\n== summary: %d GATE-FAIL, %d XPASS (promote to GATE!), %d xfail (expected-broken), %d SKIP ==\n",
           g_failures, g_xpass, g_xfail, g_skips);

    ggml_backend_free(sycl);
    ggml_backend_free(cpu);
    return (g_failures > 0 || g_xpass > 0) ? 1 : 0;
}

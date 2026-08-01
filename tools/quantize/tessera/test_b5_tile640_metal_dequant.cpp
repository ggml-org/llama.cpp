//
// test_b5_tile640_metal_dequant.cpp
//
// B5 verification: drive a Tile640 matmul through the Metal backend with
// the L1 dequant debug hook on, then compare the GPU sidecar against:
//   (1) the CPU traits->to_float re-dequant of the packed buffer
//       (dequantize_row_tessera_t640) -- this is the "cheat" the B5
//       brief calls out: it omits outliers and uses the flat layout;
//   (2) the tessera quantizer's W_hat reconstruction (ts_quantize_2d
//       .recon), which is the per-row AWQ-removed ground truth the L1
//       fitness math compares against.
//
// The delta between (1) and the GPU sidecar is the key B5 signal: it
// proves the producer now writes what the kernel really outputs, not a
// host re-dequant. Non-zero deltas come from the GPU outlier addback
// (which the CPU to_float trait does not apply) and from GPU f32
// rounding.
//
// Run: set LLAMA_TILE640_DEBUG_DEQUANT_DIR and LLAMA_TILE640_DEBUG_DEQUANT=1
// env vars before launching (the harness also sets sane defaults to
// /tmp/b5-test).
//

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "tessera-quant.h"
#include "tessera-sidecar-v3.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

// CPU reference dequant (ggml-quants.c) for the flat packed layout.
extern "C" void dequantize_row_tessera_t640(const void * x, float * y, int64_t k);

static int g_fail = 0;
static void check(const char * name, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name);
    if (!ok) g_fail++;
}

// Read the F32 data block of a TDQT v3 sidecar. Returns false on error.
static bool read_sidecar_f32(const std::string & path,
                             std::vector<float> * out,
                             int64_t * rows, int64_t * cols) {
    ts_sidecar_v3 sc;
    if (ts_sidecar_v3_read(path.c_str(), &sc, nullptr) != 0) {
        return false;
    }
    if (rows) *rows = sc.header.rows;
    if (cols) *cols = sc.header.cols;
    *out = std::move(sc.data);
    return true;
}

int main(void) {
    // ---- 1. synthetic weight: out_dim rows, in_dim cols, 1 page per row ----
    // 640 = one Tile640 page; 8 rows keeps the test cheap.
    const int64_t in_dim  = 640;
    const int64_t out_dim = 8;
    const int64_t n       = out_dim * in_dim;

    std::vector<float> w((size_t) n);
    uint32_t rng = 2654435761u;
    for (int64_t i = 0; i < n; i++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float u = (float)((rng >> 8) & 0xFFFF) / (float)0xFFFF;
        // unit variance with a few large outliers so the sparse outlier
        // addback path in the GPU dequant is exercised.
        w[(size_t) i] = (u - 0.5f) * 2.0f;
    }
    // Inject a handful of large outliers at known columns (these become
    // the sparse addback residuals the CPU to_float trait cannot see).
    for (int64_t r = 0; r < out_dim; r++) {
        w[(size_t)(r * in_dim + 17)] = 7.5f * (float)(r + 1);
        w[(size_t)(r * in_dim + 200)] = -6.0f;
    }

    // ---- 2. Tessera quantize -> 6 component tensors + W_hat recon ----
    ts_quant_params_2d params = {};
    params.alpha          = 0.5f;
    params.clip           = 0.0f;
    params.max_outliers   = 8;
    params.outlier_thresh = 1e-3f;
    params.use_imatrix    = false;
    params.use_septq      = false;
    params.awq_grid       = 20;
    params.seed           = 1234;

    ts_quant_result_2d result;
    std::vector<float> act_scales((size_t) in_dim, 1.0f);
    int rc = ts_quantize_2d(w.data(), act_scales.data(),
                            /*calib_X=*/nullptr, /*ref_output=*/nullptr,
                            /*imatrix=*/nullptr,
                            out_dim, in_dim, /*n_tokens=*/0,
                            &params, &result);
    check("ts_quantize_2d", rc == 0);
    if (rc != 0) {
        std::printf("FAIL: ts_quantize_2d rc=%d\n", rc);
        return 1;
    }

    const int64_t pages_per_row = (in_dim + 639) / 640;
    const int64_t words_per_row = pages_per_row * 32;
    check("packed size",
          (int64_t) result.packed.size() == out_dim * words_per_row);
    check("page_scales size",
          (int64_t) result.page_scales.size() == out_dim * pages_per_row);
    check("lane_scales size",
          (int64_t) result.lane_scales.size() == out_dim * pages_per_row * 32);
    check("outlier_row_offsets size",
          (int64_t) result.outlier_row_offsets.size() == out_dim + 1);
    check("recon size", (int64_t) result.recon.size() == n);

    // ---- 3. set up dequant debug hook env ----
    const char * dir = "/tmp/b5-test";
    setenv("LLAMA_TILE640_DEBUG_DEQUANT_DIR", dir, 1);
    setenv("LLAMA_TILE640_DEBUG_DEQUANT", "1", 1);
    std::string cmd = std::string("rm -rf ") + dir + " && mkdir -p " + dir;
    int sysrc = std::system(cmd.c_str());
    (void) sysrc;
    std::string sidecar_path = std::string(dir) + "/t640_weight.dequant.f32";

    // ---- 4. build a Tile640 matmul graph and run it on Metal ----
    ggml_backend_t backend = ggml_backend_metal_init();
    check("metal backend init", backend != nullptr);
    if (backend == nullptr) {
        std::printf("FAIL: no Metal backend on this host\n");
        return 1;
    }

    struct ggml_init_params ip = { /*.mem_size=*/ 8 * 1024 * 1024,
                                   /*.mem_buffer=*/ nullptr,
                                   /*.no_alloc=*/ true };
    struct ggml_context * gctx = ggml_init(ip);

    // input B: [in_dim, n_tokens=1], F16; the matmul reads it.
    const int64_t n_tokens = 1;
    struct ggml_tensor * B = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, in_dim, n_tokens);

    // 6 Tile640 weight components (must be contiguous, typed per the builder).
    struct ggml_tensor * A_packed      = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,
                                       (int64_t) result.packed.size());
    struct ggml_tensor * A_page_scales = ggml_new_tensor_1d(gctx, GGML_TYPE_F16,
                                       (int64_t) result.page_scales.size());
    struct ggml_tensor * A_lane_scales = ggml_new_tensor_1d(gctx, GGML_TYPE_I8,
                                       (int64_t) result.lane_scales.size());
    struct ggml_tensor * A_outlier_row_offsets = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,
                                       (int64_t) result.outlier_row_offsets.size());
    struct ggml_tensor * A_outlier_cols = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,
                                       (int64_t) result.outlier_cols.size());
    struct ggml_tensor * A_outlier_vals = ggml_new_tensor_1d(gctx, GGML_TYPE_F16,
                                       (int64_t) result.outlier_vals.size());

    ggml_set_name(A_packed, "t640_weight");

    // Allocate all tensors on Metal. The builder produces the result op;
    // ggml_build_forward_expand pulls in the inputs.
    struct ggml_tensor * out = ggml_tile640_matmul(gctx,
        A_packed, A_page_scales, A_lane_scales,
        A_outlier_row_offsets, A_outlier_cols, A_outlier_vals, B);

    // backend buffer covers all graph tensors
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    check("alloc ctx tensors", buf != nullptr);
    if (buf == nullptr) {
        std::printf("FAIL: alloc ctx tensors\n");
        return 1;
    }

    // Upload component data. The quantizer uses uint16 page_scales and
    // outlier_vals in raw f16 bit-pattern; tessera-quant stores them as
    // uint16 vectors, so a direct byte copy into the I32/F16/I8 tensors
    // is correct.
    auto set_tensor = [](struct ggml_tensor * t, const void * data) {
        size_t bytes = ggml_nbytes(t);
        ggml_backend_tensor_set(t, data, 0, bytes);
    };
    set_tensor(A_packed, result.packed.data());
    set_tensor(A_page_scales, result.page_scales.data());
    set_tensor(A_lane_scales, result.lane_scales.data());
    set_tensor(A_outlier_row_offsets, result.outlier_row_offsets.data());
    set_tensor(A_outlier_cols, result.outlier_cols.data());
    set_tensor(A_outlier_vals, result.outlier_vals.data());
    // B: fill with a simple F16 ramp (any non-NaN input is fine; the
    // matmul output is discarded, only the sidecar matters).
    {
        std::vector<uint16_t> bh((size_t)(in_dim * n_tokens), 0);
        for (size_t i = 0; i < bh.size(); i++) {
            // crude f16 of 0.5: sign 0, exp 14, mantissa 0 -> 0x3c00
            bh[i] = 0x3c00;
        }
        set_tensor(B, bh.data());
    }

    // Build and run the graph.
    struct ggml_cgraph * cg = ggml_new_graph(gctx);
    ggml_build_forward_expand(cg, out);

    enum ggml_status st = ggml_backend_graph_compute(backend, cg);
    check("graph compute ok", st == GGML_STATUS_SUCCESS);
    if (st != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: graph compute status=%d\n", (int) st);
        return 1;
    }

    // The L1 sidecar is written asynchronously in a Metal completed
    // handler (see metal-dump-dequant.mm); poll for the file so the test
    // does not race the GPU. The write is small (out_dim*in_dim*4 bytes)
    // so this completes within a few ms in practice.
    bool sidecar_appeared = false;
    for (int poll = 0; poll < 2000; ++poll) {
        FILE * f = fopen(sidecar_path.c_str(), "rb");
        if (f) {
            // wait for the writer to finish (full file: header + strips + data)
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fclose(f);
            // v3 size lower bound: 40-byte header + data
            if (sz >= (long)(40 + n * sizeof(float))) {
                sidecar_appeared = true;
                break;
            }
        }
        usleep(2000); // 2 ms
    }
    check("sidecar file appeared (async handler ran)", sidecar_appeared);

    // ---- 5. read back the GPU sidecar ----
    std::vector<float> gpu_deq;
    int64_t s_rows = 0, s_cols = 0;
    bool ok = read_sidecar_f32(sidecar_path, &gpu_deq, &s_rows, &s_cols);
    check("sidecar written", ok);
    if (!ok) {
        std::printf("FAIL: no sidecar at %s\n", sidecar_path.c_str());
        return 1;
    }
    check("sidecar rows == out_dim", s_rows == out_dim);
    check("sidecar cols == in_dim", s_cols == in_dim);
    check("sidecar data size", (int64_t) gpu_deq.size() == n);

    // ---- 6. CPU reference dequant from the FLAT packed layout ----
    // Build the flat per-row layout the to_float trait expects:
    //   [packed | page_scales | lane_scales] concatenated per row.
    // The GPU uses separate component tensors; this is the host layout
    // dequantize_row_tessera_t640 reads.
    const int64_t words_per_page = 32;
    std::vector<uint32_t> flat_packed((size_t)(out_dim * pages_per_row * words_per_page));
    std::vector<uint16_t> flat_page_scales((size_t)(out_dim * pages_per_row));
    std::vector<int8_t>   flat_lane_scales((size_t)(out_dim * pages_per_row * 32));
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t p = 0; p < pages_per_row; p++) {
            flat_page_scales[(size_t)(r * pages_per_row + p)] =
                result.page_scales[(size_t)(r * pages_per_row + p)];
            for (int l = 0; l < 32; l++) {
                flat_packed[(size_t)((r * pages_per_row + p) * words_per_page + l)] =
                    result.packed[(size_t)((r * pages_per_row + p) * words_per_page + l)];
                flat_lane_scales[(size_t)((r * pages_per_row + p) * 32 + l)] =
                    result.lane_scales[(size_t)((r * pages_per_row + p) * 32 + l)];
            }
        }
    }
    // Assemble the flat row layout per row and dequant one row at a time.
    std::vector<float> cpu_deq((size_t) n, 0.0f);
    for (int64_t r = 0; r < out_dim; r++) {
        std::vector<uint8_t> row_bytes;
        row_bytes.reserve((size_t)(pages_per_row * (32 * 4 + 2 + 32)));
        for (int64_t p = 0; p < pages_per_row; p++) {
            for (int l = 0; l < 32; l++) {
                uint32_t v = flat_packed[(size_t)((r * pages_per_row + p) * 32 + l)];
                row_bytes.insert(row_bytes.end(), (uint8_t *) &v, (uint8_t *) &v + 4);
            }
        }
        for (int64_t p = 0; p < pages_per_row; p++) {
            uint16_t s = flat_page_scales[(size_t)(r * pages_per_row + p)];
            row_bytes.insert(row_bytes.end(), (uint8_t *) &s, (uint8_t *) &s + 2);
        }
        for (int64_t p = 0; p < pages_per_row; p++) {
            for (int l = 0; l < 32; l++) {
                int8_t s = flat_lane_scales[(size_t)((r * pages_per_row + p) * 32 + l)];
                row_bytes.push_back((uint8_t) s);
            }
        }
        dequantize_row_tessera_t640(row_bytes.data(),
                                    cpu_deq.data() + r * in_dim, in_dim);
    }

    // ---- 7. compute deltas ----
    // GPU-vs-CPU(to_float): the key B5 metric. Non-zero expected because
    //   the GPU adds outlier residuals the CPU trait omits.
    double max_abs_gpu_cpu = 0.0, mean_abs_gpu_cpu = 0.0;
    double max_rel_gpu_cpu = 0.0;
    int64_t nz = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = std::fabs((double) gpu_deq[(size_t) i] - (double) cpu_deq[(size_t) i]);
        if (d > max_abs_gpu_cpu) max_abs_gpu_cpu = d;
        mean_abs_gpu_cpu += d;
        double denom = std::fabs((double) gpu_deq[(size_t) i]);
        if (denom > 1e-6) {
            double rel = d / denom;
            if (rel > max_rel_gpu_cpu) max_rel_gpu_cpu = rel;
        }
        if (d > 1e-9) nz++;
    }
    mean_abs_gpu_cpu /= (double) n;

    std::printf("\n=== B5 GPU-vs-CPU dequant delta (key signal) ===\n");
    std::printf("GPU sidecar vs CPU traits->to_float (flat packed, no outliers):\n");
    std::printf("  max abs delta   : %.6g\n", max_abs_gpu_cpu);
    std::printf("  mean abs delta  : %.6g\n", mean_abs_gpu_cpu);
    std::printf("  max rel delta   : %.6g\n", max_rel_gpu_cpu);
    std::printf("  elements differ : %lld / %lld\n", (long long) nz, (long long) n);

    // GPU-vs-W_hat(recon): the L1 fitness numerator basis. The recon
    // includes the AWQ scale; with alpha>0 the GPU dequant (AWQ-scaled)
    // should match recon closely except where outliers were pulled out.
    double max_abs_gpu_what = 0.0, mean_abs_gpu_what = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = std::fabs((double) gpu_deq[(size_t) i] - (double) result.recon[(size_t) i]);
        if (d > max_abs_gpu_what) max_abs_gpu_what = d;
        mean_abs_gpu_what += d;
    }
    mean_abs_gpu_what /= (double) n;
    std::printf("\nGPU sidecar vs quantizer W_hat recon:\n");
    std::printf("  max abs delta   : %.6g\n", max_abs_gpu_what);
    std::printf("  mean abs delta  : %.6g\n", mean_abs_gpu_what);

    // The KEY assertion: the GPU path MUST differ from the CPU
    // to_float path (otherwise we are still cheating). With injected
    // outliers, the delta must be non-trivial.
    check("GPU != CPU to_float (cheat removed)", max_abs_gpu_cpu > 1e-3);

    // Spot-check: at the injected outlier columns, the GPU value should
    // carry the residual while the CPU to_float shows the base dequant.
    bool outlier_seen = false;
    for (int64_t r = 0; r < out_dim && !outlier_seen; r++) {
        int64_t col = 17;
        double d = std::fabs((double) gpu_deq[(size_t)(r * in_dim + col)] -
                             (double) cpu_deq[(size_t)(r * in_dim + col)]);
        if (d > 1e-3) outlier_seen = true;
    }
    check("GPU outlier addback visible at col 17", outlier_seen);

    // ---- 8. dump a few values for the record ----
    std::printf("\nSample row 0, cols 0..4 (GPU, CPU-to_float, W_hat):\n");
    for (int c = 0; c < 5; c++) {
        std::printf("  col %3d: GPU=%+.6g  CPU=%+.6g  W_hat=%+.6g\n",
                    c,
                    (double) gpu_deq[(size_t) c],
                    (double) cpu_deq[(size_t) c],
                    (double) result.recon[(size_t) c]);
    }
    std::printf("Sample row 0, outlier cols 17 & 200:\n");
    for (int c : {17, 200}) {
        std::printf("  col %3d: GPU=%+.6g  CPU=%+.6g  W_hat=%+.6g\n",
                    c,
                    (double) gpu_deq[(size_t) c],
                    (double) cpu_deq[(size_t) c],
                    (double) result.recon[(size_t) c]);
    }

    // ---- 9. L1 fitness sanity: ts_l1_kernel_direct_t2 on the real sidecar ----
    // fitness = ||W_hat - kernel_dequant||_F^2 / ||W||_F^2
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double) result.recon[(size_t) i] - (double) gpu_deq[(size_t) i];
        num += d * d;
        den += (double) w[(size_t) i] * (double) w[(size_t) i];
    }
    float t2 = den > 0.0 ? (float) (num / den) : 0.0f;
    std::printf("\nL1 kernel-direct t_l^2 (GPU sidecar) : %.6g\n", (double) t2);
    check("L1 t2 is non-negative finite", t2 >= 0.0f && std::isfinite(t2));

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    ggml_backend_free(backend);

    std::printf("\n%s (failures=%d)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}

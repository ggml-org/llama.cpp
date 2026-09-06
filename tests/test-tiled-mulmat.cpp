#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#endif

// failed checks; main exits non-zero on any failures
static int n_failed = 0;

static float * gen_rand_f32(int64_t n) {
    float * data = (float *) malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX - 0.5f;
        data[i] *= 5.0f;
    }
    return data;
}

// updates pointers to max_err and rms_err
static void compare_f32(const float * ref, const float * out, int64_t n, float * max_err, float * rms_err) {
    *max_err = 0.0f;
    double sum_sq_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float err = fabsf(ref[i] - out[i]);
        if (err > *max_err) {
            *max_err = err;
        }
        sum_sq_err += (double)err * err;
    }
    *rms_err = sqrt(sum_sq_err / n);
}

// Fill from a flat row-major f32 source: rows * cols floats.
static void fill_tensor(struct ggml_tensor * t, const float * src, int64_t rows, int64_t cols, ggml_type qtype) {
    GGML_ASSERT(t->ne[0] == cols);
    GGML_ASSERT(rows * cols == ggml_nelements(t));
    if (qtype == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
        return;
    }
    void * q = malloc(ggml_nbytes(t));
    // some iq types (iq2_xxs, iq2_xs, iq1_s) steer their code choice with an imatrix; the
    // quantized bytes are shared by the reference and tiled paths, so a dummy imatrix is enough
    const float * imatrix = NULL;
    float * im = NULL;
    if (ggml_quantize_requires_imatrix(qtype)) {
        im = (float *) malloc(rows * cols * sizeof(float));
        for (int64_t i = 0; i < rows * cols; i++) im[i] = 1.0f;
        imatrix = im;
    }
    ggml_quantize_chunk(qtype, src, q, 0, rows, cols, imatrix);
    ggml_backend_tensor_set(t, q, 0, ggml_nbytes(t));
    free(q);
    free(im);
}

// The CPU-specific control API is resolved through the backend registry: with
// GGML_BACKEND_DL the CPU backend is a runtime-loaded module and its symbols
// are not available at link time (same pattern as get_cpu_repack_buft below)
static void cpu_set_use_ref(ggml_backend_t backend, bool use_ref) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_use_ref)(ggml_backend_t, bool) =
        (void (*)(ggml_backend_t, bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
    if (!set_use_ref) {
        fprintf(stderr, "ggml_backend_cpu_set_use_ref not available\n");
        exit(1);
    }
    set_use_ref(backend, use_ref);
}

static void cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_n_threads)(ggml_backend_t, int) =
        (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (!set_n_threads) {
        fprintf(stderr, "ggml_backend_set_n_threads not available\n");
        exit(1);
    }
    set_n_threads(backend, n_threads);
}

static void test_matmul(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, ggml_type quant_type) {
    srand(0xBEEF);

    float * src1_ref = gen_rand_f32(M * N);
    float * src0_ref = gen_rand_f32(N * K);
    float * dst_out = (float *) malloc(M * K * sizeof(float));
    float * dst_tiled = (float *) malloc(M * K * sizeof(float));

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    // ggml_mul_mat(t0, t1) computes t1 * t0^T: t0 (src0, quant) is N wide x
    // K high, t1 (src1, f32) is N wide x M high, dst = (K, M).
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, quant_type,   N, K);

    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * dst   = ggml_mul_mat(ctx, src0, src1);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    fill_tensor(src1, src1_ref, M, N, GGML_TYPE_F32);
    fill_tensor(src0, src0_ref, K, N, quant_type);

    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_out, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_tiled, 0, ggml_nbytes(dst));

    // std vs tiled: identical quantized inputs, so any large difference here is a bug in the tiled kernel
    float max_err, rms_err;
    compare_f32(dst_out, dst_tiled, M*K, &max_err, &rms_err);
    float tol = 1e-3f;

    printf("TEST %lldx%lld * %lldx%lld (%s): %s (max_err: %f, rms: %f, tolerance: %f)\n",
           (long long)M, (long long)N, (long long)N, (long long)K,
           ggml_type_name(quant_type), (max_err <= tol) ? "PASS" : "FAIL", max_err, rms_err, tol);

    // if the tiled kernel deviates from std beyond quantization tolerance,
    // dump a few offenders
    if (max_err > tol) {
        int64_t shown = 0;
        for (int64_t i = 0; i < M*K && shown < 8; ++i) {
            float err = fabsf(dst_out[i] - dst_tiled[i]);
            if (err > tol) {
                printf("  tiled vs std: i=%lld (m=%lld k=%lld) std=%f tiled=%f err=%f\n",
                       (long long)i, (long long)(i/K), (long long)(i%K), dst_out[i], dst_tiled[i], err);
                ++shown;
            }
        }
        ++n_failed;
    }

    ggml_free(ctx);
    free(src1_ref); free(src0_ref); free(dst_out); free(dst_tiled);
}

// Higher-dim (ne[2], ne[3] > 1) check, std (ggml_mul_mat) is trusted as the reference
//
// src1 (F32)  : [N, M, src1_2, src1_3]   ne0=N (reduction), ne1=M (out0)
// src0 (quant): [N, K, src0_2, src0_3]   ne0=N (reduction), ne1=K (out1)
// dst = ggml_mul_mat(src0, src1) : [K, M, src1_2, src1_3]
static void test_matmul_highdim(ggml_backend_t backend, int64_t M, int64_t N, int64_t K,
                         int64_t src1_2, int64_t src1_3,
                         int64_t src0_2, int64_t src0_3,
                         ggml_type quant_type) {
    srand(0xBEEF);

    const int64_t n_src1 = N*M*src1_2*src1_3;
    const int64_t n_src0 = N*K*src0_2*src0_3;
    const int64_t n_dst = K*M*src1_2*src1_3;

    float * src1_ref   = gen_rand_f32(n_src1);
    float * src0_ref   = gen_rand_f32(n_src0);
    float * dst_std   = (float *) malloc(n_dst * sizeof(float));
    float * dst_tiled = (float *) malloc(n_dst * sizeof(float));

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne_src1[4] = { N, M, src1_2, src1_3 };
    int64_t ne_src0[4] = { N, K, src0_2, src0_3 };
    struct ggml_tensor * src1  = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_src1);
    struct ggml_tensor * src0 = ggml_new_tensor(ctx, quant_type,  4, ne_src0);

    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * dst   = ggml_mul_mat(ctx, src0, src1);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // the batch dims fold into the row count (rows are contiguous, ne0 fastest)
    fill_tensor(src1, src1_ref, M * src1_2 * src1_3, N, GGML_TYPE_F32);
    fill_tensor(src0, src0_ref, K * src0_2 * src0_3, N, quant_type);

    // reference = stock path (use_ref keeps the op off the tiled hook), tiled
    // = the gated path; same weights, same op, run twice into separate buffers
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_std, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_tiled, 0, ggml_nbytes(dst));

    // max |dst|: scale for the quantization tolerance
    float scale = 0.0f;
    for (int64_t i = 0; i < n_dst; ++i) {
        scale = fmaxf(scale, fabsf(dst_std[i]));
    }

    // std vs tiled: identical quantized inputs, so a large difference is a bug
    float max_err, rms_err;
    compare_f32(dst_std, dst_tiled, n_dst, &max_err, &rms_err);
    float tol = (quant_type == GGML_TYPE_F32) ? 1e-4f : fmaxf(1e-3f, 1e-3f*scale);

    // which path did the tiled op take (mirrors the hard-constraint gate; the
    // harness runs with force on, so the profitability check is bypassed)
    bool tiled_kernel = (N % 256 == 0) && (src0_2 > 0) && (src0_3 > 0)
        && (src1_2 % src0_2 == 0) && (src1_3 % src0_3 == 0);

    printf("TEST %lldx%lldx%lldx%lld * %lldx%lldx%lldx%lld (%s, %s): %s (max_err: %f, rms: %f, scale: %f)\n",
           (long long)M, (long long)N, (long long)src1_2, (long long)src1_3,
           (long long)K, (long long)N, (long long)src0_2, (long long)src0_3,
           ggml_type_name(quant_type), tiled_kernel ? "tiled-kernel" : "stock",
           (max_err <= tol) ? "PASS" : "FAIL", max_err, rms_err, scale);

    if (max_err > tol) {
        int64_t shown = 0;
        for (int64_t i = 0; i < n_dst && shown < 8; ++i) {
            float err = fabsf(dst_std[i] - dst_tiled[i]);
            if (err > tol) {
                printf("  tiled vs std: i=%lld std=%f tiled=%f err=%f\n",
                       (long long)i, dst_std[i], dst_tiled[i], err);
                ++shown;
            }
        }
        ++n_failed;
    }

    ggml_free(ctx);
    free(src1_ref); free(src0_ref); free(dst_std); free(dst_tiled);
}

// MUL_MAT_ID (MoE) check, std (vec_dot) is trusted as the reference
//
// src0 (as, quant): [K, R, n_experts]   ne0 = K (reduction), ne1 = R rows per expert, ne2 = experts
// src1 (b, F32)   : [K, b_slots, batch]  ne0 = K (reduction), ne1 = b_slots (b rows, broadcast over k columns)
// ids             : [k, batch]           expert picked per (slot, batch row); k % b_slots == 0
// dst = ggml_mul_mat_id(as, b, ids) : [R, k, batch]; column (id, t) = GEMV of src1 row (id % b_slots + t*b_slots)
// against expert ids[t*k+id]. b_slots = 1 is the common MoE case (all top-k experts see one input row).
static void test_mul_mat_id(ggml_backend_t backend, int64_t K, int64_t R, int64_t n_experts,
                            int64_t k, int64_t b_slots, int64_t batch, ggml_type quant_type,
                            bool src1_strided = false) {
    srand(0xBEEF);

    const int64_t n_as  = K * R * n_experts;
    const int64_t n_b   = K * b_slots * batch;
    const int64_t n_dst = R * k * batch;

    float * as_ref  = gen_rand_f32(n_as);
    float * b_ref   = gen_rand_f32(n_b);
    int32_t * ids   = (int32_t *) malloc(k * batch * sizeof(int32_t));
    float * dst_ref = (float *) malloc(n_dst * sizeof(float));
    float * dst_tiled = (float *) malloc(n_dst * sizeof(float));

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne_as[4]  = { K, R, n_experts, 1 };
    int64_t ne_b[4]   = { K, b_slots, batch, 1 };
    int64_t ne_ids[4] = { k, batch, 1, 1 };
    struct ggml_tensor * src0  = ggml_new_tensor(ctx, quant_type,  4, ne_as);
    struct ggml_tensor * ids_t = ggml_new_tensor(ctx, GGML_TYPE_I32, 4, ne_ids);
    struct ggml_tensor * src1;
    struct ggml_tensor * src1_base = NULL;
    if (src1_strided) {
        // strided view: nb[1] = 4 bytes, the repack path reads src1 through strides into wdata
        src1_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, b_slots * batch);
        src1 = ggml_view_3d(ctx, src1_base, K, b_slots, batch, 4, (size_t) b_slots * K * 4, 0);
    } else {
        src1 = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_b);
    }

    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * dst = ggml_mul_mat_id(ctx, src0, src1, ids_t);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // deterministic balanced routing: each expert gets ~ k*batch/n_experts rows
    for (int64_t t = 0; t < batch; t++) {
        for (int64_t e = 0; e < k; e++) {
            ids[t * k + e] = (int32_t) ((t * k + e) % n_experts);
        }
    }

    fill_tensor(src1, b_ref, b_slots * batch, K, GGML_TYPE_F32);
    fill_tensor(src0, as_ref, R * n_experts, K, quant_type);
    ggml_backend_tensor_set(ids_t, ids, 0, ggml_nbytes(ids_t));

    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_ref, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, dst_tiled, 0, ggml_nbytes(dst));

    // same quantized inputs, so a large difference is a bug in the tiled path
    float max_err, rms_err;
    compare_f32(dst_ref, dst_tiled, n_dst, &max_err, &rms_err);
    float tol = 1e-3f;

    printf("TEST mul_mat_id %lldx%lldx%lld * %lldx%lldx%lld ids[%lld,%lld] (%s%s): %s (max_err: %f, rms: %f, tolerance: %f)\n",
           (long long)K, (long long)R, (long long)n_experts,
           (long long)K, (long long)b_slots, (long long)batch,
           (long long)k, (long long)batch,
           ggml_type_name(quant_type), src1_strided ? ", strided-b" : "",
           (max_err <= tol) ? "PASS" : "FAIL", max_err, rms_err, tol);

    if (max_err > tol) {
        int64_t shown = 0;
        for (int64_t i = 0; i < n_dst && shown < 8; ++i) {
            float err = fabsf(dst_ref[i] - dst_tiled[i]);
            if (err > tol) {
                printf("  tiled vs std: i=%lld (row=%lld slot=%lld batch=%lld) std=%f tiled=%f err=%f\n",
                       (long long)i, (long long)(i % R), (long long)((i / R) % k), (long long)(i / (R * k)),
                       dst_ref[i], dst_tiled[i], err);
                ++shown;
            }
        }
        ++n_failed;
    }

    ggml_free(ctx);
    free(as_ref); free(b_ref); free(ids); free(dst_ref); free(dst_tiled);
}

static double time_graph_compute(ggml_backend_t backend, struct ggml_cgraph * gf) {
#if defined(_WIN32)
    // high-res timer; clock_gettime is not portable across Windows toolchains
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    ggml_backend_graph_compute(backend, gf);
    QueryPerformanceCounter(&t1);
    return (double) (t1.QuadPart - t0.QuadPart) / (double) freq.QuadPart;
#else
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ggml_backend_graph_compute(backend, gf);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
#endif
}

// Warm up, then time n runs and return the best
static double time_graph_compute_best(ggml_backend_t backend, struct ggml_cgraph * gf, int n,
                                      void * flush_buf = NULL, size_t flush_size = 0) {

    time_graph_compute(backend, gf); // warmup
    double best = 1e30;
    for (int i = 0; i < n; ++i) {
        // Evict L3 cache first
        if (flush_buf) { memset(flush_buf, 0xAB, flush_size); }
        const double t = time_graph_compute(backend, gf);
        if (t < best) best = t;
    }
    return best;
}

// Fetch the repack extra buffer type through the public proc-address API
static ggml_backend_buffer_type_t get_cpu_repack_buft(void) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        return NULL;
    }
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    ggml_backend_dev_get_extra_bufts_t get_extra =
        (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (!get_extra) {
        return NULL;
    }
    ggml_backend_buffer_type_t * bufts = get_extra(cpu_dev);
    // the only extra buffer type the CPU backend exposes is CPU_REPACK
    return (bufts && *bufts) ? *bufts : NULL;
}

struct bench_row {
    const char * name;
    double time_std, time_repack, time_tiled;
    float max_err_repack, rmse_repack;
    float max_err_tiled, rmse_tiled;
    bool have_repack;
};

// One timed run of each path (std, repacked, tiled) plus max error and RMSE vs the std
// output. The repack column is n/a where no repack kernel exists for the type
static bench_row bench_three_way(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, ggml_type quant_type) {
    bench_row row;
    row.name = ggml_type_name(quant_type);
    row.time_std = row.time_repack = row.time_tiled = 0.0;
    row.max_err_repack = row.rmse_repack = row.max_err_tiled = row.rmse_tiled = 0.0f;
    row.have_repack = false;

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * src1     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * src0_std = ggml_new_tensor_2d(ctx, quant_type,  N, K);
    struct ggml_tensor * src0_rep = ggml_new_tensor_2d(ctx, quant_type,  N, K);

    // std and tiled are the same op on the same tensors; one graph serves
    // both timings (use_ref selects the stock path)
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, src0_std, src1);
    ggml_build_forward_expand(gf, dst);

    struct ggml_cgraph * gf_repack = ggml_new_graph(ctx);
    struct ggml_tensor * dst_repack  = ggml_mul_mat(ctx, src0_rep, src1);
    ggml_build_forward_expand(gf_repack, dst_repack);

    ggml_backend_buffer_type_t repack_buft = get_cpu_repack_buft();
    if (repack_buft && N % 8 == 0 && K % 8 == 0) {
        ggml_backend_buffer_t buf_rep = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(src0_rep));
        src0_rep->buffer = buf_rep;
        src0_rep->data   = ggml_backend_buffer_get_base(buf_rep);
        ggml_backend_buffer_init_tensor(buf_rep, src0_rep);
        row.have_repack = (src0_rep->extra != NULL);
    }

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    srand(0xBEEF);
    float * src1_data = gen_rand_f32(M * N);
    float * src0_data = gen_rand_f32(N * K);
    fill_tensor(src1, src1_data, M, N, GGML_TYPE_F32);
    fill_tensor(src0_std, src0_data, K, N, quant_type);
    if (row.have_repack) {
        // same f32 source, so both weight tensors hold identical quants; the
        // repack buffer's set_tensor repacks them in-place
        fill_tensor(src0_rep, src0_data, K, N, quant_type);
    }
    free(src1_data); free(src0_data);

    // one warmup + N timings per path; best (min) wins
    // the flush buffer is larger than the whole-chip cache, so its memset
    // evicts L3/V-Cache and each timed iteration streams weights from DRAM
    const size_t flush_size = 256 * 1024 * 1024;
    void * flush_buf = malloc(flush_size);
    const int n_reps = 5;
    cpu_set_use_ref(backend, true);
    row.time_std    = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);
    cpu_set_use_ref(backend, false);
    if (row.have_repack) {
        row.time_repack = time_graph_compute_best(backend, gf_repack, n_reps, flush_buf, flush_size);
    }
    row.time_tiled  = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);
    free(flush_buf);

    // errors vs the standard output (all paths use the same quantized weights)
    float * out_std    = (float *) malloc(M * K * sizeof(float));
    float * out_tiled  = (float *) malloc(M * K * sizeof(float));
    float * out_repack = (float *) malloc(M * K * sizeof(float));
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, out_std, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, out_tiled, 0, ggml_nbytes(dst));
    if (row.have_repack) {
        ggml_backend_graph_compute(backend, gf_repack);
        ggml_backend_tensor_get(dst_repack, out_repack, 0, ggml_nbytes(dst_repack));
        compare_f32(out_std, out_repack, M * K, &row.max_err_repack, &row.rmse_repack);
    }
    compare_f32(out_std, out_tiled, M * K, &row.max_err_tiled, &row.rmse_tiled);
    free(out_std); free(out_tiled); free(out_repack);

    ggml_free(ctx);
    return row;
}

static void print_bench_table(int64_t M, int64_t N, int64_t K, const bench_row * rows, size_t n_types) {
    const double flops = 2.0 * M * N * K;

    printf("\nBENCH %lldx%lld * %lldx%lld, min of 5 timings, 8 threads\n",
           (long long)M, (long long)N, (long long)N, (long long)K);
    printf("%-8s %10s %12s %12s %11s %11s %17s %17s %17s %17s\n",
           "type", "std TF", "repack TF", "tiled TF", "repack/std", "tiled/std",
           "max_err(repack)", "rmse(repack)", "max_err(tiled)", "rmse(tiled)");
    for (size_t i = 0; i < n_types; ++i) {
        const bench_row * r = &rows[i];
        if (r->have_repack) {
            printf("%-8s %10.3f %12.3f %12.3f %11.2f %11.2f %17.5e %17.5e %17.5e %17.5e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   flops / (r->time_repack * 1e12), flops / (r->time_tiled * 1e12),
                   r->time_std / r->time_repack, r->time_std / r->time_tiled,
                   r->max_err_repack, r->rmse_repack, r->max_err_tiled, r->rmse_tiled);
        } else {
            printf("%-8s %10.3f %12s %12.3f %11s %11.2f %17s %17s %17.5e %17.5e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   "n/a", flops / (r->time_tiled * 1e12), "n/a",
                   r->time_std / r->time_tiled,
                   "n/a", "n/a", r->max_err_tiled, r->rmse_tiled);
        }
    }
}

struct bench_row_mmid {
    const char * name;
    double time_std, time_repack, time_tiled;
    float max_err_repack, rmse_repack;
    float max_err_tiled, rmse_tiled;
    bool have_repack;
};

// MUL_MAT_ID (MoE) three-way bench: std (use_ref) vs repack (CPU_REPACK buffer) vs
// tiled kernel (default; TILED_MM_FORCE is set in main so ragged experts take it too).
// Column (id, t) = GEMV of src1 row (id % b_slots + t * b_slots) against expert ids[t*k+id];
// cne1 = k * batch / n_experts rows per expert on average
static bench_row_mmid bench_mul_mat_id(ggml_backend_t backend, int64_t K, int64_t R, int64_t n_experts,
                                       int64_t k, int64_t b_slots, int64_t batch, ggml_type quant_type) {
    bench_row_mmid row;
    row.name = ggml_type_name(quant_type);
    row.time_std = row.time_repack = row.time_tiled = 0.0;
    row.max_err_repack = row.rmse_repack = row.max_err_tiled = row.rmse_tiled = 0.0f;
    row.have_repack = false;

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    int64_t ne_as[4]  = { K, R, n_experts, 1 };
    int64_t ne_b[4]   = { K, b_slots, batch, 1 };
    int64_t ne_ids[4] = { k, batch, 1, 1 };
    struct ggml_tensor * src1     = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_b);
    struct ggml_tensor * src0_std = ggml_new_tensor(ctx, quant_type,  4, ne_as);
    struct ggml_tensor * src0_rep = ggml_new_tensor(ctx, quant_type,  4, ne_as);
    struct ggml_tensor * ids_t    = ggml_new_tensor(ctx, GGML_TYPE_I32, 4, ne_ids);

    // std and tiled are the same op on the same tensors; one graph serves
    // both timings (use_ref selects the stock path)
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * dst = ggml_mul_mat_id(ctx, src0_std, src1, ids_t);
    ggml_build_forward_expand(gf, dst);

    struct ggml_cgraph * gf_repack = ggml_new_graph(ctx);
    struct ggml_tensor * dst_repack  = ggml_mul_mat_id(ctx, src0_rep, src1, ids_t);
    ggml_build_forward_expand(gf_repack, dst_repack);

    ggml_backend_buffer_type_t repack_buft = get_cpu_repack_buft();
    if (repack_buft && R % 8 == 0) {
        ggml_backend_buffer_t buf_rep = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(src0_rep));
        src0_rep->buffer = buf_rep;
        src0_rep->data   = ggml_backend_buffer_get_base(buf_rep);
        ggml_backend_buffer_init_tensor(buf_rep, src0_rep);
        row.have_repack = (src0_rep->extra != NULL);
    }

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // deterministic balanced routing: each expert gets ~ k*batch/n_experts rows
    int32_t * ids = (int32_t *) malloc(k * batch * sizeof(int32_t));
    for (int64_t t = 0; t < batch; t++) {
        for (int64_t e = 0; e < k; e++) {
            ids[t * k + e] = (int32_t) ((t * k + e) % n_experts);
        }
    }
    srand(0xBEEF);
    float * b_data  = gen_rand_f32(K * b_slots * batch);
    float * as_data = gen_rand_f32(K * R * n_experts);
    fill_tensor(src1, b_data, b_slots * batch, K, GGML_TYPE_F32);
    fill_tensor(src0_std, as_data, R * n_experts, K, quant_type);
    if (row.have_repack) {
        // same f32 source, so both weight tensors hold identical quants; the
        // repack buffer's set_tensor repacks them in-place
        fill_tensor(src0_rep, as_data, R * n_experts, K, quant_type);
    }
    ggml_backend_tensor_set(ids_t, ids, 0, ggml_nbytes(ids_t));
    free(b_data); free(as_data);

    const size_t flush_size = 256 * 1024 * 1024;
    void * flush_buf = malloc(flush_size);
    const int n_reps = 5;
    cpu_set_use_ref(backend, true);
    row.time_std = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);
    cpu_set_use_ref(backend, false);
    if (row.have_repack) {
        row.time_repack = time_graph_compute_best(backend, gf_repack, n_reps, flush_buf, flush_size);
    }
    row.time_tiled = time_graph_compute_best(backend, gf, n_reps, flush_buf, flush_size);
    free(flush_buf);

    // errors vs the standard output (all paths use the same quantized weights)
    float * out_std    = (float *) malloc(R * k * batch * sizeof(float));
    float * out_tiled  = (float *) malloc(R * k * batch * sizeof(float));
    float * out_repack = (float *) malloc(R * k * batch * sizeof(float));
    cpu_set_use_ref(backend, true);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, out_std, 0, ggml_nbytes(dst));
    cpu_set_use_ref(backend, false);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(dst, out_tiled, 0, ggml_nbytes(dst));
    if (row.have_repack) {
        ggml_backend_graph_compute(backend, gf_repack);
        ggml_backend_tensor_get(dst_repack, out_repack, 0, ggml_nbytes(dst_repack));
        compare_f32(out_std, out_repack, R * k * batch, &row.max_err_repack, &row.rmse_repack);
    }
    compare_f32(out_std, out_tiled, R * k * batch, &row.max_err_tiled, &row.rmse_tiled);
    free(out_std); free(out_tiled); free(out_repack); free(ids);

    ggml_free(ctx);
    return row;
}

static void print_mmid_table(int64_t K, int64_t R, int64_t n_experts, int64_t k,
                             int64_t b_slots, int64_t batch, const bench_row_mmid * rows, size_t n_types) {
    const double flops = 2.0 * (double) K * R * k * batch;

    printf("\nBENCH mmid %lldx%lldx%lld k=%lld b_slots=%lld batch=%lld (cne1=%lld), min of 5 timings, 8 threads\n",
           (long long)K, (long long)R, (long long)n_experts, (long long)k,
           (long long)b_slots, (long long)batch, (long long)(k * batch / n_experts));
    printf("%-8s %10s %12s %12s %11s %11s %17s %17s %17s %17s\n",
           "type", "std TF", "repack TF", "tiled TF", "repack/std", "tiled/std",
           "max_err(repack)", "rmse(repack)", "max_err(tiled)", "rmse(tiled)");
    for (size_t i = 0; i < n_types; ++i) {
        const bench_row_mmid * r = &rows[i];
        if (r->have_repack) {
            printf("%-8s %10.3f %12.3f %12.3f %11.2f %11.2f %17.5e %17.5e %17.5e %17.5e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   flops / (r->time_repack * 1e12), flops / (r->time_tiled * 1e12),
                   r->time_std / r->time_repack, r->time_std / r->time_tiled,
                   r->max_err_repack, r->rmse_repack, r->max_err_tiled, r->rmse_tiled);
        } else {
            printf("%-8s %10.3f %12s %12.3f %11s %11.2f %17s %17s %17.5e %17.5e\n",
                   r->name,
                   flops / (r->time_std * 1e12),
                   "n/a", flops / (r->time_tiled * 1e12), "n/a",
                   r->time_std / r->time_tiled,
                   "n/a", "n/a", r->max_err_tiled, r->rmse_tiled);
        }
    }
}

int main(int argc, char ** argv) {
    bool run_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) {
            run_bench = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    // Enable tiled MM, also force tiled MM even when unprofitable for benchmarks
#if defined(_MSC_VER)
    _putenv("GGML_CPU_TILED_MM=1");
    _putenv("GGML_CPU_TILED_MM_FORCE=1");
#else
    setenv("GGML_CPU_TILED_MM", "1", 1);
    setenv("GGML_CPU_TILED_MM_FORCE", "1", 1);
#endif

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        fprintf(stderr, "failed to initialize the CPU backend\n");
        return 1;
    }
    cpu_set_n_threads(backend, 8);
    // base case, smoke each quant type
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q6_K);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q5_K);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q4_K);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q3_K);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_Q2_K);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_IQ4_XS);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_IQ2_XXS);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_IQ2_XS);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_IQ2_S);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_IQ3_XXS);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_IQ3_S);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_IQ1_S);
    test_matmul(backend, 512, 1024, 512, GGML_TYPE_IQ1_M);

    // ragged edges, both subblock lengths (Q5_K = 32, Q6_K = 16)
    test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q6_K);   // long K, int32 accumulation
    test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q6_K);    // ragged M and K
    test_matmul(backend, 16, 1024, 16, GGML_TYPE_Q5_K);
    test_matmul(backend, 18, 1024, 7, GGML_TYPE_Q5_K);       // ragged M tail, ragged K in first subblock
    test_matmul(backend, 8, 256, 8, GGML_TYPE_Q5_K);
    test_matmul(backend, 18, 1024, 256, GGML_TYPE_Q5_K);
    test_matmul(backend, 9, 256, 256, GGML_TYPE_Q5_K);
    test_matmul(backend, 8, 256, 8, GGML_TYPE_Q6_K);         // tiny M, single QK_K block
    test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q3_K);
    test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q3_K);
    test_matmul(backend, 8, 256, 8, GGML_TYPE_Q3_K);
    test_matmul(backend, 17, 1024, 257, GGML_TYPE_Q3_K);     // K past a full 256 tile
    test_matmul(backend, 257, 1024, 17, GGML_TYPE_Q3_K);     // K past a full microtile
    test_matmul(backend, 256, 1024, 8192, GGML_TYPE_Q2_K);
    test_matmul(backend, 357, 1024, 137, GGML_TYPE_Q2_K);
    test_matmul(backend, 8, 256, 8, GGML_TYPE_Q2_K);
    test_matmul(backend, 17, 1024, 257, GGML_TYPE_Q2_K);
    test_matmul(backend, 257, 1024, 17, GGML_TYPE_Q2_K);
    test_matmul(backend, 357, 1024, 137, GGML_TYPE_IQ4_XS);    // ragged M and K
    test_matmul(backend,   8,  256,   8, GGML_TYPE_IQ4_XS);    // tiny M, single QK_K block
    test_matmul(backend,  17, 1024, 257, GGML_TYPE_IQ4_XS);    // K past a full 256 tile

    // fuzz: M/K around the microtile (16) and tile (256) boundaries
    // Q4_K = subblock 32, Q6_K = subblock 16
    test_matmul(backend, 1, 1024, 1, GGML_TYPE_Q4_K);
    test_matmul(backend, 2, 1024, 3, GGML_TYPE_Q4_K);
    test_matmul(backend, 15, 1024, 15, GGML_TYPE_Q4_K);      // M and K in the first microtile
    test_matmul(backend, 17, 1024, 17, GGML_TYPE_Q4_K);      // M and K past the first microtile
    test_matmul(backend, 255, 1024, 255, GGML_TYPE_Q4_K);    // M and K one short of a tile
    test_matmul(backend, 257, 1024, 257, GGML_TYPE_Q4_K);    // M and K one past a tile
    test_matmul(backend, 271, 1024, 271, GGML_TYPE_Q4_K);    // 1 tile + 15
    test_matmul(backend, 272, 1024, 272, GGML_TYPE_Q4_K);    // 1 tile + 16
    test_matmul(backend, 511, 1024, 511, GGML_TYPE_Q4_K);    // 2 tiles - 1
    test_matmul(backend, 513, 1024, 513, GGML_TYPE_Q4_K);    // 2 tiles + 1
    test_matmul(backend, 17, 512, 257, GGML_TYPE_Q4_K);      // ragged M and K, N = 2 blocks

    test_matmul(backend, 1, 1024, 1, GGML_TYPE_Q6_K);
    test_matmul(backend, 2, 1024, 3, GGML_TYPE_Q6_K);
    test_matmul(backend, 15, 1024, 15, GGML_TYPE_Q6_K);
    test_matmul(backend, 17, 1024, 17, GGML_TYPE_Q6_K);
    test_matmul(backend, 255, 1024, 255, GGML_TYPE_Q6_K);
    test_matmul(backend, 257, 1024, 257, GGML_TYPE_Q6_K);
    test_matmul(backend, 271, 1024, 271, GGML_TYPE_Q6_K);
    test_matmul(backend, 272, 1024, 272, GGML_TYPE_Q6_K);
    test_matmul(backend, 511, 1024, 511, GGML_TYPE_Q6_K);
    test_matmul(backend, 513, 1024, 513, GGML_TYPE_Q6_K);
    test_matmul(backend, 17, 512, 257, GGML_TYPE_Q6_K);

    // MUL_MAT_ID (MoE): K = reduction (tiled gate needs K % 256 == 0), R = output rows per expert,
    // k = top-k slots, b_slots = b rows (1 = broadcast MoE, k = i11-diverse gather), cne1 = k*batch/E
    test_mul_mat_id(backend,  256,  256,   2, 8, 1,  64, GGML_TYPE_Q4_K);  // single block, cne1 = 256
    test_mul_mat_id(backend,  512,  300,   4, 2, 1, 150, GGML_TYPE_Q4_K);  // ragged R, broadcast, cne1 = 75
    test_mul_mat_id(backend,  512,  256,   4, 2, 2, 150, GGML_TYPE_Q5_K);  // i11-diverse gather, HAS_MIN
    test_mul_mat_id(backend,  512,  300,   4, 2, 1, 150, GGML_TYPE_IQ2_XXS); // iq type, imatrix quant
    test_mul_mat_id(backend,  512,  256,   4, 2, 1, 100, GGML_TYPE_Q6_K);  // ragged cne1 = 50
    test_mul_mat_id(backend,  768,  256,   4, 2, 1, 150, GGML_TYPE_Q3_K);  // K = 3 slabs
    test_mul_mat_id(backend, 1024, 2048, 128, 8, 1, 2048, GGML_TYPE_Q4_K); // large, cne1 = 128
    test_mul_mat_id(backend,  512,  256,  32, 2, 1,   8, GGML_TYPE_Q4_K);  // tiny cne1 (~0.5), forced only

    // cne1 > 256: the k-outer ring sweeps multiple 256-row windows per expert (the multi-window path)
    test_mul_mat_id(backend,  256,  256,   8, 8, 1,  512, GGML_TYPE_Q4_K);  // cne1 = 512, two full windows
    test_mul_mat_id(backend,  512,  256,   8, 8, 1,  300, GGML_TYPE_Q4_K);  // cne1 = 300, 2nd window ragged (44 rows)
    test_mul_mat_id(backend,  256,   64,   8, 8, 1,  257, GGML_TYPE_Q4_K);  // cne1 = 257, 1-row tail window, R = 1 group
    test_mul_mat_id(backend,  512,  128,   8, 8, 1,  511, GGML_TYPE_Q4_K);  // cne1 = 511, 15-row ragged tail
    test_mul_mat_id(backend,  768,  300,   8, 8, 1,  300, GGML_TYPE_Q4_K);  // 3 K slabs x 2 windows, ragged R and cne1
    test_mul_mat_id(backend,  768,  300,   8, 2, 2,  300, GGML_TYPE_Q5_K);  // cne1 = 600, 3 slabs, 3 windows (last ragged), i11-diverse, HAS_MIN

    // cne1 = 255: one short of a full ring, last ring row zero-padded by the unpack
    test_mul_mat_id(backend,  512,  256,   4, 4, 1,  255, GGML_TYPE_Q4_K);

    // long K: 8 slabs per ring row
    test_mul_mat_id(backend, 2048,  300,   8, 2, 1,  400, GGML_TYPE_Q4_K);  // cne1 = 100

    // R group edges: 64 = TILED_MMID_GROUP (ceil div, ragged group tail, single-group early return)
    test_mul_mat_id(backend,  512,   64,   4, 2, 1,  128, GGML_TYPE_Q4_K);  // ngroups = 1, most threads idle
    test_mul_mat_id(backend,  512,   65,   4, 2, 1,  128, GGML_TYPE_Q4_K);  // ceil-div boundary, 1-row tail group
    test_mul_mat_id(backend,  512,   63,   4, 2, 1,  128, GGML_TYPE_Q4_K);  // ragged group tail
    test_mul_mat_id(backend,  512,    1,   4, 1, 1,  256, GGML_TYPE_Q4_K);  // R = 1 row, cne1 = 256

    // strided (non-contiguous) F32 src1: nb[1] = 4 bytes, the repack reads through strides
    test_mul_mat_id(backend,  512,  256,   8, 8, 1,  128, GGML_TYPE_Q4_K, true);  // cne1 = 128

    // higher-dim (ne[2], ne[3] > 1)
    // equal-batch and broadcast shapes (src1_2=2*src0_2, src1_3=2*src0_3)
    test_matmul_highdim(backend, 1024, 1024, 1024, 2, 1, 2, 1, GGML_TYPE_Q4_K); // 3D, tiled
    test_matmul_highdim(backend, 1024, 1024, 1024, 1, 2, 1, 2, GGML_TYPE_Q4_K); // 3D, tiled
    test_matmul_highdim(backend,  512, 1024,  512, 2, 2, 2, 2, GGML_TYPE_Q4_K); // 4D, tiled
    test_matmul_highdim(backend, 1024, 1024, 1024, 2, 1, 1, 1, GGML_TYPE_Q4_K); // broadcast r2=2 (src1_2>src0_2)
    test_matmul_highdim(backend, 1024, 1024, 1024, 2, 2, 2, 1, GGML_TYPE_Q4_K); // broadcast r3=2 (src1_3>src0_3)
    test_matmul_highdim(backend, 1024, 1024, 1024, 2, 1, 2, 1, GGML_TYPE_Q6_K); // 3D, tiled, q6_K
    test_matmul_highdim(backend, 1024, 1024, 1024, 2, 2, 2, 2, GGML_TYPE_Q3_K); // 4D, tiled, q3_K
    test_matmul_highdim(backend, 1024, 1024, 1024, 2, 2, 2, 2, GGML_TYPE_Q2_K); // 4D, tiled, q2_K

    // benchmarks: std, repack and tiled per quant type and shape, with max error /
    // RMSE vs the std output. Every type the tiled gate accepts; the repack column is
    // n/a where no repack kernel exists for the type. TILED_MM_FORCE is set in main,
    // so the tiled column runs at every shape; the shapes vary K, the output rank

    if (run_bench) {
        const ggml_type bench_types[] = { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                          GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                                          GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M };
        const size_t n_types = sizeof(bench_types) / sizeof(bench_types[0]);
        struct { int64_t M, N, K; } shapes[] = {
            //{ 8192, 8192, 8192 },
            { 4096, 4096, 4096 },
            { 4096, 4096,   64 },
            { 4096, 4096,   32 },
            { 4096, 4096,   24 },
            { 4096, 4096,   16 },
            { 4096, 4096,   8 },
            { 4096, 4096,   1 },
        };
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
            bench_row rows[n_types];
            for (size_t i = 0; i < n_types; ++i) {
                rows[i] = bench_three_way(backend, shapes[s].M, shapes[s].N, shapes[s].K, bench_types[i]);
            }
            print_bench_table(shapes[s].M, shapes[s].N, shapes[s].K, rows, n_types);
        }

        // MUL_MAT_ID (MoE) three-way bench (std, repack, tiled); K must be a multiple of
        // 256 (the tiled slab). Every type the tiled gate accepts; the repack column is
        // n/a where no repack kernel exists for the type. cne1 = 1 is below the tiled min
        // batch (32); TILED_MM_FORCE is set in main, so that row is the forced GEMV extreme
        struct { int64_t K, R, E, k, b_slots, batch; } mmid_shapes[] = {
            { 1024, 1024,  8,   8, 1,    1 },  // cne1 = 1, single-token decode, one routed row per expert
            {  512,  512,  8,   2, 1,  128 },  // cne1 = 32, the tiled min batch
            { 1024, 1024, 16,   8, 1,   64 },  // cne1 = 32
            { 1024, 1024, 16,   8, 1,  256 },  // cne1 = 128
            { 1024, 2048, 32,   8, 1,  512 },  // cne1 = 128, wide experts
            { 2048, 1024, 16,   8, 1, 1024 },  // cne1 = 512, long dot
        };
        const ggml_type mmid_types[] = { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                         GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                                         GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M };
        const size_t n_mmid_types = sizeof(mmid_types) / sizeof(mmid_types[0]);
        for (size_t s = 0; s < sizeof(mmid_shapes) / sizeof(mmid_shapes[0]); ++s) {
            bench_row_mmid rows[n_mmid_types];
            for (size_t i = 0; i < n_mmid_types; ++i) {
                rows[i] = bench_mul_mat_id(backend, mmid_shapes[s].K, mmid_shapes[s].R, mmid_shapes[s].E,
                                           mmid_shapes[s].k, mmid_shapes[s].b_slots, mmid_shapes[s].batch, mmid_types[i]);
            }
            print_mmid_table(mmid_shapes[s].K, mmid_shapes[s].R, mmid_shapes[s].E, mmid_shapes[s].k,
                             mmid_shapes[s].b_slots, mmid_shapes[s].batch, rows, n_mmid_types);
        }
    }

    ggml_backend_free(backend);
    return n_failed ? 1 : 0;
}

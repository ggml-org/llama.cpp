// perf-tiled-mulmat: a minimal single-op driver to isolate the tiled vs iqp
// mul_mat kernels under perf, with no correctness checks or bench-table noise.
//
// It runs ONE matmul of a fixed shape (default 4096^3, Q5_K) in a tight loop so
// the profile is dominated by the kernel. The path (std / tiled / iqp) is
// selected with --path and forced with the process env switches:
//
//   std   : use_ref (stock vec_dot reference)
//   tiled : GGML_CPU_TILED_MM=1, FORCE=1, GGML_CPU_MM_PATH=tiled
//   iqp   : GGML_CPU_TILED_MM=0  (tiled master switch is process-static and the
//           tiled gate ignores MM_PATH, so tiled must be fully off for the iqp
//           panel to take the op), GGML_CPU_MM_PATH=iqp
//
// One path per process: the env switches are read once at first use, so a single
// run cannot A/B two paths. For a routing self-check, run all three and compare
// the checksum lines: three distinct checksums prove three distinct code paths.
//
// FINDING (4096^3 Q5_K, 8 threads, this part): the old A/B harness's "iqp" column
// (tiled ON + MM_PATH=iqp) is bit-identical to the tiled kernel (see --path
// harness-iqp). The tiled gate ignores MM_PATH, so that config ran tiled twice and
// the reported "tiled == iqp" was an artifact. The real kernels are not equal:
//   std   0.70 TF (vec_dot ref)
//   iqp   2.03 TF (panel, 256-bit dpbusd; tiled off)
//   tiled 4.68 TF (VNNI 8x16, 512-bit dpbusd)
// tiled/iqp ~= 2.3x, matching the dpbusd.512 = 2x dpbusd.256 madd rate (see
// microkernel.md). Both are dpbusd/compute-bound, not load-bound.
//
// Usage:
//   perf-tiled-mulmat --path tiled --M 4096 --N 4096 --K 4096 --iters 100 --threads 8
//
// Typical perf invocation (pin to the bench cores; one path at a time, e.g.)
//   taskset -c 0-15 perf stat -e cycles,instructions,cache-misses,cache-references ./build-r/tests/perf-tiled-mulmat --path tiled
//   taskset -c 0-15 perf stat -e cycles,instructions,cache-misses,cache-references ./build-r/tests/perf-tiled-mulmat --path iqp

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
#else
#include <unistd.h>
#endif

static float * gen_rand_f32(int64_t n, unsigned int seed) {
    float * data = (float *) malloc(n * sizeof(float));
    srand(seed);
    for (int64_t i = 0; i < n; ++i) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 5.0f;
    }
    return data;
}

// Fill from a flat row-major f32 source: rows * cols floats
static void fill_tensor(struct ggml_tensor * t, const float * src, int64_t rows, int64_t cols, ggml_type qtype) {
    GGML_ASSERT(t->ne[0] == cols);
    GGML_ASSERT(rows * cols == ggml_nelements(t));
    if (qtype == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
        return;
    }
    void * q = malloc(ggml_nbytes(t));
    ggml_quantize_chunk(qtype, src, q, 0, rows, cols, NULL);
    ggml_backend_tensor_set(t, q, 0, ggml_nbytes(t));
    free(q);
}

static void cpu_set_use_ref(ggml_backend_t backend, bool use_ref) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_use_ref)(ggml_backend_t, bool) =
        (void (*)(ggml_backend_t, bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
    if (!set_use_ref) { fprintf(stderr, "ggml_backend_cpu_set_use_ref not available\n"); exit(1); }
    set_use_ref(backend, use_ref);
}

static void cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    void (* set_n_threads)(ggml_backend_t, int) =
        (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (!set_n_threads) { fprintf(stderr, "ggml_backend_set_n_threads not available\n"); exit(1); }
    set_n_threads(backend, n_threads);
}

static double now_s(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double) t.QuadPart / (double) f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

enum path { PATH_STD, PATH_TILED, PATH_IQP, PATH_HARNESS_IQP };

static const char * path_name(enum path p) {
    switch (p) {
        case PATH_STD:           return "std (vec_dot ref)";
        case PATH_TILED:         return "tiled (VNNI 8x16)";
        case PATH_IQP:           return "iqp (panel) [tiled off]";
        case PATH_HARNESS_IQP:   return "harness-iqp [tiled ON + MM_PATH=iqp]";
    }
    return "?";
}
// per-path env; the iqp panel is forced cleanly by turning the tiled master switch
// off. PATH_HARNESS_IQP reproduces the old A/B harness, which set MM_PATH=iqp but
// left tiled on: the tiled gate ignores MM_PATH, so that combo actually runs tiled
// (this path exists to demonstrate that collapse)
static void path_env(enum path p, const char ** tiled_mm, const char ** force, const char ** mm_path, bool * use_ref) {
    switch (p) {
        case PATH_STD:         *tiled_mm="1"; *force="0"; *mm_path="tiled"; *use_ref=true;  break;
        case PATH_TILED:       *tiled_mm="1"; *force="1"; *mm_path="tiled"; *use_ref=false; break;
        case PATH_IQP:         *tiled_mm="0"; *force="0"; *mm_path="iqp";   *use_ref=false; break;
        case PATH_HARNESS_IQP: *tiled_mm="1"; *force="1"; *mm_path="iqp";   *use_ref=false; break;
    }
}

int main(int argc, char ** argv) {
    enum path path      = PATH_TILED;
    int64_t   M = 4096, N = 4096, K = 4096;
    int       iters   = 100;
    int       threads = 8;
    bool      flush   = false;
    unsigned  seed    = 0xBEEF;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--path"))    { if (!strcmp(argv[++i], "std"))          path = PATH_STD;
                                                  else if (!strcmp(argv[i], "tiled"))      path = PATH_TILED;
                                                  else if (!strcmp(argv[i], "iqp"))        path = PATH_IQP;
                                                  else if (!strcmp(argv[i], "harness-iqp")) path = PATH_HARNESS_IQP;
                                                  else { fprintf(stderr, "bad --path %s\n", argv[i]); return 1; } }
        else if (!strcmp(argv[i], "--M"))       M = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--N"))       N = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--K"))       K = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--iters"))   iters   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--flush"))   flush   = true;
        else if (!strcmp(argv[i], "--seed"))    seed    = (unsigned)atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }

    if (M < 8 || N % 8 != 0 || K % 256 != 0) {
        fprintf(stderr, "need M>=8, N%%8==0, K%%256==0 for the iqp/tiled gates (got M=%lld N=%lld K=%lld)\n",
                (long long)M, (long long)N, (long long)K);
        return 1;
    }

    // ---- force the selected path via process env (must be set before first use) ----
    const char * tiled_mm; const char * force; const char * mm_path; bool use_ref;
    path_env(path, &tiled_mm, &force, &mm_path, &use_ref);
    setenv("GGML_CPU_TILED_MM",       tiled_mm, 1);
    setenv("GGML_CPU_TILED_MM_FORCE", force,    1);
    setenv("GGML_CPU_MM_PATH",        mm_path,  1);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) { fprintf(stderr, "failed to init CPU backend\n"); return 1; }
    cpu_set_n_threads(backend, threads);
    cpu_set_use_ref(backend, use_ref);

    struct ggml_init_params ip = { 1024*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K,   N, K);
    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    struct ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // deterministic inputs so the checksum is reproducible across runs
    float * src1_data = gen_rand_f32(M * N, seed);
    float * src0_data = gen_rand_f32(N * K, seed + 1);
    fill_tensor(src1, src1_data, M, N, GGML_TYPE_F32);
    fill_tensor(src0, src0_data, K, N, GGML_TYPE_Q5_K);
    free(src1_data); free(src0_data);

    const double flops = 2.0 * M * N * K;
    printf("perf-tiled-mulmat: path=%s M=%lld N=%lld K=%lld iters=%d threads=%d flush=%s  (%.3f GFLOP per op)\n",
           path_name(path), (long long)M, (long long)N, (long long)K, iters, threads,
           flush ? "on" : "off", flops / 1e9);
    printf("  env: GGML_CPU_TILED_MM=%s GGML_CPU_TILED_MM_FORCE=%s GGML_CPU_MM_PATH=%s use_ref=%d\n",
           tiled_mm, force, mm_path, use_ref ? 1 : 0);
    fflush(stdout);

    // warmup so the profile is steady-state (code/JIT settled, allocator primed)
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf);

    const size_t flush_size = 256 * 1024 * 1024;
    void * flush_buf = flush ? malloc(flush_size) : NULL;

    // time only the compute; the flush memset (if any) runs before the clock so it
    // does not dilute the kernel number. now_s() overhead (~ns) is negligible vs a
    // ~30-70 ms op
    double compute_total = 0.0;
    for (int i = 0; i < iters; ++i) {
        if (flush) memset(flush_buf, 0xAB, flush_size); // evict L3/V-Cache, untimed
        const double a = now_s();
        ggml_backend_graph_compute(backend, gf);
        const double b = now_s();
        compute_total += (b - a);
    }
    if (flush) free(flush_buf);

    const double total = compute_total;
    const double avg   = total / iters;
    printf("total %7.3f s (compute only)   avg %9.4f ms/op   %8.3f TFLOP/s%s\n",
           total, avg * 1e3, flops * iters / total / 1e12,
           flush ? "  [flush: L3/V-Cache evicted each iter]" : "");

    // readback + cheap checksum: three distinct values across std/tiled/iqp prove
    // three distinct code paths actually ran (guards against a routing collapse)
    float * out = (float *) malloc(M * K * sizeof(float));
    ggml_backend_tensor_get(dst, out, 0, ggml_nbytes(dst));
    double csum = 0.0; int ccount = 0; float cmax = 0.0f;
    for (int64_t i = 0; i < M * K; i += 101) { csum += (double) out[i]; cmax = fmaxf(cmax, fabsf(out[i])); ++ccount; }
    printf("checksum: sum[every 101] = %.6f over %d elems   max|dst| = %.4f\n", csum, ccount, cmax);
    fflush(stdout);

    free(out);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}

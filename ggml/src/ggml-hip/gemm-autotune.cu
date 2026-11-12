#include "gemm-autotune.cuh"

#include <rocblas/rocblas.h>
#include <rocblas/internal/rocblas-beta.h>   // rocblas_gemm_ex_get_solutions

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

static const int64_t GGML_HIP_GEMM_AUTOTUNE_MIN_N = 256;
static const float GGML_HIP_GEMM_AUTOTUNE_MIN_GAIN = 1.03f;

struct ggml_hip_gemm_key {
    int     cc;
    int64_t m, n, k;

    bool operator<(const ggml_hip_gemm_key & other) const {
        return std::tie(cc, m, n, k) < std::tie(other.cc, other.m, other.n, other.k);
    }
};

struct ggml_hip_gemm_val {
    int32_t solution;
    float   tflops;
};

static std::map<ggml_hip_gemm_key, ggml_hip_gemm_val> g_ggml_hip_gemm_cache;
static std::mutex g_ggml_hip_gemm_mutex;

static std::string ggml_hip_gemm_cache_path() {
    static const char * const rel = "ggml/hip-gemm.tsv";   // Windows accepts '/' too

#ifdef _WIN32
    const char * base = getenv("LOCALAPPDATA");
    if (base == nullptr || *base == '\0') {
        base = getenv("APPDATA");
    }
    if (base == nullptr || *base == '\0') {
        base = getenv("TEMP");
    }
    return (base != nullptr && *base != '\0') ? std::string(base) + "/" + rel : std::string();
#else
    const char * xdg = getenv("XDG_CACHE_HOME");
    if (xdg != nullptr && *xdg != '\0') {
        return std::string(xdg) + "/" + rel;
    }
    const char * home = getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return std::string(home) + "/.cache/" + rel;
    }
    return std::string("/tmp/") + rel;
#endif
}

static std::string ggml_hip_gemm_cache_tag() {
    char version[128] = {0};
    if (rocblas_get_version_string(version, sizeof(version)) != rocblas_status_success) {
        snprintf(version, sizeof(version), "unknown");
    }
    return std::string("hip-gemm v1 rocblas=") + version;
}

static void ggml_hip_gemm_make_dirs(const std::string & path) {
    std::string dir;
    for (size_t i = 0; i < path.size(); ++i) {
        dir += path[i];
        if (path[i] != '/' && i + 1 != path.size()) {
            continue;
        }
#ifdef _WIN32
        _mkdir(dir.c_str());
#else
        mkdir(dir.c_str(), 0755);
#endif
    }
}

static void ggml_hip_gemm_cache_load() {
    static bool loaded = false;
    if (loaded) {
        return;
    }
    loaded = true;

    const std::string path = ggml_hip_gemm_cache_path();
    if (path.empty()) {
        return;
    }

    FILE * file = ggml_fopen(path.c_str(), "r");
    if (file == nullptr) {
        return;
    }

    char line[512];
    if (fgets(line, sizeof(line), file) == nullptr || std::string("# " + ggml_hip_gemm_cache_tag() + "\n") != line) {
        fclose(file);
        GGML_LOG_DEBUG("%s: %s belongs to a different rocBLAS, ignoring\n", __func__, path.c_str());
        return;
    }

    unsigned long long gcn;
    long long m, n, k;
    int solution;
    float tflops;
    int count = 0;
    while (fscanf(file, "%llx\t%lld\t%lld\t%lld\t%d\t%f\n", &gcn, &m, &n, &k, &solution, &tflops) == 6) {
        g_ggml_hip_gemm_cache[{ (int) (GGML_CUDA_CC_OFFSET_AMD + gcn), m, n, k }] = { solution, tflops };
        ++count;
    }
    fclose(file);
    GGML_LOG_DEBUG("%s: loaded %d entries from %s\n", __func__, count, path.c_str());
}

static void ggml_hip_gemm_cache_store(const ggml_hip_gemm_key & key, const ggml_hip_gemm_val & val) {
    const std::string path = ggml_hip_gemm_cache_path();
    if (path.empty()) {
        return;
    }

    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos) {
        ggml_hip_gemm_make_dirs(path.substr(0, slash));
    }

    FILE * file = ggml_fopen(path.c_str(), "a");
    if (file == nullptr) {
        return;
    }
    const bool fresh = (fseek(file, 0, SEEK_END) == 0) && ftell(file) == 0;
    if (fresh) {
        fprintf(file, "# %s\n", ggml_hip_gemm_cache_tag().c_str());
    }
    fprintf(file, "%llx\t%lld\t%lld\t%lld\t%d\t%.3f\n",
            (unsigned long long) (key.cc - GGML_CUDA_CC_OFFSET_AMD),
            (long long) key.m, (long long) key.n, (long long) key.k, val.solution, val.tflops);
    fclose(file);
}

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
static rocblas_status ggml_hip_gemm_solutions(
        rocblas_handle rb, int64_t m, int64_t n, int64_t k,
        const void * alpha, const void * a, int64_t lda,
        const void * b, int64_t ldb,
        const void * beta, void * d, int64_t ldd,
        rocblas_int * list, rocblas_int * size) {
    return rocblas_gemm_ex_get_solutions(
        rb, rocblas_operation_transpose, rocblas_operation_none,
        m, n, k, alpha,
        a, rocblas_datatype_f16_r, lda,
        b, rocblas_datatype_f16_r, ldb, beta,
        d, rocblas_datatype_f32_r, ldd,
        d, rocblas_datatype_f32_r, ldd,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        rocblas_gemm_flags_none, list, size);
}
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

static ggml_hip_gemm_val ggml_hip_gemm_measure(
        rocblas_handle rb, int64_t m, int64_t n, int64_t k,
        const void * alpha, const void * a, int64_t lda,
        const void * b, int64_t ldb,
        const void * beta, int64_t ldd) {
    const double flops = 2.0 * (double) m * (double) n * (double) k;

    void * scratch = nullptr;
    if (hipMalloc(&scratch, (size_t) ldd * n * sizeof(float)) != hipSuccess) {
        (void) hipGetLastError();
        GGML_LOG_WARN("%s: out of scratch for M=%lld N=%lld K=%lld\n",
                      __func__, (long long) m, (long long) n, (long long) k);
        return { 0, 0.0f };
    }

    auto issue = [&](rocblas_gemm_algo algo, int32_t solution) {
        return rocblas_gemm_ex(
            rb, rocblas_operation_transpose, rocblas_operation_none,
            m, n, k, alpha,
            a, rocblas_datatype_f16_r, lda,
            b, rocblas_datatype_f16_r, ldb, beta,
            scratch, rocblas_datatype_f32_r, ldd,
            scratch, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, algo, solution, rocblas_gemm_flags_none);
    };

    hipStream_t stream = nullptr;
    rocblas_get_stream(rb, &stream);

    auto measure = [&](rocblas_gemm_algo algo, int32_t solution, int iters) -> float {
        hipEvent_t t0 = nullptr;
        hipEvent_t t1 = nullptr;
        float tflops = -1.0f;

        do {
            if (issue(algo, solution) != rocblas_status_success || hipStreamSynchronize(stream) != hipSuccess) {
                break;
            }
            if (hipEventCreate(&t0) != hipSuccess || hipEventCreate(&t1) != hipSuccess) {
                break;
            }

            bool ok = hipEventRecord(t0, stream) == hipSuccess;
            for (int i = 0; i < iters && ok; ++i) {
                ok = issue(algo, solution) == rocblas_status_success;
            }
            if (!ok || hipEventRecord(t1, stream) != hipSuccess || hipEventSynchronize(t1) != hipSuccess) {
                break;
            }

            float ms = 0.0f;
            if (hipEventElapsedTime(&ms, t0, t1) != hipSuccess) {
                break;
            }
            // A near-zero reading is a broken measurement, not a fast solution.
            const double per_iter_ms = (double) ms / iters;
            if (per_iter_ms > 1e-4) {
                tflops = (float) (flops / (per_iter_ms * 1e9));
            }
        } while (false);

        if (t0 != nullptr) {
            (void) hipEventDestroy(t0);
        }
        if (t1 != nullptr) {
            (void) hipEventDestroy(t1);
        }
        (void) hipGetLastError();
        return tflops;
    };

    const float default_tflops = measure(rocblas_gemm_algo_standard, 0, 3);

    rocblas_int count = 0;
    ggml_hip_gemm_solutions(rb, m, n, k, alpha, a, lda, b, ldb, beta, scratch, ldd, nullptr, &count);

    std::vector<rocblas_int> solutions(count > 0 ? count : 0);
    if (count > 0) {
        ggml_hip_gemm_solutions(rb, m, n, k, alpha, a, lda, b, ldb, beta, scratch, ldd,
                                solutions.data(), &count);
    }

    ggml_hip_gemm_val best = { 0, default_tflops };
    int tested = 0;

    std::vector<std::pair<float, rocblas_int>> screened;
    screened.reserve(solutions.size());
    for (const rocblas_int solution : solutions) {
        const float tflops = measure(rocblas_gemm_algo_solution_index, solution, 1);
        if (tflops < 0.0f) {
            continue;
        }
        ++tested;
        screened.emplace_back(tflops, solution);
    }

    const size_t n_finalists = std::min<size_t>(screened.size(), 8);
    std::partial_sort(screened.begin(), screened.begin() + n_finalists, screened.end(),
                      [](const std::pair<float, rocblas_int> & a, const std::pair<float, rocblas_int> & b) {
                          return a.first > b.first;
                      });
    for (size_t i = 0; i < n_finalists; ++i) {
        const float tflops = measure(rocblas_gemm_algo_solution_index, screened[i].second, 3);
        if (tflops > best.tflops) {
            best = { screened[i].second, tflops };
        }
    }

    (void) hipFree(scratch);

    if (best.solution != 0 && default_tflops > 0.0f && best.tflops < default_tflops * GGML_HIP_GEMM_AUTOTUNE_MIN_GAIN) {
        best = { 0, default_tflops };
    }

    return best;
}

bool ggml_hip_gemm_autotune(
        cublasHandle_t handle, int cc,
        int64_t m, int64_t n, int64_t k,
        const void * alpha, const void * a, int64_t lda,
        const void * b, int64_t ldb,
        const void * beta, void * d, int64_t ldd,
        cudaDataType_t type_a, cudaDataType_t type_b, cudaDataType_t type_d,
        cublasComputeType_t compute_type) {
    if (n < GGML_HIP_GEMM_AUTOTUNE_MIN_N ||
        type_a != CUDA_R_16F || type_b != CUDA_R_16F ||
        type_d != CUDA_R_32F || compute_type != CUBLAS_COMPUTE_32F) {
        return false;
    }

    rocblas_handle rb = reinterpret_cast<rocblas_handle>(handle);

    hipStream_t stream = nullptr;
    if (rocblas_get_stream(rb, &stream) != rocblas_status_success) {
        return false;
    }

    hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capturing) != hipSuccess) {
        (void) hipGetLastError();   // legacy stream, nothing to query
    }

    int64_t n_bucket = GGML_HIP_GEMM_AUTOTUNE_MIN_N;
    while (n_bucket * 2 <= n) {
        n_bucket *= 2;
    }
    const ggml_hip_gemm_key key { cc, m, n_bucket, k };

    std::lock_guard<std::mutex> lock(g_ggml_hip_gemm_mutex);
    ggml_hip_gemm_cache_load();

    auto it = g_ggml_hip_gemm_cache.find(key);
    if (it == g_ggml_hip_gemm_cache.end()) {
        if (capturing != hipStreamCaptureStatusNone) {
            return false;
        }
        const ggml_hip_gemm_val val = ggml_hip_gemm_measure(rb, m, n, k, alpha, a, lda, b, ldb, beta, ldd);
        it = g_ggml_hip_gemm_cache.emplace(key, val).first;
        ggml_hip_gemm_cache_store(key, val);
    }

    if (it->second.solution == 0) {
        return false;
    }

    const rocblas_status status = rocblas_gemm_ex(
        rb, rocblas_operation_transpose, rocblas_operation_none,
        m, n, k, alpha,
        a, rocblas_datatype_f16_r, lda,
        b, rocblas_datatype_f16_r, ldb, beta,
        d, rocblas_datatype_f32_r, ldd,
        d, rocblas_datatype_f32_r, ldd,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, it->second.solution,
        rocblas_gemm_flags_none);

    if (status != rocblas_status_success) {
        GGML_LOG_WARN("%s: solution %d failed for M=%lld N=%lld K=%lld (%s), "
                      "reverting this shape to the rocBLAS default\n",
                      __func__, it->second.solution, (long long) m, (long long) n,
                      (long long) k, rocblas_status_to_string(status));
        it->second.solution = 0;
        return false;
    }
    return true;
}

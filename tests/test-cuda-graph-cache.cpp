// Regression test for the CUDA graph cache key partitioning.
//
// Two execution variants of one MUL_MAT_ID graph share the same first-node pointer but differ in the first node's extents (micro-batch ne[2] 1 <-> 4).  Under the bare first-node-pointer key both variants land in one cache entry, and the periodically stable workload below emits one "CUDA graph warmup reset" per pair switch.  With the key partitioned by first-node extents the same workload replays without any reset.
//
// The test counts the backend's own diagnostics through ggml_log_set().  Op-level numeric correctness is covered by test-backend-ops.  Skips when no CUDA device is registered.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <malloc.h> // using malloc.h with MSC/MINGW
#endif
#include <map>
#include <utility>
#include <vector>

static constexpr int N_EMBD = 256;
static constexpr int N_EXP = 8;
static constexpr int N_LAYER = 8;
static constexpr int WARM = 8;
static constexpr int ALTERNATE = 100;
static constexpr size_t META_SIZE = 32ull << 20;

static int g_failures = 0;
static std::atomic<long> g_resets{0};
static std::atomic<long> g_completes{0};

static void log_cb(enum ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "CUDA graph warmup reset")) {
        g_resets.fetch_add(1);
    } else if (strstr(text, "CUDA graph warmup complete")) {
        g_completes.fetch_add(1);
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "ggml: %s", text);
    }
}

// portable 64-byte aligned allocation (MSVC has no aligned_alloc)
static void * meta_alloc(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, 64);
#else
    return aligned_alloc(64, size);
#endif
}

static void meta_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            printf("  FAIL: %s (line %d): ", #cond, __LINE__); \
            printf(__VA_ARGS__);                               \
            printf("\n");                                      \
            g_failures++;                                      \
        }                                                      \
    } while (0)

// one gallocr per shape keeps the graph plan stable across calls.  All
// externally supplied tensors are flagged as inputs (ggml_set_input), so the
// allocator reserves their storage before the main allocation pass and no
// parameter can be aliased away by another leaf of the same call.  Every
// parameter is still re-uploaded on every call: the harness never assumes
// that leaf content persists across a compute.
struct harness {
    ggml_backend_t backend = nullptr;
    std::map<int, ggml_gallocr_t> galloc;
    void * meta = nullptr;
    std::vector<std::vector<uint8_t>> wq;

    bool init(ggml_backend_dev_t dev) {
        const size_t n = (size_t) N_EMBD * N_EMBD * N_EXP;
        std::vector<float> f(n);
        wq.resize(N_LAYER);
        for (int l = 0; l < N_LAYER; ++l) {
            for (size_t i = 0; i < n; ++i) {
                f[i] = (float)((i * 40503u + (uint32_t)(l * 2246822519u + 7)) % 4000u) / 2000.0f - 1.0f;
            }
            wq[l].resize(ggml_row_size(GGML_TYPE_Q8_0, N_EMBD) * N_EMBD * N_EXP);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), wq[l].data(), 0, n / N_EMBD, N_EMBD, nullptr);
        }
        backend = ggml_backend_dev_init(dev, nullptr);
        meta = meta_alloc(META_SIZE);
        return backend != nullptr && meta != nullptr;
    }

    void fill_f32(ggml_tensor * t, uint32_t seed) {
        const size_t n = ggml_nbytes(t) / sizeof(float);
        std::vector<float> buf(n);
        for (size_t i = 0; i < n; ++i) {
            buf[i] = (float)((i * 2654435761u + seed) % 2000u) / 1000.0f - 1.0f;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, ggml_nbytes(t));
    }

    void run_call(int bs, const void ** first_node, int64_t * first_node_ne) {
        ggml_init_params ip = {};
        ip.mem_size   = META_SIZE;
        ip.mem_buffer = meta;
        ip.no_alloc   = true;
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, bs);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, bs);
        ggml_set_input(x);
        ggml_set_input(ids);
        std::vector<std::pair<ggml_tensor *, int>> params;
        ggml_tensor * cur = x;
        for (int l = 0; l < N_LAYER; ++l) {
            ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, N_EMBD, N_EMBD, N_EXP);
            ggml_set_input(w);
            cur = ggml_mul_mat_id(ctx, w, cur, ids);
            params.emplace_back(w, 1000 + l);
            ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_EMBD);
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_EMBD);
            ggml_set_input(s);
            ggml_set_input(b);
            cur = ggml_mul(ctx, cur, s);
            cur = ggml_add(ctx, cur, b);
            params.emplace_back(s, 2000 + l);
            params.emplace_back(b, 3000 + l);
        }
        ggml_cgraph * cgraph = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(cgraph, cur);

        if (!galloc.count(bs)) {
            galloc[bs] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        }
        if (!ggml_gallocr_reserve(galloc[bs], cgraph) || !ggml_gallocr_alloc_graph(galloc[bs], cgraph)) {
            fprintf(stderr, "galloc failed\n");
            exit(1);
        }

        // re-upload every input on every call
        fill_f32(x, 12345);
        for (auto & [t, code] : params) {
            if (t->type == GGML_TYPE_Q8_0) {
                ggml_backend_tensor_set(t, wq[code - 1000].data(), 0, ggml_nbytes(t));
            } else {
                fill_f32(t, (uint32_t)(code * 2654435761u + 11));
            }
        }
        {
            std::vector<int32_t> idbuf(bs);
            for (int i = 0; i < bs; ++i) {
                idbuf[i] = (int)((i * 3 + 1) % N_EXP);
            }
            ggml_backend_tensor_set(ids, idbuf.data(), 0, ggml_nbytes(ids));
        }

        if (ggml_backend_graph_compute(backend, cgraph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            exit(1);
        }
        if (first_node) {
            // the first operation node of the graph is the coarse cache key
            ggml_tensor * n0 = ggml_graph_node(cgraph, 0);
            *first_node = n0;
            if (first_node_ne) {
                memcpy(first_node_ne, n0->ne, sizeof(int64_t) * GGML_MAX_DIMS);
            }
        }
        ggml_free(ctx);
    }

    void cleanup() {
        for (auto & kv : galloc) {
            ggml_gallocr_free(kv.second);
        }
        galloc.clear();
        if (backend) {
            ggml_backend_free(backend);
        }
        meta_free(meta);
    }
};

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    ggml_time_init();
    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();

    // find an explicitly CUDA-registered device; do not assume that the first GPU backend is CUDA
    ggml_backend_dev_t dev = nullptr;
    for (size_t r = 0; r < ggml_backend_reg_count() && !dev; ++r) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(r);
        if (strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) {
            continue;
        }
        if (ggml_backend_reg_dev_count(reg) > 0) {
            dev = ggml_backend_reg_dev_get(reg, 0);
        }
    }
    if (!dev) {
        printf("test-cuda-graph-cache: SKIP (no CUDA device registered)\n");
        return 0;
    }

    harness R;
    if (!R.init(dev)) {
        fprintf(stderr, "harness init failed\n");
        return 1;
    }

    const void * key_s1 = nullptr;
    const void * key_s4 = nullptr;
    int64_t ne_s1[GGML_MAX_DIMS] = {0};
    int64_t ne_s4[GGML_MAX_DIMS] = {0};

    R.run_call(1, nullptr, nullptr); // absorbs one-time lazy init

    // warm each variant on its own: each must demonstrate on its own that it converges to replay, and each warm phase must produce at least one warmup-complete diagnostic (positive control for the log surface)
    printf("warming S1 and S4 (%d calls each)\n", WARM);
    for (int i = 0; i < WARM; ++i) {
        R.run_call(1, &key_s1, ne_s1);
    }
    const long completes_s1 = g_completes.load();
    for (int i = 0; i < WARM; ++i) {
        R.run_call(4, &key_s4, ne_s4);
    }
    const long completes_s4 = g_completes.load();
    printf("warmup complete events: %ld after S1, %ld after S4\n", completes_s1, completes_s4);

    // preconditions for this test to be meaningful: the two variants really do collide on the old (pointer-only) key, and the new discriminator really does separate them
    CHECK(key_s1 != nullptr, "no first-node pointer captured for S1");
    CHECK(key_s4 != nullptr, "no first-node pointer captured for S4");
    CHECK(key_s1 == key_s4, "S1 and S4 first-node pointers differ (%p vs %p): the variants no longer share a cache key and this test cannot reproduce the bug", key_s1, key_s4);
    CHECK(memcmp(ne_s1, ne_s4, sizeof(ne_s1)) != 0, "S1 and S4 first-node extents are identical: the partitioned key would not separate the variants");
    CHECK(ne_s1[2] == 1 && ne_s4[2] == 4, "expected micro-batch ne[2] 1 <-> 4 (got %lld vs %lld)", (long long) ne_s1[2], (long long) ne_s4[2]);

    if (completes_s1 < 1 || completes_s4 < completes_s1 + 1) {
        printf("test-cuda-graph-cache: SKIP (missing warmup-complete events per variant; cannot judge reset behavior)\n");
        R.cleanup();
        return g_failures == 0 ? 0 : 1;
    }

    // regression window: pairs give each variant enough consecutive calls to complete warmup on its own, so under the unpartitioned key every pair switch resets the shared entry - while partitioned entries replay the whole window without a single reset
    const long resets0 = g_resets.load();
    printf("alternating S1,S1,S4,S4 x%d\n", ALTERNATE);
    for (int i = 0; i < ALTERNATE; ++i) {
        R.run_call(1, nullptr, nullptr);
        R.run_call(1, nullptr, nullptr);
        R.run_call(4, nullptr, nullptr);
        R.run_call(4, nullptr, nullptr);
    }
    const long resets = g_resets.load() - resets0;
    printf("warmup resets inside the window: %ld (over %d pair switches)\n", resets, ALTERNATE);
    CHECK(resets == 0, "alternating variants sharing one cache key must not reset the shared warmup state (resets=%ld)", resets);

    R.cleanup();
    if (g_failures == 0) {
        printf("test-cuda-graph-cache: PASS\n");
    } else {
        printf("test-cuda-graph-cache: FAIL\n");
    }
    return g_failures == 0 ? 0 : 1;
}

#include "common.cuh"

#include <cstdio>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #condition); \
        return false; \
    } \
} while (0)

static bool test_reuse() {
    ggml_backend_cuda_context ctx(0);
    const int key = 0;

    ggml_cuda_graph * graph = ctx.cuda_graph(&key, 1'000'000);
    CHECK(!graph->warmup_complete);
    graph->warmup_complete = true;
    graph->uid = 42;

    graph = ctx.cuda_graph(&key, 2'000'000);
    CHECK(graph->warmup_complete);
    CHECK(graph->uid == 42);
    CHECK(graph->last_used_time == 2'000'000);
    CHECK(ctx.cuda_graphs.size() == 1);
    return true;
}

static bool test_idle_resume() {
    ggml_backend_cuda_context ctx(0);
    const int keys[] = { 0, 1, 2 };

    for (int i = 0; i < 3; ++i) {
        ggml_cuda_graph * graph = ctx.cuda_graph(&keys[i], 1'000'000 + i);
        graph->warmup_complete = true;
        graph->uid = i + 1;
    }

    ctx.cuda_graph(&keys[0], 60'000'000);
    CHECK(ctx.cuda_graphs.size() == 3);
    CHECK(ctx.cuda_graphs.at(&keys[1])->last_used_time == 1'000'001);
    CHECK(ctx.cuda_graphs.at(&keys[2])->last_used_time == 1'000'002);

    // A graph compute can look up the same graph several times before the next split.
    ctx.cuda_graph(&keys[0], 60'000'001);
    ctx.cuda_graph(&keys[0], 60'000'002);
    ctx.cuda_graph(&keys[0], 65'000'000);
    CHECK(ctx.cuda_graphs.size() == 3);
    for (int i = 0; i < 3; ++i) {
        ggml_cuda_graph * graph = ctx.cuda_graph(&keys[i], 69'000'000 + i);
        CHECK(graph->warmup_complete);
        CHECK(graph->uid == (uint64_t) i + 1);
    }
    ctx.cuda_graph(&keys[0], 70'000'000);
    CHECK(ctx.cuda_graphs.size() == 3);
    return true;
}

static bool test_requested_graph() {
    ggml_backend_cuda_context ctx(0);
    const int keys[] = { 0, 1, 2 };

    ctx.cuda_graph(&keys[0], 1'000'000)->uid = 42;
    ctx.cuda_graph(&keys[1], 1'000'000);
    ctx.cuda_graph(&keys[2], 1'000'000);
    ctx.cuda_graph(&keys[1], 9'000'000);
    ctx.cuda_graph(&keys[1], 12'000'000);
    CHECK(ctx.cuda_graphs.size() == 3);

    ggml_cuda_graph * graph = ctx.cuda_graph(&keys[0], 14'000'000);
    CHECK(graph->uid == 42);
    CHECK(graph->last_used_time == 14'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);
    CHECK(ctx.cuda_graphs.count(&keys[1]) == 1);
    CHECK(ctx.cuda_graphs.count(&keys[2]) == 0);
    return true;
}

static bool test_eviction_after_resume() {
    ggml_backend_cuda_context ctx(0);
    const int keys[] = { 0, 1 };

    ctx.cuda_graph(&keys[0], 1'000'000);
    ctx.cuda_graph(&keys[1], 1'000'000);
    ctx.cuda_graph(&keys[0], 6'000'000);
    ctx.cuda_graph(&keys[0], 60'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);

    ctx.cuda_graph(&keys[0], 64'999'999);
    CHECK(ctx.cuda_graphs.size() == 2);
    ctx.cuda_graph(&keys[0], 65'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);
    ctx.cuda_graph(&keys[0], 69'999'999);
    CHECK(ctx.cuda_graphs.size() == 2);
    ctx.cuda_graph(&keys[0], 70'000'000);
    CHECK(ctx.cuda_graphs.size() == 1);
    CHECK(ctx.cuda_graphs.count(&keys[1]) == 0);
    return true;
}

static bool test_eviction_boundary() {
    ggml_backend_cuda_context ctx(0);
    const int keys[] = { 0, 1, 2 };

    ctx.cuda_graph(&keys[0], 1'000'000);
    ctx.cuda_graph(&keys[1], 1'000'001);
    ctx.cuda_graph(&keys[2], 6'000'000);
    ctx.cuda_graph(&keys[2], 10'999'999);
    CHECK(ctx.cuda_graphs.size() == 3);

    ctx.cuda_graph(&keys[2], 11'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);
    CHECK(ctx.cuda_graphs.count(&keys[0]) == 0);
    CHECK(ctx.cuda_graphs.count(&keys[1]) == 1);
    ctx.cuda_graph(&keys[2], 16'000'000);
    CHECK(ctx.cuda_graphs.size() == 1);
    return true;
}

static bool test_idle_preserves_stale_graphs() {
    ggml_backend_cuda_context ctx(0);
    const int keys[] = { 0, 1 };

    ctx.cuda_graph(&keys[0], 1'000'000)->uid = 42;
    ctx.cuda_graph(&keys[1], 9'000'000);
    ctx.cuda_graph(&keys[1], 12'000'000);

    // The first graph was already stale before the idle period began.
    ctx.cuda_graph(&keys[1], 22'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);
    CHECK(ctx.cuda_graphs.at(&keys[0])->last_used_time == 1'000'000);
    ctx.cuda_graph(&keys[1], 27'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);
    CHECK(ctx.cuda_graph(&keys[0], 31'999'999)->uid == 42);
    ctx.cuda_graph(&keys[1], 32'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);
    return true;
}

static bool test_idle_with_new_graph() {
    ggml_backend_cuda_context ctx(0);
    const int keys[] = { 0, 1, 2 };

    ctx.cuda_graph(&keys[0], 1'000'000)->uid = 1;
    ctx.cuda_graph(&keys[1], 2'000'000)->uid = 2;
    ctx.cuda_graph(&keys[2], 60'000'000);
    CHECK(ctx.cuda_graphs.size() == 3);
    ctx.cuda_graph(&keys[2], 65'000'000);
    CHECK(ctx.cuda_graphs.size() == 3);
    CHECK(ctx.cuda_graph(&keys[0], 69'000'000)->uid == 1);
    CHECK(ctx.cuda_graph(&keys[1], 69'000'001)->uid == 2);
    return true;
}

static bool test_repeated_idle() {
    ggml_backend_cuda_context ctx(0);
    const int keys[] = { 0, 1 };

    ctx.cuda_graph(&keys[0], 1'000'000)->uid = 42;
    ctx.cuda_graph(&keys[1], 2'000'000);
    ctx.cuda_graph(&keys[1], 12'000'000);
    ctx.cuda_graph(&keys[1], 22'000'000);
    ctx.cuda_graph(&keys[1], 32'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);
    CHECK(ctx.cuda_graphs.at(&keys[0])->uid == 42);

    ctx.cuda_graph(&keys[1], 37'000'000);
    CHECK(ctx.cuda_graphs.size() == 2);
    ctx.cuda_graph(&keys[1], 42'000'000);
    CHECK(ctx.cuda_graphs.size() == 1);
    CHECK(ctx.cuda_graphs.count(&keys[0]) == 0);
    return true;
}

static bool test_below_idle_threshold() {
    ggml_backend_cuda_context ctx(0);
    const int keys[] = { 0, 1 };

    ctx.cuda_graph(&keys[0], 1'000'000);
    ctx.cuda_graph(&keys[1], 9'000'000);
    ctx.cuda_graph(&keys[1], 18'999'999);
    CHECK(ctx.cuda_graphs.size() == 1);
    CHECK(ctx.cuda_graphs.count(&keys[0]) == 0);
    return true;
}

static bool test_cache_invalidation() {
    ggml_backend_cuda_context ctx(0);
    const int key = 0;

    ctx.cuda_graph(&key, 1'000'000)->uid = 42;
    ctx.cuda_graph(&key, 60'000'000);
    ctx.cuda_graphs.clear();
    ggml_cuda_graph * graph = ctx.cuda_graph(&key, 60'000'001);
    CHECK(graph->uid == 0);
    CHECK(!graph->warmup_complete);
    CHECK(ctx.cuda_graphs.size() == 1);
    return true;
}

int main() {
    const struct {
        const char * name;
        bool (*run)();
    } tests[] = {
        { "reuse",                       test_reuse                      },
        { "idle resume",                 test_idle_resume                },
        { "requested graph",             test_requested_graph            },
        { "eviction after resume",       test_eviction_after_resume      },
        { "eviction boundary",           test_eviction_boundary          },
        { "idle preserves stale graphs", test_idle_preserves_stale_graphs },
        { "idle with new graph",         test_idle_with_new_graph        },
        { "repeated idle",               test_repeated_idle              },
        { "below idle threshold",        test_below_idle_threshold       },
        { "cache invalidation",          test_cache_invalidation         },
    };

    bool success = true;
    for (const auto & test : tests) {
        const bool passed = test.run();
        printf("%s: %s\n", test.name, passed ? "PASS" : "FAIL");
        success &= passed;
    }
    return success ? 0 : 1;
}

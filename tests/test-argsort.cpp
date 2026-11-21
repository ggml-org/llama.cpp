#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

// mode: 0 = ggml_argsort, 1 = ggml_argsort_top_k, 2 = ggml_top_k
static bool test_argsort(int layout, ggml_sort_order order, int n_threads, int mode) {
    ggml_init_params params {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr{ggml_init(params)};
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * base = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 3, 2);
    ggml_tensor * src = base;
    if (layout == 1) {
        src = ggml_view_4d(ctx, base, 4, 2, 2, 2, base->nb[1], base->nb[2], base->nb[3], 0);
    } else if (layout == 2) {
        src = ggml_transpose(ctx, base);
    }
    ggml_tensor * out = mode == 0 ? ggml_argsort(ctx, src, order) :
                        mode == 1 ? ggml_argsort_top_k(ctx, src, 2) :
                                    ggml_top_k(ctx, src, 2);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_ptr backend_ptr{ggml_backend_cpu_init()};
    ggml_backend_t backend = backend_ptr.get();
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_backend_buffer_ptr buffer{ggml_backend_alloc_ctx_tensors(ctx, backend)};
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_supports_op(backend, out->view_src ? out->view_src : out));

    std::vector<float> values(ggml_nelements(base));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float((i * 13) % 97);
    }
    ggml_backend_tensor_set(base, values.data(), 0, ggml_nbytes(base));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    bool passed = true;
    for (int64_t i3 = 0; i3 < src->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < src->ne[1]; ++i1) {
                std::vector<std::pair<float, int32_t>> expected;
                for (int64_t i0 = 0; i0 < src->ne[0]; ++i0) {
                    expected.emplace_back(ggml_get_f32_nd(src, i0, i1, i2, i3), i0);
                }
                if (order == GGML_SORT_ORDER_ASC && mode == 0) {
                    std::sort(expected.begin(), expected.end());
                } else {
                    std::sort(expected.rbegin(), expected.rend());
                }
                std::vector<int32_t> actual(out->ne[0]);
                std::vector<int32_t> wanted(out->ne[0]);
                for (int64_t i0 = 0; i0 < out->ne[0]; ++i0) {
                    actual[i0] = ggml_get_i32_nd(out, i0, i1, i2, i3);
                    wanted[i0] = expected[i0].second;
                }
                if (mode == 2) {
                    // GGML_OP_TOP_K does not define the order of the returned indices
                    std::sort(actual.begin(), actual.end());
                    std::sort(wanted.begin(), wanted.end());
                }
                for (int64_t i0 = 0; i0 < out->ne[0]; ++i0) {
                    if (actual[i0] != wanted[i0]) {
                        if (passed) {
                            printf("layout=%d order=%d threads=%d mode=%d at [%lld,%lld,%lld,%lld]: got %d, expected %d\n",
                                   layout, order, n_threads, mode,
                                   (long long) i0, (long long) i1, (long long) i2, (long long) i3,
                                   actual[i0], wanted[i0]);
                        }
                        passed = false;
                    }
                }
            }
        }
    }
    return passed;
}

int main() {
    bool passed = true;
    for (int layout = 0; layout < 3; ++layout) {
        for (int n_threads : {1, 3}) {
            passed &= test_argsort(layout, GGML_SORT_ORDER_ASC, n_threads, 0);
            passed &= test_argsort(layout, GGML_SORT_ORDER_DESC, n_threads, 0);
            passed &= test_argsort(layout, GGML_SORT_ORDER_DESC, n_threads, 1);
            passed &= test_argsort(layout, GGML_SORT_ORDER_DESC, n_threads, 2);
        }
    }
    printf("argsort strides: %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}

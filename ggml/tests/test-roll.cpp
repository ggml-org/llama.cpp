#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <array>
#include <numeric>
#include <vector>

int64_t wrap(int64_t i, int64_t ne) {
    if (i < 0) {
        return i + ne;
    } else if (i >= ne) {
        return i - ne;
    }
    return i;
}

std::vector<float> roll_reference(
    const float * src, std::array<int64_t, 4> ne, std::array<int, 4> shift) {

    const int64_t ne0 = ne[0], ne1 = ne[1], ne2 = ne[2], ne3 = ne[3];
    std::vector<float> dst(ne0 * ne1 * ne2 * ne3);

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const int64_t i03 = wrap(i3 - shift[3], ne3);
                    const int64_t i02 = wrap(i2 - shift[2], ne2);
                    const int64_t i01 = wrap(i1 - shift[1], ne1);
                    const int64_t i00 = wrap(i0 - shift[0], ne0);

                    dst[i3 * (ne2*ne1*ne0) + i2 * (ne1*ne0) + i1 * ne0 + i0] =
                        src[i03 * (ne2*ne1*ne0) + i02 * (ne1*ne0) + i01 * ne0 + i00];
                }
            }
        }
    }
    return dst;
}

std::vector<float> f32_range(int64_t n) {
    std::vector<float> values(n);
    std::iota(values.begin(), values.end(), 0.f);
    return values;
}

bool check_equal(const std::vector<float> & result, const std::vector<float> & expected) {
    if (result.size() != expected.size()) {
        printf("result.size() = %d, expected.size() = %d\n", (int)result.size(), (int)expected.size());
        return false;
    }
    for (int i = 0; i < result.size(); i++) {
        if(std::abs(result[i] - expected[i]) > 1e-5) {
            printf("result[%d] %f != %f expected[%d]\n", i, result[i], expected[i], i);
            return false;
        }
    }
    return true;
}

bool test_roll(std::array<int64_t, 4> ne, std::array<int, 4> shift, bool permute) {
    ggml_time_init();

    ggml_init_params params {
        /*.mem_size   =*/ 64 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true
    };

    ggml_context_ptr ctx_ptr{ggml_init(params)};
    ggml_context * ctx = ctx_ptr.get();
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Build graph
    ggml_tensor * src = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
    ggml_tensor * res;
    if (!permute) {
        res = ggml_roll(ctx, src, shift[0], shift[1], shift[2], shift[3]);
    } else {
        ggml_tensor * p = ggml_permute(ctx, src, 0, 3, 1, 2);
        res = ggml_roll(ctx, p, shift[0], shift[2], shift[3], shift[1]);
        res = ggml_cont(ctx, ggml_permute(ctx, res, 0, 2, 3, 1));
    }
    ggml_build_forward_expand(gf, res);

    // Create backend & allocate buffers
    ggml_backend_ptr backend_ptr{ggml_backend_cpu_init()};
    ggml_backend_t backend = backend_ptr.get();
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_backend_buffer_ptr buffer{ggml_backend_alloc_ctx_tensors(ctx, backend)};

    std::vector<float> src_values = f32_range(ggml_nelements(src));
    ggml_backend_tensor_set(src, src_values.data(), 0, ggml_nbytes(src));

    // Execute and compare results
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> res_values(ggml_nelements(res));
    ggml_backend_tensor_get(res, res_values.data(), 0, ggml_nbytes(res));

    std::vector<float> expected = roll_reference(src_values.data(), ne, shift);

    bool passed = check_equal(res_values, expected);

    printf("ggml_roll(%d(%d), %d(%d), %d(%d), %d(%d), %s): %s\n",
        int(ne[0]), int(shift[0]),
        int(ne[1]), int(shift[1]),
        int(ne[2]), int(shift[2]),
        int(ne[3]), int(shift[3]),
        permute ? "permuted" : "contiguous",
        passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    return passed;
}

int main() {
    bool passed = true;
    passed &= test_roll({3, 7, 4, 2}, {1, 0, -1, 0}, false);
    passed &= test_roll({37, 42, 59, 2}, {-4, 3, -7, 1}, false);
    passed &= test_roll({37, 42, 59, 2}, {-4, 3, -7, 1}, true);
    return passed ? 0 : 1;
}
// Benchmark: MMVQ Q8_0 streaming DMA vs CPU on pinned host memory.
// Run (opt-in):
//   GGML_SYCL_MMVQ_BENCH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
//   ./build/bin/test-mmvq-q8-0-streaming-bench
//
// Env knobs:
//   GGML_SYCL_MMVQ_BENCH_NROWS    (default: 8192)
//   GGML_SYCL_MMVQ_BENCH_NCOLS    (default: 4096)
//   GGML_SYCL_MMVQ_BENCH_ITERS    (default: 5)
//   GGML_SYCL_MMVQ_BENCH_WARMUP   (default: 1)
//   GGML_SYCL_MMVQ_BENCH_SLICE_MB (default: 1024)
//   GGML_SYCL_MMVQ_BENCH_BUFFERS  (default: 2)
//   GGML_SYCL_MMVQ_BENCH_MODE     (manual|cache|both|mmq, default: manual)
//   GGML_SYCL_MMVQ_BENCH_MMQ_NROWS   (default: 512)
//   GGML_SYCL_MMVQ_BENCH_MMQ_NCOLS   (default: 2048)
//   GGML_SYCL_MMVQ_BENCH_MMQ_NTOKENS (default: 4)
//   GGML_SYCL_MMVQ_BENCH_MMQ_ITERS   (default: 1)
//   GGML_SYCL_MMVQ_BENCH_MMQ_WARMUP  (default: 0)
//   GGML_SYCL_MMVQ_BENCH_MMQ_BUDGET_MB (default: 1)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml-sycl/ggml-sycl-test.hpp"
#define GGML_COMMON_DECL_SYCL
#include "ggml-common.h"
#include "ggml-sycl/dpct/helper.hpp"

#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
#define WARP_SIZE GGML_SYCL_WARP_SIZE

static constexpr int MMVQ_NROWS_PER_WG = 4;
static constexpr int MMVQ_SLM_Y_QS_STRIDE = 9;
static constexpr int VDR_Q8_0_Q8_1_MMVQ = 2;

static __dpct_inline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0];
    x32 |= (int)x16[1] << 16;
    return x32;
}

static __dpct_inline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *)(x8 + sizeof(int) * i32));
}

template <int vdr>
static __dpct_inline__ float vec_dot_q8_0_q8_1_impl(const int * v, const int * u,
                                                    const float & d8_0, const float & d8_1) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }
    return d8_0 * d8_1 * sumi;
}

static __dpct_inline__ float
vec_dot_q8_0_q8_1_slm(const void * __restrict__ vbq,
                      const int * __restrict__ slm_y_qs,
                      const sycl::half2 * __restrict__ slm_y_ds,
                      const int y_block_idx, const int slm_stride, const int & iqs) {
    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
    }

    const int y_slm_offset = y_block_idx * slm_stride;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        u[i] = slm_y_qs[y_slm_offset + iqs + i];
    }

    const sycl::half2 ds8 = slm_y_ds[y_block_idx];
    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, ds8[0]);
}

static int parse_env_int(const char * name, int def) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    long val = std::strtol(env, &end, 10);
    if (end == env) {
        return def;
    }
    return static_cast<int>(val);
}

static const char * parse_env_str(const char * name, const char * def) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return def;
    }
    return env;
}

static size_t parse_env_mb(const char * name, size_t def_mb) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return def_mb;
    }
    char * end = nullptr;
    long val = std::strtol(env, &end, 10);
    if (end == env || val < 0) {
        return def_mb;
    }
    return static_cast<size_t>(val);
}

static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i];
        const double db = b[i];
        const double diff = da - db;
        mse_a_b += diff * diff;
        mse_a_0 += da * da;
    }
    return mse_a_b / (mse_a_0 > 0.0 ? mse_a_0 : 1.0);
}

static float max_diff(const float * a, const float * b, size_t n) {
    float max_d = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > max_d) {
            max_d = d;
        }
    }
    return max_d;
}

static void fill_random_q8_0(block_q8_0 * data, size_t blocks, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.2f);
    std::uniform_int_distribution<int> quant_dist(-127, 127);
    for (size_t i = 0; i < blocks; ++i) {
        const float d = scale_dist(rng);
        data[i].d = static_cast<ggml_half>(d);
        for (int j = 0; j < QK8_0; ++j) {
            data[i].qs[j] = static_cast<int8_t>(quant_dist(rng));
        }
    }
}

static void fill_random_q8_1(block_q8_1 * data, size_t blocks, std::mt19937 & rng) {
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.2f);
    std::uniform_int_distribution<int> quant_dist(-127, 127);
    for (size_t i = 0; i < blocks; ++i) {
        const float d = scale_dist(rng);
        int sum = 0;
        for (int j = 0; j < QK8_1; ++j) {
            const int q = quant_dist(rng);
            data[i].qs[j] = static_cast<int8_t>(q);
            sum += q;
        }
        data[i].data.d = static_cast<ggml_half>(d);
        data[i].data.s = static_cast<ggml_half>(d * static_cast<float>(sum));
    }
}

static void fill_random_f32(std::vector<float> & data, float scale, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float & v : data) {
        v = dist(rng);
    }
}

static float vec_dot_q8_0_q8_1_ref(const block_q8_0 * x, const block_q8_1 * y, int nb) {
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d0 = static_cast<float>(x[i].d);
        const float d1 = static_cast<float>(y[i].data.d);
        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += static_cast<int>(x[i].qs[j]) * static_cast<int>(y[i].qs[j]);
        }
        sum += d0 * d1 * static_cast<float>(sumi);
    }
    return sum;
}

static void cpu_mmvq_q8_0(const block_q8_0 * weights, const block_q8_1 * input,
                          float * output, int nrows, int ncols) {
    const int blocks_per_row = ncols / QK8_0;
    for (int row = 0; row < nrows; ++row) {
        const block_q8_0 * row_blocks = weights + static_cast<size_t>(row) * blocks_per_row;
        output[row] = vec_dot_q8_0_q8_1_ref(row_blocks, input, blocks_per_row);
    }
}

static void launch_mmvq_q8_0(const void * vx,
                             const void * vy,
                             float * dst,
                             int ncols,
                             int nrows,
                             sycl::queue & q);

struct mmvq_stream_ctx {
    const block_q8_1 * d_input = nullptr;
    float *            d_output = nullptr;
    int                ncols = 0;
    int                nrows = 0;
    size_t             bytes_per_row = 0;
};

static sycl::event mmvq_stream_slice(sycl::queue &                    queue,
                                     void *                           device_slice,
                                     size_t                           slice_bytes,
                                     size_t                           offset_bytes,
                                     const void *                     ctx_void,
                                     const std::vector<sycl::event> & deps) {
    const auto * ctx = static_cast<const mmvq_stream_ctx *>(ctx_void);
    if (!ctx || ctx->bytes_per_row == 0 || slice_bytes == 0) {
        return queue.ext_oneapi_submit_barrier();
    }
    const size_t row_start = offset_bytes / ctx->bytes_per_row;
    size_t       row_count = slice_bytes / ctx->bytes_per_row;
    if (row_start + row_count > static_cast<size_t>(ctx->nrows)) {
        row_count = static_cast<size_t>(ctx->nrows) - row_start;
    }
    if (row_count == 0) {
        return queue.ext_oneapi_submit_barrier();
    }
    if (!deps.empty() && !queue.has_property<sycl::property::queue::in_order>()) {
        sycl::event::wait(deps);
    }
    launch_mmvq_q8_0(device_slice,
                     ctx->d_input,
                     ctx->d_output + row_start,
                     ctx->ncols,
                     static_cast<int>(row_count),
                     queue);
    return queue.ext_oneapi_submit_barrier();
}

// Multi-row MMVQ kernel (copied from production path)
template <int qk,
          int qi,
          typename block_q_t,
          int vdr,
          float (*vec_dot_q_slm)(const void *, const int *, const sycl::half2 *, int, int, const int &),
          int nrows_per_wg>
static void mul_mat_vec_q_multirow(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int ncols,
                                   const int nrows,
                                   const sycl::nd_item<3> & item_ct1,
                                   int * __restrict__ slm_y_qs,
                                   sycl::half2 * __restrict__ slm_y_ds) {
    const int local_row = item_ct1.get_local_id(1);
    const int lane_id   = item_ct1.get_local_id(2);
    const int wg_idx    = item_ct1.get_group(2);
    const int row       = wg_idx * nrows_per_wg + local_row;

    const int blocks_per_row = ncols / qk;

    if (local_row == 0) {
        const block_q8_1 * y = (const block_q8_1 *) vy;
        for (int blk = lane_id; blk < blocks_per_row; blk += WARP_SIZE) {
            const int slm_offset = blk * MMVQ_SLM_Y_QS_STRIDE;
#pragma unroll
            for (int j = 0; j < QI8_1; ++j) {
                slm_y_qs[slm_offset + j] = get_int_from_int8_aligned(y[blk].qs, j);
            }
            slm_y_ds[blk] = *((const sycl::half2 *) &y[blk].ds);
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (row >= nrows) {
        return;
    }

    const block_q_t * x = (const block_q_t *) vx;

    constexpr int qi_div_vdr      = qi / vdr;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
    const int     base_iqs        = vdr * (lane_id % qi_div_vdr);
    const int     row_offset      = row * blocks_per_row;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int       i       = lane_id / qi_div_vdr;
    const int stride  = blocks_per_warp;
    const int stride4 = 4 * stride;

    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_slm(&x[ibx0], slm_y_qs, slm_y_ds, iby0, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc1 += vec_dot_q_slm(&x[ibx1], slm_y_qs, slm_y_ds, iby1, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc2 += vec_dot_q_slm(&x[ibx2], slm_y_qs, slm_y_ds, iby2, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc3 += vec_dot_q_slm(&x[ibx3], slm_y_qs, slm_y_ds, iby3, MMVQ_SLM_Y_QS_STRIDE, iqs);
        }
    }

    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);
#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_slm(&x[ibx], slm_y_qs, slm_y_ds, iby, MMVQ_SLM_Y_QS_STRIDE, iqs);
        }
    }

    float tmp = (acc0 + acc1) + (acc2 + acc3);
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

class mmvq_q8_0_bench_kernel;

static void launch_mmvq_q8_0(const void * vx,
                             const void * vy,
                             float * dst,
                             int ncols,
                             int nrows,
                             sycl::queue & q) {
    const int blocks_per_row = ncols / QK8_0;
    const int block_num_z = (nrows + MMVQ_NROWS_PER_WG - 1) / MMVQ_NROWS_PER_WG;
    const sycl::range<3> block_nums(1, 1, block_num_z);
    const sycl::range<3> block_dims(1, MMVQ_NROWS_PER_WG, WARP_SIZE);

    const int slm_y_qs_size = blocks_per_row * MMVQ_SLM_Y_QS_STRIDE;
    const int slm_y_ds_size = blocks_per_row + 1;

    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int, 1> slm_y_qs(slm_y_qs_size, cgh);
        sycl::local_accessor<sycl::half2, 1> slm_y_ds(slm_y_ds_size, cgh);

        cgh.parallel_for<mmvq_q8_0_bench_kernel>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_multirow<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                                       vec_dot_q8_0_q8_1_slm, MMVQ_NROWS_PER_WG>(
                    vx, vy, dst, ncols, nrows, item_ct1, slm_y_qs.get_pointer(), slm_y_ds.get_pointer());
            });
    });
}

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft,
                                                 ggml_tensor *               tensor,
                                                 ggml_backend_buffer_usage   usage) {
    const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static bool run_mul_mat_backend(ggml_backend_t                 backend,
                                ggml_backend_buffer_type_t     weight_buft,
                                ggml_type                      weight_type,
                                const void *                   weight_data,
                                size_t                         weight_bytes,
                                const std::vector<float> &     input_data,
                                int                            in_dim,
                                int                            out_dim,
                                int                            n_tokens,
                                std::vector<float> &           output) {
    const ggml_init_params params = { 32 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, weight_type, in_dim, out_dim);
    ggml_set_name(weights, "mmq_weights");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_tokens);
    ggml_set_name(input, "mmq_input");
    ggml_tensor * out = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(out, "mmq_out");

    ggml_backend_buffer_t weight_buf =
        alloc_tensor_buffer(weight_buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!weight_buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev && weight_buft == ggml_backend_sycl_host_buffer_type()) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weights);
    }

    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t input_buf = alloc_tensor_buffer(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t out_buf   = alloc_tensor_buffer(dev_buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (!input_buf || !out_buf) {
        ggml_backend_buffer_free(weight_buf);
        if (input_buf) ggml_backend_buffer_free(input_buf);
        if (out_buf) ggml_backend_buffer_free(out_buf);
        ggml_free(ctx);
        return false;
    }

    const size_t expected_weight_bytes = ggml_nbytes(weights);
    if (weight_bytes < expected_weight_bytes) {
        fprintf(stderr, "FAIL: weight buffer too small (got=%zu need=%zu)\n",
                weight_bytes, expected_weight_bytes);
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(out_buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(weights, weight_data, 0, expected_weight_bytes);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(out_buf);
        ggml_free(ctx);
        return false;
    }

    output.resize(static_cast<size_t>(out_dim) * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_buf);
    ggml_backend_buffer_free(out_buf);
    ggml_free(ctx);
    return true;
}

static int run_mmq_bench() {
    const int nrows   = parse_env_int("GGML_SYCL_MMVQ_BENCH_MMQ_NROWS", 512);
    const int ncols   = parse_env_int("GGML_SYCL_MMVQ_BENCH_MMQ_NCOLS", 2048);
    const int n_tokens = parse_env_int("GGML_SYCL_MMVQ_BENCH_MMQ_NTOKENS", 4);
    const int iters   = std::max(1, parse_env_int("GGML_SYCL_MMVQ_BENCH_MMQ_ITERS", 1));
    const int warmup  = std::max(0, parse_env_int("GGML_SYCL_MMVQ_BENCH_MMQ_WARMUP", 0));
    const size_t budget_mb = parse_env_mb("GGML_SYCL_MMVQ_BENCH_MMQ_BUDGET_MB", 1);

    if (ncols % QK8_0 != 0) {
        std::fprintf(stderr, "FAIL: mmq ncols must be divisible by %d\n", QK8_0);
        return 1;
    }

    bool owned_override = false;
    ggml_layout_mode override_layout = GGML_LAYOUT_AOS;
    if (!ggml_sycl::test_get_layout_override(&override_layout)) {
        ggml_sycl::test_set_layout_override(GGML_LAYOUT_AOS);
        owned_override = true;
    }
    struct OverrideGuard {
        bool owned = false;
        ~OverrideGuard() {
            if (owned) {
                ggml_sycl::test_clear_layout_override();
            }
        }
    } override_guard{ owned_override };

    setenv("GGML_SYCL_FORCE_MMQ", "1", 1);
    setenv("GGML_SYCL_UNIFIED_CACHE", "1", 1);
    if (std::getenv("GGML_SYCL_DMA_SLICE_MB") == nullptr) {
        setenv("GGML_SYCL_DMA_SLICE_MB", "2", 1);
    }
    if (std::getenv("GGML_SYCL_DMA_BUFFERS") == nullptr && std::getenv("GGML_SYCL_DMA_SLICES") == nullptr) {
        setenv("GGML_SYCL_DMA_BUFFERS", "1", 1);
    }

    ggml_sycl::set_unified_cache_budget(budget_mb * 1024ULL * 1024ULL);

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        std::fprintf(stderr, "SKIP: Could not initialize SYCL backend\n");
        return 0;
    }

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        std::fprintf(stderr, "FAIL: Could not initialize CPU backend\n");
        ggml_backend_free(sycl_backend);
        return 1;
    }

    std::mt19937 rng(1234);
    std::vector<float> input_f32(static_cast<size_t>(ncols) * n_tokens);
    fill_random_f32(input_f32, 0.5f, rng);

    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, ncols);
    const size_t weight_bytes = row_size * static_cast<size_t>(nrows);
    std::vector<uint8_t> weight_q8(weight_bytes);
    const size_t total_blocks = weight_bytes / sizeof(block_q8_0);
    fill_random_q8_0(reinterpret_cast<block_q8_0 *>(weight_q8.data()), total_blocks, rng);

    std::vector<float> cpu_out;
    std::vector<float> sycl_out;

    for (int i = 0; i < warmup; ++i) {
        run_mul_mat_backend(cpu_backend,
                            ggml_backend_get_default_buffer_type(cpu_backend),
                            GGML_TYPE_Q8_0,
                            weight_q8.data(),
                            weight_q8.size(),
                            input_f32,
                            ncols,
                            nrows,
                            n_tokens,
                            cpu_out);
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!run_mul_mat_backend(cpu_backend,
                                 ggml_backend_get_default_buffer_type(cpu_backend),
                                 GGML_TYPE_Q8_0,
                                 weight_q8.data(),
                                 weight_q8.size(),
                                 input_f32,
                                 ncols,
                                 nrows,
                                 n_tokens,
                                 cpu_out)) {
            std::fprintf(stderr, "FAIL: CPU MMQ backend compute failed\n");
            ggml_backend_free(cpu_backend);
            ggml_backend_free(sycl_backend);
            return 1;
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();

    auto gpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!run_mul_mat_backend(sycl_backend,
                                 ggml_backend_sycl_host_buffer_type(),
                                 GGML_TYPE_Q8_0,
                                 weight_q8.data(),
                                 weight_q8.size(),
                                 input_f32,
                                 ncols,
                                 nrows,
                                 n_tokens,
                                 sycl_out)) {
            std::fprintf(stderr, "FAIL: SYCL MMQ backend compute failed\n");
            ggml_backend_free(cpu_backend);
            ggml_backend_free(sycl_backend);
            return 1;
        }
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();

    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count() / iters;
    const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count() / iters;

    const double err = nmse(cpu_out.data(), sycl_out.data(), cpu_out.size());
    const float  max_d = max_diff(cpu_out.data(), sycl_out.data(), cpu_out.size());

    std::printf("MMQ Q8_0 streaming benchmark (CPU vs SYCL)\n");
    std::printf("  nrows=%d, ncols=%d, ntokens=%d\n", nrows, ncols, n_tokens);
    std::printf("  iters=%d, warmup=%d, cache_budget=%.1f MB\n", iters, warmup, budget_mb * 1.0);
    std::printf("CPU: %.3f ms/iter\n", cpu_ms);
    std::printf("GPU: %.3f ms/iter\n", gpu_ms);
    std::printf("NMSE: %.6e, max_diff: %.6f\n", err, max_d);

    const double nmse_tol = 2e-4;
    if (!std::isfinite(err) || err > nmse_tol) {
        std::fprintf(stderr, "FAIL: MMQ mismatch beyond tolerance (nmse>%.1e)\n", nmse_tol);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(sycl_backend);
        return 1;
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(sycl_backend);
    return 0;
}

int main() {
    const char * enable = std::getenv("GGML_SYCL_MMVQ_BENCH");
    if (!enable || std::atoi(enable) == 0) {
        std::printf("SKIP: set GGML_SYCL_MMVQ_BENCH=1 to run\n");
        return 0;
    }

    const char * mode = parse_env_str("GGML_SYCL_MMVQ_BENCH_MODE", "manual");
    if (std::strcmp(mode, "mmq") == 0) {
        return run_mmq_bench();
    }

    const bool use_manual = (std::strcmp(mode, "manual") == 0 || std::strcmp(mode, "both") == 0);
    const bool use_cache  = (std::strcmp(mode, "cache") == 0 || std::strcmp(mode, "both") == 0);
    if (!use_manual && !use_cache) {
        std::fprintf(stderr, "FAIL: unknown GGML_SYCL_MMVQ_BENCH_MODE='%s'\n", mode);
        return 1;
    }

    const int nrows   = parse_env_int("GGML_SYCL_MMVQ_BENCH_NROWS", 8192);
    const int ncols   = parse_env_int("GGML_SYCL_MMVQ_BENCH_NCOLS", 4096);
    const int iters   = std::max(1, parse_env_int("GGML_SYCL_MMVQ_BENCH_ITERS", 5));
    const int warmup  = std::max(0, parse_env_int("GGML_SYCL_MMVQ_BENCH_WARMUP", 1));
    const size_t slice_mb = parse_env_mb("GGML_SYCL_MMVQ_BENCH_SLICE_MB", 1024);
    const size_t buffer_count = static_cast<size_t>(std::max(1, parse_env_int("GGML_SYCL_MMVQ_BENCH_BUFFERS", 2)));

    if (ncols % QK8_0 != 0) {
        std::fprintf(stderr, "FAIL: ncols must be divisible by %d\n", QK8_0);
        return 1;
    }

    const int blocks_per_row = ncols / QK8_0;
    const size_t total_blocks = static_cast<size_t>(nrows) * blocks_per_row;
    const size_t weight_bytes = total_blocks * sizeof(block_q8_0);
    const size_t input_bytes  = static_cast<size_t>(blocks_per_row) * sizeof(block_q8_1);
    const size_t output_bytes = static_cast<size_t>(nrows) * sizeof(float);

    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});

    block_q8_0 * weights = static_cast<block_q8_0 *>(sycl::malloc_host(weight_bytes, q));
    block_q8_1 * input   = static_cast<block_q8_1 *>(sycl::malloc_host(input_bytes, q));

    if (!weights || !input) {
        std::fprintf(stderr, "FAIL: sycl::malloc_host failed\n");
        if (weights) sycl::free(weights, q);
        if (input) sycl::free(input, q);
        return 1;
    }

    std::mt19937 rng(1234);
    fill_random_q8_0(weights, total_blocks, rng);
    fill_random_q8_1(input, static_cast<size_t>(blocks_per_row), rng);

    std::vector<float> cpu_out(nrows, 0.0f);
    std::vector<float> gpu_out(nrows, 0.0f);

    for (int i = 0; i < warmup; ++i) {
        cpu_mmvq_q8_0(weights, input, cpu_out.data(), nrows, ncols);
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cpu_mmvq_q8_0(weights, input, cpu_out.data(), nrows, ncols);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();

    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    const size_t slice_bytes = slice_mb * 1024ULL * 1024ULL;
    const size_t bytes_per_row = static_cast<size_t>(blocks_per_row) * sizeof(block_q8_0);
    size_t slice_rows = std::max<size_t>(1, slice_bytes / bytes_per_row);
    if (slice_rows > static_cast<size_t>(nrows)) {
        slice_rows = static_cast<size_t>(nrows);
    }
    const size_t slice_bytes_aligned = slice_rows * bytes_per_row;

    const double cpu_ms_per = cpu_ms / iters;
    const double total_bytes = static_cast<double>(weight_bytes + output_bytes);
    const double cpu_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (cpu_ms_per / 1000.0);

    std::printf("MMVQ Q8_0 streaming benchmark (CPU vs GPU DMA)\n");
    std::printf("  nrows=%d, ncols=%d, blocks_per_row=%d\n", nrows, ncols, blocks_per_row);
    std::printf("  weights=%.2f MB, slice_rows=%zu (slice=%.1f MB)\n",
                weight_bytes / (1024.0 * 1024.0), slice_rows, slice_mb * 1.0);
    std::printf("  iters=%d, warmup=%d, buffers=%zu, mode=%s\n", iters, warmup, buffer_count, mode);
    std::printf("CPU: %.3f ms/iter, %.2f GB/s (pinned host)\n", cpu_ms_per, cpu_gbps);

    if (use_manual) {
        block_q8_1 * d_input = static_cast<block_q8_1 *>(sycl::malloc_device(input_bytes, q));
        float * d_output = static_cast<float *>(sycl::malloc_device(output_bytes, q));
        const size_t staging_blocks = slice_rows * static_cast<size_t>(blocks_per_row);
        block_q8_0 * d_weights = static_cast<block_q8_0 *>(sycl::malloc_device(staging_blocks * sizeof(block_q8_0), q));

        if (!d_input || !d_output || !d_weights) {
            std::fprintf(stderr, "FAIL: sycl::malloc_device failed (manual)\n");
            if (d_input) sycl::free(d_input, q);
            if (d_output) sycl::free(d_output, q);
            if (d_weights) sycl::free(d_weights, q);
            sycl::free(weights, q);
            sycl::free(input, q);
            return 1;
        }

        q.memcpy(d_input, input, input_bytes).wait();

        for (int i = 0; i < warmup; ++i) {
            for (size_t row = 0; row < static_cast<size_t>(nrows); row += slice_rows) {
                const size_t rows = std::min(slice_rows, static_cast<size_t>(nrows) - row);
                const size_t blocks = rows * static_cast<size_t>(blocks_per_row);
                const block_q8_0 * src = weights + row * static_cast<size_t>(blocks_per_row);
                q.memcpy(d_weights, src, blocks * sizeof(block_q8_0));
                launch_mmvq_q8_0(d_weights, d_input, d_output + row, ncols, static_cast<int>(rows), q);
            }
            q.memcpy(gpu_out.data(), d_output, output_bytes);
            q.wait();
        }

        auto gpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            for (size_t row = 0; row < static_cast<size_t>(nrows); row += slice_rows) {
                const size_t rows = std::min(slice_rows, static_cast<size_t>(nrows) - row);
                const size_t blocks = rows * static_cast<size_t>(blocks_per_row);
                const block_q8_0 * src = weights + row * static_cast<size_t>(blocks_per_row);
                q.memcpy(d_weights, src, blocks * sizeof(block_q8_0));
                launch_mmvq_q8_0(d_weights, d_input, d_output + row, ncols, static_cast<int>(rows), q);
            }
            q.memcpy(gpu_out.data(), d_output, output_bytes);
            q.wait();
        }
        auto gpu_end = std::chrono::high_resolution_clock::now();

        const double gpu_ms_per = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count() / iters;
        const double gpu_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (gpu_ms_per / 1000.0);

        float max_abs = 0.0f;
        for (int i = 0; i < nrows; ++i) {
            const float diff = std::fabs(gpu_out[i] - cpu_out[i]);
            max_abs = std::max(max_abs, diff);
        }
        const double err = nmse(cpu_out.data(), gpu_out.data(), cpu_out.size());

        std::printf("GPU(manual): %.3f ms/iter, %.2f GB/s (DMA memcpy)\n", gpu_ms_per, gpu_gbps);
        std::printf("GPU(manual) max abs diff: %.6e\n", max_abs);
        std::printf("GPU(manual) NMSE: %.6e\n", err);

        const double nmse_tol = 2e-4;
        const float max_abs_tol = 5e-2f;
        if (!std::isfinite(err) || err > nmse_tol || max_abs > max_abs_tol) {
            std::fprintf(stderr, "FAIL: GPU(manual) mismatch (nmse>%.1e or max_abs>%.2e)\n",
                         nmse_tol, max_abs_tol);
            sycl::free(d_input, q);
            sycl::free(d_output, q);
            sycl::free(d_weights, q);
            sycl::free(weights, q);
            sycl::free(input, q);
            return 1;
        }

        sycl::free(d_input, q);
        sycl::free(d_output, q);
        sycl::free(d_weights, q);
    }

    if (use_cache) {
        block_q8_1 * d_input = static_cast<block_q8_1 *>(sycl::malloc_device(input_bytes, q));
        float * d_output = static_cast<float *>(sycl::malloc_device(output_bytes, q));
        if (!d_input || !d_output) {
            std::fprintf(stderr, "FAIL: sycl::malloc_device failed (cache)\n");
            if (d_input) sycl::free(d_input, q);
            if (d_output) sycl::free(d_output, q);
            sycl::free(weights, q);
            sycl::free(input, q);
            return 1;
        }

        q.memcpy(d_input, input, input_bytes).wait();

        ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache(q);
        if (!cache) {
            std::fprintf(stderr, "FAIL: unified cache unavailable\n");
            sycl::free(d_input, q);
            sycl::free(d_output, q);
            sycl::free(weights, q);
            sycl::free(input, q);
            return 1;
        }

        auto infer_location = [&](const void * ptr) -> ggml_sycl::cache_location {
            if (!ptr) {
                return ggml_sycl::cache_location::HOST_MMAP;
            }
            const sycl::usm::alloc alloc = sycl::get_pointer_type(ptr, q.get_context());
            if (alloc == sycl::usm::alloc::device) {
                return ggml_sycl::cache_location::DEVICE;
            }
            if (alloc == sycl::usm::alloc::host || alloc == sycl::usm::alloc::shared) {
                return ggml_sycl::cache_location::HOST_PINNED;
            }
            return ggml_sycl::cache_location::HOST_MMAP;
        };

        ggml_sycl::cache_ptr_view view{};
        view.ptr      = const_cast<block_q8_0 *>(weights);
        view.size     = weight_bytes;
        view.layout   = GGML_LAYOUT_AOS;
        view.type     = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
        view.location = infer_location(view.ptr);
        if (view.location == ggml_sycl::cache_location::DEVICE) {
            std::fprintf(stderr, "FAIL: expected host-backed weights for streaming, got DEVICE\n");
            sycl::free(d_input, q);
            sycl::free(d_output, q);
            sycl::free(weights, q);
            sycl::free(input, q);
            return 1;
        }

        mmvq_stream_ctx stream_ctx{};
        stream_ctx.d_input       = d_input;
        stream_ctx.d_output      = d_output;
        stream_ctx.ncols         = ncols;
        stream_ctx.nrows         = nrows;
        stream_ctx.bytes_per_row = bytes_per_row;

        for (int i = 0; i < warmup; ++i) {
            auto result = cache->stream_dma(view,
                                            weight_bytes,
                                            slice_bytes_aligned,
                                            buffer_count,
                                            mmvq_stream_slice,
                                            &stream_ctx,
                                            {});
            if (!result.ok) {
                std::fprintf(stderr, "FAIL: unified cache streaming failed (warmup)\n");
                sycl::free(d_input, q);
                sycl::free(d_output, q);
                sycl::free(weights, q);
                sycl::free(input, q);
                return 1;
            }
            result.event.wait();
            q.memcpy(gpu_out.data(), d_output, output_bytes).wait();
        }

        auto gpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            auto result = cache->stream_dma(view,
                                            weight_bytes,
                                            slice_bytes_aligned,
                                            buffer_count,
                                            mmvq_stream_slice,
                                            &stream_ctx,
                                            {});
            if (!result.ok) {
                std::fprintf(stderr, "FAIL: unified cache streaming failed\n");
                sycl::free(d_input, q);
                sycl::free(d_output, q);
                sycl::free(weights, q);
                sycl::free(input, q);
                return 1;
            }
            result.event.wait();
            q.memcpy(gpu_out.data(), d_output, output_bytes).wait();
        }
        auto gpu_end = std::chrono::high_resolution_clock::now();

        const double gpu_ms_per = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count() / iters;
        const double gpu_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (gpu_ms_per / 1000.0);

        float max_abs = 0.0f;
        for (int i = 0; i < nrows; ++i) {
            const float diff = std::fabs(gpu_out[i] - cpu_out[i]);
            max_abs = std::max(max_abs, diff);
        }
        const double err = nmse(cpu_out.data(), gpu_out.data(), cpu_out.size());

        std::printf("GPU(cache): %.3f ms/iter, %.2f GB/s (unified cache)\n", gpu_ms_per, gpu_gbps);
        std::printf("GPU(cache) max abs diff: %.6e\n", max_abs);
        std::printf("GPU(cache) NMSE: %.6e\n", err);

        const double nmse_tol = 2e-4;
        const float max_abs_tol = 5e-2f;
        if (!std::isfinite(err) || err > nmse_tol || max_abs > max_abs_tol) {
            std::fprintf(stderr, "FAIL: GPU(cache) mismatch (nmse>%.1e or max_abs>%.2e)\n",
                         nmse_tol, max_abs_tol);
            sycl::free(d_input, q);
            sycl::free(d_output, q);
            sycl::free(weights, q);
            sycl::free(input, q);
            return 1;
        }

        sycl::free(d_input, q);
        sycl::free(d_output, q);
    }

    sycl::free(weights, q);
    sycl::free(input, q);

    return 0;
}

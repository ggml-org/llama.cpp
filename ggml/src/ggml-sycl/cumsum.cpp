#include "cumsum.hpp"
#include "common.hpp"

#include <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

static __dpct_inline__ float warp_prefix_inclusive_sum_f32(float x, const sycl::nd_item<3> & item) {
    return sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<float>());
}

static void cumsum_f32_kernel(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t  d1, const int64_t  d2, const int64_t  d3,
        const sycl::nd_item<3> & item, float * smem) {

    const int tid = item.get_local_id(2);
    const int block_size = item.get_local_range(2);
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int warps_per_block = block_size / WARP_SIZE;

    float * s_vals      = smem;
    float * s_warp_sums = smem + block_size;
    float * s_carry     = smem + block_size + warps_per_block;

    if (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t i3 = item.get_group(0);
    const int64_t i2 = item.get_group(1);
    const int64_t i1 = item.get_group(2);
    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const float * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    float       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr int num_unroll = 4;
    float temp[num_unroll];

    for (int64_t i = 0; i < ne00; i += num_unroll * block_size) {
        int64_t idx = i + tid * num_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        for (int j = 1; j < num_unroll; j++) {
            temp[j] = temp[j - 1];
            if (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        float val = (idx < ne00) ? temp[num_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        if (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        if (warp == 0) {
            float w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            float inc = warp_prefix_inclusive_sum_f32(w, item);
            if (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            if (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        float carry = s_carry[0];
        float final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[num_unroll - 1];

#pragma unroll
        for (int j = 0; j < num_unroll; j++) {
            if (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

void ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src_d = static_cast<const float *>(src0->data);
    float       * dst_d = static_cast<float *>(dst->data);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t ts = sizeof(float);
    const int64_t s01 = src0->nb[1] / ts;
    const int64_t s02 = src0->nb[2] / ts;
    const int64_t s03 = src0->nb[3] / ts;
    const int64_t d1  = dst->nb[1] / ts;
    const int64_t d2  = dst->nb[2] / ts;
    const int64_t d3  = dst->nb[3] / ts;

    const int num_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    int block_size = num_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    const int warps_per_block = block_size / WARP_SIZE;
    const int smem_size = block_size + warps_per_block + 2;

    const sycl::range<3> grid(ne03, ne02, ne01);
    const sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

#include "rwkv.hpp"

#include <sycl/sycl.hpp>

static void rwkv_lerp_f32_kernel(
        const float * x_prev,
        const float * cur,
        const float * weight,
        float * dst,
        const int64_t ne0,
        const int64_t base_total,
        const int64_t total,
        sycl::id<1> id) {
    const int64_t i = id[0];
    if (i >= total) {
        return;
    }

    const int64_t ibase = i % base_total;
    const int64_t imix  = i / base_total;
    const int64_t iembd = ibase % ne0;

    const float c = cur[ibase];
    dst[i] = c + (x_prev[ibase] - c) * weight[imix * ne0 + iembd];
}

template <int head_size>
static void rwkv_rk_f32_kernel(
        const float * cur,
        const float * k,
        const float * r,
        const float * v,
        const float * r_k,
        float * dst,
        const int64_t C,
        const int64_t H,
        sycl::nd_item<3> item,
        float * shared_mem) {
    const int64_t tid = item.get_local_id(2);
    const int64_t row = item.get_group(2);
    const int64_t h   = row % H;
    const int64_t t   = row / H;
    const int64_t off = t * C + h * head_size;

    shared_mem[tid] = k[off + tid] * r[off + tid] * r_k[h * head_size + tid];
    item.barrier(sycl::access::fence_space::local_space);

    for (int stride = head_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    dst[off + tid] = cur[off + tid] + v[off + tid] * shared_mem[0];
}

void ggml_sycl_op_rwkv_lerp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * x_prev = dst->src[0];
    const ggml_tensor * cur    = dst->src[1];
    const ggml_tensor * weight = dst->src[2];

    GGML_ASSERT(x_prev->type == GGML_TYPE_F32);
    GGML_ASSERT(cur->type    == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(x_prev, cur));
    GGML_ASSERT(ggml_is_contiguous(x_prev));
    GGML_ASSERT(ggml_is_contiguous(cur));
    GGML_ASSERT(ggml_is_contiguous(weight));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t base_total = ggml_nelements(x_prev);
    const int64_t total      = ggml_nelements(dst);
    const int64_t n_mix      = dst->ne[3];
    const int64_t ne0        = dst->ne[0];

    GGML_ASSERT(dst->ne[0] == x_prev->ne[0]);
    GGML_ASSERT(dst->ne[1] == x_prev->ne[1]);
    GGML_ASSERT(dst->ne[2] == x_prev->ne[2]);
    GGML_ASSERT(total == base_total * n_mix);
    GGML_ASSERT(weight->ne[0] == dst->ne[0]);
    GGML_ASSERT(weight->ne[1] == 1);
    GGML_ASSERT(weight->ne[2] == 1);
    GGML_ASSERT(weight->ne[3] == n_mix);
    GGML_ASSERT(ggml_nelements(weight) == dst->ne[0] * n_mix);

    const float * x_prev_d = (const float *) x_prev->data;
    const float * cur_d    = (const float *) cur->data;
    const float * weight_d = (const float *) weight->data;
    float * dst_d = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        rwkv_lerp_f32_kernel(x_prev_d, cur_d, weight_d, dst_d, ne0, base_total, total, id);
    });
}

void ggml_sycl_op_rwkv_rk(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * r   = dst->src[2];
    const ggml_tensor * v   = dst->src[3];
    const ggml_tensor * r_k = dst->src[4];

    GGML_ASSERT(cur->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type   == GGML_TYPE_F32);
    GGML_ASSERT(r->type   == GGML_TYPE_F32);
    GGML_ASSERT(v->type   == GGML_TYPE_F32);
    GGML_ASSERT(r_k->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(cur));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r_k));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(k, r));
    GGML_ASSERT(ggml_are_same_shape(k, v));
    GGML_ASSERT(ggml_are_same_shape(cur, dst));

    const int64_t head_size = k->ne[0];
    const int64_t H         = k->ne[1];
    const int64_t T         = k->ne[2];
    const int64_t C         = head_size * H;

    GGML_ASSERT(head_size == 64 || head_size == 128);
    GGML_ASSERT(r_k->ne[0] == head_size);
    GGML_ASSERT(r_k->ne[1] == H);
    GGML_ASSERT(ggml_nelements(r_k) == head_size * H);
    GGML_ASSERT(dst->ne[0] == C);
    GGML_ASSERT(dst->ne[1] == T);
    GGML_ASSERT(ggml_nelements(dst) == C * T);

    const float * cur_d = (const float *) cur->data;
    const float * k_d   = (const float *) k->data;
    const float * r_d   = (const float *) r->data;
    const float * v_d   = (const float *) v->data;
    const float * r_k_d = (const float *) r_k->data;
    float * dst_d = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    sycl::range<3> block_dims(1, 1, head_size);
    sycl::range<3> grid_dims(1, 1, H * T);

    if (head_size == 64) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> shared_mem_acc(head_size, cgh);
            cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims), [=](sycl::nd_item<3> item) {
                rwkv_rk_f32_kernel<64>(cur_d, k_d, r_d, v_d, r_k_d, dst_d, C, H, item,
                        (float *) shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
    } else {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> shared_mem_acc(head_size, cgh);
            cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims), [=](sycl::nd_item<3> item) {
                rwkv_rk_f32_kernel<128>(cur_d, k_d, r_d, v_d, r_k_d, dst_d, C, H, item,
                        (float *) shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
    }
}

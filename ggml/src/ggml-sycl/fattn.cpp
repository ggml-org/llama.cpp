#include "./fattn.hpp"
#include "./fattn_kernel.hpp"
#include "./fattn_common.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <sycl/sycl.hpp>

#define Br 32
#define Bc 32


bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (Q == nullptr || K == nullptr || V == nullptr) {
        return false;
    }
    if (Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 && V->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        return true;
    }
    // if (Q->type == GGML_TYPE_F16 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
    //     return true;
    // }

    return false;
}

template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];

    GGML_ASSERT(Q != nullptr);
    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);
    GGML_ASSERT(dst != nullptr);
    
    //not support KV_Cache yet
    GGML_ASSERT(K->ne[1] == V->ne[1]);

    //not support multi head and gqa yet 
    GGML_ASSERT(Q->ne[2] == 1);
    GGML_ASSERT(K->ne[2] == 1);
    GGML_ASSERT(V->ne[2] == 1);

    const float * Q_d   = (const float *) Q->data;
    const float * K_d   = (const float *) K->data;
    const float * V_d   = (const float *) V->data;
    float *       dst_d = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N = Q->ne[1];

    const ptrdiff_t q_row_stride = Q->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t k_row_stride = K->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t v_row_stride = V->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t o_row_stride = dst->nb[1] / (ptrdiff_t)sizeof(float);

    // const int Br = std::min((int) FLASH_ATTN_BR_MAX, (int) N);
    // const int Bc = std::min((int) FLASH_ATTN_BC_MAX, (int) N);

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    float * l_d = (float *) sycl::malloc_device(N * sizeof(float), *stream);
    float * m_d = (float *) sycl::malloc_device(N * sizeof(float), *stream);

    sycl::range<2> global(Br * Tr, Tc);
    sycl::range<2> local(Br,1);

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 2> Qtile({Br, DQK}, cgh);
        sycl::local_accessor<float, 2> Ktile({Bc, DQK}, cgh);
        sycl::local_accessor<float, 2> Vtile({Bc, DV}, cgh);
        sycl::local_accessor<float, 2> Stile({Br, Bc}, cgh);
        sycl::local_accessor<float, 1> Ptile({Br * Bc}, cgh);
        sycl::local_accessor<float, 1> m_local({Br}, cgh);
        sycl::local_accessor<float, 1> l_local({Br}, cgh);

        float* q_loc = Qtile.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* k_loc = Ktile.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* v_loc = Vtile.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* s_loc = Stile.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* p_loc = Ptile.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* m_loc = m_local.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* l_loc = l_local.template get_multi_ptr<sycl::access::decorated::no>().get();

        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            auto group = it.get_group();
            int group_id_i = group.get_group_id(0);
            int group_id_j = group.get_group_id(1);


            int row0 = group_id_i * Br;
            int col0 = group_id_j * Bc;

            if (row0 >= (int) N || col0 >= (int) N) {
                return;
            }

            const float* Q_block = Q_d + (ptrdiff_t)row0 * q_row_stride;
            const float* K_block = K_d + (ptrdiff_t)col0 * k_row_stride;
            const float* V_block = V_d + (ptrdiff_t)col0 * v_row_stride;
            float*       O_block = dst_d + (ptrdiff_t)row0 * o_row_stride;

            //this lines does not support non-contiguous tensors
            ggml_sycl_memcpy<Br * DQK>(q_loc, Q_block);
            ggml_sycl_memcpy<Bc * DQK>(k_loc, K_block);
            ggml_sycl_memcpy<Bc * DV>(v_loc, V_block);

            it.barrier(sycl::access::fence_space::local_space);

            flash_attn_mul_mat_QK_kernel<DQK>(
                it,
                Q_block, q_row_stride,
                K_block, k_row_stride,
                s_loc, (ptrdiff_t)Bc,
                Br, Bc
            );

            it.barrier(sycl::access::fence_space::local_space);

            flash_attn_softmax_kernel(
                it,
                s_loc, p_loc,
                m_loc, l_loc,
                Br, Bc,
                l_d, m_d
            );

            it.barrier(sycl::access::fence_space::local_space);

            flash_attn_mul_mat_PV_kernel<DV>(
                it,
                p_loc, (ptrdiff_t)Bc,
                V_block, v_row_stride,
                O_block, o_row_stride,
                Br,Bc
            );

            it.barrier(sycl::access::fence_space::local_space);
        });
    });

    
    stream->submit([&](sycl::handler& cgh) {
        const ptrdiff_t o_stride = o_row_stride;

        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id_row) {
            int row = id_row[0];
            float l_val = l_d[row];

            if (l_val <= 0.0f) {
                return;
            }

            float inv_l = 1.0f / l_val;
            float * o_row = dst_d + (ptrdiff_t)row * o_stride;

            for (int col = 0; col < DV; ++col) {
                o_row[col] *= inv_l;
            }
        });
    });


    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);
}


void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * V    = dst->src[2];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_sycl_op_flash_attn_2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_sycl_op_flash_attn_2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_sycl_op_flash_attn_2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_sycl_op_flash_attn_2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_sycl_op_flash_attn_2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_sycl_op_flash_attn_2<256, 256>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}


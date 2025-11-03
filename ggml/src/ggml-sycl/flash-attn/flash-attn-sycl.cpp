#include "flash-attn-sycl.h"

#include "kernels/flash-attn-kernel.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <sycl/sycl.hpp>

#define FLASH_ATTN_BR_MAX 32
#define FLASH_ATTN_BC_MAX 32

// Flash Attention: https://arxiv.org/abs/2205.14135
void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q != nullptr);
    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);
    GGML_ASSERT(dst != nullptr);

    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F32 || V->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        fprintf(stderr, "[SYCL] FLASH-ATTENTION: tensor type not supported (Q=%d, K=%d, V=%d, dst=%d)\n", Q->type, K->type, V->type, dst->type);
        return;
    }

    const float * Q_d   = (const float *) Q->data;
    const float * K_d   = (const float *) K->data;
    const float * V_d   = (const float *) V->data;
    float *       dst_d = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    const int64_t d = Q->ne[0];
    const int64_t N = Q->ne[1];

    float scale;
    float max_bias;
    float logit_softcap;
    
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const bool masked = (mask != nullptr);

    const int Br = std::min((int) FLASH_ATTN_BR_MAX, (int) N);
    const int Bc = std::min((int) FLASH_ATTN_BC_MAX, (int) N);

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    float * l_d = (float *) sycl::malloc_device(N * sizeof(float), *stream);
    float * m_d = (float *) sycl::malloc_device(N * sizeof(float), *stream);

    stream->fill(l_d, 0.0f, N);
    stream->fill(m_d, -std::numeric_limits<float>::infinity(), N);
    stream->fill(dst_d, 0.0f, N * d);
    stream->wait();

    for (int j = 0; j < Tc; ++j) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(Tr), [=](sycl::id<1> idx) {
                const int i = idx[0];
                flash_attn_tiled_kernel<FLASH_ATTN_BR_MAX, FLASH_ATTN_BC_MAX>(Q_d, K_d, V_d, dst_d, l_d, m_d, i, j, Br,
                                                                              Bc, N, d, masked, scale);
            });
        });
    }

    stream->wait();

    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);
}

bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (Q == nullptr || K == nullptr || V == nullptr) {
        return false;
    }

    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F32 || V->type != GGML_TYPE_F32) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    return true;
}

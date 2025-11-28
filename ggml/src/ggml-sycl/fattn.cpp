//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fattn.hpp"
#include "fattn-vec.hpp"
#include "fattn-mma.hpp"
#include "fattn-mma-f16.hpp"
#include "fattn-xmx-f16.hpp"

// Kernel selection is now done at runtime based on GPU capabilities.
// XMX kernel (3) is used on Intel GPUs with matrix extension support (Arc, etc.)
// MMA F16 kernel (2) is used on other SYCL devices.
// Kernel IDs:
// 0 = VEC kernel (simpler, one K/V position at a time)
// 1 = MMA kernel (tiled scalar, processes BATCH_KV positions at a time)
// 2 = MMA F16 kernel (scalar with SG_SIZE=16, named MMA but not using joint_matrix)
// 3 = XMX F16 kernel (using joint_matrix for Q@K^T acceleration)

// Check if flash attention is supported for the given operation
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // Check Q type - must be F32 or F16
    if (Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) {
        return false;
    }

    // Check K/V types - currently only F16 is supported
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }

    // Check destination type
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Check mask type if present
    if (mask && mask->type != GGML_TYPE_F16) {
        return false;
    }

    // Check head dimension - must be a supported size
    const int D = Q->ne[0];
    if (!fattn_vec_supports_head_dim(D)) {
        return false;
    }

    // MMA kernel handles high head counts (>32), vec kernel handles <=32
    // Both are now supported

    // Check that tensors are contiguous
    if (Q->nb[0] != ggml_type_size(Q->type)) {
        return false;
    }
    if (K->nb[0] != sizeof(sycl::half)) {
        return false;
    }
    if (V->nb[0] != sizeof(sycl::half)) {
        return false;
    }
    if (mask && mask->nb[0] != sizeof(sycl::half)) {
        return false;
    }

    return true;
}

// Dispatcher that selects appropriate kernel based on head dimension and GPU capabilities
template <int D, typename Q_type>
static void ggml_sycl_flash_attn_ext_dispatch_ncols(
    ggml_backend_sycl_context & ctx,
    const fattn_params & params) {

    dpct::queue_ptr stream = ctx.stream();

    // Select ncols based on batch size (ne01 = number of queries)
    const int ne01 = params.ne01;
    float logit_softcap = params.logit_softcap;

    // Runtime kernel selection based on GPU capabilities
    // Check if the device has XMX (Intel matrix extension) support
    sycl::device dev = stream->get_device();
    const bool use_xmx = gpu_has_xmx(dev);

    // Helper macro to dispatch based on softcap
    #define DISPATCH_NCOLS(NCOLS, LAUNCHER) \
        if (logit_softcap == 0.0f) { \
            LAUNCHER<D, NCOLS, false, Q_type>(params, stream); \
        } else { \
            LAUNCHER<D, NCOLS, true, Q_type>(params, stream); \
        }

    // Dispatch to XMX kernel if available, otherwise MMA F16
    if (use_xmx) {
        // XMX kernel - uses Intel joint_matrix for Q@K^T and S@V acceleration
        if (ne01 <= 1) {
            DISPATCH_NCOLS(1, launch_fattn_xmx_f16);
        } else if (ne01 <= 2) {
            DISPATCH_NCOLS(2, launch_fattn_xmx_f16);
        } else if (ne01 <= 4) {
            DISPATCH_NCOLS(4, launch_fattn_xmx_f16);
        } else {
            DISPATCH_NCOLS(8, launch_fattn_xmx_f16);
        }
    } else {
        // MMA F16 kernel - scalar fallback for non-Intel or older Intel GPUs
        if (ne01 <= 1) {
            DISPATCH_NCOLS(1, launch_fattn_mma_f16);
        } else if (ne01 <= 2) {
            DISPATCH_NCOLS(2, launch_fattn_mma_f16);
        } else if (ne01 <= 4) {
            DISPATCH_NCOLS(4, launch_fattn_mma_f16);
        } else {
            DISPATCH_NCOLS(8, launch_fattn_mma_f16);
        }
    }

    #undef DISPATCH_NCOLS
}

// Main flash attention entry point
void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];  // Attention sinks tensor (may be null)

    GGML_ASSERT(Q->type == GGML_TYPE_F32 || Q->type == GGML_TYPE_F16);
    GGML_ASSERT(K->type == GGML_TYPE_F16);
    GGML_ASSERT(V->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Extract scale, max_bias, and logit_softcap from op_params
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // If using logit_softcap, adjust scale
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // Calculate ALiBi parameters
    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // Build the params structure
    fattn_params params;

    params.Q = (const char *) Q->data;
    params.K = (const char *) K->data;
    params.V = (const char *) V->data;
    params.mask = mask ? (const char *) mask->data : nullptr;
    params.sinks = sinks ? (const char *) sinks->data : nullptr;
    params.dst = (float *) dst->data;

    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;
    params.n_head_log2 = n_head_log2;
    params.logit_softcap = logit_softcap;

    // Q dimensions: [batch, n_heads, n_queries, head_dim]
    params.ne00 = Q->ne[0];  // head_dim
    params.ne01 = Q->ne[1];  // n_queries
    params.ne02 = Q->ne[2];  // n_heads
    params.ne03 = Q->ne[3];  // batch

    params.nb01 = Q->nb[1];
    params.nb02 = Q->nb[2];
    params.nb03 = Q->nb[3];

    // K dimensions: [batch, n_kv_heads, n_kv, head_dim]
    params.ne10 = K->ne[0];  // head_dim
    params.ne11 = K->ne[1];  // n_kv (sequence length)
    params.ne12 = K->ne[2];  // n_kv_heads
    params.ne13 = K->ne[3];  // batch

    params.nb11 = K->nb[1];
    params.nb12 = K->nb[2];
    params.nb13 = K->nb[3];

    // V strides
    params.nb21 = V->nb[1];
    params.nb22 = V->nb[2];
    params.nb23 = V->nb[3];

    // Mask dimensions and strides (if present)
    // mask layout: [ne3, ne2, ne1, ne0] = [batch, heads, n_tokens_padded, n_kv]
    if (mask) {
        params.ne30 = mask->ne[0];  // n_kv
        params.ne31 = mask->ne[1];  // n_tokens_padded
        params.ne32 = mask->ne[2];  // heads
        params.ne33 = mask->ne[3];  // batch
        params.nb31 = mask->nb[1];
        params.nb32 = mask->nb[2];
        params.nb33 = mask->nb[3];

        // Assertion: ne11 (K seq len) should equal ne30 (mask's first dim)
        GGML_ASSERT(params.ne11 == params.ne30 && "K sequence length must match mask dimension");
    } else {
        params.ne30 = 0;
        params.ne31 = 0;
        params.ne32 = 0;
        params.ne33 = 0;
        params.nb31 = 0;
        params.nb32 = 0;
        params.nb33 = 0;
    }

    // Dispatch based on head dimension and Q type
    const int D = Q->ne[0];

    if (Q->type == GGML_TYPE_F32) {
        switch (D) {
            case 64:
                ggml_sycl_flash_attn_ext_dispatch_ncols<64, float>(ctx, params);
                break;
            case 128:
                ggml_sycl_flash_attn_ext_dispatch_ncols<128, float>(ctx, params);
                break;
            case 256:
                ggml_sycl_flash_attn_ext_dispatch_ncols<256, float>(ctx, params);
                break;
            default:
                GGML_ABORT("Unsupported head dimension for SYCL flash attention: %d", D);
        }
    } else if (Q->type == GGML_TYPE_F16) {
        switch (D) {
            case 64:
                ggml_sycl_flash_attn_ext_dispatch_ncols<64, sycl::half>(ctx, params);
                break;
            case 128:
                ggml_sycl_flash_attn_ext_dispatch_ncols<128, sycl::half>(ctx, params);
                break;
            case 256:
                ggml_sycl_flash_attn_ext_dispatch_ncols<256, sycl::half>(ctx, params);
                break;
            default:
                GGML_ABORT("Unsupported head dimension for SYCL flash attention: %d", D);
        }
    } else {
         GGML_ABORT("Unsupported Q type for SYCL flash attention");
    }

    // Debug: Dump input/output tensor values to compare with non-FA path
#define FATTN_DUMP_OUTPUT 0
#define FATTN_DUMP_TO_FILE 0
#if FATTN_DUMP_OUTPUT
    {
        // Wait for kernel to complete
        ctx.stream()->wait();

        // Static counter for ALL FA calls (including prefill)
        static int total_fa_calls = 0;
        total_fa_calls++;

        // Dump for first N calls to see both prefill and generation
        const int max_dumps = 50;  // Dump first 50 FA calls
        if (total_fa_calls <= max_dumps) {
            const int D = Q->ne[0];
            const int n_heads = params.ne02;
            const int n_kv_heads = params.ne12;
            const int n_kv = params.ne11;
            const int gqa = n_heads / n_kv_heads;

            // All debug output goes to dump file - no stderr spam
            fprintf(stderr, "  [FA call %d: dst=%p]\n", total_fa_calls, params.dst);

            // Calculate total output size
            const size_t output_size = (size_t)D * params.ne01 * n_heads;
            std::vector<float> host_dst(output_size);
            ctx.stream()->memcpy(host_dst.data(), params.dst, output_size * sizeof(float)).wait();

            // Dump Q input values for ALL heads (if F32)
            std::vector<float> host_Q;
            if (Q->type == GGML_TYPE_F32) {
                const size_t q_size = (size_t)D * params.ne01 * n_heads;
                host_Q.resize(q_size);
                for (int h = 0; h < n_heads; h++) {
                    for (int q = 0; q < params.ne01; q++) {
                        const float* q_ptr = (const float*)(params.Q + params.nb02 * h + params.nb01 * q);
                        ctx.stream()->memcpy(&host_Q[(h * params.ne01 + q) * D], q_ptr, D * sizeof(float)).wait();
                    }
                }
            }

            // Dump K input values for ALL KV heads (first few KV positions)
            std::vector<sycl::half> host_K;
            const int kv_dump_count = std::min(n_kv, 16);  // Dump first 16 KV positions
            const size_t k_dump_size = (size_t)D * kv_dump_count * n_kv_heads;
            host_K.resize(k_dump_size);
            for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                for (int kv = 0; kv < kv_dump_count; kv++) {
                    const sycl::half* k_ptr = (const sycl::half*)(params.K + params.nb12 * kv_h + params.nb11 * kv);
                    ctx.stream()->memcpy(&host_K[(kv_h * kv_dump_count + kv) * D], k_ptr, D * sizeof(sycl::half)).wait();
                }
            }

            // Dump V input values similarly
            std::vector<sycl::half> host_V;
            host_V.resize(k_dump_size);
            for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                for (int kv = 0; kv < kv_dump_count; kv++) {
                    const sycl::half* v_ptr = (const sycl::half*)(params.V + params.nb22 * kv_h + params.nb21 * kv);
                    ctx.stream()->memcpy(&host_V[(kv_h * kv_dump_count + kv) * D], v_ptr, D * sizeof(sycl::half)).wait();
                }
            }

            // Dump mask values if present
            std::vector<sycl::half> host_mask;
            if (params.mask) {
                const int mask_kv = std::min((int)params.ne30, 64);  // First 64 KV positions
                host_mask.resize(mask_kv * params.ne01);
                for (int q = 0; q < params.ne01; q++) {
                    const sycl::half* mask_ptr = (const sycl::half*)(params.mask + params.nb31 * q);
                    ctx.stream()->memcpy(&host_mask[q * mask_kv], mask_ptr, mask_kv * sizeof(sycl::half)).wait();
                }
            }

#if FATTN_DUMP_TO_FILE
            // Write full output to file for comparison
            char filename[256];
            snprintf(filename, sizeof(filename), "/tmp/fa_debug/fa_dump_call%03d.txt", total_fa_calls);
            FILE* f = fopen(filename, "w");
            if (f) {
                fprintf(f, "# FA call %d: ne01=%d ne02=%d ne12=%d D=%d n_kv=%d gqa=%d scale=%.6f nb13=%zu\n",
                        total_fa_calls, params.ne01, n_heads, n_kv_heads, D, n_kv, gqa, params.scale, params.nb13);
                fprintf(f, "# Q strides: nb01=%d nb02=%d nb03=%d\n", params.nb01, params.nb02, params.nb03);
                fprintf(f, "# Output tensor: [%d queries][%d heads][%d D]\n\n", params.ne01, n_heads, D);

                // Write Q input (all values)
                fprintf(f, "=== Q INPUT ===\n");
                if (!host_Q.empty()) {
                    for (int h = 0; h < n_heads; h++) {
                        for (int q = 0; q < params.ne01; q++) {
                            fprintf(f, "Q[h=%d,q=%d]: ", h, q);
                            for (int d = 0; d < D; d++) {
                                fprintf(f, "%.6f ", host_Q[(h * params.ne01 + q) * D + d]);
                            }
                            fprintf(f, "\n");
                        }
                    }
                }

                // Write K input (first kv_dump_count positions)
                fprintf(f, "\n=== K INPUT (first %d positions) ===\n", kv_dump_count);
                for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                    for (int kv = 0; kv < kv_dump_count; kv++) {
                        fprintf(f, "K[kv_h=%d,kv=%d]: ", kv_h, kv);
                        for (int d = 0; d < D; d++) {
                            fprintf(f, "%.6f ", static_cast<float>(host_K[(kv_h * kv_dump_count + kv) * D + d]));
                        }
                        fprintf(f, "\n");
                    }
                }

                // Write V input (first kv_dump_count positions)
                fprintf(f, "\n=== V INPUT (first %d positions) ===\n", kv_dump_count);
                for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                    for (int kv = 0; kv < kv_dump_count; kv++) {
                        fprintf(f, "V[kv_h=%d,kv=%d]: ", kv_h, kv);
                        for (int d = 0; d < D; d++) {
                            fprintf(f, "%.6f ", static_cast<float>(host_V[(kv_h * kv_dump_count + kv) * D + d]));
                        }
                        fprintf(f, "\n");
                    }
                }

                // Write mask if present
                if (!host_mask.empty()) {
                    fprintf(f, "\n=== MASK ===\n");
                    for (int q = 0; q < params.ne01; q++) {
                        fprintf(f, "mask[q=%d]: ", q);
                        int mask_kv = std::min((int)params.ne30, 64);
                        for (int kv = 0; kv < mask_kv; kv++) {
                            float mv = static_cast<float>(host_mask[q * mask_kv + kv]);
                            if (mv < -1e10f) {
                                fprintf(f, "-inf ");
                            } else {
                                fprintf(f, "%.1f ", mv);
                            }
                        }
                        fprintf(f, "\n");
                    }
                }

                // Write sinks if present
                if (params.sinks) {
                    fprintf(f, "\n=== SINKS ===\n");
                    std::vector<float> host_sinks(n_heads);
                    ctx.stream()->memcpy(host_sinks.data(), params.sinks, n_heads * sizeof(float)).wait();
                    fprintf(f, "sinks: ");
                    for (int h = 0; h < n_heads; h++) {
                        fprintf(f, "[h=%d]=%.6f ", h, host_sinks[h]);
                    }
                    fprintf(f, "\n");
                }

                // Write ALL output values
                // New layout: dst[d + D*(head + ne02*(query + ne01*batch))]
                // For batch=0: dst[d + D*(h + ne02*q)]
                fprintf(f, "\n=== FA OUTPUT ===\n");
                for (int h = 0; h < n_heads; h++) {
                    for (int q = 0; q < params.ne01; q++) {
                        fprintf(f, "out[h=%d,q=%d]: ", h, q);
                        for (int d = 0; d < D; d++) {
                            // New layout: d + D*(h + ne02*q)
                            fprintf(f, "%.6f ", host_dst[d + D * (h + n_heads * q)]);
                        }
                        fprintf(f, "\n");
                    }
                }

                fclose(f);
                fprintf(stderr, "  [Wrote full dump to %s]\n", filename);
            }
#endif
        }
    }
#endif
}

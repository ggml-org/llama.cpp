// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#include <arm_neon.h>
#include <assert.h>
#include <atomic>
#include <cfloat>
#include <stdexcept>
#include <stdint.h>
#include <string.h>
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <string_view>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#include <windows.h>
#include <excpt.h>
#endif

#include "kleidiai.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-threading.h"
#include "traits.h"

#include "kernels.h"

#include "kai_common.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

struct ggml_kleidiai_context {
    cpu_feature features;
    ggml_kleidiai_kernels * kernels;
} static ctx = { CPU_FEATURE_NONE, NULL };

static const char* cpu_feature_to_string(cpu_feature f) {
    switch (f) {
        case CPU_FEATURE_NONE:    return "NONE";
        case CPU_FEATURE_DOTPROD: return "DOTPROD";
        case CPU_FEATURE_I8MM:    return "I8MM";
        case CPU_FEATURE_SVE:     return "SVE";
        case CPU_FEATURE_SME:     return "SME";
        default:                  return "UNKNOWN";
    }
}

static void init_kleidiai_context(void) {

    ggml_critical_section_start();
    static bool initialized = false;

    if (!initialized) {
        initialized = true;
        const char *env_var = getenv("GGML_KLEIDIAI_SME");
        int sme_enabled = 0;

        ctx.features  = (ggml_cpu_has_dotprod()     ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                        (ggml_cpu_has_matmul_int8() ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                        (ggml_cpu_has_sve()         ? CPU_FEATURE_SVE     : CPU_FEATURE_NONE);

        if (env_var) {
            sme_enabled = atoi(env_var);
        }

        if (sme_enabled != 0) {
            ctx.features |= ggml_cpu_has_sme() ? CPU_FEATURE_SME : CPU_FEATURE_NONE;
        }
        ctx.kernels = ggml_kleidiai_select_kernels_q4_0(ctx.features);
#ifndef NDEBUG
        if (ctx.kernels) {
            GGML_LOG_DEBUG("kleidiai: using kernel with CPU feature %s\n", cpu_feature_to_string(ctx.kernels->required_cpu));
        }
#endif
    }
    ggml_critical_section_end();
}

static inline int64_t ggml_ne(const ggml_tensor * tensor, int dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    return tensor->ne[dim];
}

template <typename Variant, typename Ret, typename... Args, std::size_t... Is>
constexpr bool variant_any_invocable_impl(std::index_sequence<Is...>) {
    using V = std::remove_reference_t<Variant>;
    return (std::is_invocable_r_v<
                Ret,
                std::variant_alternative_t<Is, V>,
                Args...> || ...);
}

template <typename Variant, typename Ret, typename... Args>
constexpr bool variant_any_invocable_v =
    variant_any_invocable_impl<Variant, Ret, Args...>(
        std::make_index_sequence<
            std::variant_size_v<std::remove_reference_t<Variant>>>{});

template<typename Ret, typename Variant, typename... Args>
static inline Ret variant_call(Variant && var, Args&&... args) {
    static_assert(variant_any_invocable_v<std::remove_reference_t<Variant>, Ret, Args...>,
                  "No alternative in Variant is invocable with the provided arguments and return type.");

    return std::visit(
        [&](auto && f) -> Ret {
            using F = std::decay_t<decltype(f)>;
            if constexpr (std::is_invocable_r_v<Ret, F, Args...>) {
                return std::invoke(std::forward<decltype(f)>(f), std::forward<Args>(args)...);
            } else {
                GGML_ABORT("Invalid function type in variant_call");
                GGML_UNREACHABLE();
            }
        },
        std::forward<Variant>(var)
    );
}

namespace ggml::cpu::kleidiai {

static size_t round_down(size_t x, size_t y) {
    return y == 0 ? x : x - (x % y);
}

static void transpose_f32kxn_f16nxk(size_t n, size_t k, float * dst, const uint16_t * src, size_t rhs_stride) {
    size_t src_stride = rhs_stride / sizeof(uint16_t);
    size_t dst_stride = n;

    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            uint16_t v = *(src + k_idx + n_idx * src_stride);
            *(dst + n_idx + k_idx * dst_stride) = kai_cast_f32_f16(v);
        }
    }
}

class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, op);
        GGML_ASSERT(kernels);
        bool is_gemv = op->src[1]->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;

        size_t k = op->src[0]->ne[0];
        size_t n = op->src[0]->ne[1];
        size_t m = op->src[1]->ne[1];

        size_t mr = kernel->get_mr();
        size_t kr = kernel->get_kr();
        size_t sr = kernel->get_sr();

        if (kernels->rhs_type == GGML_TYPE_Q4_0) {
            size = variant_call<size_t>(lhs_info->packed_size, m, k, QK4_0, mr, kr, sr);
        } else if (kernels->rhs_type == GGML_TYPE_F16) {
            const int64_t lhs_batch_size0 = op->src[1]->ne[2];
            const int64_t rhs_batch_size0 = op->src[0]->ne[2];
            const int64_t r = lhs_batch_size0 / rhs_batch_size0;
            size = variant_call<size_t>(lhs_info->packed_size, m * r, k, mr, kr, sr) +
                   variant_call<size_t>(kernels->rhs_info.packed_size, n, k) +
                   k * n * sizeof(float) + n * sizeof(float);
        } else {
            GGML_ASSERT(false);
        }

        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * dst) override {
        if (dst->op == GGML_OP_MUL_MAT) {
            if (dst->src[0]->type == GGML_TYPE_Q4_0) {
                return compute_forward_q4_0(params, dst);
            } else if (dst->src[0]->type == GGML_TYPE_F16) {
                return compute_forward_fp16(params, dst);
            }
        } else if (dst->op == GGML_OP_GET_ROWS) {
            if (dst->src[0]->type == GGML_TYPE_Q4_0) {
                return compute_forward_get_rows(params, dst);
            }
        }
        return false;
    }

    bool compute_forward_fp16(ggml_compute_params * params, struct ggml_tensor * dst) {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        GGML_ASSERT(kernels);

        const bool is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;
        GGML_ASSERT(kernel);

        const int nth = params->nth;
        const int ith = params->ith;

        const int64_t lhs_batch_size0 = ne12;
        const int64_t rhs_batch_size0 = ne02;
        const int64_t batch_size      = lhs_batch_size0;

        GGML_ASSERT(rhs_batch_size0 > 0);
        GGML_ASSERT(lhs_batch_size0 % rhs_batch_size0 == 0);
        const int64_t r = lhs_batch_size0 / rhs_batch_size0;

        const int64_t m_group = ne11;
        const int64_t m       = m_group;
        const int64_t n       = ne01;
        const int64_t k       = ne00;

        const size_t lhs_stride = src1->nb[1];
        const size_t rhs_stride = src0->nb[1];
        const size_t dst_stride = dst->nb[1];

        const int64_t mr = (int64_t) kernel->get_mr();
        const int64_t nr = (int64_t) kernel->get_nr();
        const int64_t kr = (int64_t) kernel->get_kr();
        const int64_t sr = (int64_t) kernel->get_sr();

        const size_t lhs_packed_size = variant_call<size_t>(lhs_info->packed_size, (size_t)m, (size_t)k, (size_t)mr, (size_t)kr, (size_t)sr);
        const size_t rhs_packed_size = variant_call<size_t>(kernels->rhs_info.packed_size, (size_t)n, (size_t)k);
        const size_t kxn_size        = (size_t)k * (size_t)n * sizeof(float);
        const size_t bias_size       = (size_t)n * sizeof(float);

        const size_t wsize_required = lhs_packed_size + rhs_packed_size + kxn_size + bias_size;
        GGML_ASSERT(wsize_required <= params->wsize);

        uint8_t * lhs_packed = static_cast<uint8_t *>(params->wdata);
        uint8_t * rhs_packed = lhs_packed + lhs_packed_size;
        uint8_t * rhs_kxn    = rhs_packed + rhs_packed_size;
        uint8_t * bias       = rhs_kxn + kxn_size;

        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const int64_t rhs_batch_idx = batch_idx / r;
            const uint8_t * rhs_batch_base = static_cast<const uint8_t *>(src0->data) + rhs_batch_idx * src0->nb[2];
            uint8_t * dst_batch_base = static_cast<uint8_t *>(dst->data) + batch_idx * dst->nb[2];

            // LHS packing (threaded over m, honoring mr alignment and KV groups)
            {
                const int64_t m_roundup_mr = kai_roundup(m, mr);
                const int64_t num_threads  = KAI_MIN(m_roundup_mr / mr, nth);

                if (ith < num_threads) {
                    const int64_t num_m_per_thread0   = round_down((size_t)(m_roundup_mr / num_threads), (size_t)mr);
                    const int64_t num_m_per_threadN_1 = m - (num_threads - 1) * num_m_per_thread0;

                    const int64_t m_start = ith * num_m_per_thread0;
                    const int64_t m_count = (ith == num_threads - 1) ? num_m_per_threadN_1 : num_m_per_thread0;

                    // Base packed offset (aligned) and per-row stride in bytes
                    const size_t base_packed_off = variant_call<size_t>(
                        lhs_info->get_packed_offset, (size_t)m_start, (size_t)k, (size_t)mr, (size_t)kr, (size_t)sr);
                    const size_t next_block_off = variant_call<size_t>(
                        lhs_info->get_packed_offset, (size_t)(m_start + mr), (size_t)k, (size_t)mr, (size_t)kr, (size_t)sr);
                    const size_t row_stride_bytes = (next_block_off - base_packed_off) / (size_t)mr;

                    int64_t remaining = m_count;
                    int64_t cur       = m_start;

                    while (remaining > 0) {
                        const int64_t row_in_group = cur;
                        const int64_t avail        = m_group - row_in_group;
                        const int64_t take         = std::min(avail, remaining);

                        const uint8_t * lhs_batch_base = static_cast<const uint8_t *>(src1->data) + batch_idx * src1->nb[2];
                        const void * src_ptr = lhs_batch_base + (size_t)row_in_group * lhs_stride;
                        const size_t dst_off = base_packed_off + (size_t)(cur - m_start) * row_stride_bytes;
                        void * dst_ptr       = lhs_packed + dst_off;

                        variant_call<void>(lhs_info->pack_func,
                                        (size_t)take, (size_t)k, (size_t)mr, (size_t)kr, (size_t)sr,
                                        /*m_idx_start*/ 0, src_ptr, lhs_stride, dst_ptr);

                        cur       += take;
                        remaining -= take;
                    }
                }
            }

            // RHS packing (single thread), then synchronize
            if (ith == 0) {
                memset(bias, 0, (size_t)n * sizeof(float));
                transpose_f32kxn_f16nxk((size_t)n, (size_t)k,
                                        reinterpret_cast<float *>(rhs_kxn),
                                        reinterpret_cast<const uint16_t *>(rhs_batch_base),
                                        rhs_stride);

                variant_call<void>(kernels->rhs_info.pack_func,
                                   /*num_groups*/ 1, (size_t)n, (size_t)k, (size_t)nr, (size_t)kr, (size_t)sr,
                                   /*rhs_stride (bytes)*/ (size_t)(n * sizeof(float)),
                                   rhs_kxn, bias, nullptr, rhs_packed, /*extra_bytes*/ 0, /*params*/ nullptr);
            }

            ggml_barrier(params->threadpool);

            // Matmul (threaded over n)
            {
                const int64_t n_step  = (int64_t) kernel->get_n_step();
                int64_t num_threads_n = KAI_MIN(n / n_step, nth);
                if (num_threads_n <= 0) {
                    num_threads_n = 1;
                }

                if (ith < num_threads_n) {
                    const int64_t num_n_per_thread0   = round_down((size_t)(n / num_threads_n), (size_t)n_step);
                    const int64_t num_n_per_threadN_1 = n - (num_threads_n - 1) * num_n_per_thread0;

                    const int64_t n_start      = ith * num_n_per_thread0;
                    const int64_t n_to_process = (ith == num_threads_n - 1) ? num_n_per_threadN_1 : num_n_per_thread0;

                    // LHS packed base at row 0 (consistent with packing above)
                    const size_t lhs_packed_offset0 = variant_call<size_t>(
                        lhs_info->get_packed_offset, (size_t)0, (size_t)k, (size_t)mr, (size_t)kr, (size_t)sr);
                    const size_t rhs_packed_offset = variant_call<size_t>(kernel->get_rhs_packed_offset, (size_t)n_start, (size_t)k);
                    const size_t dst_offset        = kernel->get_dst_offset((size_t)0, (size_t)n_start, dst_stride);

                    const void * lhs_ptr = lhs_packed + lhs_packed_offset0;
                    const void * rhs_ptr = rhs_packed + rhs_packed_offset;
                    float * dst_ptr      = reinterpret_cast<float *>(dst_batch_base + dst_offset);

                    variant_call<void>(kernel->run_kernel,
                                       (size_t)m, (size_t)n_to_process, (size_t)k,
                                       lhs_ptr, rhs_ptr,
                                       dst_ptr, dst_stride, sizeof(float),
                                       -FLT_MAX, FLT_MAX);
                }
            }

            if (batch_idx != batch_size - 1) {
                ggml_barrier(params->threadpool);
            }
        }

        return true;
    }

    bool compute_forward_q4_0(struct ggml_compute_params * params, struct ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0);

        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        ggml_kleidiai_kernels *kernels = ggml_kleidiai_select_kernels(ctx.features, dst);
        GGML_ASSERT(kernels);

        bool is_gemv = src1->ne[1] == 1;
        kernel_info * kernel = is_gemv ? &kernels->gemv : &kernels->gemm;
        lhs_packing_info * lhs_info = is_gemv ? &kernels->gemv_lhs_info : &kernels->gemm_lhs_info;

        GGML_ASSERT(kernel);

        const int ith = params->ith;
        const int nth_raw = params->nth;
        const int nth = nth_raw > 0 ? nth_raw : 1;

        const size_t k = ne00;
        const size_t m = ne11;
        const size_t n = ne01;

        size_t mr = kernel->get_mr();
        size_t kr = kernel->get_kr();
        size_t sr = kernel->get_sr();

        const uint8_t * lhs        = static_cast<const uint8_t *>(src1->data);
        uint8_t * lhs_packed       = (uint8_t*)params->wdata;
        const uint8_t * rhs_packed = static_cast<const uint8_t *>(src0->data);

        const size_t n_step = kernel->get_n_step();
        const size_t num_n_per_thread = kai_roundup(kai_roundup(n, nth) / nth, n_step);
        const size_t n_start = ith * num_n_per_thread;

        size_t n_to_process = 0;
        if (n_start < n) {
            n_to_process = num_n_per_thread;
            if ((n_start + n_to_process) > n) {
                n_to_process = n - n_start;
            }
        }

        // Calculate number of columns to be processed per thread
        const size_t num_m_per_thread = kai_roundup(m, mr * nth) / nth;
        const size_t m_start = ith * num_m_per_thread;
        size_t m_to_process = num_m_per_thread;
        if ((m_start + m_to_process) > m) {
            m_to_process = m - m_start;
        }

        if (m_start < m) {
            // Transform LHS
            const size_t src_stride        = src1->nb[1];
            const float * src_ptr          = reinterpret_cast<const float *>(lhs + lhs_info->get_offset(m_start, dst->src[1]->nb[1]));
            const size_t lhs_packed_offset = variant_call<size_t>(lhs_info->get_packed_offset, m_start, k, QK4_0, mr, kr, sr);
            void * lhs_packed_ptr          = static_cast<void *>(lhs_packed + lhs_packed_offset);

            variant_call<void>(lhs_info->pack_func, m_to_process, k, QK4_0, mr, kr, sr, 0, src_ptr, src_stride, lhs_packed_ptr);
        }

        ggml_barrier(params->threadpool);

        // Perform the operation
        const size_t dst_stride        = dst->nb[1];
        const size_t lhs_packed_offset = variant_call<size_t>(lhs_info->get_packed_offset, 0, k, QK4_0, mr, kr, sr);
        const size_t rhs_packed_offset = variant_call<size_t>(kernel->get_rhs_packed_offset, n_start, k, QK4_0);
        const size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);
        const void * rhs_ptr           = static_cast<const void *>(rhs_packed + rhs_packed_offset);
        const void* lhs_ptr            = (const void*)((const char *)lhs_packed + lhs_packed_offset);
        float *dst_ptr                 = reinterpret_cast<float *>(static_cast<uint8_t *>(dst->data) + dst_offset);

        if (n_to_process > 0) {
            variant_call<void>(kernel->run_kernel, m, n_to_process, k, QK4_0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride,
                               sizeof(float), -FLT_MAX, FLT_MAX);
        }

        return true;
    }

    bool compute_forward_get_rows(struct ggml_compute_params * params, struct ggml_tensor * dst) {
        GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q4_0);
        GGML_ASSERT(ctx.kernels);

        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        rhs_packing_info * rhs_info = &ctx.kernels->rhs_info;
        kernel_info * kernel        = &ctx.kernels->gemm;

        const int64_t nc     = ne00;
        const int64_t nr     = ggml_nelements(src1);

        const size_t block_rows = kernel->get_nr();
        const size_t kr         = kernel->get_kr();

        const size_t num_bytes_multiplier = sizeof(uint16_t);
        const size_t packed_stride = rhs_info->packed_stride(nc, block_rows, kr, QK4_0);

        const int ith = params->ith;
        const int nth = params->nth;

        const int dr = (nr + nth - 1) / nth;
        const int ir0 = dr * ith;
        const int ir1 = MIN(ir0 + dr, nr);

        for (int64_t i = ir0; i < ir1; ++i) {
            GGML_ASSERT(src1->type == GGML_TYPE_I32);
            int64_t row_idx = ((const int32_t *)src1->data)[i];
            GGML_ASSERT(row_idx >= 0 && row_idx < src0->ne[1]);

            float *out = (float *)((char *)dst->data + i * nb1);
            rhs_info->to_float(src0->data, row_idx, nc, out, block_rows, packed_stride, kr, QK4_0, num_bytes_multiplier);
        }

        return true;
    }

public:
    int repack(struct ggml_tensor * tensor, const void * data, size_t data_size) {
        GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0);
        GGML_ASSERT(ctx.kernels);
        const size_t n = tensor->ne[1];
        const size_t k = tensor->ne[0];
        size_t nr      = ctx.kernels->gemm.get_nr();
        size_t kr      = ctx.kernels->gemm.get_kr();
        size_t sr      = ctx.kernels->gemm.get_sr();

        struct kai_rhs_pack_qs4cxs1s0_param params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        variant_call<void>(ctx.kernels->rhs_info.pack_func, 1, n, k, nr, kr, sr, QK4_0, (const uint8_t*)data, nullptr, tensor->data, 0, &params);

        return 0;
        GGML_UNUSED(data_size);
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::kleidiai

static enum ggml_status ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                       const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_KLEIDIAI";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(ctx.kernels);

    const size_t n  = tensor->ne[1];
    const size_t k  = tensor->ne[0];
    const size_t nr = ctx.kernels->gemm.get_nr();
    const size_t kr = ctx.kernels->gemm.get_kr();

    return variant_call<size_t>(ctx.kernels->rhs_info.packed_size, n, k, nr, kr, QK4_0);

    GGML_UNUSED(buft);
}

namespace ggml::cpu::kleidiai {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_GET_ROWS) &&
            op->src[0]->type == GGML_TYPE_Q4_0 &&
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type() && ctx.kernels) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if ((op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_I32) &&
                ggml_ne(op->src[1], 2) == 1 && ggml_ne(op->src[1], 3) == 1) {
                return true;
            }
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_GET_ROWS) {
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type()) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
            else if (ggml_kleidiai_select_kernels(ctx.features, op) && op->src[1]->ne[1] > 1) {
                if ((op->src[0]->nb[1] * op->src[0]->ne[1] != op->src[0]->nb[2]) ||
                    (op->src[1]->nb[1] * op->src[1]->ne[1] != op->src[1]->nb[2])) {
                    return nullptr;
                }

                return ggml::cpu::kleidiai::get_tensor_traits(NULL, NULL);
            }
        }
        return nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(void) {
    static ggml::cpu::kleidiai::extra_buffer_type ctx;
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ ggml_backend_cpu_kleidiai_buffer_type_get_alloc_size,
                           /* .is_host          = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context();

    return &ggml_backend_cpu_buffer_type_kleidiai;
}

#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ime.h"

#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ime_kernels.h"
#include "repack.h"
#include "rvv_kernels.h"
#include "scalar_kernels.h"
#include "spacemit-context.h"
#include "spacemit-env.h"
#include "spacemit-fused.h"
#include "spacemit-opnode.h"
#include "spine_mem_pool.h"
#include "vec.h"

#include "spe_kernels.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <vector>
// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#if !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#if defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#else
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        return sizeof(block_q8_0);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        return sizeof(block_q4_0) * INTER_SIZE / QK4_0;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        return (sizeof(block_q4_0) + sizeof(uint8_t)) * INTER_SIZE / QK4_1;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        return sizeof(spacemit_kernels::nrow_block_q2_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        return sizeof(spacemit_kernels::nrow_block_q3_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return sizeof(spacemit_kernels::nrow_block_q5_1<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return sizeof(spacemit_kernels::nrow_block_q5_0<1>);
    } else {
        assert(false);
        return 0;
    }
}

template <typename BLOC_TYPE> constexpr bool block_type_has_zp() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return false;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return true;
    } else {
        assert(false);
        return false;
    }
}

static size_t mmid_m4_work_size(int n_threads, const ggml_tensor * op, size_t nbw1) {
    const int64_t n_rows = op->src[2]->ne[0] * op->src[2]->ne[1];
    const int64_t n_as   = op->src[0]->ne[2];
    size_t        size   = 64;
    size += GGML_PAD(n_rows * nbw1, 64) + n_as * (spacemit_kernels::q8_hp_al_align + 8 * nbw1);
    size += GGML_PAD(n_as * sizeof(int64_t), 64);
    size += GGML_PAD((n_rows / 4 + 1 + n_as) * 3 * sizeof(int32_t) + sizeof(int64_t), 64);
    size += (size_t) n_threads * GGML_PAD((8 * op->src[1]->ne[0] + 8 * op->src[0]->ne[1]) * sizeof(float), 64);
    return size;
}

struct gemm_path_i8 {
    spacemit_kernels::gemm_kernel_quantize_def kernel;        // nullptr: no impl
    spacemit_kernels::quantize_a_row_def      quantize_a_r1;
    spacemit_kernels::quantize_a_row_def      quantize_a_r4;  // nullptr: no 4-row group packing
    spacemit_kernels::a_pack                  ap_r4;
    size_t                                    row_stride{};     // per-row stride of r1 rows in the quant buffer
    size_t                                    row_stride_r4{};  // per-row stride inside r4 groups (>= row_stride)
    spacemit_kernels::gemm_rows_def           gemm_rows;

    gemm_path_i8() = default;

    gemm_path_i8(spacemit_kernels::gemm_kernel_quantize_def k, spacemit_kernels::quantize_a_row_def r1,
                 spacemit_kernels::quantize_a_row_def r4, spacemit_kernels::a_pack ap, size_t rs, size_t rs4,
                 spacemit_kernels::gemm_rows_def rows = nullptr) :
        kernel(std::move(k)), quantize_a_r1(std::move(r1)), quantize_a_r4(std::move(r4)), ap_r4(ap),
        row_stride(rs), row_stride_r4(rs4), gemm_rows(std::move(rows)) {}
};

// Dense Q4 uses the F16 path; this table serves MoE Q4 and the other integer paths.
template <typename BLOC_TYPE, int64_t INTER_SIZE, bool MOE>
static gemm_path_i8 resolve_gemm_i8() {
    constexpr size_t blk_len = INTER_SIZE;

#if defined(RISCV64_SPACEMIT_IME2)
    if (global_spine_env_info.use_ime2) {
        using spacemit_kernels::a_pack;
        const auto std_r1  = spacemit_kernels::rvv::quantize_a_i8_r1;
        const auto std_r4  = spacemit_kernels::rvv::quantize_a_i8_r4;
        const size_t std_row = spacemit_kernels::q8_blk_size(blk_len, true);

        if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
            const size_t al_row = spacemit_kernels::q8_m4al_row_blk_size;
            if constexpr (MOE) {
                return { spacemit_kernels::ime2::gemm_i8i8_k32, std_r1, nullptr, a_pack::r1, std_row, std_row };
            } else {
                return { spacemit_kernels::ime2::gemm_i8i8_k32, std_r1,
                         spacemit_kernels::rvv::quantize_a_i8_r4_al, a_pack::r4_al, std_row, al_row };
            }
        } else if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K>) {
            if constexpr (MOE) {
                return { spacemit_kernels::ime2::gemm_i8i8_k32, std_r1, nullptr, a_pack::r1, std_row, std_row };
            } else {
                // Q6_K uses the Q8_0 N32xK32 layout.
                return { spacemit_kernels::ime2::gemm_i8i8_k32, std_r1,
                         spacemit_kernels::rvv::quantize_a_i8_r4_al, a_pack::r4_al, std_row,
                         spacemit_kernels::q8_m4al_row_blk_size };
            }
        } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                             std::is_same_v<BLOC_TYPE, block_q4_K>) {
            if constexpr (MOE && INTER_SIZE == 256) {
                const auto hp_r1   = spacemit_kernels::rvv::quantize_a_i8_r1_hp;
                const size_t hp_row = spacemit_kernels::q8_hp_blk_size(blk_len, true, true);
                // A null zero-point pointer selects Q4_0's folded-nibble encoding.
                return { spacemit_kernels::ime2::gemm_i8i4_k256, hp_r1,
                         spacemit_kernels::rvv::quantize_a_i8_r4_hp_al, a_pack::r4_al, hp_row,
                         spacemit_kernels::q8_hp_al_row_blk_size };
            } else if constexpr (MOE) {
                return { spacemit_kernels::ime2::gemm_i8i4_k32, std_r1, nullptr, a_pack::r1, std_row, std_row,
                         spacemit_kernels::ime2::gemm_i8i4_k32_rows };
            }
        } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
            const auto k_r1 = spacemit_kernels::rvv::quantize_a_i8_r1_k;
            const auto k_r4 = spacemit_kernels::rvv::quantize_a_i8_r4_k;
            const size_t k_row = spacemit_kernels::q8k_blk_size(blk_len);
            const auto kernel = std::is_same_v<BLOC_TYPE, block_q2_K> ?
                                    spacemit_kernels::ime2::gemm_i8i2k_k256 :
                                    spacemit_kernels::ime2::gemm_i8i3k_k256;
            if constexpr (MOE) {
                return { kernel, k_r1, nullptr, a_pack::r1, k_row, k_row };
            } else {
                return { kernel, k_r1, k_r4, a_pack::r4, k_row, k_row };
            }
        } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                             std::is_same_v<BLOC_TYPE, block_q5_0>) {
            if constexpr (MOE) {
                return { spacemit_kernels::ime2::gemm_i8i5_k32, std_r1, nullptr, a_pack::r1, std_row, std_row,
                         spacemit_kernels::ime2::gemm_i8i5_k32_rows };
            } else {
                return { spacemit_kernels::ime2::gemm_i8i5_k32, std_r1, std_r4, a_pack::r4, std_row, std_row };
            }
        }
    }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
    if (global_spine_env_info.use_ime1) {
        using spacemit_kernels::a_pack;
        const auto i1_r1 = spacemit_kernels::ime1::quantize_a_i8_r1;
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                      std::is_same_v<BLOC_TYPE, block_q4_K>) {
            if constexpr (MOE) {
                const size_t i1_row = spacemit_kernels::q8_blk_size(blk_len);
                return { spacemit_kernels::ime1::gemm_i8i4_k32, i1_r1, nullptr, a_pack::r1, i1_row, i1_row };
            }
        }
    }
#endif

    return {};
}

struct gemm_path_f16 {
    mlir::speir::spekernel::spe_pack_cast_function_def castpack_a;     // f32 -> f16 [K/8][16][8]
    mlir::speir::spekernel::spe_pack_cast_function_def dequantpack_b;  // Q41Block -> f16 [K/8][32][8]
    mlir::speir::spekernel::spe_reducesum_mxn_function_def reducesum;  // f16 partials -> f32 (split-K)

    // Decode uses INT32 GEMV directly over Q41Block.
    spacemit_kernels::quantize_a_row_def quantize_a_gemv;
    using dot_gemv_def   = decltype(&spacemit_kernels::ime2::gemm_kernel_i8i4_m1_q41);
    using dot_f16acc_def = decltype(&spacemit_kernels::ime2::gemm_kernel_f16f16f16_m16_vfwmadot);
    using dot_f32acc_def = decltype(&spacemit_kernels::ime2::gemm_kernel_f16f16f32_m16_vfwmadot);
    dot_gemv_def   dot_gemv;
    dot_f16acc_def dot_f16acc;  // F16 partial output.
    dot_f32acc_def dot_f32acc;  // F32 output, optionally accumulated.
};

using q41_zero_point_mode = mlir::speir::spekernel::ime2::q41_zero_point_mode;

template <typename BLOC_TYPE, size_t NB, size_t KB, q41_zero_point_mode Mode>
static gemm_path_f16 resolve_gemm_f16() {
    static_assert(std::is_same_v<BLOC_TYPE, Q41Block<32, 32, 2>>, "the f16 link has a single B layout");
    // A100 dispatch requires IME2, 128-byte vectors and RVV.
    SpineArchInfoDef march{};
    march.fields.core_arch_id = SpineCoreArchID::CoreArchA100;
    march.fields.matrix_info  = 128;
    march.fields.vector_info  = 128;
    march.fields.core_ext_info.fields.rvv = (1 << 0);
    march.fields.core_ext_info.fields.ime = (1 << 1);

    gemm_path_f16 o;
    o.castpack_a = mlir::speir::spekernel::ime2::get_spe_pack_cast_function(
        march.value, typeid(float), typeid(_spe_fp16_), 1, 1, 16, 8);
    o.dequantpack_b = mlir::speir::spekernel::ime2::get_spe_pack_cast_function(
        march.value, typeid(BLOC_TYPE), typeid(_spe_fp16_), NB, KB, NB, 8, Mode);
    o.reducesum = mlir::speir::spekernel::rvv::get_spe_reducesum_mxn_function(
        march.value, typeid(_spe_fp16_), typeid(float), 16, 32, 1, 1);
    o.quantize_a_gemv = &spacemit_kernels::rvv::quantize_a_i8_r1;
    o.dot_gemv        = &spacemit_kernels::ime2::gemm_kernel_i8i4_m1_q41;
    o.dot_f16acc      = &spacemit_kernels::ime2::gemm_kernel_f16f16f16_m16_vfwmadot;
    o.dot_f32acc      = &spacemit_kernels::ime2::gemm_kernel_f16f16f32_m16_vfwmadot;
    if (!o.castpack_a || !o.dequantpack_b || !o.reducesum) {
        GGML_ABORT("f16 link: spe dispatch getter returned null (pack=%d q41=%d reduce=%d)",
                   o.castpack_a ? 1 : 0, o.dequantpack_b ? 1 : 0, o.reducesum ? 1 : 0);
    }
    return o;
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> class tensor_traits : public ggml::spacemit::tensor_traits_base {
    const gemm_path_i8 dense_ops_;
    const gemm_path_i8 moe_ops_;

public:
    tensor_traits() : dense_ops_(resolve_gemm_i8<BLOC_TYPE, INTER_SIZE, false>()),
                      moe_ops_(resolve_gemm_i8<BLOC_TYPE, INTER_SIZE, true>()) {}

private:
    static bool hp_aligned() {
        if constexpr ((std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                       std::is_same_v<BLOC_TYPE, block_q4_K>) &&
                      INTER_SIZE == 256) {
            return true;
        } else {
            return false;
        }
    }

    // Full r4_al groups use 48 B/row; r1 tails use 38 B/row.
    static bool q8_al_layout() {
        if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0> || std::is_same_v<BLOC_TYPE, block_q6_K>) {
            return true;
        } else {
            return false;
        }
    }

    static bool a_aligned() { return q8_al_layout(); }

    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               (q8_al_layout() ? spacemit_kernels::q8_m4al_row_blk_size :
                                           spacemit_kernels::q8_blk_size(QK4_0, true));
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));
                    if (a_aligned()) {
                        size = GGML_PAD(size, spacemit_kernels::q8_hp_al_align) + spacemit_kernels::q8_hp_al_align;
                    }

                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    const int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * sizeof(int64_t);

                    size = GGML_PAD(size, sizeof(int64_t));

                    if constexpr (INTER_SIZE == 256 && !std::is_same_v<BLOC_TYPE, block_q2_K> &&
                                  !std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        const size_t nbw1 = spacemit_kernels::div_round_up(op->src[1]->ne[0], 256) *
                                            (hp_aligned() ? spacemit_kernels::q8_hp_al_row_blk_size :
                                                            spacemit_kernels::q8_hp_blk_size(256, true, true));
                        size += mmid_m4_work_size(n_threads, op, nbw1);
                        if (hp_aligned()) {
                            size += spacemit_kernels::q8_hp_al_align;
                        }
                    }

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        forward_mul_mat(ctx, op);
                        return true;
                    case GGML_TYPE_F32:
                        if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                            forward_mul_mat(ctx, op);
                            return true;
                        }
                        return false;
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        return false;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        forward_mul_mat_id(ctx, op);
                        return true;
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        return false;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat(ggml::spacemit::context & ctx, ggml_tensor * op) const {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = ctx.ith;
        int nth = ctx.nth;

        [[maybe_unused]] const enum ggml_type type = src0->type;

        void *        w_data  = (void *) src0->data;
        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const int64_t gemm_m = ne11 * ne12 * ne13;
        const int64_t gemm_k = ne10;
        const int64_t gemm_n = ne01;

        const gemm_path_i8 & path = dense_ops_;
        if (!path.kernel) {
            GGML_ABORT("no kernel implementation found for the block type");
        }
        const auto & gemm_kernel      = path.kernel;
        const auto & quantize_a_i8_r1 = path.quantize_a_r1;
        const auto & quantize_a_i8_r4 = path.quantize_a_r4;
        const auto   a_pack_r4         = path.ap_r4;

        // Reserve the larger row stride for mixed r4_al groups and r1 tails.
        const int64_t block_stride_a = path.quantize_a_r4 ? path.row_stride_r4 : path.row_stride;

        const int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        const int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        const int64_t row_stride_a        = a_k_blks * block_stride_a;
        const int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        if (ith == 0 && ctx.workspace_size < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(ctx.workspace);
        if (a_aligned()) {
            const int64_t ws_align = spacemit_kernels::q8_hp_al_align;
            ws_ptr                = (ws_ptr + ws_align - 1) & ~(uintptr_t) (ws_align - 1);
        }

        void *        tcm_buffer      = ctx.shared.data;
        const int64_t tcm_buffer_size = (int64_t) ctx.shared.size;

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        const int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        const int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const int64_t per_mb_rows_wsize = row_align * row_stride_a;
        const int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < static_cast<int64_t>(spine_init_barrier_count));
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        if (gemm_m == 1) {
            const int64_t blk_stride_1row =
                q8_al_layout() ? (int64_t) spacemit_kernels::q8_blk_size(a_blk_len, true) : block_stride_a;
            int task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            int a_blk_start     = ith * task_per_thread;
            int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
            if (a_blk_start < a_blk_end) {
                quantize_a_i8_r1(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * blk_stride_1row);
            }
        } else {
            int task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            int m_row_blk_start = ith * task_per_thread;
            int m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (int) row_blks);
            for (int m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                int m_idx             = m_row_blk * row_align;
                int rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                if (rows_tobe_handled == row_align && quantize_a_i8_r4 != nullptr) {
                    const float * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_i8_r4(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } else {
                    while (rows_tobe_handled) {
                        const float * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_i8_r1(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ctx.sync();

        const int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        const int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        const int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        if (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        if (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            for (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    float *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(rows_remaining >= 4 ? a_pack_r4 : spacemit_kernels::a_pack::r1,
                                                        b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } else if (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            if ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            // Direct IME-from-DRAM GEMV path, decode only (gemm_m == 1), q4_0 only:
            // q8_0 always loses to the staged TCM path here.
            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256 && NB_COLS == 32) {
                if (gemm_m == 1 && a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                    constexpr int64_t tile_cols = 4 * NB_COLS;
                    for (int64_t ni = (int64_t) ith * tile_cols; ni < gemm_n; ni += tile_cols * nth) {
                        const int64_t nb_real  = std::min(gemm_n - ni, tile_cols);
                        uint8_t *     b_row    = reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b;
                        uint8_t *     b_row_zp = block_type_has_zp<BLOC_TYPE>() ? b_row : nullptr;
                        gemm_kernel(spacemit_kernels::a_pack::r1, b_blk_len, a_row, b_row, b_row_zp, output + ni, 1,
                                    nb_real, b_k_blks, gemm_n);
                    }
                    return;
                }
            }

            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

            const int64_t ni0     = (int64_t) ith * NB_COLS;
            const bool    active0 = ni0 < gemm_n;
            const int64_t nb0     = active0 ? std::min(gemm_n - ni0, (int64_t) NB_COLS) : 0;

            if (ith % 2 == 0 && active0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni0 * row_stride_b,
                                                nb0 * row_stride_b);
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            if (has_pair) {
                spine_barrier_wait(cur_barrier);
            }

            if (ith % 2 != 0 && active0) {
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni0 * row_stride_b,
                                                nb0 * row_stride_b);
            }

            const int64_t base_start = (int64_t) (ith & ~1) * NB_COLS;
            const int64_t lane_off   = (int64_t) (ith & 1) * NB_COLS;
            for (int64_t base = base_start; base < gemm_n; base += NB_COLS * nth) {
                const int64_t ni      = base + lane_off;
                const bool    active  = ni < gemm_n;
                const int64_t nb_real = active ? std::min(gemm_n - ni, (int64_t) NB_COLS) : 0;

                if (has_pair && ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                if (active) {
                    int64_t rows_remaining = gemm_m;
                    float * c_blk          = output + ni;
                    auto *  a_row_cur      = a_row;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(rows_remaining >= 4 ? a_pack_r4 : spacemit_kernels::a_pack::r1,
                                                        b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                        nb_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_cur += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }

                if (has_pair && ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                const int64_t next_ni = ni + NB_COLS * nth;
                if (next_ni < gemm_n) {
                    const int64_t next_nb = std::min(gemm_n - next_ni, (int64_t) NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    next_nb * row_stride_b);
                }
            }
        } else {
            const int64_t task_count_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            const int64_t task_count_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_count      = task_count_m * task_count_n;
            int64_t task_per_thread = (task_count + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_count);
            for (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                const auto tid_n = compute_idx / task_count_m;
                const auto tid_m = compute_idx % task_count_m;

                const int64_t m_start = tid_m * gemm_m_stride;
                const int64_t m_count = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                const int64_t n_start = tid_n * gemm_n_stride;
                const int64_t n_count = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                const int64_t n_blk = m_count == 1 ? n_count : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < n_count; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_count - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    float * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_count;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(rows_remaining >= 4 ? a_pack_r4 : spacemit_kernels::a_pack::r1,
                                                        b_blk_len, a_row, b_col_cur, b_col_zp_cur, c_blk,
                                                        rows_remaining, n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        }
    }

    void forward_mul_mat_id(ggml::spacemit::context & ctx, ggml_tensor * op) const {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = ctx.ith;
        int nth = ctx.nth;

        // row groups
        const int n_ids = ids->ne[0];  // n_expert_used
        const int n_as  = ne02;        // n_expert

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        const gemm_path_i8 & path = moe_ops_;
        if (!path.kernel) {
            GGML_ABORT("no kernel implementation found for the block type");
        }
        const auto & gemm_kernel        = path.kernel;
        const auto & gemm_rows_kernel_m2 = path.gemm_rows;
        const auto & quantize_a_i8_r1   = path.quantize_a_r1;
        const auto & quantize_a_i8_r4   = path.quantize_a_r4;
        const auto   a_pack_r4          = path.ap_r4;
        const auto   a_pack_r8          = spacemit_kernels::a_pack::r8;

        const size_t block_stride_a = path.row_stride;

        const size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        const size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        const size_t nbw1                = a_k_blks * block_stride_a;
        const size_t nbw2                = ne11 * nbw1;
        const size_t nbw3                = nbw2 * ne12;
        const size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        const uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(ctx.workspace);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        bool pad8_all = false;
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256) {
            pad8_all = ids->ne[1] > 1 && quantize_a_i8_r4 != nullptr && ne01 % 32 == 0 && hp_aligned();
        }

        if (!pad8_all && ne11 == 1) {
            for (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_i8_r1(a_blk_len, (float *) ((char *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } else if (!pad8_all) {
            for (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_i8_r1(a_blk_len, (float *) ((char *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_count          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_count         = (int32_t *) (valid_ep_count + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_count + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        const bool use_m4  = quantize_a_i8_r4 != nullptr && ne01 % 32 == 0;
        bool use_m8 = false;
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256) {
            use_m8 = use_m4 && hp_aligned();
        }
        const size_t m8_grp_bytes = a_k_blks * spacemit_kernels::q8_hp_m8_blk_size;
        const int64_t n_rows_all = (int64_t) n_ids * ids->ne[1];
        auto align64 = [](uintptr_t p) { return (p + 63) & ~(uintptr_t) 63; };
        const size_t nbw1_m4 = hp_aligned() ? a_k_blks * spacemit_kernels::q8_hp_al_row_blk_size : nbw1;
        uintptr_t m4_ptr     = align64(reinterpret_cast<uintptr_t>(matrix_rows + n_as * ne12));
        if (hp_aligned()) {
            m4_ptr = (m4_ptr + spacemit_kernels::q8_hp_al_align - 1) & ~(uintptr_t) (spacemit_kernels::q8_hp_al_align - 1);
        }
        uint8_t * m4_quant         = reinterpret_cast<uint8_t *>(m4_ptr);
        m4_ptr = align64(m4_ptr + n_rows_all * nbw1_m4 + n_as * (spacemit_kernels::q8_hp_al_align + 8 * nbw1_m4));
        int64_t * m4_expert_offset = reinterpret_cast<int64_t *>(m4_ptr);
        m4_ptr                     = align64(m4_ptr + n_as * sizeof(int64_t));
        int64_t * m4_n_groups      = reinterpret_cast<int64_t *>(m4_ptr);
        int32_t * m4_groups        = reinterpret_cast<int32_t *>(m4_n_groups + 1);
        m4_ptr = align64(m4_ptr + sizeof(int64_t) + (n_rows_all / 4 + 1 + n_as) * 3 * sizeof(int32_t));
        const size_t m4_scratch_sz = GGML_PAD((8 * ne10 + 8 * ne01) * sizeof(float), 64);
        float *      m4_a_f32      = reinterpret_cast<float *>(m4_ptr + ith * m4_scratch_sz);
        float *      m4_c_tmp      = m4_a_f32 + 8 * ne10;
        if (use_m4 && ith == 0) {
            GGML_ASSERT(m4_ptr + nth * m4_scratch_sz <= reinterpret_cast<uintptr_t>(ctx.workspace) + ctx.workspace_size);
        }

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_count_t  = 0;
            int32_t valid_act_count_t = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                valid_matrix_row_counts[valid_ep_count_t] = cur_a;
                valid_act_count_t += cne1;
                valid_ep_count_t += 1;
            }
            valid_ep_count[0]  = valid_ep_count_t;
            valid_act_count[0] = valid_act_count_t;

            if (use_m4) {
                int64_t n_groups = 0;
                int64_t offset   = 0;
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t cne1 = matrix_row_counts[cur_a];
                    offset = (offset + spacemit_kernels::q8_hp_al_align - 1) & ~(int64_t) (spacemit_kernels::q8_hp_al_align - 1);
                    m4_expert_offset[cur_a] = offset;
                    int64_t r = 0;
                    for (; use_m8 && r + 8 <= cne1; r += 8) {
                        m4_groups[3 * n_groups + 0] = cur_a;
                        m4_groups[3 * n_groups + 1] = (int32_t) r;
                        m4_groups[3 * n_groups + 2] = 8;
                        n_groups++;
                        offset += m8_grp_bytes;
                    }
                    if (pad8_all && r < cne1) {
                        m4_groups[3 * n_groups + 0] = cur_a;
                        m4_groups[3 * n_groups + 1] = (int32_t) r;
                        m4_groups[3 * n_groups + 2] = (int32_t) (cne1 - r);
                        n_groups++;
                        offset += m8_grp_bytes;
                        r = cne1;
                    }
                    for (; r + 4 <= cne1; r += 4) {
                        m4_groups[3 * n_groups + 0] = cur_a;
                        m4_groups[3 * n_groups + 1] = (int32_t) r;
                        m4_groups[3 * n_groups + 2] = 4;
                        n_groups++;
                        offset += 4 * nbw1_m4;
                    }
                }
                m4_n_groups[0] = n_groups;
            }

        }

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < static_cast<int64_t>(spine_init_barrier_count));
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ctx.sync();

        if (use_m4 && m4_n_groups[0] > 0) {
            const int64_t n_groups = m4_n_groups[0];
            for (int64_t gi = ith; gi < n_groups; gi += nth) {
                const int     cur_a = m4_groups[3 * gi + 0];
                const int64_t r0    = m4_groups[3 * gi + 1];
                const int     nr    = m4_groups[3 * gi + 2];
                for (int r = 0; r < nr; ++r) {
                    const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, r0 + r);
                    const float * src = (const float *) ((const char *) src1->data + (rm.i1 % ne11) * nb11 + rm.i2 * nb12);
                    memcpy(m4_a_f32 + r * ne10, src, ne10 * sizeof(float));
                }
                const int64_t n8  = use_m8 ? std::min<int64_t>(r0, (matrix_row_counts[cur_a] / 8) * 8) / 8 : 0;
                uint8_t *     out = m4_quant + m4_expert_offset[cur_a] + n8 * m8_grp_bytes + (r0 - n8 * 8) * nbw1_m4;
                if (nr < 8 && pad8_all) {
                    memset(m4_a_f32 + nr * ne10, 0, (8 - nr) * ne10 * sizeof(float));
                    spacemit_kernels::rvv::quantize_a_i8_r8_hp(a_blk_len, m4_a_f32, ne10, out);
                } else if (nr == 8) {
                    spacemit_kernels::rvv::quantize_a_i8_r8_hp(a_blk_len, m4_a_f32, ne10, out);
                } else {
                    quantize_a_i8_r4(a_blk_len, m4_a_f32, ne10, out);
                }
            }
            ctx.sync();
        }

        const size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const size_t expert_b_stride   = ne01 * row_stride_b;
        const size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<const uint8_t *, 2> src_workspaces;
        std::array<float *, 2>         dst_workspaces;

        const int64_t tcm_buffer_size = (int64_t) ctx.shared.size;
        void *        tcm_buffer      = ctx.shared.data;

        const auto valid_ep_count_t  = valid_ep_count[0];
        const auto valid_act_count_t = valid_act_count[0];

        int nth_es = 1;
        int nth_n  = nth;

        int ith_es = ith % nth_es;
        int ith_n  = (ith / nth_es) % nth_n;

        if (!pad8_all && valid_ep_count_t % nth == 0 && tcm_buffer != nullptr && valid_ep_count_t == n_as &&
            valid_act_count_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

            for (int64_t valid_id = ith; valid_id < valid_ep_count_t; valid_id += nth) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                const int        id          = row_mapping.i1;
                const int64_t    i11         = id % ne11;
                const int64_t    i12         = row_mapping.i2;
                const int64_t    i1          = id;
                const int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                float * c_blk    = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                if ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                if (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                if (has_pair) {
                    spine_barrier_wait(cur_barrier);
                }

                if (ith % 2 != 0) {
                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                for (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    if (has_pair && ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(spacemit_kernels::a_pack::r1, b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1,
                                nb_real, b_k_blks, ne01);

                    if (has_pair && ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    const int64_t next_ni = ni + NB_COLS;
                    if (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } else {
            for (int64_t valid_id = ith_es; valid_id < valid_ep_count_t; valid_id += nth_es) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];
                const int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                if (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    continue;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                void * extra_tcm_buffer      = tcm_buffer;

                if (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                int ir1 = src1_cur_start;

                const int64_t n_cols = src0_cur_end - src0_cur_start;
                if (pad8_all) {
                    GGML_ASSERT(use_m8 && n_cols % 32 == 0);
                }
                if (use_m4 && n_cols % 32 == 0 && (cne1 >= 4 || pad8_all)) {
                    const uint8_t * a_grp = m4_quant + m4_expert_offset[cur_a];
                    for (; use_m8 && ir1 + 8 <= src1_cur_end; ir1 += 8, a_grp += m8_grp_bytes) {
                        gemm_kernel(a_pack_r8, b_blk_len, a_grp, src0_cur, nullptr, m4_c_tmp, 8, n_cols, b_k_blks,
                                    n_cols);
                        for (int r = 0; r < 8; ++r) {
                            const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + r);
                            float * dst_row = (float *) ((char *) dst->data + (rm.i1 * nb1 + rm.i2 * nb2)) + src0_cur_start;
                            memcpy(dst_row, m4_c_tmp + r * n_cols, n_cols * sizeof(float));
                        }
                    }
                    if (pad8_all && ir1 < src1_cur_end) {
                        const int nr = (int) (src1_cur_end - ir1);
                        gemm_kernel(a_pack_r8, b_blk_len, a_grp, src0_cur, nullptr, m4_c_tmp, 8, n_cols, b_k_blks,
                                    n_cols);
                        for (int r = 0; r < nr; ++r) {
                            const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + r);
                            float * dst_row = (float *) ((char *) dst->data + (rm.i1 * nb1 + rm.i2 * nb2)) + src0_cur_start;
                            memcpy(dst_row, m4_c_tmp + r * n_cols, n_cols * sizeof(float));
                        }
                        ir1 = src1_cur_end;
                    }
                    for (; ir1 + 4 <= src1_cur_end; ir1 += 4, a_grp += 4 * nbw1_m4) {
                        gemm_kernel(a_pack_r4, b_blk_len, a_grp, src0_cur, b_col_zp, m4_c_tmp, 4, n_cols, b_k_blks,
                                    n_cols);
                        for (int r = 0; r < 4; ++r) {
                            const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + r);
                            float * dst_row = (float *) ((char *) dst->data + (rm.i1 * nb1 + rm.i2 * nb2)) + src0_cur_start;
                            memcpy(dst_row, m4_c_tmp + r * n_cols, n_cols * sizeof(float));
                        }
                    }
                }

                if (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    do {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        int iir1 = ir1;
                        for (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            const int id = row_mapping.i1;  // selected expert index

                            const int64_t i11 = id % ne11;
                            const int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        if (gemm_rows_kernel_m2 != nullptr) {
                            for (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (float *) ((char *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                gemm_rows_kernel_m2(spacemit_kernels::a_pack::r1, b_blk_len, src_workspaces.data(),
                                                   src0_cur, b_col_zp, dst_workspaces.data(), 1,
                                                   src0_cur_end - src0_cur_start, b_k_blks, ne01);
                            }
                        }

                        for (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            float * y = (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                        src0_cur_start;
                            gemm_kernel(spacemit_kernels::a_pack::r1, b_blk_len, quant_a_tile_buffer, src0_cur,
                                        b_col_zp, y, 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } while (ir1 < src1_cur_end);
                } else {
                    if (gemm_rows_kernel_m2 != nullptr) {
                        for (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            for (int iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                const int id = row_mapping.i1;  // selected expert index

                                const int64_t i11 = id % ne11;
                                const int64_t i12 = row_mapping.i2;  // row index in src1

                                const int64_t i1 = id;               // selected expert index
                                const int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            gemm_rows_kernel_m2(spacemit_kernels::a_pack::r1, b_blk_len, src_workspaces.data(),
                                               src0_cur, b_col_zp, dst_workspaces.data(), 1,
                                               src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    for (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        const int id = row_mapping.i1;  // selected expert index

                        const int64_t i11 = id % ne11;
                        const int64_t i12 = row_mapping.i2;  // row index in src1

                        const int64_t i1 = id;               // selected expert index
                        const int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        float * y = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                        gemm_kernel(spacemit_kernels::a_pack::r1, b_blk_len, src1_col, src0_cur, b_col_zp, y, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }

#undef MMID_MATRIX_ROW
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

class tensor_traits_common : public ggml::spacemit::tensor_traits_base {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32) {
                    const auto * traits = ggml_get_type_traits_cpu(op->src[0]->type);
                    if (traits && traits->vec_dot) {
                    const size_t row_size = ggml_row_size(traits->vec_dot_type, op->src[1]->ne[0]);
                    const size_t bytes = row_size * op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3];
                        if (op->src[1]->type != traits->vec_dot_type) {
                            size = bytes;
                            return true;
                        }
                    }
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                {
                    const int     n_tasks = n_threads;
                    const int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    const int64_t DK      = op->src[1]->ne[0];
                    const int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = sizeof(float) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK +
                                      ggml::spacemit::cache_line_size_f32) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = sizeof(float) * n_tasks * (DK + 2 * DV + 64);
                    // split-KV decode partials (up to 4 query tokens, FA_SPLIT_KV_MAX_Q)
                    const size_t n_q = (size_t) std::min<int64_t>(op->src[0]->ne[1], 4);
                    decode = MAX(decode, sizeof(float) * (n_q * neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV)));

                    size = MAX(prefill, decode);
                }
                return true;
            case GGML_OP_ROPE:
                {
                    // forward_rope_impl caches sin/cos per thread unless the
                    // small-F32-NEOX legacy path (stack cache) is taken
                    const int  mode    = ggml_get_op_params_i32(op, 2);
                    const bool legacy  = op->src[0]->type == GGML_TYPE_F32 &&
                                        mode == GGML_ROPE_TYPE_NEOX && op->ne[0] <= 128 &&
                                        op->src[0]->nb[0] == sizeof(float);
                    size = legacy ? 0
                                  : (size_t) n_threads *
                                        (op->ne[0] + ggml::spacemit::cache_line_size_f32) * sizeof(float);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->src[0]->nb[0] == sizeof(_Float16) &&
                    op->src[1]->nb[0] == sizeof(float)) {
                    spacemit_kernels::rvv::forward_mul_mat_f16_f32(ctx, op);
                    return true;
                }
                if (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot != nullptr) {
                    spacemit_kernels::scalar::forward_mul_mat(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::scalar::forward_norm_f16(ctx, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_RMS_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::scalar::forward_rms_norm_f16(ctx, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_ADD:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SUB:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_MUL:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_DIV:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SCALE:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    spacemit_kernels::rvv::forward_scale_f32(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(op)) {
                    case GGML_UNARY_OP_TANH:
                        spacemit_kernels::rvv::forward_unary_tanh_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_GELU:
                        spacemit_kernels::rvv::forward_unary_gelu_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SILU:
                        spacemit_kernels::rvv::forward_unary_silu_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_EXP:
                        spacemit_kernels::rvv::forward_unary_exp_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SIGMOID:
                        spacemit_kernels::rvv::forward_unary_sigmoid_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_NEG:
                        spacemit_kernels::rvv::forward_unary_neg_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SOFTPLUS:
                        spacemit_kernels::rvv::forward_unary_softplus_f32(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SOFT_MAX:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    spacemit_kernels::rvv::forward_soft_max_f32(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_GLU:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    switch (ggml_get_glu_op(op)) {
                        case GGML_GLU_OP_GEGLU:
                            spacemit_kernels::rvv::forward_glu_geglu_f32(ctx, op);
                            return true;
                        case GGML_GLU_OP_SWIGLU:
                            spacemit_kernels::rvv::forward_glu_swiglu_f32(ctx, op);
                            return true;
                        default:
                            break;
                    }
                }
                return false;
            case GGML_OP_FLASH_ATTN_EXT:
                // Use the RVV implementation for its supported F32-Q/F16-KV
                // shapes; retain the CPU-equivalent implementation for all
                // other layouts and dtypes.
                if (forward_flash_attn_ext_f16(ctx, op)) {
                    return true;
                }
                spacemit_kernels::scalar::forward_flash_attn_ext(ctx, op);
                return true;
            case GGML_OP_ROPE:
                spacemit_kernels::rvv::forward_rope(ctx, op);
                return true;
            case GGML_OP_CONT:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(ctx, op);
                        return true;
                    }
                    if (op->type == src0->type && ggml_is_contiguous(op) &&
                        ggml_nelements(op) == ggml_nelements(src0) &&
                        (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16)) {
                        if (op->type == GGML_TYPE_F32) {
                            spacemit_kernels::rvv::forward_cont_general<float>(ctx, op);
                        } else {
                            spacemit_kernels::rvv::forward_cont_general<_Float16>(ctx, op);
                        }
                        return true;
                    }
                    return false;
                }
            case GGML_OP_CPY:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (ggml_nelements(op) == 0) {
                        return true;
                    }
                    if (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(ctx, op);
                        return true;
                    }
                    if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_strided_f32(ctx, op);
                        return true;
                    }
                    // row-contiguous F32 copy with arbitrary outer strides (GDN rollback snapshots)
                    if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                        op->nb[0] == sizeof(float) && src0->nb[0] == sizeof(float) &&
                        op->ne[0] == src0->ne[0] && op->ne[1] == src0->ne[1] &&
                        op->ne[2] == src0->ne[2] && op->ne[3] == src0->ne[3]) {
                        spacemit_kernels::rvv::forward_cpy_f32_row_strided(ctx, op);
                        return true;
                    }
                    if (ggml_is_contiguous(op) && ggml_is_contiguous(src0) &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        if (op->type == GGML_TYPE_F16 && src0->type == GGML_TYPE_F32) {
                            spacemit_kernels::rvv::forward_cpy_f32_to_f16(ctx, op);
                            return true;
                        }
                        if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F16) {
                            spacemit_kernels::rvv::forward_cpy_f16_to_f32(ctx, op);
                            return true;
                        }
                    }
                    return false;
                }
            case GGML_OP_SET_ROWS:
                spacemit_kernels::rvv::forward_set_rows(ctx, op);
                return true;
            case GGML_OP_REPEAT:
                {
                    const bool rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    const bool broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    if (rows_equal && broadcast_or_equal) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    if (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            case GGML_OP_SUM_ROWS:
                {
                    if (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<float>(ctx, op);
                    } else {
                        return false;
                    }
                }
                return true;
            case GGML_OP_GET_ROWS:
                {
                    if (op->src[0]->type == op->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            case GGML_OP_CONCAT:
                {
                    if (spacemit_kernels::rvv::forward_concat_dim0_transposed_f32(ctx, op, 0)) {
                        return true;
                    }
                    const int32_t dim = ggml_get_op_params_i32(op, 0);
                    if (dim == 0 && op->type == op->src[0]->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            // Scalar ops dispatched to scalar_kernels
            case GGML_OP_L2_NORM:
                spacemit_kernels::scalar::forward_l2_norm_f32(ctx, op);
                return true;
            case GGML_OP_FILL:
                spacemit_kernels::scalar::forward_fill_f32(ctx, op);
                return true;
            case GGML_OP_CUMSUM:
                spacemit_kernels::scalar::forward_cumsum_f32(ctx, op);
                return true;
            case GGML_OP_PAD:
                spacemit_kernels::scalar::forward_pad_f32(ctx, op);
                return true;
            case GGML_OP_TRI:
                spacemit_kernels::scalar::forward_tri_f32(ctx, op);
                return true;
            case GGML_OP_DIAG:
                spacemit_kernels::scalar::forward_diag_f32(ctx, op);
                return true;
            case GGML_OP_SET:
                spacemit_kernels::scalar::forward_set_f32(ctx, op);
                return true;
            case GGML_OP_SOLVE_TRI:
                spacemit_kernels::scalar::forward_solve_tri_f32(ctx, op);
                return true;
            case GGML_OP_GATED_DELTA_NET:
                spacemit_kernels::rvv::forward_gated_delta_net(ctx, op);
                return true;
            case GGML_OP_ARGSORT:
                spacemit_kernels::scalar::forward_argsort_f32(ctx, op);
                return true;
            case GGML_OP_CLAMP:
                spacemit_kernels::scalar::forward_clamp_f32(ctx, op);
                return true;
            case GGML_OP_SSM_CONV:
                if (!spacemit_kernels::rvv::forward_ssm_conv_f32(ctx, op)) {
                    spacemit_kernels::scalar::forward_ssm_conv_f32(ctx, op);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward_fused(ggml::spacemit::context & ctx, const spacemit_opnode & node) const override {
        switch (node.opcode) {
            case SPACEMIT_OP_RMS_NORM_MUL:
                // out = rms_norm(x) * w in one pass; node.node = RMS_NORM, node.fused[0] = MUL
                spacemit_kernels::rvv::forward_rms_norm_mul_f32(ctx, node.node, node.fused[0]);
                return true;
            default:
                return false;
        }
    }

    bool forward_flash_attn_ext_f16(ggml::spacemit::context & ctx, ggml_tensor * dst) const {
        const ggml_tensor * q = dst->src[0];
        const ggml_tensor * k = dst->src[1];
        const ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        const int64_t DK = nek0;
        const int64_t DV = nev0;

        const bool supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        const bool supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        const bool supported_shape = (DK > 0 && DK <= 256 && DV > 0 && DV <= 256);
        const bool supported_vlen  = (__riscv_vlenb() == 128);

        if (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            return false;
        }

        // total rows in q
        const int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        const int ith = ctx.ith;
        const int nth = ctx.nth;

        static constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        const bool               use_tiled = neq1 >= Q_TILE_SZ;

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        // decode / MTP verification (1-4 query tokens): split the KV range across threads so each K/V row is
        // read once, then merge the partials (ported from ggml-cpu/spacemit)
        const int n_split = (!use_tiled && ctx.shared.data != nullptr) ?
                                spacemit_kernels::rvv::flash_attn_ext_f16_split_kv_chunks_vlen1024_vf16(dst, nth) :
                                0;
        if (n_split > 0) {
            float * partials = (float *) ctx.workspace;
            spacemit_kernels::rvv::forward_flash_attn_ext_f16_split_kv_partial_vlen1024_vf16(
                ctx, dst, n_split, partials, ctx.shared.data, ctx.shared.size);
            ctx.sync();
            spacemit_kernels::rvv::forward_flash_attn_ext_f16_split_kv_reduce_vlen1024_vf16(ctx, dst, n_split, partials);
            return true;
        }

        int64_t nchunk = nth;

        // The number of elements in each chunk
        const int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk) {
            const int64_t ir0 = dr * current_chunk;
            const int64_t ir1 = MIN(ir0 + dr, nr);

            if (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    ctx, dst, ir0, ir1);
            } else {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    ctx, dst, ir0, ir1);
            }

            current_chunk += nth;
        }

        return true;
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};

// Dense 2-D Q4 uses Q41Block/F16; MoE experts retain the integer path.
static bool use_f16_link(const ggml_tensor * cur) {
#if defined(RISCV64_SPACEMIT_IME2)
    if (!global_spine_env_info.use_ime2 || cur->ne[2] != 1 || cur->ne[3] != 1 || cur->ne[1] % 32 != 0 ||
        cur->ne[0] % 32 != 0) {
        return false;
    }
    return cur->type == GGML_TYPE_Q4_0 || cur->type == GGML_TYPE_Q4_1 || cur->type == GGML_TYPE_Q4_K;
#else
    GGML_UNUSED(cur);
    return false;
#endif
}

constexpr size_t Q4_0_F16_SPLIT_K_MIN     = 2048;
constexpr size_t Q4_0_F16_SPLIT_N_M_MAX   = 256;

static bool q4_0_f16_split_n(const size_t K, const size_t M) {
    return K < Q4_0_F16_SPLIT_K_MIN && M <= Q4_0_F16_SPLIT_N_M_MAX;
}

// Q4_1/Q4_K retain nibbles but round offsets to uint8 zero points; the 608B mapping is lossy.
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, q41_zero_point_mode Mode>
class tensor_traits_f16 : public ggml::spacemit::tensor_traits_base {
    static_assert(std::is_same_v<BLOC_TYPE, Q41Block<32, 32, 2>>, "f16 traits expect a Q41Block weight layout");
    const gemm_path_f16 ops_;

public:
    tensor_traits_f16() : ops_(resolve_gemm_f16<BLOC_TYPE, NB_COLS, INTER_SIZE, Mode>()) {}

private:
    static constexpr size_t F16_M_TILE = 16;
    static constexpr size_t F16_N_TILE = 32;

    static constexpr size_t KB = (size_t) INTER_SIZE;   // K elements per block
    static constexpr size_t NB = (size_t) NB_COLS;      // rows per block
    static constexpr size_t KB_BYTES   = sizeof(BLOC_TYPE);
    static constexpr size_t ROW_PITCH  = KB_BYTES / NB;  // bytes per row per block

    static size_t partial_workspace_size(size_t M, size_t N, int n_threads) {
        GGML_ASSERT(n_threads > 0);
        const size_t m_pad = GGML_PAD(M, F16_M_TILE);
        const size_t n_pad = GGML_PAD(N, F16_N_TILE);
        if (m_pad == 0 || n_pad == 0) {
            return 0;
        }
        // Include alignment and the largest cluster base-pointer table.
        constexpr size_t metadata_size = 64 + 16 * sizeof(uint64_t);
        const size_t slices = static_cast<size_t>(n_threads);
        GGML_ASSERT(m_pad <= (SIZE_MAX - metadata_size) / n_pad / sizeof(_Float16) / slices);
        return slices * m_pad * n_pad * sizeof(_Float16) + metadata_size;
    }

    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        const size_t N         = static_cast<size_t>(op->src[0]->ne[1]);
        const size_t M         = static_cast<size_t>(op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3]);
        const size_t K         = static_cast<size_t>(op->src[1]->ne[0]);
        const size_t gemv_size = ((K + KB - 1) / KB) * 38;
        size = std::max(partial_workspace_size(M, N, n_threads), GGML_PAD(gemv_size, sizeof(int64_t)));
        return true;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        const size_t K             = static_cast<size_t>(src1->ne[0]);
        const size_t M             = static_cast<size_t>(src1->ne[1] * src1->ne[2] * src1->ne[3]);

        const size_t N             = static_cast<size_t>(src0->ne[1]);
        const size_t a_src_stride  = src1->nb[1];  // bytes
        const size_t a_src_stride0 = src1->nb[0];  // bytes
        GGML_ASSERT(a_src_stride0 == sizeof(float));
        GGML_ASSERT(dst->nb[0] == sizeof(float));

        const int ith = ctx.ith;
        const int nth = ctx.nth;

        if (ith == 0 && getenv("GGML_SPACEMIT_Q41_TRACE") != nullptr) {
            fprintf(stderr, "q41_mul_mat: N=%zu M=%zu K=%zu (%s)\n", N, M, K, M == 1 ? "decode" : "prefill");
        }

        const size_t out_row_stride = dst->nb[1];  // bytes

        constexpr size_t m_tile = F16_M_TILE;
        constexpr size_t n_tile = F16_N_TILE;
        const size_t     mt_count = (M + m_tile - 1) / m_tile;
        const size_t     nt_count = (N + n_tile - 1) / n_tile;
        const size_t partial_workspace = ctx.workspace_size;

        void *      tcm_buffer      = ctx.shared.data;
        const int64_t tcm_buffer_size = ctx.shared.size;
        if (tcm_buffer == nullptr || tcm_buffer_size <= 0) {
            GGML_ABORT("q4_0_f16: no shared TCM buffer");
        }
        const size_t tcm_size = static_cast<size_t>(tcm_buffer_size);

        const gemm_path_f16 & ops = ops_;
        const auto & pack_func   = ops.castpack_a;
        const auto & q41_func    = ops.dequantpack_b;
        const auto   gemv_func   = ops.dot_gemv;
        const auto & reduce_func = ops.reducesum;

        const auto * a_src = reinterpret_cast<const uint8_t *>(src1->data);
        const auto * b_src = reinterpret_cast<const uint8_t *>(src0->data);
        auto *       out   = reinterpret_cast<uint8_t *>(dst->data);

        // Decode quantizes A once; workers compute separate N32 tiles without TCM staging.
        if (M == 1 && K % KB == 0) {
            constexpr size_t a_blk_len      = 32;
            const size_t     a_blk_stride   = spacemit_kernels::q8_blk_size(a_blk_len, true);
            const size_t     a_k_blks       = K / KB;
            const size_t     k_blks         = K / KB;
            const size_t     gemv_ws_needed = a_k_blks * a_blk_stride;
            if (ctx.workspace_size >= gemv_ws_needed) {
                auto * quant_a = reinterpret_cast<uint8_t *>(ctx.workspace);

                int task_per_thread = spacemit_kernels::div_round_up((int) a_k_blks, nth);
                int a_blk_start     = ith * task_per_thread;
                int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
                if (a_blk_start < a_blk_end) {
                    ops.quantize_a_gemv(
                        a_blk_len, reinterpret_cast<const float *>(a_src) + a_blk_start * a_blk_len,
                        (a_blk_end - a_blk_start) * a_blk_len, quant_a + a_blk_start * a_blk_stride);
                }

                ctx.sync();

                const size_t b_row_stride = k_blks * ROW_PITCH;
                constexpr size_t tile_cols = 32;
                for (int64_t ni = (int64_t) ith * tile_cols; ni < (int64_t) N;
                     ni += (int64_t) tile_cols * nth) {
                    const int64_t nb_real = std::min((int64_t) N - ni, (int64_t) tile_cols);
                    const uint8_t * b_row = b_src + ni * b_row_stride;
                    auto *          c_row = reinterpret_cast<float *>(out) + ni;
                    gemv_func(quant_a, b_row, c_row, (size_t) nb_real, k_blks);
                }
                return true;
            }
        }

        if (q4_0_f16_split_n(K, M) && K % KB == 0) {
            // Keep all padded A rows in TCM and reuse them across this worker's N tiles.
            const size_t m_pad  = (M + m_tile - 1) & ~size_t{ m_tile - 1 };
            const size_t kb_res = (tcm_size / ((m_pad + n_tile) * sizeof(_Float16))) & ~size_t{ 31 };
            const size_t kb_a   = std::min(K, kb_res);
            if (kb_a >= 32 && (m_pad + n_tile) * kb_a * sizeof(_Float16) <= tcm_size) {
                auto * a_tile = reinterpret_cast<_Float16 *>(tcm_buffer);
                auto * b_tile = a_tile + m_pad * kb_a;
                const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
                const size_t m_tiles = (M + m_tile - 1) / m_tile;
                for (size_t k0 = 0; k0 < K; k0 += kb_a) {
                    const size_t kk = std::min(kb_a, K - k0);
                    for (size_t mt = 0; mt < m_tiles; ++mt) {
                        const size_t tm0 = mt * m_tile;
                        pack_func((void *) (a_src + tm0 * a_src_stride),
                                  static_cast<int64_t>(k0 * sizeof(float)),
                                  static_cast<int64_t>(a_src_stride),
                                  (void *) (a_tile + mt * m_tile * kb_a), 0,
                                  static_cast<int64_t>(kb_a * m_tile * sizeof(_Float16)),
                                  static_cast<int64_t>(std::min(m_tile, M - tm0)),
                                  static_cast<int64_t>(kk),
                                  1, static_cast<int64_t>(kk / 8));
                    }
                    for (size_t nt_i = (size_t) ith; nt_i < nt_count; nt_i += (size_t) nth) {
                        const size_t n0 = nt_i * n_tile;
                        const size_t nc = std::min(n_tile, N - n0);
                        q41_func((void *) (b_src + nt_i * b_row_stride + (k0 / KB) * KB_BYTES),
                                 0,
                                 static_cast<int64_t>(b_row_stride),
                                 (void *) b_tile, 0,
                                 static_cast<int64_t>(kk / 8) * 512,
                                 1, static_cast<int64_t>(kk / KB),
                                 1, static_cast<int64_t>(kk / 8));
                        for (size_t mt = 0; mt < m_tiles; ++mt) {
                            const size_t tm0    = mt * m_tile;
                            float *     c_tile = reinterpret_cast<float *>(out + tm0 * out_row_stride) + n0;
                            ops.dot_f32acc(
                                kk, b_tile, a_tile + mt * m_tile * kb_a, c_tile,
                                std::min(m_tile, M - tm0), nc, N, k0 != 0);
                        }
                    }
                }
                return true;
            }
            // If A cannot fit, keep one B tile resident and stream M groups through the A slot.
            const size_t bytes_per_k  = (m_tile + n_tile) * sizeof(_Float16);
            const size_t kb           = std::min(K, (tcm_size / bytes_per_k) & ~size_t{ 31 });
            const size_t b_tile_bytes = n_tile * kb * sizeof(_Float16);
            const size_t a_budget     = tcm_size > b_tile_bytes ? tcm_size - b_tile_bytes : 0;
            const size_t m_fit        = kb >= 32 ? (a_budget / (kb * sizeof(_Float16))) & ~size_t{ 15 } : 0;
            if (m_fit >= m_tile) {
                auto * a_tile = reinterpret_cast<_Float16 *>(tcm_buffer);
                auto * b_tile = a_tile + m_fit * kb;
                const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
                for (size_t nt_i = (size_t) ith; nt_i < nt_count; nt_i += (size_t) nth) {
                    const size_t n0 = nt_i * n_tile;
                    const size_t nc = std::min(n_tile, N - n0);
                    for (size_t k0 = 0; k0 < K; k0 += kb) {
                        const size_t kk = std::min(kb, K - k0);
                        q41_func((void *) (b_src + nt_i * b_row_stride + (k0 / KB) * KB_BYTES),
                                 0,
                                 static_cast<int64_t>(b_row_stride),
                                 (void *) b_tile, 0,
                                 static_cast<int64_t>(kk / 8) * 512,
                                 1, static_cast<int64_t>(kk / KB),
                                 1, static_cast<int64_t>(kk / 8));
                        for (size_t m0 = 0; m0 < M; m0 += m_fit) {
                            const size_t m_count = std::min(m_fit, M - m0);
                            const size_t m_tiles = (m_count + m_tile - 1) / m_tile;
                            for (size_t mt = 0; mt < m_tiles; ++mt) {
                                const size_t tm0 = m0 + mt * m_tile;
                                pack_func((void *) (a_src + tm0 * a_src_stride),
                                          static_cast<int64_t>(k0 * sizeof(float)),
                                          static_cast<int64_t>(a_src_stride),
                                          (void *) (a_tile + mt * m_tile * kb), 0,
                                          static_cast<int64_t>(kb * m_tile * sizeof(_Float16)),
                                          static_cast<int64_t>(std::min(m_tile, M - tm0)),
                                          static_cast<int64_t>(kk),
                                          1, static_cast<int64_t>(kk / 8));
                                float * c_tile = reinterpret_cast<float *>(out + tm0 * out_row_stride) + n0;
                                ops.dot_f32acc(
                                    kk, b_tile, a_tile + mt * m_tile * kb, c_tile,
                                    std::min(m_tile, M - tm0), nc, N, k0 != 0);
                            }
                        }
                    }
                }
                return true;
            }
        }

        // Split M across workers; accumulate K chunks directly into F32 output.
        if (K < Q4_0_F16_SPLIT_K_MIN && M > Q4_0_F16_SPLIT_N_M_MAX && K % KB == 0) {
            const size_t bytes_per_k  = (m_tile + n_tile) * sizeof(_Float16);
            const size_t kb           = std::min(K, (tcm_size / bytes_per_k) & ~size_t{ 31 });
            const size_t b_tile_bytes = n_tile * kb * sizeof(_Float16);
            const size_t a_budget     = tcm_size > b_tile_bytes ? tcm_size - b_tile_bytes : 0;
            const size_t m_fit        = kb >= 32 ? (a_budget / (kb * sizeof(_Float16))) & ~size_t{ 15 } : 0;
            if (m_fit >= m_tile) {
                size_t mb_count = (M + m_fit - 1) / m_fit;
                if (mb_count < (size_t) nth) {
                    mb_count = (size_t) nth;
                }
                size_t m_block = 0;
                do {
                    // M16 padding must not create an empty final block.
                    m_block = (((M + mb_count - 1) / mb_count) + 15) & ~size_t{ 15 };
                    if ((mb_count - 1) * m_block < M) {
                        break;
                    }
                    --mb_count;
                } while (mb_count > 1);
                if (m_block <= m_fit && m_block >= m_tile) {
                    auto * a_tile = reinterpret_cast<_Float16 *>(tcm_buffer);
                    auto * b_tile = a_tile + m_block * kb;
                    const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
                    for (size_t mb_i = (size_t) ith; mb_i < mb_count; mb_i += (size_t) nth) {
                        const size_t mb      = mb_i * m_block;
                        const size_t m_count = std::min(m_block, M - mb);
                        const size_t m_tiles = (m_count + m_tile - 1) / m_tile;
                        for (size_t k0 = 0; k0 < K; k0 += kb) {
                            const size_t kk = std::min(kb, K - k0);
                            for (size_t mt = 0; mt < m_tiles; ++mt) {
                                const size_t tm0 = mb + mt * m_tile;
                                pack_func((void *) (a_src + tm0 * a_src_stride),
                                          static_cast<int64_t>(k0 * sizeof(float)),
                                          static_cast<int64_t>(a_src_stride),
                                          (void *) (a_tile + mt * m_tile * kb), 0,
                                          static_cast<int64_t>(kb * m_tile * sizeof(_Float16)),
                                          static_cast<int64_t>(std::min(m_tile, M - tm0)),
                                          static_cast<int64_t>(kk),
                                          1, static_cast<int64_t>(kk / 8));
                            }
                            for (size_t nt = 0; nt < nt_count; ++nt) {
                                const size_t n0 = nt * n_tile;
                                const size_t nc = std::min(n_tile, N - n0);
                                q41_func((void *) (b_src + nt * b_row_stride + (k0 / KB) * KB_BYTES),
                                         0,
                                         static_cast<int64_t>(b_row_stride),
                                         (void *) b_tile, 0,
                                         static_cast<int64_t>(kk / 8) * 512,
                                         1, static_cast<int64_t>(kk / KB),
                                         1, static_cast<int64_t>(kk / 8));
                                for (size_t mt = 0; mt < m_tiles; ++mt) {
                                    const size_t tm0   = mb + mt * m_tile;
                                    float *       c_tile = reinterpret_cast<float *>(out + tm0 * out_row_stride) + n0;
                                    ops.dot_f32acc(
                                        kk, b_tile, a_tile + mt * m_tile * kb, c_tile,
                                        std::min(m_tile, M - tm0), nc, N, k0 != 0);
                                }
                            }
                        }
                    }
                    return true;
                }
            }
        }

        // Each two-core cluster shares a K slice and reads both B tiles from the paired TCM blocks.
        if ((nth % 2) == 0 && nth <= 16 && K % KB == 0) {
            const size_t g  = (size_t) nth / 2;
            const size_t ks = K / g;
            const size_t b_slot = n_tile * ks;                    // f16 per B slot
            const size_t n_chunk = n_tile * 2;                    // 64 columns per cluster chunk
            const size_t chunks  = (N + n_chunk - 1) / n_chunk;
            const auto m_fit_for = [&](size_t slots) {
                const size_t b_bytes  = slots * b_slot * sizeof(_Float16);
                const size_t a_budget = tcm_size > b_bytes + 64 ? tcm_size - b_bytes - 64 : 0;
                return (a_budget / (ks * sizeof(_Float16))) & ~size_t{ 15 };
            };
            const auto n_passes = [&](size_t m_fit) {
                const size_t m_cap = m_fit * 2;
                return (M + m_cap - 1) / m_cap;
            };
            const size_t m_fit1 = m_fit_for(1);
            const size_t m_fit2 = m_fit_for(2);
            const size_t slots =
                chunks > 1 && m_fit2 >= m_tile && n_passes(m_fit2) == n_passes(m_fit1) ? 2 : 1;
            const size_t m_fit = slots == 2 ? m_fit2 : m_fit1;
            // Prefer split-K only when the cluster A budget is small and split-K needs fewer M passes.
            bool route_split_k = false;
            if (m_fit >= m_tile && m_fit1 < 128 && K % (KB * (size_t) nth) == 0) {
                const size_t ksg  = (size_t) K / (size_t) nth;
                const size_t bg   = n_tile * ksg * sizeof(_Float16);
                const size_t mfg  = tcm_size > bg
                                        ? ((tcm_size - bg) / (ksg * sizeof(_Float16))) & ~size_t{ 15 }
                                        : 0;
                if (mfg >= m_tile) {
                    const size_t mbg = std::min(std::max(M, m_tile), mfg);
                    route_split_k    = (M + mbg - 1) / mbg < n_passes(m_fit);
                }
            }
            // Per-thread TCM: [m_fit x ks A][slots x 32 x ks B][64B flags].
            const size_t slice_stride = mt_count * nt_count * m_tile * n_tile;
            const size_t metadata_size = 64 + static_cast<size_t>(nth) * sizeof(uint64_t);
            if (!route_split_k && K % (KB * g) == 0 && m_fit >= m_tile &&
                partial_workspace >= metadata_size &&
                g <= (partial_workspace - metadata_size) / sizeof(_Float16) / slice_stride) {
                const size_t group    = (size_t) ith / 2;
                const size_t intra    = (size_t) ith % 2;
                const size_t sib      = (size_t) ith ^ 1;
                const size_t slice_k0 = group * ks;
                auto * partials_h = reinterpret_cast<_Float16 *>(
                    (reinterpret_cast<uintptr_t>(ctx.workspace) + 63) & ~uintptr_t{ 63 });
                uint64_t * bases = reinterpret_cast<uint64_t *>(partials_h + g * slice_stride);
                auto * a_blk = reinterpret_cast<_Float16 *>(tcm_buffer);
                auto * b_blk = a_blk + m_fit * ks;             // slot 0; slot 1 at +b_slot
                volatile uint64_t * f_ready =
                    reinterpret_cast<volatile uint64_t *>(reinterpret_cast<uint8_t *>(tcm_buffer) + tcm_size - 64);
                volatile uint64_t * f_done = f_ready + 1;

                bases[ith] = (uint64_t) tcm_buffer;
                *f_ready   = 0;
                *f_done    = 0;
                ctx.sync();
                const auto * sib_base = reinterpret_cast<const uint8_t *>(bases[sib]);
                const auto * sib_b_blk =
                    reinterpret_cast<const _Float16 *>(sib_base + m_fit * ks * sizeof(_Float16));
                volatile uint64_t * sib_ready =
                    reinterpret_cast<volatile uint64_t *>(const_cast<uint8_t *>(sib_base) + tcm_size - 64);
                volatile uint64_t * sib_done = sib_ready + 1;

                const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
                const size_t m_cap   = m_fit * 2;

                // Both threads use the same rotated visit order and B-slot index.
                auto dequant_chunk = [&](size_t pos) {
                    const size_t chunk = (pos + group) % chunks;
                    const size_t nt    = (chunk * n_chunk) / n_tile + intra;
                    if (nt >= nt_count) {
                        return;
                    }
                    q41_func((void *) (b_src + nt * b_row_stride +
                                           (slice_k0 / KB) * KB_BYTES),
                             0,
                             static_cast<int64_t>(b_row_stride),
                             (void *) (b_blk + (pos % slots) * b_slot), 0,
                             static_cast<int64_t>((ks / 8) * 512),
                             1, static_cast<int64_t>(ks / KB),
                             1, static_cast<int64_t>(ks / 8));
                };

                uint64_t seq = 0;   // Monotonic across passes.
                for (size_t mb = 0; mb < M; mb += m_cap) {
                    const size_t rows   = std::min(m_cap, M - mb);
                    size_t       r0     = std::min(std::min(((rows + 31) / 32) * 16, rows), m_fit);
                    const size_t row0   = mb + (intra == 0 ? 0 : r0);
                    const size_t rows_t = intra == 0 ? r0 : rows - r0;
                    const size_t m_tiles = (rows_t + m_tile - 1) / m_tile;
                    // Publish B before packing A; the previous pass must have released this slot.
                    dequant_chunk(0);
                    __sync_synchronize();
                    *f_ready = seq + 1;
                    for (size_t mt = 0; mt < m_tiles; ++mt) {
                        const size_t tm0 = row0 + mt * m_tile;
                        pack_func((void *) (a_src + tm0 * a_src_stride),
                                  static_cast<int64_t>(slice_k0 * sizeof(float)),
                                  static_cast<int64_t>(a_src_stride),
                                  (void *) (a_blk + mt * m_tile * ks), 0,
                                  static_cast<int64_t>(ks * m_tile * sizeof(_Float16)),
                                  static_cast<int64_t>(std::min(m_tile, rows_t - mt * m_tile)),
                                  static_cast<int64_t>(ks),
                                  1, static_cast<int64_t>(ks / 8));
                    }
                    for (size_t j = 0; j < chunks; ++j) {
                        const size_t chunk = (j + group) % chunks;
                        const size_t n0    = chunk * n_chunk;
                        const size_t nt    = n0 / n_tile + intra;
                        const size_t nt_s  = n0 / n_tile + (1 - intra); // sibling's tile
                        ++seq;
                        if (nt < nt_count) {
                            const size_t nc = std::min(n_tile, N - nt * n_tile);
                            for (size_t mt = 0; mt < m_tiles; ++mt) {
                                const size_t tm0 = row0 + mt * m_tile;
                                _Float16 *     c_blk =
                                    partials_h + ((tm0 / m_tile) * nt_count + nt) * m_tile * g * n_tile +
                                    group * n_tile;
                                ops.dot_f16acc(
                                    ks, b_blk + (j % slots) * b_slot, a_blk + mt * m_tile * ks, c_blk,
                                    std::min(m_tile, rows_t - mt * m_tile), nc, g * n_tile, false);
                            }
                        }
                        while (*sib_ready < seq) ;
                        __sync_synchronize();
                        if (nt_s < nt_count) {
                            const size_t nc = std::min(n_tile, N - nt_s * n_tile);
                            for (size_t mt = 0; mt < m_tiles; ++mt) {
                                const size_t tm0 = row0 + mt * m_tile;
                                _Float16 *     c_blk =
                                    partials_h + ((tm0 / m_tile) * nt_count + nt_s) * m_tile * g * n_tile +
                                    group * n_tile;
                                ops.dot_f16acc(
                                    ks, sib_b_blk + (j % slots) * b_slot, a_blk + mt * m_tile * ks, c_blk,
                                    std::min(m_tile, rows_t - mt * m_tile), nc, g * n_tile, false);
                            }
                        }
                        __sync_synchronize();
                        *f_done = seq;
                        // Refill only after the sibling releases the slot: seq for one slot, seq-1 for two.
                        if (j + 1 < chunks) {
                            while (*sib_done < seq - (slots - 1)) ;
                            dequant_chunk(j + 1);
                            __sync_synchronize();
                            *f_ready = seq + 1;
                        }
                    }
                    // The next pass overwrites slot 0; wait for its final reader.
                    while (*sib_done < seq) ;
                }

                ctx.sync();
                constexpr size_t reduce_chunk = 4;  // tiles per chunk
                const size_t     tile_count   = mt_count * nt_count;
                const size_t     n_rchunks    = (tile_count + reduce_chunk - 1) / reduce_chunk;
                for (size_t c = (size_t) ith; c < n_rchunks; c += (size_t) nth) {
                    const size_t t0 = c * reduce_chunk;
                    const size_t t1 = std::min(tile_count, t0 + reduce_chunk);
                    for (size_t ti = t0; ti < t1; ++ti) {
                        const size_t     mt      = ti / nt_count;
                        const size_t     nt      = ti % nt_count;
                        const _Float16 * tile    = partials_h + ti * m_tile * g * n_tile;
                        const size_t     count_m = std::min(m_tile, M - mt * m_tile);
                        const size_t     count_n = std::min(n_tile, N - nt * n_tile);
                        auto *           out_tile =
                            reinterpret_cast<float *>(out + (mt * m_tile) * out_row_stride) + nt * n_tile;
                        reduce_func((void *) tile, 0, 0, 0,
                                    (void *) out_tile, 0,
                                    static_cast<int64_t>(out_row_stride),
                                    1, 1, static_cast<int64_t>(g),
                                    static_cast<int64_t>(count_m), static_cast<int64_t>(count_n));
                    }
                }
                return true;
            }
        }

        // K slices must align to K32, fit padded M16/N32 tiles in TCM and fit partials in workspace.
        size_t s_count = 0;
        size_t ks      = 0;
        for (size_t s = static_cast<size_t>(nth); s >= 1; --s) {
            if (K % (KB * s) != 0) {
                continue;
            }
            const size_t cand_ks  = K / s;
            const size_t b_bytes  = n_tile * cand_ks * sizeof(_Float16);
            const size_t a_budget = tcm_size > b_bytes ? tcm_size - b_bytes : 0;
            const size_t m_fit    = (a_budget / (cand_ks * sizeof(_Float16))) & ~size_t{ 15 };
            if (m_fit < m_tile) {
                continue;
            }
            const size_t slice_stride = mt_count * nt_count * m_tile * n_tile;
            if (partial_workspace < 64 ||
                s > (partial_workspace - 64) / sizeof(_Float16) / slice_stride) {
                continue;
            }
            s_count = s;
            ks      = cand_ks;
            break;
        }
        if (s_count == 0) {
            // Stream K32 chunks into disjoint output tiles without a partials buffer.
            const size_t bytes_per_k = (m_tile + n_tile) * sizeof(_Float16);
            const size_t kb = std::min(K, (tcm_size / bytes_per_k) & ~(KB - 1));
            GGML_ASSERT(kb >= KB && K % KB == 0);
            auto * a_tile = reinterpret_cast<_Float16 *>(tcm_buffer);
            auto * b_tile = a_tile + m_tile * kb;
            const size_t b_row_stride = (K / KB) * KB_BYTES;
            const size_t tile_count = mt_count * nt_count;
            for (size_t ti = static_cast<size_t>(ith); ti < tile_count; ti += static_cast<size_t>(nth)) {
                const size_t m0 = (ti / nt_count) * m_tile;
                const size_t nt = ti % nt_count;
                const size_t n0 = nt * n_tile;
                const size_t mc = std::min(m_tile, M - m0);
                const size_t nc = std::min(n_tile, N - n0);
                auto * c_tile = reinterpret_cast<float *>(out + m0 * out_row_stride) + n0;
                for (size_t k0 = 0; k0 < K; k0 += kb) {
                    const size_t kk = std::min(kb, K - k0);
                    pack_func((void *) (a_src + m0 * a_src_stride),
                              static_cast<int64_t>(k0 * sizeof(float)),
                              static_cast<int64_t>(a_src_stride),
                              (void *) a_tile, 0,
                              static_cast<int64_t>(kb * m_tile * sizeof(_Float16)),
                              static_cast<int64_t>(mc), static_cast<int64_t>(kk),
                              1, static_cast<int64_t>(kk / 8));
                    q41_func((void *) (b_src + nt * b_row_stride + (k0 / KB) * KB_BYTES),
                             0, static_cast<int64_t>(b_row_stride),
                             (void *) b_tile, 0,
                             static_cast<int64_t>(kk / 8) * 512,
                             1, static_cast<int64_t>(kk / KB),
                             1, static_cast<int64_t>(kk / 8));
                    ops.dot_f32acc(kk, b_tile, a_tile, c_tile, mc, nc,
                                   out_row_stride / sizeof(float), k0 != 0);
                }
            }
            return true;
        }

        const size_t a_budget = tcm_size - n_tile * ks * sizeof(_Float16);
        const size_t m_fit    = (a_budget / (ks * sizeof(_Float16))) & ~size_t{ 15 };
        // Reserve all padded M16 rows before B to prevent A/B overlap in TCM.
        const size_t m_block  = std::min(mt_count * m_tile, m_fit);
        const size_t nc_count = nt_count;

        // Align partials to 64B so each F16 row store stays within one DDR bank.
        auto * partials =
            reinterpret_cast<_Float16 *>((reinterpret_cast<uintptr_t>(ctx.workspace) + 63) & ~uintptr_t{ 63 });
        auto * a_blk    = reinterpret_cast<_Float16 *>(tcm_buffer);
        auto * b_blk    = a_blk + m_block * ks;
        const size_t slice_k0 = static_cast<size_t>(ith) * ks;

        if (static_cast<size_t>(ith) < s_count) {
            const size_t b_row_stride = (K / KB) * KB_BYTES;  // bytes per N32 tile row
            for (size_t m0 = 0; m0 < M; m0 += m_block) {
                const size_t m_count = std::min(m_block, M - m0);
                const size_t m_tiles = (m_count + m_tile - 1) / m_tile;
                for (size_t mt = 0; mt < m_tiles; ++mt) {
                    const size_t tm0 = m0 + mt * m_tile;
                    pack_func((void *) (a_src + tm0 * a_src_stride),
                              static_cast<int64_t>(slice_k0 * sizeof(float)),
                              static_cast<int64_t>(a_src_stride),
                              (void *) (a_blk + mt * m_tile * ks), 0,
                              static_cast<int64_t>(ks * m_tile * sizeof(_Float16)),
                              static_cast<int64_t>(std::min(m_tile, M - tm0)),
                              static_cast<int64_t>(ks),
                              1, static_cast<int64_t>(ks / 8));
                }
                // Rotate N traversal by worker to spread simultaneous stores across tile blocks.
                const size_t skew = static_cast<size_t>(ith);
                for (size_t j = 0; j < nc_count; ++j) {
                    const size_t nt = (j + skew) % nc_count;
                    const size_t n0 = nt * n_tile;
                    const size_t nc = std::min(n_tile, N - n0);
                    q41_func((void *) (b_src + nt * b_row_stride + (slice_k0 / KB) * KB_BYTES),
                             0,
                             static_cast<int64_t>(b_row_stride),
                             (void *) b_blk, 0,
                             static_cast<int64_t>(ks / 8) * 512,
                             1, static_cast<int64_t>(ks / KB),
                             1, static_cast<int64_t>(ks / 8));
                    for (size_t i = 0; i < m_tiles; ++i) {
                        const size_t mt  = (i + skew) % m_tiles;
                        const size_t tm0 = m0 + mt * m_tile;
                        // Partials use [M/16][nc][16][s_count][32]; each slice row occupies 64B.
                        _Float16 * c_blk = partials +
                                           ((tm0 / m_tile) * nc_count + nt) * m_tile * s_count * n_tile +
                                           static_cast<size_t>(ith) * n_tile;
                        ops.dot_f16acc(
                            ks, b_blk, a_blk + mt * m_tile * ks, c_blk, std::min(m_tile, M - tm0), nc,
                            s_count * n_tile, false);
                    }
                }
            }
        }

        ctx.sync();

        // Assign adjacent reduction tiles to workers to avoid strided DDR-bank aliasing.
        constexpr size_t reduce_chunk = 4;  // tiles per chunk
        const size_t     tile_count   = mt_count * nt_count;
        const size_t     n_chunks     = (tile_count + reduce_chunk - 1) / reduce_chunk;
        if (static_cast<size_t>(ith) < s_count) {
            for (size_t c = static_cast<size_t>(ith); c < n_chunks; c += s_count) {
                const size_t t0 = c * reduce_chunk;
                const size_t t1 = std::min(tile_count, t0 + reduce_chunk);
                for (size_t t = t0; t < t1; ++t) {
                    const size_t      mt      = t / nt_count;
                    const size_t      nt      = t % nt_count;
                    const _Float16 *  tile    = partials + t * m_tile * s_count * n_tile;
                    const size_t      count_m = std::min(m_tile, M - mt * m_tile);
                    const size_t      count_n = std::min(n_tile, N - nt * n_tile);
                    auto *            out_tile = reinterpret_cast<float *>(out + (mt * m_tile) * out_row_stride) +
                                                nt * n_tile;
                    reduce_func((void *) tile, 0, 0, 0,
                                (void *) out_tile, 0,
                                static_cast<int64_t>(out_row_stride),
                                1, 1, static_cast<int64_t>(s_count),
                                static_cast<int64_t>(count_m), static_cast<int64_t>(count_n));
                }
            }
        }

        return true;
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        GGML_LOG_DEBUG("%s: repack tensor %s (type %s) with q4_0_q41block_f16\n", __func__, t->name, ggml_type_name(t->type));
        switch (t->type) {
            case GGML_TYPE_Q4_0:
                return ggml::cpu::riscv64_spacemit::repack_q4_0_to_q41(t, data, data_size);
            case GGML_TYPE_Q4_1:
                return ggml::cpu::riscv64_spacemit::repack_q4_1_to_q41(t, data, data_size);
            case GGML_TYPE_Q4_K:
                return ggml::cpu::riscv64_spacemit::repack_q4_k_to_q41(t, data, data_size);
            default:
                GGML_ABORT("q41_f16: unsupported weight type %s", ggml_type_name(t->type));
        }
    }

    void unpack(const ggml_tensor * t, void * data, size_t data_size) const override {
        GGML_LOG_DEBUG("%s: unpack tensor %s from q4_0_q41block_f16\n", __func__, t->name);
        auto ok = ggml::cpu::riscv64_spacemit::unpack_q41_to_q4_0_common(t, data, data_size);
        GGML_ASSERT(ok == 0);
    }
};

// Q5_0 weights converted exactly to Q8_0 at load time and run on the Q8_0 IME path
class tensor_traits_q5_0_as_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        return repack_q5_0_to_q8_0_32(t, data, data_size);
    }
};

class tensor_traits_f32_as_q8_0 : public tensor_traits<block_q8_0, 32, 32> {
    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        return repack_f32_to_q8_0_32(t, data, data_size);
    }

    void unpack(const ggml_tensor * t, void * data, size_t data_size) const override {
        const int ok = unpack_q8_0_32_to_f32(t, data, data_size);
        GGML_ASSERT(ok == 0);
    }
};

template <typename T> static const T & traits_instance() {
    static const T traits;
    return traits;
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
static const auto & i8_traits() {
    return traits_instance<tensor_traits<BLOC_TYPE, INTER_SIZE, NB_COLS>>();
}

template <q41_zero_point_mode Mode>
static const auto & f16_traits() {
    return traits_instance<tensor_traits_f16<Q41Block<32, 32, 2>, 32, 32, Mode>>();
}

// Impl By RVV
static const tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_optimal_repack_type(const ggml_tensor * cur) {
    using ggml::cpu::riscv64_spacemit::f16_traits;
    using ggml::cpu::riscv64_spacemit::i8_traits;
    using ggml::cpu::riscv64_spacemit::q41_zero_point_mode;
    using ggml::cpu::riscv64_spacemit::traits_instance;

    switch (cur->type) {
        case GGML_TYPE_Q2_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q2_K, 256, 32>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q3_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q3_K, 256, 32>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (ggml::cpu::riscv64_spacemit::use_f16_link(cur)) {
                    return &f16_traits<q41_zero_point_mode::fixed_8>();
                }

                if (cur->ne[2] > 1 && cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q4_0, 256, 32>();
                }

                if (cur->ne[2] > 1 && cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q4_0, 32, 32>();
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[2] > 1 && cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &i8_traits<block_q4_0, 32, 16>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (ggml::cpu::riscv64_spacemit::use_f16_link(cur)) {
                    return &f16_traits<q41_zero_point_mode::per_row>();
                }

                if (cur->ne[2] > 1 && cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q4_1, 256, 32>();
                }

                if (cur->ne[2] > 1 && cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q4_1, 32, 32>();
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[2] > 1 && cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &i8_traits<block_q4_1, 32, 16>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (ggml::cpu::riscv64_spacemit::use_f16_link(cur)) {
                    return &f16_traits<q41_zero_point_mode::per_row>();
                }

                if (cur->ne[2] > 1 && cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q4_K, 256, 32>();
                }

                if (cur->ne[2] > 1 && cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q4_K, 32, 32>();
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[2] > 1 && cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &i8_traits<block_q4_K, 32, 16>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q6_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q6_K, 32, 32>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q8_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q8_0, 32, 32>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q5_K, 32, 32>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &i8_traits<block_q5_1, 32, 32>();
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // exact conversion to Q8_0 at load time, run on the Q8_0 IME path
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % QK8_0 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &traits_instance<ggml::cpu::riscv64_spacemit::tensor_traits_q5_0_as_q8_0>();
                }
#endif
            }
            break;
        case GGML_TYPE_F32:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                const size_t nl = strlen(cur->name);
                if (strstr(cur->name, "ffn_gate_inp") != nullptr && strstr(cur->name, "shexp") == nullptr &&
                    nl > 7 && strcmp(cur->name + nl - 7, ".weight") == 0 && ggml_n_dims(cur) == 2 &&
                    cur->ne[1] % 32 == 0 && cur->ne[0] % QK8_0 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &traits_instance<ggml::cpu::riscv64_spacemit::tensor_traits_f32_as_q8_0>();
                }
#endif
            }
            break;
        default:
            break;
    }

    return nullptr;
}

size_t ggml_spacemit_nbytes(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    GGML_UNUSED(buft);

    const auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        return total;
    };

    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        return plain_nbytes();
    }

    const size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    const auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        return total;
    };

    const auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        if (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        return total;
    };

    size_t nbytes = row_nbytes;
    switch (tensor->type) {
        case GGML_TYPE_Q4_K:
            // Q41Block needs 19 B per row/K32; each Q4_K block contains eight K32 windows.
            nbytes = ggml::cpu::riscv64_spacemit::use_f16_link(tensor)
                         ? remap_block_nbytes(sizeof(block_q4_K), 8 * sizeof(ggml::cpu::riscv64_spacemit::Q41Block<32, 32, 2>) / 32, 32)
                         : (tensor->ne[2] > 1 ? remap_block_nbytes(sizeof(block_q4_K), sizeof(block_q4_1) * 8)
                                              : add_strided_nbytes(row_nbytes, 1, 1));
            break;
        case GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(block_q8_0) * 8, 32);
            break;
        case GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(sizeof(block_q8_0), sizeof(block_q8_0), 32);
            break;
        case GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(sizeof(block_q2_K), sizeof(spacemit_kernels::nrow_block_q2_k<1>));
            break;
        case GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(sizeof(block_q3_K), sizeof(spacemit_kernels::nrow_block_q3_k<1>));
            break;
        case GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(sizeof(block_q5_K), sizeof(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            break;
        case GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(sizeof(block_q5_1), sizeof(spacemit_kernels::nrow_block_q5_1<1>));
            break;
        case GGML_TYPE_Q5_0:
            if (const auto * repack = ggml_spacemit_get_optimal_repack_type(tensor);
                repack && repack ==
                    &ggml::cpu::riscv64_spacemit::traits_instance<ggml::cpu::riscv64_spacemit::tensor_traits_q5_0_as_q8_0>()) {
                nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(block_q8_0), 32);
            } else {
                nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(spacemit_kernels::nrow_block_q5_0<1>));
            }
            break;
        case GGML_TYPE_Q4_0:
            // Q41Block needs 19 B per row/K32, padded to 32 rows.
            nbytes = ggml::cpu::riscv64_spacemit::use_f16_link(tensor)
                         ? remap_block_nbytes(sizeof(block_q4_0), sizeof(ggml::cpu::riscv64_spacemit::Q41Block<32, 32, 2>) / 32, 32)
                         : add_strided_nbytes(row_nbytes, 1, 1);
            break;
        case GGML_TYPE_Q4_1:
            nbytes = ggml::cpu::riscv64_spacemit::use_f16_link(tensor)
                         ? remap_block_nbytes(sizeof(block_q4_1), sizeof(ggml::cpu::riscv64_spacemit::Q41Block<32, 32, 2>) / 32, 32)
                         : add_strided_nbytes(row_nbytes, 1, 1);
            break;
        default:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
    }

    return std::max(nbytes, ggml_nbytes(tensor));
}

bool ggml_spacemit_compute_opnode(ggml::spacemit::context & ctx, const spacemit_opnode & node) {
    // Single-op opnodes with a special opcode: the two halves of the concat -> ssm_conv fusion.
    // They run at their original graph positions (see try_fuse_node); the allocator plans buffer
    // reuse on that order, so they must not be merged into one opnode at the concat position.
    switch (node.opcode) {
        case SPACEMIT_OP_CONCAT_STATE:
            // only the trailing columns read back through views (the conv state update)
            if (!spacemit_kernels::rvv::forward_concat_dim0_transposed_f32(ctx, node.node,
                                                                           (int64_t) node.kernel_params[0])) {
                GGML_ABORT("fused concat state update: unsupported shape");
            }
            return true;
        case SPACEMIT_OP_SSM_CONV_CONCAT:
            // the concat only materialized its trailing columns; read its inputs directly
            if (!spacemit_kernels::rvv::forward_ssm_conv_concat_f32(ctx, node.node)) {
                GGML_ABORT("fused ssm_conv: unsupported shape");
            }
            return true;
        default:
            break;
    }
    // Fused opnodes dispatch through the fused-op table; single-op opnodes keep
    // the existing traits path.
    if (!node.fused.empty()) {
        return ggml_spacemit_compute_fused(ctx, node);
    }
    return ggml_spacemit_compute_forward(ctx, node.node);
}

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_tensor_traits(const ggml_tensor * op) {
    if (op == nullptr) {
        return nullptr;
    }

    const auto * common = &ggml::cpu::riscv64_spacemit::rvv_impl;
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            if (op->src[0] && op->src[1] && ggml_n_dims(op->src[0]) == 2 &&
                op->src[1]->type == GGML_TYPE_F32) {
                const auto * traits = static_cast<const ggml::spacemit::tensor_traits_base *>(op->src[0]->extra);
                if (traits) return traits;
                const auto * repack = ggml_spacemit_get_optimal_repack_type(op->src[0]);
                if (repack) return repack;
                // F16 dense weight: same condition as compute_forward
                if (op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[0]->nb[0] == sizeof(_Float16) &&
                    op->src[1]->nb[0] == sizeof(float)) {
                    return common;
                }
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0] && op->src[1] && ggml_n_dims(op->src[0]) == 3 &&
                op->src[1]->type == GGML_TYPE_F32) {
                const auto * traits = static_cast<const ggml::spacemit::tensor_traits_base *>(op->src[0]->extra);
                return traits ? traits : ggml_spacemit_get_optimal_repack_type(op->src[0]);
            }
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)) {
                return common;
            }
            break;
        case GGML_OP_SCALE:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32) {
                return common;
            }
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0] && op->src[1] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == op->src[0]->type && op->type == op->src[0]->type &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) &&
                op->nb[0] == ggml_type_size(op->type) && ggml_can_repeat(op->src[1], op->src[0]) &&
                ggml_are_same_shape(op->src[0], op)) {
                return common;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->op_params[3] == GGML_PREC_F32 || op->op_params[3] == GGML_PREC_DEFAULT) &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[1]->ne[0] > 0 && op->src[2]->ne[0] > 0) {
                return common;
            }
            break;
        case GGML_OP_ROPE:
            if (op->src[0] && op->src[1]) {
                float freq_base;
                memcpy(&freq_base,  op->op_params + 5, sizeof(float));
                const int n_dims = ggml_get_op_params_i32(op, 1);
                const int mode   = ggml_get_op_params_i32(op, 2);
                const bool src2_ok = op->src[2] == nullptr ||
                    (op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]));
                const bool mode_ok = (mode == GGML_ROPE_TYPE_NORMAL ||
                                      mode == GGML_ROPE_TYPE_NEOX   ||
                                      mode == GGML_ROPE_TYPE_MROPE  ||
                                      mode == GGML_ROPE_TYPE_IMROPE);
                const bool c1 = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                const bool c2 = op->type == op->src[0]->type;
                const bool c3 = op->src[1]->type == GGML_TYPE_I32;
                const bool c4 = src2_ok;
                const bool c5 = op->src[0]->nb[0] == ggml_type_size(op->src[0]->type);
                const bool c6 = op->nb[0] == ggml_type_size(op->type);
                const bool c7 = freq_base > 0.0f;
                const bool c8 = n_dims > 0 && n_dims <= op->ne[0] && n_dims <= 512 && n_dims % 2 == 0;
                const bool c9 = mode_ok;
                if (c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9) {
                    return common;
                }
            }
            break;
        case GGML_OP_CONT:
            if (op->src[0] && op->type == op->src[0]->type) {
                // permute case
                if (op->nb[0] != op->src[0]->nb[0] && op->nb[0] == op->src[0]->nb[1] &&
                    op->ne[3] * op->ne[2] * op->nb[2] == op->src[0]->ne[3] * op->src[0]->ne[2] * op->src[0]->nb[2]) {
                    return common;
                }
                // general stride-aware copy for F32/F16
                if ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                    ggml_is_contiguous(op) &&
                    ggml_nelements(op) == ggml_nelements(op->src[0])) {
                    return common;
                }
            }
            break;
        case GGML_OP_CPY:
            if (op->src[0] && ggml_nelements(op) == 0) {
                return common;
            }
            if (op->src[0] && op->type == op->src[0]->type && op->nb[0] == op->src[0]->nb[1] &&
                op->src[0]->nb[0] != op->src[0]->nb[1] && ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            if (op->src[0] && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            // row-contiguous F32 copy with arbitrary outer strides (GDN rollback-snapshot copies)
            if (op->src[0] && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                op->nb[0] == sizeof(float) && op->src[0]->nb[0] == sizeof(float) &&
                op->ne[0] == op->src[0]->ne[0] && op->ne[1] == op->src[0]->ne[1] &&
                op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3]) {
                return common;
            }
            if (op->src[0] &&
                ((op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32) ||
                 (op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F16)) &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            break;
        case GGML_OP_SET_ROWS:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_I32 || op->src[1]->type == GGML_TYPE_I64) &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[2]->type == op->type &&
                ggml_are_same_shape(op, op->src[2]) && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->src[2]->nb[0] == ggml_type_size(op->src[2]->type) && op->nb[0] == ggml_type_size(op->type) &&
                op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) && op->ne[0] == op->src[0]->ne[0] &&
                op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3] &&
                op->src[0]->ne[2] % op->src[1]->ne[1] == 0 && op->src[0]->ne[3] % op->src[1]->ne[2] == 0) {
                return common;
            }
            break;
        case GGML_OP_REPEAT:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)) {
                const bool rows_equal = ggml_nrows(op->src[0]) == ggml_nrows(op);
                const bool n0_matches = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];
                if ((rows_equal && n0_matches) ||
                    (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0])) {
                    return common;
                }
            }
            break;
        case GGML_OP_SUM_ROWS:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                return common;
            }
            break;
        case GGML_OP_GET_ROWS:
            if (op->src[0] && op->src[1] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->type == op->src[0]->type &&
                op->src[1]->type == GGML_TYPE_I32 && op->ne[0] == op->src[0]->ne[0] &&
                op->src[0]->ne[2] == op->src[1]->ne[1] &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                ggml_nrows(op) == ggml_nelements(op->src[1])) {
                return common;
            }
            break;
        case GGML_OP_CONCAT:
            if (op->src[0] && op->src[1] && ggml_get_op_params_i32(op, 0) == 0 &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == op->src[0]->type && op->type == op->src[0]->type &&
                op->nb[0] == ggml_type_size(op->type) &&
                op->nb[1] == ggml_type_size(op->type) * (op->src[0]->ne[0] + op->src[1]->ne[0])) {
                return common;
            }
            break;
        case GGML_OP_UNARY:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                (ggml_get_unary_op(op) == GGML_UNARY_OP_TANH    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_GELU    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SILU    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_EXP     ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_NEG     ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS)) {
                return common;
            }
            break;
        case GGML_OP_SOFT_MAX:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float)) {
                return common;
            }
            break;
        case GGML_OP_GLU:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32) {
                if (ggml_get_glu_op(op) == GGML_GLU_OP_GEGLU) {
                    return common;
                }
                if (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && ggml_is_contiguous_1(op->src[0]) &&
                    op->type == GGML_TYPE_F32 && op->nb[0] == sizeof(float)) {
                    const int32_t swapped = ggml_get_op_params_i32(op, 1);
                    if (op->src[1]) {
                        if (swapped == 0 && op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous_1(op->src[1]) &&
                            ggml_are_same_shape(op->src[0], op->src[1]) && ggml_are_same_shape(op, op->src[0])) {
                            return common;
                        }
                    } else if (op->ne[0] == op->src[0]->ne[0] / 2 && ggml_nrows(op) == ggml_nrows(op->src[0])) {
                        return common;
                    }
                }
            }
            break;
        default:
            break;
    }

    // Scalar ops: route to common so compute_forward dispatches to scalar_kernels
    switch (op->op) {
        case GGML_OP_L2_NORM:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_FILL:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16))
                return common;
            break;
        case GGML_OP_CUMSUM:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_PAD:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_TRI:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]))
                return common;
            break;
        case GGML_OP_DIAG:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_SET:
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_SOLVE_TRI:
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_GATED_DELTA_NET:
            if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2) {
                return common;
            }
            break;
        case GGML_OP_SSM_CONV:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_ARGSORT:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(int32_t))
                return common;
            break;
        case GGML_OP_CLAMP:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) && ggml_are_same_shape(op, op->src[0]))
                return common;
            break;
        case GGML_OP_MUL_MAT:
            // Generic CPU dot fallback for dense or unsupported quantized
            // layouts which do not have a repacked Spacemit trait.
            if (op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot != nullptr) {
                return common;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->op_params[3] == GGML_PREC_F32 || op->op_params[3] == GGML_PREC_DEFAULT) &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[1]->ne[0] > 0 && op->src[2]->ne[0] > 0) {
                return common;
            }
            break;
        default:
            break;
    }

    return nullptr;
}

int ggml_riscv64_spacemit_repack_tensor(ggml_tensor * tensor, const void * data, size_t size) {
    auto traits = (ggml::spacemit::tensor_traits_base *) tensor->extra;
    if (traits) {
        return traits->repack(tensor, data, size);
    }
    memcpy(tensor->data, data, size);
    return 0;
}

void ggml_riscv64_spacemit_unpack_tensor(const ggml_tensor * tensor, void * data, size_t size) {
    auto traits = (const ggml::spacemit::tensor_traits_base *) tensor->extra;
    if (traits) {
        traits->unpack(tensor, data, size);
        return;
    }
    memcpy(data, tensor->data, size);
}

bool ggml_spacemit_get_work_size(int n_threads, const ggml_tensor * op, size_t * size) {
    const auto * traits = ggml_spacemit_get_tensor_traits(op);
    return traits != nullptr && traits->work_size(n_threads, op, *size);
}

bool ggml_spacemit_compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const auto * traits = ggml_spacemit_get_tensor_traits(op);
    return traits != nullptr && traits->compute_forward(ctx, op);
}

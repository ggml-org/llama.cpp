// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

// KleidiAI micro-kernels
#include <cstddef>

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"

#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"

#include "kai/kai_common.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "kernels.h"

#define NELEMS(x) (sizeof(x) / sizeof(*x))

template<size_t(*Fn)(size_t,size_t,size_t)>
static inline size_t kernel_offs_fn3(size_t a, size_t b, size_t c) {
    return Fn(a, b, c);
}

template<size_t(*Fn)(size_t,size_t)>
static inline size_t kernel_offs_fn2(size_t a, size_t b, size_t) {
    return Fn(a, b);
}

template<void(*Fn)(size_t,size_t,size_t,size_t,const void*,const void*,float*,size_t,size_t,float,float)>
static inline void kernel_run_fn11(size_t m, size_t n, size_t k, size_t bl,
                                     const void* lhs, const void* rhs, void* dst,
                                     size_t dst_stride_row, size_t dst_stride_col,
                                     float clamp_min, float clamp_max) {
    Fn(m, n, k, bl, lhs, rhs, static_cast<float*>(dst), dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

template<void(*Fn)(size_t,size_t,size_t,const void*,const void*,void*,size_t,size_t,float,float)>
static inline void kernel_run_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                   const void* lhs, const void* rhs, void* dst,
                                   size_t dst_stride_row, size_t dst_stride_col,
                                   float clamp_min, float clamp_max) {
    Fn(m, n, k, lhs, rhs, dst, dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

template <void (*Fn)(size_t, size_t, size_t, const void *, size_t, const void *, void *, size_t, size_t, float, float)>
static inline void kernel_run_lhs_stride_fn10(size_t       m,
                                              size_t       n,
                                              size_t       k,
                                              size_t       lhs_stride,
                                              const void * lhs,
                                              const void * rhs,
                                              void *       dst,
                                              size_t       dst_stride_row,
                                              size_t       dst_stride_col,
                                              float        clamp_min,
                                              float        clamp_max) {
    Fn(m, n, k, lhs, lhs_stride, rhs, dst, dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

template<void(*Fn)(size_t,size_t,size_t,const void*,const void*,float*,size_t,size_t,float,float)>
static inline void kernel_run_float_fn10(size_t m, size_t n, size_t k, size_t /*bl*/,
                                         const void* lhs, const void* rhs, void* dst,
                                         size_t dst_stride_row, size_t dst_stride_col,
                                         float clamp_min, float clamp_max) {
    Fn(m, n, k, lhs, rhs, static_cast<float*>(dst), dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
static inline size_t lhs_ps_fn6(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    return Fn(m, k, bl, mr, kr, sr);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
static inline size_t lhs_ps_fn5(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    return Fn(m, k, mr, kr, sr);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t)>
static inline size_t lhs_offs_fn6(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    return Fn(m_idx, k, bl, mr, kr, sr);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
static inline size_t lhs_offs_fn5(size_t m_idx, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr) {
    return Fn(m_idx, k, mr, kr, sr);
}

template<void(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,const float*,size_t,void*)>
static inline void lhs_pack_float_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_start, const void* lhs, size_t lhs_stride, void* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_start, static_cast<const float*>(lhs), lhs_stride, lhs_packed);
}

template<void(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,const void*,size_t,void*)>
static inline void lhs_pack_void_fn10(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                           size_t m_idx_start, const void* lhs, size_t lhs_stride, void* lhs_packed) {
    Fn(m, k, bl, mr, kr, sr, m_idx_start, lhs, lhs_stride, lhs_packed);
}

template<void(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,const void*,size_t,void*)>
static inline void lhs_pack_void_fn9(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                             size_t m_idx_start, const void* lhs, size_t lhs_stride, void* lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride, lhs_packed);
}

template<void(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,const float*,size_t,void*)>
static inline void lhs_pack_float_fn9_no_bl(size_t m, size_t k, size_t /*bl*/, size_t mr, size_t kr, size_t sr,
                                            size_t m_idx_start, const void * lhs, size_t lhs_stride, void * lhs_packed) {
    Fn(m, k, mr, kr, sr, m_idx_start, static_cast<const float*>(lhs), lhs_stride, lhs_packed);
}

template<size_t(*Fn)(size_t,size_t,size_t,size_t,size_t)>
static inline size_t rhs_ps_fn5(size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    return Fn(n, k, nr, kr, bl);
}

template<size_t(*Fn)(size_t,size_t)>
static inline size_t rhs_ps_fn2(size_t n, size_t k, size_t /*nr*/, size_t /*kr*/, size_t /*bl*/) {
    return Fn(n, k);
}

template<void(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,const uint8_t*,const float*,void*,size_t,const struct kai_rhs_pack_qs4cxs1s0_param*)>
static inline void rhs_pack_fn12(size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                                      size_t /*rhs_stride*/, const void* rhs, const void* bias, const void* /*scale*/,
                                      void* rhs_packed, size_t extra_bytes, const void* params) {
    Fn(num_groups, n, k, nr, kr, sr, bl,
       static_cast<const uint8_t*>(rhs),
       static_cast<const float*>(bias),
       rhs_packed, extra_bytes,
       static_cast<const kai_rhs_pack_qs4cxs1s0_param*>(params));
}

template<void(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,const int8_t*,const float*,const float*,void*,size_t,const struct kai_rhs_pack_qsi8cx_params*)>
static inline void rhs_pack_scale_fn12(size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                       size_t /*rhs_stride*/, const void* rhs, const void* bias, const void* scale,
                                       void* rhs_packed, size_t extra_bytes, const void* params) {
    Fn(num_groups, n, k, nr, kr, sr,
       static_cast<const int8_t*>(rhs),
       static_cast<const float*>(bias),
       static_cast<const float*>(scale),
       rhs_packed, extra_bytes,
       static_cast<const kai_rhs_pack_qsi8cx_params*>(params));
}

template<void(*Fn)(size_t,size_t,size_t,size_t,size_t,size_t,size_t,const void*,const void*,const void*,void*,size_t,const void*)>
static inline void rhs_pack_fn13(size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t /*bl*/,
                                               size_t rhs_stride, const void* rhs, const void* bias, const void* scale,
                                               void* rhs_packed, size_t extra_bytes, const void* params) {
    Fn(num_groups, n, k, nr, kr, sr, rhs_stride, rhs, bias, scale, rhs_packed, extra_bytes, params);
}

static constexpr size_t KLEIDIAI_Q4_K_BLOCK_LENGTH = QK_K / 8;

static_assert(QK_K == 256, "KleidiAI Q4_K kernels require 256-value super-blocks");
static_assert(sizeof(block_q4_K) == 144, "Unexpected ggml Q4_K block layout");
static_assert(offsetof(block_q4_K, scales) == 4, "Unexpected ggml Q4_K metadata offset");
static_assert(offsetof(block_q4_K, qs) == 16, "Unexpected ggml Q4_K data offset");

static struct kai_matmul_uker_config q4_k_matmul_config() {
    struct kai_matmul_uker_config config = {};
    config.format.bl = KLEIDIAI_Q4_K_BLOCK_LENGTH;
    return config;
}

template<struct kai_matmul_uker_api (*GetApi)(void)>
static size_t q4_k_get_m_step() {
    const struct kai_matmul_uker_api api = GetApi();
    const struct kai_matmul_uker_config config = q4_k_matmul_config();
    return api.get_step(&config).m;
}

template<struct kai_matmul_uker_api (*GetApi)(void)>
static size_t q4_k_get_n_step() {
    const struct kai_matmul_uker_api api = GetApi();
    const struct kai_matmul_uker_config config = q4_k_matmul_config();
    return api.get_step(&config).n;
}

template<size_t Value>
static size_t q4_k_get_constant() {
    return Value;
}

template<struct kai_matmul_uker_api (*GetApi)(void)>
static size_t q4_k_get_dst_offset(size_t m_idx, size_t n_idx, size_t dst_stride) {
    const struct kai_matmul_uker_api api = GetApi();
    const struct kai_matmul_uker_config config = q4_k_matmul_config();
    const struct kai_matmul_uker_dst_dim_args index = { m_idx, n_idx };
    const struct kai_matmul_uker_dst_stride_args stride = { dst_stride };
    return api.get_dst_offset(&config, &index, &stride);
}

template<struct kai_matmul_uker_api (*GetApi)(void)>
static size_t q4_k_get_lhs_packed_offset(size_t m_idx, size_t k, size_t bl) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);

    const struct kai_matmul_uker_api api = GetApi();
    const struct kai_matmul_uker_config config = q4_k_matmul_config();
    const struct kai_matmul_uker_lhs_dim_args shape = { m_idx, k };
    const struct kai_matmul_uker_lhs_dim_args index = { m_idx, 0 };
    const struct kai_matmul_uker_lhs_stride_args stride = api.get_lhs_stride(&config, &shape);
    return api.get_lhs_offset(&config, &index, &stride);
}

template<struct kai_matmul_uker_api (*GetApi)(void)>
static size_t q4_k_get_rhs_packed_offset(size_t n_idx, size_t k, size_t bl) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);

    const struct kai_matmul_uker_api api = GetApi();
    const struct kai_matmul_uker_config config = q4_k_matmul_config();
    const struct kai_matmul_uker_rhs_dim_args shape = { n_idx, k };
    const struct kai_matmul_uker_rhs_dim_args index = { n_idx, 0 };
    const struct kai_matmul_uker_rhs_stride_args stride = api.get_rhs_stride(&config, &shape);
    return api.get_rhs_offset(&config, &index, &stride);
}

template<struct kai_matmul_uker_api (*GetApi)(void)>
static void q4_k_run_matmul(
        size_t m, size_t n, size_t k, size_t bl,
        const void * lhs_packed, const void * rhs_packed,
        void * dst, size_t dst_stride_row, size_t dst_stride_col,
        float clamp_min, float clamp_max) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(dst_stride_col == sizeof(float));

    const struct kai_matmul_uker_api api = GetApi();
    const struct kai_matmul_uker_config config = q4_k_matmul_config();
    const struct kai_matmul_uker_lhs_dim_args lhs_shape = { m, k };
    const struct kai_matmul_uker_rhs_dim_args rhs_shape = { n, k };

    struct kai_matmul_uker_args args = {};
    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.shape = { m, n, k };
    args.operand.lhs.ptr = lhs_packed;
    args.operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);
    args.operand.rhs.ptr = rhs_packed;
    args.operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
    args.operand.dst.ptr = dst;
    args.operand.dst.stride.m = dst_stride_row;
    args.activation.clamp.min_ptr = &clamp_min;
    args.activation.clamp.max_ptr = &clamp_max;
    api.run(&config, &args);
}

static struct kai_matmul_pack_lhs_uker_config q4_k_lhs_pack_config() {
    struct kai_matmul_pack_lhs_uker_config config = {};
    return config;
}

static size_t q4_k_lhs_get_offset(size_t m_idx, size_t lhs_stride) {
    const struct kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    const struct kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    const struct kai_matmul_pack_lhs_uker_lhs_dim_args index = { m_idx, 0 };
    const struct kai_matmul_pack_lhs_uker_lhs_stride_args stride = { lhs_stride };
    return api.get_lhs_offset(&config, &index, &stride);
}

static size_t q4_k_lhs_get_packed_offset(
        size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    const struct kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    const struct kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = { m_idx, k };
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args index = { m_idx, 0 };
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = api.get_lhs_packed_stride(&config, &shape);
    return api.get_lhs_packed_offset(&config, &index, &stride);
}

static size_t q4_k_lhs_get_packed_size(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);

    const struct kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    const struct kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = { m, k };
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = api.get_lhs_packed_stride(&config, &shape);
    return api.get_lhs_packed_size(&config, &shape, &stride);
}

static void q4_k_lhs_pack(
        size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_start, const void * lhs, size_t lhs_stride, void * lhs_packed) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(mr == 1 && kr == 4 && sr == 2);
    GGML_ASSERT(m_idx_start == 0);

    const struct kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    const struct kai_matmul_pack_lhs_uker_config config = q4_k_lhs_pack_config();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args packed_shape = { m, k };

    struct kai_matmul_pack_lhs_uker_args args = {};
    args.flags = 0;
    args.shape = { m, k };
    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride.m = lhs_stride;
    args.operand.lhs_packed.ptr = lhs_packed;
    args.operand.lhs_packed.stride = api.get_lhs_packed_stride(&config, &packed_shape);
    api.run(&config, &args);
}

static struct kai_matmul_pack_rhs_uker_config q4_k_rhs_pack_config() {
    struct kai_matmul_pack_rhs_uker_config config = {};
    return config;
}

static size_t q4_k_rhs_get_packed_size(size_t n, size_t k, size_t nr, size_t /*kr*/, size_t bl) {
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);

    const struct kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    const struct kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_step(&config).n == nr);
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args shape = { n, k };
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = api.get_rhs_packed_stride(&config, &shape);
    return api.get_rhs_packed_size(&config, &shape, &stride);
}

static void q4_k_rhs_pack(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t /*kr*/, size_t /*sr*/, size_t bl,
        size_t rhs_stride, const void * rhs, const void * bias, const void * scale,
        void * rhs_packed, size_t extra_bytes, const void * params) {
    GGML_ASSERT(num_groups == 1);
    GGML_ASSERT(bl == KLEIDIAI_Q4_K_BLOCK_LENGTH);
    GGML_ASSERT(bias == nullptr);
    GGML_ASSERT(scale == nullptr);
    GGML_ASSERT(extra_bytes == 0);
    GGML_ASSERT(params == nullptr);

    const struct kai_matmul_pack_rhs_uker_api api =
        kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    const struct kai_matmul_pack_rhs_uker_config config = q4_k_rhs_pack_config();
    GGML_ASSERT(api.get_step(&config).n == nr);
    const struct kai_matmul_pack_rhs_uker_rhs_dim_args rhs_shape = { n, k };
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args packed_shape = { n, k };

    struct kai_matmul_pack_rhs_uker_args args = {};
    args.flags = 0;
    args.shape = { n, k };
    args.operand.rhs.ptr = rhs;
    args.operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
    if (rhs_stride != 0) {
        GGML_ASSERT(rhs_stride >= args.operand.rhs.stride.n);
        args.operand.rhs.stride.n = rhs_stride;
    }
    args.operand.rhs_packed.ptr = rhs_packed;
    args.operand.rhs_packed.stride = api.get_rhs_packed_stride(&config, &packed_shape);
    api.run(&config, &args);
}

static ggml_kleidiai_kernels gemm_gemv_kernels[] = {
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa>,
        },

        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_run_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32_neon>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2 | CPU_FEATURE_FP16,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME2 GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_fn10<kai_run_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* SME2 GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ nullptr,
            /* .get_rhs_packed_offset_ex = */ nullptr,
            /* .run_kernel_ex         = */ nullptr,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_bf16p2vlx2_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_size_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F16,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#if defined(__APPLE__)
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#else
    {
        /* SVE i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* SVE dotprod GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SVE | CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn3<kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn3<kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_fn11<kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn10<kai_run_lhs_quant_pack_qsi8d32p_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
            /* .pack_func_ex          = */ &rhs_pack_fn12<kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
#endif
    { /* Sentinel */ }
};

static ggml_kleidiai_kernels gemm_gemv_kernels_q4_k[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ &q4_k_get_m_step<kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa>,
            /* .get_n_step            = */ &q4_k_get_n_step<kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa>,
            /* .get_mr                = */ &q4_k_get_m_step<kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa>,
            /* .get_nr                = */ &q4_k_get_n_step<kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa>,
            /* .get_kr                = */ &q4_k_get_constant<4>,
            /* .get_sr                = */ &q4_k_get_constant<2>,
            /* .get_dst_offset        = */ &q4_k_get_dst_offset<kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa>,
            /* .get_lhs_offset_ex     = */ &q4_k_get_lhs_packed_offset<kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &q4_k_get_rhs_packed_offset<kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa>,
            /* .run_kernel_ex         = */ &q4_k_run_matmul<kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa>,
        },
        {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f16pmrx2_f32_neon,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn6<kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon>,
            /* .packed_size_ex        = */ &lhs_ps_fn6<kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn10<kai_run_lhs_pack_f16pmrx2_f32_neon>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ &q4_k_get_m_step<kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot>,
            /* .get_n_step            = */ &q4_k_get_n_step<kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot>,
            /* .get_mr                = */ &q4_k_get_m_step<kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot>,
            /* .get_nr                = */ &q4_k_get_n_step<kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot>,
            /* .get_kr                = */ &q4_k_get_constant<4>,
            /* .get_sr                = */ &q4_k_get_constant<2>,
            /* .get_dst_offset        = */ &q4_k_get_dst_offset<kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot>,
            /* .get_lhs_offset_ex     = */ &q4_k_get_lhs_packed_offset<kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot>,
            /* .get_rhs_packed_offset_ex = */ &q4_k_get_rhs_packed_offset<kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot>,
            /* .run_kernel_ex         = */ &q4_k_run_matmul<kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot>,
        },
        {
            /* .get_offset            = */ q4_k_lhs_get_offset,
            /* .get_packed_offset_ex  = */ q4_k_lhs_get_packed_offset,
            /* .packed_size_ex        = */ q4_k_lhs_get_packed_size,
            /* .pack_func_ex          = */ q4_k_lhs_pack,
        },
        {
            /* .packed_size_ex        = */ q4_k_rhs_get_packed_size,
            /* .pack_func_ex          = */ q4_k_rhs_pack,
            /* .repack_mode           = */ RHS_REPACK_SINGLE_ONLY,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2 | CPU_FEATURE_FP16,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q4_K,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

static ggml_kleidiai_kernels gemm_gemv_kernels_q8[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* SME GEMV (pure SME, no SME2 required) */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* I8MM GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* I8MM GEMV (dotprod fallback) */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_I8MM | CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* DOTPROD GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* DOTPROD GEMV */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
            /* .run_kernel_ex         = */ &kernel_run_float_fn10<kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32>,
            /* .pack_func_ex          = */ &lhs_pack_float_fn9_no_bl<kai_run_lhs_quant_pack_qai8dxp_f32>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn5<kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
            /* .pack_func_ex          = */ &rhs_pack_scale_fn12<kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon>,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_Q8_0,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

static ggml_kleidiai_kernels ggml_kleidiai_kernels_f32[] = {
    {
        /* SME2 GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_fn10<kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME2 GEMV */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_mr                = */ kai_get_m_step_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
            /* .run_kernel_ex         = */ &kernel_run_lhs_stride_fn10<kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla>,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ nullptr,
            /* .get_packed_offset_ex  = */ nullptr,
            /* .packed_size_ex        = */ nullptr,
            /* .pack_func_ex          = */ nullptr,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME2,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    {
        /* SME GEMM */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ &kernel_offs_fn2<kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .get_rhs_packed_offset_ex = */ &kernel_offs_fn2<kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
            /* .run_kernel_ex         = */ &kernel_run_fn10<kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa>,
        },
        /* .gemm_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* SME GEMV */
        {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa,
            /* .get_lhs_offset_ex     = */ nullptr,
            /* .get_rhs_packed_offset_ex = */ nullptr,
            /* .run_kernel_ex         = */ nullptr,
        },
        /* .gemv_lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
            /* .get_packed_offset_ex  = */ &lhs_offs_fn5<kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme>,
            /* .packed_size_ex        = */ &lhs_ps_fn5<kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme>,
            /* .pack_func_ex          = */ &lhs_pack_void_fn9<kai_run_lhs_pack_f32p2vlx1_f32_sme>,
        },
        /* .rhs_info = */ {
            /* .packed_size_ex        = */ &rhs_ps_fn2<kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
            /* .pack_func_ex          = */ &rhs_pack_fn13<kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme>,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
        /* .lhs_type           = */ GGML_TYPE_F32,
        /* .rhs_type           = */ GGML_TYPE_F32,
        /* .op_type            = */ GGML_TYPE_F32,
    },
    { /* Sentinel */ }
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, const ggml_tensor * tensor) {
    ggml_kleidiai_kernels * kernel = nullptr;

    if (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] != nullptr && tensor->src[1] != nullptr) {
        auto try_table = [&](auto & table) {
            for (size_t i = 0; i < NELEMS(table) - 1; ++i) {
                if ((cpu_features & table[i].required_cpu) == table[i].required_cpu &&
                    table[i].lhs_type == tensor->src[1]->type &&
                    table[i].rhs_type == tensor->src[0]->type &&
                    table[i].op_type  == tensor->type) {
                    kernel = &table[i];
                    return true;
                }
            }
            return false;
        };

        if (tensor->src[0]->type == GGML_TYPE_Q4_K) {
            try_table(gemm_gemv_kernels_q4_k);
        } else if (tensor->src[0]->type == GGML_TYPE_Q8_0 || tensor->src[0]->type == GGML_TYPE_Q6_K) {
            const ggml_type original_type = tensor->src[0]->type;
            if (original_type == GGML_TYPE_Q6_K) {
                for (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
                    if ((cpu_features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu &&
                        gemm_gemv_kernels_q8[i].lhs_type == tensor->src[1]->type &&
                        gemm_gemv_kernels_q8[i].rhs_type == GGML_TYPE_Q8_0 &&
                        gemm_gemv_kernels_q8[i].op_type == tensor->type) {
                        kernel = &gemm_gemv_kernels_q8[i];
                        break;
                    }
                }
            } else {
                try_table(gemm_gemv_kernels_q8);
            }
        } else if (tensor->src[0]->type == GGML_TYPE_F32) {
            try_table(ggml_kleidiai_kernels_f32);
        } else {
            try_table(gemm_gemv_kernels);
        }
    }

    return kernel;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    for (size_t i = 0; i < NELEMS(gemm_gemv_kernels) - 1; ++i) {
        if ((features & gemm_gemv_kernels[i].required_cpu) == gemm_gemv_kernels[i].required_cpu &&
            gemm_gemv_kernels[i].rhs_type == GGML_TYPE_Q4_0) {
            kernels = &gemm_gemv_kernels[i];
            break;
        }
    }

    return kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    for (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q4_k) - 1; ++i) {
        if ((features & gemm_gemv_kernels_q4_k[i].required_cpu) == gemm_gemv_kernels_q4_k[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q4_k[i];
            break;
        }
    }

    return kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    for (size_t i = 0; i < NELEMS(gemm_gemv_kernels_q8) - 1; ++i) {
        if ((features & gemm_gemv_kernels_q8[i].required_cpu) == gemm_gemv_kernels_q8[i].required_cpu) {
            kernels = &gemm_gemv_kernels_q8[i];
            break;
        }
    }

    return kernels;
}

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    for (size_t i = 0; i < NELEMS(ggml_kleidiai_kernels_f32) - 1; ++i) {
        if ((features & ggml_kleidiai_kernels_f32[i].required_cpu) == ggml_kleidiai_kernels_f32[i].required_cpu) {
            kernels = &ggml_kleidiai_kernels_f32[i];
            break;
        }
    }
    return kernels;
}

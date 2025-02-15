// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

enum cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8
};
inline cpu_feature& operator|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    return lhs;
}
inline cpu_feature operator|(cpu_feature lhs, cpu_feature rhs) {
    return static_cast<cpu_feature>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

struct kernel_info {
    size_t (*get_m_step)(void);
    size_t (*get_n_step)(void);
    size_t (*get_mr)(void);
    size_t (*get_nr)(void);
    size_t (*get_kr)(void);
    size_t (*get_sr)(void);
    size_t (*get_lhs_offset)(size_t m_idx, size_t k, size_t bl);
    size_t (*get_rhs_packed_offset)(size_t n_idx, size_t k, size_t bl);
    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t stride);
    size_t (*get_dst_size)(size_t m, size_t n);
    void (*run_kernel)(size_t m, size_t n, size_t k, size_t bl, const void* lhs_packed, const void* rhs_packed,
                         float* dst, size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max);
};

struct lhs_packing_info {
    size_t (*get_offset)(size_t m_idx, size_t lhs_stride);
    size_t (*get_packed_offset)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);
    size_t (*packed_size)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);
    void (*pack_func)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs,
                      size_t lhs_stride, void* lhs_packed);
    bool require_aligned_m_idx;
};

struct rhs_packing_info {
    size_t (*packed_size)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);
    void (*pack_func)(size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const uint8_t* rhs,
                      const float* bias, void* rhs_packed, size_t extra_bytes, const struct kai_rhs_pack_qs4cxs1s0_param* params);
};

struct ggml_kleidiai_kernels {
    kernel_info gemm;
    kernel_info gemv;
    lhs_packing_info lhs_info;
    rhs_packing_info rhs_info;

    cpu_feature required_cpu;
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features);

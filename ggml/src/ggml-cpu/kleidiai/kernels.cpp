// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

// KleidiAI micro-kernels
#include "kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai_common.h"

#include "kernels.h"

#define NELEMS(x) sizeof(x) / sizeof(*x)
static ggml_kleidiai_kernels gemm_gemv_kernels[] = {
#if defined(__ARM_FEATURE_SME)
    {
        /* SME GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa,
        },
        /* SME GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot,
        },
        /* .lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .get_packed_offset     = */ kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .packed_size           = */ kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon,
            /* .pack_func             = */ kai_run_lhs_quant_pack_qsi8d32p_f32_neon,
        },
        /* .rhs_info = */ {
            /* .packed_size = */ kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon,
            /* .pack_func   = */ kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon,
        },
        /* .required_cpu       = */ CPU_FEATURE_SME,
    },
#endif
#if defined(__APPLE__)
#if defined(__ARM_FEATURE_DOTPROD)
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
        },
        /* .lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset     = */ kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .packed_size           = */ kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32,
            /* .pack_func             = */ kai_run_lhs_quant_pack_qsi8d32p_f32,
        },
        /* .rhs_info = */ {
            /* .packed_size = */ kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0,
            /* .pack_func   = */ kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
    },
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
        },
        /* i8mm GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
        },
        /* .lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset     = */ kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .packed_size           = */ kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32,
            /* .pack_func             = */ kai_run_lhs_quant_pack_qsi8d32p_f32,
        },
        /* .rhs_info = */ {
            /* .packed_size = */ kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0,
            /* .pack_func   = */ kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD | CPU_FEATURE_I8MM,
    },
#endif
#else
#if defined(__ARM_FEATURE_MATMUL_INT8)
    {
        /* i8mm GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
        },
        /* i8mm GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
        },
        /* .lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset     = */ kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .packed_size           = */ kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32,
            /* .pack_func             = */ kai_run_lhs_quant_pack_qsi8d32p_f32,
        },
        /* .rhs_info = */ {
            /* .packed_size = */ kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0,
            /* .pack_func   = */ kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD | CPU_FEATURE_I8MM,
    },
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    {
        /* DOTPROD GEMM */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod,
        },
        /* DOTPROD GEMV */
        /* .kern_info = */ {
            /* .get_m_step            = */ kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_n_step            = */ kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_mr                = */ kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_nr                = */ kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_kr                = */ kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_sr                = */ kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_lhs_offset        = */ kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_rhs_packed_offset = */ kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_offset        = */ kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .get_dst_size          = */ kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
            /* .run_kernel            = */ kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod,
        },
        /* .lhs_info = */ {
            /* .get_offset            = */ kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .get_packed_offset     = */ kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32,
            /* .packed_size           = */ kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32,
            /* .pack_func             = */ kai_run_lhs_quant_pack_qsi8d32p_f32,
        },
        /* .rhs_info = */ {
            /* .packed_size = */ kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0,
            /* .pack_func   = */ kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0,
        },
        /* .required_cpu       = */ CPU_FEATURE_DOTPROD,
    },
#endif
#endif
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature features) {
    ggml_kleidiai_kernels * kernels = nullptr;

    for (size_t i = 0; i < NELEMS(gemm_gemv_kernels); ++i) {
        if ((features & gemm_gemv_kernels[i].required_cpu) == gemm_gemv_kernels[i].required_cpu) {
            kernels = &gemm_gemv_kernels[i];
            break;
        }
    }

    return kernels;
}

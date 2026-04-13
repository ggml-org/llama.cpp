//******************************************************************************
// ET Uberkernel POC
// Executes a recorded sequence of simple ET kernels inside one kernel launch.
//******************************************************************************

#include <stdint.h>

#include "ggml-et-uberkernel-common.h"
#include "ggml-et-uberkernel-kernel-map.h"
#include "platform.h"

#define entry_point el_map_f32_entry
#include "el_map_f32.c"
#undef entry_point

#define entry_point glu_f32_entry
#include "glu_f32.c"
#undef entry_point

#define entry_point unary_f32_entry
#include "unary_f32.c"
#undef entry_point

#define entry_point rope_f32_entry
#include "rope_f32.c"
#undef entry_point

// Clean up ROPE macros
#undef GGML_ROPE_TYPE_NEOX
#undef GGML_ROPE_TYPE_MROPE
#undef GGML_ROPE_TYPE_IMROPE
#undef MAX_ROPE_HALF_DIMS
#undef ROPE_VEC_WIDTH
#undef ROPE_PI
#undef ROPE_TWO_PI
#undef ROPE_PI_OVER_2
#undef ROPE_INV_TWO_PI

#define entry_point rms_norm_f32_entry
#include "rms_norm_f32.c"
#undef entry_point

#define entry_point rms_norm_mul_f32_entry
#include "rms_norm_mul_f32.c"
#undef entry_point

#define entry_point mul_mat_f16_entry
#include "mul_mat_f16.c"
#undef entry_point

#define entry_point mul_mat_f32_entry
#include "mul_mat_f32.c"
#undef entry_point

#define entry_point mul_mat_Q8_0_entry
#include "mul_mat_Q8_0.c"
#undef entry_point

// Clean up Q8_0 macros before matrix engine includes
#undef STRIDE_M
#undef STRIDE_M_KSPLIT
#undef KSPLIT_MIN_K_BLOCKS
#undef KSPLIT_SMALL_ROWS_K_BLOCKS
#undef KSPLIT_MAX_ROWS
#undef TILE_KB
#undef KSPLIT_GROUP_ROWS
#undef SIMPLE_X2_ROWS

// ── Matrix-engine / flash-attn kernels call setup_cache_scp(), which is a
// one-shot operation that hangs if issued twice.  We call the real function
// once at uberkernel entry and suppress all internal calls.
static inline void __attribute__((always_inline))
uberkernel_setup_cache_scp(void) { setup_cache_scp(); }
#define setup_cache_scp() ((void)0)

// ── MUL_MAT: F16 matrix engine (TILE_K=32) ─────────────────────────────────

#define entry_point mul_mat_f16_matrix_engine_entry
#include "mul_mat_f16_matrix_engine.c"
#undef entry_point

// Clean up F16 ME macros (TILE_K=32) before F32 ME redefines them (TILE_K=16)
#undef NUM_COMPUTE_SHIRES
#undef MINIONS_PER_SHIRE
#undef TILE_M
#undef TILE_N
#undef TILE_K
#undef CACHEOP_MAX
#undef REP_RATE
#undef A_L1_START
#undef B_L1_START
#undef SCP_BPANEL_SIZE
#undef SCP_READY_OFF
#undef SCP_CONSUMED_OFF
#undef SCP_PER_MINION

#define entry_point mul_mat_f32_matrix_engine_entry
#include "mul_mat_f32_matrix_engine.c"
#undef entry_point

#undef NUM_COMPUTE_SHIRES
#undef MINIONS_PER_SHIRE
#undef TILE_K
#undef TILE_M
#undef TILE_N
#undef CACHEOP_MAX
#undef REP_RATE

// Evict a contiguous region from both L1 and L2 so subsequent loads fetch
// from L3/DRAM.  Both L1 and L2 are incoherent on ET-SoC-1 (L2 is per-shire),
// so every op must evict its inputs before reading if a prior op in the same
// uberkernel batch may have written to them via fsw.ps or tensor_store.
//
// Handles regions larger than the 16-line hardware limit by issuing multiple
// evict_past_l2 calls.
static void evict_region_past_l2(const void *addr, size_t bytes) {
    if (!addr || bytes == 0) return;

    const uint64_t CL = 64;
    uint64_t base = (uint64_t)addr & ~(CL - 1);
    uint64_t end  = ((uint64_t)addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t nlines = (end - base) / CL;

    FENCE;
    for (uint64_t off = 0; off < nlines; off += 16) {
        uint64_t batch = nlines - off;
        if (batch > 16) batch = 16;
        evict_past_l2((const void *)(base + off * CL), batch, CL);
    }
    WAIT_CACHEOPS;
}

// Compute contiguous byte footprint of a tensor (ne[0..3] * element_size).
static inline size_t tensor_bytes(const struct ggml_tensor *t) {
    return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

int entry_point(struct ggml_et_uberkernel_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || !params) {
        return -1;
    }

    // Enable L1 SCP once upfront — ME kernels need it, and the enable is a
    // one-shot operation that hangs if issued twice.  Non-ME kernels tolerate
    // the halved L1D (performance, not correctness).
    uberkernel_setup_cache_scp();

    struct ggml_et_uberkernel_inst * insts =
        (struct ggml_et_uberkernel_inst *)(uintptr_t) params->insts;
    uint8_t * params_blob = (uint8_t *)(uintptr_t) params->params_blob;

    if (!insts || !params_blob || params->inst_stride < sizeof(struct ggml_et_uberkernel_inst)) {
        return -1;
    }

    for (uint32_t i = 0; i < params->num_insts; ++i) {
        struct ggml_et_uberkernel_inst * inst =
            (struct ggml_et_uberkernel_inst *)((uint8_t *) insts + (i * params->inst_stride));
        void * inst_params = params_blob + inst->params_offset;
        int rc = -1;

        switch (inst->kernel_id) {

            case GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32: {
                struct ggml_et_binary_params *p = (struct ggml_et_binary_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                rc = el_map_f32_entry(p, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_GLU_F32: {
                struct ggml_et_glu_params *p = (struct ggml_et_glu_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                if (p->src1.data) {
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                }
                rc = glu_f32_entry(p, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_UNARY_F32: {
                struct ggml_et_unary_params *p = (struct ggml_et_unary_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = unary_f32_entry(p, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_ROPE_F32: {
                struct ggml_et_rope_params *p = (struct ggml_et_rope_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = rope_f32_entry(p, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32: {
                struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = rms_norm_f32_entry(p, env);
                break;
            }
            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32: {
                struct ggml_et_rms_norm_mul_params *p = (struct ggml_et_rms_norm_mul_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = rms_norm_mul_f32_entry(p, env);
                break;
            }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16: {
                struct ggml_et_binary_params *p = (struct ggml_et_binary_params *) inst_params;
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                rc = mul_mat_f16_entry(p, env);
                break;
            }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE: {
                struct ggml_et_binary_params *p = (struct ggml_et_binary_params *) inst_params;
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                rc = mul_mat_f16_matrix_engine_entry(p, env);
                break;
            }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32: {
                struct ggml_et_binary_params *p = (struct ggml_et_binary_params *) inst_params;
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                rc = mul_mat_f32_entry(p, env);
                break;
            }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE: {
                struct ggml_et_binary_params *p = (struct ggml_et_binary_params *) inst_params;
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                rc = mul_mat_f32_matrix_engine_entry(p, env);
                break;
            }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0: {
                struct ggml_et_binary_params *p = (struct ggml_et_binary_params *) inst_params;
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                rc = mul_mat_Q8_0_entry(p, env);
                break;
            }

            default:
                return -1;
        }

        if (rc != 0) {
            return rc;
        }

        et_barrier(ET_BARRIER_GLOBAL);
    }

    return 0;
}

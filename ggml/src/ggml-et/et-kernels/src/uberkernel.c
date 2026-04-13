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

#define entry_point rms_norm_f32_entry
#include "rms_norm_f32.c"
#undef entry_point

#define entry_point rms_norm_mul_f32_entry
#include "rms_norm_mul_f32.c"
#undef entry_point

#define entry_point unary_f32_entry
#include "unary_f32.c"
#undef entry_point

// Evict a contiguous region from both L1 and L2 so subsequent loads fetch
// from L3/DRAM.  Handles regions larger than the 16-line hardware limit by
// issuing multiple evict_past_l2 calls.
static void evict_region_past_l2(const void *addr, size_t bytes) {
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

int entry_point(struct ggml_et_uberkernel_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || !params) {
        return -1;
    }

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
            case GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                rc = el_map_f32_entry((struct ggml_et_binary_params *) inst_params, env);
                break;
            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32: {
                // Evict src0 from L1+L2 before reduction to avoid stale cache reads.
                // Prior ops may have written src0's memory via fsw.ps (bypasses L1D),
                // and both L1 and L2 are incoherent on ET-SoC-1.
                struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                size_t src0_bytes = (size_t)p->src0.ne[0] * p->src0.ne[1]
                                  * p->src0.ne[2] * p->src0.ne[3] * sizeof(float);
                evict_region_past_l2(p->src0.data, src0_bytes);
                rc = rms_norm_f32_entry(p, env);
                break;
            }
            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32: {
                struct ggml_et_rms_norm_mul_params *p = (struct ggml_et_rms_norm_mul_params *) inst_params;
                size_t src0_bytes = (size_t)p->src0.ne[0] * p->src0.ne[1]
                                  * p->src0.ne[2] * p->src0.ne[3] * sizeof(float);
                evict_region_past_l2(p->src0.data, src0_bytes);
                rc = rms_norm_mul_f32_entry(p, env);
                break;
            }
            case GGML_ET_UBERKERNEL_KERNEL_UNARY_F32:
                rc = unary_f32_entry((struct ggml_et_unary_params *) inst_params, env);
                break;
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

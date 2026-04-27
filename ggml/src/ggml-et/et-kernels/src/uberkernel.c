#include <stdint.h>

#include "ggml-et-uberkernel-common.h"
#include "ggml-et-uberkernel-kernel-map.h"
#include "ggml_tensor.h"
#include "platform.h"


struct ggml_et_glu_params;
struct ggml_et_unary_params;
struct ggml_et_rope_params;
struct ggml_et_rms_norm_params;
struct ggml_et_rms_norm_mul_params;
struct ggml_et_softmax_params;
struct ggml_et_set_rows_params;
struct ggml_et_get_rows_params;
struct ggml_et_cont_params;

extern int el_map_f32_entry(struct ggml_et_binary_params *, void *);
extern int glu_f32_entry(struct ggml_et_glu_params *, void *);
extern int unary_f32_entry(struct ggml_et_unary_params *, void *);
extern int rope_f32_entry(struct ggml_et_rope_params *, void *);
extern int rms_norm_f32_entry(struct ggml_et_rms_norm_params *, void *);
extern int rms_norm_mul_f32_entry(struct ggml_et_rms_norm_mul_params *, void *);
extern int softmax_f32_entry(struct ggml_et_softmax_params *, void *);
extern int set_rows_f32_entry(struct ggml_et_set_rows_params *, void *);
extern int get_rows_f32_entry(struct ggml_et_get_rows_params *, void *);
extern int cont_f32_entry(struct ggml_et_cont_params *, void *);
extern int mul_mat_f16_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f16_matrix_engine_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f32_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f32_matrix_engine_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_Q8_0_entry(struct ggml_et_mm_q8_params *, void *);

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

struct uber_glu_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

struct uber_unary_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct uber_rope_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
};

struct uber_rms_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct uber_rms_norm_mul_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_softmax_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
};

struct uber_set_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_get_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_cont_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};
int entry_point(struct ggml_et_uberkernel_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || !params) {
        return -1;
    }

    // Enable L1 SCP once upfront - _me kernels need it, and the enable is a
    // one-shot operation that hangs if issued twice.
    setup_cache_scp();

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
                struct uber_glu_params *p = (struct uber_glu_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                if (p->src1.data) {
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                }
                rc = glu_f32_entry((struct ggml_et_glu_params *) inst_params, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_UNARY_F32: {
                struct uber_unary_params *p = (struct uber_unary_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = unary_f32_entry((struct ggml_et_unary_params *) inst_params, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_ROPE_F32: {
                struct uber_rope_params *p = (struct uber_rope_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = rope_f32_entry((struct ggml_et_rope_params *) inst_params, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32: {
                struct uber_rms_norm_params *p = (struct uber_rms_norm_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = rms_norm_f32_entry((struct ggml_et_rms_norm_params *) inst_params, env);
                break;
            }
            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32: {
                struct uber_rms_norm_mul_params *p = (struct uber_rms_norm_mul_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = rms_norm_mul_f32_entry((struct ggml_et_rms_norm_mul_params *) inst_params, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32: {
                struct uber_softmax_params *p = (struct uber_softmax_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                if (p->src1.data) {
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                }
                if (p->src2.data) {
                    evict_region_past_l2(p->src2.data, tensor_bytes(&p->src2));
                }
                rc = softmax_f32_entry((struct ggml_et_softmax_params *) inst_params, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32: {
                struct uber_set_rows_params *p = (struct uber_set_rows_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                rc = set_rows_f32_entry((struct ggml_et_set_rows_params *) inst_params, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_GET_ROWS_F32: {
                struct uber_get_rows_params *p = (struct uber_get_rows_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                rc = get_rows_f32_entry((struct ggml_et_get_rows_params *) inst_params, env);
                break;
            }

            case GGML_ET_UBERKERNEL_KERNEL_CONT_F32: {
                struct uber_cont_params *p = (struct uber_cont_params *) inst_params;
                evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                rc = cont_f32_entry((struct ggml_et_cont_params *) inst_params, env);
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
                struct ggml_et_mm_q8_params *p = (struct ggml_et_mm_q8_params *) inst_params;
                evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                if (p->bias.data) {
                    evict_region_past_l2(p->bias.data, tensor_bytes(&p->bias));
                }
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

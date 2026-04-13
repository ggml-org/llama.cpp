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

#define entry_point unary_f32_entry
#include "unary_f32.c"
#undef entry_point

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

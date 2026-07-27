#pragma once

#include <stdint.h>

struct ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_offset;
    uint32_t params_size;
};

struct ggml_et_uberkernel_params {
    uint32_t num_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
};

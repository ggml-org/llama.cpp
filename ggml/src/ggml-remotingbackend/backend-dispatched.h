#pragma once

#include <cstdint>
#include <cstddef>

#include <ggml-backend.h>

#include "backend-utils.h"
#include "shared/venus_cs.h"
#include "shared/apir_backend.h"

uint32_t backend_dispatch_initialize(void *ggml_backend_reg_fct_p);

typedef uint32_t (*backend_dispatch_t)(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec);

/* *** */

uint32_t backend_reg_get_device_count(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec);

static inline const char *backend_dispatch_command_name(ApirBackendCommandType type)
{
    switch (type) {
    case APIR_COMMAND_TYPE_GET_DEVICE_COUNT: return "backend_reg__get_device_count";
    default: return "unknown";
    }
}

static const backend_dispatch_t apir_backend_dispatch_table[APIR_BACKEND_DISPATCH_TABLE_COUNT] = {
    [APIR_COMMAND_TYPE_GET_DEVICE_COUNT] = backend_reg_get_device_count,
};

#pragma once

#include <cstdint>
#include <cstddef>

#include <ggml-backend.h>

#include "backend-utils.h"
#include "backend-convert.h"
#include "shared/apir_backend.h"
#include "shared/apir_cs.h"
#include "shared/apir_cs_ggml.h"


typedef uint32_t (*backend_dispatch_t)(apir_encoder *       enc,
                                       apir_decoder *       dec,
                                       virgl_apir_context * ctx);

#include "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(void * ggml_backend_reg_fct_p, void * ggml_backend_init_fct_p);

#pragma once

#include <cstdint>
#include <cstddef>

#include <ggml-backend.h>

#include "backend-convert.h"
#include "backend-virgl-apir.h"
#include "shared/apir_backend.h"
#include "shared/apir_cs.h"
#include "shared/apir_cs_ggml.h"

struct virgl_apir_context {
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

typedef uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

#include "backend-dispatched.gen.h"

uint32_t backend_dispatch_initialize(void * ggml_backend_reg_fct_p);

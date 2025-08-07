#pragma once

#include "traits.h"
#include "ggml.h"
#include <riscv_vector.h>

// #include <cstddef>
// GGML internal header
ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(void);
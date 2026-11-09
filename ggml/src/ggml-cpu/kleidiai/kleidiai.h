// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(void);

const struct ggml_backend_weight_cache_i * ggml_backend_cpu_weight_cache_get_interface(void);

#ifdef  __cplusplus
}
#endif

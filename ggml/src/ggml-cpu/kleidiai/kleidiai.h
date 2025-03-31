// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(void);

#ifdef  __cplusplus
}
#endif

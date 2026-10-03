// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-alloc.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(void);

struct ggml_kleidiai_sme2_flash_attn_workspace {
    size_t work_size;
    size_t thread_size;
};

bool ggml_kleidiai_sme2_flash_attn_get_workspace(const struct ggml_tensor * op,
                                                 struct ggml_kleidiai_sme2_flash_attn_workspace * workspace);

bool ggml_kleidiai_sme2_flash_attn_prepare_q(const float * q,
                                             size_t        m,
                                             size_t        k,
                                             size_t        dv,
                                             void *        work_data,
                                             size_t        work_size);

bool ggml_kleidiai_sme2_flash_attn_qk(float *       dst,
                                      const float * rhs,
                                      size_t        m,
                                      size_t        n,
                                      size_t        k,
                                      void *        work_data,
                                      size_t        work_size);

bool ggml_kleidiai_sme2_flash_attn_av(float *       dst,
                                      const float * lhs,
                                      const float * rhs,
                                      size_t        m,
                                      size_t        n,
                                      size_t        k,
                                      void *        work_data,
                                      size_t        work_size);

#ifdef  __cplusplus
}
#endif

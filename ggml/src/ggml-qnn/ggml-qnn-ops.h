/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#pragma once

#include "ggml-qnn-impl.h"
void ggml_qnn_general_node(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_mul_mat(ggml_backend_qnn_context * ctx, ggml_tensor * dst);

void ggml_qnn_repeat(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_div(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_leaky_relu(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_concat(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_arange(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_sqr(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_clamp(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_scale(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_argsort(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_group_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_acc(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_sum_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_upsample_nearest2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_pad(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_pool2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_dup(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_rms_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_diag_mask(ggml_backend_qnn_context * ctx, ggml_tensor * dst, float value);
void ggml_qnn_im2col(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_timestep_embedding(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_cpy(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_softmax(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_get_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
void ggml_qnn_rope(ggml_backend_qnn_context * ctx, ggml_tensor * dst);

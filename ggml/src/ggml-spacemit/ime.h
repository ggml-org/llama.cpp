#pragma once

#include "ggml-alloc.h"
#include "spacemit-context.h"

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_optimal_repack_type(const ggml_tensor * cur);
const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_tensor_traits(const ggml_tensor * op);
bool ggml_spacemit_get_work_size(int n_threads, const ggml_tensor * op, size_t * size);
bool ggml_spacemit_compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op);
int ggml_riscv64_spacemit_repack_tensor(ggml_tensor * tensor, const void * data, size_t size);
void ggml_riscv64_spacemit_unpack_tensor(const ggml_tensor * tensor, void * data, size_t size);
size_t ggml_spacemit_nbytes(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);

#ifndef GGML_SYCL_FATTN_SPARSE_HPP
#define GGML_SYCL_FATTN_SPARSE_HPP

#include "common.hpp"

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Reports whether ggml_sycl_flash_attn_ext_sparse() would actually take the sparse path for
// this node (enabled and applicable), without running it. The sparse kernel calls a blocking
// stream->wait(), which is not legal while a SYCL graph is recording.
bool ggml_sycl_flash_attn_ext_sparse_applicable(const ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_SPARSE_HPP

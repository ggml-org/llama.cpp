#ifndef GGML_SYCL_QSA_SCORE_HPP
#define GGML_SYCL_QSA_SCORE_HPP

#include "common.hpp"

bool ggml_sycl_can_fuse_qsa_score(const ggml_cgraph * cgraph, int node_idx);
int  ggml_sycl_qsa_score_absorbs(const ggml_cgraph * cgraph, int node_idx);
int  ggml_sycl_fuse_qsa_score(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_QSA_SCORE_HPP

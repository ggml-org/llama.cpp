#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#include "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

void ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      const ggml_tensor       * src,
                      const int64_t           * ne,
                      const zdnn_data_layouts   layout);

void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, void * buffer);

int64_t ggml_zdnn_q_group_size(const ggml_tensor * tensor);

void ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);

void ggml_zdnn_load_weights(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);

bool ggml_zdnn_ztensor_in_alloc(const ggml_tensor * tensor);

void ggml_zdnn_q_weights_desc(zdnn_tensor_desc * pre_tfm_desc,
                              zdnn_tensor_desc * tfm_desc,
                        const ggml_tensor      * tensor);

void ggml_zdnn_init_tensor_desc(zdnn_tensor_desc & pre_tfm_desc,
                                zdnn_tensor_desc & tfm_desc,
                          const ggml_tensor      * tensor);

// how quantized weights are stacked in their ztensor, see ggml_zdnn_q_weights_desc
struct ggml_zdnn_q_layout {
    int64_t n_stacks;  // n_groups per matrix, times the matrices along ne2
    int64_t n_groups;  // groups along ne0 of each matrix
    int64_t gs;        // values per group, all of ne0 when not stacked
};

ggml_zdnn_q_layout ggml_zdnn_q_weights_layout(const ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);

void ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP

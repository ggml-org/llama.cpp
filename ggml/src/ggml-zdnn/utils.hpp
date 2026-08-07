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

void ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP

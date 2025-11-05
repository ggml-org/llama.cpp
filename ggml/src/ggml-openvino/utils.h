#include "ggml-backend-impl.h"
#include "ggml-decoder.h"
#include "ggml-impl.h"

#include <algorithm>
#include <openvino/runtime/core.hpp>

enum ggml_status openvino_frontend_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);

size_t checksum(const void * data, size_t size);

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor);

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, void * output_dst);

template <typename T>
std::vector<T> pad_input(const ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_value) {
    std::vector<T> padded_data(padded_rows * padded_cols, pad_value);
    size_t rows = tensor->ne[1];
    size_t cols = tensor->ne[0];
    T * data = static_cast<T *>(tensor->data);

    for (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        for (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded_data[i * padded_cols + j] = data[i * cols + j];
        }
    }
    return padded_data;
}

void set_zero_diagonal(std::vector<float> & matrix, size_t dim);

const ggml_tensor * get_inp_pos_tensor(struct ggml_cgraph * cgraph);

bool get_is_first_token(const ggml_tensor * inp_pos);

ov::AnyMap get_ov_compile_config(const std::string & device);

std::map<ggml_type, ExtraQuantType> get_types_to_requant(const std::string & device);

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name);
ov::Tensor get_ov_input_tensor_static(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                      const std::string & param_name,
                                      int j,
                                      int input_len);

ov::Tensor get_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & result_name);

bool is_naive(struct ggml_cgraph * cgraph);

enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config);

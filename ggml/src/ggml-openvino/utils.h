#include "ggml-backend-impl.h"
#include "ggml-decoder.h"

enum ggml_status openvino_frontend_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph);

size_t checksum(const void* data, size_t size);

void print_input_tensor_info(const std::string& name, const ov::Tensor& tensor);

void print_output_tensor_info(const std::string& name,
                              const ov::Tensor& tensor,
                              std::map<std::string, void*>& output_dst);
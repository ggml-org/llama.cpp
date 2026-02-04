#include "ggml-backend-impl.h"
#include "ggml-decoder.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstddef>
#include <openvino/runtime/core.hpp>
#include <string>

struct graph_key {
    int n_nodes;
    std::string first_node_name;
    std::string last_node_name;

    graph_key(const ggml_cgraph * cgraph) : n_nodes(cgraph->n_nodes) {
        if (n_nodes > 0) {
            first_node_name = cgraph->nodes[0]->name;
            last_node_name = cgraph->nodes[n_nodes - 1]->name;
        }
    }

    bool operator==(const graph_key & other) const {
        return n_nodes == other.n_nodes && first_node_name == other.first_node_name &&
               last_node_name == other.last_node_name;
    }
};

struct graph_key_hash {
    size_t operator()(const graph_key & key) const {
        size_t h = std::hash<int>{}(key.n_nodes);
        if (key.n_nodes > 0) {
            h ^= std::hash<std::string>{}(key.first_node_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>{}(key.last_node_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

enum ggml_status ov_graph_compute(struct ggml_cgraph * cgraph);

enum ggml_status ov_graph_compute_dynamic(struct ggml_cgraph * cgraph, const std::string & device, bool stateful = false);
enum ggml_status ov_graph_compute_static(struct ggml_cgraph * cgraph);

size_t checksum(const void * data, size_t size);

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor);

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst);

template <typename T>
std::vector<T> pad_input(const T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_value) {
    std::vector<T> padded(padded_rows * padded_cols, pad_value);

    for (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        for (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    return padded;
}

template <typename T>
std::vector<T> pad_input(const ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_value) {
    return pad_input<T>(reinterpret_cast<const T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_value);
}

void set_zero_diagonal(std::vector<float> & matrix, size_t rows, size_t cols);

const ggml_tensor * get_inp_pos_tensor(struct ggml_cgraph * cgraph);

bool get_is_prefill(const ggml_tensor * inp_pos);

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index);

ov::Tensor get_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & result_name);

bool is_naive(struct ggml_cgraph * cgraph);

enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config);

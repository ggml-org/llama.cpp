#include "ggml-decoder.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct graph_key {
    int n_nodes;
    std::string first_node_name;
    std::string last_node_name;
    std::vector<std::string> input_src_names;

    graph_key(const ggml_cgraph * cgraph) : n_nodes(cgraph->n_nodes) {
        if (n_nodes > 0) {
            first_node_name = cgraph->nodes[0]->name;
            last_node_name = cgraph->nodes[n_nodes - 1]->name;
        }

        auto get_input_key_name = [](const ggml_cgraph * graph, const ggml_tensor * tensor) {
            std::string name = tensor->name;
            const size_t hash_pos = ggml_hash_find(&graph->visited_hash_set, tensor);
            if (((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || GgmlOvDecoder::is_kvcache(tensor, nullptr)) &&
                hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(graph->visited_hash_set.used, hash_pos)) {
                name += "#" + std::to_string(hash_pos);
            }
            return name;
        };

        std::vector<std::string> node_names;
        node_names.reserve(cgraph->n_nodes);
        for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
            node_names.emplace_back(cgraph->nodes[node_idx]->name);
        }

        for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
            const ggml_tensor * node = cgraph->nodes[node_idx];
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                const ggml_tensor * src = node->src[src_idx];
                if (src == nullptr || src->name[0] == '\0') {
                    continue;
                }

                const std::string src_name = get_input_key_name(cgraph, src);
                if (std::find(node_names.begin(), node_names.end(), src_name) != node_names.end()) {
                    continue;
                }
                if (src_name.find("weight") != std::string::npos) {
                    continue;
                }

                input_src_names.push_back(std::to_string(node_idx) + ":" + std::to_string(src_idx) + ":" + src_name);
            }
        }
    }

    bool operator==(const graph_key & other) const {
        return n_nodes == other.n_nodes && first_node_name == other.first_node_name &&
               last_node_name == other.last_node_name && input_src_names == other.input_src_names;
    }
};

struct graph_key_hash {
    size_t operator()(const graph_key & key) const {
        size_t hash = std::hash<int>{}(key.n_nodes);
        if (key.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(key.first_node_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(key.last_node_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        for (const auto & input_src_name : key.input_src_names) {
            hash ^= std::hash<std::string>{}(input_src_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

struct ov_runtime_context {
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache_prefill;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache;
    //TODO: Stateful is only supported for single request at a time.
    //      Simultanous stateful inference request support to be added.
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_input_name_map;
    std::atomic<int> backend_count;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0), backend_count(0) {}

    void clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_input_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_input_name_map.clear();
        stateful_kv_size = 0;
    }

    void clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

enum ggml_status ov_graph_compute(struct ggml_cgraph * cgraph, ggml_backend_t backend);

enum ggml_status ov_graph_compute_dynamic(struct ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);
enum ggml_status ov_graph_compute_static(struct ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);

size_t checksum(const void * data, size_t size);

bool save_ggml_tensor_data_to_txt(const ggml_tensor * tensor, const std::string & file_path);

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

const ggml_tensor * get_inp_pos_tensor(struct ggml_cgraph * cgraph);

bool get_is_prefill(const ggml_tensor * inp_pos);

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index);

ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                   std::shared_ptr<ov::InferRequest> infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor);

bool is_naive(struct ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(struct ggml_cgraph * cgraph);

enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config);

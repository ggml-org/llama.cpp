#include "ggml-decoder.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
struct graph_key {
    int n_nodes;
    int n_leaves;
    std::string first_node_name;
    std::string last_node_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> input_srcs;

    graph_key(const ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        if (n_nodes > 0) {
            first_node_name = cgraph->nodes[0]->name;
            last_node_name = cgraph->nodes[n_nodes - 1]->name;
        }

        static const bool full_key = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        if (!full_key && !include_inputs) {
            return;
        }

        std::unordered_set<const ggml_tensor *> node_set;
        node_set.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            node_set.insert(cgraph->nodes[i]);
        }

        for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
            const ggml_tensor * node = cgraph->nodes[node_idx];
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                const ggml_tensor * src = node->src[src_idx];
                if (src == nullptr || src->buffer == nullptr || node_set.count(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    continue;
                }

                input_srcs.push_back(std::to_string(node_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(node));
            }
        }
    }

    bool operator==(const graph_key & other) const {
        return n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_node_name == other.first_node_name &&
               last_node_name == other.last_node_name && input_srcs == other.input_srcs;
    }
};

struct graph_key_hash {
    size_t operator()(const graph_key & key) const {
        size_t hash = std::hash<int>{}(key.n_nodes);
        hash ^= std::hash<int>{}(key.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        if (key.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(key.first_node_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(key.last_node_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        for (const auto & s : key.input_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

struct ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
struct ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_count = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
struct ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache_prefill;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_input_name_map;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

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

int64_t get_inp_pos_n_tokens(struct ggml_cgraph * cgraph, const ggml_tensor * inp_pos);

bool get_is_prefill(struct ggml_cgraph * cgraph, const ggml_tensor * inp_pos);

bool is_naive(struct ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(struct ggml_cgraph * cgraph);

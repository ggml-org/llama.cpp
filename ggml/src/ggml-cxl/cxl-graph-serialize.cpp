#include "cxl-graph-serialize.h"
#include "ggml-impl.h"

#include <unordered_map>
#include <unordered_set>
#include <cstring>

// Collect all unique tensors reachable from the graph nodes
static void collect_tensors(const ggml_cgraph * cgraph,
                            std::vector<const ggml_tensor *> & tensors,
                            std::unordered_map<const ggml_tensor *, uint32_t> & tensor_idx) {
    std::unordered_set<const ggml_tensor *> visited;

    // DFS to collect all tensors
    std::vector<const ggml_tensor *> stack;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        stack.push_back(cgraph->nodes[i]);
    }

    while (!stack.empty()) {
        const ggml_tensor * t = stack.back();
        stack.pop_back();

        if (!t || visited.count(t)) {
            continue;
        }
        visited.insert(t);

        uint32_t idx = (uint32_t)tensors.size();
        tensors.push_back(t);
        tensor_idx[t] = idx;

        // Visit sources
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (t->src[j]) {
                stack.push_back(t->src[j]);
            }
        }

        // Visit view source
        if (t->view_src) {
            stack.push_back(t->view_src);
        }
    }
}

static void serialize_tensor(const ggml_tensor * t,
                             cxl_serialized_tensor * out) {
    memset(out, 0, sizeof(*out));

    out->id   = (uint64_t)(uintptr_t)t;
    out->type = (uint32_t)t->type;

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        out->ne[i] = (uint32_t)t->ne[i];
        out->nb[i] = (uint32_t)t->nb[i];
    }

    out->op    = (uint32_t)t->op;
    out->flags = t->flags;

    memcpy(out->op_params, t->op_params, sizeof(out->op_params));

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (t->src[i]) {
            out->src[i] = (uint64_t)(uintptr_t)t->src[i];
        } else {
            out->src[i] = 0;
        }
    }

    if (t->view_src) {
        out->view_src  = (uint64_t)(uintptr_t)t->view_src;
        out->view_offs = (uint64_t)t->view_offs;
    }

    // data points to device memory (the opaque device pointer)
    out->data = (uint64_t)(uintptr_t)t->data;

    memcpy(out->name, t->name, GGML_MAX_NAME);
}

bool cxl_graph_serialize(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    if (!cgraph || cgraph->n_nodes == 0) {
        return false;
    }

    // Collect all unique tensors
    std::vector<const ggml_tensor *> tensors;
    std::unordered_map<const ggml_tensor *, uint32_t> tensor_idx;
    collect_tensors(cgraph, tensors, tensor_idx);

    // Calculate total output size
    size_t total_size = sizeof(cxl_graph_header)
                      + tensors.size() * sizeof(cxl_serialized_tensor)
                      + cgraph->n_nodes * sizeof(uint64_t);

    output.resize(total_size);
    uint8_t * ptr = output.data();

    // Write header
    cxl_graph_header header;
    header.magic     = CXL_GRAPH_MAGIC;
    header.version   = CXL_GRAPH_VERSION;
    header.n_nodes   = (uint32_t)cgraph->n_nodes;
    header.n_tensors = (uint32_t)tensors.size();
    memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    // Write tensor table
    for (const ggml_tensor * t : tensors) {
        cxl_serialized_tensor st;
        serialize_tensor(t, &st);
        memcpy(ptr, &st, sizeof(st));
        ptr += sizeof(st);
    }

    // Write node list (tensor IDs for each compute node)
    for (int i = 0; i < cgraph->n_nodes; i++) {
        uint64_t id = (uint64_t)(uintptr_t)cgraph->nodes[i];
        memcpy(ptr, &id, sizeof(id));
        ptr += sizeof(id);
    }

    return true;
}

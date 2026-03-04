#pragma once

// Graph serialization for CXL backend
// Serializes GGML compute graphs into a binary format that can be sent
// to the CXL device's host-side GPU backend for execution.

#include "ggml.h"

#include <vector>
#include <cstdint>

// Serialization format:
//
// [Header]
//   uint32_t magic       = 0x47475347 ("GGSG" - GGML Graph Serialization)
//   uint32_t version     = 1
//   uint32_t n_nodes     = number of compute nodes
//   uint32_t n_tensors   = total number of unique tensors (nodes + sources)
//
// [Tensor Table] (n_tensors entries)
//   For each tensor:
//     uint64_t id         = tensor pointer (used as unique ID)
//     uint32_t type       = ggml_type
//     uint32_t ne[4]      = shape
//     uint32_t nb[4]      = strides (byte offsets)
//     uint32_t op         = ggml_op
//     int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]
//     int32_t  flags
//     uint64_t src[GGML_MAX_SRC]  = source tensor IDs (0 if none)
//     uint64_t view_src   = view source tensor ID
//     uint64_t view_offs  = view offset
//     uint64_t data       = device memory pointer for tensor data
//     char     name[GGML_MAX_NAME]
//
// [Node List] (n_nodes entries)
//   uint64_t tensor_id   = tensor ID for each compute node

#define CXL_GRAPH_MAGIC   0x47475347  // "GGSG"
#define CXL_GRAPH_VERSION 1

#pragma pack(push, 1)
struct cxl_graph_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_nodes;
    uint32_t n_tensors;
};

struct cxl_serialized_tensor {
    uint64_t id;
    uint32_t type;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];
    char     padding[4]; // align to 8 bytes
};
#pragma pack(pop)

// Serialize a compute graph into a byte buffer
// Returns true on success
bool cxl_graph_serialize(const ggml_cgraph * cgraph, std::vector<uint8_t> & output);

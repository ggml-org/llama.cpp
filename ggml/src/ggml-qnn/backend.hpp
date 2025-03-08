
#pragma once

#ifndef NDEBUG
#    include <atomic>
#endif

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ggml-backend.h"
#include "ggml-qnn.h"
#include "ggml.h"
#include "graph.hpp"
#include "qnn-lib.hpp"

namespace qnn {
typedef std::unordered_map<std::string, std::unique_ptr<qnn::qnn_graph>> qnn_graph_cache_t;
}  // namespace qnn

struct ggml_backend_qnn_device_context {
    // initialize in constructor
    QNNBackend  device;
    size_t      threads;
    std::string name;
    std::string lib_name;

    // initialize in qnn init
    qnn::qcom_socinfo                   socinfo = {};
    uint64_t                            supported_types;
    std::shared_ptr<qnn::qnn_instance>  instance;
    std::shared_ptr<qnn::qnn_interface> qnn_interface;

    qnn::qnn_graph_cache_t qnn_graph_cache;

#ifndef NDEBUG
    std::atomic_uint32_t supported_op_count   = 0;
    std::atomic_uint32_t unsupported_op_count = 0;
#endif

    explicit ggml_backend_qnn_device_context(QNNBackend device, size_t threads, const char * name,
                                             const char * lib_name, uint64_t supported_types) :
        device(device),
        threads(threads),
        name(name),
        lib_name(lib_name),
        supported_types(supported_types) {}
};

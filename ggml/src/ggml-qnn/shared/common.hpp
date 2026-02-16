#pragma once

#include <cstdint>
#include <memory>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

enum backend_index_type {
    QNN_BACKEND_CPU = 0,
    QNN_BACKEND_GPU,
    QNN_BACKEND_NPU,

    HEXAGON_BACKEND,

    TOTAL_BACKEND_COUNT,
    QNN_BACKEND_COUNT = HEXAGON_BACKEND,
};

class backend_device_proxy {
  public:
    virtual ~backend_device_proxy() = default;

    virtual const ggml_backend_device_i & get_iface() const = 0;
    virtual void *                        get_context()     = 0;
};

using backend_device_proxy_ptr = std::shared_ptr<backend_device_proxy>;

backend_device_proxy_ptr create_qnn_backend_context(backend_index_type device);
backend_device_proxy_ptr create_hexagon_backend_context(backend_index_type device);

namespace common {

size_t get_system_total_memory_in_bytes();
size_t get_system_free_memory_in_bytes();

}  // namespace common

#define DISABLE_COPY(class_name)                 \
    class_name(const class_name &)     = delete; \
    void operator=(const class_name &) = delete

#define DISABLE_MOVE(class_name)            \
    class_name(class_name &&)     = delete; \
    void operator=(class_name &&) = delete

#define DISABLE_COPY_AND_MOVE(class_name) \
    DISABLE_COPY(class_name);             \
    DISABLE_MOVE(class_name)

#define LOG_ERROR(...) (GGML_LOG_ERROR(__VA_ARGS__))
#define LOG_WARN(...)  (GGML_LOG_WARN(__VA_ARGS__))
#define LOG_INFO(...)  (GGML_LOG_INFO(__VA_ARGS__))

#ifndef NDEBUG
#    define LOG_DEBUG(...) (GGML_LOG_DEBUG(__VA_ARGS__))
#else
#    define LOG_DEBUG(...)
#endif

#pragma once

#include <HAP_farf.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "hexagon_npu.h"

#define DEVICE_LOG_ERROR(...) FARF(FATAL, __VA_ARGS__)
#define DEVICE_LOG_WARN(...)  FARF(ERROR, __VA_ARGS__)
#define DEVICE_LOG_INFO(...)  FARF(HIGH, __VA_ARGS__)

#ifdef _DEBUG
#    undef FARF_LOW
#    define FARF_LOW              1
#    define DEVICE_LOG_DEBUG(...) FARF(LOW, __VA_ARGS__)
#else
#    define DEVICE_LOG_DEBUG(...) (void) 0
#endif

// TODO: reuse the declaration at host
#define DISABLE_COPY(class_name)                 \
    class_name(const class_name &)     = delete; \
    void operator=(const class_name &) = delete

#define DISABLE_MOVE(class_name)            \
    class_name(class_name &&)     = delete; \
    void operator=(class_name &&) = delete

#define DISABLE_COPY_AND_MOVE(class_name) \
    DISABLE_COPY(class_name);             \
    DISABLE_MOVE(class_name)

#define NPU_UNUSED(x) (void) (x)

namespace hexagon {

inline constexpr const char * op_get_name(npu_device_tensor_op op) {
    switch (op) {
        case NPU_OP_MUL_MAT:
            return "MUL_MAT";
        case NPU_OP_ADD:
            return "ADD";
        case NPU_OP_SUB:
            return "SUB";
        case NPU_OP_MUL:
            return "MUL";
        default:
            return "UNKNOWN";
    }
}

inline constexpr std::pair<int64_t, int64_t> get_thread_work_slice(int64_t total, size_t tidx, size_t tcnt) {
    const auto elements_per_thread = (total + tcnt - 1) / tcnt;
    const auto start               = tidx * elements_per_thread;
    const auto end                 = std::min<int64_t>(start + elements_per_thread, total);
    return { start, end };
}

}  // namespace hexagon

#pragma once

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <cstdint>
#include <cstring>
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

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING

template <size_t _buffer_count> class npu_scoped_timer {
  public:
    enum { kBufferCount = _buffer_count };

    explicit npu_scoped_timer(const char * log_prefix, const char * sub_proc_log_prefix) {
        strncpy(_log_prefix, log_prefix, kBufferCount - 1);
        if (sub_proc_log_prefix != nullptr) {
            strncpy(_sub_proc_log_prefix, sub_proc_log_prefix, kBufferCount - 1);
        }

        _begin_cycles  = HAP_perf_get_qtimer_count();
        _begin_pcycles = HAP_perf_get_pcycles();
    }

    npu_scoped_timer(npu_scoped_timer && other) { *this = std::move(other); }

    ~npu_scoped_timer() { print(); }

    void operator=(npu_scoped_timer && other) {
        strncpy(_log_prefix, other._log_prefix, kBufferCount - 1);
        strncpy(_sub_proc_log_prefix, other._sub_proc_log_prefix, kBufferCount - 1);
        _begin_cycles    = other._begin_cycles;
        _sub_proc_cycles = other._sub_proc_cycles;
        _sub_proc_count  = other._sub_proc_count;
    }

    void add_sub_proc_cycles(uint64_t cycles, uint64_t pcycles) {
        _sub_proc_cycles += cycles;
        _sub_proc_pcycles += pcycles;
        _sub_proc_count++;
    }

    void print() const {
        auto total_cycles  = HAP_perf_get_qtimer_count() - _begin_cycles;
        auto total_pcycles = HAP_perf_get_pcycles() - _begin_pcycles;
        auto duration      = HAP_perf_qtimer_count_to_us(total_cycles);

        if (_sub_proc_count > 0) {
            auto sub_proc_duration = HAP_perf_qtimer_count_to_us(_sub_proc_cycles);
            DEVICE_LOG_WARN("[profiler]%s, pcyc: %llu, dur: %lluus, [%s]cnt: %llu, pcyc: %llu, dur: %lluus\n",
                            _log_prefix, total_pcycles, duration, _sub_proc_log_prefix, _sub_proc_count,
                            _sub_proc_pcycles, sub_proc_duration);
        } else {
            DEVICE_LOG_WARN("[profiler]%s, pcyc: %llu, dur: %lluus\n", _log_prefix, total_pcycles, duration);
        }
    }

  private:
    char     _log_prefix[kBufferCount]          = {};
    char     _sub_proc_log_prefix[kBufferCount] = {};
    uint64_t _begin_cycles                      = 0;
    uint64_t _begin_pcycles                     = 0;
    uint64_t _sub_proc_cycles                   = 0;
    uint64_t _sub_proc_pcycles                  = 0;
    uint64_t _sub_proc_count                    = 0;

    DISABLE_COPY(npu_scoped_timer);
};

template <size_t _buffer_count> class npu_sub_process_scoped_timer {
  public:
    using npu_scoped_timer = npu_scoped_timer<_buffer_count>;

    explicit npu_sub_process_scoped_timer(npu_scoped_timer & timer) : _timer(timer) {
        _begin_cycles  = HAP_perf_get_qtimer_count();
        _begin_pcycles = HAP_perf_get_pcycles();
    }

    ~npu_sub_process_scoped_timer() {
        _timer.add_sub_proc_cycles(HAP_perf_get_qtimer_count() - _begin_cycles,
                                   HAP_perf_get_pcycles() - _begin_pcycles);
    }

  private:
    npu_scoped_timer & _timer;
    uint64_t           _begin_cycles  = 0;
    uint64_t           _begin_pcycles = 0;

    DISABLE_COPY_AND_MOVE(npu_sub_process_scoped_timer);
};

inline auto make_scoped_perf_timer(const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[512];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return npu_scoped_timer<512>(buffer, nullptr);
}

#endif

}  // namespace hexagon

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
#    define DEVICE_SCOPED_PERFORMANCE_TRACKER(fmt, ...) \
        auto __npu_timer_##__LINE__ = hexagon::make_scoped_perf_timer(fmt, __VA_ARGS__)
#else
#    define DEVICE_SCOPED_PERFORMANCE_TRACKER(fmt, ...) ((void) 0)
#endif

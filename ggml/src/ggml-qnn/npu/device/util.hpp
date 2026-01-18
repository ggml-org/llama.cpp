#pragma once

#include "hexagon_npu.h"

#include <AEEStdDef.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <qurt.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

#define DEVICE_LOG_ERROR(...) hexagon::log_error(__VA_ARGS__)
#define DEVICE_LOG_WARN(...)  hexagon::log_message(__VA_ARGS__)
#define DEVICE_LOG_INFO(...)  hexagon::log_message(__VA_ARGS__)

#ifdef _DEBUG
#    define DEVICE_LOG_DEBUG(...) hexagon::log_message(__VA_ARGS__)
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

__attribute__((format(printf, 1, 2))) inline void log_error(const char * format, ...) {
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

__attribute__((format(printf, 1, 2))) inline void log_message(const char * format, ...) {
    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
}

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
        case NPU_OP_RMS_NORM:
            return "RMS_NORM";
        case NPU_OP_FLASH_ATTN:
            return "FLASH_ATTN_EXT";
        case NPU_OP_ROPE:
            return "ROPE";
        case NPU_OP_GLU:
            return "GLU";
        case NPU_OP_GET_ROWS:
            return "GET_ROWS";
        case NPU_OP_SET_ROWS:
            return "SET_ROWS";
        case NPU_OP_CPY:
            return "CPY";
        default:
            return "UNKNOWN";
    }
}

inline bool is_transposed_or_permuted(const npu_device_nb_type & nb) {
    // Check if the tensor is transposed or permuted
    return (nb[0] > nb[1]) || (nb[1] > nb[2]) || (nb[2] > nb[3]);
}

inline bool is_same_shape(const npu_device_ne_type & src, const npu_device_ne_type & dst) {
    for (size_t i = 0; i < DEVICE_TENSOR_MAX_DIMS; ++i) {
        if (src[i] != dst[i]) {
            return false;
        }
    }

    return true;
}

inline bool is_same_shape(const npu_device_tensor_spec & src, const npu_device_tensor_spec & dst) {
    return is_same_shape(src.ne, dst.ne);
}

class qurt_mutex {
  public:
    qurt_mutex() { qurt_mutex_init(&_mutex); }

    ~qurt_mutex() { qurt_mutex_destroy(&_mutex); }

    void lock() { qurt_mutex_lock(&_mutex); }

    void unlock() { qurt_mutex_unlock(&_mutex); }

  private:
    qurt_mutex_t _mutex;

    DISABLE_COPY_AND_MOVE(qurt_mutex);
};

class power_utils {
  public:
    power_utils() {
        _context_ptr = HAP_utils_create_context();
        if (_context_ptr == nullptr) {
            DEVICE_LOG_ERROR("Failed to create power context\n");
        }
    }

    ~power_utils() {
        if (_context_ptr != nullptr) {
            HAP_utils_destroy_context(_context_ptr);
        }
    }

    unsigned int get_clock_speed_hz() const {
        if (!is_valid()) {
            DEVICE_LOG_ERROR("Power context is not initialized\n");
            return 0;
        }

        HAP_power_response_t response = {};
        response.type                 = HAP_power_get_clk_Freq;
        auto ret                      = HAP_power_get(_context_ptr, &response);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to get clock speed: %d\n", ret);
            return 0;
        }

        return response.clkFreqHz;
    }

    bool get_dvcs_enabled() const {
        if (!is_valid()) {
            DEVICE_LOG_ERROR("Power context is not initialized\n");
            return false;
        }

        HAP_power_response_t response = {};
        response.type                 = HAP_power_get_dcvsEnabled;
        auto ret                      = HAP_power_get(_context_ptr, &response);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to get DVCS enabled: %d\n", ret);
            return false;
        }

        return response.dcvsEnabled;
    }

    void set_dvcs_performance_mode(bool enable) {
        if (!is_valid()) {
            DEVICE_LOG_ERROR("Power context is not initialized\n");
            return;
        }

        HAP_power_request_t request     = {};
        request.type                    = HAP_power_set_DCVS_v3;
        request.dcvs_v3.set_dcvs_enable = enable ? TRUE : FALSE;
        request.dcvs_v3.dcvs_enable     = enable ? TRUE : FALSE;
        request.dcvs_v3.set_core_params = TRUE;
        if (enable) {
            request.dcvs_v3.dcvs_option               = HAP_DCVS_V2_PERFORMANCE_MODE;
            request.dcvs_v3.set_bus_params            = TRUE;
            request.dcvs_v3.bus_params.min_corner     = HAP_DCVS_VCORNER_MAX;
            request.dcvs_v3.bus_params.max_corner     = HAP_DCVS_VCORNER_MAX;
            request.dcvs_v3.bus_params.target_corner  = HAP_DCVS_VCORNER_MAX;
            request.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_MAX;
            request.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_MAX;
            request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
            request.dcvs_v3.set_sleep_disable         = TRUE;
            request.dcvs_v3.sleep_disable             = TRUE;
        }

        auto ret = HAP_power_set(_context_ptr, &request);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to set DVCS performance mode: %d\n", ret);
        }
    }

    void set_sleep_mode(bool enable) {
        if (!is_valid()) {
            DEVICE_LOG_ERROR("Power context is not initialized\n");
            return;
        }

        boolean sleep_disable = enable ? FALSE : TRUE;
        auto    ret           = HAP_power_set_sleep_mode(_context_ptr, sleep_disable);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to set sleep mode: %d\n", ret);
        }
    }

    bool is_valid() const { return _context_ptr != nullptr; }

  private:
    void * _context_ptr = nullptr;

    DISABLE_COPY_AND_MOVE(power_utils);
};

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING

struct sub_process_data {
    char     log_prefix[32] = {};
    uint64_t proc_cycles    = 0;
    uint64_t proc_pcycles   = 0;
    uint64_t proc_count     = 0;
};

template <size_t _buffer_count> class npu_scoped_timer {
  public:
    enum {
        kBufferCount  = _buffer_count,
        kSubProcCount = 4,
    };

    explicit npu_scoped_timer(const char * log_prefix) {
        strncpy(_log_prefix, log_prefix, kBufferCount - 1);
        _begin_cycles  = HAP_perf_get_qtimer_count();
        _begin_pcycles = HAP_perf_get_pcycles();
    }

    npu_scoped_timer(npu_scoped_timer && other) { *this = std::move(other); }

    ~npu_scoped_timer() { print(); }

    void operator=(npu_scoped_timer && other) {
        strncpy(_log_prefix, other._log_prefix, kBufferCount - 1);
        _begin_cycles  = other._begin_cycles;
        _begin_pcycles = other._begin_pcycles;
        memcpy(&_sub_proc_data, &other._sub_proc_data, sizeof(_sub_proc_data));
    }

    void add_sub_proc_cycles(size_t sub_proc_idx, const char * sub_proc_prefix, uint64_t cycles, uint64_t pcycles) {
        auto & sub_proc_data = _sub_proc_data[sub_proc_idx];
        sub_proc_data.proc_cycles += cycles;
        sub_proc_data.proc_pcycles += pcycles;

        if (!sub_proc_data.proc_count) {
            strncpy(sub_proc_data.log_prefix, sub_proc_prefix, sizeof(sub_proc_data.log_prefix) - 1);
        }

        sub_proc_data.proc_count++;
    }

    void print() const {
        static_assert(kSubProcCount == 4, "Sub process count must be 4 for logging format");

        auto total_cycles  = HAP_perf_get_qtimer_count() - _begin_cycles;
        auto total_pcycles = HAP_perf_get_pcycles() - _begin_pcycles;
        auto duration      = HAP_perf_qtimer_count_to_us(total_cycles);

        int sub_proc_count = 0;
        for (int i = kSubProcCount; i > 0; --i) {
            if (_sub_proc_data[i - 1].proc_count > 0) {
                sub_proc_count = i;
                break;
            }
        }

        auto sub_proc0_duration = HAP_perf_qtimer_count_to_us(_sub_proc_data[0].proc_cycles);
        auto sub_proc1_duration = HAP_perf_qtimer_count_to_us(_sub_proc_data[1].proc_cycles);
        auto sub_proc2_duration = HAP_perf_qtimer_count_to_us(_sub_proc_data[2].proc_cycles);
        auto sub_proc3_duration = HAP_perf_qtimer_count_to_us(_sub_proc_data[3].proc_cycles);

        switch (sub_proc_count) {
            case 4:
                DEVICE_LOG_WARN(
                    "[profiler]%s, pcyc: %llu, dur: %lluus, [%s]cnt: %llu, dur: %lluus, "
                    "[%s]cnt: %llu, dur: %lluus, [%s]cnt: %llu, dur: %lluus, "
                    "[%s]cnt: %llu, dur: %lluus\n",
                    _log_prefix, (unsigned long long) total_pcycles, (unsigned long long) duration,
                    _sub_proc_data[0].log_prefix, (unsigned long long) _sub_proc_data[0].proc_count,
                    (unsigned long long) sub_proc0_duration, _sub_proc_data[1].log_prefix,
                    (unsigned long long) _sub_proc_data[1].proc_count, (unsigned long long) sub_proc1_duration,
                    _sub_proc_data[2].log_prefix, (unsigned long long) _sub_proc_data[2].proc_count,
                    (unsigned long long) sub_proc2_duration, _sub_proc_data[3].log_prefix,
                    (unsigned long long) _sub_proc_data[3].proc_count, (unsigned long long) sub_proc3_duration);
                break;
            case 3:
                DEVICE_LOG_WARN(
                    "[profiler]%s, pcyc: %llu, dur: %lluus, [%s]cnt: %llu, dur: %lluus, "
                    "[%s]cnt: %llu, dur: %lluus, [%s]cnt: %llu, dur: %lluus\n",
                    _log_prefix, (unsigned long long) total_pcycles, (unsigned long long) duration,
                    _sub_proc_data[0].log_prefix, (unsigned long long) _sub_proc_data[0].proc_count,
                    (unsigned long long) sub_proc0_duration, _sub_proc_data[1].log_prefix,
                    (unsigned long long) _sub_proc_data[1].proc_count, (unsigned long long) sub_proc1_duration,
                    _sub_proc_data[2].log_prefix, (unsigned long long) _sub_proc_data[2].proc_count,
                    (unsigned long long) sub_proc2_duration);
                break;
            case 2:
                DEVICE_LOG_WARN(
                    "[profiler]%s, pcyc: %llu, dur: %lluus, [%s]cnt: %llu, dur: %lluus, "
                    "[%s]cnt: %llu, dur: %lluus\n",
                    _log_prefix, (unsigned long long) total_pcycles, (unsigned long long) duration,
                    _sub_proc_data[0].log_prefix, (unsigned long long) _sub_proc_data[0].proc_count,
                    (unsigned long long) sub_proc0_duration, _sub_proc_data[1].log_prefix,
                    (unsigned long long) _sub_proc_data[1].proc_count, (unsigned long long) sub_proc1_duration);
                break;
            case 1:
                DEVICE_LOG_WARN("[profiler]%s, pcyc: %llu, dur: %lluus, [%s]cnt: %llu, dur: %lluus\n", _log_prefix,
                                (unsigned long long) total_pcycles, (unsigned long long) duration,
                                _sub_proc_data[0].log_prefix, (unsigned long long) _sub_proc_data[0].proc_count,
                                (unsigned long long) sub_proc0_duration);
                break;
            default:
            case 0:
                DEVICE_LOG_WARN("[profiler]%s, pcyc: %llu, dur: %lluus\n", _log_prefix,
                                (unsigned long long) total_pcycles, (unsigned long long) duration);
                break;
        }
    }

  private:
    char             _log_prefix[kBufferCount]     = {};
    uint64_t         _begin_cycles                 = 0;
    uint64_t         _begin_pcycles                = 0;
    sub_process_data _sub_proc_data[kSubProcCount] = {};

    DISABLE_COPY(npu_scoped_timer);
};

template <size_t _buffer_count, size_t _sub_idx> class npu_sub_process_scoped_timer {
  public:
    static_assert(_sub_idx < npu_scoped_timer<_buffer_count>::kSubProcCount,
                  "Sub process index must be less than kSubProcCount");
    using npu_scoped_timer = npu_scoped_timer<_buffer_count>;

    explicit npu_sub_process_scoped_timer(npu_scoped_timer & timer, const char * prefix) :
        _timer(timer),
        _prefix(prefix) {
        _begin_cycles  = HAP_perf_get_qtimer_count();
        _begin_pcycles = HAP_perf_get_pcycles();
    }

    ~npu_sub_process_scoped_timer() {
        _timer.add_sub_proc_cycles(_sub_idx, _prefix, HAP_perf_get_qtimer_count() - _begin_cycles,
                                   HAP_perf_get_pcycles() - _begin_pcycles);
    }

  private:
    npu_scoped_timer & _timer;
    const char *       _prefix        = nullptr;
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
    return npu_scoped_timer<1024>(buffer);
}

#endif

}  // namespace hexagon

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
#    define _MAKE_VARIABLE_NAME2(name, postfix) name##postfix
#    define _MAKE_VARIABLE_NAME(name, postfix)  _MAKE_VARIABLE_NAME2(name, postfix)
#    define DEVICE_SCOPED_PERFORMANCE_TRACKER(fmt, ...) \
        auto _MAKE_VARIABLE_NAME(__npu_timer_, __LINE__) = hexagon::make_scoped_perf_timer(fmt, __VA_ARGS__)
#else
#    define DEVICE_SCOPED_PERFORMANCE_TRACKER(fmt, ...) ((void) 0)
#endif

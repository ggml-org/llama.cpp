#pragma once

#include <QnnCommon.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "logger.hpp"
#include "qnn-types.hpp"

namespace qnn {

#ifdef GGML_QNN_ENABLE_PERFORMANCE_TRACKING

class qnn_scoped_timer {
  public:
    qnn_scoped_timer(const std::string & log_prefix) : _log_prefix(std::move(log_prefix)) {
        _begin_us = ggml_time_us();
    }

    qnn_scoped_timer(qnn_scoped_timer && other) {
        _begin_us   = other._begin_us;
        _log_prefix = std::move(other._log_prefix);
    }

    ~qnn_scoped_timer() { print(); }

    void operator=(qnn_scoped_timer && other) {
        _begin_us   = other._begin_us;
        _log_prefix = std::move(other._log_prefix);
    }

    void print() const {
        auto duration = (ggml_time_us() - _begin_us) / 1000.0;
        QNN_LOG_INFO("[profiler]%s, duration: %.4f ms\n", _log_prefix.c_str(), duration);
    }


  private:
    int64_t     _begin_us = 0LL;
    std::string _log_prefix;

    qnn_scoped_timer(const qnn_scoped_timer &) = delete;
    void operator=(const qnn_scoped_timer &)   = delete;
};

inline qnn_scoped_timer make_scope_perf_timer(const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return qnn_scoped_timer(buffer);
}

#else

inline void make_scope_perf_timer(const char *, ...) {}

#endif

// forward declaration of qnn_interface
class qnn_interface;

class qnn_event_tracer {
  public:
    // ref:
    //   https://github.com/pytorch/executorch/blob/ae3d558d5e6aa04fc52a3065399fe6a773702f52/backends/qualcomm/serialization/qc_schema.py#L53
    //   https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices
    enum sdk_profile_level { PROFILE_OFF = 0, PROFILE_BASIC, PROFILE_DETAIL, PROFILE_OP_TRACE };

    explicit qnn_event_tracer(const std::string & prefix, std::shared_ptr<qnn_interface> interface,
                              Qnn_BackendHandle_t backend_handle, sdk_profile_level level);
    ~qnn_event_tracer();

    Qnn_ProfileHandle_t get_handle() const { return _handle; }

    void print_profile_events();

  private:
    std::shared_ptr<qnn_interface> _interface;
    Qnn_ProfileHandle_t            _handle = nullptr;
    std::string                    _prefix;

    DISABLE_COPY(qnn_event_tracer);
    DISABLE_MOVE(qnn_event_tracer);
};

using qnn_event_tracer_ptr = std::shared_ptr<qnn_event_tracer>;

}  // namespace qnn

#ifdef GGML_QNN_ENABLE_PERFORMANCE_TRACKING
#    define QNN_SCOPED_PERFORMANCE_TRACKER(fmt, ...) \
        auto __qnn_timer_##__LINE__ = qnn::make_scope_perf_timer(fmt, __VA_ARGS__)
#else
#    define QNN_SCOPED_PERFORMANCE_TRACKER(fmt, ...) ((void) 0)
#endif

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "common.hpp"
#include "ggml-impl.h"

namespace profiler {

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING

class scoped_timer {
  public:
    scoped_timer(const std::string & log_prefix) : _log_prefix(std::move(log_prefix)) { _begin_us = ggml_time_us(); }

    scoped_timer(scoped_timer && other) {
        _begin_us   = other._begin_us;
        _log_prefix = std::move(other._log_prefix);
    }

    ~scoped_timer() { print(); }

    void operator=(scoped_timer && other) {
        _begin_us   = other._begin_us;
        _log_prefix = std::move(other._log_prefix);
    }

    void print() const {
        auto duration = ggml_time_us() - _begin_us;
        GGML_LOG_INFO("[profiler]%s, dur: %lld us\n", _log_prefix.c_str(), (long long) duration);
    }


  private:
    int64_t     _begin_us = 0LL;
    std::string _log_prefix;

    DISABLE_COPY(scoped_timer);
};

inline scoped_timer make_scope_perf_timer(const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return scoped_timer(buffer);
}

#endif

}  // namespace profiler

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
#    define SCOPED_PERFORMANCE_TRACKER(fmt, ...) \
        auto __scoped_timer_##__LINE__ = profiler::make_scope_perf_timer(fmt, __VA_ARGS__)
#    define PROFILER_LOG_DEBUG(fmt, ...) GGML_LOG_INFO("[profiler]" fmt, __VA_ARGS__)
#else
#    define SCOPED_PERFORMANCE_TRACKER(fmt, ...) ((void) 0)
#    define PROFILER_LOG_DEBUG(...)              ((void) 0)
#endif

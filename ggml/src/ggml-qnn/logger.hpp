#pragma once

#include <QnnLog.h>

#include <cstdint>

#include "ggml-impl.h"
#include "ggml.h"

namespace qnn {
void sdk_logcallback(const char * fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp);
}  // namespace qnn

#define QNN_LOG_ERROR(...) (GGML_LOG_ERROR(__VA_ARGS__))
#define QNN_LOG_WARN(...)  (GGML_LOG_WARN(__VA_ARGS__))
#define QNN_LOG_INFO(...)  (GGML_LOG_INFO(__VA_ARGS__))

#ifndef NDEBUG
#    define QNN_LOG_DEBUG(...) (GGML_LOG_DEBUG(__VA_ARGS__))
#else
#    define QNN_LOG_DEBUG(...)
#endif

#pragma once

#include "ggml-backend.h"

// internal API
void ggml_et_log_internal(ggml_log_level level, const char* file, int line, const char* fmt, ...);

#define ET_LOG_INTERNAL(level, ...) ggml_et_log_internal(level, __FILE_NAME__, __LINE__, __VA_ARGS__)
#define ET_LOG(...) ET_LOG_INTERNAL(GGML_LOG_LEVEL_TRACE, __VA_ARGS__)
#define ET_LOG_DEBUG(...) ET_LOG_INTERNAL(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define ET_LOG_INFO(...) ET_LOG_INTERNAL(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define ET_LOG_WARN(...) ET_LOG_INTERNAL(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define ET_LOG_ERROR(...) ET_LOG_INTERNAL(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

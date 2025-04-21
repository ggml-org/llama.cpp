
#include "logger.hpp"

#ifndef NDEBUG

#    include <mutex>

#    include "QnnInterface.h"
#    include "QnnTypes.h"
#    include "System/QnnSystemInterface.h"

void qnn::sdk_logcallback(const char * fmt, QnnLog_Level_t level, uint64_t /*timestamp*/, va_list argp) {
    static std::mutex log_mutex;
    static char       s_ggml_qnn_logbuf[4096];

    char log_level_desc;
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            log_level_desc = 'E';
            break;
        case QNN_LOG_LEVEL_WARN:
            log_level_desc = 'W';
            break;
        case QNN_LOG_LEVEL_INFO:
            log_level_desc = 'I';
            break;
        case QNN_LOG_LEVEL_DEBUG:
            log_level_desc = 'D';
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            log_level_desc = 'V';
            break;
        default:
            log_level_desc = 'U';
            break;
    }

    {
        std::lock_guard<std::mutex> lock(log_mutex);
        int                         size = vsnprintf(s_ggml_qnn_logbuf, sizeof(s_ggml_qnn_logbuf), fmt, argp);
        if (size > 0 && s_ggml_qnn_logbuf[size - 1] != '\n') {
            QNN_LOG_INFO("[%c]%s\n", log_level_desc, s_ggml_qnn_logbuf);
        } else {
            QNN_LOG_INFO("[%c]%s", log_level_desc, s_ggml_qnn_logbuf);
        }
    }
}
#else
void qnn::sdk_logcallback(const char *, QnnLog_Level_t, uint64_t, va_list) {}
#endif

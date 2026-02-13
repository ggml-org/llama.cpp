#include "ggml-threading.h"
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
#if defined(__x86_64__) || defined(_M_X64)
    if (_xbegin() == _XBEGIN_STARTED) {
        return;
    }
#endif
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
#if defined(__x86_64__) || defined(_M_X64)
    if (_xtest()) {
        _xend();
        return;
    }
#endif
    ggml_critical_section_mutex.unlock();
}

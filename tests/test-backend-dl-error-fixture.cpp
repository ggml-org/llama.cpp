#include "ggml-backend-impl.h"

static const char * fixture_name(ggml_backend_reg_t) {
    return "DL_TEST";
}

static size_t fixture_device_count(ggml_backend_reg_t) {
    return 0;
}

extern "C" __declspec(dllexport) ggml_backend_reg_t ggml_backend_init(void) {
    static ggml_backend_reg reg = {
        GGML_BACKEND_API_VERSION,
        { fixture_name, fixture_device_count, nullptr, nullptr },
        nullptr,
    };
    return &reg;
}

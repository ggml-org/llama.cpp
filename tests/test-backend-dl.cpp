#ifdef GGML_TEST_BACKEND_DL_DEP

extern "C" __declspec(dllexport) int ggml_test_dl_ping(void) {
    return 42;
}

#elif defined(GGML_TEST_BACKEND_DL_PLUGIN)

#include "ggml-backend-impl.h"

extern "C" __declspec(dllimport) int ggml_test_dl_ping(void);

static const char * ggml_backend_test_dl_reg_get_name(ggml_backend_reg_t) {
    return "test-dl";
}

static size_t ggml_backend_test_dl_reg_get_device_count(ggml_backend_reg_t) {
    return 0;
}

static ggml_backend_dev_t ggml_backend_test_dl_reg_get_device(ggml_backend_reg_t, size_t) {
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_test_dl_reg_i = {
    /* .get_name         = */ ggml_backend_test_dl_reg_get_name,
    /* .get_device_count = */ ggml_backend_test_dl_reg_get_device_count,
    /* .get_device       = */ ggml_backend_test_dl_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static ggml_backend_reg_t ggml_backend_test_dl_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_test_dl_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_test_dl_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_test_dl_ping)

#else

#include "ggml-backend.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <plugin.dll>\n", argv[0]);
        return 1;
    }

    ggml_backend_reg_t reg = ggml_backend_load(argv[1]);
    if (reg == nullptr) {
        fprintf(stderr, "failed to load backend from %s\n", argv[1]);
        return 1;
    }

    printf("loaded backend %s\n", ggml_backend_reg_name(reg));
    return 0;
}

#endif

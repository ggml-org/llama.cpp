#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

// FastRPC remote.h: DSPRPC_GET_DSP_INFO, CDSP_DOMAIN_ID and remote_dsp_attributes.
struct capability {
    uint32_t domain;
    uint32_t attribute;
    uint32_t value;
};

int main(void) {
    void * driver = dlopen("libcdsprpc.so", RTLD_NOW);
    if (!driver) {
        fprintf(stderr, "%s\n", dlerror());
        return 1;
    }
    int (*control)(uint32_t, void *, uint32_t) = dlsym(driver, "remote_handle_control");
    if (!control) {
        fprintf(stderr, "%s\n", dlerror());
        dlclose(driver);
        return 1;
    }
    const char * names[] = {"ARCH_VER", "HMX_SUPPORT_DEPTH", "HMX_SUPPORT_SPATIAL"};
    int failed = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        struct capability query = {3, 6 + i, 0};
        int status = control(2, &query, sizeof(query));
        printf("%s status=0x%x value=0x%x\n", names[i], (unsigned)status, query.value);
        failed |= status != 0;
    }
    dlclose(driver);
    return failed;
}

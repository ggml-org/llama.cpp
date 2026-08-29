#include "rdna3-auto-policy.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void check(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

} // namespace

int main() {
    using flag = ggml_cuda_rdna3_auto_flag;
    check(ggml_cuda_rdna3_auto_parse(nullptr) == flag::disabled, "unset RDNA3 Auto flag must be disabled");
    for (const char * value : { "1", "on", "true", "yes" }) {
        check(ggml_cuda_rdna3_auto_parse(value) == flag::enabled, "enabled RDNA3 Auto spelling rejected");
    }
    for (const char * value : { "0", "off", "false", "no" }) {
        check(ggml_cuda_rdna3_auto_parse(value) == flag::disabled, "disabled RDNA3 Auto spelling rejected");
    }
    for (const char * value : { "", "TRUE", "garbage" }) {
        check(ggml_cuda_rdna3_auto_parse(value) == flag::invalid, "invalid RDNA3 Auto spelling accepted");
    }
    std::puts("RDNA3 auto policy parser: PASS");
    return 0;
}

#include <cstdio>
#include <cstring>

#include "ggml-sycl/ggml-sycl-test.hpp"

int main() {
    const char * packed_name = ggml_sycl::test_layout_name(GGML_LAYOUT_ONEDNN_PACKED);
    if (!packed_name) {
        std::printf("FAIL: packed layout name is null\n");
        return 1;
    }
    if (std::strcmp(packed_name, "onednn_packed") != 0) {
        std::printf("FAIL: expected onednn_packed, got %s\n", packed_name);
        return 1;
    }
    const char * woq_name = ggml_sycl::test_layout_name(GGML_LAYOUT_ONEDNN_WOQ);
    if (!woq_name) {
        std::printf("FAIL: woq layout name is null\n");
        return 1;
    }
    if (std::strcmp(woq_name, "onednn_woq") != 0) {
        std::printf("FAIL: expected onednn_woq, got %s\n", woq_name);
        return 1;
    }
    std::printf("PASS: onednn packed + woq layout names resolved\n");
    return 0;
}

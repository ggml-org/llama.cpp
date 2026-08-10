#include "llama.h"
#include <cstdio>

int main(void) {
    printf("[test-cmake] Using llama.cpp version %s\n", llama_version());
    printf("[test-cmake] Initializing backend...\n");
    llama_backend_init();
    printf("[test-cmake] Backend initialized.\n");
    llama_backend_free();
    return 0;
}

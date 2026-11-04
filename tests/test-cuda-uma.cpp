// checks which CUDA devices take their free memory from /proc/meminfo instead of cudaMemGetInfo
#include "ggml-cuda/uma.cuh"

#include <cstdio>

int main() {
    struct test_case {
        const char * name;
        bool integrated;
        bool unified_memory_env;
        bool wsl;
        bool want;
    };

    const test_case cases[] = {
        { "discrete",                             false, false, false, false },
        { "discrete, unified memory env",         false, true,  false, true  },
        { "integrated",                           true,  false, false, true  },
        { "integrated, unified memory env",       true,  true,  false, true  },
        { "WSL discrete",                         false, false, true,  false },
        { "WSL discrete, unified memory env",     false, true,  true,  true  },
        { "WSL integrated",                       true,  false, true,  false },
        { "WSL integrated, unified memory env",   true,  true,  true,  false },
    };

    int failures = 0;
    for (const test_case & tc : cases) {
        const bool got = ggml_cuda_uma_free_from_host(tc.integrated, tc.unified_memory_env, tc.wsl);
        if (got != tc.want) {
            fprintf(stderr, "%s: got %d, want %d\n", tc.name, got, tc.want);
            failures++;
        }
    }
    return failures == 0 ? 0 : 1;
}

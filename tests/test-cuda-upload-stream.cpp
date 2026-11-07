// test-cuda-upload-stream.cpp
// Regression test for the llama.cpp CUDA upload-stream event-lifetime fix
// (ggml-cuda.cu: cudaStreamWaitEvent on the ring slot before reuse).
//
// Build as a plain C++ TU against ggml + ggml-cuda (no nvcc needed): all device
// work goes through the ggml backend API. Run with GGML_CUDA_UPLOAD_STREAM=1.
//
// What it catches: set_tensor_async writes on up_stream, get_tensor_async reads on
// the compute stream. Without the per-copy event wait the compute stream overtakes
// the upload and the read returns stale/garbage bytes. With the fix every read
// matches the pattern just written. Also checks that the stream + 64 events are
// destroyed (no VRAM leak across create/destroy cycles).
//
// Small footprint: a 1 MiB tensor, any CUDA device (no 8 GiB required).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static unsigned long long checksum(const unsigned char * p, int n) {
    unsigned long long s = 0;
    for (int i = 0; i < n; ++i) {
        s += (unsigned long long) p[i] * (unsigned long long) (i + 1);
    }
    return s;
}

// setenv is absent on MSVC; _putenv_s is the Windows equivalent.
static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main() {
    set_env("GGML_CUDA_UPLOAD_STREAM", "1");

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        fprintf(stderr, "SKIP: no CUDA backend\n");
        return 0; // no CUDA device -> not a failure
    }

    const int N = 1 << 20; // 1 MiB
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, N);
    if (buf == nullptr) {
        fprintf(stderr, "FAIL: buffer alloc\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, N);
    ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));

    std::vector<unsigned char> hin(N), hout(N);
    const int ITER = 255; // > n_up_events (64): forces ring-slot reuse
    int bad = 0;

    for (int i = 0; i < ITER; ++i) {
        const unsigned char v = (unsigned char) (i & 0xFF);
        memset(hin.data(), v, N);
        const unsigned long long want = checksum(hin.data(), N);

        ggml_backend_tensor_set_async(backend, t, hin.data(), 0, N);  // up_stream
        ggml_backend_tensor_get_async(backend, t, hout.data(), 0, N); // compute stream
        ggml_backend_synchronize(backend);

        const unsigned long long got = checksum(hout.data(), N);
        if (got != want || memcmp(hin.data(), hout.data(), N) != 0) {
            fprintf(stderr, "MISMATCH iter %d: want %llu got %llu (event reused before completion)\n",
                    i, want, got);
            ++bad;
        }
    }

    // Leak probe: 100 create/destroy cycles must not drop free VRAM
    // (the destructor must destroy up_stream and the 64 up_events).
    size_t free0 = 0, total0 = 0;
    ggml_backend_cuda_get_device_memory(0, &free0, &total0);
    for (int k = 0; k < 100; ++k) {
        ggml_backend_t b = ggml_backend_cuda_init(0);
        ggml_backend_buffer_t bb = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(0), N);
        struct ggml_tensor * tt = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, N);
        ggml_backend_tensor_alloc(bb, tt, ggml_backend_buffer_get_base(bb));
        ggml_backend_tensor_set_async(b, tt, hin.data(), 0, N); // creates up_stream + events
        ggml_backend_synchronize(b);
        ggml_backend_buffer_free(bb);
        ggml_backend_free(b);
    }
    size_t free1 = 0, total1 = 0;
    ggml_backend_cuda_get_device_memory(0, &free1, &total1);
    if (free1 + (16u << 20) < free0) {
        fprintf(stderr, "LEAK: free VRAM dropped %zu MiB over 100 cycles\n",
                (size_t) ((free0 - free1) >> 20));
        ++bad;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (bad) {
        printf("FAIL (%d)\n", bad);
        return 1;
    }
    printf("OK: %d copies verified, no leak\n", ITER);
    return 0;
}

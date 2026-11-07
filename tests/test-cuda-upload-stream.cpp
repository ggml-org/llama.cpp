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

    const int N = 1 << 22; // 4 MiB: long enough for the compute stream to overtake the upload
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, N);
    if (buf == nullptr) {
        fprintf(stderr, "FAIL: buffer alloc\n");
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, N);
    ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));

    std::vector<std::vector<unsigned char>> hin(200, std::vector<unsigned char>(N));
    std::vector<std::vector<unsigned char>> hout(200, std::vector<unsigned char>(N));
    const int ITER = 200; // > n_up_events (64): forces ring-slot reuse
    int bad = 0;

    // No synchronize inside the loop: the compute stream must overtake the upload
    // if the per-copy event wait is missing.
    for (int i = 0; i < ITER; ++i) {
        memset(hin[i].data(), (unsigned char) (i & 0xFF), N);
        ggml_backend_tensor_set_async(backend, t, hin[i].data(), 0, N);  // up_stream
        ggml_backend_tensor_get_async(backend, t, hout[i].data(), 0, N); // compute stream
    }
    ggml_backend_synchronize(backend);

    for (int i = 0; i < ITER; ++i) {
        const unsigned char v = (unsigned char) (i & 0xFF);
        if (memcmp(hin[i].data(), hout[i].data(), N) != 0) {
            fprintf(stderr, "MISMATCH iter %d: expected 0x%02x (event reused before completion)\n", i, v);
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
        struct ggml_init_params lp = { ggml_tensor_overhead() * 8, nullptr, true };
        struct ggml_context * lctx = ggml_init(lp);
        struct ggml_tensor * tt = ggml_new_tensor_1d(lctx, GGML_TYPE_I8, N);
        ggml_backend_tensor_alloc(bb, tt, ggml_backend_buffer_get_base(bb));
        ggml_backend_tensor_set_async(b, tt, hin[0].data(), 0, N); // creates up_stream + events
        ggml_backend_synchronize(b);
        ggml_backend_buffer_free(bb);
        ggml_backend_free(b);
        ggml_free(lctx);
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

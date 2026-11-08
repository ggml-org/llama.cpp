// test-cuda-upload-stream.cpp
// Regression test for the CUDA upload-stream event-lifetime fix: with GGML_CUDA_UPLOAD_STREAM=1
// ggml_backend_cuda_set_tensor_async issues the copy on a dedicated stream and the compute stream
// waits on the per-copy event, so a read of the tensor observes the bytes of that copy.
//
// Only the public ggml backend API is used, so the test links against ggml alone and runs on any
// build where the CUDA backend is registered. It sets GGML_CUDA_UPLOAD_STREAM before the first
// copy because the flag is read once per process, and exits 0 with a SKIP line when there is no
// GPU device (a CPU-only or GPU-less runner must not report a failure).
//
// What it catches: the upload runs on up_stream while the read is queued on the compute stream.
// Without the wait on the ring slot the compute stream overtakes the upload and the read returns
// the bytes of an earlier copy. Every input buffer is distinct and ITER exceeds the 64-slot ring,
// so each slot is reused while an earlier copy is still in flight.

#include "ggml.h"
#include "ggml-backend.h"

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

// byte(i) is injective modulo 256 for the ITER used here, so a stale read of copy j never matches
// the expected byte of copy i.
static unsigned char fill_byte(int i) {
    return (unsigned char) (i * 37 + 11);
}

int main() {
    set_env("GGML_CUDA_UPLOAD_STREAM", "1");

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        printf("SKIP: no GPU backend device\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        printf("SKIP: cannot init the GPU backend\n");
        return 0;
    }

    const size_t N    = 1u << 20; // 1 MiB per copy
    const int    ITER = 128;      // > the 64 upload events: every ring slot is reused while pending
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (buft == nullptr) {
        printf("SKIP: the GPU device has no buffer type\n");
        ggml_backend_free(backend);
        return 0;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(buft, N);
    if (buf == nullptr) {
        printf("FAIL: buffer alloc\n");
        ggml_backend_free(backend);
        return 1;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) N);
    ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));

    // ITER distinct host buffers on each side: an async copy reads and writes host memory until it
    // completes, so reusing one buffer between iterations would be a race in the test itself.
    std::vector<unsigned char> in((size_t) ITER * N);
    std::vector<unsigned char> out((size_t) ITER * N);

    for (int i = 0; i < ITER; ++i) {
        unsigned char * src = in.data() + (size_t) i * N;
        memset(src, fill_byte(i), N);
        ggml_backend_tensor_set_async(backend, t, src, 0, N);                    // up_stream
        ggml_backend_tensor_get_async(backend, t, out.data() + (size_t) i * N, 0, N); // compute stream
    }
    ggml_backend_synchronize(backend);

    int bad = 0;
    for (int i = 0; i < ITER; ++i) {
        const unsigned char * got = out.data() + (size_t) i * N;
        const unsigned char want = fill_byte(i);
        for (size_t k = 0; k < N; ++k) {
            if (got[k] != want) {
                printf("FAIL: copy %d read 0x%02x at %zu, expected 0x%02x (the compute stream "
                       "overtook the upload or a ring slot was reused early)\n", i, got[k], k, want);
                ++bad;
                break;
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (bad) {
        return 1;
    }
    printf("OK: %d copies of %zu MiB verified on the upload stream\n", ITER, N >> 20);
    return 0;
}

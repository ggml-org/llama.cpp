#include "testing.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstring>
#include <string>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define KV_MMAP_TEST_SUPPORTED
#endif // defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))

static const std::string test_path = "/tmp/test-kv-cache-mmap.bin";

static void cleanup() {
#ifdef KV_MMAP_TEST_SUPPORTED
    unlink(test_path.c_str());
#endif
}

int main() {
    testing t;

#ifndef KV_MMAP_TEST_SUPPORTED
    fprintf(stderr, "mmap KV cache not supported on this platform, skipping\n");
    return 0;
#else

    t.test("buffer_from_ptr wraps user memory", [&](testing & t) {
        const size_t buf_size = 64 * 1024;
        void * backing = aligned_alloc(64, buf_size);
        GGML_ASSERT(backing != nullptr);
        memset(backing, 0xab, buf_size);

        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(backing, buf_size);
        t.assert_true("buffer created",  buf != nullptr);
        t.assert_true("base matches",    ggml_backend_buffer_get_base(buf) == backing);
        t.assert_true("size matches",    ggml_backend_buffer_get_size(buf) == buf_size);

        ggml_backend_buffer_free(buf);
        free(backing);
    });

    t.test("tallocr seats tensor into user buffer", [&](testing & t) {
        const size_t buf_size = 64 * 1024;
        void * backing = aligned_alloc(64, buf_size);
        GGML_ASSERT(backing != nullptr);
        memset(backing, 0xab, buf_size);

        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(backing, buf_size);

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 4,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        t.assert_true("context created", ctx != nullptr);

        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        t.assert_true("tensor created", tensor != nullptr);

        ggml_tallocr tallocr = ggml_tallocr_new(buf);
        ggml_tallocr_alloc(&tallocr, tensor);

        uintptr_t data_addr = (uintptr_t) tensor->data;
        uintptr_t base_addr = (uintptr_t) backing;
        t.assert_true("tensor data inside backing region",
                data_addr >= base_addr && data_addr < base_addr + buf_size);

        // write through tensor, verify via backing pointer
        float * floats = (float *) tensor->data;
        for (int i = 0; i < 128; i++) {
            floats[i] = (float) i * 2.5f;
        }

        size_t offset = data_addr - base_addr;
        const float * readback = (const float *) ((uint8_t *) backing + offset);
        bool match = true;
        for (int i = 0; i < 128; i++) {
            if (readback[i] != (float) i * 2.5f) {
                match = false;
                break;
            }
        }
        t.assert_true("write through tensor visible via backing pointer", match);

        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
        free(backing);
    });

    t.test("multiple tensors allocated sequentially", [&](testing & t) {
        const size_t buf_size = 256 * 1024;
        void * backing = aligned_alloc(64, buf_size);
        GGML_ASSERT(backing != nullptr);

        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(backing, buf_size);

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);

        // simulate 4 layers of K tensors
        const int n_layers = 4;
        ggml_tensor * tensors[4];
        ggml_tallocr tallocr = ggml_tallocr_new(buf);

        for (int i = 0; i < n_layers; i++) {
            tensors[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 256);
            ggml_tallocr_alloc(&tallocr, tensors[i]);
        }

        uintptr_t base = (uintptr_t) backing;

        // verify all tensors are inside the buffer and non-overlapping
        for (int i = 0; i < n_layers; i++) {
            uintptr_t addr = (uintptr_t) tensors[i]->data;
            t.assert_true("tensor inside buffer",
                    addr >= base && addr + ggml_nbytes(tensors[i]) <= base + buf_size);

            if (i > 0) {
                uintptr_t prev_end = (uintptr_t) tensors[i-1]->data + ggml_nbytes(tensors[i-1]);
                t.assert_true("tensor does not overlap previous", addr >= prev_end);
            }
        }

        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
        free(backing);
    });

    t.test("mmap file created and survives reopening", [&](testing & t) {
        cleanup();

        const size_t file_size = 32 * 1024;

        // create a file-backed mmap region, write data, unmap
        {
            int fd = open(test_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
            t.assert_true("open succeeded", fd >= 0);
            t.assert_true("ftruncate succeeded", ftruncate(fd, (off_t) file_size) == 0);

            void * base = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            t.assert_true("mmap succeeded", base != MAP_FAILED);

            // write a pattern
            uint8_t * bytes = (uint8_t *) base;
            for (size_t i = 0; i < file_size; i++) {
                bytes[i] = (uint8_t)(i & 0xff);
            }

            msync(base, file_size, MS_SYNC);
            munmap(base, file_size);
            close(fd);
        }

        // reopen and verify data survived
        {
            int fd = open(test_path.c_str(), O_RDONLY);
            t.assert_true("reopen succeeded", fd >= 0);

            void * base = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
            t.assert_true("remap succeeded", base != MAP_FAILED);

            const uint8_t * bytes = (const uint8_t *) base;
            bool match = true;
            for (size_t i = 0; i < file_size; i++) {
                if (bytes[i] != (uint8_t)(i & 0xff)) {
                    match = false;
                    break;
                }
            }
            t.assert_true("data survived close and reopen", match);

            munmap(base, file_size);
            close(fd);
        }

        // verify file size on disk
        {
            struct stat sb;
            t.assert_true("stat succeeded", stat(test_path.c_str(), &sb) == 0);
            t.assert_equal("file size matches", (long long) file_size, (long long) sb.st_size);
        }

        cleanup();
    });

    t.test("ggml tensor write persists through mmap file", [&](testing & t) {
        cleanup();

        const size_t n_floats  = 256;
        const size_t data_size = n_floats * sizeof(float);
        const size_t file_size = data_size + 1024; // padding for alignment

        // create file, allocate tensor, write values
        {
            int fd = open(test_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
            t.assert_true("open", fd >= 0);
            ftruncate(fd, (off_t) file_size);

            void * base = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            t.assert_true("mmap", base != MAP_FAILED);

            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, file_size);
            ggml_init_params params = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * ctx = ggml_init(params);

            ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) n_floats);
            ggml_tallocr tallocr = ggml_tallocr_new(buf);
            ggml_tallocr_alloc(&tallocr, tensor);

            float * data = (float *) tensor->data;
            for (size_t i = 0; i < n_floats; i++) {
                data[i] = (float) i * 3.14f;
            }

            ggml_free(ctx);
            ggml_backend_buffer_free(buf);
            msync(base, file_size, MS_SYNC);
            munmap(base, file_size);
            close(fd);
        }

        // reopen file, allocate tensor at same offset, verify values
        {
            int fd = open(test_path.c_str(), O_RDONLY);
            t.assert_true("reopen", fd >= 0);

            void * base = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
            t.assert_true("remap", base != MAP_FAILED);

            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, file_size);
            ggml_init_params params = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * ctx = ggml_init(params);

            ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) n_floats);
            ggml_tallocr tallocr = ggml_tallocr_new(buf);
            ggml_tallocr_alloc(&tallocr, tensor);

            const float * data = (const float *) tensor->data;
            bool match = true;
            for (size_t i = 0; i < n_floats; i++) {
                if (data[i] != (float) i * 3.14f) {
                    match = false;
                    break;
                }
            }
            t.assert_true("tensor data persisted across sessions", match);

            ggml_free(ctx);
            ggml_backend_buffer_free(buf);
            munmap(base, file_size);
            close(fd);
        }

        cleanup();
    });

    return t.summary();

#endif // KV_MMAP_TEST_SUPPORTED
}

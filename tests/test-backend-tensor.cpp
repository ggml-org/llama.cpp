#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool run_tensor_n_case(
        std::vector<ggml_tensor *> & tensors,
        const std::vector<size_t> & offsets,
        const std::vector<size_t> & sizes,
        const std::vector<size_t> & data_offsets) {
    const size_t n = tensors.size();

    std::vector<uint8_t> data(256, 0);
    std::vector<uint8_t> out(256, 0);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(data.data() + data_offsets[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(tensors.data(), data.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    ggml_backend_tensor_get_n(tensors.data(), out.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        if (memcmp(data.data() + data_offsets[i], out.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    std::vector<uint8_t> slice(256, 0);
    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(slice.data(), 0, slice.size());
        ggml_backend_tensor_get(tensors[i], slice.data(), offsets[i], sizes[i]);
        if (memcmp(slice.data(), data.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    return true;
}

static size_t align_up(size_t x, size_t align) {
    return (x + align - 1) / align * align;
}

static bool test_backend_tensor_n_mixed_buft(ggml_backend_t backend) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*16 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * t0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * s0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * s1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * s2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t buft0 = ggml_backend_dev_buffer_type(dev);

    const size_t align0 = ggml_backend_buft_get_alignment(buft0);
    const size_t sz_t0 = ggml_backend_buft_get_alloc_size(buft0, t0);
    const size_t sz_t2 = ggml_backend_buft_get_alloc_size(buft0, t2);
    const size_t buf0_size = align_up(sz_t0, align0) + sz_t2;

    ggml_backend_buffer_ptr buf0(ggml_backend_buft_alloc_buffer(buft0, buf0_size));
    if (!buf0) {
        return false;
    }

    char * base0 = (char *) ggml_backend_buffer_get_base(buf0.get());
    if (ggml_backend_tensor_alloc(buf0.get(), t0, base0) != GGML_STATUS_SUCCESS) {
        return false;
    }
    char * p2 = base0 + align_up(sz_t0, align0);
    if (ggml_backend_tensor_alloc(buf0.get(), t2, p2) != GGML_STATUS_SUCCESS) {
        return false;
    }

    const size_t host_align = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    std::vector<uint8_t> host(1024 + host_align, 0);
    uint8_t * host_ptr = (uint8_t *)(((uintptr_t)host.data() + host_align - 1) / host_align * host_align);
    ggml_backend_buffer_ptr buf1(ggml_backend_cpu_buffer_from_ptr(host_ptr, 1024));
    if (!buf1) {
        return false;
    }

    ggml_backend_buffer_type_t buft1 = ggml_backend_buffer_get_type(buf1.get());
    const size_t align1 = ggml_backend_buft_get_alignment(buft1);
    const size_t sz_t1 = ggml_backend_buft_get_alloc_size(buft1, t1);
    const size_t sz_s0 = ggml_backend_buft_get_alloc_size(buft1, s0);
    const size_t sz_s1 = ggml_backend_buft_get_alloc_size(buft1, s1);

    char * base1 = (char *) ggml_backend_buffer_get_base(buf1.get());
    char * p = base1;
    if (ggml_backend_tensor_alloc(buf1.get(), t1, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_t1, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s0, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_s0, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s1, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_s1, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s2, p) != GGML_STATUS_SUCCESS) {
        return false;
    }

    uint8_t src0[12], src1[8], src2[16];
    memset(src0, 0xA1, sizeof(src0));
    memset(src1, 0xB2, sizeof(src1));
    memset(src2, 0xC3, sizeof(src2));

    ggml_backend_tensor_set(s0, src0, 0, sizeof(src0));
    ggml_backend_tensor_set(s1, src1, 0, sizeof(src1));
    ggml_backend_tensor_set(s2, src2, 0, sizeof(src2));

    {
        std::vector<ggml_tensor *> tensors = { t0, t1, t2 };
        std::vector<size_t> offsets = { 0, 0, 0 };
        std::vector<size_t> sizes = { sizeof(src0), sizeof(src1), sizeof(src2) };
        std::vector<size_t> data_offsets = { 0, sizeof(src0), sizeof(src0) + sizeof(src1) };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<const ggml_tensor *> srcs = { s0, s1, s2 };
        std::vector<ggml_tensor *> dsts = { t0, t1, t2 };
        ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), srcs.size());

        const uint8_t * expected[3] = { src0, src1, src2 };
        const size_t sizes[3] = { sizeof(src0), sizeof(src1), sizeof(src2) };
        std::vector<uint8_t> out(64, 0);
        for (size_t i = 0; i < 3; i++) {
            memset(out.data(), 0, out.size());
            ggml_backend_tensor_get(dsts[i], out.data(), 0, sizes[i]);
            if (memcmp(out.data(), expected[i], sizes[i]) != 0) {
                return false;
            }
        }
    }

    return true;
}

static bool test_backend_tensor_n(ggml_backend_t backend) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 1024;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * t0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * t3 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t_big = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 6);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        return false;
    }

    {
        std::vector<ggml_tensor *> tensors = { t0, t1, t2, t0 };
        std::vector<size_t> offsets = { 0, 0, 0, 0 };
        std::vector<size_t> sizes = { 3*4, 2*4, 4*4, 0 };
        std::vector<size_t> data_offsets = { 0, 3*4, 5*4, 64 };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> tensors = { t_big, t_big, t_big };
        std::vector<size_t> offsets = { 0, 8, 16 };
        std::vector<size_t> sizes = { 8, 8, 8 };
        std::vector<size_t> data_offsets = { 0, 8, 16 };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> tensors = { t0, t3 };
        std::vector<size_t> offsets = { 0, 0 };
        std::vector<size_t> sizes = { 3*4, 3*4 };
        std::vector<size_t> data_offsets = { 0, 3*4 };

        if (!run_tensor_n_case(tensors, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    return true;
}

int main() {
    ggml_backend_load_all();

    const size_t n_devs = ggml_backend_dev_count();
    size_t n_ok = 0;

    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);

        printf("%s ", name);
        fflush(stdout);

        ggml_backend_ptr backend(ggml_backend_dev_init(dev, NULL));
        if (!backend) {
            printf("FAILED\n");
            continue;
        }

        const bool ok = test_backend_tensor_n(backend.get()) && test_backend_tensor_n_mixed_buft(backend.get());
        printf("%s\n", ok ? "PASSED" : "FAILED");
        if (ok) {
            n_ok++;
        }
    }

    printf("%zu/%zu backends passed\n", n_ok, n_devs);

    return n_ok == n_devs ? 0 : 1;
}

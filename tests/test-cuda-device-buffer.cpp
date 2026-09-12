#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-backend-memory.h"
#include "testing.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>

using device_buffer_type_fn = decltype(&ggml_backend_cuda_device_buffer_type);
using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

// Check the allocation itself, including interior pointers used by arena views.
static void check_pointer(testing & t, void * ptr, cudaMemoryType type, int device) {
    cudaPointerAttributes attributes = {};
    if (!t.assert_true(cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess)) return;
    t.assert_true(attributes.type == type);
    t.assert_equal(device, attributes.device);
}

// Exercise the inherited tensor and clear callbacks without adding a model dependency.
static void check_buffer(testing & t, ggml_backend_buffer_t buffer) {
    constexpr size_t count = 256;
    ggml_context_ptr ctx(ggml_init({4096, nullptr, true}));
    if (!t.assert_true(ctx != nullptr)) return;
    auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, count);
    if (!t.assert_true(ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS)) return;
    std::vector<float> values(count, 3.25f), read(count);
    ggml_backend_tensor_set(tensor, values.data(), 0, count*sizeof(float));
    ggml_backend_tensor_get(tensor, read.data(), 0, count*sizeof(float));
    t.assert_true(values == read);
    ggml_backend_buffer_clear(buffer, 0xa5);
    std::vector<uint8_t> bytes(count*sizeof(float));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    t.assert_true(std::all_of(bytes.begin(), bytes.end(), [](uint8_t x) { return x == 0xa5; }));
}

int main() {
    ggml_backend_load_all();
    auto * reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    testing t;
    auto factory = reinterpret_cast<device_buffer_type_fn>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_device_buffer_type"));
    t.test("device_local_factory_is_discoverable", [&](testing & t) {
        t.assert_true(factory != nullptr);
    });
    if (!factory) return t.summary();

    const int count = static_cast<int>(ggml_backend_reg_dev_count(reg));
    t.test("invalid_device_is_rejected", [&](testing & t) {
        t.assert_true(factory(-1) == nullptr);
        t.assert_true(factory(count) == nullptr);
        t.assert_true(factory(std::numeric_limits<int>::max()) == nullptr);
    });

    for (int id = 0; id < count; ++id) {
        auto * device = ggml_backend_reg_dev_get(reg, id);
        auto * local = factory(id);
        auto * legacy = ggml_backend_dev_buffer_type(device);
        if (!t.assert_true(local != nullptr)) continue;
        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        if (!t.assert_true(backend != nullptr)) continue;
        constexpr size_t size = 65536;
        int physical = -1;
        t.test("device_" + std::to_string(id) + "_identity_and_allocation", [&](testing & t) {
            t.assert_true(local == factory(id));
            t.assert_true(local != legacy);
            t.assert_true(ggml_backend_buft_get_device(local) == device);
            t.assert_true(ggml_backend_supports_buft(backend.get(), local));
            t.assert_true(!ggml_backend_buft_is_host(local));
            t.assert_equal(ggml_backend_buft_get_alignment(legacy), ggml_backend_buft_get_alignment(local));
            for (int other = 0; other < count; ++other) {
                t.assert_true(ggml_backend_dev_supports_buft(ggml_backend_reg_dev_get(reg, other), local) == (other == id));
            }
            ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(local, size));
            if (!t.assert_true(buffer != nullptr)) return;
            // Allocation selects the mapped physical device, which may differ from the GGML ordinal.
            if (!t.assert_true(cudaGetDevice(&physical) == cudaSuccess)) return;
            t.assert_true(ggml_backend_buffer_get_type(buffer.get()) == local);
            t.assert_true(ggml_backend_buffer_supports_views(buffer.get()));
            t.assert_equal(size, ggml_backend_buffer_get_size(buffer.get()));
            t.assert_equal(size_t(0), reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer.get())) % ggml_backend_buft_get_alignment(local));
            check_pointer(t, ggml_backend_buffer_get_base(buffer.get()), cudaMemoryTypeDevice, physical);
            check_buffer(t, buffer.get());
        });
        if (physical < 0) continue;

        t.test("device_" + std::to_string(id) + "_legacy_uvm_is_unchanged", [&](testing & t) {
            ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(legacy, size));
            if (!t.assert_true(buffer != nullptr)) return;
            const auto expected = std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") ? cudaMemoryTypeManaged : cudaMemoryTypeDevice;
            check_pointer(t, ggml_backend_buffer_get_base(buffer.get()), expected, physical);
            check_buffer(t, buffer.get());
        });

        t.test("device_" + std::to_string(id) + "_async_tensor_io", [&](testing & t) {
            ggml_context_ptr ctx(ggml_init({4096, nullptr, true}));
            if (!t.assert_true(ctx != nullptr)) return;
            auto * tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, 4);
            ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(local, size));
            if (!t.assert_true(buffer != nullptr)) return;
            if (!t.assert_true(ggml_backend_tensor_alloc(buffer.get(), tensor, ggml_backend_buffer_get_base(buffer.get())) == GGML_STATUS_SUCCESS)) return;
            std::vector<float> values(64, 7.0f), read(64);
            ggml_backend_tensor_set_async(backend.get(), tensor, values.data(), 0, 64*sizeof(float));
            ggml_backend_tensor_get_async(backend.get(), tensor, read.data(), 0, 64*sizeof(float));
            ggml_backend_synchronize(backend.get());
            t.assert_true(values == read);
            std::vector<float> packed(32, 2.0f), packed_read(32);
            ggml_backend_tensor_set_2d_async(backend.get(), tensor, packed.data(), 0, 8*sizeof(float), 4, 16*sizeof(float), 8*sizeof(float));
            ggml_backend_tensor_get_2d_async(backend.get(), tensor, packed_read.data(), 0, 8*sizeof(float), 4, 16*sizeof(float), 8*sizeof(float));
            ggml_backend_tensor_get_async(backend.get(), tensor, read.data(), 0, 64*sizeof(float));
            ggml_backend_synchronize(backend.get());
            t.assert_true(packed == packed_read);
            for (size_t n = 0; n < read.size(); ++n) t.assert_equal(n % 16 < 8 ? 2.0f : 7.0f, read[n]);
        });

        t.test("device_" + std::to_string(id) + "_graph_and_legacy_copy", [&](testing & t) {
            ggml_context_ptr ctx(ggml_init({16384, nullptr, true}));
            if (!t.assert_true(ctx != nullptr)) return;
            auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 256);
            auto * output = ggml_scale(ctx.get(), input, 2.0f);
            auto * copy = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 256);
            auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
            ggml_build_forward_expand(graph, output);
            ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(local, size));
            ggml_backend_buffer_ptr ordinary(ggml_backend_buft_alloc_buffer(legacy, size));
            if (!t.assert_true(buffer != nullptr && ordinary != nullptr)) return;
            auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer.get()));
            if (!t.assert_true(ggml_backend_tensor_alloc(buffer.get(), input, base) == GGML_STATUS_SUCCESS)) return;
            if (!t.assert_true(ggml_backend_tensor_alloc(buffer.get(), output, base + 4096) == GGML_STATUS_SUCCESS)) return;
            if (!t.assert_true(ggml_backend_tensor_alloc(ordinary.get(), copy, ggml_backend_buffer_get_base(ordinary.get())) == GGML_STATUS_SUCCESS)) return;
            std::vector<float> values(256, 3.25f), read(256);
            ggml_backend_tensor_set(copy, values.data(), 0, 256*sizeof(float));
            ggml_backend_tensor_copy(copy, input);
            if (!t.assert_true(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS)) return;
            ggml_backend_tensor_copy(output, copy);
            ggml_backend_tensor_get(copy, read.data(), 0, 256*sizeof(float));
            t.assert_true(std::all_of(read.begin(), read.end(), [](float x) { return x == 6.5f; }));
        });

        t.test("device_" + std::to_string(id) + "_quant_padding_matches_legacy", [&](testing & t) {
            ggml_context_ptr ctx(ggml_init({16384, nullptr, true}));
            if (!t.assert_true(ctx != nullptr)) return;
            for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
                auto * tensor = ggml_new_tensor_1d(ctx.get(), type, 32);
                const size_t bytes = ggml_backend_buft_get_alloc_size(local, tensor);
                t.assert_equal(ggml_backend_buft_get_alloc_size(legacy, tensor), bytes);
                ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(local, bytes));
                if (!t.assert_true(buffer != nullptr)) return;
                ggml_backend_buffer_clear(buffer.get(), 0xa5);
                if (!t.assert_true(ggml_backend_tensor_alloc(buffer.get(), tensor, ggml_backend_buffer_get_base(buffer.get())) == GGML_STATUS_SUCCESS)) return;
                std::vector<uint8_t> read(bytes);
                if (!t.assert_true(cudaMemcpy(read.data(), tensor->data, bytes, cudaMemcpyDeviceToHost) == cudaSuccess)) return;
                for (size_t b = 0; b < bytes; ++b) {
                    t.assert_equal(uint8_t(b < ggml_nbytes(tensor) ? 0xa5 : 0), read[b]);
                }
            }
        });

        t.test("device_" + std::to_string(id) + "_arena_lease_retains_device_storage", [&](testing & t) {
            arena_ptr arena(ggml_backend_memory_arena_new(local, size), ggml_backend_memory_arena_free);
            if (!t.assert_true(arena != nullptr)) return;
            const size_t alignment = ggml_backend_buft_get_alignment(local);
            auto * parent = ggml_backend_memory_arena_parent(arena.get());
            auto * base = static_cast<char *>(ggml_backend_buffer_get_base(parent));
            check_pointer(t, base, cudaMemoryTypeDevice, physical);
            if (!t.assert_true(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE))) return;
            if (!t.assert_true(ggml_backend_memory_arena_reserve_at(arena.get(), 1, alignment, 4096, alignment, 0, nullptr))) return;
            if (!t.assert_true(ggml_backend_memory_arena_commit(arena.get()))) return;
            lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(), 1), ggml_backend_memory_lease_free);
            if (!t.assert_true(lease != nullptr)) return;
            auto * view = ggml_backend_memory_lease_buffer(lease.get());
            t.assert_true(ggml_backend_buffer_get_type(view) == local);
            t.assert_true(ggml_backend_buffer_get_base(view) == base + alignment);
            check_pointer(t, ggml_backend_buffer_get_base(view), cudaMemoryTypeDevice, physical);
            ggml_backend_buffer_clear(parent, 0x11);
            arena.reset();
            check_buffer(t, view);
            std::vector<uint8_t> bytes(size);
            if (!t.assert_true(cudaMemcpy(bytes.data(), base, size, cudaMemcpyDeviceToHost) == cudaSuccess)) return;
            t.assert_true(std::all_of(bytes.begin(), bytes.begin() + alignment, [](uint8_t x) { return x == 0x11; }));
            t.assert_true(std::all_of(bytes.begin() + alignment + 4096, bytes.end(), [](uint8_t x) { return x == 0x11; }));
            lease.reset();
        });

        t.test("device_" + std::to_string(id) + "_allocation_failure_is_recoverable", [&](testing & t) {
            // This impossible request fails without consuming the machine's available VRAM.
            ggml_backend_buffer_ptr failed(ggml_backend_buft_alloc_buffer(local, SIZE_MAX));
            t.assert_true(failed == nullptr);
            t.assert_true(cudaGetLastError() == cudaSuccess);
            arena_ptr arena(ggml_backend_memory_arena_new(local, SIZE_MAX), ggml_backend_memory_arena_free);
            t.assert_true(arena == nullptr);
            t.assert_true(cudaGetLastError() == cudaSuccess);
            ggml_backend_buffer_ptr recovered(ggml_backend_buft_alloc_buffer(local, size));
            if (!t.assert_true(recovered != nullptr)) return;
            check_pointer(t, ggml_backend_buffer_get_base(recovered.get()), cudaMemoryTypeDevice, physical);
            check_buffer(t, recovered.get());
        });

        t.test("device_" + std::to_string(id) + "_zero_size_uses_common_empty_buffer", [&](testing & t) {
            ggml_backend_buffer_ptr empty(ggml_backend_buft_alloc_buffer(local, 0));
            if (!t.assert_true(empty != nullptr)) return;
            t.assert_equal(size_t(0), ggml_backend_buffer_get_size(empty.get()));
            t.assert_true(ggml_backend_buffer_get_type(empty.get()) == local);
            arena_ptr arena(ggml_backend_memory_arena_new(local, 0), ggml_backend_memory_arena_free);
            t.assert_true(arena == nullptr);
        });
    }
    return t.summary();
}

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "testing.h"

#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>

using host_type_fn = ggml_backend_buffer_type_t (*)(int);
using host_register_fn = ggml_backend_buffer_t (*)(int, ggml_backend_buffer_t);

// A mapped allocation must support actual device writes, not just host memcpy.
static void mapped_write(testing & t, ggml_backend_buffer_t buffer, uint8_t value) {
    void * host = ggml_backend_buffer_get_base(buffer);
    void * mapped = nullptr;
    if (!t.assert_true(cudaHostGetDevicePointer(&mapped, host, 0) == cudaSuccess)) return;
    t.assert_true(cudaMemset(mapped, value, ggml_backend_buffer_get_size(buffer)) == cudaSuccess);
    t.assert_true(cudaDeviceSynchronize() == cudaSuccess);
    auto * bytes = static_cast<uint8_t *>(host);
    t.assert_true(std::all_of(bytes, bytes + ggml_backend_buffer_get_size(buffer), [value](uint8_t x) { return x == value; }));
}

int main() {
    ggml_backend_load_all();
    auto * reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || !ggml_backend_reg_dev_count(reg)) return 77;
    testing t;
    auto type_fn = reinterpret_cast<host_type_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_host_buffer_type"));
    auto register_fn = reinterpret_cast<host_register_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_host_buffer_register"));
    t.test("strict_host_hooks_exist", [&](testing & t) { t.assert_true(type_fn && register_fn); });
    if (!type_fn || !register_fn) return t.summary();
    auto * type = type_fn(0);
    if (!t.assert_true(type != nullptr)) return t.summary();
    if (std::getenv("GGML_CUDA_NO_PINNED")) {
        t.test("disabled_pinning_does_not_fall_back", [&](testing & t) {
            ggml_backend_buffer_ptr owner(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 8192));
            ggml_backend_buffer_ptr allocated(ggml_backend_buft_alloc_buffer(type, 4096));
            ggml_backend_buffer_ptr registered(register_fn(0, owner.get()));
            t.assert_true(!allocated && !registered);
        });
        return t.summary();
    }
    t.test("invalid_devices_and_owners", [&](testing & t) {
        t.assert_true(type_fn(-1) == nullptr);
        t.assert_true(type_fn(int(ggml_backend_reg_dev_count(reg))) == nullptr);
        t.assert_true(register_fn(0, nullptr) == nullptr);
    });
    t.test("allocated_mapping_flags_and_last_view_lifetime", [&](testing & t) {
        ggml_backend_buffer_ptr root(ggml_backend_buft_alloc_buffer(type, 8192));
        if (!t.assert_true(root != nullptr)) return;
        t.assert_true(ggml_backend_buffer_get_type(root.get()) == type);
        t.assert_true(ggml_backend_buffer_is_host(root.get()));
        unsigned int flags = 0;
        t.assert_true(cudaHostGetFlags(&flags, ggml_backend_buffer_get_base(root.get())) == cudaSuccess);
        t.assert_true((flags & cudaHostAllocMapped) != 0);
#ifdef _WIN32
        t.assert_true((flags & cudaHostAllocWriteCombined) == 0);
#else
        t.assert_true((flags & cudaHostAllocWriteCombined) != 0);
#endif
        mapped_write(t, root.get(), 0x31);
        ggml_backend_buffer_ptr duplicate(register_fn(0, root.get()));
        t.assert_true(duplicate == nullptr);
        mapped_write(t, root.get(), 0x32);
        ggml_backend_buffer_ptr view(ggml_backend_buffer_view(root.get(), 128, 4096));
        if (!t.assert_true(view != nullptr)) return;
        root.reset();
        ggml_backend_buffer_clear(view.get(), 0x52);
        t.assert_equal(uint8_t(0x52), *static_cast<uint8_t *>(ggml_backend_buffer_get_base(view.get())));
    });
    t.test("registered_storage_retains_owner_and_unregisters_once", [&](testing & t) {
        ggml_backend_buffer_ptr owner(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 16384));
        auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(owner.get()));
        size_t offset = (4096 - base % 4096) % 4096;
        ggml_backend_buffer_ptr page(ggml_backend_buffer_view(owner.get(), offset, 8192));
        ggml_backend_buffer_ptr registered(register_fn(0, page.get()));
        if (!t.assert_true(registered != nullptr)) return;
        mapped_write(t, registered.get(), 0x19);
        // An existing registration must not be adopted or undone by a failed second registration.
        ggml_backend_buffer_ptr duplicate(register_fn(0, page.get()));
        t.assert_true(duplicate == nullptr);
        mapped_write(t, registered.get(), 0x21);
        registered.reset();
        ggml_backend_buffer_ptr again(register_fn(0, page.get()));
        if (!t.assert_true(again != nullptr)) return;
        owner.reset(); page.reset();
        mapped_write(t, again.get(), 0x29);
    });
    t.test("allocation_failure_is_recoverable", [&](testing & t) {
        ggml_backend_buffer_ptr failed(ggml_backend_buft_alloc_buffer(type, SIZE_MAX));
        t.assert_true(failed == nullptr);
        t.assert_true(cudaGetLastError() == cudaSuccess);
        ggml_backend_buffer_ptr valid(ggml_backend_buft_alloc_buffer(type, 4096));
        if (t.assert_true(valid != nullptr)) mapped_write(t, valid.get(), 0x42);
    });
    return t.summary();
}

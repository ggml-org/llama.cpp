#include "../src/llama-kv-stream-host.h"
#include "../src/llama-kv-stream-binding.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"

#include <cstring>
#include <limits>

static llama_kv_stream_host_config config(int k = GGML_TYPE_Q8_0, int v = GGML_TYPE_Q4_0) {
    return {71, {k, v, 256, 256, 4, 256, 128},
        {{k, true, true, true, true}, {v, true, true, true, true}, true, true}, 257, 3};
}

struct allocation_probe {
    ggml_backend_buffer_type type = *ggml_backend_cpu_buffer_type();
    size_t calls = 0;
    bool fail = false, fallback = false, throws = false;
    allocation_probe() {
        type.context = this;
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
            auto & p = *static_cast<allocation_probe *>(type->context);
            ++p.calls;
            if (p.throws) throw std::bad_alloc();
            if (p.fail) return ggml_backend_buffer_t(nullptr);
            auto * result = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (result && !p.fallback) result->buft = type;
            return result;
        };
    }
};

int main(int argc, char ** argv) {
    testing t;
    ggml_backend_dev_t device = nullptr;
    auto * storage_type = ggml_backend_cpu_buffer_type();
    if (argc > 1 && std::strcmp(argv[1], "--cuda") == 0) {
        ggml_backend_load_all();
        auto * reg = ggml_backend_reg_by_name("CUDA");
        if (!t.assert_true(reg && ggml_backend_reg_dev_count(reg))) return t.summary();
        device = ggml_backend_reg_dev_get(reg, 0);
        storage_type = llama_kv_stream_host_buffer_type(device);
        if (!t.assert_true(storage_type != nullptr)) return t.summary();
    }
    t.test("separate_planes_padded_context_and_layer_bounds", [&](testing & t) {
        auto c = config();
        auto host = llama_kv_stream_host::create(c, storage_type);
        if (!t.assert_true(host != nullptr)) return;
        t.assert_equal(uint64_t(71), host->cache_id());
        t.assert_equal(size_t(512), host->layout().tokens);
        t.assert_equal(size_t(557056), host->layout().k_bytes);
        t.assert_equal(size_t(294912), host->layout().v_bytes);
        t.assert_equal(size_t(851968), host->stride());
        t.assert_equal(size_t(2555904), host->bytes());
        for (uint32_t i = 0; i < c.layers; ++i) {
            llama_kv_stream_host_layer layer;
            if (!t.assert_true(host->layer(i, layer))) return;
            t.assert_equal(size_t(0), reinterpret_cast<uintptr_t>(layer.k) % c.shape.alignment);
            t.assert_equal(size_t(0), reinterpret_cast<uintptr_t>(layer.v) % c.shape.alignment);
            t.assert_equal(host->layout().v_offset, size_t(static_cast<char *>(layer.v) - static_cast<char *>(layer.k)));
            auto * bytes = static_cast<uint8_t *>(layer.k);
            t.assert_true(std::all_of(bytes, bytes + host->stride(), [](uint8_t b) { return b == 0; }));
        }
        llama_kv_stream_host_layer invalid{reinterpret_cast<void *>(1), reinterpret_cast<void *>(2)};
        t.assert_true(!host->layer(c.layers, invalid));
        t.assert_true(invalid.k == reinterpret_cast<void *>(1));
    });
    t.test("all_online_quant_pairs_use_generic_row_sizes", [](testing & t) {
        for (int k : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL}) {
            for (int v : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL}) {
                auto c = config(k, v); c.context_tokens = 1; c.layers = 2;
                auto host = llama_kv_stream_host::create(c, ggml_backend_cpu_buffer_type());
                if (!t.assert_true(host != nullptr)) continue;
                t.assert_equal(ggml_row_size(ggml_type(k), 256)*4*256, host->layout().k_bytes);
                t.assert_equal(ggml_row_size(ggml_type(v), 256)*4*256, host->layout().v_bytes);
                t.assert_equal(host->stride()*2, host->bytes());
            }
        }
    });
    t.test("invalid_or_overflowing_metadata_never_allocates", [](testing & t) {
        allocation_probe p;
        for (int bad = 0; bad < 8; ++bad) {
            auto c = config();
            if (bad == 0) c.cache_id = 0;
            if (bad == 1) c.layers = 0;
            if (bad == 2) c.context_tokens = 0;
            if (bad == 3) c.context_tokens = SIZE_MAX;
            if (bad == 4) c.capabilities.k.online_write = false;
            if (bad == 5) c.shape.type_v = -1;
            if (bad == 6) c.shape.alignment = 3;
            if (bad == 7) { c.layers = UINT32_MAX; c.context_tokens = size_t(1) << 30; }
            t.assert_true(!llama_kv_stream_host::create(c, &p.type));
        }
        t.assert_equal(size_t(0), p.calls);
    });
    t.test("allocation_failure_and_pageable_fallback_are_rejected", [](testing & t) {
        allocation_probe p;
        p.fail = true;
        t.assert_true(!llama_kv_stream_host::create(config(), &p.type));
        p.fail = false; p.fallback = true;
        t.assert_true(!llama_kv_stream_host::create(config(), &p.type));
        p.fallback = false;
        t.assert_true(llama_kv_stream_host::create(config(), &p.type) != nullptr);
        t.assert_equal(size_t(3), p.calls);
        p.throws = true;
        t.assert_true(!llama_kv_stream_host::create(config(), &p.type));
    });
    t.test("import_preserves_bytes_and_retains_buffer", [](testing & t) {
        auto c = config(); c.context_tokens = 1; c.layers = 1;
        auto original = llama_kv_stream_host::create(c, ggml_backend_cpu_buffer_type());
        if (!t.assert_true(original != nullptr)) return;
        ggml_backend_buffer_clear(original->buffer(), 0x31);
        auto host = llama_kv_stream_host::from_buffer(c, ggml_backend_cpu_buffer_type(), original->buffer());
        if (!t.assert_true(host != nullptr)) return;
        original.reset();
        llama_kv_stream_host_layer layer;
        t.assert_true(host->layer(0, layer));
        t.assert_equal(uint8_t(0x31), *static_cast<uint8_t *>(layer.k));
        t.assert_equal(uint8_t(0x31), *static_cast<uint8_t *>(layer.v));
        ggml_backend_buffer_ptr small(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 128));
        t.assert_true(!llama_kv_stream_host::from_buffer(c, ggml_backend_cpu_buffer_type(), small.get()));
        allocation_probe other;
        t.assert_true(!llama_kv_stream_host::from_buffer(c, &other.type, host->buffer()));
        t.assert_true(!llama_kv_stream_host::from_buffer(c, nullptr, host->buffer()));
    });
    t.test("host_lifetime_is_independent_of_device_binding", [&](testing & t) {
        auto host = llama_kv_stream_host::create(config(), storage_type);
        if (!t.assert_true(host != nullptr)) return;
        std::weak_ptr<llama_kv_stream_host> weak = host;
        struct resources : llama_memory_executable { std::shared_ptr<llama_kv_stream_host> host; };
        struct completion : llama_memory_executor_backend { bool drain() override { return true; } } done;
        auto * arena = ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(), 1024*1024);
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena, 0));
        auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena)));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena, 1, (128 - base % 128) % 128, 851968, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena));
        auto * lease = ggml_backend_memory_arena_acquire(arena, 1);
        llama_kv_stream_binding binding(host->cache_id(), ggml_backend_cpu_buffer_type());
        llama_kv_stream_policy_config policy;
        policy.shape = host->config().shape; policy.capabilities = host->config().capabilities;
        policy.pool_bytes = 851968; policy.layers = 1;
        if (!t.assert_true(binding.bind(lease, policy, [&](const auto &) {
            auto result = std::make_unique<resources>(); result->host = host; return result;
        }))) { ggml_backend_memory_lease_free(lease); ggml_backend_memory_arena_free(arena); return; }
        ggml_backend_memory_lease_free(lease);
        ggml_backend_memory_arena_free(arena);
        auto pin = binding.acquire();
        host.reset();
        t.assert_true(!weak.expired());
        t.assert_true(binding.detach(done).status == llama_memory_executor_status::pending);
        pin.reset();
        t.assert_true(binding.detach(done).status == llama_memory_executor_status::retired);
        t.assert_true(weak.expired());
    });
    t.test("unaligned_plane_sizes_have_aligned_layer_stride", [](testing & t) {
        auto c = config(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
        c.shape.head_dim_k = c.shape.head_dim_v = 64;
        c.shape.heads = c.shape.page_tokens = 1;
        c.context_tokens = 3;
        auto host = llama_kv_stream_host::create(c, ggml_backend_cpu_buffer_type());
        if (!t.assert_true(host != nullptr)) return;
        t.assert_equal(size_t(120), host->layout().k_bytes);
        t.assert_equal(size_t(128), host->layout().v_offset);
        t.assert_equal(size_t(144), host->layout().v_bytes);
        t.assert_equal(size_t(384), host->stride());
        llama_kv_stream_host_layer first, second;
        t.assert_true(host->layer(0, first) && host->layer(1, second));
        t.assert_equal(host->stride(), size_t(static_cast<char *>(second.k) - static_cast<char *>(first.k)));
        t.assert_equal(size_t(0), reinterpret_cast<uintptr_t>(second.v) % 128);
    });
    t.test("no_host_adapter_fallback_for_cpu", [](testing & t) {
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        t.assert_true(llama_kv_stream_host_buffer_type(ggml_backend_get_device(cpu.get())) == nullptr);
        t.assert_true(llama_kv_stream_host_register(nullptr, nullptr) == nullptr);
    });
    if (device) t.test("registered_adapter_preserves_imported_planes", [&](testing & t) {
        auto c = config(); c.layers = 1;
        ggml_backend_buffer_ptr owner(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 1024*1024 + 4096));
        if (!t.assert_true(owner != nullptr)) return;
        ggml_backend_buffer_clear(owner.get(), 0x41);
        auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(owner.get()));
        ggml_backend_buffer_ptr page(ggml_backend_buffer_view(owner.get(), (4096 - base % 4096) % 4096, 1024*1024));
        ggml_backend_buffer_ptr pinned(llama_kv_stream_host_register(device, page.get()));
        if (!t.assert_true(pinned != nullptr)) return;
        auto host = llama_kv_stream_host::from_buffer(c, storage_type, pinned.get());
        if (!t.assert_true(host != nullptr)) return;
        owner.reset(); page.reset(); pinned.reset();
        llama_kv_stream_host_layer layer;
        t.assert_true(host->layer(0, layer));
        t.assert_equal(uint8_t(0x41), *static_cast<uint8_t *>(layer.k));
        t.assert_equal(uint8_t(0x41), *static_cast<uint8_t *>(layer.v));
    });
    return t.summary();
}

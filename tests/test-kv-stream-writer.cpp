#include "../src/llama-kv-stream-resident.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include "../src/llama-memory-executor-cuda.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"
#include "../ggml/src/ggml-backend-impl.h"
#include <stdexcept>

#include <chrono>
#include <cmath>
#include <cstring>

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct writer_fixture {
    ggml_backend_t backend;
    bool cuda;
    std::shared_ptr<llama_kv_stream_host> host;
    std::shared_ptr<llama_kv_stream_content> content;
    llama_kv_stream_policy_config policy;
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    std::unique_ptr<llama_kv_stream_binding> binding;
    llama_memory_execution pin;
    llama_kv_stream_resident * resident = nullptr;

    // Fixed Qwen-like geometry and budget, independent of the writer implementation being compared.
    writer_fixture(ggml_backend_t backend, bool cuda, ggml_type k = GGML_TYPE_Q8_0, ggml_type v = GGML_TYPE_Q4_0, bool tiny = false) : backend(backend), cuda(cuda) {
        llama_kv_stream_host_config c{91, {k, v, 256, 256, 4, 256, 128},
            {{k, true, true, true, true}, {v, true, true, true, true}, true, true}, 1024, 1};
        auto * device = ggml_backend_get_device(backend);
        if (cuda) {
            auto * reg = ggml_backend_dev_backend_reg(device);
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
            if (get && get() && get()->version >= 3) c.capabilities = get()->capabilities(backend,k,v);
        }
        auto * ht = cuda ? llama_kv_stream_host_buffer_type(device) : ggml_backend_cpu_buffer_type();
        auto * dt = cuda ? llama_kv_stream_device_buffer_type(device) : ggml_backend_cpu_buffer_type();
        GGML_ASSERT(ht && dt);
        host = llama_kv_stream_host::create(c, ht); GGML_ASSERT(host);
        content = std::make_shared<llama_kv_stream_content>(host);
        ggml_kv_stream_layout page; GGML_ASSERT(ggml_kv_stream_layout_make(c.shape, 256, page).status == ggml_kv_stream_status::success);
        policy.shape = c.shape; policy.capabilities = c.capabilities; policy.layers = 1;
        ggml_kv_stream_execution execution;
        GGML_ASSERT(ggml_kv_stream_resolve(c.shape,c.capabilities,256,execution).status == ggml_kv_stream_status::success);
        policy.pool_bytes = page.bytes*(tiny ? 8 : 16)+execution.conversion.bytes; policy.initial_ring_slots = tiny ? 1 : 0;
        arena.reset(ggml_backend_memory_arena_new(dt, policy.pool_bytes + 256)); GGML_ASSERT(arena);
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena.get())));
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), 0));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 3, 128 + (128 - base%128)%128, policy.pool_bytes, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 3));
        binding = std::make_unique<llama_kv_stream_binding>(91, dt);
        GGML_ASSERT(binding->bind(lease.get(), policy, [&](const auto & view) {
            auto r = llama_kv_stream_resident::create(view, content, backend); resident = r.get(); return r;
        }));
        pin = binding->acquire(); GGML_ASSERT(pin);
        GGML_ASSERT(resident->synchronize(1024));
    }
};

struct encoder {
    ggml_backend_t backend;
    bool cuda;
    size_t rows, row_bytes;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * source = nullptr, * output = nullptr;
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<llama_memory_cuda_executor> execution;

    // Ordinary SET_ROWS is the independent GPU quantization control; its source buffer is already on the device.
    encoder(writer_fixture & f, size_t rows, ggml_type type) : backend(f.backend), cuda(f.cuda), rows(rows), row_bytes(ggml_row_size(type, 1024)) {
        context.reset(ggml_init({65536, nullptr, true}));
        source = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1024, int64_t(rows));
        auto * indices = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, int64_t(rows));
        auto * dst = ggml_new_tensor_2d(context.get(), type, 1024, int64_t(rows));
        output = ggml_set_rows(context.get(), dst, source, indices);
        GGML_ASSERT(ggml_backend_supports_op(backend, output));
        graph = ggml_new_graph_custom(context.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend)); GGML_ASSERT(buffer);
        std::vector<float> input(rows*1024);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i%1021)*.07f);
        std::vector<int64_t> idx(rows); for (size_t i = 0; i < rows; ++i) idx[i] = int64_t(i);
        ggml_backend_tensor_set(source, input.data(), 0, input.size()*sizeof(float));
        ggml_backend_tensor_set(indices, idx.data(), 0, idx.size()*sizeof(int64_t));
        if (cuda) { execution = std::make_unique<llama_memory_cuda_executor>(backend); GGML_ASSERT(execution->bind(graph, {}, 1)); }
    }
    void run() {
        if (cuda) { GGML_ASSERT(execution->compute_async({}, 1) == GGML_STATUS_SUCCESS); GGML_ASSERT(execution->drain()); }
        else GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }
};

// Existing coalesced/row-wise callers of the committed content and resident APIs, with identical GPU quantization.
static void baseline(writer_fixture & f, encoder & e, bool rowwise, std::vector<uint8_t> & bytes) {
    e.run();
    if (rowwise) for (size_t r = 0; r < e.rows; ++r) ggml_backend_tensor_get(e.output, bytes.data() + r*e.row_bytes, r*e.row_bytes, e.row_bytes);
    else ggml_backend_tensor_get(e.output, bytes.data(), 0, bytes.size());
    llama_kv_stream_write write;
    GGML_ASSERT(f.content->prepare({{0, ggml_kv_stream_operand::k, 0, bytes.data(), bytes.size()}}, write));
    GGML_ASSERT(f.content->commit(write));
    GGML_ASSERT(f.resident->synchronize(1024));
}

// Compare canonical bytes and the independently addressed resident plane.
static void check_bytes(testing & t, writer_fixture & f, bool value, size_t first, const std::vector<uint8_t> & expected) {
    llama_kv_stream_host_layer host; f.host->layer(0, host);
    const size_t stride = value ? f.host->layout().v_token_bytes : f.host->layout().k_token_bytes;
    t.assert_true(std::memcmp(static_cast<char *>(value ? host.v : host.k) + first*stride, expected.data(), expected.size()) == 0);
    llama_kv_stream_policy_layout layout;
    GGML_ASSERT(llama_kv_stream_policy_layout_make(f.policy, f.binding->view()->initial_policy, 0, layout).status == llama_kv_stream_policy_status::success);
    auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
    ggml_context_ptr context(ggml_init({4096, nullptr, true}));
    auto * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, int64_t(expected.size()));
    const size_t offset = layout.layers[0].offset + (value ? layout.layers[0].planes.v_offset : 0) + first*stride;
    GGML_ASSERT(ggml_backend_tensor_alloc(buffer, tensor, static_cast<char *>(f.binding->view()->base) + offset) == GGML_STATUS_SUCCESS);
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    t.assert_true(actual == expected);
}

// Fail after a real D2D copy, so rollback must repair a partially changed mirror.
struct copy_fault {
    ggml_backend_buffer_t buffer;
    ggml_backend_t backend;
    decltype(ggml_backend_buffer_i::set_tensor) original_set;
    decltype(ggml_backend_i::cpy_tensor_async) original_async;
    int calls = 0;
    inline static copy_fault * active = nullptr;
    copy_fault(ggml_backend_t backend, ggml_backend_buffer_t buffer) : buffer(buffer), backend(backend),
        original_set(buffer->iface.set_tensor), original_async(backend->iface.cpy_tensor_async) {
        GGML_ASSERT(!active); active = this;
        if (original_async) backend->iface.cpy_tensor_async = [](ggml_backend_t a, ggml_backend_t b, const ggml_tensor * src, ggml_tensor * dst) {
            const bool result = active->original_async(a, b, src, dst);
            if (++active->calls == 2) throw std::runtime_error("injected queued second tile failure");
            return result;
        };
        else buffer->iface.set_tensor = [](ggml_backend_buffer_t b, ggml_tensor * dst, const void * data, size_t offset, size_t size) {
            active->original_set(b, dst, data, offset, size);
            if (++active->calls == 2) throw std::runtime_error("injected second tile failure");
        };
    }
    ~copy_fault() { buffer->iface.set_tensor = original_set; backend->iface.cpy_tensor_async = original_async; active = nullptr; }
};

// Observe final input-buffer release without changing its native free behavior.
struct release_probe {
    ggml_backend_buffer_t buffer;
    decltype(ggml_backend_buffer_i::free_buffer) original;
    int frees = 0;
    inline static release_probe * active = nullptr;
    explicit release_probe(ggml_backend_buffer_t buffer) : buffer(buffer), original(buffer->iface.free_buffer) {
        GGML_ASSERT(!active); active = this;
        buffer->iface.free_buffer = [](ggml_backend_buffer_t b) { ++active->frees; active->original(b); };
    }
    ~release_probe() { if (!frees) buffer->iface.free_buffer = original; active = nullptr; }
};

int main(int argc, char ** argv) {
    const bool baseline_only = argc > 1 && std::strcmp(argv[1], "--bench-baseline") == 0;
    const bool bench = baseline_only || (argc > 1 && std::strcmp(argv[1], "--bench") == 0);
    const bool cuda = bench || (argc > 1 && std::strcmp(argv[1], "--cuda") == 0);
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev, nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    if (bench) {
        std::cout << "policy,batch,median_us,p95_us,d2h_bytes,d2h_calls,h2d_bytes,h2d_calls,d2d_bytes,d2d_calls\n";
        for (size_t rows : {size_t(32), size_t(256), size_t(512)}) {
            writer_fixture f(backend.get(), true); encoder e(f, rows, GGML_TYPE_Q8_0);
            std::vector<uint8_t> bytes(rows*e.row_bytes);
            if (!baseline_only) GGML_ASSERT(f.resident->configure_writes(512));
            for (int mode = 0; mode < (baseline_only ? 2 : 3); ++mode) {
                const bool rowwise = mode == 1, staged = mode == 2;
                std::vector<double> times;
                for (int repeat = 0; repeat < 110; ++repeat) {
                    const auto start = std::chrono::steady_clock::now();
                    if (staged) {
                        GGML_ASSERT(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
                        GGML_ASSERT(f.resident->synchronize(1024));
                    } else baseline(f, e, rowwise, bytes);
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                    if (repeat >= 10) times.push_back(us);
                }
                std::sort(times.begin(), times.end());
                const auto stats = f.resident->last_write_stats();
                std::cout << (staged ? "staged" : rowwise ? "rowwise" : "coalesced") << ',' << rows << ',' << times[50] << ',' << times[95] << ','
                    << bytes.size() << ',' << (staged ? stats.d2h_calls : rowwise ? rows : 1) << ',' << f.resident->last_upload_bytes() << ',' << f.resident->last_upload_calls()
                    << ',' << (staged ? stats.d2d_bytes : 0) << ',' << (staged ? stats.d2d_calls : 0) << '\n';
            }
        }
        return 0;
    }
    testing t;
    t.test("ordinary_set_rows_baseline_populates_authoritative_cache", [&](testing & t) {
        writer_fixture f(backend.get(), cuda); encoder e(f, 33, GGML_TYPE_Q8_0);
        std::vector<uint8_t> bytes(e.rows*e.row_bytes);
        baseline(f, e, false, bytes);
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        t.assert_true(std::memcmp(host.k, bytes.data(), bytes.size()) == 0);
        t.assert_equal(bytes.size(), f.resident->last_upload_bytes());
    });
    t.test("generated_ticket_is_atomic_and_cancellable", [&](testing & t) {
        writer_fixture f(backend.get(), cuda);
        uint8_t value = 7; llama_kv_stream_write ticket;
        const auto gen = f.content->generation();
        t.assert_true(!f.content->prepare_generated({{0, ggml_kv_stream_operand::k, 0, nullptr, 1}}, [](const auto &, void * out) {
            *static_cast<uint8_t *>(out) = 7; return false;
        }, ticket));
        t.assert_equal(gen, f.content->generation());
        t.assert_true(f.content->prepare_generated({{0, ggml_kv_stream_operand::k, 0, nullptr, 1}}, [&](const auto &, void * out) {
            *static_cast<uint8_t *>(out) = value; return true;
        }, ticket));
        ticket.cancel();
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        t.assert_equal(uint8_t(0), *static_cast<uint8_t *>(host.k));
    });
    t.test("staged_rows_match_stock_quantization_and_avoid_h2d", [&](testing & t) {
        writer_fixture f(backend.get(), cuda); encoder e(f, 33, GGML_TYPE_Q8_0);
        e.run();
        std::vector<uint8_t> expected(e.rows*e.row_bytes);
        ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 17, e.source));
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        t.assert_true(std::memcmp(static_cast<uint8_t *>(host.k) + 17*e.row_bytes, expected.data(), expected.size()) == 0);
        t.assert_true(f.resident->synchronize(1024));
        t.assert_equal(size_t(0), f.resident->last_upload_bytes());
        auto stats = f.resident->last_write_stats();
        t.assert_equal(expected.size(), stats.d2h_bytes);
        t.assert_equal(expected.size(), stats.d2d_bytes);
        t.assert_equal(size_t(1), stats.graph_submissions);
    });
    t.test("quant_pairs_partial_batches_and_scratch_reuse", [&](testing & t) {
        for (auto pair : {std::pair{GGML_TYPE_F16, GGML_TYPE_F16}, std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0},
                std::pair{GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}, std::pair{GGML_TYPE_Q5_1, GGML_TYPE_Q4_1},
                std::pair{GGML_TYPE_BF16, GGML_TYPE_Q5_0}, std::pair{GGML_TYPE_IQ4_NL, GGML_TYPE_F32}}) {
            writer_fixture f(backend.get(), cuda, pair.first, pair.second);
            if (!t.assert_true(f.resident->configure_writes(512))) continue;
            for (bool value : {false, true}) for (size_t rows : {size_t(1), size_t(7), size_t(256), size_t(257), size_t(512)}) {
                encoder e(f, rows, value ? pair.second : pair.first); e.run();
                std::vector<uint8_t> expected(rows*e.row_bytes);
                ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
                if (!t.assert_true(f.resident->write_rows(0, value ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k, 3, e.source))) continue;
                check_bytes(t, f, value, 3, expected);
                t.assert_true(f.resident->synchronize(1024));
                t.assert_equal(size_t(0), f.resident->last_upload_bytes());
                const auto stats = f.resident->last_write_stats();
                t.assert_equal(expected.size(), stats.host_payload_bytes);
                t.assert_equal(expected.size(), stats.d2h_bytes);
                t.assert_equal(expected.size(), stats.d2d_bytes);
                t.assert_true(stats.device_scratch_bytes < f.policy.pool_bytes);
            }
        }
    });
    t.test("tiled_failure_preserves_host_and_restores_mirror", [&](testing & t) {
        writer_fixture f(backend.get(), cuda, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true);
        encoder e(f, 512, GGML_TYPE_Q8_0); e.run();
        std::vector<uint8_t> expected(e.rows*e.row_bytes), zero(expected.size(), 0);
        ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        const auto generation = f.content->generation();
        bool caught = false;
        {
            copy_fault fault(backend.get(), ggml_backend_memory_lease_buffer(f.lease.get()));
            try { f.resident->write_rows(0, ggml_kv_stream_operand::k, 5, e.source); }
            catch (const std::runtime_error &) { caught = true; }
            t.assert_equal(2, fault.calls);
        }
        t.assert_true(caught && !f.resident->ready(1024));
        t.assert_equal(generation, f.content->generation());
        t.assert_true(f.resident->synchronize(1024));
        check_bytes(t, f, false, 5, zero);
        t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 5, e.source));
        check_bytes(t, f, false, 5, expected);
        auto stats = f.resident->last_write_stats();
        t.assert_true(stats.tile_rows < 512 && stats.graph_submissions > 1);
        t.assert_true(stats.device_scratch_bytes <= f.policy.pool_bytes/8);
    });
    t.test("queued_reader_completes_before_resident_overwrite", [&](testing & t) {
        writer_fixture f(backend.get(), cuda, GGML_TYPE_F32, GGML_TYPE_F32);
        encoder e(f, 32, GGML_TYPE_F32);
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        std::vector<float> old(32*1024), next(old.size(), 7.0f), observed(old.size());
        std::memcpy(old.data(), host.k, old.size()*sizeof(float));
        ggml_backend_tensor_set(e.source, next.data(), 0, next.size()*sizeof(float));
        llama_kv_stream_policy_layout layout;
        GGML_ASSERT(llama_kv_stream_policy_layout_make(f.policy, f.binding->view()->initial_policy, 0, layout).status == llama_kv_stream_policy_status::success);
        ggml_context_ptr context(ggml_init({65536, nullptr, true}));
        auto * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, int64_t(old.size()));
        GGML_ASSERT(ggml_backend_tensor_alloc(ggml_backend_memory_lease_buffer(f.lease.get()), input,
            static_cast<char *>(f.binding->view()->base) + layout.layers[0].offset) == GGML_STATUS_SUCCESS);
        auto * output = ggml_scale(context.get(), input, 1.0f);
        auto * graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_ptr output_buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
        if (cuda) {
            llama_memory_cuda_executor reader(backend.get());
            t.assert_true(reader.bind(graph, {f.lease.get()}, 1));
            t.assert_true(reader.compute_async({f.lease.get()}, 1) == GGML_STATUS_SUCCESS);
            t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
            t.assert_true(reader.drain());
        } else {
            t.assert_true(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
            t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
        }
        ggml_backend_tensor_get(output, observed.data(), 0, observed.size()*sizeof(float));
        t.assert_true(observed == old);
        t.assert_true(std::memcmp(host.k, next.data(), next.size()*sizeof(float)) == 0);
    });
    t.test("invalid_source_and_bounds_do_not_mutate_content", [&](testing & t) {
        writer_fixture f(backend.get(), cuda); encoder e(f, 33, GGML_TYPE_Q8_0);
        const auto generation = f.content->generation();
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
        t.assert_true(!f.resident->configure_writes(0));
        t.assert_true(f.resident->configure_writes(32));
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
        t.assert_true(f.resident->configure_writes(512));
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 1000, e.source));
        t.assert_true(!f.resident->write_rows(1, ggml_kv_stream_operand::k, 0, e.source));
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::none, 0, e.source));
        auto invalid = *e.source; --invalid.ne[0];
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, &invalid));
        t.assert_equal(generation, f.content->generation());
        f.resident->release_write_workspace();
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
    });
    t.test("completed_source_graph_is_not_recomputed", [&](testing & t) {
        writer_fixture f(backend.get(), cuda); encoder e(f, 32, GGML_TYPE_Q8_0); e.run();
        std::vector<uint8_t> expected(e.rows*e.row_bytes);
        ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        const auto saved = *e.source;
        auto input = saved;
        e.source->op = GGML_OP_SCALE; e.source->src[0] = &input;
        const float scale = 3.0f; std::memcpy(e.source->op_params, &scale, sizeof(scale));
        const bool ok = f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source);
        *e.source = saved;
        t.assert_true(ok);
        check_bytes(t, f, false, 0, expected);
    });
    t.test("generated_failure_preserves_existing_ticket_and_admission", [&](testing & t) {
        writer_fixture f(backend.get(), cuda);
        uint8_t value = 4; llama_kv_stream_write ticket;
        t.assert_true(f.content->prepare({{0, ggml_kv_stream_operand::k, 0, &value, 1}}, ticket));
        int calls = 0;
        t.assert_true(!f.content->prepare_generated({{0, ggml_kv_stream_operand::v, 0, &value, 1}}, [&](const auto &, void *) { ++calls; return true; }, ticket));
        t.assert_equal(0, calls);
        t.assert_true(!f.content->prepare_generated({{0, ggml_kv_stream_operand::v, 0, nullptr, 1}}, [&](const auto &, void *) -> bool {
            t.assert_true(!f.content->invalidate());
            throw std::bad_alloc();
        }, ticket));
        t.assert_true(f.content->commit(ticket));
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        t.assert_equal(value, *static_cast<uint8_t *>(host.k));
        t.assert_equal(uint8_t(0), *static_cast<uint8_t *>(host.v));
    });
    t.test("release_writer_drops_cached_source_before_phase_reclaim", [&](testing & t) {
        writer_fixture f(backend.get(), cuda);
        auto e = std::make_unique<encoder>(f, 32, GGML_TYPE_Q8_0);
        release_probe probe(e->buffer.get());
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e->source));
        e.reset();
        t.assert_equal(0, probe.frees);
        f.resident->release_write_workspace();
        t.assert_equal(1, probe.frees);
        t.assert_true(ggml_backend_memory_arena_lease_count(f.arena.get()) != 0);
        t.assert_true(f.resident->synchronize(1024));
    });
    return t.summary();
}

#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-kv-stream-logical-cache.h"
#include "../src/llama-kv-stream-layer-lease.h"
#include "../src/llama-kv-stream-mtp-proxy.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../src/llama-context-memory.h"
#include <random>

// Quantization error is not adapter error: compare the exact bytes from an ordinary CUDA producer graph.
static std::vector<uint8_t> reference_bytes(ggml_backend_t backend,const std::vector<float> & data,ggml_type type,float scale) {
    const size_t rows = data.size()/512;
    ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
    auto * source = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,512,rows);
    auto * indices = ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I64,rows);
    auto * destination = ggml_new_tensor_2d(ctx.get(),type,512,rows);
    auto * output = ggml_set_rows(ctx.get(),destination,ggml_scale(ctx.get(),source,scale),indices);
    auto * graph = ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(graph,output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend)); GGML_ASSERT(buffer);
    std::vector<int64_t> ids(rows); for (size_t i = 0; i < rows; ++i) ids[i] = int64_t(i);
    ggml_backend_tensor_set(source,data.data(),0,data.size()*sizeof(float));
    ggml_backend_tensor_set(indices,ids.data(),0,ids.size()*sizeof(int64_t));
    llama_memory_cuda_executor executor(backend); GGML_ASSERT(executor.bind(graph,{},1));
    GGML_ASSERT(executor.compute_async({},1) == GGML_STATUS_SUCCESS && executor.drain());
    std::vector<uint8_t> result(ggml_nbytes(output)); ggml_backend_tensor_get(output,result.data(),0,result.size()); return result;
}

struct parent_view_fault {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_t,size_t,size_t);
    inline static parent_view_fault * active = nullptr;
    ggml_backend_buffer_t parent;
    factory original;
    size_t failures;
    size_t calls = 0;
    std::function<void()> on_create;

    parent_view_fault(ggml_backend_buffer_t parent,size_t failures) :
        parent(parent),original(parent ? parent->view_buffer : nullptr),failures(failures) {
        GGML_ASSERT(parent && original && !active);
        active = this;
        parent->view_buffer = create;
    }
    ~parent_view_fault() {
        parent->view_buffer = original;
        active = nullptr;
    }
    static ggml_backend_buffer_t create(
            ggml_backend_buffer_t parent,size_t offset,size_t size) {
        GGML_ASSERT(active && parent == active->parent);
        ++active->calls;
        if (active->on_create) active->on_create();
        if (active->failures) {
            --active->failures;
            return nullptr;
        }
        return active->original(parent,offset,size);
    }
};
struct backend_drain_fault {
    inline static backend_drain_fault * active = nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::synchronize) original;
    size_t failures, calls = 0;
    backend_drain_fault(ggml_backend_t backend,size_t failures) :
        backend(backend),original(backend->iface.synchronize),failures(failures) {
        GGML_ASSERT(!active); active = this;
        backend->iface.synchronize = [](ggml_backend_t backend) {
            ++active->calls;
            if (active->failures) {
                --active->failures;
                throw std::runtime_error("injected backend drain failure");
            }
            if (active->original) active->original(backend);
        };
    }
    ~backend_drain_fault() { backend->iface.synchronize = original; active = nullptr; }
};
struct index_read_fault {
    ggml_backend_buffer_t buffer;
    decltype(ggml_backend_buffer_i::get_tensor) original;
    inline static index_read_fault * active = nullptr;
    explicit index_read_fault(ggml_backend_buffer_t buffer) :
        buffer(buffer), original(buffer->iface.get_tensor) {
        GGML_ASSERT(!active);
        active = this;
        buffer->iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor *,
                void *, size_t, size_t) {
            throw std::runtime_error("injected index read failure");
        };
    }
    ~index_read_fault() {
        buffer->iface.get_tensor = original;
        active = nullptr;
    }
};
struct proxy_d2d_fault {
    ggml_backend_t backend;
    decltype(ggml_backend_i::cpy_tensor_async) original;
    inline static proxy_d2d_fault * active = nullptr;
    size_t calls = 0;
    explicit proxy_d2d_fault(ggml_backend_t backend) :
        backend(backend), original(backend->iface.cpy_tensor_async) {
        GGML_ASSERT(!active);
        active = this;
        backend->iface.cpy_tensor_async = [](ggml_backend_t, ggml_backend_t,
                const ggml_tensor *, ggml_tensor *) {
            ++active->calls;
            throw std::runtime_error("injected MTP D2D failure");
            return false;
        };
    }
    ~proxy_d2d_fault() {
        backend->iface.cpy_tensor_async = original;
        active = nullptr;
    }
};

// Observe synchronous resident refresh without changing its transfer behavior.
struct resident_upload_probe {
    using setter = void (*)(ggml_backend_buffer_t,ggml_tensor *,const void *,size_t,size_t);
    inline static resident_upload_probe * active = nullptr;
    ggml_backend_buffer_t buffer;
    setter original;
    size_t bytes = 0, calls = 0;
    resident_upload_probe(ggml_backend_buffer_t buffer) : buffer(buffer),original(buffer->iface.set_tensor) {
        GGML_ASSERT(!active && original);
        active = this;
        buffer->iface.set_tensor = [](ggml_backend_buffer_t buffer,ggml_tensor * tensor,
                const void * data,size_t offset,size_t bytes) {
            GGML_ASSERT(active && buffer == active->buffer);
            active->bytes += bytes;
            ++active->calls;
            active->original(buffer,tensor,data,offset,bytes);
        };
    }
    ~resident_upload_probe() { buffer->iface.set_tensor = original; active = nullptr; }
};

// Small real shared parent; the target and each borrower have independent schedulers.
struct serial_workspace_fixture {
    fixture host;
    ggml_backend_ptr cpu{ggml_backend_cpu_init()};
    ggml_backend_buffer_type_t type;
    size_t alignment;
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> types;
    ggml_backend_sched_ptr sched;
    llama_compute_workspace_plan plan;
    std::unique_ptr<llama_kv_stream_model> model;
    std::unique_ptr<llama_context_memory> owner;

    explicit serial_workspace_fixture(ggml_backend_t backend,const std::function<bool()> & capture = {}) :
        host(backend,true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1),
        type(llama_kv_stream_device_buffer_type(ggml_backend_get_device(backend))),
        alignment(ggml_backend_buft_get_alignment(type)),backends{backend,cpu.get()},
        types{type,ggml_backend_cpu_buffer_type()},
        sched(ggml_backend_sched_new(backends.data(),types.data(),2,256,false,true)) {
        auto bootstrap = host.policy;
        bootstrap.pool_bytes = 0;
        size_t minimum = 0;
        GGML_ASSERT(llama_kv_stream_policy_minimum_pool_bytes(bootstrap,minimum).status == llama_kv_stream_policy_status::success);
        llama_kv_stream_model_config config{backend,host.host->config(),minimum,256,4};
        config.shared_device_memory_bytes = 8*1048576;
        model = llama_kv_stream_model::create(config);
        plan.groups = {{type,2*1048576,alignment,0}};
        plan.phase_sizes = {{2*1048576},{1048576}};
        GGML_ASSERT(model && sched);
        owner = llama_context_memory::create(sched.get(),backends,plan,model.get(),nullptr,false,capture);
        GGML_ASSERT(owner);
    }
};

struct serial_workspace_child {
    ggml_backend_ptr backend, cpu{ggml_backend_cpu_init()};
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_ptr sched;
    std::unique_ptr<llama_context_memory> owner;

    explicit serial_workspace_child(serial_workspace_fixture & parent) :
        backend(ggml_backend_dev_init(ggml_backend_get_device(parent.backends[0]),nullptr)),
        backends{backend.get(),cpu.get()},
        sched(ggml_backend_sched_new(backends.data(),parent.types.data(),2,256,false,true)) {}

    bool attach(serial_workspace_fixture & parent,size_t prefill,size_t decode) {
        auto plan = parent.plan;
        plan.groups[0].size = std::max(prefill,decode);
        plan.phase_sizes = {{prefill},{decode}};
        owner = llama_context_memory::create(sched.get(),backends,plan,nullptr,parent.owner.get());
        return owner != nullptr;
    }
};

// Simulate a missing native family without changing private streamed-kernel capabilities.
struct native_attention_probe {
    inline static native_attention_probe * active = nullptr;
    ggml_backend_dev_t dev;
    decltype(ggml_backend_device_i::supports_op) original;
    uint32_t deny;
    std::vector<uint32_t> queries;
    std::vector<int64_t> padded_tokens;
    native_attention_probe(ggml_backend_dev_t dev,uint32_t deny) : dev(dev),original(dev->iface.supports_op),deny(deny) {
        GGML_ASSERT(!active); active = this;
        dev->iface.supports_op = [](ggml_backend_dev_t dev,const ggml_tensor * op) {
            if (op->op == GGML_OP_FLASH_ATTN_EXT) {
                const auto rows = uint32_t(op->src[0]->ne[1]);
                active->queries.push_back(rows);
                active->padded_tokens.push_back(op->src[1]->ne[1]);
                if (rows == active->deny) return false;
            }
            return active->original(dev,op);
        };
    }
    ~native_attention_probe() { dev->iface.supports_op = original; active = nullptr; }
};

struct kv_allocation_probe {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t,size_t);
    inline static kv_allocation_probe * active = nullptr;
    ggml_backend_buffer_type_t device, host;
    factory device_alloc, host_alloc;
    size_t calls = 0;
    ggml_backend_buffer_type_t fail = nullptr;
    explicit kv_allocation_probe(ggml_backend_dev_t dev) : device(llama_kv_stream_device_buffer_type(dev)),
        host(llama_kv_stream_host_buffer_type(dev)),device_alloc(device->iface.alloc_buffer),host_alloc(host->iface.alloc_buffer) {
        GGML_ASSERT(device != host && !active); active = this;
        device->iface.alloc_buffer = host->iface.alloc_buffer = allocate;
    }
    static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type,size_t bytes) {
        GGML_ASSERT(active && (type == active->device || type == active->host));
        ++active->calls;
        if (type == active->fail) return nullptr;
        return (type == active->device ? active->device_alloc : active->host_alloc)(type,bytes);
    }
    ~kv_allocation_probe() { device->iface.alloc_buffer = device_alloc; host->iface.alloc_buffer = host_alloc; active = nullptr; }
};

int main(int argc,char ** argv) {
    const bool native_only = argc > 1 && std::strcmp(argv[1], "--cuda-native-admission") == 0;
    const bool native_reduced = argc > 1 && std::strcmp(argv[1], "--cuda-native-reduced") == 0;
    const bool auxiliary_only = argc > 1 && std::strcmp(argv[1], "--cuda-auxiliary-cache") == 0;
    const bool cancel_only = argc > 1 && std::strcmp(argv[1], "--cuda-mtp-cancel") == 0;
    const bool lease_only = argc > 1 && std::strcmp(argv[1], "--cuda-mtp-lease") == 0;
    const bool admission_only = argc > 1 && std::strcmp(argv[1], "--cuda-mtp-admission") == 0;
    const bool wide_ubatch_only = argc > 1 && std::strcmp(argv[1], "--cuda-mtp-wide-ubatch") == 0;
    const bool suspend_only = argc > 1 && std::strcmp(argv[1], "--cuda-suspend") == 0;
    const bool resume_only = argc > 1 && std::strcmp(argv[1], "--cuda-resume") == 0;
    const bool vision_suspend_only = argc > 1 && std::strcmp(argv[1], "--cuda-vision-suspend") == 0;
    const bool phase_loan_only = argc > 1 && std::strcmp(argv[1], "--cuda-phase-loan") == 0;
    const bool serial_scratch_only = argc > 1 && std::strcmp(argv[1], "--cuda-serial-scratch") == 0;
    testing t;
    if (native_only) t.set_filter("native_attention_admission_.*");
    if (native_reduced) t.set_filter("native_attention_reduced_.*");
    if (serial_scratch_only) t.set_filter("serial_draft_growth_.*");
    if (phase_loan_only) t.set_filter("suspended_parent_tracks_phase_loans_until_the_last_lease_returns");
    if (vision_suspend_only) t.set_filter("suspended_kv_parent_lends_full_capacity_and_blocks_early_restore");
    if (suspend_only) t.set_filter("shared_kv_suspension_releases_device_grants_without_changing_host");
    if (resume_only) t.set_filter("shared_kv_resume_acquires_fresh_phase_grants_and_keeps_host_identity");
    if (admission_only) t.set_filter("mtp_prefill_admission_demotes_resident_pages_before_population");
    if (wide_ubatch_only) t.set_filter("mtp_prefill_uses_configured_microbatch_rows");
    if (auxiliary_only) t.set_filter("auxiliary_mtp_cache_shares_physical_policy_without_merging_identity");
    if (lease_only) t.set_filter("populated_mtp_lease_reuses_one_upload_for_tg1_to_tg4");
    if (cancel_only) t.set_filter("mtp_proxy_cancellation_drains_pending_writes");
    if (argc < 2 || (!native_reduced && !native_only && !serial_scratch_only && !phase_loan_only && !vision_suspend_only && !resume_only && !suspend_only && !auxiliary_only && !lease_only && !cancel_only && !admission_only && !wide_ubatch_only && std::strcmp(argv[1],"--cuda"))) {
        t.assert_true(!llama_kv_stream_model::create({})); return t.summary();
    }
    ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr)), cpu(ggml_backend_cpu_init());
    if (native_reduced) t.test("native_attention_reduced_build_rejects_mixed_pair_and_keeps_equal_pair", [&](testing & t) {
        fixture mixed(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        {
            kv_allocation_probe allocations(dev);
            uint32_t unavailable = 99;
            auto model = llama_kv_stream_model::create({backend.get(),mixed.host->config(),mixed.policy.pool_bytes,256,4},&unavailable);
            t.assert_true(!model);
            t.assert_equal(uint32_t(1),unavailable);
            t.assert_equal(size_t(0),allocations.calls);
        }
        fixture equal(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,513,false,1);
        uint32_t unavailable = 99;
        auto model = llama_kv_stream_model::create({backend.get(),equal.host->config(),equal.policy.pool_bytes,256,4},&unavailable);
        t.assert_true(bool(model));
        t.assert_equal(uint32_t(0),unavailable);
    });
    t.test("native_attention_admission_rejects_missing_width_before_allocating", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        for (uint32_t rows : {1u,2u,3u,4u,8u,16u,64u,128u,256u}) {
            native_attention_probe native(dev,rows);
            kv_allocation_probe allocations(dev);
            uint32_t unavailable = 99;
            auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),f.policy.pool_bytes,256,4},&unavailable);
            t.assert_true(!model);
            t.assert_equal(rows,unavailable);
            t.assert_equal(size_t(0),allocations.calls);
        }
    });
    t.test("native_attention_admission_probes_only_reachable_query_widths", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        for (uint32_t maximum : {1u,2u,3u,4u,256u}) {
            native_attention_probe native(dev,0);
            uint32_t unavailable = 99;
            auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),f.policy.pool_bytes,maximum,4},&unavailable);
            if (!t.assert_true(bool(model))) return;
            t.assert_equal(uint32_t(0),unavailable);
            t.assert_equal(size_t(maximum),native.queries.size());
            for (uint32_t expected = 1; expected <= maximum; ++expected)
                t.assert_equal(expected,native.queries[expected-1]);
            t.assert_true(std::all_of(native.queries.begin(),native.queries.end(),[&](uint32_t rows) { return rows <= maximum; }));
            t.assert_true(std::all_of(native.padded_tokens.begin(),native.padded_tokens.end(),[](int64_t tokens) { return tokens == 768; }));
        }
    });
    t.test("native_attention_admission_does_not_label_allocation_failure_as_missing_code", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        for (bool host : {true,false}) {
            kv_allocation_probe allocations(dev);
            allocations.fail = host ? allocations.host : allocations.device;
            uint32_t unavailable = 99;
            auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),f.policy.pool_bytes,256,4},&unavailable);
            t.assert_true(!model);
            t.assert_equal(uint32_t(0),unavailable);
            t.assert_true(allocations.calls > 0);
        }
    });
    t.test("serial_draft_growth_covers_independent_phase_maxima", [&](testing & t) {
        for (const auto & sizes : {std::pair<size_t,size_t>{3,1},{2,3},{3,4},{4,2},{2,1},{3,3}}) {
            serial_workspace_fixture f(backend.get());
            serial_workspace_child child(f);
            if (!t.assert_true(child.attach(f,sizes.first*1048576,sizes.second*1048576))) return;
            for (auto phase : {llama_memory_text_phase::prefill,llama_memory_text_phase::decode,llama_memory_text_phase::prefill}) {
                t.assert_true(f.owner->prepare_serial_target());
                const auto signalled = f.owner->signal_text_phase({phase,phase == llama_memory_text_phase::decode ? 1u : 256u,true,true,false});
                if (!t.assert_true(signalled.status == llama_memory_text_phase_status::changed)) return;
                llama_context_memory_diagnostics d;
                if (!t.assert_true(f.owner->diagnostics(d))) return;
                const size_t wanted = std::max(phase == llama_memory_text_phase::prefill ? size_t(2) : size_t(1),
                    phase == llama_memory_text_phase::prefill ? sizes.first : sizes.second)*1048576;
                t.assert_equal(wanted,d.workspace_bytes);
                t.assert_equal(d.parent_bytes,d.workspace_bytes+d.kv_pool_bytes+d.kv_writer_bytes+d.kv_attention_bytes+d.unused_bytes);
                t.assert_true(child.owner->prepare_serial_draft(phase));
                const auto end = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(f.owner->shared_parent()))+wanted;
                t.assert_true(end <= reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(f.model->mtp_attention_workspace())));
                t.assert_true(end <= reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(f.model->mtp_writer_workspace())));
                t.assert_true(end <= reinterpret_cast<uintptr_t>(f.model->binding_view().base));
            }
        }
    });
    t.test("serial_draft_growth_rejects_live_borrowers_and_retries_after_detach", [&](testing & t) {
        serial_workspace_fixture f(backend.get());
        serial_workspace_child first(f), second(f);
        if (!t.assert_true(first.attach(f,2*1048576,1048576))) return;
        const auto generation = f.owner->shared_arena_generation();
        const auto pool = f.model->pool_grant_bytes();
        t.assert_true(first.owner->prepare_serial_draft(llama_memory_text_phase::prefill));
        ggml_context_ptr context(ggml_init({16384,nullptr,true}));
        auto * input = ggml_new_tensor_1d(context.get(),GGML_TYPE_F32,16);
        auto * output = ggml_scale(context.get(),input,3.0f);
        ggml_set_input(input); ggml_set_output(input); ggml_set_output(output);
        auto * graph = ggml_new_graph_custom(context.get(),32,false);
        ggml_build_forward_expand(graph,output);
        ggml_backend_sched_set_tensor_backend(first.sched.get(),input,first.backend.get());
        ggml_backend_sched_set_tensor_backend(first.sched.get(),output,first.backend.get());
        if (!t.assert_true(ggml_backend_sched_alloc_graph(first.sched.get(),graph))) return;
        float data[16], result[16];
        std::fill(data,data+16,2.0f);
        ggml_backend_tensor_set(input,data,0,sizeof(data));
        if (!t.assert_true(first.owner->compute_async(graph) == GGML_STATUS_SUCCESS)) return;
        ggml_backend_tensor_get_async(first.backend.get(),output,result,0,sizeof(result));
        t.assert_true(!second.attach(f,3*1048576,2*1048576));
        t.assert_equal(generation,f.owner->shared_arena_generation());
        t.assert_equal(pool,f.model->pool_grant_bytes());
        t.assert_true(first.owner->serial_ready());
        first.owner->synchronize();
        t.assert_true(std::all_of(result,result+16,[](float value) { return value == 6.0f; }));
        first.owner.reset();
        t.assert_true(f.owner->prepare_serial_target());
        t.assert_true(second.attach(f,3*1048576,2*1048576));
        t.assert_true(second.owner->prepare_serial_draft(llama_memory_text_phase::prefill));
        second.owner.reset();
        t.assert_true(first.attach(f,3*1048576,2*1048576));
        t.assert_true(first.owner->prepare_serial_draft(llama_memory_text_phase::decode));
    });
    t.test("serial_draft_growth_capture_failure_closes_admission", [&](testing & t) {
        for (int failure : {0,1,2}) for (bool recovery : {false,true}) {
            bool fail = false;
            serial_workspace_fixture f(backend.get(),[&] {
                if (!fail) return true;
                if (failure == 2) return false;
                if (failure == 1) throw std::runtime_error("capture publication failure");
                throw std::bad_alloc();
            });
            const auto generation = f.owner->shared_arena_generation();
            serial_workspace_child child(f);
            fail = true;
            if (recovery) {
                parent_view_fault fault(f.owner->shared_parent(),0);
                fault.on_create = [&] { if (fault.calls == 2) fault.failures = 1; };
                t.assert_true(!child.attach(f,3*1048576,1048576));
                t.assert_true(fault.calls >= 2);
            } else t.assert_true(!child.attach(f,3*1048576,1048576));
            t.assert_true(recovery ? f.owner->shared_arena_generation() >= generation : f.owner->shared_arena_generation() > generation);
            t.assert_true(!f.owner->valid());
            t.assert_true(!f.owner->prepare_serial_target());
            t.assert_true(!f.owner->serial_ready());
            fail = false;
        }
    });
    t.test("serial_draft_growth_seeded_lifecycle_preserves_bounds_and_host_state", [&](testing & t) {
        std::mt19937 random(0x6b765632);
        serial_workspace_fixture f(backend.get());
        ggml_backend_buffer_clear(f.model->buffer(),0x35);
        if (!t.assert_true(f.model->restore(257))) return;
        std::vector<uint8_t> host(f.model->host()->bytes());
        std::memcpy(host.data(),ggml_backend_buffer_get_base(f.model->host()->buffer()),host.size());
        for (size_t iteration = 0; iteration < 32; ++iteration) {
            serial_workspace_child child(f);
            const size_t prefill = (2+random()%3)*1048576;
            const size_t decode = (1+random()%4)*1048576;
            if (iteration%7 == 0) {
                const auto generation = f.owner->shared_arena_generation();
                t.assert_true(!child.attach(f,8*1048576,1048576));
                t.assert_equal(generation,f.owner->shared_arena_generation());
            }
            if (!t.assert_true(child.attach(f,prefill,decode))) return;
            for (size_t step = 0; step < 6; ++step) {
                const auto phase = random()%2 ? llama_memory_text_phase::prefill : llama_memory_text_phase::decode;
                if (!t.assert_true(f.owner->prepare_serial_target())) return;
                const auto result = f.owner->signal_text_phase({phase,phase == llama_memory_text_phase::decode ? 1u : 256u,true,true,false});
                if (!t.assert_true(result.status == llama_memory_text_phase_status::changed || result.status == llama_memory_text_phase_status::unchanged)) return;
                if (!t.assert_true(child.owner->prepare_serial_draft(phase))) return;
                llama_context_memory_diagnostics d;
                if (!t.assert_true(f.owner->diagnostics(d))) return;
                t.assert_equal(size_t(8*1048576),d.parent_bytes);
                t.assert_equal(d.parent_bytes,d.workspace_bytes+d.kv_pool_bytes+d.kv_writer_bytes+d.kv_attention_bytes+d.unused_bytes);
                t.assert_equal(size_t(257),f.model->tokens());
                t.assert_true(std::memcmp(host.data(),ggml_backend_buffer_get_base(f.model->host()->buffer()),host.size()) == 0);
            }
            child.owner.reset();
            if (!t.assert_true(f.owner->prepare_serial_target() && f.owner->suspend_kv())) return;
            std::vector<ggml_backend_memory_lease_t> loans;
            if (!t.assert_true(f.owner->lend_suspended({4096},loans))) return;
            auto * retained = ggml_backend_memory_lease_retain(loans[0]);
            ggml_backend_memory_lease_free(loans[0]);
            t.assert_true(!f.owner->resume_kv(llama_memory_text_phase::prefill));
            auto blocked = llama_context_memory::borrow_workspace(child.sched.get(),child.backends,
                {{f.type,3*1048576,f.alignment,0}},*f.owner);
            t.assert_true(!blocked);
            ggml_backend_memory_lease_free(retained);
            if (!t.assert_true(f.owner->resume_kv(llama_memory_text_phase::prefill))) return;
            t.assert_true(std::memcmp(host.data(),ggml_backend_buffer_get_base(f.model->host()->buffer()),host.size()) == 0);
        }
    });
    t.test("serial_draft_growth_protects_live_kv_scratch", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        llama_kv_stream_policy_config bootstrap = f.policy;
        bootstrap.pool_bytes = 0;
        size_t minimum = 0;
        if (!t.assert_true(llama_kv_stream_policy_minimum_pool_bytes(bootstrap,minimum).status == llama_kv_stream_policy_status::success)) return;
        llama_kv_stream_model_config config{backend.get(),f.host->config(),minimum,256,4};
        config.shared_device_memory_bytes = 8*1048576;
        auto model = llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(model))) return;
        auto * type = llama_kv_stream_device_buffer_type(dev);
        const size_t alignment = ggml_backend_buft_get_alignment(type);
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        ggml_backend_buffer_type_t types[] = {type,ggml_backend_cpu_buffer_type()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types,2,256,false,true));
        llama_compute_workspace_plan plan;
        plan.groups = {{type,2*1048576,alignment,0}};
        plan.phase_sizes = {{2*1048576},{1048576}};
        auto owner = llama_context_memory::create(sched.get(),backends,plan,model.get());
        if (!t.assert_true(bool(owner))) return;
        auto * parent = owner->shared_parent();
        const auto initial_pool = model->pool_grant_bytes();
        ggml_backend_ptr draft_backend(ggml_backend_dev_init(dev,nullptr)), draft_cpu(ggml_backend_cpu_init());
        std::vector<ggml_backend_t> draft_backends{draft_backend.get(),draft_cpu.get()};
        ggml_backend_sched_ptr draft_sched(ggml_backend_sched_new(draft_backends.data(),types,2,256,false,true));
        auto draft_plan = plan;
        draft_plan.groups[0].size = 3*1048576;
        draft_plan.phase_sizes = {{3*1048576},{1048576}};
        const auto generation = owner->shared_arena_generation();
        const auto graph_revision = owner->graph_binding_revision();
        auto invalid = draft_plan;
        invalid.groups[0].size = invalid.phase_sizes[0][0] = config.shared_device_memory_bytes;
        t.assert_true(!llama_context_memory::create(draft_sched.get(),draft_backends,invalid,nullptr,owner.get()));
        t.assert_equal(generation,owner->shared_arena_generation());
        t.assert_equal(initial_pool,model->pool_grant_bytes());
        {
            parent_view_fault fault(parent,1);
            t.assert_true(!llama_context_memory::create(draft_sched.get(),draft_backends,draft_plan,nullptr,owner.get()));
            t.assert_true(fault.calls > 0);
        }
        t.assert_true(owner->valid());
        t.assert_equal(initial_pool,model->pool_grant_bytes());
        t.assert_equal(graph_revision,owner->graph_binding_revision());
        {
            parent_view_fault fault(parent,0);
            fault.on_create = [&] { if (fault.calls == 2) fault.failures = 1; };
            t.assert_true(!llama_context_memory::create(draft_sched.get(),draft_backends,draft_plan,nullptr,owner.get()));
            t.assert_true(fault.calls >= 2);
        }
        t.assert_true(owner->valid());
        t.assert_equal(initial_pool,model->pool_grant_bytes());
        t.assert_true(owner->graph_binding_revision() > graph_revision);
        auto draft = llama_context_memory::create(draft_sched.get(),draft_backends,draft_plan,nullptr,owner.get());
        if (!t.assert_true(bool(draft))) return;
        t.assert_true(draft->borrows_serial_parent());
        t.assert_true(owner->shared_parent() == parent);
        t.assert_equal(config.shared_device_memory_bytes,owner->shared_parent_capacity());
        t.assert_equal(initial_pool-1048576,model->pool_grant_bytes());
        if (!t.assert_true(draft->prepare_serial_draft(llama_memory_text_phase::prefill))) return;
        const auto scratch_end = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(draft->workspace_leases()[0])))+3*1048576;
        for (auto * buffer : {model->mtp_attention_workspace(),model->mtp_writer_workspace()}) {
            t.assert_true(buffer && scratch_end <= reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer)));
        }
        t.assert_true(scratch_end <= reinterpret_cast<uintptr_t>(model->binding_view().base));
        t.assert_true(owner->prepare_serial_target());
        t.assert_true(owner->signal_text_phase({llama_memory_text_phase::prefill,513,true,true,false}).status == llama_memory_text_phase_status::changed);
        t.assert_true(owner->signal_text_phase({llama_memory_text_phase::decode,1,true,true,false}).status == llama_memory_text_phase_status::changed);
        llama_context_memory_diagnostics diagnostics;
        t.assert_true(owner->diagnostics(diagnostics));
        t.assert_equal(size_t(1048576),diagnostics.workspace_bytes);
        t.assert_true(draft->prepare_serial_draft(llama_memory_text_phase::decode));
        t.assert_true(owner->prepare_serial_target());
        t.assert_true(owner->signal_text_phase({llama_memory_text_phase::prefill,128,true,true,false}).status == llama_memory_text_phase_status::changed);
        t.assert_equal(initial_pool-1048576,model->pool_grant_bytes());
    });
    t.test("serial_draft_growth_admits_exact_budget_and_rejects_one_byte_short", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        auto bootstrap = f.policy;
        bootstrap.pool_bytes = 0;
        size_t minimum = 0;
        if (!t.assert_true(llama_kv_stream_policy_minimum_pool_bytes(bootstrap,minimum).status == llama_kv_stream_policy_status::success)) return;
        auto * type = llama_kv_stream_device_buffer_type(dev);
        const size_t alignment = ggml_backend_buft_get_alignment(type);
        auto align = [&](size_t bytes) { return (bytes+alignment-1)/alignment*alignment; };
        llama_kv_stream_memory_requirements requirements;
        auto probe = llama_kv_stream_model::create({backend.get(),f.host->config(),minimum,256,4});
        if (!t.assert_true(probe && probe->memory_requirements(requirements))) return;
        probe.reset();
        for (const auto & sizes : {std::pair<size_t,size_t>{3,1},{2,3},{3,4},{4,2},{3,3}}) {
            const size_t exact = std::max(sizes.first,sizes.second)*1048576+
                align(std::max(requirements.attention_prefill_bytes,requirements.attention_decode_bytes))+
                align(requirements.writer_bytes)+minimum;
            for (size_t budget : {exact-1,exact}) {
                llama_kv_stream_model_config config{backend.get(),f.host->config(),minimum,256,4};
                config.shared_device_memory_bytes = budget;
                auto model = llama_kv_stream_model::create(config);
                if (!t.assert_true(bool(model))) return;
                std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
                ggml_backend_buffer_type_t types[] = {type,ggml_backend_cpu_buffer_type()};
                ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types,2,256,false,true));
                llama_compute_workspace_plan plan;
                plan.groups = {{type,2*1048576,alignment,0}};
                plan.phase_sizes = {{2*1048576},{1048576}};
                auto owner = llama_context_memory::create(sched.get(),backends,plan,model.get());
                if (!t.assert_true(bool(owner))) return;
                const auto before = model->pool_grant_bytes();
                ggml_backend_ptr draft_backend(ggml_backend_dev_init(dev,nullptr)), draft_cpu(ggml_backend_cpu_init());
                std::vector<ggml_backend_t> draft_backends{draft_backend.get(),draft_cpu.get()};
                ggml_backend_sched_ptr draft_sched(ggml_backend_sched_new(draft_backends.data(),types,2,256,false,true));
                plan.groups[0].size = std::max(sizes.first,sizes.second)*1048576;
                plan.phase_sizes = {{sizes.first*1048576},{sizes.second*1048576}};
                auto draft = llama_context_memory::create(draft_sched.get(),draft_backends,plan,nullptr,owner.get());
                t.assert_true(bool(draft) == (budget == exact));
                t.assert_equal(budget,owner->shared_parent_capacity());
                if (!draft) t.assert_equal(before,model->pool_grant_bytes());
            }
        }
    });
    t.test("suspended_parent_tracks_phase_loans_until_the_last_lease_returns", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,4,4});
        auto * type = llama_kv_stream_device_buffer_type(dev);
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        ggml_backend_buffer_type_t types[] = {type,ggml_backend_cpu_buffer_type()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types,2,256,false,true));
        llama_compute_workspace_plan plan;
        plan.groups = {{type,1048576,ggml_backend_buft_get_alignment(type),0}};
        plan.phase_sizes = {{1048576},{32768}};
        auto owner = llama_context_memory::create(sched.get(),backends,plan,model.get());
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(!owner->has_speculative_consumer());
        ggml_backend_buffer_clear(model->buffer(),0);
        if (!t.assert_true(model->restore(257))) return;
        const auto host = model->host();
        std::vector<uint8_t> bytes(host->bytes());
        std::memcpy(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size());
        std::vector<ggml_backend_memory_lease_t> loan;
        t.assert_true(!owner->lend_suspended({65536,32768},loan));
        if (!t.assert_true(owner->suspend_kv())) return;
        t.assert_true(!owner->lend_suspended({owner->shared_parent_capacity(),1},loan) && loan.empty());
        t.assert_true(!owner->lend_suspended({0},loan) && loan.empty());
        if (!t.assert_true(owner->lend_suspended({65536,32768},loan))) return;
        t.assert_equal(size_t(2),loan.size());
        llama_context_memory_diagnostics diagnostics;
        t.assert_true(owner->diagnostics(diagnostics));
        t.assert_equal(size_t(98304),diagnostics.borrowed_phase_bytes);
        t.assert_equal(owner->shared_parent_capacity()-98304,diagnostics.unused_bytes);
        ggml_backend_memory_region a{},b{};
        t.assert_true(ggml_backend_memory_lease_get_region(loan[0],&a) && ggml_backend_memory_lease_get_region(loan[1],&b));
        t.assert_true(a.offset+a.size <= b.offset && b.offset+b.size <= owner->shared_parent_capacity());
        t.assert_true(!owner->resume_kv(llama_memory_text_phase::decode));
        auto * retained = ggml_backend_memory_lease_retain(loan[0]);
        for (auto * lease : loan) ggml_backend_memory_lease_free(lease);
        loan.clear();
        t.assert_true(!owner->resume_kv(llama_memory_text_phase::decode));
        std::vector<ggml_backend_memory_lease_t> another;
        t.assert_true(!owner->lend_suspended({1024},another));
        ggml_backend_memory_lease_free(retained);
        if (!t.assert_true(owner->resume_kv(llama_memory_text_phase::decode))) return;
        t.assert_equal(size_t(257),model->tokens());
        t.assert_true(std::memcmp(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size()) == 0);
        if (!t.assert_true(owner->suspend_kv())) return;
        {
            parent_view_fault fault(owner->shared_parent(),1);
            t.assert_true(!owner->lend_suspended({65536,32768},loan) && loan.empty());
        }
        t.assert_true(owner->resume_kv(llama_memory_text_phase::prefill));
        if (!t.assert_true(owner->suspend_kv() && owner->lend_suspended({65536,32768},loan))) return;
        owner.reset();
        t.assert_true(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(loan[0])) != nullptr);
        ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(loan[0]),0x5a);
        for (auto * lease : loan) ggml_backend_memory_lease_free(lease);
        loan.clear();
    });
    t.test("suspended_kv_parent_lends_full_capacity_and_blocks_early_restore", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,4,4});
        if (!t.assert_true(bool(model))) return;
        auto * type = llama_kv_stream_device_buffer_type(dev);
        const size_t alignment = ggml_backend_buft_get_alignment(type);
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        ggml_backend_buffer_type_t types[] = {type,ggml_backend_cpu_buffer_type()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types,2,256,false,true));
        llama_compute_workspace_plan plan;
        plan.groups = {{type,1048576,alignment,0}};
        plan.phase_sizes = {{1048576},{32768}};
        auto owner = llama_context_memory::create(sched.get(),backends,plan,model.get());
        if (!t.assert_true(bool(owner))) return;
        ggml_backend_buffer_clear(model->buffer(),0);
        if (!t.assert_true(model->restore(257))) return;
        const auto host = model->host();
        const auto identity = host->cache_id();
        std::vector<uint8_t> bytes(host->bytes());
        std::memcpy(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size());
        ggml_backend_ptr vision_backend(ggml_backend_dev_init(dev,nullptr));
        ggml_backend_ptr vision_cpu(ggml_backend_cpu_init());
        std::vector<ggml_backend_t> vision_backends{vision_backend.get(),vision_cpu.get()};
        ggml_backend_sched_ptr vision_sched(ggml_backend_sched_new(vision_backends.data(),types,2,256,false,true));
        std::vector<ggml_backend_memory_workspace_group> groups{{type,owner->shared_parent_capacity(),alignment,0}};
        // Cancellation before eviction changes neither admission nor host state.
        t.assert_true(!llama_context_memory::borrow_workspace(vision_sched.get(),vision_backends,groups,*owner));
        t.assert_true(owner->serial_ready() && model->complete());
        for (int round = 0; round < 3; ++round) {
            if (!t.assert_true(owner->suspend_kv())) return;
            auto oversized = groups; oversized[0].size += alignment;
            t.assert_true(!llama_context_memory::borrow_workspace(vision_sched.get(),vision_backends,oversized,*owner));
            {
                parent_view_fault fault(owner->shared_parent(),1);
                bool reentered = false;
                fault.on_create = [&] { reentered |= owner->resume_kv(llama_memory_text_phase::prefill); };
                t.assert_true(!llama_context_memory::borrow_workspace(vision_sched.get(),vision_backends,groups,*owner));
                t.assert_true(fault.calls > 0 && !reentered);
            }
            auto vision = llama_context_memory::borrow_workspace(vision_sched.get(),vision_backends,groups,*owner);
            if (!t.assert_true(bool(vision))) return;
            t.assert_true(!owner->resume_kv(llama_memory_text_phase::prefill));
            t.assert_true(vision->prepare_serial_consumer(llama_memory_text_phase::prefill));
            t.assert_true(vision->serial_ready() && !owner->serial_ready());
            t.assert_true(!vision->prepare_serial_consumer(llama_memory_text_phase::decode));
            t.assert_true(!owner->prepare_serial_target() && !owner->resume_kv(llama_memory_text_phase::decode));
            t.assert_true(!llama_context_memory::borrow_workspace(vision_sched.get(),vision_backends,groups,*owner));
            auto * lease = vision->workspace_leases().front();
            t.assert_equal(owner->shared_parent_capacity(),ggml_backend_buffer_get_size(ggml_backend_memory_lease_buffer(lease)));
            ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(lease),0xa5);
            t.assert_true(vision->retire_graph());
            t.assert_true(!owner->resume_kv(llama_memory_text_phase::prefill));
            // Returning a cancelled or completed vision stage uses the same drain/retire path.
            vision.reset();
            t.assert_true(owner->kv_device_suspended() && model->device_grant_bytes() == 0);
            {
                parent_view_fault fault(owner->shared_parent(),1);
                t.assert_true(!owner->resume_kv(llama_memory_text_phase::prefill));
                t.assert_true(fault.calls > 0);
            }
            t.assert_true(owner->kv_device_suspended() && !owner->serial_ready());
            t.assert_true(owner->resume_kv(llama_memory_text_phase::prefill));
            t.assert_true(model->complete() && owner->serial_ready());
            t.assert_equal(size_t(257),model->tokens());
            t.assert_equal(identity,host->cache_id());
            t.assert_true(std::memcmp(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size()) == 0);
        }
        // A one-shot drain failure can restore the old layout; failure during reverse drain is terminal.
        {
            backend_drain_fault fault(backend.get(),1);
            t.assert_true(!owner->suspend_kv());
            t.assert_true(fault.calls > 1);
        }
        t.assert_true(owner->valid() && model->complete() && owner->serial_ready());
        {
            backend_drain_fault fault(backend.get(),2);
            t.assert_true(!owner->suspend_kv());
            t.assert_true(fault.calls >= 2);
        }
        t.assert_true(!owner->valid() && !owner->serial_ready());
        t.assert_true(!owner->suspend_kv() && !owner->resume_kv(llama_memory_text_phase::prefill));
        t.assert_true(!llama_context_memory::borrow_workspace(vision_sched.get(),vision_backends,groups,*owner));
        t.assert_true(std::memcmp(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size()) == 0);
    });
    t.test("shared_kv_resume_acquires_fresh_phase_grants_and_keeps_host_identity", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        llama_kv_stream_model_config config{backend.get(),f.host->config(),page.bytes*4,4,4};
        config.auxiliary_cache_layers = 1;
        auto model = llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(model))) return;
        auto * type = llama_kv_stream_device_buffer_type(dev);
        auto * host_type = ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        std::vector<ggml_backend_buffer_type_t> types{type,host_type};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types.data(),2,256,false,true));
        llama_compute_workspace_plan plan;
        plan.groups = {{type,1048576,ggml_backend_buft_get_alignment(type),0},
            {host_type,8192,ggml_backend_buft_get_alignment(host_type),1}};
        plan.phase_sizes = {{1048576,8192},{32768,4096}};
        auto owner = llama_context_memory::create(sched.get(),backends,plan,model.get());
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(owner->has_speculative_consumer());
        ggml_backend_buffer_clear(model->buffer(),0);
        if (!t.assert_true(model->restore(257))) return;
        t.assert_true(!owner->resume_kv(llama_memory_text_phase::decode));
        ggml_backend_ptr child_backend(ggml_backend_dev_init(dev,nullptr)), child_cpu(ggml_backend_cpu_init());
        std::vector<ggml_backend_t> child_backends{child_backend.get(),child_cpu.get()};
        ggml_backend_sched_ptr child_sched(ggml_backend_sched_new(child_backends.data(),types.data(),2,256,false,true));
        auto child = llama_context_memory::borrow_workspace(child_sched.get(),child_backends,
            {{type,8192,ggml_backend_buft_get_alignment(type),0}},*owner);
        if (!t.assert_true(bool(child))) return;
        const auto host = model->host();
        const auto consumer = model->memory_consumer();
        const auto id = host->cache_id();
        std::vector<uint8_t> bytes(host->bytes());
        std::memcpy(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size());
        size_t previous_capacity = model->pool_grant_bytes();
        uintptr_t previous_base = uintptr_t(model->binding_view().base);
        uint64_t previous_revision = model->binding_view().revision;
        uint64_t previous_generation = owner->shared_arena_generation();
        for (auto phase : {llama_memory_text_phase::decode,llama_memory_text_phase::prefill,
                llama_memory_text_phase::decode,llama_memory_text_phase::decode}) {
            if (!t.assert_true(owner->suspend_kv())) return;
            t.assert_true(!owner->resume_kv(llama_memory_text_phase::unspecified));
            if (!t.assert_true(owner->resume_kv(phase))) return;
            const auto binding = model->binding_view();
            t.assert_true(model->complete() && !model->device_suspended() && owner->serial_ready());
            t.assert_true(child->workspace_leases().empty() && !child->serial_ready());
            t.assert_true(binding.revision > previous_revision && binding.cache_id == id);
            t.assert_true(owner->shared_arena_generation() > previous_generation);
            if (phase != llama_memory_text_phase::decode || previous_capacity == page.bytes*4) {
                t.assert_true(uintptr_t(binding.base) != previous_base);
                t.assert_true(binding.capacity != previous_capacity);
            }
            t.assert_true(model->host() == host && model->memory_consumer() == consumer);
            t.assert_equal(size_t(257),model->tokens());
            t.assert_true(std::memcmp(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size()) == 0);
            t.assert_true(!owner->resume_kv(phase));
            previous_base = uintptr_t(binding.base);
            previous_capacity = binding.capacity;
            previous_revision = binding.revision;
            previous_generation = owner->shared_arena_generation();
        }
        if (!t.assert_true(owner->suspend_kv())) return;
        bool reentered = false;
        t.assert_true(!owner->resume_kv(llama_memory_text_phase::decode,[&] {
            reentered |= owner->resume_kv(llama_memory_text_phase::decode);
            reentered |= owner->prepare_serial_target();
            return false;
        }));
        t.assert_true(!reentered && model->device_suspended() && !owner->serial_ready());
        t.assert_equal(size_t(0),model->device_grant_bytes());
        t.assert_true(owner->resume_kv(llama_memory_text_phase::decode));
        if (!t.assert_true(owner->suspend_kv())) return;
        const auto auxiliary = model->auxiliary_cache();
        if (!t.assert_true(auxiliary && auxiliary->begin(1))) return;
        t.assert_true(!owner->resume_kv(llama_memory_text_phase::decode) && model->device_suspended());
        if (!t.assert_true(auxiliary->cancel())) return;
        t.assert_true(owner->resume_kv(llama_memory_text_phase::decode));
        t.assert_true(child->prepare_serial_consumer(llama_memory_text_phase::prefill));
        t.assert_true(child->serial_ready() && !owner->serial_ready());
        t.assert_true(owner->prepare_serial_target());
        if (!t.assert_true(owner->suspend_kv())) return;
        t.assert_true(!owner->resume_kv(llama_memory_text_phase::decode,[]() -> bool {
            throw std::runtime_error("injected graph reconstruction failure");
        }));
        t.assert_true(model->device_suspended() && model->device_grant_bytes() == 0);
        t.assert_true(owner->resume_kv(llama_memory_text_phase::decode));
    });
    t.test("shared_kv_suspension_releases_device_grants_without_changing_host", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        llama_kv_stream_model_config config{backend.get(),f.host->config(),page.bytes*4,4,4};
        config.auxiliary_cache_layers = 1;
        auto model = llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(model))) return;
        auto * type = llama_kv_stream_device_buffer_type(dev);
        auto * host_type = ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        std::vector<ggml_backend_buffer_type_t> types{type,host_type};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types.data(),2,256,false,true));
        llama_compute_workspace_plan plan;
        plan.groups = {{type,65536,ggml_backend_buft_get_alignment(type),0},
            {host_type,8192,ggml_backend_buft_get_alignment(host_type),1}};
        plan.phase_sizes = {{65536,8192},{32768,4096}};
        auto owner = llama_context_memory::create(sched.get(),backends,plan,model.get());
        if (!t.assert_true(bool(owner))) return;
        if (!t.assert_true(model->begin(1,1,true))) return;
        const auto active_grants = model->device_grant_bytes();
        t.assert_true(!owner->suspend_kv() && !model->device_suspended());
        t.assert_equal(active_grants,model->device_grant_bytes());
        model->abort();
        ggml_backend_buffer_clear(model->buffer(),0);
        if (!t.assert_true(model->restore(257))) return;
        const auto auxiliary = model->auxiliary_cache();
        if (!t.assert_true(auxiliary && auxiliary->begin(1))) return;
        t.assert_true(!owner->suspend_kv() && !model->device_suspended());
        if (!t.assert_true(auxiliary->cancel())) return;
        auto checkpoint = auxiliary->checkpoint();
        checkpoint.tokens = 257;
        if (!t.assert_true(auxiliary->restore(checkpoint) && model->acquire_mtp_layer())) return;
        const auto auxiliary_identity = auxiliary->identity();
        std::vector<uint8_t> auxiliary_bytes(auxiliary->host()->bytes());
        std::memcpy(auxiliary_bytes.data(),ggml_backend_buffer_get_base(auxiliary->host()->buffer()),auxiliary_bytes.size());
        const auto cache = model->host();
        const auto cache_id = cache->cache_id();
        const auto consumer = model->memory_consumer();
        std::vector<uint8_t> expected(cache->bytes());
        std::memcpy(expected.data(),ggml_backend_buffer_get_base(cache->buffer()),expected.size());
        const auto revision = model->binding_view().revision;
        auto * parent = owner->shared_parent();
        ggml_context_ptr graph_ctx(ggml_init({16384,nullptr,true}));
        auto * marker = ggml_new_tensor_1d(graph_ctx.get(),GGML_TYPE_F32,4);
        const auto old = model->binding_view();
        if (!t.assert_true(ggml_backend_tensor_alloc(old.buffer,marker,old.base) == GGML_STATUS_SUCCESS)) return;
        const float source[4] = {1,2,3,4};
        float copied[4] = {};
        ggml_backend_tensor_set_async(backend.get(),marker,source,0,sizeof(source));
        ggml_backend_tensor_get_async(backend.get(),marker,copied,0,sizeof(copied));
        ggml_backend_ptr child_backend(ggml_backend_dev_init(dev,nullptr)), child_cpu(ggml_backend_cpu_init());
        std::vector<ggml_backend_t> child_backends{child_backend.get(),child_cpu.get()};
        ggml_backend_sched_ptr child_sched(ggml_backend_sched_new(child_backends.data(),types.data(),2,256,false,true));
        auto child = llama_context_memory::borrow_workspace(child_sched.get(),child_backends,
            {{type,8192,ggml_backend_buft_get_alignment(type),0}},*owner);
        if (!t.assert_true(bool(child))) return;
        struct completion_probe : llama_memory_executor_backend {
            llama_context_memory & owner;
            bool fail = true, reentered = false;
            explicit completion_probe(llama_context_memory & owner) : owner(owner) {}
            bool drain() override { reentered |= owner.suspend_kv(); return !fail; }
        } completion(*owner);
        const auto grants = model->device_grant_bytes();
        t.assert_true(!owner->suspend_kv(&completion));
        t.assert_true(!completion.reentered && !model->device_suspended());
        t.assert_equal(grants,model->device_grant_bytes());
        completion.fail = false;
        if (!t.assert_true(owner->suspend_kv(&completion))) return;
        t.assert_true(!completion.reentered);
        t.assert_true(std::memcmp(source,copied,sizeof(source)) == 0);
        t.assert_true(owner->kv_device_suspended() && model->device_suspended());
        t.assert_true(model->host() == cache && cache->cache_id() == cache_id);
        t.assert_true(model->memory_consumer() == consumer);
        t.assert_true(!model->has_mtp_layer() && model->auxiliary_cache() == auxiliary);
        t.assert_equal(auxiliary_identity.generation,auxiliary->identity().generation);
        t.assert_equal(size_t(257),auxiliary->tokens());
        t.assert_true(std::memcmp(auxiliary_bytes.data(),ggml_backend_buffer_get_base(auxiliary->host()->buffer()),auxiliary_bytes.size()) == 0);
        t.assert_equal(size_t(257),model->tokens());
        t.assert_equal(size_t(0),model->device_grant_bytes());
        t.assert_equal(size_t(0),model->pool_grant_bytes());
        t.assert_equal(size_t(0),model->writer_grant_bytes());
        t.assert_equal(size_t(0),model->attention_grant_bytes());
        t.assert_true(model->binding_view().buffer == nullptr);
        t.assert_true(owner->workspace_leases().empty());
        t.assert_true(child->workspace_leases().empty());
        t.assert_true(!child->serial_ready() && !child->prepare_serial_consumer(llama_memory_text_phase::prefill));
        t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(sched.get(),backend.get()));
        t.assert_true(owner->shared_parent() == parent);
        t.assert_true(std::memcmp(expected.data(),ggml_backend_buffer_get_base(cache->buffer()),expected.size()) == 0);
        llama_context_memory_diagnostics diagnostics;
        if (!t.assert_true(owner->diagnostics(diagnostics))) return;
        t.assert_true(diagnostics.kv_device_suspended && diagnostics.layout_revision > revision);
        t.assert_equal(diagnostics.parent_bytes,diagnostics.unused_bytes);
        t.assert_true(!model->begin(258,1,true) && !model->reset(false) && !model->truncate(256));
        t.assert_true(!owner->prepare_serial_target() && !owner->serial_ready());
        t.assert_true(owner->signal_text_phase({llama_memory_text_phase::decode,1,true,true,false}).status ==
            llama_memory_text_phase_status::transition_failed);
        t.assert_true(owner->suspend_kv());
        t.assert_true(model->set_workspaces({}));
        t.assert_true(!model->set_workspaces({nullptr}));
        t.assert_true(std::memcmp(expected.data(),ggml_backend_buffer_get_base(cache->buffer()),expected.size()) == 0);
    });
    t.test("fixed_compute_and_kv_grants_share_one_device_parent", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page;
        ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;

        auto * device_type = llama_kv_stream_device_buffer_type(dev);
        auto * cpu_type = ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        std::vector<ggml_backend_buffer_type_t> types{device_type,cpu_type};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(
            backends.data(),types.data(),types.size(),256,false,true));
        if (!t.assert_true(bool(sched))) return;

        const size_t device_alignment = ggml_backend_buft_get_alignment(device_type);
        const size_t cpu_alignment = ggml_backend_buft_get_alignment(cpu_type);
        llama_compute_workspace_plan plan;
        plan.groups = {
            {device_type,2*1048576,device_alignment,0},
            {cpu_type,8192,cpu_alignment,1},
        };
        plan.phase_sizes = {
            {2*1048576,8192},
            {1048576,4096},
        };

        llama_kv_stream_memory_requirements requirements;
        if (!t.assert_true(model->memory_requirements(requirements))) return;
        auto owner = llama_context_memory::create(sched.get(),backends,plan,model.get());
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(owner->shares_kv_memory());
        t.assert_true(model->uses_shared_memory());
        t.assert_true(!model->prepare_shared_memory());
        t.assert_true(owner->shared_parent() == model->shared_parent());
        t.assert_true(owner->shared_parent() != nullptr);
        t.assert_equal(owner->shared_parent_capacity(),
            ggml_backend_buffer_get_size(owner->shared_parent()));
        const size_t expected_grants = requirements.pool_bytes + requirements.writer_bytes +
            std::max(requirements.attention_prefill_bytes,requirements.attention_decode_bytes);
        t.assert_equal(expected_grants,model->device_grant_bytes());
        t.assert_equal(plan.groups[0].size,
            ggml_backend_sched_get_buffer_size(sched.get(),backend.get()));
        size_t expected_parent = plan.groups[0].size;
        for (size_t bytes : {requirements.pool_bytes,requirements.writer_bytes,
                std::max(requirements.attention_prefill_bytes,requirements.attention_decode_bytes)}) {
            expected_parent = (expected_parent+device_alignment-1)&~(device_alignment-1);
            expected_parent += bytes;
        }
        t.assert_equal(expected_parent,owner->shared_parent_capacity());
        auto * base = ggml_backend_buffer_get_base(owner->shared_parent());
        const auto initial_generation = owner->shared_arena_generation();
        const auto initial_pool = model->pool_grant_bytes();
        const auto initial_writer = model->writer_grant_bytes();
        const auto initial_attention = model->attention_grant_bytes();
        t.assert_equal(requirements.pool_bytes,initial_pool);
        t.assert_equal(requirements.writer_bytes,initial_writer);
        t.assert_equal(requirements.attention_prefill_bytes,initial_attention);
        t.assert_equal(uint64_t(0),owner->phase_transition_count());
        ggml_backend_buffer_clear(model->buffer(),0);
        t.assert_true(model->restore(513));
        t.assert_equal(size_t(513),model->tokens());

        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::prefill,513,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(0),owner->phase_transition_count());
        t.assert_equal(initial_generation,owner->shared_arena_generation());

        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::decode,1,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(1),owner->phase_transition_count());
        t.assert_true(owner->shared_arena_generation() > initial_generation);
        t.assert_equal(plan.phase_sizes[1][0],
            ggml_backend_sched_get_buffer_size(sched.get(),backend.get()));
        t.assert_true(model->pool_grant_bytes() > initial_pool);
        t.assert_equal(initial_writer,model->writer_grant_bytes());
        t.assert_equal(requirements.attention_decode_bytes,model->attention_grant_bytes());
        t.assert_equal(model->pool_grant_bytes()+model->writer_grant_bytes()+
            model->attention_grant_bytes(),model->device_grant_bytes());
        const auto decode_generation = owner->shared_arena_generation();
        const auto decode_pool = model->pool_grant_bytes();
        t.out << "shared pool bytes: prefill=" << initial_pool << " decode=" << decode_pool
              << " reclaimed=" << decode_pool-initial_pool << '\n';

        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::decode,1,true,true,false}).status ==
            llama_memory_text_phase_status::unchanged);
        t.assert_equal(uint64_t(1),owner->phase_transition_count());
        t.assert_equal(decode_generation,owner->shared_arena_generation());
        t.assert_equal(decode_pool,model->pool_grant_bytes());

        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::prefill,128,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(2),owner->phase_transition_count());
        t.assert_equal(plan.phase_sizes[0][0],
            ggml_backend_sched_get_buffer_size(sched.get(),backend.get()));
        t.assert_equal(initial_pool,model->pool_grant_bytes());
        t.assert_equal(initial_attention,model->attention_grant_bytes());
        t.assert_true(base == ggml_backend_buffer_get_base(owner->shared_parent()));

        auto * consumer = model->memory_consumer();
        t.assert_true(consumer != nullptr);
        t.assert_true(model->reset(false));
        t.assert_true(consumer == model->memory_consumer());
        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::decode,1,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(3),owner->phase_transition_count());
        t.assert_true(owner->signal_text_phase({
            llama_memory_text_phase::prefill,128,true,true,false}).status ==
            llama_memory_text_phase_status::changed);
        t.assert_equal(uint64_t(4),owner->phase_transition_count());
        t.assert_equal(size_t(0),model->tokens());
        t.assert_true(model->complete());

        owner.reset();
        t.assert_true(!model->uses_shared_memory());
        t.assert_true(model->shared_parent() == nullptr);
    });

    t.test("shared_budget_finds_exact_fit_and_validates_all_phases", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        llama_kv_stream_policy_config bootstrap = f.policy;
        bootstrap.pool_bytes = 0;
        size_t minimum_pool = 0;
        if (!t.assert_true(llama_kv_stream_policy_minimum_pool_bytes(
                bootstrap,minimum_pool).status ==
                llama_kv_stream_policy_status::success)) return;
        auto * device_type = llama_kv_stream_device_buffer_type(dev);
        auto * cpu_type = ggml_backend_cpu_buffer_type();
        const size_t alignment = ggml_backend_buft_get_alignment(device_type);
        llama_compute_workspace_plan plan;
        plan.groups = {
            {device_type,2*1048576,alignment,0},
            {cpu_type,8192,ggml_backend_buft_get_alignment(cpu_type),1},
        };
        plan.phase_sizes = {{2*1048576,8192},{1048576,4096}};

        const auto attempt = [&](size_t budget,bool exercise) {
            llama_kv_stream_model_config config{
                backend.get(),f.host->config(),minimum_pool,256,4};
            config.shared_device_memory_bytes = budget;
            auto model = llama_kv_stream_model::create(config);
            if (!model) return false;
            std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
            std::vector<ggml_backend_buffer_type_t> types{device_type,cpu_type};
            ggml_backend_sched_ptr sched(ggml_backend_sched_new(
                backends.data(),types.data(),types.size(),256,false,true));
            auto owner = llama_context_memory::create(
                sched.get(),backends,plan,model.get());
            if (!owner) return false;
            if (!exercise) return true;
            t.assert_equal(budget,owner->shared_parent_capacity());
            t.assert_equal(budget,
                ggml_backend_buffer_get_size(owner->shared_parent()));
            const auto prefill_pool = model->pool_grant_bytes();
            t.assert_true(prefill_pool >= minimum_pool);
            llama_context_memory_diagnostics prefill;
            t.assert_true(owner->diagnostics(prefill));
            t.assert_true(prefill.phase == llama_memory_text_phase::prefill);
            t.assert_equal(budget,prefill.parent_bytes);
            t.assert_equal(plan.phase_sizes[0][0],prefill.workspace_bytes);
            t.assert_equal(prefill_pool,prefill.kv_pool_bytes);
            t.assert_equal(budget,prefill.workspace_bytes+prefill.kv_pool_bytes+
                prefill.kv_writer_bytes+prefill.kv_attention_bytes+prefill.unused_bytes);
            t.assert_equal(size_t(0),prefill.reclaimed_workspace_bytes);
            t.assert_true(prefill.executable_storage_external);
            auto * base = ggml_backend_buffer_get_base(owner->shared_parent());
            t.assert_true(owner->signal_text_phase({
                llama_memory_text_phase::prefill,513,true,true,false}).status ==
                llama_memory_text_phase_status::changed);
            t.assert_true(owner->signal_text_phase({
                llama_memory_text_phase::decode,1,true,true,false}).status ==
                llama_memory_text_phase_status::changed);
            t.assert_true(model->pool_grant_bytes() > prefill_pool);
            llama_context_memory_diagnostics decode;
            t.assert_true(owner->diagnostics(decode));
            t.assert_true(decode.phase == llama_memory_text_phase::decode);
            t.assert_equal(plan.phase_sizes[1][0],decode.workspace_bytes);
            t.assert_equal(plan.groups[0].size-plan.phase_sizes[1][0],
                decode.reclaimed_workspace_bytes);
            t.assert_equal(model->pool_grant_bytes(),decode.kv_pool_bytes);
            t.assert_equal(uint64_t(1),decode.transition_count);
            t.assert_true(decode.last_transition_us > 0);
            t.assert_true(base ==
                ggml_backend_buffer_get_base(owner->shared_parent()));
            t.assert_true(owner->signal_text_phase({
                llama_memory_text_phase::prefill,7,true,true,false}).status ==
                llama_memory_text_phase_status::changed);
            t.assert_equal(prefill_pool,model->pool_grant_bytes());
            return true;

        };

        size_t low = plan.groups[0].size-alignment;
        size_t high = 4*1048576;
        if (!t.assert_true(attempt(high,false))) return;
        while (high-low > alignment) {
            size_t middle = low+(high-low)/2;
            middle -= middle%alignment;
            if (middle <= low) middle = low+alignment;
            if (attempt(middle,false)) high = middle;
            else low = middle;
        }
        t.assert_equal(alignment,high-low);
        t.assert_true(!attempt(low,false));
        t.assert_true(attempt(high,true));
        t.out << "minimum shared parent bytes=" << high
              << ", rejected previous granule=" << low << '\n';
    });

    t.test("interrupted_phase_transition_recovers_and_retries", [&](testing & t) {
        {
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
            ggml_kv_stream_layout page;
            ggml_kv_stream_layout_make(f.policy.shape,256,page);
            auto model = llama_kv_stream_model::create(
                {backend.get(),f.host->config(),page.bytes*4,256,4});
            if (!t.assert_true(bool(model))) return;

            auto * device_type = llama_kv_stream_device_buffer_type(dev);
            auto * cpu_type = ggml_backend_cpu_buffer_type();
            std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
            std::vector<ggml_backend_buffer_type_t> types{device_type,cpu_type};
            ggml_backend_sched_ptr sched(ggml_backend_sched_new(
                backends.data(),types.data(),types.size(),256,false,true));
            llama_compute_workspace_plan plan;
            plan.groups = {
                {device_type,2*1048576,ggml_backend_buft_get_alignment(device_type),0},
                {cpu_type,8192,ggml_backend_buft_get_alignment(cpu_type),1},
            };
            plan.phase_sizes = {{2*1048576,8192},{1048576,4096}};
            auto owner = llama_context_memory::create(
                sched.get(),backends,plan,model.get());
            if (!t.assert_true(bool(owner))) return;
            const auto initial_pool = model->pool_grant_bytes();
            auto * base = ggml_backend_buffer_get_base(owner->shared_parent());
            ggml_backend_buffer_clear(model->buffer(),0);
            if (!t.assert_true(model->restore(513))) return;
            t.assert_true(owner->signal_text_phase({
                llama_memory_text_phase::prefill,513,true,true,false}).status ==
                llama_memory_text_phase_status::changed);

            size_t calls = 0;
            {
                parent_view_fault fault(
                    owner->shared_parent(),1);
                const auto failed = owner->signal_text_phase({
                    llama_memory_text_phase::decode,1,true,true,false});
                t.assert_true(failed.status ==
                    llama_memory_text_phase_status::transition_failed);
                calls = fault.calls;
            }
            t.assert_true(calls > 0);
            t.assert_true(base ==
                ggml_backend_buffer_get_base(owner->shared_parent()));
            t.assert_equal(initial_pool,model->pool_grant_bytes());
            t.assert_equal(size_t(513),model->tokens());
            t.assert_true(owner->text_phase().phase ==
                llama_memory_text_phase::prefill);

            const auto retry = owner->signal_text_phase({
                llama_memory_text_phase::decode,1,true,true,false});
            t.assert_true(model->complete());
            t.assert_true(retry.status ==
                llama_memory_text_phase_status::changed);
            t.assert_true(model->pool_grant_bytes() > initial_pool);
            t.assert_true(owner->signal_text_phase({
                llama_memory_text_phase::prefill,7,true,true,false}).status ==
                llama_memory_text_phase_status::changed);
            t.assert_equal(initial_pool,model->pool_grant_bytes());
        }
    });

    t.test("shared_parent_mismatch_rejects_and_restores_private_model", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page;
        ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;
        const size_t private_bytes = model->granted_bytes();

        auto * ordinary = ggml_backend_get_default_buffer_type(backend.get());
        auto * cpu_type = ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
        std::vector<ggml_backend_buffer_type_t> types{ordinary,cpu_type};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(
            backends.data(),types.data(),types.size(),256,false,true));
        llama_compute_workspace_plan plan;
        plan.groups = {
            {ordinary,1048576,ggml_backend_buft_get_alignment(ordinary),0},
            {cpu_type,8192,ggml_backend_buft_get_alignment(cpu_type),1},
        };
        plan.phase_sizes = {
            {1048576,8192},
            {524288,4096},
        };
        t.assert_true(!llama_context_memory::create(sched.get(),backends,plan,model.get()));
        t.assert_true(!model->uses_shared_memory());
        t.assert_true(model->shared_parent() == nullptr);
        t.assert_true(model->complete());
        t.assert_equal(private_bytes,model->granted_bytes());
    });

    t.test("failed_scratch_replacement_keeps_host_state_and_can_retry", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model=llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;
        const auto initial=model->granted_bytes(); auto host=model->host();
        auto * type=llama_kv_stream_device_buffer_type(dev);
        const auto allocate=type->iface.alloc_buffer;
        type->iface.alloc_buffer=[](ggml_backend_buffer_type_t,size_t)->ggml_backend_buffer_t { return nullptr; };
        const bool began=model->begin(1,1,true);
        type->iface.alloc_buffer=allocate;
        t.assert_true(!began && model->complete() && model->tokens()==0 && model->host()==host);
        t.assert_true(model->granted_bytes()<initial);
        t.assert_true(model->begin(1,1,true)); model->abort();
        t.assert_true(model->reset(false)); t.assert_equal(initial,model->granted_bytes());
    });
    t.test("suffix_truncation_preserves_allocation_and_uploads_only_tail", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,1024,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model=llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*8,256,4});
        if (!t.assert_true(bool(model))) return;
        ggml_backend_buffer_clear(model->buffer(),91);
        t.assert_true(model->restore(252));
        if (!t.assert_true(model->begin(254,2,true))) return;
        model->abort();
        const auto view=model->binding_view();
        const auto granted=model->granted_bytes();
        auto * type=llama_kv_stream_device_buffer_type(dev);
        const auto allocate=type->iface.alloc_buffer;
        type->iface.alloc_buffer=[](ggml_backend_buffer_type_t,size_t)->ggml_backend_buffer_t { return nullptr; };
        const bool truncated=model->truncate(250);
        type->iface.alloc_buffer=allocate;
        if (!t.assert_true(truncated && model->complete())) return;
        t.assert_true(model->binding_view().buffer == view.buffer);
        t.assert_equal(granted,model->granted_bytes());
        t.assert_equal(size_t(250),model->tokens());
        t.assert_true(model->truncate(250));
        t.assert_true(!model->truncate(251));
        resident_upload_probe probe(view.buffer);
        if (!t.assert_true(model->begin(252,2,true))) return;
        t.assert_equal(size_t(2),probe.calls);
        t.assert_equal(6*(model->host()->layout().k_token_bytes+model->host()->layout().v_token_bytes),probe.bytes);
        t.assert_true(model->binding_view().buffer == view.buffer);
        model->abort();
    });
    t.test("suffix_truncation_reopens_frontier_without_erasing_host_bytes", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model=llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;
        ggml_backend_buffer_clear(model->buffer(),0);
        t.assert_true(model->restore(513));
        auto * raw=static_cast<uint8_t *>(ggml_backend_buffer_get_base(model->host()->buffer()));
        raw[model->host()->bytes()-1]=91;
        t.assert_true(!model->truncate(514));
        t.assert_true(model->truncate(256));
        t.assert_equal(size_t(256),model->tokens());
        t.assert_equal(uint8_t(91),raw[model->host()->bytes()-1]);
        t.assert_true(model->begin(257,1,true)); model->abort();
    });
    t.test("ordinary_graph_dispatches_kv_producers_and_streamed_attention", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,1);
        ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),page.bytes*4,256,4});
        if (!t.assert_true(bool(model))) return;
        const auto prefill_grants = model->granted_bytes();
        f.host = model->host();
        size_t active = 0;
        size_t step = 0;
        for (uint32_t rows : {256u,256u,1u,1u}) {
            const bool replay = step++ == 3;
            if (replay) {
                t.assert_true(model->truncate(256));
                active = 256;
            }
            const size_t first = active; active += rows;
            ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
            auto * source = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,512,rows);
            auto * indices = ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I64,rows);
            ggml_backend_buffer_ptr inputs(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
            std::vector<float> data(rows*512); std::vector<int64_t> ids(rows);
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = .25f*std::sin(float((first*512+i)%677)*.07f)+(replay ? .125f : 0);
            const auto expected_k = reference_bytes(backend.get(),data,GGML_TYPE_Q8_0,2);
            const auto expected_v = reference_bytes(backend.get(),data,GGML_TYPE_Q4_0,3);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = int64_t(first+i);
            ggml_backend_tensor_set(source,data.data(),0,data.size()*sizeof(float));
            ggml_backend_tensor_set(indices,ids.data(),0,ids.size()*sizeof(int64_t));
            auto * k = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q8_0,512,f.host->layout().tokens);
            auto * v = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q4_0,512,f.host->layout().tokens);
            llama_kv_stream_host_layer planes; f.host->layer(0,planes);
            t.assert_true(ggml_backend_tensor_alloc(model->buffer(),k,planes.k) == GGML_STATUS_SUCCESS);
            t.assert_true(ggml_backend_tensor_alloc(model->buffer(),v,planes.v) == GGML_STATUS_SUCCESS);
            auto * graph = ggml_new_graph_custom(ctx.get(),128,false);
            auto * k_source = ggml_scale(ctx.get(),source,2);
            auto * v_source = ggml_scale(ctx.get(),source,3);
            ggml_build_forward_expand(graph,k_source); ggml_build_forward_expand(graph,v_source);
            auto * k_write = ggml_set_rows(ctx.get(),k,k_source,indices);
            auto * v_write = ggml_set_rows(ctx.get(),v,v_source,indices); v_write->src[3] = k_source;
            ggml_build_forward_expand(graph,k_write); ggml_build_forward_expand(graph,v_write);
            block_inputs attn(f,active,rows);
            auto * key = ggml_view_3d(ctx.get(),k,256,attn.padded,2,f.host->layout().k_token_bytes,f.host->layout().k_row_bytes,0);
            auto * value = ggml_view_3d(ctx.get(),v,256,attn.padded,2,f.host->layout().v_token_bytes,f.host->layout().v_row_bytes,0);
            auto * output = ggml_flash_attn_ext(ctx.get(),attn.q,key,value,attn.mask,1.0f/16,0,0);
            ggml_flash_attn_ext_set_prec(output,GGML_PREC_F32);
            ggml_build_forward_expand(graph,output);
            ggml_backend_t backends[]{backend.get(),cpu.get()};
            ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends,nullptr,2,128,false,true));
            // Validate before allocation can abort on an unsupported preallocated destination.
            if (!t.assert_true(ggml_backend_supports_op(backend.get(),k_write) && ggml_backend_supports_op(backend.get(),v_write))) return;
            if (!t.assert_true(ggml_backend_sched_alloc_graph(sched.get(),graph))) return;
            if (!t.assert_true(model->begin(active,rows,rows == 1))) return;
            if (rows == 1) {
                llama_kv_stream_runtime_diagnostics onset;
                t.assert_true(model->runtime_diagnostics(onset));
                t.assert_true(onset.streaming_active);
                t.assert_true(onset.active_pages > onset.resident_pages_per_layer);
                t.assert_true(onset.ring_slots > 0);
            }
            if (rows == 1) t.assert_true(model->granted_bytes() < prefill_grants);
            if (!t.assert_true(ggml_backend_sched_graph_compute(sched.get(),graph) == GGML_STATUS_SUCCESS)) return;
            t.assert_true(model->complete()); t.assert_equal(active,model->tokens());
            t.assert_true(!std::memcmp(static_cast<const char *>(planes.k)+first*f.host->layout().k_token_bytes,expected_k.data(),expected_k.size()));
            t.assert_true(!std::memcmp(static_cast<const char *>(planes.v)+first*f.host->layout().v_token_bytes,expected_v.data(),expected_v.size()));
            std::vector<float> actual(ggml_nelements(output)); ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
            close_values(t,oracle(f,0,active,rows,attn.qdata),actual,1e-3f);
        }
        auto * type=llama_kv_stream_device_buffer_type(dev);
        const auto allocate=type->iface.alloc_buffer;
        type->iface.alloc_buffer=[](ggml_backend_buffer_type_t,size_t)->ggml_backend_buffer_t { return nullptr; };
        const bool truncated=model->truncate(256);
        type->iface.alloc_buffer=allocate;
        t.assert_true(truncated && model->complete());
        t.assert_equal(size_t(256),model->tokens());
    });
    t.test("auxiliary_mtp_cache_shares_physical_policy_without_merging_identity", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 513, false, 2, 2, 7);
        ggml_kv_stream_layout page;
        if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape, 256, page).status ==
                ggml_kv_stream_status::success)) return;
        llama_kv_stream_model_config config;
        config.backend = backend.get();
        config.host = f.host->config();
        config.pool_bytes = 7*page.bytes;
        config.max_batch_rows = 1;
        config.query_heads = 4;
        config.auxiliary_cache_layers = 1;
        auto model = llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(model))) return;
        auto mtp = model->auxiliary_cache();
        if (!t.assert_true(bool(mtp))) return;
        t.assert_true(model->host() != mtp->host());
        t.assert_true(model->host()->cache_id() != mtp->host()->cache_id());
        t.assert_equal(uint32_t(2), model->host()->config().layers);
        t.assert_equal(uint32_t(1), mtp->host()->config().layers);
        t.assert_equal(model->host()->config().context_tokens, mtp->host()->config().context_tokens);
        t.assert_equal(size_t(0), mtp->tokens());
        t.assert_equal(size_t(0), mtp->frontiers().device);
        t.assert_true(ggml_backend_buffer_get_type(mtp->host()->buffer()) ==
            llama_kv_stream_host_buffer_type(dev));
        const auto view = model->binding_view();
        t.assert_equal(uint32_t(3), view.config.layers);
        if (!t.assert_equal(size_t(2), view.config.caches.size())) return;
        t.assert_equal(model->host()->cache_id(), view.config.caches[0].id);
        t.assert_equal(uint32_t(2), view.config.caches[0].layers);
        t.assert_equal(mtp->host()->cache_id(), view.config.caches[1].id);
        t.assert_equal(uint32_t(1), view.config.caches[1].layers);
        llama_kv_stream_policy_state state;
        if (!t.assert_true(llama_kv_stream_policy_initialize(view.config, state).status ==
                llama_kv_stream_policy_status::success)) return;
        llama_kv_stream_policy_layout layout;
        t.assert_true(llama_kv_stream_policy_layout_make(view.config, state, 0, layout).status ==
            llama_kv_stream_policy_status::success);
        auto guard_owner = llama_kv_stream_layer_lease_owner::create(view.lease,
            {view.config, state, 1, view.revision, mtp->identity().generation});
        if (!t.assert_true(bool(guard_owner))) return;
        auto guard = guard_owner->hold_ring();
        if (!t.assert_true(bool(guard))) return;
        t.assert_true(model->set_ring_guard(guard));
        t.assert_true(!guard_owner->can_repartition());
        t.assert_true(model->set_ring_guard({}));
        guard.reset();
        t.assert_true(guard_owner->can_repartition());
        t.assert_true(model->begin(1, 1, false));
        model->abort();
        const auto retained_host = mtp->host();
        model.reset();
        t.assert_true(mtp->host() == retained_host && mtp->host()->cache_id() != 0);

        config.auxiliary_cache_layers = 2;
        t.assert_true(!llama_kv_stream_model::create(config));
        config.auxiliary_cache_layers = 1;
        config.pool_bytes = 3*page.bytes;
        t.assert_true(!llama_kv_stream_model::create(config));
    });
    t.test("populated_mtp_lease_reuses_one_upload_for_tg1_to_tg4", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 513, false, 2, 2, 7);
        ggml_kv_stream_layout page;
        if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape, 256, page).status ==
                ggml_kv_stream_status::success)) return;
        llama_kv_stream_model_config config;
        config.backend = backend.get();
        config.host = f.host->config();
        config.pool_bytes = 7*page.bytes;
        config.max_batch_rows = 4;
        config.query_heads = 4;
        config.auxiliary_cache_layers = 1;
        auto model = llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(model))) return;
        auto mtp = model->auxiliary_cache();
        if (!t.assert_true(bool(mtp))) return;
        const auto & layout = mtp->host()->layout();
        llama_kv_stream_host_layer seed;
        if (!t.assert_true(f.host->layer(0, seed))) return;
        std::vector<uint8_t> keys(static_cast<const uint8_t *>(seed.k),
            static_cast<const uint8_t *>(seed.k) + 257*layout.k_token_bytes);
        std::vector<uint8_t> values(static_cast<const uint8_t *>(seed.v),
            static_cast<const uint8_t *>(seed.v) + 257*layout.v_token_bytes);
        llama_kv_stream_write write;
        if (!t.assert_true(mtp->begin(257) && mtp->content()->prepare({
                {0, ggml_kv_stream_operand::k, 0, keys.data(), keys.size()},
                {0, ggml_kv_stream_operand::v, 0, values.data(), values.size()}}, write) &&
                mtp->publish_host(write) && mtp->finish())) return;
        ggml_backend_buffer_clear(model->buffer(), 0);
        if (!t.assert_true(model->restore(258))) return;
        t.assert_true(model->acquire_mtp_layer(4));
        t.assert_equal(size_t(262), model->mtp_reserved_tokens());
        t.assert_true(model->has_mtp_layer());
        t.assert_true(!model->truncate(256));
        t.assert_true(!model->prepare_shared_memory());
        const auto copied = model->mtp_layer_population();
        t.assert_true(copied.bytes > 0 && copied.calls > 0);
        for (uint32_t width = 1; width <= 4; ++width) {
            t.assert_true(model->mtp_layer_plan(width) != nullptr);
            t.assert_equal(copied.bytes, model->mtp_layer_population().bytes);
        }
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 7 && ops->spans && ops->spans_workspace)) return;
        for (uint32_t width = 1; width <= 4; ++width) {
            block_inputs input(f, 257, width);
            const auto expected = stock_attention(f, input, 0);
            ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
            auto * key_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 512, input.padded);
            auto * value_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_0, 512, input.padded);
            auto * key = ggml_view_3d(ctx.get(), key_storage, 256, input.padded, 2,
                ggml_row_size(GGML_TYPE_Q8_0, 512), ggml_row_size(GGML_TYPE_Q8_0, 256), 0);
            auto * value = ggml_view_3d(ctx.get(), value_storage, 256, input.padded, 2,
                ggml_row_size(GGML_TYPE_Q4_0, 512), ggml_row_size(GGML_TYPE_Q4_0, 256), 0);
            auto * node = ggml_flash_attn_ext(ctx.get(), input.q, key, value,
                input.mask, 1.0f/16, 0, 0);
            if (!t.assert_true(node != nullptr)) return;
            ggml_flash_attn_ext_set_prec(node, GGML_PREC_F32);
            ggml_tensor op = *node;
            op.buffer = input.output->buffer;
            op.data = input.output->data;
            auto * plan = model->mtp_layer_plan(width);
            ggml_kv_stream_span_plan_view view;
            if (!t.assert_true(ggml_kv_stream_span_plan_get_view(plan, view))) return;
            t.assert_equal(size_t(257), view.active_tokens);
            t.assert_equal(size_t(width), view.query_tokens);
            size_t workspace_bytes = 0;
            if (!t.assert_true(ops->spans_workspace(backend.get(), &op, plan, workspace_bytes))) return;
            ggml_backend_buffer_ptr workspace(ggml_backend_buft_alloc_buffer(
                llama_kv_stream_device_buffer_type(dev), workspace_bytes));
            if (!t.assert_true(bool(workspace) &&
                    ops->spans(backend.get(), &op, plan, workspace.get()))) return;
            ggml_backend_synchronize(backend.get());
            t.out << "MTP retained span TG" << width << ": ";
            close_values(t, expected, input.read(), 1e-5f);
        }
        std::vector<uint8_t> tail_k(static_cast<const uint8_t *>(seed.k) + 257*layout.k_token_bytes,
            static_cast<const uint8_t *>(seed.k) + 258*layout.k_token_bytes);
        std::vector<uint8_t> tail_v(static_cast<const uint8_t *>(seed.v) + 257*layout.v_token_bytes,
            static_cast<const uint8_t *>(seed.v) + 258*layout.v_token_bytes);
        llama_kv_stream_write tail;
        if (!t.assert_true(mtp->begin(1) && mtp->content()->prepare({
                {0, ggml_kv_stream_operand::k, 257*layout.k_token_bytes, tail_k.data(), tail_k.size()},
                {0, ggml_kv_stream_operand::v, 257*layout.v_token_bytes, tail_v.data(), tail_v.size()}}, tail) &&
                mtp->publish_host(tail) && mtp->finish())) return;
        if (!t.assert_true(model->advance_mtp_layer_tail())) return;
        const auto extended = model->mtp_layer_population();
        t.assert_equal(copied.bytes + layout.k_token_bytes + layout.v_token_bytes, extended.bytes);
        for (uint32_t width = 1; width <= 4; ++width) {
            ggml_kv_stream_span_plan_view renewed;
            if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                    model->mtp_layer_plan(width), renewed))) return;
            t.assert_equal(size_t(258), renewed.active_tokens);
        }
        for (size_t token = 258; token < 262; ++token) {
            std::vector<uint8_t> next_k(static_cast<const uint8_t *>(seed.k) +
                    token*layout.k_token_bytes,
                static_cast<const uint8_t *>(seed.k) + (token + 1)*layout.k_token_bytes);
            std::vector<uint8_t> next_v(static_cast<const uint8_t *>(seed.v) +
                    token*layout.v_token_bytes,
                static_cast<const uint8_t *>(seed.v) + (token + 1)*layout.v_token_bytes);
            llama_kv_stream_write next;
            if (!t.assert_true(mtp->begin(1) && mtp->content()->prepare({
                    {0, ggml_kv_stream_operand::k, token*layout.k_token_bytes,
                        next_k.data(), next_k.size()},
                    {0, ggml_kv_stream_operand::v, token*layout.v_token_bytes,
                        next_v.data(), next_v.size()}}, next) &&
                    mtp->publish_host(next) && mtp->finish() &&
                    model->advance_mtp_layer_tail())) return;
            t.assert_equal(token + 1, mtp->tokens());
            t.assert_equal(copied.bytes + (token + 1 - 257)*
                (layout.k_token_bytes + layout.v_token_bytes),
                model->mtp_layer_population().bytes);
            ggml_kv_stream_span_plan_view draft_plan;
            if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                    model->mtp_layer_plan(1), draft_plan))) return;
            t.assert_equal(token + 1, draft_plan.active_tokens);
        }
        const auto before_reject = model->mtp_layer_population();
        if (!t.assert_true(model->truncate_mtp_layer(260))) return;
        t.assert_equal(size_t(260), mtp->tokens());
        t.assert_true(model->has_mtp_layer());
        t.assert_equal(size_t(262), model->mtp_reserved_tokens());
        t.assert_equal(before_reject.bytes, model->mtp_layer_population().bytes);
        ggml_kv_stream_span_plan_view accepted_plan;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                model->mtp_layer_plan(4), accepted_plan))) return;
        t.assert_equal(size_t(260), accepted_plan.active_tokens);
        std::vector<uint8_t> accepted_k(static_cast<const uint8_t *>(seed.k) +
                260*layout.k_token_bytes,
            static_cast<const uint8_t *>(seed.k) + 261*layout.k_token_bytes);
        std::vector<uint8_t> accepted_v(static_cast<const uint8_t *>(seed.v) +
                260*layout.v_token_bytes,
            static_cast<const uint8_t *>(seed.v) + 261*layout.v_token_bytes);
        llama_kv_stream_write accepted;
        if (!t.assert_true(mtp->begin(1) && mtp->content()->prepare({
                {0, ggml_kv_stream_operand::k, 260*layout.k_token_bytes,
                    accepted_k.data(), accepted_k.size()},
                {0, ggml_kv_stream_operand::v, 260*layout.v_token_bytes,
                    accepted_v.data(), accepted_v.size()}}, accepted) &&
                mtp->publish_host(accepted) && mtp->finish() &&
                model->advance_mtp_layer_tail())) return;
        t.assert_equal(before_reject.bytes + layout.k_token_bytes + layout.v_token_bytes,
            model->mtp_layer_population().bytes);
        t.assert_true(!model->acquire_mtp_layer());
        t.assert_true(model->release_mtp_layer());
        t.assert_true(model->mtp_layer_plan(1) == nullptr);
        t.assert_true(mtp->truncate(256));
        t.assert_true(model->acquire_mtp_layer());
        t.assert_equal(size_t(0),model->mtp_layer_population().bytes);
        t.assert_true(model->release_mtp_layer());
        t.assert_true(mtp->restore(mtp->checkpoint()));
        const auto upload=backend->iface.set_tensor_async;
        backend->iface.set_tensor_async=[](ggml_backend_t,ggml_tensor *,const void *,size_t,size_t) {
            throw std::runtime_error("injected MTP resident refresh failure");
        };
        const bool failed_population=model->acquire_mtp_layer();
        backend->iface.set_tensor_async=upload;
        t.assert_true(!failed_population && !model->has_mtp_layer() && model->complete());
        t.assert_true(model->acquire_mtp_layer());
        t.assert_equal(256*(layout.k_token_bytes+layout.v_token_bytes),model->mtp_layer_population().bytes);
        t.assert_true(model->release_mtp_layer());
        t.assert_true(model->prepare_shared_memory() && model->resume_private_memory());
        t.assert_true(model->acquire_mtp_layer());
        t.assert_equal(256*(layout.k_token_bytes+layout.v_token_bytes),model->mtp_layer_population().bytes);
        model->abort();
        t.assert_true(model->release_mtp_layer());
        t.assert_true(!model->has_mtp_layer());
        t.assert_true(model->reset(false));
    });
    t.test("mtp_prefill_admission_demotes_resident_pages_before_population", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 6145, false, 2, 2, 29);
        ggml_kv_stream_layout page;
        if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape, 256, page).status ==
                ggml_kv_stream_status::success)) return;
        llama_kv_stream_model_config config;
        config.backend = backend.get();
        config.host = f.host->config();
        config.pool_bytes = 29 * page.bytes;
        config.max_batch_rows = 4;
        config.query_heads = 4;
        config.auxiliary_cache_layers = 1;
        auto model = llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(model))) return;
        auto mtp = model->auxiliary_cache();
        llama_kv_stream_host_layer source;
        if (!t.assert_true(mtp && f.host->layer(0, source))) return;
        const auto & layout = mtp->host()->layout();
        llama_kv_stream_write write;
        if (!t.assert_true(mtp->begin(5120) && mtp->content()->prepare({
                {0, ggml_kv_stream_operand::k, 0, source.k, 5120 * layout.k_token_bytes},
                {0, ggml_kv_stream_operand::v, 0, source.v, 5120 * layout.v_token_bytes}}, write) &&
                mtp->publish_host(write) && mtp->finish())) return;
        ggml_backend_buffer_clear(model->buffer(), 0);
        if (!t.assert_true(model->restore(5121))) return;
        llama_kv_stream_runtime_diagnostics before, after;
        if (!t.assert_true(model->runtime_diagnostics(before))) return;
        t.assert_true(before.ring_slots < 21 - before.resident_pages_per_layer);
        const auto parent = model->binding_view().buffer;
        if (!t.assert_true(model->acquire_mtp_layer(4))) return;
        if (!t.assert_true(model->runtime_diagnostics(after))) return;
        t.assert_true(after.ring_slots > before.ring_slots);
        t.assert_true(after.resident_pages_per_layer < before.resident_pages_per_layer);
        t.assert_true(after.layout_revision > before.layout_revision);
        t.assert_equal(before.pool_bytes, after.pool_bytes);
        t.assert_true(parent == model->binding_view().buffer);
        t.assert_equal(size_t(5125), model->mtp_reserved_tokens());
        for (uint32_t width = 1; width <= 4; ++width) {
            ggml_kv_stream_span_plan_view plan;
            if (!t.assert_true(ggml_kv_stream_span_plan_get_view(model->mtp_layer_plan(width), plan))) return;
            t.assert_equal(size_t(5120), plan.active_tokens);
            t.assert_equal(size_t(2), plan.count);
            block_inputs input(f, 5120, width);
            const auto expected = stock_attention(f, input, 0);
            ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
            auto * ks = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 512, input.padded);
            auto * vs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_0, 512, input.padded);
            auto * k = ggml_view_3d(ctx.get(), ks, 256, input.padded, 2,
                ggml_row_size(GGML_TYPE_Q8_0, 512), ggml_row_size(GGML_TYPE_Q8_0, 256), 0);
            auto * v = ggml_view_3d(ctx.get(), vs, 256, input.padded, 2,
                ggml_row_size(GGML_TYPE_Q4_0, 512), ggml_row_size(GGML_TYPE_Q4_0, 256), 0);
            auto * node = ggml_flash_attn_ext(ctx.get(), input.q, k, v, input.mask, 1.0f/16, 0, 0);
            ggml_flash_attn_ext_set_prec(node, GGML_PREC_F32);
            ggml_tensor op = *node;
            op.buffer = input.output->buffer;
            op.data = input.output->data;
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                    "ggml_backend_kv_stream_partial_ops"));
            const auto * ops = get ? get() : nullptr;
            size_t bytes = 0;
            if (!t.assert_true(ops && ops->spans_workspace(backend.get(), &op,
                    model->mtp_layer_plan(width), bytes))) return;
            ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(
                llama_kv_stream_device_buffer_type(dev), bytes));
            if (!t.assert_true(scratch && ops->spans(backend.get(), &op,
                    model->mtp_layer_plan(width), scratch.get()))) return;
            ggml_backend_synchronize(backend.get());
            close_values(t, expected, input.read(), 1e-5f);
        }
        t.assert_true(model->release_mtp_layer());
        t.assert_equal(size_t(5121), model->tokens());
        t.assert_equal(size_t(5120), mtp->tokens());
        t.assert_true(model->acquire_mtp_layer(4));
        llama_kv_stream_runtime_diagnostics repeated;
        if (!t.assert_true(model->runtime_diagnostics(repeated))) return;
        t.assert_equal(after.layout_revision, repeated.layout_revision);
        t.assert_true(model->release_mtp_layer());

        config.pool_bytes = 7 * page.bytes;
        auto insufficient = llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(insufficient))) return;
        auto pending = insufficient->auxiliary_cache();
        if (!t.assert_true(pending->begin(5120) && pending->content()->prepare({
                {0, ggml_kv_stream_operand::k, 0, source.k, 5120 * layout.k_token_bytes},
                {0, ggml_kv_stream_operand::v, 0, source.v, 5120 * layout.v_token_bytes}}, write) &&
                pending->publish_host(write) && pending->finish())) return;
        ggml_backend_buffer_clear(insufficient->buffer(), 0);
        if (!t.assert_true(insufficient->restore(5121) &&
                insufficient->runtime_diagnostics(before))) return;
        t.assert_true(!insufficient->acquire_mtp_layer(4));
        t.assert_true(!insufficient->has_mtp_layer());
        t.assert_true(insufficient->complete());
        if (!t.assert_true(insufficient->runtime_diagnostics(after))) return;
        t.assert_equal(before.layout_revision, after.layout_revision);
        t.assert_equal(before.ring_slots, after.ring_slots);
        t.assert_equal(size_t(5121), insufficient->tokens());
        t.assert_equal(size_t(5120), pending->tokens());
    });

    t.test("mtp_prefill_uses_configured_microbatch_rows", [&](testing & t) {
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,1024,false,2,2,7);
        ggml_kv_stream_layout page;
        if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape,256,page).status ==
                ggml_kv_stream_status::success)) return;
        llama_kv_stream_model_config config;
        config.backend=backend.get();
        config.host=f.host->config();
        config.pool_bytes=7*page.bytes;
        config.max_batch_rows=512;
        config.query_heads=4;
        config.auxiliary_cache_layers=1;
        auto model=llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(model))) return;
        auto cache=model->auxiliary_cache();
        auto proxy=llama_kv_stream_mtp_proxy::create(dev,cache,model.get());
        if (!t.assert_true(bool(proxy))) return;
        ggml_context_ptr ctx(ggml_init({8192,nullptr,true}));
        auto * k=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,512,512);
        auto * v=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,512,512);
        auto * indices=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I64,512);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(k && v && indices && bool(source))) return;
        std::vector<float> data(512*512);
        for (size_t i=0;i<data.size();++i) data[i]=.2f*std::sin(float(i%401)*.13f);
        std::vector<int64_t> positions(512);
        for (size_t i=0;i<positions.size();++i) positions[i]=int64_t(i);
        ggml_backend_tensor_set(k,data.data(),0,data.size()*sizeof(float));
        ggml_backend_tensor_set(v,data.data(),0,data.size()*sizeof(float));
        ggml_backend_tensor_set(indices,positions.data(),0,positions.size()*sizeof(int64_t));
        ggml_tensor k_op{},v_op{};
        k_op.op=v_op.op=GGML_OP_SET_ROWS;
        k_op.src[0]=k; k_op.src[1]=indices; k_op.src[2]=proxy->key();
        v_op.src[0]=v; v_op.src[1]=indices; v_op.src[2]=proxy->value(); v_op.src[3]=k;
        auto * owner=proxy->key()->buffer;
        if (!t.assert_true(ggml_backend_execution_supports(owner,dev,&k_op) &&
                ggml_backend_execution_supports(owner,dev,&v_op))) return;
        block_inputs input(f,512,512);
        auto * key=ggml_view_3d(ctx.get(),proxy->key(),256,512,2,
            ggml_row_size(GGML_TYPE_Q8_0,512),ggml_row_size(GGML_TYPE_Q8_0,256),0);
        auto * value=ggml_view_3d(ctx.get(),proxy->value(),256,512,2,
            ggml_row_size(GGML_TYPE_Q4_0,512),ggml_row_size(GGML_TYPE_Q4_0,256),0);
        if (!t.assert_true(key && value)) return;
        ggml_tensor attention=*input.output;
        attention.op=GGML_OP_FLASH_ATTN_EXT;
        attention.src[0]=input.q; attention.src[1]=key;
        attention.src[2]=value; attention.src[3]=input.mask;
        if (!t.assert_true(ggml_backend_execution_supports(owner,dev,&attention))) return;
        if (!t.assert_equal(GGML_STATUS_SUCCESS,
                ggml_backend_execution_compute(owner,backend.get(),&k_op)) ||
                !t.assert_equal(GGML_STATUS_SUCCESS,
                ggml_backend_execution_compute(owner,backend.get(),&v_op)) ||
                !t.assert_true(proxy->complete_publication())) return;
        t.assert_equal(size_t(512),cache->tokens());
        const auto expected_k=reference_bytes(backend.get(),data,GGML_TYPE_Q8_0,1);
        const auto expected_v=reference_bytes(backend.get(),data,GGML_TYPE_Q4_0,1);
        llama_kv_stream_host_layer host;
        if (!t.assert_true(cache->host()->layer(0,host))) return;
        t.assert_true(!std::memcmp(host.k,expected_k.data(),expected_k.size()));
        t.assert_true(!std::memcmp(host.v,expected_v.data(),expected_v.size()));
        config.max_batch_rows=128;
        auto limited_model=llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(limited_model))) return;
        auto limited_proxy=llama_kv_stream_mtp_proxy::create(
            dev,limited_model->auxiliary_cache(),limited_model.get());
        if (!t.assert_true(bool(limited_proxy))) return;
        ggml_tensor too_wide=k_op;
        too_wide.src[2]=limited_proxy->key();
        t.assert_true(!ggml_backend_execution_supports(limited_proxy->key()->buffer,dev,&too_wide));
    });
    t.test("mtp_proxy_cancellation_drains_pending_writes", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 513, false, 2, 2, 7);
        ggml_kv_stream_layout page;
        if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape, 256, page).status ==
                ggml_kv_stream_status::success)) return;
        llama_kv_stream_model_config config;
        config.backend = backend.get();
        config.host = f.host->config();
        config.pool_bytes = 7*page.bytes;
        config.max_batch_rows = 4;
        config.query_heads = 4;
        config.auxiliary_cache_layers = 1;
        auto model = llama_kv_stream_model::create(config);
        if (!t.assert_true(bool(model))) return;
        auto cache = model->auxiliary_cache();
        llama_kv_stream_host_layer seed;
        if (!t.assert_true(bool(cache) && f.host->layer(0, seed))) return;
        const auto & layout = cache->host()->layout();
        std::vector<uint8_t> keys(static_cast<const uint8_t *>(seed.k),
            static_cast<const uint8_t *>(seed.k) + 256*layout.k_token_bytes);
        std::vector<uint8_t> values(static_cast<const uint8_t *>(seed.v),
            static_cast<const uint8_t *>(seed.v) + 256*layout.v_token_bytes);
        llama_kv_stream_write write;
        ggml_backend_buffer_clear(model->buffer(), 0);
        if (!t.assert_true(cache->begin(256) && cache->content()->prepare({
                {0, ggml_kv_stream_operand::k, 0, keys.data(), keys.size()},
                {0, ggml_kv_stream_operand::v, 0, values.data(), values.size()}}, write) &&
                cache->publish_host(write) && cache->finish() &&
                model->restore(257) && model->acquire_mtp_layer(1))) return;
        auto proxy = llama_kv_stream_mtp_proxy::create(dev, cache, model.get());
        if (!t.assert_true(bool(proxy))) return;
        ggml_context_ptr source_context(ggml_init({8192, nullptr, true}));
        if (!t.assert_true(bool(source_context))) return;
        auto * k = ggml_new_tensor_2d(source_context.get(), GGML_TYPE_F32, 512, 1);
        auto * v = ggml_new_tensor_2d(source_context.get(), GGML_TYPE_F32, 512, 1);
        auto * indices = ggml_new_tensor_1d(source_context.get(), GGML_TYPE_I64, 1);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(source_context.get(), backend.get()));
        if (!t.assert_true(k && v && indices && bool(source))) return;
        std::vector<float> row(512, 0.25f);
        const int64_t index = 256;
        ggml_backend_tensor_set(k, row.data(), 0, row.size()*sizeof(float));
        ggml_backend_tensor_set(v, row.data(), 0, row.size()*sizeof(float));
        ggml_backend_tensor_set(indices, &index, 0, sizeof(index));
        ggml_tensor k_op{}, v_op{};
        k_op.op = v_op.op = GGML_OP_SET_ROWS;
        k_op.src[0] = k; k_op.src[1] = indices; k_op.src[2] = proxy->key();
        v_op.src[0] = v; v_op.src[1] = indices; v_op.src[2] = proxy->value(); v_op.src[3] = k;
        auto * owner = proxy->key()->buffer;
        if (!t.assert_true(proxy->arm(256, 1) &&
                ggml_backend_execution_compute(owner, backend.get(), &k_op) == GGML_STATUS_SUCCESS)) return;
        if (!t.assert_true(proxy->remove_suffix(256, SIZE_MAX))) return;
        t.assert_equal(size_t(256), cache->tokens());
        if (!t.assert_true(proxy->arm(256, 1) &&
                ggml_backend_execution_compute(owner, backend.get(), &k_op) == GGML_STATUS_SUCCESS &&
                ggml_backend_execution_compute(owner, backend.get(), &v_op) == GGML_STATUS_SUCCESS)) return;
        t.assert_equal(size_t(256), cache->tokens());
        t.assert_equal(size_t(256), cache->frontiers().host);
        if (!t.assert_true(proxy->remove_suffix(256, SIZE_MAX))) return;
        t.assert_equal(size_t(256), cache->frontiers().reserved);
        t.assert_equal(size_t(256), cache->frontiers().host);
        t.assert_equal(size_t(256), cache->tokens());
        t.assert_true(!model->has_mtp_layer());
        t.assert_true(model->acquire_mtp_layer(1));
        proxy.reset();
        t.assert_true(!model->has_mtp_layer());
        if (!t.assert_true(model->acquire_mtp_layer(1))) return;
        proxy = llama_kv_stream_mtp_proxy::create(dev, cache, model.get());
        if (!t.assert_true(bool(proxy) && proxy->arm(256, 1))) return;
        owner = proxy->key()->buffer;
        {
            index_read_fault fault(source.get());
            t.assert_equal(GGML_STATUS_FAILED,
                ggml_backend_execution_compute(owner, backend.get(), &k_op));
        }
        if (!t.assert_true(!proxy->arm(256, 1))) return;
        t.assert_true(proxy->remove_suffix(0, SIZE_MAX));
        t.assert_equal(size_t(0), cache->tokens());
        t.assert_true(!model->has_mtp_layer());
        llama_kv_stream_write retry;
        if (!t.assert_true(cache->begin(256) && cache->content()->prepare({
                {0, ggml_kv_stream_operand::k, 0, keys.data(), keys.size()},
                {0, ggml_kv_stream_operand::v, 0, values.data(), values.size()}}, retry) &&
                cache->publish_host(retry) && cache->finish() &&
                model->acquire_mtp_layer(1) && proxy->arm(256, 1))) return;
        if (!t.assert_equal(GGML_STATUS_SUCCESS,
                ggml_backend_execution_compute(owner, backend.get(), &k_op))) return;
        {
            proxy_d2d_fault fault(backend.get());
            t.assert_equal(GGML_STATUS_FAILED,
                ggml_backend_execution_compute(owner, backend.get(), &v_op));
            t.assert_true(fault.calls > 0);
        }
        t.assert_equal(size_t(256), cache->tokens());
        t.assert_true(!proxy->arm(256, 1));
        if (!t.assert_true(proxy->remove_suffix(0, SIZE_MAX))) return;
        t.assert_equal(size_t(0), cache->frontiers().reserved);
        t.assert_equal(size_t(0), cache->tokens());
        t.assert_true(!model->has_mtp_layer());
    });
    return t.summary();
}

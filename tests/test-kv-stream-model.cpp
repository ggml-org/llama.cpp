#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-kv-stream-logical-cache.h"
#include "../src/llama-kv-stream-layer-lease.h"
#include "../src/llama-kv-stream-mtp-proxy.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../src/llama-context-memory.h"

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
        if (active->failures) {
            --active->failures;
            return nullptr;
        }
        return active->original(parent,offset,size);
    }
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

int main(int argc,char ** argv) {
    const bool auxiliary_only = argc > 1 && std::strcmp(argv[1], "--cuda-auxiliary-cache") == 0;
    const bool cancel_only = argc > 1 && std::strcmp(argv[1], "--cuda-mtp-cancel") == 0;
    const bool lease_only = argc > 1 && std::strcmp(argv[1], "--cuda-mtp-lease") == 0;
    const bool admission_only = argc > 1 && std::strcmp(argv[1], "--cuda-mtp-admission") == 0;
    testing t;
    if (admission_only) t.set_filter("mtp_prefill_admission_demotes_resident_pages_before_population");
    if (auxiliary_only) t.set_filter("auxiliary_mtp_cache_shares_physical_policy_without_merging_identity");
    if (lease_only) t.set_filter("populated_mtp_lease_reuses_one_upload_for_tg1_to_tg4");
    if (cancel_only) t.set_filter("mtp_proxy_cancellation_drains_pending_writes");
    if (argc < 2 || (!auxiliary_only && !lease_only && !cancel_only && !admission_only && std::strcmp(argv[1],"--cuda"))) {
        t.assert_true(!llama_kv_stream_model::create({})); return t.summary();
    }
    ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr)), cpu(ggml_backend_cpu_init());
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

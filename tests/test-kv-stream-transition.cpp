#include "kv-stream-block-test.h"
#include "../src/llama-kv-stream-session.h"

namespace {

constexpr llama_memory_domain_id device_domain = 17;
constexpr llama_memory_resource_id pool_resource = 9;
constexpr llama_memory_stage_id prefill_stage = 81;
constexpr llama_memory_stage_id decode_stage = 82;

llama_memory_transition_target target_for(fixture & f, size_t bytes, bool decode) {
    auto * parent = ggml_backend_memory_arena_parent(f.arena.get());
    const size_t alignment = ggml_backend_buffer_get_alignment(parent);
    const auto allocation = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
    llama_memory_transition_target target;
    target.plan.domains = {{device_domain, allocation, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}};
    target.plan.resources = {{pool_resource, device_domain, allocation, llama_memory_content::reconstructible}};
    target.plan.stages = {{decode ? decode_stage : prefill_stage, {},
        {{pool_resource, bytes, bytes, alignment, LLAMA_MEMORY_ACCESS_READ_WRITE,
          LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS}}}};
    target.plan.inputs = {pool_resource};
    target.plan.outputs = {pool_resource};
    target.stage = decode ? decode_stage : prefill_stage;
    target.budgets = {{device_domain, allocation, ggml_backend_memory_arena_capacity(f.arena.get()), alignment}};
    return target;
}

std::unique_ptr<llama_kv_stream_session> make_session(
        fixture & f, block_workspace & writer, block_workspace & partial, bool resumed = false) {
    llama_kv_stream_session_config config{f.policy, 256, 4, false, true, resumed};
    config.pool_resource = pool_resource;
    config.prefill_stage = prefill_stage;
    config.decode_stage = decode_stage;
    return llama_kv_stream_session::create(
        f.backend, f.content, config, f.lease.get(), writer.lease.get(), partial.lease.get());
}

size_t minimum_pool(const fixture & f) {
    ggml_kv_stream_layout page;
    ggml_kv_stream_execution execution;
    GGML_ASSERT(ggml_kv_stream_layout_make(f.policy.shape, 256, page).status == ggml_kv_stream_status::success);
    GGML_ASSERT(ggml_kv_stream_resolve(
        f.policy.shape, f.policy.capabilities, 256, execution).status == ggml_kv_stream_status::success);
    return (f.policy.layers + 1)*page.bytes + execution.conversion.bytes;
}

struct view_probe {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_t, size_t, size_t);
    inline static view_probe * active = nullptr;
    ggml_backend_buffer_t parent;
    factory original;
    bool fail;
    size_t calls = 0;

    view_probe(ggml_backend_memory_arena_t arena, bool fail = false) :
        parent(ggml_backend_memory_arena_parent(arena)),
        original(parent->view_buffer), fail(fail) {
        GGML_ASSERT(!active && original);
        active = this;
        parent->view_buffer = create;
    }
    ~view_probe() {
        parent->view_buffer = original;
        active = nullptr;
    }
    static ggml_backend_buffer_t create(ggml_backend_buffer_t parent, size_t offset, size_t size) {
        GGML_ASSERT(active && parent == active->parent);
        ++active->calls;
        return active->fail ? nullptr : active->original(parent, offset, size);
    }
};

struct binding_fault : llama_memory_consumer {
    ggml_backend_buffer_t writer;
    decltype(ggml_backend_buffer_i::init_tensor) original;
    bool armed = false;
    bool fail_recovery = false;

    explicit binding_fault(ggml_backend_memory_lease_t lease) :
        writer(ggml_backend_memory_lease_buffer(lease)), original(writer->iface.init_tensor) {}
    ~binding_fault() override { restore(); }

    void restore() {
        if (armed) writer->iface.init_tensor = original;
        armed = false;
    }

    struct proposal : llama_memory_preparation {
        binding_fault & owner;
        bool recovery;
        proposal(binding_fault & owner, bool recovery) : owner(owner), recovery(recovery) {}
        bool quiesce(const std::vector<llama_memory_resource_id> &) override { return true; }
        bool drain() override { return true; }
        bool invalidate() override { return true; }
        bool release() override { return true; }
        bool bind(const std::vector<llama_memory_region_binding> &) override {
            if (recovery && !owner.fail_recovery) {
                owner.restore();
            } else {
                owner.armed = true;
                owner.writer->iface.init_tensor = [](ggml_backend_buffer_t, ggml_tensor *) {
                    return GGML_STATUS_ALLOC_FAILED;
                };
            }
            return true;
        }
        bool activate() override { return true; }
        bool prepare_recovery(std::unique_ptr<llama_memory_preparation> & output) override {
            output = std::make_unique<proposal>(owner, true);
            return true;
        }
    };

    bool prepare(const llama_memory_transition_target &, const llama_memory_layout &,
            std::unique_ptr<llama_memory_preparation> & output) override {
        output = std::make_unique<proposal>(*this, false);
        return true;
    }
};

struct activation_fault : llama_memory_consumer {
    bool fail = true;
    struct proposal : llama_memory_preparation {
        activation_fault & owner;
        explicit proposal(activation_fault & owner) : owner(owner) {}
        bool quiesce(const std::vector<llama_memory_resource_id> &) override { return true; }
        bool drain() override { return true; }
        bool invalidate() override { return true; }
        bool release() override { return true; }
        bool bind(const std::vector<llama_memory_region_binding> &) override { return true; }
        bool activate() override { return !owner.fail; }
        bool prepare_recovery(std::unique_ptr<llama_memory_preparation> &) override { return true; }
    };
    bool prepare(const llama_memory_transition_target &, const llama_memory_layout &,
            std::unique_ptr<llama_memory_preparation> & output) override {
        output = std::make_unique<proposal>(*this);
        return true;
    }
};

struct transition_inputs {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * k;
    ggml_tensor * v;
    transition_inputs(ggml_backend_t backend, size_t rows) {
        context.reset(ggml_init({4096, nullptr, true}));
        k = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 512, int64_t(rows));
        v = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 512, int64_t(rows));
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        GGML_ASSERT(buffer);
        std::vector<float> data(rows*512, 0.25f);
        ggml_backend_tensor_set(k, data.data(), 0, data.size()*sizeof(float));
        ggml_backend_tensor_set(v, data.data(), 0, data.size()*sizeof(float));
    }
};

bool generate_next(testing & t, fixture & f, llama_kv_stream_session & session) {
    transition_inputs input(f.backend, 1);
    block_inputs attn(f, 514, 1);
    if (!t.assert_true(session.begin(514, 1, true))) return false;
    for (uint32_t layer = 0; layer < f.policy.layers; ++layer) {
        if (!t.assert_true(session.produce(layer, input.k, input.v)) ||
                !t.assert_true(session.attention(layer, attn.q, attn.mask, attn.output, 1.0f/16))) return false;
        close_values(t, oracle(f, layer, 514, 1, attn.qdata), attn.read(), 1e-3f);
    }
    return t.assert_equal(size_t(514), session.tokens());
}

} // namespace

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && !std::strcmp(argv[1], "--cuda");
    ggml_backend_ptr backend;
    if (cuda) {
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_name("CUDA0");
        if (!device) return 1;
        backend.reset(ggml_backend_dev_init(device, nullptr));
    } else {
        backend.reset(ggml_backend_cpu_init());
    }
    testing t;

    if (cuda) t.test("same_parent_resize_releases_old_pool_before_commit", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 769, false, 4);
        block_workspace writer(f, 32768, 19), partial(f, f.host->layout().bytes, 29);
        auto session = make_session(f, writer, partial);
        if (!t.assert_true(bool(session)) || !t.assert_true(session->restore(513))) return;
        const auto generation = f.content->generation();
        const auto old_revision = session->layout_revision();
        llama_memory_transition transition({session.get()});
        auto target = target_for(f, minimum_pool(f), true);
        if (!t.assert_true(transition.prepare(target).status == llama_memory_transition_status::prepared)) return;
        f.lease.reset();
        {
            view_probe probe(f.arena.get());
            const auto result = transition.activate({{device_domain,
                LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, f.arena.get()}});
            if (!t.assert_true(result.status == llama_memory_transition_status::activated)) return;
            t.assert_true(probe.calls > 0);
            t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.arena.get()));
        }
        t.assert_equal(minimum_pool(f), session->binding_view().capacity);
        t.assert_equal(old_revision + 1, session->layout_revision());
        t.assert_equal(generation, f.content->generation());
        t.assert_true(generate_next(t, f, *session));
    });

    if (cuda) t.test("failed_view_creation_recovers_old_pool_from_host", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 769, false, 4);
        block_workspace writer(f, 32768, 19), partial(f, f.host->layout().bytes, 29);
        auto session = make_session(f, writer, partial);
        if (!t.assert_true(bool(session)) || !t.assert_true(session->restore(513))) return;
        const auto original = session->binding_view();
        const auto generation = f.content->generation();
        llama_memory_transition transition({session.get()});
        auto target = target_for(f, minimum_pool(f), true);
        if (!t.assert_true(transition.prepare(target).status == llama_memory_transition_status::prepared)) return;
        f.lease.reset();
        {
            view_probe fault(f.arena.get(), true);
            const auto failed = transition.activate({{device_domain,
                LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, f.arena.get()}});
            t.assert_true(failed.status == llama_memory_transition_status::activation_failed);
            t.assert_true(failed.failed_at == llama_memory_transition_state::committing);
        }
        if (!t.assert_true(transition.recover().status == llama_memory_transition_status::recovered)) return;
        t.assert_equal(original.capacity, session->binding_view().capacity);
        t.assert_true(session->layout_revision() > original.revision);
        t.assert_equal(generation, f.content->generation());
        t.assert_true(!session->failed());
        t.assert_true(generate_next(t, f, *session));
    });

    if (cuda) t.test("failed_binding_creation_recovers_old_pool_metadata", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 769, false, 4);
        block_workspace writer(f, 32768, 19), partial(f, f.host->layout().bytes, 29);
        auto session = make_session(f, writer, partial);
        if (!t.assert_true(bool(session)) || !t.assert_true(session->restore(513))) return;
        const auto original = session->binding_view();
        const auto original_policy = session->policy();
        binding_fault fault(writer.lease.get());
        llama_memory_transition transition({&fault, session.get()});
        auto target = target_for(f, minimum_pool(f), true);
        if (!t.assert_true(transition.prepare(target).status == llama_memory_transition_status::prepared)) return;
        f.lease.reset();
        const auto failed = transition.activate({{device_domain,
            LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, f.arena.get()}});
        t.assert_true(failed.status == llama_memory_transition_status::activation_failed);
        t.assert_true(failed.failed_at == llama_memory_transition_state::binding);
        t.assert_equal(size_t(1), failed.consumer);
        if (!t.assert_true(transition.recover().status == llama_memory_transition_status::recovered)) return;
        t.assert_equal(original.capacity, session->binding_view().capacity);
        t.assert_equal(original_policy.budget.pages, session->policy().budget.pages);
        t.assert_true(session->layout_revision() > original.revision);
        t.assert_true(!session->failed());
        t.assert_true(generate_next(t, f, *session));
    });

    if (cuda) t.test("later_activation_failure_reactivates_original_pool", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 769, false, 4);
        block_workspace writer(f, 32768, 19), partial(f, f.host->layout().bytes, 29);
        auto session = make_session(f, writer, partial);
        if (!t.assert_true(bool(session)) || !t.assert_true(session->restore(513))) return;
        const auto original = session->binding_view();
        activation_fault fault;
        llama_memory_transition transition({session.get(), &fault});
        auto target = target_for(f, minimum_pool(f), true);
        if (!t.assert_true(transition.prepare(target).status == llama_memory_transition_status::prepared)) return;
        f.lease.reset();
        const auto failed = transition.activate({{device_domain,
            LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, f.arena.get()}});
        t.assert_true(failed.status == llama_memory_transition_status::activation_failed);
        t.assert_true(failed.failed_at == llama_memory_transition_state::activating);
        t.assert_equal(size_t(1), failed.consumer);
        fault.fail = false;
        if (!t.assert_true(transition.recover().status == llama_memory_transition_status::recovered)) return;
        t.assert_equal(original.capacity, session->binding_view().capacity);
        t.assert_true(session->layout_revision() >= original.revision + 2);
        t.assert_true(generate_next(t, f, *session));
    });

    if (cuda) t.test("failed_recovery_poison_closes_session", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 769, false, 4);
        block_workspace writer(f, 32768, 19), partial(f, f.host->layout().bytes, 29);
        auto session = make_session(f, writer, partial);
        if (!t.assert_true(bool(session)) || !t.assert_true(session->restore(513))) return;
        binding_fault fault(writer.lease.get());
        llama_memory_transition transition({&fault, session.get()});
        auto target = target_for(f, minimum_pool(f), true);
        if (!t.assert_true(transition.prepare(target).status == llama_memory_transition_status::prepared)) return;
        f.lease.reset();
        const auto failed = transition.activate({{device_domain,
            LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, f.arena.get()}});
        t.assert_true(failed.status == llama_memory_transition_status::activation_failed);
        t.assert_true(failed.failed_at == llama_memory_transition_state::binding);
        fault.fail_recovery = true;
        const auto recovery = transition.recover();
        t.assert_true(recovery.status == llama_memory_transition_status::recovery_failed);
        t.assert_true(recovery.failed_at == llama_memory_transition_state::binding);
        t.assert_true(session->failed());
        t.assert_true(!session->begin(514, 1, true));
    });

    if (cuda) t.test("invalid_resize_and_poisoned_session_are_rejected_before_release", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 769, false, 4);
        block_workspace writer(f, 32768, 19), partial(f, f.host->layout().bytes, 29);
        auto session = make_session(f, writer, partial);
        if (!t.assert_true(bool(session)) || !t.assert_true(session->restore(513))) return;
        const auto view = session->binding_view();
        llama_memory_transition transition({session.get()});
        auto invalid = target_for(f, minimum_pool(f) - 128, true);
        t.assert_true(transition.prepare(invalid).status == llama_memory_transition_status::consumer_failed);
        t.assert_equal(view.base, session->binding_view().base);
        t.assert_equal(view.revision, session->binding_view().revision);
        session->abort();
        auto valid = target_for(f, minimum_pool(f), true);
        t.assert_true(transition.prepare(valid).status == llama_memory_transition_status::consumer_failed);
        t.assert_true(session->failed());
    });

    return t.summary();
}

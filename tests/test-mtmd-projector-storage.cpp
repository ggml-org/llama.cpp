#include "../tools/mtmd/mtmd-projector-storage.h"
#include "../tools/mtmd/mtmd-helper.h"
#include "../tools/mtmd/mtmd-embeddings.h"
#include "../tools/mtmd/mtmd-workspace.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "testing.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <thread>
#include <atomic>

struct weight_buffer_probe {
    inline static weight_buffer_probe * active = nullptr;
    ggml_backend_buffer_t buffer;
    decltype(ggml_backend_buffer_i::free_buffer) original;
    size_t frees = 0;
    explicit weight_buffer_probe(ggml_backend_buffer_t buffer) : buffer(buffer),original(buffer->iface.free_buffer) {
        GGML_ASSERT(!active); active = this;
        buffer->iface.free_buffer = [](ggml_backend_buffer_t buffer) {
            ++active->frees;
            if (active->original) active->original(buffer);
        };
    }
    ~weight_buffer_probe() { if (!frees) buffer->iface.free_buffer = original; active = nullptr; }
};

struct weight_allocation_fault {
    ggml_backend_buffer_type_t type = ggml_backend_cpu_buffer_type();
    decltype(ggml_backend_buffer_type_i::alloc_buffer) original = type->iface.alloc_buffer;
    weight_allocation_fault() { type->iface.alloc_buffer = [](ggml_backend_buffer_type_t,size_t) -> ggml_backend_buffer_t { return nullptr; }; }
    ~weight_allocation_fault() { type->iface.alloc_buffer = original; }
};

// Keep fake backing arrays allocated so every new resident buffer has a different address.
struct changing_weight_allocator {
    inline static changing_weight_allocator * active = nullptr;
    ggml_backend_buffer_type type = *ggml_backend_cpu_buffer_type();
    std::vector<std::vector<uint8_t>> backing;
    changing_weight_allocator() {
        GGML_ASSERT(!active); active = this;
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t type,size_t bytes) {
            const size_t alignment = ggml_backend_buft_get_alignment(type);
            active->backing.emplace_back(bytes+alignment);
            const auto begin = uintptr_t(active->backing.back().data());
            auto * buffer = ggml_backend_cpu_buffer_from_ptr(reinterpret_cast<void *>((begin+alignment-1)/alignment*alignment),bytes);
            buffer->buft = type;
            return buffer;
        };
    }
    ~changing_weight_allocator() { active = nullptr; }
};

struct source_fixture {
    ggml_context_ptr tensors{ggml_init({65536,nullptr,false})};
    gguf_context_ptr gguf{gguf_init_empty()};
    ggml_tensor * a = ggml_new_tensor_1d(tensors.get(),GGML_TYPE_F32,32);
    ggml_tensor * b = ggml_new_tensor_1d(tensors.get(),GGML_TYPE_F16,32);
    size_t data_offset = 0;
    source_fixture() {
        ggml_set_name(a,"v.first"); ggml_set_name(b,"a.second");
        std::memset(a->data,0x3c,ggml_nbytes(a)); std::memset(b->data,0x5a,ggml_nbytes(b));
        gguf_add_tensor(gguf.get(),a); gguf_add_tensor(gguf.get(),b);
    }
    std::shared_ptr<const mtmd_projector_source> source(bool truncated = false) {
        FILE * file = std::tmpfile(); GGML_ASSERT(file);
        GGML_ASSERT(gguf_write_to_file_ptr(gguf.get(),file,truncated));
        std::rewind(file);
        gguf_context_ptr parsed(gguf_init_from_file_ptr(file,{true,nullptr})); GGML_ASSERT(parsed);
        data_offset = gguf_get_data_offset(parsed.get());
        return mtmd_projector_source::from_file(file,parsed.get());
    }
    ggml_context_ptr context(bool both = true) {
        ggml_context_ptr result(ggml_init({65536,nullptr,true}));
        ggml_set_name(ggml_dup_tensor(result.get(),a),a->name);
        if (both) ggml_set_name(ggml_dup_tensor(result.get(),b),b->name);
        return result;
    }
};

int main(int argc,char ** argv) {
    const char * model_path = nullptr, * projector_path = nullptr, * baseline = nullptr, * save = nullptr;
    bool cuda = false;
    bool phase_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i],"--cuda")) cuda = true;
        else if (!std::strcmp(argv[i],"--phase-only")) phase_only = true;
        else if (!std::strcmp(argv[i],"--model") && i+1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i],"--mmproj") && i+1 < argc) projector_path = argv[++i];
        else if (!std::strcmp(argv[i],"--baseline") && i+1 < argc) baseline = argv[++i];
        else if (!std::strcmp(argv[i],"--save-baseline") && i+1 < argc) save = argv[++i];
        else return 2;
    }
    testing t;
    if (phase_only) {
        if (!model_path || !projector_path) return 2;
        t.set_filter("real_deferred_projector_uses_disjoint_weight_and_compute_grants");
    }
    t.test("source_manifest_and_bytes_outlive_the_loader_metadata", [](testing & t) {
        std::shared_ptr<const mtmd_projector_source> source;
        {
            source_fixture f; source = f.source();
            if (!t.assert_true(bool(source))) return;
            const auto * a = source->find("v.first");
            if (!t.assert_true(a != nullptr)) return;
            t.assert_equal(size_t(128),a->bytes);
            t.assert_true(a->type == GGML_TYPE_F32 && a->shape[0] == 32);
            t.assert_equal(f.data_offset + gguf_get_tensor_offset(f.gguf.get(),0),a->offset);
        }
        std::vector<uint8_t> output(128,0);
        t.assert_true(source->read("v.first",output.data(),output.size()));
        t.assert_true(std::all_of(output.begin(),output.end(),[](uint8_t b) { return b == 0x3c; }));
        t.assert_true(!source->read("missing",output.data(),output.size()));
        t.assert_true(!source->read("v.first",output.data(),127));
        t.assert_true(!source->read(nullptr,output.data(),128));
        t.assert_true(!source->read("v.first",nullptr,128));
        t.assert_true(!source->find(nullptr));
        t.assert_true(std::all_of(output.begin(),output.end(),[](uint8_t b) { return b == 0x3c; }));
    });
    t.test("metadata_rejects_wrong_shapes_names_types_and_bound_descriptors", [](testing & t) {
        source_fixture f; auto source = f.source();
        if (!t.assert_true(bool(source))) return;
        auto changed = f.context(); ggml_get_first_tensor(changed.get())->ne[0] = 16;
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        changed = f.context(); ggml_set_name(ggml_get_first_tensor(changed.get()),"missing");
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        changed = f.context(); ggml_get_first_tensor(changed.get())->type = GGML_TYPE_F16;
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        changed = f.context(); ggml_get_first_tensor(changed.get())->data = f.a->data;
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        changed = f.context(); ggml_set_name(ggml_get_next_tensor(changed.get(),ggml_get_first_tensor(changed.get())),"v.first");
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        t.assert_true(!mtmd_projector_metadata::create({},source));
        t.assert_true(!mtmd_projector_metadata::create(f.context(),{}));
        t.assert_true(bool(mtmd_projector_metadata::create(f.context(false),source)));
    });
    t.test("shared_weights_release_once_and_metadata_remains_unbound", [](testing & t) {
        source_fixture f;
        auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        if (!t.assert_true(bool(metadata))) return;
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type());
        if (!t.assert_true(bool(resident))) return;
        auto * a = ggml_get_first_tensor(metadata->context());
        t.assert_true(a->buffer && a->data);
        std::vector<uint8_t> bytes(128); ggml_backend_tensor_get(a,bytes.data(),0,bytes.size());
        t.assert_true(std::all_of(bytes.begin(),bytes.end(),[](uint8_t b) { return b == 0x3c; }));
        t.assert_true(ggml_backend_buffer_get_usage(resident->buffer()) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto another = resident;
        weight_buffer_probe probe(resident->buffer());
        t.assert_true(!mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type()));
        std::weak_ptr<mtmd_projector_weights> weak = resident;
        resident.reset(); t.assert_true(!weak.expired() && a->data != nullptr);
        another.reset(); t.assert_true(weak.expired());
        t.assert_equal(size_t(1),probe.frees);
        for (auto * tensor = a; tensor; tensor = ggml_get_next_tensor(metadata->context(),tensor))
            t.assert_true(!tensor->buffer && !tensor->data && !tensor->extra);
        t.assert_true(metadata->source()->read("v.first",bytes.data(),bytes.size()));
    });
    t.test("failed_and_cancelled_loading_returns_candidate_bindings", [](testing & t) {
        source_fixture f;
        auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        if (!t.assert_true(bool(metadata))) return;
        t.assert_true(!mtmd_projector_weights::allocate(metadata,nullptr));
        {
            weight_allocation_fault fault;
            t.assert_true(!mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type()));
        }
        auto cancel = [](float,void *) { return false; };
        t.assert_true(!mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),false,cancel));
        size_t callbacks = 0;
        auto late = [](float,void * p) { return ++*static_cast<size_t *>(p) == 1; };
        t.assert_true(!mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),false,late,&callbacks));
        t.assert_equal(size_t(2),callbacks);
        auto throwing = [](float progress,void *) { if (progress > 0) throw std::runtime_error("cancel"); return true; };
        try { mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),false,throwing); t.assert_true(false); }
        catch (const std::runtime_error &) { t.assert_true(true); }
        for (auto * tensor = ggml_get_first_tensor(metadata->context()); tensor; tensor = ggml_get_next_tensor(metadata->context(),tensor))
            t.assert_true(!tensor->buffer && !tensor->data && !tensor->extra);
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type());
        t.assert_true(bool(resident));
        auto truncated = mtmd_projector_metadata::create(f.context(),f.source(true));
        t.assert_true(truncated && !mtmd_projector_weights::allocate(truncated,ggml_backend_cpu_buffer_type()));
        t.assert_true(!ggml_get_first_tensor(truncated->context())->data);
    });
    t.test("loading_callbacks_cannot_reenter_the_same_metadata", [](testing & t) {
        source_fixture f; auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        if (!t.assert_true(bool(metadata))) return;
        struct probe { std::shared_ptr<mtmd_projector_metadata> metadata; size_t calls = 0; bool admitted = false; } p{metadata};
        auto callback = [](float,void * data) {
            auto & p = *static_cast<probe *>(data); ++p.calls;
            p.admitted |= bool(mtmd_projector_weights::allocate(p.metadata,ggml_backend_cpu_buffer_type()));
            return true;
        };
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),false,callback,&p);
        t.assert_true(resident && !p.admitted && p.calls == 3);
    });
    t.test("temporary_metadata_and_skipped_upload_keep_lifetime_boundaries", [](testing & t) {
        source_fixture f;
        auto cancel = [](float p,void *) { return p == 0; };
        t.assert_true(!mtmd_projector_weights::allocate(mtmd_projector_metadata::create(f.context(),f.source()),
            ggml_backend_cpu_buffer_type(),false,cancel));
        auto metadata = mtmd_projector_metadata::create(f.context(),f.source(true));
        if (!t.assert_true(bool(metadata))) return;
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),true);
        if (!t.assert_true(bool(resident))) return;
        auto weak = std::weak_ptr<mtmd_projector_metadata>(metadata);
        metadata.reset();
        t.assert_true(!weak.expired());
        resident.reset();
        t.assert_true(weak.expired());
    });
    t.test("shared_source_reads_do_not_race_the_file_cursor", [](testing & t) {
        source_fixture f; auto source = f.source();
        if (!t.assert_true(bool(source))) return;
        std::atomic<bool> ok{true};
        auto read = [&](const char * name,size_t bytes,uint8_t value) {
            std::vector<uint8_t> data(bytes);
            for (int i = 0; i < 64; ++i) {
                if (!source->read(name,data.data(),bytes) ||
                        !std::all_of(data.begin(),data.end(),[&](uint8_t b) { return b == value; })) ok = false;
            }
        };
        std::thread first(read,"v.first",128,0x3c),second(read,"a.second",64,0x5a);
        first.join(); second.join();
        t.assert_true(ok.load());
    });
    t.test("invalid_source_handles_and_missing_paths_are_rejected", [](testing & t) {
        source_fixture f;
        t.assert_true(!mtmd_projector_source::from_file(nullptr,f.gguf.get()));
        t.assert_true(!mtmd_projector_source::from_file(std::tmpfile(),nullptr));
        t.assert_true(!mtmd_projector_source::open(nullptr,f.gguf.get()));
        t.assert_true(!mtmd_projector_source::open("",f.gguf.get()));
        t.assert_true(!mtmd_projector_source::from_file(std::tmpfile(),f.gguf.get()));
    });
    t.test("residency_drains_and_retires_before_unbinding_and_rejects_stale_generations", [](testing & t) {
        source_fixture f;
        auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        auto * tensor = ggml_get_first_tensor(metadata->context());
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type());
        std::unique_ptr<mtmd_projector_residency> owner;
        std::vector<std::string> order;
        bool reentered = false;
        auto drain = [&] {
            order.push_back("drain");
            t.assert_true(tensor->data && tensor->buffer && !owner->ready());
            reentered |= owner->begin(owner->generation()) || owner->unload() || owner->reload() || bool(owner->acquire());
            return true;
        };
        auto invalidate = [&] { order.push_back("invalidate"); t.assert_true(tensor->data != nullptr); return true; };
        owner = mtmd_projector_residency::create(std::move(resident),ggml_backend_cpu_buffer_type(),{drain,invalidate});
        if (!t.assert_true(bool(owner))) return;
        const auto first = owner->generation();
        auto reader = owner->acquire();
        t.assert_true(!owner->unload() && owner->ready() && order.empty());
        reader.reset();
        if (!t.assert_true(owner->begin(first))) return;
        t.assert_true(!owner->begin(first) && !owner->unload());
        owner->end();
        if (!t.assert_true(owner->unload())) return;
        t.assert_true(!reentered && order == std::vector<std::string>({"drain","invalidate"}));
        t.assert_true(!tensor->data && !tensor->buffer && !tensor->extra && !owner->ready());
        t.assert_true(!owner->begin(first) && !owner->acquire());
        t.assert_true(owner->unload() && order.size() == 2);
        t.assert_true(owner->reload() && owner->ready() && owner->generation() > first);
        t.assert_true(!owner->begin(first));
        t.assert_true(owner->begin(owner->generation())); owner->end();
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
        t.assert_true(std::all_of(bytes.begin(),bytes.end(),[](uint8_t b) { return b == 0x3c; }));
        // Drain callbacks inspect the owner; finish teardown while that pointer is still assigned.
        t.assert_true(owner->unload());
    });
    t.test("failed_retirement_and_reload_keep_admission_closed_until_retry", [](testing & t) {
        source_fixture f;
        auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type());
        bool drain_ok = false, retire_ok = false;
        auto owner = mtmd_projector_residency::create(std::move(resident),ggml_backend_cpu_buffer_type(),
            {[&] { return drain_ok; },[&] { return retire_ok; }});
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(!owner->unload() && !owner->ready());
        t.assert_true(!owner->reload());
        drain_ok = true;
        t.assert_true(!owner->unload() && !owner->ready());
        retire_ok = true;
        t.assert_true(owner->unload());
        {
            weight_allocation_fault fault;
            t.assert_true(!owner->reload() && !owner->ready());
        }
        auto cancel = [](float,void *) { return false; };
        t.assert_true(!owner->reload(cancel));
        auto throwing = [](float,void *) -> bool { throw std::runtime_error("cancel"); };
        t.assert_true(!owner->reload(throwing));
        t.assert_true(!ggml_get_first_tensor(metadata->context())->data);
        t.assert_true(owner->reload() && owner->ready());
        t.assert_true(owner->unload());
    });
    t.test("repeated_reload_moves_bindings_without_replacing_tensor_metadata", [](testing & t) {
        changing_weight_allocator allocator;
        source_fixture f; auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        auto * tensor = ggml_get_first_tensor(metadata->context());
        auto owner = mtmd_projector_residency::create(mtmd_projector_weights::allocate(metadata,&allocator.type),
            &allocator.type,{[] { return true; },[] { return true; }});
        if (!t.assert_true(bool(owner))) return;
        for (int i = 0; i < 4; ++i) {
            const auto address = uintptr_t(tensor->data), generation = owner->generation();
            t.assert_true(owner->begin(generation)); owner->end();
            if (!t.assert_true(owner->unload() && owner->reload())) return;
            t.assert_true(uintptr_t(tensor->data) != address && ggml_get_first_tensor(metadata->context()) == tensor);
            t.assert_true(!owner->begin(generation));
            std::vector<uint8_t> bytes(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
            t.assert_true(std::all_of(bytes.begin(),bytes.end(),[](uint8_t b) { return b == 0x3c; }));
        }
        t.assert_true(owner->unload());
    });
    t.test("a_reader_retained_during_retirement_keeps_weights_bound_and_admission_closed", [](testing & t) {
        source_fixture f; auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type());
        std::weak_ptr<mtmd_projector_weights> weak = resident;
        std::shared_ptr<mtmd_projector_weights> reader;
        auto owner = mtmd_projector_residency::create(std::move(resident),ggml_backend_cpu_buffer_type(),
            {[] { return true; },[&] { reader = weak.lock(); return true; }});
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(!owner->unload() && reader && !owner->ready());
        t.assert_true(ggml_get_first_tensor(metadata->context())->data != nullptr);
        reader.reset();
        t.assert_true(owner->unload() && !ggml_get_first_tensor(metadata->context())->data);
    });
    t.test("measurement_only_weights_cannot_enter_the_live_residency_lifecycle", [](testing & t) {
        source_fixture f; auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        auto weights = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),true);
        t.assert_true(bool(weights));
        t.assert_true(!mtmd_projector_residency::create(std::move(weights),ggml_backend_cpu_buffer_type(),
            {[] { return true; },[] { return true; }}));
    });
    t.test("leased_weights_use_bounded_storage_without_another_backend_allocation", [](testing & t) {
        source_fixture f; auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        using arena_ptr = std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
        using lease_ptr = std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
        arena_ptr arena(ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(),4096),ggml_backend_memory_arena_free);
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(),0));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(),1,alignment,1024,alignment,0,nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(),2,2048,alignment,alignment,0,nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(),1),ggml_backend_memory_lease_free);
        lease_ptr tiny(ggml_backend_memory_arena_acquire(arena.get(),2),ggml_backend_memory_lease_free);
        auto owner = mtmd_projector_residency::create_unloaded(metadata,ggml_backend_cpu_buffer_type(),
            {[] { return true; },[] { return true; }});
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(!owner->ready() && !owner->reload_in(tiny.get()));
        {
            weight_allocation_fault no_extra_allocation;
            t.assert_true(owner->reload_in(lease.get()) && owner->ready());
        }
        auto * tensor = ggml_get_first_tensor(metadata->context());
        const auto begin = uintptr_t(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(lease.get())));
        t.assert_true(uintptr_t(tensor->data) >= begin && uintptr_t(tensor->data)+ggml_nbytes(tensor) <= begin+1024);
        auto reader = owner->acquire();
        lease.reset(); tiny.reset();
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(arena.get()));
        t.assert_true(!owner->unload());
        reader.reset();
        if (!t.assert_true(owner->unload())) return;
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(arena.get()));
        t.assert_true(!tensor->data && metadata->source() != nullptr);
    });
    if (model_path && projector_path) t.test("real_projector_encoding_matches_the_original_eager_loader", [&](testing & t) {
        ggml_backend_load_all();
        auto model_params = llama_model_default_params();
        model_params.no_alloc = true; model_params.n_gpu_layers = 0;
        model_params.load_mode = LLAMA_LOAD_MODE_NONE; model_params.use_extra_bufts = false;
        llama_model_ptr model(llama_model_load_from_file(model_path,model_params));
        if (!t.assert_true(bool(model))) return;
        auto params = mtmd_context_params_default();
        struct encode_probe { mtmd_context * ctx = nullptr; mtmd_batch * batch = nullptr; bool reentered = false; size_t calls = 0; } probe;
        params.cb_eval_user_data = &probe;
        params.cb_eval = [](ggml_tensor *,bool ask,void * data) {
            auto & p = *static_cast<encode_probe *>(data);
            if (ask && p.ctx && !p.calls++) {
                p.reentered |= mtmd_unload_projector_weights(p.ctx) || mtmd_reload_projector_weights(p.ctx) ||
                    bool(mtmd_acquire_projector_weights(p.ctx)) || mtmd_batch_encode(p.batch) == 0;
            }
            return false;
        };
        params.use_gpu = cuda; params.warmup = false; params.n_threads = 4;
        params.image_min_tokens = 64; params.image_max_tokens = 256;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        mtmd::context_ptr projector(mtmd_init_from_file(projector_path,model.get(),params));
        if (!t.assert_true(bool(projector))) return;
        std::vector<uint8_t> pixels(256*256*3);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = uint8_t((i*7+i/101)%256);
        mtmd::bitmap_ptr image(mtmd_bitmap_init(256,256,pixels.data()));
        const auto * bitmap = image.get();
        const std::string prompt = mtmd_get_marker(projector.get());
        mtmd_input_text input{prompt.c_str(),prompt.size(),true,true};
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        if (!t.assert_equal(0,mtmd_tokenize(projector.get(),chunks.get(),&input,&bitmap,1))) return;
        mtmd::batch_ptr batch(mtmd_batch_init(projector.get()));
        probe.ctx = projector.get(); probe.batch = batch.get();
        const mtmd_input_chunk * chunk = nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto * entry = mtmd_input_chunks_get(chunks.get(),i);
            if (mtmd_input_chunk_get_type(entry) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                chunk = entry;
                if (!t.assert_equal(0,mtmd_batch_add_chunk(batch.get(),entry))) return;
            }
        }
        if (!t.assert_true(chunk != nullptr) || !t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
        t.assert_true(probe.calls > 0 && !probe.reentered);
        mtmd_embedding_view output;
        if (!t.assert_true(mtmd_batch_acquire_output_embd(batch.get(),chunk,output))) return;
        const size_t bytes = output.n_tokens()*output.n_embd()*sizeof(float);
        if (save) {
            std::ofstream file(save,std::ios::binary);
            file.write(reinterpret_cast<const char *>(output.data()),std::streamsize(bytes));
            t.assert_true(bool(file));
        }
        if (baseline) {
            std::vector<uint8_t> expected(bytes);
            std::ifstream file(baseline,std::ios::binary);
            file.read(reinterpret_cast<char *>(expected.data()),std::streamsize(bytes));
            if (!t.assert_true(bool(file) && file.peek() == EOF)) return;
            t.assert_true(std::memcmp(expected.data(),output.data(),bytes) == 0);
        }
        if (!save) {
            auto weights = mtmd_acquire_projector_weights(projector.get());
            if (!t.assert_true(bool(weights))) return;
            auto metadata = weights->metadata();
            auto * tensor = ggml_get_first_tensor(metadata->context());
            t.assert_true(!mtmd_unload_projector_weights(projector.get()));
            weights.reset();
            const std::vector<float> embeddings(output.data(),output.data()+bytes/sizeof(float));
            for (int round = 0; round < 3; ++round) {
                for (int warm = 0; warm < 2; ++warm) if (!t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
                if (!t.assert_true(mtmd_unload_projector_weights(projector.get()))) return;
                t.assert_true(!tensor->data && !tensor->buffer && !tensor->extra);
                t.assert_true(!mtmd_acquire_projector_weights(projector.get()));
                t.assert_true(mtmd_batch_encode(batch.get()) != 0);
                std::vector<ggml_backend_memory_workspace_group> groups;
                t.assert_true(!mtmd_batch_measure_compute_workspace(batch.get(),groups));
                t.assert_true(std::equal(embeddings.begin(),embeddings.end(),output.data()));
                struct reload_probe { mtmd_context * ctx; mtmd_batch * batch; bool reentered = false; size_t calls = 0; } p{projector.get(),batch.get()};
                auto callback = [](float,void * data) {
                    auto & p = *static_cast<reload_probe *>(data); ++p.calls;
                    p.reentered |= mtmd_unload_projector_weights(p.ctx) || mtmd_reload_projector_weights(p.ctx) ||
                        bool(mtmd_acquire_projector_weights(p.ctx)) || mtmd_batch_encode(p.batch) == 0;
                    return true;
                };
                if (!t.assert_true(mtmd_reload_projector_weights(projector.get(),callback,&p))) return;
                t.assert_true(p.calls > 1 && !p.reentered && tensor->data && tensor->buffer);
                if (!t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
                mtmd_embedding_view restored;
                if (!t.assert_true(mtmd_batch_acquire_output_embd(batch.get(),chunk,restored))) return;
                t.assert_true(std::memcmp(restored.data(),embeddings.data(),bytes) == 0);
            }
            weights = mtmd_acquire_projector_weights(projector.get());
            const size_t sample_bytes = std::min(size_t(64),ggml_nbytes(tensor));
            std::vector<uint8_t> expected(sample_bytes),actual(sample_bytes);
            ggml_backend_tensor_get(tensor,expected.data(),0,sample_bytes);
            batch.reset(); projector.reset();
            // The resident owner outlives its projector scheduler and backend handle.
            t.assert_true(tensor->data && tensor->buffer);
            ggml_backend_tensor_get(tensor,actual.data(),0,sample_bytes);
            t.assert_true(actual == expected);
            weights.reset();
            t.assert_true(!tensor->buffer && !tensor->data && !tensor->extra);
            std::vector<uint8_t> file_bytes(ggml_nbytes(tensor));
            t.assert_true(metadata->source()->read(tensor->name,file_bytes.data(),file_bytes.size()));
            t.assert_true(std::equal(expected.begin(),expected.end(),file_bytes.begin()));
            t.assert_true(output.data() != nullptr);
        }
    });
    if (phase_only) t.test("real_deferred_projector_uses_disjoint_weight_and_compute_grants", [&](testing & t) {
        ggml_backend_load_all();
        auto mp = llama_model_default_params(); mp.no_alloc = true; mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_NONE; mp.use_extra_bufts = false;
        llama_model_ptr model(llama_model_load_from_file(model_path,mp));
        if (!t.assert_true(bool(model))) return;
        auto params = mtmd_context_params_default(); params.use_gpu = cuda; params.warmup = true;
        params.image_min_tokens = 64; params.image_max_tokens = 256;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        mtmd::context_ptr projector;
        {
            weight_allocation_fault no_eager_host_buffer;
            projector.reset(mtmd_init_from_file_deferred(projector_path,model.get(),params));
        }
        if (!t.assert_true(bool(projector) && !mtmd_acquire_projector_weights(projector.get()))) return;
        auto * type = ggml_backend_cpu_buffer_type();
        if (cuda) {
            auto * dev = ggml_backend_dev_by_name("CUDA0");
            using factory_t = ggml_backend_buffer_type_t (*)(int);
            auto factory = reinterpret_cast<factory_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                "ggml_backend_cuda_device_buffer_type"));
            if (!t.assert_true(factory != nullptr)) return;
            type = factory(0);
        }
        using arena_ptr = std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
        using lease_ptr = std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
        arena_ptr arena(ggml_backend_memory_arena_new(type,1024*1048576),ggml_backend_memory_arena_free);
        if (!t.assert_true(bool(arena))) return;
        std::vector<uint8_t> pixels(256*256*3);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = uint8_t((i*7+i/101)%256);
        mtmd::bitmap_ptr image(mtmd_bitmap_init(256,256,pixels.data())); const auto * bitmap = image.get();
        const std::string prompt = mtmd_get_marker(projector.get());
        mtmd_input_text input{prompt.c_str(),prompt.size(),true,true};
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        if (!t.assert_equal(0,mtmd_tokenize(projector.get(),chunks.get(),&input,&bitmap,1))) return;
        mtmd::batch_ptr batch(mtmd_batch_init(projector.get())); const mtmd_input_chunk * chunk = nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto * entry = mtmd_input_chunks_get(chunks.get(),i);
            if (mtmd_input_chunk_get_type(entry) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                chunk = entry; if (!t.assert_equal(0,mtmd_batch_add_chunk(batch.get(),entry))) return;
            }
        }
        mtmd_vision_phase_requirements plan;
        auto * parent = ggml_backend_memory_arena_parent(arena.get());
        ggml_backend_buffer_clear(parent,0x6d);
        ggml_context_ptr canary_context(ggml_init({65536,nullptr,true}));
        auto * canary = ggml_new_tensor_1d(canary_context.get(),GGML_TYPE_F32,16);
        GGML_ASSERT(ggml_backend_tensor_alloc(parent,canary,ggml_backend_buffer_get_base(parent)) == GGML_STATUS_SUCCESS);
        std::vector<uint8_t> canary_before(64),canary_after(64);
        ggml_backend_tensor_get(canary,canary_before.data(),0,64);
        const auto usage = ggml_backend_buffer_get_usage(parent);
        if (!t.assert_true(mtmd_batch_measure_vision_phase(batch.get(),parent,plan))) return;
        ggml_backend_tensor_get(canary,canary_after.data(),0,64);
        t.assert_true(canary_before == canary_after && ggml_backend_buffer_get_usage(parent) == usage);
        t.assert_true(plan.weight_bytes+plan.device_compute_bytes <= 1024*1048576);
        t.assert_true(plan.groups.size() == 1 || (plan.groups.size() == 2 && plan.host_compute_bytes > 0));
        auto tiny = ggml_backend_buffer_ptr(ggml_backend_buffer_view(ggml_backend_memory_arena_parent(arena.get()),0,1048576));
        mtmd_vision_phase_requirements unchanged; unchanged.weight_bytes = 123;
        t.assert_true(!mtmd_batch_measure_vision_phase(batch.get(),tiny.get(),unchanged) && unchanged.weight_bytes == 123);
        auto weights_only = ggml_backend_buffer_ptr(ggml_backend_buffer_view(parent,0,plan.weight_bytes));
        t.assert_true(!mtmd_batch_measure_vision_phase(batch.get(),weights_only.get(),unchanged) && unchanged.weight_bytes == 123);
        ggml_backend_tensor_get(canary,canary_after.data(),0,64);
        t.assert_true(canary_after == canary_before && !mtmd_acquire_projector_weights(projector.get()));
        const size_t alignment = ggml_backend_buft_get_alignment(type);
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(),0));
        GGML_ASSERT(ggml_backend_memory_arena_reserve(arena.get(),91,plan.weight_bytes,alignment,0,nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_reserve(arena.get(),92,plan.device_compute_bytes,alignment,0,nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease_ptr weights(ggml_backend_memory_arena_acquire(arena.get(),91),ggml_backend_memory_lease_free);
        lease_ptr device_compute(ggml_backend_memory_arena_acquire(arena.get(),92),ggml_backend_memory_lease_free);
        if (!t.assert_true(mtmd_reload_projector_weights_in(projector.get(),weights.get()))) return;
        std::vector<ggml_backend_memory_lease_t> compute;
        std::vector<arena_ptr> host; std::vector<lease_ptr> host_leases;
        for (size_t i = 0; i < plan.groups.size(); ++i) {
            const auto & group = plan.groups[i];
            if (group.buft == type) compute.push_back(device_compute.get());
            else {
                host.emplace_back(ggml_backend_memory_arena_new(group.buft,group.size),ggml_backend_memory_arena_free);
                GGML_ASSERT(ggml_backend_memory_arena_begin(host.back().get(),0));
                GGML_ASSERT(ggml_backend_memory_arena_reserve(host.back().get(),i+1,group.size,group.alignment,0,nullptr));
                GGML_ASSERT(ggml_backend_memory_arena_commit(host.back().get()));
                host_leases.emplace_back(ggml_backend_memory_arena_acquire(host.back().get(),i+1),ggml_backend_memory_lease_free);
                compute.push_back(host_leases.back().get());
            }
        }
        if (!t.assert_true(mtmd_attach_compute_workspace(projector.get(),compute)) || !t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
        mtmd_embedding_view output;
        if (!t.assert_true(mtmd_batch_acquire_output_embd(batch.get(),chunk,output))) return;
        const size_t bytes = output.n_tokens()*output.n_embd()*sizeof(float);
        if (baseline) {
            std::vector<uint8_t> expected(bytes); std::ifstream file(baseline,std::ios::binary);
            file.read(reinterpret_cast<char *>(expected.data()),std::streamsize(bytes));
            if (!t.assert_true(bool(file) && file.peek() == EOF)) return;
            t.assert_true(std::memcmp(expected.data(),output.data(),bytes) == 0);
        }
        auto reader = mtmd_acquire_projector_weights(projector.get());
        if (!t.assert_true(bool(reader))) return;
        auto metadata = reader->metadata(); auto * tensor = ggml_get_first_tensor(metadata->context());
        const size_t sample = std::min(size_t(64),ggml_nbytes(tensor));
        std::vector<uint8_t> before(sample),after(sample); ggml_backend_tensor_get(tensor,before.data(),0,sample);
        batch.reset(); projector.reset(); weights.reset(); device_compute.reset(); host_leases.clear(); host.clear();
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(arena.get()));
        tiny.reset(); weights_only.reset();
        arena.reset();
        ggml_backend_tensor_get(tensor,after.data(),0,sample);
        t.assert_true(after == before && output.data() != nullptr);
        reader.reset(); t.assert_true(!tensor->data && !tensor->buffer);
    });
    return t.summary();
}

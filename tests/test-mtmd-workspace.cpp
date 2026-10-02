#include "../tools/mtmd/mtmd-workspace.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "testing.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct workspace_graph {
    ggml_context_ptr ctx{ggml_init({1024*1024,nullptr,true})};
    ggml_tensor * input;
    ggml_tensor * output;
    ggml_cgraph * graph;
    explicit workspace_graph(size_t rows) {
        input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,64,rows);
        ggml_set_input(input);
        output=ggml_scale(ctx.get(),input,2);
        ggml_set_output(output);
        graph=ggml_new_graph_custom(ctx.get(),64,false);
        ggml_build_forward_expand(graph,output);
    }
};

struct workspace_grant {
    arena_ptr arena{nullptr,ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr,ggml_backend_memory_lease_free};
    workspace_grant(ggml_backend_buffer_type_t type,size_t bytes,uint64_t id=11) {
        const size_t alignment=ggml_backend_buft_get_alignment(type);
        arena.reset(ggml_backend_memory_arena_new(type,bytes+2*alignment));
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena.get(),0));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(),id,alignment,bytes,alignment,0,nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(),id));
        GGML_ASSERT(lease);
    }
};

struct workspace_fixture {
    ggml_backend_ptr first{ggml_backend_cpu_init()}, second{ggml_backend_cpu_init()};
    std::vector<ggml_backend_t> backends{first.get(),second.get()};
    std::vector<ggml_backend_buffer_type_t> types{ggml_backend_cpu_buffer_type(),ggml_backend_cpu_buffer_type()};
    ggml_backend_sched_ptr sched{ggml_backend_sched_new(backends.data(),types.data(),2,256,false,true)};
    mtmd_compute_workspace workspace{sched.get(),backends,256};
};

struct synchronization_probe {
    inline static synchronization_probe * active=nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::synchronize) original;
    ggml_backend_memory_arena_t arena;
    mtmd_compute_workspace & owner;
    std::vector<size_t> leases;
    bool reentrant_release=false;
    synchronization_probe(ggml_backend_t backend,ggml_backend_memory_arena_t arena,mtmd_compute_workspace & owner) :
        backend(backend),original(backend->iface.synchronize),arena(arena),owner(owner) {
        GGML_ASSERT(!active); active=this;
        backend->iface.synchronize=[](ggml_backend_t backend) {
            active->leases.push_back(ggml_backend_memory_arena_lease_count(active->arena));
            active->reentrant_release |= active->owner.release();
            if (active->original) active->original(backend);
        };
    }
    ~synchronization_probe() { backend->iface.synchronize=original; active=nullptr; }
};

static void verify_graph(testing & t, mtmd_compute_workspace & owner, workspace_graph & g) {
    const size_t elements=ggml_nelements(g.input);
    std::vector<float> input(elements,3),output(elements,0);
    ggml_backend_tensor_set(g.input,input.data(),0,elements*sizeof(float));
    if (!t.assert_equal(GGML_STATUS_SUCCESS,owner.compute_async(g.graph)) || !t.assert_true(owner.drain())) return;
    ggml_backend_tensor_get(g.output,output.data(),0,elements*sizeof(float));
    t.assert_true(std::all_of(output.begin(),output.end(),[](float value) { return value == 6; }));
}

int main(int argc,char ** argv) {
    const char * model_path=nullptr;
    const char * projector_path=nullptr;
    bool cuda=false;
    for (int i=1;i<argc;++i) {
        if (!std::strcmp(argv[i],"--cuda")) cuda=true;
        else if (!std::strcmp(argv[i],"--model") && i+1<argc) model_path=argv[++i];
        else if (!std::strcmp(argv[i],"--mmproj") && i+1<argc) projector_path=argv[++i];
        else return 2;
    }
    testing t;
    t.test("measures_actual_batch_without_allocating_and_normalizes_aliases", [](testing & t) {
        workspace_fixture f;
        if (!t.assert_true(f.workspace.supported())) return;
        std::vector<ggml_backend_memory_workspace_group> small,large;
        workspace_graph a(8),b(64);
        if (!t.assert_true(f.workspace.measure(a.graph,small)) || !t.assert_true(f.workspace.measure(b.graph,large))) return;
        t.assert_equal(size_t(1),small.size());
        t.assert_equal(size_t(1),large.size());
        t.assert_true(large[0].size > small[0].size);
        t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(f.sched.get(),f.second.get()));
    });
    t.test("borrowed_graph_executes_and_release_returns_the_lease", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(16);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}))) return;
        grant.lease.reset();
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        if (!t.assert_true(f.workspace.alloc_graph(g.graph))) return;
        verify_graph(t,f.workspace,g);
        t.assert_true(f.workspace.release());
        t.assert_true(!f.workspace.ready());
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        t.assert_true(f.workspace.release());
    });
    t.test("measurement_restores_graph_metadata_after_backend_optimization", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(16);
        const auto before=*g.output;
        f.first->iface.graph_optimize=[](ggml_backend_t,ggml_cgraph * graph) {
            ggml_graph_node(graph,ggml_graph_n_nodes(graph)-1)->op_params[0]=0;
        };
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        t.assert_equal(0,std::memcmp(&before,g.output,sizeof(before)));
        f.first->iface.graph_optimize=nullptr;
        workspace_grant grant(groups[0].buft,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}) && f.workspace.alloc_graph(g.graph))) return;
        verify_graph(t,f.workspace,g);
    });
    t.test("invalid_grant_preserves_attachment_and_larger_graph_is_bounded", [](testing & t) {
        workspace_fixture f;
        workspace_graph small(8),large(1024);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(small.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size),tiny(groups[0].buft,groups[0].alignment);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}))) return;
        t.assert_true(!f.workspace.attach({tiny.lease.get()}));
        t.assert_true(!f.workspace.attach({nullptr}));
        t.assert_true(f.workspace.ready());
        t.assert_true(!f.workspace.alloc_graph(large.graph));
        t.assert_equal(groups[0].size,ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        if (!t.assert_true(f.workspace.alloc_graph(small.graph))) return;
        verify_graph(t,f.workspace,small);
    });
    t.test("rejects_invalid_graphs_and_requires_explicit_release_before_measure", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(8);
        std::vector<ggml_backend_memory_workspace_group> groups;
        t.assert_true(!f.workspace.measure(nullptr,groups));
        t.assert_true(groups.empty());
        t.assert_true(!f.workspace.attach({}));
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}))) return;
        const auto before=groups[0].size;
        t.assert_true(!f.workspace.measure(g.graph,groups));
        t.assert_equal(before,groups[0].size);
        t.assert_true(f.workspace.release());
        t.assert_true(f.workspace.measure(g.graph,groups));
    });
    t.test("attachment_failure_returns_our_lease_and_preserves_foreign_owner", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(16);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant a(groups[0].buft,groups[0].size,11),foreign(groups[0].buft,groups[0].size,99);
        if (!t.assert_true(ggml_backend_sched_attach_memory_lease(f.sched.get(),f.first.get(),foreign.lease.get()))) return;
        t.assert_true(!f.workspace.attach({a.lease.get()}));
        t.assert_true(!f.workspace.ready());
        a.lease.reset(); foreign.lease.reset();
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(a.arena.get()));
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(foreign.arena.get()));
        t.assert_true(ggml_backend_sched_detach_memory_lease(f.sched.get(),f.first.get()));
    });
    t.test("release_drains_execution_before_returning_storage_and_rejects_reentry", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(32);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}) && f.workspace.alloc_graph(g.graph))) return;
        grant.lease.reset();
        const std::vector<float> input(ggml_nelements(g.input),1);
        ggml_backend_tensor_set(g.input,input.data(),0,ggml_nbytes(g.input));
        if (!t.assert_equal(GGML_STATUS_SUCCESS,f.workspace.compute_async(g.graph))) return;
        synchronization_probe probe(f.first.get(),grant.arena.get(),f.workspace);
        t.assert_true(f.workspace.release());
        if (!t.assert_true(!probe.leases.empty())) return;
        t.assert_equal(size_t(1),probe.leases.front());
        t.assert_true(!probe.reentrant_release);
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        t.assert_equal(GGML_STATUS_FAILED,f.workspace.compute_async(g.graph));
    });
    t.test("rejects_wrong_type_and_count_before_changing_current_grants", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(16);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size);
        auto distinct=*ggml_backend_cpu_buffer_type();
        workspace_grant wrong(&distinct,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}))) return;
        t.assert_true(!f.workspace.attach({wrong.lease.get()}));
        t.assert_true(!f.workspace.attach({grant.lease.get(),grant.lease.get()}));
        t.assert_true(f.workspace.ready());
        t.assert_true(f.workspace.alloc_graph(g.graph));
        verify_graph(t,f.workspace,g);
    });
    t.test("larger_grants_survive_graph_retirement_and_support_a_new_batch", [](testing & t) {
        workspace_fixture f;
        workspace_graph a(16),b(24);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(a.graph,groups))) return;
        workspace_grant grant(groups[0].buft,2*groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}) && f.workspace.alloc_graph(a.graph))) return;
        verify_graph(t,f.workspace,a);
        t.assert_true(f.workspace.retire_graph());
        t.assert_true(f.workspace.ready());
        t.assert_equal(GGML_STATUS_FAILED,f.workspace.compute_async(a.graph));
        if (!t.assert_true(f.workspace.alloc_graph(b.graph))) return;
        verify_graph(t,f.workspace,b);
    });
    t.test("invalid_scheduler_and_excess_graph_metadata_are_rejected", [](testing & t) {
        mtmd_compute_workspace invalid(nullptr,{},0);
        std::vector<ggml_backend_memory_workspace_group> output;
        workspace_graph g(8);
        t.assert_true(!invalid.supported());
        t.assert_true(!invalid.measure(g.graph,output));
        t.assert_true(invalid.release());
        workspace_fixture f;
        for (int i=0;i<260;++i) g.output=ggml_scale(g.ctx.get(),g.output,1.1f);
        g.graph=ggml_new_graph_custom(g.ctx.get(),512,false);
        ggml_build_forward_expand(g.graph,g.output);
        t.assert_true(!f.workspace.measure(g.graph,output));
        t.assert_true(output.empty());
    });
    if (model_path && projector_path) t.test("real_vision_batches_match_legacy_with_borrowed_workspace", [&](testing & t) {
        auto log=[](ggml_log_level level,const char * text,void *) {
            if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(text,stderr);
        };
        llama_log_set(log,nullptr);
        mtmd_log_set(log,nullptr);
        ggml_backend_load_all();
        auto model_params=llama_model_default_params();
        model_params.vocab_only=true;
        llama_model_ptr model(llama_model_load_from_file(model_path,model_params));
        if (!t.assert_true(bool(model))) return;
        auto params=mtmd_context_params_default();
        params.use_gpu=cuda; params.warmup=false; params.n_threads=2;
        params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.image_min_tokens=64; params.image_max_tokens=4096;
        mtmd::context_ptr stock(mtmd_init_from_file(projector_path,model.get(),params));
        mtmd::context_ptr borrowed(mtmd_init_from_file(projector_path,model.get(),params));
        if (!t.assert_true(bool(stock) && bool(borrowed))) return;
        auto * device=cuda ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
        auto * compute_type=ggml_backend_cpu_buffer_type();
        if (cuda) {
            auto * reg=device ? ggml_backend_dev_backend_reg(device) : nullptr;
            using factory_t=ggml_backend_buffer_type_t (*)(int);
            auto factory=reg ? reinterpret_cast<factory_t>(ggml_backend_reg_get_proc_address(reg,
                "ggml_backend_cuda_device_buffer_type")) : nullptr;
            compute_type=nullptr;
            if (factory) for (size_t i=0;i<ggml_backend_reg_dev_count(reg);++i) {
                if (ggml_backend_reg_dev_get(reg,i) == device) { compute_type=factory(int(i)); break; }
            }
        }
        if (!t.assert_true(compute_type != nullptr)) return;
        size_t previous_bytes=0;
        for (auto shape : {std::pair{320u,320u},std::pair{640u,384u},std::pair{1024u,768u},std::pair{1280u,1024u}}) {
            std::vector<uint8_t> pixels(size_t(shape.first)*shape.second*3);
            for (size_t i=0;i<pixels.size();++i) pixels[i]=uint8_t((i*13+i/101)%256);
            mtmd::bitmap_ptr bitmap(mtmd_bitmap_init(shape.first,shape.second,pixels.data()));
            mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
            const auto * image=bitmap.get();
            const std::string prompt=mtmd_get_marker(stock.get());
            const mtmd_input_text input{prompt.c_str(),prompt.size(),true,true};
            if (!t.assert_equal(0,mtmd_tokenize(stock.get(),chunks.get(),&input,&image,1))) return;
            const mtmd_input_chunk * chunk=nullptr;
            for (size_t i=0;i<mtmd_input_chunks_size(chunks.get());++i) {
                auto * next=mtmd_input_chunks_get(chunks.get(),i);
                if (mtmd_input_chunk_get_type(next) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { chunk=next; break; }
            }
            if (!t.assert_true(chunk != nullptr)) return;
            mtmd::batch_ptr reference(mtmd_batch_init(stock.get())),batch(mtmd_batch_init(borrowed.get()));
            if (!t.assert_equal(0,mtmd_batch_add_chunk(reference.get(),chunk)) ||
                    !t.assert_equal(0,mtmd_batch_add_chunk(batch.get(),chunk)) ||
                    !t.assert_equal(0,mtmd_batch_encode(reference.get()))) return;
            const size_t elements=mtmd_input_chunk_get_n_tokens(chunk)*llama_model_n_embd_inp(model.get());
            auto * expected_ptr=mtmd_batch_get_output_embd(reference.get(),chunk);
            if (!t.assert_true(expected_ptr != nullptr)) return;
            const std::vector<float> expected(expected_ptr,expected_ptr+elements);
            std::vector<ggml_backend_memory_workspace_group> groups;
            if (!t.assert_true(mtmd_batch_measure_compute_workspace(batch.get(),groups,compute_type)) || !t.assert_true(!groups.empty())) return;
            if (!t.assert_true(groups[0].buft == compute_type)) return;
            size_t bytes=0;
            std::vector<std::unique_ptr<workspace_grant>> grants;
            std::vector<ggml_backend_memory_lease_t> leases;
            for (size_t i=0;i<groups.size();++i) {
                bytes+=groups[i].size;
                grants.push_back(std::make_unique<workspace_grant>(groups[i].buft,groups[i].size,11+i));
                leases.push_back(grants.back()->lease.get());
            }
            t.assert_true(bytes > previous_bytes);
            previous_bytes=bytes;
            auto sentinel=groups;
            t.assert_true(!mtmd_batch_measure_compute_workspace(nullptr,sentinel,compute_type));
            t.assert_equal(groups[0].size,sentinel[0].size);
            workspace_grant tiny(groups[0].buft,groups[0].alignment,91);
            auto invalid=leases; invalid[0]=tiny.lease.get();
            t.assert_true(!mtmd_attach_compute_workspace(borrowed.get(),invalid));
            if (!t.assert_true(mtmd_attach_compute_workspace(borrowed.get(),leases)) ||
                    !t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
            float * actual=mtmd_batch_get_output_embd(batch.get(),chunk);
            if (!t.assert_true(actual != nullptr)) return;
            if (!t.assert_true(std::all_of(actual,actual+elements,[](float value) { return std::isfinite(value); }))) return;
            float error=0;
            for (size_t i=0;i<elements;++i) {
                error=std::max(error,std::abs(actual[i]-expected[i]));
            }
            t.out << "vision " << shape.first << 'x' << shape.second << ": workspace=" << bytes/1048576.0
                  << " MiB, max_error=" << error << '\n';
            t.assert_true(error <= 1e-5f);
            if (shape.first == 320) {
                std::vector<uint8_t> large_pixels(1280*1024*3,127);
                mtmd::bitmap_ptr large_bitmap(mtmd_bitmap_init(1280,1024,large_pixels.data()));
                const auto * large_image=large_bitmap.get();
                mtmd::input_chunks_ptr large_chunks(mtmd_input_chunks_init());
                if (!t.assert_equal(0,mtmd_tokenize(borrowed.get(),large_chunks.get(),&input,&large_image,1))) return;
                mtmd::batch_ptr too_large(mtmd_batch_init(borrowed.get()));
                for (size_t i=0;i<mtmd_input_chunks_size(large_chunks.get());++i) {
                    auto * next=mtmd_input_chunks_get(large_chunks.get(),i);
                    if (mtmd_input_chunk_get_type(next) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                        if (!t.assert_equal(0,mtmd_batch_add_chunk(too_large.get(),next))) return;
                    }
                }
                t.assert_true(mtmd_batch_encode(too_large.get()) != 0);
                if (!t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
                actual=mtmd_batch_get_output_embd(batch.get(),chunk);
            }
            t.assert_true(mtmd_release_compute_workspace(borrowed.get()));
            for (auto & grant : grants) {
                grant->lease.reset();
                t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant->arena.get()));
            }
            t.assert_true(std::equal(expected.begin(),expected.end(),actual,
                [](float a,float b) { return std::abs(a-b) <= 1e-5f; }));
        }
    });
    return t.summary();
}

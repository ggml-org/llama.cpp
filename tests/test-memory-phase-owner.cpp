#include "memory-transition-test.h"
#include "../src/llama-memory-phase-owner.h"
#include "../src/llama-memory-workspace.h"
#include "../src/llama-memory-recurrent-spill.h"

#include <cstring>
int main(int argc, char ** argv) {
    testing t;
    t.test("serial_target_and_mtp_handoff_reuses_one_arena", [](testing & t) {
        fixture f(t,false);
        auto target=f.target;
        target.plan.stages.push_back({11,{10},{{11,f.a,f.a,f.a,LLAMA_MEMORY_ACCESS_WRITE,0},
            {12,2*f.a,2*f.a,f.a,LLAMA_MEMORY_ACCESS_READ_WRITE,0}}});
        auto draft=target; draft.stage=11;
        const llama_memory_phase_cache_identity target_id{101,1}, mtp_id{202,1};
        auto owner=llama_memory_phase_owner::create({
            {llama_memory_work_phase::target_verify,target},
            {llama_memory_work_phase::mtp_draft,draft}},
            f.arenas,{&f.workspace,&f.preserved},target_id,mtp_id);
        if (!t.assert_true(bool(owner))) return;
        const auto first=owner->enter(llama_memory_work_phase::target_verify,1,target_id,mtp_id);
        if (!t.assert_true(first.status == llama_memory_phase_owner_status::changed)) return;
        const auto admission=owner->admit();
        t.assert_true(admission != 0);
        t.assert_true(owner->enter(llama_memory_work_phase::mtp_draft,1,target_id,mtp_id).status ==
            llama_memory_phase_owner_status::busy);
        const llama_memory_phase_cache_identity advanced_target{101,2};
        t.assert_true(owner->finish(admission,advanced_target,mtp_id));
        const auto second=owner->enter(llama_memory_work_phase::mtp_draft,1,advanced_target,mtp_id);
        t.assert_true(second.status == llama_memory_phase_owner_status::changed);
        const int activations=f.workspace.activations;
        t.assert_true(owner->enter(llama_memory_work_phase::mtp_draft,1,advanced_target,mtp_id).status ==
            llama_memory_phase_owner_status::unchanged);
        t.assert_equal(activations,f.workspace.activations);
        t.assert_true(owner->enter(llama_memory_work_phase::target_verify,1,advanced_target,{202,0}).status ==
            llama_memory_phase_owner_status::stale_identity);
        t.assert_true(owner->snapshot().phase == llama_memory_work_phase::mtp_draft);
    });
    t.test("failed_handoff_recovers_old_phase_and_independent_generations", [](testing & t) {
        fixture f(t,false);
        auto target=f.target;
        target.plan.stages.push_back({11,{10},{{11,f.a,f.a,f.a,LLAMA_MEMORY_ACCESS_WRITE,0},
            {12,2*f.a,2*f.a,f.a,LLAMA_MEMORY_ACCESS_READ_WRITE,0}}});
        auto draft=target; draft.stage=11;
        const llama_memory_phase_cache_identity target_id{301,1}, mtp_id{302,1};
        auto owner=llama_memory_phase_owner::create({
            {llama_memory_work_phase::target_verify,target},
            {llama_memory_work_phase::mtp_draft,draft}},
            f.arenas,{&f.workspace,&f.preserved},target_id,mtp_id);
        if (!t.assert_true(bool(owner) && owner->enter(llama_memory_work_phase::target_verify,1,target_id,mtp_id).status ==
                llama_memory_phase_owner_status::changed)) return;
        f.workspace.hook=[&](const std::string & name) {
            if (name == "activate") f.workspace.fail_at="activate";
        };
        const auto failed=owner->enter(llama_memory_work_phase::mtp_draft,1,target_id,mtp_id);
        t.assert_true(failed.status == llama_memory_phase_owner_status::transition_failed);
        t.assert_true(owner->snapshot().phase == llama_memory_work_phase::target_verify);
        t.assert_true(owner->target_identity().generation == 1 && owner->mtp_identity().generation == 1);
        f.workspace.hook={}; f.workspace.fail_at.clear();
        t.assert_true(owner->enter(llama_memory_work_phase::mtp_draft,1,target_id,mtp_id).status ==
            llama_memory_phase_owner_status::changed);
        const auto ticket=owner->admit();
        t.assert_true(ticket != 0);
        t.assert_true(!owner->finish(ticket,{301,0},{302,2}));
        t.assert_true(owner->finish(ticket,target_id,{302,2}));
        t.assert_true(owner->target_identity().generation == 1 && owner->mtp_identity().generation == 2);
        t.assert_true(owner->enter(llama_memory_work_phase::target_verify,1,target_id,mtp_id).status ==
            llama_memory_phase_owner_status::stale_identity);
        t.assert_true(owner->enter(llama_memory_work_phase::target_verify,1,target_id,{302,2}).status ==
            llama_memory_phase_owner_status::changed);
    });

    t.test("verification_and_catchup_widths_are_bounded_by_maximum_grant", [](testing & t) {
        fixture f(t,false);
        const llama_memory_phase_cache_identity target_id{401,1}, mtp_id{402,1};
        auto owner=llama_memory_phase_owner::create({
            {llama_memory_work_phase::target_verify,f.target,4},
            {llama_memory_work_phase::mtp_catchup,f.target,4},
            {llama_memory_work_phase::mtp_draft,f.target,1}},
            f.arenas,{&f.workspace,&f.preserved},target_id,mtp_id);
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(owner->enter(llama_memory_work_phase::target_verify,4,target_id,mtp_id).status ==
            llama_memory_phase_owner_status::changed);
        const auto activations=f.workspace.activations;
        for (uint32_t width=1;width<=4;++width) {
            t.assert_true(owner->enter(llama_memory_work_phase::target_verify,width,target_id,mtp_id).status ==
                llama_memory_phase_owner_status::unchanged);
        }
        t.assert_equal(activations,f.workspace.activations);
        t.assert_true(owner->enter(llama_memory_work_phase::target_verify,5,target_id,mtp_id).status ==
            llama_memory_phase_owner_status::invalid_signal);
        t.assert_true(owner->snapshot().phase == llama_memory_work_phase::target_verify);
        for (uint32_t width=1;width<=4;++width) {
            auto result=owner->enter(llama_memory_work_phase::mtp_catchup,width,target_id,mtp_id);
            t.assert_true(result.status == (width == 1 ?
                llama_memory_phase_owner_status::changed : llama_memory_phase_owner_status::unchanged));
        }
        t.assert_true(owner->enter(llama_memory_work_phase::mtp_draft,2,target_id,mtp_id).status ==
            llama_memory_phase_owner_status::invalid_signal);
        t.assert_true(owner->enter(llama_memory_work_phase::mtp_draft,1,target_id,mtp_id).status ==
            llama_memory_phase_owner_status::changed);
    });

    t.test("workspace_budget_uses_peak_serial_width_and_counts_extra_staging", [](testing & t) {
        llama_memory_phase_workspace_measurement m;
        m.alignment=256;
        m.target_prefill_graph_bytes=700;
        m.target_verify_graph_bytes={100,600,200,400};
        m.mtp_catchup_graph_bytes={200,300,150,250};
        m.mtp_draft_graph_bytes=260;
        m.rollback_stage_bytes=128;
        m.mtp_stage_bytes=128;
        llama_memory_phase_workspace_budget budget;
        if (!t.assert_true(llama_memory_phase_workspace_budget_make(m,budget))) return;
        t.assert_equal(size_t(768),budget.target_prefill_bytes);
        t.assert_equal(size_t(768),budget.target_verify_bytes);
        t.assert_equal(size_t(512),budget.mtp_catchup_bytes);
        t.assert_equal(size_t(512),budget.mtp_draft_bytes);
        t.assert_equal(size_t(768),budget.shared_parent_bytes);
        t.assert_equal(size_t(256),budget.reclaimed_during_mtp_catchup_bytes);
        t.assert_equal(size_t(256),budget.reclaimed_during_mtp_draft_bytes);
        const auto prior=budget;
        m.target_verify_graph_bytes[3]=SIZE_MAX;
        t.assert_true(!llama_memory_phase_workspace_budget_make(m,budget));
        t.assert_equal(prior.shared_parent_bytes,budget.shared_parent_bytes);
        m.target_verify_graph_bytes[3]=400;
        m.alignment=3;
        t.assert_true(!llama_memory_phase_workspace_budget_make(m,budget));
    });

    t.test("two_real_cpu_schedulers_handoff_one_physical_workspace", [](testing & t) {
        ggml_backend_ptr target_backend(ggml_backend_cpu_init());
        ggml_backend_ptr mtp_backend(ggml_backend_cpu_init());
        auto * buft=ggml_backend_cpu_buffer_type();
        ggml_backend_t target_array[]={target_backend.get()};
        ggml_backend_t mtp_array[]={mtp_backend.get()};
        ggml_backend_buffer_type_t types[]={buft};
        ggml_backend_sched_ptr target_sched(ggml_backend_sched_new(target_array,types,1,256,false,true));
        ggml_backend_sched_ptr mtp_sched(ggml_backend_sched_new(mtp_array,types,1,256,false,true));
        if (!t.assert_true(bool(target_sched) && bool(mtp_sched))) return;
        const size_t alignment=ggml_backend_buft_get_alignment(buft);
        const size_t bytes=4096;
        arena_ptr parent(ggml_backend_memory_arena_new(buft,bytes),ggml_backend_memory_arena_free);
        if (!t.assert_true(bool(parent))) return;
        llama_memory_transition_target target;
        target.plan.domains.push_back({1,LLAMA_MEMORY_ALLOCATION_HOST,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
        target.plan.stages={{10,{},{}},{20,{10},{}}};
        target.stage=10;
        target.budgets.push_back({1,LLAMA_MEMORY_ALLOCATION_HOST,bytes,alignment});
        llama_memory_workspace target_workspace(target_sched.get(),{
            {{buft,bytes,alignment,0},{11,1,LLAMA_MEMORY_ALLOCATION_HOST,llama_memory_content::discardable},
                {{10,bytes},{20,0}}}},
            {[] { return true; }, [] { return true; }});
        llama_memory_workspace mtp_workspace(mtp_sched.get(),{
            {{buft,bytes,alignment,0},{12,1,LLAMA_MEMORY_ALLOCATION_HOST,llama_memory_content::discardable},
                {{10,0},{20,bytes}}}},
            {[] { return true; }, [] { return true; }});
        if (!t.assert_true(target_workspace.register_resources(target.plan,{10,20}) &&
                mtp_workspace.register_resources(target.plan,{10,20}))) return;
        auto mtp=target; mtp.stage=20;
        const llama_memory_phase_cache_identity target_id{501,1}, mtp_id{502,1};
        auto owner=llama_memory_phase_owner::create({
            {llama_memory_work_phase::target_verify,target,4},
            {llama_memory_work_phase::mtp_catchup,mtp,4}},
            {{1,LLAMA_MEMORY_ALLOCATION_HOST,parent.get()}},
            {&target_workspace,&mtp_workspace},target_id,mtp_id);
        if (!t.assert_true(bool(owner))) return;
        auto evaluate=[](ggml_backend_sched_t sched, ggml_backend_t backend) {
            ggml_context_ptr ctx(ggml_init({16384,nullptr,true}));
            if (!ctx) return false;
            auto * input=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
            auto * output=ggml_scale(ctx.get(),input,2.0f);
            auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
            ggml_build_forward_expand(graph,output);
            ggml_backend_sched_set_tensor_backend(sched,input,backend);
            ggml_backend_sched_set_tensor_backend(sched,output,backend);
            if (!ggml_backend_sched_reserve(sched,graph) ||
                    !ggml_backend_sched_alloc_graph(sched,graph)) return false;
            const float data[4]={1,2,3,4};
            float result[4]={};
            ggml_backend_tensor_set(input,data,0,sizeof(data));
            if (ggml_backend_sched_graph_compute(sched,graph) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(output,result,0,sizeof(result));
            ggml_backend_sched_reset(sched);
            for (int i=0;i<4;++i) if (result[i] != 2*data[i]) return false;
            return true;
        };
        if (!t.assert_true(owner->enter(llama_memory_work_phase::target_verify,4,target_id,mtp_id).status ==
                llama_memory_phase_owner_status::changed)) return;
        t.assert_equal(size_t(1),target_workspace.leases().size());
        t.assert_equal(size_t(0),mtp_workspace.leases().size());
        t.assert_true(evaluate(target_sched.get(),target_backend.get()));
        const void * target_base=ggml_backend_buffer_get_base(
            ggml_backend_memory_lease_buffer(target_workspace.leases()[0]));
        const auto admission=owner->admit();
        t.assert_true(admission != 0);
        t.assert_true(owner->enter(llama_memory_work_phase::mtp_catchup,1,target_id,mtp_id).status ==
            llama_memory_phase_owner_status::busy);
        t.assert_true(owner->finish(admission,target_id,mtp_id));
        if (!t.assert_true(owner->enter(llama_memory_work_phase::mtp_catchup,4,target_id,mtp_id).status ==
                llama_memory_phase_owner_status::changed)) return;
        t.assert_equal(size_t(0),target_workspace.leases().size());
        t.assert_equal(size_t(1),mtp_workspace.leases().size());
        t.assert_true(evaluate(mtp_sched.get(),mtp_backend.get()));
        const void * mtp_base=ggml_backend_buffer_get_base(
            ggml_backend_memory_lease_buffer(mtp_workspace.leases()[0]));
        t.assert_true(target_base == mtp_base);
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(parent.get()));
        owner.reset();
        t.assert_true(target_workspace.close() && mtp_workspace.close());
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(parent.get()));
    });
    if (argc > 1 && std::strcmp(argv[1],"--cuda") == 0) t.test(
        "two_cuda_schedulers_compute_on_one_exclusive_arena", [](testing & t) {
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(device != nullptr)) return;
        auto * registry=ggml_backend_dev_backend_reg(device);
        using release_t=void (*)(ggml_backend_t);
        auto release=reinterpret_cast<release_t>(
            ggml_backend_reg_get_proc_address(registry,"ggml_backend_cuda_graph_release_all"));
        if (!t.assert_true(release != nullptr)) return;
        ggml_backend_ptr target_backend(ggml_backend_dev_init(device,nullptr));
        ggml_backend_ptr mtp_backend(ggml_backend_dev_init(device,nullptr));
        auto * buft=ggml_backend_dev_buffer_type(device);
        ggml_backend_ptr target_cpu(ggml_backend_cpu_init());
        ggml_backend_ptr mtp_cpu(ggml_backend_cpu_init());
        ggml_backend_t target_array[]={target_backend.get(),target_cpu.get()};
        ggml_backend_t mtp_array[]={mtp_backend.get(),mtp_cpu.get()};
        ggml_backend_buffer_type_t types[]={buft,ggml_backend_cpu_buffer_type()};
        ggml_backend_sched_ptr target_sched(ggml_backend_sched_new(target_array,types,2,256,false,true));
        ggml_backend_sched_ptr mtp_sched(ggml_backend_sched_new(mtp_array,types,2,256,false,true));
        if (!t.assert_true(bool(target_sched) && bool(mtp_sched))) return;
        const size_t alignment=ggml_backend_buft_get_alignment(buft);
        const size_t bytes=4096;
        arena_ptr parent(ggml_backend_memory_arena_new(buft,bytes),ggml_backend_memory_arena_free);
        if (!t.assert_true(bool(parent))) return;
        llama_memory_transition_target target;
        target.plan.domains.push_back({1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
        target.plan.stages={{10,{},{}},{20,{10},{}}};
        target.stage=10;
        target.budgets.push_back({1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,bytes,alignment});
        llama_memory_workspace target_workspace(target_sched.get(),{
            {{buft,bytes,alignment,0},{11,1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,llama_memory_content::discardable},
                {{10,bytes},{20,0}}}},
            {[] { return true; },[&] { release(target_backend.get()); return true; }});
        llama_memory_workspace mtp_workspace(mtp_sched.get(),{
            {{buft,bytes,alignment,0},{12,1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,llama_memory_content::discardable},
                {{10,0},{20,bytes}}}},
            {[] { return true; },[&] { release(mtp_backend.get()); return true; }});
        if (!t.assert_true(target_workspace.register_resources(target.plan,{10,20}) &&
                mtp_workspace.register_resources(target.plan,{10,20}))) return;
        auto mtp=target; mtp.stage=20;
        const llama_memory_phase_cache_identity target_id{601,1}, mtp_id{602,1};
        auto owner=llama_memory_phase_owner::create({
            {llama_memory_work_phase::target_verify,target,4},
            {llama_memory_work_phase::mtp_catchup,mtp,4}},
            {{1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,parent.get()}},
            {&target_workspace,&mtp_workspace},target_id,mtp_id);
        if (!t.assert_true(bool(owner))) return;
        const auto evaluate=[&](ggml_backend_sched_t sched,ggml_backend_t backend) {
            ggml_context_ptr ctx(ggml_init({16384,nullptr,true}));
            auto * input=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
            auto * output=ggml_scale(ctx.get(),input,2.0f);
            auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
            ggml_build_forward_expand(graph,output);
            ggml_backend_sched_set_tensor_backend(sched,input,backend);
            ggml_backend_sched_set_tensor_backend(sched,output,backend);
            if (!ggml_backend_sched_reserve(sched,graph) ||
                    !ggml_backend_sched_alloc_graph(sched,graph)) return false;
            const float data[4]={1,2,3,4};
            float result[4]={};
            ggml_backend_tensor_set(input,data,0,sizeof(data));
            if (ggml_backend_sched_graph_compute(sched,graph) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(output,result,0,sizeof(result));
            ggml_backend_sched_reset(sched);
            for (int i=0;i<4;++i) if (result[i] != 2*data[i]) return false;
            return true;
        };
        if (!t.assert_true(owner->enter(llama_memory_work_phase::target_verify,4,target_id,mtp_id).status ==
                llama_memory_phase_owner_status::changed)) return;
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,1,snapshots))) return;
        auto bank=llama_recurrent_spill_bank::create(
            snapshots,ggml_backend_dev_host_buffer_type(device),701,0);
        if (!t.assert_true(bool(bank) &&
                bank->enable_publication(device,buft,2))) return;
        ggml_context_ptr capture_ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_2d(capture_ctx.get(),GGML_TYPE_F32,4,1);
        auto * s=ggml_new_tensor_2d(capture_ctx.get(),GGML_TYPE_F32,8,1);
        auto * write_r=ggml_cpy(capture_ctx.get(),r,bank->staging_tensor(0,false,1));
        auto * write_s=ggml_cpy(capture_ctx.get(),s,bank->staging_tensor(0,true,1));
        auto * capture_graph=ggml_new_graph_custom(capture_ctx.get(),32,false);
        ggml_build_forward_expand(capture_graph,ggml_cpy(capture_ctx.get(),write_r,
            bank->publication_tensor(0,false,1)));
        ggml_build_forward_expand(capture_graph,ggml_cpy(capture_ctx.get(),write_s,
            bank->publication_tensor(0,true,1)));
        ggml_backend_sched_set_tensor_backend(target_sched.get(),r,target_backend.get());
        ggml_backend_sched_set_tensor_backend(target_sched.get(),s,target_backend.get());
        if (!t.assert_true(ggml_backend_sched_alloc_graph(target_sched.get(),capture_graph))) return;
        const float expected_r[4]={1,2,3,4};
        const float expected_s[8]={8,7,6,5,4,3,2,1};
        ggml_backend_tensor_set(r,expected_r,0,sizeof(expected_r));
        ggml_backend_tensor_set(s,expected_s,0,sizeof(expected_s));
        if (!t.assert_true(bank->begin_staged_capture({701,0,1},2) &&
                ggml_backend_sched_graph_compute_async(target_sched.get(),capture_graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready())) return;
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        if (!t.assert_true(evaluate(target_sched.get(),target_backend.get()))) return;
        const void * first=ggml_backend_buffer_get_base(
            ggml_backend_memory_lease_buffer(target_workspace.leases()[0]));
        if (!t.assert_true(owner->enter(llama_memory_work_phase::mtp_catchup,4,target_id,mtp_id).status ==
                llama_memory_phase_owner_status::changed)) return;
        t.assert_equal(size_t(0),target_workspace.leases().size());
        t.assert_equal(size_t(1),mtp_workspace.leases().size());
        const void * second=ggml_backend_buffer_get_base(
            ggml_backend_memory_lease_buffer(mtp_workspace.leases()[0]));
        t.assert_true(first == second);
        t.assert_true(evaluate(mtp_sched.get(),mtp_backend.get()));
        if (!t.assert_true(bank->complete_capture())) return;
        const auto * actual_r=bank->snapshot_data(0,false,1,1);
        const auto * actual_s=bank->snapshot_data(0,true,1,1);
        t.assert_true(actual_r && !std::memcmp(actual_r,expected_r,sizeof(expected_r)));
        t.assert_true(actual_s && !std::memcmp(actual_s,expected_s,sizeof(expected_s)));

        owner.reset();
        t.assert_true(target_workspace.close() && mtp_workspace.close());
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(parent.get()));
    });


    return t.summary();
}

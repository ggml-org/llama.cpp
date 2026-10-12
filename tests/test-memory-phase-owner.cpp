#include "memory-transition-test.h"
#include "../src/llama-memory-phase-owner.h"
#include "../src/llama-memory-workspace.h"
#include "../src/llama-memory-recurrent-spill.h"
#include "../src/llama-memory-bounded-kv.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-alloc.h"

#include <cstring>
#include <type_traits>

static_assert(!std::is_copy_constructible<llama_memory_bounded_kv>::value,"a cache must have one write owner");
static_assert(!std::is_move_constructible<llama_memory_bounded_kv>::value,"pending preparations borrow the owner address");
int main(int argc, char ** argv) {
    testing t;
    t.test("bounded_KV_rejects_inconsistent_backend_quotes_and_invalid_geometry", [](testing & t) {
        llama_memory_cache_requirement cache;
        cache.identity={605,1}; cache.role=llama_memory_cache_role::bounded_resident;
        cache.layers=2; cache.history_rows=4096; cache.window_rows=16; cache.headroom_rows=16; cache.row_padding=32;
        cache.format={GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,32,32,1,false};
        cache.row_bytes=52;
        auto padded=*ggml_backend_cpu_buffer_type();
        padded.iface.get_alloc_size=[](ggml_backend_buffer_type_t,const ggml_tensor * tensor) { return ggml_nbytes(tensor)+13; };
        llama_memory_bounded_kv_layout layout;
        if (!t.assert_true(llama_memory_bounded_kv_layout_make(cache,{3,9},&padded,layout))) return;
        t.assert_true(layout.allocation.bytes > cache.row_bytes*32*2);
        t.assert_true(layout.layers[0].v_offset >= layout.layers[0].k_offset+layout.layers[0].k_bytes+13);
        const auto before=layout;
        padded.iface.get_alloc_size_n=[](ggml_backend_buffer_type_t,ggml_tensor **,int) { return size_t(1); };
        t.assert_true(!llama_memory_bounded_kv_layout_make(cache,{3,9},&padded,layout));
        t.assert_equal(before.allocation.bytes,layout.allocation.bytes);
        auto * cpu=ggml_backend_cpu_buffer_type();
        t.assert_true(!llama_memory_bounded_kv_layout_make(cache,{3,3},cpu,layout));
        t.assert_true(!llama_memory_bounded_kv_layout_make(cache,{3},cpu,layout));
        auto bad=cache; bad.role=llama_memory_cache_role::full_history;
        t.assert_true(!llama_memory_bounded_kv_layout_make(bad,{3,9},cpu,layout));
        bad=cache; bad.format.key_head_dim=31;
        t.assert_true(!llama_memory_bounded_kv_layout_make(bad,{3,9},cpu,layout));
        bad=cache; bad.row_padding=3;
        t.assert_true(!llama_memory_bounded_kv_layout_make(bad,{3,9},cpu,layout));
        bad=cache; bad.window_rows=UINT32_MAX;
        t.assert_true(!llama_memory_bounded_kv_layout_make(bad,{3,9},cpu,layout));
        bad=cache; bad.format.kv_heads=UINT32_MAX; bad.format.key_head_dim=UINT32_MAX-31;
        t.assert_true(!llama_memory_bounded_kv_layout_make(bad,{3,9},cpu,layout));
        t.assert_equal(before.allocation.bytes,layout.allocation.bytes);
    });
    t.test("bounded_KV_measures_stock_window_geometry_and_backend_padding", [](testing & t) {
        for (bool gemma : {false,true}) for (ggml_type key : {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,GGML_TYPE_Q5_0,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1})
            for (ggml_type value : {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,GGML_TYPE_Q5_0,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1}) {
                llama_memory_cache_requirement cache;
                cache.identity={601,1}; cache.role=llama_memory_cache_role::bounded_resident;
                cache.layers=gemma ? 3 : 5; cache.history_rows=262144;
                cache.window_rows=gemma ? 1024 : 2048; cache.headroom_rows=256; cache.row_padding=256;
                cache.format={key,value,gemma ? 256u : 128u,gemma ? 256u : 128u,gemma ? 16u : 8u,false};
                cache.row_bytes=ggml_row_size(key,cache.format.key_head_dim*cache.format.kv_heads)+
                    ggml_row_size(value,cache.format.value_head_dim*cache.format.kv_heads);
                std::vector<uint32_t> ids(cache.layers);
                for (uint32_t i=0;i<cache.layers;++i) ids[i]=i*3;
                llama_memory_bounded_kv_layout layout;
                if (!t.assert_true(llama_memory_bounded_kv_layout_make(cache,ids,ggml_backend_cpu_buffer_type(),layout))) return;
                t.assert_equal(gemma ? 1280u : 2304u,layout.allocation.capacity_rows);
                t.assert_true(layout.allocation.bytes >= cache.row_bytes*cache.layers*layout.allocation.capacity_rows);
                const auto bytes=layout.allocation.bytes;
                cache.history_rows=131072;
                t.assert_true(llama_memory_bounded_kv_layout_make(cache,ids,ggml_backend_cpu_buffer_type(),layout));
                t.assert_equal(bytes,layout.allocation.bytes);
                const auto before=layout.allocation;
                ++cache.row_bytes;
                t.assert_true(!llama_memory_bounded_kv_layout_make(cache,ids,ggml_backend_cpu_buffer_type(),layout));
                t.assert_equal(before.bytes,layout.allocation.bytes);
            }
    });
    t.test("bounded_KV_pins_preserve_storage_and_close_stale_writer_admission", [](testing & t) {
        auto * buft=ggml_backend_cpu_buffer_type();
        llama_memory_cache_requirement cache;
        cache.identity={602,1}; cache.role=llama_memory_cache_role::bounded_resident;
        cache.layers=1; cache.history_rows=4096; cache.window_rows=16; cache.headroom_rows=16; cache.row_padding=32;
        cache.format={GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,32,32,1,false};
        cache.row_bytes=ggml_row_size(GGML_TYPE_Q8_0,32)+ggml_row_size(GGML_TYPE_Q4_0,32);
        llama_memory_bounded_kv_layout layout;
        if (!t.assert_true(llama_memory_bounded_kv_layout_make(cache,{17},buft,layout))) return;
        const size_t a=layout.allocation.alignment, bytes=layout.allocation.bytes;
        const auto make=[&](size_t size,uint32_t flags) {
            arena_ptr arena(ggml_backend_memory_arena_new(buft,bytes+2*a),ggml_backend_memory_arena_free);
            if (!arena || !ggml_backend_memory_arena_begin(arena.get(),0) ||
                    !ggml_backend_memory_arena_reserve_at(arena.get(),88,a,size,a,flags,nullptr) ||
                    !ggml_backend_memory_arena_commit(arena.get())) return arena_ptr(nullptr,ggml_backend_memory_arena_free);
            return arena;
        };
        auto arena=make(bytes,GGML_BACKEND_MEMORY_REGION_PERSISTENT);
        if (!t.assert_true(bool(arena))) return;
        lease_ptr grant(ggml_backend_memory_arena_acquire(arena.get(),88),ggml_backend_memory_lease_free);
        for (int mutation=0;mutation<5;++mutation) {
            auto forged=layout;
            switch (mutation) {
                case 0: --forged.allocation.bytes; break;
                case 1: ++forged.layers[0].v_offset; break;
                case 2: ++forged.allocation.capacity_rows; break;
                case 3: forged.allocation.format.value_type=GGML_TYPE_F16; break;
                case 4: ++forged.allocation.cache.generation; break;
            }
            llama_memory_bounded_kv invalid(std::move(forged));
            t.assert_true(!invalid.bind(grant.get()) && !invalid.read({602,1}));
        }
        auto consumer=std::make_unique<llama_memory_bounded_kv>(layout);
        if (!t.assert_true(consumer->bind(grant.get()))) return;
        auto * buffer=ggml_backend_memory_lease_buffer(grant.get());
        ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(arena.get()),0x5a);
        auto read=consumer->read({602,1});
        t.assert_true(bool(read) && read.buffer() == buffer);
        t.assert_true(!consumer->read({602,2}) && !consumer->read({603,1}));
        t.assert_equal(uint64_t(0),consumer->begin_write({602,1}));
        t.assert_true(!consumer->reconstruct({602,1}));
        t.assert_true(consumer->bind(grant.get()));
        read.reset();
        const auto ticket=consumer->begin_write({602,1});
        if (!t.assert_true(ticket != 0)) return;
        ggml_context_ptr tensors(ggml_init({4096+3*ggml_tensor_overhead(),nullptr,true}));
        auto * k=ggml_new_tensor_3d(tensors.get(),cache.format.key_type,32,32,1);
        auto * v=ggml_new_tensor_3d(tensors.get(),cache.format.value_type,32,32,1);
        ggml_tallocr allocator{};
        if (!t.assert_true(ggml_tallocr_new_range(&allocator,buffer,0,bytes))) return;
        if (!t.assert_true(ggml_tallocr_alloc(&allocator,k) == GGML_STATUS_SUCCESS &&
                ggml_tallocr_alloc(&allocator,v) == GGML_STATUS_SUCCESS)) return;
        const auto base=uintptr_t(ggml_backend_buffer_get_base(buffer));
        t.assert_equal(layout.layers[0].k_offset,uintptr_t(k->data)-base);
        t.assert_equal(layout.layers[0].v_offset,uintptr_t(v->data)-base);
        std::vector<uint8_t> payload(std::max(ggml_nbytes(k),ggml_nbytes(v)),0xf7);
        ggml_backend_tensor_set(k,payload.data(),0,ggml_nbytes(k));
        ggml_backend_tensor_set(v,payload.data(),0,ggml_nbytes(v));
        auto * overflow=ggml_new_tensor_1d(tensors.get(),GGML_TYPE_F32,32);
        const auto offset=allocator.offset;
        t.assert_true(ggml_tallocr_alloc(&allocator,overflow) == GGML_STATUS_ALLOC_FAILED && !overflow->data);
        t.assert_equal(offset,allocator.offset);
        const auto * parent=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena.get())));
        t.assert_true(std::all_of(parent,parent+a,[](uint8_t value){return value == 0x5a;}));
        t.assert_true(std::all_of(parent+a+bytes,parent+2*a+bytes,[](uint8_t value){return value == 0x5a;}));
        ggml_backend_buffer_clear(buffer,0x5a);
        t.assert_true(!consumer->read({602,1}) && !consumer->bind(grant.get()));
        t.assert_true(!consumer->finish_write(ticket+1,true));
        if (!t.assert_true(consumer->finish_write(ticket,true))) return;
        t.assert_equal(uint64_t(2),consumer->identity().generation);
        t.assert_true(!consumer->read({602,1}));
        const auto failed=consumer->begin_write({602,2});
        t.assert_true(failed && consumer->finish_write(failed,false));
        t.assert_true(!consumer->read({602,2}) && !consumer->begin_write({602,2}));
        t.assert_true(consumer->reconstruct({602,2}));
        auto moved=make(bytes,GGML_BACKEND_MEMORY_REGION_PERSISTENT);
        lease_ptr other(ggml_backend_memory_arena_acquire(moved.get(),88),ggml_backend_memory_lease_free);
        t.assert_true(!consumer->bind(other.get()));
        read=consumer->read({602,3});
        grant.reset(); consumer.reset(); arena.reset();
        t.assert_true(bool(read) && read.layout()->cache.identity.generation == 3);
        const auto * data=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(read.buffer()));
        t.assert_true(data && std::all_of(data,data+bytes,[](uint8_t v){return v == 0x5a;}));
        read.reset();
        for (auto invalid : {std::pair{bytes-1,uint32_t(GGML_BACKEND_MEMORY_REGION_PERSISTENT)},std::pair{bytes,uint32_t(0)}}) {
            auto bad=make(invalid.first,invalid.second);
            lease_ptr short_grant(ggml_backend_memory_arena_acquire(bad.get(),88),ggml_backend_memory_lease_free);
            llama_memory_bounded_kv fresh(layout);
            t.assert_true(!fresh.bind(short_grant.get()) && !fresh.read({602,1}));
        }
    });
    t.test("bounded_KV_participates_in_phase_changes_without_moving_or_clearing", [](testing & t) {
        fixture f(t,false);
        llama_memory_cache_requirement cache;
        cache.identity={606,1}; cache.role=llama_memory_cache_role::bounded_resident;
        cache.layers=1; cache.history_rows=4096; cache.window_rows=16; cache.headroom_rows=16; cache.row_padding=32;
        cache.format={GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,32,32,1,false}; cache.row_bytes=52;
        llama_memory_bounded_kv_layout layout;
        if (!t.assert_true(llama_memory_bounded_kv_layout_make(cache,{7},ggml_backend_cpu_buffer_type(),layout))) return;
        const auto a=layout.allocation.alignment, bytes=layout.allocation.bytes;
        arena_ptr arena(ggml_backend_memory_arena_new(layout.buft,bytes+4*a),ggml_backend_memory_arena_free);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),55,0,bytes,a,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        lease_ptr grant(ggml_backend_memory_arena_acquire(arena.get(),55),ggml_backend_memory_lease_free);
        llama_memory_bounded_kv owner(layout,{55,1,LLAMA_MEMORY_ALLOCATION_HOST,llama_memory_content::preserve});
        if (!t.assert_true(owner.bind(grant.get()))) return;
        auto target=f.target;
        target.plan.stages.push_back({11,{10},{{11,f.a,f.a,f.a,LLAMA_MEMORY_ACCESS_WRITE,0},
            {12,2*f.a,2*f.a,f.a,LLAMA_MEMORY_ACCESS_READ_WRITE,0}}});
        if (!t.assert_true(owner.register_resources(target.plan,{10,11}))) return;
        target.budgets[0].capacity=bytes+4*a;
        target.fixed.clear();
        ggml_backend_memory_region region{};
        if (!t.assert_true(ggml_backend_memory_lease_get_region(grant.get(),&region))) return;
        target.fixed.push_back({0,region});
        ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(arena.get()),0x5a);
        auto pin=owner.read({606,1});
        llama_memory_transition transition({&owner});
        const std::vector<llama_memory_transition_arena> arenas={{1,LLAMA_MEMORY_ALLOCATION_HOST,arena.get()}};
        for (auto stage : {10u,11u,10u}) {
            target.stage=stage;
            if (!t.assert_true(transition.prepare(target).status == status::prepared)) return;
            t.assert_equal(uint64_t(0),owner.begin_write({606,1}));
            if (!t.assert_true(transition.activate(arenas).status == status::activated)) return;
            const auto * data=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(pin.buffer()));
            t.assert_true(std::all_of(data,data+bytes,[](uint8_t value){return value == 0x5a;}));
        }
        target.stage=11;
        pin.reset();
        t.assert_true(transition.prepare(target).status == status::prepared);
        t.assert_equal(uint64_t(0),owner.begin_write({606,1}));
        t.assert_true(transition.cancel());
        t.assert_true(bool(owner.read({606,1})));
        struct failing_participant : llama_memory_consumer {
            bool fail=false;
            struct work : llama_memory_preparation {
                failing_participant & owner;
                explicit work(failing_participant & owner) : owner(owner) {}
                bool quiesce(const std::vector<llama_memory_resource_id> &) override { return true; }
                bool drain() override { return true; }
                bool invalidate() override { return true; }
                bool release() override { return true; }
                bool bind(const std::vector<llama_memory_region_binding> &) override { return true; }
                bool activate() override { return !owner.fail; }
                bool prepare_recovery(std::unique_ptr<llama_memory_preparation> &) override { return true; }
            };
            bool prepare(const llama_memory_transition_target &,const llama_memory_layout &,
                    std::unique_ptr<llama_memory_preparation> & output) override {
                output=std::make_unique<work>(*this); return true;
            }
        } failure;
        llama_memory_transition recovery({&owner,&failure});
        target.stage=10;
        if (!t.assert_true(recovery.prepare(target).status == status::prepared &&
                recovery.activate(arenas).status == status::activated)) return;
        failure.fail=true; target.stage=11;
        if (!t.assert_true(recovery.prepare(target).status == status::prepared &&
                recovery.activate(arenas).status == status::activation_failed)) return;
        t.assert_true(recovery.recover().status == status::recovered);
        t.assert_true(bool(owner.read({606,1})));
        const auto ticket_after_recovery=owner.begin_write({606,1});
        if (!t.assert_true(ticket_after_recovery != 0)) return;
        t.assert_true(owner.finish_write(ticket_after_recovery,true));
        t.assert_true(!owner.read({606,1}) && owner.read({606,2}));
        target.fixed.front().region.offset=a;
        t.assert_true(transition.prepare(target).status == status::consumer_failed);
        t.assert_true(bool(owner.read({606,2})));
        pin.reset(); grant.reset();
        const auto ticket=owner.begin_write({606,2});
        t.assert_true(ticket && owner.finish_write(ticket,true));
        t.assert_true(transition.prepare(target).status == status::consumer_failed);
    });
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


    t.test("measured_workloads_fit_exactly_and_report_missing_kernels_separately", [](testing & t) {
        using phase = llama_memory_work_phase;
        llama_memory_cache_requirement cache{{201, 1}, {}, llama_memory_cache_role::bounded_resident,
                                           5, 262144, 2048, 256, 256, 1664};
        cache.format = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,128,128,8,false};
        std::vector<llama_memory_workload_requirement> work{
            {phase::draft_feature_injection,cache.identity,256,0,0,0,256,true,true,false},
            {phase::draft_block_prediction,cache.identity,8,8,8,0,256,true,true,false},
            {phase::target_verify,cache.identity,8,8,8,7,256,true,true,true}};
        std::vector<llama_memory_workload_measurement> measured;
        const std::vector<llama_memory_cache_allocation_measurement> allocation{{cache.identity,19169280,16,2304,cache.format}};
        for (const auto & r : work) measured.push_back({r, 100, 20, 25, 16, true, false, false});
        measured[1].graph_bytes = 200;
        measured[2].live_device_bytes = 80;
        llama_memory_workload_budget budget;
        auto result = llama_memory_workload_budget_make({cache}, allocation, work, measured, 0, 19169504, budget);
        t.assert_true(result.status == llama_memory_workload_budget_status::success);
        t.assert_equal(size_t(19169280), budget.persistent_cache_bytes);
        t.assert_equal(size_t(224), budget.workspace_peak_bytes);
        t.assert_equal(size_t(19169504), budget.parent_minimum_bytes);
        t.assert_equal(size_t(25), budget.host_feature_peak_bytes);
        const auto before = budget;
        result = llama_memory_workload_budget_make({cache}, allocation, work, measured, 0, 19169503, budget);
        t.assert_true(result.status == llama_memory_workload_budget_status::insufficient_budget);
        t.assert_equal(size_t(1), result.additional_bytes);
        t.assert_equal(before.parent_minimum_bytes, budget.parent_minimum_bytes);
        measured[1].execution_available = false;
        result = llama_memory_workload_budget_make({cache}, allocation, work, measured, 0, 1, budget);
        t.assert_true(result.status == llama_memory_workload_budget_status::missing_kernel);
        t.assert_equal(size_t(1), result.workload);
        measured[1].execution_available = true;
        measured[1].workload.query_rows = 4;
        result = llama_memory_workload_budget_make({cache}, allocation, work, measured, 0, 19169504, budget);
        t.assert_true(result.status == llama_memory_workload_budget_status::missing_measurement);
        measured[1].workload = work[1];
        measured[1].fallback_selected = true;
        result = llama_memory_workload_budget_make({cache}, allocation, work, measured, 0, 19169504, budget);
        t.assert_true(result.status == llama_memory_workload_budget_status::unbudgeted_fallback);
        measured[1].fallback_measured = true;
        t.assert_true(llama_memory_workload_budget_make({cache}, allocation, work, measured, 0, 19169504, budget).status ==
                      llama_memory_workload_budget_status::success);
        measured[1].live_device_bytes = SIZE_MAX;
        result = llama_memory_workload_budget_make({cache}, allocation, work, measured, 0, SIZE_MAX, budget);
        t.assert_true(result.status == llama_memory_workload_budget_status::invalid_measurement);
        auto padded_allocation = allocation;
        padded_allocation[0].alignment = 64;
        measured[1].live_device_bytes = 20;
        result = llama_memory_workload_budget_make({cache}, padded_allocation, work, measured, 1, 19169567, budget);
        t.assert_true(result.status == llama_memory_workload_budget_status::insufficient_budget);
        t.assert_equal(size_t(1), result.additional_bytes);
    });

    t.test("workload_admission_rejects_stale_and_duplicate_cache_catalogs", [](testing & t) {
        llama_memory_cache_requirement cache{{201, 2}, {}, llama_memory_cache_role::full_history,
                                           16, 262144, 0, 0, 256, 1664};
        cache.format = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,4,false};
        llama_memory_workload_requirement work{llama_memory_work_phase::target_verify,cache.identity,8,8,8,7,256,true,true,true};
        llama_memory_workload_measurement measured{work, 100, 20, 25, 16, true, false, false};
        const std::vector<llama_memory_cache_allocation_measurement> allocation{{cache.identity,0,16,262144,cache.format}};
        llama_memory_workload_budget budget;
        t.assert_true(llama_memory_workload_budget_make({cache}, allocation, {work}, {measured}, 0, 128, budget).status ==
                      llama_memory_workload_budget_status::success);
        t.assert_equal(size_t(0), budget.persistent_cache_bytes);
        const auto padded = llama_memory_workload_budget_make({cache}, allocation, {work}, {measured}, 1, 143, budget);
        t.assert_true(padded.status == llama_memory_workload_budget_status::insufficient_budget);
        t.assert_equal(size_t(1), padded.additional_bytes);
        t.assert_true(llama_memory_workload_budget_make({cache}, allocation, {work}, {measured}, 1, 144, budget).status ==
                      llama_memory_workload_budget_status::success);
        t.assert_equal(size_t(144), budget.parent_minimum_bytes);
        t.assert_true(llama_memory_workload_budget_make({cache,cache}, {allocation[0],allocation[0]}, {work}, {measured}, 0, 128, budget).status ==
                      llama_memory_workload_budget_status::invalid_cache);
        work.cache.generation++;
        t.assert_true(llama_memory_workload_budget_make({cache}, allocation, {work}, {measured}, 0, 128, budget).status ==
                      llama_memory_workload_budget_status::stale_cache);
        work.cache.id++;
        t.assert_true(llama_memory_workload_budget_make({cache}, allocation, {work}, {measured}, 0, 128, budget).status ==
                      llama_memory_workload_budget_status::missing_cache);
    });

    t.test("measured_cache_padding_and_aliases_are_accounted_once", [](testing & t) {
        llama_memory_cache_requirement source{{201,2},{},llama_memory_cache_role::bounded_resident,
                                            5,262144,2048,256,256,1664};
        source.format = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,128,128,8,false};
        auto alias = source;
        alias.identity = {202,1};
        alias.role = llama_memory_cache_role::read_only_source;
        alias.source = source.identity;
        alias.layers = 1;
        llama_memory_workload_requirement work{llama_memory_work_phase::target_verify,alias.identity,8,8,8,0,256,true,false,false};
        llama_memory_workload_measurement measurement{work,100,20,0,64,true,false,false};
        std::vector<llama_memory_cache_allocation_measurement> allocations{
            {source.identity,19169312,64,2304,source.format},{alias.identity,0,64,2304,alias.format}};
        llama_memory_workload_budget budget;
        auto result = llama_memory_workload_budget_make({source,alias},allocations,{work},{measurement},0,19169472,budget);
        t.assert_true(result.status == llama_memory_workload_budget_status::success);
        t.assert_equal(size_t(19169312), budget.persistent_cache_bytes);
        t.assert_equal(size_t(19169472), budget.parent_minimum_bytes);
        allocations[1].bytes = 1;
        t.assert_true(llama_memory_workload_budget_make({source,alias},allocations,{work},{measurement},0,SIZE_MAX,budget).status ==
                      llama_memory_workload_budget_status::invalid_measurement);
        allocations[1].bytes = 0;
        allocations[0].bytes = 19169279;
        t.assert_true(llama_memory_workload_budget_make({source,alias},allocations,{work},{measurement},0,SIZE_MAX,budget).status ==
                      llama_memory_workload_budget_status::invalid_measurement);
        allocations[0].bytes = 19169312;
        allocations[0].cache.generation++;
        t.assert_true(llama_memory_workload_budget_make({source,alias},allocations,{work},{measurement},0,SIZE_MAX,budget).status ==
                      llama_memory_workload_budget_status::missing_measurement);
        allocations[0].cache = source.identity;
        allocations[0].capacity_rows--;
        t.assert_true(llama_memory_workload_budget_make({source,alias},allocations,{work},{measurement},0,SIZE_MAX,budget).status ==
                      llama_memory_workload_budget_status::missing_measurement);
        allocations[0].capacity_rows++;
        allocations[0].format.key_head_dim *= 2;
        allocations[0].format.kv_heads /= 2;
        t.assert_true(llama_memory_workload_budget_make({source,alias},allocations,{work},{measurement},0,SIZE_MAX,budget).status ==
                      llama_memory_workload_budget_status::missing_measurement);
    });

    return t.summary();
}

#include "../src/llama-memory-recurrent-spill.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include <cstring>
#include <array>

int main(int argc, char ** argv) {
    testing t;
    t.test("layout_matches_stock_recurrent_rows_and_counts_exact_bytes", [](testing & t) {
        const std::vector<llama_recurrent_spill_layer> layers{
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,true},
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,true},
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,false},
        };
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,3,layout))) return;
        t.assert_equal(size_t(98304),layout.layers[0].r_row_bytes);
        t.assert_equal(size_t(3145728),layout.layers[0].s_row_bytes);
        t.assert_equal(size_t(6488064),layout.device_current_bytes);
        t.assert_equal(size_t(19464192),layout.host_snapshot_bytes);
        t.assert_equal(size_t(25952256),layout.stock_device_bytes);
        t.assert_equal(layout.device_current_bytes+layout.host_snapshot_bytes,layout.stock_device_bytes);
        t.assert_equal(size_t(3244032),layout.device_stage_bytes);
        t.assert_true(!layout.layers[2].enabled);
        t.assert_equal(layout.layers[0].s_offset+3*layout.layers[0].s_row_bytes,layout.layers[1].r_offset);
        ggml_context * ctx=ggml_init({32768,nullptr,true});
        auto * r=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,24576,4);
        auto * s=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,786432,4);
        t.assert_equal(2*(ggml_nbytes(r)+ggml_nbytes(s)),layout.stock_device_bytes);
        ggml_free(ctx);
    });
    t.test("two_slot_stage_is_bounded_by_one_layer_not_all_recurrent_layers", [](testing & t) {
        std::vector<llama_recurrent_spill_layer> layers(
            48,{GGML_TYPE_F32,GGML_TYPE_F32,30720,786432,true});
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,3,snapshots))) return;
        t.assert_equal(size_t(149.625*1024*1024),snapshots.device_current_bytes);
        t.assert_equal(size_t(448.875*1024*1024),snapshots.host_snapshot_bytes);
        llama_recurrent_stage_layout stage;
        if (!t.assert_true(llama_recurrent_stage_layout_make(snapshots,2,256,stage))) return;
        t.assert_equal(size_t(9805824),stage.bytes_per_slot);
        t.assert_equal(size_t(19611648),stage.total_bytes);
        size_t offset=0,bytes=0;
        t.assert_true(llama_recurrent_stage_region(stage,0,false,1,0,offset,bytes));
        t.assert_equal(size_t(0),offset);
        t.assert_equal(size_t(122880),bytes);
        t.assert_true(llama_recurrent_stage_region(stage,47,true,3,1,offset,bytes));
        t.assert_equal(size_t(9805824+368640+2*3145728),offset);
        t.assert_equal(size_t(3145728),bytes);
        auto inconsistent=stage;
        inconsistent.bytes_per_slot=1;
        offset=123; bytes=456;
        t.assert_true(!llama_recurrent_stage_region(inconsistent,0,false,1,0,offset,bytes));
        t.assert_equal(size_t(123),offset);
        t.assert_equal(size_t(456),bytes);
        t.assert_true(!llama_recurrent_stage_region(stage,48,true,1,0,offset,bytes));
        t.assert_true(!llama_recurrent_stage_region(stage,0,true,4,0,offset,bytes));
        t.assert_true(!llama_recurrent_stage_region(stage,0,true,1,2,offset,bytes));
        const auto original=stage.total_bytes;
        t.assert_true(!llama_recurrent_stage_layout_make(snapshots,2,3,stage));
        t.assert_equal(original,stage.total_bytes);
        t.assert_true(!llama_recurrent_stage_layout_make(
            snapshots,3,size_t(1) << (sizeof(size_t)*8-1),stage));
        t.assert_equal(original,stage.total_bytes);
        auto malformed=snapshots;
        malformed.layers[0].r_row_bytes=0;
        t.assert_true(!llama_recurrent_stage_layout_make(malformed,2,256,stage));
        t.assert_equal(original,stage.total_bytes);
    });

    t.test("typed_stage_tensors_alias_only_their_planned_ring_regions", [](testing & t) {
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,16,32,true}},1,3,snapshots))) return;
        auto stage=llama_recurrent_stage_buffer::create(snapshots,2,ggml_backend_cpu_buffer_type());
        if (!t.assert_true(bool(stage))) return;
        t.assert_equal(stage->layout().total_bytes,stage->bytes());
        auto * base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage->buffer()));
        size_t offset=0,bytes=0;
        if (!t.assert_true(llama_recurrent_stage_region(
                stage->layout(),0,false,2,1,offset,bytes))) return;
        auto * r=stage->tensor(0,false,2,1);
        t.assert_true(r && r->buffer == stage->buffer() && r->data == base+offset);
        t.assert_equal(bytes,ggml_nbytes(r));
        if (!t.assert_true(llama_recurrent_stage_region(
                stage->layout(),0,true,3,1,offset,bytes))) return;
        auto * s=stage->tensor(0,true,3,1);
        t.assert_true(s && s->buffer == stage->buffer() && s->data == base+offset);
        t.assert_equal(bytes,ggml_nbytes(s));
        t.assert_true(stage->tensor(1,false,1,0) == nullptr);
        t.assert_true(stage->tensor(0,true,4,0) == nullptr);
        t.assert_true(stage->tensor(0,true,1,2) == nullptr);
    });

    t.test("invalid_geometry_and_overflow_are_transactional", [](testing & t) {
        llama_recurrent_spill_layout layout;
        const std::vector<llama_recurrent_spill_layer> good{{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}};
        if (!t.assert_true(llama_recurrent_spill_layout_make(good,1,2,layout))) return;
        const auto bytes=layout.host_snapshot_bytes;
        t.assert_true(!llama_recurrent_spill_layout_make(good,UINT32_MAX,UINT32_MAX,layout));
        t.assert_equal(bytes,layout.host_snapshot_bytes);
        const std::vector<llama_recurrent_spill_layer> bad{{GGML_TYPE_Q8_0,GGML_TYPE_F32,1,8,true}};
        t.assert_true(!llama_recurrent_spill_layout_make(bad,1,2,layout));
        t.assert_equal(bytes,layout.host_snapshot_bytes);
    });
    t.test("host_bank_publishes_and_restores_exact_selected_state", [](testing & t) {
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,2,layout))) return;
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),7,0);
        if (!t.assert_true(bool(bank))) return;
        t.assert_equal(layout.host_snapshot_bytes,bank->host_bytes());
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        const float expected_r[4]={1,2,3,4};
        t.assert_true(bank->snapshot_pristine(0,false,1));
        t.assert_true(bank->snapshot_pristine(0,true,2));
        const float expected_s[8]={9,8,7,6,5,4,3,2};
        ggml_backend_tensor_set(r,expected_r,0,sizeof(expected_r));
        ggml_backend_tensor_set(s,expected_s,0,sizeof(expected_s));
        if (!t.assert_true(bank->begin_capture({7,0,1}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->capture(backend.get(),0,true,1,s,0) && bank->publish_capture())) return;
        t.assert_true(!bank->begin_restore({7,0,1},1));
        if (!t.assert_true(bank->complete_capture() && bank->begin_restore({7,0,1},1))) return;
        t.assert_true(bank->restore(backend.get(),0,false,dr,0));
        t.assert_true(bank->restore(backend.get(),0,true,ds,0));
        if (!t.assert_true(bank->publish_restore() && bank->complete_restore())) return;
        float actual_r[4]={},actual_s[8]={};
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        t.assert_true(std::memcmp(actual_r,expected_r,sizeof(expected_r)) == 0);
        t.assert_true(std::memcmp(actual_s,expected_s,sizeof(expected_s)) == 0);
        t.assert_equal(uint64_t(1),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        t.assert_true(!bank->begin_capture({7,1,2}));
        t.assert_true(bank->begin_capture({7,0,2}));
        t.assert_true(bank->capture(backend.get(),0,false,1,r,0));
        bank->cancel();
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        bank->reset(false);
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        bank->reset(true);
        t.assert_true(bank->snapshot_pristine(0,false,1));
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(bank->snapshot_pristine(0,false,2));
        if (!t.assert_true(bank->begin_restore({7,0,1},1) &&
                bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        const float zero_r[4]={},zero_s[8]={};
        t.assert_true(std::memcmp(actual_r,zero_r,sizeof(actual_r)) == 0);
        t.assert_true(std::memcmp(actual_s,zero_s,sizeof(actual_s)) == 0);

    });

    t.test("mixed_generation_restore_keeps_unchanged_recurrent_plane", [](testing & t) {
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,4,true}},1,2,layout))) return;
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),19,0);
        if (!t.assert_true(bool(bank))) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        const float first_r[4]={1,2,3,4}, first_s[4]={5,6,7,8};
        const float second_r[4]={9,10,11,12};
        ggml_backend_tensor_set(r,first_r,0,sizeof(first_r));
        ggml_backend_tensor_set(s,first_s,0,sizeof(first_s));
        if (!t.assert_true(bank->begin_capture({19,0,1}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->capture(backend.get(),0,true,1,s,0) &&
                bank->publish_capture() && bank->complete_capture())) return;
        ggml_backend_tensor_set(r,second_r,0,sizeof(second_r));
        if (!t.assert_true(bank->begin_capture({19,0,2}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->publish_capture() && bank->complete_capture())) return;
        if (!t.assert_true(bank->begin_restore({19,0,2},1))) return;
        if (!t.assert_true(bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        float actual_r[4]={},actual_s[4]={};
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        t.assert_true(std::memcmp(actual_r,second_r,sizeof(actual_r)) == 0);
        t.assert_true(std::memcmp(actual_s,first_s,sizeof(actual_s)) == 0);
    });


    t.test("cpu_graph_publishes_recurrent_snapshots_through_two_slot_stage", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),31,0);
        if (!t.assert_true(bool(bank))) return;
        if (!t.assert_true(bank->enable_publication(ggml_backend_get_device(backend.get()),
                    ggml_backend_cpu_buffer_type(),2))) return;
        auto * dst_r=bank->publication_tensor(0,false,1);
        auto * dst_s=bank->publication_tensor(0,true,1);
        auto * stage_r=bank->staging_tensor(0,false,1);
        auto * stage_s=bank->staging_tensor(0,true,1);
        if (!t.assert_true(dst_r && dst_s && stage_r && stage_s &&
                bank->staged_publication_enabled())) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * src_r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * src_s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(source))) return;
        const float expected_r[4]={1,2,3,4};
        const float expected_s[8]={8,7,6,5,4,3,2,1};
        ggml_backend_tensor_set(src_r,expected_r,0,sizeof(expected_r));
        ggml_backend_tensor_set(src_s,expected_s,0,sizeof(expected_s));
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        auto * copied_r=ggml_cpy(ctx.get(),src_r,stage_r);
        auto * copied_s=ggml_cpy(ctx.get(),src_s,stage_s);
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),copied_r,dst_r));
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),copied_s,dst_s));
        if (!t.assert_true(bank->begin_staged_capture({31,0,1},2))) return;
        if (!t.assert_true(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        const auto * actual_r=bank->snapshot_data(0,false,1,1);
        const auto * actual_s=bank->snapshot_data(0,true,1,1);
        t.assert_true(actual_r && !std::memcmp(actual_r,expected_r,sizeof(expected_r)));
        t.assert_true(actual_s && !std::memcmp(actual_s,expected_s,sizeof(expected_s)));
        auto * partial=ggml_new_graph_custom(ctx.get(),16,false);
        ggml_build_forward_expand(partial,ggml_cpy(ctx.get(),copied_r,dst_r));
        if (!t.assert_true(bank->begin_staged_capture({31,0,2},2) &&
                ggml_backend_graph_compute(backend.get(),partial) == GGML_STATUS_SUCCESS)) return;
        t.assert_true(!bank->publish_capture());
        bank->cancel();
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        if (!t.assert_true(bank->begin_staged_capture({31,0,2},2) &&
                ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        t.assert_true(bank->snapshot_data(0,false,1,2) != nullptr);
        t.assert_true(bank->snapshot_data(0,true,1,2) != nullptr);
    });

    t.test("three_layers_reuse_two_slots_only_after_d2h_consumption", [&](testing & t) {
        ggml_backend_load_all();
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        auto * host_type=ggml_backend_dev_host_buffer_type(device);
        if (!host_type) host_type=ggml_backend_cpu_buffer_type();
        llama_recurrent_spill_layout layout;
        std::vector<llama_recurrent_spill_layer> layers(
            3,{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true});
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host_type,33,0);
        if (!t.assert_true(bool(bank) &&
                bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        std::array<ggml_tensor *,3> r{},s{};
        for (size_t layer=0;layer<3;++layer) {
            r[layer]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,4,1);
            s[layer]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,8,1);
        }
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(source))) return;
        std::array<std::array<float,4>,3> expected_r{};
        std::array<std::array<float,8>,3> expected_s{};
        auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
        for (size_t layer=0;layer<3;++layer) {
            for (size_t i=0;i<4;++i) expected_r[layer][i]=float(100*layer+i);
            for (size_t i=0;i<8;++i) expected_s[layer][i]=float(-100*layer-int(i));
            ggml_backend_tensor_set(r[layer],expected_r[layer].data(),0,ggml_nbytes(r[layer]));
            ggml_backend_tensor_set(s[layer],expected_s[layer].data(),0,ggml_nbytes(s[layer]));
            auto * staged_r=ggml_cpy(ctx.get(),r[layer],bank->staging_tensor(layer,false,1));
            auto * staged_s=ggml_cpy(ctx.get(),s[layer],bank->staging_tensor(layer,true,1));
            ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged_r,
                bank->publication_tensor(layer,false,1)));
            ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged_s,
                bank->publication_tensor(layer,true,1)));
        }
        if (!t.assert_true(bank->begin_staged_capture({33,0,1},2) &&
                ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        for (size_t layer=0;layer<3;++layer) {
            const auto * actual_r=bank->snapshot_data(layer,false,1,1);
            const auto * actual_s=bank->snapshot_data(layer,true,1,1);
            t.assert_true(actual_r && !std::memcmp(actual_r,expected_r[layer].data(),ggml_nbytes(r[layer])));
            t.assert_true(actual_s && !std::memcmp(actual_s,expected_s[layer].data(),ggml_nbytes(s[layer])));
        }
    });

    if (argc > 1 && !std::strcmp(argv[1],"--cuda")) t.test(
        "cuda_graph_publishes_staged_snapshots_without_host_graph_sync", [](testing & t) {
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,1024,2048,true}},1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(
            layout,ggml_backend_dev_host_buffer_type(device),32,0);
        if (!t.assert_true(bool(bank) &&
                bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * src_r=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1024,1);
        auto * src_s=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2048,1);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(source))) return;
        std::vector<float> expected_r(1024),expected_s(2048);
        for (size_t i=0;i<expected_r.size();++i) expected_r[i]=float(i)*.125f;
        for (size_t i=0;i<expected_s.size();++i) expected_s[i]=float(i)*-.0625f;
        ggml_backend_tensor_set(src_r,expected_r.data(),0,ggml_nbytes(src_r));
        ggml_backend_tensor_set(src_s,expected_s.data(),0,ggml_nbytes(src_s));
        auto * write_r=ggml_cpy(ctx.get(),src_r,bank->staging_tensor(0,false,1));
        auto * write_s=ggml_cpy(ctx.get(),src_s,bank->staging_tensor(0,true,1));
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),write_r,bank->publication_tensor(0,false,1)));
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),write_s,bank->publication_tensor(0,true,1)));
        if (!t.assert_true(bank->begin_staged_capture({32,0,1},2))) return;
        if (!t.assert_true(ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready())) return;
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        if (!t.assert_true(bank->complete_capture())) return;
        const auto * actual_r=bank->snapshot_data(0,false,1,1);
        const auto * actual_s=bank->snapshot_data(0,true,1,1);
        t.assert_true(actual_r && !std::memcmp(actual_r,expected_r.data(),ggml_nbytes(src_r)));
        t.assert_true(actual_s && !std::memcmp(actual_s,expected_s.data(),ggml_nbytes(src_s)));
    });

    if (argc > 1 && !std::strcmp(argv[1],"--cuda")) t.test("cuda_pinned_host_capture_and_restore_keep_publication_order", [](testing & t) {
        ggml_backend_load_all();
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto * host_type=ggml_backend_dev_host_buffer_type(dev);
        if (!t.assert_true(host_type && ggml_backend_buft_is_host(host_type))) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,1024,2048,true}},1,2,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host_type,8,0);
        if (!t.assert_true(bool(bank))) return;
        auto stage=llama_recurrent_stage_buffer::create(layout,2,ggml_backend_dev_buffer_type(dev));
        if (!t.assert_true(bool(stage))) return;
        t.assert_true(!ggml_backend_buffer_is_host(stage->buffer()));
        t.assert_equal(stage->layout().total_bytes,stage->bytes());
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1024);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,2048);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1024);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,2048);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        std::vector<float> expected_r(1024),expected_s(2048),actual_r(1024),actual_s(2048);
        for (size_t i=0;i<expected_r.size();++i) expected_r[i]=float(i)*.125f;
        for (size_t i=0;i<expected_s.size();++i) expected_s[i]=float(i)*-.0625f;
        ggml_backend_tensor_set(r,expected_r.data(),0,ggml_nbytes(r));
        ggml_backend_tensor_set(s,expected_s.data(),0,ggml_nbytes(s));
        if (!t.assert_true(bank->begin_capture({8,0,1}) &&
                bank->capture(backend.get(),0,false,2,r,0) &&
                bank->capture(backend.get(),0,true,2,s,0) && bank->publish_capture())) return;
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,2));
        if (!t.assert_true(bank->complete_capture() && bank->begin_restore({8,0,1},2))) return;
        if (!t.assert_true(bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        ggml_backend_tensor_get(dr,actual_r.data(),0,ggml_nbytes(dr));
        ggml_backend_tensor_get(ds,actual_s.data(),0,ggml_nbytes(ds));
        t.assert_true(std::memcmp(expected_r.data(),actual_r.data(),ggml_nbytes(r)) == 0);
        t.assert_true(std::memcmp(expected_s.data(),actual_s.data(),ggml_nbytes(s)) == 0);
    });

    return t.summary();
}

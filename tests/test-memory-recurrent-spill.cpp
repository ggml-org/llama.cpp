#include "../src/llama-memory-recurrent-spill.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include <cstring>

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

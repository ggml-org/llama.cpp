#include "../ggml/src/ggml-cuda/fattn-tile-resume.h"
#include "ggml-cpp.h"
#include "testing.h"

#include <cstring>

int main() {
    testing t;
    const ggml_cuda_fattn_tile_resume_geometry geometry{256,4,4,2,2,1,128,64,3,8};
    t.test("tile_resume_layout_preserves_accumulator_width_and_bounds",[&](testing & t) {
        ggml_cuda_fattn_tile_resume_layout plan;
        if (!t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(geometry,529,plan))) return;
        t.assert_equal(size_t(40),plan.thread_bytes);
        t.assert_equal(size_t(61440),plan.state_bytes);
        t.assert_equal(size_t(256),plan.state_offset);
        t.assert_equal(size_t(94720),plan.bytes);
        auto half = geometry; half.acc_bytes = 4;
        ggml_cuda_fattn_tile_resume_layout h;
        t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(half,529,h));
        t.assert_equal(size_t(24),h.thread_bytes);
        t.assert_equal(size_t(36864),h.state_bytes);
        auto saved = plan;
        for (int kind = 0; kind < 7; ++kind) {
            auto bad = geometry;
            if (kind == 0) bad.nthreads = 96;
            if (kind == 1) bad.ncols1 = 3;
            if (kind == 2) bad.acc_bytes = 2;
            if (kind == 3) bad.query_heads = 3;
            if (kind == 4) bad.nbatch_fa = 31;
            if (kind == 5) bad.queries = 5;
            if (kind == 6) bad.splits = UINT32_MAX;
            t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(bad,529,plan));
            t.assert_true(std::memcmp(&saved,&plan,sizeof(plan)) == 0);
        }
        t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(geometry,0,plan));
        t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(geometry,size_t(INT32_MAX)+1,plan));
        auto grouped = geometry; grouped.ncols2 = 2;
        t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(grouped,529,plan));
    });
    t.test("wave_admission_is_transactional_and_preserves_logical_coordinates",[&](testing & t) {
        ggml_cuda_fattn_tile_resume_layout plan;
        if (!t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(geometry,529,plan))) return;
        ggml_backend_buffer_ptr input(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),131072));
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),plan.bytes));
        ggml_kv_stream_span spans[2] = {{input.get(),input.get(),64,17,128,32768},
            {input.get(),input.get(),81,47,9216,41984}};
        ggml_kv_stream_span_plan_view view;
        view.shape = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,4,256,128};
        view.spans = spans; view.count = 2; view.active_tokens = 529; view.query_tokens = 2;
        ggml_cuda_fattn_tile_resume_descriptor descriptor;
        t.assert_true(ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),64,128,descriptor));
        t.assert_equal(64,descriptor.control.first); t.assert_equal(128,descriptor.control.end);
        t.assert_equal(529,descriptor.table.tokens);
        t.assert_true(!descriptor.control.reset && !descriptor.control.finish);
        const auto saved = descriptor;
        auto reject = [&](size_t first,size_t end,ggml_backend_buffer_t workspace) {
            t.assert_true(!ggml_cuda_fattn_tile_resume_wave_make(view,plan,workspace,first,end,descriptor));
            t.assert_true(std::memcmp(&saved,&descriptor,sizeof(descriptor)) == 0);
        };
        reject(65,128,scratch.get()); reject(64,127,scratch.get()); reject(64,64,scratch.get());
        reject(128,64,scratch.get()); reject(64,530,scratch.get());
        ggml_backend_buffer_ptr short_scratch(ggml_backend_buffer_view(scratch.get(),0,plan.bytes-1));
        reject(64,128,short_scratch.get()); reject(64,128,nullptr);
        spans[1].token_begin = 82; reject(64,128,scratch.get()); spans[1].token_begin = 81;
        auto forged = plan; forged.state_bytes -= 4;
        t.assert_true(!ggml_cuda_fattn_tile_resume_wave_make(view,forged,scratch.get(),64,128,descriptor));
        spans[0] = {scratch.get(),scratch.get(),64,64,0,0}; view.count = 1;
        reject(64,128,scratch.get());
        spans[0] = {input.get(),input.get(),512,17,128,32768};
        t.assert_true(ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),512,529,descriptor));
        t.assert_true(descriptor.control.finish && !descriptor.control.reset);
        spans[0] = {input.get(),input.get(),0,64,128,32768};
        t.assert_true(ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),0,64,descriptor));
        t.assert_true(descriptor.control.reset && !descriptor.control.finish);
    });
    t.test("masked_padding_keeps_native_extent_without_extending_physical_grants",[&](testing & t) {
        auto grouped=geometry; grouped.kv_heads=2; grouped.ncols1=2; grouped.ncols2=2;
        ggml_cuda_fattn_tile_resume_layout plan;
        if (!t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(grouped,768,plan))) return;
        ggml_backend_buffer_ptr input(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),2*1024*1024));
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),plan.bytes));
        ggml_kv_stream_span source{input.get(),input.get(),0,529,0,1024*1024};
        ggml_kv_stream_span_plan_view view{{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,2,256,128},&source,1,529,2};
        ggml_cuda_fattn_tile_resume_descriptor descriptor;
        t.assert_true(!ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),0,768,descriptor));
        t.assert_true(ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),0,768,descriptor,true));
        t.assert_equal(768,descriptor.table.tokens);
        t.assert_equal(529,descriptor.table.spans[0].tokens);
        t.assert_true(descriptor.control.reset && descriptor.control.finish);
        source.tokens=528;
        t.assert_true(!ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),0,768,descriptor,true));
        source.tokens=529;
        t.assert_true(!ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),0,704,descriptor,true));
        t.assert_true(!ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),576,768,descriptor,true));
        view.active_tokens=512; source.tokens=512;
        t.assert_true(!ggml_cuda_fattn_tile_resume_wave_make(view,plan,scratch.get(),0,768,descriptor,true));
    });
    t.test("read_completion_and_publication_are_separate_transactions",[&](testing & t) {
        ggml_cuda_fattn_tile_resume_cursor empty(0,64), invalid_tile(529,0);
        t.assert_true(!empty.ready_to_publish() && !empty.publish(true));
        t.assert_true(!invalid_tile.begin(529) && !invalid_tile.ready_to_publish());
        ggml_cuda_fattn_tile_resume_cursor cursor(529,64);
        t.assert_true(!cursor.begin(63)); t.assert_true(cursor.begin(64));
        t.assert_true(!cursor.can_reuse() && !cursor.begin(128) && !cursor.publish(true));
        t.assert_equal(size_t(0),cursor.next());
        t.assert_true(cursor.complete(true)); t.assert_true(cursor.can_reuse());
        t.assert_equal(size_t(64),cursor.next());
        t.assert_true(cursor.begin(529)); t.assert_true(cursor.complete(true));
        t.assert_true(cursor.ready_to_publish()); t.assert_true(cursor.publish(true));
        t.assert_true(!cursor.publish(true) && !cursor.begin(529));
        ggml_cuda_fattn_tile_resume_cursor cancelled(529,64);
        t.assert_true(cancelled.begin(64)); cancelled.cancel();
        t.assert_true(!cancelled.can_reuse()); t.assert_true(!cancelled.complete(true));
        t.assert_true(cancelled.can_reuse() && !cancelled.publish(true) && !cancelled.begin(128));
        ggml_cuda_fattn_tile_resume_cursor failure(529,64);
        t.assert_true(failure.begin(64)); t.assert_true(!failure.complete(false));
        t.assert_true(failure.can_reuse() && !failure.begin(128));
        ggml_cuda_fattn_tile_resume_cursor publication_failure(529,64);
        t.assert_true(publication_failure.begin(529)); t.assert_true(publication_failure.complete(true));
        t.assert_true(!publication_failure.publish(false) && !publication_failure.publish(true));
    });
    return t.summary();
}

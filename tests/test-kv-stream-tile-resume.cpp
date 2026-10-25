#include "../ggml/src/ggml-cuda/fattn-tile-spans.h"
#include "../ggml/src/ggml-cuda/fattn-mma-resume.h"
#include "ggml-cpp.h"
#include "testing.h"

#include <cstring>
#include <algorithm>
#include <vector>

int main() {
    testing t;
    t.test("tile_split_scratch_bound_covers_native_search_at_every_history", [&](testing & t) {
        for (uint32_t occupancy : {1u,2u,4u,8u}) for (uint32_t sm : {1u,2u,17u,70u})
            for (uint32_t destinations : {4u,8u,32u,128u,512u,4096u}) {
            const uint32_t wave=sm*occupancy;
            const auto bound=std::max(occupancy,ggml_cuda_fattn_tile_resume_splits(wave,destinations,occupancy,wave));
            for (uint32_t tiles=1;tiles<=4096;tiles+=tiles < 64 ? 1 : 31) {
                uint32_t expected=std::min(tiles,occupancy), best=0;
                uint64_t best_waves=0;
                for (uint32_t candidate=expected;candidate<=tiles;++candidate) {
                    const uint64_t blocks=uint64_t(destinations)*candidate, waves=(blocks+wave-1)/wave;
                    const auto efficiency=uint32_t(100*blocks/(waves*wave));
                    if (best >= 95 && waves > best_waves) break;
                    if (efficiency > best) { best=efficiency; best_waves=waves; expected=candidate; }
                }
                const auto actual=ggml_cuda_fattn_tile_resume_splits(tiles,destinations,occupancy,wave);
                t.assert_equal(expected,actual);
                t.assert_true(actual <= bound);
            }
        }
        t.assert_equal(0u,ggml_cuda_fattn_tile_resume_splits(0,1,1,1));
        t.assert_equal(0u,ggml_cuda_fattn_tile_resume_splits(1,0,1,1));
        t.assert_equal(0u,ggml_cuda_fattn_tile_resume_splits(1,1,0,1));
        t.assert_equal(0u,ggml_cuda_fattn_tile_resume_splits(1,1,1,0));
        t.assert_equal(0u,ggml_cuda_fattn_tile_resume_splits(1,1,2,1));
        t.assert_equal(0u,ggml_cuda_fattn_tile_resume_splits(UINT32_MAX,UINT32_MAX,1,1));
    });
    t.test("tile_prefill_checkpoints_stay_bounded_by_query_tiles", [&](testing & t) {
        for (uint32_t width : {256u,512u}) for (uint32_t queries : {9u,16u,17u,31u,128u,129u,255u,256u}) {
            ggml_cuda_fattn_tile_resume_geometry g{width,32,4,queries,width == 512 ? 2u : 4u,8,128,64,7,4};
            ggml_cuda_fattn_tile_resume_layout short_history, long_history;
            if (!t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(g,4096,short_history))) continue;
            t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(g,262144,long_history));
            t.assert_equal(short_history.bytes,long_history.bytes);
            t.assert_equal(size_t((queries+g.ncols1-1)/g.ncols1)*(32/8)*7*128*short_history.thread_bytes,
                short_history.state_bytes);
            const auto saved=long_history;
            g.queries=UINT32_MAX;
            t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(g,262144,long_history));
            t.assert_true(std::memcmp(&saved,&long_history,sizeof(saved)) == 0);
        }
        ggml_cuda_fattn_tile_resume_geometry invalid{512,32,4,4,4,8,128,64,7,4};
        ggml_cuda_fattn_tile_resume_layout untouched;
        t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(invalid,4096,untouched));
    });
    t.test("wide_mma_and_tile_checkpoints_quote_native_accumulator_bytes", [&](testing & t) {
        for (uint32_t columns : {1u,2u,4u,8u}) for (uint32_t groups : {2u,8u}) {
            if (columns < 4 && groups != 8) continue;
            ggml_cuda_fattn_mma_resume_layout narrow,wide;
            if (!t.assert_true(ggml_cuda_fattn_mma_resume_layout_make(12,17,128,columns,groups,narrow,41,true))) return;
            if (!t.assert_true(ggml_cuda_fattn_mma_resume_layout_make(12,17,128,columns,groups,wide,41,true,512))) return;
            t.assert_equal(narrow.slots,wide.slots);
            t.assert_equal(narrow.map_entries,wide.map_entries);
            t.assert_equal(columns*groups <= 32 ? size_t(272) : size_t(528),wide.thread_bytes);
            t.assert_equal(wide.slots*128*wide.thread_bytes,wide.bytes);
            const auto saved=wide;
            for (uint32_t invalid : {0u,255u,UINT32_MAX}) {
                t.assert_true(!ggml_cuda_fattn_mma_resume_layout_make(12,17,128,columns,groups,wide,41,true,invalid));
                t.assert_true(std::memcmp(&saved,&wide,sizeof(wide)) == 0);
            }
        }
        ggml_cuda_fattn_tile_resume_geometry g{512,32,4,4,1,8,64,64,17,4};
        ggml_cuda_fattn_tile_resume_layout tile;
        if (!t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(g,8192,tile))) return;
        t.assert_equal(size_t(4*(8+8*4)),tile.thread_bytes);
        t.assert_equal(size_t(512*32*4*4),tile.output_bytes);
        t.assert_true(tile.state_offset+tile.state_bytes <= tile.partial_offset);
        t.assert_true(tile.output_offset+tile.output_bytes <= tile.bytes);
    });
    t.test("mma_compact_checkpoints_only_store_multitile_native_slices",[&](testing & t) {
        for (size_t destinations : {1u,2u,7u}) for (size_t tiles : {1u,2u,7u,41u})
            for (size_t blocks : {1u,3u,17u,64u}) {
                if (blocks > destinations*tiles) continue;
                ggml_cuda_fattn_mma_resume_layout layout;
                t.assert_true(ggml_cuda_fattn_mma_resume_layout_make(destinations,blocks,128,4,8,layout,tiles,true));
                std::vector<uint32_t> expected(destinations+blocks-1,UINT32_MAX);
                size_t slots=0;
                for (size_t block=0;block<blocks;++block) {
                    const size_t begin=block*destinations*tiles/blocks, stop=(block+1)*destinations*tiles/blocks;
                    for (size_t d=begin/tiles;d<=(stop-1)/tiles;++d) {
                        const size_t first=std::max(begin,d*tiles), end=std::min(stop,(d+1)*tiles);
                        if (end-first > 1) expected[d+block]=uint32_t(slots++);
                    }
                }
                t.assert_equal(slots,layout.slots);
                t.assert_equal(slots*128*272,layout.bytes);
                const bool uniform=blocks%destinations == 0 && destinations*tiles/blocks >= 2;
                t.assert_equal(uniform,layout.block_indexed);
                t.assert_equal(slots && slots != expected.size() && !uniform ? expected.size() : size_t(0),layout.map_entries);
                std::vector<uint32_t> actual(expected.size(),0xa5a5a5a5);
                if (slots) {
                    t.assert_true(!ggml_cuda_fattn_mma_resume_state_map(destinations,blocks,tiles,actual.data(),actual.size()-1));
                    t.assert_true(std::all_of(actual.begin(),actual.end(),[](uint32_t x){return x == 0xa5a5a5a5;}));
                    t.assert_true(ggml_cuda_fattn_mma_resume_state_map(destinations,blocks,tiles,actual.data(),actual.size()));
                    t.assert_true(actual == expected);
                }
            }
        ggml_cuda_fattn_mma_resume_layout invalid;
        t.assert_true(!ggml_cuda_fattn_mma_resume_layout_make(INT32_MAX,3,128,4,8,invalid,2,true));
        t.assert_true(!ggml_cuda_fattn_mma_resume_layout_make(2,5,128,4,8,invalid,2,true));
        t.assert_true(!ggml_cuda_fattn_mma_resume_layout_make(2,3,128,4,8,invalid,0,true));
        uint32_t sentinel=123;
        t.assert_true(!ggml_cuda_fattn_mma_resume_state_map(SIZE_MAX,3,2,&sentinel,1));
        t.assert_equal(uint32_t(123),sentinel);
    });
    t.test("mma_active_wave_blocks_preserve_original_stream_k_intersections",[&](testing & t) {
        for (int destinations : {1,2,7}) for (int tiles : {1,7,41}) for (int blocks : {1,3,17}) {
            if (blocks > destinations*tiles) continue;
            for (int first=0;first<tiles;++first) for (int end=first+1;end<=tiles;++end)
                for (int block=0;block<blocks;++block) {
                    const int begin=block*destinations*tiles/blocks, stop=(block+1)*destinations*tiles/blocks;
                    bool oracle=false;
                    for (int tile=begin;tile<stop;++tile) oracle|=tile%tiles >= first && tile%tiles < end;
                    t.assert_equal(oracle,ggml_cuda_fattn_mma_resume_block_active(destinations,tiles,blocks,block,first,end));
                }
        }
        t.assert_true(!ggml_cuda_fattn_mma_resume_block_active(0,41,3,0,0,1));
        t.assert_true(!ggml_cuda_fattn_mma_resume_block_active(7,41,3,3,0,1));
        t.assert_true(!ggml_cuda_fattn_mma_resume_block_active(INT32_MAX,41,3,0,0,1));
        t.assert_true(!ggml_cuda_fattn_mma_resume_block_active(7,41,3,0,4,4));
    });
    t.test("mma_resume_layout_keeps_stream_k_slices_distinct_and_bounded",[&](testing & t) {
        ggml_cuda_fattn_mma_resume_layout layout;
        t.assert_true(ggml_cuda_fattn_mma_resume_layout_make(12,17,128,8,8,layout));
        t.assert_equal(size_t(272),layout.thread_bytes);
        t.assert_equal(size_t(28),layout.slots);
        t.assert_equal(size_t(974848),layout.bytes);
        ggml_cuda_fattn_mma_resume_layout singleton;
        t.assert_true(ggml_cuda_fattn_mma_resume_layout_make(2,4,128,4,8,singleton,2));
        t.assert_equal(size_t(0),singleton.bytes);
        t.assert_true(ggml_cuda_fattn_mma_resume_layout_make(2,4,128,4,8,singleton,3));
        t.assert_true(singleton.bytes > 0);
        const auto saved=layout;
        for (int kind=0;kind<7;++kind) {
            t.assert_true(!ggml_cuda_fattn_mma_resume_layout_make(
                kind == 0 ? 0 : kind == 1 ? SIZE_MAX : 12,
                kind == 2 ? 0 : 17, kind == 3 ? 33 : 128,
                kind == 4 ? 3 : 8, kind == 5 ? 4 : kind == 6 ? 0 : 8,layout));
            t.assert_true(std::memcmp(&saved,&layout,sizeof(layout)) == 0);
        }
        for (size_t destinations : {1u,3u,12u}) for (size_t blocks : {1u,7u,17u}) {
            constexpr size_t tiles=41;
            if (blocks > destinations*tiles) continue;
            t.assert_true(ggml_cuda_fattn_mma_resume_layout_make(destinations,blocks,128,4,2,layout));
            std::vector<bool> used(layout.slots,false);
            for (size_t block=0;block<blocks;++block) {
                const size_t first=block*destinations*tiles/blocks;
                const size_t end=(block+1)*destinations*tiles/blocks;
                for (size_t tile=first/tiles;tile<=(end-1)/tiles;++tile) {
                    const size_t slot=tile+block;
                    t.assert_true(slot < layout.slots && !used[slot]);
                    used[slot]=true;
                }
            }
        }
    });
    t.test("mma_resume_waves_validate_logical_tiles_without_reading_padding",[&](testing & t) {
        ggml_cuda_kv_span sources[3]={{reinterpret_cast<const char *>(uintptr_t(4096)),reinterpret_cast<const char *>(uintptr_t(8192)),
            256,17,128,64,64,32},
            {reinterpret_cast<const char *>(uintptr_t(12288)),reinterpret_cast<const char *>(uintptr_t(16384)),
            273,47,128,64,64,32}};
        const auto valid=[&](size_t first,size_t end,size_t active,size_t capacity) {
            return ggml_cuda_fattn_mma_resume_wave_validate(sources,2,active,capacity,64,first,end,2,64,32);
        };
        t.assert_true(valid(256,320,529,768));
        t.assert_true(!valid(255,320,529,768));
        t.assert_true(!valid(256,319,529,768));
        t.assert_true(!valid(256,256,529,768));
        t.assert_true(!valid(256,320,529,1024));
        sources[0].first=512; sources[1].first=529;
        sources[1].tokens=0;
        t.assert_true(ggml_cuda_fattn_mma_resume_wave_validate(sources,1,529,768,64,512,768,2,64,32));
        t.assert_true(!ggml_cuda_fattn_mma_resume_wave_validate(sources,1,529,768,64,512,704,2,64,32));
        t.assert_true(!ggml_cuda_fattn_mma_resume_wave_validate(sources,1,SIZE_MAX,SIZE_MAX,64,512,768,2,64,32));
        sources[0].tokens=16;
        t.assert_true(!ggml_cuda_fattn_mma_resume_wave_validate(sources,1,529,768,64,512,768,2,64,32));
    });
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
            if (kind == 5) bad.queries = 0;
            if (kind == 6) bad.splits = UINT32_MAX;
            t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(bad,529,plan));
            t.assert_true(std::memcmp(&saved,&plan,sizeof(plan)) == 0);
        }
        t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(geometry,0,plan));
        t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(geometry,size_t(INT32_MAX)+1,plan));
        auto grouped = geometry; grouped.ncols2 = 2;
        t.assert_true(!ggml_cuda_fattn_tile_resume_layout_make(grouped,529,plan));
    });
    t.test("wide_tile_resume_keeps_stock_query_tiles_and_checked_state",[&](testing & t) {
        for (uint32_t queries : {5u,6u,7u,8u}) {
            auto wide = geometry;
            wide.kv_heads = 2;
            wide.queries = queries;
            wide.ncols1 = 8;
            wide.ncols2 = 2;
            ggml_cuda_fattn_tile_resume_layout plan;
            t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(wide,768,plan));
            t.assert_equal(size_t(122880),plan.state_bytes);
            wide.query_heads = 32;
            wide.kv_heads = 4;
            wide.ncols1 = 4;
            wide.ncols2 = 8;
            t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(wide,768,plan));
            t.assert_equal(size_t(983040),plan.state_bytes);
        }
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

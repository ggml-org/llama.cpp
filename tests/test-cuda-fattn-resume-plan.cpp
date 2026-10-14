#include "fattn-tile-resume.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    const ggml_cuda_fattn_tile_resume_geometry geometry{256,4,4,2,2,1,128,64,3,8};
    ggml_cuda_fattn_tile_resume_layout plan;
    assert(ggml_cuda_fattn_tile_resume_layout_make(geometry,529,plan));
    assert(plan.thread_bytes == 40 && plan.state_bytes == 61440);
    auto half = geometry; half.acc_bytes = 4;
    ggml_cuda_fattn_tile_resume_layout h;
    assert(ggml_cuda_fattn_tile_resume_layout_make(half,529,h));
    assert(h.thread_bytes == 24 && h.state_bytes == 36864);
    const auto saved = plan;
    for (int kind = 0; kind < 9; ++kind) {
        auto bad = geometry;
        if (kind == 0) bad.nthreads = 96;
        if (kind == 1) bad.ncols1 = 3;
        if (kind == 2) bad.acc_bytes = 2;
        if (kind == 3) bad.query_heads = 3;
        if (kind == 4) bad.nbatch_fa = 31;
        if (kind == 5) bad.queries = 9;
        if (kind == 6) bad.splits = UINT32_MAX;
        if (kind == 7) bad.width = 0;
        if (kind == 8) bad.kv_heads = 0;
        assert(!ggml_cuda_fattn_tile_resume_layout_make(bad,529,plan));
        assert(std::memcmp(&saved,&plan,sizeof(plan)) == 0);
    }
    assert(!ggml_cuda_fattn_tile_resume_layout_make(geometry,0,plan));
    assert(!ggml_cuda_fattn_tile_resume_layout_make(geometry,size_t(INT32_MAX)+1,plan));
    auto grouped = geometry; grouped.ncols2 = 2;
    assert(!ggml_cuda_fattn_tile_resume_layout_make(grouped,529,plan));
    alignas(128) char source[131072] = {}, workspace[131072] = {};
    ggml_cuda_fattn_tile_span_table table;
    table.count = 2; table.tokens = 529;
    table.spans[0] = {source,source+65536,128,17,272*4,144*4,272,144};
    table.spans[1] = {source+18496,source+75328,145,47,272*4,144*4,272,144};
    ggml_cuda_fattn_tile_resume_descriptor descriptor;
    assert(ggml_cuda_fattn_tile_resume_wave_make(table,plan,workspace,plan.bytes,272,144,128,192,descriptor));
    assert(descriptor.control.first == 128 && descriptor.control.end == 192 && !descriptor.control.reset && !descriptor.control.finish);
    const auto descriptor_saved = descriptor;
    for (int kind = 0; kind < 8; ++kind) {
        auto bad = table;
        auto bad_layout = plan;
        if (kind == 0) bad.spans[1].first++;
        if (kind == 1) bad.spans[0].k = workspace;
        if (kind == 2) bad_layout.state_bytes--;
        if (kind == 3) bad.spans[1].tokens++;
        if (kind == 4) bad.spans[0].k_token_stride = 271;
        if (kind == 5) bad.count = 4;
        if (kind == 6) bad.tokens++;
        assert(!ggml_cuda_fattn_tile_resume_wave_make(bad,bad_layout,workspace,plan.bytes-(kind == 7),272,144,128,192,descriptor));
        assert(std::memcmp(&descriptor_saved,&descriptor,sizeof(descriptor)) == 0);
    }
    assert(!ggml_cuda_fattn_tile_resume_wave_make(table,plan,workspace,plan.bytes,272,144,129,192,descriptor));
    assert(!ggml_cuda_fattn_tile_resume_wave_make(table,plan,workspace,plan.bytes,272,144,128,191,descriptor));
    ggml_cuda_fattn_tile_resume_cursor cursor(529,128);
    assert(!cursor.begin(127) && cursor.begin(128) && !cursor.can_reuse());
    assert(!cursor.begin(256) && !cursor.publish(true) && cursor.next() == 0);
    assert(cursor.complete(true) && cursor.next() == 128 && cursor.can_reuse());
    assert(cursor.begin(529) && cursor.complete(true) && cursor.publish(true));
    assert(!cursor.publish(true) && !cursor.begin(529));
    ggml_cuda_fattn_tile_resume_cursor cancelled(529,128);
    assert(cancelled.begin(128)); cancelled.cancel();
    assert(!cancelled.can_reuse() && !cancelled.complete(true) && cancelled.can_reuse());
    assert(!cancelled.begin(529) && !cancelled.publish(true));
    ggml_cuda_fattn_tile_resume_cursor failure(529,128);
    assert(failure.begin(529) && !failure.complete(false) && !failure.publish(true));
    ggml_cuda_fattn_tile_resume_cursor publish_failure(529,128);
    assert(publish_failure.begin(529) && publish_failure.complete(true));
    assert(!publish_failure.publish(false) && !publish_failure.publish(true));
    ggml_cuda_fattn_tile_resume_cursor generation(529,128,42);
    assert(!generation.begin(128,41) && generation.begin(128,42));
    assert(!generation.complete(true,41) && !generation.can_reuse() && generation.next() == 0);
    assert(generation.complete(true,42) && generation.begin(529,42) && generation.complete(true,42));
    assert(!generation.publish(true,41) && generation.publish(true,42));
    auto padded_geometry = geometry;
    padded_geometry.ncols2 = 2; padded_geometry.kv_heads = 2;
    ggml_cuda_fattn_tile_resume_layout padded;
    assert(ggml_cuda_fattn_tile_resume_layout_make(padded_geometry,768,padded));
    table.count = 1; table.tokens = 529;
    table.spans[0] = {source,source+65536,0,64,272*2,144*2,272,144};
    assert(!ggml_cuda_fattn_tile_resume_wave_make(table,padded,workspace,padded.bytes,272,144,0,64,descriptor));
    assert(ggml_cuda_fattn_tile_resume_wave_make(table,padded,workspace,padded.bytes,272,144,0,64,descriptor,true));
    table.spans[0].first = 512; table.spans[0].tokens = 17;
    assert(ggml_cuda_fattn_tile_resume_wave_make(table,padded,workspace,padded.bytes,272,144,512,768,descriptor,true));
    assert(descriptor.table.tokens == 768 && descriptor.table.spans[0].tokens == 17 && descriptor.control.finish);
    assert(!ggml_cuda_fattn_tile_resume_wave_make(table,padded,workspace,padded.bytes,272,144,512,576,descriptor,true));
    assert(!ggml_cuda_fattn_tile_resume_wave_make(table,padded,workspace,padded.bytes,272,144,576,768,descriptor,true));
    auto overflowing = table;
    overflowing.spans[0].k = reinterpret_cast<const char *>(UINTPTR_MAX-15);
    assert(!ggml_cuda_fattn_tile_resume_wave_make(overflowing,padded,workspace,padded.bytes,272,144,512,768,descriptor,true));
    std::puts("Resume planning: state sizing, checked waves and completion/publication lifecycle passed");
}

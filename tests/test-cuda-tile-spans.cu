#include "../ggml/src/ggml-cuda/fattn-tile.cuh"
#include "../ggml/src/ggml-cuda/fattn-tile-spans.h"
#include "../ggml/src/ggml-cuda/fattn-tile-span-access.cuh"
#include "../ggml/src/ggml-cuda/fattn-tile-resume.cuh"
#include "ggml-cpp.h"
#include "testing.h"

#include <cstring>
#include <limits>
#include <atomic>
#include <thread>

struct read_gate {
    std::atomic<bool> entered{false}, released{false};
    // This callback holds a real stream dependency and never calls CUDA itself.
    static void CUDART_CB wait(void * opaque) {
        auto & gate = *static_cast<read_gate *>(opaque);
        gate.entered = true;
        while (!gate.released.load()) std::this_thread::yield();
    }
};

enum class resume_fault { none, cancellation, read_completion, publication };

struct attention_comparison {
    bool finite = false, exact = false, accepted = false;
    double max_abs = 0, relative_l2 = 0;
};

// Conversion/canary checks remain exact; this policy applies only to final attention outputs.
static attention_comparison compare_attention(const std::vector<float> & stock, const std::vector<float> & span, int compiled_cc) {
    attention_comparison result;
    if (stock.empty() || stock.size() != span.size()) return result;
    double error_squared = 0, reference_squared = 0;
    for (size_t i = 0; i < stock.size(); ++i) {
        if (!std::isfinite(stock[i]) || !std::isfinite(span[i])) return result;
        const double error = double(stock[i])-double(span[i]);
        result.max_abs = std::max(result.max_abs,std::abs(error));
        error_squared += error*error;
        reference_squared += double(stock[i])*double(stock[i]);
    }
    result.finite = true;
    result.exact = std::memcmp(stock.data(),span.data(),stock.size()*sizeof(float)) == 0;
    result.relative_l2 = reference_squared ? std::sqrt(error_squared/reference_squared) :
        (error_squared ? std::numeric_limits<double>::infinity() : 0);
    // Bound SM61 FP32 rounding noise by both absolute error and whole-output relative error.
    result.accepted = result.exact || (compiled_cc == 610 && result.max_abs <= 1e-8 &&
        result.relative_l2 <= 8*std::numeric_limits<float>::epsilon());
    return result;
}

static void comparison_policy_tests(testing & t) {
    const float base = .01f, adjacent = std::nextafter(base,1.0f);
    t.assert_true(compare_attention({base},{base},1200).accepted);
    t.assert_true(compare_attention({base},{adjacent},610).accepted);
    t.assert_true(!compare_attention({base},{adjacent},1200).accepted);
    t.assert_true(!compare_attention({base},{adjacent},860).accepted);
    t.assert_true(!compare_attention({base},{adjacent},600).accepted);
    t.assert_true(!compare_attention({1.0f},{std::nextafter(1.0f,2.0f)},610).accepted);
    t.assert_true(!compare_attention({1e-9f},{2e-9f},610).accepted);
    t.assert_true(compare_attention({0.0f},{0.0f},610).accepted);
    t.assert_true(!compare_attention({0.0f},{1e-10f},610).accepted);
    t.assert_true(!compare_attention({base},{INFINITY},610).accepted);
    t.assert_true(!compare_attention({INFINITY},{INFINITY},610).accepted);
    t.assert_true(!compare_attention({base},{NAN},610).accepted);
    t.assert_true(!compare_attention({base},{},610).accepted);
    t.assert_true(!compare_attention({},{},610).accepted);
}

static size_t attention_cases = 0, attention_nonexact = 0;
static double attention_max_abs = 0, attention_max_relative_l2 = 0;
static size_t resume_cases = 0, resume_nonexact = 0;
static double resume_max_abs = 0, resume_max_relative_l2 = 0;

// Launch the same stock specialization and split grid for native and span-backed K/V.
template<int D, int columns, int gqa, typename KV>
static void launch_attention(const float * q, const void * k, const void * v, const half * mask, const float * sinks,
        float * output, float2 * meta, int queries, int tokens, int splits, cudaStream_t stream = nullptr) {
    constexpr int heads = gqa > 2 ? 2*gqa : 4, kv_heads = heads/gqa;
    const int compiled = ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
    const int nthreads = ggml_cuda_fattn_tile_get_nthreads(D,D,columns*gqa,compiled);
    const dim3 grid((queries+columns-1)/columns,splits,heads/gqa);
    flash_attn_tile<D,D,columns,gqa,false,KV><<<grid,dim3(32,nthreads/32),0,stream>>>(
        reinterpret_cast<const char *>(q),reinterpret_cast<const char *>(k),reinterpret_cast<const char *>(v),
        reinterpret_cast<const char *>(mask),reinterpret_cast<const char *>(sinks),nullptr,output,meta,
        1.0f/std::sqrt(float(D)),0,1,1,2,0,
        D,init_fastdiv_values(queries),heads,1,D*4,queries*D*4,queries*heads*D*4,
        D,tokens,kv_heads,1,kv_heads*D*2,D*2,int64_t(tokens)*kv_heads*D*2,
        kv_heads*D*2,D*2,int64_t(tokens)*kv_heads*D*2,
        tokens,queries,1,tokens*2,queries*tokens*2,int64_t(queries)*tokens*2);
    CUDA_CHECK(cudaGetLastError());
}

template<int D, int gqa, ggml_type K, ggml_type V, bool resumed = false>
static void attention_case(testing & t, int queries, int tokens, const std::vector<int> & cuts, bool masked, size_t tiles_per_wave = 1,
        resume_fault fault = resume_fault::none) {
    constexpr size_t guard = 128;
    constexpr int heads = gqa > 2 ? 2*gqa : 4, kv_heads = heads/gqa;
    const int splits = tokens < 256 ? 1 : 3;
    std::vector<void *> allocations;
    auto allocate = [&](size_t bytes) {
        void * pointer = nullptr; CUDA_CHECK(cudaMalloc(&pointer,bytes)); allocations.push_back(pointer); return pointer;
    };
    auto encode = [&](ggml_type type, float phase) {
        std::vector<float> values(size_t(tokens)*kv_heads*D);
        for (size_t i = 0; i < values.size(); ++i) values[i] = .2f*std::sin(float(i%401)*.13f+phase);
        std::vector<uint8_t> bytes(size_t(tokens)*kv_heads*ggml_row_size(type,D));
        if (type == GGML_TYPE_F32) std::memcpy(bytes.data(),values.data(),bytes.size());
        else ggml_quantize_chunk(type,values.data(),bytes.data(),0,tokens*kv_heads,D,nullptr);
        return bytes;
    };
    const auto keys = encode(K,.3f), values = encode(V,.7f);
    auto converted = [&](ggml_type type,const std::vector<uint8_t> & bytes) {
        auto * encoded = allocate(bytes.size()); auto * result = static_cast<half *>(allocate(size_t(tokens)*kv_heads*D*2));
        CUDA_CHECK(cudaMemcpy(encoded,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
        if (type == GGML_TYPE_F16) CUDA_CHECK(cudaMemcpy(result,encoded,bytes.size(),cudaMemcpyDeviceToDevice));
        else ggml_get_to_fp16_cuda(type)(encoded,result,size_t(tokens)*kv_heads*D,nullptr);
        return result;
    };
    const half * nk = converted(K,keys), * nv = converted(V,values);
    std::vector<float> query(size_t(queries)*heads*D);
    for (size_t i = 0; i < query.size(); ++i) query[i] = .2f*std::sin(float(i%401)*.13f+.1f);
    auto * q = static_cast<float *>(allocate(query.size()*4));
    CUDA_CHECK(cudaMemcpy(q,query.data(),query.size()*4,cudaMemcpyHostToDevice));
    half * mask = nullptr;
    if (masked) {
        std::vector<ggml_fp16_t> m(size_t(tokens)*queries);
        for (int row = 0; row < queries; ++row) for (int col = 0; col < tokens; ++col)
            m[row*tokens+col] = ggml_fp32_to_fp16(col <= tokens-queries+row && col%7 != 3 ? 0 : -INFINITY);
        mask = static_cast<half *>(allocate(m.size()*2)); CUDA_CHECK(cudaMemcpy(mask,m.data(),m.size()*2,cudaMemcpyHostToDevice));
    }
    ggml_cuda_fattn_tile_span_table table{}; table.count = cuts.size()-1; table.tokens = tokens;
    table.contiguous_k = table.contiguous_v = true;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<ggml_kv_stream_span> spans(table.count);
    struct grant { uint8_t * allocation; size_t bytes; std::vector<uint8_t> expected; };
    std::vector<grant> grants;
    // Reverse physical placement models a wrapped suffix without changing logical iteration order.
    for (int i = table.count-1; i >= 0; --i) {
        const size_t kb = size_t(cuts[i+1]-cuts[i])*kv_heads*ggml_row_size(K,D);
        const size_t vb = size_t(cuts[i+1]-cuts[i])*kv_heads*ggml_row_size(V,D);
        const size_t vo = (kb+127)/128*128;
        buffers.emplace_back(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CUDA0")),vo+vb+2*guard));
        if (!t.assert_true(buffers.back() != nullptr)) return;
        auto * storage = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffers.back().get()));
        std::vector<uint8_t> bytes(vo+vb+2*guard,0xa5);
        std::memcpy(bytes.data()+guard,keys.data()+size_t(cuts[i])*kv_heads*ggml_row_size(K,D),kb);
        std::memcpy(bytes.data()+guard+vo,values.data()+size_t(cuts[i])*kv_heads*ggml_row_size(V,D),vb);
        CUDA_CHECK(cudaMemcpy(storage,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
        spans[i] = {buffers.back().get(),buffers.back().get(),size_t(cuts[i]),size_t(cuts[i+1]-cuts[i]),guard,guard+vo};
        grants.push_back({storage,bytes.size(),std::move(bytes)});
    }
    ggml_kv_stream_span_plan_view view;
    view.shape = {K,V,D,D,kv_heads,256,128}; view.spans = spans.data(); view.count = spans.size();
    view.active_tokens = tokens; view.query_tokens = queries;
    if (!t.assert_true(ggml_cuda_fattn_tile_spans_make(view,table))) return;
    table.contiguous_k = table.contiguous_v = true;
    ggml_cuda_fattn_tile_span_workspace layout;
    if (!t.assert_true(ggml_cuda_fattn_tile_span_workspace_make(heads,queries,D,splits,layout))) return;
    const size_t out_bytes = size_t(heads)*queries*D*4;
    constexpr int modes = resumed ? 4 : 2;
    std::vector<float> results[modes];
    std::vector<float2> metas[modes];
    const int device_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int compiled_cc = ggml_cuda_highest_compiled_arch(device_cc);
    ggml_cuda_fattn_tile_resume_layout resume_layout;
    if constexpr (resumed) {
        const uint32_t columns = queries <= 2 ? 2 : 4;
        const ggml_cuda_fattn_tile_resume_geometry geometry{D,heads,kv_heads,uint32_t(queries),columns,gqa,
            uint32_t(ggml_cuda_fattn_tile_get_nthreads(D,D,columns*gqa,compiled_cc)),
            uint32_t(ggml_cuda_fattn_tile_get_nbatch_fa(D,D,columns*gqa,compiled_cc)),uint32_t(splits),
            uint32_t(fast_fp16_available(device_cc) ? sizeof(half2) : sizeof(float2))};
        if (!t.assert_true(ggml_cuda_fattn_tile_resume_layout_make(geometry,tokens,resume_layout))) return;
    }
    for (int spanned = 0; spanned < modes; ++spanned) {
        const size_t scratch_bytes = spanned < 2 ? layout.bytes : resume_layout.bytes;
        ggml_backend_buffer_ptr scratch_parent(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CUDA0")),scratch_bytes+2*guard));
        auto * scratch = static_cast<uint8_t *>(ggml_backend_buffer_get_base(scratch_parent.get()));
        ggml_backend_buffer_ptr scratch_view(ggml_backend_buffer_view(scratch_parent.get(),guard,scratch_bytes));
        auto * storage = static_cast<uint8_t *>(allocate(out_bytes+2*guard));
        CUDA_CHECK(cudaMemset(scratch,0x5a,scratch_bytes+2*guard)); CUDA_CHECK(cudaMemset(storage,0x5a,out_bytes+2*guard));
        CUDA_CHECK(cudaMemcpy(scratch+guard,&table,sizeof(table),cudaMemcpyHostToDevice));
        auto * output = reinterpret_cast<float *>(storage+guard);
        auto * parts = spanned < 2 ? (splits == 1 ? output : reinterpret_cast<float *>(scratch+guard+layout.partial_offset)) :
            reinterpret_cast<float *>(scratch+guard+resume_layout.partial_offset);
        auto * meta = reinterpret_cast<float2 *>(scratch+guard+(spanned < 2 ? layout.meta_offset : resume_layout.meta_offset));
        float * final_output = spanned < 2 ? output : reinterpret_cast<float *>(scratch+guard+resume_layout.output_offset);
        if constexpr (resumed) if (spanned >= 2) {
            const size_t wave_tokens = spanned == 2 ? size_t(tokens) : resume_layout.geometry.nbatch_fa*tiles_per_wave;
            const size_t capacity = std::min(size_t(tokens),wave_tokens);
            const size_t k_plane = (capacity*kv_heads*ggml_row_size(K,D)+3*128+127)/128*128;
            const size_t ring_bytes = k_plane+capacity*kv_heads*ggml_row_size(V,D)+3*128;
            ggml_backend_buffer_ptr ring(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CUDA0")),ring_bytes+2*guard));
            auto * ring_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(ring.get()));
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            cudaEvent_t read_done;
            CUDA_CHECK(cudaEventCreateWithFlags(&read_done,cudaEventDisableTiming));
            bool injected = false;
            const int attempts = spanned == 3 && fault == resume_fault::publication ? 2 : 1;
            CUDA_CHECK(cudaMemset(scratch+guard+resume_layout.state_offset,0xff,resume_layout.state_bytes));
            for (int attempt = 0; attempt < attempts; ++attempt) {
            ggml_cuda_fattn_tile_resume_cursor cursor(tokens,resume_layout.geometry.nbatch_fa);
            while (cursor.next() < size_t(tokens)) {
                const size_t first = cursor.next(), end = std::min(size_t(tokens),first+wave_tokens);
                CUDA_CHECK(cudaMemset(ring_base,0xa5,ring_bytes+2*guard));
                std::vector<size_t> boundaries{first};
                if (end-first > 2) {boundaries.push_back(first+1); boundaries.push_back(end-1);}
                boundaries.push_back(end);
                std::vector<ggml_kv_stream_span> sources(boundaries.size()-1);
                size_t ko = guard, vo = guard+k_plane;
                for (size_t physical = 0; physical < sources.size(); ++physical) {
                    const size_t index = sources.size()-1-physical;
                    const size_t begin = boundaries[index], count = boundaries[index+1]-begin;
                    const size_t kb = count*kv_heads*ggml_row_size(K,D), vb = count*kv_heads*ggml_row_size(V,D);
                    CUDA_CHECK(cudaMemcpy(ring_base+ko,keys.data()+begin*kv_heads*ggml_row_size(K,D),kb,cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(ring_base+vo,values.data()+begin*kv_heads*ggml_row_size(V,D),vb,cudaMemcpyHostToDevice));
                    sources[index] = {ring.get(),ring.get(),begin,count,ko,vo};
                    ko += (kb+127)/128*128; vo += (vb+127)/128*128;
                }
                auto window = view; window.spans = sources.data(); window.count = sources.size();
                ggml_cuda_fattn_tile_resume_descriptor descriptor;
                if (!t.assert_true(ggml_cuda_fattn_tile_resume_wave_make(window,resume_layout,scratch_view.get(),first,end,descriptor))) return;
                descriptor.table.contiguous_k = descriptor.table.contiguous_v = true;
                if (!t.assert_true(cursor.begin(end))) return;
                CUDA_CHECK(cudaMemcpy(scratch+guard,&descriptor,sizeof(descriptor),cudaMemcpyHostToDevice));
                read_gate gate;
                const bool gated = spanned == 3 && fault != resume_fault::none && !injected && first == 0;
                if (gated) CUDA_CHECK(cudaLaunchHostFunc(stream,read_gate::wait,&gate));
                using reader = ggml_cuda_fattn_tile_resume_kv<K,V>;
                if (queries <= 2) launch_attention<D,2,gqa,reader>(q,scratch+guard,nullptr,mask,nullptr,parts,meta,queries,tokens,splits,stream);
                else launch_attention<D,4,gqa,reader>(q,scratch+guard,nullptr,mask,nullptr,parts,meta,queries,tokens,splits,stream);
                CUDA_CHECK(cudaEventRecord(read_done,stream));
                t.assert_true(!cursor.can_reuse() && !cursor.begin(end));
                if (gated) {
                    const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(2);
                    while (!gate.entered.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
                    t.assert_true(gate.entered.load());
                    t.assert_equal(cudaErrorNotReady,cudaEventQuery(read_done));
                    if (fault == resume_fault::cancellation) cursor.cancel();
                    t.assert_true(!cursor.can_reuse());
                    gate.released = true;
                }
                CUDA_CHECK(cudaEventSynchronize(read_done));
                if (gated && (fault == resume_fault::cancellation || fault == resume_fault::read_completion)) {
                    t.assert_true(!cursor.complete(fault != resume_fault::read_completion) && cursor.can_reuse() && !cursor.publish(true));
                    std::vector<uint8_t> public_bytes(out_bytes);
                    CUDA_CHECK(cudaMemcpy(public_bytes.data(),output,out_bytes,cudaMemcpyDeviceToHost));
                    t.assert_true(std::all_of(public_bytes.begin(),public_bytes.end(),[](uint8_t x){return x==0x5a;}));
                    injected = true;
                    cursor = ggml_cuda_fattn_tile_resume_cursor(tokens,resume_layout.geometry.nbatch_fa);
                    CUDA_CHECK(cudaMemset(scratch+guard+resume_layout.partial_offset,0x5a,scratch_bytes-resume_layout.partial_offset));
                    continue;
                }
                if (!t.assert_true(cursor.complete(true))) return;
                std::vector<uint8_t> public_bytes(out_bytes);
                CUDA_CHECK(cudaMemcpy(public_bytes.data(),output,out_bytes,cudaMemcpyDeviceToHost));
                t.assert_true(std::all_of(public_bytes.begin(),public_bytes.end(),[](uint8_t x){return x==0x5a;}));
                if (end < size_t(tokens)) {
                    std::vector<uint8_t> untouched(scratch_bytes-resume_layout.partial_offset);
                    CUDA_CHECK(cudaMemcpy(untouched.data(),scratch+guard+resume_layout.partial_offset,untouched.size(),cudaMemcpyDeviceToHost));
                    t.assert_true(std::all_of(untouched.begin(),untouched.end(),[](uint8_t x){return x==0x5a;}));
                }
                std::vector<uint8_t> ring_data(ring_bytes+2*guard);
                CUDA_CHECK(cudaMemcpy(ring_data.data(),ring_base,ring_data.size(),cudaMemcpyDeviceToHost));
                t.assert_true(std::all_of(ring_data.begin(),ring_data.begin()+guard,[](uint8_t x){return x==0xa5;}));
                t.assert_true(std::all_of(ring_data.end()-guard,ring_data.end(),[](uint8_t x){return x==0xa5;}));
                for (const auto & source : sources) {
                    const size_t kb = source.tokens*kv_heads*ggml_row_size(K,D), vb = source.tokens*kv_heads*ggml_row_size(V,D);
                    t.assert_true(std::memcmp(ring_data.data()+source.k_offset,keys.data()+source.token_begin*kv_heads*ggml_row_size(K,D),kb) == 0);
                    t.assert_true(std::memcmp(ring_data.data()+source.v_offset,values.data()+source.token_begin*kv_heads*ggml_row_size(V,D),vb) == 0);
                }
            }
            if (!t.assert_true(cursor.ready_to_publish())) return;
            if (splits > 1) {
                flash_attn_combine_results<D><<<dim3(queries,heads,1),D,splits*sizeof(float2)>>>(parts,meta,final_output,splits);
                CUDA_CHECK(cudaGetLastError());
            } else CUDA_CHECK(cudaMemcpy(final_output,parts,out_bytes,cudaMemcpyDeviceToDevice));
            if (spanned == 3 && fault == resume_fault::publication && !injected) {
                t.assert_true(!cursor.publish(false) && !cursor.publish(true));
                std::vector<uint8_t> public_bytes(out_bytes);
                CUDA_CHECK(cudaMemcpy(public_bytes.data(),output,out_bytes,cudaMemcpyDeviceToHost));
                t.assert_true(std::all_of(public_bytes.begin(),public_bytes.end(),[](uint8_t x){return x==0x5a;}));
                CUDA_CHECK(cudaMemset(scratch+guard+resume_layout.partial_offset,0x5a,scratch_bytes-resume_layout.partial_offset));
                injected = true;
                continue;
            }
            if (!t.assert_true(cursor.publish(true))) return;
            CUDA_CHECK(cudaMemcpy(output,final_output,out_bytes,cudaMemcpyDeviceToDevice));
            }
            CUDA_CHECK(cudaEventDestroy(read_done));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        if (spanned < 2) {
#define LAUNCH(columns,KV) launch_attention<D,columns,gqa,KV>(q,spanned ? scratch+guard : static_cast<const void *>(nk),nv,mask,nullptr,parts,meta,queries,tokens,splits)
        if (spanned) {
            using reader = ggml_cuda_fattn_tile_span_kv<K,V>;
            if (queries <= 2) { LAUNCH(2,reader); } else { LAUNCH(4,reader); }
        } else {
            if (queries <= 2) { LAUNCH(2,ggml_cuda_fattn_tile_native_rows); } else { LAUNCH(4,ggml_cuda_fattn_tile_native_rows); }
        }
#undef LAUNCH
        if (splits > 1) {
            flash_attn_combine_results<D><<<dim3(queries,heads,1),D,splits*sizeof(float2)>>>(parts,meta,output,splits);
            CUDA_CHECK(cudaGetLastError());
        }
        }
        results[spanned].resize(out_bytes/4);
        CUDA_CHECK(cudaMemcpy(results[spanned].data(),output,out_bytes,cudaMemcpyDeviceToHost));
        if (splits > 1) {
            metas[spanned].resize(size_t(heads)*queries*splits);
            CUDA_CHECK(cudaMemcpy(metas[spanned].data(),meta,metas[spanned].size()*sizeof(float2),cudaMemcpyDeviceToHost));
        }
        for (auto bounded : {std::pair{scratch,scratch_bytes},std::pair{storage,out_bytes}}) {
            std::vector<uint8_t> bytes(bounded.second+2*guard);
            CUDA_CHECK(cudaMemcpy(bytes.data(),bounded.first,bytes.size(),cudaMemcpyDeviceToHost));
            t.assert_true(std::all_of(bytes.begin(),bytes.begin()+guard,[](uint8_t x){return x==0x5a;}));
            t.assert_true(std::all_of(bytes.end()-guard,bytes.end(),[](uint8_t x){return x==0x5a;}));
        }
    }
    t.assert_true(std::all_of(results[0].begin(),results[0].end(),[](float x){return std::isfinite(x);}));
    const auto comparison = compare_attention(results[0],results[1],compiled_cc);
    ++attention_cases; attention_nonexact += !comparison.exact;
    attention_max_abs = std::max(attention_max_abs,comparison.max_abs);
    attention_max_relative_l2 = std::max(attention_max_relative_l2,comparison.relative_l2);
    t.assert_true(compiled_cc == 610 ? "stock/span bounded FP32 output" : "stock/span output bits",comparison.accepted);
    if (splits > 1) t.assert_true("stock/span metadata bits",
        std::memcmp(metas[0].data(),metas[1].data(),metas[0].size()*sizeof(float2)) == 0);
    if (!comparison.exact) {
        t.out << "D=" << D << " gqa=" << gqa << " K=" << ggml_type_name(K) << " V=" << ggml_type_name(V)
            << " TG=" << queries << " ctx=" << tokens << " spans=" << table.count << " mask=" << masked
            << " max_abs=" << comparison.max_abs << " relative_l2=" << comparison.relative_l2 << '\n';
        if (splits > 1) {
            size_t max_mismatch = 0, sum_mismatch = 0;
            for (size_t i = 0; i < metas[0].size(); ++i) {
                max_mismatch += std::memcmp(&metas[0][i].x,&metas[1][i].x,4) != 0;
                sum_mismatch += std::memcmp(&metas[0][i].y,&metas[1][i].y,4) != 0;
            }
            t.out << "meta max mismatch=" << max_mismatch << " sum mismatch=" << sum_mismatch << '\n';
        }
    }
    if constexpr (resumed) {
        t.assert_true("single-wave/multi-wave output bits",std::memcmp(results[2].data(),results[3].data(),out_bytes) == 0);
        const auto resumed_comparison = compare_attention(results[0],results[3],compiled_cc);
        ++resume_cases; resume_nonexact += !resumed_comparison.exact;
        resume_max_abs = std::max(resume_max_abs,resumed_comparison.max_abs);
        resume_max_relative_l2 = std::max(resume_max_relative_l2,resumed_comparison.relative_l2);
        t.assert_true("stock/resumed numerical policy",resumed_comparison.accepted);
        if (splits > 1) t.assert_true("single-wave/multi-wave metadata bits",
            std::memcmp(metas[2].data(),metas[3].data(),metas[2].size()*sizeof(float2)) == 0);
        if (splits > 1) t.assert_true("stock/resumed metadata bits",
            std::memcmp(metas[0].data(),metas[3].data(),metas[0].size()*sizeof(float2)) == 0);
    }
    for (auto & grant : grants) {
        std::vector<uint8_t> bytes(grant.bytes);
        CUDA_CHECK(cudaMemcpy(bytes.data(),grant.allocation,bytes.size(),cudaMemcpyDeviceToHost));
        t.assert_true(bytes == grant.expected);
    }
    for (auto pointer : allocations) CUDA_CHECK(cudaFree(pointer));
}

template<ggml_type type, typename T, int width>
static __global__ void span_load_probe(ggml_cuda_fattn_tile_span_table table, T * destination, int first, int live, int head, bool value) {
    const ggml_cuda_fattn_tile_span_rows<type> rows{&table,first,0,head,value};
    flash_attn_tile_load_tile<32,4,7,width,4,true>(rows,destination,live);
}

// CPU-backed grants exercise metadata validation without accessing accelerator payloads.
static void invalid_grants(testing & t) {
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),8192));
    ggml_kv_stream_span spans[3] = {{buffer.get(),buffer.get(),0,3,128,4096},
        {buffer.get(),buffer.get(),3,5,1024,5120},{buffer.get(),buffer.get(),8,1,2048,6144}};
    ggml_kv_stream_span_plan_view view;
    view.shape = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,64,64,2,256,128};
    view.spans = spans; view.count = 3; view.active_tokens = 9; view.query_tokens = 4;
    ggml_cuda_fattn_tile_span_table table;
    t.assert_true(ggml_cuda_fattn_tile_spans_make(view,table));
    t.assert_equal(3,table.count); t.assert_equal(9,table.tokens);
    const auto saved = table;
    auto reject = [&] { t.assert_true(!ggml_cuda_fattn_tile_spans_make(view,table));
        t.assert_true(std::memcmp(&saved,&table,sizeof(table)) == 0); };
    spans[1].token_begin = 4; reject(); spans[1].token_begin = 3;
    spans[2].tokens = 0; reject(); spans[2].tokens = 1;
    spans[0].k_offset = 8192-203; reject(); spans[0].k_offset = 128;
    spans[0].v_offset = SIZE_MAX; reject(); spans[0].v_offset = 4096;
    spans[0].k_buffer = nullptr; reject(); spans[0].k_buffer = buffer.get();
    view.count = 4; reject(); view.count = 3;
    view.active_tokens = INT32_MAX+size_t(1); reject(); view.active_tokens = 9;
    view.shape.type_k = GGML_TYPE_IQ4_NL; reject(); view.shape.type_k = GGML_TYPE_Q8_0;
    view.shape.head_dim_k = 63; reject(); view.shape.head_dim_k = 64;
    view.query_tokens = 10; reject(); view.query_tokens = 4;
    t.assert_true(ggml_cuda_fattn_tile_spans_make(view,table));
    ggml_cuda_fattn_tile_span_workspace workspace;
    t.assert_true(ggml_cuda_fattn_tile_span_workspace_make(4,4,256,3,workspace));
    t.assert_equal(size_t(256),workspace.partial_offset);
    t.assert_equal(size_t(256+4*4*3*256*4),workspace.meta_offset);
    t.assert_equal(workspace.meta_offset+4*4*3*8,workspace.bytes);
    const auto workspace_saved = workspace;
    for (auto shape : {std::vector<size_t>{0,1,64,1},{4,5,64,1},{4,4,SIZE_MAX,3},{SIZE_MAX,4,64,3}}) {
        t.assert_true(!ggml_cuda_fattn_tile_span_workspace_make(shape[0],shape[1],shape[2],shape[3],workspace));
        t.assert_true(std::memcmp(&workspace,&workspace_saved,sizeof(workspace)) == 0);
    }
    t.assert_true(ggml_cuda_fattn_tile_span_workspace_make(4,4,256,1,workspace));
    t.assert_equal(size_t(256),workspace.bytes);
}

template<ggml_type type, typename T, int width>
static void check_conversion(testing & t, int first, int live, const std::vector<int> & cuts, bool value, bool contiguous = true) {
    constexpr int tokens = 19, heads = 2, padding = 4, guard = 128;
    constexpr bool packed = std::is_same<T,half2>::value;
    constexpr size_t row_elements = (packed ? width/2 : width)+padding;
    constexpr size_t tile_bytes = 7*row_elements*sizeof(T);
    std::vector<float> input(tokens*heads*width);
    for (size_t i = 0; i < input.size(); ++i) input[i] = .23f*std::sin(float(i%401)*.31f);
    const size_t row_bytes = ggml_row_size(type,width);
    std::vector<uint8_t> encoded(tokens*heads*row_bytes);
    if (type == GGML_TYPE_F32) std::memcpy(encoded.data(),input.data(),encoded.size());
    else ggml_quantize_chunk(type,input.data(),encoded.data(),0,tokens*heads,width,nullptr);
    void * full = nullptr; half * reference = nullptr; uint8_t * destination = nullptr;
    CUDA_CHECK(cudaMalloc(&full,encoded.size()));
    CUDA_CHECK(cudaMalloc(&reference,input.size()*sizeof(half)));
    CUDA_CHECK(cudaMemcpy(full,encoded.data(),encoded.size(),cudaMemcpyHostToDevice));
    if constexpr (type == GGML_TYPE_F16) CUDA_CHECK(cudaMemcpy(reference,full,encoded.size(),cudaMemcpyDeviceToDevice));
    else if (contiguous) ggml_get_to_fp16_cuda(type)(full,reference,input.size(),nullptr);
    else {
        const int64_t stride = heads*row_bytes/ggml_type_size(type);
        ggml_get_to_fp16_nc_cuda(type)(full,reference,width*heads,tokens,1,1,stride,stride*tokens,stride*tokens,nullptr);
    }
    std::vector<ggml_fp16_t> converted(input.size());
    CUDA_CHECK(cudaMemcpy(converted.data(),reference,converted.size()*sizeof(half),cudaMemcpyDeviceToHost));
    ggml_cuda_fattn_tile_span_table table{};
    table.count = cuts.size()-1; table.tokens = tokens;
    table.contiguous_k = table.contiguous_v = contiguous;
    std::vector<uint8_t *> allocations;
    std::vector<size_t> allocation_bytes;
    for (int i = 0; i < table.count; ++i) {
        const size_t bytes = size_t(cuts[i+1]-cuts[i])*heads*row_bytes;
        uint8_t * storage = nullptr;
        CUDA_CHECK(cudaMalloc(&storage,bytes+2*guard));
        CUDA_CHECK(cudaMemset(storage,0xa5,bytes+2*guard));
        CUDA_CHECK(cudaMemcpy(storage+guard,encoded.data()+size_t(cuts[i])*heads*row_bytes,bytes,cudaMemcpyHostToDevice));
        allocations.push_back(storage); allocation_bytes.push_back(bytes);
        table.spans[i] = {reinterpret_cast<const char *>(storage+guard),reinterpret_cast<const char *>(storage+guard),
            cuts[i],cuts[i+1]-cuts[i],int64_t(heads*row_bytes),int64_t(heads*row_bytes),int64_t(row_bytes),int64_t(row_bytes)};
    }
    CUDA_CHECK(cudaMalloc(&destination,tile_bytes+2*guard));
    CUDA_CHECK(cudaMemset(destination,0x5a,tile_bytes+2*guard));
    span_load_probe<type,T,width><<<1,dim3(32,4)>>>(table,reinterpret_cast<T *>(destination+guard),first,live,1,value);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint8_t> actual(tile_bytes+2*guard), expected(tile_bytes+2*guard,0x5a);
    CUDA_CHECK(cudaMemcpy(actual.data(),destination,actual.size(),cudaMemcpyDeviceToHost));
    for (int row = 0; row < 7; ++row) for (int col = 0; col < width; ++col) {
        const auto h = row < live ? converted[((first+row)*heads+1)*width+col] : ggml_fp16_t(0);
        auto * out = expected.data()+guard+row*row_elements*sizeof(T)+col*(packed ? 2 : 4);
        if (packed) std::memcpy(out,&h,2);
        else { const float f = ggml_fp16_to_fp32(h); std::memcpy(out,&f,4); }
    }
    t.assert_true(std::string(ggml_type_name(type))+" width="+std::to_string(width)+" first="+std::to_string(first)+" live="+std::to_string(live),actual == expected);
    for (size_t i = 0; i < allocations.size(); ++i) {
        std::vector<uint8_t> bytes(allocation_bytes[i]+2*guard);
        CUDA_CHECK(cudaMemcpy(bytes.data(),allocations[i],bytes.size(),cudaMemcpyDeviceToHost));
        t.assert_true(std::all_of(bytes.begin(),bytes.begin()+guard,[](uint8_t x){return x==0xa5;}));
        t.assert_true(std::all_of(bytes.end()-guard,bytes.end(),[](uint8_t x){return x==0xa5;}));
        t.assert_true(std::memcmp(bytes.data()+guard,encoded.data()+size_t(cuts[i])*heads*row_bytes,allocation_bytes[i]) == 0);
        CUDA_CHECK(cudaFree(allocations[i]));
    }
    CUDA_CHECK(cudaFree(destination)); CUDA_CHECK(cudaFree(reference)); CUDA_CHECK(cudaFree(full));
}

int main(int argc, char ** argv) {
    testing t;
    t.test("attention_comparison_policy",comparison_policy_tests);
    if (argc == 2 && !std::strcmp(argv[1],"--policy-only")) return t.summary();
    if (argc != 1) return 2;
    t.test("bounded_tile_span_metadata",invalid_grants);
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return t.failures ? t.summary() : 77;
    ggml_backend_load_all();
    t.test("encoded_tiles_match_stock_conversion",[&](testing & t) {
        for (const auto & cuts : {std::vector<int>{0,19},std::vector<int>{0,4,19},std::vector<int>{0,4,9,19}})
        for (auto extent : {std::pair{0,7},std::pair{2,7},std::pair{12,7},std::pair{17,2},std::pair{19,0}})
        for (bool value : {false,true}) {
#define CHECK_TYPE(type) check_conversion<type,half2,64>(t,extent.first,extent.second,cuts,value); check_conversion<type,float,64>(t,extent.first,extent.second,cuts,value)
            CHECK_TYPE(GGML_TYPE_F16); CHECK_TYPE(GGML_TYPE_BF16); CHECK_TYPE(GGML_TYPE_F32);
            CHECK_TYPE(GGML_TYPE_Q8_0); CHECK_TYPE(GGML_TYPE_Q4_0); CHECK_TYPE(GGML_TYPE_Q4_1);
            CHECK_TYPE(GGML_TYPE_Q5_0); CHECK_TYPE(GGML_TYPE_Q5_1);
#undef CHECK_TYPE
        }
        for (bool value : {false,true}) {
            check_conversion<GGML_TYPE_Q4_0,half2,64>(t,2,7,{0,4,9,19},value,false);
            check_conversion<GGML_TYPE_Q4_0,float,64>(t,17,2,{0,4,9,19},value,false);
        }
    });
    t.test("complete_layer_span_attention_matches_stock_policy",[&](testing & t) {
        for (int queries : {1,2,3,4}) for (int tokens : {31,513,8192}) for (bool mask : {false,true}) {
            const std::vector<std::vector<int>> cuts = mask ? std::vector<std::vector<int>>{{0,tokens}} :
                std::vector<std::vector<int>>{{0,1,tokens},{0,17,29,tokens},{0,tokens/2,tokens-1,tokens}};
            for (const auto & spans : cuts) {
#define CHECK_ATTENTION(K,V) attention_case<64,1,K,V>(t,queries,tokens,spans,mask); attention_case<256,1,K,V>(t,queries,tokens,spans,mask)
                CHECK_ATTENTION(GGML_TYPE_F16,GGML_TYPE_F16);
                CHECK_ATTENTION(GGML_TYPE_F32,GGML_TYPE_BF16);
                CHECK_ATTENTION(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0);
                CHECK_ATTENTION(GGML_TYPE_Q5_1,GGML_TYPE_Q4_1);
#undef CHECK_ATTENTION
            }
        }
        for (int queries : {1,2,3,4}) for (const auto & spans : {std::vector<int>{0,256},std::vector<int>{0,128,256},std::vector<int>{0,64,128,256}}) {
            attention_case<64,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(t,queries,256,spans,true);
            attention_case<256,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(t,queries,256,spans,true);
        }
        for (int queries : {1,2,3,4}) {
            attention_case<64,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(t,queries,8192,{0,64,4096,8192},true);
            attention_case<256,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(t,queries,8192,{0,64,4096,8192},true);
            attention_case<256,4,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(t,queries,8192,{0,64,4096,8192},true);
            attention_case<256,8,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(t,queries,8192,{0,64,4096,8192},true);
        }
        for (int queries : {1,2,3,4}) for (int tokens : {31,513,8192}) {
            attention_case<40,1,GGML_TYPE_F16,GGML_TYPE_F16>(t,queries,tokens,{0,1,tokens-1,tokens},true);
            attention_case<72,1,GGML_TYPE_F32,GGML_TYPE_BF16>(t,queries,tokens,{0,1,tokens-1,tokens},true);
        }
    });
    t.test("tile_resume_preserves_split_order_across_refills",[&](testing & t) {
        for (int queries : {1,2,3,4}) for (int tokens : {31,529,2049}) for (size_t tiles : {1,2,5}) {
            attention_case<64,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,true>(t,queries,tokens,{0,17,tokens-1,tokens},true,tiles);
            attention_case<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,true>(t,queries,tokens,{0,17,tokens-1,tokens},true,tiles);
        }
        for (int queries : {1,2,3,4}) {
            attention_case<256,2,GGML_TYPE_F16,GGML_TYPE_F16,true>(t,queries,768,{0,17,511,768},true,1);
            attention_case<256,4,GGML_TYPE_F32,GGML_TYPE_BF16,true>(t,queries,768,{0,17,511,768},true,2);
            attention_case<256,8,GGML_TYPE_Q5_1,GGML_TYPE_Q4_1,true>(t,queries,768,{0,17,511,768},true,3);
        }
    });
    t.test("tile_resume_fences_cancellation_and_failed_publication",[&](testing & t) {
        for (int queries : {1,2,3,4}) for (auto fault : {resume_fault::cancellation,resume_fault::read_completion,resume_fault::publication}) {
            attention_case<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,true>(t,queries,529,{0,17,511,529},true,1,fault);
        }
        attention_case<64,1,GGML_TYPE_F16,GGML_TYPE_F16,true>(t,2,529,{0,17,511,529},false,2);
        attention_case<40,1,GGML_TYPE_F16,GGML_TYPE_F16,true>(t,3,529,{0,17,511,529},true,1);
        attention_case<72,1,GGML_TYPE_F32,GGML_TYPE_BF16,true>(t,4,529,{0,17,511,529},true,5);
    });
    std::printf("attention comparison: cases=%zu nonexact=%zu max_abs=%.9g max_relative_l2=%.9g\n",
        attention_cases,attention_nonexact,attention_max_abs,attention_max_relative_l2);
    std::printf("resume comparison: cases=%zu nonexact=%zu max_abs=%.9g max_relative_l2=%.9g\n",
        resume_cases,resume_nonexact,resume_max_abs,resume_max_relative_l2);
    return t.summary();
}

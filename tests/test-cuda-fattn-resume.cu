#include "fattn-vec.cuh"
#include "fattn-tile-resume.h"
#include "kv-stream-attention-dispatch.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct device_allocation {
    char * data = nullptr;
    size_t bytes;
    explicit device_allocation(size_t size) : bytes(size) {check_cuda(cudaMalloc(&data,size));}
    ~device_allocation() {if (data) (void) cudaFree(data);}
    device_allocation(const device_allocation &) = delete;
    device_allocation & operator=(const device_allocation &) = delete;
    std::vector<char> read() const {
        std::vector<char> host(bytes);
        check_cuda(cudaMemcpy(host.data(),data,bytes,cudaMemcpyDeviceToHost));
        return host;
    }
};

static void check_guard(const std::vector<char> & bytes) {
    if (!std::all_of(bytes.begin(),bytes.begin()+128,[](char v){return v == char(0x5a);}) ||
            !std::all_of(bytes.end()-128,bytes.end(),[](char v){return v == char(0x5a);}))
        throw std::runtime_error("resume scratch guard changed");
}

static float vector_max_error = 0;

// V2's vector family used a 1e-6 stock bound; checkpointing itself must introduce no bit changes.
static void check_vector_stock(const std::vector<char> & expected, const std::vector<char> & actual) {
    for (size_t i = 128; i < expected.size()-128; i += 4) {
        float a, b;
        std::memcpy(&a,expected.data()+i,4); std::memcpy(&b,actual.data()+i,4);
        if (std::memcmp(expected.data()+i,actual.data()+i,4)) {
            if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > 1e-6f)
                throw std::runtime_error("resumed vector exceeds its existing stock numerical bound");
            vector_max_error = std::max(vector_max_error,std::abs(a-b));
        }
    }
}

// Each wave reuses the same physical ring only after its actual CUDA read fence.
template<int width, int columns, ggml_type key, ggml_type value, bool tail_safe = true, bool compact_mask = false>
static void check_vector(int active, int queries, int splits, int wave, bool sink, int mask_first = 0) {
    constexpr int heads = 12, kv_heads = 2, guard = 128;
    const int capacity = (active+127)/128*128;
    const int k_row = int(ggml_row_size(key,width)), v_row = int(ggml_row_size(value,width));
    std::vector<char> encoded[2];
    for (int plane = 0; plane < 2; ++plane) {
        const auto type = plane ? value : key;
        const int row = plane ? v_row : k_row;
        encoded[plane].resize(size_t(capacity)*kv_heads*row,0);
        std::vector<float> input(size_t(capacity)*width);
        for (int head = 0; head < kv_heads; ++head) {
            for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i*19+head*11+plane*7)%251)-125)/128;
            ggml_quantize_chunk(type,input.data(),encoded[plane].data()+size_t(head)*capacity*row,0,capacity,width,nullptr);
        }
    }
    device_allocation dense_k(encoded[0].size()), dense_v(encoded[1].size());
    check_cuda(cudaMemcpy(dense_k.data,encoded[0].data(),encoded[0].size(),cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dense_v.data,encoded[1].data(),encoded[1].size(),cudaMemcpyHostToDevice));
    device_allocation q(size_t(width)*queries*heads*4), mask(size_t(capacity)*queries*2), sinks(heads*4), bounds(size_t(queries)*4*sizeof(int32_t));
    std::vector<float> host_q(size_t(width)*queries*heads), host_sinks(heads);
    for (size_t i = 0; i < host_q.size(); ++i) host_q[i] = float(int((i*23)%263)-131)/128;
    for (int head = 0; head < heads; ++head) host_sinks[head] = float(head-6)/8;
    std::vector<ggml_fp16_t> host_mask(size_t(capacity)*queries);
    for (int query = 0; query < queries; ++query) for (int token = 0; token < capacity; ++token)
        host_mask[size_t(query)*capacity+token] = ggml_fp32_to_fp16(token >= mask_first && token <= active-queries+query ? 0 : -INFINITY);
    std::vector<int32_t> intervals(size_t(queries)*4);
    for (int query = 0; query < queries; ++query) {
        intervals[4*query] = mask_first;
        intervals[4*query+1] = active-queries+query+1;
        intervals[4*query+2] = capacity;
        intervals[4*query+3] = 1;
    }
    check_cuda(cudaMemcpy(q.data,host_q.data(),q.bytes,cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(mask.data,host_mask.data(),mask.bytes,cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(sinks.data,host_sinks.data(),sinks.bytes,cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(bounds.data,intervals.data(),bounds.bytes,cudaMemcpyHostToDevice));
    const size_t output_bytes = size_t(width)*queries*heads*splits*4, meta_bytes = size_t(queries)*heads*splits*8;
    device_allocation outputs[4] = {device_allocation(output_bytes+2*guard),device_allocation(output_bytes+2*guard),device_allocation(output_bytes+2*guard),device_allocation(output_bytes+2*guard)};
    device_allocation metadata[4] = {device_allocation(meta_bytes+2*guard),device_allocation(meta_bytes+2*guard),device_allocation(meta_bytes+2*guard),device_allocation(meta_bytes+2*guard)};
    device_allocation final[4] = {device_allocation(size_t(width)*queries*heads*4+2*guard),device_allocation(size_t(width)*queries*heads*4+2*guard),device_allocation(size_t(width)*queries*heads*4+2*guard),device_allocation(size_t(width)*queries*heads*4+2*guard)};
    // Reserve the worst native lane count; all active state and guard bytes are checked below.
    const int lanes = value == GGML_TYPE_F16 || value == GGML_TYPE_BF16 ? 8 : std::min(width/4,32);
    const size_t state_bytes = size_t((queries+columns-1)/columns)*heads*splits*128*columns*(2+width/lanes)*4;
    device_allocation state(state_bytes+2*guard);
    device_allocation ring_k(size_t(wave)*kv_heads*k_row), ring_v(size_t(wave)*kv_heads*v_row);
    cudaStream_t stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t ready = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check_cuda(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking));
    check_cuda(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
    check_cuda(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    try {
        check_cuda(cudaMemsetAsync(state.data,0x5a,state.bytes,stream));
        const dim3 grid((queries+columns-1)/columns,splits,heads), block(32,4);
        for (int implementation = 0; implementation < 4; ++implementation) {
            check_cuda(cudaMemsetAsync(outputs[implementation].data,0x5a,outputs[implementation].bytes,stream));
            check_cuda(cudaMemsetAsync(metadata[implementation].data,0x5a,metadata[implementation].bytes,stream));
            auto * out = reinterpret_cast<float *>(outputs[implementation].data+guard);
            auto * meta = reinterpret_cast<float2 *>(metadata[implementation].data+guard);
            const char * k_data = dense_k.data, * v_data = dense_v.data;
            const char * mask_data = implementation == 0 || !compact_mask ? mask.data : nullptr;
            const auto * mask_bounds = compact_mask ? reinterpret_cast<const int32_t *>(bounds.data) : nullptr;
            int k_head = capacity*k_row, v_head = capacity*v_row;
#define VECTOR_ARGS q.data,k_data,v_data,mask_data,sink ? sinks.data : nullptr,nullptr,out,meta, \
    1.0f/16,0.0f,1.0f,1.0f,0u,0.0f,width,init_fastdiv_values(queries),heads,1, \
    width*4,queries*width*4,heads*queries*width*4,width,capacity,kv_heads,1, \
    k_row,k_head,int64_t(k_head)*kv_heads,v_row,v_head,int64_t(v_head)*kv_heads, \
    queries,1,1,capacity*2,capacity*queries*2,int64_t(capacity)*queries*2
            if (implementation == 0) {
                flash_attn_ext_vec<width,columns,key,value,false><<<grid,block,0,stream>>>(VECTOR_ARGS);
                check_cuda(cudaGetLastError());
                check_cuda(cudaStreamSynchronize(stream));
            } else if (implementation == 1) {
                flash_attn_ext_vec_resume<width,columns,key,value,tail_safe,compact_mask><<<grid,block,0,stream>>>(VECTOR_ARGS,
                    reinterpret_cast<float *>(state.data+guard),0,tail_safe ? active : capacity,true,true,mask_bounds);
                check_cuda(cudaGetLastError());
                check_cuda(cudaStreamSynchronize(stream));
            } else {
                const int limit = tail_safe ? active : capacity;
                ggml_cuda_fattn_tile_resume_cursor cursor(limit,128);
                for (int first = 0; first < limit; first += wave) {
                    const int end = std::min(first+wave,limit), live = end-first;
                    if (!cursor.begin(end)) throw std::runtime_error("invalid vector wave");
                    k_data = ring_k.data; v_data = ring_v.data;
                    k_head = wave*k_row; v_head = wave*v_row;
                    for (int head = 0; head < kv_heads; ++head) {
                        check_cuda(cudaMemcpyAsync(ring_k.data+head*k_head,encoded[0].data()+size_t(head*capacity+first)*k_row,size_t(live)*k_row,cudaMemcpyHostToDevice,copy_stream));
                        check_cuda(cudaMemcpyAsync(ring_v.data+head*v_head,encoded[1].data()+size_t(head*capacity+first)*v_row,size_t(live)*v_row,cudaMemcpyHostToDevice,copy_stream));
                    }
                    check_cuda(cudaEventRecord(ready,copy_stream));
                    check_cuda(cudaStreamWaitEvent(stream,ready,0));
                    const int bulk=implementation == 3 ? int(ggml_cuda_kv_stream_vector_bulk_tokens(first,size_t(live))) : 0;
                    if (bulk) {
                        flash_attn_ext_vec_resume<width,columns,key,value,false,compact_mask><<<grid,block,0,stream>>>(VECTOR_ARGS,
                            reinterpret_cast<float *>(state.data+guard),first,first+bulk,first == 0,first+bulk == limit,mask_bounds);
                        check_cuda(cudaGetLastError());
                    }
                    if (bulk < live) {
                        k_data += bulk*k_row; v_data += bulk*v_row;
                        flash_attn_ext_vec_resume<width,columns,key,value,tail_safe,compact_mask><<<grid,block,0,stream>>>(VECTOR_ARGS,
                            reinterpret_cast<float *>(state.data+guard),first+bulk,end,first+bulk == 0,end == limit,mask_bounds);
                    }
                    check_cuda(cudaGetLastError());
                    check_cuda(cudaEventRecord(done,stream));
                    if (cursor.can_reuse()) throw std::runtime_error("ring reused before completion");
                    check_cuda(cudaEventSynchronize(done));
                    if (!cursor.complete(true)) throw std::runtime_error("vector read completion failed");
                    if (end != limit) {
                        const auto unpublished = outputs[implementation].read();
                        if (!std::all_of(unpublished.begin(),unpublished.end(),[](char v){return v == char(0x5a);}))
                            throw std::runtime_error("partial wave published output");
                    }
                }
                if (!cursor.publish(true) || cursor.publish(true)) throw std::runtime_error("publication must occur once");
            }
#undef VECTOR_ARGS
            check_cuda(cudaMemsetAsync(final[implementation].data,0x5a,final[implementation].bytes,stream));
            if (splits > 1) {
                flash_attn_combine_results<width><<<dim3(queries,heads,1),width,splits*sizeof(float2),stream>>>(
                    out,meta,reinterpret_cast<float *>(final[implementation].data+guard),splits);
                check_cuda(cudaGetLastError());
            } else check_cuda(cudaMemcpyAsync(final[implementation].data+guard,out,size_t(width)*queries*heads*4,cudaMemcpyDeviceToDevice,stream));
            check_cuda(cudaStreamSynchronize(stream));
        }
        const auto stock = outputs[0].read(), expected = outputs[1].read(), actual = outputs[2].read();
        const auto stock_meta = metadata[0].read(), expected_meta = metadata[1].read(), actual_meta = metadata[2].read();
        check_guard(stock); check_guard(stock_meta);
        const auto stock_final = final[0].read(), resumed_final = final[2].read();
        check_guard(stock_final); check_guard(resumed_final);
        check_vector_stock(stock_final,resumed_final);
        if constexpr (compact_mask) {
            if (stock_final != resumed_final) throw std::runtime_error("compact vector mask changes stock output bits");
        }
        for (const auto * planes : {&outputs,&metadata,&final}) {
            const auto mixed=(*planes)[3].read();
            check_guard(mixed);
            const auto baseline=(*planes)[1].read();
            if (mixed != baseline) {
                size_t mismatch=0; float maximum=0;
                for (size_t i=guard;i<baseline.size()-guard;i+=4) {
                    float a,b; std::memcpy(&a,baseline.data()+i,4); std::memcpy(&b,mixed.data()+i,4);
                    maximum=std::max(maximum,std::abs(a-b));
                    if (std::memcmp(baseline.data()+i,mixed.data()+i,4) && mismatch++ < 3)
                        std::fprintf(stderr,"bulk/tail plane=%d float=%zu baseline=%a mixed=%a\n",int(planes == &outputs ? 0 : planes == &metadata ? 1 : 2),(i-guard)/4,double(a),double(b));
                }
                std::fprintf(stderr,"bulk/tail width=%d TG%d active=%d splits=%d wave=%d guarded=%d pair=%s/%s mismatches=%zu max_abs=%g\n",width,queries,active,splits,wave,int(tail_safe),ggml_type_name(key),ggml_type_name(value),mismatch,double(maximum));
                throw std::runtime_error("bulk/tail vector dispatch changes output or reduction metadata");
            }
        }
        check_guard(expected); check_guard(actual); check_guard(expected_meta); check_guard(actual_meta); check_guard(state.read());
        if (expected != actual || expected_meta != actual_meta) {
            for (int plane = 0; plane < 2; ++plane) {
                const auto & a = plane ? expected_meta : expected;
                const auto & b = plane ? actual_meta : actual;
                size_t mismatch = 0; float maximum = 0;
                for (size_t i = guard; i < a.size()-guard; i += 4) {
                    float av, bv;
                    std::memcpy(&av,a.data()+i,4); std::memcpy(&bv,b.data()+i,4);
                    if (std::memcmp(a.data()+i,b.data()+i,4)) {
                        if (mismatch++ < 3) std::fprintf(stderr,"plane=%d float=%zu expected=%a actual=%a\n",plane,(i-guard)/4,double(av),double(bv));
                        maximum = std::max(maximum,std::abs(av-bv));
                    }
                }
                std::fprintf(stderr,"plane=%d mismatches=%zu max_abs=%g\n",plane,mismatch,double(maximum));
            }
            std::fprintf(stderr,"vector mismatch: width=%d TG%d splits=%d active=%d wave=%d %s/%s\n",
                width,queries,splits,active,wave,ggml_type_name(key),ggml_type_name(value));
            throw std::runtime_error("vector checkpoint changes output or reduction metadata");
        }
    } catch (...) {
        (void) cudaEventDestroy(ready); (void) cudaEventDestroy(done);
        (void) cudaStreamDestroy(copy_stream); (void) cudaStreamDestroy(stream); throw;
    }
    check_cuda(cudaEventDestroy(ready)); check_cuda(cudaEventDestroy(done));
    check_cuda(cudaStreamDestroy(copy_stream)); check_cuda(cudaStreamDestroy(stream));
}

#include "fattn-resume-tile-test.h"

template<ggml_type key, ggml_type value>
static size_t check_vector_pair(bool quick) {
    size_t cases = 0;
    for (int active : {257,1025,8193}) for (int queries : {1,2}) for (int splits : {3,17}) {
        if (quick && (active != 257 || splits != 3)) continue;
        if (queries == 1) check_vector<256,1,key,value>(active,queries,splits,128,true);
        else check_vector<256,2,key,value>(active,queries,splits,128,true);
        ++cases;
    }
    return cases;
}

template<int width, ggml_type key, ggml_type value>
static size_t check_tile_pair(ggml_backend_t backend, bool quick) {
    size_t cases = 0;
    for (int queries : {1,2,3,4}) for (int active : {257,1025}) for (int splits : {1,17}) for (uint32_t flags : {0u,3u}) {
        if (quick && (active != 257 || splits != 17 || flags != 3)) continue;
        if (queries <= 2) {
            check_tile<width,2,1,key,value>(backend,active,queries,splits,1,3,true,flags);
            check_tile<width,1,2,key,value>(backend,active,queries,splits,4,2,false,flags);
        } else {
            check_tile<width,4,1,key,value>(backend,active,queries,splits,1,3,true,flags);
            check_tile<width,4,2,key,value>(backend,active,queries,splits,4,2,false,flags);
        }
        cases += 2;
    }
    return cases;
}

int main(int argc, char ** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    const bool quick = argc > 1 && !std::strcmp(argv[1],"--quick");
    try {
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_name("CUDA0");
        if (!device) return 77;
        ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
        if (!backend) return 77;
#ifdef FATTN_RESUME_CASE40
        check_tile<40,4,1,GGML_TYPE_F16,GGML_TYPE_F16>(backend.get(),257,3,1,256,3,true,0);
        check_tile<40,4,1,GGML_TYPE_F16,GGML_TYPE_F16>(backend.get(),257,3,1,1,3,true,0);
        check_tile<40,2,1,GGML_TYPE_F16,GGML_TYPE_F16>(backend.get(),257,1,1,1,3,true,0);
        check_tile<72,2,1,GGML_TYPE_F32,GGML_TYPE_BF16>(backend.get(),257,1,1,1,3,true,0);
        check_tile<72,4,2,GGML_TYPE_F32,GGML_TYPE_BF16>(backend.get(),257,3,1,1,3,true,0);
        std::puts("Padded-head tile resume matches stock");
#else
        if (argc > 1 && !std::strcmp(argv[1],"--case40")) {
            check_tile<40,4,1,GGML_TYPE_F16,GGML_TYPE_F16>(backend.get(),257,3,1,256,3,true,0);
            check_tile<40,4,1,GGML_TYPE_F16,GGML_TYPE_F16>(backend.get(),257,3,1,1,3,true,0);
            return 0;
        }
        size_t tile_cases = 0;
        for (int active : {257,1025,8193}) for (int queries : {5,6,7,8}) for (int splits : {1,3,17})
                for (int wave : {1,4}) for (int spans : {1,2,3}) for (bool interleaved : {false,true}) {
            if (quick && (active != 257 || splits != 3 || wave != 1 || spans != 3 || !interleaved)) continue;
            check_tile<256,8,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(backend.get(),active,queries,splits,wave,spans,interleaved,3);
            check_tile<256,4,8,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,16>(backend.get(),active,queries,splits,wave,spans,interleaved,3);
            tile_cases += 2;
        }
        for (int active : {257,1025,8193}) for (int queries : {1,2,3,4}) for (int splits : {1,3,17})
                for (int wave : {1,4}) for (int spans : {1,2,3}) for (bool interleaved : {false,true}) for (uint32_t flags : {0u,3u}) {
            if (quick && (active != 257 || splits != 3 || wave != 1 || spans != 3 || !interleaved)) continue;
            if (queries <= 2) check_tile<256,2,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(backend.get(),active,queries,splits,wave,spans,interleaved,flags);
            else check_tile<256,4,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(backend.get(),active,queries,splits,wave,spans,interleaved,flags);
            ++tile_cases;
        }
        tile_cases += check_tile_pair<256,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(backend.get(),quick);
        tile_cases += check_tile_pair<256,GGML_TYPE_Q4_0,GGML_TYPE_Q8_0>(backend.get(),quick);
        tile_cases += check_tile_pair<256,GGML_TYPE_Q4_1,GGML_TYPE_Q5_1>(backend.get(),quick);
        tile_cases += check_tile_pair<256,GGML_TYPE_Q5_1,GGML_TYPE_Q4_1>(backend.get(),quick);
        tile_cases += check_tile_pair<256,GGML_TYPE_Q5_0,GGML_TYPE_Q4_0>(backend.get(),quick);
        tile_cases += check_tile_pair<256,GGML_TYPE_BF16,GGML_TYPE_F16>(backend.get(),quick);
        tile_cases += check_tile_pair<256,GGML_TYPE_F16,GGML_TYPE_BF16>(backend.get(),quick);
        tile_cases += check_tile_pair<256,GGML_TYPE_F32,GGML_TYPE_F32>(backend.get(),quick);
        tile_cases += check_tile_pair<40,GGML_TYPE_F16,GGML_TYPE_F16>(backend.get(),quick);
        tile_cases += check_tile_pair<64,GGML_TYPE_BF16,GGML_TYPE_F16>(backend.get(),quick);
        tile_cases += check_tile_pair<72,GGML_TYPE_F32,GGML_TYPE_BF16>(backend.get(),quick);
        std::printf("Native resume tile: %zu bit-exact stock cases passed\n",tile_cases);
        size_t cases = 2;
        check_vector<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,false>(257,1,1,512,true);
        check_vector<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(257,1,1,512,true);
        for (int active : {257,1025,8193}) for (int queries : {1,2}) for (int splits : {1,3,17}) for (int wave : {128,256,512}) {
            if (quick && (active != 257 || splits != 3 || wave != 128)) continue;
            if (queries == 1) check_vector<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(active,queries,splits,wave,true);
            else check_vector<256,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(active,queries,splits,wave,true);
            ++cases;
        }
        cases += check_vector_pair<GGML_TYPE_F16,GGML_TYPE_F16>(quick);
        cases += check_vector_pair<GGML_TYPE_BF16,GGML_TYPE_BF16>(quick);
        cases += check_vector_pair<GGML_TYPE_Q5_1,GGML_TYPE_Q4_1>(quick);
        cases += check_vector_pair<GGML_TYPE_Q4_1,GGML_TYPE_Q5_1>(quick);
        cases += check_vector_pair<GGML_TYPE_Q5_0,GGML_TYPE_Q4_0>(quick);
        cases += check_vector_pair<GGML_TYPE_Q4_0,GGML_TYPE_F16>(quick);
        check_vector<64,1,GGML_TYPE_F16,GGML_TYPE_F16>(257,1,17,128,true);
        check_vector<128,2,GGML_TYPE_BF16,GGML_TYPE_BF16>(1025,2,17,256,false);
        cases += 2;
        for (int active : {257,8193}) for (int first : {0,127}) {
            check_vector<256,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,true,true>(active,1,17,256,true,first);
            check_vector<256,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,true,true>(active,2,17,256,false,first);
            check_vector<256,1,GGML_TYPE_Q5_1,GGML_TYPE_Q4_1,false,true>(active,1,3,512,false,first);
            check_vector<256,2,GGML_TYPE_F16,GGML_TYPE_Q4_0,true,true>(active,2,3,128,true,first);
            cases += 4;
        }
        std::printf("Native resume vector: %zu bit-exact wave cases passed, stock max_abs=%g (existing bound 1e-6)\n",cases,double(vector_max_error));
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr,"%s\n",error.what()); return 1;
    }
}

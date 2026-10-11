#pragma once

#include "fattn-tile.cuh"
#include "fattn-tile-resume.cuh"
#include "fattn-stock-conversion-test.h"

// Compare native split accumulators and final reduction while repeatedly rebinding a guarded ring.
template<int width, int columns, int groups, ggml_type key, ggml_type value, int heads = 12>
static void check_tile(ggml_backend_t backend, int active, int queries, int splits, int wave_tiles, int span_count, bool interleaved, uint32_t flags) {
    constexpr int kv_heads = 2, guard = 128;
    const int capacity = (active+255)/256*256, extent = groups > 1 ? capacity : active;
    const int k_row = int(ggml_row_size(key,width)), v_row = int(ggml_row_size(value,width));
    std::vector<char> encoded[2];
    std::vector<ggml_fp16_t> reference[2];
    for (int plane = 0; plane < 2; ++plane) {
        const auto type = plane ? value : key;
        const int row = plane ? v_row : k_row;
        encoded[plane].resize(size_t(capacity)*kv_heads*row,0);
        std::vector<float> input(size_t(active)*width);
        for (int head = 0; head < kv_heads; ++head) {
            for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i*19+head*11+plane*7)%251)-125)/128;
            ggml_quantize_chunk(type,input.data(),encoded[plane].data()+size_t(head)*capacity*row,0,active,width,nullptr);
        }
        reference[plane] = stock_f16_rows(backend,type,encoded[plane],capacity,kv_heads,!(flags&(1u<<plane)),width);
    }
    device_allocation dense_k(reference[0].size()*2), dense_v(reference[1].size()*2);
    check_cuda(cudaMemcpy(dense_k.data,reference[0].data(),dense_k.bytes,cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dense_v.data,reference[1].data(),dense_v.bytes,cudaMemcpyHostToDevice));
    device_allocation q(size_t(width)*queries*heads*4), mask(size_t(capacity)*queries*2), sinks(heads*4);
    std::vector<float> host_q(size_t(width)*queries*heads), host_sinks(heads);
    for (size_t i = 0; i < host_q.size(); ++i) host_q[i] = float(int((i*23)%263)-131)/128;
    for (int head = 0; head < heads; ++head) host_sinks[head] = float(head-6)/8;
    std::vector<ggml_fp16_t> host_mask(size_t(capacity)*queries);
    for (int query = 0; query < queries; ++query) for (int token = 0; token < capacity; ++token)
        host_mask[size_t(query)*capacity+token] = ggml_fp32_to_fp16(token <= active-queries+query ? 0 : -INFINITY);
    check_cuda(cudaMemcpy(q.data,host_q.data(),q.bytes,cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(mask.data,host_mask.data(),mask.bytes,cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(sinks.data,host_sinks.data(),sinks.bytes,cudaMemcpyHostToDevice));
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties,0));
    const int cc = properties.major*100+properties.minor*10;
    const int threads = ggml_cuda_fattn_tile_get_nthreads(width,width,columns*groups,cc);
    const int tile = ggml_cuda_fattn_tile_get_nbatch_fa(width,width,columns*groups,cc), wave = tile*wave_tiles;
    const ggml_cuda_fattn_tile_resume_geometry geometry{width,heads,kv_heads,uint32_t(queries),columns,groups,
        uint32_t(threads),uint32_t(tile),uint32_t(splits),fast_fp16_available(cc) ? 4u : 8u};
    ggml_cuda_fattn_tile_resume_layout layout;
    if (!ggml_cuda_fattn_tile_resume_layout_make(geometry,extent,layout)) throw std::runtime_error("invalid tile resume layout");
    device_allocation scratch(layout.bytes+2*guard);
    const size_t out_bytes = size_t(width)*queries*heads*splits*4, meta_bytes = size_t(queries)*heads*splits*8;
    device_allocation outputs[2] = {device_allocation(out_bytes+2*guard),device_allocation(out_bytes+2*guard)};
    device_allocation metadata[2] = {device_allocation(meta_bytes+2*guard),device_allocation(meta_bytes+2*guard)};
    device_allocation final[2] = {device_allocation(size_t(width)*queries*heads*4+2*guard),device_allocation(size_t(width)*queries*heads*4+2*guard)};
    device_allocation ring(size_t(wave)*kv_heads*(k_row+v_row)+12*guard);
    cudaStream_t stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t ready = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check_cuda(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking));
    check_cuda(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
    check_cuda(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    try {
        check_cuda(cudaMemsetAsync(scratch.data,0x5a,scratch.bytes,stream));
        const dim3 grid((queries+columns-1)/columns,splits,heads/groups), block(32,threads/32);
        for (int implementation = 0; implementation < 2; ++implementation) {
            check_cuda(cudaMemsetAsync(outputs[implementation].data,0x5a,outputs[implementation].bytes,stream));
            check_cuda(cudaMemsetAsync(metadata[implementation].data,0x5a,metadata[implementation].bytes,stream));
            auto * out = reinterpret_cast<float *>(outputs[implementation].data+guard);
            auto * meta = reinterpret_cast<float2 *>(metadata[implementation].data+guard);
            const char * k_data = dense_k.data, * v_data = dense_v.data;
#define TILE_ARGS q.data,k_data,v_data,mask.data,sinks.data,nullptr,out,meta, \
    1.0f/16,0.0f,1.0f,1.0f,0u,0.0f,width,init_fastdiv_values(queries),heads,1, \
    width*4,queries*width*4,heads*queries*width*4,width,extent,kv_heads,1, \
    width*2,capacity*width*2,int64_t(capacity)*width*kv_heads*2, \
    width*2,capacity*width*2,int64_t(capacity)*width*kv_heads*2, \
    queries,1,1,capacity*2,capacity*queries*2,int64_t(capacity)*queries*2
            if (implementation == 0) {
                flash_attn_tile<width,width,columns,groups,false><<<grid,block,0,stream>>>(TILE_ARGS);
                check_cuda(cudaGetLastError()); check_cuda(cudaStreamSynchronize(stream));
            } else {
                ggml_cuda_fattn_tile_resume_cursor cursor(extent,tile);
                for (int first = 0; first < extent;) {
                    const int end = first+wave >= active ? extent : std::min(first+wave,extent);
                    const int live = std::min(end,active)-first, count = live < 3 ? 1 : span_count;
                    if (!cursor.begin(end)) throw std::runtime_error("invalid tile wave");
                    const std::vector<int> cuts = count == 1 ? std::vector<int>{0,live} : count == 2 ?
                        std::vector<int>{0,std::min(17,live/2),live} : std::vector<int>{0,std::min(17,live/2),std::min(127,live-1),live};
                    ggml_cuda_fattn_tile_span_table table;
                    table.count = count; table.tokens = extent == active ? extent : active;
                    table.contiguous_k = flags&1; table.contiguous_v = flags&2;
                    std::vector<char> host_ring(ring.bytes,char(0x5a));
                    size_t offset = 0;
                    for (int physical = 0; physical < count; ++physical) {
                        const int index = count == 3 ? (physical == 0 ? 0 : physical == 1 ? 2 : 1) : physical;
                        const int rows = cuts[index+1]-cuts[index];
                        const size_t k = offset+guard, v = k+size_t(rows)*kv_heads*k_row+2*guard;
                        auto & span = table.spans[index];
                        span = {ring.data+k,ring.data+v,first+cuts[index],rows,
                            k_row*(interleaved ? kv_heads : 1),v_row*(interleaved ? kv_heads : 1),
                            k_row*(interleaved ? 1 : rows),v_row*(interleaved ? 1 : rows)};
                        for (int row = 0; row < rows; ++row) for (int head = 0; head < kv_heads; ++head) {
                            std::memcpy(host_ring.data()+k+row*span.k_token_stride+head*span.k_head_stride,
                                encoded[0].data()+size_t(head*capacity+first+cuts[index]+row)*k_row,k_row);
                            std::memcpy(host_ring.data()+v+row*span.v_token_stride+head*span.v_head_stride,
                                encoded[1].data()+size_t(head*capacity+first+cuts[index]+row)*v_row,v_row);
                        }
                        offset = v+size_t(rows)*kv_heads*v_row+guard;
                    }
                    ggml_cuda_fattn_tile_resume_descriptor descriptor;
                    if (!ggml_cuda_fattn_tile_resume_wave_make(table,layout,scratch.data+guard,layout.bytes,
                            k_row,v_row,first,end,descriptor,extent != active)) throw std::runtime_error("tile wave admission failed");
                    check_cuda(cudaMemcpyAsync(ring.data,host_ring.data(),ring.bytes,cudaMemcpyHostToDevice,copy_stream));
                    check_cuda(cudaMemcpyAsync(scratch.data+guard,&descriptor,sizeof(descriptor),cudaMemcpyHostToDevice,copy_stream));
                    check_cuda(cudaEventRecord(ready,copy_stream));
                    check_cuda(cudaStreamWaitEvent(stream,ready,0));
                    k_data = scratch.data+guard; v_data = nullptr;
                    flash_attn_tile<width,width,columns,groups,false,ggml_cuda_fattn_tile_resume_kv<key,value>><<<grid,block,0,stream>>>(TILE_ARGS);
                    check_cuda(cudaGetLastError()); check_cuda(cudaEventRecord(done,stream));
                    if (cursor.can_reuse()) throw std::runtime_error("tile ring reused before its read fence");
                    check_cuda(cudaEventSynchronize(done));
                    if (!cursor.complete(true)) throw std::runtime_error("tile read completion failed");
                    if (ring.read() != host_ring) throw std::runtime_error("tile source or guard changed");
                    if (end != extent) {
                        const auto unpublished = outputs[1].read();
                        if (!std::all_of(unpublished.begin(),unpublished.end(),[](char v){return v == char(0x5a);}))
                            throw std::runtime_error("tile wave published incomplete output");
                    }
                    first = end;
                }
                if (!cursor.publish(true) || cursor.publish(true)) throw std::runtime_error("tile publication must occur once");
            }
#undef TILE_ARGS
            check_cuda(cudaMemsetAsync(final[implementation].data,0x5a,final[implementation].bytes,stream));
            if (splits > 1) {
                flash_attn_combine_results<width><<<dim3(queries,heads,1),width,splits*sizeof(float2),stream>>>(
                    out,meta,reinterpret_cast<float *>(final[implementation].data+guard),splits);
                check_cuda(cudaGetLastError());
            } else check_cuda(cudaMemcpyAsync(final[implementation].data+guard,out,size_t(width)*queries*heads*4,cudaMemcpyDeviceToDevice,stream));
            check_cuda(cudaStreamSynchronize(stream));
        }
        check_guard(scratch.read());
        for (const auto * planes : {&outputs,&metadata,&final}) {
            const auto expected = (*planes)[0].read(), actual = (*planes)[1].read();
            check_guard(expected); check_guard(actual);
            if (expected != actual) {
                size_t mismatches = 0; float max_abs = 0;
                for (size_t i = guard; i < expected.size()-guard; i += 4) {
                    float a, b;
                    std::memcpy(&a,expected.data()+i,4); std::memcpy(&b,actual.data()+i,4);
                    if (std::memcmp(expected.data()+i,actual.data()+i,4)) {
                        if (mismatches++ < 3) std::fprintf(stderr,"tile float=%zu expected=%a actual=%a\n",(i-guard)/4,double(a),double(b));
                        max_abs = std::max(max_abs,std::abs(a-b));
                    }
                }
                std::fprintf(stderr,"tile mismatches=%zu max_abs=%g\n",mismatches,double(max_abs));
                std::fprintf(stderr,"tile mismatch width=%d TG%d groups=%d splits=%d active=%d spans=%d flags=%u %s/%s\n",
                    width,queries,groups,splits,active,span_count,flags,ggml_type_name(key),ggml_type_name(value));
                throw std::runtime_error("resumed tile output or native reduction metadata differs from stock");
            }
        }
    } catch (...) {
        (void) cudaEventDestroy(ready); (void) cudaEventDestroy(done);
        (void) cudaStreamDestroy(copy_stream); (void) cudaStreamDestroy(stream); throw;
    }
    check_cuda(cudaEventDestroy(ready)); check_cuda(cudaEventDestroy(done));
    check_cuda(cudaStreamDestroy(copy_stream)); check_cuda(cudaStreamDestroy(stream));
}

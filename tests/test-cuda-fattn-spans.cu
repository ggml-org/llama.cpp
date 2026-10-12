#include "fattn-mma-f16.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

static void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct device_allocation {
    char * data = nullptr;
    explicit device_allocation(size_t bytes) { check_cuda(cudaMalloc(&data, bytes)); }
    ~device_allocation() { if (data) (void) cudaFree(data); }
    device_allocation(const device_allocation &) = delete;
    device_allocation & operator=(const device_allocation &) = delete;
};

static bool benchmark_mma = false;

// Hold the first upload without a clock-based delay, then fence every physical-ring reuse.
struct mma_wave_queue {
    cudaStream_t copy = nullptr, compute = nullptr;
    cudaEvent_t ready = nullptr, consumed = nullptr;
    ggml_cuda_kv_span * host = nullptr;
    std::atomic<bool> released{false};
    mma_wave_queue() {
        check_cuda(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
        check_cuda(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));
        check_cuda(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
        check_cuda(cudaEventCreateWithFlags(&consumed,cudaEventDisableTiming));
        check_cuda(cudaMallocHost(&host,3*sizeof(*host)));
    }
    ~mma_wave_queue() {
        released.store(true,std::memory_order_release);
        if (copy) (void) cudaStreamSynchronize(copy);
        if (compute) (void) cudaStreamSynchronize(compute);
        if (host) (void) cudaFreeHost(host);
        if (ready) (void) cudaEventDestroy(ready);
        if (consumed) (void) cudaEventDestroy(consumed);
        if (copy) (void) cudaStreamDestroy(copy);
        if (compute) (void) cudaStreamDestroy(compute);
    }
    static void hold(void * data) {
        auto * gate=static_cast<std::atomic<bool> *>(data);
        while (!gate->load(std::memory_order_acquire)) std::this_thread::yield();
    }
};

// Compare complete shared tiles, including untouched padding and guard bytes.
template<int stride, int rows, bool spanned, bool asynchronous>
static __global__ void load_probe(const half2 * dense, const ggml_cuda_kv_span * spans, int count,
        bool value, int head, int first, int feature, int width, int live, char * output) {
    constexpr int guard = 128, tile_bytes = stride*rows*int(sizeof(half2));
    extern __shared__ char storage[];
    const int thread = threadIdx.y*blockDim.x+threadIdx.x;
    const int threads = blockDim.x*blockDim.y;
    for (int i = thread; i < tile_bytes+2*guard; i += threads) storage[i] = char(0x5a);
    __syncthreads();
    auto * tile = reinterpret_cast<half2 *>(storage+guard);
    if constexpr (spanned) {
        flash_attn_ext_f16_load_span_tile<stride,4,rows,asynchronous>(
            spans, count, value, head, first, feature, tile, width, live);
    } else {
        flash_attn_ext_f16_load_tile<stride,4,rows,false,true,false>(
            dense+head*257*128+feature, tile, width, 128, first, live, nullptr);
    }
    if constexpr (asynchronous) cp_async_wait_all();
    __syncthreads();
    for (int i = thread; i < tile_bytes+2*guard; i += threads) output[i] = storage[i];
}

#include "fattn-quant-test.h"

// Pack independently guarded resident/ring intervals with the wrapped interval first in physical memory.
struct span_fixture {
    static constexpr int heads = 2, pairs = 128, guard = 128;
    int tokens, capacity;
    device_allocation dense, packed, descriptors;
    std::vector<char> original;
    std::vector<ggml_cuda_kv_span> spans;
    bool value;
    span_fixture(int count, bool interleaved, bool value, int active = 257, int padded = 257, bool aligned = false) :
        tokens(active), capacity(padded), dense(size_t(padded)*heads*pairs*sizeof(half2)),
        packed(2*size_t(active)*heads*pairs*sizeof(half2)+12*guard), descriptors(3*sizeof(ggml_cuda_kv_span)), value(value) {
        std::vector<ggml_fp16_t> source[2];
        for (int plane = 0; plane < 2; ++plane) {
            source[plane].resize(size_t(tokens)*heads*pairs*2);
            for (size_t i = 0; i < source[plane].size(); ++i) source[plane][i] = i%17 == 0 ? ggml_fp16_t(0x8000) :
                ggml_fp32_to_fp16(float(int((i*19+plane*7)%251)-125)/128.0f);
        }
        std::vector<ggml_fp16_t> padded_source(size_t(capacity)*heads*pairs*2, 0);
        for (int head = 0; head < heads; ++head) std::memcpy(
            padded_source.data()+size_t(head)*capacity*pairs*2,
            source[value].data()+size_t(head)*tokens*pairs*2,size_t(tokens)*pairs*4);
        check_cuda(cudaMemcpy(dense.data, padded_source.data(), padded_source.size()*2, cudaMemcpyHostToDevice));
        original.resize(2*source[0].size()*2+12*guard, char(0x5a));
        std::vector<int> cuts = count == 1 ? std::vector<int>{0,tokens} :
            count == 2 ? std::vector<int>{0,17,tokens} : aligned && tokens >= 768 ?
            std::vector<int>{0,256,512,tokens} : std::vector<int>{0,17,std::min(127,tokens-1),tokens};
        spans.resize(count);
        size_t cursor = 0;
        for (int physical = 0; physical < count; ++physical) {
            const int index = count == 3 ? (physical == 0 ? 0 : physical == 1 ? 2 : 1) : physical;
            const int length = cuts[index+1]-cuts[index];
            const size_t plane_bytes = size_t(length)*heads*pairs*4;
            const size_t k_offset = cursor+guard, v_offset = k_offset+plane_bytes+2*guard;
            auto & span = spans[index];
            span = {packed.data+k_offset, packed.data+v_offset, cuts[index], length,
                interleaved ? heads*pairs*4 : pairs*4, interleaved ? heads*pairs*4 : pairs*4,
                interleaved ? pairs*4 : length*pairs*4, interleaved ? pairs*4 : length*pairs*4};
            for (int row = 0; row < length; ++row) for (int head = 0; head < heads; ++head) {
                const size_t offset = size_t(row)*span.k_token_stride+size_t(head)*span.k_head_stride;
                const size_t src = size_t(head*tokens+cuts[index]+row)*pairs*2;
                std::memcpy(original.data()+k_offset+offset, source[0].data()+src, pairs*4);
                std::memcpy(original.data()+v_offset+offset, source[1].data()+src, pairs*4);
            }
            cursor = v_offset+plane_bytes+guard;
        }
        if (!ggml_cuda_kv_spans_f16_validate(spans.data(), spans.size(), tokens, heads, pairs*2)) throw std::runtime_error("invalid fixture spans");
        check_cuda(cudaMemcpy(packed.data, original.data(), original.size(), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(descriptors.data, spans.data(), spans.size()*sizeof(spans[0]), cudaMemcpyHostToDevice));
    }
    void check_source() {
        std::vector<char> actual(original.size());
        check_cuda(cudaMemcpy(actual.data(), packed.data, actual.size(), cudaMemcpyDeviceToHost));
        if (actual != original) throw std::runtime_error("span source or guard changed");
    }
};

template<int stride, int rows, bool asynchronous>
static size_t check_loads(bool quick) {
    constexpr size_t bytes = size_t(stride)*rows*sizeof(half2)+256;
    device_allocation control{bytes}, spanned{bytes};
    std::vector<char> expected(bytes), actual(bytes);
    size_t cases = 0;
    for (int count : {1,2,3}) for (bool interleaved : {false,true}) for (bool value : {false,true}) {
        span_fixture fixture(count,interleaved,value);
        for (int head : {0,1}) for (int first : {0,16,17,31,63,126,127,224,256,257}) {
            for (int feature : {0,4,32}) for (int live : {0,1,rows-1,rows}) {
                const int valid = std::min(live,fixture.tokens-first);
                const int width = feature == 0 ? 128 : 32;
                if (quick && (head || feature || first != 16 || live != rows)) continue;
                const dim3 block(32,4);
                load_probe<stride,rows,false,false><<<1,block,bytes>>>(
                    reinterpret_cast<const half2 *>(fixture.dense.data), nullptr, 0, value, head, first, feature, width, valid, control.data);
                check_cuda(cudaGetLastError());
                load_probe<stride,rows,true,asynchronous><<<1,block,bytes>>>(
                    nullptr,reinterpret_cast<const ggml_cuda_kv_span *>(fixture.descriptors.data),count,
                    value,head,first,feature,width,valid,spanned.data);
                check_cuda(cudaGetLastError());
                check_cuda(cudaMemcpy(expected.data(), control.data, bytes, cudaMemcpyDeviceToHost));
                check_cuda(cudaMemcpy(actual.data(), spanned.data, bytes, cudaMemcpyDeviceToHost));
                if (actual != expected) throw std::runtime_error("FP16 span tile or guard differs from stock");
                ++cases;
            }
        }
        fixture.check_source();
    }
    return cases;
}

// Run identical stock split/fixup schedules over contiguous and fragmented FP16 inputs.
template<int columns, ggml_type key_type = GGML_TYPE_F16, ggml_type value_type = GGML_TYPE_F16>
static void check_attention(int queries, int active, int count, bool interleaved, int split_mode, float gain,
        quant_span_fixture * encoded = nullptr) {
    constexpr int dim = 256, ratio = 6, heads = span_fixture::heads*ratio;
    const int capacity = (active+255)/256*256;
    span_fixture key(count,interleaved,false,active,capacity,benchmark_mma), value(count,interleaved,true,active,capacity,benchmark_mma);
    device_allocation prototype(size_t(capacity)*span_fixture::heads*dim*2);
    check_cuda(cudaMemset(prototype.data,0,size_t(capacity)*span_fixture::heads*dim*2));
    device_allocation q(size_t(dim)*queries*heads*4), mask(size_t(capacity)*queries*2);
    std::vector<float> q_host(size_t(dim)*queries*heads);
    for (size_t i = 0; i < q_host.size(); ++i) q_host[i] = gain*float(int((i*23)%263)-131)/256.0f;
    std::vector<ggml_fp16_t> mask_host(size_t(capacity)*queries);
    for (int query = 0; query < queries; ++query) for (int row = 0; row < capacity; ++row) {
        mask_host[size_t(query)*capacity+row] = ggml_fp32_to_fp16(row <= active-queries+query ? 0.0f : -INFINITY);
    }
    check_cuda(cudaMemcpy(q.data,q_host.data(),q_host.size()*4,cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(mask.data,mask_host.data(),mask_host.size()*2,cudaMemcpyHostToDevice));
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties,0));
    const int cc = properties.major*100+properties.minor*10, ncols = columns*8;
    const int threads = ggml_cuda_fattn_mma_get_nthreads(dim,dim,ncols,cc), warps = threads/32;
    const int rows = ggml_cuda_fattn_mma_get_nbatch_fa(dim,dim,ncols,cc);
    const int stages = ggml_cuda_fattn_mma_get_nstages(dim,dim,columns,8,cc);
    const int k2 = ggml_cuda_fattn_mma_get_nbatch_K2(dim,dim,ncols,cc), v2 = ggml_cuda_fattn_mma_get_nbatch_V2(dim,dim,ncols,cc);
    const bool swizzled = ggml_cuda_fattn_mma_get_swizzled(dim,dim,ncols,cc);
    const size_t k_stride = k2+(swizzled ? 0 : 4), v_stride = v2+(swizzled ? 0 : 4);
    const size_t kv_bytes = size_t(rows)*(stages > 1 ? k_stride+v_stride : std::max(k_stride,v_stride))*4;
    const size_t q_bytes = size_t(ncols)*(dim/2+4)*4, mask_bytes = size_t(columns)*(rows/2+4)*4;
    const size_t combine = size_t(warps)*std::min(ncols,get_cols_per_warp(cc))*
        (ggml_cuda_fattn_mma_get_nbatch_combine(dim,dim,ncols,cc)+4)*4;
    const size_t shared = std::max(combine,ggml_cuda_fattn_mma_get_Q_in_reg(dim,dim,ncols,cc) ?
        std::max(q_bytes,kv_bytes+mask_bytes) : q_bytes+kv_bytes+mask_bytes);
    const size_t span_offset = (shared+15)/16*16, span_shared = span_offset+3*sizeof(ggml_cuda_kv_span);
    auto control_kernel = flash_attn_ext_f16<dim,dim,columns,8,false,false,false>;
    auto span_kernel = flash_attn_ext_f16_spans<dim,dim,columns,8,false,false,key_type,value_type>;
    auto resume_kernel = flash_attn_ext_f16_spans<dim,dim,columns,8,false,false,key_type,value_type,true>;
    check_cuda(cudaFuncSetAttribute(control_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared)));
    check_cuda(cudaFuncSetAttribute(span_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(span_shared)));
    check_cuda(cudaFuncSetAttribute(resume_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(span_shared)));
    int occupancy = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,control_kernel,threads,shared));
    int span_occupancy = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&span_occupancy,span_kernel,threads,span_shared));
    if (!span_occupancy) throw std::runtime_error("span kernel has no resource-valid resident block");
    ggml_cuda_fattn_stream_k_plan plan;
    const int tiles = capacity/rows, dst_tiles = span_fixture::heads;
    if (!ggml_cuda_fattn_stream_k_plan_make(dst_tiles,tiles,occupancy*properties.multiProcessorCount,
            cc >= GGML_CUDA_CC_ADA_LOVELACE,false,plan)) throw std::runtime_error("invalid stock launch plan");
    const int blocks = split_mode == 1 ? 1 : split_mode == 2 ? 3 : plan.blocks;
    constexpr size_t guard = 128;
    const size_t out_bytes = size_t(dim)*queries*heads*4, fixup_bytes = size_t(blocks)*ncols*(2+dim/2)*sizeof(float2);
    ggml_cuda_fattn_mma_resume_layout resume_layout;
    if (!ggml_cuda_fattn_mma_resume_layout_make(dst_tiles,blocks,threads,columns,8,resume_layout))
        throw std::runtime_error("invalid MMA resume layout");
    device_allocation state(resume_layout.bytes+2*guard);
    check_cuda(cudaMemset(state.data,0x5a,resume_layout.bytes+2*guard));
    const size_t key_row=ggml_row_size(key_type,dim), value_row=ggml_row_size(value_type,dim);
    device_allocation ring(size_t(rows)*span_fixture::heads*(key_row+value_row)+12*guard);
    device_allocation outputs[4] = {device_allocation(out_bytes+2*guard),device_allocation(out_bytes+2*guard),device_allocation(out_bytes+2*guard),device_allocation(out_bytes+2*guard)};
    device_allocation fixups[4] = {device_allocation(fixup_bytes+2*guard),device_allocation(fixup_bytes+2*guard),device_allocation(fixup_bytes+2*guard),device_allocation(fixup_bytes+2*guard)};
    std::vector<char> output_bytes[4], metadata[4];
    for (int implementation = 0; implementation < 4; ++implementation) {
        check_cuda(cudaMemset(outputs[implementation].data,0x5a,out_bytes+2*guard));
        check_cuda(cudaMemset(outputs[implementation].data+guard,0,out_bytes));
        check_cuda(cudaMemset(fixups[implementation].data,0x5a,fixup_bytes+2*guard));
        check_cuda(cudaMemset(fixups[implementation].data+guard,0,fixup_bytes));
        auto * out = reinterpret_cast<float *>(outputs[implementation].data+guard);
        auto * fixup = reinterpret_cast<float2 *>(fixups[implementation].data+guard);
        const int q1 = dim*4, q2 = queries*q1, q3 = heads*q2;
        const int k1 = dim*2, k2_stride = capacity*k1, k3 = span_fixture::heads*k2_stride;
        const int m1 = capacity*2, m2 = queries*m1;
        const char * k_data = implementation == 0 ? (encoded ? encoded->dense_key.data : key.dense.data) : prototype.data;
        const char * v_data = implementation == 0 ? (encoded ? encoded->dense_value.data : value.dense.data) : prototype.data;
#define ATTENTION_ARGS q.data,k_data,v_data,mask.data,nullptr,nullptr,out,fixup, \
    1.0f/16.0f,0.0f,1.0f,1.0f,0u,0.0f,dim,init_fastdiv_values(queries),heads,1,q1,q2,q3, \
    dim,capacity,span_fixture::heads,1,k1,k2_stride,int64_t(k3),k1,k2_stride,int64_t(k3), \
    queries,1,1,m1,m2,int64_t(m2)
        if (implementation == 0) control_kernel<<<blocks,dim3(32,warps),shared>>>(ATTENTION_ARGS);
        else if (implementation == 1) span_kernel<<<blocks,dim3(32,warps),span_shared>>>(ATTENTION_ARGS,
            reinterpret_cast<const ggml_cuda_kv_span *>(encoded ? encoded->descriptors.data : key.descriptors.data),
            count,int(span_offset),encoded ? encoded->conversion_flags : 3u,ggml_cuda_fattn_mma_resume_control{});
        else if (implementation == 3) {
            check_cuda(cudaMemcpy(key.descriptors.data,key.spans.data(),key.spans.size()*sizeof(key.spans[0]),cudaMemcpyHostToDevice));
            resume_kernel<<<blocks,dim3(32,warps),span_shared>>>(ATTENTION_ARGS,
                reinterpret_cast<const ggml_cuda_kv_span *>(encoded ? encoded->descriptors.data : key.descriptors.data),
                count,int(span_offset),encoded ? encoded->conversion_flags : 3u,{state.data+guard,0,capacity});
        } else {
            check_cuda(cudaStreamSynchronize(nullptr));
            mma_wave_queue queue;
            const auto & sources=encoded ? encoded->spans : key.spans;
            for (int first=0;first<active;first+=rows) {
                const int end=first+rows >= active ? capacity : first+rows, live=std::min(end,active)-first;
                if (live <= 0) throw std::runtime_error("MMA fixture has an empty physical wave");
                // Reuse one small physical ring; neither the native extent nor the split assignment changes.
                check_cuda(cudaMemsetAsync(ring.data,0x5a,size_t(rows)*span_fixture::heads*(key_row+value_row)+12*guard,queue.copy));
                if (!first) check_cuda(cudaLaunchHostFunc(queue.copy,mma_wave_queue::hold,&queue.released));
                std::vector<ggml_cuda_kv_span> wave;
                size_t offset=0;
                for (const auto & source : sources) {
                    const int begin=std::max(first,source.first), stop=std::min(first+live,source.first+source.tokens);
                    if (begin >= stop) continue;
                    const int length=stop-begin;
                    const size_t kb=offset+guard, vb=kb+size_t(length)*span_fixture::heads*key_row+2*guard;
                    ggml_cuda_kv_span part{ring.data+kb,ring.data+vb,begin,length,
                        int64_t(key_row),int64_t(value_row),int64_t(length*key_row),int64_t(length*value_row)};
                    for (int head=0;head<span_fixture::heads;++head) {
                        check_cuda(cudaMemcpy2DAsync(const_cast<char *>(part.k)+head*part.k_head_stride,key_row,
                            source.k+head*source.k_head_stride+(begin-source.first)*source.k_token_stride,
                            source.k_token_stride,key_row,length,cudaMemcpyDeviceToDevice,queue.copy));
                        check_cuda(cudaMemcpy2DAsync(const_cast<char *>(part.v)+head*part.v_head_stride,value_row,
                            source.v+head*source.v_head_stride+(begin-source.first)*source.v_token_stride,
                            source.v_token_stride,value_row,length,cudaMemcpyDeviceToDevice,queue.copy));
                    }
                    wave.push_back(part);
                    offset=vb+size_t(length)*span_fixture::heads*value_row+guard;
                }
                std::memcpy(queue.host,wave.data(),wave.size()*sizeof(wave[0]));
                check_cuda(cudaMemcpyAsync(key.descriptors.data,queue.host,wave.size()*sizeof(wave[0]),cudaMemcpyHostToDevice,queue.copy));
                check_cuda(cudaEventRecord(queue.ready,queue.copy));
                check_cuda(cudaStreamWaitEvent(queue.compute,queue.ready,0));
                resume_kernel<<<blocks,dim3(32,warps),span_shared,queue.compute>>>(ATTENTION_ARGS,
                    reinterpret_cast<const ggml_cuda_kv_span *>(key.descriptors.data),int(wave.size()),int(span_offset),
                    encoded ? encoded->conversion_flags : 3u,{state.data+guard,first,end});
                check_cuda(cudaGetLastError());
                check_cuda(cudaEventRecord(queue.consumed,queue.compute));
                if (!first) {
                    if (cudaEventQuery(queue.ready) != cudaErrorNotReady)
                        throw std::runtime_error("delayed MMA copy completed before its gate opened");
                    queue.released.store(true,std::memory_order_release);
                }
                check_cuda(cudaEventSynchronize(queue.consumed));
                if (end < capacity && blocks == dst_tiles*tiles) {
                    std::vector<char> unpublished(out_bytes);
                    check_cuda(cudaMemcpy(unpublished.data(),outputs[implementation].data+guard,out_bytes,cudaMemcpyDeviceToHost));
                    if (!std::all_of(unpublished.begin(),unpublished.end(),[](char x){return x == 0;}))
                        throw std::runtime_error("singleton MMA slice published before the last wave");
                }
            }
        }
        if (benchmark_mma && implementation < 2) {
            const auto launch=[&] {
                if (!implementation) control_kernel<<<blocks,dim3(32,warps),shared>>>(ATTENTION_ARGS);
                else span_kernel<<<blocks,dim3(32,warps),span_shared>>>(ATTENTION_ARGS,
                    reinterpret_cast<const ggml_cuda_kv_span *>(encoded ? encoded->descriptors.data : key.descriptors.data),
                    count,int(span_offset),encoded ? encoded->conversion_flags : 3u,ggml_cuda_fattn_mma_resume_control{});
            };
            for (int i=0;i<16;++i) launch();
            cudaEvent_t start=nullptr, stop=nullptr;
            check_cuda(cudaEventCreate(&start)); check_cuda(cudaEventCreate(&stop));
            check_cuda(cudaEventRecord(start));
            for (int i=0;i<128;++i) launch();
            check_cuda(cudaEventRecord(stop)); check_cuda(cudaEventSynchronize(stop));
            float elapsed=0;
            check_cuda(cudaEventElapsedTime(&elapsed,start,stop));
            check_cuda(cudaEventDestroy(start)); check_cuda(cudaEventDestroy(stop));
            std::printf("MMA timing K=%s V=%s TG%d tokens=%d spans=%d implementation=%s us=%.3f scratch=%zu\n",
                ggml_type_name(key_type),ggml_type_name(value_type),queries,capacity,count,
                implementation ? "spanned" : "stock",double(elapsed)*1000/128,resume_layout.bytes);
        }
#undef ATTENTION_ARGS
        check_cuda(cudaGetLastError());
        metadata[implementation].resize(fixup_bytes+2*guard);
        check_cuda(cudaMemcpy(metadata[implementation].data(),fixups[implementation].data,metadata[implementation].size(),cudaMemcpyDeviceToHost));
        if (blocks%dst_tiles == 0 && blocks > dst_tiles) {
            flash_attn_stream_k_fixup_uniform<dim,columns,8><<<dim3(dst_tiles,columns,8),dim>>>(
                out,fixup,queries,heads,span_fixture::heads,blocks,ratio,blocks/dst_tiles,
                init_fastdiv_values(dst_tiles),init_fastdiv_values(1),init_fastdiv_values(1));
        } else if (dst_tiles%blocks != 0) {
            flash_attn_stream_k_fixup_general<dim,columns,8><<<dim3(blocks,columns,8),dim>>>(
                out,fixup,queries,heads,ratio,tiles*dst_tiles,
                init_fastdiv_values(tiles*dst_tiles),init_fastdiv_values(tiles),init_fastdiv_values(tiles),init_fastdiv_values(tiles));
        }
        check_cuda(cudaGetLastError());
        output_bytes[implementation].resize(out_bytes+2*guard);
        check_cuda(cudaMemcpy(output_bytes[implementation].data(),outputs[implementation].data,output_bytes[implementation].size(),cudaMemcpyDeviceToHost));
        for (const auto * bytes : {&output_bytes[implementation],&metadata[implementation]}) {
            if (!std::all_of(bytes->begin(),bytes->begin()+guard,[](char x){return x == char(0x5a);}) ||
                    !std::all_of(bytes->end()-guard,bytes->end(),[](char x){return x == char(0x5a);})) throw std::runtime_error("MMA output or fixup guard changed");
        }
    }
    if (metadata[0] != metadata[1]) throw std::runtime_error("MMA split accumulator or softmax metadata drift");
    if (output_bytes[0] != output_bytes[1]) throw std::runtime_error("MMA final output bits differ from stock");
    if (metadata[0] != metadata[3]) throw std::runtime_error("one-wave resumed MMA split metadata differs from stock");
    if (output_bytes[0] != output_bytes[3]) throw std::runtime_error("one-wave resumed MMA differs from stock before checkpointing");
    if (metadata[0] != metadata[2]) throw std::runtime_error("resumed MMA split accumulator or softmax metadata drift");
    if (output_bytes[0] != output_bytes[2]) {
        for (size_t i=guard;i<guard+out_bytes;i+=sizeof(float)) {
            uint32_t a,b; float x,y;
            std::memcpy(&a,output_bytes[0].data()+i,4); std::memcpy(&b,output_bytes[2].data()+i,4);
            if (a == b) continue;
            std::memcpy(&x,&a,4); std::memcpy(&y,&b,4);
            std::fprintf(stderr,"MMA first output mismatch: element=%zu stock=%08x resumed=%08x abs=%g\n",
                (i-guard)/4,a,b,double(std::abs(x-y)));
            break;
        }
        throw std::runtime_error("resumed MMA final output bits differ from stock");
    }
    std::vector<char> checkpoint(resume_layout.bytes+2*guard);
    check_cuda(cudaMemcpy(checkpoint.data(),state.data,checkpoint.size(),cudaMemcpyDeviceToHost));
    if (!std::all_of(checkpoint.begin(),checkpoint.begin()+guard,[](char x){return x == char(0x5a);}) ||
            !std::all_of(checkpoint.end()-guard,checkpoint.end(),[](char x){return x == char(0x5a);}))
        throw std::runtime_error("MMA checkpoint guard changed");
    bool nonzero = false;
    for (size_t i = 0; i < out_bytes; i += sizeof(float)) {
        float result;
        std::memcpy(&result,output_bytes[1].data()+guard+i,sizeof(result));
        if (!std::isfinite(result)) throw std::runtime_error("nonfinite MMA output");
        nonzero |= result != 0.0f;
    }
    if (!nonzero) throw std::runtime_error("vacuous zero-output attention fixture");
    key.check_source(); value.check_source();
    if (encoded) encoded->check_source();
}

static size_t check_attention_matrix(bool quick) {
    size_t cases = 0;
    for (int queries : {1,2,3,4,5,6,7,8}) for (int active : {31,257,1023,8192}) for (int count : {1,2,3}) {
        for (bool interleaved : {false,true}) for (int split : {0,1,2}) for (float gain : {1.0f,16.0f}) {
            if (benchmark_mma) {
                if ((queries != 3 && queries != 4 && queries != 8) || active != 8192 || count == 2 || !interleaved || split || gain != 1.0f) continue;
            } else if (quick && (active != 257 || count != 3 || interleaved || gain != 1.0f)) continue;
            try {
                if (queries == 1) check_attention<1>(queries,active,count,interleaved,split,gain);
                else if (queries == 2) check_attention<2>(queries,active,count,interleaved,split,gain);
                else if (queries <= 4) check_attention<4>(queries,active,count,interleaved,split,gain);
                else check_attention<8>(queries,active,count,interleaved,split,gain);
            } catch (...) {
                std::fprintf(stderr,"MMA case: TG%d active=%d spans=%d interleaved=%d split=%d gain=%g\n",
                    queries,active,count,int(interleaved),split,double(gain));
                throw;
            }
            ++cases;
        }
    }
    return cases;
}

template<ggml_type key, ggml_type value>
static size_t check_encoded_attention_pair(ggml_backend_t backend, bool quick, bool deep) {
    size_t cases = 0;
    for (int queries : {2,3,4,5,6,7,8}) for (int active : {31,257,1023,8192}) for (int count : {1,2,3}) {
        for (bool interleaved : {false,true}) for (int split : {0,1,2}) for (uint32_t flags : {0u,1u,2u,3u}) {
            if (benchmark_mma) {
                if ((queries != 3 && queries != 4 && queries != 8) || active != 8192 || count == 2 || !interleaved || split || flags != 0) continue;
            } else {
                if (!deep && (active != 257 || count != 3 || split || interleaved)) continue;
                if (quick && (active != 257 || count != 3 || interleaved || split)) continue;
            }
            quant_span_fixture fixture(backend,key,value,active,(active+255)/256*256,count,interleaved,flags,false,benchmark_mma);
            try {
                if (queries == 2) check_attention<2,key,value>(queries,active,count,interleaved,split,1.0f,&fixture);
                else if (queries <= 4) check_attention<4,key,value>(queries,active,count,interleaved,split,1.0f,&fixture);
                else check_attention<8,key,value>(queries,active,count,interleaved,split,1.0f,&fixture);
            } catch (...) {
                std::fprintf(stderr,"encoded MMA case: K=%s V=%s TG%d active=%d spans=%d interleaved=%d split=%d flags=%u\n",
                    ggml_type_name(key),ggml_type_name(value),queries,active,count,int(interleaved),split,flags);
                throw;
            }
            ++cases;
        }
    }
    return cases;
}

template<ggml_type key>
static size_t check_encoded_attention_values(ggml_backend_t backend, bool quick) {
    size_t cases = 0;
#define CHECK_VALUE(V) cases += check_encoded_attention_pair<key,GGML_TYPE_##V>(backend,quick,key == GGML_TYPE_Q8_0 && GGML_TYPE_##V == GGML_TYPE_Q4_0)
    CHECK_VALUE(F16); CHECK_VALUE(BF16); CHECK_VALUE(Q8_0); CHECK_VALUE(Q4_0);
    CHECK_VALUE(Q4_1); CHECK_VALUE(Q5_0); CHECK_VALUE(Q5_1);
#undef CHECK_VALUE
    return cases;
}

template<ggml_type type>
static size_t check_encoded_tile_type(ggml_backend_t backend, bool quick) {
    size_t cases = check_encoded_tiles<type,128,false>(backend,quick)+check_encoded_tiles<type,132,false>(backend,quick);
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties,0));
    if (properties.major >= 8) cases += check_encoded_tiles<type,128,true>(backend,quick)+check_encoded_tiles<type,132,true>(backend,quick);
    return cases;
}

int main(int argc, char ** argv) {
    benchmark_mma=argc == 2 && !std::strcmp(argv[1],"--benchmark");
    const bool quick = argc == 2 && !std::strcmp(argv[1],"--quick");
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name("CUDA0",nullptr));
    if (!backend) return 77;
    try {
        if (argc == 2 && !std::strcmp(argv[1],"--case-mma-checkpoint")) {
            check_attention<1>(1,1023,3,false,1,16.0f);
            std::puts("MMA checkpoint stress case matches stock");
            return 0;
        }
        if (argc == 2 && !std::strcmp(argv[1],"--case-mma-checkpoint-tg2")) {
            check_attention<2>(2,8192,3,false,1,16.0f);
            std::puts("MMA TG2 checkpoint stress case matches stock");
            return 0;
        }
#ifdef FATTN_MMA_RESUME_ONLY
        const auto cases=check_attention_matrix(quick)+
            check_encoded_attention_pair<GGML_TYPE_Q8_0,GGML_TYPE_Q4_0>(backend.get(),quick,true)+
            check_encoded_attention_pair<GGML_TYPE_BF16,GGML_TYPE_BF16>(backend.get(),quick,false);
        std::printf("Native MMA resume: %zu cases passed, zero output/metadata mismatches\n",cases);
#else
        const auto conversion_cases = check_decoding<GGML_TYPE_Q8_0>(backend.get(),quick)+
            check_decoding<GGML_TYPE_Q4_0>(backend.get(),quick)+check_decoding<GGML_TYPE_Q4_1>(backend.get(),quick)+
            check_decoding<GGML_TYPE_Q5_0>(backend.get(),quick)+check_decoding<GGML_TYPE_Q5_1>(backend.get(),quick)+
            check_decoding<GGML_TYPE_F16>(backend.get(),quick)+check_decoding<GGML_TYPE_BF16>(backend.get(),quick);
        std::printf("Encoded conversion: %zu cases passed, zero byte mismatches\n",conversion_cases);
        const auto staged_cases = check_encoded_tile_type<GGML_TYPE_Q8_0>(backend.get(),quick)+
            check_encoded_tile_type<GGML_TYPE_Q4_0>(backend.get(),quick)+check_encoded_tile_type<GGML_TYPE_Q4_1>(backend.get(),quick)+
            check_encoded_tile_type<GGML_TYPE_Q5_0>(backend.get(),quick)+check_encoded_tile_type<GGML_TYPE_Q5_1>(backend.get(),quick)+
            check_encoded_tile_type<GGML_TYPE_BF16>(backend.get(),quick)+check_encoded_tile_type<GGML_TYPE_F16>(backend.get(),quick);
        std::printf("Encoded staging: %zu cases passed, zero byte mismatches\n",staged_cases);
        const auto quant_attention = check_encoded_attention_values<GGML_TYPE_Q8_0>(backend.get(),quick)+
            check_encoded_attention_values<GGML_TYPE_Q4_0>(backend.get(),quick)+check_encoded_attention_values<GGML_TYPE_Q4_1>(backend.get(),quick)+
            check_encoded_attention_values<GGML_TYPE_Q5_0>(backend.get(),quick)+check_encoded_attention_values<GGML_TYPE_Q5_1>(backend.get(),quick)+
            check_encoded_attention_values<GGML_TYPE_BF16>(backend.get(),quick)+check_encoded_attention_values<GGML_TYPE_F16>(backend.get(),quick);
        std::printf("Encoded MMA: %zu cases passed, zero output/metadata mismatches\n",quant_attention);
        size_t cases = check_loads<128,32,false>(quick)+check_loads<132,64,false>(quick);
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties,0));
        if (properties.major >= 8) cases += check_loads<128,32,true>(quick)+check_loads<132,64,true>(quick);
        std::printf("FP16 span tile probes: %zu cases passed, zero byte mismatches\n",cases);
        const auto attention_cases = check_attention_matrix(quick);
        std::printf("FP16 span MMA: %zu cases passed, zero output/metadata mismatches\n",attention_cases);
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FP16 span test failed: %s\n",error.what());
        return 1;
    }
}

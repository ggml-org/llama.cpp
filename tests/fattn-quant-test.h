#pragma once

#include "fattn-stock-conversion-test.h"

// Pack the same encoded rows into guarded, reordered intervals and retain the stock reference.
struct quant_span_fixture {
    static constexpr int heads = 2, dimension = 256, guard = 128;
    int tokens, capacity;
    ggml_type key_type, value_type;
    uint32_t conversion_flags;
    std::vector<char> original;
    std::vector<ggml_cuda_kv_span> spans;
    std::vector<ggml_fp16_t> reference[2];
    device_allocation packed, descriptors, dense_key, dense_value;
    quant_span_fixture(ggml_backend_t backend, ggml_type key, ggml_type value, int active, int padded,
            int count, bool interleaved, uint32_t flags, bool stress = false, bool aligned = false) : tokens(active), capacity(padded), key_type(key), value_type(value), conversion_flags(flags),
        packed(size_t(active)*heads*(ggml_row_size(key,dimension)+ggml_row_size(value,dimension))+12*guard),
        descriptors(3*sizeof(ggml_cuda_kv_span)),dense_key(size_t(padded)*heads*dimension*2),dense_value(size_t(padded)*heads*dimension*2) {
        const ggml_type types[2] = {key,value};
        std::vector<char> encoded[2];
        size_t row_bytes[2];
        for (int plane = 0; plane < 2; ++plane) {
            row_bytes[plane] = ggml_row_size(types[plane],dimension);
            encoded[plane].resize(size_t(padded)*heads*row_bytes[plane],0);
            std::vector<float> input(size_t(active)*dimension);
            for (int head = 0; head < heads; ++head) {
                for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i*19+plane*7+head*11)%251)-125)/128.0f;
                ggml_quantize_chunk(types[plane],input.data(),encoded[plane].data()+size_t(head)*padded*row_bytes[plane],0,active,dimension,nullptr);
                if (stress) {
                    // Exercise signed zeros, subnormal scales and large finite encoded values.
                    const ggml_fp16_t scales[8] = {0,0x8000,1,0x8001,ggml_fp32_to_fp16(0.1f),ggml_fp32_to_fp16(-0.1f),
                        ggml_fp32_to_fp16(256.0f),ggml_fp32_to_fp16(-256.0f)};
                    const uint16_t bf16_values[8] = {0,0x8000,1,0x8001,0x3eab,0xbeab,0x477f,0xc77f};
                    const uint16_t f16_values[8] = {0,0x8000,1,0x8001,0x3555,0xb555,0x7bff,0xfbff};
                    const size_t stride = ggml_type_size(types[plane]);
                    const size_t blocks = size_t(active)*dimension/ggml_blck_size(types[plane]);
                    for (size_t block = 0; block < blocks; ++block) {
                        char * dest = encoded[plane].data()+size_t(head)*padded*row_bytes[plane]+block*stride;
                        const auto code = (block+plane+head)%8;
                        const uint16_t bits = types[plane] == GGML_TYPE_BF16 ? bf16_values[code] :
                            types[plane] == GGML_TYPE_F16 ? f16_values[code] : scales[code];
                        std::memcpy(dest,&bits,2);
                        if (types[plane] == GGML_TYPE_Q4_1 || types[plane] == GGML_TYPE_Q5_1) {
                            const uint16_t offset = scales[(code+3)%8];
                            std::memcpy(dest+2,&offset,2);
                        }
                    }
                }
            }
            reference[plane] = stock_f16_rows(backend,types[plane],encoded[plane],padded,heads,!(flags&(1u<<plane)));
        }
        check_cuda(cudaMemcpy(dense_key.data,reference[0].data(),reference[0].size()*2,cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(dense_value.data,reference[1].data(),reference[1].size()*2,cudaMemcpyHostToDevice));
        original.resize(size_t(active)*heads*(row_bytes[0]+row_bytes[1])+12*guard,char(0x5a));
        const std::vector<int> cuts = count == 1 ? std::vector<int>{0,active} : count == 2 ?
            std::vector<int>{0,17,active} : aligned && active >= 768 ?
            std::vector<int>{0,256,512,active} : std::vector<int>{0,17,std::min(127,active-1),active};
        spans.resize(count);
        size_t cursor = 0;
        for (int physical = 0; physical < count; ++physical) {
            const int index = count == 3 ? (physical == 0 ? 0 : physical == 1 ? 2 : 1) : physical;
            const int rows = cuts[index+1]-cuts[index];
            const size_t k = cursor+guard, v = k+size_t(rows)*heads*row_bytes[0]+2*guard;
            auto & span = spans[index];
            span = {packed.data+k,packed.data+v,cuts[index],rows,
                int64_t(row_bytes[0]*(interleaved ? heads : 1)),int64_t(row_bytes[1]*(interleaved ? heads : 1)),
                int64_t(row_bytes[0]*(interleaved ? 1 : rows)),int64_t(row_bytes[1]*(interleaved ? 1 : rows))};
            for (int row = 0; row < rows; ++row) for (int head = 0; head < heads; ++head) {
                std::memcpy(original.data()+k+row*span.k_token_stride+head*span.k_head_stride,
                    encoded[0].data()+size_t(head*padded+cuts[index]+row)*row_bytes[0],row_bytes[0]);
                std::memcpy(original.data()+v+row*span.v_token_stride+head*span.v_head_stride,
                    encoded[1].data()+size_t(head*padded+cuts[index]+row)*row_bytes[1],row_bytes[1]);
            }
            cursor = v+size_t(rows)*heads*row_bytes[1]+guard;
        }
        if (!ggml_cuda_kv_spans_validate(spans.data(),count,active,heads,row_bytes[0],row_bytes[1])) throw std::runtime_error("encoded span validation failed");
        check_cuda(cudaMemcpy(packed.data,original.data(),original.size(),cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(descriptors.data,spans.data(),spans.size()*sizeof(spans[0]),cudaMemcpyHostToDevice));
    }
    void check_source() {
        std::vector<char> actual(original.size());
        check_cuda(cudaMemcpy(actual.data(),packed.data,actual.size(),cudaMemcpyDeviceToHost));
        if (actual != original) throw std::runtime_error("quantized source or guard changed");
    }
};

template<ggml_type type>
static __global__ void decode_probe(const ggml_cuda_kv_span * spans, int count, bool value, int tokens, int head, bool contiguous, half2 * output) {
    const int pair = blockIdx.x*blockDim.x+threadIdx.x;
    if (pair >= tokens*128) return;
    const int row = pair/128, feature = 2*(pair%128);
    const int index = flash_attn_ext_f16_span_index(spans,count,row);
    const char * source = nullptr;
    if (index >= 0) {
        const auto & span = spans[index];
        source = (value ? span.v : span.k)+int64_t(row-span.first)*(value ? span.v_token_stride : span.k_token_stride)+
            int64_t(head)*(value ? span.v_head_stride : span.k_head_stride);
    }
    output[pair] = ggml_cuda_fattn_decode_pair<type>(source,feature,contiguous);
}

template<ggml_type type>
static size_t check_decoding(ggml_backend_t backend, bool quick) {
    size_t cases = 0;
    for (int count : {1,2,3}) for (bool interleaved : {false,true}) for (bool strided : {false,true}) for (bool stress : {false,true}) {
        if (quick && (count != 3 || interleaved)) continue;
        quant_span_fixture fixture(backend,type,type,257,512,count,interleaved,strided ? 0u : 3u,stress);
        device_allocation output(size_t(512)*256*2+256);
        for (int head : {0,1}) for (bool value : {false,true}) {
            check_cuda(cudaMemset(output.data,0x5a,size_t(512)*256*2+256));
            decode_probe<type><<<(512*128+255)/256,256>>>(reinterpret_cast<const ggml_cuda_kv_span *>(fixture.descriptors.data),
                count,value,512,head,!strided,reinterpret_cast<half2 *>(output.data+128));
            check_cuda(cudaGetLastError());
            std::vector<char> actual(size_t(512)*256*2+256);
            check_cuda(cudaMemcpy(actual.data(),output.data,actual.size(),cudaMemcpyDeviceToHost));
            const auto * expected = fixture.reference[value].data()+size_t(head)*512*256;
            if (std::memcmp(actual.data()+128,expected,size_t(512)*256*2)) {
                std::fprintf(stderr,"conversion mismatch: type=%s spans=%d interleaved=%d strided=%d head=%d V=%d stress=%d\n",
                    ggml_type_name(type),count,int(interleaved),int(strided),head,int(value),int(stress));
                size_t mismatches = 0;
                for (size_t i = 0; i < size_t(512)*256; ++i) {
                    ggml_fp16_t got;
                    std::memcpy(&got,actual.data()+128+2*i,2);
                    if (got != expected[i]) {
                        if (mismatches < 8) std::fprintf(stderr," conversion element=%zu actual=%04x expected=%04x\n",i,unsigned(got),unsigned(expected[i]));
                        ++mismatches;
                    }
                }
                std::fprintf(stderr," conversion mismatches=%zu\n",mismatches);
                throw std::runtime_error("converted rows differ from stock FA");
            }
            if (!std::all_of(actual.begin(),actual.begin()+128,[](char x){return x == char(0x5a);}) ||
                    !std::all_of(actual.end()-128,actual.end(),[](char x){return x == char(0x5a);})) throw std::runtime_error("conversion output guard changed");
            ++cases;
        }
        fixture.check_source();
    }
    return cases;
}

template<ggml_type type, int stride, bool staged>
static __global__ void encoded_tile_probe(const ggml_cuda_kv_span * spans, int count, bool value, int head,
        int first, int feature, int width, int live, bool contiguous, char * output) {
    constexpr int rows = 32, raw_bytes = ggml_cuda_fattn_encoded_traits<type>::row_bytes(128)*rows;
    constexpr int tile_bytes = stride*rows*4, guard = 128, total = raw_bytes+tile_bytes+4*guard;
    extern __shared__ char shared[];
    const int thread = threadIdx.y*32+threadIdx.x;
    for (int i = thread; i < total; i += 128) shared[i] = char(0x5a);
    __syncthreads();
    auto * raw = shared+guard;
    auto * tile = reinterpret_cast<half2 *>(shared+raw_bytes+3*guard);
    if constexpr (staged) {
        ggml_cuda_fattn_prefetch_encoded_tile<type,4,rows>(spans,count,value,head,first,raw,128,live);
        cp_async_wait_all();
        __syncthreads();
        ggml_cuda_fattn_finish_encoded_tile<type,stride,4,rows>(raw,tile,128,feature,width,contiguous);
    } else ggml_cuda_fattn_load_encoded_tile<type,stride,4,rows>(
        spans,count,value,head,first,feature,tile,width,live,contiguous);
    __syncthreads();
    for (int i = thread; i < total; i += 128) output[i] = shared[i];
}

template<ggml_type type, int stride, bool staged>
static size_t check_encoded_tiles(ggml_backend_t backend, bool quick) {
    constexpr int rows = 32, raw_bytes = ggml_cuda_fattn_encoded_traits<type>::row_bytes(128)*rows;
    constexpr int tile_bytes = stride*rows*4, total = raw_bytes+tile_bytes+512;
    device_allocation actual_device(total), expected_device(tile_bytes+256), control(32*128*4);
    std::vector<char> actual(total), expected(tile_bytes+256);
    size_t cases = 0;
    for (int count : {1,2,3}) for (bool interleaved : {false,true}) for (bool strided : {false,true}) {
        if (quick && (count != 3 || interleaved)) continue;
        quant_span_fixture fixture(backend,type,type,257,512,count,interleaved,strided ? 0u : 3u);
        for (bool value : {false,true}) for (int head : {0,1}) for (int first : {0,16,127,240,257}) {
            for (int feature : {0,4,32}) for (int live : {0,1,31,32}) {
                if (quick && (head || first != 16 || feature || live != 32)) continue;
                const int valid = std::min(live,std::max(0,fixture.tokens-first)), width = feature == 0 ? 128 : 32;
                std::vector<ggml_fp16_t> control_rows(rows*256);
                for (int row = 0; row < rows; ++row) {
                    const int source_row = row < valid ? first+row : 511;
                    std::memcpy(control_rows.data()+row*256,
                        fixture.reference[value].data()+size_t(head*512+source_row)*256,512);
                }
                check_cuda(cudaMemcpy(control.data,control_rows.data(),control_rows.size()*2,cudaMemcpyHostToDevice));
                load_probe<stride,rows,false,false><<<1,dim3(32,4),tile_bytes+256>>>(
                    reinterpret_cast<const half2 *>(control.data),nullptr,0,false,0,0,feature,width,rows,expected_device.data);
                encoded_tile_probe<type,stride,staged><<<1,dim3(32,4),total>>>(
                    reinterpret_cast<const ggml_cuda_kv_span *>(fixture.descriptors.data),count,value,head,
                    first,feature,width,valid,!strided,actual_device.data);
                check_cuda(cudaGetLastError());
                check_cuda(cudaMemcpy(expected.data(),expected_device.data,expected.size(),cudaMemcpyDeviceToHost));
                check_cuda(cudaMemcpy(actual.data(),actual_device.data,actual.size(),cudaMemcpyDeviceToHost));
                if (std::memcmp(actual.data()+raw_bytes+256,expected.data(),expected.size())) throw std::runtime_error("encoded staging tile differs from stock conversion");
                if (!std::all_of(actual.begin(),actual.begin()+128,[](char x){return x == char(0x5a);}) ||
                        !std::all_of(actual.begin()+raw_bytes+128,actual.begin()+raw_bytes+256,[](char x){return x == char(0x5a);})) throw std::runtime_error("compressed staging guard changed");
                ++cases;
            }
        }
        fixture.check_source();
    }
    return cases;
}

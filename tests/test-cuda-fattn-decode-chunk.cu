#include "fattn-mma-quant-chunk.cuh"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Compare aligned/unaligned eight-value groups and zero padding to the original converter's bits.
template<ggml_type type>
static __global__ void compare_chunks(const char * rows, int row_bytes, int count, half2 * output) {
    const int chunk = int(blockIdx.x*blockDim.x+threadIdx.x);
    if (chunk >= (count+1)*125) return;
    const int row = chunk/125, first = chunk%125*2;
    const char * source = row == count ? nullptr : rows+row*row_bytes;
    half2 decoded[4];
    ggml_cuda_fattn_decode_strided_chunk8<type>(source,first,decoded);
    for (int pair = 0; pair < 4; ++pair) {
        output[chunk*8+pair] = ggml_cuda_fattn_decode_pair<type>(source,first+2*pair,false);
        output[chunk*8+4+pair] = decoded[pair];
    }
}

// Exercise signed packed extrema and subnormal/negative/maximum finite FP16 block scales.
template<ggml_type type>
static void qualify() {
    constexpr int count = 256, width = 256;
    const size_t row_bytes = ggml_row_size(type,width);
    std::vector<char> rows(row_bytes*count);
    std::vector<float> values(count*width);
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i*19%251)-125)/128;
    ggml_quantize_chunk(type,values.data(),rows.data(),0,count,width,nullptr);
    if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) {
        const uint16_t scales[] = {0,0x8000,1,0x8001,0x03ff,0x0400,0x3555,0x3c00,0xbc00,0x7bff};
        constexpr int bytes = ggml_cuda_fattn_encoded_traits<type>::bytes;
        for (int row = 0; row < count; ++row) for (int block = 0; block < width/32; ++block) {
            char * encoded = rows.data()+row*row_bytes+block*bytes;
            const uint16_t scale = scales[(row+block)%10];
            std::memcpy(encoded,&scale,2);
            for (int byte = 2; byte < bytes; ++byte) encoded[byte] = char(row+byte*31+block*7);
        }
    }
    char * device_rows = nullptr;
    half2 * device_output = nullptr;
    const size_t output_bytes = (count+1)*125*8*sizeof(half2);
    check(cudaMalloc(&device_rows,rows.size()));
    check(cudaMalloc(&device_output,output_bytes));
    check(cudaMemcpy(device_rows,rows.data(),rows.size(),cudaMemcpyHostToDevice));
    compare_chunks<type><<<((count+1)*125+127)/128,128>>>(device_rows,int(row_bytes),count,device_output);
    check(cudaGetLastError());
    std::vector<uint32_t> output(output_bytes/4);
    check(cudaMemcpy(output.data(),device_output,output_bytes,cudaMemcpyDeviceToHost));
    check(cudaFree(device_output));
    check(cudaFree(device_rows));
    for (size_t chunk = 0; chunk < (count+1)*125; ++chunk) for (int pair = 0; pair < 4; ++pair)
        if (output[chunk*8+pair] != output[chunk*8+4+pair])
            throw std::runtime_error("eight-value conversion differs from stock pair bits");
    std::printf("%s: %d chunks bit-exact\n",ggml_type_name(type),(count+1)*125);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        qualify<GGML_TYPE_Q8_0>(); qualify<GGML_TYPE_Q4_0>();
        qualify<GGML_TYPE_Q4_1>(); qualify<GGML_TYPE_Q5_0>(); qualify<GGML_TYPE_Q5_1>();
        qualify<GGML_TYPE_F16>(); qualify<GGML_TYPE_BF16>(); qualify<GGML_TYPE_F32>();
    } catch (const std::exception & error) {
        std::fprintf(stderr,"%s\n",error.what()); return 1;
    }
    return 0;
}

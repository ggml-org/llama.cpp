#include "../ggml/src/ggml-cuda/fattn-tile.cuh"
#include "testing.h"

#include <cstring>
#include <type_traits>

struct counted_rows {
    const half2 * data;
    int stride;
    unsigned int * visits;

    // A custom reader proves the shared loader uses the seam, including zero-fill without invalid reads.
    template<int bytes>
    __device__ __forceinline__ void load(half2 * destination, int row, int pair, bool valid, const half2 * zero) const {
        if (valid) atomicAdd(visits+row,1u);
        const ggml_cuda_fattn_tile_contiguous_f16_rows contiguous{data,stride};
        contiguous.template load<bytes>(destination,row,pair,valid,zero);
    }
};

template<typename T, int width>
static __global__ void load_probe(const half2 * source, int stride, T * destination, unsigned int * visits, int valid) {
    const counted_rows rows{source,stride,visits};
    flash_attn_tile_load_tile<32,4,7,width,4,true>(rows,destination,valid);
}

// Check stock's zero-array spelling independently of reader dispatch.
static __global__ void stock_zero_probe(half2 * output) {
    const __align__(16) half2 zero[4] = {{0.0f,0.0f}};
    ggml_cuda_memcpy_1<16>(output,zero);
}

// Guard the whole allocation; shared-tile padding must stay untouched while absent rows become zero.
template<typename T, int width>
static void check(testing & t, int live) {
    constexpr bool packed = std::is_same<T,half2>::value;
    constexpr size_t rows = 7, padding = 4, guard = 16;
    constexpr size_t row_elements = (packed ? width/2 : width)+padding;
    constexpr size_t bytes = rows*row_elements*sizeof(T);
    const int stride = width/2+8;
    std::vector<ggml_fp16_t> source(size_t(live)*size_t(stride)*2);
    const ggml_fp16_t patterns[] = {0x0000,0x8000,0x3c00,0xbc00,0x7bff,0xfbff,0x0001,0x03ff,0x0400,0x3555};
    for (size_t i = 0; i < source.size(); ++i) source[i] = patterns[i%10];
    half2 * device_source = nullptr;
    uint8_t * storage = nullptr;
    unsigned int * visits = nullptr;
    CUDA_CHECK(cudaMalloc(&storage,bytes+2*guard));
    CUDA_CHECK(cudaMalloc(&visits,rows*sizeof(unsigned int)));
    if (!source.empty()) {
        CUDA_CHECK(cudaMalloc(&device_source,source.size()*sizeof(ggml_fp16_t)));
        CUDA_CHECK(cudaMemcpy(device_source,source.data(),source.size()*sizeof(ggml_fp16_t),cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(storage,0x5a,bytes+2*guard));
    CUDA_CHECK(cudaMemset(visits,0,rows*sizeof(unsigned int)));
    load_probe<T,width><<<1,dim3(32,4)>>>(device_source,stride,reinterpret_cast<T *>(storage+guard),visits,live);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint8_t> actual(bytes+2*guard), expected(bytes+2*guard,0x5a);
    std::vector<unsigned int> counts(rows);
    CUDA_CHECK(cudaMemcpy(actual.data(),storage,actual.size(),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts.data(),visits,rows*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    int physical = 0; cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&physical)); CUDA_CHECK(cudaGetDeviceProperties(&properties,physical));
    const int compiled = ggml_cuda_highest_compiled_arch(properties.major*100+properties.minor*10);
    const unsigned int copy_bytes = compiled < 700 ? 8 : 16;
    for (size_t row = 0; row < rows; ++row) {
        t.assert_equal(row < size_t(live) ? unsigned(width*(packed ? 2 : 4))/copy_bytes : 0u,counts[row]);
        for (size_t col = 0; col < width; ++col) {
            const auto encoded = row < size_t(live) ? source[row*size_t(stride)*2+col] : ggml_fp16_t(0);
            auto * output = expected.data()+guard+row*row_elements*sizeof(T)+col*(packed ? 2 : 4);
            if (packed) std::memcpy(output,&encoded,sizeof(encoded));
            else { const float value = ggml_fp16_to_fp32(encoded); std::memcpy(output,&value,sizeof(value)); }
        }
    }
    if (!t.assert_true(std::string(packed ? "half" : "float")+" width="+std::to_string(width)+" live="+std::to_string(live),actual == expected)) {
        for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) {
            t.out << "first mismatched byte=" << i << " expected=" << unsigned(expected[i]) << " actual=" << unsigned(actual[i]) << '\n';
            break;
        }
    }
    if (device_source) CUDA_CHECK(cudaFree(device_source));
    CUDA_CHECK(cudaFree(visits)); CUDA_CHECK(cudaFree(storage));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    testing t;
    t.test("stock_zero_array_initializes_every_copy_lane", [&](testing & t) {
        half2 * output = nullptr;
        CUDA_CHECK(cudaMalloc(&output,16));
        CUDA_CHECK(cudaMemset(output,0x5a,16));
        stock_zero_probe<<<1,1>>>(output);
        CUDA_CHECK(cudaGetLastError());
        uint8_t bytes[16];
        CUDA_CHECK(cudaMemcpy(bytes,output,16,cudaMemcpyDeviceToHost));
        t.assert_true(std::all_of(bytes,bytes+16,[](uint8_t x) { return x == 0; }));
        CUDA_CHECK(cudaFree(output));
    });
    t.test("custom_tile_reader_preserves_bits_rounding_zero_fill_and_padding", [&](testing & t) {
        for (int live : {0,1,5,7}) {
            check<half2,40>(t,live); check<float,40>(t,live);
            check<half2,64>(t,live); check<float,64>(t,live);
            check<half2,256>(t,live); check<float,256>(t,live);
        }
    });
    return t.summary();
}

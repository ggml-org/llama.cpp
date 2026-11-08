#include "mmq-sm90.cuh"

#ifdef GGML_CUDA_MMQ_SM90

#include <cstddef>
#include <cuda/ptx>

namespace {

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 256;
static constexpr int Q6_K_BLOCK_BYTES = sizeof(block_q6_K);
static constexpr int WEIGHT_TILE_BYTES = TILE_M * 64 * sizeof(half);
static constexpr int INPUT_TILE_BYTES = TILE_N * 64 * sizeof(half);
static constexpr int INPUT_OFFSET = 2 * WEIGHT_TILE_BYTES;
static constexpr int PACKED_OFFSET = INPUT_OFFSET + 4 * INPUT_TILE_BYTES;
static constexpr int PACKED_TILE_BYTES = 4 * TILE_M * 64;
static constexpr int SHARED_BYTES = PACKED_OFFSET + 2 * PACKED_TILE_BYTES;

static_assert(QK_K == 256 && Q6_K_BLOCK_BYTES == 210, "Q6_K layout changed");

__host__ __device__ constexpr int sw128_offset(int row, int column) {
    return (row * 64 + column) ^ ((row & 7) * 8);
}

__device__ __forceinline__ uint64_t wgmma_descriptor(char * tile) {
    unsigned address = unsigned(__cvta_generic_to_shared(tile));
    return uint64_t(address >> 4) | (uint64_t(1) << 16) | (uint64_t(64) << 32) | (uint64_t(1) << 62);
}

__host__ __device__ constexpr int wgmma_output_row(int thread, int index) {
    return (thread / 32) * 16 + (thread % 32) / 4 + ((index / 2) % 2) * 8;
}

__host__ __device__ constexpr int wgmma_output_column(int thread, int index) {
    return (thread % 4) * 2 + index % 2 + (index / 4) * 8;
}

template<int value>
struct quarter_tag {
    static constexpr int value_ = value;
};

__device__ __forceinline__ void wgmma_operand_fence(float (&accumulator)[64]) {
    // keep compiler from moving accumulator accesses across async WGMMA
    #pragma unroll
    for (int index = 0; index < 64; ++index) asm volatile("" : "+f"(accumulator[index]) :: "memory");
}

__device__ __forceinline__ void wgmma_mma(uint64_t a, uint64_t b, float (&accumulator)[64]) {
    asm volatile(
        "{ .reg .pred accumulate; setp.eq.u32 accumulate, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, accumulate, 1, 1, 0, 0; }"
        : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3]),
          "+f"(accumulator[4]), "+f"(accumulator[5]), "+f"(accumulator[6]), "+f"(accumulator[7]),
          "+f"(accumulator[8]), "+f"(accumulator[9]), "+f"(accumulator[10]), "+f"(accumulator[11]),
          "+f"(accumulator[12]), "+f"(accumulator[13]), "+f"(accumulator[14]), "+f"(accumulator[15]),
          "+f"(accumulator[16]), "+f"(accumulator[17]), "+f"(accumulator[18]), "+f"(accumulator[19]),
          "+f"(accumulator[20]), "+f"(accumulator[21]), "+f"(accumulator[22]), "+f"(accumulator[23]),
          "+f"(accumulator[24]), "+f"(accumulator[25]), "+f"(accumulator[26]), "+f"(accumulator[27]),
          "+f"(accumulator[28]), "+f"(accumulator[29]), "+f"(accumulator[30]), "+f"(accumulator[31]),
          "+f"(accumulator[32]), "+f"(accumulator[33]), "+f"(accumulator[34]), "+f"(accumulator[35]),
          "+f"(accumulator[36]), "+f"(accumulator[37]), "+f"(accumulator[38]), "+f"(accumulator[39]),
          "+f"(accumulator[40]), "+f"(accumulator[41]), "+f"(accumulator[42]), "+f"(accumulator[43]),
          "+f"(accumulator[44]), "+f"(accumulator[45]), "+f"(accumulator[46]), "+f"(accumulator[47]),
          "+f"(accumulator[48]), "+f"(accumulator[49]), "+f"(accumulator[50]), "+f"(accumulator[51]),
          "+f"(accumulator[52]), "+f"(accumulator[53]), "+f"(accumulator[54]), "+f"(accumulator[55]),
          "+f"(accumulator[56]), "+f"(accumulator[57]), "+f"(accumulator[58]), "+f"(accumulator[59]),
          "+f"(accumulator[60]), "+f"(accumulator[61]), "+f"(accumulator[62]), "+f"(accumulator[63])
        : "l"(a), "l"(b));
}

__device__ __forceinline__ void tma_initialize(uint64_t * barrier) {
    cuda::ptx::mbarrier_init(barrier, 1);
}

__device__ __forceinline__ void tma_wait(uint64_t * barrier, unsigned phase) {
    while (!cuda::ptx::mbarrier_try_wait_parity(barrier, phase)) {}
}

__global__ void convert_q6_k_sm90_activations(const float * input, __half * output, size_t count) {
    size_t vector = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector < count / 4) {
        float4 values = reinterpret_cast<const float4 *>(input)[vector];
        __half2_raw first = __floats2half2_rn(values.x, values.y);
        __half2_raw second = __floats2half2_rn(values.z, values.w);
        reinterpret_cast<uint2 *>(output)[vector] = make_uint2(
            uint32_t(first.x) | (uint32_t(first.y) << 16), uint32_t(second.x) | (uint32_t(second.y) << 16));
    }
}

template<int quarter>
__device__ __forceinline__ void issue_input_columns256(
        const CUtensorMap & map,
        char * storage,
        uint64_t * barriers,
        int block) {
    if (threadIdx.x == 0) {
        char * destination = storage + INPUT_OFFSET + quarter * INPUT_TILE_BYTES;
        int reduction = block * 256 + quarter * 64;
        int row = blockIdx.y * TILE_N;
        cuda::ptx::mbarrier_arrive_expect_tx(
            cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, barriers + quarter, 32768);
        const int32_t coordinates[] = {reduction, row};
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_cluster, cuda::ptx::space_global, destination, &map, coordinates, barriers + quarter);
    }
}

struct packed_quad_cache {
    uint32_t low[2][2];
    uint32_t high[2][2];
    float d[2];
};

__device__ __forceinline__ uint2 packed_half_four(
        uint32_t low,
        uint32_t high,
        int low_shift,
        int high_shift,
        float scale) {
    unsigned codes = ((low >> low_shift) & 0x0f0f0f0fu) | (((high >> high_shift) & 0x03030303u) << 4);
    float first = __uint_as_float(0x4b000000u | (codes & 255)) - 8388640.0f;
    float second = __uint_as_float(0x4b000000u | ((codes >> 8) & 255)) - 8388640.0f;
    float third = __uint_as_float(0x4b000000u | ((codes >> 16) & 255)) - 8388640.0f;
    float fourth = __uint_as_float(0x4b000000u | (codes >> 24)) - 8388640.0f;
    __half2_raw a = __floats2half2_rn(scale * first, scale * second);
    __half2_raw b = __floats2half2_rn(scale * third, scale * fourth);
    return make_uint2(uint32_t(a.x) | (uint32_t(a.y) << 16), uint32_t(b.x) | (uint32_t(b.y) << 16));
}

__device__ __forceinline__ unsigned packed_address_sw64(char * storage, int row, int offset) {
    unsigned base = unsigned(__cvta_generic_to_shared(storage));
    return base + (offset / 64) * 4096 + row * 64 + ((offset % 64) ^ (((row >> 1) & 3) * 16));
}

__device__ __forceinline__ uint8_t packed_byte_sw64(char * storage, int row, int offset) {
    unsigned value;
    unsigned address = packed_address_sw64(storage, row, offset);
    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return uint8_t(value);
}

__device__ __forceinline__ uint16_t packed_word_sw64(char * storage, int row, int offset) {
    uint16_t value;
    unsigned address = packed_address_sw64(storage, row, offset);
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(value) : "r"(address) : "memory");
    return value;
}

__device__ __forceinline__ void packed_issue_sw64(
        const CUtensorMap & map,
        char * destination,
        uint64_t * barrier,
        int block,
        int row) {
    int first_byte = (block * Q6_K_BLOCK_BYTES) & ~15;
    if (threadIdx.x == 0) {
        cuda::ptx::mbarrier_arrive_expect_tx(
            cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, barrier, 16384);
        for (int plane = 0; plane < 4; ++plane) {
            int coordinate = first_byte + plane * 64;
            const int32_t coordinates[] = {coordinate, row};
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_cluster, cuda::ptx::space_global,
                destination + plane * 4096, &map, coordinates, barrier);
        }
    }
}

__device__ __forceinline__ uint32_t packed_u32_sw64(char * storage, int row, int offset) {
    unsigned address = packed_address_sw64(storage, row, offset);
    uint32_t value;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return value;
}

template<bool two_byte_offset>
__device__ __forceinline__ uint2 packed_octet_sw64(char * storage, int row, int offset) {
    if constexpr (two_byte_offset) {
        uint32_t first = packed_u32_sw64(storage, row, offset - 2);
        uint32_t second = packed_u32_sw64(storage, row, offset + 2);
        uint32_t third = packed_u32_sw64(storage, row, offset + 6);
        return make_uint2((first >> 16) | (second << 16), (second >> 16) | (third << 16));
    } else {
        return make_uint2(packed_u32_sw64(storage, row, offset), packed_u32_sw64(storage, row, offset + 4));
    }
}

template<int quarter, bool two_byte_offset>
__device__ __forceinline__ void unpack_quad_sw64(
        char * packed,
        __half * shared_a,
        packed_quad_cache & cache,
        int block,
        int row_start,
        int m) {
    int column = (threadIdx.x % 8) * 8;
    int lane = column % 32;
    int group = column / 32;
    constexpr int half = quarter / 2;
    constexpr int low_shift = (quarter % 2) * 4;
    int high_shift = low_shift + group * 2;
    int offset = (block * Q6_K_BLOCK_BYTES) & 15;
    uint8_t scales[2];
    #pragma unroll 2
    for (int index = 0; index < 2; ++index) {
        int row = threadIdx.x / 8 + index * 32;
        if constexpr (quarter == 0) cache.d[index] = __half2float(__ushort_as_half(
            packed_word_sw64(packed, row, offset + int(offsetof(block_q6_K, d)))));
        int scale_index = offset + int(offsetof(block_q6_K, scales)) + half * 8 +
                          (quarter % 2) * 4 + group * 2 + lane / 16;
        scales[index] = packed_byte_sw64(packed, row, scale_index);
        if constexpr (quarter % 2 == 0) {
            uint2 low = packed_octet_sw64<two_byte_offset>(packed, row, offset + half * 64 + group * 32 + lane);
            uint2 high = packed_octet_sw64<two_byte_offset>(
                packed, row, offset + int(offsetof(block_q6_K, qh)) + half * 32 + lane);
            cache.low[index][0] = low.x;
            cache.low[index][1] = low.y;
            cache.high[index][0] = high.x;
            cache.high[index][1] = high.y;
        }
    }
    #pragma unroll 2
    for (int index = 0; index < 2; ++index) {
        int row = threadIdx.x / 8 + index * 32;
        float d = cache.d[index];
        float scale = d * int(static_cast<int8_t>(scales[index]));
        unsigned mask = -unsigned(row_start + row < m);
        uint32_t converted[4];
        #pragma unroll 2
        for (int word = 0; word < 2; ++word) {
            uint2 halves = packed_half_four(
                cache.low[index][word], cache.high[index][word], low_shift, high_shift, scale);
            converted[word * 2] = halves.x & mask;
            converted[word * 2 + 1] = halves.y & mask;
        }
        uint4 values = make_uint4(converted[0], converted[1], converted[2], converted[3]);
        *reinterpret_cast<uint4 *>(shared_a + sw128_offset(row, column)) = values;
    }
}

template<int quarter, bool two_byte_offset>
__device__ __forceinline__ void prepare_columns256_sw64(
        const CUtensorMap & map,
        char * storage,
        uint64_t * barriers,
        packed_quad_cache & cache,
        int block,
        int m,
        int k) {
    int packed_bank = block % 2;
    char * raw = storage + PACKED_OFFSET + packed_bank * PACKED_TILE_BYTES;
    if constexpr (quarter == 0) {
        if (block == 0) packed_issue_sw64(map, raw, barriers + 4 + packed_bank, block, blockIdx.x * TILE_M);
        tma_wait(barriers + 4 + packed_bank, (block / 2) % 2);
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        __syncthreads();
        if (block + 1 < k / 256) packed_issue_sw64(
            map, storage + PACKED_OFFSET + (1 - packed_bank) * PACKED_TILE_BYTES,
            barriers + 4 + (1 - packed_bank), block + 1, blockIdx.x * TILE_M);
    }
    auto shared_a = reinterpret_cast<__half *>(storage + (quarter % 2) * WEIGHT_TILE_BYTES);
    unsigned phase = block % 2;
    bool ready = cuda::ptx::mbarrier_try_wait_parity(barriers + quarter, phase);
    unpack_quad_sw64<quarter, two_byte_offset>(raw, shared_a, cache, block, blockIdx.x * TILE_M, m);
    while (!ready) {
        ready = cuda::ptx::mbarrier_try_wait_parity(barriers + quarter, phase);
    }
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
}

__global__ void mul_mat_q6_k_sm90(
        const __grid_constant__ CUtensorMap map,
        const __grid_constant__ CUtensorMap input_map,
        float * output,
        int m,
        int n,
        int k) {
    extern __shared__ __align__(1024) char sw64_storage[];
    __shared__ __align__(8) uint64_t barriers[6];
    if (threadIdx.x == 0) {
        for (int bank = 0; bank < 6; ++bank) tma_initialize(barriers + bank);
        cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
    }
    __syncthreads();
    issue_input_columns256<0>(input_map, sw64_storage, barriers, 0);
    issue_input_columns256<1>(input_map, sw64_storage, barriers, 0);
    issue_input_columns256<2>(input_map, sw64_storage, barriers, 0);
    issue_input_columns256<3>(input_map, sw64_storage, barriers, 0);
    int group = threadIdx.x / 128;
    packed_quad_cache cache{};
    prepare_columns256_sw64<0, false>(map, sw64_storage, barriers, cache, 0, m, k);
    __syncthreads();
    float accumulator[64] = {};
    for (int first_block = 0; first_block < k / 256; first_block += 2) {
        auto block_step = [&](auto parity_tag) {
            constexpr bool two_byte_offset = decltype(parity_tag)::value_ != 0;
            int block = first_block + int(two_byte_offset);
            auto step = [&](auto quarter_index) {
                constexpr int quarter = decltype(quarter_index)::value_;
                uint64_t descriptor_a = wgmma_descriptor(sw64_storage + (quarter % 2) * WEIGHT_TILE_BYTES);
                uint64_t descriptor_b = wgmma_descriptor(
                    sw64_storage + INPUT_OFFSET + quarter * INPUT_TILE_BYTES + group * (INPUT_TILE_BYTES / 2));
                wgmma_operand_fence(accumulator);
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int reduction = 0; reduction < 4; ++reduction)
                    wgmma_mma(descriptor_a + reduction * 2, descriptor_b + reduction * 2, accumulator);
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                if constexpr (quarter < 3)
                    prepare_columns256_sw64<quarter + 1, two_byte_offset>(
                        map, sw64_storage, barriers, cache, block, m, k);
                else if (block + 1 < k / 256)
                    prepare_columns256_sw64<0, !two_byte_offset>(map, sw64_storage, barriers, cache, block + 1, m, k);
                // Finish both warp groups before reusing the current operand tiles.
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                wgmma_operand_fence(accumulator);
                __syncthreads();
                if (block + 1 < k / 256) issue_input_columns256<quarter>(input_map, sw64_storage, barriers, block + 1);
            };
            step(quarter_tag<0>{}); step(quarter_tag<1>{}); step(quarter_tag<2>{}); step(quarter_tag<3>{});
        };
        block_step(quarter_tag<0>{});
        if (first_block + 1 < k / 256) block_step(quarter_tag<1>{});
    }
    for (int index = 0; index < 64; ++index) {
        int row = blockIdx.x * TILE_M + wgmma_output_row(threadIdx.x % 128, index);
        int column = blockIdx.y * TILE_N + group * 128 + wgmma_output_column(threadIdx.x % 128, index);
        if (row < m && column < n) output[column * m + row] = accumulator[index];
    }
}

} // namespace

bool ggml_cuda_should_use_mmq_q6_k_sm90(
        int device_id,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
#if defined(GGML_CUDA_FORCE_MMQ) || defined(GGML_CUDA_FORCE_CUBLAS)
    GGML_UNUSED(device_id);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#else
    const auto & device = ggml_cuda_info().devices[device_id];
    const bool shape_gate_up = src0->ne[0] == 4096 && src0->ne[1] == 12288;
    const bool shape_down = src0->ne[0] == 12288 && src0->ne[1] == 4096;
    const int64_t n = src1->ne[1];
    if (device.cc != GGML_CUDA_CC_HOPPER || device.smpbo < SHARED_BYTES + 2048 ||
            src0->type != GGML_TYPE_Q6_K || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            (!shape_gate_up && !shape_down) || (n != 256 && n != 512 && n != 1024 && n != 2048) ||
            src1->ne[0] != src0->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != n ||
            !ggml_is_matrix(src0) || !ggml_is_matrix(src1) || !ggml_is_matrix(dst) ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            ggml_get_op_params_i32(dst, 0) != GGML_PREC_UNDEFINED ||
            ggml_get_op_params_i32(dst, 1) != GGML_HINT_NONE ||
            ggml_get_op_params_i32(dst, 3) != GGML_PREC_UNDEFINED ||
            getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") != nullptr || getenv("GGML_CUDA_MMQ_PREC") != nullptr) {
        return false;
    }
    for (const ggml_tensor * tensor : {src0, src1, static_cast<const ggml_tensor *>(dst)}) {
        if (!tensor->buffer ||
                ggml_backend_buffer_get_type(tensor->buffer) != ggml_backend_cuda_buffer_type(device_id)) {
            return false;
        }
    }
    if (uintptr_t(src0->data) % 16 != 0 || uintptr_t(src1->data) % 16 != 0) {
        return false;
    }

    return true;
#endif
}

bool ggml_cuda_mul_mat_q6_k_sm90(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (!ggml_cuda_should_use_mmq_q6_k_sm90(ctx.device, src0, src1, dst)) {
        return false;
    }

    const int64_t n = src1->ne[1];
    const int m = int(src0->ne[1]);
    const int k = int(src0->ne[0]);
    ggml_cuda_pool_alloc<half> workspace(ctx.pool(), size_t(n) * k);
    CUtensorMap weight_map, input_map;
    uint64_t weight_dimensions[] = {uint64_t(k / QK_K) * Q6_K_BLOCK_BYTES, uint64_t(m)};
    uint64_t weight_strides[] = {weight_dimensions[0]};
    uint32_t weight_box[] = {64, TILE_M};
    uint32_t element_strides[] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(
        &weight_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, src0->data,
        weight_dimensions, weight_strides, weight_box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    uint64_t input_dimensions[] = {uint64_t(k), uint64_t(n)};
    uint64_t input_strides[] = {k * sizeof(half)};
    uint32_t input_box[] = {64, TILE_N};
    CU_CHECK(cuTensorMapEncodeTiled(
        &input_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, workspace.get(),
        input_dimensions, input_strides, input_box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q6_k_sm90, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES));
    const cudaStream_t stream = ctx.stream();
    convert_q6_k_sm90_activations<<<(size_t(n) * k / 4 + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(src1->data), workspace.get(), size_t(n) * k);
    CUDA_CHECK(cudaGetLastError());
    mul_mat_q6_k_sm90<<<dim3(m / TILE_M, n / TILE_N), 256, SHARED_BYTES, stream>>>(
        weight_map, input_map, static_cast<float *>(dst->data), m, int(n), k);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif // GGML_CUDA_MMQ_SM90

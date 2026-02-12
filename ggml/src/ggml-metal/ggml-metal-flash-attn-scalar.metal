// Flash Attention SCALAR fallback for GPUs without simdgroup matrix multiply
// Ported from Vulkan flash_attn.comp (SCALAR path)
// Runs on any Metal GPU without requiring simdgroup_mm (AMD, older Apple, etc.)

#define SCALAR_BR 4
#define SCALAR_BC 64
#define SCALAR_THREADS 128
#define SCALAR_D_SPLIT 4

template<int DK, int DV>
inline void flash_attn_ext_scalar_impl(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q    [[buffer(1)]],
    device const char * k    [[buffer(2)]],
    device const char * v    [[buffer(3)]],
    device const char * mask [[buffer(4)]],
    device       char * dst  [[buffer(5)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tid   [[thread_index_in_threadgroup]])
{
    const uint i   = tgpig.x;
    const uint iq2 = tgpig.y;
    const uint iq3 = tgpig.z;

    constexpr int BR = SCALAR_BR;
    constexpr int BC = SCALAR_BC;
    constexpr int D_SPLIT = SCALAR_D_SPLIT;
    constexpr int DK4 = DK / 4;
    constexpr int DV4 = DV / 4;
    constexpr int HSK_per_thread = DK / D_SPLIT;
    constexpr int HSV_per_thread = DV / D_SPLIT;
    constexpr int cols_per_iter = SCALAR_THREADS / D_SPLIT;
    constexpr int cols_per_thread = BC / cols_per_iter;

    const uint N  = args.ne01;
    const uint KV = args.ne11;
    const float scale = args.scale;

    const ulong q_byte_offset   = iq2 * args.nb02 + iq3 * args.nb03;
    const ulong k_byte_offset   = iq2 * args.nb12 + iq3 * args.nb13;
    const ulong v_byte_offset   = iq2 * args.nb22 + iq3 * args.nb23;
    const ulong dst_byte_offset = iq2 * args.nb02 + iq3 * args.nb03;

    device const float4 * q_ptr   = (device const float4 *)((device const char *)q   + q_byte_offset);
    device const half4  * k_ptr   = (device const half4  *)((device const char *)k   + k_byte_offset);
    device const half4  * v_ptr   = (device const half4  *)((device const char *)v   + v_byte_offset);
    device       float4 * dst_ptr = (device       float4 *)((device       char *)dst + dst_byte_offset);

    const uint d_tid   = tid % D_SPLIT;
    const uint col_tid = tid / D_SPLIT;

    threadgroup float4 * Qf      = (threadgroup float4 *) shmem;
    threadgroup float  * tmpsh    = (threadgroup float  *) (Qf + BR * DK4);
    threadgroup float4 * tmpshv4  = (threadgroup float4 *) (Qf + BR * DK4);

    // Load Q tile to shared memory
    for (uint idx = tid; idx < BR * DK4; idx += SCALAR_THREADS) {
        uint r = idx / DK4;
        uint d = idx % DK4;
        if (i * BR + r < N && d < DK4) {
            Qf[r * DK4 + d] = q_ptr[(i * BR + r) * DK4 + d] * scale;
        } else {
            Qf[r * DK4 + d] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 Of[BR][HSV_per_thread / 4];
    float Lf[BR];
    float Mf[BR];

    constexpr float NEG_FLT_MAX_HALF = -3.4028234e+38f / 2.0f;

    for (uint r = 0; r < BR; ++r) {
        for (uint d = 0; d < HSV_per_thread / 4; ++d) {
            Of[r][d] = float4(0.0f);
        }
        Lf[r] = 0.0f;
        Mf[r] = NEG_FLT_MAX_HALF;
    }

    const uint num_tiles = (KV + BC - 1) / BC;

    for (uint j = 0; j < num_tiles; ++j) {
        float Sf[BR][cols_per_thread];
        for (uint r = 0; r < BR; ++r) {
            for (uint c = 0; c < cols_per_thread; ++c) {
                Sf[r][c] = 0.0f;
            }
        }

        // S = Q @ K^T
        for (uint c = 0; c < cols_per_thread; ++c) {
            uint kv_idx = j * BC + c * cols_per_iter + col_tid;
            if (kv_idx >= KV) continue;
            for (uint d = 0; d < HSK_per_thread / 4; ++d) {
                uint k_idx = kv_idx * DK4 + d * D_SPLIT + d_tid;
                float4 K_Tf = float4(k_ptr[k_idx]);
                for (uint r = 0; r < BR; ++r) {
                    Sf[r][c] += dot(Qf[r * DK4 + d * D_SPLIT + d_tid], K_Tf);
                }
            }
        }

        // Reduce scores across D_SPLIT threads
        for (uint c = 0; c < cols_per_thread; ++c) {
            for (uint r = 0; r < BR; ++r) {
                tmpsh[tid] = Sf[r][c];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = D_SPLIT / 2; s > 0; s >>= 1) {
                    if (d_tid < s) {
                        tmpsh[col_tid * D_SPLIT + d_tid] += tmpsh[col_tid * D_SPLIT + d_tid + s];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                Sf[r][c] = tmpsh[col_tid * D_SPLIT];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Online softmax
        float Pf[BR][cols_per_thread], eMf[BR];
        for (uint r = 0; r < BR; ++r) {
            float rowmaxf = NEG_FLT_MAX_HALF;
            for (uint c = 0; c < cols_per_thread; ++c) {
                uint kv_idx = j * BC + c * cols_per_iter + col_tid;
                if (kv_idx < KV) {
                    rowmaxf = max(rowmaxf, Sf[r][c]);
                }
            }
            float Moldf = Mf[r];
            Mf[r] = max(rowmaxf, Moldf);
            for (uint c = 0; c < cols_per_thread; ++c) {
                Pf[r][c] = exp(Sf[r][c] - Mf[r]);
            }
            eMf[r] = exp(Moldf - Mf[r]);
            float rowsumf = 0.0f;
            for (uint c = 0; c < cols_per_thread; ++c) {
                uint kv_idx = j * BC + c * cols_per_iter + col_tid;
                if (kv_idx < KV) {
                    rowsumf += Pf[r][c];
                }
            }
            Lf[r] = eMf[r] * Lf[r] + rowsumf;
        }

        // Scale previous output
        for (uint d = 0; d < HSV_per_thread / 4; ++d) {
            for (uint r = 0; r < BR; ++r) {
                Of[r][d] = eMf[r] * Of[r][d];
            }
        }

        // O += P @ V
        for (uint c = 0; c < cols_per_thread; ++c) {
            uint kv_idx = j * BC + c * cols_per_iter + col_tid;
            if (kv_idx >= KV) continue;
            for (uint d = 0; d < HSV_per_thread / 4; ++d) {
                uint v_idx = kv_idx * DV4 + d * D_SPLIT + d_tid;
                float4 Vf = float4(v_ptr[v_idx]);
                for (uint r = 0; r < BR; ++r) {
                    Of[r][d] += Pf[r][c] * Vf;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final reduction across col_tid groups within each d_tid lane
    for (uint r = 0; r < BR; ++r) {
        tmpsh[tid] = Mf[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = cols_per_iter / 2; s > 0; s >>= 1) {
            if (col_tid < s) {
                uint idx = col_tid * D_SPLIT + d_tid;
                tmpsh[idx] = max(tmpsh[idx], tmpsh[(col_tid + s) * D_SPLIT + d_tid]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float rowmaxf = tmpsh[d_tid];
        float Moldf = Mf[r];
        Mf[r] = max(rowmaxf, Moldf);
        float eMf = exp(Moldf - Mf[r]);
        Lf[r] = eMf * Lf[r];

        tmpsh[tid] = Lf[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = cols_per_iter / 2; s > 0; s >>= 1) {
            if (col_tid < s) {
                uint idx = col_tid * D_SPLIT + d_tid;
                tmpsh[idx] = tmpsh[idx] + tmpsh[(col_tid + s) * D_SPLIT + d_tid];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        Lf[r] = tmpsh[d_tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = 0; d < HSV_per_thread / 4; ++d) {
            Of[r][d] = eMf * Of[r][d];
            tmpshv4[tid] = Of[r][d];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = cols_per_iter / 2; s > 0; s >>= 1) {
                if (col_tid < s) {
                    uint idx = col_tid * D_SPLIT + d_tid;
                    Of[r][d] += tmpshv4[(col_tid + s) * D_SPLIT + d_tid];
                    tmpshv4[idx] = Of[r][d];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            Of[r][d] = tmpshv4[d_tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write output: O / L
    for (uint r = 0; r < BR; ++r) {
        if (i * BR + r < N && col_tid == 0) {
            const float inv_L = 1.0f / Lf[r];
            for (uint d = 0; d < HSV_per_thread / 4; ++d) {
                uint dst_idx = (i * BR + r) * DV4 + d * D_SPLIT + d_tid;
                dst_ptr[dst_idx] = Of[r][d] * inv_L;
            }
        }
    }
}

// Kernel wrappers for common head sizes
[[host_name("kernel_flash_attn_ext_scalar_dk64_dv64")]]
kernel void kernel_flash_attn_ext_scalar_dk64_dv64(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q    [[buffer(1)]],
    device const char * k    [[buffer(2)]],
    device const char * v    [[buffer(3)]],
    device const char * mask [[buffer(4)]],
    device       char * dst  [[buffer(5)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tid   [[thread_index_in_threadgroup]])
{
    flash_attn_ext_scalar_impl<64, 64>(args, q, k, v, mask, dst, shmem, tgpig, tid);
}

[[host_name("kernel_flash_attn_ext_scalar_dk128_dv128")]]
kernel void kernel_flash_attn_ext_scalar_dk128_dv128(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q    [[buffer(1)]],
    device const char * k    [[buffer(2)]],
    device const char * v    [[buffer(3)]],
    device const char * mask [[buffer(4)]],
    device       char * dst  [[buffer(5)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tid   [[thread_index_in_threadgroup]])
{
    flash_attn_ext_scalar_impl<128, 128>(args, q, k, v, mask, dst, shmem, tgpig, tid);
}

[[host_name("kernel_flash_attn_ext_scalar_dk256_dv256")]]
kernel void kernel_flash_attn_ext_scalar_dk256_dv256(
    constant ggml_metal_kargs_flash_attn_ext & args [[buffer(0)]],
    device const char * q    [[buffer(1)]],
    device const char * k    [[buffer(2)]],
    device const char * v    [[buffer(3)]],
    device const char * mask [[buffer(4)]],
    device       char * dst  [[buffer(5)]],
    threadgroup  float * shmem [[threadgroup(0)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tid   [[thread_index_in_threadgroup]])
{
    flash_attn_ext_scalar_impl<256, 256>(args, q, k, v, mask, dst, shmem, tgpig, tid);
}

import common;
import types;
import flash_attn_loader;

[vk::constant_id( 0)] const uint WorkGroupSize = 128;
[vk::constant_id( 1)] const uint Br = 1;
[vk::constant_id( 2)] const uint Bc = 32;
[vk::constant_id( 3)] const uint HSK = 32;
[vk::constant_id( 4)] const uint HSV = 32;
[vk::constant_id( 5)] const uint Clamp = 0;
[vk::constant_id( 6)] const uint D_split = 16;
[vk::constant_id( 7)] const uint row_split = 1;
[vk::constant_id( 8)] const uint SubGroupSize = 32;
[vk::constant_id( 9)] const uint SHMEM_STAGING = 0;
[vk::constant_id(10)] const uint Flags = 0;
[vk::constant_id(11)] const uint LIMIT_OCCUPANCY_SHMEM = 0;

static const bool USE_MASK_OPT    = (Flags & 1) != 0;
static const bool MASK_ENABLE     = (Flags & 2) != 0;
static const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
static const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
static const uint HSK_pad = (HSK + 15) & ~15;
static const uint HSV_pad = (HSV + 15) & ~15;

static const bool KV_bounds_check = Clamp != 0;

struct PushConstants {
    uint N;
    uint KV;

    uint ne1;
    uint ne2;
    uint ne3;

    uint neq2;
    uint neq3;
    uint nek2;
    uint nek3;
    uint nev2;
    uint nev3;
    uint nem1;
    uint nem2;
    uint nem3;

    uint nb01;
    uint nb02;
    uint nb03;
    uint nb11;
    uint nb12;
    uint nb13;
    uint nb21;
    uint nb22;
    uint nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint mask_n_head_log2;
    float m0;
    float m1;

    uint gqa_ratio;
    uint split_kv;
    uint k_num;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> p;

static const uint HSK_per_thread = HSK / D_split;
static const uint HSV_per_thread = HSV / D_split;

static const uint rows_per_thread = Br / row_split;
static const uint cols_per_iter = WorkGroupSize / D_split / row_split;
static const uint cols_per_thread = Bc / cols_per_iter;
static const uint num_subgroups = SubGroupSize == 0 ? 0 : WorkGroupSize / SubGroupSize;

[vk::binding(0)] StructuredBuffer<vector<float, 4>> data_qv4;

[vk::binding(1)] StructuredBuffer<vector<half, 4>> data_kh;
[vk::binding(1)] StructuredBuffer<vector<float, 4>> data_kf;
[vk::binding(1)] StructuredBuffer<block_q8_0_packed16> data_kq8_0;
[vk::binding(1)] StructuredBuffer<block_q4_0_packed16> data_kq4_0;

[vk::binding(2)] StructuredBuffer<vector<half, 4>> data_vh;
[vk::binding(2)] StructuredBuffer<vector<float, 4>> data_vf;
[vk::binding(2)] StructuredBuffer<block_q8_0_packed16> data_vq8_0;
[vk::binding(2)] StructuredBuffer<block_q4_0_packed16> data_vq4_0;

[vk::binding(3)] StructuredBuffer<half> data_m;
[vk::binding(4)] StructuredBuffer<float> data_s;
[vk::binding(5)] RWStructuredBuffer<D_TYPE> data_o;
[vk::binding(5)] RWStructuredBuffer<vector<D_TYPE, 4>> data_ov4;
[vk::binding(6)] StructuredBuffer<uint> data_mask_opt;

// If SubGroupSize is set to 0 then only use shmem reductions
static const uint tmpsh_size = (SubGroupSize > 0) ? (row_split == 1 ? num_subgroups * D_split : num_subgroups) : WorkGroupSize;
groupshared float tmpsh[tmpsh_size];
struct ShMemFloat: ISharedMemory<float, 1> {
    static float get(uint idx) {
        return tmpsh[idx];
    }
    static void set(uint idx, float value) {
        tmpsh[idx] = value;
    }
}
groupshared vector<FLOAT, 4> tmpshv4[tmpsh_size];
struct ShMemFloat4: ISharedMemory<FLOAT, 4> {
    static vector<FLOAT, 4> get(uint idx) {
        return tmpshv4[idx];
    }
    static void set(uint idx, vector<FLOAT, 4> value) {
        tmpshv4[idx] = value;
    }
}

static const uint masksh_stride = Br + 1;
groupshared FLOAT masksh[Bc * masksh_stride];

static const uint qf_stride = HSK / 4 + 1;
groupshared vector<FLOAT, 4> Qf[Br * qf_stride];

static const uint D = HSK > HSV ? HSK : HSV;
static const uint kvsh_stride = D / 4 + 1;
groupshared vector<FLOAT, 4> kvsh[SHMEM_STAGING != 0 ? Bc * kvsh_stride : 1];

groupshared vector<float, 4> occupancy_limiter[LIMIT_OCCUPANCY_SHMEM > 0 ? LIMIT_OCCUPANCY_SHMEM : 1];

struct Indices {
    uint i, N, KV, split_k_index, Tr, start_j, end_j,
             gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
             q_stride, k_stride, v_stride, m_stride;
};

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

uint ceil_div(uint a, uint b) {
    return (a + b - 1) / b;
}

// Store column zero. This is used to save per-row m and L values for split_k.
T perElemOpStoreCol0<T: __BuiltinFloatingPointType>(const uint r, const uint32_t c, const T elem, const uint32_t o_offset, const uint32_t iq2, const uint32_t N)
{
    if (r < N && c == 0) {
        uint offset = iq2 + r;
        data_o[o_offset + offset] = floatCast<D_TYPE>(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
T perElemOpComputeSlope<T: __BuiltinFloatingPointType>(const uint r, const uint32_t c, const T elem, const uint32_t iq2)
{
    const uint h = iq2 + (r % p.gqa_ratio);

    uint n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const T base = T(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return T(pow(base, T(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
T perElemOpGetSink<T: __BuiltinFloatingPointType>(const in uint r, const in uint32_t c, const in T elem, const in uint32_t iq2)
{
    const uint h = iq2 + (r % p.gqa_ratio);

    return T(data_s[h]);
}

Indices init_indices(const vector<uint, 3> wgid)
{
    Indices idcs;
    idcs.N = p.N;
    idcs.KV = p.KV;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            idcs.i = 0;
            // batch and split_k share wgid.x
            idcs.gqa_iq1 = wgid.x / p.k_num;
            idcs.split_k_index = wgid.x % p.k_num;
        } else {
            idcs.gqa_iq1 = 0;
            idcs.split_k_index = wgid.x % p.k_num;
            idcs.i = wgid.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        idcs.i = 0;
        idcs.gqa_iq1 = wgid.x;
        idcs.split_k_index = 0;
    } else {
        idcs.i = wgid.x;
        idcs.gqa_iq1 = 0;
        idcs.split_k_index = 0;
    }

    idcs.Tr = ceil_div(idcs.N, Br);

    idcs.start_j = idcs.split_k_index * p.split_kv / Bc;
    idcs.end_j = ceil_div(min(idcs.KV, (idcs.split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to wgid.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    idcs.iq2 = wgid.y * p.gqa_ratio;
    idcs.iq3 = wgid.z;

    // broadcast factors
    idcs.rk2 = p.neq2/p.nek2;
    idcs.rk3 = p.neq3/p.nek3;

    idcs.rv2 = p.neq2/p.nev2;
    idcs.rv3 = p.neq3/p.nev3;

    // k indices
    idcs.ik3 = idcs.iq3 / idcs.rk3;
    idcs.ik2 = idcs.iq2 / idcs.rk2;

    // v indices
    idcs.iv3 = idcs.iq3 / idcs.rv3;
    idcs.iv2 = idcs.iq2 / idcs.rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    idcs.q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    idcs.k_stride = p.nb11;
    idcs.v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    idcs.m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : idcs.KV;

    return idcs;
}

#if defined(DATA_A_F32)
typealias KLoader = ScalarKVLoader<float>;
typealias VLoader = ScalarKVLoader<float>;
#elif defined(DATA_A_Q8_0)
typealias KLoader = Q8_0KVLoader;
typealias VLoader = Q8_0KVLoader;
#elif defined(DATA_A_Q4_0)
typealias KLoader = Q4_0KVLoader;
typealias VLoader = Q4_0KVLoader;
#else //if defined(DATA_A_F16)
typealias KLoader = ScalarKVLoader<half>;
typealias VLoader = ScalarKVLoader<half>;
#endif

// Store the output when doing grouped query attention.
// Rows index by Q's dimension 2, and the first N rows are valid.
void gqaStore<T: __BuiltinFloatingPointType>(const in uint32_t r, const in uint32_t c, const in vector<T, 4> elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N) {
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = vector<D_TYPE, 4>(elems);
}

[shader("compute")]
[numthreads(WorkGroupSize, 1, 1)]
void main(
    vector<uint, 3> wgid : SV_GroupID,
    uint tid : SV_GroupIndex,
) {
    const Indices idcs = init_indices(wgid);

    const uint subgroup_invocation_id = WaveGetLaneIndex();
    const uint subgroup_id = tid / WaveGetLaneCount();

    const uint threads_per_rowgroup = WorkGroupSize / row_split;
    const uint row_tid = tid / threads_per_rowgroup;
    const uint rowgroup_tid = tid % threads_per_rowgroup;
    const uint d_tid = tid % D_split;
    const uint col_tid = (tid % threads_per_rowgroup) / D_split;

    if (LIMIT_OCCUPANCY_SHMEM > 0) {
        // This just exists to avoid the occupancy_limiter array getting optimized out
        occupancy_limiter[tid] = vector<float, 4>(tid);

        GroupMemoryBarrierWithGroupSync();

        if (all(occupancy_limiter[tid] == vector<float, 4>(99999.0))) {
            data_ov4[0] = vector<D_TYPE, 4>(occupancy_limiter[tid]);
        }
    }

#define tile_row(r) (row_tid * rows_per_thread + (r))

    uint q_offset = idcs.gqa_iq1*p.nb01 + (idcs.iq2*p.nb02 + idcs.iq3*p.nb03) / 4;

    [unroll] for (uint idx = 0; idx < Br * HSK / 4; idx += WorkGroupSize) {
        uint d = (idx + tid) % (HSK / 4);
        uint r = (idx + tid) / (HSK / 4);
        if (r < Br && d < HSK / 4 &&
            idcs.i * Br + r < idcs.N) {
            Qf[r * qf_stride + d] = vector<FLOAT, 4>(data_qv4[q_offset / 4 + (idcs.i * Br + r) * idcs.q_stride / 4 + d] * p.scale);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    vector<FLOAT, 4> Of[rows_per_thread][HSV_per_thread / 4];
    [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
        [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
            Of[r][d] = vector<FLOAT, 4>(0.0);
        }
    }

    float Lf[rows_per_thread], Mf[rows_per_thread];

    // Use -FLT_MAX/2 rather than -inf to reduce the possibility of NaNs, e.g. when computing Mold-M.
    const float NEG_FLT_MAX_OVER_2 = asfloat(0xFEFFFFFF);

    [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
        Lf[r] = 0;
        Mf[r] = NEG_FLT_MAX_OVER_2;
    }

    ACC_TYPE slope[rows_per_thread];
    [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
        slope[r] = ACC_TYPE(1.0);
    }

    // ALiBi
    if (p.max_bias > 0.0f) {
        [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
            slope[r] = perElemOpComputeSlope(tile_row(r), col_tid, ACC_TYPE(0), idcs.iq2);
        }
    }

    const uint mo_stride = ceil_div(idcs.KV, 16 * Bc);
    // mo_offset will point to the tile starting at row i*Br and col 0
    uint mo_offset = mo_stride * idcs.i;

    uint k_offset = idcs.ik2*p.nb12 + idcs.ik3*p.nb13;
    uint v_offset = idcs.iv2*p.nb22 + idcs.iv3*p.nb23;

#if defined(DATA_A_F32)
    KLoader kloader = KLoader(data_kf, k_offset, idcs.k_stride);
    VLoader vloader = VLoader(data_vf, v_offset, idcs.v_stride);
#elif defined(DATA_A_Q4_0)
    KLoader kloader = KLoader(data_kq4_0, k_offset, idcs.k_stride);
    VLoader vloader = VLoader(data_vq4_0, v_offset, idcs.v_stride);
#elif defined(DATA_A_Q8_0)
    KLoader kloader = KLoader(data_kq8_0, k_offset, idcs.k_stride);
    VLoader vloader = VLoader(data_vq8_0, v_offset, idcs.v_stride);
#else //if defined(DATA_A_F16)
    KLoader kloader = KLoader(data_kh, k_offset, idcs.k_stride);
    VLoader vloader = VLoader(data_vh, v_offset, idcs.v_stride);
#endif

    uint m_offset = idcs.gqa_iq1*idcs.KV;
    if (p.nem2 != 1 || p.nem3 != 1) {
        m_offset += ((idcs.iq3 % p.nem3) * p.nem2 + (idcs.iq2 % p.nem2)) * p.nem1 * idcs.KV;
        mo_offset += ((idcs.iq3 % p.nem3) * p.nem2 + (idcs.iq2 % p.nem2)) * ceil_div(p.nem1, Br) * mo_stride;
    }

    uint mask_opt = 0;
    uint mask_opt_idx = ~0;
    uint mask_opt_bits = 0;

    [loop]
    for (uint j = idcs.start_j; j < idcs.end_j; ++j) {
        if (MASK_ENABLE) {
            if (USE_MASK_OPT && mask_opt_idx != j / 16) {
                mask_opt_idx = j / 16;
                mask_opt = data_mask_opt[mo_offset + mask_opt_idx];
            }
            mask_opt_bits = (mask_opt >> ((j % 16) * 2)) & 0x3;
            if (mask_opt_bits == MASK_OPT_ALL_NEG_INF) {
                // skip this block
                continue;
            }
            // Only load if the block is not all zeros
            if (mask_opt_bits != MASK_OPT_ALL_ZERO) {
                bool nem1_bounds_check = !(p.gqa_ratio > 1) && (p.nem1 % Br) != 0;

                float max_mask = NEG_FLT_MAX_OVER_2;
                GroupMemoryBarrierWithGroupSync();
                [unroll] for (uint idx = 0; idx < Bc * Br; idx += WorkGroupSize) {
                    uint c = (idx + tid) % Bc;
                    uint r = (idx + tid) / Bc;
                    if (idx + tid < Bc * Br) {
                        if ((!KV_bounds_check || j * Bc + c < idcs.KV) && (!nem1_bounds_check || idcs.i * Br + r < p.nem1)) {
                            FLOAT m = FLOAT(data_m[m_offset + (idcs.i * Br + r) * idcs.m_stride + (j * Bc + c)]);
                            masksh[c * masksh_stride + r] = m;
                            max_mask = max(max_mask, float(m));
                        } else {
                            masksh[c * masksh_stride + r] = FLOAT(0);
                        }
                    }
                }
                // skip the block if the mask is entirely -inf
                bool all_less = WaveActiveAllTrue(max_mask <= NEG_FLT_MAX_OVER_2);
                GroupMemoryBarrierWithGroupSync();
                if (WaveIsFirstLane()) {
                    tmpsh[subgroup_id] = all_less ? NEG_FLT_MAX_OVER_2 : 0.0f;
                }
                GroupMemoryBarrierWithGroupSync();
                [unroll] for (uint s = 0; s < WaveGetNumWaves(); ++s) {
                    max_mask = max(max_mask, tmpsh[s]);
                }
                if (max_mask <= NEG_FLT_MAX_OVER_2) {
                    continue;
                }
            }
        }

        ACC_TYPE Sf[rows_per_thread][cols_per_thread];
        [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
            [unroll] for (uint c = 0; c < cols_per_thread; ++c) {
                Sf[r][c] = ACC_TYPE(0.0);
            }
        }

        if (SHMEM_STAGING != 0) {
            GroupMemoryBarrierWithGroupSync();
            [unroll] for (uint idx = 0; idx < Bc * HSK / 4; idx += WorkGroupSize) {
                uint d = (idx + tid) % (HSK / 4);
                uint c = (idx + tid) / (HSK / 4);
                if (idx + WorkGroupSize <= Bc * HSK / 4 || c < Bc) {
                    vector<FLOAT, 4> K_Tf = vector<FLOAT, 4>(0);
                    if (!KV_bounds_check || j * Bc + c < idcs.KV) {
                        K_Tf = kloader.load(j * Bc + c, d);
                    }

                    kvsh[c * kvsh_stride + d] = K_Tf;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // More d iterations means Q register caching becomes relevant
        // Few iterations means the additional registers needed are worse than the speed-up from caching
        if (HSK_per_thread / 4 > 4) {
            [unroll] for (uint d = 0; d < HSK_per_thread / 4; ++d) {
                vector<FLOAT, 4> Q_cache[rows_per_thread];
                [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                    Q_cache[r] = Qf[tile_row(r) * qf_stride + d * D_split + d_tid];
                }

                [unroll] for (uint c = 0; c < cols_per_thread; ++c) {
                    if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= idcs.KV) {
                        continue;
                    }

                    vector<FLOAT, 4> K_Tf;
                    if (SHMEM_STAGING != 0) {
                        K_Tf = kvsh[(c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                    } else {
                        K_Tf = kloader.load(j * Bc + c * cols_per_iter + col_tid, d * D_split + d_tid);
                    }
                    [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                        Sf[r][c] += ACC_TYPE(dot(Q_cache[r], K_Tf));
                    }
                }
            }
        } else {
            [unroll] for (uint c = 0; c < cols_per_thread; ++c) {
                if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= idcs.KV) {
                    continue;
                }

                [unroll] for (uint d = 0; d < HSK_per_thread / 4; ++d) {
                    vector<FLOAT, 4> K_Tf;
                    if (SHMEM_STAGING != 0) {
                        K_Tf = kvsh[(c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                    } else {
                        K_Tf = kloader.load(j * Bc + c * cols_per_iter + col_tid, d * D_split + d_tid);
                    }
                    [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                        Sf[r][c] += ACC_TYPE(dot(Qf[tile_row(r) * qf_stride + d * D_split + d_tid], K_Tf));
                    }
                }
            }
        }

        [unroll] for (uint c = 0; c < cols_per_thread; ++c) {
            // Compute sum across the D_split
            [unroll] for (uint s = D_split / 2; s > 0; s >>= 1) {
                [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                    Sf[r][c] += WaveReadLaneAt(Sf[r][c], subgroup_invocation_id ^ s);
                }
            }
        }

        if (LOGIT_SOFTCAP) {
            [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                [unroll] for (uint c = 0; c < cols_per_thread; ++c) {
                    Sf[r][c] = ACC_TYPE(p.logit_softcap * tanh(Sf[r][c]));
                }
            }
        }

        if (MASK_ENABLE && mask_opt_bits != MASK_OPT_ALL_ZERO) {
            [unroll] for (uint c = 0; c < cols_per_thread; ++c) {
                [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                    FLOAT mvf = masksh[(c * cols_per_iter + col_tid) * masksh_stride + tile_row(r)];

                    Sf[r][c] += slope[r]*mvf;
                }
            }
        }

        float eMf[rows_per_thread];
        [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
            float rowmaxf = NEG_FLT_MAX_OVER_2;
            [unroll] for (uint c = 0; c < cols_per_thread; ++c) {
                if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= idcs.KV) {
                    continue;
                }
                rowmaxf = max(rowmaxf, float(Sf[r][c]));
            }
            float Moldf = Mf[r];

            // M = max(rowmax, Mold)
            // P = e^(S - M)
            // eM = e^(Mold - M)
            Mf[r] = max(rowmaxf, Moldf);
            eMf[r] = exp(Moldf - Mf[r]);
            Lf[r] = eMf[r]*Lf[r];
        }

        [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
            [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                Of[r][d] = FLOAT(eMf[r]) * Of[r][d];
            }
        }

        if (SHMEM_STAGING != 0) {
            GroupMemoryBarrierWithGroupSync();
            [unroll] for (uint idx = 0; idx < Bc * HSV / 4; idx += WorkGroupSize) {
                uint d = (idx + tid) % (HSV / 4);
                uint c = (idx + tid) / (HSV / 4);
                if (idx + WorkGroupSize <= Bc * HSV / 4 || c < Bc) {
                    vector<FLOAT, 4> V_Tf = vector<FLOAT, 4>(0);
                    if (!KV_bounds_check || j * Bc + c < idcs.KV) {
                        V_Tf = vloader.load(j * Bc + c, d);
                    }

                    kvsh[c * kvsh_stride + d] = V_Tf;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        [unroll] for (uint c = 0; c < cols_per_thread; ++c) {
            if (KV_bounds_check && j * Bc + c * cols_per_iter + col_tid >= idcs.KV) {
                continue;
            }

            FLOAT Pf[rows_per_thread];
            [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                Pf[r] = FLOAT(exp(float(Sf[r][c]) - Mf[r]));
                Lf[r] += Pf[r];
            }

            [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
                vector<FLOAT, 4> Vf;
                if (SHMEM_STAGING != 0) {
                    Vf = kvsh[(c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                } else {
                    Vf = vloader.load(j * Bc + c * cols_per_iter + col_tid, d * D_split + d_tid);
                }
                [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                    Of[r][d] += vector<FLOAT, 4>(Pf[r] * Vf);
                }
            }
        }
    }

    // prevent race on tmpsh
    GroupMemoryBarrierWithGroupSync();

    // reduce across threads

    [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
        float rowmaxf = Mf[r];

        // Compute max across the row
        rowmaxf = reduce<float, 1, MaxOp<float, 1>, ShMemFloat>(rowmaxf, D_split, threads_per_rowgroup, tid, SubGroupSize);

        float Moldf = Mf[r];

        // M = max(rowmax, Mold)
        // eM = e^(Mold - M)
        Mf[r] = max(rowmaxf, Moldf);
        float eMf = exp(Moldf - Mf[r]);

        Lf[r] = eMf*Lf[r];

        // Compute sum across the row
        Lf[r] = reduce<float, 1, SumOp<float, 1>, ShMemFloat>(Lf[r], D_split, threads_per_rowgroup, tid, SubGroupSize);

        [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
            Of[r][d] = FLOAT(eMf) * Of[r][d];

            Of[r][d] = reduce<FLOAT, 4, SumOp<FLOAT, 4>, ShMemFloat4>(Of[r][d], D_split, threads_per_rowgroup, tid, SubGroupSize, OLD_AMD_WINDOWS);
        }
    }


    // If there is split_k, then the split_k resolve shader does the final
    // division by L. Store the intermediate O value and per-row m and L values.
    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            // note: O and Q have swapped coord 1,2.
            uint o_offset = HSV * p.ne1 * (idcs.split_k_index + p.k_num * (idcs.gqa_iq1 + p.ne2 * idcs.iq3)) / 4;

            [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                const uint row = tile_row(r);
                if (row < idcs.N) {
                    [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
                        gqaStore(row, d * D_split + d_tid, Of[r][d], o_offset, idcs.iq2, idcs.N);
                    }
                }
            }

            o_offset = HSV * p.ne1 * p.k_num * p.ne2 * p.ne3 + p.ne1 * 2 * (idcs.split_k_index + p.k_num * (idcs.gqa_iq1 + p.ne2 * idcs.iq3));
            [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                const uint row = tile_row(r);
                if (row < idcs.N) {
                    perElemOpStoreCol0(row, 0u, ACC_TYPE(Lf[r]), o_offset, idcs.iq2, idcs.N);
                    perElemOpStoreCol0(row, 0u, ACC_TYPE(Mf[r]), o_offset + p.ne1, idcs.iq2, idcs.N);
                }
            }
        } else {
            [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
                const uint row = tile_row(r);
                const uint global_row = idcs.i * Br + row;

                if (global_row < idcs.N) {
                    uint o_offset = HSV * p.ne1 * (idcs.split_k_index + p.k_num * (global_row + p.ne2 * idcs.iq3)) / 4;

                    [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
                        data_ov4[o_offset + idcs.iq2 * HSV/4 + d * D_split + d_tid] = vector<D_TYPE, 4>(Of[r][d]);
                    }
                }

                if (global_row < idcs.N && d_tid == 0 && col_tid == 0) {
                    uint lm_offset = HSV * p.ne1 * p.k_num * p.ne2 * p.ne3 + p.ne1 * 2 * (idcs.split_k_index + p.k_num * (global_row + p.ne2 * idcs.iq3));
                    data_o[lm_offset + idcs.iq2] = D_TYPE(Lf[r]);
                    data_o[lm_offset + p.ne1 + idcs.iq2] = D_TYPE(Mf[r]);
                }
            }
        }
        return;
    }

    if ((p.mask_n_head_log2 & SINK_ENABLE_BIT) != 0) {
        [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
            float sink = perElemOpGetSink(tile_row(r), 0u, ACC_TYPE(0), idcs.iq2);

            float ms = 1.0f;
            float vs = 1.0f;

            if (sink > Mf[r]) {
                ms = exp(Mf[r] - sink);

                [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
                    Of[r][d] *= FLOAT(ms);
                }
            } else {
                vs = exp(sink - Mf[r]);
            }

            Lf[r] = Lf[r]*ms + vs;
        }
    }

    float Lfrcp[rows_per_thread];
    [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
        Lfrcp[r] = (Lf[r] == 0.0) ? 0.0 : (1.0 / Lf[r]);
    }

    [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
        [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
            Of[r][d] *= FLOAT(Lfrcp[r]);
#if defined(FLOAT_TYPE_MAX)
            Of[r][d] = clamp(Of[r][d], -FLOAT_TYPE_MAX, FLOAT_TYPE_MAX);
#endif
        }
    }

    uint o_offset = (idcs.gqa_iq1*p.ne1*HSV + idcs.iq3*p.ne2*p.ne1*HSV) / 4;

    if (p.gqa_ratio > 1) {
        [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
            const uint row = tile_row(r);
            if (row < idcs.N) {
                [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
                    gqaStore(row, d * D_split + d_tid, Of[r][d], o_offset, idcs.iq2, idcs.N);
                }
            }
        }
    } else {
        [unroll] for (uint r = 0; r < rows_per_thread; ++r) {
            const uint row = tile_row(r);
            if (idcs.i * Br + row < idcs.N) {
                [unroll] for (uint d = 0; d < HSV_per_thread / 4; ++d) {
                    data_ov4[o_offset + (idcs.iq2 * HSV + (idcs.i * Br + row) * p.ne1 * HSV) / 4 + d * D_split + d_tid] = vector<D_TYPE, 4>(Of[r][d]);
                }
            }
        }
    }
}

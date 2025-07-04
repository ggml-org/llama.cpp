#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_128
#endif

#define OPWM 64
#define OPWN 64
#define CPWK 8
#define OPTM 4
#define OPTN 8

#define WG_M (OPWM / OPTM)
#define WG_N (OPWN / OPTN)
#define VEC_K (CPWK / 4)

REQD_SUBGROUP_SIZE_128
__kernel void mul_mat_f16_f32(
    const int M, const int N, const int K,
    __global const void* A_void, ulong A_offset,
    __global const void* B_void, ulong B_offset,
    __global       void* C_void, ulong C_offset) {

    __global const half*  A = (__global const half* )((__global const char*)A_void + A_offset);
    __global const float* B = (__global const float*)((__global const char*)B_void + B_offset);
    __global       float* C = (__global       float*)((__global       char*)C_void + C_offset);

    const int lidm = get_local_id(0);
    const int lidn = get_local_id(1);
    const int lid = lidn * WG_M + lidm;

    const int offsetM = get_group_id(0) * OPWM;
    const int offsetN = get_group_id(1) * OPWN;

    __local half4  Alocal[OPWM][VEC_K];
    __local float4 Blocal[OPWN][VEC_K];

    float sum[OPTM][OPTN];

    for (int wm = 0; wm < OPTM; wm++) {
        for (int wn = 0; wn < OPTN; wn++) {
            sum[wm][wn] = 0.0f;
        }
    }

    const int numTiles = (K + CPWK - 1) / CPWK;

    const int load_row_a = lid % OPWM;
    const int load_vec_k_a = lid / OPWM;
    const int global_row_a = offsetM + load_row_a;

    const int load_row_b = lid % OPWN;
    const int load_vec_k_b = lid / OPWN;
    const int global_row_b = offsetN + load_row_b;

    for (int t = 0; t < numTiles; t++) {
        const int k_start = t * CPWK;
        const int k_vec_start_a = k_start + load_vec_k_a * 4;
        const int k_vec_start_b = k_start + load_vec_k_b * 4;

        if (global_row_a < M && k_vec_start_a < K) {
            if (k_vec_start_a + 3 < K) {
                Alocal[load_row_a][load_vec_k_a] = vload4(0, A + global_row_a * K + k_vec_start_a);
            } else {
                half4 tempA = (half4)(0.0h);
                if (k_vec_start_a < K) tempA.s0 = A[global_row_a * K + k_vec_start_a];
                if (k_vec_start_a + 1 < K) tempA.s1 = A[global_row_a * K + k_vec_start_a + 1];
                if (k_vec_start_a + 2 < K) tempA.s2 = A[global_row_a * K + k_vec_start_a + 2];
                Alocal[load_row_a][load_vec_k_a] = tempA;
            }
        } else {
            Alocal[load_row_a][load_vec_k_a] = (half4)(0.0h);
        }

        if (global_row_b < N && k_vec_start_b < K) {
            if (k_vec_start_b + 3 < K) {
                Blocal[load_row_b][load_vec_k_b] = vload4(0, B + global_row_b * K + k_vec_start_b);
            } else {
                float4 tempB = (float4)(0.0f);
                if (k_vec_start_b < K) tempB.s0 = B[global_row_b * K + k_vec_start_b];
                if (k_vec_start_b + 1 < K) tempB.s1 = B[global_row_b * K + k_vec_start_b + 1];
                if (k_vec_start_b + 2 < K) tempB.s2 = B[global_row_b * K + k_vec_start_b + 2];
                Blocal[load_row_b][load_vec_k_b] = tempB;
            }
        } else {
            Blocal[load_row_b][load_vec_k_b] = (float4)(0.0f);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k_vec = 0; k_vec < VEC_K; k_vec++) {
            float4 a_fvecs[OPTM];
            int current_row_a = lidm;
            for (int wm = 0; wm < OPTM; wm++) {
                a_fvecs[wm] = convert_float4(Alocal[current_row_a][k_vec]);
                current_row_a += WG_M;
            }

            float4 b_fvecs[OPTN];
            int current_row_b = lidn;
            for (int wn = 0; wn < OPTN; wn++) {
                b_fvecs[wn] = Blocal[current_row_b][k_vec];
                current_row_b += WG_N;
            }

            for (int wm = 0; wm < OPTM; wm++) {
                for (int wn = 0; wn < OPTN; wn++) {
                    sum[wm][wn] += dot(a_fvecs[wm], b_fvecs[wn]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wm = 0; wm < OPTM; wm++) {
        int globalRow = offsetM + lidm + wm * WG_M;
        if (globalRow < M) {
            for (int wn = 0; wn < OPTN; wn++) {
                int globalCol = offsetN + lidn + wn * WG_N;
                if (globalCol < N) {
                    C[globalCol * M + globalRow] = sum[wm][wn];
                }
            }
        }
    }
}

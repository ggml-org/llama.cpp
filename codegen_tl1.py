import argparse

def gen_body_core_code(bm, by):
    length = 4
    all_code = ""
    for i in range(length):
        core_code = "\n\
            uint8x16_t vec_a_{0} = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + {0} * 16);\n\
            uint8x16_t vec_a{0}_top = vshrq_n_u8(vec_a_{0}, 4);\n\
            uint8x16_t vec_a{0}_bot = vandq_u8(vec_a_{0}, vec_mask);\n\
            int8x16_t  vec_v_{0}_left_tmp0 = vqtbl1q_s8(vec_lut[{1} * k + {2}], vec_a{0}_top);\n\
            int8x16_t  vec_v_{0}_left_tmp1 = vqtbl1q_s8(vec_lut[{1} * k + {3}], vec_a{0}_top);\n\
            int8x16_t  vec_v_{0}_right_tmp0 = vqtbl1q_s8(vec_lut[{1} * k + {4}], vec_a{0}_bot);\n\
            int8x16_t  vec_v_{0}_right_tmp1 = vqtbl1q_s8(vec_lut[{1} * k + {5}], vec_a{0}_bot);\n\
            int8x16x2_t  vec_v_left_{0} = vzipq_s8(vec_v_{0}_left_tmp1, vec_v_{0}_left_tmp0);\n\
            int8x16x2_t  vec_v_right_{0} = vzipq_s8(vec_v_{0}_right_tmp1, vec_v_{0}_right_tmp0);\n\
            vec_c[{6}] += vec_v_left_{0}.val[0];\n\
            vec_c[{6}] += vec_v_right_{0}.val[0];\n\
            vec_c[{7}] += vec_v_left_{0}.val[1];\n\
            vec_c[{7}] += vec_v_right_{0}.val[1];\n\
        ".format(i, 2 * by // 2, (4 * i) % (2 * by // 2), (4 * i + 1) % (2 * by // 2), (4 * i + 2) % (2 * by // 2), (4 * i + 3) % (2 * by // 2), (i * 2) // (by // 2) * 2 + 0, (i * 2) // (by // 2) * 2 + 1)
        
        all_code = "".join([all_code, core_code])

    all_code = "".join([all_code, "\n       }\n\n"])

    for i in range(bm // 8):
        core_code = "\
        int32x4_t vec_v_bot_low_low_{0} = vmovl_s16(vget_low_s16(vec_c[{0}]));\n\
        int32x4_t vec_v_bot_low_high_{0} = vmovl_high_s16(vec_c[{0}]);\n\
        vst1q_s32(c + i + {1}, vld1q_s32(c + i + {1}) + vec_v_bot_low_low_{0});\n\
        vst1q_s32(c + i + {2}, vld1q_s32(c + i + {2}) + vec_v_bot_low_high_{0});\n".format(i, i * 8, i * 8 + 4)
        all_code = "".join([all_code, core_code])

    return all_code

def gen_tbl_impl(pre, BM, BK, bm, by):

    kernel_code = "\
#include <arm_neon.h>\n\
\n\
#define BM{0} {1}\n\
#define BBK{0} {2}\n\
inline void tbl_impl_{0}(int32_t* c, int8_t* lut, uint8_t* a, tmac_float_type* scale, tmac_float_type* lut_scale) {{\n\
#ifdef __ARM_NEON\n\
    const int KK = BBK{0} / 2;\n\
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);\n\
    const int8x16_t vec_zero = vdupq_n_s16(0x0000);\n\
    int8x16_t vec_lut[2 * KK];\n\
".format(pre, BM, BK)
    
    kernel_code = "".join([kernel_code, "    int16x8_t vec_c[{}];".format(bm // 8)])

    kernel_code = "".join([kernel_code, "\n\
#pragma unroll\n\
    for (int k = 0; k < 2 * KK; k++) {\n\
        vec_lut[k] = vld1q_s8(lut + k * 16);\n\
    }\n"])

    pre_core_code = "\n\
#pragma unroll\n\
    for (int i = 0; i < BM{}; i += {}) {{\n\
        #pragma unroll\n\
        for (int i=0; i<{}; i++) {{\n\
            vec_c[i] = vandq_s16(vec_c[i], vec_zero);\n\
        }}\n".format(pre, bm, bm // 8)

    body_core_pre_code = "\n\
#pragma unroll\n\
        for (int k = 0; k < KK / {}; k++) {{\n\
            ".format(by // 2)

    body_core_post_code = "\n\
    }\n\
\
#endif\n\
}\n"


    kernel_code = "".join([kernel_code, pre_core_code, body_core_pre_code, gen_body_core_code(bm, by), body_core_post_code])
    return kernel_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gen impl')
    parser.add_argument('--BMEMD',default="input", type=int)
    parser.add_argument('--BKEMD',default="input", type=int)
    parser.add_argument('--bmEMD',default="input", type=int)
    parser.add_argument('--byEMD',default="input", type=int)
    parser.add_argument('--BMGQA',default="input", type=int)
    parser.add_argument('--BKGQA',default="input", type=int)
    parser.add_argument('--bmGQA',default="input", type=int)
    parser.add_argument('--byGQA',default="input", type=int)
    args = parser.parse_args()

    k1_code = gen_tbl_impl("EMD", args.BMEMD, args.BKEMD, args.bmEMD, args.byEMD)
    k2_code = gen_tbl_impl("GQA", args.BMGQA, args.BKGQA, args.bmGQA, args.byGQA)

    with open("ggml/include/inline_func.h", 'w') as f:     # 写文件，开始的时候会先清空原文件，参考w的用法。如果不用with open，只是open，要注意最后将文件关闭。
        f.write(''.join(k1_code))
        f.write(''.join(k2_code))
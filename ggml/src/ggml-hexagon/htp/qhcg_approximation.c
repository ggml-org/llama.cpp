/**=============================================================================
@file
    qhcg_approximation.c

@brief
    Calculate polynomial approximation of the function below in
    floating-point arithmetic using HVX instructions.

    Function: gelu(x)

    Function is approximated in specified input range from -6.0 to 6.0,
    where inputs and outputs are arrays of 32-bit float values.

    Approximation is performed using the following method:

    1) Input range is split into 16 equidistant segments
    2) For each segment, Numpy's polynomial package is used to find the best
       polynomial approximation of order N with the corresponding C0, C1, ..., Cn.
    3) VLUT instructions are used to select appropriate coefficients for each input sample
    4) Horner's method is used to compute polynomial values:
       f(x) = ((((Cn*x + Cn-1)*x + Cn-2)*x + ...)*x + C1)*x + C0

Copyright (c) 2020 Qualcomm Technologies Incorporated.
All Rights Reserved. Qualcomm Proprietary and Confidential.
=============================================================================**/

#if __HVX_ARCH__ >= 68

#include "qhcg_approximation.h"
#include "qhcg_internal.h"

#define BLOCK_SIZE       (8*1024/128)  /* vector chunks */
#define L2FETCH_AHEAD    (BLOCK_SIZE)

/* Polynomial coefficients */
static const float c0_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
-0.1025868397073178,-1.1184356646199394,-1.9705895994321767,0.11469604839384463,0.40991447569341943,0.00424292239610935,-0.0017846707638177889,4.125901398310816e-09,
9.718309490480692e-11,-0.0015488336803479719,0.001064556481209511,0.3906162486717146,0.19084584900320978,-1.911422745140333,-1.1879384314707315,-0.10823562636002611,
};
static const float c1_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
-0.1234196807250312,-1.5042580469229814,-2.7701977888429816,1.1561921948215528,1.73891533063333,0.49580124294548433,0.4867587290479026,0.500000435462697,
0.4999997919981341,0.5116842338641109,0.5163606020356294,-0.6867154811454343,-0.31551789326265844,3.6694157536939014,2.6042137343731855,1.1304321895807614,
};
static const float c2_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
-0.06367500510012546,-0.8689061926460069,-1.674005553705795,1.5013658408230053,1.9798213609930566,0.33731544915026324,0.35673778512915555,0.398953295788538,
0.3989496120997857,0.3611051680040998,0.31742994078248077,1.9193992198306873,1.6441036493618186,-1.600477678714911,-0.9304878890577859,-0.06740463140212431,
};
static const float c3_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
-0.01826132753437031,-0.27938965958246625,-0.56347555462781,0.8662803078586866,1.0748504969694623,-0.13840364789720844,-0.07490683960610874,0.00011501805987770841,
-8.89610380930177e-05,0.06815977365648013,0.1564140217086786,-1.036053072449464,-0.9372597866516783,0.5336910940777527,0.3004584208315817,0.019362956684359556,
};
static const float c4_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
-0.003143995202268695,-0.05400134785573118,-0.11405541169720136,0.2730279204613817,0.3235877725936627,-0.21757221119731435,-0.14645680741997966,-0.06620698974306806,
-0.06630082288474698,-0.14025595963442758,-0.22733077791076023,0.30866276792496655,0.29418673249390104,-0.10682071320119783,-0.05832443091378418,-0.003339162830362702,
};
static const float c5_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
-0.0003249391771954532,-0.006273424264889337,-0.01387698764918236,0.04913223606829913,0.05545197372332761,-0.09031312961799431,-0.04973154630108704,0.001825035162087615,
-0.0016453620553813022,0.046340833813673266,0.09347637717015225,-0.0520121796723486,-0.0529133082948728,0.012823220702040979,0.00680542921713579,0.00034567793526667706,
};
static const float c6_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
-1.866614032852734e-05,-0.0004055443985444202,-0.0009392734765841164,0.004770459893478637,0.00504881095326808,-0.016904688419800747,-0.0049839929986692675,0.013321931926939602,
0.01314779303860721,-0.003962384692377502,-0.017472688915232914,0.004609031329816739,0.005145502689303376,-0.0008540539868357813,-0.0004419008815610675,-1.989001488248583e-05,
};
static const float c7_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
-4.5974928158662533e-07,-1.1252676844375516e-05,-2.7270249210246306e-05,0.0001949129249064408,0.00018774479051508836,-0.001238293494672944,0.000199977799551612,0.0029325624122584024,
-0.0028654073893250895,-0.00033083897498689484,0.0012818786224478341,-0.00016368340082111905,-0.0002108418119120288,2.431836142521883e-05,1.2317036094266618e-05,4.906925630164402e-07,
};

/**
 * @brief       Polynomial approximation of gelu(x) function.
 * @param[in]   input   Input array of elements in IEEE 32-bit floating-point format.
 * @param[out]  output  Output array of elements in IEEE 32-bit floating-point format.
 * @param[in]   length  Number of elements in input/output arrays.
 * @return      Returns 0 on successful execution. Otherwise -1.
 */
int32_t qhcg_approximation(float *restrict input, float *restrict output, uint32_t size, 
    float limit_left, float limit_right)

{
    HVX_Vector *input_v_ptr;
    HVX_UVector *output_v_ptr;
    HVX_Vector input_min_v_f;
    HVX_Vector input_max_v_f;

    HVX_Vector input_shifted_v_qf32;
    HVX_Vector input_scaled_v_qf32;
    HVX_Vector scale_v;
    HVX_Vector input_v_qf32;
    HVX_Vector const16_0_v_sf;
    HVX_Vector zero_v_sf;
    HVX_Vector mask_idx1_v, mask_idx2_v;
    HVX_Vector tmp_v, idx1_v, idx2_v;
    HVX_Vector output_v;
    HVX_Vector slinep;
    HVX_Vector slinec;
    HVX_Vector sline;
    HVX_Vector sline_tmp;    
    HVX_Vector sout;
    int32_t block, l2fetch_block;
    int32_t leftover = size & 31;
    int32_t vectors_in_rounddown = size / 32;
    int32_t leftover_size = leftover * sizeof(float);
    HVX_DV c0_coeff_dv;
    HVX_VectorPair c0_coeff_vp;
    HVX_Vector c0_coeff_v;
    HVX_DV c1_coeff_dv;
    HVX_VectorPair c1_coeff_vp;
    HVX_Vector c1_coeff_v;
    HVX_DV c2_coeff_dv;
    HVX_VectorPair c2_coeff_vp;
    HVX_Vector c2_coeff_v;
    HVX_DV c3_coeff_dv;
    HVX_VectorPair c3_coeff_vp;
    HVX_Vector c3_coeff_v;
    HVX_DV c4_coeff_dv;
    HVX_VectorPair c4_coeff_vp;
    HVX_Vector c4_coeff_v;
    HVX_DV c5_coeff_dv;
    HVX_VectorPair c5_coeff_vp;
    HVX_Vector c5_coeff_v;
    HVX_DV c6_coeff_dv;
    HVX_VectorPair c6_coeff_vp;
    HVX_Vector c6_coeff_v;
    HVX_DV c7_coeff_dv;
    HVX_VectorPair c7_coeff_vp;
    HVX_Vector c7_coeff_v;

    HVX_Vector zero_vec    = Q6_V_vsplat_R(0x00000000);

    /* Check input arguments. Return error status if some argument has invalid value */
    if ((input == 0) || (output == 0) || (size == 0))
    {
        return -1;
    }

    input_v_ptr = (HVX_Vector *) input;
    output_v_ptr = (HVX_UVector *) output;

    /*
     * If input data is not aligned to HVX vector size, compose aligned vectors
     * from data loaded in slinep and slinec
     */
    slinep = *input_v_ptr++;

    /*
     * Splat scale factor in order to be used later for finding indexes of coefficients.
     * Scale factor is represented in IEEE 16-bit floating-point format and it is
     * calculated using the following formula:
     *    scale_factor = (16.0 / (b0 - a0))
     * NOTE: Calculated value is slightly decreased in order to avoid out of bound
     *       indexes during VLUT lookup.
     */
    scale_v = Q6_V_vsplat_R(0x3faaaaa9);

    /*
     * Vector of zeroes used as neutral element in sf to qf32 conversions.
     * NOTE: Some of conversions (i.e conversion of scale factor and coefficients)
     *       can be avoided in real-time, but this is not done in order to don't
     *       sacrify code readibility in expense of insignificant performance improvement.
     */
    zero_v_sf = Q6_V_vzero();

    /* Mask for extracting only 4 bits of mantissa */
    mask_idx1_v = Q6_V_vsplat_R(0x0000000F);
    mask_idx2_v = Q6_V_vsplat_R(0x00000010);

    /* 16.0 in IEEE 16-bit floating-point representation */
    const16_0_v_sf = Q6_V_vsplat_R(0x41800000);

    /*
     * Prepare vector of input_min values, that is used later in shifting input range.
     * input_min is low boundary of specified input range.
     */
    int32_t input_min_bits = *((int32_t *) &limit_left);
    int32_t input_max_bits = *((int32_t *) &limit_right);
    
    input_min_v_f = Q6_V_vsplat_R(input_min_bits);
    input_max_v_f = Q6_V_vsplat_R(input_max_bits);

    /* Convert scale factor from sf to q32. Use the same vector for both formats */
    scale_v = Q6_Vqf32_vadd_VsfVsf(scale_v, zero_v_sf);

    /* Load coefficients */
    c0_coeff_v = *((HVX_Vector *)(c0_coeffs));
    c1_coeff_v = *((HVX_Vector *)(c1_coeffs));
    c2_coeff_v = *((HVX_Vector *)(c2_coeffs));
    c3_coeff_v = *((HVX_Vector *)(c3_coeffs));
    c4_coeff_v = *((HVX_Vector *)(c4_coeffs));
    c5_coeff_v = *((HVX_Vector *)(c5_coeffs));
    c6_coeff_v = *((HVX_Vector *)(c6_coeffs));
    c7_coeff_v = *((HVX_Vector *)(c7_coeffs));

    /* Convert coefficients from sf to qf32 format. Use the same vector for both representations */
    c0_coeff_v = Q6_Vqf32_vadd_VsfVsf(c0_coeff_v, zero_v_sf);
    c1_coeff_v = Q6_Vqf32_vadd_VsfVsf(c1_coeff_v, zero_v_sf);
    c2_coeff_v = Q6_Vqf32_vadd_VsfVsf(c2_coeff_v, zero_v_sf);
    c3_coeff_v = Q6_Vqf32_vadd_VsfVsf(c3_coeff_v, zero_v_sf);
    c4_coeff_v = Q6_Vqf32_vadd_VsfVsf(c4_coeff_v, zero_v_sf);
    c5_coeff_v = Q6_Vqf32_vadd_VsfVsf(c5_coeff_v, zero_v_sf);
    c6_coeff_v = Q6_Vqf32_vadd_VsfVsf(c6_coeff_v, zero_v_sf);
    c7_coeff_v = Q6_Vqf32_vadd_VsfVsf(c7_coeff_v, zero_v_sf);

    /* Split 32-bit coefficients to lower and upper part in order to obtain them later with VLUT16. */
    c0_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
    c1_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
    c2_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
    c3_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c3_coeff_v);
    c4_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c4_coeff_v);
    c5_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c5_coeff_v);
    c6_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c6_coeff_v);
    c7_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c7_coeff_v);

    /*
     * Handle number of whole vectors in input data.
     * Don't process last vector in order to avoid out-of-boundary load.
     */
    for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE)
    {
        block = Q6_R_min_RR(i, BLOCK_SIZE);
        l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

        if (l2fetch_block > 0)
        {
            l2fetch(input_v_ptr + L2FETCH_AHEAD, 128, 128, l2fetch_block, 0);
        }

        /* Process one vector at a time */
        for (int32_t j = 0; j < block; ++j)
        {
            slinec = *input_v_ptr++;

            /* Compose vector of input data from slinec and slinep */
            sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
            sline_tmp = sline;
            /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
            input_shifted_v_qf32 = Q6_Vqf32_vsub_VsfVsf(sline, input_min_v_f);

            /*
             * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
             * in order to get corresponding coefficient indexes
             */
            input_scaled_v_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v_qf32, scale_v);

            /*
             * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
             * to [16.0,32.0) in order to convert float indexes to integer values.
             * Float values, represented in IEEE 754, in range [16.0,32.0] have the
             * same exponent, which means 4 MSB of mantissa carry information about
             * integer index.
             */
            input_scaled_v_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v_qf32, const16_0_v_sf);

            /* Convert back from qf32 to sf in order to extract integer index */
            tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v_qf32);

            /* Only 4 MSB bits of mantissa represent segment index */
            idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);

            idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
            idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
            idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

            /* Obtain the polynomial coefficients from lookup table */
            c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
            c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
            c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
            c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
            c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
            c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
            c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
            c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);
            c4_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c4_coeff_dv.VV), 1);
            c4_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c4_coeff_vp, idx2_v, Q6_V_hi_W(c4_coeff_dv.VV), 1);
            c5_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c5_coeff_dv.VV), 1);
            c5_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c5_coeff_vp, idx2_v, Q6_V_hi_W(c5_coeff_dv.VV), 1);
            c6_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c6_coeff_dv.VV), 1);
            c6_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c6_coeff_vp, idx2_v, Q6_V_hi_W(c6_coeff_dv.VV), 1);
            c7_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c7_coeff_dv.VV), 1);
            c7_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c7_coeff_vp, idx2_v, Q6_V_hi_W(c7_coeff_dv.VV), 1);

            /* Convert input from sf vector to qf32 vector for Horner's method*/
            input_v_qf32 = Q6_Vqf32_vadd_VsfVsf(sline, zero_v_sf);

            /* Perform evaluation of polynomial using Horner's method */
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c7_coeff_vp), input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c6_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c5_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c4_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c3_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c2_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c1_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c0_coeff_vp));

            // /* Store results to the output buffer and convert from qf32 to sf */
            // *((HVX_UVector *)(output_v_ptr++)) = Q6_Vsf_equals_Vqf32(output_v);


            /* Convert from qf32 to sf, store output and go to handle leftover */
            HVX_Vector output_v_f32 =  Q6_Vsf_equals_Vqf32(output_v);
            HVX_VectorPred pred_cap_left = Q6_Q_vcmp_gt_VsfVsf(input_min_v_f, sline_tmp); // 1 if input_min_v_f > sline_tmp
            output_v_f32 = Q6_V_vmux_QVV(pred_cap_left, zero_vec, output_v_f32); // if sline_tmp> input_min_v_f, set to zero
           
            HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(sline_tmp, input_max_v_f); // 1 if sline_tmp > input_max_v_f
            output_v_f32 = Q6_V_vmux_QVV(pred_cap_right, sline_tmp, output_v_f32); // if sline_tmp> input_max_v_f, set to whatever the sline_tmp was
           
            *((HVX_UVector *)(output_v_ptr++)) = output_v_f32;


            /* Prepare slinep for next iteration */
            slinep = slinec;
        }
    }

    /* Handle last whole vector from input data */
    if (vectors_in_rounddown > 0)
    {
        slinec = is_aligned(input_v_ptr, 128) && leftover == 0 ? slinep : *input_v_ptr++;
        sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
        sline_tmp = sline;

        /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
        input_shifted_v_qf32 = Q6_Vqf32_vsub_VsfVsf(sline, input_min_v_f);

        /* Scale shifted input range from [0, input_max - input_min] to [0,16.0) */
        input_scaled_v_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v_qf32, scale_v);

        /*
         * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
         * to [16.0,32.0) in order to convert float indexes to integer values.
         * Float values, represented in IEEE 754, in range [16.0,32.0] have the
         * same exponent, which means 4 MSB of mantissa carry information about
         * integer index.
         */
        input_scaled_v_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v_qf32, const16_0_v_sf);

        /* Convert back from qf32 to sf in order to extract integer index */
        tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v_qf32);

        /* Only 4 MSB bits of mantissa represent segment index */
        idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);

        /* Ensure only 4 MSB bits of mantissa are used as indexes */
        idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
        idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
        idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

        /* Obtain the polynomial coefficients from lookup table */
        c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
        c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
        c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
        c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
        c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
        c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
        c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
        c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);
        c4_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c4_coeff_dv.VV), 1);
        c4_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c4_coeff_vp, idx2_v, Q6_V_hi_W(c4_coeff_dv.VV), 1);
        c5_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c5_coeff_dv.VV), 1);
        c5_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c5_coeff_vp, idx2_v, Q6_V_hi_W(c5_coeff_dv.VV), 1);
        c6_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c6_coeff_dv.VV), 1);
        c6_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c6_coeff_vp, idx2_v, Q6_V_hi_W(c6_coeff_dv.VV), 1);
        c7_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c7_coeff_dv.VV), 1);
        c7_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c7_coeff_vp, idx2_v, Q6_V_hi_W(c7_coeff_dv.VV), 1);

        /* Convert input from sf vector to qf32 vector for Horner's method*/
        input_v_qf32 = Q6_Vqf32_vadd_VsfVsf(sline, zero_v_sf);

        /* Perform evaluation of polynomial using Horner's method */
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c7_coeff_vp), input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c6_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c5_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c4_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c3_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c2_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c1_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c0_coeff_vp));

        /* Convert from qf32 to sf, store output and go to handle leftover */
        HVX_Vector output_v_f32 =  Q6_Vsf_equals_Vqf32(output_v);
        HVX_VectorPred pred_cap_left = Q6_Q_vcmp_gt_VsfVsf(input_min_v_f, sline_tmp); // 1 if input_min_v_f > sline_tmp
        output_v_f32 = Q6_V_vmux_QVV(pred_cap_left, zero_vec, output_v_f32); // if sline_tmp> input_min_v_f, set to zero

        HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(sline_tmp, input_max_v_f); // 1 if sline_tmp > input_max_v_f
        output_v_f32 = Q6_V_vmux_QVV(pred_cap_right, sline_tmp, output_v_f32); // if sline_tmp> input_max_v_f, set to whatever the sline_tmp was

        *((HVX_UVector *)(output_v_ptr++)) = output_v_f32;

        slinep = slinec;
    }

    /* Handle leftover elements */
    if (leftover > 0)
    {
        slinec = (is_in_one_chunk(input_v_ptr, leftover_size, 128)
                   ? slinep
                   : *input_v_ptr++);

        sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);
        sline_tmp = sline;

        /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
        input_shifted_v_qf32 = Q6_Vqf32_vsub_VsfVsf(sline, input_min_v_f);

        /* Scale shifted input range from [0, input_max - input_min] to [0,16.0) */
        input_scaled_v_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v_qf32, scale_v);

        /*
         * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
         * to [16.0,32.0) in order to convert float indexes to integer values.
         * Float values, represented in IEEE 754, in range [16.0,32.0] have the
         * same exponent, which means 4 MSB of mantissa carry information about
         * integer index.
         */
        input_scaled_v_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v_qf32, const16_0_v_sf);

        /* Convert back from qf32 to sf in order to extract integer index */
        tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v_qf32);

        /* Only 4 MSB bits of mantissa represent segment index */
        idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);

        /* Ensure only 4 MSB bits of mantissa are used as indexes */
        idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
        idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
        idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

        /* Obtain the polynomial coefficients from lookup table */
        c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
        c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
        c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
        c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
        c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
        c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
        c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
        c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);
        c4_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c4_coeff_dv.VV), 1);
        c4_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c4_coeff_vp, idx2_v, Q6_V_hi_W(c4_coeff_dv.VV), 1);
        c5_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c5_coeff_dv.VV), 1);
        c5_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c5_coeff_vp, idx2_v, Q6_V_hi_W(c5_coeff_dv.VV), 1);
        c6_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c6_coeff_dv.VV), 1);
        c6_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c6_coeff_vp, idx2_v, Q6_V_hi_W(c6_coeff_dv.VV), 1);
        c7_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c7_coeff_dv.VV), 1);
        c7_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c7_coeff_vp, idx2_v, Q6_V_hi_W(c7_coeff_dv.VV), 1);

        /* Convert input from sf vector to qf32 vector for Horner's method*/
        input_v_qf32 = Q6_Vqf32_vadd_VsfVsf(sline, zero_v_sf);

        /* Perform evaluation of polynomial using Horner's method */
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c7_coeff_vp), input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c6_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c5_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c4_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c3_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c2_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c1_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c0_coeff_vp));

        /* Convert from qf32 to sf */
        // sout = Q6_Vsf_equals_Vqf32(output_v);
        HVX_Vector output_v_f32 =  Q6_Vsf_equals_Vqf32(output_v);
        HVX_VectorPred pred_cap_left = Q6_Q_vcmp_gt_VsfVsf(input_min_v_f, sline_tmp); // 1 if input_min_v_f > sline_tmp
        output_v_f32 = Q6_V_vmux_QVV(pred_cap_left, zero_vec, output_v_f32); // if sline_tmp> input_min_v_f, set to zero


        HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(sline_tmp, input_max_v_f); // 1 if sline_tmp > input_max_v_f
        output_v_f32 = Q6_V_vmux_QVV(pred_cap_right, sline_tmp, output_v_f32); // if sline_tmp> input_max_v_f, set to whatever the sline_tmp was

        sout = output_v_f32;
        /* Store output */
        vstu_variable(output_v_ptr, leftover_size, sout);
    }

    return 0;
}

#endif /* __HVX_ARCH__ >= 68 */

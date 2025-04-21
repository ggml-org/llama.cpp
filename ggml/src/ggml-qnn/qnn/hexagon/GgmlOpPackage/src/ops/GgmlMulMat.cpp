//==============================================================================
// Auto Generated Code for GgmlOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_GgmlMulMat);

// op execute function declarations
template <typename TensorType>
GraphStatus ggmlmulmatImpl(TensorType & out_0, const TensorType & in_0, const TensorType & in_1);

// forward declaration of sample cost function
static float ggmlmulmatCostFunc(const Op * op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((ggmlmulmatImpl<Tensor>), "GgmlMulMat")
 */
DEF_PACKAGE_OP((ggmlmulmatImpl<Tensor>), "GgmlMulMat")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((ggmlmulmatImpl<PlainFloatTensor>), "GgmlMulMat", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((ggmlmulmatImpl<PlainFloatTensor>),
 * "GgmlMulMat", ggmlmulmatCostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax: DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core documentations
 */

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax: DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
     * order of parameters listed determines the order of parameters passed into op execution functions
 * if an op does not have a parameter order definition, parameter order passed into Qnn_addNode
 *   will be passed into op execution functions
 * if an op has a parameter order definition, any parameter passed into Qnn_addNode with unlisted
     *   name will be abandoned
 * if two or more op packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANDATORY refers to whether this parameter is required to be provided at Qnn_addNode
 * DEFAULT is used when MANDATORY is false
 *     if provided as Qnn_Param_t*,
 *       DEFAULT will be used for graph construction when this parameter is not provided at
 *       Qnn_addNode
 *     if provided as nullptr,
 *       graph construction will skip this parameter when this parameter is not provided at
 *       Qnn_addNode
 */

namespace {

constexpr const size_t kBytesPerVector  = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kFloatsPerVector = kBytesPerVector / sizeof(float);
constexpr const size_t kAlignMask       = kBytesPerVector - 1;

inline size_t unaligned_bytes(const void * addr) {
    return ((size_t) addr) & kAlignMask;
}

inline bool is_addr_aligned(void * addr) {
    return unaligned_bytes(addr) == 0;
}

inline float vec_dot_product_f32(const float * src0, const float * src1, size_t count) {
    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / kFloatsPerVector);
    HVX_Vector * iptr1     = ((HVX_Vector *) src1);
    HVX_Vector   prev0     = *iptr0++;
    HVX_Vector   prev1     = *iptr1++;
    HVX_Vector   sum       = Q6_V_vzero();

    // TODO: prefetch?
    while (iptr0 < iptr0_end) {
        HVX_Vector curr0 = *iptr0++;
        HVX_Vector curr1 = *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        sum              = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, s1), sum);
        prev0            = curr0;
        prev1            = curr1;
    }

    if ((iptr0_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also: https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        HVX_Vector curr0 = is_addr_aligned(iptr0) ? prev0 : *iptr0++;
        HVX_Vector curr1 = is_addr_aligned(iptr1) ? prev1 : *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        sum              = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, s1), sum);
        prev0            = curr0;
        prev1            = curr1;
    }

    const size_t leftover       = count % kFloatsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 = (leftover_bytes + unaligned_bytes(iptr0) > kBytesPerVector) ? *iptr0 : prev0;
        curr0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 = (leftover_bytes + unaligned_bytes(iptr1) > kBytesPerVector) ? *iptr1 : prev1;
        curr1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        sum = Q6_Vqf32_vadd_Vqf32Vqf32(
            Q6_V_valign_VVR(Q6_Vqf32_vmpy_VsfVsf(curr0, curr1), Q6_V_vzero(), leftover_bytes), sum);
    }

    // TODO: do we have a better way to do the reduction?
    for (size_t i = kFloatsPerVector / 2; i > 0; i /= 2) {
        sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vror_VR(sum, i * sizeof(float)));
    }

    float result;
    q6op_vstu_variable_ARV(&result, sizeof(float), Q6_Vsf_equals_Vqf32(sum));
    return result;
}

template <typename TensorType>
inline GraphStatus mul_mat_2d_f32(TensorType & out_0, const TensorType & in_0, const TensorType & in_1) {
    // TODO: handle strides?
    if (in_1.dim(1) != in_0.dim(1)) {
        return GraphStatus::ErrorDimensions;
    }

    size_t dims[4] = { in_1.dim(0), in_0.dim(0) };
    out_0.set_dims(dims);

    auto in0_ptr = (float *) in_0.raw_data_const();
    auto in1_ptr = (float *) in_1.raw_data_const();
    auto out_ptr = (float *) out_0.raw_data();

    for (size_t i = 0; i < dims[0]; i++) {
        // TODO: prefetch?
        auto * in1_row = in1_ptr + i * in_1.dim(1);
        auto * out_row = out_ptr + i * dims[1];
        for (size_t j = 0; j < dims[1]; j++) {
            *out_row++ = vec_dot_product_f32(in0_ptr + j * in_0.dim(1), in1_row, in_0.dim(1));
        }
    }

    return GraphStatus::Success;
}

}  // namespace

/* execute functions for ops */

template <typename TensorType>
GraphStatus ggmlmulmatImpl(TensorType & out_0, const TensorType & in_0, const TensorType & in_1) {
    if (!in_0.raw_data_const() || !in_1.raw_data_const() || !out_0.raw_data()) {
        return GraphStatus::ErrorBadInput;
    }

    if (in_0.rank() != in_1.rank()) {
        return GraphStatus::ErrorRank;
    }

    auto rank = in_0.rank();
    switch (rank) {
        case 4:
        case 3:
            // TODO: add implementation
            return GraphStatus::ErrorUnsupported;
        case 2:
            return mul_mat_2d_f32(out_0, in_0, in_1);
    }

    return GraphStatus::ErrorRank;
}

__attribute__((unused)) static float ggmlmulmatCostFunc(const Op * op) {
    /*
   * add code here
   * */

    float cost = 0.0;  // add cost computation here
    return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_GgmlMulMat);

#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// pipelines

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> pipeline;
};

@interface ggml_metal_pipeline_wrapper : NSObject

@property (nonatomic, assign) struct ggml_metal_pipeline pipeline;

@end

@implementation ggml_metal_pipeline_wrapper
- (void) dealloc {
    [_pipeline.pipeline release];
    [super dealloc];
}
@end

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;
};

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
}

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

enum ggml_metal_pipeline_type {
    GGML_METAL_PIPELINE_TYPE_ADD_ID,
    GGML_METAL_PIPELINE_TYPE_REPEAT_F32,
    GGML_METAL_PIPELINE_TYPE_REPEAT_F16,
    GGML_METAL_PIPELINE_TYPE_REPEAT_I32,
    GGML_METAL_PIPELINE_TYPE_REPEAT_I16,
    GGML_METAL_PIPELINE_TYPE_SCALE,
    GGML_METAL_PIPELINE_TYPE_SCALE_4,
    GGML_METAL_PIPELINE_TYPE_CLAMP,
    GGML_METAL_PIPELINE_TYPE_TANH,
    GGML_METAL_PIPELINE_TYPE_RELU,
    GGML_METAL_PIPELINE_TYPE_SIGMOID,
    GGML_METAL_PIPELINE_TYPE_GELU,
    GGML_METAL_PIPELINE_TYPE_GELU_4,
    GGML_METAL_PIPELINE_TYPE_GELU_ERF,
    GGML_METAL_PIPELINE_TYPE_GELU_ERF_4,
    GGML_METAL_PIPELINE_TYPE_GELU_QUICK,
    GGML_METAL_PIPELINE_TYPE_GELU_QUICK_4,
    GGML_METAL_PIPELINE_TYPE_SILU,
    GGML_METAL_PIPELINE_TYPE_SILU_4,
    GGML_METAL_PIPELINE_TYPE_ELU,
    GGML_METAL_PIPELINE_TYPE_ABS,
    GGML_METAL_PIPELINE_TYPE_SGN,
    GGML_METAL_PIPELINE_TYPE_STEP,
    GGML_METAL_PIPELINE_TYPE_HARDSWISH,
    GGML_METAL_PIPELINE_TYPE_HARDSIGMOID,
    GGML_METAL_PIPELINE_TYPE_EXP,
    GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F16,
    GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F16_4,
    GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F32,
    GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F32_4,
    GGML_METAL_PIPELINE_TYPE_DIAG_MASK_INF,
    GGML_METAL_PIPELINE_TYPE_DIAG_MASK_INF_8,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_F32,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_F16,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_BF16,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_0,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_1,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_0,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_1,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q8_0,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_MXFP4,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q2_K,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q3_K,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_K,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_K,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q6_K,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_XXS,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_XS,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ3_XXS,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ3_S,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_S,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ1_S,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ1_M,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ4_NL,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ4_XS,
    GGML_METAL_PIPELINE_TYPE_GET_ROWS_I32,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_F32,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_F16,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_BF16,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q8_0,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q4_0,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q4_1,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q5_0,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q5_1,
    GGML_METAL_PIPELINE_TYPE_SET_ROWS_IQ4_NL,
    GGML_METAL_PIPELINE_TYPE_L2_NORM,
    GGML_METAL_PIPELINE_TYPE_GROUP_NORM,
    GGML_METAL_PIPELINE_TYPE_NORM,
    GGML_METAL_PIPELINE_TYPE_SSM_CONV_F32,
    GGML_METAL_PIPELINE_TYPE_SSM_SCAN_F32,
    GGML_METAL_PIPELINE_TYPE_SSM_SCAN_F32_GROUP,
    GGML_METAL_PIPELINE_TYPE_RWKV_WKV6_F32,
    GGML_METAL_PIPELINE_TYPE_RWKV_WKV7_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_F32_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_F32_F32_C4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_C4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_1ROW,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_L4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_C4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_1ROW,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_L4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_BF16,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_1_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_1_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q8_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_MXFP4_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_3,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_5,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q2_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q3_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_Q6_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_XXS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_XS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ3_XXS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ3_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ1_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ1_M_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ4_NL_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ4_XS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F32_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F16_F32,
  //GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F16_F32_1ROW,
  //GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F16_F32_L4,
  //GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F16_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_BF16_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_1_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_1_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q8_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_MXFP4_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q2_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q3_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q6_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_XXS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_XS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ3_XXS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ3_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ1_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ1_M_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ4_NL_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ4_XS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_F32_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_F16_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_BF16_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_1_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_1_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q8_0_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_MXFP4_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q2_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q3_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_Q6_K_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_XXS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_XS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ3_XXS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ3_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ1_S_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ1_M_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ4_NL_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ4_XS_F32,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_1,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_2,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_4,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_6,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_8,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_10,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_F32_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_F16_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_BF16_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_0_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_1_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_0_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_1_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q8_0_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MXFP4_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q2_K_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q3_K_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_K_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_K_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q6_K_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_XXS_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_XS_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ3_XXS_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ3_S_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_S_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ1_S_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ1_M_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ4_NL_F16,
    GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ4_XS_F16,
    GGML_METAL_PIPELINE_TYPE_ROPE_NORM_F32,
    GGML_METAL_PIPELINE_TYPE_ROPE_NORM_F16,
    GGML_METAL_PIPELINE_TYPE_ROPE_MULTI_F32,
    GGML_METAL_PIPELINE_TYPE_ROPE_MULTI_F16,
    GGML_METAL_PIPELINE_TYPE_ROPE_VISION_F32,
    GGML_METAL_PIPELINE_TYPE_ROPE_VISION_F16,
    GGML_METAL_PIPELINE_TYPE_ROPE_NEOX_F32,
    GGML_METAL_PIPELINE_TYPE_ROPE_NEOX_F16,
    GGML_METAL_PIPELINE_TYPE_IM2COL_F16,
    GGML_METAL_PIPELINE_TYPE_IM2COL_F32,
    GGML_METAL_PIPELINE_TYPE_IM2COL_EXT_F16,
    GGML_METAL_PIPELINE_TYPE_IM2COL_EXT_F32,
    GGML_METAL_PIPELINE_TYPE_CONV_TRANSPOSE_1D_F32_F32,
    GGML_METAL_PIPELINE_TYPE_CONV_TRANSPOSE_1D_F16_F32,
    GGML_METAL_PIPELINE_TYPE_UPSCALE_F32,
    GGML_METAL_PIPELINE_TYPE_PAD_F32,
    GGML_METAL_PIPELINE_TYPE_PAD_REFLECT_1D_F32,
    GGML_METAL_PIPELINE_TYPE_ARANGE_F32,
    GGML_METAL_PIPELINE_TYPE_TIMESTEP_EMBEDDING_F32,
    GGML_METAL_PIPELINE_TYPE_ARGSORT_F32_I32_ASC,
    GGML_METAL_PIPELINE_TYPE_ARGSORT_F32_I32_DESC,
    GGML_METAL_PIPELINE_TYPE_LEAKY_RELU_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_F16,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_BF16,
    GGML_METAL_PIPELINE_TYPE_CPY_F16_F16,
    GGML_METAL_PIPELINE_TYPE_CPY_F16_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_BF16_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_BF16_BF16,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_I32,
    GGML_METAL_PIPELINE_TYPE_CPY_I32_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_Q8_0,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_Q4_0,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_Q4_1,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_Q5_0,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_Q5_1,
    GGML_METAL_PIPELINE_TYPE_CPY_F32_IQ4_NL,
    GGML_METAL_PIPELINE_TYPE_CPY_Q4_0_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_Q4_0_F16,
    GGML_METAL_PIPELINE_TYPE_CPY_Q4_1_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_Q4_1_F16,
    GGML_METAL_PIPELINE_TYPE_CPY_Q5_0_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_Q5_0_F16,
    GGML_METAL_PIPELINE_TYPE_CPY_Q5_1_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_Q5_1_F16,
    GGML_METAL_PIPELINE_TYPE_CPY_Q8_0_F32,
    GGML_METAL_PIPELINE_TYPE_CPY_Q8_0_F16,
    GGML_METAL_PIPELINE_TYPE_CONCAT,
    GGML_METAL_PIPELINE_TYPE_SQR,
    GGML_METAL_PIPELINE_TYPE_SQRT,
    GGML_METAL_PIPELINE_TYPE_SIN,
    GGML_METAL_PIPELINE_TYPE_COS,
    GGML_METAL_PIPELINE_TYPE_NEG,
    GGML_METAL_PIPELINE_TYPE_REGLU,
    GGML_METAL_PIPELINE_TYPE_GEGLU,
    GGML_METAL_PIPELINE_TYPE_SWIGLU,
    GGML_METAL_PIPELINE_TYPE_SWIGLU_OAI,
    GGML_METAL_PIPELINE_TYPE_GEGLU_ERF,
    GGML_METAL_PIPELINE_TYPE_GEGLU_QUICK,
    GGML_METAL_PIPELINE_TYPE_SUM_ROWS,
    GGML_METAL_PIPELINE_TYPE_MEAN,
    GGML_METAL_PIPELINE_TYPE_POOL_2D_AVG_F32,
    GGML_METAL_PIPELINE_TYPE_POOL_2D_MAX_F32,
    GGML_METAL_PIPELINE_TYPE_ARGMAX,

    GGML_METAL_PIPELINE_TYPE_COUNT
};


struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;

    // used to enable concurrent execution of ops in the command buffers
    ggml_mem_ranges_t mem_ranges;
};

struct ggml_metal {
    id<MTLDevice>       device;
    id<MTLLibrary>      library;
    id<MTLCommandQueue> queue; // currently a pointer to the device queue, but might become separate queue [TAG_QUEUE_PER_BACKEND]

    //struct ggml_metal_device_props props_dev;
    ggml_metal_device_t ctx_dev;

    dispatch_queue_t d_queue;

    // the set of pre-compiled pipelines for this context
    struct ggml_metal_pipeline pipelines[GGML_METAL_PIPELINE_TYPE_COUNT];

    // additional, inference-time compiled pipelines
    NSMutableDictionary * pipelines_ext;

    bool use_bfloat;
    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    bool capture_next_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t ctx_dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    res->device = ggml_metal_device_get_device(ctx_dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[res->device name] UTF8String]);

    res->library = ggml_metal_device_get_library(ctx_dev);
    if (res->library == nil) {
        GGML_LOG_ERROR("%s: error: metal library is nil\n", __func__);
        return NULL;
    }

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    res->queue = ggml_metal_device_get_queue(ctx_dev);
    if (res->queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->ctx_dev = ctx_dev;

    struct ggml_metal_device_props props_dev = ggml_metal_device_get_props(ctx_dev);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_bfloat      = props_dev.has_bfloat;
    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use bfloat         = %s\n", __func__, res->use_bfloat         ? "true" : "false");
    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_next_compute = false;
    res->capture_started = false;
    res->capture_scope = nil;

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;

        if (res->use_concurrency) {
            res->cmd_bufs[i].mem_ranges = ggml_mem_ranges_init(res->debug_graph);
        }
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    // load default pipelines
    {
        NSError * error = nil;

        for (int i = 0; i < GGML_METAL_PIPELINE_TYPE_COUNT; ++i) {
            res->pipelines[i].pipeline = nil;
        }

#define GGML_METAL_ADD_PIPELINE(e, name, supported) \
        if (supported) { \
            struct ggml_metal_pipeline * pipeline = &res->pipelines[e]; \
            id<MTLFunction> function = [res->library newFunctionWithName:@"kernel_"#name]; \
            pipeline->pipeline = [res->device newComputePipelineStateWithFunction:function error:&error]; \
            GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, "kernel_"#name, (void *) pipeline->pipeline, \
                    (int) pipeline->pipeline.maxTotalThreadsPerThreadgroup, \
                    (int) pipeline->pipeline.threadExecutionWidth); \
            [function release]; \
            if (error) { \
                GGML_LOG_ERROR("%s: error: load pipeline error: %s\n", __func__, [[error description] UTF8String]); \
                return NULL; \
            } \
        } else { \
            GGML_LOG_WARN("%s: skipping %-40s (not supported)\n", __func__, "kernel_"#name); \
        }

        const bool has_simdgroup_mm        = props_dev.has_simdgroup_mm;
        const bool has_simdgroup_reduction = props_dev.has_simdgroup_reduction;
        const bool has_bfloat              = props_dev.has_bfloat;

        // simd_sum and simd_max requires MTLGPUFamilyApple7

        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ADD_ID,                          add_id,                          true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_REPEAT_F32,                      repeat_f32,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_REPEAT_F16,                      repeat_f16,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_REPEAT_I32,                      repeat_i32,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_REPEAT_I16,                      repeat_i16,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SCALE,                           scale,                           true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SCALE_4,                         scale_4,                         true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CLAMP,                           clamp,                           true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_TANH,                            tanh,                            true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_RELU,                            relu,                            true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SIGMOID,                         sigmoid,                         true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GELU,                            gelu,                            true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GELU_4,                          gelu_4,                          true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GELU_ERF,                        gelu_erf,                        true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GELU_ERF_4,                      gelu_erf_4,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GELU_QUICK,                      gelu_quick,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GELU_QUICK_4,                    gelu_quick_4,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SILU,                            silu,                            true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SILU_4,                          silu_4,                          true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ELU,                             elu,                             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ABS,                             abs,                             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SGN,                             sgn,                             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_STEP,                            step,                            true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_HARDSWISH,                       hardswish,                       true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_HARDSIGMOID,                     hardsigmoid,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_EXP,                             exp,                             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F16,                    soft_max_f16,                    has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F16_4,                  soft_max_f16_4,                  has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F32,                    soft_max_f32,                    has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F32_4,                  soft_max_f32_4,                  has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_DIAG_MASK_INF,                   diag_mask_inf,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_DIAG_MASK_INF_8,                 diag_mask_inf_8,                 true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_F32,                    get_rows_f32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_F16,                    get_rows_f16,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_BF16,                   get_rows_bf16,                   has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_0,                   get_rows_q4_0,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_1,                   get_rows_q4_1,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_0,                   get_rows_q5_0,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_1,                   get_rows_q5_1,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q8_0,                   get_rows_q8_0,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_MXFP4,                  get_rows_mxfp4,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q2_K,                   get_rows_q2_K,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q3_K,                   get_rows_q3_K,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_K,                   get_rows_q4_K,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_K,                   get_rows_q5_K,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q6_K,                   get_rows_q6_K,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_XXS,                get_rows_iq2_xxs,                true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_XS,                 get_rows_iq2_xs,                 true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ3_XXS,                get_rows_iq3_xxs,                true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ3_S,                  get_rows_iq3_s,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_S,                  get_rows_iq2_s,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ1_S,                  get_rows_iq1_s,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ1_M,                  get_rows_iq1_m,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ4_NL,                 get_rows_iq4_nl,                 true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ4_XS,                 get_rows_iq4_xs,                 true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GET_ROWS_I32,                    get_rows_i32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_F32,                    set_rows_f32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_F16,                    set_rows_f16,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_BF16,                   set_rows_bf16,                   has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q8_0,                   set_rows_q8_0,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q4_0,                   set_rows_q4_0,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q4_1,                   set_rows_q4_1,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q5_0,                   set_rows_q5_0,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q5_1,                   set_rows_q5_1,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SET_ROWS_IQ4_NL,                 set_rows_iq4_nl,                 true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_L2_NORM,                         l2_norm,                         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GROUP_NORM,                      group_norm,                      has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_NORM,                            norm,                            has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SSM_CONV_F32,                    ssm_conv_f32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SSM_SCAN_F32,                    ssm_scan_f32,                    has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SSM_SCAN_F32_GROUP,              ssm_scan_f32_group,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_RWKV_WKV6_F32,                   rwkv_wkv6_f32,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_RWKV_WKV7_F32,                   rwkv_wkv7_f32,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_F32_F32,                  mul_mv_f32_f32,                  has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_F32_F32_C4,               mul_mv_f32_f32_c4,               true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32,                 mul_mv_bf16_f32,                 has_simdgroup_reduction && has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_C4,              mul_mv_bf16_f32_c4,              has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_1ROW,            mul_mv_bf16_f32_1row,            has_simdgroup_reduction && has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_L4,              mul_mv_bf16_f32_l4,              has_simdgroup_reduction && has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_BF16,                mul_mv_bf16_bf16,                has_simdgroup_reduction && has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32,                  mul_mv_f16_f32,                  has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_C4,               mul_mv_f16_f32_c4,               true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_1ROW,             mul_mv_f16_f32_1row,             has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_L4,               mul_mv_f16_f32_l4,               has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F16,                  mul_mv_f16_f16,                  has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_0_F32,                 mul_mv_q4_0_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_1_F32,                 mul_mv_q4_1_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_0_F32,                 mul_mv_q5_0_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_1_F32,                 mul_mv_q5_1_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q8_0_F32,                 mul_mv_q8_0_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_MXFP4_F32,                mul_mv_mxfp4_f32,                has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_2,         mul_mv_ext_f32_f32_r1_2,         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_3,         mul_mv_ext_f32_f32_r1_3,         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_4,         mul_mv_ext_f32_f32_r1_4,         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_5,         mul_mv_ext_f32_f32_r1_5,         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_2,         mul_mv_ext_f16_f32_r1_2,         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_3,         mul_mv_ext_f16_f32_r1_3,         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_4,         mul_mv_ext_f16_f32_r1_4,         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_5,         mul_mv_ext_f16_f32_r1_5,         has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_2,        mul_mv_ext_q4_0_f32_r1_2,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_3,        mul_mv_ext_q4_0_f32_r1_3,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_4,        mul_mv_ext_q4_0_f32_r1_4,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_5,        mul_mv_ext_q4_0_f32_r1_5,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_2,        mul_mv_ext_q4_1_f32_r1_2,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_3,        mul_mv_ext_q4_1_f32_r1_3,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_4,        mul_mv_ext_q4_1_f32_r1_4,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_5,        mul_mv_ext_q4_1_f32_r1_5,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_2,        mul_mv_ext_q5_0_f32_r1_2,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_3,        mul_mv_ext_q5_0_f32_r1_3,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_4,        mul_mv_ext_q5_0_f32_r1_4,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_5,        mul_mv_ext_q5_0_f32_r1_5,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_2,        mul_mv_ext_q5_1_f32_r1_2,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_3,        mul_mv_ext_q5_1_f32_r1_3,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_4,        mul_mv_ext_q5_1_f32_r1_4,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_5,        mul_mv_ext_q5_1_f32_r1_5,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_2,        mul_mv_ext_q8_0_f32_r1_2,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_3,        mul_mv_ext_q8_0_f32_r1_3,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_4,        mul_mv_ext_q8_0_f32_r1_4,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_5,        mul_mv_ext_q8_0_f32_r1_5,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_2,       mul_mv_ext_mxfp4_f32_r1_2,       has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_3,       mul_mv_ext_mxfp4_f32_r1_3,       has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_4,       mul_mv_ext_mxfp4_f32_r1_4,       has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_5,       mul_mv_ext_mxfp4_f32_r1_5,       has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_2,        mul_mv_ext_q4_K_f32_r1_2,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_3,        mul_mv_ext_q4_K_f32_r1_3,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_4,        mul_mv_ext_q4_K_f32_r1_4,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_5,        mul_mv_ext_q4_K_f32_r1_5,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_2,        mul_mv_ext_q5_K_f32_r1_2,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_3,        mul_mv_ext_q5_K_f32_r1_3,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_4,        mul_mv_ext_q5_K_f32_r1_4,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_5,        mul_mv_ext_q5_K_f32_r1_5,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_2,        mul_mv_ext_q6_K_f32_r1_2,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_3,        mul_mv_ext_q6_K_f32_r1_3,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_4,        mul_mv_ext_q6_K_f32_r1_4,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_5,        mul_mv_ext_q6_K_f32_r1_5,        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_2,      mul_mv_ext_iq4_nl_f32_r1_2,      has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_3,      mul_mv_ext_iq4_nl_f32_r1_3,      has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_4,      mul_mv_ext_iq4_nl_f32_r1_4,      has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_5,      mul_mv_ext_iq4_nl_f32_r1_5,      has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q2_K_F32,                 mul_mv_q2_K_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q3_K_F32,                 mul_mv_q3_K_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_K_F32,                 mul_mv_q4_K_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_K_F32,                 mul_mv_q5_K_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_Q6_K_F32,                 mul_mv_q6_K_f32,                 has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_XXS_F32,              mul_mv_iq2_xxs_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_XS_F32,               mul_mv_iq2_xs_f32,               has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ3_XXS_F32,              mul_mv_iq3_xxs_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ3_S_F32,                mul_mv_iq3_s_f32,                has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_S_F32,                mul_mv_iq2_s_f32,                has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ1_S_F32,                mul_mv_iq1_s_f32,                has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ1_M_F32,                mul_mv_iq1_m_f32,                has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ4_NL_F32,               mul_mv_iq4_nl_f32,               has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ4_XS_F32,               mul_mv_iq4_xs_f32,               has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F32_F32,               mul_mv_id_f32_f32,               has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F16_F32,               mul_mv_id_f16_f32,               has_simdgroup_reduction);
        //GGML_METAL_ADPIPELINEEL(GGML_METAPIPELINEEL_TYPE_MUL_MV_ID_F16_F32_1ROW,          mul_mv_id_f16_f32_1row,          has_simdgroup_reduction);
        //GGML_METAL_ADPIPELINEEL(GGML_METAPIPELINEEL_TYPE_MUL_MV_ID_F16_F32_L4,            mul_mv_id_f16_f32_l4,            has_simdgroup_reduction);
        //GGML_METAL_ADPIPELINEEL(GGML_METAPIPELINEEL_TYPE_MUL_MV_ID_F16_F16,               mul_mv_id_f16_f16,               has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_BF16_F32,              mul_mv_id_bf16_f32,              has_simdgroup_reduction && has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_0_F32,              mul_mv_id_q4_0_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_1_F32,              mul_mv_id_q4_1_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_0_F32,              mul_mv_id_q5_0_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_1_F32,              mul_mv_id_q5_1_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q8_0_F32,              mul_mv_id_q8_0_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_MXFP4_F32,             mul_mv_id_mxfp4_f32,             has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q2_K_F32,              mul_mv_id_q2_K_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q3_K_F32,              mul_mv_id_q3_K_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_K_F32,              mul_mv_id_q4_K_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_K_F32,              mul_mv_id_q5_K_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q6_K_F32,              mul_mv_id_q6_K_f32,              has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_XXS_F32,           mul_mv_id_iq2_xxs_f32,           has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_XS_F32,            mul_mv_id_iq2_xs_f32,            has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ3_XXS_F32,           mul_mv_id_iq3_xxs_f32,           has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ3_S_F32,             mul_mv_id_iq3_s_f32,             has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_S_F32,             mul_mv_id_iq2_s_f32,             has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ1_S_F32,             mul_mv_id_iq1_s_f32,             has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ1_M_F32,             mul_mv_id_iq1_m_f32,             has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ4_NL_F32,            mul_mv_id_iq4_nl_f32,            has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ4_XS_F32,            mul_mv_id_iq4_xs_f32,            has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_F32_F32,                  mul_mm_f32_f32,                  has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_F16_F32,                  mul_mm_f16_f32,                  has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_BF16_F32,                 mul_mm_bf16_f32,                 has_simdgroup_mm && has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_0_F32,                 mul_mm_q4_0_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_1_F32,                 mul_mm_q4_1_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_0_F32,                 mul_mm_q5_0_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_1_F32,                 mul_mm_q5_1_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q8_0_F32,                 mul_mm_q8_0_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_MXFP4_F32,                mul_mm_mxfp4_f32,                has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q2_K_F32,                 mul_mm_q2_K_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q3_K_F32,                 mul_mm_q3_K_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_K_F32,                 mul_mm_q4_K_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_K_F32,                 mul_mm_q5_K_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_Q6_K_F32,                 mul_mm_q6_K_f32,                 has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_XXS_F32,              mul_mm_iq2_xxs_f32,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_XS_F32,               mul_mm_iq2_xs_f32,               has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ3_XXS_F32,              mul_mm_iq3_xxs_f32,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ3_S_F32,                mul_mm_iq3_s_f32,                has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_S_F32,                mul_mm_iq2_s_f32,                has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ1_S_F32,                mul_mm_iq1_s_f32,                has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ1_M_F32,                mul_mm_iq1_m_f32,                has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ4_NL_F32,               mul_mm_iq4_nl_f32,               has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ4_XS_F32,               mul_mm_iq4_xs_f32,               has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_1,       mul_mm_id_map0_f16_ne20_1,       has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_2,       mul_mm_id_map0_f16_ne20_2,       has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_4,       mul_mm_id_map0_f16_ne20_4,       has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_6,       mul_mm_id_map0_f16_ne20_6,       has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_8,       mul_mm_id_map0_f16_ne20_8,       has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_10,      mul_mm_id_map0_f16_ne20_10,      has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_16,      mul_mm_id_map0_f16_ne20_16,      has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_F32_F16,               mul_mm_id_f32_f16,               has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_F16_F16,               mul_mm_id_f16_f16,               has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_BF16_F16,              mul_mm_id_bf16_f16,              has_simdgroup_mm && has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_0_F16,              mul_mm_id_q4_0_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_1_F16,              mul_mm_id_q4_1_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_0_F16,              mul_mm_id_q5_0_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_1_F16,              mul_mm_id_q5_1_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q8_0_F16,              mul_mm_id_q8_0_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MXFP4_F16,             mul_mm_id_mxfp4_f16,             has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q2_K_F16,              mul_mm_id_q2_K_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q3_K_F16,              mul_mm_id_q3_K_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_K_F16,              mul_mm_id_q4_K_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_K_F16,              mul_mm_id_q5_K_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q6_K_F16,              mul_mm_id_q6_K_f16,              has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_XXS_F16,           mul_mm_id_iq2_xxs_f16,           has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_XS_F16,            mul_mm_id_iq2_xs_f16,            has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ3_XXS_F16,           mul_mm_id_iq3_xxs_f16,           has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ3_S_F16,             mul_mm_id_iq3_s_f16,             has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_S_F16,             mul_mm_id_iq2_s_f16,             has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ1_S_F16,             mul_mm_id_iq1_s_f16,             has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ1_M_F16,             mul_mm_id_iq1_m_f16,             has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ4_NL_F16,            mul_mm_id_iq4_nl_f16,            has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ4_XS_F16,            mul_mm_id_iq4_xs_f16,            has_simdgroup_mm);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ROPE_NORM_F32,                   rope_norm_f32,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ROPE_NORM_F16,                   rope_norm_f16,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ROPE_MULTI_F32,                  rope_multi_f32,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ROPE_MULTI_F16,                  rope_multi_f16,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ROPE_VISION_F32,                 rope_vision_f32,                 true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ROPE_VISION_F16,                 rope_vision_f16,                 true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ROPE_NEOX_F32,                   rope_neox_f32,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ROPE_NEOX_F16,                   rope_neox_f16,                   true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_IM2COL_F16,                      im2col_f16,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_IM2COL_F32,                      im2col_f32,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_IM2COL_EXT_F16,                  im2col_ext_f16,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_IM2COL_EXT_F32,                  im2col_ext_f32,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CONV_TRANSPOSE_1D_F32_F32,       conv_transpose_1d_f32_f32,       true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CONV_TRANSPOSE_1D_F16_F32,       conv_transpose_1d_f16_f32,       true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_UPSCALE_F32,                     upscale_f32,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_PAD_F32,                         pad_f32,                         true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_PAD_REFLECT_1D_F32,              pad_reflect_1d_f32,              true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_TIMESTEP_EMBEDDING_F32,          timestep_embedding_f32,          true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ARANGE_F32,                      arange_f32,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ARGSORT_F32_I32_ASC,             argsort_f32_i32_asc,             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ARGSORT_F32_I32_DESC,            argsort_f32_i32_desc,            true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_LEAKY_RELU_F32,                  leaky_relu_f32,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_F32,                     cpy_f32_f32,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_F16,                     cpy_f32_f16,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_BF16,                    cpy_f32_bf16,                    has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F16_F32,                     cpy_f16_f32,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F16_F16,                     cpy_f16_f16,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_BF16_F32,                    cpy_bf16_f32,                    has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_BF16_BF16,                   cpy_bf16_bf16,                   has_bfloat);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_I32,                     cpy_f32_i32,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_I32_F32,                     cpy_i32_f32,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_Q8_0,                    cpy_f32_q8_0,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_Q4_0,                    cpy_f32_q4_0,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_Q4_1,                    cpy_f32_q4_1,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_Q5_0,                    cpy_f32_q5_0,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_Q5_1,                    cpy_f32_q5_1,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_F32_IQ4_NL,                  cpy_f32_iq4_nl,                  true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q4_0_F32,                    cpy_q4_0_f32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q4_0_F16,                    cpy_q4_0_f16,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q4_1_F32,                    cpy_q4_1_f32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q4_1_F16,                    cpy_q4_1_f16,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q5_0_F32,                    cpy_q5_0_f32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q5_0_F16,                    cpy_q5_0_f16,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q5_1_F32,                    cpy_q5_1_f32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q5_1_F16,                    cpy_q5_1_f16,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q8_0_F32,                    cpy_q8_0_f32,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CPY_Q8_0_F16,                    cpy_q8_0_f16,                    true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_CONCAT,                          concat,                          true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SQR,                             sqr,                             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SQRT,                            sqrt,                            true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SIN,                             sin,                             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_COS,                             cos,                             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_NEG,                             neg,                             true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_REGLU,                           reglu,                           true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GEGLU,                           geglu,                           true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SWIGLU,                          swiglu,                          true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SWIGLU_OAI,                      swiglu_oai,                      true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GEGLU_ERF,                       geglu_erf,                       true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_GEGLU_QUICK,                     geglu_quick,                     true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_SUM_ROWS,                        sum_rows,                        has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_MEAN,                            mean,                            has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_ARGMAX,                          argmax,                          has_simdgroup_reduction);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_POOL_2D_AVG_F32,                 pool_2d_avg_f32,                 true);
        GGML_METAL_ADD_PIPELINE(GGML_METAL_PIPELINE_TYPE_POOL_2D_MAX_F32,                 pool_2d_max_f32,                 true);
    }

    res->pipelines_ext = [[NSMutableDictionary alloc] init];

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }

        if (ctx->cmd_bufs[i].mem_ranges) {
            ggml_mem_ranges_free(ctx->cmd_bufs[i].mem_ranges);
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    for (int i = 0; i < GGML_METAL_PIPELINE_TYPE_COUNT; ++i) {
        [ctx->pipelines[i].pipeline release];
    }

    if (ctx->pipelines_ext) {
        [ctx->pipelines_ext release];
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    free(ctx);
}

void * ggml_metal_get_pipeline(ggml_metal_t ctx, const char * name) {
    NSString * key = [NSString stringWithUTF8String:name];

    ggml_metal_pipeline_wrapper * obj = [ctx->pipelines_ext objectForKey:key];
    if (obj) {
        return obj.pipeline.pipeline;
    }

    return nil;
}

void * ggml_metal_compile_pipeline(ggml_metal_t ctx, const char * base, const char * name, ggml_metal_cv_t cv) {
    id<MTLComputePipelineState> res = nil;

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        // TODO: make sure it is thread-safe to compile pipelines in parallel
        id<MTLFunction> mtl_function = [ctx->library newFunctionWithName:base_func constantValues:(cv ? cv->obj : nil) error:&error];
        if (!mtl_function) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);

            return nil;
        }

        struct ggml_metal_pipeline pipeline = {
            /*.pipeline =*/ [ctx->device newComputePipelineStateWithFunction:mtl_function error:&error],
        };

        ggml_metal_pipeline_wrapper * obj = [[ggml_metal_pipeline_wrapper alloc] init];
        obj.pipeline = pipeline;

        res = obj.pipeline.pipeline;

        NSString * key = [NSString stringWithUTF8String:name];
        [ctx->pipelines_ext setObject:obj forKey:key];

        [mtl_function release];
        [obj release];

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name, (void *) pipeline.pipeline,
                (int) pipeline.pipeline.maxTotalThreadsPerThreadgroup,
                (int) pipeline.pipeline.threadExecutionWidth);
    }

    return res;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    // release any completed command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                GGML_ABORT("fatal error");
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }
}

// TODO: temporary shim
static id<MTLBuffer> ggml_metal_get_buffer(const struct ggml_tensor * t, size_t * offs) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    struct ggml_metal_buffer_id res = ggml_metal_buffer_get_id(buffer->context, t);

    *offs = res.offs;

    return res.metal;
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLBuffer> buf_src = [ctx->device newBufferWithBytes:data
                                                         length:size
                                                        options:MTLResourceStorageModeShared];

        size_t buf_dst_offset = 0;
        id<MTLBuffer> buf_dst = ggml_metal_get_buffer(tensor, &buf_dst_offset);

        if (buf_dst == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        buf_dst_offset += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:buf_dst
              destinationOffset:buf_dst_offset
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buf_dst = [ctx->device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        size_t buf_src_offset = 0;
        id<MTLBuffer> buf_src = ggml_metal_get_buffer(tensor, &buf_src_offset);

        if (buf_src == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        buf_src_offset += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:buf_src_offset
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

struct ggml_metal_encode_context {
    id<MTLComputeCommandEncoder> encoder;

    ggml_metal_t ctx;

    ggml_mem_ranges_t mem_ranges;
};

static bool ggml_metal_encode_concurrency_reset(struct ggml_metal_encode_context * ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    [ctx->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_encode_concurrency_check(struct ggml_metal_encode_context * ctx, const struct ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_encode_concurrency_add(struct ggml_metal_encode_context * ctx, const struct ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

static int ggml_metal_encode_node(struct ggml_metal_encode_context * ctx_enc, int idx, int idx_end) {
    id<MTLComputeCommandEncoder> encoder = ctx_enc->encoder;

    ggml_metal_t ctx = ctx_enc->ctx;

    struct ggml_cgraph * gf = ctx->gf;

    enum ggml_op ops[8];

    struct ggml_tensor ** nodes = ggml_graph_nodes(gf) + idx;
    struct ggml_tensor *  node  = nodes[0];

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    struct ggml_tensor * src0 = node->src[0];
    struct ggml_tensor * src1 = node->src[1];
    struct ggml_tensor * src2 = node->src[2];
    struct ggml_tensor * dst  = node;

    if (ggml_is_empty(dst)) {
        return 1;
    }

    switch (dst->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
            } return 1;
        default:
            {
            } break;
    }

    if (!ggml_metal_device_supports_op(ctx->ctx_dev, dst)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(dst));
        GGML_ABORT("unsupported op");
    }

    const int64_t  ne00 = src0 ? src0->ne[0] : 0;
    const int64_t  ne01 = src0 ? src0->ne[1] : 0;
    const int64_t  ne02 = src0 ? src0->ne[2] : 0;
    const int64_t  ne03 = src0 ? src0->ne[3] : 0;

    const uint64_t nb00 = src0 ? src0->nb[0] : 0;
    const uint64_t nb01 = src0 ? src0->nb[1] : 0;
    const uint64_t nb02 = src0 ? src0->nb[2] : 0;
    const uint64_t nb03 = src0 ? src0->nb[3] : 0;

    const int64_t  ne10 = src1 ? src1->ne[0] : 0;
    const int64_t  ne11 = src1 ? src1->ne[1] : 0;
    const int64_t  ne12 = src1 ? src1->ne[2] : 0;
    const int64_t  ne13 = src1 ? src1->ne[3] : 0;

    const uint64_t nb10 = src1 ? src1->nb[0] : 0;
    const uint64_t nb11 = src1 ? src1->nb[1] : 0;
    const uint64_t nb12 = src1 ? src1->nb[2] : 0;
    const uint64_t nb13 = src1 ? src1->nb[3] : 0;

    const int64_t  ne20 = src2 ? src2->ne[0] : 0;
    const int64_t  ne21 = src2 ? src2->ne[1] : 0;
    const int64_t  ne22 = src2 ? src2->ne[2] : 0; GGML_UNUSED(ne22);
    const int64_t  ne23 = src2 ? src2->ne[3] : 0; GGML_UNUSED(ne23);

    const uint64_t nb20 = src2 ? src2->nb[0] : 0; GGML_UNUSED(nb20);
    const uint64_t nb21 = src2 ? src2->nb[1] : 0;
    const uint64_t nb22 = src2 ? src2->nb[2] : 0;
    const uint64_t nb23 = src2 ? src2->nb[3] : 0; GGML_UNUSED(nb23);

    const int64_t  ne0  =  dst ?  dst->ne[0] : 0;
    const int64_t  ne1  =  dst ?  dst->ne[1] : 0;
    const int64_t  ne2  =  dst ?  dst->ne[2] : 0;
    const int64_t  ne3  =  dst ?  dst->ne[3] : 0;

    const uint64_t nb0  =  dst ?  dst->nb[0] : 0;
    const uint64_t nb1  =  dst ?  dst->nb[1] : 0;
    const uint64_t nb2  =  dst ?  dst->nb[2] : 0;
    const uint64_t nb3  =  dst ?  dst->nb[3] : 0;

    size_t offs_src[GGML_MAX_SRC];

    id<MTLBuffer> id_src[GGML_MAX_SRC];

    enum ggml_type srct[GGML_MAX_SRC];

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        offs_src[i] = 0;
        id_src[i] = node->src[i] ? ggml_metal_get_buffer(node->src[i], &offs_src[i]) : nil;
        srct[i]   = node->src[i] ? node->src[i]->type : GGML_TYPE_COUNT;
    }

    // TODO: tmp shorthands - remove
    size_t offs_src0 = offs_src[0];
    size_t offs_src1 = offs_src[1];
    size_t offs_src2 = offs_src[2];

    id<MTLBuffer> id_src0 = id_src[0];
    id<MTLBuffer> id_src1 = id_src[1];
    id<MTLBuffer> id_src2 = id_src[2];

    const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
    const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
    const enum ggml_type src2t = src2 ? src2->type : GGML_TYPE_COUNT;
    const enum ggml_type dstt  = dst  ? dst->type  : GGML_TYPE_COUNT;

    size_t offs_dst = 0;

    id<MTLBuffer> id_dst = dst ? ggml_metal_get_buffer(dst, &offs_dst) : nil;

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        const bool is_concurrent = ggml_metal_encode_concurrency_check(ctx_enc, node);

        if (!is_concurrent) {
            ggml_metal_encode_concurrency_reset(ctx_enc);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(dst->op), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            if (src0) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src0t), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(src0), src0->name);
            }
            if (src1) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src1t), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(src1), src1->name);
            }
            if (dst) {
                GGML_LOG_DEBUG("%s: dst  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(dstt), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        dst->name);
            }
        }
    }

    struct ggml_metal_device_props props_dev = ggml_metal_device_get_props(ctx->ctx_dev);

    switch (dst->op) {
        case GGML_OP_CONCAT:
            {
                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CONCAT].pipeline;

                const int32_t dim = ((const int32_t *) dst->op_params)[0];

                ggml_metal_kargs_concat args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.ne12 =*/ ne12,
                    /*.ne13 =*/ ne13,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                    /*.dim  =*/ dim,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                const int nth = MIN(1024, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                GGML_ASSERT(src0t == GGML_TYPE_F32);
                GGML_ASSERT(src1t == GGML_TYPE_F32);

                GGML_ASSERT(ggml_is_contiguous_rows(src0));
                GGML_ASSERT(ggml_is_contiguous_rows(src1));

                const size_t offs = 0;

                bool bcast_row = false;

                ggml_metal_kargs_bin args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.ne12 =*/ ne12,
                    /*.ne13 =*/ ne13,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                    /*.offs =*/ offs,
                    /*.o1   =*/ { offs_src1 },
                };

                // c[0] = add(a,    b[0])
                // c[1] = add(c[0], b[1])
                // c[2] = add(c[1], b[2])
                // ...
                if (ctx->use_fusion) {
                    ops[0] = GGML_OP_ADD;
                    ops[1] = GGML_OP_ADD;
                    ops[2] = GGML_OP_ADD;
                    ops[3] = GGML_OP_ADD;
                    ops[4] = GGML_OP_ADD;
                    ops[5] = GGML_OP_ADD;
                    ops[6] = GGML_OP_ADD;
                    ops[7] = GGML_OP_ADD;

                    size_t offs_fuse;
                    id<MTLBuffer> id_fuse;

                    // note: in metal, we sometimes encode the graph in parallel so we have to avoid fusing nodes
                    //       across splits. idx_end indicates the last node in the current split
                    for (n_fuse = 0; n_fuse <= 6 && idx + n_fuse + 1 < idx_end; ++n_fuse) {
                        if (!ggml_can_fuse(gf, idx + n_fuse, ops + n_fuse, 2)) {
                            break;
                        }

                        if (nodes[n_fuse] != nodes[n_fuse + 1]->src[0]) {
                            break;
                        }

                        // b[0] === b[1] === ...
                        if (!ggml_are_same_layout(nodes[n_fuse]->src[1], nodes[n_fuse + 1]->src[1])) {
                            break;
                        }

                        // only fuse nodes if src1 is in the same Metal buffer
                        id_fuse = ggml_metal_get_buffer(nodes[n_fuse + 1]->src[1], &offs_fuse);
                        if (id_fuse != id_src1) {
                            break;
                        }

                        ctx->fuse_cnt[nodes[n_fuse + 1]->op]++;

                        args.o1[n_fuse + 1] = offs_fuse;
                    }

                    ++n_fuse;

                    if (ctx->debug_fusion > 1 && n_fuse > 1) {
                        GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
                    }
                }

                id<MTLComputePipelineState> pipeline = nil;

                if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
                    GGML_ASSERT(ggml_is_contiguous(src0));

                    // src1 is a row
                    GGML_ASSERT(ne11 == 1);

                    pipeline = ggml_metal_op_bin_get_pipeline(ctx, dst->op, n_fuse, true);

                    bcast_row = true;
                } else {
                    pipeline = ggml_metal_op_bin_get_pipeline(ctx, dst->op, n_fuse, false);
                }

                if (n_fuse > 1) {
                    id_dst = ggml_metal_get_buffer(nodes[n_fuse - 1], &offs_dst);

                    for (int i = 1; i < n_fuse; ++i) {
                        if (!ggml_metal_encode_concurrency_check(ctx_enc, nodes[i])) {
                            ggml_metal_encode_concurrency_reset(ctx_enc);

                            break;
                        }
                    }
                }

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_src1 offset:0         atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                if (bcast_row) {
                    const int64_t n = ggml_nelements(dst)/4;

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } else {
                    int nth = 32;

                    while (16*nth < ne0 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                        nth *= 2;
                    }

                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                }
            } break;
        case GGML_OP_ADD_ID:
            {
                GGML_ASSERT(src0t == GGML_TYPE_F32);
                GGML_ASSERT(src1t == GGML_TYPE_F32);
                GGML_ASSERT(src2t == GGML_TYPE_I32);
                GGML_ASSERT(dstt  == GGML_TYPE_F32);

                GGML_ASSERT(ggml_is_contiguous_rows(src0));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ADD_ID].pipeline;

                ggml_metal_kargs_add_id args = {
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb11 =*/ nb11,
                    /*.nb21 =*/ nb21,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                [encoder setBuffer:id_src2 offset:offs_src2 atIndex:3];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:4];

                const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne00);

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_REPEAT:
            {
                id<MTLComputePipelineState> pipeline;

                switch (src0t) {
                    case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_REPEAT_F32].pipeline; break;
                    case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_REPEAT_F16].pipeline; break;
                    case GGML_TYPE_I32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_REPEAT_I32].pipeline; break;
                    case GGML_TYPE_I16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_REPEAT_I16].pipeline; break;
                    default: GGML_ABORT("fatal error");
                }

                ggml_metal_kargs_repeat args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_ACC:
            {
                GGML_ASSERT(src0t == GGML_TYPE_F32);
                GGML_ASSERT(src1t == GGML_TYPE_F32);
                GGML_ASSERT(dstt  == GGML_TYPE_F32);

                GGML_ASSERT(ggml_is_contiguous(src0));
                GGML_ASSERT(ggml_is_contiguous(src1));

                const size_t pnb1 = ((const int32_t *) dst->op_params)[0];
                const size_t pnb2 = ((const int32_t *) dst->op_params)[1];
                const size_t pnb3 = ((const int32_t *) dst->op_params)[2];
                const size_t offs = ((const int32_t *) dst->op_params)[3];

                const bool inplace = (bool) ((const int32_t *) dst->op_params)[4];

                if (!inplace) {
                    // run a separete kernel to cpy src->dst
                    // not sure how to avoid this
                    // TODO: make a simpler cpy_bytes kernel

                    const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].pipeline;

                    ggml_metal_kargs_cpy args = {
                        /*.ne00 =*/ ne00,
                        /*.ne01 =*/ ne01,
                        /*.ne02 =*/ ne02,
                        /*.ne03 =*/ ne03,
                        /*.nb00 =*/ nb00,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.nb03 =*/ nb03,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                        /*.ne2  =*/ ne2,
                        /*.ne3  =*/ ne3,
                        /*.nb0  =*/ nb0,
                        /*.nb1  =*/ nb1,
                        /*.nb2  =*/ nb2,
                        /*.nb3  =*/ nb3,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                    const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne00);

                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];

                    ggml_metal_encode_concurrency_reset(ctx_enc);
                }

                ggml_metal_kargs_bin args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ pnb1,
                    /*.nb02 =*/ pnb2,
                    /*.nb03 =*/ pnb3,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.ne12 =*/ ne12,
                    /*.ne13 =*/ ne13,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ pnb1,
                    /*.nb2  =*/ pnb2,
                    /*.nb3  =*/ pnb3,
                    /*.offs =*/ offs,
                    /*.o1   =*/ { offs_src1},
                };

                //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ADD].pipeline;
                const id<MTLComputePipelineState> pipeline = ggml_metal_op_bin_get_pipeline(ctx, GGML_OP_ADD, 1, false);

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_src1 offset:0         atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne00);

                [encoder dispatchThreadgroups:MTLSizeMake(ne11, ne12, ne13) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_SCALE:
            {
                GGML_ASSERT(ggml_is_contiguous(src0));

                float scale;
                float bias;
                memcpy(&scale, ((const int32_t *) dst->op_params) + 0, sizeof(float));
                memcpy(&bias,  ((const int32_t *) dst->op_params) + 1, sizeof(float));

                int64_t n = ggml_nelements(dst);

                id<MTLComputePipelineState> pipeline = nil;

                if (n % 4 == 0) {
                    n /= 4;
                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SCALE_4].pipeline;
                } else {
                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SCALE].pipeline;
                }

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0   offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst    offset:offs_dst  atIndex:1];
                [encoder setBytes:&scale length:sizeof(scale) atIndex:2];
                [encoder setBytes:&bias  length:sizeof(bias)  atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_CLAMP:
            {
                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CLAMP].pipeline;

                float min;
                float max;
                memcpy(&min, ((const int32_t *) dst->op_params) + 0, sizeof(float));
                memcpy(&max, ((const int32_t *) dst->op_params) + 1, sizeof(float));

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&min   length:sizeof(min) atIndex:2];
                [encoder setBytes:&max   length:sizeof(max) atIndex:3];

                const int64_t n = ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                // we are not taking into account the strides, so for now require contiguous tensors
                GGML_ASSERT(ggml_is_contiguous(src0));

                case GGML_UNARY_OP_TANH:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_TANH].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_RELU:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_RELU].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_SIGMOID:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SIGMOID].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_GELU:
                {
                    int64_t n = ggml_nelements(dst);

                    id<MTLComputePipelineState> pipeline = nil;

                    if (n % 4 == 0) {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GELU_4].pipeline;
                        n /= 4;
                    } else {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GELU].pipeline;
                    }

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_GELU_ERF:
                {
                    int64_t n = ggml_nelements(dst);

                    id<MTLComputePipelineState> pipeline = nil;

                    if (n % 4 == 0) {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GELU_ERF_4].pipeline;
                        n /= 4;
                    } else {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GELU_ERF].pipeline;
                    }

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_GELU_QUICK:
                {
                    int64_t n = ggml_nelements(dst);

                    id<MTLComputePipelineState> pipeline = nil;

                    if (n % 4 == 0) {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GELU_QUICK_4].pipeline;
                        n /= 4;
                    } else {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GELU_QUICK].pipeline;
                    }

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_SILU:
                {
                    int64_t n = ggml_nelements(dst);

                    id<MTLComputePipelineState> pipeline = nil;

                    if (n % 4 == 0) {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SILU_4].pipeline;
                        n /= 4;
                    } else {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SILU].pipeline;
                    }

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_ELU:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ELU].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_NEG:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_NEG].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_ABS:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ABS].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_SGN:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SGN].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_STEP:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_STEP].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_HARDSWISH:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_HARDSWISH].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_HARDSIGMOID:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_HARDSIGMOID].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_UNARY_OP_EXP:
                {
                    id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_EXP].pipeline;

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                default:
                {
                    GGML_LOG_WARN("%s: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(dst->op));
                    GGML_ABORT("fatal error");
                }
            } break;
        case GGML_OP_GLU:
            {
                GGML_ASSERT(ggml_is_contiguous_1(src0));

                if (src1) {
                    GGML_ASSERT(ggml_are_same_shape(src0, src1));
                }

                id<MTLComputePipelineState> pipeline = nil;

                switch (ggml_get_glu_op(node)) {
                    case GGML_GLU_OP_REGLU:
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_REGLU].pipeline;
                        break;
                    case GGML_GLU_OP_GEGLU:
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GEGLU].pipeline;
                        break;
                    case GGML_GLU_OP_SWIGLU:
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SWIGLU].pipeline;
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SWIGLU_OAI].pipeline;
                        break;
                    case GGML_GLU_OP_GEGLU_ERF:
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GEGLU_ERF].pipeline;
                        break;
                    case GGML_GLU_OP_GEGLU_QUICK:
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GEGLU_QUICK].pipeline;
                        break;
                    default:
                        GGML_ABORT("fatal error");
                }

                const int32_t swp = ggml_get_op_params_i32(dst, 1);
                const float alpha = ggml_get_op_params_f32(dst, 2);
                const float limit = ggml_get_op_params_f32(dst, 3);

                const int32_t i00 = swp ? ne0 : 0;
                const int32_t i10 = swp ? 0 : ne0;

                ggml_metal_kargs_glu args = {
                    /*.ne00 =*/ ne00,
                    /*.nb01 =*/ nb01,
                    /*.ne10 =*/ src1 ? ne10 : ne00,
                    /*.nb11 =*/ src1 ? nb11 : nb01,
                    /*.ne0  =*/ ne0,
                    /*.nb1  =*/ nb1,
                    /*.i00  =*/ src1 ? 0 : i00,
                    /*.i10  =*/ src1 ? 0 : i10,
                    /*.alpha=*/ alpha,
                    /*.limit=*/ limit
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                if (src1) {
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                } else {
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                }
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                [encoder setBytes:&args length:sizeof(args) atIndex:3];

                const int64_t nrows = ggml_nrows(src0);

                const int32_t nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne00/2);

                [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_SQR:
            {
                GGML_ASSERT(ggml_is_contiguous(src0));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SQR].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                const int64_t n = ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_SQRT:
            {
                GGML_ASSERT(ggml_is_contiguous(src0));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SQRT].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                const int64_t n = ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_SIN:
            {
                GGML_ASSERT(ggml_is_contiguous(src0));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SIN].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                const int64_t n = ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_COS:
            {
                GGML_ASSERT(ggml_is_contiguous(src0));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_COS].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                const int64_t n = ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));

                id<MTLComputePipelineState> pipeline = nil;

                switch (dst->op) {
                    case GGML_OP_SUM_ROWS:
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SUM_ROWS].pipeline;
                        break;
                    case GGML_OP_MEAN:
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MEAN].pipeline;
                        break;
                    default:
                        GGML_ABORT("fatal error");
                }

                int nth = 32; // SIMD width

                while (nth < ne00 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                    nth *= 2;
                }

                nth = MIN(nth, (int) pipeline.maxTotalThreadsPerThreadgroup);
                nth = MIN(nth, ne00);

                ggml_metal_kargs_sum_rows args = {
                   /*.ne00 =*/ ne00,
                   /*.ne01 =*/ ne01,
                   /*.ne02 =*/ ne02,
                   /*.ne03 =*/ ne03,
                   /*.nb00 =*/ nb00,
                   /*.nb01 =*/ nb01,
                   /*.nb02 =*/ nb02,
                   /*.nb03 =*/ nb03,
                   /*.ne10 =*/ ne10,
                   /*.ne11 =*/ ne11,
                   /*.ne12 =*/ ne12,
                   /*.ne13 =*/ ne13,
                   /*.nb10 =*/ nb10,
                   /*.nb11 =*/ nb11,
                   /*.nb12 =*/ nb12,
                   /*.nb13 =*/ nb13,
                   /*.ne0  =*/ ne0,
                   /*.ne1  =*/ ne1,
                   /*.ne2  =*/ ne2,
                   /*.ne3  =*/ ne3,
                   /*.nb0  =*/ nb0,
                   /*.nb1  =*/ nb1,
                   /*.nb2  =*/ nb2,
                   /*.nb3  =*/ nb3,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_SOFT_MAX:
            {
                GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

                int nth = 32; // SIMD width

                id<MTLComputePipelineState> pipeline = nil;

                const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

                if (ne00%4 == 0) {
                    while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
                        nth *= 2;
                    }
                    if (use_f16) {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F16_4].pipeline;
                    } else {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F32_4].pipeline;
                    }
                } else {
                    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
                        nth *= 2;
                    }
                    if (use_f16) {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F16].pipeline;
                    } else {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SOFT_MAX_F32].pipeline;
                    }
                }

                float scale;
                float max_bias;

                memcpy(&scale,    ((const int32_t *) dst->op_params) + 0, sizeof(scale));
                memcpy(&max_bias, ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));

                const uint32_t n_head      = src0->ne[2];
                const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

                const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
                const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

                id<MTLBuffer> h_src0 = id_src0;

                // softmax

                ggml_metal_kargs_soft_max args = {
                    /*.ne00        =*/ ne00,
                    /*.ne01        =*/ ne01,
                    /*.ne02        =*/ ne02,
                    /*.nb01        =*/ nb01,
                    /*.nb02        =*/ nb02,
                    /*.nb03        =*/ nb03,
                    /*.ne11        =*/ ne11,
                    /*.ne12        =*/ ne12,
                    /*.ne13        =*/ ne13,
                    /*.nb11        =*/ nb11,
                    /*.nb12        =*/ nb12,
                    /*.nb13        =*/ nb13,
                    /*.nb1         =*/ nb1,
                    /*.nb2         =*/ nb2,
                    /*.nb3         =*/ nb3,
                    /*.scale       =*/ scale,
                    /*.max_bias    =*/ max_bias,
                    /*.m0          =*/ m0,
                    /*.m1          =*/ m1,
                    /*.n_head_log2 =*/ n_head_log2,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:h_src0 offset:offs_src0      atIndex:0];
                if (id_src1) {
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                } else {
                    [encoder setBuffer:h_src0 offset:offs_src0  atIndex:1];
                }
                if (id_src2) {
                    [encoder setBuffer:id_src2 offset:offs_src2 atIndex:2];
                } else {
                    [encoder setBuffer:h_src0 offset:offs_src0  atIndex:2];
                }
                [encoder setBuffer:id_dst offset:offs_dst       atIndex:3];
                [encoder setBytes:&args   length:sizeof(args)   atIndex:4];

                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                const int n_past = ((const int32_t *)(dst->op_params))[0];

                id<MTLComputePipelineState> pipeline = nil;

                if (ne00%8 == 0) {
                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_DIAG_MASK_INF_8].pipeline;
                } else {
                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_DIAG_MASK_INF].pipeline;
                }

                ggml_metal_kargs_diag_mask_inf args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.n_past =*/ n_past,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args  length:sizeof(args) atIndex:2];

                if (ne00%8 == 0) {
                    [encoder dispatchThreadgroups:MTLSizeMake(ne00*ne01*ne02/8, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                }
                else {
                    [encoder dispatchThreadgroups:MTLSizeMake(ne00, ne01, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                }
            } break;
        case GGML_OP_SSM_CONV:
            {
                GGML_ASSERT(src0t == GGML_TYPE_F32);
                GGML_ASSERT(src1t == GGML_TYPE_F32);

                GGML_ASSERT(ggml_is_contiguous(src0));
                GGML_ASSERT(ggml_is_contiguous(src1));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SSM_CONV_F32].pipeline;

                ggml_metal_kargs_ssm_conv args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0    atIndex:0];
                [encoder setBuffer:id_src1 offset:offs_src1    atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst     atIndex:2];
                [encoder setBytes:&args    length:sizeof(args) atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne1, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_SSM_SCAN:
            {
                struct ggml_tensor * src3 = node->src[3];
                struct ggml_tensor * src4 = node->src[4];
                struct ggml_tensor * src5 = node->src[5];
                struct ggml_tensor * src6 = node->src[6];

                GGML_ASSERT(src3);
                GGML_ASSERT(src4);
                GGML_ASSERT(src5);
                GGML_ASSERT(src6);

                size_t offs_src3 = 0;
                size_t offs_src4 = 0;
                size_t offs_src5 = 0;
                size_t offs_src6 = 0;

                id<MTLBuffer> id_src3 = src3 ? ggml_metal_get_buffer(src3, &offs_src3) : nil;
                id<MTLBuffer> id_src4 = src4 ? ggml_metal_get_buffer(src4, &offs_src4) : nil;
                id<MTLBuffer> id_src5 = src5 ? ggml_metal_get_buffer(src5, &offs_src5) : nil;
                id<MTLBuffer> id_src6 = src6 ? ggml_metal_get_buffer(src6, &offs_src6) : nil;

                const int64_t  ne30 = src3->ne[0];
                const int64_t  ne31 = src3->ne[1]; GGML_UNUSED(ne31);

                const uint64_t nb30 = src3->nb[0]; GGML_UNUSED(nb30);
                const uint64_t nb31 = src3->nb[1];

                const int64_t  ne40 = src4->ne[0]; GGML_UNUSED(ne40);
                const int64_t  ne41 = src4->ne[1];
                const int64_t  ne42 = src4->ne[2]; GGML_UNUSED(ne42);
                const int64_t  ne43 = src4->ne[3]; GGML_UNUSED(ne43);

                const uint64_t nb40 = src4->nb[0]; GGML_UNUSED(nb40);
                const uint64_t nb41 = src4->nb[1];
                const uint64_t nb42 = src4->nb[2];
                const uint64_t nb43 = src4->nb[3];

                const int64_t  ne50 = src5->ne[0]; GGML_UNUSED(ne50);
                const int64_t  ne51 = src5->ne[1]; GGML_UNUSED(ne51);
                const int64_t  ne52 = src5->ne[2]; GGML_UNUSED(ne52);
                const int64_t  ne53 = src5->ne[3]; GGML_UNUSED(ne53);

                const uint64_t nb50 = src5->nb[0]; GGML_UNUSED(nb50);
                const uint64_t nb51 = src5->nb[1];
                const uint64_t nb52 = src5->nb[2];
                const uint64_t nb53 = src5->nb[3];

                const int64_t  ne60 = src6->ne[0]; GGML_UNUSED(ne60);

                const uint64_t nb60 = src6->nb[0]; GGML_UNUSED(nb60);

                const int64_t d_state      = ne00;
                const int64_t d_inner      = ne01;
                const int64_t n_head       = ne02;
                const int64_t n_group      = ne41;
                const int64_t n_seq_tokens = ne12;
                const int64_t n_seqs       = ne13;

                id<MTLComputePipelineState> pipeline = nil;

                if (ne30 == 1) {
                    // Mamba-2
                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SSM_SCAN_F32_GROUP].pipeline;
                } else {
                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SSM_SCAN_F32].pipeline;
                }

                ggml_metal_kargs_ssm_scan args = {
                    /*.d_state      =*/ d_state,
                    /*.d_inner      =*/ d_inner,
                    /*.n_head       =*/ n_head,
                    /*.n_group      =*/ n_group,
                    /*.n_seq_tokens =*/ n_seq_tokens,
                    /*.n_seqs       =*/ n_seqs,
                    /*.s_off        =*/ ggml_nelements(src1) * sizeof(float),
                    /*.nb01         =*/ nb01,
                    /*.nb02         =*/ nb02,
                    /*.nb03         =*/ nb03,
                    /*.nb11         =*/ nb11,
                    /*.nb12         =*/ nb12,
                    /*.nb13         =*/ nb13,
                    /*.nb21         =*/ nb21,
                    /*.nb22         =*/ nb22,
                    /*.nb31         =*/ nb31,
                    /*.nb41         =*/ nb41,
                    /*.nb42         =*/ nb42,
                    /*.nb43         =*/ nb43,
                    /*.nb51         =*/ nb51,
                    /*.nb52         =*/ nb52,
                    /*.nb53         =*/ nb53,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                [encoder setBuffer:id_src2 offset:offs_src2 atIndex:2];
                [encoder setBuffer:id_src3 offset:offs_src3 atIndex:3];
                [encoder setBuffer:id_src4 offset:offs_src4 atIndex:4];
                [encoder setBuffer:id_src5 offset:offs_src5 atIndex:5];
                [encoder setBuffer:id_src6 offset:offs_src6 atIndex:6];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:7];
                [encoder setBytes:&args    length:sizeof(args) atIndex:8];

                // One shared memory bucket for each simd group in the threadgroup
                // NOTE: Metal kernels require the buffer size to be multiple of 16 bytes
                //  https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
                if (d_state >= 32) {
                    GGML_ASSERT((int64_t)(d_state / 32) <= 32);
                    const int64_t shmem_size = 32;
                    GGML_ASSERT(d_state <= (int64_t)pipeline.maxTotalThreadsPerThreadgroup);
                    [encoder setThreadgroupMemoryLength:(shmem_size)*sizeof(float) atIndex:0];
                }

                if (ne30 == 1) {
                    // Mamba-2
                    [encoder dispatchThreadgroups:MTLSizeMake(d_inner, n_head, n_seqs) threadsPerThreadgroup:MTLSizeMake(d_state, 1, 1)];
                } else {
                    GGML_ASSERT(d_inner == 1);
                    [encoder dispatchThreadgroups:MTLSizeMake(n_head, n_seqs, 1) threadsPerThreadgroup:MTLSizeMake(d_state, 1, 1)];
                }
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                const int64_t B = dst->src[5]->ne[1];
                const int64_t T = dst->src[0]->ne[2];
                const int64_t C = dst->ne[0];
                const int64_t H = dst->src[0]->ne[1];

                GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                size_t offs_src3 = 0;
                size_t offs_src4 = 0;
                size_t offs_src5 = 0;

                id<MTLBuffer> id_src3 = dst->src[3] ? ggml_metal_get_buffer(dst->src[3], &offs_src3) : nil;
                id<MTLBuffer> id_src4 = dst->src[4] ? ggml_metal_get_buffer(dst->src[4], &offs_src4) : nil;
                id<MTLBuffer> id_src5 = dst->src[5] ? ggml_metal_get_buffer(dst->src[5], &offs_src5) : nil;

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_RWKV_WKV6_F32].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                [encoder setBuffer:id_src2 offset:offs_src2 atIndex:2];
                [encoder setBuffer:id_src3 offset:offs_src3 atIndex:3];
                [encoder setBuffer:id_src4 offset:offs_src4 atIndex:4];
                [encoder setBuffer:id_src5 offset:offs_src5 atIndex:5];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:6];

                [encoder setBytes:&B length:sizeof(B) atIndex:7];
                [encoder setBytes:&T length:sizeof(T) atIndex:8];
                [encoder setBytes:&C length:sizeof(C) atIndex:9];
                [encoder setBytes:&H length:sizeof(H) atIndex:10];

                [encoder dispatchThreadgroups:MTLSizeMake(B * H, 1, 1) threadsPerThreadgroup:MTLSizeMake(C/ H, 1, 1)];
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t B = dst->src[6]->ne[1];
                const int64_t T = dst->src[0]->ne[2];
                const int64_t C = dst->ne[0];
                const int64_t H = dst->src[0]->ne[1];

                GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                size_t offs_src3 = 0;
                size_t offs_src4 = 0;
                size_t offs_src5 = 0;
                size_t offs_src6 = 0;

                id<MTLBuffer> id_src3 = dst->src[3] ? ggml_metal_get_buffer(dst->src[3], &offs_src3) : nil;
                id<MTLBuffer> id_src4 = dst->src[4] ? ggml_metal_get_buffer(dst->src[4], &offs_src4) : nil;
                id<MTLBuffer> id_src5 = dst->src[5] ? ggml_metal_get_buffer(dst->src[5], &offs_src5) : nil;
                id<MTLBuffer> id_src6 = dst->src[6] ? ggml_metal_get_buffer(dst->src[6], &offs_src6) : nil;

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_RWKV_WKV7_F32].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                [encoder setBuffer:id_src2 offset:offs_src2 atIndex:2];
                [encoder setBuffer:id_src3 offset:offs_src3 atIndex:3];
                [encoder setBuffer:id_src4 offset:offs_src4 atIndex:4];
                [encoder setBuffer:id_src5 offset:offs_src5 atIndex:5];
                [encoder setBuffer:id_src6 offset:offs_src6 atIndex:6];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:7];

                [encoder setBytes:&B length:sizeof(B) atIndex:8];
                [encoder setBytes:&T length:sizeof(T) atIndex:9];
                [encoder setBytes:&C length:sizeof(C) atIndex:10];
                [encoder setBytes:&H length:sizeof(H) atIndex:11];

                [encoder dispatchThreadgroups:MTLSizeMake(B * H, 1, 1) threadsPerThreadgroup:MTLSizeMake(C/ H, 1, 1)];
            } break;
        case GGML_OP_MUL_MAT:
            {
                GGML_ASSERT(ne00 == ne10);

                GGML_ASSERT(ne12 % ne02 == 0);
                GGML_ASSERT(ne13 % ne03 == 0);

                const uint32_t r2 = ne12/ne02;
                const uint32_t r3 = ne13/ne03;

                // find the break-even point where the matrix-matrix kernel becomes more efficient compared
                // to the matrix-vector kernel
                const int ne11_mm_min = 8;

                // first try to use small-batch mat-mv kernels
                // these should be efficient for BS [2, ~8]
                if (src1t == GGML_TYPE_F32 && (ne00%128 == 0) &&
                    (
                     (
                      (
                       src0t == GGML_TYPE_F32  || // TODO: helper function
                       src0t == GGML_TYPE_F16  ||
                       src0t == GGML_TYPE_Q4_0 ||
                       src0t == GGML_TYPE_Q4_1 ||
                       src0t == GGML_TYPE_Q5_0 ||
                       src0t == GGML_TYPE_Q5_1 ||
                       src0t == GGML_TYPE_Q8_0 ||
                       src0t == GGML_TYPE_MXFP4 ||
                       src0t == GGML_TYPE_IQ4_NL ||
                       false) && (ne11 >= 2 && ne11 <= 8)
                     ) ||
                     (
                      (
                       src0t == GGML_TYPE_Q4_K ||
                       src0t == GGML_TYPE_Q5_K ||
                       src0t == GGML_TYPE_Q6_K ||
                       false) && (ne11 >= 4 && ne11 <= 8)
                     )
                    )
                   ) {
                    // TODO: determine the optimal parameters based on grid utilization
                    //       I still don't know why we should not always use the maximum available threads:
                    //
                    //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
                    //
                    //       my current hypothesis is that the work grid is not evenly divisible for different nsg
                    //       values and there can be some tail effects when nsg is high. need to confirm this
                    //
                    const int nsg    = 2;                 // num simdgroups per threadgroup

                    // num threads along row per simdgroup
                    int nxpsg = 0;
                    if (ne00 % 256 == 0 && ne11 < 3) {
                        nxpsg = 16;
                    } else if (ne00 % 128 == 0) {
                        nxpsg = 8;
                    } else {
                        nxpsg = 4;
                    }

                    const int nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
                    const int r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
                          int r1ptg  = 4;                 // num src1 rows per threadgroup

                    // note: not sure how optimal are those across all different hardware. there might be someting cleverer
                    switch (ne11) {
                        case 2:
                            r1ptg = 2; break;
                        case 3:
                        case 6:
                            r1ptg = 3; break;
                        case 4:
                        case 7:
                        case 8:
                            r1ptg = 4; break;
                        case 5:
                            r1ptg = 5; break;
                    };

                    id<MTLComputePipelineState> pipeline = nil;

                    switch (src0->type) {
                        case GGML_TYPE_F32:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F32_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_F16:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_F16_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_Q4_0:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_0_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_Q4_1:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_1_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_Q5_0:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_0_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_Q5_1:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_1_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_Q8_0:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q8_0_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_MXFP4:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_MXFP4_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_Q4_K:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q4_K_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_Q5_K:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q5_K_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_Q6_K:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_Q6_K_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        case GGML_TYPE_IQ4_NL:
                            switch (r1ptg) {
                                case 2: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_2].pipeline; break;
                                case 3: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_3].pipeline; break;
                                case 4: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_4].pipeline; break;
                                case 5: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_5].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            } break;
                        default: GGML_ABORT("not implemented");
                    }

                    ggml_metal_kargs_mul_mv_ext args = {
                        /*.ne00  =*/ ne00,
                        /*.ne01  =*/ ne01,
                        /*.ne02  =*/ ne02,
                        /*.nb00  =*/ nb00,
                        /*.nb01  =*/ nb01,
                        /*.nb02  =*/ nb02,
                        /*.nb03  =*/ nb03,
                        /*.ne10  =*/ ne10,
                        /*.ne11  =*/ ne11,
                        /*.ne12  =*/ ne12,
                        /*.nb10  =*/ nb10,
                        /*.nb11  =*/ nb11,
                        /*.nb12  =*/ nb12,
                        /*.nb13  =*/ nb13,
                        /*.ne0   =*/ ne0,
                        /*.ne1   =*/ ne1,
                        /*.r2    =*/ r2,
                        /*.r3    =*/ r3,
                        /*.nsg   =*/ nsg,
                        /*.nxpsg =*/ nxpsg,
                        /*.r1ptg =*/ r1ptg,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                    //printf("ne01 = %lld nr0ptg = %d\n", ne01, nr0ptg);
                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + r0ptg - 1)/r0ptg, (ne11 + r1ptg - 1)/r1ptg, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                } else
                // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                if (props_dev.supports_gpu_family_apple7 &&
                    !ggml_is_transposed(src0) &&
                    !ggml_is_transposed(src1) &&
                    src1t == GGML_TYPE_F32 &&
                    ne00 % 32 == 0 && ne00 >= 64 &&
                    (ne11 > ne11_mm_min || (ggml_is_quantized(src0t) && ne12 > 1))) {
                    //printf("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

                    // some Metal matrix data types require aligned pointers
                    // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
                    switch (src0->type) {
                        case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
                        case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
                        case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
                        default: break;
                    }

                    id<MTLComputePipelineState> pipeline = nil;

                    switch (src0->type) {
                        case GGML_TYPE_F32:     pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_F32_F32    ].pipeline; break;
                        case GGML_TYPE_F16:     pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_F16_F32    ].pipeline; break;
                        case GGML_TYPE_BF16:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_BF16_F32   ].pipeline; break;
                        case GGML_TYPE_Q4_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_0_F32   ].pipeline; break;
                        case GGML_TYPE_Q4_1:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_1_F32   ].pipeline; break;
                        case GGML_TYPE_Q5_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_0_F32   ].pipeline; break;
                        case GGML_TYPE_Q5_1:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_1_F32   ].pipeline; break;
                        case GGML_TYPE_Q8_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q8_0_F32   ].pipeline; break;
                        case GGML_TYPE_MXFP4:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_MXFP4_F32  ].pipeline; break;
                        case GGML_TYPE_Q2_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q2_K_F32   ].pipeline; break;
                        case GGML_TYPE_Q3_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q3_K_F32   ].pipeline; break;
                        case GGML_TYPE_Q4_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q4_K_F32   ].pipeline; break;
                        case GGML_TYPE_Q5_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q5_K_F32   ].pipeline; break;
                        case GGML_TYPE_Q6_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_Q6_K_F32   ].pipeline; break;
                        case GGML_TYPE_IQ2_XXS: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_XXS_F32].pipeline; break;
                        case GGML_TYPE_IQ2_XS:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_XS_F32 ].pipeline; break;
                        case GGML_TYPE_IQ3_XXS: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ3_XXS_F32].pipeline; break;
                        case GGML_TYPE_IQ3_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ3_S_F32  ].pipeline; break;
                        case GGML_TYPE_IQ2_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ2_S_F32  ].pipeline; break;
                        case GGML_TYPE_IQ1_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ1_S_F32  ].pipeline; break;
                        case GGML_TYPE_IQ1_M:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ1_M_F32  ].pipeline; break;
                        case GGML_TYPE_IQ4_NL:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ4_NL_F32 ].pipeline; break;
                        case GGML_TYPE_IQ4_XS:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_IQ4_XS_F32 ].pipeline; break;
                        default: GGML_ABORT("MUL MAT-MAT not implemented");
                    }

                    ggml_metal_kargs_mul_mm args = {
                        /*.ne00 =*/ ne00,
                        /*.ne02 =*/ ne02,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.nb03 =*/ nb03,
                        /*.ne12 =*/ ne12,
                        /*.nb10 =*/ nb10,
                        /*.nb11 =*/ nb11,
                        /*.nb12 =*/ nb12,
                        /*.nb13 =*/ nb13,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                        /*.r2   =*/ r2,
                        /*.r3   =*/ r3,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args    length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0    atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1    atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst     atIndex:3];

                    [encoder setThreadgroupMemoryLength:8192 atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake((ne11 + 31)/32, (ne01 + 63)/64, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    id<MTLComputePipelineState> pipeline = nil;

                    int nsg = 0; // number of simdgroups
                    int nr0 = 0; // number of src0 rows per simdgroup
                    int nr1 = 1; // number of src1 rows per threadgroup

                    size_t smem = 0; // shared memory

                    // use custom matrix x vector kernel
                    switch (src0t) {
                        case GGML_TYPE_F32:
                            {
                                GGML_ASSERT(src1t == GGML_TYPE_F32);
                                nsg = 1;
                                nr0 = 1;
                                nr1 = 4;
                                if (ne00 == 4) {
                                    nr0 = 32;
                                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_F32_F32_C4].pipeline;
                                } else {
                                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_F32_F32].pipeline;
                                }
                            } break;
                        case GGML_TYPE_F16:
                            {
                                nsg = 1;
                                nr0 = 1;
                                if (src1t == GGML_TYPE_F32) {
                                    if (ne00 == 4) {
                                        nr0 = 32;
                                        nr1 = 4;
                                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_C4].pipeline;
                                    } else if (ne11 * ne12 < 4) {
                                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_1ROW].pipeline;
                                    } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32_L4].pipeline;
                                        nr1 = ne11;
                                    } else {
                                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F32].pipeline;
                                        nr1 = 4;
                                    }
                                } else {
                                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_F16_F16].pipeline;
                                    nr1 = 4;
                                }
                            } break;
                        case GGML_TYPE_BF16:
                            {
                                nsg = 1;
                                nr0 = 1;
                                if (src1t == GGML_TYPE_F32) {
                                    if (ne00 == 4) {
                                        nr0 = 32;
                                        nr1 = 4;
                                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_C4].pipeline;
                                    } else if (ne11 * ne12 < 4) {
                                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_1ROW].pipeline;
                                    } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32_L4].pipeline;
                                        nr1 = ne11;
                                    } else {
                                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_F32].pipeline;
                                        nr1 = 4;
                                    }
                                } else {
                                    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_BF16_BF16].pipeline;
                                    nr1 = 4;
                                }
                            } break;
                        case GGML_TYPE_Q4_0:
                            {
                                nsg = N_SG_Q4_0;
                                nr0 = N_R0_Q4_0;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_0_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q4_1:
                            {
                                nsg = N_SG_Q4_1;
                                nr0 = N_R0_Q4_1;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_1_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q5_0:
                            {
                                nsg = N_SG_Q5_0;
                                nr0 = N_R0_Q5_0;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_0_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q5_1:
                            {
                                nsg = N_SG_Q5_1;
                                nr0 = N_R0_Q5_1;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_1_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q8_0:
                            {
                                nsg = N_SG_Q8_0;
                                nr0 = N_R0_Q8_0;
                                smem = 32*sizeof(float)*N_R0_Q8_0;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q8_0_F32].pipeline;
                            } break;
                        case GGML_TYPE_MXFP4:
                            {
                                nsg = N_SG_MXFP4;
                                nr0 = N_R0_MXFP4;
                                smem = 32*sizeof(float);
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_MXFP4_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q2_K:
                            {
                                nsg = N_SG_Q2_K;
                                nr0 = N_R0_Q2_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q2_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q3_K:
                            {
                                nsg = N_SG_Q3_K;
                                nr0 = N_R0_Q3_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q3_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q4_K:
                            {
                                nsg = N_SG_Q4_K;
                                nr0 = N_R0_Q4_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q4_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q5_K:
                            {
                                nsg = N_SG_Q5_K;
                                nr0 = N_R0_Q5_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q5_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q6_K:
                            {
                                nsg = N_SG_Q6_K;
                                nr0 = N_R0_Q6_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_Q6_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ2_XXS:
                            {
                                nsg = N_SG_IQ2_XXS;
                                nr0 = N_R0_IQ2_XXS;
                                smem = 256*8+128;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_XXS_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ2_XS:
                            {
                                nsg = N_SG_IQ2_XS;
                                nr0 = N_R0_IQ2_XS;
                                smem = 512*8+128;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_XS_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ3_XXS:
                            {
                                nsg = N_SG_IQ3_XXS;
                                nr0 = N_R0_IQ3_XXS;
                                smem = 256*4+128;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ3_XXS_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ3_S:
                            {
                                nsg = N_SG_IQ3_S;
                                nr0 = N_R0_IQ3_S;
                                smem = 512*4;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ3_S_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ2_S:
                            {
                                nsg = N_SG_IQ2_S;
                                nr0 = N_R0_IQ2_S;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ2_S_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ1_S:
                            {
                                nsg = N_SG_IQ1_S;
                                nr0 = N_R0_IQ1_S;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ1_S_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ1_M:
                            {
                                nsg = N_SG_IQ1_M;
                                nr0 = N_R0_IQ1_M;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ1_M_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ4_NL:
                            {
                                nsg = N_SG_IQ4_NL;
                                nr0 = N_R0_IQ4_NL;
                                smem = 32*sizeof(float);
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ4_NL_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ4_XS:
                            {
                                nsg = N_SG_IQ4_XS;
                                nr0 = N_R0_IQ4_XS;
                                smem = 32*sizeof(float);
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_IQ4_XS_F32].pipeline;
                            } break;
                        default:
                            {
                                GGML_LOG_ERROR("Asserting on type %d\n", (int)src0t);
                                GGML_ABORT("not implemented");
                            }
                    };

                    ggml_metal_kargs_mul_mv args = {
                        /*.ne00 =*/ ne00,
                        /*.ne01 =*/ ne01,
                        /*.ne02 =*/ ne02,
                        /*.nb00 =*/ nb00,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.nb03 =*/ nb03,
                        /*.ne10 =*/ ne10,
                        /*.ne11 =*/ ne11,
                        /*.ne12 =*/ ne12,
                        /*.nb10 =*/ nb10,
                        /*.nb11 =*/ nb11,
                        /*.nb12 =*/ nb12,
                        /*.nb13 =*/ nb13,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                        /*.r2   =*/ r2,
                        /*.r3   =*/ r3,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                    if (smem > 0) {
                        [encoder setThreadgroupMemoryLength:smem atIndex:0];
                    }

                    if (src0t == GGML_TYPE_Q8_0) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nr0 - 1)/(nr0), (ne11 + nr1 - 1)/nr1, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                    } else {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nr0*nsg - 1)/(nr0*nsg), (ne11 + nr1 - 1)/nr1, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                    }
                }
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                // src2 = ids
                GGML_ASSERT(src2t == GGML_TYPE_I32);

                GGML_ASSERT(!ggml_is_transposed(src0));
                GGML_ASSERT(!ggml_is_transposed(src1));

                GGML_ASSERT(src1t == GGML_TYPE_F32);

                GGML_ASSERT(ne03 == 1);
                GGML_ASSERT(ne13 == 1);

                const uint32_t r2 = 1;
                const uint32_t r3 = 1;

                // find the break-even point where the matrix-matrix kernel becomes more efficient compared
                // to the matrix-vector kernel
                // ne20 = n_used_experts
                // ne21 = n_rows (batch size)
                const int ne21_mm_id_min = 32;

                // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                if (props_dev.supports_gpu_family_apple7 &&
                    ne00 % 32 == 0 && ne00 >= 64 &&
                    (ne21 >= ne21_mm_id_min)) {
                    GGML_ASSERT(ne00 % 4 == 0);

                    // some Metal matrix data types require aligned pointers
                    // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
                    switch (src0->type) {
                        case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
                        case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
                        case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
                        default: break;
                    }

                    // extra buffers for intermediate id mapping
                    size_t offs_tpe = offs_dst + ggml_nbytes(dst);
                    size_t offs_ids = offs_tpe + ggml_metal_op_mul_mat_id_extra_tpe(dst);

                    {
                        ggml_metal_kargs_mul_mm_id_map0 args = {
                            ne02,
                            ne10,
                            ne11, // n_expert_used (bcast)
                            nb11,
                            nb12,
                            ne21, // n_tokens
                            ne20, // n_expert_used
                            nb21,
                        };

                        id<MTLComputePipelineState> pipeline = nil;

                        pipeline = nil;

                        switch (ne20) {
                            case 1:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_1 ].pipeline; break;
                            case 2:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_2 ].pipeline; break;
                            case 4:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_4 ].pipeline; break;
                            case 6:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_6 ].pipeline; break;
                            case 8:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_8 ].pipeline; break;
                            case 10: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_10].pipeline; break;
                            case 16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MAP0_F16_NE20_16].pipeline; break;
                            default: GGML_ABORT("missing specialization for ne20 = %d", (int) ne20);
                        }

                        GGML_ASSERT(ne02 <= (int) pipeline.maxTotalThreadsPerThreadgroup);

                        const size_t smem = ne02*ne20*sizeof(uint16_t);

                        GGML_ASSERT(smem <= props_dev.max_theadgroup_memory_size);

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBytes:&args    length:sizeof(args) atIndex:0];
                        [encoder setBuffer:id_src2 offset:offs_src2    atIndex:1];
                        [encoder setBuffer:id_dst  offset:offs_tpe     atIndex:2];
                        [encoder setBuffer:id_dst  offset:offs_ids     atIndex:3];
                        [encoder setThreadgroupMemoryLength:smem atIndex:0];

                        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(ne02, 1, 1)];
                    }

                    // this barrier is always needed because the next kernel has to wait for the id maps to be computed
                    ggml_metal_encode_concurrency_reset(ctx_enc);

                    {
                        id<MTLComputePipelineState> pipeline = nil;

                        switch (src0->type) {
                            case GGML_TYPE_F32:     pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_F32_F16    ].pipeline; break;
                            case GGML_TYPE_F16:     pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_F16_F16    ].pipeline; break;
                            case GGML_TYPE_BF16:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_BF16_F16   ].pipeline; break;
                            case GGML_TYPE_Q4_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_0_F16   ].pipeline; break;
                            case GGML_TYPE_Q4_1:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_1_F16   ].pipeline; break;
                            case GGML_TYPE_Q5_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_0_F16   ].pipeline; break;
                            case GGML_TYPE_Q5_1:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_1_F16   ].pipeline; break;
                            case GGML_TYPE_Q8_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q8_0_F16   ].pipeline; break;
                            case GGML_TYPE_MXFP4:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_MXFP4_F16  ].pipeline; break;
                            case GGML_TYPE_Q2_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q2_K_F16   ].pipeline; break;
                            case GGML_TYPE_Q3_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q3_K_F16   ].pipeline; break;
                            case GGML_TYPE_Q4_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q4_K_F16   ].pipeline; break;
                            case GGML_TYPE_Q5_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q5_K_F16   ].pipeline; break;
                            case GGML_TYPE_Q6_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_Q6_K_F16   ].pipeline; break;
                            case GGML_TYPE_IQ2_XXS: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_XXS_F16].pipeline; break;
                            case GGML_TYPE_IQ2_XS:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_XS_F16 ].pipeline; break;
                            case GGML_TYPE_IQ3_XXS: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ3_XXS_F16].pipeline; break;
                            case GGML_TYPE_IQ3_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ3_S_F16  ].pipeline; break;
                            case GGML_TYPE_IQ2_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ2_S_F16  ].pipeline; break;
                            case GGML_TYPE_IQ1_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ1_S_F16  ].pipeline; break;
                            case GGML_TYPE_IQ1_M:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ1_M_F16  ].pipeline; break;
                            case GGML_TYPE_IQ4_NL:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ4_NL_F16 ].pipeline; break;
                            case GGML_TYPE_IQ4_XS:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MM_ID_IQ4_XS_F16 ].pipeline; break;
                            default: GGML_ABORT("MUL_MAT_ID not implemented");
                        }

                        ggml_metal_kargs_mul_mm_id args = {
                            /*.ne00  =*/ ne00,
                            /*.ne02  =*/ ne02,
                            /*.nb01  =*/ nb01,
                            /*.nb02  =*/ nb02,
                            /*.nb03  =*/ nb03,
                            /*.ne11  =*/ ne11, // n_expert_used (bcast)
                            /*.nb10  =*/ nb10,
                            /*.nb11  =*/ nb11,
                            /*.nb12  =*/ nb12,
                            /*.nb13  =*/ nb13,
                            /*.ne20  =*/ ne20, // n_expert_used
                            /*.ne21  =*/ ne21, // n_tokens
                            /*.ne0   =*/ ne0,
                            /*.ne1   =*/ ne1,
                            /*.r2    =*/ r2,
                            /*.r3    =*/ r3,
                        };

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBytes:&args    length:sizeof(args) atIndex:0];
                        [encoder setBuffer:id_src0 offset:offs_src0    atIndex:1];
                        [encoder setBuffer:id_src1 offset:offs_src1    atIndex:2];
                        [encoder setBuffer:id_dst  offset:offs_tpe     atIndex:3];
                        [encoder setBuffer:id_dst  offset:offs_ids     atIndex:4];
                        [encoder setBuffer:id_dst  offset:offs_dst     atIndex:5];

                        [encoder setThreadgroupMemoryLength:8192 atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne21 + 31)/32, (ne01 + 63)/64, ne02) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                } else {
                    id<MTLComputePipelineState> pipeline = nil;

                    int nsg = 0; // number of simdgroups
                    int nr0 = 0; // number of src0 rows per simdgroup
                    int nr1 = 1; // number of src1 rows per threadgroup

                    size_t smem = 0; // shared memory

                    // use custom matrix x vector kernel
                    switch (src0t) {
                        case GGML_TYPE_F32:
                            {
                                GGML_ASSERT(src1t == GGML_TYPE_F32);
                                nsg = 1;
                                nr0 = 1;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F32_F32].pipeline;
                            } break;
                        case GGML_TYPE_F16:
                            {
                                GGML_ASSERT(src1t == GGML_TYPE_F32);
                                nsg = 1;
                                nr0 = 1;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_F16_F32].pipeline;
                            } break;
                        case GGML_TYPE_BF16:
                            {
                                GGML_ASSERT(src1t == GGML_TYPE_F32);
                                nsg = 1;
                                nr0 = 1;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_BF16_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q4_0:
                            {
                                nsg = N_SG_Q4_0;
                                nr0 = N_R0_Q4_0;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_0_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q4_1:
                            {
                                nsg = N_SG_Q4_1;
                                nr0 = N_R0_Q4_1;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_1_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q5_0:
                            {
                                nsg = N_SG_Q5_0;
                                nr0 = N_R0_Q5_0;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_0_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q5_1:
                            {
                                nsg = N_SG_Q5_1;
                                nr0 = N_R0_Q5_1;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_1_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q8_0:
                            {
                                nsg = N_SG_Q8_0;
                                nr0 = N_R0_Q8_0;
                                smem = 32*sizeof(float)*N_R0_Q8_0;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q8_0_F32].pipeline;
                            } break;
                        case GGML_TYPE_MXFP4:
                            {
                                nsg = N_SG_MXFP4;
                                nr0 = N_R0_MXFP4;
                                smem = 32*sizeof(float);
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_MXFP4_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q2_K:
                            {
                                nsg = N_SG_Q2_K;
                                nr0 = N_R0_Q2_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q2_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q3_K:
                            {
                                nsg = N_SG_Q3_K;
                                nr0 = N_R0_Q3_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q3_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q4_K:
                            {
                                nsg = N_SG_Q4_K;
                                nr0 = N_R0_Q4_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q4_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q5_K:
                            {
                                nsg = N_SG_Q5_K;
                                nr0 = N_R0_Q5_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q5_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_Q6_K:
                            {
                                nsg = N_SG_Q6_K;
                                nr0 = N_R0_Q6_K;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_Q6_K_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ2_XXS:
                            {
                                nsg = N_SG_IQ2_XXS;
                                nr0 = N_R0_IQ2_XXS;
                                smem = 256*8+128;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_XXS_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ2_XS:
                            {
                                nsg = N_SG_IQ2_XS;
                                nr0 = N_R0_IQ2_XS;
                                smem = 512*8+128;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_XS_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ3_XXS:
                            {
                                nsg = N_SG_IQ3_XXS;
                                nr0 = N_R0_IQ3_XXS;
                                smem = 256*4+128;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ3_XXS_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ3_S:
                            {
                                nsg = N_SG_IQ3_S;
                                nr0 = N_R0_IQ3_S;
                                smem = 512*4;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ3_S_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ2_S:
                            {
                                nsg = N_SG_IQ2_S;
                                nr0 = N_R0_IQ2_S;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ2_S_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ1_S:
                            {
                                nsg = N_SG_IQ1_S;
                                nr0 = N_R0_IQ1_S;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ1_S_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ1_M:
                            {
                                nsg = N_SG_IQ1_M;
                                nr0 = N_R0_IQ1_M;
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ1_M_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ4_NL:
                            {
                                nsg = N_SG_IQ4_NL;
                                nr0 = N_R0_IQ4_NL;
                                smem = 32*sizeof(float);
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ4_NL_F32].pipeline;
                            } break;
                        case GGML_TYPE_IQ4_XS:
                            {
                                nsg = N_SG_IQ4_XS;
                                nr0 = N_R0_IQ4_XS;
                                smem = 32*sizeof(float);
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_MUL_MV_ID_IQ4_XS_F32].pipeline;
                            } break;
                        default:
                            {
                                GGML_LOG_ERROR("Asserting on type %d\n", (int)src2t);
                                GGML_ABORT("not implemented");
                            }
                    };

                    if (ggml_is_quantized(src0t)) {
                        GGML_ASSERT(ne00 >= nsg*nr0);
                    }

                    ggml_metal_kargs_mul_mv_id args = {
                        /*.nei0 =*/ ne20,
                        /*.nei1 =*/ ne21,
                        /*.nbi1 =*/ nb21,
                        /*.ne00 =*/ ne00,
                        /*.ne01 =*/ ne01,
                        /*.ne02 =*/ ne02,
                        /*.nb00 =*/ nb00,
                        /*.nb01 =*/ nb01,
                        /*.nb02 =*/ nb02,
                        /*.ne10 =*/ ne10,
                        /*.ne11 =*/ ne11,
                        /*.ne12 =*/ ne12,
                        /*.ne13 =*/ ne13,
                        /*.nb10 =*/ nb10,
                        /*.nb11 =*/ nb11,
                        /*.nb12 =*/ nb12,
                        /*.ne0  =*/ ne0,
                        /*.ne1  =*/ ne1,
                        /*.nb1  =*/ nb1,
                    };

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args) atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];
                    [encoder setBuffer:id_src2 offset:offs_src2 atIndex:4];

                    const int64_t _ne1 = 1;
                    const int64_t ne123 = ne20*ne21;

                    if (smem > 0) {
                        [encoder setThreadgroupMemoryLength:smem atIndex:0];
                    }

                    if (src0t == GGML_TYPE_Q8_0) {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                    } else {
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                    }
                }
            } break;
        case GGML_OP_GET_ROWS:
            {
                id<MTLComputePipelineState> pipeline = nil;

                switch (src0->type) {
                    case GGML_TYPE_F32:     pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_F32    ].pipeline; break;
                    case GGML_TYPE_F16:     pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_F16    ].pipeline; break;
                    case GGML_TYPE_BF16:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_BF16   ].pipeline; break;
                    case GGML_TYPE_Q4_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_0   ].pipeline; break;
                    case GGML_TYPE_Q4_1:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_1   ].pipeline; break;
                    case GGML_TYPE_Q5_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_0   ].pipeline; break;
                    case GGML_TYPE_Q5_1:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_1   ].pipeline; break;
                    case GGML_TYPE_Q8_0:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q8_0   ].pipeline; break;
                    case GGML_TYPE_MXFP4:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_MXFP4  ].pipeline; break;
                    case GGML_TYPE_Q2_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q2_K   ].pipeline; break;
                    case GGML_TYPE_Q3_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q3_K   ].pipeline; break;
                    case GGML_TYPE_Q4_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q4_K   ].pipeline; break;
                    case GGML_TYPE_Q5_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q5_K   ].pipeline; break;
                    case GGML_TYPE_Q6_K:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_Q6_K   ].pipeline; break;
                    case GGML_TYPE_IQ2_XXS: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_XXS].pipeline; break;
                    case GGML_TYPE_IQ2_XS:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_XS ].pipeline; break;
                    case GGML_TYPE_IQ3_XXS: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ3_XXS].pipeline; break;
                    case GGML_TYPE_IQ3_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ3_S  ].pipeline; break;
                    case GGML_TYPE_IQ2_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ2_S  ].pipeline; break;
                    case GGML_TYPE_IQ1_S:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ1_S  ].pipeline; break;
                    case GGML_TYPE_IQ1_M:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ1_M  ].pipeline; break;
                    case GGML_TYPE_IQ4_NL:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ4_NL ].pipeline; break;
                    case GGML_TYPE_IQ4_XS:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_IQ4_XS ].pipeline; break;
                    case GGML_TYPE_I32:     pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GET_ROWS_I32    ].pipeline; break;
                    default: GGML_ABORT("not implemented");
                }

                ggml_metal_kargs_get_rows args = {
                    /*.ne00 =*/ ne00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.ne10 =*/ ne10,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb1 =*/ nb1,
                    /*.nb2 =*/ nb2,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args    length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0    atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1    atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst     atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake(ne10, ne11, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } break;
        case GGML_OP_SET_ROWS:
            {
                id<MTLComputePipelineState> pipeline = nil;

                switch (dst->type) {
                    case GGML_TYPE_F32:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_F32   ].pipeline; break;
                    case GGML_TYPE_F16:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_F16   ].pipeline; break;
                    case GGML_TYPE_BF16:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_BF16  ].pipeline; break;
                    case GGML_TYPE_Q8_0:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q8_0  ].pipeline; break;
                    case GGML_TYPE_Q4_0:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q4_0  ].pipeline; break;
                    case GGML_TYPE_Q4_1:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q4_1  ].pipeline; break;
                    case GGML_TYPE_Q5_0:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q5_0  ].pipeline; break;
                    case GGML_TYPE_Q5_1:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_Q5_1  ].pipeline; break;
                    case GGML_TYPE_IQ4_NL: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_SET_ROWS_IQ4_NL].pipeline; break;
                    default: GGML_ABORT("not implemented");
                }

                const int32_t nk0 = ne0/ggml_blck_size(dst->type);

                int nth = 32; // SIMD width

                while (nth < nk0 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                    nth *= 2;
                }

                int nrptg = 1;
                if (nth > nk0) {
                    nrptg = (nth + nk0 - 1)/nk0;
                    nth   = nk0;

                    if (nrptg*nth > (int) pipeline.maxTotalThreadsPerThreadgroup) {
                        nrptg--;
                    }
                }

                nth = MIN(nth, nk0);

                ggml_metal_kargs_set_rows args = {
                    /*.nk0  =*/ nk0,
                    /*.ne01 =*/ ne01,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne11 =*/ ne11,
                    /*.ne12 =*/ ne12,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args    length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0    atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1    atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst     atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nrptg - 1)/nrptg, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, nrptg, 1)];
            } break;
        case GGML_OP_RMS_NORM:
            {
                GGML_ASSERT(ne00 % 4 == 0);
                GGML_ASSERT(ggml_is_contiguous_rows(src0));

                float eps;
                memcpy(&eps, dst->op_params, sizeof(float));

                ggml_metal_kargs_rms_norm args = {
                    /*.ne00   =*/ ne00,
                    /*.ne00_4 =*/ ne00/4,
                    /*.nb1    =*/ nb1,
                    /*.nb2    =*/ nb2,
                    /*.nb3    =*/ nb3,
                    /*.eps    =*/ eps,
                    /*.nef1   =*/ { ne01 },
                    /*.nef2   =*/ { ne02 },
                    /*.nef3   =*/ { ne03 },
                    /*.nbf1   =*/ { nb01 },
                    /*.nbf2   =*/ { nb02 },
                    /*.nbf3   =*/ { nb03 },
                };

                size_t offs_fuse[2] = { 0, 0 };
                id<MTLBuffer> id_fuse[2] = { id_src0, id_src0 };

                // d[0] = rms_norm(a)
                // d[1] = mul(d[0], b)
                // d[2] = add(d[1], c)
                if (ctx->use_fusion) {
                    ops[0] = GGML_OP_RMS_NORM;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = GGML_OP_ADD;

                    for (n_fuse = 0; n_fuse <= 1 && idx + n_fuse + 1 < idx_end; ++n_fuse) {
                        if (!ggml_can_fuse(gf, idx + n_fuse, ops + n_fuse, 2)) {
                            break;
                        }

                        if (nodes[n_fuse] != nodes[n_fuse + 1]->src[0]) {
                            break;
                        }

                        if (nodes[n_fuse + 1]->src[1]->ne[0] != node->ne[0]) {
                            break;
                        }

                        if (!ggml_is_contiguous_rows(nodes[n_fuse + 1]->src[1])) {
                            break;
                        }

                        if (nodes[n_fuse + 1]->type != GGML_TYPE_F32) {
                            break;
                        }

                        ctx->fuse_cnt[nodes[n_fuse + 1]->op]++;

                        id_fuse[n_fuse] = ggml_metal_get_buffer(nodes[n_fuse + 1]->src[1], &offs_fuse[n_fuse]);

                        args.nef1[n_fuse + 1] = nodes[n_fuse + 1]->src[1]->ne[1];
                        args.nef2[n_fuse + 1] = nodes[n_fuse + 1]->src[1]->ne[2];
                        args.nef3[n_fuse + 1] = nodes[n_fuse + 1]->src[1]->ne[3];

                        args.nbf1[n_fuse + 1] = nodes[n_fuse + 1]->src[1]->nb[1];
                        args.nbf2[n_fuse + 1] = nodes[n_fuse + 1]->src[1]->nb[2];
                        args.nbf3[n_fuse + 1] = nodes[n_fuse + 1]->src[1]->nb[3];
                    }

                    ++n_fuse;

                    if (ctx->debug_fusion > 1 && n_fuse > 1) {
                        if (n_fuse == 2) {
                            GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL\n", __func__);
                        }
                        if (n_fuse == 3) {
                            GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL + ADD\n", __func__);
                        }
                    }
                }

                if (n_fuse > 1) {
                    id_dst = ggml_metal_get_buffer(nodes[n_fuse - 1], &offs_dst);

                    for (int i = 1; i < n_fuse; ++i) {
                        if (!ggml_metal_encode_concurrency_check(ctx_enc, nodes[i])) {
                            ggml_metal_encode_concurrency_reset(ctx_enc);

                            break;
                        }
                    }
                }

                const id<MTLComputePipelineState> pipeline = ggml_metal_op_rms_norm_get_pipeline(ctx, node, n_fuse);

                int nth = 32; // SIMD width

                while (nth < ne00/4 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                    nth *= 2;
                }

                nth = MIN(nth, (int) pipeline.maxTotalThreadsPerThreadgroup);
                nth = MIN(nth, ne00/4);

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args)       atIndex:0];
                [encoder setBuffer:id_src0    offset:offs_src0    atIndex:1];
                [encoder setBuffer:id_fuse[0] offset:offs_fuse[0] atIndex:2];
                [encoder setBuffer:id_fuse[1] offset:offs_fuse[1] atIndex:3];
                [encoder setBuffer:id_dst     offset:offs_dst     atIndex:4];

                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_L2_NORM:
            {
                GGML_ASSERT(ne00 % 4 == 0);
                GGML_ASSERT(ggml_is_contiguous_1(src0));

                float eps;
                memcpy(&eps, dst->op_params, sizeof(float));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_L2_NORM].pipeline;

                int nth = 32; // SIMD width

                while (nth < ne00/4 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                    nth *= 2;
                }

                nth = MIN(nth, (int) pipeline.maxTotalThreadsPerThreadgroup);
                nth = MIN(nth, ne00/4);

                ggml_metal_kargs_l2_norm args = {
                    /*.ne00   =*/ ne00,
                    /*.ne00_4 =*/ ne00/4,
                    /*.nb01   =*/ nb01,
                    /*.eps    =*/ eps,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                const int64_t nrows = ggml_nrows(src0);

                [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_GROUP_NORM:
            {
                GGML_ASSERT(ggml_is_contiguous(src0));

                float eps;
                memcpy(&eps, dst->op_params + 1, sizeof(float));

                const int32_t n_groups = ((const int32_t *) dst->op_params)[0];

                int nth = 32; // SIMD width

                //while (nth < ne00/4 && nth < 1024) {
                //    nth *= 2;
                //}

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_GROUP_NORM].pipeline;

                ggml_metal_kargs_group_norm args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.n_groups =*/ n_groups,
                    /*.eps =*/ eps,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0  offset:offs_src0        atIndex:0];
                [encoder setBuffer:id_dst   offset:offs_dst         atIndex:1];
                [encoder setBytes:&args     length:sizeof(args)     atIndex:2];
                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                [encoder dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_NORM:
            {
                GGML_ASSERT(ne00 % 4 == 0);
                GGML_ASSERT(ggml_is_contiguous_1(src0));

                float eps;
                memcpy(&eps, dst->op_params, sizeof(float));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_NORM].pipeline;

                int nth = 32; // SIMD width

                while (nth < ne00/4 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                    nth *= 2;
                }

                nth = MIN(nth, (int) pipeline.maxTotalThreadsPerThreadgroup);
                nth = MIN(nth, ne00/4);

                ggml_metal_kargs_norm args = {
                    /*.ne00   =*/ ne00,
                    /*.ne00_4 =*/ ne00/4,
                    /*.nb01   =*/ nb01,
                    /*.eps    =*/ eps,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                const int64_t nrows = ggml_nrows(src0);

                [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_ROPE:
            {
                // make sure we have one or more position id(ne10) per token(ne02)
                GGML_ASSERT(ne10 % ne02 == 0);
                GGML_ASSERT(ne10 >= ne02);

                const int nth = MIN(1024, ne00);

                const int n_past     = ((const int32_t *) dst->op_params)[0];
                const int n_dims     = ((const int32_t *) dst->op_params)[1];
                const int mode       = ((const int32_t *) dst->op_params)[2];
                // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
                const int n_ctx_orig = ((const int32_t *) dst->op_params)[4];

                float freq_base;
                float freq_scale;
                float ext_factor;
                float attn_factor;
                float beta_fast;
                float beta_slow;

                memcpy(&freq_base,   (const int32_t *) dst->op_params +  5, sizeof(float));
                memcpy(&freq_scale,  (const int32_t *) dst->op_params +  6, sizeof(float));
                memcpy(&ext_factor,  (const int32_t *) dst->op_params +  7, sizeof(float));
                memcpy(&attn_factor, (const int32_t *) dst->op_params +  8, sizeof(float));
                memcpy(&beta_fast,   (const int32_t *) dst->op_params +  9, sizeof(float));
                memcpy(&beta_slow,   (const int32_t *) dst->op_params + 10, sizeof(float));

                const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
                const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
                const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

                // mrope
                const int sect_0 = ((const int32_t *) dst->op_params)[11];
                const int sect_1 = ((const int32_t *) dst->op_params)[12];
                const int sect_2 = ((const int32_t *) dst->op_params)[13];
                const int sect_3 = ((const int32_t *) dst->op_params)[14];

                id<MTLComputePipelineState> pipeline = nil;

                if (is_neox) {
                    switch (src0->type) {
                        case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ROPE_NEOX_F32].pipeline; break;
                        case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ROPE_NEOX_F16].pipeline; break;
                        default: GGML_ABORT("fatal error");
                    };
                } else if (is_mrope && !is_vision) {
                    GGML_ASSERT(ne10*4 >= ne02); // need at least 4 pos per token
                    switch (src0->type) {
                        case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ROPE_MULTI_F32].pipeline; break;
                        case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ROPE_MULTI_F16].pipeline; break;
                        default: GGML_ABORT("fatal error");
                    };
                } else if (is_vision) {
                    GGML_ASSERT(ne10*4 >= ne02); // need at least 4 pos per token
                    switch (src0->type) {
                        case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ROPE_VISION_F32].pipeline; break;
                        case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ROPE_VISION_F16].pipeline; break;
                        default: GGML_ABORT("fatal error");
                    };
                } else {
                    switch (src0->type) {
                        case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ROPE_NORM_F32].pipeline; break;
                        case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ROPE_NORM_F16].pipeline; break;
                        default: GGML_ABORT("fatal error");
                    };
                }

                ggml_metal_kargs_rope args = {
                    /*.ne00        =*/ ne00,
                    /*.ne01        =*/ ne01,
                    /*.ne02        =*/ ne02,
                    /*.ne03        =*/ ne03,
                    /*.nb00        =*/ nb00,
                    /*.nb01        =*/ nb01,
                    /*.nb02        =*/ nb02,
                    /*.nb03        =*/ nb03,
                    /*.ne0         =*/ ne0,
                    /*.ne1         =*/ ne1,
                    /*.ne2         =*/ ne2,
                    /*.ne3         =*/ ne3,
                    /*.nb0         =*/ nb0,
                    /*.nb1         =*/ nb1,
                    /*.nb2         =*/ nb2,
                    /*.nb3         =*/ nb3,
                    /*.n_past      =*/ n_past,
                    /*.n_dims      =*/ n_dims,
                    /*.n_ctx_orig  =*/ n_ctx_orig,
                    /*.freq_base   =*/ freq_base,
                    /*.freq_scale  =*/ freq_scale,
                    /*.ext_factor  =*/ ext_factor,
                    /*.attn_factor =*/ attn_factor,
                    /*.beta_fast   =*/ beta_fast,
                    /*.beta_slow   =*/ beta_slow,
                    /* sect_0      =*/ sect_0,
                    /* sect_1      =*/ sect_1,
                    /* sect_2      =*/ sect_2,
                    /* sect_3      =*/ sect_3,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args)     atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0     atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1     atIndex:2];
                if (id_src2 != nil) {
                    [encoder setBuffer:id_src2 offset:offs_src2 atIndex:3];
                } else {
                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:3];
                }
                [encoder setBuffer:id_dst  offset:offs_dst      atIndex:4];

                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_IM2COL:
            {
                GGML_ASSERT(ggml_is_contiguous(src1));
                GGML_ASSERT(src1->type == GGML_TYPE_F32);
                GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

                const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
                const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
                const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
                const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
                const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
                const int32_t d1 = ((const int32_t *)(dst->op_params))[5];

                const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

                const int32_t N  = src1->ne[is_2D ? 3 : 2];
                const int32_t IC = src1->ne[is_2D ? 2 : 1];
                const int32_t IH = is_2D ? src1->ne[1] : 1;
                const int32_t IW =         src1->ne[0];

                const int32_t KH = is_2D ? src0->ne[1] : 1;
                const int32_t KW =         src0->ne[0];

                const int32_t OH = is_2D ? dst->ne[2] : 1;
                const int32_t OW =         dst->ne[1];

                const int32_t CHW = IC * KH * KW;

                const uint64_t ofs0 = src1->nb[is_2D ? 3 : 2] / 4;
                const uint64_t ofs1 = src1->nb[is_2D ? 2 : 1] / 4;

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_IM2COL_F32].pipeline;

                const bool is_gt_mttpt = ((size_t)(N * KH * KW)) > pipeline.maxTotalThreadsPerThreadgroup;

                switch (dst->type) {
                    case GGML_TYPE_F32: {
                        pipeline = (is_gt_mttpt ?
                                    ctx->pipelines[GGML_METAL_PIPELINE_TYPE_IM2COL_EXT_F32].pipeline
                                    :
                                    ctx->pipelines[GGML_METAL_PIPELINE_TYPE_IM2COL_F32].pipeline);
                    } break;
                    case GGML_TYPE_F16: {
                        pipeline = (is_gt_mttpt ?
                                    ctx->pipelines[GGML_METAL_PIPELINE_TYPE_IM2COL_EXT_F16].pipeline
                                    :
                                    ctx->pipelines[GGML_METAL_PIPELINE_TYPE_IM2COL_F16].pipeline);
                    } break;
                    default: GGML_ABORT("fatal error");
                };

                ggml_metal_kargs_im2col args = {
                    /*.ofs0 =*/ ofs0,
                    /*.ofs1 =*/ ofs1,
                    /*.IW   =*/ IW,
                    /*.IH   =*/ IH,
                    /*.CHW  =*/ CHW,
                    /*.s0   =*/ s0,
                    /*.s1   =*/ s1,
                    /*.p0   =*/ p0,
                    /*.p1   =*/ p1,
                    /*.d0   =*/ d0,
                    /*.d1   =*/ d1,
                    /*.N    =*/ N,
                    /*.KH   =*/ KH,
                    /*.KW   =*/ KW,
                    /*.KHW  =*/ KH * KW,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src1 offset:offs_src1       atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst        atIndex:1];
                [encoder setBytes:&args length:sizeof(args)       atIndex:2];

                if (is_gt_mttpt) {
                    const uint64_t n_threads = MIN(pipeline.maxTotalThreadsPerThreadgroup, (uint64_t)N);

                    const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

                    [encoder dispatchThreadgroups:MTLSizeMake(quotient * CHW, OH, OW) threadsPerThreadgroup:MTLSizeMake(n_threads, 1, 1)];
                } else {
                    [encoder dispatchThreadgroups:MTLSizeMake(IC, OH, OW) threadsPerThreadgroup:MTLSizeMake(N, KH, KW)];
                }
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                GGML_ASSERT(ggml_is_contiguous(src0));
                GGML_ASSERT(ggml_is_contiguous(src1));
                GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
                GGML_ASSERT(src1->type == GGML_TYPE_F32);
                GGML_ASSERT( dst->type == GGML_TYPE_F32);

                const int32_t s0 = ((const int32_t *)(dst->op_params))[0];

                const int32_t IC = src1->ne[1];
                const int32_t IL = src1->ne[0];

                const int32_t K  = src0->ne[0];

                const int32_t OL = dst->ne[0];
                const int32_t OC = dst->ne[1];

                id<MTLComputePipelineState> pipeline;

                switch (src0->type) {
                    case GGML_TYPE_F32: {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CONV_TRANSPOSE_1D_F32_F32].pipeline;
                    } break;
                    case GGML_TYPE_F16: {
                        pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CONV_TRANSPOSE_1D_F16_F32].pipeline;
                    } break;
                    default: GGML_ABORT("fatal error");
                };

                ggml_metal_kargs_conv_transpose_1d args = {
                    /*.IC =*/ IC,
                    /*.IL =*/ IL,
                    /*.K  =*/ K,
                    /*.s0 =*/ s0,
                    /*.nb0 =*/ nb0,
                    /*.nb1 =*/ nb1,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0         atIndex:0];
                [encoder setBuffer:id_src1 offset:offs_src1         atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst          atIndex:2];
                [encoder setBytes:&args    length:sizeof(args)       atIndex:3];

                [encoder dispatchThreadgroups:MTLSizeMake(OL, OC, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_UPSCALE:
            {
                GGML_ASSERT(src0->type == GGML_TYPE_F32);

                const float sf0 = (float)ne0/src0->ne[0];
                const float sf1 = (float)ne1/src0->ne[1];
                const float sf2 = (float)ne2/src0->ne[2];
                const float sf3 = (float)ne3/src0->ne[3];

                const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_UPSCALE_F32].pipeline;

                ggml_metal_kargs_upscale args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0 =*/ ne0,
                    /*.ne1 =*/ ne1,
                    /*.ne2 =*/ ne2,
                    /*.ne3 =*/ ne3,
                    /*.nb0 =*/ nb0,
                    /*.nb1 =*/ nb1,
                    /*.nb2 =*/ nb2,
                    /*.nb3 =*/ nb3,
                    /*.sf0 =*/ sf0,
                    /*.sf1 =*/ sf1,
                    /*.sf2 =*/ sf2,
                    /*.sf3 =*/ sf3
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_PAD:
            {
                GGML_ASSERT(src0->type == GGML_TYPE_F32);

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_PAD_F32].pipeline;

                ggml_metal_kargs_pad args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0 =*/ ne0,
                    /*.ne1 =*/ ne1,
                    /*.ne2 =*/ ne2,
                    /*.ne3 =*/ ne3,
                    /*.nb0 =*/ nb0,
                    /*.nb1 =*/ nb1,
                    /*.nb2 =*/ nb2,
                    /*.nb3 =*/ nb3
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                const int nth = MIN(1024, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                GGML_ASSERT(src0->type == GGML_TYPE_F32);

                const int32_t p0 = ((const int32_t *)(dst->op_params))[0];
                const int32_t p1 = ((const int32_t *)(dst->op_params))[1];

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_PAD_REFLECT_1D_F32].pipeline;

                ggml_metal_kargs_pad_reflect_1d args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0 =*/ ne0,
                    /*.ne1 =*/ ne1,
                    /*.ne2 =*/ ne2,
                    /*.ne3 =*/ ne3,
                    /*.nb0 =*/ nb0,
                    /*.nb1 =*/ nb1,
                    /*.nb2 =*/ nb2,
                    /*.nb3 =*/ nb3,
                    /*.p0 =*/ p0,
                    /*.p1 =*/ p1
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                const int nth = MIN(1024, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_ARANGE:
            {
                GGML_ASSERT(dst->type == GGML_TYPE_F32);

                float start;
                float step;

                memcpy(&start, ((const int32_t *) dst->op_params) + 0, sizeof(float));
                memcpy(&step,  ((const int32_t *) dst->op_params) + 2, sizeof(float));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ARANGE_F32].pipeline;

                ggml_metal_kargs_arange args = {
                    /*.ne0 =*/ ne0,
                    /*.start =*/ start,
                    /*.step =*/ step
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:0];
                [encoder setBytes:&args length:sizeof(args) atIndex:1];

                const int nth = MIN(1024, ne0);

                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                GGML_ASSERT(src0->type == GGML_TYPE_F32);

                const int dim        = dst->op_params[0];
                const int max_period = dst->op_params[1];

                const int half = dim / 2;

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_TIMESTEP_EMBEDDING_F32].pipeline;

                ggml_metal_kargs_timestep_embedding args = {
                    /*.nb1 =*/ nb1,
                    /*.dim =*/ dim,
                    /*.max_period =*/ max_period
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];

                const int nth = MIN(1024, half);

                [encoder dispatchThreadgroups:MTLSizeMake(ne00, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
        case GGML_OP_ARGSORT:
            {
                GGML_ASSERT(src0->type == GGML_TYPE_F32);
                GGML_ASSERT( dst->type == GGML_TYPE_I32);

                const int nrows = ggml_nrows(src0);

                enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

                // bitonic sort requires the number of elements to be power of 2
                int64_t ne00_padded = 1;
                while (ne00_padded < ne00) {
                    ne00_padded *= 2;
                }

                // Metal kernels require the buffer size to be multiple of 16 bytes
                // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
                const int mem_size = GGML_PAD(ne00_padded*sizeof(int32_t), 16);

                id<MTLComputePipelineState> pipeline = nil;

                switch (order) {
                    case GGML_SORT_ORDER_ASC:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ARGSORT_F32_I32_ASC].pipeline;  break;
                    case GGML_SORT_ORDER_DESC: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ARGSORT_F32_I32_DESC].pipeline; break;
                    default: GGML_ABORT("fatal error");
                };

                ggml_metal_kargs_argsort args = {
                    /*.ncols =*/ ne00,
                    /*.ncols_pad =*/ ne00_padded
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args length:sizeof(args) atIndex:2];
                [encoder setThreadgroupMemoryLength:mem_size atIndex:0];

                [encoder dispatchThreadgroups:MTLSizeMake(1, nrows, 1) threadsPerThreadgroup:MTLSizeMake(ne00_padded, 1, 1)];
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                GGML_ASSERT(src0->type == GGML_TYPE_F32);

                float slope;
                memcpy(&slope, dst->op_params, sizeof(float));

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_LEAKY_RELU_F32].pipeline;

                ggml_metal_kargs_leaky_relu args = {
                    /*.slope =*/ slope
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0   atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst    atIndex:1];
                [encoder setBytes:&args length:sizeof(args)   atIndex:2];

                const int64_t n = ggml_nelements(dst);

                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                GGML_ASSERT(ne00 % 4  == 0);
                GGML_ASSERT(ne11 % 32 == 0);

                GGML_ASSERT(src0->type == GGML_TYPE_F32);
                GGML_ASSERT(src1->type == src2->type);

                //GGML_ASSERT(ggml_are_same_shape (src1, src2));
                GGML_ASSERT(ne11 == ne21);
                GGML_ASSERT(ne12 == ne22);

                struct ggml_tensor * src3 = node->src[3]; // mask
                struct ggml_tensor * src4 = node->src[4]; // sinks

                size_t offs_src3 = 0;
                size_t offs_src4 = 0;

                id<MTLBuffer> id_src3 = src3 ? ggml_metal_get_buffer(src3, &offs_src3) : nil;
                id<MTLBuffer> id_src4 = src4 ? ggml_metal_get_buffer(src4, &offs_src4) : nil;

                GGML_ASSERT(!src3 || src3->type == GGML_TYPE_F16);
                GGML_ASSERT(!src3 || src3->ne[1] >= GGML_PAD(src0->ne[1], 8) &&
                        "the Flash-Attention Metal kernel requires the mask to be padded to 8 and at least n_queries big");

                const int64_t  ne30 = src3 ? src3->ne[0] : 0; GGML_UNUSED(ne30);
                //const int64_t  ne31 = src3 ? src3->ne[1] : 0;
                const int64_t  ne32 = src3 ? src3->ne[2] : 0; GGML_UNUSED(ne32);
                const int64_t  ne33 = src3 ? src3->ne[3] : 0; GGML_UNUSED(ne33);

                const uint64_t nb30 = src3 ? src3->nb[0] : 0; GGML_UNUSED(nb30);
                const uint64_t nb31 = src3 ? src3->nb[1] : 0;
                const uint64_t nb32 = src3 ? src3->nb[2] : 0; GGML_UNUSED(nb32);
                const uint64_t nb33 = src3 ? src3->nb[3] : 0; GGML_UNUSED(nb33);

                float scale;
                float max_bias;
                float logit_softcap;

                memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
                memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
                memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));

                if (logit_softcap != 0.0f) {
                    scale /= logit_softcap;
                }

                const bool has_mask  = src3 != NULL;
                const bool has_sinks = src4 != NULL;
                const bool has_bias  = max_bias != 0.0f;
                const bool has_scap  = logit_softcap != 0.0f;

                const uint32_t n_head      = src0->ne[2];
                const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

                const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
                const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

                GGML_ASSERT(ne01 < 65536);

                if (!ggml_metal_op_flash_attn_ext_use_vec(dst)) {
                    // half8x8 kernel
                    const int64_t nqptg = 8;  // queries per threadgroup    !! sync with kernel template arguments !!
                    const int64_t ncpsg = 64; // cache values per simdgroup !! sync with kernel template arguments !!

                    GGML_ASSERT(nqptg <= 32);
                    GGML_ASSERT(nqptg  % 8  == 0);
                    GGML_ASSERT(ncpsg  % 32 == 0);

                    const int is_q = ggml_is_quantized(src1->type) ? 1 : 0;

                    // 2*(2*ncpsg)
                    // ncpsg soft_max values + ncpsg mask values
                    //
                    // 16*32*(nsg)
                    // the shared memory needed for the simdgroups to load the KV cache
                    // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
                    //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + is_q*(16*32*(nsg)))*(sizeof(float)/2), 16))

                    //int64_t nsgmax = 4;
                    //
                    //if (is_q) {
                    //    nsgmax = 2;
                    //    while (true) {
                    //        const size_t smem = FATTN_SMEM(nsgmax);
                    //        if (smem > props_dev.max_theadgroup_memory_size) {
                    //            break;
                    //        }
                    //        nsgmax *= 2;
                    //    }
                    //    nsgmax /= 2;
                    //}

                    // simdgroups per threadgroup (a.k.a. warps)
                    //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
                    int32_t nsg = 4;

                    const size_t smem = FATTN_SMEM(nsg);

                    ggml_metal_kargs_flash_attn_ext args = {
                        /*.ne01          =*/ ne01,
                        /*.ne02          =*/ ne02,
                        /*.ne03          =*/ ne03,
                        /*.nb01          =*/ nb01,
                        /*.nb02          =*/ nb02,
                        /*.nb03          =*/ nb03,
                        /*.ne11          =*/ ne11,
                        /*.ne_12_2       =*/ ne12,
                        /*.ne_12_3       =*/ ne13,
                        /*.ns10          =*/ nb11/nb10,
                        /*.nb11          =*/ nb11,
                        /*.nb12          =*/ nb12,
                        /*.nb13          =*/ nb13,
                        /*.ns20          =*/ nb21/nb20,
                        /*.nb21          =*/ nb21,
                        /*.nb22          =*/ nb22,
                        /*.nb23          =*/ nb23,
                        /*.ne32          =*/ ne32,
                        /*.ne33          =*/ ne33,
                        /*.nb31          =*/ nb31,
                        /*.nb32          =*/ nb32,
                        /*.nb33          =*/ nb33,
                        /*.ne1           =*/ ne1,
                        /*.ne2           =*/ ne2,
                        /*.ne3           =*/ ne3,
                        /*.scale         =*/ scale,
                        /*.max_bias      =*/ max_bias,
                        /*.m0            =*/ m0,
                        /*.m1            =*/ m1,
                        /*.n_head_log2   =*/ n_head_log2,
                        /*.logit_softcap =*/ logit_softcap,
                    };

                    id<MTLComputePipelineState> pipeline = ggml_metal_op_flash_attn_ext_get_pipeline(ctx, node, has_mask, has_sinks, has_bias, has_scap, nsg);

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args)     atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0     atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1     atIndex:2];
                    [encoder setBuffer:id_src2 offset:offs_src2     atIndex:3];
                    if (id_src3) {
                        [encoder setBuffer:id_src3 offset:offs_src3 atIndex:4];
                    } else {
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:4];
                    }
                    if (id_src4) {
                        [encoder setBuffer:id_src4 offset:offs_src4 atIndex:5];
                    } else {
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:5];
                    }

                    [encoder setBuffer:id_dst offset:offs_dst atIndex:6];

                    //printf("smem: %zu, max: %zu, nsg = %d, ne02 = %d, ne12 = %d\n", smem, props_dev.max_theadgroup_memory_size, (int) nsg, ne02, ne12);
                    GGML_ASSERT(smem <= props_dev.max_theadgroup_memory_size);
                    [encoder setThreadgroupMemoryLength:smem atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nqptg - 1)/nqptg, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
#undef FATTN_SMEM
                } else {
                    // half4x4 kernel
                    const int64_t nqptg = 1;  // queries per threadgroup    !! sync with kernel template arguments !!
                    const int64_t ncpsg = 32; // cache values per simdgroup !! sync with kernel template arguments !!
                    const int64_t nkpsg = 1*ncpsg;

                    GGML_ASSERT(nqptg <= 32);
                    GGML_ASSERT(nqptg  % 1  == 0);
                    GGML_ASSERT(ncpsg  % 32 == 0);

                    // ne00 + 2*ncpsg*(nsg)
                    // for each query, we load it as f16 in shared memory (ne00)
                    // and store the soft_max values and the mask
                    //
                    // ne20*(nsg)
                    // each simdgroup has a full f32 head vector in shared mem to accumulate results
                    //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(GGML_PAD(ne00, 128) + 4*ncpsg*(nsg)) + 2*GGML_PAD(ne20, 128)*(nsg))*(sizeof(float)/2), 16))

                    int64_t nsgmax = 2;
                    while (true) {
                        const size_t smem = FATTN_SMEM(nsgmax);
                        // avoid using more than half of the threadgroup memory - can cause slow downs especially for large head sizes
                        if (smem > props_dev.max_theadgroup_memory_size/2) {
                            break;
                        }
                        nsgmax *= 2;
                    }
                    nsgmax /= 2;

                    // simdgroups per threadgroup (a.k.a. warps)
                    //const int64_t nsgt = MAX(2, MIN(nsgmax, MIN((ne11 + nkpsg - 1)/(nkpsg), (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32)));
                    const int64_t nsgt = MAX(2, MIN(nsgmax, MIN((ne11 + nkpsg - 1)/(nkpsg), (int64_t) 1024/32)));

                    int64_t nsg = 1;
                    while (nsg <= nsgt) {
                        nsg *= 2;
                    }
                    nsg /= 2;

                    // workgroups
                    // each workgroup handles nsg*nkpsg cache values
                    int32_t nwg = 1;
                    if (false) {
                        // for small KV caches, we could launch a single workgroup and write the results directly to dst/
                        // however, this does not lead to significant improvement, so disabled
                        nwg = 1;
                        nsg = 4;
                    } else {
                        nwg = 32;
                        nsg = 1;
                        while (2*nwg*nsg*nkpsg < ne11 && nsg < 4) {
                            nsg *= 2;
                        }
                    }

                    ggml_metal_kargs_flash_attn_ext_vec args = {
                        /*.ne01          =*/ ne01,
                        /*.ne02          =*/ ne02,
                        /*.ne03          =*/ ne03,
                        /*.nb01          =*/ nb01,
                        /*.nb02          =*/ nb02,
                        /*.nb03          =*/ nb03,
                        /*.ne11          =*/ ne11,
                        /*.ne_12_2       =*/ ne12,
                        /*.ne_12_3       =*/ ne13,
                        /*.ns10          =*/ nb11/nb10,
                        /*.nb11          =*/ nb11,
                        /*.nb12          =*/ nb12,
                        /*.nb13          =*/ nb13,
                        /*.ns20          =*/ nb21/nb20,
                        /*.nb21          =*/ nb21,
                        /*.nb22          =*/ nb22,
                        /*.nb23          =*/ nb23,
                        /*.ne32          =*/ ne32,
                        /*.ne33          =*/ ne33,
                        /*.nb31          =*/ nb31,
                        /*.nb32          =*/ nb32,
                        /*.nb33          =*/ nb33,
                        /*.ne1           =*/ ne1,
                        /*.ne2           =*/ ne2,
                        /*.ne3           =*/ ne3,
                        /*.scale         =*/ scale,
                        /*.max_bias      =*/ max_bias,
                        /*.m0            =*/ m0,
                        /*.m1            =*/ m1,
                        /*.n_head_log2   =*/ n_head_log2,
                        /*.logit_softcap =*/ logit_softcap,
                    };

                    id<MTLComputePipelineState> pipeline = ggml_metal_op_flash_attn_ext_vec_get_pipeline(ctx, node, has_mask, has_sinks, has_bias, has_scap, nsg, nwg);

                    GGML_ASSERT(nsg*32 <= (int) pipeline.maxTotalThreadsPerThreadgroup);

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&args length:sizeof(args)     atIndex:0];
                    [encoder setBuffer:id_src0 offset:offs_src0     atIndex:1];
                    [encoder setBuffer:id_src1 offset:offs_src1     atIndex:2];
                    [encoder setBuffer:id_src2 offset:offs_src2     atIndex:3];
                    if (id_src3) {
                        [encoder setBuffer:id_src3 offset:offs_src3 atIndex:4];
                    } else {
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:4];
                    }
                    if (id_src4) {
                        [encoder setBuffer:id_src4 offset:offs_src4 atIndex:5];
                    } else {
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:5];
                    }

                    const size_t smem = FATTN_SMEM(nsg);

                    //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, props_dev.max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
                    GGML_ASSERT(smem <= props_dev.max_theadgroup_memory_size);

                    if (nwg == 1) {
                        // using 1 workgroup -> write the result directly into dst
                        [encoder setBuffer:id_dst offset:offs_dst atIndex:6];

                        [encoder setThreadgroupMemoryLength:smem atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
                    } else {
                        // sanity checks
                        GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
                        GGML_ASSERT(ne1*ne2*ne3 <= (1u << 31));

                        // write the results from each workgroup into a temp buffer
                        const size_t offs_tmp = offs_dst + ggml_nbytes(dst);
                        [encoder setBuffer:id_dst offset:offs_tmp atIndex:6];

                        [encoder setThreadgroupMemoryLength:smem atIndex:0];
                        [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg) threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];

                        // sync the 2 kernels
                        ggml_metal_encode_concurrency_reset(ctx_enc);

                        // reduce the results from the workgroups
                        {
                            const int32_t nrows = ne1*ne2*ne3;

                            ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                                nrows,
                            };

                            id<MTLComputePipelineState> pipeline0 = ggml_metal_op_flash_attn_ext_vec_reduce_get_pipeline(ctx, node, ne20, nwg);

                            [encoder setComputePipelineState:pipeline0];
                            [encoder setBytes:&args0   length:sizeof(args0) atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_tmp      atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst      atIndex:2];

                            //printf("ne1 = %d, ne2 = %d, ne3 = %d, ne20 = %d\n", ne1, ne2, ne3, ne20);
                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32*nwg, 1, 1)];
                        }
                    }
#undef FATTN_SMEM
                }
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            {
                id<MTLComputePipelineState> pipeline = nil;

                switch (src0t) {
                    case GGML_TYPE_F32:
                        {
                            GGML_ASSERT(ne0 % ggml_blck_size(dst->type) == 0);

                            switch (dstt) {
                                case GGML_TYPE_F32:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].pipeline; break;
                                case GGML_TYPE_I32:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_I32].pipeline; break;
                                case GGML_TYPE_F16:    pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F16].pipeline; break;
                                case GGML_TYPE_BF16:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_BF16].pipeline; break;
                                case GGML_TYPE_Q8_0:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_Q8_0].pipeline; break;
                                case GGML_TYPE_Q4_0:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_Q4_0].pipeline; break;
                                case GGML_TYPE_Q4_1:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_Q4_1].pipeline; break;
                                case GGML_TYPE_Q5_0:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_Q5_0].pipeline; break;
                                case GGML_TYPE_Q5_1:   pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_Q5_1].pipeline; break;
                                case GGML_TYPE_IQ4_NL: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_IQ4_NL].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    case GGML_TYPE_I32:
                        {
                            switch (dstt) {
                                case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_I32_F32].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    case GGML_TYPE_F16:
                        {
                            switch (dstt) {
                                case GGML_TYPE_F32:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F16_F32].pipeline; break;
                                case GGML_TYPE_F16:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F16_F16].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    case GGML_TYPE_BF16:
                        {
                            switch (dstt) {
                                case GGML_TYPE_F32:  pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_BF16_F32].pipeline; break;
                                case GGML_TYPE_BF16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_BF16_BF16].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    case GGML_TYPE_Q4_0:
                        {
                            switch (dstt) {
                                case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q4_0_F32].pipeline; break;
                                case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q4_0_F16].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    case GGML_TYPE_Q4_1:
                        {
                            switch (dstt) {
                                case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q4_1_F32].pipeline; break;
                                case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q4_1_F16].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    case GGML_TYPE_Q5_0:
                        {
                            switch (dstt) {
                                case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q5_0_F32].pipeline; break;
                                case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q5_0_F16].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    case GGML_TYPE_Q5_1:
                        {
                            switch (dstt) {
                                case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q5_1_F32].pipeline; break;
                                case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q5_1_F16].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    case GGML_TYPE_Q8_0:
                        {
                            switch (dstt) {
                                case GGML_TYPE_F32: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q8_0_F32].pipeline; break;
                                case GGML_TYPE_F16: pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_Q8_0_F16].pipeline; break;
                                default: GGML_ABORT("not implemented");
                            };
                        } break;
                    default: GGML_ABORT("not implemented");
                }

                GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

                // TODO: support
                //const int32_t nk00 = ne00/ggml_blck_size(dst->type);
                const int32_t nk00 = ne00;

                int nth = 32; // SIMD width

                while (nth < nk00 && nth < (int) pipeline.maxTotalThreadsPerThreadgroup) {
                    nth *= 2;
                }

                nth = MIN(nth, (int) pipeline.maxTotalThreadsPerThreadgroup);

                // when rows are small, we can batch them together in a single threadgroup
                int nrptg = 1;

                // TODO: relax this constraint in the future
                if (ggml_blck_size(src0->type) == 1 && ggml_blck_size(dst->type) == 1) {
                    if (nth > nk00) {
                        nrptg = (nth + nk00 - 1)/nk00;
                        nth   = nk00;

                        if (nrptg*nth > (int) pipeline.maxTotalThreadsPerThreadgroup) {
                            nrptg--;
                        }
                    }
                }

                nth = MIN(nth, nk00);

                ggml_metal_kargs_cpy args = {
                    /*.ne00 =*/ nk00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + nrptg - 1)/nrptg, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, nrptg, 1)];
            } break;
        case GGML_OP_POOL_2D:
            {
                GGML_ASSERT(ggml_is_contiguous(src0));
                GGML_ASSERT(src0t == GGML_TYPE_F32 && src0t == dstt);

                const int32_t * opts = dst->op_params;
                enum ggml_op_pool op = opts[0];

                id<MTLComputePipelineState> pipeline = nil;
                switch (src0t) {
                    case GGML_TYPE_F32: {
                        switch(op) {
                            case GGML_OP_POOL_AVG:
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_POOL_2D_AVG_F32].pipeline; break;
                            case GGML_OP_POOL_MAX:
                                pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_POOL_2D_MAX_F32].pipeline; break;
                            default: GGML_ASSERT(false && "not implemented");
                        }
                    } break;
                    default: GGML_ASSERT(false && "not implemented");
                }

                const int32_t k0 = opts[1];
                const int32_t k1 = opts[2];
                const int32_t s0 = opts[3];
                const int32_t s1 = opts[4];
                const int32_t p0 = opts[5];
                const int32_t p1 = opts[6];

                const int64_t IH = src0->ne[1];
                const int64_t IW = src0->ne[0];

                const int64_t N  = dst->ne[3];
                const int64_t OC = dst->ne[2];
                const int64_t OH = dst->ne[1];
                const int64_t OW = dst->ne[0];

                const int64_t parallel_elements = N * OC * OH * OW;
                const int64_t n_threads = MIN((int64_t)[pipeline maxTotalThreadsPerThreadgroup], parallel_elements);
                const int64_t n_tg = (parallel_elements + n_threads - 1) / n_threads;

                ggml_metal_kargs_pool_2d args_pool_2d = {
                    /* .k0 = */ k0,
                    /* .k1 = */ k1,
                    /* .s0 = */ s0,
                    /* .s1 = */ s1,
                    /* .p0 = */ p0,
                    /* .p1 = */ p1,
                    /* .IH = */ IH,
                    /* .IW = */ IW,
                    /* .OH = */ OH,
                    /* .OW = */ OW,
                    /* .parallel_elements = */ parallel_elements
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                [encoder setBytes:&args_pool_2d length:sizeof(args_pool_2d) atIndex:2];

                [encoder dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(n_threads, 1, 1)];
            } break;
        case GGML_OP_ARGMAX:
            {
                GGML_ASSERT(src0->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_contiguous_1(src0));
                GGML_ASSERT(nb00 == ggml_type_size(src0->type));

                const int64_t nrows = ggml_nrows(src0);

                int nth = 32; // SIMD width
                while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
                    nth *= 2;
                }

                id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_ARGMAX].pipeline;

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:3];
                [encoder setThreadgroupMemoryLength:32*sizeof(float)   atIndex:0];
                [encoder setThreadgroupMemoryLength:32*sizeof(int32_t) atIndex:1];

                [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            } break;
       default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(dst->op));
                GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_encode_concurrency_add(ctx_enc, nodes[i])) {
            ggml_metal_encode_concurrency_reset(ctx_enc);
        }
    }

    return n_fuse;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = 64;

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        const bool should_capture = ctx->capture_next_compute;
        if (should_capture) {
            ctx->capture_next_compute = false;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                // create capture scope
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:ctx->device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithFormat:@"/tmp/perf-metal.gputrace"]];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (!should_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        struct ggml_mem_ranges * mem_ranges = ctx->cmd_bufs[cb_idx].mem_ranges;
        if (mem_ranges) {
            ggml_mem_ranges_reset(mem_ranges);
        }

        id<MTLComputeCommandEncoder> encoder;

        if (ctx->use_concurrency) {
            encoder = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
        } else {
            encoder = [cmd_buf computeCommandEncoder];
        }

        int node_start = 0;
        int node_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            node_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            node_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        const bool should_capture = ctx->capture_next_compute;

        struct ggml_metal_encode_context ctx_enc = {
            /*.encoder    =*/ encoder,
            /*.ctx        =*/ ctx,
            /*.mem_ranges =*/ mem_ranges,
        };

        for (int idx = node_start; idx < node_end;) {
            if (should_capture) {
                [encoder pushDebugGroup:[NSString stringWithCString:ggml_op_desc(ggml_graph_node(ctx->gf, idx)) encoding:NSUTF8StringEncoding]];
            }

            const int res = ggml_metal_encode_node(&ctx_enc, idx, node_end);
            if (idx + res > node_end) {
                GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                        "https://github.com/ggml-org/llama.cpp/pull/14849");
            }

            if (should_capture) {
                [encoder popDebugGroup];
            }

            if (res == 0) {
                break;
            }

            idx += res;
        }

        [encoder endEncoding];

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->device != nil);

    return [ctx->device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_next_compute = true;
}

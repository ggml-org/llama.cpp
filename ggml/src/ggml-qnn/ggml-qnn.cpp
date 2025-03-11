/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Qualcomm QNN SDK and reference tech guides could be found at:
 * https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
 * https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
 *
 * this single-source-file or self-contained implementation of ggml-qnn backend has 10 sections:
 * section-1  forward/prototype declaration
 * section-2  global vars, macros, data structures
 * section-3  ggml-qnn internal troubleshooting function/class
 * section-4  helper function for WoA(Windows on ARM)
 * section-5  general helper function
 * section-6  QNN helper function
 * section-7  ggml-qnn backend helper function / class
 * section-8  implementation of ggml-qnn backend according to ggml's backend subsystem
 * section-9  implementation of offload ggml op to QNN backend
 * section-10 illustrate why the second approach is actual an fake at the moment
 *
 * currently provide following ggml op' QNN backend implementation:
 * - GGML_OP_ADD:    this is a simple skeleton, can expand other ggml ops according to expertise
 * - GGML_OP_MUL:    this is a simple skeleton, can expand other ggml ops according to expertise
 * - GGML_OP_MUL_MAT:this is a complicated skeleton, can expand other complex ggml ops accordingly
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#if defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdatomic.h>
#endif

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <tuple>
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <regex>
#include <random>
#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <unordered_set>
#include <utility>
#include <future>
#if (defined __ANDROID__) || (defined ANDROID)
#include "android/log.h"
#endif

#if !defined(__ANDROID__) && !defined(__linux__)
#include <wchar.h>
#include <malloc.h>
#include <Windows.h>
#endif

#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnGraph.h"
#include "QnnProperty.h"
#include "QnnTensor.h"
#include "QnnInterface.h"
#include "Saver/QnnSaver.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"

#include "ggml-qnn.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

// =================================================================================================
//  section-1: forward/prototype declaration
// =================================================================================================
class  qnn_instance;
struct ggml_backend_qnn_context;
typedef void (* ggmlqnn_op_func_t)(ggml_backend_qnn_context * ctx, ggml_tensor * op);
static Qnn_Tensor_t * ggmlqnn_create_general_tensor(const ggml_tensor * tensor, const char * name,
                                             Qnn_TensorType_t qnn_tensor_type,
                                             Qnn_DataType_t qnn_data_type,
                                             uint32_t rank, uint32_t * dims,
                                             void * data, uint32_t data_size,
                                             bool b_transpose = false);
static enum ggml_status ggmlqnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);
static void ggmlqnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...);
static inline bool ggmlqnn_is_valid_params(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

//op functions:
//done
static void ggml_qnn_general_node(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_mul_mat(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
//todo
static void ggml_qnn_repeat(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_div(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_leaky_relu(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_concat(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_arange(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_sqr(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_clamp(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_scale(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_argsort(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_group_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_acc(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_sum_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_pad(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_pool2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_dup(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_rms_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_cpy(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_rope(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_im2col(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_softmax(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_get_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_upsample_nearest2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_timestep_embedding(ggml_backend_qnn_context * ctx, ggml_tensor * dst);
static void ggml_qnn_diag_mask(ggml_backend_qnn_context * ctx, ggml_tensor * dst, float value);

// =================================================================================================
//  section-2: global var, macro, data structure
// =================================================================================================
// the following two vars can be fetched from [qnn_runtimelib_path]/ggml-qnn.cfg
// [general]
// print_qnn_internal_log=0
// inference_approach=0
static int g_print_qnn_internal_log         = 0; // enable/disable QNN's internal log
static int g_inference_approach             = 0; // 0: general approach,similar to ggml-sycl or ggml-cann 1: mapping entire ggml cgraph to QNN graph
static const char * g_qnn_cfgfilename       = "ggml-qnn.cfg";

#if defined(__ANDROID__)
//Android command line program
static const char * g_qnn_runtimelib_path   = "/data/local/tmp/";
#elif defined(__linux__)
static const char * g_qnn_runtimelib_path   = "/tmp/";
#elif defined(_WIN32)
static const char * g_qnn_runtimelib_path   = "C:\\";
#endif

#if !defined(__ANDROID__) && !defined(__linux__)
static std::atomic<int32_t> g_ggmltensor_idx(0); //ensure every QNN tensor name is unique
#else
static int32_t g_ggmltensor_idx = 0; //ensure every QNN tensor name is unique
#endif

#if 0//def NDEBUG
#define GGMLQNN_DEBUG                                   0
#define ENABLE_QNNBACKEND_PERF                          0
#define GGMLQNN_PRINT_OP_ADD_LOG                        0  // GGML_OP_ADD already verified with QNN-CPU / QNN-GPU / QNN-NPU
#define GGMLQNN_PRINT_OP_MUL_MAT_LOG                    0
#else
#define GGMLQNN_DEBUG                                   1  // for troubleshooting QNN backend
#define ENABLE_QNNBACKEND_PERF                          0
#define GGMLQNN_PRINT_OP_ADD_LOG                        0  // GGML_OP_ADD already verified with QNN-CPU / QNN-GPU / QNN-NPU
#define GGMLQNN_PRINT_OP_MUL_MAT_LOG                    1
#endif
#define GGML_QNN_LOGBUF_LEN                             4096

#define GGMLQNN_LOG_ERROR(...)                          ggmlqnn_log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLQNN_LOG_WARN(...)                           ggmlqnn_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLQNN_LOG_INFO(...)                           ggmlqnn_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGMLQNN_DEBUG
#define GGMLQNN_LOG_DEBUG(...)                          ggmlqnn_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLQNN_LOG_DEBUG(...)
#endif

#define GGMLQNN_MEM_ADD(alignment)                      (sizeof (size_t) + alignment)
#define GGMLQNN_MEM_MASK(alignment)                     ((uintptr_t)alignment - 1)
#define GQCGT                                           ggmlqnn_create_general_tensor
#define QNN_VER_PTR(x)                                  (&((x).v1))
#define RPCMEM_DEFAULT_FLAGS                            1
#define RPCMEM_HEAP_ID_SYSTEM                           25

#define QNN_TENSOR_GET_ID(tensor)                       get_qnn_tensorid(tensor)
#define QNN_TENSOR_GET_NAME(tensor)                     get_qnn_tensorname(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)                     get_qnn_tensortype(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)              get_qnn_tensor_dataformat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)                get_qnn_tensor_datatype(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor)             get_qnn_tensor_quantparams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)                     get_qnn_tensor_rank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)               get_qnn_tensor_dimensions(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)                 get_qnn_tensor_memtype(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF(tensor)               get_qnn_tensor_clientbuf(tensor)
#define QNN_TENSOR_GET_MEM_HANDLE(tensor)               get_qnn_tensor_memhandle(tensor)

#define QNN_TENSOR_SET_ID(tensor, value)                set_qnn_tensor_id(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)              set_qnn_tensor_name(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)              set_qnn_tensor_type(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)       set_qnn_tensor_dataformat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)         set_qnn_tensor_datatype(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value)      set_qnn_tensor_quantparams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)              set_qnn_tensor_rank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)        set_qnn_tensor_dimensions(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)          set_qnn_tensor_memtype(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)        set_qnn_tensor_clientbuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value)        set_qnn_tensor_memhandle(tensor, value)

#define DISABLE_COPY(class_name)                                                \
    class_name(const class_name &) = delete;                                    \
    void operator=(const class_name &) = delete

#define DISABLE_MOVE(class_name)                                                \
    class_name(class_name &&) = delete;                                         \
    void operator=(class_name &&) = delete

#define CHECK_QNN_API(error, result)                                            \
    do {                                                                        \
        error = (result);                                                       \
        if (QNN_SUCCESS != error) {                                             \
            if (error == QNN_COMMON_ERROR_NOT_SUPPORTED) {                      \
                GGMLQNN_LOG_WARN("WARNING: QNN feature/API not supported\n");   \
            } else {                                                            \
                GGMLQNN_LOG_INFO("QNN API error = %d(%s)\n", error, ggmlqnn_get_qnnerror_string(error));  \
            }                                                                   \
        }                                                                       \
    } while (0)

#define GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst)                              \
    do {                                                                        \
        if (!ggmlqnn_is_valid_params((ctx), (src0), (src1), (dst))) {           \
            return;                                                             \
        }                                                                       \
    } while (0)

using pfn_rpc_mem_init                          = void (*)(void);
using pfn_rpc_mem_deinit                        = void (*)(void);
using pfn_rpc_mem_alloc                         = void *(*)(int, uint32_t, int);
using pfn_rpc_mem_free                          = void (*)(void *);
using pfn_rpc_mem_to_fd                         = int (*)(void *);
using _pfn_QnnSaver_initialize                  = decltype(QnnSaver_initialize);
using _pfn_QnnInterface_getProviders            = decltype(QnnInterface_getProviders);
using _pfn_QnnSystemInterface_getProviders      = decltype(QnnSystemInterface_getProviders);
using qnn_res_t                                 = std::tuple<Qnn_GraphHandle_t, std::vector< Qnn_Tensor_t *>>;
using qnn_tensors_t                             = std::vector< Qnn_Tensor_t *>;

enum class ggml_qnn_profile_level {
    profile_off     = 0,
    profile_basic   = 1,
    profile_detail  = 2
};

enum qcom_htp_arch {
    NONE = 0,
    V68 = 68,
    V69 = 69,
    V73 = 73,
    V75 = 75,
    V79 = 79,
};

enum qcom_chipset_soc_model {
    UNKNOWN_SM = 0,
    SM7450 = 41,  // v69, 7 Gen1
    SM8350 = 30,  // v68, 888
    SM8450 = 36,  // v69, SD 8 Gen 1
    SM8475 = 42,  // v69, SD 8+ Gen 1
    SM8550 = 43,  // v73, SD 8 Gen 2
    SM8650 = 57,  // v75, SD 8 Gen 3
    SM8750 = 69,  // v79, SD 8 Gen 4
#if !defined(__ANDROID__) && !defined(__linux__)
    SC7280X     = 44,
    SC8280X     = 37,
    SC8380XP    = 60,
#endif
};

struct qcom_socinfo {
    uint32_t soc_model;
    size_t htp_arch;
    size_t vtcm_size_in_mb;
    char soc_desc[GGML_MAX_NAME];
};

struct ggml_backend_qnn_context {
    int device;
    int threads;
    char name[GGML_MAX_NAME];
    char desc[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    qnn_instance * instance;
    struct ggml_backend * backend;
    QNN_INTERFACE_VER_TYPE raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE raw_system_interface;
    struct qcom_socinfo           socinfo;

    std::unique_ptr<char[]> work_data;
    std::vector<std::future<void>> tasks;
    size_t work_size    = 0;
    size_t desired_size = 0;
    int n_threads       = GGML_DEFAULT_N_THREADS;
};

struct qnn_op_caps_t {
    const char * qnn_op_name        = nullptr;
    const size_t input_param_count  = 0;
    const char * qnn_param_name     = nullptr;
};

//file:///opt/qcom/aistack/qairt/2.31.0.250130/docs/QNN/general/overview.html#tbl-supported-snapdragon-devices
static struct qcom_socinfo g_qnn_soc_info_table[] = {
        /* Qualcomm SnapDragon 7 Gen 1 */
        {
                .soc_model         = SM7450,
                .htp_arch          = V69,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 7 Gen 1"},

        /* Qualcomm SnapDragon 888 */
        {
                .soc_model         = SM8350,
                .htp_arch          = V68,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 888 "},

        /* Qualcomm SnapDragon 8 Gen 1 */
        {
                .soc_model         = SM8450,
                .htp_arch          = V69,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 1"},

        /* Qualcomm SnapDragon 8 Gen 1+ */
        {
                .soc_model         = SM8475,
                .htp_arch          = V69,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 1+"},

        /* Qualcomm SnapDragon 8 Gen 2 */
        {
                .soc_model         = SM8550,
                .htp_arch          = V73,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 2"},

        /* Qualcomm SnapDragon 8 Gen 3 */
        {
                .soc_model         = SM8650,
                .htp_arch          = V75,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 3 "},

        /* Qualcomm SnapDragon 8 Gen 4 */
        {
                .soc_model         = SM8750,
                .htp_arch          = V79,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 4"},

#if !defined(__ANDROID__) && !defined(__linux__)
        /* Qualcomm SnapDragon 7c Gen 2 */
        {
                .soc_model         = SC7280X,
                .htp_arch          = V68,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 7c Gen 2"},

        /* Qualcomm SnapDragon 8cx Gen 3 */
        {
                .soc_model         = SC8280X,
                .htp_arch          = V68,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8cx Gen 3"},

        /* Qualcomm SnapDragon 8cx Gen 4 */
        {
                .soc_model         = SC8380XP,
                .htp_arch          = V73,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8cx Gen 4"},
#endif

};

// file:///opt/qcom/aistack/qairt/2.31.0.250130/docs/QNN/general/quantization.html
// CPU - Choose a non-quantized model.Quantized models are currently incompatible with the CPU backend
// GPU - Choose a non-quantized model.Quantized models are currently incompatible with the GPU backend
// HTP - Choose a quantized model. Quantized models are required when running on the HTP backend
// DSP - Choose a quantized model. Quantized models are required when running on the DSP backend
// HTA - Choose a quantized model. Quantized models are required when running on the HTA backend
static struct ggml_backend_qnn_context g_qnn_mgr[GGML_QNN_MAX_DEVICES] = {
        [QNN_BACKEND_CPU] = {.device               = 0,
                .threads              = 1,
                .name                 = "qnn-cpu",
                .desc                 = "Qualcomm Kryo CPU",
#if !defined(__ANDROID__) && !defined(__linux__)
                .lib                  = "QnnCpu.dll",
#else
                .lib                  = "libQnnCpu.so",
#endif
                .instance             = nullptr,
                .backend              = nullptr,
                .raw_interface        = {},
                .raw_system_interface = {},
                .socinfo              = {}},

        [QNN_BACKEND_GPU] = {.device               = 1,
                .threads              = 1,
                .name                 = "qnn-gpu",
                .desc                 = "Qualcomm Adreno GPU",
#if !defined(__ANDROID__) && !defined(__linux__)
                .lib                  = "QnnGpu.dll",
#else
                .lib                  = "libQnnGpu.so",
#endif
                .instance             = nullptr,
                .backend              = nullptr,
                .raw_interface        = {},
                .raw_system_interface = {},
                .socinfo              = {}},

        [QNN_BACKEND_NPU] = {.device               = 2,
                .threads              = 1,
                .name                 = "qnn-npu",
                .desc                 = "Qualcomm NPU(Hexagon Tensor Processor)",
#if !defined(__ANDROID__) && !defined(__linux__)
                .lib                  = "QnnHtp.dll",
#else
                .lib                  = "libQnnHtp.so",
#endif
                .instance             = nullptr,
                .backend              = nullptr,
                .raw_interface        = {},
                .raw_system_interface = {},
                .socinfo              = {}},
};

static const qnn_op_caps_t ggmlqnn_k_op_caps[] = {
        {}, // GGML_OP_NONE
        {}, // GGML_OP_DUP
        {
                // GGML_OP_ADD
                QNN_OP_ELEMENT_WISE_ADD,
                2,
        },
        {}, // GGML_OP_ADD1
        {}, // GGML_OP_ACC
        {}, // GGML_OP_SUB
        {
                // GGML_OP_MUL
                QNN_OP_ELEMENT_WISE_MULTIPLY,
                2,
        },
        {}, // GGML_OP_DIV
        {}, // GGML_OP_SQR
        {}, // GGML_OP_SQRT
        {}, // GGML_OP_LOG
        {}, // GGML_OP_SIN
        {}, // GGML_OP_COS
        {}, // GGML_OP_SUM
        {}, // GGML_OP_SUM_ROWS
        {}, // GGML_OP_MEAN
        {}, // GGML_OP_ARGMAX
        {}, // GGML_OP_COUNT_EQUAL
        {}, // GGML_OP_REPEAT
        {}, // GGML_OP_REPEAT_BACK
        {}, // GGML_OP_CONCAT
        {}, // GGML_OP_SILU_BACK
        {}, // GGML_OP_NORM
        {}, // GGML_OP_RMS_NORM
        {}, // GGML_OP_RMS_NORM_BACK
        {}, // GGML_OP_GROUP_NORM
        {
                // GGML_OP_MUL_MAT
                QNN_OP_MAT_MUL,
                2,
        },
        {}, // GGML_OP_MUL_MAT_ID
        {}, // GGML_OP_OUT_PROD
        {}, // GGML_OP_SCALE
        {}, // GGML_OP_SET
        {}, // GGML_OP_CPY
        {}, // GGML_OP_CONT
        {}, // GGML_OP_RESHAPE
        {}, // GGML_OP_VIEW
        {}, // GGML_OP_PERMUTE
        {}, // GGML_OP_TRANSPOSE
        {}, // GGML_OP_GET_ROWS
        {}, // GGML_OP_GET_ROWS_BACK
        {}, // GGML_OP_DIAG
        {}, // GGML_OP_DIAG_MASK_INF
        {}, // GGML_OP_DIAG_MASK_ZERO
        {}, // GGML_OP_SOFT_MAX
        {}, // GGML_OP_SOFT_MAX_BACK
        {}, // GGML_OP_ROPE
        {}, // GGML_OP_ROPE_BACK
        {}, // GGML_OP_CLAMP
        {}, // GGML_OP_CONV_TRANSPOSE_1D
        {}, // GGML_OP_IM2COL
        {}, // GGML_OP_IM2COL_BACK
        {}, // GGML_OP_CONV_TRANSPOSE_2D
        {}, // GGML_OP_POOL_1D
        {}, // GGML_OP_POOL_2D
        {}, // GGML_OP_POOL_2D_BACK
        {}, // GGML_OP_UPSCALE
        {}, // GGML_OP_PAD
        {}, // GGML_OP_PAD_REFLECT_1D
        {}, // GGML_OP_ARANGE
        {}, // GGML_OP_TIMESTEP_EMBEDDING
        {}, // GGML_OP_ARGSORT
        {}, // GGML_OP_LEAKY_RELU
        {}, // GGML_OP_FLASH_ATTN_EXT
        {}, // GGML_OP_FLASH_ATTN_BACK
        {}, // GGML_OP_SSM_CONV
        {}, // GGML_OP_SSM_SCAN
        {}, // GGML_OP_WIN_PART
        {}, // GGML_OP_WIN_UNPART
        {}, // GGML_OP_GET_REL_POS
        {}, // GGML_OP_ADD_REL_POS
        {}, // GGML_OP_RWKV_WKV6
        {}, // GGML_OP_GATED_LINEAR_ATTN
        {}, // GGML_OP_UNARY
        {}, // GGML_OP_MAP_UNARY
        {}, // GGML_OP_MAP_BINARY
        {}, // GGML_OP_MAP_CUSTOM1_F32
        {}, // GGML_OP_MAP_CUSTOM2_F32
        {}, // GGML_OP_MAP_CUSTOM3_F32
        {}, // GGML_OP_MAP_CUSTOM1
        {}, // GGML_OP_MAP_CUSTOM2
        {}, // GGML_OP_MAP_CUSTOM3
        {}, // GGML_OP_CROSS_ENTROPY_LOSS
        {}, // GGML_OP_CROSS_ENTROPY_LOSS_BACK
        {}, // GGML_OP_OPT_STEP_ADAMW
        {}, // GGML_UNARY_OP_ABS
        {}, // GGML_UNARY_OP_SGN
        {}, // GGML_UNARY_OP_NEG
        {}, // GGML_UNARY_OP_STEP
        {}, // GGML_UNARY_OP_TANH
        {}, // GGML_UNARY_OP_ELU
        {}, // GGML_UNARY_OP_RELU
        {}, // GGML_UNARY_OP_SIGMOID
        {}, // GGML_UNARY_OP_GELU
        {}, // GGML_UNARY_OP_GELU_QUICK
        {}, // GGML_UNARY_OP_SILU
        {}, // GGML_UNARY_OP_HARDSWISH
        {}, // GGML_UNARY_OP_HARDSIGMOID
        {}, // GGML_UNARY_OP_EXP
};

// =================================================================================================
//  section-3: ggml-qnn internal troubleshooting function/class
// =================================================================================================
static void ggmlqnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex ggmlqnn_log_internal_mutex;
    static char s_ggmlqnn_log_internal_buf[GGML_QNN_LOGBUF_LEN];

    GGML_UNUSED(file);
#if !(defined __ANDROID__) || !(defined ANDROID)
    GGML_UNUSED(level);
#endif
    {
        std::lock_guard<std::mutex> lock(ggmlqnn_log_internal_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_ggmlqnn_log_internal_buf, GGML_QNN_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_ggmlqnn_log_internal_buf + len_prefix, GGML_QNN_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGML_QNN_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            __android_log_print(ANDROID_LOG_INFO, "ggml-qnn", "%s\n", s_ggmlqnn_log_internal_buf);
            if (GGML_LOG_LEVEL_INFO == level) {
                printf("%s\n", s_ggmlqnn_log_internal_buf);
            }
#else
            //for Snapdragon based WoA(Windows on ARM) device or Linux
            printf("%s\n", s_ggmlqnn_log_internal_buf);
#endif
        }
        va_end(args);
    }
}

#if ENABLE_QNNBACKEND_PERF
class qnn_perf {
public:
    qnn_perf(const std::string & perf_name) : _perf_name(std::move(perf_name)) {};
    qnn_perf() = delete;
    qnn_perf(const qnn_perf & ) = delete;
    qnn_perf & operator= (const qnn_perf & ) = delete;

    void start() {
        _begin_time = ggml_time_us();
    }

    void info() {
        _end_time = ggml_time_us();
        _duration = (_end_time - _begin_time);
        GGMLQNN_LOG_DEBUG("duration of %s : %lld microseconds\n", _perf_name.c_str(), _duration);
    }

private:
    int64_t _begin_time = 0LL;
    int64_t _end_time   = 0LL;
    int64_t _duration   = 0LL;
    std::string _perf_name;
};
#else
class qnn_perf {
public:
    qnn_perf(const std::string & perf_name) {
        GGML_UNUSED(perf_name);
    }
    qnn_perf() = delete;
    qnn_perf(const qnn_perf & ) = delete;
    qnn_perf & operator= (const qnn_perf & ) = delete;

    void start() {}
    void info() {}
};
#endif

class qnn_cfg {
public:
    void dump(std::function<void(const std::string &, const std::string &, const std::string &)> worker) {
        if (!_load_success) {
            GGMLQNN_LOG_INFO("qnn cfg file %s not loadded", _cfg_filename.c_str());
            return;
        }
        auto iter = _qnn_cfg.begin();
        while (iter != _qnn_cfg.end()) {
            auto kv_iter = iter->second.begin();
            while (kv_iter != iter->second.end()) {
                worker(iter->first, kv_iter->first, kv_iter->second);
                ++kv_iter;
            }
            ++iter;
        }
    }

    bool load(const std::string & file_name) {
        if (file_name == "") {
            return false;
        }
        _cfg_filename = file_name;
        std::ifstream in;
        std::string line;
        in.open(file_name.c_str());
        if (not in.is_open()) {
            GGMLQNN_LOG_WARN("can't open file %s", file_name.c_str());
            return false;
        }
        while (getline(in, line)) {
            std::string section, key, value;
            if (not parse_line(line, section, key, value)) {
                continue;
            }
            set_section_keyvalue(section, key, value);
        }
        _load_success = true;
        return true;
    }

    void get_stringvalue(const std::string & section, const std::string & key, std::string & value, std::string default_value) {
        value = default_value;
        if (_qnn_cfg.find(section) == _qnn_cfg.end()) {
            return;
        }
        if (_qnn_cfg[section].find(key) == _qnn_cfg[section].end()) {
            return;
        }
        value = _qnn_cfg[section][key];
    }

    void get_intvalue(const std::string & section, const std::string & key, int & value, int default_value) {
        value = default_value;
        if (_qnn_cfg.find(section) == _qnn_cfg.end()) {
            return;
        }
        if (_qnn_cfg[section].find(key) == _qnn_cfg[section].end()) {
            return;
        }
        value = atol(_qnn_cfg[section][key].c_str());
    }

private:
    void ltrim(std::string & str) {
        if (str.empty()) return;
        size_t len = 0;
        char* temp = (char*)str.c_str();
        while (*temp && isblank(*temp)) {
            ++len;
            ++temp;
        }
        if (len > 0) str.erase(0, len);
    }

    void rtrim(std::string & str) {
        if (str.empty()) return;
        size_t len = str.length();
        size_t pos = len;
        while (pos > 0) {
            if (not isblank(str[pos - 1])) {
                break;
            }
            --pos;
        }
        if (pos != len) str.erase(pos);
    }

    void trim(std::string& str) {
        ltrim(str);
        rtrim(str);
    }

    void set_section_keyvalue(std::string & section, std::string & key, std::string & value) {
        if (_qnn_cfg.find(section) == _qnn_cfg.end()) {
            std::unordered_map<std::string, std::string> kv_map;
            _qnn_cfg[section] = kv_map;
        }
        if (key != "" && value != "") _qnn_cfg[section][key] = value;
    }

    bool parse_line(std::string & line, std::string & section, std::string & key, std::string & value) {
        static std::string cur_section = "";
        std::string nodes[2] = {"#", ";"};
        for (int i = 0; i < 2; ++i) {
            std::string::size_type pos = line.find(nodes[i]);
            if (pos != std::string::npos) line.erase(pos);
        }
        trim(line);
        if (line == "") return false;
        if (line[0] == '[' && line[line.size() - 1] == ']') {
            section = line.substr(1, line.size() - 2);
            trim(section);
            cur_section = section;
            return false;
        }
        if (cur_section == "") return false;
        bool is_key = true;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '=') {
                is_key = false;
                continue;
            }
            if (is_key) {
                key += line[i];
            } else {
                value += line[i];
            }
        }
        section = cur_section;
        trim(key);
        trim(value);
        return true;
    }
private:
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> _qnn_cfg;
    bool _load_success = false;
    std::string _cfg_filename;
};

// =================================================================================================
//  section-4: helper function for WoA(Window on ARM)
// =================================================================================================
#if !defined(__ANDROID__) && !defined(__linux__)
#define RTLD_GLOBAL 0x100
#define RTLD_LOCAL  0x000
#define RTLD_LAZY   0x000
#define RTLD_NOW    0x001
static void *       dlopen(const char * filename, int flag);
static int          dlclose(void * handle);
static void *       dlsym(void* handle, const char* name);
static const char * dlerror(void);

static const char * last_func = nullptr;
static long last_err;
static void * dlopen(const char * dll, int flags) {
  HINSTANCE h = LoadLibraryA(dll);
  GGML_UNUSED(flags);
  if (h == NULL) {
    last_err  = GetLastError();
    last_func = "dlopen";
  }
  return h;
}

static int dlclose(void * h) {
  if (!FreeLibrary((HINSTANCE)h)) {
    last_err  = GetLastError();
    last_func = "dlclose";
    return -1;
  }
  return 0;
}

static void * dlsym(void * h, const char * name) {
  FARPROC p = GetProcAddress((HINSTANCE)h, name);
  if (!p) {
    last_err  = GetLastError();
    last_func = "dlsym";
  }
  return (void*)(intptr_t)p;
}

static const char * dlerror(void) {
  static char str[512];
  if (!last_err) return nullptr;

  snprintf(str, 512, "%s error #%ld", last_func, last_err);
  last_err  = 0;
  last_func = NULL;

  return str;
}
#endif

// =================================================================================================
//  section-5: general helper function
// =================================================================================================
//the following 3 helper funcs are used to ensure every QNN tensor name is unique
static void ggmqnn_reset_tensoridx() {
    g_ggmltensor_idx = 0;
}

static void ggmqnn_inc_tensoridx() {
    g_ggmltensor_idx++;
}

static int32_t ggmqnn_get_tensoridx() {
#if !defined(__ANDROID__) && !defined(__linux__)
    return g_ggmltensor_idx.load();
#else
    return g_ggmltensor_idx;
#endif
}

static void * ggmlqnn_mallocz_aligned(size_t size, size_t alignment) {
    uint8_t * buffer    = NULL;
    size_t * sp         = NULL;
    buffer = static_cast<uint8_t *>(calloc(1, size + GGMLQNN_MEM_ADD(alignment)));
    if (!buffer)
        return NULL;
    sp = (size_t *)buffer;
    *sp = size;
    buffer = (uint8_t *)(((uintptr_t) buffer + GGMLQNN_MEM_ADD(alignment)) & ~GGMLQNN_MEM_MASK(alignment));
    buffer[-1] = buffer - (uint8_t *)sp;
    return buffer;
}

static void * ggmlqnn_malloc_aligned(size_t size, size_t alignment) {
    uint8_t * buffer = NULL;
    size_t * sp = NULL;
    buffer = static_cast<uint8_t *>(malloc(size + GGMLQNN_MEM_ADD(alignment)));
    if (!buffer)
        return NULL;
    sp = (size_t *)buffer;
    *sp = size;
    buffer = (uint8_t *)(((uintptr_t) buffer + GGMLQNN_MEM_ADD(alignment)) & ~GGMLQNN_MEM_MASK(alignment));
    buffer[-1] = buffer - (uint8_t *)sp;
    return buffer;
}

static void ggmqnn_free_aligned(void * ptr) {
    uint8_t * old = (uint8_t *)ptr;
    if (!old)
        return;
    old -= old[-1];
    free(old);
}

static intptr_t ggmlqnn_align_to(size_t alignment, intptr_t offset) {
    return offset % alignment == 0 ? offset
                                   : offset +
                                     (static_cast<intptr_t>(alignment) -
                                      offset % static_cast<intptr_t>(alignment));
}

static size_t ggmlqnn_get_system_total_memory_in_bytes() {
#if defined(__ANDROID__) || defined(__linux__)
    struct sysinfo info = {};
    if (0 == sysinfo(&info)) {
        return (info.totalram + info.totalswap) * info.mem_unit;
    }
    size_t pages      = (size_t)sysconf(_SC_PHYS_PAGES);
    size_t page_size  = (size_t)sysconf(_SC_PAGE_SIZE);

    return pages * page_size;
#else
    //FIXME: Snapdragon based WoA(Windows on ARM)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        GGMLQNN_LOG_INFO("total physical mem:%llu Mb", statex.ullTotalPhys >> 20);
        GGMLQNN_LOG_INFO("avail physical mem:%llu Mb", statex.ullAvailPhys >> 20);
        return statex.ullTotalPhys;
    }
    return 0;
#endif
}

static size_t ggmlqnn_get_system_free_memory_in_bytes() {
#if defined(__ANDROID__) || defined(__linux__)
    struct sysinfo info = {};
    if (0 == sysinfo(&info)) {
        return (info.freeram + info.freeswap) * info.mem_unit;
    }
    size_t avail_pages = (size_t)sysconf(_SC_AVPHYS_PAGES);
    size_t page_size   = (size_t)sysconf(_SC_PAGE_SIZE);

    return avail_pages * page_size;
#else
    //FIXME: Snapdragon based WoA(Windows on ARM)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        GGMLQNN_LOG_INFO("total physical mem:%llu Mb", statex.ullTotalPhys >> 20);
        GGMLQNN_LOG_INFO("avail physical mem:%llu Mb", statex.ullAvailPhys >> 20);
        return statex.ullAvailPhys;
    }
    return 0;
#endif
}

static size_t ggmlqnn_memscpy(void * dst, size_t dst_size, const void * src, size_t copy_size) {
    if (!dst || !src || !dst_size || !copy_size)
        return 0;

    size_t min_size = dst_size < copy_size ? dst_size : copy_size;

    memcpy(dst, src, min_size);

    return min_size;
}

static char * ggmlqnn_strndup(const char * source, size_t maxlen) {
#if defined(__ANDROID__) || defined(__linux__)
    return strndup(source, maxlen);
#else
    //FIXME:behaviour is not exactly same to Android&Linux
    GGML_UNUSED(maxlen);
    return strdup(source);
#endif
}

static void * ggmlqnn_host_malloc(size_t buffer_size, size_t page_size) {
    void * data = nullptr;
#if defined(__ANDROID__) || defined(__linux__)
    int result = posix_memalign((void **)&data, page_size, buffer_size);
    if (result != 0) {
        GGMLQNN_LOG_WARN("%s: error: posix_memalign failed\n", __func__);
        return nullptr;
    }
#else
    //GGMLQNN_LOG_DEBUG("buffer_size %d, page_size %d\n", buffer_size, page_size);
    data = ggmlqnn_malloc_aligned(buffer_size, page_size);
    if (nullptr == data) {
        GGMLQNN_LOG_WARN("%s: error: host_malloc failed\n", __func__);
    }
#endif

    return data;
}

static void ggmlqnn_load_cfg() {
    std::string cfg_filename = std::string(g_qnn_runtimelib_path) + std::string(g_qnn_cfgfilename);
    GGMLQNN_LOG_INFO("load ggml-qnn config from %s", cfg_filename.c_str());
    qnn_cfg qnncfg_instance;
    qnncfg_instance.load(cfg_filename);
    qnncfg_instance.dump([](const std::string & section, const std::string & key, const std::string value) {
        std::ostringstream  tmposs;
        tmposs << "section[" << section << "],[" << key << "] = [" << value << "]" << std::endl;
        GGMLQNN_LOG_INFO("%s", tmposs.str().c_str());
    });
    std::string npu_inference_datatype;
    qnncfg_instance.get_intvalue("general", "print_qnn_internal_log", g_print_qnn_internal_log, 0);
    qnncfg_instance.get_intvalue("general", "inference_approach", g_inference_approach, 0);
    qnncfg_instance.get_stringvalue("npu", "npu_inference_datatype", npu_inference_datatype, "fp32");
    GGMLQNN_LOG_INFO("print_qnn_internal_log=%d", g_print_qnn_internal_log);
    GGMLQNN_LOG_INFO("inference_approach=%d", g_inference_approach);
    GGMLQNN_LOG_INFO("npu inference data type=%s", npu_inference_datatype.c_str());
}

// =================================================================================================
//  section-6: QNN helper function
// =================================================================================================
static inline uint32_t get_qnn_tensorid(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.id;
    }

    return 0u;
}

static inline const char * get_qnn_tensorname(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.name;
    }
    return nullptr;
}

static inline Qnn_TensorType_t get_qnn_tensortype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.type;
    }
    return QNN_TENSOR_TYPE_UNDEFINED;
}

static inline Qnn_TensorDataFormat_t get_qnn_tensor_dataformat(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataFormat;
    }
    return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

static inline Qnn_DataType_t get_qnn_tensor_datatype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataType;
    }
    return QNN_DATATYPE_UNDEFINED;
}

static inline Qnn_QuantizeParams_t get_qnn_tensor_quantparams(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.quantizeParams;
    }
    return QNN_QUANTIZE_PARAMS_INIT;
}

static inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.rank;
    }
    return 0u;
}

static inline uint32_t * get_qnn_tensor_dimensions(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dimensions;
    }
    return nullptr;
}

static inline Qnn_TensorMemType_t get_qnn_tensor_memtype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.memType;
    }
    return QNN_TENSORMEMTYPE_UNDEFINED;
}

static inline void set_qnn_tensor_id(Qnn_Tensor_t & tensor, uint32_t id) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.id = id;
    }
}

static inline void set_qnn_tensor_name(Qnn_Tensor_t & tensor, const char * name) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.name = name;
    }
}

static inline void set_qnn_tensor_type(Qnn_Tensor_t & tensor, Qnn_TensorType_t type) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.type = type;
    }
}

static inline void set_qnn_tensor_dataformat(Qnn_Tensor_t & tensor, Qnn_TensorDataFormat_t format) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataFormat = format;
    }
}

static inline void set_qnn_tensor_datatype(Qnn_Tensor_t & tensor, Qnn_DataType_t dataType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataType = dataType;
    }
}

static inline void set_qnn_tensor_quantparams(Qnn_Tensor_t & tensor, Qnn_QuantizeParams_t params) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.quantizeParams = params;
    }
}

static inline void set_qnn_tensor_rank(Qnn_Tensor_t & tensor, uint32_t rank) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.rank = rank;
    }
}

static inline void set_qnn_tensor_dimensions(Qnn_Tensor_t & tensor, uint32_t * dims) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dimensions = dims;
    }
}

static inline void set_qnn_tensor_memtype(Qnn_Tensor_t & tensor, Qnn_TensorMemType_t memType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memType = memType;
    }
}

static inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t & tensor, Qnn_ClientBuffer_t clientBuf) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.clientBuf = clientBuf;
    }
}

static inline void set_qnn_tensor_memhandle(Qnn_Tensor_t & tensor, Qnn_MemHandle_t handle) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memHandle = handle;
    }
}

static int deep_copy_qnn_tensors(Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
    int err = 0;

    dst.version = src.version;
    QNN_TENSOR_SET_NAME(dst, ggmlqnn_strndup(QNN_TENSOR_GET_NAME(src), std::string(QNN_TENSOR_GET_NAME(src)).size()));
    if (nullptr == QNN_TENSOR_GET_NAME(dst)) {
        return 1;
    }
    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
    QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSOR_GET_MEM_TYPE(src));

    if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_RAW) {
        Qnn_ClientBuffer_t client_buf = {nullptr, 0};
        QNN_TENSOR_SET_CLIENT_BUF(dst, client_buf);
    } else if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
        QNN_TENSOR_SET_MEM_HANDLE(dst, nullptr);
    } else {
        return 1;
    }

    Qnn_QuantizeParams_t src_qparam      = QNN_TENSOR_GET_QUANT_PARAMS(src);
    Qnn_QuantizationEncoding_t encoding  = src_qparam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        Qnn_QuantizeParams_t src_qparam_cpy       = src_qparam;
        Qnn_AxisScaleOffset_t & axis_scale_offset = src_qparam_cpy.axisScaleOffsetEncoding;
        Qnn_ScaleOffset_t ** scale_offset         = &axis_scale_offset.scaleOffset;
        size_t scale_offset_size = axis_scale_offset.numScaleOffsets * sizeof(Qnn_ScaleOffset_t);
        *scale_offset            = (Qnn_ScaleOffset_t *)malloc(scale_offset_size);
        ggmlqnn_memscpy(*scale_offset,
                        scale_offset_size,
                        src_qparam.axisScaleOffsetEncoding.scaleOffset,
                        scale_offset_size);
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        Qnn_QuantizeParams_t src_qparam_cpy           = src_qparam;
        Qnn_BwAxisScaleOffset_t & bwaxis_scale_offset = src_qparam_cpy.bwAxisScaleOffsetEncoding;
        size_t scale_size                          = bwaxis_scale_offset.numElements * sizeof(float);
        float ** scales                            = &bwaxis_scale_offset.scales;
        int32_t ** offsets                         = &bwaxis_scale_offset.offsets;
        *scales                                    = (float *)malloc(scale_size);
        ggmlqnn_memscpy(*scales, scale_size, src_qparam.bwAxisScaleOffsetEncoding.scales, scale_size);

        if (bwaxis_scale_offset.offsets != nullptr) {
            size_t offset_size = bwaxis_scale_offset.numElements * sizeof(int32_t);
            *offsets           = (int32_t *)malloc(offset_size);
            ggmlqnn_memscpy(*offsets, offset_size, src_qparam.bwAxisScaleOffsetEncoding.offsets, offset_size);
        }
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else {
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam);
    }

    uint32_t rank = QNN_TENSOR_GET_RANK(src);
    QNN_TENSOR_SET_RANK(dst, rank);
    size_t dim_size       = GGML_MAX_DIMS * sizeof(uint32_t);
    uint32_t * dimensions = (uint32_t *)malloc(dim_size);
    if (nullptr == dimensions) {
        GGMLQNN_LOG_WARN("deep_copy_qnn_tensors() allocation error while copying tensor %s\n", QNN_TENSOR_GET_NAME(src));
        return 1;
    }
    ggmlqnn_memscpy(dimensions, dim_size, QNN_TENSOR_GET_DIMENSIONS(src), dim_size);
    QNN_TENSOR_SET_DIMENSIONS(dst, dimensions);

    return err;
}

static int free_qnn_tensor(Qnn_Tensor_t * tensor) {
    int err = 0;
    free((void *) QNN_TENSOR_GET_NAME(*tensor));
    Qnn_QuantizeParams_t src_qparam     = QNN_TENSOR_GET_QUANT_PARAMS(*tensor);
    Qnn_QuantizationEncoding_t encoding = src_qparam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        free(src_qparam.axisScaleOffsetEncoding.scaleOffset);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        free(src_qparam.bwAxisScaleOffsetEncoding.scales);
        if (src_qparam.bwAxisScaleOffsetEncoding.offsets != nullptr) {
            free(src_qparam.bwAxisScaleOffsetEncoding.offsets);
        }
    }
    free(QNN_TENSOR_GET_DIMENSIONS(*tensor));
    free(tensor);

    return err;
}

static const char * ggmlqnn_get_qnnerror_string(Qnn_ErrorHandle_t qnn_error_code) {
    // file:///opt/qcom/aistack/qairt/2.31.0.250130/docs/QNN/general/api_error_codes.html
    switch (qnn_error_code) {
        case QNN_SUCCESS:
            return "QNN_SUCCESS";
        case QNN_COMMON_ERROR_GENERAL:
            return "QNN_COMMON_ERROR_GENERAL";

            // QnnGraph_Error_t
        case QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE:
            return "QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE";
        case QNN_GRAPH_ERROR_MEM_ALLOC:
            return "QNN_GRAPH_ERROR_MEM_ALLOC";
        case QNN_GRAPH_ERROR_INVALID_ARGUMENT:
            return "QNN_GRAPH_ERROR_INVALID_ARGUMENT";
        case QNN_GRAPH_ERROR_INVALID_HANDLE:
            return "QNN_GRAPH_ERROR_INVALID_HANDLE";
        case QNN_GRAPH_ERROR_GRAPH_DOES_NOT_EXIST:
            return "QNN_GRAPH_ERROR_GRAPH_DOES_NOT_EXIST";
        case QNN_GRAPH_ERROR_INVALID_NAME:
            return "QNN_GRAPH_ERROR_INVALID_NAME";
        case QNN_GRAPH_ERROR_INVALID_TENSOR:
            return "QNN_GRAPH_ERROR_INVALID_TENSOR";
        case QNN_GRAPH_ERROR_INVALID_OP_CONFIG:
            return "QNN_GRAPH_ERROR_INVALID_OP_CONFIG";
        case QNN_GRAPH_ERROR_SET_PROFILE:
            return "QNN_GRAPH_ERROR_SET_PROFILE";
        case QNN_GRAPH_ERROR_UNCONNECTED_NODE:
            return "QNN_GRAPH_ERROR_UNCONNECTED_NODE";
        case QNN_GRAPH_ERROR_CREATE_FAILED:
            return "QNN_GRAPH_ERROR_CREATE_FAILED";
        case QNN_GRAPH_ERROR_OPTIMIZATION_FAILED:
            return "QNN_GRAPH_ERROR_OPTIMIZATION_FAILED";
        case QNN_GRAPH_ERROR_FINALIZE_FAILED:
            return "QNN_GRAPH_ERROR_FINALIZE_FAILED";
        case QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED:
            return "QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED";
        case QNN_GRAPH_ERROR_GRAPH_FINALIZED:
            return "QNN_GRAPH_ERROR_GRAPH_FINALIZED";
        case QNN_GRAPH_ERROR_EXECUTION_ASYNC_FIFO_FULL:
            return "QNN_GRAPH_ERROR_EXECUTION_ASYNC_FIFO_FULL";
        case QNN_GRAPH_ERROR_SIGNAL_IN_USE:
            return "QNN_GRAPH_ERROR_SIGNAL_IN_USE";
        case QNN_GRAPH_ERROR_ABORTED:
            return "QNN_GRAPH_ERROR_ABORTED";
        case QNN_GRAPH_ERROR_PROFILE_IN_USE:
            return "QNN_GRAPH_ERROR_PROFILE_IN_USE";
        case QNN_GRAPH_ERROR_TIMED_OUT:
            return "QNN_GRAPH_ERROR_TIMED_OUT";
        case QNN_GRAPH_ERROR_SUBGRAPH:
            return "QNN_GRAPH_ERROR_SUBGRAPH";
        case QNN_GRAPH_ERROR_DISABLED:
            return "QNN_GRAPH_ERROR_DISABLED";
        case QNN_GRAPH_ERROR_DYNAMIC_TENSOR_SHAPE:
            return "QNN_GRAPH_ERROR_DYNAMIC_TENSOR_SHAPE";
        case QNN_GRAPH_ERROR_TENSOR_SPARSITY:
            return "QNN_GRAPH_ERROR_TENSOR_SPARSITY";
        case QNN_GRAPH_ERROR_EARLY_TERMINATION:
            return "QNN_GRAPH_ERROR_EARLY_TERMINATION";
        case QNN_GRAPH_ERROR_INVALID_CONTEXT:
            return "QNN_GRAPH_ERROR_INVALID_CONTEXT";

            //QQnnTensor_Error_t
            //Invalid context/graph handle in creating tensor
        case QNN_TENSOR_ERROR_INVALID_HANDLE:
            return "QNN_TENSOR_ERROR_INVALID_HANDLE";
            //Tensor with specified credentials not registered with a context/graph
        case QNN_TENSOR_ERROR_DOES_NOT_EXIST:
            return "QNN_TENSOR_ERROR_DOES_NOT_EXIST";
            // (deprecated) Tensor has already been registered with backend
        case QNN_TENSOR_ERROR_ALREADY_EXISTS:
            return "QNN_TENSOR_ERROR_ALREADY_EXISTS";
            // Invalid tensor param.
        case QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM:
            return "QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM";
            // This tensor param is currently unsupported
        case QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM:
            return "QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM";
            // Tensor provided for update is invalid
        case QNN_TENSOR_ERROR_INCOMPATIBLE_TENSOR_UPDATE:
            return "QNN_TENSOR_ERROR_INCOMPATIBLE_TENSOR_UPDATE";

            // QnnOpPackage_Error_t
        case QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED:
            return "QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED";
        case QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED:
            return "QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED";
        case QNN_OP_PACKAGE_ERROR_INVALID_HANDLE:
            return "QNN_OP_PACKAGE_ERROR_INVALID_HANDLE";
        case QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE:
            return "QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE";
        case QNN_OP_PACKAGE_ERROR_INVALID_INFO:
            return "QNN_OP_PACKAGE_ERROR_INVALID_INFO";
        case QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE:
            return "QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE";
        case QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT:
            return "QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT";

        default:
            return "unknown QNN error";
    }
}

static Qnn_OpConfig_t ggmlqnn_create_op_config(const char * name, const char * package, const char * type,
                                       Qnn_Param_t * params, uint32_t num_params,
                                       Qnn_Tensor_t * inputs, uint32_t num_inputs,
                                       Qnn_Tensor_t * outputs, uint32_t num_outputs) {
    Qnn_OpConfigV1_t v1 = {name, package, type,
                           num_params, params,
                           num_inputs, inputs,
                           num_outputs, outputs
    };
    Qnn_OpConfig_t opcfg = {QNN_OPCONFIG_VERSION_1, {v1}};

    return opcfg;
}

// =================================================================================================
//  section-7:ggml-qnn backend helper function / class
// =================================================================================================
static const char * ggmlqnn_get_socmodel_desc(uint32_t soc_model) {
    switch (soc_model) {
        case SM7450:
            return "SM7450";
        case SM8350:
            return "SM8350";
        case SM8450:
            return "SM8450";
        case SM8475:
            return "SM8475";
        case SM8550:
            return "SM8550";
        case SM8650:
            return "SM8650";
        case SM8750:
            return "SM8750";
        default:
            return "unknown";
    }
}

static const char * ggmlqnn_get_htparch_desc(size_t htp_arch) {
    switch (htp_arch) {
        case V68:
            return "QCOM_HTP_V68";
        case V69:
            return "QCOM_HTP_V69";
        case V73:
            return "QCOM_HTP_V73";
        case V75:
            return "QCOM_HTP_V75";
        case V79:
            return "QCOM_HTP_V79";
        default:
            return "unknown";
    }
}

static struct qcom_socinfo * ggmlqnn_get_socinfo_from_socmodel(uint32_t soc_model) {
    size_t items = sizeof(g_qnn_soc_info_table) / sizeof(g_qnn_soc_info_table[0]);
    for (size_t idx = 0; idx < items; idx++) {
        if (soc_model == g_qnn_soc_info_table[idx].soc_model) {
            return &g_qnn_soc_info_table[idx];
        }
    }
    return nullptr;
}

static const char * ggmlqnn_get_ggml_type_name(ggml_type type) {
    const auto * traits = ggml_get_type_traits(type);
    return traits->type_name;
}

// ref:explanation of k-quants, https://github.com/ggerganov/llama.cpp/pull/1684
static Qnn_DataType_t ggmlqnn_datatype_from_ggml_datatype(enum ggml_type ggmltype) {
    switch (ggmltype) {
        case GGML_TYPE_F16:
            return QNN_DATATYPE_FLOAT_16;
        case GGML_TYPE_F32:
            return QNN_DATATYPE_FLOAT_32;
        case GGML_TYPE_I8:
            return QNN_DATATYPE_INT_8;
        case GGML_TYPE_Q8_0:
            return QNN_DATATYPE_SFIXED_POINT_8;
        case GGML_TYPE_Q4_0:
            return QNN_DATATYPE_SFIXED_POINT_4;
        default:
            break;
    }
    return QNN_DATATYPE_UNDEFINED;
}

static ggml_type ggml_datatype_from_qnn_datatype(Qnn_DataType_t qnn_type) {
    switch (qnn_type) {
        case QNN_DATATYPE_FLOAT_32:
            return GGML_TYPE_F32;
        case QNN_DATATYPE_FLOAT_16:
            return GGML_TYPE_F16;
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_INT_32:
            return GGML_TYPE_I32;
        case QNN_DATATYPE_INT_16:
            return GGML_TYPE_I16;
        case QNN_DATATYPE_INT_8:
            return GGML_TYPE_I8;
        case QNN_DATATYPE_SFIXED_POINT_8:
            return GGML_TYPE_Q8_0;
        case QNN_DATATYPE_SFIXED_POINT_4:
            return GGML_TYPE_Q4_0;
        default:
            break;
    }
    return GGML_TYPE_COUNT;
}

static void ggmlqnn_get_qnn_dimensions_from_ggml_dimensions(uint32_t * qnn_dimensions, const uint32_t * ggml_dimensions, uint32_t rank) {
    if (rank > GGML_MAX_DIMS) {
        GGMLQNN_LOG_WARN("invalid params");
        return;
    }
    if (nullptr == qnn_dimensions || nullptr == ggml_dimensions) {
        GGMLQNN_LOG_WARN("invalid params");
        return;
    }
    for (size_t idx = 0; idx < GGML_MAX_DIMS; idx++)
        qnn_dimensions[idx] = ggml_dimensions[idx];

    if (rank >= 2) {
        qnn_dimensions[rank - 1] = ggml_dimensions[rank - 2];
        qnn_dimensions[rank - 2] = ggml_dimensions[rank - 1];
    }
}

static void * ggmlqnn_type_trait(ggml_backend_qnn_context * ctx, ggml_tensor * op) {
    const ggml_tensor * src0        = op->src[0];
    const ggml_tensor * src1        = op->src[1];
    ggml_tensor * dst               = op;
    const enum ggml_type src0_type  = src0->type;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);
    GGML_ASSERT(nb00 == ggml_type_size(src0_type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    const int64_t ne_plane = ne01 * ne00;
    const size_t desired_size = ((GGML_TYPE_F32 == src0_type) ? 0 : ne03 * ne02 * ne_plane * sizeof(float));
    ctx->desired_size   = desired_size;
    if (ctx->work_size < desired_size) {
        ctx->work_data.reset(new char[desired_size]);
        ctx->work_size  = desired_size;
    }
    ctx->n_threads = std::thread::hardware_concurrency();
    void * wdata = ctx->work_data.get();
    // convert src0 to float
    if (src0_type != GGML_TYPE_F32) {
        const auto * type_traits        = ggml_get_type_traits(src0_type);
        ggml_to_float_t const to_float  = type_traits->to_float;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const void * x          = (char *)src0->data + i02 * nb02 + i03 * nb03;
                float * const wplane    = (float *)wdata + i02 * ne_plane + i03 * ne02 * ne_plane;

                const int min_cols_per_thread = 4096;
                const int min_rows_per_thread = std::max((int)(min_cols_per_thread / ne00), 1);
                const int n_threads = std::max(
                        std::min(ctx->n_threads, (int)(ne01 / min_rows_per_thread)), 1);
                for (int i = 1; i < n_threads; i++) {
                    const int64_t start = i * ne01 / n_threads;
                    const int64_t end   = (i + 1) * ne01 / n_threads;
                    if (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            for (int64_t i01 = start; i01 < end; i01++) {
                                to_float((const char *)x + i01 * nb01, wplane + i01 * ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    const int64_t start = 0;
                    const int64_t end = ne01 / n_threads;
                    for (int64_t i01 = start; i01 < end; i01++) {
                        to_float((const char *) x + i01 * nb01, wplane + i01 * ne00, ne00);
                    }
                }
            }
        }

        // wait for all tasks to finish
        for (auto &task: ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
    }
    return wdata;
}

static void ggmlqnn_append_tensor_dimensions(const ggml_tensor * tensor, std::string & output) {
    char buffer[256] = {};
    const char * type_name = ggmlqnn_get_ggml_type_name(tensor->type);
    int len = 0;
    switch (ggml_n_dims(tensor)) {
        case 1:
            len = snprintf(buffer, sizeof(buffer), "%ldx1%s", (long)tensor->ne[0], type_name);
            break;
        case 2:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1], type_name);
            break;
        case 3:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], type_name);
            break;
        case 4:
        default:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ldx%ld%s", (long)tensor->ne[0], (long)tensor->ne[1],
                           (long)tensor->ne[2], (long)tensor->ne[3], type_name);
            break;
    }
    GGML_ASSERT(len > 0 && len < (int)sizeof(buffer));
    output.append(buffer, len);
}

static size_t ggmlqnn_get_opcaps_size() {
    return std::size(ggmlqnn_k_op_caps);
}

static size_t ggmlqnn_get_op_index(const ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_UNARY) {
        return static_cast<size_t>(GGML_OP_COUNT) + static_cast<size_t>(ggml_get_unary_op(tensor));
    }

    return tensor->op;
}

static size_t ggmlqnn_get_op_input_param_count(const ggml_tensor * op) {
    auto op_index = ggmlqnn_get_op_index(op);
    GGML_ASSERT(op_index < std::size(ggmlqnn_k_op_caps));
    return ggmlqnn_k_op_caps[op_index].input_param_count;
}

static void ggmlqnn_get_graphkey_from_op(const ggml_tensor * op, std::string & output) {
    GGML_ASSERT(op->op != GGML_OP_NONE);
    output += ggml_op_desc(op);
    output += ggmlqnn_get_ggml_type_name(op->type);
    size_t param_count = ggmlqnn_get_op_input_param_count(op);
    for (size_t i = 0; i < param_count; ++i) {
        auto * input = op->src[i];
        if (!input) {
            break;
        }
        output += '_';
        ggmlqnn_append_tensor_dimensions(input, output);
    }
}

static void ggmlqnn_get_opkey_with_srcop_desc(const ggml_tensor * op, std::string & output) {
    output += ggml_op_desc(op);
    output += '(';
    if (op->src[0]) {
        output += ggml_op_desc(op->src[0]);
    }
    for (size_t i = 1; i < GGML_MAX_DIMS && op->src[i]; ++i) {
        output += ',';
        output += ggml_op_desc(op->src[i]);
    }
    output += ')';
}

static void ggmlqnn_get_graphkey_from_cgraph(const ggml_cgraph * cgraph, std::string & output) {
    if (nullptr == cgraph || 0 == cgraph->n_nodes) {
        GGMLQNN_LOG_WARN("empty ggml computational graph");
        return;
    }

    bool is_start = true;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        if (ggml_is_empty(op)) {
            GGMLQNN_LOG_WARN("empty op in graph, skipping");
            continue;
        }

        if (op->op == GGML_OP_NONE) {
            GGMLQNN_LOG_WARN("GGML_OP_NONE in graph, skipping");
            continue;
        }

        if (is_start) {
            ggmlqnn_get_graphkey_from_op(cgraph->nodes[0], output);
            is_start = false;
        } else {
            output += '#';
            ggmlqnn_get_opkey_with_srcop_desc(op, output);
        }
    }

    if (cgraph->n_nodes > 1) {
        auto * last_op = cgraph->nodes[cgraph->n_nodes - 1];
        output += ggmlqnn_get_ggml_type_name(last_op->type);
        output += '_';
        ggmlqnn_append_tensor_dimensions(last_op, output);
    }
}

template<typename Fn>
Fn ggmlqnn_load_qnn_functionpointers(void * handle, const char * function_name) {
    return reinterpret_cast<Fn>(dlsym(handle, function_name));
}

class qnn_interface {
#define DEFINE_SHIM_FUNCTION_INTERFACE(F, pointer_name)           \
  template <typename... Args>                                     \
  inline auto qnn_##F(Args... args) const {                       \
    return (_qnn_interface->QNN_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                             \
  }


#define DEFINE_SHIM_FUNCTION_SYS_INTERFACE(F, pointer_name)                  \
  template <typename... Args>                                                \
  inline auto qnn_##F(Args... args) const {                                  \
    return (_qnn_sys_interface->QNN_SYSTEM_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                                        \
  }

    friend class qnn_instance;

public:
    qnn_interface() = default;

    // QnnBackend
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_create, backendCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_free, backendFree)

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_register_op_package, backendRegisterOpPackage)

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_validate_op_config, backendValidateOpConfig)

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_get_api_version, backendGetApiVersion)

    // QnnDevice
    DEFINE_SHIM_FUNCTION_INTERFACE(device_create, deviceCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(device_free, deviceFree)

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_infrastructure, deviceGetInfrastructure)

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_platform_info, deviceGetPlatformInfo)

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_info, deviceGetInfo)

    // QnnContext
    DEFINE_SHIM_FUNCTION_INTERFACE(context_create, contextCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary_size, contextGetBinarySize)

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary, contextGetBinary)

    DEFINE_SHIM_FUNCTION_INTERFACE(context_create_from_binary, contextCreateFromBinary)

    DEFINE_SHIM_FUNCTION_INTERFACE(context_free, contextFree)

    // QnnGraph
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_create, graphCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_add_node, graphAddNode)

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_finalize, graphFinalize)

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_execute, graphExecute)

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_retrieve, graphRetrieve)

    // QnnLog
    DEFINE_SHIM_FUNCTION_INTERFACE(log_create, logCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(log_free, logFree)

    DEFINE_SHIM_FUNCTION_INTERFACE(log_set_log_level, logSetLogLevel)

    // QnnProfile
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_create, profileCreate)

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_events, profileGetEvents)

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_sub_events, profileGetSubEvents)

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_event_data, profileGetEventData)

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_free, profileFree)

    // QnnMem
    DEFINE_SHIM_FUNCTION_INTERFACE(mem_register, memRegister)

    DEFINE_SHIM_FUNCTION_INTERFACE(mem_de_register, memDeRegister)

    // QnnProperty
    DEFINE_SHIM_FUNCTION_INTERFACE(property_has_capability, propertyHasCapability)

    // QnnTensor
    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_context_tensor, tensorCreateContextTensor)

    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_graph_tensor, tensorCreateGraphTensor)

    // QnnSystem
    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_create, systemContextCreate)

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_get_binary_info, systemContextGetBinaryInfo)

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree)

    void set_qnn_interface(const QnnInterface_t * qnn_interface) {
        _qnn_interface = qnn_interface;
    }

    void set_qnn_system_interface(const QnnSystemInterface_t * qnn_sys_interface) {
        _qnn_sys_interface = qnn_sys_interface;
    }

    uint32_t get_backend_id() const {
        return _qnn_interface->backendId;
    }

    bool is_loaded() const {
        return ((_qnn_sys_interface != nullptr) && (_qnn_interface != nullptr));
    }

private:
    const QnnInterface_t * _qnn_interface           = nullptr;

    const QnnSystemInterface_t * _qnn_sys_interface = nullptr;
};

class qnn_instance {
public:
    using BackendIdType = decltype(QnnInterface_t{}.backendId);

    explicit qnn_instance(const std::string & lib_path, const std::string & backend_name,
                          const std::string & model_name) :
            _lib_path(std::move(lib_path)),
            _backend_name(std::move(backend_name)),
            _model_name(std::move(model_name)) {}

    ~qnn_instance() {
    }

    int qnn_init(const QnnSaver_Config_t ** saver_config);

    int qnn_finalize();

    const qnn_interface & get_qnn_interface() {
        if (!_qnn_interface.is_loaded()) {
            GGMLQNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_interface;
    }

    const QNN_INTERFACE_VER_TYPE & get_qnn_raw_interface() {
        if (!_qnn_interface.is_loaded()) {
            GGMLQNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_interface;
    }

    const QNN_SYSTEM_INTERFACE_VER_TYPE & get_qnn_raw_system_interface() {
        if (!_qnn_interface.is_loaded()) {
            GGMLQNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_system_interface;
    }

    Qnn_LogHandle_t get_qnn_log_handle() { return _qnn_log_handle; }

    Qnn_ProfileHandle_t get_qnn_profile_handle() { return _qnn_profile_handle; }

    Qnn_DeviceHandle_t get_qnn_device_handle() { return _qnn_device_handle; }

    Qnn_BackendHandle_t get_qnn_backend_handle() { return _qnn_backend_handle; }

    Qnn_ContextHandle_t get_qnn_context_handle() { return _qnn_context_handle; }

    QnnSystemContext_Handle_t get_qnn_system_handle() { return _qnn_system_handle; }

    Qnn_GraphHandle_t get_qnn_graph_handle() { return _qnn_graph_handle; }

    int init_qnn_graph(const char * graph_name,
                       bool debug,
                       uint8_t do_node_validation = 1,
                       const QnnGraph_Config_t ** graph_configs = nullptr
    );
    int init_qnn_graph(const std::string & graph_name, QNNBackend device, size_t vtcm_size_in_mb = 8, size_t hvx_threads = 8);

    int finalize_qnn_graph();

    bool is_valid_graph() const { return _qnn_graph_handle != nullptr; }

    int init_htp_perfinfra();

    int set_rpc_polling();

    int set_high_performance_mode();

    std::string & get_qnn_graph_name() { return _graph_name; }

    bool is_rpcmem_initialized() {
        return _rpcmem_initialized;
    }

    void set_rpcmem_initialized(bool initialized) {
        _rpcmem_initialized = initialized;
    }

    size_t get_rpcmem_capacity() { return _rpcmem_capacity; }
    size_t get_rpcmem_usage() { return _rpcmem_usage; }

    int32_t rpcmem_to_fd(void * buf);

    int register_rpcmem(void * p_data, Qnn_Tensor_t * p_tensor);
    Qnn_MemHandle_t  register_rpcmem(void * p_data, const uint32_t rank, uint32_t * dimensions, Qnn_DataType_t data_type);

    void unregister_rpcmem();
    void unregister_rpcmem(Qnn_MemHandle_t mem_handle);

    void * alloc_rpcmem(size_t bytes, size_t alignment);
    void * get_rpcmem_from_memhandle(Qnn_MemHandle_t mem_handle);

    void free_rpcmem(void * buf);
    void free_rpcmem();

    bool is_rpcmem_allocated(void * buf);

    bool is_rpcmem_registered(Qnn_MemHandle_t handle) {
        return _qnn_mem_set.count(handle) != 0U;
    }

    bool enable_qnn_rpc() {
        return _enable_qnn_rpc;
    }

    QNNBackend get_device_id() {
        return _device_id;
    }

public:
    std::map<std::string, std::tuple<Qnn_GraphHandle_t, std::vector< Qnn_Tensor_t *>>> _qnn_graph_map;

private:
    int load_system();

    int unload_system();

    int load_backend(std::string & lib_path, const QnnSaver_Config_t ** saver_config);

    int unload_backend();

    void set_qnn_raw_interface(QNN_INTERFACE_VER_TYPE & raw_interface) {
        _qnn_raw_interface = raw_interface;
    }

    void set_qnn_raw_system_interface(QNN_SYSTEM_INTERFACE_VER_TYPE & raw_interface) {
        _qnn_raw_system_interface = raw_interface;
    }

    void * alloc_rpcmem_internal(size_t bytes, size_t alignment);

    void htp_print_info();

    void htp_probe_device_meminfo();

    void print_backend_info();

    void htp_set_memory_grow_size(size_t size = 1ul * 1024 * 1024);

    void htp_enter_performance_mode();

    void htp_set_n_hvx_threads(size_t n_threads);

private:
    static constexpr const int _required_num_providers = 1;

private:
    std::string     _lib_path;
    std::string     _backend_name;
    std::string     _model_name; // name of prebuilt QNN model, might be used in the future
    BackendIdType   _backend_id;

    bool _debug_tensor                      = false; // flag to indicate if requested graph is to be run in debug mode
    bool _do_node_validations               = true;  // flag to indicate whether all add_node calls need to be validated
    QnnLog_Level_t _qnn_log_level           = QNN_LOG_LEVEL_DEBUG;

    ggml_qnn_profile_level _profile_level   = ggml_qnn_profile_level::profile_detail;

    void * _system_lib_handle               = nullptr;
    void * _loaded_lib_handle               = nullptr;
    const QnnInterface_t * _loaded_backend  = nullptr;

    Qnn_GraphHandle_t _qnn_graph_handle     = nullptr;

    Qnn_LogHandle_t _qnn_log_handle         = nullptr;

    Qnn_ProfileHandle_t _qnn_profile_handle = nullptr;

    Qnn_DeviceHandle_t _qnn_device_handle   = nullptr;

    Qnn_BackendHandle_t _qnn_backend_handle = nullptr;

    Qnn_ContextHandle_t _qnn_context_handle = nullptr;

    QnnSystemContext_Handle_t _qnn_system_handle = nullptr;

    QnnHtpDevice_PerfInfrastructure_t * _qnn_htp_perfinfra = nullptr;
    uint32_t _qnn_htp_powerconfig_id  = 1;
    uint32_t _qnn_htp_device_id       = 0;
    uint32_t _qnn_htp_core_id         = 0;

    uint32_t _qnn_rpc_pollingtime     = 9999; // 0-10000 us for high performing

    qnn_interface _qnn_interface;
    QNN_INTERFACE_VER_TYPE _qnn_raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE _qnn_raw_system_interface;

    std::unordered_map<void *, Qnn_MemHandle_t> _qnn_mem_set;
    std::unordered_map<void *, Qnn_MemHandle_t> _qnn_rpc_buffer_to_handles;

    std::atomic_bool _rpcmem_initialized{false};
    pfn_rpc_mem_alloc _pfn_rpc_mem_alloc;
    pfn_rpc_mem_free _pfn_rpc_mem_free;
    pfn_rpc_mem_to_fd _pfn_rpc_mem_to_fd;
    pfn_rpc_mem_init  _pfn_rpc_mem_init;
    pfn_rpc_mem_deinit _pfn_rpc_mem_deinit;
    std::unordered_map<void *, void *> _rpcmem_store_map;
    std::unordered_map<void *, size_t> _rpcmem_usage_map;
    size_t                             _rpcmem_usage    = 0;   // mempool usage in Mbytes
    size_t                             _rpcmem_capacity = 512; // mempool size  in Mbytes

    std::string _graph_name;
    QNNBackend _device_id;
    void * _rpc_lib_handle      = nullptr;
    bool       _enable_qnn_rpc  = false; //TODO:unknown issue with QNN RPC feature

    DISABLE_COPY(qnn_instance);
    DISABLE_MOVE(qnn_instance);
};

void * qnn_instance::alloc_rpcmem_internal(size_t bytes, size_t alignment) {
    if (!_rpcmem_initialized) {
        GGMLQNN_LOG_WARN("rpc memory not initialized\n");
        return nullptr;
    }

    auto allocate_bytes = static_cast<int32_t>(bytes + alignment);
    void * buf = _pfn_rpc_mem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocate_bytes);
    if (nullptr == buf) {
        GGMLQNN_LOG_WARN("failed to allocate rpc memory\n");
        return nullptr;
    }

    auto aligned_buf = reinterpret_cast<void *>(ggmlqnn_align_to(alignment,
                                                reinterpret_cast<intptr_t>(buf)));
    bool status = _rpcmem_store_map.insert(std::pair<void *, void *>(aligned_buf, buf)).second;
    if (!status) {
        GGMLQNN_LOG_WARN("failed to allocate rpc memory\n");
        _pfn_rpc_mem_free(buf);
    }
    return aligned_buf;
}

void * qnn_instance::alloc_rpcmem(size_t bytes, size_t alignment) {
    if (_rpcmem_usage > (_rpcmem_capacity - 8)) { // reserve 8Mbytes in rpc mempool
        GGMLQNN_LOG_WARN("rpc mempool capcaity: %d MB, usage: %d MB", _rpcmem_capacity, _rpcmem_usage);
        return nullptr;
    }

    auto aligned_buf = alloc_rpcmem_internal(bytes, alignment);
    if (nullptr == aligned_buf)
        return nullptr;
    _rpcmem_usage_map.insert(std::pair<void *, size_t>(aligned_buf, bytes));

    size_t rpcmem_usage_in_bytes = _rpcmem_usage * (1 << 20);
    rpcmem_usage_in_bytes += bytes;
    _rpcmem_usage = rpcmem_usage_in_bytes / ( 1 << 20);
    return aligned_buf;
}

void qnn_instance::free_rpcmem(void * buf) {
    size_t rpcbuffer_size = 0;
    if (!_rpcmem_initialized) {
        GGMLQNN_LOG_WARN("rpc memory not initialized\n");
    } else if (0 == _rpcmem_store_map.count(buf)) {
        GGMLQNN_LOG_WARN("no allocated tensor\n");
    } else {
        GGMLQNN_LOG_DEBUG("free rpc mem %p", _rpcmem_store_map[buf]);
        for (std::unordered_map<void *, size_t>::iterator it = _rpcmem_usage_map.begin();
             it != _rpcmem_usage_map.end();
             it++) {
            void * rpcbuffer = it->first;
            if (buf == rpcbuffer) {
                rpcbuffer_size = it->second;
                size_t rpcmem_usage_in_bytes = _rpcmem_usage * (1 << 20);
                rpcmem_usage_in_bytes -= rpcbuffer_size;
                _rpcmem_usage = rpcmem_usage_in_bytes / ( 1 << 20);
            }
        }
        if (rpcbuffer_size != 0) {
            _rpcmem_usage_map.erase(buf);
        } else {
            GGMLQNN_LOG_WARN("it shouldn't happen, pls check why?");
        }
        _pfn_rpc_mem_free(_rpcmem_store_map[buf]);
        _rpcmem_store_map.erase(buf);
    }
}

void qnn_instance::free_rpcmem() {
    if (_rpcmem_store_map.empty()) {
        GGMLQNN_LOG_WARN("no rpcmem allocated\n");
        return;
    }

    for (std::unordered_map<void *, void *>::iterator it = _rpcmem_store_map.begin();
         it != _qnn_mem_set.end();
         it++) {
        void * rpcbuffer = it->second;
        GGMLQNN_LOG_DEBUG("free rpc buffer %p", rpcbuffer);
        _pfn_rpc_mem_free(rpcbuffer);
    }
    _rpcmem_store_map.clear();
    _rpcmem_usage_map.clear();
    _rpcmem_usage = 0;
}

int32_t qnn_instance::rpcmem_to_fd(void * buf) {
    int32_t mem_fd = -1;
    if (!is_rpcmem_initialized()) {
        GGMLQNN_LOG_WARN("rpc memory not initialized\n");
    } else {
        mem_fd = _pfn_rpc_mem_to_fd(buf);
    }

    return mem_fd;
}

int qnn_instance::register_rpcmem(void * p_data, Qnn_Tensor_t * p_tensor) {
    if (nullptr == p_data || (nullptr == p_tensor)) {
        GGMLQNN_LOG_WARN("invalid param\n");
        return 1;
    }

    if (!is_rpcmem_initialized()) {
        GGMLQNN_LOG_WARN("rpc memory not initialized\n");
        return 2;
    }

    if (is_rpcmem_registered((QNN_VER_PTR(*p_tensor)->memHandle))) {
        GGMLQNN_LOG_WARN("tensor %s has been registered shared memory\n", (QNN_VER_PTR(*p_tensor)->name));
        return 3;
    }

    int32_t mem_fd = rpcmem_to_fd(p_data);
    if (-1 == mem_fd) {
        GGMLQNN_LOG_WARN("failed to get file descriptor\n");
        return 4;
    }
    GGMLQNN_LOG_DEBUG("mem_fd %d\n", mem_fd);
    Qnn_MemDescriptor_t descriptor = {
            {QNN_VER_PTR(*p_tensor)->rank, QNN_VER_PTR(*p_tensor)->dimensions, nullptr},
            QNN_VER_PTR(*p_tensor)->dataType,
            QNN_MEM_TYPE_ION,
            {{mem_fd}}};
    Qnn_MemHandle_t handle = nullptr;
    int error = QNN_SUCCESS;
    error = _qnn_interface.qnn_mem_register(
            _qnn_context_handle,
            &descriptor,
            /*numDescriptors=*/1,
            &handle);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to register shared memory, error %d, %s\n", QNN_GET_ERROR_CODE(error), strerror(error));
        return 5;
    } else {
        GGMLQNN_LOG_INFO("tensor %s successfully register shared memory\n", (QNN_VER_PTR(*p_tensor)->name));
    }
    QNN_VER_PTR(*p_tensor)->memHandle = handle;
    _qnn_mem_set.insert((std::pair<void*, Qnn_MemHandle_t>(p_data, handle)));

    return 0;
}

Qnn_MemHandle_t  qnn_instance::register_rpcmem(void * p_data, const uint32_t rank, uint32_t * dimensions, Qnn_DataType_t data_type) {
    if (!p_data) {
        GGMLQNN_LOG_WARN("invalid param");
        return nullptr;
    }

    if (!is_rpcmem_initialized()) {
        GGMLQNN_LOG_WARN("rpc memory not initialized");
        return nullptr;
    }

    if (is_rpcmem_registered(p_data)) {
        GGMLQNN_LOG_WARN("rpc memory already registered");
        return _qnn_rpc_buffer_to_handles[p_data];
    }

    auto mem_fd = rpcmem_to_fd(p_data);
    if (mem_fd == -1) {
        GGMLQNN_LOG_WARN("failed to get file descriptor");
        return nullptr;
    }

    GGMLQNN_LOG_DEBUG("mem_fd %d", mem_fd);
    Qnn_MemDescriptor_t descriptor = {
            {rank, dimensions, nullptr},
            data_type, QNN_MEM_TYPE_ION,
            {{mem_fd}}
    };
    Qnn_MemHandle_t handle = nullptr;
    auto error = _qnn_interface.qnn_mem_register(_qnn_context_handle, &descriptor, /*numDescriptors=*/1, &handle);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to register shared memory, error %d, %s", QNN_GET_ERROR_CODE(error), strerror(error));
        return nullptr;
    }

    _qnn_rpc_buffer_to_handles.insert({p_data, handle});
    GGMLQNN_LOG_DEBUG("successfully register shared memory handler: %p", handle);
    return handle;
}

void * qnn_instance::get_rpcmem_from_memhandle(Qnn_MemHandle_t mem_handle) {
    for (std::unordered_map<void *, Qnn_MemHandle_t>::iterator it = _qnn_mem_set.begin();
         it != _qnn_mem_set.end();
         it++) {
        Qnn_MemHandle_t mem_handle = it->second;
        if (it->second == mem_handle) {
            return it->first;
        }
    }
    GGMLQNN_LOG_WARN("can't find rpcmem from qnn mem handle %p", mem_handle);
    return nullptr;
}

void qnn_instance::unregister_rpcmem() {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (_qnn_mem_set.empty()) {
        GGMLQNN_LOG_WARN("no rpcmem registered\n");
    }

    for (std::unordered_map<void *, Qnn_MemHandle_t>::iterator it = _qnn_mem_set.begin();
         it != _qnn_mem_set.end();
         it++) {
        Qnn_MemHandle_t mem_handle = it->second;
        error = _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to unregister shared memory, error %d\n", QNN_GET_ERROR_CODE(error));
        } else {
            GGMLQNN_LOG_DEBUG("unregister shared memory ok");
        }
    }
    _qnn_mem_set.clear();
}

void qnn_instance::unregister_rpcmem(Qnn_MemHandle_t mem_handle) {
    Qnn_ErrorHandle_t error = _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to unregister shared memory, error %d", QNN_GET_ERROR_CODE(error));
    }

    auto it = std::find_if(_qnn_mem_set.begin(), _qnn_mem_set.end(),
                           [mem_handle](const auto &kv) { return kv.second == mem_handle; });
    if (it == _qnn_mem_set.end()) {
        GGMLQNN_LOG_WARN("failed to find shared memory handler: %p", mem_handle);
        return;
    }

    _qnn_mem_set.erase(it);
}

bool qnn_instance::is_rpcmem_allocated(void * buf) {
    return _rpcmem_store_map.count(buf) != 0U;
}

int qnn_instance::load_backend(std::string & lib_path, const QnnSaver_Config_t ** saver_config) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    GGMLQNN_LOG_DEBUG("lib_path:%s\n", lib_path.c_str());

    void * lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (nullptr == lib_handle) {
        GGMLQNN_LOG_WARN("can not open QNN library %s, with error: %s", lib_path.c_str(), dlerror());
        return 1;
    }

    auto get_providers = ggmlqnn_load_qnn_functionpointers<_pfn_QnnInterface_getProviders *>(
                               lib_handle,
                               "QnnInterface_getProviders");
    if (nullptr == get_providers) {
        GGMLQNN_LOG_WARN("can not load symbol QnnInterface_getProviders : %s", dlerror());
        return 2;
    }

    std::uint32_t num_providers = 0;
    const QnnInterface_t ** provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to get providers, error %d", QNN_GET_ERROR_CODE(error));
        return 3;
    }
    GGMLQNN_LOG_DEBUG("num_providers=%d\n", num_providers);
    if (num_providers != _required_num_providers) {
        GGMLQNN_LOG_WARN("providers is %d instead of required %d", num_providers, _required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        GGMLQNN_LOG_WARN("failed to get qnn interface providers\n");
        return 5;
    }
    bool found_valid_interface = false;
    QNN_INTERFACE_VER_TYPE qnn_interface;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_API_VERSION_MAJOR == provider_list[idx]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= provider_list[idx]->apiVersion.coreApiVersion.minor) {
            found_valid_interface = true;
            qnn_interface = provider_list[idx]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }

    if (!found_valid_interface) {
        GGMLQNN_LOG_WARN("unable to find a valid qnn interface\n");
        return 6;
    } else {
        GGMLQNN_LOG_INFO("find a valid qnn interface\n");
    }
    set_qnn_raw_interface(qnn_interface);

    BackendIdType backend_id = provider_list[0]->backendId;
    _loaded_backend     = provider_list[0];
    _loaded_lib_handle  = lib_handle;
    _backend_id         = backend_id;

    auto saver_initialize =
            ggmlqnn_load_qnn_functionpointers<_pfn_QnnSaver_initialize *>(_loaded_lib_handle, "QnnSaver_initialize");
    if (nullptr != saver_initialize) {
        error = saver_initialize(saver_config);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to saver_initialize，error %d", QNN_GET_ERROR_CODE(error));
            return 7;
        }
    } else {
        GGMLQNN_LOG_WARN("saver_initialize is null\n");
    }

    return 0;
}

int qnn_instance::unload_backend() {
    int dlclose_error = 0;
    dlclose_error = dlclose(_loaded_lib_handle);
    if (dlclose_error != 0) {
        GGMLQNN_LOG_WARN("failed to close QNN backend %d, error %s\n", _backend_id, dlerror());
    }

    return 0;
}

int qnn_instance::load_system() {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

#if !defined(__ANDROID__) && !defined(__linux__)
    std::string system_lib_path = _lib_path + "QnnSystem.dll";
#else
    std::string system_lib_path = _lib_path + "libQnnSystem.so";
#endif
    GGMLQNN_LOG_DEBUG("system_lib_path:%s\n", system_lib_path.c_str());

    _system_lib_handle = dlopen(system_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (nullptr == _system_lib_handle) {
        GGMLQNN_LOG_WARN("can not open QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
        //re-try with default path of QNN binary runtime lib
        _lib_path = std::string(g_qnn_runtimelib_path);
#if !defined(__ANDROID__) && !defined(__linux__)
        system_lib_path = _lib_path + "QnnSystem.dll";
#else
        system_lib_path = _lib_path + "libQnnSystem.so";
#endif
        _system_lib_handle = dlopen(system_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (nullptr == _system_lib_handle) {
            GGMLQNN_LOG_WARN("can not open QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
            return 1;
        }
    }

    auto * get_providers = reinterpret_cast<_pfn_QnnSystemInterface_getProviders *>(dlsym(
            _system_lib_handle, "QnnSystemInterface_getProviders"));
    if (nullptr == get_providers) {
        GGMLQNN_LOG_WARN("can not load QNN symbol QnnSystemInterface_getProviders: %s\n", dlerror());
        return 2;
    }

    uint32_t num_providers = 0;
    const QnnSystemInterface_t ** provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to get providers, error %d\n", QNN_GET_ERROR_CODE(error));
        return 3;
    }

    if (num_providers != _required_num_providers) {
        GGMLQNN_LOG_WARN("providers is %d instead of required %d\n", num_providers, _required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        GGMLQNN_LOG_WARN("can not get providers\n");
        return 5;
    }

    QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface;
    bool found_valid_system_interface = false;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_SYSTEM_API_VERSION_MAJOR ==
            provider_list[idx]->systemApiVersion.major &&
            QNN_SYSTEM_API_VERSION_MINOR <=
            provider_list[idx]->systemApiVersion.minor) {
            found_valid_system_interface = true;
            qnn_system_interface = provider_list[idx]->QNN_SYSTEM_INTERFACE_VER_NAME;
            break;
        }
    }
    if (!found_valid_system_interface) {
        GGMLQNN_LOG_WARN("unable to find a valid qnn system interface\n");
        return 6;
    } else {
        GGMLQNN_LOG_INFO("find a valid qnn system interface\n");
    }
    set_qnn_raw_system_interface(qnn_system_interface);

    _qnn_interface.set_qnn_system_interface(provider_list[0]);

    _qnn_interface.qnn_system_context_create(&_qnn_system_handle);
    if (nullptr == _qnn_system_handle) {
        GGMLQNN_LOG_WARN("can not create QNN system contenxt\n");
    } else {
        GGMLQNN_LOG_INFO("initialize qnn system successfully\n");
    }

    return 0;
}

int qnn_instance::unload_system() {
    int result = 0;

    if (nullptr == _system_lib_handle) {
        GGMLQNN_LOG_DEBUG("system lib handle is null\n");
        return 1;
    }

    if (nullptr != _qnn_system_handle) {
        result = _qnn_interface.qnn_system_context_free(_qnn_system_handle);
        if (result != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN system context\n");
        }
        _qnn_system_handle = nullptr;
    }

    int dlclose_error = dlclose(_system_lib_handle);
    if (dlclose_error != 0) {
        GGMLQNN_LOG_WARN("failed to close QnnSystem library, error %s\n", dlerror());
        return 2;
    }

    _system_lib_handle = nullptr;

    return result;
}

static void ggml_qnn_logcallback(const char * fmt,
                                 QnnLog_Level_t level,
                                 uint64_t timestamp,
                                 va_list argp) {

    if (0 == g_print_qnn_internal_log)
        return;

    static std::mutex log_mutex;
    static unsigned char s_ggml_qnn_logbuf[GGML_QNN_LOGBUF_LEN];

    const char * log_level_desc = "";
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            log_level_desc = " ERROR ";
            break;
        case QNN_LOG_LEVEL_WARN:
            log_level_desc = "WARNING";
            break;
        case QNN_LOG_LEVEL_INFO:
            log_level_desc = "  INFO ";
            break;
        case QNN_LOG_LEVEL_DEBUG:
            log_level_desc = " DEBUG ";
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            log_level_desc = "VERBOSE";
            break;
        case QNN_LOG_LEVEL_MAX:
            log_level_desc = "UNKNOWN";
            break;
    }

    double ms = (double) timestamp / 1000000.0;
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        memset(s_ggml_qnn_logbuf, 0, GGML_QNN_LOGBUF_LEN);
        vsnprintf(reinterpret_cast<char *const>(s_ggml_qnn_logbuf), GGML_QNN_LOGBUF_LEN, fmt, argp);
        GGMLQNN_LOG_DEBUG("%8.1fms [%-7s] %s\n", ms, log_level_desc, s_ggml_qnn_logbuf);
    }
}

int qnn_instance::qnn_init(const QnnSaver_Config_t ** saver_config) {
    BackendIdType backend_id = QNN_BACKEND_ID_NULL;
    GGMLQNN_LOG_DEBUG("enter qni_init\n");

    _device_id = QNN_BACKEND_GGML;
    if (_backend_name.find("QnnCpu") != std::string::npos) {
        _device_id = QNN_BACKEND_CPU;
    }
    if (_backend_name.find("QnnGpu") != std::string::npos) {
        _device_id = QNN_BACKEND_GPU;
    }
    if (_backend_name.find("QnnHtp") != std::string::npos) {
        _device_id = QNN_BACKEND_NPU;
    }
    if (QNN_BACKEND_GGML == _device_id) {
        GGMLQNN_LOG_INFO("user specified qnn backend is ggml, skip QNN initialize");
        return 0;
    }

    if (0 != load_system()) {
        GGMLQNN_LOG_WARN("can not load QNN system lib, pls check why?\n");
        return 1;
    } else {
        GGMLQNN_LOG_DEBUG("load QNN system lib successfully\n");
    }

    std::string backend_lib_path = _lib_path + _backend_name;

    int is_load_ok = load_backend(backend_lib_path, saver_config);
    if (0 != is_load_ok) {
        GGMLQNN_LOG_WARN("failed to load QNN backend\n");
        return 2;
    }

    _qnn_interface.set_qnn_interface(_loaded_backend);
#if 1
    _qnn_interface.qnn_log_create(ggml_qnn_logcallback, _qnn_log_level, &_qnn_log_handle);
#else
    _qnn_raw_interface.logCreate(ggml_qnn_logcallback, _qnn_log_level, &_qnn_log_handle);
#endif
    if (nullptr == _qnn_log_handle) {
        GGMLQNN_LOG_WARN("why failed to initialize qnn log\n"); //NPU backend not work on Qualcomm SoC based low-end phone
        return 3;
    } else {
        GGMLQNN_LOG_DEBUG("initialize qnn log successfully\n");
    }

    std::vector<const QnnBackend_Config_t *> temp_backend_config;
    _qnn_interface.qnn_backend_create(_qnn_log_handle,
                      temp_backend_config.empty() ? nullptr : temp_backend_config.data(),
                      &_qnn_backend_handle);
    if (nullptr == _qnn_backend_handle) {
        GGMLQNN_LOG_WARN("why failed to initialize qnn backend\n");
        return 4;
    } else {
        GGMLQNN_LOG_DEBUG("initialize qnn backend successfully\n");
    }

    if (nullptr != _qnn_raw_interface.propertyHasCapability) {
        auto qnnstatus = _qnn_raw_interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnstatus) {
            GGMLQNN_LOG_WARN("device property is not supported\n");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnstatus) {
            GGMLQNN_LOG_WARN("device property is not known to backend\n");
        }
    }

    auto qnnstatus = _qnn_raw_interface.deviceCreate(
            _qnn_log_handle, nullptr, &_qnn_device_handle);
    if (QNN_SUCCESS != qnnstatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnstatus) {
        GGMLQNN_LOG_WARN("failed to create QNN device\n");
    } else {
        GGMLQNN_LOG_INFO("create device successfully\n");
    }

    if (ggml_qnn_profile_level::profile_off != _profile_level) {
        GGMLQNN_LOG_INFO("profiling turned on; level = %d", _profile_level);
        if (ggml_qnn_profile_level::profile_basic == _profile_level) {
            GGMLQNN_LOG_INFO("basic profiling requested. creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_BASIC, &_qnn_profile_handle)) {
                GGMLQNN_LOG_WARN("unable to create profile handle in the backend\n");
                return 5;
            } else {
                GGMLQNN_LOG_DEBUG("initialize qnn profile successfully\n");
            }
        } else if (ggml_qnn_profile_level::profile_detail == _profile_level) {
            GGMLQNN_LOG_INFO("detailed profiling requested. Creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_DETAILED, &_qnn_profile_handle)) {
                GGMLQNN_LOG_WARN("unable to create profile handle in the backend\n");
                return 6;
            } else {
                GGMLQNN_LOG_DEBUG("initialize qnn profile successfully\n");
            }
        }
    }

#if defined(__ANDROID__) || defined(__linux__)
    _rpc_lib_handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
#else
    _rpc_lib_handle = dlopen("libcdsprpc.dll", RTLD_NOW | RTLD_LOCAL);
#endif
    if (nullptr == _rpc_lib_handle) {
        GGMLQNN_LOG_WARN("failed to load qualcomm's rpc lib, error:%s\n", dlerror());
        return 7;
    } else {
        GGMLQNN_LOG_DEBUG("load rpcmem lib successfully\n");
        set_rpcmem_initialized(true);
    }
    _pfn_rpc_mem_init   = reinterpret_cast<pfn_rpc_mem_init>(dlsym(_rpc_lib_handle, "rpcmem_init"));
    _pfn_rpc_mem_deinit = reinterpret_cast<pfn_rpc_mem_deinit>(dlsym(_rpc_lib_handle, "rpcmem_deinit"));
    _pfn_rpc_mem_alloc  = reinterpret_cast<pfn_rpc_mem_alloc>(dlsym(_rpc_lib_handle,"rpcmem_alloc"));
    _pfn_rpc_mem_free   = reinterpret_cast<pfn_rpc_mem_free>(dlsym(_rpc_lib_handle, "rpcmem_free"));
    _pfn_rpc_mem_to_fd  = reinterpret_cast<pfn_rpc_mem_to_fd>(dlsym(_rpc_lib_handle,"rpcmem_to_fd"));
    if (nullptr == _pfn_rpc_mem_alloc || nullptr == _pfn_rpc_mem_free
        || nullptr == _pfn_rpc_mem_to_fd) {
        GGMLQNN_LOG_WARN("unable to access symbols in QNN RPC lib. dlerror(): %s", dlerror());
        dlclose(_rpc_lib_handle);
        return 8;
    }

    if (nullptr != _pfn_rpc_mem_init) // make Qualcomm's SoC based low-end phone happy
        _pfn_rpc_mem_init();

    std::vector<const QnnContext_Config_t *> temp_context_config;
    _qnn_interface.qnn_context_create(_qnn_backend_handle, _qnn_device_handle,
                               temp_context_config.empty() ? nullptr : temp_context_config.data(),
                               &_qnn_context_handle);
    if (nullptr == _qnn_context_handle) {
        GGMLQNN_LOG_WARN("why failed to initialize qnn context, error:%s\n", strerror(errno));
        return 9;
    } else {
        GGMLQNN_LOG_DEBUG("initialize qnn context successfully\n");
    }

    if (_backend_name.find("Htp") != std::string::npos) {
        htp_print_info();

        htp_probe_device_meminfo();

        if (0 != init_htp_perfinfra()) {
            GGMLQNN_LOG_WARN("initialize HTP performance failure");
        }
#if 0
        if (0 != set_rpc_polling()) {
            GGMLQNN_LOG_WARN("set RPC polling failure");
        }
        if (0 != set_high_performance_mode()) {
            GGMLQNN_LOG_WARN("set HTP high performance mode failure");
        }
#else
        htp_set_memory_grow_size();
        htp_enter_performance_mode();
#endif
        if (enable_qnn_rpc()) {
            GGMLQNN_LOG_INFO("NPU RPC feature enabled");
        } else {
            GGMLQNN_LOG_INFO("NPU RPC feature disabled");
        }
    }

    print_backend_info();

    GGMLQNN_LOG_DEBUG("leave qni_init\n");

    return 0;
}

int qnn_instance::qnn_finalize() {
    int ret_status = 0;
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    GGMLQNN_LOG_DEBUG("enter %s\n", __func__);
    ggmqnn_reset_tensoridx();

    free_rpcmem();
    unregister_rpcmem();

    if (nullptr != _pfn_rpc_mem_deinit)
        _pfn_rpc_mem_deinit();

    if (dlclose(_rpc_lib_handle) != 0) {
        GGMLQNN_LOG_WARN("failed to unload qualcomm's rpc lib, error:%s\n", dlerror());
    } else {
        GGMLQNN_LOG_DEBUG("succeed to close rpcmem lib\n");
    }

    if (nullptr != _qnn_context_handle) {
        error = _qnn_interface.qnn_context_free(_qnn_context_handle, _qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN context_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_context_handle = nullptr;
    }

    if (nullptr != _qnn_profile_handle) {
        error = _qnn_interface.qnn_profile_free(_qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN profile_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_profile_handle = nullptr;
    }

    if (nullptr != _qnn_device_handle) {
        error = _qnn_interface.qnn_device_free(_qnn_device_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN device_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_device_handle = nullptr;
    }

    if (nullptr != _qnn_backend_handle) {
        error = _qnn_interface.qnn_backend_free(_qnn_backend_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN backend_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_backend_handle = nullptr;

    }

    if (nullptr != _qnn_log_handle) {
        error = _qnn_interface.qnn_log_free(_qnn_log_handle);
        if (error != QNN_SUCCESS) {
            GGMLQNN_LOG_WARN("failed to free QNN log_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_log_handle = nullptr;
    }

    unload_backend();

    unload_system();
    GGMLQNN_LOG_DEBUG("leave %s\n", __func__);

    return ret_status;
}

int qnn_instance::init_qnn_graph(const std::string & graph_name, QNNBackend device, size_t vtcm_size_in_mb, size_t hvx_threads) {
    _graph_name = graph_name;
    _device_id = device;

    GGMLQNN_LOG_DEBUG("[%s][%s]created", ggml_backend_qnn_get_devname(device), graph_name.c_str());

    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    Qnn_GraphHandle_t graph_handle = nullptr;
    if (device == QNN_BACKEND_NPU) {
        QnnHtpGraph_CustomConfig_t hvx_config;
        hvx_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
        hvx_config.numHvxThreads = hvx_threads;
        QnnGraph_Config_t graph_hvx_config;
        graph_hvx_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_hvx_config.customConfig = &hvx_config;

        QnnHtpGraph_CustomConfig_t dlbc_config;
        dlbc_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        dlbc_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
        dlbc_config.optimizationOption.floatValue = 1.0; // set to 0.0 to turn off DLBC
        QnnGraph_Config_t graph_dlbc_config;
        graph_dlbc_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_dlbc_config.customConfig = &dlbc_config;

        QnnHtpGraph_CustomConfig_t opt_config;
        opt_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        opt_config.optimizationOption.floatValue = 1; // 1 / 3
        QnnGraph_Config_t graph_opt_config;
        graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_opt_config.customConfig = &opt_config;

        QnnHtpGraph_CustomConfig_t vtcm_config;
        vtcm_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
        vtcm_config.vtcmSizeInMB = vtcm_size_in_mb;
        QnnGraph_Config_t graph_vtcm_config;
        graph_vtcm_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_vtcm_config.customConfig = &vtcm_config;

        QnnHtpGraph_CustomConfig_t fp16_config;
        fp16_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        fp16_config.precision = QNN_PRECISION_FLOAT16;
        QnnGraph_Config_t graph_fp16_config;
        graph_fp16_config.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_fp16_config.customConfig = &fp16_config;

        const QnnGraph_Config_t * graph_configs[] = {&graph_hvx_config, &graph_dlbc_config, &graph_vtcm_config,
                                                    &graph_opt_config, &graph_fp16_config, nullptr};
        error = _qnn_interface.qnn_graph_create(_qnn_context_handle, graph_name.c_str(), graph_configs, &graph_handle);
    } else {
        error = _qnn_interface.qnn_graph_create(_qnn_context_handle, graph_name.c_str(), nullptr, &graph_handle);
    }

    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_ERROR("[%s][%s]failed to create qnn graph, error: %s",
                      ggml_backend_qnn_get_devname(device), graph_name.c_str(),
                      ggmlqnn_get_qnnerror_string(error));
        return error;
    }

    GGMLQNN_LOG_DEBUG("[%s]create graph %s succeed", ggml_backend_qnn_get_devname(device), graph_name.c_str());
    _qnn_graph_handle = graph_handle;
    if (device == QNN_BACKEND_NPU) {
        htp_set_n_hvx_threads(hvx_threads);
    }
    return QNN_SUCCESS;
}

int qnn_instance::init_qnn_graph(const char * graph_name, bool debug, uint8_t do_node_validation,
                                 const QnnGraph_Config_t ** graph_configs) {
    int result = 0;

    if (nullptr == graph_name) {
        GGMLQNN_LOG_WARN("graph name is null\n");
        return 1;
    }

    if (!_graph_name.empty()) {
        GGMLQNN_LOG_WARN("qnn model for graph %s already initialized\n", graph_name);
        return 2;
    }

    if (!do_node_validation) {
        GGMLQNN_LOG_WARN("node validation disabled, backend will not perform op validation prior to adding node\n");
    }

    _graph_name = graph_name;
    _debug_tensor = debug;
    _do_node_validations = do_node_validation;

    result = _qnn_raw_interface.graphCreate(_qnn_context_handle,
                                            graph_name,
                                            graph_configs,
                                            &_qnn_graph_handle);
    if (result != QNN_GRAPH_NO_ERROR || nullptr == _qnn_graph_handle) {
        GGMLQNN_LOG_WARN("failed to create graph in qnn context\n");
        return 3;
    } else {
        GGMLQNN_LOG_INFO("succeed to create graph %s, %p\n", graph_name, _qnn_graph_handle);
    }

    return 0;
}

int qnn_instance::finalize_qnn_graph() {
    if (nullptr != _qnn_graph_handle) {
        if (_qnn_raw_interface.graphFinalize(_qnn_graph_handle,
                                             _qnn_profile_handle, nullptr)
                                             != QNN_GRAPH_NO_ERROR) {
            GGMLQNN_LOG_WARN("finalizing graph failure\n");
            return 1;
        }
    } else {
        GGMLQNN_LOG_DEBUG("qnn graph handle is null\n");
    }

    return 0;
}

int qnn_instance::init_htp_perfinfra() {
    QnnDevice_Infrastructure_t device_infra = nullptr;
    int error = _qnn_raw_interface.deviceGetInfrastructure(&device_infra);
    if (error != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to get qnn device infra\n");
        return 1;
    }

    QnnHtpDevice_Infrastructure_t * htp_infra = static_cast<QnnHtpDevice_Infrastructure_t *>(device_infra);
    QnnHtpDevice_PerfInfrastructure_t * htp_perfinfra = &htp_infra->perfInfra;
    uint32_t power_configid = 1;
    uint32_t device_id      = 0;
    uint32_t core_id        = 0;
    htp_perfinfra->createPowerConfigId(device_id, core_id, &power_configid);
    _qnn_htp_perfinfra      = htp_perfinfra;
    _qnn_htp_powerconfig_id = power_configid;
    //FIXME:hardcode to 0 and 0 although it's correct
    _qnn_htp_device_id      = device_id;
    _qnn_htp_core_id        = core_id;

    return 0;
}

int qnn_instance::set_rpc_polling() {
    if (_qnn_rpc_pollingtime > 0) {
        QnnHtpPerfInfrastructure_PowerConfig_t rpc_pollingtime;
        memset(&rpc_pollingtime, 0, sizeof(rpc_pollingtime));
        rpc_pollingtime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        rpc_pollingtime.rpcPollingTimeConfig = _qnn_rpc_pollingtime;
        const QnnHtpPerfInfrastructure_PowerConfig_t * power_configs[] = {&rpc_pollingtime, nullptr};
        if (_qnn_htp_perfinfra) {
            _qnn_htp_perfinfra->setPowerConfig(_qnn_htp_powerconfig_id, power_configs);
        }
    }
    return 0;
}

int qnn_instance::set_high_performance_mode() {
    if (nullptr == _qnn_htp_perfinfra) {
        GGMLQNN_LOG_DEBUG("perf intra is null\n");
        return 1;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t power_config;
    memset(&power_config, 0, sizeof(power_config));
    power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    power_config.dcvsV3Config.dcvsEnable = 0;
    power_config.dcvsV3Config.setDcvsEnable = 1;
    power_config.dcvsV3Config.contextId = _qnn_htp_powerconfig_id;
    power_config.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    power_config.dcvsV3Config.setSleepLatency = 1; // True to consider Latency parameter otherwise False
    power_config.dcvsV3Config.setBusParams = 1; // True to consider Bus parameter otherwise False
    power_config.dcvsV3Config.setCoreParams = 1; // True to consider Core parameter otherwise False
    power_config.dcvsV3Config.sleepDisable = 0; // True to consider sleep/LPM modes, False to enable
    power_config.dcvsV3Config.setSleepDisable = 0; // True to consider sleep disable/enable parameter otherwise False
    // set Sleep latency parameter
    uint32_t latencyValue = 40;
    power_config.dcvsV3Config.sleepLatency = latencyValue; // range 40-2000 micro sec
    // set Bus Clock Parameters (refer QnnHtpPerfInfrastructure_VoltageCorner_t enum)
    power_config.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    power_config.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    power_config.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    // set Core Clock Parameters (refer QnnHtpPerfInfrastructure_VoltageCorner_t enum)
    power_config.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    power_config.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    power_config.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    // set power config with different performance parameters
    const QnnHtpPerfInfrastructure_PowerConfig_t * power_configs[] = {&power_config, nullptr};

    _qnn_htp_perfinfra->setPowerConfig(_qnn_htp_powerconfig_id, power_configs);

    return 0;
}

void qnn_instance::htp_print_info() {
    const QnnDevice_PlatformInfo_t * p_info = nullptr;
    _qnn_raw_interface.deviceGetPlatformInfo(nullptr, &p_info);
    GGMLQNN_LOG_INFO("HTP device counts %d", p_info->v1.numHwDevices);
    QnnDevice_HardwareDeviceInfo_t * infos = p_info->v1.hwDevices;
    for (size_t i = 0; i < p_info->v1.numHwDevices; i++) {
        GGMLQNN_LOG_INFO("HTP deviceID:%d, deviceType:%d, numCores %d", infos[i].v1.deviceId,
                         infos[i].v1.deviceType, infos[i].v1.numCores);
        QnnDevice_DeviceInfoExtension_t devinfo = infos[i].v1.deviceInfoExtension;
        QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = devinfo->onChipDevice;
        QnnHtpDevice_Arch_t htp_arch = chipinfo.arch;
        GGMLQNN_LOG_INFO("HTP_TYPE:%d(%s)", devinfo->devType,
                         (devinfo->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) ? "QNN_HTP_DEVICE_TYPE_ON_CHIP" : "QNN_HTP_DEVICE_TYPE_UNKNOWN");
        GGMLQNN_LOG_INFO("qualcomm soc_model:%d(%s), htp_arch:%d(%s), vtcm_size:%d MB，" \
                             "dlbc_support:%d, signedpd_support:%d", \
                             chipinfo.socModel, ggmlqnn_get_socmodel_desc(chipinfo.socModel), \
                             htp_arch, ggmlqnn_get_htparch_desc(htp_arch), chipinfo.vtcmSize, \
                             chipinfo.dlbcSupport, chipinfo.signedPdSupport);
        struct qcom_socinfo * socinfo = ggmlqnn_get_socinfo_from_socmodel(chipinfo.socModel);
        g_qnn_mgr[QNN_BACKEND_NPU].socinfo = { chipinfo.socModel, htp_arch, chipinfo.vtcmSize, {}};
        if (nullptr != socinfo) {
            memcpy(g_qnn_mgr[QNN_BACKEND_NPU].socinfo.soc_desc, socinfo->soc_desc, sizeof(socinfo->soc_desc));
            GGMLQNN_LOG_INFO("soc info:%s", socinfo->soc_desc);
        } else {
            memcpy(g_qnn_mgr[QNN_BACKEND_NPU].socinfo.soc_desc, "unknown", 7);
            GGMLQNN_LOG_INFO("soc info:unknown");
        }
    }
    _qnn_raw_interface.deviceFreePlatformInfo(nullptr, p_info);
}

void qnn_instance::htp_probe_device_meminfo() {
    size_t candidate_size   = 0;
    uint8_t * rpc_buffer    = nullptr;
    const int SIZE_IN_MB    = (1 << 20);
    size_t probe_slots[]    = {1024, 1536, 2048 - 48, 2048};
    size_t probe_counts     = sizeof(probe_slots) / sizeof(size_t);
    for (size_t idx = 0; idx < probe_counts; idx++) {
        rpc_buffer = static_cast<uint8_t *>(alloc_rpcmem_internal(probe_slots[idx] * SIZE_IN_MB, 4));
        if (nullptr == rpc_buffer) {
            GGMLQNN_LOG_DEBUG("alloc rpcmem %d (MB) failure, %s\n", probe_slots[idx], strerror(errno));
            break;
        } else {
            candidate_size = probe_slots[idx];
            free_rpcmem(rpc_buffer);
            rpc_buffer = nullptr;
        }
    }
    if (candidate_size > _rpcmem_capacity)
        _rpcmem_capacity = candidate_size;

    free_rpcmem();
    _rpcmem_usage = 0;
    GGMLQNN_LOG_INFO("capacity of rpc ion memory %d MB\n", _rpcmem_capacity);
}

void qnn_instance::print_backend_info() {
    auto print_property = [&](const char * name, QnnProperty_Key_t property) {
        auto ret = _qnn_raw_interface.propertyHasCapability(property);

        const char * status = "Unknown";
        if (ret == QNN_PROPERTY_SUPPORTED) {
            status = "Yes";
        } else if (ret == QNN_PROPERTY_NOT_SUPPORTED) {
            status = "No";
        }

        GGMLQNN_LOG_INFO("%s: %s", name, status);
    };

    GGMLQNN_LOG_INFO("QNN backend properties:");
    print_property("Create context from binary list", QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_LIST_ASYNC);
    print_property("Dynamic batch", QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE);
    print_property("Early termination", QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION);
    print_property("Dynamic dimensions", QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS);
    print_property("Blockwise quantization", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCK);
    print_property("Blockwise quantization with expansion", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION);
    print_property("Vector quantization", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_VECTOR);
    print_property("Tensor sparsity", QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY);
    print_property("Updateable application tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_APP_TENSORS);
    print_property("Updateable native tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_NATIVE_TENSORS);
    print_property("Updateable static tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_STATIC_TENSORS);
    print_property("Qnn group device", QNN_PROPERTY_GROUP_DEVICE);
}

void qnn_instance::htp_set_memory_grow_size(size_t size) {
    QnnHtpPerfInfrastructure_MemoryConfig_t grow_size_config = {
            .option            = QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE,
            .memGrowSizeConfig = (uint32_t)size,
    };

    const QnnHtpPerfInfrastructure_MemoryConfig_t *memory_config[] = {
            &grow_size_config,
            nullptr,
    };
    Qnn_ErrorHandle_t ret = _qnn_htp_perfinfra->setMemoryConfig(_qnn_htp_device_id, _qnn_htp_core_id, memory_config);
    if (ret != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to set HTP memory config");
    } else {
        GGMLQNN_LOG_INFO("succeed to set HTP memory config");
    }
}

void qnn_instance::htp_enter_performance_mode() {
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs_v3_config = {
            .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
            .dcvsV3Config =
                    {
                            .contextId = _qnn_htp_powerconfig_id,

                            .setDcvsEnable = 1,
                            .dcvsEnable    = 0,

                            .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,

                            .setSleepLatency = 1,
                            .sleepLatency    = 40,

                            .setSleepDisable = 1,
                            .sleepDisable    = 1,

                            .setBusParams           = 1,
                            .busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                            .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                            .busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,

                            .setCoreParams           = 1,
                            .coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                            .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                            .coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                    },
    };

    QnnHtpPerfInfrastructure_PowerConfig_t hmx_config = {
            .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2,
            .hmxV2Config =
                    {
                            .hmxPickDefault         = 0,
                            .hmxVoltageCornerMin    = DCVS_EXP_VCORNER_MAX,
                            .hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX,
                            .hmxVoltageCornerMax    = DCVS_EXP_VCORNER_MAX,
                            .hmxPerfMode            = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH,
                    },
    };

    QnnHtpPerfInfrastructure_PowerConfig_t rpc_ctrl_config = {
            .option                  = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY,
            .rpcControlLatencyConfig = 100,
    };

    QnnHtpPerfInfrastructure_PowerConfig_t rpc_poll_config = {
            .option               = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME,
            .rpcPollingTimeConfig = 9999,
    };

    const QnnHtpPerfInfrastructure_PowerConfig_t *power_configs[] = {
            &dcvs_v3_config,
            &hmx_config,
            &rpc_ctrl_config,
            &rpc_poll_config,
            nullptr,
    };
    Qnn_ErrorHandle_t ret = _qnn_htp_perfinfra->setPowerConfig(_qnn_htp_powerconfig_id, power_configs);
    if (ret != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to set HTP power config");
    } else {
        GGMLQNN_LOG_INFO("succeed to set HTP power config");
    }
}

void qnn_instance::htp_set_n_hvx_threads(size_t n_threads) {
    QnnHtpGraph_CustomConfig_t htp_hvx_thread_config = {
            .option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS,
            .numHvxThreads = n_threads,
    };

    QnnGraph_Config_t hvx_thread_config = {
            .option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM,
            .customConfig = &htp_hvx_thread_config,
    };

    const QnnGraph_Config_t * graph_configs[] = {&hvx_thread_config, nullptr};
    Qnn_ErrorHandle_t ret     = _qnn_raw_interface.graphSetConfig(_qnn_graph_handle, graph_configs);
    if (ret != QNN_SUCCESS) {
        GGMLQNN_LOG_WARN("failed to set QNN graph config: set hvx threads %d", n_threads);
    } else {
        GGMLQNN_LOG_INFO("succeed to set QNN graph config: set hvx threads %d", n_threads);
    }
}

static uint8_t * ggmlqnn_create_rpc_buffer(qnn_instance * instance, const ggml_tensor * ggml_tensor, Qnn_Tensor_t * qnn_tensor, bool b_copydata) {
    if (nullptr == instance || nullptr == ggml_tensor || nullptr == qnn_tensor) {
        GGMLQNN_LOG_WARN("invalid params\n");
        return nullptr;
    }

    uint8_t * qnn_rpcbuffer = static_cast<uint8_t *>(instance->alloc_rpcmem(ggml_nbytes(ggml_tensor), 4));
    if (nullptr == qnn_rpcbuffer) {
        GGMLQNN_LOG_WARN("alloc rpcmem failure, %s\n", strerror(errno));
        return nullptr;
    } else {
        GGMLQNN_LOG_DEBUG("alloc rpcmem %p successfully\n", qnn_rpcbuffer);
    }
    if (b_copydata)
        memcpy(qnn_rpcbuffer, ggml_tensor->data, ggml_nbytes(ggml_tensor));
    instance->register_rpcmem(qnn_rpcbuffer, qnn_tensor);
    return qnn_rpcbuffer;
}

static void ggmlqnn_print_tensors_info(const char * func_name, ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    //skip sanity check of params because of performance concern
    if (nullptr != func_name && nullptr != ctx) {
        GGMLQNN_LOG_DEBUG("call %s in dev %s\n", func_name, ctx->name);
    }
    GGMLQNN_LOG_DEBUG("%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                      src0->name,
                      src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                      src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    GGMLQNN_LOG_DEBUG("%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                      src1->name,
                      src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                      src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    GGMLQNN_LOG_DEBUG("%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                      dst->name,
                      dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                      dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    GGMLQNN_LOG_DEBUG("\n");
}

static void ggmlqnn_dump_op_info(const struct ggml_tensor * tensor) {
    //skip sanity check of params because of performance concern
    const struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor       * src1 = tensor->src[1];
    struct ggml_tensor       * dst  = const_cast<ggml_tensor *>(tensor);
    GGMLQNN_LOG_DEBUG("op name:%s, tensor type:%s", ggml_op_name(tensor->op), ggml_type_name(tensor->type));
    ggmlqnn_print_tensors_info(nullptr, nullptr, src0, src1, dst);
}

static Qnn_Tensor_t * ggmlqnn_create_general_tensor(const ggml_tensor * tensor, const char * name,
                                                    Qnn_TensorType_t qnn_tensor_type,
                                                    Qnn_DataType_t qnn_data_type,
                                                    uint32_t rank, uint32_t * dims,
                                                    void * data, uint32_t data_size,
                                                    bool b_transpose) {
    Qnn_ErrorHandle_t error         = QNN_SUCCESS;
    char tensor_name[GGML_MAX_NAME] = {};

    //ensure the tensor name is unique
    if (nullptr == name) {
        snprintf(tensor_name, GGML_MAX_NAME, "tensor_%-8d", ggmqnn_get_tensoridx());
    } else {
        snprintf(tensor_name, GGML_MAX_NAME, "tensor_%s%-8d", name, ggmqnn_get_tensoridx());
    }
    GGMLQNN_LOG_DEBUG("init_tensor %d", ggmqnn_get_tensoridx());
    ggmqnn_inc_tensoridx();

    uint32_t reverse_dims[GGML_MAX_DIMS]    = {};
    uint32_t transpose_dims[GGML_MAX_DIMS]  = {};
    uint32_t * tensor_dims                  = nullptr;
    //case 1:use dims info from ggml tensor
    if (nullptr != tensor) {
        //there are different dimension order between ggml tensor and qnn tensor
        for (size_t idx = 0; idx < rank; idx++) {
            reverse_dims[idx] = (uint32_t)tensor->ne[rank - 1 - idx];
        }
        tensor_dims = reverse_dims;
    }
    //case 2: use user's specified tensor_dims
    if (nullptr != dims) {
        tensor_dims = dims;
    }
    //case 3: transpose for dst tensor
    if (b_transpose) {
        GGML_ASSERT(tensor != nullptr); //ensure ggml_tensor is not nullptr for this special case

        ggmlqnn_get_qnn_dimensions_from_ggml_dimensions(transpose_dims, reverse_dims, ggml_n_dims(tensor));
        tensor_dims = transpose_dims;
#if 0
        for (size_t idx = 0; idx < 4; idx++) {
            GGMLQNN_LOG_DEBUG("origin dim[%d]=%d\n", idx, reverse_dims[idx]);
        }
        for (size_t idx = 0; idx < 4; idx++) {
            GGMLQNN_LOG_DEBUG("trans  dim[%d]=%d\n", idx, transpose_dims[idx]);
        }
#endif
    }

    Qnn_Tensor_t qnn_tensor = {
            .version= QNN_TENSOR_VERSION_1,
            {.v1= {
                    .id = 0,
                    .name = tensor_name,
                    .type = qnn_tensor_type,
                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType = qnn_data_type,
                    .quantizeParams = {.encodingDefinition = QNN_DEFINITION_UNDEFINED,
                            .quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED,
                            {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                    .rank = rank,
                    .dimensions = tensor_dims,
                    .memType = QNN_TENSORMEMTYPE_RAW,
                    .clientBuf = {.data = nullptr, .dataSize = 0}
            }
            }
    };
    if (nullptr != name) {
        QNN_VER_PTR(qnn_tensor)->name = name;
    }
    Qnn_Tensor_t * p_qnn_tensor = (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    if (nullptr == p_qnn_tensor) {
        GGMLQNN_LOG_WARN("calloc failed");
        return nullptr;
    }
    error = deep_copy_qnn_tensors(qnn_tensor, * p_qnn_tensor);
    if (error != QNN_SUCCESS) {
        free(p_qnn_tensor);
        GGMLQNN_LOG_WARN("init tensor failed");
        return  nullptr;
    }
    QNN_VER_PTR(*p_qnn_tensor)->clientBuf = {data, data_size};

    return p_qnn_tensor;
}

static Qnn_Tensor_t * ggmlqnn_create_compute_tensor(qnn_instance * instance, Qnn_GraphHandle_t graph_handle, const ggml_tensor * tensor, Qnn_TensorType_t tensor_type) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    uint32_t dimensions[]   = {(uint32_t) tensor->ne[0], (uint32_t) tensor->ne[1],
                               (uint32_t) tensor->ne[2], (uint32_t) tensor->ne[3]};
    Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
    Qnn_TensorType_t qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;

    if (0 == tensor->flags) {
        qnn_tensor_type = tensor_type;
    } else {
        if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
            qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
        } else if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
            qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
        }
    }

    qnn_data_type = ggmlqnn_datatype_from_ggml_datatype(tensor->type);
    Qnn_Tensor_t * p_qnn_tensor = ggmlqnn_create_general_tensor(tensor, nullptr,
                                                                qnn_tensor_type, qnn_data_type,
                                                                ggml_n_dims(tensor), dimensions,
                                                                nullptr, 0);

    bool enable_npu_rpc = (instance->enable_qnn_rpc() && instance->get_device_id() == QNN_BACKEND_NPU);
    if (enable_npu_rpc) {
        QNN_VER_PTR(*p_qnn_tensor)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        QNN_VER_PTR(*p_qnn_tensor)->clientBuf = {.data=nullptr, .dataSize=0};
    }
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = instance->get_qnn_raw_interface();
    CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_qnn_tensor));

    return p_qnn_tensor;
}

// =================================================================================================
//  section-8: implementation of ggml-qnn backend
// =================================================================================================
//TODO: refine this function as it is a performance hotspot/bottleneck function
static bool ggml_qnn_can_handle_op(const ggml_backend_qnn_context * ctx, const struct ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_NONE) {
        return true;
    }
    if (ggml_is_empty(tensor) || tensor->op == GGML_OP_RESHAPE
        || tensor->op == GGML_OP_TRANSPOSE
        || tensor->op == GGML_OP_VIEW
        || tensor->op == GGML_OP_PERMUTE
        ) {
        return false;
    }

    //TODO: add other op here
    bool supported_op = ((tensor->op == GGML_OP_ADD)
                         || (tensor->op == GGML_OP_MUL_MAT)
                         || (tensor->op == GGML_OP_MUL)
                        );
    if (!supported_op) {
        return false;
    }

    struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor * src1 = tensor->src[1];

    const int64_t ne00  = tensor->src[0]->ne[0];
    const int64_t ne01  = tensor->src[0]->ne[1];

    const int64_t ne10  = tensor->src[1]->ne[0];
    const int64_t ne11  = tensor->src[1]->ne[1];

    const int64_t ne0   = tensor->ne[0];
    const int64_t ne1   = tensor->ne[1];

    const uint32_t src0_rank = ggml_n_dims(src0);
    const uint32_t src1_rank = ggml_n_dims(src1);
    GGML_UNUSED(ne01);
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);

    if (tensor->op == GGML_OP_ADD) {
        //ggmlqnn_dump_op_info(tensor);
        if (!ggml_are_same_shape(src0, src1)) {
            return false;
        }
        if (ne00 < 32)
            return false;
        return (src0->type == GGML_TYPE_F32) && (src1->type == GGML_TYPE_F32);
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        //ggmlqnn_dump_op_info(tensor);
        if (src0_rank != src1_rank) // make QNN SDK happy
            return false;
        if (src0_rank < 2) // QNN's limitation, make QNN SDK happy
            return false;
        if (4 == src0_rank) //TODO: 4D matrix mulmat in CT
            return false;
        if ((src1->ne[2] != src0->ne[2]) || (src1->ne[3] != src0->ne[3])) // make QNN SDK happy
            return false;

        if (ctx->device == QNN_BACKEND_NPU)
            if (2 == src0_rank)
                return (src0->type == GGML_TYPE_F32
                    || src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0
                    || src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q8_K
                   ) && (src1->type == GGML_TYPE_F32) && (tensor->type == GGML_TYPE_F32);
           else
                return (src0->type == GGML_TYPE_F32) && (src1->type == GGML_TYPE_F32) && (tensor->type == GGML_TYPE_F32);
        else
            return (src0->type == GGML_TYPE_F32   || ggml_is_quantized(src0->type))
                    && (src1->type == GGML_TYPE_F32) && (tensor->type == GGML_TYPE_F32);
    }

    if (tensor->op == GGML_OP_MUL) {
        //ggmlqnn_dump_op_info(tensor);
        if (ctx->device == QNN_BACKEND_NPU)
            return false;
        if ((src0_rank != 2) || (src1_rank != 2)) //TODO: 3D and 4D matrix
            return false;
        return  (src0->type == GGML_TYPE_F32)
                && (src1->type == GGML_TYPE_F32)
                && (tensor->type == src1->type);
    }

    return false;
}

static bool ggml_qnn_compute_forward(ggml_backend_t backend, struct ggml_tensor * dst) {
    ggmlqnn_op_func_t func                = nullptr;
    ggml_backend_qnn_context * ctx        = (ggml_backend_qnn_context *)backend->context;

    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_qnn_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_qnn_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_qnn_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            func = ggml_qnn_general_node;
            break;
        case GGML_OP_ACC:
            ggml_qnn_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            func = ggml_qnn_general_node;
            break;
        case GGML_OP_DIV:
            ggml_qnn_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    break;
                case GGML_UNARY_OP_SILU:
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    break;
                case GGML_UNARY_OP_TANH:
                    break;
                case GGML_UNARY_OP_RELU:
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_qnn_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_qnn_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_qnn_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_qnn_upsample_nearest2d(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_qnn_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_qnn_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_qnn_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_qnn_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_qnn_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_qnn_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            return false;
        case GGML_OP_SCALE:
            ggml_qnn_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_qnn_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_qnn_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_qnn_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_qnn_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_qnn_diag_mask(ctx, dst, -INFINITY);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_qnn_softmax(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_qnn_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_qnn_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_qnn_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_qnn_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_qnn_argsort(ctx, dst);
            break;
        default:
            return false;
    }

    if (nullptr != func)
        func(ctx, dst);

    return true;
}

struct ggml_backend_qnn_buffer_context {
    ~ggml_backend_qnn_buffer_context() {
        if (buffer) {
            free(buffer);
        }

        for (auto * sub_buffer : sub_buffers) {
            free(sub_buffer);
        }

        for (auto * qnn_tensor : qnn_tensors) {
            free_qnn_tensor(qnn_tensor);
        }

        sub_buffers.clear();
        qnn_tensors.clear();
    }
    void * buffer       = nullptr;

    struct ggml_backend_qnn_context * backend_ctx = nullptr;

    size_t buffer_size  = 0;
    std::vector<void *> sub_buffers;
    std::vector<Qnn_Tensor_t *> qnn_tensors;
};

static void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_qnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;
    return ctx->buffer;
}

static enum ggml_status ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;
    GGML_UNUSED(tensor);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_qnn_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                  struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset((char *)tensor->data + offset, value, size);
}

static void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *)tensor->data + offset, size);
}

static bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

static void ggml_backend_qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;
    memset(ctx->buffer, value, ctx->buffer_size);
}

static ggml_backend_buffer_i ggml_backend_qnn_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_qnn_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_qnn_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_qnn_buffer_init_tensor,
        /* .memset_tensor   = */ ggml_backend_qnn_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_qnn_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_qnn_buffer_get_tensor,
        /* .cpy_tensor      = */ ggml_backend_qnn_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_qnn_buffer_clear,
        /* .reset           = */ nullptr,
};

static const char * ggml_backend_qnn_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "qnn-buffer";
}

static ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(
                                  ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_qnn_buffer_context * ctx = new ggml_backend_qnn_buffer_context;

    size_t size_page = 0;
#if defined(__ANDROID__) || defined(__linux__)
    size_page = sysconf(_SC_PAGESIZE);
#else
    SYSTEM_INFO systeminfo;
    GetSystemInfo(&systeminfo);
    size_page = systeminfo.dwPageSize;
#endif
    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }
    ctx->buffer         = ggmlqnn_host_malloc(size_aligned, size_page);
    ctx->buffer_size    = size_aligned;
    if (nullptr == ctx->buffer) {
        GGMLQNN_LOG_WARN("%s: failed to allocate %d MiB\n", __func__, size / (1 << 20));
        return nullptr;
    } else {
        GGMLQNN_LOG_DEBUG("%s: allocate %d MiB\n", __func__, size_aligned / (1 << 20));
    }

    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface, ctx, size);
}

static size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

static size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);

    return (2 * (1 << 29));
}

static bool ggml_backend_qnn_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const char * ggml_backend_qnn_name(ggml_backend_t backend) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;
    return g_qnn_mgr[ctx->device].name;
}

static void ggml_backend_qnn_free(ggml_backend_t backend) {
    GGMLQNN_LOG_DEBUG("enter %s", __func__ );
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;
    GGMLQNN_LOG_DEBUG("idx %d, name:%s", ctx->device, g_qnn_mgr[ctx->device].name);

    qnn_instance * instance = (qnn_instance*)g_qnn_mgr[ctx->device].instance;
    if (instance != nullptr) {
        std::map<std::string, std::tuple<Qnn_GraphHandle_t, std::vector<Qnn_Tensor_t*>>>::iterator graph_it;

        for (graph_it = instance->_qnn_graph_map.begin();
             graph_it != instance->_qnn_graph_map.end(); graph_it++) {
            auto & graph_item = graph_it->second;
            Qnn_GraphHandle_t & graph_handle = std::get<0>(graph_item);
            qnn_tensors_t &  tensors = std::get<1>(graph_item);
            for (auto tensor_it = tensors.begin(); tensor_it != tensors.end(); ++tensor_it) {
                free_qnn_tensor(*tensor_it);
            }
            GGML_UNUSED(graph_handle);
            GGMLQNN_LOG_DEBUG("graph type:%s", graph_it->first.c_str());
        }
        instance->_qnn_graph_map.clear();

        instance->qnn_finalize();
        delete instance;
        g_qnn_mgr[ctx->device].instance = nullptr;
    }

    if (g_qnn_mgr[ctx->device].backend != nullptr) {
        delete backend;
        g_qnn_mgr[ctx->device].backend = nullptr;
    }
    GGMLQNN_LOG_DEBUG("leave %s", __func__ );
}

static enum ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status result         = GGML_STATUS_SUCCESS;
    ggml_backend_qnn_context * ctx  = (ggml_backend_qnn_context *) backend->context;
    GGML_UNUSED(ctx);
    //GGMLQNN_LOG_DEBUG("device %d", ctx->device);
    //GGMLQNN_LOG_DEBUG("cgraph->n_nodes %d", cgraph->n_nodes);

    if (0 == g_inference_approach) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE
                || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW
                || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                continue;
            }
            bool ok = ggml_qnn_compute_forward(backend, node);
            if (!ok) {
                GGMLQNN_LOG_DEBUG("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
            }
        }
    } else {
        //offload entire cgraph to QNN CPU & GPU & NPU
        return ggmlqnn_graph_compute(backend, cgraph);
    }

    return result;
}

static const char * ggml_backend_qnn_device_get_name(ggml_backend_dev_t dev) {
    struct ggml_backend_qnn_context *ctx = static_cast<ggml_backend_qnn_context *>(dev->context);
    if (nullptr == ctx) {
        GGMLQNN_LOG_ERROR("pls check why ctx is null");
        return "unknown";
    }
    return ctx->name;
}

static const char * ggml_backend_qnn_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_qnn_context * ctx = static_cast<ggml_backend_qnn_context *>(dev->context);
    static char qnn_device_desc[256];
    if (nullptr == ctx) {
        GGMLQNN_LOG_ERROR("pls check why ctx is null");
        return "unknown";
    }
    if (0 == strncmp(ctx->name, "qnn-npu", 7)) {
        const char * soc_info = ggmlqnn_get_socmodel_desc(ctx->socinfo.soc_model);
        const char * htp_arch = ggmlqnn_get_htparch_desc(ctx->socinfo.htp_arch);
        std::string dev_desc = std::string(ctx->desc)
                + std::string(soc_info) + "_" + std::string(htp_arch)
                + "," + std::string(ctx->socinfo.soc_desc);
        memset(qnn_device_desc, 0, 256);
        memcpy(qnn_device_desc, dev_desc.c_str(), strlen(dev_desc.c_str()));
        return qnn_device_desc;
    } else {
        return ctx->desc;
    }
}

static void ggml_backend_qnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    struct ggml_backend_qnn_context * ctx = static_cast<ggml_backend_qnn_context *>(dev->context);
    if ((nullptr == ctx) || (ctx->device > QNN_BACKEND_GGML)) {
        GGMLQNN_LOG_ERROR("pls check params");
        *free = 0;
        *total = 0;
    }

    if (QNN_BACKEND_CPU == ctx->device || QNN_BACKEND_GGML == ctx->device) {
        *total = ggmlqnn_get_system_total_memory_in_bytes();
        *free = ggmlqnn_get_system_free_memory_in_bytes();
    } else if (QNN_BACKEND_GPU == ctx->device) {
        //TODO: probe GPU info in Qualcomm Adreno GPU
        *total = ggmlqnn_get_system_total_memory_in_bytes();
        *free = ggmlqnn_get_system_free_memory_in_bytes();
    } else if (QNN_BACKEND_NPU == ctx->device) {
        size_t rpc_ion_memsize = ctx->instance->get_rpcmem_capacity();
        size_t rpc_ion_usage = ctx->instance->get_rpcmem_usage();
        GGMLQNN_LOG_DEBUG("rpc memsize %d", rpc_ion_memsize);
        GGMLQNN_LOG_DEBUG("rpc usage %d", rpc_ion_usage);
        *total = rpc_ion_memsize * (1 << 20);
        *free = (rpc_ion_memsize - rpc_ion_usage) * (1 << 20);
    }
}

static enum ggml_backend_dev_type ggml_backend_qnn_device_get_type(ggml_backend_dev_t dev) {
    struct ggml_backend_qnn_context * ctx = static_cast<ggml_backend_qnn_context *>(dev->context);
    if (QNN_BACKEND_CPU == ctx->device)
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    else if (QNN_BACKEND_GPU == ctx->device)
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    else if (QNN_BACKEND_NPU == ctx->device)
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    else
        return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static void ggml_backend_qnn_device_get_props(ggml_backend_dev_t dev,
                                              struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_qnn_device_get_name(dev);
    props->description = ggml_backend_qnn_device_get_description(dev);
    props->type        = ggml_backend_qnn_device_get_type(dev);
    ggml_backend_qnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
            /* .async                 = */ false,
            /* .host_buffer           = */ false,
            /* .buffer_from_host_ptr  = */ true,
            /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_qnn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    if (nullptr == params) {
        params = 0;
    }
    ggml_backend_t qnn_backend = ggml_backend_qnn_init((int)(intptr_t)params, g_qnn_runtimelib_path);

    return qnn_backend;

}

static ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t device_index) {
    if (device_index >= GGML_QNN_MAX_DEVICES) {
        GGMLQNN_LOG_DEBUG("ggml_backend_qnn_buffer_type error: device_index:%d is out of range [0, %d]\n",
                      device_index, GGML_QNN_MAX_DEVICES - 1);
        return nullptr;
    }

    static struct ggml_backend_buffer_type ggml_backend_buffer_type_qnn = {
            /* .iface   = */ {
                                     /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                                     /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
                                     /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
                                     /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
                                     /* .get_alloc_size   = */ nullptr,// defaults to ggml_nbytes
                                     /* .is_host          = */ ggml_backend_qnn_buffer_is_host
                             },
            /* .device  = */ nullptr,
            /* .context = */ nullptr,
    };

    return &ggml_backend_buffer_type_qnn;
}

static ggml_backend_buffer_type_t ggml_backend_qnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) dev->context;
    return ggml_backend_qnn_buffer_type(ctx->device);
}

static ggml_backend_buffer_t ggml_backend_qnn_device_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_qnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) dev->context;
    return (ggml_qnn_can_handle_op(ctx,op));
}

static bool ggml_backend_qnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static struct ggml_backend_device_i ggml_backend_qnn_device_interface = {
        /* .get_name             = */ ggml_backend_qnn_device_get_name,
        /* .get_description      = */ ggml_backend_qnn_device_get_description,
        /* .get_memory           = */ ggml_backend_qnn_device_get_memory,
        /* .get_type             = */ ggml_backend_qnn_device_get_type,
        /* .get_props            = */ ggml_backend_qnn_device_get_props,
        /* .init_backend         = */ ggml_backend_qnn_device_init_backend,
        /* .get_buffer_type      = */ ggml_backend_qnn_device_get_buffer_type,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ ggml_backend_qnn_device_buffer_from_host_ptr,
        /* .supports_op          = */ ggml_backend_qnn_device_supports_op,
        /* .supports_buft        = */ ggml_backend_qnn_device_supports_buft,
        /* .offload_op           = */ nullptr,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
};

static ggml_backend_i ggml_backend_qnn_interface = {
        /* .get_name                = */ ggml_backend_qnn_name,
        /* .free                    = */ ggml_backend_qnn_free,
        /* .set_tensor_async        = */ nullptr,
        /* .get_tensor_async        = */ nullptr,
        /* .cpy_tensor_async        = */ nullptr,
        /* .synchronize             = */ nullptr,
        /* .graph_plan_create       = */ nullptr,
        /* .graph_plan_free         = */ nullptr,
        /* .graph_plan_update       = */ nullptr,
        /* .graph_plan_compute      = */ nullptr,
        /* .graph_compute           = */ ggml_backend_qnn_graph_compute,
        /* .event_record            = */ nullptr,
        /* .event_wait              = */ nullptr,
};

//FIXME: this guid is not make sense
static ggml_guid_t ggml_backend_qnn_guid() {
    static ggml_guid guid = {
            0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
            0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09
    };
    return &guid;
}

bool ggml_backend_is_qnn(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_qnn_guid());
}

void ggml_backend_qnn_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_qnn(backend));

    struct ggml_backend_qnn_context * ctx = (struct ggml_backend_qnn_context *)backend->context;
    ctx->threads = n_threads;
}

int ggml_backend_qnn_get_device_count() {
    return GGML_QNN_MAX_DEVICES;
}

struct ggml_backend_qnn_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_qnn_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "ggml-qnn";
}

static size_t ggml_backend_qnn_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_QNN_MAX_DEVICES;
}

static ggml_backend_dev_t ggml_backend_qnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_UNUSED(index);

    GGMLQNN_LOG_DEBUG("index %d", index);
    ggml_backend_qnn_reg_context * ctx = (ggml_backend_qnn_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_qnn_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (nullptr == name)
        return nullptr;

    const char * slot_name =  "ggml_backend_set_n_threads";
    //avoid buffer attack rather than strcmp
    if (0 == memcmp(name, slot_name, strlen(slot_name))) {
        return (void *)ggml_backend_qnn_set_n_threads;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_qnn_reg_interface = {
        /* .get_name          = */ ggml_backend_qnn_reg_get_name,
        /* .get_device_count  = */ ggml_backend_qnn_reg_get_device_count,
        /* .get_device        = */ ggml_backend_qnn_reg_get_device,
        /* .get_proc_address  = */ ggml_backend_qnn_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_qnn_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;
    GGMLQNN_LOG_DEBUG("enter ggml_backend_qnn_reg");
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_qnn_reg_context * ctx = new ggml_backend_qnn_reg_context;

            for (int i = 0; i < ggml_backend_qnn_get_device_count(); i++) {
                ggml_backend_dev_t dev = new ggml_backend_device {
                        /* .iface       = */ ggml_backend_qnn_device_interface,
                        /* .reg         = */ &reg,
                        /* .context     = */ &g_qnn_mgr[i]
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                    /* .api_version = */ GGML_BACKEND_API_VERSION,
                    /* .iface       = */ ggml_backend_qnn_reg_interface,
                    /* .context     = */ ctx
            };
        }

        initialized = true;
    }
    GGMLQNN_LOG_DEBUG("leave ggml_backend_qnn_reg");

    return &reg;
}

/**
 *
 * @param device            0: QNN_BACKEND_CPU 1: QNN_BACKEND_GPU 2: QNN_BACKEND_NPU
 * @param qnn_lib_path      QNN binrary runtime library path, such as "/data/local/tmp/" on Android or specified in JNI layer
 * @return
 */
ggml_backend_t ggml_backend_qnn_init(size_t device, const char * qnn_lib_path) {
    int result = 0;

    if (nullptr == qnn_lib_path)
        return nullptr;

    GGMLQNN_LOG_DEBUG("device %d", device);
    GGMLQNN_LOG_DEBUG("qnn_lib_path %s", qnn_lib_path);
    if (device >= GGML_QNN_MAX_DEVICES) {
        GGMLQNN_LOG_ERROR("invalid device %d", device);
        return nullptr;
    }

    if (nullptr != g_qnn_mgr[device].backend) {
        GGMLQNN_LOG_WARN("qnn backend %d(%s) already loaded", device, ggml_backend_qnn_get_devname(device));
        return g_qnn_mgr[device].backend;
    }

    ggmlqnn_load_cfg();

#if defined(__ANDROID__)
    std::string path = qnn_lib_path;
    if (QNN_BACKEND_NPU == device) {
        if (0 == setenv("LD_LIBRARY_PATH",
                        (path +
                         ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images").c_str(),
                        1)) {
            GGMLQNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            GGMLQNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
        if (0 == setenv("ADSP_LIBRARY_PATH",
                        (path +
                         ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp").c_str(),
                        1)) {
            GGMLQNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            GGMLQNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
    } else {
        if (0 == setenv("LD_LIBRARY_PATH",
                        (path +
                         ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images").c_str(),
                        1)) {
            GGMLQNN_LOG_INFO("%s backend setenv successfully\n", ggml_backend_qnn_get_devname(device));
        } else {
            GGMLQNN_LOG_ERROR("%s backend setenv failure\n", ggml_backend_qnn_get_devname(device));
        }
    }
#endif

    qnn_instance * instance = nullptr;
    instance = new qnn_instance(qnn_lib_path, g_qnn_mgr[device].lib, "");
    result = instance->qnn_init(nullptr);
    if (0 != result) {
        GGMLQNN_LOG_WARN("init qnn subsystem failed with qnn backend %s, pls check why\n", ggml_backend_qnn_get_devname(device));
        delete instance;
        return nullptr;
    }
    qnn_interface qnn_interface                             = instance->get_qnn_interface();
    if (!qnn_interface.is_loaded()) {
        GGMLQNN_LOG_WARN("qnn subsystem failure\n");
        delete instance;
        return nullptr;
    }

    std::string device_name = ggml_backend_qnn_get_devname(device);
    GGMLQNN_LOG_INFO("qnn device name %s", device_name.c_str());
    g_qnn_mgr[device].instance                  = instance;
    g_qnn_mgr[device].raw_interface             = instance->get_qnn_raw_interface();
    g_qnn_mgr[device].raw_system_interface      = instance->get_qnn_raw_system_interface();

    ggml_backend_t qnn_backend = new ggml_backend{
            /* .guid      = */ ggml_backend_qnn_guid(),
            /* .iface     = */ ggml_backend_qnn_interface,
            /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_qnn_reg(), device),
            /* .context   = */ &g_qnn_mgr[device]
    };
    g_qnn_mgr[device].backend   = qnn_backend;

    return qnn_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_qnn_reg)

// =================================================================================================
//  section-9: general approach: offload GGML op to QNN backend
// =================================================================================================
static inline uint32_t ggmlqnn_get_tensor_data_size(const ggml_tensor * tensor) {
    /*
    size_t data_size = ggml_row_size(tensor->type, tensor->ne[0]);
    size_t n_dims = ggml_get_tensor_rank(tensor);
    for (int i = 1; i < n_dims; i++) {
        data_size *= tensor->ne[i];
    }

    return data_size;
    */
    return ggml_nbytes(tensor);
}

static inline bool ggmlqnn_is_valid_params(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) {
    if ((nullptr == ctx) || (nullptr == src0) || (nullptr == src1) || (nullptr == dst)) {
        GGMLQNN_LOG_WARN("invalid params\n");
        return false;
    }

    qnn_instance * instance = ctx->instance;
    if (nullptr == instance) {
        GGMLQNN_LOG_WARN("invalid params\n");
        return false;
    }

    return true;
}

/*
 * provide a general skeleton to offload ggml op to QNN backend: a single node contains 2 input
 * tensor and 1 output tensor
*/
void ggml_qnn_general_node(ggml_backend_qnn_context * ctx, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    qnn_instance * instance                     = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    const ggml_tensor * src0                    = op->src[0];
    const ggml_tensor * src1                    = op->src[1];
    ggml_tensor * dst                           = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;
    size_t qnn_op_index                         = ggmlqnn_get_op_index(op);
    GGML_ASSERT(qnn_op_index < ggmlqnn_get_opcaps_size());
    const char * qnn_op_name                    = ggmlqnn_k_op_caps[qnn_op_index].qnn_op_name;
    std::string ggml_op_name_string             = std::string("ggml_") + ggml_op_name(op->op);
    const char * ggml_op_name                   = ggml_op_name_string.c_str();

    qnn_perf op_perf                            = qnn_perf(ggml_op_name);
    op_perf.start();

    //ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);
    bool enable_npu_rpc = instance->enable_qnn_rpc() && ctx->device == QNN_BACKEND_NPU;

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        //retrieve computational resource from cached QNN graph
        qnn_res_t & graph_item  = instance->_qnn_graph_map[graph_name];
        graph_handle            = std::get<0>(graph_item);
        qnn_tensors_t & tensor  = std::get<1>(graph_item);
        p_tensor0               = tensor[0];
        p_tensor1               = tensor[1];
        p_tensor2               = tensor[2];
    } else {
        GGMLQNN_LOG_INFO("graph name %s", graph_name.c_str());
        GGML_ASSERT(instance->get_device_id() == ctx->device);
        //create QNN graph
        error = instance->init_qnn_graph(graph_name, static_cast<QNNBackend>(ctx->device), 8, 4);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_WARN("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        graph_handle = instance->get_qnn_graph_handle();

        //create computational tensor
        p_tensor0 = ggmlqnn_create_compute_tensor(instance, graph_handle, src0, QNN_TENSOR_TYPE_APP_WRITE);
        p_tensor1 = ggmlqnn_create_compute_tensor(instance, graph_handle, src1, QNN_TENSOR_TYPE_APP_WRITE);
        p_tensor2 = ggmlqnn_create_compute_tensor(instance, graph_handle, dst,  QNN_TENSOR_TYPE_APP_READ);

        //compose QNN graph
        Qnn_Tensor_t tensor_inputs[] = {
                *p_tensor0,
                *p_tensor1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *p_tensor2
        };
        Qnn_OpConfig_t op_config = {
                QNN_OPCONFIG_VERSION_1, {
                        ggml_op_name,
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_op_name,
                        0,
                        nullptr,
                        2,
                        tensor_inputs,
                        1,
                        tensor_outputs
                }
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, op_config));
        //finalize QNN graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));

        //cache QNN graph
        qnn_tensors_t ggml_op_add_tensors;
        ggml_op_add_tensors.reserve(3);
        ggml_op_add_tensors.push_back(p_tensor0);
        ggml_op_add_tensors.push_back(p_tensor1);
        ggml_op_add_tensors.push_back(p_tensor2);
        auto  graph_item = std::make_tuple(graph_handle, ggml_op_add_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    }

    if (enable_npu_rpc) {
        uint8_t * qnn_buffer_0 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor0)->memHandle));
        GGMLQNN_LOG_INFO("qnn_rpcbuffer_0 = %p\n", qnn_buffer_0);
        if (nullptr != qnn_buffer_0) {
            memcpy(qnn_buffer_0, src0->data, ggml_nbytes(src0));
        }

        uint8_t * qnn_buffer_1 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor1)->memHandle));
        GGMLQNN_LOG_INFO("qnn_rpcbuffer_1 = %p\n", qnn_buffer_1);
        if (nullptr != qnn_buffer_1) {
            memcpy(qnn_buffer_1, src1->data, ggml_nbytes(src1));
        }
    } else {
        QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggmlqnn_get_tensor_data_size(src0)};
        QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggmlqnn_get_tensor_data_size(src1)};
        QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggmlqnn_get_tensor_data_size(dst)};
    }

    Qnn_Tensor_t tensor_inputs[] = {
            *p_tensor0,
            *p_tensor1
    };
    Qnn_Tensor_t tensor_outputs[] = {
            *p_tensor2
    };
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                        tensor_inputs, 2,
                                                        tensor_outputs, 1,
                                                        nullptr, nullptr));
    if (enable_npu_rpc) {
        //TODO:NPU RPC feature will failed with test-backend-ops
        uint8_t * qnn_buffer_2 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor2)->memHandle));
        if (nullptr != qnn_buffer_2) {
            memcpy(dst->data, qnn_buffer_2, ggml_nbytes(dst));
        }
    }

#if GGMLQNN_PRINT_OP_ADD_LOG
    op_perf.info();
#endif
}

/*
 * this function is AI-assisted code from Grok 3 for purpose of offload 4d matrix mulmat to QNN backend
 * various UT has verified and succeed but failed in CT of test-backend-ops
 *
 * the logic of ggml_qnn_mul_mat_4d is similar to ggml_qnn_mul_mat but much more complicated
 * than ggml_qnn_mul_mat, so it's a standalone function.
 * it will be combined with ggml_qnn_mul_mat in the future
 */
static void ggml_qnn_mul_mat_4d(ggml_backend_qnn_context *ctx, ggml_tensor *op) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    bool graph_initialized = false;
    qnn_perf op_perf = qnn_perf("ggml_qnn_mul_mat_4d");
    qnn_instance *instance = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface = ctx->raw_interface;

    const ggml_tensor *src0 = op->src[0];
    const ggml_tensor *src1 = op->src[1];
    ggml_tensor *dst = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    GGML_ASSERT(ggml_n_dims(src0) == 4 && ggml_n_dims(src1) == 4);
    op_perf.start();

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    GGMLQNN_LOG_DEBUG("graph name %s\n", graph_name.c_str());

    ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);

    Qnn_GraphHandle_t graph_handle = nullptr;
    Qnn_Tensor_t *p_tensor0 = nullptr;
    Qnn_Tensor_t *p_reshape0_out = nullptr;
    Qnn_Tensor_t *p_tile0_out = nullptr;
    Qnn_Tensor_t *p_tensor1 = nullptr;
    Qnn_Tensor_t *p_permute1_out = nullptr;
    Qnn_Tensor_t *p_reshape1_out = nullptr;
    Qnn_Tensor_t *p_matmul_out = nullptr;
    Qnn_Tensor_t *p_reshape2_out = nullptr;

    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        qnn_res_t &graph_item = instance->_qnn_graph_map[graph_name];
        graph_handle = std::get<0>(graph_item);
        qnn_tensors_t &tensors = std::get<1>(graph_item);
        p_tensor0 = tensors[0];
        p_reshape0_out = tensors[1];
        p_tile0_out = tensors[2];
        p_tensor1 = tensors[3];
        p_permute1_out = tensors[4];
        p_reshape1_out = tensors[5];
        p_matmul_out = tensors[6];
        p_reshape2_out = tensors[7];
    } else {
        CHECK_QNN_API(error, qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(),
                                                           graph_name.c_str(), NULL, &graph_handle));

        // Define dimensions
        uint32_t K = src0->ne[0];               // Inner dimension
        uint32_t M = src0->ne[1];               // Rows of src0
        uint32_t N = src1->ne[1];               // Columns of src1
        uint32_t B0 = src0->ne[2] * src0->ne[3]; // src0 batch
        uint32_t B1 = src1->ne[2] * src1->ne[3]; // src1 batch (drives output)

        // Validate K only
        GGML_ASSERT(src0->ne[0] == src1->ne[0]); // K must match

        // src0: [K, M, H0, B0] -> QNN: [B0, H0, M, K]
        uint32_t src0_dims[] = {static_cast<uint32_t>(src0->ne[3]), static_cast<uint32_t>(src0->ne[2]), static_cast<uint32_t>(src0->ne[1]), static_cast<uint32_t>(src0->ne[0])};
        p_tensor0 = GQCGT(src0, "input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, 4,
                          src0_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));

        // Reshape src0 to [B0, M, K]
        uint32_t reshape0_out_dims[] = {B0, M, K};
        p_reshape0_out = GQCGT(nullptr, "reshape0_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                               reshape0_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_reshape0_out));
        Qnn_Tensor_t reshape0_inputs[] = {*p_tensor0};
        Qnn_Tensor_t reshape0_outputs[] = {*p_reshape0_out};
        Qnn_OpConfig_t reshape0_op = ggmlqnn_create_op_config("reshape0", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                              QNN_OP_RESHAPE, nullptr, 0,
                                                              reshape0_inputs, 1, reshape0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape0_op));

        // Tile src0 to match B1: [B0, M, K] -> [B1, M, K]
        uint32_t tile0_out_dims[] = {B1, M, K};
        p_tile0_out = GQCGT(nullptr, "tile0_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                            tile0_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tile0_out));
        uint32_t tile_multiples[] = {B1 / B0, 1, 1};
        uint32_t tile_dims[] = {3};
        Qnn_Tensor_t *p_tile_multiples = GQCGT(nullptr, "tile_multiples", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                               tile_dims, tile_multiples, sizeof(tile_multiples));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tile_multiples));
        Qnn_Param_t tile_params[] = {{QNN_PARAMTYPE_TENSOR, "multiples", .tensorParam = *p_tile_multiples}};
        Qnn_Tensor_t tile0_inputs[] = {*p_reshape0_out};
        Qnn_Tensor_t tile0_outputs[] = {*p_tile0_out};
        Qnn_OpConfig_t tile0_op = ggmlqnn_create_op_config("tile0", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                           QNN_OP_TILE, tile_params, 1,
                                                           tile0_inputs, 1, tile0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, tile0_op));

        // src1: [N, K, H1, B1] -> QNN: [B1, H1, N, K]
        uint32_t src1_dims[] = {static_cast<uint32_t>(src1->ne[3]), static_cast<uint32_t>(src1->ne[2]), static_cast<uint32_t>(src1->ne[1]), static_cast<uint32_t>(src1->ne[0])};
        p_tensor1 = GQCGT(src1, "input1", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, 4,
                          src1_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));

        // Permute src1 to [B1, H1, K, N]
        uint32_t perm_data[] = {0, 1, 3, 2};
        uint32_t perm_dims[] = {4};
        Qnn_Tensor_t *p_perm = GQCGT(nullptr, "perm", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                     perm_dims, perm_data, sizeof(perm_data));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_perm));
        uint32_t permute1_out_dims[] = {static_cast<uint32_t>(src1->ne[3]), static_cast<uint32_t>(src1->ne[2]), static_cast<uint32_t>(src1->ne[0]), static_cast<uint32_t>(src1->ne[1])};
        p_permute1_out = GQCGT(nullptr, "permute1_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 4,
                               permute1_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_permute1_out));
        Qnn_Param_t permute1_params[] = {{QNN_PARAMTYPE_TENSOR, "perm", .tensorParam = *p_perm}};
        Qnn_Tensor_t permute1_inputs[] = {*p_tensor1};
        Qnn_Tensor_t permute1_outputs[] = {*p_permute1_out};
        Qnn_OpConfig_t permute1_op = ggmlqnn_create_op_config("permute1", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                              QNN_OP_TRANSPOSE, permute1_params, 1,
                                                              permute1_inputs, 1, permute1_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, permute1_op));

        // Reshape src1 to [B1, K, N]
        uint32_t reshape1_out_dims[] = {B1, K, N};
        p_reshape1_out = GQCGT(nullptr, "reshape1_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                               reshape1_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_reshape1_out));
        Qnn_Tensor_t reshape1_inputs[] = {*p_permute1_out};
        Qnn_Tensor_t reshape1_outputs[] = {*p_reshape1_out};
        Qnn_OpConfig_t reshape1_op = ggmlqnn_create_op_config("reshape1", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                              QNN_OP_RESHAPE, nullptr, 0,
                                                              reshape1_inputs, 1, reshape1_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape1_op));

        // MatMul: [B1, M, K] x [B1, K, N] -> [B1, M, N]
        uint32_t matmul_out_dims[] = {B1, M, N};
        p_matmul_out = GQCGT(nullptr, "matmul_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                             matmul_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_matmul_out));
        Qnn_Tensor_t matmul_inputs[] = {*p_tile0_out, *p_reshape1_out};
        Qnn_Tensor_t matmul_outputs[] = {*p_matmul_out};
        Qnn_OpConfig_t matmul_op = ggmlqnn_create_op_config("matmul", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                            QNN_OP_MAT_MUL, nullptr, 0,
                                                            matmul_inputs, 2, matmul_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, matmul_op));

        // Output: [N, M, H1, B1] -> QNN: [B1, H1, M, N]
        uint32_t reshape2_out_dims[] = {static_cast<uint32_t>(dst->ne[3]), static_cast<uint32_t>(dst->ne[2]), static_cast<uint32_t>(dst->ne[1]), static_cast<uint32_t>(dst->ne[0])};
        p_reshape2_out = GQCGT(dst, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, 4,
                               reshape2_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_reshape2_out));
        Qnn_Tensor_t reshape2_inputs[] = {*p_matmul_out};
        Qnn_Tensor_t reshape2_outputs[] = {*p_reshape2_out};
        Qnn_OpConfig_t reshape2_op = ggmlqnn_create_op_config("reshape2", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                              QNN_OP_RESHAPE, nullptr, 0,
                                                              reshape2_inputs, 1, reshape2_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape2_op));

        // Finalize
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, NULL, NULL));

        // Cache
        qnn_tensors_t ggml_op_mulmat_tensors = {p_tensor0, p_reshape0_out, p_tile0_out, p_tensor1, p_permute1_out, p_reshape1_out, p_matmul_out, p_reshape2_out};
        instance->_qnn_graph_map[graph_name] = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
    }

    // Execute
    QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, static_cast<uint32_t>(ggml_nbytes(src0))};
    QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, static_cast<uint32_t>(ggml_nbytes(src1))};
    QNN_VER_PTR(*p_reshape2_out)->clientBuf = {dst->data, static_cast<uint32_t>(ggml_nbytes(dst))};

    Qnn_Tensor_t input_tensors[] = {*p_tensor0, *p_tensor1};
    Qnn_Tensor_t output_tensors[] = {*p_reshape2_out};
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle, input_tensors, 2,
                                                        output_tensors, 1, NULL, NULL));

#if 0
    // Log dst for debugging
    float *dst_data = (float *)dst->data;
    GGMLQNN_LOG_DEBUG("dst shape: [%d, %d, %d, %d]\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    for (int i = 0; i < dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3]; i++) {
        GGMLQNN_LOG_DEBUG("dst[%d] = %f\n", i, dst_data[i]);
    }
#endif

    op_perf.info();
}

/*
 * @brief performs matrix multiplication with FP32 & quantized weights and floating-point inputs
 *        using the QNN backend. this function performs matrix multiplication of the input tensor
 *        `src1` and the weight tensor `src0`, handling transposing, and quantization as needed,
 *        and stores the result in the destination tensor `dst`.
 *
         there are two key-points in properly handling how to offload mulmat to the QNN backend in ggml-qnn
         1. transpose
            a 3x2 f32 matrix which means 3 rows and 2 columns. in ggml, it could be created from:
            struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
            which like this:
            +---+---+
            | 0 | 1 |
            +---+---+
            | 2 | 3 |
            +---+---+
            | 4 | 5 |
            +---+---+
            with
                ne[0] = 2
                ne[1] = 3
            there are different dimension order between ggml tensor and qnn tensor

          2. QNN's MatMul can only support input tensors with rank >= 2

             in the all, there is gap between ggml mulmat and QNN mulmat,we need to perform a transpose
             operation when offloading mulmat to QNN backend. this implementation will handle transpose
             in func ggml_qnn_create_general_tensor()
 *
 *        this function is a good example to illustrated the second technical approach "mapping the
 *        entire ggml computational graph to QNN graph" without complex C++ encapsulation. or another
 *        pipeline of "how to utilize the Hexagon NPU maximally through QNN SDK", details could be found at
 *        https://github.com/ggml-org/llama.cpp/pull/12049#issuecomment-2678308360
 *
 * @param ctx     the context of ggml-qnn backend
 * @param op      the destination tensor where the result of the matrix multiplication will be stored.
 *
 * @note the logic of ggml_qnn_mul_mat is similar to ggml_qnn_general_node but much more complicated
 *       than ggml_qnn_general_node. so it's a standalone function. accordingly, this is another
 *       typical skeleton for offload other ggml ops to QNN backend. MUL_MAT take most of the compute
 *       time (about 95%).so to speed up llama inference, should focus on this func. there are three kinds
 *       of MUL_MAT to compute:
 *       mul_mat_f32:     both src0 and src1 are F32, this will be naturally handled in QNN backend
 *       mul_mat_f16_f32: src0 is F16 and src1 is F32, f16 in src0 -> f32 in src0', then src0' * src1
 *       mul_mat_q_f32:   src0 is quantized (Q4_0, Q4_1, Q6_K...)
 *                        and src1 is F32, src0 -> f32 in src0', then src0' * src1
*/
void ggml_qnn_mul_mat(ggml_backend_qnn_context * ctx, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    qnn_perf op_perf                            = qnn_perf("ggml_qnn_mul_mat");
    qnn_instance * instance                     = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    Qnn_Tensor_t * p_param_tensor               = nullptr;
    Qnn_Tensor_t * p_tensor2_transpose          = nullptr;
    const ggml_tensor * src0                    = op->src[0];
    const ggml_tensor * src1                    = op->src[1];
    ggml_tensor       * dst                     = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;
    op_perf.start();

    const enum ggml_type src0_type              = src0->type;
    const uint32_t src0_rank                    = ggml_n_dims(src0);
    const uint32_t src1_rank                    = ggml_n_dims(src1);
    GGML_ASSERT(src0_rank == src1_rank);
    GGML_ASSERT(src0_rank >= 2); //QNN SDK's limitation, make QNN SDK happy
    if (4 == src0_rank) {
        return ggml_qnn_mul_mat_4d(ctx, op);
    }
    void * wdata                                = ggmlqnn_type_trait(ctx, op);
    const size_t desired_size                   = ctx->desired_size;

    ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        //retrieve computational resource from cached QNN graph
        qnn_res_t & graph_item  = instance->_qnn_graph_map[graph_name];
        graph_handle            = std::get<0>(graph_item);
        qnn_tensors_t & tensors = std::get<1>(graph_item);
        p_tensor0               = tensors[0];
        p_tensor1               = tensors[1];
        p_tensor2               = tensors[2];
        p_param_tensor          = tensors[3];
        p_tensor2_transpose     = tensors[4];
    } else {
        //create QNN graph
        GGMLQNN_LOG_INFO("graph name %s", graph_name.c_str());
        error = instance->init_qnn_graph(graph_name, static_cast<QNNBackend>(ctx->device), 8, 4);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_WARN("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        graph_handle = instance->get_qnn_graph_handle();

        //create computational tensor
        p_tensor0 = GQCGT(src0, nullptr, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor1 = GQCGT(src1, nullptr, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor2 = GQCGT(dst, nullptr, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));

        //create param tensor for offload 2d/3d/4d matrix multiplication
        const uint32_t param_tensor_data[GGML_MAX_DIMS][GGML_MAX_DIMS] = {
                {0},
                {1, 0},
                {0, 2, 1},
                {0, 1, 3, 2},
        };
        uint32_t param_tensor_dims[1] = {src0_rank};
        p_param_tensor = GQCGT(nullptr, "param", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1, param_tensor_dims, (void *)(param_tensor_data[src0_rank - 1]), src0_rank * sizeof(uint32_t));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_param_tensor));

        //create transpose tensor
        p_tensor2_transpose = GQCGT(dst, "transpose", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0, true);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2_transpose));

        //compose QNN graph: add mulmat node
        Qnn_Param_t out_0_params[]   = {{QNN_PARAMTYPE_SCALAR, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, .scalarParam = {QNN_DATATYPE_BOOL_8, .bool8Value = 1}}};
        Qnn_Tensor_t out_0_inputs[]  = {*p_tensor0, *p_tensor1};
        Qnn_Tensor_t out_0_outputs[] = {*p_tensor2_transpose};
        Qnn_OpConfig_t out_0         = ggmlqnn_create_op_config("mulmat_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, out_0_params, 1, out_0_inputs, 2, out_0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_0));

        //compose QNN graph: add transpose node
        Qnn_Param_t out_trans1_0_params[]   = { {QNN_PARAMTYPE_TENSOR, "perm", .tensorParam = *p_param_tensor}};
        Qnn_Tensor_t out_trans1_0_inputs[]  = {*p_tensor2_transpose};
        Qnn_Tensor_t out_trans1_0_outputs[] = {*p_tensor2};
        Qnn_OpConfig_t out_trans1_0         = ggmlqnn_create_op_config("mulmat_transpose_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE, out_trans1_0_params, 1, out_trans1_0_inputs, 1, out_trans1_0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_trans1_0));

        //finalize QNN graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));

        //cache QNN graph
        qnn_tensors_t ggml_op_mulmat_tensors;
        ggml_op_mulmat_tensors.reserve(5);
        ggml_op_mulmat_tensors.push_back(p_tensor0);
        ggml_op_mulmat_tensors.push_back(p_tensor1);
        ggml_op_mulmat_tensors.push_back(p_tensor2);
        ggml_op_mulmat_tensors.push_back(p_param_tensor);
        ggml_op_mulmat_tensors.push_back(p_tensor2_transpose);
        auto  graph_item = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    }

    if (src0_type != GGML_TYPE_F32) {
        QNN_VER_PTR(*p_tensor0)->clientBuf = {wdata, static_cast<uint32_t>(desired_size)};
    } else {
        QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggmlqnn_get_tensor_data_size(src0)};
    }
    QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggmlqnn_get_tensor_data_size(src1)};
    QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggmlqnn_get_tensor_data_size(dst)};

    Qnn_Tensor_t tensor_inputs[] = {
            *p_tensor0,
            *p_tensor1
    };
    Qnn_Tensor_t tensor_outputs[] = {
            *p_tensor2
    };
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                        tensor_inputs, 2,
                                                        tensor_outputs, 1,
                                                        nullptr, nullptr));
    op_perf.info();
}

void ggml_qnn_repeat(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_div(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_leaky_relu(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_concat(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_arange(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_sqr(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_clamp(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_scale(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_argsort(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_group_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_acc(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_sum_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_upsample_nearest2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_pad(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_pool2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_dup(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_rms_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_diag_mask(ggml_backend_qnn_context * ctx, ggml_tensor * dst, float value) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(value);
}

void ggml_qnn_im2col(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_timestep_embedding(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_cpy(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    ggml_qnn_dup(ctx, dst);
}

void ggml_qnn_softmax(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_get_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_rope(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

// =================================================================================================
//  section-10: second approach: mapping ggml computational cgraph to QNN graph
// =================================================================================================
// details: https://github.com/ggml-org/llama.cpp/pull/12326#issuecomment-2712838649
// ref:     https://github.com/kantv-ai/kantv/blob/kantv-poc-with-qnn/core/ggml/jni/Inception_v3.cpp#L20634
// TODO: mapping entire ggml cgraph to a single QNN graph
static enum ggml_status ggmlqnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status ggml_result                = GGML_STATUS_SUCCESS;
    Qnn_ErrorHandle_t qnn_error                 = QNN_SUCCESS;
    qnn_perf op_perf                            = qnn_perf("ggmlqnn_graph_compute");
    qnn_instance * instance                     = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    ggml_backend_qnn_context * ctx              = (ggml_backend_qnn_context *) backend->context;
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;
    op_perf.start();

    //now we got the entire ggml cgraph
    GGMLQNN_LOG_DEBUG("qnn device %d(%s)", ctx->device, ggml_backend_qnn_get_devname(ctx->device));
    GGMLQNN_LOG_DEBUG("cgraph->n_nodes %d", cgraph->n_nodes);
    int num_nodes = std::min(5, cgraph->n_nodes);
    //for (int i = 0; i < cgraph->n_nodes; i++) {
    for (int i = 0; i < num_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        GGMLQNN_LOG_DEBUG("%s: op %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
    }

    //now we'll offload the entire ggml cgraph to a single opcfg QNN graph
    std::string graph_name;
    ggmlqnn_get_graphkey_from_cgraph(cgraph, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        //retrieve computational resource from cached QNN graph
        qnn_res_t & graph_item  = instance->_qnn_graph_map[graph_name];
        graph_handle            = std::get<0>(graph_item);
    } else {
        //create QNN graph
        GGMLQNN_LOG_INFO("graph name %s", graph_name.c_str());
        qnn_error = instance->init_qnn_graph(graph_name, static_cast<QNNBackend>(ctx->device), 8, 4);
        if (QNN_SUCCESS != qnn_error) {
            GGMLQNN_LOG_WARN("can't create qnn graph handle with graph name %s, error = %d(%s)\n", graph_name.c_str(), qnn_error,
                             ggmlqnn_get_qnnerror_string(qnn_error));
            return ggml_result;
        }
        graph_handle = instance->get_qnn_graph_handle();
        //TODO: compose a single opcfg QNN graph

        //TODO: finalize QNN graph
        //CHECK_QNN_API(qnn_error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));

        //cache QNN graph
        qnn_tensors_t ggml_op_mulmat_tensors;
        ggml_op_mulmat_tensors.reserve(0);
        auto  graph_item = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    }
    //exec QNN graph

    GGMLQNN_LOG_DEBUG("the second inference approach \"mapping cgraph to QNN graph\" is actually not supported now");

    return ggml_result;
}

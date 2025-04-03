/*
 * Copyright (c) 2023-2025 The ggml authors
 *
 * Qualcomm QNN SDK and reference tech guides could be found at:
 * https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
 * Qualcomm Hexagon SDK and reference tech guides could be found at:
 * https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
 *
 * this single-source-file or self-contained implementation of ggml-hexagon backend has 8 sections:
 * section-1  forward/prototype declaration, global vars, macros, data structures
 * section-2  internal troubleshooting function/class
 * section-3  helper function for WoA(Windows on ARM)
 * section-4  general helper function
 * section-5  QNN helper function/class
 * section-6  implementation of hwaccel approach through QNN: offload ggmlop to QNN
 * section-7  cDSP helper function
 * section-8  implementation of ggml-hexagon backend according to specification in ggml backend subsystem
 *
 * currently provide following ggml op' implementation through QNN:
 * - GGML_OP_ADD/GGML_OP_SUB/GGML_OP_MUL/GGML_OP_DIV/GGML_OP_LOG/GGML_OP_SQRT:
 *   this is a simple hwaccel skeleton, can expand other ggml ops according to expertise
 * - GGML_OP_MUL_MAT:
 *   this is a complicated hwaccel skeleton, can expand other ggml ops accordingly
 *
 *  currently provide following ggml op' implementation through cDSP in hexagon-kernels:
 * - GGML_OP_ADD & GGML_OP_MUL_MAT:
 *   this is a hwaccel skeleton, can expand other ggml ops accordingly
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

#if defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdatomic.h>
#endif

#if !defined(__ANDROID__) && !defined(__linux__)
#include <wchar.h>
#include <malloc.h>
#include <Windows.h>
#endif

#if defined(__ANDROID__)
#include "android/log.h"

#include "rpcmem.h"
#include "remote.h"
#include "os_defines.h"
#include "domain.h"
#include "AEEStdErr.h"
#include "HAP_power.h"
#include "HAP_farf.h"
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

#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "kernels/ggmlop_ap_skel.h"

// =================================================================================================
//  section-1: forward/prototype declaration, global vars, macros, data structures
// =================================================================================================
class  qnn_instance;
struct ggml_backend_hexagon_context;

#if 0//def NDEBUG
#define GGMLHEXAGON_DEBUG                               0
#else
#define GGMLHEXAGON_DEBUG                               1
#endif

#define GGMLHEXAGON_LOGBUF_LEN                          4096
#define GGMLHEXAGON_TMPBUF_LEN                          256

#define GGMLHEXAGON_LOG_ERROR(...)                      ggmlhexagon_log_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_WARN(...)                       ggmlhexagon_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_INFO(...)                       ggmlhexagon_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGMLHEXAGON_DEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)                      ggmlhexagon_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#endif

#define QNN_VER_PTR(x)                                  (&((x).v1))
#define RPCMEM_DEFAULT_FLAGS                            1
#define RPCMEM_HEAP_ID_SYSTEM                           25
#define SIZE_IN_MB                                      (1 << 20)
#define STATUS_CONTEXT                                  0x12345678

#define CHECK_QNN_API(error, result)                                            \
    do {                                                                        \
        error = (result);                                                       \
        if (QNN_SUCCESS != error) {                                             \
            if (error == QNN_COMMON_ERROR_NOT_SUPPORTED) {                      \
                GGMLHEXAGON_LOG_WARN("WARNING: QNN feature/API not supported\n");   \
            } else {                                                            \
                GGMLHEXAGON_LOG_INFO("QNN API error = %d(%s)\n", error, ggmlqnn_get_qnnerror_string(error));  \
            }                                                                   \
        }                                                                       \
    } while (0)

#define GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst)                              \
    do {                                                                        \
        if (g_hexagon_appcfg.hwaccel_approach != HWACCEL_CDSP) {                    \
            if (!ggmlqnn_is_valid_params((ctx), (src0), (src1), (dst))) {       \
                return;                                                         \
            }                                                                   \
        }                                                                       \
    } while (0)                                                                 \

// =================================================================================================
//  section-1: data type, data structure, global vars
// =================================================================================================
using pfn_rpc_mem_init                          = void (*)(void);
using pfn_rpc_mem_deinit                        = void (*)(void);
using pfn_rpc_mem_alloc                         = void *(*)(int, uint32_t, int);
using pfn_rpc_mem_free                          = void (*)(void *);
using pfn_rpc_mem_to_fd                         = int (*)(void *);
using _pfn_QnnSaver_initialize                  = decltype(QnnSaver_initialize);
using _pfn_QnnInterface_getProviders            = decltype(QnnInterface_getProviders);
using _pfn_QnnSystemInterface_getProviders      = decltype(QnnSystemInterface_getProviders);

//QNN resource management for the general approach through QNN
using qnn_tensors_t                             = std::vector< Qnn_Tensor_t >;
using qnn_ptensors_t                            = std::vector< Qnn_Tensor_t *>;
using qnn_singlenode_res_t                      = std::tuple<Qnn_GraphHandle_t, qnn_ptensors_t>;

typedef void (* ggmlqnn_op_func_t)(ggml_backend_hexagon_context * ctx, ggml_tensor * op);
typedef int  (* notify_callback_fn)(void * context, int domain, int session, remote_rpc_status_flags_t status);
typedef int  (* ggmlhexagon_op_func_t)(remote_handle64 handle, const dsptensor * src0, const dsptensor * src1, dsptensor * dst);

enum qnn_index_type {
    QNN_TENSOR_INDEX = 0,
    QNN_OPCFG_INDEX  = 1,
};

enum qnn_profile_level {
    PROFILE_OFF     = 0,
    PROFILE_BASIC   = 1,
    PROFILE_DETAIL  = 2,
};

//0: general approach through QNN:offload ggmlop to QNN
//1: special approach through QNN-SINGLEGRAPH:mapping entire ggml cgraph to a single QNN graph
//2: general approach through Hexagon cDSP:offload ggmlop to Hexagon cDSP directly
enum hwaccel_approach_type {
    HWACCEL_QNN                     = 0,
    HWACCEL_QNN_SINGLEGRAPH         = 1,
    HWACCEL_CDSP                    = 2,
};

enum hexagon_dsp_type {
    HEXAGON_ADSP    = 0,
    HEXAGON_MDSP    = 1,
    HEXAGON_SDSP    = 2,
    HEXAGON_CDSP    = 3,
    HEXAGON_CDSP1   = 4,
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
    SM8750 = 69,  // v79, SD 8 Elite(aka 8 Gen 4)
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

struct ggml_backend_hexagon_context {
    int device;
    char name[GGML_MAX_NAME];
    char desc[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    qnn_instance * instance;
    struct ggml_backend * backend;
    QNN_INTERFACE_VER_TYPE raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE raw_system_interface;
    struct qcom_socinfo           socinfo;

    //QNN resource management for the general approach through QNN
    std::map<std::string, qnn_singlenode_res_t> qnn_singlenode_graph_map;

    //quantize data -> fp32
    std::unique_ptr<char[]> work_data;
    std::vector<std::future<void>> tasks;
    size_t work_size;
    size_t desired_size;
    int n_threads;

    //Hexagon resource management for the general approach through Hexagaon cDSP
    size_t rpc_mempool_capacity;
    size_t rpc_mempool_len;
    size_t rpc_mempool_usage;
    void * rpc_mempool;
    int rpc_mempool_handle;
    remote_handle64 ggmlop_handle;
    int domain_id;
};

struct qnn_op_caps {
    bool supported;
    ggml_op op;
    const size_t input_param_count;
    const char * qnn_op_name;
};

struct hexagon_op_caps {
    bool supported;
    ggml_op op;
    const size_t input_param_count;
    const char * hexagon_op_name;
    ggmlhexagon_op_func_t dsp_op_func;
};

struct hexagon_appcfg_t {
    int print_qnn_internal_log; // enable/disable QNN's internal log
    int enable_perf;            // enable/disable perf of op function
    int print_tensors_info;     // enable/disable print tensors info in op function
    int dump_op_info;           // enable/disable dump op info in handle_op
    int enable_q_mulmat;        // enable/disable offload quantized mulmat
    int precision_mode;         // 0: default 1:fp16
    int hvx_threads;
    int vtcm_size_in_mb;
    int enable_dlbc;
    int hwaccel_approach;       // 0: HWACCEL_QNN 1: HWACCEL_QNN_SINGLEGRAPH 2: HWACCEL_CDSP
    int hexagon_backend;        // 0: HEXAGON_BACKEND_QNNCPU 1: HEXAGON_BACKEND_QNNGPU 2: HEXAGON_BACKEND_QNNNPU / HEXAGON_BACKEND_CDSP
    int enable_rpc_ion_mempool; // enable/disable rpc ion memory pool
    int enable_rpc_dma_mempool; // enable/disable rpc dma memory pool
    const char * cfgfilename;
    const char * runtime_libpath;
    char ggml_hexagon_version[GGMLHEXAGON_TMPBUF_LEN];
};

static struct hexagon_appcfg_t g_hexagon_appcfg = {
        .print_qnn_internal_log = 0,
        .enable_perf            = 0,
        .print_tensors_info     = 0,
        .dump_op_info           = 0,
        .enable_q_mulmat        = 0,
        .precision_mode         = 0,
        .hvx_threads            = 4,
        .vtcm_size_in_mb        = 8,
        .enable_dlbc            = 1,
        .hwaccel_approach       = HWACCEL_CDSP,
        .hexagon_backend        = HEXAGON_BACKEND_CDSP,
        .enable_rpc_ion_mempool = 0,
        .enable_rpc_dma_mempool = 0,
        .cfgfilename            = "ggml-hexagon.cfg",
#if defined(__ANDROID__)
//Android command line program
        .runtime_libpath        = "/data/local/tmp/",
#elif defined(__linux__)
        .qnn_runtimelib_path    = "/tmp/",
#elif defined(_WIN32)
        .qnn_runtimelib_path    = "C:\\",
#endif
        .ggml_hexagon_version   = {"1.00"},
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
                .soc_desc          = "Qualcomm SnapDragon 8 Elite(aka 8 Gen 4)"},

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
static struct ggml_backend_hexagon_context g_hexagon_mgr[GGML_HEXAGON_MAX_DEVICES] = {
        {       .device               = 0,
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

        {       .device               = 1,
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

        {       .device               = 2,
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

static domain hexagon_supported_domains[] = {
        {ADSP_DOMAIN_ID, ADSP_DOMAIN},
        {MDSP_DOMAIN_ID, MDSP_DOMAIN},
        {SDSP_DOMAIN_ID, SDSP_DOMAIN},
        {CDSP_DOMAIN_ID, CDSP_DOMAIN},
        {CDSP1_DOMAIN_ID, CDSP1_DOMAIN}
};

//supported ggml op by HWACCEL_QNN
static constexpr const qnn_op_caps ggmlqnn_k_op_caps[] = {
        {true,  GGML_OP_NONE, 0, nullptr},
        {false, GGML_OP_DUP, 0, nullptr},
        {true,  GGML_OP_ADD, 2, QNN_OP_ELEMENT_WISE_ADD},
        {false, GGML_OP_ADD1, 0, nullptr},
        {false, GGML_OP_ACC, 0, nullptr},
        {true,  GGML_OP_SUB, 2, QNN_OP_ELEMENT_WISE_SUBTRACT},
        {true,  GGML_OP_MUL, 2, QNN_OP_ELEMENT_WISE_MULTIPLY},
        {true,  GGML_OP_DIV, 2, QNN_OP_ELEMENT_WISE_DIVIDE},
        {false, GGML_OP_SQR, 0, nullptr},
        {true,  GGML_OP_SQRT, 1, QNN_OP_ELEMENT_WISE_SQUARE_ROOT},
        {true,  GGML_OP_LOG, 1, QNN_OP_ELEMENT_WISE_LOG},
        {false, GGML_OP_SIN, 0, nullptr},
        {false, GGML_OP_COS, 0, nullptr},
        {false, GGML_OP_SUM, 0, nullptr},
        {false, GGML_OP_SUM_ROWS, 0, nullptr},
        {false, GGML_OP_MEAN, 0, nullptr},
        {false, GGML_OP_ARGMAX, 0, nullptr},
        {false, GGML_OP_COUNT_EQUAL, 0, nullptr},
        {false, GGML_OP_REPEAT, 0, nullptr},
        {false, GGML_OP_REPEAT_BACK, 0, nullptr},
        {false, GGML_OP_CONCAT, 0, nullptr},
        {false, GGML_OP_SILU_BACK, 0, nullptr},
        {false, GGML_OP_NORM, 0, nullptr},
        {false, GGML_OP_RMS_NORM, 0, nullptr},
        {false, GGML_OP_RMS_NORM_BACK, 0, nullptr},
        {false, GGML_OP_GROUP_NORM, 0, nullptr},
        {false, GGML_OP_L2_NORM, 0, nullptr},
        {true,  GGML_OP_MUL_MAT, 2, QNN_OP_MAT_MUL},
        {false, GGML_OP_MUL_MAT_ID, 0, nullptr},
        {false, GGML_OP_OUT_PROD, 0, nullptr},
        {false, GGML_OP_SCALE, 0, nullptr},
        {false, GGML_OP_SET, 0, nullptr},
        {false, GGML_OP_CPY, 0, nullptr},
        {false, GGML_OP_CONT, 0, nullptr},
        {false, GGML_OP_RESHAPE, 0, nullptr},
        {false, GGML_OP_VIEW, 0, nullptr},
        {false, GGML_OP_PERMUTE, 0, nullptr},
        {false, GGML_OP_TRANSPOSE, 0, nullptr},
        {false, GGML_OP_GET_ROWS, 0, nullptr},
        {false, GGML_OP_GET_ROWS_BACK, 0, nullptr},
        {false, GGML_OP_DIAG, 0, nullptr},
        {false, GGML_OP_DIAG_MASK_INF, 0, nullptr},
        {false, GGML_OP_DIAG_MASK_ZERO, 0, nullptr},
        {false, GGML_OP_SOFT_MAX, 0, nullptr},
        {false, GGML_OP_SOFT_MAX_BACK, 0, nullptr},
        {false, GGML_OP_ROPE, 0, nullptr},
        {false, GGML_OP_ROPE_BACK, 0, nullptr},
        {false, GGML_OP_CLAMP, 0, nullptr},
        {false, GGML_OP_CONV_TRANSPOSE_1D, 0, nullptr},
        {false, GGML_OP_IM2COL, 0, nullptr},
        {false, GGML_OP_IM2COL_BACK, 0, nullptr},
        {false, GGML_OP_CONV_TRANSPOSE_2D, 0, nullptr},
        {false, GGML_OP_POOL_1D, 0, nullptr},
        {false, GGML_OP_POOL_2D, 0, nullptr},
        {false, GGML_OP_POOL_2D_BACK, 0, nullptr},
        {false, GGML_OP_UPSCALE, 0, nullptr},
        {false, GGML_OP_PAD, 0, nullptr},
        {false, GGML_OP_PAD_REFLECT_1D, 0, nullptr},
        {false, GGML_OP_ARANGE, 0, nullptr},
        {false, GGML_OP_TIMESTEP_EMBEDDING, 0, nullptr},
        {false, GGML_OP_ARGSORT, 0, nullptr},
        {false, GGML_OP_LEAKY_RELU, 0, nullptr},
        {false, GGML_OP_FLASH_ATTN_EXT, 0, nullptr},
        {false, GGML_OP_FLASH_ATTN_BACK, 0, nullptr},
        {false, GGML_OP_SSM_CONV, 0, nullptr},
        {false, GGML_OP_SSM_SCAN, 0, nullptr},
        {false, GGML_OP_WIN_PART, 0, nullptr},
        {false, GGML_OP_WIN_UNPART, 0, nullptr},
        {false, GGML_OP_GET_REL_POS, 0, nullptr},
        {false, GGML_OP_ADD_REL_POS, 0, nullptr},
        {false, GGML_OP_RWKV_WKV6, 0, nullptr},
        {false, GGML_OP_GATED_LINEAR_ATTN, 0, nullptr},
        {false, GGML_OP_RWKV_WKV7, 0, nullptr},
        {false, GGML_OP_UNARY, 0, nullptr},
        {false, GGML_OP_MAP_UNARY, 0, nullptr},
        {false, GGML_OP_MAP_BINARY, 0, nullptr},
        {false, GGML_OP_MAP_CUSTOM1_F32, 0, nullptr},
        {false, GGML_OP_MAP_CUSTOM2_F32, 0, nullptr},
        {false, GGML_OP_MAP_CUSTOM3_F32, 0, nullptr},
        {false, GGML_OP_MAP_CUSTOM1, 0, nullptr},
        {false, GGML_OP_MAP_CUSTOM2, 0, nullptr},
        {false, GGML_OP_MAP_CUSTOM3, 0, nullptr},
        {false, GGML_OP_CROSS_ENTROPY_LOSS, 0, nullptr},
        {false, GGML_OP_CROSS_ENTROPY_LOSS_BACK, 0, nullptr},
        {false, GGML_OP_OPT_STEP_ADAMW, 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_ABS), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_SGN), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_NEG), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_STEP), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_TANH), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_ELU), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_RELU), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_SIGMOID), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_GELU), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_GELU_QUICK), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_SILU), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_HARDSWISH), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_HARDSIGMOID), 0, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_EXP), 0, nullptr}
};

static_assert(ggmlqnn_k_op_caps[GGML_OP_NONE].supported,    "GGML_OP_NONE is not true");
static_assert(ggmlqnn_k_op_caps[GGML_OP_ADD].supported,     "GGML_OP_ADD is not true");
static_assert(ggmlqnn_k_op_caps[GGML_OP_MUL].supported,     "GGML_OP_MUL is not true");
static_assert(ggmlqnn_k_op_caps[GGML_OP_MUL_MAT].supported, "GGML_OP_MUL_MAT is not true");
static_assert(std::size(ggmlqnn_k_op_caps) == (static_cast<size_t>(GGML_OP_COUNT) + static_cast<size_t>(GGML_UNARY_OP_COUNT)),
              "pls check ggmlqnn_k_op_caps and ensure is corresponding to latest ggml.h");

//supported ggml op by HWACCEL_CDSP
static constexpr const hexagon_op_caps ggmlhexagon_k_op_caps[] = {
        {true,  GGML_OP_NONE, 0, nullptr, nullptr},
        {false, GGML_OP_DUP, 0, nullptr, nullptr},
        {true,  GGML_OP_ADD, 2, "ggmlop_dsp_add", ggmlop_dsp_add},
        {false, GGML_OP_ADD1, 0, nullptr, nullptr},
        {false, GGML_OP_ACC, 0, nullptr, nullptr},
        {true,  GGML_OP_SUB, 2, "ggmlop_dsp_sub", ggmlop_dsp_sub},
        {true,  GGML_OP_MUL, 2, "ggmlop_dsp_mul", ggmlop_dsp_mul},
        {true,  GGML_OP_DIV, 2, "ggmlop_dsp_div", ggmlop_dsp_div},
        {false, GGML_OP_SQR, 0, nullptr, nullptr},
        {false,  GGML_OP_SQRT, 0, nullptr, nullptr},
        {false,  GGML_OP_LOG, 0, nullptr, nullptr},
        {false, GGML_OP_SIN, 0, nullptr, nullptr},
        {false, GGML_OP_COS, 0, nullptr, nullptr},
        {false, GGML_OP_SUM, 0, nullptr, nullptr},
        {false, GGML_OP_SUM_ROWS, 0, nullptr, nullptr},
        {false, GGML_OP_MEAN, 0, nullptr, nullptr},
        {false, GGML_OP_ARGMAX, 0, nullptr, nullptr},
        {false, GGML_OP_COUNT_EQUAL, 0, nullptr, nullptr},
        {false, GGML_OP_REPEAT, 0, nullptr, nullptr},
        {false, GGML_OP_REPEAT_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_CONCAT, 0, nullptr, nullptr},
        {false, GGML_OP_SILU_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_NORM, 0, nullptr, nullptr},
        {false, GGML_OP_RMS_NORM, 0, nullptr, nullptr},
        {false, GGML_OP_RMS_NORM_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_GROUP_NORM, 0, nullptr, nullptr},
        {false, GGML_OP_L2_NORM, 0, nullptr, nullptr},
        {true,  GGML_OP_MUL_MAT, 2, "ggmlop_dsp_mulmat", ggmlop_dsp_mulmat},
        {false, GGML_OP_MUL_MAT_ID, 0, nullptr, nullptr},
        {false, GGML_OP_OUT_PROD, 0, nullptr, nullptr},
        {false, GGML_OP_SCALE, 0, nullptr, nullptr},
        {false, GGML_OP_SET, 0, nullptr, nullptr},
        {false, GGML_OP_CPY, 0, nullptr, nullptr},
        {false, GGML_OP_CONT, 0, nullptr, nullptr},
        {false, GGML_OP_RESHAPE, 0, nullptr, nullptr},
        {false, GGML_OP_VIEW, 0, nullptr, nullptr},
        {false, GGML_OP_PERMUTE, 0, nullptr, nullptr},
        {false, GGML_OP_TRANSPOSE, 0, nullptr, nullptr},
        {false, GGML_OP_GET_ROWS, 0, nullptr, nullptr},
        {false, GGML_OP_GET_ROWS_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_DIAG, 0, nullptr, nullptr},
        {false, GGML_OP_DIAG_MASK_INF, 0, nullptr, nullptr},
        {false, GGML_OP_DIAG_MASK_ZERO, 0, nullptr, nullptr},
        {false, GGML_OP_SOFT_MAX, 0, nullptr, nullptr},
        {false, GGML_OP_SOFT_MAX_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_ROPE, 0, nullptr, nullptr},
        {false, GGML_OP_ROPE_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_CLAMP, 0, nullptr, nullptr},
        {false, GGML_OP_CONV_TRANSPOSE_1D, 0, nullptr, nullptr},
        {false, GGML_OP_IM2COL, 0, nullptr, nullptr},
        {false, GGML_OP_IM2COL_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_CONV_TRANSPOSE_2D, 0, nullptr, nullptr},
        {false, GGML_OP_POOL_1D, 0, nullptr, nullptr},
        {false, GGML_OP_POOL_2D, 0, nullptr, nullptr},
        {false, GGML_OP_POOL_2D_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_UPSCALE, 0, nullptr, nullptr},
        {false, GGML_OP_PAD, 0, nullptr, nullptr},
        {false, GGML_OP_PAD_REFLECT_1D, 0, nullptr, nullptr},
        {false, GGML_OP_ARANGE, 0, nullptr, nullptr},
        {false, GGML_OP_TIMESTEP_EMBEDDING, 0, nullptr, nullptr},
        {false, GGML_OP_ARGSORT, 0, nullptr, nullptr},
        {false, GGML_OP_LEAKY_RELU, 0, nullptr, nullptr},
        {false, GGML_OP_FLASH_ATTN_EXT, 0, nullptr, nullptr},
        {false, GGML_OP_FLASH_ATTN_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_SSM_CONV, 0, nullptr, nullptr},
        {false, GGML_OP_SSM_SCAN, 0, nullptr, nullptr},
        {false, GGML_OP_WIN_PART, 0, nullptr, nullptr},
        {false, GGML_OP_WIN_UNPART, 0, nullptr, nullptr},
        {false, GGML_OP_GET_REL_POS, 0, nullptr, nullptr},
        {false, GGML_OP_ADD_REL_POS, 0, nullptr, nullptr},
        {false, GGML_OP_RWKV_WKV6, 0, nullptr, nullptr},
        {false, GGML_OP_GATED_LINEAR_ATTN, 0, nullptr, nullptr},
        {false, GGML_OP_RWKV_WKV7, 0, nullptr, nullptr},
        {false, GGML_OP_UNARY, 0, nullptr, nullptr},
        {false, GGML_OP_MAP_UNARY, 0, nullptr, nullptr},
        {false, GGML_OP_MAP_BINARY, 0, nullptr, nullptr},
        {false, GGML_OP_MAP_CUSTOM1_F32, 0, nullptr, nullptr},
        {false, GGML_OP_MAP_CUSTOM2_F32, 0, nullptr, nullptr},
        {false, GGML_OP_MAP_CUSTOM3_F32, 0, nullptr, nullptr},
        {false, GGML_OP_MAP_CUSTOM1, 0, nullptr, nullptr},
        {false, GGML_OP_MAP_CUSTOM2, 0, nullptr, nullptr},
        {false, GGML_OP_MAP_CUSTOM3, 0, nullptr, nullptr},
        {false, GGML_OP_CROSS_ENTROPY_LOSS, 0, nullptr, nullptr},
        {false, GGML_OP_CROSS_ENTROPY_LOSS_BACK, 0, nullptr, nullptr},
        {false, GGML_OP_OPT_STEP_ADAMW, 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_ABS), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_SGN), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_NEG), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_STEP), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_TANH), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_ELU), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_RELU), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_SIGMOID), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_GELU), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_GELU_QUICK), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_SILU), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_HARDSWISH), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_HARDSIGMOID), 0, nullptr, nullptr},
        {false, static_cast<ggml_op>(GGML_UNARY_OP_EXP), 0, nullptr, nullptr}
};

static_assert(ggmlhexagon_k_op_caps[GGML_OP_NONE].supported,    "GGML_OP_NONE is not true");
static_assert(ggmlhexagon_k_op_caps[GGML_OP_ADD].supported,     "GGML_OP_ADD is not true");
static_assert(ggmlhexagon_k_op_caps[GGML_OP_MUL].supported,     "GGML_OP_MUL is not true");
static_assert(ggmlhexagon_k_op_caps[GGML_OP_MUL_MAT].supported, "GGML_OP_MUL_MAT is not true");
static_assert(std::size(ggmlhexagon_k_op_caps) == (static_cast<size_t>(GGML_OP_COUNT) + static_cast<size_t>(GGML_UNARY_OP_COUNT)),
              "pls check ggmlhexagon_k_op_caps and ensure is corresponding to latest ggml.h");

static int32_t g_qnntensor_idx = 0; //ensure every QNN tensor name is unique
static int32_t g_qnnopcfg_idx  = 0; //ensure every QNN opconfig name is unique

// =================================================================================================
//  section-2: ggml-hexagon internal troubleshooting function/class
// =================================================================================================
static void ggmlhexagon_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex ggmlhexagon_log_internal_mutex;
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];

    GGML_UNUSED(file);
#if !(defined __ANDROID__) || !(defined ANDROID)
    GGML_UNUSED(level);
#endif
    {
        std::lock_guard<std::mutex> lock(ggmlhexagon_log_internal_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix, GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            __android_log_print(ANDROID_LOG_INFO, "ggml-hexagon", "%s\n", s_ggmlhexagon_log_internal_buf);
            if (GGML_LOG_LEVEL_INFO == level) {
                printf("%s\n", s_ggmlhexagon_log_internal_buf);
            }
#else
            //for Snapdragon based WoA(Windows on ARM) device or Linux
            printf("%s\n", s_ggmlhexagon_log_internal_buf);
#endif
        }
        va_end(args);
    }
}

static void ggmlhexagon_print_tensors_info(const char * func_name, const ggml_backend_hexagon_context * ctx,
                const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    //skip sanity check of params because of performance concern
    if (0 == g_hexagon_appcfg.dump_op_info) {
        if (0 == g_hexagon_appcfg.print_tensors_info)
            return;
    }

    if (nullptr != func_name && nullptr != ctx) {
        GGMLHEXAGON_LOG_DEBUG("call %s in dev %s\n", func_name, ctx->name);
    }
    if (nullptr != src0) {
        GGMLHEXAGON_LOG_DEBUG(
                "%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                src0->name,
                src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
                src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    }
    if (nullptr != src1) {
        GGMLHEXAGON_LOG_DEBUG(
                "%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                src1->name,
                src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
                src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    }
    GGMLHEXAGON_LOG_DEBUG("%-6s: type = %i (%s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi, %5zi)",
                      dst->name,
                      dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                      dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    GGMLHEXAGON_LOG_DEBUG("\n");
}

static void ggmlhexagon_dump_op_info(const struct ggml_tensor * tensor) {
    //skip sanity check of params because of performance concern
    if (0 == g_hexagon_appcfg.dump_op_info)
        return;

    const struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor       * src1 = tensor->src[1];
    struct ggml_tensor       * dst  = const_cast<ggml_tensor *>(tensor);
    GGMLHEXAGON_LOG_DEBUG("op name:%s, tensor type:%s", ggml_op_name(tensor->op), ggml_type_name(tensor->type));
    ggmlhexagon_print_tensors_info(nullptr, nullptr, src0, src1, dst);
}

static void ggmlhexagon_dump_tensor_elements(const ggml_tensor * tensor) {
    float value = 0;
    std::ostringstream tmposs;
    if (tensor->type == GGML_TYPE_F32) {
        for (int h = 0; h < tensor->ne[3]; h++) {
            for (int i = 0; i < tensor->ne[2]; i++) {
                for (int j = 0; j < tensor->ne[1]; j++) {
                    for (int k = 0; k < tensor->ne[0]; k++) {
                        value = ((float *) tensor->data)[h * tensor->ne[2] + i * tensor->ne[1] +
                                                         j * tensor->ne[0] + k];
                        tmposs << std::setw(8) << std::fixed << std::setprecision(2) << value
                               << " ";
                    }
                    if (strlen(tmposs.str().c_str()) <= (GGMLHEXAGON_LOGBUF_LEN - 96)) {
                        GGMLHEXAGON_LOG_DEBUG("%s\n", tmposs.str().c_str());
                    }
                    tmposs.clear();
                    tmposs.str("");
                }
            }
        }
    }

    GGMLHEXAGON_LOG_DEBUG("\n");
}

static void ggmlhexagon_dump_tensor(const ggml_tensor * tensor, const char * name) {
    GGMLHEXAGON_LOG_DEBUG("dump ggml tensor %s(%s)\n", name, tensor->name);
    GGMLHEXAGON_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64", nb = (%5zi, %5zi, %5zi, %5zi)\n",
                      name,
                      tensor->type, ggml_type_name(tensor->type),
                      tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                      tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[2]);
    ggmlhexagon_dump_tensor_elements(tensor);

    GGMLHEXAGON_LOG_DEBUG("\n");
}

static const char * ggmlhexagon_get_hwaccel_approach_name(int hwaccle_approach) {
    switch (hwaccle_approach) {
        case HWACCEL_QNN:
            return "HWACCEL_QNN";
        case HWACCEL_QNN_SINGLEGRAPH:
            return "HWACCEL_QNN_SINGLEGRAPH";
        case HWACCEL_CDSP:
            return "HWACCEL_CDSP";
        default:
            return "unknown hwaccel approach";
    }
}

static void ggmlhexagon_get_timestring(char * p_currenttime) {
#if defined(__ANDROID__) || defined(__linux__)
    time_t n_seconds    = 0;
    struct tm now_time;

    if (nullptr == p_currenttime)
        return;

    time(&n_seconds);
    localtime_r(&n_seconds, &now_time);
    snprintf(p_currenttime, GGMLHEXAGON_TMPBUF_LEN, "%04d-%02d-%02d,%02d:%02d:%02d",
             now_time.tm_year + 1900, now_time.tm_mon + 1, now_time.tm_mday,
             now_time.tm_hour, now_time.tm_min, now_time.tm_sec);
#else
    //TODO: WoA
#endif
}

static void ggmlhexagon_probe_dspinfo(ggml_backend_hexagon_context * ctx);
static void ggmlhexagon_print_running_timestamp(ggml_backend_hexagon_context * ctx) {
    char timestamp[GGMLHEXAGON_TMPBUF_LEN];
    memset(timestamp, 0, GGMLHEXAGON_TMPBUF_LEN);

    GGMLHEXAGON_LOG_INFO("ggml_hexagon_version:             %s", g_hexagon_appcfg.ggml_hexagon_version);
    GGMLHEXAGON_LOG_INFO("hwaccel approach:                 %d(%s)", g_hexagon_appcfg.hwaccel_approach,
                     ggmlhexagon_get_hwaccel_approach_name(g_hexagon_appcfg.hwaccel_approach));
    GGMLHEXAGON_LOG_INFO("hexagon_backend:                  %d(%s)", g_hexagon_appcfg.hexagon_backend,
                     ggml_backend_hexagon_get_devname(g_hexagon_appcfg.hexagon_backend));
    ggmlhexagon_get_timestring(timestamp);
    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
        GGMLHEXAGON_LOG_INFO("offload quantize GGML_OP_MUL_MAT: %s", g_hexagon_appcfg.enable_q_mulmat ? "YES" : "NO");
        GGMLHEXAGON_LOG_INFO("using rpc ion memory pool:        %s", g_hexagon_appcfg.enable_rpc_ion_mempool ? "YES" : "NO");
        GGMLHEXAGON_LOG_INFO("using rpc dma memory pool:        %s", g_hexagon_appcfg.enable_rpc_dma_mempool ? "YES" : "NO");
        ggmlhexagon_probe_dspinfo(ctx);
    } else {
        GGMLHEXAGON_LOG_INFO("offload quantize GGML_OP_MUL_MAT: %s", g_hexagon_appcfg.enable_q_mulmat ? "YES" : "NO");
    }
    GGMLHEXAGON_LOG_INFO("running timestamp:%s", timestamp);
}

class hexagon_perf {
public:
    hexagon_perf(const std::string & perf_name) : _perf_name(std::move(perf_name)) {}
    hexagon_perf() = delete;
    hexagon_perf(const hexagon_perf & ) = delete;
    hexagon_perf & operator= (const hexagon_perf & ) = delete;

    void start() {
        if (0 == g_hexagon_appcfg.enable_perf)
            return;
        _begin_time = ggml_time_us();
    }

    void info() {
        if (0 == g_hexagon_appcfg.enable_perf)
            return;
        _end_time = ggml_time_us();
        _duration = (_end_time - _begin_time);
        GGMLHEXAGON_LOG_DEBUG("duration of %s : %lld microseconds\n", _perf_name.c_str(), _duration);
    }

private:
    int64_t _begin_time = 0LL;
    int64_t _end_time   = 0LL;
    int64_t _duration   = 0LL;
    std::string _perf_name;
};

class hexagon_appcfg {
public:
    void dump(std::function<void(const std::string &, const std::string &, const std::string &)> worker) {
        if (!_load_success) {
            GGMLHEXAGON_LOG_INFO("qnn cfg file %s not loaded", _cfg_filename.c_str());
            return;
        }
        auto iter = _hexagon_appcfg.begin();
        while (iter != _hexagon_appcfg.end()) {
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
            GGMLHEXAGON_LOG_WARN("can't open file %s", file_name.c_str());
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
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            return;
        }
        if (_hexagon_appcfg[section].find(key) == _hexagon_appcfg[section].end()) {
            return;
        }
        value = _hexagon_appcfg[section][key];
    }

    void get_intvalue(const std::string & section, const std::string & key, int & value, int default_value) {
        value = default_value;
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            return;
        }
        if (_hexagon_appcfg[section].find(key) == _hexagon_appcfg[section].end()) {
            return;
        }
        value = atol(_hexagon_appcfg[section][key].c_str());
    }

private:
    void ltrim(std::string & str) {
        if (str.empty()) return;
        size_t len  = 0;
        const char * temp = str.c_str();
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

    void trim(std::string & str) {
        ltrim(str);
        rtrim(str);
    }

    void set_section_keyvalue(std::string & section, std::string & key, std::string & value) {
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            std::unordered_map<std::string, std::string> kv_map;
            _hexagon_appcfg[section] = kv_map;
        }
        if (key != "" && value != "") _hexagon_appcfg[section][key] = value;
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
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> _hexagon_appcfg;
    bool _load_success = false;
    std::string _cfg_filename;
};

// =================================================================================================
//  section-3: helper function for WoA(Window on ARM)
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
//  section-4: general helper function
// =================================================================================================
static const char * ggmlhexagon_get_socmodel_desc(uint32_t soc_model) {
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

//0x68 -> 68, 0x69 -> 69, 0x73 -> 73, 0x75 -> 75, 0x79 -> 79
static size_t ggmlhexagon_htparch_hex_to_decimal(size_t htp_arch) {
    //naive algorithm
    int a = htp_arch / 16;
    int b = htp_arch % 16;
    return a * 10 + b;
}

static const char * ggmlhexagon_get_htparch_desc(size_t htp_arch) {
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

static struct qcom_socinfo * ggmlhexagon_get_socinfo_from_socmodel(uint32_t soc_model) {
    size_t items = sizeof(g_qnn_soc_info_table) / sizeof(g_qnn_soc_info_table[0]);
    for (size_t idx = 0; idx < items; idx++) {
        if (soc_model == g_qnn_soc_info_table[idx].soc_model) {
            return &g_qnn_soc_info_table[idx];
        }
    }
    return nullptr;
}

static struct qcom_socinfo * ggmlhexagon_get_socinfo_from_socmodel(size_t htp_arch) {
    size_t items = sizeof(g_qnn_soc_info_table) / sizeof(g_qnn_soc_info_table[0]);
    for (size_t idx = 0; idx < items; idx++) {
        if (htp_arch == g_qnn_soc_info_table[idx].htp_arch) {
            return &g_qnn_soc_info_table[idx];
        }
    }
    return nullptr;
}

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

static inline bool ggmlqnn_is_valid_params(ggml_backend_hexagon_context * ctx, const ggml_tensor * src0,
                                           const ggml_tensor * src1, ggml_tensor * dst) {
    if ((nullptr == ctx) || (nullptr == src0) || (nullptr == dst)) {
        GGMLHEXAGON_LOG_WARN("invalid params\n");
        return false;
    }

    qnn_instance * instance = ctx->instance;
    if (nullptr == instance) {
        GGMLHEXAGON_LOG_WARN("invalid params\n");
        return false;
    }

    return true;
}

static size_t ggmlhexagon_get_system_total_memory_in_bytes() {
#if defined(__ANDROID__) || defined(__linux__)
    struct sysinfo info = {};
    if (0 == sysinfo(&info)) {
        return (info.totalram + info.totalswap) * info.mem_unit;
    }
    size_t pages      = (size_t)sysconf(_SC_PHYS_PAGES);
    size_t page_size  = (size_t)sysconf(_SC_PAGE_SIZE);

    return pages * page_size;
#else
    //TODO: Snapdragon based WoA(Windows on ARM)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        GGMLHEXAGON_LOG_INFO("total physical mem:%llu Mb", statex.ullTotalPhys >> 20);
        GGMLHEXAGON_LOG_INFO("avail physical mem:%llu Mb", statex.ullAvailPhys >> 20);
        return statex.ullTotalPhys;
    }
    return 0;
#endif
}

static size_t ggmlhexagon_get_system_free_memory_in_bytes() {
#if defined(__ANDROID__) || defined(__linux__)
    struct sysinfo info = {};
    if (0 == sysinfo(&info)) {
        return (info.freeram + info.freeswap) * info.mem_unit;
    }
    size_t avail_pages = (size_t)sysconf(_SC_AVPHYS_PAGES);
    size_t page_size   = (size_t)sysconf(_SC_PAGE_SIZE);

    return avail_pages * page_size;
#else
    //TODO: Snapdragon based WoA(Windows on ARM)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        GGMLHEXAGON_LOG_INFO("total physical mem:%llu Mb", statex.ullTotalPhys >> 20);
        GGMLHEXAGON_LOG_INFO("avail physical mem:%llu Mb", statex.ullAvailPhys >> 20);
        return statex.ullAvailPhys;
    }
    return 0;
#endif
}

static bool ggmlhexagon_same_types(const ggml_backend_hexagon_context * ctx, const ggml_tensor * op_tensor) {
    GGML_UNUSED(ctx);
    ggml_tensor * src0 = op_tensor->src[0];
    ggml_tensor * src1 = op_tensor->src[1];
    if (nullptr != src1) {
        if (src0->type != op_tensor->type || src1->type != op_tensor->type) {
            return false;
        }
    } else {
        if (src0->type != op_tensor->type) {
            return false;
        }
    }

    if (src0->type != GGML_TYPE_F32)
        return false;

    return true;
}

static const char * ggmlhexagon_get_ggml_type_name(ggml_type type) {
    const auto * traits = ggml_get_type_traits(type);
    return traits->type_name;
}

static void ggmlhexagon_append_tensor_dimensions(const ggml_tensor * tensor, std::string & output) {
    char buffer[GGMLHEXAGON_TMPBUF_LEN] = {};
    const char * type_name = ggmlhexagon_get_ggml_type_name(tensor->type);
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

static size_t ggmlhexagon_get_op_index(const ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_UNARY) {
        return static_cast<size_t>(GGML_OP_COUNT) + static_cast<size_t>(ggml_get_unary_op(tensor));
    }

    return tensor->op;
}

static size_t ggmlhexagon_get_op_input_param_count(const ggml_tensor * op) {
    auto op_index = ggmlhexagon_get_op_index(op);
    GGML_ASSERT(op_index < std::size(ggmlqnn_k_op_caps));
    return ggmlhexagon_k_op_caps[op_index].input_param_count;
}

static void ggmlhexagon_get_opkey_from_op(const ggml_tensor * op, std::string & output) {
    GGML_ASSERT(op->op != GGML_OP_NONE);
    output += ggml_op_desc(op);
    output += ggmlhexagon_get_ggml_type_name(op->type);
    size_t param_count = ggmlhexagon_get_op_input_param_count(op);
    for (size_t i = 0; i < param_count; ++i) {
        auto * input = op->src[i];
        if (!input) {
            break;
        }
        output += '_';
        ggmlhexagon_append_tensor_dimensions(input, output);
    }
}

static void * ggmlhexagon_type_trait(ggml_backend_hexagon_context * ctx, ggml_tensor * op) {
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

static void ggmlhexagon_set_runtime_path(size_t device, const std::string & path) {
#if defined(__ANDROID__)
    if ((HEXAGON_BACKEND_QNNNPU == device) || (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach)) {
        std::string lib_runtime_path = path + ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images";
        if (0 == setenv("LD_LIBRARY_PATH", lib_runtime_path.c_str(), 1)) {
            GGMLHEXAGON_LOG_DEBUG("setenv LD_LIBRARY_PATH %s successfully", lib_runtime_path.c_str());
        } else {
            GGMLHEXAGON_LOG_ERROR("setenv LD_LIBRARY_PATH %s failure", lib_runtime_path.c_str());
        }

        std::string adsp_runtime_path = path + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp";
        if (0 == setenv("ADSP_LIBRARY_PATH", adsp_runtime_path.c_str(), 1)) {
            GGMLHEXAGON_LOG_DEBUG("setenv ADSP_LIBRARY_PATH %s successfully", adsp_runtime_path.c_str());
        } else {
            GGMLHEXAGON_LOG_ERROR("setenv ADSP_LIBRARY_PATH %s failure", adsp_runtime_path.c_str());
        }
    } else {
        if (0 == setenv("LD_LIBRARY_PATH",
                        (path +
                         ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images").c_str(),
                        1)) {
            GGMLHEXAGON_LOG_DEBUG("%s backend setenv successfully\n",
                                 ggml_backend_hexagon_get_devname(device));
        } else {
            GGMLHEXAGON_LOG_ERROR("%s backend setenv failure\n",
                                  ggml_backend_hexagon_get_devname(device));
        }
    }
#endif
}

static void ggmlhexagon_load_cfg() {
    //this function can be called in various scenarios
    static bool initialized = false;
    if (initialized) {
        GGMLHEXAGON_LOG_DEBUG("hexagon appcfg file already loaded\n");
        return;
    }
    char time_string[GGMLHEXAGON_TMPBUF_LEN];
    memset(time_string, 0, GGMLHEXAGON_TMPBUF_LEN);
    ggmlhexagon_get_timestring(time_string);
    GGMLHEXAGON_LOG_DEBUG("program running start time:%s", time_string);
    std::string cfg_filename = std::string(g_hexagon_appcfg.runtime_libpath) + std::string(g_hexagon_appcfg.cfgfilename);
    GGMLHEXAGON_LOG_INFO("load hexagon appcfg from %s", cfg_filename.c_str());
    hexagon_appcfg qnncfg_instance;
    qnncfg_instance.load(cfg_filename);
    qnncfg_instance.dump([](const std::string & section, const std::string & key, const std::string value) {
        std::ostringstream  tmposs;
        tmposs << "section[" << std::setw(10) << std::left << section << "],[" << std::setw(25) << std::left << key << "] = [" << value << "]";
        GGMLHEXAGON_LOG_INFO("%s", tmposs.str().c_str());
    });
    std::string precision_mode;
    std::string ggml_hexagon_version;
    qnncfg_instance.get_stringvalue("general", "ggml_hexagon_version", ggml_hexagon_version, "1.00");
    qnncfg_instance.get_intvalue("general", "print_qnn_internal_log", g_hexagon_appcfg.print_qnn_internal_log, 0);
    qnncfg_instance.get_intvalue("general", "enable_perf", g_hexagon_appcfg.enable_perf, 1);
    qnncfg_instance.get_intvalue("general", "print_tensors_info", g_hexagon_appcfg.print_tensors_info, 0);
    qnncfg_instance.get_intvalue("general", "dump_op_info", g_hexagon_appcfg.dump_op_info, 0);
    qnncfg_instance.get_intvalue("general", "hwaccel_approach", g_hexagon_appcfg.hwaccel_approach, HWACCEL_CDSP);
    qnncfg_instance.get_intvalue("general", "hexagon_backend", g_hexagon_appcfg.hexagon_backend, HEXAGON_BACKEND_CDSP);
    qnncfg_instance.get_intvalue("general", "enable_q_mulmat", g_hexagon_appcfg.enable_q_mulmat, 0);
    qnncfg_instance.get_intvalue("qnn", "hvx_threads", g_hexagon_appcfg.hvx_threads, 4);
    qnncfg_instance.get_intvalue("qnn", "vtcm_size_in_mb", g_hexagon_appcfg.vtcm_size_in_mb, 8);
    qnncfg_instance.get_intvalue("qnn", "enable_dlbc", g_hexagon_appcfg.enable_dlbc, 1);
    qnncfg_instance.get_stringvalue("qnn", "precision_mode", precision_mode, "fp32");
    qnncfg_instance.get_intvalue("cdsp", "enable_rpc_ion_mempool", g_hexagon_appcfg.enable_rpc_ion_mempool, 1);
    qnncfg_instance.get_intvalue("cdsp", "enable_rpc_dma_mempool", g_hexagon_appcfg.enable_rpc_dma_mempool, 0);
    GGMLHEXAGON_LOG_INFO("internal ggml_hexagon_version=%s", g_hexagon_appcfg.ggml_hexagon_version);
    GGMLHEXAGON_LOG_INFO("external ggml_hexagon_version=%s", ggml_hexagon_version.c_str());
    GGMLHEXAGON_LOG_INFO("hwaccel_approach=%d(%s)", g_hexagon_appcfg.hwaccel_approach,
                         ggmlhexagon_get_hwaccel_approach_name(g_hexagon_appcfg.hwaccel_approach));
    GGMLHEXAGON_LOG_INFO("hexagon_backend=%d(%s)", g_hexagon_appcfg.hexagon_backend,
                         ggml_backend_hexagon_get_devname(g_hexagon_appcfg.hexagon_backend));
    GGMLHEXAGON_LOG_INFO("runtime libpath=%s", g_hexagon_appcfg.runtime_libpath);

    if (precision_mode.find("fp16") != std::string::npos) {
        g_hexagon_appcfg.precision_mode = 1;
    } else {
        g_hexagon_appcfg.precision_mode = 0;
    }

    ggmlhexagon_set_runtime_path(HEXAGON_BACKEND_CDSP, g_hexagon_appcfg.runtime_libpath);

    initialized = true;
}

static bool ggmlhexagon_check_valid_appcfg() {
    bool is_valid_appcfg = true;

    GGMLHEXAGON_LOG_DEBUG("user's specified hwaccel approach=%d(%s)", g_hexagon_appcfg.hwaccel_approach,
                          ggmlhexagon_get_hwaccel_approach_name(g_hexagon_appcfg.hwaccel_approach));
    GGMLHEXAGON_LOG_DEBUG("user's specified hexagon_backend=%d", g_hexagon_appcfg.hexagon_backend);
    if (g_hexagon_appcfg.hexagon_backend >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_INFO("using default ggml backend");
        is_valid_appcfg = false;
    }

    if (HWACCEL_QNN_SINGLEGRAPH == g_hexagon_appcfg.hwaccel_approach) {
        GGMLHEXAGON_LOG_INFO("HWACCEL_QNN_SINGLEGRAPH not supported");
        is_valid_appcfg = false;
    }

    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
        if (HEXAGON_BACKEND_CDSP != g_hexagon_appcfg.hexagon_backend) {
            GGMLHEXAGON_LOG_INFO("hwaccel_approach HWACCEL_CDSP must match with hexagon_backend HEXAGON_BACKEND_CDSP");
            is_valid_appcfg = false;
        }

        if ((1 == g_hexagon_appcfg.enable_rpc_ion_mempool) && (1 == g_hexagon_appcfg.enable_rpc_dma_mempool)) {
            GGMLHEXAGON_LOG_INFO("rpc ion mempool and rpc dma mempool cannot be enabled at the same time");
            is_valid_appcfg = false;
        }

        if (1 == g_hexagon_appcfg.enable_rpc_dma_mempool) {
            GGMLHEXAGON_LOG_INFO("rpc dma mempool not supported");
            is_valid_appcfg = false;
        }
    }

    if (!is_valid_appcfg) {
        GGMLHEXAGON_LOG_INFO("it seems there is wrong configuration in ggml-hexagon.cfg, will using the default ggml backend accordingly");
    }
    return is_valid_appcfg;
}

// =================================================================================================
//  section-5: QNN helper function/class
// =================================================================================================
//ensure every QNN tensor/opcfg name is unique, threadsafe is not required at the moment
static void ggmlqnn_reset_idx() {
    g_qnntensor_idx = 0;
    g_qnnopcfg_idx = 0;
}

static void ggmlqnn_inc_idx(int idx_type) {
    switch (idx_type) {
        case QNN_TENSOR_INDEX:
            g_qnntensor_idx++;
            break;
        case QNN_OPCFG_INDEX:
            g_qnnopcfg_idx++;
            break;
        default:
            break;
    }
}

static int32_t ggmlqnn_get_idx(int idx_type) {
    switch (idx_type) {
        case QNN_TENSOR_INDEX:
            return g_qnntensor_idx;
        case QNN_OPCFG_INDEX:
            return g_qnnopcfg_idx;
        default:
            break;
    }

    //it's not make sense, just for fix compiler warning
    return g_qnntensor_idx;
}

static intptr_t ggmlqnn_align_to(size_t alignment, intptr_t offset) {
    return offset % alignment == 0 ? offset
                                   : offset +
                                     (static_cast<intptr_t>(alignment) -
                                      offset % static_cast<intptr_t>(alignment));
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
    //TODO:behaviour is not exactly same to Android&Linux
    GGML_UNUSED(maxlen);
    return strdup(source);
#endif
}

static inline uint32_t ggmlqnn_get_tensorid(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.id;
    }
    return 0u;
}

static inline const char * ggmlqnn_get_tensorname(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.name;
    }
    return nullptr;
}

static inline Qnn_TensorType_t ggmlqnn_get_tensortype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.type;
    }
    return QNN_TENSOR_TYPE_UNDEFINED;
}

static inline Qnn_TensorDataFormat_t ggmlqnn_get_tensor_dataformat(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataFormat;
    }
    return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

static inline Qnn_DataType_t ggmlqnn_get_tensor_datatype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataType;
    }
    return QNN_DATATYPE_UNDEFINED;
}

static inline Qnn_QuantizeParams_t ggmlqnn_get_tensor_quantparams(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.quantizeParams;
    }
    return QNN_QUANTIZE_PARAMS_INIT;
}

static inline uint32_t ggmlqnn_get_tensor_rank(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.rank;
    }
    return 0u;
}

static inline uint32_t * ggmlqnn_get_tensor_dimensions(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dimensions;
    }
    return nullptr;
}

static inline Qnn_TensorMemType_t ggmlqnn_get_tensor_memtype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.memType;
    }
    return QNN_TENSORMEMTYPE_UNDEFINED;
}

static inline void ggmlqnn_set_tensor_id(Qnn_Tensor_t & tensor, uint32_t id) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.id = id;
    }
}

static inline void ggmlqnn_set_tensor_name(Qnn_Tensor_t & tensor, const char * name) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.name = name;
    }
}

static inline void ggmlqnn_set_tensor_type(Qnn_Tensor_t & tensor, Qnn_TensorType_t type) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.type = type;
    }
}

static inline void ggmlqnn_set_tensor_dataformat(Qnn_Tensor_t & tensor, Qnn_TensorDataFormat_t format) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataFormat = format;
    }
}

static inline void ggmlqnn_set_tensor_datatype(Qnn_Tensor_t & tensor, Qnn_DataType_t dataType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataType = dataType;
    }
}

static inline void ggmlqnn_set_tensor_quantparams(Qnn_Tensor_t & tensor, Qnn_QuantizeParams_t params) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.quantizeParams = params;
    }
}

static inline void ggmlqnn_set_tensor_rank(Qnn_Tensor_t & tensor, uint32_t rank) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.rank = rank;
    }
}

static inline void ggmlqnn_set_tensor_dimensions(Qnn_Tensor_t & tensor, uint32_t * dims) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dimensions = dims;
    }
}

static inline void ggmlqnn_set_tensor_memtype(Qnn_Tensor_t & tensor, Qnn_TensorMemType_t memType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memType = memType;
    }
}

static inline void ggmlqnn_set_tensor_clientbuf(Qnn_Tensor_t & tensor, Qnn_ClientBuffer_t clientBuf) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.clientBuf = clientBuf;
    }
}

static inline void ggmlqnn_set_tensor_memhandle(Qnn_Tensor_t & tensor, Qnn_MemHandle_t handle) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memHandle = handle;
    }
}

static int ggmlqnn_deep_copy_qnntensor(Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
    int err = 0;

    dst.version = src.version;
    ggmlqnn_set_tensor_name(dst, ggmlqnn_strndup(ggmlqnn_get_tensorname(src), std::string(ggmlqnn_get_tensorname(src)).size()));
    if (nullptr == ggmlqnn_get_tensorname(dst)) {
        return 1;
    }
    ggmlqnn_set_tensor_id(dst, ggmlqnn_get_tensorid(src));
    ggmlqnn_set_tensor_type(dst, ggmlqnn_get_tensortype(src));
    ggmlqnn_set_tensor_dataformat(dst, ggmlqnn_get_tensor_dataformat(src));
    ggmlqnn_set_tensor_datatype(dst, ggmlqnn_get_tensor_datatype(src));
    ggmlqnn_set_tensor_memtype(dst, ggmlqnn_get_tensor_memtype(src));

    if (ggmlqnn_get_tensor_memtype(src) == QNN_TENSORMEMTYPE_RAW) {
        Qnn_ClientBuffer_t client_buf = {nullptr, 0};
        ggmlqnn_set_tensor_clientbuf(dst, client_buf);
    } else if (ggmlqnn_get_tensor_memtype(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
        ggmlqnn_set_tensor_memhandle(dst, nullptr);
    } else {
        return 1;
    }

    Qnn_QuantizeParams_t src_qparam      = ggmlqnn_get_tensor_quantparams(src);
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
        ggmlqnn_set_tensor_quantparams(dst, src_qparam_cpy);
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
        ggmlqnn_set_tensor_quantparams(dst, src_qparam_cpy);
    } else {
        ggmlqnn_set_tensor_quantparams(dst, src_qparam);
    }

    uint32_t rank = ggmlqnn_get_tensor_rank(src);
    ggmlqnn_set_tensor_rank(dst, rank);
    size_t dim_size       = GGML_MAX_DIMS * sizeof(uint32_t);
    uint32_t * dimensions = (uint32_t *)malloc(dim_size);
    if (nullptr == dimensions) {
        GGMLHEXAGON_LOG_WARN("deep_copy_qnn_tensors() allocation error while copying tensor %s\n", ggmlqnn_get_tensorname(src));
        return 1;
    }
    ggmlqnn_memscpy(dimensions, dim_size, ggmlqnn_get_tensor_dimensions(src), dim_size);
    ggmlqnn_set_tensor_dimensions(dst, dimensions);

    return err;
}

static int ggmlqnn_free_qnntensor(Qnn_Tensor_t * tensor) {
    int err = 0;
    free((void *) ggmlqnn_get_tensorname(*tensor));
    Qnn_QuantizeParams_t src_qparam     = ggmlqnn_get_tensor_quantparams(*tensor);
    Qnn_QuantizationEncoding_t encoding = src_qparam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        free(src_qparam.axisScaleOffsetEncoding.scaleOffset);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        free(src_qparam.bwAxisScaleOffsetEncoding.scales);
        if (src_qparam.bwAxisScaleOffsetEncoding.offsets != nullptr) {
            free(src_qparam.bwAxisScaleOffsetEncoding.offsets);
        }
    }
    free(ggmlqnn_get_tensor_dimensions(*tensor));
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

static void ggmlqnn_get_qnn_dimensions_from_ggml_dimensions(uint32_t * qnn_dimensions, const uint32_t * ggml_dimensions, uint32_t rank) {
    if (rank > GGML_MAX_DIMS) {
        GGMLHEXAGON_LOG_WARN("invalid params");
        return;
    }
    if (nullptr == qnn_dimensions || nullptr == ggml_dimensions) {
        GGMLHEXAGON_LOG_WARN("invalid params");
        return;
    }
    for (size_t idx = 0; idx < GGML_MAX_DIMS; idx++)
        qnn_dimensions[idx] = ggml_dimensions[idx];

    if (rank >= 2) {
        qnn_dimensions[rank - 1] = ggml_dimensions[rank - 2];
        qnn_dimensions[rank - 2] = ggml_dimensions[rank - 1];
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
            GGMLHEXAGON_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_interface;
    }

    const QNN_INTERFACE_VER_TYPE & get_qnn_raw_interface() {
        if (!_qnn_interface.is_loaded()) {
            GGMLHEXAGON_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_interface;
    }

    const QNN_SYSTEM_INTERFACE_VER_TYPE & get_qnn_raw_system_interface() {
        if (!_qnn_interface.is_loaded()) {
            GGMLHEXAGON_LOG_WARN("pls check why _qnn_interface is not loaded\n");
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
    int init_qnn_graph(const std::string & graph_name, HEXAGONBackend device, size_t vtcm_size_in_mb = 8, size_t hvx_threads = 8);

    int finalize_qnn_graph();

    bool is_valid_graph() const { return _qnn_graph_handle != nullptr; }

    int htp_init_perfinfra();

    int htp_set_rpc_polling();

    int htp_set_high_performance_mode();

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

    HEXAGONBackend get_device_id() {
        return _device_id;
    }

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

    void htp_probe_rpc_meminfo();

    void htp_print_info();

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

    qnn_profile_level _profile_level        = PROFILE_OFF;

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
    size_t                             _rpcmem_usage    = 0;   // mempool usage in bytes
    size_t                             _rpcmem_capacity = 0;   // mempool size  in bytes

    std::string _graph_name;
    HEXAGONBackend _device_id;
    void * _rpc_lib_handle      = nullptr;
    bool       _enable_qnn_rpc  = false; //TODO:unknown issue with QNN RPC feature

    qnn_instance(const qnn_instance &) = delete;
    void operator=(const qnn_instance &) = delete;

    qnn_instance(qnn_instance &&) = delete;
    void operator=(qnn_instance &&) = delete;
};

void * qnn_instance::alloc_rpcmem_internal(size_t bytes, size_t alignment) {
    if (!_rpcmem_initialized) {
        GGMLHEXAGON_LOG_WARN("rpc memory not initialized\n");
        return nullptr;
    }

    auto allocate_bytes = static_cast<int32_t>(bytes + alignment);
    void * buf = _pfn_rpc_mem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocate_bytes);
    if (nullptr == buf) {
        GGMLHEXAGON_LOG_WARN("failed to allocate rpc memory\n");
        return nullptr;
    }

    auto aligned_buf = reinterpret_cast<void *>(ggmlqnn_align_to(alignment,
                                                reinterpret_cast<intptr_t>(buf)));
    bool status = _rpcmem_store_map.insert(std::pair<void *, void *>(aligned_buf, buf)).second;
    if (!status) {
        GGMLHEXAGON_LOG_WARN("failed to allocate rpc memory\n");
        _pfn_rpc_mem_free(buf);
    }
    return aligned_buf;
}

void * qnn_instance::alloc_rpcmem(size_t bytes, size_t alignment) {
    if (_rpcmem_usage > (_rpcmem_capacity - (8 * SIZE_IN_MB))) { // reserve 8Mbytes in rpc mempool
        GGMLHEXAGON_LOG_WARN("rpc mempool capacity: %d MiB, usage: %d MiB", _rpcmem_capacity / SIZE_IN_MB, _rpcmem_usage / SIZE_IN_MB);
        return nullptr;
    }

    auto aligned_buf = alloc_rpcmem_internal(bytes, alignment);
    if (nullptr == aligned_buf)
        return nullptr;
    _rpcmem_usage_map.insert(std::pair<void *, size_t>(aligned_buf, bytes));

    _rpcmem_usage += bytes;
    return aligned_buf;
}

void qnn_instance::free_rpcmem(void * buf) {
    size_t rpcbuffer_size = 0;
    if (!_rpcmem_initialized) {
        GGMLHEXAGON_LOG_WARN("rpc memory not initialized\n");
    } else if (0 == _rpcmem_store_map.count(buf)) {
        GGMLHEXAGON_LOG_WARN("no allocated tensor\n");
    } else {
        GGMLHEXAGON_LOG_DEBUG("free rpc mem %p", _rpcmem_store_map[buf]);
        for (std::unordered_map<void *, size_t>::iterator it = _rpcmem_usage_map.begin();
             it != _rpcmem_usage_map.end();
             it++) {
            void * rpcbuffer = it->first;
            if (buf == rpcbuffer) {
                rpcbuffer_size = it->second;
                _rpcmem_usage -= rpcbuffer_size;
            }
        }
        if (rpcbuffer_size != 0) {
            _rpcmem_usage_map.erase(buf);
        } else {
            GGMLHEXAGON_LOG_WARN("it shouldn't happen, pls check why?");
        }
        _pfn_rpc_mem_free(_rpcmem_store_map[buf]);
        _rpcmem_store_map.erase(buf);
    }
}

void qnn_instance::free_rpcmem() {
    if (_rpcmem_store_map.empty()) {
        GGMLHEXAGON_LOG_WARN("no rpcmem allocated\n");
        return;
    }

    for (std::unordered_map<void *, void *>::iterator it = _rpcmem_store_map.begin();
         it != _qnn_mem_set.end();
         it++) {
        void * rpcbuffer = it->second;
        GGMLHEXAGON_LOG_DEBUG("free rpc buffer %p", rpcbuffer);
        _pfn_rpc_mem_free(rpcbuffer);
    }
    _rpcmem_store_map.clear();
    _rpcmem_usage_map.clear();
    _rpcmem_usage = 0;
}

int32_t qnn_instance::rpcmem_to_fd(void * buf) {
    int32_t mem_fd = -1;
    if (!is_rpcmem_initialized()) {
        GGMLHEXAGON_LOG_WARN("rpc memory not initialized\n");
    } else {
        mem_fd = _pfn_rpc_mem_to_fd(buf);
    }

    return mem_fd;
}

int qnn_instance::register_rpcmem(void * p_data, Qnn_Tensor_t * p_tensor) {
    if (nullptr == p_data || (nullptr == p_tensor)) {
        GGMLHEXAGON_LOG_WARN("invalid param\n");
        return 1;
    }

    if (!is_rpcmem_initialized()) {
        GGMLHEXAGON_LOG_WARN("rpc memory not initialized\n");
        return 2;
    }

    if (is_rpcmem_registered((QNN_VER_PTR(*p_tensor)->memHandle))) {
        GGMLHEXAGON_LOG_WARN("tensor %s has been registered shared memory\n", (QNN_VER_PTR(*p_tensor)->name));
        return 3;
    }

    int32_t mem_fd = rpcmem_to_fd(p_data);
    if (-1 == mem_fd) {
        GGMLHEXAGON_LOG_WARN("failed to get file descriptor\n");
        return 4;
    }
    GGMLHEXAGON_LOG_DEBUG("mem_fd %d\n", mem_fd);
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
        GGMLHEXAGON_LOG_WARN("failed to register shared memory, error %d, %s\n", QNN_GET_ERROR_CODE(error), strerror(error));
        return 5;
    } else {
        GGMLHEXAGON_LOG_INFO("tensor %s successfully register shared memory\n", (QNN_VER_PTR(*p_tensor)->name));
    }
    QNN_VER_PTR(*p_tensor)->memHandle = handle;
    _qnn_mem_set.insert((std::pair<void*, Qnn_MemHandle_t>(p_data, handle)));

    return 0;
}

Qnn_MemHandle_t  qnn_instance::register_rpcmem(void * p_data, const uint32_t rank, uint32_t * dimensions, Qnn_DataType_t data_type) {
    if (!p_data) {
        GGMLHEXAGON_LOG_WARN("invalid param");
        return nullptr;
    }

    if (!is_rpcmem_initialized()) {
        GGMLHEXAGON_LOG_WARN("rpc memory not initialized");
        return nullptr;
    }

    if (is_rpcmem_registered(p_data)) {
        GGMLHEXAGON_LOG_WARN("rpc memory already registered");
        return _qnn_rpc_buffer_to_handles[p_data];
    }

    int32_t mem_fd = rpcmem_to_fd(p_data);
    if (mem_fd == -1) {
        GGMLHEXAGON_LOG_WARN("failed to get file descriptor");
        return nullptr;
    }

    GGMLHEXAGON_LOG_DEBUG("mem_fd %d", mem_fd);
    Qnn_MemDescriptor_t descriptor = {
            {rank, dimensions, nullptr},
            data_type, QNN_MEM_TYPE_ION,
            {{mem_fd}}
    };
    Qnn_MemHandle_t handle = nullptr;
    Qnn_ErrorHandle_t error = _qnn_interface.qnn_mem_register(_qnn_context_handle, &descriptor, /*numDescriptors=*/1, &handle);
    if (error != QNN_SUCCESS) {
        GGMLHEXAGON_LOG_WARN("failed to register shared memory, error %d, %s", QNN_GET_ERROR_CODE(error), strerror(error));
        return nullptr;
    }

    _qnn_rpc_buffer_to_handles.insert({p_data, handle});
    GGMLHEXAGON_LOG_DEBUG("successfully register shared memory handler: %p", handle);
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
    GGMLHEXAGON_LOG_WARN("can't find rpcmem from qnn mem handle %p", mem_handle);
    return nullptr;
}

void qnn_instance::unregister_rpcmem() {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (_qnn_mem_set.empty()) {
        GGMLHEXAGON_LOG_WARN("no rpcmem registered\n");
    }

    for (std::unordered_map<void *, Qnn_MemHandle_t>::iterator it = _qnn_mem_set.begin();
         it != _qnn_mem_set.end();
         it++) {
        Qnn_MemHandle_t mem_handle = it->second;
        error = _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
        if (error != QNN_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed to unregister shared memory, error %d\n", QNN_GET_ERROR_CODE(error));
        } else {
            GGMLHEXAGON_LOG_DEBUG("unregister shared memory ok");
        }
    }
    _qnn_mem_set.clear();
}

void qnn_instance::unregister_rpcmem(Qnn_MemHandle_t mem_handle) {
    Qnn_ErrorHandle_t error = _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
    if (error != QNN_SUCCESS) {
        GGMLHEXAGON_LOG_WARN("failed to unregister shared memory, error %d", QNN_GET_ERROR_CODE(error));
    }

    auto it = std::find_if(_qnn_mem_set.begin(), _qnn_mem_set.end(),
                           [mem_handle](const auto &kv) { return kv.second == mem_handle; });
    if (it == _qnn_mem_set.end()) {
        GGMLHEXAGON_LOG_WARN("failed to find shared memory handler: %p", mem_handle);
        return;
    }

    _qnn_mem_set.erase(it);
}

bool qnn_instance::is_rpcmem_allocated(void * buf) {
    return _rpcmem_store_map.count(buf) != 0U;
}

int qnn_instance::load_backend(std::string & lib_path, const QnnSaver_Config_t ** saver_config) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    GGMLHEXAGON_LOG_DEBUG("lib_path:%s\n", lib_path.c_str());

    void * lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (nullptr == lib_handle) {
        GGMLHEXAGON_LOG_WARN("can not open QNN library %s, with error: %s", lib_path.c_str(), dlerror());
        return 1;
    }

    auto get_providers = ggmlqnn_load_qnn_functionpointers<_pfn_QnnInterface_getProviders *>(
                               lib_handle,
                               "QnnInterface_getProviders");
    if (nullptr == get_providers) {
        GGMLHEXAGON_LOG_WARN("can not load symbol QnnInterface_getProviders : %s", dlerror());
        return 2;
    }

    std::uint32_t num_providers = 0;
    const QnnInterface_t ** provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        GGMLHEXAGON_LOG_WARN("failed to get providers, error %d", QNN_GET_ERROR_CODE(error));
        return 3;
    }
    GGMLHEXAGON_LOG_DEBUG("num_providers=%d\n", num_providers);
    if (num_providers != _required_num_providers) {
        GGMLHEXAGON_LOG_WARN("providers is %d instead of required %d", num_providers, _required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        GGMLHEXAGON_LOG_WARN("failed to get qnn interface providers\n");
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
        GGMLHEXAGON_LOG_WARN("unable to find a valid qnn interface\n");
        return 6;
    } else {
        GGMLHEXAGON_LOG_INFO("find a valid qnn interface\n");
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
            GGMLHEXAGON_LOG_WARN("failed to saver_initialize，error %d", QNN_GET_ERROR_CODE(error));
            return 7;
        }
    } else {
        GGMLHEXAGON_LOG_WARN("saver_initialize is null\n");
    }

    return 0;
}

int qnn_instance::unload_backend() {
    int dlclose_error = 0;
    dlclose_error = dlclose(_loaded_lib_handle);
    if (dlclose_error != 0) {
        GGMLHEXAGON_LOG_WARN("failed to close QNN backend %d, error %s\n", _backend_id, dlerror());
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
    GGMLHEXAGON_LOG_DEBUG("system_lib_path:%s\n", system_lib_path.c_str());

    _system_lib_handle = dlopen(system_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (nullptr == _system_lib_handle) {
        GGMLHEXAGON_LOG_WARN("can not open QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
        //re-try with default path of QNN binary runtime lib
        _lib_path = std::string(g_hexagon_appcfg.runtime_libpath);
#if !defined(__ANDROID__) && !defined(__linux__)
        system_lib_path = _lib_path + "QnnSystem.dll";
#else
        system_lib_path = _lib_path + "libQnnSystem.so";
#endif
        _system_lib_handle = dlopen(system_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (nullptr == _system_lib_handle) {
            GGMLHEXAGON_LOG_WARN("can not open QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
            return 1;
        }
    }

    auto * get_providers = reinterpret_cast<_pfn_QnnSystemInterface_getProviders *>(dlsym(
            _system_lib_handle, "QnnSystemInterface_getProviders"));
    if (nullptr == get_providers) {
        GGMLHEXAGON_LOG_WARN("can not load QNN symbol QnnSystemInterface_getProviders: %s\n", dlerror());
        return 2;
    }

    uint32_t num_providers = 0;
    const QnnSystemInterface_t ** provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        GGMLHEXAGON_LOG_WARN("failed to get providers, error %d\n", QNN_GET_ERROR_CODE(error));
        return 3;
    }

    if (num_providers != _required_num_providers) {
        GGMLHEXAGON_LOG_WARN("providers is %d instead of required %d\n", num_providers, _required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        GGMLHEXAGON_LOG_WARN("can not get providers\n");
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
        GGMLHEXAGON_LOG_WARN("unable to find a valid qnn system interface\n");
        return 6;
    } else {
        GGMLHEXAGON_LOG_INFO("find a valid qnn system interface\n");
    }
    set_qnn_raw_system_interface(qnn_system_interface);

    _qnn_interface.set_qnn_system_interface(provider_list[0]);

    _qnn_interface.qnn_system_context_create(&_qnn_system_handle);
    if (nullptr == _qnn_system_handle) {
        GGMLHEXAGON_LOG_WARN("can not create QNN system contenxt\n");
    } else {
        GGMLHEXAGON_LOG_INFO("initialize qnn system successfully\n");
    }

    return 0;
}

int qnn_instance::unload_system() {
    int result = 0;

    if (nullptr == _system_lib_handle) {
        GGMLHEXAGON_LOG_DEBUG("system lib handle is null\n");
        return 1;
    }

    if (nullptr != _qnn_system_handle) {
        result = _qnn_interface.qnn_system_context_free(_qnn_system_handle);
        if (result != QNN_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed to free QNN system context\n");
        }
        _qnn_system_handle = nullptr;
    }

    int dlclose_error = dlclose(_system_lib_handle);
    if (dlclose_error != 0) {
        GGMLHEXAGON_LOG_WARN("failed to close QnnSystem library, error %s\n", dlerror());
        return 2;
    }

    _system_lib_handle = nullptr;

    return result;
}

static void ggmlqnn_sdk_logcallback(const char * fmt,
                                 QnnLog_Level_t level,
                                 uint64_t timestamp,
                                 va_list argp) {

    if (0 == g_hexagon_appcfg.print_qnn_internal_log)
        return;

    static std::mutex log_mutex;
    static unsigned char s_ggmlqnn_sdk_logbuf[GGMLHEXAGON_LOGBUF_LEN];

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
        memset(s_ggmlqnn_sdk_logbuf, 0, GGMLHEXAGON_LOGBUF_LEN);
        vsnprintf(reinterpret_cast<char *const>(s_ggmlqnn_sdk_logbuf), GGMLHEXAGON_LOGBUF_LEN, fmt, argp);
        GGMLHEXAGON_LOG_DEBUG("%8.1fms [%-7s] %s\n", ms, log_level_desc, s_ggmlqnn_sdk_logbuf);
    }
}

int qnn_instance::qnn_init(const QnnSaver_Config_t ** saver_config) {
    GGMLHEXAGON_LOG_DEBUG("enter qni_init\n");

    _device_id = HEXAGON_BACKEND_GGML;
    if (_backend_name.find("QnnCpu") != std::string::npos) {
        _device_id = HEXAGON_BACKEND_QNNCPU;
    }
    if (_backend_name.find("QnnGpu") != std::string::npos) {
        _device_id = HEXAGON_BACKEND_QNNGPU;
    }
    if (_backend_name.find("QnnHtp") != std::string::npos) {
        _device_id = HEXAGON_BACKEND_QNNNPU;
    }
    if (HEXAGON_BACKEND_GGML == _device_id) {
        GGMLHEXAGON_LOG_INFO("user specified qnn backend is ggml, skip QNN initialize");
        return 0;
    }

    if (0 != load_system()) {
        GGMLHEXAGON_LOG_WARN("can not load QNN system lib, pls check why?\n");
        return 1;
    } else {
        GGMLHEXAGON_LOG_DEBUG("load QNN system lib successfully\n");
    }

    std::string backend_lib_path = _lib_path + _backend_name;

    int is_load_ok = load_backend(backend_lib_path, saver_config);
    if (0 != is_load_ok) {
        GGMLHEXAGON_LOG_WARN("failed to load QNN backend\n");
        return 2;
    }

    _qnn_interface.set_qnn_interface(_loaded_backend);
#if 1
    _qnn_interface.qnn_log_create(ggmlqnn_sdk_logcallback, _qnn_log_level, &_qnn_log_handle);
#else
    _qnn_raw_interface.logCreate(ggmlqnn_sdk_logcallback, _qnn_log_level, &_qnn_log_handle);
#endif
    if (nullptr == _qnn_log_handle) {
        GGMLHEXAGON_LOG_WARN("why failed to initialize qnn log\n"); //NPU backend not work on Qualcomm SoC based low-end phone
        return 3;
    } else {
        GGMLHEXAGON_LOG_DEBUG("initialize qnn log successfully\n");
    }

    std::vector<const QnnBackend_Config_t *> temp_backend_config;
    _qnn_interface.qnn_backend_create(_qnn_log_handle,
                      temp_backend_config.empty() ? nullptr : temp_backend_config.data(),
                      &_qnn_backend_handle);
    if (nullptr == _qnn_backend_handle) {
        GGMLHEXAGON_LOG_WARN("why failed to initialize qnn backend\n");
        return 4;
    } else {
        GGMLHEXAGON_LOG_DEBUG("initialize qnn backend successfully\n");
    }

    if (nullptr != _qnn_raw_interface.propertyHasCapability) {
        auto qnnstatus = _qnn_raw_interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnstatus) {
            GGMLHEXAGON_LOG_WARN("device property is not supported\n");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnstatus) {
            GGMLHEXAGON_LOG_WARN("device property is not known to backend\n");
        }
    }

    Qnn_ErrorHandle_t qnnstatus = QNN_SUCCESS;
    if (_device_id == HEXAGON_BACKEND_QNNNPU) {
        const QnnDevice_PlatformInfo_t * p_info = nullptr;
        qcom_socinfo soc_info = {};
        qnnstatus = _qnn_raw_interface.deviceGetPlatformInfo(nullptr, &p_info);
        if (QNN_SUCCESS == qnnstatus) {
            GGMLHEXAGON_LOG_INFO("device counts %d\n", p_info->v1.numHwDevices);
            QnnDevice_HardwareDeviceInfo_t *         infos    = p_info->v1.hwDevices;
            QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = {};
            for (uint32_t i = 0; i < p_info->v1.numHwDevices; i++) {
                GGMLHEXAGON_LOG_INFO("deviceID:%d, deviceType:%d, numCores %d\n", (int) infos[i].v1.deviceId,
                             (int) infos[i].v1.deviceType, (int) infos[i].v1.numCores);
                QnnDevice_DeviceInfoExtension_t devinfo = infos[i].v1.deviceInfoExtension;
                chipinfo                                = devinfo->onChipDevice;
                size_t htp_arch                         = (size_t) chipinfo.arch;
                GGMLHEXAGON_LOG_INFO("htp_type:%d(%s)\n", devinfo->devType,
                             (devinfo->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) ? "ON_CHIP" : "");
                soc_info = { chipinfo.socModel, htp_arch, chipinfo.vtcmSize, {} };
            }
            _qnn_raw_interface.deviceFreePlatformInfo(nullptr, p_info);
        } else {
            GGMLHEXAGON_LOG_WARN("failed to get platform info, are we in emulator?\n");
            soc_info = { NONE, UNKNOWN_SM, 0, {} };
        }

        QnnHtpDevice_CustomConfig_t soc_customconfig;
        soc_customconfig.option    = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        soc_customconfig.socModel  = soc_info.soc_model;
        QnnDevice_Config_t soc_devconfig;
        soc_devconfig.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        soc_devconfig.customConfig = &soc_customconfig;

        /*
        QnnHtpDevice_CustomConfig_t arch_customconfig;
        arch_customconfig.option        = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
        arch_customconfig.arch.arch     = (QnnHtpDevice_Arch_t)soc_info.htp_arch;
        arch_customconfig.arch.deviceId = 0;
        QnnDevice_Config_t arch_devconfig;
        arch_devconfig.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        arch_devconfig.customConfig = &arch_customconfig;
        */
        const QnnDevice_Config_t * p_deviceconfig[] = { &soc_devconfig, nullptr };
        qnnstatus = _qnn_raw_interface.deviceCreate(_qnn_log_handle, p_deviceconfig, &_qnn_device_handle);
    } else {
        qnnstatus = _qnn_interface.qnn_device_create(_qnn_log_handle, nullptr, &_qnn_device_handle);
    }
    if (QNN_SUCCESS != qnnstatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnstatus) {
        GGMLHEXAGON_LOG_WARN("failed to create QNN device\n");
    } else {
        GGMLHEXAGON_LOG_INFO("create device successfully\n");
    }

    if (PROFILE_OFF != _profile_level) {
        GGMLHEXAGON_LOG_INFO("profiling turned on; level = %d", _profile_level);
        if (PROFILE_BASIC == _profile_level) {
            GGMLHEXAGON_LOG_INFO("basic profiling requested. creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_BASIC, &_qnn_profile_handle)) {
                GGMLHEXAGON_LOG_WARN("unable to create profile handle in the backend\n");
                return 5;
            } else {
                GGMLHEXAGON_LOG_DEBUG("initialize qnn profile successfully\n");
            }
        } else if (PROFILE_DETAIL == _profile_level) {
            GGMLHEXAGON_LOG_INFO("detailed profiling requested. Creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_DETAILED, &_qnn_profile_handle)) {
                GGMLHEXAGON_LOG_WARN("unable to create profile handle in the backend\n");
                return 6;
            } else {
                GGMLHEXAGON_LOG_DEBUG("initialize qnn profile successfully\n");
            }
        }
    }

#if defined(__ANDROID__) || defined(__linux__)
    std::filesystem::path full_path(std::string(g_hexagon_appcfg.runtime_libpath) + "libcdsprpc.so");
    full_path /= std::filesystem::path("libcdsprpc.so").filename();
    _rpc_lib_handle = dlopen(full_path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (nullptr == _rpc_lib_handle) {
        GGMLHEXAGON_LOG_WARN("failed to load %s\n", full_path.c_str());
        _rpc_lib_handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    }
#else
    _rpc_lib_handle = dlopen("libcdsprpc.dll", RTLD_NOW | RTLD_LOCAL);
#endif
    if (nullptr == _rpc_lib_handle) {
        GGMLHEXAGON_LOG_WARN("failed to load qualcomm's rpc lib, error:%s\n", dlerror());
        return 7;
    } else {
        GGMLHEXAGON_LOG_DEBUG("load rpcmem lib successfully\n");
        set_rpcmem_initialized(true);
    }
    _pfn_rpc_mem_init   = reinterpret_cast<pfn_rpc_mem_init>(dlsym(_rpc_lib_handle, "rpcmem_init"));
    _pfn_rpc_mem_deinit = reinterpret_cast<pfn_rpc_mem_deinit>(dlsym(_rpc_lib_handle, "rpcmem_deinit"));
    _pfn_rpc_mem_alloc  = reinterpret_cast<pfn_rpc_mem_alloc>(dlsym(_rpc_lib_handle,"rpcmem_alloc"));
    _pfn_rpc_mem_free   = reinterpret_cast<pfn_rpc_mem_free>(dlsym(_rpc_lib_handle, "rpcmem_free"));
    _pfn_rpc_mem_to_fd  = reinterpret_cast<pfn_rpc_mem_to_fd>(dlsym(_rpc_lib_handle,"rpcmem_to_fd"));
    if (nullptr == _pfn_rpc_mem_alloc || nullptr == _pfn_rpc_mem_free || nullptr == _pfn_rpc_mem_to_fd) {
        GGMLHEXAGON_LOG_WARN("unable to access symbols in QNN RPC lib, dlerror(): %s", dlerror());
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
        GGMLHEXAGON_LOG_WARN("why failed to initialize qnn context, error:%s\n", strerror(errno));
        return 9;
    } else {
        GGMLHEXAGON_LOG_DEBUG("initialize qnn context successfully\n");
    }

    if (_backend_name.find("Htp") != std::string::npos) {
        htp_print_info();
        htp_probe_rpc_meminfo();

        if (0 != htp_init_perfinfra()) {
            GGMLHEXAGON_LOG_WARN("initialize HTP performance failure");
        }

        htp_enter_performance_mode();
        htp_set_memory_grow_size();

        if (enable_qnn_rpc()) {
            GGMLHEXAGON_LOG_INFO("NPU RPC feature enabled with QNN-NPU backend");
        } else {
            GGMLHEXAGON_LOG_INFO("NPU RPC feature disabled with QNN-NPU backend");
        }
    }

    print_backend_info();

    GGMLHEXAGON_LOG_DEBUG("leave qni_init\n");

    return 0;
}

int qnn_instance::qnn_finalize() {
    int ret_status = 0;
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    GGMLHEXAGON_LOG_INFO("enter %s\n", __func__);
    ggmlqnn_reset_idx();

    free_rpcmem();
    unregister_rpcmem();

    if (nullptr != _pfn_rpc_mem_deinit)
        _pfn_rpc_mem_deinit();

    if (0 != dlclose(_rpc_lib_handle)) {
        GGMLHEXAGON_LOG_WARN("failed to unload qualcomm's rpc lib, error:%s\n", dlerror());
    } else {
        GGMLHEXAGON_LOG_DEBUG("succeed to close rpcmem lib\n");
    }

    if (nullptr != _qnn_context_handle) {
        error = _qnn_interface.qnn_context_free(_qnn_context_handle, _qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed to free QNN context_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_context_handle = nullptr;
    }

    if (nullptr != _qnn_profile_handle) {
        error = _qnn_interface.qnn_profile_free(_qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed to free QNN profile_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_profile_handle = nullptr;
    }

    if (nullptr != _qnn_device_handle) {
        error = _qnn_interface.qnn_device_free(_qnn_device_handle);
        if (error != QNN_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed to free QNN device_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_device_handle = nullptr;
    }

    if (nullptr != _qnn_backend_handle) {
        error = _qnn_interface.qnn_backend_free(_qnn_backend_handle);
        if (error != QNN_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed to free QNN backend_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_backend_handle = nullptr;

    }

    if (nullptr != _qnn_log_handle) {
        error = _qnn_interface.qnn_log_free(_qnn_log_handle);
        if (error != QNN_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed to free QNN log_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_log_handle = nullptr;
    }

    unload_backend();
    unload_system();

    GGMLHEXAGON_LOG_INFO("leave %s\n", __func__);
    return ret_status;
}

int qnn_instance::init_qnn_graph(const std::string & graph_name, HEXAGONBackend device, size_t vtcm_size_in_mb, size_t hvx_threads) {
    _graph_name = graph_name;
    _device_id = device;

    //GGMLHEXAGON_LOG_DEBUG("[%s][%s]created", ggml_backend_hexagon_get_devname(device), graph_name.c_str());

    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    if (HEXAGON_BACKEND_QNNNPU == device) {
        QnnHtpGraph_CustomConfig_t hvx_config;
        hvx_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
        hvx_config.numHvxThreads = hvx_threads;
        QnnGraph_Config_t graph_hvx_config;
        graph_hvx_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_hvx_config.customConfig = &hvx_config;

        QnnHtpGraph_CustomConfig_t dlbc_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        dlbc_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        dlbc_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
        if (0 == g_hexagon_appcfg.enable_dlbc)
            dlbc_config.optimizationOption.floatValue = 0.0; // set to 0.0 to turn off DLBC
        else
            dlbc_config.optimizationOption.floatValue = 1.0; // set to 1.0 to turn on  DLBC
        QnnGraph_Config_t graph_dlbc_config;
        graph_dlbc_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_dlbc_config.customConfig = &dlbc_config;

        QnnHtpGraph_CustomConfig_t opt_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        opt_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        opt_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        opt_config.optimizationOption.floatValue = 1; // 1 / 3
        QnnGraph_Config_t graph_opt_config;
        graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_opt_config.customConfig = &opt_config;

        QnnHtpGraph_CustomConfig_t vtcm_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        vtcm_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
        vtcm_config.vtcmSizeInMB = vtcm_size_in_mb;
        QnnGraph_Config_t graph_vtcm_config;
        graph_vtcm_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_vtcm_config.customConfig = &vtcm_config;

        std::vector<const QnnGraph_Config_t *> graph_configs;
        graph_configs.push_back(&graph_hvx_config);
        graph_configs.push_back(&graph_dlbc_config);
        graph_configs.push_back(&graph_vtcm_config);
        graph_configs.push_back(&graph_opt_config);
        if (1 == g_hexagon_appcfg.precision_mode) {
            QnnHtpGraph_CustomConfig_t fp16_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
            fp16_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
            fp16_config.precision = QNN_PRECISION_FLOAT16;
            QnnGraph_Config_t graph_fp16_config;
            graph_fp16_config.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_fp16_config.customConfig = &fp16_config;
            graph_configs.push_back(&graph_fp16_config);
        }
        graph_configs.push_back(nullptr);
        error = _qnn_interface.qnn_graph_create(_qnn_context_handle, graph_name.c_str(), graph_configs.data(), &_qnn_graph_handle);
        //GGMLHEXAGON_LOG_DEBUG("[%s][%s]created graph %p", ggml_backend_hexagon_get_devname(device), graph_name.c_str(), _qnn_graph_handle);
    } else {
        error = _qnn_interface.qnn_graph_create(_qnn_context_handle, graph_name.c_str(), nullptr, &_qnn_graph_handle);
    }
    if (QNN_SUCCESS != error) {
        GGMLHEXAGON_LOG_ERROR("[%s][%s]failed to create qnn graph, error: %s",
                      ggml_backend_hexagon_get_devname(device), graph_name.c_str(),
                      ggmlqnn_get_qnnerror_string(error));
        return error;
    }

    GGMLHEXAGON_LOG_DEBUG("[%s]create graph %s succeed", ggml_backend_hexagon_get_devname(device), graph_name.c_str());
    if (HEXAGON_BACKEND_QNNNPU == device) {
        htp_set_n_hvx_threads(hvx_threads);
    }
    return QNN_SUCCESS;
}

int qnn_instance::init_qnn_graph(const char * graph_name, bool debug, uint8_t do_node_validation,
                                 const QnnGraph_Config_t ** graph_configs) {
    Qnn_ErrorHandle_t result = 0;

    if (nullptr == graph_name) {
        GGMLHEXAGON_LOG_WARN("graph name is null\n");
        return 1;
    }

    if (!_graph_name.empty()) {
        GGMLHEXAGON_LOG_WARN("qnn model for graph %s already initialized\n", graph_name);
        return 2;
    }

    if (!do_node_validation) {
        GGMLHEXAGON_LOG_WARN("node validation disabled, backend will not perform op validation prior to adding node\n");
    }

    _graph_name             = graph_name;
    _debug_tensor           = debug;
    _do_node_validations    = do_node_validation;

    result = _qnn_raw_interface.graphCreate(_qnn_context_handle,
                                            graph_name,
                                            graph_configs,
                                            &_qnn_graph_handle);
    if (QNN_GRAPH_NO_ERROR != result || nullptr == _qnn_graph_handle) {
        GGMLHEXAGON_LOG_WARN("failed to create graph in qnn context\n");
        return 3;
    } else {
        GGMLHEXAGON_LOG_DEBUG("succeed to create graph %s, %p\n", graph_name, _qnn_graph_handle);
    }

    return 0;
}

int qnn_instance::finalize_qnn_graph() {
    if (nullptr != _qnn_graph_handle) {
        if (_qnn_raw_interface.graphFinalize(_qnn_graph_handle,
                                             _qnn_profile_handle, nullptr)
                                             != QNN_GRAPH_NO_ERROR) {
            GGMLHEXAGON_LOG_WARN("finalizing graph failure\n");
            return 1;
        }
    } else {
        GGMLHEXAGON_LOG_DEBUG("qnn graph handle is null\n");
    }

    return 0;
}

int qnn_instance::htp_init_perfinfra() {
    QnnDevice_Infrastructure_t device_infra = nullptr;
    Qnn_ErrorHandle_t error = _qnn_raw_interface.deviceGetInfrastructure(&device_infra);
    if (QNN_SUCCESS != error) {
        GGMLHEXAGON_LOG_WARN("failed to get qnn device infra\n");
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
    //TODO:hardcode to 0 and 0 although it's correct
    _qnn_htp_device_id      = device_id;
    _qnn_htp_core_id        = core_id;

    return 0;
}

void qnn_instance::htp_probe_rpc_meminfo() {
    size_t candidate_size   = 0;
    uint8_t * rpc_buffer    = nullptr;
    size_t probe_slots[]    = {1024, 1536, 2048 - 48, 2048};
    size_t probe_counts     = sizeof(probe_slots) / sizeof(size_t);
    for (size_t idx = 0; idx < probe_counts; idx++) {
        rpc_buffer = static_cast<uint8_t *>(alloc_rpcmem_internal(probe_slots[idx] * SIZE_IN_MB, 4));
        if (nullptr == rpc_buffer) {
            GGMLHEXAGON_LOG_DEBUG("alloc rpcmem %d (MiB) failure during probe rpc memory info, reason: %s\n", probe_slots[idx], strerror(errno));
            break;
        } else {
            candidate_size = probe_slots[idx];
            free_rpcmem(rpc_buffer);
            rpc_buffer = nullptr;
        }
    }
    if (candidate_size > _rpcmem_capacity)
        _rpcmem_capacity = candidate_size * SIZE_IN_MB;

    free_rpcmem();
    _rpcmem_usage = 0;
    GGMLHEXAGON_LOG_INFO("capacity of rpc ion memory %d MiB\n", _rpcmem_capacity / SIZE_IN_MB);
}

void qnn_instance::htp_print_info() {
    const QnnDevice_PlatformInfo_t * p_info = nullptr;
    _qnn_raw_interface.deviceGetPlatformInfo(nullptr, &p_info);
    GGMLHEXAGON_LOG_DEBUG("HTP device counts %d", p_info->v1.numHwDevices);
    QnnDevice_HardwareDeviceInfo_t * infos = p_info->v1.hwDevices;
    for (size_t i = 0; i < p_info->v1.numHwDevices; i++) {
        GGMLHEXAGON_LOG_DEBUG("HTP deviceID:%d, deviceType:%d, numCores %d", infos[i].v1.deviceId,
                         infos[i].v1.deviceType, infos[i].v1.numCores);
        QnnDevice_DeviceInfoExtension_t devinfo = infos[i].v1.deviceInfoExtension;
        QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = devinfo->onChipDevice;
        QnnHtpDevice_Arch_t htp_arch = chipinfo.arch;
        GGMLHEXAGON_LOG_DEBUG("HTP_TYPE:%d(%s)", devinfo->devType,
                         (devinfo->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) ? "QNN_HTP_DEVICE_TYPE_ON_CHIP" : "QNN_HTP_DEVICE_TYPE_UNKNOWN");
        GGMLHEXAGON_LOG_DEBUG("qualcomm soc_model:%d(%s), htp_arch:%d(%s), vtcm_size:%d MiB，" \
                             "dlbc_support:%d, signedpd_support:%d", \
                             chipinfo.socModel, ggmlhexagon_get_socmodel_desc(chipinfo.socModel), \
                             htp_arch, ggmlhexagon_get_htparch_desc(htp_arch), chipinfo.vtcmSize, \
                             chipinfo.dlbcSupport, chipinfo.signedPdSupport);
        struct qcom_socinfo * socinfo = ggmlhexagon_get_socinfo_from_socmodel(chipinfo.socModel);
        g_hexagon_mgr[HEXAGON_BACKEND_QNNNPU].socinfo = { chipinfo.socModel, htp_arch, chipinfo.vtcmSize, {}};
        if (nullptr != socinfo) {
            memcpy(g_hexagon_mgr[HEXAGON_BACKEND_QNNNPU].socinfo.soc_desc, socinfo->soc_desc, sizeof(socinfo->soc_desc));
            GGMLHEXAGON_LOG_DEBUG("soc info:%s", socinfo->soc_desc);
        } else {
            memcpy(g_hexagon_mgr[HEXAGON_BACKEND_QNNNPU].socinfo.soc_desc, "unknown", 7);
            GGMLHEXAGON_LOG_DEBUG("soc info:unknown");
        }
    }
    _qnn_raw_interface.deviceFreePlatformInfo(nullptr, p_info);
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

        GGMLHEXAGON_LOG_INFO("%s: %s", name, status);
    };

    GGMLHEXAGON_LOG_INFO("QNN backend properties:");
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
    Qnn_ErrorHandle_t result = _qnn_htp_perfinfra->setMemoryConfig(_qnn_htp_device_id, _qnn_htp_core_id, memory_config);
    if (QNN_SUCCESS != result) {
        GGMLHEXAGON_LOG_WARN("failed to set HTP memory config");
    } else {
        GGMLHEXAGON_LOG_INFO("succeed to set HTP memory config");
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
    Qnn_ErrorHandle_t result     = _qnn_raw_interface.graphSetConfig(_qnn_graph_handle, graph_configs);
    if (QNN_SUCCESS != result) {
        GGMLHEXAGON_LOG_WARN("failed to set QNN graph config: set hvx threads %d", n_threads);
    } else {
        //GGMLHEXAGON_LOG_DEBUG("succeed to set QNN graph config: set hvx threads %d", n_threads);
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

    const QnnHtpPerfInfrastructure_PowerConfig_t * power_configs[] = {
            &dcvs_v3_config,
            &hmx_config,
            &rpc_ctrl_config,
            &rpc_poll_config,
            nullptr,
    };
    Qnn_ErrorHandle_t ret = _qnn_htp_perfinfra->setPowerConfig(_qnn_htp_powerconfig_id, power_configs);
    if (ret != QNN_SUCCESS) {
        GGMLHEXAGON_LOG_WARN("failed to set HTP power config");
    } else {
        GGMLHEXAGON_LOG_INFO("succeed to set HTP power config");
    }
}

static uint8_t * ggmlqnn_create_rpc_buffer(qnn_instance * instance, const ggml_tensor * ggml_tensor, Qnn_Tensor_t * qnn_tensor, bool b_copydata) {
    if (nullptr == instance || nullptr == ggml_tensor || nullptr == qnn_tensor) {
        GGMLHEXAGON_LOG_WARN("invalid params\n");
        return nullptr;
    }

    uint8_t * qnn_rpcbuffer = static_cast<uint8_t *>(instance->alloc_rpcmem(ggml_nbytes(ggml_tensor), 4));
    if (nullptr == qnn_rpcbuffer) {
        GGMLHEXAGON_LOG_WARN("alloc rpcmem failure, %s\n", strerror(errno));
        return nullptr;
    } else {
        GGMLHEXAGON_LOG_DEBUG("alloc rpcmem %p successfully\n", qnn_rpcbuffer);
    }
    if (b_copydata)
        memcpy(qnn_rpcbuffer, ggml_tensor->data, ggml_nbytes(ggml_tensor));
    instance->register_rpcmem(qnn_rpcbuffer, qnn_tensor);
    return qnn_rpcbuffer;
}

static Qnn_OpConfig_t ggmlqnn_create_op_config(const char * name, const char * package, const char * type,
                                               Qnn_Param_t * params, uint32_t num_params,
                                               Qnn_Tensor_t * inputs, uint32_t num_inputs,
                                               Qnn_Tensor_t * outputs, uint32_t num_outputs) {

    char opcfg_name[GGML_MAX_NAME] = {};

    //ensure the opcfg name is unique
    if (nullptr == name) {
        snprintf(opcfg_name, GGML_MAX_NAME, "opcfg_%-8d", ggmlqnn_get_idx(QNN_OPCFG_INDEX));
    } else {
        snprintf(opcfg_name, GGML_MAX_NAME, "opcfg_%s_%-8d", name, ggmlqnn_get_idx(QNN_OPCFG_INDEX));
    }
    //GGMLHEXAGON_LOG_DEBUG("create qnn opconfig %s", opcfg_name);
    ggmlqnn_inc_idx(QNN_OPCFG_INDEX);

    Qnn_OpConfigV1_t v1 = {opcfg_name, package, type,
                           num_params, params,
                           num_inputs, inputs,
                           num_outputs, outputs
    };
    Qnn_OpConfig_t opcfg = {QNN_OPCONFIG_VERSION_1, {v1}};

    return opcfg;
}

static Qnn_Tensor_t * ggmlqnn_create_general_tensor(qnn_instance * instance, Qnn_GraphHandle_t graph_handle,
                                                    const ggml_tensor * tensor, const char * name,
                                                    Qnn_TensorType_t qnn_tensor_type,
                                                    Qnn_DataType_t qnn_data_type,
                                                    uint32_t rank, uint32_t * dims,
                                                    void * data, uint32_t data_size,
                                                    bool b_transpose = false) {
    Qnn_ErrorHandle_t error         = QNN_SUCCESS;
    char tensor_name[GGML_MAX_NAME] = {};

    //ensure the tensor name is unique
    if (nullptr == name) {
        snprintf(tensor_name, GGML_MAX_NAME, "tensor_%-8d", ggmlqnn_get_idx(QNN_TENSOR_INDEX));
    } else {
        snprintf(tensor_name, GGML_MAX_NAME, "tensor_%s%-8d", name, ggmlqnn_get_idx(QNN_TENSOR_INDEX));
    }
    GGMLHEXAGON_LOG_DEBUG("init_tensor %s", tensor_name);
    ggmlqnn_inc_idx(QNN_TENSOR_INDEX);

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
    }

    Qnn_Tensor_t qnn_tensor = {
            .version = QNN_TENSOR_VERSION_1,
            .v1 = {
                    .id = 0,
                    .name = tensor_name,
                    .type = qnn_tensor_type,
                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType = qnn_data_type,
                    .quantizeParams = {.encodingDefinition = QNN_DEFINITION_UNDEFINED,
                            .quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED},
                    .rank = rank,
                    .dimensions = tensor_dims,
                    .memType = QNN_TENSORMEMTYPE_RAW,
                    .clientBuf = {.data = nullptr, .dataSize = 0}
            }
    };
    Qnn_Tensor_t * p_qnn_tensor = (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    if (nullptr == p_qnn_tensor) {
        GGMLHEXAGON_LOG_WARN("calloc failed");
        return nullptr;
    }
    error = ggmlqnn_deep_copy_qnntensor(qnn_tensor, *p_qnn_tensor);
    if (error != QNN_SUCCESS) {
        free(p_qnn_tensor);
        GGMLHEXAGON_LOG_WARN("init tensor failed");
        return  nullptr;
    }

    bool enable_npu_rpc = (instance->enable_qnn_rpc() && instance->get_device_id() == HEXAGON_BACKEND_QNNNPU);
    if (enable_npu_rpc) {
        QNN_VER_PTR(*p_qnn_tensor)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        QNN_VER_PTR(*p_qnn_tensor)->clientBuf = {.data=nullptr, .dataSize=0};
    } else {
        QNN_VER_PTR(*p_qnn_tensor)->clientBuf = {data, data_size};
    }
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = instance->get_qnn_raw_interface();
    CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_qnn_tensor));

    return p_qnn_tensor;
}

static Qnn_Tensor_t * ggmlqnn_create_compute_tensor(qnn_instance * instance, Qnn_GraphHandle_t graph_handle,
                          const ggml_tensor * tensor, Qnn_TensorType_t tensor_type) {
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
    Qnn_Tensor_t * p_qnn_tensor = ggmlqnn_create_general_tensor(instance, graph_handle, tensor, nullptr,
                                      qnn_tensor_type, qnn_data_type,
                                      ggml_n_dims(tensor), dimensions,
                                      nullptr, 0);
    return p_qnn_tensor;
}

// =================================================================================================
//  section-6: hwaccel approach through QNN: offload GGML op to QNN backend
// =================================================================================================
/*
 * provide a general skeleton to offload ggml op to QNN backend: perform element-wise
 * operation on 1/2 input tensors and 1 output tensors
*/
static void ggmlqnn_compute_elementwise(ggml_backend_hexagon_context * ctx, ggml_tensor * op) {
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
    size_t qnn_op_index                         = ggmlhexagon_get_op_index(op);
    const char * qnn_op_name                    = ggmlqnn_k_op_caps[qnn_op_index].qnn_op_name;
    size_t input_param_count                    = ggmlqnn_k_op_caps[qnn_op_index].input_param_count;
    std::string ggml_op_name_string             = std::string("ggml_") + ggml_op_name(op->op);
    const char * ggml_op_name                   = ggml_op_name_string.c_str();

    std::string graph_name;
    ggmlhexagon_get_opkey_from_op(op, graph_name);

    hexagon_perf op_perf(graph_name);
    op_perf.start();

    bool enable_npu_rpc = instance->enable_qnn_rpc() && ctx->device == HEXAGON_BACKEND_QNNNPU;
    if (ctx->qnn_singlenode_graph_map.find(graph_name) != ctx->qnn_singlenode_graph_map.end()) {
        //retrieve computational resource from cached QNN graph
        qnn_singlenode_res_t & graph_item = ctx->qnn_singlenode_graph_map[graph_name];
        graph_handle                      = std::get<0>(graph_item);
        qnn_ptensors_t & ptensors         = std::get<1>(graph_item);
        p_tensor0  = ptensors[0];
        if (2 == input_param_count) {
            p_tensor1 = ptensors[1];
            p_tensor2 = ptensors[2];
        } else {
            //now p_tensor1 is nullptr
            p_tensor2 = ptensors[1];
        }
    } else {
        GGML_ASSERT(instance->get_device_id() == ctx->device);
        GGMLHEXAGON_LOG_INFO("graph name %s", graph_name.c_str());
        //create QNN graph
        error = instance->init_qnn_graph(graph_name, static_cast<HEXAGONBackend>(ctx->device),
                                         g_hexagon_appcfg.vtcm_size_in_mb,
                                         g_hexagon_appcfg.hvx_threads);
        if (QNN_SUCCESS != error) {
            GGMLHEXAGON_LOG_WARN("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        graph_handle = instance->get_qnn_graph_handle();

        //GGMLHEXAGON_LOG_DEBUG("graph_handle %p", graph_handle);
        //create computational tensor
        p_tensor0 = ggmlqnn_create_compute_tensor(instance, graph_handle, src0, QNN_TENSOR_TYPE_APP_WRITE);
        if (2 == input_param_count) {
            p_tensor1 = ggmlqnn_create_compute_tensor(instance, graph_handle, src1, QNN_TENSOR_TYPE_APP_WRITE);
        }
        p_tensor2 = ggmlqnn_create_compute_tensor(instance, graph_handle, dst, QNN_TENSOR_TYPE_APP_READ);

        //compose QNN graph
        qnn_tensors_t input_tensors;
        input_tensors.reserve(input_param_count);
        input_tensors.push_back(*p_tensor0);
        if (2 == input_param_count) {
            input_tensors.push_back(*p_tensor1);
        }
        Qnn_Tensor_t output_tensors[] = {
                *p_tensor2
        };
        Qnn_OpConfig_t op_config = ggmlqnn_create_op_config(ggml_op_name,
                                                            QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                            qnn_op_name, nullptr, 0,
                                                            input_tensors.data(),
                                                            input_param_count, output_tensors,
                                                            1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, op_config));
        //finalize QNN graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));

        //cache QNN graph
        qnn_ptensors_t qnn_elementwise_tensors;
        qnn_elementwise_tensors.reserve(input_param_count + 1);

        qnn_elementwise_tensors.push_back(p_tensor0);
        if (2 == input_param_count) {
            qnn_elementwise_tensors.push_back(p_tensor1);
        }
        qnn_elementwise_tensors.push_back(p_tensor2);
        auto graph_item = std::make_tuple(graph_handle, qnn_elementwise_tensors);
        ctx->qnn_singlenode_graph_map[graph_name] = graph_item;
    }

    if (enable_npu_rpc) {
        uint8_t * qnn_buffer_0 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(
                QNN_VER_PTR(*p_tensor0)->memHandle));
        GGMLHEXAGON_LOG_DEBUG("qnn_rpcbuffer_0 = %p\n", qnn_buffer_0);
        if (nullptr != qnn_buffer_0) {
            memcpy(qnn_buffer_0, src0->data, ggml_nbytes(src0));
        }

        if (2 == input_param_count) {
            uint8_t * qnn_buffer_1 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(
                    QNN_VER_PTR(*p_tensor1)->memHandle));
            GGMLHEXAGON_LOG_DEBUG("qnn_rpcbuffer_1 = %p\n", qnn_buffer_1);
            if (nullptr != qnn_buffer_1) {
                memcpy(qnn_buffer_1, src1->data, ggml_nbytes(src1));
            }
        }
    } else {
        QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggmlqnn_get_tensor_data_size(src0)};
        if (2 == input_param_count) {
            QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggmlqnn_get_tensor_data_size(src1)};
        }
        QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggmlqnn_get_tensor_data_size(dst)};
    }

    qnn_tensors_t input_tensors;
    input_tensors.reserve(input_param_count);
    input_tensors.push_back(*p_tensor0);
    if (2 == input_param_count) {
        input_tensors.push_back(*p_tensor1);
    }
    Qnn_Tensor_t output_tensors[] = {
            *p_tensor2
    };
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                        input_tensors.data(), input_param_count,
                                                        output_tensors, 1,
                                                        nullptr, nullptr));
    if (enable_npu_rpc) {
        uint8_t * qnn_buffer_2 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor2)->memHandle));
        if (nullptr != qnn_buffer_2) {
            memcpy(dst->data, qnn_buffer_2, ggml_nbytes(dst));
        }
    }

    op_perf.info();
}

/*
 * this function is AI-assisted code from Grok 3 for purpose of offload 4d matrix mulmat to QNN backend
 * various UT has verified and succeed but failed in CT of test-backend-ops
 *
 * the logic of ggmlqnn_compute_mul_mat_4d is similar to ggmlqnn_compute_mul_mat but much more complicated
 * than ggmlqnn_compute_mul_mat, so it's a standalone function.
 * it will be combined with ggmlqnn_compute_mul_mat in the future
 */
static void ggmlqnn_compute_mul_mat_4d(ggml_backend_hexagon_context * ctx, ggml_tensor * op) {
    Qnn_ErrorHandle_t error     = QNN_SUCCESS;
    qnn_instance * instance     = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface = ctx->raw_interface;

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor * dst        = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    GGML_ASSERT(ggml_n_dims(src0) == 4 && ggml_n_dims(src1) == 4);

    hexagon_perf op_perf("ggmlqnn_compute_mul_mat_4d");
    op_perf.start();

    std::string graph_name;
    ggmlhexagon_get_opkey_from_op(op, graph_name);
    GGMLHEXAGON_LOG_DEBUG("graph name %s\n", graph_name.c_str());

    ggmlhexagon_print_tensors_info(__func__, ctx, src0, src1, dst);

    Qnn_GraphHandle_t graph_handle  = nullptr;
    Qnn_Tensor_t * p_tensor0        = nullptr;
    Qnn_Tensor_t * p_reshape0_out   = nullptr;
    Qnn_Tensor_t * p_tile0_out      = nullptr;
    Qnn_Tensor_t * p_tensor1        = nullptr;
    Qnn_Tensor_t * p_permute1_out   = nullptr;
    Qnn_Tensor_t * p_reshape1_out   = nullptr;
    Qnn_Tensor_t * p_matmul_out     = nullptr;
    Qnn_Tensor_t * p_reshape2_out   = nullptr;

    if (ctx->qnn_singlenode_graph_map.find(graph_name) != ctx->qnn_singlenode_graph_map.end()) {
        qnn_singlenode_res_t & graph_item   = ctx->qnn_singlenode_graph_map[graph_name];
        graph_handle                        = std::get<0>(graph_item);
        qnn_ptensors_t & tensors            = std::get<1>(graph_item);
        p_tensor0                           = tensors[0];
        p_reshape0_out                      = tensors[1];
        p_tile0_out                         = tensors[2];
        p_tensor1                           = tensors[3];
        p_permute1_out                      = tensors[4];
        p_reshape1_out                      = tensors[5];
        p_matmul_out                        = tensors[6];
        p_reshape2_out                      = tensors[7];
    } else {
        CHECK_QNN_API(error, qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(), graph_name.c_str(), NULL, &graph_handle));

        // Define dimensions
        uint32_t K = src0->ne[0];               // Inner dimension
        uint32_t M = src0->ne[1];               // Rows of src0
        uint32_t N = src1->ne[1];               // Columns of src1
        uint32_t B0 = src0->ne[2] * src0->ne[3]; // src0 batch
        uint32_t B1 = src1->ne[2] * src1->ne[3]; // src1 batch (drives output)

        // Validate K only
        GGML_ASSERT(src0->ne[0] == src1->ne[0]); // K must match

        // src0: [K, M, H0, B0] -> QNN: [B0, H0, M, K]
        uint32_t src0_dims[] = {static_cast<uint32_t>(src0->ne[3]), static_cast<uint32_t>(src0->ne[2]),
                                static_cast<uint32_t>(src0->ne[1]), static_cast<uint32_t>(src0->ne[0])
        };
        p_tensor0 = ggmlqnn_create_general_tensor(instance, graph_handle, src0, "input0",
                                                  QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, 4,
                                                  src0_dims, nullptr, 0);

        // Reshape src0 to [B0, M, K]
        uint32_t reshape0_out_dims[] = {B0, M, K};
        p_reshape0_out = ggmlqnn_create_general_tensor(instance, graph_handle, nullptr, "reshape0_out",
                                                       QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                                                       reshape0_out_dims, nullptr, 0);

        Qnn_Tensor_t reshape0_inputs[]  = {*p_tensor0};
        Qnn_Tensor_t reshape0_outputs[] = {*p_reshape0_out};
        Qnn_OpConfig_t reshape0_op      = ggmlqnn_create_op_config("reshape0", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                   QNN_OP_RESHAPE, nullptr, 0,
                                                                   reshape0_inputs, 1, reshape0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape0_op));

        // Tile src0 to match B1: [B0, M, K] -> [B1, M, K]
        uint32_t tile0_out_dims[] = {B1, M, K};
        p_tile0_out = ggmlqnn_create_general_tensor(instance, graph_handle, nullptr, "tile0_out",
                                                    QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                                                    tile0_out_dims, nullptr, 0);

        uint32_t tile_multiples[] = {B1 / B0, 1, 1};
        uint32_t tile_dims[] = {3};
        Qnn_Tensor_t * p_tile_multiples = ggmlqnn_create_general_tensor(instance, graph_handle, nullptr, "tile_multiples",
                                                                        QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                                                        tile_dims, tile_multiples, sizeof(tile_multiples));

        Qnn_Param_t tile_params[]       = {{.paramType = QNN_PARAMTYPE_TENSOR, .name = "multiples", .tensorParam = *p_tile_multiples}};
        Qnn_Tensor_t tile0_inputs[]     = {*p_reshape0_out};
        Qnn_Tensor_t tile0_outputs[]    = {*p_tile0_out};
        Qnn_OpConfig_t tile0_op         = ggmlqnn_create_op_config("tile0", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                   QNN_OP_TILE, tile_params, 1,
                                                                   tile0_inputs, 1, tile0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, tile0_op));

        // src1: [N, K, H1, B1] -> QNN: [B1, H1, N, K]
        uint32_t src1_dims[] = {static_cast<uint32_t>(src1->ne[3]), static_cast<uint32_t>(src1->ne[2]),
                                static_cast<uint32_t>(src1->ne[1]), static_cast<uint32_t>(src1->ne[0])
        };
        p_tensor1 = ggmlqnn_create_general_tensor(instance, graph_handle, src1, "input1",
                                                  QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, 4,
                                                  src1_dims, nullptr, 0);


        // Permute src1 to [B1, H1, K, N]
        uint32_t perm_data[] = {0, 1, 3, 2};
        uint32_t perm_dims[] = {4};
        Qnn_Tensor_t * p_perm = ggmlqnn_create_general_tensor(instance, graph_handle, nullptr, "perm",
                                                              QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                                              perm_dims, perm_data, sizeof(perm_data));

        uint32_t permute1_out_dims[] = {static_cast<uint32_t>(src1->ne[3]), static_cast<uint32_t>(src1->ne[2]),
                                        static_cast<uint32_t>(src1->ne[0]), static_cast<uint32_t>(src1->ne[1])
        };
        p_permute1_out = ggmlqnn_create_general_tensor(instance, graph_handle, nullptr, "permute1_out",
                                                       QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 4,
                                                       permute1_out_dims, nullptr, 0);

        Qnn_Param_t permute1_params[]   = {{.paramType = QNN_PARAMTYPE_TENSOR, .name = "perm", .tensorParam = *p_perm}};
        Qnn_Tensor_t permute1_inputs[]  = {*p_tensor1};
        Qnn_Tensor_t permute1_outputs[] = {*p_permute1_out};
        Qnn_OpConfig_t permute1_op      = ggmlqnn_create_op_config("permute1", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                   QNN_OP_TRANSPOSE, permute1_params, 1,
                                                                   permute1_inputs, 1, permute1_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, permute1_op));

        // Reshape src1 to [B1, K, N]
        uint32_t reshape1_out_dims[] = {B1, K, N};
        p_reshape1_out = ggmlqnn_create_general_tensor(instance, graph_handle, nullptr, "reshape1_out",
                                                       QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                                                       reshape1_out_dims, nullptr, 0);

        Qnn_Tensor_t reshape1_inputs[]  = {*p_permute1_out};
        Qnn_Tensor_t reshape1_outputs[] = {*p_reshape1_out};
        Qnn_OpConfig_t reshape1_op      = ggmlqnn_create_op_config("reshape1", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                   QNN_OP_RESHAPE, nullptr, 0,
                                                                   reshape1_inputs, 1, reshape1_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape1_op));

        // MatMul: [B1, M, K] x [B1, K, N] -> [B1, M, N]
        uint32_t matmul_out_dims[] = {B1, M, N};
        p_matmul_out = ggmlqnn_create_general_tensor(instance, graph_handle, nullptr, "matmul_out",
                                                     QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                                                     matmul_out_dims, nullptr, 0);

        Qnn_Tensor_t matmul_inputs[]    = {*p_tile0_out, *p_reshape1_out};
        Qnn_Tensor_t matmul_outputs[]   = {*p_matmul_out};
        Qnn_OpConfig_t matmul_op        = ggmlqnn_create_op_config("matmul", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                   QNN_OP_MAT_MUL, nullptr, 0,
                                                                   matmul_inputs, 2, matmul_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, matmul_op));

        // Output: [N, M, H1, B1] -> QNN: [B1, H1, M, N]
        uint32_t reshape2_out_dims[] = {static_cast<uint32_t>(dst->ne[3]), static_cast<uint32_t>(dst->ne[2]),
                                        static_cast<uint32_t>(dst->ne[1]), static_cast<uint32_t>(dst->ne[0])
        };
        p_reshape2_out = ggmlqnn_create_general_tensor(instance, graph_handle, dst, "output",
                                                       QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, 4,
                                                       reshape2_out_dims, nullptr, 0);

        Qnn_Tensor_t reshape2_inputs[]  = {*p_matmul_out};
        Qnn_Tensor_t reshape2_outputs[] = {*p_reshape2_out};
        Qnn_OpConfig_t reshape2_op      = ggmlqnn_create_op_config("reshape2", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                   QNN_OP_RESHAPE, nullptr, 0,
                                                                   reshape2_inputs, 1, reshape2_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape2_op));

        // Finalize
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, NULL, NULL));

        // Cache
        qnn_ptensors_t ggml_op_mulmat_tensors = {p_tensor0, p_reshape0_out, p_tile0_out, p_tensor1,
                                                 p_permute1_out, p_reshape1_out, p_matmul_out, p_reshape2_out
        };
        ctx->qnn_singlenode_graph_map[graph_name] = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
    }

    // Execute
    QNN_VER_PTR(*p_tensor0)->clientBuf      = {src0->data, static_cast<uint32_t>(ggml_nbytes(src0))};
    QNN_VER_PTR(*p_tensor1)->clientBuf      = {src1->data, static_cast<uint32_t>(ggml_nbytes(src1))};
    QNN_VER_PTR(*p_reshape2_out)->clientBuf = {dst->data, static_cast<uint32_t>(ggml_nbytes(dst))};

    Qnn_Tensor_t input_tensors[]    = {*p_tensor0, *p_tensor1};
    Qnn_Tensor_t output_tensors[]   = {*p_reshape2_out};
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle, input_tensors, 2, output_tensors, 1, NULL, NULL));

    op_perf.info();
}

/*
 * @brief performs matrix multiplication with FP32 & quantized weights and floating-point inputs
 *        using the QNN backend. this function performs matrix multiplication of the input tensor
 *        `src1` and the weight tensor `src0`, handling transposing, and quantization as needed,
 *        and stores the result in the destination tensor `dst`.
 *
         there are two key-points in properly handling how to offload mulmat to the QNN
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
             in func ggmlqnn_compute_create_general_tensor()

 * @param ctx     the context of backend
 * @param op      the destination tensor where the result of the matrix multiplication will be stored.
 *
 * @note the logic of ggmlqnn_compute_mul_mat is similar to ggmlqnn_compute_op_two_tensors but much more complicated
 *       than ggmlqnn_compute_op_two_tensors. so it's a standalone function. accordingly, this is another
 *       typical skeleton for offload other ggml ops to QNN backend. MUL_MAT take most of the compute
 *       time (about 95%).so to speed up llama inference, should focus on this func. there are three kinds
 *       of MUL_MAT to compute:
 *       mul_mat_f32:     both src0 and src1 are F32, this will be naturally handled in QNN backend
 *       mul_mat_f16_f32: src0 is F16 and src1 is F32, f16 in src0 -> f32 in src0', then src0' * src1
 *       mul_mat_q_f32:   src0 is quantized (Q4_0, Q4_1, Q6_K...)
 *                        and src1 is F32, src0 -> f32 in src0', then src0' * src1
*/
static void ggmlqnn_compute_mul_mat(ggml_backend_hexagon_context * ctx, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
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

    const enum ggml_type src0_type              = src0->type;
    const uint32_t src0_rank                    = ggml_n_dims(src0);
    const uint32_t src1_rank                    = ggml_n_dims(src1);

    ggmlhexagon_print_tensors_info(__func__, ctx, src0, src1, dst);

    std::string graph_name;
    ggmlhexagon_get_opkey_from_op(op, graph_name);

    hexagon_perf op_perf(graph_name);
    op_perf.start();

    GGML_ASSERT(src0_rank == src1_rank);
    GGML_ASSERT(src0_rank >= 2); //QNN SDK's limitation, make QNN SDK happy
    if (4 == src0_rank) {
        return ggmlqnn_compute_mul_mat_4d(ctx, op);
    }

    void * wdata                                = ggmlhexagon_type_trait(ctx, op);
    const size_t desired_size                   = ctx->desired_size;

    if (ctx->qnn_singlenode_graph_map.find(graph_name) != ctx->qnn_singlenode_graph_map.end()) {
        //retrieve computational resource from cached QNN graph
        qnn_singlenode_res_t & graph_item = ctx->qnn_singlenode_graph_map[graph_name];
        graph_handle = std::get<0>(graph_item);
        qnn_ptensors_t &tensors = std::get<1>(graph_item);
        p_tensor0 = tensors[0];
        p_tensor1 = tensors[1];
        p_tensor2 = tensors[2];
        p_param_tensor = tensors[3];
        p_tensor2_transpose = tensors[4];
    } else {
        //create QNN graph
        GGMLHEXAGON_LOG_INFO("graph name %s", graph_name.c_str());
        error = instance->init_qnn_graph(graph_name, static_cast<HEXAGONBackend>(ctx->device),
                                         g_hexagon_appcfg.vtcm_size_in_mb,
                                         g_hexagon_appcfg.hvx_threads);
        if (QNN_SUCCESS != error) {
            GGMLHEXAGON_LOG_WARN("can't create qnn graph handle with graph name %s, error = %d\n",
                                 graph_name.c_str(), error);
            return;
        }
        graph_handle = instance->get_qnn_graph_handle();

        //create computational tensor
        p_tensor0 = ggmlqnn_create_general_tensor(instance, graph_handle, src0, nullptr,
                                                  QNN_TENSOR_TYPE_APP_WRITE,
                                                  QNN_DATATYPE_FLOAT_32, src0_rank,
                                                  nullptr, nullptr, 0);
        p_tensor1 = ggmlqnn_create_general_tensor(instance, graph_handle, src1, nullptr,
                                                  QNN_TENSOR_TYPE_APP_WRITE,
                                                  QNN_DATATYPE_FLOAT_32, src0_rank,
                                                  nullptr, nullptr, 0);
        p_tensor2 = ggmlqnn_create_general_tensor(instance, graph_handle, dst, nullptr,
                                                  QNN_TENSOR_TYPE_APP_READ,
                                                  QNN_DATATYPE_FLOAT_32, src0_rank,
                                                  nullptr, nullptr, 0);

        //create param tensor for offload 2d/3d/4d matrix multiplication
        const uint32_t param_tensor_data[GGML_MAX_DIMS][GGML_MAX_DIMS] = {
                {0},
                {1, 0},
                {0, 2, 1},
                {0, 1, 3, 2},
        };
        uint32_t param_tensor_dims[1] = {src0_rank};
        p_param_tensor = ggmlqnn_create_general_tensor(instance, graph_handle, nullptr, "param",
                                                       QNN_TENSOR_TYPE_STATIC,
                                                       QNN_DATATYPE_UINT_32, 1,
                                                       param_tensor_dims,
                                                       (void *) (param_tensor_data[src0_rank - 1]),
                                                       src0_rank * sizeof(uint32_t));

        //create transpose tensor
        p_tensor2_transpose = ggmlqnn_create_general_tensor(instance, graph_handle, dst,
                                                            "transpose",
                                                            QNN_TENSOR_TYPE_NATIVE,
                                                            QNN_DATATYPE_FLOAT_32, src0_rank,
                                                            nullptr, nullptr, 0, true);

        //compose QNN graph: add mulmat node
        Qnn_Param_t out_0_params[] = {
                {.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, .scalarParam = {
                        .dataType = QNN_DATATYPE_BOOL_8, .bool8Value = 1}}};
        Qnn_Tensor_t out_0_inputs[] = {*p_tensor0, *p_tensor1};
        Qnn_Tensor_t out_0_outputs[] = {*p_tensor2_transpose};
        Qnn_OpConfig_t out_0 = ggmlqnn_create_op_config("mulmat_opconfig",
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_MAT_MUL, out_0_params, 1,
                                                        out_0_inputs, 2, out_0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, out_0));

        //compose QNN graph: add transpose node
        Qnn_Param_t out_trans1_0_params[] = {
                {.paramType = QNN_PARAMTYPE_TENSOR, .name = "perm", .tensorParam = *p_param_tensor}};
        Qnn_Tensor_t out_trans1_0_inputs[] = {*p_tensor2_transpose};
        Qnn_Tensor_t out_trans1_0_outputs[] = {*p_tensor2};
        Qnn_OpConfig_t out_trans1_0 = ggmlqnn_create_op_config("mulmat_transpose_opconfig",
                                                               QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                               QNN_OP_TRANSPOSE,
                                                               out_trans1_0_params, 1,
                                                               out_trans1_0_inputs, 1,
                                                               out_trans1_0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, out_trans1_0));

        //finalize QNN graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));

        //cache QNN graph
        qnn_ptensors_t ggml_op_mulmat_tensors;
        ggml_op_mulmat_tensors.reserve(5);
        ggml_op_mulmat_tensors.push_back(p_tensor0);
        ggml_op_mulmat_tensors.push_back(p_tensor1);
        ggml_op_mulmat_tensors.push_back(p_tensor2);
        ggml_op_mulmat_tensors.push_back(p_param_tensor);
        ggml_op_mulmat_tensors.push_back(p_tensor2_transpose);
        auto graph_item = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
        ctx->qnn_singlenode_graph_map[graph_name] = graph_item;
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

static void ggmlqnn_compute_repeat(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_div(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_leaky_relu(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_concat(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_arange(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_sqr(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_clamp(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_scale(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_argsort(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_norm(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_group_norm(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_acc(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_sum_rows(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_upsample_nearest2d(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_pad(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_pool2d(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_dup(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_rms_norm(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_diag_mask(ggml_backend_hexagon_context * ctx, ggml_tensor * dst, float value) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(value);
}

static void ggmlqnn_compute_im2col(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_timestep_embedding(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_cpy(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    ggmlqnn_compute_dup(ctx, dst);
}

static void ggmlqnn_compute_softmax(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_get_rows(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

static void ggmlqnn_compute_rope(ggml_backend_hexagon_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

// =================================================================================================
//  section-7: cDSP helper function
// =================================================================================================
static const char * ggmlhexagon_get_dsp_name(int domain_id) {
    switch (domain_id) {
        case HEXAGON_ADSP:
            return "Hexagon-aDSP";
        case HEXAGON_MDSP:
            return "Hexagon-mDSP";
        case HEXAGON_SDSP:
            return "Hexagon-sDSP";
        case HEXAGON_CDSP:
            return "Hexagon-cDSP";
        case HEXAGON_CDSP1:
            return "Hexagon-cDSP1";
        default:
            return "Hexagon-unknown";
    }
}

static int ggmlhexagon_pd_status_notifier_callback(void * context, int domain, int session, remote_rpc_status_flags_t status){
    int error = AEE_SUCCESS;
    switch (status){
        case  FASTRPC_USER_PD_UP:
            GGMLHEXAGON_LOG_DEBUG("PD is up\n");
            break;
        case  FASTRPC_USER_PD_EXIT:
            GGMLHEXAGON_LOG_DEBUG("PD closed\n");
            break;
        case  FASTRPC_USER_PD_FORCE_KILL:
            GGMLHEXAGON_LOG_DEBUG("PD force kill\n");
            break;
        case  FASTRPC_USER_PD_EXCEPTION:
            GGMLHEXAGON_LOG_DEBUG("PD exception\n");
            break;
        case  FASTRPC_DSP_SSR:
            GGMLHEXAGON_LOG_DEBUG("DSP SSR\n");
            break;
        default :
            error =  AEE_EBADITEM;
            break;
    }
    return error;
}

static domain * ggmlhexagon_get_domain(int domain_id) {
    int size = sizeof(hexagon_supported_domains) / sizeof(domain);

    for (int i = 0; i < size; i++) {
        if (hexagon_supported_domains[i].id == domain_id)
            return &hexagon_supported_domains[i];
    }

    return nullptr;
}

static bool ggmlhexagon_is_cdsp(int domain_id) {
    return (domain_id == HEXAGON_CDSP) || (domain_id == HEXAGON_CDSP1);
}

static bool ggmlhexagon_is_valid_domain_id(int domain_id, int compute_only) {
    int size = sizeof(hexagon_supported_domains) / sizeof(domain);

    if (0 != compute_only) {
        return ggmlhexagon_is_cdsp(domain_id);
    }

    for (int i = 0; i < size; i++) {
        if (hexagon_supported_domains[i].id == domain_id)
            return true;
    }

    return false;
}

static int ggmlhexagon_get_domains_info(const char * domain_type, int * num_domains, fastrpc_domain ** domains_info) {
    int hexagon_err = AEE_SUCCESS;
    int ss_info     = 0;
    void * buffer   = nullptr;
    ss_info = strcmp(domain_type, "NSP")? HPASS: NSP;
    system_req_payload req;
    memset(&req, 0, sizeof(system_req_payload));
    req.id = FASTRPC_GET_DOMAINS;
    req.sys.domains = nullptr;
    fastrpc_domain * domain = nullptr;

    if (ss_info != 0) {
        req.sys.flags = DOMAINS_LIST_FLAGS_SET_TYPE(req.sys.flags, ss_info);
    } else {
        req.sys.flags =0;
    }

#ifdef _WIN32
    hexagon_err = AEE_EUNSUPPORTED;
    goto bail;
#endif

    hexagon_err = remote_system_request(&req);
    if (hexagon_err != AEE_SUCCESS) {
        GGMLHEXAGON_LOG_DEBUG("failure in remote_system_request call: %d", hexagon_err);
        goto bail;
    }
    //allocate memory for domain-info array
    req.sys.max_domains = req.sys.num_domains;
    buffer = calloc(req.sys.num_domains, sizeof(fastrpc_domain));
    if (nullptr == buffer) {
        hexagon_err = AEE_ENOMEMORY;
        GGMLHEXAGON_LOG_DEBUG("unable to allocate memory for req.sys.domains");
        goto bail;
    }
    req.sys.domains = static_cast<fastrpc_domain *>(buffer);
    hexagon_err = remote_system_request(&req);
    if (hexagon_err != AEE_SUCCESS) {
        GGMLHEXAGON_LOG_DEBUG("failure in remote_system_request call: %d.\n", hexagon_err);
        goto bail;
    }

    for (int i = 0; i < req.sys.num_domains; i++) {
        //verify that only requested type domains were returned
        domain = &req.sys.domains[i];
        if (domain->type != ss_info) {
            hexagon_err = -1;
            GGMLHEXAGON_LOG_DEBUG("incorrect data received from remote_system_request.\n");
            goto bail;
        }
    }
    *domains_info = req.sys.domains;
    *num_domains  = req.sys.num_domains;

bail:
    if (hexagon_err && !req.sys.domains) {
        free(req.sys.domains);
    }
    return hexagon_err;
}

static int ggmlhexagon_get_dsp_support(int * domain) {
    int hexagon_error = AEE_SUCCESS;
    *domain = HEXAGON_CDSP;

    if (remote_handle_control) {
        struct remote_dsp_capability dsp_capability_domain = {HEXAGON_CDSP, DOMAIN_SUPPORT, 0};
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
            goto bail;
        }

        if (0 == dsp_capability_domain.capability) {
            dsp_capability_domain.domain       = HEXAGON_ADSP;
            dsp_capability_domain.attribute_ID = DOMAIN_SUPPORT;
            dsp_capability_domain.capability   = 0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
            if(dsp_capability_domain.capability) {
                *domain = HEXAGON_ADSP;
            }
        }

        if (hexagon_error != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_DEBUG("get_dsp_support failed with error 0x%x", hexagon_error);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_get_vtcm_info(int domain, uint32_t attr, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;

    if (attr == VTCM_PAGE || attr == VTCM_COUNT) {
    } else {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("unsupported attr, only VTCM_PAGE and VTCM_COUNT supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == HEXAGON_ADSP || domain == HEXAGON_CDSP) {
            /*
            * query the DSP for VTCM information
            * since the ADSP does not have a dedicated VTCM, we expect the output to be 0
            */
            struct remote_dsp_capability dsp_capability_vtcm_dsp;
            dsp_capability_vtcm_dsp.domain       = (uint32_t)domain;
            dsp_capability_vtcm_dsp.attribute_ID = attr;
            dsp_capability_vtcm_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_vtcm_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                GGMLHEXAGON_LOG_DEBUG("running the use case without checking the capability");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_vtcm_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("get_vtcm_info failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("unsupported domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static bool ggmlhexagon_is_unsignedpd_supported(int domain_id) {
    int hexagon_error = AEE_SUCCESS;
    if (remote_handle_control) {
        struct remote_dsp_capability dsp_capability_domain = {static_cast<uint32_t>(domain_id), UNSIGNED_PD_SUPPORT, 0};
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_WARN("FastRPC Capability API is not supported on this device. Falling back to signed pd");
            return false;
        }

        if (hexagon_error) {
            GGMLHEXAGON_LOG_WARN("error 0x%x: FastRPC Capability API failed. falling back to signed pd", hexagon_error);
            return false;
        }

        if (dsp_capability_domain.capability == 1) {
            return true;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device.falling back to signed pd");
        return false;
    }

    return false;
}

static bool ggmlhexagon_get_unsignedpd_support(void) {
    return ggmlhexagon_is_unsignedpd_supported(HEXAGON_CDSP);
}

static bool ggmlhexagon_is_async_fastrpc_supported(int domain) {
    int hexagon_error = AEE_SUCCESS;
    if (remote_handle_control) {
        if (domain == HEXAGON_CDSP) {
            /*
            * Query the DSP for ASYNC_FASTRPC_SUPPORT information
            * Async fastrpc is supported only on CDSP
            */
            struct remote_dsp_capability dsp_capability_async_support;
            dsp_capability_async_support.domain       = (uint32_t)domain;
            dsp_capability_async_support.attribute_ID = ASYNC_FASTRPC_SUPPORT;
            dsp_capability_async_support.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_async_support, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_WARN("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (dsp_capability_async_support.capability == 1) {
                return true;
            }

            if (hexagon_error != AEE_SUCCESS){
                GGMLHEXAGON_LOG_WARN("failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_WARN("async FastRPC is not supported on domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return false;
}

static void ggmlhexagon_set_rpc_latency(remote_handle64 handle, int qos, int latency) {
    int hexagon_error = AEE_SUCCESS;

    if (remote_handle_control) {
        struct remote_rpc_control_latency data;
/*
        qos          |  latency
        -----------------------
        RPC_PM_QOS   |  100
        RPC_POLL_QOS |  1000
*/
        data.enable   = qos;
        data.latency  = latency;
        hexagon_error = remote_handle64_control(handle, DSPRPC_CONTROL_LATENCY, (void*)&data, sizeof(data));
        if (hexagon_error != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed with error 0x%x", hexagon_error);
            goto bail;
        } else {
            GGMLHEXAGON_LOG_INFO("set rpc qos %d, latency %d\n", qos, latency);
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return;
}

static bool ggmlhexagon_is_status_notification_supported(int domain) {
    int hexagon_error = AEE_SUCCESS;

    if (remote_handle_control) {
        /*
        * Query the DSP for STATUS_NOTIFICATION_SUPPORT information
        * DSP User PD status notification Support
        */
        struct remote_dsp_capability dsp_capability_status_notification_support;
        dsp_capability_status_notification_support.domain       = (uint32_t)domain;
        dsp_capability_status_notification_support.attribute_ID = STATUS_NOTIFICATION_SUPPORT;
        dsp_capability_status_notification_support.capability   = (uint32_t)0;
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_status_notification_support, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_WARN("FastRPC Capability API is not supported on this device");
            hexagon_error = AEE_SUCCESS;
            goto bail;
        } else if (1 == dsp_capability_status_notification_support.capability) {
            return true;
        }

        if (hexagon_error != AEE_SUCCESS){
            GGMLHEXAGON_LOG_WARN("failed with error 0x%x", hexagon_error);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return false;
}

static int ggmlhexagon_get_hmx_support_info(int domain, uint32_t attr, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;

    if (attr != HMX_SUPPORT_SPATIAL && attr != HMX_SUPPORT_DEPTH) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_WARN("unsupported attr, only HMX_SUPPORT_SPATIAL and HMX_SUPPORT_DEPTH supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == HEXAGON_CDSP) {
            /*
            * Query the DSP for HMX SUPPORT information
            * HMX is supported on CDSP only
            */
            struct remote_dsp_capability dsp_capability_hmx_dsp;
            dsp_capability_hmx_dsp.domain       = (uint32_t)domain;
            dsp_capability_hmx_dsp.attribute_ID = attr;
            dsp_capability_hmx_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_hmx_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            }
            else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_hmx_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("get_hmx_support_info failed with Error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("HMX support is not there for domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_get_hvx_arch_ver(int domain, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;
    if(remote_handle_control) {
        /*
        * Query the Hexagon processor architecture version information
        */
        struct remote_dsp_capability dsp_capability_arch_ver;
        dsp_capability_arch_ver.domain       = (uint32_t)domain;
        dsp_capability_arch_ver.attribute_ID = ARCH_VER;
        dsp_capability_arch_ver.capability   = (uint32_t)0;
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_arch_ver, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
            hexagon_error = AEE_SUCCESS;
            goto bail;
        } else if (hexagon_error == AEE_SUCCESS) {
            *capability = dsp_capability_arch_ver.capability & 0xFF;
        } else {
            GGMLHEXAGON_LOG_DEBUG("get_hex_arch_ver failed with error 0x%x", hexagon_error);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_get_hvx_support_info(int domain, uint32_t attr, uint32_t * capability)
{
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;
    if (attr == HVX_SUPPORT_64B) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("latest targets have 128 byte HVX register, use HVX_SUPPORT_128B instead of HVX_SUPPORT_64B");
        goto bail;
    }

    if (attr != HVX_SUPPORT_128B) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("unsupported attr. only HVX_SUPPORT_128B supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == HEXAGON_CDSP) {
            /*
            * Query the DSP for HVX SUPPORT information
            * HVX is supported on CDSP only
            */
            struct remote_dsp_capability dsp_capability_hvx_dsp;
            dsp_capability_hvx_dsp.domain       = (uint32_t)domain;
            dsp_capability_hvx_dsp.attribute_ID = attr;
            dsp_capability_hvx_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_hvx_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF)==(AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_hvx_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("HVX support is not available on domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_request_status_notifications(int domain_id, void * context, notify_callback_fn call_back_fn) {
    int hexagon_error = AEE_SUCCESS;
    struct remote_rpc_notif_register notif;
    bool status_notification_support;

    notif.context     = context;
    notif.domain      = domain_id;
    notif.notifier_fn = call_back_fn;

    status_notification_support = ggmlhexagon_is_status_notification_supported(domain_id);
    if (status_notification_support) {
        hexagon_error = remote_session_control(FASTRPC_REGISTER_STATUS_NOTIFICATIONS, (void*)&notif, sizeof(notif));
        if (hexagon_error != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_DEBUG("error 0x%x: remote_session_control failed to enable status notifications", hexagon_error);
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
    }

    return hexagon_error;
}

static void ggmlhexagon_init_rpcmempool(ggml_backend_hexagon_context * ctx) {
    size_t candidate_size   = 0;
    uint8_t * rpc_buffer    = nullptr;
    size_t probe_slots[]    = {1024, 1536, 2000, 2048};
    size_t probe_counts     = sizeof(probe_slots) / sizeof(size_t);
    for (size_t idx = 0; idx < probe_counts; idx++) {
        rpc_buffer = static_cast<uint8_t *>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (probe_slots[idx] * SIZE_IN_MB)));
        if (nullptr == rpc_buffer) {
            GGMLHEXAGON_LOG_DEBUG("alloc rpcmem %d (MiB) failure during probe rpc memory info, reason: %s\n", probe_slots[idx], strerror(errno));
            break;
        } else {
            candidate_size = probe_slots[idx];
            rpcmem_free(rpc_buffer);
            rpc_buffer = nullptr;
        }
    }
    ctx->rpc_mempool_capacity = candidate_size * SIZE_IN_MB;
    GGMLHEXAGON_LOG_DEBUG("rpc memory capacity %ld(%d MiB) for device %d",
                          ctx->rpc_mempool_capacity, ctx->rpc_mempool_capacity / SIZE_IN_MB, ctx->device);
    GGMLHEXAGON_LOG_INFO("capacity of rpc memory %d MiB", ctx->rpc_mempool_capacity / SIZE_IN_MB);

    if ((g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
        //FIXME: reasonable rpc memory pool size through a better approach rather than hardcoded size
        ctx->rpc_mempool_len = 1024 * SIZE_IN_MB;
        if (ctx->rpc_mempool_len > ctx->rpc_mempool_capacity) {
            GGMLHEXAGON_LOG_WARN("rpc mempool is too big");
            return;
        }
        //FIXME: it seems there is unknown issue with DMA memory pool
        ctx->rpc_mempool = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                                        ctx->rpc_mempool_len);
        if (nullptr == ctx->rpc_mempool) {
            GGMLHEXAGON_LOG_WARN("alloc rpc memorypool %d failed", ctx->rpc_mempool_len);
            return;
        } else {
            GGMLHEXAGON_LOG_DEBUG("alloc rpc memorypool %p successfully %ld(%d MiB)",
                                  ctx->rpc_mempool, ctx->rpc_mempool_len,
                                  ctx->rpc_mempool_len / SIZE_IN_MB);
        }
        ctx->rpc_mempool_handle = rpcmem_to_fd(ctx->rpc_mempool);
        GGMLHEXAGON_LOG_DEBUG("rpc mempool handle %d", ctx->rpc_mempool_handle);
        remote_register_buf(ctx->rpc_mempool, ctx->rpc_mempool_len, ctx->rpc_mempool_handle);
    }

    if ((g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP) && (1 == g_hexagon_appcfg.enable_rpc_dma_mempool)) {
        //TODO
    }

    return;
}

static void ggmlhexagon_deinit_rpcmempool(ggml_backend_hexagon_context * ctx) {
    if ((g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
        if (ctx->rpc_mempool) {
            //deregister rpc memory pool
            remote_register_buf(ctx->rpc_mempool, ctx->rpc_mempool_len, -1);
            GGMLHEXAGON_LOG_DEBUG("free rpc mempool %p", ctx->rpc_mempool);
            rpcmem_free(ctx->rpc_mempool);
            ctx->rpc_mempool = nullptr;
            ctx->rpc_mempool_len = 0;
            ctx->rpc_mempool_capacity = 0;
        }
    }
}

static void ggmlhexagon_probe_dspinfo(ggml_backend_hexagon_context * ctx) {
    uint32_t dsp_version = 0;
    ggmlhexagon_get_hvx_arch_ver(ctx->domain_id, &dsp_version);

    if (dsp_version == 0x68 || dsp_version == 0x69 || dsp_version == 0x73 || dsp_version == 0x75 || dsp_version == 0x79) {
        GGMLHEXAGON_LOG_INFO("dsp arch version 0x%x", dsp_version);
        //0x68 -> 68, 0x69 -> 69, 0x73 -> 73, 0x75 -> 75, 0x79 -> 79
        size_t htp_arch = ggmlhexagon_htparch_hex_to_decimal(dsp_version);
        GGMLHEXAGON_LOG_DEBUG("dsp arch version %d", htp_arch);
        struct qcom_socinfo * socinfo = ggmlhexagon_get_socinfo_from_socmodel(htp_arch);
        if (nullptr != socinfo) {
            //got fully description of SoC when hwaccel approach is HWACCEL_CDSP
            GGMLHEXAGON_LOG_INFO("device info: %s, %s", socinfo->soc_desc, ggmlhexagon_get_htparch_desc(htp_arch));
        }
    } else {
        GGMLHEXAGON_LOG_WARN("error: dsp arch version 0x%x is not supported", dsp_version);
    }

    uint32_t vtcm_count = 0;
    uint32_t vtcm_page  = 0;
    ggmlhexagon_get_vtcm_info(ctx->domain_id, VTCM_COUNT, &vtcm_count);
    ggmlhexagon_get_vtcm_info(ctx->domain_id, VTCM_PAGE, &vtcm_page);
    GGMLHEXAGON_LOG_INFO("vtcm_count %d", vtcm_count);
    GGMLHEXAGON_LOG_INFO("vtcm_page %d", vtcm_page);

    uint32_t hmx_depth = 0;
    uint32_t hmx_spatial = 0;
    ggmlhexagon_get_hmx_support_info(ctx->domain_id, HMX_SUPPORT_DEPTH, &hmx_depth);
    ggmlhexagon_get_hmx_support_info(ctx->domain_id, HMX_SUPPORT_SPATIAL, &hmx_spatial);
    GGMLHEXAGON_LOG_INFO("hmx_depth %d", hmx_depth);
    GGMLHEXAGON_LOG_INFO("hmx_spatial %d", hmx_spatial);

    uint32_t hvx_support_128b = 0;
    ggmlhexagon_get_hvx_support_info(ctx->domain_id, HVX_SUPPORT_128B, &hvx_support_128b);
    GGMLHEXAGON_LOG_INFO("hvx_support_128b %d", hvx_support_128b);

    GGMLHEXAGON_LOG_INFO("unsigned pd supported %d", ggmlhexagon_get_unsignedpd_support());
    GGMLHEXAGON_LOG_INFO("async fastrpc supported %d", ggmlhexagon_is_async_fastrpc_supported(ctx->domain_id));
}

static void ggmlhexagon_deinit_cdsp(ggml_backend_hexagon_context * ctx) {
    int hexagon_error  = AEE_SUCCESS;
    GGMLHEXAGON_LOG_INFO("enter %s", __func__);
    if (0 != ctx->ggmlop_handle) {
        hexagon_error = ggmlop_dsp_close(ctx->ggmlop_handle);
        if (AEE_SUCCESS != hexagon_error) {
            GGMLHEXAGON_LOG_WARN("error 0x%x: failed to close ggmlop dsp handle", hexagon_error);
        } else {
            ctx->ggmlop_handle = 0;
        }
    }

    ggmlhexagon_deinit_rpcmempool(ctx);

    ctx->domain_id             = -1;
    GGMLHEXAGON_LOG_INFO("leave %s", __func__);
}

static int ggmlhexagon_init_dsp(ggml_backend_hexagon_context * ctx) {
    int hexagon_error               = AEE_SUCCESS;

    int domain_id                   = HEXAGON_CDSP;
    const char * domain_type        = "NSP";

    int unsignedpd_flag             = 1;
    bool is_unsignedpd_enabled      = false;
    int use_logical_id              = 0;
    int core_id                     = -1;
    fastrpc_domain * domains_info   = NULL;
    int num_domains                 = -1;

    domain * my_domain              = NULL;
    char * uri                      = NULL;

    char * ggmlop_domain_uri        = NULL;
    int    ggmlop_domain_uri_len    = 0;

    if (nullptr == ctx)
        return 1;
    GGMLHEXAGON_LOG_INFO("init Hexagon cDSP with backend %d(%s)", ctx->device, ggml_backend_hexagon_get_devname(ctx->device));
    if (0 != ctx->ggmlop_handle) {
        GGMLHEXAGON_LOG_DEBUG("already init Hexagon cDSP with backend %d(%s)", ctx->device, ggml_backend_hexagon_get_devname(ctx->device));
        return 0;
    }
    ctx->ggmlop_handle = 0;

    if (-1 == domain_id) {
        if (nullptr != domain_type) {
            if ((strcmp(domain_type, "NSP") != 0 && strcmp(domain_type, "HPASS") != 0)) {
                GGMLHEXAGON_LOG_WARN("invalid domain_type %s. possible values are NSP or HPASS", domain_type);
                goto bail;
            } else {
                hexagon_error = ggmlhexagon_get_domains_info(domain_type, &num_domains, &domains_info);
                if (hexagon_error == AEE_EUNSUPPORTED) {
                    GGMLHEXAGON_LOG_DEBUG("API is not supported on this target so cannot get domains info from the device. falling back to legacy approach of using default domain id");
                    hexagon_error = ggmlhexagon_get_dsp_support(&domain_id);
                    if (hexagon_error != AEE_SUCCESS) {
                        GGMLHEXAGON_LOG_DEBUG("error: 0x%x, defaulting to cDSP domain", hexagon_error);
                    }
                } else if (hexagon_error != AEE_SUCCESS) {
                    GGMLHEXAGON_LOG_DEBUG("error in getting domains information");
                    goto bail;
                } else {
                    if (core_id != -1) {
                        if (core_id < 0 || core_id >= num_domains) {
                            GGMLHEXAGON_LOG_DEBUG("invalid core_id = %d for %s. core_id should be between 0 to %d", core_id, domain_type, num_domains - 1);
                            hexagon_error = AEE_EBADPARM;
                            goto bail;
                        }
                    } else {
                        core_id = 0;
                    }
                    use_logical_id = 1;
                    domain_id = domains_info[core_id].id;
                }
            }
        } else {
            GGMLHEXAGON_LOG_DEBUG("DSP domain is not provided, retrieving DSP information using Remote APIs");
            hexagon_error = ggmlhexagon_get_dsp_support(&domain_id);
            if (hexagon_error != AEE_SUCCESS) {
                GGMLHEXAGON_LOG_DEBUG("error: 0x%x, defaulting to cDSP domain", hexagon_error);
            }
        }
    }

    if (0 == use_logical_id) {
        if (!ggmlhexagon_is_valid_domain_id(domain_id, 0)) {
            hexagon_error = AEE_EBADPARM;
            GGMLHEXAGON_LOG_DEBUG("error 0x%x: invalid domain %d", hexagon_error, domain_id);
            goto bail;
        }

        my_domain = ggmlhexagon_get_domain(domain_id);
        if (nullptr == my_domain) {
            GGMLHEXAGON_LOG_DEBUG("unable to get domain struct %d",  domain_id);
            goto bail;
        }
        uri = my_domain->uri;
    }
    GGMLHEXAGON_LOG_DEBUG("temporary domain uri=%s\n", uri);

    if (1 == unsignedpd_flag) {
        is_unsignedpd_enabled = ggmlhexagon_is_unsignedpd_supported(domain_id);
        if (!is_unsignedpd_enabled) {
            GGMLHEXAGON_LOG_DEBUG("overriding user request for unsigned PD, only signed offload is allowed on domain %d", domain_id);
            unsignedpd_flag = 0;
        }
    }

    ctx->domain_id = domain_id;
    GGMLHEXAGON_LOG_INFO("using Hexagon domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
    GGMLHEXAGON_LOG_INFO("unsignedpd_enabled %d", is_unsignedpd_enabled);
    if (is_unsignedpd_enabled) {
        if (remote_session_control) {
            struct remote_rpc_control_unsigned_module data;
            data.enable = 1;
            data.domain = domain_id;
            hexagon_error = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));
            GGMLHEXAGON_LOG_DEBUG("remote_session_control returned %d for configuring unsigned PD success", hexagon_error);
            if (AEE_SUCCESS != hexagon_error) {
                GGMLHEXAGON_LOG_DEBUG("error 0x%x: remote_session_control failed", hexagon_error);
            }
        } else {
            GGMLHEXAGON_LOG_DEBUG("unsigned PD not supported on this device");
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("error 0x%x: remote_session_control interface is not supported on this device", hexagon_error);
        }
    }

    hexagon_error = ggmlhexagon_request_status_notifications(domain_id, (void *)STATUS_CONTEXT, ggmlhexagon_pd_status_notifier_callback);
    if (AEE_SUCCESS != hexagon_error) {
        if (AEE_EUNSUPPORTEDAPI != hexagon_error) {
            GGMLHEXAGON_LOG_WARN("error 0x%x: hexagon_request_status_notifications failed", hexagon_error);
        }
        GGMLHEXAGON_LOG_WARN("error 0x%x: failed to compute on domain %d", hexagon_error, domain_id);
        goto bail;
    }

    ggmlop_domain_uri_len   = strlen(ggmlop_URI) + MAX_DOMAIN_NAMELEN;
    ggmlop_domain_uri       = (char *)malloc(ggmlop_domain_uri_len);
    snprintf(ggmlop_domain_uri, ggmlop_domain_uri_len, "%s%s", ggmlop_URI, uri);
    GGMLHEXAGON_LOG_DEBUG("ggmlop domain uri:%s", ggmlop_domain_uri);
    hexagon_error = ggmlop_dsp_open(ggmlop_domain_uri, &ctx->ggmlop_handle);
    if (AEE_SUCCESS == hexagon_error) {
        GGMLHEXAGON_LOG_INFO("succeed to open domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
        //FIXME: only support offload fp32 GGML_OP_MUL_MAT to cDSP
        GGMLHEXAGON_LOG_INFO("only support offload fp32 GGML_OP_ADD and fp32 GGML_OP_MUL_MAT to cDSP currently");
        ggmlhexagon_probe_dspinfo(ctx);
        ggmlop_dsp_setclocks(ctx->ggmlop_handle, HAP_DCVS_VCORNER_TURBO_PLUS, 40, 1);
        ggmlhexagon_set_rpc_latency(ctx->ggmlop_handle, RPC_POLL_QOS, 100);
        ggmlhexagon_init_rpcmempool(ctx);
    } else {
        GGMLHEXAGON_LOG_INFO("error 0x%x: failed to open domain %d(%s)", hexagon_error, domain_id,
                             ggmlhexagon_get_dsp_name(domain_id));
        goto bail;
    }

    //ensure test-backend-ops get the correct backend name when hwaccel approach is 2(HWACCEL_CDSP)
    memcpy(g_hexagon_mgr[ctx->device].name, "Hexagon-cDSP", strlen("Hexagon-cDSP"));

    return 0;

bail:
    if (ggmlop_domain_uri) {
        free(ggmlop_domain_uri);
    }

    ggmlhexagon_deinit_cdsp(ctx);

    return -1;
}

static void ggmlhexagon_compute(ggml_backend_hexagon_context * ctx, struct ggml_tensor * op) {
    //skip sanity check because already checked in other place
    struct dsptensor dsptensor_0;
    struct dsptensor dsptensor_1;
    struct dsptensor dsptensor_2;
    std::string op_name;
    ggmlhexagon_get_opkey_from_op(op, op_name);

    hexagon_perf op_perf(op_name);
    op_perf.start();

    int hexagon_error               = AEE_SUCCESS;
    ggmlhexagon_op_func_t op_func   = nullptr;
    size_t input_tensor_count       = 2;

    ggml_tensor * src0  = op->src[0];
    ggml_tensor * src1  = op->src[1];
    ggml_tensor * dst   = op;

    input_tensor_count  =  ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op)].input_param_count;
    op_func             =  ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op)].dsp_op_func;
    if (nullptr == op_func) {
        GGMLHEXAGON_LOG_DEBUG("op GGML_OP_%s and dsp func %s not supported on cCSP", ggml_op_name(op->op), ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op)].hexagon_op_name);
        return;
    }

    //FIXME:try to fully understand the tech detail in qidl:
    // qidl is a binary tool to generate some very complicated and hard-to customized bridge-layer codes
    // between ARM-AP and cDSP. the mechanism in qidl/FastRPC is exactly similar to mechanism in TEE.
    // try to find a better/efficient approach to exchange necessary data between ARM-AP side and cDSP side.
    // manually modifying the important data structure ggml_tensor in ggml.h is not make-sense and not acceptable.
    dsptensor_0.data        = src0->data;
    dsptensor_0.data_len    = ggml_nbytes(src0);
    dsptensor_0.type        = src0->type;

    dsptensor_0.ne[0] = src0->ne[0];
    dsptensor_0.ne[1] = src0->ne[1];
    dsptensor_0.ne[2] = src0->ne[2];
    dsptensor_0.ne[3] = src0->ne[3];

    dsptensor_0.nb[0] = src0->nb[0];
    dsptensor_0.nb[1] = src0->nb[1];
    dsptensor_0.nb[2] = src0->nb[2];
    dsptensor_0.nb[3] = src0->nb[3];

    if (2 == input_tensor_count) {
        dsptensor_1.data        = src1->data;
        dsptensor_1.type        = src1->type;
        dsptensor_1.data_len    = ggml_nbytes(src1);

        dsptensor_1.ne[0] = src1->ne[0];
        dsptensor_1.ne[1] = src1->ne[1];
        dsptensor_1.ne[2] = src1->ne[2];
        dsptensor_1.ne[3] = src1->ne[3];

        dsptensor_1.nb[0] = src1->nb[0];
        dsptensor_1.nb[1] = src1->nb[1];
        dsptensor_1.nb[2] = src1->nb[2];
        dsptensor_1.nb[3] = src1->nb[3];
    }

    dsptensor_2.data        = dst->data;
    dsptensor_2.data_len    = ggml_nbytes(dst);
    dsptensor_2.type        = dst->type;

    dsptensor_2.ne[0] = dst->ne[0];
    dsptensor_2.ne[1] = dst->ne[1];
    dsptensor_2.ne[2] = dst->ne[2];
    dsptensor_2.ne[3] = dst->ne[3];

    dsptensor_2.nb[0] = dst->nb[0];
    dsptensor_2.nb[1] = dst->nb[1];
    dsptensor_2.nb[2] = dst->nb[2];
    dsptensor_2.nb[3] = dst->nb[3];

    hexagon_error = op_func(ctx->ggmlop_handle, &dsptensor_0, &dsptensor_1, &dsptensor_2);
    if (AEE_SUCCESS != hexagon_error) {
        GGMLHEXAGON_LOG_WARN("ggmlop %s computation fail on cdsp", ggml_op_name(op->op));
    }

    op_perf.info();
    return;
}

// =================================================================================================
//  section-8: implementation of ggml-hexagon backend according to specification in ggml backend subsystem
// =================================================================================================
static bool ggmlhexagon_can_handle_op_through_cdsp(ggml_backend_dev_t dev, const struct ggml_tensor * op_tensor) {
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)dev->context;
    GGML_UNUSED(ctx);
    if (op_tensor->op == GGML_OP_NONE) {
        return true;
    }

    if (!ggmlhexagon_k_op_caps[ggmlhexagon_get_op_index(op_tensor)].supported) {
        return false;
    }

    const struct ggml_tensor * src0 = op_tensor->src[0];
    const struct ggml_tensor * src1 = op_tensor->src[1];
    switch (op_tensor->op) {
        case GGML_OP_ADD:
        {
            if (!ggml_are_same_shape(src0, src1)) {
                return false;
            }

            //TODO: offload quantize GGML_OP_ADD to cDSP
            return ggmlhexagon_same_types(ctx, op_tensor);
        }
        case GGML_OP_MUL_MAT:
        {
            ggmlhexagon_dump_op_info(op_tensor);
            if (1 == g_hexagon_appcfg.enable_q_mulmat)
                return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q6_K
                       ) && (src1->type == GGML_TYPE_F32) && (op_tensor->type == GGML_TYPE_F32);
            else
                return (src0->type == GGML_TYPE_F32) && (src1->type == GGML_TYPE_F32) && (op_tensor->type == GGML_TYPE_F32);
        }
        default:
            break;
    }
    return false;
}

static bool ggmlhexagon_can_handle_op_through_qnn(ggml_backend_dev_t dev, const struct ggml_tensor * op_tensor) {
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)dev->context;
    if (op_tensor->op == GGML_OP_NONE) {
        return true;
    }

    if (!ggmlqnn_k_op_caps[ggmlhexagon_get_op_index(op_tensor)].supported) {
        return false;
    }

    struct ggml_tensor * src0 = op_tensor->src[0];
    struct ggml_tensor * src1 = op_tensor->src[1];
    int64_t ne00        = 0;
    uint32_t src0_rank  = 0;
    uint32_t src1_rank  = 0;
    if (nullptr != src0) {
        src0_rank = ggml_n_dims(src0);
        ne00      = src0->ne[0];
    }
    if (nullptr != src1) {
        src1_rank = ggml_n_dims(src1);
    }

    switch (op_tensor->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        {
            if (!ggml_are_same_shape(src0, src1)) {
                return false;
            }

            if (ne00 < 32)
                return false;

            return ggmlhexagon_same_types(ctx, op_tensor);
        }

        case GGML_OP_DIV:
        case GGML_OP_MUL: {
            if (ctx->device == HEXAGON_BACKEND_QNNNPU)
                return false;

            if (!ggml_are_same_shape(src0, src1)) {
                return false;
            }

            if ((src0_rank != 2) || (src1_rank != 2)) //TODO: 3D and 4D matrix mul
                return false;

            return ggmlhexagon_same_types(ctx, op_tensor);
        }
        case GGML_OP_MUL_MAT:
        {
            ggmlhexagon_dump_op_info(op_tensor);
            if (src0_rank != src1_rank) // make QNN SDK happy
                return false;

            if (src0_rank < 2) // QNN's limitation, make QNN SDK happy
                return false;

            if (4 == src0_rank) //TODO: 4D matrix mulmat in CT
                return false;

            if ((src1->ne[2] != src0->ne[2]) || (src1->ne[3] != src0->ne[3])) // make QNN SDK happy
                return false;

            if (ctx->device == HEXAGON_BACKEND_QNNNPU) {
                if (1 == g_hexagon_appcfg.enable_q_mulmat)
                    return (src0->type == GGML_TYPE_F32
                        || src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0
                        || src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q8_K
                        ) && (src1->type == GGML_TYPE_F32) && (op_tensor->type == GGML_TYPE_F32);
                else
                    return (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op_tensor->type == GGML_TYPE_F32);
            } else {
                return (src0->type == GGML_TYPE_F32 || ggml_is_quantized(src0->type))
                        && (src1->type == GGML_TYPE_F32) && (op_tensor->type == GGML_TYPE_F32);
            }
        }
        case GGML_OP_LOG:
        {
            if (ctx->device == HEXAGON_BACKEND_QNNNPU)
                return false;
        }
        case GGML_OP_SQRT:
        default:
            return ggmlhexagon_same_types(ctx, op_tensor);
    }
}

static bool ggmlhexagon_compute_forward(ggml_backend_t backend, struct ggml_tensor * dst) {
    ggmlqnn_op_func_t func          = nullptr;
    ggml_backend_hexagon_context * ctx  = (ggml_backend_hexagon_context *)backend->context;

    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
        ggmlhexagon_compute(ctx, dst);
        return true;
    }

    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggmlqnn_compute_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggmlqnn_compute_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggmlqnn_compute_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
            func = ggmlqnn_compute_elementwise;
            break;
        case GGML_OP_ACC:
            ggmlqnn_compute_acc(ctx, dst);
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
            ggmlqnn_compute_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggmlqnn_compute_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggmlqnn_compute_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggmlqnn_compute_upsample_nearest2d(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggmlqnn_compute_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggmlqnn_compute_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggmlqnn_compute_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggmlqnn_compute_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggmlqnn_compute_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggmlqnn_compute_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            return false;
        case GGML_OP_SCALE:
            ggmlqnn_compute_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggmlqnn_compute_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggmlqnn_compute_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggmlqnn_compute_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggmlqnn_compute_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggmlqnn_compute_diag_mask(ctx, dst, -INFINITY);
            break;
        case GGML_OP_SOFT_MAX:
            ggmlqnn_compute_softmax(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggmlqnn_compute_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggmlqnn_compute_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggmlqnn_compute_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggmlqnn_compute_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggmlqnn_compute_argsort(ctx, dst);
            break;
        default:
            return false;
    }

    if (nullptr != func)
        func(ctx, dst);

    return true;
}

struct ggml_backend_hexagon_buffer_context {
    ~ggml_backend_hexagon_buffer_context() {
        if (buffer) {
            if ((g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
                //do nonthing here because rpc mempool was used for HWACCEL_CDSP
            } else {
                ggml_aligned_free(buffer, 0);
            }
        }

        for (auto * sub_buffer : sub_buffers) {
            free(sub_buffer);
        }

        sub_buffers.clear();
    }
    void * buffer       = nullptr;

    struct ggml_backend_hexagon_context * backend_ctx = nullptr;

    size_t buffer_size  = 0;
    std::vector<void *> sub_buffers;
};

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    return ctx->buffer;
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    GGML_UNUSED(tensor);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_hexagon_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                  struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset((char *)tensor->data + offset, value, size);
}

static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *)tensor->data + offset, size);
}

static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    memset(ctx->buffer, value, ctx->buffer_size);
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
        /* .memset_tensor   = */ ggml_backend_hexagon_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
        /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_hexagon_buffer_clear,
        /* .reset           = */ nullptr,
};

static const char * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    if ((g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
        return "hexagon-ion-buffer";
    }
    if ((g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP) && (1 == g_hexagon_appcfg.enable_rpc_dma_mempool)) {
        return "hexagon-dma-buffer";
    }
    return "hexagon-normal-buffer";
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
           ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    GGML_ASSERT(nullptr != ctx);
    GGMLHEXAGON_LOG_DEBUG("device %d(%s)", ctx->device, ggml_backend_hexagon_get_devname(ctx->device));

    ggml_backend_hexagon_buffer_context * buffer_ctx = new ggml_backend_hexagon_buffer_context;

    size_t size_page = 0;
#if defined(__ANDROID__) || defined(__linux__)
    size_page = sysconf(_SC_PAGESIZE);
#else
    SYSTEM_INFO systeminfo;
    GetSystemInfo(&systeminfo);
    size_page = systeminfo.dwPageSize;
#endif
    size_t size_aligned = size;
    if (0 != (size_aligned % size_page)) {
        size_aligned += (size_page - (size_aligned % size_page));
    }
    if ((HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
        GGML_ASSERT(size + ctx->rpc_mempool_usage <= ctx->rpc_mempool_len);
        buffer_ctx->buffer = (static_cast<char*>(ctx->rpc_mempool)) + ctx->rpc_mempool_usage;
        GGMLHEXAGON_LOG_DEBUG("size %d(%d MiB), buffer_ctx->buffer %p", size, size / SIZE_IN_MB, buffer_ctx->buffer);
        GGML_ASSERT(nullptr != buffer_ctx->buffer);
        ctx->rpc_mempool_usage += size_aligned;
    } else {
        buffer_ctx->buffer = ggml_aligned_malloc(size_aligned);
    }
    buffer_ctx->buffer_size = size_aligned;
    if (nullptr == buffer_ctx->buffer) {
        GGMLHEXAGON_LOG_WARN("%s: failed to allocate %d MiB\n", __func__, size / SIZE_IN_MB);
        return nullptr;
    } else {
        //GGMLHEXAGON_LOG_DEBUG("%s: succeed to allocate %d MiB\n", __func__, size / SIZE_IN_MB);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_hexagon_buffer_interface, buffer_ctx, size);
}

static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    GGML_ASSERT(nullptr != ctx);
    if ((HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
        GGML_ASSERT(ctx->rpc_mempool_len > (8 * SIZE_IN_MB));
        return ctx->rpc_mempool_len - (8 * SIZE_IN_MB);
    } else {
        //TODO:this is an experimental value for LLM models
        return (1024 * SIZE_IN_MB);
    }
}

static bool ggml_backend_buft_is_hexagon(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_hexagon_buffer_type_name;
}

static bool ggml_backend_hexagon_buffer_is_host(ggml_backend_buffer_type_t buft) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    GGML_ASSERT(nullptr != ctx);
    if ((HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
        //FIXME: return false here is make sense in this scenario although this is not key-point at the moment
        //       fix it after solving other urgent tasks
        //return false;
    }
    return true;
}

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *) backend->context;
    return g_hexagon_mgr[ctx->device].name;
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)backend->context;

    qnn_instance * instance = (qnn_instance*)g_hexagon_mgr[ctx->device].instance;
    if (nullptr != instance) {
        std::map<std::string, qnn_singlenode_res_t>::iterator singlenode_graph_it;
        for (singlenode_graph_it = ctx->qnn_singlenode_graph_map.begin();
             singlenode_graph_it != ctx->qnn_singlenode_graph_map.end(); singlenode_graph_it++) {
            auto & graph_res = singlenode_graph_it->second;
            Qnn_GraphHandle_t & graph_handle    = std::get<0>(graph_res);
            qnn_ptensors_t    & ptensors        = std::get<1>(graph_res);
            for (auto tensor_it = ptensors.begin(); tensor_it != ptensors.end(); ++tensor_it) {
                ggmlqnn_free_qnntensor(*tensor_it);
            }
            GGML_UNUSED(graph_handle);
            GGMLHEXAGON_LOG_DEBUG("clean up graph:%s", singlenode_graph_it->first.c_str());
        }
        ctx->qnn_singlenode_graph_map.clear();

        instance->qnn_finalize();
        delete instance;
        g_hexagon_mgr[ctx->device].instance = nullptr;
    }

    if (nullptr != g_hexagon_mgr[ctx->device].backend) {
        //print timestamp and dsp information before deinit cdsp, useful for troubleshooting
        ggmlhexagon_print_running_timestamp(ctx);
        if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
            ggmlhexagon_deinit_cdsp(ctx);
        }

        delete backend;
        g_hexagon_mgr[ctx->device].backend = nullptr;
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
}

static enum ggml_status ggmlhexagon_backend_graph_compute_general(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status result         = GGML_STATUS_SUCCESS;
    ggml_backend_hexagon_context * ctx  = (ggml_backend_hexagon_context *)backend->context;
    GGML_UNUSED(ctx);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        bool ok = ggmlhexagon_compute_forward(backend, node);
        if (!ok) {
            GGMLHEXAGON_LOG_DEBUG("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
    }

    return result;
}

static const char * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
    if (nullptr == ctx) {
        GGMLHEXAGON_LOG_ERROR("pls check why ctx is null");
        return "unknown";
    }
    return ctx->name;
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
    static char hexagon_device_desc[GGMLHEXAGON_TMPBUF_LEN];
    if (nullptr == ctx) {
        GGMLHEXAGON_LOG_ERROR("pls check why ctx is null");
        return "unknown";
    }

    if (0 == strncmp(ctx->name, "qnn-npu", 7)) {
        const char * soc_info = ggmlhexagon_get_socmodel_desc(ctx->socinfo.soc_model);
        const char * htp_arch = ggmlhexagon_get_htparch_desc(ctx->socinfo.htp_arch);
        std::string dev_desc = std::string(ctx->desc)
                + std::string(soc_info) + "_" + std::string(htp_arch)
                + "," + std::string(ctx->socinfo.soc_desc);
        memset(hexagon_device_desc, 0, GGMLHEXAGON_TMPBUF_LEN);
        memcpy(hexagon_device_desc, dev_desc.c_str(), strlen(dev_desc.c_str()));
        return hexagon_device_desc;
    } else {
        return ctx->desc;
    }
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
    if ((nullptr == ctx) || (ctx->device > HEXAGON_BACKEND_GGML)) {
        GGMLHEXAGON_LOG_ERROR("pls check params");
        *free = 0;
        *total = 0;
    }

    if (HEXAGON_BACKEND_QNNCPU == ctx->device || HEXAGON_BACKEND_GGML == ctx->device) {
        *total = ggmlhexagon_get_system_total_memory_in_bytes();
        *free = ggmlhexagon_get_system_free_memory_in_bytes();
    } else if (HEXAGON_BACKEND_QNNGPU == ctx->device) {
        //TODO: probe GPU info in Qualcomm Adreno GPU
        *total = ggmlhexagon_get_system_total_memory_in_bytes();
        *free = ggmlhexagon_get_system_free_memory_in_bytes();
    } else if (HEXAGON_BACKEND_QNNNPU == ctx->device) {
        size_t rpc_ion_memsize = 0;
        size_t rpc_ion_usage   = 0;
        if (HWACCEL_CDSP != g_hexagon_appcfg.hwaccel_approach) {
            rpc_ion_memsize = ctx->instance->get_rpcmem_capacity();
            rpc_ion_usage   = ctx->instance->get_rpcmem_usage();
        } else {
            rpc_ion_memsize = ctx->rpc_mempool_capacity;
            rpc_ion_usage   = ctx->rpc_mempool_usage;
        }
        *total = rpc_ion_memsize;
        *free = (rpc_ion_memsize - rpc_ion_usage);
        GGMLHEXAGON_LOG_DEBUG("rpc memsize %d MiB", rpc_ion_memsize / SIZE_IN_MB);
        GGMLHEXAGON_LOG_DEBUG("rpc usage %d MiB\n\n", rpc_ion_usage / SIZE_IN_MB);
    }
}

static enum ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    }

    if (HEXAGON_BACKEND_QNNCPU == ctx->device)
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    else if (HEXAGON_BACKEND_QNNGPU == ctx->device)
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    else if (HEXAGON_BACKEND_QNNNPU == ctx->device)
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    else
        return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static void ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev,
                                              struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
            /* .async                 = */ false,
            /* .host_buffer           = */ true,
            /* .buffer_from_host_ptr  = */ false,
            /* .events                = */ false,
    };

    if ((HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
        //don't use system memory in this scenario
        props->caps.host_buffer       = false;
    }
}

static ggml_backend_t ggml_backend_hexagon_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    size_t dev_index = 0;

    //case-1: test-backend-ops or other similar scenario: calling ggml_backend_dev_init(dev, reinterpret_cast<const char *>(i)) directly in user's code
    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    if (nullptr == params) {
        GGMLHEXAGON_LOG_DEBUG("program specified param is nullptr");
        dev_index = (g_hexagon_appcfg.hexagon_backend > 0) ? g_hexagon_appcfg.hexagon_backend : 0;
        if (dev_index >= GGML_HEXAGON_MAX_DEVICES) {
            GGMLHEXAGON_LOG_INFO("assume the default ggml backend");
            return nullptr;
        }
    } else {
        GGMLHEXAGON_LOG_INFO("program specified param is not nullptr");
        //user's program calling ggml_backend_hexagon_device_init_backend directly
        dev_index = (int)(intptr_t)params;
        g_hexagon_appcfg.hexagon_backend = dev_index;
        GGMLHEXAGON_LOG_INFO("program specified dev_index %d\n", dev_index);
    }
    GGMLHEXAGON_LOG_DEBUG("hexagon_backend=%d", dev_index);
    ggml_backend_t hexagon_backend = ggml_backend_hexagon_init(dev_index, g_hexagon_appcfg.runtime_libpath);
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);

    return hexagon_backend;

}

static ggml_backend_buffer_type_t ggml_backend_hexagon_buffer_type(size_t device_index) {
    if (device_index >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_DEBUG("ggml_backend_hexagon_buffer_type error: device_index:%d is out of range [0, %d]\n",
                      device_index, GGML_HEXAGON_MAX_DEVICES - 1);
        return nullptr;
    }

    static struct ggml_backend_buffer_type ggml_backend_buffer_type_hexagon = {
            /* .iface   = */ {
                                     /* .get_name         = */ ggml_backend_hexagon_buffer_type_name,
                                     /* .alloc_buffer     = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
                                     /* .get_alignment    = */ ggml_backend_hexagon_buffer_type_get_alignment,
                                     /* .get_max_size     = */ ggml_backend_hexagon_buffer_type_get_max_size,
                                     /* .get_alloc_size   = */ nullptr,// defaults to ggml_nbytes
                                     /* .is_host          = */ ggml_backend_hexagon_buffer_is_host
                             },
            /* .device  = */ nullptr,
            /* .context = */ &g_hexagon_mgr[device_index],
    };

    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
        //here is the trick:
        //there only 1 backend_device when g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP
        //and we need to re-use the g_hexagon_mgr
        //so context is g_hexagon_mgr[HEXAGON_BACKEND_CDSP] rather than g_hexagon_mgr[0]
        ggml_backend_buffer_type_hexagon.context = &g_hexagon_mgr[HEXAGON_BACKEND_CDSP];
    }

    return &ggml_backend_buffer_type_hexagon;
}

static const char * ggml_backend_hexagon_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "Hexagon_Host";
}

static const char * ggml_backend_hexagon_host_buffer_name(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return "Hexagon_Host";
}

static void ggml_backend_hexagon_host_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, 0);
}

static void * ggml_hexagon_host_malloc(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_aligned_malloc(size);
}

static ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * host_ptr = ggml_hexagon_host_malloc(buft, size);

    if (nullptr == host_ptr) {
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(host_ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_hexagon_host_buffer_free;

    return buffer;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_hexagon_buffer_type_host = {
            /* .iface    = */ {
                                      /* .get_name         = */ ggml_backend_hexagon_host_buffer_type_name,
                                      /* .alloc_buffer     = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
                                      /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                                      /* .get_max_size     = */ nullptr,
                                      /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                                      /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                              },
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_hexagon_reg(), 0),
            /* .context  = */ nullptr,
    };

    return &ggml_backend_hexagon_buffer_type_host;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_hexagon_host_buffer_type();
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)dev->context;
    return ggml_backend_hexagon_buffer_type(ctx->device);
}

static ggml_backend_buffer_t ggml_backend_hexagon_device_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if ((HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
        if (ggml_backend_buft_is_hexagon(buft)) {
            ggml_backend_hexagon_context * dev_ctx  = (ggml_backend_hexagon_context *)dev->context;
            ggml_backend_hexagon_context * buft_ctx = (ggml_backend_hexagon_context *)buft->context;
            return buft_ctx->device == dev_ctx->device;
        }
    }

    return ggml_backend_buft_is_host(buft);
}

static struct ggml_backend_device_i ggml_backend_hexagon_device_interface = {
        /* .get_name             = */ ggml_backend_hexagon_device_get_name,
        /* .get_description      = */ ggml_backend_hexagon_device_get_description,
        /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
        /* .get_type             = */ ggml_backend_hexagon_device_get_type,
        /* .get_props            = */ ggml_backend_hexagon_device_get_props,
        /* .init_backend         = */ ggml_backend_hexagon_device_init_backend,
        /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
        /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
        /* .buffer_from_host_ptr = */ ggml_backend_hexagon_device_buffer_from_host_ptr,
        /* .supports_op          = */ nullptr,
        /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
        /* .offload_op           = */ nullptr,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
};

static ggml_backend_i ggml_backend_hexagon_interface = {
        /* .get_name                = */ ggml_backend_hexagon_name,
        /* .free                    = */ ggml_backend_hexagon_free,
        /* .set_tensor_async        = */ nullptr,
        /* .get_tensor_async        = */ nullptr,
        /* .cpy_tensor_async        = */ nullptr,
        /* .synchronize             = */ nullptr,
        /* .graph_plan_create       = */ nullptr,
        /* .graph_plan_free         = */ nullptr,
        /* .graph_plan_update       = */ nullptr,
        /* .graph_plan_compute      = */ nullptr,
        /* .graph_compute           = */ nullptr,
        /* .event_record            = */ nullptr,
        /* .event_wait              = */ nullptr,
};

//FIXME: this guid is not make sense
static ggml_guid_t ggml_backend_hexagon_guid() {
    static ggml_guid guid = {
            0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
            0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09
    };
    return &guid;
}

bool ggml_backend_is_hexagon(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hexagon_guid());
}

static void ggml_backend_hexagon_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_hexagon(backend));

    struct ggml_backend_hexagon_context * ctx = (struct ggml_backend_hexagon_context *)backend->context;
    ctx->n_threads = n_threads;
}

int ggml_backend_hexagon_get_device_count() {
    if (g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP) {
        GGML_ASSERT(g_hexagon_appcfg.hexagon_backend == HEXAGON_BACKEND_CDSP);
        return 1;
    } else {
        return GGML_HEXAGON_MAX_DEVICES;
    }
}

struct ggml_backend_hexagon_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "ggml-hexagon";
}

static size_t ggml_backend_hexagon_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
        GGML_ASSERT(g_hexagon_appcfg.hexagon_backend == HEXAGON_BACKEND_CDSP);
        //here is the trick:
        //there only 1 backend_device when g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP
        //so return 1
        return 1;
    } else {
        return GGML_HEXAGON_MAX_DEVICES;
    }
}

static ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_UNUSED(index);

    GGMLHEXAGON_LOG_DEBUG("index %d", index);
    ggml_backend_hexagon_reg_context * ctx = (ggml_backend_hexagon_reg_context *)reg->context;
    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
        GGML_ASSERT(g_hexagon_appcfg.hexagon_backend == HEXAGON_BACKEND_CDSP);
        //here is the trick:
        //there only 1 backend_device when g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP
        //so return ctx->devices[0]
        return ctx->devices[0];
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

static void * ggml_backend_hexagon_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (nullptr == name)
        return nullptr;

    const char * slot_name =  "ggml_backend_set_n_threads";
    if (0 == memcmp(name, slot_name, strlen(slot_name))) {
        return (void *)ggml_backend_hexagon_set_n_threads;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hexagon_reg_interface = {
        /* .get_name          = */ ggml_backend_hexagon_reg_get_name,
        /* .get_device_count  = */ ggml_backend_hexagon_reg_get_device_count,
        /* .get_device        = */ ggml_backend_hexagon_reg_get_device,
        /* .get_proc_address  = */ ggml_backend_hexagon_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;
    GGMLHEXAGON_LOG_DEBUG("enter ggml_backend_hexagon_reg");

    //case-2: normal scenario, such as llama-cli or UI applicaton
    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_hexagon_reg_context * ctx = new ggml_backend_hexagon_reg_context;

            for (int i = 0; i < ggml_backend_hexagon_get_device_count(); i++) {
                if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
                    ggml_backend_hexagon_device_interface.supports_op = ggmlhexagon_can_handle_op_through_cdsp;
                } else {
                    ggml_backend_hexagon_device_interface.supports_op = ggmlhexagon_can_handle_op_through_qnn;
                }
                if ((HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) && (1 == g_hexagon_appcfg.enable_rpc_ion_mempool)) {
                    //don't use system memory in this scenario
                    ggml_backend_hexagon_device_interface.get_host_buffer_type = nullptr;
                }

                GGMLHEXAGON_LOG_DEBUG("create backend device for device %d", i);
                ggml_backend_dev_t dev = new ggml_backend_device{
                        /* .iface       = */ ggml_backend_hexagon_device_interface,
                        /* .reg         = */ &reg,
                        /* .context     = */ &g_hexagon_mgr[i]
                };
                if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
                    //here is the trick:
                    //there only 1 backend_device when g_hexagon_appcfg.hwaccel_approach == HWACCEL_CDSP
                    //and we need to re-use the g_hexagon_mgr
                    //so context is g_hexagon_mgr[HEXAGON_BACKEND_CDSP] rather than g_hexagon_mgr[0]
                    dev->context = &g_hexagon_mgr[HEXAGON_BACKEND_CDSP];
                }
                ctx->devices.push_back(dev);

                //here is the trick: make cDSP rpc memory pool happy because ggml's backend subsystem need this
                if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
                    GGML_ASSERT(HEXAGON_BACKEND_CDSP == g_hexagon_appcfg.hexagon_backend);
                    int result = ggmlhexagon_init_dsp(&g_hexagon_mgr[HEXAGON_BACKEND_CDSP]);
                    if (0 != result) {
                        GGMLHEXAGON_LOG_INFO("init hexagon dsp failure");
                    }
                    GGML_ASSERT(0 == result);
                }
            }

            reg = ggml_backend_reg {
                    /* .api_version = */ GGML_BACKEND_API_VERSION,
                    /* .iface       = */ ggml_backend_hexagon_reg_interface,
                    /* .context     = */ ctx
            };
        }

        initialized = true;
    }
    GGMLHEXAGON_LOG_DEBUG("leave ggml_backend_hexagon_reg");

    return &reg;
}

const char * ggml_backend_hexagon_get_devname(size_t dev_num) {
    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
       if (HEXAGON_BACKEND_CDSP == dev_num)
            return "HEXAGON_BACKEND_CDSP";
    }

    //here is the trick: fall back for various scenarios
    switch (dev_num) {
        case HEXAGON_BACKEND_QNNCPU:
            return "HEXAGON_BACKEND_QNN_CPU";
        case HEXAGON_BACKEND_QNNGPU:
            return "HEXAGON_BACKEND_QNN_GPU";
        case HEXAGON_BACKEND_QNNNPU:
            return "HEXAGON_BACKEND_QNN_NPU";
        case HEXAGON_BACKEND_GGML:
            return "ggml"; //"fake" QNN backend, used for compare performance between hexagon backend and the default ggml backend
        default:
            return "unknown";
    }
}

static qnn_instance * ggmlqnn_init_qnn_instance(size_t device, const char * qnn_lib_path) {
    int result = 0;
    GGMLHEXAGON_LOG_INFO("hwaccel approach=%d(%s)", g_hexagon_appcfg.hwaccel_approach,
                     ggmlhexagon_get_hwaccel_approach_name(g_hexagon_appcfg.hwaccel_approach));

    qnn_instance * instance = nullptr;
    instance = new qnn_instance(qnn_lib_path, g_hexagon_mgr[device].lib, "");
    result = instance->qnn_init(nullptr);
    if (0 != result) {
        GGMLHEXAGON_LOG_WARN("init qnn subsystem failed with qnn backend %s, pls check why\n",
                         ggml_backend_hexagon_get_devname(device));
        delete instance;
        return nullptr;
    }
    qnn_interface qnn_interface = instance->get_qnn_interface();
    if (!qnn_interface.is_loaded()) {
        GGMLHEXAGON_LOG_WARN("qnn subsystem failure\n");
        delete instance;
        return nullptr;
    }

    std::string device_name = ggml_backend_hexagon_get_devname(device);
    GGMLHEXAGON_LOG_INFO("qnn device name %s", device_name.c_str());
    g_hexagon_mgr[device].instance = instance;
    g_hexagon_mgr[device].raw_interface = instance->get_qnn_raw_interface();
    g_hexagon_mgr[device].raw_system_interface = instance->get_qnn_raw_system_interface();

    return instance;
}

/**
 *
 * @param device            0: HEXAGON_BACKEND_QNNCPU 1: HEXAGON_BACKEND_QNNGPU 2: HEXAGON_BACKEND_QNNNPU/HEXAGON_BACKEND_CDSP
 * @param runtime_libpath   binary runtime library path, such as "/data/local/tmp/" on Android or specified in user's code
 * @return
 */
ggml_backend_t ggml_backend_hexagon_init(size_t device, const char * runtime_libpath) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);
    if (nullptr == runtime_libpath)
        return nullptr;

    //case-3: calling ggml_backend_hexagon_init() directly in user's code
    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    GGMLHEXAGON_LOG_DEBUG("device %d", device);
    GGMLHEXAGON_LOG_DEBUG("runtime libpath %s", runtime_libpath);
    if (device >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_ERROR("invalid device %d", device);
        return nullptr;
    }

    if (0 != memcmp(runtime_libpath, g_hexagon_appcfg.runtime_libpath, strlen(g_hexagon_appcfg.runtime_libpath))) {
        //re-setting runtime libpath
        ggmlhexagon_set_runtime_path(device, runtime_libpath);
    }

    if (nullptr != g_hexagon_mgr[device].backend) {
        GGMLHEXAGON_LOG_DEBUG("backend %d(%s) already loaded", device,
                         ggml_backend_hexagon_get_devname(device));
        GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);
        return g_hexagon_mgr[device].backend;
    }

    //don't initialize QNN when hwaccel approach is offload ggml op to Hexagon cDSP directly
    if (HWACCEL_CDSP != g_hexagon_appcfg.hwaccel_approach) {
        qnn_instance * instance = ggmlqnn_init_qnn_instance(device, runtime_libpath);
        if (nullptr == instance)
            return nullptr;
    }
    ggml_backend_hexagon_interface.graph_compute = ggmlhexagon_backend_graph_compute_general;
    ggml_backend_t hexagon_backend = new ggml_backend{
            /* .guid      = */ ggml_backend_hexagon_guid(),
            /* .iface     = */ ggml_backend_hexagon_interface,
            /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_hexagon_reg(), device),
            /* .context   = */ &g_hexagon_mgr[device]
    };

    g_hexagon_mgr[device].backend = hexagon_backend;
    if (HWACCEL_CDSP == g_hexagon_appcfg.hwaccel_approach) {
        int result = ggmlhexagon_init_dsp(&g_hexagon_mgr[device]);
        if (0 != result) {
            GGMLHEXAGON_LOG_INFO("init hexagon dsp failure");
            ggml_backend_hexagon_free(hexagon_backend);
            return nullptr;
        }
    } else {
        //got fully description of SoC when hwaccel approach is HWACCEL_QNN and backend is HEXAGON_BACKEND_QNNNPU
        GGMLHEXAGON_LOG_INFO("device name %s", ggml_backend_hexagon_device_get_description(hexagon_backend->device));
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__);

    return hexagon_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)

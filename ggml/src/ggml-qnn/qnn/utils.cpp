
#include "utils.hpp"

#include <cstdlib>

#include "ggml-qnn.h"
#include "qnn-types.hpp"
#include "QnnGraph.h"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <sys/sysinfo.h>
#    include <unistd.h>
#endif

namespace {

template <typename _Ty> _Ty align_to_generic(size_t alignment, _Ty offset) {
    return offset % alignment == 0 ? offset :
                                     offset + (static_cast<_Ty>(alignment) - (offset % static_cast<_Ty>(alignment)));
}

}  // namespace

namespace qnn {

qnn_dimension_array_t get_internal_dimension(const ggml_dimension_array_t & dims, uint32_t rank) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS should be 4");
    GGML_ASSERT(rank <= GGML_MAX_DIMS && rank > 0);

    qnn_dimension_array_t internal_dims = {};
    /*
     * Both the ggml and qnn tensor in memory are stored as row-major format.
     * But the dimensions of the tensor are stored in different order.
     * For example, a 2x3 matrix:
     *   [
     *     [1, 2, 3],
     *     [4, 5, 6],
     *   ]
     * The ggml tensor will have dimensions [3, 2], while the qnn tensor will have dimensions [2, 3].
     */
    for (uint32_t i = 0; i < rank; i++) {
        internal_dims[i] = std::max<uint32_t>((uint32_t) dims[rank - 1 - i], 1);
    }

    return internal_dims;
}

qnn_dimension_array_t get_view_internal_dimension(const ggml_tensor * tensor, size_t & element_offset_out) {
    element_offset_out = 0;

    auto * parent_tensor = tensor;
    while (parent_tensor->view_src) {
        element_offset_out += parent_tensor->view_offs;
        parent_tensor = parent_tensor->view_src;
    }

    const auto rank        = get_ggml_tensor_rank(tensor);
    const auto parent_rank = get_ggml_tensor_rank(parent_tensor);
    GGML_ASSERT(parent_tensor->type == tensor->type);
    GGML_ASSERT(parent_rank == rank);

    const auto block_size = ggml_blck_size(tensor->type);
    element_offset_out =
        element_offset_out * block_size / tensor->nb[0];  // calculate the element offset in the view tensor

    return get_internal_dimension(parent_tensor->ne, parent_rank);
}

// TODO: mapping more ggml data type to QNN data type
// ref:explanation of k-quants, https://github.com/ggerganov/llama.cpp/pull/1684
Qnn_DataType_t qnn_datatype_from_ggml_datatype(ggml_type ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_F32:
            return QNN_DATATYPE_FLOAT_32;
        case GGML_TYPE_F16:
            return QNN_DATATYPE_FLOAT_16;
        case GGML_TYPE_I32:
            return QNN_DATATYPE_INT_32;
        case GGML_TYPE_I16:
            return QNN_DATATYPE_INT_16;
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

ggml_type ggml_datatype_from_qnn_datatype(Qnn_DataType_t qnn_type) {
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

size_t qnn_datatype_size(Qnn_DataType_t qnn_type) {
    switch (qnn_type) {
        case QNN_DATATYPE_FLOAT_32:
            return sizeof(float);
        case QNN_DATATYPE_FLOAT_16:
            return sizeof(uint16_t);
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_INT_32:
            return sizeof(int32_t);
        case QNN_DATATYPE_INT_16:
            return sizeof(int16_t);
        case QNN_DATATYPE_INT_8:
            return sizeof(int8_t);
        case QNN_DATATYPE_SFIXED_POINT_8:
            return sizeof(int8_t);
        case QNN_DATATYPE_SFIXED_POINT_4:
            return sizeof(int8_t);
        default:
            break;
    }
    return 0;
}

const char * qnn_datatype_to_string(Qnn_DataType_t qnn_type) {
    switch (qnn_type) {
        case QNN_DATATYPE_FLOAT_32:
            return "QNN_DATATYPE_FLOAT_32";
        case QNN_DATATYPE_FLOAT_16:
            return "QNN_DATATYPE_FLOAT_16";
        case QNN_DATATYPE_UINT_32:
            return "QNN_DATATYPE_UINT_32";
        case QNN_DATATYPE_INT_32:
            return "QNN_DATATYPE_INT_32";
        case QNN_DATATYPE_INT_16:
            return "QNN_DATATYPE_INT_16";
        case QNN_DATATYPE_INT_8:
            return "QNN_DATATYPE_INT_8";
        case QNN_DATATYPE_SFIXED_POINT_8:
            return "QNN_DATATYPE_SFIXED_POINT_8";
        case QNN_DATATYPE_SFIXED_POINT_4:
            return "QNN_DATATYPE_SFIXED_POINT_4";
        default:
            break;
    }

    return "QNN_DATATYPE_UNDEFINED";
}

uint32_t get_ggml_tensor_rank(const ggml_tensor * tensor) {
    uint32_t rank = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if ((0 != tensor->ne[i]) && (1 != tensor->ne[i])) {
            rank++;
        }
    }
    return rank;
}

const char * get_ggml_type_name(ggml_type type) {
    const auto * traits = ggml_get_type_traits(type);
    return traits->type_name;
}

const char * get_backend_name(backend_index_type device) {
    switch (device) {
        case QNN_BACKEND_CPU:
            return "qnn-cpu";
        case QNN_BACKEND_GPU:
            return "qnn-gpu";
        case QNN_BACKEND_NPU:
            return "qnn-npu";
        case QNN_BACKEND_COUNT:
        default:
            return "unknown";
    }
}

const char * get_backend_desc(backend_index_type device) {
    switch (device) {
        case QNN_BACKEND_CPU:
            return "CPU";
        case QNN_BACKEND_GPU:
            return "Adreno GPU";
        case QNN_BACKEND_NPU:
            return "Hexagon NPU";
        case QNN_BACKEND_COUNT:
        default:
            return "unknown";
    }
}

const char * get_chipset_desc(uint32_t soc_model) {
    switch (soc_model) {
        case SM8350:
            return "Snapdragon 888/888+";
        case SM8450:
            return "Snapdragon 8 Gen 1";
        case SM8475:
            return "Snapdragon 8 Gen 1+";
        case SM8550:
            return "Snapdragon 8 Gen 2";
        case SM7675:
            return "Snapdragon 7+ Gen 3";
        case SM8635:
            return "Snapdragon 8s Gen 3";
        case SM8650:
            return "Snapdragon 8 Gen 3";
        case SM8750:
            return "Snapdragon 8 Elite";
        case EMULATOR_AARCH64:
            return "AArch64 Emulator";
        case EMULATOR_X64:
            return "x86_64 Emulator";
        default:
            return "unknown";
    }
}

const char * get_chipset_model(uint32_t soc_model) {
    switch (soc_model) {
        case SM8350:
            return "SM8350";
        case SM8450:
            return "SM8450";
        case SA8295:
            return "SA8295";
        case SM8475:
            return "SM8475";
        case SM8550:
            return "SM8550";
        case SSG2115P:
            return "SSG2115P";
        case SM7675:
            return "SM7675";
        case SM8635:
            return "SM8635";
        case SM8650:
            return "SM8650";
        case SM8750:
            return "SM8750";
        case EMULATOR_AARCH64:
            return "AARCH64EMU";
        case EMULATOR_X64:
            return "X64EMU";
        default:
            return "unknown";
    }
}

const char * get_htparch_desc(size_t htp_arch) {
    switch (htp_arch) {
        case V68:
            return "HTP_V68";
        case V69:
            return "HTP_V69";
        case V73:
            return "HTP_V73";
        case V75:
            return "HTP_V75";
        case V79:
            return "HTP_V79";
        default:
            return "unknown";
    }
}

intptr_t align_to(size_t alignment, intptr_t offset) {
    return align_to_generic<intptr_t>(alignment, offset);
}

uint32_t get_ggml_tensor_data_size(const ggml_tensor * tensor) {
    return (uint32_t) ggml_nbytes(tensor);
}

const char * get_qnn_tensor_type_name(Qnn_TensorType_t type) {
    switch (type) {
        case QNN_TENSOR_TYPE_APP_WRITE:
            return "QNN_TENSOR_TYPE_APP_WRITE";
        case QNN_TENSOR_TYPE_APP_READ:
            return "QNN_TENSOR_TYPE_APP_READ";
        case QNN_TENSOR_TYPE_APP_READWRITE:
            return "QNN_TENSOR_TYPE_APP_READWRITE";
        case QNN_TENSOR_TYPE_STATIC:
            return "QNN_TENSOR_TYPE_STATIC";
        case QNN_TENSOR_TYPE_NATIVE:
            return "QNN_TENSOR_TYPE_NATIVE";
        case QNN_TENSOR_TYPE_UNDEFINED:
            return "QNN_TENSOR_TYPE_UNDEFINED";
        case QNN_TENSOR_TYPE_NULL:
            return "QNN_TENSOR_TYPE_NULL";
        default:
            break;
    }

    return "unknown";
}

#ifdef _WIN32
static void * _align_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}

static size_t _get_page_size() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

void align_free(void * ptr) {
    _aligned_free(ptr);
}
#else
static void * _align_alloc(size_t alignment, size_t size) {
    return std::aligned_alloc(alignment, size);
}

static size_t _get_page_size() {
    return sysconf(_SC_PAGESIZE);
}

void align_free(void * ptr) {
    std::free(ptr);
}
#endif

void * page_align_alloc(size_t size) {
    const size_t alignment    = _get_page_size();
    size_t       size_aligned = align_to_generic<size_t>(alignment, size);
    void *       data         = _align_alloc(alignment, size_aligned);
    if (!data) {
        QNN_LOG_WARN("_align_alloc failed, alignment: %ld, size: %ld, size_aligned: %ld\n", alignment, size,
                     size_aligned);
        return nullptr;
    }

    QNN_LOG_DEBUG("_align_alloc success, alignment: %ld, size: %ld, size_aligned: %ld\n", alignment, size,
                  size_aligned);
    return data;
}

// =================================================================================================
//
//  QNN backend internal helper functions
//
// =================================================================================================
// TODO: only support GGML_OP_ADD/GGML_OP_MUL/GGML_OP_MUL_MAT
const char * opname_from_ggmlop(enum ggml_op ggmlop) {
    switch (ggmlop) {
        case GGML_OP_ADD:
            return QNN_OP_ELEMENT_WISE_ADD;
        case GGML_OP_MUL:
            return QNN_OP_ELEMENT_WISE_MULTIPLY;
        case GGML_OP_MUL_MAT:
            return QNN_OP_MAT_MUL;
        default:
            break;
    }
    return nullptr;
}

const char * get_qnn_error_string(Qnn_ErrorHandle_t error) {
    // A complete list of error codes can be found at here:
    // https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/api_error_codes.html
    thread_local static char error_code[128] = {};
    switch (error) {
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
            if (error >= QNN_GRAPH_MIN_ERROR && error < QNN_GRAPH_MAX_ERROR) {
                snprintf(error_code, sizeof(error_code), "UNKNOWN_GRAPH_ERROR_%d", int(error - QNN_GRAPH_MIN_ERROR));
            } else {
                snprintf(error_code, sizeof(error_code), "%d", int(error));
            }
            return error_code;
    }
}

}  // namespace qnn

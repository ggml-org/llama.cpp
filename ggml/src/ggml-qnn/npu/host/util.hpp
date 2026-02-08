#include "ggml-impl.h"
#include "hexagon_npu.h"
#include "rpc-interface.hpp"

namespace hexagon {

enum npu_device_tensor_op        op_to_npu_op(ggml_op op);
const char *                     get_npu_op_desc(enum npu_device_tensor_op op);
enum npu_device_tensor_data_type type_to_npu_type(ggml_type type);

// TODO: merge with qcom_htp_arch
enum hexagon_dsp_arch {
    NONE = 0,
    V68,
    V69,
    V73,
    V75,
    V79,  // SD 8 Gen 4 (SM8750)
};

hexagon_dsp_arch get_dsp_arch(common::rpc_interface_ptr rpc_interface, uint32_t domain_id);

const char * get_dsp_arch_desc(hexagon_dsp_arch arch);

void enable_unsigned_dsp_module(common::rpc_interface_ptr rpc_interface, uint32_t domain_id);
void set_fast_rpc_stack_size(common::rpc_interface_ptr rpc_interface, uint32_t domain_id, uint32_t stack_size);

void get_op_tensor_desc(const ggml_tensor * dst, char * out, size_t max_len);

constexpr const size_t kMaxNpuRpcStructSize = 100;  // TODO: figure out the actual size

}  // namespace hexagon

#include <algorithm>
#include "ggml-cpu.h"

// static kernel selection for fixed-length kernels
static int ggml_get_riscv_v_kernel_idx() {
    int vlen = ggml_cpu_get_riscv_vlen();
    vlen = std::min(vlen, 256);
    return vlen / 128;
}

extern "C" {
    int ggml_rvv_kernel_idx = ggml_get_riscv_v_kernel_idx();
}

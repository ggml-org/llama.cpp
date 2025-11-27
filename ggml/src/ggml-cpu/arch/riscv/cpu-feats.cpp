#include <algorithm>

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

// static kernel selection for fixed-length kernels
static int ggml_get_riscv_v_kernel_idx() {
    int vlen = ggml_cpu_get_riscv_vlen();
    vlen = std::min(vlen, 256);
    return vlen / 128;
}

extern "C" int kernel_idx = ggml_get_riscv_v_kernel_idx();

#if defined(__riscv) && __riscv_xlen == 64
#include <sys/auxv.h>

//https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/hwcap.h#L24
#ifndef COMPAT_HWCAP_ISA_V
#define COMPAT_HWCAP_ISA_V (1 << ('V' - 'A'))
#endif

struct riscv64_features {
    bool has_rvv = false;

    riscv64_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);

        has_rvv = !!(hwcap & COMPAT_HWCAP_ISA_V);
    }
};

static int ggml_backend_cpu_riscv64_score() {
    int score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    if (!rf.has_rvv) { return 0; }
    score += 1 << 1;
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64

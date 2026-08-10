#pragma once

#if defined(__aarch64__) || defined(_M_ARM64)

#if defined(__linux__)
#include <sys/auxv.h>
#include <sys/prctl.h>

#if !defined(HWCAP2_SVE2)
#define HWCAP2_SVE2 (1ULL << 1)
#endif

#if !defined(HWCAP_FPHP)
#define HWCAP_FPHP (1 << 9)
#endif

#if !defined(HWCAP_ASIMDHP)
#define HWCAP_ASIMDHP (1 << 10)
#endif

#if !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1ULL << 13)
#endif

#if !defined(HWCAP_ASIMDDP)
#define HWCAP_ASIMDDP (1 << 20)
#endif

#if !defined(HWCAP_SVE)
#define HWCAP_SVE (1 << 22)
#endif

#if !defined(HWCAP2_SME)
#define HWCAP2_SME (1ULL << 23)
#endif

#if !defined(HWCAP2_SME2)
#define HWCAP2_SME2 (1ULL << 37)
#endif

#if !defined(PR_SVE_GET_VL)
#define PR_SVE_GET_VL 51
#endif

#if !defined(PR_SVE_VL_LEN_MASK)
#define PR_SVE_VL_LEN_MASK 0xffff
#endif

#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>

#if !defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif

#if !defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE_INSTRUCTIONS_AVAILABLE 46
#endif

#if !defined(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE 47
#endif

#if !defined(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE 66
#endif

#if !defined(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE 67
#endif

#if !defined(PF_ARM_SME_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SME_INSTRUCTIONS_AVAILABLE 70
#endif

#if !defined(PF_ARM_SME2_INSTRUCTIONS_AVAILABLE)
#define PF_ARM_SME2_INSTRUCTIONS_AVAILABLE 71
#endif

#endif

typedef struct ggml_feats_arch64_runtime {
    bool has_dotprod;
    bool has_fp16;
    bool has_sve;
    bool has_sve2;
    bool has_i8mm;
    bool has_sme;
    bool has_sme2;
    int sve_cnt;
} ggml_feats_arch64_runtime_t;

static inline ggml_feats_arch64_runtime_t ggml_get_aarch64_runtime_features(void) {
    ggml_feats_arch64_runtime_t runtime_feat = {};

#if defined(__linux__)
    const unsigned long hwcap  = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);

    runtime_feat.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    runtime_feat.has_fp16    = !!(hwcap & HWCAP_FPHP) && !!(hwcap & HWCAP_ASIMDHP);;
    runtime_feat.has_sve     = !!(hwcap & HWCAP_SVE);
    runtime_feat.has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
    runtime_feat.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    runtime_feat.has_sme     = !!(hwcap2 & HWCAP2_SME);
    runtime_feat.has_sme2    = !!(hwcap2 & HWCAP2_SME2);

    if (runtime_feat.has_sve) {
        const int vl = prctl(PR_SVE_GET_VL);
        if (vl >= 0) {
            runtime_feat.sve_cnt = vl & PR_SVE_VL_LEN_MASK;
        }
    }
#elif defined(__APPLE__)
    int oldp = 0;
    size_t size = sizeof(oldp);

    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_dotprod = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_FP16", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_fp16 = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SVE", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sve = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SVE2", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sve2 = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_i8mm = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sme = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &size, nullptr, 0) == 0) {
        runtime_feat.has_sme2 = static_cast<bool>(oldp);
    }

    // Apple does not support userspace non-streaming SVE; keep SVE vector length unknown.
    runtime_feat.sve_cnt = 0;
#elif defined (_WIN32)
    runtime_feat.has_dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_fp16    = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sve     = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sve2    = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_i8mm    = IsProcessorFeaturePresent(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sme     = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_AVAILABLE) != 0;
    runtime_feat.has_sme2    = IsProcessorFeaturePresent(PF_ARM_SME2_INSTRUCTIONS_AVAILABLE) != 0;

    // Windows exposes SVE feature presence, but not the runtime SVE vector length here.
    runtime_feat.sve_cnt = 0;
#endif

    return runtime_feat;
}

#endif // defined(__aarch64__) || defined(_M_ARM64)

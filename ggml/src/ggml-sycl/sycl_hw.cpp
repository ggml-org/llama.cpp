#include "sycl_hw.hpp"

using namespace std;

struct sycl_arch_entry {
    const char* name;
    sycl_intel_gpu_family gpu_family;
    sycl_xe_family xe_family;
};

/*defined in
* /opt/intel/oneapi/compiler/latest/include/sycl/ext/oneapi/experimental/device_architecture.def
*/
static map<gpu_arch, sycl_arch_entry> arch2name = {
    {gpu_arch::intel_gpu_bdw,     {"intel_gpu_bdw",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_skl,     {"intel_gpu_skl",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_kbl,     {"intel_gpu_kbl",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_cfl,     {"intel_gpu_cfl",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_apl,     {"intel_gpu_apl",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_glk,     {"intel_gpu_glk",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_whl,     {"intel_gpu_whl",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_aml,     {"intel_gpu_aml",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_cml,     {"intel_gpu_cml",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_icllp,   {"intel_gpu_icllp",   GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_ehl,     {"intel_gpu_ehl",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_PRE_XE}},
    {gpu_arch::intel_gpu_tgllp,   {"intel_gpu_tgllp",   GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_XE_LP}},
    {gpu_arch::intel_gpu_rkl,     {"intel_gpu_rkl",     GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_XE_LP}},
    {gpu_arch::intel_gpu_adl_s,   {"intel_gpu_adl_s",   GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_XE_LP}},
    {gpu_arch::intel_gpu_adl_p,   {"intel_gpu_adl_p",   GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_XE_LP}},
    {gpu_arch::intel_gpu_adl_n,   {"intel_gpu_adl_n",   GPU_FAMILY_IGPU_NON_XE,      XE_FAMILY_XE_LP}},
    {gpu_arch::intel_gpu_dg1,     {"intel_gpu_dg1",     GPU_FAMILY_DGPU_CLIENT_GAME, XE_FAMILY_XE_LP}},
    {gpu_arch::intel_gpu_acm_g10, {"intel_gpu_acm_g10", GPU_FAMILY_DGPU_CLIENT_GAME, XE_FAMILY_XE_HPG}},
    {gpu_arch::intel_gpu_acm_g11, {"intel_gpu_acm_g11", GPU_FAMILY_DGPU_CLIENT_GAME, XE_FAMILY_XE_HPG}},
    {gpu_arch::intel_gpu_acm_g12, {"intel_gpu_acm_g12", GPU_FAMILY_DGPU_CLIENT_GAME, XE_FAMILY_XE_HPG}},
    {gpu_arch::intel_gpu_pvc,     {"intel_gpu_pvc",     GPU_FAMILY_DGPU_CLOUD,       XE_FAMILY_XE_HPC}},
    {gpu_arch::intel_gpu_pvc_vg,  {"intel_gpu_pvc_vg",  GPU_FAMILY_DGPU_CLOUD,       XE_FAMILY_XE_HPC}},
    {gpu_arch::intel_gpu_mtl_u,   {"intel_gpu_mtl_u",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE_LPG}},
    {gpu_arch::intel_gpu_mtl_h,   {"intel_gpu_mtl_h",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE_LPG}},
    {gpu_arch::intel_gpu_arl_h,   {"intel_gpu_arl_h",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE_LPGPLUS}},
    {gpu_arch::intel_gpu_bmg_g21, {"intel_gpu_bmg_g21", GPU_FAMILY_DGPU_CLIENT_GAME, XE_FAMILY_XE2_HPG}},
    {gpu_arch::intel_gpu_bmg_g31, {"intel_gpu_bmg_g31", GPU_FAMILY_DGPU_CLIENT_GAME, XE_FAMILY_XE2_HPG}},
    {gpu_arch::intel_gpu_lnl_m,   {"intel_gpu_lnl_m",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE2_LPG}},
    {gpu_arch::intel_gpu_ptl_h,   {"intel_gpu_ptl_h",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE3_LPG}},
    {gpu_arch::intel_gpu_ptl_u,   {"intel_gpu_ptl_u",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE3_LPG}},
    {gpu_arch::intel_gpu_wcl,     {"intel_gpu_wcl",     GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE3_LPG}},
    {gpu_arch::intel_gpu_nvl_s,   {"intel_gpu_nvl_s",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE3_LPG}},
    {gpu_arch::intel_gpu_nvl_u,   {"intel_gpu_nvl_u",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE3_LPG}},
    {gpu_arch::intel_gpu_nvl_p,   {"intel_gpu_nvl_p",   GPU_FAMILY_IGPU_XE,          XE_FAMILY_XE3P_LPG}}
};

/* indexed by sycl_xe_family, family grouping follows `ocloc ids <family>` */
static const sycl_xe_family_caps xe_family_caps[XE_FAMILY_COUNT] = {
    /* name          simd  dpas  2d_io */
    {"unknown",      0,    0,    false}, // XE_FAMILY_UNKNOWN
    {"pre-xe",       8,    0,    false}, // XE_FAMILY_PRE_XE
    {"xe-lp",        8,    0,    false}, // XE_FAMILY_XE_LP
    {"xe-lpg",       8,    0,    false}, // XE_FAMILY_XE_LPG
    {"xe-lpgplus",   8,    8,    false}, // XE_FAMILY_XE_LPGPLUS
    {"xe-hpg",       8,    8,    false}, // XE_FAMILY_XE_HPG
    {"xe-hpc",       16,   16,   true},  // XE_FAMILY_XE_HPC
    {"xe2-lpg",      16,   16,   true},  // XE_FAMILY_XE2_LPG
    {"xe2-hpg",      16,   16,   true},  // XE_FAMILY_XE2_HPG
    {"xe3-lpg",      16,   16,   true},  // XE_FAMILY_XE3_LPG
    {"xe3p-lpg",     16,   16,   true},  // XE_FAMILY_XE3P_LPG
    {"xe3p-xpc",     16,   16,   true},  // XE_FAMILY_XE3P_XPC
};

const sycl_xe_family_caps & get_xe_family_caps(sycl_xe_family family) {
    if (family < XE_FAMILY_UNKNOWN || family >= XE_FAMILY_COUNT) {
        return xe_family_caps[XE_FAMILY_UNKNOWN];
    }
    return xe_family_caps[family];
}

bool is_xe_family_compiled(sycl_xe_family family) {
#ifndef GGML_SYCL_XE_FAMILY_AOT
    (void) family;
    return true;
#else
    switch (family) {
#ifdef GGML_SYCL_XE_FAMILY_XE_LP
        case XE_FAMILY_XE_LP:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE_LPG
        case XE_FAMILY_XE_LPG:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE_LPGPLUS
        case XE_FAMILY_XE_LPGPLUS:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE_HPG
        case XE_FAMILY_XE_HPG:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE_HPC
        case XE_FAMILY_XE_HPC:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE2_LPG
        case XE_FAMILY_XE2_LPG:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE2_HPG
        case XE_FAMILY_XE2_HPG:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE3_LPG
        case XE_FAMILY_XE3_LPG:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE3P_LPG
        case XE_FAMILY_XE3P_LPG:
#endif
#ifdef GGML_SYCL_XE_FAMILY_XE3P_XPC
        case XE_FAMILY_XE3P_XPC:
#endif
            return true;
        default:
            return false;
    }
#endif
}

sycl_hw_info get_device_hw_info(sycl::device* device_ptr) {
    sycl_hw_info res;
    int32_t id =
        device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    res.device_id = id;

    res.name = device_ptr->get_info<sycl::info::device::name>();

    syclex::architecture arch =
        device_ptr->get_info<syclex::info::device::architecture>();
    res.arch = arch;

    map<syclex::architecture, sycl_arch_entry>::iterator it =
        arch2name.find(res.arch);
    if (it != arch2name.end()) {
        res.arch_name = it->second.name;
        res.gpu_family = it->second.gpu_family;
        res.xe_family = it->second.xe_family;
    } else {
        res.arch_name = "unknown";
        res.gpu_family = GPU_FAMILY_UKNOWN;
        res.xe_family = XE_FAMILY_UNKNOWN;
    }

    return res;
}

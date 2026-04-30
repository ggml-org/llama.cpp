#include "sycl_hw.hpp"

using namespace std;

/*defined in
* /opt/intel/oneapi/compiler/latest/include/sycl/ext/oneapi/experimental/device_architecture.def
*/
static map<gpu_arch, std::pair<const char*, sycl_intel_gpu_family>> arch2name = {
    {gpu_arch::intel_gpu_bdw,     {"intel_gpu_bdw",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_skl,     {"intel_gpu_skl",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_kbl,     {"intel_gpu_kbl",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_cfl,     {"intel_gpu_cfl",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_apl,     {"intel_gpu_apl",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_glk,     {"intel_gpu_glk",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_whl,     {"intel_gpu_whl",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_aml,     {"intel_gpu_aml",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_cml,     {"intel_gpu_cml",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_icllp,   {"intel_gpu_icllp",   GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_ehl,     {"intel_gpu_ehl",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_tgllp,   {"intel_gpu_tgllp",   GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_rkl,     {"intel_gpu_rkl",     GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_adl_s,   {"intel_gpu_adl_s",   GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_adl_p,   {"intel_gpu_adl_p",   GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_adl_n,   {"intel_gpu_adl_n",   GPU_FAMILY_IGPU_NON_XE}},
    {gpu_arch::intel_gpu_dg1,     {"intel_gpu_dg1",     GPU_FAMILY_DGPU_CLIENT_GAME}},
    {gpu_arch::intel_gpu_acm_g10, {"intel_gpu_acm_g10", GPU_FAMILY_DGPU_CLIENT_GAME}},
    {gpu_arch::intel_gpu_acm_g11, {"intel_gpu_acm_g11", GPU_FAMILY_DGPU_CLIENT_GAME}},
    {gpu_arch::intel_gpu_acm_g12, {"intel_gpu_acm_g12", GPU_FAMILY_DGPU_CLIENT_GAME}},
    {gpu_arch::intel_gpu_pvc,     {"intel_gpu_pvc",     GPU_FAMILY_DGPU_CLOUD}},
    {gpu_arch::intel_gpu_pvc_vg,  {"intel_gpu_pvc_vg",  GPU_FAMILY_DGPU_CLOUD}},
    {gpu_arch::intel_gpu_mtl_u,   {"intel_gpu_mtl_u",   GPU_FAMILY_IGPU_XE}},
    {gpu_arch::intel_gpu_mtl_h,   {"intel_gpu_mtl_h",   GPU_FAMILY_IGPU_XE}},
    {gpu_arch::intel_gpu_arl_h,   {"intel_gpu_arl_h",   GPU_FAMILY_IGPU_XE}},
    {gpu_arch::intel_gpu_bmg_g21, {"intel_gpu_bmg_g21", GPU_FAMILY_DGPU_CLIENT_GAME}},
    {gpu_arch::intel_gpu_bmg_g31, {"intel_gpu_bmg_g31", GPU_FAMILY_DGPU_CLIENT_GAME}},
    {gpu_arch::intel_gpu_lnl_m,   {"intel_gpu_lnl_m",   GPU_FAMILY_IGPU_XE}},
    {gpu_arch::intel_gpu_ptl_h,   {"intel_gpu_ptl_h",   GPU_FAMILY_IGPU_XE}},
    {gpu_arch::intel_gpu_ptl_u,   {"intel_gpu_ptl_u",   GPU_FAMILY_IGPU_XE}},
    {gpu_arch::intel_gpu_wcl,     {"intel_gpu_wcl",     GPU_FAMILY_IGPU_XE}}
};


sycl_hw_info get_device_hw_info(sycl::device* device_ptr) {
    sycl_hw_info res;
    int32_t id =
        device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    res.device_id = id;

    res.name = device_ptr->get_info<sycl::info::device::name>();

    syclex::architecture arch =
        device_ptr->get_info<syclex::info::device::architecture>();
    res.arch = arch;

    map<syclex::architecture,
        std::pair<const char*, sycl_intel_gpu_family>>::iterator it =
        arch2name.find(res.arch);
    if (it != arch2name.end()) {
        res.arch_name = it->second.first;
        res.gpu_family = it->second.second;
    } else {
        res.arch_name = "unknown";
        res.gpu_family = GPU_FAMILY_UKNOWN;
    }

    return res;
}

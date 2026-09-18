// SPDX-License-Identifier: MIT
#pragma once
#include <sycl/sycl.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
namespace maple_w2 {
// The successful v0.2 run still queried device strings at every enqueue.
// Cache *validated device identities*, not a global "GPU present" flag.
// MAPLE_W2_DEVICE_CHECK_EVERY_CALL=1 is a diagnostic A/B switch, not a safety bypass.
inline void validate_dg2_device(const sycl::device& d) {
    static const bool every_call=[] { const char* p=std::getenv("MAPLE_W2_DEVICE_CHECK_EVERY_CALL"); return p && std::string(p)=="1"; }();
    thread_local std::vector<sycl::device> validated;
    if(!every_call) for(const auto& v:validated) if(v==d) return;
    const auto name=d.get_info<sycl::info::device::name>();
    if(!d.is_gpu() || d.get_info<sycl::info::device::vendor_id>()!=0x8086 || !d.has(sycl::aspect::fp16) ||
       (name.find("A750")==std::string::npos && name.find("A770")==std::string::npos))
        throw std::invalid_argument("requires Intel Arc A750/A770, DG2 ExecSize8");
    if(!every_call) validated.push_back(d);
}
} // namespace maple_w2

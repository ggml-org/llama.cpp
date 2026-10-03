#ifndef SYCL_HW_HPP
#define SYCL_HW_HPP

#include <algorithm>
#include <stdio.h>
#include <vector>
#include <map>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
using gpu_arch = sycl::ext::oneapi::experimental::architecture;

// It's used to mark the GPU computing capacity
// The value must flow the order of performance.
enum sycl_intel_gpu_family {
  GPU_FAMILY_UKNOWN = -1,
  // iGPU without Xe core, before Meteor Lake iGPU(Xe)
  GPU_FAMILY_IGPU_NON_XE = 0,
  // iGPU with Xe core, Meteor Lake iGPU or newer.
  GPU_FAMILY_IGPU_XE = 1,
  // dGPU for gaming in client/data center (DG1/FLex 140 or newer).
  GPU_FAMILY_DGPU_CLIENT_GAME = 2,
  // dGPU for AI in cloud, PVC or newer.
  GPU_FAMILY_DGPU_CLOUD = 3
};

// Intel GPU IP family, named after the ocloc family targets.
enum sycl_xe_family {
  XE_FAMILY_UNKNOWN = 0,
  XE_FAMILY_PRE_XE,     // Gen8 - Gen11
  XE_FAMILY_XE_LP,      // TGL, RKL, ADL, DG1
  XE_FAMILY_XE_LPG,     // MTL, ARL-U/S
  XE_FAMILY_XE_LPGPLUS, // ARL-H
  XE_FAMILY_XE_HPG,     // DG2
  XE_FAMILY_XE_HPC,     // PVC
  XE_FAMILY_XE2_LPG,    // LNL
  XE_FAMILY_XE2_HPG,    // BMG
  XE_FAMILY_XE3_LPG,    // PTL, WCL, NVL-S/U
  XE_FAMILY_XE3P_LPG,   // NVL-P
  XE_FAMILY_XE3P_XPC,   // CRI, no SYCL arch enum yet
  XE_FAMILY_COUNT
};

// Static hardware traits of one Xe family. 0 means unknown or not present.
struct sycl_xe_family_caps {
  const char* name;  // ocloc family name
  int simd_width;    // native EU SIMD width
  int dpas_n;        // DPAS execution size (XMX), 0 = no XMX
  bool block_2d_io;  // 2D block load/store
};

struct sycl_hw_info {
  syclex::architecture arch;
  const char* arch_name;
  int32_t device_id;
  std::string name;
  sycl_intel_gpu_family gpu_family;
  sycl_xe_family xe_family;
};

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);

// Look up the static traits of an Xe family.
const sycl_xe_family_caps & get_xe_family_caps(sycl_xe_family family);

#endif // SYCL_HW_HPP

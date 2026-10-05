# Does Mesa Affect Intel Arc A770 Compute Performance in llama.cpp on Arch Linux?

**Verdict:** Mesa only matters if you run llama.cpp's **Vulkan** backend. That backend runs your A770 through Mesa's ANV driver (the `vulkan-intel` package), so the Mesa version can change pp and tg a lot, in either direction. On the **SYCL** path Mesa is not loaded on the compute path at all. There the stack that matters is `level-zero-loader` → `intel-compute-runtime` (NEO) → `intel-graphics-compiler` (IGC), plus the DPC++/oneDNN/oneMKL versions you build with, the kernel driver (i915 vs xe), firmware, and PCIe/ReBAR/power settings. The desktop running on the RDNA2 iGPU through RADV/radeonsi does not affect Arc compute, except that a Vulkan build will see both GPUs, so you have to pin the device.

## TL;DR

- **SYCL (your main backend):** Mesa has no effect. The packages that matter are `intel-compute-runtime` + `intel-graphics-compiler` at runtime and `intel-oneapi-dpcpp-cpp`/oneDNN/oneMKL at build time. llama.cpp's own docs report that a newer oneAPI release (2026.1) gave 3.1× prompt processing over the 2025.3-based build on an Arc B570, which shows that compiler version is a first-order variable.
- **Vulkan:** Mesa (ANV) is the driver, so its version is the single biggest package variable. On an A770, Mesa 24.0.9 → 24.3.1 cut tg128 from 54.5 to 37.3 t/s while raising pp. llama.cpp also deliberately leaves cooperative matrix off for DG2. Mesa 26.1's large Vulkan speedups were measured on Xe2 (Battlemage), not Alchemist, so don't assume they carry over to your card.
- **System level:** ReBAR is effectively mandatory (compute-runtime on xe rejects small-BAR DG2). ASPM L1 and the choice of i915 vs xe set idle power. Monitoring tools can hold the GPU awake. Pin the device in every backend so the RADV iGPU never joins a Vulkan split.

## Key Findings

### 1. Which userspace driver is actually loaded, by backend

| Backend | Loader | Userspace driver on the A770 | JIT/shader compiler | Does Mesa matter? |
|---|---|---|---|---|
| SYCL (DPC++ → Unified Runtime → Level Zero) | `level-zero-loader` (`libze_loader.so`) | `intel-compute-runtime` (`libze_intel_gpu.so`) | IGC compiles SPIR-V → Xe ISA at first run, or AOT if you build with `GGML_SYCL_DEVICE_ARCH` | **No** |
| Vulkan | `vulkan-icd-loader` | Mesa ANV (`vulkan-intel`, `libvulkan_intel.so`) | Mesa's own Intel backend compiler (brw/NIR); SPIR-V produced at build time by `glslc` (shaderc) | **Yes, directly** |
| OpenCL (llama.cpp OpenCL backend) | `ocl-icd` | `intel-compute-runtime` (NEO OpenCL ICD) **or** Mesa rusticl (on top of iris), if installed | IGC (NEO) or Mesa (rusticl) | Only if rusticl is the ICD selected |

The mechanism behind the SYCL row: the SYCL runtime never touches Mesa. Even when SYCL lists `opencl:gpu` devices, those also come from NEO. llama.cpp's SYCL docs state that the backend "always runs on Level Zero running time"\[1\] and recommend `ONEAPI_DEVICE_SELECTOR="level_zero:0"`.\[2\] llama.cpp's OpenCL backend is tuned mainly for Qualcomm Adreno. On Arc it is a curiosity, not a performance path. If you install rusticl, `ocl-icd` will expose more platforms (and could expose the RADV/radeonsi iGPU). That is a device-selection hazard more than a speed lever.

### 2. Ranked package inventory

Impact key: **H** = shown to move t/s by tens of percent or more; **M** = plausible or occasionally shown; **L** = indirect, stability, or startup only; **—** = not on the path. Rows marked † are inferred, not benchmarked.

| Rank | Arch package(s) | SYCL | Vulkan | OpenCL | Notes |
|---|---|---|---|---|---|
| 1 | `intel-oneapi-dpcpp-cpp` / `intel-oneapi-base-toolkit` (DPC++/icpx, oneDNN `intel-oneapi-dnnl`, `intel-oneapi-mkl-sycl`) | **H** (build) | — | — | The compiler generates your kernels. oneAPI 2026.1 gave 3.1× pp on B570 vs 2025.3. oneDNN is the default GEMM for pp.\[1\] |
| 2 | `intel-compute-runtime` (NEO) | **H** | — | **H** (NEO ICD) | The matching upgrade gave SYCL +28% pp2048 / +4% tg128 on a B70 (Jonathan Mann's benchmark write-up). Historically regressions coincided with kernel changes. |
| 3 | `intel-graphics-compiler` (IGC) | **H** | — | **H** | JIT backend for every SYCL kernel. Must match the compute-runtime release. |
| 4 | `mesa` / `vulkan-intel` (ANV) | — | **H** | L† (rusticl) | Same pkgbase, so versions move in lockstep. Changes driver code, the shader compiler, and the feature exposure llama.cpp keys off. |
| 5 | `shaderc` (`glslc`), `spirv-headers`, `spirv-tools`, `vulkan-headers` | — | **H** (build) | — | A glslc too old for `GL_EXT_integer_dot_product` silently compiles out the MMVQ path, making tg 5–10× slower (measured on Xe2).\[3\] |
| 6 | `linux` (kernel), i915 vs xe | M | M | M | Sets memory management, scheduling, GuC SLPC frequency policy and runtime PM. Linux 7.1 gave no Vulkan llama.cpp gain on B580.\[4\] |
| 7 | `linux-firmware-intel` (DG2 GuC/HuC/GSC) | M† | M† | M† | GuC SLPC chooses the GT frequency under both drivers. Mostly a stability factor.\[5\] |
| 8 | `level-zero-loader`, `level-zero-headers` | L | — | — | Thin dispatch layer. A mismatch breaks device discovery, not speed. |
| 9 | `vulkan-icd-loader` | — | L | — | Device enumeration and selection only. |
| 10 | `ocl-icd` | L | — | L | ICD dispatch. Matters for which platforms appear. |
| 11 | `intel-gmmlib` | L† | — | L† | NEO memory-layout library, pulled in as a dependency. |
| 12 | `intel-gpu-tools`, `xpu-smi`/`xpumanager` (AUR), `nvtop` | L (observer) | L | L | Monitoring tools can keep the GT out of RC6 (see §4).\[6\] |
| — | AUR: `intel-compute-runtime-bin`, `intel-graphics-compiler-git`, `intel-oneapi-basekit-2025`, `mesa-git` | as parent | as parent | as parent | Useful for bisecting. The `-bin` runtime has lagged the repo version.\[7\] |

### 3. Version-specific evidence

**Mesa ANV / Vulkan on A770**
- **Cooperative matrix is off on DG2 by design.** ANV has exposed `VK_KHR_cooperative_matrix` on DG2 since Mesa 24.0, using DPAS.\[8\] However, Intel contributor rillomas measured A770 pp512 at 260.74 t/s with coopmat forced on vs 961.3 t/s without (27%) in llama.cpp Discussion #13530 (2025-05-28, build b5200, i9-12900K + Arc A770), adding that "The performance regression issue you mentioned in #12690 with Arc A770 is acknowledged but we don't have an ETA for the fix yet." PR #14001 (approved 2025-06-05) therefore enabled coopmat for Intel **Xe2 only**.\[9\]\[10\] The 2026 code reads `return arch == vk_device_architecture::INTEL_XE2;`, and a device counts as Xe2 only when `minSubgroupSize == 16`.\[11\]\[12\] DG2 reports 8, so your log shows `matrix cores: none`.\[13\] A llama.cpp maintainer said that "Intel engineers have also looked at this and decided to leave Alchemist behind."\[14\] The practical consequence: Mesa coopmat work (including Mesa 26.1's NV-coopmat2-related ANV additions) does not reach the A770's matmul path unless you patch that gate.
- **Mesa can regress tg on DG2.** In PR #10721 (2024-12), A770 / llama 7B Q4_0 ran pp512 142.42 / tg128 54.51 t/s on Mesa 24.0.9. On Mesa 24.3.1 it ran pp512 373.20 / tg128 37.26 t/s, a ~32% tg loss alongside a 2.6× pp gain.\[15\] Mesa git 25.0 threw `vk::DeviceLostError`.\[15\] On 2025-01-11 llama.cpp Vulkan maintainer 0cc4m wrote in Discussion #10879: "I suspect the mesa version, there was something in newer mesa versions that slowed down tg on Intel." That is old data. Treat it as evidence that the Mesa version is a real variable on DG2, not as a description of Mesa 26.x behavior.
- **Mesa 26.1 (2026-05-06) gives large gains on Xe2. They are unverified on DG2.** Phoronix ("Linux 7.1 Helping Intel Arc Battlemage Graphics Achieve Better Performance", 2026-06-08, B580) found "significant speed-ups to find with Intel ANV for Llama.cpp on Mesa 26.1, especially around the text generation speed": in its Vulkan tg128 tests on Llama-3.1-Tulu-3-8B, Mistral-7B, gpt-oss-20b and DeepSeek-R1-Distill-Llama-8B (all Q8_0), "Linux 7.1 Git + Mesa 26.1 was the fastest," while the kernel alone gave no gain. Jonathan Mann's B70 write-up ("Don't trust your old benchmarks", Manjaro 26.0.4) reports that "Vulkan decode doubled from 37.8 to 76 t/s after the Mesa 26.1 upgrade." He later noted that the `matrix cores:` label is not a reliable indicator of the fast path.\[16\] Both results are Xe2, where coopmat is enabled. I found no published DG2 A/B for Mesa 26.0 → 26.1/26.2. *Inference:* gains in the scalar and integer-dot shader paths from compiler improvements are plausible on DG2, but the coopmat-driven part should not carry over.
- **Mesa 26.2.x.** Mesa 26.2.0 shipped 2026-08-05.\[17\]\[18\] Intel's 2026 Xe flash-attention Vulkan work (llama.cpp PR #24406) includes a fix for "test op failure on A770 Linux with 26.2.3 mesa driver."\[19\] That shows active A770+ANV interaction in 2026, but it is a correctness fix, not a benchmark.
- **Stability.** Issue #29526 reports an A770 on Vulkan with Mesa 25.2.8 at tg ≈ 26 t/s that degrades to empty replies and fence timeouts after ~7–8 h of uptime.\[20\]

**compute-runtime / IGC / oneAPI for SYCL**
- llama.cpp's SYCL docs: "oneAPI 2026.1 improves the SYCL build performance: measured with the same code on Arc B570, prompt processing 1331 vs 434 t/s (3.1x) vs the 2025.3-based release build."\[1\] This is Xe2, but it shows that DPC++/oneDNN version can be worth multiples. Verified oneAPI releases in the docs: 2025.3.3, 2025.2.1, 2025.1, 2024.1. llama.cpp's own 2025-02 Q4_0 reorder work took the A770 from 42 to 55 t/s (+30%).\[1\]\[21\] That came from code, not packages.
- A B70 report (issue #21517) found the same results across compute-runtime 26.05.37020.3/IGC 2.28.4 and 26.09.37435.1/IGC 2.30.1. Toggling `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0/1` had no effect on tg.\[22\] So runtime upgrades are not always a lever. When kernel code is the bottleneck, they do little.\[22\]
- Historical (2024, likely outdated): Linux 6.8 combined with older compute-runtime broke or slowed A770 Level Zero (compute-runtime issues #710/#726). One user noted the fix "lowered performance a lot."\[2\]
- Long-run stability: issue #29527 reports an A770 SYCL/Level Zero fence deadlock after ~18 h with oneAPI 2026.1.\[23\]

**Arch packaging state (verify locally with `pacman -Qi`)**
- `intel-compute-runtime` 26.31.39395.13-1 (built 2026-08-22).\[24\] Upstream built 26.31 against IGC v2.40.13.\[25\] Arch's `intel-graphics-compiler` package page showed 1:2.38.2-1 (built 2026-07-26),\[26\] which is the IGC that upstream paired with runtime 26.27. Arch's security tracker shows 1:2.41.5-1.\[27\] The sources conflict, so check what you actually have installed. An IGC older than the runtime's validated pairing is a known way to get JIT failures or odd performance.
- `intel-oneapi-base-toolkit` 2025.3.2 (the metapackage) vs split `intel-oneapi-dpcpp-cpp` 2026.0.0_947-1. Both are in extra, and the split packages conflict with the basekit.\[28\]\[29\] Your DPC++ version therefore depends on which set you installed. Know it before comparing numbers.
- Upstream compute-runtime lists Alchemist as "Production" on Level Zero 1.17, validated on Ubuntu 26.04 with kernel 7.0.\[25\]

### 4. System-level factors

- **Resizable BAR / Above 4G decoding:** required. With the xe driver, compute-runtime silently reported 0 platforms on an A770 with a 256 MB BAR, and enabling ReBAR in the BIOS fixed it (compute-runtime issue #905).\[30\] i915 tolerates small BAR, but host↔VRAM transfers (model load, any CPU-offload layers) suffer. Check with `lspci -vv -s <BDF>` for `Region 2 ... [size=16G]`.
- **i915 vs xe for DG2:** i915 is the default. xe requires `i915.force_probe=!56a0 xe.force_probe=56a0`, and Mesa prints "Support for this platform is experimental with Xe KMD."\[31\]\[32\] xe gets the A770 into runtime `suspended` (~1 W idle headless), whereas i915 reports runtime PM `unsupported`.\[30\] One SYCL dual-A770 report found xe plus IOMMU made `--split-mode tensor` viable.\[31\] In llama.cpp Discussion #23313, user noise-field (2026-08-17, two GPUs at PCIe Gen4 x8/x8, commit 4df29be) concluded that "Xe driver gives a huge boost over i915, with decode rivaling Vulkan and a much faster prefill," but I could not confirm this was a single A770, so treat it as unverified for your card. The case for xe is idle power, not speed.
- **ASPM / PCIe link:** Intel Support article 000092564 ("High Power Consumption when Intel® Arc™ Graphics Card is Idle") states that "For Intel® Arc™ Graphics, ASPM L1 is required for all power states greater than G2". Without it, an A770 idles at ~40 W. Enable PEG ASPM L1 in the BIOS\[33\] and check `LnkCtl` and `LnkSta` (should be 16 GT/s x16) with `lspci -vv`. ASPM does not reduce sustained throughput. A downtrained link (x8/x4, Gen3) mainly hurts loading and hybrid offload, not full-GPU tg.
- **GPU frequency:** GuC SLPC picks frequencies between the min/max requests. On xe these are under `/sys/class/drm/cardN/device/tile0/gt0/freq0/{min_freq,max_freq,act_freq,rp0_freq}`, plus `throttle/`.\[5\] On i915 they are `/sys/class/drm/cardN/gt/gt0/rps_{min,max,boost}_freq_mhz`. Intel's oneAPI GPU optimization guide recommends pinning min=max=RP0 for compute benchmarking.\[34\] For decode, which issues many small kernels, raising `min_freq` is the single most plausible way to reduce variance. That is inferred; benchmark it.
- **Monitoring perturbs the GPU:** `intel_gpu_top` doesn't support xe devices (use `gputop`).\[35\] nvtop PR #519 by stolk (2026-09-30, "intel: read the xe GT frequency from sysfs, not a perf event") found that nvtop keeps the xe frequency perf event open for its whole run, holding forcewake, so the GT "never enters RC6", costing "4–36 W extra per card." Close monitors during idle-power tests and keep them identical across A/B runs.
- **AMD iGPU as primary:** this is harmless for SYCL, which only sees Intel Level Zero devices. On Vulkan, llama.cpp enumerates both RADV and ANV and may split layers across them. A llama.cpp maintainer diagnosed exactly this pattern (dGPU plus iGPU) as a slowdown on an A770 system in 2026-08.\[36\] Pin the device.
- **Build linkage:** one #23313 participant found that `-DBUILD_SHARED_LIBS=0` reduced A770 SYCL performance.\[31\] This is anecdotal.

### 5. Environment variables and tunables

| Variable / flag | Backend | Effect | Measured perf impact? |
|---|---|---|---|
| `ONEAPI_DEVICE_SELECTOR=level_zero:0` | SYCL | Restricts SYCL to the A770 via Level Zero | Prevents bad splits |
| `--device SYCL0` / `-sm none -mg 0` | SYCL | Single-device run | Yes, when an iGPU would otherwise join\[36\] |
| `GGML_SYCL_F16=ON` (build) | SYCL | FP16 kernels. Docs: "recommended for better performance in most cases"\[1\] | Yes; model-dependent |
| `GGML_SYCL_DEVICE_ARCH=acm-g10` (build) | SYCL | AOT compiles for DG2 and skips the ~20 s JIT at startup\[37\] | Startup; codegen may differ from JIT (test it) |
| `GGML_SYCL_GRAPH` (build, default ON) / `GGML_SYCL_DISABLE_GRAPH` (runtime, default 1) | SYCL | SYCL command graphs replay decode kernels. Off by default: "no better performance" yet\[38\] | Workload-dependent; relevant to decode launch overhead |
| `GGML_SYCL_DISABLE_OPT` (being renamed `GGML_SYCL_ENABLE_OPT`) | SYCL | Toggles weight-reorder optimization (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0)\[39\]\[40\] | Yes (the reorder gave A770 +30% on Q4_0); also a correctness workaround |
| `GGML_SYCL_DISABLE_DNN`, `GGML_SYCL_PRIORITIZE_DMMV`, `GGML_SYCL_USE_LEVEL_ZERO_API` | SYCL | GEMM library, mat-vec kernel choice, allocation API\[1\]\[41\] | Can change pp/tg; benchmark |
| `ZES_ENABLE_SYSMAN=1` | SYCL | Enables free-memory query (for `-sm layer`)\[38\] | No throughput effect |
| `SYCL_CACHE_PERSISTENT=1` | SYCL | Persistent JIT kernel cache | Startup only; IPEX-LLM docs blame it for a "program was built for 1 devices" error\[42\] |
| `UR_L0_USE_IMMEDIATE_COMMANDLISTS` / legacy `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | SYCL | Immediate vs batched command lists | No tg effect in #21517 (B70); worth a DG2 A/B\[22\] |
| NEO debug keys (`NEOReadDebugKeys=1` + keys) | SYCL/OpenCL | Driver internals | Debug only; not a tuning interface |
| `GGML_VK_VISIBLE_DEVICES=<idx>` / `--device VulkanN` | Vulkan | Hides the RADV iGPU from ggml\[43\] | Prevents splits |
| `VK_DRIVER_FILES` (formerly `VK_ICD_FILENAMES`)`=/usr/share/vulkan/icd.d/intel_icd.x86_64.json` | Vulkan | Loads only ANV for the llama.cpp process | Hygiene; KDE/Wayland unaffected (per-process) |
| `MESA_VK_DEVICE_SELECT=8086:56a0` | Vulkan | Mesa device-select layer reorders devices | Ordering only |
| `GGML_VK_DISABLE_COOPMAT` / `GGML_VK_DISABLE_COOPMAT2` / `GGML_VK_DISABLE_MMVQ` | Vulkan | Kernel-path switches | Mostly no-ops on DG2 (coopmat already off); MMVQ toggle is diagnostic\[13\]\[44\] |
| `-ub`/`--ubatch-size` | both | Physical batch size | On a B70, 512 → 2048 raised pp2048 from 1,131 to 1,824 t/s with no tg cost\[16\] |

## Recommendations

1. **For your SYCL fork, pin and record the triple, not Mesa:** `intel-oneapi-dpcpp-cpp` (+dnnl/mkl), `intel-compute-runtime`, `intel-graphics-compiler`, plus kernel and `linux-firmware-intel`. Add them to `IgnorePkg` during a tuning campaign, and log `sycl-ls` output (it prints the NEO version) with every llama-bench result.
2. **Treat DPC++ upgrades as the biggest SYCL A/B.** Rebuild the same commit under two oneAPI versions before attributing a change to your kernels. Try both JIT and `GGML_SYCL_DEVICE_ARCH=acm-g10` AOT builds.
3. **Keep IGC matched to compute-runtime.** Use the pairing in the upstream release notes (e.g. 26.31 ↔ IGC 2.40.13).\[25\] If Arch's IGC lags, take both from the Arch Linux Archive at a matching date, or build both from AUR.
4. **Run a Vulkan control build** with current `shaderc`. Mesa 26.1/26.2 may have moved the Vulkan baseline on DG2 without anyone publishing it, and decode on Vulkan has been competitive with or faster than SYCL on A770 for MoE models (issue #19918: tg 68 vs 10 t/s on gpt-oss-20B).\[45\] Confirm `int dot: 1` in the ggml_vulkan banner.
5. **Stay on i915** for throughput work unless you need idle power or multi-GPU experiments. If you test xe, keep ReBAR on and compare against an identical package set.
6. **Pin GT min frequency** during benchmarks to cut variance, then re-test at defaults, because SLPC behavior is part of real-world decode.

## How to verify on Arch

**Confirm the loaded driver**
```bash
# SYCL / Level Zero: expect [level_zero:gpu] ... A770 ... [1.x.<NEO build>]
source /opt/intel/oneapi/setvars.sh; sycl-ls
./build/bin/llama-ls-sycl-device
# Vulkan: expect "Intel open-source Mesa driver", driverInfo = Mesa version
vulkaninfo --summary
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json vulkaninfo --summary
# OpenCL: which ICDs ocl-icd sees
ls /etc/OpenCL/vendors/; clinfo -l
# Kernel driver, BAR, link
lspci -k -vv -s $(lspci -D | awk '/56a0|A770/{print $1}') | grep -E 'driver|Region 2|LnkSta|LnkCtl'
pacman -Q mesa vulkan-intel intel-compute-runtime intel-graphics-compiler level-zero-loader intel-oneapi-dpcpp-cpp linux linux-firmware-intel shaderc
```

**Benchmark methodology**
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./llama-bench -m model.gguf -ngl 99 -sm none -mg 0 \
  -p 512,2048 -n 128 -fa 0,1 -r 10 -o md
GGML_VK_VISIBLE_DEVICES=<arc_idx> VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json \
  ./llama-bench -m model.gguf -ngl 99 -p 512,2048 -n 128 -fa 0,1 -r 10 -o md
```
- Report pp and tg separately, and use `-d` (depth) runs if long-context decode matters to you. Use ≥10 repetitions and a fixed GT min frequency or fixed defaults. Do one warm-up run so the JIT cache and shader cache are populated. Keep monitors closed or identical between runs.
- Use the community baseline model (llama-2-7b Q4_0) so you can compare against discussions #10879 (Vulkan) and #23313 (SYCL).

**Pinning or downgrading via the Arch Linux Archive**
```bash
# One package (mesa and vulkan-intel must move together; same pkgbase)
sudo pacman -U https://archive.archlinux.org/packages/m/mesa/mesa-<ver>-x86_64.pkg.tar.zst \
               https://archive.archlinux.org/packages/v/vulkan-intel/vulkan-intel-<ver>-x86_64.pkg.tar.zst
# Whole-repo snapshot for a matched NEO+IGC pair: point mirrorlist at
# Server=https://archive.archlinux.org/repos/YYYY/MM/DD/$repo/os/$arch  then pacman -Syyuu
# Freeze during testing: /etc/pacman.conf -> IgnorePkg = mesa vulkan-intel intel-compute-runtime intel-graphics-compiler
```

**Monitoring**
- i915: `intel_gpu_top -d drm:/dev/dri/renderD12X`. On xe: `gputop` from `intel-gpu-tools`, or `xpu-smi` (AUR `xpumanager`) for frequency, power and memory bandwidth.
- Read `act_freq` and `throttle/reasons` from sysfs during tg to catch power or thermal throttling.
- Check `cat /sys/bus/pci/devices/<BDF>/power/runtime_status` for idle behavior.

## Caveats

- **Confidence levels.** High confidence that Mesa is not on the SYCL/Level Zero path; that follows from the architecture, not from a benchmark. High confidence that Mesa matters for Vulkan; there are A770 measurements, but they are from 2024. Medium-to-low confidence on how Mesa 26.x performs on DG2 specifically; the published 2026 evidence is Xe2 (B580/B70).
- **Inferred, not benchmarked:** firmware impact, gmmlib impact, an i915 vs xe throughput difference on a single A770, gains from frequency pinning, and the claim that AOT vs JIT changes codegen.
- **Weaker sources.** Some evidence comes from individual user reports (blog posts, GitHub issues) on different hardware. Phoronix reported Mesa 26.1 gains only qualitatively on the page reviewed. A sourcing conflict remains on Arch's current IGC version (2.38.2 on the package page vs 2.41.5 on the security tracker).
- **Things change fast.** llama.cpp's Intel gating (`INTEL_XE2`) and env-var names (`GGML_SYCL_DISABLE_OPT` → `ENABLE_OPT`) are moving. Re-check against the commit you build.

## Sources

1. [llama.cpp/docs/backend/SYCL.md at master · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
2. [github.com](https://github.com/ggerganov/llama.cpp/issues/7042)
3. [GitHub - chaosmonkies/llamacpp-vulkan-intel-arc-slow-quant: Why llama.cpp Vulkan runs quantized models \~10x too slow on Intel Arc/Battlemage (old glslc disables the MMVQ integer-dot path) and how to fix it · GitHub](https://github.com/chaosmonkies/llamacpp-vulkan-intel-arc-slow-quant)
4. [Linux 7.1 Helping Intel Arc Battlemage Graphics Achieve Better Performance - Phoronix](https://www.phoronix.com/review/intel-b580-linux-71/3)
5. [Xe GT Frequency Management — The Linux Kernel documentation](https://docs.kernel.org/gpu/xe/xe_gt_freq.html)
6. [intel: read the xe GT frequency from sysfs, not a perf event by stolk · Pull Request #519 · Syllo/nvtop](https://github.com/Syllo/nvtop/pull/519)
7. [intel-compute-runtime package versions - Repology](https://repology.org/project/intel-compute-runtime/versions)
8. [Intel's Vulkan Linux Driver Now Exposes Cooperative Matrix Support - Phoronix](https://www.phoronix.com/news/Intel-ANV-Cooperative-Matrix)
9. [Vulkan: VK\_KHR\_cooperative\_matrix support to speed up prompt processing by 0cc4m · Pull Request #10597 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/pull/10597)
10. [PR #14001 vulkan: Enable VK\_KHR\_cooperative\_matrix extension for Intel Xe2 GPUs - SemanticDiff](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/14001/overview)
11. [Intel Arc 140T (Arrow Lake H, Xe2) not detected as INTEL\_XE2 — cooperative matrix disabled despite driver support · Issue #20776 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues/20776)
12. [Vulkan Backend (Cross-Platform)](https://deepwiki.com/ggml-org/llama.cpp/5.3-example-programs)
13. [Misc. bug: Intel UHD 770 graphics crashes on Vulkan TOP\_K unit test · Issue #26219 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues/26219)
14. [When will llama.cpp's vulkan provide support for Intel Arc's matrix core?](https://github.com/ggml-org/llama.cpp/issues/12690)
15. [Vulkan: Add VK\_EXT\_subgroup\_size\_control support to ensure full subgroups for coopmats by 0cc4m · Pull Request #10721 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/pull/10721)
16. [Don't trust your old benchmarks: a weekend of testing on the Intel Arc B70 — Jonathan Mann](https://jonathanmann.tech/blog/intel-arc-b70-llama-cpp-benchmarks/)
17. [Mesa 26.2.0 Release Notes / 2026-08-05 — The Mesa 3D Graphics Library latest documentation](https://docs.mesa3d.org/relnotes/26.2.0.html)
18. [Mesa (computer graphics)](<https://en.wikipedia.org/wiki/Mesa_(computer_graphics)>)
19. [\[pull\] master from ggml-org:master by pull\[bot\] · Pull Request #11 · firas9941/llama.cpp](https://github.com/firas9941/llama.cpp/pull/11)
20. [\[Bug\] Vulkan backend: A770 long-running decode degradation (empty EOS replies) after \~7-8h · Issue #29526 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues/29526)
21. [OpenTransformer's picture](https://huggingface.co/OpenTransformer/llama.cpp-prismml/blob/main/docs/backend/SYCL.md)
22. [\[SYCL\] Q8\_0 quantization \~4x slower than Q4\_K\_M on Intel Arc Pro B70 (Xe2/Battlemage) — kernel efficiency issue · Issue #21517 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues/21517)
23. [\[Bug\] SYCL/Level Zero backend: A770 long-running decode fence deadlock after \~18h · Issue #29527 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues/29527)
24. [Arch Linux - intel-compute-runtime 26.31.39395.13-1 (x86\_64)](https://archlinux.org/packages/extra/x86_64/intel-compute-runtime/)
25. [Releases · intel/compute-runtime](https://github.com/intel/compute-runtime/releases)
26. [Arch Linux - intel-graphics-compiler 1:2.38.2-1 (x86\_64)](https://archlinux.org/packages/extra/x86_64/intel-graphics-compiler/)
27. [intel-graphics-compiler - Arch Linux](https://security.archlinux.org/package/intel-graphics-compiler)
28. [Arch Linux - intel-oneapi-base-toolkit 2025.3.2.21+99f4837a\_25b7\_425d\_a897\_60af022676ea-1 (x86\_64)](https://archlinux.org/packages/extra/x86_64/intel-oneapi-base-toolkit/)
29. [Arch Linux - intel-oneapi-dpcpp-cpp 2026.0.0\_947-1 (x86\_64)](https://archlinux.org/packages/extra/x86_64/intel-oneapi-dpcpp-cpp/)
30. [\[GSD-12441\] DG2 (Arc A770) not detected by compute-runtime when using xe kernel driver](https://github.com/intel/compute-runtime/issues/905)
31. [Performance of llama.cpp on Intel GPU with SYCL backend · ggml-org/llama.cpp · Discussion #23313](https://github.com/ggml-org/llama.cpp/discussions/23313)
32. [Xe](https://www.kernel.org/doc/html/v6.8/gpu/rfc/xe.html)
33. [Question Intel ARC A770 Performance and idle power draw after Feb. 2023 driver update](https://forums.anandtech.com/threads/intel-arc-a770-performance-and-idle-power-draw-after-feb-2023-driver-update.2610507/post-40948829)
34. [Configuring GPU Device](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-1/configuring-gpu-device.html)
35. [intel\_gpu\_top(1) — Arch manual pages](https://man.archlinux.org/man/intel_gpu_top.1.en)
36. [Best Llama.cpp Backend for NUC 12 Extreme + Intel Arc A770 · ggml-org/llama.cpp · Discussion #26823](https://github.com/ggml-org/llama.cpp/discussions/26823)
37. [GitHub - alucryd/arc-llama: Plug-and-play llama.cpp runtime for Intel Arc GPUs. Auto-detects your card, picks safe SYCL defaults, and exposes an OpenAI-compatible API. · GitHub](https://github.com/alucryd/arc-llama)
38. [llama.cpp: docs/backend/SYCL.md](https://fossies.org/linux/llama.cpp/docs/backend/SYCL.md)
39. [sycl: fix check\_graph\_compatibility() to allow graphs for MoE decode (CONCAT dim!=3, MUL\_MAT\_ID fused path) by Captain-Tripps · Pull Request #25089 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/pull/25089)
40. [Eval bug: \[SYCL\] Intel Xe2 (Battlemage) B70: Weight corruption/nonsense output without GGML\_SYCL\_DISABLE\_OPT=1 · Issue #21893 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues/21893)
41. [Implementation:Ggml org Ggml Sycl backend - Leeroopedia](https://leeroopedia.com/index.php/Implementation:Ggml_org_Ggml_Sycl_backend)
42. [ipex-llm/docs/mddocs/Quickstart/llama\_cpp\_quickstart.md at main · intel/ipex-llm](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md)
43. [Performance of llama.cpp with Vulkan · ggml-org/llama.cpp · Discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
44. [Eval bug: Gemma 4 26B A4B (QAT) full-GPU offload corrupts output on Vulkan (Radeon 890M gfx1150) - isolated to fused MMVQ kernel path · Issue #27007 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues/27007)
45. [\[SYCL\]\[Intel\] Low performance on MoE models - SYCL is slower than VULKAN (A770) · Issue #19918 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/issues/19918)

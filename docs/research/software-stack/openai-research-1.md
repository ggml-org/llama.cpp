# Intel Arc Compute Performance on Arch Linux with an AMD iGPU Driving the Display

## Executive summary

The most important distinction is **which llama.cpp GPU backend you are actually using**. On Intel Arc, that determines whether Mesa is central to performance or almost irrelevant:

| llama.cpp path | Main userspace driver stack | Does Mesa version affect compute speed? | What to optimize first |
|---|---|---:|---|
| **Vulkan** | `ggml-vulkan` → Vulkan loader → **Mesa ANV** → kernel `i915`/`xe` | **Yes, directly** | `vulkan-intel`, Mesa/ANV, kernel, firmware |
| **SYCL / oneAPI Level Zero** | `ggml-sycl` → DPC++/SYCL → Level Zero loader → **Intel NEO Compute Runtime** → kernel `i915`/`xe` | **Generally no** | oneAPI version, `intel-compute-runtime`, IGC, kernel, firmware |
| **Intel OpenCL / NEO** | OpenCL app → `ocl-icd` → **Intel NEO Compute Runtime** → kernel | **Generally no** | `intel-compute-runtime`, IGC, kernel, firmware |
| **Mesa Rusticl OpenCL** | OpenCL → `ocl-icd` → Rusticl → Gallium **Iris** → kernel | **Yes** | `opencl-mesa`, Mesa, kernel |
| **VA-API / media** | `libva` → `intel-media-driver` | **No relevance to llama.cpp compute** | Do not tune for LLM inference |

Mesa's Intel Vulkan driver, **ANV**, is packaged by Arch as `vulkan-intel`; therefore Mesa releases can materially change llama.cpp Vulkan performance through shader compilation, memory-management behavior, subgroup/compute decisions, and driver optimizations. Mesa 26.x release work includes ANV compute-related changes, so “Mesa doesn't matter because the Arc isn't driving a monitor” is incorrect for Vulkan. citeturn3view0turn12search30

Conversely, when llama.cpp is using **SYCL/Level Zero**, the important Intel userspace driver is **NEO**, distributed on Arch as `intel-compute-runtime`. That package contains both Intel's Level Zero GPU driver (`libze_intel_gpu.so`) and GPU OpenCL implementation (`libigdrcl.so`); it is independent of Mesa ANV. citeturn8view3turn18search12 In fact, current llama.cpp upstream data show that upgrading the **oneAPI toolchain itself** can move Arc performance far more than changing Mesa: on an Arc B570, upstream reports prompt processing rising from roughly **434 to 1331 tokens/s — about 3.1× — when moving from oneAPI 2025.3 to 2026.1**. citeturn11view0

Your AMD iGPU driving the desktop is **not inherently a disadvantage for Arc compute**. Intel Compute Runtime explicitly operates through DRM render devices and requires access to `/dev/dri/renderD*`; those render nodes do not require the Intel card to own the desktop or a display connector. citeturn8view3 For compute-only use, **PRIME render offload is normally unnecessary**. What matters is that llama.cpp selects the Arc device rather than the AMD iGPU, and that your user can open the Arc render node. The llama.cpp SYCL documentation specifically recommends access through the `render`/`video` groups and supports explicit Level Zero device selection. citeturn11view0

As of **2026-10-05**, a strong Arch baseline is the current coherent rolling stack: Linux **7.2.8.arch1-2**, Mesa/ANV **26.2.4**, `libdrm` **2.4.134**, Intel Compute Runtime **26.35.39758.10**, `level-zero-loader` **1.32.0**, and Intel firmware **20260916**. citeturn5view0turn15search24turn3view0turn14search4turn18search12turn19search16turn5view2 There is no evidence for a universally “magic” kernel/Mesa pair worth pinning on Arch; staying current is generally preferable, especially for recent Arc hardware. For **SYCL**, however, the oneAPI version deserves explicit benchmarking rather than simply assuming the Arch package is optimal, because Arch's current `intel-oneapi-toolkit` package is still the **2026.0.1** series while llama.cpp reports its major B570 gain with **oneAPI 2026.1**. citeturn13search9turn11view0

One hardware detail remains unspecified: **the exact Arc model**. That matters for the `i915` versus `xe` discussion and for which driver optimizations apply. I assume “AMD Ryzen X7900X3D” means a **Ryzen 9 7900X3D**, but the precise AMD CPU model is largely irrelevant to Arc compute once the AMD iGPU is simply hosting the desktop.

## How the Intel Arc compute stacks actually fit together

The cleanest way to reason about this system is to separate the **kernel-mode driver**, **userspace compute/Vulkan driver**, and **API loader/toolchain** layers.

For Vulkan llama.cpp, the path is approximately:

```text
llama.cpp / ggml-vulkan
        │
        ▼
vulkan-icd-loader
        │
        ▼
Mesa ANV  ← Arch package: vulkan-intel
        │
        ▼
DRM render node (/dev/dri/renderD*)
        │
        ▼
i915 or xe kernel driver
        │
        ▼
GuC / GPU firmware
        │
        ▼
Intel Arc GPU
```

`vulkan-intel` comes from the Mesa source package and is explicitly Arch's open-source Vulkan driver for Intel GPUs. Its current package version tracks Mesa at `1:26.2.4-1`. citeturn3view0turn15search24 That makes Mesa version **directly performance-relevant** for llama.cpp's Vulkan backend even if the Arc has no display connected.

The SYCL path is fundamentally different:

```text
llama.cpp / ggml-sycl
        │
        ▼
Intel oneAPI DPC++ / SYCL runtime
        │
        ├── Level Zero loader
        │
        ▼
Intel Compute Runtime / NEO
  libze_intel_gpu.so
        │
        ▼
Intel Graphics Compiler (IGC)
Intel GMM library
        │
        ▼
DRM render node (/dev/dri/renderD*)
        │
        ▼
i915 or xe
        │
        ▼
Intel Arc GPU
```

Intel calls this userspace component the **Compute Runtime**, or **NEO**. The current Arch `intel-compute-runtime` package implements both oneAPI Level Zero and Intel GPU OpenCL and installs `libze_intel_gpu.so`, `libigdrcl.so`, and an `intel.icd` OpenCL vendor file. citeturn8view3turn18search12 Its required Arch dependencies include `intel-gmmlib` and `intel-graphics-compiler`; `igsc` is an optional dependency specifically associated with discrete-GPU firmware enumeration through Level Zero. citeturn18search6turn18search7

This separation is crucial: **upgrading Mesa cannot upgrade NEO, IGC, or the oneAPI DPC++ runtime**. If an Arc SYCL workload gets substantially faster after a system update, the cause may instead be `intel-compute-runtime`, `intel-graphics-compiler`, the oneAPI runtime/compiler, kernel DRM changes, or firmware. citeturn18search6turn11view0

### Iris, Gallium, Rusticl, and Clover

**Iris** is Mesa's modern Intel Gallium driver. Its primary relevance is OpenGL; it is *not* ANV and is *not* the Level Zero driver. Mesa's newer **Rusticl** OpenCL implementation can run through Gallium drivers such as Iris, and Mesa documents `RUSTICL_ENABLE=iris` as the mechanism for enabling that path. citeturn12search23

On current Arch, `opencl-mesa` installs a `rusticl.icd` file and `libRusticlOpenCL.so`. In other words, the current Mesa OpenCL package is a **Rusticl** package, not a reason to pursue historical Clover. citeturn15search0turn15search16

**Clover** is the older Gallium OpenCL implementation. For a modern Intel Arc llama.cpp system, it should be regarded as legacy and irrelevant. The practical choices are NEO/Level Zero, NEO/OpenCL where needed, Mesa Rusticl for experimentation, or ANV/Vulkan. citeturn15search16turn18search12

Likewise, **Beignet** is an obsolete Intel OpenCL path and should not be installed for Arc. Current Intel Arc support lives in Intel Compute Runtime/NEO; even the surviving AUR references to Beignet are legacy compatibility options rather than the current Intel GPU stack. citeturn22search11turn8view3

## Arch package and component impact matrix

The table below distinguishes packages that actually execute llama.cpp GPU work from loaders, compatibility libraries, video components, and diagnostics.

| Component | Arch package / name | Effect on llama.cpp Arc performance | Assessment |
|---|---|---|---|
| **Linux DRM kernel driver** | `linux`; in-kernel `i915` / `xe` | **High**. Scheduling, VM/local-memory handling, submission, power/frequency management and GPU support all live below userspace. Current Arch Linux is 7.2.8.arch1-2. citeturn5view0turn6search2 | **Keep current.** Particularly important for recent Arc generations. |
| **Intel firmware blobs** | `linux-firmware-intel`; normally via `linux-firmware` | **High for correctness; potentially performance-relevant.** Mesa documents GuC firmware requirements for modern Intel graphics including Arc Alchemist and recommends GuC ≥70.6.3 for optimal operation. Current Arch Intel firmware bundle is 20260916-1. citeturn5view1turn5view2turn12search9 | **Install/update.** |
| **Mesa common stack / Iris Gallium** | `mesa` 1:26.2.4-1 | **Low for NEO/SYCL; indirect or direct for Mesa paths.** Iris is not the Level Zero driver. citeturn15search24turn12search16 | Needed for your AMD-driven desktop anyway; don't expect it to speed up SYCL. |
| **Mesa ANV Vulkan driver** | `vulkan-intel` 1:26.2.4-1 | **High if llama.cpp uses Vulkan; irrelevant to SYCL.** It is Mesa's Intel Vulkan implementation. citeturn3view0 | **Essential for Vulkan.** |
| **Vulkan loader** | `vulkan-icd-loader` 1.4.363.0-1 | Required for Vulkan device/ICD dispatch but normally **not a throughput bottleneck**. citeturn16search0 | Install for Vulkan; update for compatibility, not expected t/s gains. |
| **llama/ggml Vulkan backend** | `ggml-vulkan`; current package seen at 0.25.3-2 | This is the application backend that actually issues Vulkan compute. It depends on `vulkan-icd-loader`. citeturn13search23 | **Essential if using packaged Vulkan backend.** |
| **Intel Compute Runtime / NEO** | `intel-compute-runtime` 26.35.39758.10-1 | **High for SYCL/Level Zero and Intel OpenCL.** Contains the Level Zero Intel GPU driver and OpenCL GPU runtime. citeturn18search12turn8view3 | **Core Arc compute package.** |
| **Intel Graphics Compiler** | `intel-graphics-compiler` | **High/medium for NEO.** NEO requires IGC for Intel GPU code generation. citeturn18search6turn8view3 | Let `pacman -Syu` keep it synchronized with NEO. |
| **Intel GMM library** | `intel-gmmlib` 22.10.0-1 | Memory-management support used by Intel's compute/media stack; usually an **indirect** rather than tunable performance component. citeturn18search2turn18search6 | Keep current; no separate tuning. |
| **Level Zero loader** | `level-zero-loader` 1.32.0-1 | Required API dispatch for Level Zero, but generally **not where kernel throughput is won or lost**. `ggml-sycl` depends on it. citeturn19search16 | **Required for SYCL/L0.** |
| **Level Zero headers** | `level-zero-headers` 1.32.0-1 | Build-time API headers only. citeturn19search9 | Runtime performance: **none**. |
| **`level-zero`** | Arch package base, split into `level-zero-loader` + `level-zero-headers` | There is no need to hunt for a separate runtime package named simply `level-zero`; these are the Arch split packages. citeturn19search9 | Use `level-zero-loader` at runtime. |
| **Intel oneAPI toolkit** | `intel-oneapi-toolkit` 2026.0.1…-2 | **Potentially very high for SYCL.** llama.cpp reports large backend performance differences across oneAPI versions. Arch's package is currently from the 2026.0.1 series. citeturn13search9turn11view0 | **Important for SYCL; benchmark version explicitly.** |
| **ggml SYCL backend** | `ggml-sycl` | Uses oneAPI/SYCL; Arch dependency metadata shows `level-zero-loader` and oneAPI integration. citeturn19search16turn13search9 | **Essential only for SYCL builds.** |
| **OpenCL ICD loader** | `ocl-icd` 2.3.5-1 | Finds/dispatches OpenCL implementations. Throughput impact is effectively negligible once dispatched. `intel-compute-runtime` and `opencl-mesa` are available ICD providers. citeturn3view3 | Install when using/testing OpenCL. |
| **Intel GPU OpenCL** | provided by `intel-compute-runtime` | **Relevant for OpenCL applications.** No extra Arc GPU package named `intel-opencl-runtime` is needed. NEO installs `intel.icd` and `libigdrcl.so`. citeturn18search12 | Prefer this Intel OpenCL implementation for Arc. |
| **Mesa Rusticl OpenCL** | `opencl-mesa` 1:26.2.4-1 | Mesa version directly affects this path because it uses Rusticl/Gallium. citeturn15search0turn15search16 | Optional/experimental for this use case; **not my preferred Arc llama.cpp path**. |
| **AUR `intel-opencl-runtime`** | `intel-opencl-runtime` AUR 2024.2.1-1 | The AUR entry is an **Intel CPU OpenCL runtime for Core/Xeon**, not the Arc GPU NEO runtime. Your host CPU is AMD anyway. citeturn22search2turn22search6 | **Irrelevant; do not install for Arc.** |
| **Beignet / Clover** | legacy/AUR or historical Mesa components | Obsolete paths compared with NEO and Rusticl for contemporary Arc. Current Arch `opencl-mesa` contains Rusticl, not Clover. citeturn15search16turn22search11 | **Irrelevant.** |
| **`libdrm`** | `libdrm` 2.4.134-1 | Userspace DRM interface; **important to ANV/Mesa compatibility**, but not normally an independent throughput lever. Interestingly, current Arch NEO lists `libdrm` only as optional for VA-API/OpenCL sharing. citeturn14search4turn18search6 | Keep current; don't expect standalone t/s gains. |
| **IGSC library** | `igsc` 0.9.5-14 | Firmware-update/control support for Intel discrete GPUs; NEO lists it for discrete-GPU firmware enumeration through Level Zero. citeturn18search7turn16search14 | **Recommended for Arc**, but not a hot-path performance library. |
| **VA-API API library** | `libva` 2.24.1-1 | Video acceleration API, not llama tensor compute. citeturn16search2 | **No llama.cpp performance benefit.** |
| **Intel VA-API media driver** | `intel-media-driver` 26.3.5-1 | Intel video codec/processing driver (`iHD_drv_video.so`), not Level Zero/Vulkan tensor compute. citeturn16search1turn16search5 | **Irrelevant unless doing video work.** |
| **Legacy Intel VA driver** | `libva-intel-driver` | Intended for older Intel graphics families, not the Arc compute stack. citeturn16search13 | **Do not install for Arc compute.** |
| **OpenCL diagnostic** | `clinfo` 3.0.25.02.14-1 | Enumerates platforms/devices; no runtime speed effect. citeturn14search5 | Useful diagnostic only. |
| **Intel DRM diagnostics** | `intel-gpu-tools` 2.5-1 | Monitoring/testing tools for Intel DRM; no intrinsic speed increase. citeturn17search3 | Very useful for verifying actual GPU load/frequency. |
| **ROCm** | `rocm-*` | AMD compute stack, not needed to compute on an Intel Arc. | **Irrelevant to Arc.** Do not add it merely because the desktop GPU is AMD. |
| **“firmware-misc”** | No corresponding Arch requirement | That naming belongs to other distro packaging schemes; Arch splits firmware primarily under `linux-firmware-*`, including `linux-firmware-intel`. citeturn5view1turn5view2 | Use Arch's firmware packages. |
| **“intel-level-zero-gpu-manager”** | No current official Arch package found by that exact name | On Arch, the **Intel Level Zero GPU driver is supplied by `intel-compute-runtime`** and the generic loader by `level-zero-loader`. citeturn18search12turn19search16 | Do not chase this name on Arch. |

A particularly useful naming clarification is that Intel's packaging documentation for Debian/RPM-based distributions may refer to packages such as **`intel-level-zero-gpu`**. That should not be translated literally to Arch package names: on Arch, `intel-compute-runtime` provides `libze_intel_gpu.so`, and `level-zero-loader` provides the generic Level Zero loader. citeturn4search3turn18search12turn19search16

## Mesa version impact by backend

### Vulkan: Mesa can absolutely change llama.cpp performance

For Vulkan, ANV is not just some desktop-rendering baggage. It is the userspace driver handling Vulkan compute pipeline creation, SPIR-V lowering/compilation, resource and memory behavior, command submission interfaces, synchronization, and GPU-specific optimization. Arch's `vulkan-intel` package is built from the Mesa source tree and currently follows Mesa **26.2.4** exactly. citeturn3view0turn15search24

Mesa release notes demonstrate ongoing Intel compute work rather than a frozen driver. Mesa 26.0, for example, carried ANV changes involving compute shared-memory reporting, subgroup/SIMD decisions, internal compute kernels, and Xe2 behavior. citeturn12search30 Therefore, a Mesa upgrade can produce anything from no measurable llama.cpp difference to a substantial regression or gain depending on the model's shaders and Arc generation. It is not sensible to predict a fixed percentage from the Mesa version number alone.

The actionable interpretation is:

**If you run llama.cpp Vulkan, test Mesa/ANV versions. If you run llama.cpp SYCL/Level Zero, Mesa should not be the first suspect.**

Arch's current `vulkan-intel 26.2.4-1` is a sensible baseline; I found no strong evidence that an older Mesa release is systematically faster enough on Arc llama.cpp to justify pinning it. citeturn3view0 Given Arch's rolling stack and the pace of Intel ANV development, my default recommendation is to remain on the current stable repository package, reverting only when you can reproduce a regression with `llama-bench`.

### SYCL and Level Zero: oneAPI/NEO matter much more than Mesa

llama.cpp's SYCL documentation says its Intel path uses DPC++ and Level Zero and is primarily designed for Intel GPUs. citeturn11view0 The upstream performance history gives unusually good evidence that software-stack version matters here.

On Arc A770, llama.cpp reported a Q4_0 optimization raising generation performance from roughly **42 to 55 tokens/s**, approximately a 30% improvement. citeturn11view0 More strikingly, in 2026 the project reported **1331 versus 434 tokens/s prompt processing** on Arc B570 with oneAPI 2026.1 versus 2025.3—about a **3.1× difference**. citeturn11view0

That means an Arc user benchmarking “Mesa versions” while actually running `GGML_SYCL` could spend considerable effort tuning the wrong component.

The critical SYCL chain is closer to:

**llama.cpp commit → oneAPI/DPC++ → NEO version → IGC version → kernel KMD → firmware → hardware**

rather than:

**llama.cpp → Mesa**.

### OpenCL: distinguish NEO from Mesa OpenCL

OpenCL is especially easy to misconfigure because multiple ICDs can coexist. `ocl-icd` is merely the loader. Intel's current Arc GPU OpenCL implementation is inside `intel-compute-runtime`, while `opencl-mesa` supplies Mesa Rusticl. citeturn3view3turn18search12turn15search16

`clinfo` can therefore show multiple platforms if several ICD files exist. That is normal; it does not mean they are interchangeable in performance. citeturn14search5

The AUR package named `intel-opencl-runtime` is particularly misleading for this question: its current description is an **Intel oneAPI OpenCL runtime for Intel Core and Xeon processors**, i.e. CPU OpenCL, not the Arc GPU NEO runtime. citeturn22search6 On your AMD CPU it provides no useful Arc acceleration.

For llama.cpp on an Arc, I would rank the compute paths for serious testing as **Vulkan and SYCL/Level Zero first**, NEO OpenCL only where the specific backend needs it, and Rusticl/OpenCL as an experimental alternative rather than the performance default.

## Kernel, firmware, DRM and the non-primary Arc GPU

### The Arc does not need to render the desktop

Intel's own Compute Runtime documentation explicitly states that applications need permission to the DRM **render nodes** at `/dev/dri/renderD*`. citeturn8view3 That is precisely the mechanism that permits compute-capable clients to access a GPU without making it the display/master DRM device.

So a topology such as this is entirely legitimate:

```text
AMD Ryzen iGPU
  └─ amdgpu
      └─ Wayland/X11 compositor + monitors

Intel Arc dGPU
  └─ i915 or xe
      └─ /dev/dri/renderDXYZ
          ├─ ANV / Vulkan / llama.cpp
          └─ NEO / Level Zero / llama.cpp
```

The Intel card being secondary or completely headless does **not** force computation onto the AMD iGPU, nor does ANV or NEO require an attached monitor. The real failure mode on a mixed-GPU system is **device selection**: both Intel and AMD may be exposed by the Vulkan loader, so llama.cpp must actually choose the Intel physical device. The same principle applies to SYCL device enumeration. llama.cpp's SYCL documentation shows Level Zero devices separately and supports `ONEAPI_DEVICE_SELECTOR`, including selectors such as `level_zero:0`. citeturn11view0

This is also why **PRIME is mostly beside the point here**. PRIME is useful for graphics render-offload/display workflows. Pure Vulkan/Level Zero compute can directly target the Arc's render node; there is no reason to render the desktop on the Intel card just to make compute work.

### Permissions matter more than PRIME

Check the actual nodes:

```bash
ls -l /dev/dri/
ls -l /dev/dri/by-path/
id
```

With a static group-based setup, the Arc render node should ordinarily be usable through the `render` group. llama.cpp's Intel SYCL documentation recommends adding the user to both `render` and `video` on Linux. citeturn11view0

```bash
sudo usermod -aG render,video "$USER"
```

Log out completely and back in after changing supplementary groups.

Do not blindly add custom udev rules. First inspect the node:

```bash
stat -c '%n  owner=%U  group=%G  mode=%a' /dev/dri/renderD*
```

A historical systemd 258 regression illustrates why this check matters: affected systems ended up with render nodes owned `root:root` and mode `0600`, causing Vulkan initialization failures for non-root users. citeturn20search22 That is an **access/correctness problem**, not a performance tuning technique; if your workload already accesses the Arc normally, altering udev permissions will not increase tokens/s.

### Map the render node to the correct physical GPU

On a dual-GPU machine, do not assume `renderD128` means Intel. Enumeration order can change. Use PCI topology:

```bash
lspci -nnk | grep -A4 -E 'VGA|3D|Display'
ls -l /dev/dri/by-path/
```

Then check each compute API independently:

```bash
vulkaninfo --summary
clinfo
sycl-ls
```

`clinfo` is an official Arch diagnostic package for enumerating OpenCL platforms and properties. citeturn14search5 `sycl-ls` is the relevant oneAPI/SYCL check; llama.cpp's documentation uses it to distinguish `[level_zero:gpu]` and `[opencl:gpu]` devices. citeturn11view0

For current llama.cpp builds, also inspect the application's own device list rather than trusting API ordering:

```bash
llama-cli --list-devices
```

Then explicitly select the Intel Arc in your actual workload.

### `i915` versus `xe`

Current Linux exposes two Intel graphics kernel drivers, **i915** and **Xe**; Arch's Intel graphics documentation acknowledges both KMS drivers. citeturn6search2 Which is appropriate depends strongly on the Arc generation, which you did not specify.

For **Arc A-series / Alchemist**, `i915` has the longer production history. For newer GPU generations, Xe is increasingly the intended KMD. I would **not switch an otherwise working Arc from i915 to Xe, or vice versa, merely because llama.cpp exists**. The backend-visible userspace stack and the specific Arc generation should drive that decision.

The practical rule is:

```bash
lspci -nnk
```

and inspect:

```text
Kernel driver in use: i915
```

or:

```text
Kernel driver in use: xe
```

Then leave the default alone unless you are investigating a documented driver-specific bug or feature. Running the newest stable Arch kernel is more defensible than forcing a different KMD because of generic online tuning advice. Current Arch stable is Linux 7.2.8.arch1-2 as of 2026-10-05. citeturn5view0

Intel documentation also describes disabling i915 hangcheck for compute kernels that intentionally execute for exceptionally long periods, because hang detection can interpret very long execution as a stuck GPU. citeturn7search9 This is **not a normal llama.cpp performance tweak**. Typical inference kernels should not require disabling GPU recovery, and doing so makes real hangs harder to recover from.

### Firmware has two meanings

There are two related but different things here.

The kernel-loaded GPU firmware comes from Arch's `linux-firmware-intel`, normally installed through the broader `linux-firmware` meta package. Current Arch packages are `20260916-1`. citeturn5view1turn5view2 Mesa's ANV documentation specifically notes modern Intel hardware's GuC firmware dependency and recommends GuC 70.6.3 or newer for optimal behavior on relevant hardware including Arc Alchemist. citeturn12search9

Separately, **IGSC** is Intel's Graphics System Controller firmware update/access library for discrete GPUs. Arch packages it as `igsc`; Intel Compute Runtime marks it optional for discrete-GPU firmware enumeration through Level Zero. citeturn18search7turn16search14 Installing `igsc` is reasonable on an Arc compute system, but its presence by itself does not make matrix kernels run faster.

## Performance evidence and the stack I would run

There is an instructive tension in the available llama.cpp measurements: **neither Vulkan nor SYCL can be declared universally fastest across Intel Arc generations and software versions**.

An April 2026 llama.cpp community report comparing Battlemage B50/B70 found the Mesa Vulkan backend dramatically faster than the reporter's SYCL setup. On Gemma-family Q4_K testing, prompt processing was approximately **1169 versus 352 tokens/s** and token generation approximately **40.2 versus 13.3 tokens/s**, Vulkan versus SYCL respectively—roughly 3.3× and 3.0×. citeturn11view1

That result needs a large warning label: the issue was marked unconfirmed, the reporter described an older DDR4/PCIe 3.0 system, and the result predates llama.cpp's subsequent oneAPI 2026.1 performance report. citeturn11view1turn11view0 It is therefore evidence that **the Vulkan backend can be excellent on Intel Arc**, not evidence that Vulkan is intrinsically three times faster than SYCL.

Later llama.cpp upstream testing showed the opposite lesson: software improvements to the SYCL/oneAPI path can transform performance, with the aforementioned **3.1× prompt-processing increase on B570 using oneAPI 2026.1 versus 2025.3**. citeturn11view0

That leads to a much stronger recommendation than trying to infer performance from API branding:

> **On your actual Arc, benchmark Vulkan/ANV and current SYCL/Level Zero with the same llama.cpp commit, model, quantization, context, and GPU offload settings. Keep whichever is faster and stable.**

For Arch as of 2026-10-05, I would use this as the **Vulkan baseline**:

| Layer | Recommended baseline |
|---|---|
| Kernel | `linux` **7.2.8.arch1-2** citeturn5view0 |
| Firmware | `linux-firmware-intel` **20260916-1** citeturn5view2 |
| Mesa | `mesa` **26.2.4-1** citeturn15search24 |
| Intel Vulkan | `vulkan-intel` **26.2.4-1** citeturn3view0 |
| DRM userspace | `libdrm` **2.4.134-1** citeturn14search4 |
| Vulkan loader | `vulkan-icd-loader` **1.4.363.0-1** citeturn16search0 |
| llama backend | Current `GGML_VULKAN` / `ggml-vulkan` citeturn13search23 |

And this as the **SYCL baseline**:

| Layer | Recommended baseline |
|---|---|
| Kernel | Current Arch `linux` citeturn5view0 |
| Firmware | Current `linux-firmware-intel` citeturn5view2 |
| Intel GPU compute driver | `intel-compute-runtime` **26.35.39758.10-1** citeturn18search12 |
| Intel compiler | Repo-current `intel-graphics-compiler`, synchronized through full Arch upgrades citeturn18search6 |
| GMM | `intel-gmmlib` **22.10.0-1** citeturn18search2 |
| Level Zero loader | `level-zero-loader` **1.32.0-1** citeturn19search16 |
| GSC support | `igsc` **0.9.5-14** citeturn18search7 |
| oneAPI | **Target/test 2026.1 where feasible**, particularly for B-series; Arch official `intel-oneapi-toolkit` is presently 2026.0.1…-2. citeturn11view0turn13search9 |

That oneAPI discrepancy is the biggest version issue I would investigate on a **Battlemage** card. Arch's official toolkit is convenient and internally packaged, but upstream llama.cpp's measured 2026.1 improvement is large enough that a serious performance comparison should include it. I would test it in an isolated build/runtime environment rather than partially replacing random oneAPI libraries in `/usr`; mixed oneAPI/NEO/IGC installations are exactly the sort of configuration that can make benchmark results uninterpretable.

For **Alchemist A-series**, I would start with the completely current Arch stack rather than chasing AUR driver builds. llama.cpp has already demonstrated substantial A770 gains from software/backend work, and the modern NEO/ANV stacks support Alchemist directly. citeturn11view0turn8view3

I would use `mesa-git`, `intel-compute-runtime-git`, or binary/AUR alternatives only as **A/B test instruments for a known upstream fix or regression**, not as default “faster” packages. Current official Arch packages already track very recent upstream releases. citeturn15search24turn18search12

## Actionable Arch Linux checklist

### Start with a fully coherent system upgrade

On Arch, partial upgrades are the wrong baseline for a graphics/compute stack. Bring kernel, firmware, Mesa, DRM, loaders, and Intel runtime dependencies forward together:

```bash
sudo pacman -Syu
```

As of 2026-10-05, that places the principal relevant official packages near Linux 7.2.8, Mesa/ANV 26.2.4, `libdrm` 2.4.134, Compute Runtime 26.35, Level Zero loader 1.32, and firmware 20260916. citeturn5view0turn15search24turn14search4turn18search12turn19search16turn5view2

### For a Vulkan-first llama.cpp setup

Install the current kernel/firmware and Intel Vulkan stack:

```bash
sudo pacman -S \
    linux \
    linux-firmware \
    mesa \
    libdrm \
    vulkan-intel \
    vulkan-icd-loader \
    vulkan-tools \
    intel-gpu-tools
```

`vulkan-intel` is the key package here: that is **ANV**, and therefore the package whose Mesa version directly affects Intel Arc Vulkan compute. citeturn3view0 Your AMD desktop can remain on `amdgpu`/Mesa; there is no need to move the display to Arc.

For a source build of llama.cpp:

```bash
cmake -B build \
    -DGGML_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)"
```

llama.cpp/ggml supports the `GGML_VULKAN` build path, and Arch separately packages a `ggml-vulkan` backend. citeturn10search14turn13search23

Verify that Vulkan sees the Arc:

```bash
vulkaninfo --summary
./build/bin/llama-cli --list-devices
```

Do **not** assume the first Vulkan GPU is the Arc simply because the Arc is discrete; with an AMD Vulkan ICD installed, both may enumerate.

### For a SYCL/Level Zero comparison

Install the Intel compute stack:

```bash
sudo pacman -S \
    intel-compute-runtime \
    level-zero-loader \
    intel-oneapi-toolkit \
    igsc \
    intel-gpu-tools
```

`intel-compute-runtime` automatically brings in its required Intel GMM and Intel Graphics Compiler dependencies; the oneAPI toolkit depends on the Level Zero loader. citeturn18search6turn13search9

For OpenCL diagnostics as well:

```bash
sudo pacman -S ocl-icd clinfo
```

The NEO package already supplies the Intel GPU OpenCL ICD, so **do not install the AUR `intel-opencl-runtime` for the Arc**; that AUR package is for CPU OpenCL on Intel Core/Xeon. citeturn18search12turn22search6

Check the Intel APIs:

```bash
sycl-ls
clinfo
```

For an explicit Level Zero device in the SYCL path:

```bash
export ONEAPI_DEVICE_SELECTOR='level_zero:0'
```

The exact index must match your `sycl-ls` output; llama.cpp documents this Level Zero selector mechanism. citeturn11view0

A source build is conceptually:

```bash
cmake -B build-sycl \
    -DGGML_SYCL=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build-sycl --config Release -j"$(nproc)"
```

The SYCL backend and Level Zero path are documented by llama.cpp and supported on Arch Linux. citeturn11view0turn10search14

### Fix render-node access, not PRIME

Inspect:

```bash
ls -l /dev/dri/
ls -l /dev/dri/by-path/
id
```

If your account lacks render-node access:

```bash
sudo usermod -aG render,video "$USER"
```

Then completely log out and log in. Intel Compute Runtime explicitly requires access to `/dev/dri/renderD*`, while llama.cpp recommends the render/video groups for its Intel SYCL configuration. citeturn8view3turn11view0

No `DRI_PRIME=1` should be required for Level Zero. For Vulkan, prefer explicit llama.cpp device selection instead of relying on PRIME as a hidden device-selection mechanism.

### Do not install these in pursuit of llama.cpp speed

Avoid adding `beignet`, legacy Clover packages, `libva-intel-driver`, the AUR `intel-opencl-runtime`, or ROCm merely for Intel Arc inference. Current Arch Mesa OpenCL is Rusticl; current Intel Arc GPU OpenCL/Level Zero is NEO; VA-API is a separate media stack; and the AUR `intel-opencl-runtime` targets Intel CPU OpenCL. citeturn15search16turn18search12turn16search13turn22search6

Likewise, `libva` and `intel-media-driver` are perfectly reasonable packages to have for video decoding/encoding, but upgrading them should not improve llama.cpp token throughput because they implement the VA-API media path rather than ANV or NEO compute. citeturn16search1turn16search2

### Benchmark the two meaningful backends

Use the same llama.cpp revision, GGUF, quantization, context size, batch settings, and GPU offload settings for both runs. `llama-bench` is preferable to judging by interactive feel because prompt processing and token generation can react very differently to a backend.

A useful experiment matrix is:

```text
A. Vulkan + current Mesa/ANV
B. SYCL + current Arch oneAPI/NEO
C. SYCL + oneAPI 2026.1, if your Arc is B-series and B is significantly slower
```

Record at least:

```text
llama.cpp commit
Arc model
kernel version
kernel driver: i915 or xe
Mesa/vulkan-intel version
intel-compute-runtime version
oneAPI version
model + quantization
pp512
tg128
GPU layers/offload
```

The community Battlemage result showing Vulkan around **1169 pp512 / 40.2 tg128** versus SYCL around **352 / 13.3** proves that large backend differences can occur in a real installation, while llama.cpp's later B570 result showing a **3.1× oneAPI-version improvement** proves that those differences are not immutable properties of Vulkan versus SYCL. citeturn11view1turn11view0

### Bottom line

For your topology—**AMD iGPU drives the monitor, Intel Arc is compute-only**—the packages with the greatest chance of materially changing llama.cpp performance are:

**Vulkan path:** `vulkan-intel`/Mesa ANV → kernel → Intel firmware → llama.cpp Vulkan backend. citeturn3view0turn12search30

**SYCL path:** oneAPI version → `intel-compute-runtime`/NEO → `intel-graphics-compiler` → kernel → Intel firmware. Mesa is largely outside this execution path. citeturn8view3turn18search6turn11view0

The Arc not being the primary display is **not a reason to expect lower compute performance**. The things to verify are that the Arc has a working DRM render node, your user can access it, llama.cpp selects it rather than the AMD iGPU, and you are benchmarking the backend whose driver stack you think you are tuning. citeturn8view3turn11view0

For an Arch system today, my default would be **Linux 7.2.x + Mesa/ANV 26.2.x + current Intel firmware for Vulkan**, and **the same kernel/firmware plus current NEO/IGC and a deliberate oneAPI 2026.1 test for SYCL**, especially if the unspecified Arc card is Battlemage. I would not pin an older Mesa, switch the desktop to Arc, configure PRIME, install Beignet, install the Intel CPU OpenCL AUR package, or add ROCm without a specific reproducible problem showing why. citeturn5view0turn15search24turn18search12turn11view0
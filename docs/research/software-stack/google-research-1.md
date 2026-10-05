# **Architectural Optimization and Package Dependency Dynamics for Llama.cpp Acceleration on Intel Arc GPUs under Arch Linux**

In modern high-performance local large language model (LLM) inference, software execution stack interactions frequently dictate real-world performance ceilings. When deploying llama.cpp on an Arch Linux distribution utilizing a heterogeneous dual-GPU topology - where desktop display rendering and composition are driven by an integrated AMD Ryzen 9 7900X3D GPU via the amdgpu driver, while a discrete Intel Arc GPU operates headlessly strictly as a GPGPU compute accelerator - the operational boundaries between user-space graphics drivers, compute runtimes, and low-level kernel modules become vital1.  
A primary area of configuration ambiguity is whether user-space display driver frameworks, such as Mesa, exert any influence over a discrete graphics adapter that is completely decoupled from display output. Evaluating this requires analyzing the software dependency graph across distinct execution backends (GGML\_VULKAN versus GGML\_SYCL), mapping user-space driver interfaces to kernel Direct Rendering Manager (DRM) modules, and examining empirical performance data across software revisions1.

## **Direct Impact of Mesa on Headless Intel Arc Compute Performance**

The question of whether the system Mesa package version impacts Intel Arc discrete GPU performance when the card is not rendering display output yields a definitive answer: Mesa plays a critical, performance-defining role, but its impact depends entirely on the compiled compute backend of llama.cpp1.  
Although traditional mental models treat Mesa solely as a desktop graphics driver for OpenGL and X11 or Wayland display presentation, under Arch Linux Mesa serves as the host distribution framework for vulkan-intel, also known as the ANV driver2. When llama.cpp is built using its Vulkan backend (-DGGML\_VULKAN=ON), the system bypassing display composition does not bypass Mesa1. The Vulkan compute kernels used for tensor matrix multiplication (![][image1]) execute directly through the ANV driver interface provided by Mesa, rendering the user-space driver version pivotal to hardware compute throughput1.

### **The Vulkan Driver Execution Mechanism and Mesa 26.1 Acceleration**

In Vulkan-based LLM inference, matrix operations can use cooperative-matrix extensions when the backend and GPU expose them.
The cited Mesa 26.1 throughput gain was measured on a Battlemage B70. It must not be generalized to Alchemist/A770 without a same-hardware A/B; Mesa changes can improve or regress throughput depending on the GPU and workload.

### **SYCL Backend Execution Isolation from Mesa**

When llama.cpp is built using Intel's native oneAPI SYCL backend (-DGGML\_SYCL=ON), Mesa's direct influence on token generation throughput drops to near zero1. The SYCL execution pipeline routes calls through the Level Zero driver interface (libze\_loader.so) and the Intel Compute Runtime (intel-compute-runtime), bypassing the Mesa ANV driver layer completely4. Under the SYCL path, Mesa updates affect only secondary system dependencies, whereas direct compute gains are governed by updates to intel-compute-runtime and the intel-graphics-compiler1.

## **Comprehensive Package Ecosystem and Performance Dependencies**

Maximizing the throughput of an Intel Arc discrete GPU on Arch Linux requires managing a multi-tiered system dependency graph spanning user-space libraries, intermediate compilers, compute runtimes, and kernel modules3. Because the operating system display server runs on the AMD Ryzen integrated GPU via amdgpu, the Intel software ecosystem operates cleanly in a headless compute role2.  
The following software packages directly govern, or possess the potential to influence, Intel Arc LLM inference performance:

| Package Name | System Layer | Engine Target | Operational Mechanism & Performance Influence |
| :---- | :---- | :---- | :---- |
| mesa / vulkan-intel | User-space Driver | Vulkan (GGML\_VULKAN) | Delivers the Intel ANV Vulkan compute driver2. Version 26.1+ introduces NV\_coopmat2 lowering routines, doubling token decode rates and enabling efficient multi-slot execution scaling1. |
| intel-compute-runtime | Compute Runtime | SYCL (GGML\_SYCL) / OpenCL | Implements the Level Zero and OpenCL driver interfaces (libze\_loader.so)8. Governs low-level memory allocation strategies and hardware kernel submission overhead for SYCL workloads4. |
| intel-graphics-compiler | JIT / IR Compiler | SYCL / OpenCL | Translates SPIR-V intermediate representations into native Xe ISA assembly10. Upstream compiler releases optimize instruction packing and register allocation within XMX execution pipelines7. |
| intel-gmmlib | Memory Management | SYCL & Vulkan | Intel Graphics Memory Management Library10. Manages low-level page allocations, buffer aliasing, and Base Address Register (BAR) aperture mappings across PCIe bus interfaces8. |
| level-zero / level-zero-headers | API Abstraction Layer | SYCL (GGML\_SYCL) | Low-overhead direct-to-metal user-space system interface8. Missing build-time headers cause CMake to disable GGML\_SYCL\_SUPPORT\_LEVEL\_ZERO\_API, forcing higher-overhead OpenCL fallback paths8. |
| linux / linux-cachyos | Linux Kernel | System Core | Houses the Direct Rendering Manager (DRM) kernel drivers3. Selecting between the legacy i915.ko module and the modern xe.ko module alters memory scheduling and preemptive GPU context switching3. |
| linux-firmware | Hardware Microcode | Hardware Firmware | Delivers updated GuC (Graphics Microcontroller) and HuC microcode binaries7. GuC handles hardware-based kernel submission scheduling and power-state management7. |
| intel-deep-learning-essentials | Math Libraries | SYCL (GGML\_SYCL) | Integrates oneDNN and oneMKL primitives optimized for Intel XMX matrix tensor operations, significantly improving prompt prefill processing speeds4. |

### **Kernel Driver Mechanics: Transitioning from i915 to xe**

A critical architectural decision on Arch Linux is selecting the active kernel DRM driver handling the Intel Arc card3. Modern kernels (Linux 6.12+) support both the legacy i915 module and the modern xe module, which was designed explicitly for Xe-based architectures (Alchemist, Battlemage, and integrated Xe graphics)3.  
The xe driver eliminates decades of legacy platform debt accumulated in i915, introducing a modernized Virtual Memory Address (VMA) binding model and offloading microsecond-level hardware scheduling directly to the onboard GuC microcontroller7. For intensive llama.cpp compute workloads, explicitly enforcing the xe driver via kernel boot parameters mitigates context-switching latency bottlenecks and prevents engine resets during deep context evaluations3.

## **Architectural Comparison: SYCL versus Vulkan Backends**

Selecting between the GGML\_SYCL and GGML\_VULKAN compilation backends on Arch Linux involves evaluating distinct trade-offs across prompt processing (prefill) latency, multi-tenant concurrent scaling, and runtime driver stability1.  
The SYCL backend demonstrates clear performance advantages during prompt processing (PP) or prefill phases1. By leveraging Intel's specialized oneDNN math libraries and native XMX instruction pipelines, GGML\_SYCL achieves exceptionally high matrix processing throughput during initial context ingestion, consistently outperforming Vulkan in single-stream prefill evaluations1.  
Conversely, during single-token generation (TG) or decode phases, recent Mesa developments have shifted the advantage toward Vulkan1. With Mesa 26.1+ exposing NV\_coopmat2 support, Vulkan matches or slightly exceeds single-stream SYCL generation speeds while operating with reduced user-space driver overhead1.  
Under concurrent multi-slot workloads - such as serving parallel client queries through llama-server - the execution models of the two backends diverge dramatically1. Vulkan exhibits near-linear performance scaling as additional context slots are allocated1. As parallel execution slots scale from one to eight, Vulkan utilizes internal queue submission parallelism, enabling aggregate decode throughput on high-end Arc GPUs to reach 176 tokens/second1. In contrast, SYCL's batched-decode kernels encounter kernel-launch queue bottlenecks on discrete Intel hardware, causing multi-slot decode performance to plateau near 100 tokens/second aggregate1. This architectural gap allows Vulkan to deliver up to a 70% throughput advantage under heavily concurrent server loads1.  
The following empirical performance summary highlights cross-backend throughput metrics gathered on an Intel Arc GPU executing Qwen3.6-35B model variants1:

| Inference Metric / Workload Phase | Vulkan (Mesa 26.0) | Vulkan (Mesa 26.1+) | Vulkan (Mesa 26.1 \+ \--ubatch 2048\) | SYCL (Compute Runtime 26.x) |
| :---- | :---- | :---- | :---- | :---- |
| **Prefill pp512** (tokens/second) | 1,288 t/s | 1,225 t/s | 1,310 t/s | **1,485 t/s** |
| **Prefill pp2048** (tokens/second) | 1,098 t/s | 1,172 t/s | **1,824 t/s** | 1,210 t/s |
| **Single-Stream Decode tg128** | 37.8 t/s | 76.0 t/s | **76.0 t/s** | 77.3 t/s |
| **4-Slot Aggregate Decode** | \~240 t/s | 128 t/s | **132 t/s** | 92 t/s |
| **8-Slot Aggregate Decode** | - | 170 t/s | **176 t/s** | 100 t/s |
| **8-Slot Total System Throughput** | - | 526 t/s | **624 t/s** | 350 t/s |

## **System Configuration, Runtime Tuning, and Bug Mitigations**

Achieving maximum stability and inference performance on an Arch Linux system equipped with an Intel Arc discrete GPU requires establishing specific environment variables, kernel parameters, and runtime execution flags3.

### **Mitigating the SYCL Persistent Cache Segfault Bug**

A widespread issue affecting SYCL deployments on Linux stems from a persistent compilation cache bug within the oneAPI runtime (intel/llvm\#21972)8. When llama.cpp dynamically loads shared compute libraries (libggml-sycl.so), the SYCL JIT persistent code cache path frequently encounters a NULL pointer dereference8. This triggers immediate segmentation faults (SIGSEGV) or kernel resets (xe bcs engine reset) during model loading, even when offloading zero model layers to the GPU8.  
To resolve this failure, persistent JIT caching must be explicitly disabled in the process environment prior to execution8:

Bash  
export SYCL\_CACHE\_PERSISTENT=0

Disabling persistent caching forces in-memory kernel compilation during process initialization, completely preventing model-loading crashes without impacting active tensor execution speeds8.

### **Mandatory Runtime Environment Variables**

For stable SYCL execution on Arch Linux, the shell environment must be initialized with the following configuration profile8:

Bash  
\# Source the oneAPI runtime environment  
source /opt/intel/oneapi/setvars.sh \--force

\# Force oneAPI to explicitly bind to the Level Zero Intel discrete GPU device  
export ONEAPI\_DEVICE\_SELECTOR=level\_zero:0

\# Enable System Management metrics (exposes power, thermals, and clocks via Sysman API)  
export ZES\_ENABLE\_SYSMAN=1

\# Disable persistent SYCL kernel cache to prevent JIT segfaults  
export SYCL\_CACHE\_PERSISTENT=0

Note: Custom environment overrides such as ZEX\_NUMBER\_OF\_CCS or SYCL\_UR\_USE\_LEVEL\_ZERO\_V2=0 should be avoided, as they frequently break hardware device enumeration across modern xe driver interfaces8.

### **Enforcing Kernel Module Options**

To ensure the modern xe driver manages the Intel Arc discrete GPU rather than falling back to the legacy i915 driver, administrator parameters must be specified in the bootloader configuration (such as systemd-boot or GRUB) or defined within /etc/modprobe.d/3.  
Assuming the Intel Arc GPU PCI device ID is identified via lspci \-nn as 56a0 (e.g., Arc A770) or corresponding series IDs (56b0, 56c0)3, append the following parameters to the kernel boot command line3:  
i915.force\_probe=\!56a0 xe.force\_probe=56a0  
Alternatively, create the configuration file /etc/modprobe.d/intel\_xe.conf containing18:  
options i915 force\_probe=\!56a0 options xe force\_probe=56a0  
After modifying these options, regenerate the initial ramdisk via sudo mkinitcpio \-P and reboot to confirm driver assignment13.

### **Llama.cpp Physical Batch Tuning (ubatch)**

Because Intel Arc GPUs feature high memory bandwidth bus architectures paired with XMX matrix engines, standard software default parameters often underutilize the underlying hardware1. The default physical batch size (--ubatch-size 512\) in llama.cpp introduces a severe compute-bound bottleneck during prompt processing1.  
Increasing \--ubatch-size to 2048 dramatically enhances prompt processing throughput1:

Bash  
llama-server \-m model.gguf \-ngl 99 \--ubatch-size 2048 \-c 32768 \--host 0.0.0.0 \--port 8080

Expanding physical batching from 512 to 2048 increases prefill processing speed on 2,048-token prompts from 1,172 t/s to 1,824 t/s (\~55% performance gain) without introducing any latency penalty during single-token generation1.

## **Conclusions and Practical Deployment Guidance**

Deploying llama.cpp on an Arch Linux system utilizing an Intel Arc GPU for headless compute alongside an AMD Ryzen iGPU yields several clear architectural conclusions1.  
First, Mesa is a vital performance determinant whenever llama.cpp is built with the Vulkan backend1. Because Mesa delivers the vulkan-intel ANV driver, upgrading Mesa to version 26.1+ provides an immediate double in single-stream token generation performance due to native NV\_coopmat2 cooperative matrix support1.  
Second, selecting the optimal compilation backend depends on the primary execution pattern1. For multi-tenant API serving, parallel agent tasks, or workloads requiring maximum single-token generation speed, the Vulkan backend paired with Mesa 26.1+ represents the optimal technical path1. For single-tenant workloads characterized by massive prompt prefill phases or model quantization tasks, the SYCL backend leveraging oneDNN and oneMKL remains highly competitive1.  
Finally, neither the xe KMD nor `SYCL_CACHE_PERSISTENT=0` is a universal stability requirement for Alchemist. This repository's current known-good A770 setup uses xe with copy-engine off and `SYCL_CACHE_PERSISTENT=1`; choose i915/xe and cache policy from same-stack measurements rather than the B70-derived recommendation.

#### **Works cited**

> 1. Don't trust your old benchmarks: a weekend of testing on the Intel, [https\://jonathanmann.tech/blog/intel-arc-b70-llama-cpp-benchmarks/](https://jonathanmann.tech/blog/intel-arc-b70-llama-cpp-benchmarks/)  
> 2. Vulkan \- ArchWiki, [https\://wiki.archlinux.org/title/Vulkan](https://wiki.archlinux.org/title/Vulkan)  
> 3. Intel xe drivers \- Kernel \- CachyOS Forum, [https\://discuss.cachyos.org/t/intel-xe-drivers/11177](https://discuss.cachyos.org/t/intel-xe-drivers/11177)  
> 4. benchmarks-llama.cpp/docs/backend/SYCL.md at ... \- GitHub, [https\://github.com/Liquid4All/benchmarks-llama.cpp/blob/benchmarks/docs/backend/SYCL.md](https://github.com/Liquid4All/benchmarks-llama.cpp/blob/benchmarks/docs/backend/SYCL.md)  
> 5. Intel Arc Pro B70 (32GB) for Local LLMs: llama.cpp (SYCL/Vulkan, [https\://www\.reddit.com/r/LocalLLM/comments/1tr8znu/intel\_arc\_pro\_b70\_32gb\_for\_local\_llms\_llamacpp/](https://www.reddit.com/r/LocalLLM/comments/1tr8znu/intel_arc_pro_b70_32gb_for_local_llms_llamacpp/)  
> 6. Intel Arc : r/archlinux \- Reddit, [https\://www\.reddit.com/r/archlinux/comments/1ael0wb/intel\_arc/](https://www.reddit.com/r/archlinux/comments/1ael0wb/intel_arc/)  
> 7. Intel Arc 140V on Linux: the best GPU control panel apps and driver, [https\://botmonster.com/self-hosting/intel-arc-140v-linux-gpu-control-panel/](https://botmonster.com/self-hosting/intel-arc-140v-linux-gpu-control-panel/)  
> 8. Intel Arc Pro B70 32GB: Running Qwen3.6-35B on llama.cpp SYCL, [https\://sergiiob.dev/posts/intel-arc-pro-b70-sycl-llama-cpp-qwen35/](https://sergiiob.dev/posts/intel-arc-pro-b70-sycl-llama-cpp-qwen35/)  
> 9. intel-compute-runtime \- archlinux.de, [https\://www\.archlinux.de/packages/extra/x86\_64/intel-compute-runtime](https://www.archlinux.de/packages/extra/x86_64/intel-compute-runtime)  
> 10. intel-compute-runtime 26.35.39758.10-1 (x86\_64) \- Arch Linux, [https\://archlinux.org/packages/extra/x86\_64/intel-compute-runtime/](https://archlinux.org/packages/extra/x86_64/intel-compute-runtime/)  
> 11. AUR (en) \- llama.cpp-sycl \- Arch Linux User Repository, [https\://aur.archlinux.org/packages/llama.cpp-sycl](https://aur.archlinux.org/packages/llama.cpp-sycl)  
> 12. intel-graphics-compiler-legacy-bin \- AUR (en), [https\://aur.archlinux.org/packages/intel-graphics-compiler-legacy-bin](https://aur.archlinux.org/packages/intel-graphics-compiler-legacy-bin)  
> 13. How to Install and Configure Intel Graphics Drivers on Arch Linux, [https\://www\.siberoloji.com/how-to-install-and-configure-intel-graphics-drivers-on-arch-linux/](https://www.siberoloji.com/how-to-install-and-configure-intel-graphics-drivers-on-arch-linux/)  
> 14. AUR (en) \- intel-compute-runtime-legacy \- Arch Linux, [https\://aur.archlinux.org/packages/intel-compute-runtime-legacy](https://aur.archlinux.org/packages/intel-compute-runtime-legacy)  
> 15. Install Vulkan support for Intel graphics : r/archlinux \- Reddit, [https\://www\.reddit.com/r/archlinux/comments/hnz13i/install\_vulkan\_support\_for\_intel\_graphics/](https://www.reddit.com/r/archlinux/comments/hnz13i/install_vulkan_support_for_intel_graphics/)  
> 16. Intel | Discovery \- EndeavourOS Forum, [https\://discovery.endeavouros.com/intel/](https://discovery.endeavouros.com/intel/)  
> 17. Re: Arc B580 and X-Plane 12 \- Intel Community, [https\://community.intel.com/t5/Graphics/Arc-B580-and-X-Plane-12/m-p/1730274?profile.language=pt](https://community.intel.com/t5/Graphics/Arc-B580-and-X-Plane-12/m-p/1730274?profile.language=pt)  
> 18. Intel graphics \- ArchWiki, [https\://wiki.archlinux.org/title/Intel\_graphics](https://wiki.archlinux.org/title/Intel_graphics)  
> 19. I've recently been trying to get IPEX working as well, apparently, [https\://news.ycombinator.com/item?id=42501596](https://news.ycombinator.com/item?id=42501596)  
> 20. Llama.cpp \- ArchWiki, [https\://wiki.archlinux.org/title/Llama.cpp](https://wiki.archlinux.org/title/Llama.cpp)  
> 21. Intel Xe Arc drivers : r/linux\_gaming \- Reddit, [https\://www\.reddit.com/r/linux\_gaming/comments/1oq5z6b/intel\_xe\_arc\_drivers/](https://www.reddit.com/r/linux_gaming/comments/1oq5z6b/intel_xe_arc_drivers/)  
> 22. Enable Intel XE drivers for multiple cards \- Manjaro Linux Forum, [https\://forum.manjaro.org/t/enable-intel-xe-drivers-for-multiple-cards/175692](https://forum.manjaro.org/t/enable-intel-xe-drivers-for-multiple-cards/175692)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGsAAAAZCAYAAAA2VdDGAAAEUklEQVR4Xu2YW4hWVRTH/xllZYFFqURZoUgvPZQ9ZeBBiG5SSfRS0qBGib2Egt00sh40iIguQpJOoaBWokgqeQEfNErBIDWj60BEIUOWUQ+K5PrP2ptvnfXtc843880g2P7BnznfWvvsPef89/UAmUwm879noeg/o+/L6TZeRbn8LtHdLhb1ZriHHHK530SXuJjV/XrbkNiMcl3vl9Nt7EW5/NJStsxi0UofdPCd+Oep0uFwz6B4TfQztIJpLmfZCC1zTDTK5XpC7mMXJ6NFD4j+ET0qutTkboTeRwMvNPFuuEC0H/pMf6DcnuUG0ZfQ9le5XAq+3H7U/59fidaIrg2/+b+cgLZxdYjx/hdEv4ffg4K9aQW0wrdcLnKHaBG0DEeK52Fo7kOfMPzoA9AH4H19Lt4t7OHxmR5xuchzaD2TnQlS3ILWiJjhcpYfRGNdjB2R911uYheFeJ3xSWjWQ9CGjkMr8rwtuh7NZvX6hIH1e6JZP/lEl9CsqdC6N7lcZIuoQGdmLRF9BC37jstZjvoA0maRHaJxLtYIzXpQtBxa6cxyesD9T8J1k1mrfcKQWhOjWalR1w00iz18n+i06KpyGreKXkbnZh0UXQeti9NXakRcJtrqg8KvSJv1ruhmF2skmnUbtNJ15TTuE80P101m1S3o3/kAWmalRl03RLOegdb/VDk9sE7zRRVoNmsKdBSQbdDyfCedUmXWkIhmkSOiv1GuuFd0Zbjuxqy6kZUyshuiWVzoWf8ek+OiH0dAgWaznkersz4BLV83g3h+wQiZ9RK04p7wmw3EKZBUmTULzWalprqRNitesw2uuYQbBG7DSRFydWZ9gdbawk7LqZBre2oqTDHsZnGDQSZBK94efs8WPR6uSZVZNJs5blurGMrI4hR0pw92AA2Ks8GT0DY4Qgi36Vx/SBFyVWZNhJ7FLHEqvMvFqxgxswh70hnReOiOyTZSZda90Fzd1v0bH0CzWXNFc3ywA6xZ/HsKulPjTjeuP6RAvVn+w4FV0wE5MuxmcRqLxLMH4+tNnFSZxX/kL5TXBgvXjr0+iGaz3hPd44MdQLPsDpCdju3wMPq0iRchXmUWd5M3uRinRJrPLTnXvyaG3Sw7sjhFsPJ/oRsHS5VZhD2NXymu8QnhFdE8H0S9WVxzePrnNnuw2JFFHoO2cxLls00R4imzuMYd8MHAp9D7pvtEgmEziz2Do4fz+AQT5wjph37Di1wMbfRrE7OwJ7MnUnZBflH0Gdo/UZG4W+tz8clofbeL60unXCH6FrqD43WMcZfLl2zhusM2/EGX//cGtJePPAu97w2fcPD98lzGsv6sNyhS83ERcgtEH4Rr8ifay7L3esZARxhHJXdNPAosgx4YLXUfcr1SX1Sq8B9yqQg7JddAcnvIeXGW4RmKIzDG+C3Uwo7MNZ05PuNutJ/j1qK97qjPTbnzktdFOxvEF53JZDKZTCaTyWQy54qzCyBn5v0heE4AAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,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>
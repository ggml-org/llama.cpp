# LLVM OpenMP on Windows

`GGML_OPENMP_FETCH` builds LLVM's host OpenMP runtime, `libomp`, from pinned source archives. The option is used by the Windows LLVM CPU builds and is disabled by default.

## Release configuration

The fetched runtime is built as a shared, normal host runtime. The configuration deliberately disables optional diagnostics and tooling:

| Setting | Value | Reason |
| --- | --- | --- |
| `LIBOMP_ENABLE_ASSERTIONS` | `OFF` | Standalone libomp otherwise enables assertions by default. |
| `LIBOMP_STATS` | `OFF` | Do not collect runtime statistics. |
| `OPENMP_ENABLE_LIBOMP_PROFILING` | `OFF` | Do not add time profiling instrumentation. |
| `LIBOMP_USE_DEBUGGER` | `OFF` | Do not build the optional debugger interface. |
| `LIBOMP_OMPT_SUPPORT` | `OFF` | Do not build the OpenMP Tools interface. |
| `LIBOMP_OMPD_SUPPORT` | `OFF` | Do not build the OpenMP debugging interface. |
| `LIBOMP_USE_ITT_NOTIFY` | `OFF` | Do not add Intel ITT tracing hooks. |
| `LIBOMP_USE_HWLOC` | `OFF` | Avoid an external topology dependency. |
| `OPENMP_ENABLE_LIBOMPTARGET` | `OFF` | Build only the CPU host runtime. |
| `LIBOMP_FORTRAN_MODULES` | `OFF` | Do not require a Fortran compiler. |

CI builds the `omp` target with the Release configuration. With the Windows GNU-style Clang driver used by the llama.cpp toolchains, CMake's Release configuration supplies optimized code generation and defines `NDEBUG`.

The generated `omp.h` directory is a system include for llama.cpp targets. Warnings from LLVM sources are also disabled. These warning settings do not alter optimization or OpenMP behavior.

## Comparison with LLVM 20.1.8 packages

LLVM's official Windows packaging script builds OpenMP as part of the LLVM tree. The following comparison is based on the `llvmorg-20.1.8` release script and OpenMP CMake defaults.

| Property | Official x64 package | Official WoA package | `GGML_OPENMP_FETCH` |
| --- | --- | --- | --- |
| Build configuration | Release | Release | Release |
| Assertions | Off through `LLVM_ENABLE_ASSERTIONS=OFF` | Off through `LLVM_ENABLE_ASSERTIONS=OFF` | Explicitly off |
| LTO | Off | Off | Off in the release workflows |
| PGO-use flag | Enabled globally | Not enabled | Not enabled |
| Stats and profiling | Off | Off | Explicitly off |
| Debugger and OMPT/OMPD | Off on Windows | Off on Windows | Explicitly off |
| ITT notifications | On by default | On by default | Explicitly off |
| Adaptive x86 locks | Compiler feature detection | Not applicable | Compiler feature detection |
| Architecture selection | x86-64 target default | AArch64 target default | x86-64 target default or the llama.cpp ARMv8.7-A toolchain target |

The official x64 package generates a profile by using an instrumented Clang to compile part of Clang Sema, then passes the resulting profile to the complete package build. LLVM's CMake code applies the profile-use flag globally, so it also reaches the OpenMP subproject. The training workload does not execute libomp, however, so this is not a libomp-specific profile and there is no evidence that it improves libomp scheduling or synchronization. The official WoA package does not run the PGO stage.

LLVM's release script does not enable `LLVM_ENABLE_LTO`; its default is `OFF`. Enabling LTO or copying the x64 package's PGO setup would add build complexity without a demonstrated OpenMP benefit.

The official packages leave ITT notifications enabled through the libomp default. The fetched build disables them because llama.cpp does not use the ITT integration. This removes optional tracing code and cannot remove OpenMP parallel execution, affinity, barrier, or scheduling behavior.

## Runtime optimizations retained

Most important libomp performance behavior is implemented in the runtime and does not depend on how the source archive is obtained. The fetched build retains:

- Worker thread and hot-team reuse.
- Spin, yield and sleep behavior controlled by `KMP_BLOCKTIME` and `OMP_WAIT_POLICY`.
- Topology discovery and thread affinity controlled by `OMP_PROC_BIND`, `OMP_PLACES` and `KMP_AFFINITY`.
- Runtime selection of barrier and scheduling behavior.
- Compiler feature checks for x86 adaptive locks, RTM and WAITPKG support. Hardware-dependent paths remain runtime gated where required.

No build uses `-march=native` for the x64 release runtime. The WoA fetch build inherits `-march=armv8.7-a` from `cmake/arm64-windows-llvm.cmake`, which is also the baseline for the llama.cpp binaries in that package.

## Performance validation

Compare the fetched and official runtimes using the same llama.cpp build, model, thread counts, power policy and environment. Replacing only `libomp.dll` after linking is useful for isolating runtime behavior because both libraries expose the same LLVM OpenMP ABI.

Run `llama-bench` with at least one thread, the number of physical cores and the number of logical processors. Include prompt processing and token generation, use at least ten repetitions, and compare the distribution rather than only the fastest result. For example:

```powershell
llama-bench.exe -m model.gguf -t 1,8,16 -p 512 -n 128 -r 10
```

Also test repeated small parallel regions. Their barrier and wake-up costs are more sensitive to the OpenMP runtime than long matrix operations. Record `OMP_WAIT_POLICY`, `KMP_BLOCKTIME`, `OMP_PROC_BIND` and `OMP_PLACES`, because changing them can outweigh compiler-level differences.

For each LLVM update:

1. Update both source archive hashes.
2. Review changes to LLVM's Windows release script and libomp defaults.
3. Confirm the generated configuration keeps assertions, statistics and profiling disabled.
4. Build and run x64 on Windows x64.
5. Cross-build WoA and run the result on native Windows ARM64.
6. Benchmark against the corresponding official `libomp.dll` before changing optimization or instrumentation settings.

## Upstream references

- [LLVM 20.1.8 Windows release build script](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/utils/release/build_llvm_release.bat)
- [LLVM 20.1.8 OpenMP top-level CMake configuration](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/openmp/CMakeLists.txt)
- [LLVM 20.1.8 libomp CMake configuration](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/openmp/runtime/CMakeLists.txt)
- [LLVM profile and LTO option handling](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/llvm/cmake/modules/HandleLLVMOptions.cmake)
- [LLVM OpenMP runtime environment variables](https://openmp.llvm.org/design/Runtimes.html)

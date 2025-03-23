## llama-server-one
Based on [llama.cpp](https://github.com/ggml-org/llama.cpp).

Brad Hutchings<br/>
brad@bradhutchings.com

**THIS REPO IS NOT QUITE READY FOR PUBIC USE. I WILL DELETE THIS NOTICE WHEN IT IS READY.**

---
### Project Goals

The goal of this project is to build a single `llama-server-one executable` file that can run "anywhere":
- x86_64 Windows
- x86_64 Linux
- ARM Windows
- ARM Linux
- ARM MacOS

I am inspired by the [llamafile project](https://github.com/Mozilla-Ocho/llamafile). The main drawback of that project is that it has not kept up-to-date with llama.cpp and therefore, does not always support the latest models when llama.cpp supports them. Support for new models in llamafile takes work and time.

I want to use the MIT license as used by llama.cpp.

GPU support is not important to me and can be handled by platform specific builds of llama.cpp. CPU inference is quite adequate for many private end-user applications.

The ability to package support files, such as a custom web, UI into the executable file is important to me. This is implemented.

The ability to package default arguments, in an "args" file, into the executable file is important to me. This is implemented.

The ability to read arguments from a file adjacent to the executable file is important to me. This is implemented.

The ability to package a gguf model into the executable file is important to me. This is not implemented yet.

I welcome any of my changes being implemented in the official llama.cpp.

---
### Modifications to llama.cpp

To get this from the llama.cpp source base, there are few files that need to be modified:

1. [Makefile](Makefile) -- Extensive modifications to bring up to date, as it is deprecated in favor of a CMake system, and to support COSMOCC.

2. [src/llama-context.cpp](src/llama-context.cpp) -- COSMOCC doesn't have std::fill in its Standard Templates Library.

3. [examples/server/server.cpp](examples/server/server.cpp) -- Support embedded or adjacent "args" file, fix Cosmo name conflict with "defer" task member, add additional meta data to `model_meta`.

---
### Building llama-server

My start-to-finish guide for building `llama-server` with Cosmo is in the [Building-ls1.md](docs/Building-ls1.md) file.

---
### Packaging llama-server-one

My guide for packaging a `llama-server-one` executable is in the [Packaging-ls1.md](docs/Packaging-ls1.md) file.

---
### Reference
Here are some projects and pages you should be familiar with if you want to get the most out of `llama-server-one`:
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Georgi Gerganov and his team are the rock stars who are making the plumbing so LLMs can be available for developers of all kinds. The `llama.cpp` project is the industry standard for inference. I only fork it here because I want to make it a little better for my applications while preserving all its goodness.
- [llamafile](https://github.com/Mozilla-Ocho/llamafile) - `Llamafile` lets you distribute and run LLMs with a single file. It is a Mozilla Foundation project that brough the Cosmopolitan C Library and llama.cpp together. It has some popular GPU support. It is based on an older version of llama.cpp and does not support all of the latest models supported by llama.cpp. Llamafile is an inspiration for this project.
- [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) - `Cosmopolitan` is a project for building cross-platform binaries that run on x86_64 and ARM architectures, supporting Linux, Windows, macOS, and other operating systems. Like `llamafile`, I use Cosmo compile cross-platform executables of `llama.cpp` targets, including `llama-server`.

---
### To Do List

- A good way to manage a forked repo where we update from upstream and patch files as needed. I don't quite understand how to manage this. We should update llama.cpp.once regularly (monthly? weekly?) and as popular new models are supported.
- Package gguf file into executable file. The zip item needs to be aligned for mmap. There is a zipalign.c tool source in llamafile that seems loosely inspired by the Android zipalign too. I feel like there should be a more generic solution for this problem.
- GPU support without a complicated kludge, and that can support all supported platform / CPU / GPU triads. Perhaps a plugin system with shared library dispatch? Invoking dev tools on Apple Metal like llamafile does is "complicated".
- Code signing instructions. Might have to sign executables within the zip package, plus the package itself.
- Clean up remaining build warnings, either by fixing source (i.e. Cosmo) or finding the magical compiler flags.
- Copy the `cosmo_args` function into `server.cpp` so it could potentially be incorporated upstream in non-Cosmo builds.

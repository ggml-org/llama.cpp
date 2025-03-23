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

I am inspired by the llamafile project. The main drawback of that project is that it doesn't keep up-to-date with llama.cpp and therefore, does not always support the latest models when llama.cpp supports them. Support for new models in llamafile takes work and time.

I want to use the MIT license as used by llama.cpp.

GPU support is not important to me and can be handled by platform specific builds of llama.cpp. CPU inference is quite adequate for many private end-user applications.

The ability to package support files, such as a custom web, UI into the executable file is important to me. This is implemented.

The ability to package default arguments, in a .args file, into the executable file is important to me. This is implemented.

The ability to read llama-server-once-args from a file adjacent to the executable file is important to me. This is implemented.

The ability to package a gguf model into the executable file is important to me. This is not implemented yet.

I welcome any of my changes being implemented in the official llama.cpp.

---
### Modifications to llama.cpp

To get this from the llama.cpp source base, there are few files that need to be modified:

1. [Makefile](Makefile) -- extensive modifications to bring up to date, as it is deprecated in favor of a CMake system, and to support COSMOCC.

2. [src/llama-context.cpp](src/llama-context.cpp) -- COSMOCC doesn't have std::fill in its Standard Templates Library.

3. [examples/server/server.cpp](examples/server/server.cpp) -- support embedded or adjacent .args file, fix name conflict with "defer" task member, add additional meta data to `model_meta`.

---
### Building llama-server-one

My start-to-finish guide for building 'llama-server` with Cosmo and customizing a `llama-server-one` executable is in the [BUILD-INSTRUCTIONS.md](BUILD-INSTRUCTIONS.md) file.

---
### To Do List

- A good way to manage a forked repo where we update from upstream and patch files as needed. I don't quite understand how to manage this. We should update llama.cpp.once regularly (monthly? weekly?) and as popular new models are supported.

- Package gguf file into executable file. The zip item needs to be aligned for mmap. There is a zipalign.c tool source in llamafile that seems loosely inspired by the Android zipalign too. I feel like there should be a more generic solution for this problem.

- GPU support without a complicated kludge, and that can support all supported platform / CPU / GPU triads. Perhaps a plugin system with shared library dispatch? Invoking dev tools on Apple Metal like llamafile does is "complicated".

- Code signing instructions.

---
name: build-troubleshooting
description: >
  Troubleshooting guide for llama.cpp build issues on Windows with Vulkan backend.
  Covers locked DLLs, ExternalProject shader-gen quirks, environment setup, stamp
  file management, and common compilation errors. Use when builds fail, link errors
  appear, or shader generation behaves unexpectedly.
---

# llama.cpp Build Troubleshooting (Windows + Vulkan)

## Build Environment Setup

### The vcvarsall.bat Problem

`vcvarsall.bat` fails when invoked from bash/MSYS2. Set environment manually:

```bat
@echo off
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set INCLUDE=%MSVC%\include;%SDK%\ucrt;%SDK%\shared;%SDK%\um;%SDK%\winrt;C:\VulkanSDK\1.4.341.1\Include
set LIB=%MSVC%\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%MSVC%\bin\Hostx64\x64
```

Always use `cmd.exe //c "path\\to\\build.bat"` from bash, never try to set these in bash directly (breaks PATH, loses standard commands like `tail`).

### The Build Batch File Pattern

Use a `.bat` file with three stages:

```bat
@echo off
REM ... environment setup ...

echo === Step 1: Rebuild vulkan-shaders-gen ===
cd /d <build-dir>\ggml\src\ggml-vulkan\vulkan-shaders-gen-prefix\src\vulkan-shaders-gen-build
ninja -j 16 2>&1

echo === Step 2: Run shader-gen to regenerate shaders ===
cd /d <build-dir>\ggml\src\ggml-vulkan
vulkan-shaders-gen-prefix\src\vulkan-shaders-gen-build\vulkan-shaders-gen.exe ^
  --glslc "C:\VulkanSDK\1.4.341.1\Bin\glslc.exe" ^
  --input-dir "<source>\ggml\src\ggml-vulkan\vulkan-shaders" ^
  --output-dir "." --target-hpp "" --no-clean 2>&1

echo === Step 3: Build main targets ===
cd /d <build-dir>
ninja -j 16 bin/llama-bench.exe bin/llama-cli.exe 2>&1
```

## Common Errors and Fixes

### LNK1104: cannot open file 'bin\ggml-base.dll'

**Cause**: A running llama process holds a lock on the DLL.

**Fix**:
```bash
taskkill //F //IM llama-server.exe
taskkill //F //IM llama-bench.exe
taskkill //F //IM llama-cli.exe
```

Always kill ALL llama processes before building. Benchmarks can leave zombie processes. Check with:
```bash
tasklist 2>/dev/null | grep -i "llama\|ggml"
```

### static_assert failed: 'GGML_OP_COUNT != N'

**Cause**: Added a new GGML op but didn't update the static_assert count.

**Fix**: There are TWO static_asserts in `ggml/src/ggml.c` - one after the name table (~line 1052) and one after the symbol table (~line 1161). Update both from N to N+1.

### error C2078: too many initializers

**Cause**: Added an entry to the GGML_OP_NAME or GGML_OP_SYMBOL table at the wrong position or added a duplicate.

**Fix**: The entry must be at the exact position matching its enum value. Count from the top. Also check for duplicates if you're continuing from a previous session that may have already added the entry.

### Shader-gen "Error opening file for writing"

**Cause**: The `--target-hpp ""` flag or `--no-clean` flag interaction. This error from shader-gen step 2 is often harmless - the main ninja build (step 3) will re-run shader-gen as its first step anyway.

**Workaround**: Ignore the error if step 3 succeeds. The main build system manages shader-gen via ExternalProject and handles it correctly.

### Shader changes not picked up

**Cause**: Shader-gen is an ExternalProject with its own build directory. Changing `.comp` shader files doesn't trigger a rebuild of shader-gen itself.

**Fix for .comp changes** (shader content): The main build picks these up automatically via file timestamps.

**Fix for vulkan-shaders-gen.cpp changes** (adding new shaders): Must rebuild shader-gen separately:
```bat
cd /d build-win\ggml\src\ggml-vulkan\vulkan-shaders-gen-prefix\src\vulkan-shaders-gen-build
ninja -j 16
```

### Touch stamp files to skip shader-gen sub-build

When you only changed C++ code (not shaders), skip the slow shader-gen rebuild:
```bat
echo. > build-win\ggml\src\ggml-vulkan\vulkan-shaders-gen-prefix\src\vulkan-shaders-gen-stamp\vulkan-shaders-gen-build
echo. > build-win\ggml\src\ggml-vulkan\vulkan-shaders-gen-prefix\src\vulkan-shaders-gen-stamp\vulkan-shaders-gen-install
```

### std::chrono crash on MSVC

**Cause**: Using `std::chrono::high_resolution_clock` in ggml-vulkan.cpp can cause issues with MSVC.

**Fix**: Use `ggml_time_us()` instead for all timing in ggml code.

### warning C4003: not enough arguments for function-like macro

**Cause**: Known warnings in ggml-vulkan.cpp for IM2COL and CREATE_CONVS macros. Harmless, don't fix.

## Build Directories

| Directory | Purpose |
|-----------|---------|
| `build-win/` | Primary Windows Vulkan build |
| `build-hip/` | Windows HIP/ROCm build |
| `build-master/` | Master branch baseline for A/B comparison |
| `build-win/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build/` | Shader-gen ExternalProject build dir |
| `build-win/ggml/src/ggml-vulkan/` | Generated shader .cpp files output |

## Build Speed Tips

- Use `ninja -j 16` for parallel compilation
- Target specific binaries: `ninja -j 16 bin/llama-bench.exe bin/llama-cli.exe` instead of building everything
- Touch stamp files when only changing C++ (not shaders)
- Kill all llama processes BEFORE building to avoid link errors at the very end of a long build

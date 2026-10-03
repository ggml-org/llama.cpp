# llama.cpp Vulkan build on Fedora

This setup is for a Fedora machine using the Mesa/RADV Vulkan driver. The build script assumes the `llama.cpp` repository already exists and has been updated manually. It **does not run `git pull`**.

The script builds the current checkout with `GGML_VULKAN=ON`, creates a versioned standalone llama.cpp runtime bundle, copies the built ELF executables and llama.cpp shared libraries, sets `$ORIGIN` RPATH with `patchelf`, verifies key binaries, and prints the final bundle name/path.

## 1. Prerequisites

For the Fedora setup used here, install the build/runtime dependencies once:

```bash
sudo dnf install gcc-c++ cmake ninja-build vulkan-tools vulkan-loader-devel mesa-vulkan-drivers glslc spirv-headers-devel ccache patchelf
```

Verify Vulkan before building:

```bash
vulkaninfo --summary
```

For the RX 7900 XTX setup, the device should appear through the Mesa `radv` driver.

## 2. Configure the script

Open `build_llama_vulkan.sh` and adjust the variables near the top if your directories differ:

```bash
REPO_DIR="/home/tmatkovic/repos/llama.cpp"
OUTPUT_ROOT="/home/tmatkovic/llama.cpp/vulkan"
CLEAN_BUILD=true
OVERWRITE_BUNDLE=false
JOBS="$(nproc)"
```

`CLEAN_BUILD=true` removes the old CMake `build` directory before configuring. This avoids stale build outputs. `ccache` still provides a cache across builds, so repeated master builds can remain relatively fast.

Existing versioned runtime bundles under `OUTPUT_ROOT` are not deleted. A new bundle gets a name based on the binary's actual version/build output, for example:

```text
llama-0.5.0-dev-11369
```

The script refuses to overwrite a bundle with the same name unless `OVERWRITE_BUNDLE=true` is explicitly set.

## 3. Update llama.cpp manually

The script intentionally leaves source control to you. Update the repository first when desired, for example:

```bash
cd /home/tmatkovic/repos/llama.cpp
git pull
```

Then run the build script.

## 4. Run the build script

Make it executable once:

```bash
chmod +x build_llama_vulkan.sh
```

Run it:

```bash
./build_llama_vulkan.sh
```

The script will:

1. Check required commands and Vulkan availability.
2. Optionally remove the old CMake build directory.
3. Configure Ninja/CMake with `-DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON`.
4. Build llama.cpp using the configured number of parallel jobs.
5. Read the version/build number from the newly built `llama-server`.
6. Create a temporary versioned bundle under `/home/tmatkovic/llama.cpp/vulkan`.
7. Copy top-level ELF outputs from `build/bin`, including `llama-server`, `llama-cli`, `llama-bench`, other built tools, and llama.cpp shared libraries/symlinks.
8. Set each suitable ELF file's RPATH to `$ORIGIN`, so bundled programs find bundled llama.cpp libraries without `LD_LIBRARY_PATH`.
9. Verify `llama-server`, `llama-cli`, and `llama-bench`, check for unresolved libraries with `ldd`, and confirm that the Vulkan device list is available.
10. Write `BUILD_INFO.txt` into the bundle and publish the final directory only after verification succeeds.

At the end it prints output similar to:

```text
Bundle name: llama-0.5.0-dev-11369
Bundle path: /home/tmatkovic/llama.cpp/vulkan/llama-0.5.0-dev-11369
```

It also prints the `LLAMA_VULKAN_DIR` line you can paste into `~/.bashrc`:

```bash
LLAMA_VULKAN_DIR="/home/tmatkovic/llama.cpp/vulkan/llama-0.5.0-dev-11369"
```

Your Bash functions can then launch the server using:

```bash
"$LLAMA_VULKAN_DIR/llama-server" ...
```

## Notes

The bundle is self-contained with respect to the llama.cpp/ggml libraries produced by this build. Fedora system libraries and drivers such as glibc, libstdc++, `libvulkan.so`, and Mesa/RADV are intentionally **not** copied into it; those remain managed by Fedora.

The current upstream llama.cpp Vulkan build uses CMake with `GGML_VULKAN=ON`; the Vulkan backend requires Vulkan/glslc and SPIR-V headers. The standalone bundling/RPATH step in this script is a local packaging convenience rather than an upstream llama.cpp installation format.

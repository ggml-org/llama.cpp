# The Vulkan backend for Android(Termux)

## Setup

First, download and install the Termux app from its [GitHub releases page](https://github.com/termux/termux-app/releases/latest).

> [!IMPORTANT]
> Before proceeding to the next step, make sure you have switched the package sources to a mirror closest to you (e.g. via `termux-change-repo`). Otherwise, package downloads may be extremely slow or fail entirely. Additionally, before starting the build, run `termux-wake-lock` to acquire a wake lock on your device. Otherwise, Android may kill Termux or throttle it in the background, causing the build to fail unexpectedly. Note that the wake lock is only active for the current session; if you exit Termux or reboot your device, you must run this command again before performing another build.

Then update the Termux environment and install the dependencies.
```
$ apt update && apt upgrade
$ apt install git ccache cmake clang ninja vulkan-headers vulkan-loader-android vulkan-tools glslang spirv-headers spirv-tools vulkan-extension-layer shaderc -y
```

Check your environment.
```
$ vulkaninfo --summary
# Output something like this:
Devices:
========
GPU0:
        apiVersion         = 1.1.177
        driverVersion      = 32.1.0
        vendorID           = 0x13b5
        deviceID           = 0x90800011
        deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
        deviceName         = Mali-G77 MC9
        driverID           = DRIVER_ID_ARM_PROPRIETARY
        driverName         = Mali-G77 MC9
        driverInfo         = v1.r32p1-00bet5.f7b017e64923a972161adc84dd689d3f
        conformanceVersion = 1.2.6.0

```

But you at least need Vulkan 1.2. If you don't meet this requirement, you can check out the blog below, though stable operation isn't guaranteed.

## How to build

### Using build.py script (Recommended)

The easiest way to build llama.cpp is by using the `scripts/termux-vulkan/build.py` script. It will automatically perform native compilation for the SoC.

Build:
```
$ cd ./scripts/termux-vulkan
# You have to run the compile script in this directory
$ python build.py --release # Or --debug
```

### Manual CMake Build

You can also manually compile it after installing the dependencies, the method is the same as on desktop Linux.

## How to run

Use it the same way you usually use llama.cpp.

## Benchmark data

Please note that this data was measured on the low-speed path; for details, see [Blog](https://jetsontan.github.io/blog/llamacpp-on-old-vulkan-driver.html).

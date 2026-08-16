# ARM Mali Vulkan validation

This document describes the capability and validation path for ARM Mali Vulkan devices.
It does not treat a software Vulkan implementation as a Mali GPU emulator.

## Runtime capability report

Build llama.cpp with Vulkan support, then run:

```bash
GGML_VK_DEBUG=1 ./bin/llama-bench --list-devices
```

For an ARM Mali device, the Vulkan backend reports:

- ARM Mali architecture classification
- subgroup size
- FP16 and integer dot-product availability
- maximum compute shared memory
- ARM shader core rates from `VK_ARM_shader_core_properties`
- shader core count, warps per core, and mask from `VK_ARM_shader_core_builtins`
- cooperative matrix status

The ARM properties are queried only when the corresponding device extension is
advertised. Devices without the extension remain on the generic path.

References:

- https://docs.vulkan.org/refpages/latest/refpages/source/VK_ARM_shader_core_builtins.html
- https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM.html
- https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceShaderCorePropertiesARM.html

## Correctness validation

Run the Vulkan backend operation tests on the target device:

```bash
./bin/test-backend-ops test -b Vulkan0 -o ADD -j 1
./bin/test-backend-ops test -b Vulkan0 -o SET_ROWS -j 1
./bin/test-backend-ops grad -b Vulkan0 -o SET_ROWS -j 1
```

The Mali-G720 validation used for this change passed:

```text
ADD:            99/99 tests passed
SET_ROWS:       319/319 tests passed
SET_ROWS grad:  19626/19626 tests passed
```

## Model benchmark

Use the same model, backend, batch sizes, and repetition count for every
candidate tuning change:

```bash
./bin/llama-bench \
  -m /path/to/model.gguf \
  -ngl 99 -p 128 -n 128 -r 5 -o json
```

Record the device name, Vulkan API version, driver version, subgroup size,
shader compiler versions, build commit, model quantization, and all benchmark
samples. A single run is not sufficient for a tuning claim.

The backend keeps the generic 32-wide fallback for other architectures. For an ARM Mali device, the warptile configuration uses the device's actual subgroup size instead of forcing the 32-wide lower bound. This is guarded by both the ARM vendor architecture classification and the advertised ARM shader-core extension.

On the reference Mali-G720 device, two five-sample runs with the same SmolLM2 Q4_K_M workload measured the following ranges:

```text
prompt/matmul:  generic 154.27 tok/s -> ARM subgroup 162.34 tok/s (+5.24%)
generation:     generic  60.58 tok/s -> ARM subgroup  59.58 tok/s (-1.65%)
```

The generation result is not claimed as an improvement. More Mali devices and
larger model/batch matrices are still needed before tuning this path further.

```text
Mali-G720-Immortalis MC12
Vulkan API 1.3.247
ARM driver v1.r44p1
subgroup size 16
shared memory 32768 bytes
integer dot product enabled
cooperative matrix unavailable
external host memory unavailable
ARM shader core count 12
ARM warps per core 64
ARM rates: pixel=4 texel=8 fma=128
```

## Emulator and simulator boundary

The Android Emulator can run generic software Vulkan with:

```bash
emulator @AVD -gpu swiftshader_indirect
emulator @AVD -gpu lavapipe
```

It can also use `-gpu host`, which passes through the development host GPU;
that is not Mali emulation. These modes require a desktop emulator host and
do not reproduce the Android ARM Mali driver ABI.

Mesa PanVK is an open Mali-family Vulkan driver on supported Linux hardware,
but it is not the proprietary Android r44p1 driver used by the reference phone.
It may help test generic Mali-family shader behavior, but it cannot stand in
for this device's performance result.

References:

- https://developer.android.com/studio/run/emulator-acceleration
- https://developer.android.com/studio/run/emulator-commandline
- https://docs.mesa3d.org/panfrost.html

llvmpipe, SwiftShader, and Android Emulator software modes are useful for
Vulkan API and shader correctness checks, but they do not emulate ARM Mali
hardware, ARM shader-core properties, Mali scheduling, or Mali performance.
Android Emulator GPU modes likewise do not provide a Mali-G720 driver
contract.

Therefore:

- software Vulkan can validate generic fallback behavior;
- it cannot validate `VK_ARM_shader_core_properties` behavior;
- it cannot validate Mali subgroup or workgroup performance;
- real Mali devices remain required for Mali performance claims.

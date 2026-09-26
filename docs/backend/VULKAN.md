# Opt-in Adreno 750 matrix-vector compatibility candidate

`-DGGML_VULKAN_ADRENO_750_SHMEM=ON` enables an Android-only compatibility
candidate for a narrowly selected proprietary Adreno 750 driver and F32
matrix-vector prefill specialization. It is OFF by default. The selection
requires vendor `0x5143`, driver ID `8`, driver version `2150604839`, physical
name `Adreno (TM) 750`, the F32 subgroup-no-shared-memory shader, specialization
`64/1/5`, full required subgroups of64, and at least1280 bytes shared memory.
It substitutes the existing shared-memory reduction without changing other
shader selections. These properties do not uniquely identify a phone or OS
firmware. No numeric device ID is assumed.

This candidate is not a general Qualcomm subgroup workaround or a promise that
other prompt shapes, models, driver versions, or devices are qualified. Its
floating-point reduction order differs. Keep the option disabled unless the
exact artifact and intended workload have been qualified on the target device.

`test-vulkan-adreno-compat` checks guard boundaries and588 real tensor cases on
CPU by default. Passing an explicit backend name (for example `Vulkan0`) runs
that backend and fails if it is unavailable; it does not silently use CPU.
CPU passes do not qualify Vulkan. The cases cover F32 matrix-vector/matrix
shapes, odd/aligned K widths, surrounding column counts, and zero/one/two bias
adds. Driver fusion/rounding and actual shader attribution remain device gates.
Run `python3 tests/test-vulkan-adreno-routing.py --compiler c++` on the host to
compile the actual selection prefix and check pointer/length wiring under
Android ON/OFF and non-Android configurations. These stubbed routing tests do
not execute Vulkan.

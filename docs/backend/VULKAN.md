# Opt-in Adreno 750 matrix-vector compatibility candidate

`-DGGML_VULKAN_ADRENO_750_SHMEM=ON` enables an Android-only compatibility
candidate for a narrowly selected proprietary Adreno 750 driver. It is OFF by
default. The selection requires vendor `0x5143`, driver ID `8`, driver version
`2150604839`, and physical name `Adreno (TM) 750`. On that device, every
dequantize matrix-vector pipeline (`mul_mat_vec*`, `mul_mat_vec_id*`, all weight
types) is built with the existing shared-memory reduction instead of the
subgroup and hybrid reductions, and without a required full subgroup size, as
on devices without subgroup arithmetic. Other shader selections are unchanged.
These properties do not uniquely identify a phone or OS firmware. No numeric
device ID is assumed.

This candidate is not a general Qualcomm subgroup workaround or a promise that
other models, driver versions, or devices are qualified. Its floating-point
reduction order differs. Keep the option disabled unless the exact artifact and
intended workload have been qualified on the target device.

`test-vulkan-adreno-compat` checks guard boundaries and 588 real tensor cases on
CPU by default. Passing an explicit backend name (for example `Vulkan0`) runs
that backend and fails if it is unavailable; it does not silently use CPU.
CPU passes do not qualify Vulkan. The cases cover F32 matrix-vector/matrix
shapes, odd/aligned K widths, surrounding column counts, and zero/one/two bias
adds. Driver fusion/rounding and actual shader attribution remain device gates.
Run `python3 tests/test-vulkan-adreno-routing.py --compiler c++` on the host to
compile the actual matrix-vector reduction selection and check it under
Android ON/OFF and non-Android configurations. These stubbed routing tests do
not execute Vulkan.

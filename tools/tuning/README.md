# ggml-metal-tuning

Offline kernel tuner for the Metal backend.
It sweeps a kernel's config grid on the machine it runs on and prints pasteable table rows for `ggml/src/ggml-metal/ggml-metal-tuning.cpp`.

This is not a test: it never reports pass/fail on performance.
A non-zero exit code means bad arguments or a wrong environment (no Metal device, missing proc bridges), never a perf result.

| tuner | tunes | table |
|---|---|---|
| `fa-vec` | flash-attn vec `(Q, NE)` per `(dtype, head size, KV depth, batch width)` | `fa_vec_tuned_table` |
| `fa` | flash-attn (non-vec) `(Q, NSG)` per `(head size, KV depth)` | `fa_tuned_table` |

## Adding a device to the FA-vec table

Build on the target machine:

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --target ggml-metal-tuning -j
cmake --build build --target test-backend-ops -j
```

Sweep the grid (6 dtypes x 10 head sizes x 4 KV depths x 9 batch widths; a few hours):

```bash
./build/bin/ggml-metal-tuning fa-vec > fa_vec_rows.txt 2> fa_vec_sweep.log
```

`fa_vec_rows.txt` holds nothing but table rows: the min-max-regret target, the aggregate benefit gate, the short-KV drop and the pointwise compression are already applied.
The rows carry the SKU token the runtime reported, but `fa_vec_tuned_table` is keyed by Apple GPU family, so that column has to be retagged before the rows compile.
Your family number is on the `MTLGPUFamilyApple<N>` line the backend logs at init, near the top of `fa_vec_sweep.log`; `N` is the value, and the `MTLGPUFamilyCommon`/`MTLGPUFamilyMetal` lines beside it are not it.
If your log is the only sweep for that family, its rows become the family's segment; where the family already has rows, post the log and let the two be compared before anything is replaced.
A config represents a bucket only if it is no slower than the baseline config at every point that bucket covers, so a config that wins on average but loses at one batch width leaves its bucket at baseline.
`fa_vec_sweep.log` holds the per-cell timings, bucket coverage, noise floor, any cooldown activity, and every config the no-harm rule refused together with the point that refused it.
Post both, always: the rows now speak for every device in the family, so the log is what makes them reviewable.

Long sweeps can be split.
`--dtype f16,q4_0` and `--dk 128,192` restrict the grid, and the emitted rows for one `(dtype, head size)` do not depend on the others.
Concatenating the shard outputs in the order the full grid would visit them gives the same rows a single run prints.

Then validate the numerics, where Metal is compared against the CPU reference:

```bash
./build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b MTL0
```

This forces every legal `(Q, NE)` on `dk=128` and `dk=576`.
The tuner itself does no numerical checks, so the other head sizes have no automated numerical coverage.

If the device is not in `enum ggml_metal_device_id` yet, register it in `ggml/src/ggml-metal/ggml-metal-device.{h,m}` first.
The tuner emits whatever token the runtime reports for the machine, so an unregistered device emits `GGML_METAL_DEVICE_GENERIC` and its rows would apply to every unknown device.

## Adding a device to the FA table

```bash
./build/bin/ggml-metal-tuning fa > fa_rows.txt 2> fa_sweep.log
```

The sweep times the baseline tile against the wide tile with 4 and with 8 simdgroups at GQA 8, F16 K/V, over 8 KV depths (3 of them in the first bucket) x up to 4 batch widths per head size, and again with 8 query heads from 32 to 1024 tiles (30-45 minutes on M5, depending on how many cells are re-measured).
Unlike the FA-vec table, rows stay keyed by the SKU token the tuner emits: `tiles_min` depends on how many GPU cores the device has, which varies within a family.
A launch is counted in dispatched wide tiles: `ceil(batch/16) x query heads x streams`.
A KV-depth bucket gets a row only for a config that is at least 2% faster in aggregate over the launches of 1024 tiles or more and has no clear loss (more than 1.5%) at any of them, so a device where the wide tile does not pay off emits nothing and stays at baseline.
The last number of a row is `tiles_min`: the smallest sampled launch in that KV-depth bucket at or above which every small launch of full tiles wins and no partial-tile launch clearly loses. Below it the baseline tile is kept. A partial last tile also pays for the rows it pads, so it only raises `tiles_min` on a clear loss, never on a result near baseline.
A win must clear 1.5% and a loss must exceed it. A cell whose first measurement is within 3% of baseline is measured two more times, and the decision uses the median ratio over the three, so that both sides of the 1.5% cutoff rest on the same number of measurements. A cell that is neither a win nor a loss moves `tiles_min` up at a small full-tile launch; at a large launch its time simply stays in the aggregate that the 2% gate is applied to.
The 8-simdgroup variant replaces the 4-simdgroup one only when it is at least 3% faster, so that rows do not flip on noise.
Rows of one head size collapse into a single default row only when every bucket picks the same config and the same `tiles_min`.
`test-backend-ops test -o FLASH_ATTN_EXT -b MTL0` forces the wide tile regardless of the table.

## Thermal throttling

Long sweeps heat the GPU, and a throttled measurement is indistinguishable from a slow kernel.
The tuner re-measures a fixed baseline config every four candidates as an anchor.
When the anchor drifts more than `--cool-drift` (10% by default) from the coolest anchor seen in that cell, the tuner:

1. discards every candidate measured since the last clean anchor,
2. sleeps with exponential backoff until the anchor comes back within `--cool-eps` (3%),
3. re-measures the discarded candidates.

If it cannot cool down within `--cool-max-wait` seconds, or a cell needs more than `--cool-max-retry` rounds, that cell is dropped from the table and reported on stderr.

`--no-cooldown` only warns on drift and keeps the measurement.
Use it to reproduce a sweep taken without cooling.

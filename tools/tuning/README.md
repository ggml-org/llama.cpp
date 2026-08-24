# ggml-metal-tuning

Offline kernel tuner for the Metal backend.
It sweeps a kernel's config grid on the machine it runs on and prints pasteable table rows for `ggml/src/ggml-metal/ggml-metal-tuning.cpp`.

This is not a test: it never reports pass/fail on performance.
A non-zero exit code means bad arguments or a wrong environment (no Metal device, missing proc bridges), never a perf result.

| tuner | tunes | table |
|---|---|---|
| `fa-vec` | flash-attn vec `(Q, NE)` per `(dtype, head size, KV depth, batch width)` | `fa_vec_tuned_table` |

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

`fa_vec_rows.txt` holds nothing but table rows, ready to paste into `fa_vec_tuned_table`: the min-max-regret target, the aggregate benefit gate, the short-KV drop and the pointwise compression are already applied.
A config represents a bucket only if it is no slower than the baseline config at every point that bucket covers, so a config that wins on average but loses at one batch width leaves its bucket at baseline.
`fa_vec_sweep.log` holds the per-cell timings, bucket coverage, noise floor, any cooldown activity, and every config the no-harm rule refused together with the point that refused it.
Post both: the log is what makes the rows reviewable.

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

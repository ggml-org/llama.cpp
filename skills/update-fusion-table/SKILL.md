---
name: update-fusion-table
description: Regenerate a per-device test-fusion baseline (tests/fusion/MTL.csv, ...) after fusion patterns or the dummy-model set change. Use when test-fusion --check fails.
---

# Update a fusion baseline

`tests/fusion/<BACKEND>.csv` is the `test-fusion` golden baseline: `arch,moe,mode,label,count`,
where `count` is how many times a fusion label matched per graph run. See `tests/fusion/README.md`.

Re-record only when the counts moved for the reason the contributor intended. A row failing on
`nmse_fus` (> 1e-4) is a fused-vs-unfused correctness bug, not baseline drift - fix it, never
record it away.

## Steps

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON       # enable the target backend
cmake --build build --config Release --target test-llama-archs --target test-fusion -j

rm -rf build-ci-models && mkdir -p build-ci-models              # always regenerate
./build/bin/test-llama-archs -o build-ci-models

D=MTL0; B=tests/fusion/MTL.csv; T=/tmp/$(basename $B)   # keep the target backend/device in sync
./build/bin/test-fusion --models build-ci-models --device $D --check $B   # read the FAIL rows
./build/bin/test-fusion --models build-ci-models --device $D --record $T
diff $B $T
cp $T $B
./build/bin/test-fusion --models build-ci-models --device $D --check $B   # 0 failed, exit 0
```

- Read the diff before applying it. Every added, removed or changed row must be explainable by the
  change under review; an unexplained row means the run picked up something unintended.
- `expected 0` in a FAIL row = no baseline row for that label (new arch, or a fusion newly
  firing); non-zero `expected` = count changed; `fusion ... is no longer matched` = row stopped
  firing.
- `--record` rewrites the whole file, so never record a partial run (`--model FILE`, a filtered
  models dir) - it drops the rows of every architecture that was not run.
- `--record` marks every row `ok`; only the `--check` run validates.

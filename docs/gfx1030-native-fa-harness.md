# gfx1030-native FlashAttention harness

`scripts/benchmark-gfx1030-native-fa.py` is a guarded, repeatable harness for comparing stock and native gfx1030 FlashAttention.

## Safety behavior

The default invocation is a dry run. Actual GPU work requires both:

```text
--run --allow-gpu
```

A real run refuses to start if `llama-server`, `llama-cli`, `llama-bench`, or `test-backend-ops` is already running. Stop the active server before starting a run.

The harness never builds the tree and never disables FlashAttention. It controls only:

- `GGML_HIP_GFX1030_NATIVE` (unset for stock, `1` for native)
- `GGML_CUDA_DISABLE_GRAPHS` (unset for graphs on, `1` for graphs off)

It preserves relevant RCCL/HIP environment values in the manifest.

## Dry run

```bash
MODEL=/path/to/model.gguf
python3 scripts/benchmark-gfx1030-native-fa.py \
  --model "$MODEL" \
  --output-dir /path/to/gfx1030-native-fa-runs/dry-run \
  --profile
```

This prints the exact stock/native, graphs-on/off, correctness, benchmark, and focused rocprofv3 commands without touching the GPUs.

## Real run

After stopping any server and confirming the GPUs are free:

```bash
MODEL=/path/to/model.gguf
python3 scripts/benchmark-gfx1030-native-fa.py \
  --model "$MODEL" \
  --output-dir /path/to/gfx1030-native-fa-runs/$(date -u +%Y%m%d-%H%M%S) \
  --graphs both --repetitions 3 --prompt-sizes 512,4096,16384 \
  --profile --run --allow-gpu
```

Stock/native order alternates on each repetition. The harness runs the existing `FLASH_ATTN_EXT` backend correctness suite, captures `llama-bench` JSON in stdout logs, and optionally profiles a focused 256-dimension/4096-token case with `/opt/rocm/core-7.14/bin/rocprofv3`.

## Artifacts

Each run contains:

- `manifest.json`: command, environment, model, build, git, timing, and return-code metadata
- `commands.txt`: monitor-rerunnable command list
- `tests/`: correctness logs
- `bench/`: stock/native benchmark stdout/stderr logs containing JSON results
- `profiles/`: optional rocprofv3 traces and counters
- `system/`: before/after `rocm-smi` and `rocminfo` snapshots

Verify a completed run without using the GPUs:

```bash
cd /home/edwin/llama.cpp-rdna2
python3 scripts/verify-gfx1030-native-fa-run.py \
  /home/edwin/models/gfx1030-native-fa-runs/<run> --require-profiles
```

The verifier checks all command return codes, 12 `5/5` backend results, 12 four-case benchmark JSON files, and both stock/native profiler kernel dispatches.

The same harness is mirrored into the diagnostic worktree; diagnostic MTP/FA tracing remains source-controlled separately.
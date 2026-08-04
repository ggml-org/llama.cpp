# moe-qwen35b run state

## Status: PARTIAL - pipeline RUNNING in background, will produce artifact

The tessera quantization pipeline is running correctly and will produce a fresh
AWQ-unified artifact at `/tmp/qwen35moe-awq-unified.gguf` (~12 GB) in
approximately 60-70 more minutes from the timestamp below. The process is
nohup'd and survives agent exit.

## Exact invocation (the deliverable)

```bash
cd /Users/user/Developer/GitHub/tessera && \
./build/bin/llama-quantize \
  --tessera-imatrix "/Volumes/Julian T7/models/Qwen3.6-35B-A3B-Tile640-AWQ-refined.imatrix.gguf" \
  --tessera-awq-alpha 0.5 \
  --tessera-evolve-iters 4 \
  --tessera-evolve-population 8 \
  --tessera-evolve-islands 2 \
  --progress-file /tmp/ts-progress.jsonl \
  "/Volumes/Julian T7/models/Qwen3.6-35B-A3B-f16.gguf" \
  "/tmp/qwen35moe-awq-unified.gguf" \
  TESSERA_T640
```

Logs: /tmp/ts-stdout.log, /tmp/ts-stderr.log, /tmp/ts-progress.jsonl
PID file: /tmp/ts-run.pid (pid 89237 at launch)

## Pipeline entry point (reverse-engineered)

The tessera AWQ pipeline is NOT a subcommand. It is triggered by passing the
pseudo-ftype `TESSERA_T640` (or `TESSERA_T640_3D`) as the ftype argument to the
standard `llama-quantize` binary. The normal 2-arg form `<input> TESSERA_T640`
names the output `ggml-model-TESSERA_T640.gguf`; the 3-arg form
`<input> <output> TESSERA_T640` lets you name it. `tools/quantize/main.cpp`
-> `tools/quantize/quantize.cpp:llama_quantize` -> `ts_dispatch_run`
(`tools/quantize/tessera/tessera-dispatch.cpp:1040`).

The `--tessera-*` flags are parsed by `common/arg.cpp` (~line 4148+) into
`common_tessera_params`, surfaced via `common_get_tessera_params()`.

## DuckDB resume state

NO DuckDB store exists for this run (--quantize-db was NOT passed). The DuckDB
store only persists GA results and family warm-start seeds; the quantize phase
(step 7, the long part) has NO incremental checkpointing - it writes the entire
output GGUF in a single `gguf_write_to_file` call at the very end. So if the
running process is killed, the quantize phase restarts from scratch (the GA
phase would be faster on rerun if --quantize-db were used).

To make a future run resumable for the GA phase, add:
  --quantize-db /tmp/tessera-qwen35moe.duckdb

## Stages completed (as of last check)
1. SETUP - done (mmap'd 66 GB source via streaming-weight-load, no_alloc=true)
2. ga-prep (collect 391 GA layers from 753 tensors) - done in ~45s
3. ga-screen - done
4. ga-evolve (per-tensor alpha search, 391 tensors) - done in ~740s
5. quantize (write 753 quantized tensors) - IN PROGRESS, ~160/753, ~70 min ETA

The GA used iters=4/pop=8/islands=2 with the refined imatrix providing
per-channel act_scales. Family warm-start accelerated later tensors.

## Verification command (run after artifact appears)
```bash
./build/bin/llama-bench -m /tmp/qwen35moe-awq-unified.gguf -p 16 -n 1 -ngl 0
```

## Build / baseline notes
- HEAD: 45fec5098 on evolve-baseline/w3 ("restore untracked tessera WIP for
  buildable tree"). The tessera pipeline code is untracked WIP under
  tools/quantize/tessera/, present in this working tree.
- Baseline sha bbfc3493d is NOT an ancestor of HEAD; bbfc3493d is the wave-5
  g1 branch tip. The tessera WIP lives on evolve-baseline/w3.
- Build verified: `cmake --build build --target llama-quantize -- -j8` succeeds
  (one harmless -Wswitch warning about GGML_TYPE_TESSERA_T640 in ops.cpp).

## No code changes were made
This was pure pipeline-driving. No evolve-review branch needed.

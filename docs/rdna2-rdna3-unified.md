# Unified RDNA2/RDNA3 ROCm branch

This branch starts from `llama.cpp-rdna2` master and selectively ports the
qualified parts of the local RDNA3 fork. It is intentionally not a merge or
whole-commit cherry-pick.

## Provenance

- RDNA2 base: `d14c03343c03493eb5bee6df0aa38a3f8382b416`
- RDNA3 review tip: `1024b32c200d38bb3b6d95de0f78e0836590f3c8`
- common ancestor: `29be82a64991601a562e3db635c2f443acc8892e`
- reviewed RDNA3 core source: `961f1b30765cfa981a02a4323dc631eec01d0b0e`

The broad RDNA3 core commit also removed V620 P2P/RCCL policy, NVIDIA GB10
MMVQ policy, and several gfx1030 controls. Those removals are rejected here.
Current `common/spec_sidecar.*` and `src/spec_sidecar/` code supersedes the
older BridgeSpec sidecar directory from the RDNA3 fork.

## Porting matrix

### Shared and retained

- speculative CPU SGEMM dispatch keeps decode and small verification batches
  on a consistent accumulation path;
- flash-attention tile reuse barriers are correctness fixes shared by HIP
  architectures;
- current sidecar ABI, per-sequence state, target-stream attachment, provider
  probing, and stacked speculative implementations remain from RDNA2 master;
- current V620 P2P/RCCL tuner, fused-prefix all-reduce, gfx1030 MMVQ policies,
  and NVIDIA policies remain intact.

### gfx1030 / RDNA2 only

- `GGML_HIP_RDNA2_AUTO` and existing `GGML_HIP_GFX1030_*` policy;
- V620 topology-aware P2P/RCCL behavior;
- NVFP4 fast-scale decode and MMVQ six-row/width-eight policies;
- Q8_1 unary/matmul fusion. This remains structurally restricted to RDNA2.

### gfx1100 / RDNA3 opt-in

- `GGML_HIP_RDNA3_NATIVE=1`: enables the native gfx11 Q4_0 dot8 MMVQ path;
- `GGML_HIP_RDNA3_Q8_CACHE=1`: enables graph-scoped Q8 activation reuse,
  independently of the rejected RDNA3 unary-Q8 fusion;
- `GGML_HIP_RDNA3_Q8_CACHE_TELEMETRY=1`: cache diagnostics;
- `GGML_HIP_RDNA3_GDN_CHUNKED=1`: chunked GDN prefill;
- gfx1100-specific flash-attention launch configurations;
- Q4_K/Q5_K/Q6_K aligned LDS MMQ loads, compiled only into gfx11 builds;
- consistent RDNA3 MMVQ reduction widths for decode and speculative verify.

### Rejected or deferred

- `HSA_OVERRIDE_GFX_VERSION=10.3.0` on native gfx1100;
- `GGML_TP_SHARDED_OUTPUT` (not present in either final source tree);
- older `sidecars/bridgespec` duplication;
- reverted DFlash residual/output-trimming experiment;
- RDNA3 unary-MUL/Q8_1 fusion, which previously produced corrupt repeated
  tokens;
- experimental multi-op GDN/sibling/residual fusions until independent output
  parity tests pass;
- broad replacement of current MMVQ, CUDA dispatcher, or communication code.

All gfx1100 features are opt-in at runtime. The safe launch profile leaves
native, Q8 cache, and chunked GDN paths off. Promote a path only after comparing
fixed-seed output against that profile.

## Build

Build each architecture separately:

```bash
./scripts/build-rdna-unified.sh --arch gfx1100 \
  --rocm /opt/rocm/core-10.0 --jobs 2

# On a native gfx1030 host/build worker:
./scripts/build-rdna-unified.sh --arch gfx1030 --jobs 2
```

The script refuses an inherited `HSA_OVERRIDE_GFX_VERSION` for gfx1100. RCCL
is off unless explicitly requested with `GGML_HIP_RCCL=ON`. Embedded/prebuilt
Web UI assets are disabled so rebuilds do not fall back to an unpinned network
artifact; the OpenAI-compatible HTTP API remains available.

## Verified model

The server model is pinned to Hugging Face revision
`04a41723de3622e56bb499676ebaaacaa430f345`:

- `Qwen3.8-27B-Q4_0-AutoRound-Code.gguf`:
  `6f02e53c762a4a29a795a2346704c07f35c8a8ae7b74967aa1c0fda6bf047100`
- `mmproj-model.gguf`:
  `9da757136cb044abdf552334c56f2dcb63839799ea54c705ba4bcee807abdad2`

Do not start inference unless `SOURCE.txt` records that revision and both
hashes.

## Prepare current MTP sidecar assets

```bash
python3 -m venv .venv-spec-sidecar
. .venv-spec-sidecar/bin/activate
python -m pip install -e ./gguf-py numpy
python tools/spec-sidecar/prepare_assets.py mtp \
  --target "$HOME/models/Qwen3.8-27B-Q4-AutoRound-Code-GGUF/Qwen3.8-27B-Q4_0-AutoRound-Code.gguf" \
  --ids "$HOME/models/.manifests/qwen38-sidecar/draft_vocab_ids-c954724104a7856a07abb7031cc4af780ae7f5bf.json" \
  --output build-gfx1100-unified/bin/spec-sidecar-mtp
python tools/spec-sidecar/validate_assets.py mtp \
  build-gfx1100-unified/bin/spec-sidecar-mtp
```

This target does not embed `*.nextn.draft_vocab_ids`. The listed 40,960-ID
JSON is pinned to source commit `c954724104a7856a07abb7031cc4af780ae7f5bf`
and has SHA-256
`b64b6dfcf5441eb995ddf77d3d37b018e91b88c56ad1b4c5774ad8fbfac1c388`.
Its source metadata and a pinned Apache-2.0 license copy are under
`~/models/.manifests/qwen38-sidecar/`.

## Two-GPU launch

The launcher discovers GPU ordinals dynamically through AMD SMI and accepts
only the complete RX 7900 XT identity tuple. Unknown devices are logged and
skipped; BDFs and ordinals are not hard-coded.

```bash
# Validate command and identity without starting the server:
./scripts/run-qwen38-rdna-unified.sh --profile safe --dry-run

# Conservative target-only tensor smoke:
CTX_SIZE=8192 ./scripts/run-qwen38-rdna-unified.sh \
  --profile safe --no-sidecar

# Recommended stacked MTP + ngram-map-k4v:
./scripts/run-qwen38-rdna-unified.sh --profile safe

# Layer split remains available for faster startup/lower full-context VRAM:
./scripts/run-qwen38-rdna-unified.sh --profile safe --split-mode layer --kv-type q8_0
```

Profiles:

- `safe`: all new gfx1100 runtime paths off;
- `native`: native Q4_0 MMVQ on, Q8 cache and chunked GDN off;
- `experimental`: native MMVQ, Q8 cache, and chunked GDN on. It is retained
  for A/B work and is not the recommended profile.

The launcher defaults to true tensor mode with F16 KV and a `1,1` split. It
uses `draft-mtp,ngram-map-k4v`, N/M sizes 12/48, draft maximum 3, the first
dynamically matched GPU for the drafter, and a 4096 draft ubatch. Tensor mode
currently reports that its internal all-reduce cannot initialize and safely
uses the generic meta-backend butterfly fallback. Layer/Q8 mode remains a
supported fallback.

Sidecar KV is capped at 131,072 positions; requests beyond that ceiling fall
back to the authoritative target. Above 131,072 target context, the 1.76 GiB
vision projector is loaded but runs on CPU because F16 target KV leaves too
little VRAM to offload it. It deliberately sets `--cache-ram 0` while the host
has only 16 GB installed; do not restore a 65535 MiB RAM cache until stable
capacity is installed and tested.

## Initial two-GPU evidence

At 8,192 context on two 180 W RX 7900 XT cards, a deterministic 39-token code
response was byte-identical in safe, native, experimental, and MTP-sidecar
modes. Measured decode rates were:

- layer/Q8 safe target: 28.51 tokens/s;
- layer/Q8 native target: 20.58 tokens/s;
- layer/Q8 experimental native + Q8 cache + chunked GDN: 20.61 tokens/s;
- layer/Q8 safe target + MTP/ngram: 64.09-65.14 tokens/s;
- tensor/F16 safe target: 37.54 tokens/s;
- tensor/F16 safe target + MTP/ngram: 87.01-87.38 tokens/s after warmup;
- 262,144-context tensor/F16 + MTP/ngram with CPU projector: 64.25 tokens/s.

MTP accepted 30/33 draft tokens (0.90909) in each code run. Therefore the
production recommendation is the **safe tensor target plus current MTP
sidecar**. The native dot8/cache controls remain opt-in research paths; their
correct output does not justify enabling them on this Q4 AutoRound model.

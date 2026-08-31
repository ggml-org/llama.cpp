# speculative sidecar Qwen3.8-27B sidecars

This tree carries an optional, host-mediated speculative sidecar integration for the
Qwen3.8-27B MTP and DFlash2 drafters. The target model remains authoritative:
the sidecar proposes token IDs and the normal target verifier accepts or
rejects them.

The sidecars are intentionally **not** enabled by default. Runtime activation
requires the exact opt-in `SPEC_SIDECAR=1`; an unset value, `0`, or any other
value leaves the sidecar code dormant and preserves native speculative
selection. They are model-specific, support up to eight isolated sequences,
and support both keyed stochastic and greedy text inference. The host treats
them as stateful speculative implementations rather than stateless token
generators.

## Prepare the 27B artifacts

The preparation tools require `gguf-py` from this checkout and NumPy:

```sh
python3 -m pip install -e ./gguf-py
```

Obtain the 40,960-ID draft vocabulary from the matching, licensed source and
keep it outside the source tree. Then prepare the artifacts from a compatible
Qwen3.8-27B Q4_0 target and the matching DFlash2 Q4_K_M draft:

```sh
python3 tools/spec-sidecar/prepare_assets.py mtp \
  --target /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  --ids /absolute/artifacts/draft_vocab_ids.json \
  --output /absolute/artifacts/spec-sidecar-mtp

python3 tools/spec-sidecar/prepare_assets.py dflash \
  --target /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  --draft /absolute/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --ids /absolute/artifacts/draft_vocab_ids.json \
  --output /absolute/artifacts/spec-sidecar-dflash

python3 tools/spec-sidecar/validate_assets.py mtp /absolute/artifacts/spec-sidecar-mtp
python3 tools/spec-sidecar/validate_assets.py dflash /absolute/artifacts/spec-sidecar-dflash
```

The target must have Q4_0 token embeddings, a Q6_K output head, vocabulary
size 248,320, and one MTP block at index 64. The DFlash artifacts must provide
the 81-tensor Qwen3.8-27B DFlash2 schema. Do not mix an ID table, sliced head, and
weights from different preparation runs. The generated MTP `*-spec-sidecar.gguf`
is retained as a prepared derivative, but the sidecar-only path uses the target
for hidden-state extraction and does not create a native MTP draft context. It
uses the original full-vocabulary target because sliced native MTP-head loading is
not enabled in this integration yet.

## Prepare the Qwen3.6 35B-A3B MoE artifacts

`qwen35moe-mtp` is a separate compatibility provider for the Qwen3.6/Qwen3.5
MoE model identified by GGUF as `qwen35moe` (`35B-A3B`). It cannot reuse the
dense Qwen3.8-27B provider: the MoE target has a 2,048-wide hidden state, 40
trunk blocks plus one MTP block, 16/2 attention heads, and an 8-of-256 expert
MTP FFN. The preparation step converts the trained MTP block and output head to
the provider's Q4_0/F32 artifact layout and uses the validated 40,960-row draft
vocabulary ID table:

```sh
python3 tools/spec-sidecar/prepare_assets.py qwen35moe-mtp \
  --target /absolute/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  --ids /absolute/artifacts/draft_vocab_ids.bin \
  --output /absolute/artifacts/spec-sidecar-qwen35moe-mtp

python3 tools/spec-sidecar/validate_assets.py qwen35moe-mtp \
  /absolute/artifacts/spec-sidecar-qwen35moe-mtp
```

The MoE sidecar is currently an **explicit-path experimental compatibility
provider**. It is not selected merely because its DLL is beside
`llama-server`; this prevents an unqualified provider from replacing native
MTP. Set both variables below when deliberately testing it. Without them,
Qwen3.6 35B-A3B retains native MTP loading:

```sh
export SPEC_SIDECAR=1
export LLAMA_SPEC_QWEN35MOE_HIP_SIDECAR=/absolute/build/bin/spec_qwen35moe_mtp_sidecar.so
export LLAMA_SPEC_QWEN35MOE_HIP_WEIGHTS=/absolute/artifacts/spec-sidecar-qwen35moe-mtp
# The ID path defaults to $LLAMA_SPEC_QWEN35MOE_HIP_WEIGHTS/draft_head_ids.bin.
```

The current implementation is correctness/lifecycle validated but not yet a
production speed recommendation. Its full-vocabulary target output is reduced
to the 40,960 IDs in the artifact, so acceptance and throughput must be
qualified against native MTP on the user's exact model and prompt family.

## Build

Normal builds do not compile the sidecars. Enable them explicitly in a HIP
build and select the actual GPU architecture:

```sh
cmake -S . -B build-spec-sidecar \
  -G Ninja \
  -DGGML_HIP=ON \
  -DLLAMA_BUILD_SPEC_SIDECARS=ON \
  -DLLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES=gfx1030 \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
cmake --build build-spec-sidecar \
  --target spec-sidecar-hip-mtp spec-sidecar-hip-dflash llama-server
```

The resulting libraries are `spec_hip_sidecar.so`, `spec_dflash_sidecar.so`,
and `spec_qwen35moe_mtp_sidecar.so` in the build `bin` directory. For automatic
runtime discovery of the qualified providers, put the prepared bundles at
`bin/spec-sidecar-mtp` and `bin/spec-sidecar-dflash` beside those libraries (or
under the documented installed `share/llama.cpp/spec-sidecar` layout). The
Qwen35MoE bundle is intentionally explicit-path only until its end-to-end speed
and acceptance matrix is complete. These targets are optional because each
provider contains fixed model dimensions and is not a replacement for the
normal HIP backend.

## Run MTP

The default bundle layout needs only the master gate; provider paths are
optional overrides:

```sh
export SPEC_SIDECAR=1

./build-spec-sidecar/bin/llama-server \
  -m /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  -np 1 --no-context-shift \
  --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots
```

## Run DFlash2

When the DFlash sidecar probe succeeds, no DFlash GGUF model or draft context
is loaded by the host; the sidecar loads all prepared controller artifacts.
The matching GGUF may still be supplied when native fallback is desired, but it
is not needed for the sidecar-only path:

```sh
export SPEC_SIDECAR=1

./build-spec-sidecar/bin/llama-server \
  -m /absolute/models/Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 7 --spec-draft-p-min 0 \
  -np 1 --no-context-shift \
  --ctx-checkpoints 0 --cache-ram 0 --no-cache-idle-slots
```

## State, safety, and current limits

- Up to eight sidecar sequences are supported. Each sequence has an isolated
  logical cursor and KV namespace; KV storage is allocated lazily per active
  sequence. Larger `-np` values use the native drafter or target-only fallback.
- Greedy drafting uses `temperature=0` and `p_min=0`. For `temperature>0`,
  both sidecars sample from a compact top-k q distribution and return that q to
  the target residual verifier. The proposal RNG is a deterministic keyed
  stream derived from the request seed, sequence, position, sidecar kind, and
  draft step; target acceptance/rejection RNG remains owned by the main
  sampler. `p_min` is applied to the sampled q probability. The gfx1030
  Qwen35/MTP provider uses rocPRIM device top-k when its headers are available;
  otherwise it retains the portable two-stage device reduction. This provider-
  local optimization is independent of the Qwen4Exp sidecar.
- Text-only, contiguous positions are the supported sidecar input. Vision
  batches, unsupported interleaving, and migration disable the sidecar safely.
  With a single HIP target ubatch on the matching device, the host passes
  borrowed target device pointers and attaches the sidecar to the target HIP
  stream; the target context defers those host output copies until a host
  getter is requested. Otherwise it uses the synchronized host-copy path.
- The ABI exposes sequence-scoped `state_size`, `get_state`, `set_state`,
  `reset_state`, `truncate_state`, `commit_state`, and `rebase_state` for both
  sidecars. Snapshots contain only a position cursor plus an epoch; the large
  device KV cache is not serialized or copied. Target-derived rows are staged
  in pending KV and only the accepted prefix is copied into persistent KV.
  The speculative manager wraps state by implementation type, so stacked
  implementations cannot consume one another's state.
- Prompt and ordinary target rows are implicitly committed; target
  verification stages rows and acceptance commits only the accepted prefix.
  Checkpoint rollback discards pending rows and restores the cursor, slot reset
  starts a new epoch, and context shifting rebases committed device KV rows.
  Any failed update or restore enters target-only mode instead of guessing at
  state.
- N-gram and other speculative implementations may remain stacked with MTP
  or DFlash. A sidecar stages/commits target rows even when another
  implementation wins, so it can take over on a later round.
- Prompt-cache and external slot-file restore do not persist the sidecar's
  device KV contents. If a restored target state does not receive a complete
  contiguous sidecar prefill, the sidecar rejects the gap and the host uses
  target-only mode for correctness.
- The target verifier remains the correctness authority. A sidecar ABI/artifact
  probe runs before draft construction. A successful probe selects sidecar-only
  mode and avoids loading the host draft model/context; a later HIP
  initialization/runtime failure disables drafting and enters target-only mode
  rather than loading a late or potentially desynchronized native cache.
- Validate the activation log and artifact set before making performance
  comparisons. speculative sidecar's published numbers are Windows/RX7900 XTX
  research evidence, not RDNA2/Linux qualification.

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

## Build

The unified RDNA build script compiles the MTP and DFlash sidecars by default
(`BUILD_SIDECARS=ON`) and passes the selected architecture. Plain CMake builds
do not compile them; enable them explicitly in a HIP build and select the
actual GPU architecture:

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

The resulting libraries are `spec_hip_sidecar.so` and
`spec_dflash_sidecar.so` in the build `bin` directory. For automatic runtime
discovery, put the prepared bundles at `bin/spec-sidecar-mtp` and
`bin/spec-sidecar-dflash` beside those libraries (or under the documented
installed `share/llama.cpp/spec-sidecar` layout). The sidecar target is
optional because it contains fixed Qwen3.8-27B dimensions and is not a
replacement for the normal HIP backend.

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
  sampler. `p_min` is applied to the sampled q probability. The HIP MTP
  provider uses capability-gated rocPRIM top-k when available and retains the
  portable reduction fallback; this is compiled independently for gfx1030 and
  gfx1100.
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
  implementations cannot consume one another's state. DFlash committed KV
  storage starts at 16K rows and grows geometrically up to the hard
  `LLAMA_SPEC_HIP_MAX_POS` ceiling (131072 by default); growth recaptures the
  per-sequence graph once, and reset releases grown storage.
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
  probe runs before draft construction. A successful DFlash probe permits the
  target tensor-parallel output head to remain sharded; if the promised probe
  later fails, host-draft fallback is refused rather than risking a sharded-head
  mismatch. A successful probe selects sidecar-only mode and avoids loading the
  host draft model/context; a later HIP initialization/runtime failure disables
  drafting and enters target-only mode rather than loading a late or potentially
  desynchronized native cache.
- Validate the activation log and artifact set before making performance
  comparisons. speculative sidecar's published numbers are Windows/RX7900 XTX
  research evidence, not RDNA2/Linux qualification.

# Adaptive KV V2 upstream history replay

## References

- Original fork ancestor: `ece963f41b0b02d7a0d61436ae365762c073a4c8`.
- Original published V2: `05c0a5e454210ed18bf2486dfb7a6c7eba4b1cc7`.
- Pinned upstream base: `10a60cf303566e10d6a7a2774c17d2085503d87b`.
- Qualified port reference: `ac16245a776835f3656ad3fed98dba06230f89cc` on `integration/adaptive-kv-stream-v2-upstream-port`.
- Reconstructed branch: `rebase/adaptive-kv-stream-v2-upstream`.

The published V2 branch is not moved or pushed by this reconstruction. The original history, the reference snapshot and the saved future-model documentation stash remain available.

## Commit organization

The [commit map](docs/development/adaptive-kv-upstream-replay.csv) accounts for all 134 non-merge feature commits in dependency order. Original authors, author dates and titles are retained; commit bodies are omitted at the user's request. The three original merge wrappers are flattened, not the feature changes brought in by those merges. Replayed hashes differ because their parents now descend from pinned upstream.

Two small compatibility commits follow the memory foundation to adapt reservation state, unity-build grouping, Meta tests and the native batch API. The rebuilt foundation passes its allocator, buffer, arena, Meta and context tests before later consumers are replayed.

The original feature-development history is retained, including its earlier kernel implementations. Separate port commits then supersede those kernels and integrate updated upstream behavior:

1. Graph/batch lifecycle, two graph slots, visual MTP transport, recurrent history admission and construction-time MTP graph shape.
2. Private attention metadata outside stock parameter words, and CUDA graph source-type tracking.
3. Stock launch planning and checked resident/ring/wrap span geometry.
4. Updated MMA loading, quant conversion, dense pipelines and ordinary sparse behavior.
5. Stock-order vector/tile state resume and fused rounding.
6. Selected-pair factories, owner dispatch and fast aligned blocks with guarded tails.
7. Server UTF-8 repair that preserves token mappings and vision replay.
8. Matching-profile numerical/performance fixtures and restored architecture metadata checks.

Intermediate historical snapshots are not individually qualified upstream builds. The foundation checkpoint and the final reconstructed implementation are tested explicitly. Native sparse attention remains intact; unimplemented sparse streaming remains rejected. This port does not implement streamed Gemma or DFlash2.

## Tree audit

Compare the reconstructed branch with the reference without relying on a build's Git banner:

```sh
git diff --exit-code ac16245a7 rebase/adaptive-kv-stream-v2-upstream -- \
  common ggml include src tools/mtmd tools/server ':!tools/server/tests'
```

These runtime source paths must be identical. Intentional additions outside those paths retain the published graphs, vision HTTP harnesses, compatibility build documentation and the GPU-independent compiled-feature test. The obsolete tile-only device fixtures are replaced by the unified native/span/resume fixtures; their original source remains in history. README timings from the pre-port version are labeled historical.

The detailed qualification evidence and limits are in [the port roadmap](UPSTREAM_ATTENTION_PORT_ROADMAP.md). History replacement and any subsequent push require a separate user decision after the reconstructed branch is reviewed.

The final host build and 63 focused host tests pass (two GPU-only skips). The CUDA rebuild passes stock snapshot, span, resume and owner tests; all four real-model query widths pass 32-Ki/256-token checkpoint replay with zero final-logit and recurrent-state differences. Fresh MTP 0-3 server continuations retain the reference token IDs, grants and acceptance counts. Production is restored healthy after qualification.

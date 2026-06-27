---
---
## Design

### Overview

A minimal reference plugin for the deterministic-draft SDK. The plugin applies a regex filter (`^[A-Za-z]+$`) to draft tokens. It lives under `external/plugins/` and is built as part of the SDK build defined in `external/CMakeLists.txt`.

### Layout

- `external/CMakeLists.txt` — Root SDK build. Produces `libdeterministic_draft_spec.so`, copies headers to `external/include/`, and copies plugin `.so` artifacts to `external/lib/`.
- `external/plugins/deterministic_regex_plugin.h` — Plugin contract header.
- `external/plugins/deterministic_regex_plugin.c` — Implementation.
- `external/plugins/CMakeLists.txt` — Plugin build rules.

### Interface

- `deterministic_draft_create` → Allocate regex filter state.
- `deterministic_draft_destroy` → Free state; passing `NULL` harmless.
- `deterministic_draft_set_vocab` — Store token strings for regex matching.
- `deterministic_draft_fill_bitmask` — Apply regex against stored token strings.
- `deterministic_draft_commit` — No-op; regex stateless per step.
- `deterministic_draft_reset` — Clear per-slot state to enforce generation isolation.
- `deterministic_draft_get_capabilities` — BITMASK only.
- `deterministic_draft_get_jump_forward` — Unused; not supported.

### Runtime flow

   1. Host/Serviceloader loads `libdeterministic_draft_spec.so`, resolves SDK symbols via dlopen/dlsym.
   2. Host selects plugin (e.g. `--det-draft-model external/plugins/... `).
3. Host → `deterministic_draft_create`
4. Host → `deterministic_draft_set_vocab` (token strings registered)
5. Per step:
   - Host → `deterministic_draft_fill_bitmask` → plugin iterates vocab entries, regex-match, set valid bits
   - Host samples/drafts from constrained pool
   - Host → `deterministic_draft_commit` (no-op)
6. Host → `deterministic_draft_reset` between generations
7. Host → `deterministic_draft_destroy` at shutdown

### Regex rule

`^[A-Za-z]+$` — allows only purely alphabetical tokens. Anchored to avoid partial matches.

### Test flow

CLI invocation with draft types MTP/DSPARK/DFLASH/EAGLE3:
```
llama-server -m <model> --spec <type> --det-draft-model <path>
```
Every generated token must be alphabetical. Filter boundaries validated against common deterministic tokens ("0", "_", "1", etc.).

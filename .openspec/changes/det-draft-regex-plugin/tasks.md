---
---
## 1. Plugin scaffold

- [x] 1.1 Create `external/plugins/deterministic_regex_plugin.h` and `external/plugins/deterministic_regex_plugin.c`
- [x] 1.2 Wire `external/plugins/CMakeLists.txt` (add to SDK build, targets plugin library)

## 2. SPI implementation

- [x] 2.1 `deterministic_draft_create` / `deterministic_draft_destroy` — state lifecycle
- [x] 2.2 `deterministic_draft_set_vocab` — register token strings
- [x] 2.3 `deterministic_draft_fill_bitmask` — apply `^[A-Za-z]+$` regex against stored vocab
- [x] 2.4 `deterministic_draft_commit` — no-op
- [x] 2.5 `deterministic_draft_reset` — clear per-slot state
- [x] 2.6 `deterministic_draft_get_capabilities` — return BITMASK only

## 3. Integration test

- [x] 3.1 Build SDK library (`libdeterministic_draft_spec.so`) + plugin `.so`
- [x] 3.2 Verify `llama-server` can load and activate plugin via `--det-draft-model`
- [x] 3.3 Confirm alphabetical filter rejects non-alpha tokens (test "0", "_", "1" etc.)

## 4. Documentation

- [x] 4.1 Add `external/plugins/README.md` — how to build and use the skeleton plugin

## 5. Root README for external/

- [ ] 5.1 Create `external/README.md` with build instructions (llama.cpp → external SDK → plugin .so)
- [ ] 5.2 Create `external/README.md` with run instructions for each draft type (MTP, DSPARK, DFLASH, EAGLE3)

# Deterministic Regex Plugin

A minimal reference plugin for the deterministic-draft SDK. Applies a letter-and-whitespace predicate to draft tokens, constraining generation to letters, spaces, and newlines only.

## Files

- `deterministic_regex_plugin.h` — Plugin contract header
- `deterministic_regex_plugin.c` — Plugin implementation
- `CMakeLists.txt` — Build rules

## Usage

```bash
# Build SDK + plugin
cmake --build build/ --target deterministic_draft_spec_plugin

# Load plugin with llama-server
llama-server -m <model> --spec-type draft-mtp --det-draft-model external/plugins/lib/libdeterministic_draft_spec_plugin.so
```

## Compatibility

Works with draft types: MTP, DSPARK, DFLASH, EAGLE3.

## Filter

Filter: admits only letters and whitespace (space, newline, tab); rejects digits and punctuation.

## Declining rounds

The rule describes letters, so it has nothing to say about a numeric answer. Rather than truncating every draft in such a round -- which pushes the model into emitting rule content and constrains the bonus token too -- the plugin returns `DETERMINISTIC_DRAFT_REST_SKIP` and the host runs the round as plain speculative decoding.

The decision is the plugin's own, from its match rate over a rolling window. It starts with no opinion, turns enforcement on once a window of drafts is mostly rule-valid, and turns it back off when the rate collapses.

| Environment variable | Default | Meaning |
|---|---|---|
| `DET_DRAFT_REGEX_WINDOW` | `256` | Draft tokens per window. `0` disables the check (always enforce). |
| `DET_DRAFT_REGEX_MIN_MATCH` | `0.5` | Fraction of a window that must be rule-valid to enforce. |

The window is a proxy: a weak drafter also collapses the rate, and output that is rule-valid but semantically wrong keeps it high.

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

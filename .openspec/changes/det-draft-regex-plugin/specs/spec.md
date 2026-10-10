---
---
## Purpose

A minimal reference plugin for the deterministic-draft SDK, filter-constraining draft tokens via a regex rule, usable as a skeleton to prove end-to-end integration.

## ADDED Requirements

### Requirement: Plugin exposes required SPI symbols
Since the host loads via dlopen/dlsym.

#### Scenario: Fresh instantiation
- **WHEN** the host dlopens the plugin and calls `deterministic_draft_create()`
- **THEN** the plugin returns a non-null opaque `DeterministicDraftPlugin*` handle

#### Scenario: Cleanup
- **WHEN** the host calls `deterministic_draft_destroy(handle)`
- **THEN** the plugin frees all internal state; passing `NULL` is a no-op

### Requirement: Plugin provides bitmask-based token constraint
Declares which capability flags are supported.

#### Scenario: Capability query
- **WHEN** the host calls `deterministic_draft_get_capabilities(handle)`
- **THEN** the returned bitmask includes `CAPABILITY_BITMASK` and does not include `CAPABILITY_JUMP_FORWARD`

### Requirement: Vocabulary registration before filtering
Plugin needs tokenizer vocab to map ids to token strings.

#### Scenario: Vocabulary set once
- **WHEN** the host calls `deterministic_draft_set_vocab(handle, tokens, n_tokens)` after creation
- **THEN** the plugin stores a mapping of token id to token string and returns success

### Requirement: Filter keeps only alphabetical tokens
Core regex rule.

#### Scenario: Alphabetical-only formatting
- **WHEN** the plugin fills the bitmask via `fill_bitmask` for any slot
- **THEN** only token ids whose text matches the regex `^[A-Za-z]+$` are set as valid

### Requirement: Per-slot isolation
Multiple concurrent inference slots cannot interfere with each other.

#### Scenario: Multi-slot
- **WHEN** the host uses slot `slot_id=0` and slot `slot_id=1` simultaneously
- **THEN** each maintains independent state; committing tokens on one slot does not affect the other's `fill_bitmask` output

### Requirement: Draft-type compatibility
The plugin is usable with all existing speculative draft types.

#### Scenario: All draft types
- **WHEN** the host runs with draft types MTP, DSPARK, DFLASH, or EAGLE3
- **THEN** the plugin filters tokens correctly under each type

### Requirement: Reset restores initial state
Host calls reset between generations.

#### Scenario: Between generations
- **WHEN** the host calls `deterministic_draft_reset(handle, slot_id)`
- **THEN** the plugin clears internal state so the next `fill_bitmask` behaves identically to a fresh instance

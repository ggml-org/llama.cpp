# Verifiable Inference (commit-and-open) Protocol

This document specifies the **verifiable inference proof protocol** implemented in this repository’s `llama-server` integration and the reference verifier at `tools/server/vi_verify.py`.

This protocol is inspired by and adapted from the paper [Towards Verifiable AI with Lightweight
Cryptographic Proofs of Inference](https://eprint.iacr.org/2026/541.pdf) by Anchuri et al. (see that PDF for the motivating design, security discussion, and the broader “verifiable inference” framing). The implementation here is a practical, engineering-focused “commit-and-open over a GGML execution trace” variant.

The goal is to let a client obtain a **cryptographic transcript** that *probabilistically* attests the server executed a particular inference computation, by:

- **Committing** to a sequence of intermediate tensor outputs (a “trace”) via a Merkle root.
- **Opening** a small, sampled subset of those trace leaves (raw bytes), plus Merkle paths, so a client can recompute and check the commitment.

## Threat model and scope

- **What this proves**: integrity of a sampled subset of intermediate tensor outputs *for a specific inference execution*; the client checks that those outputs are consistent with a single Merkle commitment.
- **What this does not prove**:
  - That the model weights are the ones you expect (unless you separately attest the model artifact).
  - That the server used a specific hardware/implementation (only that the opened bytes match the committed bytes).
  - Full correctness of *all* trace nodes (only the opened subset).
- **Security level**: probabilistic (sampling-based). Increasing `samples` increases detection probability at the cost of larger openings and more I/O.

## Definitions

- **Trace leaf**: a GGML node output tensor observed during inference evaluation (via an eval callback).
- **Leaf hash**: SHA-256 hash over a canonical header + opened tensor bytes.
- **Merkle tree**: binary Merkle tree over the ordered list of leaf hashes.
- **Opening**: for a chosen leaf index, the server provides:
  - tensor metadata (name/op/type/shape/nbytes),
  - raw tensor bytes (via a URL),
  - Merkle path sibling hashes to recompute the root.

## Hashing primitives

All hashes are SHA-256.

### Leaf hash

Leaf hash must match `common/vi-proof.cpp` (and is mirrored in `tools/server/vi_verify.py`).

Given:
- `tensor.name` (string)
- `tensor.op` (string)
- `tensor.type` (string)
- `tensor.ne` (4 integers)
- `tensor.nbytes` (integer)
- `opened_bytes` (byte array)

Compute:

1. Construct the header bytes:
   - ASCII prefix `llama.cpp/vi-leaf/v1\0`
   - then `name\0op\0type\0`
   - then `"ne0,ne1,ne2,ne3\0"`
   - then `"nbytes\0"`
   - All encoded as UTF-8.
2. `leaf_hash_sha256 = sha256(header || opened_bytes)`

### Merkle node hash

For siblings `L`, `R` (each 32 bytes):

- `parent = sha256(L || R)`

Merkle proof recomputation is positional:
- If the current index is even: `cur = sha256(cur || sib)`
- If odd: `cur = sha256(sib || cur)`
- Then `idx //= 2`

## Sampling (which leaves are opened)

To avoid re-running inference, the server selects the set of leaves to open **up-front** from the client-provided `seed`.

- Let \(k = \) requested `samples` (clamped to \(\ge 0\)).
- During trace collection, the server maintains a **reservoir sample** of size \(k\) over the stream of collected leaves.
- Each opened record stores the final selected leaf’s:
  - `leaf_index` (0-based index in the collected trace order)
  - `name_occurrence` (0-based occurrence count among collected leaves with the same `tensor.name`)

This yields a uniformly distributed subset of size \(k\) without knowing `trace_leaves` in advance.

## Transport: async two-step download

Proof transport is **two-step** and **asynchronous**:

- **Step 1**: normal completion response returns quickly and includes only a pending proof handle.
- **Step 2**: client later fetches the proof manifest and opening bytes, then verifies locally.

### Step 1: request

Clients request a proof by including a `proof` object in the request JSON (OpenAI-compatible endpoints and native endpoints share this field).

Example:

```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false,
  "max_tokens": 8,
  "proof": {
    "enabled": true,
    "samples": 16,
    "seed": 123,
    "tensor_filter": [],
    "max_proof_bytes": 16000000
  }
}
```

Fields:
- `enabled` (bool): enable proof production.
- `samples` (int): requested number of openings \(k\).
- `seed` (uint32): sampling seed (and included in the transcript derivation).
- `tensor_filter` (array of strings): optional regex-like filters applied to tensor names in the trace callback.
- `max_proof_bytes` (int64): cap on total bytes written for openings (defensive limit).

### Step 1: response (fast path)

The completion response includes:

```json
{
  "... normal completion fields ...": "...",
  "proof": {
    "enabled": true,
    "proof_id": "opaque-id",
    "status": "pending"
  }
}
```

`proof_id` is an opaque identifier used for later retrieval.

### Step 2a: proof manifest

Client polls:

- `GET /vi/proofs/{proof_id}`

Responses:

- Pending:

```json
{
  "proof": {
    "proof_id": "opaque-id",
    "status": "pending"
  }
}
```

- Ready (manifest):

```json
{
  "proof": {
    "scheme": "commit-and-open over ggml node outputs (demo)",
    "proof_id": "opaque-id",
    "status": "ready",
    "trace_leaves": 3546,
    "merkle_root_sha256": "…",
    "challenge_seed_sha256": "…",
    "openings": [
      {
        "leaf_index": 89,
        "name_occurrence": 0,
        "tensor": {
          "name": "node_89",
          "op": "ADD",
          "type": "f32",
          "ne": [768, 28, 1, 1],
          "nbytes": 86016
        },
        "opened_bytes_sha256": "…",
        "opened_bytes_nbytes": 86016,
        "opened_bytes_url": "/vi/proofs/{proof_id}/openings/0",
        "leaf_hash_sha256": "…",
        "merkle_path_siblings_sha256": ["…", "…"]
      }
    ]
  }
}
```

- Error:

```json
{
  "proof": {
    "proof_id": "opaque-id",
    "status": "error",
    "error": "human-readable error string"
  }
}
```

### Step 2b: opening bytes

For each opening index `i` in `openings[]`:

- `GET /vi/proofs/{proof_id}/openings/{i}`
- Returns `application/octet-stream` of the raw tensor bytes for that opening.

## Client-side verification procedure

Given the **ready** manifest and downloaded opening bytes:

For each opening:
1. Download `opened_bytes` from `opened_bytes_url`.
2. Compute `sha256(opened_bytes)` and compare to `opened_bytes_sha256`.
3. Compute `leaf_hash` using the leaf hash definition above and compare to `leaf_hash_sha256`.
4. Recompute the Merkle root from:
   - `leaf_index`
   - `leaf_hash`
   - `merkle_path_siblings_sha256`
   and verify it equals `merkle_root_sha256`.

If all openings pass, the proof verifies.

### Reference verifier

`tools/server/vi_verify.py` implements the verification logic described above and is intended as a correctness reference for clients.

## Server-side storage and lifecycle

- Opened bytes are stored as **temporary on-disk files** under a server-managed proof directory.
- Proofs have:
  - a **TTL** (time-to-live),
  - a **global max-bytes** cache cap,
  - garbage collection that deletes proof directories on eviction/expiry.

Clients should treat `proof_id` as ephemeral: proofs may expire and become unavailable.

## Limitations (current)

- `stream=true` is currently not supported when requesting a proof (server returns an error).
- Some request shapes (e.g. multimodal) may be rejected for proof generation depending on server capabilities.

## Practical guidance

- **If you care about latency**: keep `samples` small (e.g. 1–16) and avoid overly broad `tensor_filter`.
- **If you care about assurance**: increase `samples` and consider fixing model artifacts via separate integrity mechanisms (model hash / artifact provenance).


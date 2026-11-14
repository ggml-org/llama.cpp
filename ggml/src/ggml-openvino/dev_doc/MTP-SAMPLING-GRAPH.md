# Stable multislot MTP graph

The captured UD draft graph contains 107 nodes regardless of which of four slots are active. It is the single MTP attention/MoE layer, not the recurrent trunk. Two speculative tokens require two consecutive draft calls, batched over active sequences.

```text
MTP token IDs + input hidden state
             |
  0-84: MTP layer and normalization, N active sequences
             |
     85: logits [248320, N]
     86: PAD adds one safe row
             |
             +-- 87-91:   slot 0 sampling
             +-- 92-96:   slot 1 sampling
             +-- 97-101:  slot 2 sampling
             +-- 102-106: slot 3 sampling
```

Each branch contains VIEW, two Reshapes, TopK (ten candidate IDs), and GET_ROWS (their scores). Ten is the candidate count, not the speculative-token count.

| Active slots | Sampling COMPUTE flags, slots 0-3 | Raw/current nodes | Old filtered nodes | Packed sampling_rows |
|---|---|---:|---:|---|
| 3 | `[0,0,0,1]` | 107 | 92 | `[0,0,0,0]` |
| 0,2 | `[1,0,1,0]` | 107 | 97 | `[0,0,1,0]` |
| All four | `[1,1,1,1]` | 107 | 107 | `[0,1,2,3]` |

Old counts apply the old COMPUTE filter to the captures. Unselected nodes remain in the raw cgraph. Previously, removing them changed translated topology, slot-specific row Parameter names and output ports; even equal-sized subsets such as `{0,1}` and `{0,2}` differed. This could trigger up to 15 draft variants, with compilation/import on first use or a missing-cache error in cache-only mode.

Now all pure top-k branches remain. Their names include stable graph ordinals even when inactive. One int64 `sampling_rows` Parameter supplies row indices at runtime. For `{0,2}`, the two logits rows belong to slots 0 and 2:

```text
sampling_rows = [0, 0, 1, 0]
slot 0 -> row 0 -> normal candidate buffers
slot 1 -> row 0 -> private scratch, ignored
slot 2 -> row 1 -> normal candidate buffers
slot 3 -> row 0 -> private scratch, ignored
```

The vector is not an active mask. Host-side COMPUTE flags select normal versus scratch output destinations. PAD provides a valid row when there are no logits. Inactive outputs use device USM on GPU and do not affect sampler state or acceptance. Other sampler chains retain their selected-branch semantics.

Only small sampling work executes for inactive slots. The MTP layer and output projection still process N active rows. Top-k is separate per branch, not yet one batched TopK. The observed draft interface has 10 inputs and 12 outputs: eight candidate ID/score outputs, hidden state, full logits and two KV Results.

Dump shapes use GGML order `[width, rows, dim2, dim3]`; OV reverses it. `view_offs` is in bytes: one logits row occupies `248320 * 4 = 993280` bytes. Buffer labels describe GGML storage, not the device executing imported GPU operations.

The captures used cache-only mode, four slots, context 8192 and the default two-token MTP settings.

# llama.cpp/examples/predicted

Demonstration of predicted output generation with recovery. See `patch.sh` for an example and `lookup.sh` for comparison with lookup decoding.

# Algorithm

- `n_past`: cumulative number of tokens sampled (including the prompt)
- `use_draft`: whether we're using the draft or not
- `id_last`: the last token that has been sampled
- `batch_idx`: current index in the current batch (-1 means we need to decode a new batch)

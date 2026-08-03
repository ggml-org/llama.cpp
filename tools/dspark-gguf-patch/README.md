# dspark-gguf-patch

Tools to prepare dspark draft models for llama.cpp's dflash loader.

The shipped `dspark_gemma4_12b_q4pure.gguf` (and any pre-PR-#25173 dspark draft)
needs three patches before llama.cpp can load it:

1. **Arch rename**: `general.architecture: dspark → dflash`. The upstream PR
   folded DSpark into DFlash — same graph, extra markov/conf tensors detected
   by presence. Use the legacy `dspark` arch name and the loader rejects it
   with `unknown model architecture: 'dspark'`.

2. **Tensor renames**: the original converter used dotted names; the
   canonical (post-#25173) names use underscores:
   - `markov.w1.weight` → `markov_w1.weight`
   - `markov.w2.weight` → `markov_w2.weight`
   - `confidence.proj.weight` → `conf_proj.weight`
   - `confidence.proj.bias` → `conf_proj.bias`

3. **MQA V injection**: gemma4 12B dspark is MQA (`head_count_kv=1`), so
   V == K. The original converter omitted V. The dflash loader now requires
   explicit V tensors, so we copy K's raw bytes into a new V entry at the
   end of the data section.

4. **Hparam prefix**: `dspark.*` hparam keys → `dflash.*` for the standard
   dflash hparams (keep `dspark.markov_*` etc. as dspark-specific).

5. **SWA disable**: the shipped .gguf has `dflash.attention.sliding_window=1024`
   but `sliding_window_pattern=[False]*n_layer`. The dflash loader's SWA
   enable path conflicts with the `is_swa_any()` assert at `create_memory`
   time. Set `sliding_window=0` to disable.

## Usage

```bash
# Step 1: rewrite dspark model (creates <name>_v2.gguf with renames + V injection)
python3 rewrite_dspark_gguf.py

# Step 2: disable SWA on the rewritten model
python3 disable_swa.py
```

Both scripts are idempotent and back up the input file the first time.

## Verifying the output

```bash
PYTHONPATH=$LLAMA_CPP/gguf-py python3 -c "
import sys
sys.path.insert(0, '$LLAMA_CPP/gguf-py')
from gguf import GGUFReader
r = GGUFReader('dspark_gemma4_12b_q4pure_v2.gguf')
print('tensors:', len(r.tensors))
for t in r.tensors:
    if 'markov' in t.name or 'conf_proj' in t.name or 'attn_v' in t.name:
        print(f'  {t.name}  shape={t.shape}  dtype={t.tensor_type.name}')
"
```

Expected: `markov_w1.weight`, `markov_w2.weight`, `conf_proj.{weight,bias}`,
and 5× `blk.{N}.attn_v.weight` tensors with the same shape as their
corresponding K.

## Running spec calibration

```bash
$LLAMA_CPP_BUILD/bin/llama-imatrix \
    -m /Volumes/Julian\ T7/models/unsloth-nonaut/gemma-4-12b-it-Q4_0.gguf \
    --model-draft dspark_gemma4_12b_q4pure_v2.gguf \
    --spec-steps 4 \
    --telemetry-out dspark_q40_acceptance.jsonl \
    -f $CORPUS \
    -c 512 -b 512 -ub 512 \
    -o dspark_q40_spec.imatrix.gguf
```

Telemetry JSONL conforms to `llama.tessera.spec.v1`:
`{seq_id, step_idx, prime_token, drafted, accepted, drafted_tokens[],
accepted_tokens[], confidence[]}` (top-k fields added when
`--telemetry-topk > 0`) — one record per spec step.

## Files

- `rewrite_dspark_gguf.py` — main patcher (renames + V injection + hparam prefix).
- `disable_swa.py` — secondary patcher (sets `sliding_window=0`).

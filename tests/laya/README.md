# Laya multilingual golden fixtures

Reference outputs produced by the PyTorch reference implementation
(`laya.agent.Agent`, requires the `laya` package and PyTorch) for the laya
multilingual checkpoint.

## Fixtures

`golden/<case>.json` — one file per fixed case. Each contains:

- `state` / `questions` — the exact inputs fed to `Agent.system_one`.
- `per_question.<qid>`:
  - `qtype` — `choice` / `score` / `noul`
  - `options` — rendered option texts in label-index order
  - `input_ids` — tokenized sequence (build_sequence output)
  - `marker_pos` — marker (option) positions in the sequence
  - `raw_logits` — scorer logits at the marker positions (pre temperature)
  - `act_logits` — raw action-head logits
  - `act_probability` — softmax(act)[0]
- `answers` — the post-processed `system_one` answer dict
  (choice/score/noul + probabilities + confidence + action).

`manifest.json` — case name -> file mapping.

## Cases

| case | qtype | lang | options |
|---|---|---|---|
| choice_single_zh | choice | zh | single (3 options) |
| choice_multi_zh  | choice | zh | multi (3 options) |
| score_zh         | score  | zh | 4 levels |
| noul_zh          | noul   | zh | 2 (false/true) |
| choice_single_en | choice | en | single (3 options) |
| score_en         | score  | en | 4 levels |
| noul_en          | noul   | en | 2 (false/true) |

The `zh` cases specifically exercise the Chinese path; the `en` cases cover the
English path. No case contains real brand or entity names.

## Regenerating

```bash
python3 tests/laya/gen_fixtures.py
```

Requires the `laya` package and PyTorch (set `LAYA_PY` if the interpreter is
not `python3` on `PATH`) and the multilingual checkpoint at `$LAYA_MODEL_DIR`.

## Quantization

Produce the F16 GGUF, then quantize with precision protection:

```bash
# 1. F16 GGUF
python convert_hf_to_gguf.py "$LAYA_MODEL_DIR" \
    --outfile laya-f16.gguf --outtype f16

# 2. quantize (builds llama-quantize if needed)
./tests/laya/quantize.sh
```

### Precision protection

The following tensor families are never quantized (kept F16, or F32 where the
conversion already emits F32 — norms and 1-D biases):

- `token_embd.weight`
- all `*_norm.weight` / `*_norm.bias` (encoder + decision-head LayerNorms)
- `type_emb.weight`
- `scorer.*` / `act_head.*`

Everything else (encoder `blk.*` matrices and the decision-head transformer
matrices `head.*.attn_qkv|attn_output|ffn_up|ffn_down`) is quantized with the
standard k-quant mixture. This is enforced via

```bash
llama-quantize --token-embedding-type f16 \
    --tensor-type 'type_emb\.weight=f16' \
    --tensor-type 'scorer\..*\.weight=f16' \
    --tensor-type 'act_head\..*\.weight=f16' \
    laya-f16.gguf laya-q4_k_m.gguf Q4_K_M
```

### Verification

`tests/laya/verify_quantize.py` checks tensor list/shape consistency, that no
protected tensor got quantized, and reports per-tensor deviation
(dequantized vs F16 reference):

```bash
PYTHONPATH=gguf-py "${LAYA_PY:-python3}" tests/laya/verify_quantize.py .
```

## End-to-end verification

`tests/laya/e2e.sh` runs the whole chain
(safetensors -> F16 GGUF -> k-quant -> inference -> checks):

```bash
./tests/laya/e2e.sh                 # full pipeline (idempotent, byte-reproducible)
./tests/laya/e2e.sh --skip-convert  # reuse laya-f16.gguf
```

### Precision regression

`tests/laya/verify_precision.py` runs `llama-laya-cli` on every golden case
for the F16 model and each quantization tier and reports the deviation of the
scorer logits, the answer probabilities, the decision fields
(choice / score / noul) and the action-head probability, against the PyTorch
reference and against F16 (the implementation-consistent baseline):

```bash
python3 tests/laya/verify_precision.py ./build/bin/llama-laya-cli
```

### Performance and stability

`tests/laya/bench.py` uses the CLI's in-process `--bench N` mode to isolate
the forward pass (model load excluded) and reports single-question and
4-question latency, batching behaviour, thread scaling and determinism:

```bash
python3 tests/laya/bench.py ./build/bin/llama-laya-cli --runs 30
```

### Demo

```bash
./build/bin/llama-laya-cli -m laya-f16.gguf -f tests/laya/demo_input.json -t 8
```

`tests/laya/demo_input.json` holds a dialogue plus `choice` / `score` /
`noul` questions; all three are answered from a single forward pass. The
stdout JSON contains `answers` (per question), the raw per-question tensors
(`per_question`) used for golden comparison, and, with `--bench`, a `bench`
timing block. `-t` sets the CPU thread count.

### Known deviations

* Known fix: the encoder attention output projection (`attn_output.weight`
  / `Wo`) was loaded but never applied in the encoder graph, which produced
  O(1) wrong scorer logits and wrong choices. It is now applied, and the
  multi-sequence marker gather is offset by each sequence's start
  (`seq_start`) so batched/multi-question forwards read the right tokens.
* Residual F16-vs-PyTorch scorer-logit deviation is 0.01-0.13 on the golden
  cases. The encoder attention is extremely saturated (pre-softmax scores up
  to ~55), so tiny summation-order differences are amplified; the decision
  (argmax) is unaffected on every case and tier.
* Quantization error is monotonic at the tensor level (max relative error
  ~3.85% / 2.16% / 0.38% for Q4_K_M / Q5_K_M / Q8_0). At the logit level it
  is not monotonic (a saturated attention position can swing), but
  choice/argmax agrees with PyTorch on all cases for all tiers.
* Batching several questions into one forward is not faster in this runtime:
  the flattened encoder attends over the concatenation of all sequences, so
  the dense `kq` matrix grows quadratically with total tokens. For a handful
  of questions, call the model once per question.


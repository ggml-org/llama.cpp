# Qwen35 three-dataset MTP calibration recipe

The locked pipeline is `scripts/build-qwen35-three-dataset-mtp-recipe.py`.
It downloads pinned Hugging Face revisions in parallel, samples locally with a
stable SHA-256 seed, globally deduplicates records, creates a deterministic
10% held-out split, allocates equal chunk budgets, collects per-source MTP
imatrices, merges them, and quantizes from BF16. It never overwrites a run and
does not delete artifacts.

## Full run

```bash
PY=/home/edwin/venvs/qwen-convert/bin/python
SCRIPT=/home/edwin/llama.cpp-rdna2/scripts/build-qwen35-three-dataset-mtp-recipe.py

$PY "$SCRIPT" \
  nvidia/Open-SWE-Traces \
  nvidia/Nemotron-SFT-Math-v4 \
  nvidia/ChatQA2-Long-SFT-data \
  --run-id qwen35-mtp-3datasets-i1000-b512-s20260808 \
  --batch-size 512 \
  --context-size 512 \
  --iterations 1000 \
  --seed 20260808 \
  --holdout-fraction 0.10 \
  --download-workers 3 \
  --candidate-records 4000
```

The 1000 chunks are split equally across the three IDs. Math is split equally
between COT/TIR; ChatQA2 is split equally between `long_sft` and
`NarrativeQA_131072`. The Open-SWE source is the `openhands/qwen35_122b`
subset, sampled across all 23 shards rather than only the first shard.

The first run writes `revisions.json`. To reproduce the exact source versions,
use the same command with a new `--run-id` and:

```bash
--revisions-file /home/edwin/models/qwen35-calibration-runs/qwen35-mtp-3datasets-i1000-b512-s20260808/revisions.json
```

## Fallback

With no dataset IDs, the command uses local `wiki.train.raw`, batch 512,
context 512, and 100 calibration chunks:

```bash
$PY "$SCRIPT" \
  --run-id qwen35-mtp-wikitext-fallback-i100-b512-s20260808 \
  --batch-size 512 --context-size 512 --iterations 100
```

Record clipping is automatic by default: the pipeline chooses a cap from the
iteration budget and keeps at least 64 candidate records per dataset. Naturally
short records are unchanged. Override it only when deliberately testing a
specific cap.

The run directory contains the manifest, pinned revisions, selected-record
manifests, calibration/held-out text, per-source imatrices, merged imatrix,
quantization overrides, logs, and the output model.
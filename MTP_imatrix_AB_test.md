# MTP head calibration: A/B test design

Status: run 2026-09-04, results below. `MTP_imatrix_AB_test.sh` reproduces the whole thing: imatrix, five quantizations, trunk hash proof, agreement runs, summary table. Every step skips when its output exists. Needs the `--process-mtp` patch in `tools/imatrix/imatrix.cpp` (commit 8de9abba1 plus the ubatch cap for the head context).

Done on 2026-09-04: `kld-base/qwen3.8-27b-imatrix-cal-mtp.gguf` (calibration_datav3.txt only, 129 chunks, head included) and the quant from it now lives as `qwen3.8-27b-q6_k-gateup-q5_k.gguf` (the served file). That file is variant B on the calibration corpus and `qwen3.8-27b-q6_k-gateup-q5_k-imat-cal.gguf` is its A, up to the 120-vs-129 chunk drift in the trunk; the steps below still describe the 1000-chunk corpus.

## Question

Does an importance matrix for block 64 (the MTP head of Qwen3.8-27B) make the quantized head draft better?
And if it does, how far down can the head go in bits before drafting gets worse?

## What cannot measure it

The trunk verifies every draft token, so the head cannot change what the model emits.
KLD, same-top-p and perplexity of the final output stay identical for a bf16 head and a garbage head.
The head is a speed knob only. The metric must be about drafting.

## Metric

Primary: **greedy agreement on a fixed text**. At temperature 0 a draft is accepted iff the head's argmax equals the trunk's argmax. Run both over the same tokens, teacher-forced, and count. With a fixed trunk and a fixed text this is deterministic and free of content drift between variants.

Report per variant:

| Field | Meaning |
| --- | --- |
| agreement | head argmax == trunk argmax, the depth-1 acceptance rate |
| proposed at p_min | share of positions where the head is confident enough to draft, and how many of those agree |
| trunk hits corpus, head hits corpus | sanity: the head should trail the trunk by a few points, not collapse |
| mean KL(trunk, head) | the same thing with the full distribution, for sampled decoding |

Agreement is the headline. The live server's accept rate mixes in `p_min` gating and chained drafts; see Step 4 for why it is not the measurement.

## Variants

All variants share the same trunk bytes: pure Q6_K, gate and up at Q5_K, quantized from the same bf16 file with the same imatrix file. Only block 64 differs.

| Variant | Block 64 | Imatrix for block 64 | Role |
| --- | --- | --- | --- |
| A | Q4_0 | no (`--exclude-weights blk.64.`) | baseline, same as the file in use today |
| B | Q4_0 | yes | the test |
| C | BF16 | not applicable | ceiling: total headroom |
| D | IQ3_S | yes | how low the head can go |
| E | IQ2_XS | yes | where it breaks |

D and E are where an imatrix changes a quant the most. If B is within noise of A, that is not a negative result for the imatrix; it means Q4_0 does not use it much. C tells whether there is anything left to gain at all.

Block 64 is about 425M parameters: roughly 240 MB at Q4_0, 180 MB at IQ3_S, 850 MB at BF16.

## Step 1: collect the imatrix with the head

Same corpus and chunking as `kld-base/qwen3.8-27b-imatrix-cal.gguf` (5120-token chunks, one sequence per batch), plus the head and the output tensor. One pass produces trunk, output and head entries together. No merge step. `imatrix-cal.sh` in the repo root holds this command followed by the two quantizations of Step 2.

```
taskset -c 0,2,4,6,8,10,12,14 \
    llama-imatrix \
    -m qwen3.8-27b-bf16.gguf \
    -f calibration_datav3.txt \
    -o kld-base/qwen3.8-27b-imatrix-cal-mtp.gguf \
    -dev CUDA0 -ngl 20 \
    -c 5120 -np 1 -b 5120 -ub 5120 \
    --no-ppl --process-output --process-mtp
```

`--process-output` is fine here because every variant is quantized from this one file, so `output.weight` is identical across variants by construction.

`-ngl 28`, the setting of the older files, ran out of VRAM with the head: the head context's ubatch is capped at the chunk length, so at 5120 both contexts hold a full-vocab f32 logits tile for 5120 rows on the device. `-ngl 20` fits. Placement changes summation order only.

Sanity checks:

```
gguf-dump --no-tensors kld-base/qwen3.8-27b-imatrix-cal-mtp.gguf | grep imatrix.chunk   # chunk_size 5120, chunk_count 24
gguf-dump kld-base/qwen3.8-27b-imatrix-cal-mtp.gguf | grep -c 'blk\.64\.'             # 16: 8 matmul weights x (in_sum2, counts)
```

`chunk_count` is the largest count in the file divided by the chunk size, and `output.weight` is fed by the trunk and by the head, so its count is doubled: 24 there, 12 (61440 tokens) on every other tensor. The head's 8 entries are its four attention projections, three FFN matrices and `nextn.eh_proj`; the norms and the shared head norm are not matmuls and have no entry.

Do not expect the trunk entries to match an older imatrix file, even one made with the same settings: CUDA reruns are not bit-stable, and a one-ulp change flips quantization choices. Every variant in this experiment must come from this one file. Check `imatrix.chunk_size` before mixing imatrix files at all; on 2026-09-04 two files from the same corpus turned out to differ in chunk length (5120 vs 512), which changed every trunk tensor.

## Step 2: quantize the variants

One type file per variant. Only the first line differs.

```
cat > tmp/quant-types-mtp-B.txt <<'EOF'
blk\.64\.=q4_0
\.ffn_gate\.weight$=q5_k
\.ffn_up\.weight$=q5_k
EOF
```

C uses `blk\.64\.=bf16`, D uses `iq3_s`, E uses `iq2_xs`. A uses the B type file with the imatrix excluded for block 64.

```
IMX=kld-base/qwen3.8-27b-imatrix-cal-mtp.gguf
SRC=qwen3.8-27b-bf16.gguf
OUT=qwen3.8-27b-q6_k-gateup-q5_k

llama-quantize --imatrix $IMX --pure --tensor-type-file tmp/quant-types-mtp-B.txt --exclude-weights blk.64. $SRC $OUT-mtpA.gguf Q6_K
llama-quantize --imatrix $IMX --pure --tensor-type-file tmp/quant-types-mtp-B.txt                            $SRC $OUT-mtpB.gguf Q6_K
llama-quantize --imatrix $IMX --pure --tensor-type-file tmp/quant-types-mtp-C.txt                            $SRC $OUT-mtpC.gguf Q6_K
llama-quantize --imatrix $IMX --pure --tensor-type-file tmp/quant-types-mtp-D.txt                            $SRC $OUT-mtpD.gguf Q6_K
llama-quantize --imatrix $IMX --pure --tensor-type-file tmp/quant-types-mtp-E.txt                            $SRC $OUT-mtpE.gguf Q6_K
```

Five files at about 20 GB each. Quantize is I/O bound here, so run them one after another.

## Step 3: prove the trunk is identical

Hash every tensor, drop block 64 and the file name, diff against A. Any difference outside block 64 voids the comparison.

```
for v in A B C D E; do
    python gguf-py/gguf/scripts/gguf_hash.py $OUT-mtp$v.gguf | grep -v 'blk\.64\.' | sed -E 's/  [^ ]+:/  /' > tmp/hash-$v.txt
done
for v in B C D E; do diff -q tmp/hash-A.txt tmp/hash-$v.txt && echo "$v trunk identical"; done
```

Older files are out of the comparison: `qwen3.8-27b-q6_k-gateup-q5_k-imat-cal.gguf` came from an imatrix without the output tensor, and `-imat-mtp-cal.gguf` from a 512-chunk one. Their trunks differ from A in every quantized matrix (checked 2026-09-04: 497 tensors differ, only norms, ssm_a, conv1d, dt bias and the embeddings match). A is the baseline.

## Step 4: measure agreement, teacher-forced

`llama-mtp-agree` (examples/mtp-agree) runs the trunk over a fixed text with every token as an output, then runs the head over the same tokens fed with the trunk state of the previous position, exactly as the speculative driver feeds it. At position k both predict token k+1, and a greedy trunk accepts the head's draft iff the two argmaxes agree. So the agreement rate is the depth-1 acceptance rate on a fixed text, with no sampling noise and no batch-shape effects. It also reports the proposal rate at `p_min` and the precision of proposed drafts, both argmaxes' hit rate on the corpus, and the mean KL divergence from trunk to head.

```
llama-mtp-agree -m $OUT-mtpA.gguf -f tmp/mtp-agree-val.txt -c 1024 -b 1024 -ngl 99 -fa on
```

The text is held out: 600 KB of the pile validation split (`zstdcat val.jsonl.zst | jq -r .text | head -c 600000`), about 150k tokens, none of it in the calibration set. About 6 minutes per variant on the 5090.

Depth-1 only: the live driver chains up to `n_max` drafts, and later positions run on the head's own previous outputs. The chain's acceptance is bounded by the depth-1 rate, so a variant that does not move depth 1 cannot move the chain.

### The live-server measurement, and why it is secondary

`tmp/mtp-ab-run.sh <letter>` sends ten prompts through the chat endpoint, greedy, 512 tokens, and `tmp/mtp-ab-compare.sh A B` sums the server timings. It was the first plan, and it failed its own identity check: every prompt's output diverged between A and B within a few hundred characters. Verification guarantees the same distribution in exact arithmetic, but the trunk verifies drafts in batches whose length depends on the head, and different batch shapes give slightly different CUDA logits, so near-tie argmaxes flip. Past the divergence the variants are scored on different texts, and per-prompt swings of 10 to 20 percent drown a sum over ten prompts. It stays as a smoke test of the whole stack, not as the measurement.

## Decision rule

Over 150k scored positions the agreement rate has a standard error near 0.001, so differences of 0.005 are real.

- B over A: agreement up by less than 0.005 means the imatrix does nothing useful for a Q4_0 head. Keep Q4_0 and drop the head pass.
- C over A: this is the ceiling. If C beats A by less than 0.01, the head is already solved at Q4_0 and only D and E are interesting.
- D and E: the lowest type within 0.01 of A becomes the production head type. The VRAM gain is small but real at 262k context.

## Results (2026-09-04)

151,256 positions of held-out pile text, one trunk (hash-proven identical), five heads. `./MTP_imatrix_AB_test.sh` reproduces the table.

| variant | block 64 | agreement | proposed at p_min 0.8 | precision | head hits corpus | KL(trunk, head) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Q4_0, no imatrix | 0.7183 | 0.3645 | 0.9721 | 0.5196 | 0.5627 |
| B | Q4_0 + imatrix | 0.7200 | 0.3658 | 0.9719 | 0.5203 | 0.5605 |
| C | BF16 | 0.7239 | 0.3699 | 0.9710 | 0.5214 | 0.5513 |
| D | IQ3_S + imatrix | 0.7018 | 0.3582 | 0.9686 | 0.5122 | 0.6235 |
| E | IQ2_XS + imatrix | 0.6330 | 0.3075 | 0.9632 | 0.4764 | 0.9175 |

Trunk hits corpus is 0.5756 in every run, as it must be.

All five runs reproduce bit for bit on a second pass, so the numbers are properties of the files, not of the run.

Paired test on the 148 chunks, each variant against A, difference in per-chunk agreement rate:

| variant | mean difference | standard error | t |
| --- | --- | --- | --- |
| B | +0.0017 | 0.0005 | +3.5 |
| C | +0.0056 | 0.0005 | +10.7 |
| D | -0.0165 | 0.0007 | -24.4 |
| E | -0.0853 | 0.0013 | -65.7 |

Chunks are the unit, so within-text correlation is inside each observation and not assumed away. Pairing removes the text's own variance, which is why the standard error is three times smaller than the independence estimate over positions.

Read against the decision rule:

- **The imatrix is measurable and useless.** B over A is +0.0017, seven standard errors from the 0.005 line and a sixth of a percent of drafts. Keep Q4_0 and drop the head pass from the pipeline.
- **The whole head headroom is half a percent of drafts.** C over A is +0.0056, which with a geometric chain is roughly 3.55 against 3.62 expected tokens per step, a two percent decode effect at the ceiling.
- **IQ3_S costs three times the Q4_0 penalty** for about 60 MB of VRAM. IQ2_XS collapses. Q4_0 stays the production head type.

## Noise and confounds

- Acceptance at temperature 0 has no sampling noise. Do not average over seeds, there is nothing to average.
- Throughput is noisy on this box because of user activity. Quote it, do not decide on it.
- Prefill is shared and out of scope. Only decode changes.
- `-c 5120` chunks mean the head sees at most 5119 tokens of context per row during calibration, the same limit the trunk imatrix has.
- The KV cache types of the draft context apply to block 64's attention. A lossy V cache lowers acceptance for every variant alike and shrinks the headroom the imatrix can show. Measure with 16-bit draft K and V; bring the quantized draft cache back once the head type is settled.
- Serve greedy. At temperature 1.0 a draft counts as accepted only when it equals the sampled token, which adds the trunk's entropy as noise on top of the head's precision.

## Reproducing the calibration

The dataset is public, but the imatrix depends on how it is fed. Same text, different chunking, gives a different imatrix and a different quant in every matrix. Anyone repeating this study needs the whole recipe, not just the file name.

| Item | Value |
| --- | --- |
| dataset | `calibration_datav3.txt` (Bartowski's calibration set), 279,515 bytes, 2481 lines |
| sha256 | `200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026` |
| model | `qwen3.8-27b-bf16.gguf`, converted from the bf16 safetensors with `convert_hf_to_gguf.py` |
| chunking | `-c 5120 -np 1 -b 5120 -ub 5120`: 5120-token chunks, one sequence per batch, 12 chunks, tail dropped |
| placement | `-dev CUDA0 -ngl 20`, remaining layers on CPU, P-cores pinned with `taskset -c 0,2,4,6,8,10,12,14` |
| flags | `--no-ppl --process-output --process-mtp` |
| tool | `llama-imatrix` from this tree with the `--process-mtp` patch (commits 8de9abba1, 09c106858) |

Check the result against the metadata the tool writes: `imatrix.chunk_size` 5120, `imatrix.chunk_count` 12, `imatrix.datasets` naming the file. If these differ, the imatrix is a different one, whatever the corpus.

Placement and pinning do not change the statistics in exact arithmetic, but they change summation order, so a re-run on other hardware will not reproduce the file bit for bit. Expect the quantized tensors to differ slightly and the KLD and acceptance numbers to move within noise. Comparisons in this study hold only between variants quantized from one and the same imatrix file.

## Done: teacher-forced head tool

`examples/mtp-agree` is the tool sketched here at first: the imatrix head feed with the collector swapped for argmax and KL comparison against the trunk. A comparison against a bf16 head is variant C, run through the same tool.

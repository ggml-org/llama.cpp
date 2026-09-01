# Self-Speculative Biased Decoding

Reference implementation of [Self-Speculative Biased Decoding for Faster
Re-translation](https://arxiv.org/abs/2509.21740) (Zeng, Deng, Shu, Wang), as a
llama.cpp example.

Re-translation is the simplest way to translate speech as it arrives: every time
a few more words of source show up, translate the whole prefix again. It is easy
to reason about and it gets the final sentence right, but it is wasteful, because
each answer is close to the one before it and is decoded from scratch anyway.

SSBD reuses the previous answer as a speculative draft, checks it in a single
forward pass, and resumes decoding from the first place the model disagrees. No
draft model, no second set of weights: the target model checks its own earlier
output.

Three mechanisms, which pay off in different places:

1. **Draft reuse.** The previous answer is checked in one forward pass. Tokens
   the model agrees with are kept, the rest are dropped from the KV cache.
   This buys throughput.
2. **Biased verification.** While checking, the probability of the expected
   token is raised to `p = (1 - beta)*p + beta`, so the model is nudged to stand
   by what it already said. This buys throughput and stability, at a quality
   cost that is small at low beta.
3. **Holding back the tail.** The last few tokens of a partial answer lean on
   source that has not arrived. Withholding them, or merely not showing them,
   removes most of the remaining flicker.

## Quick start

```sh
cmake -B build && cmake --build build --target llama-self-spec-bias

# put a gguf where the preset looks for it
ln -s /path/to/towerplus-2b-Q4_K_M.gguf examples/self-spec-bias/bench/models/towerplus-2b-Q4_K_M.gguf

bash examples/self-spec-bias/bench/run-towerplus.sh            # all 1012 sentences
QUICK=1 bash examples/self-spec-bias/bench/run-towerplus.sh    # 20 sentences
```

That fetches FLORES-200 devtest, cuts each sentence into growing prefixes,
runs every setting, and scores each for speed, draft acceptance, output
stability and translation quality.

To drive the decoder directly:

```sh
llama-self-spec-bias -m model.gguf -f source.txt --stream-interval 3 \
    --draft-bias-beta 0.2 \
    --in-prefix "<start_of_turn>user
Translate the following English source text to Chinese.
English: " \
    --in-suffix "
Chinese: <end_of_turn>
<start_of_turn>model
"
```

`source.txt` holds one complete sentence per line. Each line is expanded into
growing word prefixes, so a 12 word line with `--stream-interval 3` becomes 4
requests. The prompt for each request is `--in-prefix` + partial source +
`--in-suffix`. Only greedy sampling is supported, so `--temp` must be 0.

## Results

FLORES-200 devtest, all 1012 sentences cut every 3 words into 7628 requests,
English to Chinese, TowerPlus 2B Q4_K_M on Metal with Apple Silicon M4 Pro, `-n 200`:

| Setting | flags | accepted | output | erasure | BLEU | chrF | COMET |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `no-cache` | `--no-draft-reuse --no-prompt-cache-prefix` | n/a | 76.83 t/s | 1.43 | 42.51 | 36.69 | 0.8829 |
| **`baseline`** | `--no-draft-reuse` | n/a | 76.83 t/s | 1.43 | 42.51 | 36.69 | 0.8829 |
| `greedy-verify` | `--draft-bias-beta 0` | 58.5% | 98.41 t/s | 1.43 | 42.54 | 36.58 | 0.8826 |
| `bias-02` | `--draft-bias-beta 0.2` | 76.1% | 111.27 t/s | 0.86 | 42.68 | 36.61 | 0.8823 |
| `bias-03` | `--draft-bias-beta 0.3` | 81.3% | 116.82 t/s | 0.69 | 41.63 | 35.62 | 0.8776 |

Reading down the table, each mechanism buys something different:

- **Prefix caching** saves prompt evaluation, so it moves time to first token
  but not the output rate. `baseline` against `no-cache` is flat on every
  metric here.
- **Draft reuse** buys output rate, 76.8 to 98.4 t/s, and nothing else. With a
  greedy check the text follows greedy decoding, so quality and stability do
  not move.
- **The bias** buys stability. Erasure falls from 1.43 to 0.86 at beta 0.2,
  meaning the answer is rewritten far less as the source grows, which is what a
  reader of live output actually notices. It also raises acceptance, hence the
  further speed gain.

Quality holds until it does not. COMET is flat through beta 0.2 and starts to
slip at 0.3, where BLEU and chrF drop as well.

## Choosing beta

For one biased token the draft token is picked when

```
p_max - p_bias < beta / (1 - beta)
```

so `beta = 0.2` overrides the model whenever the gap is below 0.25.

**`beta >= 0.5` is not a bias, it is enforcement.** At `beta = 0.5` the bound is
1.0 and the gap can never exceed 1.0, so the draft is always accepted and the
model has no say. The useful soft range is `0 < beta < 0.5`.

With a high beta the model can no longer fix word order, which matters for
language pairs that reorder: an English sentence ending in "last week" becomes
Chinese that should start with the time phrase, and a forced draft strands it at
the end.

## Holding back the tail

The tail of a partial answer is its least reliable part. Two settings keep it
from the reader, and they are alternatives, not layers:

- `--output-mask-k N` really withholds the last N tokens of a partial answer.
  They are not shown and not drafted from either, so the model decides them
  again with more input. The last answer of a line is sent whole. **This is
  mask-k in the usual sense**: when a re-translation paper says mask-k, this is
  the one it means.
- `--display-mask-k N` in `bench/score.py` leaves decoding alone and only hides
  the tail when erasure is measured, modelling a reader shown all but the last
  N tokens. This is the display only variant the paper proposes.

Withholding the tail costs speed, because a token that is not sent cannot be
drafted from on the next step and has to be decoded again. Hiding it does not.
TowerPlus, beta 0.2, 20 sentences:

| policy | k | drafted | accepted | of output | output | erasure | COMET |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `--output-mask-k` | 0 | 3331 | 2572 | 63.6% | 127.5 t/s | 0.98 | 0.8709 |
| `--output-mask-k` | 3 | 2854 | 2384 | 59.1% | 124.5 t/s | 0.57 | 0.8668 |
| `--output-mask-k` | 5 | 2596 | 2251 | 55.1% | 120.6 t/s | 0.39 | 0.8686 |
| `--display-mask-k` | 3 | 3331 | 2572 | 63.6% | 128.2 t/s | 0.56 | 0.8709 |
| `--display-mask-k` | 5 | 3331 | 2572 | 63.6% | 128.2 t/s | 0.39 | 0.8709 |

At the same k the erasure is the same and the display side keeps the whole
speedup. That is the argument for display only masking: the tokens stay usable
as draft even though nobody sees them.

Two things to keep in mind. The acceptance rate out of drafted tokens rises with
`--output-mask-k`, from 77.2% to 86.7%, while the share of the answer that came
from the draft falls from 63.6% to 55.1%. The rate is a ratio, and the second
number is the one that tracks speed.

Erasure numbers elsewhere in this file are the `--display-mask-k 0` case, the
pessimistic one: every token shown the moment it is produced.

## Options

| Option | Meaning |
| --- | --- |
| `--stream-interval N` | expand each line into prefixes every N words, 0 to disable |
| `--draft-bias-beta F` | bias used while checking the draft, 0 = greedy check |
| `--target-bias-beta F` | bias used while decoding past the draft |
| `--output-mask-k N` | mask-k: hold back N tokens from the end of a partial answer |
| `--no-draft-reuse` | do not reuse the previous answer |
| `--no-prompt-cache-prefix` | do not reuse the KV cache of the common prompt prefix |
| `-o FNAME` | write jsonl, one object per input line, for scoring |

## Segmentation

How the source is cut into requests is a policy, and there are many: a fixed
word interval, an ASR chunk boundary, a wait-k rule, a learned segmenter. The
decoder does not implement any of them. It reads a stream that already says
where the cuts are, so a new policy never needs a change here.

### stream.jsonl

One json object per source line:

```json
{
  "id": "flores200.devtest.en:0",
  "source": "The weather is nice today and we plan to go hiking now.",
  "segmentation": {"policy": "interval", "n": 3},
  "stream_ins": [
    "The weather is",
    "The weather is nice today and",
    "The weather is nice today and we plan to",
    "The weather is nice today and we plan to go hiking now."
  ]
}
```

| field | required | meaning |
| --- | --- | --- |
| `stream_ins` | yes | the requests, in order. Each is the source known so far |
| `id` | no | join key for references and for diffing runs, defaults to `<file>:<line>` |
| `source` | no | the whole source line, defaults to the last `stream_ins` |
| `segmentation` | no | how the cuts were made, carried into the output so a run says what it was |

Pass it with `-f stream.jsonl` and `stream_ins` is used exactly as given, no
`--stream-interval` involved. The file is read straight from disk rather than
through the usual `-f` escape processing, so backslashes inside the json
survive.

`stream_ins` does not have to be growing prefixes. It is whatever the caller
decides to send. The decoder only assumes that consecutive entries usually
extend one another, and it checks: if one does not extend the last, the draft
is dropped and that request is decoded from scratch. So a restart, a correction
from ASR, or a cut in the middle of a word all behave correctly, they just do
not get the speedup for that step.

### segment.py

`bench/segment.py` turns plain text into a stream:

```sh
python3 bench/segment.py --input source.txt --output stream.jsonl \
    --policy interval --n 3 --id-prefix source.txt
```

| option | meaning |
| --- | --- |
| `--policy` | `interval` cuts every `--n` words, `whole` emits one request per line |
| `--n` | words per cut, for `interval` |
| `--id-prefix` | id stem, defaults to the input file name |

**Only word intervals are implemented**, because that is what the paper
measures and it needs no extra dependencies. It is deliberately the least
interesting part: add a function to `POLICIES` in `segment.py` and a new policy
is one entry long. Anything harder, an ASR aligner or a trained segmenter,
belongs in your own tool that writes the same jsonl, and nothing here needs to
know about it.

For convenience the decoder also accepts plain text with `--stream-interval N`,
which applies the interval policy internally and skips the separate file. That
is only a shortcut for the simple case; `bench/sweep.sh` always goes through
`segment.py` so that every run records the segmentation it used.

## Scoring

With `-o out.jsonl` the example writes one json object per input line:

```json
{"id": "...", "source": "...", "segmentation": {...}, "stream_ins": [...], "stream_outs": [...]}
```

`stream_ins[i]` is the partial source of request `i` and `stream_outs[i]` is the
answer to it. The completed translation of a line is `stream_outs[-1]`.

References are not written, because the example has no reference to write. Join
them by id in whatever scores the output. That is what `bench/score.py` does:
quality on the last answer of each line, stability across the answers within a
line.

Normalized erasure deliberately does not vary with the language. It compares a
hypothesis against its own earlier versions, so it uses one multilingual
tokenizer and the numbers stay comparable across pairs. BLEU cannot do that,
which is why the tokenizer it used is recorded in each `results.json`.

## Reproducing

```
bench/
  run-towerplus.sh    preset: TowerPlus 2B
  run-qwen3.sh        preset: Qwen3 4B
  preset-common.sh    shared body of the presets
  sweep.sh            engine: one stream, several settings, scored
  get-flores.sh       fetch FLORES-200 devtest
  segment.py          text + policy -> stream.jsonl
  score.py            erasure, BLEU, chrF, COMET
```

The stages are separate on purpose:

```
segment.py   source text + policy   -> stream.jsonl
llama-self-spec-bias  stream.jsonl       -> <setting>.jsonl
score.py     <setting>.jsonl + refs -> <setting>.results.json
```

Everything is keyed by `id`, so the three files are joined by key and a mismatch
is an error rather than a silent shift.

A preset declares a model file name and the prompt that model expects, then
sources `preset-common.sh`. Copy one to measure your own model, the file is
about fifteen lines. Models are looked up by name in `bench/models/`, which is
not in git, so put the gguf there or symlink it. `MODEL=` overrides with a path
of your own. Models are never downloaded.

`TGT_LANG` picks the language pair, and with it the FLORES files to fetch, the
language names put into the prompt and the BLEU tokenizer:

```sh
TGT_LANG=de bash bench/run-towerplus.sh
TGT_LANG=ja bash bench/run-qwen3.sh
```

A preset writes `{src}` and `{tgt}` in its prompt rather than a language name,
so one preset serves every pair. `lang_info` in `preset-common.sh` holds the
table; add a row for another language.

`CONFS` selects the settings. A setting may carry a hold back suffix:

```sh
CONFS="baseline greedy-verify bias-02 bias-03 bias-02-omask3 bias-02-dmask3" \
    bash bench/run-towerplus.sh
```

`N_SENT=100` limits the run. `SCORE=0` measures speed only and then nothing
outside the python standard library is needed. Scoring needs
`pip install -r bench/requirements.txt`; sacrebleu alone covers erasure, BLEU
and chrF, COMET is a separate large download.

## Citation

If you find this useful, please cite the paper:

```bibtex
@misc{zeng2026selfspeculativebiaseddecodingfaster,
      title={Self-Speculative Biased Decoding for Faster Re-Translation},
      author={Linxiao Zeng and Haoyun Deng and Kangyuan Shu and Shizhen Wang},
      year={2026},
      eprint={2509.21740},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.21740},
}
```

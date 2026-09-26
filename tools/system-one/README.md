# System One

A System One model does not write text. Give it a **state** and a set of **typed questions**,
and one forward pass returns a calibrated distribution over each question's declared options.
Nothing is sampled and no token is generated, so the answer cannot be malformed by
construction -- there is no reply to parse and no schema to validate.

This directory holds the `system-one` library, which builds that prompt and reads those
answers, and `llama-system-one`, a CLI over it.

## Three question types

| type | the answer | read it as |
|---|---|---|
| `noul` | P(true) for an assertion | a threshold: act at > 0.8, ask again below 0.4 |
| `choice` | a distribution over declared options | routing, classification, intent |
| `score` | a distribution over ordered levels, plus its index-weighted mean | severity, satisfaction, priority |

Every answer also carries `confidence = 1 − H(p)/ln K` and its raw logits, because calibration
is the product: a consumer that only takes the argmax could have used anything.

## llama-system-one

```bash
llama-system-one -m model.gguf \
    --state "Support ticket: I was charged twice for the same subscription. Please refund the duplicate." \
    --noul   "refund:The customer is asking for a refund." \
    --choice "queue:Which team should handle this ticket?:billing=payments and refunds,technical=a product fault,account=login or profile settings" \
    --score  "urgency:How urgent is this?:routine,soon,urgent,critical"
```

```
refund  (noul, confidence 0.578)
  P(true) = 0.9141

queue  (choice, confidence 0.910)
  -> billing
     billing                  0.9824
     technical                0.0129
     account                  0.0046

urgency  (score, confidence 0.369)
  score = 1.8675 on 0..3  (0 routine .. 3 critical)
     routine                  0.0802
     soon                     0.0803
     urgent                   0.7312
     critical                 0.1083

138 prompt tokens, 1 sequence(s), 0 tokens generated
```

Three decisions out of one prompt: whether to open a refund case, which queue to route it to,
and how to prioritise it -- each a number a caller can threshold, and no text to parse.

**The answers are the checkpoint's, not this tool's.** That run is a Qwen3-based checkpoint;
the same questions put to a 270M model trained on zh-TW data return P(true) 0.28 for the
refund assertion, which is wrong. A System One model is only calibrated on the distribution it
was trained for, so check a checkpoint against your own questions, in the language you will
actually serve, before trusting its numbers.

| option | |
|---|---|
| `--state TEXT` / `--state-file FNAME` | the context the questions are asked about |
| `--noul KEY:INSTRUCTIONS` | a true/false question; repeatable |
| `--choice KEY:INSTRUCTIONS:opt[=desc][,...]` | pick one declared option; repeatable |
| `--score KEY:INSTRUCTIONS:level[,...]` | ordered levels, and the mean of their indices |
| `--system-one-request FNAME` | the whole request as JSON, instead of the flags |
| `--system-one-template-file FNAME` | override the checkpoint's System One template |
| `--labels A,B,C` | override the labels an answer is read at (default A-Za-z) |
| `--system-one-temperature T` | rescale every answer's logits by T (default 1, skipped) |
| `--json` | print the answers as JSON |

`--json` is the same shape either front end returns:

```jsonc
{"model": "...",
 "answers": {
   "refund":  {"noul": 0.9141, "confidence": 0.5775, "logits": [9.7280, 12.0933]},
   "queue":   {"choice": "billing",
               "probabilities": {"billing": 0.9824, "technical": 0.0129, "account": 0.0046},
               "confidence": 0.9104, "logits": [13.4224, 9.0917, 8.0633]},
   "urgency": {"score": 1.8675, "legend": ["routine", "soon", "urgent", "critical"],
               "probabilities": [0.0802, 0.0803, 0.7312, 0.1083],
               "confidence": 0.3691, "logits": [9.8189, 9.8197, 12.0286, 10.1188]}},
 "usage": {"input_tokens": 138, "output_tokens": 0, "sequences": 1}}
```

`output_tokens: 0` is the point of the model class. `--system-one-request` takes that request
side back, which is how a prompt is kept identical between this CLI and a server.

The instructions in `--noul` / `--choice` / `--score` may not contain a colon; that is what the
JSON request is for, along with anything else more elaborate than a one-off from the shell.

## The prompt format belongs to the checkpoint

It is a Jinja template stored in the GGUF as the named chat template `system_one`, rendered
with llama.cpp's own Jinja engine. There is no built-in default, so a model without one is an
error. `--system-one-template-file` overrides it, and usually needs `--labels` with it: the labels say *where* an answer is read, so a format that writes
`(1)(2)(3)` has to say so or the reply is a distribution over `A`, `B`, `C` that nothing wrote.

This is **not** the chat template. A System One template is rendered with a state, questions
and option labels, never with a message list.

Segment boundaries matter as much as the text. The reference implementations encode the state
and each question block separately and concatenate the ids, which differs from a single-pass
tokenization at the seams; the template marks those seams with `system_one.segment_separator`
(U+001E) and this side tokenizes each piece on its own. On the 255-state reference set that
costs exactly one token per prompt against a single-pass encode -- and matching it is what
makes the numbers comparable to the reference implementation at all.

## Readouts

Where the answer sits is **derived from what the checkpoint is**, and there is no metadata key
for it: a model carrying a classification head is read through it, a bidirectional one at a
mask token, a causal one at the next token. So a checkpoint read the wrong way is a checkpoint
describing itself wrongly -- almost always `{arch}.attention.causal`, which the graph builds
its attention mask from too.

| readout | the answer is | cost |
|---|---|---|
| `letter_slot` | the next-token distribution at the end of each question's segment | one pass |
| `masked_slot` | the distribution at a mask token inside it (bidirectional models) | one pass |
| `rank_head` | the model's classification head scoring one sequence per option | K passes |

Only `letter_slot` can reuse a prefix across calls: bidirectional attention makes every
position depend on what follows it, so changing the tail of a state changes its head.

## Temperature

`temperature` rescales every answer's logits before the probabilities are taken; it is one
number for the whole request. A checkpoint ships with its calibration already folded into its
weights, so T = 1 is the normal case and is skipped outright -- the knob is for a caller that
has fitted a T on its own distribution. The raw logits are in every answer, so anything more
elaborate is the caller's to compute.

It is not `--temp`, the sampling temperature, which means nothing here: nothing is sampled.

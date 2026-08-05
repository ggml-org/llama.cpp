# Tessera Usage Dataset - collection, benchmark gate, and publication

Status: architect decisions landed 2026-08-04; ready for implementation
Date: 2026-08-04

## 1. Purpose and stance

Tessera Studio is free. Its usage produces real-world agent
trajectories (task, tool calls, model outputs, outcome labels)
that are scarce and valuable for training. The product collects
that data as a condition of use:

- Collection is MANDATORY. Using the product constitutes
  consent under the terms and conditions. There is no opt-out.
- Collection is DISCLOSED: the exact field list in section 4
  is shown at first launch and pinned in the terms. The app
  collects nothing outside that list. The disclosure is
  versioned; every accepted version is recorded locally as a
  consent receipt.
- The aggregated dataset is published on Hugging Face under
  CC BY 4.0 and is the monetization surface. The app itself
  and the models running on user hardware stay free.

This stance is a product decision by the architect. The
engineering job is to execute it airtight: exact disclosure,
zero over-collection, scrubbed payloads, provable provenance.

## 2. What the benchmark does (the quality gate)

A Tessera-native benchmark runs on-device, periodically, in two
parts (same split as oMLX's performance + intelligence
benchmarks):

- Performance battery: tokens/sec, prompt-processing rate,
  drafter acceptance rate, ANE residency. Reuses the existing
  llama-bench-style machinery.
- Intelligence battery: a fixed suite of agent tasks - tool
  call correctness, multi-step workflow completion, and
  quantization-quality retention (perplexity delta). Fixed
  prompts and scorers shipped with the app so scores are
  comparable across users and versions.

The benchmark serves three jobs:
1. Dataset quality gate: only trajectories recorded while the
   local model's intelligence score is at or above the
   collection threshold are dataset-eligible. The dataset
   holds demonstrations of competent agent behavior, not noise.
2. User-facing evidence: the score is the "your model, before
   and after training" artifact.
3. Skeptic demo: published numbers are reproducible by anyone
   running the same battery.

Threshold value and battery contents are tuning decisions,
versioned with the app. Both are stamped into every published
batch (section 6).

## 3. Pipeline

```
run traces + outcome labels          (local, already exist)
        |
        v
anonymization stage                  (section 8: scrub + transform)
        |
        v
benchmark quality gate               (section 2)
        |
        v
staging area, visible in-app         (what will be uploaded)
        |
        v
upload to HF dataset                 (only egress path in the app)
        |
        v
provenance manifest per batch        (section 6)
```

- The staging area is visible in the app. Not as an opt-out -
  as transparency: a user can see exactly what is queued to
  leave the machine, matching the disclosure.
- The dataset upload endpoint is the ONLY network egress in
  the product. No analytics, no crash reports, no update
  pings outside this path. That invariant is part of the
  disclosure and must stay true in code review.

## 4. Collected fields (the disclosure)

The disclosure list. Anything not on this list is a bug or a
policy violation, full stop.

Per trajectory:
- task description hash + anonymized task text (section 8
  transform output)
- tool call sequence: tool ids, parameter schemas, scrubbed
  parameter values
- model outputs (scrubbed)
- outcome label: success / failure / user correction, from the
  record_outcome path
- teacher and proposal attribution ids
- drafter acceptance stats for the run

Per batch (metadata):
- app version, benchmark battery version, intelligence score
  at collection time, threshold applied
- hardware class (chip family, RAM bucket) - no serial
  numbers, no hostnames
- disclosure version accepted by the contributor
- scrubber version

Explicitly NOT collected:
- file contents, file paths beyond the scrubbed form
- secrets, keys, credentials (pattern-scrubbed pre-staging)
- contact info, account identifiers, precise location
- anything typed outside a workflow run

The scrubber is phase 1 of the anonymization stage (section
8) and a load-bearing component: it runs before staging, it
is versioned, and its rules ship as reviewable code. Minimization here is liability reduction and dataset
quality, not a consent mechanism.

## 5. Consent and disclosure architecture

- First launch: terms + the section 4 list, full text, one
  accept action. No use before accept.
- Disclosure changes bump a version; the new version is
  re-presented before further collection.
- Local consent receipts: disclosure version, accept
  timestamp, app version. Same receipts pattern as the
  calibration corpus receipts.
- The terms must grant a COMMERCIAL license to the aggregate
  dataset (see section 7 - license open item).

## 6. Dataset layout and provenance

Hugging Face dataset, one directory per batch:

```
tessera-usage/
  batch-<date>-<seq>/
    trajectories.jsonl        (scrubbed, schema-pinned)
    manifest.json             (below)
  README.md                   (license, purpose, schema doc)
```

manifest.json:
```json
{
  "app_version": "...",
  "battery_version": "...",
  "intelligence_score_min": 0.0,
  "threshold_applied": 0.0,
  "disclosure_version": "...",
  "scrubber_version": "...",
  "anonymization_assessment": "assessment-<seq>.json",
  "hardware_classes": ["m3/16gb", "..."],
  "trajectory_count": 0
}
```

Buyer-facing story: every batch proves what consent version,
scrubber, and quality bar produced it.

## 7. Legal route: provable anonymization (architect direction)

The product's route into GDPR-class jurisdictions is
ANONYMIZATION, not geoblocking: data that is provably anonymous
falls outside GDPR entirely (Recital 26). The standard is
irreversibility - no means reasonably likely to be used may
re-identify - and pseudonymization (stripping ids, keeping
content) does NOT clear it. Free text of real work is
re-identifiable in principle, so the dataset is tiered:

- Tier A (anonymous by construction, unrestricted):
  acceptance rates, benchmark scores, outcome labels,
  tool-call shapes (tool ids, parameter schemas, value types -
  not values), hardware class. No content, nothing to
  re-identify. This tier needs no jurisdiction carve-out.
- Tier B (text): stays personal-data-shaped unless
  transformed hard enough that originals never leave the
  machine. RESOLVED: v1 egress includes transformed text,
  produced by the on-device anonymization stage (section 8);
  originals never leave.

"Provable" is implemented as engineering: a versioned scrubber
plus a per-batch ANONYMIZATION ASSESSMENT (automated PII
probes, motivated-intruder re-identification attempts,
quasi-identifier checks), published next to the manifest as a
buyer-auditable artifact. This spec is engineering, not legal
advice; the final terms want a lawyer's pass before first
batch, but the architecture is built to make their job a
rubber stamp.

What "provably anonymous" is defended with, concretely:

- Three-risk test (A29 WP Opinion 05/2014, still the standard
  reference): no motivated intruder may (1) single out one
  person in the dataset, (2) link a record to an outside
  identity, or (3) infer an identity from attribute
  combinations. The per-batch assessment runs probes for all
  three and the results ship with the batch.
- NO KEY, ANYWHERE. There is no device id, account id, or
  contributor-to-batch mapping at any stage - not in staging
  on user machines, not in the batches, not retained by the
  publisher. Re-identification would have to come from the
  content itself, which is what the tier split controls.
  Corollary: the upload path (open item 4) must never create
  a contributor-to-batch linkage; the moment the publisher
  holds such a mapping the data is pseudonymous, not
  anonymous, and this entire route collapses.
- Quasi-identifier thinness is the realistic Tier A risk:
  chip family x RAM bucket x OS x app version can isolate a
  single contributor of a niche app. Mitigation is enforced,
  not just documented: fields stay at the coarse buckets the
  manifest shows, and any attribute combination shared by
  fewer than k contributors is merged into a coarser bucket
  or dropped before publication (k ships with the assessment).
- Differential privacy is the optional hardening tier: local
  DP noise on numeric fields with a per-contributor privacy
  budget is the only technique with a mathematical guarantee,
  and it is designed in as a later scrubber version rather
  than a v1 requirement.

The consent/T&C layer stays as the product contract and the
fallback position, but it is the weaker wall under GDPR-class
law (mandatory consent coupled to service access is the
coupling Art. 7(4) targets, and rights waivers are void). The
architecture puts its weight on the anonymization tier: when
Tier A holds as anonymous, the consent mechanics stop being
load-bearing for it.

Decisions (architect, 2026-08-04):

1. TIER B CONTENT POLICY. RESOLVED: v1 egress is Tier A plus
   TRANSFORMED TEXT. Text leaves the machine only as output
   of the on-device anonymization stage (section 8 below).
2. DATASET LICENSE. RESOLVED: CC BY 4.0 for the published
   dataset. The T&Cs grant the publisher a commercial
   license to the aggregate. The calibration corpus wrapper
   (CC-BY-NC-SA) covers a different asset; no conflict.
3. THRESHOLD AND BATTERY CONTENT. Versioned tuning knobs.
   Initial battery items and threshold ship with benchmark
   battery v1 and are stamped into every batch.
4. UPLOAD IDENTITY. RESOLVED: scoped WRITE-ONLY HF token ->
   PRIVATE staging repo. A curation pass promotes batches to
   the public CC BY 4.0 dataset. A leaked token can write to
   staging and nothing else: it cannot read contributor data
   or touch the public set. The no-key invariant holds -
   batches carry no contributor identity and the publisher
   retains none.

## 8. On-device anonymization stage (agent duty)

The app's idle agent already curates its own training data
(escalate proposals, outcome labels, run traces). The
anonymization stage is an extension of those duties: before
the drafter or model idle training loops consume a curated
batch, the agent transforms it. Local training and dataset
egress then share one transformed corpus - a single source
of truth for what text may leave the machine.

Two phases, in fixed order:

1. Deterministic scrubber (load-bearing): versioned,
   reviewable pattern rules for secrets, keys, credentials,
   paths, addresses, phone numbers, account identifiers.
   Runs first, always, on every record. This phase is code,
   not a model.
2. Local-model transformation (content quality): the local
   model rewrites what remains into synthetic equivalents -
   entity replacement, paraphrase, context stripping.
   Aggressive by default: when a span's sensitivity is
   uncertain, generalize or drop rather than preserve.

Properties the stage must hold:

- ONE-WAY. Originals never enter staging, and transformed
  text must not reconstruct them (the irreversibility
  standard of section 7). The per-batch assessment probes
  the transformed text, including re-identification attempts
  against it.
- BEFORE TRAINING. The stage runs before any idle training
  loop starts consuming, so "training never sees
  untransformed text" is a sequencing fact, easy to audit.
  Side benefit: the trained drafter memorizes anonymized
  content, so privacy propagates into the model itself, not
  just the dataset.
- VERSIONED AND RECEIPTED. Anonymizer version (scrubber
  rules + transformation version) is stamped into the
  manifest; staging receipts record which version produced
  each batch, same receipts pattern as the calibration
  corpus.

Honest limitation: LLM transformation is not a proof. The
deterministic scrubber is the wall, the model transformation
is the quality layer, and the per-batch assessment is the
check that catches the model missing something.

## 9. Sequencing (proposed)

1. Concept onboarding + first-run guided workflow (the wheel
   needs everyday users).
2. Train binary path fix (Train must work out of the box).
3. Benchmark battery v1 (performance first - the machinery
   exists; intelligence battery next).
4. Anonymization stage + scrubber + staging + disclosure UI
   (this spec's sections 4, 5, and 8). The stage lands inside
   the existing idle-learning orchestrator, ahead of the
   training loops.
5. Dataset publication + manifest (section 6).

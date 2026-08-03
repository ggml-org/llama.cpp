# Tessera Self-Improving Loop - Design and Implementation Plan

Status: design / proposal. Not code. Captures an architect working
session (2026-07-31) and maps it onto the existing Tessera surface.
AI-assisted design capture; the design choices are the architect's.

Cross-references:
- Current state: [`PROJECT-STATUS.md`](PROJECT-STATUS.md)
- Studio app surface: [`tessera-studio-design.md`](tessera-studio-design.md)
- Runtime-aware ground-truth pipeline (L1-L6): [`runtime-aware-pipeline.md`](runtime-aware-pipeline.md), [`pipeline-design.md`](pipeline-design.md)
- C++ port + sidecar: [`c++-port-design.md`](c++-port-design.md)
- Research alignment + fitness form: [`research-alignment-2026-07-30.md`](research-alignment-2026-07-30.md)
- GGUF metadata spec: [`tessera.md`](tessera.md)

---

## 0. North star

The goal is not a model that knows everything. It is a model that
**knows what it does not know**: one that replaces confident
hallucination with calibrated "I am not sure, but I know how to reason
about it and go find out." Bigger models try to memorize the world;
this system tries to *navigate* it.

Every mechanism below serves that one shift:

- Confidence-triggered escalation = calibrated uncertainty.
- Retrieve-before-escalate = "go find out."
- Verifiable-outcome grounding = "and then check you were right."
- Reasoning distillation = "learn the method, not the answer."

One-sentence system summary: a private, on-device, self-improving
coding harness that drafts with two complementary speculative heads on
a Qwen-class MoE trunk, harvests its own training signal from local
usage, grounds every update in verifiable real-world outcomes,
escalates only the genuinely-hard tail to a stronger cloud teacher
under a strict privacy boundary, and routes each kind of knowledge to
the substrate that suits it.

---

## 1. This is mostly assembly, not invention

The single most important framing for this plan: **Tessera already
owns most of the substrate.** The fork runs both drafters, already
captures the exact accept/reject + per-position distribution telemetry
this loop needs, already has a planned drafter LoRA pipeline, already
has the runtime-ground-truth pipeline, and already has the agent +
tool + approval + settings surface. The work here is wiring those
together and adding a bounded delta, not building a learning system
from scratch.

| Loop component | Status | Where |
| --- | --- | --- |
| Accept/reject trace (drafter signal + fast metric) | EXISTS | `llama.tessera.spec.v1` (cheap payload), `tools/imatrix/imatrix.cpp` |
| Per-position verifier+drafter top-k (rejection-sampling input) | EXISTS | `llama.tessera.spec.v1` (top-k payload, when `--telemetry-topk > 0`), `tools/imatrix/imatrix.cpp` |
| DFlash drafter running (~30% accept) | EXISTS | imatrix spec path |
| DSpark drafter running (33% accept Q4_0) | EXISTS | `tools/dspark-gguf-patch/` + imatrix spec path |
| Acceptance metric module (C++) | EXISTS | `tools/quantize/tessera/tessera-acceptance.{h,cpp}` |
| Corpus handling (C++) | EXISTS | `tools/quantize/tessera/tessera-corpus.{h,cpp}` |
| A/B harness (C++) | EXISTS | `tools/quantize/tessera/tessera-ab-harness.{h,cpp}` |
| Evolutionary quantizer (`evolve`) | EXISTS | `tools/tessera/per_tensor_calibrate.py` |
| Agent loop + tool protocol + registry | EXISTS | `TesseraStudio/Sources/TesseraCore/Agent/*.swift` |
| Approval engine (auto/notify/prompt/denied) | EXISTS | `TesseraCore/Agent/TesseraApprovalEngine.swift` |
| Settings + privacy/telemetry keys | EXISTS | `TesseraCore/Settings/TesseraSettings.swift` (`telemetryEnabled` defaults false) |
| Runtime enum (onDevice/mlx/privateCloud) | EXISTS | `TesseraCore/Models/TesseraRuntime.swift` |
| Drafter co-adaptation via rejection-sampling LoRA | PLANNED | `PROJECT-STATUS.md` Priority 3 (`tools/tessera/dspark_pytorch.py`, peft/accelerate, target 33%->50%) |
| Runtime ground truth (kernel dequant fidelity) | PLANNED / IN FLIGHT | runtime-aware pipeline L1-L6 |
| Web search / retrieval channel | SEARCH PROVIDERS IMPLEMENTED; escalation wiring DESIGNED | `tessera-studio-design.md` 5.4 (provider-shaped, keyless DuckDuckGo default), `TesseraWebSearch.swift` + `TesseraSearchProvider.swift` |
| Reasoning / CoT from day 1 | DESIGNED | `tessera-studio-design.md` 5.7, `ThinkingBlock.swift` |
| Schema-versioned evidence / receipts pattern | EXISTS (pattern) | `llama.tessera.spec.v1`, sidecar v3 |
| Escalation subsystem (tier 1 + tier 2) | NEW | this plan, section 4.1 |
| Curation pipeline | NEW | section 4.2 |
| Knowledge stores (reasoning playbook, reference store) | NEW | section 4.3 |
| World-signal capture (coding-task verifiable rewards) | NEW | section 4.4 |
| Background idle adaptation engine | NEW | section 4.5 |
| Unified self-speculating model | NEW (research) | section 4.6 |
| North-star batched-throughput eval | NEW | section 4.7 |
| Training-data receipts / audit / purge | NEW | section 4.8 |

**Design constraint (from AGENTS.md):** match llama.cpp production
style, prefer reusing existing infrastructure over new components, no
gratuitous abstractions. This plan leans hard on that: the new work
extends existing modules and the existing `TesseraTool` protocol
rather than introducing parallel subsystems.

---

## 2. The knowledge taxonomy

Five channels, each routed to the substrate that suits it. This table
is the organizing spine of the whole design.

| Signal | Substrate | Timescale | Recursion safety | Status |
| --- | --- | --- | --- | --- |
| Token acceptance (fast) | Drafter heads (DFlash/DSpark LoRA) | per-session | SAFE: target model is external ground truth; recurse hot | EXISTS signal / PLANNED training |
| Verifiable outcomes (tests/build/commit) | The reward that gates ALL updates | slow | This IS the ground truth | NEW capture (section 4.4) |
| Object + meta reasoning (from escalation) | Capability distillation + reasoning playbook | per-escalation | Meta-layer audited against object layer + outcomes | NEW (4.1, 4.3) |
| Personal traces (style/distribution) | Trunk LoRA | idle window | DANGER: ground on human outcomes, never model exhaust | NEW capture (4.4) |
| Live reference knowledge (docs/examples) | Versioned, expiring reference store + foraging-skill distillation | on-demand + TTL | Volatile: store, never fuse into weights | DESIGNED retrieval / NEW store (4.3) |

**The one rule that decides where anything goes:**

> Volatile, factual knowledge -> explicit retrieval store.
> Stable, behavioral knowledge -> weights.

**The grounding principle:**

> The agent curates and proposes. The world judges. No closed loop
> where the model is curator, judge, and student at once.

**One pipeline, not two (ratified).** This loop is the runtime-aware
pipeline (L1-L6) extended one level up the stack, not a separate
system. L1-L6 ground the *quantizer* in the kernel's actual dequant
output - the runtime ground truth, replacing the synthetic
`_ternary_reconstruct` proxy. This loop grounds the *drafter, trunk,
and knowledge* in the world's actual output (acceptance traces, tests,
builds, commits). Same thesis - close every optimization on real
ground truth, never a proxy - at two levels of the stack. The receipts
/ schema-versioned-evidence substrate (sidecar v3,
`llama.tessera.spec.v1`) is shared across both: the acceptance telemetry
is the L1-equivalent for drafter quality, and the world signals are the
L4-equivalent (end-to-end probe) for capability.

Reference docs are the most volatile knowledge, so they are the purest
case for the store. The reasoning playbook and the reference store are
both instances of the original "engram" instinct in its correct form:
a personal, retrievable *memory trace*, not the DeepSeek pretraining
module.

---

## 3. Architecture overview

```
                       +------------------------------------------+
                       |            Tessera Studio (Swift)          |
                       |  agent loop + tools + approval + settings  |
                       +------------------------------------------+
                              |                          ^
        tool calls            v                          | events / receipts
   +-------------------------------------------------------------+
   |                    Learning services (new)                    |
   |                                                               |
   |  [1] Harness capture -----> [2] Curation -----> [3] Stores    |
   |      generations                 scrub              reasoning |
   |      accept/reject (EXIST)       dedup/quality      playbook  |
   |      world signals [4]           retention          reference |
   |                                                     LoRA set  |
   |                                                               |
   |  [5] Escalation -----+   [6] Retrieval (DESIGNED web search)  |
   |      tier1 describe  |        retrieve BEFORE escalate        |
   |      tier2 anonymize |                                        |
   |      trigger/router  |                                        |
   +----------------------|----------------------------------------+
                          v
            +-------------------------------+
            | [7] Background adaptation      |
            |     engine (idle + on power)   |
            |     drafter LoRA (PLANNED P3)  |
            |     trunk LoRA (world-gated)   |
            |     head co-train (research)   |
            +-------------------------------+
                          |
                          v  adapted bundle
            +-------------------------------+
            | Tessera serve: quantize (EXIST)|
            | calibrate/evolve (EXIST)       |
            | convert/evaluate (EXIST)       |
            | spec-decode w/ acceptance(EXIST)|
            +-------------------------------+
                          |
                          v
            verifiable outcomes (tests/build/commit) ---> [4]
```

Where it lives:
- **Swift (`TesseraCore`)**: capture orchestration, curation policy,
  stores, escalation/retrieval tools, background scheduler, receipts,
  UI surfaces. Extends the existing agent/tool/approval/settings.
- **C++ (`tools/quantize/tessera/`)**: acceptance metric (exists),
  corpus (exists), anonymizer (new, if tier 2 lands), north-star
  throughput eval (extends `tessera-ab-harness` + `tessera-acceptance`).
- **Python (`tools/tessera/`)**: the adaptation engine reuses the
  PLANNED `dspark_pytorch.py` + peft/accelerate rejection-sampling
  harness (Priority 3) and extends it to trunk LoRA + head co-train.

---

## 4. New components (the delta)

### 4.1 Escalation subsystem (NEW)

Purpose: get a stronger teacher's *reasoning* on the hard tail, under
a strict privacy boundary, and turn it into training signal.

Two tiers:

- **Tier 1 (default, fully private):** the agent emits a
  natural-language problem *frame* plus a small *structured diagnostic
  envelope* (type signatures, failing test names, redacted error text,
  observed-vs-expected, stack shape). No source code crosses the
  boundary. The teacher returns reasoning + method (+ an explicit
  "how to reason about this class of problem" meta-explanation). The
  student applies it locally; the outcome is verified against the
  world; the successful `(method -> student's own working application)`
  pair is the highest-value artifact.
- **Tier 2 (rare, opt-in, scrubbed):** when the problem is genuinely
  implementation-bound and the frame cannot capture it, build a
  pseudoanonymized worktree. Anonymizer pipeline: symbols (consistent,
  type-preserving, public-aware, ephemeral per-escalation mapping with
  the key kept local), plus strings, comments, constants, and paths.
  De-anonymize the answer locally; shred the payload after extracting
  the reasoning.

Trigger / router: a calibrated confidence/competence estimate decides
"beyond me." Seed it from the DSpark confidence head (exists as
`conf_proj` in the dspark arch). The router is itself learnable from
escalation outcomes (escalated+committed = correct; escalated+discarded
= over; not-escalated+failed = under).

Privacy boundary (load-bearing): two trust domains. The local domain
is total; the cloud domain is an explicit, opt-in, scrubbed exception.
The no-egress invariant is enforced and tested, not assumed. Secret
scrubbing happens at egress. Provider must be a zero-retention tier.
This extends the existing `remoteAPIBaseURL` / `remoteAPIKey` /
`remoteModelName` settings and the `telemetryEnabled` default-false
posture.

Consent model (ratified): egress tools default to `prompt`, but the
user may configure a **scoped allowlist** (per problem class and/or per
teacher) so trusted, recurring escalations do not prompt every call.
The allowlist is user-set, explicit, inspectable, and never widens
automatically.

Integration: new `TesseraTool`s (section 5), new settings (section 6),
a `TesseraEscalationService` in `TesseraCore` (Swift), reusing the
existing `LLMProvider` remote path for the teacher call.

Teacher selection is an ENSEMBLE, not a single pick (resolved
2026-07-31). The teacher pool is exactly the providers the user has
configured API keys for (section 6, `learning.teachers`), treated as a
dynamic pool rather than a chosen oracle. For a hard-tail instance the
service fans the SAME frame+envelope out to every available teacher,
collects N proposals, and - because the world gate is cheap and
authoritative - trials all of them against the verifiable outcome and
learns from all of them, instead of betting on one teacher. A recurring
per-teacher assessment keeps a live quality estimate (fraction of
proposals that pass the world gate, plus a reasoning-externalization
score per R6); teachers that stop being useful drift down and are used
less. This kills R3 (teacher bias) structurally: no single teacher can
become gospel, and the verbosity / reasoning-externalization criterion
of R6 is MEASURED per teacher on a recurring basis instead of chosen
once at config time.

Apple Foundation Models is the default teacher (resolved 2026-07-31).
The ensemble above is "whatever providers the user has keys for," which
leaves a cold-start gap: a fresh install with no API keys has no teacher
at all. Apple Foundation Models (AFM) closes it. AFM is available on
macOS 26+ with NO API key and no account, runs on-device or on Apple's
Private Cloud Compute (PCC) under Apple's attestation guarantees, and is
therefore the always-available FLOOR of the teacher pool: the default
drafter tier and the default escalation teacher when the user has
configured nothing else. Third-party cloud teachers (Claude/GPT) remain
the higher-capability, higher-egress tier the user opts into by adding
keys. The ordering is deliberate: AFM is the low-egress default precisely
because teacher distillation is the one real egress in an otherwise-local
system (privacy boundary, above), and PCC barely egresses at all. AFM
enters `learning.teachers` as a synthetic, keyless entry the service
always treats as available; it is the seed of the AION mediator already
specified in `tessera-studio-design.md` 14.11, promoted from annotation
hint to first-class teacher.

### 4.2 Curation pipeline (NEW)

Purpose: turn raw harvested traces into usable, safe training signal.

- Agent-driven curation (the agent loop already exists): dedup,
  quality scoring, clean `(prompt -> good outcome)` extraction,
  preference-pair generation, informativeness scoring (drop platitudes
  in the meta-layer).
- Secret scrubbing for stored data (not just egress): API keys,
  `.env` contents, tokens that landed in context. Non-negotiable for a
  coding agent.
- Retention: ring buffer with a retention policy; adaptation fires on
  idle + on-power + enough-new-signal, not on every delta.

Integration: extends `tools/quantize/tessera/tessera-corpus.{h,cpp}`
for the corpus side; a `TesseraCurationService` (Swift) for policy.

### 4.3 Knowledge stores (NEW)

Two explicit, inspectable, editable stores (the "engram" memory
traces), both preferring retrieval over weight fusion:

- **Reasoning playbook:** meta-reasoning strategies indexed by
  problem class. Primary channel for escalation meta-reasoning.
  Editable/deletable, inspectable by the user, collapse-immune.
- **Reference knowledge store:** looked-up docs/examples with
  provenance, library version, and a TTL. Volatile by nature; never
  fused into weights. Extends the DESIGNED web-search retrieval into a
  persistent, versioned cache.

Integration: new `TesseraCore` store types alongside the existing
`ConversationStore` / `CalibrationSession` pattern; surfaced via tools
(section 5) and a viewer in the Mac Studio surface.

### 4.4 World-signal capture (NEW)

Purpose: the external, verifiable reward for coding tasks. This is the
behavioral extension of the runtime-aware pipeline's ground-truth
discipline (section 2): L1-L6 ground the quantizer in kernel output;
these signals ground capability in world output. One pipeline, shared
receipts substrate - not two separate ground truths.

Capture (intrusive but local; the no-egress boundary makes this fine):
- Build / test / typecheck / lint pass-fail.
- Git: committed vs reverted; the surviving diff.
- Editor telemetry: post-agent edits (implicit correction) vs
  untouched (implicit approval).
- Agent-action accept/revert at the task level.

Routing rule (anti-collapse): drafter heads train freely on production
traces (safe); the trunk LoRA trains ONLY on human-outcome signal,
never on the model's own generations. Conflating them builds a
self-consumption collapse engine; separating them builds the moat.

Integration: a `TesseraWorldSignalObserver` (Swift, Mac-first) feeding
the curation pipeline; outcomes recorded as receipts (4.8).

### 4.5 Background idle adaptation engine (NEW)

Purpose: run adaptation in the idle + on-power window, on-device.

Feasibility anchor: an A3B-class MoE has ~3B *active* params, so
forward passes are cheap enough for background work on Apple silicon;
MLX is the on-device training substrate (the `mlx` runtime already
exists in `TesseraRuntime`). Scope:
- Drafter LoRA: reuse PLANNED Priority 3 rejection-sampling harness.
- Trunk LoRA: small rank, world-gated, reversible.
- Head co-train: research track (4.6).

Gating: idle + on-power + data-readiness. Acceptance-rate delta is the
online reward: measure before/after each step, keep improvements,
revert regressions. This plugs straight into the existing `evaluate`
tool and `tessera-acceptance`.

Integration: a `TesseraAdaptationScheduler` (Swift) over the existing
"background behaviour" settings surface; calls into the Python
adaptation engine (which reuses Priority 3).

### 4.6 Unified self-speculating model (NEW, funded research track)

Purpose: collapse trunk + drafters into one self-speculating model so
drafter and trunk cannot desync (calibration drift solved by
construction) and only one weight set is resident (more KV headroom
for batched concurrent agents).

Correct reading: NOT a weight-merge of three architecturally-different
models (causal MoE, non-causal block-diffusion DFlash, semi-AR DSpark
do not average). Instead: one backbone (the trunk) with two drafting
*heads* grafted on, each initialized from its pretrained drafter and
co-distilled against the trunk's hidden states, regime-routed (DFlash
head for parallel predictable blocks like code; DSpark head for
batched-serving confidence scheduling). Lineage: MTP heads, EAGLE
shared-weight drafting, self-speculative / layer-skip.

This is the heaviest, riskiest piece. It is FUNDED NOW (ratified) and
runs as a PARALLEL track (section 7, track R) alongside Phases 2-6.
Funding it now is synergistic with MVP-first, not contradictory: the
Phase-1 flywheel produces exactly the acceptance traces + per-position
distributions that are the co-distillation signal the grafted heads
need, so the MVP feeds track R rather than competing with it. The MVP
still goes first because it is the fastest way to validate the loop and
to start generating that signal. The fallback that keeps the whole
system viable if R underperforms: trunk + two separate drafter models
(what the fork already runs), accepting the calibration-drift and
memory-budget costs and managing them by re-co-training drafters after
trunk updates.

### 4.7 North-star eval (NEW)

Purpose: the single ruler every choice is judged against.

Headline metric: **sustained batched tokens/sec across N concurrent
coding agents, at acceptable quality, on a fixed memory budget.**
Secondary: capability-growth-over-escalations (does success rate on
the escalated problem class climb over N escalations?), acceptance
rate, and a general-competence regression guard (confirm adaptation
did not lobotomize the base).

Capability eval is MULTI-DIMENSIONAL (ratified direction). Rather than
a single problem class, the fitness has several behavioral axes, each
chosen to be MECHANISTICALLY INDEPENDENT - one axis per subsystem, so
the evolutionary search gets real gradient on each gene rather than
correlated noise:
- Mechanical correctness (failing-test + compiler/type-error instances)
  -> probes base capability + drafter acceptance.
- API currency (deprecated-API migration instances) -> probes the
  retrieval / reference-store channel.
- Hard-tail reasoning (escalation instances) -> probes the escalation
  + reasoning-distillation channel.
- Personal-style fit -> probes the trunk LoRA / personal-distribution
  channel.
- General competence (broad held-out set) -> the collapse guard.
This is a MAP-Elites-style behavioral quality-diversity archive: the
capability-loop analogue of the quantizer's regime-indexed archive
(research-alignment G4), with the multi-axis score as quality and
behavioral-regime descriptors as the archive cells. Multi-axis fitness
is also a STRUCTURAL anti-collapse defense: a candidate that spikes one
axis while cratering others scores poorly on the composite.

Aggregation is a LENS, not a decision (resolved 2026-07-31). The
expensive artifact is the per-candidate multi-axis score VECTOR; a
weighted-sum scalar and a Pareto non-domination front are two readings
of the same numbers, so we keep both and A/B them (reusing
`tessera-ab-harness`) to see which drives better learning dynamics,
rather than picking one a priori. The axes are not all the same kind:
general competence is a GUARD axis - a hard regression constraint (must
not drop more than epsilon), not a trade-off weight - while the
optimization axes (mechanical, API-currency, hard-tail, personal-style)
are where trade-offs actually live and where the weighted-sum vs Pareto
lenses genuinely compete. This maps onto machinery that already exists:
the MAP-Elites archive is the Pareto-flavored lens (best per behavioral
cell), the drafter rejection-sampling scalar is the weighted-sum lens.
No new aggregation subsystem - just name that both lenses are live and
measured against each other.

Integration: extends the existing `evaluate` tool, `tessera-ab-harness`,
and `tessera-acceptance`. The A/B compare surface (exists in the studio
design) is the natural place to render it.

### 4.8 Training-data receipts / audit / purge (NEW)

Purpose: make the self-improving system inspectable and deletable.

Extend the existing schema-versioned-evidence pattern (sidecar v3,
`llama.tessera.spec.v1`) to harvested training data: what was collected,
from which session, what it trained, and purge-on-demand. This fits
Tessera's existing receipts identity and is a trust requirement for a
system that learns from everything the user touches.

---

## 5. New TesseraTools

All conform to the existing `TesseraTool` protocol
(`TesseraCore/Agent/TesseraTool.swift`): `name`, `description`,
`parameters: JSONSchema`, `defaultApprovalLevel: ApprovalLevel`,
`execute(arguments:) async throws -> ToolResult`. Registered in
`TesseraToolRegistry.default` alongside the 8 v1 tools.

| Tool | Purpose | Approval | Rationale |
| --- | --- | --- | --- |
| `lookup_docs` | Retrieve current docs/examples for a query; cache to reference store with provenance+TTL | `auto` | Local + public; low risk |
| `query_playbook` | Retrieve reasoning strategies for a problem class | `auto` | Local read |
| `record_outcome` | Record a verifiable world outcome (build/test/commit/revert) | `notify` | Local write; keep user informed |
| `escalate_reasoning` | Tier 1: send frame+envelope to teacher, collect reasoning | `prompt` | Egress; explicit consent per call (overridable to notify once trusted) |
| `anonymize_worktree` | Tier 2: build pseudoanonymized worktree | `prompt` | Pre-egress; show what will be sent |
| `escalate_with_code` | Tier 2: send anonymized worktree to teacher | `prompt` | Egress of (scrubbed) code; highest sensitivity |
| `run_adaptation` | Trigger a background adaptation step now | `notify` | Local; long-running |
| `inspect_learning` | Show what was harvested, what was trained, acceptance deltas | `auto` | Local read; the transparency surface |
| `purge_training_data` | Delete harvested data / playbook entries / LoRA set | `prompt` | Destructive; confirm scope |

Approval levels use the existing engine; `prompt` drives the existing
ApprovalSheet. Egress tools default to `prompt` so the privacy
boundary is consent-gated from day one. Per the ratified consent model
(section 4.1), the user may set a scoped allowlist so trusted recurring
escalations run without per-call prompts; the allowlist is explicit and
never self-widening.

---

## 6. New settings keys + defaults

Extend `TesseraSettingsKey` / `TesseraSettingsDefault`
(`TesseraCore/Settings/TesseraSettings.swift`). Privacy-safe defaults.

| Key | Default | Notes |
| --- | --- | --- |
| `learning.enabled` | false | Master switch; opt-in |
| `learning.escalationEnabled` | false | Egress is opt-in |
| `learning.teachers` | [] (empty JSON array) | The escalation ENSEMBLE: every provider the user has configured keys for (id / label / baseURL / apiKey / model / zeroRetention / weight). Escalation fans out to all of them (section 4.1). Supersedes the old single-provider keys. |
| `learning.anonymizerAggressiveness` | "balanced" | enum: light/balanced/aggressive (the quality dial) |
| `learning.captureScopes` | "build,test,git" | Which world signals; editor/screen off by default |
| `learning.idleAdaptation` | false | Idle-triggered background adaptation (gated by learning.enabled) |
| `learning.onPowerOnly` | true | Background adaptation power gate |
| `learning.dataRetentionDays` | 90 | Ring-buffer retention |
| `learning.referenceTTLDays` | 30 | Reference-store freshness |
| `learning.maxConcurrentAgents` | 4 | For the north-star workload |
| `learning.guardEpsilon` | 0.02 | Collapse-guard regression tolerance (general-competence axis) |
| `learning.assessmentIntervalHours` | 24 | Recurring per-teacher assessment cadence |

`telemetryEnabled` stays false and continues to govern any EXTERNAL
telemetry; the learning egress is separate and governed by
`learning.escalationEnabled`.

Capture vs egress vs learning (resolved 2026-07-31). "Always-on
telemetry" and "privacy-first" are not in tension once three layers are
separated. (1) CAPTURE - the receipt stream - is ON by default and local:
every agent action already produces a schema-versioned receipt
(`tessera-studio-design.md` 14.10), and the accept/reject + world-signal
capture (4.4) extends it. This is the fuel; it never leaves the device.
(2) LEARNING - turning receipts into LoRA training - is opt-in via
`learning.enabled`, idle + on-power gated (4.5). (3) EGRESS - sending
anything to a teacher - is opt-in via `learning.escalationEnabled` and
approval-gated (4.1). Capture, learning, and egress are three separate
gates. The receipts are simultaneously the LoRA dataset (the
accept/reject signal IS the label) and the autonomy-calibration data the
approval engine learns trust from (`tessera-studio-design.md` 15): one
receipt stream, two learners. So the flywheel runs on data the app
already records for audit; nothing new is captured, and the only new
question is whether the user opts in to letting it train.

---

## 7. Phasing

Sequencing principle: **prove the flywheel turns with the thinnest
slice and mostly-existing parts BEFORE investing in the heavy research
(unified model).** The drafter LoRA pipeline (Priority 3) is the
foundation and is already planned; the MVP rides on it.

### Phase 0 - Foundation reuse + scaffolding

Goal: confirm the existing substrate is wired and add empty shells.
- Reuses: spec_calib telemetry, dspark patcher, acceptance module,
  agent loop/tools/approval/settings.
- New: register the section-5 tool shells (stubs), add section-6
  settings keys (defaults off), add the `TesseraCore` service skeletons
  (Escalation/Curation/Stores/WorldSignal/AdaptationScheduler) as
  empty types behind the tool protocol.
- Build gate: `swift build` (TesseraStudio package) green; C++ side
  `cmake --build build --target llama-quantize` green (the real gate;
  `tools/quantize/tessera/test_all.sh` bypasses CMake and is NOT
  sufficient).
- Acceptance: new tools appear in the registry and system prompt;
  settings surface shows the learning section, all off.
- Size: ~600 LoC Swift (mostly stubs + settings).

### Phase 1 - Prove the flywheel under a multi-dimensional eval (the MVP)

Goal: the single experiment that proves or disproves the loop. Detail
in section 8.
- Reuses: DSpark drafter (running), spec_calib v2/v3 telemetry, the
  PLANNED Priority 3 rejection-sampling LoRA harness, acceptance
  metric, `evaluate`.
- New: Tier-1 escalation only (no anonymizer); minimal curation;
  reasoning playbook store (single class); world-signal capture
  limited to build/test pass-fail + git commit; acceptance-delta
  online reward.
- Acceptance: on the primary axis (failing-test resolution), drafter
  acceptance climbs (target: the Priority-3 sanity bar, 33%->>=50%,
  driven by the escalation feed) AND student success rate climbs over N
  escalations, with the other behavioral axes + the general-competence
  guard held non-regressing. The multi-axis eval harness (section 4.7)
  is stood up in this phase even though the first proof is single-axis.
- Risk: description bottleneck (R1), collapse (R2). Both monitored.
- Size: ~1,200 LoC Swift + reuse of Priority-3 Python.

### Phase 2 - Retrieval + reference store + retrieve-before-escalate

Goal: close the "go find out" loop and purify the escalation corpus.
- Reuses: IMPLEMENTED web search providers (`TesseraWebSearch.swift` facade over `TesseraSearchProvider.swift`; keyless DuckDuckGo default, SearXNG/Tavily opt-in).
- New: reference store (provenance + TTL), the retrieve->still-stuck?->
  escalate ordering in the escalation router, foraging-signal capture.
- Acceptance: measurable fraction of would-be escalations resolved by
  retrieval; escalation corpus shifts to genuinely reasoning-bound
  problems (higher reasoning-distillation value).
- Size: ~700 LoC Swift.

### Phase 3 - Full curation + world-signal capture + receipts/purge

Goal: broaden the grounding and make it inspectable/deletable.
- Reuses: tessera-corpus, schema-versioned-evidence pattern.
- New: full curation (dedup/quality/preference-pairs/informativeness),
  secret scrubbing (store + egress), editor/CI world signals, training
  receipts + `inspect_learning` + `purge_training_data`.
- Acceptance: scrubbing catches injected secrets in a red-team test;
  receipts trace every adapted artifact to its source sessions; purge
  removes them verifiably.
- Size: ~1,000 LoC Swift + ~300 C++ (scrubbing in corpus).

### Phase 4 - Background idle adaptation engine (hardened)

Goal: make adaptation autonomous and safe in the idle window.
- Reuses: Priority-3 harness, `mlx` runtime, background-behaviour
  settings surface.
- New: `TesseraAdaptationScheduler` (idle + on-power + data-ready
  gating), trunk LoRA (world-gated), acceptance-delta keep/revert,
  thermal/power awareness.
- Acceptance: adaptation runs unattended in idle windows without
  degrading foreground responsiveness; revert-on-regression fires
  correctly on an induced regression.
- Size: ~900 LoC Swift + Python engine extension.

### Phase 5 - Tier 2 anonymized worktree escalation

Goal: cover the implementation-bound tail without leaking code.
- Reuses: escalation service from Phase 1.
- New: anonymizer pipeline (symbols consistent/type-preserving/
  public-aware/ephemeral; strings/comments/constants/paths), the
  anonymizer quality dial (section 9 R1 experiment first), local
  de-anonymization, payload shred.
- Acceptance: anonymizer quality curve measured (reasoning quality vs
  aggressiveness); zero secrets in an adversarial payload review;
  tier-2 escalations improve success on implementation-bound tasks.
- Size: ~800 LoC Swift + ~500 C++ (anonymizer).

### Phase 6 - Multi-class + north-star eval + dashboard

Goal: generalize and instrument the headline metric.
- Reuses: `evaluate`, `tessera-ab-harness`, `tessera-acceptance`, A/B
  surface.
- New: multi-class playbook + routing, the batched-throughput
  north-star workload (N concurrent coding agents, fixed memory
  budget), capability-growth-over-escalations chart, learning
  dashboard in the Mac Studio surface.
- Acceptance: north-star metric reported and trending; dashboard shows
  harvest -> train -> acceptance-delta receipts per class.
- Size: ~1,000 LoC Swift + ~400 C++.

### Track R (funded now, parallel) - Unified self-speculating model

Funded now (ratified); staffed in parallel from the start, alongside
Phases 2-6. Not on the MVP critical path, and fed by the Phase-1
acceptance traces (its co-distillation signal).
- R.1 Graft one drafting head (DSpark-seeded) onto the trunk; co-distill
  against trunk hidden states; measure acceptance vs the separate-drafter
  baseline.
- R.2 Add the DFlash-seeded head; regime routing; measure batched
  throughput + KV headroom vs separate drafters.
- R.3 If R.1/R.2 beat the baseline, migrate the adaptation engine to
  update heads jointly with the trunk; else keep separate drafters and
  re-co-train after trunk updates.
- Exit criterion: unified bundle beats separate-drafter setup on the
  north-star metric at equal quality, OR it does not and the fallback
  (already running) remains the shipping path. Either outcome is fine;
  the system does not depend on this track.

---

## 8. The thin end-to-end slice (MVP), detailed

This is the experiment Phase 1 runs. It is designed to learn the most
with the least new machinery, riding on parts that already exist or
are already planned.

Setup:
- Evaluation is MULTI-DIMENSIONAL from the start (section 4.7): a small
  held-out set per behavioral axis (mechanical correctness, API
  currency, hard-tail reasoning, personal-style fit, general
  competence), each axis probing a different subsystem. Use progressive
  evaluation (cheap proxy on most candidates, full multi-axis on
  survivors) to respect the idle-compute budget - this reuses the
  existing island-GA progressive-eval infrastructure.
- SEQUENCING NUANCE: multi-axis MEASUREMENT from day 1, but prove the
  loop moves a needle on the simplest, most verifiable axis FIRST
  (failing-test resolution: red test -> green, binary reward), then
  open the full multi-axis evolutionary search. Do not multi-axis
  OPTIMIZE before the core escalation->distill->acceptance-climb loop
  is proven to move any axis.
- Drafter: existing DSpark (33% baseline on Q4_0). Trunk: existing
  Tessera-quantized target. No unified model.
- Adaptation: the PLANNED Priority-3 rejection-sampling LoRA harness,
  fed by escalation-produced signal instead of (or alongside) the
  imatrix telemetry.

Loop under test:
1. Agent hits a class instance it cannot solve (confidence trigger).
2. Tier-1 escalation: frame + structured envelope -> teacher reasoning
   + meta-method.
3. Student applies the method to the real (private) code.
4. World gate: does it build / pass tests / get committed?
5. On success: store `(method -> student's working application)`; feed
   the rejection-sampling drafter update; record acceptance delta.

Metric + success threshold:
- Primary: drafter acceptance on the class climbs across N escalations
  (hold the Priority-3 sanity bar, 33%->>=50%, as the first gate).
- Secondary: student success rate on held-out class instances climbs.
- Guard: general-competence eval (existing `evaluate`) does not
  regress beyond a small epsilon.

What we learn:
- If acceptance + success climb without regression: the flywheel
  turns; proceed to Phases 2-6.
- If acceptance climbs but success does not: the drafter learns but
  capability does not transfer -> the description bottleneck (R1) or
  reasoning-vs-implementation gap is biting; instrument the application
  step and tighten the envelope.
- If general competence regresses: collapse risk (R2) is real;
  re-anchor trunk updates strictly on world signal and reduce rank.

---

## 9. Risks and de-risking experiments

| ID | Risk | Mitigation | De-risking experiment |
| --- | --- | --- | --- |
| R1 | Description bottleneck: the stuck model misframes the problem; teacher returns correct-for-wrong-problem reasoning distilled as gospel | Frame + structured diagnostic envelope (types, failing tests, redacted errors); teacher clarifying-questions used to train articulation | Escalate the same hard tasks at varying frame completeness; measure teacher-reasoning usefulness vs frame quality |
| R2 | Model collapse / self-consumption on the trunk | Trunk LoRA grounded ONLY on human/world outcomes, never model generations; drafter recursion is safe (target is external) | General-competence regression guard every adaptation step; revert on regression |
| R3 | Teacher bias / generic-wrong-for-this-repo code | Teacher proposes, world disposes: cloud output still passes build/test/commit gate before training | Measure fraction of teacher outputs that fail the local world gate; ensure they are filtered, not distilled |
| R4 | Privacy at egress: the hard tail carries the heaviest payloads | Two trust domains; no-egress invariant enforced+tested; scrub at egress; zero-retention provider; egress tools default `prompt` | Red-team: inject secrets, confirm zero reach the teacher; audit provider retention posture |
| R5 | Unified-model head grafting is research-hard | Separate parallel track R; viable fallback (separate drafters) already runs | R.1 single-head graft acceptance vs separate-drafter baseline |
| R6 | Reasoning-summary degradation (APIs hide raw CoT) | Reasoning-elicitation prompt template; verbosity-selected teacher; keep object layer to audit meta layer | Compare distilled capability from summary-only vs elicited-rationale; pick the teacher that externalizes process |
| R7 | Reward latency (fast acceptance vs slow correctness) | Route fast signal to drafter, slow signal to trunk; different timescales, different targets | Track fast/slow signal agreement; alert on sustained divergence |
| R8 | Scope / identity tension (training engine vs quant studio) | RESOLVED (ratified): the adaptation engine lives IN-REPO under `tools/tessera/` as a first-class Tessera component; the loop is an extension of the runtime-aware pipeline, not a sibling product | n/a - decided |
| R9 | Anonymizer degrades reasoning (names carry semantics) | Consistent + type-preserving + public-aware anonymization; quality dial | The Phase-5 prereq: reasoning quality vs anonymizer aggressiveness curve |

---

## 10. Effort and sequencing summary

| Phase | Rough new Swift | Rough new C++/Py | Depends on | Parallelizable |
| --- | --- | --- | --- | --- |
| 0 scaffolding | ~600 | 0 | nothing | first |
| 1 MVP flywheel | ~1,200 | reuse P3 | 0, Priority 3 | critical path |
| 2 retrieval+store | ~700 | 0 | 1 | with 3 |
| 3 curation+receipts | ~1,000 | ~300 | 1 | with 2 |
| 4 background engine | ~900 | Py ext | 1,3 | after 3 |
| 5 tier-2 anonymizer | ~800 | ~500 | 1 (R9 first) | with 4/6 |
| 6 multi-class+eval | ~1,000 | ~400 | 2,3,4 | last |
| R unified model | research | research | 1 (fed by its traces) | funded now, parallel |

Critical path: 0 -> 1 -> (2 || 3) -> 4 -> 6. Track R is funded now and
staffed in parallel (ratified); Phase 5 runs off the critical path. The
MVP (Phase 1) is still deliberately first and small: it is the go/no-go
on the thesis AND it produces the acceptance traces that feed track R's
co-distillation.

---

## 11. Ratified decisions (2026-07-31)

The architect ratified the full scope and settled the open questions:

1. **Scope:** tackle the FULL scope as designed.
2. **Escalation consent:** a user-set **scoped allowlist** is the UX
   (egress defaults to `prompt`; trusted recurring escalations may be
   allowlisted per class / per teacher). See section 4.1.
3. **Adaptation engine location:** IN-REPO under `tools/tessera/`, a
   first-class Tessera implementation (resolves R8). Not a sibling.
4. **Capture scope defaults:** confirmed - build / test / git on;
   editor and screen off by default.
5. **Unified model (track R):** FUNDED NOW (overrules the earlier
   "defer" recommendation). Runs parallel, fed by the Phase-1
   acceptance traces; MVP still goes first. See sections 4.6, 7, 10.
6. **Relationship to the runtime-aware pipeline:** this loop is an
   EXTENSION of the L1-L6 pipeline, not a separate ground truth. One
   pipeline, two levels (quantizer grounded in kernel output;
   capability grounded in world output), shared receipts substrate.
   See section 2.

7. **Evaluation is multi-dimensional** (refines the earlier "pick one
   class" framing): the fitness has several mechanistically-independent
   behavioral axes (section 4.7), so the evolutionary search sees
   trade-offs across genes. The first proof still focuses the
   OPTIMIZATION on the simplest axis (failing-test resolution) while
   measuring all axes.
8. **Aggregation is a lens, not a fork** (resolves the earlier
   "weighted sum vs Pareto" open item): the multi-axis score vector is
   the substrate; a weighted-sum scalar and a Pareto non-domination
   front are two readings of the same data, so both stay live and are
   A/B'd against each other (section 4.7). Guard axes (general
   competence) are hard regression constraints, not trade-off weights;
   the weighted-sum-vs-Pareto question only applies to the optimization
   axes.
9. **Teacher selection is an ensemble** (resolves the earlier "which
   teacher" open item): the teacher pool is whatever providers the user
   has configured keys for (`learning.teachers`); the same frame fans
   out to all available teachers, every proposal is trialed against the
   world gate and learned from, and a recurring per-teacher assessment
   keeps a live quality estimate that gates future use (section 4.1).
10. **Apple Foundation Models is the default teacher** (resolves the
    cold-start gap in decision 9): AFM (macOS 26+, no API key, on-device
    or Private Cloud Compute) is the always-available floor of the teacher
    pool and the default drafter tier; third-party cloud teachers are the
    higher-capability, higher-egress opt-in tier. See section 4.1.
11. **Telemetry capture is on by default and local; learning and egress
    are opt-in** (the "always-on telemetry" decision): the receipt stream
    is the LoRA fuel and is always recorded locally, but turning it into
    training is gated by `learning.enabled` and teacher egress by
    `learning.escalationEnabled`. Capture, learning, and egress are three
    separate gates. See section 6.
12. **Autonomy is calibrated, not fixed** (joint with
    `tessera-studio-design.md` 15): the approval policy is a learned
    projection over the receipt history - needy at first, auto-continuing
    on consistently-allowed action-classes, prompting on novel/edge
    cases - under a one-way ratchet (learning only ever grants MORE
    autonomy on OBSERVED-SAFE patterns; a new consequential/irreversible
    action-class always prompts) plus a scoped, time-boxed, logged YOLO
    override. The receipts that train the model also train the approval
    policy: one stream, two learners.

Remaining open items:

- **Held-out set sizes per axis** - needs a first calibration pass.
- **AFM teacher quality on the hard tail** - is Apple Foundation Models
  good enough as the default teacher for the genuinely-hard escalations,
  or does the hard tail always need a third-party teacher? The recurring
  per-teacher assessment (section 4.1) measures this.

---

## 12. Design provenance

This plan is the capstone of a working session that started from
"joint-finetune Qwen MoE + DFlash + DSpark on a ChatGPT export, all
learning to use engrams" and resolved, step by step, into the layered
system above. The through-line, supplied by the architect: ground
everything in the world, keep volatile knowledge explicit, let the
agent curate but never judge itself, and treat calibrated uncertainty
("I don't know, but I can reason it out and find out") as the actual
product - not raw scale.

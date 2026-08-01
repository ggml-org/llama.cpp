# Tessera Studio Harness - Open-Source Absorption Map

_Date: 2026-07-31. Source: seven open-source agent repos scouted and mapped
by three parallel analysis passes. Per-cluster evidence (with file:line
citations) lives in the scout reports and is the ground truth for the
individual claims:_

- _`tessera-scout/reports/cluster-A-computer-use.md` - open-interpreter (Rust Codex fork), self-operating-computer_
- _`tessera-scout/reports/cluster-B-perception.md` - UI-TARS, OpenAdapt_
- _`tessera-scout/reports/cluster-C-browser-research-assistant.md` - browser-use, gpt-researcher, openclaw_

_This document is the source of truth for HOW these findings reshape the
Studio harness plan. Where this and another plan doc disagree, this wins
until the other is updated._

## 0. Purpose and the one framing

The question was: what does Tessera Studio need to be a genuinely good
general-agent harness - not just a coding agent, and not by copying what
OpenAI/Google ship, but by absorbing what the open-source community has
proven and differentiating on what only Tessera has.

The framing that organizes everything: **one machine, two payloads.** The
SAME agent loop serves (a) an outward general-purpose Mac agent and (b) an
inward model-improvement flywheel. This is the ratified thesis of
`self-improving-loop-design.md` section 1 ("one pipeline, not two"). The
seven repos were read strictly through that lens: absorb only what slots
into the one existing loop as _one more `TesseraTool`, one more settings
key, or one more markdown file_ - never as a parallel subsystem. That is
the whole test, and it is the project's `AGENTS.md` constraint ("no new
abstractions, match existing style, reuse existing infrastructure") made
operational.

## 1. Ground truth: what already exists vs what is new

Critical correction to the starting assumption. Studio is NOT greenfield.
The inward flywheel is substantially built and building green. The outward
capabilities are not. This decides the entire sequencing.

| Layer | Status | Where |
| --- | --- | --- |
| Agent loop + tool protocol + registry | EXISTS | `TesseraCore/Agent/{TesseraAgentLoop,TesseraTool,TesseraToolRegistry}.swift` |
| Approval engine (auto/notify/prompt/denied) | EXISTS | `TesseraCore/Agent/TesseraApprovalEngine.swift` |
| Learning spine (17 services/stores) | EXISTS | `TesseraCore/Learning/*.swift` (curation, world-signal, capability eval, escalation, playbook, reference store, foraging, anonymizer, adaptation scheduler) |
| Learning tools (9) | EXISTS | `TesseraCore/Tools/Learning/*.swift` |
| Engine providers (local + remote) | EXISTS | `TesseraCore/Engine/{LlamaLLMProvider,RemoteLLMProvider,TesseraEngineBridge}.swift` |
| Settings + privacy keys (`telemetryEnabled` default false) | EXISTS | `TesseraCore/Settings/TesseraSettings.swift` |
| Quantizer/eval tools (evolve, calibrate, evaluate, sidecar) | EXISTS | `TesseraCore/Tools/*.swift` |
| **Safety/verification hardening** (layered permission, fail-closed guardian, denial circuit-breaker, pre/post state-diff verification) | NEW | extends `TesseraApprovalEngine` + `TesseraWorldSignalObserver` |
| **Computer-use capability** (eyes + hands) | NEW | new `TesseraComputerUseTool` + perception module |
| **Browser capability** (live-web reach) | NEW | new `TesseraBrowserTool` (WKWebView) |
| **Research capability** (go find out) | NEW | new `TesseraResearchTool` over a to-be-built `TesseraWebSearch` |
| **Skills + persona** (extensibility + identity) | NEW | new `TesseraSkillLoader` + `Skills/` + `SOUL.md` |
| Web search channel (Tavily) | DESIGNED only | `tessera-studio-design.md` 5.6 (`TesseraWebSearch.swift` not yet built) |

## 2. The unified absorption map

Deduplicated and re-ranked across all three clusters. Three independent
analyses converged on the same spine, which is the strongest signal here.

### Theme 1 - The safety/verification spine (world-gate made concrete)

The highest-confidence theme: all three clusters produced it independently.
This is Tessera's "agent curates, world judges, never self-judge" principle
turned into implementable contracts.

| ID | Pattern | Source | Maps to | Pri |
| --- | --- | --- | --- | --- |
| S1 | "Verify a real state change; a self-reported success is not enough" - capture pre-state, act, capture post-state, diff | open-interpreter `qa-testing/SKILL.md:80-99`; OpenAdapt `strategies/base.py:140-176` | `TesseraWorldSignalObserver` + agent loop verification seam | P0 |
| S2 | Fail-closed review contract: assess the exact planned action, return strict JSON, fail closed on timeout/malformed | open-interpreter `guardian/mod.rs:1-12,45-67` | `TesseraApprovalEngine` (cheap rule-based default; model-review only for the ambiguous tail - battery) | P0 |
| S3 | Denial circuit-breaker: 3 consecutive or 10-of-last-50 denials -> interrupt the turn | open-interpreter `guardian/mod.rs:47-49,90-126` | collapse guard / agent loop | P0 |
| S4 | Layered permission: approval-policy x permission-profile x sandbox-enforceability -> safety check; fail-safe to AskUser; "only auto-approve when a sandbox can actually be enforced"; hardlink-aware path containment | open-interpreter `safety.rs:20-110` | `TesseraApprovalEngine` hardening | P0 |
| S5 | Scoped approval gating: per-session/per-sender scope, command level (all/safety/strict), tools allow/deny, "unavailable tools hidden by gating, not left to fail" | openclaw `config/bundled-channel-config-metadata.generated.ts` | `TesseraApprovalEngine` extension | P1 |

### Theme 2 - Eyes and hands (computer-use perception + execution)

Cluster B's contribution. UI-TARS = perception half; OpenAdapt = replay half.

| ID | Pattern | Source | Maps to | Pri |
| --- | --- | --- | --- | --- |
| H1 | Record/replay loop as the Mac computer-use architecture: ScreenCaptureKit (screen) -> Accessibility (a11y) -> agent decides -> CGEvent executes | OpenAdapt `strategies/base.py:56-127` | new `TesseraComputerUseTool` + perception module | P0 |
| H2 | Recorded-task data model as skill-capture receipts: action + screenshot ref + a11y `element_state` + window context, FK-linked, timestamped | OpenAdapt `models.py:46-170` | receipts substrate (sidecar-v3 shape); the flywheel's durable artifact | P0 |
| H3 | Model-native absolute-coordinate grounding in a canonical resized frame; fixed platform action space as structured output | UI-TARS `action_parser.py:115-143,241-266`, `prompt.py:3-60` | perception module (Vision supplies pixels, the multimodal trunk supplies the point) | P0/P1 |
| H4 | Element-anchored re-grounding (re-locate target by description in the current frame, not absolute coords) + event reduction (raw HID -> semantic actions) | OpenAdapt `strategies/visual.py:187-247`, `events.py:757-795` | replay/skill path + `TesseraCurationService` | P1 |
| H5 | Perception grounding ladder (raw coords -> OCR text-click -> labeled boxes, per model capability) + screenshot compression | self-operating-computer `prompts.py:11-196`, `apis.py:163-165` | perception module; battery-aligned | P1 |
| H6 | Capture-time PII scrubbing (scrub a11y text + screenshots before store) | OpenAdapt `models.py` scrub methods, `scrub.py` | no-egress boundary; LOAD-BEARING, not optional | P0 (privacy gate) |

### Theme 3 - Reach and knowledge (browser + research)

Cluster C's contribution.

| ID | Pattern | Source | Maps to | Pri |
| --- | --- | --- | --- | --- |
| K1 | Per-claim citation + "never fabricate / data-grounding" contract: every claim carries `([cite](url))`; never cite a source not in context; only report observed data | gpt-researcher `prompts.py:262-316`; browser-use `system_prompt.md:1-22` | prompt text + a verifier check (drafter/verifier confirms each citation resolves). CHEAPEST item, highest soul-alignment | P0 |
| K2 | Research sub-agent loop: plan -> fan-out search -> URL-dedup -> LLM-curate -> cite-and-synthesize | gpt-researcher `agent.py:331,451`, `skills/researcher.py:97,801` | new `TesseraResearchTool` over `TesseraWebSearch` + reference store. No new retriever, no new subsystem | P0 |
| K3 | Indexed-DOM page-state repr (`[index] <tag attrs>`) as an embedded-WKWebView browser tool; model addresses elements by index | browser-use `dom/serializer/serializer.py:966` | new `TesseraBrowserTool` (WKWebView + injected serializer JS) | P0/P1 |
| K4 | Two-layer page-change re-ground guard: when the world changes (URL/focus), drop the queued plan and re-observe | browser-use `service.py:2733-2800` | agent loop action-execution path (GENERAL, benefits every tool batch) | P0 |
| K5 | Source-curation step (LLM scores relevance/credibility/currency, retains original, returns JSON) + plain-NL sub-queries with operator-banning | gpt-researcher `prompts.py:213-259,319-350` | reference knowledge store with provenance + TTL | P1 |

### Theme 4 - Identity and extensibility (skills + persona)

| ID | Pattern | Source | Maps to | Pri |
| --- | --- | --- | --- | --- |
| I1 | Skills directory + `SKILL.md` manifest (frontmatter `os`/`requires`/`install` + "When to Use / When NOT to Use" body) + filesystem loader | openclaw `skills/apple-reminders/SKILL.md`, `src/node-host/skills.ts:35-143` | new `TesseraSkillLoader` + `Skills/` (bundle + Documents). Markdown + a loader; the format is an emerging standard (Mavis/Anthropic skills) | P0 |
| I2 | `SOUL.md` persona: per-agent workspace file, read every turn in prompt order, "follow unless higher-priority overrides" | openclaw `src/agents/workspace.ts:49`, `system-prompt.ts:251-252` | per-conversation/per-agent persona in Settings; injected ahead of tool/skill text. Outward agent vs inward flywheel get different souls | P1 |
| I3 | Harness = per-model prompt + constrained tool surface + context-budget rules (no item >10K tokens, incremental history only) | open-interpreter `harness/routing.rs:5-90`, `AGENTS.md:91-100` | agent loop config/data for squeezing capability from small on-device models; KV-headroom discipline | P1 |
| I4 | Local-first config/state posture: `0o700` state dir, secrets via env/file/exec, `doctor --fix` migrations | openclaw `config/config-journal-snapshot.ts:35-49` | Settings + receipts/audit; the privacy posture Tessera already wants | P1 |

### Theme 5 - Doctrine (not code)

| ID | Pattern | Source | Pri |
| --- | --- | --- | --- |
| D1 | Product doctrine as agent-design rules: "tool results are prompts - return what the model needs next, not a bare ack"; "collapse act-then-observe pairs into one tool result"; "never dead-end the agent - failure text states what to try next"; "defaults are the product" | openclaw `AGENTS.md` Product Doctrine | P2 - bake into tool-result formatting + agent loop |

## 3. What we deliberately do NOT absorb

As valuable as the absorbs. The seven repos share a common set of things to
leave on the floor:

- **Anyone else's agent loop.** open-interpreter's is so bloated its own
  maintainers warn against growing it (`codex-rs` `AGENTS.md:72-83`);
  self-operating-computer's is a 2023 toy with no safety, memory, or
  evidence. Tessera has its own. Take shapes, not loops.
- **Cloud / vendor / server infrastructure.** Provider adapter zoos, MCP/ACP
  servers, cloud sync, FastAPI/NextJS/Docker deploy stacks, the openclaw
  Gateway-as-daemon. Tessera is on-device-first, single-machine,
  no-egress-by-default.
- **Heavy Python/CUDA/CV/training stacks.** FastSAM, vLLM, pyautogui,
  Tesseract, SQLAlchemy, embedding vector stores, VLM pretraining. Absorb
  the interface idea, implement natively (Accessibility / ScreenCaptureKit /
  Vision / CGEvent).
- **Unsigned-binary supply chains.** The QA skill that `curl|sh`-installs
  driver CLIs will not pass App Store review. Implement the input driver
  natively or vendor a signed tool.
- **Self-judging evaluation.** self-operating-computer uses GPT-4o to judge
  GPT-4o - the exact closed loop the world-gate forbids. Reuse the eval
  SHAPE; make the judge an independent model or a world signal.
- **Anti-safety prompt hacks.** self-operating-computer's "don't say you're
  unable to assist" coercion. Explicitly do not absorb.

## 4. The cross-cutting risk: privacy is the product

All three clusters independently flagged privacy/security as the real risk,
not the license. A computer-use agent records screens (passwords/PHI by
construction); a browser agent runs inside logged-in sessions (banking,
mail); user-editable `SOUL.md`/`SKILL.md` that a web tool can influence is a
prompt-injection surface. The mitigations are non-negotiable before any
capability ships:

- Capture-time PII scrubbing (H6) - load-bearing.
- Every consequential computer-use / browser action routes through the
  approval engine at `prompt` (S2/S4).
- No-egress boundary holds; `telemetryEnabled` stays default-false.
- Bootstrap files (SOUL.md/SKILL.md) stay user-owned and out of any egress
  path; preserve the higher-priority-override ordering as the injection
  mitigation.

All seven licenses are permissive (MIT / Apache-2.0); pattern-absorption is
clean-room with negligible exposure. Keep attribution + NOTICE for any
verbatim port. Do not reuse "Codex" / "Open Interpreter" trademarks.

## 5. Sequencing - implementation waves

The inward flywheel exists; the new work is the outward capabilities plus
the safety spine that both payloads share. Sequence so the safety spine
lands first (everything depends on it), then the cheap high-soul wins, then
the heavier native capabilities.

**Wave 1 - Safety spine + cheap high-soul wins (P0, low contention).**
Builds and verifies against the existing green package:
- S1-S4: approval-engine hardening + fail-closed verifier + circuit-breaker
  + pre/post state-diff verification seam.
- K1: citation + never-fabricate contract (prompt + verifier check).
- I1: skills directory + `SKILL.md` loader.
- K2: research tool over a newly-built `TesseraWebSearch`.

**Wave 2 - Native capabilities (P0/P1, macOS-first).** Heavier, needs the
spine from Wave 1 in front of every action:
- H1-H6: computer-use tool (ScreenCaptureKit + Accessibility + CGEvent),
  model-native grounding, skill-capture receipts, PII scrub.
- K3-K4: browser tool (WKWebView + indexed-DOM serializer) + re-ground guard.

**Wave 3 - Identity + polish (P1/P2).**
- I2-I4: SOUL.md persona, harness profiles + context-budget rules,
  local-first config posture + doctor migrations.
- S5, H4/H5, K5, D1: scoped gating, element re-grounding + grounding ladder,
  source curation, product-doctrine bake-in.

**Integration discipline:** the workers share `TesseraToolRegistry.default`
and `TesseraSettings`, so they create new files and leave the one-line
registration edits for sequential integration (one cherry-pick at a time,
regenerate the build between), exactly as the Prism FFI campaign does it.

## 6. The differentiation, restated

Every scouted repo is a harness pointing at a model it does not own, or a
model with no harness. Tessera is the only one that owns BOTH the optimized
local inference AND the harness, and where the two co-evolve: the agent used
by day is the same agent that improves the model by night, with a receipt
for every step. The absorbed patterns are chosen to strengthen exactly that
loop - safety spine, eyes/hands, reach/knowledge, identity - while the
things we skip (cloud, vendor, self-judging, unsigned binaries) are the
things that would dilute it.

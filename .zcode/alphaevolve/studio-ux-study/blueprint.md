# Tessera Studio - SOTA UX Blueprint

Run: `studio-ux-study` (single-instance, design-only).
Date: 2026-08-02.
Baseline SHA: 10222c950.
Scope: a single authoritative design document for turning Tessera
Studio (the SwiftUI Mac app, 229 Swift files, ~65K LOC) into a
state-of-the-art user experience that is legible to BOTH
non-technical users AND well-versed ML experts. No code changes;
this is the spec a future implementation wave consumes.

This blueprint is grounded in (a) a full read of the existing app
source, (b) the prior context in `docs/audit-2026-07-29.md`,
`docs/agent-patterns-research.md`,
`docs/research-autonomy-calibration-2026-07-31.md`,
`docs/tessera-studio-design.md`, and `docs/self-improving-loop-design.md`,
and (c) web research on SOTA peers and Apple-native patterns. Every
recommendation cites either the current Swift file it touches or a
specific peer that demonstrated the win. All links are in the
`## Sources` section at the end and cited inline by URL.

ASCII only throughout, per repo rule.

---

## 0. TL;DR

The app today is **expert-only by accident**: a clean four-destination
shell (`ContentView.swift`) wraps a deep pipeline (calibrate, evolve,
quantize, evaluate, convert, acceptance gates, L2 divergence, A/B
receipts, MAP-Elites archive) and a serious agent loop with a layered
safety spine. Nothing is hidden; nothing guides a non-technical user;
the Playground's `PlaceholderLLMProvider` ships as the default. The
single highest-leverage move is to **keep the four destinations, add a
"mode" axis on top of them (Simple / Standard / Studio), and re-skin
every existing expert surface as the unwrapped depth of an
already-approachable Simple view.** Concretely: the Library becomes a
two-tier model picker (chat-ready cards on top, hardware/policy/sidecar
inspector on disclosure); the Playground gains model scope, reasoning
collapse, and tool-call receipts (already in the agent loop, just not
surfaced); the Runs view becomes a lineage timeline first, table
second; the Analytics dashboards move behind a "compare two runs"
intent rather than living as file-importers. The Learning subsystem
stays where it is (Settings -> Autonomy, plus a read-only
"Learning" tab) but its receipts become a first-class trust surface
the user sees at every approval. Fourteen components, eight flows,
four phases, all anchored to existing Swift files.

---

## 1. Current-state audit

### 1.1 What the app does well today

- **A restrained four-destination shell.** `TesseraStudioMac/App/
  ContentView.swift:5-21` defines a `Destination` enum with exactly
  four cases: Library, Playground, Runs, Learning. This is the
  Foundation-Lab three-destination pattern from the agent-patterns
  research (`docs/agent-patterns-research.md`), extended by one. It is
  the right skeleton and should not be re-litigated.
- **A real agent loop with streaming events.** `Agent/TesseraAgentLoop.
  swift:62-94` emits a typed `AgentEvent` stream (thinking / text /
  toolCall / toolResult / error / done). The Playground consumes it
  (`Views/PlaygroundView.swift:113-134`). This is more sophisticated
  than what most local-LLM competitors ship.
- **A layered, auditable safety spine.** `Agent/TesseraSafetyDecision.
  swift`, `Agent/TesseraApprovalEngine.swift:125-173`,
  `Agent/TesseraDenialCircuitBreaker`, and the structural
  `TesseraActionClass` classifier are state-of-the-art for an on-device
  agent. The autonomy-calibration research doc explicitly rates the
  industry ("every major coding agent ships a permission system; none
  of them learns") and Tessera's spine is already at the frontier.
- **Receipts everywhere.** The runtime-aware pipeline emits four
  schema-versioned JSON reports: `tessera.map-elites-archive.v1`,
  `llama.tessera.acceptance.v1`, the A/B harness receipt, and
  `llama.tessera.runtime-probe.v1`. Each has a dedicated view
  (`Views/ArchiveBrowserView.swift`, `Views/AcceptanceGateView.swift`,
  `Views/ABReceiptView.swift`, `Views/L2DivergenceView.swift`) and a
  unified `AnalyticsDashboardView.swift` that auto-routes by schema.
  This is the "show your work" substrate most apps lack.
- **Honest defaults.** `Tools/TesseraResearchTool.swift:21` runs web
  search at approval `.prompt`. `Tools/CalibrateTool.swift`,
  `QuantizeTool.swift`, `EvolveTool.swift`, `ConvertTool.swift` all
  default to `.prompt`. `EvaluateTool.swift` defaults to `.notify`.
  Only `ListModelsTool` and `InspectSidecarTool` are `.auto`. This is
  conservative and correct.
- **A real chat history drawer** (`Views/ChatHistoryDrawer.swift`) with
  SwiftData persistence (`Models/Conversation.swift`), date filtering,
  rename, search across title/model/tool, and export to Markdown/JSON.
  This is at parity with the AWS sample and the design doc 5.7.
- **A live telemetry drawer** (`Views/TelemetryDrawer.swift`) with
  sparklines for throughput, memory, GPU, kernel dispatch. Polled at
  500ms; rolling window of 60 samples. The infrastructure is
  already there.

### 1.2 What is broken or awkward for non-technical users

- **The Playground is unusable out of the box.**
  `Agent/TesseraAgentLoop.swift:328-380` ships a `PlaceholderLLMProvider`
  that echoes input. The default `TesseraLLMProviderFactory` selection
  is `placeholder` unless the user has filled `llmProviderType`,
  `remoteAPIBaseURL` etc. in Settings (`Settings/TesseraSettingsKey`).
  So a brand-new user lands on "Ask the Tessera agent...", types
  something, and gets back "I'm the Tessera Studio agent (placeholder)
  ... Try asking me to list models." This is the single worst first-run
  experience in the app. LM Studio, Jan, Msty, and GPT4All all boot to
  a working model within two clicks.
- **Onboarding is a static brochure.** `Views/OnboardingView.swift`
  has three pages: Welcome, Set Up Your Models (a directory text-field
  plus a disabled "Download a Starter Model" button at line 86-88),
  and Meet the Agent. Page two does not actually download anything.
  There is no model onboarding, no first-inference demo, no engine
  setup. The disabled button is a known unfinished surface.
- **The Library hides how to actually use a model.**
  `Views/LibraryView.swift` shows a `LazyVGrid` of `ModelCardView`s
  with badges and metadata (Family/Params/Bits/Size). There is no
  "Chat with this", no "Quantize this", no default action - the entire
  view is read-only metadata. `scanModels()` (line 63) hard-codes
  `~/Models/tessera` and `~/Models`, and the `effectiveBits` field is
  fabricated from filename heuristics (line 84: `isTessera ? 3.5 :
  4.5`) rather than read from the GGUF metadata that the engine
  actually validates.
- **Every tool surfaces as text in a chat bubble.**
  `Views/ChatBubbleView.swift` renders tool calls via
  `ToolCallView(record:)`. The Approval sheet
  (`Views/ApprovalSheet.swift`) shows the tool name and arguments in
  monospaced caption but no plain-language summary, no risk badge, no
  "what this will do" preview. A non-technical user approving
  `quantize { model_path, output_path, policy_path }` is approving
  gibberish.
- **Telemetry is always-on, always-technical.** `TelemetryDrawer` is
  docked at the bottom of every screen
  (`ContentView.swift:96`). It shows tok/s, MB, GPU %, kernel ms.
  For a non-technical user, this is noise on every screen, with no
  plain-language interpretation. There is no "calm" mode.
- **The four destinations are all expert-flavored.** "Library",
  "Runs", and "Learning" are nouns that mean nothing to a
  non-technical user. Compare Apple HIG ML guidance: "AI suggestions
  are useful, but they are not the user's primary intent" - they
  should defer, not headline.

### 1.3 What is awkward for ML experts

- **The four pipeline views are file-importers, not explorers.**
  `Views/AcceptanceGateView.swift:51-59`, `Views/ABReceiptView.swift:
  54-62`, `Views/L2DivergenceView.swift:66-74` all start with an empty
  state and a "Load Report..." fileImporter. There is no connection
  between a `RunRecord` (which already carries the analytics report
  envelope in `RunRecord.analyticsReport`, used in `Views/RunsView.swift:
  172-180`) and these views. The expert has to know where the JSON
  lives on disk.
- **No side-by-side comparison surface.** The A/B Receipt view
  (`Views/ABReceiptView.swift`) compares offline-proxy vs kernel-direct
  fitness for ONE run. There is no view that compares two runs, two
  models, or two engines on the same prompt. The design doc 1.1 calls
  out "the A/B moment" as the hero of the iPhone demo; the Mac app
  does not have it.
- **Telemetry has no per-tensor or per-layer granularity.**
  `Models/TelemetrySample.swift` carries tok/s, MB, GPU %, kernel ms,
  ANE MW. There is no per-tensor L2 divergence in the live telemetry,
  no per-kernel latency LUT (the design doc 5.3 calls for both). The
  expert gets the same four sparklines as the non-technical user.
- **The Learning tab is a List of LabeledContent rows.**
  `TesseraStudioMac/Views/LearningDashboardView.swift` is a flat
  `List` with sections for Capability / Adaptation / Teachers /
  Foraging / Curation, each row a `LabeledContent`. No charts (the
  comment at line 7 explicitly says "no Charts framework"), no
  timeline, no relationship between rows. For 4.8K LOC of subsystem,
  this is undersold.
- **No lineage view.** A `RunRecord` is a row in a table
  (`Views/RunsView.swift:31-56`). There is no parent/child
  relationship shown between, say, the calibrate run that produced
  the imatrix, the evolve run that consumed it, the quantize run that
  consumed the policy, and the evaluate run that gated the result.
  The receipts exist in the data model (`Models/QuantizationReceipt.
  swift`) but are not graphed.
- **Settings hides the autonomous-training dials.**
  `TesseraStudioMac/Views/SettingsView.swift:50-58` exposes a flat
  `TabView` (General/Agent/Model/Autonomy/Advanced). The Autonomy tab
  has the learned-permission ratchet snapshots, YOLO session
  controls, etc. - but the relationship between "what I approve here"
  and "what the agent does in the Playground" is invisible at the
  point of action. The autonomy research doc warns about this:
  disuse (under-trust) and misuse (over-trust) are both failure
  modes, and the calibration must be visible at the moment of action.
- **The ChatBubble loses the streaming token-level latency.** The
  agent loop streams `AgentEvent.text` chunks
  (`TesseraAgentLoop.swift:132-134`); the Playground accumulates them
  into `streamingText` (`PlaygroundView.swift:117`) and renders a
  single `ChatBubbleView` with `isStreaming: true`. The per-token
  timing that the engine knows is dropped on the floor; only the
  rolled-up `TelemetrySample.tokensPerSecond` survives.

### 1.4 Summary of the gap

The app has the **machinery** of a SOTA dual-audience product (agent
loop, layered safety, schema-versioned receipts, telemetry, four
schemas of analytics) but the **presentation** is uniformly expert,
flat, and file-driven. There is no progressive disclosure; everything
is shown at the same depth to everyone.

---

## 2. Design principles

These eight principles govern every recommendation below. Each is
grounded in either a SOTA peer's demonstrated win or an Apple-native
pattern.

**P1. Calm default, depth on demand.** Steal directly from Apple's
Keynote/Numbers inspector pattern (`support.apple.com/guide/keynote/
tan391376b09/mac`): the right-hand Format/Animate/Document inspector
shows only what is relevant to the selection, with deeper controls
behind disclosure triangles. The non-technical user sees the calm
default; the expert unwraps the depth. Never destroy information -
just defer it.

**P2. The model is a document.** A loaded model is the central object
of the app, like a document in Pages. It has a model card (the
document's title page), a chat surface (the document's working
canvas), an inspector (the document's Format panel), and a history
(the document's versions). Every screen is "what is this model doing
right now" or "what has this model done". LM Studio gets this right
with the model browser; Msty gets it right with "buckets" tied to
models (`msty.ai`).

**P3. Every action is reversible or previewable.** An extension of
the autonomy-calibration research's negativity-bias finding
(`docs/research-autonomy-calibration-2026-07-31.md` section 3): trust
is destroyed faster than it is built. So every destructive action
(quantize, convert, purge training data, evolve with a new policy)
offers either a dry-run preview ("this will produce a 3.8-bit GGUF
at ~4.1 GB, replacing nothing") or a one-click revert. Never a
preview-less commit.

**P4. Two languages, one screen.** Every expert surface has a
plain-language summary at the top and the numeric detail below it.
The AcceptanceGateView already half-does this (the verdict header at
line 63-83 says PASS/FAIL before the bars). Generalize: every
analytics card opens with a single sentence ("the composite policy
beats the best single proxy by 7.2% on held-out tensors"), then the
chart. This is the Apple Intelligence "Deference Principle": AI
output occupies a secondary visual tier
(`artofstyleframe.com/blog/designing-for-apple-intelligence-ui-2026/`).

**P5. Receipts are the trust surface.** Every action the agent takes
is already logged through the learned-permission ratchet
(`Agent/TesseraApprovalEngine.swift:179-197` records every gate
outcome). Surface those receipts at the moment of action - in the
chat, not in Settings. This is what Cursor's Auto-review gets right
and what Claude Code's 93% approval rate gets wrong
(`docs/research-autonomy-calibration-2026-07-31.md` section 1).

**P6. Telemetry is opt-in by granularity, not by presence.** The
bottom telemetry drawer is always present; what changes is the depth.
At the top level it is one number ("fast" / "moderate" / "slow" with
tok/s hidden). At the unwrapped level it is the existing four
sparklines. At the deepest level it is per-tensor, per-kernel, with
the L2 divergence hot list. The user chooses the depth once and the
app remembers per audience mode.

**P7. The agent is a co-pilot, not a servant.** Apple HIG ML says AI
suggestions "are not the user's primary intent"
(`artofstyleframe.com/blog/designing-for-apple-intelligence-ui-2026/`).
So the agent never takes over the screen; it offers, the user
accepts, and the result is rendered deferentially. The agent's
"thinking" is collapsed by default (the design doc 5.5 already
specifies this for CoT; extend it to all agent reasoning).

**P8. Native, not scientific.** The ML results are numbers; the
surface should feel like Numbers.app, not like Jupyter. Use SF
Rounded for headline metrics, SF Pro for body, system colors with
restrained accent (purple, inherited from the existing
`OnboardingView` accent), monospaced digits only where the value is
the point (gauges, tables, code). Dark mode is a first-class target
because the existing app uses `.background`, `.quaternary`, and
`.secondary` consistently - the bones are already there.

---

## 3. Information architecture

### 3.1 The mode axis (the dual-audience mechanism)

The single biggest move: introduce a **mode axis** on top of the
existing four destinations. Mode is a setting (`tessera.settings.
audienceMode`) with three values:

- **Simple.** For the non-technical user. Chat-first. No pipeline
  surface. Calm telemetry. Plain-language status. The Library
  becomes a model picker, not a metadata grid. The agent runs
  silently unless it needs approval.
- **Standard.** The default for someone who has chatted and wants to
  try a Tessera-quantized model. Library shows effective bits,
  badges, recommended action. Pipeline surfaces exist behind a
  disclosure ("Make this smaller / faster"). Receipts are visible
  but collapsed.
- **Studio.** The expert mode. Every existing expert surface is
  unwrapped: pipeline editor, analytics dashboards, per-tensor L2,
  A/B comparison, the Learning dashboard. Telemetry is at full
  depth. Settings exposes the Autonomy tab.

The mode is switchable from the toolbar (a segmented control with
three dots) and via a "More" menu. Mode switches are animated with a
brief crossfade; no data is destroyed. Mode is per-window on Mac.

**Why three modes and not two.** Cursor's Ask/Edit/Agent modes
(`medium.com/@roberto.g_infante/mastering-cursor-ide-10-best-practices`)
demonstrate that two modes (ask vs do) is too coarse: the "I want to
explore" user and the "I want to ship" user have different needs. The
middle "Standard" mode is where most users will live; it is the
LM-Studio-style "polished desktop GUI" experience that
`llmcheck.net/software` identifies as the Mac default.

### 3.2 The reshaped destination map

The four destinations persist but their content is mode-dependent:

```
                Simple            Standard            Studio
                ------            --------            ------
Library    ->  Pick a model    -> Pick + taste    -> Pick + inspect
Playground ->  Just chat       -> Chat + tools    -> Chat + agent
Runs       ->  (hidden)        -> Recent + verdict-> Lineage + analytics
Learning   ->  (hidden)        -> (hidden)        -> Receipts + autonomy
```

In **Simple mode**, the sidebar shows only **Playground** (relabeled
"Chat") and **Library** (relabeled "Models"). Runs and Learning are
hidden. The Telemetry drawer shrinks to a single "fast/moderate/slow"
chip. The chat surface has no approval sheets - everything that
would prompt is either auto-handled or surfaced as a plain-language
inline suggestion.

In **Standard mode**, all four destinations appear. Runs becomes
useful: it shows recent runs as cards with a verdict (PASS/FAIL) and
a one-line summary. The Learning tab remains hidden until the user
opts in (it is an advanced concept).

In **Studio mode**, every existing expert surface is unwrapped. The
Learning tab appears with the full receipt dashboard. Settings
exposes the Autonomy tab. The toolbar gains a "Compare" button
(opens the new A/B Compare view).

### 3.3 The inspector panel (the depth mechanism)

Every destination gets a right-hand inspector panel modeled on
Keynote's Format inspector. The inspector is the universal
progressive-disclosure surface:

- In **Library**, the inspector shows the selected model's full
  metadata: GGUF path, .mlmodelc status, sidecar policy, imatrix,
  last run, effective bits, kernel version. In Simple mode the
  inspector is hidden; in Standard it is a "Details" disclosure; in
  Studio it is always-visible.
- In **Playground**, the inspector is the "model scope" panel:
  current engine, context length, GPU layers, temperature, system
  prompt, approval level for the active session. This is where the
  expert unwraps inference knobs; the non-technical user never sees
  it.
- In **Runs**, the inspector shows the selected run's full receipt,
  the lineage chain, the export options. Today's `RunDetailSheet`
  (`Views/RunsView.swift:158-237`) becomes the inspector content.
- In **Learning**, the inspector shows the selected receipt's
  provenance: who approved, what gate fired, what world-signal
  grounded it.

The inspector follows Apple's "Format" panel convention: a vertical
strip on the right edge of the window, ~280pt wide, with
disclosure-triangle sections. It collapses to a thin strip with a
single "i" button when not needed.

---

## 4. Primary flows

Each flow is specified for both audiences: what the non-technical
user sees (Simple mode) and what the expert unwraps (Studio mode).

### 4.1 Onboarding (non-technical)

**Today:** three static pages, disabled download button
(`OnboardingView.swift:86-88`).

**New flow (six steps, all named):**

1. **Welcome.** Plain-language: "Tessera runs AI on your Mac, fully
   private. Pick a model and start chatting." One button: "Choose a
   starter model."
2. **Model pick.** A grid of three pre-qualified models (bundled or
   one-click-download): a small fast one (e.g. Gemma 3 4B Q4), a
   medium Tessera-quantized one, and a reasoning-capable one. Each
   has a plain-language description ("Fast - good for quick chats",
   "Balanced - Tessera-tuned for Apple Silicon", "Smart - reasons
   step by step"). This is the LM Studio model-browser pattern.
3. **Engine check.** A non-interactive progress card: "Setting up
   the engine...", then "Ready." No knobs.
4. **First chat.** The user lands on the Playground with a
   pre-populated first message: "Hi! What can you help me with?" The
   model streams a reply. The chat surface is in Simple mode: no
   telemetry, no token-budget bar, no tool calls.
5. **Mode prompt.** After the first chat, a sheet: "Want more
   control? Switch to Standard mode any time." with a one-tap
   dismiss. This is the only place mode is mentioned by name in
   onboarding.
6. **Done.** The user is in Simple mode. The sidebar shows only
   Chat and Models.

**What the expert unwraps:** from any step, an "Advanced" link in
the corner opens the existing directory-picker and a provider
selector (remote API / on-device / placeholder). On finish, the
expert is dropped into Standard or Studio mode per their choice.

This addresses the worst problem in the app today (the
PlaceholderLLMProvider default) by giving every new user a working
model before they ever see Settings.

### 4.2 First chat

**Simple mode:**

- The Playground shows: the conversation, an input bar, a small
  "Tessera" header with the model name. No token-budget bar. No
  telemetry drawer. No tool-call rendering. The agent's
  `AgentEvent.toolCall` events are silently swallowed if the model
  is the bundled starter (it does not call tools); if a research
  model is loaded and tool calls happen, they are rendered as
  "Looking something up..." with a single tap to expand.
- Reasoning blocks (when the model is reasoning-capable) collapse
  by default per design doc 5.5; tapping "Thought for 4.2s" expands.
- The only chrome is a small "i" button (top-right) that opens the
  inspector, which in Simple mode just shows "Model: <name>,
  Engine: On-Device".

**Studio mode:**

- The full TokenBudgetView (`Views/TokenBudgetView.swift`) appears.
- The Telemetry drawer is docked and expanded by default.
- Every `AgentEvent` is rendered: thinking as a faint italic, text
  as the bubble, toolCall as a `ToolCallView` with arguments, and a
  new per-token timing strip on the bubble (the engine's latency
  per token, currently dropped).
- Approval sheets (`Views/ApprovalSheet.swift`) include a risk badge
  (low/medium/high) and a plain-language summary, not just the raw
  arguments.

### 4.3 Downloading a model

**Today:** there is no model downloader in the app.
`LibraryView.scanModels()` only lists files in `~/Models/tessera`
and `~/Models`. The `LoadModelTool` requires a path that already
exists on disk.

**New flow:**

- Library gains a "Add Model" button (top-right of the toolbar) that
  opens a sheet with three options: "From Hugging Face", "From a
  File", "From a URL". The Hugging Face path is a search field plus
  a list of results; this is the LM Studio model browser pattern
  (`lmstudio.ai`).
- Downloads show progress in the model card itself (a circular
  progress over the badge area). The card is greyed out until the
  download completes, then becomes a normal card.
- In **Simple mode**, only the starter catalog is shown; no search.
- In **Studio mode**, the inspector shows the GGUF metadata, the
  policy/sidecar status, and a "Quantize this" affordance that
  jumps to the pipeline editor pre-populated with this model.

### 4.4 Running the quantize pipeline (expert, Studio mode only)

**Today:** the agent will run `quantize` if you type "quantize ..."
into the Playground. The PlaceholderLLMProvider recognizes the
keyword (`TesseraAgentLoop.swift:357-367`). There is no pipeline
editor in the UI; the design doc 5.3 specifies one but it was never
built.

**New flow:**

1. From a model card, the expert clicks "Quantize" (Studio mode
   only). This opens the new **Pipeline Editor** (replaces the
   never-built `QuantizationPlanEditor` from design doc 2.1).
2. The Pipeline Editor is a horizontal stepper with five stages:
   Calibrate, Evolve, Quantize, Convert, Evaluate. Each stage is a
   card with inputs (corpus path, target bits, generations, etc.)
   and an output preview ("will produce: imatrix.dat, ~120 MB").
3. Each stage maps to one of the existing tools: `CalibrateTool`,
   `EvolveTool`, `QuantizeTool`, `ConvertTool`, `EvaluateTool`. The
   editor does not replace the tools; it composes them into a plan
   and runs them via the agent loop with auto-approval (the user
   has explicitly chosen this pipeline, so per-stage approval is
   not needed - but the agent loop's safety spine still runs and
   will reject forbidden actions).
4. The run creates a `RunRecord` (already in the data model) and
   appears in the Runs view. As stages complete, their receipts
   are appended to the run's `analyticsReport` envelope
   (`RunRecord.analyticsReport` already supports all four schemas).
5. The Evaluate stage's verdict becomes the run's
   `acceptanceVerdict` (`Views/RunsView.swift:42-49` already
   renders this).

The pipeline editor is the answer to the design doc's never-built
QuantizationPlanEditor. It is also where the expert gets
reproducibility: every plan is saved to `~/Library/Application
Support/TesseraStudio/plans/<id>.json` (per design doc 5.3) and
can be re-run or shared.

### 4.5 Inspecting an acceptance gate (expert)

**Today:** `Views/AcceptanceGateView.swift` opens empty with a
fileImporter. The expert has to find the JSON on disk.

**New flow:**

- From a RunRecord in RunsView (Studio mode), the expert clicks the
  run row to open the inspector. The inspector's "Receipt" section
  shows the acceptance verdict inline (the existing
  `RunDetailSheet.receipt` flow at line 165-167 already does this).
- The expert clicks "Open in Analytics" to push the full
  AcceptanceGateView as a detail view, pre-populated with the run's
  verdict (no fileImporter). The view itself stays as it is - it
  is already good.
- The new affordance: a "Compare with another run" button at the
  top of the view that opens the A/B Compare surface (see 4.6)
  with this run pre-loaded on one side.

### 4.6 Comparing two runs (expert)

**Today:** there is no two-run comparison view. The ABReceiptView
compares offline-proxy vs kernel-direct for one run.

**New flow:**

- A new top-level **Compare** view (Studio mode toolbar button, also
  reachable from any RunRecord's context menu). Layout: two columns,
  each holding a run picker. Selecting a run on each side renders
  the four analytics surfaces side-by-side: archive occupancy,
  acceptance verdict, A/B receipt, L2 divergence.
- The headline is a single composite verdict: "Run B is 7.2% better
  on held-out tensors, 0.4 bits lower, 2.1 GB smaller, 12% slower
  at decode." This is the design doc 1.1 "A/B moment" extended to
  any two runs.
- The view supports exporting the comparison as a single PDF receipt
  (extending the existing `ReceiptPDFRenderer` referenced in
  `Views/RunsView.swift:239`).

### 4.7 Exporting an artifact

**Today:** `Views/ExportView.swift` (280 LOC) is the export surface,
reachable from `ContentView`'s `.sheet(item: $exportItem)` for chat
conversations (`ContentView.swift:58`). The `RunDetailSheet` has
"Export Receipt as PDF" and "Export Charts as PNG" buttons
(`Views/RunsView.swift:206-208`).

**New flow:** the export surface is unified. A single "Export" menu
in the toolbar (Studio mode) or a Share-style sheet (Simple mode)
offers:

- Conversation as Markdown / JSON / PDF (existing).
- Run receipt as PDF (existing).
- Run charts as PNG (existing).
- Model artifacts: the quantized GGUF, the .mlmodelc, the policy
  JSON, the imatrix (new - today these are just paths in the file
  system).
- A "reproducibility bundle" - a single .zip with the run's full
  receipt, the policy, the imatrix, the model hash, and a
  re-run command. This is the W&B "Reports" pattern
  (`wandb.ai/wandb/intro/reports/A-Few-of-Our-Favorite-W-B-Reports`)
  adapted for a local-first app.

---

## 5. Specific component designs

Twelve named components. Each references the existing Swift file it
extends or replaces.

### 5.1 AudienceModeToggle (new)

**Purpose:** the three-segment toolbar control that switches between
Simple / Standard / Studio.

**Behavior:** segmented `Picker` with three SF Symbols (a person, a
person with a gear, a beaker). On switch: crossfade, persist to
`@AppStorage("tessera.settings.audienceMode")`, animate the sidebar
filter (Runs/Learning appear/disappear), resize the Telemetry drawer.
Never destroys data; if a tool is mid-flight in Studio mode and the
user switches to Simple, the tool keeps running but its UI is hidden
until completion.

**Audience:** everyone. This is the dual-audience mechanism's front
door.

**Replaces:** nothing; new. Lives in `TesseraStudioMac/App/
ContentView.swift`'s toolbar.

### 5.2 ModelCardV2 (extends `LibraryView.ModelCardView`)

**Purpose:** the model card becomes the dual-audience surface for a
model.

**Behavior:**

- Top row: model name, a single primary badge ("Ready" / "Tessera" /
  "Stock"), and a context menu.
- Body (Standard / Studio): effective bits, family, size, ANE/Metal
  badge, sidecar status - the existing rows.
- Footer action area (mode-dependent):
  - Simple: a single "Chat" button.
  - Standard: "Chat" + "Details" disclosure.
  - Studio: "Chat" + "Quantize" + "Inspect" + "Compare".
- Long-press / right-click: full context menu (rename, delete,
  export, show in Finder).

**States:** loading (download progress ring), ready (default),
error (red badge with explanation), stale (the GGUF was deleted
out from under us).

**Audience:** everyone; the footer action set changes by mode.

**Replaces:** `Views/LibraryView.swift:118-165` (`ModelCardView`).

### 5.3 ModelInspector (new)

**Purpose:** the right-hand Format-inspector for a selected model.

**Behavior:** vertical strip, ~280pt, with disclosure-triangle
sections: Identity (name, family, params, hash), Quantization
(effective bits, qtype, kernel version, sidecar path), Calibration
(imatrix path, corpus, token count, modality scales), Runtime (last
loaded, last eval, tok/s typical), Lineage (parent run, child
artifacts).

**Audience:** Standard (collapsed by default), Studio (open). Hidden
in Simple.

**Replaces:** the static metadata grid in `ModelCardView` (which
becomes the always-visible summary); the inspector is the unwrapped
depth.

### 5.4 ChatSurfaceV2 (extends `PlaygroundView`)

**Purpose:** the chat surface, mode-aware.

**Behavior:**

- Top bar (always): model name, an "i" button (opens
  ModelInspector), a mode-aware menu (engine selector in Studio,
  hidden in Simple).
- Top bar (Studio only): token budget pill (existing
  `TokenBudgetView`, miniaturized), reasoning-mode toggle.
- Conversation: `ChatBubbleView`s, with three additions:
  - Tool calls render with a plain-language summary first
    ("Quantizing Gemma 4 12B...") and a disclosure to show the raw
    arguments.
  - Reasoning blocks render with the existing `ThinkingBlock`
    pattern (design doc 5.5).
  - Each assistant bubble carries a per-token timing strip on
    hover (the engine knows; we currently drop it).
- Input bar: TextField + send (existing), plus a "tools" picker in
  Standard+ that toggles which tools the agent may call this session.
- Approval sheets (`Views/ApprovalSheet.swift`) gain a risk badge
  and a plain-language summary; the existing raw-arguments view
  moves to a disclosure.

**Audience:** everyone; chrome varies by mode.

**Replaces:** `Views/PlaygroundView.swift` (kept structurally,
re-skinned).

### 5.5 TelemetryInspector (extends `TelemetryDrawer`)

**Purpose:** the bottom drawer becomes a multi-granularity surface.

**Behavior:** three depth tiers, controlled by mode and a "depth"
segmented control in the drawer handle:

- **Tier 1 (Simple).** A single chip in the bottom-right corner:
  "fast" (green), "moderate" (yellow), "slow" (red), based on tok/s
  thresholds. No drawer.
- **Tier 2 (Standard).** The existing four sparklines (throughput,
  memory, GPU, kernel) - the current default. Plus a one-line
  plain-language status ("running smoothly", "memory tight").
- **Tier 3 (Studio).** Tier 2 plus per-tensor L2 divergence hot
  list, per-kernel latency LUT, and a reasoning-channel breakdown
  (CoT vs final answer power draw, per design doc 5.5). New tabs
  along the drawer's top edge.

The existing `TelemetryMonitor` (`Views/TelemetryDrawer.swift:9-55`)
already polls at 500ms; Tier 3 extends the sample type rather than
the poll loop.

**Audience:** everyone; depth varies by mode.

**Replaces:** `Views/TelemetryDrawer.swift`.

### 5.6 PipelineStepper (new)

**Purpose:** the five-stage pipeline editor described in flow 4.4.

**Behavior:** a horizontal `HStack` of five stage cards
(Calibrate / Evolve / Quantize / Convert / Evaluate). Each card has
an icon, a status (pending / running / done / failed), and an
expandable inputs panel. A "Run" button at the right end starts the
sequence via the agent loop. A "Save Plan" button persists the
configuration.

**States:** per stage: pending (grey), running (animated), done
(green check, with a one-line summary of what it produced), failed
(red, with a tap-to-see-error), skippable (some stages can be
turned off, e.g. skip Convert if the target is a GGUF).

**Audience:** Studio only.

**Replaces:** the never-built `QuantizationPlanEditor` (design doc
2.1). Composes the existing tools rather than replacing them.

### 5.7 RunLineageView (new)

**Purpose:** a graph view of how runs, models, and artifacts relate.

**Behavior:** a `Canvas` or `ScrollView` with nodes for each run and
each artifact (imatrix, policy, GGUF, .mlmodelc, eval report), and
directed edges showing the "produced" relationship. Selecting a node
shows its detail in the inspector. The lineage is reconstructed from
the existing `RunRecord` provenance fields.

**Layout:** left-to-right time axis; nodes stack vertically by
parent. Color-coded by type (run = blue, model = purple, artifact =
green, eval = orange).

**Audience:** Studio only.

**Replaces:** the `RunsView` table view as the default Studio Runs
surface (the table is retained as a list-mode toggle).

### 5.8 AcceptanceReceiptCard (extends `AcceptanceGateView`)

**Purpose:** the acceptance verdict becomes a portable receipt card.

**Behavior:** the existing `verdictHeader` and `test1Panel` (lines
63-118 of `AcceptanceGateView.swift`) become a self-contained card
that can appear in three places: inline in a RunRecord's inspector,
as the top of the full `AcceptanceGateView`, and as a chip in the
chat surface when an evaluate tool completes.

The card always opens with a single plain-language sentence:
"PASS - the composite policy beats the best single proxy by 7.2%
on held-out tensors." Then the existing visualization. This is
Principle P4 in action.

**Audience:** Standard sees the sentence + a small PASS/FAIL chip.
Studio sees the full card.

**Replaces:** parts of `Views/AcceptanceGateView.swift`,
`Views/RunsView.swift` (the inline verdict rendering).

### 5.9 ABCompareView (new)

**Purpose:** the two-run comparison surface described in flow 4.6.

**Behavior:** two-column layout, each column with a run picker and a
rendered analytics surface. A composite summary at the top:
composite score, effective bits, file size, decode tok/s - with
deltas. An "Export comparison as PDF" button (extends
`ReceiptPDFRenderer`).

**Audience:** Studio only.

**Replaces:** nothing; new top-level view. Reaches into the existing
analytics rendering code (`ArchiveBrowserView`,
`AcceptanceGateView`, `ABReceiptView`, `L2DivergenceView`).

### 5.10 ApprovalSheetV2 (extends `ApprovalSheet`)

**Purpose:** the approval surface becomes legible.

**Behavior:** the existing `Views/ApprovalSheet.swift` is re-laid
out:

- A risk badge (low / medium / high) sourced from
  `TesseraActionVerifier.ruleBasedRisk` (already called in the agent
  loop at `TesseraAgentLoop.swift:184`).
- A plain-language summary: "The agent wants to quantize Gemma 4
  12B. This will take ~10 minutes and produce a 4 GB file. It will
  not modify the original model." Generated from a per-tool
  template.
- The raw arguments in a disclosure (the current top-level view
  becomes the disclosure's content).
- The approve / deny buttons stay, but with a third: "Always allow
  this kind of action" - which writes a user override via the
  existing `TesseraApprovalEngine.setOverride`
  (`Agent/TesseraApprovalEngine.swift:47`). This is the explicit
  ratchet-up affordance that Principle P5 demands at the moment of
  action.
- The receipt is logged (already done by the agent loop's
  `recordOutcome` call) and a small "Logged" indicator appears.

**Audience:** everyone who sees an approval (any mode where a
tool's approval level is `.prompt`).

**Replaces:** `Views/ApprovalSheet.swift`.

### 5.11 QuantizationKnobPanel (new)

**Purpose:** the expert's per-tensor policy editor.

**Behavior:** a sheet or inspector section that shows the current
policy as a sortable table: tensor name, current qtype, current
alpha, current sensitivity rank (from the L2/L3/L4 layers of the
runtime-aware pipeline). The expert can override individual qtypes
or alphas; the panel shows the predicted effective bits and
predicted perplexity impact (from the E2E fidelity predictor, per
design doc 5.3).

This is the AWQ-evolve knob surface. Today evolve is a single LLM
tool call with five integer arguments (`Tools/EvolveTool.swift`).
The panel decomposes it into a visible, tweakable policy.

**Audience:** Studio only.

**Replaces:** nothing; new. Composes the `evolve` tool with a
policy reader/writer.

### 5.12 LearningReceiptsPanel (extends `LearningDashboardView`)

**Purpose:** the Learning subsystem surfaces as receipts, not just
internal state.

**Behavior:** the existing flat `List` in
`TesseraStudioMac/Views/LearningDashboardView.swift` is rebuilt as
four card sections, each with a chart:

- **Capability.** A radar chart of the multi-axis capability score
  (the existing `TesseraCapabilityScore.axisNames`). Current score
  overlaid on the previous, so drift is visible.
- **Adaptation.** A timeline of adaptation runs (the
  `TesseraAdaptationRecord` sequence) with guard verdicts.
- **Teachers.** A bar chart of effective weights and pass
  fractions. The world-gate pass fraction is the headline number.
- **Foraging + Curation.** A stacked bar over time showing how the
  source mix has shifted from remote toward local (the
  `TesseraForagingSource` enum is local-playbook / local-reference
  / remote - the design intent is local-over-remote).

Plus a fifth section: **Approval receipts** - the most recent N
approval decisions the agent logged, with the action class, the
gate decision, the user choice, and the source (rule-based ratchet
vs learned). This is what Principle P5 puts at the trust surface.

**Audience:** Studio only.

**Replaces:** `TesseraStudioMac/Views/LearningDashboardView.swift`.

---

## 6. The Learning subsystem question

### 6.1 What it is

The 4.8K-LOC Learning subsystem (`Sources/TesseraCore/Learning/`) is
the implementation of the self-improving loop
(`docs/self-improving-loop-design.md`). It is NOT a model trainer in
the traditional sense; it is the **autonomy + escalation +
curation + knowledge** substrate that makes the agent trustworthy
over time. Concretely, it owns:

- **The learned-permission ratchet**
  (`TesseraAutonomyService.swift`, `TesseraAutonomyDataModel.swift`,
  `TesseraAutonomyContracts.swift`, 1.1K LOC) - a per-action-class
  permission store that learns from approval/denial history. This is
  the unique feature; no competitor has it.
- **The escalation ensemble** (`TesseraEscalationService.swift`,
  `TesseraTeacherAssessor.swift`, `TesseraApproverNetwork.swift`,
  ~850 LOC) - a confidence-triggered teacher router that fans
  genuinely-hard cases out to a small ensemble, weighted by a
  world-gated pass fraction.
- **The curation pipeline** (`TesseraCurationService.swift`,
  `TesseraAnonymizerService.swift`, ~420 LOC) - the dedup, quality,
  preference-pair formation, and secret-scrubbing for the loops
  that train.
- **The knowledge stores** (`TesseraReferenceKnowledgeStore.swift`,
  `TesseraReasoningPlaybookStore.swift`, ~200 LOC) - volatile,
  expiring, lookup-able.
- **The training orchestrator + adaptation scheduler**
  (`TesseraTrainingOrchestrator.swift`,
  `TesseraAdaptationScheduler.swift`, ~550 LOC) - shells out to the
  CLI tools for finetune + adapt, with the capability-eval guard.
  Note the honesty ceiling: actual LoRA training is a plug-in point
  that returns `adapted=false` in v1.
- **The drafting head scaffold** (`TesseraTrackR.swift`) - one trunk
  with DFlash-seeded and DSpark-seeded heads; explicitly a scaffold
  today.
- **The world-signal observer + miscalibration detector**
  (`TesseraWorldSignalObserver.swift`,
  `TesseraMiscalibrationDetector.swift`) - the ground truth that
  gates every update, plus a regime-shift detector that tightens
  autonomy when approval behavior flips.
- **Nine Learning tools** (`Tools/Learning/*.swift`) - exposed to
  the agent loop: `LookupDocsTool`, `QueryPlaybookTool`,
  `RecordOutcomeTool`, `EscalateReasoningTool`,
  `AnonymizeWorktreeTool`, `EscalateWithCodeTool`,
  `RunAdaptationTool`, `RunTrainingTool`, `InspectLearningTool`,
  `PurgeTrainingDataTool`.

### 6.2 How it should surface (and to whom)

The Learning subsystem is **invisible to non-technical users and
mostly invisible to standard users.** It surfaces in three places,
all Studio-only, all framed as transparency rather than as
controls:

1. **At the moment of approval (Principle P5).** When the
   `ApprovalSheetV2` (component 5.10) appears, it shows the
   learned-permission state for this action class: "You've approved
   this kind of action 12 times; the agent has handled it
   correctly each time." Or, after a regime shift: "You've denied
   this kind of action 3 times in a row - the agent will keep
   asking." This is the receipt that the autonomy research doc says
   is missing from every shipping competitor.
2. **In the Learning tab (component 5.12).** A Studio-only
   transparency dashboard showing capability scores, adaptation
   runs, teacher weights, foraging mix, and recent approval
   receipts. No "train now" button; the training is a scheduled,
   idle-window activity that the user opts into via Settings.
3. **In Settings -> Autonomy** (already exists, keep it). The
   learned-permission entries, the YOLO session controls, the
   miscalibration-detector state. This is where the user purges
   learning data (`PurgeTrainingDataTool` is already a tool).

### 6.3 What NOT to surface

- The drafting head scaffold (`TesseraTrackR`) and the training
  orchestrator's CLI shelling (`TesseraTrainingOrchestrator`) are
  internals. They do not get a UI in any mode. If the user wants to
  see them, they read the receipts.
- The capability-eval guard's per-axis scores are interesting to
  the architect but not to a user; the radar chart (component 5.12)
  is the right abstraction.
- The reference knowledge store and the reasoning playbook store
  are not directly browsable. They appear only as the provenance of
  an agent answer ("this answer drew on your local reference
  store, 3 entries").

### 6.4 The simple version

For the non-technical user the entire Learning subsystem is one
sentence, shown the first time the agent escalates: "Tessera learns
your preferences over time. You can review or delete what it has
learned in Settings." With a "Got it" button. Nothing else.

---

## 7. Visual language

This is not pixel-pushing; it is the tonal/typographic/color
direction.

### 7.1 Tone

Native, calm, slightly technical-but-friendly. The reference points
are:

- **Apple Numbers** for the calm-default inspector + table aesthetic
  (Principle P1, P8). Numbers uses SF Pro, system colors, no
  decoration; the format panel on the right is the model for the
  ModelInspector.
- **Keynote** for the Format-inspector panel convention (right-hand
  strip, disclosure triangles) - the same pattern Apple's HIG
  documents for ML apps (`developer.apple.com/design/
  human-interface-guidelines/machine-learning`).
- **Linear** for the dense-list-with-calm-typography pattern. Linear
  shows a lot of information without feeling crowded; it is the
  reference for the Runs list and the Learning dashboard.

The app should NOT look like:

- **Jupyter / VS Code**. The ML results are numbers, but the
  surface should not feel scientific. No monospaced body text, no
  gridlines everywhere, no terminal-green-on-black for telemetry.
- **A chat app**. The Playground is a chat surface, but the rest of
  the app is a workbench; it should not feel like Discord or
  Slack.

### 7.2 Type

- **Headline metrics** (tok/s, effective bits, PPL, the composite
  score, the world-gate pass fraction): SF Rounded, bold, large.
  This is the Apple-Intelligence "use rounded for the headline
  number" pattern.
- **Body**: SF Pro, regular, dynamic type. Existing app already
  uses this.
- **Captions / metadata**: SF Pro, caption, secondary color.
- **Monospaced**: SF Mono, ONLY where the value is the point:
  gauge readings, table cells with numbers, code, file paths,
  argument names in approval sheets. The existing
  `.monospacedDigit()` usage in `TelemetryDrawer.swift`,
  `AcceptanceGateView.swift`, etc. is correct; extend it
  consistently.

### 7.3 Color

- **System colors as the base** (`Color.primary`, `.secondary`,
  `.tertiary`, `.background`, `.quaternary`, `.tint`). The existing
  app uses these consistently - the bones are right.
- **Accent: purple**, inherited from the existing
  `OnboardingView.swift:46` (`foregroundStyle(.purple)`). Purple is
  Tessera's existing brand color; keep it. Used for: the AudienceMode
  toggle's Studio segment, primary CTAs, the agent's thinking state.
- **Semantic colors**:
  - Green: pass, ready, fast, approved-low-risk.
  - Yellow: warning, moderate, ask.
  - Red: fail, slow, denied, high-risk, irreversible.
  - Blue: the user (chat bubbles), information.
  - Orange: telemetry accent (matches existing
    `TelemetryDrawer.swift` kernel category).
- **Dark mode**: first-class. The existing `.background.opacity(0.5)`
  card pattern (`AcceptanceGateView.swift:203`) needs a small tweak:
  in dark mode the opacity should be 0.3 to avoid the card
  disappearing into the background. Audit every
  `.background(.background.opacity(...))` and
  `.background(.quaternary.opacity(...))`.

### 7.4 The Apple-Intelligence shimmer (sparingly)

Apple's iOS 18+ shimmer (a 2px animated outline using a four-color
`AngularGradient` rotating at 1.8s/loop) is the native AI
processing indicator
(`artofstyleframe.com/blog/designing-for-apple-intelligence-ui-2026/`).
Use it for the **first-token wait** in the Playground (the engine
is "thinking" before the first chunk streams) and for
**agent-initiated long-running actions** (a quantize run). Do NOT
use it for sustained telemetry, for results that have already
arrived, or as a permanent badge. Fades out over 0.4s when the
result renders. This is a directly stolen pattern, with the exact
timing from the cited article.

For non-Apple Intelligence models (any third-party model the user
loads), use the third-party alternative pattern: persistent corner
sparkles OR a slow breathing glow on the affected element, NOT
Apple's signature shimmer. This is the "whose AI are you trusting"
transparency rule.

### 7.5 Accessibility

- **Dynamic Type** end-to-end. Existing views largely comply; the
  `TelemetryDrawer`'s fixed `.frame(height: 32)` sparklines and
  `AcceptanceGateView`'s fixed `.frame(height: 110)` charts need
  to scale with Dynamic Type or provide a compressed alternative.
- **VoiceOver**: every model card, every run row, every analytics
  card needs a spoken summary that leads with the conclusion ("PASS
  run, Gemma 4 12B, 3.8 effective bits") rather than the visual
  structure. This is also Principle P4 (one sentence first).
- **Reduced motion**: the shimmer, the breathing-dot in the chat
  history, and any auto-expanding disclosures get static
  equivalents. The `agent-patterns-research.md` file's BreathingDot
  pattern already has the right shape; just gate it on
  `@Environment(\.accessibilityReduceMotion)`.
- **Color contrast**: the existing `.opacity(0.15)` backgrounds for
  badges (`LibraryView.swift:175`) are borderline in dark mode;
  audit with the Audit tab in Xcode (the Chrome DevTools MCP skill
  is the web analog).

---

## 8. Implementation phasing

Four phases. Each is a shippable milestone. Each cites which existing
views change and which new components land. Phases are ordered for
maximum leverage early.

### Phase 1: Make the app work for a brand-new user (2-3 weeks)

**Goal:** a non-technical user can install Tessera Studio and chat
within two minutes, with no Settings visit.

**Changes:**

- Wire a real default LLM provider. The PlaceholderLLMProvider
  (`TesseraAgentLoop.swift:328`) becomes a last-resort fallback;
  the factory (`TesseraLLMProviderFactory`) defaults to the
  on-device path with a starter model if one is bundled, else
  prompts for the model pick in onboarding.
- Replace `OnboardingView` (flow 4.1). Six steps, ending in a
  working chat. The "Download a Starter Model" button at line 86
  becomes functional.
- Add the `AudienceModeToggle` (component 5.1) to the toolbar.
  Default mode: Simple.
- Filter the sidebar by mode in `ContentView.swift`: Simple shows
  only Chat + Models.
- Reskin `LibraryView`'s `ModelCardView` (component 5.2) with a
  Chat button on the card.
- Shrink the `TelemetryDrawer` to Tier 1 (a single chip) in Simple
  mode.

**New components:** AudienceModeToggle, ModelCardV2 (partial).

**Files touched:** `Views/OnboardingView.swift`,
`TesseraStudioMac/App/ContentView.swift`, `Views/LibraryView.swift`,
`Views/TelemetryDrawer.swift`, `Engine/TesseraLLMProviderFactory.
swift`.

**Validation:** a brand-new user (recruited from a non-technical
pool) goes from install to first chat in under two minutes, with no
documentation.

### Phase 2: Make the agent legible (2-3 weeks)

**Goal:** every action the agent takes is legible to the user at the
moment of action; the receipts are visible.

**Changes:**

- Ship `ApprovalSheetV2` (component 5.10) with risk badge,
  plain-language summary, and "always allow this kind" affordance.
- Add a plain-language-summary template per tool
  (`TesseraTool` protocol gains a `func plainLanguageSummary(args:)
  String`). Each tool ships its template.
- Render tool calls in `ChatBubbleView` with the summary-first
  disclosure pattern.
- Add the per-token timing strip to assistant bubbles in Studio
  mode.
- Add the Learning receipt chip to the ApprovalSheetV2 ("you've
  approved this kind N times").

**New components:** ApprovalSheetV2.

**Files touched:** `Views/ApprovalSheet.swift`,
`Views/ChatBubbleView.swift`, `Views/ToolCallView.swift`,
`Agent/TesseraTool.swift`, every file under `Tools/` (to add the
summary template).

**Validation:** a non-technical user can correctly predict what a
tool call will do from the approval sheet alone, without reading
the arguments.

### Phase 3: Make the expert surface unwrappable (4-6 weeks)

**Goal:** the expert can run the pipeline, inspect every receipt,
and compare runs without leaving the app.

**Changes:**

- Ship the `ModelInspector` (component 5.3) as a right-hand panel.
- Ship the `PipelineStepper` (component 5.6). Composes the existing
  pipeline tools.
- Ship `RunLineageView` (component 5.7) as the Studio-default Runs
  surface (table retained as a toggle).
- Ship `ABCompareView` (component 5.9) at the top level of Runs.
- Ship `AcceptanceReceiptCard` (component 5.8) inline in run rows
  and inspectors.
- Connect the existing analytics views (`AcceptanceGateView`,
  `ABReceiptView`, `L2DivergenceView`, `ArchiveBrowserView`) to
  RunRecords so they open pre-populated, not via fileImporter.
- Ship the `QuantizationKnobPanel` (component 5.11).
- Add Tier 3 to the `TelemetryInspector` (component 5.5).

**New components:** ModelInspector, PipelineStepper, RunLineageView,
ABCompareView, AcceptanceReceiptCard, QuantizationKnobPanel,
TelemetryInspector Tier 3.

**Files touched:** `Views/RunsView.swift`, all four analytics
views, `Views/TelemetryDrawer.swift`, `Models/RunRecord` (the
analyticsReport plumbing), `Tools/EvolveTool.swift` (to support
the panel's policy read/write).

**Validation:** an expert can take a BF16 GGUF from download to a
Tessera-quantized .mlmodelc with an acceptance verdict, all in-app,
with every step's receipt visible.

### Phase 4: Make the model a document (3-4 weeks)

**Goal:** the model becomes the central object; lineage, artifacts,
and exports unify around it.

**Changes:**

- Ship the unified Export surface (flow 4.7).
- Ship the model downloader (flow 4.3) - Hugging Face search,
  file picker, URL download with progress.
- Ship the `ChatSurfaceV2` (component 5.4) full reskin: per-token
  timing, reasoning collapse, tools picker.
- Ship the `LearningReceiptsPanel` (component 5.12) with charts.
- Polish dark mode, accessibility, the Apple-Intelligence shimmer.

**New components:** ChatSurfaceV2 (full), LearningReceiptsPanel,
unified ExportView, model downloader.

**Files touched:** `Views/PlaygroundView.swift`,
`Views/ExportView.swift`, `TesseraStudioMac/Views/
LearningDashboardView.swift`, `Views/LibraryView.swift`.

**Validation:** the dual-audience promise is met: a non-technical
user can use the app for weeks in Simple mode without ever knowing
the pipeline exists; an expert can unwrap into Studio mode and
access the full SOTA workbench.

---

## 9. Risks and open questions

This is an honest list. Items here are things a prototype would need
to validate; they are not blockers but they are undecided.

- **R1: Mode switching may confuse users.** Three modes is more than
  Apple typically ships (Apple tends to two: standard + advanced).
  Cursor's three-mode Ask/Edit/Agent works because the modes are
  about *what the agent does*, not *what the user sees*. Tessera's
  modes are about what the user sees, which is closer to Numbers's
  single-mode-with-inspector pattern. A prototype should validate
  whether the mode toggle is necessary at all, or whether the
  inspector panel alone is enough. My recommendation is to ship the
  toggle in Phase 1 and A/B test it against a no-toggle variant in
  Phase 4.
- **R2: The PipelineStepper may overlap the agent loop.** Today the
  agent can run any tool from chat. The PipelineStepper is a
  structured way to compose five specific tools. If both paths
  exist, the user may be confused about which to use. Resolution:
  the PipelineStepper is the Studio-only structured path; the chat
  agent remains the general path. They share the same tool
  implementations, the same safety spine, and the same receipts.
- **R3: The Learning subsystem's honesty ceiling.** The training
  orchestrator returns `adapted=false` in v1
  (`TesseraTrainingOrchestrator.swift` per the design doc). Surfacing
  the Learning dashboard before training is real risks
  disappointing the expert who unwraps it. Resolution: the
  dashboard should show what IS real today (the receipts, the
  capability eval, the foraging mix, the curation pipeline) and
  clearly mark what is a plug-in point.
- **R4: Per-token timing in the chat bubble.** The engine knows
  per-token latency; the agent loop's `AgentEvent.text` chunk does
  not carry it. Adding it requires either a new event variant or
  extending `AgentEvent.text` to carry timing metadata. This is a
  protocol change with downstream test impact.
- **R5: The Apple-Intelligence shimmer is iOS 18+ / macOS 15+.
  ** Tessera Studio targets macOS 15 and iOS 18 per `Package.swift`,
  so this is fine, but the third-party-model breathing-glow
  alternative needs its own implementation. Steal the exact
  AngularGradient recipe from the cited article; do not improvise.
- **R6: Hugging Face download requires a network egress policy.**
  The autonomy research doctrine is "no egress by default" (the
  web search path was reshaped to a keyless DuckDuckGo default for
  exactly this reason, per design doc 5.4). A model download is a
  large, traceable egress. The download flow needs to disclose the
  egress, offer a private mirror configuration, and respect the
  same approval discipline as the research tool. Open question:
  is HF download approval-level `.prompt` or `.notify`? My
  recommendation: `.prompt` for the first download from a model
  author, `.notify` thereafter (mirrors the autonomy ratchet).
- **R7: The Runs lineage view depends on provenance fields that may
  not be populated.** `RunRecord` carries config and metrics but
  its parent/child relationship to other runs is implicit (output
  path of one run = input path of another). The lineage view needs
  either explicit parent-run IDs added to RunRecord or a path-based
  inference pass at view time. Path-based is fragile; explicit is
  better but requires a model migration.
- **R8: I did not validate with real non-technical users.** This is
  a blueprint from reading the source and the research. Every
  Phase-1 change should be validated with a 5-user study before
  Phase 2 commits. Specifically the onboarding flow (4.1) and the
  Simple mode chrome (4.2) are the highest-risk surfaces.
- **R9: The "TesseraRuntime" enum conflates backend and device.**
  `Models/TesseraRuntime.swift` has `.onDevice` (CoreML ANE),
  `.mlx`, `.privateCloud`. The visual badges in `ModelInfo.icon`
  map these to icons. But a Tessera-quantized model can run on
  Metal OR ANE; the runtime is chosen at load time, not at the
  model. The ModelCardV2 needs to reflect "this model supports:
  ANE, Metal" rather than "this model IS on_device". Open question:
  is this a model-level property or a run-level property?
- **R10: iOS scope.** This blueprint is Mac-focused. The iOS app
  (`TesseraStudioiOS/`) has its own ContentView and SettingsView
  and the design doc 1.1 specifies a chat-first iPhone demo. The
  mode axis translates (Simple is the iPhone default) but the
  inspector panel does not (no room). The iOS app needs a separate
  design pass that this blueprint does not provide.

---

## Sources

Primary internal:

- `/Users/user/Developer/GitHub/tessera/docs/audit-2026-07-29.md`
- `/Users/user/Developer/GitHub/tessera/docs/agent-patterns-research.md`
- `/Users/user/Developer/GitHub/tessera/docs/research-autonomy-calibration-2026-07-31.md`
- `/Users/user/Developer/GitHub/tessera/docs/tessera-studio-design.md`
- `/Users/user/Developer/GitHub/tessera/docs/self-improving-loop-design.md`
- `/Users/user/Developer/GitHub/tessera/TesseraStudio/Sources/TesseraCore/` (full source walk)
- `/Users/user/Developer/GitHub/tessera/TesseraStudio/Sources/TesseraStudioMac/` (full source walk)

Apple / native patterns:

- Apple Machine Learning HIG: https://developer.apple.com/design/human-interface-guidelines/machine-learning
- Apple Generative AI HIG: https://developer.apple.com/design/human-interface-guidelines/generative-ai
- Apple HIG hub (progressive disclosure pattern): https://developer.apple.com/design/human-interface-guidelines
- WWDC25 Discover ML and AI frameworks: https://developer.apple.com/videos/play/wwdc2025/360/
- WWDC25 Meet the Foundation Models framework: https://developer.apple.com/videos/play/wwdc2025/286/
- WWDC25 Prompt design and safety for on-device foundation models: https://developer.apple.com/videos/play/wwdc2025/248/
- Apple Foundation Models 2025 updates (research): https://machinelearning.apple.com/research/apple-foundation-models-2025-updates
- Apple Intelligence Foundation Language Models (arXiv): https://arxiv.org/pdf/2507.13575
- Designing for Apple Intelligence (concrete shimmer / deference patterns): https://artofstyleframe.com/blog/designing-for-apple-intelligence-ui-2026/
- Apple Keynote inspector panel: https://support.apple.com/guide/keynote/show-or-hide-sidebars-tan391376b09/mac
- WWDC22 The craft of SwiftUI API design (progressive disclosure): https://developer.apple.com/videos/play/wwdc2022/10059/
- Nielsen Norman Group, Progressive Disclosure: https://www.nngroup.com/articles/progressive-disclosure/

Local-LLM competitors:

- LM Studio (official): https://lmstudio.ai/
- LM Studio docs (model browser, GPU offload slider): https://lmstudio.ai/docs
- LM Studio multi-GPU controls blog: https://lmstudio.ai/blog/lmstudio-v0.3.14
- LM Studio lms load CLI: https://lmstudio.ai/docs/cli/local-models/load
- Best Software to Run Local AI on Mac 2026: https://llmcheck.net/software
- Open WebUI LM Studio comparison: https://docs.openwebui.com/alternatives/lm-studio/
- Msty (official): https://msty.ai/
- Msty reviews (Mason James, Medium): https://masonjames.com/blog/play-msty-for-me/ and https://medium.com/@elenasm/exploring-msty-studio-a-practical-ui-for-managing-local-llms-dc43d6780084
- Best Local AI App for Non-Technical Users (PromptQuorum): https://www.promptquorum.com/power-local-llm/local-ai-app-non-technical-users
- Ollama vs LM Studio vs Jan vs GPT4All (PromptQuorum): https://www.promptquorum.com/local-llms/local-llm-one-click-installers
- Free LLM Desktop Tools Comparison (SailingByte): https://sailingbyte.com/blog/the-ultimate-comparison-of-free-desktop-tools-for-running-local-llms/
- r/LocalLLaMA LM Studio vs Ollama vs GPT4All vs AnythingLLM: https://www.reddit.com/r/LocalLLM/comments/1bd9qqb/exploring_local_llm_managers_lmstudio_ollama/

Dual-audience expert surfaces:

- Cursor Ask / Edit / Agent modes overview: https://medium.com/@roberto.g_infante/mastering-cursor-ide-10-best-practices-building-a-daily-task-manager-app-0b26524411c1
- Cursor Agent Mode guide (Cowork.ink, includes YOLO): https://cowork.ink/blog/cursor-agent-mode/
- Raycast AI (command-K + chat): https://www.raycast.com/core-features/ai
- Linear x Raycast integration: https://linear.app/integrations/raycast

Conversation + artifact:

- Claude Artifacts explained: https://blog.gopenai.com/claude-artifacts-explained-the-feature-that-changes-how-you-use-ai-c13891cbf0b3
- ChatGPT Canvas vs Claude Artifacts for coding (r/ClaudeAI): https://www.reddit.com/r/ClaudeAI/comments/1fvydtc/chatgpt_canvas_vs_claude_artifacts_for_coding/
- Replit "Send to Replit" workflow: https://docs.replit.com/design/import-claude-designs-into-replit-design

ML observability:

- Weights and Biases vs MLflow (ContraCollective, UX focus): https://contracollective.com/blog/weights-biases-vs-mlflow-mlops-experiment-tracking-2026
- W&B Reports (collaborative dashboards): https://wandb.ai/wandb/intro/reports/A-Few-of-Our-Favorite-W-B-Reports--VmlldzozMTAzNjQ3
- MLflow vs W&B vs ZenML (ZenML): https://www.zenml.io/blog/mlflow-vs-weights-and-biases
- MLflow UI for run analysis (apxml): https://apxml.com/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/using-mlflow-ui
- Databricks MLflow run visualizations: https://docs.databricks.com/aws/en/mlflow/visualize-runs
- Arize Phoenix LLM tracing and observability: https://arize.com/blog/llm-tracing-and-observability-with-arize-phoenix/
- Arize Phoenix span and latency (resource hub): https://arize.com/resource-hub/ai-roi-framework-observability/
- Langfuse vs Phoenix (ZenMR): https://www.zenml.io/blog/langfuse-vs-phoenix

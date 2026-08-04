# Tessera Studio UX study - orientation facts

## The app at a glance
- TesseraStudio/ - a SwiftUI Mac (and iOS) app, 229 Swift files, ~65K LOC
- Package.swift describes a v2 architecture: CLlama (C bridge to libllama via dlopen),
  CTesseraFFI (C bridge to tessera.xcframework), TesseraCore (platform-independent:
  models, tools, agent, views), TesseraStudioMac (macOS shell), TesseraStudioiOS.
- Inference runs ON-DEVICE via libllama (the C bridge), not a server connection.

## Source map (read these to understand what exists)
- TesseraStudio/Sources/TesseraCore/Agent/ (8 files, 1280 LOC) - the agent loop,
  approval engine, safety decision, circuit breaker, action class/verifier
- TesseraStudio/Sources/TesseraCore/Tools/ (14 files, 1497 LOC) - the 10 tools:
  Calibrate, Quantize, Evaluate, Evolve, Convert, LoadModel, ListModels,
  InspectSidecar, TesseraResearch, WebSearch (+ DuckDuckGo/SearXNG/Tavily providers)
- TesseraStudio/Sources/TesseraCore/Views/ (16 files, 2855 LOC) - the UI surface:
  OnboardingView, PlaygroundView, LibraryView, RunsView, AnalyticsDashboardView,
  TelemetryDrawer, AcceptanceGateView, L2DivergenceView, ABReceiptView,
  ApprovalSheet, ChatBubbleView, ChatHistoryDrawer, ExportView, TokenBudgetView,
  ArchiveBrowserView + Renderers/
- TesseraStudio/Sources/TesseraCore/Learning/ (32 files, 4794 LOC) - the largest
  subsystem; understand what it does and whether it has a UI
- TesseraStudio/Sources/TesseraCore/Models/ (8 files, 1185 LOC)
- TesseraStudio/Sources/TesseraCore/Engine/ (7 files, 1185 LOC) - engine bridge
- TesseraStudio/Sources/TesseraStudioMac/Views/ - LearningDashboardView, SettingsView

## Docs to read FIRST (they have prior context)
- docs/audit-2026-07-29.md - known issues audit (section 6 mentioned auto-MTP)
- docs/agent-patterns-research.md - agent design research
- docs/research-autonomy-calibration-2026-07-31.md - autonomy calibration research
- docs/research-alignment-2026-07-30.md
- docs/tessera-studio-design.md (referenced in Package.swift; may or may not exist -
  check) - the original design doc
- TesseraStudio/README.md or similar - any project-level docs

## The two-audience tension (the core of the task)
Tessera Studio exposes a quantization/calibration/evolution pipeline (highly
technical: imatrix, AWQ, alpha search, GA, acceptance gates, L2 divergence) AND
an on-device chat agent (the Playground). The two audiences have OPPOSITE needs:
- NON-TECHNICAL users want: pick a model -> chat. Hide the pipeline. Sensible
  defaults. Plain-language status. No jargon. Apple-like polish.
- ML EXPERTS want: full visibility into every pipeline stage, numeric telemetry,
  per-tensor granularity, ability to tune every knob, exportable artifacts,
  reproducibility (seed tracking, lineage), comparison views (A/B, L2).
A SOTA UX serves both via progressive disclosure (calm default, depth on demand).
This is the central design problem to solve.

## Reference apps to study (web research encouraged)
- LM Studio, Ollama, Msty, GPT4All, Jan, Backyard AI - the direct competitors
- Apple's Human Interface Guidelines for ML/AI apps (2024-2026 era)
- Notion AI, Linear AI, Raycast AI - for the "agent that does things" UX patterns
- Cursor / Copilot / Zed - for the "expert surface + assistant" dual-audience pattern
- Vercel v0, ChatGPT, Claude - for conversation + artifact patterns

## Constraints
- DO NOT do heavy compute. The MoE quantize pipeline is running in the background
  (pid 89237) using most of the 16 GB RAM. No builds of the app, no model loads,
  no concurrent llama-bench. This is a READ + RESEARCH + DESIGN task, not a build task.
- DO NOT edit source. This is a study; produce a blueprint, not code changes.
- ASCII only in any output (repo rule).

## Baseline
- sha: 10222c950 (current main, with all the wave cherry-picks). Read the app
  source from this tree.

## Honest scope
This is a research + design task. The output is a single substantial design
document. Do not try to also implement changes - that is a separate future wave
that will use your blueprint as its spec.

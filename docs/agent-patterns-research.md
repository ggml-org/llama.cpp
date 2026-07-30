# Tessera Studio: Agent Pattern Research (Task 4)

**Date:** 2026-07-30
**Branch:** tessera/track-studio-app-scoping (off b93e218a3)
**Status:** Research only. No doc changes committed. Held for architect direction.

## Sources studied

| Source | LoC | Relevance |
|---|---|---|
| `/Users/user/Developer/GitHub/PrismAgent/` agent files | ~2,200 | Direct port candidate |
| `macOS26/Agent` (Agent!) at `/tmp/tessera-mac-agent-research/Agent` | 49,790 Swift | Most production reference |
| `LastByteLLC/junco` | 36,359 Swift | On-device Swift coding agent |
| `terryso/motive` | 39,504 Swift | Menu bar agent with 3 trust levels |
| `terryso/open-agent-sdk-swift` | 168,946 Swift | Reference SDK architecture (10 ADs) |
| `rudrankriyam/Foundation-Models-Framework-Example` (Foundation Lab) | 7,988 Swift (app) | **Direct parallel to Tessera Studio** |
| `Tom-Ryder/AgentRunKit` | 71,141 Swift | Type-safe tool framework |
| `singe/seldon` | 1,447 Swift | Smallest viable reference |
| `1amageek/SwiftAgent` | 47,518 Swift | Declarative agents |

## Six themes

1. **App shell:** Full window with focused toolbar (Foundation Lab 3-destination model), NOT Agent!'s 15-button header
2. **Agent loop:** 4 patterns (AsyncStream actor / micro-conversations / 86 flat tools / 34 tiered tools) — none fit Tessera Studio v1
3. **Safety:** 6 tiers (PrismAgent) > 3 trust levels (Motive) > 0 (workbenches) — for v1 with no agent loop, we need 0
4. **Plan mode:** Calibration is a linear pipeline, not a multi-step plan; show pipeline as `NavigationSplitView` detail
5. **On-device constraints:** Token-budget visualization, micro-conversations, Apple AI as mediator — adopt token-badge from Foundation Lab
6. **Provider abstraction:** TesseraRuntime enum (onDevice/mlx/privateCloud) mirroring `FoundationModelRuntime`

## 50 concrete patterns table

See full response in session transcript. Top picks:

| # | Pattern | Source | Tessera target | LoC |
|---|---|---|---|---|
| 1 | 3-destination shell | Foundation Lab | `StudioShell.swift` | ~300 |
| 2 | `@MainActor @Observable` VM | `Foundation Lab/ViewModels/ChatViewModel.swift:15-65` | `ChatViewModel.swift` | 200 |
| 5 | `availability` enum gating | `Foundation Lab/ViewModels/ChatViewModel.swift:70-82` | `CalibrationEngine.swift` | 13 |
| 13 | Tool protocol with Codable + JSON Schema | `PrismAgent/.../ToolTypes.swift`, `open-agent-sdk-swift/.../architecture.md:119-133` | `TesseraTool.swift` (Phase 2) | ~150 |
| 16 | Tool message UI: file content / directory / search / web / error | `PrismAgent/PrismAgent/ToolMessageView.swift:62-98` | `RichMessageView.swift` | 362 |
| 17 | Tool call banner with icon-by-prefix | `PrismAgent/PrismAgent/ToolMessageView.swift:103-138` | `TesseraToolBanner.swift` | 36 |
| 18 | Audit receipts (CUAAuditReceipt pattern) | `PrismAgent/PrismAgent/ComputerUseAdapter.swift:58-75` | `CalibrationReceipt.swift` | 18 |
| 24 | Sub-agent orchestrator (parallel workers) | `PrismAgent/PrismAgent/SubAgentOrchestrator.swift:1-100` | `QuantizationWorkerPool.swift` | 100 |
| 28 | LLM token-usage popover with cached_tokens | `Agent/README.md:40` | `StudioUsageView.swift` | n/a |
| 29 | Prompt versioning with `v{version}` and `READ ONLY` | `Agent/docs/TECHNICAL.md:347-363` | `StudioSystemPrompt.swift` | n/a |
| 31 | Fallback chain (provider A → B → C) | Agent! toolbar #10 | `FallbackChainView.swift` | n/a |
| 33 | `@Generable` for calibration configs | `Foundation-Models-Framework-Example/README.md:121-123` | `CalibrationConfig.swift` | n/a |
| 36 | `@`-file targeting (`quantize @Gemma-4-12B`) | junco `README.md:101-107` | `TesseraInputField.swift` | ~30 |
| 39 | SwiftData for chat history | `Agent/AgentViewModel.swift:39-40` | `StudioChatMessage.swift` | n/a |
| 48 | TesseraRuntime enum | `Foundation Lab/ViewModels/ChatViewModel.swift:37` | `TesseraRuntime.swift` | 3 |
| 50 | GUI app + companion CLI from same package | `Foundation Lab/README.md:166-180` | `tessera` CLI (Phase 2) | n/a |

## 20 patterns to SKIP entirely

| Pattern | Source | Why skip |
|---|---|---|
| 15-button toolbar w/ popovers | Agent! `HeaderSectionView.swift:53-150` | Visual noise; Foundation Lab's 3-destination is cleaner |
| XPC user agent + privileged daemon | Agent! `docs/TECHNICAL.md:296-300` | Studio is foreground single-user; no root ops |
| MCP server config UI | Agent! `docs/TECHNICAL.md:253-275` | Not in v1 scope |
| Computer use (CGEvent + Accessibility) | `PrismAgent/PrismAgent/ComputerUseAdapter.swift:1-504` | Out of scope; dangerous |
| AppleScript bridge | `PrismAgent/PrismAgent/ScriptBridge.swift:1-42` | Not needed |
| Accessibility scan | `PrismAgent/PrismAgent/AccessibilityEngine.swift:1-101` | Not needed |
| Action overlay (floating NSPanel) | `PrismAgent/PrismAgent/AgentActionOverlay.swift:1-249` | Foreground app |
| iMessage remote | `Agent/docs/TECHNICAL.md:187-247` | Not Tessera use case |
| Voice hotword | `Agent/README.md:208` | Not hands-busy workflow |
| Plan mode (PlanDocument / flat 86 tools) | PrismAgent, Agent! | Calibration is linear pipeline |
| TUI mode | seldon | GUI users first |
| Global hotkey | Motive | Windowed app; not needed |
| LoRA adapter training | junco | Calibration, not fine-tuning |
| Server mode (Linux) | Foundation Lab | macOS/iOS only for v1 |
| JSONL repo-map | Agent! | We don't edit code |
| `fmas` Python tooling | Foundation Lab `Tools/AdapterStudio/` | Swift-native; no Python sidecar |
| Chat history drawer (already adopted) | AWS sample | Already in design doc §5.6 |
| iOS BackgroundKeepAlive / Live Activity | sample-mobile-ai-assistant | macOS only |
| Sub-agent dispatcher for chat tool calls | PrismAgent `AgentToolDispatcher.swift:1-292` | Sub-agent pool yes; chat dispatcher no |
| Tessera-core LoRA | junco | Out of scope |

## Open question: Should Tessera Studio get an agent loop?

**Recommendation: SKIP for v1, EVALUATE for v2.**

| For | Against |
|---|---|
| Calibration is multi-step pipeline; agent could automate | 6-8 weeks of work |
| PrismAgent has building blocks (AgentLoop, ToolRegistry, ApprovalEngine) — reuse | Studio v1 value is the workbench, not autonomy |
| Agent! proves users want this | Agent! is a coding agent; use cases differ |
| Tessera C FFI already exposes token control | Reasoning C FFI was for chat, not for agent loop |

**If v1 includes it:** minimum viable is 5 tools (`list_models`, `calibrate`, `quantize`, `evaluate`, `export`) + `TesseraTool` protocol + `TesseraToolRegistry` + `TesseraAgentLoop` + `TesseraApprovalEngine` + `RichMessageView` = ~830 LoC new Swift + ~200 LoC C FFI = 2-week sprint.

**If v2:** ship the workbench + chat + reasoning + web search + rich renderers in v1; layer agent loop on top reusing PrismAgent's data structures in 3-4 weeks.

## Proposed doc changes (held for architect direction)

If greenlight to commit, on top of cc4415fb6:

| Section | Change | LoC |
|---|---|---|
| §3.2 | Add `tessera_tool_*` C API | +150 |
| NEW §5.11 | Workbench destinations (Library/Playground/Runs) | +200 |
| NEW §5.12 | Token budget visualization | +80 |
| NEW §5.13 | Rich message renderers | +250 |
| §6 | Add R28 agent-loop scope expansion | +30 |
| §11 | Add 11.19 TesseraTool protocol question | +20 |
| §13 | Note no new entitlements needed | +40 |
| **Total** | | **+770** |

## 1 callout

**AION (Apple Intelligence Mediator) pattern from Agent! README.md:24-30** — Apple AI observes the conversation and adds `[ AI]`-prefixed annotations, doesn't act as the LLM. Worth borrowing for Tessera Studio's chat: when user asks about a model, Apple AI can inject a `modelCard` annotation (free, on-device, no API cost). Pattern is small (~100 LoC chat-side annotation).

# UX implementation wave 1 - phase 1 of the blueprint

## The blueprint (READ FIRST)
/Users/user/Developer/GitHub/tessera/.zcode/alphaevolve/studio-ux-study/blueprint.md
This wave implements Phase 1 (section 8). Read the full blueprint; this preflight
only captures the must-know facts and the constraints.

## Phase 1 goal
A non-technical user can install Tessera Studio and chat within two minutes,
with no Settings visit. You are implementing THIS phase only - do not start
Phase 2+ work (ApprovalSheetV2, ABCompareView, etc.).

## Phase 1 scope (from blueprint section 8)
1. Wire a real default LLM provider. PlaceholderLLMProvider
   (TesseraStudio/Sources/TesseraCore/Agent/TesseraAgentLoop.swift:328) becomes
   a last-resort fallback; the factory defaults to the on-device path.
2. Replace OnboardingView with a six-step first-run flow ending in working chat.
   The "Download a Starter Model" button at line 86 becomes functional.
3. Add AudienceModeToggle (component 5.1) to the toolbar. Default mode: Simple.
4. Filter the sidebar by mode in ContentView.swift: Simple shows only Chat + Models.
5. Reskin LibraryView's ModelCardView (component 5.2) with a Chat button on the card.
6. Shrink TelemetryDrawer to Tier 1 (a single chip) in Simple mode.

## Files (read each before editing)
- TesseraStudio/Sources/TesseraCore/Views/OnboardingView.swift
- TesseraStudio/Sources/TesseraStudioMac/App/ContentView.swift
- TesseraStudio/Sources/TesseraCore/Views/LibraryView.swift
- TesseraStudio/Sources/TesseraCore/Views/TelemetryDrawer.swift
- TesseraStudio/Sources/TesseraCore/Engine/TesseraLLMProviderFactory.swift (or wherever the factory lives - grep for it)
- TesseraStudio/Sources/TesseraCore/Agent/TesseraAgentLoop.swift (PlaceholderLLMProvider at :328)

## New components to add (from blueprint component 5.1 / 5.2)
- AudienceModeToggle.swift - toolbar toggle with three states (Simple/Standard/Studio), persisted in UserDefaults
- ModelCardV2 (extension or replacement of ModelCardView) - adds a prominent Chat button

## Baseline
- sha: 10222c950 (main). Branch your worktree off THIS.
- The TesseraStudio package builds via `swift build` from the TesseraStudio/ dir,
  OR `xcodebuild` via the .xcodeproj. Try swift build first (faster, no Xcode GUI).

## CRITICAL resource constraint
- The MoE quantize pipeline (wave 6) is running and using most of the 16 GB RAM.
- DO NOT load any model, run any benchmark, or do anything inference-y. Pure
  Swift edits + a `swift build` (CPU only, no model). If swift build also
  competes too hard, slow down - correctness over speed.
- DO NOT run two swift build processes concurrently.
- NEVER run llama-server or llama-bench.

## Mechanics
- Single gene. Budget: 60 min OR Phase 1 shippable, whichever first.
- One worktree off 10222c950. ASCII only in code + comments (repo rule: no
  em-dash, no unicode arrows, use -, ->, x, ...).
- Commits on evolve/ux-impl-w1/* only. NEVER master/main. Never push, never gh.
- Never weaken a test or assertion to pass.

## Build verification
- swift build (from TesseraStudio/) must succeed.
- swift test (if it runs without a model) must pass.
- If swift build is impossible in the worktree for an environmental reason you
  can't fix, document the exact failure and ship the source changes with the
  build marked "unverified" - source correctness is still valuable.

## Output contract
- review branch evolve-review/ux-impl-w1 off 10222c950 with Phase 1 changes.
- Run artifacts: .zcode/alphaevolve/ux-impl-w1/{gene-ledger.json, changes.md, best.md, integration/patches/g1.patch}
- Final message: which of the 6 Phase 1 sub-items landed, build status (swift build pass/fail),
  new components added, files touched, what you skipped and why, bugs/quirks in the existing Swift.

Be honest about build status. A claimed build pass that doesn't reproduce is worse
than an honest "source done, build couldn't verify because X". Begin.

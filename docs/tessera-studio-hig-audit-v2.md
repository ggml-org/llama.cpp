# Tessera Studio - HIG audit v2 (2026-08-04)

Re-audit against the expanded `apple-hig` skill (152 KB revision: added
writing, app icons, motion, dark mode; components 13.14-13.25; patterns
14.9-14.14 incl. run-completion routing + notification suppression;
inputs 16.1-16.2; state-lifetime 4.11-4.13; composite surfaces 4.14).

Scope: main @ 738bed892 (post Tier 1 + Tier 2 + scene-lived store +
cross-destination ping). Audit method: code inspection (no visual pass);
file:line refs are `TesseraStudio/Sources/`-relative unless noted.

## Verdict

Tier 1 and Tier 2 work HOLDS under the expanded criteria. The new
state-lifetime, composite-surface, and run-routing sections all validate
the architecture that landed after Tier 2. The new findings concentrate
in the newly-covered sections: destructive-action confirmation (14.1),
Reduce Motion (2.7/3.6), writing (2.9), numeric input (10.15/13.16),
drag feedback (14.6), and window sizing (4.1).

**5 Tier 1 findings (2 data-loss, 1 accessibility, 1 correctness,
1 misleading control), 9 Tier 2, 8 Tier 3.**

## What passes under the new sections

- **Run-completion routing (14.10.x table)**: suppress only when
  frontmost AND Workflows surface visible; cancels never ping;
  outcome re-presents on return via stored `.finished` phase. Matches
  the table row-for-row. The key-window edge (window A finishes while
  user looks at window B) is the documented v1 deferral in 14.12.x.
  `interruptionLevel` unset = system default `.active` = what the
  table prescribes (make explicit; see T2-9).
- **State lifetime (4.11-4.13)**: WorkflowEditorStore is exactly the
  scene-owned pattern - four hydration sites, derived `document`,
  derived `isEdited`, presentation state view-local, undo closures
  capture the class. Matches the decision table for document model,
  selection, run phase, and undo.
- **Composite surface (4.14)**: WorkflowsView = NavigationSplitView +
  toolbar + inspector + run sheet; every component carries real user
  attention (no orphans); View menu owns the toggles. Right-sizing
  test passes.
- **Settings scene (10.12/14.8)**: `Settings { SettingsView() }`
  present, Cmd-, auto-bound, no manual menu item, no appearance
  toggle (correct per 2.8), Keychain for the secret (14.3).
- **Search (14.14)**: `.searchable` on the palette and Library with
  prompts.
- **Empty states (13.8)**: ContentUnavailableView in Library, Runs,
  palette-filter, parameter panel, chat history, and the analytics
  views.

## Tier 1 - close before ship

### T1-1 "Purge all learning data" fires with no confirmation
`SettingsView.swift:338-341`. Irreversible bulk delete executes
directly in the button action. HIG 14.1 ("warn about unexpected,
irreversible data loss"), 13.5 (destructive alerts), principle 2
("help people recover from mistakes"). There is currently NO
`confirmationDialog` in the entire codebase. Fix: `.confirmationDialog`
or `.alert` with the destructive role on "Purge" and a Cancel button;
destructive must not be the default button. "Reset all grants"
(:334) also fires immediately - recoverable, so lower urgency, but
give it the same treatment for consistency.

### T1-2 Chat conversation delete fires with no confirmation
`ChatHistoryDrawer.swift:123,142` - `Button("Delete", role:
.destructive) { delete(convo) }` runs directly. Same HIG items as
T1-1. User's conversation history is user data.

### T1-3 Reduce Motion is not respected anywhere
Zero `accessibilityReduceMotion` reads in the app. HIG 2.7/3.6/12.4:
when Reduce Motion is on, replace transitions with fades or instant
changes. Affected: connection-error banner `.transition(.move)` +
`.animation` (WorkflowsView.swift:442-447), onboarding page turns
(`withAnimation` OnboardingView.swift:148,157), sheet presentation,
selection styling. Fix: read the environment flag at the animation
sites; `withAnimation(reduceMotion ? nil : ...)` and drop the banner
transition under Reduce Motion.

### T1-4 Numeric node parameters are stored as JSON strings
`WorkflowParameterPanelView.swift:98-103,129-140` - integer and
number fields both bind through `bindingForString`, so a node's
`{"samples": 100}` parameter round-trips to the executor as
`{"samples": "100"}`. This is a data-correctness bug wearing a UI
hat (tools may reject or mis-parse string numbers), and it violates
the numeric-input decision tree (10.15): bounded integers get
Stepper + TextField; large-range / decimal gets a TextField with a
number format. Fix: `bindingForNumber` that writes `.number` /
`.integer` JSONValues, parse-on-display, and Stepper pairing where
the schema gives min/max.

### T1-5 File > Save As... does the same thing as Save
`TesseraStudioMacApp.swift` command block: SaveAsWorkflowMenuItem
reuses the save code path ("same code path today" per the comment).
A menu item that lies about its behavior is worse than an absent one
(14.9: save panel provides name + location chooser). Fix: implement
Save As (re-present fileExporter with the current name prefilled,
then markSaved at the new URL) or remove the menu item until it is
real.

## Tier 2 - polish before public beta

- **T2-1 Window min/max sizes missing (4.1)**. Only `.defaultSize`
  is set; the window can be shrunk until the nested split views
  collapse. Add `.frame(minWidth:minHeight:)` on ContentView (e.g.
  900x560) - onboarding already does this (560x460).
- **T2-2 New/Open over an edited document has no guard (14.7, 13.5)**.
  Old Tier 3 item, now actionable: `isEdited` is derived on the
  store, so New/Open can present a Save/Discard/Cancel alert before
  hydrating. Destructive replacement of unsaved work.
- **T2-3 Onboarding gaps (14.2, 2.9)**. No Skip button ("make the
  tutorial optional"); "Set Up Your Models" uses a possessive
  (2.9: "Set Up Models"); the model-directory field lacks the
  Browse... picker that Tier 2 added everywhere else; "Download a
  Starter Model" is permanently disabled - implement or remove
  (dead controls erode trust).
- **T2-4 "YOLO" jargon in the Permissions tab (2.9)**. "Start YOLO",
  "End YOLO", "Last YOLO ran..." - plain-language rule ("write for
  everyone... no jargon"). Candidate: "Autonomous session" /
  "Start autonomous session". Architect call - YOLO is recognized
  agent-culture slang, but the tab was literally renamed away from
  "Autonomy" for clarity.
- **T2-5 Playground Cancel has role: .destructive
  (13.1 button roles)**. `PlaygroundView.swift:89` - cancelling a
  run is not data destruction; `.destructive` renders it red and
  mis-signals. Use `.cancel` (also gets Esc behavior).
- **T2-6 Drag-to-wire feedback is post-hoc (14.6)**. During the wire
  drag, valid input ports don't highlight and invalid targets don't
  show `circle.slash`; validation only lands as a banner after the
  drop. Add: highlight compatible ports while `pendingConnection` is
  live (port-type match is already known), red ring / slash on
  incompatible ones. Also `currentCanvasSize()` returns a hard-coded
  2000x2000 (WorkflowCanvasView.swift:120-125) - thread the real
  GeometryReader size through.
- **T2-7 Tooltip coverage + wording (14.11)**. Only the four file/run
  toolbar buttons have `.help()`. Missing: New Node button, View-menu
  toggles (palette/inspector/telemetry), telemetry drawer handle,
  canvas affordances (drag-to-wire hint). Wording: 14.11 says begin
  with a verb and don't repeat the control name - "New workflow" on
  the New-Workflow button repeats the name; prefer "Create an empty
  workflow". 60-75 char budget.
- **T2-8 Palette drag-onto-canvas absent (14.6/10.13)**. Adding a
  node requires the toolbar button; HIG "support drag and drop
  everywhere you can" - palette rows are the natural drag source
  (`.draggable` node-type payload + `.dropDestination` on the
  canvas). The §14.11 TipKit first-use tip pairs with this.
- **T2-9 Make the notification interruption level explicit
  (14.12)**. `content.interruptionLevel = .active` in the notifier -
  currently the default, but explicit is greppable and documents the
  routing decision next to the suppression logic.

## Tier 3 - backlog (revalidated against new sections)

- **T3-1 Pointer styles (4.10, 16.2)**: zero `.pointerStyle` in the
  app. Grab cursor over nodes, resize over the HSplitView divider,
  link cursor on docs links.
- **T3-2 Keyboard alternative for canvas moves (16.2)**: custom
  gestures need a non-gesture alternative - arrow-key nudge on the
  selected node (also makes node moving Full-Keyboard-Access
  reachable).
- **T3-3 Esc handling (12.8)**: run sheet Close + connection-error
  Dismiss should bind `.keyboardShortcut(.cancelAction)`.
- **T3-4 Grid snap for node drags** (carried from Tier 3 backlog).
- **T3-5 Canvas zoom/pan (13.20)**: nodes can be dragged out of the
  visible canvas and lost; ScrollView + magnification, zoom-to-fit.
- **T3-6 Gauges in the run sheet (13.25)**: ANE/GPU/memory
  utilization when telemetry exposes them.
- **T3-7 Layered app icon (2.10)**: Icon Composer, background +
  foreground layers, default/dark/tinted variants, recognizable at
  60x60, no text.
- **T3-8 Status bar (4.1 bottom bar)**: node/edge counts + selection
  in a `.safeAreaInset(edge: .bottom)` bar - small, status-only.

## Suggested commit slices (§12.9 tagging pattern)

1. `Tessera Studio: confirm destructive actions (HIG T1-1, T1-2, T2-2)`
   - the confirmation-dialog slice: purge, conversation delete,
   New/Open-while-edited. Smallest, highest value.
2. `Tessera Studio: respect Reduce Motion (HIG T1-3)` - one env flag,
   four animation sites.
3. `Tessera Studio: numeric parameter fields round-trip as numbers
   (HIG T1-4)` - binding + Stepper pairing.
4. `Tessera Studio: real Save As or none (HIG T1-5)` - architect
   picks.
5. `Tessera Studio: window floor, onboarding polish, writing pass
   (HIG T2-1, T2-3, T2-4, T2-5)` - the cheap bundle.
6. `Tessera Studio: wire-drag feedback + palette drag-to-add
   (HIG T2-6, T2-8)` - the canvas-feel slice, pairs with T3-1/T3-4.
7. `Tessera Studio: tooltip sweep (HIG T2-7)` - mechanical.

## Carry-over status from audit v1

All 7 v1 Tier 1 items and all 14 v1 Tier 2 items remain closed
(re-verified: UTI, Edit/Help menus, a11y labels, contrast, run
lifecycle, notification, Edited marker, multi-window, restoration,
Keychain, NavigationSplitView, .toolbar, View menu, .searchable,
file pickers, Permissions rename). The v1 Tier 3 backlog items that
survive into this audit: pointer styles (T3-1), grid snap (T3-4),
status bar (T3-8), layered icon (T3-7), canvas zoom (T3-5, was
"canvas-size"), history drawer as real column (not re-flagged - the
drawer works, low value), .fill symbol variants and @FocusState
(unchanged, minor).

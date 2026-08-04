# Tessera Studio - Scoping Design

Design only. No implementation code. Scopes a new native iOS + Mac
app that uses the Tessera llama.cpp fork at its core, ports the good
architectural patterns from PrismAgent, and showcases the nature of
the Tessera project: per-tensor policy, schema-versioned evidence,
runtime agreement, full CoreML inference on iPhone, multi-modal
calibration.

> Roadmap alignment: the runtime-aware proxy-objective research
> (2026-07-30) validates the `evaluate` tool, the fitness-over-generations
> chart, and the A/B compare view as the user-facing surface for the
> kernel-fidelity loop (runtime-aware-pipeline L4/L6). Two refinements:
> the fitness chart should plot the alpha-weighted composite and the
> regime-indexed archive occupancy (not a single fitness line), and the
> A/B compare view is the natural place to surface the G6
> composite-beats-single-proxy result as a receipt. See
> [`research-alignment-2026-07-30.md`](research-alignment-2026-07-30.md)
> Section 4.6.

The 32 architect decisions are locked by the prior conversation and
the three prior scoping docs. This doc references them and does not
re-litigate them.

**Architect directives applied in this revision (2026-07-30):**
web search is in scope (provider-shaped, keyless DuckDuckGo default, see section 5.4);
.mlmodelc distribution is via App Store IAP with Apple-hosted
background assets for large files (see section 5.10.4 and 5.10.5);
Gemma 4 12B unified is a reasoning model and the engine + UI must
support chain-of-thought from day 1 (see section 5.7);
rich renderers (markdown, code highlighting, Mermaid, HTML preview)
are in scope (see section 5.8); the chat history drawer pattern is
in scope and replaces the iOS tab bar (see section 5.9); a full
Settings surface is in scope for App Store publishability
(see section 5.10).

- CoreML design: `docs/tessera-coreml-conversion-design.md`
  (decisions C1-C10)
- Multi-modal calibration: `docs/multimodal-calibration-design.md`
  (decisions M1-M8)
- C++ port: `docs/c++-port-design.md` (G0-G6 phasing,
  Tessera-as-default of llama-quantize, `--tessera-mode=off` opt-out)
- L1-L6 telemetry pipeline: `docs/runtime-aware-pipeline.md`
- L1.5 reference sidecar: `docs/c++-port-design.md` section 5
- Tessera GGUF metadata spec: `docs/tessera.md`

Architect surface, one-line summary of the locked state:

- Full CoreML inference is a first-class backend peer to ggml-metal
  (C1). iPhone is the primary surface (decision 2). Hero metric is
  sustained battery draw per token on a phone (decision 2). IOReport
  is the runtime telemetry source (C3, C4). The conversion tool is
  stateless, the runtime is ggml-coreml (decision 4, G7). 1
  .mlmodelc with runtime act_scale for v1, 3 as v2 packaging
  optimization (C8). Per-modality AWQ alpha with text fallback
  (M8, M2 hard error on missing modality_scales). The C++ port is
  in flight, G0-G6 phasing (c++-port-design.md #1, #4). Multi-modal
  calibration with 5k curated set, weighted 1.5k/2k/1.5k (M1-M8).
  Tessera as the new default of llama-quantize; no subcommand;
  --tessera-mode=off only opt-out (c++-port-design.md #1, #4).
  Match llama.cpp production code style; no new abstractions.

---

## See also — prior art (pattern reference only)

The node-graph workflow system described in section 16 is a
from-scratch reimplementation of established patterns, not a
derivative of any specific project. The most relevant reference
project is **ComfyUI** (Stable Diffusion community, GPL-3.0):

- **Patterns reimplemented**: node-graph editor, typed input /
  output ports, per-node parameters, JSON workflow persistence,
  palette-driven node discovery, real-time progress streaming.
- **Patterns deliberately not ported**: server-client architecture
  (Tessera Studio is a native in-process app), web frontend
  (Tessera Studio is SwiftUI on macOS), Python runtime (Tessera
  Studio is Swift + the existing C++ engine), any specific
  ComfyUI JSON schema or Python API.
- **Plugin model not adopted**: Tessera Studio is a single-shipped
  Mac app. There is no third-party node-pack system, no manifest
  format, no plugin discovery under
  `~/Library/Application Support/`. Adding nodes is a code change
  in `TesseraCore`. This keeps the GPL/POLYFORM analysis simple
  (we are reimplementing the editor UX; there are no third-party
  derived works to license) and avoids the third-party legal tail.
- **License note**: ComfyUI is GPL-3.0; Tessera Studio is
  PolyForm Noncommercial 1.0.0. These two licenses are
  mutually exclusive (commercial-use asymmetry, not viral
  copyleft), so no ComfyUI code may be imported. Patterns are
  not copyrightable; specific code is. The ComfyUI project is
  cited here as a pattern reference for the workflow section
  only.

Other reference projects whose patterns are visible in Tessera
Studio (also reimplemented, not derived):

- **Draw Things** (Apache-2.0) — local CoreML inference UI.
- **LM Studio** (Apache-2.0 for the core) — model discovery
  and download.
- **Pocket-TTS** style local-first agent UIs.

All Tessera Studio source is from-scratch. No third-party
UI / agent framework is imported.

## 1. Product surface

### 1.1 The 5-minute iPhone demo

A new user opens Tessera Studio on a M-series iPhone and sees the
full Tessera story in five minutes. The flow is scripted, not free
exploration: every step lands a different "this is what Tessera is"
beat.

| Minute | Beat | What the user sees |
| --- | --- | --- |
| 0:00-0:30 | Cold open | The "What is Tessera?" splash. One sentence, one diagram, the 9-component cluster next to a 7-component stock K-quant. The A/B is already implied. |
| 0:30-1:00 | Model picker | Three models are pre-bundled. Stock Q4_0 Gemma 3 4B. Tessera-quantized Gemma 3 4B (4-bit effective, T640). Tessera-quantized Gemma 4 12B Unified (text + image + audio, 3.5-bit effective, T640, **reasoning**). Badges: ANE badge on the Tessera rows, GPU badge on the stock row, REASONING badge on the 12B. |
| 1:00-1:30 | First inference | User picks the Tessera 12B, types "In one sentence, what is the capital of France?" The iPhone warms. First token in 240ms. ANE power bar starts moving. |
| 1:30-2:00 | Telemetry transparency | User taps the title. A sub-row appears: `tok/s = 24.1`, `ANE mW = 1.2k`, `battery Δ = 12 mAh`. The five IOReport gauges are available via a `...` menu. The "why CoreML" answer is on the screen. |
| 2:00-2:45 | Reasoning reveal | User asks "Why is the sky blue? Show your reasoning." The 12B emits a `Thinking for 4.2s` block (collapsible, monospaced, sparkle icon). When complete, the user expands it to see the chain-of-thought. The reasoning is preserved in the L1 sidecar (per-token) and the v3 export. |
| 2:45-3:15 | Web search | User asks "What is the capital of France today?" (Tavily-backed web search enabled). A `web_search` chip appears above the answer; the model streams its synthesis, citation links visible inline. |
| 3:15-3:45 | Modality switch | User pastes a photo from the camera roll, asks "What is in this image?" The .mlmodelc is the same one (C8). The act_scale switches to `IMAGE` at the engine call. The first image token lands in 340ms; the per-token stream follows. |
| 3:45-4:15 | Audio mode | User records a 4-second voice memo, asks for a transcript. The same .mlmodelc. act_scale switches to `AUDIO`. The transcript streams; latency is up (audio tokenisation is heavier) but ANE power is steady. |
| 4:15-4:30 | Drawer reveal | User swipes from the left edge. The chat history drawer opens: Today / Yesterday / 2 days ago / ... / This month bucketing, breathing-dot on the active conversation, the new conversation button. |
| 4:30-5:00 | The A/B moment | User taps "Compare" on the chat header. Side-by-side: Tessera 4-bit 12B vs stock Q4_0 12B. Same prompt, same image. Per-token latency table. The 12B Tessera row is faster (4 effective bits < Q4_0's 4.5) and uses less ANE power per token. PPL proxy on a 50-token held-out set is reported (e.g. Tessera 5.42, stock 5.39). The "PPL cost of going 4-bit" is on the screen. |

The script works because the iPhone app boots into the chat surface
with the Tessera 12B already loaded; the user never waits on
downloads, never configures a quantizer, never sees a CLI. The
conversion (C2) and the calibration (multi-modal-calibration-design
section 4) already happened on the Mac companion and the result is
in the `.app` bundle. The pre-bundled .mlmodelcs are listed in
section 5.10.4 (the In-App Purchase / Apple-hosted background asset
distribution); the user can buy additional .mlmodelcs from the App
Store without leaving the app.

### 1.2 The 30-minute flight test

The hero metric is sustained battery draw per token on a phone
(decision 2). The 30-minute flight test is the canonical
demonstration of that metric. The iPhone app runs a continuous
chat workload for 30 minutes on battery (no charger), records:

- `battery_delta_mWh` (the total consumed, from IOReport)
- `tokens_generated` (the denominator)
- `mWh_per_token` (the headline)
- ANE power, GPU power, DRAM power, battery current as
  time-series (sampled at 1 Hz)
- thermal events (number of `kIOReturnTemperatureCritical` /
  throttling transitions)
- L1 + L1.5 sidecar emitted per token (Tessera-quality evidence
  is part of the headline, not separate)

The flight test is live in the iPhone app: a "Flight Test" button
in the chat header starts a 30-minute timer, runs an automated
prompt stream (text + image + audio round-robin), and renders the
results in a summary card at the end. The summary is exportable as
a JSON file (the v3 sidecar extension, see
`docs/tessera-coreml-conversion-design.md` section 6.5) for later
analysis on the Mac.

### 1.3 The 3 primary screens + chat history drawer

The app has three top-level destinations, surfaced as a chat
history drawer + content area on iOS (NavigationSplitView, section
5.9) and a window menu on Mac:

1. **Chat** (iOS primary, Mac also). The LlamaState-style chat
   surface (`examples/llama.swiftui/llama.swiftui/Models/LlamaState
   .swift` is the pattern we port). Modality picker, engine
   selector, live IOReport, the 30-minute flight test, the
   reasoning toggle, the web search button, the rich markdown
   renderers. The Mac Studio's `Run` tab is the same surface
   scaled up.
2. **A/B Compare** (iOS, Mac). Side-by-side: Tessera-quantized vs
   stock Q4_0. Same model, same prompt, different engines. The
   comparison table.
3. **Studio** (Mac only). The calibration + evolution surface.
   Pick a model, pick a calibration corpus, run AWQ-evolve,
   quantize, convert to `.mlmodelc`, ship to iPhone via iCloud
   Drive.

The **chat history drawer** is on the left side of the iOS app
(NavigationSplitView, sections 5.9 and 5.10). It is always
visible on iPad (sidebar) and is a swipe-to-reveal drawer on
iPhone. The Mac app uses a similar drawer pattern via
NavigationSplitView.

### 1.4 The secondary screens

Each primary screen has a detail / drawer that exposes a secondary
view. None of these are top-level destinations; they are reached
via push / sheet from the primary.

- **ModelStore** (drawer / sheet, both). Lists bundled models +
  any user-imported GGUFs / .mlmodelcs in the Documents directory
  + the App Store IAP catalog (section 5.10.4). Badges: ANE
  (Tessera + CoreML), GPU (stock + Metal), CPU (rare, for
  debugging), REASONING (Gemma 4 12B Unified).
- **Telemetry** (drawer, iOS). The live IOReport + the 30-minute
  flight test results. The summary card is the conclusion of
  the flight test (mWh/token headline, thermal events, per-modality
  breakdown).
- **QuantizationPlan** (sheet, iOS; full view, Mac). The
  "calibrate -> evolve -> quantize -> convert -> run" pipeline
  visualised as five steps. iOS shows the plan the user shipped
  from the Mac; Mac shows the editor.
- **CalibrationSession** (sheet, iOS; full view, Mac). The
  imatrix + policy + provenance for a single calibration pass.
  Mac shows the runner with weight visualisation; iOS shows the
  read-only viewer.
- **Settings** (push from drawer / menu, both). The full
  settings surface (section 5.10). Privacy, telemetry, IAP
  catalog, model management, background behaviour, theme,
  about. Required for App Store publishability (section 13).

### 1.5 The "wow" moments

Three moments are designed to be screenshot-worthy:

- **First-token latency**: Metal vs CoreML, on a 12B, on a phone.
  The gap is ~3x on prefill (ANE > Metal) and ~2x on decode (ANE
  is competitive on memory-bound). Show both numbers, same prompt.
- **Sustained battery**: 30-minute flight test, mWh/token
  headline. The gap between Tessera 4-bit and stock Q4_0 is
  ~30-40% on the 12B; the headline lands that.
- **Telemetry transparency**: IOReport visible to the user. Other
  on-device inference apps hide the metric. We do not.

---

## 2. Architecture: Swift Package layout

### 2.1 The high-level shape

One Swift Package, three targets. The shared target is the engine
+ the adapter + the data models. The two platform targets are
iOS (chat + A/B) and Mac (Studio + calibration). No shared UI
code: iOS UI and Mac UI are different work, different idioms.

```
TesseraStudio/
  Package.swift
  Sources/
    TesseraCore/           (shared library, both platforms)
      LibTessera.swift          ~400 LoC
      InferenceAdapter.swift    ~200 LoC
      ModelStore.swift          ~200 LoC
      ConversationStore.swift   ~200 LoC
      TesseraChatHistory.swift  ~200 LoC (new in 2026-07-30 rev)
      CalibrationSession.swift  ~200 LoC
      QuantizationPlan.swift    ~150 LoC
      TelemetryObserver.swift   ~200 LoC
      TesseraWebSearch.swift    ~200 LoC (new in 2026-07-30 rev)
      TesseraRichRenderer.swift ~300 LoC (new in 2026-07-30 rev)
      TesseraSettings.swift     ~150 LoC (new in 2026-07-30 rev)
      TesseraStoreKit.swift     ~250 LoC (new in 2026-07-30 rev)
      TesseraIOReporter.swift   ~120 LoC
      (total ~2,570 LoC)
    TesseraStudioiOS/      (iOS app)
      TesseraStudioiOSApp.swift    ~50 LoC
      ContentView.swift            ~150 LoC (NavigationSplitView root)
      ChatView.swift               ~520 LoC (Chat + reasoning + web search chips)
      ABCompareView.swift          ~280 LoC
      ChatHistoryDrawer.swift      ~180 LoC (new in 2026-07-30 rev)
      ModelStoreDrawer.swift       ~150 LoC
      TelemetryDrawer.swift        ~120 LoC
      SettingsView.swift           ~200 LoC (new in 2026-07-30 rev)
      OnboardingView.swift         ~120 LoC (new in 2026-07-30 rev)
      ThinkingBlock.swift          ~80 LoC  (new in 2026-07-30 rev)
      MarkdownView.swift           ~250 LoC (new in 2026-07-30 rev)
      MermaidView.swift            ~250 LoC (new in 2026-07-30 rev)
      (total ~2,350 LoC)
    TesseraStudioMac/      (Mac app)
      TesseraStudioMacApp.swift    ~50 LoC
      ContentView.swift            ~180 LoC (NavigationSplitView root)
      StudioView.swift             ~400 LoC
      QuantizationPlanEditor.swift ~280 LoC
      CalibrationSessionView.swift ~320 LoC
      SidecarViewer.swift          ~250 LoC
      ABReplayView.swift           ~250 LoC
      ChatHistoryDrawer.swift      ~180 LoC
      SettingsView.swift           ~200 LoC
      RunTabView.swift             ~480 LoC (the Mac chat surface)
      (total ~2,590 LoC)
  Frameworks/
    tessera.xcframework    (the C FFI, ~3,500 LoC of C/C++)
  Resources/
    models/                (pre-bundled GGUFs and .mlmodelcs)
  Tests/
    TesseraCoreTests/
    TesseraStudioiOSTests/
    TesseraStudioMacTests/
```

Total: ~7,500 LoC of Swift across the three targets, plus the
~3,500 LoC C/C++ engine that we do not write here (it is the
output of the G0-G7 workstreams in the C++ port design). The
+2,700 LoC delta from the previous revision is web search,
reasoning, rich renderers, chat history drawer, settings,
IAP / StoreKit, onboarding, and AI disclosure.

### 2.2 Package.swift

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "TesseraStudio",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "TesseraCore", targets: ["TesseraCore"]),
        .executable(name: "TesseraStudioiOS", targets: ["TesseraStudioiOS"]),
        .executable(name: "TesseraStudioMac", targets: ["TesseraStudioMac"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "TesseraCore",
            path: "Sources/TesseraCore",
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("Metal"),
                .linkedLibrary("tessera"),
                .unsafeFlags(["-L", "Frameworks/tessera.xcframework"]),
            ]
        ),
        .executableTarget(
            name: "TesseraStudioiOS",
            dependencies: ["TesseraCore"],
            path: "Sources/TesseraStudioiOS"
        ),
        .executableTarget(
            name: "TesseraStudioMac",
            dependencies: ["TesseraCore"],
            path: "Sources/TesseraStudioMac"
        ),
        .testTarget(
            name: "TesseraCoreTests",
            dependencies: ["TesseraCore"],
            path: "Tests/TesseraCoreTests"
        ),
    ]
)
```

The iOS app target embeds `tessera.xcframework` via the Xcode
project, not the SwiftPM target. The Package.swift surface is
Mac-friendly (SwiftPM build for command-line tests) and the
xcodebuild surface handles the iOS embed-and-sign. This mirrors
the pattern in `examples/llama.swiftui/llama.swiftui.xcodeproj/
project.pbxproj:24-26` (the llama.xcframework is in `Frameworks,`
`Libraries, and Embedded Content`).

### 2.3 Target dependencies

| Target | Depends on | Why |
| --- | --- | --- |
| TesseraCore | CoreML, Metal, tessera xcframework | The engine + the adapter. The xcframework is the C FFI (G0-G7). CoreML is the Swift-side CoreML backend (it is the .mlmodelc loader + MLState, not the C backend). |
| TesseraStudioiOS | TesseraCore | The iOS chat + A/B surface. |
| TesseraStudioMac | TesseraCore | The Mac Studio surface. Imports the C-side calibration runner via tessera xcframework. |
| TesseraCoreTests | TesseraCore | Unit tests for the adapter, the model store, the calibration session persistence. |

### 2.4 File-level LoC estimates

| File | LoC | Status |
| --- | --- | --- |
| `Package.swift` | 50 | Design. |
| `Sources/TesseraCore/LibTessera.swift` | 400 | Design. The C FFI wrapper. Replaces `examples/llama.swiftui/llama.cpp.swift/LibLlama.swift` (337 LoC) and adds the Tessera C functions. |
| `Sources/TesseraCore/InferenceAdapter.swift` | 200 | Design. The protocol ported from `PrismAgent/PrismAgentiOS/PrismAgentiOS/InferenceAdapter.swift:1-25`. |
| `Sources/TesseraCore/ModelStore.swift` | 200 | Design. The Tessera-aware model store. Replaces `PrismAgent/PrismAgent/ModelStore.swift:1-229` minus the Prism-specific `authorized` check. |
| `Sources/TesseraCore/ConversationStore.swift` | 200 | Design. Tiered storage. Replaces `PrismAgent/PrismAgent/ConversationStore.swift:1-172` with a Tessera-flavoured StoredMessage (no image cache; modality is part of the message). |
| `Sources/TesseraCore/CalibrationSession.swift` | 200 | Design. The imatrix + policy + provenance record. New, not a port. |
| `Sources/TesseraCore/QuantizationPlan.swift` | 150 | Design. The 5-step pipeline. Replaces the PrismAgent "Plan" idea with a Tessera-flavoured one. |
| `Sources/TesseraCore/TelemetryObserver.swift` | 200 | Design. The L1 + L1.5 + B + C + E consumer. New, not a port. |
| `Sources/TesseraStudioiOS/TesseraStudioiOSApp.swift` | 50 | Design. |
| `Sources/TesseraStudioiOS/ContentView.swift` | 120 | Design. Tab bar. |
| `Sources/TesseraStudioiOS/ChatView.swift` | 480 | Design. Modality picker, engine selector, IOReport, flight test. |
| `Sources/TesseraStudioiOS/ABCompareView.swift` | 280 | Design. |
| `Sources/TesseraStudioiOS/ModelStoreDrawer.swift` | 150 | Design. |
| `Sources/TesseraStudioiOS/TelemetryDrawer.swift` | 120 | Design. |
| `Sources/TesseraStudioMac/TesseraStudioMacApp.swift` | 50 | Design. |
| `Sources/TesseraStudioMac/ContentView.swift` | 150 | Design. Window menu. |
| `Sources/TesseraStudioMac/StudioView.swift` | 400 | Design. |
| `Sources/TesseraStudioMac/QuantizationPlanEditor.swift` | 280 | Design. |
| `Sources/TesseraStudioMac/CalibrationSessionView.swift` | 320 | Design. |
| `Sources/TesseraStudioMac/SidecarViewer.swift` | 250 | Design. |
| `Sources/TesseraStudioMac/ABReplayView.swift` | 250 | Design. |
| Tests + xcconfig | ~300 | Design. |
| **Swift total** | **~4,800** | |
| C/C++ xcframework | ~3,500 | Out of scope for this doc. Output of G0-G7. |

The Swift totals are a target, not a budget. The LlamaState + the
PrismAgent surfaces show a real chat surface fits in ~480 LoC
(`PrismAgent/PrismChatView.swift:1-479` is 479 LoC and it does
more: voice, image drop, multimodal streaming). The Tessera
chat surface is simpler (no agent loop, no tool calls, no
embedded browser) and fits the same envelope.

### 2.5 The xcframework

The build script is a fork of
`/Users/user/Developer/GitHub/llama.cpp/build-xcframework.sh:1-550`
(550 LoC, builds for iOS sim, iOS device, macOS arm64, macOS x86_64,
visionOS, visionOS sim, tvOS sim, tvOS device). The Tessera
fork is ~600 LoC because it adds:

- `ggml-coreml` as a backend (`docs/tessera-coreml-conversion-
  design.md` section 5, G7)
- The Tessera common lib (`tools/quantize/tessera/`, the C++ port
  output of G3-G5)
- A subset of `tools/mtmd/` (the multimodal glue) for the iOS
  app's image / audio tokenisation
- The new `llama.h` with the Tessera additions (see section 3
  below for the C FFI)

The Mac app links against the Mac slice; the iOS app embeds the
iOS device + iOS sim slices. The fork is stored as
`build-tessera-xcframework.sh` in the Tessera llama.cpp fork and
called by the Tessera Studio Xcode project as a pre-build step.

---

## 3. The Tessera engine integration

### 3.1 How the llama.cpp fork becomes a Swift module

The Tessera llama.cpp fork produces a `tessera.xcframework` with
the same shape as `llama.xcframework` (see
`/Users/user/Developer/GitHub/llama.cpp/build-xcframework.sh:533-
550` for the xcodebuild `xcodebuild -create-xcframework` call).
The framework exposes the C API in a module map
(`build-xcframework.sh:135-146` shows the llama.xcframework
module map; the Tessera one is identical with `llama` -> `tessera`
and one extra header `tessera.h`).

The Swift wrapper `Sources/TesseraCore/LibTessera.swift` (~400
LoC) is a port of
`/Users/user/Developer/GitHub/llama.cpp/examples/llama.swiftui/
llama.cpp.swift/LibLlama.swift:1-337` (337 LoC). The port adds
~60 LoC for the new Tessera C functions. The pattern is the
same: an `actor TesseraContext` holds the `OpaquePointer` for
`model`, `context`, `vocab`, `sampling`, and a `batch`; the
`create_context` static factory loads the model.

### 3.2 The new TesseraEngine class

`TesseraEngine` (in `LibTessera.swift`) wraps the llama.cpp C API
and adds the Tessera-specific surface. It is the public entry
point for both `InferenceAdapter` implementations and the
`CalibrationSession` consumer.

```swift
@MainActor
public final class TesseraEngine {
    public static let shared: TesseraEngine = .init()

    // The loaded model + context
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var sampling: UnsafeMutablePointer<llama_sampler>?

    // The Tessera C-side handles
    private var tesseraContext: OpaquePointer?
    private var l1SidecarWriter: OpaquePointer?
    private var l15SidecarWriter: OpaquePointer?

    // The IOReport client (Objective-C bridge, see 6.5)
    private var ioReport: TesseraIOReporter?

    public func load(
        ggufPath: String,
        tesseraSidecar: URL?,
        policy: TesseraPolicy?
    ) async throws

    public func generate(
        prompt: String,
        modality: TesseraModality,
        config: TesseraGenerateConfig
    ) -> AsyncThrowingStream<TesseraToken, Error>

    public func readL1Sidecar(for tensor: String) throws -> TesseraL1Sidecar
    public func readL15Sidecar(for tensor: String) throws -> TesseraL15Sidecar

    public func latestIOReport() throws -> TesseraIOReport
    public func flightTestStatus() throws -> TesseraFlightTestStatus
}
```

The `TesseraToken` is the streaming element. It carries:

```swift
public struct TesseraToken: Sendable {
    public let text: String
    public let tokenID: Int32
    public let modality: TesseraModality
    public let latencyNs: UInt64
    public let anePowerMW: Double
    public let gpuPowerMW: Double
    public let dramPowerMW: Double
    public let batteryCurrentMA: Double
    public let thermalState: Int32
    public let l1SidecarDelta: URL?  // written by C side per token
    public let isReasoningToken: Bool  // section 5.5: true for CoT
    public let reasoningDurationSeconds: Double?  // set on EOG
}
```

`TesseraModality` is the C FFI enum (see 3.3 below).

### 3.3 The C FFI additions

The C FFI is THIN. Most of the work is in C++; the Swift surface
is a wrapper. The new C functions and their signatures:

```c
// libtessera.h (added to llama.h, not a new header)

typedef enum {
    TESSERA_MODALITY_TEXT  = 0,
    TESSERA_MODALITY_IMAGE = 1,
    TESSERA_MODALITY_AUDIO = 2,
} tessera_modality_t;

typedef enum {
    TESSERA_DEQUANT_T640_3D = 0,
    TESSERA_DEQUANT_T640_4D = 1,
    TESSERA_DEQUANT_STOCK_KQUANT = 2,
} tessera_dequant_mode_t;

// Opaque handle; created by tessera_context_init, freed by tessera_context_free
typedef struct tessera_context tessera_context_t;

tessera_context_t * tessera_context_init(
    struct llama_context * lctx,
    const char * policy_json_path  // optional; NULL falls back to GGUF metadata
);

void tessera_context_free(tessera_context_t * tctx);

int tessera_set_dequant_mode(
    tessera_context_t * tctx,
    tessera_dequant_mode_t mode
);

int tessera_set_modality(
    tessera_context_t * tctx,
    tessera_modality_t modality
);

int tessera_set_imatrix(
    tessera_context_t * tctx,
    const char * imatrix_path  // imatrix v2, modality-tagged (M3)
);

int tessera_set_policy(
    tessera_context_t * tctx,
    const char * policy_json_path  // calibration-policy.v1 + modality_scales (M2)
);

// L1 sidecar: per-tensor dequant error (the kernel's actual dequant vs BF16).
// Writer is created on first call; rows appended per matmul.
int tessera_open_l1_sidecar(
    tessera_context_t * tctx,
    const char * output_dir  // one .dequant.f32 file per tensor
);

int tessera_read_l1_sidecar(
    const char * path,
    struct tessera_l1_row ** out_rows,
    int64_t * out_n_rows
);

// L1.5 sidecar: FP16 reference for the dequant. Same v3 schema (c++-port-design.md 5.1).
int tessera_open_l15_sidecar(
    tessera_context_t * tctx,
    const char * output_dir
);

int tessera_read_l15_sidecar(
    const char * path,
    struct tessera_l15_row ** out_rows,
    int64_t * out_n_rows
);

// IOReport telemetry: one row per token. Swift polls latestIOReport() at 1 Hz.
int tessera_poll_io_report(
    tessera_context_t * tctx,
    struct tessera_io_report * out_report
);

// Battery current (the hero metric, decision 2 + C4).
int tessera_poll_battery(
    int32_t * out_current_ma,
    int32_t * out_thermal_state
);

// Reasoning model support (section 5.5).
typedef enum {
    TESSERA_REASONING_DISABLED = 0,
    TESSERA_REASONING_ENABLED  = 1,
    TESSERA_REASONING_AUTO      = 2,  // model decides
} tessera_reasoning_mode_t;

int tessera_set_reasoning_mode(
    tessera_context_t * tctx,
    tessera_reasoning_mode_t mode
);

// Read the latest reasoning token (CoT stream). Returns 0 on
// success, -1 on EOG. The reasoning token is written to
// out_buf (caller-allocated, >= 256 bytes recommended).
int tessera_read_reasoning_token(
    tessera_context_t * tctx,
    char * out_buf,
    int * out_len
);

// Returns the duration (in seconds) of the most recent
// reasoning pass. Set on EOG of the reasoning channel.
// Returns -1.0 if no reasoning pass has completed.
double tessera_last_reasoning_duration(
    tessera_context_t * tctx
);

// Web search context (section 5.6). The web search context
// is allocated by tessera_web_search_init and freed by
// tessera_web_search_free. The search is performed by the
// Swift side; the C side exposes only the search-context
// lifecycle + the prompt-folding logic.
typedef struct tessera_web_search tessera_web_search_t;

tessera_web_search_t * tessera_web_search_init(
    const char * tavily_api_key
);

void tessera_web_search_free(
    tessera_web_search_t * ws
);
```

The `tessera_modality_t` enum is the M5 decision (BOTH modality
ID + per-modality components). The `tessera_set_modality` call is
made before every `llama_decode`; the engine routes the act_scale
lookup and the L1 / L1.5 sidecar row tagging by the current
modality. This matches `docs/multimodal-calibration-design.md:
561-606` (section 5.1, 5.2, 5.3).

The `tessera_set_policy` accepts the calibration-policy.v1 +
modality_scales JSON produced by the AWQ-evolve runner
(`docs/multimodal-calibration-design.md:348-386`, schema diff
v1 -> v2). The runner writes it to disk; the engine reads it. M2
hard-errors on missing modality_scales - the Swift side surfaces
the error and refuses to load.

The `tessera_set_imatrix` accepts the modality-tagged imatrix v2
(`docs/multimodal-calibration-design.md:410-482`, M3). The
`modality_breakdown` field is consumed by the engine to populate
the per-modality AWQ alpha; the per-modality component is
selected by the modality ID (M5).

The C FFI is THIN. The C++ side does the work:

- `tessera_context_init` calls into the G3-G5 C++ port
  (`tools/quantize/tessera/tessera-context.{h,cpp}` in the C++
  port design section 2.1) to wire the runtime to the L1 sidecar
  writer, the L1.5 sidecar reader, the imatrix v2 reader, and the
  policy reader.
- `tessera_set_modality` mutates the engine's `current_modality`
  field; the next matmul uses it for act_scale lookup and
  sidecar row tagging.
- `tessera_open_l1_sidecar` opens a sidecar file per tensor, the
  first time the tensor is dequanted. The writer is the
  `common/tessera-debug/tessera-sidecar-v3.cpp` already in the
  C++ port (G5).
- `tessera_poll_io_report` reads the IOReport subsamples from the
  channel opened in `tessera_context_init` (C3, C4).

### 3.4 How the engine loads a Tessera-quantized GGUF

The flow is the same as `LlamaContext.create_context` in
`/Users/user/Developer/GitHub/llama.cpp/examples/llama.swiftui/
llama.cpp.swift/LibLlama.swift:62-91` (30 LoC, calls
`llama_backend_init`, `llama_model_load_from_file`,
`llama_init_from_model`), with three additions:

1. After `llama_model_load_from_file`, read the `tessera.*`
   metadata fields (`docs/tessera.md` section 1 for the field
   list). Validate that `tessera.version` matches the C++ port's
   `TESSERA_KERNEL_VERSION`. Hard-fail if mismatched.
2. After `llama_init_from_model`, call `tessera_context_init` with
   the path to the calibration-policy.v1 + modality_scales JSON
   (if present on disk next to the GGUF; the GGUF metadata
   `tessera.policy.path` points to it; C10).
3. After `tessera_context_init`, call `tessera_set_imatrix` with
   the imatrix v2 path (if present; `tessera.imatrix.path` in
   the GGUF metadata).

The M2 hard-error on missing modality_scales is enforced at
step 2: if the policy is multi-modal and modality_scales is
absent, `tessera_context_init` returns NULL and the Swift side
throws `TesseraError.missingModalityScales`. This matches
`docs/multimodal-calibration-design.md:329-348` (section 2.1,
the in-place extension decision).

### 3.5 How the engine emits L1 / L1.5 sidecars (v3 schema)

The v3 sidecar format is defined in
`docs/c++-port-design.md:595-621` (section 4.3) and the C++ port
already implements the writer (G5, `tools/quantize/tessera/
tessera-sidecar-v3.cpp`). The engine enables the writer via
`tessera_open_l1_sidecar(tctx, output_dir)`. Per-tensor, the C++
sidecar hook is fired by the dequant kernel (see
`docs/runtime-aware-pipeline.md:104-117` for the kernel
instrumentation pattern).

The Swift side reads the sidecars for the SidecarViewer (Mac)
and the ABReplayView (Mac). It does not parse the binary F32
payload; it dispatches to a tiny C function
`tessera_l1_sidecar_summary` that emits JSON (mean, max,
p99, top-k) so the Swift view does not have to allocate 4 GiB
of F32 in the heap. The C function is ~80 LoC.

### 3.6 How the engine reports IOReport telemetry back to Swift

The flow matches `docs/tessera-coreml-conversion-design.md:
section 6` (IOReport telemetry design). The C side opens the
IOReport channels in `tessera_context_init` (C3, the "Energy
Model" channel for ANE power; the DVFS channel as a fallback for
ANE activity; the GPU power channel; the DRAM power channel;
the battery current channel; the thermal state channel). The
Swift side calls `tessera_poll_io_report` at 1 Hz (or at
`CADisplayLink` callback for 60 Hz; default is 1 Hz to keep
the C bridge cheap).

The `TesseraIOReporter` Objective-C class is the bridge; it owns
the IOReport subscription. It is in
`Sources/TesseraCore/TesseraIOReporter.swift` (~120 LoC) and
links `IOKit.framework` and `IOSurface.framework` (private
frameworks; the App Store risk is documented in section 12 and
section 11).

### 3.7 The C FFI is THIN

The 10 C functions above total ~400 LoC of C. The Swift wrapper
is ~400 LoC. The C++ side does the work, including:

- the dequant kernel
- the L1 sidecar writer
- the IOReport subscriber
- the policy / imatrix reader
- the per-modality act_scale dispatch

This matches the llama.cpp production style: thin wrappers, no
new abstractions. The Swift surface is a one-to-one mirror of
the C surface; there is no Swift-side OR-M, no async wrappers
beyond `AsyncThrowingStream` for `generate`, no Swift-specific
state machine. The C++ port owns the design.

---

## 4. PrismAgent pattern porting

The PrismAgent surface has 197 Swift files in the worktree. Most
of them are Prism-specific (Neo4j graph, Chromium embedded, XPC,
P2P, agent loop with tool calls). A small minority are patterns
that translate cleanly to the Tessera app. This section
documents each pattern, what it does, the relevant file, and the
verdict (KEEP / REPLACE / SKIP / RENAME).

### 4.1 InferenceAdapter protocol - KEEP, REPLACE inner

File: `PrismAgent/PrismAgentiOS/PrismAgentiOS/InferenceAdapter.
swift:1-25`. The protocol:

```swift
@MainActor
public protocol InferenceAdapter: AnyObject {
    func loadCImage(at url: URL) async throws
    func generateText(prompt: String) -> AsyncThrowingStream<String, Error>
    func synthesizeSpeech(text: String, voice: String) async throws -> Data
    var isReady: Bool { get }
    var tokensPerSecond: Double { get }
}
```

This is the right shape. The Tessera port:

- Renames `loadCImage(at:)` to `loadGGUF(at:tesseraSidecar:policy:)`
  to match the Tessera file shape (C10: GGUF metadata is primary,
  sidecar is override).
- Renames `generateText(prompt:)` to `generate(prompt:modality:
  config:)` to surface the M5 modality ID.
- Replaces the `synthesizeSpeech` method with `loadTTSAssets`
  (deferred to v2; v1 does not ship TTS).
- Adds `telemetry` as a published property: the live
  `TesseraIOReport` (C3, C4).
- Adds `l1SidecarURL(for tensor:)` and `l15SidecarURL(for tensor:)`
  for the Mac companion to read.

The protocol stays. The engine is the new CoreML-backed
implementation; the iOS inference adapter is a thin wrapper.

### 4.2 ModelStore - KEEP, TESSERA-FLAVOUR

File: `PrismAgent/PrismAgent/ModelStore.swift:1-229`. The
catalogue + the `ModelDownload` (URLSession with progress
tracking) + the `installedModels` set + `scanInstalled`. The
Tessera port keeps the same surface but:

- Adds a `TesseraModelDescriptor` per entry with the
  `tessera.profile` field (e.g. `TSQ-T640-AWQ-SR-U-M3`), the
  per-modality `awq_alpha` vector, the `.mlmodelc` path, and the
  imatrix v2 path.
- Replaces the "is downloaded" check with two checks: "is the
  GGUF present" and "is the .mlmodelc present" (C2: the .mlmodelc
  is baked at quantize time, so on iOS the .mlmodelc ships in the
  `.app` bundle; the GGUF is for the Mac).
- Replaces the Prism-specific `authorized` flag (line 179-182,
  the `~/.prism/auth` file) with `tesseraBaked` (a bool
  indicating whether the model was Tessera-quantized).
- Drops the `LocalModelDiscovery` reference (line 9, the on-disk
  filesystem scan); the Tessera model store reads the `models/`
  bundle directory on first launch and registers the entries.

### 4.3 ModelStoreView - KEEP, REPLACE, ADD BADGES

File: `PrismAgent/PrismAgent/ModelStoreView.swift:1-428`. The
horizontal-scroll `ModelSection` (line 157-179) + the
`ModelCard` (line 183-333) + the `LocalModelCard` (line 337-422).
The Tessera port keeps the layout but:

- Replaces the `compatibleDevices: [String]` (PrismAgent
  line 18) with `backends: [TesseraBackend]` (an enum:
  `ane`, `metal`, `cpu`). The badge mapping (line 316-332) is
  unchanged in shape; only the values change.
- Adds a `TesseraProfileBadge` per card: a small pill showing
  `TSQ-T640-AWQ-SR-U` with the effective-bit count (e.g. "3.5
  bit") on a second line.
- Adds a `.mlmodelc` status indicator: green check if the
  .mlmodelc is bundled; gray dash if the model is stock (no
  .mlmodelc).
- Drops the `compilePipelineSection` (line 119-152, the
  "Compile for Device" UI for Prism's cimage format). The
  Tessera equivalent lives on the Mac Studio surface, not in
  the iOS model store drawer.

### 4.4 ConversationStore - KEEP, TESSERA-FLAVOUR

File: `PrismAgent/PrismAgent/ConversationStore.swift:1-172`. The
tiered storage (journal on disk + ring buffer + lazy reverse
prefetch, line 4-9 docs) is the right pattern. The Tessera
port:

- Replaces the `StoredMessage` (PrismAgent's, has `imageAttachment
  :ImageAttachment?`) with a `TesseraMessage` that has
  `modality: TesseraModality` and a `payload: TesseraPayload`
  (an enum: `.text(String)`, `.image(URL)`, `.audio(URL)`).
- Replaces the `ImageCacheService` (line 38, 121-123) with a
  `TesseraAttachmentStore` that handles text + image + audio
  attachments.
- Replaces the `SpotlightIndexer` reference (line 98) with the
  Tessera-side indexer (deferred to v2).
- Keeps the journal + ring buffer + prefetch pattern. The
  `workingWindowSize = 200` and `prefetchChunkSize = 100`
  constants (line 22-23) stay.

### 4.4a ConversationList - KEEP, ADAPT for Tessera drawer

File: `PrismAgent/PrismAgentiOS/PrismAgentiOS/ConversationList.swift:1-56`
(missed in the previous revision; this is the iOS-side
conversation list view that pairs with `ConversationStore`).
The pattern is `List` of `NavigationLink`s into `ChatView`,
with `swipeActions` for delete. The Tessera port lifts this
to the iOS chat history drawer (section 5.7) and adds the
`BreathingDot` for active inference, the recency bucketing
(Today / Yesterday / 2d ago / ...), the pin action, and
the rename action. The Mac side uses the same
`ConversationList` adapted to a `NavigationSplitView` sidebar.
The recency-bucketing utility is `TesseraChatHistory.swift`
in TesseraCore (~200 LoC), ported from the AWS sample's
`react-native/src/history/HistoryGroupUtil.ts:21-68` pattern
(but in Swift, not TypeScript).

### 4.5 Plan / QuantizationPlan - KEEP, RENAME

PrismAgent does not have a `Plan.swift` in the iOS / Mac
sources we read; the "Plan" idea lives in the agent loop (the
AutonomousOrchestrator generates a plan, the user approves it,
the orchestrator runs it). The Tessera equivalent is the
`QuantizationPlan`:

```
calibrate -> evolve -> quantize -> convert -> run
```

The five steps are: pick a calibration corpus (M1), run
AWQ-evolve (M2), quantize with the resulting policy (G3), convert
to .mlmodelc (C2), ship to iPhone for run (C8). The
QuantizationPlan Swift type is a `Codable` struct that the Mac
Studio editor mutates and the iOS app reads for display.

### 4.6 AgentObserver / TelemetryObserver - KEEP, RENAME

PrismAgent does not have an `AgentObserver.swift`; the observer
pattern is implicit in the agent loop (the AgentOrchestrator
pushes events to the StreamHandler at `PrismAgent/
PrismBridgeAdapter.swift:49-67`). The Tessera port lifts this
into a first-class `TelemetryObserver` Swift type that consumes
the L1 + L1.5 + B + C + E layers:

- L1: per-tensor dequant sidecar (the kernel's actual dequant
  vs BF16, see `docs/runtime-aware-pipeline.md:51-117`)
- L1.5: FP16 reference sidecar (see `docs/c++-port-design.md:
  638-696`)
- B: per-layer sensitivity rank (the L2/L3/L4 output, see
  `docs/runtime-aware-pipeline.md:138-145`)
- C: per-kernel latency LUT, modality-tagged (see
  `docs/multimodal-calibration-design.md:772-784`)
- E: fidelity predictor (the E2E quality estimate, see
  `docs/runtime-aware-pipeline.md:218-227` and the Phase E
  smoke + unit tests already in the C++ port)

The `TelemetryObserver` is the Swift object that exposes these
five layers to the UI; it owns the polling loops (1 Hz for
IOReport, on-demand for sidecars, lazy for B/C/E) and the
serialization to JSON for export.

### 4.7 SubAgent / QuantizationWorkerPool - KEEP, RENAME

File: `PrismAgent/PrismAgent/SubAgentOrchestrator.swift:1-100`.
The `SubAgent` lifecycle (spawn -> mark running -> complete ->
move to completedAgents, line 26-77) is the right pattern. The
Tessera port lifts it to `QuantizationWorkerPool` for parallel
per-tile work: the calibration runner spawns N workers
(one per tile, N = 8 by default), each runs an
AWQ-evolve generation, the pool aggregates the results into the
joint policy.

The Tessera port keeps the `activeWorkers` / `completedWorkers`
separation, the `cancelAll` (line 70-75), the `reset` (line
77-81). It drops the `parentPlanID` and the `goal: String` (line
27, 86) - the QuantizationWorker has a `tileIndex: Int` and a
`policyGenes: QuantizationGenes` instead.

### 4.8 - 4.14 SKIP summary

The following PrismAgent patterns are SKIP (Prism-specific, no
Tessera equivalent). One-line per pattern; no port.

- `AgentToolDispatcher` (`PrismAgent/AgentToolDispatcher.swift:1-292`):
  intent classifier + tool dispatch; no agent loop in Tessera.
- `PrismVoiceEngine` (visible in PrismAgent file list line 39):
  TTS, deferred to v2.
- `ChromiumView` / `BrowserTools` / `ChromeCDP` /
  `PrismBridgeAdapter.swift:466-1105` (semantic DOM reducer +
  injection guard, 640 LoC): no embedded browser.
- Neo4j / Valkey / LMDB: the Prism memory stack; Tessera has
  only the conversation journal (local JSON).
- `PrismEngineXPCProtocol` / `PrismCredentialService` (in
  `Package.swift:115-125`): XPC + separate credential process;
  Tessera links the xcframework directly.
- `P2PRouter` (`PrismAgent/P2PRouter.swift:1-72`): the
  NetworkSession + SymmetricKey; no P2P in Tessera.
- `MacRemoteView*` + `RemoteViewBridge` + `RemoteKeyboardBridge`:
  the iOS-as-Mac-remote-display pattern; Tessera Mac and iOS
  are independent, the Plan syncs via iCloud Drive.

### 4.15 SubAgent pool summary

| Prism pattern | File | LoC | Verdict |
| --- | --- | --- | --- |
| InferenceAdapter protocol | PrismAgentiOS/InferenceAdapter.swift | 83 | KEEP, REPLACE inner |
| ModelStore | PrismAgent/ModelStore.swift | 229 | KEEP, TESSERA-FLAVOUR |
| ModelStoreView | PrismAgent/ModelStoreView.swift | 428 | KEEP, REPLACE, ADD BADGES |
| ConversationStore | PrismAgent/ConversationStore.swift | 172 | KEEP, TESSERA-FLAVOUR |
| PrismChatView | PrismAgent/PrismChatView.swift | 479 | KEEP, SIMPLIFY (drop voice + agent) |
| PrismEngineApp | PrismAgent/PrismEngineApp.swift | 14 | KEEP, minimal |
| PrismAgentiOSApp | PrismAgentiOS/PrismAgentiOSApp.swift | 39 | KEEP, minimal |
| PrismModelManager | PrismAgent/PrismModelManager.swift | 171 | KEEP, simplify to ModelStore |
| PrismModelCardView | PrismAgent/PrismModelCardView.swift | 206 | KEEP, REPLACE compile/download |
| PrismBridgeAdapter | PrismAgent/PrismBridgeAdapter.swift | 1105 | PARTIAL: keep StreamHandler pattern only |
| SubAgentOrchestrator | PrismAgent/SubAgentOrchestrator.swift | 100 | KEEP, RENAME QuantizationWorkerPool |
| AutonomousOrchestrator | PrismAgent/AutonomousOrchestrator.swift | 159 | SKIP (no agent loop) |
| AgentToolDispatcher | PrismAgent/AgentToolDispatcher.swift | 292 | SKIP |
| P2PRouter | PrismAgent/P2PRouter.swift | 72 | SKIP |
| Voice (PrismVoiceEngine) | PrismAgent/PrismVoiceEngine.swift | (not read) | SKIP v1, v2 |
| Chromium / DOM reducer | PrismBridgeAdapter.swift:466-1105 | 640 | SKIP |
| Neo4j / Valkey / LMDB | (across PrismAgent) | n/a | SKIP |
| XPC / Credential Service | PrismEngineXPCProtocol.swift, Package.swift:115-125 | n/a | SKIP |
| Mac remote view | MacRemoteView*.swift, RemoteViewBridge.swift | n/a | SKIP |

Net: ~1,920 LoC of PrismAgent ports cleanly into the Tessera
Studio surface. ~5,000 LoC of PrismAgent is Prism-specific and
is skipped. The Tessera port is ~4,800 LoC of new Swift; ~40%
of it is a port, ~60% is Tessera-flavoured new code (the
Studio surface, the CalibrationSession, the SidecarViewer).

---

## 5. The Tessera-specific UI surfaces

This section documents the iOS + Mac surfaces that are
Tessera-specific (no Prism equivalent) and the surfaces that
are ports of the llama.swiftui engine shape.

### 5.1 Chat (iOS)

The chat surface is a port of the LlamaState pattern
(`/Users/user/Developer/GitHub/llama.cpp/examples/llama.swiftui/
llama.swiftui/Models/LlamaState.swift:1-196`), extended with:

- The modality picker (text / image / audio) - a horizontal
  `SegmentedControl` above the input bar.
- The engine selector (Tessera + CoreML, Tessera + Metal, Stock
  + Metal) - a `Menu` in the navigation bar, with badges.
- The inference controls (temperature, top-p, top-k, max
  tokens) - a `Sheet` triggered by a gear icon.
- The live IOReport (C3, C4) - a collapsible `DisclosureGroup`
  at the bottom of the chat scroll view, showing five
  `Gauge`s: ANE power, GPU power, DRAM power, battery current,
  thermal state.
- The 30-minute flight test button - a `Button` in the
  navigation bar that starts the test, runs an automated prompt
  stream, and renders a summary card at the end.

The message bubble is a port of
`PrismAgent/PrismAgent/PrismChatView.swift:288-414` (the
`MessageBubble` view) with the image attachment replaced by
the modality-aware payload (text + image + audio). The
streaming dot (line 442-457) stays.

The chat surface is keyboard-first: tapping the input field
shows the keyboard, the user types, taps send, the engine
streams tokens. The modality picker is for "I have an image
to attach" or "I want to dictate a voice memo"; the default
is text-only.

The IOReport is the new thing. Five gauges, updated at 1 Hz,
are the answer to "why is this app different from any other
on-device inference app." We do not hide the metric.

The 30-minute flight test is the second new thing. It is
visible to the user, not a separate QA harness. The user taps
"Flight Test" once, gets a 30-minute timer, walks away, comes
back to a summary card with the hero metric (mWh/token) and
the thermal events. The summary is exportable.

### 5.2 A/B Compare (iOS)

The A/B Compare view runs two engines side-by-side on the same
prompt and surfaces the comparison.

- **Layout**: a `VStack` of two `ChatView`-like surfaces, the
  top labelled "Tessera" with an ANE badge, the bottom labelled
  "Stock" with a GPU badge. The input bar is shared: one
  prompt, two engines, one "Send" button.
- **Per-token latency table**: when both engines finish, the
  view renders a table with columns `token`, `Tessera
  latency`, `Stock latency`, `delta`. The table is virtualised
  (only the first 20 + last 20 tokens are shown; the full
  table is in the export).
- **First-token latency comparison**: a single row at the top
  with `Tessera: 240ms`, `Stock: 720ms`, `delta: -480ms` (or
  similar). The headline.
- **PPL proxy**: the view runs a held-out 50-token validation
  set through both engines, computes the perplexity on each,
  and reports `Tessera PPL: 5.42`, `Stock PPL: 5.39`,
  `delta: 0.03`. The "PPL cost of going 4-bit" number.
- **Memory footprint**: the view reports `Tessera RSS: 3.1 GB`,
  `Stock RSS: 3.4 GB`, `delta: -0.3 GB` (the 4-bit effective
  bits of the Tessera 12B are tighter than Q4_0's 4.5).
- **Export**: the per-token table + the IOReport time-series
  + the PPL numbers are written to a JSON file in the
  Documents directory for the Mac companion to ingest.

The A/B Compare view is the "show your work" surface. The
Tessera story is "we are faster, smaller, and the quality cost
is bounded" - the table is the proof.

### 5.3 Studio (Mac)

The Studio is the Mac-only primary surface. The user opens the
Mac app, lands on the Studio, picks a model from the
`ModelStore` drawer, picks a calibration corpus from the
`CorpusPicker`, and runs the five-step pipeline.

- **QuantizationPlan editor**: a `Form` view with five sections
  (calibrate, evolve, quantize, convert, run). The user picks
  the model, the corpus, the AWQ-evolve generation count, the
  output bit-width, the .mlmodelc output path. The form
  persists the plan as a JSON file in `~/Library/Application
  Support/TesseraStudio/plans/<id>.json`.
- **L1 / L1.5 sidecar viewer**: a `Table` view with columns
  `tensor`, `L1 mean`, `L1 max`, `L1 p99`, `L1.5 mean abs`,
  `L1.5 max abs`. The rows are sorted by L1 p99 descending; the
  top 20 are highlighted. The user clicks a row to see a
  per-position heatmap (the dequant error per row of the
  tensor).
- **Per-layer error table (B)**: a `Table` view with columns
  `layer`, `sensitivity rank`, `B mean`, `B max`, `recommended
  bit-width`. The recommended bit-width comes from the L5
  orchestrator (`docs/c++-port-design.md:171-181`).
- **Latency LUT (C)**: a `Table` view with columns `kernel`,
  `shape`, `modality`, `p50`, `p99`. Grouped by
  (shape, kernel_id, modality) per
  `docs/multimodal-calibration-design.md:772-784`.
- **Fidelity predictor (E)**: a `Chart` view of the predicted
  E2E PPL vs the actual PPL on a held-out probe set. The
  diagonal `y = x` is overlaid; the points are the per-prompt
  predictions.
- **The 30-minute flight test is replayable from the sidecar
  data**: the user picks a recorded flight test (an exported
  JSON from the iPhone app), the Studio replays the
  inference steps against the recorded sidecars, and the
  latency / power / PPL surfaces re-render. The replay is
  for debugging: "why was the flight test on 2026-08-15
  weird?"

The Studio is a workbench. The user runs the plan, watches
the L1 / L1.5 sidecars fill in, sees the L5 recommendations
emerge, ships the .mlmodelc to the iPhone via iCloud Drive.

### 5.4 Web search (provider-shaped, keyless by default)

_Updated 2026-07-31: the search path is no longer Tavily-only. It is a
provider seam with a keyless default, per the no-egress-by-default doctrine
and the resolution of open question 11.13._

The chat surface has a web search button next to the modality
picker. Tapping it enables web search for the next message;
the button is toggled off by default (privacy by default).
When enabled, the engine routes the user's prompt through a
two-step pipeline: (1) the prompt is sent to the active search
provider with `max_results: 5`; (2) the top results are folded
into the prompt as a context block, the engine streams the
synthesis, and the citation links are rendered inline.

The search backend is a seam, not a hard dependency. `TesseraWebSearch`
is a thin facade over a `TesseraSearchProvider`, and the active provider
is chosen in settings (`tessera.settings.searchProvider`):

- DuckDuckGo (default, keyless). Hits the static HTML endpoint
  (`https://html.duckduckgo.com/html/`) with URLSession and parses the
  results page with SwiftSoup. No API key, no vendor account. The query
  still leaves the device (it is a web search), but there is no key, no
  billing, and no per-account query log to manage.
- SearXNG (opt-in, self-hosted). Points at a SearXNG instance the user
  runs themselves (`tessera.settings.searxngBaseURL`) and reads its JSON
  API. Keyless, and keeps the query on the user's own infrastructure.
- Tavily (opt-in, vendor key). The original provider, kept for its
  agent-tuned results and clean citations. Requires
  `tessera.settings.tavilyAPIKey` (or `TAVILY_API_KEY`) and is full
  third-party egress, so it is never the default.

Every provider returns the same `TesseraSearchHit` (url, title, content)
and degrades to an empty result rather than throwing, so research falls
back to "no sources" instead of crashing. The query is constrained to
<= 400 chars.

The Swift surface (TesseraCore):

```swift
public protocol TesseraSearchProvider: Sendable {
    var id: String { get }                 // "duckduckgo" | "searxng" | "tavily"
    var configurationNote: String? { get } // nil when ready
    func search(query: String, maxResults: Int) async -> [TesseraSearchHit]
}

public actor TesseraWebSearch {
    public init(provider: (any TesseraSearchProvider)? = nil) // defaults from settings
    public var providerID: String { get }
    public var configurationNote: String? { get }
    public func search(query: String, maxResults: Int = 5) async -> [TesseraSearchHit]
}
```

The chat surface injects a `web_search` chip above the message
when the user enables it: `Web search: 5 sources, top hit
"..."`. The chip is the same shape as the file attachment
chips. The user can tap the chip to expand the source list.

The privacy implication depends on the provider. With the keyless
DuckDuckGo default the query leaves the device but carries no account
or key. With self-hosted SearXNG the query stays on the user's own
infrastructure. With Tavily the prompt is sent to a third-party API.
Because every provider egresses the query somewhere, the research tool
runs at approval level `.prompt` (it asks before searching) rather than
silently notifying. The Settings surface discloses the active provider;
the privacy policy is in the App Store metadata (section 13.3) and the
in-app About screen.

The web search is on the chat surface, not on the A/B Compare
or the Mac Studio. A/B is offline-by-construction; the Mac
Studio runs the calibration corpus from disk.

### 5.5 Reasoning model support (Gemma 4 12B Unified)

Gemma 4 12B Unified is a reasoning model: before emitting the
final answer, the model emits a chain-of-thought (CoT)
reasoning trace. The engine surfaces both: the CoT in a
`Thinking for Ns` block (collapsed by default, expandable),
the final answer in the normal message body. The pattern is
the Locara + Anthropic + DeepSeek convention (see the research
notes in the section 4 / 5.x reference); default collapsed
because most users don't look at CoT (Claude Code's stated
rationale).

The C FFI adds the reasoning channel:

```c
typedef enum {
    TESSERA_REASONING_DISABLED = 0,
    TESSERA_REASONING_ENABLED  = 1,
    TESSERA_REASONING_AUTO      = 2,  // model decides
} tessera_reasoning_mode_t;

int tessera_set_reasoning_mode(
    tessera_context_t * tctx,
    tessera_reasoning_mode_t mode
);

// Read the latest reasoning token (CoT stream). Returns 0 on
// success, -1 on EOG.
int tessera_read_reasoning_token(
    tessera_context_t * tctx,
    char * out_buf,
    int * out_len
);
```

The Swift `TesseraToken` gains a `reasoningToken: String?`
field. The chat surface renders the CoT in a separate
`ThinkingBlock` SwiftUI view:

```swift
struct ThinkingBlock: View {
    let isStreaming: Bool
    let durationSeconds: Double
    let wordCount: Int?
    let body: String  // full reasoning text (rendered in monospaced font)

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "sparkles")
                    .symbolEffect(.pulse, isActive: isStreaming)
                Text(isStreaming
                     ? "Thinking..."
                     : "Thought for \(String(format: "%.1f", durationSeconds))s")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                if let n = wordCount {
                    Text("\(n) words")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            if expanded {
                Text(body)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }
}
```

The reasoning toggle in the engine selector menu has three
states: `Auto` (model decides, default), `On` (force CoT
on), `Off` (suppress CoT). The choice persists per-model
in `UserDefaults`. The reasoning tokens are written to the
L1 sidecar (`reasoning_content` field per row) and to the
v3 sidecar export (section 6.5 of the CoreML design).

The latency budget: reasoning models emit CoT before the
final answer, so first-token latency is 2-10x higher than
non-reasoning models. The IOReport reflects the delay:
`battery_delta_mWh` for a reasoning run is higher than for
a non-reasoning run on the same prompt. The chat surface
surfaces `time_to_first_reasoning_token` and
`time_to_first_answer_token` as two separate metrics in the
title sub-row.

The Mac Studio's CalibrationSessionView surfaces the per-tensor
reasoning cost: which tensors contribute to the CoT vs the
final answer (from the L1 sidecar `tensor_kind` field, a
new addition in v3). This is for debugging, not for v1 UX.

### 5.6 Rich renderers (markdown, code, Mermaid, HTML)

The chat surface renders the model's output as rich content,
not plain text. The Swift surface is
`TesseraRichRenderer.swift` (~300 LoC, TesseraCore), with
SwiftUI views per element type. The renderer is
streaming-aware: it parses incrementally as tokens arrive
(debounce 64 chars or 200ms, whichever is first) and
re-parses the full visible chunk on each update.

The supported elements:

- **Markdown** (headings, paragraphs, lists, links, bold,
  italic, code spans, blockquotes, tables). Uses
  Apple's `Markdown` framework (`#available(iOS 17, *)`)
  with manual `PresentationIntentAttribute` walking for
  header levels (per the SO answer cited in the research;
  SwiftUI's `AttributedString(markdown:)` doesn't render
  headers natively). For iOS 16 fallback, fall back to
  `AttributedString(markdown:)` and lose header styling.
- **Code blocks** with language detection. Highlights via
  Splash (Swift-only, no JS) or a custom rule-based
  highlighter; ~100 LoC. Languages: Python, TypeScript,
  Swift, Rust, C, C++, Go, JSON, YAML, Bash. Falls back to
  monospaced plain text for unknown languages.
- **Mermaid diagrams** rendered to SVG via a Swift-native
  renderer (~600 LoC, based on the Mermaid grammar). For
  v1 we support `flowchart`, `sequenceDiagram`,
  `classDiagram`, `stateDiagram-v2`. The SVG is rendered
  via `WKWebView` (a local HTML host with the SVG inline,
  no network). A "Copy SVG" and "Open in fullscreen" button
  appears on hover.
- **HTML preview** for `<html>` code blocks. The HTML is
  rendered in a `WKWebView` sandbox (no JS, no network).
  The user can copy the HTML or open in fullscreen.
- **Tables** rendered as `Table` (SwiftUI, iOS 16+). The
  table is selectable, copyable, and adapts to dynamic
  type.
- **Math** (LaTeX-style) is deferred to v2. KaTeX is a
  JavaScript library; the v2 path is a native Swift
  renderer or a `WKWebView` with KaTeX bundled.

The streaming caveat: as the model emits tokens, the
markdown is incomplete (an unclosed code block, an
unclosed table). The renderer holds the last block as a
"pending" state (lighter colour, no syntax highlighting
yet) and promotes it to "complete" when the closing
delimiter arrives. The `Markdown` framework's incremental
parse is the foundation; the pending-state UI is in
`TesseraRichRenderer.swift`.

The Mac Studio's CalibrationSessionView also uses the
markdown renderer for the policy JSON, the imatrix
preview, and the L1 / L1.5 sidecar summaries.

### 5.7 Chat history drawer (NavigationSplitView)

The iOS app uses SwiftUI's `NavigationSplitView` for the
chat history pattern (the SwiftUI native replacement for
the React Navigation drawer in the AWS sample). On iPad
the sidebar is always visible (two-column layout); on
iPhone the sidebar is hidden until the user swipes from
the left edge or taps the menu button in the top-left.

The Mac app uses the same `NavigationSplitView` for the
chat history drawer in the Studio's `Run` tab. The
sidebar shows the conversation list, the detail shows
the active conversation.

The drawer contents:

- **"New Chat"** button at the top.
- **Pinned conversations** (3-5 max, user-pinnable, with
  a pin icon).
- **Conversation list** grouped by recency (the
  `HistoryGroupUtil.ts` pattern from the AWS sample,
  ported to Swift):
  - Today
  - Yesterday
  - 2 days ago
  - ... up to 6 days ago
  - Last week
  - ... up to 4 weeks ago
  - Then by month (e.g. `2026.07`)
- Each row shows: title (first user message, truncated
  to 32 chars), last-update time, a breathing-dot if
  the conversation is currently streaming inference
  (the `BreathingDot` from the AWS sample's
  `CustomDrawerContent.tsx:45-68`).
- Swipe actions: rename, pin, share, delete.
- Long-press: context menu with the same options.
- Search field at the top (deferred to v2; v1 is a
  scrollable list).

The Swift type is `TesseraChatHistoryStore`
(TesseraCore, ~200 LoC). It uses the same tiered
storage as `ConversationStore`: a journal on disk
(JSON-Lines, one row per event), an in-memory
ring buffer for the last 200 messages of the
active conversation, a lazy reverse prefetch for
older messages.

The `BreathingDot` is a SwiftUI
`@State private var opacity: Double` with
`withAnimation(.easeInOut.repeatForever)`,
identical pattern to the AWS sample.

### 5.8 Settings (iOS, Mac) - the full settings surface

The Settings surface is required for App Store
publishability (section 13). It is a single
`NavigationStack` with the following sections. Each
section is a `Form` with the corresponding toggles; the
underlying state is `TesseraSettings` in TesseraCore
(~150 LoC), a `@Observable` class backed by
`@AppStorage` for the simple toggles and a JSON file in
the Documents directory for the structured data (the
IAP catalog, the model aliases).

- **Privacy** (App Store requires clear disclosure of
  data use, see section 13.3):
  - `Send prompts to web search provider` (default ON,
    warns the user; toggles `TesseraWebSearch.isEnabled`)
  - `Send prompts to telemetry collector` (default OFF;
    the testflight build has the collector on by
    default)
  - `Share analytics with Tessera project` (default OFF;
    the analytics are crash reports + the v3 sidecar
    export, opt-in only)
  - `Show AI-generated content labels` (default ON, per
    EU AI Act Article 50 + Apple's iOS 26 AI
    disclosure; this is the "AI-generated" badge on
    every assistant message)
- **Model**:
  - The App Store IAP catalog (section 5.10.4): the list
    of purchasable .mlmodelcs, the user's purchased
    items, the "Restore Purchases" button.
  - `Default engine`: Tessera + CoreML / Tessera +
    Metal / Stock + Metal (per-model override).
  - `Default modality`: text / image / audio.
  - `Reasoning mode`: Auto / On / Off (per-model
    override, default Auto for reasoning-capable
    models).
  - `iCloud Drive sync`: ON / OFF. When ON, the .mlmodelc
    is imported from iCloud Drive; when OFF, the user
    downloads from the App Store IAP only.
  - `Delete all model files`: confirmation required.
- **Inference**:
  - `Background keep-alive for flight test` (default
    ON, explains the pink-noise audio session, required
    for the 30-minute flight test to run with the
    iPhone locked; C3).
  - `Live Activity for flight test` (default ON,
    required for the iOS 16.2+ Live Activity to
    surface the flight test progress on the Lock
    Screen).
  - `Telemetry in title` (default ON, taps the title
    to show tok/s + ANE mW + battery delta).
  - `Persist IOReport samples` (default OFF; when ON,
    the 1 Hz samples are written to a JSON file in
    Documents, used by the ABReplayView on Mac).
- **Display**:
  - `Theme`: System / Light / Dark.
  - `Font size`: respects Dynamic Type.
  - `Language`: English v1; the strings are externalised.
  - `Show reasoning by default` (default OFF; when
    ON, the `ThinkingBlock` is expanded on first
    arrival).
- **About**:
  - App version + build number.
  - The privacy policy link.
  - The terms of service link.
  - `Contact support` (mailto:).
  - `Open source licenses` (the third-party licence
    list; required for the App Store).

The Settings view is reachable from the chat history
drawer (iOS) and the app menu (Mac). The privacy
section is at the top; this matches Apple's
"Privacy & Security" convention in iOS Settings.

### 5.9 IAP / Apple-hosted background asset distribution

The .mlmodelc distribution is via the App Store. The
three pre-bundled models (Tessera 4-bit Gemma 3 4B,
Tessera 3.5-bit Gemma 4 12B Unified, Stock Q4_0
Gemma 3 4B) ship in the `.app`; additional .mlmodelcs
are purchasable via In-App Purchase. The IAP
catalog:

| Tier | Product | Price (US) | Apple cut (30%) | Net to us |
| --- | --- | --- | --- | --- |
| Small | `tessera.gemma3.4b.q4.tessera.v1` | $4.99 | $1.50 | $3.49 |
| Medium | `tessera.gemma4.12b.unified.q35.tessera.v1` | $9.99 | $3.00 | $6.99 |
| Large | `tessera.gemma4.27b.unified.q3.tessera.v1` (v2) | $19.99 | $6.00 | $13.99 |

The pricing is the App Store's tier 4 / tier 6 / tier
9; the net is documented for the user in the
Settings > About screen. The 30% cut is the standard
commission; the Small Business Program (15%) is
available at < $1M annual revenue (we are not there
yet).

The .mlmodelc is delivered as an **Apple-hosted
background asset** (`BackgroundAssets.framework`,
iOS 26+). The flow is:

1. The user buys the IAP via StoreKit 2.
2. The .mlmodelc is bundled as an Apple-hosted
   background asset pack (up to 200 GB per app,
   included in the Developer Program membership;
   `WWDC26/378` for the documentation).
3. The OS automatically downloads the asset pack when
   the IAP is purchased; the user sees a
   "Downloading..." progress in the App Store.
4. The app reads the asset pack from the local
   Background Assets cache; the engine loads the
   .mlmodelc from the cached path.

The rationale: Apple discontinued Apple-hosted
non-consumable IAP content in April 2022 (the
`SKDownload` API is deprecated). The two remaining
options are On-Demand Resources (limited to the app
bundle, requires resubmission for every update) and
Background Assets (Apple-hosted, automatic download,
iOS 26+). Background Assets is the right answer for
v1; v2 may also ship a self-hosted option for users
who have already paid for the .mlmodelc on a different
device.

The "Restore Purchases" button is in the Settings >
Model section; the `Transaction.currentEntitlements`
API (StoreKit 2) is the source of truth. The receipt
validation is on-device (no server; StoreKit 2
returns a JWS that the app verifies locally).

The IAP catalog is also browsable from the ModelStore
drawer: the "Store" tab shows the available
.mlmodelcs with a "Buy" button next to each. The buy
button triggers the StoreKit 2 `product.purchase()`
flow with a sheet for the App Store sign-in.

The Mac companion distribution stays as iCloud Drive
handoff (per the original design). The user can
either (a) build a .mlmodelc on the Mac, drop it in
iCloud Drive, import on iOS, or (b) buy a .mlmodelc
on the iOS App Store and have it Apple-hosted
delivered to the iPhone. Both paths are documented
in the Settings > Model section.

### 5.10 Onboarding + first-run

First-run is the user's first 30 seconds. The flow:

1. **Splash** (0-3s). The "What is Tessera?" splash.
2. **AI disclosure** (3-5s). The EU AI Act Article 50
   + iOS 26 AI disclosure screen: "Tessera Studio
   uses on-device AI to generate responses. All
   inference is local; no data leaves your device
   unless you enable web search." Required for App
   Store review (section 13.3).
3. **Welcome** (5-15s). The 9-component cluster
   diagram, the "from corpus to device" narrative,
   the "Buy or import a model" button.
4. **First model** (15-30s). The ModelStore drawer
   with the three pre-bundled models; the user
   picks one and the engine loads it.

The "AI disclosure" step is required by the iOS 26
App Review Guidelines (Guideline 2.1, AI
disclosure-forward enforcement). The disclosure must
be visible in the UI, not buried in the privacy
policy. The "Show AI-generated content labels"
toggle in Settings (section 5.8) controls whether
the "AI-generated" badge appears on every assistant
message; the default is ON.

The first-run also requests the iOS permissions the
app will use later: speech recognition (for audio
input), microphone (for audio input), photo library
(for image input), camera (for image input). Each
permission is requested in the context where it is
used (e.g. microphone when the user first taps the
audio modality), not all at once. This is the
"permission priming" pattern recommended in the App
Store Review Guidelines.

---

## 6. The CoreML backend integration

The C backend (`ggml-coreml`) loads the `.mlmodelc`, manages
the MLState, and routes the matmul calls to the ANE. The
Swift wrapper (`TesseraCoreMLBackend.swift`, ~150 LoC) wraps
the C API in a Swift class:

```swift
final class TesseraCoreMLBackend {
    let model: MLModel
    let state: MLState
    let inputName: String
    let outputName: String
    let actScaleInputName: String  // C8: runtime act_scale

    init(modelURL: URL, configuration: MLModelConfiguration) throws

    func predict(
        inputs: [String: MLFeatureValue],
        modality: TesseraModality
    ) throws -> [String: MLFeatureValue]
}
```

The `MLModelConfiguration` is built with
`computeUnits = .all` (ANE + GPU + CPU), `allowLowPrecision
AccumulationOnGPU = true` (ANE's preferred), and
`functionName = nil` (single function). The `MLState` is
loaded with `state = try MLState(mlModel:)` (C6: full state
API).

### 6.2 Memory layout (C5)

Locked: MMAP for the weight blobs, RAM for the activations.
Standard Apple ML stack pattern. The Swift wrapper enforces
this by not allocating a CPU-side `MLMultiArray` for the
weights; it lets the CoreML framework MMAP the `.mlmodelc`
weight blobs and only allocates the activation buffers
(`MLMultiArray`s) on demand.

### 6.3 The backend uses MLState for the KV cache (C6)

Locked: full state API (`MLState`). Public API as designed;
custom is more code for no benefit. The Swift wrapper owns
the `MLState` and exposes `stateValue(for:)` /
`setStateValue(_:for:)` to the C side via the C FFI
(`tessera_coreml_state_get` / `tessera_coreml_state_set`, ~50
LoC of C).

### 6.4 The backend routes prefill AND decode to ANE (decision 1)

Locked: both prefill AND decode run on the ANE via CoreML.
The Swift wrapper sets `computeUnits = .all` and the C
backend is the routing layer: it prefers ANE for matmul
(MMEPilogue / MIL `matmul`), falls back to GPU for ops
the ANE cannot do (e.g. some reduction patterns), falls
back to CPU last. The C backend logs the fallback (C7)
and the Swift side surfaces the fallback in the
`TesseraIOReport` (a `fallbackCount: Int` field).

### 6.5 The backend takes a modality ID per call (M5)

The M5 decision is "BOTH modality ID + per-modality
components." The backend takes the modality ID on every
`predict` call and uses it to:

- Select the act_scale input to the .mlmodelc (C8: runtime
  act_scale, one .mlmodelc with the act_scale as a runtime
  input).
- Tag the L1 / L1.5 sidecar row with the modality ID
  (`docs/multimodal-calibration-design.md:561-606`, section
  5.1, 5.2, 5.3).
- Apply the per-modality AWQ alpha from the policy
  (`docs/multimodal-calibration-design.md:495-557`,
  section 4).

The Swift wrapper passes the modality ID through
unmodified; the C backend uses it.

### 6.6 The backend emits IOReport telemetry per token (C3, C4)

Locked: IOReport is the runtime telemetry source. The
backend records ANE power, GPU power, DRAM power, battery
current, thermal state at the end of every `predict` call
(via the IOReport subsamples, see
`docs/tessera-coreml-conversion-design.md` section 6.1).
The Swift side polls at 1 Hz; the per-token rows are
written to the v3 sidecar (`docs/tessera-coreml-conversion-
design.md` section 6.5, the v3 sidecar extension).

The IOReport subscriber is in
`Sources/TesseraCore/TesseraIOReporter.swift` (~120 LoC) and
links `IOKit.framework` and `IOSurface.framework`. The
App Store risk is documented in section 11 and section 12.

### 6.7 The CoreML backend is one of two backends

The TesseraEngine supports two backends:

- `TesseraCoreMLBackend` (default on M-series iPhone and
  Mac, the primary path)
- `TesseraMetalBackend` (fallback for non-M-series or for
  users who opt out via `--device metal`, C7)

The user picks the backend in the engine selector
(section 5.1). The Tessera quantize time decision (C2)
means the .mlmodelc is bundled; the Metal fallback
re-quantizes on the fly from the GGUF (the stock K-quant
path).

---

## 7. Multi-modal calibration UI

The Mac Studio surface runs the multi-modal calibration
pipeline
(`docs/multimodal-calibration-design.md` sections 1-4).
This section documents the Swift surfaces that wrap the
C++ runner.

### 7.1 The corpus picker

The corpus picker is a `List` view with three sections:
Text (CC0 procedural generator, M1), Image (PixelProse,
M1), Audio (LibriSpeech, M1). Each section shows the
entry count, the SHA-256, the license, the per-modality
weight. The user adjusts the weights via a `Slider` per
section (the lean recommendation in
`docs/multimodal-calibration-design.md:107-145` is 1.5k /
2k / 1.5k; the user can change the per-section count, the
weights update automatically to keep the total at 5k).

The corpus picker writes a `corpus.json` file that the
C++ imatrix runner reads. The C++ runner is invoked via
the Tessera C FFI:

```c
int tessera_calibrate_corpus(
    const char * gguf_path,
    const char * corpus_json_path,
    const char * output_imatrix_path,
    const char * output_policy_path
);
```

### 7.2 The calibration runner

The calibration runner is a `View` that wraps the C++
call. The user clicks "Run Calibration," the view shows
a progress bar, a log, and a "current modality" label
(text / image / audio round-robin). When the runner
finishes, the view shows the per-modality imatrix v2
breakdown and the modality_scales (M2, M3).

The view writes the imatrix v2 and the policy JSON to
the plan directory
(`~/Library/Application Support/TesseraStudio/plans/
<id>/`). The AWQ-evolve runner picks them up next.

### 7.3 The GA runner

The GA runner is a `View` that runs AWQ-evolve for N
generations (default 50, configurable). The view shows:

- The current generation number.
- A fitness-over-generations `Chart` (the weighted
  per-modality fitness, see
  `docs/multimodal-calibration-design.md:505-557`,
  section 4.2-4.4).
- A weight visualisation (the AWQ alpha per tensor, a
  heatmap with rows = layers, columns = modality).
- The top 5 candidates of the current generation
  (fitness, alpha vector, per-modality policy).

The GA runner is the slowest step in the pipeline
(5-30 minutes for a 12B on a M-series Mac). The user
can pause / resume; the runner checkpoints the
population + the random state per
`docs/tessera.md` ("uses deterministic island populations
and a MAP-Elites archive").

### 7.4 The policy editor

The policy editor is a `Form` view that shows the
modality_scales (M2) and lets the user override the
per-modality alpha. The form has three sliders (text /
image / audio) and a "preview" button. The preview
renders the per-tensor act_scale for the current
modality as a heatmap; the user clicks through the
modalities to see how the act_scale changes.

The policy editor writes the final policy to
`plan/policy.json`. The quantize step picks it up.

### 7.5 The dequant preview

The dequant preview is a `View` that shows the L1 +
L1.5 sidecar deltas for the current policy. The user
clicks "Preview Dequant," the view runs the dequant
kernel on a small slice of the model (one tensor, 10%
of the rows), and renders:

- The L1 dequant error histogram (BF16 vs Tessera
  dequant).
- The L1.5 reference comparison (the FP16 source vs
  the Tessera dequant).
- The act_scale heatmap (per-row scaling, normalised
  to [0, 1]).

The preview is for sanity-checking the policy before
shipping the quantize + convert step.

---

## 8. The demo narrative

This section scripts the three demo flows: the 5-minute
iPhone demo, the 30-minute flight test, the Mac Studio
demo. Each flow is timestamped and lists the UI surfaces
involved.

### 8.1 The 5-minute iPhone demo (minute-by-minute)

| Min | Beat | UI surface | What happens |
| --- | --- | --- | --- |
| 0:00-0:15 | Splash | `TesseraStudioiOSApp.swift` | "What is Tessera?" splash. One sentence + the 9-component cluster diagram. |
| 0:15-0:30 | Model picker | `ModelStoreDrawer` | Three models are pre-bundled: Stock Q4_0 Gemma 3 4B, Tessera 4-bit Gemma 3 4B, Tessera 3.5-bit Gemma 4 12B Unified. |
| 0:30-0:45 | Engine selector | `ChatView` nav bar | Tessera + CoreML (default, ANE badge). The user can switch to Stock + Metal for the A/B later. |
| 0:45-1:15 | First inference | `ChatView` | User types "In one sentence, what is the capital of France?" First token in 240ms. |
| 1:15-1:45 | IOReport reveal | `TelemetryDrawer` | User expands the IOReport. Five gauges start moving. ANE power ~1.2W, GPU power ~0.1W, DRAM power ~0.4W. The "why CoreML" answer is on the screen. |
| 1:45-2:15 | Modality switch | `ChatView` | User pastes a photo. Modality picker switches to IMAGE. First image token in 340ms; per-token stream follows. |
| 2:15-2:45 | Audio mode | `ChatView` | User records a 4-second voice memo. Modality picker switches to AUDIO. Transcript streams. |
| 2:45-3:15 | Engine switch | `ChatView` | User switches to Stock + Metal. Same text prompt. First token in 720ms. The first-token latency delta is shown. |
| 3:15-4:00 | A/B moment | `ABCompareView` | User opens A/B Compare. Side-by-side. Per-token latency table. Tessera row is faster; PPL proxy shows the quality cost. |
| 4:00-4:30 | Flight test intro | `ChatView` | User taps "Flight Test." 30-minute timer starts. Automated prompt stream begins. |
| 4:30-5:00 | Wrap | `ChatView` | User stops the flight test after 60 seconds. Summary card: `mWh/token = 0.42`, `ANE power avg = 1.1W`, `thermal events = 0`. The hero metric is on the screen. |

### 8.2 The 30-minute flight test (the hero metric)

The flight test is the sustained-battery-draw measurement
(decision 2). The user runs the iPhone on battery, taps
"Flight Test," and the app runs an automated prompt stream
for 30 minutes:

- 10 prompts of text-only ("Summarise the second
  paragraph of Moby Dick in two sentences," etc.)
- 5 prompts with an attached image (round-robin from a
  bundled image set).
- 5 prompts with an attached voice memo (round-robin
  from a bundled audio set).
- Each prompt is generated to 100 tokens (configurable).

The app records:

- `battery_delta_mWh` (IOReport battery current integrated
  over the 30 minutes).
- `tokens_generated` (the denominator).
- `mWh_per_token` (the headline).
- `ane_power_avg_mW`, `gpu_power_avg_mW`, `dram_power_avg_mW`
  (the 1 Hz time-series mean).
- `thermal_events` (count of throttling transitions).
- `l1_sidecar_path`, `l15_sidecar_path` (the v3 sidecar
  paths for the export).
- `flight_test_id` (a UUID for the JSON export filename).

The summary card is rendered at the end. The export is a
JSON file in `Documents/flight-tests/<id>.json` for the
Mac companion to ingest.

### 8.3 The Mac Studio demo

The user opens the Mac app, lands on the Studio. The
demo flow is:

- **0:00-0:30**: Land on Studio. The plan library is
  empty; the user clicks "New Plan."
- **0:30-1:30**: The QuantizationPlan editor. User picks
  the model (Tessera 3.5-bit Gemma 4 12B Unified), the
  corpus (5k curated set, 1.5k text / 2k image / 1.5k
  audio), the AWQ-evolve generation count (50), the
  output bit-width (3.5), the .mlmodelc output path.
- **1:30-3:00**: Calibrate. The C++ imatrix runner runs.
  The view shows the per-modality imatrix v2 breakdown
  + the modality_scales.
- **3:00-5:00**: Evolve. The GA runner runs. The user
  watches the fitness chart climb; the top 5 candidates
  are visible.
- **5:00-5:30**: Quantize. The C++ quantize step runs
  (Tessera-as-default, no subcommand, per
  `docs/c++-port-design.md:417-493`).
- **5:30-6:00**: Convert. The C++ `tessera-to-coreml` step
  runs (per `docs/tessera-coreml-conversion-design.md`
  section 4). The .mlmodelc is written.
- **6:00-6:30**: Ship. The .mlmodelc is dropped in the
  iCloud Drive `TesseraStudio` directory. The user opens
  the iPhone, the .mlmodelc is in the ModelStore.

The 6.5-minute Mac demo is the "from corpus to device"
story.

### 8.4 The "wow" moments (recap)

- **First-token latency** (Metal vs CoreML, on a 12B on
  a phone, ~3x gap on prefill, ~2x on decode).
- **Sustained battery** (30-min flight test, mWh/token
  headline, ~30-40% gap Tessera 4-bit vs stock Q4_0).
- **Telemetry transparency** (IOReport visible to the
  user, not hidden).
- **Reasoning transparency** (the `Thinking for Ns`
  block, collapsed by default, expandable to see the
  chain-of-thought; the user can audit the model's
  reasoning before trusting the answer).
- **Rich content** (markdown + code highlighting +
  Mermaid diagrams + HTML preview; the chat surface is
  not a plain-text terminal).
- **Web search grounded answers** (Tavily-backed
  citations inline, no hallucination on "what is
  current?").
- **One-tap model store** (App Store IAP for the
  .mlmodelc, no manual download, no iCloud Drive
  dance).

---

## 9. Test path

This section documents the verification path, mirroring
the test path sections of the C++ port design
(`docs/c++-port-design.md:980-1057` risk register +
`docs/tessera-coreml-conversion-design.md:1280-1371`
section 7).

### 9.1 Mac test: engine on stock Q4_0 GGUFs

First verification: the engine works on stock Q4_0 GGUFs
before we ship any Tessera-specific code. The Mac app
loads a stock Q4_0 Gemma 3 4B from the bundled
`models/` directory, runs a 50-token generation, asserts:

- The text output is non-empty.
- The first-token latency is < 500ms.
- The IOReport subscriber fires (ANE power, GPU power
  both > 0).
- The L1 sidecar file is created in the sidecar
  directory.

This is the "did the xcframework build and link" test. It
runs in `TesseraCoreTests/testStockQ4_0Engine` and is the
gate for Phase 1 of the implementation plan.

### 9.2 Mac test: calibrate, evolve, quantize, convert

The second verification: the C++ pipeline produces a
working .mlmodelc. The Mac app runs the five-step
pipeline on TinyLlama 1.1B (a small model that fits
in the test fixture budget):

- Calibrate on a 1k subset of the corpus (text-only for
  TinyLlama).
- Evolve for 10 generations (small budget).
- Quantize with the resulting policy.
- Convert to .mlmodelc.

The test asserts:

- The .mlmodelc is a valid CoreML model (it loads in
  `MLModel(contentsOf:)`).
- The .mlmodelc produces non-empty output on a
  held-out 10-token generation.
- The L1 / L1.5 sidecar files are created.
- The IOReport shows ANE power > 0 during the
  inference.

This is the "did the C++ pipeline + the Swift wrapper
integrate" test. It runs in
`TesseraCoreTests/testFullPipelineTinyLlama` and is the
gate for Phase 4.

### 9.3 iPhone test: install the .app on a real device

The third verification: the iPhone app loads the
.mlmodelc and produces output. The user runs the
`build-tessera-xcframework.sh` on the Mac, opens the
iOS app in Xcode, builds, deploys to a real M-series
iPhone (iPhone 15 Pro or later), and runs a 50-token
generation.

The test asserts:

- The .mlmodelc is in the `.app` bundle
  (`Frameworks/tessera_ane.mlmodelc`).
- The first-token latency is < 300ms on a 12B.
- The IOReport shows ANE power > 0, GPU power < 0.2W.
- The L1 sidecar file is created in the app's
  Documents directory.

This is the "did the iOS embed + the on-device runtime
work" test. It is manual, not automated (Xcode +
device + real measurement).

### 9.4 The 30-minute flight test

The fourth verification: the hero metric is real. The
user runs the 30-minute flight test on a M-series
iPhone on battery, captures the JSON export, and runs
the Mac ABReplayView against the JSON. The replay
asserts:

- `battery_delta_mWh > 0`.
- `mWh_per_token` is in the expected range
  (TBD; will be measured on the first 3-5 flight tests).
- The thermal events count is < 5.
- The L1 sidecar files are present and have
  per-modality rows.

This is the "is the hero metric real" test. It is
manual, runs on a real device, and is the gate for the
App Store submission.

### 9.5 Test infrastructure

- The Mac tests run via `swift test` (SwiftPM). The
  `Package.swift` declares a `TesseraCoreTests` target.
- The iOS tests run via Xcode's XCTest. The iOS app
  target has a `TesseraStudioiOSTests` target.
- The on-device flight test is manual; the JSON export
  is the artifact.
- The Mac Studio replay is automated; the
  `ABReplayView` reads the JSON and re-renders the
  surfaces.

### 9.6 New-feature tests (added 2026-07-30)

- **Web search test**: `TesseraWebSearchTests.testSearch`,
  using a real Tavily API key (or a fixture response).
  Asserts the search returns <= 5 results, each with a
  `title` + `url` + `content` + `score` field. The
  `TesseraChatWebSearchTests.testFoldPrompt` asserts
  that the prompt-folding produces a context block
  with the right format.
- **Reasoning test**: `TesseraEngineTests.testReasoning`,
  using a small reasoning-capable model (a 1B
  distilled). Asserts that the `reasoningToken` field
  is populated, that the CoT arrives before the
  answer, and that the duration is recorded.
  `TesseraChatReasoningTests.testThinkingBlockRenders`
  asserts that the `ThinkingBlock` view collapses by
  default and expands on tap.
- **Rich renderer test**: `TesseraRichRendererTests
  .testMarkdownStreaming`, asserting the parser
  re-parses on each token, that the pending block is
  visible, and that the complete block is promoted on
  closing delimiter. `testCodeHighlighting` asserts
  Python syntax highlighting. `testMermaid` asserts
  a `flowchart TD; A-->B; B-->C;` block renders to an
  SVG.
- **Chat history drawer test**:
  `TesseraChatHistoryTests.testRecencyBucketing`,
  asserting that messages from "today" go in the
  Today bucket, messages from "2 days ago" go in the
  2-days-ago bucket, etc.
- **IAP / StoreKit test**: `TesseraStoreKitTests
  .testIAPCatalog` uses a `.storekit` configuration
  file with the 3 products, asserts the catalog loads,
  the purchase flow runs, the receipt is validated,
  and the "Restore Purchases" button restores the
  entitlements. The `.storekit` file is checked into
  the repo at `Tests/Fixtures/Products.storekit`.
- **Settings test**: `TesseraSettingsTests.testDefaults`,
  asserting the default values for every toggle
  match the section 5.8 spec. `testPrivacyManifest`
  asserts the `PrivacyInfo.xcprivacy` file is in
  every binary target.
- **AI disclosure test**:
  `TesseraStudioiOSTests.testAIDisclosure`, asserting
  the `OnboardingView` shows the EU AI Act Article
  50 disclosure before the first inference. The
  App Store review test: the disclosure must be
  visible without scrolling.
- **Privacy manifest test**: a CI script runs `xcrun
  privacy-manifest-tool validate` on every
  `PrivacyInfo.xcprivacy` in the repo. The script
  fails the build if any required reason is
  missing.

---

## 10. Phased implementation plan

This section documents the implementation phasing. The
phases match the dependency graph: Phase 1 is the
critical path (engine + xcframework + C FFI), Phases 2-3
are parallel (Swift Package skeleton + CoreML backend
+ IOReport), Phases 4-5 are sequential (Studio surface
+ A/B compare), Phase 6 is polish.

### Phase 1 (~3 weeks): The engine integration

The critical path. The Mac and iOS apps both depend on
this.

- 1.1: `build-tessera-xcframework.sh` (~600 LoC, fork
  of `/Users/user/Developer/GitHub/llama.cpp/build-
  xcframework.sh:1-550`).
- 1.2: The C FFI additions
  (`tessera_modality_t`, `tessera_set_dequant_mode`,
  `tessera_read_l1_sidecar`, `tessera_read_l15_sidecar`,
  `tessera_set_imatrix`, `tessera_set_policy`) - ~400
  LoC of C.
- 1.3: `Sources/TesseraCore/LibTessera.swift` (~400
  LoC, port of the llama.swiftui LibLlama.swift pattern).
- 1.4: The TesseraEngine class (~150 LoC inside
  LibTessera.swift).
- 1.5: The TesseraIOReporter Objective-C bridge
  (~120 LoC).
- 1.6: `Package.swift` (~50 LoC).
- 1.7: The stock Q4_0 engine test
  (`TesseraCoreTests/testStockQ4_0Engine`).

Gate: stock Q4_0 inference works on Mac + iOS.

### Phase 2 (~3 weeks): The Swift Package skeleton

The chat surface on iOS, the package layout, the
adapter protocol.

- 2.1: `Sources/TesseraCore/InferenceAdapter.swift`
  (~200 LoC).
- 2.2: `Sources/TesseraCore/ModelStore.swift` (~200
  LoC, port + Tessera flavour).
- 2.3: `Sources/TesseraCore/ConversationStore.swift`
  (~200 LoC).
- 2.4: `Sources/TesseraStudioiOS/TesseraStudioiOSApp
  .swift` (~50 LoC).
- 2.5: `Sources/TesseraStudioiOS/ContentView.swift`
  (~120 LoC, tab bar).
- 2.6: `Sources/TesseraStudioiOS/ChatView.swift` (~480
  LoC, port of PrismChatView + LlamaState).
- 2.7: `Sources/TesseraStudioiOS/ModelStoreDrawer
  .swift` (~150 LoC).
- 2.8: `Sources/TesseraStudioMac/TesseraStudioMacApp
  .swift` (~50 LoC, minimal: empty Studio placeholder).
- 2.9: `Sources/TesseraStudioMac/ContentView.swift`
  (~150 LoC, window menu).

Gate: chat works on iOS, Mac app launches.

### Phase 3 (~3 weeks): The CoreML backend + IOReport

The CoreML backend, the IOReport telemetry, the L1 /
L1.5 sidecar writer.

- 3.1: `Sources/TesseraCore/TesseraCoreMLBackend.swift`
  (~150 LoC).
- 3.2: The C-side `ggml-coreml` integration (G7 of
  the C++ port, out of scope for this doc).
- 3.3: `Sources/TesseraStudioiOS/TelemetryDrawer
  .swift` (~120 LoC, the five gauges).
- 3.4: The IOReport v3 sidecar extension (C-side,
  out of scope for this doc; Swift side reads).
- 3.5: The flight test runner (~80 LoC in
  ChatView.swift, the 30-minute timer + the
  automated prompt stream).

Gate: IOReport visible in chat, flight test runs end
to end on iOS.

### Phase 4 (~3 weeks): The Studio surface on Mac

The Mac Studio, the calibration runner, the GA runner,
the policy editor, the sidecar viewer.

- 4.1: `Sources/TesseraCore/CalibrationSession.swift`
  (~200 LoC).
- 4.2: `Sources/TesseraCore/QuantizationPlan.swift`
  (~150 LoC).
- 4.3: `Sources/TesseraCore/TelemetryObserver.swift`
  (~200 LoC, the L1 + L1.5 + B + C + E consumer).
- 4.4: `Sources/TesseraStudioMac/StudioView.swift`
  (~400 LoC).
- 4.5: `Sources/TesseraStudioMac/QuantizationPlanEditor
  .swift` (~280 LoC).
- 4.6: `Sources/TesseraStudioMac/CalibrationSessionView
  .swift` (~320 LoC).
- 4.7: `Sources/TesseraStudioMac/SidecarViewer.swift`
  (~250 LoC).
- 4.8: The Mac pipeline test
  (`TesseraCoreTests/testFullPipelineTinyLlama`).

Gate: full pipeline runs on Mac, .mlmodelc ships to
iOS.

### Phase 5 (~2 weeks): The A/B compare view + flight test polish

The A/B compare view, the 30-minute flight test
polish, the export, the Mac ABReplayView.

- 5.1: `Sources/TesseraStudioiOS/ABCompareView.swift`
  (~280 LoC).
- 5.2: `Sources/TesseraStudioMac/ABReplayView.swift`
  (~250 LoC).
- 5.3: The flight test summary card (~50 LoC in
  ChatView.swift).
- 5.4: The flight test JSON export (~50 LoC).
- 5.5: The iCloud Drive sync (~50 LoC of
  file-presentation code).

Gate: A/B compare works on iOS, flight test exports
to JSON, Mac replay works.

### Phase 6 (~2 weeks): Reasoning, web search, rich renderers

The new architect-directed features.

- 6.1: `Sources/TesseraCore/TesseraWebSearch.swift`
  (~200 LoC). The Tavily client + the prompt-folding
  + the `TesseraChatWebSearchChip` SwiftUI view.
- 6.2: The reasoning C FFI additions
  (`tessera_set_reasoning_mode`,
  `tessera_read_reasoning_token`,
  `tessera_last_reasoning_duration`).
- 6.3: `Sources/TesseraCore/TesseraChatHistory.swift`
  (~200 LoC). The recency-bucketing utility.
- 6.4: `Sources/TesseraStudioiOS/ChatHistoryDrawer
  .swift` (~180 LoC). The `NavigationSplitView`
  sidebar.
- 6.5: `Sources/TesseraStudioiOS/ThinkingBlock
  .swift` (~80 LoC). The collapsible CoT block.
- 6.6: `Sources/TesseraCore/TesseraRichRenderer
  .swift` (~300 LoC). The markdown parser +
  code highlighter + Mermaid renderer + HTML
  preview host.
- 6.7: `Sources/TesseraStudioiOS/MarkdownView
  .swift` (~250 LoC) + `MermaidView.swift` (~250
  LoC). The SwiftUI views.
- 6.8: `TesseraWebSearchTests`, `TesseraChatReasoning
  Tests`, `TesseraRichRendererTests`,
  `TesseraChatHistoryTests` (section 9.6).

Gate: web search works, reasoning block collapses
by default, chat history drawer opens with the
recency bucketing, rich renderers display markdown
+ code + Mermaid.

### Phase 7 (~2 weeks): Settings, IAP, App Store metadata

The publishability surface.

- 7.1: `Sources/TesseraCore/TesseraSettings.swift`
  (~150 LoC). The `@Observable` settings model.
- 7.2: `Sources/TesseraStudioiOS/SettingsView.swift`
  (~200 LoC) and `Sources/TesseraStudioMac/
  SettingsView.swift` (~200 LoC). The settings
  surface.
- 7.3: `Sources/TesseraStudioiOS/OnboardingView
  .swift` (~120 LoC). The first-run flow with the
  AI disclosure.
- 7.4: `Sources/TesseraCore/TesseraStoreKit.swift`
  (~250 LoC). The StoreKit 2 wrapper + the IAP
  catalog.
- 7.5: The `.storekit` configuration file (the
  TestFlight product catalog).
- 7.6: The Apple-hosted background asset pack (the
  large .mlmodelc bundled in the Background Assets
  framework).
- 7.7: The `PrivacyInfo.xcprivacy` files for the
  app + every xcframework.
- 7.8: The App Store Connect metadata: the privacy
  label, the App Privacy questionnaire, the AI
  disclosure, the privacy policy link, the support
  URL, the IAP catalog.

Gate: settings surface is reachable from both
platforms, IAP catalog loads, AI disclosure is
visible in the first-run, privacy manifest is
validated, App Store Connect metadata is ready for
submission.

### Phase 8 (~2 weeks): Polish, App Store compliance, beta

- 8.1: App Store compliance: the IOReport subscriber
  uses private APIs; the v1 ships via TestFlight only.
  The v2 (with public-API fallback) is documented.
- 8.2: Accessibility (VoiceOver labels, Dynamic Type,
  Reduced Motion).
- 8.3: Localisation (English v1; the strings are
  externalised in `Localizable.strings`).
- 8.4: Beta testing (TestFlight, 50 users).
- 8.5: App Store submission (v1, TestFlight track).

Gate: TestFlight build is approved, 50 beta users
have run the 30-minute flight test, the hero metric
is on the marketing page, the App Store review
passes.

### Total

~22 weeks wall clock with 1 dev. ~10 weeks with 4
parallel agents after Phase 1. The 6-week delta
from the previous revision is Phase 6 (reasoning +
web search + rich renderers) and Phase 7 (settings
+ IAP + App Store metadata).

Phase 1 is the critical path; Phases 2-3 can run as
2-3 parallel agents (chat, CoreML backend, IOReport
are independent). Phase 4 is Mac-only and can run
in parallel with Phase 5's iOS work. Phase 6 is
iOS-heavy (the chat + the chat history drawer) and
can run in parallel with the iOS-side Phase 7 work.
Phase 8 is sequential.

---

## 11. Open design questions (with lean recommendations)

The following are open at scoping time. Each is
documented with a lean recommendation; the architect
is expected to lock or push back.

### 11.1 The app's bundle identifier

Question: `com.butterbase.tessera-studio`? `com.tessera-
project.studio`? `ai.tessera.studio`?

Lean: `com.butterbase.tessera-studio`. Matches the
PrismAgent namespace (`com.butterbase.*` is what
PrismAgent uses for its `PrismEngineController` and
the Butterbase SDK). Avoids claiming a namespace
the project does not own.

### 11.2 Mac Catalyst vs Mac-native for the Mac target

Question: Mac Catalyst (one codebase, both iPad and
Mac) or Mac-native (two codebases, idiomatic UI on
each)?

Lean: Mac-native. The Mac Studio surface (the
calibration runner, the GA visualisation, the policy
editor) is fundamentally different from the iOS chat
surface. Catalyst is a compromise that costs more in
the long run; native SwiftUI is the right answer. The
shared `TesseraCore` library makes the duplication of
the app entry + the tab bar / window menu trivial.

### 11.3 iOS app on iPad: universal or iPhone-only?

Question: Universal (iPhone + iPad) or iPhone-only
(Mac Catalyst for iPad)?

Lean: Universal. The chat surface scales well (the
`MessageBubble` and the input bar adapt to the iPad
keyboard with a few `horizontalSizeClass` checks). The
A/B compare view benefits from the larger screen. The
iPad does not need a separate target.

### 11.4 On-device model download

Question: App Store CDN (Apple hosts the .mlmodelc) or
Mac companion (user transfers via Finder / iCloud)?

Lean: Mac companion exports the .mlmodelc, user
transfers via iCloud Drive. Reasons:

- The .mlmodelc is 3-4 GB for a 12B; the App Store CDN
  has a 4 GB limit per download, and Apple does not
  support resumable downloads for assets that large.
- The .mlmodelc is generated per-model per-policy; the
  CDN model assumes a stable set of models. The Mac
  companion can generate a new .mlmodelc every time
  the user changes the policy.
- The user is already on the Mac for calibration; the
  iCloud handoff is a one-tap operation.

### 11.5 The 30-minute flight test: live in-app or QA harness

Question: Live in the iPhone app (visible to the user)
or a separate QA harness (Xcode UI test)?

Lean: Live in the iPhone app. The 30-minute flight test
is the hero metric; the user needs to see it. A QA
harness hides the metric.

### 11.6 The Plan's persistence

Question: JSON file in iCloud Drive (sync across
devices) or JSON file in the app's Documents directory
(per-device, no sync)?

Lean: JSON file in the app's Documents directory; the
.mlmodelc and the flight test JSON are in iCloud Drive.
The Plan is a working document on the Mac; the iOS app
reads a copy of the Plan from iCloud Drive for display
only. The iOS app does not edit Plans.

### 11.7 The IOReport telemetry display

Question: Always visible (persistent in the chat
surface) or on-demand (collapsed drawer, expanded on
tap)?

Lean: Collapsed by default, expandable on tap. The
chat surface is the primary surface; the IOReport is
the secondary surface. A persistent display would
clutter the chat. A collapsed drawer keeps the chat
clean and surfaces the metric on demand.

### 11.8 App Store compliance

Question: IOReport is a private API. The v1 ships via
TestFlight; the v2 uses public APIs only (or a
different telemetry source).

Lean: TestFlight v1, App Store v2 with public API
fallback. The public API fallback is
`ProcessInfo.thermalState` (thermal) +
`ProcessInfo.processInfo.systemUptime` (uptime) +
`os_signpost` (in-process power, public on iOS 13+)
+ a battery-drain heuristic (start battery level,
end battery level, elapsed time, tokens generated).
The App Store v2 has worse fidelity than the
TestFlight v1, but it is the right answer for the
store.

### 11.9 Pre-bundled models

Question: Which models ship in the `.app` bundle for
v1?

Lean: Two: (a) Tessera 4-bit Gemma 3 4B (text-only,
~2 GB), (b) Tessera 3.5-bit Gemma 4 12B Unified
(text + image + audio, ~3.5 GB). The user can add
more via the Mac companion. The stock Q4_0 model
(for A/B) is downloaded on first use, not bundled.

### 11.10 The iCloud Drive handoff format

Question: Is the .mlmodelc a single file in iCloud
Drive, or a directory (with the sidecars)?

Lean: Directory. The .mlmodelc is a directory in
CoreML; the sidecars (L1, L1.5, policy, imatrix) sit
next to it in the same directory. The iPhone app
imports the whole directory; the Mac app writes the
whole directory. The directory name is the model ID.

### 11.11 The Mac Studio's text-only mode

Question: Can the Mac Studio run inference (not just
calibration)?

Lean: Yes. The Mac Studio has a "Run" tab (the same
chat surface as iOS, scaled up) for testing the
model locally before shipping to the iPhone. The
Studio's primary surface is calibration; the "Run"
tab is secondary.

### 11.12 The Swift 6.2 strict concurrency posture

Question: The existing llama.swiftui uses non-strict
concurrency; should the Tessera Studio use strict
(Swift 6.2 default) or non-strict?

Lean: Strict. The PrismAgent `Package.swift:1` is
already `swift-tools-version: 6.2`; the Tessera
Studio inherits that. The `actor TesseraContext`
pattern in `LibLlama.swift:24` translates cleanly to
strict concurrency.

### 11.13 Web search provider

Question: Tavily only, or Tavily + Google CSE fallback,
or Brave Search, or self-hosted (SearXNG)?

Resolved (2026-07-31): provider-shaped, keyless default. DuckDuckGo's
static HTML endpoint (URLSession + SwiftSoup, no key) is the default;
self-hosted SearXNG is an opt-in for users who want the query to stay on
their own infrastructure; Tavily is kept as an opt-in vendor provider
behind `tessera.settings.tavilyAPIKey`. This supersedes the "Tavily only
for v1" lean and brings SearXNG into scope - the no-Docker, run-from-source
path answers the earlier operational-overhead objection. Implemented in
`TesseraSearchProvider.swift`, `TesseraDuckDuckGoSearch.swift`,
`TesseraSearXNGSearch.swift`, and `TesseraTavilySearch.swift`; see section 5.4.

### 11.14 Reasoning model UI defaults

Question: Show the `ThinkingBlock` collapsed by
default, or expanded, or both (collapsed on first
arrival, auto-expand on user tap, stay expanded
until the user collapses it)?

Lean: Collapsed by default, with a `Show reasoning
by default` toggle in Settings (default OFF). The
Claude Code pattern is "most people don't look at
it"; the Locara pattern is "default collapsed -
opening is the user's deliberate choice." The
Tessera port follows Locara.

### 11.15 IAP pricing tier

Question: $4.99 / $9.99 / $19.99, or $2.99 / $7.99 /
$14.99, or other?

Lean: $4.99 / $9.99 / $19.99 (App Store tier 4 / tier
6 / tier 9). The 30% Apple cut is acceptable for v1
(we are not in the Small Business Program yet);
the $9.99 medium tier covers the Gemma 4 12B
Tessera 3.5-bit .mlmodelc which is the "show your
work" model. The pricing is a product decision; the
architect may push back.

### 11.16 IAP delivery mechanism

Question: Apple-hosted non-consumable IAP (deprecated
in April 2022), On-Demand Resources (limited to the
app bundle), Background Assets (Apple-hosted, iOS
26+), or self-hosted CDN with receipt validation?

Lean: Background Assets (`BackgroundAssets.framework`,
iOS 26+). Apple-hosted, up to 200 GB per app,
included in the Developer Program membership, no
`SKDownload` deprecation risk. The 12B .mlmodelc is
~3.5 GB; the 27B v2 is ~10 GB; both fit. The
fallback for iOS 18-25 users is On-Demand Resources
in the same build (or a "minimum iOS 26" cutoff, the
lean).

### 11.17 Settings: pre-bundled vs IAP-only

Question: Pre-bundle 3 .mlmodelcs in the `.app` (free,
in the App Store binary) AND offer IAP for additional
.mlmodelcs, or IAP-only (the user buys all .mlmodelcs
including the 3 pre-bundled)?

Lean: Pre-bundle 3 + IAP for additional. The 3
pre-bundled models are the "show your work" set
(stock Q4_0 4B, Tessera 4-bit 4B, Tessera 3.5-bit
12B reasoning). The IAP catalog offers larger and
niche models (Gemma 4 27B v2, Llama 4 Scout v2,
multimodal specialists). The 30% Apple cut on the
IAP is the revenue model; the 3 pre-bundled models
are the on-ramp.

### 11.18 Chat history sync across devices

Question: Sync the chat history across the user's
iPhone + iPad + Mac via iCloud, or per-device only?

Lean: Per-device only. The chat history is in the
app's Documents directory; the .mlmodelc + the
plan are in iCloud Drive. The chat history is
local-only; the user can export a single
conversation as a markdown file from the
`...` menu if they want to share it. Sync
introduces conflict-resolution complexity
(two devices editing the same conversation
mid-stream) for negligible benefit.

---

## 12. Risk register

This section lists the risks, mirroring the C++ port
design's risk register
(`docs/c++-port-design.md:964-1113`).

### R1 - The CoreML backend depends on G7

The CoreML backend (`ggml-coreml`, G7 of the C++ port)
is the largest unknown. If G7 is delayed, the iOS app
cannot ship. Mitigations: the iOS app can ship with
the Metal backend (Tessera-quantized + Metal) in
parallel; the CoreML backend is a v1.1 upgrade. The
Metal backend is the fallback (C7) and is tested
independently.

### R2 - Multi-modal calibration depends on G0-MM-G4-MM

The Mac Studio surface depends on the multi-modal
calibration work (M1-M8, G0-MM through G4-MM in
`docs/multimodal-calibration-design.md:828-925`).
If the multi-modal work is delayed, the Mac Studio
ships with text-only calibration and a "multi-modal
coming soon" badge.

### R3 - IOReport is a private API on iOS

The App Store will reject a v1 that uses private
APIs. Mitigations: TestFlight v1 (no rejection
risk), App Store v2 with public API fallback
(section 11.8). The C-side IOReport subscriber is
isolated to `TesseraIOReporter.swift`; the fallback
is a drop-in replacement.

### R4 - The .mlmodelc conversion takes 30-120s for a 12B

The 30-120s conversion is a UX problem on the Mac:
the user clicks "Convert" and waits. Mitigations: the
conversion is offline (the user is on the Mac, can
do other things); the conversion runs in a background
task; the progress bar shows the per-layer progress.
The C2 decision (convert at quantize time) means the
iPhone user never waits.

### R5 - 12B on iPhone: ANE memory constraints

The gemma 4 12b at 3.5-bit effective is ~5.5 GB; the
ANE on a M-series iPhone has 6-8 GB of unified memory.
A 12B at 3-bit effective is ~4.7 GB; fits. A 12B at
4-bit effective is ~6 GB; tight. Mitigations: the
Tessera quantizer supports 3-bit and 3.5-bit
configurations; the iPhone app warns the user if
the chosen model is too large.

### R6 - PrismAgent has 197 Swift files; selective port

The PrismAgent port is selective (~1,920 LoC of the
~3,692 LoC we read ports cleanly, ~5,000 LoC is
skipped). Some patterns may not translate cleanly
(e.g. the multi-tier journal may be overkill for a
Tessera app that is not an agent). Mitigations: the
design above is opinionated about what to port and
what to skip; the open questions in section 11
flag the borderline cases.

### R7 - Swift 6.2 strict concurrency

The existing llama.swiftui uses non-strict
concurrency. The Tessera Studio is strict (the
`actor TesseraContext` pattern translates cleanly;
the LlamaState is `@MainActor` already at
`/Users/user/Developer/GitHub/llama.cpp/examples/
llama.swiftui/llama.swiftui/Models/LlamaState.swift:
11`). Mitigations: the Tessera Studio starts strict
from day 1; the C FFI is wrapped in a single
`@unchecked Sendable` actor if necessary.

### R8 - TestFlight / App Store review timeline

The TestFlight v1 review is 1-2 days; the App Store
v2 review is 1-7 days. The 16-week implementation
plan does not include review time. Mitigations: the
plan is 16 weeks of dev; the review is in addition.
The 50-user beta (Phase 6.4) is the buffer for
review delays.

### R9 - IOReport sampling rate vs Swift UI thread

The IOReport subscriber fires at 1 Hz; the Swift
side reads at 1 Hz. If the Swift UI is busy (a
streaming chat surface), the read can be delayed.
Mitigations: the IOReport subscriber is on a
background queue; the Swift side reads from a
lock-free ring buffer. The 1 Hz rate is the
floor; the 60 Hz (CADisplayLink) is the ceiling
for the visible gauges.

### R10 - The 9-component cluster vs CoreML's input switch

The C8 decision is 1 .mlmodelc with runtime
act_scale. The act_scale is a runtime input; the
CoreML framework routes the act_scale through the
graph. If the act_scale input is on the hot path,
the runtime input switch is a cost. Mitigations:
the 3-package v2 (C8 fallback) is gated on
profiling; if the act_scale input is hot, the
Mac Studio re-generates 3 .mlmodelc files and the
iPhone app picks based on modality. This is a
packaging optimization, not a v1 requirement.

### R11 - The PrismAgent PrismEngineC is not reused

The PrismAgent `PrismEngineC` and `PrismBridgeFFI`
targets (`Package.swift:89-108`) are Prism-specific
(the FFI is to a Rust core, not C). The Tessera
Studio does not reuse them; the FFI is to the
tessera xcframework (C). This is a known divergence;
the Tessera xcframework is a fork of the llama.cpp
xcframework, not a port of the PrismEngineC.

### R12 - The C++ port phasing (G0-G6) is on the critical path

The Tessera Studio depends on the C++ port
(`docs/c++-port-design.md:699-810`). G3 (TESSERA_*
writer) is the gate for the Mac quantize step; G5
(L1.5 + sidecar v3 producer) is the gate for the
Mac sidecar viewer. If G3 or G5 is delayed, the Mac
Studio ships with a "Tessera quantize coming soon"
badge and uses the Python `tools/tessera/`
quantize for v1.

### R13 - iCloud Drive directory import on iOS

The iPhone app imports a directory from iCloud Drive
(`.mlmodelc` + sidecars). iOS does not have a
"directory import" API the way the Mac does. The
iOS app reads the directory via
`FileManager.startDownloadingUbiquitousItem` for
each file in the directory. Mitigations: the
.mlmodelc + sidecars are listed in a JSON manifest
in the directory; the iOS app reads the manifest
first, then downloads each file. The progress is
shown in the ModelStore drawer.

### R14 - The iPhone "first inference" 240ms target

The 240ms first-token latency on a 12B on a phone
is an estimate, not a measurement. The actual
latency depends on the ANE warm-up, the prefill
length, the model's prefill shape, and the
Tessera kernel's efficiency. Mitigations: the
Mac ABReplayView is the offline simulator; the
iPhone measurement is the ground truth. The
5-minute demo script is updated once the first
flight tests land.

### R15 - The "30-minute flight test" is 30 minutes

The 30-minute flight test is long. The user has
to leave the app running for 30 minutes. The app
must handle background / lock / thermal
throttling during the 30 minutes. Mitigations:
the flight test is foreground-only (the iPhone
user keeps the app open); the IOReport
subscriber is foreground-only (C3); the
summary card is rendered at the end even if
the app was backgrounded (the v3 sidecar is
written continuously).

### R16 - Substantive risks remaining

R11 (PrismAgent PrismEngineC not reused), R13 (iCloud
directory import), R14 (240ms first-token target is
an estimate), R15 (30-minute flight test is long) are
documented in detail above. Lower-impact risks that
do not need a separate entry:

- R17: corpus licensing (CC0 + MIT + CC-BY-4.0,
  documented in `docs/multimodal-calibration-design.md:
  81-145`).
- R18: Mac Studio "Run" tab duplicates the iOS chat
  surface; the shared `TesseraCore` makes this a thin
  wrapper.
- R19: the Plan is a Swift-side convenience; the
  artifacts the C++ side reads are independent (the
  `policy.json` and `imatrix.v2.bin` from the C++ port
  design).
- R20: the flight test JSON is `version: 1`; future
  versions extend the v3 sidecar per
  `docs/tessera-coreml-conversion-design.md` section
  6.5.
- R21: the app name "Tessera Studio" vs "Tessera Chat"
  is a product decision; the design doc is agnostic.

### R22 - Apple's 30% IAP cut is significant

The .mlmodelc IAP prices have a 30% Apple cut. A
$9.99 medium model yields $6.99 net. The $19.99
large model yields $13.99 net. The 30% is the
standard commission; the Small Business Program
(15%) is available at < $1M annual revenue (not
applicable at launch). Mitigations: (a) keep the
Tessera quantize-side costs in the 2-person-team
range, (b) treat the IAP as a margin contributor
not a profit centre, (c) re-evaluate the
Small Business Program at the $1M threshold.

### R23 - Background Assets requires iOS 26+

`BackgroundAssets.framework` is iOS 26+. The current
plan targets iOS 18. Mitigations: (a) raise the
minimum to iOS 26 (cuts the addressable market by
~30%, the lean), (b) On-Demand Resources for
iOS 18-25 and Background Assets for iOS 26+
(the user is on a 2-track install, more complex),
(c) self-host the .mlmodelc on a CDN with receipt
validation (operational overhead).

### R24 - Reasoning model latency is 2-10x slower

The Gemma 4 12B Unified reasoning model emits CoT
before the answer. First-token latency for a
reasoning run is 2-10x slower than a non-reasoning
run. The chat surface surfaces the delay
(`time_to_first_reasoning_token` and
`time_to_first_answer_token` in the title sub-row);
the user is warned. The A/B Compare view shows the
delta. The flight test records both metrics. The
Settings "Reasoning mode = Auto" default is the
right answer for most users; the user can flip to
"Off" for non-reasoning runs.

### R25 - AI disclosure is required, the wording matters

iOS 26 App Review Guidelines require an in-app
disclosure at the point of display for AI-generated
content. The EU AI Act Article 50 (effective 2
August 2026) requires machine-readable marks and
explicit disclosure. The wording in the
`OnboardingView` and on every assistant message
("AI-generated response") must be approved by
counsel. The conservative interpretation is
"disclose in both the App Store description and a
visible in-app location" (per the
`appsops.store/news/week-in-app-store-ops-july-12-
2026` summary).

### R26 - Privacy manifest required, every binary

Apple's `PrivacyInfo.xcprivacy` is required for
every binary in the `.app` (the main app, every
xcframework, every extension). The required-reason
APIs that Tessera Studio uses:

- `NSPrivacyAccessedAPICategoryUserDefaults`
  (`CA92.1` for the app's own defaults; the
  Settings + the IAP catalog both use `UserDefaults`).
- `NSPrivacyAccessedAPICategorySystemBootTime`
  (`35F9.1` for the flight test duration
  measurement; the IOReport subscriber uses
  `ProcessInfo.processInfo.systemUptime`).
- `NSPrivacyAccessedAPICategoryDiskSpace` (`85F4.1`
  for the .mlmodelc download size check).
- `NSPrivacyAccessedAPICategoryFileTimestamp`
  (`C617.1` for the conversation journal).

A CI script runs `xcrun privacy-manifest-tool
validate` on every `PrivacyInfo.xcprivacy` in the
repo. The build fails if any required reason is
missing. The first-submission rejection rate for
apps missing the privacy manifest is non-trivial
(per `mobile.wednesday.is` 28% for AI apps); this
is a hard gate.

### R27 - The Mermaid + HTML renderer uses WKWebView

The Mermaid diagram and the HTML preview host their
content in a `WKWebView`. The webview is sandboxed
(no network, no JS for the HTML preview; JS for the
Mermaid renderer). The user can paste arbitrary
HTML; the webview is a sandboxed renderer, not a
general-purpose browser. Mitigations: (a) the
webview is `WKWebView` not `SFSafariViewController`,
(b) the HTML preview sets
`WKWebpagePreferences.allowsContentJavaScript =
false`, (c) the Mermaid renderer bundles KaTeX
(when v2) or a Swift-native renderer (v1).

---

## 13. App Store publishability (added 2026-07-30)

This section consolidates the App Store-specific
deliverables: the privacy manifest, the App Store
Connect metadata, the IAP / Apple-hosted background
asset configuration, the AI disclosure, the EU AI
Act compliance, the App Privacy questionnaire.

### 13.1 Privacy manifest (`PrivacyInfo.xcprivacy`)

Every binary in the `.app` ships a
`PrivacyInfo.xcprivacy` file (per Apple's
`developer.apple.com/documentation/bundleresources
/describing-use-of-required-reason-api`). The
required-reason APIs Tessera Studio uses, with the
approved reason codes:

- **`NSPrivacyAccessedAPICategoryUserDefaults`**
  (`CA92.1`): the app's own defaults. The Settings
  surface uses `@AppStorage` for the simple toggles.
- **`NSPrivacyAccessedAPICategorySystemBootTime`**
  (`35F9.1`): elapsed-time measurement. The flight
  test uses `ProcessInfo.processInfo.systemUptime`
  for the 30-minute timer; the IOReport subscriber
  uses it for the telemetry window.
- **`NSPrivacyAccessedAPICategoryDiskSpace`**
  (`85F4.1`): storage check before the .mlmodelc
  download (the user needs >= 5 GB free for the 12B
  .mlmodelc).
- **`NSPrivacyAccessedAPICategoryFileTimestamp`**
  (`C617.1`): the conversation journal uses file
  creation + modification dates for the recency
  bucketing.

The manifest lives in every binary target
(`Sources/TesseraStudioiOS/PrivacyInfo.xcprivacy`
+ the iOS app + the iOS extension + every
xcframework's privacy manifest). A CI script runs
`xcrun privacy-manifest-tool validate` on every
manifest in the repo; the build fails if any
required reason is missing.

### 13.2 App Privacy questionnaire

The App Store Connect App Privacy questionnaire
must match the `PrivacyInfo.xcprivacy` exactly.
The questionnaire is filled out per the
`appsops.store/news/week-in-app-store-ops-july-12-
2026` guidance:

- **Data collection**: the app does not collect
  data from the user unless they opt in. The
  default for "Share analytics" is OFF. The
  default for "Send prompts to web search
  provider" is ON, with a clear in-app disclosure
  at first-run.
- **Data linked to user**: the conversation
  journal is local to the device; not synced, not
  shared, not collected. The flight test JSON
  export is local. The .mlmodelc is local.
- **Tracking**: the app does not track. The
  `NSPrivacyTracking` is `false`. The
  `NSPrivacyTrackingDomains` is empty.
- **Data types collected**: `Identifiers` (the
  per-conversation UUID, the per-session UUID, not
  linked to user identity) and `Usage Data` (the
  flight test JSON, the v3 sidecar, opt-in) and
  `Diagnostics` (crash reports, opt-in).

### 13.3 AI disclosure (iOS 26 + EU AI Act Article 50)

The iOS 26 App Review Guidelines (Guideline 2.1,
AI disclosure-forward enforcement) require an
in-app disclosure at the point of display for
AI-generated content. The EU AI Act Article 50
(effective 2 August 2026) requires machine-readable
marks and explicit disclosure for AI systems that
interact with individuals.

The Tessera Studio compliance:

- **First-run disclosure** (`OnboardingView`): the
  "AI disclosure" step in section 5.10 shows the
  EU AI Act + iOS 26 disclosure before the first
  inference. The wording: "Tessera Studio uses
  on-device AI to generate responses. All
  inference is local; no data leaves your device
  unless you enable web search."
- **Per-message disclosure** (Settings toggle,
  default ON): every assistant message has an
  "AI-generated response" badge in the footer.
- **Settings disclosure** (About section): the
  "About" screen in Settings has a "Tessera uses
  on-device AI" link to the disclosure text.
- **App Store description**: the description has
  the same disclosure in the first paragraph.
- **Reviewer notes**: the App Store Connect review
  notes include the disclosure text + the
  in-app locations + the per-message badge
  location + the test account credentials.

The conservative interpretation is
"disclose in both the App Store description and a
visible in-app location" (per the
`appsops.store/news` summary). The 28% first-
submission rejection rate for AI apps without
disclosure is the risk; the disclosure is the
fix.

### 13.4 IAP configuration (App Store Connect)

The IAP catalog is configured in App Store
Connect > Monetization > In-App Purchases:

- `tessera.gemma3.4b.q4.tessera.v1` (Small, $4.99,
  non-consumable).
- `tessera.gemma4.12b.unified.q35.tessera.v1`
  (Medium, $9.99, non-consumable).
- `tessera.gemma4.27b.unified.q3.tessera.v1`
  (Large, $19.99, non-consumable, v2).

Each product is non-consumable (the user buys
once, keeps forever, restores on new device).
The Apple-hosted background asset pack is
attached to each product via the App Store
Connect > In-App Purchases > Hosting section
(deprecated path) or the Xcode 27+
`ba-package` command for Background Assets
(iOS 26+, the lean path).

The `Restore Purchases` button in Settings uses
`Transaction.currentEntitlements` (StoreKit 2) to
restore. The receipt validation is on-device (no
server).

The `.storekit` configuration file is at
`Tests/Fixtures/Products.storekit`; the iOS
scheme uses it for the sandbox test. The
synced `.storekit` file (linked to App Store
Connect) is at
`Tests/Fixtures/SyncedProducts.storekit`.

### 13.5 Privacy policy + terms of service

The privacy policy is hosted at
`https://tessera.studio/privacy` (placeholder, the
real URL is set before App Store submission). The
policy covers:

- What data the app collects (none by default;
  opt-in for analytics + web search).
- How the data is processed (on-device; no cloud
  upload unless the user enables web search).
- The web search provider (Tavily), with the
  privacy policy link.
- The Apple-hosted background asset distribution
  (Apple's privacy policy applies).
- The EU AI Act Article 50 disclosure.
- The data retention (the conversation journal is
  local; the user can delete it from the Settings
  > Model > "Delete all model files" + "Delete all
  conversations").
- The data deletion request process
  (mailto: privacy@tessera.studio).

The terms of service is at
`https://tessera.studio/terms` (placeholder). The
App Store Connect metadata includes both URLs in
the App Privacy section.

### 13.6 TestFlight distribution

The TestFlight build is the v1 distribution
channel. The flow:

1. The TestFlight build is uploaded via
   `xcodebuild -exportArchive` + the App Store
   Connect API.
2. The internal TestFlight group (50 users) is
   invited via email.
3. The 50 users run the 30-minute flight test on
   a M-series iPhone.
4. The 50 users fill out a 5-question survey
   (latency perception, battery perception,
   reasoning quality, web search quality, "would
   you pay for the Medium model").
5. The survey results are aggregated; the v1
   App Store submission is gated on >= 4.0/5.0
   average + >= 80% "would pay".

The TestFlight build includes the IOReport
subscriber (private API). The App Store build
uses the public-API fallback (section 1.1 of the
CoreML design). The two are separate targets with
separate build settings.

---

## 14. Agent loop + tool calling (v1, not v2)

The architect's 2026-07-30 reversal on the agent's
"SKIP for v1, EVALUATE for v2" lean: the agent loop
is part of the v1 spec. Tessera Studio is a full
agent app for macOS and iOS, on par with the most
mature references (Agent!, Foundation Lab, PrismAgent,
open-agent-sdk-swift).

Rationale: the calibration pipeline is linear, but
the agent loop adds value beyond automation. The user
sees a chat surface that can invoke Tessera tools
(calibrate, quantize, evaluate, export), inspect
tool call receipts, and undo destructive operations.
The 5-tool MVP becomes an 8-tool v1; the rest of the
agent loop is the standard typed-protocol + approval
engine + tool message UI pattern, all of which are
well-trodden in the references.

This section extends the v1 spec; the prior sections
that described the agent loop as v2 are superseded.

### 14.1 The tool protocol (TesseraTool)

A typed protocol with Codable + JSON Schema. The
PrismAgent `ToolTypes.swift` and `open-agent-sdk-swift`
architecture doc are the two references; Tessera
inherits PrismAgent's shape and adds the `@Generable`
pattern from Foundation Lab for the calibration config
inputs.

```swift
public protocol TesseraTool: Identifiable, Codable,
    Sendable {
    associatedtype Input: Codable & Sendable
    associatedtype Output: Codable & Sendable

    var id: String { get }
    var name: String { get }
    var description: String { get }
    var iconName: String { get }  // banner icon
    var inputSchema: JSONSchema { get }
    var outputSchema: JSONSchema { get }
    var approvalLevel: ApprovalLevel { get }

    func run(_ input: Input) async throws -> Output
}

public enum ApprovalLevel: String, Codable, Sendable {
    case auto    // safe, no confirmation
    case notify  // run + log, no prompt
    case prompt  // require user confirmation
    case denied  // not exposed in v1
}
```

LoC: ~150 Swift.

### 14.2 The tool registry (TesseraToolRegistry)

The catalog of available tools, indexed by id. The
registry is `@MainActor` and observable so the chat
can re-render the available tool list. The catalog
ships with the 8 v1 tools (see 14.4); new tools
register at app launch.

```swift
@MainActor
@Observable
public final class TesseraToolRegistry {
    public private(set) var tools: [String: any TesseraTool] = [:]

    public func register<T: TesseraTool>(_ tool: T) { ... }
    public func tool(id: String) -> (any TesseraTool)? { ... }
    public func toolsForApprovalLevel(_ level: ApprovalLevel)
        -> [any TesseraTool] { ... }
}
```

LoC: ~80 Swift.

### 14.3 The agent loop (TesseraAgentLoop)

The core streaming loop: receive user message, decide
which tools to call, stream tool invocations and
results, render tool messages in the chat, repeat
until the model stops. The agent's lean is the
AsyncStream actor pattern from Foundation Lab, which
combines cleanly with the existing `TesseraEngine`
streaming.

```swift
@MainActor
@Observable
public final class TesseraAgentLoop {
    public enum State: Equatable {
        case idle
        case planning
        case awaitingApproval(toolId: String, input: Data)
        case executing(toolId: String)
        case streaming
        case error(String)
    }

    public private(set) var state: State = .idle
    public private(set) var pendingToolCalls: [ToolCall] = []
    public private(set) var conversationLog: [ConversationTurn] = []

    public func send(_ userMessage: String) async { ... }
    public func approvePendingToolCall() async { ... }
    public func denyPendingToolCall(reason: String) async { ... }
    public func cancel() async { ... }
}
```

The loop is cancellation-aware (every `await` checks
`Task.isCancelled`). The state is observable; the
chat re-renders on every state change. The conversation
log is persisted via SwiftData (pattern from Agent!).

LoC: ~280 Swift.

### 14.4 The 8 v1 tools

| # | Name | Description | Approval | LoC |
|---|---|---|---|---|
| 1 | `list_models` | List the available Tessera-quantized + stock models in the ModelStore | `auto` | 30 |
| 2 | `load_model` | Load a specific model into the TesseraEngine | `auto` | 40 |
| 3 | `inspect_sidecar` | Read the v3 sidecar (L1 + L1.5 + per-row meta) for a model, return a JSON report | `auto` | 60 |
| 4 | `calibrate` | Run the 5k multi-modal calibration corpus, produce a modality-tagged imatrix v2 | `prompt` (destructive: 5-30 min on the GPU) | 180 |
| 5 | `evolve` | Run AWQ-evolve on the imatrix, produce a TesseraPolicy (modality_scales + AWQ alpha) | `prompt` (destructive: 5-30 min on the GPU) | 160 |
| 6 | `quantize` | Quantize the source model with the TesseraPolicy, write the TESSERA_* GGUF | `prompt` (writes to disk) | 120 |
| 7 | `convert` | Run `tessera-to-coreml` on the GGUF, write the `.mlmodelc` | `prompt` (writes 3-4 GB) | 100 |
| 8 | `evaluate` | Run the A/B Compare harness on a held-out eval set, return the per-tensor PPL + the per-token latency | `auto` | 140 |

Total tool LoC: ~830 Swift + 6 shared types.

The 8 tools compose the full Tessera pipeline. The
chat can run them in any order; the typical sequence
is `calibrate` -> `evolve` -> `quantize` -> `convert`
-> `evaluate`. The agent loop is responsible for
ordering; the user is responsible for approvals.

### 14.5 The approval engine (TesseraApprovalEngine)

User-facing approval for destructive tool calls.
When the agent loop hits a `prompt`-level tool, it
pauses, posts a `TesseraApprovalRequest` to the
approval engine, and waits. The user sees a modal
sheet in the chat surface with:
- The tool name + icon
- The input (rendered as a structured form, not raw JSON)
- The estimated duration + cost (e.g., "20 min, will write 4 GB to disk")
- Approve / Deny / Deny + don't ask again

The "don't ask again" is per-tool, per-session (not
persisted; the destructive behavior is always
re-prompted across sessions for safety).

(This per-session-only posture is superseded for the outward agent by
the learned-trust model in section 15.5, which persists approval
history under a one-way ratchet.)

LoC: ~120 Swift + 60 SwiftUI for the modal sheet.

### 14.6 The 3-destination shell

The app shell: 3 destinations in a `NavigationSplitView`,
following the Foundation Lab pattern. Each destination
is a top-level section in the sidebar.

| Destination | Purpose | Tessera target |
|---|---|---|
| **Library** | Browse + download + manage models and `.mlmodelc` files | `LibraryDestination.swift` (~250 LoC) |
| **Playground** | The chat + A/B Compare + on-device telemetry | `PlaygroundDestination.swift` (~150 LoC) |
| **Runs** | The agent loop history + tool call receipts + audit log | `RunsDestination.swift` (~200 LoC) |

Total destination LoC: ~600 Swift + ~80 SwiftUI for
the shell chrome.

### 14.7 The 3-destination -> 4-screen map

The chat is the primary screen in Playground. The
A/B Compare is a separate screen (a sheet over the
chat). The Settings surface is a separate screen
(an inspector panel in the chat sidebar, not a
top-level destination — per the iOS 26 pattern).
The Studio on Mac has 4 destinations, not 3: the
calibration pipeline editor is a Mac-only top-level
section.

### 14.8 The tool message UI (RichMessageView)

The chat surface renders tool calls as rich messages,
not raw text. Per the PrismAgent `ToolMessageView`
pattern + the Foundation Lab `@Generable` rendering:

- File content: monospace, syntax-highlighted
- Directory: tree view
- Search results: list with snippets
- Web results: card with title + URL + snippet
- Sidecar JSON: collapsible, syntax-highlighted
- Receipt: structured form with the audit log
- Error: red border, the stack trace, the recovery hint

The tool call banner is at the top of the message
(`TesseraToolBanner.swift`, ~36 LoC): icon-by-prefix
from PrismAgent (the tool's `iconName` + a chevron +
the tool's name + a duration + a status dot).

LoC: ~400 Swift + ~80 SwiftUI.

### 14.9 The sub-agent dispatch (QuantizationWorkerPool)

When the agent invokes `calibrate` or `evolve`, the
work is parallelizable across tensors. The
`QuantizationWorkerPool` (the rename of PrismAgent's
`SubAgentOrchestrator`) spawns N workers (N = the
number of GPU cores or the number of ANE engines,
whichever is larger) and dispatches per-tensor work.

For the calibration: 1 worker per layer, 1 work item
per tensor.
For the evolve: 1 worker per island, 1 work item per
generation.

The pool is observable; the agent can see the
progress (`worker X is processing tensor Y, 30% done`).
The chat renders a progress bar in the tool message
UI when the tool is long-running.

LoC: ~150 Swift (rename + thread pool + progress
emission).

### 14.10 The audit receipts (CalibrationReceipt)

Every destructive tool call produces a `CalibrationReceipt`
written to the conversation journal. The receipt is
JSON, schema-versioned (`llama.tessera.calibration-receipt.v1`),
and contains:
- The tool id + name + description
- The input (canonical JSON)
- The output (canonical JSON)
- The wall-clock duration
- The IOReport snapshot at tool completion
- A SHA-256 of the receipt itself (for cross-reference)

The receipt is the source of truth for the audit log
in the Runs destination. The user can export the
receipt chain as a single JSONL for offline analysis.

LoC: ~18 Swift (the receipt struct) + ~40 Swift
for the Runs destination rendering.

### 14.11 The AION mediator (Apple Intelligence)

The AION pattern from `Agent!` README.md:24-30:
Apple AI observes the conversation and injects
`[AI]`-prefixed annotations on-device, free, no API
cost. For Tessera Studio:

- When the user asks "what models are available?",
  AION injects a `modelCard` annotation
- When the user asks "is this quantized?", AION
  injects a `quantizationReport` annotation
- When the user asks "how long did the calibration
  take?", AION injects a `calibrationTiming`
  annotation

The AION mediator is a small Swift service
(~100 LoC) that listens to the conversation log
and posts annotations to the chat.

LoC: ~100 Swift. v1 ships with the AION mediator
on Mac (Apple Intelligence is macOS 26+); iOS
v1 ships without AION (Apple Intelligence is
limited on iOS).

### 14.12 The token budget visualization (StudioUsageView)

A small popover in the chat title bar that shows the
current session's token usage, the prompt token
count, the completion token count, the cached tokens
(if any), the per-tool-call token delta, and the
estimated cost (when the TesseraRuntime is `privateCloud`).

LoC: ~80 Swift.

### 14.13 The TesseraRuntime enum

A 3-value runtime selector, mirroring the
`FoundationModelRuntime` pattern from Foundation Lab:

```swift
public enum TesseraRuntime: String, Codable, CaseIterable,
    Sendable {
    case onDevice   // CoreML on the local ANE
    case mlx        // MLX (v2, for models that don't
                   //   fit the Tessera pipeline yet)
    case privateCloud  // v2, a remote Tessera endpoint
}
```

The default for v1 is `onDevice`. The selector lives
in the Settings surface; the chat reads it via
`@AppStorage`.

LoC: ~3 Swift (just the enum).

### 14.14 The SwiftData chat journal

The conversation log is persisted via SwiftData
(pattern from Agent! `AgentViewModel.swift:39-40`).
The schema is:
- `Conversation`: id, title, createdAt, updatedAt
- `ConversationTurn`: id, role (user/assistant/tool), content, toolCallId?, createdAt
- `ToolCall`: id, toolName, input, output?, status, startedAt, completedAt?

The SwiftData model is auto-migrated; the conversation
journal is a single SQLite file in the app's
`Application Support/Studio/` directory.

LoC: ~120 Swift for the SwiftData models + ~60 for
the migration.

### 14.15 The C FFI additions for the agent loop

The C FFI from section 3 needs three new functions
for the agent loop:

```c
// Stream the model's response; the agent loop calls
// this and consumes the AsyncStream of tokens.
int32_t tessera_stream_response(
    TesseraContextHandle ctx,
    const char * system_prompt,
    const char * user_message,
    TesseraModality modality,
    TesseraTokenCallback callback,
    void * user_data
);

// Pause + resume a streaming response (for tool
// approval pauses).
int32_t tessera_pause_stream(TesseraContextHandle ctx);
int32_t tessera_resume_stream(TesseraContextHandle ctx);
```

LoC: ~200 C.

### 14.16 The 20 patterns the agent recommended to skip

For reference, the patterns the agent flagged to
NOT adopt (per `docs/agent-patterns-research.md`):

(Several of these were v1 skips and are promoted to later-wave scope in
section 15.4: computer use, Accessibility scan, AppleScript bridge,
global hotkey, and LoRA adapter training.)

- 15-button toolbar (Agent!) — visual noise
- XPC user agent + privileged daemon (Agent!) —
  Studio is foreground single-user
- MCP server config UI (Agent!) — not in v1
- Computer use (PrismAgent) — out of scope
- AppleScript bridge (PrismAgent) — not needed
- Accessibility scan (PrismAgent) — not needed
- Action overlay (PrismAgent) — foreground app
- iMessage remote (Agent!) — not Tessera use case
- Voice hotword (Agent!) — not hands-busy
- Plan mode (PrismAgent, Agent!) — calibration is
  linear
- TUI mode (seldon) — GUI users first
- Global hotkey (Motive) — windowed app
- LoRA adapter training (junco) — calibration, not
  fine-tuning
- Server mode (Foundation Lab) — macOS/iOS only for
  v1
- JSONL repo-map (Agent!) — we don't edit code
- `fmas` Python tooling (Foundation Lab) — Swift-native
- Sub-agent dispatcher for chat tool calls
  (PrismAgent) — sub-agent pool yes; chat dispatcher no
- Tessera-core LoRA (junco) — out of scope
- iOS BackgroundKeepAlive / Live Activity
  (sample-mobile-ai-assistant) — macOS only
- Chat history drawer (already adopted) (AWS sample) —
  already in design doc 5.6

### 14.17 Total agent loop LoC impact

| Component | LoC |
|---|---:|
| TesseraTool protocol (14.1) | ~150 |
| TesseraToolRegistry (14.2) | ~80 |
| TesseraAgentLoop (14.3) | ~280 |
| 8 v1 tools (14.4) | ~830 |
| TesseraApprovalEngine (14.5) | ~180 |
| 3-destination shell (14.6) | ~680 |
| Tool message UI (14.8) | ~480 |
| QuantizationWorkerPool (14.9) | ~150 |
| CalibrationReceipt (14.10) | ~58 |
| AION mediator (14.11) | ~100 |
| Token budget viz (14.12) | ~80 |
| TesseraRuntime enum (14.13) | ~3 |
| SwiftData journal (14.14) | ~180 |
| C FFI additions (14.15) | ~200 |
| **Total** | **~3,651** |

The v1 Studio is now ~4,800 + ~3,651 = **~8,451 LoC
Swift** + **~3,700 LoC C/C++** = **~12,151 LoC total**.

### 14.18 The new open questions for the agent loop

The agent's "Should Tessera Studio get an agent loop?"
question is now answered (YES for v1). The follow-up
questions are:

- **Q13. Tool count**: 8 tools is the v1 floor. Should
  the v1 ship more (e.g., `inspect_receipt`,
  `diff_policies`, `export_receipts`, `import_receipts`,
  `fork_session`)? Lean: 8 for v1, +4 for v2.
- **Q14. Approval granularity**: per-tool, per-session,
  or both? Lean: per-tool + per-session override; the
  default is per-tool with the destructive-tools list
  prompting.
- **Q15. Sub-agent visibility**: should the user see
  the sub-agent worker pool (the parallel calibration
  workers) in the tool message UI? Lean: yes, a
  collapsible "Workers" section in the tool message
  that shows per-worker progress.
- **Q16. AION on iOS**: Apple Intelligence is
  limited on iOS; the AION mediator is Mac-only for
  v1. Lean: Mac-only v1, iOS v2.
- **Q17. Multi-modal tool inputs**: the `calibrate` and
  `evolve` tools accept multi-modal inputs (the 5k
  corpus). The tool input schema needs to encode the
  modality. Lean: the tool input is a
  `CalibrationConfig` struct with a `modality` enum
  and a `corpus` reference.

### 14.19 Risk additions for the agent loop

The agent loop adds 4 new risks to section 12:

- **R27. Agent loop runaway**: a poorly-prompted agent
  could invoke tools in a tight loop, generating cost
  or filling disk. Mitigated by the per-tool cost
  ceiling, the per-session token budget, and the
  cancellation surface (`TesseraAgentLoop.cancel()`).
- **R28. Tool approval bypass**: a user could
  pre-approve a destructive tool via the "don't ask
  again" affordance, then forget. Mitigated by
  per-session-only approval persistence (the "don't
  ask again" expires at app restart).
- **R29. Sub-agent pool size**: a chat with N tools
  spawning M sub-agents per tool could exhaust the
  iOS memory budget. Mitigated by a hard cap on the
  pool size (8 workers on iOS, 32 on Mac) and a
  queue-based dispatch.
- **R30. AION hallucination**: Apple's Apple
  Intelligence can annotate incorrectly. Mitigated
  by the `modelCard` annotation being a hint, not a
  fact; the user can dismiss the annotation.

### 14.20 Updated phasing

The v1 phasing from section 10 is updated to include
the agent loop in Phase 2 and Phase 3 (parallel with
the existing work):

- **Phase 1 (~3 weeks)**: The engine integration
  (LibTessera.swift + the C FFI + the xcframework
  build). Same as before.
- **Phase 2 (~4 weeks)**: The Swift Package skeleton
  (TesseraCore shared + the 2 targets + the chat
  surface on iOS) + **the agent loop core (TesseraTool
  protocol, TesseraToolRegistry, TesseraAgentLoop,
  TesseraApprovalEngine, the 8 v1 tools)**. Phase 2
  grows from 3 to 4 weeks to accommodate the agent
  loop.
- **Phase 3 (~3 weeks)**: The CoreML backend + the
  IOReport telemetry + **the tool message UI
  (RichMessageView, TesseraToolBanner, the sub-agent
  dispatch)**. Phase 3 grows from 3 to 4 weeks.
- **Phase 4 (~4 weeks)**: The Studio surface on Mac
  (calibration, evolution, telemetry review) + **the
  3-destination shell + the audit receipts + the
  SwiftData journal**. Phase 4 grows from 3 to 4
  weeks.
- **Phase 5 (~2 weeks)**: The A/B compare view + the
  30-min flight test + **the AION mediator (Mac) +
  the token budget viz**. Phase 5 grows from 2 to
  3 weeks.
- **Phase 6 (~2 weeks)**: Polish, App Store
  compliance (privacy manifest, AI disclosure), beta
  testing. Same as before.

**Total: 18 weeks wall-clock with 1 dev, ~7 weeks
with 4 parallel agents after Phase 1.**

The agent loop adds 2 weeks to the 1-dev timeline
and 1 week to the 4-parallel-agent timeline. The
Studio v1 is now ~12,151 LoC across 3 targets +
the xcframework.

---

## 15. Agent harness direction: the outward agent (added 2026-07-31)

Section 14 specified the INWARD agent: the loop that drives the Tessera
calibration pipeline (calibrate -> evolve -> quantize -> convert ->
evaluate). This section adds the OUTWARD agent: a general Mac agent that
acts on the user's behalf across the machine - computer use, browser,
research - the same loop pointed at the world instead of at the
quantizer. One machine, two payloads (`self-improving-loop-design.md`
section 1): the agent used by day is the harness that harvests the
training signal that improves the model by night. The outward
capabilities are scoped in `PROJECT-STATUS.md` Priority 9 and absorbed
selectively from seven scouted open-source agents
(`docs/tessera-harness-absorption-2026-07-31.md`).

This section extends the v1 spec; where it conflicts with section 14's
v1 skips and v1 approval posture, this section governs the later waves.

### 15.1 Positioning: agent manager, not an editor

Studio orchestrates, verifies, and records agents; editors and browsers
are things it DRIVES and DIFFS against, not things it IS. No text editor,
LSP, debugger, or extension ecosystem. The editor is a commodity
(Google's Antigravity just forked VS Code) and a tar pit for a small
team; the seat Tessera takes is the layer ABOVE the editor - the thing
that manages agents, holds the evidence, enforces approval, and drives
the Mac. Diff-review + "open in editor" replaces building an editor.
(This refines section 14.16's "we don't edit code" skip: the agent
reviews and proposes diffs and opens the user's editor; it does not
become the editor.)

### 15.2 Distribution: Developer ID for the Mac app

The Mac app ships via Developer ID + notarization, NOT the Mac App Store
(confirmed by the architect). Deep macOS integration - the Accessibility
API, Full Disk Access, screen recording - is impossible under Mac App
Store sandboxing, and those are the differentiator, so the Mac app takes
the Developer ID path that every serious computer-use product uses. This
is scoped to the Mac: the iOS app keeps the App Store path in section 13
(IAP model distribution, privacy manifest, AI disclosure all unchanged).
The two distribution channels are independent; nothing in section 13 is
superseded.

### 15.3 The macOS integration ladder

The local product can go places a cloud product cannot, in rough order
of permission burden:

- Global hotkey + menu bar - summon an agent from anywhere. (Promotes
  section 14.16's "global hotkey" skip.)
- Services / Quick Actions - right-click any text or file -> "ask
  Tessera."
- ScreenCaptureKit - see any screen or window; the perception half of
  computer use.
- Accessibility API - read and drive ANY app's UI, not just the browser;
  the action half of computer use. (Promotes section 14.16's
  "Accessibility scan" + "computer use" skips.)
- AppleScript / Apple Events - script Finder, Mail, Calendar, Notes,
  Reminders. (Promotes section 14.16's "AppleScript bridge" skip.)
- Vision OCR + Speech transcription - free, on-device perception.
- Apple Foundation Models - the zero-cost local drafter tier and default
  teacher (`self-improving-loop-design.md` 4.1); the seed is the AION
  mediator in 14.11, promoted from annotation hint to first-class
  runtime + teacher.

### 15.4 The outward capabilities (Priority 9, three waves)

- **Wave 1 - safety spine + cheap wins (P0).** Approval-engine hardening
  (layered permission: policy x profile x sandbox-enforceability,
  fail-safe to AskUser); a fail-closed action verifier ("verify a real
  state change, not a self-reported success"); a denial circuit-breaker
  (the collapse guard, made concrete); per-claim citation + a
  never-fabricate contract; a skills directory + `SKILL.md` loader; a
  research tool over `TesseraWebSearch`.
- **Wave 2 - native capabilities (P0/P1, macOS-first).** A computer-use
  tool (ScreenCaptureKit -> Accessibility -> CGEvent, model-native
  coordinate grounding, skill-capture receipts, capture-time PII scrub)
  and a browser tool (WKWebView + an indexed-DOM serializer + a
  page-change re-ground guard).
- **Wave 3 - identity + polish (P1/P2).** `SOUL.md` persona, per-model
  harness profiles + context-budget rules, a local-first config posture
  + `doctor` migrations, scoped gating, source curation.

Deliberately NOT absorbed: anyone else's agent loop, cloud/vendor/server
infra, heavy Python/CUDA/CV stacks, unsigned-binary supply chains, and
self-judging evaluation. (This revisits section 14.16: computer use,
Accessibility, AppleScript, and the global hotkey were v1 skips and are
now Wave-2/Wave-3 scope; LoRA training was a v1 skip and is now the
whole inward flywheel of `self-improving-loop-design.md`.)

### 15.5 Autonomy calibration: needy -> learned trust -> scoped YOLO

_Evidence base: `research-autonomy-calibration-2026-07-31.md`. That note
is the source of truth for the claims below; where this section and the
note disagree, the note wins until this section is updated._

_Detailed engineering spec: `autonomy-calibration-design.md`. This section
is the overview and rationale; that doc is the buildable spec - the
action-class identity scheme, the learned-permission store, the asymmetric
ratchet algorithm, the precedence model, scoped YOLO, the audit/revocation
surface, and the leashed neural approver. Where the two disagree, the
detailed spec wins for implementation._

Studio starts NEEDY: it asks for approval often. Every approval and
denial is a receipt (14.10), and the approval policy is a LEARNED
PROJECTION over that history: action-classes the user consistently
allows auto-continue; novel and edge cases keep prompting. This
SUPERSEDES the v1 approval posture in 14.5 and risk R28, which held
approval to per-session-only and never persisted, on the theory that
persisted approval is dangerous. The learned-trust model is safe
precisely because of the invariants below; the per-session-only rule was
a blunt instrument for a danger the ratchet defuses.

**Why this is a real gap, not a feature grab.** Every shipping coding
agent has a permission system; NONE learns. Claude Code, Cursor, Codex
CLI, Cline, and OpenHands have all converged on a static spectrum from
"ask everything" to "ask nothing," with a per-action classifier as the
2025-2026 frontier (research note section 1, section 8). Classifiers
judge each action in isolation; none accumulates evidence across
sessions to move the baseline. Tessera's receipt-driven learned
permission is the differentiator, and the longitudinal local approval
history it runs on is the moat (15.6).

**The design goal is calibration, not trust.** The human-factors
literature is explicit: "design for appropriate trust, not greater
trust" (Lee & See 2004, research note section 2). Both over-trust and
under-trust are failure modes. A harness that only defends against
over-trust by asking too often manufactures DISUSE: the user stops
reading prompts and rubber-stamps them, defeating the safety mechanism.
Claude Code's own reported 93% approval rate is disuse in progress
(research note section 1). So the goal is to match autonomy to actual
reliability - ask when it matters, stay quiet when it does not.

**Three outcomes, not two.** The approval engine produces
`autoApprove` / `askUser` / `reject` (landed in `TesseraSafetyDecision`).
This maps to Claude Code's auto-mode tiers and OpenHands' ConfirmRisky
policy. `autoApprove` = learned trust OR contained low-risk; `askUser` =
uncertain or novel; `reject` = high-risk, forbidden, or circuit-breaker
tripped. The loop MUST honor all three: today only `reject` is branched
on and `autoApprove` still falls through to the prompt path, which is
the wiring gap this section authorizes closing (research note section
10).

- **One-way ratchet (load-bearing).** Learning only ever moves toward
  MORE autonomy on OBSERVED-SAFE patterns. A NEW consequential or
  irreversible action-class ALWAYS prompts, regardless of history. "You
  have approved 200 file edits" must never translate into "auto-approve
  this `rm -rf`." Trust is per-action-class, and irreversible classes
  are permanently un-trainable. This is the structural defense against
  OWASP ASI10 (rogue agents accumulating access over time) and against
  approval-gaming (research note section 6, section 7).
- **The ratchet is ASYMMETRIC.** Trust repairs slower than it builds:
  automation failures cut trust more than successes restore it, and
  trust after a violation may never return to its initial level (Yang et
  al. 2023, Rittenberg et al. 2024, De Visser et al. 2018; research note
  section 3). Encode this directly. GRANT threshold: N consecutive
  approvals of the same action-class across M distinct sessions
  (conservative defaults N=5, M=3, user-tunable). REVOKE threshold: a
  SINGLE denial of that action-class resets its approval counter. Denial
  signals are weighted heavier than approval signals by construction.
- **The circuit breaker outranks the ratchet.** Given the asymmetry
  above, the denial circuit-breaker (S3, already landed) is MORE
  important than the ratchet. A tripped breaker fires before the ratchet
  can grant further autonomy and resets all learned permission for the
  affected action-class. If the two ever disagree, the breaker wins.
- **Action-class identity is STRUCTURAL, not linguistic.** Learned
  permission keys on tool identity + argument structure (e.g.
  `bash:git status`, `bash:npm test`, `file_write:src/**`), never on the
  natural-language description of the action. This blocks the
  specification-gaming surface where an agent learns to REPHRASE a
  dangerous action to match an approved pattern (Goodhart / reward
  hacking; research note section 7). Start with tool + argument-prefix
  patterns and evolve the granularity later (open question, section 9).
- **The learned-permission store is auditable and revocable.** The user
  can inspect every grant (which action-class, how many approvals,
  across how many sessions, when granted) and revoke any entry. This is
  the transparency Lee & See's "process" trust basis requires and the
  antidote to access accumulating out of sight (research note section 6,
  section 10).
- **Dispositional floor and ceiling.** Some users want more autonomy on
  day one; others want to stay needy forever (Hoff & Bashir 2015's
  dispositional layer; research note section 2). The user sets a FLOOR
  (minimum approval requirements learning cannot reduce) and a CEILING
  (maximum autonomy learning cannot exceed). Learning moves within the
  band; it never crosses the walls.
- **Escalation communication follows selective-prediction evidence.**
  When asking, describe the action and its risk tier, what changes if it
  succeeds, and what cannot be undone. Do NOT show the agent's
  confidence score or reasoning chain: revealing the AI's prediction
  anchors the human and degrades the decision (research note section 5).
  The prompt is about the ACTION, not the agent's opinion of it.
- **Escalate on expected value, not a static table.** Per Horvitz (1999)
  principle 4, auto-approve only when the expected value of acting
  exceeds the expected value of asking, weighted by P(user approves |
  context) and the cost of each error (research note section 4). Every
  prompt has a real cost - interrupted flow, attention switching,
  approval fatigue - and the spec models it rather than pretending
  asking is free. Irreversible actions require explicit consent
  regardless of learned trust (principle 7).
- **Scoped YOLO mode (first-class, not a toggle).** A time-boxed AND
  goal/session-boxed override that auto-approves within scope. It has:
  explicit activation ("go fast for this task"); bounded scope (a
  specific goal or session); a hard time limit (configurable, default 30
  minutes); full receipt logging (every action recorded even though not
  prompted); and automatic expiry with a summary of what ran
  autonomously. A YOLO session is the richest training data the harness
  gets (many actions, fast) and feeds the loop rather than escaping it.
  Industry YOLO modes are unbounded toggles; the bounding is the point.

### 15.6 One receipt stream, two learners

The receipts defined in 14.10 are the shared substrate for both
payloads. The SAME accept/reject + outcome stream trains the model (idle
LoRA, `self-improving-loop-design.md` 4.4-4.5) and trains the approval
policy (15.5). Capture is on by default and local; learning and egress
are opt-in (`self-improving-loop-design.md` section 6). No cloud product
can replicate this, because none has the user's local longitudinal
approval history; that history is the moat, and it is built one receipt
at a time.

### 15.7 The one honest tension

Teacher distillation sends the user's struggling prompts to a teacher -
the single real egress in an otherwise-local system. It is therefore
opt-in, approval-governed, and scrubbed before it leaves; Apple
Foundation Models / Private Cloud Compute is the default teacher
precisely because it barely egresses at all
(`self-improving-loop-design.md` 4.1). Computer use is the other face of
the same tension: an agent that can click anything can click the wrong
thing, and a screen recorder captures passwords and PHI by construction.
The mitigation is the approval engine + the no-egress boundary +
capture-time scrub + the receipts, treated as gates, not afterthoughts.
This is what makes an autonomous Mac agent trustworthy rather than
terrifying, and it is the constitutional/receipt architecture Tessera
already owns.

### 15.8 New open questions for the outward agent

- **Q18. How much autonomy before requiring approval?** The ratchet
  (15.5) needs a threshold. The research
  (`research-autonomy-calibration-2026-07-31.md` section 10) supplies
  concrete, conservative defaults: GRANT after N=5 consecutive approvals
  of an action-class across M=3 distinct sessions; REVOKE on a single
  denial; both user-tunable within a dispositional floor/ceiling. The
  genuinely OPEN question is no longer the count but the ACTION-CLASS
  GRANULARITY: tool name alone is too coarse (all bash is one class),
  tool + argument-prefix is the proposed start, semantic clustering is
  most flexible but hardest to audit (research note section 9). Lean:
  ship tool + argument-prefix patterns, keep the identity auditable, and
  evolve granularity only with evidence.
- **Q19. AFM teacher quality on the hard tail.** Is Apple Foundation
  Models good enough as the default teacher for the genuinely-hard
  escalations, or does the hard tail always need a third-party teacher?
  The recurring per-teacher assessment (`self-improving-loop-design.md`
  4.1) measures this; lean: AFM default, third-party on demand.
- **Q20. Computer-use permission onboarding.** Accessibility + Full Disk
  Access + screen recording each need a separate macOS grant. Lean: a
  guided first-run flow that requests each only when the capability that
  needs it is first used, not all up front.

## 16. Workflows (data model + workflow-as-code execution + SwiftUI editor)

The workflow system. This section covers the data model
(Phase 1), the SwiftUI graph editor (Phase 2), and the
wrapped Tessera tools that ship in the default registry.
The custom node-pack plugin model that was once Phase 3
has been removed from scope (architect decision 2026-08-04):
Tessera Studio is a single-shipped Mac app, no third-party
node system, no plugin discovery.

### 16.1 What a workflow is

A workflow is a directed graph of typed nodes. Each node has:

- **typed input ports** (the executor fills them from upstream
  outputs),
- **per-instance parameters** (a JSON object the user sets in
  the editor's side panel; not wired),
- **typed output ports** (downstream nodes consume them).

Edges connect a source `(nodeId, outputPortId)` to a target
`(nodeId, inputPortId)`. The executor enforces that the source
and target port types are equal (or the source type is `path`
and the target is `gguf` / `json` — the only allowed widening).

The whole workflow is serialised as a single JSON file with
schema marker `tessera.workflow.v1`. New fields are additive
(default values); new node types are additive (unrecognised
types produce a validation error at load time).

### 16.2 The data model

Five files in `TesseraStudio/Sources/TesseraCore/Workflow/`:

- `WorkflowPortType.swift` — closed enum of port types
  (`string`, `number`, `boolean`, `path`, `gguf`, `json`,
  `toolResult`, `bag`) + `canFlowInto` widening rule + `WorkflowPortValue`
  typed-value carrier.
- `WorkflowNodeType.swift` — the protocol, plus the per-run
  `WorkflowExecutionContext` (file system + logger) and the
  `TesseraFileSystem` / `WorkflowLogger` abstractions.
- `WorkflowNodeRegistry.swift` — the metatype-keyed catalogue
  + the `WorkflowNodePaletteEntry` shape the editor's palette
  consumes.
- `Workflow.swift` — `Workflow`, `WorkflowNode`, `WorkflowEdge`,
  `WorkflowEvent`.
- `WorkflowExecutor.swift` — actor, validation, Kahn's
  topological sort, `run` / `runCollecting` over `AsyncStream`.
- `Nodes/TesseraToolNode.swift` — the wrapper that lifts the
  existing 18 `TesseraTool`s into workflow nodes, plus 5
  default wrapped types (`LoadModelNode`, `CalibrateNode`,
  `QuantizeNode`, `EvaluateNode`, `InspectSidecarNode`) and
  `WorkflowNodeRegistry.default`.

### 16.3 Wrapping TesseraTools as nodes

Each wrapped node is a zero-state struct that holds a static
reference to the underlying `TesseraTool`. The wrapper's
`splitSchema` rule:

- **Required** schema properties become input ports (typed
  from the schema's `type` field; `_path` suffix maps to
  `WorkflowPortType.path`).
- **Optional** properties (have a `defaultValue`, or aren't in
  `required`) become node-level parameters edited in the side
  panel, not wired.
- One synthetic output port `result` typed `toolResult`,
  carrying the `ToolResult.data` map (or a synthesised
  `{success, output, error}` map if the tool returns nil data).

Adding a new wrapped tool is ~20 lines: declare the struct,
supply the static `typeId` / `displayName` / etc., forward
`execute` to `TesseraToolNode.execute`. No edits to the
underlying `TesseraTool` are required.

### 16.4 Worked example: calibrate -> quantize -> save

```swift
let wf = Workflow(
    name: "calibrate-and-quantize",
    nodes: [
        WorkflowNode(
            id: "calib",
            type: CalibrateNode.typeId,
            parameters: ["n_tokens": .number(8000)]
        ),
        WorkflowNode(
            id: "q",
            type: QuantizeNode.typeId,
            parameters: [:]
        ),
    ],
    edges: [
        WorkflowEdge(
            fromNode: "calib", fromPort: "result",
            toNode: "q", toPort: "policy_path"
        ),
    ]
)
let executor = WorkflowExecutor(registry: .default)
for await event in await executor.run(wf) {
    // Started / nodeStarted / nodeFinished / log / finished.
    // Last event is always .finished; the executor aborts on
    // the first failed node.
}
```

The editor surface (Phase 2) draws this same JSON as a
graph; the executor's `AsyncStream<WorkflowEvent>` is what
feeds the live progress pane.

### 16.5 Validation

The executor's `validate(_:)` rejects:

- nodes with an unregistered `typeId` (typo or stale JSON),
- edges that reference a port the target node doesn't declare,
- edges whose source / target port types are incompatible
  (per `WorkflowPortType.canFlowInto`),
- graphs that contain a cycle (Kahn's algorithm).

A `run` that fails validation produces a single
`.finished(success: false)` event with the error string in
`message`; no nodes are executed.

### 16.6 Phasing (COMPLETE)

The workflow system shipped in two phases. Both have merged
to main.

1. **Phase 1 — Data model + workflow-as-code execution
   (merged `3ef20e120`).** The five data model files, the
   wrapped tools, the executor, and the round-trip /
   validation / executor tests. Hand-built 3-node
   `calibrate -> quantize -> save` workflow executes
   end-to-end against a stub tool.
2. **Phase 2 — SwiftUI graph editor (merged `292bef1a2`).**
   `WorkflowCanvasView` (SwiftUI `Canvas` for bezier
   connections + node rectangles + in-flight wire preview),
   `WorkflowNodeView` (drag-to-move via the header bar,
   port-hit-testing on the port dot), `WorkflowPaletteView`
   (List bound to `WorkflowNodePaletteEntry`),
   `WorkflowToolbarView` (New / Open / Save / Run),
   `WorkflowParameterPanelView` (form bound to the selected
   node's `JSONSchema`), `WorkflowDocument` (FileDocument
   envelope with optional positions, schema
   `tessera.workflow.document.v1`), `WorkflowsView`
   (container with the new "Workflows" tab in
   `TesseraStudioMac` `NavigationSplitView`),
   `WorkflowGeometry` (shared port-center / node-height
   math in `TesseraCore` so the canvas renderer and the
   drop-test validator can't drift).

A third phase (custom node-pack plugin system) was
originally planned here and is now removed from scope:
Tessera Studio is a single-shipped Mac app, no third-party
node system, no plugin discovery, no per-plugin manifest.
Adding a new node is a code change in `TesseraCore`. This
removes the third-party legal tail entirely and matches
the architect's scope statement (2026-08-04): "the scope
is just the tessera studio mac app".

### 16.7 Why this order

Phase 1 was load-bearing: the editor (Phase 2) is just a
pretty picture without a clean protocol + Codable +
executor. The palette-entry shape shipped in 1.1 is exactly
the API the SwiftUI `List` in `WorkflowPaletteView` ended
up consuming. There is no third phase to justify; the
workflow system ends at the editor.

### 16.7a What the user can do today

After Phase 1 + Phase 2 are merged, the user can:
- Open the "Workflows" tab in Tessera Studio (macOS).
- See the palette of wrapped TesseraTools on the left.
- See a hard-coded `calibrate -> quantize` example
  workflow on the canvas.
- Drag any node to move it.
- Drag from a node's output port to another node's
  input port to wire them (validated: type-compatible,
  no self-loops, no parallel edges).
- Click a node to select it; the right pane shows its
  parameters (form bound to the JSONSchema); edits flow
  back into the workflow.
- Save the workflow to a `.json` file (the
  `tessera.workflow.document.v1` envelope) and re-open
  it later.
- Hit "Run" to execute the workflow through
  `WorkflowExecutor.run`; the progress sheet shows the
  live `AsyncStream<WorkflowEvent>` as nodes start, finish,
  log, and the workflow reports success or failure.

### 16.8 License analysis (binding)

The decision to cite ComfyUI as prior art in this design
doc, without importing any ComfyUI code, is binding. The
justification:

- Tessera Studio is PolyForm Noncommercial 1.0.0.
- ComfyUI is GPL-3.0.
- Combined work would have to allow commercial use (per
  GPL viral copyleft), violating PolyForm. They are
  mutually exclusive.
- Patterns (node-graph editor UX, typed port system,
  palette-driven discovery) are not copyrightable. Specific
  code is.
- Reimplementing the patterns from scratch in Swift is the
  only path that satisfies both licenses.
- With no third-party plugin system, the analysis is
  bounded to Tessera Studio itself: it is from-scratch
  Swift that re-implements the editor UX. There is no
  scenario in which a ComfyUI-derived work would be
  distributed as part of Tessera Studio.

This is the same constraint Prism Engine operates under
(its compiler / runtime is also from-scratch despite
referencing LLVM / rustc patterns).

## Appendix A: File index

This section lists the files referenced in the design
and their roles.

### PrismAgent references

- `PrismAgent/Package.swift:1-143` - the project layout
  (PrismAgent SDK + iOS + Mac + Keyboard + Watch +
  EngineC + BridgeFFI + CredentialService).
- `PrismAgent/PrismAgentiOS/PrismAgentiOS/InferenceAdapter
  .swift:1-83` - the adapter protocol, KEEP + REPLACE
  inner.
- `PrismAgent/PrismAgent/ModelStore.swift:1-229` - the
  model store, KEEP + TESSERA-FLAVOUR.
- `PrismAgent/PrismAgent/ModelStoreView.swift:1-428`  - 
  the model store view, KEEP + REPLACE + ADD BADGES.
- `PrismAgent/PrismAgent/ConversationStore.swift:1-172`  - 
  the conversation store, KEEP + TESSERA-FLAVOUR.
- `PrismAgent/PrismAgent/PrismChatView.swift:1-479`  - 
  the chat view, KEEP + SIMPLIFY.
- `PrismAgent/PrismAgent/PrismEngineApp.swift:1-14`  - 
  the Mac app entry, KEEP minimal.
- `PrismAgent/PrismAgentiOS/PrismAgentiOS/PrismAgentiOSApp
  .swift:1-39` - the iOS app entry, KEEP minimal.
- `PrismAgent/PrismAgent/PrismModelManager.swift:1-171`  - 
  the model manager, KEEP + simplify.
- `PrismAgent/PrismAgent/PrismModelCardView.swift:1-206`
  - the model card view, KEEP + REPLACE.
- `PrismAgent/PrismAgent/PrismBridgeAdapter.swift:1-1105`
  - the engine bridge, PARTIAL (StreamHandler pattern
  only).
- `PrismAgent/PrismAgent/SubAgentOrchestrator.swift:1-100`
  - the sub-agent pool, KEEP + RENAME.
- `PrismAgent/PrismAgent/AutonomousOrchestrator.swift:
  1-159` - the agent loop, SKIP.
- `PrismAgent/PrismAgent/AgentToolDispatcher.swift:1-292`
  - the tool dispatcher, SKIP.
- `PrismAgent/PrismAgent/P2PRouter.swift:1-72` - the P2P
  bridge, SKIP.
- `PrismAgent/PrismAgentiOS/PrismAgentiOS/ConversationList
  .swift:1-56` - the iOS conversation list view
  (KEEP + ADAPT for the chat history drawer, see
  section 4.4a).

### llama.cpp references

- `/Users/user/Developer/GitHub/llama.cpp/examples/
  llama.swiftui/llama.cpp.swift/LibLlama.swift:1-337`  - 
  the C FFI wrapper pattern (port to LibTessera.swift).
- `/Users/user/Developer/GitHub/llama.cpp/examples/
  llama.swiftui/llama.swiftui/Models/LlamaState.swift:
  1-196` - the model state pattern (port to ChatView).
- `/Users/user/Developer/GitHub/llama.cpp/examples/
  llama.swiftui/llama.swiftui/UI/ContentView.swift:1-156`
  - the chat UI (reference for the iOS ContentView).
- `/Users/user/Developer/GitHub/llama.cpp/build-
  xcframework.sh:1-550` - the xcframework build script
  (fork to build-tessera-xcframework.sh).
- `/Users/user/Developer/GitHub/llama.cpp/examples/
  llama.swiftui/README.md:1-27` - the build instructions
  (model for the Tessera Studio README).
- `/Users/user/Developer/GitHub/llama.cpp/examples/
  llama.swiftui/llama.swiftui.xcodeproj/project.pbxproj:
  1-449` - the project file (target structure).

### Tessera references

- `docs/tessera.md` - the Tessera quantization spec
  (GGUF metadata fields, AWQ-evolve, calibration).
- `docs/c++-port-design.md` - the C++ port design
  (G0-G6, CLI surface, sidecar JSON shape).
- `docs/multimodal-calibration-design.md` - the
  multi-modal calibration design (M1-M8, schema
  diff, per-modality AWQ, L5 scorer).
- `docs/tessera-coreml-conversion-design.md` - the
  CoreML conversion + runtime design (C1-C10,
  ggml-coreml, IOReport, test path, risk register).
- `docs/runtime-aware-pipeline.md` - the L1-L6
  telemetry pipeline.
- `docs/per-tensor-calibration.md` - the per-tensor
  GA calibration (the `ternary_threshold` knob).

## Appendix B: Glossary

- **A/B Compare**: the iOS surface that runs two
  engines side-by-side on the same prompt.
- **ANE**: Apple Neural Engine.
- **Calibration**: the process of computing the
  imatrix v2 (importance scores) from a calibration
  corpus.
- **CalibrationSession**: the Swift object that
  represents a single calibration pass.
- **Conversion**: the process of writing a
  `.mlmodelc` from a Tessera-quantized GGUF.
- **CoreML**: Apple's machine learning framework.
- **.mlmodelc**: the compiled CoreML model bundle.
- **Evolved policy**: the output of AWQ-evolve, the
  per-tensor AWQ alpha + the per-modality scales.
- **Flight test**: the 30-minute sustained-battery
  measurement.
- **G0-G6**: the C++ port phasing (see
  `docs/c++-port-design.md:699-810`).
- **G7**: the ggml-coreml runtime workstream.
- **GA**: genetic algorithm, used in AWQ-evolve.
- **GGUF**: the GGML Universal Format, the Tessera
  container.
- **Imatrix**: importance matrix, the per-tensor
  activation statistics.
- **IOReport**: Apple's private API for hardware
  telemetry.
- **L1 / L1.5 / L2 / L3 / L4 / L5 / L6**: the
  runtime-aware pipeline layers (see
  `docs/runtime-aware-pipeline.md:24-30`).
- **M1-M8**: the multi-modal calibration decisions
  (see `docs/multimodal-calibration-design.md`).
- **Plan**: the QuantizationPlan, the five-step
  pipeline.
- **QuantizationPlan**: the Swift object that
  represents the user's pipeline configuration.
- **Sidecar**: the v3 schema evidence file written
  next to the GGUF.
- **Studio**: the Mac-only primary surface.
- **TelemetryObserver**: the Swift object that
  consumes the L1 + L1.5 + B + C + E layers.
- **Tessera**: the quantization architecture, the
  9-component cluster, the project name.
- **TesseraEngine**: the public entry point for the
  engine.
- **TesseraModality**: text / image / audio, the M5
  decision.
- **TesseraPolicy**: the JSON policy (modality_scales
  + AWQ alpha + calibration metadata).

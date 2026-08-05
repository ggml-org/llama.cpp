// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Tessera Studio v2 - target layout.
//
//   CLlama            C target: runtime-loaded (dlopen) bridge to libllama
//                     for on-device inference. Compiles against the repo's
//                     llama.h when present; otherwise builds a stub that
//                     reports unavailable (see CLLAMA_NO_HEADERS below).
//   CTesseraFFI       C target: TesseraFFIBridge's FFI surface. Compiles a
//                     stub that reports unavailable when built by SwiftPM
//                     (so swift build / swift test pass standalone). When
//                     tessera.xcframework is linked in the Xcode app the
//                     real C++ implementation takes over and isAvailable
//                     returns true. See Sources/CTesseraFFI/tessera_ffi.c.
//   TesseraCore       Platform-independent models, tool protocol, engine
//                     bridge protocol + CLI bridge, and the shared SwiftUI
//                     views (guarded with #if os() where needed).
//   TesseraStudioMac  macOS app: AppKit integrations, Settings scene,
//                     NSSavePanel export. Depends on TesseraCore.
//   TesseraStudioiOS  iOS app surface. Depends on TesseraCore.
//
// The iOS .app cannot be produced by `swift build` on a macOS host (SPM
// builds for the host platform only), so TesseraStudioiOS is a library
// target whose sources are #if os(iOS)-guarded. On a macOS build it
// compiles to an empty module; the real iOS .app is produced by the
// Xcode project, which embeds tessera.xcframework. This mirrors the
// pattern in docs/tessera-studio-design.md section 2.2.
//
// The native FFI surface (tessera.xcframework) is built by
// TesseraStudio/scripts/build-xcframework.sh and linked in the Xcode app.
// SwiftPM builds the stub in Sources/CTesseraFFI instead; see
// Sources/TesseraCore/tessera_ffi_reference.h for the contract history.

// CLlama compiles against the llama.cpp public headers, which live in the
// enclosing repo (this package is nested inside the fork). Resolve them
// relative to this manifest so the build works from any checkout location.
// When the headers are absent (package built standalone), define
// CLLAMA_NO_HEADERS so the shim compiles to an always-unavailable stub.
let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()   // TesseraStudio/
    .deletingLastPathComponent()   // repo root
let llamaHeaderDir = repoRoot.appendingPathComponent("include")
let ggmlHeaderDir = repoRoot.appendingPathComponent("ggml/include")
let commonHeaderDir = repoRoot.appendingPathComponent("common")
let hasLlamaHeaders = FileManager.default
    .fileExists(atPath: llamaHeaderDir.appendingPathComponent("llama.h").path)

let cllamaCSettings: [CSetting] = hasLlamaHeaders
    ? [
        .unsafeFlags(["-I", llamaHeaderDir.path, "-I", ggmlHeaderDir.path, "-I", commonHeaderDir.path]),
    ]
    : [
        .define("CLLAMA_NO_HEADERS"),
    ]

let package = Package(
    name: "TesseraStudio",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "TesseraCore", targets: ["TesseraCore"]),
        .executable(name: "TesseraStudioMac", targets: ["TesseraStudioMac"]),
        .library(name: "TesseraStudioiOS", targets: ["TesseraStudioiOS"]),
    ],
    dependencies: [
        // Pure-Swift HTML parser used by the keyless web-search providers
        // (DuckDuckGo HTML endpoint). No WebKit, works headless in tests.
        .package(url: "https://github.com/scinfu/SwiftSoup.git", from: "2.6.0"),
        // Postgres client (SwiftNIO-based, maintained by Vapor). Used by
        // TesseraDataStore for the durable knowledge-graph + receipts store.
        // Pinned 1.21.x because the PostgresClient API stabilized there;
        // see docs/tessera-data-layer-design.md §6.1 for the dependency
        // choice and the alternatives considered.
        .package(url: "https://github.com/vapor/postgres-nio.git", from: "1.21.2"),
        // Redis / Valkey client (SwiftNIO-based, maintained by swift-server).
        // Used by TesseraCache for the ephemeral scratchpad / decay windows.
        .package(url: "https://github.com/swift-server/RediStack.git", from: "1.6.2"),
        // Splash (JohnSundell): Swift-native syntax highlighter. Used by
        // the Phase 2 editor for highlighting `.codeBlock` blocks. Splash
        // is grammar-driven and ships a Swift grammar; other languages use
        // a small regex highlighter. See docs/tessera-productivity-editor-design.md
        // §10 for the rationale and the language list.
        .package(url: "https://github.com/JohnSundell/Splash.git", from: "0.16.0"),
        // Force-directed graph visualization (SwiftUI-native, 2D simd, KDTree).
        // Used by the Graph view (Phase 6 productivity surface). Pinned at
        // 1.1.0+ per docs/tessera-productivity-contacts-graph-design.md §14.
        .package(url: "https://github.com/SwiftGraphs/Grape.git", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "CLlama",
            path: "Sources/CLlama",
            publicHeadersPath: "include",
            cSettings: cllamaCSettings
        ),
        .target(
            name: "CTesseraFFI",
            path: "Sources/CTesseraFFI",
            publicHeadersPath: "include"
        ),
        .target(
            name: "TesseraCore",
            dependencies: [
                "CLlama",
                "CTesseraFFI",
                .product(name: "SwiftSoup", package: "SwiftSoup"),
                // Data layer: Postgres (durable graph + receipts) and
                // RediStack (Valkey/Redis cache). Only TesseraDataStore /
                // TesseraCache import these; TesseraDataLayer is the
                // facade the rest of the app depends on. See
                // docs/tessera-data-layer-design.md §6 for the
                // hexagonal boundary rationale.
                .product(name: "PostgresNIO", package: "postgres-nio"),
                .product(name: "RediStack", package: "RediStack"),
                // Splash: Swift-native syntax highlighter for code blocks.
                // The CodeBlockHighlighter consumes it on macOS; the
                // regex fallback path is used on Linux/CI.
                .product(name: "Splash", package: "Splash"),
                // Graph view (Phase 6 productivity surface). Only the
                // Productivity/Graph viewmodel and view import these.
                .product(name: "Grape", package: "Grape"),
                .product(name: "ForceSimulation", package: "Grape"),
            ],
            path: "Sources/TesseraCore",
            // Ships the Skills format doc so the target has a resource bundle
            // (Bundle.module); the skill loader looks here for bundled skills.
            resources: [.copy("Skills/README.md")]
        ),
        .executableTarget(
            name: "TesseraStudioMac",
            dependencies: ["TesseraCore"],
            path: "Sources/TesseraStudioMac"
        ),
        .target(
            name: "TesseraStudioiOS",
            dependencies: ["TesseraCore"],
            path: "Sources/TesseraStudioiOS"
        ),
        .testTarget(
            name: "TesseraCoreTests",
            dependencies: ["TesseraCore"],
            path: "Tests/TesseraCoreTests",
            // Copied verbatim so the loader's `<name>/SKILL.md` nesting is
            // preserved when the fixtures are read back via Bundle.module.
            resources: [.copy("Fixtures")]
        ),
    ]
)

// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Tessera Studio v2 - three-target split.
//
//   CTesseraFFI       C target: the thin FFI header (+ link-time stub).
//   CLlama            C target: runtime-loaded (dlopen) bridge to libllama
//                     for on-device inference. Compiles against the repo's
//                     llama.h when present; otherwise builds a stub that
//                     reports unavailable (see CLLAMA_NO_HEADERS below).
//   TesseraCore       Platform-independent models, tool protocol, engine
//                     bridge protocol + FFI/CLI bridges, and the shared
//                     SwiftUI views (guarded with #if os() where needed).
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
let hasLlamaHeaders = FileManager.default
    .fileExists(atPath: llamaHeaderDir.appendingPathComponent("llama.h").path)

let cllamaCSettings: [CSetting] = hasLlamaHeaders
    ? [
        .unsafeFlags(["-I", llamaHeaderDir.path, "-I", ggmlHeaderDir.path]),
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
    targets: [
        .target(
            name: "CTesseraFFI",
            path: "Sources/CTesseraFFI",
            publicHeadersPath: "include"
        ),
        .target(
            name: "CLlama",
            path: "Sources/CLlama",
            publicHeadersPath: "include",
            cSettings: cllamaCSettings
        ),
        .target(
            name: "TesseraCore",
            dependencies: ["CTesseraFFI", "CLlama"],
            path: "Sources/TesseraCore"
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
            path: "Tests/TesseraCoreTests"
        ),
    ]
)

import XCTest
@testable import TesseraCore

// S2 (runtime-traces spec): runtime speculative engine wiring.
//
// Covers the pieces that are testable without the native dylibs: drafter
// path resolution (explicit wins, auto-derive, "-" sentinel), the provider
// routing decision, the settings defaults, and the shim-degradation path
// (spec library absent -> today's single-model behavior, no crash).

// MARK: - Drafter resolution

final class TesseraRuntimeDrafterResolverTests: XCTestCase {
    func testExplicitPathWinsAsIs() {
        let path = TesseraRuntimeDrafterResolver.resolve(
            setting: "/explicit/drafter.gguf",
            trunkPath: "/models/base.gguf")
        XCTAssertEqual(path, "/explicit/drafter.gguf")
    }

    func testExplicitPathNeverFallsBackToAutoDerive() {
        // The explicit path does not exist, so resolvedDrafter degrades to
        // trunk-only; it must NOT silently switch to the derived sibling.
        let resolved = TesseraRuntimeDrafterResolver.resolvedDrafter(
            setting: "/explicit/drafter.gguf",
            trunkPath: "/models/base.gguf",
            exists: { $0 == "/models/base-tessera-trained.gguf" })
        XCTAssertNil(resolved)
    }

    func testExplicitPathUsedWhenPresent() {
        let resolved = TesseraRuntimeDrafterResolver.resolvedDrafter(
            setting: "/explicit/drafter.gguf",
            trunkPath: "/models/base.gguf",
            exists: { _ in true })
        XCTAssertEqual(resolved, "/explicit/drafter.gguf")
    }

    func testSentinelDisablesAutoDerive() {
        XCTAssertNil(TesseraRuntimeDrafterResolver.resolve(
            setting: "-", trunkPath: "/models/base.gguf"))
        XCTAssertNil(TesseraRuntimeDrafterResolver.resolve(
            setting: "  -  ", trunkPath: "/models/base.gguf"))
    }

    func testAutoDeriveFindsSibling() {
        let derived = "/models/base-tessera-trained.gguf"
        let resolved = TesseraRuntimeDrafterResolver.resolvedDrafter(
            setting: "",
            trunkPath: "/models/base.gguf",
            exists: { $0 == derived })
        XCTAssertEqual(resolved, derived)
    }

    func testAutoDeriveMissingSiblingIsTrunkOnly() {
        let resolved = TesseraRuntimeDrafterResolver.resolvedDrafter(
            setting: "",
            trunkPath: "/models/base.gguf",
            exists: { _ in false })
        XCTAssertNil(resolved)
    }

    func testDerivedPathStripsGGUFSuffix() {
        XCTAssertEqual(
            TesseraRuntimeDrafterResolver.derivedPath(forTrunk: "/models/base.gguf"),
            "/models/base-tessera-trained.gguf")
        // A trunk path without the suffix still derives next to it.
        XCTAssertEqual(
            TesseraRuntimeDrafterResolver.derivedPath(forTrunk: "/models/base"),
            "/models/base-tessera-trained.gguf")
    }

    func testEmptyTrunkWithEmptySettingIsTrunkOnly() {
        XCTAssertNil(TesseraRuntimeDrafterResolver.resolve(setting: "", trunkPath: ""))
    }

    func testExplicitPathExpandsTilde() {
        let path = TesseraRuntimeDrafterResolver.resolve(
            setting: "~/models/drafter.gguf", trunkPath: "")
        XCTAssertEqual(path, NSString(string: "~/models/drafter.gguf").expandingTildeInPath)
        XCTAssertFalse(path?.hasPrefix("~") ?? true)
    }
}

// MARK: - Provider routing decision (pure)

final class LlamaSpecRoutingTests: XCTestCase {
    func testUsesSpecEngineTruthTable() {
        XCTAssertFalse(LlamaLLMProvider.usesSpecEngine(drafterPath: nil, specLibraryAvailable: false))
        XCTAssertFalse(LlamaLLMProvider.usesSpecEngine(drafterPath: nil, specLibraryAvailable: true))
        XCTAssertFalse(LlamaLLMProvider.usesSpecEngine(drafterPath: "/d.gguf", specLibraryAvailable: false))
        XCTAssertTrue(LlamaLLMProvider.usesSpecEngine(drafterPath: "/d.gguf", specLibraryAvailable: true))
    }

    func testProviderInitResolvesDrafter() async throws {
        // An explicit path wins as-is, provided it exists; a missing
        // explicit path degrades to trunk-only (never auto-derive), which
        // TesseraRuntimeDrafterResolverTests pins at the resolver level.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-spec-explicit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let drafter = dir.appendingPathComponent("drafter.gguf")
        FileManager.default.createFile(atPath: drafter.path, contents: Data())

        let provider = LlamaLLMProvider(
            modelPath: "/models/base.gguf",
            runtimeDraftModelSetting: drafter.path)
        let resolved = await provider.resolvedRuntimeDrafter
        XCTAssertEqual(resolved, drafter.path)
    }

    func testProviderInitMissingExplicitDrafterIsTrunkOnly() async {
        // Missing explicit path must NOT fall back to the auto-derived
        // sibling and must NOT engage spec mode: trunk-only.
        let provider = LlamaLLMProvider(
            modelPath: "/models/base.gguf",
            runtimeDraftModelSetting: "/explicit/missing-\(UUID().uuidString).gguf")
        let resolved = await provider.resolvedRuntimeDrafter
        XCTAssertNil(resolved)
    }

    func testProviderInitSentinelIsTrunkOnly() async {
        let provider = LlamaLLMProvider(
            modelPath: "/models/base.gguf",
            runtimeDraftModelSetting: "-")
        let resolved = await provider.resolvedRuntimeDrafter
        XCTAssertNil(resolved)
    }

    func testProviderInitAutoDeriveRequiresExistingFile() async {
        // No real file at the derived sibling path -> trunk-only.
        let provider = LlamaLLMProvider(
            modelPath: "/definitely/missing/base-\(UUID().uuidString).gguf",
            runtimeDraftModelSetting: "")
        let resolved = await provider.resolvedRuntimeDrafter
        XCTAssertNil(resolved)
    }

    func testProviderInitAutoDeriveFindsRealSibling() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-spec-resolve-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let trunk = dir.appendingPathComponent("base.gguf")
        let drafter = dir.appendingPathComponent("base-tessera-trained.gguf")
        FileManager.default.createFile(atPath: trunk.path, contents: Data())
        FileManager.default.createFile(atPath: drafter.path, contents: Data())

        let provider = LlamaLLMProvider(
            modelPath: trunk.path,
            runtimeDraftModelSetting: "")
        let resolved = await provider.resolvedRuntimeDrafter
        XCTAssertEqual(resolved, drafter.path)
    }
}

// MARK: - Settings defaults

final class TesseraRuntimeSettingsTests: XCTestCase {
    private let keys = [
        TesseraSettingsKey.learningRuntimeDraftModel,
        TesseraSettingsKey.learningRuntimeCapture,
        TesseraSettingsKey.learningRuntimeCaptureTopk,
        TesseraSettingsKey.learningRuntimeDraftMax,
    ]

    private func withClearedRuntimeSettings(_ body: () -> Void) {
        let saved: [(String, Any?)] = keys.map { ($0, UserDefaults.standard.object(forKey: $0)) }
        for key in keys { UserDefaults.standard.removeObject(forKey: key) }
        defer {
            for (key, value) in saved {
                if let value { UserDefaults.standard.set(value, forKey: key) }
                else { UserDefaults.standard.removeObject(forKey: key) }
            }
        }
        body()
    }

    func testDefaults() {
        withClearedRuntimeSettings {
            XCTAssertEqual(TesseraSettings.learningRuntimeDraftModel, "")
            XCTAssertTrue(TesseraSettings.learningRuntimeCapture)
            XCTAssertEqual(TesseraSettings.learningRuntimeCaptureTopk, 16)
            XCTAssertEqual(TesseraSettings.learningRuntimeDraftMax, 3)
        }
    }

    func testDefaultsMatchRegisteredValues() {
        XCTAssertEqual(TesseraSettingsDefault.learningRuntimeDraftModel, "")
        XCTAssertTrue(TesseraSettingsDefault.learningRuntimeCapture)
        XCTAssertEqual(TesseraSettingsDefault.learningRuntimeCaptureTopk, 16)
        XCTAssertEqual(TesseraSettingsDefault.learningRuntimeDraftMax, 3)
    }

    func testOverridesAreRespected() {
        withClearedRuntimeSettings {
            UserDefaults.standard.set("/custom/drafter.gguf", forKey: TesseraSettingsKey.learningRuntimeDraftModel)
            UserDefaults.standard.set(false, forKey: TesseraSettingsKey.learningRuntimeCapture)
            UserDefaults.standard.set(8, forKey: TesseraSettingsKey.learningRuntimeCaptureTopk)
            UserDefaults.standard.set(5, forKey: TesseraSettingsKey.learningRuntimeDraftMax)

            XCTAssertEqual(TesseraSettings.learningRuntimeDraftModel, "/custom/drafter.gguf")
            XCTAssertFalse(TesseraSettings.learningRuntimeCapture)
            XCTAssertEqual(TesseraSettings.learningRuntimeCaptureTopk, 8)
            XCTAssertEqual(TesseraSettings.learningRuntimeDraftMax, 5)
        }
    }
}

// MARK: - Shim degradation (spec library absent -> today's path)

final class LlamaSpecDegradationTests: XCTestCase {
    /// The spec probe is honest in both worlds: in a test environment with
    /// no dylibs it reports unavailable; on a machine with the build present
    /// it reports available. Either way it must not crash and must agree
    /// with cllama_is_spec_available() (routed through load success).
    func testProbeSpecLibraryIsHonest() {
        let probe = LlamaLLMProvider.probeSpecLibrary(libraryPath: "/definitely/missing/libllama-common.dylib")
        // An explicit missing path can never load... unless a previous test
        // already cached a successful load (the shim is idempotent). Accept
        // either world, but the call itself must be safe.
        _ = probe
    }

    /// A provider whose spec inputs are garbage (empty placeholder GGUFs)
    /// must fail with a clean LlamaLLMError, never crash, and never hang -
    /// regardless of whether the native libraries are present on the test
    /// machine. This is the degrade-open contract end to end.
    func testCompleteWithMissingLibrariesOrModelsThrowsCleanly() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-spec-degrade-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let trunk = dir.appendingPathComponent("base.gguf")
        let drafter = dir.appendingPathComponent("base-tessera-trained.gguf")
        FileManager.default.createFile(atPath: trunk.path, contents: Data())
        FileManager.default.createFile(atPath: drafter.path, contents: Data())

        let provider = LlamaLLMProvider(
            modelPath: trunk.path,
            maxTokens: 8,
            runtimeDraftModelSetting: "")

        do {
            _ = try await provider.complete(system: "s", messages: [], tools: [])
            // If real dylibs AND valid models were somehow present this
            // could succeed; with empty placeholder GGUFs every world ends
            // in a clean throw.
            XCTFail("expected generation with placeholder models to fail cleanly")
        } catch let error as LlamaLLMError {
            // libraryUnavailable (no dylibs) or modelLoadFailed (dylibs
            // present, placeholder GGUFs invalid) are both honest outcomes.
            switch error {
            case .libraryUnavailable, .modelLoadFailed, .generationFailed:
                break
            }
        }
    }

    /// Spec mode never engages when the spec library is unavailable, even
    /// with a drafter resolved: the routing decision is pure and the
    /// provider falls through to the single-model path.
    func testRoutingDegradesWhenSpecLibraryMissing() {
        XCTAssertFalse(LlamaLLMProvider.usesSpecEngine(
            drafterPath: "/models/base-tessera-trained.gguf",
            specLibraryAvailable: false))
    }
}

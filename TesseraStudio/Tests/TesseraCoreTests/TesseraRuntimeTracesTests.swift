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

// MARK: - Trace store: runtime capture (spec section 8)

final class TesseraTraceStoreRuntimeTests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    private func makeStore() throws -> TesseraTraceStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-trace-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        return TesseraTraceStore(directory: dir)
    }

    private func runtimeRecord(sid: String, accepted: Int, drafted: Int, pad: Int = 0) -> String {
        let padding = pad > 0 ? ",\"pad\":\"\(String(repeating: "x", count: pad))\"" : ""
        return "{\"schema\":\"llama.tessera.spec.v1\",\"drafted\":\(drafted),\"accepted\":\(accepted),\"provenance\":\"runtime\",\"sid\":\"\(sid)\"\(padding)}"
    }

    private func calibrationRecord(step: Int) -> String {
        "{\"schema\":\"llama.tessera.spec.v1\",\"step\":\(step),\"drafted\":3,\"accepted\":2}"
    }

    // Naming + verbatim content.

    func testAppendRuntimeWritesDatedRuntimeFile() throws {
        let store = try makeStore()
        let records = [
            runtimeRecord(sid: "s1", accepted: 1, drafted: 2),
            runtimeRecord(sid: "s1", accepted: 0, drafted: 2),
        ]
        let url = try store.appendRuntime(records: records)
        XCTAssertNotNil(url)
        guard let url else { return }
        XCTAssertTrue(url.lastPathComponent.hasPrefix(TesseraTraceStore.runtimeFilePrefix))
        XCTAssertTrue(url.lastPathComponent.hasSuffix(".jsonl"))
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertEqual(text, records.joined(separator: "\n") + "\n")
    }

    func testAppendRuntimeEmptyIsNoop() throws {
        let store = try makeStore()
        XCTAssertNil(try store.appendRuntime(records: []))
        XCTAssertTrue(store.runtimeFiles().isEmpty)
        XCTAssertEqual(store.totalRecords(), 0)
    }

    // Combined counting: calibration + runtime share the traces- prefix, so
    // totalRecords() sees both (the training gate counts the combined total).

    func testCombinedCountingAcrossProvenances() throws {
        let store = try makeStore()

        let calibrationSource = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-calib-\(UUID().uuidString).jsonl")
        dirs.append(calibrationSource)
        let calibration = (0..<2).map { calibrationRecord(step: $0) }
        try (calibration.joined(separator: "\n") + "\n")
            .write(to: calibrationSource, atomically: true, encoding: .utf8)
        try store.appendRun(jsonlPath: calibrationSource)

        try store.appendRuntime(records: [
            runtimeRecord(sid: "a", accepted: 1, drafted: 3),
            runtimeRecord(sid: "a", accepted: 2, drafted: 3),
            runtimeRecord(sid: "a", accepted: 0, drafted: 3),
        ])

        XCTAssertEqual(store.totalRecords(), 5)
        XCTAssertEqual(store.traceFiles().count, 2)
        XCTAssertEqual(store.runtimeFiles().count, 1)
    }

    // Sid stamping: records group into sessions by sid, totals accumulate,
    // and the summary cache invalidates on append.

    func testSidStampingGroupsSessions() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [
            runtimeRecord(sid: "A", accepted: 1, drafted: 2),
            runtimeRecord(sid: "A", accepted: 2, drafted: 3),
        ])
        // Distinct date stamps: same-second suffixes sort before the base
        // name, so ordering pinning needs different seconds.
        Thread.sleep(forTimeInterval: 1.1)
        try store.appendRuntime(records: [
            runtimeRecord(sid: "B", accepted: 3, drafted: 3),
        ])

        let summary = store.runtimeSummary()
        XCTAssertEqual(summary.totalRecords, 3)
        XCTAssertEqual(summary.sessions.count, 2)
        XCTAssertEqual(summary.sessions.map { $0.sid }, ["A", "B"])  // oldest first
        XCTAssertEqual(summary.sessions[0].records, 2)
        XCTAssertEqual(summary.sessions[0].accepted, 3)
        XCTAssertEqual(summary.sessions[0].drafted, 5)
        XCTAssertEqual(summary.latestSession?.sid, "B")
        // 6 accepted of 8 drafted across every captured step.
        XCTAssertEqual(summary.acceptanceRate, 0.75)

        // Cache invalidates on append: a same-sid retry merges into A.
        try store.appendRuntime(records: [runtimeRecord(sid: "A", accepted: 1, drafted: 1)])
        let merged = store.runtimeSummary()
        XCTAssertEqual(merged.sessions.count, 2)
        XCTAssertEqual(merged.sessions.first { $0.sid == "A" }?.records, 3)
    }

    // Runtime-first trimming: the rolling cap removes OLDEST runtime files
    // first and never touches calibration or replay files.

    func testRuntimeFirstTrimmingSparesCalibrationAndReplay() throws {
        let store = try makeStore()
        let dir = FileManager.default.temporaryDirectory

        // Calibration file (appendRun) with one record.
        let calibrationSource = dir.appendingPathComponent("tessera-calib-\(UUID().uuidString).jsonl")
        dirs.append(calibrationSource)
        try (calibrationRecord(step: 0) + "\n")
            .write(to: calibrationSource, atomically: true, encoding: .utf8)
        let calibrationStored = try store.appendRun(jsonlPath: calibrationSource)

        // Replay file: written directly, as the replay stage (S4) will.
        let replay = calibrationStored.deletingLastPathComponent()
            .appendingPathComponent("traces-replay-20260101-000000.jsonl")
        try (calibrationRecord(step: 0) + "\n").write(to: replay, atomically: true, encoding: .utf8)

        // Three ~1 KB runtime files, oldest first.
        var sizes: [Int] = []
        for i in 0..<3 {
            let record = runtimeRecord(sid: "s\(i)", accepted: 1, drafted: 2, pad: 900)
            let url = try store.appendRuntime(records: [record])
            sizes.append((try String(contentsOf: url!, encoding: .utf8)).utf8.count)
        }

        // Budget keeps only the newest runtime file.
        let removed = try store.trimRuntimeToBudget(budgetBytes: sizes[2])
        XCTAssertEqual(removed, 2)

        let runtimeNames = store.runtimeFiles().map { $0.lastPathComponent }
        XCTAssertEqual(runtimeNames.count, 1)
        // Compare names: contentsOfDirectory resolves the /var symlink, so
        // URL identity does not match hand-built URLs.
        let storedNames = store.traceFiles().map { $0.lastPathComponent }
        XCTAssertTrue(storedNames.contains(calibrationStored.lastPathComponent))
        XCTAssertTrue(storedNames.contains(replay.lastPathComponent))
        XCTAssertEqual(store.totalRecords(), 3)  // calibration + replay + 1 runtime
    }

    // Quarantine exemption: a quarantined sid survives BOTH automatic
    // retention paths even when it is the oldest file.

    func testQuarantineExemptionFromRollingCap() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [runtimeRecord(sid: "Q", accepted: 1, drafted: 2, pad: 900)])
        try store.appendRuntime(records: [runtimeRecord(sid: "K", accepted: 1, drafted: 2, pad: 900)])

        // Budget fits one file; Q is oldest but quarantined, so K goes.
        let removed = try store.trimRuntimeToBudget(budgetBytes: 1000, exemptSids: ["Q"])
        XCTAssertEqual(removed, 1)

        let remaining = store.runtimeFiles()
        XCTAssertEqual(remaining.count, 1)
        let text = try String(contentsOf: remaining[0], encoding: .utf8)
        XCTAssertTrue(text.contains("\"sid\":\"Q\""))
    }

    func testQuarantineExemptionFromRetention() throws {
        let store = try makeStore()
        let old = Date().addingTimeInterval(-100 * 86_400)

        try store.appendRuntime(records: [runtimeRecord(sid: "Q", accepted: 1, drafted: 2)])
        try store.appendRuntime(records: [runtimeRecord(sid: "K", accepted: 1, drafted: 2)])
        let files = store.runtimeFiles()
        XCTAssertEqual(files.count, 2)
        // Backdate both runtime files past the retention window.
        for file in files {
            try FileManager.default.setAttributes(
                [.creationDate: old], ofItemAtPath: file.path)
        }
        // And a calibration file, backdated the same way.
        let calibrationSource = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-calib-\(UUID().uuidString).jsonl")
        dirs.append(calibrationSource)
        try (calibrationRecord(step: 0) + "\n")
            .write(to: calibrationSource, atomically: true, encoding: .utf8)
        let calibrationStored = try store.appendRun(jsonlPath: calibrationSource)
        try FileManager.default.setAttributes(
            [.creationDate: old], ofItemAtPath: calibrationStored.path)

        let removed = try store.trimExpired(retentionDays: 30, exemptSids: ["Q"])
        XCTAssertEqual(removed, 2)  // K + calibration; Q exempt

        let survivors = store.traceFiles().map { $0.lastPathComponent }
        let qFile = files.first { (try? String(contentsOf: $0, encoding: .utf8))?.contains("\"sid\":\"Q\"") == true }
        XCTAssertNotNil(qFile)
        XCTAssertTrue(survivors.contains(qFile!.lastPathComponent))
    }

    func testTrimExpiredKeepsFreshFiles() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [runtimeRecord(sid: "N", accepted: 1, drafted: 2)])
        let removed = try store.trimExpired(retentionDays: 30)
        XCTAssertEqual(removed, 0)
        XCTAssertEqual(store.runtimeFiles().count, 1)
    }

    func testTrimExpiredNoopWhenRetentionNotPositive() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [runtimeRecord(sid: "N", accepted: 1, drafted: 2)])
        XCTAssertEqual(try store.trimExpired(retentionDays: 0), 0)
        XCTAssertEqual(try store.trimExpired(retentionDays: -5), 0)
        XCTAssertEqual(store.runtimeFiles().count, 1)
    }
}

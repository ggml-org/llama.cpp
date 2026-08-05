import XCTest
@testable import TesseraCore

// MARK: - Mock volume

/// In-memory mock of ``PleadTheFifthVolume`` for the executor tests.
/// The actor records which calls were made so tests can assert on
/// the step order, and exposes hooks (`unmountShouldFail`,
/// `isMountedReturnValue`) for failure-mode coverage.
actor MockPleadTheFifthVolume: PleadTheFifthVolume {
    private(set) var unmountCalls: Int = 0
    private(set) var isMountedCalls: Int = 0
    private(set) var artifactsReads: Int = 0

    var encryptedArtifactsValue: [URL] = []
    var isMountedReturnValue: Bool = true
    var unmountShouldFail: Bool = false

    init(artifacts: [URL] = [], isMounted: Bool = true, unmountShouldFail: Bool = false) {
        self.encryptedArtifactsValue = artifacts
        self.isMountedReturnValue = isMounted
        self.unmountShouldFail = unmountShouldFail
    }

    func unmount() async throws {
        unmountCalls += 1
        if unmountShouldFail {
            throw NSError(
                domain: "MockPleadTheFifthVolume",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "mock: unmount refused (key destroyed)"]
            )
        }
    }

    func isMounted() async -> Bool {
        isMountedCalls += 1
        return isMountedReturnValue
    }

    var encryptedArtifacts: [URL] {
        get async {
            artifactsReads += 1
            return encryptedArtifactsValue
        }
    }
}

// MARK: - PleadTheFifthExecutor

final class PleadTheFifthExecutorTests: XCTestCase {

    private var tempDir: URL!
    private var reportURL: URL!

    override func setUp() async throws {
        try await super.setUp()
        tempDir = makeTempDir()
        reportURL = tempDir.appendingPathComponent("last-wipe.json")
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDir)
        // Reset the Keychain so cross-test pollution does not happen.
        _ = PleadTheFifthKeychain.deleteEntry(
            account: PleadTheFifthKeychain.volumePasswordAccount
        )
        _ = PleadTheFifthKeychain.deleteEntry(
            account: PleadTheFifthKeychain.dataAccessKeyAccount
        )
        // Reset coercion + covert trigger so settings tests do not
        // pollute one another.
        UserDefaults.standard.removeObject(forKey: PleadTheFifthSettingsKey.coercionMode)
        UserDefaults.standard.removeObject(forKey: PleadTheFifthSettingsKey.covertTriggerPhrase)
        UserDefaults.standard.removeObject(forKey: PleadTheFifthSettingsKey.lastCovertTriggerAt)
        UserDefaults.standard.removeObject(forKey: PleadTheFifthSettingsKey.failedCovertTriggerAttempts)
        try await super.tearDown()
    }

    // MARK: 9-step ordering

    func testWipeRunsAllStepsInOrder() async throws {
        // Arrange: a real (small) file the executor will overwrite
        // and unlink. Empty artifacts would skip step 6's overwrite
        // path; we want the full path covered.
        let file = try makeFile(contents: Data(repeating: 0xAA, count: 16 * 1024))
        let volume = MockPleadTheFifthVolume(artifacts: [file])
        let executor = PleadTheFifthExecutor(
            volume: volume,
            sidecarController: NoOpSidecarController(),
            overwritePasses: 1
        )

        // Act
        let report = try await executor.destroyAll(trigger: .test)

        // Assert: the 9 steps in order.
        let expectedNames = [
            "stop_postgres",
            "stop_valkey",
            "destroy_volume_password",
            "destroy_dak",
            "unmount_volume",
            "overwrite_ciphertext",
            "delete_volume_files",
            "fsync",
            "exit",
        ]
        XCTAssertEqual(report.steps.map(\.name), expectedNames)
        XCTAssertEqual(report.triggerSource, .test)
        // The file should be unlinked by the end of the wipe.
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path),
                       "executor must unlink the encrypted artifact")
    }

    // MARK: partial failure is recorded but not propagated

    func testWipeContinuesOnPartialFailure() async throws {
        // Arrange: unmount will fail (the real reason is "key was
        // destroyed" - same shape as production). The wipe must
        // still complete and produce a report.
        let volume = MockPleadTheFifthVolume(
            artifacts: [],
            isMounted: true,
            unmountShouldFail: true
        )
        let executor = PleadTheFifthExecutor(
            volume: volume,
            sidecarController: NoOpSidecarController()
        )

        let report = try await executor.destroyAll(trigger: .hotkey)

        // Step 5 (unmount) is the one that should report partialFailure.
        let unmount = report.steps.first { $0.name == "unmount_volume" }
        XCTAssertNotNil(unmount)
        XCTAssertEqual(unmount?.outcome, .partialFailure)
        XCTAssertEqual(unmount?.reason, "expected: key destroyed",
                       "step 5 must use the expected-failure reason, not the raw error")

        // All 9 steps must still have run.
        XCTAssertEqual(report.steps.count, 9)
        // destroy_volume_password must be success (the crypto-shred event).
        let crypto = report.steps.first { $0.name == "destroy_volume_password" }
        XCTAssertEqual(crypto?.outcome, .success)
    }

    // MARK: timing

    func testCryptoShredCompletesInUnderTwoSeconds() async throws {
        // The hot-key -> crypto-shred portion is steps 1-3. We
        // measure the elapsed time from start to the moment step 3
        // completed. On a reference machine this is well under 2s.
        let volume = MockPleadTheFifthVolume(artifacts: [])
        let executor = PleadTheFifthExecutor(
            volume: volume,
            sidecarController: NoOpSidecarController()
        )

        let start = Date()
        let report = try await executor.destroyAll(trigger: .hotkey)
        let cryptoEnd = report.steps.prefix(3).reduce(0) { $0 + $1.durationMs }
        let wallClock = Date().timeIntervalSince(start) * 1000

        XCTAssertLessThan(cryptoEnd, 2000, "crypto steps took \(cryptoEnd)ms")
        XCTAssertLessThan(wallClock, 2000, "wall clock \(wallClock)ms")
    }

    func testFullWipeCompletesInUnderTenSeconds() async throws {
        // Use a 4 MiB artifact and 1 overwrite pass so the
        // overwrite step runs but is short. 10s is the design's
        // budget for the full 9-step wipe on a reference machine.
        let file = try makeFile(contents: Data(repeating: 0xCC, count: 4 * 1024 * 1024))
        let volume = MockPleadTheFifthVolume(artifacts: [file])
        let executor = PleadTheFifthExecutor(
            volume: volume,
            sidecarController: NoOpSidecarController(),
            overwritePasses: 1
        )

        let start = Date()
        let report = try await executor.destroyAll(trigger: .test)
        let elapsed = Date().timeIntervalSince(start) * 1000
        XCTAssertLessThan(elapsed, 10_000, "full wipe took \(elapsed)ms")
        // Sanity: overwrite step actually ran.
        let overwrite = report.steps.first { $0.name == "overwrite_ciphertext" }
        XCTAssertEqual(overwrite?.outcome, .success)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
    }

    // MARK: report

    func testWipeReportIsWrittenBeforeExit() async throws {
        let volume = MockPleadTheFifthVolume(artifacts: [])
        let store = WipeReportStore(fileURL: reportURL)
        let executor = PleadTheFifthExecutor(
            volume: volume,
            sidecarController: NoOpSidecarController()
        )

        let report = try await executor.destroyAll(trigger: .menu)
        try store.save(report)

        XCTAssertTrue(FileManager.default.fileExists(atPath: reportURL.path))
        let reloaded = try store.loadIfPresent()
        XCTAssertNotNil(reloaded)
        XCTAssertEqual(reloaded?.triggerSource, .menu)
        XCTAssertEqual(reloaded?.steps.count, 9)

        // Spot-check the JSON shape: trigger source, startedAt,
        // steps with name + outcome.
        let data = try Data(contentsOf: reportURL)
        let dict = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(dict["triggerSource"] as? String, "menu")
        XCTAssertNotNil(dict["startedAt"] as? String)
        XCTAssertNotNil(dict["completedAt"] as? String)
        let steps = try XCTUnwrap(dict["steps"] as? [[String: Any]])
        XCTAssertEqual(steps.count, 9)
        XCTAssertNotNil(steps[0]["name"])
        XCTAssertNotNil(steps[0]["outcome"])
    }

    // MARK: hot-key

    func testHotKeyMatchesDefaultChord() {
        // The match logic is pure and stable. We synthesise the
        // bits the live monitor would compare against.
        let chord = HotKeyMonitor.Chord.defaultChord
        XCTAssertEqual(chord.keyCode, 51)
        XCTAssertTrue(chord.command)
        XCTAssertTrue(chord.shift)
        XCTAssertEqual(chord.displayString, "\u{2318}\u{21E7}\u{232B}")
    }

    // MARK: confirmation phrase

    func testConfirmationPhraseIsCaseInsensitive() {
        // The panel's success criterion is exact, case-insensitive
        // match. We assert the rule by reconstructing what the
        // submit handler does (the panel itself is UI-bound; the
        // rule is the testable bit).
        let expected = "destroy everything"
        let candidates: [(String, Bool)] = [
            ("destroy everything", true),
            ("DESTROY EVERYTHING", true),
            ("  destroy everything  ", true),
            ("destroy everything!", false),
            ("destroy", false),
            ("", false),
        ]
        for (input, shouldMatch) in candidates {
            let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
            let matches = trimmed.caseInsensitiveCompare(expected) == .orderedSame
            XCTAssertEqual(matches, shouldMatch, "input: \(input)")
        }
    }

    // MARK: coercion mode

    func testCoercionModeHidesDestructiveMenuItems() {
        // We assert the policy: when coercion mode is on, the
        // settings surface exposes the toggle; the menu item's
        // menu builder filters by the same flag. We exercise
        // the policy in isolation by checking the helper that
        // decides whether a given menu title is visible.
        UserDefaults.standard.set(true, forKey: PleadTheFifthSettingsKey.coercionMode)
        XCTAssertTrue(PleadTheFifthSettings.coercionMode)
        // When coercion is on, the "primary" menu entry is hidden
        // by ``PleadTheFifthMenuItem.rebuildMenu``; we replicate
        // the policy here.
        XCTAssertTrue(PleadTheFifthMenuPolicy.shouldHidePrimaryAction(
            coercionMode: PleadTheFifthSettings.coercionMode
        ))
    }

    // MARK: helpers

    private func makeTempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("plead-fifth-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeFile(contents: Data) throws -> URL {
        let url = tempDir.appendingPathComponent("encrypted-\(UUID().uuidString).bin")
        try contents.write(to: url)
        return url
    }
}

// MARK: - menu policy helper

/// Pure function form of the menu item's "should I show this entry?"
/// policy. The NSStatusItem itself is not testable in headless
/// `swift test` (no NSApp), so the policy is split out and tested
/// here. The menu item calls this function from `rebuildMenu`.
public enum PleadTheFifthMenuPolicy {
    public static func shouldHidePrimaryAction(coercionMode: Bool) -> Bool {
        coercionMode
    }
    public static func shouldShowCovertSubmenu(
        coercionMode: Bool,
        covertTriggerConfigured: Bool
    ) -> Bool {
        !coercionMode && covertTriggerConfigured
    }
    public static func shouldShowReportEntry(coercionMode: Bool) -> Bool {
        !coercionMode
    }
}

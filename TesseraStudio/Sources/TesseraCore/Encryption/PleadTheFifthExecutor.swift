import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// The 9-step "Plea the Fifth" wipe. The actor runs the documented
/// sequence (see docs/tessera-plead-the-fifth-design.md sections 6.5
/// and 7.1) and produces a ``WipeReport`` for the user's audit trail.
///
/// The crypto-shred property is achieved at step 3 (destroying the
/// volume password in the Keychain). Every step after that is defense
/// in depth; failures are recorded but do not abort the wipe.
public actor PleadTheFifthExecutor {

    public struct WipeReport: Codable, Sendable, Equatable {
        public let startedAt: Date
        public let completedAt: Date
        public let triggerSource: TriggerSource
        public let steps: [WipeStep]

        /// True when every step completed with `success`. Note that
        /// step 5 (unmount) is EXPECTED to report `partialFailure`
        /// after the key is destroyed - that is the design, not a
        /// bug. Callers that care about the "is the data gone" question
        /// should look at step 3 instead of this flag.
        public var succeeded: Bool { steps.allSatisfy { $0.outcome == .success } }
    }

    public enum TriggerSource: String, Codable, Sendable {
        case hotkey
        case menu
        case covert
        case test
    }

    public enum WipeOutcome: String, Codable, Sendable {
        case success
        case partialFailure
        case aborted
    }

    public struct WipeStep: Codable, Sendable, Equatable {
        public let name: String
        public let outcome: WipeOutcome
        public let durationMs: Int
        public let reason: String?
    }

    public enum WipeError: Error, LocalizedError {
        case alreadyRunning

        public var errorDescription: String? {
            switch self {
            case .alreadyRunning: "A wipe is already running."
            }
        }
    }

    private let volume: PleadTheFifthVolume
    private let sidecarController: SidecarController
    private let randomFill: SecureOverwrite.RandomFill
    private let overwritePasses: Int
    private let exitHook: (@Sendable () -> Void)?

    private var inFlight: Bool = false

    /// Production initializer. `exitHook` runs after the report is
    /// built so callers (the menu bar, the hot-key handler) can flush
    /// the report to disk and then ask the OS to terminate the process.
    public init(
        volume: PleadTheFifthVolume,
        sidecarController: SidecarController = NoOpSidecarController(),
        overwritePasses: Int = 3,
        exitHook: (@Sendable () -> Void)? = nil
    ) {
        self.volume = volume
        self.sidecarController = sidecarController
        self.randomFill = SecureOverwrite.defaultRandomFill
        self.overwritePasses = overwritePasses
        self.exitHook = exitHook
    }

    /// Test-only initializer. Exposes the random-fill seam and the
    /// overwrite pass count so tests can stage small files and
    /// deterministic data without touching the real RNG.
    init(
        volume: PleadTheFifthVolume,
        sidecarController: SidecarController,
        randomFill: @escaping SecureOverwrite.RandomFill,
        overwritePasses: Int,
        exitHook: (@Sendable () -> Void)? = nil
    ) {
        self.volume = volume
        self.sidecarController = sidecarController
        self.randomFill = randomFill
        self.overwritePasses = overwritePasses
        self.exitHook = exitHook
    }

    /// Run the wipe. Returns the ``WipeReport`` on completion.
    ///
    /// The executor never throws on a per-step failure - the whole
    /// point is that a partial failure in step 5+ does not stop the
    /// crypto-shred event in step 3. A throw here means the executor
    /// itself could not start (already running).
    public func destroyAll(trigger: TriggerSource) async throws -> WipeReport {
        if inFlight { throw WipeError.alreadyRunning }
        inFlight = true
        defer { inFlight = false }

        let startedAt = Date()
        var steps: [WipeStep] = []
        steps.reserveCapacity(9)

        // Step 1: stop the Postgres sidecar. No-op in v1.
        steps.append(await runStep("stop_postgres") {
            try await self.sidecarController.stopPostgres()
        })

        // Step 2: stop the Valkey sidecar. No-op in v1.
        steps.append(await runStep("stop_valkey") {
            try await self.sidecarController.stopValkey()
        })

        // Step 3: destroy the volume password in Keychain. This is
        // the crypto-shred event. After this returns success, the
        // encrypted data on disk is cryptographically unrecoverable,
        // regardless of what happens to steps 4-9.
        steps.append(await runStep("destroy_volume_password") {
            let ok = PleadTheFifthKeychain.deleteEntry(
                account: PleadTheFifthKeychain.volumePasswordAccount
            )
            if !ok { throw WipeStepError.destroyFailed("volume-password") }
        })

        // Step 4: destroy the DAK (optional, future). Best-effort;
        // missing entry is OK.
        steps.append(await runStep("destroy_dak") {
            _ = PleadTheFifthKeychain.deleteEntry(
                account: PleadTheFifthKeychain.dataAccessKeyAccount
            )
        })

        // Step 5: unmount the volume. Will likely fail because the
        // key is gone - that is the design. We record the failure
        // as `partialFailure` with reason "key destroyed" so the
        // report reader knows it is expected, not a bug.
        steps.append(await runStep(
            "unmount_volume",
            expectedFailureReason: "expected: key destroyed"
        ) {
            let mounted = await self.volume.isMounted()
            if !mounted { throw WipeStepError.notMounted }
            do {
                try await self.volume.unmount()
            } catch {
                throw WipeStepError.unmountFailed("\(error)")
            }
        })

        // Step 6: overwrite the encrypted artifacts. The volume's
        // `encryptedArtifacts` may be empty (volume already detached,
        // or volume file gone) - that is fine, treat as success.
        steps.append(await runStep("overwrite_ciphertext") {
            let artifacts = await self.volume.encryptedArtifacts
            guard !artifacts.isEmpty else { return }
            try SecureOverwrite.randomPassesThenDelete(
                paths: artifacts,
                passes: self.overwritePasses,
                randomFill: self.randomFill
            )
        })

        // Step 7: delete the volume files (the `delete_volume_files`
        // step). The overwrite path already unlinks the file, so
        // this step is a second unlink for any artifacts the
        // overwrite did not handle (e.g. if step 6 was skipped
        // because `encryptedArtifacts` was empty - in that case we
        // still want to unlink any known side files). For v1 the
        // overwrite covers everything; we leave the step in the
        // report so the JSON contract matches the design.
        steps.append(await runStep("delete_volume_files") {
            let artifacts = await self.volume.encryptedArtifacts
            for url in artifacts {
                let path = url.path
                _ = path.withCString { Darwin.unlink($0) }
            }
        })

        // Step 8: fsync the parent directory of the first artifact.
        // This is what makes the unlink durable.
        steps.append(await runStep("fsync") {
            let artifacts = await self.volume.encryptedArtifacts
            guard let first = artifacts.first else { return }
            try SecureOverwrite.fsyncParentDirectory(of: first)
        })

        // Step 9: exit. The actor's job is to run the 8 destructive
        // steps and produce a report. Process termination is the
        // caller's responsibility (the menu item, the hot-key
        // handler) so the report can be written to disk BEFORE
        // exit(0) fires. The exit hook here is an optional seam
        // for any in-actor cleanup (none in v1). The dispatched
        // 100ms grace before exit lives in the caller, not here.
        steps.append(await runStep("exit") {
            self.exitHook?()
        })

        return WipeReport(
            startedAt: startedAt,
            completedAt: Date(),
            triggerSource: trigger,
            steps: steps
        )
    }

    // MARK: - private

    /// Errors the per-step closure raises. Mapped to outcome by
    /// `runStep`; never propagated.
    private enum WipeStepError: Error {
        case destroyFailed(String)
        case notMounted
        case unmountFailed(String)
    }

    /// Run a single step, time it, and translate thrown errors into
    /// `partialFailure` outcomes. `expectedFailureReason` is the reason
    /// string used when the step throws; pass it for steps where a
    /// failure is part of the design (e.g. step 5 unmount after the
    /// key is gone).
    private func runStep(
        _ name: String,
        expectedFailureReason: String? = nil,
        body: @Sendable () async throws -> Void
    ) async -> WipeStep {
        let start = Date()
        do {
            try await body()
            return WipeStep(
                name: name,
                outcome: .success,
                durationMs: elapsedMs(since: start),
                reason: nil
            )
        } catch {
            return WipeStep(
                name: name,
                outcome: .partialFailure,
                durationMs: elapsedMs(since: start),
                reason: expectedFailureReason ?? "\(error)"
            )
        }
    }

    private func elapsedMs(since start: Date) -> Int {
        Int((Date().timeIntervalSince(start) * 1000).rounded())
    }
}

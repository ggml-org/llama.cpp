import Foundation
#if canImport(IOKit.ps)
import IOKit.ps
#endif

/// Fires a recurring idle session-curation pass (runtime-traces spec section
/// 12.5): same envelope as the training sweeps - idle-gated, on-power,
/// resumable. The stage itself degrades open on every missing dependency,
/// so this scheduler only applies the user's envelope switches.
public final class TesseraSessionCurationScheduler: @unchecked Sendable {
    private let stage: TesseraSessionCurationStage
    private let lock = NSLock()
    private var task: Task<Void, Never>?

    /// Terminal hook for each sweep that reaches the stage (including
    /// no-ops), so the UI layer can refresh. Called on an arbitrary thread.
    public var onFinished: (@Sendable (TesseraSessionCurationReport) -> Void)?

    public init(stage: TesseraSessionCurationStage) {
        self.stage = stage
    }

    /// Launch the recurring loop. Idempotent. Same cadence as the training
    /// scheduler: both are idle duties of one flywheel.
    public func start() {
        lock.lock(); defer { lock.unlock() }
        guard task == nil else { return }
        task = Task.detached { [weak self] in
            while !Task.isCancelled {
                let hours = TesseraSettings.learningTrainingIntervalHours
                try? await Task.sleep(nanoseconds: UInt64(hours) * 3_600 * 1_000_000_000)
                guard !Task.isCancelled else { break }
                await self?.sweep()
            }
        }
    }

    public func stop() {
        lock.lock(); defer { lock.unlock() }
        task?.cancel()
        task = nil
    }

    /// One idle curation pass. Returns the sweep report, or nil when a
    /// scheduler-level gate (curation off, on-power requirement) stopped it
    /// before any state was touched.
    @discardableResult
    public func sweep() async -> TesseraSessionCurationReport? {
        guard TesseraSettings.learningSessionCuration else { return nil }
        if TesseraSettings.learningOnPowerOnly && !isOnPower() {
            print("[tessera.curation] idle sweep: skipped, not on power (learning.onPowerOnly=true)")
            return nil
        }
        let report = await stage.sweep()
        print("[tessera.curation] idle sweep: analyzed=\(report.analyzed) promoted=\(report.promoted) quarantined=\(report.quarantined) dropped=\(report.dropped) replay=\(report.replayRecords)\(report.note.map { " (\($0))" } ?? "")")
        onFinished?(report)
        return report
    }

    // MARK: - Power gate

    /// Same power gate as the training / adaptation schedulers (design 4.5):
    /// AC or UPS counts as powered, battery does not. Degrades OPEN where
    /// the power source cannot be queried.
    func isOnPower() -> Bool {
        #if os(macOS) && canImport(IOKit.ps)
        guard let infoRef = IOPSCopyPowerSourcesInfo() else { return true }
        let info = infoRef.takeRetainedValue()
        guard let typeRef = IOPSGetProvidingPowerSourceType(info) else { return true }
        let type = typeRef.takeRetainedValue() as String
        return type != "Battery Power"
        #else
        return true
        #endif
    }
}

import Foundation

/// Fires a recurring teacher-assessment sweep on the learning.assessmentIntervalHours
/// cadence. Each sweep reads the current assessments and logs a summary line;
/// the actual per-trial updates happen in TesseraTeacherAssessor.recordTrial()
/// (called by the escalation service when world outcomes arrive). This scheduler
/// exists so the assessment state is periodically surfaced and stale teachers
/// are flagged, not left silently accumulating.
public final class TesseraAssessmentScheduler: @unchecked Sendable {
    /// A teacher with samples but a pass fraction below this is flagged degraded.
    private static let degradedThreshold = 0.3

    private let assessor: any TesseraTeacherAssessing
    private let lock = NSLock()
    private var task: Task<Void, Never>?

    public init(assessor: any TesseraTeacherAssessing = TesseraLearningCenter.shared.assessor) {
        self.assessor = assessor
    }

    /// Launch the recurring sweep loop. Idempotent: calling it while already
    /// running is a no-op. Each iteration sleeps assessmentIntervalHours then
    /// sweeps; the loop runs until stop() cancels it.
    public func start() {
        lock.lock(); defer { lock.unlock() }
        guard task == nil else { return }
        task = Task.detached { [weak self] in
            while !Task.isCancelled {
                let hours = TesseraSettings.learningAssessmentIntervalHours
                try? await Task.sleep(nanoseconds: UInt64(hours) * 3_600 * 1_000_000_000)
                guard !Task.isCancelled else { break }
                self?.sweep()
            }
        }
    }

    public func stop() {
        lock.lock(); defer { lock.unlock() }
        task?.cancel()
        task = nil
    }

    /// Read the current assessments, flag any teacher with samples > 0 and
    /// worldGatePassFraction < 0.3 as degraded, log a summary line, and return
    /// the assessments. Degrade-open: assessments() is non-throwing and the
    /// loop only ever exits on cancellation, so one bad pass cannot kill it.
    @discardableResult
    public func sweep() -> [TesseraTeacherAssessment] {
        let assessments = assessor.assessments()
        let degraded = assessments.filter {
            $0.samples > 0 && $0.worldGatePassFraction < Self.degradedThreshold
        }
        for teacher in degraded {
            print("[tessera.assessment] teacher \(teacher.teacherId) degraded: pass fraction \(String(format: "%.2f", teacher.worldGatePassFraction)) over \(teacher.samples) samples")
        }
        print("[tessera.assessment] sweep: \(assessments.count) teacher(s), \(degraded.count) degraded")

        // Idle-window work: incrementally (re)train the leashed approver
        // network on the approval receipt stream (autonomy-calibration-
        // design.md 11.5). Cheap, fully local, fail-closed; no-ops below
        // the warmup threshold and rolls back on calibration collapse.
        if TesseraLearningCenter.shared.autonomy.trainApprover(denialWeight: 5.0) {
            print("[tessera.assessment] approver network retrained; calibration guard passed")
        }

        return assessments
    }
}

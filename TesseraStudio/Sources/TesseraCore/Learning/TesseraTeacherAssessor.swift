import Foundation

/// File-backed teacher assessor: keeps the recurring per-teacher quality
/// estimate that gates future use (design 4.1, the structural defense
/// against R3 teacher bias). Each world-gated trial updates a running
/// pass fraction and nudges the routing weight toward it.
public final class TesseraTeacherAssessor: TesseraTeacherAssessing, @unchecked Sendable {
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let file = "assessments.json"

    public init() {
        self.store = TesseraLearningStore()
    }

    public func assessments() -> [TesseraTeacherAssessment] {
        lock.lock(); defer { lock.unlock() }
        let map = store.load([String: TesseraTeacherAssessment].self, from: Self.file, default: [:])
        return Array(map.values)
    }

    public func recordTrial(proposal: TesseraTeacherProposal, passedWorldGate: Bool) throws {
        lock.lock(); defer { lock.unlock() }
        var map = store.load([String: TesseraTeacherAssessment].self, from: Self.file, default: [:])
        var assessment = map[proposal.teacherId] ?? TesseraTeacherAssessment(teacherId: proposal.teacherId)

        let pass = passedWorldGate ? 1.0 : 0.0
        // Running average of the world-gate pass fraction.
        assessment.worldGatePassFraction =
            (assessment.worldGatePassFraction * Double(assessment.samples) + pass) / Double(assessment.samples + 1)
        assessment.samples += 1
        // Nudge the routing weight toward the pass fraction, floored so a
        // cold or struggling teacher is never fully zeroed out.
        assessment.effectiveWeight = 0.25 + 0.75 * assessment.worldGatePassFraction
        assessment.lastUpdated = Date()

        map[proposal.teacherId] = assessment
        try store.save(map, to: Self.file)
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = store.load([String: TesseraTeacherAssessment].self, from: Self.file, default: [:]).count
        try store.delete(Self.file)
        return count
    }
}

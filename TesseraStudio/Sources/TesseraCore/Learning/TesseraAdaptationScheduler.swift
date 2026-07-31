import Foundation

/// Background idle adaptation scheduler (design 4.5). v1 is deliberately
/// honest: a dry run reports what WOULD fire and performs no training, and a
/// real run shells out to the tessera-adapt harness only if it is installed.
/// We never fake a training step.
public struct TesseraAdaptationScheduler: TesseraAdaptationScheduling {
    /// Minimum recent outcomes before an adaptation step is considered due.
    private static let signalThreshold = 20

    public init() {}

    public func isDue() -> Bool {
        guard TesseraSettings.learningIdleAdaptation else { return false }
        let recent = TesseraLearningCenter.shared.worldSignals.recent(limit: Self.signalThreshold)
        return recent.count >= Self.signalThreshold
    }

    public func runAdaptation(dryRun: Bool) async throws -> TesseraLearningReceipt {
        let signalCount = TesseraLearningCenter.shared.worldSignals.recent(limit: Self.signalThreshold).count

        if dryRun {
            return TesseraLearningReceipt(
                kind: "adaptation",
                summary: "Dry run: an adaptation step would fire on \(signalCount) recent outcome(s). No training performed.",
                payload: [
                    "dryRun": .bool(true),
                    "signalCount": .number(Double(signalCount)),
                ]
            )
        }

        // Real step: delegate to the tessera-adapt harness if present.
        let runner = ProcessRunner()
        do {
            let result = try await runner.run(executable: "tessera-adapt", arguments: ["--from-learning-store"])
            if result.exitCode == 0 {
                return TesseraLearningReceipt(
                    kind: "adaptation",
                    summary: "Adaptation step completed via tessera-adapt on \(signalCount) outcome(s).",
                    payload: [
                        "dryRun": .bool(false),
                        "signalCount": .number(Double(signalCount)),
                        "exitCode": .number(Double(result.exitCode)),
                    ]
                )
            }
            let detail = String(result.stderr.prefix(200))
            return TesseraLearningReceipt(
                kind: "adaptation",
                summary: "Adaptation harness exited \(result.exitCode): \(detail)",
                payload: [
                    "dryRun": .bool(false),
                    "exitCode": .number(Double(result.exitCode)),
                ]
            )
        } catch {
            // Missing binary (or non-macOS platform): report honestly.
            return TesseraLearningReceipt(
                kind: "adaptation",
                summary: "Adaptation harness unavailable (tessera-adapt not found or not runnable); no training performed.",
                payload: [
                    "dryRun": .bool(false),
                    "harnessFound": .bool(false),
                ]
            )
        }
    }
}

import Foundation
#if canImport(IOKit.ps)
import IOKit.ps
#endif

/// Background idle adaptation scheduler (design 4.5). This wave makes the
/// DECISION pipeline real: gather the latest multi-axis capability score,
/// serialize it to the harness instances format, shell out to --tessera-adapt
/// (which runs the collapse guard and writes a schema-versioned receipt),
/// parse the receipt, and persist the decision.
///
/// Honesty ceiling: the actual LoRA TRAINING is a plug-in point. The harness
/// returns adapted=false in v1 and this scheduler reports that plainly; what
/// is real is the gate + guard + receipt + record. We never fake a training
/// step, and we degrade honestly when the binary is missing (mirroring
/// TesseraAnonymizerService) or when no capability eval is on record.
public struct TesseraAdaptationScheduler: TesseraAdaptationScheduling {
    /// Minimum recent outcomes before an adaptation step is considered due.
    private static let signalThreshold = 20
    /// Ring-buffer cap on persisted adaptation decisions.
    private static let recordCapacity = 200
    private static let recordsFile = "adaptation-records.json"
    private static let receiptSchema = "llama.tessera.adapt.v1"

    private let store: TesseraLearningStore

    public init() {
        self.store = TesseraLearningStore()
    }

    public func isDue() -> Bool {
        guard TesseraSettings.learningIdleAdaptation else { return false }
        let recent = TesseraLearningCenter.shared.worldSignals.recent(limit: Self.signalThreshold)
        return recent.count >= Self.signalThreshold
    }

    public func lastAdaptation() -> TesseraAdaptationRecord? {
        loadRecords().last
    }

    public func runAdaptation(dryRun: Bool) async throws -> TesseraLearningReceipt {
        // Gate 1: idle adaptation must be enabled.
        guard TesseraSettings.learningIdleAdaptation else {
            return decisionReceipt(
                "Idle adaptation is disabled (learning.idleAdaptation=false); no step taken.",
                record: nil, dryRun: dryRun, signalCount: signalCount()
            )
        }

        // Gate 2: power, when required.
        if TesseraSettings.learningOnPowerOnly && !isOnPower() {
            return decisionReceipt(
                "On-power gate: not on power (learning.onPowerOnly=true); no step taken.",
                record: nil, dryRun: dryRun, signalCount: signalCount()
            )
        }

        // Gather the multi-axis capability input. Without a scored eval on
        // record we cannot derive a score honestly, so we stop rather than
        // fabricate one.
        guard let eval = TesseraCapabilityEvalStore().latest() else {
            return decisionReceipt(
                "No capability eval on record; run `evaluate` with capability_eval=true and scored results first. No step taken.",
                record: nil, dryRun: dryRun, signalCount: signalCount()
            )
        }

        let epsilon = TesseraSettings.learningGuardEpsilon
        // The previous decision's score is the baseline the guard regresses
        // against; the first run has none, so the guard passes trivially.
        let baseline = lastAdaptation()?.score
        let score = eval.score

        let binary = TesseraHarnessBinary.path
        guard FileManager.default.isExecutableFile(atPath: binary) else {
            let record = TesseraAdaptationRecord(
                dryRun: dryRun, guardPassed: false, adapted: false, epsilon: epsilon,
                score: score, hasBaseline: baseline != nil, backend: "unavailable",
                note: "harness binary not found at \(binary); no training performed"
            )
            try? appendRecord(record)
            return decisionReceipt(
                "Adaptation harness unavailable (\(binary) not found); no training performed.",
                record: record, dryRun: dryRun, signalCount: signalCount()
            )
        }

        // Serialize the eval (instances format) + baseline and shell out. The
        // harness reads the eval from a file and writes the receipt to a file;
        // both are ephemeral and removed before returning.
        let dir = NSTemporaryDirectory()
        let evalPath = (dir as NSString).appendingPathComponent("tessera-adapt-\(UUID().uuidString).eval.json")
        let receiptPath = (dir as NSString).appendingPathComponent("tessera-adapt-\(UUID().uuidString).receipt.json")
        defer {
            try? FileManager.default.removeItem(atPath: evalPath)
            try? FileManager.default.removeItem(atPath: receiptPath)
        }

        do {
            let data = try TesseraCapabilityEvalService()
                .serializeInstancesJSON(tallies: eval.tallies, baseline: baseline)
            try data.write(to: URL(fileURLWithPath: evalPath), options: .atomic)
        } catch {
            let record = TesseraAdaptationRecord(
                dryRun: dryRun, guardPassed: false, adapted: false, epsilon: epsilon,
                score: score, hasBaseline: baseline != nil, backend: "error",
                note: "could not stage eval for the harness"
            )
            try? appendRecord(record)
            return decisionReceipt(
                "Could not stage the eval for the adaptation harness; no training performed.",
                record: record, dryRun: dryRun, signalCount: signalCount()
            )
        }

        var arguments = [
            "--tessera-adapt", evalPath,
            "--tessera-adapt-out", receiptPath,
            "--tessera-adapt-epsilon", String(epsilon),
        ]
        if dryRun {
            arguments.append("--tessera-adapt-dry-run")
        }

        let result: ProcessResult
        do {
            result = try await ProcessRunner().run(executable: binary, arguments: arguments)
        } catch {
            let record = TesseraAdaptationRecord(
                dryRun: dryRun, guardPassed: false, adapted: false, epsilon: epsilon,
                score: score, hasBaseline: baseline != nil, backend: "unavailable",
                note: "harness process unavailable (\(error.localizedDescription))"
            )
            try? appendRecord(record)
            return decisionReceipt(
                "Adaptation harness unavailable (\(error.localizedDescription)); no training performed.",
                record: record, dryRun: dryRun, signalCount: signalCount()
            )
        }

        // Exit codes (quantize.cpp): 0 = guard passed, 1 = guard FAILED /
        // blocked, 2 = error. The receipt is written for 0 and 1 alike.
        let parsed = Self.parseAdaptReceipt(at: receiptPath)
        let record: TesseraAdaptationRecord
        let summary: String

        switch result.exitCode {
        case 0:
            let adapted = parsed?.adapted ?? false
            record = TesseraAdaptationRecord(
                dryRun: dryRun, guardPassed: true, adapted: adapted, epsilon: epsilon,
                score: score, hasBaseline: baseline != nil, backend: "harness",
                note: adapted
                    ? "guard passed; adapter produced"
                    : "guard passed; training not wired in v1 (adapted=false)"
            )
            summary = "Adaptation \(dryRun ? "dry run" : "step"): collapse guard PASSED (epsilon \(String(format: "%g", epsilon))). "
                + (adapted ? "Adapter produced." : "No training performed (training plug-in point; adapted=false).")
        case 1:
            record = TesseraAdaptationRecord(
                dryRun: dryRun, guardPassed: false, adapted: false, epsilon: epsilon,
                score: score, hasBaseline: baseline != nil, backend: "harness",
                note: "collapse guard FAILED: general-competence regression beyond epsilon; adaptation blocked"
            )
            summary = "Adaptation \(dryRun ? "dry run" : "step"): collapse guard FAILED "
                + "(general competence \(String(format: "%.3f", score.generalCompetence)) vs baseline "
                + "\(baseline.map { String(format: "%.3f", $0.generalCompetence) } ?? "none"), epsilon \(String(format: "%g", epsilon))). "
                + "Adaptation blocked; no training performed."
        default:
            let detail = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            record = TesseraAdaptationRecord(
                dryRun: dryRun, guardPassed: false, adapted: false, epsilon: epsilon,
                score: score, hasBaseline: baseline != nil, backend: "error",
                note: "harness exited \(result.exitCode)\(detail.isEmpty ? "" : ": " + String(detail.prefix(200)))"
            )
            summary = "Adaptation harness exited \(result.exitCode); no training performed."
        }

        try? appendRecord(record)
        return decisionReceipt(summary, record: record, dryRun: dryRun, signalCount: signalCount())
    }

    // MARK: - Power gate

    /// Power gate (design 4.5 / Phase 4: idle + on-power). On macOS this reads
    /// the live power source via IOPowerSources; AC or UPS counts as powered,
    /// battery does not. PLUG-IN POINT: where the power source cannot be
    /// queried (non-macOS, or the call fails) we assume powered so the gate
    /// degrades OPEN rather than silently blocking adaptation forever.
    func isOnPower() -> Bool {
        #if os(macOS) && canImport(IOKit.ps)
        guard let infoRef = IOPSCopyPowerSourcesInfo() else { return true }
        let info = infoRef.takeRetainedValue()
        guard let typeRef = IOPSGetProvidingPowerSourceType(info) else { return true }
        let type = typeRef.takeRetainedValue() as String
        // kIOPSBatteryPowerKey ("Battery Power") is not exported to Swift; the
        // literal is its documented value. Anything else (AC / UPS) is powered.
        return type != "Battery Power"
        #else
        return true
        #endif
    }

    // MARK: - Records

    private func signalCount() -> Int {
        TesseraLearningCenter.shared.worldSignals.recent(limit: Self.signalThreshold).count
    }

    private func loadRecords() -> [TesseraAdaptationRecord] {
        store.load([TesseraAdaptationRecord].self, from: Self.recordsFile, default: [])
    }

    private func appendRecord(_ record: TesseraAdaptationRecord) throws {
        var records = loadRecords()
        records.append(record)
        if records.count > Self.recordCapacity {
            records.removeFirst(records.count - Self.recordCapacity)
        }
        try store.save(records, to: Self.recordsFile)
    }

    private func decisionReceipt(
        _ summary: String,
        record: TesseraAdaptationRecord?,
        dryRun: Bool,
        signalCount: Int
    ) -> TesseraLearningReceipt {
        var payload: [String: JSONValue] = [
            "dryRun": .bool(dryRun),
            "signalCount": .number(Double(signalCount)),
        ]
        if let record {
            payload["guardPassed"] = .bool(record.guardPassed)
            payload["adapted"] = .bool(record.adapted)
            payload["epsilon"] = .number(record.epsilon)
            payload["backend"] = .string(record.backend)
            payload["hasBaseline"] = .bool(record.hasBaseline)
        }
        return TesseraLearningReceipt(kind: "adaptation", summary: summary, payload: payload)
    }

    // MARK: - Receipt parsing

    private struct HarnessAdaptScore: Codable {
        let mechanical: Double?
        let api_currency: Double?
        let hard_tail: Double?
        let personal_style: Double?
        let general_competence: Double?
    }

    private struct HarnessAdaptReceipt: Codable {
        let schema: String?
        let timestamp: String?
        let dry_run: Bool?
        let guard_epsilon: Double?
        let guard_passed: Bool?
        let adapted: Bool?
        let has_baseline: Bool?
        let score: HarnessAdaptScore?
    }

    /// Parse the harness adaptation receipt. Returns nil when the file is
    /// missing, malformed, or carries an unexpected schema; the caller treats
    /// that as "no readable receipt" and relies on the process exit code.
    private static func parseAdaptReceipt(at path: String) -> HarnessAdaptReceipt? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
        guard let decoded = try? JSONDecoder().decode(HarnessAdaptReceipt.self, from: data) else { return nil }
        if let schema = decoded.schema, schema != receiptSchema { return nil }
        return decoded
    }
}

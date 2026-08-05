import Foundation

/// One spec-decoding step of a captured runtime session, as parsed from a
/// llama.tessera.spec.v1 record.
public struct TesseraRuntimeSessionStep: Sendable, Equatable {
    public let stepIdx: Int
    public let drafted: Int
    public let accepted: Int
    public let acceptedTokens: [Int32]

    public init(stepIdx: Int, drafted: Int, accepted: Int, acceptedTokens: [Int32]) {
        self.stepIdx = stepIdx
        self.drafted = drafted
        self.accepted = accepted
        self.acceptedTokens = acceptedTokens
    }
}

/// One captured runtime session: every runtime record sharing a sid, ordered
/// by step_idx. A retried flush can write the same step twice; duplicate
/// step indices collapse to the first occurrence.
public struct TesseraRuntimeSession: Sendable, Equatable {
    public let sid: String
    public let steps: [TesseraRuntimeSessionStep]

    public init(sid: String, steps: [TesseraRuntimeSessionStep]) {
        self.sid = sid
        self.steps = steps
    }

    /// The session's accepted token sequence: each step's accepted drafts
    /// plus its bonus token, concatenated in step order. This is what the
    /// curation stage decodes to UTF-8.
    public var acceptedTokens: [Int32] {
        steps.flatMap { $0.acceptedTokens }
    }
}

/// Reads runtime trace files back into per-sid sessions (runtime-traces spec
/// section 12.2 decode step input). Pure file parsing - tolerant of
/// malformed lines and records without a sid, which are skipped.
public enum TesseraSessionTraceReader {
    /// Sessions across every runtime file in the store, oldest capture first.
    public static func sessions(in store: TesseraTraceStore) -> [TesseraRuntimeSession] {
        sessions(inFiles: store.runtimeFiles())
    }

    public static func sessions(inFiles files: [URL]) -> [TesseraRuntimeSession] {
        var order: [String] = []
        var stepsBySid: [String: [Int: TesseraRuntimeSessionStep]] = [:]
        for file in files {
            guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
            text.enumerateLines { line, _ in
                guard !line.trimmingCharacters(in: .whitespaces).isEmpty,
                      let step = parseLine(line) else { return }
                if stepsBySid[step.sid] == nil { order.append(step.sid) }
                // First write of a step wins; a retried flush re-appends the
                // identical record.
                stepsBySid[step.sid, default: [:]][step.step.stepIdx] = step.step
            }
        }
        return order.map { sid in
            let steps = (stepsBySid[sid] ?? [:])
                .values
                .sorted { $0.stepIdx < $1.stepIdx }
            return TesseraRuntimeSession(sid: sid, steps: steps)
        }
    }

    private static func parseLine(_ line: String) -> (sid: String, step: TesseraRuntimeSessionStep)? {
        guard let data = line.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let sid = obj["sid"] as? String, !sid.isEmpty else {
            return nil
        }
        let stepIdx = (obj["step_idx"] as? NSNumber)?.intValue ?? 0
        let drafted = (obj["drafted"] as? NSNumber)?.intValue ?? 0
        let accepted = (obj["accepted"] as? NSNumber)?.intValue ?? 0
        let acceptedTokens = ((obj["accepted_tokens"] as? [NSNumber]) ?? []).map { $0.int32Value }
        return (sid, TesseraRuntimeSessionStep(
            stepIdx: stepIdx, drafted: drafted,
            accepted: accepted, acceptedTokens: acceptedTokens))
    }
}

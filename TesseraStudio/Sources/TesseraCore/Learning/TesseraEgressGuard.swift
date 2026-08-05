import Foundation

/// The record-level egress guard (runtime-traces spec sections 8, 9, 12.5).
/// The filename prefix is the first egress-filter line (the orchestrator
/// never stages traces-runtime- files); the provenance field is the second.
/// Fail-closed: anything that is not plainly allowed is dropped.
public enum TesseraEgressGuard {
    /// True when one JSONL record may leave the machine as training or
    /// dataset fuel. Calibration records carry no provenance field and
    /// pass. Replay records pass only with the exact promotion stamp
    /// ("provenance":"replay" plus "replayed_from":"runtime"). Runtime
    /// records and anything unknown, unstamped, or unparseable drop:
    /// runtime captures reach training exclusively through the curation
    /// stage's replay (spec section 9 invariant).
    public static func allows(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        // Calibration output never mentions provenance; skip the parse.
        guard trimmed.contains("\"provenance\"") else { return true }
        guard let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let dict = obj as? [String: Any] else { return false }
        guard let provenance = dict["provenance"] as? String else { return false }
        guard provenance == "replay" else { return false }
        return (dict["replayed_from"] as? String) == "runtime"
    }
}

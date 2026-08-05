import Foundation

/// The record-level egress guard (runtime-traces spec sections 8, 9, 12.5).
/// The filename prefix is the first egress-filter line (the orchestrator
/// never stages traces-runtime- or traces-s2s- files); the provenance field
/// is the second. Fail-closed: anything that is not plainly allowed is
/// dropped.
public enum TesseraEgressGuard {
    /// Provenances that never leave the machine, by name. Runtime captures
    /// reach training exclusively through the curation stage's replay
    /// (runtime-traces spec section 9). S2S records are Tier B LOCAL-ONLY:
    /// their codes are voice-bearing, so they are never dataset fuel (s2s
    /// design sections 4.1, 4.2).
    private static let localOnlyProvenances: Set<String> = ["runtime", "s2s"]

    /// True when one JSONL record may leave the machine as training or
    /// dataset fuel. Calibration records carry no provenance field and
    /// pass. Replay records pass only with the exact promotion stamp
    /// ("provenance":"replay" plus "replayed_from":"runtime"). Local-only
    /// provenances (runtime, s2s) and anything unknown, unstamped, or
    /// unparseable drop.
    public static func allows(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        // Calibration output never mentions provenance; skip the parse.
        guard trimmed.contains("\"provenance\"") else { return true }
        guard let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let dict = obj as? [String: Any] else { return false }
        guard let provenance = dict["provenance"] as? String else { return false }
        guard !localOnlyProvenances.contains(provenance) else { return false }
        guard provenance == "replay" else { return false }
        return (dict["replayed_from"] as? String) == "runtime"
    }
}

import Foundation

/// The outcome of an anonymization pass. `map` is the local de-anonymization
/// key (pseudonym -> original) and is ONLY populated for a real symbol-level
/// pass; a degraded fallback carries an empty map and `usedFallback == true`
/// so callers never mistake a secret-scrub for a symbol anonymization.
public struct TesseraAnonymizerResult: Sendable, Equatable {
    public let text: String              // anonymized (or scrubbed) payload
    public let map: [String: String]     // pseudonym -> original; empty on fallback
    public let anonymizer: String        // "symbol-level" | "curation-scrub"
    public let level: String             // configured dial; "" on fallback
    public let usedFallback: Bool
    public let note: String              // honest explanation when fallback was used

    public init(
        text: String,
        map: [String: String],
        anonymizer: String,
        level: String,
        usedFallback: Bool,
        note: String = ""
    ) {
        self.text = text
        self.map = map
        self.anonymizer = anonymizer
        self.level = level
        self.usedFallback = usedFallback
        self.note = note
    }
}

/// Tier-2 symbol-level anonymizer (design 4.1 / Phase 5 / R9). Shells out to
/// the C++ tessera-anonymizer carried by the llama-quantize binary: the
/// payload is written to a temp file, the binary prints the anonymized text to
/// stdout and writes the pseudonym -> original map as JSON, and the map is
/// parsed back so a teacher's answer can be de-anonymized locally later.
///
/// Honesty ceiling: when the binary is missing or the process fails, this does
/// NOT fake a symbol-level pass. It degrades to curation.scrub (a secret
/// scrubber, not a symbol anonymizer) and reports the fallback explicitly.
public struct TesseraAnonymizerService: Sendable {
    /// Default installed location of the binary that carries the anonymizer.
    public static let defaultBinaryPath = "/usr/local/bin/llama-quantize"
    private static let mapSchema = "llama.tessera.anonymizer.v1"
    private static let mapsFile = "anonymizer-maps.json"

    private let store: TesseraLearningStore

    public init() {
        self.store = TesseraLearningStore()
    }

    /// The configured binary path, or the installed default when unset.
    public var binaryPath: String {
        let configured = TesseraSettings.learningAnonymizerBinary
        return configured.isEmpty ? Self.defaultBinaryPath : configured
    }

    // MARK: - Anonymize

    public func anonymize(
        _ payload: String,
        level: String = TesseraSettings.learningAnonymizerAggressiveness
    ) async -> TesseraAnonymizerResult {
        let binary = binaryPath
        guard FileManager.default.isExecutableFile(atPath: binary) else {
            return fallback(payload, level: level,
                note: "anonymizer binary not found at \(binary); degraded to curation.scrub")
        }

        // The binary reads the payload from a file and writes the map to a
        // file; both are ephemeral and removed before returning.
        let dir = NSTemporaryDirectory()
        let inputPath = (dir as NSString).appendingPathComponent("tessera-anon-\(UUID().uuidString).in")
        let mapPath = (dir as NSString).appendingPathComponent("tessera-anon-\(UUID().uuidString).map")
        defer {
            try? FileManager.default.removeItem(atPath: inputPath)
            try? FileManager.default.removeItem(atPath: mapPath)
        }

        do {
            try payload.write(toFile: inputPath, atomically: true, encoding: .utf8)
        } catch {
            return fallback(payload, level: level,
                note: "could not stage payload for the anonymizer; degraded to curation.scrub")
        }

        let result: ProcessResult
        do {
            result = try await ProcessRunner().run(
                executable: binary,
                arguments: [
                    "--tessera-anonymize", inputPath,
                    "--tessera-anonymize-level", level,
                    "--tessera-anonymize-map", mapPath,
                ]
            )
        } catch {
            return fallback(payload, level: level,
                note: "anonymizer process unavailable (\(error.localizedDescription)); degraded to curation.scrub")
        }

        guard result.exitCode == 0 else {
            let detail = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return fallback(payload, level: level,
                note: "anonymizer exited \(result.exitCode)\(detail.isEmpty ? "" : ": " + detail); degraded to curation.scrub")
        }

        guard let map = Self.parseMap(at: mapPath) else {
            return fallback(payload, level: level,
                note: "anonymizer ran but its de-anonymization map was unreadable; degraded to curation.scrub")
        }

        return TesseraAnonymizerResult(
            text: result.stdout,
            map: map,
            anonymizer: "symbol-level",
            level: level,
            usedFallback: false
        )
    }

    // MARK: - De-anonymize

    /// Reverse the transform by whole-identifier replacement of each pseudonym
    /// (map key) with its original (map value). Word boundaries keep a short
    /// pseudonym from matching inside a longer identifier.
    public func deAnonymize(text: String, map: [String: String]) -> String {
        var out = text
        for (pseudonym, original) in map {
            let pattern = "\\b\(NSRegularExpression.escapedPattern(for: pseudonym))\\b"
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let range = NSRange(out.startIndex..<out.endIndex, in: out)
            // Escape any backslash/$ in the original so they are not read as
            // regex-replacement metacharacters.
            let safe = original
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "$", with: "\\$")
            out = regex.stringByReplacingMatches(in: out, options: [], range: range, withTemplate: safe)
        }
        return out
    }

    // MARK: - Map persistence

    /// Persist the de-anonymization map under an escalation id so a teacher's
    /// answer can be de-anonymized later. The map is the local-only key that
    /// makes tier-2 egress reversible; it never leaves the machine.
    public func persistMap(_ map: [String: String], forEscalation escalationId: String) throws {
        var all = store.load([String: [String: String]].self, from: Self.mapsFile, default: [:])
        all[escalationId] = map
        try store.save(all, to: Self.mapsFile)
    }

    public func loadMap(forEscalation escalationId: String) -> [String: String]? {
        store.load([String: [String: String]].self, from: Self.mapsFile, default: [:])[escalationId]
    }

    // MARK: - Helpers

    private func fallback(_ payload: String, level: String, note: String) -> TesseraAnonymizerResult {
        TesseraAnonymizerResult(
            text: TesseraLearningCenter.shared.curation.scrub(payload),
            map: [:],
            anonymizer: "curation-scrub",
            level: "",
            usedFallback: true,
            note: note
        )
    }

    /// Parse the binary's map JSON ({"schema","level","symbols":{pseudo:original}}).
    /// Returns nil when the file is missing, malformed, or carries no symbols -
    /// a symbol-level pass without a readable map is treated as a failure.
    private static func parseMap(at path: String) -> [String: String]? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
        struct MapFile: Codable {
            let schema: String?
            let symbols: [String: String]?
        }
        guard let decoded = try? JSONDecoder().decode(MapFile.self, from: data) else { return nil }
        if let schema = decoded.schema, schema != mapSchema { return nil }
        return decoded.symbols ?? [:]
    }
}

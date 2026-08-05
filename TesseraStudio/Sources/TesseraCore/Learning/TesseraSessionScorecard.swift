import Foundation

/// Per-step spec statistics recorded by the runtime capture, as read back
/// from llama.tessera.spec.v1 records during curation.
public struct TesseraSessionStepStats: Sendable, Equatable {
    public var drafted: Int
    public var accepted: Int

    public init(drafted: Int, accepted: Int) {
        self.drafted = drafted
        self.accepted = accepted
    }
}

/// The analysis of one decoded runtime session (runtime-traces spec section
/// 12.3). Every field is derived, never configured: decoded text plus the
/// recorded step statistics go in, the scorecard comes out.
public struct TesseraSessionAnalysis: Sendable, Equatable {
    /// Decoded accepted-token count.
    public var tokenCount: Int
    /// accepted / drafted across the session's steps.
    public var acceptanceRate: Double
    /// Mean accepted drafts per step.
    public var meanAcceptedRun: Double
    /// Word n-gram self-overlap in the decoded text (0 = no repetition).
    public var repetitionRatio: Double
    /// Fraction of decoded pieces that are empty or undecodable junk
    /// (EOS-ish / out-of-distribution garbage).
    public var garbageRatio: Double
    /// Scrub rule ids that matched the decoded text (read-only probe).
    public var probeHits: [String]
    /// Dedup fingerprint of the normalized decoded text.
    public var fingerprint: String
    /// Every recorded token id decodes against the current trunk vocab.
    public var modelCompatible: Bool

    public init(
        tokenCount: Int,
        acceptanceRate: Double,
        meanAcceptedRun: Double,
        repetitionRatio: Double,
        garbageRatio: Double,
        probeHits: [String],
        fingerprint: String,
        modelCompatible: Bool
    ) {
        self.tokenCount = tokenCount
        self.acceptanceRate = acceptanceRate
        self.meanAcceptedRun = meanAcceptedRun
        self.repetitionRatio = repetitionRatio
        self.garbageRatio = garbageRatio
        self.probeHits = probeHits
        self.fingerprint = fingerprint
        self.modelCompatible = modelCompatible
    }
}

/// Terminal curation outcome for one session (spec section 12.4).
public enum TesseraSessionVerdict: String, Codable, Sendable {
    case promoted
    case quarantined
    case dropped
    /// User-initiated purge (spec section 12.4): the session's records were
    /// removed from the trace store on request. Quarantined sessions are
    /// exempt from automatic retention entirely, so this verdict is the
    /// only path that removes them. Latest-wins keeps the session out of
    /// the quarantine list and out of any future re-analysis.
    case purged
}

/// A verdict plus the reasons that produced it, in ledger form.
public struct TesseraSessionJudgement: Sendable, Equatable {
    public let verdict: TesseraSessionVerdict
    public let reasons: [String]

    public init(verdict: TesseraSessionVerdict, reasons: [String]) {
        self.verdict = verdict
        self.reasons = reasons
    }
}

/// Pure quality/sensitivity/duplication/compatibility scoring for decoded
/// runtime sessions. Thresholds are deliberately conservative: quarantine is
/// reviewable and local-only, while a leaked sensitivity hit is not, so the
/// sensitivity check outranks every quality consideration.
public enum TesseraSessionScorecard {
    /// Sessions below this decoded token count are noise.
    public static let minTokens = 64
    /// Below this acceptance rate the session teaches the drafter little.
    public static let minAcceptance = 0.10
    /// Above this n-gram self-overlap the session is a repetition loop.
    public static let maxRepetition = 0.60
    /// Above this garbage-piece ratio the decode is mostly EOS/junk.
    public static let maxGarbage = 0.30
    /// Word n-gram size for the repetition ratio.
    public static let ngramSize = 4

    // MARK: - Metric derivations (each pure and individually testable)

    public static func acceptanceRate(steps: [TesseraSessionStepStats]) -> Double {
        let drafted = steps.reduce(0) { $0 + $1.drafted }
        guard drafted > 0 else { return 0 }
        let accepted = steps.reduce(0) { $0 + $1.accepted }
        return Double(accepted) / Double(drafted)
    }

    public static func meanAcceptedRun(steps: [TesseraSessionStepStats]) -> Double {
        guard !steps.isEmpty else { return 0 }
        let accepted = steps.reduce(0) { $0 + $1.accepted }
        return Double(accepted) / Double(steps.count)
    }

    /// 1 - unique/total word n-grams; texts shorter than one n-gram have
    /// nothing to overlap and score 0.
    public static func repetitionRatio(of text: String, ngramSize n: Int = ngramSize) -> Double {
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard words.count > n else { return 0 }
        var total = 0
        var seen = Set<[String]>()
        for i in 0...(words.count - n) {
            seen.insert(Array(words[i..<(i + n)]))
            total += 1
        }
        return 1.0 - Double(seen.count) / Double(total)
    }

    /// A piece is garbage when it is empty (EOS detokenizes to nothing) or
    /// carries no printable content and is not plain whitespace. Newlines
    /// and spaces are normal text, not junk.
    public static func isGarbage(_ piece: String) -> Bool {
        if piece.isEmpty { return true }
        let hasPrintable = piece.contains { !$0.isASCII || $0.isPrintableASCII }
        if hasPrintable { return false }
        return !piece.allSatisfy { $0.isWhitespace }
    }

    public static func garbageRatio(pieces: [String]) -> Double {
        guard !pieces.isEmpty else { return 0 }
        let garbage = pieces.filter(isGarbage).count
        return Double(garbage) / Double(pieces.count)
    }

    /// Deterministic dedup fingerprint: FNV-1a over the lowercased,
    /// whitespace-collapsed text. Stable across launches (String.hashValue
    /// is not), so a repeated session is recognized on a later sweep.
    public static func fingerprint(of text: String) -> String {
        let normalized = text.lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in normalized.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return String(hash, radix: 16)
    }

    // MARK: - Verdict

    /// Precedence: model-mismatch (nothing else is trustworthy), then
    /// sensitivity (quarantine outranks quality), then duplication, then the
    /// quality floors.
    public static func judge(_ analysis: TesseraSessionAnalysis, isDuplicate: Bool) -> TesseraSessionJudgement {
        guard analysis.modelCompatible else {
            return TesseraSessionJudgement(verdict: .dropped, reasons: ["model-mismatch"])
        }
        if !analysis.probeHits.isEmpty {
            return TesseraSessionJudgement(
                verdict: .quarantined,
                reasons: analysis.probeHits.map { "probe:\($0)" })
        }
        if isDuplicate {
            return TesseraSessionJudgement(verdict: .dropped, reasons: ["duplicate"])
        }
        var failures: [String] = []
        if analysis.tokenCount < minTokens { failures.append("below-token-floor") }
        if analysis.acceptanceRate < minAcceptance { failures.append("low-acceptance") }
        if analysis.repetitionRatio > maxRepetition { failures.append("high-repetition") }
        if analysis.garbageRatio > maxGarbage { failures.append("garbage") }
        guard failures.isEmpty else {
            return TesseraSessionJudgement(verdict: .dropped, reasons: failures)
        }
        return TesseraSessionJudgement(
            verdict: .promoted,
            reasons: ["probe:none", "dedup:kept", "low-repetition"])
    }
}

private extension Character {
    var isPrintableASCII: Bool {
        guard let ascii = asciiValue else { return false }
        return ascii >= 0x20 && ascii < 0x7F
    }
}

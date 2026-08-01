import Foundation

// MARK: - Curation products

/// A (chosen, rejected) preference pair for one problem class, derived from a
/// pass vs a fail on the world gate. Pair FORMATION is real and testable; the
/// DPO-style training consumer is a marked plug-in point (design 4.2).
public struct TesseraPreferencePair: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let problemClass: String
    public let chosen: TesseraWorldOutcome
    public let rejected: TesseraWorldOutcome

    public init(
        id: String = UUID().uuidString,
        problemClass: String,
        chosen: TesseraWorldOutcome,
        rejected: TesseraWorldOutcome
    ) {
        self.id = id
        self.problemClass = problemClass
        self.chosen = chosen
        self.rejected = rejected
    }
}

/// A brief rollup of curation state for the transparency surface: how much is
/// stored, how many duplicates were skipped, how many preference pairs are
/// formable now, and the mean heuristic quality of what is stored.
public struct TesseraCurationSummary: Codable, Sendable, Equatable {
    public var stored: Int
    public var dedupSkipped: Int
    public var preferencePairs: Int
    public var meanQuality: Double

    public init(stored: Int = 0, dedupSkipped: Int = 0, preferencePairs: Int = 0, meanQuality: Double = 0) {
        self.stored = stored
        self.dedupSkipped = dedupSkipped
        self.preferencePairs = preferencePairs
        self.meanQuality = meanQuality
    }
}
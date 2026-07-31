import Foundation

// MARK: - Eval instances

/// One held-out evaluation instance on a single capability axis. The first
/// proof optimizes the simplest axis (mechanical: red test -> green, binary
/// reward), but instances exist for every axis from day 1 (design 4.7 / 8).
public struct TesseraEvalInstance: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let axis: String        // one of TesseraCapabilityScore.axisNames
    public let prompt: String      // the task / description put to the model
    public let expectedSignal: String // verifiable condition that means "passed"

    public init(id: String = UUID().uuidString, axis: String, prompt: String, expectedSignal: String = "") {
        self.id = id
        self.axis = axis
        self.prompt = prompt
        self.expectedSignal = expectedSignal
    }
}

/// The outcome of running one instance against a model. Producing these
/// requires a live inference run; this wave does NOT run them - a caller /
/// future runner supplies them (see honesty note in EvaluateTool).
public struct TesseraEvalInstanceResult: Codable, Sendable, Equatable {
    public let instanceId: String
    public let axis: String
    public let passed: Bool

    public init(instanceId: String, axis: String, passed: Bool) {
        self.instanceId = instanceId
        self.axis = axis
        self.passed = passed
    }
}

/// On-disk exchange format for a caller-supplied eval run: the instances
/// that were run plus the results the run produced.
public struct TesseraEvalInstanceFile: Codable, Sendable, Equatable {
    public var instances: [TesseraEvalInstance]
    public var results: [TesseraEvalInstanceResult]

    public init(instances: [TesseraEvalInstance] = [], results: [TesseraEvalInstanceResult] = []) {
        self.instances = instances
        self.results = results
    }
}

// MARK: - Lens comparison

/// Both readings of the same score vector (ratified decision #8): the
/// weighted-sum scalar lens and the Pareto non-domination lens, plus whether
/// the two lenses pick different winners. This is the A/B surface.
public struct TesseraLensComparison: Codable, Sendable, Equatable {
    public var weightedSumA: Double
    public var weightedSumB: Double
    public var scalarWinner: String   // "a" | "b" | "tie"
    public var aDominatesB: Bool
    public var bDominatesA: Bool
    public var neither: Bool          // neither dominates (Pareto-incomparable or equal)
    public var paretoWinner: String   // "a" | "b" | "tie" (tie == non-domination)
    public var disagree: Bool         // the two lenses pick different winners

    public init(
        weightedSumA: Double,
        weightedSumB: Double,
        scalarWinner: String,
        aDominatesB: Bool,
        bDominatesA: Bool,
        neither: Bool,
        paretoWinner: String,
        disagree: Bool
    ) {
        self.weightedSumA = weightedSumA
        self.weightedSumB = weightedSumB
        self.scalarWinner = scalarWinner
        self.aDominatesB = aDominatesB
        self.bDominatesA = bDominatesA
        self.neither = neither
        self.paretoWinner = paretoWinner
        self.disagree = disagree
    }
}

// MARK: - Service

/// Turns per-axis instance results into a TesseraCapabilityScore and reads
/// the same vector through two lenses (weighted-sum scalar, Pareto
/// non-domination). Stateless; the score VECTOR is the substrate, the lenses
/// are just projections (design 4.7).
public struct TesseraCapabilityEvalService {
    public init() {}

    /// Each axis = pass fraction over that axis's instances. Axes with zero
    /// instances score 0. Results naming an unknown axis are ignored.
    public func score(from results: [TesseraEvalInstanceResult]) -> TesseraCapabilityScore {
        var passed: [String: Int] = [:]
        var total: [String: Int] = [:]
        for result in results {
            guard TesseraCapabilityScore.axisNames.contains(result.axis) else { continue }
            total[result.axis, default: 0] += 1
            if result.passed { passed[result.axis, default: 0] += 1 }
        }

        func fraction(_ axis: String) -> Double {
            guard let denominator = total[axis], denominator > 0 else { return 0 }
            return Double(passed[axis] ?? 0) / Double(denominator)
        }

        return TesseraCapabilityScore(
            mechanical: fraction("mechanical"),
            apiCurrency: fraction("apiCurrency"),
            hardTail: fraction("hardTail"),
            personalStyle: fraction("personalStyle"),
            generalCompetence: fraction("generalCompetence")
        )
    }

    /// Read the same two scores through both lenses and report whether they
    /// agree on a winner. generalCompetence is a guard axis: it participates
    /// in Pareto domination (a regression there can block a candidate) but is
    /// excluded from the weighted sum by TesseraCapabilityScore.weightedSum.
    public func compareLenses(
        _ a: TesseraCapabilityScore,
        _ b: TesseraCapabilityScore,
        weights: [String: Double]
    ) -> TesseraLensComparison {
        let sumA = a.weightedSum(weights: weights)
        let sumB = b.weightedSum(weights: weights)

        let scalarWinner: String
        if sumA > sumB { scalarWinner = "a" }
        else if sumB > sumA { scalarWinner = "b" }
        else { scalarWinner = "tie" }

        let aDominatesB = a.dominates(b)
        let bDominatesA = b.dominates(a)
        let paretoWinner: String
        if aDominatesB { paretoWinner = "a" }
        else if bDominatesA { paretoWinner = "b" }
        else { paretoWinner = "tie" }

        return TesseraLensComparison(
            weightedSumA: sumA,
            weightedSumB: sumB,
            scalarWinner: scalarWinner,
            aDominatesB: aDominatesB,
            bDominatesA: bDominatesA,
            neither: !aDominatesB && !bDominatesA,
            paretoWinner: paretoWinner,
            disagree: scalarWinner != paretoWinner
        )
    }

    /// Human-readable A/B summary for the tessera-ab-harness concept:
    /// "weighted-sum says X, Pareto says Y, agree/disagree". Here a ==
    /// baseline and b == candidate.
    public func abLensReport(
        baseline: TesseraCapabilityScore,
        candidate: TesseraCapabilityScore,
        weights: [String: Double]
    ) -> String {
        let comparison = compareLenses(baseline, candidate, weights: weights)

        func label(_ winner: String) -> String {
            switch winner {
            case "a": return "baseline"
            case "b": return "candidate"
            default: return "tie"
            }
        }

        let sumA = String(format: "%.3f", comparison.weightedSumA)
        let sumB = String(format: "%.3f", comparison.weightedSumB)
        let verdict = comparison.disagree ? "DISAGREE" : "agree"
        return "weighted-sum says \(label(comparison.scalarWinner)) (\(sumA) vs \(sumB)), "
            + "Pareto says \(label(comparison.paretoWinner)) -> lenses \(verdict)"
    }
}

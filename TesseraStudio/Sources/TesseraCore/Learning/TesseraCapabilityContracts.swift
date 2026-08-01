import Foundation

// MARK: - Multi-axis capability score

/// A per-candidate behavioral score vector. The vector is the substrate;
/// the weighted-sum scalar and Pareto non-domination below are two lenses
/// on the same numbers (ratified decision #8). generalCompetence is a
/// GUARD axis (hard regression constraint), not a trade-off weight.
public struct TesseraCapabilityScore: Codable, Sendable, Equatable {
    public var mechanical: Double        // failing-test + compiler/type-error instances
    public var apiCurrency: Double       // deprecated-API migration instances
    public var hardTail: Double          // escalation instances
    public var personalStyle: Double     // trunk LoRA / personal-distribution fit
    public var generalCompetence: Double // broad held-out set; the collapse guard

    public static let axisNames = ["mechanical", "apiCurrency", "hardTail", "personalStyle", "generalCompetence"]
    /// Axes that participate in trade-offs (everything except the guard).
    public static let optimizationAxisNames = ["mechanical", "apiCurrency", "hardTail", "personalStyle"]

    public init(
        mechanical: Double = 0,
        apiCurrency: Double = 0,
        hardTail: Double = 0,
        personalStyle: Double = 0,
        generalCompetence: Double = 0
    ) {
        self.mechanical = mechanical
        self.apiCurrency = apiCurrency
        self.hardTail = hardTail
        self.personalStyle = personalStyle
        self.generalCompetence = generalCompetence
    }

    public var vector: [Double] {
        [mechanical, apiCurrency, hardTail, personalStyle, generalCompetence]
    }

    public subscript(axis: String) -> Double {
        switch axis {
        case "mechanical": return mechanical
        case "apiCurrency": return apiCurrency
        case "hardTail": return hardTail
        case "personalStyle": return personalStyle
        case "generalCompetence": return generalCompetence
        default: return 0
        }
    }

    /// Weighted-sum lens over the OPTIMIZATION axes only. The guard axis
    /// is handled by `passesGuard`, never traded off here.
    public func weightedSum(weights: [String: Double]) -> Double {
        var sum = 0.0
        var weightTotal = 0.0
        for axis in Self.optimizationAxisNames {
            let w = weights[axis] ?? 1.0
            sum += w * self[axis]
            weightTotal += w
        }
        return weightTotal > 0 ? sum / weightTotal : 0.0
    }

    /// Pareto lens: true if self dominates other across ALL axes (>= on
    /// every axis, > on at least one).
    public func dominates(_ other: TesseraCapabilityScore) -> Bool {
        let a = vector
        let b = other.vector
        var strictlyBetter = false
        for i in 0..<a.count {
            if a[i] < b[i] { return false }
            if a[i] > b[i] { strictlyBetter = true }
        }
        return strictlyBetter
    }

    /// Guard check: general competence must not drop more than epsilon
    /// below the baseline. A nil baseline passes trivially.
    public func passesGuard(baseline: TesseraCapabilityScore?, epsilon: Double) -> Bool {
        guard let baseline else { return true }
        return generalCompetence >= baseline.generalCompetence - epsilon
    }
}
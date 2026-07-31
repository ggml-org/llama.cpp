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

// MARK: - Harness exchange types

/// Per-axis pass/fail tally. The C++ harness (ts_capability_score_load)
/// reduces {"pass":N,"fail":M} to a pass fraction; this is the unit both the
/// capability-eval and adapt inputs are serialized in.
public struct TesseraAxisTally: Codable, Sendable, Equatable {
    public var pass: Int
    public var fail: Int

    public init(pass: Int = 0, fail: Int = 0) {
        self.pass = pass
        self.fail = fail
    }

    public var total: Int { pass + fail }
    public var fraction: Double { total > 0 ? Double(pass) / Double(total) : 0 }
}

/// The result of scoring a set of instance results, tagged with which backend
/// produced it. The C++ harness is the source of truth when its binary is
/// present; the in-process Swift reduction is the honest fallback (same
/// pass-fraction math, just computed locally).
public struct TesseraCapabilityEvalOutcome: Sendable, Equatable {
    public let score: TesseraCapabilityScore
    public let tallies: [String: TesseraAxisTally]   // keyed by Swift axis name
    public let weightedSum: Double
    public let backend: String                        // "harness" | "swift"
    public let note: String

    public init(
        score: TesseraCapabilityScore,
        tallies: [String: TesseraAxisTally],
        weightedSum: Double,
        backend: String,
        note: String
    ) {
        self.score = score
        self.tallies = tallies
        self.weightedSum = weightedSum
        self.backend = backend
        self.note = note
    }
}

// MARK: - Service

/// Turns per-axis instance results into a TesseraCapabilityScore and reads
/// the same vector through two lenses (weighted-sum scalar, Pareto
/// non-domination). The score VECTOR is the substrate, the lenses are just
/// projections (design 4.7).
///
/// Scoring has two backends: the C++ harness (--tessera-capability-eval) is
/// the source of truth when its binary is installed; the Swift reduction is
/// the fallback when it is not. Both reduce per-axis pass/fail to the same
/// fractions, so the fallback is a degradation of location, not of method.
public struct TesseraCapabilityEvalService {
    /// Swift axis name -> C++ harness JSON key. The harness uses snake_case
    /// (ts_capability_score field names); the Swift score uses camelCase.
    public static let axisToHarnessKey: [String: String] = [
        "mechanical": "mechanical",
        "apiCurrency": "api_currency",
        "hardTail": "hard_tail",
        "personalStyle": "personal_style",
        "generalCompetence": "general_competence",
    ]

    /// Uniform weights over the four optimization axes, matching the harness
    /// (quantize.cpp). The guard axis is excluded by weightedSum, never here.
    public static let uniformWeights: [String: Double] = [
        "mechanical": 0.25, "apiCurrency": 0.25, "hardTail": 0.25, "personalStyle": 0.25,
    ]

    private static let capabilityOutSchema = "llama.tessera.capability.v1"

    public init() {}

    /// Each axis = pass fraction over that axis's instances. Axes with zero
    /// instances score 0. Results naming an unknown axis are ignored.
    public func score(from results: [TesseraEvalInstanceResult]) -> TesseraCapabilityScore {
        let tallies = tally(from: results)
        return TesseraCapabilityScore(
            mechanical: tallies["mechanical"]?.fraction ?? 0,
            apiCurrency: tallies["apiCurrency"]?.fraction ?? 0,
            hardTail: tallies["hardTail"]?.fraction ?? 0,
            personalStyle: tallies["personalStyle"]?.fraction ?? 0,
            generalCompetence: tallies["generalCompetence"]?.fraction ?? 0
        )
    }

    /// Reduce per-instance results to per-axis pass/fail tallies. Every axis
    /// is present (zeroed) so the harness - which requires all five - always
    /// gets a complete vector. Results naming an unknown axis are ignored.
    public func tally(from results: [TesseraEvalInstanceResult]) -> [String: TesseraAxisTally] {
        var tallies: [String: TesseraAxisTally] = [:]
        for axis in TesseraCapabilityScore.axisNames { tallies[axis] = TesseraAxisTally() }
        for result in results {
            guard TesseraCapabilityScore.axisNames.contains(result.axis) else { continue }
            var t = tallies[result.axis] ?? TesseraAxisTally()
            if result.passed { t.pass += 1 } else { t.fail += 1 }
            tallies[result.axis] = t
        }
        return tallies
    }

    /// Serialize per-axis tallies to the harness instances JSON: schema_version
    /// 1, all five axes as {"pass","fail"}, and optional baseline fractions.
    /// This is the shared input shape for BOTH --tessera-capability-eval and
    /// --tessera-adapt (ts_capability_score_load consumes both). All five axes
    /// are always emitted; a missing axis would make the harness fail loudly.
    public func serializeInstancesJSON(
        tallies: [String: TesseraAxisTally],
        baseline: TesseraCapabilityScore? = nil
    ) throws -> Data {
        var axes: [String: HarnessAxisCounts] = [:]
        for axis in TesseraCapabilityScore.axisNames {
            let key = Self.axisToHarnessKey[axis] ?? axis
            let t = tallies[axis] ?? TesseraAxisTally()
            axes[key] = HarnessAxisCounts(pass: t.pass, fail: t.fail)
        }

        var baselineObject: [String: Double]?
        if let baseline {
            var b: [String: Double] = [:]
            for axis in TesseraCapabilityScore.axisNames {
                b[Self.axisToHarnessKey[axis] ?? axis] = baseline[axis]
            }
            baselineObject = b
        }

        return try JSONEncoder().encode(
            HarnessInstancesFile(schema_version: 1, axes: axes, baseline: baselineObject)
        )
    }

    /// Score instance results via the C++ harness when its binary is present
    /// (source of truth), falling back to the in-process Swift reduction
    /// otherwise. Never fabricates: with no results, every axis is 0 because
    /// zero instances were scored, and the note says which backend ran.
    public func scoreResults(
        _ results: [TesseraEvalInstanceResult],
        baseline: TesseraCapabilityScore? = nil
    ) async -> TesseraCapabilityEvalOutcome {
        let tallies = tally(from: results)
        let swiftScore = score(from: results)
        let swiftSum = swiftScore.weightedSum(weights: Self.uniformWeights)

        func fallback(_ note: String) -> TesseraCapabilityEvalOutcome {
            TesseraCapabilityEvalOutcome(
                score: swiftScore, tallies: tallies, weightedSum: swiftSum,
                backend: "swift", note: note
            )
        }

        let binary = TesseraHarnessBinary.path
        guard FileManager.default.isExecutableFile(atPath: binary) else {
            return fallback("harness binary not found at \(binary); scored in Swift")
        }

        // The binary reads the instances from a file and writes the score to a
        // file; both are ephemeral and removed before returning.
        let dir = NSTemporaryDirectory()
        let inPath = (dir as NSString).appendingPathComponent("tessera-cap-\(UUID().uuidString).in.json")
        let outPath = (dir as NSString).appendingPathComponent("tessera-cap-\(UUID().uuidString).out.json")
        defer {
            try? FileManager.default.removeItem(atPath: inPath)
            try? FileManager.default.removeItem(atPath: outPath)
        }

        do {
            let data = try serializeInstancesJSON(tallies: tallies, baseline: baseline)
            try data.write(to: URL(fileURLWithPath: inPath), options: .atomic)
        } catch {
            return fallback("could not stage instances for the harness; scored in Swift")
        }

        let result: ProcessResult
        do {
            result = try await ProcessRunner().run(
                executable: binary,
                arguments: [
                    "--tessera-capability-eval", inPath,
                    "--tessera-capability-out", outPath,
                ]
            )
        } catch {
            return fallback("harness process unavailable (\(error.localizedDescription)); scored in Swift")
        }

        guard result.exitCode == 0 else {
            let detail = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return fallback("harness exited \(result.exitCode)\(detail.isEmpty ? "" : ": " + detail); scored in Swift")
        }

        guard let out = Self.parseCapabilityOut(at: outPath), let harnessScore = out.score else {
            return fallback("harness ran but its score output was unreadable; scored in Swift")
        }

        return TesseraCapabilityEvalOutcome(
            score: Self.scoreFromHarness(harnessScore),
            tallies: tallies,
            weightedSum: out.weighted_sum ?? swiftSum,
            backend: "harness",
            note: "scored by \(binary)"
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

    // MARK: Harness JSON shapes

    private struct HarnessAxisCounts: Codable {
        let pass: Int
        let fail: Int
    }

    private struct HarnessInstancesFile: Codable {
        let schema_version: Int
        let axes: [String: HarnessAxisCounts]   // keyed by harness (snake_case) key
        let baseline: [String: Double]?         // keyed by harness key; optional
    }

    private struct HarnessScoreObject: Codable {
        let mechanical: Double?
        let api_currency: Double?
        let hard_tail: Double?
        let personal_style: Double?
        let general_competence: Double?
    }

    private struct HarnessCapabilityOut: Codable {
        let schema: String?
        let score: HarnessScoreObject?
        let weighted_sum: Double?
        let has_baseline: Bool?
        let baseline: HarnessScoreObject?
    }

    private static func scoreFromHarness(_ s: HarnessScoreObject) -> TesseraCapabilityScore {
        TesseraCapabilityScore(
            mechanical: s.mechanical ?? 0,
            apiCurrency: s.api_currency ?? 0,
            hardTail: s.hard_tail ?? 0,
            personalStyle: s.personal_style ?? 0,
            generalCompetence: s.general_competence ?? 0
        )
    }

    /// Parse the harness score output. Returns nil when the file is missing,
    /// malformed, carries an unexpected schema, or has no score - a harness
    /// pass without a readable score is treated as a failure (caller falls
    /// back to Swift).
    private static func parseCapabilityOut(at path: String) -> HarnessCapabilityOut? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
        guard let decoded = try? JSONDecoder().decode(HarnessCapabilityOut.self, from: data) else { return nil }
        if let schema = decoded.schema, schema != capabilityOutSchema { return nil }
        guard decoded.score != nil else { return nil }
        return decoded
    }
}

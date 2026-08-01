import Foundation

// MARK: - Approver network (autonomy spec section 11)

/// A small, continuously-trained, per-user confidence estimator. Predicts
/// P(user approves | structural features). Leashed: it predicts, never
/// grants; it fails closed; it operates only within the safe envelope the
/// rule-based invariant layer defines (section 11.1).
///
/// Architecture: a shallow MLP (input -> 16 relu -> 8 relu -> 1 sigmoid).
/// Pure Swift, no external ML framework. Trains in milliseconds on the
/// local receipt stream. Fully local, no egress.
public struct TesseraApproverNetwork: Codable, Sendable, Equatable {

    // MARK: Architecture

    /// Feature vector size (section 11.3).
    public static let featureCount = 20
    /// Hidden layer widths.
    static let h1 = 16
    static let h2 = 8

    // Weights and biases (row-major).
    var w1: [Double]   // [h1 x featureCount]
    var b1: [Double]   // [h1]
    var w2: [Double]   // [h2 x h1]
    var b2: [Double]   // [h2]
    var w3: [Double]   // [1 x h2]
    var b3: Double

    /// Training metadata.
    public private(set) var trainedOnReceipts: Int
    public private(set) var lastTrainedAt: Date?
    /// Calibration metrics from the last collapse-guard check.
    public private(set) var lastECE: Double?
    public private(set) var lastDenialRecall: Double?

    // MARK: Init

    /// Xavier-initialized random network.
    public init(seed: UInt64 = 42) {
        var rng = SplitMix64(seed: seed)
        let s1 = (2.0 / Double(Self.featureCount + Self.h1)).squareRoot()
        let s2 = (2.0 / Double(Self.h1 + Self.h2)).squareRoot()
        let s3 = (2.0 / Double(Self.h2 + 1)).squareRoot()
        w1 = (0..<Self.h1 * Self.featureCount).map { _ in rng.nextGaussian() * s1 }
        b1 = [Double](repeating: 0, count: Self.h1)
        w2 = (0..<Self.h2 * Self.h1).map { _ in rng.nextGaussian() * s2 }
        b2 = [Double](repeating: 0, count: Self.h2)
        w3 = (0..<Self.h2).map { _ in rng.nextGaussian() * s3 }
        b3 = 0
        trainedOnReceipts = 0
        lastTrainedAt = nil
        lastECE = nil
        lastDenialRecall = nil
    }

    // MARK: Forward pass

    /// Predict P(approve) for a feature vector. Returns a value in [0, 1].
    public func predict(_ features: [Double]) -> Double {
        let a1 = reluForward(features, w: w1, b: b1, rows: Self.h1, cols: Self.featureCount)
        let a2 = reluForward(a1, w: w2, b: b2, rows: Self.h2, cols: Self.h1)
        let logit = dot(w3, a2) + b3
        return sigmoid(logit)
    }

    /// Forward pass with cached activations for backprop.
    private func forwardWithCache(_ x: [Double]) -> (a1: [Double], a2: [Double], out: Double) {
        let z1 = matVec(w1, x, rows: Self.h1, cols: Self.featureCount)
        let a1 = zip(z1, b1).map { Swift.max(0, $0 + $1) }
        let z2 = matVec(w2, a1, rows: Self.h2, cols: Self.h1)
        let a2 = zip(z2, b2).map { Swift.max(0, $0 + $1) }
        let logit = dot(w3, a2) + b3
        return (a1, a2, sigmoid(logit))
    }

    // MARK: Training (section 11.5)

    /// Train on (features, label) pairs. Labels: 1 = approved, 0 = denied.
    /// Cost-sensitive: denials are weighted by `denialWeight` (section 11.8)
    /// to counter the ~93% approval base rate.
    public mutating func train(
        features: [[Double]],
        labels: [Double],
        denialWeight: Double = 5.0,
        epochs: Int = 20,
        learningRate: Double = 0.01
    ) {
        precondition(features.count == labels.count)
        guard !features.isEmpty else { return }

        for _ in 0..<epochs {
            // Accumulate gradients over the batch.
            var gw1 = [Double](repeating: 0, count: w1.count)
            var gb1 = [Double](repeating: 0, count: b1.count)
            var gw2 = [Double](repeating: 0, count: w2.count)
            var gb2 = [Double](repeating: 0, count: b2.count)
            var gw3 = [Double](repeating: 0, count: w3.count)
            var gb3 = 0.0

            for (x, y) in zip(features, labels) {
                let (a1, a2, out) = forwardWithCache(x)
                // Cost-sensitive weight: denials count more.
                let weight = y < 0.5 ? denialWeight : 1.0
                // Binary cross-entropy gradient at the output.
                let dOut = weight * (out - y)

                // Output layer gradients.
                for j in 0..<Self.h2 {
                    gw3[j] += dOut * a2[j]
                }
                gb3 += dOut

                // Hidden layer 2 gradients (relu derivative).
                var dA2 = [Double](repeating: 0, count: Self.h2)
                for j in 0..<Self.h2 {
                    let z2j = dotRow(w2, j, a1, cols: Self.h1) + b2[j]
                    dA2[j] = z2j > 0 ? dOut * w3[j] : 0
                }
                for j in 0..<Self.h2 {
                    for k in 0..<Self.h1 {
                        gw2[j * Self.h1 + k] += dA2[j] * a1[k]
                    }
                    gb2[j] += dA2[j]
                }

                // Hidden layer 1 gradients (relu derivative).
                var dA1 = [Double](repeating: 0, count: Self.h1)
                for k in 0..<Self.h1 {
                    var sum = 0.0
                    for j in 0..<Self.h2 {
                        sum += dA2[j] * w2[j * Self.h1 + k]
                    }
                    let z1k = dotRow(w1, k, x, cols: Self.featureCount) + b1[k]
                    dA1[k] = z1k > 0 ? sum : 0
                }
                for k in 0..<Self.h1 {
                    for i in 0..<Self.featureCount {
                        gw1[k * Self.featureCount + i] += dA1[k] * x[i]
                    }
                    gb1[k] += dA1[k]
                }
            }

            // SGD update (average over batch).
            let n = Double(features.count)
            for i in 0..<w1.count { w1[i] -= learningRate * gw1[i] / n }
            for i in 0..<b1.count { b1[i] -= learningRate * gb1[i] / n }
            for i in 0..<w2.count { w2[i] -= learningRate * gw2[i] / n }
            for i in 0..<b2.count { b2[i] -= learningRate * gb2[i] / n }
            for i in 0..<w3.count { w3[i] -= learningRate * gw3[i] / n }
            b3 -= learningRate * gb3 / n
        }

        trainedOnReceipts += features.count
        lastTrainedAt = Date()
    }

    // MARK: Collapse guard (section 11.6)

    /// Measure calibration on a held-out set. Returns (ECE, denialRecall).
    /// ECE: expected calibration error over 10 bins.
    /// Denial recall: fraction of actual denials where prediction < 0.5.
    public func calibrationMetrics(
        features: [[Double]],
        labels: [Double]
    ) -> (ece: Double, denialRecall: Double) {
        guard !features.isEmpty else { return (ece: 1.0, denialRecall: 0.0) }

        let bins = 10
        var binCorrect = [Double](repeating: 0, count: bins)
        var binConf = [Double](repeating: 0, count: bins)
        var binCount = [Int](repeating: 0, count: bins)

        var denials = 0
        var denialsCaught = 0

        for (x, y) in zip(features, labels) {
            let raw = predict(x)
            // A diverged candidate net must fail the guard, not trap the
            // process: non-finite predictions clamp to 0.5 (low confidence).
            let p = raw.isFinite ? min(max(raw, 0.0), 1.0) : 0.5
            let bin = min(Int(p * Double(bins)), bins - 1)
            binConf[bin] += p
            binCorrect[bin] += y
            binCount[bin] += 1

            if y < 0.5 {
                denials += 1
                if p < 0.5 { denialsCaught += 1 }
            }
        }

        var ece = 0.0
        let n = Double(features.count)
        for i in 0..<bins where binCount[i] > 0 {
            let avgConf = binConf[i] / Double(binCount[i])
            let avgAcc = binCorrect[i] / Double(binCount[i])
            ece += Double(binCount[i]) / n * abs(avgConf - avgAcc)
        }

        let denialRecall = denials > 0 ? Double(denialsCaught) / Double(denials) : 1.0
        return (ece: ece, denialRecall: denialRecall)
    }

    /// Check the collapse guard. If calibration degraded beyond epsilon or
    /// denial recall dropped below threshold, roll back to `previous` and
    /// return false. Otherwise update stored metrics and return true.
    public mutating func checkCollapseGuard(
        holdoutFeatures: [[Double]],
        holdoutLabels: [Double],
        previous: TesseraApproverNetwork,
        epsilonECE: Double = 0.15,
        minDenialRecall: Double = 0.5
    ) -> Bool {
        let (ece, dr) = calibrationMetrics(features: holdoutFeatures, labels: holdoutLabels)
        lastECE = ece
        lastDenialRecall = dr

        if ece > epsilonECE || dr < minDenialRecall {
            // Roll back.
            self = previous
            return false
        }
        return true
    }

    // MARK: Math helpers

    private func sigmoid(_ x: Double) -> Double {
        1.0 / (1.0 + Foundation.exp(-x))
    }

    private func dot(_ a: [Double], _ b: [Double]) -> Double {
        zip(a, b).reduce(into: 0.0) { $0 += $1.0 * $1.1 }
    }

    private func dotRow(_ w: [Double], _ row: Int, _ x: [Double], cols: Int) -> Double {
        var sum = 0.0
        let base = row * cols
        for i in 0..<cols { sum += w[base + i] * x[i] }
        return sum
    }

    private func matVec(_ w: [Double], _ x: [Double], rows: Int, cols: Int) -> [Double] {
        (0..<rows).map { dotRow(w, $0, x, cols: cols) }
    }

    private func reluForward(_ x: [Double], w: [Double], b: [Double], rows: Int, cols: Int) -> [Double] {
        let z = matVec(w, x, rows: rows, cols: cols)
        return zip(z, b).map { Swift.max(0, $0 + $1) }
    }
}

// MARK: - Feature extraction (section 11.3)

/// Extracts the structural feature vector from an action + context.
/// NEVER reads natural language. Structural only.
public enum TesseraApproverFeatures {

    /// Number of hash buckets for the action-class embedding.
    static let classEmbedDim = 8

    /// Build the feature vector for a pending action + context.
    public static func extract(
        actionClass: String,
        risk: TesseraActionRisk,
        sandboxed: Bool,
        entry: TesseraLearnedPermission?,
        yoloActive: Bool,
        recentDenialRate: Double,
        secondsSinceLastDenial: Double?,
        config: TesseraPermissionConfig
    ) -> [Double] {
        var f = [Double]()

        // Action-class embedding (feature hashing, 8 dims).
        let emb = classEmbedding(actionClass)
        f.append(contentsOf: emb)

        // Risk level (ordinal 0-3, normalized).
        f.append(Double(risk.severity) / 3.0)

        // Sandbox-contained.
        f.append(sandboxed ? 1.0 : 0.0)

        // Ratchet state.
        f.append(logNorm(Double(entry?.consecutiveApprovals ?? 0)))
        f.append(logNorm(Double(entry?.distinctSessions ?? 0)))
        f.append(entry?.granted == true ? 1.0 : 0.0)
        f.append(logNorm(Double(entry?.totalDenials ?? 0)))

        // Session context.
        f.append(yoloActive ? 1.0 : 0.0)
        f.append(recentDenialRate)
        f.append(secondsSinceLastDenial.map { logNorm($0) } ?? 0.0)

        // Dispositional band.
        let floorOrd: Double
        switch config.floor {
        case .restricted: floorOrd = 0.0
        case .standard: floorOrd = 0.5
        case .elevated: floorOrd = 1.0
        }
        f.append(floorOrd)
        f.append(config.ceiling == .anyNonIrreversible ? 1.0 : 0.0)

        // Pad to featureCount if needed.
        while f.count < TesseraApproverNetwork.featureCount {
            f.append(0.0)
        }
        return Array(f.prefix(TesseraApproverNetwork.featureCount))
    }

    /// Feature-hash an action class into a fixed-size embedding.
    static func classEmbedding(_ actionClass: String) -> [Double] {
        var emb = [Double](repeating: 0, count: classEmbedDim)
        // Hash character trigrams into buckets.
        let scalars = Array(actionClass.unicodeScalars)
        for i in 0..<scalars.count {
            let end = min(i + 3, scalars.count)
            let trigram = String(String.UnicodeScalarView(scalars[i..<end]))
            // FNV-1a hashes span the full UInt64 range; reduce in the UInt64
            // domain (a direct Int() conversion traps above Int.max).
            let h = UInt64(TesseraActionClass.stableHash(trigram), radix: 16) ?? 0
            emb[Int(h % UInt64(classEmbedDim))] += 1.0
        }
        // L2 normalize.
        let norm = (emb.reduce(into: 0.0) { $0 += $1 * $1 }).squareRoot()
        if norm > 0 {
            for i in 0..<emb.count { emb[i] /= norm }
        }
        return emb
    }

    /// Log-normalize a non-negative value: log(1 + x) / log(1 + max).
    private static func logNorm(_ x: Double, max: Double = 1000) -> Double {
        Foundation.log(1 + Swift.max(0, x)) / Foundation.log(1 + max)
    }
}

// MARK: - Deterministic RNG

/// SplitMix64: a small, fast, deterministic PRNG for reproducible init.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state = state &+ 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }

    mutating func nextGaussian() -> Double {
        let u1 = Double(next() >> 11) / Double(UInt64.max >> 11)
        let u2 = Double(next() >> 11) / Double(UInt64.max >> 11)
        return (-2.0 * Foundation.log(Swift.max(u1, 1e-10))).squareRoot()
            * Foundation.cos(2.0 * .pi * u2)
    }
}

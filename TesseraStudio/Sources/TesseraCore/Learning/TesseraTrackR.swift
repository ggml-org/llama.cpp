import Foundation

// Track R: unified self-speculating model (design doc section on unified model).
// One trunk with DFlash-seeded and DSpark-seeded drafting heads, regime-routed.
// This file is the SCAFFOLD: protocols + no-op defaults + clear plug-in points.
// Actual head grafting and regime routing require live training runs.

public enum TesseraDraftingRegime: String, Codable, Sendable {
    case dflash   // DFlash-seeded head: fast, lower acceptance
    case dspark   // DSpark-seeded head: rejection-sampling LoRA, higher acceptance target
    case unified  // fused trunk+head, the end-state
}

public struct TesseraDraftingHead: Codable, Sendable, Identifiable {
    public var id: String { regime.rawValue }
    public let regime: TesseraDraftingRegime
    public var modelPath: String?       // nil until a real head is grafted
    public var acceptanceRate: Double   // running mean from production traces
    public var samples: Int
    public var lastUpdated: Date?

    public init(
        regime: TesseraDraftingRegime,
        modelPath: String? = nil,
        acceptanceRate: Double = 0,
        samples: Int = 0,
        lastUpdated: Date? = nil
    ) {
        self.regime = regime
        self.modelPath = modelPath
        self.acceptanceRate = acceptanceRate
        self.samples = samples
        self.lastUpdated = lastUpdated
    }
}

public protocol TesseraHeadRouting: Sendable {
    func heads() -> [TesseraDraftingHead]
    func recordAcceptance(regime: TesseraDraftingRegime, accepted: Bool)
    // Route to the best available head for a given throughput budget.
    // v1: always returns .dflash (the only head with a real implementation).
    func routeRegime(targetTokensPerSec: Double) -> TesseraDraftingRegime
}

/// No-op default for the service locator: reports no grafted heads and routes
/// every budget to .dflash. Keeps the center compiling before a real scaffold
/// is installed, mirroring the other TesseraNoop* defaults.
public struct TesseraNoopHeadRouting: TesseraHeadRouting {
    public init() {}
    public func heads() -> [TesseraDraftingHead] { [] }
    public func recordAcceptance(regime: TesseraDraftingRegime, accepted: Bool) {}
    public func routeRegime(targetTokensPerSec: Double) -> TesseraDraftingRegime { .dflash }
}

/// No-op scaffold. All methods are honest stubs with clear plug-in comments.
public final class TesseraTrackRScaffold: TesseraHeadRouting, @unchecked Sendable {
    private let lock = NSLock()
    private var headsByRegime: [TesseraDraftingRegime: TesseraDraftingHead]

    public init() {
        // Both seeded heads start cold: no grafted model, zero samples.
        headsByRegime = [
            .dflash: TesseraDraftingHead(regime: .dflash),
            .dspark: TesseraDraftingHead(regime: .dspark),
        ]
    }

    public func heads() -> [TesseraDraftingHead] {
        lock.lock(); defer { lock.unlock() }
        return headsByRegime.values.sorted { $0.regime.rawValue < $1.regime.rawValue }
    }

    public func recordAcceptance(regime: TesseraDraftingRegime, accepted: Bool) {
        lock.lock(); defer { lock.unlock() }
        var head = headsByRegime[regime] ?? TesseraDraftingHead(regime: regime)
        let value = accepted ? 1.0 : 0.0
        // Running mean of the acceptance rate, same shape as the teacher
        // assessor's world-gate pass fraction.
        head.acceptanceRate = (head.acceptanceRate * Double(head.samples) + value) / Double(head.samples + 1)
        head.samples += 1
        head.lastUpdated = Date()
        headsByRegime[regime] = head
    }

    public func routeRegime(targetTokensPerSec: Double) -> TesseraDraftingRegime {
        // PLUG-IN POINT: regime routing. v1 has only one real head, so every
        // budget routes to .dflash. Once a DSpark head is grafted and its
        // acceptance rate is tracked, route high-acceptance budgets to .dspark
        // and latency-bound budgets to .dflash here.
        .dflash
    }
}

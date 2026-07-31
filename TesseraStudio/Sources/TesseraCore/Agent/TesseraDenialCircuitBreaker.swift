import Foundation

/// Denial circuit-breaker (S3): the collapse guard made concrete. Tracks recent
/// action outcomes and trips when denials cluster, signalling that the turn or
/// loop should be interrupted. Trips on 3 consecutive denials OR 10 denials in
/// the last 50 actions. Pure and testable.
public final class TesseraDenialCircuitBreaker: @unchecked Sendable {
    /// Consecutive denials that trip the breaker.
    public static let consecutiveLimit = 3
    /// Sliding window size for the rate-based rule.
    public static let windowSize = 50
    /// Denials within `windowSize` that trip the breaker.
    public static let windowLimit = 10

    private let lock = NSLock()
    /// Recent outcomes, newest-last; `true` means denied. Capped at `windowSize`.
    private var outcomes: [Bool] = []

    public init() {}

    /// Record the outcome of one action (`true` = denied).
    public func record(denied: Bool) {
        lock.lock(); defer { lock.unlock() }
        outcomes.append(denied)
        if outcomes.count > Self.windowSize {
            outcomes.removeFirst(outcomes.count - Self.windowSize)
        }
    }

    /// Whether the breaker is tripped: 3 consecutive denials, or at least
    /// `windowLimit` denials within the last `windowSize` actions.
    public var isTripped: Bool {
        lock.lock(); defer { lock.unlock() }

        var consecutive = 0
        for denied in outcomes.reversed() {
            guard denied else { break }
            consecutive += 1
        }
        if consecutive >= Self.consecutiveLimit { return true }

        let windowDenials = outcomes.reduce(0) { $0 + ($1 ? 1 : 0) }
        return windowDenials >= Self.windowLimit
    }

    /// Clear all recorded outcomes.
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        outcomes.removeAll()
    }
}

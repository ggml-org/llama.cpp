import Foundation

// MARK: - Miscalibration detector (autonomy spec section 12)

/// Detects regime shifts in the user's approval behavior: a class (or the
/// global stream) that was consistently approved flips to consistently
/// denied. Response: TIGHTEN. Suspend learned promotion and surface a
/// non-anchoring notice. Complements the circuit breaker (which reacts to
/// denial CLUSTERS) by reacting to denial REGIME SHIFTS.
///
/// Dumb, auditable, receipt-only. No psychology.
public struct TesseraMiscalibrationDetector: Codable, Sendable {

    // MARK: Config

    /// Rolling window size.
    public var windowSize: Int
    /// Approval rate above this is "consistently approved."
    public var hiThreshold: Double
    /// Approval rate below this is "consistently denied."
    public var loThreshold: Double

    // MARK: State

    /// Per-class rolling window of recent outcomes (true = approved).
    private var classWindows: [String: [Bool]]
    /// Global rolling window.
    private var globalWindow: [Bool]
    /// Classes observed in a consistently-approveded regime at least once.
    /// A regime shift is only meaningful for something that WAS approved;
    /// this latch records that, and makes the detector able to fire (the
    /// naive "first half vs second half of one window" comparison is
    /// unsatisfiable when lo < hi/2).
    private var highRegimeSeen: Set<String>
    /// Whether the global stream was ever in a high-approval regime.
    private var globalHighRegimeSeen: Bool
    /// Classes currently tightened (learned promotion suspended).
    public private(set) var tightenedClasses: Set<String>
    /// Whether the global stream is tightened.
    public private(set) var globalTightened: Bool

    // MARK: Init

    public init(
        windowSize: Int = 20,
        hiThreshold: Double = 0.8,
        loThreshold: Double = 0.3
    ) {
        self.windowSize = windowSize
        self.hiThreshold = hiThreshold
        self.loThreshold = loThreshold
        self.classWindows = [:]
        self.globalWindow = []
        self.highRegimeSeen = []
        self.globalHighRegimeSeen = false
        self.tightenedClasses = []
        self.globalTightened = false
    }

    // MARK: Recording

    /// Record an outcome and check for regime shift. Returns true if a
    /// tightening was triggered (caller should notify the user).
    @discardableResult
    public mutating func record(actionClass: String, approved: Bool) -> Bool {
        var triggered = false

        // Per-class window.
        var cw = classWindows[actionClass] ?? []
        cw.append(approved)
        if cw.count > windowSize { cw.removeFirst(cw.count - windowSize) }
        classWindows[actionClass] = cw

        // Global window.
        globalWindow.append(approved)
        if globalWindow.count > windowSize {
            globalWindow.removeFirst(globalWindow.count - windowSize)
        }

        // Check per-class regime shift: was consistently approved, now
        // consistently denied. TIGHTEN (section 12).
        if cw.count >= windowSize {
            let rate = Double(cw.filter { $0 }.count) / Double(cw.count)
            if rate > hiThreshold {
                highRegimeSeen.insert(actionClass)
            }
            if rate < loThreshold,
               highRegimeSeen.contains(actionClass),
               !tightenedClasses.contains(actionClass) {
                tightenedClasses.insert(actionClass)
                triggered = true
            }
            // Auto-recover: if the rate climbs back above hi, un-tighten.
            if rate > hiThreshold && tightenedClasses.contains(actionClass) {
                tightenedClasses.remove(actionClass)
            }
        }

        // Check global regime shift.
        if globalWindow.count >= windowSize {
            let rate = Double(globalWindow.filter { $0 }.count) / Double(globalWindow.count)
            if rate > hiThreshold {
                globalHighRegimeSeen = true
            }
            if rate < loThreshold, globalHighRegimeSeen, !globalTightened {
                globalTightened = true
                triggered = true
            }
            if rate > hiThreshold && globalTightened {
                globalTightened = false
            }
        }

        return triggered
    }

    // MARK: Queries

    /// Whether learned promotion is suspended for a class (either per-class
    /// or global tightening).
    public func isTightened(_ actionClass: String) -> Bool {
        globalTightened || tightenedClasses.contains(actionClass)
    }

    /// Current approval rate for a class (or nil if not enough data).
    public func approvalRate(for actionClass: String) -> Double? {
        guard let cw = classWindows[actionClass], !cw.isEmpty else { return nil }
        return Double(cw.filter { $0 }.count) / Double(cw.count)
    }

    /// Current global approval rate.
    public func globalApprovalRate() -> Double? {
        guard !globalWindow.isEmpty else { return nil }
        return Double(globalWindow.filter { $0 }.count) / Double(globalWindow.count)
    }

    /// Reset all state (e.g. on purge).
    public mutating func reset() {
        classWindows = [:]
        globalWindow = []
        highRegimeSeen = []
        globalHighRegimeSeen = false
        tightenedClasses = []
        globalTightened = false
    }
}

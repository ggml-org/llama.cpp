import Foundation

// MARK: - Pending action

/// A lightweight description of a planned action, assessed before it runs.
public struct PendingAction: Sendable, Equatable {
    public let toolName: String
    public let arguments: [String: JSONValue]

    public init(toolName: String, arguments: [String: JSONValue] = [:]) {
        self.toolName = toolName
        self.arguments = arguments
    }
}

// MARK: - Verifier decision

/// The strict structured verdict returned by an action verifier (S2).
public struct VerifierDecision: Sendable, Equatable {
    public let riskLevel: TesseraActionRisk
    public let authorized: Bool
    public let rationale: String

    public init(riskLevel: TesseraActionRisk, authorized: Bool, rationale: String) {
        self.riskLevel = riskLevel
        self.authorized = authorized
        self.rationale = rationale
    }
}

// MARK: - Verifier protocol

/// Assesses the exact planned action before it runs (S2). Implementations must
/// fail CLOSED: any error, timeout, or malformed input yields `authorized=false`.
public protocol ActionVerifying: Sendable {
    func verify(_ action: PendingAction) -> VerifierDecision
}

// MARK: - Rule-based verifier

/// The default cheap, rule-based verifier (S2). No model call on the healthy
/// path; the injectable `assess` closure is the documented extension point
/// where a model review could be added for the ambiguous tail only.
public struct TesseraActionVerifier: ActionVerifying {
    /// Maps an action to a risk level. May throw; a throw fails closed.
    private let assess: @Sendable (PendingAction) throws -> TesseraActionRisk

    public init(assess: @escaping @Sendable (PendingAction) throws -> TesseraActionRisk = { try TesseraActionVerifier.ruleBasedRisk(for: $0) }) {
        self.assess = assess
    }

    public func verify(_ action: PendingAction) -> VerifierDecision {
        let risk: TesseraActionRisk
        do {
            risk = try assess(action)
        } catch {
            // Fail closed: an erroring assessment never authorizes.
            return VerifierDecision(
                riskLevel: .high,
                authorized: false,
                rationale: "Verification failed closed: \(error)"
            )
        }
        let authorized = risk < .high
        let rationale = authorized
            ? "Risk '\(risk.rawValue)' is within the authorized threshold."
            : "Risk '\(risk.rawValue)' exceeds the authorized threshold."
        return VerifierDecision(riskLevel: risk, authorized: authorized, rationale: rationale)
    }

    /// The default rule-based risk classifier. Read-only verbs are low risk,
    /// mutating/execution verbs are medium, destructive verbs are high, and
    /// unknown tools are treated cautiously (medium), never trusted as low.
    public static func ruleBasedRisk(for action: PendingAction) throws -> TesseraActionRisk {
        let name = action.toolName.lowercased()

        let destructive = ["delete", "remove", "purge", "drop", "erase", "format"]
        if destructive.contains(where: { name.contains($0) }) { return .high }

        let mutating = ["quantize", "calibrate", "evolve", "write", "execute", "run", "shell", "deploy", "export"]
        if mutating.contains(where: { name.contains($0) }) { return .medium }

        let readOnly = ["list", "inspect", "get", "read", "show", "describe"]
        if readOnly.contains(where: { name.contains($0) }) { return .low }

        return .medium
    }
}

// MARK: - Action evidence (S1)

/// A source of state snapshots. A tool captures a pre-state, acts, captures a
/// post-state, and the verifier compares them to confirm a real change (S1).
/// GUI capture is a Wave 2 concern; this is the contract plus a dict-based
/// concrete implementation for tests.
public protocol ActionEvidence: Sendable {
    func snapshot() -> [String: String]
}

/// Dictionary-backed evidence. The capture closure re-reads the current state
/// on each `snapshot()`, so the same instance can produce pre and post states.
public struct TesseraDictionaryEvidence: ActionEvidence {
    private let capture: @Sendable () -> [String: String]

    public init(_ capture: @escaping @Sendable () -> [String: String]) {
        self.capture = capture
    }

    /// Evidence over a fixed, immutable dictionary.
    public init(static values: [String: String]) {
        self.capture = { values }
    }

    public func snapshot() -> [String: String] {
        capture()
    }
}

// MARK: - State-diff verification

extension TesseraActionVerifier {
    /// Confirms the expected key actually changed between two snapshots (S1).
    /// A self-reported success is not enough: the value must differ. Adding or
    /// removing the key counts as a change; equal values (or both absent) do not.
    public static func verifyStateChange(pre: [String: String], post: [String: String], expect key: String) -> Bool {
        pre[key] != post[key]
    }
}

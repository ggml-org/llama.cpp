import Foundation

// MARK: - Action risk

/// Severity of a pending action, shared by the layered-permission decision
/// (S4) and the fail-closed verifier (S2). Ordered by severity so callers can
/// compare thresholds (e.g. `risk < .high`).
public enum TesseraActionRisk: String, Codable, CaseIterable, Sendable, Comparable {
    case low
    case medium
    case high
    /// Never permitted regardless of policy or sandboxing.
    case forbidden

    public var severity: Int {
        switch self {
        case .low: return 0
        case .medium: return 1
        case .high: return 2
        case .forbidden: return 3
        }
    }

    public static func < (lhs: TesseraActionRisk, rhs: TesseraActionRisk) -> Bool {
        lhs.severity < rhs.severity
    }
}

// MARK: - Permission profile

/// How trusted the surrounding tooling/context is. This modulates the safety
/// decision but never overrides the sandbox fail-safe: even an elevated
/// profile only auto-approves a contained, low-risk action.
public enum TesseraPermissionProfile: String, Codable, CaseIterable, Sendable {
    /// Never auto-approve; everything that is not rejected asks the user.
    case restricted
    /// The default fail-safe behaviour.
    case standard
    /// Trusted tooling. Currently behaves like `standard`; reserved for the
    /// scoped-approval gating follow-up (S5).
    case elevated
}

// MARK: - Safety check

/// The layered-permission verdict (S4): the single gate every action passes
/// through before it runs.
public enum TesseraSafetyCheck: String, Codable, CaseIterable, Sendable {
    /// Run without user interaction.
    case autoApprove
    /// Run only after explicit user approval.
    case askUser
    /// Never run.
    case reject
}

// MARK: - Safety decision

/// A layered-permission decision (S4). Computed from
/// approval-policy x permission-profile x sandbox-enforceability x action-risk.
///
/// Fail-safe rule: auto-approve ONLY when the action can actually be
/// contained (sandboxed) AND its risk is low. Everything else asks the user;
/// forbidden actions and disabled tools are rejected outright. Pure and fully
/// unit-testable.
public struct TesseraSafetyDecision: Sendable, Equatable {
    public let approvalPolicy: ApprovalLevel
    public let permissionProfile: TesseraPermissionProfile
    public let sandboxEnforceable: Bool
    public let actionRisk: TesseraActionRisk

    public init(
        approvalPolicy: ApprovalLevel,
        permissionProfile: TesseraPermissionProfile,
        sandboxEnforceable: Bool,
        actionRisk: TesseraActionRisk
    ) {
        self.approvalPolicy = approvalPolicy
        self.permissionProfile = permissionProfile
        self.sandboxEnforceable = sandboxEnforceable
        self.actionRisk = actionRisk
    }

    /// The computed safety check.
    public var check: TesseraSafetyCheck {
        // Forbidden actions and disabled tools never run.
        if actionRisk == .forbidden || approvalPolicy == .denied {
            return .reject
        }
        // An explicit prompt policy always asks the user.
        if approvalPolicy == .prompt {
            return .askUser
        }
        // A restricted profile never auto-approves.
        if permissionProfile == .restricted {
            return .askUser
        }
        // Fail-safe: only auto-approve when contained AND low risk.
        if sandboxEnforceable && actionRisk == .low {
            return .autoApprove
        }
        return .askUser
    }
}

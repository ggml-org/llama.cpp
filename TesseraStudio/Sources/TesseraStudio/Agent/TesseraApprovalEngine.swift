import Foundation
import Observation

/// Approval levels for tool execution.
enum ApprovalLevel: String, Codable, CaseIterable, Sendable {
    /// Execute without user interaction.
    case auto
    /// Execute and show a notification after the fact.
    case notify
    /// Require explicit user approval before execution.
    case prompt
    /// Never execute; the tool is disabled.
    case denied
}

/// Manages per-tool approval levels and user overrides.
/// The prompt level triggers an ApprovalSheet in the UI.
@Observable
@MainActor
final class TesseraApprovalEngine {
    /// User overrides keyed by tool name. Falls back to the tool's default.
    private(set) var overrides: [String: ApprovalLevel] = [:]

    /// Pending approval request (drives the ApprovalSheet presentation).
    private(set) var pendingRequest: PendingApproval?

    /// Callback continuation for the pending request.
    private var continuation: CheckedContinuation<Bool, Never>?

    struct PendingApproval: Identifiable {
        let id = UUID()
        let toolName: String
        let arguments: [String: JSONValue]
        let level: ApprovalLevel
    }

    init() {
        loadOverrides()
    }

    /// The effective approval level for a tool.
    func level(for toolName: String, default defaultLevel: ApprovalLevel) -> ApprovalLevel {
        overrides[toolName] ?? defaultLevel
    }

    /// Set a user override for a tool.
    func setOverride(_ level: ApprovalLevel, for toolName: String) {
        overrides[toolName] = level
        saveOverrides()
    }

    /// Remove a user override, reverting to the tool's default.
    func clearOverride(for toolName: String) {
        overrides.removeValue(forKey: toolName)
        saveOverrides()
    }

    /// Request approval for a tool call. Returns true if approved.
    func requestApproval(toolName: String, arguments: [String: JSONValue]) async -> Bool {
        let tool = TesseraToolRegistry.default.tool(named: toolName)
        let defaultLevel = tool?.defaultApprovalLevel ?? .prompt
        let level = level(for: toolName, default: defaultLevel)

        switch level {
        case .auto:
            return true
        case .notify:
            // In production: post a local notification
            return true
        case .denied:
            return false
        case .prompt:
            return await withCheckedContinuation { cont in
                self.continuation = cont
                self.pendingRequest = PendingApproval(
                    toolName: toolName,
                    arguments: arguments,
                    level: level
                )
            }
        }
    }

    /// Called by the ApprovalSheet when the user responds.
    func resolvePending(approved: Bool) {
        pendingRequest = nil
        continuation?.resume(returning: approved)
        continuation = nil
    }

    // MARK: - Persistence

    private static let storageKey = "tessera.approval.overrides"

    private func saveOverrides() {
        let raw = overrides.mapValues(\.rawValue)
        UserDefaults.standard.set(raw, forKey: Self.storageKey)
    }

    private func loadOverrides() {
        guard let raw = UserDefaults.standard.dictionary(forKey: Self.storageKey) as? [String: String] else { return }
        overrides = raw.compactMapValues { ApprovalLevel(rawValue: $0) }
    }
}

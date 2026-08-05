import Foundation

// MARK: - AgentContext

/// The per-document context the agent sees at prompt time
/// (per spec §6.4). The agent's `LLMProvider.complete(...)`
/// call is given a system prompt that serializes this struct
/// into the shape the model understands.
///
/// The struct is value-typed and `Sendable` so it can be passed
/// across actor boundaries without an extra layer of state. The
/// state machine rebuilds it on every `enqueue` / `markApplied`
/// so the agent's view of the world is always current.
///
/// **Newest-first.** The `pending` array is ordered with the
/// newest item at index 0 (matching the chat panel's front of
/// the queue). The `recentReceipts` array is ordered oldest
/// first (matching the chain order).
///
/// **Cap.** The `pending` and `recentReceipts` arrays are
/// bounded (default `pending` cap 50, `recentReceipts` cap 50)
/// so the prompt doesn't grow unboundedly. The caps are
/// configured on the state machine; the context struct just
/// carries what the state machine chose to include.
public struct AgentContext: Codable, Sendable, Hashable {

    public let documentID: UUID
    public let pending: [ChatQueueItem]
    public let recentReceipts: [Receipt]
    public let documentAST: DocumentAST
    public let builtAt: Date

    public init(
        documentID: UUID,
        pending: [ChatQueueItem],
        recentReceipts: [Receipt],
        documentAST: DocumentAST,
        builtAt: Date = Date()
    ) {
        self.documentID = documentID
        self.pending = pending
        self.recentReceipts = recentReceipts
        self.documentAST = documentAST
        self.builtAt = builtAt
    }

    /// True iff the agent has pending work to do.
    public var hasPending: Bool { !pending.isEmpty }

    /// The first pending item (the head of the queue). nil when
    /// the queue is empty.
    public var frontPending: ChatQueueItem? { pending.first }

    /// Render the context as a stable, human-readable prompt
    /// section. The LLM's system prompt calls this; the model
    /// then reasons about the context.
    public func asPromptSection(now: Date = Date()) -> String {
        var lines: [String] = []
        lines.append("<agent_context>")
        lines.append("  <document_id>\(documentID.uuidString)</document_id>")
        lines.append("  <built_at>\(ISO8601DateFormatter().string(from: builtAt))</built_at>")
        lines.append("  <pending>")
        for item in pending {
            let state = item.state.rawValue
            let marker = item.isSuperseded ? " (superseded)" : ""
            lines.append("    <message id=\"\(item.id.uuidString)\" state=\"\(state)\"\(marker)>")
            lines.append("      \(item.message)")
            lines.append("    </message>")
        }
        lines.append("  </pending>")
        lines.append("  <recent_receipts>")
        for receipt in recentReceipts {
            lines.append("    <receipt id=\"\(receipt.id.uuidString)\" timestamp=\"\(receipt.timestamp)\">")
            lines.append("      \(receipt.summary)")
            lines.append("    </receipt>")
        }
        lines.append("  </recent_receipts>")
        lines.append("</agent_context>")
        return lines.joined(separator: "\n")
    }
}

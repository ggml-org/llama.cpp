import Foundation

// MARK: - TextCursor

/// A position in the document. Phase 1 only needs the data
/// structure (per spec §6.5 and the brief §5); Phase 2's
/// `TesseraTextContentManager` will translate this into a
/// `NSTextLocation` for the platform text view.
///
/// Two cursors can exist in the same document at the same time:
/// the user cursor (the standard text caret) and the agent cursor
/// (where the agent is currently editing). The mutation engine
/// doesn't distinguish them -- it's a UI concern. The data model
/// supports both as values of this type.
public struct TextCursor: Codable, Sendable, Equatable, Hashable {
    /// The block the cursor is in.
    public let blockID: UUID
    /// The character offset within the block's flattened text.
    /// For inline content, this is the offset into the concatenation
    /// of all `InlineRun.text` values. For empty blocks, the only
    /// valid offset is 0.
    public let offset: Int
    /// Upstream vs downstream affinity. The standard text-editing
    /// distinction for ambiguous positions (e.g., at a line
    /// boundary). The phase-2 text content manager uses this to
    /// pick the right rendering side; Phase 1 just stores it.
    public let affinity: Affinity

    public init(blockID: UUID, offset: Int, affinity: Affinity = .downstream) {
        self.blockID = blockID
        self.offset = offset
        self.affinity = affinity
    }

    public enum Affinity: String, Codable, Sendable, Hashable {
        case upstream
        case downstream
    }
}

// MARK: - CursorPair

/// A pair of cursors (user + agent) for the same document. The
/// data model carries both; the editor (Phase 2) is responsible
/// for keeping the offsets current as the document mutates.
///
/// A document can have either cursor missing (no current user
/// focus, no agent edit in flight). The pair is `nil` for either
/// member when the cursor is not active.
public struct CursorPair: Codable, Sendable, Equatable, Hashable {
    public var user: TextCursor?
    public var agent: TextCursor?

    public init(user: TextCursor? = nil, agent: TextCursor? = nil) {
        self.user = user
        self.agent = agent
    }

    /// True if no cursors are present.
    public var isEmpty: Bool { user == nil && agent == nil }

    /// The total number of cursors present.
    public var count: Int { (user == nil ? 0 : 1) + (agent == nil ? 0 : 1) }
}

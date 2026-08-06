import Foundation

// MARK: - NoteChatCommand

/// The vocabulary of chat-panel commands that target the
/// Notes surface. The chat panel's command queue (Phase 3)
/// produces `ChatQueueItem`s whose `message` field carries
/// the user's text; the agent parses the message into one
/// of these typed commands and routes it to the
/// ``NotesViewModel``.
///
/// v1 supports the four most common operations. The
/// parsing rules are deliberately permissive — the agent
/// does the heavy lifting; the command enum is the typed
/// boundary the chat panel and the notes surface agree on.
///
/// **"create a new note titled 'Meeting notes for Q3
/// review'"** — the agent extracts the title and calls
/// ``createNote(title:body:tags:)``. The body starts empty
/// (the user types the body themselves or asks the agent to
/// fill it in a follow-up command).
///
/// **"summarize this article" + a doc selection** — the
/// agent reads the selected document's body, summarizes it,
/// and calls ``createNote(title:body:tags:)`` with the
/// summary as the body. The summary is a `DocumentAST` of
/// one or two paragraphs.
///
/// **"add a tag 'q3-2026' to this note"** — the agent
/// extracts the tag and calls ``addTag(_:to:)``. The tag
/// is normalized through `Note.normalizeTags`.
///
/// **"pin this note" / "archive this note"** — the agent
/// calls ``togglePinned`` or ``toggleArchived``. The toggle
/// preserves the existing state when the user re-issues
/// the command.
///
/// The chat panel is per-document (the spec §6); the notes
/// surface is a sibling surface. The chat panel for a
/// document's chat queue stays per-document; the chat panel
/// for the notes surface is a "global" chat (one queue per
/// surface, not per-note) so the agent can create + edit
/// notes that don't exist yet.
public enum NoteChatCommand: Codable, Sendable, Hashable {
    /// Create a new note with a title + optional body + tags.
    case createNote(title: String, tags: [String])
    /// Edit the currently-active note: replace the body AST.
    case replaceBody(noteID: UUID, body: DocumentAST)
    /// Set a single tag on the note. Idempotent.
    case addTag(noteID: UUID, tag: String)
    /// Remove a single tag from the note. Idempotent.
    case removeTag(noteID: UUID, tag: String)
    /// Set the full tag list. Replaces the existing list.
    case setTags(noteID: UUID, tags: [String])
    /// Pin / unpin / archive / unarchive a note. The booleans
    /// encode the new state directly so the agent doesn't have
    /// to ask the model "what's the current state".
    case setPinned(noteID: UUID, pinned: Bool)
    case setArchived(noteID: UUID, archived: Bool)
    /// Link the note to another graph entity. The
    /// `targetEntityID` is the entity to link to; the chat
    /// panel resolves "this article" / "this contact" to an
    /// entity id before dispatching.
    case link(noteID: UUID, targetEntityID: UUID, linkType: String)
    /// Delete the note. The receipt is `note_delete`. The
    /// command is the only irreversible one — the chat panel
    /// shows a confirmation dialog before dispatching.
    case delete(noteID: UUID)

    // MARK: - Parsing

    /// Best-effort parse of a chat message into a
    /// `NoteChatCommand`. The parser is intentionally
    /// simple: it catches the four canonical phrasings from
    /// the spec ("create a new note titled X", "summarize
    /// this article", "add a tag X to this note", "pin /
    /// archive this note") and leaves the long tail to the
    /// agent model. The returned optional has a `confidence`
    /// hint via the `requiresAgentConfirmation` flag — the
    /// chat panel surfaces "Did you mean to …?" for low
    /// confidence.
    ///
    /// The parser is the static surface the chat panel
    /// uses for "scaffold the typed command" before the
    /// agent refines it. v2 may swap this for a grammar.
    public static func parse(
        message: String,
        activeNoteID: UUID? = nil,
        targetEntityID: UUID? = nil
    ) -> ParsedCommand? {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let lower = trimmed.lowercased()

        // Order matters: "unpin" / "unarchive" must be
        // checked BEFORE the affirmative forms so the
        // substring check doesn't false-match.

        // "create a new note titled X"
        if let title = extractCreateNoteTitle(from: trimmed) {
            return ParsedCommand(
                command: .createNote(title: title, tags: []),
                requiresAgentConfirmation: false
            )
        }

        // "add a tag 'X' to this note"
        if let tag = extractAddTag(from: trimmed), let noteID = activeNoteID {
            return ParsedCommand(
                command: .addTag(noteID: noteID, tag: tag),
                requiresAgentConfirmation: false
            )
        }

        // "remove tag 'X' from this note"
        if let tag = extractRemoveTag(from: trimmed), let noteID = activeNoteID {
            return ParsedCommand(
                command: .removeTag(noteID: noteID, tag: tag),
                requiresAgentConfirmation: false
            )
        }

        // "unpin this note" — must be checked before "pin
        // this note" (substring).
        if activeNoteID != nil {
            if lower.contains("unpin this note") {
                return ParsedCommand(
                    command: .setPinned(noteID: activeNoteID!, pinned: false),
                    requiresAgentConfirmation: false
                )
            }
            if lower.contains("pin this note") {
                return ParsedCommand(
                    command: .setPinned(noteID: activeNoteID!, pinned: true),
                    requiresAgentConfirmation: false
                )
            }
            if lower.contains("unarchive this note") {
                return ParsedCommand(
                    command: .setArchived(noteID: activeNoteID!, archived: false),
                    requiresAgentConfirmation: false
                )
            }
            if lower.contains("archive this note") {
                return ParsedCommand(
                    command: .setArchived(noteID: activeNoteID!, archived: true),
                    requiresAgentConfirmation: false
                )
            }
        }

        // "link this note to <entity>" — agent resolves the
        // target entity id before dispatching. The parser
        // only flags the intent.
        if lower.contains("link this note"), activeNoteID != nil, targetEntityID != nil {
            return ParsedCommand(
                command: .link(
                    noteID: activeNoteID!,
                    targetEntityID: targetEntityID!,
                    linkType: "related_to"
                ),
                requiresAgentConfirmation: false
            )
        }

        // "summarize this article" — agent runs the body
        // extraction, the command is a create with no body
        // (the body is filled in by a follow-up replaceBody
        // command). The chat panel shows a "synthesizing…"
        // chip.
        if lower.contains("summarize") {
            return ParsedCommand(
                command: .createNote(title: "Summary", tags: []),
                requiresAgentConfirmation: true
            )
        }

        return nil
    }

    // MARK: - Apply

    /// Apply this command to a `NotesViewModel`. The command
    /// is the typed boundary between the chat panel and the
    /// notes surface; the apply is the side-effecting part.
    /// Returns the affected `Note` (or nil for commands that
    /// don't return one).
    @discardableResult
    public func apply(to viewModel: NotesViewModel) async throws -> Note? {
        switch self {
        case .createNote(let title, let tags):
            return try await viewModel.chatCreateNote(title: title, body: .empty, tags: tags)

        case .replaceBody(let noteID, let body):
            return try await viewModel.chatEditNote(noteID: noteID) { store in
                try await store.setBody(body, for: noteID)
            }

        case .addTag(let noteID, let tag):
            return try await viewModel.chatEditNote(noteID: noteID) { store in
                try await store.addTag(tag, to: noteID)
            }

        case .removeTag(let noteID, let tag):
            return try await viewModel.chatEditNote(noteID: noteID) { store in
                try await store.removeTag(tag, from: noteID)
            }

        case .setTags(let noteID, let tags):
            return try await viewModel.chatEditNote(noteID: noteID) { store in
                try await store.setTags(tags, for: noteID)
            }

        case .setPinned(let noteID, let pinned):
            return try await viewModel.chatEditNote(noteID: noteID) { store in
                if pinned {
                    return try await store.pin(noteID)
                } else {
                    return try await store.unpin(noteID)
                }
            }

        case .setArchived(let noteID, let archived):
            return try await viewModel.chatEditNote(noteID: noteID) { store in
                if archived {
                    return try await store.archive(noteID)
                } else {
                    return try await store.unarchive(noteID)
                }
            }

        case .link(let noteID, let targetEntityID, let linkType):
            return try await viewModel.chatEditNote(noteID: noteID) { store in
                _ = try await store.link(noteID: noteID, to: targetEntityID, linkType: linkType)
                // The link method's return value is the link,
                // not the note; the editor re-reads.
                if let fresh = try await store.get(id: noteID) {
                    return fresh
                }
                throw NoteStoreError.noteNotFound(id: noteID)
            }

        case .delete(let noteID):
            // The delete command goes through the store, not
            // the view model — the view model has the
            // selection-clearing logic, but the chat panel
            // doesn't have a "currently selected note" so we
            // just delete the row.
            _ = try await viewModel.store.delete(id: noteID)
            return nil
        }
    }

    // MARK: - Parsing helpers

    private static func extractCreateNoteTitle(from message: String) -> String? {
        // Match: "create a new note titled 'X'" or
        //        "create a new note titled X"  (single-line)
        //        "create a note titled X"
        let lower = message.lowercased()
        guard lower.contains("create") && lower.contains("note") else { return nil }
        let patterns = [
            "create a new note titled ",
            "create a note titled ",
            "new note titled ",
        ]
        for pat in patterns {
            if let range = message.range(of: pat, options: .caseInsensitive) {
                let after = message[range.upperBound...]
                let raw = after.trimmingCharacters(in: .whitespacesAndNewlines)
                return stripQuotes(raw)
            }
        }
        return nil
    }

    private static func extractAddTag(from message: String) -> String? {
        let lower = message.lowercased()
        guard lower.contains("add") && lower.contains("tag") else { return nil }
        let patterns = [
            "add a tag ",
            "add tag ",
        ]
        for pat in patterns {
            if let range = message.range(of: pat, options: .caseInsensitive) {
                let after = message[range.upperBound...]
                let raw = after.trimmingCharacters(in: .whitespacesAndNewlines)
                // Strip the "to this note" / "to the note" tail.
                let cleaned = raw.replacingOccurrences(
                    of: " to this note",
                    with: "",
                    options: .caseInsensitive
                ).replacingOccurrences(
                    of: " to the note",
                    with: "",
                    options: .caseInsensitive
                ).trimmingCharacters(in: .whitespacesAndNewlines)
                return stripQuotes(cleaned)
            }
        }
        return nil
    }

    private static func extractRemoveTag(from message: String) -> String? {
        let lower = message.lowercased()
        guard lower.contains("remove") && lower.contains("tag") else { return nil }
        let patterns = [
            "remove tag ",
            "remove the tag ",
        ]
        for pat in patterns {
            if let range = message.range(of: pat, options: .caseInsensitive) {
                let after = message[range.upperBound...]
                let raw = after.trimmingCharacters(in: .whitespacesAndNewlines)
                let cleaned = raw.replacingOccurrences(
                    of: " from this note",
                    with: "",
                    options: .caseInsensitive
                ).replacingOccurrences(
                    of: " from the note",
                    with: "",
                    options: .caseInsensitive
                ).trimmingCharacters(in: .whitespacesAndNewlines)
                return stripQuotes(cleaned)
            }
        }
        return nil
    }

    private static func stripQuotes(_ s: String) -> String {
        var out = s
        let quoteChars: Set<Character> = ["'", "\"", "`"]
        while let first = out.first, quoteChars.contains(first) {
            out.removeFirst()
        }
        while let last = out.last, quoteChars.contains(last) {
            out.removeLast()
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - ParsedCommand

/// The result of a ``NoteChatCommand/parse`` call. Wraps the
/// typed command with a confirmation hint so the chat panel
/// can ask "Did you mean …?" for low-confidence parses.
public struct ParsedCommand: Sendable, Hashable {
    public let command: NoteChatCommand
    /// True when the parse is heuristic and the agent should
    /// re-confirm. The chat panel surfaces a "Did you mean…?"
    /// chip.
    public let requiresAgentConfirmation: Bool

    public init(command: NoteChatCommand, requiresAgentConfirmation: Bool) {
        self.command = command
        self.requiresAgentConfirmation = requiresAgentConfirmation
    }
}

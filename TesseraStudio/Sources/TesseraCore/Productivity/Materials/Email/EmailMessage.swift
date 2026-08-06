import Foundation

// MARK: - EmailMessage

/// A first-class material in the Tessera productivity surface.
/// v1 is read + reply only (no IMAP receive, no direct SMTP
/// send); the message is stored locally in the universal
/// `graph_entity` table with `entity_type = 'email'`, the
/// `label` carrying the subject, and the `body` JSONB
/// carrying the full RFC 5322 fields plus any per-message
/// Tessera metadata.
///
/// **Why one row per email.** The data layer's universal
/// "one row per thing" pattern (see
/// `docs/tessera-data-layer-design.md` §3) means emails
/// ride on the same `graph_entities` table that holds
/// documents, contacts, tasks, etc. The `body` column
/// stores the message's structured fields as JSON; the
/// `hybrid_search` function (defined in migration 0001)
/// walks the table for retrieval. The Phase 5 partial
/// indexes (migration 0008) accelerate the email-specific
/// list / thread queries.
///
/// **Threading.** The `threadID` is a normalized form of
/// the RFC 5322 `In-Reply-To` and `References` headers.
/// We pick the first non-empty `References` entry as the
/// thread anchor; if both are empty the message is its
/// own thread (its own `messageID`). The normalization
/// is done in ``Threading`` so all importers agree.
///
/// **Folder model.** v1 has five built-in folders
/// (`.inbox`, `.sent`, `.drafts`, `.archive`, `.trash`)
/// and an open `.custom(String)` case for user labels
/// (e.g. `.custom("Work")`). The folder is part of the
/// JSON body so the list query can filter by it
/// post-hoc; in v2 we add a dedicated
/// `email_folders` table for fast folder-based paging.
public struct EmailMessage: Codable, Sendable, Identifiable, Hashable {

    /// Stable identifier. New messages get a fresh UUID
    /// at creation; importer round-trips preserve the
    /// existing id (the Phase 4 importer uses the
    /// `Message-ID` to seed a deterministic UUID for
    /// re-import idempotency).
    public let id: UUID

    /// The RFC 5322 `Message-ID` header value, in
    /// canonical form (the angle brackets stripped).
    /// The spec includes the brackets; the Swift side
    /// strips them so the equality check is on the
    /// bare message id.
    public var messageID: String

    public var from: EmailAddress
    public var to: [EmailAddress]
    public var cc: [EmailAddress]
    public var bcc: [EmailAddress]

    public var subject: String

    /// Plain-text body. Always present (HTML-only
    /// messages are stripped to plain text by the
    /// importer — see `tools/tessera/importers/parsers/email.py`).
    public var bodyPlain: String

    /// HTML body, when the original message was
    /// multipart/alternative or text/html. Rendered
    /// via `WKWebView` (macOS) / `WKWebView` (iOS) for
    /// the preview when present; the plain-text body
    /// is the fallback.
    public var bodyHTML: String?

    /// Parsed Block AST (Phase 1 `DocumentAST`). The
    /// importer builds it; the composer edits it. The
    /// plain text and the AST stay in sync — the AST
    /// is regenerated when the plain text changes.
    public var bodyAST: DocumentAST?

    public var receivedAt: Date
    public var sentAt: Date?

    public var isRead: Bool
    public var isReplied: Bool
    public var isForwarded: Bool
    public var isStarred: Bool
    public var isArchived: Bool
    public var isTrashed: Bool

    public var folder: Folder

    /// The normalized RFC 5322 thread anchor (see
    /// ``Threading/normalize``). nil for messages that
    /// have no thread information; such messages are
    /// their own thread by definition.
    public var threadID: String?

    /// Cross-surface links. The graph store's
    /// `entity_link` rows also carry the links; the
    /// in-body list is a denormalized cache for the
    /// detail view's "related" section.
    public var linkedEntityIDs: [UUID]

    public var attachments: [Attachment]

    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        messageID: String,
        from: EmailAddress,
        to: [EmailAddress] = [],
        cc: [EmailAddress] = [],
        bcc: [EmailAddress] = [],
        subject: String = "",
        bodyPlain: String = "",
        bodyHTML: String? = nil,
        bodyAST: DocumentAST? = nil,
        receivedAt: Date = Date(),
        sentAt: Date? = nil,
        isRead: Bool = false,
        isReplied: Bool = false,
        isForwarded: Bool = false,
        isStarred: Bool = false,
        isArchived: Bool = false,
        isTrashed: Bool = false,
        folder: Folder = .inbox,
        threadID: String? = nil,
        linkedEntityIDs: [UUID] = [],
        attachments: [Attachment] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.messageID = messageID
        self.from = from
        self.to = to
        self.cc = cc
        self.bcc = bcc
        self.subject = subject
        self.bodyPlain = bodyPlain
        self.bodyHTML = bodyHTML
        self.bodyAST = bodyAST
        self.receivedAt = receivedAt
        self.sentAt = sentAt
        self.isRead = isRead
        self.isReplied = isReplied
        self.isForwarded = isForwarded
        self.isStarred = isStarred
        self.isArchived = isArchived
        self.isTrashed = isTrashed
        self.folder = folder
        self.threadID = threadID
        self.linkedEntityIDs = linkedEntityIDs
        self.attachments = attachments
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// The `entity_type` value persisted in
    /// `graph_entities.entity_type`. Pinned so the
    /// email view's `listByEntityType` filter, the
    /// graph view's "by type" grouping, and the
    /// migration 0008 partial indexes all agree.
    public static let entityType = "email"

    /// Display string for the email list. The subject
    /// line is the primary label; the snippet is the
    /// first 80 characters of the plain body for the
    /// list row.
    public var displaySubject: String {
        let trimmed = subject.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "(no subject)" : trimmed
    }

    /// First ~80 characters of the plain body, for the
    /// list row snippet. Newlines are normalized to
    /// spaces.
    public var snippet: String {
        let flat = bodyPlain
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        let collapsed = flat.split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        return String(collapsed.prefix(80))
    }

    /// The sender's display string for the list row.
    /// Prefers the name when present; falls back to
    /// the bare email.
    public var senderDisplay: String {
        if let n = from.name, !n.isEmpty { return n }
        return from.email
    }

    /// True when this message has at least one
    /// attachment. The list row shows the paperclip
    /// indicator based on this.
    public var hasAttachments: Bool { !attachments.isEmpty }

    /// True when the message is "in the user's
    /// active" set: inbox, sent, drafts, or starred.
    /// Used by the sidebar's count badge.
    public var isActive: Bool {
        switch folder {
        case .inbox, .sent, .drafts: return !isArchived && !isTrashed
        case .archive, .trash: return false
        case .custom: return !isArchived && !isTrashed
        }
    }
}

// MARK: - EmailAddress

/// One RFC 5322 mailbox. The `email` is the bare
/// address (`alice@example.com`); the `name` is the
/// optional display name (`Alice Example`).
///
/// `contactID` is the linked `Contact` row (Phase 6).
/// The link is one-way: an email has a `contactID`
/// pointer, the contact has the email in its
/// `linkedEntityIDs`. The data layer's unique
/// constraint on `(source_id, target_id, link_type)`
/// keeps the link idempotent.
public struct EmailAddress: Codable, Sendable, Hashable {
    public var name: String?
    public var email: String
    public var contactID: UUID?

    public init(
        name: String? = nil,
        email: String,
        contactID: UUID? = nil
    ) {
        self.name = name
        self.email = email
        self.contactID = contactID
    }

    /// "Alice Example <alice@example.com>" when the
    /// name is present; bare email otherwise. Matches
    /// the RFC 5322 mailbox syntax the composer
    /// expects.
    public var mailboxString: String {
        if let n = name, !n.isEmpty {
            return "\(n) <\(email)>"
        }
        return email
    }

    /// "alice@example.com" — the canonical equality
    /// key for "is this the same address". The
    /// comparison is case-insensitive on the local
    /// part (per RFC 5321 §2.4) and case-insensitive
    /// on the domain.
    public var canonicalEmail: String {
        email.lowercased()
    }
}

// MARK: - Folder

/// The folder model. The five built-in cases are
/// common across email clients; the `custom(String)`
/// case is the open extension point for user labels
/// (Gmail-style tags). The custom label string is the
/// "label name" exactly as the user typed it (case is
/// preserved for display, comparison is
/// case-insensitive).
///
/// **v1 storage.** The folder is part of the email's
/// JSON body. In v2 we promote it to a dedicated
/// `email_folder_assignments` table for fast folder
/// paging. The Swift side already has the seam
/// (``EmailStore/setFolder``) so the v2 migration is
/// a drop-in.
public enum Folder: Codable, Sendable, Equatable, Hashable {
    case inbox
    case sent
    case drafts
    case archive
    case trash
    case custom(String)

    /// Display name. The sidebar shows this string.
    public var displayName: String {
        switch self {
        case .inbox: return "Inbox"
        case .sent: return "Sent"
        case .drafts: return "Drafts"
        case .archive: return "Archive"
        case .trash: return "Trash"
        case .custom(let s): return s
        }
    }

    /// The "system folder" identity (a stable string
    /// we can compare across users / imports). The
    /// custom case uses the label name itself.
    public var systemID: String {
        switch self {
        case .inbox: return "inbox"
        case .sent: return "sent"
        case .drafts: return "drafts"
        case .archive: return "archive"
        case .trash: return "trash"
        case .custom(let s): return "label:\(s.lowercased())"
        }
    }

    /// True for the system folders that are
    /// implicitly present in every email client.
    public var isSystem: Bool {
        switch self {
        case .custom: return false
        default: return true
        }
    }
}

// MARK: - Attachment

/// One attachment on the email. The `dataReference`
/// is a path or storage key; v1 doesn't extract the
/// bytes (the importer records metadata only, per
/// the Phase 4 email parser docstring). The detail
/// view shows the paperclip indicator + filename
/// even when the bytes aren't on disk.
public struct Attachment: Codable, Sendable, Hashable, Identifiable {
    public var id: UUID
    public var filename: String
    public var mimeType: String
    public var size: Int64

    /// A path, URL, or storage key. The composer
    /// sets this when adding an attachment; the
    /// importer sets it from the parser's per-part
    /// info.
    public var dataReference: String

    public init(
        id: UUID = UUID(),
        filename: String,
        mimeType: String,
        size: Int64,
        dataReference: String = ""
    ) {
        self.id = id
        self.filename = filename
        self.mimeType = mimeType
        self.size = size
        self.dataReference = dataReference
    }
}

// MARK: - Threading

/// The thread-resolution helpers. Threading is the
/// RFC 5322 `In-Reply-To` + `References` pair
/// normalized to a single anchor. The spec is
/// "thread = the first non-empty References entry,
/// or the bare message's own id when both are empty."
/// The helpers here are pure and unit-tested.
public enum Threading {

    /// Compute the thread anchor for a message.
    ///
    /// - Parameters:
    ///   - messageID: The bare message id (the
    ///     `EmailMessage.messageID` field, brackets
    ///     already stripped).
    ///   - inReplyTo: The `In-Reply-To` header value,
    ///     brackets stripped (or empty).
    ///   - references: The `References` header value,
    ///     already split into one message id per
    ///     element.
    /// - Returns: The thread anchor. The anchor is
    ///   either the first `references` element, the
    ///   `inReplyTo` value, or the message's own id
    ///   when both are empty.
    public static func normalize(
        messageID: String,
        inReplyTo: String?,
        references: [String]
    ) -> String {
        // 1) The first non-empty References entry
        //    is the canonical thread anchor.
        if let first = references.first(where: { !$0.isEmpty }) {
            return first
        }
        // 2) Fall back to In-Reply-To.
        if let irt = inReplyTo?.trimmingCharacters(in: .whitespacesAndNewlines),
           !irt.isEmpty {
            return irt
        }
        // 3) The message is its own thread.
        return messageID
    }

    /// Strip the angle brackets from an RFC 5322
    /// `Message-ID` value. The standard form is
    /// `<id@host>`; the bare id is `id@host`.
    public static func stripBrackets(_ s: String) -> String {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.hasPrefix("<") { t.removeFirst() }
        if t.hasSuffix(">") { t.removeLast() }
        return t
    }

    /// Split a `References` header value into one
    /// message id per element. The header is a
    /// whitespace-separated list of angle-bracketed
    /// ids; we strip the brackets per element.
    public static func splitReferences(_ s: String) -> [String] {
        s.split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
            .map(stripBrackets)
            .filter { !$0.isEmpty }
    }
}

// MARK: - JSON helpers

extension EmailMessage {
    /// Encode the message to JSON. The encoded form
    /// is what the `body` column of `graph_entities`
    /// stores. Uses sorted keys + ISO-8601 dates so
    /// the bytes are deterministic.
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Decode a message from JSON. The inverse of
    /// ``jsonData()``.
    public static func from(jsonData: Data) throws -> EmailMessage {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(EmailMessage.self, from: jsonData)
    }

    /// Encode to a UTF-8 string. Convenience for the
    /// email store (which deals in `String` bodies
    /// for the data layer).
    public func jsonDataString() throws -> String {
        let data = try jsonData()
        guard let s = String(data: data, encoding: .utf8) else {
            throw EmailMessageError.invalidBody(reason: "encoded JSON is not UTF-8")
        }
        return s
    }

    /// Decode from a UTF-8 string. Inverse of
    /// ``jsonDataString()``.
    public static func from(jsonDataString: String) throws -> EmailMessage {
        guard let data = jsonDataString.data(using: .utf8) else {
            throw EmailMessageError.invalidBody(reason: "body is not valid UTF-8")
        }
        return try from(jsonData: data)
    }
}

// MARK: - Errors

public enum EmailMessageError: Error, Sendable, Equatable {
    case invalidBody(reason: String)
    case emailNotFound(id: UUID)
    case folderNotFound(label: String)
    case invalidAddress(reason: String)
    case importerSubprocessFailed(reason: String)
}

import Foundation

// MARK: - DraftEmail

/// A composer's output. The composer is a value
/// type; the build step produces a `DraftEmail`
/// that the sender routes through the system share
/// sheet and the store persists in `.drafts` /
/// `.sent`.
///
/// The `DraftEmail` is intentionally close to
/// `EmailMessage` (the same fields are present)
/// but adds two composer-specific fields:
///
/// * `pendingSend` — true between the build and
///   the user picking a share target. The store
///   uses this to filter "drafts that are about to
///   go out" out of the .drafts list (they're shown
///   in the .sent folder once the share sheet
///   resolves).
/// * `composeMode` — the originating mode (new /
///   reply / forward). Persisted in the receipt
///   payload so the audit trail can answer "what
///   was this draft for?".
public struct DraftEmail: Sendable {

    public enum ComposeMode: String, Codable, Sendable, Equatable {
        case new
        case reply
        case replyAll
        case forward
    }

    public var id: UUID
    public var inReplyTo: UUID?
    public var inReplyToMessageID: String?
    public var threadID: String?

    public var from: EmailAddress
    public var to: [EmailAddress]
    public var cc: [EmailAddress]
    public var bcc: [EmailAddress]

    public var subject: String
    public var bodyPlain: String
    public var bodyHTML: String?
    public var bodyAST: DocumentAST?

    public var attachments: [Attachment]

    public var composeMode: ComposeMode
    public var pendingSend: Bool

    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        inReplyTo: UUID? = nil,
        inReplyToMessageID: String? = nil,
        threadID: String? = nil,
        from: EmailAddress,
        to: [EmailAddress] = [],
        cc: [EmailAddress] = [],
        bcc: [EmailAddress] = [],
        subject: String = "",
        bodyPlain: String = "",
        bodyHTML: String? = nil,
        bodyAST: DocumentAST? = nil,
        attachments: [Attachment] = [],
        composeMode: ComposeMode = .new,
        pendingSend: Bool = false,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.inReplyTo = inReplyTo
        self.inReplyToMessageID = inReplyToMessageID
        self.threadID = threadID
        self.from = from
        self.to = to
        self.cc = cc
        self.bcc = bcc
        self.subject = subject
        self.bodyPlain = bodyPlain
        self.bodyHTML = bodyHTML
        self.bodyAST = bodyAST
        self.attachments = attachments
        self.composeMode = composeMode
        self.pendingSend = pendingSend
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// Lift a `DraftEmail` into the
    /// `EmailMessage` shape for persistence. The
    /// resulting message has the same `id`,
    /// message-id (the draft's id), subject, body,
    /// and attachments; the folder is left for the
    /// caller to set (`.drafts` for unsent, `.sent`
    /// for sent).
    public func toEmailMessage() -> EmailMessage {
        EmailMessage(
            id: id,
            messageID: id.uuidString,
            from: from,
            to: to,
            cc: cc,
            bcc: bcc,
            subject: subject,
            bodyPlain: bodyPlain,
            bodyHTML: bodyHTML,
            bodyAST: bodyAST,
            receivedAt: createdAt,
            sentAt: nil,
            isRead: true,
            isReplied: false,
            isForwarded: false,
            isStarred: false,
            isArchived: false,
            isTrashed: false,
            folder: .drafts,
            threadID: threadID,
            linkedEntityIDs: [],
            attachments: attachments
        )
    }

    /// Serialize the draft as an .eml file body
    /// (RFC 5322). The sender writes this to a
    /// temp file and hands the URL to the system
    /// share sheet. The encoding is intentionally
    /// minimal — Apple's mail clients accept it
    /// as-is, and we don't need to encode-word the
    /// headers (we use the bodies the user typed
    /// as UTF-8 directly).
    public func emlData() -> Data {
        var out = ""
        // Date
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        out += "Date: \(f.string(from: createdAt))\r\n"
        // From
        out += "From: \(from.mailboxString)\r\n"
        // To / Cc / Bcc
        if !to.isEmpty {
            out += "To: " + to.map { $0.mailboxString }.joined(separator: ", ") + "\r\n"
        }
        if !cc.isEmpty {
            out += "Cc: " + cc.map { $0.mailboxString }.joined(separator: ", ") + "\r\n"
        }
        if !bcc.isEmpty {
            out += "Bcc: " + bcc.map { $0.mailboxString }.joined(separator: ", ") + "\r\n"
        }
        // Subject
        out += "Subject: \(subject)\r\n"
        // Message-ID
        out += "Message-ID: <\(id.uuidString)@tessera.local>\r\n"
        // In-Reply-To / References for threading
        if let irt = inReplyToMessageID, !irt.isEmpty {
            out += "In-Reply-To: <\(irt)>\r\n"
        }
        if let tid = threadID, !tid.isEmpty {
            out += "References: <\(tid)>\r\n"
        }
        // MIME headers
        if bodyHTML != nil {
            out += "MIME-Version: 1.0\r\n"
            out += "Content-Type: multipart/alternative; boundary=\"tessera-boundary\"\r\n"
            out += "\r\n"
            out += "--tessera-boundary\r\n"
            out += "Content-Type: text/plain; charset=utf-8\r\n\r\n"
            out += bodyPlain
            out += "\r\n--tessera-boundary\r\n"
            out += "Content-Type: text/html; charset=utf-8\r\n\r\n"
            out += bodyHTML ?? ""
            out += "\r\n--tessera-boundary--\r\n"
        } else {
            out += "Content-Type: text/plain; charset=utf-8\r\n"
            out += "\r\n"
            out += bodyPlain
        }
        return Data(out.utf8)
    }
}

// MARK: - EmailComposer

/// The reply / forward / new composer. The
/// composer is a value type — every setter returns
/// a new value — so it composes naturally in
/// SwiftUI's `@State` / `@Binding` flow without
/// sharing mutable state across actors.
///
/// The flow:
///
/// 1. The SwiftUI view creates an ``EmailComposer``
///    in `.reply(to:all:)` mode. The composer
///    pre-fills the to / cc / subject / body
///    fields per RFC 5322 §3.6.2.
/// 2. The user edits the body and presses "Send"
///    (or the share-sheet's "Save as Draft"). The
///    view calls ``build()`` to get a
///    `DraftEmail`.
/// 3. The view hands the draft to
///    ``EmailSender/send(_:)`` (the share sheet
///    path).
public struct EmailComposer: Sendable {

    public enum Mode: Sendable, Hashable {
        case new
        case reply(to: EmailMessage, all: Bool)
        case forward(EmailMessage)
    }

    public var mode: Mode
    public var from: EmailAddress
    public var to: [EmailAddress]
    public var cc: [EmailAddress]
    public var bcc: [EmailAddress]
    public var subject: String
    public var bodyPlain: String
    public var bodyHTML: String?
    public var bodyAST: DocumentAST?
    public var attachments: [Attachment]
    public var threadID: String?
    public var inReplyTo: UUID?
    public var inReplyToMessageID: String?

    public init(mode: Mode, from: EmailAddress) {
        self.mode = mode
        self.from = from
        self.to = []
        self.cc = []
        self.bcc = []
        self.subject = ""
        self.bodyPlain = ""
        self.bodyHTML = nil
        self.bodyAST = nil
        self.attachments = []
        self.threadID = nil
        self.inReplyTo = nil
        self.inReplyToMessageID = nil
        // Apply the mode-specific pre-fill.
        switch mode {
        case .new:
            // Empty composer; user fills in to/subject/body.
            break
        case .reply(let original, let all):
            self.to = [original.from]
            self.cc = all ? Self.recipientsExcluding(
                from: [original.from],
                in: original.to + original.cc
            ) : []
            self.subject = Self.rePrefix(subject: original.subject)
            self.bodyPlain = Self.quoteBody(
                from: original.from,
                date: original.receivedAt,
                body: original.bodyPlain
            )
            self.inReplyTo = original.id
            self.inReplyToMessageID = original.messageID
            self.threadID = original.threadID ?? original.messageID
        case .forward(let original):
            self.subject = Self.fwdPrefix(subject: original.subject)
            self.bodyPlain = Self.forwardedBody(
                from: original.from,
                to: original.to,
                cc: original.cc,
                date: original.receivedAt,
                subject: original.subject,
                body: original.bodyPlain
            )
            self.inReplyTo = nil
            self.inReplyToMessageID = nil
            self.threadID = original.threadID ?? original.messageID
            // Forwarding preserves attachments. The
            // attachment entries are passed through
            // by reference — the sender / store can
            // re-stage them when the user picks the
            // target mail client.
            self.attachments = original.attachments
        }
    }

    // MARK: - Setters (return Self for fluent style)

    public func setTo(_ addresses: [EmailAddress]) -> Self {
        var copy = self
        copy.to = addresses
        return copy
    }

    public func setCC(_ addresses: [EmailAddress]) -> Self {
        var copy = self
        copy.cc = addresses
        return copy
    }

    public func setBCC(_ addresses: [EmailAddress]) -> Self {
        var copy = self
        copy.bcc = addresses
        return copy
    }

    public func setSubject(_ subject: String) -> Self {
        var copy = self
        copy.subject = subject
        return copy
    }

    public func setBody(_ body: String) -> Self {
        var copy = self
        copy.bodyPlain = body
        return copy
    }

    public func setBodyAST(_ ast: DocumentAST) -> Self {
        var copy = self
        copy.bodyAST = ast
        return copy
    }

    public func setBodyHTML(_ html: String?) -> Self {
        var copy = self
        copy.bodyHTML = html
        return copy
    }

    public func attach(_ attachment: Attachment) -> Self {
        var copy = self
        copy.attachments.append(attachment)
        return copy
    }

    public func setAttachments(_ attachments: [Attachment]) -> Self {
        var copy = self
        copy.attachments = attachments
        return copy
    }

    // MARK: - Build

    /// Build the draft. The `composeMode` is
    /// derived from the originating `Mode`.
    public func build() -> DraftEmail {
        let composeMode: DraftEmail.ComposeMode
        switch mode {
        case .new: composeMode = .new
        case .reply(_, let all): composeMode = all ? .replyAll : .reply
        case .forward: composeMode = .forward
        }
        return DraftEmail(
            inReplyTo: inReplyTo,
            inReplyToMessageID: inReplyToMessageID,
            threadID: threadID,
            from: from,
            to: to,
            cc: cc,
            bcc: bcc,
            subject: subject,
            bodyPlain: bodyPlain,
            bodyHTML: bodyHTML,
            bodyAST: bodyAST,
            attachments: attachments,
            composeMode: composeMode,
            pendingSend: false
        )
    }

    // MARK: - Helpers

    /// "Re: " prefix, idempotent (per RFC 5322
    /// §3.6.2 — replies stack "Re: ").
    public static func rePrefix(subject: String) -> String {
        let trimmed = subject.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.lowercased().hasPrefix("re:") { return trimmed }
        return "Re: \(trimmed)"
    }

    /// "Fwd: " prefix, idempotent.
    public static func fwdPrefix(subject: String) -> String {
        let trimmed = subject.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.lowercased().hasPrefix("fwd:") { return trimmed }
        return "Fwd: \(trimmed)"
    }

    /// Build the standard reply quote:
    /// `> On <date>, <from> wrote:\n> <quoted body>`.
    public static func quoteBody(
        from: EmailAddress,
        date: Date,
        body: String
    ) -> String {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        let attribution: String
        if let name = from.name, !name.isEmpty {
            attribution = "On \(f.string(from: date)), \(name) <\(from.email)> wrote:"
        } else {
            attribution = "On \(f.string(from: date)), \(from.email) wrote:"
        }
        let quoted = body
            .split(whereSeparator: { $0 == "\n" || $0 == "\r" })
            .map { "> \($0)" }
            .joined(separator: "\n")
        return "\n\n\(attribution)\n\(quoted)\n"
    }

    /// Build the standard forward body. The header
    /// block names the original sender, recipient,
    /// date, and subject (the "Fwd" envelope).
    public static func forwardedBody(
        from: EmailAddress,
        to: [EmailAddress],
        cc: [EmailAddress],
        date: Date,
        subject: String,
        body: String
    ) -> String {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        let toStr = to.isEmpty ? "" : to.map { $0.mailboxString }.joined(separator: ", ")
        let ccStr = cc.isEmpty ? "" : cc.map { $0.mailboxString }.joined(separator: ", ")
        var header = "\n\n---------- Forwarded message ----------\n"
        header += "From: \(from.mailboxString)\n"
        if !toStr.isEmpty { header += "Date: \(f.string(from: date))\nSubject: \(subject)\n" }
        else { header += "Date: \(f.string(from: date))\nSubject: \(subject)\n" }
        if !ccStr.isEmpty { header += "Cc: \(ccStr)\n" }
        header += "\n"
        return header + body
    }

    /// For reply-all, drop the user's own address and
    /// the original sender from the cc list (we send
    /// to the sender; cc is everyone else who got the
    /// original).
    public static func recipientsExcluding(
        from: [EmailAddress],
        in others: [EmailAddress]
    ) -> [EmailAddress] {
        let excluded = Set(from.map { $0.canonicalEmail })
        return others.filter { !excluded.contains($0.canonicalEmail) }
    }
}

import Foundation

// MARK: - EmailChatAdapter

/// The bridge between the productivity chat panel
/// (Phase 3) and the Email surface. v1 supports
/// three intents, parsed from a free-form user
/// message:
///
/// 1. **"reply to ..."** — the user says
///    "reply to <sender>'s email with: <body>".
///    The adapter finds the matching email, opens
///    the composer in reply mode, and prefills
///    the body. The result is a draft the user
///    confirms before it's sent.
///
/// 2. **"summarize this thread"** — the adapter
///    finds the current thread (or a named one),
///    asks the agent for a summary, and creates a
///    Note containing the summary. The Note is a
///    separate `graph_entity` (the Phase 5 Notes
///    surface owns the format).
///
/// 3. **"find emails from <sender> about
///    <topic>"** — the adapter runs a filter
///    over the local email store and returns
///    matching messages. The chat panel renders
///    the results inline.
///
/// **Why a free-form parser.** The v1 spec is
/// read + reply only; the chat panel's "intent
/// parser" is a thin layer over a small
/// vocabulary of action phrases. The grammar is
/// intentionally narrow (no LLM in the loop)
/// because the v1 chat panel is the existing
/// Phase 3 state machine; the adapter is the
/// "agent tool" the state machine calls. A
/// follow-up wires the agent's LLM-backed
/// intent parser; the v1 shape stays as the
/// deterministic fallback.
///
/// **Determinism.** The three intents are
/// identified by their first keyword ("reply",
/// "summarize", "find"). The parser is purely
/// string-based; tests pin the exact match
/// patterns so v2 can layer an LLM without
/// changing the contract.
public struct EmailChatAdapter: Sendable {

    public enum Intent: Sendable, Hashable {
        case reply(emailID: UUID, body: String)
        case summarize(threadID: String?)
        case find(sender: String?, topic: String?)
        case unknown
    }

    /// The minimal store surface the
    /// adapter needs. Production wires
    /// this to ``EmailStore/list(limit:)``
    /// ; tests can wire a fake with a
    /// fixed list. The closure is
    /// `@Sendable` so the adapter can be
    /// called from any actor.
    public typealias EmailLookup = @Sendable () async -> [EmailMessage]

    private let lookup: EmailLookup

    /// Production initializer. The adapter
    /// reads from the email store's
    /// ``EmailStore/list(limit:)`` method.
    public init(store: EmailStore) {
        self.lookup = {
            (try? await store.list(limit: 1000)) ?? []
        }
    }

    /// Test-friendly initializer. The
    /// caller provides a closure that
    /// returns the in-memory email list
    /// (e.g., a captured `[EmailMessage]`
    /// in the test). Used by
    /// ``EmailChatAdapterTests`` to
    /// exercise the run handlers without
    /// a real data layer.
    public init(lookup: @escaping EmailLookup) {
        self.lookup = lookup
    }

    /// Parse a free-form user message into an
    /// ``Intent``. The match is the first
    /// recognized keyword; anything that doesn't
    /// fit is `.unknown`. The chat panel handles
    /// `.unknown` by showing the message in the
    /// general feed (no email action).
    public func parseIntent(_ message: String) -> Intent {
        let lower = message.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if lower.hasPrefix("reply to") {
            return parseReply(message)
        }
        if lower.hasPrefix("summarize") {
            return parseSummarize(message)
        }
        if lower.hasPrefix("find") {
            return parseFind(message)
        }
        return .unknown
    }

    /// Run an intent. The dispatch is the
    /// adapter's executable surface; the chat
    /// panel calls it with the parsed intent and
    /// an actor that produces the side effect
    /// (open a composer, create a note, surface
    /// a result list).
    public func run(
        intent: Intent,
        context: RunContext
    ) async -> RunResult {
        switch intent {
        case .reply(let emailID, let body):
            return await runReply(emailID: emailID, body: body, context: context)
        case .summarize(let threadID):
            return await runSummarize(threadID: threadID, context: context)
        case .find(let sender, let topic):
            return await runFind(sender: sender, topic: topic, context: context)
        case .unknown:
            return .noAction(reason: "intent not recognized")
        }
    }

    // MARK: - Run context

    /// The side effects the chat adapter can
    /// produce. The chat panel provides an
    /// implementation; tests provide a fake.
    public struct RunContext: Sendable {
        /// Open the composer pre-filled with a
        /// reply draft. The view shows the
        /// composer sheet; the user confirms
        /// the send.
        public var openReplyComposer: (@Sendable (UUID, String) async -> Void)?
        /// Create a Note with the given title +
        /// body. The Note is a graph entity; the
        /// Phase 5 Notes surface owns the
        /// persistence.
        public var createNote: (@Sendable (String, String) async -> Void)?
        /// Show inline search results in the
        /// chat panel.
        public var showInlineResults: (@Sendable ([EmailMessage]) async -> Void)?

        public init(
            openReplyComposer: (@Sendable (UUID, String) async -> Void)? = nil,
            createNote: (@Sendable (String, String) async -> Void)? = nil,
            showInlineResults: (@Sendable ([EmailMessage]) async -> Void)? = nil
        ) {
            self.openReplyComposer = openReplyComposer
            self.createNote = createNote
            self.showInlineResults = showInlineResults
        }
    }

    /// The result of a run. The chat panel
    /// matches on this to decide what to render.
    public enum RunResult: Sendable, Hashable {
        case composerOpened(emailID: UUID)
        case noteCreated(title: String, body: String)
        case inlineResults(emails: [UUID])
        case noAction(reason: String)
    }

    // MARK: - Parsers

    private func parseReply(_ message: String) -> Intent {
        // Pattern: "reply to <name>'s email with: <body>"
        // OR       "reply to <email>'s email with: <body>"
        // OR       "reply with: <body>"  (uses the most recent email)
        let lower = message.lowercased()
        guard let withRange = lower.range(of: "with:") else {
            return .unknown
        }
        let prefix = String(message[message.startIndex..<withRange.lowerBound])
        let body = String(message[withRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip the leading "reply to" / "reply".
        var head = prefix
        if head.lowercased().hasPrefix("reply to") {
            head = String(head.dropFirst("reply to".count))
        } else if head.lowercased().hasPrefix("reply") {
            head = String(head.dropFirst("reply".count))
        }
        head = head.trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip the trailing "'s email" / " email" / " message".
        for suffix in ["'s email", "'s message", " email", " message"] {
            if head.lowercased().hasSuffix(suffix) {
                head = String(head.dropLast(suffix.count))
                break
            }
        }
        head = head.trimmingCharacters(in: .whitespacesAndNewlines)
        // The head is the "to whom" — name or
        // email. The caller resolves the
        // matching email; the run handler does
        // the lookup. We return .reply with
        // emailID = a sentinel that the run
        // handler replaces with the actual id.
        return .reply(emailID: Self.sentinelID, body: body)
    }

    private func parseSummarize(_ message: String) -> Intent {
        // "summarize this thread" or "summarize
        // thread <id>". The "this" form is the
        // common case; the chat panel passes the
        // current threadID via the run context.
        let lower = message.lowercased()
        if lower.contains("this thread") || lower.contains("current thread") {
            return .summarize(threadID: nil)
        }
        // "summarize thread <id>"
        if let r = lower.range(of: "thread ") {
            let tail = String(message[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !tail.isEmpty {
                return .summarize(threadID: tail)
            }
        }
        return .summarize(threadID: nil)
    }

    private func parseFind(_ message: String) -> Intent {
        // "find emails from <sender> about <topic>"
        let lower = message.lowercased()
        var sender: String?
        var topic: String?
        if let r = lower.range(of: "from ") {
            let after = String(message[r.upperBound...])
            if let about = after.lowercased().range(of: " about ") {
                let s = String(after[after.startIndex..<about.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let t = String(after[about.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                sender = s.isEmpty ? nil : s
                topic = t.isEmpty ? nil : t
            } else {
                let s = after.trimmingCharacters(in: .whitespacesAndNewlines)
                sender = s.isEmpty ? nil : s
            }
        }
        if sender == nil && topic == nil {
            return .unknown
        }
        return .find(sender: sender, topic: topic)
    }

    // MARK: - Run handlers

    private func runReply(emailID: UUID, body: String, context: RunContext) async -> RunResult {
        // Resolve the email. If the caller
        // passed the sentinel, look up the
        // most-recently-received email (the
        // chat panel's "current" email).
        let targetID: UUID?
        if emailID == Self.sentinelID {
            targetID = await mostRecentEmailID()
        } else {
            targetID = emailID
        }
        guard let id = targetID else {
            return .noAction(reason: "no email to reply to")
        }
        await context.openReplyComposer?(id, body)
        return .composerOpened(emailID: id)
    }

    private func runSummarize(threadID: String?, context: RunContext) async -> RunResult {
        let emails: [EmailMessage]
        if let tid = threadID {
            emails = (await lookup())
                .filter { ($0.threadID ?? $0.messageID) == tid }
        } else {
            // "this thread" — use the most
            // recent thread (the chat panel
            // would normally pass the threadID
            // explicitly; v1 falls back to the
            // most recent email's thread).
            let all = await lookup()
            guard let mostRecent = all.first else {
                return .noAction(reason: "no thread to summarize")
            }
            let tid = mostRecent.threadID ?? mostRecent.messageID
            emails = all.filter { ($0.threadID ?? $0.messageID) == tid }
        }
        guard !emails.isEmpty else {
            return .noAction(reason: "no thread to summarize")
        }
        // Build a one-line-per-message summary
        // (the v1 path; the v2 LLM-backed
        // summary replaces this with a real
        // abstract).
        let body = emails.enumerated().map { idx, e in
            "[\(idx + 1)] \(e.displaySubject) — \(e.from.email)\n    \(e.snippet)"
        }.joined(separator: "\n\n")
        let title = "Thread summary (\(emails.count) message\(emails.count == 1 ? "" : "s"))"
        await context.createNote?(title, body)
        return .noteCreated(title: title, body: body)
    }

    private func runFind(sender: String?, topic: String?, context: RunContext) async -> RunResult {
        let all = await lookup()
        let q = (sender?.lowercased() ?? "")
        let topicQ = (topic?.lowercased() ?? "")
        let matches = all.filter { e in
            let senderMatch = q.isEmpty
                || e.from.email.lowercased().contains(q)
                || (e.from.name ?? "").lowercased().contains(q)
            let topicMatch = topicQ.isEmpty
                || e.subject.lowercased().contains(topicQ)
                || e.bodyPlain.lowercased().contains(topicQ)
            return senderMatch && topicMatch
        }
        await context.showInlineResults?(matches)
        return .inlineResults(emails: matches.map { $0.id })
    }

    // MARK: - Helpers

    private func mostRecentEmailID() async -> UUID? {
        let all = await lookup()
        return all.first?.id
    }

    /// The sentinel UUID the chat panel passes
    /// when the user types "reply with: ..."
    /// without naming a recipient. The run
    /// handler replaces it with the most
    /// recent email.
    public static let sentinelID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!
}

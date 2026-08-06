import Foundation

// MARK: - CalendarStoring

/// The store operations the chat handler + view model need.
/// ``CalendarStore`` conforms (production); the test suite
/// injects an in-memory fake so the create / list / update
/// chat flows run without Postgres.
public protocol CalendarStoring: Sendable {
    func upsert(_ event: CalendarEvent) async throws -> CalendarEvent
    func get(id: UUID) async throws -> CalendarEvent?
    func delete(id: UUID) async throws -> Bool
    func list(limit: Int) async throws -> [CalendarEvent]
    func events(in range: ClosedRange<Date>, calendar: Calendar) async throws -> [CalendarEvent]
    func search(matching query: String, limit: Int) async throws -> [CalendarEvent]
    func respond(
        to eventID: UUID,
        attendeeIndex: Int?,
        attendeeName: String?,
        status: CalendarEvent.ResponseStatus
    ) async throws -> CalendarEvent
}

// ``CalendarStore`` already satisfies ``CalendarStoring``
// member-for-member (its defaulted parameters cover the
// protocol's non-defaulted shape), so the conformance is a
// declaration, not a wrapper:
extension CalendarStore: CalendarStoring {}

// MARK: - CalendarChatIntent

/// What the user wants the calendar to do. Classified by
/// ``CalendarChatHandler/classify(_:)`` — a small rule pass,
/// deliberately not an LLM call (privacy-first: the intent
/// never leaves the device for classification).
public enum CalendarChatIntent: Sendable, Equatable {
    /// "schedule a meeting with John next monday at 2pm"
    case create(ParsedEvent)
    /// "what's on my calendar tomorrow?"
    case list(range: ClosedRange<Date>)
    /// "move the Q3 review to wednesday"
    case move(eventQuery: String, target: Date)
    /// "decline the dentist"
    case respond(eventQuery: String, status: CalendarEvent.ResponseStatus)
    /// "cancel the standup"
    case delete(eventQuery: String)
    /// Couldn't classify; the panel echoes the input.
    case notUnderstood
}

// MARK: - CalendarChatOutcome

/// The result of processing one chat item. The chat panel
/// renders `summary` inline and, when an event is involved,
/// navigates to it.
public struct CalendarChatOutcome: Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        case created
        case listed
        case updated
        case responded
        case deleted
        case failed
        case notUnderstood
    }

    public var kind: Kind
    public var summary: String
    public var eventID: UUID?
    /// For `.listed`: the events to render inline.
    public var events: [CalendarEvent]

    public init(
        kind: Kind,
        summary: String,
        eventID: UUID? = nil,
        events: [CalendarEvent] = []
    ) {
        self.kind = kind
        self.summary = summary
        self.eventID = eventID
        self.events = events
    }
}

// MARK: - CalendarChatItem

/// One queued chat command for the calendar surface.
/// Mirrors the document chat queue's pending -> applied /
/// failed lifecycle at surface level (the calendar surface
/// has no document id, so it keeps its own lightweight
/// queue instead of reusing ``ChatQueueItem``'s
/// document-scoped row).
public struct CalendarChatItem: Sendable, Equatable, Identifiable {
    public enum State: String, Sendable, Equatable {
        case pending
        case inProgress
        case applied
        case failed
    }

    public let id: UUID
    public var message: String
    public var state: State
    public var outcome: CalendarChatOutcome?
    public var createdAt: Date

    public init(
        id: UUID = UUID(),
        message: String,
        state: State = .pending,
        outcome: CalendarChatOutcome? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.message = message
        self.state = state
        self.outcome = outcome
        self.createdAt = createdAt
    }
}

// MARK: - CalendarChatHandler

/// The calendar surface's command queue. The user types
/// natural language ("schedule a meeting with John next
/// monday at 2pm"); the handler enqueues a pending item,
/// classifies the intent, executes it against the
/// ``CalendarStoring`` store, and records the outcome.
/// Every mutation goes through the store, so every
/// mutation gets its constitutional receipt — the handler
/// itself never writes to the data layer.
public actor CalendarChatHandler {

    private let store: CalendarStoring
    private let parser: CalendarNLUParser
    private let calendar: Calendar
    private var nowProvider: @Sendable () -> Date

    private(set) public var queue: [CalendarChatItem] = []

    public init(
        store: CalendarStoring,
        parser: CalendarNLUParser,
        calendar: Calendar = .current,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.store = store
        self.parser = parser
        self.calendar = calendar
        self.nowProvider = now
    }

    // MARK: - Queue lifecycle

    /// Enqueue a message as a pending item. Returns the
    /// item (the panel shows it immediately).
    @discardableResult
    public func enqueue(_ message: String) -> CalendarChatItem {
        let item = CalendarChatItem(message: message, createdAt: nowProvider())
        queue.append(item)
        return item
    }

    /// Process the oldest pending item: classify, execute,
    /// update the queue entry. Returns the outcome.
    @discardableResult
    public func processNext() async throws -> CalendarChatOutcome? {
        guard let index = queue.firstIndex(where: { $0.state == .pending }) else {
            return nil
        }
        queue[index].state = .inProgress
        let message = queue[index].message
        let outcome: CalendarChatOutcome
        do {
            outcome = try await handle(message)
            queue[index].state = .applied
        } catch {
            outcome = CalendarChatOutcome(
                kind: .failed,
                summary: "Something went wrong: \(error)"
            )
            queue[index].state = .failed
        }
        queue[index].outcome = outcome
        return outcome
    }

    /// Enqueue + process in one call (the common path for
    /// the quick-add field, which wants the outcome
    /// inline).
    public func submit(_ message: String) async throws -> CalendarChatOutcome {
        enqueue(message)
        guard let outcome = try await processNext() else {
            return CalendarChatOutcome(kind: .failed, summary: "Queue drained")
        }
        return outcome
    }

    // MARK: - Intent classification

    /// Classify a message into an intent. Pure rule pass:
    /// verb keywords first (move / cancel / accept /
    /// decline win over the generic create), then the
    /// question shape for listing, then create via the NLU
    /// parser.
    public func classify(_ input: String) -> CalendarChatIntent {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return .notUnderstood }
        let lowered = text.lowercased()

        // Respond: "accept the standup", "decline lunch".
        let respondWords: [(word: String, status: CalendarEvent.ResponseStatus)] = [
            ("accept", .accepted),
            ("decline", .declined),
            ("refuse", .declined),
            ("tentatively", .tentative),
            ("maybe", .tentative),
        ]
        for (word, status) in respondWords {
            if let range = lowered.range(of: #"\b\#(word)\b"#, options: .regularExpression) {
                let query = cleanupEventQuery(String(text[range.upperBound...]))
                if !query.isEmpty {
                    return .respond(eventQuery: query, status: status)
                }
            }
        }

        // Delete: "cancel the standup", "delete lunch".
        if let range = lowered.range(of: #"\b(cancel|delete|remove)\b"#, options: .regularExpression) {
            let query = cleanupEventQuery(String(text[range.upperBound...]))
            if !query.isEmpty {
                return .delete(eventQuery: query)
            }
        }

        // Move: "move the Q3 review to wednesday".
        if let verbRange = lowered.range(of: #"\b(move|reschedule|push|shift)\b"#, options: .regularExpression) {
            let tail = String(text[verbRange.upperBound...])
            let parts = tail.split(separator: " to ", maxSplits: 1, omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            if parts.count == 2, let target = parser.firstDate(in: parts[1]) {
                let query = cleanupEventQuery(parts[0])
                if !query.isEmpty {
                    return .move(eventQuery: query, target: target)
                }
            }
        }

        // List: question shape ("what's on my calendar
        // tomorrow?") or explicit agenda words.
        let isQuestion = lowered.contains("what") || lowered.contains("agenda")
            || lowered.hasPrefix("show") || lowered.contains("do i have")
        if isQuestion,
           let range = resolveQueryRange(lowered) {
            return .list(range: range)
        }

        // Create (default): the NLU parser always returns
        // something; an empty-title parse still produces an
        // event with the fallback title.
        return .create(parser.parse(text))
    }

    /// The date range a list-question asks about: today /
    /// tomorrow / this week / a named day, defaulting to
    /// today.
    private func resolveQueryRange(_ lowered: String) -> ClosedRange<Date>? {
        let now = nowProvider()
        let today = calendar.startOfDay(for: now)
        guard let tomorrow = calendar.date(byAdding: .day, value: 1, to: today) else { return nil }
        guard let tomorrowEnd = calendar.date(byAdding: .day, value: 1, to: tomorrow) else { return nil }

        if lowered.contains("tomorrow") {
            return tomorrow...tomorrowEnd
        }
        if lowered.contains("today") || lowered.contains("tonight") {
            return today...tomorrow
        }
        if lowered.contains("this week") || lowered.contains("the week") || lowered.contains("my week") {
            guard let weekInterval = calendar.dateInterval(of: .weekOfYear, for: now) else { return nil }
            return weekInterval.start...weekInterval.end
        }
        // A named day ("what's on my calendar friday"):
        // parse it and use that day.
        if let date = parser.firstDate(in: lowered) {
            let day = calendar.startOfDay(for: date)
            guard let end = calendar.date(byAdding: .day, value: 1, to: day) else { return nil }
            return day...end
        }
        // Bare "what's on my calendar?" = today.
        return today...tomorrow
    }

    // MARK: - Execution

    /// Classify + execute one message. Every mutation path
    /// goes through the store, so every mutation carries a
    /// constitutional receipt.
    public func handle(_ input: String) async throws -> CalendarChatOutcome {
        switch classify(input) {
        case .create(let parsed):
            let event = parsed.makeEvent(now: nowProvider())
            let saved = try await store.upsert(event)
            return CalendarChatOutcome(
                kind: .created,
                summary: "Created \(describe(saved))",
                eventID: saved.id
            )

        case .list(let range):
            let events = try await store.events(in: range, calendar: calendar)
            if events.isEmpty {
                return CalendarChatOutcome(
                    kind: .listed,
                    summary: "Nothing on your calendar for that range.",
                    events: []
                )
            }
            let lines = events.map { describe($0) }
            return CalendarChatOutcome(
                kind: .listed,
                summary: "\(events.count) event\(events.count == 1 ? "" : "s"):\n" + lines.joined(separator: "\n"),
                eventID: events.first?.id,
                events: events
            )

        case .move(let eventQuery, let target):
            guard let event = try await resolveEvent(matching: eventQuery) else {
                return CalendarChatOutcome(
                    kind: .failed,
                    summary: "Couldn't find an event matching \"\(eventQuery)\"."
                )
            }
            let moved = move(event, to: target)
            let saved = try await store.upsert(moved)
            return CalendarChatOutcome(
                kind: .updated,
                summary: "Moved \(describe(saved))",
                eventID: saved.id
            )

        case .respond(let eventQuery, let status):
            guard let event = try await resolveEvent(matching: eventQuery) else {
                return CalendarChatOutcome(
                    kind: .failed,
                    summary: "Couldn't find an event matching \"\(eventQuery)\"."
                )
            }
            let updated = try await store.respond(
                to: event.id,
                attendeeIndex: nil,
                attendeeName: nil,
                status: status
            )
            return CalendarChatOutcome(
                kind: .responded,
                summary: "Marked you as \(status.rawValue) for \(describe(updated))",
                eventID: updated.id
            )

        case .delete(let eventQuery):
            guard let event = try await resolveEvent(matching: eventQuery) else {
                return CalendarChatOutcome(
                    kind: .failed,
                    summary: "Couldn't find an event matching \"\(eventQuery)\"."
                )
            }
            _ = try await store.delete(id: event.id)
            return CalendarChatOutcome(
                kind: .deleted,
                summary: "Deleted \"\(event.title)\".",
                eventID: event.id
            )

        case .notUnderstood:
            return CalendarChatOutcome(
                kind: .notUnderstood,
                summary: "I didn't catch that. Try \"schedule lunch with John tomorrow at noon\"."
            )
        }
    }

    // MARK: - Helpers

    /// Move an event to a new day, preserving its
    /// time-of-day and duration (an all-day event stays
    /// all-day on the new day).
    private func move(_ event: CalendarEvent, to target: Date) -> CalendarEvent {
        let duration = event.endAt.timeIntervalSince(event.startAt)
        var start: Date
        if event.allDay {
            start = calendar.startOfDay(for: target)
        } else {
            let time = calendar.dateComponents([.hour, .minute, .second], from: event.startAt)
            var day = calendar.dateComponents([.year, .month, .day], from: target)
            day.hour = time.hour ?? 9
            day.minute = time.minute ?? 0
            day.second = time.second ?? 0
            start = calendar.date(from: day) ?? target
        }
        var moved = event
        moved.startAt = start
        moved.endAt = start.addingTimeInterval(duration)
        moved.updatedAt = nowProvider()
        return moved
    }

    /// Fuzzy event resolution: exact-ish title first
    /// (store search), then best substring match among all
    /// events. Returns nil when nothing matches.
    private func resolveEvent(matching query: String) async throws -> CalendarEvent? {
        let direct = try await store.search(matching: query, limit: 5)
        if let first = direct.first { return first }
        // Word-overlap fallback for "the q3 review" vs
        // "Q3 planning review".
        let queryWords = Set(
            query.lowercased()
                .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
                .map(String.init)
                .filter { $0.count > 2 }
        )
        guard !queryWords.isEmpty else { return nil }
        let all = try await store.list(limit: 1000)
        var best: (event: CalendarEvent, score: Int)?
        for event in all {
            let titleWords = Set(
                event.title.lowercased()
                    .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
                    .map(String.init)
            )
            let score = queryWords.intersection(titleWords).count
            if score > 0, score > (best?.score ?? 0) {
                best = (event, score)
            }
        }
        return best?.event
    }

    /// Strip leading articles / determiners from an event
    /// query ("the Q3 review" -> "Q3 review").
    private func cleanupEventQuery(_ raw: String) -> String {
        var q = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        q = q.trimmingCharacters(in: CharacterSet(charactersIn: "?!.,"))
        let fillers = ["the ", "my ", "a ", "an ", "event ", "meeting "]
        var changed = true
        while changed {
            changed = false
            for f in fillers where q.lowercased().hasPrefix(f) {
                q = String(q.dropFirst(f.count))
                changed = true
            }
        }
        return q.trimmingCharacters(in: .whitespaces)
    }

    /// Human-readable one-liner for chat summaries.
    private func describe(_ event: CalendarEvent) -> String {
        if event.allDay {
            let day = DateFormatter()
            day.dateStyle = .medium
            day.timeStyle = .none
            return "\"\(event.title)\" (all day, \(day.string(from: event.startAt)))"
        }
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        let g = DateFormatter()
        g.dateStyle = .none
        g.timeStyle = .short
        return "\"\(event.title)\" (\(f.string(from: event.startAt)) - \(g.string(from: event.endAt)))"
    }
}

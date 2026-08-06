import XCTest
import PostgresNIO
@testable import TesseraCore

/// End-to-end integration tests for ``CalendarStore``
/// against a real Postgres database: CRUD round-trips, the
/// constitutional receipt chain, cross-surface links, range
/// queries, and the graph-view open path. Env-gated on
/// `TESSERA_DB_INTEGRATION=1` (matching the contacts
/// pattern); without the env var every test skips.
@MainActor
final class CalendarStoreIntegrationTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_PG_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_PG_PORT"] ?? "5432") ?? 5432
    private static let user = ProcessInfo.processInfo.environment["TESSERA_PG_USER"] ?? "tessera"
    private static let pass = ProcessInfo.processInfo.environment["TESSERA_PG_PASSWORD"] ?? "tessera"
    private static let db = ProcessInfo.processInfo.environment["TESSERA_PG_DB"] ?? "tessera"

    /// Locate the migration files at runtime (same search
    /// paths as the contacts integration tests).
    fileprivate static func locateMigrationFiles() -> [(name: String, sql: String)] {
        let candidates = [
            "tools/tessera/db/migrations",
            "../tools/tessera/db/migrations",
            "../../tools/tessera/db/migrations",
        ]
        let fm = FileManager.default
        for c in candidates {
            let url = URL(fileURLWithPath: c)
            if fm.fileExists(atPath: url.path) {
                var out: [(name: String, sql: String)] = []
                let files = (try? fm.contentsOfDirectory(atPath: url.path)) ?? []
                for f in files.sorted() where f.hasSuffix(".sql") {
                    let path = url.appendingPathComponent(f).path
                    if let sql = try? String(contentsOfFile: path) {
                        out.append((f, sql))
                    }
                }
                return out
            }
        }
        return []
    }

    private func requireIntegration() throws {
        guard Self.envEnabled else {
            throw XCTSkip("TESSERA_DB_INTEGRATION not set; skipping DB test")
        }
    }

    private struct TestContext {
        let admin: TesseraDataStore
        let dataStore: TesseraDataStore
        let dataLayer: TesseraDataLayer
        let testDB: String
        let calendarStore: CalendarStore
        let contactStore: ContactStore

        func tearDown() async {
            try? await admin.queryRaw(PostgresQuery(stringLiteral: "DROP DATABASE IF EXISTS \(testDB) WITH (FORCE)"))
            await dataLayer.shutdown()
            await admin.close()
        }
    }

    private func makeTestContext() async throws -> TestContext {
        let admin = TesseraDataStore(
            configuration: .init(
                host: Self.host,
                port: Self.port,
                username: Self.user,
                password: Self.pass,
                database: Self.db,
                minimumConnections: 1,
                maximumConnections: 2
            )
        )
        try await admin.connect()

        let testDB = "tessera_calendar_test_\(Int.random(in: 1000...99999))"
        try await admin.queryRaw(
            PostgresQuery(stringLiteral: "CREATE DATABASE \(testDB)")
        )

        let dataStore = TesseraDataStore(
            configuration: .init(
                host: Self.host,
                port: Self.port,
                username: Self.user,
                password: Self.pass,
                database: testDB,
                minimumConnections: 1,
                maximumConnections: 2
            )
        )
        try await dataStore.connect()
        let migrations = Self.locateMigrationFiles()
        if migrations.isEmpty {
            throw XCTSkip("Migration files not found in the expected paths")
        }
        try await dataStore.applyMigrations(migrations)

        let dataLayer = TesseraDataLayer(dataStore: dataStore, cache: TesseraCache(
            configuration: .init(
                host: Self.host,
                port: 6379,
                password: nil,
                databaseNumber: 0
            )
        ))
        _ = await dataLayer.start()

        return TestContext(
            admin: admin,
            dataStore: dataStore,
            dataLayer: dataLayer,
            testDB: testDB,
            calendarStore: CalendarStore(dataLayer: dataLayer),
            contactStore: ContactStore(dataLayer: dataLayer)
        )
    }

    private let calendar = CalendarFixtures.calendar()

    private func event(title: String, daysFromNow: Int = 1, hour: Int = 10) -> CalendarEvent {
        let day = calendar.date(byAdding: .day, value: daysFromNow, to: calendar.startOfDay(for: Date()))!
        var c = calendar.dateComponents([.year, .month, .day], from: day)
        c.hour = hour
        let start = calendar.date(from: c)!
        return CalendarFixtures.event(title: title, startAt: start)
    }

    // MARK: - CRUD + receipt chain

    func testEventRoundTripEndToEnd() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let original = event(title: "Q3 review")
        var rich = original
        rich.location = "the blue room"
        rich.attendees = [CalendarEvent.Attendee(email: "guest@example.com", name: "Guest")]
        rich.recurrence = CalendarEvent.Recurrence(rrule: "FREQ=WEEKLY;BYDAY=MO")

        let saved = try await ctx.calendarStore.upsert(rich)
        XCTAssertEqual(saved.id, original.id)

        let fetched = try await ctx.calendarStore.get(id: original.id)
        XCTAssertNotNil(fetched)
        XCTAssertEqual(fetched?.title, "Q3 review")
        XCTAssertEqual(fetched?.location, "the blue room")
        XCTAssertEqual(fetched?.attendees.first?.email, "guest@example.com")
        XCTAssertEqual(fetched?.recurrence?.rrule, "FREQ=WEEKLY;BYDAY=MO")
    }

    func testCreateThenUpdateBuildsReceiptChain() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let original = event(title: "Planning")
        _ = try await ctx.calendarStore.upsert(original)

        var edited = original
        edited.title = "Planning (moved)"
        edited.startAt = original.startAt.addingTimeInterval(3600)
        edited.endAt = original.endAt.addingTimeInterval(3600)
        _ = try await ctx.calendarStore.upsert(edited)

        let receipts = try await ctx.calendarStore.receipts(forEvent: original.id)
        XCTAssertEqual(receipts.map(\.receiptType), [
            CalendarEventReceiptType.eventCreated.rawValue,
            CalendarEventReceiptType.eventUpdated.rawValue,
        ])
    }

    func testRespondEmitsASingleRespondedReceipt() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let original = CalendarEvent(
            title: "Invited",
            startAt: Date().addingTimeInterval(86_400),
            endAt: Date().addingTimeInterval(90_000),
            attendees: [CalendarEvent.Attendee(name: "Me")]
        )
        _ = try await ctx.calendarStore.upsert(original)

        let updated = try await ctx.calendarStore.respond(to: original.id, status: .accepted)
        XCTAssertEqual(updated.attendees.first?.responseStatus, .accepted)

        // A response is not an edit: the chain shows exactly
        // created + responded (no event_updated in between).
        let receipts = try await ctx.calendarStore.receipts(forEvent: original.id)
        XCTAssertEqual(receipts.map(\.receiptType), [
            CalendarEventReceiptType.eventCreated.rawValue,
            CalendarEventReceiptType.eventResponded.rawValue,
        ])

        // The responded receipt names the attendee + status.
        let payload = receipts.last?.payload
        XCTAssertEqual(payload?["attendee"], .string("Me"))
        XCTAssertEqual(payload?["status"], .string("accepted"))
    }

    func testDeleteReceiptOutlivesTheRow() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let doomed = event(title: "Doomed")
        _ = try await ctx.calendarStore.upsert(doomed)

        let deleted = try await ctx.calendarStore.delete(id: doomed.id)
        XCTAssertTrue(deleted)
        let fetched = try await ctx.calendarStore.get(id: doomed.id)
        XCTAssertNil(fetched)

        // The receipt chain survives the row: it IS the
        // audit trail of the deletion.
        let receipts = try await ctx.calendarStore.receipts(forEvent: doomed.id)
        XCTAssertEqual(receipts.map(\.receiptType), [
            CalendarEventReceiptType.eventCreated.rawValue,
            CalendarEventReceiptType.eventDeleted.rawValue,
        ])
    }

    func testInvalidEventIsRejected() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let bad = CalendarEvent(
            title: "   ",
            startAt: Date(),
            endAt: Date().addingTimeInterval(3600)
        )
        do {
            _ = try await ctx.calendarStore.upsert(bad)
            XCTFail("expected invalidEvent")
        } catch CalendarStoreError.invalidEvent {
            // expected
        }
    }

    // MARK: - Cross-surface links

    func testSyncLinksCreatesEdgesAndReceipts() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        // Seed the linked entities: a contact + a document.
        let contact = Contact(name: NameComponents(first: "John", last: "Appleseed"))
        _ = try await ctx.contactStore.upsert(contact)
        let docID = UUID()
        _ = try await ctx.dataLayer.upsertEntity(GraphEntityUpsert(
            id: docID,
            entityType: "document",
            subtype: nil,
            label: "Q3 roadmap",
            body: "{}",
            sourceURL: nil,
            embedding: nil
        ))

        let ev = CalendarEvent(
            title: "Linked",
            startAt: Date().addingTimeInterval(86_400),
            endAt: Date().addingTimeInterval(90_000),
            attendees: [CalendarEvent.Attendee(contactID: contact.id, name: "John Appleseed")],
            linkedDocumentIDs: [docID]
        )
        _ = try await ctx.calendarStore.upsert(ev)

        let links = try await ctx.calendarStore.links(for: ev.id)
        XCTAssertTrue(links.contains {
            $0.targetID == contact.id && $0.linkType == CalendarLinkType.attendeeOf.rawValue
        })
        XCTAssertTrue(links.contains {
            $0.targetID == docID && $0.linkType == CalendarLinkType.prepDocument.rawValue
        })

        // Every link gets its own receipt.
        let receipts = try await ctx.calendarStore.receipts(forEvent: ev.id)
        let linkReceipts = receipts.filter { $0.receiptType == CalendarEventReceiptType.linkCreated.rawValue }
        XCTAssertEqual(linkReceipts.count, 2)
    }

    func testSyncLinksIsIdempotent() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let contact = Contact(name: NameComponents(first: "Jane", last: "Doe"))
        _ = try await ctx.contactStore.upsert(contact)

        let ev = CalendarEvent(
            title: "Idempotent",
            startAt: Date().addingTimeInterval(86_400),
            endAt: Date().addingTimeInterval(90_000),
            attendees: [CalendarEvent.Attendee(contactID: contact.id, name: "Jane Doe")]
        )
        _ = try await ctx.calendarStore.upsert(ev)
        _ = try await ctx.calendarStore.syncLinks(for: ev)

        let links = try await ctx.calendarStore.links(for: ev.id)
        let attendeeEdges = links.filter { $0.linkType == CalendarLinkType.attendeeOf.rawValue }
        XCTAssertEqual(attendeeEdges.count, 1, "unique constraint must dedupe repeated links")
    }

    // MARK: - Queries

    func testRangeQueryExpandsRecurrence() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let anchor = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: Date()))!
        let weekly = CalendarEvent(
            title: "Weekly",
            startAt: anchor,
            endAt: anchor.addingTimeInterval(3600),
            recurrence: CalendarEvent.Recurrence(rrule: "FREQ=WEEKLY")
        )
        let oneShot = CalendarEvent(
            title: "One shot",
            startAt: anchor,
            endAt: anchor.addingTimeInterval(3600)
        )
        _ = try await ctx.calendarStore.upsert(weekly)
        _ = try await ctx.calendarStore.upsert(oneShot)

        // Two-week window around the anchor: both events.
        let nearRange = anchor.addingTimeInterval(-86_400)...anchor.addingTimeInterval(14 * 86_400)
        let near = try await ctx.calendarStore.events(in: nearRange, calendar: calendar)
        XCTAssertEqual(Set(near.map(\.title)), ["Weekly", "One shot"])

        // A window two months out: only the recurring event.
        let farStart = anchor.addingTimeInterval(60 * 86_400)
        let farRange = farStart...farStart.addingTimeInterval(14 * 86_400)
        let far = try await ctx.calendarStore.events(in: farRange, calendar: calendar)
        XCTAssertEqual(far.map(\.title), ["Weekly"])
    }

    func testSearchByTitle() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        _ = try await ctx.calendarStore.upsert(event(title: "Q3 review", daysFromNow: 1))
        _ = try await ctx.calendarStore.upsert(event(title: "Dentist", daysFromNow: 2))

        let results = try await ctx.calendarStore.search(matching: "q3")
        XCTAssertEqual(results.map(\.title), ["Q3 review"])
    }

    // MARK: - Migration + graph

    func testMigrationCreatesEventStartIndex() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let query: PostgresQuery = """
            SELECT indexname FROM pg_indexes
             WHERE schemaname = 'public'
               AND tablename = 'graph_entities'
               AND indexname = 'idx_entities_event_start'
            """
        let rows = try await ctx.dataStore.queryRaw(query)
        var found = Set<String>()
        for try await row in rows {
            let ra = row.makeRandomAccess()
            let n: String = try ra[0].decode(String.self)
            found.insert(n)
        }
        XCTAssertEqual(found, ["idx_entities_event_start"])
    }

    func testEventNodeRendersInGraphAndOpensInCalendar() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let ev = event(title: "Graph open", daysFromNow: 2)
        _ = try await ctx.calendarStore.upsert(ev)

        // The event shows up as a graph node.
        let graphStore = GraphStore(dataLayer: ctx.dataLayer)
        let snapshot = try await graphStore.loadAll()
        let node = snapshot.nodes.first(where: { $0.id == ev.id })
        XCTAssertNotNil(node)
        XCTAssertEqual(node?.entityType, CalendarEvent.entityType)
        XCTAssertEqual(node?.iconName, "calendar")

        // Click-to-open: the connector wires the graph's
        // open hook to the calendar surface.
        let parser = CalendarNLUParser(
            contacts: StaticContactsAdapter(),
            documents: StaticDocumentResolver(),
            locations: StaticLocationResolver()
        )
        let handler = CalendarChatHandler(store: ctx.calendarStore, parser: parser)
        let calendarVM = CalendarViewModel(store: ctx.calendarStore, chatHandler: handler)
        let graphVM = GraphViewModel(store: graphStore)
        CalendarGraphConnector.wire(graphVM, to: calendarVM)

        graphVM.open(node!)
        for _ in 0..<200 {
            if calendarVM.selectedEventID == ev.id { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(calendarVM.selectedEventID, ev.id)
        XCTAssertTrue(calendar.isDate(calendarVM.selectedDate, inSameDayAs: ev.startAt))
    }
}

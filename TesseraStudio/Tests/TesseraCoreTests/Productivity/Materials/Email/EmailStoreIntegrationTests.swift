import XCTest
import PostgresNIO
@testable import TesseraCore

/// End-to-end integration tests for
/// ``EmailStore`` against a real Postgres
/// database. The test is env-gated on
/// `TESSERA_DB_INTEGRATION=1` (matching the
/// `ContactStoreIntegrationTests` pattern).
/// When the env var is not set, every test
/// calls `XCTSkip(...)` so `swift test` works
/// in environments without a running DB.
///
/// The v1 worker fills in the basic upsert +
/// receipt + fetch flow; the
/// `markRead` / `setStarred` / `setFolder` /
/// `link` mutations are exercised in
/// follow-up waves (each is one extra test
/// in this file once the data layer migration
/// is in place).
final class EmailStoreIntegrationTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_PG_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_PG_PORT"] ?? "5432") ?? 5432
    private static let user = ProcessInfo.processInfo.environment["TESSERA_PG_USER"] ?? "tessera"
    private static let pass = ProcessInfo.processInfo.environment["TESSERA_PG_PASSWORD"] ?? "tessera"
    private static let db = ProcessInfo.processInfo.environment["TESSERA_PG_DB"] ?? "tessera"

    /// Locate the migration files at runtime so
    /// the test works whether `swift test` is
    /// run from the package root, the test
    /// target dir, or the repo root.
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
                let sorted = files.sorted()
                for f in sorted where f.hasSuffix(".sql") {
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
        let dataLayer: TesseraDataLayer
        let testDB: String
        let emailStore: EmailStore

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

        let testDB = "tessera_email_test_\(Int.random(in: 1000...99999))"
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

        // Apply all migrations in the repo.
        let migrations = Self.locateMigrationFiles()
        try await dataStore.applyMigrations(migrations)

        let dataLayer = TesseraDataLayer(
            dataStore: dataStore,
            cache: TesseraCache(configuration: .init())
        )
        _ = await dataLayer.start()

        let emailStore = EmailStore(dataLayer: dataLayer)

        return TestContext(
            admin: admin,
            dataLayer: dataLayer,
            testDB: testDB,
            emailStore: emailStore
        )
    }

    /// Upsert → fetch round-trip is the
    /// load-bearing happy path. Verifies that
    /// the email's JSON body survives the
    /// Postgres round-trip, the receipt is
    /// appended, and the receipt chain is
    /// queryable.
    func testUpsertAndFetch() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let email = EmailMessage(
            messageID: "msg@example.com",
            from: EmailAddress(name: "Alice", email: "alice@example.com"),
            to: [EmailAddress(email: "bob@example.com")],
            subject: "Hello",
            bodyPlain: "Body",
            receivedAt: date,
            folder: .inbox,
            createdAt: date,
            updatedAt: date
        )
        let stored = try await ctx.emailStore.upsert(email)
        XCTAssertEqual(stored.id, email.id)

        let fetched = try await ctx.emailStore.get(id: email.id)
        XCTAssertNotNil(fetched)
        XCTAssertEqual(fetched?.messageID, email.messageID)
        XCTAssertEqual(fetched?.subject, email.subject)

        // The receipt chain is the audit
        // trail; it should have at least the
        // upsert receipt.
        let receipts = try await ctx.emailStore.receipts(forEmail: email.id)
        XCTAssertFalse(receipts.isEmpty, "expected at least one receipt")
        let types = Set(receipts.map { $0.receiptType })
        XCTAssertTrue(types.contains(EmailReceiptType.upsert.rawValue))
    }

    /// Mark-as-read produces a receipt.
    /// The receipt payload records the
    /// prior / next state so the audit
    /// trail captures the transition.
    func testMarkReadProducesReceipt() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let email = EmailMessage(
            messageID: "msg-r@example.com",
            from: EmailAddress(email: "alice@example.com"),
            subject: "Hello",
            receivedAt: date,
            isRead: false,
            folder: .inbox,
            createdAt: date,
            updatedAt: date
        )
        _ = try await ctx.emailStore.upsert(email)
        _ = try await ctx.emailStore.markRead(email.id, read: true)
        let receipts = try await ctx.emailStore.receipts(forEmail: email.id)
        let types = receipts.map { $0.receiptType }
        XCTAssertTrue(types.contains(EmailReceiptType.read.rawValue))
    }

    /// setFolder produces a folder-typed
    /// receipt. The payload's `nextFolder`
    /// field carries the destination.
    func testSetFolderProducesReceipt() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let email = EmailMessage(
            messageID: "msg-f@example.com",
            from: EmailAddress(email: "alice@example.com"),
            subject: "Archive me",
            receivedAt: date,
            folder: .inbox,
            createdAt: date,
            updatedAt: date
        )
        _ = try await ctx.emailStore.upsert(email)
        _ = try await ctx.emailStore.setFolder(email.id, folder: .archive)
        let receipts = try await ctx.emailStore.receipts(forEmail: email.id)
        let types = receipts.map { $0.receiptType }
        XCTAssertTrue(types.contains(EmailReceiptType.archived.rawValue))
    }

    /// The full receipt chain is the audit
    /// trail. After several mutations
    /// (upsert, mark read, set starred,
    /// archive), the chain has a receipt
    /// for each. This is the
    /// "the receipt chain shows the email's
    /// full history" test from the spec.
    func testReceiptChainShowsFullHistory() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let email = EmailMessage(
            messageID: "msg-history@example.com",
            from: EmailAddress(email: "alice@example.com"),
            subject: "History",
            receivedAt: date,
            isRead: false,
            isStarred: false,
            folder: .inbox,
            createdAt: date,
            updatedAt: date
        )
        _ = try await ctx.emailStore.upsert(email)
        _ = try await ctx.emailStore.markRead(email.id, read: true)
        _ = try await ctx.emailStore.setStarred(email.id, starred: true)
        _ = try await ctx.emailStore.setFolder(email.id, folder: .archive)

        let receipts = try await ctx.emailStore.receipts(forEmail: email.id)
        let types = receipts.map { $0.receiptType }
        // Every mutation in the history
        // produced a receipt. The chain
        // contains them all in the order
        // they were appended.
        XCTAssertTrue(types.contains(EmailReceiptType.upsert.rawValue))
        XCTAssertTrue(types.contains(EmailReceiptType.read.rawValue))
        XCTAssertTrue(types.contains(EmailReceiptType.starred.rawValue))
        XCTAssertTrue(types.contains(EmailReceiptType.archived.rawValue))
    }

    /// Linking an email to another graph
    /// entity creates an ``entity_link``
    /// row and appends a
    /// ``link_created`` receipt. The
    /// ``link`` method is the seam the
    /// detail view's "related" section
    /// uses to surface cross-surface links.
    func testLinkEmailToEntity() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let email = EmailMessage(
            messageID: "msg-link@example.com",
            from: EmailAddress(email: "alice@example.com"),
            subject: "Link me",
            receivedAt: date,
            folder: .inbox,
            createdAt: date,
            updatedAt: date
        )
        _ = try await ctx.emailStore.upsert(email)

        // Create a second entity to link to.
        let otherID = UUID()
        let link = try await ctx.emailStore.link(
            emailID: email.id,
            to: otherID,
            linkType: "mentioned_in"
        )
        XCTAssertEqual(link.sourceID, email.id)
        XCTAssertEqual(link.targetID, otherID)
        XCTAssertEqual(link.linkType, "mentioned_in")

        // The link receipt is in the chain.
        let receipts = try await ctx.emailStore.receipts(forEmail: email.id)
        let types = receipts.map { $0.receiptType }
        XCTAssertTrue(types.contains(EmailReceiptType.linkCreated.rawValue))
    }

    /// ``recordReply`` flips the
    /// ``isReplied`` flag on the original,
    /// persists the draft in ``.sent``,
    /// and appends a ``replied`` receipt
    /// to the original. This is the
    /// "send routes through the share
    /// sheet" test (we don't actually
    /// present the share sheet in the
    /// integration test, but we do verify
    /// the side effects the share-sheet
    /// path would produce).
    func testRecordReplyPersistsDraftAndReceipt() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let original = EmailMessage(
            messageID: "msg-orig@example.com",
            from: EmailAddress(email: "alice@example.com"),
            subject: "Original",
            receivedAt: date,
            folder: .inbox,
            threadID: "msg-orig@example.com",
            createdAt: date,
            updatedAt: date
        )
        _ = try await ctx.emailStore.upsert(original)

        // The reply draft.
        let draft = EmailMessage(
            id: UUID(),
            messageID: "reply-id",
            from: EmailAddress(email: "me@example.com"),
            to: [EmailAddress(email: "alice@example.com")],
            subject: "Re: Original",
            bodyPlain: "Got it, thanks",
            receivedAt: date,
            folder: .drafts,
            threadID: original.threadID,
            createdAt: date,
            updatedAt: date
        )
        let sent = try await ctx.emailStore.recordReply(
            to: original.id,
            draft: draft
        )
        XCTAssertNotNil(sent)
        XCTAssertEqual(sent?.folder, .sent)

        // The original's isReplied flipped.
        let updated = try await ctx.emailStore.get(id: original.id)
        XCTAssertEqual(updated?.isReplied, true)

        // The reply receipt is in the chain.
        let receipts = try await ctx.emailStore.receipts(forEmail: original.id)
        let types = receipts.map { $0.receiptType }
        XCTAssertTrue(types.contains(EmailReceiptType.replied.rawValue))
    }
}

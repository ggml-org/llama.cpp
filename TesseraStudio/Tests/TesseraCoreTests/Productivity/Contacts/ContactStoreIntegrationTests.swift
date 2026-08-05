import XCTest
import PostgresNIO
@testable import TesseraCore

/// End-to-end integration tests for ``ContactStore``
/// against a real Postgres database. The test is
/// env-gated on `TESSERA_DB_INTEGRATION=1` (matching the
/// `SchemaMigrationTests` pattern). When the env var is
/// not set, every test calls `XCTSkip(...)` so `swift
/// test` works in environments without a running DB.
final class ContactStoreIntegrationTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_PG_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_PG_PORT"] ?? "5432") ?? 5432
    private static let user = ProcessInfo.processInfo.environment["TESSERA_PG_USER"] ?? "tessera"
    private static let pass = ProcessInfo.processInfo.environment["TESSERA_PG_PASSWORD"] ?? "tessera"
    private static let db = ProcessInfo.processInfo.environment["TESSERA_PG_DB"] ?? "tessera"

    /// Locate the migration files at runtime so the test
    /// works whether `swift test` is run from the
    /// package root, the test target dir, or the repo
    /// root.
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

        let testDB = "tessera_contact_test_\(Int.random(in: 1000...99999))"
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
            // Skip if migrations are not found in the
            // expected paths. Common in CI when the
            // migration files are not yet vendored into
            // the test target.
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
        let contactStore = ContactStore(dataLayer: dataLayer)
        return TestContext(
            admin: admin, dataLayer: dataLayer, testDB: testDB, contactStore: contactStore
        )
    }

    // MARK: - Tests

    func testContactRoundTripEndToEnd() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let contact = Contact(
            subtype: .person,
            name: NameComponents(first: "Ada", last: "Lovelace"),
            emails: [LabeledEmail(label: .work, value: "ada@analyticalengine.org")],
            organization: "Analytical Engine Co."
        )
        let saved = try await ctx.contactStore.upsert(contact)
        XCTAssertEqual(saved.id, contact.id)

        let fetched = try await ctx.contactStore.get(id: contact.id)
        XCTAssertNotNil(fetched)
        XCTAssertEqual(fetched?.displayName, "Ada Lovelace")
        XCTAssertEqual(fetched?.emails.first?.value, "ada@analyticalengine.org")
        XCTAssertEqual(fetched?.organization, "Analytical Engine Co.")
    }

    func testContactReceiptIsAppended() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let contact = Contact(
            subtype: .person,
            name: NameComponents(first: "Receipt", last: "Test")
        )
        _ = try await ctx.contactStore.upsert(contact)
        let receipts = try await ctx.contactStore.receipts(forContact: contact.id)
        XCTAssertFalse(receipts.isEmpty, "Every upsert should produce a receipt")
        XCTAssertEqual(receipts.first?.receiptType, ContactReceiptType.upsert.rawValue)
    }

    func testContactSearchByName() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        // Insert several contacts; verify the prefix
        // search returns the right one.
        for (i, first) in ["Alan", "Ada", "Grace", "Linus"].enumerated() {
            let c = Contact(
                subtype: .person,
                name: NameComponents(first: first, last: "Last\(i)")
            )
            _ = try await ctx.contactStore.upsert(c)
        }
        let adaResults = try await ctx.contactStore.search(matching: "Ada")
        XCTAssertFalse(adaResults.isEmpty)
        XCTAssertTrue(adaResults.contains { $0.name.first == "Ada" })
    }

    func testContactEgressPolicyFailsClosed() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let contact = Contact(
            subtype: .person,
            name: NameComponents(first: "Egress", last: "Test")
        )
        _ = try await ctx.contactStore.upsert(contact)
        let data = Data("fake vcard".utf8)
        do {
            _ = try await ctx.contactStore.exportVCard(
                contact,
                preEncodedVCard: data,
                provenance: "training"  // Not on the allow-list
            )
            XCTFail("Expected egressDenied")
        } catch ContactStoreError.egressDenied {
            // expected
        }
    }

    func testContactEgressPolicyAllowsUserExport() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let contact = Contact(
            subtype: .person,
            name: NameComponents(first: "Egress", last: "Allowed")
        )
        _ = try await ctx.contactStore.upsert(contact)
        let data = Data("fake vcard".utf8)
        let exported = try await ctx.contactStore.exportVCard(
            contact,
            preEncodedVCard: data,
            provenance: "user_explicit_export"
        )
        XCTAssertEqual(exported, data)
        // Verify the export was receipt-logged.
        let receipts = try await ctx.contactStore.receipts(forContact: contact.id)
        XCTAssertTrue(receipts.contains { $0.receiptType == ContactReceiptType.contactExport.rawValue })
    }

    func testContactNameQueryFastFor10k() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        // Insert 10k contacts.
        for i in 0..<10_000 {
            let c = Contact(
                subtype: .person,
                name: NameComponents(first: "First\(i)", last: "Last\(i)"),
                emails: [LabeledEmail(label: .work, value: "user\(i)@example.com")]
            )
            _ = try await ctx.contactStore.upsert(c)
        }
        // Search for the last entry; should be O(log n)
        // with the 0003 index.
        let start = Date()
        let results = try await ctx.contactStore.search(matching: "First9999")
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertFalse(results.isEmpty)
        XCTAssertLessThan(elapsed, 1.0, "10k contact search took \(elapsed)s")
    }
}

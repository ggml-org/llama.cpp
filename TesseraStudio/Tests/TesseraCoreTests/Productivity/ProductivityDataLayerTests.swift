import XCTest
import PostgresNIO
import CryptoKit
@testable import TesseraCore

/// Integration tests for the productivity surface's data-layer
/// extensions: `appendReceiptToChain`, `receiptChain`,
/// `latestChainIndex`, `loadChatQueue`, `saveChatQueue`.
///
/// All tests are env-gated (`TESSERA_DB_INTEGRATION=1`) and skip
/// gracefully when Postgres is unreachable. The test fixture
/// creates a throwaway database, applies both `0001_init.sql` AND
/// `0002_productivity_receipts.sql`, and tears down on exit.
final class ProductivityDataLayerTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_PG_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_PG_PORT"] ?? "5432") ?? 5432
    private static let user = ProcessInfo.processInfo.environment["TESSERA_PG_USER"] ?? "tessera"
    private static let pass = ProcessInfo.processInfo.environment["TESSERA_PG_PASSWORD"] ?? "tessera"
    private static let db = ProcessInfo.processInfo.environment["TESSERA_PG_DB"] ?? "tessera"

    fileprivate static func locateMigrationFile(_ name: String) -> String? {
        let fm = FileManager.default
        let candidates = [
            "tools/tessera/db/migrations/\(name)",
            "../tools/tessera/db/migrations/\(name)",
            "../../tools/tessera/db/migrations/\(name)",
        ]
        for c in candidates {
            if fm.fileExists(atPath: c) { return c }
        }
        return nil
    }

    private func requireIntegration() throws {
        guard Self.envEnabled else {
            throw XCTSkip("TESSERA_DB_INTEGRATION not set; skipping DB test")
        }
    }

    private struct TestContext {
        let admin: TesseraDataStore
        let target: TesseraDataStore
        let testDB: String

        func tearDown() async {
            try? await admin.queryRaw(PostgresQuery(stringLiteral: "DROP DATABASE IF EXISTS \(testDB) WITH (FORCE)"))
            await target.close()
            await admin.close()
        }
    }

    private func makeStore(database: String) async throws -> TesseraDataStore {
        let store = TesseraDataStore(
            configuration: .init(
                host: Self.host,
                port: Self.port,
                username: Self.user,
                password: Self.pass,
                database: database,
                minimumConnections: 1,
                maximumConnections: 2
            )
        )
        try await store.connect()
        return store
    }

    private func makeTestContext() async throws -> TestContext {
        let admin = try await makeStore(database: Self.db)
        let testDB = "tessera_prod_test_\(Int.random(in: 1000...99999))"
        try await admin.queryRaw(PostgresQuery(stringLiteral: "CREATE DATABASE \(testDB)"))
        let target = try await makeStore(database: testDB)

        // Apply both 0001 and 0002 migrations.
        guard let m1Path = Self.locateMigrationFile("0001_init.sql") else {
            XCTFail("could not locate 0001_init.sql")
            throw NSError(domain: "test", code: 1)
        }
        guard let m2Path = Self.locateMigrationFile("0002_productivity_receipts.sql") else {
            XCTFail("could not locate 0002_productivity_receipts.sql")
            throw NSError(domain: "test", code: 1)
        }
        let m1SQL = try String(contentsOfFile: m1Path, encoding: .utf8)
        let m2SQL = try String(contentsOfFile: m2Path, encoding: .utf8)
        try await target.applyMigrations([
            (name: "0001_init.sql", sql: m1SQL),
            (name: "0002_productivity_receipts.sql", sql: m2SQL),
        ])
        return TestContext(admin: admin, target: target, testDB: testDB)
    }

    private func makeTestDocument(target: TesseraDataStore) async throws -> UUID {
        let entity = try await target.upsertEntity(GraphEntityUpsert(
            entityType: "document",
            subtype: "doc",
            label: "Test Document",
            body: ""
        ))
        return entity.id
    }

    // MARK: - Schema

    func test0002CreatesAllTables() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let expected = ["receipt_chain", "chat_queues"]
        let query: PostgresQuery = """
            SELECT tablename FROM pg_tables
             WHERE schemaname = 'public'
               AND tablename = ANY(\(expected))
            """
        let rows = try await ctx.target.queryRaw(query)
        var found = Set<String>()
        for try await row in rows {
            let ra = row.makeRandomAccess()
            let n: String = try ra[0].decode(String.self)
            found.insert(n)
        }
        XCTAssertEqual(found, Set(expected))
    }

    func test0002CreatesAllIndexes() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let expected = [
            "idx_receipt_chain_doc",
            "idx_chat_queues_updated_at",
        ]
        let query: PostgresQuery = """
            SELECT indexname FROM pg_indexes
             WHERE schemaname = 'public'
               AND indexname = ANY(\(expected))
            """
        let rows = try await ctx.target.queryRaw(query)
        var found = Set<String>()
        for try await row in rows {
            let ra = row.makeRandomAccess()
            let n: String = try ra[0].decode(String.self)
            found.insert(n)
        }
        XCTAssertEqual(found, Set(expected))
    }

    // MARK: - Receipt chain

    func testAppendReceiptToChain() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let documentID = try await makeTestDocument(target: ctx.target)

        let key = Curve25519.Signing.PrivateKey()
        let signature = try key.signature(for: Data("test".utf8))
        let inserted = try await ctx.target.appendReceiptToChain(
            documentID: documentID,
            receiptType: "test",
            payload: ["summary": .string("hello")],
            signature: signature
        )
        XCTAssertNotNil(inserted.id)
        let chain = try await ctx.target.receiptChain(documentID: documentID)
        XCTAssertEqual(chain.count, 1)
        XCTAssertEqual(chain[0].chainIndex, 0)
        XCTAssertEqual(chain[0].receipt.id, inserted.id)
    }

    func testChainIndexIsMonotonic() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let documentID = try await makeTestDocument(target: ctx.target)
        let key = Curve25519.Signing.PrivateKey()
        for i in 0..<5 {
            let sig = try key.signature(for: Data("r\(i)".utf8))
            _ = try await ctx.target.appendReceiptToChain(
                documentID: documentID,
                receiptType: "test",
                payload: ["i": .number(Double(i))],
                signature: sig
            )
        }
        let chain = try await ctx.target.receiptChain(documentID: documentID)
        XCTAssertEqual(chain.count, 5)
        XCTAssertEqual(chain.map { $0.chainIndex }, [0, 1, 2, 3, 4])
    }

    func testChainIsPerDocument() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let docA = try await makeTestDocument(target: ctx.target)
        let docB = try await makeTestDocument(target: ctx.target)
        let key = Curve25519.Signing.PrivateKey()
        _ = try await ctx.target.appendReceiptToChain(
            documentID: docA, receiptType: "t", payload: ["k": .string("v")]
        )
        _ = try await ctx.target.appendReceiptToChain(
            documentID: docB, receiptType: "t", payload: ["k": .string("v")]
        )
        _ = try await ctx.target.appendReceiptToChain(
            documentID: docA, receiptType: "t", payload: ["k": .string("v")]
        )
        let chainA = try await ctx.target.receiptChain(documentID: docA)
        let chainB = try await ctx.target.receiptChain(documentID: docB)
        XCTAssertEqual(chainA.count, 2)
        XCTAssertEqual(chainB.count, 1)
        // Chain indices start at 0 for each document.
        XCTAssertEqual(chainA.map { $0.chainIndex }, [0, 1])
        XCTAssertEqual(chainB.map { $0.chainIndex }, [0])
    }

    func testLatestChainIndexNilForFreshDocument() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let documentID = try await makeTestDocument(target: ctx.target)
        let latest = try await ctx.target.latestChainIndex(documentID: documentID)
        XCTAssertNil(latest)
    }

    func testLatestChainIndexAfterAppend() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let documentID = try await makeTestDocument(target: ctx.target)
        for _ in 0..<3 {
            _ = try await ctx.target.appendReceiptToChain(
                documentID: documentID, receiptType: "t", payload: [:]
            )
        }
        let latest = try await ctx.target.latestChainIndex(documentID: documentID)
        XCTAssertEqual(latest, 2)
    }

    func testReceiptChainRespectsLimit() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let documentID = try await makeTestDocument(target: ctx.target)
        for _ in 0..<10 {
            _ = try await ctx.target.appendReceiptToChain(
                documentID: documentID, receiptType: "t", payload: [:]
            )
        }
        let limited = try await ctx.target.receiptChain(documentID: documentID, limit: 3)
        XCTAssertEqual(limited.count, 3)
        XCTAssertEqual(limited.map { $0.chainIndex }, [0, 1, 2])
    }

    // MARK: - Chat queue

    func testLoadChatQueueDefaultEmpty() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let documentID = try await makeTestDocument(target: ctx.target)
        let json = try await ctx.target.loadChatQueue(documentID: documentID)
        XCTAssertEqual(json, "[]")
    }

    func testSaveAndLoadChatQueue() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let documentID = try await makeTestDocument(target: ctx.target)
        let itemsJSON = #"[{"id":"\(UUID().uuidString)","order":0,"message":"hi"}]"#
        try await ctx.target.saveChatQueue(documentID: documentID, itemsJSON: itemsJSON)
        let loaded = try await ctx.target.loadChatQueue(documentID: documentID)
        XCTAssertEqual(loaded, itemsJSON)
    }

    func testSaveChatQueueOverwrites() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }
        let documentID = try await makeTestDocument(target: ctx.target)
        let v1 = #"[{"id":"a","order":0,"message":"v1"}]"#
        let v2 = #"[{"id":"b","order":0,"message":"v2"}]"#
        try await ctx.target.saveChatQueue(documentID: documentID, itemsJSON: v1)
        try await ctx.target.saveChatQueue(documentID: documentID, itemsJSON: v2)
        let loaded = try await ctx.target.loadChatQueue(documentID: documentID)
        XCTAssertEqual(loaded, v2)
    }
}

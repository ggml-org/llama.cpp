import XCTest
import PostgresNIO
@testable import TesseraCore

/// Integration tests for the `hybrid_search` SQL function, called
/// from ``TesseraDataStore/hybridSearch(...)``.
///
/// These tests seed a deterministic 5-entity / 4-link fixture (the
/// same fixture that ships in `tools/tessera/db/seeds/seed.sql`),
/// then assert the ranking matches the expected order for a known
/// query.
///
/// All tests are env-gated (`TESSERA_DB_INTEGRATION=1`) and skip
/// gracefully when Postgres is unreachable.
final class HybridSearchTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_PG_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_PG_PORT"] ?? "5432") ?? 5432
    private static let user = ProcessInfo.processInfo.environment["TESSERA_PG_USER"] ?? "tessera"
    private static let pass = ProcessInfo.processInfo.environment["TESSERA_PG_PASSWORD"] ?? "tessera"
    private static let db = ProcessInfo.processInfo.environment["TESSERA_PG_DB"] ?? "tessera"

    // Anchor entity from the seed fixture.
    private static let anchorID = UUID(uuidString: "b0000000-0000-0000-0000-000000000001")!
    private static let projectID = UUID(uuidString: "b0000000-0000-0000-0000-000000000002")!
    private static let topicID = UUID(uuidString: "b0000000-0000-0000-0000-000000000003")!
    private static let personID = UUID(uuidString: "b0000000-0000-0000-0000-000000000004")!
    private static let documentID = UUID(uuidString: "b0000000-0000-0000-0000-000000000005")!

    private func requireIntegration() throws {
        guard Self.envEnabled else {
            throw XCTSkip("TESSERA_DB_INTEGRATION not set; skipping DB test")
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

    /// Locate the seed file.
    private static func locateSeedFile() -> String? {
        let fm = FileManager.default
        let candidates = [
            "tools/tessera/db/seeds/seed.sql",
            "../tools/tessera/db/seeds/seed.sql",
            "../../tools/tessera/db/seeds/seed.sql",
        ]
        for c in candidates {
            if fm.fileExists(atPath: c) { return c }
        }
        return nil
    }

    /// Locate the migration file.
    private static func locateMigrationFile() -> String? {
        let fm = FileManager.default
        let candidates = [
            "tools/tessera/db/migrations/0001_init.sql",
            "../tools/tessera/db/migrations/0001_init.sql",
            "../../tools/tessera/db/migrations/0001_init.sql",
        ]
        for c in candidates {
            if fm.fileExists(atPath: c) { return c }
        }
        return nil
    }

    /// Test context returned by ``makeStoreWithSeed()`` -- the test
    /// owns the lifecycle and is responsible for calling
    /// ``TestContext/tearDown()`` before the test returns.
    private struct TestContext {
        let admin: TesseraDataStore
        let store: TesseraDataStore
        let testDB: String

        func tearDown() async {
            // Drop the throwaway DB. With FORCE for active connections.
            try? await admin.queryRaw(PostgresQuery(stringLiteral: "DROP DATABASE IF EXISTS \(testDB) WITH (FORCE)"))
            await store.close()
            await admin.close()
        }
    }

    /// Apply the migration + seed to a fresh throwaway database and
    /// return a connected `TesseraDataStore` for that DB.
    private func makeStoreWithSeed() async throws -> TestContext {
        let admin = try await makeStore(database: Self.db)
        let testDB = "tessera_hybrid_test_\(Int.random(in: 10000...99999))"
        try await admin.queryRaw(PostgresQuery(stringLiteral: "CREATE DATABASE \(testDB)"))

        let store = try await makeStore(database: testDB)

        guard let migrationPath = Self.locateMigrationFile() else {
            await admin.close()
            await store.close()
            XCTFail("could not locate 0001_init.sql")
            throw NSError(domain: "test", code: 1)
        }
        let migrationSQL = try String(contentsOfFile: migrationPath, encoding: .utf8)
        try await store.applyMigrations([(name: "0001_init.sql", sql: migrationSQL)])

        guard let seedPath = Self.locateSeedFile() else {
            await admin.close()
            await store.close()
            XCTFail("could not locate seed.sql")
            throw NSError(domain: "test", code: 1)
        }
        let seedSQL = try String(contentsOfFile: seedPath, encoding: .utf8)
        try await store.applyMigrations([(name: "seed.sql", sql: seedSQL)])

        return TestContext(admin: admin, store: store, testDB: testDB)
    }

    /// From the anchor, all 4 reachable entities should appear in
    /// the result set with depth 3.
    func testHybridSearchReturnsAllReachableEntities() async throws {
        try requireIntegration()
        let ctx = try await makeStoreWithSeed()
        let results: [HybridSearchResult]
        do {
            results = try await ctx.store.hybridSearch(
                anchor: Self.anchorID,
                queryText: "hybrid retrieval",
                queryEmbedding: nil,
                maxDepth: 3
            )
        } catch {
            XCTFail("hybridSearch threw: \(String(reflecting: error))")
            await ctx.tearDown()
            return
        }
        await ctx.tearDown()

        let ids = Set(results.map { $0.entityID })
        XCTAssertTrue(ids.contains(Self.projectID), "project missing from results")
        XCTAssertTrue(ids.contains(Self.topicID), "topic missing from results")
        XCTAssertTrue(ids.contains(Self.personID), "person missing from results")
        XCTAssertTrue(ids.contains(Self.documentID), "document missing from results")
    }

    /// The RRF ranking should put the project (depth 1) above the
    /// topic (depth 2), and the topic above the document / person
    /// (depth 3).
    func testHybridSearchRankingByDepth() async throws {
        try requireIntegration()
        let ctx = try await makeStoreWithSeed()
        let results = try await ctx.store.hybridSearch(
            anchor: Self.anchorID,
            queryText: nil,
            queryEmbedding: nil,
            maxDepth: 3
        )
        await ctx.tearDown()

        XCTAssertGreaterThanOrEqual(results.count, 3)
        let projectIdx = results.firstIndex { $0.entityID == Self.projectID }
        let topicIdx = results.firstIndex { $0.entityID == Self.topicID }
        let docIdx = results.firstIndex { $0.entityID == Self.documentID }
        XCTAssertNotNil(projectIdx)
        XCTAssertNotNil(topicIdx)
        XCTAssertNotNil(docIdx)
        XCTAssertLessThan(projectIdx!, topicIdx!, "project should rank above topic")
        XCTAssertLessThan(topicIdx!, docIdx!, "topic should rank above document")
    }

    /// No vector query means vectorScore is 0.
    func testHybridSearchVectorScore() async throws {
        try requireIntegration()
        let ctx = try await makeStoreWithSeed()
        let results = try await ctx.store.hybridSearch(
            anchor: Self.anchorID,
            queryText: nil,
            queryEmbedding: nil,
            maxDepth: 3
        )
        await ctx.tearDown()

        guard let project = results.first(where: { $0.entityID == Self.projectID }) else {
            XCTFail("project missing from results")
            return
        }
        XCTAssertEqual(project.vectorScore, 0)
    }

    /// The depth limit truncates the reachable set. With maxDepth=1,
    /// only the project (1 hop from the anchor) should appear.
    func testHybridSearchDepthLimit() async throws {
        try requireIntegration()
        let ctx = try await makeStoreWithSeed()
        let results = try await ctx.store.hybridSearch(
            anchor: Self.anchorID,
            queryText: "hybrid retrieval",
            queryEmbedding: nil,
            maxDepth: 1
        )
        await ctx.tearDown()

        let ids = Set(results.map { $0.entityID })
        XCTAssertEqual(ids, [Self.projectID], "depth=1 should only return the direct edge")
    }
}

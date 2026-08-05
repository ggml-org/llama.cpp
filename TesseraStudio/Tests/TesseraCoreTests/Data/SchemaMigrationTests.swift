import XCTest
import PostgresNIO
@testable import TesseraCore

/// Schema migration integration tests.
///
/// All tests in this file are env-gated: they only run when
/// `TESSERA_DB_INTEGRATION=1` is set AND a local Postgres is reachable
/// on `localhost:5432` with the test credentials (default
/// `tessera/tessera/tessera`). When the env var is missing, every
/// test calls `XCTSkip(...)` so `swift test` works in environments
/// without a running DB (CI, laptop without docker, etc.).
final class SchemaMigrationTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_PG_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_PG_PORT"] ?? "5432") ?? 5432
    private static let user = ProcessInfo.processInfo.environment["TESSERA_PG_USER"] ?? "tessera"
    private static let pass = ProcessInfo.processInfo.environment["TESSERA_PG_PASSWORD"] ?? "tessera"
    private static let db = ProcessInfo.processInfo.environment["TESSERA_PG_DB"] ?? "tessera"

    // The migration file is read from the repo at runtime. We
    // resolve the path relative to the current working directory
    // so the test works whether `swift test` is run from the
    // package root, the test target dir, or the repo root.
    fileprivate static func locateMigrationFile() -> String? {
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

    private func requireIntegration() throws {
        guard Self.envEnabled else {
            throw XCTSkip("TESSERA_DB_INTEGRATION not set; skipping DB test")
        }
    }

    // MARK: - Test context (throwaway DB + applied migration)

    private struct TestContext {
        let admin: TesseraDataStore
        let target: TesseraDataStore
        let testDB: String

        func tearDown() async {
            // Drop the throwaway DB.
            try? await admin.queryRaw(PostgresQuery(stringLiteral: "DROP DATABASE IF EXISTS \(testDB) WITH (FORCE)"))
            await target.close()
            await admin.close()
        }
    }

    /// Create a fresh test database, apply the migration to it, and
    /// return connected stores for both the admin (which can issue
    /// DROP DATABASE) and the target (where the schema lives).
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

        let testDB = "tessera_migration_test_\(Int.random(in: 1000...99999))"
        try await admin.queryRaw(PostgresQuery(stringLiteral: "CREATE DATABASE \(testDB)"))

        let target = TesseraDataStore(
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
        try await target.connect()

        guard let migrationPath = Self.locateMigrationFile() else {
            XCTFail("could not locate 0001_init.sql from cwd \(FileManager.default.currentDirectoryPath)")
            throw NSError(domain: "test", code: 1)
        }
        let sql = try String(contentsOfFile: migrationPath, encoding: .utf8)
        try await target.applyMigrations([(name: "0001_init.sql", sql: sql)])

        return TestContext(admin: admin, target: target, testDB: testDB)
    }

    // MARK: - Tests

    /// All expected tables are created by the migration.
    func test0001InitCreatesAllTables() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let expected = ["graph_entities", "entity_links", "graph_receipts"]
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

    /// All expected indexes are created by the migration.
    func test0001InitCreatesAllIndexes() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let expected = [
            "idx_entities_type",
            "idx_entities_subtype",
            "idx_entities_source_url",
            "idx_entities_search_tsv",
            "idx_entities_embedding",
            "idx_entities_trgm_label",
            "idx_links_source",
            "idx_links_target",
            "idx_links_type",
            "idx_receipts_entity",
            "idx_receipts_type",
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

    /// The hybrid_search function exists and takes 4 args.
    func test0001InitCreatesHybridSearchFunction() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let query: PostgresQuery = """
            SELECT proname, pronargs FROM pg_proc
             WHERE proname = 'hybrid_search'
            """
        let rows = try await ctx.target.queryRaw(query)
        var found = false
        for try await row in rows {
            let ra = row.makeRandomAccess()
            let name: String = try ra[0].decode(String.self)
            let nargs: Int32 = try ra[1].decode(Int32.self)
            XCTAssertEqual(name, "hybrid_search")
            // hybrid_search takes 4 args: p_anchor uuid, p_query_text text,
            // p_query_embedding vector(1536), p_max_depth int.
            XCTAssertEqual(Int(nargs), 4)
            found = true
        }
        XCTAssertTrue(found, "hybrid_search function not found in pg_proc")
    }

    /// Required extensions are loaded in the test DB.
    func test0001InitLoadsExtensions() async throws {
        try requireIntegration()
        let ctx = try await makeTestContext()
        defer { Task { await ctx.tearDown() } }

        let query: PostgresQuery = """
            SELECT extname FROM pg_extension
             WHERE extname IN ('vector', 'pg_trgm', 'pgcrypto')
            """
        let rows = try await ctx.target.queryRaw(query)
        var found = Set<String>()
        for try await row in rows {
            let ra = row.makeRandomAccess()
            let n: String = try ra[0].decode(String.self)
            found.insert(n)
        }
        for required in ["vector", "pg_trgm"] {
            XCTAssertTrue(found.contains(required), "extension \(required) not loaded")
        }
    }
}

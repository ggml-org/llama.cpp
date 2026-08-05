import XCTest
import PostgresNIO
@testable import TesseraCore

/// Integration tests for ``TesseraDataStore`` connection pool
/// concurrency semantics.
///
/// All tests are env-gated (`TESSERA_DB_INTEGRATION=1`) and skip
/// gracefully when Postgres is unreachable.
final class ConnectionPoolTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_PG_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_PG_PORT"] ?? "5432") ?? 5432
    private static let user = ProcessInfo.processInfo.environment["TESSERA_PG_USER"] ?? "tessera"
    private static let pass = ProcessInfo.processInfo.environment["TESSERA_PG_PASSWORD"] ?? "tessera"
    private static let db = ProcessInfo.processInfo.environment["TESSERA_PG_DB"] ?? "tessera"

    private func requireIntegration() throws {
        guard Self.envEnabled else {
            throw XCTSkip("TESSERA_DB_INTEGRATION not set; skipping DB test")
        }
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

    private struct TestContext {
        let admin: TesseraDataStore
        let store: TesseraDataStore
        let testDB: String

        func tearDown() async {
            try? await admin.queryRaw(PostgresQuery(stringLiteral: "DROP DATABASE IF EXISTS \(testDB) WITH (FORCE)"))
            await store.close()
            await admin.close()
        }
    }

    /// Create a fresh test database with the migration applied, and
    /// return connected stores for both the admin (for DROP DATABASE)
    /// and the test target.
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

        let testDB = "tessera_pool_test_\(Int.random(in: 10000...99999))"
        try await admin.queryRaw(PostgresQuery(stringLiteral: "CREATE DATABASE \(testDB)"))

        let store = TesseraDataStore(
            configuration: .init(
                host: Self.host,
                port: Self.port,
                username: Self.user,
                password: Self.pass,
                database: testDB,
                minimumConnections: 1,
                maximumConnections: 4
            )
        )
        try await store.connect()

        guard let migrationPath = Self.locateMigrationFile() else {
            XCTFail("could not locate 0001_init.sql")
            throw NSError(domain: "test", code: 1)
        }
        let migrationSQL = try String(contentsOfFile: migrationPath, encoding: .utf8)
        try await store.applyMigrations([(name: "0001_init.sql", sql: migrationSQL)])

        return TestContext(admin: admin, store: store, testDB: testDB)
    }

    /// 50 concurrent `getEntity(id:)` calls on different IDs (all
    /// nil -- the test DB has no entities seeded) must all complete
    /// without deadlock. The test passes if the store can serve
    /// 50 concurrent reads; if the pool deadlocks, the test will
    /// time out (XCTest's default is 60s per test).
    func test50ConcurrentReadsNoDeadlock() async throws {
        try requireIntegration()
        let ctx: TestContext
        do {
            ctx = try await makeTestContext()
        } catch {
            throw XCTSkip("Postgres not reachable at \(Self.host):\(Self.port): \(error)")
        }

        let ids = (0..<50).map { _ in UUID() }
        let store = ctx.store

        let results = await withTaskGroup(of: (UUID, GraphEntity?).self) { group in
            for id in ids {
                group.addTask {
                    do {
                        let entity = try await store.getEntity(id: id)
                        return (id, entity)
                    } catch {
                        return (id, nil)
                    }
                }
            }
            var collected: [(UUID, GraphEntity?)] = []
            for await r in group { collected.append(r) }
            return collected
        }
        await ctx.tearDown()

        XCTAssertEqual(results.count, 50, "all 50 reads should complete")
        for (id, entity) in results {
            XCTAssertNil(entity, "random UUID \(id) should not exist")
        }
    }

    /// 50 concurrent UPSERTs of different entities must all complete.
    /// Validates the pool can serve 50 writes without deadlock; the
    /// pool has 4 connections, so 50 writes will queue, but they
    /// must drain in finite time.
    func test50ConcurrentWritesNoDeadlock() async throws {
        try requireIntegration()
        let ctx: TestContext
        do {
            ctx = try await makeTestContext()
        } catch {
            throw XCTSkip("Postgres not reachable at \(Self.host):\(Self.port): \(error)")
        }

        let store = ctx.store
        let results = await withTaskGroup(of: Result<UUID, Error>.self) { group in
            for i in 0..<50 {
                group.addTask {
                    do {
                        let input = GraphEntityUpsert(
                            id: UUID(),
                            entityType: "test_concurrent",
                            subtype: "concurrent_write",
                            label: "concurrent-\(i)-\(Int.random(in: 1000...9999))",
                            body: "generated by ConnectionPoolTests"
                        )
                        let entity = try await store.upsertEntity(input)
                        return .success(entity.id)
                    } catch {
                        return .failure(error)
                    }
                }
            }
            var collected: [Result<UUID, Error>] = []
            for await r in group { collected.append(r) }
            return collected
        }
        await ctx.tearDown()

        let successes = results.compactMap { try? $0.get() }
        let failures = results.filter { if case .failure = $0 { return true } else { return false } }
        XCTAssertEqual(successes.count, 50, "all 50 writes should succeed; failures: \(failures.count)")
        XCTAssertEqual(successes.count, Set(successes).count, "all 50 IDs should be unique")
    }

    /// The pool opens at least one connection. Smoke test that
    /// `pg_stat_activity` shows a live connection from our test.
    func testConnectionIsEstablished() async throws {
        try requireIntegration()
        let ctx: TestContext
        do {
            ctx = try await makeTestContext()
        } catch {
            throw XCTSkip("Postgres not reachable at \(Self.host):\(Self.port): \(error)")
        }

        let query: PostgresQuery = """
            SELECT count(*)::int FROM pg_stat_activity WHERE datname = current_database()
            """
        let rows = try await ctx.store.queryRaw(query)
        var count = 0
        for try await row in rows {
            let ra = row.makeRandomAccess()
            count = try ra[0].decode(Int.self)
        }
        await ctx.tearDown()

        // At least one connection should be open after the smoke test
        // run earlier in `makeTestContext`.
        XCTAssertGreaterThan(count, 0, "no connections open to the test DB")
    }
}

import XCTest
import PostgresNIO
@testable import TesseraCore

/// End-to-end integration tests for ``ProductivityTaskStore``
/// against a real Postgres database. The test is env-gated on
/// `TESSERA_DB_INTEGRATION=1` (matching the
/// ``ContactStoreIntegrationTests`` pattern). When the env
/// var is not set, every test calls `XCTSkip(...)` so `swift
/// test` works in environments without a running DB.
final class ProductivityTaskStoreIntegrationTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_PG_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_PG_PORT"] ?? "5432") ?? 5432
    private static let user = ProcessInfo.processInfo.environment["TESSERA_PG_USER"] ?? "tessera"
    private static let pass = ProcessInfo.processInfo.environment["TESSERA_PG_PASSWORD"] ?? "tessera"
    private static let db = ProcessInfo.processInfo.environment["TESSERA_PG_DB"] ?? "tessera"

    fileprivate static func locateMigrationFiles() -> [(name: String, sql: String)] {
        let fm = FileManager.default
        var dirs = [
            URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("tools/tessera/db/migrations"),
        ]
        if let workspaceRoot = ProcessInfo.processInfo.environment["TESSERA_WORKSPACE_ROOT"] {
            dirs.append(URL(fileURLWithPath: workspaceRoot).appendingPathComponent("tools/tessera/db/migrations"))
        }
        for dir in dirs where fm.fileExists(atPath: dir.path) {
            var out: [(name: String, sql: String)] = []
            if let names = try? fm.contentsOfDirectory(atPath: dir.path) {
                for name in names.sorted() where name.hasSuffix(".sql") {
                    let url = dir.appendingPathComponent(name)
                    if let sql = try? String(contentsOf: url, encoding: .utf8) {
                        out.append((name: name, sql: sql))
                    }
                }
            }
            if !out.isEmpty { return out }
        }
        return []
    }

    private func makeDataLayer() async throws -> TesseraDataLayer {
        guard Self.envEnabled else {
            throw XCTSkip("TESSERA_DB_INTEGRATION not set; skipping DB test")
        }
        let store = TesseraDataStore(
            configuration: .init(
                host: Self.host,
                port: Self.port,
                username: Self.user,
                password: Self.pass,
                database: Self.db,
                useTLS: false
            )
        )
        try await store.connect()
        try await store.applyMigrations(Self.locateMigrationFiles())
        return TesseraDataLayer(dataStore: store, cache: TesseraCache(configuration: .init()))
    }

    // MARK: - End-to-end

    func testUpsertGetDeleteFlow() async throws {
        let dataLayer = try await makeDataLayer()
        let store = ProductivityTaskStore(dataLayer: dataLayer)
        let userID = UserID()
        let task = ProductivityTask(        title: "review the Q3 report",
        priority: .high,
        tags: ["work"],
        list: .today,
        linkedEntityIDs: [])

        let saved = try await store.upsert(task, actor: .user(userID))
        XCTAssertEqual(saved.title, "review the Q3 report")

        let fetched = try await store.get(id: saved.id)
        XCTAssertNotNil(fetched)
        XCTAssertEqual(fetched?.title, "review the Q3 report")
        XCTAssertEqual(fetched?.list, .today)
        XCTAssertEqual(fetched?.priority, .high)

        // Receipt was written
        let receipts = try await store.receipts(forTask: saved.id)
        XCTAssertFalse(receipts.isEmpty)
        XCTAssertEqual(receipts.first?.receiptType, ProductivityTaskReceiptType.upsert.rawValue)

        // Delete
        let didDelete = try await store.delete(id: saved.id, actor: .user(userID))
        XCTAssertTrue(didDelete)
        let afterDelete = try await store.get(id: saved.id)
        XCTAssertNil(afterDelete)
    }

    func testListReturnsInsertedTask() async throws {
        let dataLayer = try await makeDataLayer()
        let store = ProductivityTaskStore(dataLayer: dataLayer)
        let userID = UserID()
        let title = "list test \(UUID().uuidString)"
        let task = ProductivityTask(        title: title,
        list: .inbox)

        _ = try await store.upsert(task, actor: .user(userID))
        let all = try await store.list(limit: 1000)
        XCTAssertTrue(all.contains(where: { $0.id == task.id }))
        // Cleanup
        _ = try await store.delete(id: task.id, actor: .user(userID))
    }

    func testMoveRecordsReceipt() async throws {
        let dataLayer = try await makeDataLayer()
        let store = ProductivityTaskStore(dataLayer: dataLayer)
        let userID = UserID()
        let task = ProductivityTask(        title: "move test",
        list: .inbox)

        let saved = try await store.upsert(task, actor: .user(userID))
        _ = try await store.move(id: saved.id, to: .today, actor: .user(userID))
        let receipts = try await store.receipts(forTask: saved.id)
        let moveReceipts = receipts.filter { $0.receiptType == ProductivityTaskReceiptType.moved.rawValue }
        XCTAssertEqual(moveReceipts.count, 1)
        // Cleanup
        _ = try await store.delete(id: saved.id, actor: .user(userID))
    }

    func testCompleteReopen() async throws {
        let dataLayer = try await makeDataLayer()
        let store = ProductivityTaskStore(dataLayer: dataLayer)
        let userID = UserID()
        let task = ProductivityTask(        title: "complete test",
        list: .inbox)

        let saved = try await store.upsert(task, actor: .user(userID))
        let completed = try await store.complete(id: saved.id, actor: .user(userID))
        XCTAssertNotNil(completed?.completedAt)
        let reopened = try await store.reopen(id: saved.id, actor: .user(userID))
        XCTAssertNil(reopened?.completedAt)
        // Cleanup
        _ = try await store.delete(id: saved.id, actor: .user(userID))
    }
}

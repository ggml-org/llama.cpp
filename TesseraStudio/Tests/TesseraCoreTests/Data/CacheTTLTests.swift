import XCTest
@testable import TesseraCore

/// Integration tests for ``TesseraCache`` TTL semantics.
///
/// All tests are env-gated (`TESSERA_DB_INTEGRATION=1`) and skip
/// gracefully when Valkey / Redis is unreachable.
final class CacheTTLTests: XCTestCase {

    private static let envEnabled: Bool = {
        ProcessInfo.processInfo.environment["TESSERA_DB_INTEGRATION"] == "1"
    }()

    private static let host = ProcessInfo.processInfo.environment["TESSERA_VALKEY_HOST"] ?? "localhost"
    private static let port = Int(ProcessInfo.processInfo.environment["TESSERA_VALKEY_PORT"] ?? "6379") ?? 6379
    private static let password = ProcessInfo.processInfo.environment["TESSERA_VALKEY_PASSWORD"]
    private static let dbNumber = Int(ProcessInfo.processInfo.environment["TESSERA_VALKEY_DB"] ?? "0") ?? 0

    private func requireIntegration() throws {
        guard Self.envEnabled else {
            throw XCTSkip("TESSERA_DB_INTEGRATION not set; skipping cache test")
        }
    }

    private func makeCache() -> TesseraCache {
        TesseraCache(
            configuration: .init(
                host: Self.host,
                port: Self.port,
                password: Self.password,
                databaseNumber: Self.dbNumber,
                namespace: "test-ttl-\(Int.random(in: 1000...9999))"
            )
        )
    }

    /// SET with TTL=1s, then assert the key is gone after 1.5s.
    /// This is the load-bearing TTL behaviour the rest of the cache
    /// contract relies on.
    func testSetTTLExpire() async throws {
        try requireIntegration()
        let cache = makeCache()
        do {
            try await cache.connect()
        } catch {
            throw XCTSkip("Valkey not reachable at \(Self.host):\(Self.port): \(error)")
        }
        defer { Task { await cache.close() } }

        let key = cache.key("ttl-test")
        try await cache.set(key, value: "hello", ttlSeconds: 1)

        // Confirm the value is there.
        let first = try await cache.get(key)
        XCTAssertEqual(first, "hello", "value should be present right after set")

        // Sleep past the TTL.
        try await Task.sleep(nanoseconds: 1_500_000_000)

        // The key should now be absent.
        let second = try await cache.get(key)
        XCTAssertNil(second, "value should be gone after TTL")
    }

    /// SET without TTL persists across a short sleep.
    func testSetWithoutTTL() async throws {
        try requireIntegration()
        let cache = makeCache()
        do {
            try await cache.connect()
        } catch {
            throw XCTSkip("Valkey not reachable at \(Self.host):\(Self.port): \(error)")
        }
        defer { Task { await cache.close() } }

        let key = cache.key("no-ttl-test")
        try await cache.set(key, value: "persistent", ttlSeconds: 0)

        try await Task.sleep(nanoseconds: 500_000_000)

        let value = try await cache.get(key)
        XCTAssertEqual(value, "persistent", "value should persist without TTL")

        // Clean up.
        _ = try? await cache.del([key])
    }

    /// DEL removes the key.
    func testDel() async throws {
        try requireIntegration()
        let cache = makeCache()
        do {
            try await cache.connect()
        } catch {
            throw XCTSkip("Valkey not reachable at \(Self.host):\(Self.port): \(error)")
        }
        defer { Task { await cache.close() } }

        let key = cache.key("del-test")
        try await cache.set(key, value: "to-be-deleted", ttlSeconds: 0)
        let removed = try await cache.del([key])
        XCTAssertEqual(removed, 1)
        let value = try await cache.get(key)
        XCTAssertNil(value, "value should be gone after del")
    }

    /// INCR + EXPIRE compose: a counter that auto-resets.
    func testIncrAndExpire() async throws {
        try requireIntegration()
        let cache = makeCache()
        do {
            try await cache.connect()
        } catch {
            throw XCTSkip("Valkey not reachable at \(Self.host):\(Self.port): \(error)")
        }
        defer { Task { await cache.close() } }

        let key = cache.key("counter")
        _ = try? await cache.del([key])  // reset

        let n1 = try await cache.incr(key)
        XCTAssertEqual(n1, 1)
        let n2 = try await cache.incr(key)
        XCTAssertEqual(n2, 2)

        // Set a short expiry; reading TTL should reflect it.
        let expired = try await cache.expire(key, after: 1)
        XCTAssertTrue(expired)

        let ttl = try await cache.ttl(key)
        XCTAssertGreaterThanOrEqual(ttl, 0)
        XCTAssertLessThanOrEqual(ttl, 1)

        _ = try? await cache.del([key])
    }

    /// ZADD + ZRANGEBYSCORE round-trip. The decay window is the
    /// production use case; this is the minimum test.
    func testZaddZRangeByScore() async throws {
        try requireIntegration()
        let cache = makeCache()
        do {
            try await cache.connect()
        } catch {
            throw XCTSkip("Valkey not reachable at \(Self.host):\(Self.port): \(error)")
        }
        defer { Task { await cache.close() } }

        let key = cache.key("decay")
        _ = try? await cache.zremrangebyscore(key, min: -.infinity, max: .infinity)  // reset

        let added = try await cache.zadd(key, members: [
            (member: "alpha", score: 1.0),
            (member: "beta", score: 2.0),
            (member: "gamma", score: 3.0),
        ])
        XCTAssertEqual(added, 3, "all three new members should be added")

        let range = try await cache.zrangebyscore(key, min: 1.5, max: 2.5, withScores: true)
        XCTAssertEqual(range.map { $0.member }, ["beta"])
        XCTAssertEqual(range.map { $0.score }, [2.0])
    }

    /// SET-NX (setIfAbsent) returns true the first time, false the second.
    func testSetIfAbsent() async throws {
        try requireIntegration()
        let cache = makeCache()
        do {
            try await cache.connect()
        } catch {
            throw XCTSkip("Valkey not reachable at \(Self.host):\(Self.port): \(error)")
        }
        defer { Task { await cache.close() } }

        let key = cache.key("setnx-test")
        _ = try? await cache.del([key])  // reset

        let first = try await cache.setIfAbsent(key, value: "first", ttlSeconds: 0)
        XCTAssertTrue(first, "first SET NX should succeed")

        let second = try await cache.setIfAbsent(key, value: "second", ttlSeconds: 0)
        XCTAssertFalse(second, "second SET NX should fail")

        let value = try await cache.get(key)
        XCTAssertEqual(value, "first", "the first value should win")

        _ = try? await cache.del([key])
    }
}

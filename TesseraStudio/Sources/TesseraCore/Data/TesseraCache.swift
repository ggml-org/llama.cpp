import Foundation
import Logging
@preconcurrency import NIOCore
import NIOPosix
import NIOSSL
@preconcurrency import RediStack

// MARK: - Domain types (no Redis types leak)

/// A single entry returned by ``TesseraCache/pop(timeout:)``. `nil`
/// means the timeout expired before an element was available. The
/// shape mirrors a Redis list BRPOP: the source key + the popped value.
public struct CachePopResult: Sendable, Equatable {
    public let key: String
    public let value: String
    public init(key: String, value: String) {
        self.key = key
        self.value = value
    }
}

/// One (member, score) pair returned by a sorted-set range query.
public struct CacheScoredMember: Sendable, Equatable {
    public let member: String
    public let score: Double
    public init(member: String, score: Double) {
        self.member = member
        self.score = score
    }
}

/// Errors raised by ``TesseraCache``. The underlying error is preserved
/// on `connectionFailed` / `commandFailed` for diagnostics.
public enum TesseraCacheError: Error, Sendable, Equatable {
    case closed
    case connectionFailed
    case commandFailed(String)
    case typeMismatch(expected: String, got: String)

    public static func == (lhs: TesseraCacheError, rhs: TesseraCacheError) -> Bool {
        switch (lhs, rhs) {
        case (.closed, .closed): return true
        case (.connectionFailed, .connectionFailed): return true
        case (.commandFailed(let a), .commandFailed(let b)): return a == b
        case (.typeMismatch(let a, let b), .typeMismatch(let c, let d)): return a == c && b == d
        default: return false
        }
    }
}

// MARK: - Cache

/// Valkey / Redis cache actor for ephemeral state: agent scratchpad,
/// capture cache, decay windows, session state, idempotency keys.
///
/// Backed by `RediStack`'s ``RedisConnectionPool`` (NIO-native; the
/// community-maintained successor to the original swift-server client).
/// All key names are automatically prefixed with `tessera:<namespace>:` so
/// the cache can be shared with other apps on the same Valkey without
/// collision (the namespace is per-cache-instance).
///
/// **Hexagonal boundary**: this is the ONLY file in TesseraCore that
/// imports `RediStack`. The rest of the app depends on ``TesseraDataLayer``
/// which exposes a domain-shaped API. The productivity surface never
/// sees a `RESPValue` or a `RedisConnection`.
public actor TesseraCache {

    public struct Configuration: Sendable {
        public var host: String
        public var port: Int
        public var password: String?
        public var databaseNumber: Int
        public var poolSize: Int
        public var namespace: String

        public init(
            host: String = "localhost",
            port: Int = 6379,
            password: String? = nil,
            databaseNumber: Int = 0,
            poolSize: Int = 4,
            namespace: String = "default"
        ) {
            self.host = host
            self.port = port
            self.password = password
            self.databaseNumber = databaseNumber
            self.poolSize = poolSize
            self.namespace = namespace
        }

        /// Build from a `redis://[:pw@]host:port/db` URL. Returns nil
        /// for malformed URLs.
        public static func from(connectionString: String) -> Configuration? {
            guard let url = URL(string: connectionString) else { return nil }
            let host = url.host ?? "localhost"
            let port = url.port ?? 6379
            let password = url.password
            let database = Int(url.path.split(separator: "/").last.map(String.init) ?? "0") ?? 0
            return Configuration(
                host: host,
                port: port,
                password: password,
                databaseNumber: database
            )
        }
    }

    private var logger: Logger
    nonisolated let configuration: Configuration
    private var pool: RedisConnectionPool?
    private var isClosed: Bool = false
    private let eventLoop: EventLoop
    private let eventLoopGroup: EventLoopGroup

    public init(
        configuration: Configuration = .init(),
        eventLoopGroup: EventLoopGroup = TesseraCache.makeDefaultGroup(),
        logger: Logger = .init(label: "tessera.data.valkey")
    ) {
        self.configuration = configuration
        self.eventLoopGroup = eventLoopGroup
        self.eventLoop = eventLoopGroup.next()
        self.logger = logger
    }

    deinit {
        // Best-effort: pool's deinit handles close.
        pool = nil
    }

    /// Replace the logger. Useful for tests.
    public func setLogger(_ logger: Logger) {
        self.logger = logger
    }

    /// Build the default event loop group. The pool reuses this
    /// `EventLoopGroup` for its NIO channels.
    public static func makeDefaultGroup() -> EventLoopGroup {
        MultiThreadedEventLoopGroup(numberOfThreads: 1)
    }

    // MARK: - Lifecycle

    /// Open the connection pool. Idempotent.
    public func connect() async throws {
        guard pool == nil else { return }
        guard !isClosed else { throw TesseraCacheError.closed }

        let address: SocketAddress
        do {
            address = try SocketAddress.makeAddressResolvingHost(
                configuration.host,
                port: configuration.port
            )
        } catch {
            throw TesseraCacheError.connectionFailed
        }

        let poolConfig = RedisConnectionPool.Configuration(
            initialServerConnectionAddresses: [address],
            maximumConnectionCount: .maximumActiveConnections(configuration.poolSize),
            connectionFactoryConfiguration: .init(
                connectionInitialDatabase: configuration.databaseNumber,
                connectionPassword: configuration.password
            )
        )

        let pool = RedisConnectionPool(
            configuration: poolConfig,
            boundEventLoop: eventLoop
        )
        pool.activate(logger: logger)
        self.pool = pool
    }

    /// Close the pool. Idempotent.
    public func close() async {
        guard let pool else { return }
        let promise = eventLoop.makePromise(of: Void.self)
        pool.close(promise: promise, logger: logger)
        do {
            try await promise.futureResult.get()
        } catch {
            logger.warning("TesseraCache close failed: \(error)")
        }
        self.pool = nil
        self.isClosed = true
    }

    // MARK: - Key naming

    /// Build a namespaced key from a namespace + parts array.
    /// Synchronous so callers can use it outside the actor (e.g. in
    /// test setup that doesn't need the cache to be connected).
    public static func namespacedKey(namespace: String, parts: [String]) -> String {
        let joined = parts.joined(separator: ":")
        return "tessera:\(namespace):\(joined)"
    }

    /// Build the namespaced key. All public methods funnel through
    /// this so callers can't accidentally bypass the prefix.
    public nonisolated func key(_ parts: String...) -> String {
        Self.namespacedKey(namespace: configuration.namespace, parts: parts)
    }

    // MARK: - Strings

    /// GET a key. Returns nil if the key is absent.
    public func get(_ key: String) async throws -> String? {
        guard let pool else { throw TesseraCacheError.closed }
        do {
            let value = try await pool.get(RedisKey(key)).get()
            return Self.unwrapString(value)
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// SET a key with optional TTL. TTL is in seconds; 0 means no expiry.
    public func set(_ key: String, value: String, ttlSeconds: Int = 0) async throws {
        guard let pool else { throw TesseraCacheError.closed }
        do {
            if ttlSeconds > 0 {
                _ = try await pool.setex(
                    RedisKey(key),
                    to: value,
                    expirationInSeconds: ttlSeconds
                ).get()
            } else {
                _ = try await pool.set(RedisKey(key), to: value).get()
            }
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// SET-NX (set if not exists) with optional TTL. Returns true if
    /// the key was created; false if it already existed. The TTL is
    /// omitted from the SET command when `ttlSeconds == 0` (Redis
    /// rejects `EX 0` as an invalid expire time).
    @discardableResult
    public func setIfAbsent(_ key: String, value: String, ttlSeconds: Int = 0) async throws -> Bool {
        guard let pool else { throw TesseraCacheError.closed }
        // Build the SET command via the lower-level send API so we
        // can pass NX (and EX only when ttlSeconds > 0).
        var args: [RESPValue] = [
            .init(from: RedisKey(key)),
            .init(from: value),
            .init(from: "NX"),
        ]
        if ttlSeconds > 0 {
            args.append(.init(from: "EX"))
            args.append(.init(from: Int64(ttlSeconds)))
        }
        do {
            let resp = try await pool.send(command: "SET", with: args).get()
            switch resp {
            case .simpleString(let buf) where String(buffer: buf) == "OK":
                return true
            case .null:
                return false
            default:
                throw TesseraCacheError.commandFailed("SET NX returned unexpected RESP: \(resp)")
            }
        } catch let err as TesseraCacheError {
            throw err
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// DEL one or more keys. Returns the number of keys deleted.
    @discardableResult
    public func del(_ keys: [String]) async throws -> Int {
        guard let pool else { throw TesseraCacheError.closed }
        do {
            return try await pool.delete(keys.map { RedisKey($0) }).get()
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// INCR a key (creating it at 0 if absent). Returns the new value.
    @discardableResult
    public func incr(_ key: String) async throws -> Int {
        guard let pool else { throw TesseraCacheError.closed }
        do {
            return try await pool.increment(RedisKey(key)).get()
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// EXPIRE a key. Returns true if the expiry was set; false if the
    /// key does not exist.
    @discardableResult
    public func expire(_ key: String, after seconds: Int) async throws -> Bool {
        guard let pool else { throw TesseraCacheError.closed }
        do {
            return try await pool.expire(RedisKey(key), after: .seconds(Int64(seconds))).get()
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// TTL of a key. -2 means "key does not exist"; -1 means "no expiry".
    public func ttl(_ key: String) async throws -> Int {
        guard let pool else { throw TesseraCacheError.closed }
        let args: [RESPValue] = [.init(from: RedisKey(key))]
        do {
            let resp = try await pool.send(command: "TTL", with: args).get()
            return Self.unwrapInt(resp)
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    // MARK: - Lists (queues)

    /// LPUSH one or more values. Returns the new list length.
    @discardableResult
    public func push(_ key: String, values: [String]) async throws -> Int {
        guard let pool else { throw TesseraCacheError.closed }
        do {
            return try await pool.lpush(values, into: RedisKey(key)).get()
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// BRPOP with timeout. Returns nil on timeout. `timeoutSeconds = 0`
    /// means block forever (consistent with the Redis protocol).
    public func pop(timeoutSeconds: Int, from keys: [String]) async throws -> CachePopResult? {
        guard let pool else { throw TesseraCacheError.closed }
        do {
            let resp = try await pool.brpop(
                from: keys.map { RedisKey($0) },
                timeout: .seconds(Int64(timeoutSeconds))
            ).get()
            guard let (key, value) = resp else { return nil }
            let keyStr = key.rawValue
            let valueStr = Self.unwrapString(value) ?? ""
            return CachePopResult(key: keyStr, value: valueStr)
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    // MARK: - Sorted sets (decay windows)

    /// ZADD one or more (member, score) pairs. Returns the number of
    /// NEW members added (i.e. not previously present).
    @discardableResult
    public func zadd(_ key: String, members: [(member: String, score: Double)]) async throws -> Int {
        guard let pool else { throw TesseraCacheError.closed }
        var args: [RESPValue] = [.init(from: RedisKey(key))]
        for m in members {
            args.append(.init(from: m.score))
            args.append(.init(from: m.member))
        }
        do {
            let resp = try await pool.send(command: "ZADD", with: args).get()
            return Self.unwrapInt(resp)
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// ZRANGEBYSCORE: members with score in `[min, max]`, ordered by
    /// score ascending. `withScores = true` returns the score alongside
    /// each member.
    public func zrangebyscore(
        _ key: String,
        min: Double,
        max: Double,
        withScores: Bool = false,
        limit: Int? = nil
    ) async throws -> [CacheScoredMember] {
        guard let pool else { throw TesseraCacheError.closed }
        var args: [RESPValue] = [
            .init(from: RedisKey(key)),
            .init(from: min),
            .init(from: max),
        ]
        if withScores {
            args.append(.init(from: "WITHSCORES"))
        }
        if let limit {
            args.append(.init(from: "LIMIT"))
            args.append(.init(from: Int64(0)))
            args.append(.init(from: Int64(limit)))
        }
        do {
            let resp = try await pool.send(command: "ZRANGEBYSCORE", with: args).get()
            return Self.unwrapScoredMembers(resp)
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    /// ZREMRANGEBYSCORE: remove members with score in `[min, max]`.
    /// Returns the number of members removed.
    @discardableResult
    public func zremrangebyscore(_ key: String, min: Double, max: Double) async throws -> Int {
        guard let pool else { throw TesseraCacheError.closed }
        let args: [RESPValue] = [
            .init(from: RedisKey(key)),
            .init(from: min),
            .init(from: max),
        ]
        do {
            let resp = try await pool.send(command: "ZREMRANGEBYSCORE", with: args).get()
            return Self.unwrapInt(resp)
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    // MARK: - EVAL (Lua scripts)

    /// EVAL a Lua script. Pass `keys` and `args` separately; the
    /// caller's values flow through `RESPValue` conversion. Use
    /// sparingly; the typed methods above cover the common cases.
    public func eval(
        _ script: String,
        keys: [String] = [],
        args: [String] = []
    ) async throws -> RESPValue {
        guard let pool else { throw TesseraCacheError.closed }
        var respArgs: [RESPValue] = [
            .init(from: script),
            .init(from: Int64(keys.count)),
        ]
        respArgs.append(contentsOf: keys.map { .init(from: RedisKey($0)) })
        respArgs.append(contentsOf: args.map { .init(from: $0) })
        do {
            return try await pool.send(command: "EVAL", with: respArgs).get()
        } catch {
            throw TesseraCacheError.commandFailed(String(describing: error))
        }
    }

    // MARK: - Internals

    private static func unwrapString(_ value: RESPValue) -> String? {
        switch value {
        case .null: return nil
        case .bulkString(let bytes?): return String(buffer: bytes)
        case .bulkString(.none): return nil
        case .simpleString(let bytes): return String(buffer: bytes)
        default: return nil
        }
    }

    private static func unwrapInt(_ value: RESPValue) -> Int {
        switch value {
        case .integer(let i): return Int(i)
        case .bulkString(let bytes?): return Int(String(buffer: bytes)) ?? 0
        case .bulkString(.none): return 0
        case .simpleString(let bytes): return Int(String(buffer: bytes)) ?? 0
        default: return 0
        }
    }

    private static func unwrapScoredMembers(_ value: RESPValue) -> [CacheScoredMember] {
        // ZRANGEBYSCORE WITHSCORES returns a flat array
        // [member1, score1, member2, score2, ...].
        guard case .array(let arr) = value else { return [] }
        var out: [CacheScoredMember] = []
        var i = 0
        while i + 1 < arr.count {
            let member = stringFromRESP(arr[i])
            let score = doubleFromRESP(arr[i + 1])
            if let member, let score {
                out.append(CacheScoredMember(member: member, score: score))
            }
            i += 2
        }
        return out
    }

    private static func stringFromRESP(_ v: RESPValue) -> String? {
        switch v {
        case .bulkString(let bytes?): return String(buffer: bytes)
        case .bulkString(.none): return nil
        case .simpleString(let bytes): return String(buffer: bytes)
        default: return nil
        }
    }

    private static func doubleFromRESP(_ v: RESPValue) -> Double? {
        switch v {
        case .bulkString(let bytes?): return Double(String(buffer: bytes))
        case .bulkString(.none): return nil
        case .simpleString(let bytes): return Double(String(buffer: bytes))
        case .integer(let i): return Double(i)
        default: return nil
        }
    }
}

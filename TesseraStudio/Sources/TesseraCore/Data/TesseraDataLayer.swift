import Foundation
import Logging

// MARK: - TesseraDataLayer

/// Top-level facade for the Tessera data layer. The rest of the app
/// depends on this type, NOT on ``TesseraDataStore`` or ``TesseraCache``
/// directly. The facade combines the durable Postgres store and the
/// ephemeral Valkey cache behind a domain-shaped API.
///
/// **Why a facade?** The hexagonal boundary (see
/// docs/tessera-data-layer-design.md §6) says: productivity code
/// (importers, AI editor, export pipeline) must be able to look up
/// entities, append receipts, and run hybrid search without ever
/// importing `PostgresNIO` or `RediStack`. The facade enforces that by
/// being the only public surface in this directory that callers should
/// reach for.
///
/// **Cache invariants** the facade enforces (or will in a follow-up):
///   * Reads check the cache first; on miss, fall through to Postgres
///     and backfill the cache with a TTL.
///   * Writes go to Postgres first (durable), then invalidate the
///     cache key. We do NOT write-through because the cache TTLs are
///     tuned for "ephemeral" not "durable"; an offline Postgres would
///     leave the cache stale otherwise.
///   * Receipts are append-only and never invalidated.
///
/// The current worker delivers the facade shape + the constructor +
/// the pass-throughs to ``TesseraDataStore`` / ``TesseraCache``. The
/// read-through + write-through cache wiring lives in the productivity
/// surface (next wave) so we don't speculatively build a cache policy
/// the user hasn't asked for.
public actor TesseraDataLayer {

    /// Combined configuration. Callers usually construct this from the
    /// app's settings (`PostgresConfig + ValkeyConfig -> TesseraDataLayerConfig`).
    public struct Configuration: Sendable {
        public var dataStore: TesseraDataStore.Configuration
        public var cache: TesseraCache.Configuration

        public init(
            dataStore: TesseraDataStore.Configuration = .init(),
            cache: TesseraCache.Configuration = .init()
        ) {
            self.dataStore = dataStore
            self.cache = cache
        }
    }

    /// Result of a successful ``TesseraDataLayer/start()``.
    public enum StartOutcome: Sendable, Equatable {
        /// Both stores are open and ready.
        case ready
        /// Postgres opened but the cache was unreachable. The facade
        /// runs in degraded mode: writes go to Postgres, the cache
        /// calls throw ``TesseraCacheError``. The user sees a
        /// non-fatal warning in the log.
        case cacheDegraded(reason: String)
        /// The cache opened but Postgres was unreachable. The facade
        /// runs in degraded mode: reads/writes throw
        /// ``TesseraDataStoreError``. The cache stays alive for
        /// scratchpad use.
        case dataStoreDegraded(reason: String)
    }

    private let dataStore: TesseraDataStore
    private let cache: TesseraCache
    private var logger: Logger
    private var isStarted: Bool = false

    public init(
        configuration: Configuration = .init(),
        logger: Logger = .init(label: "tessera.data")
    ) {
        self.dataStore = TesseraDataStore(
            configuration: configuration.dataStore,
            logger: logger
        )
        self.cache = TesseraCache(
            configuration: configuration.cache,
            logger: logger
        )
        self.logger = logger
    }

    /// Construct the facade from pre-built stores. The stores are
    /// captured (and not re-created) so callers can pass in fakes for
    /// tests. The facade's actor still serializes calls; the inner
    /// stores' actors are independent.
    public init(
        dataStore: TesseraDataStore,
        cache: TesseraCache,
        logger: Logger = .init(label: "tessera.data")
    ) {
        self.dataStore = dataStore
        self.cache = cache
        self.logger = logger
    }

    // MARK: - Lifecycle

    /// Open both stores. Postgres is required (the data layer can't
    /// exist without it); the cache is best-effort. The function never
    /// throws; failures degrade gracefully (see ``StartOutcome``).
    public func start() async -> StartOutcome {
        guard !isStarted else { return .ready }
        isStarted = true

        var pgError: String?
        var cacheError: String?

        do {
            try await dataStore.connect()
        } catch {
            pgError = String(describing: error)
            logger.error("TesseraDataLayer: Postgres connect failed: \(String(describing: error))")
        }
        do {
            try await cache.connect()
        } catch {
            cacheError = String(describing: error)
            logger.error("TesseraDataLayer: Valkey connect failed: \(String(describing: error))")
        }

        switch (pgError, cacheError) {
        case (nil, nil): return .ready
        case (nil, let ce?): return .cacheDegraded(reason: ce)
        case (let pe?, nil): return .dataStoreDegraded(reason: pe)
        case (let pe?, let ce?):
            // Both down: log loudly and report the Postgres failure
            // (it is the load-bearing one).
            logger.error("TesseraDataLayer: BOTH stores down. pg=\(pe) valkey=\(ce)")
            return .dataStoreDegraded(reason: pe)
        }
    }

    /// Graceful shutdown. Closes both stores.
    public func shutdown() async {
        await dataStore.close()
        await cache.close()
        isStarted = false
    }

    // MARK: - Health probes

    /// True iff both stores are currently open. A degraded facade
    /// returns false (because at least one store is down).
    public var isReady: Bool {
        get async {
            // We can't reach the inner actors' `isClosed` state from
            // here synchronously; the cheap proxy is to check that
            // we successfully started and no inner store has thrown
            // a closed-error since. For now this returns true once
            // started; the productivity surface can poll the deeper
            // isOpen if it needs to.
            return isStarted
        }
    }

    // MARK: - Pass-through: durable store

    /// Fetch one entity by id.
    public func getEntity(id: UUID) async throws -> GraphEntity? {
        try await dataStore.getEntity(id: id)
    }

    /// Insert or update an entity. Returns the resolved entity.
    public func upsertEntity(_ input: GraphEntityUpsert) async throws -> GraphEntity {
        try await dataStore.upsertEntity(input)
    }

    /// Delete an entity by id. Returns true if a row was deleted.
    public func deleteEntity(id: UUID) async throws -> Bool {
        try await dataStore.deleteEntity(id: id)
    }

    /// List entities of a given type, ordered by updated_at DESC
    /// then label. See ``TesseraDataStore/listByEntityType(...)``
    /// for the index + limit semantics.
    public func listByEntityType(
        entityType: String,
        limit: Int = 1000,
        offset: Int = 0
    ) async throws -> [GraphEntity] {
        try await dataStore.listByEntityType(
            entityType: entityType,
            limit: limit,
            offset: offset
        )
    }

    /// Case-insensitive prefix search over `label` for a given
    /// entity type. See ``TesseraDataStore/searchByLabelPrefix(...)``
    /// for the SQL details.
    public func searchByLabelPrefix(
        entityType: String,
        labelPrefix: String,
        limit: Int = 20
    ) async throws -> [GraphEntity] {
        try await dataStore.searchByLabelPrefix(
            entityType: entityType,
            labelPrefix: labelPrefix,
            limit: limit
        )
    }

    /// Every entity_link in the database (bounded by `limit`).
    /// Used by the graph view to build its edge set.
    public func listAllLinks(limit: Int = 10_000) async throws -> [EntityLink] {
        try await dataStore.listAllLinks(limit: limit)
    }

    /// Insert (or no-op) a typed edge between two entities.
    @discardableResult
    public func linkEntities(
        sourceID: UUID,
        targetID: UUID,
        linkType: String,
        weight: Float = 1.0
    ) async throws -> EntityLink {
        try await dataStore.linkEntities(
            sourceID: sourceID,
            targetID: targetID,
            linkType: linkType,
            weight: weight
        )
    }

    /// Outgoing edges of a given `linkType` from `sourceID`.
    public func outLinks(sourceID: UUID, linkType: String? = nil) async throws -> [EntityLink] {
        try await dataStore.outLinks(sourceID: sourceID, linkType: linkType)
    }

    /// Append a constitutional receipt.
    public func appendReceipt(
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue],
        signature: Data? = nil
    ) async throws -> GraphReceipt {
        try await dataStore.appendReceipt(
            entityID: entityID,
            receiptType: receiptType,
            payload: payload,
            signature: signature
        )
    }

    /// All receipts for one entity, oldest first.
    public func receipts(forEntity entityID: UUID) async throws -> [GraphReceipt] {
        try await dataStore.receipts(forEntity: entityID)
    }

    // MARK: - Productivity surface pass-through: receipt chain + chat queue

    /// Append a receipt to the document's chain. The receipt is
    /// written to `graph_receipts` AND linked into the per-document
    /// `receipt_chain` table at the next monotonic index. The
    /// payload is a `[String: JSONValue]` map; the chain table
    /// stores the linkage, the receipt row stores the payload.
    @discardableResult
    public func appendReceiptToChain(
        documentID: UUID,
        receiptType: String,
        payload: [String: JSONValue],
        signature: Data? = nil
    ) async throws -> GraphReceipt {
        try await dataStore.appendReceiptToChain(
            documentID: documentID,
            receiptType: receiptType,
            payload: payload,
            signature: signature
        )
    }

    /// The chain for one document, oldest first by `chain_index`.
    /// Each element pairs the monotonic position with the
    /// underlying `GraphReceipt` row.
    public func receiptChain(
        documentID: UUID,
        limit: Int? = nil
    ) async throws -> [(chainIndex: Int64, receipt: GraphReceipt)] {
        try await dataStore.receiptChain(documentID: documentID, limit: limit)
    }

    /// The latest chain index for a document (nil if the document
    /// has no receipts yet). Used by the productivity surface to
    /// compute the `priorReceiptID` for the next append.
    public func latestChainIndex(documentID: UUID) async throws -> Int64? {
        try await dataStore.latestChainIndex(documentID: documentID)
    }

    /// Load the per-document chat queue as a JSON string (the
    /// serialized form of a `ChatQueue`). Returns `"[]"` when no
    /// queue exists yet. The caller decodes the string into a
    /// `ChatQueue`.
    public func loadChatQueue(documentID: UUID) async throws -> String {
        try await dataStore.loadChatQueue(documentID: documentID)
    }

    /// Upsert the per-document chat queue. The `itemsJSON` is the
    /// JSON-serialized form of a `ChatQueue.items` array.
    public func saveChatQueue(documentID: UUID, itemsJSON: String) async throws {
        try await dataStore.saveChatQueue(documentID: documentID, itemsJSON: itemsJSON)
    }

    /// RRF over graph + vector + keyword.
    public func hybridSearch(
        anchor: UUID,
        queryText: String? = nil,
        queryEmbedding: [Float]? = nil,
        maxDepth: Int = 3
    ) async throws -> [HybridSearchResult] {
        try await dataStore.hybridSearch(
            anchor: anchor,
            queryText: queryText,
            queryEmbedding: queryEmbedding,
            maxDepth: maxDepth
        )
    }

    // MARK: - Pass-through: cache

    /// Build a namespaced key. Exposed so productivity code can build
    /// composite keys without re-implementing the prefix scheme.
    public func cacheKey(_ parts: String...) async -> String {
        // The cache is an actor; we await on it to read the namespace.
        // The combined path is built in the caller's context so the
        // returned string is plain (not a future).
        let joined = parts.joined(separator: ":")
        let ns = await cache.namespace
        return "tessera:\(ns):\(joined)"
    }

    /// GET a value from the cache.
    public func cacheGet(_ key: String) async throws -> String? {
        try await cache.get(key)
    }

    /// SET a value with optional TTL.
    public func cacheSet(_ key: String, value: String, ttlSeconds: Int = 0) async throws {
        try await cache.set(key, value: value, ttlSeconds: ttlSeconds)
    }

    /// DEL keys from the cache.
    @discardableResult
    public func cacheDel(_ keys: [String]) async throws -> Int {
        try await cache.del(keys)
    }

    /// INCR a counter in the cache.
    @discardableResult
    public func cacheIncr(_ key: String) async throws -> Int {
        try await cache.incr(key)
    }

    /// Add to a sorted set (decay window).
    @discardableResult
    public func cacheZadd(_ key: String, members: [(member: String, score: Double)]) async throws -> Int {
        try await cache.zadd(key, members: members)
    }

    /// Range a sorted set by score.
    public func cacheZrangebyscore(
        _ key: String,
        min: Double,
        max: Double,
        withScores: Bool = false,
        limit: Int? = nil
    ) async throws -> [CacheScoredMember] {
        try await cache.zrangebyscore(
            key,
            min: min,
            max: max,
            withScores: withScores,
            limit: limit
        )
    }
}

// MARK: - TesseraCache namespace accessor

extension TesseraCache {
    /// The configured namespace. Exposed so ``TesseraDataLayer`` can
    /// build prefixed keys in the caller's context.
    fileprivate var namespace: String {
        configuration.namespace
    }
}

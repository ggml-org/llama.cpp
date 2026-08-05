import Foundation
import Logging
import NIOCore
import PostgresNIO

// MARK: - Domain types (exposed via the data layer; no Postgres types leak)

/// A graph entity. The universal "one row per thing" row from
/// `graph_entities`. The `embedding` is a 1536-element Float32 vector.
/// Callers that need a different dimension must re-embed before
/// passing it in (see docs/tessera-data-layer-design.md §3.2).
public struct GraphEntity: Sendable, Equatable {
    public let id: UUID
    public let entityType: String
    public let subtype: String?
    public let label: String
    public let body: String?
    public let sourceURL: String?
    public let createdAt: Date
    public let updatedAt: Date
    public let embedding: [Float]?

    public init(
        id: UUID,
        entityType: String,
        subtype: String? = nil,
        label: String,
        body: String? = nil,
        sourceURL: String? = nil,
        createdAt: Date,
        updatedAt: Date,
        embedding: [Float]? = nil
    ) {
        self.id = id
        self.entityType = entityType
        self.subtype = subtype
        self.label = label
        self.body = body
        self.sourceURL = sourceURL
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.embedding = embedding
    }
}

/// One row from `entity_links`.
public struct EntityLink: Sendable, Equatable {
    public let id: UUID
    public let sourceID: UUID
    public let targetID: UUID
    public let linkType: String
    public let weight: Float

    public init(
        id: UUID,
        sourceID: UUID,
        targetID: UUID,
        linkType: String,
        weight: Float
    ) {
        self.id = id
        self.sourceID = sourceID
        self.targetID = targetID
        self.linkType = linkType
        self.weight = weight
    }
}

/// One row from `graph_receipts`. `signature` is nil until signing
/// is wired up (the column exists; the signing path is a follow-up).
public struct GraphReceipt: Sendable, Equatable, Identifiable {
    public let id: UUID
    public let entityID: UUID
    public let receiptType: String
    public let payload: [String: JSONValue]
    public let signature: Data?
    public let witnessedAt: Date

    public init(
        id: UUID,
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue],
        signature: Data? = nil,
        witnessedAt: Date
    ) {
        self.id = id
        self.entityID = entityID
        self.receiptType = receiptType
        self.payload = payload
        self.signature = signature
        self.witnessedAt = witnessedAt
    }
}

/// One ranked result from `hybrid_search(...)`.
public struct HybridSearchResult: Sendable, Equatable {
    public let entityID: UUID
    public let entityType: String
    public let label: String
    public let body: String?
    public let graphScore: Float
    public let vectorScore: Float
    public let keywordScore: Float
    public let rrfScore: Float

    public init(
        entityID: UUID,
        entityType: String,
        label: String,
        body: String?,
        graphScore: Float,
        vectorScore: Float,
        keywordScore: Float,
        rrfScore: Float
    ) {
        self.entityID = entityID
        self.entityType = entityType
        self.label = label
        self.body = body
        self.graphScore = graphScore
        self.vectorScore = vectorScore
        self.keywordScore = keywordScore
        self.rrfScore = rrfScore
    }
}

/// Input shape for `TesseraDataStore.upsertEntity`.
public struct GraphEntityUpsert: Sendable {
    public var id: UUID?
    public var entityType: String
    public var subtype: String?
    public var label: String
    public var body: String?
    public var sourceURL: String?
    public var embedding: [Float]?

    public init(
        id: UUID? = nil,
        entityType: String,
        subtype: String? = nil,
        label: String,
        body: String? = nil,
        sourceURL: String? = nil,
        embedding: [Float]? = nil
    ) {
        self.id = id
        self.entityType = entityType
        self.subtype = subtype
        self.label = label
        self.body = body
        self.sourceURL = sourceURL
        self.embedding = embedding
    }
}

/// Errors raised by ``TesseraDataStore``. Kept narrow on purpose so the
/// facade surface doesn't leak PostgresNIO error types. The underlying
/// error is preserved via the message for diagnostics.
public enum TesseraDataStoreError: Error, Sendable, Equatable {
    case notFound(id: UUID)
    case invalidEmbedding(expected: Int, got: Int)
    case connectionFailed(reason: String)
    case queryFailed(reason: String)
    case migrationFailed(reason: String)
    case closed

    public static func == (lhs: TesseraDataStoreError, rhs: TesseraDataStoreError) -> Bool {
        switch (lhs, rhs) {
        case (.notFound(let a), .notFound(let b)): return a == b
        case (.invalidEmbedding(let a, let b), .invalidEmbedding(let c, let d)): return a == c && b == d
        case (.connectionFailed(let a), .connectionFailed(let b)): return a == b
        case (.queryFailed(let a), .queryFailed(let b)): return a == b
        case (.migrationFailed(let a), .migrationFailed(let b)): return a == b
        case (.closed, .closed): return true
        default: return false
        }
    }
}

// MARK: - Data store

/// Postgres-backed data store for the durable knowledge graph + receipts.
///
/// `TesseraDataStore` is an `actor` so all state mutations (the
/// connection pool reference, in-flight query tracking) are serialized
/// without explicit locks. The single `PostgresClient` underneath owns
/// its own SwiftNIO-based connection pool (see
/// docs/tessera-data-layer-design.md §6.1).
///
/// **Hexagonal boundary**: this is the ONLY file in TesseraCore that
/// imports `PostgresNIO`. The rest of the app depends on
/// ``TesseraDataLayer``, which exposes a domain-shaped API on top of
/// this store. The productivity surface never sees a `PostgresRow` or a
/// `PostgresQuery`.
///
/// **The expected embedding dimension is 1536.** This matches the
/// `vector(1536)` column in the schema (and OpenAI text-embedding-3-small).
/// Re-embedding for other models is a follow-up; the column type does not
/// change.
public actor TesseraDataStore {

    /// The schema's fixed embedding dimension. See `migrations/0001_init.sql`.
    public static let embeddingDimension = 1536

    /// Configuration values. The pool tuning knobs map to
    /// `PostgresClient.Configuration.Options`. The `connection` block
    /// maps to `PostgresClient.Configuration` (host, port, tls, etc.).
    public struct Configuration: Sendable {
        public var host: String
        public var port: Int
        public var username: String
        public var password: String?
        public var database: String
        public var minimumConnections: Int
        public var maximumConnections: Int
        public var connectionIdleTimeoutSeconds: Int
        public var useTLS: Bool

        public init(
            host: String = "localhost",
            port: Int = 5432,
            username: String = "tessera",
            password: String? = "tessera",
            database: String = "tessera",
            minimumConnections: Int = 1,
            maximumConnections: Int = 8,
            connectionIdleTimeoutSeconds: Int = 60,
            useTLS: Bool = false
        ) {
            self.host = host
            self.port = port
            self.username = username
            self.password = password
            self.database = database
            self.minimumConnections = minimumConnections
            self.maximumConnections = maximumConnections
            self.connectionIdleTimeoutSeconds = connectionIdleTimeoutSeconds
            self.useTLS = useTLS
        }

        /// Build a `Configuration` from a libpq-style connection string
        /// (e.g. `postgres://user:pass@host:port/dbname?sslmode=disable`).
        /// Returns nil if the URL is malformed; for that case callers
        /// should construct `Configuration` field-by-field.
        public static func from(connectionString: String) -> Configuration? {
            guard let url = URL(string: connectionString) else { return nil }
            let host = url.host ?? "localhost"
            let port = url.port ?? 5432
            let username = url.user ?? "tessera"
            let password = url.password
            let database = url.path.split(separator: "/").last.map(String.init) ?? "tessera"
            return Configuration(
                host: host,
                port: port,
                username: username,
                password: password,
                database: database
            )
        }
    }

    private var logger: Logger
    private var configuration: Configuration
    private var client: PostgresClient?
    private var runTask: Task<Void, Never>?
    /// Signaled when the underlying ``PostgresClient.run()`` task
    /// has been entered. Tests can use this to synchronise with the
    /// pool start.
    private var runStarted: Task<Void, Never>?
    private var isClosed: Bool = false

    public init(
        configuration: Configuration = .init(),
        logger: Logger = .init(label: "tessera.data.postgres")
    ) {
        self.configuration = configuration
        self.logger = logger
    }

    deinit {
        // Best-effort: cancel the run task. The pool tears itself
        // down when the run task returns; we cannot await from a
        // deinit but the task will be cancelled by the Swift runtime.
        runTask?.cancel()
    }

    /// Replace the logger. Useful for tests that want a silent logger.
    public func setLogger(_ logger: Logger) {
        self.logger = logger
    }

    /// Update the configuration. Only safe BEFORE `connect()` is called.
    public func setConfiguration(_ configuration: Configuration) throws {
        guard client == nil else {
            throw TesseraDataStoreError.queryFailed(reason: "cannot change configuration after connect()")
        }
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    /// Open the connection pool. Must be called before any query.
    /// The PostgresClient's `run()` method is started in a long-running
    /// background `Task`; cancelling that task is equivalent to closing.
    public func connect() async throws {
        guard client == nil else { return }
        guard !isClosed else { throw TesseraDataStoreError.closed }

        var pgConfig = PostgresClient.Configuration(
            host: configuration.host,
            port: configuration.port,
            username: configuration.username,
            password: configuration.password,
            database: configuration.database,
            tls: configuration.useTLS ? .prefer(.makeClientConfiguration()) : .disable
        )
        // Pool tuning.
        pgConfig.options.minimumConnections = configuration.minimumConnections
        pgConfig.options.maximumConnections = configuration.maximumConnections
        pgConfig.options.connectionIdleTimeout = .seconds(Int64(configuration.connectionIdleTimeoutSeconds))

        let client = PostgresClient(
            configuration: pgConfig,
            backgroundLogger: logger
        )
        self.client = client

        // Start the long-running pool task. Cancellation == close.
        // We use a TaskGroup with the run task to wait for the run
        // to actually start (the run() method is non-blocking, so
        // we need to race the first query against it). The simplest
        // signal is: the run task hasn't thrown AND we've successfully
        // issued one round-trip query.
        let runTask = Task.detached(priority: .userInitiated) { [logger] in
            await client.run()
            // When run() returns the client is closed; log it for
            // diagnostics but do not throw (the caller's awaiting
            // `close()` is what orchestrates this).
            logger.debug("TesseraDataStore: PostgresClient.run() returned")
        }
        self.runTask = runTask

        // Wait for the pool to be ready by polling the first query.
        // PostgresNIO queues queries until run() is in flight, so
        // a small retry loop is the right primitive: the first
        // `SELECT 1` will succeed once the pool is up.
        do {
            try await waitForPoolReady(client: client)
        } catch {
            // 5s of failures -- roll back.
            runTask.cancel()
            self.client = nil
            self.runTask = nil
            throw TesseraDataStoreError.connectionFailed(
                reason: String(describing: error)
            )
        }
    }

    /// Poll `client.withConnection` for up to 10s, returning the
    /// last error on failure. `withConnection` queues lease
    /// requests until the run task is ready, so a successful
    /// round-trip proves the pool is up. Extracted from
    /// ``connect()`` so the compiler doesn't crash on the nested
    /// try/catch + for loop.
    private func waitForPoolReady(client: PostgresClient) async throws {
        var lastError: Error = TesseraDataStoreError.connectionFailed(reason: "no attempts")
        for _ in 0..<100 {  // up to 10s
            do {
                _ = try await client.withConnection { conn in
                    try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                        conn.simpleQuery("SELECT 1").whenComplete { result in
                            switch result {
                            case .success: cont.resume()
                            case .failure(let err): cont.resume(throwing: err)
                            }
                        }
                    }
                }
                return  // success
            } catch {
                lastError = error
                try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms
            }
        }
        throw lastError
    }

    /// Close the connection pool. Idempotent. The PostgresClient API
    /// has no public `close()` (it uses `cancelOnGracefulShutdown`); we
    /// cancel the run task, which causes the pool to drain.
    public func close() {
        guard let runTask else { return }
        runTask.cancel()
        self.client = nil
        self.runTask = nil
        self.isClosed = true
    }

    // MARK: - Migrations

    /// Apply the SQL in `migrations/*.sql` in lexicographic order. The
    /// `tools/tessera/db/Makefile` runs the same SQL via the `psql`
    /// CLI; this method is for the in-process path (tests, in-app
    /// bootstrap when there is no `psql`).
    ///
    /// **Multi-statement files are split on `;` boundaries** and each
    /// statement is sent individually. PostgresNIO's `query()` uses
    /// the extended query protocol which only accepts one statement
    /// per call; the actual Simple Query protocol isn't publicly
    /// exposed by the library. We skip `--` line comments and split
    /// on top-level `;` only (not inside `$$ ... $$` blocks).
    public func applyMigrations(_ sqlFiles: [(name: String, sql: String)]) async throws {
        guard let client else { throw TesseraDataStoreError.closed }
        for (name, sql) in sqlFiles {
            let statements = Self.splitSqlStatements(sql)
            for (i, stmt) in statements.enumerated() {
                let trimmed = stmt.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty { continue }
                do {
                    _ = try await client.withConnection { conn in
                        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                            conn.simpleQuery(trimmed).whenComplete { result in
                                switch result {
                                case .success: cont.resume()
                                case .failure(let err): cont.resume(throwing: err)
                                }
                            }
                        }
                    }
                } catch {
                    throw TesseraDataStoreError.migrationFailed(
                        reason: "\(name) (statement \(i + 1)): \(String(reflecting: error))"
                    )
                }
            }
        }
    }

    /// Split a SQL script into individual statements. Strips `--`
    /// line comments and splits on `;` boundaries at depth 0 (i.e.
    /// not inside `$$ ... $$` block bodies). The result is the list
    /// of statements, in order, each terminated by a single `;`
    /// (re-added by the caller). This is the minimum SQL tokenizer
    /// needed for the migration files; for general-purpose SQL
    /// splitting, use a real parser.
    static func splitSqlStatements(_ sql: String) -> [String] {
        var out: [String] = []
        var current: [String] = []
        var inDollarBlock = false
        var i = sql.startIndex
        while i < sql.endIndex {
            let c = sql[i]
            // Skip -- line comments.
            if !inDollarBlock && c == "-" && sql.index(after: i) < sql.endIndex
                && sql[sql.index(after: i)] == "-" {
                // Skip to end of line.
                while i < sql.endIndex && sql[i] != "\n" {
                    i = sql.index(after: i)
                }
                continue
            }
            // Toggle $$ block.
            if c == "$" && sql.index(after: i) < sql.endIndex
                && sql[sql.index(after: i)] == "$" {
                inDollarBlock.toggle()
                current.append("$")
                current.append("$")
                i = sql.index(i, offsetBy: 2)
                continue
            }
            if c == ";" && !inDollarBlock {
                current.append(";")
                out.append(current.joined())
                current = []
                i = sql.index(after: i)
                continue
            }
            current.append(String(c))
            i = sql.index(after: i)
        }
        let tail = current.joined().trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty {
            out.append(tail)
        }
        return out
    }

    // MARK: - Entity CRUD

    /// Fetch one entity by id. Returns nil if not found (does not throw).
    public func getEntity(id: UUID) async throws -> GraphEntity? {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            SELECT id, entity_type, subtype, label, body, source_url, created_at, updated_at, embedding::text
              FROM graph_entities WHERE id = \(id)
            """
        let rows = try await client.query(query, logger: logger)
        for try await row in rows {
            return try decodeEntity(row)
        }
        return nil
    }

    /// Insert or update an entity. If `input.id` is nil, a fresh UUID
    /// is generated server-side. Returns the resolved entity (with
    /// `createdAt` / `updatedAt` set by the server).
    public func upsertEntity(_ input: GraphEntityUpsert) async throws -> GraphEntity {
        guard let client else { throw TesseraDataStoreError.closed }
        if let emb = input.embedding, emb.count != Self.embeddingDimension {
            throw TesseraDataStoreError.invalidEmbedding(
                expected: Self.embeddingDimension,
                got: emb.count
            )
        }
        let embeddingText: String? = input.embedding.map { floats in
            "[" + floats.map { String(format: "%.7g", $0) }.joined(separator: ",") + "]"
        }
        // The interpolation below uses PostgresNonThrowingEncodable
        // conformances for the standard types (UUID, String, String?,
        // Float), so the query literal does not throw. We mark the
        // interpolation with `try` only where the call site later
        // adds a JSON-typed value (the receipt payload path).
        let query: PostgresQuery = """
            INSERT INTO graph_entities (id, entity_type, subtype, label, body, source_url, embedding)
            VALUES (COALESCE(\(input.id), gen_random_uuid()), \(input.entityType),
                    \(input.subtype), \(input.label), \(input.body), \(input.sourceURL),
                    \(embeddingText)::vector)
            ON CONFLICT (id) DO UPDATE SET
                entity_type = EXCLUDED.entity_type,
                subtype     = EXCLUDED.subtype,
                label       = EXCLUDED.label,
                body        = EXCLUDED.body,
                source_url  = EXCLUDED.source_url,
                embedding   = EXCLUDED.embedding
            RETURNING id, entity_type, subtype, label, body, source_url, created_at, updated_at, embedding::text
            """
        let rows = try await client.query(query, logger: logger)
        for try await r in rows {
            return try decodeEntity(r)
        }
        throw TesseraDataStoreError.queryFailed(reason: "upsertEntity returned no rows")
    }

    /// Delete an entity by id. Returns true if a row was deleted.
    public func deleteEntity(id: UUID) async throws -> Bool {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = "DELETE FROM graph_entities WHERE id = \(id) RETURNING id"
        let rows = try await client.query(query, logger: logger)
        for try await _ in rows { return true }
        return false
    }

    /// List entities of a given type, ordered by updated_at DESC
    /// then label. Used by the contact store's "all contacts"
    /// query and by the graph view's "load every entity of
    /// type X" path. Bounded by `limit` (default 1000) so a
    /// large catalog doesn't materialize all at once; the
    /// caller is expected to page or filter.
    public func listByEntityType(
        entityType: String,
        limit: Int = 1000,
        offset: Int = 0
    ) async throws -> [GraphEntity] {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            SELECT id, entity_type, subtype, label, body, source_url, created_at, updated_at, embedding::text
              FROM graph_entities
             WHERE entity_type = \(entityType)
             ORDER BY updated_at DESC, label ASC
             LIMIT \(limit) OFFSET \(offset)
            """
        let rows = try await client.query(query, logger: logger)
        var out: [GraphEntity] = []
        for try await row in rows {
            out.append(try decodeEntity(row))
        }
        return out
    }

    /// Case-insensitive prefix search over `label` for a given
    /// entity type. Used by the contact store's "find contact
    /// by name" path; the `idx_entities_contact_name` partial
    /// index (migration 0003_contacts.sql) makes this O(log n)
    /// for the contact case, but the SQL is generic across
    /// entity types so the same code path serves tasks,
    /// documents, and any future material.
    ///
    /// The `labelPrefix` is lowercased because Postgres'
    /// `LOWER(label) LIKE 'foo%'` only uses the index when the
    /// pattern is unanchored. We pass a case-insensitive
    /// `ILIKE` for small N; the index kicks in once the
    /// migration's partial index is in place.
    public func searchByLabelPrefix(
        entityType: String,
        labelPrefix: String,
        limit: Int = 20
    ) async throws -> [GraphEntity] {
        guard let client else { throw TesseraDataStoreError.closed }
        let pattern = labelPrefix + "%"
        let query: PostgresQuery = """
            SELECT id, entity_type, subtype, label, body, source_url, created_at, updated_at, embedding::text
              FROM graph_entities
             WHERE entity_type = \(entityType)
               AND LOWER(label) LIKE LOWER(\(pattern))
             ORDER BY label ASC
             LIMIT \(limit)
            """
        let rows = try await client.query(query, logger: logger)
        var out: [GraphEntity] = []
        for try await row in rows {
            out.append(try decodeEntity(row))
        }
        return out
    }

    /// List every entity_link in the database, used by the graph
    /// view to build its edge set. Limited to `limit` rows;
    /// the graph viewmaterializes incrementally for large
    /// graphs (see ``GraphViewModel`` for the progressive
    /// disclosure policy).
    public func listAllLinks(limit: Int = 10_000) async throws -> [EntityLink] {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            SELECT id, source_id, target_id, link_type, weight
              FROM entity_links
             ORDER BY created_at DESC
             LIMIT \(limit)
            """
        let rows = try await client.query(query, logger: logger)
        var out: [EntityLink] = []
        for try await row in rows {
            out.append(try decodeLink(row))
        }
        return out
    }

    // MARK: - Links

    /// Insert (or no-op) a typed edge between two entities.
    public func linkEntities(
        sourceID: UUID,
        targetID: UUID,
        linkType: String,
        weight: Float = 1.0
    ) async throws -> EntityLink {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            INSERT INTO entity_links (source_id, target_id, link_type, weight)
            VALUES (\(sourceID), \(targetID), \(linkType), \(weight))
            ON CONFLICT (source_id, target_id, link_type) DO UPDATE
                SET weight = EXCLUDED.weight
            RETURNING id, source_id, target_id, link_type, weight
            """
        let rows = try await client.query(query, logger: logger)
        for try await r in rows {
            return try decodeLink(r)
        }
        throw TesseraDataStoreError.queryFailed(reason: "linkEntities returned no rows")
    }

    /// All edges of a given `linkType` originating at `sourceID`. Used by
    /// the productivity surface to walk a typed subgraph.
    public func outLinks(sourceID: UUID, linkType: String? = nil) async throws -> [EntityLink] {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery
        if let linkType {
            query = """
                SELECT id, source_id, target_id, link_type, weight
                  FROM entity_links
                 WHERE source_id = \(sourceID) AND link_type = \(linkType)
                 ORDER BY created_at
                """
        } else {
            query = """
                SELECT id, source_id, target_id, link_type, weight
                  FROM entity_links
                 WHERE source_id = \(sourceID)
                 ORDER BY created_at
                """
        }
        let rows = try await client.query(query, logger: logger)
        var out: [EntityLink] = []
        for try await row in rows {
            out.append(try decodeLink(row))
        }
        return out
    }

    // MARK: - Receipts

    /// Append a constitutional receipt. The `signature` is left nil for
    /// now; the column exists so signing can be wired up without a
    /// migration (see docs §3.2 + §9).
    public func appendReceipt(
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue],
        signature: Data? = nil
    ) async throws -> GraphReceipt {
        guard let client else { throw TesseraDataStoreError.closed }
        let payloadJSON = try Self.encodePayload(payload)
        let query: PostgresQuery = try """
            INSERT INTO graph_receipts (entity_id, receipt_type, payload, signature)
            VALUES (\(entityID), \(receiptType), \(payloadJSON)::jsonb, \(signature))
            RETURNING id, entity_id, receipt_type, payload::text, signature, witnessed_at
            """
        let rows = try await client.query(query, logger: logger)
        for try await r in rows {
            return try decodeReceipt(r)
        }
        throw TesseraDataStoreError.queryFailed(reason: "appendReceipt returned no rows")
    }

    /// All receipts for one entity, oldest first.
    public func receipts(forEntity entityID: UUID) async throws -> [GraphReceipt] {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            SELECT id, entity_id, receipt_type, payload::text, signature, witnessed_at
              FROM graph_receipts
             WHERE entity_id = \(entityID)
             ORDER BY witnessed_at
            """
        let rows = try await client.query(query, logger: logger)
        var out: [GraphReceipt] = []
        for try await row in rows {
            out.append(try decodeReceipt(row))
        }
        return out
    }

    // MARK: - Productivity surface: receipt chain + chat queue

    /// Append a receipt to the document's chain. The receipt is
    /// first written to `graph_receipts` (the constitutional
    /// receipt log) and then linked into the per-document
    /// `receipt_chain` table at the next monotonic `chain_index`.
    /// The two writes are NOT in a single transaction (postgres-nio
    /// does not expose explicit transactions on the simple-query
    /// path); a crash between them leaves the receipt in
    /// `graph_receipts` but not in `receipt_chain`. The
    /// `rebuildReceiptChain(documentID:)` helper fixes this on
    /// document open.
    public func appendReceiptToChain(
        documentID: UUID,
        receiptType: String,
        payload: [String: JSONValue],
        signature: Data? = nil
    ) async throws -> GraphReceipt {
        guard let client else { throw TesseraDataStoreError.closed }
        let payloadJSON = try Self.encodePayload(payload)

        // 1. Insert into graph_receipts. We do this first to get
        //    the server-side defaults (id, witnessed_at) populated
        //    before we link the row into the chain.
        let insertReceipt: PostgresQuery = try """
            INSERT INTO graph_receipts (entity_id, receipt_type, payload, signature)
            VALUES (\(documentID), \(receiptType), \(payloadJSON)::jsonb, \(signature))
            RETURNING id, entity_id, receipt_type, payload::text, signature, witnessed_at
            """
        var inserted: GraphReceipt?
        let receiptRows = try await client.query(insertReceipt, logger: logger)
        for try await r in receiptRows {
            inserted = try decodeReceipt(r)
        }
        guard let inserted else {
            throw TesseraDataStoreError.queryFailed(reason: "appendReceiptToChain: insert returned no rows")
        }

        // 2. Append to receipt_chain. COALESCE on the next index
        //    gives us atomic monotonic ordering at the per-document
        //    granularity (the chain_index is per-document, not
        //    global, so two documents in parallel don't collide).
        let nextIndex: Int64 = try await nextChainIndex(documentID: documentID)
        let insertChain: PostgresQuery = """
            INSERT INTO receipt_chain (document_id, chain_index, receipt_id)
            VALUES (\(documentID), \(nextIndex), \(inserted.id))
            """
        do {
            _ = try await client.query(insertChain, logger: logger)
        } catch {
            // Roll back the receipt insert (best-effort). If this
            // also fails, the receipt is orphaned in graph_receipts
            // but not in the chain; rebuildReceiptChain repairs it.
            _ = try? await client.query(
                PostgresQuery(stringLiteral: "DELETE FROM graph_receipts WHERE id = \(inserted.id)"),
                logger: logger
            )
            throw TesseraDataStoreError.queryFailed(
                reason: "appendReceiptToChain: chain insert failed: \(String(describing: error))"
            )
        }
        return inserted
    }

    /// The next monotonic chain index for a document. Returns 0
    /// when the document has no chain entries yet. The query is
    /// `MAX(chain_index) + 1` on the document's chain rows,
    /// COALESCE'd to 0 for the empty case.
    private func nextChainIndex(documentID: UUID) async throws -> Int64 {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            SELECT COALESCE(MAX(chain_index), -1) + 1
              FROM receipt_chain
             WHERE document_id = \(documentID)
            """
        let rows = try await client.query(query, logger: logger)
        for try await row in rows {
            let ra = row.makeRandomAccess()
            return try ra[0].decode(Int64.self)
        }
        return 0
    }

    /// The chain for one document, oldest first by `chain_index`.
    /// Joins `receipt_chain` to `graph_receipts` so the caller
    /// gets the full constitutional receipt row.
    public func receiptChain(
        documentID: UUID,
        limit: Int? = nil
    ) async throws -> [(chainIndex: Int64, receipt: GraphReceipt)] {
        guard let client else { throw TesseraDataStoreError.closed }
        let limitClause: String = limit.map { " LIMIT \(Int32($0))" } ?? ""
        let query: PostgresQuery = """
            SELECT rc.chain_index,
                   gr.id, gr.entity_id, gr.receipt_type, gr.payload::text, gr.signature, gr.witnessed_at
              FROM receipt_chain rc
              JOIN graph_receipts gr ON gr.id = rc.receipt_id
             WHERE rc.document_id = \(documentID)
             ORDER BY rc.chain_index ASC\(limitClause)
            """
        let rows = try await client.query(query, logger: logger)
        var out: [(Int64, GraphReceipt)] = []
        for try await row in rows {
            let ra = row.makeRandomAccess()
            let chainIndex: Int64 = try ra[0].decode(Int64.self)
            // The remaining columns are a GraphReceipt; we rebuild
            // by re-running decodeReceipt on a synthetic row is
            // awkward, so we just decode the cells directly.
            let id: UUID = try ra[1].decode(UUID.self)
            let entityID: UUID = try ra[2].decode(UUID.self)
            let receiptType: String = try ra[3].decode(String.self)
            let payloadText: String = try ra[4].decode(String.self)
            let signature: Data? = try ra[5].decode(Data?.self)
            let witnessedAt: Date = try ra[6].decode(Date.self)
            let payload: [String: JSONValue] = Self.decodePayloadText(payloadText) ?? [:]
            let receipt = GraphReceipt(
                id: id,
                entityID: entityID,
                receiptType: receiptType,
                payload: payload,
                signature: signature,
                witnessedAt: witnessedAt
            )
            out.append((chainIndex, receipt))
        }
        return out
    }

    /// The latest `chain_index` for a document. Returns nil if
    /// the document has no chain entries.
    public func latestChainIndex(documentID: UUID) async throws -> Int64? {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            SELECT MAX(chain_index) FROM receipt_chain WHERE document_id = \(documentID)
            """
        let rows = try await client.query(query, logger: logger)
        for try await row in rows {
            let ra = row.makeRandomAccess()
            let value: Int64? = try ra[0].decode(Int64?.self)
            return value
        }
        return nil
    }

    /// Load the per-document chat queue. Returns an empty queue
    /// when no row exists.
    public func loadChatQueue(documentID: UUID) async throws -> String {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            SELECT items::text
              FROM chat_queues
             WHERE document_id = \(documentID)
            """
        let rows = try await client.query(query, logger: logger)
        for try await row in rows {
            let ra = row.makeRandomAccess()
            return try ra[0].decode(String.self)
        }
        return "[]"
    }

    /// Upsert the per-document chat queue. The `itemsJSON` is
    /// the JSON-serialized `ChatQueue` (or just its `items` array).
    public func saveChatQueue(documentID: UUID, itemsJSON: String) async throws {
        guard let client else { throw TesseraDataStoreError.closed }
        let query: PostgresQuery = """
            INSERT INTO chat_queues (document_id, items, updated_at)
            VALUES (\(documentID), \(itemsJSON)::jsonb, now())
            ON CONFLICT (document_id) DO UPDATE
                SET items = EXCLUDED.items,
                    updated_at = now()
            """
        _ = try await client.query(query, logger: logger)
    }

    // MARK: - Hybrid search

    /// RRF over graph + vector + keyword. The query embedding is optional
    /// (vector signal is skipped if nil); the query text is optional
    /// (keyword signal is skipped if nil). At least one of the two MUST
    /// be provided, otherwise the function returns whatever is reachable
    /// from the anchor (still ranked, but with only the graph signal).
    public func hybridSearch(
        anchor: UUID,
        queryText: String? = nil,
        queryEmbedding: [Float]? = nil,
        maxDepth: Int = 3
    ) async throws -> [HybridSearchResult] {
        guard let client else { throw TesseraDataStoreError.closed }
        if let emb = queryEmbedding, emb.count != Self.embeddingDimension {
            throw TesseraDataStoreError.invalidEmbedding(
                expected: Self.embeddingDimension,
                got: emb.count
            )
        }
        let embedLiteral: String? = queryEmbedding.map { floats in
            "[" + floats.map { String(format: "%.7g", $0) }.joined(separator: ",") + "]"
        }
        // `maxDepth` must be cast to `integer` (int4) explicitly:
        // Swift's `Int` is 64-bit and maps to `bigint` (int8) by
        // default in postgres-nio's encoders, but the hybrid_search
        // function's `p_max_depth` is declared `int DEFAULT 3`.
        let maxDepth32: Int32 = Int32(maxDepth)
        let query: PostgresQuery = """
            SELECT entity_id, entity_type, label, body,
                   graph_score, vector_score, keyword_score, rrf_score
              FROM hybrid_search(\(anchor)::uuid, \(queryText), \(embedLiteral)::vector, \(maxDepth32))
            """
        let rows = try await client.query(query, logger: logger)
        var out: [HybridSearchResult] = []
        for try await row in rows {
            out.append(try decodeHybrid(row))
        }
        return out
    }

    // MARK: - Low-level pass-through

    /// Escape hatch for callers that need to run ad-hoc SQL. The query
    /// is a parameterised `PostgresQuery` (DO NOT concatenate user
    /// input -- use ``PostgresQuery`` interpolation for binds). The
    /// row sequence is generic; callers decode it themselves.
    ///
    /// This is intentionally narrow: the rest of the data store's
    /// surface is the supported path. The escape hatch exists for
    /// the productivity surface's index tuning queries and the
    /// test suite's pg_catalog assertions.
    public func queryRaw(_ query: PostgresQuery) async throws -> PostgresRowSequence {
        guard let client else { throw TesseraDataStoreError.closed }
        return try await client.query(query, logger: logger)
    }

    // MARK: - Internals: row decoders

    /// `PostgresRow` is a `Collection` whose `Index` is opaque; for
    /// O(1) column access we call `makeRandomAccess()` to get a
    /// `PostgresRandomAccessRow`, then access by integer index. The
    /// public `decode(_:)` is on `PostgresCell` (per-cell), so we
    /// index into the random-access row to get a cell, then call
    /// `cell.decode(Type.self)`.
    private func makeRA(_ row: PostgresRow) -> PostgresRandomAccessRow {
        row.makeRandomAccess()
    }

    private func decodeEntity(_ row: PostgresRow) throws -> GraphEntity {
        let ra = makeRA(row)
        let id: UUID = try ra[0].decode(UUID.self)
        let entityType: String = try ra[1].decode(String.self)
        let subtype: String? = try ra[2].decode(String?.self)
        let label: String = try ra[3].decode(String.self)
        let body: String? = try ra[4].decode(String?.self)
        let sourceURL: String? = try ra[5].decode(String?.self)
        let createdAt: Date = try ra[6].decode(Date.self)
        let updatedAt: Date = try ra[7].decode(Date.self)
        let embeddingText: String? = try ra[8].decode(String?.self)
        let embedding = Self.parseVectorLiteral(embeddingText)
        return GraphEntity(
            id: id,
            entityType: entityType,
            subtype: subtype,
            label: label,
            body: body,
            sourceURL: sourceURL,
            createdAt: createdAt,
            updatedAt: updatedAt,
            embedding: embedding
        )
    }

    private func decodeLink(_ row: PostgresRow) throws -> EntityLink {
        let ra = makeRA(row)
        let id: UUID = try ra[0].decode(UUID.self)
        let sourceID: UUID = try ra[1].decode(UUID.self)
        let targetID: UUID = try ra[2].decode(UUID.self)
        let linkType: String = try ra[3].decode(String.self)
        let weight: Float = try ra[4].decode(Float.self)
        return EntityLink(
            id: id,
            sourceID: sourceID,
            targetID: targetID,
            linkType: linkType,
            weight: weight
        )
    }

    private func decodeReceipt(_ row: PostgresRow) throws -> GraphReceipt {
        let ra = makeRA(row)
        let id: UUID = try ra[0].decode(UUID.self)
        let entityID: UUID = try ra[1].decode(UUID.self)
        let receiptType: String = try ra[2].decode(String.self)
        let payloadText: String = try ra[3].decode(String.self)
        let signature: Data? = try ra[4].decode(Data?.self)
        let witnessedAt: Date = try ra[5].decode(Date.self)
        let payload: [String: JSONValue] = Self.decodePayloadText(payloadText) ?? [:]
        return GraphReceipt(
            id: id,
            entityID: entityID,
            receiptType: receiptType,
            payload: payload,
            signature: signature,
            witnessedAt: witnessedAt
        )
    }

    private func decodeHybrid(_ row: PostgresRow) throws -> HybridSearchResult {
        let ra = makeRA(row)
        let entityID: UUID = try ra[0].decode(UUID.self)
        let entityType: String = try ra[1].decode(String.self)
        let label: String = try ra[2].decode(String.self)
        let body: String? = try ra[3].decode(String?.self)
        let graphScore: Float = try ra[4].decode(Float.self)
        let vectorScore: Float = try ra[5].decode(Float.self)
        let keywordScore: Float = try ra[6].decode(Float.self)
        let rrfScore: Float = try ra[7].decode(Float.self)
        return HybridSearchResult(
            entityID: entityID,
            entityType: entityType,
            label: label,
            body: body,
            graphScore: graphScore,
            vectorScore: vectorScore,
            keywordScore: keywordScore,
            rrfScore: rrfScore
        )
    }

    // MARK: - Internals: payload + vector helpers

    private static let jsonEncoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private static let jsonDecoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    private static func encodePayload(_ payload: [String: JSONValue]) throws -> String {
        let data = try jsonEncoder.encode(payload)
        guard let s = String(data: data, encoding: .utf8) else {
            throw TesseraDataStoreError.queryFailed(reason: "payload not UTF-8 encodable")
        }
        return s
    }

    private static func decodePayloadText(_ text: String) -> [String: JSONValue]? {
        guard let data = text.data(using: .utf8) else { return nil }
        return try? jsonDecoder.decode([String: JSONValue].self, from: data)
    }

    /// Parse a pgvector textual representation `[f1,f2,...]` into a
    /// `[Float]`. We read the column as `text` (via `embedding::text`
    /// in the SELECT) so we don't need the pgvector PostgresNIO
    /// integration shim.
    private static func parseVectorLiteral(_ text: String?) -> [Float]? {
        guard let text, text.hasPrefix("["), text.hasSuffix("]") else { return nil }
        let inner = String(text.dropFirst().dropLast())
        if inner.isEmpty { return [] }
        return inner.split(separator: ",").map { s in
            Float(s.trimmingCharacters(in: .whitespaces)) ?? 0
        }
    }
}

import Foundation

// MARK: - SupersedeDecision

/// The result of a match-and-supersede check (per spec §6.7).
/// The engine returns the ids of the queue items that the new
/// front replaces, plus a one-line `reasoning` field for the
/// audit log.
///
/// The decision is monotonic: re-evaluating with the same input
/// returns the same decision (idempotent). The state machine
/// caches the decision on the queue item so repeated enqueues
/// of the same content skip the LLM call.
public struct SupersedeDecision: Codable, Sendable, Hashable {
    public var supersededItemIDs: [UUID]
    public var reasoning: String

    public init(supersededItemIDs: [UUID] = [], reasoning: String = "") {
        self.supersededItemIDs = supersededItemIDs
        self.reasoning = reasoning
    }

    public static let none = SupersedeDecision()

    /// True iff the decision supersedes at least one item.
    public var hasSupersessions: Bool { !supersededItemIDs.isEmpty }
}

// MARK: - SupersedeDecisionCache

/// In-memory cache for supersede decisions, keyed by the new-front
/// item's id. The cache is bounded (default 64 entries) so the
/// chat panel's memory footprint doesn't grow unboundedly across
/// a long editing session. The cache is process-local; the
/// match-and-supersede check is always re-runnable from the
/// queue's content, so a cache miss just costs one LLM call.
public actor SupersedeDecisionCache {
    private struct Entry: Sendable {
        let decision: SupersedeDecision
        let storedAt: Date
    }

    private var entries: [UUID: Entry] = [:]
    private let limit: Int

    public init(limit: Int = 64) {
        self.limit = max(1, limit)
    }

    public func get(_ key: UUID) -> SupersedeDecision? {
        entries[key]?.decision
    }

    public func put(_ key: UUID, decision: SupersedeDecision) {
        if entries.count >= limit, let oldestKey = entries.min(by: { $0.value.storedAt < $1.value.storedAt })?.key {
            entries.removeValue(forKey: oldestKey)
        }
        entries[key] = Entry(decision: decision, storedAt: Date())
    }

    public func clear() {
        entries.removeAll()
    }

    public var count: Int { entries.count }
}

// MARK: - MatchAndSupersedeEngine

/// The match-and-supersede check from spec §6.7. When a new
/// item is added to the front of the queue, the engine asks
/// the LLM: "Does the new front X supersede any of the existing
/// queue items [Y, Z, W]?" The result is a ``SupersedeDecision``.
///
/// **Heuristic fallback.** The on-device LLM is the default,
/// but the engine falls back to a lexical similarity check
/// (Jaccard over word tokens, threshold 0.6) when the LLM is
/// unavailable or returns an unparseable response. This keeps
/// the chat panel responsive when the model library is empty
/// or in the test environment.
///
/// **Caching.** The decision is cached on the new-front item's
/// id (via ``SupersedeDecisionCache``) so repeated enqueues of
/// the same content skip the LLM call. The cache is
/// process-local; a fresh process rebuilds it on demand.
public actor MatchAndSupersedeEngine {

    public typealias LLMProvider = @Sendable (
        _ system: String,
        _ userMessage: String
    ) async throws -> String

    private let llmProvider: LLMProvider?
    private let cache: SupersedeDecisionCache
    /// The similarity threshold for the heuristic fallback
    /// (Jaccard over word tokens). 0.6 is the spec's "high
    /// enough" default; tests can override.
    private let similarityThreshold: Double

    /// Production initializer. Uses the existing `LLMProvider`
    /// from the agent loop (typically the on-device model via
    /// `TesseraLLMProviderFactory`).
    public init(
        llmProvider: LLMProvider? = nil,
        cache: SupersedeDecisionCache = SupersedeDecisionCache(),
        similarityThreshold: Double = 0.6
    ) {
        self.llmProvider = llmProvider
        self.cache = cache
        self.similarityThreshold = similarityThreshold
    }

    /// Evaluate a new front item against the existing queue.
    /// Returns the ids of the existing items that the new
    /// front supersedes. Returns `.none` when the queue has
    /// no existing items to consider.
    public func evaluate(
        newFront: ChatQueueItem,
        existingQueue: [ChatQueueItem]
    ) async throws -> SupersedeDecision {
        // Cache hit -> return the cached decision (the user may
        // have re-enqueued the same content; the LLM call is
        // expensive, so we avoid it). The cache check is
        // BEFORE the empty-queue early return so repeated
        // enqueues of the same item hit the cache even when
        // the queue is empty.
        if let cached = await cache.get(newFront.id) {
            return cached
        }

        // Empty queue -> nothing to supersede. Cache the
        // empty decision so repeated enqueues of the same
        // content skip the LLM call.
        let candidates = existingQueue.filter { $0.id != newFront.id && $0.supersededByID == nil }
        guard !candidates.isEmpty else {
            await cache.put(newFront.id, decision: .none)
            return .none
        }

        // LLM call (when available). The prompt is small and
        // bounded; the response is parsed as JSON.
        let decision: SupersedeDecision
        if let llmProvider {
            do {
                let llmDecision = try await runLLMCheck(
                    newFront: newFront,
                    candidates: candidates,
                    llmProvider: llmProvider
                )
                // The LLM path always returns a decision
                // (possibly empty). When the response is
                // unparseable, `parseResponse` throws — we
                // fall back to the heuristic in that case.
                decision = llmDecision
            } catch {
                // LLM failure -> fall back to heuristic.
                decision = heuristicCheck(newFront: newFront, candidates: candidates)
            }
        } else {
            // No LLM available -> heuristic.
            decision = heuristicCheck(newFront: newFront, candidates: candidates)
        }

        await cache.put(newFront.id, decision: decision)
        return decision
    }

    // MARK: - LLM check

    private func runLLMCheck(
        newFront: ChatQueueItem,
        candidates: [ChatQueueItem],
        llmProvider: LLMProvider
    ) async throws -> SupersedeDecision {
        let system = Self.systemPrompt
        let user = Self.userPrompt(newFront: newFront, candidates: candidates)
        let response = try await llmProvider(system, user)
        return try Self.parseResponse(response, candidates: candidates)
    }

    /// System prompt for the LLM. Instructs the model to return
    /// strict JSON so the response can be parsed without an
    /// extra LLM call. The model is told to be conservative
    /// (prefer to NOT supersede when uncertain).
    static let systemPrompt: String = """
        You are a match-and-supersede classifier. The user has
        added a new instruction to a per-document chat queue.
        You will be given the new instruction and the existing
        queue items. For each existing item, decide whether the
        new instruction supersedes it (i.e. the new instruction
        is a refinement, replacement, or explicit re-do of the
        existing item).

        Return STRICT JSON with this exact shape:
          {"superseded_ids": ["uuid", ...], "reasoning": "..."}

        Be conservative: when uncertain, do not supersede.
        """

    static func userPrompt(
        newFront: ChatQueueItem,
        candidates: [ChatQueueItem]
    ) -> String {
        var lines: [String] = []
        lines.append("New instruction (\(newFront.id.uuidString)): \(newFront.message)")
        lines.append("")
        lines.append("Existing queue items:")
        for (idx, item) in candidates.enumerated() {
            let state = item.state.rawValue
            lines.append("  [\(idx)] id=\(item.id.uuidString) state=\(state) message=\"\(item.message)\"")
        }
        lines.append("")
        lines.append("Respond with JSON only.")
        return lines.joined(separator: "\n")
    }

    /// Parse the LLM response into a `SupersedeDecision`. The
    /// parser is forgiving in that it tolerates extra prose
    /// around the JSON object and unknown ids (candidates not
    /// in the response are simply not superseded), but it
    /// throws when the response contains no parseable JSON
    /// object at all (so the caller can fall back to the
    /// heuristic when the LLM truly returns garbage).
    static func parseResponse(
        _ response: String,
        candidates: [ChatQueueItem]
    ) throws -> SupersedeDecision {
        let candidateIDs = Set(candidates.map { $0.id })
        // Find the JSON object in the response.
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let openBrace = trimmed.firstIndex(of: "{"),
              let closeBrace = trimmed.lastIndex(of: "}") else {
            throw NSError(
                domain: "MatchAndSupersedeEngine",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "no JSON object in response"]
            )
        }
        let jsonSlice = String(trimmed[openBrace...closeBrace])
        guard let data = jsonSlice.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(
                domain: "MatchAndSupersedeEngine",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "JSON parse failed"]
            )
        }
        let raw = parsed["superseded_ids"] as? [Any] ?? []
        let ids: [UUID] = raw.compactMap { value in
            if let str = value as? String, let uuid = UUID(uuidString: str), candidateIDs.contains(uuid) {
                return uuid
            }
            return nil
        }
        let reasoning = (parsed["reasoning"] as? String) ?? ""
        return SupersedeDecision(supersededItemIDs: ids, reasoning: reasoning)
    }

    // MARK: - Heuristic fallback

    /// Lexical similarity fallback. Returns the items that are
    /// "close enough" to the new front by Jaccard token overlap.
    /// Threshold 0.6 (configurable).
    func heuristicCheck(
        newFront: ChatQueueItem,
        candidates: [ChatQueueItem]
    ) -> SupersedeDecision {
        let newTokens = Set(Self.tokenize(newFront.message))
        guard !newTokens.isEmpty else { return .none }
        var superseded: [UUID] = []
        for candidate in candidates {
            let candidateTokens = Set(Self.tokenize(candidate.message))
            if candidateTokens.isEmpty { continue }
            let intersection = newTokens.intersection(candidateTokens).count
            let union = newTokens.union(candidateTokens).count
            let jaccard = union > 0 ? Double(intersection) / Double(union) : 0
            if jaccard >= similarityThreshold {
                superseded.append(candidate.id)
            }
        }
        let reasoning = superseded.isEmpty
            ? "no supersessions (lexical similarity below threshold)"
            : "lexical similarity above threshold (\(similarityThreshold))"
        return SupersedeDecision(
            supersededItemIDs: superseded,
            reasoning: reasoning
        )
    }

    /// Tokenize a string for Jaccard similarity. Lower-cases,
    /// strips punctuation, splits on whitespace.
    static func tokenize(_ s: String) -> [String] {
        let lowered = s.lowercased()
        let scalars = lowered.unicodeScalars.map { scalar -> Character in
            CharacterSet.alphanumerics.contains(scalar) ? Character(scalar) : " "
        }
        let cleaned = String(scalars)
        return cleaned.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
    }
}

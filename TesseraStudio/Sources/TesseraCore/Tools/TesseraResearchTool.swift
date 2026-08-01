import Foundation

/// Research sub-agent tool (harness absorption K2): plan plain-language
/// sub-queries, fan out to web search, dedup by URL, curate sources with the
/// model (K5), then synthesize a cited answer under the never-fabricate
/// citation contract (K1) and verify every citation resolves to a curated
/// source. Model calls reuse the app's configured `LLMProvider` (the same path
/// as the agent loop); search goes through `TesseraWebSearch`.
public struct TesseraResearchTool: TesseraTool {
    public let name = "research"
    public let description = "Research a question on the web: plans sub-queries, runs a web search (keyless DuckDuckGo by default; SearXNG or Tavily if configured), curates sources, and writes a cited answer where every claim links to a real source. Returns the answer plus the source list."
    // Web search sends the query to an external search engine, so this egresses
    // regardless of provider and asks for approval rather than just notifying.
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "query": SchemaProperty(
                type: "string",
                description: "The research question to answer."
            ),
            "max_subqueries": SchemaProperty(
                type: "integer",
                description: "Number of plain-language sub-queries to plan. Default 3.",
                defaultValue: "3"
            ),
            "max_results": SchemaProperty(
                type: "integer",
                description: "Max search hits per sub-query. Default 5.",
                defaultValue: "5"
            ),
        ],
        required: ["query"]
    )

    private let llmProvider: any LLMProvider
    private let webSearch: TesseraWebSearch

    public init(
        llmProvider: (any LLMProvider)? = nil,
        webSearch: TesseraWebSearch? = nil
    ) {
        self.llmProvider = llmProvider ?? TesseraLLMProviderFactory.makeFromSettings()
        self.webSearch = webSearch ?? TesseraWebSearch()
    }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let query = arguments["query"]?.stringValue, !query.isEmpty else {
            return .fail("query is required")
        }
        let maxSubqueries = arguments["max_subqueries"]?.numberValue.map { max(1, Int($0)) } ?? 3
        let maxResults = arguments["max_results"]?.numberValue.map { max(1, Int($0)) } ?? 5

        // Graceful: no search key -> honest empty result, no crash.
        if let note = await webSearch.configurationNote {
            return .ok("Research could not run. \(note)", data: [
                "answer": .string(""),
                "sources": .array([]),
                "note": .string(note),
            ])
        }

        // (a) plan plain-language sub-queries (operator-banning).
        let subqueries = await planSubQueries(query: query, count: maxSubqueries)

        // (b) fan out + dedup by URL across sub-queries.
        var collected: [TesseraSearchHit] = []
        for subquery in subqueries {
            collected += await webSearch.search(query: subquery, maxResults: maxResults)
        }
        let deduped = Self.deduplicateByURL(collected)
        guard !deduped.isEmpty else {
            return .ok("Research found no sources for: \(query)", data: [
                "answer": .string(""),
                "sources": .array([]),
                "subqueries": .array(subqueries.map { .string($0) }),
            ])
        }

        // (c) curate: score relevance/credibility/currency, keep originals verbatim.
        let curated = await curate(query: query, hits: deduped)

        // (d) synthesize a cited answer under the K1 contract.
        let draft = await synthesize(query: query, sources: curated)

        // (e) verify: strip any citation whose URL is not in the curated set.
        let verified = Self.verifyCitations(answer: draft, sourceURLs: Set(curated.map(\.url)))

        // K5 landing spot: cache the cited answer with provenance + TTL. A store
        // failure must never break the research result.
        cacheReference(query: query, answer: verified.answer)

        let sourceLines = curated.map { "- \($0.title): \($0.url)" }.joined(separator: "\n")
        let output = verified.answer + "\n\nSources:\n" + sourceLines

        var data: [String: JSONValue] = [
            "answer": .string(verified.answer),
            "sources": .array(curated.map { hit in
                .object([
                    "url": .string(hit.url),
                    "title": .string(hit.title),
                    "content": .string(hit.content),
                ])
            }),
            "subqueries": .array(subqueries.map { .string($0) }),
            "stripped_citations": .number(Double(verified.stripped.count)),
        ]
        if !verified.stripped.isEmpty {
            data["stripped_urls"] = .array(verified.stripped.map { .string($0) })
        }
        return .ok(output, data: data)
    }

    // MARK: - Pipeline steps (model calls reuse the configured LLMProvider)

    private func planSubQueries(query: String, count: Int) async -> [String] {
        let user = "Question: \(query)\nReturn \(count) sub-queries."
        guard let raw = await complete(system: Self.planSystemPrompt, user: user),
              let parsed = Self.parseQueries(raw) else {
            return [query]  // degrade to the raw question
        }
        let cleaned = parsed.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        return cleaned.isEmpty ? [query] : Array(cleaned.prefix(count))
    }

    private func curate(query: String, hits: [TesseraSearchHit]) async -> [TesseraSearchHit] {
        let user = "Question: \(query)\n\nSources (JSON):\n\(Self.encodeHits(hits))"
        guard let raw = await complete(system: Self.curateSystemPrompt, user: user),
              let keep = Self.parseKeep(raw) else {
            return hits  // never drop everything on a parse/model failure
        }
        let keepSet = Set(keep.map(Self.normalizeURL))
        let curated = hits.filter { keepSet.contains(Self.normalizeURL($0.url)) }
        return curated.isEmpty ? hits : curated
    }

    private func synthesize(query: String, sources: [TesseraSearchHit]) async -> String {
        let user = "Question: \(query)\n\nSources (JSON):\n\(Self.encodeHits(sources))"
        guard let raw = await complete(system: Self.synthesizeSystemPrompt, user: user) else {
            return "The sources did not yield a grounded answer for: \(query)."
        }
        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func complete(system: String, user: String) async -> String? {
        do {
            let response = try await llmProvider.complete(
                system: system,
                messages: [LLMMessage(role: "user", content: user)],
                tools: []
            )
            return response.content
        } catch {
            return nil
        }
    }

    private func cacheReference(query: String, answer: String) {
        guard !answer.isEmpty else { return }
        let ttl = TesseraSettings.learningReferenceTTLDays
        try? TesseraReferenceKnowledgeStore().cache(query: query, content: answer, ttlDays: ttl)
    }

    // MARK: - Pure helpers (testable without network or model)

    /// Dedup hits by normalized URL, preserving first-seen order.
    static func deduplicateByURL(_ hits: [TesseraSearchHit]) -> [TesseraSearchHit] {
        var seen = Set<String>()
        var out: [TesseraSearchHit] = []
        for hit in hits where seen.insert(Self.normalizeURL(hit.url)).inserted {
            out.append(hit)
        }
        return out
    }

    /// K1 verifier: every inline markdown link `[text](url)` whose url is not in
    /// `sourceURLs` is a fabricated citation - replace it with its link text and
    /// record the url. Links that resolve are left untouched.
    static func verifyCitations(answer: String, sourceURLs: Set<String>) -> (answer: String, stripped: [String]) {
        let allowed = Set(sourceURLs.map(Self.normalizeURL))
        guard let regex = try? NSRegularExpression(pattern: "\\[([^\\]]*)\\]\\(([^)]*)\\)") else {
            return (answer, [])
        }
        let source = answer as NSString
        let matches = regex.matches(in: answer, range: NSRange(location: 0, length: source.length))
        var output = answer
        var stripped: [String] = []
        // Reverse order keeps earlier ranges valid while we mutate the tail.
        for match in matches.reversed() {
            guard match.numberOfRanges >= 3,
                  let fullRange = Range(match.range, in: output),
                  let textRange = Range(match.range(at: 1), in: answer),
                  let urlRange = Range(match.range(at: 2), in: answer) else { continue }
            let url = String(answer[urlRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if allowed.contains(Self.normalizeURL(url)) { continue }
            stripped.append(url)
            output.replaceSubrange(fullRange, with: String(answer[textRange]))
        }
        return (output, stripped)
    }

    /// URL match key: trimmed, trailing slashes dropped. Deliberately light so a
    /// cited url that only differs by a trailing slash still resolves.
    static func normalizeURL(_ url: String) -> String {
        var out = url.trimmingCharacters(in: .whitespacesAndNewlines)
        while out.hasSuffix("/") { out.removeLast() }
        return out
    }

    // MARK: - Prompts (plain constants; the plan prompt carries the K5 operator ban)

    static let planSystemPrompt = """
        You break a research question into plain-language search sub-queries.
        Return JSON only: {"queries": ["...", "..."]}.
        Rules:
        - Each query is short, natural language a person would type.
        - Do NOT use search operators: no site:, no filetype:, no OR, no AND, no quoted operators, no minus-exclude.
        - Cover distinct facets of the question; no near-duplicates.
        """

    static let curateSystemPrompt = """
        You curate retrieved sources for a research question.
        Score each source for relevance, credibility, and currency, then keep the ones worth citing.
        Return JSON only: {"keep": ["<url>", ...]} listing the URLs you keep.
        Do NOT summarize, rewrite, or edit the source content. Keep originals verbatim. Only select.
        """

    static let synthesizeSystemPrompt = """
        You write a grounded research answer using ONLY the provided sources.
        Citation contract (mandatory):
        - Every substantive claim carries an inline citation like ([claim text](source_url)).
        - Cite ONLY URLs present in the provided sources. Never cite a source that was not given.
        - Do NOT fill gaps from your own training knowledge. If the sources do not support a claim, omit it.
        - If the sources are insufficient, say so plainly rather than inventing content.
        """

    // MARK: - Model output parsing

    private static func encodeHits(_ hits: [TesseraSearchHit]) -> String {
        let items = hits.map { hit -> [String: String] in
            ["url": hit.url, "title": hit.title, "content": hit.content]
        }
        guard let data = try? JSONEncoder().encode(items),
              let str = String(data: data, encoding: .utf8) else {
            return "[]"
        }
        return str
    }

    /// Strip optional ```json fences, then decode. Models often wrap JSON output.
    private static func jsonPayload(_ raw: String) -> Data? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let fenceStart = text.range(of: "```") {
            text = String(text[fenceStart.upperBound...])
            if let fenceEnd = text.range(of: "```") {
                text = String(text[..<fenceEnd.lowerBound])
            }
            if text.hasPrefix("json") { text = String(text.dropFirst(4)) }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8)
    }

    private static func parseQueries(_ raw: String) -> [String]? {
        struct Plan: Decodable { let queries: [String]? }
        guard let data = jsonPayload(raw) else { return nil }
        return (try? JSONDecoder().decode(Plan.self, from: data))?.queries
    }

    private static func parseKeep(_ raw: String) -> [String]? {
        struct Curation: Decodable { let keep: [String]? }
        guard let data = jsonPayload(raw) else { return nil }
        return (try? JSONDecoder().decode(Curation.self, from: data))?.keep
    }
}

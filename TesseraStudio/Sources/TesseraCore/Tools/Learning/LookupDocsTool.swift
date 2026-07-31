import Foundation

/// Looks up cached reference docs/examples for a query. v1 reads the local
/// reference knowledge store; real web-search wiring lands in Phase 2.
public struct LookupDocsTool: TesseraTool {
    public let name = "lookup_docs"
    public let description = "Look up cached reference docs and examples for a query from the local reference knowledge store."
    public let defaultApprovalLevel = ApprovalLevel.auto

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "query": SchemaProperty(
                type: "string",
                description: "The search query to look up in the reference store."
            ),
        ],
        required: ["query"]
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let query = arguments["query"]?.stringValue, !query.isEmpty else {
            return .fail("query is required")
        }

        let center = TesseraLearningCenter.shared
        let hits = center.reference.lookup(query: query)

        // WEB PLUG-IN POINT (design Phase 2): there is no web-search/Tavily
        // surface in this codebase yet, so this tool reads the LOCAL reference
        // store only - it does not fabricate a web client. A future web step
        // would run right here when the local store misses, then cache its hits
        // into the reference store so the next lookup resolves locally:
        //
        //   if hits.isEmpty {
        //       for result in try await webSearch(query) {
        //           try center.reference.cache(query: query, content: result,
        //                                      ttlDays: TesseraSettings.learningReferenceTTLDays)
        //       }
        //       hits = center.reference.lookup(query: query)  // now local
        //   }
        //
        // No web egress happens on this path today.

        guard !hits.isEmpty else {
            return .ok("No cached reference docs for \"\(query)\".", data: ["hits": .number(0)])
        }

        // Foraging capture: a lookup hit resolved locally from the reference
        // store. Telemetry only - a store failure must not break the lookup.
        try? center.foraging.record(problemClass: query, source: .localReference, teacherIds: [])

        let body = hits.enumerated()
            .map { "[\($0.offset + 1)] \($0.element)" }
            .joined(separator: "\n\n")
        return .ok("Found \(hits.count) reference doc(s) for \"\(query)\":\n\n\(body)", data: [
            "hits": .number(Double(hits.count)),
        ])
    }
}

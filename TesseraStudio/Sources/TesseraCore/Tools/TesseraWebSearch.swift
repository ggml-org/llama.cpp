import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// One web search hit: the URL, its title, and the retrieved content snippet.
/// Content is kept verbatim - curation and synthesis happen downstream, never here.
public struct TesseraSearchHit: Codable, Sendable, Equatable {
    public let url: String
    public let title: String
    public let content: String

    public init(url: String, title: String, content: String) {
        self.url = url
        self.title = title
        self.content = content
    }
}

/// Web search facade used by the research tool. It delegates to a single
/// `TesseraSearchProvider` chosen from settings: keyless DuckDuckGo by default,
/// self-hosted SearXNG or vendor Tavily as explicit opt-ins. The public surface
/// (configurationNote, search) is unchanged so callers do not care which backend
/// is active. Foundation + URLSession only; HTML parsing lives in the providers.
public actor TesseraWebSearch {
    private let provider: any TesseraSearchProvider

    public init(provider: (any TesseraSearchProvider)? = nil) {
        self.provider = provider ?? TesseraSearchProviders.makeDefault()
    }

    /// Stable id of the active provider ("duckduckgo" | "searxng" | "tavily").
    public var providerID: String { provider.id }

    /// A clear note when the active provider cannot search; nil when configured.
    public var configurationNote: String? { provider.configurationNote }

    /// Run a search. Returns an empty array when unconfigured or on any network
    /// or decoding failure - research degrades to "no sources" rather than crashing.
    public func search(query: String, maxResults: Int = 5) async -> [TesseraSearchHit] {
        await provider.search(query: query, maxResults: maxResults)
    }
}

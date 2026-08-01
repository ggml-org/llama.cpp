import Foundation

/// Which web-search backend the agent research path uses. DuckDuckGo is the
/// keyless default; SearXNG (self-hosted) and Tavily (vendor key) are explicit
/// opt-ins. Selection lives in settings (see TesseraSettings.searchProvider).
public enum TesseraSearchProviderKind: String, CaseIterable, Sendable {
    case duckduckgo
    case searxng
    case tavily
}

/// A backend that turns a query into structured search hits. Providers are the
/// seam that keeps web search keyless-by-default: the facade picks one from
/// settings and every provider degrades to an empty result rather than crashing.
public protocol TesseraSearchProvider: Sendable {
    /// Stable identifier surfaced in results and receipts ("duckduckgo" | "searxng" | "tavily").
    var id: String { get }

    /// A clear note when the provider cannot search (e.g. missing key or base
    /// URL); nil when it is ready. Lets callers show an honest message.
    var configurationNote: String? { get }

    /// Run a search. Returns [] on any network, decoding, or configuration
    /// failure - research degrades to "no sources", it never throws.
    func search(query: String, maxResults: Int) async -> [TesseraSearchHit]
}

/// Builds the provider selected in settings. Keyless DuckDuckGo is the floor;
/// the other two only activate when the user configures them.
public enum TesseraSearchProviders {
    public static func makeDefault() -> any TesseraSearchProvider {
        switch TesseraSettings.searchProvider {
        case .duckduckgo:
            return TesseraDuckDuckGoSearch()
        case .searxng:
            return TesseraSearXNGSearch(baseURL: TesseraSettings.searxngBaseURL)
        case .tavily:
            return TesseraTavilySearch(apiKey: TesseraSettings.tavilyAPIKey)
        }
    }
}

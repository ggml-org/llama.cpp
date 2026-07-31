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

/// Tavily-backed web search service (design 5.4). Foundation + URLSession only.
///
/// The API key comes from the `TAVILY_API_KEY` environment variable by default;
/// the integrator may also pass one explicitly (proposed settings key:
/// `tessera.settings.tavilyAPIKey`). When no key is configured the service is
/// graceful: `search` returns an empty array and `configurationNote` explains
/// why, so callers never crash and can surface an honest message.
public actor TesseraWebSearch {
    private let apiKey: String
    private let baseURL: URL
    private let session: URLSession

    public init(
        apiKey: String = ProcessInfo.processInfo.environment["TAVILY_API_KEY"] ?? "",
        baseURL: URL = URL(string: "https://api.tavily.com/search")!,
        session: URLSession = .shared
    ) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.session = session
    }

    /// A clear note when the service cannot search (no API key); nil when configured.
    public var configurationNote: String? {
        guard apiKey.isEmpty else { return nil }
        return "Web search disabled: set TAVILY_API_KEY (settings key tessera.settings.tavilyAPIKey) to enable Tavily search."
    }

    /// Run a search. Returns an empty array when unconfigured or on any network
    /// or decoding failure - research degrades to "no sources" rather than crashing.
    public func search(query: String, maxResults: Int = 5) async -> [TesseraSearchHit] {
        guard !apiKey.isEmpty else { return [] }
        let trimmed = String(query.prefix(400))  // Tavily: keep queries concise
        guard !trimmed.isEmpty else { return [] }

        var request = URLRequest(url: baseURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = TavilyRequest(
            apiKey: apiKey,
            query: trimmed,
            maxResults: max(1, maxResults),
            includeAnswer: false,
            searchDepth: "basic",
            timeRange: "month"
        )
        guard let data = try? JSONEncoder().encode(body) else { return [] }
        request.httpBody = data

        let responseData: Data
        let response: URLResponse
        do {
            (responseData, response) = try await session.data(for: request)
        } catch {
            return []
        }
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            return []
        }
        guard let decoded = try? JSONDecoder().decode(TavilyResponse.self, from: responseData) else {
            return []
        }
        return (decoded.results ?? []).compactMap { hit in
            guard let url = hit.url, !url.isEmpty else { return nil }
            return TesseraSearchHit(url: url, title: hit.title ?? "", content: hit.content ?? "")
        }
    }
}

// MARK: - Wire format

private struct TavilyRequest: Encodable {
    let apiKey: String
    let query: String
    let maxResults: Int
    let includeAnswer: Bool
    let searchDepth: String
    let timeRange: String

    enum CodingKeys: String, CodingKey {
        case apiKey = "api_key"
        case query
        case maxResults = "max_results"
        case includeAnswer = "include_answer"
        case searchDepth = "search_depth"
        case timeRange = "time_range"
    }
}

private struct TavilyResponse: Decodable {
    struct Hit: Decodable {
        let url: String?
        let title: String?
        let content: String?
    }
    let results: [Hit]?
}

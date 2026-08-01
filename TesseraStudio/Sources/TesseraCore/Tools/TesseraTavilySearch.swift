import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Tavily-backed web search (design 5.4), demoted to an explicit opt-in provider.
/// This is the only provider that needs a vendor API key and logs queries with a
/// third party, so it is never the default. When no key is configured it is
/// graceful: `search` returns [] and `configurationNote` explains why.
public struct TesseraTavilySearch: TesseraSearchProvider {
    public let id = "tavily"

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

    public var configurationNote: String? {
        apiKey.isEmpty
            ? "Web search disabled: set TAVILY_API_KEY (settings key tessera.settings.tavilyAPIKey) to enable Tavily search."
            : nil
    }

    public func search(query: String, maxResults: Int) async -> [TesseraSearchHit] {
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

        do {
            let (responseData, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                return []
            }
            return Self.parseResults(data: responseData, maxResults: max(1, maxResults))
        } catch {
            return []
        }
    }

    // MARK: - Pure parsing (testable without network)

    static func parseResults(data: Data, maxResults: Int) -> [TesseraSearchHit] {
        guard let decoded = try? JSONDecoder().decode(TavilyResponse.self, from: data) else { return [] }
        let hits = (decoded.results ?? []).compactMap { hit -> TesseraSearchHit? in
            guard let url = hit.url, !url.isEmpty else { return nil }
            return TesseraSearchHit(url: url, title: hit.title ?? "", content: hit.content ?? "")
        }
        return Array(hits.prefix(maxResults))
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

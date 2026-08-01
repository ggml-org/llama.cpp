import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Keyless search against a self-hosted SearXNG instance (design: optional
/// provider). SearXNG is an open-source metasearch engine that exposes a JSON
/// API with no API key; the user runs it themselves and points this provider at
/// its base URL. No base URL configured -> graceful empty results plus a note.
public struct TesseraSearXNGSearch: TesseraSearchProvider {
    public let id = "searxng"

    private let baseURL: String
    private let session: URLSession

    public init(baseURL: String, session: URLSession = .shared) {
        var trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        while trimmed.hasSuffix("/") { trimmed.removeLast() }
        self.baseURL = trimmed
        self.session = session
    }

    public var configurationNote: String? {
        baseURL.isEmpty
            ? "Web search disabled: set a self-hosted SearXNG base URL (settings key tessera.settings.searxngBaseURL) to enable keyless SearXNG search."
            : nil
    }

    public func search(query: String, maxResults: Int) async -> [TesseraSearchHit] {
        guard !baseURL.isEmpty else { return [] }
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let endpoint = URL(string: baseURL + "/search") else { return [] }

        var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "q", value: String(trimmed.prefix(400))),
            URLQueryItem(name: "format", value: "json"),
        ]
        guard let url = components?.url else { return [] }

        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                return []
            }
            return Self.parseResults(data: data, maxResults: max(1, maxResults))
        } catch {
            return []
        }
    }

    // MARK: - Pure parsing (testable without network)

    /// Decode SearXNG's `{"results": [{url,title,content,...}]}` payload.
    static func parseResults(data: Data, maxResults: Int) -> [TesseraSearchHit] {
        struct Payload: Decodable {
            struct Item: Decodable {
                let url: String?
                let title: String?
                let content: String?
            }
            let results: [Item]?
        }
        guard let decoded = try? JSONDecoder().decode(Payload.self, from: data) else { return [] }
        let hits = (decoded.results ?? []).compactMap { item -> TesseraSearchHit? in
            guard let url = item.url, !url.isEmpty else { return nil }
            return TesseraSearchHit(url: url, title: item.title ?? "", content: item.content ?? "")
        }
        return Array(hits.prefix(maxResults))
    }
}

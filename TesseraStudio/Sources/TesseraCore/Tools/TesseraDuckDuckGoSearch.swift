import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import SwiftSoup

/// Keyless web search against DuckDuckGo's static HTML endpoint. No API key, no
/// vendor account - the default provider. URLSession fetches the results page,
/// SwiftSoup parses it into structured hits. Parsing is a pure static function
/// so it is unit-testable without any network.
public struct TesseraDuckDuckGoSearch: TesseraSearchProvider {
    public let id = "duckduckgo"

    private let session: URLSession
    private let endpoint: URL

    public init(
        session: URLSession = .shared,
        endpoint: URL = URL(string: "https://html.duckduckgo.com/html/")!
    ) {
        self.session = session
        self.endpoint = endpoint
    }

    /// Keyless: always ready, so there is never a configuration note.
    public var configurationNote: String? { nil }

    public func search(query: String, maxResults: Int) async -> [TesseraSearchHit] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "q", value: String(trimmed.prefix(400)))]
        guard let url = components?.url else { return [] }

        var request = URLRequest(url: url)
        request.setValue(Self.userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
        request.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                return []
            }
            guard let html = String(data: data, encoding: .utf8) else { return [] }
            return Self.parseResults(html: html, maxResults: max(1, maxResults))
        } catch {
            return []
        }
    }

    // MARK: - Pure parsing (testable without network)

    /// Parse a DuckDuckGo HTML results page into hits. Defensive about markup:
    /// it first reads organic `div.result` containers, then falls back to pairing
    /// `a.result__a` links with `.result__snippet` nodes by index if the container
    /// layout changes. Ads (`result--ad`) and unresolved internal redirects are
    /// dropped. Returns whatever was gathered; never throws.
    static func parseResults(html: String, maxResults: Int) -> [TesseraSearchHit] {
        var hits: [TesseraSearchHit] = []
        guard let doc = try? SwiftSoup.parse(html) else { return [] }

        if let containers = try? doc.select("div.result").array() {
            for container in containers {
                if hits.count >= maxResults { break }
                let classes = (try? container.className()) ?? ""
                if classes.contains("result--ad") { continue }
                guard let link = try? container.select("a.result__a").first() else { continue }
                let href = (try? link.attr("href")) ?? ""
                guard let url = decodeRedirectURL(href) else { continue }
                let title = (try? link.text()) ?? ""
                let snippet = (try? container.select(".result__snippet").first()?.text()) ?? ""
                hits.append(TesseraSearchHit(url: url, title: title, content: snippet))
            }
        }

        // Fallback: markup changed, pair links and snippets positionally.
        if hits.isEmpty {
            let links = (try? doc.select("a.result__a").array()) ?? []
            let snippets = (try? doc.select(".result__snippet").array()) ?? []
            for (index, link) in links.enumerated() {
                if hits.count >= maxResults { break }
                let href = (try? link.attr("href")) ?? ""
                guard let url = decodeRedirectURL(href) else { continue }
                let title = (try? link.text()) ?? ""
                let snippet = index < snippets.count ? ((try? snippets[index].text()) ?? "") : ""
                hits.append(TesseraSearchHit(url: url, title: title, content: snippet))
            }
        }

        return hits
    }

    /// DuckDuckGo wraps each result href in a redirect of the form
    /// `//duckduckgo.com/l/?uddg=<percent-encoded url>&rut=...`. Extract and
    /// decode the `uddg` value to recover the real destination. Hrefs without a
    /// redirect wrapper are resolved directly; DuckDuckGo-internal links (ads,
    /// unresolved redirects) are rejected.
    static func decodeRedirectURL(_ href: String) -> String? {
        let trimmed = href.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let range = trimmed.range(of: "uddg=") {
            let value = String(trimmed[range.upperBound...].prefix(while: { $0 != "&" }))
            guard let decoded = value.removingPercentEncoding, !decoded.isEmpty else { return nil }
            return filterInternal(decoded)
        }
        if trimmed.hasPrefix("//") {
            return filterInternal("https:" + trimmed)
        }
        if trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://") {
            return filterInternal(trimmed)
        }
        return nil
    }

    /// Drop links that still point back at DuckDuckGo (ads or unresolved
    /// redirects); a real result always lands on an external host.
    private static func filterInternal(_ url: String) -> String? {
        guard let host = URL(string: url)?.host?.lowercased() else { return nil }
        if host == "duckduckgo.com" || host.hasSuffix(".duckduckgo.com") { return nil }
        return url
    }

    // A normal desktop Safari UA; a missing or bot-like UA trips DDG's anomaly gate.
    static let userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        + "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
}

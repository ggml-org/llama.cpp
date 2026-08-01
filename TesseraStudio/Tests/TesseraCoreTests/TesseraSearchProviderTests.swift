import XCTest
@testable import TesseraCore

final class TesseraSearchProviderTests: XCTestCase {

    // MARK: - DuckDuckGo redirect decoding

    func testDecodeRedirectURLUnwrapsUddg() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
        XCTAssertEqual(TesseraDuckDuckGoSearch.decodeRedirectURL(href), "https://example.com/page")
    }

    func testDecodeRedirectURLPassesDirectLinks() {
        XCTAssertEqual(TesseraDuckDuckGoSearch.decodeRedirectURL("https://direct.example/a"), "https://direct.example/a")
        XCTAssertEqual(TesseraDuckDuckGoSearch.decodeRedirectURL("//direct.example/b"), "https://direct.example/b")
    }

    func testDecodeRedirectURLRejectsInternalAndJunk() {
        // A redirect that never resolved still points at DuckDuckGo.
        XCTAssertNil(TesseraDuckDuckGoSearch.decodeRedirectURL("//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js&rut=x"))
        XCTAssertNil(TesseraDuckDuckGoSearch.decodeRedirectURL(""))
        XCTAssertNil(TesseraDuckDuckGoSearch.decodeRedirectURL("javascript:void(0)"))
    }

    // MARK: - DuckDuckGo HTML parsing

    private let ddgHTML = """
        <html><body>
        <div class="result results_links results_links_deep web-result">
          <div class="links_main links_deep result__body">
            <h2 class="result__title">
              <a rel="noopener" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fswift.org%2F&amp;rut=abc123">Swift.org</a>
            </h2>
            <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fswift.org%2F">Swift is a general-purpose programming language.</a>
          </div>
        </div>
        <div class="result results_links results_links_deep web-result">
          <div class="result__body">
            <h2 class="result__title">
              <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.apple.com%2Fswift%2F&amp;rut=def456">Swift - Apple</a>
            </h2>
            <div class="result__snippet">Swift is a powerful and intuitive programming language.</div>
          </div>
        </div>
        <div class="result result--ad">
          <div class="result__body">
            <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fad.example%2F&amp;rut=x">Sponsored</a></h2>
            <a class="result__snippet">An ad snippet.</a>
          </div>
        </div>
        </body></html>
        """

    func testParseDuckDuckGoExtractsOrganicHitsAndDropsAds() {
        let hits = TesseraDuckDuckGoSearch.parseResults(html: ddgHTML, maxResults: 10)
        XCTAssertEqual(hits.count, 2)
        XCTAssertEqual(hits[0].url, "https://swift.org/")
        XCTAssertEqual(hits[0].title, "Swift.org")
        XCTAssertEqual(hits[0].content, "Swift is a general-purpose programming language.")
        XCTAssertEqual(hits[1].url, "https://www.apple.com/swift/")
        XCTAssertEqual(hits[1].title, "Swift - Apple")
        XCTAssertFalse(hits.contains { $0.url.contains("ad.example") })
    }

    func testParseDuckDuckGoRespectsMaxResults() {
        XCTAssertEqual(TesseraDuckDuckGoSearch.parseResults(html: ddgHTML, maxResults: 1).count, 1)
    }

    func testParseDuckDuckGoAnomalyPageYieldsNothing() {
        let anomaly = "<html><body><div class='anomaly'>Please solve the challenge to continue.</div></body></html>"
        XCTAssertTrue(TesseraDuckDuckGoSearch.parseResults(html: anomaly, maxResults: 5).isEmpty)
    }

    // MARK: - SearXNG JSON parsing

    func testParseSearXNGDropsEmptyURLAndCapsResults() {
        let json = """
            {"results":[
              {"url":"https://a.example","title":"A","content":"aa"},
              {"url":"","title":"Empty","content":"x"},
              {"url":"https://b.example","title":"B","content":"bb"}
            ]}
            """
        let hits = TesseraSearXNGSearch.parseResults(data: Data(json.utf8), maxResults: 5)
        XCTAssertEqual(hits.map(\.url), ["https://a.example", "https://b.example"])
        XCTAssertEqual(TesseraSearXNGSearch.parseResults(data: Data(json.utf8), maxResults: 1).count, 1)
    }

    func testParseSearXNGInvalidJSONYieldsNothing() {
        XCTAssertTrue(TesseraSearXNGSearch.parseResults(data: Data("not json".utf8), maxResults: 5).isEmpty)
    }

    // MARK: - Tavily JSON parsing

    func testParseTavily() {
        let json = """
            {"results":[{"url":"https://t.example","title":"T","content":"tt"}]}
            """
        let hits = TesseraTavilySearch.parseResults(data: Data(json.utf8), maxResults: 5)
        XCTAssertEqual(hits.count, 1)
        XCTAssertEqual(hits[0].url, "https://t.example")
        XCTAssertEqual(hits[0].content, "tt")
    }

    // MARK: - Provider selection and configuration notes

    func testDefaultProviderIsKeylessDuckDuckGo() {
        // No settings written in the test process, so the factory falls back to
        // the keyless default rather than a vendor backend.
        let provider = TesseraSearchProviders.makeDefault()
        XCTAssertEqual(provider.id, "duckduckgo")
        XCTAssertNil(provider.configurationNote)
    }

    func testSearXNGConfigurationNoteTracksBaseURL() {
        XCTAssertNotNil(TesseraSearXNGSearch(baseURL: "").configurationNote)
        XCTAssertNil(TesseraSearXNGSearch(baseURL: "http://localhost:8888").configurationNote)
    }

    func testTavilyConfigurationNoteTracksKey() {
        XCTAssertNotNil(TesseraTavilySearch(apiKey: "").configurationNote)
        XCTAssertNil(TesseraTavilySearch(apiKey: "tvly-secret").configurationNote)
    }

    // MARK: - Facade

    func testFacadeExposesActiveProvider() async {
        let search = TesseraWebSearch(provider: TesseraDuckDuckGoSearch())
        let id = await search.providerID
        let note = await search.configurationNote
        XCTAssertEqual(id, "duckduckgo")
        XCTAssertNil(note)
    }

    func testFacadeUnconfiguredSearXNGSearchesEmptyWithoutNetwork() async {
        let search = TesseraWebSearch(provider: TesseraSearXNGSearch(baseURL: ""))
        let note = await search.configurationNote
        let hits = await search.search(query: "anything", maxResults: 3)
        XCTAssertNotNil(note)
        XCTAssertTrue(hits.isEmpty)
    }
}

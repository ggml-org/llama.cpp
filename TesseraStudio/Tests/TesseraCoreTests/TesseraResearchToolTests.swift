import XCTest
@testable import TesseraCore

final class TesseraResearchToolTests: XCTestCase {
    private func hit(_ url: String, title: String = "t", content: String = "c") -> TesseraSearchHit {
        TesseraSearchHit(url: url, title: title, content: content)
    }

    // MARK: - URL dedup

    func testDeduplicateByURLDropsCrossSubqueryDuplicates() {
        let hits = [
            hit("https://a.example/1", title: "first"),
            hit("https://b.example/2"),
            hit("https://a.example/1", title: "dup-of-first"),   // same URL, later sub-query
            hit("https://c.example/3"),
            hit("https://b.example/2", title: "dup-of-second"),
        ]
        let deduped = TesseraResearchTool.deduplicateByURL(hits)
        XCTAssertEqual(deduped.map(\.url), [
            "https://a.example/1",
            "https://b.example/2",
            "https://c.example/3",
        ])
        // First occurrence wins.
        XCTAssertEqual(deduped[0].title, "first")
        XCTAssertEqual(deduped[1].title, "t")
    }

    func testDeduplicateByURLIsTrailingSlashInsensitive() {
        let hits = [
            hit("https://a.example/page"),
            hit("https://a.example/page/"),
        ]
        XCTAssertEqual(TesseraResearchTool.deduplicateByURL(hits).count, 1)
    }

    func testDeduplicateEmpty() {
        XCTAssertTrue(TesseraResearchTool.deduplicateByURL([]).isEmpty)
    }

    // MARK: - Citation verifier (K1)

    func testVerifyCitationsStripsFabricatedAndKeepsValid() {
        let sources: Set<String> = ["https://good.example/article"]
        let answer = """
            The capital is Paris ([Paris is the capital](https://good.example/article)). \
            It has a population of nine million ([nine million people](https://fabricated.example/nowhere)).
            """

        let result = TesseraResearchTool.verifyCitations(answer: answer, sourceURLs: sources)

        // The fabricated citation is detected and stripped.
        XCTAssertEqual(result.stripped, ["https://fabricated.example/nowhere"])
        XCTAssertFalse(result.answer.contains("https://fabricated.example/nowhere"))
        // The valid citation survives intact.
        XCTAssertTrue(result.answer.contains("https://good.example/article"))
        // The fabricated link's readable text is preserved; only the bad URL goes.
        XCTAssertTrue(result.answer.contains("nine million people"))
        XCTAssertFalse(result.answer.contains("](https://fabricated.example/nowhere)"))
    }

    func testVerifyCitationsNoLinksIsNoOp() {
        let answer = "A plain answer with no citations."
        let result = TesseraResearchTool.verifyCitations(answer: answer, sourceURLs: ["https://x.example"])
        XCTAssertEqual(result.answer, answer)
        XCTAssertTrue(result.stripped.isEmpty)
    }

    func testVerifyCitationsAllValidStripsNothing() {
        let answer = "Claim one ([a](https://a.example)) and claim two ([b](https://b.example))."
        let result = TesseraResearchTool.verifyCitations(
            answer: answer,
            sourceURLs: ["https://a.example", "https://b.example"]
        )
        XCTAssertEqual(result.answer, answer)
        XCTAssertTrue(result.stripped.isEmpty)
    }

    // MARK: - Sub-query operator banning (K5)

    func testPlanPromptBansSearchOperators() {
        let prompt = TesseraResearchTool.planSystemPrompt
        XCTAssertTrue(prompt.contains("site:"))
        XCTAssertTrue(prompt.contains("filetype:"))
        XCTAssertTrue(prompt.contains("OR"))
        XCTAssertTrue(prompt.contains("AND"))
        XCTAssertTrue(prompt.lowercased().contains("do not use search operators"))
    }

    func testSynthesizePromptCarriesCitationContract() {
        let prompt = TesseraResearchTool.synthesizeSystemPrompt
        XCTAssertTrue(prompt.contains("([claim text](source_url))"))
        XCTAssertTrue(prompt.lowercased().contains("only"))
        XCTAssertTrue(prompt.lowercased().contains("training knowledge"))
    }
}

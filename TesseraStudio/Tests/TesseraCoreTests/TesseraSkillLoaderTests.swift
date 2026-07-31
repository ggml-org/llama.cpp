import XCTest
@testable import TesseraCore

// Skills subsystem (absorption I1) tests. Loads the committed SKILL.md
// fixtures bundled with this target, plus a bad manifest built inline in a
// temp dir. Nothing here touches the network or the learning store.

final class TesseraSkillLoaderTests: XCTestCase {
    private var fixturesSkillsURL: URL {
        Bundle.module.resourceURL!
            .appendingPathComponent("Fixtures", isDirectory: true)
            .appendingPathComponent("Skills", isDirectory: true)
    }

    func testLoadsBothFixtureSkills() {
        let loader = TesseraSkillLoader(searchDirectories: [fixturesSkillsURL])
        let names = loader.skills().map(\.name).sorted()
        XCTAssertEqual(names, ["apple-reminders", "git-status"])
    }

    func testParsesFrontmatterAndBody() throws {
        let loader = TesseraSkillLoader(searchDirectories: [fixturesSkillsURL])
        let skill = try XCTUnwrap(loader.skill(named: "apple-reminders"))

        XCTAssertEqual(skill.description, "Create, list, and complete Apple Reminders via the reminders CLI.")
        XCTAssertEqual(skill.emoji, "\u{23F0}") // alarm clock emoji, escaped to keep this file ASCII
        XCTAssertEqual(skill.supportedOSes, ["darwin"])
        XCTAssertEqual(skill.requiredBins, ["reminders"])
        XCTAssertEqual(skill.installSteps, ["brew install reminders"])

        XCTAssertTrue(skill.whenToUse.contains("Reminder"))
        XCTAssertTrue(skill.whenNotToUse.contains("calendar"))
        XCTAssertTrue(skill.setup.contains("brew install reminders"))
        XCTAssertTrue(skill.commonCommands.contains("reminders add"))
        XCTAssertTrue(skill.rawBody.contains("## When to Use"))
    }

    func testMultiOSFrontmatterParsesAsList() throws {
        let loader = TesseraSkillLoader(searchDirectories: [fixturesSkillsURL])
        let skill = try XCTUnwrap(loader.skill(named: "git-status"))
        XCTAssertEqual(skill.supportedOSes, ["darwin", "linux"])
        XCTAssertEqual(skill.requiredBins, ["git"])
    }

    func testMismatchedDirectoryNameIsSkipped() throws {
        // A folder whose name disagrees with the frontmatter name must be
        // skipped, not surfaced.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-skill-test-\(UUID().uuidString)", isDirectory: true)
        let bad = tmp.appendingPathComponent("wrong-folder-name", isDirectory: true)
        try FileManager.default.createDirectory(at: bad, withIntermediateDirectories: true)
        let markdown = """
        ---
        name: actual-name
        description: A skill whose folder does not match its frontmatter name.
        ---

        ## When to Use
        Never, in this test.
        """
        try markdown.write(to: bad.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let loader = TesseraSkillLoader(searchDirectories: [tmp])
        XCTAssertTrue(loader.skills().isEmpty, "mismatched dir/name must be skipped")
        XCTAssertNil(loader.skill(named: "actual-name"))
    }

    func testSystemPromptFragmentMatchesQuery() {
        let loader = TesseraSkillLoader(searchDirectories: [fixturesSkillsURL])

        let reminders = loader.systemPromptFragment(for: "reminders")
        XCTAssertTrue(reminders.contains("apple-reminders"))
        XCTAssertTrue(reminders.contains("## When to Use"))
        XCTAssertFalse(reminders.contains("git-status"), "git skill must not match a reminders query")

        let git = loader.systemPromptFragment(for: "git")
        XCTAssertTrue(git.contains("git-status"))
        XCTAssertFalse(git.contains("apple-reminders"))
    }

    func testSystemPromptFragmentEmptyForNoMatchOrBlankQuery() {
        let loader = TesseraSkillLoader(searchDirectories: [fixturesSkillsURL])
        XCTAssertEqual(loader.systemPromptFragment(for: "kubernetes"), "")
        XCTAssertEqual(loader.systemPromptFragment(for: "   "), "")
    }
}

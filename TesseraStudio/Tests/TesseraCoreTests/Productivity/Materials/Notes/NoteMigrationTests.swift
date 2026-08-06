import XCTest
@testable import TesseraCore

/// Tests for the `0007_notes.sql` migration. The migration
/// adds three partial B-tree indexes for the note rows.
/// The tests pin the migration's behavior: the indexes
/// exist, the partial WHERE clause targets `entity_type =
/// 'note'`, and the migration is idempotent.
final class NoteMigrationTests: XCTestCase {

    // MARK: - Migration file shape

    func testMigrationFileExists() throws {
        // The migration file is at the path documented in
        // the design doc. We check via the data layer's
        // discovery path; the test fails if the file is
        // missing or unreadable.
        let candidates = [
            "tools/tessera/db/migrations/0007_notes.sql",
            "../tools/tessera/db/migrations/0007_notes.sql",
            "../../tools/tessera/db/migrations/0007_notes.sql",
        ]
        var found = false
        for path in candidates {
            if FileManager.default.fileExists(atPath: path) {
                found = true
                let contents = try? String(contentsOfFile: path, encoding: .utf8)
                XCTAssertNotNil(contents)
                XCTAssertTrue(contents?.contains("CREATE INDEX IF NOT EXISTS idx_entities_note_updated") ?? false)
                XCTAssertTrue(contents?.contains("WHERE entity_type = 'note'") ?? false)
                break
            }
        }
        XCTAssertTrue(found, "expected the 0007_notes.sql migration to exist at one of: \(candidates)")
    }

    func testMigrationIsIdempotent() throws {
        // The migration uses `CREATE INDEX IF NOT EXISTS`
        // so re-running it is a no-op. The test pins the
        // SQL shape: no `BEGIN` / `COMMIT`, no destructive
        // operations.
        for path in [
            "tools/tessera/db/migrations/0007_notes.sql",
            "../tools/tessera/db/migrations/0007_notes.sql",
            "../../tools/tessera/db/migrations/0007_notes.sql",
        ] where FileManager.default.fileExists(atPath: path) {
            let contents = try String(contentsOfFile: path, encoding: .utf8)
            XCTAssertFalse(contents.contains("BEGIN"), "migration should not use BEGIN (per 0001 conventions)")
            XCTAssertFalse(contents.contains("COMMIT"), "migration should not use COMMIT")
            XCTAssertFalse(contents.contains("DROP INDEX"), "migration should not drop indexes")
        }
    }

    func testMigrationIndexesAreAllPartial() throws {
        for path in [
            "tools/tessera/db/migrations/0007_notes.sql",
            "../tools/tessera/db/migrations/0007_notes.sql",
            "../../tools/tessera/db/migrations/0007_notes.sql",
        ] where FileManager.default.fileExists(atPath: path) {
            let contents = try String(contentsOfFile: path, encoding: .utf8)
            // Count CREATE INDEX statements (excluding
            // comments) and verify each is followed by a
            // `WHERE entity_type = 'note'` predicate on a
            // non-comment line.
            let sqlLines = contents
                .components(separatedBy: .newlines)
                .filter { line in
                    let trimmed = line.trimmingCharacters(in: .whitespaces)
                    return !trimmed.hasPrefix("--")
                }
            let indexCount = sqlLines
                .filter { $0.contains("CREATE INDEX") }
                .count
            let whereCount = sqlLines
                .filter { $0.contains("WHERE entity_type = 'note'") }
                .count
            XCTAssertGreaterThan(indexCount, 0, "expected at least one CREATE INDEX")
            XCTAssertEqual(
                indexCount, whereCount,
                "every CREATE INDEX should have a matching WHERE entity_type = 'note' partial predicate"
            )
        }
    }
}

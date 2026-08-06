import XCTest
@testable import TesseraCore

/// Tests for ``CodeStore`` without a Postgres
/// connection. The store's no-data-layer init gives
/// us an in-memory index + a no-op receipt path; the
/// tests exercise the mutation flow, the search /
/// list, and the rename / delete paths.
final class CodeStoreTests: XCTestCase {

    // MARK: - upsert + get

    func testUpsertAddsNewFile() async throws {
        let store = CodeStore()
        let file = CodeFile(
            path: "/tmp/main.swift",
            body: "let x = 1\n"
        )
        let result = try await store.upsert(file)
        XCTAssertEqual(result.id, file.id)
        let stored = store.get(path: file.path)
        XCTAssertEqual(stored?.id, file.id)
    }

    func testUpsertReplacesExisting() async throws {
        let store = CodeStore()
        let first = CodeFile(path: "/tmp/main.swift", body: "let x = 1\n")
        _ = try await store.upsert(first)
        let second = CodeFile(
            id: first.id,
            path: "/tmp/main.swift",
            body: "let x = 2\n"
        )
        _ = try await store.upsert(second)
        let stored = store.get(path: "/tmp/main.swift")
        XCTAssertEqual(stored?.body, "let x = 2\n")
    }

    // MARK: - list + search

    func testListAllReturnsAllFiles() async throws {
        let store = CodeStore()
        _ = try await store.upsert(CodeFile(path: "/tmp/a.swift", body: ""))
        _ = try await store.upsert(CodeFile(path: "/tmp/b.py", body: ""))
        _ = try await store.upsert(CodeFile(path: "/tmp/c.json", body: ""))
        XCTAssertEqual(store.listAll().count, 3)
    }

    func testListByLanguageFilters() async throws {
        let store = CodeStore()
        _ = try await store.upsert(CodeFile(path: "/tmp/a.swift", body: ""))
        _ = try await store.upsert(CodeFile(path: "/tmp/b.py", body: ""))
        _ = try await store.upsert(CodeFile(path: "/tmp/c.swift", body: ""))
        XCTAssertEqual(store.list(language: "swift").count, 2)
        XCTAssertEqual(store.list(language: "python").count, 1)
    }

    func testSearchByPathSubstring() async throws {
        let store = CodeStore()
        _ = try await store.upsert(CodeFile(path: "/tmp/Apple.swift", body: ""))
        _ = try await store.upsert(CodeFile(path: "/tmp/Banana.swift", body: ""))
        let hits = store.search("apple")
        XCTAssertEqual(hits.count, 1)
        XCTAssertEqual(hits.first?.filename, "Apple.swift")
    }

    // MARK: - apply (mutation engine integration)

    func testApplyReplacesBody() async throws {
        let store = CodeStore()
        let file = CodeFile(path: "/tmp/main.swift", body: "let x = 1\n")
        _ = try await store.upsert(file)
        let result = try await store.apply(
            .replaceCodeBlock(fileID: file.id, newBody: "let x = 99\n"),
            to: file.id
        )
        XCTAssertEqual(result.updated.body, "let x = 99\n")
        XCTAssertEqual(result.preBody, "let x = 1\n")
        // The store's index reflects the update.
        let stored = store.get(path: "/tmp/main.swift")
        XCTAssertEqual(stored?.body, "let x = 99\n")
    }

    func testApplyFailsForUnknownFile() async {
        let store = CodeStore()
        do {
            _ = try await store.apply(
                .replaceCodeBlock(fileID: UUID(), newBody: "x"),
                to: UUID()
            )
            XCTFail("expected throw")
        } catch CodeStoreError.fileNotFound {
            // expected
        } catch {
            XCTFail("expected fileNotFound, got \(error)")
        }
    }

    // MARK: - rename

    func testRenameUpdatesPath() async throws {
        let store = CodeStore()
        let file = CodeFile(path: "/tmp/old.swift", body: "x")
        _ = try await store.upsert(file)
        let renamed = try await store.rename(id: file.id, to: "/tmp/new.swift")
        XCTAssertEqual(renamed.path, "/tmp/new.swift")
        XCTAssertEqual(renamed.filename, "new.swift")
        XCTAssertNil(store.get(path: "/tmp/old.swift"))
        XCTAssertNotNil(store.get(path: "/tmp/new.swift"))
    }

    func testRenameFailsForUnknownID() async {
        let store = CodeStore()
        do {
            _ = try await store.rename(id: UUID(), to: "/tmp/x.swift")
            XCTFail("expected throw")
        } catch CodeStoreError.fileNotFound {
            // expected
        } catch {
            XCTFail("expected fileNotFound, got \(error)")
        }
    }

    // MARK: - delete

    func testDeleteRemovesFromIndex() async throws {
        let store = CodeStore()
        let file = CodeFile(path: "/tmp/main.swift", body: "x")
        _ = try await store.upsert(file)
        try await store.delete(id: file.id)
        XCTAssertNil(store.get(id: file.id))
        XCTAssertNil(store.get(path: "/tmp/main.swift"))
    }

    func testDeleteFailsForUnknownID() async {
        let store = CodeStore()
        do {
            try await store.delete(id: UUID())
            XCTFail("expected throw")
        } catch {
            // expected
        }
    }

    // MARK: - tag

    func testTagIsAddedAndRemoved() async throws {
        let store = CodeStore()
        let file = CodeFile(path: "/tmp/main.swift", body: "x")
        _ = try await store.upsert(file)
        _ = try await store.apply(
            .addTag(fileID: file.id, tag: "core"),
            to: file.id
        )
        let tagged = store.get(id: file.id)
        XCTAssertEqual(tagged?.tags, ["core"])
        _ = try await store.apply(
            .removeTag(fileID: file.id, tag: "core"),
            to: file.id
        )
        let untagged = store.get(id: file.id)
        XCTAssertEqual(untagged?.tags, [])
    }

    // MARK: - receipts (no data layer = empty)

    func testReceiptsWithNoDataLayerReturnsEmpty() async throws {
        let store = CodeStore()
        let file = CodeFile(path: "/tmp/main.swift", body: "x")
        _ = try await store.upsert(file)
        let receipts = try await store.receipts(forFile: file.id)
        XCTAssertEqual(receipts.count, 0)
    }
}

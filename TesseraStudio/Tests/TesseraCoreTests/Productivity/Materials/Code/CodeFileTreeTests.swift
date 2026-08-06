import XCTest
@testable import TesseraCore

/// Tests for ``CodeFileTreeBuilder``. The builder is
/// pure (no I/O); the tests construct `CodeFile`s by
/// hand and assert the produced tree's structure.
final class CodeFileTreeTests: XCTestCase {

    private let builder = CodeFileTreeBuilder()

    // MARK: - Empty

    func testEmptyFileSetProducesRootOnly() {
        let root = URL(fileURLWithPath: "/tmp/empty")
        let tree = builder.build(root: root, files: [])
        XCTAssertEqual(tree.root.id, root.standardizedFileURL.path)
        XCTAssertTrue(tree.root.children?.isEmpty ?? true)
    }

    // MARK: - Single file

    func testSingleFileProducesLeaf() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let file = CodeFile(path: "/tmp/project/main.swift", body: "print(1)")
        let tree = builder.build(root: root, files: [file])
        let children = tree.root.children ?? []
        XCTAssertEqual(children.count, 1)
        XCTAssertEqual(children[0].name, "main.swift")
        XCTAssertTrue(children[0].isFile)
        XCTAssertEqual(children[0].file?.id, file.id)
    }

    // MARK: - Nested directories

    func testNestedDirectoriesArePreserved() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let file = CodeFile(path: "/tmp/project/Sources/Foo.swift", body: "")
        let tree = builder.build(root: root, files: [file])
        let children = tree.root.children ?? []
        XCTAssertEqual(children.count, 1)
        let sourcesDir = children[0]
        XCTAssertEqual(sourcesDir.name, "Sources")
        XCTAssertTrue(sourcesDir.isDirectory)
        let grandchildren = sourcesDir.children ?? []
        XCTAssertEqual(grandchildren.count, 1)
        XCTAssertEqual(grandchildren[0].name, "Foo.swift")
    }

    func testDeeplyNestedDirectoriesArePreserved() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let file = CodeFile(path: "/tmp/project/a/b/c/d/foo.swift", body: "")
        let tree = builder.build(root: root, files: [file])
        let children = tree.root.children ?? []
        XCTAssertEqual(children.count, 1)
        let a = children[0]
        XCTAssertEqual(a.name, "a")
        let b = a.children?.first
        XCTAssertEqual(b?.name, "b")
        let c = b?.children?.first
        XCTAssertEqual(c?.name, "c")
        let d = c?.children?.first
        XCTAssertEqual(d?.name, "d")
        let foo = d?.children?.first
        XCTAssertEqual(foo?.name, "foo.swift")
    }

    // MARK: - Sort order

    func testDirectoriesBeforeFiles() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let files = [
            CodeFile(path: "/tmp/project/zzz.swift", body: ""),
            CodeFile(path: "/tmp/project/aaa.swift", body: ""),
            CodeFile(path: "/tmp/project/Sub/file.swift", body: ""),
        ]
        let tree = builder.build(root: root, files: files)
        let children = tree.root.children ?? []
        // Sub comes first (directory), then files sorted
        // case-insensitively.
        XCTAssertEqual(children[0].name, "Sub")
        XCTAssertEqual(children[1].name, "aaa.swift")
        XCTAssertEqual(children[2].name, "zzz.swift")
    }

    func testFilesSortedCaseInsensitively() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let files = [
            CodeFile(path: "/tmp/project/Zebra.swift", body: ""),
            CodeFile(path: "/tmp/project/apple.swift", body: ""),
            CodeFile(path: "/tmp/project/Banana.swift", body: ""),
        ]
        let tree = builder.build(root: root, files: files)
        let children = tree.root.children ?? []
        XCTAssertEqual(children.map(\.name), ["apple.swift", "Banana.swift", "Zebra.swift"])
    }

    // MARK: - Stable IDs

    func testIDsAreStableAcrossRebuilds() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let file = CodeFile(path: "/tmp/project/main.swift", body: "")
        let tree1 = builder.build(root: root, files: [file])
        let tree2 = builder.build(root: root, files: [file])
        let id1 = tree1.root.children?.first?.id
        let id2 = tree2.root.children?.first?.id
        XCTAssertEqual(id1, id2)
    }

    // MARK: - Tree helpers

    func testFlattenedReturnsDepthFirstOrder() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let files = [
            CodeFile(path: "/tmp/project/main.swift", body: ""),
            CodeFile(path: "/tmp/project/Sources/Foo.swift", body: ""),
        ]
        let tree = builder.build(root: root, files: files)
        let flat = tree.flattened()
        XCTAssertEqual(flat.first?.id, tree.root.id)
        // After the root, the order is depth-first.
        let names = flat.dropFirst().map(\.name)
        XCTAssertTrue(names.contains("main.swift"))
        XCTAssertTrue(names.contains("Sources"))
        XCTAssertTrue(names.contains("Foo.swift"))
    }

    func testNodeLookupByID() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let file = CodeFile(path: "/tmp/project/main.swift", body: "")
        let tree = builder.build(root: root, files: [file])
        let node = tree.node(withID: file.path)
        XCTAssertEqual(node?.file?.id, file.id)
    }

    func testNodeLookupReturnsNilForMissingID() {
        let root = URL(fileURLWithPath: "/tmp/project")
        let tree = builder.build(root: root, files: [])
        XCTAssertNil(tree.node(withID: "/does/not/exist"))
    }

    // MARK: - Icon names

    func testSwiftIcon() {
        let file = CodeFile(path: "/tmp/x.swift", body: "")
        let node = CodeFileTreeNode(
            id: "/tmp/x.swift", relativePath: "x.swift", name: "x.swift",
            isDirectory: false, file: file, children: nil, depth: 0
        )
        XCTAssertEqual(node.iconName, "swift")
    }

    func testPythonIcon() {
        let file = CodeFile(path: "/tmp/x.py", body: "")
        let node = CodeFileTreeNode(
            id: "/tmp/x.py", relativePath: "x.py", name: "x.py",
            isDirectory: false, file: file, children: nil, depth: 0
        )
        XCTAssertEqual(node.iconName, "chevron.left.forwardslash.chevron.right")
    }

    func testDirectoryIcon() {
        let node = CodeFileTreeNode(
            id: "/tmp/d", relativePath: "d", name: "d",
            isDirectory: true, children: [], depth: 0
        )
        XCTAssertEqual(node.iconName, "folder")
    }
}

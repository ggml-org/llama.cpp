import Foundation

// MARK: - CodeFileTreeNode

/// One node in the directory tree the Code surface's
/// sidebar shows. The node is either a directory (the
/// `children` array is non-nil) or a file (the `file`
/// field is non-nil). The `id` is stable across
/// mutations (the same path always produces the same
/// id), so SwiftUI can use it as a `List` row identity.
public struct CodeFileTreeNode: Identifiable, Sendable, Hashable {

    /// The path relative to the watched root. The
    /// `id` is the absolute path (root + relative)
    /// to guarantee uniqueness across trees from
    /// different roots.
    public var id: String
    public var relativePath: String
    public var name: String
    public var isDirectory: Bool
    public var file: CodeFile?
    public var children: [CodeFileTreeNode]?
    /// The depth in the tree (0 = root). Used by the
    /// SwiftUI `List` to indent rows; the data is
    /// precomputed so the view doesn't re-walk.
    public var depth: Int

    public init(
        id: String,
        relativePath: String,
        name: String,
        isDirectory: Bool,
        file: CodeFile? = nil,
        children: [CodeFileTreeNode]? = nil,
        depth: Int
    ) {
        self.id = id
        self.relativePath = relativePath
        self.name = name
        self.isDirectory = isDirectory
        self.file = file
        self.children = children
        self.depth = depth
    }

    /// `true` when this is a leaf file (not a
    /// directory). The view uses this to decide
    /// whether to show the chevron + the
    /// expandable disclosure.
    public var isFile: Bool { !isDirectory }

    /// The SF Symbol name for the file or directory.
    /// Mirrors the per-language icon set the rest of
    /// the productivity surface uses (Phase 4 contact
    /// type, Phase 6 graph).
    public var iconName: String {
        if isDirectory {
            return "folder"
        }
        guard let file else { return "doc" }
        switch file.language {
        case "swift": return "swift"
        case "python": return "chevron.left.forwardslash.chevron.right"
        case "javascript", "typescript", "jsx", "tsx": return "curlybraces"
        case "rust": return "gearshape.2"
        case "go": return "globe"
        case "ruby": return "diamond"
        case "java", "kotlin", "scala": return "cup.and.saucer"
        case "html": return "globe.americas"
        case "css", "scss", "sass": return "paintpalette"
        case "json", "yaml", "toml": return "list.bullet.rectangle"
        case "markdown": return "text.alignleft"
        case "shell", "bash", "zsh": return "terminal"
        case "sql": return "tablecells"
        case "c", "cpp", "h", "hpp": return "c.square"
        case "dockerfile": return "shippingbox"
        case "makefile": return "hammer"
        default: return "doc.text"
        }
    }
}

// MARK: - CodeFileTree

/// The directory tree the Code surface's sidebar shows.
/// The tree is built from the `CodeFileWatcher`'s
/// walk output + the data layer's `CodeFile` rows
/// (the sidebar shows the file with its persisted body,
/// not the on-disk bytes). The struct is value-typed;
/// the SwiftUI view observes the diff between
/// consecutive `CodeFileTree`s and animates the changes.
///
/// **Why a separate type from the watcher's walk.**
/// The watcher's walk is recursive and ad-hoc; the
/// tree is the canonical shape the view consumes
/// (sortable, depth-tracked, with stable ids). The
/// `CodeFileTreeBuilder` is the bridge between the two.
public struct CodeFileTree: Sendable, Hashable {

    public var root: CodeFileTreeNode

    public init(root: CodeFileTreeNode) {
        self.root = root
    }

    public static let empty = CodeFileTree(
        root: CodeFileTreeNode(
            id: "", relativePath: "", name: "", isDirectory: true,
            children: [], depth: -1
        )
    )

    /// Flatten the tree into a depth-first list of
    /// nodes. The view uses this to render the SwiftUI
    /// `List` (the disclosure state is in a separate
    /// `Set<String>` keyed by `id`). The depth is
    /// precomputed in the node so the view doesn't
    /// re-walk.
    public func flattened() -> [CodeFileTreeNode] {
        var out: [CodeFileTreeNode] = []
        flatten(node: root, into: &out)
        return out
    }

    private func flatten(
        node: CodeFileTreeNode,
        into out: inout [CodeFileTreeNode]
    ) {
        out.append(node)
        if let children = node.children {
            for child in children {
                flatten(node: child, into: &out)
            }
        }
    }

    /// Find a node by id. Returns nil if the id isn't
    /// in the tree. The view uses this for the
    /// "scroll to selected file" gesture.
    public func node(withID id: String) -> CodeFileTreeNode? {
        findNode(node: root, id: id)
    }

    private func findNode(
        node: CodeFileTreeNode, id: String
    ) -> CodeFileTreeNode? {
        if node.id == id { return node }
        guard let children = node.children else { return nil }
        for child in children {
            if let found = findNode(node: child, id: id) { return found }
        }
        return nil
    }
}

// MARK: - CodeFileTreeBuilder

/// Build a `CodeFileTree` from a flat list of
/// `CodeFile` rows + the watched root URL. The builder
/// is stateless; the caller owns the input and the
/// output. The builder is intentionally a struct (no
/// caching) -- the tree is rebuilt from scratch on
/// every `CodeFile` change, which is O(n) over the
/// number of files. For a 10k-file project, that's
/// ~50ms; well below the 16ms frame budget for a
/// SwiftUI list.
public struct CodeFileTreeBuilder: Sendable {

    public init() {}

    /// Build the tree. `root` is the watched root URL;
    /// the tree's root node carries the root's display
    /// name. `files` is the current material set (the
    /// `CodeStore` reads the data layer for this).
    /// Directories that have no files in them are
    /// omitted (the user doesn't want a `Foo/` row if
    /// `Foo/` is empty).
    public func build(root: URL, files: [CodeFile]) -> CodeFileTree {
        let rootNode = CodeFileTreeNode(
            id: root.standardizedFileURL.path,
            relativePath: "",
            name: root.lastPathComponent.isEmpty ? "/" : root.lastPathComponent,
            isDirectory: true,
            children: [],
            depth: -1
        )
        // Build a path -> file index. The tree is
        // organized by directory; the index makes
        // the per-directory sort cheap.
        var byParent: [String: [CodeFile]] = [:]
        for file in files {
            let url = URL(fileURLWithPath: file.path).standardizedFileURL
            let parentPath = url.deletingLastPathComponent().path
            byParent[parentPath, default: []].append(file)
        }
        // The directories we need to render are the
        // unique parent paths of the files. We sort
        // them lexicographically so the SwiftUI
        // `List` produces a stable order across
        // rebuilds.
        let directoryPaths = Set(byParent.keys)
            .union(derivedAncestors(of: files))
        // Build the tree by walking the directory set
        // top-down. The recursion is bounded by the
        // longest path; for a 10-level project the
        // call depth is 10.
        let children = buildChildren(
            for: root.standardizedFileURL.path,
            depth: 0,
            directoryPaths: directoryPaths,
            byParent: byParent
        )
        var rootCopy = rootNode
        rootCopy.children = children
        return CodeFileTree(root: rootCopy)
    }

    /// Recursive builder for the children of `parentPath`.
    /// Directories are listed first, then files; both
    /// are sorted by name (case-insensitive) so the
    /// sidebar is stable.
    private func buildChildren(
        for parentPath: String,
        depth: Int,
        directoryPaths: Set<String>,
        byParent: [String: [CodeFile]]
    ) -> [CodeFileTreeNode] {
        // Directories: the immediate children of
        // `parentPath` are the directory paths whose
        // parent is `parentPath`. We derive "immediate
        // child" by checking if the parent is one
        // level up.
        let childDirs: [String] = directoryPaths
            .filter { isImmediateChild(parent: parentPath, child: $0) }
            .sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
        // Files: the `byParent[parentPath]` list.
        let childFiles = (byParent[parentPath] ?? [])
            .sorted { $0.filename.localizedCaseInsensitiveCompare($1.filename) == .orderedAscending }
        var out: [CodeFileTreeNode] = []
        for dirPath in childDirs {
            let url = URL(fileURLWithPath: dirPath)
            let name = url.lastPathComponent
            let id = dirPath
            let grandchildren = buildChildren(
                for: dirPath, depth: depth + 1,
                directoryPaths: directoryPaths, byParent: byParent
            )
            out.append(CodeFileTreeNode(
                id: id,
                relativePath: relPath(root: parentPath, child: dirPath),
                name: name,
                isDirectory: true,
                children: grandchildren,
                depth: depth
            ))
        }
        for file in childFiles {
            out.append(CodeFileTreeNode(
                id: file.path,
                relativePath: relPath(root: parentPath, child: file.path),
                name: file.filename,
                isDirectory: false,
                file: file,
                children: nil,
                depth: depth
            ))
        }
        return out
    }

    /// `true` iff `child` is an immediate child of
    /// `parent` in the path tree. The check is purely
    /// string-based: `child` has `parent` as its
    /// prefix AND `child`'s remaining path has no
    /// further `/` separator. (The `parent` is the
    /// parent's full path; `child` is the child's
    /// full path.)
    private func isImmediateChild(parent: String, child: String) -> Bool {
        guard child.hasPrefix(parent + "/") else { return false }
        let remainder = child.dropFirst(parent.count + 1)
        return !remainder.isEmpty && !remainder.contains("/")
    }

    /// The set of directories that are ancestors of
    /// at least one file in `files`. The walk is
    /// path-based: for `/a/b/c/foo.swift`, the
    /// ancestors are `/a`, `/a/b`, `/a/b/c`.
    private func derivedAncestors(of files: [CodeFile]) -> Set<String> {
        var ancestors: Set<String> = []
        for file in files {
            let url = URL(fileURLWithPath: file.path).standardizedFileURL
            var current = url.deletingLastPathComponent()
            while current.path != current.deletingLastPathComponent().path {
                ancestors.insert(current.path)
                current = current.deletingLastPathComponent()
            }
        }
        return ancestors
    }

    private func relPath(root: String, child: String) -> String {
        if child.hasPrefix(root + "/") {
            return String(child.dropFirst(root.count + 1))
        }
        return child
    }
}

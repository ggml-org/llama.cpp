import Foundation
import CryptoKit

// MARK: - CodeFile

/// One source file as a Tessera Material. The Code surface
/// (per `docs/tessera-productivity-design.md` §12.10) treats
/// source code as a first-class material: the user can open
/// it, edit it, search it, link it to documents and contacts,
/// and have the agent propose diffs against it.
///
/// **Storage shape.** A `CodeFile` is a `graph_entity` row
/// with `entity_type = 'code'` and `subtype` = the language
/// tag (`swift`, `python`, `typescript`, ...). The `body`
/// column carries the source as JSON (a `text` field -- the
/// source itself, NOT a `Block` AST; code is plain text).
///
/// **Why plain text, not a `Block` AST?** Documents have a
/// rich structure (headings, lists, code blocks, tables). Code
/// files are a single opaque text block; introducing a
/// per-language AST would balloon the data model without
/// adding value (the agent's reasoning operates on ranges +
/// find/replace, not on a structured tree). The `Mutation`
/// extension in this file is the `replaceCodeBlock` /
/// `replaceCodeRange` / `insertCodeAt` family, which is the
/// code surface's equivalent of the document mutation API.
///
/// **Identification.** The `id` is the `graph_entities.id`
/// (a UUID). The on-disk `path` is the canonical human-readable
/// identity; the `id` is what the receipt chain + links key
/// off. Two `CodeFile`s with the same `path` are the same
/// file (a re-import upserts in place, the new receipt chain
/// appends to the existing id).
///
/// **Checksum.** `checksum` is the SHA-256 of the body. The
/// receipt's `preMutationSnapshot` carries the prior body so
/// undo restores the prior bytes; the checksum is the
/// change-detection primitive the file-system watcher uses
/// (events with an unchanged checksum are dropped).
public struct CodeFile: Codable, Sendable, Identifiable, Hashable {

    /// The graph-entity id. Stable across renames of the
    /// underlying file (the rename emits a `codeFileRenamed`
    /// receipt that updates `path`, not the id).
    public let id: UUID

    /// Absolute path on disk. The watched root is the parent
    /// of every code material the user has imported; paths
    /// outside the root are an `importError` (not a security
    /// boundary, but a data-integrity check).
    public var path: String

    /// `URL(path).lastPathComponent`, denormalized for
    /// display in the sidebar and the graph label. The
    /// `path` is the source of truth; `filename` is the
    /// cache.
    public var filename: String

    /// The detected language tag. Matches the tags
    /// `CodeBlockHighlighter` knows about so the editor's
    /// syntax highlighting reuses the same logic. The
    /// string is what `graph_entities.subtype` carries; the
    /// `Language` enum's raw values are the canonical set.
    public var language: String

    /// The source text. Stored as JSONB in
    /// `graph_entities.body` with a single `text` field
    /// (the JSON wrapper keeps the schema extensible --
    /// future "code metadata" fields attach without a
    /// migration).
    public var body: String

    /// File size in bytes. The watcher compares it to the
    /// on-disk size as a fast "did the file change at all"
    /// check before recomputing the checksum.
    public var size: Int64

    /// The file's last-modified timestamp (from
    /// `URLResourceValues.contentModificationDate`). The
    /// watcher refreshes it on every event; the receipt
    /// payload includes the mtime delta for audit.
    public var modifiedAt: Date

    /// SHA-256 of the body, formatted as
    /// `"sha256:<hex>"` (the same shape as
    /// `DocumentAST.contentHash()`). The watcher and the
    /// import path both verify the checksum after
    /// reading the file; the receipt chain is content-
    /// addressed via this field.
    public var checksum: String

    /// IDs of other graph entities this file is linked
    /// to. Documents the file implements, contacts the
    /// file's authors / maintainers, related code files.
    /// The graph view uses these as edges; the file's
    /// detail panel lists them in "Related".
    public var linkedEntityIDs: [UUID]

    /// Free-form tags. Used for grouping (`#auth`,
    /// `#core`, `#experimental`) and for the user-driven
    /// pin / archive flags. The receipt chain records tag
    /// changes as `codeFileTagged` / `codeFileUntagged`
    /// receipts.
    public var tags: [String]

    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        path: String,
        filename: String? = nil,
        language: String? = nil,
        body: String,
        size: Int64? = nil,
        modifiedAt: Date? = nil,
        checksum: String? = nil,
        linkedEntityIDs: [UUID] = [],
        tags: [String] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.path = path
        self.filename = filename ?? Self.filenameFromPath(path)
        self.language = language ?? Self.detectLanguage(forPath: path)
        self.body = body
        self.size = size ?? Int64(body.utf8.count)
        self.modifiedAt = modifiedAt ?? Date()
        self.checksum = checksum ?? Self.computeChecksum(of: body)
        self.linkedEntityIDs = linkedEntityIDs
        self.tags = tags
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    // MARK: - Constants

    /// The `graph_entities.entity_type` value. Pinned so the
    /// data layer filter and the CodeFile store agree.
    public static let entityType = "code"

    /// The subtype value for files whose language we can't
    /// detect (no matching extension). The CodeBlockHighlighter
    /// falls back to a plain monospaced render for these.
    public static let unknownLanguage = "plain"

    // MARK: - Computed

    /// Subtype persisted in `graph_entities.subtype`. Always
    /// the language string (or `unknownLanguage` for the
    /// fallback). The view uses this to pick the
    /// per-language SF Symbol icon.
    public var subtypeString: String { language }

    /// `true` when the file is a known source language
    /// (not the `plain` fallback). The editor uses this to
    /// decide whether to enable syntax highlighting.
    public var hasKnownLanguage: Bool { language != Self.unknownLanguage }

    // MARK: - Path helpers

    /// Derive the `filename` from the path. Returns the
    /// last path component; falls back to the whole path
    /// for inputs that aren't valid URLs.
    public static func filenameFromPath(_ path: String) -> String {
        let url = URL(fileURLWithPath: path)
        return url.lastPathComponent.isEmpty ? path : url.lastPathComponent
    }

    // MARK: - Language detection

    /// Map a file extension to a language tag. The mapping
    /// mirrors the language list in
    /// `docs/tessera-productivity-design.md` §12.10
    /// ("swift", "python", "typescript", "sql", "json",
    /// "yaml", "markdown", "shell", "rust", "go", ...).
    /// Extensions not in the table fall through to
    /// `unknownLanguage`.
    public static func detectLanguage(forPath path: String) -> String {
        let url = URL(fileURLWithPath: path)
        // pathExtension is lowercased by Foundation
        // (the docs don't promise this but the macOS
        // implementation does; we lowercase again to be
        // safe across platforms).
        let ext = url.pathExtension.lowercased()
        return languageForExtension(ext) ?? unknownLanguage
    }

    /// Pure extension-to-language mapping. The function is
    /// `static` so tests can call it without constructing
    /// a `CodeFile`.
    public static func languageForExtension(_ ext: String) -> String? {
        switch ext {
        // The mapping is exhaustive enough for the
        // languages the design doc calls out; unknown
        // extensions return nil.
        case "swift": return "swift"
        case "py", "pyi": return "python"
        case "js", "mjs", "cjs": return "javascript"
        case "ts", "tsx", "jsx": return "typescript"
        case "sql": return "sql"
        case "json": return "json"
        case "yaml", "yml": return "yaml"
        case "md", "markdown": return "markdown"
        case "sh", "bash", "zsh": return "shell"
        case "rs": return "rust"
        case "go": return "go"
        case "c", "h": return "c"
        case "cpp", "cc", "cxx", "hpp", "hxx": return "cpp"
        case "rb": return "ruby"
        case "java": return "java"
        case "kt", "kts": return "kotlin"
        case "scala", "sc": return "scala"
        case "hs": return "haskell"
        case "lua": return "lua"
        case "ex", "exs": return "elixir"
        case "r": return "r"
        case "m": return "matlab"
        case "dockerfile": return "dockerfile"
        case "mk", "makefile": return "makefile"
        case "toml": return "toml"
        case "xml": return "xml"
        case "html", "htm": return "html"
        case "css": return "css"
        case "scss", "sass": return "scss"
        case "vue": return "vue"
        case "svelte": return "svelte"
        case "proto": return "protobuf"
        case "graphql", "gql": return "graphql"
        case "dart": return "dart"
        case "swiftinterface": return "swift"
        default: return nil
        }
    }

    // MARK: - Checksum

    /// SHA-256 of `body`, formatted as `"sha256:<hex>"`.
    /// The `"sha256:"` prefix matches the shape
    /// `DocumentAST.contentHash()` uses so the receipt
    /// chain has one consistent format for content
    /// addressing.
    public static func computeChecksum(of body: String) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(body.utf8))
        let digest = hasher.finalize()
        return "sha256:" + digest.map { String(format: "%02x", $0) }.joined()
    }

    /// `true` when the receiver's `body` and the
    /// `expectedChecksum` agree. The watcher calls
    /// this after a file event to filter out no-op
    /// notifications (the kernel sometimes emits a
    /// `.write` event when only the atime changed).
    public func bodyMatches(checksum expectedChecksum: String) -> Bool {
        checksum == expectedChecksum
    }
}

// MARK: - JSON helpers

extension CodeFile {
    /// Encode the file to JSON. The encoded form is what
    /// the `body` column of `graph_entities` stores. Uses
    /// sorted keys + ISO-8601 dates so the bytes are
    /// deterministic (the property the receipt chain
    /// relies on for content hashing).
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Decode a file from JSON. The inverse of
    /// ``jsonData()``.
    public static func from(jsonData: Data) throws -> CodeFile {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(CodeFile.self, from: jsonData)
    }

    /// UTF-8 string form of the JSON. Convenience for the
    /// store path (the data layer deals in `String` bodies
    /// for the `graph_entities.body` JSONB column).
    func jsonDataString() throws -> String {
        let data = try jsonData()
        guard let s = String(data: data, encoding: .utf8) else {
            throw CodeFileError.invalidBody(reason: "encoded JSON is not UTF-8")
        }
        return s
    }

    /// Inverse of ``jsonDataString()``.
    static func from(jsonDataString: String) throws -> CodeFile {
        guard let data = jsonDataString.data(using: .utf8) else {
            throw CodeFileError.invalidBody(reason: "body is not valid UTF-8")
        }
        return try from(jsonData: data)
    }
}

// MARK: - Errors

public enum CodeFileError: Error, Sendable, Equatable {
    case invalidBody(reason: String)
    case invalidPath(reason: String)
    case fileNotFound(path: String)
    case unsupportedLanguage(String)
}

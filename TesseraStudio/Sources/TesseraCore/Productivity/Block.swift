import Foundation
import CryptoKit

// MARK: - AnyCodable

/// Type-erased JSON value for block attributes and document meta.
///
/// This is a type alias for ``JSONValue`` (defined in `TesseraTool.swift`):
/// `JSONValue` is already `Codable, Sendable, Equatable, Hashable` and
/// covers every value the spec's block attributes need (string, number,
/// bool, array, object, null). The spec writes `AnyCodable` to match
/// community conventions (FlightSchool's `AnyCodable` package), so we
/// expose the alias at the productivity surface to keep the API name
/// recognisable while reusing the existing implementation.
public typealias AnyCodable = JSONValue

// MARK: - BlockType

/// The set of block types the editor understands. Each case carries
/// the data shape its block stores in `attributes` / `content` /
/// `children`. The spec's section 4.1 is the source of truth.
public enum BlockType: String, Codable, Sendable, Hashable, CaseIterable {
    /// Headings h1-h6. `attributes["level"]` is `AnyCodable.number(1..6)`.
    case heading
    /// Plain prose. `content` holds the inline runs.
    case paragraph
    /// List container. `attributes["style"]` is `"unordered" | "ordered" | "task"`.
    /// `attributes["items"]` is `AnyCodable.array` of block IDs (deprecated in
    /// favour of `children` -- see spec §4.1; we keep the field for round-trip
    /// compatibility with v0 imports).
    case list
    /// A single item in a list. `content` holds the inline runs.
    case listItem
    /// Table. `attributes["rows"]`, `attributes["cols"]`, `attributes["cells"]`
    /// (a nested array of block IDs). The spec's `cells: [[BlockID]]` is the
    /// 1:1 mirror of the visible grid.
    case table
    /// One cell in a table. `content` holds the inline runs.
    case tableCell
    /// Image. `attributes["source"]` is the URL, `attributes["alt"]` the
    /// accessibility text, `width` / `height` are optional pixel hints.
    case image
    /// Code block. `attributes["language"]` is the optional language tag;
    /// `content` is a single run whose `text` holds the source.
    case codeBlock
    /// Callout (Notion-style). `attributes["emoji"]` and
    /// `attributes["color"]` are optional styling.
    case callout
    /// Horizontal divider. `content` is empty.
    case divider
    /// Block quote. `content` holds the inline runs; `attributes["cite"]`
    /// is the optional citation source.
    case quote
    /// Collapsible section. `attributes["expanded"]` is the bool state;
    /// `children` is the ordered list of contained block IDs.
    case toggle
    /// Display equation. `attributes["latex"]` is the LaTeX source.
    case equation
}

// MARK: - InlineRun

/// A contiguous run of inline content with a uniform set of
/// annotations (bold / italic / link / ...). The editor renders each
/// run as a single styled span; the agent's mutation API reasons
/// about runs at this granularity.
public struct InlineRun: Codable, Sendable, Hashable {
    /// Inline annotation tags. Some variants carry an associated value
    /// (link URL, color hex); those are encoded as a tagged JSON object
    /// via ``InlineRun/AnnotatedAnnotation`` so the on-disk shape stays
    /// human-readable.
    public enum Annotation: Codable, Sendable, Hashable {
        case bold
        case italic
        case underline
        case strikethrough
        case code
        case `subscript`
        case `superscript`
        case link(URL)
        case color(hex: String)
    }

    public var text: String
    public var annotations: [Annotation]

    public init(text: String, annotations: [Annotation] = []) {
        self.text = text
        self.annotations = annotations
    }
}

// MARK: - Block

/// One block in the document. A block is a leaf or container; the
/// `type` field drives the renderer, the `attributes` field carries
/// the type-specific data, `content` holds the inline runs for leaf
/// blocks (paragraph, listItem, tableCell, codeBlock, quote), and
/// `children` holds the ordered list of child IDs for container
/// blocks (list, toggle, table, callout).
///
/// The flat `[UUID: Block]` map + `rootChildren: [UUID]` ordered list
/// in ``DocumentAST`` is what the spec uses (matching Notion's
/// internal shape). Each block carries a `parentID` for fast upward
/// walks without re-deriving the tree.
public struct Block: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var type: BlockType
    public var attributes: [String: AnyCodable]
    public var content: [InlineRun]
    public var children: [UUID]
    public var parentID: UUID?

    public init(
        id: UUID = UUID(),
        type: BlockType,
        attributes: [String: AnyCodable] = [:],
        content: [InlineRun] = [],
        children: [UUID] = [],
        parentID: UUID? = nil
    ) {
        self.id = id
        self.type = type
        self.attributes = attributes
        self.content = content
        self.children = children
        self.parentID = parentID
    }
}

// MARK: - DocumentAST

/// The full document model. Stored in the data layer as the `body`
/// JSONB column of a `graph_entity` (entity_type = "document").
/// The `blocks` map gives O(1) lookup; `rootChildren` is the
/// ordered list of top-level block IDs.
public struct DocumentAST: Codable, Sendable, Hashable {
    public var blocks: [UUID: Block]
    public var rootChildren: [UUID]

    public init(blocks: [UUID: Block] = [:], rootChildren: [UUID] = []) {
        self.blocks = blocks
        self.rootChildren = rootChildren
    }

    /// The empty document. Useful as a starting point for new
    /// documents and for tests.
    public static let empty = DocumentAST()

    // MARK: - Codable

    /// Custom Codable: Swift's `JSONDecoder` does not decode
    /// `Dictionary<UUID, V>` from a JSON object directly (it
    /// expects an array of [key, value, key, value, ...] pairs).
    /// We use a JSON object with stringified UUID keys for
    /// round-trip-ability with standard JSON tools, and convert
    /// at the boundary.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // Decode the blocks as [String: Block] and convert to UUID keys.
        let rawBlocks = try container.decode([String: Block].self, forKey: .blocks)
        var blocks: [UUID: Block] = [:]
        for (key, value) in rawBlocks {
            guard let uuid = UUID(uuidString: key) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .blocks,
                    in: container,
                    debugDescription: "invalid UUID '\(key)' as block key"
                )
            }
            blocks[uuid] = value
        }
        self.blocks = blocks
        self.rootChildren = try container.decode([UUID].self, forKey: .rootChildren)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        // Encode as [String: Block] so the JSON shape is an object
        // with stringified UUID keys.
        let rawBlocks = blocks.reduce(into: [String: Block]()) { acc, entry in
            acc[entry.key.uuidString] = entry.value
        }
        try container.encode(rawBlocks, forKey: .blocks)
        try container.encode(rootChildren, forKey: .rootChildren)
    }

    private enum CodingKeys: String, CodingKey {
        case blocks
        case rootChildren
    }

    // MARK: Tree helpers

    /// Children of `parentID` in document order. Returns the root
    /// children when `parentID` is `nil`.
    public func children(of parentID: UUID?) -> [UUID] {
        guard let parentID else { return rootChildren }
        return blocks[parentID]?.children ?? []
    }

    /// `true` if `blockID` is present in the document.
    public func contains(_ blockID: UUID) -> Bool {
        blocks[blockID] != nil
    }

    /// Walks the tree depth-first and returns every block ID, root
    /// first. Includes the root children; the order is stable.
    public func depthFirstOrder() -> [UUID] {
        var out: [UUID] = []
        out.reserveCapacity(blocks.count)
        for root in rootChildren { walk(root, into: &out) }
        return out
    }

    private func walk(_ id: UUID, into out: inout [UUID]) {
        guard let block = blocks[id] else { return }
        out.append(id)
        for child in block.children { walk(child, into: &out) }
    }
}

// MARK: - JSON helpers

extension DocumentAST {
    /// Encode the AST to JSON. Uses a stable key ordering and ISO-8601
    /// dates so that two semantically-equal ASTs produce byte-identical
    /// JSON (the property the receipt chain relies on for content
    /// hashing). The encoder excludes `parentID` from the round-trip
    /// because the parent is implicit in the tree shape; storing it
    /// would create ambiguity if a stale value was written.
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Decode an AST from JSON. The inverse of ``jsonData()``.
    public static func from(jsonData: Data) throws -> DocumentAST {
        let decoder = JSONDecoder()
        return try decoder.decode(DocumentAST.self, from: jsonData)
    }

    /// SHA-256 content hash of the canonical JSON form. The receipt
    /// infrastructure embeds this hash in the C2PA `c2pa.hash.data`
    /// assertion. The hash is computed over ``jsonData()`` (sorted
    /// keys) so two semantically-equal ASTs produce the same hash.
    public func contentHash() throws -> String {
        let data = try jsonData()
        var hasher = SHA256()
        hasher.update(data: data)
        let digest = hasher.finalize()
        return "sha256:" + digest.map { String(format: "%02x", $0) }.joined()
    }
}

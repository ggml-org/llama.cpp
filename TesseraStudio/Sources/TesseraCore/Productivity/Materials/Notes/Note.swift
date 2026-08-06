import Foundation

// MARK: - Note

/// A first-class Material in the Tessera productivity surface — a
/// Bear-style Markdown note. Notes are stored as `graph_entity` rows
/// with `entity_type = 'note'` and `subtype = 'markdown'`, matching
/// the universal "one row per thing" pattern (see
/// `docs/tessera-data-layer-design.md` §3). The `body` column carries
/// the note's `DocumentAST` JSON (Phase 1's Block AST); `label` is the
/// display title.
///
/// Notes ride the same constitutional-receipt backbone as every
/// other material: every mutation produces a signed receipt. The
/// ``NoteStore`` is the seam that enforces that invariant for the
/// note-specific mutation vocabulary (pin, archive, tag, link).
///
/// The struct is intentionally self-contained: no references to the
/// data layer, no framework imports beyond `Foundation`. The store
/// takes a `Note` and writes it through the data layer facade.
public struct Note: Codable, Sendable, Identifiable, Hashable {

    /// Stable identifier. New notes get a fresh UUID at creation;
    /// importer / round-trip paths preserve the id so re-imports
    /// upsert in place.
    public let id: UUID

    /// Display title. The note editor auto-derives this from the
    /// first heading in the body; the user can override it via
    /// the editor. The `label` column of `graph_entities` mirrors
    /// this field (the data layer stores the title as the row's
    /// label so the graph view + search by label work without
    /// decoding the AST).
    public var title: String

    /// The Block AST (Phase 1). The note editor binds to this; the
    /// same `TesseraEditorView` that powers the Documents surface
    /// powers the note editor with `EditorMode.notes` so the
    /// toolbar promotes callouts/quotes and drops tables.
    public var body: DocumentAST

    /// Free-form tags. v1 uses explicit tag input (no `#hashtag`
    /// parsing — that's a v2 follow-up). Tags are normalized to
    /// lowercase + trimmed; duplicates are removed.
    public var tags: [String]

    /// The moment the user pinned this note, or `nil` if not
    /// pinned. The Pinned list view filters on `pinnedAt != nil`;
    /// the All list shows pinned notes with a "pinned" badge.
    public var pinnedAt: Date?

    /// The moment the user archived this note, or `nil` if not
    /// archived. Archived notes are hidden from the All + Pinned
    /// lists and only appear in the Archived list.
    public var archivedAt: Date?

    /// Other graph entities this note is linked to — documents
    /// (the article the note summarizes), contacts (people
    /// mentioned), calendar events (the meeting the note is from),
    /// tasks (action items), other notes. The linking goes through
    /// `entity_links` (Phase 1's data layer); the IDs live on the
    /// note as denormalized cache for the detail view's
    /// "related" section so we don't need a join on every render.
    public var linkedEntityIDs: [UUID]

    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        title: String = "",
        body: DocumentAST = .empty,
        tags: [String] = [],
        pinnedAt: Date? = nil,
        archivedAt: Date? = nil,
        linkedEntityIDs: [UUID] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.body = body
        self.tags = Self.normalizeTags(tags)
        self.pinnedAt = pinnedAt
        self.archivedAt = archivedAt
        self.linkedEntityIDs = linkedEntityIDs
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    // MARK: - Display

    /// The `entity_type` value persisted in `graph_entities.entity_type`.
    /// Pinned so the note view's `hybrid_search` filter and the
    /// graph view's "by type" grouping all agree.
    public static let entityType = "note"

    /// The `subtype` value persisted in `graph_entities.subtype`.
    public static let subtype = "markdown"

    public var subtypeString: String { Self.subtype }

    /// True iff this note is currently pinned. Convenience over
    /// `pinnedAt != nil` so call sites read better.
    public var isPinned: Bool { pinnedAt != nil }

    /// True iff this note is currently archived. Archived notes
    /// are hidden from the All + Pinned views.
    public var isArchived: Bool { archivedAt != nil }

    /// Display title with sensible fallbacks. The note editor
    /// auto-derives the title from the first heading in the body,
    /// but until the user types anything, this returns a placeholder
    /// ("Untitled") for the list + graph view rows.
    public var displayTitle: String {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        // Fall back to the first heading in the body, if any.
        if let firstHeading = Note.firstHeadingText(in: body), !firstHeading.isEmpty {
            return firstHeading
        }
        return "Untitled"
    }

    /// Snippet for the list view: the first `maxLength` characters
    /// of the body's plain-text representation, with markdown
    /// decoration stripped. Pure function — used by the list rows
    /// AND by the graph view's hover preview.
    public func snippet(maxLength: Int = 200) -> String {
        Note.plainTextSnippet(from: body, maxLength: maxLength)
    }

    /// Word count of the note's body, used by the focus mode
    /// status bar. Word count is the number of whitespace-separated
    /// tokens in the plain text of the body; empty AST returns 0.
    public var wordCount: Int {
        Note.wordCount(of: body)
    }

    /// Estimated reading time in whole minutes. Computed as
    /// `ceil(wordCount / 250)` — 250 words per minute is the
    /// commonly-cited average silent-reading speed (Brysbaert,
    /// 2019, "How many words do we read per minute? A review and
    /// meta-analysis of reading rate"). Returns 1 minute for any
    /// non-empty note (so a 5-word note doesn't read as 0 min).
    public var readingTimeMinutes: Int {
        let words = wordCount
        guard words > 0 else { return 0 }
        return max(1, Int((Double(words) / 250.0).rounded(.up)))
    }

    // MARK: - Tag normalization

    /// Normalize a list of tags. Lowercases, trims whitespace,
    /// drops empties, and de-duplicates while preserving the
    /// input order. Used by the init and the store's `setTags`
    /// mutation so the persisted form is always clean.
    public static func normalizeTags(_ tags: [String]) -> [String] {
        var seen: Set<String> = []
        var out: [String] = []
        for raw in tags {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard !trimmed.isEmpty else { continue }
            if seen.insert(trimmed).inserted {
                out.append(trimmed)
            }
        }
        return out
    }

    // MARK: - Plain text + heading extraction

    /// Extract the plain text of the first heading (h1-h6) in the
    /// AST. Returns nil when the body has no headings — the
    /// caller falls back to "Untitled" or the user's `title`.
    public static func firstHeadingText(in ast: DocumentAST) -> String? {
        for id in ast.rootChildren {
            guard let block = ast.blocks[id] else { continue }
            if block.type == .heading {
                let text = block.content.map { $0.text }.joined()
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        return nil
    }

    /// Convert the AST to plain text for the snippet / word-count
    /// helpers. Walks the tree depth-first, joining each block's
    /// inline runs with a space. Lists and tables get one inline
    /// run joined by "\n". Code blocks are preserved as-is (we
    /// don't substitute them with a placeholder).
    public static func plainText(of ast: DocumentAST) -> String {
        var out: [String] = []
        for id in ast.rootChildren {
            appendPlainText(blockID: id, ast: ast, into: &out)
        }
        return out.joined(separator: "\n")
    }

    private static func appendPlainText(
        blockID: UUID,
        ast: DocumentAST,
        into out: inout [String]
    ) {
        guard let block = ast.blocks[blockID] else { return }
        switch block.type {
        case .paragraph, .heading, .quote, .listItem, .tableCell:
            let text = block.content.map { $0.text }.joined()
            if !text.isEmpty { out.append(text) }
        case .codeBlock:
            let text = block.content.map { $0.text }.joined()
            if !text.isEmpty { out.append(text) }
        case .callout:
            // Render callouts as a single line: the callout's
            // emoji prefix (if any) + content.
            var line = ""
            if let emoji = block.attributes["emoji"]?.stringValue, !emoji.isEmpty {
                line += emoji + " "
            }
            line += block.content.map { $0.text }.joined()
            if !line.isEmpty { out.append(line) }
        case .list:
            for child in block.children {
                appendPlainText(blockID: child, ast: ast, into: &out)
            }
        case .table:
            for child in block.children {
                appendPlainText(blockID: child, ast: ast, into: &out)
            }
        case .toggle, .image, .divider, .equation:
            break
        }
        for child in block.children where block.type != .list && block.type != .table {
            appendPlainText(blockID: child, ast: ast, into: &out)
        }
    }

    /// Plain-text snippet capped at `maxLength` characters. The
    /// snippet collapses newlines and excess whitespace so the
    /// result fits in a single line of the list row.
    public static func plainTextSnippet(from ast: DocumentAST, maxLength: Int = 200) -> String {
        let raw = plainText(of: ast)
        let collapsed = raw
            .replacingOccurrences(of: "\n", with: " ")
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        if collapsed.count <= maxLength { return collapsed }
        let endIndex = collapsed.index(collapsed.startIndex, offsetBy: maxLength)
        return String(collapsed[..<endIndex]) + "…"
    }

    /// Word count of the AST. Splits on whitespace; empty AST
    /// returns 0.
    public static func wordCount(of ast: DocumentAST) -> Int {
        let text = plainText(of: ast)
        guard !text.isEmpty else { return 0 }
        return text
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .count
    }
}

// MARK: - JSON helpers

extension Note {
    /// Encode the note to JSON. The encoded form is what the
    /// `body` column of `graph_entities` stores. Uses sorted
    /// keys + ISO-8601 dates so the bytes are deterministic.
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Decode a note from JSON. Inverse of ``jsonData()``.
    public static func from(jsonData: Data) throws -> Note {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(Note.self, from: jsonData)
    }
}

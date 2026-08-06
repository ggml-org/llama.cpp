import Foundation

// MARK: - CodeOutlineItem

/// One entry in a file's outline (function, class, struct,
/// method, ...). The struct is what the Code surface's
/// outline panel renders; clicking the row scrolls the
/// editor to the line.
///
/// **Why regex, not an LSP.** Phase 5 ships a regex-based
/// extractor; v2 swaps in a Language Server Protocol
/// integration for proper go-to-definition + completion.
/// The regex extractor is fast (microseconds per file) and
/// covers the most common cases for the languages the
/// design doc calls out (Swift, Python, TypeScript,
/// JavaScript, Rust, Go, Ruby, Java). Languages without a
/// regex table get an empty outline + a "no outline
/// available" hint in the panel.
public struct CodeOutlineItem: Codable, Sendable, Identifiable, Hashable {

    public enum Kind: String, Codable, Sendable, Hashable, CaseIterable {
        case function
        case method
        case `class`
        case `struct`
        case `enum`
        case proto
        case `extension`
        case interface
        case namespace
        case property
        case constant
        case typealiasKind
        case macro
    }

    /// The display label. For a Swift method this is
    /// `"func foo() -> Int"`; for a class this is
    /// `"class Foo"`. The label is the matched line's
    /// stripped signature (no body).
    public var label: String

    /// The kind (function, class, ...). Drives the
    /// SF Symbol in the outline panel.
    public var kind: Kind

    /// The 1-indexed line number where the item starts.
    public var line: Int

    /// The 1-indexed line number where the item ends
    /// (exclusive for blocks). `nil` for single-line
    /// items (constants, typealiases, ...).
    public var endLine: Int?

    /// The parent outline item's id, if this is a
    /// nested declaration. The outline panel uses
    /// this to build the tree (methods inside a
    /// class). `nil` for top-level items.
    public var parentID: UUID?

    /// The depth in the outline tree (0 = top-level
    /// class/struct/function; 1 = method inside a
    /// class; ...). The view uses this to indent
    /// nested items.
    public var depth: Int

    public let id: UUID

    public init(
        id: UUID = UUID(),
        label: String,
        kind: Kind,
        line: Int,
        endLine: Int? = nil,
        parentID: UUID? = nil,
        depth: Int = 0
    ) {
        self.id = id
        self.label = label
        self.kind = kind
        self.line = line
        self.endLine = endLine
        self.parentID = parentID
        self.depth = depth
    }
}

// MARK: - CodeOutline

/// A file's outline. The struct is the result of
/// ``CodeOutlineExtractor/extract(source:language:)``;
/// the view consumes the `items` (the flat list with
/// `parentID` linkage) and the `language` (so the panel
/// shows a "no outline for `xyz`" hint when the language
/// isn't recognized).
public struct CodeOutline: Codable, Sendable, Hashable {
    public var language: String
    public var items: [CodeOutlineItem]

    public init(language: String, items: [CodeOutlineItem]) {
        self.language = language
        self.items = items
    }

    public static let empty = CodeOutline(language: "plain", items: [])

    /// `true` when the extractor had no regex table for
    /// the language and returned an empty outline.
    public var isEmpty: Bool { items.isEmpty }
}

// MARK: - CodeOutlineExtractor

/// The regex-based outline extractor. The extractor
/// holds a static set of per-language regex tables; the
/// `extract(source:language:)` method picks the right
/// table and produces a `CodeOutline`.
///
/// **Design.** A regex-based outline is inherently
/// approximate (it can't disambiguate a `function` from
/// a `function call` inside a comment, for example). The
/// extractor's job is to produce a useful-enough outline
/// for the user to navigate a 500-line file in two
/// seconds. The LSP-based v2 will produce a precise
/// outline; this regex path is the v1.
public struct CodeOutlineExtractor: Sendable {

    public init() {}

    public func extract(source: String, language: String) -> CodeOutline {
        let normalized = language.lowercased()
        let rules = Self.rules(for: normalized)
        if rules.isEmpty {
            return CodeOutline(language: normalized, items: [])
        }
        let lines = source.replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var items: [CodeOutlineItem] = []
        // The stack tracks the currently-open block
        // (the last `case .x:` we matched). The endLine
        // is filled in when a block ends (a dedent in
        // Python, a closing brace in C-like languages,
        // or `end` in Ruby).
        var openStack: [(id: UUID, kind: CodeOutlineItem.Kind, startLine: Int, indent: Int, depth: Int)] = []
        for (i, line) in lines.enumerated() {
            let lineNumber = i + 1
            let stripped = line.trimmingCharacters(in: .whitespaces)
            if stripped.isEmpty || stripped.hasPrefix("//") || stripped.hasPrefix("#") {
                continue
            }
            for rule in rules {
                guard let regex = try? NSRegularExpression(pattern: rule.pattern) else { continue }
                let range = NSRange(line.startIndex..., in: line)
                if let match = regex.firstMatch(in: line, range: range),
                   match.numberOfRanges >= 2,
                   let labelRange = Range(match.range(at: 1), in: line) {
                    let label = String(line[labelRange])
                        .trimmingCharacters(in: .whitespaces)
                    let indent = Self.leadingWhitespaceCount(in: line)
                    let parent = Self.findOpenParent(
                        in: openStack, atIndent: indent
                    )
                    let id = UUID()
                    let endLine = Self.computeEndLine(
                        for: rule.kind,
                        source: lines,
                        startingAt: lineNumber,
                        language: normalized
                    )
                    // Compute the depth as
                    // `parentDepth + 1` (or 0 for
                    // top-level items). The parent is
                    // the most recent openStack entry
                    // with a smaller indent.
                    let parentDepth: Int = parent.map { p in
                        openStack.first(where: { $0.id == p.id })?.depth ?? 0
                    } ?? -1
                    let actualDepth = parentDepth + 1
                    let item = CodeOutlineItem(
                        id: id,
                        label: label,
                        kind: rule.kind,
                        line: lineNumber,
                        endLine: endLine,
                        parentID: parent?.id,
                        depth: actualDepth
                    )
                    items.append(item)
                    if endLine != nil {
                        // Block-scoped kind; push to the
                        // stack so nested declarations
                        // attach to it.
                        openStack.append((id, rule.kind, lineNumber, indent, actualDepth))
                    }
                    break  // first rule wins
                }
            }
            // Pop the stack at line ends (best-effort
            // dedent/brace matching).
            if let top = openStack.last,
               let endLine = items.first(where: { $0.id == top.id })?.endLine,
               endLine == lineNumber + 1 {
                openStack.removeLast()
            }
        }
        return CodeOutline(language: normalized, items: items)
    }

    /// The number of leading whitespace characters in
    /// `line`. Indentation is significant for the
    /// Python extractor (a def nested in a class has
    /// a larger indent).
    static func leadingWhitespaceCount(in line: String) -> Int {
        var count = 0
        for ch in line {
            if ch == " " || ch == "\t" {
                count += 1
            } else {
                break
            }
        }
        return count
    }

    /// Find the open block that contains `indent`. The
    /// stack is searched from the top (most recent
    /// push) for the first item whose indent is less
    /// than `indent`. That item is the parent; items
    /// with a smaller indent are siblings of an
    /// ancestor.
    static func findOpenParent(
        in stack: [(id: UUID, kind: CodeOutlineItem.Kind, startLine: Int, indent: Int, depth: Int)],
        atIndent indent: Int
    ) -> (id: UUID, kind: CodeOutlineItem.Kind, startLine: Int, indent: Int, depth: Int)? {
        for item in stack.reversed() where item.indent < indent {
            return item
        }
        return nil
    }

    /// Compute the end line for a block-scoped kind.
    /// The strategy is language-specific:
    ///   * C-like (Swift, Rust, Go, Java, JS/TS) — the
    ///     end is the next line whose leading brace
    ///     count goes back to the opening level.
    ///   * Python — the end is the next line with an
    ///     indent <= the opening indent.
    ///   * Ruby — the end is the next `end` at the
    ///     opening indent.
    /// The algorithm is a single forward scan and is
    /// O(file length) per item; for typical files
    /// (a few hundred lines) this is microseconds.
    static func computeEndLine(
        for kind: CodeOutlineItem.Kind,
        source: [String],
        startingAt lineNumber: Int,
        language: String
    ) -> Int? {
        // Single-line kinds have no end.
        switch kind {
        case .property, .constant, .typealiasKind:
            return nil
        default: break
        }
        let isCLike = ["swift", "rust", "go", "java", "javascript", "typescript",
                        "c", "cpp", "kotlin", "scala"].contains(language)
        let isPython = language == "python"
        let isRuby = language == "ruby"
        // The opening line's indentation is the
        // block's indent. We use that to find the
        // end (Python) or to find the matching
        // closing brace (C-like).
        guard lineNumber >= 1, lineNumber <= source.count else { return nil }
        let openingLine = source[lineNumber - 1]
        let openingIndent = leadingWhitespaceCount(in: openingLine)
        var braceBalance: Int? = isCLike ? 0 : nil
        for i in lineNumber..<source.count {
            let line = source[i]
            let stripped = line.trimmingCharacters(in: .whitespaces)
            if stripped.isEmpty || stripped.hasPrefix("//") || stripped.hasPrefix("#") {
                continue
            }
            if isCLike {
                braceBalance = Self.updateBraceBalance(
                    line: line, current: braceBalance ?? 0
                )
                if let bal = braceBalance, bal <= 0 && i >= lineNumber {
                    return i + 1
                }
            } else if isPython {
                let indent = leadingWhitespaceCount(in: line)
                if i > lineNumber - 1 && indent <= openingIndent &&
                   !stripped.isEmpty {
                    return i
                }
            } else if isRuby {
                if i > lineNumber - 1 && stripped == "end" {
                    let indent = leadingWhitespaceCount(in: line)
                    if indent == openingIndent { return i + 1 }
                }
            }
        }
        return source.count
    }

    /// Update the brace balance for a line. Opening
    /// braces count +1, closing braces count -1; the
    /// function is brace-only (it doesn't try to
    /// strip comments or strings — the C-like
    /// languages' syntax highlighting is a
    /// separate path).
    static func updateBraceBalance(line: String, current: Int) -> Int {
        var balance = current
        for ch in line {
            if ch == "{" { balance += 1 }
            else if ch == "}" { balance -= 1 }
        }
        return balance
    }

    // MARK: - Per-language rules

    /// A regex rule for the outline extractor. The
    /// `pattern` captures the label (group 1) and the
    /// `kind` is the outline entry's kind. The regex
    /// is matched per line; multiline patterns (rare
    /// in practice) would need a different approach.
    fileprivate struct OutlineRule {
        let kind: CodeOutlineItem.Kind
        let pattern: String
    }

    /// Return the rule list for `language`. Languages
    /// without a rule list return an empty list (the
    /// outline is empty for the file).
    fileprivate static func rules(for language: String) -> [OutlineRule] {
        switch language {
        case "swift":
            return [
                OutlineRule(kind: .class, pattern: #"^\s*(?:public\s+|private\s+|fileprivate\s+|internal\s+|open\s+|final\s+)*class\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .struct, pattern: #"^\s*(?:public\s+|private\s+|fileprivate\s+|internal\s+|open\s+)*struct\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .enum, pattern: #"^\s*(?:public\s+|private\s+|fileprivate\s+|internal\s+|open\s+)*enum\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .proto, pattern: #"^\s*(?:public\s+|private\s+|fileprivate\s+|internal\s+)*protocol\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .extension, pattern: #"^\s*extension\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:public\s+|private\s+|fileprivate\s+|internal\s+|open\s+|static\s+|final\s+|override\s+|async\s+)*func\s+([A-Za-z_][A-Za-z0-9_]*\s*\(.*)"#),
                OutlineRule(kind: .property, pattern: #"^\s*(?:public\s+|private\s+|fileprivate\s+|internal\s+|open\s+|static\s+|final\s+|override\s+|lazy\s+|var\s+)+([A-Za-z_][A-Za-z0-9_]*\s*[:=])"#),
                OutlineRule(kind: .typealiasKind, pattern: #"^\s*(?:public\s+|private\s+|fileprivate\s+|internal\s+)*typealias\s+([A-Za-z_][A-Za-z0-9_]*)"#),
            ]
        case "python":
            return [
                OutlineRule(kind: .class, pattern: #"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*\s*\(.*)"#),
            ]
        case "javascript", "typescript", "jsx", "tsx":
            return [
                OutlineRule(kind: .class, pattern: #"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*\s*\(.*)"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s+)?"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s+)?[A-Za-z_$][A-Za-z0-9_$]*\s*=>"#),
            ]
        case "rust", "rs":
            return [
                OutlineRule(kind: .struct, pattern: #"^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .enum, pattern: #"^\s*(?:pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*\s*\(.*)"#),
                OutlineRule(kind: .macro, pattern: #"^\s*macro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)"#),
            ]
        case "go":
            return [
                OutlineRule(kind: .function, pattern: #"^\s*func\s+(?:\([^)]*\)\s+)?([A-Za-z_][A-Za-z0-9_]*\s*\(.*)"#),
                OutlineRule(kind: .struct, pattern: #"^\s*type\s+([A-Z][A-Za-z0-9_]*)\s+struct"#),
                OutlineRule(kind: .interface, pattern: #"^\s*type\s+([A-Z][A-Za-z0-9_]*)\s+interface"#),
            ]
        case "java":
            return [
                OutlineRule(kind: .class, pattern: #"^\s*(?:public\s+|private\s+|protected\s+)*class\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .interface, pattern: #"^\s*(?:public\s+|private\s+|protected\s+)*interface\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+|abstract\s+|synchronized\s+)*[A-Za-z_<>?\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*\s*\(.*)"#),
            ]
        case "ruby", "rb":
            return [
                OutlineRule(kind: .class, pattern: #"^\s*class\s+([A-Z][A-Za-z0-9_:]*)(?:\s+<\s+[A-Z][A-Za-z0-9_:]*)?\s*$"#),
                OutlineRule(kind: .function, pattern: #"^\s*def\s+([A-Za-z_][A-Za-z0-9_?!=]*\s*(?:\([^)]*\))?)"#),
            ]
        case "kotlin", "kt":
            return [
                OutlineRule(kind: .class, pattern: #"^\s*(?:public\s+|private\s+|internal\s+)*class\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:public\s+|private\s+|internal\s+|suspend\s+|inline\s+|override\s+)*fun\s+([A-Za-z_][A-Za-z0-9_]*\s*\(.*)"#),
            ]
        case "c", "h":
            return [
                OutlineRule(kind: .function, pattern: #"^\s*(?:static\s+|inline\s+|extern\s+)*[A-Za-z_][A-Za-z0-9_\s\*]+\s+([A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\))\s*\{"#),
                OutlineRule(kind: .struct, pattern: #"^\s*struct\s+([A-Za-z_][A-Za-z0-9_]*)"#),
            ]
        case "cpp", "cc", "cxx", "hpp", "hxx":
            return [
                OutlineRule(kind: .class, pattern: #"^\s*(?:template\s*<[^>]*>\s+)?class\s+([A-Z][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .struct, pattern: #"^\s*struct\s+([A-Za-z_][A-Za-z0-9_]*)"#),
                OutlineRule(kind: .function, pattern: #"^\s*(?:virtual\s+|static\s+|inline\s+|explicit\s+|constexpr\s+)*[A-Za-z_][A-Za-z0-9_:<>\s\*&]+\s+([A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\))\s*(?:const)?\s*\{"#),
            ]
        default:
            return []
        }
    }
}

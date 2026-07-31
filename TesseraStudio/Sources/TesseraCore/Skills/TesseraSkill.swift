import Foundation

// MARK: - Skill errors

/// Errors surfaced while parsing a `SKILL.md` manifest.
public enum TesseraSkillError: Error, LocalizedError {
    case missingFrontmatter
    case missingField(String)

    public var errorDescription: String? {
        switch self {
        case .missingFrontmatter: "SKILL.md is missing YAML frontmatter (--- ... ---)."
        case .missingField(let field): "SKILL.md frontmatter is missing required field '\(field)'."
        }
    }
}

// MARK: - Skill

/// A parsed agent skill: a `SKILL.md` manifest (YAML frontmatter plus a
/// markdown body) loaded from disk (absorption I1). The frontmatter carries
/// the machine-readable metadata; the body carries the prose the agent reads.
/// Skills are injected into the system prompt on demand - they are markdown
/// plus a loader, never a new subsystem, and never leave the machine.
public struct TesseraSkill: Sendable, Equatable {
    public let name: String
    public let description: String
    public let emoji: String?
    public let supportedOSes: [String]
    public let requiredBins: [String]
    public let installSteps: [String]
    public let whenToUse: String
    public let whenNotToUse: String
    public let setup: String
    public let commonCommands: String
    public let rawBody: String
    public let sourceURL: URL

    public init(
        name: String,
        description: String,
        emoji: String? = nil,
        supportedOSes: [String] = [],
        requiredBins: [String] = [],
        installSteps: [String] = [],
        whenToUse: String = "",
        whenNotToUse: String = "",
        setup: String = "",
        commonCommands: String = "",
        rawBody: String,
        sourceURL: URL
    ) {
        self.name = name
        self.description = description
        self.emoji = emoji
        self.supportedOSes = supportedOSes
        self.requiredBins = requiredBins
        self.installSteps = installSteps
        self.whenToUse = whenToUse
        self.whenNotToUse = whenNotToUse
        self.setup = setup
        self.commonCommands = commonCommands
        self.rawBody = rawBody
        self.sourceURL = sourceURL
    }

    /// Parse a manifest read from disk.
    public static func parse(contentsOf url: URL) throws -> TesseraSkill {
        let text = try String(contentsOf: url, encoding: .utf8)
        return try parse(markdown: text, sourceURL: url)
    }

    /// Parse a manifest from its raw markdown text.
    public static func parse(markdown: String, sourceURL: URL) throws -> TesseraSkill {
        let (frontmatter, body) = splitFrontmatter(markdown)
        guard let frontmatter else { throw TesseraSkillError.missingFrontmatter }

        let fields = FrontmatterParser.parse(frontmatter)
        guard let name = fields["name"]?.scalarValue, !name.isEmpty else {
            throw TesseraSkillError.missingField("name")
        }

        let bins = fields["requires"]?.mapValue?["bins"]?.listValue ?? []
        let sections = parseSections(body)

        return TesseraSkill(
            name: name,
            description: fields["description"]?.scalarValue ?? "",
            emoji: fields["emoji"]?.scalarValue,
            supportedOSes: fields["os"]?.listValue ?? [],
            requiredBins: bins,
            installSteps: fields["install"]?.listValue ?? [],
            whenToUse: sections.whenToUse,
            whenNotToUse: sections.whenNotToUse,
            setup: sections.setup,
            commonCommands: sections.commonCommands,
            rawBody: body,
            sourceURL: sourceURL
        )
    }

    /// Splits a manifest into its frontmatter block (between the leading
    /// `---` fences) and the markdown body that follows. A manifest without
    /// a leading fence returns nil frontmatter and the whole text as body.
    static func splitFrontmatter(_ markdown: String) -> (frontmatter: String?, body: String) {
        let normalized = markdown.replacingOccurrences(of: "\r\n", with: "\n")
        let lines = normalized.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)

        var start = 0
        while start < lines.count, lines[start].trimmingCharacters(in: .whitespaces).isEmpty {
            start += 1
        }
        guard start < lines.count, lines[start].trimmingCharacters(in: .whitespaces) == "---" else {
            return (nil, normalized)
        }

        var end = start + 1
        while end < lines.count, lines[end].trimmingCharacters(in: .whitespaces) != "---" {
            end += 1
        }
        guard end < lines.count else { return (nil, normalized) }

        let frontmatter = lines[(start + 1)..<end].joined(separator: "\n")
        let body = lines[(end + 1)...].joined(separator: "\n").trimmingCharacters(in: .newlines)
        return (frontmatter, body)
    }

    /// Extracts the four conventional body sections by their `## ` headings.
    /// Heading match is case-insensitive and ignores punctuation, so
    /// "When NOT to Use" maps to the whenNotToUse field. Unknown headings
    /// are left in rawBody but not surfaced as fields.
    static func parseSections(_ body: String) -> (whenToUse: String, whenNotToUse: String, setup: String, commonCommands: String) {
        var sections: [String: String] = [:]
        var currentKey: String?
        var buffer: [String] = []

        func flush() {
            guard let key = currentKey else { return }
            sections[key] = buffer.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        }

        for raw in body.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("## ") {
                flush()
                buffer = []
                currentKey = normalizeHeading(String(trimmed.dropFirst(3)))
            } else {
                buffer.append(String(raw))
            }
        }
        flush()

        return (
            sections["whentouse"] ?? "",
            sections["whennottouse"] ?? "",
            sections["setup"] ?? "",
            sections["commoncommands"] ?? ""
        )
    }

    private static func normalizeHeading(_ heading: String) -> String {
        heading.lowercased().filter { $0.isLetter || $0.isNumber }
    }
}

// MARK: - Frontmatter parser

/// A parsed frontmatter value: a scalar, a list of scalars, or a nested map
/// one level deep (enough for `requires.bins`).
fileprivate enum FrontmatterValue {
    case scalar(String)
    case list([String])
    case map([String: FrontmatterValue])

    var scalarValue: String? {
        if case .scalar(let value) = self { return value }
        return nil
    }

    /// Lists coerce a lone scalar to a one-element list so `os: darwin`
    /// and `os: ["darwin"]` behave the same.
    var listValue: [String] {
        switch self {
        case .list(let items): return items
        case .scalar(let value): return value.isEmpty ? [] : [value]
        case .map: return []
        }
    }

    var mapValue: [String: FrontmatterValue]? {
        if case .map(let value) = self { return value }
        return nil
    }
}

/// Minimal hand-rolled parser for the simple YAML subset used by SKILL.md
/// frontmatter: `key: value` scalars, inline `[a, b]` lists, indented
/// `- item` block lists, and one level of nesting. No external YAML library.
fileprivate enum FrontmatterParser {
    private typealias Line = (indent: Int, text: String)

    static func parse(_ text: String) -> [String: FrontmatterValue] {
        var lines: [Line] = []
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
            let indent = line.prefix(while: { $0 == " " }).count
            lines.append((indent, trimmed))
        }
        var index = 0
        return parseBlock(lines: lines, index: &index, indent: 0)
    }

    private static func parseBlock(lines: [Line], index: inout Int, indent: Int) -> [String: FrontmatterValue] {
        var map: [String: FrontmatterValue] = [:]
        while index < lines.count {
            let line = lines[index]
            if line.indent < indent { break }
            if line.indent > indent {
                // Defensive: a stray over-indented line is skipped.
                index += 1
                continue
            }
            guard let colon = line.text.firstIndex(of: ":") else {
                index += 1
                continue
            }
            let key = String(line.text[..<colon]).trimmingCharacters(in: .whitespaces)
            let rest = String(line.text[line.text.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            index += 1

            if !rest.isEmpty {
                map[key] = parseScalarOrInlineList(rest)
                continue
            }

            // Empty value: either a block list or a nested map, decided by
            // the first more-indented line that follows.
            if index < lines.count, lines[index].indent > indent {
                let childIndent = lines[index].indent
                if lines[index].text.hasPrefix("-") {
                    map[key] = parseBlockList(lines: lines, index: &index, indent: childIndent)
                } else {
                    map[key] = .map(parseBlock(lines: lines, index: &index, indent: childIndent))
                }
            } else {
                map[key] = .scalar("")
            }
        }
        return map
    }

    private static func parseBlockList(lines: [Line], index: inout Int, indent: Int) -> FrontmatterValue {
        var items: [String] = []
        while index < lines.count, lines[index].indent == indent, lines[index].text.hasPrefix("-") {
            var item = lines[index].text
            item.removeFirst()
            items.append(stripQuotes(item.trimmingCharacters(in: .whitespaces)))
            index += 1
        }
        return .list(items)
    }

    private static func parseScalarOrInlineList(_ rest: String) -> FrontmatterValue {
        if rest.hasPrefix("[") && rest.hasSuffix("]") {
            let inner = String(rest.dropFirst().dropLast())
            let items = inner
                .split(separator: ",")
                .map { stripQuotes($0.trimmingCharacters(in: .whitespaces)) }
                .filter { !$0.isEmpty }
            return .list(items)
        }
        return .scalar(stripQuotes(rest))
    }

    private static func stripQuotes(_ value: String) -> String {
        guard value.count >= 2 else { return value }
        let quoted = (value.hasPrefix("\"") && value.hasSuffix("\""))
            || (value.hasPrefix("'") && value.hasSuffix("'"))
        guard quoted else { return value }
        var out = value
        out.removeFirst()
        out.removeLast()
        return out
    }
}

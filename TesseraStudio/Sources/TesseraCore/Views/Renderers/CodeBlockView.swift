import SwiftUI

/// A syntax-highlighted code block with monospace font, line numbers,
/// a language label, and a copy button. Language comes from the fence tag.
public struct CodeBlockView: View {
    public let code: String
    public let language: String

    @State private var copied = false

    public init(code: String, language: String = "") {
        self.code = code
        self.language = language
    }

    private var lines: [String] {
        // Drop a single trailing newline so we don't render an empty last line.
        var trimmed = code
        if trimmed.hasSuffix("\n") { trimmed.removeLast() }
        return trimmed.components(separatedBy: "\n")
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView(.horizontal, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                        HStack(alignment: .top, spacing: 12) {
                            Text("\(index + 1)")
                                .foregroundStyle(.tertiary)
                                .frame(minWidth: 28, alignment: .trailing)
                            Text(CodeSyntaxHighlighter.highlight(line, language: language))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .padding(10)
            }
        }
        .font(.system(.caption, design: .monospaced))
        .background(Color(.sRGB, white: 0.1, opacity: 1).opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary))
    }

    private var header: some View {
        HStack {
            Text(language.isEmpty ? "code" : language)
                .font(.caption2.bold())
                .foregroundStyle(.secondary)
            Spacer()
            Button(action: copy) {
                Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                    .font(.caption2)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private func copy() {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        #elseif os(iOS)
        UIPasteboard.general.string = code
        #endif
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
    }
}

/// A small rule-based syntax highlighter. Covers keywords, strings,
/// comments, and numbers for common languages; falls back to plain
/// monospaced text for unknown languages (design doc 5.6).
public enum CodeSyntaxHighlighter {
    private static let keywordSets: [String: Set<String>] = [
        "swift": ["func", "let", "var", "struct", "class", "enum", "protocol", "import", "return", "if", "else", "guard", "for", "while", "switch", "case", "public", "private", "static", "self", "init", "some", "async", "await", "throws", "try"],
        "python": ["def", "class", "import", "from", "return", "if", "elif", "else", "for", "while", "with", "as", "lambda", "yield", "pass", "break", "continue", "True", "False", "None", "and", "or", "not", "in", "is"],
        "rust": ["fn", "let", "mut", "struct", "enum", "impl", "trait", "use", "pub", "return", "if", "else", "for", "while", "loop", "match", "self", "Self", "async", "await", "move", "ref", "const", "static"],
        "c": ["int", "char", "void", "float", "double", "long", "short", "unsigned", "signed", "struct", "union", "enum", "typedef", "return", "if", "else", "for", "while", "switch", "case", "break", "continue", "const", "static"],
        "go": ["func", "package", "import", "return", "if", "else", "for", "range", "switch", "case", "type", "struct", "interface", "map", "chan", "go", "defer", "var", "const", "nil", "true", "false"],
    ]

    public static func highlight(_ line: String, language: String) -> AttributedString {
        var result = AttributedString(line)
        result.foregroundColor = .primary

        let lang = language.lowercased()
        let commentPrefix = commentToken(for: lang)
        let keywords = keywordSets[lang] ?? keywordSets["c"] ?? []

        // Comments: dim everything from the comment token onward.
        if let prefix = commentPrefix, let range = line.range(of: prefix) {
            if let attrRange = attrRange(of: range, in: line, result: result) {
                result[attrRange].foregroundColor = .secondary
            }
        }

        applyPattern(#""[^"]*""#, to: line, in: &result, color: .orange)   // strings
        applyPattern(#"\b\d+(\.\d+)?\b"#, to: line, in: &result, color: .teal) // numbers

        for keyword in keywords {
            let pattern = "\\b\(NSRegularExpression.escapedPattern(for: keyword))\\b"
            applyPattern(pattern, to: line, in: &result, color: .purple)
        }

        return result
    }

    private static func commentToken(for lang: String) -> String? {
        switch lang {
        case "python", "yaml", "yml", "bash", "sh": return "#"
        case "swift", "rust", "c", "cpp", "c++", "go", "typescript", "ts", "javascript", "js", "json": return "//"
        default: return nil
        }
    }

    private static func applyPattern(_ pattern: String, to line: String, in result: inout AttributedString, color: Color) {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return }
        let nsLine = line as NSString
        for match in regex.matches(in: line, range: NSRange(location: 0, length: nsLine.length)) {
            guard let swiftRange = Range(match.range, in: line),
                  let attrRange = attrRange(of: swiftRange, in: line, result: result) else { continue }
            result[attrRange].foregroundColor = color
        }
    }

    private static func attrRange(
        of range: Range<String.Index>,
        in line: String,
        result: AttributedString
    ) -> Range<AttributedString.Index>? {
        Range<AttributedString.Index>(range, in: result)
    }
}

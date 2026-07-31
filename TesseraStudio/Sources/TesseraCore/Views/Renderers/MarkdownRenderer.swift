import SwiftUI

/// Parses and renders Markdown inside chat bubbles.
///
/// Code fences are routed to `CodeBlockView`; prose is rendered line-by-line
/// so headers, lists, and blockquotes get distinct styling (SwiftUI's
/// `AttributedString(markdown:)` handles inline bold/italic/code/links but
/// not block-level headers). See design doc 5.6.
public struct MarkdownRenderer: View {
    private let blocks: [MarkdownBlock]

    public init(_ text: String) {
        self.blocks = MarkdownBlockParser.parse(text)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .code(let language, let code):
                    CodeBlockView(code: code, language: language)
                case .prose(let lines):
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                            ProseLineView(line: line)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Blocks

enum MarkdownBlock {
    case code(language: String, code: String)
    case prose(lines: [String])
}

enum MarkdownBlockParser {
    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var proseBuffer: [String] = []
        var codeBuffer: [String] = []
        var codeLanguage = ""
        var inCode = false

        func flushProse() {
            guard !proseBuffer.isEmpty else { return }
            blocks.append(.prose(lines: proseBuffer))
            proseBuffer = []
        }

        for rawLine in text.components(separatedBy: "\n") {
            let line = rawLine
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                if inCode {
                    blocks.append(.code(language: codeLanguage, code: codeBuffer.joined(separator: "\n")))
                    codeBuffer = []
                    codeLanguage = ""
                    inCode = false
                } else {
                    flushProse()
                    let fence = line.trimmingCharacters(in: .whitespaces)
                    codeLanguage = String(fence.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                    inCode = true
                }
                continue
            }

            if inCode {
                codeBuffer.append(line)
            } else {
                proseBuffer.append(line)
            }
        }

        // An unterminated fence (streaming) is rendered as a pending code block.
        if inCode {
            blocks.append(.code(language: codeLanguage, code: codeBuffer.joined(separator: "\n")))
        }
        flushProse()
        return blocks
    }
}

// MARK: - Prose line

struct ProseLineView: View {
    let line: String

    var body: some View {
        content
    }

    @ViewBuilder
    private var content: some View {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
            Spacer().frame(height: 4)
        } else if let level = headerLevel(trimmed) {
            Text(inline(String(trimmed.drop(while: { $0 == "#" || $0 == " " }))))
                .font(headerFont(level))
                .bold()
        } else if isBullet(trimmed) {
            HStack(alignment: .top, spacing: 6) {
                Text("•")
                Text(inline(String(trimmed.dropFirst(2))))
            }
        } else if let ordered = orderedPrefix(trimmed) {
            HStack(alignment: .top, spacing: 6) {
                Text(ordered).bold()
                Text(inline(String(trimmed.dropFirst(ordered.count))))
            }
        } else if trimmed.hasPrefix("> ") {
            Text(inline(String(trimmed.dropFirst(2))))
                .italic()
                .foregroundStyle(.secondary)
                .padding(.leading, 8)
                .overlay(alignment: .leading) { Rectangle().fill(.quaternary).frame(width: 3) }
        } else {
            Text(inline(trimmed))
        }
    }

    private func inline(_ s: String) -> AttributedString {
        if let attributed = try? AttributedString(
            markdown: s,
            options: AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            return attributed
        }
        return AttributedString(s)
    }

    private func headerLevel(_ s: String) -> Int? {
        var level = 0
        for ch in s {
            if ch == "#" { level += 1 } else { break }
            if level > 6 { return nil }
        }
        guard level >= 1, level <= 6, s.count > level, s[s.index(s.startIndex, offsetBy: level)] == " " else {
            return nil
        }
        return level
    }

    private func headerFont(_ level: Int) -> Font {
        switch level {
        case 1: return .title2
        case 2: return .title3
        case 3: return .headline
        default: return .subheadline
        }
    }

    private func isBullet(_ s: String) -> Bool {
        (s.hasPrefix("- ") || s.hasPrefix("* ")) && s.count > 2
    }

    private func orderedPrefix(_ s: String) -> String? {
        // Matches "1. ", "12. " etc.
        var digits = ""
        var idx = s.startIndex
        while idx < s.endIndex, s[idx].isNumber {
            digits.append(s[idx])
            idx = s.index(after: idx)
        }
        guard !digits.isEmpty,
              idx < s.endIndex, s[idx] == ".",
              s.index(after: idx) < s.endIndex, s[s.index(after: idx)] == " " else {
            return nil
        }
        return digits + ". "
    }
}

import Foundation
#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif
#if canImport(Splash)
import Splash
#endif

// MARK: - CodeBlockHighlighter

/// Renders source code into an attributed string with syntax
/// highlighting. The highlighter composes two strategies:
///
/// 1. **Splash (JohnSundell)** for Swift — Splash is a
///    grammar-based highlighter and produces the most accurate
///    Swift highlighting we can get without a full
///    `tree-sitter` integration. We use the
///    `AttributedStringOutputFormat` to get an `NSAttributedString`
///    directly, then re-style the token colors to match the
///    editor's `SyntaxThemePalette`.
///
/// 2. **A small regex-based highlighter** for the other 9
///    languages the brief calls out (Python, JavaScript/TypeScript,
///    SQL, JSON, YAML, Markdown, Shell, Rust, Go). Splash only
///    ships a `SwiftGrammar`; the regex highlighter is the
///    pragmatic v1 path. The design doc notes the limitation
///    and the path to a real per-language grammar (either
///    Splash extensions or a `tree-sitter` integration).
///
/// **Unknown languages** fall back to a plain monospaced
/// render: a single run with the monospaced font + a subtle
/// background color. The text view still shows the source
/// as a code block; it's just not highlighted.
///
/// **Theme.** The highlighter consumes `SyntaxThemePalette` (the
/// same shape Phase 2's editor theme uses). Token colors are
/// resolved per-token; tokens the regex highlighter doesn't
/// recognize stay in the palette's `plain` color.
public struct CodeBlockHighlighter: @unchecked Sendable {

    public let theme: SyntaxThemePalette
    public let font: PlatformFont

    public init(theme: SyntaxThemePalette = .light, font: PlatformFont? = nil) {
        self.theme = theme
        if let font = font {
            self.font = font
        } else {
            #if canImport(AppKit)
            self.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
            #elseif canImport(UIKit)
            self.font = UIFont.monospacedSystemFont(ofSize: 13, weight: .regular)
            #else
            self.font = PlatformFont()
            #endif
        }
    }

    /// Highlight `source` for the given `language` tag. The
    /// language tag is matched case-insensitively against the
    /// `SupportedLanguage` cases. A `nil` or unknown language
    /// returns a plain monospaced render.
    public func highlight(source: String, language: String?) -> NSAttributedString {
        let normalized = language?.lowercased()
        switch normalized {
        case "swift":
            return highlightSwift(source)
        case "python", "py":
            return highlightRegex(source, rules: RegexRules.python)
        case "javascript", "js", "typescript", "ts", "tsx", "jsx":
            return highlightRegex(source, rules: RegexRules.javascript)
        case "sql":
            return highlightRegex(source, rules: RegexRules.sql)
        case "json":
            return highlightRegex(source, rules: RegexRules.json)
        case "yaml", "yml":
            return highlightRegex(source, rules: RegexRules.yaml)
        case "markdown", "md":
            return highlightRegex(source, rules: RegexRules.markdown)
        case "shell", "sh", "bash", "zsh":
            return highlightRegex(source, rules: RegexRules.shell)
        case "rust", "rs":
            return highlightRegex(source, rules: RegexRules.rust)
        case "go":
            return highlightRegex(source, rules: RegexRules.go)
        case .some, .none:
            return plainMonospaced(source)
        }
    }

    // MARK: - Swift (Splash)

    private func highlightSwift(_ source: String) -> NSAttributedString {
        #if canImport(Splash)
        let splashTheme = makeSplashTheme()
        let highlighter = SyntaxHighlighter(
            format: AttributedStringOutputFormat(theme: splashTheme),
            grammar: SwiftGrammar()
        )
        let attributed = highlighter.highlight(source)
        // Re-style the result with the editor's font + token
        // colors. The Splash output uses its own font; we
        // override the .font attribute on every range to the
        // editor's monospaced font.
        let mutable = NSMutableAttributedString(attributedString: attributed)
        let fullRange = NSRange(location: 0, length: mutable.length)
        mutable.addAttribute(.font, value: font, range: fullRange)
        return mutable
        #else
        return highlightRegex(source, rules: RegexRules.swiftFallback)
        #endif
    }

    #if canImport(Splash)
    private func makeSplashTheme() -> Theme {
        let splashFont = Font(size: Double(font.pointSize))
        var tokenColors: [TokenType: PlatformColor] = [:]
        tokenColors[.keyword] = PlatformColor.fromHex(theme.keyword) ?? .systemPurple
        tokenColors[.string] = PlatformColor.fromHex(theme.string) ?? .systemGreen
        tokenColors[.type] = PlatformColor.fromHex(theme.type) ?? .systemTeal
        tokenColors[.call] = PlatformColor.fromHex(theme.functionCall) ?? .systemBlue
        tokenColors[.number] = PlatformColor.fromHex(theme.number) ?? .systemOrange
        tokenColors[.comment] = PlatformColor.fromHex(theme.comment) ?? .systemGray
        tokenColors[.property] = PlatformColor.fromHex(theme.identifier) ?? .labelColor
        tokenColors[.dotAccess] = PlatformColor.fromHex(theme.operator) ?? .systemPurple
        tokenColors[.preprocessing] = PlatformColor.fromHex(theme.keyword) ?? .systemPurple
        return Theme(
            font: splashFont,
            plainTextColor: PlatformColor.fromHex(theme.plain) ?? .labelColor,
            tokenColors: tokenColors,
            backgroundColor: .clear
        )
    }
    #endif

    // MARK: - Regex (the other 9 languages)

    /// One named token rule for the regex highlighter. Each
    /// rule is a regular expression; matches are emitted as
    /// runs with the corresponding color from the palette.
    fileprivate struct TokenRule: @unchecked Sendable {
        let token: String
        let color: String
        let pattern: String
    }

    /// Render `source` by tokenizing it via the regex rules in
    /// order. The first matching rule wins for each character
    /// position. Whitespace and unmatched characters fall back
    /// to the `plain` color.
    private func highlightRegex(_ source: String, rules: [TokenRule]) -> NSAttributedString {
        let out = NSMutableAttributedString()
        // Tokenize by walking through the source and trying each
        // rule at each position. This is O(n * rules) which is
        // fine for the documents we expect (a few hundred lines
        // per code block).
        var cursor = source.startIndex
        let plainAttrs: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: PlatformColor.fromHex(theme.plain) ?? .labelColor,
        ]
        while cursor < source.endIndex {
            // Find the longest match at `cursor` from any rule.
            var best: (Range<String.Index>, TokenRule)?
            for rule in rules {
                guard let regex = try? NSRegularExpression(pattern: rule.pattern, options: []) else { continue }
                let nsCursor = utf16Offset(of: cursor, in: source)
                let searchRange = NSRange(location: nsCursor, length: source.utf16.count - nsCursor)
                if let match = regex.firstMatch(in: source, options: [], range: searchRange),
                   match.range.location == nsCursor,
                   match.range.length > 0 {
                    let start = String.Index(utf16Offset: match.range.location, in: source)
                    let end = String.Index(utf16Offset: match.range.location + match.range.length, in: source)
                    let range = start..<end
                    if best == nil || source[range].count > source[best!.0].count {
                        best = (range, rule)
                    }
                }
            }
            if let (range, rule) = best {
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: font,
                    .foregroundColor: PlatformColor.fromHex(rule.color) ?? .labelColor,
                ]
                out.append(NSAttributedString(string: String(source[range]), attributes: attrs))
                cursor = range.upperBound
            } else {
                // No rule matched at `cursor`; emit a single
                // character with the plain color and advance.
                let nextChar = source.index(after: cursor)
                out.append(NSAttributedString(string: String(source[cursor..<nextChar]), attributes: plainAttrs))
                cursor = nextChar
            }
        }
        return out
    }

    // MARK: - Plain fallback

    private func plainMonospaced(_ source: String) -> NSAttributedString {
        let attrs: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: PlatformColor.fromHex(theme.plain) ?? .labelColor,
        ]
        return NSAttributedString(string: source, attributes: attrs)
    }
}

// MARK: - Regex rules per language

extension CodeBlockHighlighter {
    /// The regex rule sets. Each language gets a small list of
    /// patterns; the patterns are intentionally simple (no
    /// full grammar) but cover the most distinctive tokens
    /// (keywords, strings, numbers, comments).
    fileprivate enum RegexRules {
        // Per-rule pattern templates. The highlighter walks
        // the source from left to right and picks the first
        // matching rule at each position.
        static let python: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "^#[^\\n]*"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "(\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?'''|\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:\\\\.|[^'\\\\\\n])*')"),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "\\b(def|class|import|from|as|return|if|elif|else|for|while|try|except|finally|with|in|is|not|and|or|None|True|False|lambda|pass|break|continue|yield|global|nonlocal|raise|async|await)\\b"),
            TokenRule(token: "number", color: SyntaxThemePalette().number, pattern: "\\b\\d+(?:\\.\\d+)?\\b"),
            TokenRule(token: "type", color: SyntaxThemePalette().type, pattern: "\\b[A-Z][A-Za-z0-9_]*\\b"),
            TokenRule(token: "functionCall", color: SyntaxThemePalette().functionCall, pattern: "\\b[a-z_][A-Za-z0-9_]*(?=\\()"),
        ]

        static let javascript: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "//[^\\n]*|/\\*[\\s\\S]*?\\*/"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "`(?:\\\\.|[^`\\\\])*`|\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:\\\\.|[^'\\\\\\n])*'"),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "\\b(const|let|var|function|class|extends|new|return|if|else|for|while|do|switch|case|break|continue|import|export|from|as|async|await|try|catch|finally|throw|typeof|instanceof|in|of|this|super|null|undefined|true|false)\\b"),
            TokenRule(token: "number", color: SyntaxThemePalette().number, pattern: "\\b\\d+(?:\\.\\d+)?\\b"),
            TokenRule(token: "type", color: SyntaxThemePalette().type, pattern: "\\b[A-Z][A-Za-z0-9_]*\\b"),
            TokenRule(token: "functionCall", color: SyntaxThemePalette().functionCall, pattern: "\\b[a-z_][A-Za-z0-9_]*(?=\\()"),
        ]

        static let sql: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "--[^\\n]*|/\\*[\\s\\S]*?\\*/"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "'(?:''|[^'])*'"),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "\\b(?i)(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP|ORDER|BY|LIMIT|OFFSET|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|INDEX|VIEW|AS|AND|OR|NOT|NULL|IS|IN|BETWEEN|LIKE|EXISTS|DISTINCT|UNION|ALL|PRIMARY|KEY|FOREIGN|REFERENCES|DEFAULT|CHECK|UNIQUE|CASCADE)\\b"),
            TokenRule(token: "number", color: SyntaxThemePalette().number, pattern: "\\b\\d+(?:\\.\\d+)?\\b"),
            TokenRule(token: "functionCall", color: SyntaxThemePalette().functionCall, pattern: "\\b[A-Z_][A-Z0-9_]*(?=\\()"),
        ]

        static let json: [TokenRule] = [
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "\"(?:\\\\.|[^\"\\\\])*\""),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "\\b(true|false|null)\\b"),
            TokenRule(token: "number", color: SyntaxThemePalette().number, pattern: "-?\\b\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b"),
        ]

        static let yaml: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "#[^\\n]*"),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "^[\\s-]*(true|false|null|yes|no|on|off)\\b"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "\"(?:\\\\.|[^\"\\\\])*\"|'(?:[^'\\\\]|\\\\.)*'"),
            TokenRule(token: "number", color: SyntaxThemePalette().number, pattern: "-?\\b\\d+(?:\\.\\d+)?\\b"),
            TokenRule(token: "identifier", color: SyntaxThemePalette().identifier, pattern: "^[\\s]*[A-Za-z_][\\w-]*(?=:)"),
        ]

        static let markdown: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "<!--[\\s\\S]*?-->"),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "^#{1,6}\\s+|^[\\s]*[-*+]\\s+|^[\\s]*\\d+\\.\\s+|^>\\s+|^---"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "`[^`]+`|```[\\s\\S]*?```"),
            TokenRule(token: "type", color: SyntaxThemePalette().type, pattern: "\\[([^\\]]+)\\]\\([^)]+\\)"),
        ]

        static let shell: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "#[^\\n]*"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:[^'\\\\\\n]|\\\\.)*'"),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "\\b(if|then|fi|else|elif|for|in|do|done|while|case|esac|function|return|exit|export|local|alias|source)\\b"),
            TokenRule(token: "functionCall", color: SyntaxThemePalette().functionCall, pattern: "\\$[A-Za-z_][A-Za-z0-9_]*|\\$\\{[^}]+\\}"),
        ]

        static let rust: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "//[^\\n]*|/\\*[\\s\\S]*?\\*/"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "\"(?:\\\\.|[^\"\\\\\\n])*\""),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "\\b(fn|let|mut|const|static|pub|use|mod|struct|enum|impl|trait|for|while|loop|if|else|match|return|break|continue|as|where|move|ref|self|Self|crate|super|async|await|dyn|true|false|None|Some|Ok|Err)\\b"),
            TokenRule(token: "number", color: SyntaxThemePalette().number, pattern: "\\b\\d+(?:\\.\\d+)?(?:[ui](?:8|16|32|64|128|size))?\\b"),
            TokenRule(token: "type", color: SyntaxThemePalette().type, pattern: "\\b[A-Z][A-Za-z0-9_]*\\b"),
            TokenRule(token: "functionCall", color: SyntaxThemePalette().functionCall, pattern: "\\b[a-z_][A-Za-z0-9_]*(?=\\()"),
        ]

        static let go: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "//[^\\n]*|/\\*[\\s\\S]*?\\*/"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "\"(?:\\\\.|[^\"\\\\\\n])*\"|`[^`]*`"),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "\\b(package|import|func|var|const|type|struct|interface|map|chan|go|defer|return|break|continue|if|else|for|range|switch|case|default|fallthrough|select|true|false|nil|iota)\\b"),
            TokenRule(token: "number", color: SyntaxThemePalette().number, pattern: "\\b\\d+(?:\\.\\d+)?\\b"),
            TokenRule(token: "type", color: SyntaxThemePalette().type, pattern: "\\b[A-Z][A-Za-z0-9_]*\\b"),
            TokenRule(token: "functionCall", color: SyntaxThemePalette().functionCall, pattern: "\\b[a-z_][A-Za-z0-9_]*(?=\\()"),
        ]

        /// Splash's Swift grammar is preferred, but the regex
        /// fallback is what tests use when Splash isn't
        /// available. The fallback uses the same token types.
        static let swiftFallback: [TokenRule] = [
            TokenRule(token: "comment", color: SyntaxThemePalette().comment, pattern: "//[^\\n]*|/\\*[\\s\\S]*?\\*/"),
            TokenRule(token: "string", color: SyntaxThemePalette().string, pattern: "\"(?:\\\\.|[^\"\\\\\\n])*\""),
            TokenRule(token: "keyword", color: SyntaxThemePalette().keyword, pattern: "\\b(func|let|var|struct|class|enum|protocol|extension|import|return|if|else|guard|for|while|switch|case|break|continue|as|is|in|where|self|Self|init|deinit|throws|throw|try|catch|do|defer|public|private|internal|fileprivate|open|static|final|true|false|nil)\\b"),
            TokenRule(token: "number", color: SyntaxThemePalette().number, pattern: "\\b\\d+(?:\\.\\d+)?\\b"),
            TokenRule(token: "type", color: SyntaxThemePalette().type, pattern: "\\b[A-Z][A-Za-z0-9_]*\\b"),
            TokenRule(token: "functionCall", color: SyntaxThemePalette().functionCall, pattern: "\\b[a-z_][A-Za-z0-9_]*(?=\\()"),
        ]
    }
}

// MARK: - String index helpers

/// Compute the UTF-16 offset of an index in a source string.
/// Matches `NSRegularExpression`'s UTF-16 coordinate space
/// (the location / length fields of `NSTextCheckingResult`
/// are UTF-16 offsets, while Swift's `String.Index` is
/// grapheme-cluster-based).
private func utf16Offset(of index: String.Index, in source: String) -> Int {
    source.utf16.distance(from: source.utf16.startIndex, to: index)
}

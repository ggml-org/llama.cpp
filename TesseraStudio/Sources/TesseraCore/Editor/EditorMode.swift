import Foundation

// MARK: - EditorMode

/// Per-surface configuration for the editor. The same
/// `TesseraEditorView` is the canvas for Documents, Notes, and
/// Code (per spec §10). Per-surface differences are
/// configuration, not different code paths.
///
/// The mode drives:
///   * Which block types the toolbar offers as one-click inserts
///     (e.g. Notes surface promotes `callout` and `quote`; Code
///     surface promotes `codeBlock`).
///   * Whether code blocks render with a monospaced font + line
///     numbers (Code surface always; Documents surface optionally).
///   * Which animation primitives the surface uses (Code surface
///     uses Text Appear per character; Documents surface uses
///     block slide-in).
///   * Whether the editor treats the document as markdown (Notes)
///     or a structured AST (Documents).
public enum EditorMode: String, Codable, Sendable, Hashable, CaseIterable {
    /// Structured documents: full AST, all block types, full
    /// formatting toolbar, block-level animations.
    case document
    /// Notes: markdown-ish surface; promotes callouts, quotes,
    /// and the lighter animation set.
    case notes
    /// Code files: promotes codeBlock; always monospaced; per-char
    /// text-appear animation; line numbers always on.
    case code
}

// MARK: - EditorTheme

/// Colors + fonts for the editor. The values are minimal here;
/// the production theme system (dark mode, accent color,
/// high-contrast) is a Phase 3 concern. The struct is the
/// seam: a value passed in by the host window so unit tests
/// can run with a fixed theme.
public struct EditorTheme: Codable, Sendable, Hashable {
    public var bodyFont: FontDescriptor
    public var monospaceFont: FontDescriptor
    public var headingFonts: [Int: FontDescriptor]   // 1...6
    public var textColorHex: String
    public var headingColorHex: String
    public var codeBackgroundColorHex: String
    public var codeForegroundColorHex: String
    public var quoteAccentColorHex: String
    public var calloutBackgroundColorHex: String
    public var agentCursorColorHex: String          // subtle blue
    public var userCursorColorHex: String           // system-default-ish
    public var dividerColorHex: String
    public var syntaxColors: SyntaxThemePalette

    public init(
        bodyFont: FontDescriptor = .system(size: 14, weight: .regular),
        monospaceFont: FontDescriptor = .monospace(size: 13, weight: .regular),
        headingFonts: [Int: FontDescriptor] = [
            1: .system(size: 28, weight: .bold),
            2: .system(size: 22, weight: .bold),
            3: .system(size: 18, weight: .semibold),
            4: .system(size: 16, weight: .semibold),
            5: .system(size: 14, weight: .semibold),
            6: .system(size: 13, weight: .semibold),
        ],
        textColorHex: String = "#1A1A1A",
        headingColorHex: String = "#000000",
        codeBackgroundColorHex: String = "#F4F4F8",
        codeForegroundColorHex: String = "#1A1A1A",
        quoteAccentColorHex: String = "#9CA3AF",
        calloutBackgroundColorHex: String = "#FEF3C7",
        agentCursorColorHex: String = "#3B82F6",
        userCursorColorHex: String = "#111827",
        dividerColorHex: String = "#D1D5DB",
        syntaxColors: SyntaxThemePalette = .light
    ) {
        self.bodyFont = bodyFont
        self.monospaceFont = monospaceFont
        self.headingFonts = headingFonts
        self.textColorHex = textColorHex
        self.headingColorHex = headingColorHex
        self.codeBackgroundColorHex = codeBackgroundColorHex
        self.codeForegroundColorHex = codeForegroundColorHex
        self.quoteAccentColorHex = quoteAccentColorHex
        self.calloutBackgroundColorHex = calloutBackgroundColorHex
        self.agentCursorColorHex = agentCursorColorHex
        self.userCursorColorHex = userCursorColorHex
        self.dividerColorHex = dividerColorHex
        self.syntaxColors = syntaxColors
    }

    public static let light = EditorTheme()
}

// MARK: - FontDescriptor

/// Platform-agnostic font description. The macOS path uses
/// `NSFont(descriptor:)`; the iOS path uses `UIFont`. The
/// `FontDescriptor` is the platform-agnostic form the renderer
/// consumes; the platform layer translates.
public struct FontDescriptor: Codable, Sendable, Hashable {
    public enum Family: String, Codable, Sendable, Hashable {
        case system
        case monospace
        case serif
        case rounded
    }
    public var family: Family
    public var size: CGFloat
    public var weight: Weight
    public var italic: Bool

    public enum Weight: String, Codable, Sendable, Hashable {
        case ultraLight, thin, light, regular, medium, semibold, bold, heavy, black
    }

    public init(family: Family = .system, size: CGFloat, weight: Weight = .regular, italic: Bool = false) {
        self.family = family
        self.size = size
        self.weight = weight
        self.italic = italic
    }

    public static func system(size: CGFloat, weight: Weight = .regular) -> FontDescriptor {
        FontDescriptor(family: .system, size: size, weight: weight)
    }
    public static func monospace(size: CGFloat, weight: Weight = .regular) -> FontDescriptor {
        FontDescriptor(family: .monospace, size: size, weight: weight)
    }
}

// MARK: - SyntaxThemePalette

/// The colors Splash uses for syntax highlighting. The palette
/// follows the same naming as Xcode's default light/dark themes
/// so the rendered code looks at home in either.
///
/// Splash is grammar-driven; the highlighter assigns the
/// palette's named roles to Splash's `SyntaxHighlighter` rules.
public struct SyntaxThemePalette: Codable, Sendable, Hashable {
    public var plain: String                  // default text
    public var `operator`: String
    public var keyword: String
    public var type: String
    public var number: String
    public var string: String
    public var identifier: String
    public var comment: String
    public var functionCall: String

    public init(
        plain: String = "#1A1A1A",
        operator: String = "#7C3AED",
        keyword: String = "#DC2626",
        type: String = "#0891B2",
        number: String = "#0EA5E9",
        string: String = "#16A34A",
        identifier: String = "#1A1A1A",
        comment: String = "#6B7280",
        functionCall: String = "#2563EB"
    ) {
        self.plain = plain
        self.operator = `operator`
        self.keyword = keyword
        self.type = type
        self.number = number
        self.string = string
        self.identifier = identifier
        self.comment = comment
        self.functionCall = functionCall
    }

    public static let light = SyntaxThemePalette()
}

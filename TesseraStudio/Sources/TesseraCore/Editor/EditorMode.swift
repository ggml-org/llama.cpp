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
    /// Code files with the default configuration. v1 used
    /// this bare `case code` form (the per-surface differences
    /// were hardcoded in the renderer). Phase 5 keeps the
    /// bare case for source-compat with the existing
    /// callers; the `.codeWithConfig(...)` case is for files
    /// whose user-driven settings diverge from the defaults.
    case code
    /// Code files with explicit per-file configuration. The
    /// renderer reads `showLineNumbers`, `codeFolding`,
    /// `multiCursor`, `findInFile`, and `minimap` from the
    /// payload. The bare `case code` is a shorthand for
    /// `.codeWithConfig(CodeEditorConfiguration())` (the
    /// defaults).
    case codeWithConfig
}

// MARK: - CodeEditorConfiguration

/// Per-file knobs for the Code surface. The default
/// values match the design doc (`docs/tessera-productivity-
/// materials-code-design.md` §4):
///
///   * Line numbers on (the gutter is the user's primary
///     navigation tool; the editor's spec lists it as
///     non-negotiable for code).
///   * Syntax highlighting on for known languages (the
///     `CodeBlockHighlighter` falls back to a plain
///     monospaced render for unknown languages).
///   * Code folding on (the design doc's `codeFolding`
///     flag).
///   * Multi-cursor on (Cmd-click to add cursors, edit
///     multiple lines at once).
///   * Find-in-file on (Cmd-F opens the inline find bar;
///     regex is supported).
///   * Minimap off by default (the design doc says
///     "off by default, opt-in" -- the minimap costs
///     ~10% of the editor's frame budget for a 500-line
///     file and is a per-user preference, not a per-file
///     one).
///
/// **Why a separate type, not a `Mutation` extension.**
/// The `EditorMode` enum is a value type the SwiftUI host
/// passes to the `TesseraEditorView`. Adding an associated
/// value to one case is the Swift idiomatic way to express
/// "this case carries extra data" -- the alternative
/// (a separate `codeConfiguration: CodeEditorConfiguration?`
/// field on the enum) would force every switch site to
/// handle the optional. The `codeWithConfig` case is the
/// production path; the bare `case code` exists so the
/// Phase 1-4 callers (which already used `case code`) keep
/// compiling and get the default configuration for free.
public struct CodeEditorConfiguration: Codable, Sendable, Hashable {
    public var showLineNumbers: Bool
    public var syntaxHighlightingLanguage: String?
    public var codeFolding: Bool
    public var multiCursor: Bool
    public var findInFile: Bool
    public var minimap: Bool

    public init(
        showLineNumbers: Bool = true,
        syntaxHighlightingLanguage: String? = nil,
        codeFolding: Bool = true,
        multiCursor: Bool = true,
        findInFile: Bool = true,
        minimap: Bool = false
    ) {
        self.showLineNumbers = showLineNumbers
        self.syntaxHighlightingLanguage = syntaxHighlightingLanguage
        self.codeFolding = codeFolding
        self.multiCursor = multiCursor
        self.findInFile = findInFile
        self.minimap = minimap
    }

    /// The default configuration. The values are
    /// conservative (line numbers on, folding on, multi-
    /// cursor on, find on, minimap off); the host view
    /// shows the user a settings panel to opt into the
    /// minimap and to opt out of folding for tiny files.
    public static let `default` = CodeEditorConfiguration()
}

extension EditorMode {
    /// Resolve the configuration for a code mode. The bare
    /// `case code` returns the default; `.codeWithConfig`
    /// with an associated configuration would be the
    /// production path, but the bare case + a per-instance
    /// `codeConfigurationOverride: CodeEditorConfiguration?`
    /// on the view is the v1 shape (the configuration is
    /// per-file, not per-mode). Non-code modes return
    /// `nil` (the caller should not ask).
    public var isCodeMode: Bool {
        switch self {
        case .code, .codeWithConfig: return true
        case .document, .notes: return false
        }
    }

    /// The default configuration for a code mode. The view
    /// uses this when no per-file override is set; the
    /// `codeConfigurationOverride` field on the view is
    /// the v1 per-file knob.
    public static let codeDefault = CodeEditorConfiguration()
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

import Foundation
#if canImport(AppKit)
import AppKit
public typealias PlatformFont = NSFont
public typealias PlatformColor = NSColor
public typealias PlatformAttributedString = NSAttributedString
public typealias PlatformMutableAttributedString = NSMutableAttributedString
#elseif canImport(UIKit)
import UIKit
public typealias PlatformFont = UIFont
public typealias PlatformColor = UIColor
public typealias PlatformAttributedString = NSAttributedString
public typealias PlatformMutableAttributedString = NSMutableAttributedString
#endif

// MARK: - BlockRenderer

/// Pure function `Block -> NSAttributedString` (per spec §4.4).
/// The renderer is the seam between the AST and the platform
/// text view. It is platform-agnostic: it consumes
/// `FontDescriptor` + `EditorTheme` + `Block` and emits an
/// `NSAttributedString`; the platform layer (AppKit / UIKit)
/// resolves the fonts and colors via the `PlatformFontResolver`
/// protocol.
///
/// The renderer is **stateless** and **side-effect-free**; every
/// block can be rendered independently. This is the property
/// the `TesseraTextContentManager` relies on: when a block
/// changes, only its element needs to be re-rendered, not the
/// whole document.
///
/// **Inline annotations.** The renderer applies the per-run
/// `InlineRun.Annotation` set as `NSAttributedString` attributes
/// (`.font`, `.underlineStyle`, `.foregroundColor`, custom
/// keys for links and code). The mapping is documented on each
/// annotation case so it's easy to extend.
///
/// **Code blocks.** When a `codeBlock` block has
/// `attributes["language"]` set, the renderer delegates to
/// `CodeBlockHighlighter` (Splash) and applies the resulting
/// runs. Without a language (or for unknown languages), the
/// renderer falls back to a monospaced single-run attribute.
///
/// **Images.** Image blocks produce an `NSAttributedString`
/// with an `NSTextAttachment` whose `image` is loaded from
/// `attributes["source"]`. Loading is best-effort: a missing
/// image becomes a placeholder attachment (so the text view
/// doesn't crash on a stale URL).
public struct BlockRenderer: Sendable {

    public let theme: EditorTheme
    public let fontResolver: PlatformFontResolver

    public init(theme: EditorTheme = .light, fontResolver: PlatformFontResolver = DefaultFontResolver()) {
        self.theme = theme
        self.fontResolver = fontResolver
    }

    /// Render one block to an attributed string. The output is
    /// never nil: even an empty `paragraph` produces an
    /// attributed string with a single empty run (so the
    /// text view shows a caret and the user can type into it).
    public func render(_ block: Block, in mode: EditorMode) -> NSAttributedString {
        switch block.type {
        case .heading:
            return renderHeading(block, mode: mode)
        case .paragraph:
            return renderParagraph(block, mode: mode)
        case .list:
            return renderListContainer(block, mode: mode)
        case .listItem:
            return renderListItem(block, mode: mode)
        case .table:
            // Tables are rendered as a placeholder; the Phase 3
            // table surface (NSTextTable / UITextView NSTextTable)
            // is its own deliverable.
            return renderTablePlaceholder(block, mode: mode)
        case .tableCell:
            return renderInline(block.content, mode: mode)
        case .image:
            return renderImage(block, mode: mode)
        case .codeBlock:
            return renderCodeBlock(block, mode: mode)
        case .callout:
            return renderCallout(block, mode: mode)
        case .divider:
            return renderDivider(block, mode: mode)
        case .quote:
            return renderQuote(block, mode: mode)
        case .toggle:
            return renderToggle(block, mode: mode)
        case .equation:
            return renderEquation(block, mode: mode)
        }
    }

    /// Convenience: render a sequence of blocks in document
    /// order. Each block is separated by a single newline so
    /// the text view's line layout works correctly.
    public func renderAll(_ blocks: [Block], in mode: EditorMode) -> NSAttributedString {
        let out = NSMutableAttributedString()
        for (idx, block) in blocks.enumerated() {
            if idx > 0 { out.append(NSAttributedString(string: "\n")) }
            out.append(render(block, in: mode))
        }
        return out
    }

    // MARK: - Per-type renderers

    private func renderHeading(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let level = (block.attributes["level"]?.intValue ?? 1).clamped(to: 1...6)
        let font = theme.headingFonts[level] ?? theme.bodyFont
        let color = PlatformColor.fromHex(theme.headingColorHex) ?? .labelColor
        let baseAttrs: [NSAttributedString.Key: Any] = [
            .font: fontResolver.font(from: font),
            .foregroundColor: color,
        ]
        return renderInlineRuns(block.content, baseAttributes: baseAttrs)
    }

    private func renderParagraph(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let font = theme.bodyFont
        let color = PlatformColor.fromHex(theme.textColorHex) ?? .labelColor
        let baseAttrs: [NSAttributedString.Key: Any] = [
            .font: fontResolver.font(from: font),
            .foregroundColor: color,
        ]
        return renderInlineRuns(block.content, baseAttributes: baseAttrs)
    }

    private func renderListContainer(_ block: Block, mode: EditorMode) -> NSAttributedString {
        // The list container itself is a marker prefix; the
        // children (listItem blocks) carry the actual content.
        // The TesseraTextContentManager flattens list containers
        // by interleaving this prefix with the rendered children
        // when building elements, so this string is rarely used
        // directly. Kept for completeness.
        let style = (block.attributes["style"]?.stringValue) ?? "unordered"
        let marker: String
        switch style {
        case "ordered": marker = "1."
        case "task":    marker = "☐"
        default:        marker = "•"
        }
        let out = NSMutableAttributedString(string: marker + " ")
        out.append(renderInline(block.content, mode: mode))
        return out
    }

    private func renderListItem(_ block: Block, mode: EditorMode) -> NSAttributedString {
        // The list item's parent is the list container; the
        // marker prefix is the list's style. We don't have the
        // parent here (the renderer is per-block), so the
        // prefix is a bullet by default. The
        // `TesseraTextContentManager` overrides this with the
        // parent's style for accurate rendering.
        let prefix = "• "
        let out = NSMutableAttributedString(string: prefix)
        out.append(renderInline(block.content, mode: mode))
        return out
    }

    private func renderTablePlaceholder(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let rows = block.attributes["rows"]?.intValue ?? 0
        let cols = block.attributes["cols"]?.intValue ?? 0
        let label = "[Table \(rows)×\(cols) — Phase 3 surface]"
        return NSAttributedString(
            string: label,
            attributes: [
                .font: fontResolver.font(from: theme.bodyFont),
                .foregroundColor: PlatformColor.secondaryLabelColor,
            ]
        )
    }

    private func renderImage(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let source = block.attributes["source"]?.stringValue ?? ""
        let alt = block.attributes["alt"]?.stringValue ?? "image"
        #if canImport(AppKit)
        let attachment = NSTextAttachment()
        if let url = URL(string: source),
           let data = try? Data(contentsOf: url),
           let image = NSImage(data: data) {
            attachment.image = image
        } else {
            attachment.image = NSImage(systemSymbolName: "photo",
                                       accessibilityDescription: alt)
        }
        #elseif canImport(UIKit)
        let attachment = NSTextAttachment()
        if let url = URL(string: source),
           let data = try? Data(contentsOf: url),
           let image = UIImage(data: data) {
            attachment.image = image
        } else {
            attachment.image = UIImage(systemName: "photo")
        }
        #else
        // Non-platform: placeholder string. The build configuration
        // that doesn't import AppKit or UIKit is a Linux CI stub;
        // images are out of scope there.
        return NSAttributedString(string: "[image: \(alt)]")
        #endif
        return NSAttributedString(attachment: attachment)
    }

    private func renderCodeBlock(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let source = block.content.first?.text ?? ""
        let language = block.attributes["language"]?.stringValue
        let highlighter = CodeBlockHighlighter(theme: theme.syntaxColors)
        let highlighted = highlighter.highlight(source: source, language: language)
        // Append a trailing newline so consecutive code blocks
        // render with separation. The text view's line layout
        // relies on this.
        let out = NSMutableAttributedString(attributedString: highlighted)
        out.append(NSAttributedString(string: "\n", attributes: [
            .font: fontResolver.font(from: theme.monospaceFont),
            .foregroundColor: PlatformColor.fromHex(theme.codeForegroundColorHex) ?? .labelColor,
        ]))
        return out
    }

    private func renderCallout(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let emoji = block.attributes["emoji"]?.stringValue ?? "💡"
        let out = NSMutableAttributedString(string: "\(emoji)  ")
        out.append(renderInline(block.content, mode: mode))
        let bg = PlatformColor.fromHex(theme.calloutBackgroundColorHex)
        if let bg = bg {
            out.addAttribute(.backgroundColor, value: bg,
                             range: NSRange(location: 0, length: out.length))
        }
        return out
    }

    private func renderDivider(_ block: Block, mode: EditorMode) -> NSAttributedString {
        // A divider is a single "—" or `─` glyph. The text view's
        // line layout treats it like any other line. The
        // `TesseraTextContentManager` adds the actual rule as a
        // paragraph border on the line range via an attribute the
        // platform layer interprets.
        return NSAttributedString(
            string: "─" + String(repeating: "─", count: 60),
            attributes: [
                .font: fontResolver.font(from: theme.bodyFont),
                .foregroundColor: PlatformColor.fromHex(theme.dividerColorHex) ?? .secondaryLabelColor,
            ]
        )
    }

    private func renderQuote(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let cite = block.attributes["cite"]?.stringValue
        let out = NSMutableAttributedString(string: "“")
        out.append(renderInline(block.content, mode: mode))
        out.append(NSAttributedString(string: "”"))
        if let cite, !cite.isEmpty {
            out.append(NSAttributedString(string: " — \(cite)"))
        }
        out.addAttribute(.foregroundColor,
                         value: PlatformColor.fromHex(theme.quoteAccentColorHex) ?? .secondaryLabelColor,
                         range: NSRange(location: 0, length: out.length))
        return out
    }

    private func renderToggle(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let expanded = block.attributes["expanded"]?.boolValue ?? true
        let marker = expanded ? "▾" : "▸"
        // The toggle's children are rendered separately by the
        // TesseraTextContentManager (interleaved with this
        // header). The renderer here just produces the header.
        let out = NSMutableAttributedString(string: "\(marker)  ")
        out.append(renderInline(block.content, mode: mode))
        return out
    }

    private func renderEquation(_ block: Block, mode: EditorMode) -> NSAttributedString {
        let latex = block.attributes["latex"]?.stringValue ?? ""
        // Equations are a Phase 3 surface (LaTeX rendering via
        // MathJax or `iosMath`). For now, render as inline code
        // so the AST round-trips visually.
        return NSAttributedString(
            string: "$\(latex)$",
            attributes: [
                .font: fontResolver.font(from: theme.monospaceFont),
                .foregroundColor: PlatformColor.fromHex(theme.codeForegroundColorHex) ?? .labelColor,
                .backgroundColor: PlatformColor.fromHex(theme.codeBackgroundColorHex) ?? .controlBackgroundColor,
            ]
        )
    }

    // MARK: - Inline run rendering

    /// Render a sequence of `InlineRun`s with annotations applied.
    /// The base attributes (font, color) are overridden by each
    /// run's annotations; the override model is the standard
    /// attributed-string cascade.
    public func renderInlineRuns(_ runs: [InlineRun], baseAttributes: [NSAttributedString.Key: Any]) -> NSAttributedString {
        let out = NSMutableAttributedString()
        for run in runs {
            var attrs = baseAttributes
            for annotation in run.annotations {
                applyAnnotation(annotation, to: &attrs)
            }
            out.append(NSAttributedString(string: run.text, attributes: attrs))
        }
        return out
    }

    private func renderInline(_ runs: [InlineRun], mode: EditorMode) -> NSAttributedString {
        let base: [NSAttributedString.Key: Any] = [
            .font: fontResolver.font(from: theme.bodyFont),
            .foregroundColor: PlatformColor.fromHex(theme.textColorHex) ?? .labelColor,
        ]
        return renderInlineRuns(runs, baseAttributes: base)
    }

    private func applyAnnotation(_ annotation: InlineRun.Annotation, to attrs: inout [NSAttributedString.Key: Any]) {
        switch annotation {
        case .bold:
            // Toggle bold on the existing font.
            if let base = attrs[.font] as? PlatformFont {
                let descriptor = base.fontDescriptor.withSymbolicTraits(base.fontDescriptor.symbolicTraits.union(.bold))
                #if canImport(AppKit)
                if let new = NSFont(descriptor: descriptor, size: base.pointSize) {
                    attrs[.font] = new
                }
                #elseif canImport(UIKit)
                if let new = UIFont(descriptor: descriptor, size: base.pointSize) {
                    attrs[.font] = new
                }
                #endif
            }
        case .italic:
            if let base = attrs[.font] as? PlatformFont {
                let descriptor = base.fontDescriptor.withSymbolicTraits(base.fontDescriptor.symbolicTraits.union(.italic))
                #if canImport(AppKit)
                if let new = NSFont(descriptor: descriptor, size: base.pointSize) {
                    attrs[.font] = new
                }
                #elseif canImport(UIKit)
                if let new = UIFont(descriptor: descriptor, size: base.pointSize) {
                    attrs[.font] = new
                }
                #endif
            }
        case .underline:
            attrs[.underlineStyle] = NSUnderlineStyle.single.rawValue
        case .strikethrough:
            attrs[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
        case .code:
            attrs[.font] = fontResolver.font(from: theme.monospaceFont)
            if let bg = PlatformColor.fromHex(theme.codeBackgroundColorHex) {
                attrs[.backgroundColor] = bg
            }
        case .subscript:
            if let base = attrs[.font] as? PlatformFont {
                #if canImport(AppKit)
                let new = NSFont(descriptor: base.fontDescriptor, size: base.pointSize * 0.75)
                #elseif canImport(UIKit)
                let new = UIFont(descriptor: base.fontDescriptor, size: base.pointSize * 0.75)
                #endif
                if let new = new {
                    attrs[.font] = new
                }
                attrs[.baselineOffset] = -base.pointSize * 0.25
            }
        case .superscript:
            if let base = attrs[.font] as? PlatformFont {
                #if canImport(AppKit)
                let new = NSFont(descriptor: base.fontDescriptor, size: base.pointSize * 0.75)
                #elseif canImport(UIKit)
                let new = UIFont(descriptor: base.fontDescriptor, size: base.pointSize * 0.75)
                #endif
                if let new = new {
                    attrs[.font] = new
                }
                attrs[.baselineOffset] = base.pointSize * 0.25
            }
        case .link(let url):
            attrs[.link] = url
            attrs[.foregroundColor] = PlatformColor.systemBlue
        case .color(let hex):
            if let c = PlatformColor.fromHex(hex) {
                attrs[.foregroundColor] = c
            }
        }
    }
}

// MARK: - PlatformFontResolver

/// Resolves a `FontDescriptor` to a platform font. The default
/// implementation uses `NSFont` / `UIFont` from the system font
/// family. Production can inject a custom resolver (e.g., to
/// fall back to a specific bundled font if a system one is
/// unavailable).
public protocol PlatformFontResolver: Sendable {
    func font(from descriptor: FontDescriptor) -> PlatformFont
}

public struct DefaultFontResolver: PlatformFontResolver {
    public init() {}

    public func font(from descriptor: FontDescriptor) -> PlatformFont {
        #if canImport(AppKit)
        let baseSize = descriptor.size
        switch descriptor.family {
        case .system:
            return NSFont.systemFont(ofSize: baseSize, weight: weight(for: descriptor.weight))
        case .monospace:
            return NSFont.monospacedSystemFont(ofSize: baseSize, weight: weight(for: descriptor.weight))
        case .serif:
            return NSFont(name: "Times New Roman", size: baseSize)
                ?? NSFont.systemFont(ofSize: baseSize, weight: weight(for: descriptor.weight))
        case .rounded:
            return NSFont(name: "SF Pro Rounded", size: baseSize)
                ?? NSFont.systemFont(ofSize: baseSize, weight: weight(for: descriptor.weight))
        }
        #elseif canImport(UIKit)
        let baseSize = descriptor.size
        switch descriptor.family {
        case .system:
            return UIFont.systemFont(ofSize: baseSize, weight: weight(for: descriptor.weight))
        case .monospace:
            return UIFont.monospacedSystemFont(ofSize: baseSize, weight: weight(for: descriptor.weight))
        case .serif:
            return UIFont(name: "Times New Roman", size: baseSize)
                ?? UIFont.systemFont(ofSize: baseSize, weight: weight(for: descriptor.weight))
        case .rounded:
            return UIFont(descriptor:
                UIFont.systemFont(ofSize: baseSize).fontDescriptor
                    .withDesign(.rounded) ?? UIFont.systemFont(ofSize: baseSize).fontDescriptor,
                size: baseSize)
        }
        #else
        return PlatformFont()  // Unreachable on supported builds
        #endif
    }

    #if canImport(AppKit)
    private func weight(for w: FontDescriptor.Weight) -> NSFont.Weight {
        switch w {
        case .ultraLight: return .ultraLight
        case .thin: return .thin
        case .light: return .light
        case .regular: return .regular
        case .medium: return .medium
        case .semibold: return .semibold
        case .bold: return .bold
        case .heavy: return .heavy
        case .black: return .black
        }
    }
    #elseif canImport(UIKit)
    private func weight(for w: FontDescriptor.Weight) -> UIFont.Weight {
        switch w {
        case .ultraLight: return .ultraLight
        case .thin: return .thin
        case .light: return .light
        case .regular: return .regular
        case .medium: return .medium
        case .semibold: return .semibold
        case .bold: return .bold
        case .heavy: return .heavy
        case .black: return .black
        }
    }
    #endif
}

// MARK: - PlatformColor helpers

extension PlatformColor {
    /// Construct a color from a `#RRGGBB` or `#RRGGBBAA` hex
    /// string. Returns nil for malformed input.
    public static func fromHex(_ hex: String) -> PlatformColor? {
        var s = hex
        if s.hasPrefix("#") { s.removeFirst() }
        guard s.count == 6 || s.count == 8 else { return nil }
        var value: UInt64 = 0
        guard Scanner(string: s).scanHexInt64(&value) else { return nil }
        #if canImport(AppKit)
        if s.count == 6 {
            let r = CGFloat((value >> 16) & 0xFF) / 255
            let g = CGFloat((value >> 8) & 0xFF) / 255
            let b = CGFloat(value & 0xFF) / 255
            return NSColor(srgbRed: r, green: g, blue: b, alpha: 1.0)
        } else {
            let r = CGFloat((value >> 24) & 0xFF) / 255
            let g = CGFloat((value >> 16) & 0xFF) / 255
            let b = CGFloat((value >> 8) & 0xFF) / 255
            let a = CGFloat(value & 0xFF) / 255
            return NSColor(srgbRed: r, green: g, blue: b, alpha: a)
        }
        #elseif canImport(UIKit)
        if s.count == 6 {
            let r = CGFloat((value >> 16) & 0xFF) / 255
            let g = CGFloat((value >> 8) & 0xFF) / 255
            let b = CGFloat(value & 0xFF) / 255
            return UIColor(red: r, green: g, blue: b, alpha: 1.0)
        } else {
            let r = CGFloat((value >> 24) & 0xFF) / 255
            let g = CGFloat((value >> 16) & 0xFF) / 255
            let b = CGFloat((value >> 8) & 0xFF) / 255
            let a = CGFloat(value & 0xFF) / 255
            return UIColor(red: r, green: g, blue: b, alpha: a)
        }
        #else
        return nil
        #endif
    }
}

// MARK: - JSONValue extension

extension JSONValue {
    /// Best-effort int read; returns nil for non-numeric values.
    /// Uses the existing `numberValue` accessor (Double) and
    /// truncates. Returns nil for non-finite values.
    public var intValue: Int? {
        guard let d = numberValue, d.isFinite, d == d.rounded() else { return nil }
        return Int(d)
    }
}

// MARK: - Comparable clamp

extension Comparable {
    /// Clamp the value to the closed range `[low, high]`.
    public func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

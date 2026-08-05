import Foundation
#if canImport(AppKit)
import AppKit
public typealias PlatformNSTextElement = NSTextElement
public typealias PlatformNSTextParagraph = NSTextParagraph
public typealias PlatformNSTextRange = NSTextRange
public typealias PlatformNSTextLocation = NSTextLocation
#elseif canImport(UIKit)
import UIKit
public typealias PlatformNSTextElement = NSTextElement
public typealias PlatformNSTextParagraph = NSTextParagraph
public typealias PlatformNSTextRange = NSTextRange
public typealias PlatformNSTextLocation = NSTextLocation
#endif

// MARK: - TesseraTextElementData

/// The platform-agnostic data carried by a `TesseraTextElement`.
/// The struct is what the rest of the editor (the
/// `TesseraTextContentManager`, the platform view layer, the
/// coalescer) operates on. The macOS / iOS
/// `TesseraTextElement` wraps this struct in an
/// `NSTextParagraph` (the concrete `NSTextElement` subclass
/// that ships an `attributedString` out of the box).
///
/// **Why a struct.** The brief requires one
/// `TesseraTextElement` per block; the struct is the value
/// that travels through the editor's pipeline. The
/// `NSTextParagraph` wrapper is the platform-typed object
/// the text view consumes (Apple's `NSTextElement` is the
/// abstract base; `NSTextParagraph` is the concrete one
/// that carries an `NSAttributedString` and an
/// `elementRange`).
///
/// **Range.** The `NSTextRange` is platform-typed (AppKit /
/// UIKit); for testable data, the struct also carries
/// `rangeStart` and `rangeEnd` as plain integers (UTF-16
/// offsets, matching the `NSAttributedString` coordinate
/// space).
public struct TesseraTextElementData: @unchecked Sendable, Hashable {
    public let blockID: UUID
    public let blockType: BlockType
    public let attributedString: NSAttributedString
    public let rangeStart: Int   // UTF-16 offset (matches NSAttributedString)
    public let rangeEnd: Int     // UTF-16 offset (exclusive)
    public let parentID: UUID?   // nil for top-level elements

    public init(
        blockID: UUID,
        blockType: BlockType,
        attributedString: NSAttributedString,
        rangeStart: Int,
        rangeEnd: Int,
        parentID: UUID? = nil
    ) {
        self.blockID = blockID
        self.blockType = blockType
        self.attributedString = attributedString
        self.rangeStart = rangeStart
        self.rangeEnd = rangeEnd
        self.parentID = parentID
    }

    public var length: Int { rangeEnd - rangeStart }

    /// The range as a `Range<Int>`, useful for the
    /// mutation engine's pre-snapshot logic and for the
    /// test's per-block assertions.
    public var intRange: Range<Int> { rangeStart..<rangeEnd }
}

// MARK: - TesseraTextElement (NSTextParagraph wrapper)

#if canImport(AppKit) || canImport(UIKit)
/// A typed wrapper that holds one `TesseraTextElementData`
/// as a platform-typed `NSTextParagraph`. The
/// `NSTextParagraph` is the concrete `NSTextElement` Apple
/// provides for "a chunk of attributed text", and it carries
/// an `elementRange` natively — exactly the shape the
/// `TesseraTextContentManager` needs.
///
/// **Why a wrapper instead of a custom NSTextElement
/// subclass.** NSTextElement's `elementRange` is a `nullable
/// strong` property; it's designed to be set after
/// construction, not provided as a designated-initializer
/// parameter. NSTextParagraph follows the same pattern
/// (its `paragraphContentRange` is computed from
/// `elementRange` + `attributedString`). Wrapping
/// `NSTextParagraph` instead of subclassing `NSTextElement`
/// directly is the path that composes with Apple's APIs
/// without fighting them.
public final class TesseraTextElement: NSTextParagraph {
    public let data: TesseraTextElementData
    public var blockID: UUID { data.blockID }
    public var blockType: BlockType { data.blockType }
    /// The element's range as a `Range<Int>` (UTF-16
    /// offsets). The platform-typed `elementRange` is the
    /// canonical form; this is the testable form.
    public var intRange: Range<Int> { data.intRange }

    public init(data: TesseraTextElementData) {
        self.data = data
        super.init(attributedString: data.attributedString)
        // Set the elementRange to the NSTextRange that
        // covers the element's bytes in the document.
        self.elementRange = makeIntTextRange(start: data.rangeStart, end: data.rangeEnd)
    }

    public required init?(coder: NSCoder) {
        self.data = TesseraTextElementData(
            blockID: UUID(),
            blockType: .paragraph,
            attributedString: NSAttributedString(),
            rangeStart: 0,
            rangeEnd: 0,
            parentID: nil
        )
        super.init(attributedString: NSAttributedString())
    }

    public override var description: String {
        "TesseraTextElement(blockID: \(data.blockID), type: \(data.blockType.rawValue), range: \(data.rangeStart)..\(data.rangeEnd))"
    }
}
#endif

// MARK: - ElementBuilder

/// Walks a `DocumentAST` and produces a sequence of
/// `TesseraTextElementData` values, one per rendered block.
/// The walker handles container blocks (`list`, `toggle`,
/// `table`, `callout`) by emitting the container as one
/// element (with a header prefix) and each child as a
/// separate element. The platform view layer interleaves
/// them via the `parentID` field.
///
/// **Single source of truth.** The walker is the only place
/// in the editor that decides how a block tree maps to a
/// linear sequence of text elements. Both the platform
/// `TesseraTextContentManager` and the test suite use it.
///
/// **Range computation.** The walker walks the elements in
/// order and accumulates the UTF-16 length of each element's
/// `attributedString` (the lengths match the
/// `NSAttributedString` coordinate space the platform text
/// view uses). The first element's range starts at 0; each
/// subsequent element starts at the previous end.
public struct ElementBuilder: Sendable {
    public let renderer: BlockRenderer
    public let mode: EditorMode

    public init(renderer: BlockRenderer = BlockRenderer(), mode: EditorMode = .document) {
        self.renderer = renderer
        self.mode = mode
    }

    /// Build the element list for a `DocumentAST`. The
    /// returned elements are in document order: depth-first
    /// over the root children. Container blocks appear once
    /// (the header); their children are emitted as siblings
    /// with the container's id in their `parentID`.
    public func buildElements(for document: DocumentAST) -> [TesseraTextElementData] {
        var out: [TesseraTextElementData] = []
        var cursor = 0
        for rootID in document.rootChildren {
            emit(blockID: rootID, in: document, parentID: nil, into: &out, cursor: &cursor)
        }
        return out
    }

    private func emit(
        blockID: UUID,
        in document: DocumentAST,
        parentID: UUID?,
        into out: inout [TesseraTextElementData],
        cursor: inout Int
    ) {
        guard let block = document.blocks[blockID] else { return }
        let rendered = renderer.render(block, in: mode)
        let length = rendered.length
        let element = TesseraTextElementData(
            blockID: block.id,
            blockType: block.type,
            attributedString: rendered,
            rangeStart: cursor,
            rangeEnd: cursor + length,
            parentID: parentID ?? block.parentID
        )
        out.append(element)
        cursor += length
        // Container blocks: emit each child as a sibling.
        for childID in block.children {
            emit(blockID: childID, in: document, parentID: block.id, into: &out, cursor: &cursor)
        }
    }
}

import Foundation
#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

// MARK: - TesseraTextContentManagerData

/// The platform-agnostic core of the editor's
/// `NSTextContentManager`. The struct holds the
/// `DocumentAST` + the element list + a `BlockRenderer`,
/// and exposes the queries the platform text view's
/// `NSTextContentManager` subclass needs. Tests drive the
/// struct directly; the AppKit / UIKit subclass is a thin
/// wrapper that calls into the struct's methods.
///
/// **Element list.** The list is rebuilt on every
/// `applyMutation` call via `ElementBuilder`. The
/// rebuild is O(n) in the document's block count, which
/// meets the brief's "1000+ blocks enumerate in < 50ms"
/// requirement (we measure this in the test suite).
///
/// **Indexing.** `elements` is a sorted array by
/// `rangeStart`. `elementAt(offset:)` is a binary search
/// (O(log n)) that returns the element whose range
/// contains the offset. The platform view layer calls
/// this on every caret move.
///
/// **Threading.** The struct is not `Sendable`: it holds
/// an in-memory `DocumentAST` that the chat panel and
/// the agent also mutate. The struct's mutation methods
/// are documented as main-thread-only; the platform
/// text view's delegate methods all run on the main
/// thread, so the threading is implicit.
public final class TesseraTextContentManagerData {

    public private(set) var document: DocumentAST
    public private(set) var elements: [TesseraTextElementData]
    public let renderer: BlockRenderer
    public let mode: EditorMode

    public init(
        document: DocumentAST = .empty,
        renderer: BlockRenderer = BlockRenderer(),
        mode: EditorMode = .document
    ) {
        self.document = document
        self.renderer = renderer
        self.mode = mode
        self.elements = ElementBuilder(renderer: renderer, mode: mode).buildElements(for: document)
    }

    /// Replace the document. Rebuilds the element list.
    public func setDocument(_ document: DocumentAST) {
        self.document = document
        self.elements = ElementBuilder(renderer: renderer, mode: mode).buildElements(for: document)
    }

    /// Apply a single mutation via the Phase 1
    /// `MutationEngine`. The engine's pre-snapshot is
    /// returned for the caller to embed in a receipt.
    @discardableResult
    public func applyMutation(_ mutation: Mutation) throws -> [UUID: Block] {
        var engine = MutationEngine()
        let pre = try engine.apply(mutation, to: &document)
        elements = ElementBuilder(renderer: renderer, mode: mode).buildElements(for: document)
        return pre
    }

    /// Apply a batch of mutations via the Phase 1
    /// `MutationEngine`. Returns the union of pre-snapshots.
    @discardableResult
    public func applyMutations(_ mutations: [Mutation]) throws -> [UUID: Block] {
        var engine = MutationEngine()
        var pre: [UUID: Block] = [:]
        var workingDocument = document
        for mutation in mutations {
            let snap = try engine.apply(mutation, to: &workingDocument)
            for (k, v) in snap { pre[k] = v }
        }
        self.document = workingDocument
        self.elements = ElementBuilder(renderer: renderer, mode: mode).buildElements(for: document)
        return pre
    }

    // MARK: - Queries

    /// The element that contains the given UTF-16 offset
    /// in the document's flat text. Returns nil when the
    /// document is empty or the offset is past the end.
    public func elementAt(offset: Int) -> TesseraTextElementData? {
        guard !elements.isEmpty else { return nil }
        // Binary search by rangeStart; pick the element
        // whose range contains the offset.
        var lo = 0
        var hi = elements.count - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            let el = elements[mid]
            if offset < el.rangeStart {
                hi = mid - 1
            } else if offset >= el.rangeEnd {
                lo = mid + 1
            } else {
                return el
            }
        }
        // Offset is past the last element; return the last
        // element if it's the immediate predecessor.
        if offset == elements.last?.rangeEnd {
            return elements.last
        }
        return nil
    }

    /// The element for the given block id. Returns nil if
    /// the block isn't in the document.
    public func element(for blockID: UUID) -> TesseraTextElementData? {
        elements.first { $0.blockID == blockID }
    }

    /// The number of elements in the document.
    public var elementCount: Int { elements.count }

    /// True iff the document has no elements (an empty
    /// document, or a document whose every block failed
    /// to render).
    public var isEmpty: Bool { elements.isEmpty }

    /// Concatenate every element's `attributedString` into
    /// a single attributed string. Used by the platform
    /// view layer when it needs the whole document as a
    /// string (e.g., for find/replace).
    public func fullAttributedString() -> NSAttributedString {
        let out = NSMutableAttributedString()
        for element in elements {
            out.append(element.attributedString)
        }
        return out
    }
}

#if canImport(AppKit) || canImport(UIKit)
/// The platform-typed `NSTextContentManager` subclass
/// that backs the editor. The subclass is a thin
/// wrapper around `TesseraTextContentManagerData`: it
/// holds the data and implements the
/// `NSTextElementProvider` methods the platform text
/// view calls (`enumerateTextElementsFromLocation:options:usingBlock:`).
///
/// **Per-block elements.** The brief requires one
/// `TesseraTextElement` per block. The data's element
/// list is rebuilt on every `applyMutation` call, and
/// the `textElement(at:)` method does a binary search
/// to return the right element for any
/// `NSTextLocation`.
///
/// **Nesting.** Container blocks (`list`, `toggle`,
/// `table`, `callout`) appear once in the element
/// list (as the header) and their children appear as
/// separate elements with the container's id in their
/// `parentID`. The `NSTextElement` protocol doesn't
/// expose the parent relationship directly; the
/// `TesseraTextElement.data.parentID` carries it.
///
/// **Empty document.** An empty `DocumentAST` produces
/// an empty element list; `enumerateTextElements`
/// returns immediately and `textElement(at:)` returns
/// `nil` for any `NSTextLocation`.
///
/// **Apply mutation.** The platform view layer calls
/// `applyMutation(_:)` (or `applyMutations(_:)`) when
/// the text view produces an edit. The data layer's
/// `MutationEngine` validates + applies, the element
/// list rebuilds, and the platform text view picks up
/// the change on the next layout pass.
public final class TesseraTextContentManager: NSTextContentManager, NSTextContentManagerDelegate {
    public let data: TesseraTextContentManagerData

    public init(data: TesseraTextContentManagerData) {
        self.data = data
        super.init()
        self.delegate = self
    }

    public init(document: DocumentAST, mode: EditorMode = .document) {
        self.data = TesseraTextContentManagerData(
            document: document,
            renderer: BlockRenderer(),
            mode: mode
        )
        super.init()
        self.delegate = self
    }

    public required init?(coder: NSCoder) {
        self.data = TesseraTextContentManagerData()
        super.init(coder: coder)
        self.delegate = self
    }

    /// Convenience: the document. Equivalent to `data.document`.
    public var document: DocumentAST { data.document }

    // MARK: - Mutation apply (called by the platform view layer)

    /// Apply a single mutation to the document. Throws
    /// `MutationError` on validation failure. The element
    /// list rebuilds; the platform text view picks up
    /// the change on the next layout pass.
    @discardableResult
    public func applyMutation(_ mutation: Mutation) throws -> [UUID: Block] {
        try data.applyMutation(mutation)
    }

    /// Apply a batch of mutations.
    @discardableResult
    public func applyMutations(_ mutations: [Mutation]) throws -> [UUID: Block] {
        try data.applyMutations(mutations)
    }

    /// The list of elements in the document. The platform
    /// text view consumes this via `enumerateTextElementsFromLocation`;
    /// the public `textElements()` helper is a convenience
    /// for tests and one-shot lookups.
    public func textElements() -> [TesseraTextElement] {
        data.elements.map { TesseraTextElement(data: $0) }
    }

    /// The element at the given `NSTextLocation`. The
    /// implementation does a binary search by the location's
    /// UTF-16 offset; the platform text view calls this
    /// via the delegate for every caret move.
    public func textElement(at location: NSTextLocation) -> TesseraTextElement? {
        guard let offset = Self.offset(of: location) else { return nil }
        guard let element = data.elementAt(offset: offset) else { return nil }
        return TesseraTextElement(data: element)
    }

    /// The element for a block id. The view layer calls
    /// this when it has a block id (e.g., from the chat
    /// panel) and needs to map it to a position in the
    /// text view.
    public func element(forBlockID blockID: UUID) -> TesseraTextElement? {
        data.element(for: blockID).map { TesseraTextElement(data: $0) }
    }

    // MARK: - NSTextElementProvider

    /// The document range. The platform text view reads this
    /// to know the document's extent; we use the total
    /// length of all element ranges.
    public override var documentRange: NSTextRange {
        let totalLength = data.elements.last?.rangeEnd ?? 0
        return makeIntTextRange(start: 0, end: totalLength)
            ?? makeIntTextRange(start: 0, end: 0)!
    }

    /// Enumerate the text elements. The platform text
    /// view calls this on the initial layout pass and
    /// after every edit; the implementation walks the
    /// `TesseraTextContentManagerData`'s element list.
    /// Container blocks are enumerated once (the header);
    /// their children appear as separate elements with
    /// the container's id in their `parentID`.
    ///
    /// The method overrides the `NSTextContentManager`'s
    /// `enumerateTextElementsFromLocation:options:usingBlock:`
    /// Objective-C method. The Swift signature matches the
    /// platform's bridging: `NSTextLocation?` in,
    /// `NSTextLocation?` out, `(NSTextElement) -> Bool`
    /// block.
    public override func enumerateTextElements(
        from location: NSTextLocation?,
        options: NSTextContentManager.EnumerationOptions = [],
        using block: (NSTextElement) -> Bool
    ) -> NSTextLocation? {
        let startOffset: Int
        if let location, let parsed = Self.offset(of: location) {
            startOffset = parsed
        } else {
            startOffset = 0
        }
        let reverse = options.contains(.reverse)
        let stream: [TesseraTextElementData]
        if reverse {
            stream = data.elements.reversed().filter { $0.rangeStart < startOffset || $0.rangeStart == 0 }
        } else {
            stream = data.elements.filter { $0.rangeStart >= startOffset }
        }
        for element in stream {
            let platform = TesseraTextElement(data: element)
            if !block(platform) { break }
        }
        // Return the location of the end of the last
        // element we enumerated. The platform uses this
        // to chain enumerations.
        return IntTextLocation(intValue: data.elements.last?.rangeEnd ?? 0)
    }

    // MARK: - NSTextContentManagerDelegate

    public func textContentManager(
        _ textContentManager: NSTextContentManager,
        textElementAt location: NSTextLocation
    ) -> NSTextElement? {
        textElement(at: location)
    }

    public func textContentManager(
        _ textContentManager: NSTextContentManager,
        shouldEnumerate textElement: NSTextElement,
        options: NSTextContentManager.EnumerationOptions = []
    ) -> Bool {
        // Enumerate every element. Container blocks are
        // emitted once (the header); their children are
        // separate elements (the `parentID` relationship
        // carries the tree shape for the chat panel /
        // receipt layer).
        return true
    }

    // MARK: - NSTextLocation helpers

    /// The UTF-16 offset of an `NSTextLocation`. Returns
    /// nil for locations whose underlying type isn't
    /// supported. The platform's plain integer locations
    /// are `NSNumber` values; the platform's richer
    /// `NSTextLocationPlus` type isn't used by the
    /// editor. Our own `IntTextLocation` is the canonical
    /// integer location in the editor.
    private static func offset(of location: NSTextLocation) -> Int? {
        if let intLoc = location as? IntTextLocation {
            return intLoc.intValue
        }
        if let n = location as? NSNumber {
            return n.intValue
        }
        if let n = location as? Int {
            return n
        }
        return nil
    }
}
#endif

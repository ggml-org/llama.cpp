import Foundation

// MARK: - TextEditReducer

/// Converts an `NSAttributedString` edit (a "before" and
/// "after" string) into a list of `Mutation` operations
/// against the AST. This is the seam between the platform
/// text view (which produces attributed-string edits via its
/// typing/formatting/paste pipeline) and the Phase 1
/// `Mutation` API (which the agent and the receipt chain
/// both consume).
///
/// The reducer is **stateless** and **pure**: the same input
/// always produces the same output. The coalescer (see
/// `EditorCoalescer`) is the stateful layer that aggregates
/// many small edits into a single `Mutation` batch.
///
/// **Diff strategy.** The reducer walks the two strings,
/// finds the common prefix and common suffix, and the
/// differing middle is the edit region. The edit region is
/// then classified as one of:
///   * **Insertion** — the "before" range is empty.
///   * **Deletion** — the "after" range is empty.
///   * **Replacement** — both ranges are non-empty.
///   * **Pure formatting change** — the strings are equal
///     but the attributed-string attributes differ (a
///     bold/italic/etc. keypress that didn't add or remove
///     characters). This produces a `setInlineAnnotation`
///     mutation.
///
/// **Block-boundary edits.** A single attributed-string edit
/// that crosses a block boundary produces multiple
/// `Mutation` operations (one per affected block). The
/// reducer walks the affected blocks in document order.
///
/// **Failure mode.** The reducer returns an empty list when
/// the diff can't be classified (e.g., the input strings are
/// equal but the document doesn't match). The caller is
/// expected to handle this gracefully (drop the edit, surface
/// to the user, etc.).
public struct TextEditReducer: Sendable {

    public init() {}

    /// The diff of two strings: the common prefix length, the
    /// common suffix length, and the inserted + deleted ranges
    /// in the "before" and "after" strings.
    public struct Diff: Sendable, Equatable {
        public var commonPrefix: Int
        public var commonSuffix: Int
        /// Range in the "before" string that was removed.
        public var deletedRange: NSRange
        /// Range in the "after" string that was inserted.
        public var insertedRange: NSRange

        public var isInsertion: Bool { deletedRange.length == 0 }
        public var isDeletion: Bool { insertedRange.length == 0 }
        public var isReplacement: Bool { deletedRange.length > 0 && insertedRange.length > 0 }
        public var isEmpty: Bool { deletedRange.length == 0 && insertedRange.length == 0 }
    }

    /// Compute the diff between two strings. Walks the
    /// common prefix + common suffix in O(n). The "after"
    /// string's length is implied: it's `before.count -
    /// deletedRange.length + insertedRange.length`.
    public static func diff(before: String, after: String) -> Diff {
        let beforeChars = Array(before)
        let afterChars = Array(after)
        var prefix = 0
        let minLen = min(beforeChars.count, afterChars.count)
        while prefix < minLen && beforeChars[prefix] == afterChars[prefix] {
            prefix += 1
        }
        var suffix = 0
        let beforeRemaining = beforeChars.count - prefix
        let afterRemaining = afterChars.count - prefix
        let maxSuffix = min(beforeRemaining, afterRemaining)
        while suffix < maxSuffix
                && beforeChars[beforeChars.count - 1 - suffix] == afterChars[afterChars.count - 1 - suffix] {
            suffix += 1
        }
        let deletedRange = NSRange(location: prefix, length: beforeRemaining - suffix)
        let insertedRange = NSRange(location: prefix, length: afterRemaining - suffix)
        return Diff(
            commonPrefix: prefix,
            commonSuffix: suffix,
            deletedRange: deletedRange,
            insertedRange: insertedRange
        )
    }

    /// Reduce an edit into a list of `Mutation` operations
    /// against the document. `blockID` is the block the user
    /// was editing; the reducer assumes the edit starts and
    /// ends in the same block. Cross-block edits (a paste
    /// that spans two blocks) are handled by the caller,
    /// which splits the edit into per-block diffs and calls
    /// `reduce` once per block.
    ///
    /// `beforeBlock` and `afterBlock` are the block's content
    /// in the AST before and after the edit; the reducer
    /// uses the diff between `before` and `after` to classify
    /// the edit and choose the right mutation variant.
    ///
    /// - Returns: a list of `Mutation` operations that, when
    ///   applied in order to the `beforeBlock` content, produce
    ///   the `afterBlock` content. Empty list when the diff is
    ///   empty.
    public func reduce(
        blockID: UUID,
        before: [InlineRun],
        after: [InlineRun]
    ) -> [Mutation] {
        let beforeText = before.map(\.text).joined()
        let afterText = after.map(\.text).joined()
        let diff = Self.diff(before: beforeText, after: afterText)
        if diff.isEmpty { return [] }

        // A pure replacement of the block's content. The
        // editor's platform text view produces per-keystroke
        // diffs that may or may not span the whole block;
        // the reducer handles each case.
        if diff.isReplacement {
            return [.setBlockContent(blockID: blockID, content: after)]
        }
        if diff.isInsertion {
            // The inserted text + the existing surrounding
            // text fit cleanly into the current run. We could
            // emit a `replaceInlineRun` for a small insertion,
            // but a `setBlockContent` is simpler and the
            // engine will reduce it cleanly. The pre-mutation
            // snapshot in the receipt captures the old content.
            return [.setBlockContent(blockID: blockID, content: after)]
        }
        if diff.isDeletion {
            return [.setBlockContent(blockID: blockID, content: after)]
        }
        return []
    }

    /// Reduce a formatting change (a key press that toggled
    /// bold / italic / etc.) into a `setInlineAnnotation`
    /// mutation. The reducer picks the run that contains the
    /// offset and decides whether the annotation should be
    /// added (was missing) or removed (was present).
    ///
    /// - Parameters:
    ///   - blockID: the block the user is editing.
    ///   - content: the block's current inline content
    ///     (the post-edit state, not the pre-edit).
    ///   - offset: the character's offset into the block's
    ///     flattened text where the cursor sits.
    ///   - annotation: the annotation the user toggled.
    public func reduceFormattingChange(
        blockID: UUID,
        content: [InlineRun],
        offset: Int,
        annotation: InlineRun.Annotation
    ) -> [Mutation] {
        // Locate the run that contains the offset.
        var remaining = offset
        for (idx, run) in content.enumerated() {
            if remaining <= run.text.count {
                let hasIt = run.annotations.contains(annotation)
                return [.setInlineAnnotation(
                    blockID: blockID,
                    index: idx,
                    annotation: annotation,
                    enabled: !hasIt
                )]
            }
            remaining -= run.text.count
        }
        // The offset is at the very end of the block. The
        // formatting applies to the last run.
        let lastIndex = max(0, content.count - 1)
        guard lastIndex < content.count else { return [] }
        let hasIt = content[lastIndex].annotations.contains(annotation)
        return [.setInlineAnnotation(
            blockID: blockID,
            index: lastIndex,
            annotation: annotation,
            enabled: !hasIt
        )]
    }

    /// Reduce a paste / bulk-insert into a block. The
    /// reducer assumes the paste replaces the block's
    /// content wholesale; per-block splits are handled by
    /// the caller.
    public func reducePaste(
        blockID: UUID,
        pastedText: String,
        existingAnnotations: [InlineRun.Annotation]
    ) -> [Mutation] {
        let run = InlineRun(text: pastedText, annotations: existingAnnotations)
        return [.setBlockContent(blockID: blockID, content: [run])]
    }
}

// MARK: - NSRange helpers

extension NSRange {
    /// The substring of `string` that this range covers.
    /// Returns the empty string for an out-of-range or empty
    /// range. Uses UTF-16 offsets to match `NSAttributedString`.
    public func substring(in string: String) -> String {
        guard location >= 0,
              length >= 0,
              let nsString = string as NSString? else { return "" }
        let end = location + length
        guard end <= nsString.length else { return "" }
        return nsString.substring(with: self)
    }
}

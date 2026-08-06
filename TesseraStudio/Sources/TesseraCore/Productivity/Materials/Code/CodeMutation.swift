import Foundation

// MARK: - CodeMutation

/// A typed operation against a ``CodeFile``. The code
/// surface's equivalent of ``Mutation`` (the document
/// mutation API). The variants are deliberately narrower
/// than the document mutation API: documents have a rich
/// block tree; code is plain text, so the operations are
/// "replace whole file", "find-and-replace a range", and
/// "insert at a position".
///
/// **Why a separate enum, not a Mutation extension.**
/// The ``Mutation`` enum is already large (15+ cases) and
/// every case is a `Codable, Sendable, Hashable` variant
/// that flows through the receipt chain's
/// `preMutationSnapshot`. Adding three cases for code
/// would force every `switch mutation` site to handle
/// them; a separate `CodeMutation` type keeps the
/// document API clean and lets the Code surface own its
/// own snapshot semantics (the pre-mutation snapshot is
/// the prior `body` string, not a `[UUID: Block]` map).
///
/// **Receipt integration.** Every `CodeMutation` is
/// receipted via the data layer's
/// `appendReceipt(entityID:receiptType:payload:)` call
/// with a `receiptType` from ``CodeReceiptType``. The
/// payload carries the `CodeMutation`'s description (the
/// `shortDescription` + the diff stats). The inverse is
/// computed from the `preMutationSnapshot` (the prior
/// `body` string the store captures at apply time).
public enum CodeMutation: Codable, Sendable, Hashable {

    /// Replace the file's body with a new string. The
    /// mutation carries the full new body (not a diff);
    /// the receipt's `preMutationSnapshot` carries the
    /// prior body so undo restores it byte-for-byte.
    case replaceCodeBlock(fileID: UUID, newBody: String)

    /// Find-and-replace a range in the file's body. The
    /// mutation is a `match` -> `replacement` pair; the
    /// engine resolves the `match` against the current
    /// body and replaces it with `replacement`. The
    /// mutation is rejected if `match` is not found or
    /// if it appears more than once (ambiguous).
    case replaceCodeRange(fileID: UUID, match: String, replacement: String)

    /// Insert `text` at `position` (a `String.Index`
    /// offset, 0-indexed) in the file's body. The
    /// mutation is rejected if `position` is out of
    /// range.
    case insertCodeAt(fileID: UUID, position: Int, text: String)

    /// Append a tag. The `CodeStore` rejects duplicates.
    case addTag(fileID: UUID, tag: String)

    /// Remove a tag.
    case removeTag(fileID: UUID, tag: String)

    /// Link the file to another graph entity. The
    /// `CodeStore` is responsible for the actual
    /// `entity_links` row; the mutation is the receipt's
    /// "what was linked" descriptor.
    case linkTo(fileID: UUID, otherEntityID: UUID, linkType: String)

    /// Unlink the file from another graph entity.
    case unlinkFrom(fileID: UUID, otherEntityID: UUID, linkType: String)

    /// The file ID this mutation targets. Every variant
    /// has a `fileID`; the accessor keeps the call
    /// sites tidy.
    public var fileID: UUID {
        switch self {
        case .replaceCodeBlock(let id, _): return id
        case .replaceCodeRange(let id, _, _): return id
        case .insertCodeAt(let id, _, _): return id
        case .addTag(let id, _): return id
        case .removeTag(let id, _): return id
        case .linkTo(let id, _, _): return id
        case .unlinkFrom(let id, _, _): return id
        }
    }

    /// The receipt type for this mutation. The store
    /// uses this to pick the `receiptType` string in the
    /// `graph_receipts` row. The mapping is exhaustive
    /// over the variants.
    public var receiptType: String {
        switch self {
        case .replaceCodeBlock: return CodeReceiptType.bodyReplaced.rawValue
        case .replaceCodeRange: return CodeReceiptType.bodyReplaced.rawValue
        case .insertCodeAt: return CodeReceiptType.bodyReplaced.rawValue
        case .addTag: return CodeReceiptType.tagged.rawValue
        case .removeTag: return CodeReceiptType.untagged.rawValue
        case .linkTo: return CodeReceiptType.linked.rawValue
        case .unlinkFrom: return CodeReceiptType.unlinked.rawValue
        }
    }

    /// A short human-readable description, used in the
    /// receipt's `summary` field. The mutation engine
    /// composes multiple summaries into a one-line
    /// summary like "3 insertions, 2 deletions".
    public var shortDescription: String {
        switch self {
        case .replaceCodeBlock(_, let body):
            return "replace file body (\(body.count) chars)"
        case .replaceCodeRange(_, let match, let replacement):
            return "replace range '\(match.prefix(30))' -> '\(replacement.prefix(30))'"
        case .insertCodeAt(_, let position, let text):
            return "insert \(text.count) chars at \(position)"
        case .addTag(_, let tag):
            return "add tag '\(tag)'"
        case .removeTag(_, let tag):
            return "remove tag '\(tag)'"
        case .linkTo(_, let other, let linkType):
            return "link to \(other) (\(linkType))"
        case .unlinkFrom(_, let other, let linkType):
            return "unlink from \(other) (\(linkType))"
        }
    }
}

// MARK: - Inverse

extension CodeMutation {
    /// Compute the inverse of this mutation against the
    /// pre-mutation snapshot. The snapshot is what
    /// ``CodeStore/apply(_:to:)`` captures at apply
    /// time: a tuple of `(priorBody: String,
    /// priorTags: [String], priorLinks: [UUID])`.
    ///
    /// The inverse restores the prior state. For a
    /// `replaceCodeBlock`, the inverse is a
    /// `replaceCodeBlock` with the prior body. For
    /// `replaceCodeRange` the inverse carries the
    /// original text; the engine re-resolves the
    /// `match` at undo time (the engine reverses
    /// `replacement` back to `match` in the now-
    /// modified body).
    public func inverse(preBody: String, preTags: [String], preLinks: [UUID]) -> [CodeMutation] {
        switch self {
        case .replaceCodeBlock(let id, _):
            return [.replaceCodeBlock(fileID: id, newBody: preBody)]

        case .replaceCodeRange(let id, let match, _):
            // The inverse replaces the new text back
            // with the original match. The engine
            // resolves `match` against the current body
            // (which now contains `replacement`) and
            // substitutes `match` back. The mutation
            // shape is the same as the original; the
            // parameters are flipped.
            return [.replaceCodeRange(fileID: id, match: match, replacement: match)]

        case .insertCodeAt(let id, let position, let text):
            // The inverse removes the inserted text.
            // We don't have a `removeCodeRange`
            // mutation (the inverse is implemented as a
            // `replaceCodeRange` that finds the inserted
            // text and replaces it with the empty
            // string). This works for the common case
            // of inserting a unique snippet; if the
            // snippet appears more than once the engine
            // rejects the undo (the user gets a
            // "ambiguous undo" message).
            return [.replaceCodeRange(
                fileID: id,
                match: text,
                replacement: ""
            )]

        case .addTag(_, let tag):
            return [.removeTag(fileID: fileID, tag: tag)]

        case .removeTag(_, let tag):
            return [.addTag(fileID: fileID, tag: tag)]

        case .linkTo(_, let other, let linkType):
            return [.unlinkFrom(fileID: fileID, otherEntityID: other, linkType: linkType)]

        case .unlinkFrom(_, let other, let linkType):
            return [.linkTo(fileID: fileID, otherEntityID: other, linkType: linkType)]
        }
    }
}

// MARK: - Apply

/// The result of applying a `CodeMutation` to a
/// `CodeFile`. The `updated` field is the post-mutation
/// file; the `preBody`, `preTags`, and `preLinks` are
/// the prior state (what the receipt needs for undo).
public struct CodeMutationApplyResult: Sendable, Hashable {
    public var updated: CodeFile
    public var preBody: String
    public var preTags: [String]
    public var preLinks: [UUID]

    public init(
        updated: CodeFile,
        preBody: String,
        preTags: [String],
        preLinks: [UUID]
    ) {
        self.updated = updated
        self.preBody = preBody
        self.preTags = preTags
        self.preLinks = preLinks
    }
}

public enum CodeMutationError: Error, Sendable, Equatable {
    case matchNotFound(match: String, inFile: String)
    case matchAmbiguous(match: String, count: Int, inFile: String)
    case positionOutOfRange(position: Int, length: Int, inFile: String)
    case tagAlreadyPresent(tag: String)
    case tagNotPresent(tag: String)
}

// MARK: - CodeMutationEngine

/// Stateless applicator + validator for ``CodeMutation``s.
/// Mirrors ``MutationEngine``'s role: in-memory only, no
/// I/O, the caller serializes access to the file. The
/// engine returns a `CodeMutationApplyResult` with the
/// post-mutation file + the pre-mutation snapshot the
/// receipt needs for undo.
public struct CodeMutationEngine: Sendable {
    public init() {}

    /// Validate a mutation against the current file
    /// state. Throws `CodeMutationError` on any
    /// invariant violation.
    public func validate(_ mutation: CodeMutation, against file: CodeFile) throws {
        switch mutation {
        case .replaceCodeBlock:
            return  // always valid (the body is unconstrained)
        case .replaceCodeRange(_, let match, _):
            if match.isEmpty {
                throw CodeMutationError.matchNotFound(match: match, inFile: file.path)
            }
            let count = Self.countOccurrences(of: match, in: file.body)
            if count == 0 {
                throw CodeMutationError.matchNotFound(match: match, inFile: file.path)
            }
            if count > 1 {
                throw CodeMutationError.matchAmbiguous(
                    match: match, count: count, inFile: file.path
                )
            }
        case .insertCodeAt(_, let position, _):
            let length = file.body.count
            guard position >= 0 && position <= length else {
                throw CodeMutationError.positionOutOfRange(
                    position: position, length: length, inFile: file.path
                )
            }
        case .addTag(_, let tag):
            if file.tags.contains(tag) {
                throw CodeMutationError.tagAlreadyPresent(tag: tag)
            }
        case .removeTag(_, let tag):
            if !file.tags.contains(tag) {
                throw CodeMutationError.tagNotPresent(tag: tag)
            }
        case .linkTo, .unlinkFrom:
            // Linking is the data layer's job; the
            // engine doesn't validate it (the link
            // table has its own unique constraints).
            return
        }
    }

    /// Apply the mutation to the file. Returns the
    /// updated file + the pre-mutation snapshot. The
    /// caller (the `CodeStore`) is responsible for
    /// persisting the updated file + writing the
    /// receipt with the snapshot.
    public func apply(_ mutation: CodeMutation, to file: CodeFile) throws -> CodeMutationApplyResult {
        try validate(mutation, against: file)
        let preBody = file.body
        let preTags = file.tags
        let preLinks = file.linkedEntityIDs
        var updated = file
        switch mutation {
        case .replaceCodeBlock(_, let newBody):
            updated.body = newBody
            updated.size = Int64(newBody.utf8.count)
            updated.checksum = CodeFile.computeChecksum(of: newBody)
            updated.updatedAt = Date()

        case .replaceCodeRange(_, let match, let replacement):
            // The engine validated the match is
            // present exactly once, so `replacing(_:with:)`
            // is safe (it replaces the first match).
            if let range = updated.body.range(of: match) {
                updated.body.replaceSubrange(range, with: replacement)
            }
            updated.size = Int64(updated.body.utf8.count)
            updated.checksum = CodeFile.computeChecksum(of: updated.body)
            updated.updatedAt = Date()

        case .insertCodeAt(_, let position, let text):
            let idx = updated.body.index(updated.body.startIndex, offsetBy: position)
            updated.body.insert(contentsOf: text, at: idx)
            updated.size = Int64(updated.body.utf8.count)
            updated.checksum = CodeFile.computeChecksum(of: updated.body)
            updated.updatedAt = Date()

        case .addTag(_, let tag):
            updated.tags.append(tag)
            updated.updatedAt = Date()

        case .removeTag(_, let tag):
            updated.tags.removeAll { $0 == tag }
            updated.updatedAt = Date()

        case .linkTo(_, let other, _):
            if !updated.linkedEntityIDs.contains(other) {
                updated.linkedEntityIDs.append(other)
                updated.updatedAt = Date()
            }

        case .unlinkFrom(_, let other, _):
            updated.linkedEntityIDs.removeAll { $0 == other }
            updated.updatedAt = Date()
        }
        return CodeMutationApplyResult(
            updated: updated,
            preBody: preBody,
            preTags: preTags,
            preLinks: preLinks
        )
    }

    /// Count non-overlapping occurrences of `needle` in
    /// `haystack`. Used by the engine to detect
    /// ambiguous `replaceCodeRange` matches.
    static func countOccurrences(of needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var searchStart = haystack.startIndex
        while searchStart < haystack.endIndex,
              let r = haystack.range(of: needle, range: searchStart..<haystack.endIndex) {
            count += 1
            searchStart = r.upperBound
        }
        return count
    }
}

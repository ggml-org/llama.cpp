import Foundation

// MARK: - CodeReceiptType

/// The set of receipt types ``CodeStore`` emits. The
/// string values are what show up in the
/// `graph_receipts.receipt_type` column; the enum keeps
/// the call sites self-documenting and makes the receipt
/// vocabulary searchable.
///
/// The taxonomy mirrors the spec's §12.10 list:
///   * `imported` — a new file was added to the materials
///   * `modified` — a file's body was changed (user or
///     agent)
///   * `deleted` — a file was removed (the receipt is
///     the voiding record)
///   * `renamed` — a file's path changed
///   * `tagged` / `untagged` — tag changes
///   * `linked` / `unlinked` — link changes
///   * `bodyReplaced` — generic body change (the agent's
///     diff-driven mutations all use this)
///   * `gitFetched` — the user triggered a git pull /
///     refresh (the receipt is informational; the data
///     is the new commit list)
public enum CodeReceiptType: String, Codable, Sendable, CaseIterable {
    /// A new file was added to the materials.
    case imported = "code_file_imported"
    /// A file's body was changed. The receipt's
    /// payload carries the diff stats (insertions /
    /// deletions / checksum before + after).
    case modified = "code_file_modified"
    /// A file was deleted. The receipt's `voidedBy`
    /// relationship voids any prior `modified`
    /// receipts in the chain (the chain is
    /// append-only; the deletion is its own receipt).
    case deleted = "code_file_deleted"
    /// A file was renamed (path changed; the id is
    /// stable). The receipt carries the old + new
    /// paths in the payload.
    case renamed = "code_file_renamed"
    /// A tag was added.
    case tagged = "code_file_tagged"
    /// A tag was removed.
    case untagged = "code_file_untagged"
    /// A link to another graph entity was created.
    case linked = "code_file_linked"
    /// A link was removed.
    case unlinked = "code_file_unlinked"
    /// Generic body replacement. The `Mutation`'s
    /// `replaceCodeBlock`, `replaceCodeRange`, and
    /// `insertCodeAt` cases all emit this receipt
    /// type (the diff stats are computed from the
    /// pre/post bodies in the apply path).
    case bodyReplaced = "code_file_body_replaced"
    /// The user triggered a git refresh (the receipt
    /// is informational; the chain is unchanged but
    /// the receipt's payload carries the latest
    /// commit list).
    case gitFetched = "code_file_git_fetched"
}

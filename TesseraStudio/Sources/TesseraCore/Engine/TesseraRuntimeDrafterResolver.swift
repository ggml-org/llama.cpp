import Foundation

/// Resolves the runtime speculative drafter GGUF for the on-device provider
/// (runtime-traces spec section 7).
///
/// The setting has three shapes:
///   - an explicit path: wins as-is (user intent is never silently
///     second-guessed, same doctrine as ``TesseraTrainBinaryResolver``)
///   - empty: auto-derive `<base>-tessera-trained.gguf` next to the base
///     model (the artifact the idle training cycle produces)
///   - the sentinel "-": auto-derive disabled, trunk-only runtime
///
/// Existence is decided by the caller's injected probe so the logic stays
/// pure; a resolved path that does not exist routes to the trunk-only path
/// (degrade open).
public enum TesseraRuntimeDrafterResolver {
    /// Setting value that disables the auto-derive.
    public static let disableSentinel = "-"

    /// Suffix of the artifact the drafter training cycle writes next to the
    /// base model.
    public static let trainedSuffix = "-tessera-trained.gguf"

    /// Resolve the setting to a candidate drafter path.
    /// Returns nil when spec decoding should not run (sentinel, or no trunk
    /// to derive from). Does NOT check existence - see
    /// ``resolvedDrafter(setting:trunkPath:exists:)``.
    public static func resolve(setting: String, trunkPath: String) -> String? {
        let trimmed = setting.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == disableSentinel {
            return nil
        }
        if !trimmed.isEmpty {
            return NSString(string: trimmed).expandingTildeInPath
        }
        return derivedPath(forTrunk: trunkPath)
    }

    /// The auto-derived drafter path for a trunk model:
    /// `<base>-tessera-trained.gguf` next to the trunk GGUF.
    public static func derivedPath(forTrunk trunkPath: String) -> String? {
        let expanded = NSString(string: trunkPath).expandingTildeInPath
        guard !expanded.isEmpty else { return nil }
        let base = expanded.hasSuffix(".gguf")
            ? String(expanded.dropLast(".gguf".count))
            : expanded
        return base + trainedSuffix
    }

    /// Full resolution: candidate path plus existence check. Returns the
    /// drafter to load, or nil for the trunk-only path.
    public static func resolvedDrafter(
        setting: String,
        trunkPath: String,
        exists: (String) -> Bool = FileManager.default.fileExists(atPath:)
    ) -> String? {
        guard let candidate = resolve(setting: setting, trunkPath: trunkPath) else {
            return nil
        }
        return exists(candidate) ? candidate : nil
    }
}

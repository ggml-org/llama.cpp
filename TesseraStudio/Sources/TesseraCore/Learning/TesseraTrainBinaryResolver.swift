import Foundation

/// Resolves the tessera-train-lk driver binary for the learning loop.
/// An explicit override wins as-is (user intent is never silently
/// second-guessed); with no override the known install locations are
/// checked in order. Always returns a path, executable or not - the
/// orchestrator's gate reports honestly when it is not executable.
public enum TesseraTrainBinaryResolver {
    /// Known install locations, in preference order.
    public static let knownLocations = [
        "/opt/homebrew/bin/tessera-train-lk",
        "/usr/local/bin/tessera-train-lk",
    ]

    /// The path the error message names when nothing resolved; matches
    /// the install instruction in the missing-binary note.
    public static let expectedLocation = "/usr/local/bin/tessera-train-lk"

    public static func resolve(
        override: String,
        isExecutable: (String) -> Bool = FileManager.default.isExecutableFile(atPath:)
    ) -> String {
        let trimmed = override.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty else { return trimmed }
        return knownLocations.first(where: isExecutable) ?? expectedLocation
    }
}

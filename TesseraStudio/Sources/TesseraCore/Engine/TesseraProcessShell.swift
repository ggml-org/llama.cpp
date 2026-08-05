import Foundation

/// Thin seam over `ProcessRunner` so the engine tools can be tested without
/// spawning a real subprocess. `ProcessRunner` conforms in production; tests
/// supply a mock that captures the executable + argv and returns a canned
/// `ProcessResult`.
///
/// Kept intentionally narrow (only the `run` shape the engine tools use) so
/// the test surface stays small and the protocol does not drag in the
/// streaming / combined variants that no tool relies on today.
public protocol TesseraProcessShell: Sendable {
    func run(
        executable: String,
        arguments: [String],
        environment: [String: String]?,
        workingDirectory: String?
    ) async throws -> ProcessResult
}

extension ProcessRunner: TesseraProcessShell {}

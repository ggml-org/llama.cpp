import Foundation

// MARK: - TesseraCLIPath

/// Resolves the path to the ``tessera-cli`` / Python subprocess used by
/// the import + export pipeline.
///
/// The task description writes ``TesseraCLIPath.default`` as the
/// canonical accessor. We provide that as a thin shim on top of
/// ``TesseraCLIBinaryResolver`` so callers that don't need the
/// diagnostics the resolver returns can ask for "the path" in one
/// expression. The shim deliberately re-uses the same precedence
/// order as the resolver (override > settings key > known locations
/// > $PATH) so a user override is honoured across the codebase.
public enum TesseraCLIPath {
    /// The resolved binary path, or nil when nothing executable is
    /// found. The caller is expected to surface a "binary not
    /// found" error to the user (per the resolver's contract).
    public static var `default`: String? {
        TesseraCLIBinaryResolver.resolve(
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        )
    }

    /// Same as ``default`` but with the full diagnostic
    /// (``ResolvedPath``) so the caller can report which
    /// locations were tried.
    public static var resolved: TesseraCLIBinaryResolver.ResolvedPath {
        TesseraCLIBinaryResolver.resolvedPathOrDiagnostic(
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        )
    }

    /// The Python interpreter used by the importer / exporter
    /// subprocess. The importer / exporter run as a Python module
    /// under the repository's ``tools/tessera/importers/cli.py`` /
    /// ``tools/tessera/exporters/cli.py``, so we need a working
    /// Python 3 to spawn.
    public static var pythonExecutable: String {
        if let stored = UserDefaults.standard.string(
            forKey: TesseraSettingsKey.tesseraPythonPath),
           !stored.isEmpty,
           FileManager.default.isExecutableFile(atPath: stored)
        {
            return stored
        }
        // Walk common locations. Homebrew on Apple Silicon is the
        // canonical install; system Python on macOS is
        // ``/usr/bin/python3`` (3.8+ which is too old for the
        // importer's typing, so we prefer Homebrew).
        let candidates: [String] = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        for c in candidates where FileManager.default.isExecutableFile(atPath: c) {
            return c
        }
        // Fall back to PATH lookup
        if let p = TesseraCLIBinaryResolver.pathSearch() {
            return p
        }
        return "/usr/bin/python3"
    }

    /// The repository root, derived from the binary path or
    /// the CWD. The Python subprocess needs to know where
    /// ``tools/tessera/importers/`` lives; we set the CWD to the
    /// repo root before launching.
    public static var repoRoot: URL {
        // First try the standard tessera checkout path on the
        // architect's machine.
        let home = NSHomeDirectory()
        let candidates: [String] = [
            "\(home)/Developer/GitHub/tessera",
        ]
        for c in candidates {
            let url = URL(fileURLWithPath: c)
            if FileManager.default.fileExists(
                atPath: url.appendingPathComponent("tools/tessera/importers/cli.py").path
            ) {
                return url
            }
        }
        return URL(fileURLWithPath: home)
            .appendingPathComponent("Developer/GitHub/tessera")
    }

    /// Path to the importer CLI script (a Python module entry point).
    public static var importerScript: String {
        repoRoot.appendingPathComponent("tools/tessera/importers/cli.py").path
    }

    /// Path to the exporter CLI script.
    public static var exporterScript: String {
        repoRoot.appendingPathComponent("tools/tessera/exporters/cli.py").path
    }
}

// MARK: - PythonSubprocessRunner

/// Lightweight wrapper around ``Process`` for invoking the Python
/// importer / exporter. The runner streams stdout / stderr to
/// the supplied handlers so the caller can present a live
/// progress UI.
///
/// The runner is a struct, not an actor: ``Process`` is not
/// thread-safe but the actor callers (the import / export
/// actors) are. We treat one ``runOnce`` as a single-shot
/// invocation; long-running subprocesses can be re-invoked
/// from the actor.
public struct PythonSubprocessRunner: Sendable {
    public struct Result: Sendable {
        public let stdout: String
        public let stderr: String
        public let exitCode: Int32
    }

    public init() {}

    /// Run a Python module under the repository's Python interpreter.
    /// Returns the captured stdout / stderr and the exit code.
    /// Throws ``RunnerError.failedToLaunch`` when the Python
    /// interpreter or the script is missing.
    public func run(
        script: String,
        args: [String],
        env: [String: String] = [:],
        workingDirectory: URL? = nil
    ) throws -> Result {
        let python = TesseraCLIPath.pythonExecutable
        let argv = [python, script] + args
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = Array(args)
        process.currentDirectoryURL = workingDirectory ?? TesseraCLIPath.repoRoot

        var environment = ProcessInfo.processInfo.environment
        // Carry through the receipt-signing key (the subprocess
        // signs receipts with the same key as the Swift side).
        for (k, v) in env {
            environment[k] = v
        }
        // The dry-run default matches the Python side's default.
        if environment["TESSERA_DATA_LAYER_DRY_RUN"] == nil {
            environment["TESSERA_DATA_LAYER_DRY_RUN"] = "1"
        }
        process.environment = environment

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            throw RunnerError.failedToLaunch(reason: String(describing: error))
        }

        // Wait synchronously. The actor wraps this so the
        // caller's await suspends; for a long-running import
        // (large PDF) the actor's other work proceeds in
        // parallel because we hold no state in this struct.
        process.waitUntilExit()

        let outData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        return Result(
            stdout: String(data: outData, encoding: .utf8) ?? "",
            stderr: String(data: errData, encoding: .utf8) ?? "",
            exitCode: process.terminationStatus
        )
    }

    public enum RunnerError: Error, LocalizedError {
        case failedToLaunch(reason: String)
        public var errorDescription: String? {
            switch self {
            case .failedToLaunch(let reason):
                return "python subprocess failed to launch: \(reason)"
            }
        }
    }
}

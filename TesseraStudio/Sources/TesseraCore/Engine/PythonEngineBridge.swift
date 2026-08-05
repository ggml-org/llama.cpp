import Foundation

// MARK: - Errors

/// Errors surfaced by ``PythonEngineBridge`` to the agent loop.
public enum PythonError: Error, LocalizedError {
    /// No Python interpreter could be located on the search path.
    /// `searched` is the ordered list of candidates the bridge tried
    /// (in priority order) for diagnostics.
    case pythonNotFound(searched: [String])
    /// The named script is missing from every ``PythonEngineBridge.discoverScriptDir()``
    /// candidate. The `searched` list names the directories that were probed.
    case scriptNotFound(name: String, searched: [URL])
    /// The subprocess exited with a non-zero status.
    /// `stderrTail` is the last few KB of stderr for the agent / UI.
    case nonZeroExit(code: Int32, stderrTail: String)

    public var errorDescription: String? {
        switch self {
        case let .pythonNotFound(searched):
            return "Python interpreter not found. Searched: \(searched.joined(separator: ", ")). " +
                "Set the TESSERA_PYTHON env var or install python3."
        case let .scriptNotFound(name, searched):
            let dirs = searched.map { $0.path }.joined(separator: ", ")
            return "Python script '\(name).py' not found. Searched: \(dirs). " +
                "Set the TESSERA_SCRIPT_DIR env var to the tools/tessera directory."
        case let .nonZeroExit(code, stderrTail):
            let tail = stderrTail.isEmpty ? "" : "\n\(stderrTail)"
            return "Python script exited with code \(code).\(tail)"
        }
    }
}

// MARK: - Stream chunk

/// One item from a Python subprocess run: a line of output from
/// stdout/stderr, or the terminal exit code. The stream ends right
/// after the single ``finished(exitCode:)`` item.
public enum PythonOutput: Sendable, Equatable {
    case stdoutLine(String)
    case stderrLine(String)
    case finished(exitCode: Int32)
}

// MARK: - Bridge

/// Discovers and runs the Python interpreter + Tessera Python tooling.
///
/// ``PythonEngineBridge`` is an ``actor`` so concurrent tool invocations
/// share the same Python path resolution (the discovery runs once per
/// process) without race conditions on the cached results. The bridge
/// shells out via ``ProcessRunner``; it does not bundle CPython.
///
/// The discovery order on macOS is:
///   1. `TESSERA_PYTHON` env var
///   2. `/usr/bin/which python3`
///   3. `/opt/homebrew/bin/python3` (Apple Silicon)
///   4. `/usr/local/bin/python3` (Intel)
///   5. `/usr/bin/python3`
///
/// Script discovery (`tools/tessera/`):
///   1. `TESSERA_SCRIPT_DIR` env var
///   2. `~/Developer/GitHub/tessera/tools/tessera/`
///   3. Walk up from `Bundle.main.bundlePath` looking for `tools/tessera/`
///   4. `/opt/tessera/tools/tessera/`
public actor PythonEngineBridge {
    public static let shared = PythonEngineBridge()

    private var cachedPython: URL?
    private var cachedScriptDir: URL?

    public init() {}

    // MARK: Python interpreter

    /// Resolves the Python interpreter URL. Throws
    /// ``PythonError/pythonNotFound(searched:)`` if every candidate is missing.
    public func discoverPython() throws -> URL {
        if let cached = cachedPython {
            return cached
        }

        var searched: [String] = []
        let env = ProcessInfo.processInfo.environment

        if let explicit = env["TESSERA_PYTHON"], !explicit.isEmpty {
            searched.append(explicit)
            let url = URL(fileURLWithPath: explicit)
            if FileManager.default.isExecutableFile(atPath: url.path) {
                cachedPython = url
                return url
            }
        }

        // /usr/bin/which python3
        if let whichURL = try? runWhich("python3") {
            searched.append("/usr/bin/which -> \(whichURL.path)")
            if FileManager.default.isExecutableFile(atPath: whichURL.path) {
                cachedPython = whichURL
                return whichURL
            }
        } else {
            searched.append("/usr/bin/which python3")
        }

        let staticCandidates = [
            "/opt/homebrew/bin/python3",  // Apple Silicon homebrew
            "/usr/local/bin/python3",     // Intel homebrew
            "/usr/bin/python3",
        ]
        for path in staticCandidates {
            searched.append(path)
            if FileManager.default.isExecutableFile(atPath: path) {
                let url = URL(fileURLWithPath: path)
                cachedPython = url
                return url
            }
        }

        throw PythonError.pythonNotFound(searched: searched)
    }

    private func runWhich(_ name: String) -> URL? {
        // Synchronous: `discoverPython` is non-async per the public contract,
        // so we shell out via Foundation's Process and block on a semaphore.
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [name]
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return nil
        }
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let text = String(data: data, encoding: .utf8) ?? ""
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard process.terminationStatus == 0, !trimmed.isEmpty else { return nil }
        return URL(fileURLWithPath: trimmed)
    }

    // MARK: Script directory

    /// Resolves the directory containing the `tools/tessera/*.py` scripts.
    public func discoverScriptDir() throws -> URL {
        if let cached = cachedScriptDir {
            return cached
        }

        let env = ProcessInfo.processInfo.environment
        let fm = FileManager.default

        var candidates: [URL] = []
        if let explicit = env["TESSERA_SCRIPT_DIR"], !explicit.isEmpty {
            candidates.append(URL(fileURLWithPath: explicit))
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        candidates.append(home
            .appendingPathComponent("Developer/GitHub/tessera/tools/tessera"))

        // Walk up from Bundle.main.bundlePath looking for a tools/tessera/ directory.
        let bundle = Bundle.main.bundleURL
        var dir = bundle.deletingLastPathComponent()
        for _ in 0..<8 {
            let probe = dir.appendingPathComponent("tools/tessera")
            if fm.fileExists(atPath: probe.path) {
                candidates.append(probe)
                break
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            dir = parent
        }
        candidates.append(URL(fileURLWithPath: "/opt/tessera/tools/tessera"))

        for url in candidates {
            if fm.fileExists(atPath: url.path) {
                cachedScriptDir = url
                return url
            }
        }

        throw PythonError.scriptNotFound(name: "*", searched: candidates)
    }

    /// Locates a specific script under the resolved ``discoverScriptDir()``.
    public func locateScript(_ name: String) throws -> URL {
        let dir = try discoverScriptDir()
        let url = dir.appendingPathComponent("\(name).py")
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        throw PythonError.scriptNotFound(name: name, searched: [dir])
    }

    // MARK: Subprocess streaming

    /// Runs the named Python script and streams stdout/stderr lines.
    /// The stream terminates with a single ``PythonOutput/finished(exitCode:)``
    /// item. The consuming task's cancellation propagates to the subprocess
    /// (the stream's `onTermination` calls `process.terminate()` on cancel).
    ///
    /// - Parameters:
    ///   - script: script basename without the `.py` suffix.
    ///   - args: CLI args to append after the script path.
    ///   - env: extra env vars merged on top of the process env.
    public func run(
        script: String,
        args: [String],
        env: [String: String] = [:]
    ) -> AsyncThrowingStream<PythonOutput, Error> {
        // Resolve python + script up-front. On failure, the returned stream
        // emits a single terminal error and finishes; the agent loop can
        // still see it via `for try await` and convert to a ToolResult.
        let python: URL
        let scriptURL: URL
        do {
            python = try discoverPython()
            scriptURL = try locateScript(script)
        } catch {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: error)
            }
        }

        let runner = ProcessRunner()
        let rawStream = runner.runStreamingCombined(
            executable: python.path,
            arguments: [scriptURL.path] + args,
            environment: env
        )
        return AsyncThrowingStream { continuation in
            let consumerTask = Task {
                do {
                    for try await chunk in rawStream {
                        try Task.checkCancellation()
                        switch chunk {
                        case let .output(stream, text):
                            let lineChunked = chunkLines(text)
                            for line in lineChunked {
                                switch stream {
                                case .stdout:
                                    continuation.yield(.stdoutLine(line))
                                case .stderr:
                                    continuation.yield(.stderrLine(line))
                                }
                            }
                        case let .exited(code):
                            continuation.yield(.finished(exitCode: code))
                            continuation.finish()
                            return
                        }
                    }
                    // The raw stream finished without yielding `.exited`.
                    // That happens when the subprocess was killed by
                    // the cancellation handler; the user-visible signal
                    // here is CancellationError, not a normal end.
                    try Task.checkCancellation()
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                consumerTask.cancel()
            }
        }
    }

    /// Synchronous variant: collect every line into stdout/stderr buffers and
    /// return the final ``ProcessResult``-like tuple. Used by the high-level
    /// ``PythonToolWrapper`` when the agent loop wants the whole transcript
    /// for a single JSON parse.
    public func runCollect(
        script: String,
        args: [String],
        env: [String: String] = [:]
    ) async throws -> (exitCode: Int32, stdout: String, stderr: String) {
        var stdout = ""
        var stderr = ""
        var exitCode: Int32 = -1
        for try await item in run(script: script, args: args, env: env) {
            try Task.checkCancellation()
            switch item {
            case let .stdoutLine(line): stdout += line + "\n"
            case let .stderrLine(line): stderr += line + "\n"
            case let .finished(code): exitCode = code
            }
        }
        return (exitCode, stdout, stderr)
    }
}

// MARK: - Helpers

/// Splits a `ProcessChunk.output` blob (which may carry multiple lines
/// or a partial line) into newline-terminated strings. We deliberately
/// re-attach the trailing partial line so the next call can flush it.
private func chunkLines(_ text: String) -> [String] {
    var lines: [String] = []
    var current = ""
    for ch in text {
        if ch == "\n" {
            lines.append(current)
            current = ""
        } else if ch == "\r" {
            // swallow carriage return; the next newline will close the line
        } else {
            current.append(ch)
        }
    }
    if !current.isEmpty {
        lines.append(current)
    }
    return lines
}

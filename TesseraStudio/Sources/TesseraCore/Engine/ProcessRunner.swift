import Foundation

/// Result of a subprocess execution.
public struct ProcessResult: Sendable {
    public let exitCode: Int32
    public let stdout: String
    public let stderr: String

    public init(exitCode: Int32, stdout: String, stderr: String) {
        self.exitCode = exitCode
        self.stdout = stdout
        self.stderr = stderr
    }
}

/// Which standard stream a chunk arrived on.
public enum ProcessStream: Sendable {
    case stdout
    case stderr
}

/// One item from a combined streaming run: a chunk of output from one of the
/// two streams, or the terminal exit code. The stream ends right after the
/// single .exited chunk.
public enum ProcessChunk: Sendable {
    case output(ProcessStream, String)
    case exited(Int32)
}

/// Runs CLI tools as subprocesses, capturing stdout/stderr.
///
/// ProcessRunner lives in TesseraCore (not the Mac target) so the shared
/// tools can compile cross-platform; it is a macOS-only *capability* at
/// runtime. On iOS the methods throw `unavailableOnPlatform` and the engine
/// bridge uses the C FFI instead (see docs/tessera-studio-design.md 2.3).
public final class ProcessRunner: Sendable {
    /// The base directory where tessera CLI tools are installed.
    private let toolDirectory: String

    public init(toolDirectory: String = "/usr/local/bin") {
        self.toolDirectory = toolDirectory
    }

    /// Run a command and wait for completion, returning the result.
    public func run(
        executable: String,
        arguments: [String] = [],
        environment: [String: String]? = nil,
        workingDirectory: String? = nil
    ) async throws -> ProcessResult {
        #if os(macOS)
        return try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()

            // Resolve the executable path
            let execPath: String
            if executable.contains("/") {
                execPath = executable
            } else {
                execPath = (toolDirectory as NSString).appendingPathComponent(executable)
            }

            process.executableURL = URL(fileURLWithPath: execPath)
            process.arguments = arguments
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            if let env = environment {
                var merged = ProcessInfo.processInfo.environment
                for (k, v) in env { merged[k] = v }
                process.environment = merged
            }

            if let wd = workingDirectory {
                process.currentDirectoryURL = URL(fileURLWithPath: wd)
            }

            process.terminationHandler = { proc in
                let outData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                let result = ProcessResult(
                    exitCode: proc.terminationStatus,
                    stdout: String(data: outData, encoding: .utf8) ?? "",
                    stderr: String(data: errData, encoding: .utf8) ?? ""
                )
                continuation.resume(returning: result)
            }

            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
            }
        }
        #else
        throw ProcessRunnerError.unavailableOnPlatform
        #endif
    }

    /// Run a command and stream stdout lines via AsyncStream.
    public func runStreaming(
        executable: String,
        arguments: [String] = [],
        environment: [String: String]? = nil
    ) -> AsyncThrowingStream<String, Error> {
        #if os(macOS)
        return AsyncThrowingStream { continuation in
            let process = Process()
            let pipe = Pipe()

            let execPath: String
            if executable.contains("/") {
                execPath = executable
            } else {
                execPath = (toolDirectory as NSString).appendingPathComponent(executable)
            }

            process.executableURL = URL(fileURLWithPath: execPath)
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = FileHandle.nullDevice

            if let env = environment {
                var merged = ProcessInfo.processInfo.environment
                for (k, v) in env { merged[k] = v }
                process.environment = merged
            }

            pipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty else {
                    continuation.finish()
                    return
                }
                if let line = String(data: data, encoding: .utf8) {
                    continuation.yield(line)
                }
            }

            process.terminationHandler = { _ in
                pipe.fileHandleForReading.readabilityHandler = nil
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                process.terminate()
            }

            do {
                try process.run()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        #else
        return AsyncThrowingStream { continuation in
            continuation.finish(throwing: ProcessRunnerError.unavailableOnPlatform)
        }
        #endif
    }

    /// Run a command and stream BOTH stdout and stderr chunks, ending with a
    /// single .exited(code) chunk. For long-running steps whose output is
    /// parsed live (progress) while the full capture is still needed for the
    /// terminal record. Cancel the consuming task to terminate the process.
    public func runStreamingCombined(
        executable: String,
        arguments: [String] = [],
        environment: [String: String]? = nil,
        workingDirectory: String? = nil
    ) -> AsyncThrowingStream<ProcessChunk, Error> {
        #if os(macOS)
        return AsyncThrowingStream { continuation in
            let process = Process()
            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            let stateLock = NSLock()
            var finished = false

            func finishOnce() {
                stateLock.lock(); defer { stateLock.unlock() }
                guard !finished else { return }
                finished = true
                continuation.finish()
            }

            let execPath: String
            if executable.contains("/") {
                execPath = executable
            } else {
                execPath = (toolDirectory as NSString).appendingPathComponent(executable)
            }

            process.executableURL = URL(fileURLWithPath: execPath)
            process.arguments = arguments
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            if let env = environment {
                var merged = ProcessInfo.processInfo.environment
                for (k, v) in env { merged[k] = v }
                process.environment = merged
            }
            if let wd = workingDirectory {
                process.currentDirectoryURL = URL(fileURLWithPath: wd)
            }

            func attach(_ pipe: Pipe, _ stream: ProcessStream) {
                pipe.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    // EOF: the termination handler drains the tail and finishes.
                    guard !data.isEmpty else { return }
                    if let text = String(data: data, encoding: .utf8) {
                        continuation.yield(.output(stream, text))
                    }
                }
            }
            attach(stdoutPipe, .stdout)
            attach(stderrPipe, .stderr)

            process.terminationHandler = { proc in
                stdoutPipe.fileHandleForReading.readabilityHandler = nil
                stderrPipe.fileHandleForReading.readabilityHandler = nil
                // Drain what is left so the tail (final losses / errors) is not lost.
                let outTail = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                let errTail = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                if !outTail.isEmpty, let text = String(data: outTail, encoding: .utf8) {
                    continuation.yield(.output(.stdout, text))
                }
                if !errTail.isEmpty, let text = String(data: errTail, encoding: .utf8) {
                    continuation.yield(.output(.stderr, text))
                }
                continuation.yield(.exited(proc.terminationStatus))
                finishOnce()
            }

            continuation.onTermination = { @Sendable _ in
                process.terminate()
            }

            do {
                try process.run()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        #else
        return AsyncThrowingStream { continuation in
            continuation.finish(throwing: ProcessRunnerError.unavailableOnPlatform)
        }
        #endif
    }
}

public enum ProcessRunnerError: Error, LocalizedError {
    case unavailableOnPlatform

    public var errorDescription: String? {
        switch self {
        case .unavailableOnPlatform:
            "ProcessRunner is only available on macOS. On iOS, use the C FFI bridge."
        }
    }
}
